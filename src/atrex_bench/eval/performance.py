"""Stage 2: Performance profiling for the candidate module."""

from __future__ import annotations

import contextlib
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule


# record_function label that scopes a single model.forward() invocation.
# Used by the kernel-attribution path so we only credit GPU events that
# actually fired inside the candidate forward (and NOT, e.g., the bench
# loop's per-iter ``clone_model_inputs`` which triggers Memcpy DtoD on
# the device timeline before the forward starts).
_MODEL_FORWARD_LABEL = "atrex_bench_model_forward"

from atrex_bench.eval._runtime import (
    ShapeSpec,
    clone_model_inputs,
    deterministic_input_seed,
    get_device,
    import_module_from_path,
    instantiate_model_module,
    load_model_init_inputs,
    load_reference_inputs,
    load_shape_call_inputs,
    load_shape_init_inputs,
    load_shape_spec,
    resolve_input_module,
    seed_all_input_rngs,
    sync_device,
    validate_reference_module,
    write_input_artifact,  # kept import for backward compat; unused here
)
from atrex_bench.eval._timeout import CandidateTimeoutError, candidate_timeout

_DEFAULT_CANDIDATE_TIMEOUT_S = 60


@dataclass(frozen=True)
class PerformanceSample:
    """One bench iteration's end-to-end forward time.

    Sample order in the parent list is the iteration index; no separate
    ``iteration`` field per the data schema spec, Section 7.
    """

    end_to_end_time_ms: float | None = None


@dataclass(frozen=True)
class KernelTimingEvent:
    """Aggregated device-side timing for one kernel symbol.

    Captured by ``_measure_runner_samples`` only when ``collect_kernel_events``
    is true; otherwise the parent ``PerformanceShapeResult.kernel_events`` list
    is empty.

    Values are normalised to PER MODEL-FORWARD CALL averages (the profiler
    breakdown loop runs N forwards; we divide the raw aggregates by N).
    This lets the attribution ratio compare like-for-like against
    ``samples[0].end_to_end_time_ms`` (which is also per-forward).

      * ``device_time_us`` -- average GPU device time spent in this kernel
        per single model.forward() call.
      * ``calls`` -- average number of launches of this kernel per single
        model.forward() call (rounded to nearest int; raw float available
        as ``device_time_us / per-launch-avg`` if needed).
    """

    name: str
    device_time_us: float
    calls: int


@dataclass(frozen=True)
class PerformanceShapeResult:
    """Per-shape performance result returned by ``benchmark_performance``.

    Performance is not tracked in ``passed`` (correctness-pass implies the
    candidate runs); whether perf actually ran is derivable from
    ``samples`` being non-empty.

    ``observed_kernels`` is the serialized payload from the runtime flydsl
    decorator tracker (``_flydsl_tracker.observed_kernel_symbols_serializable``).
    It carries authoritative ground-truth about which symbols were registered
    via flydsl's ``@kernel`` during the candidate's execution, so the
    classifier in ``kernel_attribution`` doesn't have to guess from source.
    None when the tracker wasn't installed (e.g. flydsl not importable or
    older runs predating this field).
    """

    input_artifact: dict[str, str] | None = None
    samples: list[PerformanceSample] = field(default_factory=list)
    kernel_events: list[KernelTimingEvent] = field(default_factory=list)
    error: str | None = None
    observed_kernels: dict[str, list[str]] | None = None


def _profile_activities(device: torch.device) -> list[ProfilerActivity]:
    """Select profiler activities for the active device."""
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        # PyTorch ROCm uses the CUDA device/profiler namespace too.
        activities.append(ProfilerActivity.CUDA)
    return activities


def _build_profiler_schedule(warmup_iters: int, bench_iters: int):
    """Build the profiler schedule and suppress the no-warmup warning for warmup=0."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Profiler won't be using warmup, this can skew profiler results",
            category=UserWarning,
        )
        return schedule(wait=0, warmup=warmup_iters, active=bench_iters, repeat=1)


_PROFILER_WRAPPER_PREFIXES = ("ProfilerStep",)


def _collect_model_forward_spans(prof: profile) -> list[tuple[float, float]]:
    """Return (start_us, end_us) ranges of every model-forward marker on the GPU timeline.

    Each invocation of the candidate's ``forward`` is wrapped in
    ``record_function(_MODEL_FORWARD_LABEL)``; the profiler emits one event
    for the marker on the CPU timeline AND one mirror event on the CUDA
    timeline whose ``time_range`` covers all GPU work launched inside the
    scope. We use the CUDA-side mirror because we're filtering CUDA leaf
    events by start-time.
    """
    spans: list[tuple[float, float]] = []
    for event in prof.events():
        if str(getattr(event, "device_type", None)) != "DeviceType.CUDA":
            continue
        if event.name != _MODEL_FORWARD_LABEL:
            continue
        time_range = getattr(event, "time_range", None)
        if time_range is None:
            continue
        spans.append((float(time_range.start), float(time_range.end)))
    return spans


def _in_any_span(point_us: float, spans: list[tuple[float, float]]) -> bool:
    """True iff ``point_us`` falls within at least one (start, end) span."""
    for start, end in spans:
        if start <= point_us <= end:
            return True
    return False


def _aggregate_kernel_events(prof: profile | None) -> list[KernelTimingEvent]:
    """Aggregate device-side kernel events by name from the profiler.

    Filters:
      * keep only ``DeviceType.CUDA`` events (PyTorch's ROCm build uses the
        same namespace, so this catches HIP kernels too);
      * drop events with ``self_device_time_total <= 0`` (CPU-side dispatch
        records that attribute device time to a child);
      * drop profiler-side wrapper events (``ProfilerStep*``) which are
        containers, not real kernels;
      * drop the ``model_forward`` markers themselves (they are summaries
        of their child kernels);
      * keep only kernels whose start time falls within a model-forward
        span — this excludes harness-side GPU activity such as the per-iter
        ``clone_model_inputs`` DtoD memcpy, which fires outside the candidate
        forward but inside the profiling cycle.
    """
    if prof is None:
        return []

    spans = _collect_model_forward_spans(prof)

    aggregate: dict[str, dict[str, float | int]] = {}
    for event in prof.events():
        device_type = getattr(event, "device_type", None)
        if str(device_type) != "DeviceType.CUDA":
            continue
        self_device_us = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
        if self_device_us <= 0.0:
            continue
        name = event.name
        if name == _MODEL_FORWARD_LABEL:
            continue
        if any(name.startswith(prefix) for prefix in _PROFILER_WRAPPER_PREFIXES):
            continue
        if spans:
            time_range = getattr(event, "time_range", None)
            if time_range is None:
                continue
            if not _in_any_span(float(time_range.start), spans):
                continue
        bucket = aggregate.setdefault(name, {"device_time_us": 0.0, "calls": 0})
        bucket["device_time_us"] = float(bucket["device_time_us"]) + self_device_us
        bucket["calls"] = int(bucket["calls"]) + 1

    return [
        KernelTimingEvent(
            name=name,
            device_time_us=float(stats["device_time_us"]),
            calls=int(stats["calls"]),
        )
        for name, stats in aggregate.items()
    ]


_PROFILER_BREAKDOWN_ITERS = 5
_DEFAULT_PERF_TIMEOUT_S = 600.0


def _measure_runner_samples(
    model: torch.nn.Module,
    inputs,
    device: torch.device,
    *,
    warmup_iters: int,
    bench_iters: int,
    collect_kernel_events: bool = False,
    perf_timeout_s: int | float | None = _DEFAULT_PERF_TIMEOUT_S,
) -> tuple[list[PerformanceSample], list[KernelTimingEvent]]:
    """Measure end-to-end forward times for one model variant.

    End-to-end timing uses ``triton.testing.do_bench`` (CUDA events, no
    per-iter ``torch.cuda.synchronize`` overhead). When
    ``collect_kernel_events`` is true an extra short profiler loop runs
    afterwards to gather per-kernel device time for the flydsl breakdown
    (its per-iter wall is inaccurate, intentionally not used for
    ``end_to_end_time_ms``).

    The WHOLE perf phase (do_bench + optional breakdown loop) is wrapped
    in a single SIGALRM ``perf_timeout_s`` budget.
    """
    from triton.testing import do_bench

    benchmark_inputs = clone_model_inputs(inputs)

    def _bench_fn():
        model(*benchmark_inputs.args, **benchmark_inputs.kwargs)

    with torch.inference_mode(), candidate_timeout(perf_timeout_s):
        elapsed_ms = do_bench(
            _bench_fn,
            warmup=warmup_iters,
            rep=bench_iters,
        )

        samples = [PerformanceSample(end_to_end_time_ms=elapsed_ms)]

        if not collect_kernel_events:
            return samples, []

        # Profiler loop for kernel breakdown only. Already warm from
        # do_bench, no extra warmup. Per-iter wall is intentionally
        # NOT used for end_to_end_time_ms (do_bench is the source).
        profiler_ctx: contextlib.AbstractContextManager = profile(
            activities=_profile_activities(device),
            schedule=_build_profiler_schedule(0, _PROFILER_BREAKDOWN_ITERS),
            acc_events=True,
        )
        with profiler_ctx as prof:
            for _ in range(_PROFILER_BREAKDOWN_ITERS):
                breakdown_inputs = clone_model_inputs(inputs)
                sync_device(device)
                with record_function(_MODEL_FORWARD_LABEL):
                    model(*breakdown_inputs.args, **breakdown_inputs.kwargs)
                    sync_device(device)
                if prof is not None:
                    prof.step()

        raw_kernel_events = _aggregate_kernel_events(prof)
        # Normalise to per-forward averages so consumers can compare
        # against the per-forward end_to_end_time_ms from do_bench.
        kernel_events = [
            KernelTimingEvent(
                name=ev.name,
                device_time_us=ev.device_time_us / _PROFILER_BREAKDOWN_ITERS,
                calls=max(1, round(ev.calls / _PROFILER_BREAKDOWN_ITERS)),
            )
            for ev in raw_kernel_events
        ]
        return samples, kernel_events


def benchmark_performance(
    candidate_path: Path,
    reference_path: Path,
    *,
    shape_id: str = "0",
    warmup_iters: int = 10,
    bench_iters: int = 100,
    device: str = "auto",
    artifact_path: Path | None = None,
    artifact_root: Path | None = None,
    collect_kernel_events: bool = False,
    candidate_timeout_s: int | float | None = _DEFAULT_CANDIDATE_TIMEOUT_S,
    perf_timeout_s: int | float | None = _DEFAULT_PERF_TIMEOUT_S,
) -> PerformanceShapeResult:
    """Benchmark the candidate module under one shape configuration."""
    if warmup_iters < 0:
        return PerformanceShapeResult(error="warmup_iters must be non-negative")
    if bench_iters < 1:
        return PerformanceShapeResult(error="bench_iters must be at least 1")

    try:
        resolved_device = get_device(device)
        reference_module = import_module_from_path(
            reference_path,
            "atrex_performance_reference",
        )
        validate_reference_module(reference_module)
        input_module = resolve_input_module(
            reference_path,
            reference_module,
            module_prefix="atrex_performance_input",
        )
        shape: ShapeSpec | None
        if (reference_path.parent / "shapes.json").is_file():
            shape = load_shape_spec(reference_path, shape_id)
            reference_init_inputs = load_shape_init_inputs(shape, resolved_device)
        else:
            shape = None
            reference_init_inputs = load_model_init_inputs(input_module, resolved_device)
        with candidate_timeout(candidate_timeout_s):
            loaded_candidate = instantiate_model_module(
                candidate_path,
                resolved_device,
                module_prefix="atrex_performance_candidate",
                init_inputs=reference_init_inputs,
            )
        # Seed before generating inputs so the per-shape perf inputs are
        # reproducible from the recorded seed alone (no .pt files needed).
        perf_seed = deterministic_input_seed("performance", shape_id, 0)
        seed_all_input_rngs(perf_seed)
        if shape is not None:
            inputs = load_shape_call_inputs(input_module, shape, resolved_device)
        else:
            inputs = load_reference_inputs(input_module, resolved_device)
        artifact = {"seed": perf_seed, "format": "manual_seed"}
        samples, kernel_events = _measure_runner_samples(
            loaded_candidate.model,
            inputs,
            resolved_device,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            collect_kernel_events=collect_kernel_events,
            perf_timeout_s=perf_timeout_s,
        )
    except CandidateTimeoutError as timeout_error:
        return PerformanceShapeResult(
            input_artifact=artifact,
            error=str(timeout_error),
        )
    except Exception:
        return PerformanceShapeResult(error=traceback.format_exc())

    return PerformanceShapeResult(
        input_artifact=artifact,
        samples=samples,
        kernel_events=kernel_events,
    )


def benchmark_reference_torch_compile(
    reference_path: Path,
    *,
    shape_id: str = "0",
    warmup_iters: int = 10,
    bench_iters: int = 100,
    device: str = "auto",
    artifact_path: Path | None = None,
    artifact_root: Path | None = None,
) -> PerformanceShapeResult:
    """Benchmark ``torch.compile(reference_model)`` under one shape configuration.

    The first compiled-model invocation is deliberately outside the measured
    samples so Inductor compilation time is not counted as kernel runtime.
    The usual warmup/bench loop then measures steady-state forward latency.
    """
    if warmup_iters < 0:
        return PerformanceShapeResult(error="warmup_iters must be non-negative")
    if bench_iters < 1:
        return PerformanceShapeResult(error="bench_iters must be at least 1")
    if not callable(getattr(torch, "compile", None)):
        return PerformanceShapeResult(error="torch.compile is not available")

    try:
        resolved_device = get_device(device)
        reference_module = import_module_from_path(
            reference_path,
            "atrex_torch_compile_reference",
        )
        validate_reference_module(reference_module)
        input_module = resolve_input_module(
            reference_path,
            reference_module,
            module_prefix="atrex_torch_compile_input",
        )
        shape: ShapeSpec | None
        if (reference_path.parent / "shapes.json").is_file():
            shape = load_shape_spec(reference_path, shape_id)
            init_inputs = load_shape_init_inputs(shape, resolved_device)
        else:
            shape = None
            init_inputs = load_model_init_inputs(input_module, resolved_device)

        model_inputs = clone_model_inputs(init_inputs)
        reference_model = reference_module.Model(
            *model_inputs.args,
            **model_inputs.kwargs,
        ).to(resolved_device).eval()

        perf_seed = deterministic_input_seed("performance", shape_id, 0)
        seed_all_input_rngs(perf_seed)
        if shape is not None:
            inputs = load_shape_call_inputs(input_module, shape, resolved_device)
        else:
            inputs = load_reference_inputs(input_module, resolved_device)

        artifact = {"seed": perf_seed, "format": "manual_seed"}

        compiled_model = torch.compile(reference_model)

        # Trigger torch.compile / Inductor compilation outside timed samples.
        compile_inputs = clone_model_inputs(inputs)
        with torch.inference_mode():
            compiled_model(*compile_inputs.args, **compile_inputs.kwargs)
            sync_device(resolved_device)

        samples, _ = _measure_runner_samples(
            compiled_model,
            inputs,
            resolved_device,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            collect_kernel_events=False,
        )
    except Exception:
        return PerformanceShapeResult(error=traceback.format_exc())

    return PerformanceShapeResult(
        input_artifact=artifact,
        samples=samples,
    )
