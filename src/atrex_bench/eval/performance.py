"""Stage 2: Performance profiling for the candidate module."""

from __future__ import annotations

import contextlib
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from atrex_bench.eval._runtime import (
    ShapeSpec,
    clone_model_inputs,
    deterministic_input_seed,
    flatten_outputs,
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
    write_input_artifact,  # noqa: F401 - backward-compatible module export
)
from atrex_bench.eval._timeout import CandidateTimeoutError, candidate_timeout
from atrex_bench.eval.correctness import (
    _compare_output_tensors,
    _flatten_output_name,
)

# record_function label that scopes a single model.forward() invocation.
# Used by the kernel-attribution path so we only credit GPU events that
# actually fired inside the candidate forward (and NOT, e.g., the bench
# loop's per-iter ``clone_model_inputs`` which triggers Memcpy DtoD on
# the device timeline before the forward starts).
_MODEL_FORWARD_LABEL = "atrex_bench_model_forward"

_DEFAULT_CANDIDATE_TIMEOUT_S = 60
EAGER_BENCHMARK_MODE = "eager"
CUDA_GRAPH_REPLAY_BENCHMARK_MODE = "cuda_graph_replay"
SUPPORTED_BENCHMARK_MODES = frozenset(
    {
        EAGER_BENCHMARK_MODE,
        CUDA_GRAPH_REPLAY_BENCHMARK_MODE,
    }
)


@dataclass(frozen=True)
class PerformanceSample:
    """One bench iteration's end-to-end forward time.

    Sample order in the parent list is the iteration index; no separate
    ``iteration`` field per docs/data_schema.md §七.
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
    benchmark_mode: str = "eager"
    capture_time_ms: float | None = None
    cache_flush_mb: int | None = None
    graph_correctness: dict[str, object] | None = None
    error: str | None = None
    observed_kernels: dict[str, list[str]] | None = None


@dataclass(frozen=True)
class _MeasurementResult:
    samples: list[PerformanceSample] = field(default_factory=list)
    kernel_events: list[KernelTimingEvent] = field(default_factory=list)
    capture_time_ms: float | None = None
    graph_correctness: dict[str, object] | None = None


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


def _measure_eager_samples(
    *,
    bench_fn,
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
) -> _MeasurementResult:
    try:
        from triton.testing import do_bench
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
        for _ in range(max(warmup_iters, 1)):
            bench_fn()
        sync_device(device)

        reps = max(bench_iters, 1)
        start = time.perf_counter()
        for _ in range(reps):
            bench_fn()
        sync_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / reps
        return _MeasurementResult(
            samples=[PerformanceSample(end_to_end_time_ms=elapsed_ms)]
        )

    elapsed_ms = do_bench(
        bench_fn,
        warmup=warmup_iters,
        rep=bench_iters,
    )
    return _MeasurementResult(
        samples=[PerformanceSample(end_to_end_time_ms=elapsed_ms)]
    )


def _flush_cuda_cache(cache_flush_mb: int, device: torch.device) -> None:
    if cache_flush_mb <= 0:
        return
    element_size = torch.empty((), dtype=torch.int32).element_size()
    element_count = cache_flush_mb * 1024 * 1024 // element_size
    flush_buffer = torch.empty(
        element_count,
        device=device,
        dtype=torch.int32,
    )
    flush_buffer.add_(1)
    sync_device(device)


def _snapshot_output_tensors(output) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, tensor.detach().clone())
        for name, tensor in flatten_outputs(output)
    ]


def _compare_graph_outputs(
    eager_snapshot: list[tuple[str, torch.Tensor]],
    graph_output,
    *,
    atol: float,
    rtol: float,
    min_cosine: float | None,
    max_rel_l2: float | None,
) -> dict[str, object]:
    if min_cosine is not None or max_rel_l2 is not None:
        if min_cosine is not None and max_rel_l2 is not None:
            policy_type = "cosine_rel_l2"
        elif min_cosine is not None:
            policy_type = "cosine"
        else:
            policy_type = "rel_l2"
        policy = {
            "type": policy_type,
            "min_cosine": min_cosine,
            "max_rel_l2": max_rel_l2,
        }
        policy = {key: value for key, value in policy.items() if value is not None}
    else:
        policy = {"type": "allclose", "atol": atol, "rtol": rtol}
    graph_outputs = flatten_outputs(graph_output)
    if len(eager_snapshot) != len(graph_outputs):
        return {
            "passed": False,
            "outputs": [],
            "policy": policy,
            "error": (
                "Output count mismatch: "
                f"eager={len(eager_snapshot)}, graph={len(graph_outputs)}"
            ),
        }

    output_diffs = []
    for (eager_name, eager_tensor), (graph_name, graph_tensor) in zip(
        eager_snapshot,
        graph_outputs,
    ):
        if eager_name != graph_name:
            return {
                "passed": False,
                "outputs": [],
                "policy": policy,
                "error": (
                    "Output path mismatch: "
                    f"eager={eager_name!r}, graph={graph_name!r}"
                ),
            }
        output_diff = asdict(
            _compare_output_tensors(
                eager_tensor,
                graph_tensor,
                name=_flatten_output_name(eager_name),
                atol=atol,
                rtol=rtol,
            )
        )
        output_diff["cosine_similarity"] = None
        output_diff["relative_l2"] = None
        if (min_cosine is not None or max_rel_l2 is not None) and (
            torch.is_floating_point(eager_tensor)
            or torch.is_floating_point(graph_tensor)
        ):
            if eager_tensor.shape != graph_tensor.shape:
                output_diff["passed"] = False
            else:
                eager_float = eager_tensor.detach().float().flatten()
                graph_float = graph_tensor.detach().float().flatten()
                finite = bool(
                    torch.isfinite(eager_float).all()
                    and torch.isfinite(graph_float).all()
                )
                if not finite:
                    output_diff["passed"] = False
                    output_diff["error"] = "Non-finite eager or graph output"
                elif eager_float.numel() == 0:
                    output_diff["cosine_similarity"] = 1.0
                    output_diff["relative_l2"] = 0.0
                    output_diff["passed"] = True
                else:
                    denominator = eager_float.norm() * graph_float.norm()
                    if float(denominator.item()) == 0.0:
                        cosine = 1.0 if torch.equal(eager_float, graph_float) else 0.0
                    else:
                        cosine = float(
                            torch.dot(eager_float, graph_float).div(denominator).item()
                        )
                    relative_l2_denominator = eager_float.norm().clamp_min(1e-12)
                    relative_l2 = float(
                        (eager_float - graph_float)
                        .norm()
                        .div(relative_l2_denominator)
                        .item()
                    )
                    output_diff["cosine_similarity"] = cosine
                    output_diff["relative_l2"] = relative_l2
                    cosine_passed = min_cosine is None or cosine >= min_cosine
                    rel_l2_passed = (
                        max_rel_l2 is None or relative_l2 <= max_rel_l2
                    )
                    output_diff["passed"] = cosine_passed and rel_l2_passed
        output_diffs.append(output_diff)

    return {
        "passed": all(bool(output["passed"]) for output in output_diffs),
        "outputs": output_diffs,
        "policy": policy,
        "error": None,
    }


def _measure_cuda_graph_samples(
    *,
    bench_fn,
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
    cache_flush_mb: int,
    atol: float,
    rtol: float,
    min_cosine: float | None = None,
    max_rel_l2: float | None = None,
) -> _MeasurementResult:
    eager_output = None
    for _ in range(max(1, warmup_iters)):
        eager_output = bench_fn()
    sync_device(device)
    eager_snapshot = _snapshot_output_tensors(eager_output)

    capture_started = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = bench_fn()
    sync_device(device)
    capture_time_ms = (time.perf_counter() - capture_started) * 1000.0

    for _ in range(warmup_iters):
        _flush_cuda_cache(cache_flush_mb, device)
        graph.replay()
        sync_device(device)

    samples = []
    for _ in range(bench_iters):
        _flush_cuda_cache(cache_flush_mb, device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        graph.replay()
        end_event.record()
        sync_device(device)
        samples.append(
            PerformanceSample(
                end_to_end_time_ms=float(start_event.elapsed_time(end_event))
            )
        )

    graph_correctness = _compare_graph_outputs(
        eager_snapshot,
        graph_output,
        atol=atol,
        rtol=rtol,
        min_cosine=min_cosine,
        max_rel_l2=max_rel_l2,
    )
    if not graph_correctness["passed"]:
        raise RuntimeError(
            "CUDA Graph replay output differs from eager output: "
            f"{graph_correctness}"
        )

    return _MeasurementResult(
        samples=samples,
        capture_time_ms=capture_time_ms,
        graph_correctness=graph_correctness,
    )


def _measure_runner_samples(
    model: torch.nn.Module,
    inputs,
    device: torch.device,
    *,
    warmup_iters: int,
    bench_iters: int,
    collect_kernel_events: bool = False,
    perf_timeout_s: int | float | None = _DEFAULT_PERF_TIMEOUT_S,
    benchmark_mode: str = EAGER_BENCHMARK_MODE,
    cache_flush_mb: int = 1024,
    atol: float = 1e-2,
    rtol: float = 0.05,
    min_cosine: float | None = None,
    max_rel_l2: float | None = None,
) -> _MeasurementResult:
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
    benchmark_inputs = clone_model_inputs(inputs)

    def _bench_fn():
        return model(*benchmark_inputs.args, **benchmark_inputs.kwargs)

    with torch.inference_mode(), candidate_timeout(perf_timeout_s):
        if benchmark_mode == CUDA_GRAPH_REPLAY_BENCHMARK_MODE:
            return _measure_cuda_graph_samples(
                bench_fn=_bench_fn,
                device=device,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
                cache_flush_mb=cache_flush_mb,
                atol=atol,
                rtol=rtol,
                min_cosine=min_cosine,
                max_rel_l2=max_rel_l2,
            )

        measurement = _measure_eager_samples(
            bench_fn=_bench_fn,
            device=device,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )

        if not collect_kernel_events:
            return measurement

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
        return _MeasurementResult(
            samples=measurement.samples,
            kernel_events=kernel_events,
        )


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
    benchmark_mode: str = EAGER_BENCHMARK_MODE,
    cuda_graph_cache_flush_mb: int = 1024,
    atol: float = 1e-2,
    rtol: float = 0.05,
    graph_min_cosine: float | None = None,
    graph_max_rel_l2: float | None = None,
) -> PerformanceShapeResult:
    """Benchmark the candidate module under one shape configuration."""
    if warmup_iters < 0:
        return PerformanceShapeResult(
            benchmark_mode=benchmark_mode,
            error="warmup_iters must be non-negative",
        )
    if bench_iters < 1:
        return PerformanceShapeResult(
            benchmark_mode=benchmark_mode,
            error="bench_iters must be at least 1",
        )
    if benchmark_mode not in SUPPORTED_BENCHMARK_MODES:
        return PerformanceShapeResult(
            benchmark_mode=benchmark_mode,
            error=(
                f"unsupported benchmark_mode={benchmark_mode!r}; "
                f"expected one of {sorted(SUPPORTED_BENCHMARK_MODES)}"
            ),
        )
    if graph_min_cosine is not None and not 0.0 <= graph_min_cosine <= 1.0:
        return PerformanceShapeResult(
            benchmark_mode=benchmark_mode,
            error="graph_min_cosine must be between 0.0 and 1.0",
        )
    if graph_max_rel_l2 is not None and graph_max_rel_l2 < 0.0:
        return PerformanceShapeResult(
            benchmark_mode=benchmark_mode,
            error="graph_max_rel_l2 must be non-negative",
        )

    try:
        resolved_device = get_device(device)
        if (
            benchmark_mode == CUDA_GRAPH_REPLAY_BENCHMARK_MODE
            and resolved_device.type != "cuda"
        ):
            return PerformanceShapeResult(
                benchmark_mode=benchmark_mode,
                error="cuda_graph_replay requires a CUDA device",
            )
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
        measurement = _measure_runner_samples(
            loaded_candidate.model,
            inputs,
            resolved_device,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            collect_kernel_events=collect_kernel_events,
            perf_timeout_s=perf_timeout_s,
            benchmark_mode=benchmark_mode,
            cache_flush_mb=cuda_graph_cache_flush_mb,
            atol=atol,
            rtol=rtol,
            min_cosine=graph_min_cosine,
            max_rel_l2=graph_max_rel_l2,
        )
    except CandidateTimeoutError as timeout_error:
        return PerformanceShapeResult(
            input_artifact=artifact,
            benchmark_mode=benchmark_mode,
            error=str(timeout_error),
        )
    except Exception:
        return PerformanceShapeResult(
            benchmark_mode=benchmark_mode,
            cache_flush_mb=(
                cuda_graph_cache_flush_mb
                if benchmark_mode == CUDA_GRAPH_REPLAY_BENCHMARK_MODE
                else None
            ),
            error=traceback.format_exc(),
        )

    return PerformanceShapeResult(
        input_artifact=artifact,
        samples=measurement.samples,
        kernel_events=measurement.kernel_events,
        benchmark_mode=benchmark_mode,
        capture_time_ms=measurement.capture_time_ms,
        cache_flush_mb=(
            cuda_graph_cache_flush_mb
            if benchmark_mode == CUDA_GRAPH_REPLAY_BENCHMARK_MODE
            else None
        ),
        graph_correctness=measurement.graph_correctness,
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

        measurement = _measure_runner_samples(
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
        samples=measurement.samples,
    )
