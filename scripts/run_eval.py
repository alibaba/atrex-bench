"""Run the full Atrex-Bench evaluation pipeline for one operator across all its shapes."""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from atrex_bench.eval import (
    CompileResult,
    CorrectnessCase,
    CorrectnessShapeResult,
    FlydslComputeRatioShape,
    FlydslComputeRatioSummary,
    KernelTimingEvent,
    OutputDiff,
    PerformanceSample,
    PerformanceShapeResult,
    benchmark_performance,
    benchmark_reference_torch_compile,
    check_compilation,
    check_correctness,
    compute_flydsl_compute_ratio_for_shape,
    summarize_flydsl_compute_ratio,
)
from atrex_bench.eval._runtime import (
    get_accelerator_backend,
    get_core_package_versions,
    get_python_version,
    infer_target_dsl,
)
from atrex_bench.utils import get_timestamp, save_json

_SKIP_COMPILE_REASON = "Skipped because compile stage failed."
_PENDING_EVAL_REASON = "Evaluation did not complete."
_TORCH_COMPILE_EVAL_MODE = "torch_compile_reference"
_CANDIDATE_EVAL_MODE = "candidate"
_TORCH_COMPILE_SKIP_REASON = "Skipped in torch_compile_reference mode."
_DEFAULT_CONFIG_VERSION = "v1"

# Per-shape sub-worker emits stderr lines that we surface back into the
# eval_result.json reason field. Cap how many trailing lines we keep so a
# single fault doesn't bloat the result file.
_SHAPE_SUBWORKER_STDERR_TAIL_LINES = 15


_SHAPE_WALL_TIMEOUT_FLOOR_S = 60.0
_REFERENCE_OVERHEAD_BUDGET_S = 60.0


def _derived_shape_wall_timeout_s(
    candidate_timeout_s: int | float,
    perf_timeout_s: int | float,
    *,
    num_correctness_cases: int,
) -> float:
    """OS-level wall-clock ceiling for a single per-shape sub-worker.

    Two user-facing knobs, summed with a small reference overhead:
      * ``candidate_timeout_s`` bounds each candidate touch (instantiate +
        each correctness forward call). The reference is never timed out.
      * ``perf_timeout_s`` bounds the WHOLE perf phase (do_bench measurement
        + profiler-breakdown loop).

    Worst-case per-shape path:
        instantiate              <= candidate_timeout_s         (1x)
        N correctness cases      <= candidate_timeout_s * cases (Nx)
        perf phase               <= perf_timeout_s              (whole phase)
        reference cold-start     <= REFERENCE_OVERHEAD          (60s fixed)

    Default (cases=5, candidate=60s, perf=600s):
        60 + 60*5 + 600 + 60 = 1020s = 17 min/shape ceiling

    Healthy candidates run far below this; the ceiling exists to OS-SIGKILL
    C-extension hangs (MLIR compiler, wedged torch.cuda.synchronize) that
    in-Python SIGALRM cannot interrupt.
    """
    ceiling = (
        float(candidate_timeout_s) * (1 + num_correctness_cases)
        + float(perf_timeout_s)
        + _REFERENCE_OVERHEAD_BUDGET_S
    )
    return max(_SHAPE_WALL_TIMEOUT_FLOOR_S, ceiling)


def _log(message: str) -> None:
    """Emit a human-facing progress log line without changing JSON outputs."""
    print(message, file=sys.stderr, flush=True)


def _elapsed_s(start_time: float) -> float:
    """Return elapsed seconds from a perf_counter start."""
    return time.perf_counter() - start_time


def _first_line(value: str | None) -> str:
    """Return the first non-empty line from a possibly multiline string."""
    if not value:
        return ""
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _performance_sample_count(result: PerformanceShapeResult) -> int:
    """Return how many performance samples a shape produced."""
    return len(result.samples)


def _run_subprocess_with_live_stderr(
    cmd: list[str],
    *,
    cwd: str,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess while mirroring its stderr to this process in real time."""
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _drain_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            stdout_chunks.append(line)

    def _drain_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_chunks.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()

    stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


# ---------------------------------------------------------------------------
# Reference / candidate path helpers
# ---------------------------------------------------------------------------


def _resolve_reference_bundle(reference_dir: Path) -> Path:
    """Validate that reference_dir contains the new-schema 4 required files.

    Returns the resolved reference.py path. Raises FileNotFoundError when
    any required file is missing.
    """
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
    required = [
        reference_dir / "reference.py",
        reference_dir / "input.py",
        reference_dir / "shapes.json",
        reference_dir / "metadata.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required reference file not found: {path}")
    return reference_dir / "reference.py"


def _validate_candidate_path(input_path: Path) -> Path:
    """Validate that the candidate file path is a Python file that exists."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix != ".py":
        raise ValueError(f"Input file must be a Python file: {input_path}")
    return input_path


def _kernel_name(reference_dir: Path, input_path: Path) -> str:
    """Operator name for artifact paths and kernel.name; defaults to dir name."""
    if reference_dir.name:
        return reference_dir.name
    if input_path.stem:
        return input_path.stem
    return "unknown_kernel"


# ---------------------------------------------------------------------------
# JSON file loaders
# ---------------------------------------------------------------------------


def _load_metadata_json(reference_dir: Path) -> dict[str, object]:
    """Load metadata.json next to reference.py.

    Tolerates a missing file by returning an empty dict so the pre-flight
    fallback payload (built before the worker even runs) can still describe
    the kernel block with ``id``/``dtype`` set to null.
    """
    metadata_path = reference_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_shapes_json(reference_dir: Path) -> dict[str, dict[str, object]]:
    raw = json.loads((reference_dir / "shapes.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"shapes.json top level must be a dict: {reference_dir / 'shapes.json'}")
    return raw


def _shape_ids(reference_dir: Path) -> list[str]:
    """Return all shape_ids declared in shapes.json, sorted by string order."""
    return sorted(_load_shapes_json(reference_dir).keys())


# ---------------------------------------------------------------------------
# Environment / kernel block assembly
# ---------------------------------------------------------------------------


def _eval_id() -> str:
    """Unique-per-eval identifier: ``<compact UTC timestamp>-<8 hex>``.

    Example: ``"20260423T120044Z-a3f7b2e1"``.

    The timestamp prefix keeps IDs time-sortable and human-scannable;
    the 8-hex (32-bit) suffix from ``secrets.token_hex(4)`` prevents
    collisions when multiple evals start in the same wall-clock second
    (which is otherwise possible since the timestamp portion has only
    second precision).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def _gpu_info() -> tuple[str | None, str | None]:
    """Best-effort GPU name + architecture string. Returns (name, arch).

    Routed by ``get_accelerator_backend()``:
      * rocm  -> ``props.gcnArchName`` (e.g. ``gfx942:sramecc+:xnack-``)
      * cuda  -> ``sm_<major><minor>`` (e.g. ``sm_90``)

    PyTorch 2.9+ exposes ``gcnArchName`` on NVIDIA builds too — but it is
    set to the device-name string, which would duplicate
    ``gpu_name`` if we naively trusted it. Pick the right field by backend
    instead of relying on a truthiness fallback.
    """
    if not torch.cuda.is_available():
        return None, None
    try:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        backend = get_accelerator_backend()
        if backend == "rocm":
            arch = getattr(props, "gcnArchName", None) or None
        else:
            major = getattr(props, "major", None)
            minor = getattr(props, "minor", None)
            arch = (
                f"sm_{major}{minor}"
                if major is not None and minor is not None
                else None
            )
        return name, arch
    except Exception:
        return None, None


def _runtime_version() -> str | None:
    """ROCm or CUDA runtime version string from torch.version."""
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        return hip_version
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        return cuda_version
    return None


def _driver_version() -> str | None:
    """Best-effort driver version via vendor smi tool."""
    backend = get_accelerator_backend()
    if backend == "rocm":
        return _first_nonheader_line(
            ["rocm-smi", "--showdriverversion", "--csv"], skip_prefixes=("name", "device")
        )
    if backend == "cuda":
        return _first_nonheader_line(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        )
    return None


def _first_nonheader_line(cmd: list[str], skip_prefixes: tuple[str, ...] = ()) -> str | None:
    """Run cmd, return the first non-empty / non-header stdout line."""
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(lowered.startswith(prefix) for prefix in skip_prefixes):
            continue
        return line
    return None


def _ptl_state() -> str | None:
    """PTL upgrade state reported by ``amd-smi static -l`` (AMD only).

    The value (e.g. ``"N/A"`` on a non-PTL machine) is returned verbatim
    so downstream consumers can correlate performance samples with the
    machine's PTL firmware state. Returns ``None`` when the accelerator
    backend is not ROCm, when ``amd-smi`` is unavailable or fails, or
    when the output does not contain a ``PTL_STATE:`` line.
    """
    if get_accelerator_backend() != "rocm":
        return None
    try:
        completed = subprocess.run(
            ["amd-smi", "static", "-l"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if line.startswith("PTL_STATE:"):
            value = line.split(":", 1)[1].strip()
            return value or None
    return None


def _build_environment(*, clock_locked: bool) -> dict[str, object]:
    """Build the environment block per the data schema spec, Section 7.5."""
    gpu_name, gpu_arch = _gpu_info()
    versions = get_core_package_versions()
    env: dict[str, object] = {
        "gpu_name": gpu_name,
        "gpu_arch": gpu_arch,
        "accelerator_backend": get_accelerator_backend(),
        "driver_version": _driver_version(),
        "runtime_version": _runtime_version(),
        "clock_locked": clock_locked,
        "PTL_STATE": _ptl_state(),
        "torch_version": versions.get("torch"),
        "python_version": get_python_version(),
    }
    for dsl_pkg in ("triton", "gluon", "flydsl", "cutedsl"):
        version_str = versions.get(dsl_pkg)
        if version_str:
            env[f"{dsl_pkg}_version"] = version_str
    return env


def _build_kernel(reference_dir: Path) -> dict[str, object]:
    """Build the kernel block from metadata.json + dir name."""
    metadata = _load_metadata_json(reference_dir)
    return {
        "name": reference_dir.name,
        "id": metadata.get("id"),
        "dtype": metadata.get("dtype"),
    }


def _build_runner_config(
    *,
    config_version: str,
    mode: str,
    atol: float,
    rtol: float,
    num_correctness_cases: int,
    warmup_iters: int,
    bench_iters: int,
    candidate_timeout_s: int | float | None = None,
) -> dict[str, object]:
    return {
        "config_version": config_version,
        "mode": mode,
        "atol": atol,
        "rtol": rtol,
        "num_correctness_cases": num_correctness_cases,
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "candidate_timeout_s": candidate_timeout_s,
    }


# ---------------------------------------------------------------------------
# Stage result → JSON serializers
# ---------------------------------------------------------------------------


def _serialize_correctness_shapes(
    correctness: dict[str, CorrectnessShapeResult],
) -> dict[str, dict[str, object]]:
    return {
        shape_id: {"cases": [asdict(case) for case in result.cases]}
        for shape_id, result in correctness.items()
    }


def _serialize_performance_shapes(
    performance: dict[str, PerformanceShapeResult],
) -> dict[str, dict[str, object]]:
    return {
        shape_id: {
            "input_artifact": result.input_artifact,
            "samples": [asdict(sample) for sample in result.samples],
            "kernel_events": [asdict(event) for event in result.kernel_events],
            "observed_kernels": result.observed_kernels,
            "error": result.error,
        }
        for shape_id, result in performance.items()
    }


def _serialize_flydsl_compute_ratio(
    summary: FlydslComputeRatioSummary,
) -> dict[str, object]:
    """Render the top-level flydsl_compute_ratio block for eval_result.json."""
    return {
        "average": summary.average,
        "valid_shape_count": summary.valid_shape_count,
        "total_shape_count": summary.total_shape_count,
        "shapes": {
            shape_id: {
                "ratio": shape.ratio,
                "flydsl_device_time_us": shape.flydsl_device_time_us,
                "total_e2e_time_us": shape.total_e2e_time_us,
                "kernel_breakdown": [asdict(item) for item in shape.kernel_breakdown],
                "error": shape.error,
            }
            for shape_id, shape in summary.shapes.items()
        },
    }


def _is_compile_failure(result: CorrectnessShapeResult) -> bool:
    """Determine whether a failed correctness result is actually a compilation failure.

    For DSLs like flydsl, actual kernel compilation (AOT ``@kernel`` compile)
    happens during the first ``model.forward()`` call — NOT during the Python
    import checked by ``check_compilation()``.  When this per-shape compilation
    times out or crashes, the correctness stage records a failed first case
    with no outputs.  Semantically this is a *compile* failure, not a
    *correctness* failure.

    Returns True when the shape never produced tensor output (instantiation
    failed, first forward timed out, or sub-worker crashed), indicating the
    kernel could not be compiled / executed for this shape.
    """
    if result.status != "failed":
        return False
    # Empty cases → sub-worker crashed or instantiation failed before
    # any correctness case could run.
    if not result.cases:
        return True
    # First case has an error but produced no outputs → forward() call
    # failed (timeout, crash, exception) before producing any tensor.
    first_case = result.cases[0]
    if first_case.error is not None and not first_case.outputs:
        return True
    return False


def _compute_per_shape_compile(
    module_compile: CompileResult,
    correctness_per_shape: dict[str, CorrectnessShapeResult],
    compile_succeeded_per_shape: dict[str, bool | None] | None = None,
) -> dict[str, CompileResult]:
    """Derive per-shape compile status from artifact detection + heuristic fallback.

    Two sources of truth, checked in priority order:

    1. **Artifact-based** (``compile_succeeded_per_shape``): The flydsl
       tracker inside each per-shape sub-worker instruments
       ``JitFunction.__call__`` to detect whether the MLIR pass pipeline
       produced a ``CompiledArtifact``.  ``True`` means the artifact exists
       (compilation succeeded, even if execution later timed out / crashed);
       ``False`` means the pipeline did not finish.

    2. **Heuristic fallback** (``_is_compile_failure``): When the artifact
       signal is unavailable (non-flydsl candidate, sub-worker crash, or
       ``compile_succeeded`` is None), we fall back to inspecting the
       correctness result: if the first forward call produced no tensor
       output, we treat it as a compile failure.
    """
    if module_compile.status != "passed":
        return {
            shape_id: CompileResult(status="failed", reason=module_compile.reason)
            for shape_id in correctness_per_shape
        }
    succeeded_map = compile_succeeded_per_shape or {}
    per_shape: dict[str, CompileResult] = {}
    for shape_id, correctness_result in correctness_per_shape.items():
        artifact_signal = succeeded_map.get(shape_id)
        if artifact_signal is True:
            # Authoritative: compilation artifact exists.
            per_shape[shape_id] = CompileResult(status="passed", reason=None)
        elif artifact_signal is False:
            # Authoritative: compilation was attempted but artifact was NOT
            # produced — the MLIR pipeline timed out or crashed.
            per_shape[shape_id] = CompileResult(
                status="failed",
                reason=correctness_result.reason,
            )
        else:
            # No artifact signal (non-flydsl, sub-worker crash, etc.) →
            # fall back to output-based heuristic.
            if _is_compile_failure(correctness_result):
                per_shape[shape_id] = CompileResult(
                    status="failed",
                    reason=correctness_result.reason,
                )
            else:
                per_shape[shape_id] = CompileResult(status="passed", reason=None)
    return per_shape


def _serialize_passed(
    compile_result: CompileResult,
    correctness: dict[str, CorrectnessShapeResult],
    compile_succeeded_per_shape: dict[str, bool | None] | None = None,
) -> dict[str, object]:
    per_shape_compile = _compute_per_shape_compile(
        compile_result, correctness, compile_succeeded_per_shape,
    )

    return {
        "compile": {
            shape_id: {
                "status": result.status,
                "reason": result.reason,
            }
            for shape_id, result in per_shape_compile.items()
        },
        "correctness": {
            shape_id: {
                "status": result.status,
                "reason": result.reason,
            }
            for shape_id, result in correctness.items()
        },
    }


# ---------------------------------------------------------------------------
# Per-shape sub-worker: dataclass <-> JSON round-trip
# ---------------------------------------------------------------------------


def _correctness_to_payload(result: CorrectnessShapeResult) -> dict[str, object]:
    """Serialize a CorrectnessShapeResult for the sub-worker -> parent JSON channel."""
    return asdict(result)


def _correctness_from_payload(payload: dict[str, object]) -> CorrectnessShapeResult:
    """Reconstruct a CorrectnessShapeResult from the sub-worker JSON payload."""
    raw_cases = payload.get("cases") or []
    cases: list[CorrectnessCase] = []
    for raw_case in raw_cases:
        outputs = [
            OutputDiff(**raw_output) for raw_output in (raw_case.get("outputs") or [])
        ]
        cases.append(
            CorrectnessCase(
                input_artifact=raw_case.get("input_artifact"),
                outputs=outputs,
                error=raw_case.get("error"),
            )
        )
    return CorrectnessShapeResult(
        status=str(payload["status"]),
        reason=payload.get("reason"),
        cases=cases,
    )


def _performance_to_payload(result: PerformanceShapeResult) -> dict[str, object]:
    """Serialize a PerformanceShapeResult for the sub-worker -> parent JSON channel."""
    return asdict(result)


def _performance_from_payload(payload: dict[str, object]) -> PerformanceShapeResult:
    """Reconstruct a PerformanceShapeResult from the sub-worker JSON payload."""
    samples = [PerformanceSample(**raw) for raw in (payload.get("samples") or [])]
    kernel_events = [
        KernelTimingEvent(**raw) for raw in (payload.get("kernel_events") or [])
    ]
    return PerformanceShapeResult(
        input_artifact=payload.get("input_artifact"),
        samples=samples,
        kernel_events=kernel_events,
        error=payload.get("error"),
        observed_kernels=payload.get("observed_kernels"),
    )


def _summarize_subworker_failure(
    *,
    returncode: int,
    stderr: str,
) -> str:
    """Compose the ``CorrectnessShapeResult.reason`` shown when a sub-worker dies."""
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIG{signal_number}"
        head = f"Per-shape sub-worker exited with signal {signal_name}."
    else:
        head = f"Per-shape sub-worker exited with code {returncode}."
    stderr = stderr.strip()
    if not stderr:
        return head
    lines = stderr.splitlines()
    tail = lines[-_SHAPE_SUBWORKER_STDERR_TAIL_LINES:]
    return head + "\nstderr (tail):\n" + "\n".join(tail)


# ---------------------------------------------------------------------------
# Payload assembly + skipped placeholders
# ---------------------------------------------------------------------------


def _empty_correctness_for_shapes(
    shape_ids: list[str], *, status: str, reason: str | None
) -> dict[str, CorrectnessShapeResult]:
    return {
        shape_id: CorrectnessShapeResult(status=status, reason=reason)
        for shape_id in shape_ids
    }


def _empty_performance_for_shapes(
    shape_ids: list[str],
) -> dict[str, PerformanceShapeResult]:
    return {shape_id: PerformanceShapeResult() for shape_id in shape_ids}


def _build_flydsl_compute_ratio_summary(
    candidate_path: Path | None,
    dsl: str,
    performance_per_shape: dict[str, PerformanceShapeResult],
) -> FlydslComputeRatioSummary:
    """Classify per-shape kernel events and aggregate the flydsl compute ratio.

    For non-flydsl candidates every shape gets ``ratio=0.0`` with an explanatory
    ``error`` field (per the runner policy: "non-flydsl is treated as error +
    ratio 0"); the unweighted average across shapes is 0.0 in that case. For
    flydsl candidates the per-shape ratio is computed from the kernel events
    captured during the bench loop.
    """
    per_shape: dict[str, FlydslComputeRatioShape] = {}
    for shape_id, perf_result in performance_per_shape.items():
        if candidate_path is None:
            per_shape[shape_id] = FlydslComputeRatioShape(
                ratio=0.0,
                error=f"non-flydsl candidate: dsl={dsl}",
            )
            continue
        per_shape[shape_id] = compute_flydsl_compute_ratio_for_shape(
            candidate_path,
            perf_result,
            dsl=dsl,
        )
    return summarize_flydsl_compute_ratio(per_shape)


def _build_eval_payload(
    *,
    reference_dir: Path,
    candidate_path: Path | None,
    runner_config: dict[str, object],
    environment: dict[str, object],
    eval_id: str,
    compile_result: CompileResult,
    correctness_per_shape: dict[str, CorrectnessShapeResult],
    performance_per_shape: dict[str, PerformanceShapeResult],
    eval_mode: str = _CANDIDATE_EVAL_MODE,
    error: str | None = None,
    emit_flydsl_compute_ratio: bool = True,
    compile_succeeded_per_shape: dict[str, bool | None] | None = None,
) -> dict[str, object]:
    """Build a payload that matches the data schema spec, Section 7."""
    dsl = infer_target_dsl(candidate_path) if candidate_path is not None else "unknown"
    payload: dict[str, object] = {
        "kernel": _build_kernel(reference_dir),
        "dsl": dsl,
        "eval_mode": eval_mode,
        "eval_id": eval_id,
        "environment": environment,
        "runner_config": runner_config,
        "passed": _serialize_passed(
            compile_result, correctness_per_shape, compile_succeeded_per_shape,
        ),
        "correctness": {"shapes": _serialize_correctness_shapes(correctness_per_shape)},
        "performance": {"shapes": _serialize_performance_shapes(performance_per_shape)},
        "error": error,
    }
    # The flydsl ratio is only meaningful when we're evaluating a candidate
    # (not the torch.compile reference) AND when the per-shape kernel events
    # were actually collected (which is disabled via --skip-kernel-attribution).
    if eval_mode == _CANDIDATE_EVAL_MODE and emit_flydsl_compute_ratio:
        summary = _build_flydsl_compute_ratio_summary(
            candidate_path, dsl, performance_per_shape
        )
        payload["flydsl_compute_ratio"] = _serialize_flydsl_compute_ratio(summary)
    return payload


# ---------------------------------------------------------------------------
# Artifact bundle (reference.py + input.py + shapes.json + metadata.json + candidate.py)
# ---------------------------------------------------------------------------


def _build_artifact_paths(
    output_root: Path,
    kernel_name: str,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """Return (artifact_dir, eval_output_path)."""
    artifact_dir = output_root / get_timestamp(timestamp) / kernel_name
    return artifact_dir, artifact_dir / "eval_result.json"


def _archive_bundle(input_path: Path, reference_dir: Path, artifact_dir: Path) -> None:
    """Copy reference.py + input.py + shapes.json + metadata.json + candidate.py to artifact_dir."""
    if input_path.is_file():
        shutil.copy2(input_path, artifact_dir / "candidate.py")
    _archive_reference_bundle(reference_dir, artifact_dir)


def _archive_reference_bundle(reference_dir: Path, artifact_dir: Path) -> None:
    """Copy reference.py + input.py + shapes.json + metadata.json to artifact_dir."""
    for filename in ("reference.py", "input.py", "shapes.json", "metadata.json"):
        source = reference_dir / filename
        if source.is_file():
            shutil.copy2(source, artifact_dir / filename)


def _resolve_checkpoint_root(artifact_dir: Path, checkpoint_dir: Path | None) -> Path:
    """Resolve where checkpoint .pt files live (default: under artifact_dir)."""
    if checkpoint_dir is None:
        return artifact_dir
    if checkpoint_dir.is_absolute():
        return checkpoint_dir
    return artifact_dir / checkpoint_dir


# ---------------------------------------------------------------------------
# Worker-process failure handling
# ---------------------------------------------------------------------------


def _format_worker_failure(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    """Format a worker-process failure message for eval_result.json.error."""
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIG{signal_number}"
        message = f"run_eval worker exited with signal {signal_name}."
    else:
        message = f"run_eval worker exited with code {returncode}."
    details = [message]
    stderr = stderr.strip()
    stdout = stdout.strip()
    if stderr:
        details.append(f"stderr:\n{stderr}")
    if stdout:
        details.append(f"stdout:\n{stdout}")
    return "\n\n".join(details)


def _load_saved_payload(eval_output_path: Path) -> dict[str, object] | None:
    """Load a previously saved eval payload, tolerating partial / corrupt files."""
    if not eval_output_path.is_file():
        return None
    try:
        loaded = json.loads(eval_output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _annotate_failure_payload(
    payload: dict[str, object], error: str
) -> dict[str, object]:
    """Attach top-level error to an existing payload (best-effort)."""
    updated = copy.deepcopy(payload)
    updated["error"] = error
    return updated


# ---------------------------------------------------------------------------
# Per-shape sub-worker: parent-side spawn + result aggregation
# ---------------------------------------------------------------------------


def _build_single_shape_worker_command(
    *,
    candidate_path: Path,
    reference_dir: Path,
    artifact_dir: Path,
    checkpoint_root: Path,
    shape_id: str,
    atol: float,
    rtol: float,
    num_correctness_cases: int,
    warmup_iters: int,
    bench_iters: int,
    shape_result_path: Path,
    collect_kernel_events: bool,
    candidate_timeout_s: int | float,
    perf_timeout_s: int | float,
) -> list[str]:
    """Build the argv for the per-shape sub-worker invocation.

    The sub-worker re-invokes this very script with ``--single-shape-worker``
    so it gets a fresh Python interpreter (and a fresh GPU context). The
    shape's output is written to ``shape_result_path``; any GPU fault leaves
    that file absent and the parent synthesizes a failed result.
    """
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-shape-worker",
        "--input",
        str(candidate_path),
        "--reference-dir",
        str(reference_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--checkpoint-root",
        str(checkpoint_root),
        "--shape-id",
        shape_id,
        "--shape-result-output",
        str(shape_result_path),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
        "--num-correctness-cases",
        str(num_correctness_cases),
        "--warmup-iters",
        str(warmup_iters),
        "--bench-iters",
        str(bench_iters),
        "--candidate-timeout-s",
        str(candidate_timeout_s),
        "--perf-timeout-s",
        str(perf_timeout_s),
    ]
    if not collect_kernel_events:
        argv.append("--skip-kernel-attribution")
    return argv


def _load_shape_result_payload(path: Path) -> dict[str, object] | None:
    """Load the per-shape sub-worker's result JSON; return None on any error."""
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    if "correctness" not in loaded or "performance" not in loaded:
        return None
    return loaded


def _run_single_shape_subprocess(
    *,
    candidate_path: Path,
    reference_dir: Path,
    artifact_dir: Path,
    checkpoint_root: Path,
    shape_results_dir: Path,
    shape_id: str,
    atol: float,
    rtol: float,
    num_correctness_cases: int,
    warmup_iters: int,
    bench_iters: int,
    collect_kernel_events: bool,
    candidate_timeout_s: int | float,
    perf_timeout_s: int | float,
) -> tuple[CorrectnessShapeResult, PerformanceShapeResult, bool | None]:
    """Spawn one subprocess to evaluate a single shape; aggregate its result.

    Returns ``(correctness, performance, compile_succeeded)`` where
    ``compile_succeeded`` is the authoritative artifact-based signal from
    the flydsl tracker (True/False), or None when unavailable.

    On a successful run, the sub-worker writes a JSON payload that we round
    -trip back into ``CorrectnessShapeResult`` + ``PerformanceShapeResult``.
    On any fault — signal kill (SIGABRT from a GPU memory access fault is
    the original motivating case), non-zero exit, malformed JSON, or a
    missing result file — we synthesize a ``status='failed'`` result with
    the captured signal name and trailing stderr lines. The eval loop then
    moves on to the next shape; previously, a single fault skipped every
    remaining shape because the worker process itself died.

    ``shape_wall_timeout_s`` is enforced by the OS (subprocess.run's
    timeout=). On expiry the kernel SIGKILLs the sub-worker, which catches
    pathological hangs that the in-Python ``candidate_timeout`` cannot
    (e.g. flydsl's @kernel AOT compile that blocks inside an MLIR C call,
    or a CUDA/HIP kernel that wedges torch.cuda.synchronize).
    """
    shape_result_path = shape_results_dir / f"{shape_id}.json"
    if shape_result_path.exists():
        shape_result_path.unlink()

    cmd = _build_single_shape_worker_command(
        candidate_path=candidate_path,
        reference_dir=reference_dir,
        artifact_dir=artifact_dir,
        checkpoint_root=checkpoint_root,
        shape_id=shape_id,
        atol=atol,
        rtol=rtol,
        num_correctness_cases=num_correctness_cases,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        shape_result_path=shape_result_path,
        collect_kernel_events=collect_kernel_events,
        candidate_timeout_s=candidate_timeout_s,
        perf_timeout_s=perf_timeout_s,
    )
    shape_wall_timeout_s = _derived_shape_wall_timeout_s(
        candidate_timeout_s,
        perf_timeout_s,
        num_correctness_cases=num_correctness_cases,
    )

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=shape_wall_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        # OS killed the sub-worker after exceeding the wall budget.
        stderr_tail = ""
        if exc.stderr:
            tail_src = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", errors="replace")
            stderr_tail = "\nstderr (tail):\n" + "\n".join(
                tail_src.splitlines()[-_SHAPE_SUBWORKER_STDERR_TAIL_LINES:]
            )
        return (
            CorrectnessShapeResult(
                status="failed",
                reason=(
                    f"Per-shape sub-worker exceeded {shape_wall_timeout_s}s "
                    f"wall-clock budget; OS-killed (SIGKILL)." + stderr_tail
                ),
            ),
            PerformanceShapeResult(),
            None,  # compile status unknown — sub-worker killed
        )
    except Exception:
        return (
            CorrectnessShapeResult(
                status="failed",
                reason="Failed to launch per-shape sub-worker:\n" + traceback.format_exc(),
            ),
            PerformanceShapeResult(),
            None,  # compile status unknown — launch failure
        )

    payload = _load_shape_result_payload(shape_result_path)
    if payload is not None and completed.returncode == 0:
        try:
            correctness = _correctness_from_payload(payload["correctness"])
            performance = _performance_from_payload(payload["performance"])
        except (KeyError, TypeError, ValueError):
            payload = None
        else:
            compile_succeeded = payload.get("compile_succeeded")
            return correctness, performance, compile_succeeded

    reason = _summarize_subworker_failure(
        returncode=completed.returncode,
        stderr=completed.stderr,
    )
    return (
        CorrectnessShapeResult(status="failed", reason=reason),
        PerformanceShapeResult(),
        None,  # compile status unknown — sub-worker crashed
    )


# ---------------------------------------------------------------------------
# Per-shape sub-worker: in-process body (runs in the spawned interpreter)
# ---------------------------------------------------------------------------


def _run_single_shape_main(
    *,
    candidate_path: Path,
    reference_dir: Path,
    artifact_dir: Path,
    checkpoint_root: Path,
    shape_id: str,
    atol: float,
    rtol: float,
    num_correctness_cases: int,
    warmup_iters: int,
    bench_iters: int,
    shape_result_path: Path,
    collect_kernel_events: bool,
    candidate_timeout_s: int | float,
    perf_timeout_s: int | float,
) -> None:
    """Run correctness + performance for one shape; emit a JSON payload.

    The parent reads ``shape_result_path``. If this process gets killed by
    a signal (GPU memory access fault, SIGABRT, OOM-killer, …) before the
    JSON is written, the parent synthesizes a failed result for the shape.
    """
    reference_path = reference_dir / "reference.py"
    # Input artifacts are now {seed, format: "manual_seed"} stored inside
    # eval_result.json -- no per-case .pt files are written. The reference
    # implementations seed deterministically via deterministic_input_seed()
    # so the artifact alone is enough to reproduce a case.

    # Monkey-patch flydsl's ``@kernel`` BEFORE importing the candidate so we
    # observe every kernel registration at runtime regardless of how the
    # candidate spelled the decorator (bare, parametric, aliased, or built
    # via string-literal kernel generators). Falls back to a no-op when
    # flydsl is not installed (non-flydsl candidates).
    from atrex_bench.eval._flydsl_tracker import (
        flydsl_compile_succeeded,
        install_flydsl_kernel_tracker,
        observed_kernel_symbols_serializable,
        uninstall_jit_compile_tracker,
    )

    tracker_installed = install_flydsl_kernel_tracker()

    correctness_result = check_correctness(
        reference_path,
        candidate_path,
        shape_id=shape_id,
        atol=atol,
        rtol=rtol,
        num_correctness_cases=num_correctness_cases,
        candidate_timeout_s=candidate_timeout_s,
    )

    # Read the artifact-based compile signal NOW (before the wrapper is
    # removed) so the value reflects only the correctness-phase compilation.
    compile_succeeded = flydsl_compile_succeeded() if tracker_installed else None

    # Remove the JitFunction.__call__ wrapper BEFORE the performance
    # benchmark so triton.do_bench measures the candidate with zero
    # tracking overhead on the hot dispatch path.
    if tracker_installed:
        uninstall_jit_compile_tracker()

    if correctness_result.status == "passed":
        performance_result = benchmark_performance(
            candidate_path,
            reference_path,
            shape_id=shape_id,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            collect_kernel_events=collect_kernel_events,
            candidate_timeout_s=candidate_timeout_s,
            perf_timeout_s=perf_timeout_s,
        )
    else:
        performance_result = PerformanceShapeResult()

    # Attach the runtime tracker payload so the parent's
    # ``compute_flydsl_compute_ratio_for_shape`` can prefer authoritative
    # ground truth over source-AST heuristics. None when flydsl wasn't
    # importable in this sub-worker (non-flydsl candidates).
    if tracker_installed:
        observed = observed_kernel_symbols_serializable()
        performance_result = PerformanceShapeResult(
            input_artifact=performance_result.input_artifact,
            samples=performance_result.samples,
            kernel_events=performance_result.kernel_events,
            error=performance_result.error,
            observed_kernels=observed,
        )

    payload = {
        "compile_succeeded": compile_succeeded,
        "correctness": _correctness_to_payload(correctness_result),
        "performance": _performance_to_payload(performance_result),
    }
    shape_result_path.parent.mkdir(parents=True, exist_ok=True)
    shape_result_path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Torch-compile reference mode: per-shape subprocess + worker
# ---------------------------------------------------------------------------


def _build_single_shape_torch_compile_worker_command(
    *,
    reference_dir: Path,
    artifact_dir: Path,
    checkpoint_root: Path,
    shape_id: str,
    warmup_iters: int,
    bench_iters: int,
    shape_result_path: Path,
) -> list[str]:
    """Build argv for one torch-compile reference shape worker."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--torch-compile-shape-worker",
        "--reference-dir",
        str(reference_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--checkpoint-root",
        str(checkpoint_root),
        "--shape-id",
        shape_id,
        "--shape-result-output",
        str(shape_result_path),
        "--warmup-iters",
        str(warmup_iters),
        "--bench-iters",
        str(bench_iters),
    ]


def _run_single_shape_torch_compile_subprocess(
    *,
    reference_dir: Path,
    artifact_dir: Path,
    checkpoint_root: Path,
    shape_results_dir: Path,
    shape_id: str,
    warmup_iters: int,
    bench_iters: int,
) -> tuple[CorrectnessShapeResult, PerformanceShapeResult]:
    """Spawn one subprocess to benchmark torch.compile(reference) for one shape."""
    shape_result_path = shape_results_dir / f"{shape_id}.json"
    if shape_result_path.exists():
        shape_result_path.unlink()

    cmd = _build_single_shape_torch_compile_worker_command(
        reference_dir=reference_dir,
        artifact_dir=artifact_dir,
        checkpoint_root=checkpoint_root,
        shape_id=shape_id,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        shape_result_path=shape_result_path,
    )

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return (
            CorrectnessShapeResult(
                status="skipped",
                reason=_TORCH_COMPILE_SKIP_REASON,
            ),
            PerformanceShapeResult(
                error="Failed to launch torch-compile shape worker:\n"
                + traceback.format_exc()
            ),
        )

    payload = _load_shape_result_payload(shape_result_path)
    if payload is not None and completed.returncode == 0:
        try:
            correctness = _correctness_from_payload(payload["correctness"])
            performance = _performance_from_payload(payload["performance"])
        except (KeyError, TypeError, ValueError):
            payload = None
        else:
            return correctness, performance

    reason = _summarize_subworker_failure(
        returncode=completed.returncode,
        stderr=completed.stderr,
    )
    return (
        CorrectnessShapeResult(
            status="skipped",
            reason=_TORCH_COMPILE_SKIP_REASON,
        ),
        PerformanceShapeResult(error=reason),
    )


def _run_single_shape_torch_compile_main(
    *,
    reference_dir: Path,
    artifact_dir: Path,
    checkpoint_root: Path,
    shape_id: str,
    warmup_iters: int,
    bench_iters: int,
    shape_result_path: Path,
) -> None:
    """Benchmark torch.compile(reference) for one shape and emit JSON."""
    reference_path = reference_dir / "reference.py"

    performance_result = benchmark_reference_torch_compile(
        reference_path,
        shape_id=shape_id,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
    )
    correctness_result = CorrectnessShapeResult(
        status="skipped",
        reason=_TORCH_COMPILE_SKIP_REASON,
    )

    payload = {
        "correctness": _correctness_to_payload(correctness_result),
        "performance": _performance_to_payload(performance_result),
    }
    shape_result_path.parent.mkdir(parents=True, exist_ok=True)
    shape_result_path.write_text(json.dumps(payload), encoding="utf-8")


def _run_torch_compile_worker(
    *,
    reference_dir: Path,
    artifact_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    checkpoint_dir: Path | None,
    config_version: str,
    clock_locked: bool,
) -> dict[str, object]:
    """Benchmark torch.compile(reference) across all shapes and persist JSON."""
    worker_started = time.perf_counter()
    eval_output_path = artifact_dir / "eval_result.json"
    environment = _build_environment(clock_locked=clock_locked)
    runner_config = _build_runner_config(
        config_version=config_version,
        mode=_TORCH_COMPILE_EVAL_MODE,
        atol=0.0,
        rtol=0.0,
        num_correctness_cases=0,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
    )
    checkpoint_root = _resolve_checkpoint_root(artifact_dir, checkpoint_dir)
    eval_id = _eval_id()
    shape_ids = _shape_ids(reference_dir)
    _log(
        f"[eval] start mode={_TORCH_COMPILE_EVAL_MODE} op={reference_dir.name} "
        f"eval_id={eval_id} shapes={len(shape_ids)} output={eval_output_path}"
    )
    compile_result = CompileResult(status="passed", reason=None)
    correctness_per_shape = _empty_correctness_for_shapes(
        shape_ids,
        status="skipped",
        reason=_TORCH_COMPILE_SKIP_REASON,
    )
    performance_per_shape = _empty_performance_for_shapes(shape_ids)

    def _persist_partial() -> None:
        partial_payload = _build_eval_payload(
            reference_dir=reference_dir,
            candidate_path=None,
            runner_config=runner_config,
            environment=environment,
            eval_id=eval_id,
            compile_result=compile_result,
            correctness_per_shape=correctness_per_shape,
            performance_per_shape=performance_per_shape,
            eval_mode=_TORCH_COMPILE_EVAL_MODE,
        )
        save_json(partial_payload, eval_output_path)

    _persist_partial()

    shape_results_dir = artifact_dir / ".shape_results"
    shape_results_dir.mkdir(parents=True, exist_ok=True)

    for index, shape_id in enumerate(shape_ids, start=1):
        shape_started = time.perf_counter()
        _log(
            f"[shape {index}/{len(shape_ids)} id={shape_id}] "
            "torch.compile reference start"
        )
        correctness_result, performance_result = (
            _run_single_shape_torch_compile_subprocess(
                reference_dir=reference_dir,
                artifact_dir=artifact_dir,
                checkpoint_root=checkpoint_root,
                shape_results_dir=shape_results_dir,
                shape_id=shape_id,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
            )
        )
        correctness_per_shape[shape_id] = correctness_result
        performance_per_shape[shape_id] = performance_result
        sample_count = _performance_sample_count(performance_result)
        status = "passed" if performance_result.error is None and sample_count else "failed"
        detail = _first_line(performance_result.error)
        suffix = f" error={detail}" if detail else ""
        _log(
            f"[shape {index}/{len(shape_ids)} id={shape_id}] "
            f"torch.compile reference {status} samples={sample_count} "
            f"elapsed={_elapsed_s(shape_started):.2f}s{suffix}"
        )
        _persist_partial()

    shutil.rmtree(shape_results_dir, ignore_errors=True)

    final_payload = _build_eval_payload(
        reference_dir=reference_dir,
        candidate_path=None,
        runner_config=runner_config,
        environment=environment,
        eval_id=eval_id,
        compile_result=compile_result,
        correctness_per_shape=correctness_per_shape,
        performance_per_shape=performance_per_shape,
        eval_mode=_TORCH_COMPILE_EVAL_MODE,
    )
    save_json(final_payload, eval_output_path)
    ok_shapes = sum(
        1
        for result in performance_per_shape.values()
        if result.error is None and result.samples
    )
    _log(
        f"[eval] done mode={_TORCH_COMPILE_EVAL_MODE} op={reference_dir.name} "
        f"perf_ok={ok_shapes}/{len(shape_ids)} elapsed={_elapsed_s(worker_started):.2f}s"
    )
    return final_payload


# ---------------------------------------------------------------------------
# Worker entrypoint (runs inside a subprocess for fault isolation)
# ---------------------------------------------------------------------------


def _run_eval_worker(
    *,
    input_path: Path,
    reference_dir: Path,
    artifact_dir: Path,
    atol: float,
    rtol: float,
    num_correctness_cases: int,
    warmup_iters: int,
    bench_iters: int,
    checkpoint_dir: Path | None,
    config_version: str,
    clock_locked: bool,
    collect_kernel_events: bool,
    candidate_timeout_s: int | float,
    perf_timeout_s: int | float,
) -> dict[str, object]:
    """Run all stages across all shapes and persist the eval_result.json."""
    worker_started = time.perf_counter()
    candidate_path = input_path
    eval_output_path = artifact_dir / "eval_result.json"
    environment = _build_environment(clock_locked=clock_locked)
    runner_config = _build_runner_config(
        config_version=config_version,
        mode=_CANDIDATE_EVAL_MODE,
        atol=atol,
        rtol=rtol,
        num_correctness_cases=num_correctness_cases,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        candidate_timeout_s=candidate_timeout_s,
    )
    checkpoint_root = _resolve_checkpoint_root(artifact_dir, checkpoint_dir)
    # One eval_id per worker invocation; reused for every incremental save
    # so a single eval has a single stable identity in the result store.
    eval_id = _eval_id()

    shape_ids = _shape_ids(reference_dir)
    _log(
        f"[eval] start mode={_CANDIDATE_EVAL_MODE} op={reference_dir.name} "
        f"eval_id={eval_id} shapes={len(shape_ids)} output={eval_output_path}"
    )

    # Stage 0: compile
    compile_started = time.perf_counter()
    _log(f"[compile] start candidate={candidate_path}")
    compile_result = check_compilation(candidate_path)
    compile_suffix = (
        ""
        if compile_result.reason is None
        else f" reason={_first_line(compile_result.reason)}"
    )
    _log(
        f"[compile] {compile_result.status} elapsed={_elapsed_s(compile_started):.2f}s"
        f"{compile_suffix}"
    )

    if compile_result.status != "passed":
        correctness_per_shape = _empty_correctness_for_shapes(
            shape_ids, status="skipped", reason=_SKIP_COMPILE_REASON
        )
        performance_per_shape = _empty_performance_for_shapes(shape_ids)
        payload = _build_eval_payload(
            reference_dir=reference_dir,
            candidate_path=candidate_path,
            runner_config=runner_config,
            environment=environment,
            eval_id=eval_id,
            compile_result=compile_result,
            correctness_per_shape=correctness_per_shape,
            performance_per_shape=performance_per_shape,
            emit_flydsl_compute_ratio=collect_kernel_events,
        )
        save_json(payload, eval_output_path)
        _log(
            f"[eval] done mode={_CANDIDATE_EVAL_MODE} op={reference_dir.name} "
            f"compile=failed elapsed={_elapsed_s(worker_started):.2f}s"
        )
        return payload

    # Stage 1 + 2: per shape. Seed both maps with skipped placeholders and
    # save after every shape so a worker crash mid-loop leaves an
    # interpretable partial payload on disk.
    correctness_per_shape: dict[str, CorrectnessShapeResult] = (
        _empty_correctness_for_shapes(
            shape_ids, status="skipped", reason="Stage did not run before worker exited."
        )
    )
    performance_per_shape: dict[str, PerformanceShapeResult] = (
        _empty_performance_for_shapes(shape_ids)
    )
    # Per-shape artifact-based compile status reported by the flydsl tracker
    # inside each sub-worker.  None entries mean the tracker was not available
    # (non-flydsl candidate, sub-worker crash).
    compile_succeeded_per_shape: dict[str, bool | None] = {
        sid: None for sid in shape_ids
    }

    def _persist_partial() -> None:
        partial_payload = _build_eval_payload(
            reference_dir=reference_dir,
            candidate_path=candidate_path,
            runner_config=runner_config,
            environment=environment,
            eval_id=eval_id,
            compile_result=compile_result,
            correctness_per_shape=correctness_per_shape,
            performance_per_shape=performance_per_shape,
            emit_flydsl_compute_ratio=collect_kernel_events,
            compile_succeeded_per_shape=compile_succeeded_per_shape,
        )
        save_json(partial_payload, eval_output_path)

    # Persist once now so the on-disk payload reflects compile=passed even
    # if the very first per-shape stage crashes the worker.
    _persist_partial()

    shape_results_dir = artifact_dir / ".shape_results"
    shape_results_dir.mkdir(parents=True, exist_ok=True)

    for index, shape_id in enumerate(shape_ids, start=1):
        shape_started = time.perf_counter()
        _log(
            f"[shape {index}/{len(shape_ids)} id={shape_id}] "
            "correctness/performance start"
        )
        correctness_result, performance_result, compile_succeeded = _run_single_shape_subprocess(
            candidate_path=candidate_path,
            reference_dir=reference_dir,
            artifact_dir=artifact_dir,
            checkpoint_root=checkpoint_root,
            shape_results_dir=shape_results_dir,
            shape_id=shape_id,
            atol=atol,
            rtol=rtol,
            num_correctness_cases=num_correctness_cases,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            collect_kernel_events=collect_kernel_events,
            candidate_timeout_s=candidate_timeout_s,
            perf_timeout_s=perf_timeout_s,
        )
        correctness_per_shape[shape_id] = correctness_result
        performance_per_shape[shape_id] = performance_result
        compile_succeeded_per_shape[shape_id] = compile_succeeded
        sample_count = _performance_sample_count(performance_result)
        reason = _first_line(correctness_result.reason or performance_result.error)
        suffix = f" reason={reason}" if reason else ""
        _log(
            f"[shape {index}/{len(shape_ids)} id={shape_id}] "
            f"correctness={correctness_result.status} samples={sample_count} "
            f"elapsed={_elapsed_s(shape_started):.2f}s{suffix}"
        )

        _persist_partial()

    # The per-shape sub-worker JSON files are intermediate; their content is
    # already merged into eval_result.json. Best-effort cleanup.
    shutil.rmtree(shape_results_dir, ignore_errors=True)

    final_payload = _build_eval_payload(
        reference_dir=reference_dir,
        candidate_path=candidate_path,
        runner_config=runner_config,
        environment=environment,
        eval_id=eval_id,
        compile_result=compile_result,
        correctness_per_shape=correctness_per_shape,
        performance_per_shape=performance_per_shape,
        emit_flydsl_compute_ratio=collect_kernel_events,
        compile_succeeded_per_shape=compile_succeeded_per_shape,
    )
    save_json(final_payload, eval_output_path)
    passed_shapes = sum(
        1
        for result in correctness_per_shape.values()
        if result.status == "passed"
    )
    failed_shapes = sum(
        1
        for result in correctness_per_shape.values()
        if result.status == "failed"
    )
    _log(
        f"[eval] done mode={_CANDIDATE_EVAL_MODE} op={reference_dir.name} "
        f"correctness_passed={passed_shapes}/{len(shape_ids)} "
        f"failed={failed_shapes} elapsed={_elapsed_s(worker_started):.2f}s"
    )
    return final_payload


# ---------------------------------------------------------------------------
# Parent entrypoint (run_eval): pre-flight, archive, spawn worker, fault-recover
# ---------------------------------------------------------------------------


def _initial_fallback_payload(
    *,
    reference_dir: Path,
    candidate_path: Path,
    runner_config: dict[str, object],
    environment: dict[str, object],
    eval_id: str,
) -> dict[str, object]:
    """Fallback payload used before any stage runs (e.g. pre-flight failure)."""
    shape_ids = _shape_ids(reference_dir) if (reference_dir / "shapes.json").is_file() else []
    return _build_eval_payload(
        reference_dir=reference_dir,
        candidate_path=candidate_path,
        runner_config=runner_config,
        environment=environment,
        eval_id=eval_id,
        compile_result=CompileResult(status="failed", reason=_PENDING_EVAL_REASON),
        correctness_per_shape=_empty_correctness_for_shapes(
            shape_ids, status="skipped", reason=_PENDING_EVAL_REASON
        ),
        performance_per_shape=_empty_performance_for_shapes(shape_ids),
        error=_PENDING_EVAL_REASON,
    )


def _initial_torch_compile_fallback_payload(
    *,
    reference_dir: Path,
    runner_config: dict[str, object],
    environment: dict[str, object],
    eval_id: str,
) -> dict[str, object]:
    """Fallback payload for torch-compile reference mode before stages run."""
    shape_ids = _shape_ids(reference_dir) if (reference_dir / "shapes.json").is_file() else []
    return _build_eval_payload(
        reference_dir=reference_dir,
        candidate_path=None,
        runner_config=runner_config,
        environment=environment,
        eval_id=eval_id,
        compile_result=CompileResult(status="failed", reason=_PENDING_EVAL_REASON),
        correctness_per_shape=_empty_correctness_for_shapes(
            shape_ids,
            status="skipped",
            reason=_TORCH_COMPILE_SKIP_REASON,
        ),
        performance_per_shape=_empty_performance_for_shapes(shape_ids),
        eval_mode=_TORCH_COMPILE_EVAL_MODE,
        error=_PENDING_EVAL_REASON,
    )


def run_torch_compile_eval(
    reference_dir: Path,
    output_root: Path,
    *,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    checkpoint_dir: Path | None = None,
    timestamp: str | None = None,
    config_version: str = _DEFAULT_CONFIG_VERSION,
    clock_locked: bool = False,
) -> dict[str, object]:
    """Benchmark torch.compile(reference) across all shapes."""
    resolved_timestamp = get_timestamp(timestamp)
    kernel_name = _kernel_name(reference_dir, reference_dir / "reference.py")
    artifact_dir, eval_output_path = _build_artifact_paths(
        output_root=output_root,
        kernel_name=kernel_name,
        timestamp=resolved_timestamp,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runner_config = _build_runner_config(
        config_version=config_version,
        mode=_TORCH_COMPILE_EVAL_MODE,
        atol=0.0,
        rtol=0.0,
        num_correctness_cases=0,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
    )
    environment = _build_environment(clock_locked=clock_locked)

    initial_payload = _initial_torch_compile_fallback_payload(
        reference_dir=reference_dir,
        runner_config=runner_config,
        environment=environment,
        eval_id=_eval_id(),
    )
    save_json(initial_payload, eval_output_path)

    try:
        _archive_reference_bundle(reference_dir, artifact_dir)
        _resolve_reference_bundle(reference_dir)
    except Exception:
        failure_payload = _annotate_failure_payload(
            _load_saved_payload(eval_output_path) or initial_payload,
            traceback.format_exc(),
        )
        save_json(failure_payload, eval_output_path)
        return failure_payload

    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--torch-compile-worker",
        "--reference-dir",
        str(reference_dir),
        "--output",
        str(output_root),
        "--warmup-iters",
        str(warmup_iters),
        "--bench-iters",
        str(bench_iters),
        "--config-version",
        config_version,
        "--artifact-dir",
        str(artifact_dir),
    ]
    if clock_locked:
        worker_command.append("--clock-locked")
    if checkpoint_dir is not None:
        worker_command.extend(["--checkpoint-dir", str(checkpoint_dir)])

    try:
        _log(
            f"[eval] launching torch-compile worker op={reference_dir.name} "
            f"output={eval_output_path}"
        )
        completed = _run_subprocess_with_live_stderr(
            worker_command,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    except Exception:
        failure_payload = _annotate_failure_payload(
            _load_saved_payload(eval_output_path) or initial_payload,
            traceback.format_exc(),
        )
        save_json(failure_payload, eval_output_path)
        return failure_payload

    saved_payload = _load_saved_payload(eval_output_path)
    if completed.returncode != 0:
        failure_payload = _annotate_failure_payload(
            saved_payload or initial_payload,
            _format_worker_failure(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            ),
        )
        save_json(failure_payload, eval_output_path)
        return failure_payload

    if saved_payload is None:
        failure_payload = _annotate_failure_payload(
            initial_payload,
            "torch-compile worker completed but did not write a valid eval_result.json.",
        )
        save_json(failure_payload, eval_output_path)
        return failure_payload

    return saved_payload


def run_eval(
    input_path: Path,
    reference_dir: Path,
    output_root: Path,
    *,
    atol: float = 1e-2,
    rtol: float = 0.05,
    num_correctness_cases: int = 1,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    checkpoint_dir: Path | None = None,
    timestamp: str | None = None,
    config_version: str = _DEFAULT_CONFIG_VERSION,
    clock_locked: bool = False,
    collect_kernel_events: bool = True,
    candidate_timeout_s: int | float = 60,
    perf_timeout_s: int | float = 600,
) -> dict[str, object]:
    """Run the full pipeline; always persist a valid eval_result.json."""
    resolved_timestamp = get_timestamp(timestamp)
    kernel_name = _kernel_name(reference_dir, input_path)
    artifact_dir, eval_output_path = _build_artifact_paths(
        output_root=output_root,
        kernel_name=kernel_name,
        timestamp=resolved_timestamp,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runner_config = _build_runner_config(
        config_version=config_version,
        mode=_CANDIDATE_EVAL_MODE,
        atol=atol,
        rtol=rtol,
        num_correctness_cases=num_correctness_cases,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        candidate_timeout_s=candidate_timeout_s,
    )
    environment = _build_environment(clock_locked=clock_locked)

    initial_payload = _initial_fallback_payload(
        reference_dir=reference_dir,
        candidate_path=input_path,
        runner_config=runner_config,
        environment=environment,
        eval_id=_eval_id(),
    )
    save_json(initial_payload, eval_output_path)

    try:
        _archive_bundle(input_path, reference_dir, artifact_dir)
        _validate_candidate_path(input_path)
        _resolve_reference_bundle(reference_dir)
    except Exception:
        failure_payload = _annotate_failure_payload(
            _load_saved_payload(eval_output_path) or initial_payload,
            traceback.format_exc(),
        )
        save_json(failure_payload, eval_output_path)
        return failure_payload

    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input",
        str(input_path),
        "--reference-dir",
        str(reference_dir),
        "--output",
        str(output_root),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
        "--num-correctness-cases",
        str(num_correctness_cases),
        "--warmup-iters",
        str(warmup_iters),
        "--bench-iters",
        str(bench_iters),
        "--candidate-timeout-s",
        str(candidate_timeout_s),
        "--perf-timeout-s",
        str(perf_timeout_s),
        "--config-version",
        config_version,
        "--worker",
        "--artifact-dir",
        str(artifact_dir),
    ]
    if clock_locked:
        worker_command.append("--clock-locked")
    if checkpoint_dir is not None:
        worker_command.extend(["--checkpoint-dir", str(checkpoint_dir)])
    if not collect_kernel_events:
        worker_command.append("--skip-kernel-attribution")

    try:
        _log(
            f"[eval] launching worker op={reference_dir.name} "
            f"candidate={input_path} output={eval_output_path}"
        )
        completed = _run_subprocess_with_live_stderr(
            worker_command,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    except Exception:
        failure_payload = _annotate_failure_payload(
            _load_saved_payload(eval_output_path) or initial_payload,
            traceback.format_exc(),
        )
        save_json(failure_payload, eval_output_path)
        return failure_payload

    saved_payload = _load_saved_payload(eval_output_path)
    if completed.returncode != 0:
        failure_payload = _annotate_failure_payload(
            saved_payload or initial_payload,
            _format_worker_failure(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            ),
        )
        save_json(failure_payload, eval_output_path)
        return failure_payload

    if saved_payload is None:
        failure_payload = _annotate_failure_payload(
            initial_payload,
            "run_eval worker completed but did not write a valid eval_result.json.",
        )
        save_json(failure_payload, eval_output_path)
        return failure_payload

    return saved_payload


# ---------------------------------------------------------------------------
# CLI exit-code derivation
# ---------------------------------------------------------------------------


def _payload_overall_passed(payload: dict[str, object]) -> bool:
    """Derive overall pass: every shape must pass both compile AND correctness."""
    passed_block = payload.get("passed", {})
    if not isinstance(passed_block, dict):
        return False

    # Per-shape compile: every shape must have status "passed".
    compile_block = passed_block.get("compile", {})
    if not isinstance(compile_block, dict) or not compile_block:
        return False
    for shape_status in compile_block.values():
        if not isinstance(shape_status, dict) or shape_status.get("status") != "passed":
            return False

    if payload.get("eval_mode") == _TORCH_COMPILE_EVAL_MODE:
        performance = payload.get("performance", {})
        performance_shapes = (
            performance.get("shapes") if isinstance(performance, dict) else None
        )
        if not isinstance(performance_shapes, dict) or not performance_shapes:
            return False
        for shape_payload in performance_shapes.values():
            if not isinstance(shape_payload, dict):
                return False
            if shape_payload.get("error") is not None:
                return False
            samples = shape_payload.get("samples")
            if not isinstance(samples, list) or not samples:
                return False
        return True

    correctness_block = passed_block.get("correctness", {})
    if not isinstance(correctness_block, dict) or not correctness_block:
        return False
    for shape_status in correctness_block.values():
        if not isinstance(shape_status, dict) or shape_status.get("status") != "passed":
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compile/correctness/performance evaluation across all shapes"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the candidate Python file exposing Model",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing reference.py, input.py, shapes.json, "
            "and metadata.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Root output directory for timestamped evaluation artifacts. "
            "Required for the top-level CLI; ignored by --worker / --single-shape-worker."
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.05, help="Relative tolerance")
    parser.add_argument(
        "--num-correctness-cases",
        type=int,
        default=1,
        help="Number of correctness cases per shape",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Warmup iterations for profiling",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=100,
        help="Benchmark iterations for profiling",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Relative or absolute root directory for correctness/performance checkpoints",
    )
    parser.add_argument(
        "--config-version",
        type=str,
        default=_DEFAULT_CONFIG_VERSION,
        help=(
            "Versioned runner config identifier (recorded under runner_config.config_version "
            "in eval_result.json)"
        ),
    )
    parser.add_argument(
        "--clock-locked",
        action="store_true",
        help="Assert that GPU clocks are pinned during measurement",
    )
    parser.add_argument(
        "--skip-kernel-attribution",
        action="store_true",
        help=(
            "Skip the per-kernel torch.profiler attribution pass that "
            "populates the top-level 'flydsl_compute_ratio' block. The bench "
            "loop wraps each forward in torch.profiler.profile when this is "
            "off (the default); pass this flag to recover the pre-attribution "
            "bench-loop overhead at the cost of dropping the ratio metric."
        ),
    )
    parser.add_argument(
        "--candidate-timeout-s",
        type=float,
        default=60,
        help=(
            "Wall-clock timeout (seconds) for each candidate touch: import + "
            "instantiate (flydsl @kernel AOT compile) AND every individual "
            "correctness forward call. Reference is not timed out. Pass <= 0 "
            "to disable. Default 60s."
        ),
    )
    parser.add_argument(
        "--perf-timeout-s",
        type=float,
        default=600,
        help=(
            "Wall-clock timeout (seconds) for the ENTIRE perf phase per shape "
            "(do_bench end-to-end measurement + the optional profiler-breakdown "
            "loop for kernel attribution). Pass <= 0 to disable. Default 600s "
            "(10 min)."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help=(
            "Benchmark torch.compile(reference Model) across all shapes. "
            "This mode does not take --input and skips candidate compile/correctness."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--torch-compile-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--single-shape-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--torch-compile-shape-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--artifact-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--shape-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--shape-result-output", type=Path, default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.torch_compile_shape_worker:
        missing = [
            name
            for name, value in (
                ("--artifact-dir", args.artifact_dir),
                ("--checkpoint-root", args.checkpoint_root),
                ("--shape-id", args.shape_id),
                ("--shape-result-output", args.shape_result_output),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "Missing required argument(s) in torch-compile-shape-worker mode: "
                + ", ".join(missing)
            )
        _run_single_shape_torch_compile_main(
            reference_dir=args.reference_dir,
            artifact_dir=args.artifact_dir,
            checkpoint_root=args.checkpoint_root,
            shape_id=args.shape_id,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            shape_result_path=args.shape_result_output,
        )
        raise SystemExit(0)

    if args.single_shape_worker:
        missing = [
            name
            for name, value in (
                ("--input", args.input),
                ("--artifact-dir", args.artifact_dir),
                ("--checkpoint-root", args.checkpoint_root),
                ("--shape-id", args.shape_id),
                ("--shape-result-output", args.shape_result_output),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "Missing required argument(s) in single-shape-worker mode: "
                + ", ".join(missing)
            )
        _run_single_shape_main(
            candidate_path=args.input,
            reference_dir=args.reference_dir,
            artifact_dir=args.artifact_dir,
            checkpoint_root=args.checkpoint_root,
            shape_id=args.shape_id,
            atol=args.atol,
            rtol=args.rtol,
            num_correctness_cases=args.num_correctness_cases,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            shape_result_path=args.shape_result_output,
            collect_kernel_events=not args.skip_kernel_attribution,
            candidate_timeout_s=args.candidate_timeout_s,
            perf_timeout_s=args.perf_timeout_s,
        )
        raise SystemExit(0)

    if args.torch_compile_worker:
        if args.artifact_dir is None:
            raise SystemExit("Missing required --artifact-dir in torch-compile-worker mode.")
        _run_torch_compile_worker(
            reference_dir=args.reference_dir,
            artifact_dir=args.artifact_dir,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            checkpoint_dir=args.checkpoint_dir,
            config_version=args.config_version,
            clock_locked=args.clock_locked,
        )
        raise SystemExit(0)

    if args.worker:
        if args.input is None:
            raise SystemExit("Missing required --input in worker mode.")
        if args.artifact_dir is None:
            raise SystemExit("Missing required --artifact-dir in worker mode.")
        _run_eval_worker(
            input_path=args.input,
            reference_dir=args.reference_dir,
            artifact_dir=args.artifact_dir,
            atol=args.atol,
            rtol=args.rtol,
            num_correctness_cases=args.num_correctness_cases,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            checkpoint_dir=args.checkpoint_dir,
            config_version=args.config_version,
            clock_locked=args.clock_locked,
            collect_kernel_events=not args.skip_kernel_attribution,
            candidate_timeout_s=args.candidate_timeout_s,
            perf_timeout_s=args.perf_timeout_s,
        )
        raise SystemExit(0)

    if args.output is None:
        raise SystemExit("--output is required for top-level run_eval CLI.")

    if args.torch_compile:
        if args.input is not None:
            raise SystemExit("--input cannot be combined with --torch-compile.")
        timestamp = get_timestamp()
        payload = run_torch_compile_eval(
            reference_dir=args.reference_dir,
            output_root=args.output,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            checkpoint_dir=args.checkpoint_dir,
            timestamp=timestamp,
            config_version=args.config_version,
            clock_locked=args.clock_locked,
        )
        kernel_name = _kernel_name(args.reference_dir, args.reference_dir / "reference.py")
        _, eval_output_path = _build_artifact_paths(
            output_root=args.output,
            kernel_name=kernel_name,
            timestamp=timestamp,
        )
        print(f"[OUTPUT] {eval_output_path}")
        raise SystemExit(0 if _payload_overall_passed(payload) else 1)

    if args.input is None:
        raise SystemExit("--input is required unless --torch-compile is set.")

    timestamp = get_timestamp()
    payload = run_eval(
        input_path=args.input,
        reference_dir=args.reference_dir,
        output_root=args.output,
        atol=args.atol,
        rtol=args.rtol,
        num_correctness_cases=args.num_correctness_cases,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        checkpoint_dir=args.checkpoint_dir,
        timestamp=timestamp,
        config_version=args.config_version,
        clock_locked=args.clock_locked,
        collect_kernel_events=not args.skip_kernel_attribution,
        candidate_timeout_s=args.candidate_timeout_s,
        perf_timeout_s=args.perf_timeout_s,
    )
    kernel_name = _kernel_name(args.reference_dir, args.input)
    _, eval_output_path = _build_artifact_paths(
        output_root=args.output,
        kernel_name=kernel_name,
        timestamp=timestamp,
    )
    print(f"[OUTPUT] {eval_output_path}")
    raise SystemExit(0 if _payload_overall_passed(payload) else 1)


if __name__ == "__main__":
    main()
