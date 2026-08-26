"""Run the full Atrex-Bench evaluation pipeline for one operator across all its shapes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
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
from atrex_bench.eval.clock_lock import (
    ClockLockConfig,
    ClockLockError,
    ClockLockReport,
    ManagedClockLock,
    clock_lock_failure_reason,
    get_clock_lock_status,
)
from atrex_bench.eval.clock_monitor import NvidiaClockMonitor
from atrex_bench.eval.correctness import (
    CORRECTNESS_MAX_REL_L2_ENV,
    configured_max_rel_l2,
)
from atrex_bench.eval.nvidia_clock import NvidiaSmi
from atrex_bench.eval.reward_hack import (
    RewardHackDetected,
    check_cuda_event_monkey_patch,
    check_eval_integrity,
    check_thread_injection,
    snapshot_critical_functions,
)
from atrex_bench.utils import get_timestamp

_SKIP_COMPILE_REASON = "Skipped because compile stage failed."
_PENDING_EVAL_REASON = "Evaluation did not complete."
_TORCH_COMPILE_EVAL_MODE = "torch_compile_reference"
_CANDIDATE_EVAL_MODE = "candidate"
_TORCH_COMPILE_SKIP_REASON = "Skipped in torch_compile_reference mode."
_CORRECTNESS_ONLY_SKIP_REASON = "Skipped in performance_only mode."
_PERFORMANCE_ONLY_SKIP_REASON = "Skipped in correctness_only mode."
_CORRECTNESS_FAILURE_PERFORMANCE_SKIP_REASON = (
    "Skipped because correctness stage did not pass."
)
_PENDING_STAGE_REASON = "Stage did not run before worker exited."
_DEFAULT_CONFIG_VERSION = "v1"
_DEFAULT_ATOL = 1e-2
_DEFAULT_RTOL = 0.05
_DEFAULT_NUM_CORRECTNESS_CASES = 1
_DEFAULT_WARMUP_ITERS = 10
_DEFAULT_BENCH_ITERS = 100
_DEFAULT_CANDIDATE_TIMEOUT_S = 60.0
_DEFAULT_PERF_TIMEOUT_S = 600.0
_SUBPROCESS_SHUTDOWN_TIMEOUT_S = 5.0
_DEFAULT_BENCHMARK_MODE = "eager"
_DEFAULT_CUDA_GRAPH_CACHE_FLUSH_MB = 1024
_DEFAULT_GRAPH_ATOL = 1e-2
_DEFAULT_GRAPH_RTOL = 0.05
_TRUST_MODE_TRUSTED = "trusted"
_TRUST_MODE_UNTRUSTED = "untrusted"
_VALIDATION_MODE_FULL = "full"
_VALIDATION_MODE_CORRECTNESS_ONLY = "correctness_only"
_VALIDATION_MODE_PERFORMANCE_ONLY = "performance_only"
_VALIDATION_MODES = frozenset(
    {
        _VALIDATION_MODE_FULL,
        _VALIDATION_MODE_CORRECTNESS_ONLY,
        _VALIDATION_MODE_PERFORMANCE_ONLY,
    }
)
_RUNNER_CONFIG_SCHEMA_VERSION = "v1"
_EVAL_MODES = frozenset({_CANDIDATE_EVAL_MODE, _TORCH_COMPILE_EVAL_MODE})
_RUNNER_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "eval_mode",
        "input",
        "reference_dir",
        "output",
        "checkpoint_dir",
        "atol",
        "rtol",
        "correctness_max_rel_l2",
        "num_correctness_cases",
        "warmup_iters",
        "bench_iters",
        "candidate_timeout_s",
        "perf_timeout_s",
        "benchmark_mode",
        "cuda_graph_cache_flush_mb",
        "graph_atol",
        "graph_rtol",
        "graph_min_cosine",
        "graph_max_rel_l2",
        "clock_locked",
        "require_clock_locked",
        "clock_lock_mode",
        "clock_lock_device",
        "gpu_clock_mhz",
        "memory_clock_mhz",
        "clock_lock_tolerance_mhz",
        "clock_lock_settle_seconds",
        "clock_lock_command_timeout_s",
        "clock_lock_require_idle",
        "clock_lock_monitor",
        "clock_lock_sample_interval_ms",
        "clock_lock_runtime_tolerance_mhz",
        "clock_lock_fail_on_deviation",
        "skip_kernel_attribution",
        "config_version",
        "trust_mode",
        "validation_mode",
    }
)


@dataclass(frozen=True)
class StageVerdict:
    """Explicit pass/fail/skip state for a per-shape evaluation stage."""

    status: str
    reason: str | None = None

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
    validation_mode: str = _VALIDATION_MODE_FULL,
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
    correctness_budget = (
        float(candidate_timeout_s) * num_correctness_cases
        if validation_mode != _VALIDATION_MODE_PERFORMANCE_ONLY
        else 0.0
    )
    performance_budget = (
        float(perf_timeout_s)
        if validation_mode != _VALIDATION_MODE_CORRECTNESS_ONLY
        else 0.0
    )
    ceiling = (
        float(candidate_timeout_s)
        + correctness_budget
        + performance_budget
        + _REFERENCE_OVERHEAD_BUDGET_S
    )
    return max(_SHAPE_WALL_TIMEOUT_FLOOR_S, ceiling)


def _log(message: str) -> None:
    """Emit a human-facing progress log line without changing JSON outputs."""
    print(message, file=sys.stderr, flush=True)


def _save_eval_json(data: dict, path: Path) -> None:
    """Persist eval artifacts as strict JSON, rejecting NaN/Inf literals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


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


def _load_runner_config_file(config_path: Path | None) -> dict[str, object]:
    """Load a JSON runner config whose keys mirror top-level CLI options."""
    if config_path is None:
        return {}
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Runner config must be a JSON object: {config_path}")
    unknown = sorted(set(raw) - _RUNNER_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "Unsupported runner config key(s): " + ", ".join(unknown)
        )
    schema_version = raw.get("schema_version")
    if (
        schema_version is not None
        and schema_version != _RUNNER_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(
            f"schema_version must be {_RUNNER_CONFIG_SCHEMA_VERSION!r}"
        )
    return raw


def _resolve_runner_option(
    name: str,
    *,
    cli_value,
    config: dict[str, object],
    default,
):
    """Resolve one runner option with CLI > config file > built-in default."""
    if cli_value is not None:
        return cli_value
    if name in config:
        return config[name]
    return default


def _resolve_path_option(
    name: str,
    *,
    cli_value: Path | None,
    config: dict[str, object],
) -> Path | None:
    """Resolve one launch path while validating its JSON representation."""
    value = _resolve_runner_option(
        name,
        cli_value=cli_value,
        config=config,
        default=None,
    )
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty path string or null")
    return Path(value)


def _resolve_eval_mode(
    cli_torch_compile: bool,
    config: dict[str, object],
) -> str:
    """Resolve the top-level evaluator mode with CLI precedence."""
    if cli_torch_compile:
        return _TORCH_COMPILE_EVAL_MODE
    mode = config.get("eval_mode", _CANDIDATE_EVAL_MODE)
    if not isinstance(mode, str) or mode not in _EVAL_MODES:
        allowed = ", ".join(sorted(_EVAL_MODES))
        raise ValueError(f"eval_mode must be one of: {allowed}")
    return mode


def _resolve_validation_mode(
    cli_mode: str | None,
    config: dict[str, object],
) -> str:
    """Resolve the candidate stage policy with CLI precedence."""
    mode = str(
        _resolve_runner_option(
            "validation_mode",
            cli_value=cli_mode,
            config=config,
            default=_VALIDATION_MODE_FULL,
        )
    )
    if mode not in _VALIDATION_MODES:
        allowed = ", ".join(sorted(_VALIDATION_MODES))
        raise ValueError(f"validation_mode must be one of: {allowed}")
    return mode


def _validation_mode_cli_args(validation_mode: str) -> list[str]:
    """Render one canonical validation mode for child-process argv."""
    if validation_mode == _VALIDATION_MODE_CORRECTNESS_ONLY:
        return ["--correctness-only"]
    if validation_mode == _VALIDATION_MODE_PERFORMANCE_ONLY:
        return ["--performance-only"]
    return []


def _runner_config_boolean(
    name: str,
    config: dict[str, object],
    *,
    default: bool = False,
) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _resolve_clock_lock_config(
    args: argparse.Namespace,
    config: dict[str, object],
) -> ClockLockConfig:
    """Resolve and validate top-level clock ownership policy."""
    cli_mode = getattr(args, "clock_lock_mode", None)
    configured_mode = config.get("clock_lock_mode")
    mode = str(cli_mode if cli_mode is not None else configured_mode or "off")
    lock_clocks = bool(getattr(args, "lock_clocks", False))
    if lock_clocks:
        if cli_mode is not None and cli_mode != "manage":
            raise ValueError(
                "--lock-clocks cannot be combined with a non-manage "
                "--clock-lock-mode"
            )
        mode = "manage"

    clock_locked = bool(getattr(args, "clock_locked", False))
    require_clock_locked = bool(getattr(args, "require_clock_locked", False))
    worker_mode = any(
        bool(getattr(args, name, False))
        for name in (
            "worker",
            "single_shape_worker",
            "torch_compile_worker",
            "torch_compile_shape_worker",
        )
    )
    if worker_mode and mode == "manage":
        raise ValueError(
            "Managed clock locking is owned by the top-level parent process."
        )
    if not worker_mode and mode == "manage" and (
        clock_locked or require_clock_locked
    ):
        raise ValueError(
            "manage mode cannot be combined with --clock-locked or "
            "--require-clock-locked"
        )
    if require_clock_locked:
        if not worker_mode and clock_locked:
            raise ValueError(
                "--clock-locked cannot be combined with --require-clock-locked"
            )
        if mode not in {"off", "external"}:
            raise ValueError(
                "--require-clock-locked cannot be combined with manage mode"
            )
        mode = "external"
    if not worker_mode and clock_locked and mode == "external":
        raise ValueError(
            "--clock-locked cannot be combined with external clock-lock mode"
        )

    management_names = (
        "clock_lock_device",
        "gpu_clock_mhz",
        "memory_clock_mhz",
        "clock_lock_tolerance_mhz",
        "clock_lock_settle_seconds",
        "clock_lock_command_timeout_s",
        "clock_lock_require_idle",
        "clock_lock_monitor",
        "clock_lock_sample_interval_ms",
        "clock_lock_runtime_tolerance_mhz",
        "clock_lock_fail_on_deviation",
    )
    resolved_management = {
        name: _resolve_runner_option(
            name,
            cli_value=getattr(args, name, None),
            config=config,
            default=None,
        )
        for name in management_names
    }
    supplied_management = any(
        getattr(args, name, None) is not None or name in config
        for name in management_names
    )
    if mode != "manage":
        if supplied_management:
            raise ValueError(
                "GPU clock device, frequency, tolerance, timeout, and idle "
                "settings are only valid in manage mode"
            )
        return ClockLockConfig(mode=mode)

    require_idle = resolved_management["clock_lock_require_idle"]
    if require_idle is None:
        require_idle = True
    if not isinstance(require_idle, bool):
        raise ValueError("clock_lock_require_idle must be a boolean")

    monitor_enabled = resolved_management["clock_lock_monitor"]
    if monitor_enabled is None:
        monitor_enabled = True
    if not isinstance(monitor_enabled, bool):
        raise ValueError("clock_lock_monitor must be a boolean")

    sample_interval_ms = resolved_management["clock_lock_sample_interval_ms"]
    if sample_interval_ms is None:
        sample_interval_ms = 10
    if (
        not isinstance(sample_interval_ms, int)
        or isinstance(sample_interval_ms, bool)
        or sample_interval_ms <= 0
    ):
        raise ValueError(
            "clock_lock_sample_interval_ms must be a positive integer"
        )

    runtime_tolerance_mhz = resolved_management[
        "clock_lock_runtime_tolerance_mhz"
    ]
    if runtime_tolerance_mhz is None:
        runtime_tolerance_mhz = 0
    if (
        not isinstance(runtime_tolerance_mhz, int)
        or isinstance(runtime_tolerance_mhz, bool)
        or runtime_tolerance_mhz < 0
    ):
        raise ValueError(
            "clock_lock_runtime_tolerance_mhz must be a non-negative integer"
        )

    fail_on_deviation = resolved_management["clock_lock_fail_on_deviation"]
    if fail_on_deviation is None:
        fail_on_deviation = True
    if not isinstance(fail_on_deviation, bool):
        raise ValueError("clock_lock_fail_on_deviation must be a boolean")

    return ClockLockConfig(
        mode="manage",
        device_selector=resolved_management["clock_lock_device"],
        graphics_mhz=resolved_management["gpu_clock_mhz"],
        memory_mhz=resolved_management["memory_clock_mhz"],
        tolerance_mhz=(
            50
            if resolved_management["clock_lock_tolerance_mhz"] is None
            else resolved_management["clock_lock_tolerance_mhz"]
        ),
        settle_seconds=(
            3.0
            if resolved_management["clock_lock_settle_seconds"] is None
            else resolved_management["clock_lock_settle_seconds"]
        ),
        command_timeout_seconds=(
            10.0
            if resolved_management["clock_lock_command_timeout_s"] is None
            else resolved_management["clock_lock_command_timeout_s"]
        ),
        require_idle=require_idle,
        monitor_enabled=monitor_enabled,
        sample_interval_ms=sample_interval_ms,
        runtime_tolerance_mhz=runtime_tolerance_mhz,
        fail_on_deviation=fail_on_deviation,
    )


def _performance_sample_count(result: PerformanceShapeResult) -> int:
    """Return how many performance samples a shape produced."""
    return len(result.samples)


def _install_untrusted_runtime_guards() -> None:
    """Install process-local guards for evaluating untrusted submissions."""
    try:
        import torch.utils.cpp_extension as cpp_extension
    except Exception:
        return

    def _blocked_cpp_extension_load(*_args, **_kwargs):
        raise RuntimeError(
            "Runtime C++/CUDA extension loading is disabled in untrusted mode. "
            "Build native extensions before evaluation or use trusted mode."
        )

    cpp_extension.load = _blocked_cpp_extension_load
    cpp_extension.load_inline = _blocked_cpp_extension_load


def _untrusted_critical_targets() -> dict[str, object]:
    """Critical evaluator functions that untrusted code must not replace."""
    correctness_globals = getattr(check_correctness, "__globals__", {})
    performance_globals = getattr(benchmark_performance, "__globals__", {})
    candidates = {
        "run_eval.check_correctness": check_correctness,
        "run_eval.benchmark_performance": benchmark_performance,
        "correctness._compare_output_tensors": correctness_globals.get(
            "_compare_output_tensors"
        ),
        "correctness.flatten_outputs": correctness_globals.get("flatten_outputs"),
        "correctness.check_plain_tensor_outputs": correctness_globals.get(
            "check_plain_tensor_outputs"
        ),
        "performance._measure_runner_samples": performance_globals.get(
            "_measure_runner_samples"
        ),
        "performance._measure_eager_samples": performance_globals.get(
            "_measure_eager_samples"
        ),
        "performance._measure_cuda_graph_samples": performance_globals.get(
            "_measure_cuda_graph_samples"
        ),
    }
    return {name: value for name, value in candidates.items() if value is not None}


def _snapshot_untrusted_critical_functions() -> dict[str, int]:
    """Snapshot critical evaluator functions before candidate import."""
    return snapshot_critical_functions(_untrusted_critical_targets())


def _check_untrusted_integrity(snapshot: dict[str, int]) -> None:
    """Validate critical evaluator functions and timing primitive."""
    check_eval_integrity(snapshot, _untrusted_critical_targets())
    check_cuda_event_monkey_patch()


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
        try:
            for line in process.stdout:
                stdout_chunks.append(line)
        except (OSError, ValueError):
            if not process.stdout.closed:
                raise

    def _drain_stderr() -> None:
        assert process.stderr is not None
        try:
            for line in process.stderr:
                stderr_chunks.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()
        except (OSError, ValueError):
            if not process.stderr.closed:
                raise

    stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait()
    except BaseException:
        cleanup_errors: list[str] = []
        needs_kill = False
        try:
            process.terminate()
        except Exception as error:
            cleanup_errors.append(f"terminate failed: {error}")
            needs_kill = True
        else:
            try:
                process.wait(timeout=_SUBPROCESS_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                needs_kill = True
            except Exception as error:
                cleanup_errors.append(f"wait after terminate failed: {error}")
                needs_kill = True

        if needs_kill:
            try:
                process.kill()
            except Exception as error:
                cleanup_errors.append(f"kill failed: {error}")
            else:
                try:
                    process.wait(timeout=_SUBPROCESS_SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired as error:
                    cleanup_errors.append(f"wait after kill timed out: {error}")
                except Exception as error:
                    cleanup_errors.append(f"wait after kill failed: {error}")

        stdout_thread.join(timeout=_SUBPROCESS_SHUTDOWN_TIMEOUT_S)
        stderr_thread.join(timeout=_SUBPROCESS_SHUTDOWN_TIMEOUT_S)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        stdout_thread.join(timeout=_SUBPROCESS_SHUTDOWN_TIMEOUT_S)
        stderr_thread.join(timeout=_SUBPROCESS_SHUTDOWN_TIMEOUT_S)
        if cleanup_errors:
            _log("[worker] interruption cleanup warning: " + "; ".join(cleanup_errors))
        raise
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
      * rocm  -> ``props.gcnArchName``, including any feature suffixes
      * cuda  -> ``sm_<major><minor>``

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
    """Build the environment block per docs/data_schema.md §七 7.5."""
    gpu_name, gpu_arch = _gpu_info()
    versions = get_core_package_versions()
    clock_status = get_clock_lock_status()
    clock_source = clock_status.source
    if clock_source is None and clock_locked:
        clock_source = "legacy-assertion"
    env: dict[str, object] = {
        "gpu_name": gpu_name,
        "gpu_arch": gpu_arch,
        "accelerator_backend": get_accelerator_backend(),
        "driver_version": _driver_version(),
        "runtime_version": _runtime_version(),
        "clock_locked": clock_locked or clock_status.locked,
        "clock_lock_detected": clock_status.locked,
        "clock_lock_source": clock_source,
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
    validation_mode: str = _VALIDATION_MODE_FULL,
    atol: float,
    rtol: float,
    num_correctness_cases: int,
    warmup_iters: int,
    bench_iters: int,
    candidate_timeout_s: int | float | None = None,
    benchmark_mode: str = "eager",
    cuda_graph_cache_flush_mb: int = 1024,
    graph_atol: float = 1e-2,
    graph_rtol: float = 0.05,
    graph_min_cosine: float | None = None,
    graph_max_rel_l2: float | None = None,
    trust_mode: str = _TRUST_MODE_TRUSTED,
    require_clock_locked: bool = False,
) -> dict[str, object]:
    return {
        "config_version": config_version,
        "mode": mode,
        "validation_mode": validation_mode,
        "atol": atol,
        "rtol": rtol,
        "correctness_max_rel_l2": configured_max_rel_l2(),
        "num_correctness_cases": num_correctness_cases,
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "candidate_timeout_s": candidate_timeout_s,
        "benchmark_mode": benchmark_mode,
        "cuda_graph_cache_flush_mb": cuda_graph_cache_flush_mb,
        "graph_atol": graph_atol,
        "graph_rtol": graph_rtol,
        "graph_min_cosine": graph_min_cosine,
        "graph_max_rel_l2": graph_max_rel_l2,
        "trust_mode": trust_mode,
        "require_clock_locked": require_clock_locked,
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
            "benchmark_mode": result.benchmark_mode,
            "capture_time_ms": result.capture_time_ms,
            "cache_flush_mb": result.cache_flush_mb,
            "graph_correctness": result.graph_correctness,
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
    performance: dict[str, StageVerdict],
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
        "performance": {
            shape_id: {
                "status": result.status,
                "reason": result.reason,
            }
            for shape_id, result in performance.items()
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
        benchmark_mode=str(payload.get("benchmark_mode") or "eager"),
        capture_time_ms=payload.get("capture_time_ms"),
        cache_flush_mb=payload.get("cache_flush_mb"),
        graph_correctness=payload.get("graph_correctness"),
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


def _empty_stage_verdicts(
    shape_ids: list[str], *, status: str, reason: str | None
) -> dict[str, StageVerdict]:
    return {
        shape_id: StageVerdict(status=status, reason=reason)
        for shape_id in shape_ids
    }


def _performance_verdict(
    *,
    validation_mode: str,
    correctness: CorrectnessShapeResult,
    performance: PerformanceShapeResult,
) -> StageVerdict:
    if validation_mode == _VALIDATION_MODE_CORRECTNESS_ONLY:
        return StageVerdict(
            status="skipped",
            reason=_PERFORMANCE_ONLY_SKIP_REASON,
        )
    if (
        validation_mode == _VALIDATION_MODE_FULL
        and correctness.status != "passed"
    ):
        return StageVerdict(
            status="skipped",
            reason=_CORRECTNESS_FAILURE_PERFORMANCE_SKIP_REASON,
        )
    if performance.error is not None:
        return StageVerdict(status="failed", reason=performance.error)
    if performance.samples:
        return StageVerdict(status="passed")
    return StageVerdict(
        status="failed",
        reason="Performance stage produced no samples.",
    )


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
    performance_verdicts: dict[str, StageVerdict],
    eval_mode: str = _CANDIDATE_EVAL_MODE,
    error: str | None = None,
    emit_flydsl_compute_ratio: bool = True,
    compile_succeeded_per_shape: dict[str, bool | None] | None = None,
) -> dict[str, object]:
    """Build a payload that matches docs/data_schema.md §七."""
    dsl = infer_target_dsl(candidate_path) if candidate_path is not None else "unknown"
    payload: dict[str, object] = {
        "kernel": _build_kernel(reference_dir),
        "dsl": dsl,
        "eval_mode": eval_mode,
        "eval_id": eval_id,
        "environment": environment,
        "runner_config": runner_config,
        "passed": _serialize_passed(
            compile_result,
            correctness_per_shape,
            performance_verdicts,
            compile_succeeded_per_shape,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _manifest_file_entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_staging_manifest(
    *,
    artifact_dir: Path,
    candidate_path: Path | None,
    reference_dir: Path,
    runner_config: dict[str, object],
    environment: dict[str, object],
    worker_command: list[str],
) -> None:
    """Write a reproducibility manifest for the staged evaluation bundle."""
    reference_files: dict[str, object] = {}
    for filename in ("reference.py", "input.py", "shapes.json", "metadata.json"):
        path = reference_dir / filename
        if path.is_file():
            reference_files[filename] = _manifest_file_entry(path)
    manifest = {
        "candidate": (
            _manifest_file_entry(candidate_path)
            if candidate_path is not None and candidate_path.is_file()
            else None
        ),
        "reference_dir": str(reference_dir),
        "reference_files": reference_files,
        "runner_config": runner_config,
        "environment": environment,
        "worker_command": worker_command,
    }
    _save_eval_json(manifest, artifact_dir / "staging_manifest.json")


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
    benchmark_mode: str = "eager",
    cuda_graph_cache_flush_mb: int = 1024,
    graph_atol: float = 1e-2,
    graph_rtol: float = 0.05,
    graph_min_cosine: float | None = None,
    graph_max_rel_l2: float | None = None,
    trust_mode: str = _TRUST_MODE_TRUSTED,
    validation_mode: str = _VALIDATION_MODE_FULL,
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
        "--benchmark-mode",
        benchmark_mode,
        "--cuda-graph-cache-flush-mb",
        str(cuda_graph_cache_flush_mb),
        "--graph-atol",
        str(graph_atol),
        "--graph-rtol",
        str(graph_rtol),
        "--trust-mode",
        trust_mode,
    ]
    if graph_min_cosine is not None:
        argv.extend(["--graph-min-cosine", str(graph_min_cosine)])
    if graph_max_rel_l2 is not None:
        argv.extend(["--graph-max-rel-l2", str(graph_max_rel_l2)])
    if not collect_kernel_events:
        argv.append("--skip-kernel-attribution")
    argv.extend(_validation_mode_cli_args(validation_mode))
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


def _subworker_failure_results(
    *, validation_mode: str, reason: str
) -> tuple[CorrectnessShapeResult, PerformanceShapeResult, None]:
    if validation_mode == _VALIDATION_MODE_PERFORMANCE_ONLY:
        return (
            CorrectnessShapeResult(
                status="skipped",
                reason=_CORRECTNESS_ONLY_SKIP_REASON,
            ),
            PerformanceShapeResult(error=reason),
            None,
        )
    return (
        CorrectnessShapeResult(status="failed", reason=reason),
        PerformanceShapeResult(),
        None,
    )


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
    benchmark_mode: str = "eager",
    cuda_graph_cache_flush_mb: int = 1024,
    graph_atol: float = 1e-2,
    graph_rtol: float = 0.05,
    graph_min_cosine: float | None = None,
    graph_max_rel_l2: float | None = None,
    trust_mode: str = _TRUST_MODE_TRUSTED,
    validation_mode: str = _VALIDATION_MODE_FULL,
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
        benchmark_mode=benchmark_mode,
        cuda_graph_cache_flush_mb=cuda_graph_cache_flush_mb,
        graph_atol=graph_atol,
        graph_rtol=graph_rtol,
        graph_min_cosine=graph_min_cosine,
        graph_max_rel_l2=graph_max_rel_l2,
        trust_mode=trust_mode,
        validation_mode=validation_mode,
    )
    shape_wall_timeout_s = _derived_shape_wall_timeout_s(
        candidate_timeout_s,
        perf_timeout_s,
        num_correctness_cases=num_correctness_cases,
        validation_mode=validation_mode,
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
            tail_src = (
                exc.stderr
                if isinstance(exc.stderr, str)
                else exc.stderr.decode("utf-8", errors="replace")
            )
            stderr_tail = "\nstderr (tail):\n" + "\n".join(
                tail_src.splitlines()[-_SHAPE_SUBWORKER_STDERR_TAIL_LINES:]
            )
        return _subworker_failure_results(
            validation_mode=validation_mode,
            reason=(
                f"Per-shape sub-worker exceeded {shape_wall_timeout_s}s "
                f"wall-clock budget; OS-killed (SIGKILL)." + stderr_tail
            ),
        )
    except Exception:
        return _subworker_failure_results(
            validation_mode=validation_mode,
            reason="Failed to launch per-shape sub-worker:\n" + traceback.format_exc(),
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
    return _subworker_failure_results(
        validation_mode=validation_mode,
        reason=reason,
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
    benchmark_mode: str = "eager",
    cuda_graph_cache_flush_mb: int = 1024,
    graph_atol: float = 1e-2,
    graph_rtol: float = 0.05,
    graph_min_cosine: float | None = None,
    graph_max_rel_l2: float | None = None,
    trust_mode: str = _TRUST_MODE_TRUSTED,
    validation_mode: str = _VALIDATION_MODE_FULL,
) -> None:
    """Run the selected validation stages for one shape and emit JSON.

    The parent reads ``shape_result_path``. If this process gets killed by
    a signal (GPU memory access fault, SIGABRT, OOM-killer, …) before the
    JSON is written, the parent synthesizes a failed result for the shape.
    """
    reference_path = reference_dir / "reference.py"
    # Input artifacts are now {seed, format: "manual_seed"} stored inside
    # eval_result.json -- no per-case .pt files are written. The reference
    # implementations seed deterministically via deterministic_input_seed()
    # so the artifact alone is enough to reproduce a case.

    if trust_mode == _TRUST_MODE_UNTRUSTED:
        _install_untrusted_runtime_guards()
        integrity_snapshot = _snapshot_untrusted_critical_functions()
        _check_untrusted_integrity(integrity_snapshot)
    else:
        integrity_snapshot = {}

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

    threads_before = threading.active_count()

    if validation_mode == _VALIDATION_MODE_PERFORMANCE_ONLY:
        correctness_result = CorrectnessShapeResult(
            status="skipped",
            reason=_CORRECTNESS_ONLY_SKIP_REASON,
        )
    else:
        correctness_result = check_correctness(
            reference_path,
            candidate_path,
            shape_id=shape_id,
            atol=atol,
            rtol=rtol,
            num_correctness_cases=num_correctness_cases,
            candidate_timeout_s=candidate_timeout_s,
            untrusted_mode=trust_mode == _TRUST_MODE_UNTRUSTED,
        )

    # Read the artifact-based compile signal NOW (before the wrapper is
    # removed) so the value reflects only the correctness-phase compilation.
    compile_succeeded = flydsl_compile_succeeded() if tracker_installed else None

    # Remove the JitFunction.__call__ wrapper BEFORE the performance
    # benchmark so triton.do_bench measures the candidate with zero
    # tracking overhead on the hot dispatch path.
    if tracker_installed:
        uninstall_jit_compile_tracker()

    reward_hack_reason: str | None = None
    if trust_mode == _TRUST_MODE_UNTRUSTED:
        try:
            _check_untrusted_integrity(integrity_snapshot)
        except RewardHackDetected as reward_hack:
            reward_hack_reason = str(reward_hack)

    should_run_performance = (
        validation_mode == _VALIDATION_MODE_PERFORMANCE_ONLY
        or (
            validation_mode == _VALIDATION_MODE_FULL
            and correctness_result.status == "passed"
        )
    )
    if should_run_performance and reward_hack_reason is None:
        performance_result = benchmark_performance(
            candidate_path,
            reference_path,
            shape_id=shape_id,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            collect_kernel_events=collect_kernel_events,
            candidate_timeout_s=candidate_timeout_s,
            perf_timeout_s=perf_timeout_s,
            benchmark_mode=benchmark_mode,
            cuda_graph_cache_flush_mb=cuda_graph_cache_flush_mb,
            atol=graph_atol,
            rtol=graph_rtol,
            graph_min_cosine=graph_min_cosine,
            graph_max_rel_l2=graph_max_rel_l2,
        )
    else:
        performance_result = PerformanceShapeResult()

    if trust_mode == _TRUST_MODE_UNTRUSTED:
        try:
            _check_untrusted_integrity(integrity_snapshot)
            check_thread_injection(threads_before, threading.active_count())
        except RewardHackDetected as reward_hack:
            reward_hack_reason = str(reward_hack)
        if reward_hack_reason is not None:
            if validation_mode == _VALIDATION_MODE_PERFORMANCE_ONLY:
                performance_result = PerformanceShapeResult(
                    error=reward_hack_reason,
                )
            else:
                correctness_result = CorrectnessShapeResult(
                    status="failed",
                    reason=reward_hack_reason,
                )
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
            benchmark_mode=performance_result.benchmark_mode,
            capture_time_ms=performance_result.capture_time_ms,
            cache_flush_mb=performance_result.cache_flush_mb,
            graph_correctness=performance_result.graph_correctness,
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
    require_clock_locked: bool = False,
) -> dict[str, object]:
    """Benchmark torch.compile(reference) across all shapes and persist JSON."""
    worker_started = time.perf_counter()
    eval_output_path = artifact_dir / "eval_result.json"
    environment = _build_environment(clock_locked=clock_locked)
    runner_config = _build_runner_config(
        config_version=config_version,
        mode=_TORCH_COMPILE_EVAL_MODE,
        validation_mode=_VALIDATION_MODE_PERFORMANCE_ONLY,
        atol=0.0,
        rtol=0.0,
        num_correctness_cases=0,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        require_clock_locked=require_clock_locked,
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
    performance_verdicts = _empty_stage_verdicts(
        shape_ids,
        status="skipped",
        reason=_PENDING_STAGE_REASON,
    )

    clock_failure = clock_lock_failure_reason(required=require_clock_locked)
    if clock_failure is not None:
        failure_payload = _build_eval_payload(
            reference_dir=reference_dir,
            candidate_path=None,
            runner_config=runner_config,
            environment=environment,
            eval_id=eval_id,
            compile_result=CompileResult(status="failed", reason=clock_failure),
            correctness_per_shape=correctness_per_shape,
            performance_per_shape=performance_per_shape,
            performance_verdicts=_empty_stage_verdicts(
                shape_ids,
                status="skipped",
                reason=clock_failure,
            ),
            eval_mode=_TORCH_COMPILE_EVAL_MODE,
            error=clock_failure,
        )
        _save_eval_json(failure_payload, eval_output_path)
        _log(f"[eval] abort mode={_TORCH_COMPILE_EVAL_MODE} reason={clock_failure}")
        return failure_payload

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
            performance_verdicts=performance_verdicts,
            eval_mode=_TORCH_COMPILE_EVAL_MODE,
        )
        _save_eval_json(partial_payload, eval_output_path)

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
        performance_verdicts[shape_id] = _performance_verdict(
            validation_mode=_VALIDATION_MODE_PERFORMANCE_ONLY,
            correctness=correctness_result,
            performance=performance_result,
        )
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
        performance_verdicts=performance_verdicts,
        eval_mode=_TORCH_COMPILE_EVAL_MODE,
    )
    _save_eval_json(final_payload, eval_output_path)
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
    require_clock_locked: bool,
    collect_kernel_events: bool,
    candidate_timeout_s: int | float,
    perf_timeout_s: int | float,
    benchmark_mode: str = "eager",
    cuda_graph_cache_flush_mb: int = 1024,
    graph_atol: float = 1e-2,
    graph_rtol: float = 0.05,
    graph_min_cosine: float | None = None,
    graph_max_rel_l2: float | None = None,
    trust_mode: str = _TRUST_MODE_TRUSTED,
    validation_mode: str = _VALIDATION_MODE_FULL,
) -> dict[str, object]:
    """Run all stages across all shapes and persist the eval_result.json."""
    worker_started = time.perf_counter()
    candidate_path = input_path
    eval_output_path = artifact_dir / "eval_result.json"
    environment = _build_environment(clock_locked=clock_locked)
    runner_config = _build_runner_config(
        config_version=config_version,
        mode=_CANDIDATE_EVAL_MODE,
        validation_mode=validation_mode,
        atol=atol,
        rtol=rtol,
        num_correctness_cases=num_correctness_cases,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        candidate_timeout_s=candidate_timeout_s,
        benchmark_mode=benchmark_mode,
        cuda_graph_cache_flush_mb=cuda_graph_cache_flush_mb,
        graph_atol=graph_atol,
        graph_rtol=graph_rtol,
        graph_min_cosine=graph_min_cosine,
        graph_max_rel_l2=graph_max_rel_l2,
        trust_mode=trust_mode,
        require_clock_locked=require_clock_locked,
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

    clock_failure = clock_lock_failure_reason(required=require_clock_locked)
    if clock_failure is not None:
        correctness_per_shape = _empty_correctness_for_shapes(
            shape_ids, status="skipped", reason=clock_failure
        )
        performance_per_shape = _empty_performance_for_shapes(shape_ids)
        payload = _build_eval_payload(
            reference_dir=reference_dir,
            candidate_path=candidate_path,
            runner_config=runner_config,
            environment=environment,
            eval_id=eval_id,
            compile_result=CompileResult(status="failed", reason=clock_failure),
            correctness_per_shape=correctness_per_shape,
            performance_per_shape=performance_per_shape,
            performance_verdicts=_empty_stage_verdicts(
                shape_ids,
                status="skipped",
                reason=clock_failure,
            ),
            emit_flydsl_compute_ratio=(
                collect_kernel_events
                and validation_mode != _VALIDATION_MODE_CORRECTNESS_ONLY
            ),
            error=clock_failure,
        )
        _save_eval_json(payload, eval_output_path)
        _log(
            f"[eval] done mode={_CANDIDATE_EVAL_MODE} op={reference_dir.name} "
            f"clock_lock=failed elapsed={_elapsed_s(worker_started):.2f}s"
        )
        return payload

    # Stage 0: compile
    compile_started = time.perf_counter()
    _log(f"[compile] start candidate={candidate_path}")
    if trust_mode == _TRUST_MODE_UNTRUSTED:
        _install_untrusted_runtime_guards()
        integrity_snapshot = _snapshot_untrusted_critical_functions()
    else:
        integrity_snapshot = {}
    compile_result = check_compilation(candidate_path)
    if trust_mode == _TRUST_MODE_UNTRUSTED and compile_result.status == "passed":
        try:
            _check_untrusted_integrity(integrity_snapshot)
        except RewardHackDetected as reward_hack:
            compile_result = CompileResult(status="failed", reason=str(reward_hack))
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
            performance_verdicts=_empty_stage_verdicts(
                shape_ids,
                status="skipped",
                reason=_SKIP_COMPILE_REASON,
            ),
            emit_flydsl_compute_ratio=(
                collect_kernel_events
                and validation_mode != _VALIDATION_MODE_CORRECTNESS_ONLY
            ),
        )
        _save_eval_json(payload, eval_output_path)
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
    performance_verdicts = _empty_stage_verdicts(
        shape_ids,
        status="skipped",
        reason=_PENDING_STAGE_REASON,
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
            performance_verdicts=performance_verdicts,
            emit_flydsl_compute_ratio=(
                collect_kernel_events
                and validation_mode != _VALIDATION_MODE_CORRECTNESS_ONLY
            ),
            compile_succeeded_per_shape=compile_succeeded_per_shape,
        )
        _save_eval_json(partial_payload, eval_output_path)

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
            benchmark_mode=benchmark_mode,
            cuda_graph_cache_flush_mb=cuda_graph_cache_flush_mb,
            graph_atol=graph_atol,
            graph_rtol=graph_rtol,
            graph_min_cosine=graph_min_cosine,
            graph_max_rel_l2=graph_max_rel_l2,
            trust_mode=trust_mode,
            validation_mode=validation_mode,
        )
        correctness_per_shape[shape_id] = correctness_result
        performance_per_shape[shape_id] = performance_result
        performance_verdicts[shape_id] = _performance_verdict(
            validation_mode=validation_mode,
            correctness=correctness_result,
            performance=performance_result,
        )
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
        performance_verdicts=performance_verdicts,
        emit_flydsl_compute_ratio=(
            collect_kernel_events
            and validation_mode != _VALIDATION_MODE_CORRECTNESS_ONLY
        ),
        compile_succeeded_per_shape=compile_succeeded_per_shape,
    )
    _save_eval_json(final_payload, eval_output_path)
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
        performance_verdicts=_empty_stage_verdicts(
            shape_ids,
            status="skipped",
            reason=_PENDING_EVAL_REASON,
        ),
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
        performance_verdicts=_empty_stage_verdicts(
            shape_ids,
            status="skipped",
            reason=_PENDING_EVAL_REASON,
        ),
        eval_mode=_TORCH_COMPILE_EVAL_MODE,
        error=_PENDING_EVAL_REASON,
    )


def _run_torch_compile_eval_process(
    reference_dir: Path,
    output_root: Path,
    *,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    checkpoint_dir: Path | None = None,
    timestamp: str | None = None,
    config_version: str = _DEFAULT_CONFIG_VERSION,
    clock_locked: bool = False,
    require_clock_locked: bool = False,
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
        validation_mode=_VALIDATION_MODE_PERFORMANCE_ONLY,
        atol=0.0,
        rtol=0.0,
        num_correctness_cases=0,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        require_clock_locked=require_clock_locked,
    )
    environment = _build_environment(clock_locked=clock_locked)

    initial_payload = _initial_torch_compile_fallback_payload(
        reference_dir=reference_dir,
        runner_config=runner_config,
        environment=environment,
        eval_id=_eval_id(),
    )
    _save_eval_json(initial_payload, eval_output_path)

    try:
        _archive_reference_bundle(reference_dir, artifact_dir)
        _resolve_reference_bundle(reference_dir)
    except Exception:
        failure_payload = _annotate_failure_payload(
            _load_saved_payload(eval_output_path) or initial_payload,
            traceback.format_exc(),
        )
        _save_eval_json(failure_payload, eval_output_path)
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
    if require_clock_locked:
        worker_command.append("--require-clock-locked")
    if checkpoint_dir is not None:
        worker_command.extend(["--checkpoint-dir", str(checkpoint_dir)])
    _write_staging_manifest(
        artifact_dir=artifact_dir,
        candidate_path=None,
        reference_dir=reference_dir,
        runner_config=runner_config,
        environment=environment,
        worker_command=worker_command,
    )

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
        _save_eval_json(failure_payload, eval_output_path)
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
        _save_eval_json(failure_payload, eval_output_path)
        return failure_payload

    if saved_payload is None:
        failure_payload = _annotate_failure_payload(
            initial_payload,
            "torch-compile worker completed but did not write a valid eval_result.json.",
        )
        _save_eval_json(failure_payload, eval_output_path)
        return failure_payload

    return saved_payload


def _run_eval_process(
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
    require_clock_locked: bool = False,
    collect_kernel_events: bool = True,
    candidate_timeout_s: int | float = 60,
    perf_timeout_s: int | float = 600,
    benchmark_mode: str = "eager",
    cuda_graph_cache_flush_mb: int = 1024,
    graph_atol: float = 1e-2,
    graph_rtol: float = 0.05,
    graph_min_cosine: float | None = None,
    graph_max_rel_l2: float | None = None,
    trust_mode: str = _TRUST_MODE_TRUSTED,
    validation_mode: str = _VALIDATION_MODE_FULL,
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
        validation_mode=validation_mode,
        atol=atol,
        rtol=rtol,
        num_correctness_cases=num_correctness_cases,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        candidate_timeout_s=candidate_timeout_s,
        benchmark_mode=benchmark_mode,
        cuda_graph_cache_flush_mb=cuda_graph_cache_flush_mb,
        graph_atol=graph_atol,
        graph_rtol=graph_rtol,
        graph_min_cosine=graph_min_cosine,
        graph_max_rel_l2=graph_max_rel_l2,
        trust_mode=trust_mode,
        require_clock_locked=require_clock_locked,
    )
    environment = _build_environment(clock_locked=clock_locked)

    initial_payload = _initial_fallback_payload(
        reference_dir=reference_dir,
        candidate_path=input_path,
        runner_config=runner_config,
        environment=environment,
        eval_id=_eval_id(),
    )
    _save_eval_json(initial_payload, eval_output_path)

    try:
        _archive_bundle(input_path, reference_dir, artifact_dir)
        _validate_candidate_path(input_path)
        _resolve_reference_bundle(reference_dir)
    except Exception:
        failure_payload = _annotate_failure_payload(
            _load_saved_payload(eval_output_path) or initial_payload,
            traceback.format_exc(),
        )
        _save_eval_json(failure_payload, eval_output_path)
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
        "--benchmark-mode",
        benchmark_mode,
        "--cuda-graph-cache-flush-mb",
        str(cuda_graph_cache_flush_mb),
        "--graph-atol",
        str(graph_atol),
        "--graph-rtol",
        str(graph_rtol),
        "--config-version",
        config_version,
        "--trust-mode",
        trust_mode,
        "--worker",
        "--artifact-dir",
        str(artifact_dir),
    ]
    if graph_min_cosine is not None:
        worker_command.extend(["--graph-min-cosine", str(graph_min_cosine)])
    if graph_max_rel_l2 is not None:
        worker_command.extend(["--graph-max-rel-l2", str(graph_max_rel_l2)])
    if clock_locked:
        worker_command.append("--clock-locked")
    if require_clock_locked:
        worker_command.append("--require-clock-locked")
    if checkpoint_dir is not None:
        worker_command.extend(["--checkpoint-dir", str(checkpoint_dir)])
    if not collect_kernel_events:
        worker_command.append("--skip-kernel-attribution")
    worker_command.extend(_validation_mode_cli_args(validation_mode))
    _write_staging_manifest(
        artifact_dir=artifact_dir,
        candidate_path=input_path,
        reference_dir=reference_dir,
        runner_config=runner_config,
        environment=environment,
        worker_command=worker_command,
    )

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
        _save_eval_json(failure_payload, eval_output_path)
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
        _save_eval_json(failure_payload, eval_output_path)
        return failure_payload

    if saved_payload is None:
        failure_payload = _annotate_failure_payload(
            initial_payload,
            "run_eval worker completed but did not write a valid eval_result.json.",
        )
        _save_eval_json(failure_payload, eval_output_path)
        return failure_payload

    return saved_payload


def _public_clock_lock_config(
    config: ClockLockConfig | None,
    *,
    clock_locked: bool,
    require_clock_locked: bool,
) -> ClockLockConfig:
    if config is None:
        mode = "external" if require_clock_locked else "off"
        return ClockLockConfig(mode=mode)
    if config.mode == "manage" and (clock_locked or require_clock_locked):
        raise ValueError(
            "manage mode cannot be combined with clock_locked or "
            "require_clock_locked assertions"
        )
    if config.mode == "external" and clock_locked:
        raise ValueError(
            "external mode cannot be combined with a clock_locked assertion"
        )
    return config


def _build_managed_clock_session(
    *,
    config: ClockLockConfig,
    artifact_dir: Path,
) -> ManagedClockLock:
    report_path = artifact_dir / "clock_lock.json"
    return ManagedClockLock(
        config=config,
        nvidia_smi=NvidiaSmi(timeout_s=config.command_timeout_seconds),
        monitor_factory=lambda device, monitor_config: NvidiaClockMonitor(
            device_uuid=device.uuid,
            target_mhz=monitor_config.graphics_mhz,
            tolerance_mhz=monitor_config.runtime_tolerance_mhz,
            sample_interval_ms=monitor_config.sample_interval_ms,
            trace_path=artifact_dir / "clock_lock_trace.csv",
        ),
        report_callback=lambda report: _save_eval_json(
            report.to_dict(), report_path
        ),
    )


def _clock_management_error(error: str) -> str:
    return f"GPU clock management failed: {error}"


def _interruption_error(error: BaseException) -> str:
    message = f"Evaluation interrupted by {type(error).__name__}"
    details = str(error).strip()
    return f"{message}: {details}" if details else message


def _append_payload_error(payload: dict[str, object], error: str) -> None:
    previous = payload.get("error")
    if isinstance(previous, str) and previous.strip():
        if error not in previous:
            payload["error"] = f"{previous}\n\n{error}"
        return
    payload["error"] = error


def _attach_clock_lock_report(
    payload: dict[str, object],
    report: ClockLockReport,
) -> dict[str, object]:
    """Merge managed hardware evidence without mutating worker results."""
    updated = copy.deepcopy(payload)
    environment = updated.get("environment")
    if not isinstance(environment, dict):
        environment = {}
        updated["environment"] = environment
    environment["clock_locked"] = report.verified
    environment["clock_lock_verified"] = report.verified
    environment["clock_lock_source"] = report.source
    environment["clock_lock"] = report.to_dict()
    if report.error is not None and (not report.verified or not report.restored):
        _append_payload_error(updated, _clock_management_error(report.error))
    return updated


def _evaluate_with_clock_policy(
    *,
    config: ClockLockConfig,
    clock_locked: bool,
    require_clock_locked: bool,
    artifact_dir: Path,
    eval_output_path: Path,
    build_failure_payload: Callable[[str], dict[str, object]],
    evaluate: Callable[[bool, bool], dict[str, object]],
) -> dict[str, object]:
    if config.mode == "off":
        return evaluate(clock_locked, require_clock_locked)
    if config.mode == "external":
        return evaluate(False, True)

    manager = _build_managed_clock_session(
        config=config,
        artifact_dir=artifact_dir,
    )
    payload: dict[str, object] | None = None
    try:
        with manager:
            payload = evaluate(True, True)
    except ClockLockError as error:
        if payload is None:
            payload = build_failure_payload(_clock_management_error(str(error)))
        else:
            payload = copy.deepcopy(payload)
            _append_payload_error(payload, _clock_management_error(str(error)))
    except BaseException as error:
        interruption_error = _interruption_error(error)
        payload = _load_saved_payload(eval_output_path)
        if payload is None:
            payload = build_failure_payload(interruption_error)
        else:
            payload = copy.deepcopy(payload)
            _append_payload_error(payload, interruption_error)
        try:
            _save_eval_json(
                manager.report.to_dict(), artifact_dir / "clock_lock.json"
            )
            payload = _attach_clock_lock_report(payload, manager.report)
            _save_eval_json(payload, eval_output_path)
        except BaseException as persistence_error:
            try:
                _log(
                    "[eval] failed to persist interrupted clock-lock outcome: "
                    f"{persistence_error}"
                )
            except Exception:
                pass
        raise

    assert payload is not None
    _save_eval_json(manager.report.to_dict(), artifact_dir / "clock_lock.json")
    payload = _attach_clock_lock_report(payload, manager.report)
    _save_eval_json(payload, eval_output_path)
    return payload


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
    require_clock_locked: bool = False,
    clock_lock_config: ClockLockConfig | None = None,
) -> dict[str, object]:
    """Run torch.compile evaluation under one optional parent clock lock."""
    resolved_timestamp = get_timestamp(timestamp)
    resolved_clock_config = _public_clock_lock_config(
        clock_lock_config,
        clock_locked=clock_locked,
        require_clock_locked=require_clock_locked,
    )
    kernel_name = _kernel_name(reference_dir, reference_dir / "reference.py")
    artifact_dir, eval_output_path = _build_artifact_paths(
        output_root=output_root,
        kernel_name=kernel_name,
        timestamp=resolved_timestamp,
    )

    def build_failure_payload(error: str) -> dict[str, object]:
        runner_config = _build_runner_config(
            config_version=config_version,
            mode=_TORCH_COMPILE_EVAL_MODE,
            validation_mode=_VALIDATION_MODE_PERFORMANCE_ONLY,
            atol=0.0,
            rtol=0.0,
            num_correctness_cases=0,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            require_clock_locked=True,
        )
        initial = _initial_torch_compile_fallback_payload(
            reference_dir=reference_dir,
            runner_config=runner_config,
            environment=_build_environment(clock_locked=False),
            eval_id=_eval_id(),
        )
        return _annotate_failure_payload(initial, error)

    def evaluate(
        effective_clock_locked: bool,
        effective_require_clock_locked: bool,
    ) -> dict[str, object]:
        return _run_torch_compile_eval_process(
            reference_dir=reference_dir,
            output_root=output_root,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            checkpoint_dir=checkpoint_dir,
            timestamp=resolved_timestamp,
            config_version=config_version,
            clock_locked=effective_clock_locked,
            require_clock_locked=effective_require_clock_locked,
        )

    return _evaluate_with_clock_policy(
        config=resolved_clock_config,
        clock_locked=clock_locked,
        require_clock_locked=require_clock_locked,
        artifact_dir=artifact_dir,
        eval_output_path=eval_output_path,
        build_failure_payload=build_failure_payload,
        evaluate=evaluate,
    )


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
    require_clock_locked: bool = False,
    collect_kernel_events: bool = True,
    candidate_timeout_s: int | float = 60,
    perf_timeout_s: int | float = 600,
    benchmark_mode: str = "eager",
    cuda_graph_cache_flush_mb: int = 1024,
    graph_atol: float = 1e-2,
    graph_rtol: float = 0.05,
    graph_min_cosine: float | None = None,
    graph_max_rel_l2: float | None = None,
    trust_mode: str = _TRUST_MODE_TRUSTED,
    clock_lock_config: ClockLockConfig | None = None,
    validation_mode: str = _VALIDATION_MODE_FULL,
) -> dict[str, object]:
    """Run candidate evaluation under one optional parent clock lock."""
    validation_mode = _resolve_validation_mode(validation_mode, {})
    resolved_timestamp = get_timestamp(timestamp)
    resolved_clock_config = _public_clock_lock_config(
        clock_lock_config,
        clock_locked=clock_locked,
        require_clock_locked=require_clock_locked,
    )
    kernel_name = _kernel_name(reference_dir, input_path)
    artifact_dir, eval_output_path = _build_artifact_paths(
        output_root=output_root,
        kernel_name=kernel_name,
        timestamp=resolved_timestamp,
    )

    def build_failure_payload(error: str) -> dict[str, object]:
        runner_config = _build_runner_config(
            config_version=config_version,
            mode=_CANDIDATE_EVAL_MODE,
            validation_mode=validation_mode,
            atol=atol,
            rtol=rtol,
            num_correctness_cases=num_correctness_cases,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            candidate_timeout_s=candidate_timeout_s,
            benchmark_mode=benchmark_mode,
            cuda_graph_cache_flush_mb=cuda_graph_cache_flush_mb,
            graph_atol=graph_atol,
            graph_rtol=graph_rtol,
            graph_min_cosine=graph_min_cosine,
            graph_max_rel_l2=graph_max_rel_l2,
            trust_mode=trust_mode,
            require_clock_locked=True,
        )
        initial = _initial_fallback_payload(
            reference_dir=reference_dir,
            candidate_path=input_path,
            runner_config=runner_config,
            environment=_build_environment(clock_locked=False),
            eval_id=_eval_id(),
        )
        return _annotate_failure_payload(initial, error)

    def evaluate(
        effective_clock_locked: bool,
        effective_require_clock_locked: bool,
    ) -> dict[str, object]:
        return _run_eval_process(
            input_path=input_path,
            reference_dir=reference_dir,
            output_root=output_root,
            atol=atol,
            rtol=rtol,
            num_correctness_cases=num_correctness_cases,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            checkpoint_dir=checkpoint_dir,
            timestamp=resolved_timestamp,
            config_version=config_version,
            clock_locked=effective_clock_locked,
            require_clock_locked=effective_require_clock_locked,
            collect_kernel_events=collect_kernel_events,
            candidate_timeout_s=candidate_timeout_s,
            perf_timeout_s=perf_timeout_s,
            benchmark_mode=benchmark_mode,
            cuda_graph_cache_flush_mb=cuda_graph_cache_flush_mb,
            graph_atol=graph_atol,
            graph_rtol=graph_rtol,
            graph_min_cosine=graph_min_cosine,
            graph_max_rel_l2=graph_max_rel_l2,
            trust_mode=trust_mode,
            validation_mode=validation_mode,
        )

    return _evaluate_with_clock_policy(
        config=resolved_clock_config,
        clock_locked=clock_locked,
        require_clock_locked=require_clock_locked,
        artifact_dir=artifact_dir,
        eval_output_path=eval_output_path,
        build_failure_payload=build_failure_payload,
        evaluate=evaluate,
    )


# ---------------------------------------------------------------------------
# CLI exit-code derivation
# ---------------------------------------------------------------------------


def _stage_block_all_passed(
    passed_block: dict[str, object],
    stage: str,
) -> bool:
    """Return whether a per-shape stage block is present and fully passed."""
    stage_block = passed_block.get(stage)
    if not isinstance(stage_block, dict) or not stage_block:
        return False
    return all(
        isinstance(shape_status, dict)
        and shape_status.get("status") == "passed"
        for shape_status in stage_block.values()
    )


def _payload_overall_passed(payload: dict[str, object]) -> bool:
    """Derive overall pass from the stages enabled by the validation mode."""
    payload_error = payload.get("error")
    if payload_error is not None and payload_error != "":
        return False
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

    runner_config = payload.get("runner_config")
    validation_mode = (
        runner_config.get("validation_mode")
        if isinstance(runner_config, dict)
        else None
    )
    if validation_mode is None:
        return _stage_block_all_passed(passed_block, "correctness")
    if validation_mode == _VALIDATION_MODE_CORRECTNESS_ONLY:
        return _stage_block_all_passed(passed_block, "correctness")
    if validation_mode == _VALIDATION_MODE_PERFORMANCE_ONLY:
        return _stage_block_all_passed(passed_block, "performance")
    if validation_mode == _VALIDATION_MODE_FULL:
        return _stage_block_all_passed(
            passed_block,
            "correctness",
        ) and _stage_block_all_passed(passed_block, "performance")
    return False


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
        default=None,
        help=(
            "Directory containing reference.py, input.py, shapes.json, "
            "metadata.json (and roofline.json). Required via CLI or config."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Root output directory for timestamped evaluation artifacts. "
            "Required via CLI or config for top-level evaluation; ignored by "
            "--worker / --single-shape-worker."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional JSON runner config. It may contain launch paths, eval_mode, "
            "validation_mode, and public runner options. Explicit CLI flags override it."
        ),
    )
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--correctness-only",
        action="store_const",
        const=_VALIDATION_MODE_CORRECTNESS_ONLY,
        dest="validation_mode",
        default=None,
        help="Run candidate compile and correctness stages only.",
    )
    validation_group.add_argument(
        "--performance-only",
        action="store_const",
        const=_VALIDATION_MODE_PERFORMANCE_ONLY,
        dest="validation_mode",
        help="Run candidate compile and performance stages only.",
    )
    parser.add_argument("--atol", type=float, default=None, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="Relative tolerance")
    parser.add_argument(
        "--correctness-max-rel-l2",
        type=float,
        default=None,
        help=(
            "Optional global relative-L2 correctness threshold for floating "
            "outputs. Replaces element-wise allclose when set."
        ),
    )
    parser.add_argument(
        "--num-correctness-cases",
        type=int,
        default=None,
        help="Number of correctness cases per shape",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=None,
        help="Warmup iterations for profiling",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=None,
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
        default=None,
        help=(
            "Versioned runner config identifier (placed under runner_config.config_version "
            "in eval_result.json; conventionally points at configs/runner/<version>.yaml)"
        ),
    )
    parser.add_argument(
        "--clock-locked",
        action="store_true",
        help="Assert that GPU clocks are pinned during measurement",
    )
    parser.add_argument(
        "--require-clock-locked",
        action="store_true",
        help=(
            "Require an external setup step to mark GPU clocks as locked. "
            "The marker is ATREX_BENCH_CLOCKS_LOCKED=1 or SOL_EXECBENCH_CLOCKS_LOCKED=1."
        ),
    )
    parser.add_argument(
        "--clock-lock-mode",
        choices=("off", "external", "manage"),
        default=None,
        help="GPU clock policy: off, require an external marker, or manage clocks.",
    )
    parser.add_argument(
        "--lock-clocks",
        action="store_true",
        help="Convenience alias for --clock-lock-mode manage.",
    )
    parser.add_argument(
        "--clock-lock-device",
        default=None,
        help="Physical NVIDIA GPU index or GPU UUID managed by the parent process.",
    )
    parser.add_argument(
        "--gpu-clock-mhz",
        type=int,
        default=None,
        help="Exact graphics clock in MHz for managed mode.",
    )
    parser.add_argument(
        "--memory-clock-mhz",
        type=int,
        default=None,
        help=(
            "Optional exact memory clock in MHz for managed mode; omit when "
            "the GPU does not support runtime memory-clock locking."
        ),
    )
    parser.add_argument(
        "--clock-lock-tolerance-mhz",
        type=int,
        default=None,
        help="Maximum observed clock deviation in MHz; managed default is 50.",
    )
    parser.add_argument(
        "--clock-lock-settle-seconds",
        type=float,
        default=None,
        help="Delay before managed-clock verification; default is 3 seconds.",
    )
    parser.add_argument(
        "--clock-lock-command-timeout-s",
        type=float,
        default=None,
        help="Timeout for each nvidia-smi command; default is 10 seconds.",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_false",
        dest="clock_lock_require_idle",
        default=None,
        help="Allow managed locking when another compute process uses the target GPU.",
    )
    parser.add_argument(
        "--clock-lock-monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Continuously verify managed GPU clocks for the complete worker window.",
    )
    parser.add_argument(
        "--clock-lock-sample-interval-ms",
        type=int,
        default=None,
        help="Managed clock telemetry interval in milliseconds; default is 10.",
    )
    parser.add_argument(
        "--clock-lock-runtime-tolerance-mhz",
        type=int,
        default=None,
        help="Allowed measurement-window clock deviation; default is 0 MHz.",
    )
    parser.add_argument(
        "--clock-lock-fail-on-deviation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fail the evaluation when an otherwise healthy managed-clock "
            "measurement window deviates from its requested frequency."
        ),
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
        default=None,
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
        default=None,
        help=(
            "Wall-clock timeout (seconds) for the ENTIRE perf phase per shape "
            "(do_bench end-to-end measurement + the optional profiler-breakdown "
            "loop for kernel attribution). Pass <= 0 to disable. Default 600s "
            "(10 min)."
        ),
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("eager", "cuda_graph_replay"),
        default=None,
        help="Performance execution mode. CUDA Graph replay is opt-in.",
    )
    parser.add_argument(
        "--cuda-graph-cache-flush-mb",
        type=int,
        default=None,
        help=(
            "CUDA cache-flush buffer size in MiB before each replay sample. "
            "Only used by --benchmark-mode cuda_graph_replay."
        ),
    )
    parser.add_argument(
        "--graph-atol",
        type=float,
        default=None,
        help="Absolute tolerance for CUDA Graph replay versus eager output.",
    )
    parser.add_argument(
        "--graph-rtol",
        type=float,
        default=None,
        help="Relative tolerance for CUDA Graph replay versus eager output.",
    )
    parser.add_argument(
        "--graph-min-cosine",
        type=float,
        default=None,
        help=(
            "Optional minimum cosine similarity for CUDA Graph replay versus "
            "eager output. When set, it replaces allclose for floating outputs."
        ),
    )
    parser.add_argument(
        "--graph-max-rel-l2",
        type=float,
        default=None,
        help=(
            "Optional maximum relative L2 error for CUDA Graph replay versus "
            "eager output."
        ),
    )
    parser.add_argument(
        "--trust-mode",
        choices=(_TRUST_MODE_TRUSTED, _TRUST_MODE_UNTRUSTED),
        default=None,
        help=(
            "Evaluation guard profile. trusted allows production/runtime JIT paths; "
            "untrusted disables runtime cpp_extension JIT and applies process-local "
            "anti-tampering guards. It is not a sandbox; use an isolated worker for "
            "untrusted code."
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
    cli_validation_mode = args.validation_mode

    try:
        runner_file_config = _load_runner_config_file(args.config)
    except Exception as error:
        raise SystemExit(str(error)) from error
    try:
        args.input = _resolve_path_option(
            "input",
            cli_value=args.input,
            config=runner_file_config,
        )
        args.reference_dir = _resolve_path_option(
            "reference_dir",
            cli_value=args.reference_dir,
            config=runner_file_config,
        )
        args.output = _resolve_path_option(
            "output",
            cli_value=args.output,
            config=runner_file_config,
        )
        args.checkpoint_dir = _resolve_path_option(
            "checkpoint_dir",
            cli_value=args.checkpoint_dir,
            config=runner_file_config,
        )
        eval_mode = _resolve_eval_mode(args.torch_compile, runner_file_config)
        args.torch_compile = eval_mode == _TORCH_COMPILE_EVAL_MODE
        args.validation_mode = _resolve_validation_mode(
            cli_validation_mode,
            runner_file_config,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.torch_compile and cli_validation_mode is not None:
        raise SystemExit(
            "--correctness-only/--performance-only cannot be combined with "
            "--torch-compile."
        )
    if (
        args.torch_compile
        and "validation_mode" in runner_file_config
        and args.validation_mode != _VALIDATION_MODE_PERFORMANCE_ONLY
    ):
        raise SystemExit(
            "torch_compile_reference requires validation_mode=performance_only "
            "when validation_mode is set in config."
        )
    if args.reference_dir is None:
        raise SystemExit("reference_dir is required via --reference-dir or config.")

    args.atol = float(
        _resolve_runner_option(
            "atol", cli_value=args.atol, config=runner_file_config, default=_DEFAULT_ATOL
        )
    )
    args.rtol = float(
        _resolve_runner_option(
            "rtol", cli_value=args.rtol, config=runner_file_config, default=_DEFAULT_RTOL
        )
    )
    args.correctness_max_rel_l2 = _resolve_runner_option(
        "correctness_max_rel_l2",
        cli_value=args.correctness_max_rel_l2,
        config=runner_file_config,
        default=None,
    )
    args.num_correctness_cases = int(
        _resolve_runner_option(
            "num_correctness_cases",
            cli_value=args.num_correctness_cases,
            config=runner_file_config,
            default=_DEFAULT_NUM_CORRECTNESS_CASES,
        )
    )
    args.warmup_iters = int(
        _resolve_runner_option(
            "warmup_iters",
            cli_value=args.warmup_iters,
            config=runner_file_config,
            default=_DEFAULT_WARMUP_ITERS,
        )
    )
    args.bench_iters = int(
        _resolve_runner_option(
            "bench_iters",
            cli_value=args.bench_iters,
            config=runner_file_config,
            default=_DEFAULT_BENCH_ITERS,
        )
    )
    args.config_version = str(
        _resolve_runner_option(
            "config_version",
            cli_value=args.config_version,
            config=runner_file_config,
            default=_DEFAULT_CONFIG_VERSION,
        )
    )
    args.candidate_timeout_s = float(
        _resolve_runner_option(
            "candidate_timeout_s",
            cli_value=args.candidate_timeout_s,
            config=runner_file_config,
            default=_DEFAULT_CANDIDATE_TIMEOUT_S,
        )
    )
    args.perf_timeout_s = float(
        _resolve_runner_option(
            "perf_timeout_s",
            cli_value=args.perf_timeout_s,
            config=runner_file_config,
            default=_DEFAULT_PERF_TIMEOUT_S,
        )
    )
    args.benchmark_mode = str(
        _resolve_runner_option(
            "benchmark_mode",
            cli_value=args.benchmark_mode,
            config=runner_file_config,
            default=_DEFAULT_BENCHMARK_MODE,
        )
    )
    if args.benchmark_mode not in {"eager", "cuda_graph_replay"}:
        raise SystemExit(
            "benchmark_mode must be one of: eager, cuda_graph_replay"
        )
    args.cuda_graph_cache_flush_mb = int(
        _resolve_runner_option(
            "cuda_graph_cache_flush_mb",
            cli_value=args.cuda_graph_cache_flush_mb,
            config=runner_file_config,
            default=_DEFAULT_CUDA_GRAPH_CACHE_FLUSH_MB,
        )
    )
    args.graph_atol = float(
        _resolve_runner_option(
            "graph_atol",
            cli_value=args.graph_atol,
            config=runner_file_config,
            default=_DEFAULT_GRAPH_ATOL,
        )
    )
    args.graph_rtol = float(
        _resolve_runner_option(
            "graph_rtol",
            cli_value=args.graph_rtol,
            config=runner_file_config,
            default=_DEFAULT_GRAPH_RTOL,
        )
    )
    args.graph_min_cosine = _resolve_runner_option(
        "graph_min_cosine",
        cli_value=args.graph_min_cosine,
        config=runner_file_config,
        default=None,
    )
    args.graph_max_rel_l2 = _resolve_runner_option(
        "graph_max_rel_l2",
        cli_value=args.graph_max_rel_l2,
        config=runner_file_config,
        default=None,
    )
    if args.graph_min_cosine is not None:
        args.graph_min_cosine = float(args.graph_min_cosine)
    if args.graph_max_rel_l2 is not None:
        args.graph_max_rel_l2 = float(args.graph_max_rel_l2)
    try:
        args.clock_locked = bool(
            args.clock_locked
            or _runner_config_boolean("clock_locked", runner_file_config)
        )
        args.require_clock_locked = bool(
            args.require_clock_locked
            or _runner_config_boolean(
                "require_clock_locked", runner_file_config
            )
        )
        clock_lock_config = _resolve_clock_lock_config(args, runner_file_config)
        args.skip_kernel_attribution = bool(
            args.skip_kernel_attribution
            or _runner_config_boolean(
                "skip_kernel_attribution", runner_file_config
            )
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    args.trust_mode = str(
        _resolve_runner_option(
            "trust_mode",
            cli_value=args.trust_mode,
            config=runner_file_config,
            default=_TRUST_MODE_TRUSTED,
        )
    )
    if args.trust_mode not in {_TRUST_MODE_TRUSTED, _TRUST_MODE_UNTRUSTED}:
        raise SystemExit("trust_mode must be one of: trusted, untrusted")
    if args.correctness_max_rel_l2 is not None:
        os.environ[CORRECTNESS_MAX_REL_L2_ENV] = str(
            args.correctness_max_rel_l2
        )

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
            benchmark_mode=args.benchmark_mode,
            cuda_graph_cache_flush_mb=args.cuda_graph_cache_flush_mb,
            graph_atol=args.graph_atol,
            graph_rtol=args.graph_rtol,
            graph_min_cosine=args.graph_min_cosine,
            graph_max_rel_l2=args.graph_max_rel_l2,
            trust_mode=args.trust_mode,
            validation_mode=args.validation_mode,
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
            require_clock_locked=args.require_clock_locked,
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
            require_clock_locked=args.require_clock_locked,
            collect_kernel_events=not args.skip_kernel_attribution,
            candidate_timeout_s=args.candidate_timeout_s,
            perf_timeout_s=args.perf_timeout_s,
            benchmark_mode=args.benchmark_mode,
            cuda_graph_cache_flush_mb=args.cuda_graph_cache_flush_mb,
            graph_atol=args.graph_atol,
            graph_rtol=args.graph_rtol,
            graph_min_cosine=args.graph_min_cosine,
            graph_max_rel_l2=args.graph_max_rel_l2,
            trust_mode=args.trust_mode,
            validation_mode=args.validation_mode,
        )
        raise SystemExit(0)

    if args.output is None:
        raise SystemExit(
            "output is required via --output or config for top-level run_eval."
        )

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
            require_clock_locked=args.require_clock_locked,
            clock_lock_config=clock_lock_config,
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
        raise SystemExit(
            "input is required via --input or config for candidate eval mode."
        )

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
        require_clock_locked=args.require_clock_locked,
        clock_lock_config=clock_lock_config,
        collect_kernel_events=not args.skip_kernel_attribution,
        candidate_timeout_s=args.candidate_timeout_s,
        perf_timeout_s=args.perf_timeout_s,
        benchmark_mode=args.benchmark_mode,
        cuda_graph_cache_flush_mb=args.cuda_graph_cache_flush_mb,
        graph_atol=args.graph_atol,
        graph_rtol=args.graph_rtol,
        graph_min_cosine=args.graph_min_cosine,
        graph_max_rel_l2=args.graph_max_rel_l2,
        trust_mode=args.trust_mode,
        validation_mode=args.validation_mode,
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
