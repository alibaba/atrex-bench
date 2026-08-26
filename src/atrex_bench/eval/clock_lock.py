"""Clock-lock state helpers for stable benchmark evaluation."""

from __future__ import annotations

import fcntl
import math
import os
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from .clock_monitor import ClockMeasurement
from .nvidia_clock import ClockSnapshot, ComputeProcess, NvidiaDevice

ATREX_CLOCKS_LOCKED_ENV = "ATREX_BENCH_CLOCKS_LOCKED"
SOL_COMPAT_CLOCKS_LOCKED_ENV = "SOL_EXECBENCH_CLOCKS_LOCKED"
ATREX_CLOCK_DEVICE_ENV = "ATREX_BENCH_CLOCK_DEVICE"
ATREX_CLOCK_LOCK_SOURCE_ENV = "ATREX_BENCH_CLOCK_LOCK_SOURCE"
_CLOCK_LOCK_ENV_VARS = (
    ATREX_CLOCKS_LOCKED_ENV,
    SOL_COMPAT_CLOCKS_LOCKED_ENV,
)

ClockLockMode = Literal["off", "external", "manage"]


class ClockLockError(RuntimeError):
    """Raised when a requested clock-lock policy cannot be satisfied."""


@dataclass(frozen=True)
class ClockLockConfig:
    """Validated policy for one top-level evaluation clock-lock lifecycle."""

    mode: ClockLockMode = "off"
    device_selector: str | None = None
    graphics_mhz: int | None = None
    memory_mhz: int | None = None
    tolerance_mhz: int = 50
    settle_seconds: float = 3.0
    command_timeout_seconds: float = 10.0
    require_idle: bool = True
    monitor_enabled: bool | None = None
    sample_interval_ms: int = 10
    runtime_tolerance_mhz: int = 0
    fail_on_deviation: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"off", "external", "manage"}:
            raise ValueError(
                "clock lock mode must be one of: off, external, manage"
            )
        if self.device_selector is not None and (
            not isinstance(self.device_selector, str)
            or not self.device_selector.strip()
        ):
            raise ValueError("device_selector must be a non-empty string")
        if not _is_positive_int(self.tolerance_mhz):
            raise ValueError("tolerance_mhz must be a positive integer")
        if not _is_finite_number(self.settle_seconds) or self.settle_seconds < 0:
            raise ValueError("settle_seconds must be non-negative")
        if (
            not _is_finite_number(self.command_timeout_seconds)
            or self.command_timeout_seconds <= 0
        ):
            raise ValueError("command_timeout_seconds must be positive")
        if not isinstance(self.require_idle, bool):
            raise ValueError("require_idle must be a boolean")
        if self.monitor_enabled is None:
            object.__setattr__(
                self,
                "monitor_enabled",
                self.mode == "manage",
            )
        elif not isinstance(self.monitor_enabled, bool):
            raise ValueError("monitor_enabled must be a boolean")
        if not _is_positive_int(self.sample_interval_ms):
            raise ValueError("sample_interval_ms must be a positive integer")
        if not _is_nonnegative_int(self.runtime_tolerance_mhz):
            raise ValueError(
                "runtime_tolerance_mhz must be a non-negative integer"
            )
        if not isinstance(self.fail_on_deviation, bool):
            raise ValueError("fail_on_deviation must be a boolean")

        if self.mode == "manage":
            if not _is_positive_int(self.graphics_mhz):
                raise ValueError(
                    "manage mode requires a positive integer graphics_mhz value"
                )
            if self.memory_mhz is not None and not _is_positive_int(
                self.memory_mhz
            ):
                raise ValueError(
                    "memory_mhz must be a positive integer when provided"
                )
            return

        has_management_values = any(
            (
                self.device_selector is not None,
                self.graphics_mhz is not None,
                self.memory_mhz is not None,
                self.tolerance_mhz != 50,
                self.settle_seconds != 3.0,
                self.command_timeout_seconds != 10.0,
                self.require_idle is not True,
                self.monitor_enabled is not False,
                self.sample_interval_ms != 10,
                self.runtime_tolerance_mhz != 0,
                self.fail_on_deviation is not True,
            )
        )
        if has_management_values:
            raise ValueError(
                "device, frequency, tolerance, timeout, and idle settings are "
                "only valid in manage mode"
            )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class _VisibleIndexProvider(Protocol):
    def list_visible_indices(self) -> tuple[str, ...]: ...


class _NvidiaClockController(_VisibleIndexProvider, Protocol):
    def resolve_device(self, selector: str) -> NvidiaDevice: ...

    def list_compute_processes(
        self, device: NvidiaDevice
    ) -> tuple[ComputeProcess, ...]: ...

    def query_clocks(self, device: NvidiaDevice) -> ClockSnapshot: ...

    def lock_graphics(self, device: NvidiaDevice, mhz: int) -> None: ...

    def lock_memory(self, device: NvidiaDevice, mhz: int) -> None: ...

    def reset_graphics(self, device: NvidiaDevice) -> None: ...

    def reset_memory(self, device: NvidiaDevice) -> None: ...


class _ClockMonitor(Protocol):
    def start(self) -> None: ...

    def stop(self) -> ClockMeasurement: ...


def _visible_tokens(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def resolve_clock_lock_selector(
    explicit_selector: str | None,
    *,
    environ: Mapping[str, str],
    nvidia_smi: _VisibleIndexProvider,
) -> str:
    """Resolve exactly one GPU selector without initializing CUDA."""
    if explicit_selector:
        return explicit_selector.strip()

    atrex_selector = environ.get(ATREX_CLOCK_DEVICE_ENV, "").strip()
    if atrex_selector:
        return atrex_selector

    cuda_visible = environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        tokens = _visible_tokens(cuda_visible)
        if len(tokens) == 1:
            return tokens[0]
        raise ClockLockError(
            "Managed clock locking requires exactly one GPU, but "
            f"CUDA_VISIBLE_DEVICES contains {len(tokens)} entries."
        )

    nvidia_visible = environ.get("NVIDIA_VISIBLE_DEVICES", "").strip()
    if nvidia_visible and nvidia_visible.lower() not in {"all", "none", "void"}:
        tokens = _visible_tokens(nvidia_visible)
        if len(tokens) == 1:
            return tokens[0]
        raise ClockLockError(
            "Managed clock locking requires exactly one GPU, but "
            f"NVIDIA_VISIBLE_DEVICES contains {len(tokens)} entries."
        )

    indices = nvidia_smi.list_visible_indices()
    if len(indices) != 1:
        raise ClockLockError(
            "Managed clock locking requires exactly one GPU when no explicit "
            f"selector is provided; nvidia-smi reported {len(indices)} devices."
        )
    return indices[0]


@dataclass(frozen=True)
class ClockLockReport:
    """Auditable state for one managed clock-lock lifecycle."""

    mode: str
    source: str
    device: NvidiaDevice | None
    requested_graphics_mhz: int | None
    requested_memory_mhz: int | None
    tolerance_mhz: int
    fail_on_deviation: bool = True
    applied: bool = False
    verified: bool = False
    setup_verified: bool = False
    measurement_verified: bool | None = None
    verification_snapshot: ClockSnapshot | None = None
    measurement: ClockMeasurement | None = None
    restored: bool = False
    post_restore_snapshot: ClockSnapshot | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the lifecycle without losing requested/observed provenance."""
        return {
            "mode": self.mode,
            "source": self.source,
            "device": asdict(self.device) if self.device is not None else None,
            "requested": {
                "graphics_mhz": self.requested_graphics_mhz,
                "memory_mhz": self.requested_memory_mhz,
                "tolerance_mhz": self.tolerance_mhz,
            },
            "fail_on_deviation": self.fail_on_deviation,
            "applied": self.applied,
            "verified": self.verified,
            "setup_verified": self.setup_verified,
            "measurement_verified": self.measurement_verified,
            "verification_snapshot": (
                asdict(self.verification_snapshot)
                if self.verification_snapshot is not None
                else None
            ),
            "measurement": (
                self.measurement.to_dict()
                if self.measurement is not None
                else None
            ),
            "restored": self.restored,
            "post_restore_snapshot": (
                asdict(self.post_restore_snapshot)
                if self.post_restore_snapshot is not None
                else None
            ),
            "error": self.error,
        }


_MISSING_ENV_VALUE = object()


class ManagedClockLock:
    """Own a verified NVIDIA clock lock for one evaluation window."""

    def __init__(
        self,
        *,
        config: ClockLockConfig,
        nvidia_smi: _NvidiaClockController,
        monitor_factory: Callable[
            [NvidiaDevice, ClockLockConfig], _ClockMonitor
        ]
        | None = None,
        environ: MutableMapping[str, str] = os.environ,
        sleeper: Callable[[float], None] = time.sleep,
        report_callback: Callable[[ClockLockReport], None] = lambda _report: None,
        lock_directory: Path = Path(tempfile.gettempdir()),
        getpid: Callable[[], int] = os.getpid,
    ) -> None:
        if config.mode != "manage":
            raise ValueError("ManagedClockLock requires clock_lock_mode=manage")
        self.config = config
        self._nvidia_smi = nvidia_smi
        self._monitor_factory = monitor_factory
        self._environ = environ
        self._sleeper = sleeper
        self._report_callback = report_callback
        self._lock_directory = lock_directory
        self._getpid = getpid
        self._lock_handle = None
        self._device: NvidiaDevice | None = None
        self._graphics_applied = False
        self._memory_applied = False
        self._clock_monitor: _ClockMonitor | None = None
        self._monitor_started = False
        self._environment_saved = False
        self._sigterm_installed = False
        self._previous_sigterm_handler: object = signal.SIG_DFL
        self._previous_marker: str | object = _MISSING_ENV_VALUE
        self._previous_source: str | object = _MISSING_ENV_VALUE
        self.report = ClockLockReport(
            mode="manage",
            source="atrex-managed",
            device=None,
            requested_graphics_mhz=config.graphics_mhz,
            requested_memory_mhz=config.memory_mhz,
            tolerance_mhz=config.tolerance_mhz,
            fail_on_deviation=config.fail_on_deviation,
        )

    def _publish(self, **changes: object) -> None:
        self.report = replace(self.report, **changes)
        self._report_callback(self.report)

    def _publish_error(self, error_message: str) -> str:
        try:
            self._publish(error=error_message)
        except Exception as error:
            error_message = (
                f"{error_message}; publish_error_report: {error}"
            )
            self.report = replace(self.report, error=error_message)
        return error_message

    def _acquire_process_lock(self, device: NvidiaDevice) -> None:
        import hashlib

        lock_name = hashlib.sha256(device.uuid.encode("utf-8")).hexdigest()[:24]
        self._lock_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_directory / f"atrex-bench-clock-{lock_name}.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise ClockLockError(
                f"Another Atrex process is managing GPU {device.uuid}."
            ) from error
        self._lock_handle = handle

    def _release_process_lock(self) -> None:
        if self._lock_handle is None:
            return
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._lock_handle = None

    def _verify_snapshot(self, snapshot: ClockSnapshot) -> None:
        graphics_mhz = self.config.graphics_mhz
        memory_mhz = self.config.memory_mhz
        assert graphics_mhz is not None
        if abs(snapshot.graphics_mhz - graphics_mhz) > self.config.tolerance_mhz:
            raise ClockLockError(
                f"graphics clock mismatch: expected {graphics_mhz} MHz, "
                f"observed {snapshot.graphics_mhz} MHz"
            )
        if memory_mhz is not None and (
            abs(snapshot.memory_mhz - memory_mhz) > self.config.tolerance_mhz
        ):
            raise ClockLockError(
                f"memory clock mismatch: expected {memory_mhz} MHz, "
                f"observed {snapshot.memory_mhz} MHz"
            )

    def __enter__(self) -> ManagedClockLock:
        try:
            self._install_sigterm_handler()
            self._save_environment()
            selector = resolve_clock_lock_selector(
                self.config.device_selector,
                environ=self._environ,
                nvidia_smi=self._nvidia_smi,
            )
            device = self._nvidia_smi.resolve_device(selector)
            self._device = device
            self._publish(device=device)
            self._acquire_process_lock(device)

            if self.config.require_idle:
                busy_processes = tuple(
                    process
                    for process in self._nvidia_smi.list_compute_processes(device)
                    if process.pid != self._getpid()
                )
                if busy_processes:
                    details = ", ".join(
                        f"pid={process.pid} name={process.process_name}"
                        for process in busy_processes
                    )
                    raise ClockLockError(
                        f"GPU {device.uuid} is not idle; compute processes: "
                        f"{details}."
                    )

            assert self.config.graphics_mhz is not None
            self._nvidia_smi.lock_graphics(device, self.config.graphics_mhz)
            self._graphics_applied = True
            if self.config.memory_mhz is not None:
                self._nvidia_smi.lock_memory(device, self.config.memory_mhz)
                self._memory_applied = True
            self._publish(applied=True)
            self._sleeper(self.config.settle_seconds)

            snapshot = self._nvidia_smi.query_clocks(device)
            self._publish(verification_snapshot=snapshot)
            self._verify_snapshot(snapshot)
            self._publish(
                setup_verified=True,
                verified=not self.config.monitor_enabled,
            )

            if self.config.monitor_enabled:
                if self._monitor_factory is None:
                    raise ClockLockError(
                        "Managed clock monitoring is enabled but no monitor "
                        "factory was configured."
                    )
                self._clock_monitor = self._monitor_factory(device, self.config)
                self._clock_monitor.start()
                self._monitor_started = True

            self._environ[ATREX_CLOCKS_LOCKED_ENV] = "1"
            self._environ[ATREX_CLOCK_LOCK_SOURCE_ENV] = "atrex-managed"
            return self
        except BaseException as error:
            cleanup_errors = self._cleanup_resources()
            error_message = str(error)
            if cleanup_errors:
                error_message = (
                    f"{error_message}; cleanup failed: "
                    f"{'; '.join(cleanup_errors)}"
                )
            self._publish_error(error_message)
            if isinstance(error, ClockLockError):
                raise
            if isinstance(error, Exception):
                raise ClockLockError(str(error)) from error
            raise

    @staticmethod
    def _handle_sigterm(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    def _install_sigterm_handler(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        self._previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        self._sigterm_installed = True

    def _restore_sigterm_handler(self) -> None:
        if not self._sigterm_installed:
            return
        signal.signal(signal.SIGTERM, self._previous_sigterm_handler)
        self._sigterm_installed = False

    def _shield_cleanup_from_sigterm(self) -> None:
        if not self._sigterm_installed:
            return
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def _save_environment(self) -> None:
        if self._environment_saved:
            return
        self._previous_marker = self._environ.get(
            ATREX_CLOCKS_LOCKED_ENV, _MISSING_ENV_VALUE
        )
        self._previous_source = self._environ.get(
            ATREX_CLOCK_LOCK_SOURCE_ENV, _MISSING_ENV_VALUE
        )
        self._environment_saved = True

    def _restore_environment(self) -> None:
        if not self._environment_saved:
            return
        for name, previous in (
            (ATREX_CLOCKS_LOCKED_ENV, self._previous_marker),
            (ATREX_CLOCK_LOCK_SOURCE_ENV, self._previous_source),
        ):
            if previous is _MISSING_ENV_VALUE:
                self._environ.pop(name, None)
            else:
                self._environ[name] = str(previous)

    def _cleanup_clocks(self) -> list[str]:
        errors: list[str] = []
        device = self._device
        clocks_changed = self._graphics_applied or self._memory_applied
        if device is None or not clocks_changed:
            return errors

        if self._graphics_applied:
            try:
                self._nvidia_smi.reset_graphics(device)
            except Exception as error:
                errors.append(f"reset_graphics: {error}")
        if self._memory_applied:
            try:
                self._nvidia_smi.reset_memory(device)
            except Exception as error:
                errors.append(f"reset_memory: {error}")

        snapshot: ClockSnapshot | None = None
        try:
            snapshot = self._nvidia_smi.query_clocks(device)
        except Exception as error:
            errors.append(f"query_post_restore_clocks: {error}")

        self._graphics_applied = False
        self._memory_applied = False
        self._publish(
            restored=not errors,
            post_restore_snapshot=snapshot,
        )
        return errors

    def _cleanup_monitor(self) -> list[str]:
        if not self._monitor_started or self._clock_monitor is None:
            return []
        self._monitor_started = False
        try:
            measurement = self._clock_monitor.stop()
        except Exception as error:
            self._publish(measurement_verified=False, verified=False)
            return [f"stop_clock_monitor: {error}"]

        self._publish(
            measurement=measurement,
            measurement_verified=measurement.verified,
            verified=self.report.setup_verified and measurement.verified,
        )
        if measurement.verified:
            return []
        if self._allows_unverified_clock_deviation(measurement):
            return []
        return [
            "verify_clock_measurement: "
            + (measurement.error or "measurement was not verified")
        ]

    def _allows_unverified_clock_deviation(
        self,
        measurement: ClockMeasurement,
    ) -> bool:
        """Allow only a fully observed, non-safety graphics-clock downclock."""
        if self.config.fail_on_deviation:
            return False
        if measurement.sample_count <= 0 or measurement.forbidden_reason_samples:
            return False
        if (
            measurement.graphics_min_mhz is None
            or measurement.graphics_max_mhz is None
            or self.config.graphics_mhz is None
        ):
            return False
        minimum = self.config.graphics_mhz - self.config.runtime_tolerance_mhz
        maximum = self.config.graphics_mhz + self.config.runtime_tolerance_mhz
        if (
            measurement.graphics_min_mhz >= minimum
            or measurement.graphics_max_mhz > maximum
        ):
            return False
        error = measurement.error or ""
        return (
            error.startswith("graphics clock outside runtime target range")
            and "; " not in error
        )

    def _cleanup_resources(self) -> list[str]:
        errors: list[str] = []
        try:
            self._shield_cleanup_from_sigterm()
        except Exception as error:
            errors.append(f"shield_cleanup_from_sigterm: {error}")
        try:
            errors.extend(self._cleanup_monitor())
        except Exception as error:
            errors.append(f"publish_monitor_report: {error}")
        try:
            errors.extend(self._cleanup_clocks())
        except Exception as error:
            errors.append(f"publish_cleanup_report: {error}")
        try:
            self._restore_environment()
        except Exception as error:
            errors.append(f"restore_environment: {error}")
        try:
            self._release_process_lock()
        except Exception as error:
            errors.append(f"release_process_lock: {error}")
        try:
            self._restore_sigterm_handler()
        except Exception as error:
            errors.append(f"restore_sigterm_handler: {error}")
        return errors

    def __exit__(self, exc_type, _exc_value, _traceback) -> bool:
        cleanup_errors = self._cleanup_resources()
        if cleanup_errors:
            restore_failed = any(
                error.startswith(
                    ("reset_", "query_post_restore_", "publish_cleanup_")
                )
                for error in cleanup_errors
            )
            prefix = (
                "Failed to restore GPU clocks"
                if restore_failed
                else "GPU clock measurement failed"
            )
            error_message = f"{prefix}: {'; '.join(cleanup_errors)}"
            error_message = self._publish_error(error_message)
            if exc_type is None:
                raise ClockLockError(error_message)
        return False


@dataclass(frozen=True)
class ClockLockStatus:
    locked: bool
    source: str | None


def get_clock_lock_status() -> ClockLockStatus:
    """Read the external clock-lock marker set by machine setup scripts."""
    for env_name in _CLOCK_LOCK_ENV_VARS:
        if os.environ.get(env_name, "0") == "1":
            return ClockLockStatus(locked=True, source=env_name)
    return ClockLockStatus(locked=False, source=None)


def are_clocks_locked() -> bool:
    """Return whether an external setup step marked clocks as locked."""
    return get_clock_lock_status().locked


def clock_lock_failure_reason(*, required: bool) -> str | None:
    """Return a failure reason when locked clocks are required but missing."""
    if not required:
        return None
    status = get_clock_lock_status()
    if status.locked:
        return None
    return (
        "GPU clock lock is required, but no locked-clock marker was found. "
        f"Set {ATREX_CLOCKS_LOCKED_ENV}=1 after external clock setup, or "
        f"{SOL_COMPAT_CLOCKS_LOCKED_ENV}=1 for SOL-compatible launchers."
    )
