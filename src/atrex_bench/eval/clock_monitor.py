"""Continuous NVIDIA clock telemetry for managed benchmark windows."""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

CLOCK_EVENT_GPU_IDLE = 1 << 0
CLOCK_EVENT_APPLICATIONS_CLOCKS_SETTING = 1 << 1
CLOCK_EVENT_SW_POWER_CAP = 1 << 2
CLOCK_EVENT_HW_SLOWDOWN = 1 << 3
CLOCK_EVENT_SYNC_BOOST = 1 << 4
CLOCK_EVENT_SW_THERMAL_SLOWDOWN = 1 << 5
CLOCK_EVENT_HW_THERMAL_SLOWDOWN = 1 << 6
CLOCK_EVENT_HW_POWER_BRAKE_SLOWDOWN = 1 << 7

FORBIDDEN_CLOCK_EVENT_MASK = (
    CLOCK_EVENT_HW_SLOWDOWN
    | CLOCK_EVENT_SW_THERMAL_SLOWDOWN
    | CLOCK_EVENT_HW_THERMAL_SLOWDOWN
    | CLOCK_EVENT_HW_POWER_BRAKE_SLOWDOWN
)

_QUERY_FIELDS = (
    "timestamp,clocks.current.graphics,clocks.current.memory,"
    "clocks_event_reasons.active,power.draw.average,power.limit,"
    "temperature.gpu"
)


class ClockMonitorError(RuntimeError):
    """Raised when continuous clock telemetry cannot be managed safely."""


def _effective_uid() -> int:
    """Return the POSIX effective uid, or root-equivalent on non-POSIX hosts."""
    geteuid = getattr(os, "geteuid", None)
    return int(geteuid()) if geteuid is not None else 0


@dataclass(frozen=True)
class ClockSample:
    """One parsed row from an ``nvidia-smi`` loop query."""

    observed_at: str
    graphics_mhz: int
    memory_mhz: int
    reason_mask: int
    power_watts: float
    power_limit_watts: float
    temperature_c: int


@dataclass(frozen=True)
class ClockMeasurement:
    """Bounded summary of a complete managed evaluation window."""

    verified: bool
    sample_interval_ms: int
    runtime_target_mhz: int
    runtime_tolerance_mhz: int
    sample_count: int
    graphics_min_mhz: int | None
    graphics_max_mhz: int | None
    memory_min_mhz: int | None
    memory_max_mhz: int | None
    sw_power_cap_samples: int
    forbidden_reason_samples: int
    max_power_watts: float | None
    max_temperature_c: int | None
    trace_path: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible immutable measurement summary."""
        return asdict(self)


def _parse_int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid clock telemetry {field}: {value!r}."
        ) from error


def _parse_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid clock telemetry {field}: {value!r}."
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid clock telemetry {field}: {value!r}.")
    return parsed


def parse_clock_sample(line: str) -> ClockSample:
    """Parse one strict seven-column ``nvidia-smi`` telemetry row."""
    parts = tuple(part.strip() for part in line.strip().split(","))
    if len(parts) != 7 or not all(parts):
        raise ValueError(f"Invalid clock telemetry row: {line!r}.")

    try:
        reason_mask = int(parts[3], 0)
    except ValueError as error:
        raise ValueError(
            f"Invalid clock telemetry reason mask: {parts[3]!r}."
        ) from error

    return ClockSample(
        observed_at=parts[0],
        graphics_mhz=_parse_int(parts[1], "graphics clock"),
        memory_mhz=_parse_int(parts[2], "memory clock"),
        reason_mask=reason_mask,
        power_watts=_parse_float(parts[4], "power draw"),
        power_limit_watts=_parse_float(parts[5], "power limit"),
        temperature_c=_parse_int(parts[6], "temperature"),
    )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_policy(
    *, target_mhz: int, tolerance_mhz: int, sample_interval_ms: int
) -> None:
    if not _is_int(target_mhz) or target_mhz <= 0:
        raise ValueError("target_mhz must be a positive integer")
    if not _is_int(tolerance_mhz) or tolerance_mhz < 0:
        raise ValueError("tolerance_mhz must be a non-negative integer")
    if not _is_int(sample_interval_ms) or sample_interval_ms <= 0:
        raise ValueError("sample_interval_ms must be a positive integer")


def summarize_clock_samples(
    samples: Iterable[ClockSample],
    *,
    target_mhz: int,
    tolerance_mhz: int,
    sample_interval_ms: int,
    trace_path: Path,
) -> ClockMeasurement:
    """Validate and summarize clock samples for one worker lifecycle."""
    _validate_policy(
        target_mhz=target_mhz,
        tolerance_mhz=tolerance_mhz,
        sample_interval_ms=sample_interval_ms,
    )
    captured = tuple(samples)
    trace_name = trace_path.name
    if not captured:
        return ClockMeasurement(
            verified=False,
            sample_interval_ms=sample_interval_ms,
            runtime_target_mhz=target_mhz,
            runtime_tolerance_mhz=tolerance_mhz,
            sample_count=0,
            graphics_min_mhz=None,
            graphics_max_mhz=None,
            memory_min_mhz=None,
            memory_max_mhz=None,
            sw_power_cap_samples=0,
            forbidden_reason_samples=0,
            max_power_watts=None,
            max_temperature_c=None,
            trace_path=trace_name,
            error="Clock telemetry contained no valid samples.",
        )

    graphics = tuple(sample.graphics_mhz for sample in captured)
    memory = tuple(sample.memory_mhz for sample in captured)
    minimum = target_mhz - tolerance_mhz
    maximum = target_mhz + tolerance_mhz
    deviated = tuple(value for value in graphics if not minimum <= value <= maximum)
    forbidden_count = sum(
        bool(sample.reason_mask & FORBIDDEN_CLOCK_EVENT_MASK)
        for sample in captured
    )
    errors: list[str] = []
    if deviated:
        errors.append(
            "graphics clock outside runtime target range: "
            f"target={target_mhz} MHz tolerance={tolerance_mhz} MHz "
            f"observed={min(graphics)}-{max(graphics)} MHz"
        )
    if forbidden_count:
        errors.append(
            "forbidden thermal or hardware clock event reason observed in "
            f"{forbidden_count} sample(s)"
        )

    return ClockMeasurement(
        verified=not errors,
        sample_interval_ms=sample_interval_ms,
        runtime_target_mhz=target_mhz,
        runtime_tolerance_mhz=tolerance_mhz,
        sample_count=len(captured),
        graphics_min_mhz=min(graphics),
        graphics_max_mhz=max(graphics),
        memory_min_mhz=min(memory),
        memory_max_mhz=max(memory),
        sw_power_cap_samples=sum(
            bool(sample.reason_mask & CLOCK_EVENT_SW_POWER_CAP)
            for sample in captured
        ),
        forbidden_reason_samples=forbidden_count,
        max_power_watts=max(sample.power_watts for sample in captured),
        max_temperature_c=max(sample.temperature_c for sample in captured),
        trace_path=trace_name,
        error="; ".join(errors) or None,
    )


class NvidiaClockMonitor:
    """Own one device-scoped ``nvidia-smi`` telemetry subprocess."""

    def __init__(
        self,
        *,
        device_uuid: str,
        target_mhz: int,
        tolerance_mhz: int,
        sample_interval_ms: int,
        trace_path: Path,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        geteuid: Callable[[], int] | None = None,
        stop_timeout_seconds: float = 2.0,
    ) -> None:
        _validate_policy(
            target_mhz=target_mhz,
            tolerance_mhz=tolerance_mhz,
            sample_interval_ms=sample_interval_ms,
        )
        if (
            isinstance(stop_timeout_seconds, bool)
            or not isinstance(stop_timeout_seconds, (int, float))
            or not math.isfinite(stop_timeout_seconds)
            or stop_timeout_seconds <= 0
        ):
            raise ValueError("stop_timeout_seconds must be positive")
        if not device_uuid:
            raise ValueError("device_uuid must be non-empty")

        self._device_uuid = device_uuid
        self._target_mhz = target_mhz
        self._tolerance_mhz = tolerance_mhz
        self._sample_interval_ms = sample_interval_ms
        self._trace_path = trace_path
        self._popen = popen
        self._geteuid = geteuid or _effective_uid
        self._stop_timeout_seconds = float(stop_timeout_seconds)
        self._process: subprocess.Popen[str] | None = None
        self._trace_handle = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._samples: list[ClockSample] = []
        self._errors: list[str] = []
        self._stderr_lines: list[str] = []
        self._started = False
        self._first_sample = threading.Event()
        self._stdout_finished = threading.Event()

    def _command(self) -> list[str]:
        prefix = (
            ["nvidia-smi"]
            if self._geteuid() == 0
            else ["sudo", "-n", "nvidia-smi"]
        )
        return [
            *prefix,
            "-i",
            self._device_uuid,
            f"--query-gpu={_QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self._sample_interval_ms}",
        ]

    def _read_stdout(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        assert self._trace_handle is not None
        try:
            for line in self._process.stdout:
                self._trace_handle.write(line)
                self._trace_handle.flush()
                try:
                    self._samples.append(parse_clock_sample(line))
                    self._first_sample.set()
                except ValueError as error:
                    self._errors.append(str(error))
        except Exception as error:
            self._errors.append(f"clock telemetry stdout reader failed: {error}")
        finally:
            self._stdout_finished.set()

    def _read_stderr(self) -> None:
        assert self._process is not None
        assert self._process.stderr is not None
        try:
            self._stderr_lines.extend(
                line.strip() for line in self._process.stderr if line.strip()
            )
        except Exception as error:
            self._errors.append(f"clock telemetry stderr reader failed: {error}")

    def start(self) -> None:
        """Start telemetry before the managed worker is launched."""
        if self._started:
            raise ClockMonitorError("Clock telemetry monitor was already started.")
        self._started = True
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._trace_handle = self._trace_path.open("w", encoding="utf-8")
            self._process = self._popen(
                self._command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            if self._trace_handle is not None:
                self._trace_handle.close()
            raise ClockMonitorError(
                f"Clock telemetry executable was not found: {self._command()[0]}"
            ) from error

        if self._process.stdout is None or self._process.stderr is None:
            self._process.terminate()
            self._trace_handle.close()
            raise ClockMonitorError("Clock telemetry subprocess pipes were unavailable.")

        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="atrex-clock-monitor-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name="atrex-clock-monitor-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._stop_timeout_seconds
        while not self._first_sample.is_set():
            if self._stdout_finished.wait(timeout=0.01):
                break
            if time.monotonic() >= deadline:
                break

        returncode = self._process.poll() if self._process is not None else None
        if self._first_sample.is_set() and returncode is None:
            return

        measurement = self.stop()
        detail = measurement.error or "clock telemetry did not become ready"
        raise ClockMonitorError(f"Clock telemetry startup failed: {detail}")

    def _stop_process(self) -> None:
        assert self._process is not None
        returncode = self._process.poll()
        if returncode is not None:
            self._errors.append(
                f"clock telemetry process exited with code {returncode}"
            )
            return

        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=self._stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                self._errors.append(
                    "clock telemetry process did not terminate within "
                    f"{self._stop_timeout_seconds:g}s"
                )
                self._process.kill()
                try:
                    self._process.wait(timeout=self._stop_timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._errors.append(
                        "clock telemetry process did not exit after kill "
                        f"within {self._stop_timeout_seconds:g}s"
                    )
                    self._close_process_pipes()
        except Exception as error:
            self._errors.append(f"clock telemetry process cleanup failed: {error}")
            self._close_process_pipes()

    def _close_process_pipes(self) -> None:
        if self._process is None:
            return
        for stream_name in ("stdout", "stderr"):
            stream = getattr(self._process, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception as error:
                self._errors.append(
                    f"clock telemetry {stream_name} close failed: {error}"
                )

    def _join_readers(self) -> None:
        for thread, stream_name in (
            (self._stdout_thread, "stdout"),
            (self._stderr_thread, "stderr"),
        ):
            if thread is None:
                continue
            thread.join(timeout=self._stop_timeout_seconds)
            if thread.is_alive():
                self._close_process_pipes()
                thread.join(timeout=self._stop_timeout_seconds)
                if thread.is_alive():
                    self._errors.append(
                        f"clock telemetry {stream_name} reader did not stop: "
                        f"{thread.name}"
                    )

    def stop(self) -> ClockMeasurement:
        """Stop telemetry and return the final fail-closed summary."""
        if self._process is None or self._trace_handle is None:
            raise ClockMonitorError("Clock telemetry monitor is not running.")

        try:
            self._stop_process()
        finally:
            try:
                self._join_readers()
            finally:
                try:
                    self._trace_handle.close()
                except Exception as error:
                    self._errors.append(
                        f"clock telemetry trace close failed: {error}"
                    )
        if self._stderr_lines:
            self._errors.append(
                "clock telemetry stderr: " + self._stderr_lines[0]
            )

        measurement = summarize_clock_samples(
            self._samples,
            target_mhz=self._target_mhz,
            tolerance_mhz=self._tolerance_mhz,
            sample_interval_ms=self._sample_interval_ms,
            trace_path=self._trace_path,
        )
        errors = [error for error in (measurement.error, *self._errors) if error]
        if not errors:
            return measurement
        return replace(
            measurement,
            verified=False,
            error="; ".join(errors),
        )
