"""Tests for continuous NVIDIA clock telemetry."""

from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from atrex_bench.eval.clock_monitor import (
    CLOCK_EVENT_HW_POWER_BRAKE_SLOWDOWN,
    CLOCK_EVENT_HW_THERMAL_SLOWDOWN,
    CLOCK_EVENT_SW_POWER_CAP,
    CLOCK_EVENT_SW_THERMAL_SLOWDOWN,
    ClockMonitorError,
    ClockSample,
    NvidiaClockMonitor,
    parse_clock_sample,
    summarize_clock_samples,
)


@dataclass
class FakeProcess:
    stdout_text: str
    stderr_text: str = ""
    returncode: int | None = None
    timeout_on_wait: bool = False
    timeout_after_kill: bool = False
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stdout = io.StringIO(self.stdout_text)
        self.stderr = io.StringIO(self.stderr_text)

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")
        if not self.timeout_on_wait:
            self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(f"wait:{timeout}")
        if self.timeout_on_wait and "kill" not in self.events:
            raise subprocess.TimeoutExpired(["nvidia-smi"], timeout)
        if self.timeout_after_kill and "kill" in self.events:
            raise subprocess.TimeoutExpired(["nvidia-smi"], timeout)
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        self.events.append("kill")
        self.returncode = -9


@dataclass
class FakePopen:
    process: FakeProcess | BaseException
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((argv, kwargs))
        if isinstance(self.process, BaseException):
            raise self.process
        return self.process


def sample_line(
    *,
    graphics_mhz: int = 1132,
    memory_mhz: int = 3996,
    reason_mask: int = 0,
    power_watts: float = 1084.7,
    power_limit_watts: float = 1100.0,
    temperature_c: int = 56,
) -> str:
    return (
        "2026/08/01 23:59:10.226, "
        f"{graphics_mhz}, {memory_mhz}, 0x{reason_mask:016x}, "
        f"{power_watts}, {power_limit_watts}, {temperature_c}"
    )


def test_parse_clock_sample_preserves_all_hardware_fields() -> None:
    parsed = parse_clock_sample(
        sample_line(reason_mask=CLOCK_EVENT_SW_POWER_CAP)
    )

    assert parsed == ClockSample(
        observed_at="2026/08/01 23:59:10.226",
        graphics_mhz=1132,
        memory_mhz=3996,
        reason_mask=CLOCK_EVENT_SW_POWER_CAP,
        power_watts=1084.7,
        power_limit_watts=1100.0,
        temperature_c=56,
    )


@pytest.mark.parametrize(
    "line",
    [
        "",
        "2026/08/01 23:59:10.226, 1132",
        sample_line().replace("1132", "N/A", 1),
        sample_line().replace("0x0000000000000000", "not-a-mask"),
        sample_line().replace("1084.7", "N/A"),
    ],
)
def test_parse_clock_sample_rejects_incomplete_or_malformed_rows(
    line: str,
) -> None:
    with pytest.raises(ValueError, match="clock telemetry"):
        parse_clock_sample(line)


def test_stable_samples_are_measurement_verified(tmp_path: Path) -> None:
    samples = tuple(parse_clock_sample(sample_line()) for _ in range(3))

    measurement = summarize_clock_samples(
        samples,
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "clock_lock_trace.csv",
    )

    assert measurement.verified is True
    assert measurement.sample_count == 3
    assert measurement.graphics_min_mhz == 1132
    assert measurement.graphics_max_mhz == 1132
    assert measurement.memory_min_mhz == 3996
    assert measurement.memory_max_mhz == 3996
    assert measurement.sw_power_cap_samples == 0
    assert measurement.forbidden_reason_samples == 0
    assert measurement.max_power_watts == 1084.7
    assert measurement.max_temperature_c == 56
    assert measurement.trace_path == "clock_lock_trace.csv"
    assert measurement.error is None


@pytest.mark.parametrize("observed_mhz", [1125, 1140])
def test_any_clock_deviation_invalidates_measurement(
    tmp_path: Path,
    observed_mhz: int,
) -> None:
    measurement = summarize_clock_samples(
        (parse_clock_sample(sample_line(graphics_mhz=observed_mhz)),),
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "clock_lock_trace.csv",
    )

    assert measurement.verified is False
    assert "outside" in (measurement.error or "")


def test_runtime_tolerance_is_applied_symmetrically(tmp_path: Path) -> None:
    samples = (
        parse_clock_sample(sample_line(graphics_mhz=1125)),
        parse_clock_sample(sample_line(graphics_mhz=1139)),
    )

    measurement = summarize_clock_samples(
        samples,
        target_mhz=1132,
        tolerance_mhz=7,
        sample_interval_ms=10,
        trace_path=tmp_path / "clock_lock_trace.csv",
    )

    assert measurement.verified is True


def test_sw_power_cap_without_clock_deviation_is_diagnostic(
    tmp_path: Path,
) -> None:
    measurement = summarize_clock_samples(
        (
            parse_clock_sample(
                sample_line(reason_mask=CLOCK_EVENT_SW_POWER_CAP)
            ),
        ),
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "clock_lock_trace.csv",
    )

    assert measurement.verified is True
    assert measurement.sw_power_cap_samples == 1


@pytest.mark.parametrize(
    "reason_mask",
    [
        CLOCK_EVENT_SW_THERMAL_SLOWDOWN,
        CLOCK_EVENT_HW_THERMAL_SLOWDOWN,
        CLOCK_EVENT_HW_POWER_BRAKE_SLOWDOWN,
    ],
)
def test_thermal_or_hardware_reasons_invalidate_measurement(
    tmp_path: Path,
    reason_mask: int,
) -> None:
    measurement = summarize_clock_samples(
        (parse_clock_sample(sample_line(reason_mask=reason_mask)),),
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "clock_lock_trace.csv",
    )

    assert measurement.verified is False
    assert measurement.forbidden_reason_samples == 1
    assert "clock event reason" in (measurement.error or "")


def test_empty_trace_is_not_verified(tmp_path: Path) -> None:
    measurement = summarize_clock_samples(
        (),
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "clock_lock_trace.csv",
    )

    assert measurement.verified is False
    assert measurement.sample_count == 0
    assert "no valid samples" in (measurement.error or "")


@pytest.mark.parametrize(
    ("target_mhz", "tolerance_mhz", "sample_interval_ms"),
    [(0, 0, 10), (1132, -1, 10), (1132, 0, 0), (True, 0, 10)],
)
def test_summary_rejects_invalid_policy_values(
    tmp_path: Path,
    target_mhz: int,
    tolerance_mhz: int,
    sample_interval_ms: int,
) -> None:
    with pytest.raises(ValueError):
        summarize_clock_samples(
            (parse_clock_sample(sample_line()),),
            target_mhz=target_mhz,
            tolerance_mhz=tolerance_mhz,
            sample_interval_ms=sample_interval_ms,
            trace_path=tmp_path / "clock_lock_trace.csv",
        )


def test_monitor_uses_device_scoped_fixed_query_and_persists_trace(
    tmp_path: Path,
) -> None:
    lines = "\n".join([sample_line(), sample_line(reason_mask=4)]) + "\n"
    process = FakeProcess(lines)
    popen = FakePopen(process)
    trace_path = tmp_path / "clock_lock_trace.csv"
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=trace_path,
        popen=popen,
        geteuid=lambda: 0,
        stop_timeout_seconds=2.0,
    )

    monitor.start()
    measurement = monitor.stop()

    assert measurement.verified is True
    assert measurement.sample_count == 2
    assert measurement.sw_power_cap_samples == 1
    assert trace_path.read_text(encoding="utf-8") == lines
    argv, kwargs = popen.calls[0]
    assert argv == [
        "nvidia-smi",
        "-i",
        "GPU-aabb",
        "--query-gpu=timestamp,clocks.current.graphics,clocks.current.memory,"
        "clocks_event_reasons.active,power.draw.average,power.limit,"
        "temperature.gpu",
        "--format=csv,noheader,nounits",
        "--loop-ms=10",
    ]
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True
    assert kwargs["bufsize"] == 1
    assert process.events == ["terminate", "wait:2.0"]


def test_non_root_monitor_uses_noninteractive_sudo(tmp_path: Path) -> None:
    popen = FakePopen(FakeProcess(sample_line() + "\n"))
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "trace.csv",
        popen=popen,
        geteuid=lambda: 1000,
    )

    monitor.start()
    monitor.stop()

    assert popen.calls[0][0][:3] == ["sudo", "-n", "nvidia-smi"]


def test_monitor_parse_error_invalidates_measurement(tmp_path: Path) -> None:
    popen = FakePopen(FakeProcess(sample_line() + "\nbad row\n"))
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "trace.csv",
        popen=popen,
        geteuid=lambda: 0,
    )

    monitor.start()
    measurement = monitor.stop()

    assert measurement.verified is False
    assert measurement.sample_count == 1
    assert "Invalid clock telemetry row" in (measurement.error or "")


def test_monitor_early_exit_invalidates_measurement(tmp_path: Path) -> None:
    process = FakeProcess(
        sample_line() + "\n",
        stderr_text="GPU disappeared\n",
        returncode=6,
    )
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "trace.csv",
        popen=FakePopen(process),
        geteuid=lambda: 0,
    )

    with pytest.raises(ClockMonitorError, match="exited with code 6") as captured:
        monitor.start()

    assert "GPU disappeared" in str(captured.value)
    assert process.events == []


def test_monitor_rejects_empty_stream_before_worker_launch(
    tmp_path: Path,
) -> None:
    process = FakeProcess("", returncode=0)
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "trace.csv",
        popen=FakePopen(process),
        geteuid=lambda: 0,
    )

    with pytest.raises(ClockMonitorError, match="no valid samples"):
        monitor.start()


def test_monitor_timeout_kills_process_and_invalidates_measurement(
    tmp_path: Path,
) -> None:
    process = FakeProcess(sample_line() + "\n", timeout_on_wait=True)
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "trace.csv",
        popen=FakePopen(process),
        geteuid=lambda: 0,
        stop_timeout_seconds=0.5,
    )

    monitor.start()
    measurement = monitor.stop()

    assert measurement.verified is False
    assert "did not terminate" in (measurement.error or "")
    assert process.events == ["terminate", "wait:0.5", "kill", "wait:0.5"]


def test_second_wait_timeout_still_closes_resources_and_returns_failure(
    tmp_path: Path,
) -> None:
    process = FakeProcess(
        sample_line() + "\n",
        timeout_on_wait=True,
        timeout_after_kill=True,
    )
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "trace.csv",
        popen=FakePopen(process),
        geteuid=lambda: 0,
        stop_timeout_seconds=0.5,
    )

    monitor.start()
    measurement = monitor.stop()

    assert measurement.verified is False
    assert "did not exit after kill" in (measurement.error or "")
    assert monitor._trace_handle.closed is True
    assert process.events == ["terminate", "wait:0.5", "kill", "wait:0.5"]


def test_monitor_start_failure_is_actionable(tmp_path: Path) -> None:
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "trace.csv",
        popen=FakePopen(FileNotFoundError("nvidia-smi")),
        geteuid=lambda: 0,
    )

    with pytest.raises(ClockMonitorError, match="executable was not found"):
        monitor.start()


def test_monitor_rejects_repeated_start_or_stop_before_start(
    tmp_path: Path,
) -> None:
    monitor = NvidiaClockMonitor(
        device_uuid="GPU-aabb",
        target_mhz=1132,
        tolerance_mhz=0,
        sample_interval_ms=10,
        trace_path=tmp_path / "trace.csv",
        popen=FakePopen(FakeProcess(sample_line() + "\n")),
        geteuid=lambda: 0,
    )

    with pytest.raises(ClockMonitorError, match="not running"):
        monitor.stop()
    monitor.start()
    with pytest.raises(ClockMonitorError, match="already started"):
        monitor.start()
    monitor.stop()
