"""Tests for device-scoped GPU clock management."""

from __future__ import annotations

import signal
import subprocess
import threading
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from atrex_bench.eval.clock_lock import (
    ClockLockConfig,
    ClockLockError,
    ClockLockReport,
    ManagedClockLock,
    resolve_clock_lock_selector,
)
from atrex_bench.eval.clock_monitor import ClockMeasurement
from atrex_bench.eval.nvidia_clock import (
    ClockSnapshot,
    ComputeProcess,
    NvidiaClockError,
    NvidiaDevice,
    NvidiaSmi,
)


@dataclass
class ScriptedRunner:
    """Queue subprocess results while retaining the exact invocation."""

    results: list[subprocess.CompletedProcess[str] | BaseException]
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, kwargs))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_resolve_device_uses_explicit_selector_for_every_query() -> None:
    runner = ScriptedRunner([completed("2, GPU-aabb, NVIDIA Test GPU\n")])
    smi = NvidiaSmi(run=runner, geteuid=lambda: 0, timeout_s=7.0)

    device = smi.resolve_device("GPU-aabb")

    assert device == NvidiaDevice(
        selector="GPU-aabb",
        index="2",
        uuid="GPU-aabb",
        name="NVIDIA Test GPU",
    )
    assert runner.calls == [
        (
            [
                "nvidia-smi",
                "-i",
                "GPU-aabb",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            {
                "capture_output": True,
                "text": True,
                "timeout": 7.0,
                "check": False,
            },
        )
    ]


def test_non_root_commands_use_noninteractive_sudo() -> None:
    runner = ScriptedRunner([completed("0, GPU-aabb, NVIDIA Test GPU\n")])
    smi = NvidiaSmi(run=runner, geteuid=lambda: 1000)

    smi.resolve_device("0")

    assert runner.calls[0][0][:3] == ["sudo", "-n", "nvidia-smi"]


@pytest.mark.parametrize(
    "selector",
    ["", "0,1", "../../0", "GPU-aabb;id", "MIG-aabb"],
)
def test_invalid_selector_is_rejected_before_subprocess(selector: str) -> None:
    runner = ScriptedRunner([])

    with pytest.raises(NvidiaClockError, match="GPU selector"):
        NvidiaSmi(run=runner, geteuid=lambda: 0).resolve_device(selector)

    assert runner.calls == []


def test_list_visible_indices_ignores_blank_rows() -> None:
    runner = ScriptedRunner([completed("0\n\n2\n")])

    indices = NvidiaSmi(run=runner, geteuid=lambda: 0).list_visible_indices()

    assert indices == ("0", "2")
    assert runner.calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader,nounits",
    ]


@pytest.mark.parametrize(
    ("stdout", "expected_error"),
    [
        ("", "rows=0"),
        (
            "0, GPU-aabb, NVIDIA Test GPU\n1, GPU-bbcc, NVIDIA Test GPU\n",
            "rows=2",
        ),
        ("0, GPU-aabb\n", "Could not parse GPU identity"),
    ],
)
def test_resolve_device_rejects_missing_multiple_or_malformed_rows(
    stdout: str,
    expected_error: str,
) -> None:
    runner = ScriptedRunner([completed(stdout)])

    with pytest.raises(NvidiaClockError, match=expected_error):
        NvidiaSmi(run=runner, geteuid=lambda: 0).resolve_device("0")


def test_query_clocks_parses_one_device_snapshot() -> None:
    runner = ScriptedRunner([completed("1500, 3996\n")])
    smi = NvidiaSmi(
        run=runner,
        geteuid=lambda: 0,
        now=lambda: "2026-07-31T00:00:00Z",
    )
    device = NvidiaDevice("GPU-aabb", "2", "GPU-aabb", "NVIDIA Test GPU")

    snapshot = smi.query_clocks(device)

    assert snapshot == ClockSnapshot(
        graphics_mhz=1500,
        memory_mhz=3996,
        observed_at="2026-07-31T00:00:00Z",
    )
    assert runner.calls[0][0] == [
        "nvidia-smi",
        "-i",
        "GPU-aabb",
        "--query-gpu=clocks.current.graphics,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ]


@pytest.mark.parametrize("stdout", ["", "N/A, 3996\n", "1500\n", "1500, 3996\n1400, 3996\n"])
def test_query_clocks_rejects_missing_or_malformed_values(stdout: str) -> None:
    runner = ScriptedRunner([completed(stdout)])
    device = NvidiaDevice("0", "0", "GPU-aabb", "NVIDIA Test GPU")

    with pytest.raises(NvidiaClockError, match="clock values"):
        NvidiaSmi(run=runner, geteuid=lambda: 0).query_clocks(device)


def test_list_compute_processes_parses_rows() -> None:
    runner = ScriptedRunner(
        [completed("101, python, GPU-aabb\n202, model server, GPU-aabb\n")]
    )
    device = NvidiaDevice("0", "0", "GPU-aabb", "NVIDIA Test GPU")

    processes = NvidiaSmi(run=runner, geteuid=lambda: 0).list_compute_processes(device)

    assert processes == (
        ComputeProcess(pid=101, process_name="python", gpu_uuid="GPU-aabb"),
        ComputeProcess(pid=202, process_name="model server", gpu_uuid="GPU-aabb"),
    )
    assert runner.calls[0][0] == [
        "nvidia-smi",
        "-i",
        "0",
        "--query-compute-apps=pid,process_name,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]


def test_list_compute_processes_accepts_empty_output() -> None:
    runner = ScriptedRunner([completed("")])
    device = NvidiaDevice("0", "0", "GPU-aabb", "NVIDIA Test GPU")

    processes = NvidiaSmi(run=runner, geteuid=lambda: 0).list_compute_processes(device)

    assert processes == ()


def test_clock_mutations_are_device_scoped() -> None:
    runner = ScriptedRunner([completed(), completed(), completed(), completed()])
    smi = NvidiaSmi(run=runner, geteuid=lambda: 0)
    device = NvidiaDevice("GPU-aabb", "2", "GPU-aabb", "NVIDIA Test GPU")

    smi.lock_graphics(device, 1500)
    smi.lock_memory(device, 3996)
    smi.reset_graphics(device)
    smi.reset_memory(device)

    assert [argv for argv, _ in runner.calls] == [
        ["nvidia-smi", "-i", "GPU-aabb", "-lgc", "1500,1500"],
        ["nvidia-smi", "-i", "GPU-aabb", "-lmc", "3996,3996"],
        ["nvidia-smi", "-i", "GPU-aabb", "-rgc"],
        ["nvidia-smi", "-i", "GPU-aabb", "-rmc"],
    ]


def test_nonzero_command_error_includes_argv_and_first_stderr_line() -> None:
    runner = ScriptedRunner(
        [completed(stderr="permission denied\nmore detail\n", returncode=6)]
    )
    device = NvidiaDevice("0", "0", "GPU-aabb", "NVIDIA Test GPU")

    with pytest.raises(NvidiaClockError) as captured:
        NvidiaSmi(run=runner, geteuid=lambda: 0).reset_graphics(device)

    message = str(captured.value)
    assert "nvidia-smi -i 0 -rgc" in message
    assert "permission denied" in message
    assert "more detail" not in message


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_zero_exit_unsupported_clock_command_is_rejected(stream: str) -> None:
    output = (
        "Setting locked Memory clocks is not supported for GPU 0000:E2:00.0.\n"
        "Treating as warning and moving on.\n"
    )
    result = completed(**{stream: output})
    runner = ScriptedRunner([result])
    device = NvidiaDevice("0", "0", "GPU-aabb", "NVIDIA Test GPU")

    with pytest.raises(NvidiaClockError, match="not supported"):
        NvidiaSmi(run=runner, geteuid=lambda: 0).lock_memory(device, 3996)


def test_command_timeout_has_actionable_error() -> None:
    runner = ScriptedRunner([subprocess.TimeoutExpired(["nvidia-smi"], timeout=4)])

    with pytest.raises(NvidiaClockError, match="timed out after 4s"):
        NvidiaSmi(run=runner, geteuid=lambda: 0, timeout_s=4).list_visible_indices()


def test_missing_management_executable_has_actionable_error() -> None:
    runner = ScriptedRunner([FileNotFoundError("sudo")])

    with pytest.raises(NvidiaClockError, match="executable was not found: sudo"):
        NvidiaSmi(run=runner, geteuid=lambda: 1000).list_visible_indices()


@pytest.mark.parametrize("mhz", [0, -1, True])
def test_clock_mutation_rejects_nonpositive_or_boolean_frequency(mhz: int) -> None:
    runner = ScriptedRunner([])
    device = NvidiaDevice("0", "0", "GPU-aabb", "NVIDIA Test GPU")

    with pytest.raises(NvidiaClockError, match="positive integer MHz"):
        NvidiaSmi(run=runner, geteuid=lambda: 0).lock_graphics(device, mhz)

    assert runner.calls == []


def test_command_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        NvidiaSmi(run=ScriptedRunner([]), geteuid=lambda: 0, timeout_s=0)


def test_managed_config_accepts_optional_memory_frequency() -> None:
    config = ClockLockConfig(
        mode="manage",
        device_selector="GPU-aabb",
        graphics_mhz=1500,
        tolerance_mhz=50,
        settle_seconds=3.0,
        command_timeout_seconds=10.0,
        require_idle=True,
    )

    assert config.mode == "manage"
    assert config.graphics_mhz == 1500
    assert config.memory_mhz is None
    assert config.monitor_enabled is True
    assert config.sample_interval_ms == 10
    assert config.runtime_tolerance_mhz == 0


def test_nonmanaged_config_defaults_monitor_off() -> None:
    assert ClockLockConfig(mode="off").monitor_enabled is False
    assert ClockLockConfig(mode="external").monitor_enabled is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "manage", "memory_mhz": 3996},
        {"mode": "manage", "graphics_mhz": 0, "memory_mhz": 3996},
        {"mode": "manage", "graphics_mhz": True, "memory_mhz": 3996},
        {"mode": "manage", "graphics_mhz": 1500, "memory_mhz": -1},
    ],
)
def test_managed_config_rejects_missing_or_invalid_frequencies(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ClockLockConfig(**kwargs)


@pytest.mark.parametrize("mode", ["off", "external"])
def test_nonmanaged_config_rejects_management_values(mode: str) -> None:
    with pytest.raises(ValueError, match="only valid in manage mode"):
        ClockLockConfig(mode=mode, graphics_mhz=1500)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "invalid"},
        {"tolerance_mhz": 0},
        {"settle_seconds": -1},
        {"command_timeout_seconds": 0},
        {"sample_interval_ms": 0},
        {"runtime_tolerance_mhz": -1},
    ],
)
def test_clock_lock_config_rejects_invalid_policy_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ClockLockConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_selector": 7},
        {"device_selector": "   "},
        {"settle_seconds": True},
        {"settle_seconds": float("nan")},
        {"command_timeout_seconds": True},
        {"command_timeout_seconds": float("inf")},
        {"require_idle": 1},
        {"monitor_enabled": 1},
        {"sample_interval_ms": True},
        {"runtime_tolerance_mhz": True},
    ],
)
def test_managed_config_rejects_invalid_boundary_types(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ClockLockConfig(
            mode="manage",
            graphics_mhz=1500,
            memory_mhz=3996,
            **kwargs,
        )


@dataclass
class IndexOnlySmi:
    indices: tuple[str, ...]

    def list_visible_indices(self) -> tuple[str, ...]:
        return self.indices


def test_selector_resolution_prefers_explicit_value() -> None:
    selector = resolve_clock_lock_selector(
        "GPU-eecc",
        environ={
            "ATREX_BENCH_CLOCK_DEVICE": "GPU-atrex",
            "CUDA_VISIBLE_DEVICES": "1",
        },
        nvidia_smi=IndexOnlySmi(("3",)),
    )

    assert selector == "GPU-eecc"


def test_selector_resolution_uses_atrex_environment_override() -> None:
    selector = resolve_clock_lock_selector(
        None,
        environ={
            "ATREX_BENCH_CLOCK_DEVICE": "GPU-atrex",
            "CUDA_VISIBLE_DEVICES": "1",
        },
        nvidia_smi=IndexOnlySmi(("3",)),
    )

    assert selector == "GPU-atrex"


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"CUDA_VISIBLE_DEVICES": " 2 "}, "2"),
        ({"NVIDIA_VISIBLE_DEVICES": "GPU-aabb"}, "GPU-aabb"),
        ({}, "3"),
        ({"NVIDIA_VISIBLE_DEVICES": "all"}, "3"),
    ],
)
def test_selector_resolution_uses_one_visible_device(
    environ: dict[str, str], expected: str
) -> None:
    selector = resolve_clock_lock_selector(
        None,
        environ=environ,
        nvidia_smi=IndexOnlySmi(("3",)),
    )

    assert selector == expected


@pytest.mark.parametrize(
    ("environ", "indices"),
    [
        ({"CUDA_VISIBLE_DEVICES": "0,1"}, ("0", "1")),
        ({}, ("0", "1")),
        ({}, ()),
    ],
)
def test_selector_resolution_rejects_ambiguous_or_missing_device(
    environ: dict[str, str], indices: tuple[str, ...]
) -> None:
    with pytest.raises(ClockLockError, match="exactly one GPU"):
        resolve_clock_lock_selector(
            None,
            environ=environ,
            nvidia_smi=IndexOnlySmi(indices),
        )


@dataclass
class FakeNvidiaSmi:
    snapshots: list[ClockSnapshot]
    processes: tuple[ComputeProcess, ...] = ()
    failures: set[str] = field(default_factory=set)
    events: list[str] = field(default_factory=list)

    def _record(self, event: str) -> None:
        self.events.append(event)
        if event in self.failures:
            raise NvidiaClockError(f"injected failure: {event}")

    def list_visible_indices(self) -> tuple[str, ...]:
        self._record("list_visible_indices")
        return ("0",)

    def resolve_device(self, selector: str) -> NvidiaDevice:
        self._record(f"resolve:{selector}")
        return NvidiaDevice(
            selector=selector,
            index="0",
            uuid="GPU-aabb",
            name="NVIDIA Test GPU",
        )

    def list_compute_processes(
        self, device: NvidiaDevice
    ) -> tuple[ComputeProcess, ...]:
        self._record("processes")
        return self.processes

    def query_clocks(self, device: NvidiaDevice) -> ClockSnapshot:
        self._record("query_clocks")
        return self.snapshots.pop(0)

    def lock_graphics(self, device: NvidiaDevice, mhz: int) -> None:
        self._record(f"lock_graphics:{mhz}")

    def lock_memory(self, device: NvidiaDevice, mhz: int) -> None:
        self._record(f"lock_memory:{mhz}")

    def reset_graphics(self, device: NvidiaDevice) -> None:
        self._record("reset_graphics")

    def reset_memory(self, device: NvidiaDevice) -> None:
        self._record("reset_memory")


def stable_measurement(*, verified: bool = True) -> ClockMeasurement:
    return ClockMeasurement(
        verified=verified,
        sample_interval_ms=10,
        runtime_target_mhz=1132,
        runtime_tolerance_mhz=0,
        sample_count=10,
        graphics_min_mhz=1132 if verified else 1125,
        graphics_max_mhz=1132,
        memory_min_mhz=3996,
        memory_max_mhz=3996,
        sw_power_cap_samples=0,
        forbidden_reason_samples=0,
        max_power_watts=1084.7,
        max_temperature_c=56,
        trace_path="clock_lock_trace.csv",
        error=None if verified else "graphics clock outside runtime target range",
    )


@dataclass
class FakeClockMonitor:
    events: list[str]
    measurement: ClockMeasurement = field(default_factory=stable_measurement)
    start_error: BaseException | None = None
    stop_error: BaseException | None = None

    def start(self) -> None:
        self.events.append("monitor_start")
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> ClockMeasurement:
        self.events.append("monitor_stop")
        if self.stop_error is not None:
            raise self.stop_error
        return self.measurement


def _managed_lock(tmp_path, smi, **kwargs) -> ManagedClockLock:
    return ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1500,
            memory_mhz=None,
            settle_seconds=0,
            monitor_enabled=False,
        ),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
        **kwargs,
    )


def test_managed_lock_refuses_a_gpu_this_process_does_not_run_on(tmp_path) -> None:
    """A physical index is not the visible ordinal, and locking the wrong card
    on a shared node hits another tenant with no downstream signal at all."""
    smi = FakeNvidiaSmi(snapshots=[ClockSnapshot(1500, 3996, "t0")])

    with pytest.raises(ClockLockError) as excinfo:
        with _managed_lock(
            tmp_path, smi, visible_device_uuid=lambda: "GPU-ccdd"
        ):
            pass

    assert "GPU-aabb" in str(excinfo.value)
    assert "GPU-ccdd" in str(excinfo.value)
    # Refused before touching the clocks, not after.
    assert not any(event.startswith("lock_") for event in smi.events)


def test_managed_lock_accepts_the_same_uuid_written_differently(tmp_path) -> None:
    """The prefix and case differ between sources; the identity does not."""
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1500, 3996, "t0"),
            ClockSnapshot(900, 3996, "t1"),
        ]
    )

    with _managed_lock(tmp_path, smi, visible_device_uuid=lambda: "aabb"):
        pass

    assert "lock_graphics:1500" in smi.events


def test_managed_lock_skips_the_check_when_the_device_is_unknowable(tmp_path) -> None:
    """No CUDA, ROCm, or no UUID: the cross-check goes quiet instead of failing."""
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1500, 3996, "t0"),
            ClockSnapshot(900, 3996, "t1"),
        ]
    )

    with _managed_lock(tmp_path, smi, visible_device_uuid=lambda: None):
        pass

    assert "lock_graphics:1500" in smi.events


def test_managed_lock_verifies_sets_marker_and_restores(tmp_path) -> None:
    reports: list[ClockLockReport] = []
    environ = {"CUDA_VISIBLE_DEVICES": "0"}
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1500, 3996, "2026-07-31T00:00:00Z"),
            ClockSnapshot(900, 3996, "2026-07-31T00:01:00Z"),
        ]
    )

    with ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1500,
            memory_mhz=3996,
            settle_seconds=0,
            monitor_enabled=False,
        ),
        nvidia_smi=smi,
        environ=environ,
        sleeper=lambda _: None,
        report_callback=reports.append,
        lock_directory=tmp_path,
    ) as session:
        assert session.report.verified is True
        assert environ["ATREX_BENCH_CLOCKS_LOCKED"] == "1"
        assert environ["ATREX_BENCH_CLOCK_LOCK_SOURCE"] == "atrex-managed"

    assert session.report.restored is True
    assert "ATREX_BENCH_CLOCKS_LOCKED" not in environ
    assert "ATREX_BENCH_CLOCK_LOCK_SOURCE" not in environ
    report_payload = session.report.to_dict()
    assert report_payload["requested"] == {
        "graphics_mhz": 1500,
        "memory_mhz": 3996,
        "tolerance_mhz": 50,
    }
    assert report_payload["device"]["uuid"] == "GPU-aabb"
    assert smi.events == [
        "resolve:0",
        "processes",
        "lock_graphics:1500",
        "lock_memory:3996",
        "query_clocks",
        "reset_graphics",
        "reset_memory",
        "query_clocks",
    ]
    assert reports[-1] == session.report


def test_managed_graphics_only_skips_memory_mutation_and_verification(
    tmp_path,
) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1500, 3200, "2026-08-01T00:00:00Z"),
            ClockSnapshot(900, 3996, "2026-08-01T00:01:00Z"),
        ]
    )
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1500,
            memory_mhz=None,
            settle_seconds=0,
            monitor_enabled=False,
        ),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with session:
        assert session.report.verified is True

    assert session.report.requested_memory_mhz is None
    assert session.report.verification_snapshot.memory_mhz == 3200
    assert session.report.restored is True
    assert smi.events == [
        "resolve:0",
        "processes",
        "lock_graphics:1500",
        "query_clocks",
        "reset_graphics",
        "query_clocks",
    ]


def test_managed_monitor_covers_body_and_stops_before_clock_reset(
    tmp_path,
) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1132, 3996, "2026-08-02T00:00:00Z"),
            ClockSnapshot(120, 3996, "2026-08-02T00:01:00Z"),
        ]
    )
    monitor = FakeClockMonitor(smi.events)
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1132,
            monitor_enabled=True,
            settle_seconds=0,
        ),
        nvidia_smi=smi,
        monitor_factory=lambda _device, _config: monitor,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with session:
        assert session.report.setup_verified is True
        assert session.report.measurement_verified is None
        assert session.report.verified is False
        assert smi.events[-1] == "monitor_start"

    assert session.report.measurement_verified is True
    assert session.report.verified is True
    assert session.report.measurement == stable_measurement()
    assert smi.events[-3:] == ["monitor_stop", "reset_graphics", "query_clocks"]


def test_measurement_violation_fails_closed_after_clock_reset(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1132, 3996, "2026-08-02T00:00:00Z"),
            ClockSnapshot(120, 3996, "2026-08-02T00:01:00Z"),
        ]
    )
    monitor = FakeClockMonitor(
        smi.events,
        measurement=stable_measurement(verified=False),
    )
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1132,
            monitor_enabled=True,
            settle_seconds=0,
        ),
        nvidia_smi=smi,
        monitor_factory=lambda _device, _config: monitor,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="outside runtime target"):
        with session:
            pass

    assert session.report.setup_verified is True
    assert session.report.measurement_verified is False
    assert session.report.verified is False
    assert session.report.restored is True
    assert smi.events[-3:] == ["monitor_stop", "reset_graphics", "query_clocks"]


def test_measurement_downclock_is_recorded_without_failing_when_configured(
    tmp_path,
) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1132, 3996, "2026-08-02T00:00:00Z"),
            ClockSnapshot(120, 3996, "2026-08-02T00:01:00Z"),
        ]
    )
    monitor = FakeClockMonitor(
        smi.events,
        measurement=stable_measurement(verified=False),
    )
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1132,
            monitor_enabled=True,
            settle_seconds=0,
            fail_on_deviation=False,
        ),
        nvidia_smi=smi,
        monitor_factory=lambda _device, _config: monitor,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with session:
        pass

    assert session.report.measurement_verified is False
    assert session.report.verified is False
    assert session.report.restored is True
    assert session.report.error is None
    assert session.report.to_dict()["fail_on_deviation"] is False
    assert smi.events[-3:] == ["monitor_stop", "reset_graphics", "query_clocks"]


def test_permissive_deviation_policy_still_rejects_forbidden_clock_events(
    tmp_path,
) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1132, 3996, "2026-08-02T00:00:00Z"),
            ClockSnapshot(120, 3996, "2026-08-02T00:01:00Z"),
        ]
    )
    monitor = FakeClockMonitor(
        smi.events,
        measurement=replace(
            stable_measurement(verified=False),
            forbidden_reason_samples=1,
            error=(
                "graphics clock outside runtime target range; forbidden thermal "
                "or hardware clock event reason observed in 1 sample(s)"
            ),
        ),
    )
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1132,
            monitor_enabled=True,
            settle_seconds=0,
            fail_on_deviation=False,
        ),
        nvidia_smi=smi,
        monitor_factory=lambda _device, _config: monitor,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="forbidden thermal"):
        with session:
            pass

    assert session.report.measurement_verified is False
    assert session.report.restored is True


def test_permissive_deviation_policy_still_rejects_above_target_clock(
    tmp_path,
) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1132, 3996, "2026-08-02T00:00:00Z"),
            ClockSnapshot(120, 3996, "2026-08-02T00:01:00Z"),
        ]
    )
    monitor = FakeClockMonitor(
        smi.events,
        measurement=replace(
            stable_measurement(verified=False),
            graphics_min_mhz=1132,
            graphics_max_mhz=1140,
            error="graphics clock outside runtime target range",
        ),
    )
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1132,
            monitor_enabled=True,
            settle_seconds=0,
            fail_on_deviation=False,
        ),
        nvidia_smi=smi,
        monitor_factory=lambda _device, _config: monitor,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="outside runtime target range"):
        with session:
            pass

    assert session.report.measurement_verified is False
    assert session.report.restored is True


def test_monitor_start_failure_resets_managed_clock(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1132, 3996, "2026-08-02T00:00:00Z"),
            ClockSnapshot(120, 3996, "2026-08-02T00:01:00Z"),
        ]
    )
    monitor = FakeClockMonitor(
        smi.events,
        start_error=RuntimeError("monitor unavailable"),
    )
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1132,
            monitor_enabled=True,
            settle_seconds=0,
        ),
        nvidia_smi=smi,
        monitor_factory=lambda _device, _config: monitor,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="monitor unavailable"):
        with session:
            pytest.fail("body must not run")

    assert session.report.verified is False
    assert session.report.restored is True
    assert smi.events[-2:] == ["reset_graphics", "query_clocks"]


def test_monitor_stop_failure_does_not_skip_clock_reset(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1132, 3996, "2026-08-02T00:00:00Z"),
            ClockSnapshot(120, 3996, "2026-08-02T00:01:00Z"),
        ]
    )
    monitor = FakeClockMonitor(
        smi.events,
        stop_error=RuntimeError("monitor stop failed"),
    )
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            graphics_mhz=1132,
            monitor_enabled=True,
            settle_seconds=0,
        ),
        nvidia_smi=smi,
        monitor_factory=lambda _device, _config: monitor,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="monitor stop failed"):
        with session:
            pass

    assert session.report.verified is False
    assert session.report.restored is True
    assert smi.events[-3:] == ["monitor_stop", "reset_graphics", "query_clocks"]


def managed_config(*, require_idle: bool = True) -> ClockLockConfig:
    return ClockLockConfig(
        mode="manage",
        graphics_mhz=1500,
        memory_mhz=3996,
        settle_seconds=0,
        require_idle=require_idle,
        monitor_enabled=False,
    )


def successful_snapshots() -> list[ClockSnapshot]:
    return [
        ClockSnapshot(1500, 3996, "2026-07-31T00:00:00Z"),
        ClockSnapshot(900, 3996, "2026-07-31T00:01:00Z"),
    ]


def test_managed_lock_rejects_busy_gpu(tmp_path) -> None:
    busy_smi = FakeNvidiaSmi(
        snapshots=successful_snapshots(),
        processes=(ComputeProcess(202, "model server", "GPU-aabb"),),
    )
    busy_session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=busy_smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
        getpid=lambda: 101,
    )

    with pytest.raises(ClockLockError, match="pid=202 name=model server"):
        with busy_session:
            pass

    assert busy_smi.events == ["resolve:0", "processes"]
    assert "not idle" in (busy_session.report.error or "")

    idle_smi = FakeNvidiaSmi(snapshots=successful_snapshots())
    with ManagedClockLock(
        config=managed_config(),
        nvidia_smi=idle_smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    ):
        pass


def test_managed_lock_ignores_current_process(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=successful_snapshots(),
        processes=(ComputeProcess(101, "run_eval", "GPU-aabb"),),
    )

    with ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
        getpid=lambda: 101,
    ):
        pass

    assert "lock_graphics:1500" in smi.events


def test_allow_busy_gpu_skips_process_query(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=successful_snapshots(),
        processes=(ComputeProcess(202, "model server", "GPU-aabb"),),
    )

    with ManagedClockLock(
        config=managed_config(require_idle=False),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    ):
        pass

    assert "processes" not in smi.events


def test_same_device_process_lock_rejects_concurrent_session(tmp_path) -> None:
    first_smi = FakeNvidiaSmi(snapshots=successful_snapshots())
    second_smi = FakeNvidiaSmi(snapshots=successful_snapshots())

    with ManagedClockLock(
        config=managed_config(),
        nvidia_smi=first_smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    ):
        second_session = ManagedClockLock(
            config=managed_config(),
            nvidia_smi=second_smi,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            sleeper=lambda _: None,
            lock_directory=tmp_path,
        )
        with pytest.raises(ClockLockError, match="Another Atrex process"):
            with second_session:
                pass

    assert second_smi.events == ["resolve:0"]


def test_graphics_lock_failure_does_not_issue_reset(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=successful_snapshots(), failures={"lock_graphics:1500"}
    )
    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="lock_graphics"):
        with session:
            pass

    assert "reset_graphics" not in smi.events
    assert "reset_memory" not in smi.events
    assert "lock_graphics" in (session.report.error or "")


def test_memory_lock_failure_resets_graphics(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[ClockSnapshot(900, 3996, "2026-07-31T00:01:00Z")],
        failures={"lock_memory:3996"},
    )
    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="lock_memory"):
        with session:
            pass

    assert smi.events[-2:] == ["reset_graphics", "query_clocks"]
    assert session.report.restored is True


def test_unsupported_memory_warning_rolls_back_managed_graphics_lock(
    tmp_path,
) -> None:
    unsupported = (
        "This option is not supported. "
        "Please use --lock-memory-clocks-deferred instead.\n"
    )
    runner = ScriptedRunner(
        [
            completed("7, GPU-aabb, NVIDIA Test GPU\n"),
            completed(""),
            completed(),
            completed(stdout=unsupported),
            completed(),
            completed("900, 3996\n"),
        ]
    )
    session = ManagedClockLock(
        config=ClockLockConfig(
            mode="manage",
            device_selector="GPU-aabb",
            graphics_mhz=1500,
            memory_mhz=3996,
            settle_seconds=0,
        ),
        nvidia_smi=NvidiaSmi(run=runner, geteuid=lambda: 0),
        environ={},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="unsupported operation"):
        with session:
            pass

    commands = [argv for argv, _kwargs in runner.calls]
    assert commands[-2:] == [
        ["nvidia-smi", "-i", "GPU-aabb", "-rgc"],
        [
            "nvidia-smi",
            "-i",
            "GPU-aabb",
            "--query-gpu=clocks.current.graphics,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ],
    ]
    assert session.report.requested_memory_mhz == 3996
    assert session.report.verified is False
    assert session.report.restored is True
    assert "not supported" in (session.report.error or "")


def test_clock_mismatch_resets_both_clocks(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=[
            ClockSnapshot(1400, 3996, "2026-07-31T00:00:00Z"),
            ClockSnapshot(900, 3996, "2026-07-31T00:01:00Z"),
        ]
    )
    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="graphics clock mismatch"):
        with session:
            pass

    assert smi.events[-3:] == ["reset_graphics", "reset_memory", "query_clocks"]
    assert session.report.verified is False
    assert session.report.restored is True


class BodyFailure(RuntimeError):
    pass


def test_managed_lock_restores_after_body_exception(tmp_path) -> None:
    smi = FakeNvidiaSmi(snapshots=successful_snapshots())

    with pytest.raises(BodyFailure, match="candidate failed"):
        with ManagedClockLock(
            config=managed_config(),
            nvidia_smi=smi,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            sleeper=lambda _: None,
            lock_directory=tmp_path,
        ):
            raise BodyFailure("candidate failed")

    assert smi.events[-3:] == ["reset_graphics", "reset_memory", "query_clocks"]


def test_reset_failure_attempts_other_reset_and_query(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=successful_snapshots(), failures={"reset_graphics"}
    )
    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="restore GPU clocks"):
        with session:
            pass

    assert smi.events[-3:] == ["reset_graphics", "reset_memory", "query_clocks"]
    assert session.report.restored is False
    assert "reset_graphics" in (session.report.error or "")


def test_cleanup_error_does_not_mask_body_exception(tmp_path) -> None:
    smi = FakeNvidiaSmi(
        snapshots=successful_snapshots(), failures={"reset_memory"}
    )
    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(BodyFailure, match="candidate failed"):
        with session:
            raise BodyFailure("candidate failed")

    assert smi.events[-3:] == ["reset_graphics", "reset_memory", "query_clocks"]
    assert session.report.restored is False
    assert "reset_memory" in (session.report.error or "")


def test_previous_clock_marker_values_are_restored(tmp_path) -> None:
    environ = {
        "CUDA_VISIBLE_DEVICES": "0",
        "ATREX_BENCH_CLOCKS_LOCKED": "legacy",
        "ATREX_BENCH_CLOCK_LOCK_SOURCE": "launcher",
    }
    smi = FakeNvidiaSmi(snapshots=successful_snapshots())

    with ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ=environ,
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    ):
        assert environ["ATREX_BENCH_CLOCKS_LOCKED"] == "1"
        assert environ["ATREX_BENCH_CLOCK_LOCK_SOURCE"] == "atrex-managed"

    assert environ["ATREX_BENCH_CLOCKS_LOCKED"] == "legacy"
    assert environ["ATREX_BENCH_CLOCK_LOCK_SOURCE"] == "launcher"


def test_report_write_failure_does_not_block_remaining_cleanup(tmp_path) -> None:
    environ = {"CUDA_VISIBLE_DEVICES": "0"}
    smi = FakeNvidiaSmi(snapshots=successful_snapshots())

    def fail_on_restore(report: ClockLockReport) -> None:
        if report.restored:
            raise OSError("clock report filesystem is full")

    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ=environ,
        sleeper=lambda _: None,
        report_callback=fail_on_restore,
        lock_directory=tmp_path,
    )

    with pytest.raises(ClockLockError, match="publish_cleanup_report"):
        with session:
            pass

    assert "ATREX_BENCH_CLOCKS_LOCKED" not in environ
    assert "ATREX_BENCH_CLOCK_LOCK_SOURCE" not in environ
    assert "clock report filesystem is full" in (session.report.error or "")

    second_smi = FakeNvidiaSmi(snapshots=successful_snapshots())
    with ManagedClockLock(
        config=managed_config(),
        nvidia_smi=second_smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    ):
        pass


def test_sigterm_unwinds_managed_lock_and_restores_previous_handler(
    tmp_path,
) -> None:
    previous_handler = signal.getsignal(signal.SIGTERM)
    smi = FakeNvidiaSmi(snapshots=successful_snapshots())
    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(SystemExit) as exc_info:
        with session:
            temporary_handler = signal.getsignal(signal.SIGTERM)
            assert temporary_handler is not previous_handler
            assert callable(temporary_handler)
            temporary_handler(signal.SIGTERM, None)

    assert exc_info.value.code == 143
    assert session.report.restored is True
    assert smi.events[-3:] == ["reset_graphics", "reset_memory", "query_clocks"]
    assert signal.getsignal(signal.SIGTERM) is previous_handler


def test_repeated_sigterm_cannot_interrupt_clock_cleanup(tmp_path) -> None:
    previous_handler = signal.getsignal(signal.SIGTERM)
    cleanup_handlers: list[object] = []

    class RepeatedSigtermSmi(FakeNvidiaSmi):
        def reset_graphics(self, device: NvidiaDevice) -> None:
            cleanup_handler = signal.getsignal(signal.SIGTERM)
            cleanup_handlers.append(cleanup_handler)
            if callable(cleanup_handler):
                cleanup_handler(signal.SIGTERM, None)
            super().reset_graphics(device)

    smi = RepeatedSigtermSmi(snapshots=successful_snapshots())
    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    try:
        with pytest.raises(SystemExit) as exc_info:
            with session:
                temporary_handler = signal.getsignal(signal.SIGTERM)
                assert callable(temporary_handler)
                temporary_handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    assert exc_info.value.code == 143
    assert cleanup_handlers == [signal.SIG_IGN]
    assert session.report.restored is True
    assert smi.events[-3:] == ["reset_graphics", "reset_memory", "query_clocks"]
    assert signal.getsignal(signal.SIGTERM) is previous_handler


def test_keyboard_interrupt_unwinds_managed_lock(tmp_path) -> None:
    smi = FakeNvidiaSmi(snapshots=successful_snapshots())
    session = ManagedClockLock(
        config=managed_config(),
        nvidia_smi=smi,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        sleeper=lambda _: None,
        lock_directory=tmp_path,
    )

    with pytest.raises(KeyboardInterrupt):
        with session:
            raise KeyboardInterrupt

    assert session.report.restored is True
    assert smi.events[-3:] == ["reset_graphics", "reset_memory", "query_clocks"]


def test_non_main_thread_skips_sigterm_handler_but_still_restores_clocks(
    tmp_path,
) -> None:
    previous_handler = signal.getsignal(signal.SIGTERM)
    smi = FakeNvidiaSmi(snapshots=successful_snapshots())
    observed_handlers: list[object] = []
    failures: list[BaseException] = []

    def run_session() -> None:
        try:
            with ManagedClockLock(
                config=managed_config(),
                nvidia_smi=smi,
                environ={"CUDA_VISIBLE_DEVICES": "0"},
                sleeper=lambda _: None,
                lock_directory=tmp_path,
            ) as session:
                observed_handlers.append(signal.getsignal(signal.SIGTERM))
            assert session.report.restored is True
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=run_session)
    worker.start()
    worker.join(timeout=5)

    assert worker.is_alive() is False
    assert failures == []
    assert observed_handlers == [previous_handler]
    assert smi.events[-3:] == ["reset_graphics", "reset_memory", "query_clocks"]
