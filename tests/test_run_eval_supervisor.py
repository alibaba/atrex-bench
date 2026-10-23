"""Supervisors must not create a device context just to prepare fallback metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from atrex_bench.cli import run_eval as runner
from atrex_bench.eval.clock_lock import ClockLockConfig

FIXTURES = Path(__file__).parent / "fixtures"
DEVICE_FIELDS = ("gpu_name", "gpu_arch", "accelerator_backend", "driver_version", "PTL_STATE")


@pytest.fixture
def no_device_queries(monkeypatch):
    was_initialized = runner.torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        pytest.fail("supervisor queried a GPU before launching its worker")

    for name in ("_gpu_info", "get_accelerator_backend", "_driver_version", "_ptl_state"):
        monkeypatch.setattr(runner, name, forbidden)
    for name in (
        "is_available",
        "current_device",
        "get_device_name",
        "get_device_properties",
        "init",
    ):
        monkeypatch.setattr(runner.torch.cuda, name, forbidden)
    yield
    assert runner.torch.cuda.is_initialized() == was_initialized


def test_supervisor_environment_preserves_cpu_and_clock_metadata(no_device_queries, monkeypatch):
    monkeypatch.setenv("ATREX_BENCH_CLOCKS_LOCKED", "1")
    env = runner._build_environment(clock_locked=False, collect_device=False)
    assert all(env[field] is None for field in DEVICE_FIELDS)
    assert env["python_version"]
    assert env["torch_version"]
    assert env["clock_locked"] is True
    assert env["clock_lock_detected"] is True
    assert env["clock_lock_source"] == "ATREX_BENCH_CLOCKS_LOCKED"


@pytest.mark.parametrize("mode", ["candidate", "torch_compile"])
@pytest.mark.parametrize(
    "outcome", ["success", "worker_failure", "missing_result", "spawn_failure", "preflight_failure"]
)
def test_supervisor_paths_never_query_devices_and_preserve_worker_metadata(
    tmp_path,
    monkeypatch,
    no_device_queries,
    mode,
    outcome,
):
    reference = tmp_path / "reference"
    shutil.copytree(FIXTURES / "references" / "atrex_001", reference)
    output = tmp_path / "output"
    candidate = FIXTURES / "generations" / "atrex_001.py"
    timestamp = "20261023-120000"
    launched = []
    worker_env = {
        "gpu_name": "worker GPU",
        "gpu_arch": "sm_89",
        "accelerator_backend": "cuda",
        "driver_version": "worker driver",
        "PTL_STATE": None,
    }

    def spawn(command, **kwargs):
        launched.append(command)
        result_path = next(output.rglob("eval_result.json"))
        initial = json.loads(result_path.read_text())
        assert all(initial["environment"][field] is None for field in DEVICE_FIELDS)
        if outcome == "spawn_failure":
            raise OSError("worker could not start")
        if outcome == "missing_result":
            result_path.unlink()
        else:
            initial["environment"].update(worker_env)
            result_path.write_text(json.dumps(initial))
        return subprocess.CompletedProcess(command, 1 if outcome == "worker_failure" else 0, "", "")

    monkeypatch.setattr(runner, "_run_subprocess_with_live_stderr", spawn)
    if outcome == "preflight_failure":
        (reference / "input.py").unlink()
    kwargs = dict(reference_dir=reference, output_root=output, timestamp=timestamp)
    if mode == "candidate":
        result = runner.run_eval(input_path=candidate, **kwargs)
    else:
        result = runner.run_torch_compile_eval(**kwargs)
    assert len(launched) == (0 if outcome == "preflight_failure" else 1)
    if outcome in {"success", "worker_failure"}:
        assert all(result["environment"][field] == worker_env[field] for field in DEVICE_FIELDS)
    else:
        assert all(result["environment"][field] is None for field in DEVICE_FIELDS)
        assert result["error"]


@pytest.mark.parametrize("mode", ["candidate", "torch_compile"])
def test_clock_acquisition_failure_does_not_initialize_cuda_for_its_report(
    tmp_path,
    monkeypatch,
    no_device_queries,
    mode,
):
    def clock_policy(*, config, build_failure_payload, **kwargs):
        assert config.mode == "manage"
        return build_failure_payload("clock acquisition failed")

    monkeypatch.setattr(runner, "_evaluate_with_clock_policy", clock_policy)
    kwargs = dict(
        reference_dir=FIXTURES / "references" / "atrex_001",
        output_root=tmp_path,
        clock_lock_config=ClockLockConfig(mode="manage", graphics_mhz=1000),
    )
    if mode == "candidate":
        result = runner.run_eval(input_path=FIXTURES / "generations" / "atrex_001.py", **kwargs)
    else:
        result = runner.run_torch_compile_eval(**kwargs)
    assert "clock acquisition failed" in result["error"]
    assert all(result["environment"][field] is None for field in DEVICE_FIELDS)
