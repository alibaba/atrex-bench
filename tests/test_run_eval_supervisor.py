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


@pytest.mark.parametrize("options", [{}, {"collect_device": False}], ids=["default", "explicit"])
def test_supervisor_environment_preserves_cpu_and_clock_metadata(
    no_device_queries, monkeypatch, options
):
    monkeypatch.setenv("ATREX_BENCH_CLOCKS_LOCKED", "1")
    env = runner._build_environment(clock_locked=False, **options)
    assert all(env[field] is None for field in DEVICE_FIELDS)
    assert env["python_version"]
    assert env["torch_version"]
    assert env["clock_locked"] is True
    assert env["clock_lock_detected"] is True
    assert env["clock_lock_source"] == "ATREX_BENCH_CLOCKS_LOCKED"


@pytest.mark.parametrize("mode", ["candidate", "torch_compile", "abba"])
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
        result_path = Path(command[command.index("--artifact-dir") + 1]) / "eval_result.json"
        initial = json.loads(result_path.read_text())
        assert all(initial["environment"][field] is None for field in DEVICE_FIELDS)
        if outcome == "spawn_failure":
            raise OSError("worker could not start")
        if outcome == "missing_result":
            result_path.unlink()
        else:
            initial["environment"].update(worker_env)
            initial["error"] = None
            for stage in initial["passed"].values():
                for verdict in stage.values():
                    verdict.update(status="passed", reason=None)
            for shape in initial["performance"]["shapes"].values():
                shape.update(samples=[{"end_to_end_time_ms": 1.0}], error=None)
            result_path.write_text(json.dumps(initial))
        return subprocess.CompletedProcess(command, 1 if outcome == "worker_failure" else 0, "", "")

    monkeypatch.setattr(runner, "_run_subprocess_with_live_stderr", spawn)
    if outcome == "preflight_failure":
        (reference / "input.py").unlink()
    kwargs = dict(reference_dir=reference, output_root=output, timestamp=timestamp)
    if mode == "candidate":
        result = runner.run_eval(input_path=candidate, **kwargs)
    elif mode == "abba":
        result = runner.run_abba_eval(baseline_path=candidate, candidate_path=candidate, **kwargs)
    else:
        result = runner.run_torch_compile_eval(**kwargs)
    expected_launches = 4 if mode == "abba" else 1
    assert len(launched) == (0 if outcome == "preflight_failure" else expected_launches)
    payloads = [result]
    if mode == "abba":
        payloads.extend(row["result"] for row in result["abba"]["runs"])
        assert result["passed"]["abba"]["status"] == (
            "passed" if outcome == "success" else "failed"
        )
    if outcome in {"success", "worker_failure"}:
        for payload in payloads:
            assert all(
                payload["environment"][field] == worker_env[field] for field in DEVICE_FIELDS
            )
    else:
        for payload in payloads:
            assert all(payload["environment"][field] is None for field in DEVICE_FIELDS)
        assert result["error"]


@pytest.mark.parametrize("missing_leg", [0, 3], ids=["first-leg-missing", "last-leg-missing"])
def test_abba_copies_first_leg_environment_when_some_workers_have_no_report(
    tmp_path, monkeypatch, no_device_queries, missing_leg
):
    environments = []
    worker_env = {
        "gpu_name": "worker GPU",
        "gpu_arch": "sm_89",
        "accelerator_backend": "cuda",
        "driver_version": "worker driver",
        "PTL_STATE": "worker PTL",
    }

    def run_leg(**kwargs):
        environment = runner._build_environment(clock_locked=False)
        if len(environments) != missing_leg:
            environment.update(worker_env)
        environments.append(environment)
        return {"environment": environment, "error": "leg failed", "passed": {}}

    monkeypatch.setattr(runner, "_run_eval_process", run_leg)
    candidate = FIXTURES / "generations" / "atrex_001.py"
    result = runner.run_abba_eval(
        baseline_path=candidate,
        candidate_path=candidate,
        reference_dir=FIXTURES / "references" / "atrex_001",
        output_root=tmp_path,
    )

    assert len(environments) == 4
    assert result["passed"]["abba"]["status"] == "failed"
    assert result["environment"] == environments[0]
    assert result["environment"] is not environments[0]
    for index, row in enumerate(result["abba"]["runs"]):
        assert row["result"]["environment"] == environments[index]
        assert all(
            row["result"]["environment"][field] == (
                None if index == missing_leg else worker_env[field]
            )
            for field in DEVICE_FIELDS
        )
    result["environment"]["clock_lock"] = {"mode": "external"}
    assert "clock_lock" not in environments[0]


@pytest.mark.parametrize("report", [{}, {"environment": None}], ids=["absent", "null"])
def test_abba_keeps_fallback_when_worker_report_has_no_environment(
    tmp_path, monkeypatch, no_device_queries, report
):
    monkeypatch.setattr(
        runner,
        "_run_eval_process",
        lambda **kwargs: {**report, "error": "worker report incomplete"},
    )
    candidate = FIXTURES / "generations" / "atrex_001.py"
    result = runner.run_abba_eval(
        baseline_path=candidate,
        candidate_path=candidate,
        reference_dir=FIXTURES / "references" / "atrex_001",
        output_root=tmp_path,
    )
    assert result["passed"]["abba"]["status"] == "failed"
    assert all(result["environment"][field] is None for field in DEVICE_FIELDS)
    assert "worker report incomplete" in result["error"]


@pytest.mark.parametrize("mode", ["candidate", "torch_compile", "abba"])
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
    elif mode == "abba":
        candidate = FIXTURES / "generations" / "atrex_001.py"
        result = runner.run_abba_eval(baseline_path=candidate, candidate_path=candidate, **kwargs)
    else:
        result = runner.run_torch_compile_eval(**kwargs)
    assert "clock acquisition failed" in result["error"]
    assert all(result["environment"][field] is None for field in DEVICE_FIELDS)


@pytest.mark.parametrize("mode", ["candidate", "torch_compile"])
def test_workers_collect_device_metadata(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(runner, "_gpu_info", lambda: ("worker GPU", "sm_89"))
    monkeypatch.setattr(runner, "get_accelerator_backend", lambda: "cuda")
    monkeypatch.setattr(runner, "_driver_version", lambda: "worker driver")
    monkeypatch.setattr(runner, "_ptl_state", lambda: "worker PTL")
    # Stop before any GPU work while exercising each real worker entrypoint.
    monkeypatch.setattr(runner, "clock_lock_failure_reason", lambda **kwargs: "clock not locked")
    kwargs = dict(
        reference_dir=FIXTURES / "references" / "atrex_001",
        artifact_dir=tmp_path,
        warmup_iters=1,
        bench_iters=1,
        checkpoint_dir=None,
        config_version="v1",
        clock_locked=False,
        require_clock_locked=True,
    )
    if mode == "candidate":
        result = runner._run_eval_worker(
            input_path=FIXTURES / "generations" / "atrex_001.py",
            atol=0.01,
            rtol=0.05,
            num_correctness_cases=1,
            collect_kernel_events=False,
            candidate_timeout_s=1,
            perf_timeout_s=1,
            **kwargs,
        )
    else:
        result = runner._run_torch_compile_worker(**kwargs)
    assert {field: result["environment"][field] for field in DEVICE_FIELDS} == {
        "gpu_name": "worker GPU",
        "gpu_arch": "sm_89",
        "accelerator_backend": "cuda",
        "driver_version": "worker driver",
        "PTL_STATE": "worker PTL",
    }
