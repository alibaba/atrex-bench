import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from atrex_bench.eval import distributed
from atrex_bench.eval._runtime import clone_value, ModelInputs


def metadata(tmp_path, count=4, **extra):
    value = {"hardware": {"required_gpus": count}, "benchmark_contract": {
        "distributed_evaluation": {"launcher": "torchrun", "backend": "nccl", "world_size": count}}}
    value.update(extra)
    (tmp_path / "metadata.json").write_text(json.dumps(value))
    return tmp_path


@pytest.mark.parametrize("count", [2, 4, 8])
def test_launch_keeps_contract_and_arguments(tmp_path, count):
    argv = ["python", "run_eval.py", "--single-shape-worker", "--shape-id", "s1"]
    command = distributed.launch_command(argv, distributed.world_size(metadata(tmp_path, count)))
    assert command[-4:] == argv[1:]
    assert f"--nproc-per-node={count}" in command
    assert "--max-restarts=0" in command


def test_single_gpu_does_not_launch_torchrun(tmp_path):
    argv = ["python", "run_eval.py"]
    assert distributed.launch_command(argv, distributed.world_size(tmp_path)) is argv


@pytest.mark.parametrize("contract", [None, {"launcher": "torchrun", "backend": "nccl", "world_size": 2},
    {"launcher": "torchrun", "backend": "nccl", "world_size": True},
    {"launcher": "torchrun", "backend": "gloo", "world_size": 4}])
def test_invalid_or_mismatched_contract_fails_closed(tmp_path, contract):
    metadata(tmp_path, benchmark_contract={"distributed_evaluation": contract})
    with pytest.raises(ValueError):
        distributed.world_size(tmp_path)


def test_timing_uses_slowest_rank_per_iteration():
    assert distributed.max_rank_samples([[1, 4], [2, 3], [3, 2], [4, 1]]) == [4, 4]


@pytest.mark.parametrize("samples", [[], [[], []], [[1, 2], [1]], [[1], [float('nan')]], [[1], [0]]])
def test_missing_or_invalid_rank_timings_rejected(samples):
    with pytest.raises(ValueError):
        distributed.max_rank_samples(samples)


def test_process_group_not_deepcopied():
    class Group:
        def __deepcopy__(self, memo):
            raise RuntimeError("communicator must not be copied")
    group = Group()
    with patch.object(torch.distributed, "ProcessGroup", Group):
        cloned = clone_value({"x": torch.ones(2), "group": group})
    assert cloned["group"] is group


def test_collective_storage_and_off_rank_mismatch_fail():
    x = torch.ones(4)
    inputs = ModelInputs(args=(x,), kwargs={})
    contract = {"validation_requirements": ["output storage independence", "same output on all ranks"]}
    with patch.object(distributed, "active", return_value=True), \
         patch.object(distributed, "gather", side_effect=lambda value: [value] * 4):
        distributed.check_output_contract(inputs, x.clone(), contract)
        with pytest.raises(ValueError, match="aliases"):
            distributed.check_output_contract(inputs, x, contract)
    with patch.object(distributed, "active", return_value=True), \
         patch.object(distributed, "gather", return_value=["a", "a", "a", "b"]):
        with pytest.raises(ValueError, match="differs between ranks"):
            distributed.check_output_contract(inputs, x.clone(), contract)


def test_worker_command_uses_four_ranks(tmp_path):
    from atrex_bench.cli.run_eval import _build_single_shape_worker_command
    metadata(tmp_path)
    command = _build_single_shape_worker_command(candidate_path=Path("candidate.py"),
        reference_dir=tmp_path, artifact_dir=tmp_path, checkpoint_root=tmp_path,
        shape_id="0", atol=0.01, rtol=0.05, num_correctness_cases=5, warmup_iters=10,
        bench_iters=100, shape_result_path=tmp_path / "result.json", collect_kernel_events=False,
        candidate_timeout_s=60, perf_timeout_s=120)
    assert "--nproc-per-node=4" in command
    assert command[command.index("--num-correctness-cases") + 1] == "5"


def test_missing_gpu_allocation_does_not_initialize_nccl(monkeypatch):
    for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "4"}.items():
        monkeypatch.setenv(key, value)
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "device_count", return_value=1), \
         patch.object(torch.distributed, "init_process_group") as init:
        with pytest.raises(RuntimeError, match="4 distinct"):
            with distributed.rank_session(4, 60):
                pass
        init.assert_not_called()
