"""Evaluator-owned single-node NCCL launch, rank agreement and timing."""
from __future__ import annotations

import json
import hashlib
import math
import os
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def world_size(reference_dir: Path) -> int:
    path = reference_dir / "metadata.json"
    if not path.is_file():
        return 1
    metadata = json.loads(path.read_text())
    contract = metadata.get("benchmark_contract", {}).get("distributed_evaluation")
    required = metadata.get("hardware", {}).get("required_gpus", 1)
    if type(required) is not int or required < 1:
        raise ValueError("hardware.required_gpus must be a positive integer")
    if contract is None:
        if required > 1:
            raise ValueError("required_gpus > 1 requires a distributed_evaluation contract")
        return 1
    if not isinstance(contract, dict):
        raise ValueError("distributed_evaluation must be an object")
    count = contract.get("world_size")
    if (contract.get("launcher") != "torchrun" or contract.get("backend") != "nccl"
            or type(count) is not int or not 2 <= count <= 64):
        raise ValueError("distributed_evaluation requires torchrun, nccl and world_size in [2, 64]")
    if required != 1 and required != count:
        raise ValueError("hardware.required_gpus differs from distributed_evaluation.world_size")
    return count


def launch_command(argv: list[str], count: int) -> list[str]:
    if count == 1:
        return argv
    return [argv[0], "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={count}", "--max-restarts=0", *argv[1:]]


def active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def gather(value):
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, value)
    return results


@contextmanager
def rank_session(count: int, timeout_s: float):
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (int(os.environ.get("WORLD_SIZE", "0")) != count
            or not 0 <= rank < count or local_rank != rank):
        raise RuntimeError("distributed evaluator requires matching single-node torchrun ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        raise RuntimeError(f"distributed evaluator requires {count} distinct visible CUDA devices")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=count,
                            timeout=timedelta(seconds=max(1, timeout_s)))
    try:
        props = torch.cuda.get_device_properties(local_rank)
        devices = gather({"rank": rank, "local_rank": local_rank,
                          "device": torch.cuda.current_device(),
                          "uuid": str(getattr(props, "uuid", ""))})
        uuids = [item["uuid"] for item in devices]
        if all(uuids) and len(set(uuids)) != count:
            raise RuntimeError("multiple ranks bound to the same GPU")
        print("[distributed-eval] " + json.dumps({"world_size": count,
              "backend": "nccl", "ranks": devices}), flush=True)
        yield devices
    finally:
        dist.destroy_process_group()


def all_pass(passed: bool) -> bool:
    if not active():
        return passed
    vote = torch.tensor(int(passed), device="cuda", dtype=torch.int32)
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    return bool(vote.item())


def max_rank_samples(rank_samples: list[list[float]]) -> list[float]:
    if (not rank_samples or not rank_samples[0]
            or any(len(row) != len(rank_samples[0]) for row in rank_samples)
            or any(not math.isfinite(x) or x <= 0 for row in rank_samples for x in row)):
        raise ValueError("all ranks must report the same number of positive finite timings")
    return [max(values) for values in zip(*rank_samples)]


def measure(bench_fn, *, device, warmup_iters: int, bench_iters: int):
    # Fixed iteration counts on every rank. Adaptive do_bench loops can deadlock
    # collectives when different ranks choose different iteration counts.
    for _ in range(warmup_iters):
        dist.barrier()
        bench_fn()
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(bench_iters):
        dist.barrier()
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        bench_fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ranks = gather(samples)
    return max_rank_samples(ranks), ranks


def check_output_contract(inputs, output, contract: dict) -> None:
    """Enforce explicit collective storage/rank guarantees outside timing."""
    from atrex_bench.eval._runtime import flatten_outputs

    requirements = contract.get("validation_requirements", [])
    if not active():
        return
    outputs = [value for _, value in flatten_outputs(output)]
    if "same output on all ranks" in requirements:
        fingerprint = hashlib.sha256()
        for value in outputs:
            if not isinstance(value, torch.Tensor):
                raise ValueError("rank-equality contract requires tensor outputs")
            fingerprint.update(str((tuple(value.shape), value.dtype)).encode())
            fingerprint.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        if len(set(gather(fingerprint.hexdigest()))) != 1:
            raise ValueError("collective output differs between ranks")
    if "output storage independence" in requirements:
        def tensors(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from tensors(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from tensors(item)
        storage = {item.untyped_storage().data_ptr() for item in tensors((inputs.args, inputs.kwargs))}
        if any(isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() in storage
               for value in outputs):
            raise ValueError("collective output aliases an input")
