"""Process-local reward-hacking guards for untrusted evaluation mode.

These checks detect specific evaluator-tampering patterns. They are not a
security sandbox; callers must still isolate untrusted code at the worker
boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class RewardHackDetected(RuntimeError):
    """Raised when untrusted candidate code appears to tamper with evaluation."""


_ELAPSED_TIME_ID: int | None = None
try:
    _ELAPSED_TIME_ID = id(torch.cuda.Event.elapsed_time)
except Exception:
    _ELAPSED_TIME_ID = None


def check_cuda_event_monkey_patch() -> None:
    """Detect replacement of the CUDA event timing primitive."""
    if _ELAPSED_TIME_ID is None:
        return
    try:
        current_id = id(torch.cuda.Event.elapsed_time)
    except Exception:
        return
    if current_id != _ELAPSED_TIME_ID:
        raise RewardHackDetected("torch.cuda.Event.elapsed_time has been monkey-patched")


def check_thread_injection(threads_before: int, threads_after: int) -> None:
    """Detect background threads started by candidate evaluation."""
    if threads_after > threads_before:
        raise RewardHackDetected(
            "Background thread injection detected: "
            f"{threads_after} active threads after evaluation vs {threads_before} before"
        )


def check_plain_tensor_outputs(output: Any) -> None:
    """Reject lazy/proxy tensor outputs in untrusted mode.

    Scalars are allowed because Atrex references may return scalar leaves.
    Tensor subclasses are rejected: they can defer computation or override
    behavior in ways that make correctness checks observe a proxy rather than
    an actual materialized tensor.
    """
    if isinstance(output, torch.Tensor):
        if type(output) is not torch.Tensor:
            raise RewardHackDetected(
                f"Lazy/proxy tensor output detected: {type(output).__name__}"
            )
        return
    if isinstance(output, (bool, int, float)):
        return
    if isinstance(output, dict):
        for item in output.values():
            check_plain_tensor_outputs(item)
        return
    if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
        for item in output:
            check_plain_tensor_outputs(item)


def snapshot_critical_functions(targets: dict[str, Any]) -> dict[str, int]:
    """Capture identities of critical evaluator functions before candidate import."""
    return {name: id(value) for name, value in targets.items()}


def check_eval_integrity(snapshot: dict[str, int], targets: dict[str, Any]) -> None:
    """Detect replacement of critical evaluator functions."""
    for name, expected_id in snapshot.items():
        if id(targets.get(name)) != expected_id:
            raise RewardHackDetected(
                f"Eval driver integrity violated: {name!r} has been monkey-patched"
            )
