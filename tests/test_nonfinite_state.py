"""Undefined state-pool padding must remain bitwise unchanged in both executions."""

import pytest
import torch

from atrex_bench.eval.correctness import _compare_value_trees


def compare(before, ref, candidate):
    return _compare_value_trees(
        {"cache": ref},
        {"cache": candidate},
        name="input.s",
        atol=0.0,
        rtol=0.0,
        strict_types=True,
        mutation_before={"cache": before},
    )


def test_untouched_nonfinite_state_and_finite_writes():
    before = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])
    ref = before.clone()
    ref[0] = 2
    diffs = compare(before, ref, ref.clone())
    assert all(d.passed for d in diffs)
    assert diffs[0].total_elements == 1
    # This exception is specific to declared state mutation; returned NaN fails.
    assert not _compare_value_trees(ref, ref, name="output", atol=0.0, rtol=0.0)[0].passed


@pytest.mark.parametrize(
    "kind", ["new_nan", "changed_nan_bits", "finite_mismatch", "changed_infinity"]
)
def test_padding_exception_never_hides_new_corruption(kind):
    before = torch.tensor([1.0, float("nan"), float("inf")])
    ref = before.clone()
    candidate = before.clone()
    if kind == "new_nan":
        ref[0] = candidate[0] = float("nan")
    elif kind == "changed_nan_bits":
        candidate.view(torch.int32)[1] ^= 1
    elif kind == "finite_mismatch":
        candidate[0] = 2
    else:
        candidate[2] = -float("inf")
    assert not all(d.passed for d in compare(before, ref, candidate))


def test_noncontiguous_chunked_state(monkeypatch):
    import atrex_bench.eval.correctness as correctness

    monkeypatch.setattr(correctness, "_COMPARISON_CHUNK_ELEMENTS", 3)
    before = torch.tensor([1.0, float("nan"), 3.0, float("inf"), 5.0, 6.0]).reshape(2, 3).T
    ref = before.clone()
    ref[0, 0] = 9
    assert all(d.passed for d in compare(before, ref, ref.clone()))
