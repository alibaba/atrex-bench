"""Reference-relative metadata gates preserve legacy defaults and other checks."""
import json

import pytest
import torch

from atrex_bench.eval.correctness import (
    _compare_value_trees,
    _load_output_tolerance_contract,
)


def compare(tmp_path, reference, candidate, **policy):
    path = tmp_path / "reference.py"
    (tmp_path / "metadata.json").write_text(json.dumps({
        "benchmark_contract": {"correctness_tolerances": {"output": {
            "atol": 0.001, "rtol": 0.01, **policy,
        }}}
    }))
    contract = _load_output_tolerance_contract(path)
    return _compare_value_trees(
        torch.tensor(reference, dtype=torch.float64),
        torch.tensor(candidate, dtype=torch.float64),
        name="output", atol=1.0, rtol=1.0, output_tolerances=contract,
    )[0]


@pytest.mark.parametrize("ref,cand,reference_pass,candidate_pass", [
    ([1.0], [1.105], False, True),
    ([1.0], [0.905], True, False),
    ([-1.0], [-1.105], False, True),
    ([0.0], [0.001], True, True),
])
def test_direction_and_legacy_default(tmp_path, ref, cand, reference_pass, candidate_pass):
    result = compare(tmp_path, ref, cand, atol=0.001, rtol=0.1, relative_to="reference")
    assert result.passed == reference_pass
    assert result.tolerance["relative_to"] == "reference"
    legacy = compare(tmp_path, ref, cand, atol=0.001, rtol=0.1)
    assert legacy.passed == candidate_pass
    assert legacy.tolerance["relative_to"] == "candidate"


def test_l2_is_conjunctive(tmp_path):
    result = compare(tmp_path, [1.0], [1.005], relative_to="reference", max_rel_l2=0.001)
    assert result.mismatched_elements == 0
    assert result.relative_l2 > 0.001
    assert not result.passed


def test_elementwise_is_conjunctive(tmp_path):
    result = compare(tmp_path, [0.0, 100.0], [0.002, 100.0],
                     relative_to="reference", max_rel_l2=0.001)
    assert result.relative_l2 < 0.001
    assert result.mismatched_elements == 1
    assert not result.passed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(tmp_path, value):
    result = compare(tmp_path, [1.0], [value], relative_to="reference", max_rel_l2=0.001)
    assert not result.passed
    assert "Non-finite" in result.error


def test_zero_reference_requires_exact_zero_for_l2(tmp_path):
    result = compare(tmp_path, [0.0], [0.0001], relative_to="reference", max_rel_l2=0.001)
    assert result.mismatched_elements == 0
    assert not result.passed


@pytest.mark.parametrize("value", ["baseline", "", None, 1])
def test_invalid_relative_to_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="relative_to"):
        compare(tmp_path, [1.0], [1.0], relative_to=value)
