"""Regression tests for metadata-owned correctness policies."""

import json

import pytest

from atrex_bench.eval.correctness import check_correctness, metadata_owns_correctness


def _write_problem(tmp_path, *, reference_body, candidate_body, contract, inputs):
    common = (
        "import torch\nclass Model(torch.nn.Module):\n"
        "    def forward(self, x, out, workspace):\n        "
    )
    reference = tmp_path / "reference.py"
    candidate = tmp_path / "candidate.py"
    reference.write_text(
        common
        + reference_body
        + "\n\ndef get_inputs():\n    return "
        + inputs
        + "\n\ndef get_init_inputs():\n    return []\n",
        encoding="utf-8",
    )
    candidate.write_text(common + candidate_body + "\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"benchmark_contract": contract}), encoding="utf-8"
    )
    return reference, candidate


def test_per_output_tolerances_override_global_policy(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return (torch.ones_like(x), torch.ones_like(x))",
        candidate_body="return (torch.full_like(x, 1.05), torch.full_like(x, 1.005))",
        contract={
            "correctness_tolerances": {
                "output[0]": {"atol": 0.06, "rtol": 0.0},
                "output[1]": {"atol": 0.006, "rtol": 0.0},
            }
        },
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(
        reference,
        candidate,
        atol=0.0,
        rtol=0.0,
        max_rel_l2=0.0,
        device="cpu",
    )

    assert result.status == "passed", result.reason
    assert [diff.name for diff in result.cases[0].outputs] == ["output[0]", "output[1]"]
    assert metadata_owns_correctness(reference)


def test_mutation_tolerance_and_scratch_input(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="out.copy_(x + 1); workspace.fill_(1); return x",
        candidate_body="out.copy_(x + 1.05); workspace.fill_(2); return x",
        contract={
            "mutates_inputs": ["out"],
            "scratch_inputs": ["workspace"],
            "correctness_tolerances": {
                "output": {"atol": 0.0, "rtol": 0.0},
                "mutated_inputs.out": {"atol": 0.06, "rtol": 0.0},
            },
        },
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(reference, candidate, atol=0.0, rtol=0.0, device="cpu")

    assert result.status == "passed", result.reason
    assert result.cases[0].mutated_inputs[0].name == "input.out"
    assert all(diff.passed for diff in result.cases[0].unexpected_mutations)
    assert all(
        "workspace" not in diff.name for diff in result.cases[0].unexpected_mutations
    )


def test_metadata_minimum_cases_is_a_floor(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return x",
        candidate_body="return x",
        contract={"correctness_min_cases": 3},
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(
        reference, candidate, num_correctness_cases=1, device="cpu"
    )

    assert result.status == "passed", result.reason
    assert len(result.cases) == 3


def test_stale_tolerance_path_fails_closed(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return x",
        candidate_body="return x",
        contract={
            "correctness_tolerances": {
                "output.misspelled": {"atol": 0.0, "rtol": 0.0}
            }
        },
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(reference, candidate, device="cpu")

    assert result.status == "failed"
    assert "paths did not match" in result.cases[0].error


def test_missing_metadata_fields_keep_legacy_case_count(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return x",
        candidate_body="return x",
        contract={},
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(
        reference, candidate, num_correctness_cases=2, device="cpu"
    )

    assert result.status == "passed", result.reason
    assert len(result.cases) == 2
    assert not metadata_owns_correctness(reference)


@pytest.mark.parametrize("parameter", ["atol", "rtol"])
@pytest.mark.parametrize(
    "invalid_value", [float("nan"), float("inf"), float("-inf"), -0.1]
)
def test_invalid_global_tolerance_is_rejected(tmp_path, parameter, invalid_value):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return x",
        candidate_body="return torch.full_like(x, 1e9)",
        contract={},
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(
        reference,
        candidate,
        device="cpu",
        **{parameter: invalid_value},
    )

    assert result.status == "failed"
    assert result.reason == (
        f"correctness {parameter} must be finite and non-negative; "
        f"got {invalid_value!r}"
    )


def _fraction_case(tmp_path, candidate_body, *, fraction=0.005, l2=None, size=200):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return torch.ones_like(x)",
        candidate_body=candidate_body,
        contract={
            "correctness_tolerances": {
                "default": {
                    "atol": 0.01,
                    "rtol": 0.05,
                    "max_mismatch_fraction": fraction,
                    "max_rel_l2": l2,
                }
            }
        },
        inputs=f"[torch.zeros({size}), torch.zeros({size}), torch.zeros({size})]",
    )
    return check_correctness(reference, candidate, device="cpu")


@pytest.mark.parametrize("size,passed,allowed", [(199, False, 0), (200, True, 1)])
def test_fraction_uses_exact_integer_count(tmp_path, size, passed, allowed):
    result = _fraction_case(
        tmp_path, "y=torch.ones_like(x); y[0]=1.1; return y", size=size
    )

    assert (result.status == "passed") == passed
    diff = result.cases[0].outputs[0]
    assert diff.mismatched_elements == 1
    assert diff.allowed_mismatched_elements == allowed
    assert diff.tolerance["relative_to"] == "candidate"


def test_exact_path_overrides_default_policy(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return (torch.ones_like(x), torch.ones_like(x))",
        candidate_body="return (torch.full_like(x, 1.1), torch.full_like(x, 1.1))",
        contract={
            "correctness_tolerances": {
                "default": {
                    "atol": 0.01,
                    "rtol": 0.0,
                    "max_mismatch_fraction": 0.0,
                },
                "output[0]": {
                    "atol": 0.01,
                    "rtol": 0.0,
                    "max_mismatch_fraction": 1.0,
                },
            }
        },
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(reference, candidate, device="cpu")

    assert result.status == "failed"
    first, second = result.cases[0].outputs
    assert first.passed
    assert not second.passed


def test_fraction_and_l2_are_both_required(tmp_path):
    result = _fraction_case(
        tmp_path, "y=torch.ones_like(x); y[0]=100; return y", l2=0.01
    )
    diff = result.cases[0].outputs[0]

    assert result.status == "failed"
    assert diff.mismatched_elements == diff.allowed_mismatched_elements == 1
    assert diff.relative_l2 > 0.01


def test_l2_cannot_replace_fraction_check(tmp_path):
    result = _fraction_case(
        tmp_path,
        "y=torch.ones_like(x); y[:2]=1.1; return y",
        size=400,
        fraction=0.0025,
        l2=0.02,
    )

    assert result.status == "failed"
    assert result.cases[0].outputs[0].relative_l2 < 0.02


def test_l2_requires_exact_match_for_zero_reference_norm(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return torch.zeros_like(x)",
        candidate_body="y=torch.zeros_like(x); y[0]=1e-15; return y",
        contract={
            "correctness_tolerances": {
                "default": {
                    "atol": 0.0,
                    "rtol": 0.0,
                    "max_mismatch_fraction": 1.0,
                    "max_rel_l2": 1.0,
                }
            }
        },
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(reference, candidate, device="cpu")

    assert result.status == "failed"
    assert result.cases[0].outputs[0].relative_l2 < 1.0


def test_zero_fraction_preserves_candidate_relative_term(tmp_path):
    result = _fraction_case(
        tmp_path, "return torch.full_like(x, 1.062)", fraction=0
    )

    assert result.status == "passed"


@pytest.mark.parametrize("value", ["float('nan')", "float('inf')"])
def test_nonfinite_is_not_a_permitted_outlier(tmp_path, value):
    result = _fraction_case(
        tmp_path, f"y=torch.ones_like(x); y[0]={value}; return y"
    )

    assert result.status == "failed"
    assert "Non-finite" in result.cases[0].outputs[0].error


def test_discrete_output_remains_exact(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return torch.ones_like(x, dtype=torch.int32)",
        candidate_body="y=torch.ones_like(x, dtype=torch.int32); y[0]=2; return y",
        contract={
            "correctness_tolerances": {
                "default": {
                    "atol": 1.0,
                    "rtol": 1.0,
                    "max_mismatch_fraction": 0.5,
                }
            }
        },
        inputs="[torch.zeros(200), torch.zeros(200), torch.zeros(200)]",
    )

    result = check_correctness(reference, candidate, device="cpu")

    assert result.status == "failed"
    assert result.cases[0].outputs[0].allowed_mismatched_elements == 0


@pytest.mark.parametrize("reference_dtype", ["torch.int64", "torch.bool"])
@pytest.mark.parametrize("candidate_dtype", ["torch.float32", "torch.complex64"])
def test_metadata_policy_rejects_continuous_candidate_for_discrete_reference(
    tmp_path, reference_dtype, candidate_dtype
):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body=f"return torch.ones_like(x, dtype={reference_dtype})",
        candidate_body=(
            f"y=torch.ones_like(x, dtype={candidate_dtype}); y[0]=2.0; return y"
        ),
        contract={
            "correctness_tolerances": {
                "default": {
                    "atol": 0.01,
                    "rtol": 0.05,
                    "max_mismatch_fraction": 0.005,
                }
            }
        },
        inputs="[torch.zeros(200), torch.zeros(200), torch.zeros(200)]",
    )

    result = check_correctness(reference, candidate, device="cpu")

    assert result.status == "failed"
    assert "integer and boolean reference outputs" in result.cases[0].outputs[0].error


def test_metadata_policy_keeps_boolean_mismatches_exact(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return torch.ones_like(x, dtype=torch.bool)",
        candidate_body=(
            "y=torch.ones_like(x, dtype=torch.bool); y[0]=False; return y"
        ),
        contract={
            "correctness_tolerances": {
                "default": {
                    "atol": 1.0,
                    "rtol": 1.0,
                    "max_mismatch_fraction": 1.0,
                }
            }
        },
        inputs="[torch.zeros(200), torch.zeros(200), torch.zeros(200)]",
    )

    result = check_correctness(reference, candidate, device="cpu")

    assert result.status == "failed"
    assert result.cases[0].outputs[0].allowed_mismatched_elements == 0


def test_metadata_policy_keeps_float_reference_integer_candidate_compatible(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return torch.ones_like(x)",
        candidate_body=(
            "y=torch.ones_like(x, dtype=torch.int64); y[0]=2; return y"
        ),
        contract={
            "correctness_tolerances": {
                "default": {
                    "atol": 0.01,
                    "rtol": 0.05,
                    "max_mismatch_fraction": 0.005,
                }
            }
        },
        inputs="[torch.zeros(200), torch.zeros(200), torch.zeros(200)]",
    )

    result = check_correctness(reference, candidate, device="cpu")

    assert result.status == "passed", result.reason
    assert result.cases[0].outputs[0].mismatched_elements == 1


def test_legacy_policy_keeps_integer_reference_float_candidate_compatible(tmp_path):
    reference, candidate = _write_problem(
        tmp_path,
        reference_body="return torch.ones_like(x, dtype=torch.int64)",
        candidate_body=(
            "y=torch.ones_like(x, dtype=torch.float32); y[0]=1.01; return y"
        ),
        contract={},
        inputs="[torch.zeros(4), torch.zeros(4), torch.zeros(4)]",
    )

    result = check_correctness(reference, candidate, device="cpu")

    assert result.status == "passed", result.reason


@pytest.mark.parametrize("fraction", [-0.1, 1.01, float("nan"), float("inf")])
def test_invalid_fraction_is_rejected(tmp_path, fraction):
    result = _fraction_case(tmp_path, "return torch.ones_like(x)", fraction=fraction)

    assert result.status == "failed"
    assert "max_mismatch_fraction" in result.reason


@pytest.mark.parametrize("bound", [-0.1, float("nan"), float("inf")])
def test_invalid_metadata_l2_is_rejected(tmp_path, bound):
    result = _fraction_case(tmp_path, "return torch.ones_like(x)", l2=bound)

    assert result.status == "failed"
    assert "max_rel_l2" in result.reason
