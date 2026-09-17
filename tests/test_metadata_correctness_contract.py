"""Regression tests for metadata-owned correctness policies."""

import json

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
