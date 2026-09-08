"""Regression coverage for eager None, mutation and legacy output contracts."""

import json

import pytest
import torch

from atrex_bench.cli.run_eval import _correctness_from_payload, _correctness_to_payload
from atrex_bench.eval._runtime import ModelInputs, clone_model_inputs
from atrex_bench.eval.correctness import (
    _check_unchanged,
    _compare_value_trees,
    check_correctness,
)


def evaluate(
    tmp_path,
    candidate,
    *,
    reference="out.copy_(x + 1); return None",
    mutations=("out",),
    inputs="[torch.arange(8.), torch.zeros(8)]",
):
    common = "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x, out):\n        "
    (tmp_path / "reference.py").write_text(
        common
        + reference
        + "\n\ndef get_inputs():\n    return "
        + inputs
        + "\n\ndef get_init_inputs():\n    return []\n"
    )
    (tmp_path / "solution.py").write_text(common + candidate + "\n")
    metadata = (
        {} if mutations is None else {"benchmark_contract": {"mutates_inputs": list(mutations)}}
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return check_correctness(tmp_path / "reference.py", tmp_path / "solution.py", device="cpu")


def test_none_inplace_pass(tmp_path):
    result = evaluate(tmp_path, "out.copy_(x + 1); return None")
    assert result.status == "passed", result.reason
    case = result.cases[0]
    assert case.outputs[0].name == "out"
    assert case.outputs[0].passed
    assert case.mutated_inputs[0].name == "input.out"
    assert all(diff.passed for diff in case.mutated_inputs + case.unexpected_mutations)


def test_mutation_diagnostics_survive_json_worker_round_trip(tmp_path):
    result = evaluate(tmp_path, "out[:4].copy_(x[:4] + 1); x.add_(1); return None")
    payload = json.loads(json.dumps(_correctness_to_payload(result)))
    restored = _correctness_from_payload(payload)
    assert restored == result
    assert any(not diff.passed for diff in restored.cases[0].mutated_inputs)
    assert any(not diff.passed for diff in restored.cases[0].unexpected_mutations)


@pytest.mark.parametrize("candidate", ["return None", "out[:4].copy_(x[:4] + 1); return None"])
def test_missing_or_partial_mutation_fails(tmp_path, candidate):
    result = evaluate(tmp_path, candidate)
    assert result.status == "failed"
    assert any(not diff.passed for diff in result.cases[0].mutated_inputs)


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_undeclared_write_detected_on_both_sides(tmp_path, role):
    bodies = {
        "reference": "out.copy_(x + 1); return None",
        "candidate": "out.copy_(x + 1); return None",
    }
    bodies[role] = "out.copy_(x + 1); x.add_(1); return None"
    result = evaluate(tmp_path, bodies["candidate"], reference=bodies["reference"])
    assert result.status == "failed"
    assert any(
        not diff.passed and diff.name == f"{role}.input.x"
        for diff in result.cases[0].unexpected_mutations
    )


def test_unknown_mutation_name_fails_before_invocation(tmp_path):
    result = evaluate(tmp_path, "raise AssertionError('must not run')", mutations=("missing",))
    assert result.status == "failed"
    assert "Unknown mutated input names" in result.cases[0].error
    assert "must not run" not in result.cases[0].error


def test_clone_preserves_aliases_across_args_and_kwargs():
    storage = torch.arange(32.0)
    x, y = storage[3:15:2], storage[5:17:2]
    original = ModelInputs((x,), {"nested": {"y": [y]}})
    cloned = clone_model_inputs(original)
    other = clone_model_inputs(original)
    a, b = cloned.args[0], cloned.kwargs["nested"]["y"][0]
    assert a.stride() == x.stride()
    assert a.storage_offset() == 3
    assert b.storage_offset() == 5
    assert a.untyped_storage()._cdata == b.untyped_storage()._cdata
    assert a.untyped_storage()._cdata != other.args[0].untyped_storage()._cdata
    a[1] = 999
    assert b[0].item() == 999
    assert storage[5].item() == other.args[0][1].item() == 5


def test_unchanged_check_includes_hidden_storage_tail_and_nan_bytes():
    original = torch.tensor([float("nan"), 1.0, 2.0, 3.0])
    copied = original.clone()
    assert all(d.passed for d in _check_unchanged(original[:2], copied[:2], name="x"))
    copied[3] = 9
    assert not all(d.passed for d in _check_unchanged(original[:2], copied[:2], name="x"))


@pytest.mark.parametrize(
    "before,after",
    [
        ({"x": [None, 1]}, {"x": [None, 2]}),
        ({"x": None}, {"y": None}),
        ([None], [None, None]),
        ((1,), [1]),
        (None, 0),
        (torch.arange(6.0).reshape(2, 3), torch.arange(6.0).reshape(3, 2)),
        (torch.ones(2, 2), torch.ones(2, 2).T),
        (torch.ones(2), torch.ones(2).double()),
    ],
)
def test_unchanged_nested_and_metadata_mismatches(before, after):
    assert not all(d.passed for d in _check_unchanged(before, after, name="input"))


def test_unchanged_nested_values_pass():
    value = {"x": [None, (True, 1, torch.arange(3.0))]}
    cloned = clone_model_inputs(ModelInputs((value,), {})).args[0]
    assert all(d.passed for d in _check_unchanged(value, cloned, name="input"))


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "left,right",
    [
        (None, 0),
        ({"x": None}, {"y": None}),
        ([None], [None, None]),
        ({"x": None}, [None]),
        (torch.ones(2), torch.ones(3)),
    ],
)
def test_value_tree_structure_mismatches(left, right, strict):
    diffs = _compare_value_trees(left, right, name="out", atol=0, rtol=0, strict_types=strict)
    assert not all(d.passed for d in diffs)
    assert any(d.error for d in diffs)


@pytest.mark.parametrize("strict", [False, True])
def test_nested_none_leaves_pass(strict):
    value = {"a": (None, [torch.ones(2), 1, True])}
    diffs = _compare_value_trees(value, value, name="out", atol=0, rtol=0, strict_types=strict)
    assert all(d.passed for d in diffs)
    assert [d.name for d in diffs] == ["out.a[0]", "out.a[1][0]", "out.a[1][1]", "out.a[1][2]"]


@pytest.mark.parametrize(
    "reference,candidate",
    [
        ("return (x, out)", "return [x, out]"),
        ("return True", "return 1"),
        ("return x", "return x.double()"),
        ("return {'a': (None, True)}", "return {'a': [None, 1]}"),
    ],
)
@pytest.mark.parametrize("mutations", [None, ()])
def test_legacy_compatibility_and_explicit_strict_contract(
    tmp_path, reference, candidate, mutations
):
    result = evaluate(tmp_path, candidate, reference=reference, mutations=mutations)
    assert result.status == ("passed" if mutations is None else "failed"), result.reason


@pytest.mark.parametrize("mutations", [None, ()])
def test_scalar_tolerance_is_retained(tmp_path, mutations):
    result = evaluate(tmp_path, "return 1.001", reference="return 1.0", mutations=mutations)
    assert result.status == "passed", result.reason


@pytest.mark.parametrize("declare_alias", [False, True])
def test_mutation_declaration_covers_shared_storage_aliases(tmp_path, declare_alias):
    result = evaluate(
        tmp_path,
        "out.add_(1); return None",
        reference="out.add_(1); return None",
        mutations=("x", "out") if declare_alias else ("out",),
        inputs="(lambda pool: [pool[:4], pool[4:]])(torch.arange(8.))",
    )
    assert result.status == ("passed" if declare_alias else "failed"), result.reason
    if not declare_alias:
        assert {d.name for d in result.cases[0].unexpected_mutations if not d.passed} == {
            "reference.input.x",
            "candidate.input.x",
        }
