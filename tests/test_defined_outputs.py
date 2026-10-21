"""Defined output regions must not weaken the physical ABI or valid values."""

import json

import pytest

from atrex_bench.eval.correctness import check_correctness


def run(tmp_path, candidate, mask="torch.tensor([True, False, True, False])", path="output[1]"):
    reference = tmp_path / "reference.py"
    reference.write_text(
        """import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.uint8), torch.tensor([1., 0., 2., 0.])
    def valid_outputs(self, x):
        return {"""
        + repr(path)
        + ": "
        + mask
        + """}
def get_inputs():
    return [torch.tensor([3., 4.])]
def get_init_inputs():
    return []
"""
    )
    solution = tmp_path / "solution.py"
    solution.write_text(
        "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        "
        + candidate
        + "\n"
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {"benchmark_contract": {"output_mask_method": "valid_outputs", "mutates_inputs": []}}
        )
    )
    return check_correctness(reference, solution, device="cpu", atol=0.0, rtol=0.0)


def test_padding_including_nonfinite_is_ignored(tmp_path):
    r = run(
        tmp_path, 'return x.to(torch.uint8), torch.tensor([1., float("nan"), 2., float("inf")])'
    )
    assert r.status == "passed", r.reason
    assert r.cases[0].outputs[1].total_elements == 2
    assert r.cases[0].defined_outputs["output[1]"].passed
    from dataclasses import asdict

    from atrex_bench.cli.run_eval import _correctness_from_payload

    assert _correctness_from_payload(asdict(r)) == r


@pytest.mark.parametrize(
    "candidate",
    [
        "return x.to(torch.uint8), torch.tensor([9., 0., 2., 0.])",
        "return x.to(torch.uint8) + 1, torch.tensor([1., 0., 2., 0.])",
        "return x.to(torch.uint8), torch.tensor([1., 0., 2., 0.], dtype=torch.float64)",
        "return x.to(torch.uint8), torch.tensor([1., 2.])",
        'return x.to(torch.uint8), torch.tensor([float("nan"), 0., 2., 0.])',
    ],
)
def test_valid_values_and_full_abi_still_checked(tmp_path, candidate):
    assert run(tmp_path, candidate).status == "failed"


@pytest.mark.parametrize(
    "mask,path",
    [
        ("torch.ones(4)", "output[1]"),
        ("torch.zeros(4, dtype=torch.bool)", "output[1]"),
        ("torch.ones(2, dtype=torch.bool)", "output[1]"),
        ("torch.ones(4, dtype=torch.bool)", "output[8]"),
        ("None", "output[1]"),
    ],
)
def test_invalid_masks_fail_closed(tmp_path, mask, path):
    r = run(tmp_path, "return x.to(torch.uint8), torch.tensor([1., 0., 2., 0.])", mask, path)
    assert r.status == "failed"
