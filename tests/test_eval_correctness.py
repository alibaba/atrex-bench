"""Tests for Stage 1: correctness verification."""

from pathlib import Path

import torch

from atrex_bench.eval.correctness import check_correctness

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "references" / "atrex_001" / "reference.py"
CANDIDATE_PATH = Path(__file__).parent / "fixtures" / "generations" / "atrex_001.py"


def _write_python_file(tmp_path: Path, name: str, content: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _all_outputs_passed(case) -> bool:
    return all(diff.passed for diff in case.outputs)


def _passed_case_count(result) -> int:
    return sum(
        1
        for case in result.cases
        if case.error is None and case.outputs and _all_outputs_passed(case)
    )


def test_correct_output_passes_with_multiple_cases() -> None:
    result = check_correctness(
        REFERENCE_PATH,
        CANDIDATE_PATH,
        num_correctness_cases=2,
        rtol=0.05,
        device="cpu",
    )
    assert result.status == "passed"
    assert result.reason is None
    assert len(result.cases) == 2
    assert _passed_case_count(result) == 2
    for case in result.cases:
        assert case.error is None
        for diff in case.outputs:
            assert diff.passed is True
            assert diff.max_elementwise_abs_diff == 0.0


def test_relative_tolerance_is_configurable(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        return x + 1.0",
                "",
                "def get_inputs():",
                "    return [torch.zeros(4, 4)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        return (x + 1.0) * 1.03",
            ]
        ),
    )

    loose_result = check_correctness(
        reference_path,
        candidate_path,
        rtol=0.05,
        device="cpu",
    )
    strict_result = check_correctness(
        reference_path,
        candidate_path,
        rtol=0.01,
        device="cpu",
    )

    assert loose_result.status == "passed"
    assert strict_result.status == "failed"
    strict_max_rel = strict_result.cases[0].outputs[0].max_elementwise_rel_diff
    assert strict_max_rel is not None
    assert strict_max_rel > 0.01


def test_runtime_error_fails(tmp_path: Path) -> None:
    candidate_path = _write_python_file(
        tmp_path,
        "runtime_error.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        raise RuntimeError('intentional failure')",
            ]
        ),
    )

    result = check_correctness(
        REFERENCE_PATH,
        candidate_path,
        device="cpu",
    )

    assert result.status == "failed"
    assert len(result.cases) == 1
    assert _passed_case_count(result) == 0
    assert result.cases[0].error is not None
    assert "intentional failure" in result.cases[0].error


def test_correctness_uses_reference_inputs_and_init_inputs(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self, bias):",
                "        super().__init__()",
                "        self.bias = bias",
                "",
                "    def forward(self, x):",
                "        return x + self.bias",
                "",
                "def get_inputs():",
                "    return [torch.zeros(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return [1.5]",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self, bias):",
                "        super().__init__()",
                "        self.bias = bias",
                "",
                "    def forward(self, x):",
                "        return x + self.bias",
                "",
                "def get_inputs():",
                "    return [torch.full((2, 2), 7.0)]",
                "",
                "def get_init_inputs():",
                "    return [9.0]",
            ]
        ),
    )
    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
    )

    # Input artifacts are now seed-only (no .pt files written).
    assert result.status == "passed"
    artifact = result.cases[0].input_artifact
    assert artifact is not None
    assert artifact["format"] == "manual_seed"
    assert isinstance(artifact["seed"], int) and artifact["seed"] >= 0
    assert "path" not in artifact, "tensor checkpoint should NOT be written"


def test_dict_output_passes_with_by_key_pairing(tmp_path: Path) -> None:
    """Reference and candidate both return dict[str, Tensor].

    Comparison must pair tensors by key, not by dict insertion order.
    """
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return {'quantized': x * 2.0, 'scales': x + 1.0}",
                "",
                "def get_inputs():",
                "    return [torch.ones(3, 3)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    # Candidate intentionally builds the dict in REVERSE insertion order
    # to prove pairing is by-key (alphabetical via flatten_outputs) and
    # not by Python dict iteration order.
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        scales = x + 1.0",
                "        quantized = x * 2.0",
                "        return {'scales': scales, 'quantized': quantized}",
            ]
        ),
    )

    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
    )
    assert result.status == "passed", (
        f"unexpected failure: {result.reason or result.cases[0].error}"
    )
    assert result.cases[0].error is None
    output_names = sorted(diff.name for diff in result.cases[0].outputs)
    assert output_names == ["quantized", "scales"]
    for diff in result.cases[0].outputs:
        assert diff.passed is True
        assert diff.max_elementwise_abs_diff == 0.0


def test_dict_reference_vs_tuple_candidate_reports_structure_mismatch(tmp_path: Path) -> None:
    """When reference returns dict but candidate returns tuple, fail with a clear mismatch error."""
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return {'a': x * 2.0, 'b': x + 1.0}",
                "",
                "def get_inputs():",
                "    return [torch.ones(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate_tuple.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return (x * 2.0, x + 1.0)",
            ]
        ),
    )

    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
    )
    assert result.status == "failed"
    assert _passed_case_count(result) == 0
    case_error = result.cases[0].error or ""
    assert "Output structure mismatch" in case_error
    assert "dict" in case_error
    assert "tuple" in case_error
    # Detailed structural diagnostics should mention the mismatched sides clearly
    assert "reference" in case_error.lower()
    assert "candidate" in case_error.lower()


def test_dict_outputs_with_mismatched_keys_reports_key_mismatch(tmp_path: Path) -> None:
    """When both sides return dict but with different keys, fail with a key-set diff."""
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return {'quantized': x * 2.0, 'scales': x + 1.0}",
                "",
                "def get_inputs():",
                "    return [torch.ones(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate_wrong_keys.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return {'q': x * 2.0, 's': x + 1.0}",
            ]
        ),
    )

    result = check_correctness(
        reference_path,
        candidate_path,
        device="cpu",
    )
    assert result.status == "failed"
    case_error = result.cases[0].error or ""
    assert "Output structure mismatch" in case_error
    assert "dict keys differ" in case_error
    assert "missing in candidate" in case_error
    assert "extra in candidate" in case_error
