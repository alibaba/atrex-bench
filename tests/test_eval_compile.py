"""Tests for Stage 0: Compilation check."""

from pathlib import Path

from atrex_bench.eval.compile import check_compilation

VALID_CODE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)
"""

CODE_MISSING_MODEL = """
import torch
import torch.nn as nn

class AnotherModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
"""

CODE_SYNTAX_ERROR = """
def broken(
    # missing closing paren and colon
"""


def _write_source(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def test_valid_code_passes(tmp_path):
    result = check_compilation(_write_source(tmp_path, "candidate.py", VALID_CODE))
    assert result.status == "passed"
    assert result.reason is None


def test_missing_model_fails(tmp_path):
    result = check_compilation(_write_source(tmp_path, "candidate.py", CODE_MISSING_MODEL))
    assert result.status == "failed"
    assert "Model" in result.reason


def test_syntax_error_fails(tmp_path):
    result = check_compilation(_write_source(tmp_path, "candidate.py", CODE_SYNTAX_ERROR))
    assert result.status == "failed"
    assert result.reason is not None
