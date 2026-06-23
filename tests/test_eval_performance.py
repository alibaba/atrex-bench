"""Tests for Stage 2: performance profiling."""

from pathlib import Path

import pytest
import torch

from atrex_bench.eval.performance import (
    benchmark_performance,
    benchmark_reference_torch_compile,
)

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "references" / "atrex_001" / "reference.py"
CANDIDATE_PATH = Path(__file__).parent / "fixtures" / "generations" / "atrex_001.py"

# Performance benchmarking requires torch.cuda or torch.hip for do_bench timing
_gpu_available = torch.cuda.is_available()


def _write_python_file(tmp_path: Path, name: str, content: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.mark.skipif(not _gpu_available, reason="requires CUDA/HIP GPU")
def test_benchmark_returns_timing() -> None:
    result = benchmark_performance(
        CANDIDATE_PATH,
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=3,
        device="cpu",
    )
    assert result.error is None
    assert len(result.samples) >= 1
    for sample in result.samples:
        assert sample.end_to_end_time_ms is not None
        assert sample.end_to_end_time_ms > 0


@pytest.mark.skipif(not _gpu_available, reason="requires CUDA/HIP GPU")
def test_benchmark_reference_torch_compile_returns_timing(monkeypatch) -> None:
    compiled_models = []

    def fake_compile(model):
        compiled_models.append(model)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)

    result = benchmark_reference_torch_compile(
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=2,
        device="cpu",
    )

    assert result.error is None
    assert compiled_models
    assert len(result.samples) >= 1
    for sample in result.samples:
        assert sample.end_to_end_time_ms is not None
        assert sample.end_to_end_time_ms > 0


@pytest.mark.skipif(not _gpu_available, reason="requires CUDA/HIP GPU")
def test_benchmark_reference_torch_compile_writes_seed_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch, "compile", lambda model: model)

    result = benchmark_reference_torch_compile(
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=1,
        device="cpu",
    )

    # Input artifact is now seed-only (no .pt file written).
    assert result.error is None
    assert result.input_artifact is not None
    assert result.input_artifact["format"] == "manual_seed"
    assert isinstance(result.input_artifact["seed"], int)
    assert "path" not in result.input_artifact


@pytest.mark.skipif(not _gpu_available, reason="requires CUDA/HIP GPU")
def test_benchmark_writes_seed_artifact(tmp_path: Path) -> None:
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
    result = benchmark_performance(
        candidate_path,
        reference_path,
        warmup_iters=1,
        bench_iters=1,
        device="cpu",
    )

    # Input artifact is now seed-only (no .pt file written).
    assert result.error is None
    assert result.input_artifact is not None
    assert result.input_artifact["format"] == "manual_seed"
    assert isinstance(result.input_artifact["seed"], int)
    assert "path" not in result.input_artifact


def test_benchmark_error_handled(tmp_path: Path) -> None:
    candidate_path = _write_python_file(
        tmp_path,
        "broken_candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "        raise RuntimeError('init failure')",
                "",
                "    def forward(self, x):",
                "        return x",
            ]
        ),
    )

    result = benchmark_performance(
        candidate_path,
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=1,
        device="cpu",
    )
    assert result.samples == []
    assert result.error is not None
    assert "init failure" in result.error
