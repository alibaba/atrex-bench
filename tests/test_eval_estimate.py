"""Tests for the unified estimate modes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_module(tmp_path: Path, name: str, content: str) -> Path:
    module_path = tmp_path / name
    module_path.write_text(content.strip() + "\n", encoding="utf-8")
    return module_path


def test_estimate_w_theoretical_mode(tmp_path: Path) -> None:
    from atrex_bench.eval.estimate import estimate

    module_path = _write_module(
        tmp_path,
        "reference.py",
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b) + 1.0


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 8), torch.randn(8, 16)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate(
        mode="W_theoretical",
        module_path=module_path,
        device="cpu",
        strict=True,
    )

    assert result.passed is True
    assert result.mode == "W_theoretical"
    assert result.value == 1088
    assert result.units == "FLOPs"
    assert result.precision["status"] == "exact"


def test_estimate_q_semantic_lower_bound_mode(tmp_path: Path) -> None:
    from atrex_bench.eval.estimate import estimate

    module_path = _write_module(
        tmp_path,
        "reference.py",
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b) + 1.0


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 8), torch.randn(8, 16)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate(
        mode="Q_semantic_lower_bound",
        module_path=module_path,
        device="cpu",
    )

    assert result.passed is True
    assert result.mode == "Q_semantic_lower_bound"
    assert result.value == 896
    assert result.units == "bytes"
    assert result.precision["status"] == "exact"
    assert result.components["input_bytes"] == 640
    assert result.components["state_bytes"] == 0
    assert result.components["output_bytes"] == 256
    assert result.components["read_bytes"] == 640
    assert result.components["write_bytes"] == 256
    assert result.components["read_bytes"] + result.components["write_bytes"] == result.value


def test_estimate_q_semantic_lower_bound_counts_module_state(tmp_path: Path) -> None:
    from atrex_bench.eval.estimate import estimate

    module_path = _write_module(
        tmp_path,
        "stateful_reference.py",
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 8)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate(
        mode="Q_semantic_lower_bound",
        module_path=module_path,
        device="cpu",
    )

    assert result.passed is True
    assert result.value == 896
    assert result.components["input_bytes"] == 128
    assert result.components["state_bytes"] == 512
    assert result.components["output_bytes"] == 256
    assert result.components["read_bytes"] == 640
    assert result.components["write_bytes"] == 256
    assert result.components["read_bytes"] + result.components["write_bytes"] == result.value
    assert result.precision["state_accounting"] == "all_registered_parameters_and_buffers"


def test_estimate_q_profiled_impl_op_trace_mode(tmp_path: Path) -> None:
    from atrex_bench.eval.estimate import estimate

    module_path = _write_module(
        tmp_path,
        "modelnew.py",
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b) + 1.0


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 8), torch.randn(8, 16)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate(
        mode="Q_profiled_impl",
        module_path=module_path,
        device="cpu",
        profile_backend="op_trace_estimate",
    )

    assert result.passed is True
    assert result.value == 1408
    assert result.units == "bytes"
    assert result.precision["status"] == "estimated"
    assert result.precision["backend"] == "op_trace_estimate"
    assert result.components["arithmetic_intensity_flops_per_byte"] == 1088 / 1408
    assert result.components["read_bytes"] == 896
    assert result.components["write_bytes"] == 512
    assert result.components["read_bytes"] + result.components["write_bytes"] == result.value


def test_estimate_q_profiled_impl_rejects_opaque_kernels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import importlib

    estimate_module = importlib.import_module("atrex_bench.eval.estimate")

    module_path = _write_module(
        tmp_path,
        "modelnew.py",
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_inputs() -> list[torch.Tensor]:
    return [torch.randn(4, 4)]


def get_init_inputs() -> list:
    return []
""",
    )
    monkeypatch.setattr(estimate_module, "infer_target_dsl", lambda _path: "triton")

    result = estimate_module.estimate(
        mode="Q_profiled_impl",
        module_path=module_path,
        device="cpu",
    )

    assert result.passed is False
    assert result.precision["status"] == "unsupported"
    assert "opaque custom kernels" in result.error


def test_estimate_cli_q_semantic_lower_bound(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        "reference.py",
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b) + 1.0


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 8), torch.randn(8, 16)]


def get_init_inputs() -> list:
    return []
""",
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pythonpath_parts = [str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/estimate.py",
            "--mode",
            "Q_semantic_lower_bound",
            "--module",
            str(module_path),
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "Q_semantic_lower_bound"
    assert payload["value"] == 896
    assert payload["units"] == "bytes"
    assert payload["precision"]["status"] == "exact"
