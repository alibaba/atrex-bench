"""Tests for theoretical FLOPs estimation from reference modules."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_reference(tmp_path: Path, content: str) -> Path:
    reference_path = tmp_path / "reference.py"
    reference_path.write_text(content.strip() + "\n", encoding="utf-8")
    return reference_path


def test_estimate_theoretical_flops_counts_supported_ops(tmp_path: Path) -> None:
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
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

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    assert result.complete is True
    assert result.flops_complete is True
    assert result.bytes_complete is True
    assert result.total_flops == 1088
    assert result.total_bytes == 1408
    assert result.total_read_bytes == 896
    assert result.total_write_bytes == 512
    assert result.total_read_bytes + result.total_write_bytes == result.total_bytes
    assert result.arithmetic_intensity == 1088 / 1408
    assert result.counted_ops["aten.mm"] == 1024
    assert result.counted_ops["aten.add"] == 64
    assert result.counted_byte_ops["aten.mm"] == 896
    assert result.counted_byte_ops["aten.add"] == 512
    assert result.read_byte_ops["aten.mm"] == 640
    assert result.read_byte_ops["aten.add"] == 256
    assert result.write_byte_ops["aten.mm"] == 256
    assert result.write_byte_ops["aten.add"] == 256
    assert result.zero_flop_ops == {}
    assert result.uncounted_ops == []
    assert result.uncounted_op_invocations == {}
    assert result.zero_byte_ops == {}
    assert result.inputs["args"][0]["shape"] == [4, 8]
    assert result.inputs["args"][1]["shape"] == [8, 16]


def test_estimate_theoretical_flops_reports_uncounted_ops(tmp_path: Path) -> None:
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 4)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    assert result.complete is False
    assert result.flops_complete is False
    assert result.bytes_complete is True
    assert result.total_flops == 0
    assert result.total_bytes == 128
    assert result.arithmetic_intensity == 0.0
    assert "aten.sin" in result.uncounted_ops
    assert result.uncounted_op_invocations["aten.sin"] == 1
    assert result.heuristic_byte_op_invocations["aten.sin"] == 1
    assert result.counted_byte_ops["aten.sin"] == 128
    assert result.error is None


def test_estimate_theoretical_flops_strict_mode_fails_on_uncounted_ops(
    tmp_path: Path,
) -> None:
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(2, 2)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(
        reference_path,
        device="cpu",
        strict=True,
    )

    assert result.passed is False
    assert result.complete is False
    assert result.flops_complete is False
    assert result.total_flops == 0
    assert result.error is not None
    assert "aten.sin" in result.error


def test_estimate_theoretical_flops_counts_inplace_add_and_zero_flop_control_ops(
    tmp_path: Path,
) -> None:
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x == 0
        out = torch.zeros_like(x)
        if mask.any():
            out.add_(x + 1.0)
        return out


def get_inputs() -> list[torch.Tensor]:
    return [torch.zeros(2, 4)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    assert result.complete is True
    assert result.flops_complete is True
    assert result.bytes_complete is True
    assert result.total_flops == 16
    assert result.total_bytes == 242
    assert result.arithmetic_intensity == 16 / 242
    assert result.counted_ops["aten.add"] == 8
    assert result.counted_ops["aten.add_"] == 8
    assert result.counted_byte_ops["aten.eq"] == 40
    assert result.counted_byte_ops["aten.zeros_like"] == 32
    assert result.counted_byte_ops["aten.any"] == 9
    assert result.counted_byte_ops["aten._local_scalar_dense"] == 1
    assert result.counted_byte_ops["aten.add"] == 64
    assert result.counted_byte_ops["aten.add_"] == 96
    # zeros_like is write-only: zero read bytes, output-sized write bytes.
    assert "aten.zeros_like" not in result.read_byte_ops
    assert result.write_byte_ops["aten.zeros_like"] == 32
    # _local_scalar_dense is scalar-read: only reads, no write.
    assert result.read_byte_ops["aten._local_scalar_dense"] == 1
    assert "aten._local_scalar_dense" not in result.write_byte_ops
    # generic IO: read = bytes(args), write = bytes(out).
    assert result.read_byte_ops["aten.eq"] == 32
    assert result.write_byte_ops["aten.eq"] == 8
    assert result.read_byte_ops["aten.any"] == 8
    assert result.write_byte_ops["aten.any"] == 1
    assert result.read_byte_ops["aten.add"] == 32
    assert result.write_byte_ops["aten.add"] == 32
    # In-place add reads dst+src and writes dst, so read bytes exceed write bytes.
    assert result.read_byte_ops["aten.add_"] == 64
    assert result.write_byte_ops["aten.add_"] == 32
    assert result.total_read_bytes == 32 + 0 + 8 + 1 + 32 + 64
    assert result.total_write_bytes == 8 + 32 + 1 + 0 + 32 + 32
    assert result.total_read_bytes + result.total_write_bytes == result.total_bytes
    assert result.zero_flop_ops["aten._local_scalar_dense"] == 1
    assert result.zero_flop_ops["aten.any"] == 1
    assert result.zero_flop_ops["aten.eq"] == 1
    assert result.zero_byte_ops == {}
    assert result.heuristic_byte_ops == []
    assert result.uncounted_ops == []
    assert result.uncounted_op_invocations == {}


def test_estimate_theoretical_flops_treats_abs_as_zero_flop(
    tmp_path: Path,
) -> None:
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


def get_inputs() -> list[torch.Tensor]:
    return [torch.ones(2, 3)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu", strict=True)

    assert result.passed is True
    assert result.flops_complete is True
    assert result.bytes_complete is True
    assert result.uncounted_ops == []
    assert result.uncounted_op_invocations == {}
    assert result.heuristic_byte_ops == []
    assert result.heuristic_byte_op_invocations == {}
    assert result.zero_flop_ops.get("aten.abs") == 1
    assert result.counted_byte_ops.get("aten.abs") == 48
    assert result.read_byte_ops.get("aten.abs") == 24
    assert result.write_byte_ops.get("aten.abs") == 24
    assert result.total_flops == 0
    assert result.total_bytes == 48
    assert result.total_read_bytes == 24
    assert result.total_write_bytes == 24


def test_estimate_flops_cli_writes_json(tmp_path: Path) -> None:
    reference_path = _write_reference(
        tmp_path,
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
    output_path = tmp_path / "flops.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pythonpath_parts = [str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/estimate_flops.py",
            "--reference",
            str(reference_path),
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["operator_name"] == tmp_path.name
    assert payload["theoretical_flops"] == 1088
    assert payload["theoretical_data_movement_bytes"] == 1408
    assert payload["theoretical_read_bytes"] == 896
    assert payload["theoretical_write_bytes"] == 512
    assert payload["arithmetic_intensity_flops_per_byte"] == 1088 / 1408
    assert payload["precision"]["status"] == "estimated"
    assert payload["precision"]["flops_complete"] is True
    assert payload["precision"]["bytes_complete"] is True
    assert payload["precision"]["bytes_model"] == "execution_trace_tensor_io_estimate"
    assert "details" not in payload


def test_estimate_flops_cli_verbose_includes_details(tmp_path: Path) -> None:
    reference_path = _write_reference(
        tmp_path,
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
            "scripts/estimate_flops.py",
            "--reference",
            str(reference_path),
            "--device",
            "cpu",
            "--verbose",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert "details" in payload
    assert payload["details"]["counted_flop_ops"]["aten.mm"] == 1024
    assert payload["details"]["counted_byte_ops"]["aten.mm"] == 896
    assert payload["details"]["read_byte_ops"]["aten.mm"] == 640
    assert payload["details"]["write_byte_ops"]["aten.mm"] == 256


# ----- per-dtype W bucketing ---------------------------------------


def test_estimate_theoretical_flops_buckets_single_dtype_bf16(tmp_path: Path) -> None:
    """A pure bf16 mm should put 100% of FLOPs under the 'bf16' key."""
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [
        torch.randn(64, 128, dtype=torch.bfloat16),
        torch.randn(128, 96, dtype=torch.bfloat16),
    ]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    assert result.total_flops == 2 * 64 * 128 * 96
    assert result.flops_by_dtype == {"bf16": 2 * 64 * 128 * 96}
    assert sum(result.flops_by_dtype.values()) == result.total_flops


def test_estimate_theoretical_flops_buckets_mixed_dtype_separately(
    tmp_path: Path,
) -> None:
    """Separate fp32 mm and bf16 mm should land in distinct buckets."""
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        out_fp32 = torch.matmul(a, b)
        out_bf16 = torch.matmul(c, d)
        return out_fp32.float() + out_bf16.float()


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [
        torch.randn(8, 16, dtype=torch.float32),
        torch.randn(16, 32, dtype=torch.float32),
        torch.randn(8, 16, dtype=torch.bfloat16),
        torch.randn(16, 32, dtype=torch.bfloat16),
    ]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    fp32_mm = 2 * 8 * 16 * 32
    bf16_mm = 2 * 8 * 16 * 32
    fp32_add = 8 * 32  # final add() counts 1 FLOP per output element
    # bf16 bucket: just the bf16 mm.
    assert result.flops_by_dtype.get("bf16") == bf16_mm
    # fp32 bucket: fp32 mm + the final add (its inputs are fp32 after .float()).
    assert result.flops_by_dtype.get("fp32") == fp32_mm + fp32_add
    # buckets cover total_flops exactly.
    assert sum(result.flops_by_dtype.values()) == result.total_flops


def test_estimate_theoretical_flops_excludes_integer_ops_from_dtype_buckets(
    tmp_path: Path,
) -> None:
    """Integer ops contribute zero FLOPs and must not appear in flops_by_dtype."""
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
    ) -> torch.Tensor:
        # bf16 mm contributes FLOPs; int add contributes zero (FlopCounterMode
        # default registry has no formula for integer-only paths and our
        # dispatcher would tag them under no-floating-tensor anyway).
        _ = idx_a + idx_b
        return torch.matmul(a, b)


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [
        torch.randn(16, 32, dtype=torch.bfloat16),
        torch.randn(32, 64, dtype=torch.bfloat16),
        torch.arange(8, dtype=torch.int32),
        torch.arange(8, dtype=torch.int32),
    ]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    assert result.flops_by_dtype == {"bf16": 2 * 16 * 32 * 64}
    # Integer keys (int32/int64) MUST NOT be present.
    assert all(
        not k.startswith("int") for k in result.flops_by_dtype
    ), result.flops_by_dtype


def test_estimate_theoretical_flops_respects_shape_id(tmp_path: Path) -> None:
    """Passing a non-zero shape_id should select that shape's W."""
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, m: int, k: int, n: int) -> None:
        super().__init__()
        self.m, self.k, self.n = m, k, n

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
""",
    )
    (tmp_path / "shapes.json").write_text(
        json.dumps(
            {
                "0": {
                    "description": "tiny",
                    "init_kwargs": {"m": 4, "k": 8, "n": 16},
                    "input_kwargs": {"m": 4, "k": 8, "n": 16},
                },
                "1": {
                    "description": "small",
                    "init_kwargs": {"m": 16, "k": 32, "n": 64},
                    "input_kwargs": {"m": 16, "k": 32, "n": 64},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "input.py").write_text(
        """
import torch


def _make_inputs(m: int, k: int, n: int, **_) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "a": torch.randn(m, k, dtype=torch.bfloat16),
        "b": torch.randn(k, n, dtype=torch.bfloat16),
    }
""",
        encoding="utf-8",
    )

    r0 = estimate_theoretical_flops(reference_path, device="cpu", shape_id="0")
    r1 = estimate_theoretical_flops(reference_path, device="cpu", shape_id="1")

    assert r0.passed and r1.passed
    assert r0.total_flops == 2 * 4 * 8 * 16
    assert r1.total_flops == 2 * 16 * 32 * 64
    assert r0.flops_by_dtype == {"bf16": 2 * 4 * 8 * 16}
    assert r1.flops_by_dtype == {"bf16": 2 * 16 * 32 * 64}


def test_estimate_theoretical_flops_layer_norm_with_affine(tmp_path: Path) -> None:
    """``F.layer_norm`` with weight + bias yields 7 FLOPs per input element.

    See ``_native_layer_norm_flop`` for the per-element accounting:
    ``5 * N`` without affine + ``2 * N`` for the gamma/beta scale-shift.
    Without this formula, ``aten.native_layer_norm`` would land in
    ``uncounted_ops`` and total_flops would be 0 -- which is wrong because
    layer_norm DOES have real floating-point arithmetic.
    """
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias)


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 8), torch.randn(8), torch.randn(8)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    assert result.flops_complete is True
    assert result.total_flops == 7 * (4 * 8)
    assert result.counted_ops["aten.native_layer_norm"] == 7 * (4 * 8)
    assert "aten.native_layer_norm" not in result.uncounted_op_invocations


def test_estimate_theoretical_flops_layer_norm_without_affine(tmp_path: Path) -> None:
    """Drop weight/bias and the per-element factor falls from 7 to 5.

    Detected via the not-None test on the registered formula's keyword
    arguments; protects against silently double-counting affine when
    ``weight=None``.
    """
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],))


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 8)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    assert result.flops_complete is True
    assert result.total_flops == 5 * (4 * 8)
    assert result.counted_ops["aten.native_layer_norm"] == 5 * (4 * 8)


def test_estimate_theoretical_flops_topk_and_masked_fill_are_zero_flop(
    tmp_path: Path,
) -> None:
    """``topk`` (partial sort) and ``masked_fill`` (selection) carry no FP arith.

    By the FLOP convention (only counts FP multiplies / adds / divs /
    transcendentals) both ops are genuinely 0 FLOPs. Listing them
    explicitly in zero_flop_packets means the counter classifies them as
    ``zero_flop_ops`` instead of leaving them in ``uncounted_ops`` -- which
    matters for ``flops_complete`` and downstream reward-hacking checks
    that rely on the "every op is accounted for" signal.
    """
    from atrex_bench.eval.flops import estimate_theoretical_flops

    reference_path = _write_reference(
        tmp_path,
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, top_k: int = 5) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        threshold = torch.topk(logits, self.top_k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4, 16)]


def get_init_inputs() -> list:
    return []
""",
    )

    result = estimate_theoretical_flops(reference_path, device="cpu")

    assert result.passed is True
    assert result.flops_complete is True
    assert result.total_flops == 0
    assert "aten.topk" in result.zero_flop_ops
    assert "aten.masked_fill" in result.zero_flop_ops
    assert "aten.topk" not in result.uncounted_op_invocations
    assert "aten.masked_fill" not in result.uncounted_op_invocations
