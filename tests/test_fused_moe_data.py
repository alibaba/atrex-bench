"""Routing invariants for the public MoE input bundle (small CPU allocations)."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_fused_moe_routes_to_distinct_experts(monkeypatch, top_k):
    spec = importlib.util.spec_from_file_location("moe_input", ROOT / "data/fused_moe/input.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    randn = torch.randn

    def cpu_randn(*args, **kwargs):
        kwargs["device"] = "cpu"
        return randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", cpu_randn)
    torch.manual_seed(0)
    data = module._make_inputs(64, 4, 4, 4, top_k)
    ids, weights = data["topk_ids"], data["topk_weights"]
    assert ids.dtype == torch.int32
    assert weights.dtype == torch.float32
    assert ids.shape == weights.shape == (64, top_k)
    assert (ids >= 0).all() and (ids < 4).all()
    assert all(len(set(row.tolist())) == top_k for row in ids)
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(64))


def test_fused_moe_baselines_cover_shapes_without_private_fields():
    root = ROOT / "data/fused_moe"
    meta = json.loads((root / "metadata.json").read_text())
    shapes = json.loads((root / "shapes.json").read_text())
    assert meta["shapes"].keys() == shapes.keys()
    assert len(shapes) == 23
    for row in meta["shapes"].values():
        baseline = row["production_performance"]["XPU-A"]
        assert baseline["framework"] == "aiter"
        assert baseline["performance_us"] > 0
        assert not {"source_model", "source_trace", "trace_id"}.intersection(row)
    assert meta["shapes"]["0"]["production_performance"]["XPU-A"]["performance_us"] == 232.72
