"""Optional latency floors must agree across API and CLI calculation paths."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from atrex_bench.eval.roofline import (
    RooflineHardware,
    compute_roofline,
    compute_roofline_hybrid,
    load_hardware,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = RooflineHardware("test", "test", "test", "test", {"bf16_tc": 1000}, 100, "")


@pytest.mark.parametrize(
    "w,q,theory", [(1, 0, 0.001), (0, 1, 0.01), (1, 1, 0.01), (100, 1, 0.1), (0, 0, 0)]
)
@pytest.mark.parametrize("floor", [None, 0.02, 0.01])
@pytest.mark.parametrize("hybrid", [False, True])
def test_floor_preserves_theory_and_empty_work(w, q, theory, floor, hybrid):
    hw = replace(SPEC, launch_overhead_s=floor)
    result = (
        compute_roofline_hybrid({"bf16": w}, q, hw)
        if hybrid
        else compute_roofline(w, q, "bf16", hw)
    )
    expected = max(theory, floor or 0) if theory else 0
    assert result.sol_time_s == pytest.approx(expected)
    assert result.sol_time_ms == pytest.approx(expected * 1000)
    assert result.clamped_by_overhead == bool(floor and 0 < theory < floor)
    if w == 0:
        assert result.bottleneck == ("memory" if q else "no_compute")


@pytest.mark.parametrize(
    "value", [0, -1, True, False, "3e-6", float("nan"), float("inf"), -float("inf")]
)
def test_invalid_floor_rejected_in_file_and_direct_api(tmp_path, value):
    with pytest.raises(ValueError, match="launch_overhead_s"):
        replace(SPEC, launch_overhead_s=value)
    path = tmp_path / "test.yaml"
    path.write_text(
        yaml.safe_dump({"p_peak": SPEC.p_peak, "b_peak": {"hbm": 100}, "launch_overhead_s": value})
    )
    with pytest.raises(ValueError, match="launch_overhead_s"):
        load_hardware(path)


@pytest.mark.parametrize("config", [{}, {"launch_overhead_s": None}, {"launch_overhead_s": 0.02}])
def test_load_optional_floor(tmp_path, config):
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump({"p_peak": SPEC.p_peak, "b_peak": {"hbm": 100}, **config}))
    assert load_hardware(path).launch_overhead_s == config.get("launch_overhead_s")


@pytest.mark.parametrize("work", [{}, {"bf16": 0}, {"bf16": 1}, {"bf16": 1, "fp16": 1}])
@pytest.mark.parametrize("q", [0, 1, 10])
def test_cli_shape_calculation_and_floor_flag(work, q):
    spec = importlib.util.spec_from_file_location("roofline_cli", ROOT / "scripts/roofline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hw = replace(SPEC, p_peak={"bf16_tc": 1000, "fp16_tc": 1000}, launch_overhead_s=0.02)
    result, *_ = module._shape_compute(
        shape_id="0",
        shape_block={
            "semantic_W_flops": work,
            "semantic_Q_read_bytes": q,
            "semantic_Q_write_bytes": 0,
        },
        hw=hw,
        explicit_dtype=None,
    )
    theory = max(sum(work.values()) / 1000, q / 100)
    assert result.sol_time_s == pytest.approx(max(theory, 0.02) if theory else 0)
    assert module._result_to_payload(result)["clamped_by_overhead"] == (0 < theory < 0.02)


def test_hybrid_sums_compute_before_applying_floor():
    hw = replace(SPEC, p_peak={"bf16_tc": 1000, "fp16_tc": 2000}, launch_overhead_s=0.02)
    result = compute_roofline_hybrid({"bf16": 10, "fp16": 40}, 1, hw)
    assert result.sol_time_s == pytest.approx(0.03)
    assert not result.clamped_by_overhead


def test_cli_no_write_does_not_modify_cache(tmp_path, monkeypatch, capsys):
    from scripts import roofline

    op = tmp_path / "op"
    op.mkdir()
    cached = {
        "shapes": {
            "0": {
                "semantic_W_flops": {},
                "semantic_Q_read_bytes": 1,
                "semantic_Q_write_bytes": 0,
                "SOL_time_ms": {"test": 99},
            }
        }
    }
    path = op / "roofline.json"
    path.write_text(json.dumps(cached))
    hardware = tmp_path / "test.yaml"
    hardware.write_text(
        yaml.safe_dump(
            {
                "hardware": {"name": "test"},
                "p_peak": SPEC.p_peak,
                "b_peak": {"hbm": 100},
                "launch_overhead_s": 0.02,
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "roofline",
            "--operator",
            str(op),
            "--hardware",
            str(hardware),
            "--no-write",
            "--format",
            "json",
        ],
    )
    assert roofline.main() == 0
    assert json.loads(path.read_text()) == cached
    assert '"clamped_by_overhead": true' in capsys.readouterr().out
