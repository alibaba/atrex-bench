"""Static cache validation must use the same floor and the matching device latency."""

from dataclasses import replace

import pytest

from atrex_bench.eval.roofline import RooflineHardware
from scripts import static_roofline_check as check


@pytest.mark.parametrize("work", [{}, {"bf16": 0}, {"bf16": 1}])
@pytest.mark.parametrize("q", [0, 1, 100])
def test_static_floor_agrees_with_work(work, q):
    hw = RooflineHardware("test", "test", "test", "test", {"bf16_tc": 1000}, 100, "")
    block = {"semantic_W_flops": work, "semantic_Q_read_bytes": q, "semantic_Q_write_bytes": 0}
    theory = check._compute_static_roofline(shape_id="x", shape_block=block, hw=hw)
    result = check._compute_static_roofline(
        shape_id="x", shape_block=block, hw=replace(hw, launch_overhead_s=0.02)
    )
    assert result.sol_time_s == (max(theory.sol_time_s, 0.02) if theory.sol_time_s else 0)


def test_production_latency_never_falls_back_to_another_device():
    meta = {
        "shapes": {
            "0": {
                "production_performance": {
                    "A": {"performance_us": 100},
                    "B": {"performance_us": 200},
                }
            }
        }
    }
    assert check._production_perf_us(meta, "0", "A") == 100
    assert check._production_perf_us(meta, "0", "B") == 200
    assert check._production_perf_us(meta, "0", "C") is None
