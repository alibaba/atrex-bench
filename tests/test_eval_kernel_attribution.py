"""Unit tests for the flydsl kernel-attribution stage.

Tests the pure-logic parts of ``atrex_bench.eval.kernel_attribution``:
classifying kernel events against the runtime-observed flydsl symbol set,
computing the per-shape ratio, and aggregating across shapes.

The runtime tracker itself is tested in ``test_eval_flydsl_tracker.py``;
end-to-end coverage of the sub-worker wiring lives in
``test_eval_performance.py`` and ``test_run_eval.py``.
"""
from __future__ import annotations

from pathlib import Path

from atrex_bench.eval.kernel_attribution import (
    compute_flydsl_compute_ratio_for_shape,
    summarize_flydsl_compute_ratio,
)
from atrex_bench.eval.performance import (
    KernelTimingEvent,
    PerformanceSample,
    PerformanceShapeResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perf(
    events: list[KernelTimingEvent],
    *,
    observed_kernels: dict[str, list[str]] | None = None,
    e2e_ms: float = 1.0,
) -> PerformanceShapeResult:
    """Build a perf result that looks like a healthy benched shape.

    Default ``e2e_ms=1.0`` (= 1000us wall) lets event ``device_time_us``
    values be set numerically so that ``event_time / 1000us`` gives a
    convenient ratio (e.g. 500us flydsl / 1000us e2e = 0.5).
    """
    return PerformanceShapeResult(
        samples=[PerformanceSample(end_to_end_time_ms=e2e_ms)],
        kernel_events=events,
        observed_kernels=observed_kernels,
    )


def _candidate(tmp_path: Path) -> Path:
    """Build any candidate path -- contents are irrelevant after the rewrite."""
    p = tmp_path / "candidate.py"
    p.write_text("# kernel attribution no longer reads candidate source\n")
    return p


# ---------------------------------------------------------------------------
# Non-flydsl + early-exit paths (don't need observed_kernels)
# ---------------------------------------------------------------------------


def test_ratio_non_flydsl_is_zero_with_error(tmp_path: Path) -> None:
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path),
        _perf([KernelTimingEvent("foo", 100.0, 1)]),
        dsl="triton",
    )
    assert result.ratio == 0.0
    assert result.error is not None and "non-flydsl" in result.error


def test_ratio_skipped_when_perf_did_not_run(tmp_path: Path) -> None:
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path),
        PerformanceShapeResult(samples=[], kernel_events=[]),
        dsl="flydsl",
    )
    assert result.ratio is None
    assert result.error == "skipped: performance stage did not run"


def test_ratio_error_when_no_device_events_captured(tmp_path: Path) -> None:
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path),
        PerformanceShapeResult(
            samples=[PerformanceSample(end_to_end_time_ms=0.5)],
            kernel_events=[],
        ),
        dsl="flydsl",
    )
    assert result.ratio is None
    assert result.error == "no device kernel events captured"


def test_ratio_error_when_observed_kernels_missing(tmp_path: Path) -> None:
    """If observed_kernels is None we refuse to guess -- surface a clear error.

    The previous AST-based fallback silently under-counted flydsl when
    candidates used spellings the AST couldn't see. Forcing the runtime
    tracker to be present is the whole point of the rewrite, so we don't
    re-introduce a "guess from source" path on the None branch.
    """
    perf = PerformanceShapeResult(
        samples=[PerformanceSample(end_to_end_time_ms=1.0)],
        kernel_events=[KernelTimingEvent("foo", 100.0, 1)],
        observed_kernels=None,
    )
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path), perf, dsl="flydsl"
    )
    assert result.ratio is None
    assert result.error is not None
    assert "observed_kernels" in result.error
    assert "_flydsl_tracker" in result.error


# ---------------------------------------------------------------------------
# Classification using the runtime-observed symbol payload
# ---------------------------------------------------------------------------


def test_ratio_pure_flydsl_is_one_via_exact_name(tmp_path: Path) -> None:
    perf = _perf(
        [KernelTimingEvent("my_op", 1000.0, 10)],
        observed_kernels={"exact_names": ["my_op"], "func_names": []},
    )
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path), perf, dsl="flydsl"
    )
    assert result.ratio == 1.0
    assert result.flydsl_device_time_us == 1000.0
    assert result.total_e2e_time_us == 1000.0
    assert result.error is None
    assert len(result.kernel_breakdown) == 1
    assert result.kernel_breakdown[0].is_flydsl is True


def test_ratio_pure_flydsl_via_func_pattern(tmp_path: Path) -> None:
    """``func_names: ['_moe_kernel']`` matches ``_moe_kernel_<digits>``.
    500us flydsl over 1000us e2e wall -> ratio 0.5."""
    perf = _perf(
        [KernelTimingEvent("_moe_kernel_5", 500.0, 3)],
        observed_kernels={"exact_names": [], "func_names": ["_moe_kernel"]},
    )
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path), perf, dsl="flydsl"
    )
    assert result.ratio == 0.5
    assert result.kernel_breakdown[0].is_flydsl is True


def test_ratio_mixed_flydsl_and_torch(tmp_path: Path) -> None:
    """Mixed events. Total kernel device time 6000us across a 10000us e2e
    wall (10ms) -> flydsl ratio 2000/10000 = 0.2 (NOT 2000/6000 -- the
    denominator is e2e, not sum of kernel events).
    """
    events = [
        KernelTimingEvent("gdn_gate_o", 2000.0, 64),          # flydsl exact
        KernelTimingEvent("Cijk_Alik_Bljk_S_...", 3000.0, 64),  # rocBLAS GEMM
        KernelTimingEvent("void at::native::elementwise_kernel<...>", 1000.0, 64),
    ]
    perf = _perf(
        events,
        observed_kernels={"exact_names": ["gdn_gate_o"], "func_names": []},
        e2e_ms=10.0,  # 10000us wall
    )
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path), perf, dsl="flydsl"
    )
    assert abs(result.ratio - (2000.0 / 10000.0)) < 1e-9
    assert result.flydsl_device_time_us == 2000.0
    assert result.total_e2e_time_us == 10000.0
    # Breakdown is sorted by device_time desc.
    assert [k.name for k in result.kernel_breakdown] == [
        "Cijk_Alik_Bljk_S_...",
        "gdn_gate_o",
        "void at::native::elementwise_kernel<...>",
    ]
    flags = {k.name: k.is_flydsl for k in result.kernel_breakdown}
    assert flags["gdn_gate_o"] is True
    assert flags["Cijk_Alik_Bljk_S_..."] is False
    assert flags["void at::native::elementwise_kernel<...>"] is False


def test_ratio_empty_observation_yields_zero(tmp_path: Path) -> None:
    """Tracker installed but no decorations observed: trust it, ratio=0.

    The runtime tracker is authoritative; if no @kernel ever fired, the
    candidate isn't using flydsl for any device work even if it imports
    flydsl (e.g. only for utility types). The bench events all count as
    non-flydsl.
    """
    perf = _perf(
        [KernelTimingEvent("some_torch_kernel", 800.0, 4)],
        observed_kernels={"exact_names": [], "func_names": []},
    )
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path), perf, dsl="flydsl"
    )
    assert result.ratio == 0.0
    assert result.kernel_breakdown[0].is_flydsl is False


def test_ratio_multiple_observed_func_names(tmp_path: Path) -> None:
    events = [
        KernelTimingEvent("k_a_0", 100.0, 1),
        KernelTimingEvent("k_b_2", 200.0, 1),
        KernelTimingEvent("torch_relu", 700.0, 1),
    ]
    perf = _perf(
        events,
        observed_kernels={"exact_names": [], "func_names": ["k_a", "k_b"]},
    )
    result = compute_flydsl_compute_ratio_for_shape(
        _candidate(tmp_path), perf, dsl="flydsl"
    )
    # 300 / 1000 = 0.3
    assert abs(result.ratio - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# summarize_flydsl_compute_ratio
# ---------------------------------------------------------------------------


def test_summary_unweighted_mean_of_valid_shapes(tmp_path: Path) -> None:
    """e2e wall = 1ms = 1000us (default _perf helper).
    shape 0: 100us flydsl -> ratio 0.1
    shape 1:  50us flydsl -> ratio 0.05
    average = 0.075
    """
    per_shape = {
        "0": compute_flydsl_compute_ratio_for_shape(
            _candidate(tmp_path),
            _perf(
                [KernelTimingEvent("k_0", 100.0, 1)],
                observed_kernels={"exact_names": [], "func_names": ["k"]},
            ),
            dsl="flydsl",
        ),
        "1": compute_flydsl_compute_ratio_for_shape(
            _candidate(tmp_path),
            _perf(
                [
                    KernelTimingEvent("k_0", 50.0, 1),
                    KernelTimingEvent("other", 50.0, 1),
                ],
                observed_kernels={"exact_names": [], "func_names": ["k"]},
            ),
            dsl="flydsl",
        ),
    }
    summary = summarize_flydsl_compute_ratio(per_shape)
    assert summary.valid_shape_count == 2
    assert summary.total_shape_count == 2
    assert abs(summary.average - 0.075) < 1e-9


def test_summary_all_non_flydsl_yields_zero_average(tmp_path: Path) -> None:
    per_shape = {
        sid: compute_flydsl_compute_ratio_for_shape(
            _candidate(tmp_path),
            _perf([KernelTimingEvent("x", 1.0, 1)]),
            dsl="triton",
        )
        for sid in ("0", "1")
    }
    summary = summarize_flydsl_compute_ratio(per_shape)
    assert summary.average == 0.0
    assert summary.valid_shape_count == 0
    assert summary.total_shape_count == 2


def test_summary_all_skipped_yields_none(tmp_path: Path) -> None:
    per_shape = {
        sid: compute_flydsl_compute_ratio_for_shape(
            _candidate(tmp_path),
            PerformanceShapeResult(samples=[], kernel_events=[]),
            dsl="flydsl",
        )
        for sid in ("0", "1")
    }
    summary = summarize_flydsl_compute_ratio(per_shape)
    assert summary.average is None
    assert summary.valid_shape_count == 0
