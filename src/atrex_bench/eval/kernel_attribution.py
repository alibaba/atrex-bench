"""Stage 3: classify per-kernel device time as flydsl vs. non-flydsl.

We need to know, for every GPU kernel symbol that ran during the bench,
whether it came from flydsl's ``@kernel`` decorator (i.e. is candidate
work) or from elsewhere (rocBLAS GEMM, PyTorch elementwise, ...).

**Source of truth: the runtime decorator tracker** (``_flydsl_tracker``).

It monkey-patches ``flydsl.compiler.kernel_function.kernel`` BEFORE the
candidate is imported in the per-shape sub-worker, then records every
actual decoration. The recorded payload is then attached to
``PerformanceShapeResult.observed_kernels`` and consumed here.

Why not parse the candidate source? Because different AI-generated
candidates spell ``@kernel`` differently (bare ``@kernel``, parametric
``@kernel(name=...)``, ``from ... import kernel as _alias`` + ``@_alias``,
attribute-form ``@flydsl.compiler.kernel``, kernel-source generators that
build the decoration as a string literal then ``exec`` it, ...). A
source-text scanner has to enumerate all of these and inevitably misses
one for some unseen candidate -- the resulting silent under-count is
indistinguishable from real reward-hacking. The runtime patch sees the
single ground-truth call site -- flydsl's ``kernel()`` constructor --
regardless of how the candidate's source spelled the decoration.

Emitted GPU symbol per flydsl source (see ``flydsl/compiler/kernel_function.py``)
is either:

* the explicit ``name=...`` keyword argument on the decorator, used verbatim, or
* ``f"{func.__name__}_{kernel_id}"`` where ``kernel_id`` is a non-negative integer.

We tag each ``KernelTimingEvent`` as ``is_flydsl`` iff its name matches one of
the observed exact names or one of the ``<func_name>_<id>`` regex patterns.

The ratio is unweighted across shapes per request: each shape contributes its
own ``ratio`` to the top-level average and shapes are weighted equally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from atrex_bench.eval._flydsl_tracker import symbols_from_serialized
from atrex_bench.eval.performance import KernelTimingEvent, PerformanceShapeResult


@dataclass(frozen=True)
class KernelAttribution:
    """One classified kernel event."""

    name: str
    device_time_us: float
    calls: int
    is_flydsl: bool


@dataclass(frozen=True)
class FlydslComputeRatioShape:
    """Per-shape flydsl compute attribution result.

    All time fields are per single ``model.forward()`` call (microseconds).

    Fields:
      * ``ratio`` -- ``flydsl_device_time_us / total_e2e_time_us``. Equals
        the fraction of candidate end-to-end wall time spent in flydsl
        ``@kernel`` GPU work.
      * ``flydsl_device_time_us`` -- per-forward sum of GPU device time
        across kernels registered through flydsl's ``@kernel`` (runtime
        tracker observation; profiler-measured).
      * ``total_e2e_time_us`` -- per-forward end-to-end wall time, taken
        from ``samples[0].end_to_end_time_ms`` (do_bench timing). Same
        denominator used everywhere else for timing comparisons, so
        ``ratio`` is comparable across candidates and shapes.
      * ``kernel_breakdown`` -- per-kernel ``KernelAttribution`` rows sorted
        by descending per-forward device time; useful for "why is the ratio
        this number" debugging.
    """

    ratio: float | None = None
    flydsl_device_time_us: float = 0.0
    total_e2e_time_us: float = 0.0
    kernel_breakdown: list[KernelAttribution] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class FlydslComputeRatioSummary:
    """Top-level aggregate across all shapes."""

    average: float | None
    valid_shape_count: int
    total_shape_count: int
    shapes: dict[str, FlydslComputeRatioShape]


def _is_flydsl_event(
    event_name: str,
    exact_names: set[str],
    func_patterns: list[re.Pattern[str]],
) -> bool:
    if event_name in exact_names:
        return True
    for pattern in func_patterns:
        if pattern.fullmatch(event_name):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-shape attribution
# ---------------------------------------------------------------------------


def compute_flydsl_compute_ratio_for_shape(
    candidate_path: Path,
    performance_result: PerformanceShapeResult,
    *,
    dsl: str,
) -> FlydslComputeRatioShape:
    """Classify ``performance_result.kernel_events`` and compute the per-shape ratio.

    Returns an error-tagged result (with ``ratio=0.0``) for non-flydsl
    candidates per the runner's policy. A missing event list when the perf
    stage *did* run is a hard ``error`` (the denominator can't be zero in
    a healthy run).

    The set of flydsl-attributed symbols comes from the runtime decorator
    tracker (``performance_result.observed_kernels``); see the module
    docstring for the rationale. ``candidate_path`` is accepted only to
    keep the signature backward-compatible with the previous AST-based
    implementation and surface a clearer error when something upstream
    forgot to install the tracker.
    """
    del candidate_path  # not consumed any more; kept in signature for compat

    if dsl != "flydsl":
        return FlydslComputeRatioShape(
            ratio=0.0,
            error=f"non-flydsl candidate: dsl={dsl}",
        )

    if not performance_result.samples:
        return FlydslComputeRatioShape(
            ratio=None,
            error="skipped: performance stage did not run",
        )

    if not performance_result.kernel_events:
        return FlydslComputeRatioShape(
            ratio=None,
            error="no device kernel events captured",
        )

    runtime_symbols = symbols_from_serialized(performance_result.observed_kernels)
    if runtime_symbols is None:
        # Tracker payload absent. We refuse to silently fall back to
        # source-text heuristics -- those were the whole reason for the
        # rewrite -- and surface the missing-instrumentation error so the
        # caller can decide whether to re-run.
        return FlydslComputeRatioShape(
            ratio=None,
            error=(
                "missing observed_kernels payload: the runtime flydsl @kernel "
                "tracker was not installed before the candidate was imported. "
                "Re-run with an atrex_bench build that installs "
                "atrex_bench.eval._flydsl_tracker in the per-shape sub-worker."
            ),
        )

    exact_names, func_patterns = runtime_symbols

    # Denominator: per-forward end-to-end wall time from do_bench (the only
    # accurate timing source). Numerator: per-forward sum of flydsl kernels'
    # device time (kernel_events are normalised to per-forward averages in
    # _measure_runner_samples, so direct sum is per-forward).
    # Ratio thus = "fraction of candidate wall time spent inside flydsl
    # @kernel GPU work", which is the reward-hacking signal we care about.
    e2e_ms = performance_result.samples[0].end_to_end_time_ms
    if e2e_ms is None or e2e_ms <= 0.0:
        return FlydslComputeRatioShape(
            ratio=None,
            error="end_to_end_time_ms missing or non-positive in performance samples",
        )
    e2e_us = float(e2e_ms) * 1000.0

    breakdown: list[KernelAttribution] = []
    flydsl_time = 0.0
    total_kernel_time = 0.0
    for event in performance_result.kernel_events:
        is_flydsl = _is_flydsl_event(event.name, exact_names, func_patterns)
        breakdown.append(
            KernelAttribution(
                name=event.name,
                device_time_us=event.device_time_us,
                calls=event.calls,
                is_flydsl=is_flydsl,
            )
        )
        total_kernel_time += event.device_time_us
        if is_flydsl:
            flydsl_time += event.device_time_us

    breakdown.sort(key=lambda k: -k.device_time_us)

    return FlydslComputeRatioShape(
        ratio=flydsl_time / e2e_us,
        flydsl_device_time_us=flydsl_time,
        total_e2e_time_us=e2e_us,
        kernel_breakdown=breakdown,
        error=None,
    )


# ---------------------------------------------------------------------------
# Summary across shapes (unweighted mean)
# ---------------------------------------------------------------------------


def summarize_flydsl_compute_ratio(
    per_shape: dict[str, FlydslComputeRatioShape],
) -> FlydslComputeRatioSummary:
    """Compute the unweighted mean across shapes whose ratio is real.

    Rules:
      * shapes with ``error is None`` and a numeric ratio contribute to the mean
      * if no shapes contributed but every shape errored with the "non-flydsl"
        marker, the overall average is 0.0 (per "non-flydsl is error + ratio 0")
      * otherwise average is None (genuinely unknown)
    """
    valid = [
        shape.ratio
        for shape in per_shape.values()
        if shape.error is None and shape.ratio is not None
    ]
    if valid:
        average: float | None = sum(valid) / len(valid)
    elif per_shape and all(
        (shape.error or "").startswith("non-flydsl") for shape in per_shape.values()
    ):
        average = 0.0
    else:
        average = None

    return FlydslComputeRatioSummary(
        average=average,
        valid_shape_count=len(valid),
        total_shape_count=len(per_shape),
        shapes=per_shape,
    )
