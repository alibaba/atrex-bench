"""Evaluation pipeline: compile -> correctness -> performance."""

from atrex_bench.eval.compile import CompileResult, check_compilation
from atrex_bench.eval.correctness import (
    CorrectnessCase,
    CorrectnessShapeResult,
    OutputDiff,
    check_correctness,
)
from atrex_bench.eval.estimate import EstimateResult, estimate
from atrex_bench.eval.flops import FlopEstimateResult, estimate_theoretical_flops
from atrex_bench.eval.performance import (
    KernelTimingEvent,
    PerformanceSample,
    PerformanceShapeResult,
    benchmark_performance,
    benchmark_reference_torch_compile,
)
from atrex_bench.eval.kernel_attribution import (
    FlydslComputeRatioShape,
    FlydslComputeRatioSummary,
    KernelAttribution,
    compute_flydsl_compute_ratio_for_shape,
    summarize_flydsl_compute_ratio,
)
from atrex_bench.eval.roofline import (
    RooflineHardware,
    RooflineResult,
    compute_roofline,
    compute_roofline_hybrid,
    load_hardware,
)

__all__ = [
    "CompileResult",
    "CorrectnessCase",
    "CorrectnessShapeResult",
    "EstimateResult",
    "FlopEstimateResult",
    "FlydslComputeRatioShape",
    "FlydslComputeRatioSummary",
    "KernelAttribution",
    "KernelTimingEvent",
    "OutputDiff",
    "PerformanceSample",
    "PerformanceShapeResult",
    "RooflineHardware",
    "RooflineResult",
    "benchmark_performance",
    "benchmark_reference_torch_compile",
    "check_compilation",
    "check_correctness",
    "compute_flydsl_compute_ratio_for_shape",
    "compute_roofline",
    "compute_roofline_hybrid",
    "estimate",
    "estimate_theoretical_flops",
    "load_hardware",
    "summarize_flydsl_compute_ratio",
]
