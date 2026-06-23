"""Roofline (speed-of-light) analytical calculator.

Computes per-(operator, shape, SKU) roofline upper bounds from the operator's
theoretical work / data movement (``W`` FLOPs, ``Q`` bytes) and the SKU's
vendor-published peaks (``P_peak[dtype]``, ``B_peak.hbm``). Does *not* invoke
any kernel — the inputs are pure spec values plus the operator's algorithmic
estimates.

Single-precision case (one dtype):

    AI         = W / Q
    P_roof     = min(P_peak, AI * B_peak)
    SOL_time_s = W / P_roof  (== max(W / P_peak, Q / B_peak))

Hybrid (multi-dtype) case (e.g. fp8 GEMM + bf16 epilogue):

    T_compute_min = sum(W[d] / P_peak[d] for d in W)
    T_mem_min     = (Q_read + Q_write) / B_peak
    SOL_time_s    = max(T_compute_min, T_mem_min)

Reference: Williams, Waterman, Patterson 2009, "Roofline: An Insightful
Visual Performance Model for Multicore Architectures" (CACM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

# Map short dtype names (used in metadata.json / roofline.json semantic_W_flops
# keys) to the path keys used in configs/hardware/<sku>.yaml.p_peak so a single
# dtype name resolves consistently across the project.
DTYPE_PATHS: dict[str, str] = {
    "bf16": "bf16_tc",
    "fp16": "fp16_tc",
    "fp32": "fp32_strict",
    "fp32_strict": "fp32_strict",
    "fp32_tf32": "fp32_tf32_tc",
    "fp8_e4m3": "fp8_e4m3_tc",
    "fp8_e5m2": "fp8_e5m2_tc",
}

Bottleneck = Literal["compute", "memory", "balanced", "no_compute"]


@dataclass(frozen=True)
class RooflineHardware:
    """Hardware spec needed for roofline analysis.

    Loaded from configs/hardware/<sku>.yaml in the unified spec-only schema
    (commit 54d5bb8). ``p_peak`` is keyed by the path string (e.g. ``bf16_tc``)
    not the short dtype name.
    """

    sku_name: str
    sku_stem: str
    arch: str
    vendor: str
    p_peak: dict[str, int]  # path string -> FLOPs/s
    b_peak_hbm: int  # bytes/s
    source_doc: str


@dataclass(frozen=True)
class RooflineResult:
    """One roofline computation result for a (W, Q, dtype, hardware) tuple."""

    arithmetic_intensity: float  # W / Q (FLOPs per byte); inf when Q == 0
    ridge_point_ai: float  # P_peak / B_peak (machine balance)
    p_roof_flops_per_s: float  # min(P_peak, AI * B_peak)
    sol_time_s: float  # W / P_roof; 0.0 when W == 0
    sol_time_ms: float  # SOL in milliseconds
    bottleneck: Bottleneck
    p_peak_used: int
    b_peak_used: int


_RIDGE_BAND = 0.05  # +/-5% around ridge AI counts as balanced


def resolve_dtype_path(dtype: str) -> str:
    """Return the configs/hardware/<sku>.yaml.p_peak key for ``dtype``.

    Accepts either the short name ('bf16') or the path string ('bf16_tc') and
    returns the path string. Raises ``ValueError`` for unknown dtypes so a
    typo cannot silently fall through to a wrong P_peak lookup.
    """

    if dtype in DTYPE_PATHS:
        return DTYPE_PATHS[dtype]
    # Already a path? Accept identity if the value appears in DTYPE_PATHS.
    if dtype in DTYPE_PATHS.values():
        return dtype
    known = sorted(set(list(DTYPE_PATHS) + list(DTYPE_PATHS.values())))
    raise ValueError(
        f"Unknown dtype: {dtype!r}. Known dtype names / paths: {known}."
    )


def load_hardware(yaml_path: Path) -> RooflineHardware:
    """Load a SKU profile from ``configs/hardware/<sku>.yaml``.

    Validates that ``p_peak`` and ``b_peak.hbm`` contain real (non-null)
    integer values; placeholder profiles (e.g. XPU-A.yaml with ``null``
    values awaiting user fill-in) raise ``ValueError`` so partial profiles
    cannot silently produce wrong SOLs.
    """

    if not yaml_path.exists():
        raise FileNotFoundError(f"Hardware config not found: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Hardware config {yaml_path} must be a YAML mapping at top-level."
        )

    hw = raw.get("hardware") or {}
    p_peak_raw = raw.get("p_peak") or {}
    b_peak_raw = raw.get("b_peak") or {}
    source = raw.get("source") or {}

    sku_name = hw.get("name") or yaml_path.stem
    arch = hw.get("arch") or ""
    vendor = hw.get("vendor") or ""

    if not isinstance(p_peak_raw, dict) or not p_peak_raw:
        raise ValueError(
            f"{yaml_path}: 'p_peak' must be a non-empty mapping of "
            f"<dtype_path> -> FLOPs/s integers."
        )

    cleaned_p_peak: dict[str, int] = {}
    null_paths: list[str] = []
    for path_key, value in p_peak_raw.items():
        if value is None:
            null_paths.append(path_key)
            continue
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"{yaml_path}: p_peak[{path_key!r}] must be a positive number "
                f"(FLOPs/s), got {value!r}."
            )
        cleaned_p_peak[path_key] = int(value)

    if null_paths:
        raise ValueError(
            f"{yaml_path}: p_peak contains null placeholder values for "
            f"{null_paths}. Fill in the SKU spec before running roofline "
            f"calculations against it."
        )

    b_peak_hbm = b_peak_raw.get("hbm")
    if b_peak_hbm is None:
        raise ValueError(
            f"{yaml_path}: b_peak.hbm is null. Fill in the SKU spec before "
            f"running roofline calculations against it."
        )
    if not isinstance(b_peak_hbm, (int, float)) or b_peak_hbm <= 0:
        raise ValueError(
            f"{yaml_path}: b_peak.hbm must be a positive number (bytes/s), "
            f"got {b_peak_hbm!r}."
        )
    b_peak_hbm_int = int(b_peak_hbm)

    source_doc = ""
    if isinstance(source, dict):
        source_doc = str(source.get("vendor_doc") or "")

    return RooflineHardware(
        sku_name=str(sku_name),
        sku_stem=yaml_path.stem.lower(),
        arch=str(arch),
        vendor=str(vendor),
        p_peak=cleaned_p_peak,
        b_peak_hbm=b_peak_hbm_int,
        source_doc=source_doc,
    )


def _classify_bottleneck(ai: float, ridge_ai: float) -> Bottleneck:
    if ai == 0.0:
        return "no_compute"
    if math.isinf(ai):
        return "compute"
    lo, hi = ridge_ai * (1.0 - _RIDGE_BAND), ridge_ai * (1.0 + _RIDGE_BAND)
    if ai < lo:
        return "memory"
    if ai > hi:
        return "compute"
    return "balanced"


def compute_roofline(
    w_flops: int,
    q_bytes: int,
    dtype: str,
    hardware: RooflineHardware,
) -> RooflineResult:
    """Single-dtype roofline: combine W, Q, P_peak[dtype], B_peak into a SOL.

    Hybrid (multi-dtype) workloads should use ``compute_roofline_hybrid``
    instead so each dtype's compute path is summed correctly.
    """

    if w_flops < 0 or q_bytes < 0:
        raise ValueError(
            f"w_flops and q_bytes must be non-negative; "
            f"got w_flops={w_flops}, q_bytes={q_bytes}."
        )

    path = resolve_dtype_path(dtype)
    if path not in hardware.p_peak:
        raise KeyError(
            f"Hardware {hardware.sku_name!r} has no p_peak entry for "
            f"dtype path {path!r} (resolved from {dtype!r}). "
            f"Available paths: {sorted(hardware.p_peak)}."
        )

    p_peak = hardware.p_peak[path]
    b_peak = hardware.b_peak_hbm

    if q_bytes == 0:
        ai: float = math.inf
        p_roof: float = float(p_peak)
    else:
        ai = w_flops / q_bytes
        p_roof = min(float(p_peak), ai * float(b_peak))

    ridge_ai = float(p_peak) / float(b_peak)
    bottleneck = _classify_bottleneck(ai, ridge_ai)

    if w_flops == 0:
        # Two distinct W=0 sub-cases:
        #   (a) Q > 0: a real op whose ATen calls have no FlopCounterMode
        #       formula (e.g. layer_norm via aten.native_layer_norm) or that
        #       genuinely does no FP arithmetic but still moves bytes (e.g.
        #       reshape_and_cache). The op is memory-bound; SOL = Q / B_peak.
        #   (b) Q == 0: degenerate workload (no compute AND no memory traffic).
        #       Preserve the no_compute label and SOL = 0 for downstream
        #       consumers that special-case it.
        if q_bytes > 0:
            sol_time_s = q_bytes / b_peak
            bottleneck = "memory"
        else:
            sol_time_s = 0.0
            bottleneck = "no_compute"
    else:
        sol_time_s = w_flops / p_roof

    return RooflineResult(
        arithmetic_intensity=ai,
        ridge_point_ai=ridge_ai,
        p_roof_flops_per_s=p_roof,
        sol_time_s=sol_time_s,
        sol_time_ms=sol_time_s * 1000.0,
        bottleneck=bottleneck,
        p_peak_used=p_peak,
        b_peak_used=b_peak,
    )


def compute_roofline_hybrid(
    w_flops_by_dtype: dict[str, int],
    q_bytes: int,
    hardware: RooflineHardware,
) -> RooflineResult:
    """Multi-dtype roofline: T_compute = sum(W[d] / P_peak[d]).

    Use this when an operator's ``semantic_W_flops`` has more than one dtype
    key (e.g. fp8 GEMM body + bf16 epilogue). For single-dtype operators
    pass ``compute_roofline`` directly — it's a simpler interface.
    """

    if not w_flops_by_dtype:
        raise ValueError("w_flops_by_dtype must be a non-empty mapping.")
    if q_bytes < 0:
        raise ValueError(f"q_bytes must be non-negative, got {q_bytes}.")

    paths_used: dict[str, int] = {}
    t_compute_s = 0.0
    total_w = 0
    for dtype, w in w_flops_by_dtype.items():
        if w < 0:
            raise ValueError(
                f"semantic_W_flops[{dtype!r}] must be non-negative, got {w}."
            )
        path = resolve_dtype_path(dtype)
        if path not in hardware.p_peak:
            raise KeyError(
                f"Hardware {hardware.sku_name!r} has no p_peak entry for "
                f"dtype path {path!r} (resolved from {dtype!r}). "
                f"Available paths: {sorted(hardware.p_peak)}."
            )
        paths_used[path] = hardware.p_peak[path]
        if hardware.p_peak[path] > 0:
            t_compute_s += w / hardware.p_peak[path]
        total_w += w

    b_peak = hardware.b_peak_hbm
    if q_bytes == 0:
        ai: float = math.inf
    else:
        ai = total_w / q_bytes
    t_mem_s = q_bytes / b_peak if q_bytes > 0 else 0.0
    sol_time_s = max(t_compute_s, t_mem_s)

    # P_roof here is the *effective* peak — total work over the SOL time —
    # which falls below the dominant single-dtype P_peak for hybrid kernels.
    if sol_time_s > 0.0:
        p_roof = total_w / sol_time_s
    else:
        p_roof = 0.0

    # Ridge AI is dominated by the largest p_peak in the mix; report the max
    # so callers can compare AI on the same scale across hybrid / single.
    dominant_p_peak = max(paths_used.values()) if paths_used else 0
    ridge_ai = dominant_p_peak / b_peak if b_peak > 0 else math.inf
    bottleneck: Bottleneck
    if total_w == 0:
        # Same W=0 split as compute_roofline: Q>0 → memory-bound, Q=0 →
        # degenerate no_compute. The hybrid SOL math (max(T_compute, T_mem))
        # already produces the right number for the Q>0 case, only the label
        # needs to flip from no_compute → memory.
        bottleneck = "memory" if q_bytes > 0 else "no_compute"
    elif t_compute_s > t_mem_s * (1.0 + _RIDGE_BAND):
        bottleneck = "compute"
    elif t_mem_s > t_compute_s * (1.0 + _RIDGE_BAND):
        bottleneck = "memory"
    else:
        bottleneck = "balanced"

    return RooflineResult(
        arithmetic_intensity=ai,
        ridge_point_ai=ridge_ai,
        p_roof_flops_per_s=p_roof,
        sol_time_s=sol_time_s,
        sol_time_ms=sol_time_s * 1000.0,
        bottleneck=bottleneck,
        p_peak_used=dominant_p_peak,
        b_peak_used=b_peak,
    )


__all__ = [
    "DTYPE_PATHS",
    "Bottleneck",
    "RooflineHardware",
    "RooflineResult",
    "compute_roofline",
    "compute_roofline_hybrid",
    "load_hardware",
    "resolve_dtype_path",
]
