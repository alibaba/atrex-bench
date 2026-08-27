#!/usr/bin/env python3
"""Static checks for production roofline SOL numbers.

This intentionally does not import or execute operator input/reference modules.
It only keeps two checks:

1. ``roofline.json`` saved ``SOL_time_ms`` matches recomputation from
   ``semantic_W_flops`` + ``semantic_Q_*_bytes`` + hardware yaml.
2. ``T_SOL / TProd`` is above a configurable threshold, default ``0.95``.

Examples:
  python scripts/static_roofline_check.py --operator fused_moe --verbose
  python scripts/static_roofline_check.py --format json --output /tmp/check.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from atrex_bench.eval.roofline import (
        RooflineHardware,
        RooflineResult,
        apply_launch_overhead,
        compute_roofline,
        compute_roofline_hybrid,
        load_hardware,
    )
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from atrex_bench.eval.roofline import (  # type: ignore[no-redef]
        RooflineHardware,
        RooflineResult,
        apply_launch_overhead,
        compute_roofline,
        compute_roofline_hybrid,
        load_hardware,
    )


@dataclass
class Issue:
    severity: str
    code: str
    operator: str
    message: str
    shape_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value


def _load_hardware_profiles(hardware_dir: Path) -> dict[str, RooflineHardware]:
    profiles: dict[str, RooflineHardware] = {}
    if not hardware_dir.is_dir():
        return profiles
    for path in sorted(hardware_dir.glob("*.yaml")):
        hw = load_hardware(path)
        profiles[hw.sku_name] = hw
    return profiles


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _match_hardware_alias(alias: str, hardware_names: list[str]) -> list[str]:
    alias_norm = _normalize_token(alias)
    if not alias_norm:
        return []

    matches: list[str] = []
    for name in hardware_names:
        name_norm = _normalize_token(name)
        last_token = _normalize_token(name.split()[-1]) if name.split() else name_norm
        if alias_norm == name_norm or alias_norm == last_token or alias_norm in name_norm:
            matches.append(name)
    return matches


def _metadata_hardware_aliases(metadata: dict[str, Any]) -> list[str]:
    aliases: list[str] = []

    trace_filter = metadata.get("trace_filter")
    if isinstance(trace_filter, dict):
        gpu = trace_filter.get("kernel_context_gpu")
        if isinstance(gpu, str):
            aliases.append(gpu)
        elif isinstance(gpu, list):
            aliases.extend(item for item in gpu if isinstance(item, str))

    trace_gpus = metadata.get("trace_gpus")
    if isinstance(trace_gpus, list):
        aliases.extend(item for item in trace_gpus if isinstance(item, str))
    elif isinstance(trace_gpus, str):
        aliases.append(trace_gpus)

    return aliases


def _latency_hardware_keys(
    *,
    metadata: dict[str, Any],
    sol_keys: list[str],
    args: argparse.Namespace,
) -> list[str]:
    if args.check_all_sol_latency:
        return sorted(sol_keys)

    requested = args.latency_hardware or []
    if requested:
        matches: set[str] = set()
        for value in requested:
            if value in sol_keys:
                matches.add(value)
            else:
                matches.update(_match_hardware_alias(value, sol_keys))
        return sorted(matches)

    matches: set[str] = set()
    for alias in _metadata_hardware_aliases(metadata):
        matches.update(_match_hardware_alias(alias, sol_keys))
    # Public bundles need not retain trace provenance. Use exact baseline device
    # keys when available; per-shape lookup still requires a matching device.
    for shape in (metadata.get("shapes") or {}).values():
        if isinstance(shape, dict):
            perf = shape.get("production_performance")
            if isinstance(perf, dict):
                matches.update(set(perf).intersection(sol_keys))
    return sorted(matches)


def _production_perf_us(
    metadata: dict[str, Any],
    shape_id: str,
    hardware_name: str,
) -> float | None:
    shapes = metadata.get("shapes")
    if isinstance(shapes, dict):
        shape_meta = shapes.get(shape_id)
        if isinstance(shape_meta, dict):
            perf = shape_meta.get("production_performance")
            if isinstance(perf, dict):
                # Never compare one device's SOL with another device's latency.
                if "performance_us" not in perf:
                    perf = perf.get(hardware_name, {})
                value = perf.get("performance_us") if isinstance(perf, dict) else None
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)

    # Legacy fallback: top-level production_performance either keyed by shape id
    # or directly carrying one performance_us value.
    perf_root = metadata.get("production_performance")
    if isinstance(perf_root, dict):
        shape_perf = perf_root.get(shape_id)
        if isinstance(shape_perf, dict):
            value = shape_perf.get("performance_us")
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        value = perf_root.get("performance_us")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)

    return None


def _compute_static_roofline(
    *,
    shape_id: str,
    shape_block: dict[str, Any],
    hw: RooflineHardware,
) -> RooflineResult:
    semantic_w = shape_block.get("semantic_W_flops")
    if not isinstance(semantic_w, dict):
        raise ValueError(f"shape {shape_id}: semantic_W_flops must be a mapping.")

    w_by_dtype: dict[str, int] = {}
    for dtype, value in semantic_w.items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(
                f"shape {shape_id}: semantic_W_flops[{dtype!r}] must be non-negative numeric."
            )
        w_by_dtype[str(dtype)] = int(value)

    q_read = shape_block.get("semantic_Q_read_bytes")
    q_write = shape_block.get("semantic_Q_write_bytes")
    if not isinstance(q_read, (int, float)) or not isinstance(q_write, (int, float)):
        raise ValueError(f"shape {shape_id}: semantic_Q_*_bytes must be numeric.")
    if q_read < 0 or q_write < 0:
        raise ValueError(f"shape {shape_id}: semantic_Q_*_bytes must be non-negative.")

    q_total = int(q_read) + int(q_write)
    if not w_by_dtype:
        sol_time_s = q_total / hw.b_peak_hbm if hw.b_peak_hbm > 0 else 0.0
        sol_time_s, clamped = apply_launch_overhead(sol_time_s, hw)
        return RooflineResult(
            arithmetic_intensity=0.0,
            ridge_point_ai=0.0,
            p_roof_flops_per_s=0.0,
            sol_time_s=sol_time_s,
            sol_time_ms=sol_time_s * 1000.0,
            bottleneck="memory" if q_total > 0 else "no_compute",
            p_peak_used=0,
            b_peak_used=hw.b_peak_hbm,
            clamped_by_overhead=clamped,
        )

    if len(w_by_dtype) == 1:
        dtype, w_flops = next(iter(w_by_dtype.items()))
        return compute_roofline(w_flops, q_total, dtype, hw)
    return compute_roofline_hybrid(w_by_dtype, q_total, hw)


def _check_sol_matches_wq(
    *,
    operator: str,
    roof_shapes: dict[str, Any],
    hardware_by_name: dict[str, RooflineHardware],
    args: argparse.Namespace,
    issues: list[Issue],
) -> None:
    for shape_id, shape_block in roof_shapes.items():
        if not isinstance(shape_block, dict):
            issues.append(
                Issue(
                    severity="ERROR",
                    code="SOL_WQ_INCONSISTENT",
                    operator=operator,
                    shape_id=shape_id,
                    message="roofline shape entry must be a mapping.",
                )
            )
            continue

        sol = shape_block.get("SOL_time_ms")
        if not isinstance(sol, dict):
            issues.append(
                Issue(
                    severity="ERROR",
                    code="SOL_WQ_INCONSISTENT",
                    operator=operator,
                    shape_id=shape_id,
                    message="SOL_time_ms must be a hardware-name mapping.",
                )
            )
            continue

        for hw_name, saved_sol in sol.items():
            if saved_sol is None:
                continue
            hw = hardware_by_name.get(hw_name)
            if hw is None:
                issues.append(
                    Issue(
                        severity="ERROR",
                        code="SOL_WQ_INCONSISTENT",
                        operator=operator,
                        shape_id=shape_id,
                        message=f"SOL_time_ms hardware key {hw_name!r} has no hardware yaml.",
                    )
                )
                continue
            if not isinstance(saved_sol, (int, float)) or saved_sol < 0:
                issues.append(
                    Issue(
                        severity="ERROR",
                        code="SOL_WQ_INCONSISTENT",
                        operator=operator,
                        shape_id=shape_id,
                        message=f"SOL_time_ms[{hw_name!r}] must be non-negative numeric or null.",
                    )
                )
                continue

            try:
                expected = _compute_static_roofline(
                    shape_id=shape_id,
                    shape_block=shape_block,
                    hw=hw,
                ).sol_time_ms
            except Exception as exc:  # noqa: BLE001 - checker should report data problems
                issues.append(
                    Issue(
                        severity="ERROR",
                        code="SOL_WQ_INCONSISTENT",
                        operator=operator,
                        shape_id=shape_id,
                        message=f"could not recompute SOL_time_ms[{hw_name!r}]: {exc}",
                    )
                )
                continue

            diff = abs(float(saved_sol) - expected)
            rel = diff / max(abs(expected), 1e-30)
            if diff > args.sol_abs_tol_ms and rel > args.sol_rel_tol:
                issues.append(
                    Issue(
                        severity="ERROR",
                        code="SOL_WQ_INCONSISTENT",
                        operator=operator,
                        shape_id=shape_id,
                        message=f"SOL_time_ms[{hw_name!r}] does not match W/Q/hardware recompute.",
                        details={
                            "hardware": hw_name,
                            "saved_ms": saved_sol,
                            "expected_ms": expected,
                            "abs_diff_ms": diff,
                            "rel_diff": rel,
                        },
                    )
                )


def _check_sol_prod_ratio(
    *,
    operator: str,
    metadata: dict[str, Any],
    roof_shapes: dict[str, Any],
    args: argparse.Namespace,
    issues: list[Issue],
) -> None:
    for shape_id, shape_block in roof_shapes.items():
        if not isinstance(shape_block, dict):
            continue
        sol = shape_block.get("SOL_time_ms")
        if not isinstance(sol, dict):
            continue
        hw_keys = _latency_hardware_keys(
            metadata=metadata,
            sol_keys=[str(key) for key in sol],
            args=args,
        )
        if not hw_keys:
            continue

        for hw_name in hw_keys:
            perf_us = _production_perf_us(metadata, str(shape_id), hw_name)
            if perf_us is None:
                continue
            prod_ms = perf_us / 1000.0
            sol_ms = sol.get(hw_name)
            if not isinstance(sol_ms, (int, float)):
                continue
            ratio = float(sol_ms) / prod_ms
            if ratio > 1.0:
                issues.append(
                    Issue(
                        severity="WARN",
                        code="SOL_EXCEEDS_PRODUCTION",
                        operator=operator,
                        shape_id=shape_id,
                        message=(
                            f"S={ratio:.2%} > 100%: candidate runs faster "
                            f"than SOL floor. Check launch_overhead_s for "
                            f"{hw_name}."
                        ),
                        details={
                            "hardware": hw_name,
                            "sol_ms": sol_ms,
                            "production_ms": prod_ms,
                            "production_us": perf_us,
                            "sol_over_tprod": ratio,
                        },
                    )
                )
            elif ratio > args.sol_prod_threshold:
                issues.append(
                    Issue(
                        severity="WARN",
                        code="HIGH_SOL_PROD_RATIO",
                        operator=operator,
                        shape_id=shape_id,
                        message=(f"T_SOL/TProd={ratio:.4f} exceeds {args.sol_prod_threshold:.4f}."),
                        details={
                            "hardware": hw_name,
                            "sol_ms": sol_ms,
                            "production_ms": prod_ms,
                            "production_us": perf_us,
                            "sol_over_tprod": ratio,
                        },
                    )
                )


def _operator_dirs(data_root: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        return [data_root / name for name in selected]
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def _check_operator(
    *,
    op_dir: Path,
    hardware_by_name: dict[str, RooflineHardware],
    args: argparse.Namespace,
    issues: list[Issue],
) -> None:
    operator = op_dir.name
    required = {
        "metadata": op_dir / "metadata.json",
        "roofline": op_dir / "roofline.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        issues.append(
            Issue(
                severity="ERROR",
                code="SOL_WQ_INCONSISTENT",
                operator=operator,
                message=f"operator is missing required files: {missing}",
            )
        )
        return

    try:
        metadata = _load_json(required["metadata"])
        roofline = _load_json(required["roofline"])
    except Exception as exc:  # noqa: BLE001 - malformed data is a checker finding
        issues.append(
            Issue(
                severity="ERROR",
                code="SOL_WQ_INCONSISTENT",
                operator=operator,
                message=f"failed to load metadata/roofline json: {exc}",
            )
        )
        return

    if not isinstance(metadata, dict) or not isinstance(roofline, dict):
        issues.append(
            Issue(
                severity="ERROR",
                code="SOL_WQ_INCONSISTENT",
                operator=operator,
                message="metadata.json and roofline.json top-level values must be mappings.",
            )
        )
        return

    roof_shapes = roofline.get("shapes")
    if not isinstance(roof_shapes, dict):
        issues.append(
            Issue(
                severity="ERROR",
                code="SOL_WQ_INCONSISTENT",
                operator=operator,
                message="roofline.json top-level shapes must be a mapping.",
            )
        )
        return

    _check_sol_matches_wq(
        operator=operator,
        roof_shapes=roof_shapes,
        hardware_by_name=hardware_by_name,
        args=args,
        issues=issues,
    )
    _check_sol_prod_ratio(
        operator=operator,
        metadata=metadata,
        roof_shapes=roof_shapes,
        args=args,
        issues=issues,
    )


def _format_issue(issue: Issue, *, verbose: bool) -> str:
    location = issue.operator
    if issue.shape_id is not None:
        location += f" sid={issue.shape_id}"
    line = f"[{issue.severity}] {location} {issue.code}: {issue.message}"
    if verbose and issue.details:
        line += "\n  details: " + json.dumps(_jsonable(issue.details), sort_keys=True)
    return line


def _format_text(issues: list[Issue], *, verbose: bool, max_issues: int) -> str:
    counts = {"ERROR": 0, "WARN": 0}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1

    lines = [
        (
            "Static roofline check: "
            f"{counts.get('ERROR', 0)} error(s), "
            f"{counts.get('WARN', 0)} warning(s)."
        )
    ]
    if not issues:
        return "\n".join(lines)

    severity_rank = {"ERROR": 0, "WARN": 1}
    ordered = sorted(
        issues,
        key=lambda item: (
            severity_rank.get(item.severity, 99),
            item.operator,
            item.shape_id or "",
            item.code,
        ),
    )
    shown = ordered if max_issues <= 0 else ordered[:max_issues]
    lines.append("")
    lines.extend(_format_issue(issue, verbose=verbose) for issue in shown)
    if max_issues > 0 and len(ordered) > max_issues:
        lines.append(f"... {len(ordered) - max_issues} more issue(s) omitted.")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=repo_root / "data")
    parser.add_argument("--hardware-dir", type=Path, default=repo_root / "configs" / "hardware")
    parser.add_argument(
        "--operator",
        action="append",
        dest="operators",
        help="Operator directory name under --data-root. Repeat to check multiple operators.",
    )
    parser.add_argument(
        "--latency-hardware",
        action="append",
        help=(
            "Hardware name or alias for T_SOL/TProd checks. Defaults to "
            "metadata.trace_gpus / metadata.trace_filter inference."
        ),
    )
    parser.add_argument(
        "--check-all-sol-latency",
        action="store_true",
        help="Compare production latency against every SOL_time_ms hardware key.",
    )
    parser.add_argument("--sol-prod-threshold", type=float, default=0.95)
    parser.add_argument("--sol-abs-tol-ms", type=float, default=1e-9)
    parser.add_argument("--sol-rel-tol", type=float, default=1e-6)
    parser.add_argument("--warnings-as-errors", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-issues", type=int, default=120, help="0 means no limit.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    hardware_by_name = _load_hardware_profiles(args.hardware_dir)
    issues: list[Issue] = []
    if not hardware_by_name:
        issues.append(
            Issue(
                severity="ERROR",
                code="SOL_WQ_INCONSISTENT",
                operator="(global)",
                message=f"no hardware profiles found under {args.hardware_dir}",
            )
        )

    op_dirs = _operator_dirs(args.data_root, args.operators)
    if not op_dirs:
        parser.error(f"No operator directories found under {args.data_root}")
    for op_dir in op_dirs:
        _check_operator(
            op_dir=op_dir,
            hardware_by_name=hardware_by_name,
            args=args,
            issues=issues,
        )

    if args.warnings_as_errors:
        has_failure = any(issue.severity in {"ERROR", "WARN"} for issue in issues)
    else:
        has_failure = any(issue.severity == "ERROR" for issue in issues)

    if args.format == "json":
        payload = {
            "passed": not has_failure,
            "issue_counts": {
                "ERROR": sum(1 for issue in issues if issue.severity == "ERROR"),
                "WARN": sum(1 for issue in issues if issue.severity == "WARN"),
            },
            "issues": [_jsonable(asdict(issue)) for issue in issues],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    else:
        text = _format_text(issues, verbose=args.verbose, max_issues=args.max_issues) + "\n"

    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[OUTPUT] {args.output}")

    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
