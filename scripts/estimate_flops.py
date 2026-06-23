"""Estimate theoretical compute cost for a reference module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atrex_bench.eval.flops import FlopEstimateResult, estimate_theoretical_flops
from atrex_bench.utils import save_json


def _precision_status(result: FlopEstimateResult) -> str:
    """Return a concise precision label for CLI output."""
    if not result.flops_complete or not result.bytes_complete:
        return "partial"
    return "estimated"


def _build_summary_payload(
    result: FlopEstimateResult,
    *,
    verbose: bool,
) -> dict[str, object]:
    """Build the default concise CLI payload, with optional details."""
    payload: dict[str, object] = {
        "passed": result.passed,
        "operator_name": result.operator_name,
        "device": result.device,
        "theoretical_flops": result.total_flops,
        "theoretical_flops_by_dtype": dict(result.flops_by_dtype),
        "theoretical_data_movement_bytes": result.total_bytes,
        "theoretical_read_bytes": result.total_read_bytes,
        "theoretical_write_bytes": result.total_write_bytes,
        "arithmetic_intensity_flops_per_byte": result.arithmetic_intensity,
        "precision": {
            "status": _precision_status(result),
            "flops_complete": result.flops_complete,
            "bytes_complete": result.bytes_complete,
            "bytes_model": "execution_trace_tensor_io_estimate",
        },
        "error": result.error,
    }
    precision = payload["precision"]
    if isinstance(precision, dict):
        if result.uncounted_ops:
            precision["unsupported_flop_ops"] = result.uncounted_ops
        if result.heuristic_byte_ops:
            precision["heuristic_byte_ops"] = result.heuristic_byte_ops

    if verbose:
        payload["details"] = {
            "environment": result.environment,
            "inputs": result.inputs,
            "counted_flop_ops": result.counted_ops,
            "zero_flop_ops": result.zero_flop_ops,
            "uncounted_flop_op_invocations": result.uncounted_op_invocations,
            "counted_byte_ops": result.counted_byte_ops,
            "read_byte_ops": result.read_byte_ops,
            "write_byte_ops": result.write_byte_ops,
            "zero_byte_ops": result.zero_byte_ops,
            "heuristic_byte_op_invocations": result.heuristic_byte_op_invocations,
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate theoretical FLOPs and data movement for reference.py"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to reference.py",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device: auto, cpu, cuda, cuda:0, hip, hip:0 ...",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any executed aten op does not have a FLOP formula.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include detailed op breakdowns, environment, and input summaries.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--shape-id",
        type=str,
        default="0",
        help=(
            "Which shape from shapes.json (next to --reference) to use. "
            "Defaults to '0' for backward compatibility."
        ),
    )
    args = parser.parse_args()

    result = estimate_theoretical_flops(
        args.reference,
        device=args.device,
        strict=args.strict,
        shape_id=args.shape_id,
    )
    payload = _build_summary_payload(result, verbose=args.verbose)
    if args.output is not None:
        save_json(payload, args.output)
        print(f"[OUTPUT] {args.output}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
