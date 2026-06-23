"""Unified estimator for theoretical compute and implementation-related bytes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from atrex_bench.eval.estimate import SUPPORTED_ESTIMATE_MODES, estimate
from atrex_bench.utils import save_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate roofline-oriented quantities from a contract-compliant module"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=sorted(SUPPORTED_ESTIMATE_MODES),
        help="Which quantity to estimate.",
    )
    parser.add_argument(
        "--module",
        type=Path,
        required=True,
        help="Path to the module under evaluation (reference.py or modelnew.py).",
    )
    parser.add_argument(
        "--semantic-source",
        type=Path,
        default=None,
        help=(
            "Optional semantic source module used for W_theoretical or "
            "Q_semantic_lower_bound. Defaults to --module."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device: auto, cpu, cuda, cuda:0, hip, hip:0 ...",
    )
    parser.add_argument(
        "--profile-backend",
        type=str,
        default="auto",
        help="Implementation-Q backend: auto or op_trace_estimate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the selected mode depends on unsupported or heuristic estimation paths.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include detailed inputs, op breakdowns, and environment data.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--shape-id",
        type=str,
        default="0",
        help=(
            "Which shape from shapes.json (next to --module) to use. Defaults "
            "to '0' for backward compatibility. Used by per-operator roofline "
            "refresh flows when looping over every sid."
        ),
    )
    args = parser.parse_args()

    result = estimate(
        mode=args.mode,
        module_path=args.module,
        semantic_source_path=args.semantic_source,
        device=args.device,
        profile_backend=args.profile_backend,
        strict=args.strict,
        verbose=args.verbose,
        shape_id=args.shape_id,
    )
    payload = asdict(result)
    if args.output is not None:
        save_json(payload, args.output)
        print(f"[OUTPUT] {args.output}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
