"""CLI for Stage 0: validate that a candidate module imports successfully."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from atrex_bench.eval.compile import check_compilation
from atrex_bench.utils import save_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a candidate module passes compile stage"
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to the candidate Python file",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    result = asdict(check_compilation(args.candidate))
    if args.output is not None:
        save_json(result, args.output)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
