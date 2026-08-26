"""Repository wrapper for the installed Atrex-Bench evaluator CLI."""

from __future__ import annotations

import sys

from atrex_bench.cli import run_eval as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
