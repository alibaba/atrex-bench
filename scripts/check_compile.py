"""Repository wrapper for the installed Atrex-Bench compile-check CLI."""

from __future__ import annotations

import sys

from atrex_bench.cli import check_compile as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
