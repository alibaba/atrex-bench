"""Stage 0: import-time checks for candidate modules."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from atrex_bench.eval._runtime import import_module_from_path, validate_candidate_module


@dataclass(frozen=True)
class CompileResult:
    """Result of validating a candidate model file.

    ``status`` is ``"passed"`` when the module imports and exposes the
    candidate contract, otherwise ``"failed"``. ``reason`` carries the
    full traceback on failure and is ``None`` when ``status == "passed"``.
    """

    status: str
    reason: str | None = None


def check_compilation(candidate_path: Path) -> CompileResult:
    """Validate that a candidate file can be imported and exposes the required contract."""
    try:
        module = import_module_from_path(candidate_path, "atrex_candidate_compile")
        validate_candidate_module(module)
        return CompileResult(status="passed", reason=None)
    except Exception:
        return CompileResult(status="failed", reason=traceback.format_exc())
