#!/usr/bin/env python3
"""Delete everything the generation-stage agent must not see.

Run it once, right after cloning, before generating:

    git clone <repo> && cd atrex-bench
    python scripts/cleanup_for_generation.py
    python scripts/run_generate.py --operator fused_moe --backend triton

It keeps only what the agent needs to turn a reference into a kernel — each
operator's ``reference.py`` / ``input.py`` / ``shapes.json``, the ``prompt/``
templates, and the generation runner — and deletes everything else
(``metadata.json``, ``roofline.json``, ``configs/``, the evaluator, ``tests/``,
``site/``, the README, and the ``.git`` history) so provenance and SOL targets
cannot leak.

Destructive and irreversible. It strips its own checkout, so run it on a clone
used only for generation, never on the repo you evaluate from.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The only files that survive. Everything not matched here is deleted.
KEEP_FILES = {
    "pyproject.toml",
    "LICENSE",
    "NOTICE",
    "scripts/run_generate.py",
    "scripts/view_trace.py",
    "scripts/cleanup_for_generation.py",
    "src/atrex_bench/__init__.py",
    "src/atrex_bench/generate.py",
    "src/atrex_bench/utils.py",
}
KEEP_OP_FILES = {"reference.py", "input.py", "shapes.json"}


def is_kept(rel: Path) -> bool:
    """Whether a repo-relative file path is part of the agent-visible surface."""
    parts = rel.parts
    if rel.as_posix() in KEEP_FILES:
        return True
    if parts and parts[0] == "prompt":
        return True
    return len(parts) == 3 and parts[0] == "data" and parts[2] in KEEP_OP_FILES


def main() -> None:
    deleted = kept = 0
    # Deepest paths first so directories are empty by the time we reach them.
    for path in sorted(REPO_ROOT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel = path.relative_to(REPO_ROOT)
        if parts_has_git(rel):
            continue  # .git is removed wholesale below
        if path.is_dir():
            if not any(path.iterdir()):
                path.rmdir()
        elif is_kept(rel):
            kept += 1
        else:
            path.unlink()
            deleted += 1

    # Drop git history so deleted files can't be recovered with `git checkout`.
    shutil.rmtree(REPO_ROOT / ".git", ignore_errors=True)
    print(f"cleanup: kept {kept} agent-visible files, deleted {deleted} files + .git/")


def parts_has_git(rel: Path) -> bool:
    return bool(rel.parts) and rel.parts[0] == ".git"


if __name__ == "__main__":
    main()
