"""Tests for scripts/cleanup_for_generation.py — the generation-stage stripper.

The tests build a throwaway fake repo under ``tmp_path``; the real checkout is
never touched.
"""

from pathlib import Path

import pytest

from scripts import cleanup_for_generation as clean
from scripts.cleanup_for_generation import is_kept

KEEP_PATHS = [
    "pyproject.toml",
    "LICENSE",
    "NOTICE",
    "scripts/run_generate.py",
    "scripts/view_trace.py",
    "scripts/cleanup_for_generation.py",
    "src/atrex_bench/__init__.py",
    "src/atrex_bench/generate.py",
    "src/atrex_bench/utils.py",
    "prompt/triton/generate_kernel.md",
    "prompt/flydsl/constraints.md",
    "data/fused_moe/reference.py",
    "data/fused_moe/input.py",
    "data/fused_moe/shapes.json",
]

DELETE_PATHS = [
    "data/fused_moe/metadata.json",
    "data/fused_moe/roofline.json",
    "data/README.md",
    "data/operator_importance.json",
    "scripts/run_eval.py",
    "scripts/roofline.py",
    "src/atrex_bench/eval/__init__.py",
    "src/atrex_bench/eval/roofline.py",
    "tests/test_eval.py",
    "tests/fixtures/references/atrex_001/reference.py",
    "configs/hardware/XPU-A.yaml",
    "site/package.json",
    "docker/Dockerfile.rocm",
    "README.md",
    ".dockerignore",
]


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_fake_repo(root: Path) -> None:
    for rel in KEEP_PATHS + DELETE_PATHS:
        _write(root / rel)
    _write(root / ".git" / "config")  # fake history — must be removed


@pytest.mark.parametrize("rel", KEEP_PATHS)
def test_is_kept_true_for_visible_surface(rel: str) -> None:
    assert is_kept(Path(rel)) is True


@pytest.mark.parametrize("rel", DELETE_PATHS)
def test_is_kept_false_for_hidden_surface(rel: str) -> None:
    assert is_kept(Path(rel)) is False


def test_main_strips_to_visible_surface(tmp_path: Path, monkeypatch) -> None:
    _make_fake_repo(tmp_path)
    monkeypatch.setattr(clean, "REPO_ROOT", tmp_path)

    clean.main()

    for rel in KEEP_PATHS:
        assert (tmp_path / rel).exists(), f"keep file removed: {rel}"
    for rel in DELETE_PATHS:
        assert not (tmp_path / rel).exists(), f"delete file survived: {rel}"

    # Emptied directories are pruned; .git is gone; data/<op> keeps its files.
    assert not (tmp_path / ".git").exists()
    assert not (tmp_path / "configs").exists()
    assert not (tmp_path / "tests").exists()
    assert not (tmp_path / "site").exists()
    assert not (tmp_path / "src/atrex_bench/eval").exists()
    assert (tmp_path / "data/fused_moe/reference.py").exists()
