"""Tests for scripts/run_generate.py path resolution."""

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_generate import (
    DEFAULT_OUTPUT_ROOT,
    GENERATED_KERNEL_FILENAME,
    PROJECT_ROOT,
    SUPPORTED_BACKENDS,
    _collect_items,
    _parse_top_level_toml_model,
    _resolve_model,
    _resolve_template,
)


def _write_reference(data_dir: Path, name: str) -> Path:
    operator_dir = data_dir / name
    operator_dir.mkdir(parents=True)
    reference_path = operator_dir / "reference.py"
    reference_path.write_text("print('ref')\n", encoding="utf-8")
    return reference_path


def _make_args(**overrides) -> Namespace:
    defaults = {
        "backend": "triton",
        "reference": None,
        "template": None,
        "output_dir": None,
        "data_dir": Path("/unused"),
        "operator": None,
        "run_all": False,
        "cli": "codex",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_resolve_template_uses_backend_markdown_prompts(backend: str) -> None:
    template_path = _resolve_template(_make_args(backend=backend))

    assert template_path == PROJECT_ROOT / "prompt" / backend / "generate_kernel.md"


def test_collect_items_defaults_to_timestamped_output_dir(tmp_path: Path) -> None:
    reference_path = _write_reference(tmp_path, "fused_moe")
    timestamp = "20260416-120000"

    items = _collect_items(_make_args(reference=reference_path), timestamp=timestamp)

    assert items == [
        (
            reference_path,
            DEFAULT_OUTPUT_ROOT / "fused_moe" / timestamp / GENERATED_KERNEL_FILENAME,
        ),
    ]


def test_collect_items_uses_output_dir_for_single_reference(tmp_path: Path) -> None:
    reference_path = _write_reference(tmp_path / "references", "fused_moe")
    output_dir = tmp_path / "outputs" / "fused_moe"

    items = _collect_items(
        _make_args(reference=reference_path, output_dir=output_dir),
        timestamp="20260416-120000",
    )

    assert items == [
        (reference_path, output_dir / GENERATED_KERNEL_FILENAME),
    ]


def test_collect_items_uses_output_dir_for_single_operator(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    reference_path = _write_reference(data_dir, "fused_moe")
    output_dir = tmp_path / "outputs" / "custom_fused_moe"

    items = _collect_items(
        _make_args(data_dir=data_dir, operator="fused_moe", output_dir=output_dir),
        timestamp="20260416-120000",
    )

    assert items == [
        (reference_path, output_dir / GENERATED_KERNEL_FILENAME),
    ]


def test_collect_items_uses_operator_subdirs_for_all_with_output_dir(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    fused_moe_ref = _write_reference(data_dir, "fused_moe")
    attention_ref = _write_reference(data_dir, "unified_attention")
    output_dir = tmp_path / "outputs"

    items = _collect_items(
        _make_args(data_dir=data_dir, output_dir=output_dir, run_all=True),
        timestamp="20260416-120000",
    )

    assert items == [
        (fused_moe_ref, output_dir / "fused_moe" / GENERATED_KERNEL_FILENAME),
        (
            attention_ref,
            output_dir / "unified_attention" / GENERATED_KERNEL_FILENAME,
        ),
    ]


def test_collect_items_uses_same_timestamp_for_all_default_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    fused_moe_ref = _write_reference(data_dir, "fused_moe")
    attention_ref = _write_reference(data_dir, "unified_attention")
    timestamp = "20260416-120000"

    items = _collect_items(
        _make_args(data_dir=data_dir, run_all=True),
        timestamp=timestamp,
    )

    assert items == [
        (
            fused_moe_ref,
            DEFAULT_OUTPUT_ROOT / "fused_moe" / timestamp / GENERATED_KERNEL_FILENAME,
        ),
        (
            attention_ref,
            DEFAULT_OUTPUT_ROOT
            / "unified_attention"
            / timestamp
            / GENERATED_KERNEL_FILENAME,
        ),
    ]


# ---------------------------------------------------------------------------
# _parse_top_level_toml_model
# ---------------------------------------------------------------------------


def test_parse_top_level_toml_model_picks_root_key() -> None:
    text = (
        'model = "gpt-5.5-0424-global"\n'
        'model_provider = "custom"\n'
        'model_reasoning_effort = "xhigh"\n'
        '[projects."/root"]\n'
        'trust_level = "trusted"\n'
    )
    assert _parse_top_level_toml_model(text) == "gpt-5.5-0424-global"


def test_parse_top_level_toml_model_ignores_keys_inside_sections() -> None:
    """A `model` key under a [section] header must not be returned."""
    text = (
        '[fake.section]\n'
        'model = "should-be-ignored"\n'
    )
    assert _parse_top_level_toml_model(text) is None


def test_parse_top_level_toml_model_does_not_match_model_prefix_keys() -> None:
    """`model_provider`/`model_reasoning_effort` must not be picked up as `model`."""
    text = (
        'model_provider = "custom"\n'
        'model_reasoning_effort = "xhigh"\n'
    )
    assert _parse_top_level_toml_model(text) is None


def test_parse_top_level_toml_model_returns_none_for_empty_value() -> None:
    text = 'model = ""\n'
    assert _parse_top_level_toml_model(text) is None


def test_parse_top_level_toml_model_handles_comment_after_value() -> None:
    text = 'model = "gpt-x"  # current default\n'
    assert _parse_top_level_toml_model(text) == "gpt-x"


def test_parse_top_level_toml_model_handles_blank_file() -> None:
    assert _parse_top_level_toml_model("") is None


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


def test_resolve_model_reads_codex_config_toml(
    monkeypatch, tmp_path: Path
) -> None:
    """codex's model is sourced exclusively from $CODEX_HOME/config.toml."""
    fake_home = tmp_path / "codex_home"
    fake_home.mkdir()
    (fake_home / "config.toml").write_text(
        'model = "gpt-5.5-0424-global"\n'
        'model_provider = "custom"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(fake_home))

    assert _resolve_model(_make_args(cli="codex")) == "gpt-5.5-0424-global"


def test_resolve_model_raises_when_codex_config_missing(
    monkeypatch, tmp_path: Path
) -> None:
    """No config.toml => loud failure pointing at the fix.

    The bench refuses to invent a model id because the bundle's
    ``agent.model`` must match what codex itself would have selected.
    """
    fake_home = tmp_path / "codex_home"
    fake_home.mkdir()  # exists but no config.toml inside
    monkeypatch.setenv("CODEX_HOME", str(fake_home))

    with pytest.raises(ValueError) as exc_info:
        _resolve_model(_make_args(cli="codex"))

    message = str(exc_info.value)
    assert "Could not resolve a codex model" in message
    assert str(fake_home / "config.toml") in message


def test_resolve_model_raises_when_codex_config_lacks_model_key(
    monkeypatch, tmp_path: Path
) -> None:
    """config.toml without a top-level `model` => loud failure."""
    fake_home = tmp_path / "codex_home"
    fake_home.mkdir()
    (fake_home / "config.toml").write_text(
        'approval_policy = "never"\n'
        '[projects."/root"]\n'
        'trust_level = "trusted"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(fake_home))

    with pytest.raises(ValueError, match="Could not resolve a codex model"):
        _resolve_model(_make_args(cli="codex"))


def test_resolve_model_returns_none_for_claude(
    monkeypatch, tmp_path: Path
) -> None:
    """Claude path always returns None: the bench leaves model selection to
    claude's native config and recovers the actual model id from the trace
    afterwards. Even a present codex config must not leak into the claude
    branch.
    """
    fake_home = tmp_path / "codex_home"
    fake_home.mkdir()
    (fake_home / "config.toml").write_text(
        'model = "gpt-5.5-0424-global"\n', encoding="utf-8"
    )
    monkeypatch.setenv("CODEX_HOME", str(fake_home))

    assert _resolve_model(_make_args(cli="claude")) is None
