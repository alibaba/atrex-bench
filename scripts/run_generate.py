"""Generate GPU kernel implementations by calling an LLM CLI.

Usage:
    # Single operator with Claude (default output: outputs/<operator>/<timestamp>/)
    python scripts/run_generate.py --operator fused_moe --backend triton

    # Single operator with extra whitelist-staged helpers
    python scripts/run_generate.py \\
        --operator fused_moe \\
        --backend triton \\
        --stage-path scripts/check_compile.py \\
        --stage-path tests/fixtures \\
        --cli codex

    # Single operator with Codex, written to a custom directory
    python scripts/run_generate.py \\
        --operator fused_moe \\
        --backend triton \\
        --cli codex \\
        --output-dir outputs/fused_moe

    # All operators under data/, written under a custom root
    python scripts/run_generate.py \\
        --all \\
        --backend triton \\
        --cli codex \\
        --output-dir outputs/batch

    # Explicit reference path
    python scripts/run_generate.py \\
        --template prompt/triton/generate_kernel.md \\
        --reference data/fused_moe/reference.py \\
        --output-dir outputs/fused_moe \\
        --cli codex
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from atrex_bench.generate import (
    GENERATED_KERNEL_FILENAME,
    generate_kernel,
    get_generation_bundle_path,
    get_generation_trace_path,
)
from atrex_bench.utils import get_timestamp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SUPPORTED_BACKENDS = ("triton", "gluon", "flydsl", "cutedsl")


def _codex_home() -> Path:
    """Mirror codex's own resolution: ``$CODEX_HOME`` else ``~/.codex``."""

    return Path(os.environ.get("CODEX_HOME") or "~/.codex").expanduser()


def _parse_top_level_toml_model(toml_text: str) -> str | None:
    """Return the top-level ``model = "..."`` value from a codex config.toml.

    Walks the TOML line by line and only picks up ``model`` when the current
    table is the root table (i.e. before any ``[section]`` header).  This
    deliberately avoids pulling in a TOML library so the helper works on
    Python 3.10 (which lacks ``tomllib``) without adding a runtime dep.

    Limitations: handles only the basic-string syntax (``"..."``) that
    codex emits for ``model``.  Falls through to ``None`` for literal
    strings, multi-line strings, or inline-table forms — none of which
    codex writes for this key.  ``tomllib`` / ``tomli`` is consulted first
    when available so well-formed configs always parse correctly.
    """

    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            tomllib = None  # type: ignore[assignment]

    if tomllib is not None:
        try:
            data = tomllib.loads(toml_text)
        except Exception:
            data = None
        if isinstance(data, dict):
            value = data.get("model")
            if isinstance(value, str) and value:
                return value
        # Fall through to regex if tomllib parsed but produced nothing useful.

    in_root_table = True
    for raw_line in toml_text.splitlines():
        # Strip trailing comments and whitespace; preserve inline `=`.
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_root_table = False
            continue
        if not in_root_table:
            continue
        match = re.match(r'^model\s*=\s*"([^"]*)"\s*$', line)
        if match:
            value = match.group(1)
            return value or None
    return None


def _read_codex_config_model() -> str | None:
    """Return the top-level ``model`` key from ``$CODEX_HOME/config.toml``.

    Returns ``None`` if the file is missing, unreadable, or has no
    top-level ``model`` entry. Honors ``$CODEX_HOME`` so atrex-bench
    resolves the same config codex itself would load.
    """

    config_path = _codex_home() / "config.toml"
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_top_level_toml_model(text)


def _resolve_template(args: argparse.Namespace) -> Path:
    if args.template:
        return args.template
    return PROJECT_ROOT / "prompt" / args.backend / "generate_kernel.md"


def _resolve_model(args: argparse.Namespace) -> str | None:
    """Resolve the model string passed to the generator CLI.

    The bench treats each CLI's native config mechanism as the *single*
    source of truth for which model to run; atrex-bench deliberately does
    not expose a per-run override, to keep the resulting bundle's
    ``agent.model`` aligned with what the host's CLI would actually load.

    Codex:
        Returns the top-level ``model`` key from ``$CODEX_HOME/config.toml``
        (= the same default codex itself loads). Raises ``ValueError`` with
        the exact config path when the file is missing or omits ``model``,
        so the caller can surface a clear CLI error; the bench refuses to
        invent a fallback because that would risk recording a model id that
        codex itself would not have picked.

    Claude:
        Returns ``None``. Claude's stream-json trace carries the real model
        id in its ``system.init`` event and the bench backfills
        ``agent.model`` after the run, so pre-resolution is not required.
        To pin a specific claude model, configure claude's own settings
        (e.g. ``~/.claude/settings.json``) on the host.
    """

    if args.cli == "codex":
        config_model = _read_codex_config_model()
        if config_model:
            return config_model
        config_path = _codex_home() / "config.toml"
        raise ValueError(
            "Could not resolve a codex model from the host config. "
            f"Set `model = \"...\"` at the top of {config_path} "
            "(override the directory with $CODEX_HOME)."
        )
    return None


def _read_bundle_model(output_path: Path) -> str | None:
    """Pull the actual model id from the just-written generation.json."""

    bundle_path = get_generation_bundle_path(output_path)
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    agent = bundle.get("agent") or {}
    model = agent.get("model")
    return model if isinstance(model, str) and model else None


def _resolve_output_path(
    reference_path: Path,
    output_dir: Path | None,
    *,
    timestamp: str,
    use_operator_subdir: bool = False,
) -> Path:
    if output_dir is None:
        return (
            DEFAULT_OUTPUT_ROOT
            / reference_path.parent.name
            / timestamp
            / GENERATED_KERNEL_FILENAME
        )
    if use_operator_subdir:
        return output_dir / reference_path.parent.name / GENERATED_KERNEL_FILENAME
    return output_dir / GENERATED_KERNEL_FILENAME


def _collect_items(
    args: argparse.Namespace,
    *,
    timestamp: str,
) -> list[tuple[Path, Path]]:
    """Return a list of (reference_path, output_path) work items."""
    if args.reference:
        ref = args.reference
        out = _resolve_output_path(ref, args.output_dir, timestamp=timestamp)
        return [(ref, out)]

    data_dir = args.data_dir
    if args.operator:
        operators = [args.operator]
    else:
        operators = sorted(
            d.name
            for d in data_dir.iterdir()
            if d.is_dir() and (d / "reference.py").exists()
        )

    items = []
    for name in operators:
        ref = data_dir / name / "reference.py"
        if not ref.exists():
            print(f"[SKIP] {name}: reference.py not found")
            continue
        out = _resolve_output_path(
            ref,
            args.output_dir,
            timestamp=timestamp,
            use_operator_subdir=args.run_all and args.output_dir is not None,
        )
        items.append((ref, out))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GPU kernel implementations via an LLM CLI"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--operator", type=str, help="Operator name under data dir")
    group.add_argument(
        "--all", action="store_true", dest="run_all", help="Run for all operators"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        choices=SUPPORTED_BACKENDS,
        help="Backend type — selects prompt template (default: triton)",
    )
    parser.add_argument("--template", type=Path, help="Explicit prompt template path")
    parser.add_argument("--reference", type=Path, help="Explicit reference.py path")
    parser.add_argument(
        "--stage-path",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional file or directory to whitelist-stage into the isolated "
            "generation workspace. Repeatable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional output directory. Single operator/reference writes "
            f"{GENERATED_KERNEL_FILENAME} inside it; --all writes "
            f"<output_dir>/<operator>/{GENERATED_KERNEL_FILENAME}"
        ),
    )
    parser.add_argument(
        "--cli",
        type=str,
        default="claude",
        choices=["claude", "codex"],
        help=(
            "Generator CLI to invoke (default: claude). Model selection is "
            "delegated to the CLI's own native config: codex reads "
            "$CODEX_HOME/config.toml; claude reads its own settings and "
            "reports the actual model id back through stream-json. atrex-bench "
            "does not expose a per-run `--model` override to keep the bundle's "
            "agent.model aligned with what the host CLI would actually load."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only render and print the prompt, do not call the selected CLI",
    )
    parser.add_argument(
        "--mirror-trace",
        action="store_true",
        help=(
            "Echo the generator's trace stream to stdout in real time. "
            "Useful when running in the foreground; leave off when backgrounded "
            "and tail the trace file with scripts/view_trace.py instead."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--skill",
        action="store_true",
        help=(
            "Enable skill-driven generation. Appends Skill,Task to the Claude "
            "tool allowlist so an installed skill (e.g. gpu-kernel-optimizer) "
            "can be invoked and spawn its prescribed subagents. Codex already "
            "runs with --sandbox danger-full-access by default for GPU device "
            "visibility, so this flag is a no-op on the Codex sandbox tier. "
            "Default OFF preserves the strict anti-hack surface on the Claude "
            "tool allowlist; opt in only when running skill-driven evaluations. "
            "Generated kernels with --skill ON should NOT be directly compared "
            "against runs without it."
        ),
    )

    args = parser.parse_args()
    try:
        model = _resolve_model(args)
    except ValueError as exc:
        parser.error(str(exc))
    timestamp = get_timestamp()

    if not (args.operator or args.run_all or args.reference):
        parser.error("one of --operator, --all, or --reference is required")
    if args.reference and (args.operator or args.run_all):
        parser.error("--reference cannot be combined with --operator or --all")

    template = _resolve_template(args)
    if not template.exists():
        print(f"Error: template not found: {template}", file=sys.stderr)
        sys.exit(1)

    items = _collect_items(args, timestamp=timestamp)
    if not items:
        print("No operators to process.")
        return

    if args.skill:
        print(
            "[WARN] --skill is ON: enabling Skill,Task tools on the Claude "
            "side. This bypasses the default anti-hack restrictions on the "
            "Claude tool allowlist and is intended only for skill-driven "
            "generation testing. Do NOT directly compare these results with "
            "non-skill runs. (Codex always runs with --sandbox "
            "danger-full-access regardless of --skill.)",
            file=sys.stderr,
        )
        if not Path("<gpu-wiki-path>").is_dir():
            print(
                "[WARN] <gpu-wiki-path>/ not found. The gpu-kernel-optimizer skill "
                "expects this knowledge base; the agent will try to git clone it "
                "during startup. If your network/SSH key cannot reach GitLab, "
                "pre-clone with: git clone "
                "<your-gpu-wiki-repo> <gpu-wiki-path>",
                file=sys.stderr,
            )

    failed: list[str] = []
    for i, (ref, out) in enumerate(items, 1):
        name = ref.parent.name

        if args.dry_run:
            from atrex_bench.generate import render_prompt

            prompt = render_prompt(template, ref)
            print(prompt)
            continue

        print(f"[{i}/{len(items)}] Generating {name} ...")
        if args.mirror_trace:
            print(f"--- trace stream ({name}) ---")
        try:
            generate_kernel(
                template,
                ref,
                out,
                cli=args.cli,
                model=model,
                stage_paths=args.stage_path,
                mirror_to_stdout=args.mirror_trace,
                repo_root=PROJECT_ROOT,
                skill_mode=args.skill,
            )
            actual_model = _read_bundle_model(out) or model or "<default>"
            print(
                f"[{i}/{len(items)}] {name} -> {out} "
                f"(model: {actual_model}, trace: {get_generation_trace_path(out)})"
            )
        except FileNotFoundError:
            print(
                f"Error: '{args.cli}' CLI not found on PATH. Install it first.",
                file=sys.stderr,
            )
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or "unknown error").strip()
            print(
                f"[FAIL] {name}: {msg} (trace: {get_generation_trace_path(out)})",
                file=sys.stderr,
            )
            failed.append(name)

    if failed:
        print(f"\nFailed ({len(failed)}/{len(items)}): {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
