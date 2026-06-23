"""Render prompt templates and invoke an LLM CLI to generate GPU kernels."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

PLACEHOLDER = "{{REFERENCE_CODE}}"
CLAUDE_EFFORT = "max"
# Explicit tool allowlist for claude. Drops WebFetch / WebSearch / Task /
# NotebookEdit / SlashCommand / Skill etc. — none of which the kernel-writing
# workflow needs. Bash + file I/O + search is the minimum viable surface.
CLAUDE_TOOLS = "Bash,Read,Edit,Write,MultiEdit,Glob,Grep"
# Extra tools enabled only when skill_mode=True (opt-in, e.g. via --skill on
# the run_generate CLI). Skill lets the agent actually invoke installed
# Claude Code skills (e.g. gpu-kernel-optimizer); Task lets the skill spawn
# its prescribed worker / researcher / validator subagents. WebFetch /
# WebSearch are still excluded by design — the skill workflow's "public web
# fallback" stage is intentionally disabled in benchmark runs to keep
# generation comparable across runs (no internet-mediated copying).
CLAUDE_SKILL_EXTRA_TOOLS = "Skill,Task"
CODEX_APPROVAL_POLICY = "never"
# Codex reasoning effort. The supported values are low / medium / high /
# xhigh. atrex-bench pins xhigh (the deepest tier) because the kernel
# generation task is intentionally hard — backends like FlyDSL / CUTeDSL
# have no example notebooks shipped, and lower tiers consistently lead the
# agent to "play it safe" by skipping the backend kernel entirely and
# emitting a pure-PyTorch grouped path (see analysis on
# outputs/fused_moe/20260521-031231). Passing this as a `-c` override
# also surfaces the effort in the bundle's agent.command, so each run
# explicitly records the reasoning tier it consumed regardless of what
# $CODEX_HOME/config.toml says.
CODEX_REASONING_CONFIG = 'model_reasoning_effort="xhigh"'
# Codex's --sandbox workspace-write uses a bundled bubblewrap that remounts
# /dev as a fresh tmpfs without /dev/kfd or /dev/dri/*, so ROCm cannot probe
# the GPU and `torch.cuda.is_available()` returns False inside the agent's
# shell. That blocks any in-generation JIT/launch verification for backends
# like FlyDSL/CUTeDSL, which leads agents to fall back to "no-op kernel"
# compliance tricks. danger-full-access skips bwrap entirely, matching
# Claude's default unsandboxed subprocess model so codex sees the same
# host /dev/kfd, /dev/dri/*, GPU, network, and filesystem that Claude does.
# The atrex-bench anti-reward-hack surface lives in stage_generation_access
# (only prompt + reference.py + input.py + shapes.json are staged into the
# agent's cwd; metadata.json/roofline.json are deliberately withheld) and
# is independent of the kernel-level sandbox tier.
CODEX_SANDBOX_MODE = "danger-full-access"
# Codex sandbox used in skill_mode. Historically this was the only mode that
# bypassed bwrap (so skill workflows could clone gpu-wiki to <gpu-wiki-path>/
# and persist kernel_opt_<name>/ across stages). The default mode now also
# uses danger-full-access for GPU visibility, so this is kept as a separate
# constant only for documentation parity with CLAUDE_SKILL_EXTRA_TOOLS.
CODEX_SKILL_SANDBOX_MODE = "danger-full-access"
GENERATED_KERNEL_FILENAME = "generated_kernel.py"
GENERATION_BUNDLE_FILENAME = "generation.json"
GENERATION_TRACE_SUFFIX = ".trace.log"
_FILE_CHANGE_TOOL_NAMES = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
# Mirrors run_generate.SUPPORTED_BACKENDS; kept local to avoid importing the CLI
# module (which already imports this one).
_SUPPORTED_BACKENDS = ("triton", "gluon", "flydsl", "cutedsl")


@dataclass
class GenerationRun:
    """Result bundle for one generator CLI invocation."""

    cli: str
    model: str | None
    command: list[str]
    returncode: int
    output: str
    stderr: str
    trace: str
    trace_format: str


@dataclass(frozen=True)
class GenerationAccess:
    """Restricted filesystem view exposed to the generator CLI."""

    workspace_root: Path
    template_path: Path
    reference_path: Path


def _artifact_path(output_path: Path, suffix: str) -> Path:
    return output_path.parent / f"{output_path.stem}{suffix}"


def get_generation_trace_path(output_path: Path) -> Path:
    """Return the sidecar path used for raw generation traces."""

    return _artifact_path(output_path, GENERATION_TRACE_SUFFIX)


def get_generation_bundle_path(output_path: Path) -> Path:
    """Return the path of the unified ``generation.json`` bundle."""

    return output_path.parent / GENERATION_BUNDLE_FILENAME


def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp at second precision (e.g. ``2026-04-23T03:22:48Z``)."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    """Return *path* relative to *root* when possible, else its string form."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _backend_from_template(template_path: Path) -> str:
    """Infer the backend from the template's parent directory.

    Prompts live at ``prompt/<backend>/generate_kernel.md`` (and the
    ``…_with_optimizer.md`` variant), so the backend is the parent directory
    name. Falls back to the file stem for non-standard ``--template`` paths.
    """
    parent = template_path.parent.name
    return parent if parent in _SUPPORTED_BACKENDS else template_path.stem


def _empty_summary() -> dict[str, int]:
    return {"agent_messages": 0, "tool_uses": 0, "file_changes": 0, "errors": 0}


def _empty_usage() -> dict[str, object]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "duration_ms": None,
        "cost_usd": None,
    }


def _extract_actual_model(trace_text: str, cli: str) -> str | None:
    """Return the model id the CLI actually used, parsed from the trace.

    For claude (stream-json) the ``system`` init event carries ``model``
    (e.g. ``claude-opus-4-7[1m]``). Codex's JSONL stream does not surface a
    model id, so we return None and the caller falls back to ``run.model``.
    """

    if cli != "claude":
        return None
    for raw in trace_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            model = event.get("model")
            if isinstance(model, str) and model:
                return model
            return None
    return None


def _summarize_trace(trace_text: str, cli: str) -> tuple[dict, dict, int]:
    """Parse a trace JSONL transcript; return (summary, usage, event_count)."""

    summary = _empty_summary()
    usage = _empty_usage()
    event_count = 0
    for raw in trace_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_count += 1
        if cli == "claude":
            etype = event.get("type")
            if etype == "assistant":
                for block in event.get("message", {}).get("content", []) or []:
                    btype = block.get("type")
                    if btype == "text":
                        summary["agent_messages"] += 1
                    elif btype == "tool_use":
                        summary["tool_uses"] += 1
                        if block.get("name") in _FILE_CHANGE_TOOL_NAMES:
                            summary["file_changes"] += 1
            elif etype == "user":
                for block in event.get("message", {}).get("content", []) or []:
                    if block.get("type") == "tool_result" and block.get("is_error"):
                        summary["errors"] += 1
            elif etype == "result":
                u = event.get("usage") or {}
                usage["input_tokens"] = u.get("input_tokens", 0) or 0
                usage["cached_input_tokens"] = (
                    (u.get("cache_read_input_tokens", 0) or 0)
                    + (u.get("cache_creation_input_tokens", 0) or 0)
                )
                usage["output_tokens"] = u.get("output_tokens", 0) or 0
                usage["duration_ms"] = event.get("duration_ms")
                usage["cost_usd"] = event.get("total_cost_usd")
        elif cli == "codex":
            etype = event.get("type")
            if etype == "item.completed":
                item = event.get("item") or {}
                itype = item.get("type")
                if itype == "agent_message":
                    summary["agent_messages"] += 1
                elif itype == "command_execution":
                    summary["tool_uses"] += 1
                    exit_code = item.get("exit_code")
                    if exit_code not in (0, None):
                        summary["errors"] += 1
                elif itype == "file_change":
                    summary["file_changes"] += 1
            elif etype == "turn.completed":
                u = event.get("usage") or {}
                usage["input_tokens"] = u.get("input_tokens", 0) or 0
                usage["cached_input_tokens"] = u.get("cached_input_tokens", 0) or 0
                usage["output_tokens"] = u.get("output_tokens", 0) or 0
                # codex doesn't report duration_ms / cost_usd; leave defaults
    return summary, usage, event_count


def _build_generation_bundle(
    *,
    op_name: str,
    backend: str,
    output_path: Path,
    run: GenerationRun,
    trace_path: Path,
    template_path: Path,
    reference_path: Path,
    stage_paths: Iterable[Path] | None,
    repo_root: Path | None,
    prompt_text: str,
    started_at: str,
    completed_at: str,
    duration_s: float,
    skill_mode: bool = False,
    error: str | None = None,
) -> dict:
    rel = (lambda p: _safe_relative(p, repo_root)) if repo_root is not None else (lambda p: str(p))

    kernel_sha = _sha256_file(output_path)
    trace_sha = _sha256_file(trace_path) or ""
    byte_size = trace_path.stat().st_size if trace_path.exists() else 0
    trace_text = trace_path.read_text(encoding="utf-8") if trace_path.exists() else ""
    summary, usage, event_count = _summarize_trace(trace_text, run.cli)
    trace_format = "claude-stream-json" if run.cli == "claude" else "codex-jsonl"
    actual_model = _extract_actual_model(trace_text, run.cli) or run.model

    return {
        "kernel": {
            "name": op_name,
            "backend": backend,
            "path": output_path.name,
            "sha256": kernel_sha,
        },
        "agent": {
            "cli": run.cli,
            "model": actual_model,
            "command": list(run.command),
            "returncode": run.returncode,
            "skill_mode": skill_mode,
        },
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_s": duration_s,
        "prompt": {
            "template_path": rel(template_path),
            "reference_path": rel(reference_path),
            "stage_paths": [rel(p) for p in (stage_paths or ())],
            "text": prompt_text,
        },
        "trace": {
            "format": trace_format,
            "path": trace_path.name,
            "sha256": trace_sha,
            "byte_size": byte_size,
            "event_count": event_count,
            "summary": summary,
        },
        "usage": usage,
        "error": error,
    }


def _raise_for_generation_failure(run: GenerationRun) -> None:
    if run.returncode == 0:
        return
    raise subprocess.CalledProcessError(
        run.returncode,
        run.command,
        output=run.output,
        stderr=run.stderr,
    )


def _stream_subprocess(
    cmd: list[str],
    *,
    stdin_input: str,
    cwd: Path | None,
    trace_path: Path,
    stderr_path: Path,
    mirror_to_stdout: bool,
) -> int:
    """Run *cmd*, streaming stdout to *trace_path* and stderr to *stderr_path*.

    Both files are opened line-buffered so external observers (``tail -f``)
    see events as they arrive. When *mirror_to_stdout* is true, each stdout
    line is also echoed to ``sys.stdout`` so the calling terminal renders the
    trace live. Returns the subprocess exit code.
    """
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    def _write_stdin() -> None:
        try:
            proc.stdin.write(stdin_input)
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass

    def _drain(src, sink_path: Path, mirror: bool) -> None:
        with open(sink_path, "w", encoding="utf-8", buffering=1) as sink:
            for line in src:
                sink.write(line)
                if mirror:
                    sys.stdout.write(line)
                    sys.stdout.flush()

    threads = [
        threading.Thread(target=_write_stdin, name="stream-stdin", daemon=True),
        threading.Thread(
            target=_drain,
            args=(proc.stdout, trace_path, mirror_to_stdout),
            name="stream-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_drain,
            args=(proc.stderr, stderr_path, False),
            name="stream-stderr",
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    returncode = proc.wait()
    for t in threads:
        t.join()
    return returncode


def render_prompt(template_path: Path, reference_path: Path) -> str:
    """Read a prompt template and substitute the reference path placeholder.

    Args:
        template_path: Path to a prompt template containing {{REFERENCE_CODE}}.
        reference_path: Path to a PyTorch reference .py file.

    Returns:
        The fully rendered prompt string with the reference file path
        substituted into the placeholder.

    Raises:
        ValueError: If the template does not contain {{REFERENCE_CODE}}.
    """
    template = template_path.read_text()
    if PLACEHOLDER not in template:
        raise ValueError(f"Template {template_path} does not contain {PLACEHOLDER}")
    return template.replace(PLACEHOLDER, str(reference_path))


def _workspace_relative_stage_path(source_path: Path) -> Path:
    """Return the relative path used for whitelist-staged inputs."""

    source_resolved = source_path.resolve()
    cwd = Path.cwd().resolve()

    try:
        return source_resolved.relative_to(cwd)
    except ValueError:
        cleaned_parts = [
            part
            for part in source_resolved.parts
            if part not in {source_resolved.anchor, "/", ""}
        ]
        return Path("external").joinpath(*cleaned_parts)


def _copy_stage_entry(source_path: Path, destination_path: Path) -> None:
    """Copy a whitelist-staged file or directory into the temp workspace."""

    if source_path.is_dir():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, destination_path)
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


_AGENT_VISIBLE_SIBLING_FILES: tuple[str, ...] = ("input.py", "shapes.json")
"""Sibling files that auto-stage alongside reference.py per docs/data_schema.md.

Only files explicitly listed here are exposed to the candidate agent.
metadata.json and roofline.json are deliberately omitted — they hold
upstream provenance and roofline measurements respectively, and
exposing them would leak optimization hints / reward-hacking anchors.
The legacy metadata.yaml is also not staged.
"""


@contextmanager
def stage_generation_access(
    template_path: Path,
    reference_path: Path,
    stage_paths: Iterable[Path] | None = None,
) -> Iterator[GenerationAccess]:
    """Stage generation inputs plus an optional whitelist into a temp workspace.

    Always-staged inputs (per the data schema spec, Section 5):
        * the prompt template
        * reference.py
        * any sibling files in ``_AGENT_VISIBLE_SIBLING_FILES`` that exist
          next to reference.py (currently: input.py, shapes.json)

    Sibling files that are missing are skipped silently — synthetic test
    references that only ship reference.py still work.

    Explicitly NOT staged: ``metadata.json``, ``roofline.json``, and the
    legacy ``metadata.yaml`` — those carry agent-leakage-sensitive content
    (upstream provenance / roofline measurements) and stay out of the
    candidate's view by allowlist.
    """

    with tempfile.TemporaryDirectory(prefix="atrex-generate-") as tmp_dir:
        workspace_root = Path(tmp_dir)
        prompt_dir = workspace_root / "prompt"
        input_dir = workspace_root / "input"
        staged_dir = workspace_root / "staged"
        prompt_dir.mkdir()
        input_dir.mkdir()

        staged_template_path = prompt_dir / template_path.name
        staged_reference_path = input_dir / reference_path.name
        shutil.copy2(template_path, staged_template_path)
        shutil.copy2(reference_path, staged_reference_path)

        # Stage constraints.md next to the template: every prompt instructs the
        # agent to "complete the optimization goals defined in constraints.md",
        # so it must be visible in the workspace alongside the prompt.
        constraints_source = template_path.parent / "constraints.md"
        if constraints_source.is_file():
            shutil.copy2(constraints_source, prompt_dir / constraints_source.name)

        seen_paths = {
            template_path.resolve(),
            reference_path.resolve(),
        }

        # Auto-stage allowlisted sibling files (input.py, shapes.json) when
        # they live next to reference.py. Missing ones are silently skipped.
        reference_dir = reference_path.parent
        for sibling_name in _AGENT_VISIBLE_SIBLING_FILES:
            sibling_source = reference_dir / sibling_name
            if not sibling_source.is_file():
                continue
            sibling_resolved = sibling_source.resolve()
            if sibling_resolved in seen_paths:
                continue
            shutil.copy2(sibling_source, input_dir / sibling_name)
            seen_paths.add(sibling_resolved)

        for stage_path in stage_paths or ():
            stage_resolved = stage_path.resolve()
            if stage_resolved in seen_paths:
                continue
            if not stage_path.exists():
                raise FileNotFoundError(f"Whitelist stage path not found: {stage_path}")
            destination_path = staged_dir / _workspace_relative_stage_path(stage_path)
            _copy_stage_entry(stage_path, destination_path)
            seen_paths.add(stage_resolved)

        yield GenerationAccess(
            workspace_root=workspace_root,
            template_path=staged_template_path,
            reference_path=staged_reference_path,
        )


@contextmanager
def _resolve_trace_paths(
    trace_path: Path | None,
    stderr_path: Path | None,
) -> Iterator[tuple[Path, Path]]:
    """Yield writable trace/stderr paths, using a tempdir for any None."""

    if trace_path is not None and stderr_path is not None:
        yield trace_path, stderr_path
        return
    with tempfile.TemporaryDirectory(prefix="atrex-trace-") as tmp:
        tmp_path = Path(tmp)
        yield (
            trace_path if trace_path is not None else tmp_path / "trace.log",
            stderr_path if stderr_path is not None else tmp_path / "stderr.log",
        )


def _claude_tools(skill_mode: bool) -> str:
    """Return the comma-separated --tools allowlist for the current run.

    Default mode keeps the anti-hack surface (no Skill / Task / Web*).
    Skill mode appends ``Skill,Task`` so an installed skill (e.g.
    gpu-kernel-optimizer) can be invoked and spawn its prescribed
    subagents. Web tools stay excluded by design; see CLAUDE_SKILL_EXTRA_TOOLS
    for rationale.
    """
    if skill_mode:
        return f"{CLAUDE_TOOLS},{CLAUDE_SKILL_EXTRA_TOOLS}"
    return CLAUDE_TOOLS


def _run_claude(
    prompt: str,
    model: str | None = None,
    access: GenerationAccess | None = None,
    *,
    trace_path: Path | None = None,
    stderr_path: Path | None = None,
    mirror_to_stdout: bool = False,
    skill_mode: bool = False,
) -> GenerationRun:
    """Invoke ``claude -p --output-format stream-json`` and capture artifacts.

    The prompt is sent via stdin to avoid shell argument-length limits. The
    ``stream-json`` format emits one JSON event per line on stdout, matching
    the JSONL trace shape codex produces; the final assistant text is
    extracted from the terminal ``result`` event. When *access* is provided,
    Claude runs inside a temporary workspace that only contains staged
    copies of the generation inputs.

    Stdout/stderr are streamed to disk line-by-line (so external tailing sees
    progress in real time). When *trace_path* / *stderr_path* are omitted, a
    short-lived tempdir is used. When *mirror_to_stdout* is true, each stdout
    line is also echoed to ``sys.stdout``.

    When *skill_mode* is true, ``Skill`` and ``Task`` are appended to the
    ``--tools`` allowlist so installed Claude Code skills can be invoked and
    spawn their prescribed subagents.
    """
    cmd: list[str] = [
        "claude",
        "--dangerously-skip-permissions",
        "--effort",
        CLAUDE_EFFORT,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--tools",
        _claude_tools(skill_mode),
    ]
    if model:
        cmd.extend(["--model", model])
    cwd = access.workspace_root if access is not None else None
    with _resolve_trace_paths(trace_path, stderr_path) as (trace, stderr):
        returncode = _stream_subprocess(
            cmd,
            stdin_input=prompt,
            cwd=cwd,
            trace_path=trace,
            stderr_path=stderr,
            mirror_to_stdout=mirror_to_stdout,
        )
        trace_text = trace.read_text(encoding="utf-8")
        stderr_text = stderr.read_text(encoding="utf-8")
    return GenerationRun(
        cli="claude",
        model=model,
        command=cmd,
        returncode=returncode,
        output=_extract_claude_final_text(trace_text),
        stderr=stderr_text,
        trace=trace_text,
        trace_format="jsonl",
    )


def _extract_claude_final_text(stream_jsonl: str) -> str:
    """Return the assistant's final text from a claude stream-json transcript.

    Each line in ``stream_jsonl`` is one event. The terminal ``result`` event
    carries the final assembled text in its ``result`` field; if the stream
    is truncated and no ``result`` event arrives, the last assistant text
    block is used as a fallback.
    """
    final_text = ""
    last_assistant_text = ""
    for line in stream_jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "result":
            return event.get("result", "") or ""
        if event_type == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    last_assistant_text = block.get("text", "")
    return final_text or last_assistant_text


def call_claude(
    prompt: str,
    model: str | None = None,
    access: GenerationAccess | None = None,
    *,
    skill_mode: bool = False,
) -> str:
    """Invoke ``claude -p --output-format text`` and return the generated text."""

    run = _run_claude(prompt, model=model, access=access, skill_mode=skill_mode)
    _raise_for_generation_failure(run)
    return run.output


def _codex_sandbox_mode(skill_mode: bool) -> str:
    """Return the Codex --sandbox value for the current run.

    Both default and skill mode use ``danger-full-access`` so the agent's
    shell inherits the host mount namespace and can see GPU device nodes
    (``/dev/kfd``, ``/dev/dri/*``) needed for in-generation kernel JIT/launch
    verification, matching Claude's default unsandboxed subprocess model.
    See the module-level comments on ``CODEX_SANDBOX_MODE`` for the full
    rationale; the skill_mode parameter is retained for symmetry with the
    Claude side (which still toggles its tool allowlist).
    """
    return CODEX_SKILL_SANDBOX_MODE if skill_mode else CODEX_SANDBOX_MODE


def _run_codex(
    prompt: str,
    model: str | None = None,
    access: GenerationAccess | None = None,
    *,
    trace_path: Path | None = None,
    stderr_path: Path | None = None,
    mirror_to_stdout: bool = False,
    skill_mode: bool = False,
) -> GenerationRun:
    """Invoke ``codex exec`` and capture generation artifacts.

    The task prompt is sent via stdin to avoid argument-length limits. When
    *access* is provided, Codex runs inside a temporary workspace that only
    contains staged copies of the generation inputs. ``--output-last-message``
    keeps progress output out of the generated Python source that we persist to
    disk.

    Stdout/stderr are streamed to disk line-by-line. *trace_path* /
    *stderr_path* / *mirror_to_stdout* behave the same as in :func:`_run_claude`.

    ``--sandbox`` is always ``danger-full-access`` (see
    ``CODEX_SANDBOX_MODE`` for the rationale) so the agent can see host GPU
    devices during generation. *skill_mode* no longer changes the sandbox
    tier — it is forwarded to ``_codex_sandbox_mode`` for symmetry only.
    """
    with tempfile.NamedTemporaryFile(mode="r+", encoding="utf-8") as output_file:
        cmd: list[str] = [
            "codex",
            "-a",
            CODEX_APPROVAL_POLICY,
            "exec",
        ]
        if access is not None:
            cmd.extend(
                [
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--cd",
                    str(access.workspace_root),
                ]
            )
        cmd.extend(
            [
                "-c",
                CODEX_REASONING_CONFIG,
                "--sandbox",
                _codex_sandbox_mode(skill_mode),
                "--json",
                "--color",
                "never",
                "--output-last-message",
                output_file.name,
            ]
        )
        if model:
            cmd.extend(["--model", model])
        cmd.append("-")
        cwd = access.workspace_root if access is not None else None
        with _resolve_trace_paths(trace_path, stderr_path) as (trace, stderr):
            returncode = _stream_subprocess(
                cmd,
                stdin_input=prompt,
                cwd=cwd,
                trace_path=trace,
                stderr_path=stderr,
                mirror_to_stdout=mirror_to_stdout,
            )
            trace_text = trace.read_text(encoding="utf-8")
            stderr_text = stderr.read_text(encoding="utf-8")
        output_file.seek(0)
        return GenerationRun(
            cli="codex",
            model=model,
            command=cmd,
            returncode=returncode,
            output=output_file.read(),
            stderr=stderr_text,
            trace=trace_text,
            trace_format="jsonl",
        )


def call_codex(
    prompt: str,
    model: str | None = None,
    access: GenerationAccess | None = None,
    *,
    skill_mode: bool = False,
) -> str:
    """Invoke ``codex exec`` and return the generated text."""

    run = _run_codex(prompt, model=model, access=access, skill_mode=skill_mode)
    _raise_for_generation_failure(run)
    return run.output


def run_generator_cli(
    prompt: str,
    cli: str = "claude",
    model: str | None = None,
    access: GenerationAccess | None = None,
    *,
    trace_path: Path | None = None,
    stderr_path: Path | None = None,
    mirror_to_stdout: bool = False,
    skill_mode: bool = False,
) -> GenerationRun:
    """Invoke the selected generation CLI and return process artifacts."""

    common = {
        "model": model,
        "access": access,
        "trace_path": trace_path,
        "stderr_path": stderr_path,
        "mirror_to_stdout": mirror_to_stdout,
        "skill_mode": skill_mode,
    }
    if cli == "claude":
        return _run_claude(prompt, **common)
    if cli == "codex":
        return _run_codex(prompt, **common)
    raise ValueError(f"Unsupported generator CLI: {cli}")


def call_generator_cli(
    prompt: str,
    cli: str = "claude",
    model: str | None = None,
    access: GenerationAccess | None = None,
    *,
    trace_path: Path | None = None,
    stderr_path: Path | None = None,
    mirror_to_stdout: bool = False,
    skill_mode: bool = False,
) -> str:
    """Invoke the selected generation CLI and return the generated text."""
    run = run_generator_cli(
        prompt,
        cli=cli,
        model=model,
        access=access,
        trace_path=trace_path,
        stderr_path=stderr_path,
        mirror_to_stdout=mirror_to_stdout,
        skill_mode=skill_mode,
    )
    _raise_for_generation_failure(run)
    return run.output


def generate_kernel(
    template_path: Path,
    reference_path: Path,
    output_path: Path,
    cli: str = "claude",
    model: str | None = None,
    stage_paths: Iterable[Path] | None = None,
    *,
    mirror_to_stdout: bool = False,
    repo_root: Path | None = None,
    skill_mode: bool = False,
) -> Path:
    """End-to-end: render prompt, call the selected CLI, save the kernel.

    Args:
        template_path: Path to prompt template.
        reference_path: Path to reference.py.
        output_path: Where to write the generated kernel file.
        cli: Generator CLI to use (``claude`` or ``codex``).
        model: Optional model name understood by the selected CLI.
        stage_paths: Optional whitelist of extra files/directories to copy into
            the isolated temp workspace exposed to the generator CLI.
        mirror_to_stdout: If true, also echo the trace stream to ``sys.stdout``
            while the generator runs (useful for foreground monitoring).
        repo_root: Optional repo root used to relativize prompt-source paths
            recorded in the bundle. Falls back to absolute paths when omitted.
        skill_mode: When true, the generator CLI is invoked with the
            permissions a skill-driven workflow needs — for Claude this
            appends ``Skill,Task`` to the ``--tools`` allowlist so installed
            skills (e.g. gpu-kernel-optimizer) can be invoked and spawn
            subagents. Codex always runs with ``--sandbox
            danger-full-access`` regardless of this flag (see
            ``CODEX_SANDBOX_MODE``). Default is false.

    Returns:
        *output_path* after writing.

    Side effects:
        Writes the candidate kernel, the streamed ``generated_kernel.trace.log``
        JSONL sidecar, and the unified ``generation.json`` bundle next to
        *output_path*. See ``docs/generation_schema.md`` for the bundle schema.
    """
    prompt_text = render_prompt(template_path, reference_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = get_generation_trace_path(output_path)
    bundle_path = get_generation_bundle_path(output_path)
    op_name = reference_path.parent.name
    backend = _backend_from_template(template_path)

    started_at = _utc_now_iso()
    started_monotonic = datetime.now(timezone.utc)
    candidate_text = ""
    with stage_generation_access(
        template_path,
        reference_path,
        stage_paths=stage_paths,
    ) as access:
        run = run_generator_cli(
            prompt_text,
            cli=cli,
            model=model,
            access=access,
            trace_path=trace_path,
            mirror_to_stdout=mirror_to_stdout,
            skill_mode=skill_mode,
        )
        # The prompt instructs the agent to write generated_kernel.py into
        # cwd (= access.workspace_root). Read it before the temp workspace
        # is torn down by the context manager.
        workspace_kernel = access.workspace_root / GENERATED_KERNEL_FILENAME
        if workspace_kernel.is_file():
            candidate_text = workspace_kernel.read_text(encoding="utf-8")
    completed_monotonic = datetime.now(timezone.utc)
    completed_at = completed_monotonic.strftime("%Y-%m-%dT%H:%M:%SZ")
    duration_s = (completed_monotonic - started_monotonic).total_seconds()

    if candidate_text:
        output_path.write_text(candidate_text)

    bundle = _build_generation_bundle(
        op_name=op_name,
        backend=backend,
        output_path=output_path,
        run=run,
        trace_path=trace_path,
        template_path=template_path,
        reference_path=reference_path,
        stage_paths=stage_paths,
        repo_root=repo_root,
        prompt_text=prompt_text,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        skill_mode=skill_mode,
    )
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")

    _raise_for_generation_failure(run)
    return output_path
