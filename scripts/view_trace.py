"""Render a generator trace JSONL file in human-friendly form.

Accepts either the raw `*.trace.log` JSONL sidecar or a `generation.json`
bundle; in the latter case, the bundle's `trace.path` (resolved relative to
the bundle's directory) is what gets rendered.

Auto-detects the trace flavor (claude stream-json or codex --json) and
streams events with short ASCII labels:

  [init]    session metadata
  [msg]     agent message text
  [tool]    tool/command invocation
  [result]  tool/command output
  [file]    file change
  [done]    terminal usage / cost summary

Usage:

    python scripts/view_trace.py outputs/<op>/<ts>/generated_kernel.trace.log
    python scripts/view_trace.py outputs/<op>/<ts>/generation.json
    python scripts/view_trace.py <path> --no-follow
    python scripts/view_trace.py <path> --cli claude   # force flavor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

PREVIEW_CHARS = 200
POLL_INTERVAL_S = 0.2


def _truncate(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = text.rstrip("\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"... (truncated, {len(text)} chars total)"


def _detect_flavor(event: dict) -> str | None:
    etype = event.get("type", "")
    if etype in {"system", "assistant", "user", "result"}:
        return "claude"
    if etype.startswith(("thread.", "turn.", "item.")):
        return "codex"
    return None


def _render_claude(event: dict) -> Iterator[str]:
    etype = event.get("type")
    if etype == "system":
        if event.get("subtype") == "init":
            model = event.get("model", "?")
            tools = event.get("tools") or []
            cwd = event.get("cwd", "?")
            yield f"[init]   model={model} cwd={cwd} tools={len(tools)}"
        return
    if etype == "assistant":
        for block in event.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "").strip()
                if text:
                    yield f"[msg]    {_truncate(text)}"
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input") or {}
                summary = _summarize_claude_tool_input(name, inp)
                yield f"[tool]   {name}: {summary}"
        return
    if etype == "user":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") != "tool_result":
                continue
            content = block.get("content")
            text = _claude_tool_result_text(content)
            is_err = block.get("is_error", False)
            tag = "[result]" if not is_err else "[error] "
            yield f"{tag} {_truncate(text)}"
        return
    if etype == "result":
        duration_ms = event.get("duration_ms")
        usage = event.get("usage") or {}
        cost = event.get("total_cost_usd")
        bits = []
        if duration_ms is not None:
            bits.append(f"{duration_ms / 1000:.1f}s")
        if usage:
            in_tok = usage.get("input_tokens", 0)
            cached = usage.get("cache_read_input_tokens", 0) + usage.get(
                "cache_creation_input_tokens", 0
            )
            out_tok = usage.get("output_tokens", 0)
            bits.append(f"in={in_tok} cached={cached} out={out_tok}")
        if cost is not None:
            bits.append(f"${cost:.4f}")
        yield f"[done]   {' '.join(bits) if bits else 'completed'}"
        return


def _summarize_claude_tool_input(name: str, inp: dict) -> str:
    if name in {"Bash", "BashOutput"}:
        return _truncate(str(inp.get("command", "")))
    if name in {"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"}:
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        return _truncate(str(path))
    if name in {"Glob", "Grep"}:
        return _truncate(str(inp.get("pattern", "")))
    if name == "WebFetch":
        return _truncate(str(inp.get("url", "")))
    return _truncate(json.dumps(inp, ensure_ascii=False))


def _claude_tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                parts.append(chunk.get("text", ""))
        return "\n".join(parts)
    return ""


def _render_codex(event: dict) -> Iterator[str]:
    etype = event.get("type")
    if etype == "thread.started":
        yield f"[init]   thread={event.get('thread_id', '?')}"
        return
    if etype == "turn.started":
        yield "[init]   turn started"
        return
    if etype in {"item.started", "item.completed"}:
        item = event.get("item") or {}
        itype = item.get("type")
        # Only render completed items to avoid duplicate noise; show started
        # only for command_execution so a long-running command surfaces.
        if etype == "item.started" and itype != "command_execution":
            return
        if itype == "agent_message":
            text = (item.get("text") or "").strip()
            if text:
                yield f"[msg]    {_truncate(text)}"
        elif itype == "command_execution":
            cmd = item.get("command", "")
            status = item.get("status")
            if etype == "item.started":
                yield f"[tool]   {_truncate(str(cmd))}"
            else:
                exit_code = item.get("exit_code")
                output = item.get("aggregated_output", "")
                tag = "[result]" if exit_code == 0 else "[error] "
                trail = f"  (exit_code={exit_code})" if exit_code is not None else ""
                yield f"{tag} {_truncate(output)}{trail}"
        elif itype == "file_change":
            for change in item.get("changes") or []:
                kind = change.get("kind", "?")
                path = change.get("path", "?")
                yield f"[file]   {kind} {path}"
        return
    if etype == "turn.completed":
        usage = event.get("usage") or {}
        bits = []
        if usage:
            bits.append(
                "in={inp} cached={cached} out={out}".format(
                    inp=usage.get("input_tokens", 0),
                    cached=usage.get("cached_input_tokens", 0),
                    out=usage.get("output_tokens", 0),
                )
            )
        yield f"[done]   {' '.join(bits) if bits else 'completed'}"
        return


def _is_terminal(event: dict, flavor: str) -> bool:
    if flavor == "claude":
        return event.get("type") == "result"
    if flavor == "codex":
        return event.get("type") == "turn.completed"
    return False


def _iter_lines(path: Path, follow: bool) -> Iterator[str]:
    """Yield trace lines, optionally tailing the file as it grows."""

    while not path.exists():
        if not follow:
            raise FileNotFoundError(path)
        time.sleep(POLL_INTERVAL_S)
    with path.open("r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read()
            if chunk:
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    yield line
                continue
            if not follow:
                if buffer.strip():
                    yield buffer
                return
            time.sleep(POLL_INTERVAL_S)


def render(path: Path, *, follow: bool, force_cli: str | None) -> int:
    flavor = force_cli
    for raw in _iter_lines(path, follow=follow):
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if flavor is None:
            flavor = _detect_flavor(event)
            if flavor is None:
                continue
        renderer = _render_claude if flavor == "claude" else _render_codex
        for line in renderer(event):
            print(line, flush=True)
        if follow and _is_terminal(event, flavor):
            return 0
    return 0


def _resolve_input_path(path: Path) -> Path:
    """If *path* points at a generation.json bundle, return its trace sidecar."""

    if path.suffix != ".json":
        return path
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path
    trace_section = bundle.get("trace") or {}
    sidecar = trace_section.get("path")
    if not isinstance(sidecar, str):
        return path
    candidate = (path.parent / sidecar).resolve()
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a generator trace JSONL file in human-friendly form"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a *.trace.log JSONL file or a generation.json bundle",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Read once and exit instead of tailing the file as it grows",
    )
    parser.add_argument(
        "--cli",
        choices=["auto", "claude", "codex"],
        default="auto",
        help="Force the trace flavor (default: auto-detect)",
    )
    args = parser.parse_args()
    force = None if args.cli == "auto" else args.cli
    target = _resolve_input_path(args.path)
    try:
        sys.exit(render(target, follow=not args.no_follow, force_cli=force))
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError as exc:
        print(f"trace file not found: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
