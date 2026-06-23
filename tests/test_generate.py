"""Tests for prompt rendering and generation CLI dispatch."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from atrex_bench.generate import (
    CLAUDE_EFFORT,
    CLAUDE_SKILL_EXTRA_TOOLS,
    CLAUDE_TOOLS,
    CODEX_APPROVAL_POLICY,
    CODEX_REASONING_CONFIG,
    CODEX_SANDBOX_MODE,
    CODEX_SKILL_SANDBOX_MODE,
    GENERATION_BUNDLE_FILENAME,
    GenerationAccess,
    GenerationRun,
    _run_claude,
    _run_codex,
    call_claude,
    call_codex,
    call_generator_cli,
    generate_kernel,
    get_generation_bundle_path,
    get_generation_trace_path,
    render_prompt,
    stage_generation_access,
)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _install_fake_stream(monkeypatch, recorder: dict, stdout_text: str = "", returncode: int = 0):
    """Patch ``_stream_subprocess`` to write *stdout_text* to ``trace_path``."""

    def fake_stream(
        cmd,
        *,
        stdin_input,
        cwd,
        trace_path: Path,
        stderr_path: Path,
        mirror_to_stdout,
    ):
        recorder["cmd"] = cmd
        recorder["stdin_input"] = stdin_input
        recorder["cwd"] = cwd
        recorder["trace_path"] = trace_path
        recorder["stderr_path"] = stderr_path
        recorder["mirror_to_stdout"] = mirror_to_stdout
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return returncode

    monkeypatch.setattr("atrex_bench.generate._stream_subprocess", fake_stream)


def test_extract_actual_model_picks_claude_init_event():
    from atrex_bench.generate import _extract_actual_model

    trace = (
        '{"type":"system","subtype":"init","model":"claude-opus-4-7[1m]"}\n'
        '{"type":"result","subtype":"success","result":"x"}\n'
    )
    assert _extract_actual_model(trace, "claude") == "claude-opus-4-7[1m]"


def test_extract_actual_model_returns_none_for_codex():
    from atrex_bench.generate import _extract_actual_model

    # Codex JSONL has no model field anywhere; helper returns None and the
    # bundle builder falls back to run.model (which we always set for codex).
    trace = (
        '{"type":"thread.started","thread_id":"abc"}\n'
        '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
    )
    assert _extract_actual_model(trace, "codex") is None


def test_generate_kernel_records_codex_model_via_run_model_fallback(monkeypatch, tmp_path):
    """Codex's JSONL has no model field; bundle must fall back to run.model."""

    op_dir = tmp_path / "fused_moe"
    op_dir.mkdir()
    template_path = tmp_path / "triton" / "generate_kernel.md"
    template_path.parent.mkdir()
    reference_path = op_dir / "reference.py"
    output_path = tmp_path / "out" / "generated_kernel.py"
    template_path.write_text("{{REFERENCE_CODE}}")
    reference_path.write_text("print('ref')\n")

    codex_trace = (
        '{"type":"thread.started","thread_id":"abc"}\n'
        '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}\n'
    )

    def fake_run(
        prompt, cli, model, access=None, *, trace_path, stderr_path=None, mirror_to_stdout,
        **_kwargs,
    ):
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(codex_trace, encoding="utf-8")
        (access.workspace_root / "generated_kernel.py").write_text("print('ok')\n")
        return GenerationRun(
            cli=cli, model=model,
            command=["codex", "exec", "--model", model or "", "-"],
            returncode=0, output="", stderr="", trace=codex_trace, trace_format="jsonl",
        )

    monkeypatch.setattr("atrex_bench.generate.run_generator_cli", fake_run)
    generate_kernel(
        template_path, reference_path, output_path,
        cli="codex", model="gpt-5.4-0305-global",
    )
    bundle = json.loads(get_generation_bundle_path(output_path).read_text())
    assert bundle["agent"]["cli"] == "codex"
    assert bundle["agent"]["model"] == "gpt-5.4-0305-global"


def test_extract_actual_model_handles_empty_or_garbled_trace():
    from atrex_bench.generate import _extract_actual_model

    assert _extract_actual_model("", "claude") is None
    assert _extract_actual_model("not json\n{also-bad}\n", "claude") is None


def test_render_prompt_replaces_reference_code(tmp_path):
    template_path = tmp_path / "template.txt"
    reference_path = tmp_path / "reference.py"
    template_path.write_text("before\n{{REFERENCE_CODE}}\nafter\n")
    reference_path.write_text("print('kernel')\n")

    prompt = render_prompt(template_path, reference_path)

    # render_prompt substitutes the reference PATH (the agent reads the staged
    # file), not its inline content — see generate.py:render_prompt.
    assert prompt == f"before\n{reference_path}\nafter\n"


def test_flydsl_prompt_allows_internal_mixed_precision():
    # The flydsl prompt delegates to constraints.md, where this guidance now lives.
    prompt_path = Path("prompt/flydsl/constraints.md")
    text = prompt_path.read_text(encoding="utf-8")

    assert "returned tensor dtype" in text
    assert "Internal computation precision" in text
    assert "including fp8/fp4/int8 paths" in text
    assert "not as mandatory implementation choices" in text
    assert "dtype semantics" not in text


def test_flydsl_prompt_states_general_performance_goal():
    # The flydsl prompt delegates to constraints.md, where this guidance now lives.
    prompt_path = Path("prompt/flydsl/constraints.md")
    text = prompt_path.read_text(encoding="utf-8")

    assert "Performance expectations:" in text
    assert "This is an optimization task, not only a functional translation task." in text
    assert (
        "The generated implementation will be evaluated on compile success, "
        "numerical correctness, and runtime performance."
    ) in text
    assert "Among correct implementations, faster implementations are better." in text


def test_stage_generation_access_stages_whitelist_file_and_directory(tmp_path):
    template_path = tmp_path / "template.md"
    reference_path = tmp_path / "reference.py"
    helper_file = tmp_path / "scripts" / "helper.py"
    helper_dir = tmp_path / "fixtures"
    helper_dir_file = helper_dir / "sample.txt"

    template_path.write_text("{{REFERENCE_CODE}}", encoding="utf-8")
    reference_path.write_text("print('ref')\n", encoding="utf-8")
    helper_file.parent.mkdir()
    helper_file.write_text("print('helper')\n", encoding="utf-8")
    helper_dir.mkdir()
    helper_dir_file.write_text("fixture\n", encoding="utf-8")

    with stage_generation_access(
        template_path,
        reference_path,
        stage_paths=[helper_file, helper_dir],
    ) as access:
        staged_root = access.workspace_root / "staged"

        assert access.template_path.read_text(encoding="utf-8") == "{{REFERENCE_CODE}}"
        assert access.reference_path.read_text(encoding="utf-8") == "print('ref')\n"
        assert sorted(p.name for p in access.workspace_root.iterdir()) == [
            "input",
            "prompt",
            "staged",
        ]
        assert next(staged_root.rglob("helper.py")).read_text(encoding="utf-8") == (
            "print('helper')\n"
        )
        assert next(staged_root.rglob("sample.txt")).read_text(encoding="utf-8") == "fixture\n"


def test_stage_generation_access_auto_stages_sibling_input_and_shapes(tmp_path):
    """input.py and shapes.json next to reference.py auto-stage; metadata/roofline don't."""
    template_path = tmp_path / "template.md"
    op_dir = tmp_path / "op_dir"
    op_dir.mkdir()
    reference_path = op_dir / "reference.py"
    input_path = op_dir / "input.py"
    shapes_path = op_dir / "shapes.json"
    metadata_path = op_dir / "metadata.json"
    roofline_path = op_dir / "roofline.json"

    template_path.write_text("{{REFERENCE_CODE}}", encoding="utf-8")
    reference_path.write_text("# ref\n", encoding="utf-8")
    input_path.write_text("# input\n", encoding="utf-8")
    shapes_path.write_text('{"0":{}}\n', encoding="utf-8")
    metadata_path.write_text('{"id":"x"}\n', encoding="utf-8")
    roofline_path.write_text('{}\n', encoding="utf-8")

    with stage_generation_access(template_path, reference_path) as access:
        staged_input_dir = access.workspace_root / "input"
        staged_names = sorted(p.name for p in staged_input_dir.iterdir())
        assert staged_names == ["input.py", "reference.py", "shapes.json"]
        # metadata.json and roofline.json must NOT leak into the workspace
        assert not (staged_input_dir / "metadata.json").exists()
        assert not (staged_input_dir / "roofline.json").exists()


def test_stage_generation_access_skips_missing_siblings(tmp_path):
    """Synthetic references with no sibling files still stage cleanly."""
    template_path = tmp_path / "template.md"
    reference_path = tmp_path / "reference.py"
    template_path.write_text("{{REFERENCE_CODE}}", encoding="utf-8")
    reference_path.write_text("# ref\n", encoding="utf-8")

    with stage_generation_access(template_path, reference_path) as access:
        staged_input_dir = access.workspace_root / "input"
        # Only reference.py — no error raised even though sibling files are absent.
        assert sorted(p.name for p in staged_input_dir.iterdir()) == ["reference.py"]


def test_call_claude_builds_expected_command(monkeypatch):
    recorded: dict[str, object] = {}
    stream_jsonl = (
        '{"type":"system","subtype":"init"}\n'
        '{"type":"assistant","message":{"content":[{"type":"text","text":"generated"}]}}\n'
        '{"type":"result","subtype":"success","result":"generated"}\n'
    )
    _install_fake_stream(monkeypatch, recorded, stdout_text=stream_jsonl)

    output = call_claude("PROMPT", model="claude-sonnet-4-6")

    assert output == "generated"
    assert recorded["stdin_input"] == "PROMPT"
    assert recorded["cwd"] is None
    assert recorded["mirror_to_stdout"] is False
    assert recorded["cmd"] == [
        "claude",
        "--dangerously-skip-permissions",
        "--effort",
        CLAUDE_EFFORT,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--tools",
        CLAUDE_TOOLS,
        "--model",
        "claude-sonnet-4-6",
    ]


def test_call_codex_builds_expected_command(monkeypatch):
    recorded: dict[str, object] = {}

    def fake_stream(
        cmd,
        *,
        stdin_input,
        cwd,
        trace_path: Path,
        stderr_path: Path,
        mirror_to_stdout,
    ):
        recorded["cmd"] = cmd
        recorded["stdin_input"] = stdin_input
        recorded["cwd"] = cwd
        recorded["mirror_to_stdout"] = mirror_to_stdout
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("generated")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr("atrex_bench.generate._stream_subprocess", fake_stream)

    output = call_codex("PROMPT", model="gpt-5-codex")

    assert output == "generated"
    assert recorded["stdin_input"] == "PROMPT"
    assert recorded["cwd"] is None
    assert recorded["mirror_to_stdout"] is False
    cmd = recorded["cmd"]
    assert cmd[:13] == [
        "codex",
        "-a",
        CODEX_APPROVAL_POLICY,
        "exec",
        "-c",
        CODEX_REASONING_CONFIG,
        "--sandbox",
        CODEX_SANDBOX_MODE,
        "--json",
        "--color",
        "never",
        "--output-last-message",
        cmd[12],
    ]
    assert cmd[13:] == ["--model", "gpt-5-codex", "-"]
    # Pin the reasoning tier explicitly. The kernel-generation task needs
    # deep reasoning to actually attempt FlyDSL/CUTeDSL kernels instead of
    # falling back to a "safe" pure-PyTorch grouped path; anyone lowering
    # this should have to update the test on purpose.
    assert CODEX_REASONING_CONFIG == 'model_reasoning_effort="xhigh"'


def test_call_claude_skill_mode_appends_extra_tools(monkeypatch):
    """skill_mode=True must extend --tools with Skill,Task; default stays restricted."""
    recorded: dict[str, object] = {}
    stream_jsonl = (
        '{"type":"system","subtype":"init"}\n'
        '{"type":"result","subtype":"success","result":"ok"}\n'
    )
    _install_fake_stream(monkeypatch, recorded, stdout_text=stream_jsonl)

    call_claude("PROMPT", model="claude-sonnet-4-6", skill_mode=True)

    cmd = recorded["cmd"]
    tools_idx = cmd.index("--tools") + 1
    expected = f"{CLAUDE_TOOLS},{CLAUDE_SKILL_EXTRA_TOOLS}"
    assert cmd[tools_idx] == expected
    # Sanity-check: default mode stays restricted.
    recorded.clear()
    _install_fake_stream(monkeypatch, recorded, stdout_text=stream_jsonl)
    call_claude("PROMPT", model="claude-sonnet-4-6")
    cmd = recorded["cmd"]
    tools_idx = cmd.index("--tools") + 1
    assert cmd[tools_idx] == CLAUDE_TOOLS


def test_call_codex_uses_danger_full_access_sandbox(monkeypatch):
    """Codex must always run with --sandbox danger-full-access so the agent's
    shell inherits the host mount namespace (and therefore sees /dev/kfd,
    /dev/dri/*, GPU, network, ...). workspace-write would remount /dev as a
    fresh tmpfs and break in-generation FlyDSL/CUTeDSL JIT verification.

    skill_mode no longer affects the sandbox tier; both branches resolve to
    danger-full-access. The flag is still threaded through for symmetry with
    the Claude tool-allowlist toggle.
    """
    recorded: dict[str, object] = {}

    def fake_stream(
        cmd,
        *,
        stdin_input,
        cwd,
        trace_path: Path,
        stderr_path: Path,
        mirror_to_stdout,
    ):
        recorded["cmd"] = cmd
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("ok")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr("atrex_bench.generate._stream_subprocess", fake_stream)

    call_codex("PROMPT", model="gpt-5-codex", skill_mode=True)
    cmd = recorded["cmd"]
    sandbox_idx = cmd.index("--sandbox") + 1
    assert cmd[sandbox_idx] == CODEX_SKILL_SANDBOX_MODE
    assert cmd[sandbox_idx] == "danger-full-access"

    # Default mode must also resolve to danger-full-access (so the agent can
    # see /dev/kfd / GPU during generation, matching the Claude baseline).
    recorded.clear()
    monkeypatch.setattr("atrex_bench.generate._stream_subprocess", fake_stream)
    call_codex("PROMPT", model="gpt-5-codex")
    cmd = recorded["cmd"]
    sandbox_idx = cmd.index("--sandbox") + 1
    assert cmd[sandbox_idx] == CODEX_SANDBOX_MODE
    assert cmd[sandbox_idx] == "danger-full-access"


def test_run_claude_uses_restricted_workspace_when_access_provided(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    access = GenerationAccess(
        workspace_root=workspace_root,
        template_path=workspace_root / "prompt.md",
        reference_path=workspace_root / "reference.py",
    )
    stream_jsonl = (
        '{"type":"system","subtype":"init"}\n'
        '{"type":"result","subtype":"success","result":"generated"}\n'
    )
    _install_fake_stream(monkeypatch, recorded, stdout_text=stream_jsonl)

    run = _run_claude("PROMPT", model="claude-sonnet-4-6", access=access)

    assert run.output == "generated"
    assert run.trace == stream_jsonl
    assert run.trace_format == "jsonl"
    assert recorded["cwd"] == workspace_root
    cmd = recorded["cmd"]
    assert "--bare" not in cmd
    assert "--disable-slash-commands" not in cmd
    assert "--debug-file" not in cmd
    assert "--verbose" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"


def test_run_codex_uses_restricted_workspace_when_access_provided(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    access = GenerationAccess(
        workspace_root=workspace_root,
        template_path=workspace_root / "prompt.md",
        reference_path=workspace_root / "reference.py",
    )

    def fake_stream(
        cmd,
        *,
        stdin_input,
        cwd,
        trace_path: Path,
        stderr_path: Path,
        mirror_to_stdout,
    ):
        recorded["cmd"] = cmd
        recorded["cwd"] = cwd
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("generated")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr("atrex_bench.generate._stream_subprocess", fake_stream)

    run = _run_codex("PROMPT", model="gpt-5-codex", access=access)

    assert run.output == "generated"
    assert recorded["cwd"] == workspace_root
    cmd = recorded["cmd"]
    assert "--skip-git-repo-check" in cmd
    assert "--ephemeral" in cmd
    assert cmd[cmd.index("--cd") + 1] == str(workspace_root)


def test_call_generator_cli_rejects_unknown_cli():
    with pytest.raises(ValueError, match="Unsupported generator CLI"):
        call_generator_cli("PROMPT", cli="unknown")


def test_generate_kernel_writes_bundle_and_trace(monkeypatch, tmp_path):
    op_dir = tmp_path / "fused_moe"
    op_dir.mkdir()
    template_path = tmp_path / "triton" / "generate_kernel.md"
    template_path.parent.mkdir()
    reference_path = op_dir / "reference.py"
    output_path = tmp_path / "out" / "generated_kernel.py"
    template_path.write_text("{{REFERENCE_CODE}}")
    reference_path.write_text("print('ref')\n")

    # Note: the system.init event reports a model id ("claude-opus-4-7[1m]")
    # that differs from the value we'll pass via --model below. The bundle
    # must prefer the trace-reported actual model over the requested one.
    trace_payload = (
        '{"type":"system","subtype":"init","model":"claude-opus-4-7[1m]"}\n'
        '{"type":"assistant","message":{"content":['
        '{"type":"text","text":"hi"},{"type":"tool_use","name":"Bash","input":{"command":"ls"}}'
        ']}}\n'
        '{"type":"user","message":{"content":[{"type":"tool_result","is_error":false,"content":"ok"}]}}\n'
        '{"type":"result","subtype":"success","result":"print(\'generated\')\\n",'
        '"duration_ms":1234,"total_cost_usd":0.05,'
        '"usage":{"input_tokens":100,"cache_read_input_tokens":80,'
        '"cache_creation_input_tokens":10,"output_tokens":7}}\n'
    )
    recorded: dict[str, object] = {}

    def fake_run(
        prompt: str,
        cli: str,
        model: str | None,
        access: GenerationAccess | None = None,
        *,
        trace_path: Path,
        stderr_path: Path | None = None,
        mirror_to_stdout: bool,
        **_kwargs,
    ):
        recorded["prompt"] = prompt
        recorded["cli"] = cli
        recorded["model"] = model
        recorded["mirror_to_stdout"] = mirror_to_stdout
        recorded["trace_path"] = trace_path
        recorded["stderr_path"] = stderr_path
        assert access is not None
        recorded["workspace_entries"] = sorted(p.name for p in access.workspace_root.iterdir())
        recorded["template_text"] = access.template_path.read_text()
        recorded["reference_text"] = access.reference_path.read_text()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(trace_payload)
        # Per the new contract, the agent writes generated_kernel.py into cwd.
        (access.workspace_root / "generated_kernel.py").write_text("print('generated')\n")
        return GenerationRun(
            cli=cli,
            model=model,
            command=["claude", "-p"],
            returncode=0,
            output="done",  # chat text only; not the kernel source
            stderr="",
            trace=trace_payload,
            trace_format="jsonl",
        )

    monkeypatch.setattr("atrex_bench.generate.run_generator_cli", fake_run)

    result = generate_kernel(
        template_path,
        reference_path,
        output_path,
        cli="claude",
        model="claude-opus-4-7",
        repo_root=tmp_path,
    )

    assert result == output_path
    assert output_path.read_text() == "print('generated')\n"
    assert recorded["mirror_to_stdout"] is False
    assert recorded["stderr_path"] is None  # generate_kernel no longer sidecars stderr
    assert get_generation_trace_path(output_path).read_text() == trace_payload

    bundle_path = get_generation_bundle_path(output_path)
    assert bundle_path.name == GENERATION_BUNDLE_FILENAME
    bundle = json.loads(bundle_path.read_text())

    assert bundle["kernel"] == {
        "name": "fused_moe",
        "backend": "triton",
        "path": "generated_kernel.py",
        "sha256": _sha256_hex("print('generated')\n"),
    }
    assert bundle["agent"]["cli"] == "claude"
    # Trace-extracted model wins over the value passed via --model.
    assert bundle["agent"]["model"] == "claude-opus-4-7[1m]"
    assert bundle["agent"]["command"] == ["claude", "-p"]
    assert bundle["agent"]["returncode"] == 0
    assert bundle["prompt"]["template_path"] == "triton/generate_kernel.md"
    assert bundle["prompt"]["reference_path"] == "fused_moe/reference.py"
    assert bundle["prompt"]["stage_paths"] == []
    assert bundle["prompt"]["text"] == str(reference_path)
    assert bundle["trace"]["format"] == "claude-stream-json"
    assert bundle["trace"]["path"] == "generated_kernel.trace.log"
    assert bundle["trace"]["sha256"] == _sha256_hex(trace_payload)
    assert bundle["trace"]["byte_size"] == len(trace_payload.encode("utf-8"))
    assert bundle["trace"]["event_count"] == 4
    assert bundle["trace"]["summary"] == {
        "agent_messages": 1,
        "tool_uses": 1,
        "file_changes": 0,
        "errors": 0,
    }
    assert bundle["usage"] == {
        "input_tokens": 100,
        "cached_input_tokens": 90,
        "output_tokens": 7,
        "duration_ms": 1234,
        "cost_usd": 0.05,
    }
    assert bundle["error"] is None
    assert isinstance(bundle["duration_s"], float)
    assert bundle["started_at"].endswith("Z")
    assert bundle["completed_at"].endswith("Z")
    # bundle never references the dropped sidecars
    assert "stderr" not in bundle
    assert not output_path.with_name(f"{output_path.stem}.prompt.txt").exists()
    assert not output_path.with_name(f"{output_path.stem}.stderr.log").exists()
    assert not output_path.with_name(f"{output_path.stem}.generation.json").exists()


def test_generate_kernel_stages_whitelist_paths_for_cli(monkeypatch, tmp_path):
    template_path = tmp_path / "template.txt"
    reference_path = tmp_path / "reference.py"
    helper_file = tmp_path / "scripts" / "check_compile.py"
    helper_dir = tmp_path / "fixtures"
    helper_fixture = helper_dir / "sample.txt"
    output_path = tmp_path / "generated_kernel.py"
    template_path.write_text("{{REFERENCE_CODE}}", encoding="utf-8")
    reference_path.write_text("print('ref')\n", encoding="utf-8")
    helper_file.parent.mkdir()
    helper_file.write_text("print('helper')\n", encoding="utf-8")
    helper_dir.mkdir()
    helper_fixture.write_text("fixture\n", encoding="utf-8")

    recorded: dict[str, object] = {}

    def fake_run(
        prompt: str,
        cli: str,
        model: str | None,
        access: GenerationAccess | None = None,
        *,
        trace_path: Path,
        stderr_path: Path | None = None,
        mirror_to_stdout: bool,
        **_kwargs,
    ):
        assert access is not None
        staged_root = access.workspace_root / "staged"
        recorded["prompt"] = prompt
        recorded["workspace_entries"] = sorted(p.name for p in access.workspace_root.iterdir())
        recorded["staged_helper_text"] = next(staged_root.rglob("check_compile.py")).read_text(
            encoding="utf-8"
        )
        recorded["staged_fixture_text"] = next(staged_root.rglob("sample.txt")).read_text(
            encoding="utf-8"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
        return GenerationRun(
            cli=cli,
            model=model,
            command=["codex", "exec", "-"],
            returncode=0,
            output="print('generated')\n",
            stderr="",
            trace="",
            trace_format="jsonl",
        )

    monkeypatch.setattr("atrex_bench.generate.run_generator_cli", fake_run)

    result = generate_kernel(
        template_path,
        reference_path,
        output_path,
        cli="codex",
        model="gpt-5-codex",
        stage_paths=[helper_file, helper_dir],
    )

    assert result == output_path
    assert recorded == {
        "prompt": str(reference_path),
        "workspace_entries": ["input", "prompt", "staged"],
        "staged_helper_text": "print('helper')\n",
        "staged_fixture_text": "fixture\n",
    }


def test_generate_kernel_writes_bundle_on_failure_with_no_kernel(monkeypatch, tmp_path):
    """returncode != 0 and no output -> bundle exists, kernel.sha256 == null."""

    op_dir = tmp_path / "fused_moe"
    op_dir.mkdir()
    template_path = tmp_path / "triton" / "generate_kernel.md"
    template_path.parent.mkdir()
    reference_path = op_dir / "reference.py"
    output_path = tmp_path / "out" / "generated_kernel.py"
    template_path.write_text("{{REFERENCE_CODE}}")
    reference_path.write_text("print('ref')\n")

    def fake_run(
        prompt: str,
        cli: str,
        model: str | None,
        access: GenerationAccess | None = None,
        *,
        trace_path: Path,
        stderr_path: Path | None = None,
        mirror_to_stdout: bool,
        **_kwargs,
    ):
        assert access is not None
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
        return GenerationRun(
            cli=cli,
            model=model,
            command=["claude", "-p"],
            returncode=23,
            output="",
            stderr="",
            trace="",
            trace_format="jsonl",
        )

    monkeypatch.setattr("atrex_bench.generate.run_generator_cli", fake_run)

    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero exit status 23"):
        generate_kernel(
            template_path,
            reference_path,
            output_path,
            cli="claude",
            model="claude-sonnet-4-6",
            repo_root=tmp_path,
        )

    assert not output_path.exists()
    bundle = json.loads(get_generation_bundle_path(output_path).read_text())
    assert bundle["kernel"]["sha256"] is None
    assert bundle["kernel"]["name"] == "fused_moe"
    assert bundle["agent"]["returncode"] == 23
    assert bundle["error"] is None
    assert get_generation_trace_path(output_path).read_text() == ""


def test_generate_kernel_writes_bundle_on_partial_kernel(monkeypatch, tmp_path):
    """returncode != 0 but with output -> kernel still written and sha non-null."""

    op_dir = tmp_path / "fused_moe"
    op_dir.mkdir()
    template_path = tmp_path / "triton" / "generate_kernel.md"
    template_path.parent.mkdir()
    reference_path = op_dir / "reference.py"
    output_path = tmp_path / "out" / "generated_kernel.py"
    template_path.write_text("{{REFERENCE_CODE}}")
    reference_path.write_text("print('ref')\n")

    def fake_run(
        prompt, cli, model, access=None, *, trace_path, stderr_path=None, mirror_to_stdout,
        **_kwargs,
    ):
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
        # Agent wrote the kernel even though it then crashed before exiting cleanly.
        (access.workspace_root / "generated_kernel.py").write_text("partial\n")
        return GenerationRun(
            cli=cli, model=model, command=["claude", "-p"], returncode=1,
            output="", stderr="", trace="", trace_format="jsonl",
        )

    monkeypatch.setattr("atrex_bench.generate.run_generator_cli", fake_run)
    # generate_kernel still raises after recording the bundle.
    with pytest.raises(subprocess.CalledProcessError):
        generate_kernel(template_path, reference_path, output_path, cli="claude")
    assert output_path.exists()
    bundle = json.loads(get_generation_bundle_path(output_path).read_text())
    assert bundle["kernel"]["sha256"] == _sha256_hex("partial\n")


def test_generate_kernel_propagates_mirror_to_stdout(monkeypatch, tmp_path):
    template_path = tmp_path / "template.txt"
    reference_path = tmp_path / "reference.py"
    output_path = tmp_path / "generated_kernel.py"
    template_path.write_text("{{REFERENCE_CODE}}")
    reference_path.write_text("print('ref')\n")

    seen_mirror: list[bool] = []

    def fake_run(
        prompt,
        cli,
        model,
        access=None,
        *,
        trace_path,
        stderr_path=None,
        mirror_to_stdout,
        **_kwargs,
    ):
        seen_mirror.append(mirror_to_stdout)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")
        return GenerationRun(
            cli=cli,
            model=model,
            command=["x"],
            returncode=0,
            output="print('ok')\n",
            stderr="",
            trace="",
            trace_format="jsonl",
        )

    monkeypatch.setattr("atrex_bench.generate.run_generator_cli", fake_run)

    generate_kernel(
        template_path, reference_path, output_path, cli="claude", mirror_to_stdout=True
    )
    generate_kernel(template_path, reference_path, output_path, cli="claude")
    assert seen_mirror == [True, False]


def test_stream_subprocess_writes_trace_before_exit(tmp_path):
    """The trace file must contain output BEFORE the subprocess returns."""

    from atrex_bench.generate import _stream_subprocess

    trace_path = tmp_path / "trace.log"
    stderr_path = tmp_path / "stderr.log"
    # python -u forces unbuffered output so each print lands immediately.
    cmd = [
        "python",
        "-u",
        "-c",
        "import sys, time\n"
        "sys.stdout.write('first\\n'); sys.stdout.flush()\n"
        "time.sleep(0.5)\n"
        "sys.stdout.write('second\\n'); sys.stdout.flush()\n",
    ]
    import threading

    saw_partial = threading.Event()

    def watch():
        for _ in range(50):  # poll up to ~1s
            if trace_path.exists() and "first" in trace_path.read_text():
                if "second" not in trace_path.read_text():
                    saw_partial.set()
                    return
            import time as _time

            _time.sleep(0.02)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    rc = _stream_subprocess(
        cmd,
        stdin_input="",
        cwd=None,
        trace_path=trace_path,
        stderr_path=stderr_path,
        mirror_to_stdout=False,
    )
    watcher.join(timeout=2)

    assert rc == 0
    assert trace_path.read_text() == "first\nsecond\n"
    assert saw_partial.is_set(), (
        "trace file did not show 'first' before 'second' arrived — streaming failed"
    )
