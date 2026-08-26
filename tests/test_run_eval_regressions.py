"""Path and process-lifecycle regressions; these tests do not require a GPU."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from atrex_bench.cli import run_eval as runner

FIXTURES = Path(__file__).parent / "fixtures"
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")


def _stage_inputs(tmp_path):
    shutil.copytree(FIXTURES / "references" / "atrex_001", tmp_path / "reference")
    shutil.copy2(FIXTURES / "generations" / "atrex_001.py", tmp_path / "candidate.py")


@pytest.mark.parametrize("public_entry", [False, True])
@pytest.mark.parametrize("torch_compile", [False, True])
def test_launch_paths_resolved_before_artifacts_and_worker(
    tmp_path,
    monkeypatch,
    public_entry,
    torch_compile,
):
    _stage_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "fixed-timestamp" / "reference"
    commands = []

    def fake_worker(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["cwd"] != str(tmp_path)
        for flag, expected in (
            ("--reference-dir", tmp_path / "reference"),
            ("--output", tmp_path / "artifacts"),
            ("--artifact-dir", artifact_dir),
        ):
            assert cmd[cmd.index(flag) + 1] == str(expected.resolve())
        if not torch_compile:
            assert cmd[cmd.index("--input") + 1] == str((tmp_path / "candidate.py").resolve())
        assert cmd[cmd.index("--checkpoint-dir") + 1] == "checkpoints"
        # A failed worker still uses the parent's single artifact tree.
        return subprocess.CompletedProcess(cmd, 1, "", "simulated worker failure")

    real_clock_policy = runner._evaluate_with_clock_policy

    def check_clock_paths(**kwargs):
        assert kwargs["artifact_dir"] == artifact_dir.resolve()
        assert kwargs["eval_output_path"] == artifact_dir.resolve() / "eval_result.json"
        return real_clock_policy(**kwargs)

    monkeypatch.setattr(runner, "_run_subprocess_with_live_stderr", fake_worker)
    monkeypatch.setattr(runner, "_evaluate_with_clock_policy", check_clock_paths)
    if torch_compile:
        entry = (
            runner.run_torch_compile_eval
            if public_entry
            else runner._run_torch_compile_eval_process
        )
        candidate_args = {}
    else:
        entry = runner.run_eval if public_entry else runner._run_eval_process
        candidate_args = {"input_path": Path("candidate.py")}
    entry(
        **candidate_args,
        reference_dir=Path("reference"),
        output_root=Path("artifacts"),
        checkpoint_dir=Path("checkpoints"),
        timestamp="fixed-timestamp",
    )
    assert len(commands) == 1
    assert list((tmp_path / "artifacts").rglob("eval_result.json")) == [
        artifact_dir / "eval_result.json"
    ]
    assert (
        runner._resolve_checkpoint_root(artifact_dir, Path("checkpoints"))
        == artifact_dir / "checkpoints"
    )


@pytest.mark.parametrize("entry", ["cli", "sdk"])
def test_real_launch_with_relative_paths_reads_candidate(tmp_path, monkeypatch, entry):
    _stage_inputs(tmp_path)
    (tmp_path / "candidate.py").write_text("def broken(\n", encoding="utf-8")
    config = {
        "input": "candidate.py",
        "reference_dir": "reference",
        "output": "artifacts",
        "checkpoint_dir": "checkpoints",
        "validation_mode": "correctness_only",
    }
    monkeypatch.chdir(tmp_path)
    if entry == "sdk":
        from atrex_bench import evaluate

        result = evaluate(config)
    else:
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "runner.json").write_text(json.dumps(config), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "atrex_bench.cli.run_eval", "--config", "configs/runner.json"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode != 0  # Deliberate compile error, not a missing relative file.
        result_files = list((tmp_path / "artifacts").rglob("eval_result.json"))
        assert len(result_files) == 1, completed.stderr
        result = json.loads(result_files[0].read_text())
    assert "SyntaxError" in json.dumps(result)
    assert "FileNotFoundError" not in json.dumps(result)
    assert len(list((tmp_path / "artifacts").rglob("eval_result.json"))) == 1


def _shape_kwargs(tmp_path, shape_id, torch_compile):
    common = dict(
        reference_dir=tmp_path / "reference",
        artifact_dir=tmp_path / "artifacts",
        checkpoint_root=tmp_path / "checkpoints",
        shape_results_dir=tmp_path / "run" / ".shape_results",
        shape_id=shape_id,
        warmup_iters=1,
        bench_iters=1,
    )
    if torch_compile:
        return common | {"shape_wall_timeout_s": 0.3}
    return common | dict(
        candidate_path=tmp_path / "candidate.py",
        atol=0.01,
        rtol=0.05,
        num_correctness_cases=1,
        collect_kernel_events=False,
        candidate_timeout_s=0.1,
        perf_timeout_s=0.1,
    )


@pytest.mark.parametrize("torch_compile", [False, True])
@pytest.mark.parametrize("shape_id", ["../../target", "absolute", "a/b", "a\\b", "形状 1"])
def test_shape_id_cannot_delete_or_overwrite_outside_results(
    tmp_path,
    monkeypatch,
    torch_compile,
    shape_id,
):
    outside = tmp_path / "target.json"
    outside.write_text("do not touch", encoding="utf-8")
    if shape_id == "absolute":
        shape_id = str(outside.with_suffix(""))
    kwargs = _shape_kwargs(tmp_path, shape_id, torch_compile)
    kwargs["shape_results_dir"].mkdir(parents=True)
    expected_path = runner._shape_result_path(kwargs["shape_results_dir"], shape_id)
    expected_path.write_text("stale result", encoding="utf-8")
    paths = []

    def fake_worker(cmd, **_kwargs):
        result_path = Path(cmd[cmd.index("--shape-result-output") + 1])
        paths.append(result_path)
        assert result_path.parent == kwargs["shape_results_dir"]
        assert not result_path.exists(), "stale result must be removed only inside .shape_results"
        assert cmd[cmd.index("--shape-id") + 1] == shape_id
        assert outside.read_text() == "do not touch"
        result_path.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1, "", "simulated failure")

    monkeypatch.setattr(runner, "_run_subprocess_with_live_stderr", fake_worker)
    entry = (
        runner._run_single_shape_torch_compile_subprocess
        if torch_compile
        else runner._run_single_shape_subprocess
    )
    result = entry(**kwargs)
    assert "simulated failure" in (result[1].error if torch_compile else result[0].reason)
    assert paths == [expected_path]
    assert outside.read_text() == "do not touch"


def test_shape_result_names_do_not_collide_after_sanitization(tmp_path):
    ids = ["a/b", "a\\b", "a_b", "../a_b", "形状", ""]
    paths = [runner._shape_result_path(tmp_path, shape_id) for shape_id in ids]
    assert len(set(paths)) == len(ids)
    assert all(path.parent == tmp_path and len(path.stem) == 64 for path in paths)


def _assert_stopped(pid):
    # A killed orphan can briefly remain a zombie until its reaper collects it.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return
        time.sleep(0.02)
    pytest.fail(f"worker descendant {pid} is still running")


def _compiler_tree_command(pid_path, *, exit_leader=False):
    return [
        sys.executable,
        "-c",
        "import os,pathlib,subprocess,sys,time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
        "print('partial stdout', flush=True)\n"
        "print('partial stderr', file=sys.stderr, flush=True)\n"
        + ("" if exit_leader else "time.sleep(30)\n"),
    ]


@POSIX_ONLY
def test_dead_leader_inherited_pipes_still_obey_deadline(tmp_path):
    pid_path = tmp_path / "compiler.pid"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as exc:
        runner._run_subprocess_with_live_stderr(
            _compiler_tree_command(pid_path, exit_leader=True),
            cwd=str(tmp_path),
            timeout_s=0.2,
        )
    assert time.monotonic() - started < 1.5
    assert "partial stdout" in exc.value.stdout
    assert "partial stderr" in exc.value.stderr
    _assert_stopped(int(pid_path.read_text()))


@POSIX_ONLY
@pytest.mark.parametrize("torch_compile", [False, True])
def test_per_shape_timeout_kills_compiler_descendants(tmp_path, monkeypatch, torch_compile):
    pid_path = tmp_path / "compiler.pid"
    builder = (
        "_build_single_shape_torch_compile_worker_command"
        if torch_compile
        else "_build_single_shape_worker_command"
    )
    monkeypatch.setattr(runner, builder, lambda **kwargs: _compiler_tree_command(pid_path))
    monkeypatch.setattr(runner, "_derived_shape_wall_timeout_s", lambda *args, **kwargs: 0.3)
    entry = (
        runner._run_single_shape_torch_compile_subprocess
        if torch_compile
        else runner._run_single_shape_subprocess
    )
    result = entry(**_shape_kwargs(tmp_path, "0", torch_compile))
    assert "wall-clock budget" in (result[1].error if torch_compile else result[0].reason)
    _assert_stopped(int(pid_path.read_text()))


@POSIX_ONLY
def test_top_level_timeout_cleans_separately_grouped_shape_worker(tmp_path):
    pid_path = tmp_path / "compiler.pid"
    shape_command = _compiler_tree_command(pid_path)
    cmd = [
        sys.executable,
        "-c",
        "from atrex_bench.cli.run_eval import _run_subprocess_with_live_stderr as run\n"
        f"run({shape_command!r}, cwd={str(tmp_path)!r}, timeout_s=30)\n",
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        runner._run_subprocess_with_live_stderr(
            cmd,
            cwd=str(tmp_path),
            timeout_s=3,
            terminate_grace_s=runner._SUBPROCESS_TERMINATE_GRACE_S,
        )
    assert pid_path.exists(), "nested worker must have started before the outer timeout"
    _assert_stopped(int(pid_path.read_text()))


@POSIX_ONLY
def test_pipe_reader_stops_even_if_descendant_escapes_group(tmp_path, monkeypatch):
    pid_path = tmp_path / "escaped.pid"
    # Cleanup time is separate from the evaluation budget, but must be bounded.
    monkeypatch.setattr(runner, "_SUBPROCESS_SHUTDOWN_TIMEOUT_S", 0.2)
    cmd = [
        sys.executable,
        "-c",
        "import pathlib,subprocess,sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True)\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n",
    ]
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            runner._run_subprocess_with_live_stderr(cmd, cwd=str(tmp_path), timeout_s=0.2)
        assert time.monotonic() - started < 1.5
    finally:
        # Process groups cannot contain a deliberately detached descendant.
        if pid_path.exists():
            os.kill(int(pid_path.read_text()), signal.SIGKILL)


@POSIX_ONLY
@pytest.mark.parametrize("cancel_signal", [signal.SIGINT, signal.SIGTERM])
def test_cancellation_cleans_nested_shape_group(tmp_path, cancel_signal):
    pid_path = tmp_path / "compiler.pid"
    shape_command = _compiler_tree_command(pid_path)
    worker_command = [
        sys.executable,
        "-c",
        "from atrex_bench.cli.run_eval import _run_subprocess_with_live_stderr as run\n"
        f"run({shape_command!r}, cwd={str(tmp_path)!r}, timeout_s=30)\n",
    ]
    parent_command = [
        sys.executable,
        "-c",
        "from atrex_bench.cli.run_eval import _run_subprocess_with_live_stderr as run\n"
        f"run({worker_command!r}, cwd={str(tmp_path)!r}, terminate_grace_s=1)\n",
    ]
    process = subprocess.Popen(
        parent_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not pid_path.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.02)
        assert pid_path.exists(), "nested compiler must be running before cancellation"
        process.send_signal(cancel_signal)
        process.wait(timeout=5)
        _assert_stopped(int(pid_path.read_text()))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_pipe_reader_decodes_partial_utf8_and_final_unterminated_line(tmp_path):
    completed = runner._run_subprocess_with_live_stderr(
        [
            sys.executable,
            "-c",
            "import os,time\n"
            "os.write(1, b'\\xe4'); os.write(2, b'prefix ')\n"
            "time.sleep(0.05)\n"
            "os.write(1, b'\\xb8\\xad'); os.write(2, b'no newline')\n",
        ],
        cwd=str(tmp_path),
        timeout_s=2,
    )
    assert completed.stdout == "中"
    assert completed.stderr == "prefix no newline"
