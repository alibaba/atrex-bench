"""Tests for the public Atrex-Bench Python SDK."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_package_exports_run_eval_sdk_api() -> None:
    import atrex_bench

    assert callable(atrex_bench.evaluate)
    assert issubclass(atrex_bench.AtrexConfigError, ValueError)
    assert issubclass(
        atrex_bench.AtrexEvaluationError,
        atrex_bench.AtrexSDKError,
    )


def _valid_sdk_config(tmp_path: Path) -> dict[str, object]:
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("class Model:\n    pass\n", encoding="utf-8")
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    for filename in ("reference.py", "input.py", "shapes.json", "metadata.json"):
        (reference_dir / filename).write_text("{}\n", encoding="utf-8")
    return {
        "schema_version": "v1",
        "eval_mode": "candidate",
        "validation_mode": "correctness_only",
        "input": candidate_path,
        "reference_dir": reference_dir,
        "output": tmp_path / "output",
    }


def test_evaluate_rejects_non_mapping_config() -> None:
    from atrex_bench import AtrexConfigError, evaluate

    with pytest.raises(AtrexConfigError, match="Mapping"):
        evaluate("runner.json")  # type: ignore[arg-type]


def test_evaluate_normalizes_pathlike_values_without_mutating_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench import evaluate, sdk

    config = _valid_sdk_config(tmp_path)
    original = dict(config)
    captured: list[dict[str, object]] = []
    expected = {"eval_mode": "candidate", "error": None}
    monkeypatch.setattr(
        sdk,
        "_run_evaluation_process",
        lambda normalized: captured.append(normalized) or expected,
        raising=False,
    )

    result = evaluate(config)

    assert result is expected
    assert config == original
    assert captured[0]["input"] == str(original["input"])
    assert captured[0]["reference_dir"] == str(original["reference_dir"])
    assert captured[0]["output"] == str(original["output"])


def test_evaluate_keeps_relative_checkpoint_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench import evaluate, sdk

    config = _valid_sdk_config(tmp_path)
    config["checkpoint_dir"] = "checkpoints"
    (tmp_path / "checkpoints").write_text("unrelated cwd file", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    expected = {"error": None}
    monkeypatch.setattr(sdk, "_run_evaluation_process", lambda normalized: expected)

    assert evaluate(config) is expected


def test_evaluate_rejects_unwritable_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench import AtrexConfigError, evaluate, sdk

    config = _valid_sdk_config(tmp_path)
    monkeypatch.setattr(sdk.os, "access", lambda path, mode: False)
    monkeypatch.setattr(
        sdk,
        "_run_evaluation_process",
        lambda normalized: pytest.fail("unwritable output reached run_eval"),
    )

    with pytest.raises(AtrexConfigError, match="output parent is not writable"):
        evaluate(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config.pop("input"), "input is required"),
        (
            lambda config: config.update(eval_mode="torch_compile_reference", input="x.py"),
            "input cannot be set",
        ),
        (lambda config: config.update(reference_dir="missing"), "reference_dir"),
        (lambda config: config.update(output=""), "output must be a non-empty path"),
        (lambda config: config.update(extra=object()), "JSON-compatible"),
    ],
)
def test_evaluate_rejects_invalid_launch_config(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    from atrex_bench import AtrexConfigError, evaluate

    config = _valid_sdk_config(tmp_path)
    mutate(config)

    with pytest.raises(AtrexConfigError, match=message):
        evaluate(config)


class _FakeProcess:
    def __init__(self, *, returncode: int, stderr: str = "") -> None:
        self.pid = 12345
        self.returncode = returncode
        self.stderr = io.BytesIO(stderr.encode())
        self.waited = False

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def _install_fake_run_eval_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: object,
    returncode: int,
    stderr: str = "",
    pointer_target: Path | None = None,
    raw_result: str | None = None,
) -> None:
    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        assert command[:3] == [sys.executable, "-m", "atrex_bench.cli.run_eval"]
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.PIPE
        config_path = Path(command[command.index("--config") + 1])
        pointer_path = Path(command[command.index("--sdk-result-path-output") + 1])
        assert config_path.stat().st_mode & 0o777 == 0o600
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result_path = pointer_target or (
            Path(config["output"]) / "stamp" / "reference" / "eval_result.json"
        )
        if payload is not None or raw_result is not None:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                raw_result if raw_result is not None else json.dumps(payload),
                encoding="utf-8",
            )
            pointer_path.write_text(str(result_path.resolve()), encoding="utf-8")
        return _FakeProcess(returncode=returncode, stderr=stderr)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


@pytest.mark.parametrize("returncode", [0, 1])
def test_evaluate_returns_exact_eval_result_for_success_and_stage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    returncode: int,
) -> None:
    from atrex_bench import evaluate

    config = _valid_sdk_config(tmp_path)
    payload = {
        "eval_mode": "candidate",
        "error": None if returncode == 0 else "compile failed",
        "passed": {"compile": {"0": {"status": "passed"}}},
    }
    _install_fake_run_eval_process(
        monkeypatch,
        payload=payload,
        returncode=returncode,
        stderr="[eval] progress\n",
    )

    result = evaluate(config)

    assert result == payload
    assert "[eval] progress" in capsys.readouterr().err


def test_evaluate_raises_config_error_when_cli_produces_no_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench import AtrexConfigError, evaluate

    config = _valid_sdk_config(tmp_path)
    _install_fake_run_eval_process(
        monkeypatch,
        payload=None,
        returncode=1,
        stderr="Unsupported runner config key(s): extra\n",
    )

    with pytest.raises(AtrexConfigError, match="Unsupported runner config"):
        evaluate(config)


def test_evaluate_rejects_result_pointer_outside_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench import AtrexEvaluationError, evaluate

    config = _valid_sdk_config(tmp_path)
    _install_fake_run_eval_process(
        monkeypatch,
        payload={"error": None},
        returncode=0,
        pointer_target=tmp_path / "outside" / "eval_result.json",
    )

    with pytest.raises(AtrexEvaluationError, match="outside output"):
        evaluate(config)


@pytest.mark.parametrize(
    ("raw_result", "message"),
    [
        ("not-json", "valid JSON"),
        ("[]", "JSON object"),
    ],
)
def test_evaluate_rejects_malformed_eval_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_result: str,
    message: str,
) -> None:
    from atrex_bench import AtrexEvaluationError, evaluate

    config = _valid_sdk_config(tmp_path)
    _install_fake_run_eval_process(
        monkeypatch,
        payload=None,
        returncode=0,
        raw_result=raw_result,
    )

    with pytest.raises(AtrexEvaluationError, match=message):
        evaluate(config)


def test_evaluate_wraps_process_launch_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench import AtrexEvaluationError, evaluate

    config = _valid_sdk_config(tmp_path)

    def fail_to_launch(*args: object, **kwargs: object) -> _FakeProcess:
        raise OSError("cannot execute Python")

    monkeypatch.setattr(subprocess, "Popen", fail_to_launch)

    with pytest.raises(AtrexEvaluationError, match="cannot execute Python"):
        evaluate(config)


def test_evaluate_surfaces_real_cli_config_validation_error(tmp_path: Path) -> None:
    from atrex_bench import AtrexConfigError, evaluate

    config = _valid_sdk_config(tmp_path)
    config["unknown_key"] = True

    with pytest.raises(AtrexConfigError, match="Unsupported runner config key"):
        evaluate(config)


def test_evaluate_isolates_cli_stdout_and_accepts_partial_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from atrex_bench import AtrexConfigError, evaluate

    config = _valid_sdk_config(tmp_path)
    process = _FakeProcess(
        returncode=1,
        stderr="invalid config without a trailing newline",
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(AtrexConfigError, match="without a trailing newline"):
        evaluate(config)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "invalid config without a trailing newline"
    assert process.waited is True


def test_evaluate_propagates_interrupt_after_stopping_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench import evaluate, sdk

    config = _valid_sdk_config(tmp_path)
    stopped: list[_FakeProcess] = []

    class InterruptingProcess(_FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt

    process = InterruptingProcess(returncode=0)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(sdk, "_stop_process", stopped.append)

    with pytest.raises(KeyboardInterrupt):
        evaluate(config)

    assert stopped == [process]
