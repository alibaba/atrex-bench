"""Public Python SDK for the run_eval pipeline."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from codecs import getincrementaldecoder
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

_PATH_CONFIG_KEYS = frozenset(
    {"input", "reference_dir", "output", "checkpoint_dir"}
)
_REFERENCE_FILENAMES = (
    "reference.py",
    "input.py",
    "shapes.json",
    "metadata.json",
)
_EVAL_MODE_CANDIDATE = "candidate"
_EVAL_MODE_TORCH_COMPILE = "torch_compile_reference"
_EVAL_MODES = frozenset({_EVAL_MODE_CANDIDATE, _EVAL_MODE_TORCH_COMPILE})
_SDK_PROCESS_SHUTDOWN_TIMEOUT_S = 5.0
_SDK_STDERR_TAIL_LINES = 50
_SDK_STDERR_CHUNK_BYTES = 64 * 1024
_SDK_STDERR_PARTIAL_LINE_CHARS = 64 * 1024


class AtrexSDKError(Exception):
    """Base class for Python SDK failures without an eval result."""


class AtrexConfigError(AtrexSDKError, ValueError):
    """Raised when an SDK config cannot start a valid evaluation."""


class AtrexEvaluationError(AtrexSDKError, RuntimeError):
    """Raised when the evaluation process cannot produce a result payload."""


def evaluate(config: Mapping[str, object]) -> dict[str, object]:
    """Run one evaluation and return its ``eval_result.json`` payload."""

    normalized = _normalize_config(config)
    _validate_launch_paths(normalized)
    return _run_evaluation_process(normalized)


def _normalize_config(config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise AtrexConfigError("config must be a Mapping[str, object]")
    try:
        normalized = dict(config)
    except Exception as error:
        raise AtrexConfigError(f"failed to read config Mapping: {error}") from error
    if any(not isinstance(key, str) for key in normalized):
        raise AtrexConfigError("config keys must be strings")

    for name in _PATH_CONFIG_KEYS.intersection(normalized):
        value = normalized[name]
        if value is None:
            continue
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if not isinstance(value, str) or not value.strip():
            raise AtrexConfigError(
                f"{name} must be a non-empty path string or null"
            )
        # Launch paths belong to the caller's cwd, not a worker's package cwd.
        # Relative checkpoints deliberately remain relative to the run artifact.
        normalized[name] = value if name == "checkpoint_dir" else str(Path(value).resolve())

    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AtrexConfigError(
            f"config values must be JSON-compatible: {error}"
        ) from error
    return normalized


def _validate_launch_paths(config: Mapping[str, object]) -> None:
    eval_mode = config.get("eval_mode", _EVAL_MODE_CANDIDATE)
    if not isinstance(eval_mode, str) or eval_mode not in _EVAL_MODES:
        allowed = ", ".join(sorted(_EVAL_MODES))
        raise AtrexConfigError(f"eval_mode must be one of: {allowed}")

    reference_dir = _required_path(config, "reference_dir")
    if not reference_dir.is_dir():
        raise AtrexConfigError(f"reference_dir does not exist: {reference_dir}")
    missing = [
        filename
        for filename in _REFERENCE_FILENAMES
        if not (reference_dir / filename).is_file()
    ]
    if missing:
        raise AtrexConfigError(
            "reference_dir is missing required file(s): " + ", ".join(missing)
        )

    output = _required_path(config, "output")
    if output.exists() and not output.is_dir():
        raise AtrexConfigError(f"output must be a directory path: {output}")
    output_parent = _nearest_existing_parent(output)
    if not output_parent.is_dir() or not os.access(
        output_parent,
        os.W_OK | os.X_OK,
    ):
        raise AtrexConfigError(
            f"output parent is not writable: {output_parent}"
        )

    checkpoint = _optional_path(config, "checkpoint_dir")
    if (
        checkpoint is not None
        and checkpoint.is_absolute()
        and checkpoint.exists()
        and not checkpoint.is_dir()
    ):
        raise AtrexConfigError(
            f"checkpoint_dir must be a directory path: {checkpoint}"
        )

    input_path = _optional_path(config, "input")
    if eval_mode == _EVAL_MODE_TORCH_COMPILE:
        if input_path is not None:
            raise AtrexConfigError(
                "input cannot be set for eval_mode=torch_compile_reference"
            )
        return
    if input_path is None:
        raise AtrexConfigError("input is required for eval_mode=candidate")
    if not input_path.is_file():
        raise AtrexConfigError(f"input does not exist: {input_path}")
    if input_path.suffix != ".py":
        raise AtrexConfigError(f"input must be a Python file: {input_path}")


def _required_path(config: Mapping[str, object], name: str) -> Path:
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AtrexConfigError(f"{name} must be a non-empty path string")
    return Path(value)


def _optional_path(config: Mapping[str, object], name: str) -> Path | None:
    value = config.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AtrexConfigError(f"{name} must be a non-empty path string or null")
    return Path(value)


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _run_evaluation_process(config: Mapping[str, object]) -> dict[str, object]:
    output_root = _required_path(config, "output").resolve()
    with tempfile.TemporaryDirectory(prefix="atrex-bench-sdk-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        config_path = temporary_root / "run_eval.json"
        pointer_path = temporary_root / "eval_result_path.txt"
        _write_private_config(config, config_path)
        command = [
            sys.executable,
            "-m",
            "atrex_bench.cli.run_eval",
            "--config",
            str(config_path),
            "--sdk-result-path-output",
            str(pointer_path),
        ]
        process = _start_process(command)
        if process.stderr is None:
            _stop_process(process)
            raise AtrexEvaluationError(
                "run_eval process did not expose a stderr stream"
            )
        stderr_tail: deque[str] = deque(maxlen=_SDK_STDERR_TAIL_LINES)
        reader_errors: list[OSError] = []
        stderr_reader = threading.Thread(
            target=_stream_stderr,
            args=(process.stderr, stderr_tail, reader_errors),
            name="atrex-sdk-stderr",
            daemon=True,
        )
        stderr_reader.start()
        try:
            returncode = process.wait()
        except BaseException:
            _stop_process(process)
            raise
        finally:
            stderr_reader.join(timeout=_SDK_PROCESS_SHUTDOWN_TIMEOUT_S)
            process.stderr.close()

        if stderr_reader.is_alive():
            raise AtrexEvaluationError(
                "run_eval stderr reader did not stop after process exit"
            )
        if reader_errors:
            raise AtrexEvaluationError(
                f"failed to read run_eval stderr: {reader_errors[0]}"
            )
        stderr = "".join(stderr_tail).strip()
        if returncode not in {0, 1}:
            raise AtrexEvaluationError(
                _process_failure_message(returncode, stderr)
            )
        if not pointer_path.is_file():
            if returncode == 1 and "Traceback (most recent call last)" not in stderr:
                raise AtrexConfigError(
                    stderr or "run_eval rejected the SDK config"
                )
            raise AtrexEvaluationError(
                _process_failure_message(returncode, stderr)
            )
        return _load_eval_result(pointer_path, output_root)


def _stream_stderr(
    stream: BinaryIO,
    tail: deque[str],
    errors: list[OSError],
) -> None:
    decoder = getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    try:
        while chunk := stream.read(_SDK_STDERR_CHUNK_BYTES):
            text = decoder.decode(chunk)
            _mirror_stderr(text)
            pending = _append_stderr_tail(tail, pending, text)
        final_text = decoder.decode(b"", final=True)
        _mirror_stderr(final_text)
        pending = _append_stderr_tail(tail, pending, final_text)
    except OSError as error:
        errors.append(error)
    finally:
        if pending:
            tail.append(pending)


def _append_stderr_tail(
    tail: deque[str],
    pending: str,
    text: str,
) -> str:
    lines = (pending + text).splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        pending = lines.pop()
    else:
        pending = ""
    tail.extend(lines)
    return pending[-_SDK_STDERR_PARTIAL_LINE_CHARS:]


def _mirror_stderr(stderr: str) -> None:
    if not stderr:
        return
    try:
        sys.stderr.write(stderr)
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def _write_private_config(
    config: Mapping[str, object],
    config_path: Path,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(config_path, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                config,
                handle,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
    except OSError as error:
        raise AtrexEvaluationError(
            f"failed to write temporary run_eval config: {error}"
        ) from error


def _start_process(command: list[str]) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise AtrexEvaluationError(
            f"failed to launch run_eval process: {error}"
        ) from error


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGINT)
        else:  # pragma: no cover - production workers run on Linux.
            process.terminate()
        process.wait(timeout=_SDK_PROCESS_SHUTDOWN_TIMEOUT_S)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - production workers run on Linux.
            process.terminate()
        process.wait(timeout=_SDK_PROCESS_SHUTDOWN_TIMEOUT_S)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - production workers run on Linux.
            process.kill()
        process.wait(timeout=_SDK_PROCESS_SHUTDOWN_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return


def _process_failure_message(returncode: int, stderr: str) -> str:
    message = f"run_eval exited with code {returncode} without a valid result"
    return f"{message}: {stderr}" if stderr else message


def _load_eval_result(
    pointer_path: Path,
    output_root: Path,
) -> dict[str, object]:
    try:
        raw_pointer = pointer_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise AtrexEvaluationError(
            f"failed to read run_eval result pointer: {error}"
        ) from error
    pointed_path = Path(raw_pointer)
    if not raw_pointer or not pointed_path.is_absolute():
        raise AtrexEvaluationError(
            "run_eval result pointer must contain an absolute path"
        )
    eval_result_path = pointed_path.resolve()
    if eval_result_path.name != "eval_result.json":
        raise AtrexEvaluationError(
            f"run_eval result pointer is not eval_result.json: {eval_result_path}"
        )
    try:
        eval_result_path.relative_to(output_root)
    except ValueError as error:
        raise AtrexEvaluationError(
            f"run_eval result path is outside output root: {eval_result_path}"
        ) from error
    try:
        payload = json.loads(eval_result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AtrexEvaluationError(
            f"eval_result.json is not valid JSON: {error}"
        ) from error
    except OSError as error:
        raise AtrexEvaluationError(
            f"failed to read eval_result.json: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise AtrexEvaluationError("eval_result.json must contain a JSON object")
    return payload


__all__ = [
    "AtrexConfigError",
    "AtrexEvaluationError",
    "AtrexSDKError",
    "evaluate",
]
