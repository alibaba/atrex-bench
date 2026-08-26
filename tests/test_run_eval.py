"""Tests for end-to-end run_eval pipeline."""

import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atrex_bench.eval.compile import check_compilation
from atrex_bench.eval.correctness import check_correctness

EVAL_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")

# End-to-end run_eval tests require torch.cuda or torch.hip for do_bench timing
_gpu_available = torch.cuda.is_available()

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
REFERENCE_PATH = FIXTURE_ROOT / "references" / "atrex_001" / "reference.py"
INPUT_PATH = FIXTURE_ROOT / "references" / "atrex_001" / "input.py"
SHAPES_PATH = FIXTURE_ROOT / "references" / "atrex_001" / "shapes.json"
METADATA_PATH = FIXTURE_ROOT / "references" / "atrex_001" / "metadata.json"
CANDIDATE_PATH = FIXTURE_ROOT / "generations" / "atrex_001.py"


def _write_candidate_file(tmp_path: Path, name: str, content: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _build_reference_dir(
    tmp_path: Path,
    name: str = "atrex_001",
) -> Path:
    """Stage a reference dir matching the new-schema 4-file contract."""
    reference_dir = tmp_path / name
    reference_dir.mkdir(parents=True)
    shutil.copy2(REFERENCE_PATH, reference_dir / "reference.py")
    shutil.copy2(INPUT_PATH, reference_dir / "input.py")
    shutil.copy2(SHAPES_PATH, reference_dir / "shapes.json")
    shutil.copy2(METADATA_PATH, reference_dir / "metadata.json")
    return reference_dir


def test_end_to_end_stages() -> None:
    """Run stages 0-1 on the real fixture to verify the full path works."""
    compile_result = check_compilation(CANDIDATE_PATH)
    assert compile_result.status == "passed", f"Compile failed: {compile_result.reason}"

    correctness_result = check_correctness(
        REFERENCE_PATH,
        CANDIDATE_PATH,
        num_correctness_cases=2,
        device="cpu",
    )
    assert correctness_result.status == "passed", (
        f"Correctness failed: {correctness_result.reason}"
    )
    first_diff = correctness_result.cases[0].outputs[0]
    assert first_diff.max_elementwise_abs_diff is not None
    assert first_diff.max_elementwise_abs_diff < 1e-6


@pytest.mark.skipif(not _gpu_available, reason="requires CUDA/HIP GPU")
def test_run_eval_single_problem(tmp_path: Path) -> None:
    """Verify run_eval produces the expected eval_result.json structure."""
    from scripts.run_eval import run_eval

    timestamp = "20260410-103000"
    reference_dir = _build_reference_dir(tmp_path, name="operator_from_meta")

    result = run_eval(
        input_path=CANDIDATE_PATH,
        reference_dir=reference_dir,
        output_root=tmp_path,
        warmup_iters=2,
        bench_iters=3,
        num_correctness_cases=2,
        checkpoint_dir=Path("checkpoints"),
        timestamp=timestamp,
    )

    # Top-level shape per docs/data_schema.md Section 7
    assert result["kernel"]["name"] == "operator_from_meta"
    assert result["kernel"]["id"] == "atrex_901"
    assert result["kernel"]["dtype"] == "fp32"
    assert result["dsl"] == "unknown"
    assert "timestamp" not in result
    assert EVAL_ID_PATTERN.match(result["eval_id"]), result["eval_id"]
    assert "environment" in result
    assert "runner_config" in result
    assert result["runner_config"]["config_version"] == "v1"
    assert result["runner_config"]["num_correctness_cases"] == 2
    assert result["runner_config"]["warmup_iters"] == 2
    assert result["runner_config"]["bench_iters"] == 3

    # Environment block (no "device" field per finalised schema)
    env = result["environment"]
    assert "device" not in env
    assert "accelerator_backend" in env
    assert "python_version" in env
    assert "torch_version" in env
    assert "clock_locked" in env

    # Explicit per-stage verdicts for every shape.
    assert "0" in result["passed"]["compile"]
    assert result["passed"]["compile"]["0"]["status"] == "passed"
    assert result["passed"]["compile"]["0"]["reason"] is None
    assert "0" in result["passed"]["correctness"]
    assert result["passed"]["correctness"]["0"]["status"] == "passed"
    assert result["passed"]["performance"]["0"]["status"] == "passed"
    assert result["passed"]["performance"]["0"]["reason"] is None

    # Shape-major correctness/performance
    cases = result["correctness"]["shapes"]["0"]["cases"]
    assert len(cases) == 2
    assert "input_artifact" in cases[0]
    assert isinstance(cases[0]["outputs"], list)
    assert "name" in cases[0]["outputs"][0]
    assert cases[0]["outputs"][0]["name"] == "out"

    samples = result["performance"]["shapes"]["0"]["samples"]
    # do_bench returns one aggregated end-to-end timing per shape (not one
    # per bench iter -- that was the pre-do_bench perf_counter loop's shape).
    assert len(samples) == 1
    assert samples[0]["end_to_end_time_ms"] > 0
    assert "input_artifact" in result["performance"]["shapes"]["0"]

    assert result["error"] is None

    # Persisted file matches in-memory payload
    result_dir = tmp_path / timestamp / "operator_from_meta"
    result_file = result_dir / "eval_result.json"
    candidate_file = result_dir / "candidate.py"
    reference_file = result_dir / "reference.py"
    input_file = result_dir / "input.py"
    shapes_file = result_dir / "shapes.json"
    metadata_file = result_dir / "metadata.json"
    manifest_file = result_dir / "staging_manifest.json"
    assert result_file.exists()
    assert candidate_file.exists()
    assert reference_file.exists()
    assert input_file.exists()
    assert shapes_file.exists()
    assert metadata_file.exists()
    assert manifest_file.exists()
    # Input checkpoints (.pt files) are no longer written; inputs are
    # reproducible from the seed recorded in eval_result.json.
    assert not (result_dir / "checkpoints").exists()

    saved = json.loads(result_file.read_text(encoding="utf-8"))
    assert saved["kernel"]["name"] == "operator_from_meta"
    assert saved["dsl"] == "unknown"
    assert saved["passed"]["compile"]["0"]["status"] == "passed"
    assert saved["passed"]["correctness"]["0"]["status"] == "passed"
    cor_artifact = saved["correctness"]["shapes"]["0"]["cases"][0]["input_artifact"]
    assert cor_artifact["format"] == "manual_seed"
    assert isinstance(cor_artifact["seed"], int)
    assert "path" not in cor_artifact
    perf_artifact = saved["performance"]["shapes"]["0"]["input_artifact"]
    assert perf_artifact["format"] == "manual_seed"
    assert isinstance(perf_artifact["seed"], int)
    assert "path" not in perf_artifact
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["candidate"]["path"] == str(CANDIDATE_PATH)
    assert isinstance(manifest["candidate"]["sha256"], str)
    assert manifest["reference_files"]["reference.py"]["path"] == str(
        reference_dir / "reference.py"
    )
    assert manifest["runner_config"]["warmup_iters"] == 2
    assert manifest["worker_command"]


def test_torch_compile_worker_writes_shape_major_performance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    reference_dir = _build_reference_dir(tmp_path / "tc_case", name="tc_reference")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    def fake_shape_subprocess(**_kwargs):
        return (
            run_eval_module.CorrectnessShapeResult(
                status="skipped",
                reason=run_eval_module._TORCH_COMPILE_SKIP_REASON,
            ),
            run_eval_module.PerformanceShapeResult(
                input_artifact={"seed": 42, "format": "manual_seed"},
                samples=[run_eval_module.PerformanceSample(end_to_end_time_ms=1.23)],
            ),
        )

    monkeypatch.setattr(
        run_eval_module,
        "_run_single_shape_torch_compile_subprocess",
        fake_shape_subprocess,
    )

    payload = run_eval_module._run_torch_compile_worker(
        reference_dir=reference_dir,
        artifact_dir=artifact_dir,
        warmup_iters=1,
        bench_iters=1,
        checkpoint_dir=None,
        config_version="v1",
        clock_locked=False,
    )

    assert payload["eval_mode"] == "torch_compile_reference"
    assert payload["runner_config"]["mode"] == "torch_compile_reference"
    assert payload["runner_config"]["validation_mode"] == "performance_only"
    assert payload["runner_config"]["num_correctness_cases"] == 0
    assert payload["passed"]["compile"]["0"]["status"] == "passed"
    assert payload["passed"]["correctness"]["0"]["status"] == "skipped"
    assert payload["correctness"]["shapes"]["0"]["cases"] == []
    perf_shape = payload["performance"]["shapes"]["0"]
    assert perf_shape["input_artifact"] == {"seed": 42, "format": "manual_seed"}
    assert perf_shape["samples"] == [{"end_to_end_time_ms": 1.23}]
    assert perf_shape["error"] is None
    assert run_eval_module._payload_overall_passed(payload) is True

    saved = json.loads((artifact_dir / "eval_result.json").read_text(encoding="utf-8"))
    assert saved["eval_mode"] == "torch_compile_reference"
    assert saved["performance"]["shapes"]["0"]["samples"] == [
        {"end_to_end_time_ms": 1.23}
    ]


def test_torch_compile_process_records_performance_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    reference_dir = _build_reference_dir(tmp_path / "tc_process", name="tc_process")
    monkeypatch.setattr(
        run_eval_module,
        "_run_subprocess_with_live_stderr",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    payload = run_eval_module._run_torch_compile_eval_process(
        reference_dir=reference_dir,
        output_root=tmp_path / "output",
        warmup_iters=1,
        bench_iters=1,
        timestamp="20260804-000000",
    )

    assert payload["runner_config"]["validation_mode"] == "performance_only"


def test_run_eval_serializes_cuda_graph_metadata() -> None:
    from scripts import run_eval as run_eval_module

    performance = run_eval_module.PerformanceShapeResult(
        benchmark_mode="cuda_graph_replay",
        capture_time_ms=4.25,
        cache_flush_mb=1024,
        graph_correctness={"passed": True, "outputs": []},
    )

    payload = run_eval_module._serialize_performance_shapes({"0": performance})

    assert payload["0"]["benchmark_mode"] == "cuda_graph_replay"
    assert payload["0"]["capture_time_ms"] == 4.25
    assert payload["0"]["cache_flush_mb"] == 1024
    assert payload["0"]["graph_correctness"] == {
        "passed": True,
        "outputs": [],
    }


def test_observed_kernels_attachment_carries_every_field() -> None:
    """Attaching the flydsl tracker payload must not drop other fields.

    The sub-worker used to hand-copy every field here, so a field added to
    ``PerformanceShapeResult`` later was computed, then dropped on the way to
    the parent, with no error anywhere. The dataclass is introspected rather
    than spelled out so this stays honest as fields are added.
    """
    from dataclasses import fields

    from scripts import run_eval as run_eval_module

    populated = run_eval_module.PerformanceShapeResult(
        input_artifact={"0": "input.pt"},
        samples=[run_eval_module.PerformanceSample(end_to_end_time_ms=1.5)],
        kernel_events=[
            run_eval_module.KernelTimingEvent(
                name="kernel", device_time_us=2.0, calls=3
            )
        ],
        benchmark_mode="cuda_graph_replay",
        capture_time_ms=4.25,
        cache_flush_mb=1024,
        graph_correctness={"passed": True, "outputs": []},
        error="boom",
    )

    attached = run_eval_module._with_observed_kernels(populated, {"flydsl": ["k"]})

    assert attached.observed_kernels == {"flydsl": ["k"]}
    carried = [f.name for f in fields(populated) if f.name != "observed_kernels"]
    assert carried
    for name in carried:
        assert getattr(attached, name) == getattr(populated, name), name


def test_performance_payload_round_trip_keeps_every_field() -> None:
    """A performance result must survive the sub-worker JSON channel intact.

    ``_performance_to_payload`` is ``asdict``, so the decoder used to have to
    name every field; one it did not name was computed, serialized, and then
    dropped on the way back, with no error anywhere.
    """
    from scripts import run_eval as run_eval_module

    original = run_eval_module.PerformanceShapeResult(
        input_artifact={"0": "input.pt"},
        samples=[run_eval_module.PerformanceSample(end_to_end_time_ms=1.5)],
        kernel_events=[
            run_eval_module.KernelTimingEvent(
                name="kernel", device_time_us=2.0, calls=3
            )
        ],
        benchmark_mode="cuda_graph_replay",
        capture_time_ms=4.25,
        cache_flush_mb=1024,
        graph_correctness={"passed": True, "outputs": []},
        error="boom",
        observed_kernels={"flydsl": ["k"]},
    )

    restored = run_eval_module._performance_from_payload(
        run_eval_module._performance_to_payload(original)
    )

    assert restored == original


def test_correctness_payload_round_trip_keeps_every_field() -> None:
    """Same for the correctness half, at both the shape and the case level."""
    from scripts import run_eval as run_eval_module

    original = run_eval_module.CorrectnessShapeResult(
        status="failed",
        reason="mismatch",
        cases=[
            run_eval_module.CorrectnessCase(
                input_artifact={"0": "input.pt"},
                outputs=[
                    run_eval_module.OutputDiff(
                        name="out",
                        passed=False,
                        max_elementwise_abs_diff=0.5,
                        max_elementwise_rel_diff=0.25,
                        relative_l2=0.125,
                    )
                ],
                error="boom",
            )
        ],
    )

    restored = run_eval_module._correctness_from_payload(
        run_eval_module._correctness_to_payload(original)
    )

    assert restored == original


def test_performance_payload_rejects_unknown_keys() -> None:
    """A key with no matching field must raise, not be silently ignored.

    Both call sites catch ``(KeyError, TypeError, ValueError)`` and synthesize a
    failed shape, so TypeError is the contract here.
    """
    from scripts import run_eval as run_eval_module

    payload = run_eval_module._performance_to_payload(
        run_eval_module.PerformanceShapeResult()
    )
    payload["field_from_a_newer_worker"] = 1

    with pytest.raises(TypeError):
        run_eval_module._performance_from_payload(payload)


def test_torch_extension_build_dir_is_isolated_per_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each evaluation gets its own cpp_extension build dir.

    Candidates that JIT-build extensions otherwise share one cache directory and
    one build lock across concurrent evaluations, where a single wedged compiler
    blocks every other build.
    """
    from scripts import run_eval as run_eval_module

    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    artifact_dir = tmp_path / "20260804-000000" / "op"

    run_eval_module._isolate_torch_extension_build_dir(artifact_dir)

    assert os.environ["TORCH_EXTENSIONS_DIR"] == str(artifact_dir / "torch_ext")


def test_torch_extension_build_dir_respects_an_existing_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator who already pointed this somewhere keeps their choice."""
    from scripts import run_eval as run_eval_module

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/somewhere/chosen")

    run_eval_module._isolate_torch_extension_build_dir(tmp_path / "op")

    assert os.environ["TORCH_EXTENSIONS_DIR"] == "/somewhere/chosen"


def test_runner_config_records_cuda_graph_mode() -> None:
    from scripts import run_eval as run_eval_module

    config = run_eval_module._build_runner_config(
        config_version="v1",
        mode="candidate",
        atol=1e-2,
        rtol=0.05,
        num_correctness_cases=1,
        warmup_iters=10,
        bench_iters=20,
        benchmark_mode="cuda_graph_replay",
        cuda_graph_cache_flush_mb=1024,
        graph_min_cosine=0.999,
        graph_max_rel_l2=0.01,
        trust_mode="untrusted",
        require_clock_locked=True,
    )

    assert config["benchmark_mode"] == "cuda_graph_replay"
    assert config["cuda_graph_cache_flush_mb"] == 1024
    assert config["graph_atol"] == 1e-2
    assert config["graph_rtol"] == 0.05
    assert config["graph_min_cosine"] == 0.999
    assert config["graph_max_rel_l2"] == 0.01
    assert config["trust_mode"] == "untrusted"
    assert config["require_clock_locked"] is True
    assert config["validation_mode"] == "full"


def test_runner_config_records_validation_mode() -> None:
    from scripts import run_eval as run_eval_module

    config = run_eval_module._build_runner_config(
        config_version="v1",
        mode="candidate",
        validation_mode="correctness_only",
        atol=1e-2,
        rtol=0.05,
        num_correctness_cases=1,
        warmup_iters=1,
        bench_iters=1,
    )

    assert config["validation_mode"] == "correctness_only"


def test_runner_config_records_correctness_rel_l2_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    monkeypatch.setenv("ATREX_CORRECTNESS_MAX_REL_L2", "0.2")

    config = run_eval_module._build_runner_config(
        config_version="v1",
        mode="candidate",
        atol=1e-2,
        rtol=0.05,
        num_correctness_cases=1,
        warmup_iters=25,
        bench_iters=50,
    )

    assert config["correctness_max_rel_l2"] == 0.2


def test_build_environment_records_clock_lock_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    monkeypatch.setenv("ATREX_BENCH_CLOCKS_LOCKED", "1")

    env = run_eval_module._build_environment(clock_locked=True)

    assert env["clock_locked"] is True
    assert env["clock_lock_detected"] is True
    assert env["clock_lock_source"] == "ATREX_BENCH_CLOCKS_LOCKED"


def test_build_environment_treats_detected_clock_marker_as_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    monkeypatch.setenv("ATREX_BENCH_CLOCKS_LOCKED", "1")

    env = run_eval_module._build_environment(clock_locked=False)

    assert env["clock_locked"] is True
    assert env["clock_lock_detected"] is True
    assert env["clock_lock_source"] == "ATREX_BENCH_CLOCKS_LOCKED"


def test_build_environment_labels_legacy_clock_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    monkeypatch.delenv("ATREX_BENCH_CLOCKS_LOCKED", raising=False)
    monkeypatch.delenv("SOL_EXECBENCH_CLOCKS_LOCKED", raising=False)

    environment = run_eval_module._build_environment(clock_locked=True)

    assert environment["clock_locked"] is True
    assert environment["clock_lock_detected"] is False
    assert environment["clock_lock_source"] == "legacy-assertion"
    assert "clock_lock_verified" not in environment


def test_worker_rejects_missing_required_clock_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    monkeypatch.delenv("ATREX_BENCH_CLOCKS_LOCKED", raising=False)
    monkeypatch.delenv("SOL_EXECBENCH_CLOCKS_LOCKED", raising=False)
    reference_dir = _build_reference_dir(tmp_path, name="clock_required")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    payload = run_eval_module._run_eval_worker(
        input_path=CANDIDATE_PATH,
        reference_dir=reference_dir,
        artifact_dir=artifact_dir,
        atol=1e-2,
        rtol=0.05,
        num_correctness_cases=1,
        warmup_iters=1,
        bench_iters=1,
        checkpoint_dir=None,
        config_version="v1",
        clock_locked=False,
        require_clock_locked=True,
        collect_kernel_events=False,
        candidate_timeout_s=60,
        perf_timeout_s=600,
    )

    assert "GPU clock lock is required" in payload["error"]
    assert payload["passed"]["compile"]["0"]["status"] == "failed"
    assert payload["passed"]["correctness"]["0"]["status"] == "skipped"
    assert payload["passed"]["performance"]["0"]["status"] == "skipped"


def test_save_eval_json_rejects_nonfinite_values(tmp_path: Path) -> None:
    from scripts import run_eval as run_eval_module

    with pytest.raises(ValueError, match="Out of range float"):
        run_eval_module._save_eval_json({"bad": float("inf")}, tmp_path / "bad.json")

    assert not (tmp_path / "bad.json").exists()


def test_save_eval_json_replaces_same_directory_temp_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    output_path = tmp_path / "result.json"
    replacements: list[tuple[Path, Path]] = []
    real_replace = run_eval_module.os.replace

    def inspect_replace(source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == output_path.parent
        assert destination_path == output_path
        assert output_path.exists() is False
        assert json.loads(source_path.read_text(encoding="utf-8")) == {"ok": True}
        replacements.append((source_path, destination_path))
        real_replace(source, destination)

    monkeypatch.setattr(run_eval_module.os, "replace", inspect_replace)

    run_eval_module._save_eval_json({"ok": True}, output_path)

    assert len(replacements) == 1
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}


def test_load_runner_config_rejects_unknown_keys(tmp_path: Path) -> None:
    from scripts import run_eval as run_eval_module

    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "warmup_iters": 3,
                "unknown_key": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported runner config key"):
        run_eval_module._load_runner_config_file(config_path)


def test_resolve_runner_option_prefers_cli_over_config() -> None:
    from scripts import run_eval as run_eval_module

    config = {"warmup_iters": 7}

    assert (
        run_eval_module._resolve_runner_option(
            "warmup_iters",
            cli_value=3,
            config=config,
            default=10,
        )
        == 3
    )
    assert (
        run_eval_module._resolve_runner_option(
            "warmup_iters",
            cli_value=None,
            config=config,
            default=10,
        )
        == 7
    )
    assert (
        run_eval_module._resolve_runner_option(
            "bench_iters",
            cli_value=None,
            config=config,
            default=100,
        )
        == 100
    )


@pytest.mark.parametrize(
    ("cli_mode", "config", "expected"),
    [
        (None, {}, "full"),
        (None, {"validation_mode": "performance_only"}, "performance_only"),
        (
            "correctness_only",
            {"validation_mode": "performance_only"},
            "correctness_only",
        ),
    ],
)
def test_resolve_validation_mode(
    cli_mode: str | None,
    config: dict[str, object],
    expected: str,
) -> None:
    from scripts import run_eval as run_eval_module

    assert run_eval_module._resolve_validation_mode(cli_mode, config) == expected


def test_resolve_validation_mode_rejects_unknown_value() -> None:
    from scripts import run_eval as run_eval_module

    with pytest.raises(ValueError, match="validation_mode must be one of"):
        run_eval_module._resolve_validation_mode(
            None,
            {"validation_mode": "fast"},
        )


def test_validation_only_flags_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--reference-dir",
            "reference",
            "--correctness-only",
            "--performance-only",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert exc_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "expected_mode"),
    [
        (None, "full"),
        ("--correctness-only", "correctness_only"),
        ("--performance-only", "performance_only"),
    ],
)
def test_validation_mode_cli_propagates_and_exits_successfully(
    monkeypatch: pytest.MonkeyPatch,
    flag: str | None,
    expected_mode: str,
) -> None:
    from scripts import run_eval as run_eval_module

    argv = [
        "run_eval.py",
        "--input",
        "candidate.py",
        "--reference-dir",
        "reference",
        "--output",
        "output",
    ]
    if flag is not None:
        argv.append(flag)
    calls: list[dict[str, object]] = []

    def fake_run_eval(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        correctness = "skipped" if expected_mode == "performance_only" else "passed"
        performance = "skipped" if expected_mode == "correctness_only" else "passed"
        return {
            "runner_config": {"validation_mode": expected_mode},
            "passed": {
                "compile": {"0": {"status": "passed"}},
                "correctness": {"0": {"status": correctness}},
                "performance": {"0": {"status": performance}},
            },
            "error": None,
        }

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(run_eval_module, "run_eval", fake_run_eval)

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert exc_info.value.code == 0
    assert calls[0]["validation_mode"] == expected_mode


def test_config_only_candidate_launch_resolves_paths_and_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    candidate_path = tmp_path / "candidate.py"
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "eval_mode": "candidate",
                "validation_mode": "performance_only",
                "input": str(candidate_path),
                "reference_dir": str(reference_dir),
                "output": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_run_eval(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "runner_config": {"validation_mode": "performance_only"},
            "passed": {
                "compile": {"0": {"status": "passed"}},
                "correctness": {"0": {"status": "skipped"}},
                "performance": {"0": {"status": "passed"}},
            },
            "error": None,
        }

    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", str(config_path)])
    monkeypatch.setattr(run_eval_module, "run_eval", fake_run_eval)

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert exc_info.value.code == 0
    assert len(calls) == 1
    assert calls[0]["input_path"] == candidate_path
    assert calls[0]["reference_dir"] == reference_dir
    assert calls[0]["output_root"] == output_dir
    assert calls[0]["checkpoint_dir"] == checkpoint_dir
    assert calls[0]["validation_mode"] == "performance_only"


def test_config_only_torch_compile_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "output"
    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "eval_mode": "torch_compile_reference",
                "validation_mode": "performance_only",
                "reference_dir": str(reference_dir),
                "output": str(output_dir),
                "warmup_iters": 3,
                "bench_iters": 7,
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_run_torch_compile_eval(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "eval_mode": "torch_compile_reference",
            "passed": {"compile": {"0": {"status": "passed"}}},
            "performance": {
                "shapes": {"0": {"samples": [1.0], "error": None}}
            },
            "error": None,
        }

    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", str(config_path)])
    monkeypatch.setattr(
        run_eval_module,
        "run_torch_compile_eval",
        fake_run_torch_compile_eval,
    )
    monkeypatch.setattr(
        run_eval_module,
        "run_eval",
        lambda **kwargs: pytest.fail("candidate evaluator must not run"),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert exc_info.value.code == 0
    assert len(calls) == 1
    assert calls[0]["reference_dir"] == reference_dir
    assert calls[0]["output_root"] == output_dir
    assert calls[0]["warmup_iters"] == 3
    assert calls[0]["bench_iters"] == 7


def test_cli_launch_paths_and_mode_override_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    cli_reference = tmp_path / "cli-reference"
    cli_output = tmp_path / "cli-output"
    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "eval_mode": "candidate",
                "reference_dir": str(tmp_path / "config-reference"),
                "output": str(tmp_path / "config-output"),
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_run_torch_compile_eval(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "eval_mode": "torch_compile_reference",
            "passed": {"compile": {"0": {"status": "passed"}}},
            "performance": {
                "shapes": {"0": {"samples": [1.0], "error": None}}
            },
            "error": None,
        }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--config",
            str(config_path),
            "--torch-compile",
            "--reference-dir",
            str(cli_reference),
            "--output",
            str(cli_output),
        ],
    )
    monkeypatch.setattr(
        run_eval_module,
        "run_torch_compile_eval",
        fake_run_torch_compile_eval,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert exc_info.value.code == 0
    assert calls[0]["reference_dir"] == cli_reference
    assert calls[0]["output_root"] == cli_output
    assert "input_path" not in calls[0]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"schema_version": "v2"}, "schema_version must be 'v1'"),
        ({"eval_mode": "fast"}, "eval_mode must be one of"),
        ({"reference_dir": 1}, "reference_dir must be a non-empty path string"),
    ],
)
def test_config_launch_rejects_invalid_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
    message: str,
) -> None:
    from scripts import run_eval as run_eval_module

    config_path = tmp_path / "runner.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", str(config_path)])

    with pytest.raises(SystemExit, match=message):
        run_eval_module.main()


def test_config_only_launch_requires_reference_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "eval_mode": "candidate",
                "input": str(tmp_path / "candidate.py"),
                "output": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", str(config_path)])

    with pytest.raises(SystemExit, match="reference_dir is required"):
        run_eval_module.main()


def test_config_torch_compile_rejects_candidate_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "eval_mode": "torch_compile_reference",
                "input": str(tmp_path / "candidate.py"),
                "reference_dir": str(tmp_path / "reference"),
                "output": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", str(config_path)])

    with pytest.raises(SystemExit, match="input cannot be combined"):
        run_eval_module.main()


def test_config_torch_compile_rejects_non_performance_validation_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "eval_mode": "torch_compile_reference",
                "validation_mode": "full",
                "reference_dir": str(tmp_path / "reference"),
                "output": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", str(config_path)])

    with pytest.raises(SystemExit, match="requires validation_mode=performance_only"):
        run_eval_module.main()


def test_torch_compile_rejects_candidate_validation_only_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--torch-compile",
            "--correctness-only",
            "--reference-dir",
            "reference",
            "--output",
            "output",
        ],
    )

    with pytest.raises(SystemExit, match="cannot be combined with --torch-compile"):
        run_eval_module.main()


def _clock_lock_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "clock_lock_mode": None,
        "lock_clocks": False,
        "clock_lock_device": None,
        "gpu_clock_mhz": None,
        "memory_clock_mhz": None,
        "clock_lock_tolerance_mhz": None,
        "clock_lock_settle_seconds": None,
        "clock_lock_command_timeout_s": None,
        "clock_lock_require_idle": None,
        "clock_lock_monitor": None,
        "clock_lock_sample_interval_ms": None,
        "clock_lock_runtime_tolerance_mhz": None,
        "clock_locked": False,
        "require_clock_locked": False,
        "worker": False,
        "single_shape_worker": False,
        "torch_compile_worker": False,
        "torch_compile_shape_worker": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_clock_lock_config_defaults_to_off() -> None:
    from scripts import run_eval as run_eval_module

    config = run_eval_module._resolve_clock_lock_config(_clock_lock_args(), {})

    assert config.mode == "off"


def test_clock_lock_config_loads_managed_json_values() -> None:
    from scripts import run_eval as run_eval_module

    config = run_eval_module._resolve_clock_lock_config(
        _clock_lock_args(),
        {
            "clock_lock_mode": "manage",
            "clock_lock_device": "GPU-aabb",
            "gpu_clock_mhz": 1500,
            "memory_clock_mhz": 3996,
            "clock_lock_tolerance_mhz": 25,
            "clock_lock_settle_seconds": 1.5,
            "clock_lock_command_timeout_s": 8.0,
            "clock_lock_require_idle": False,
            "clock_lock_monitor": True,
            "clock_lock_sample_interval_ms": 20,
            "clock_lock_runtime_tolerance_mhz": 7,
        },
    )

    assert config.mode == "manage"
    assert config.device_selector == "GPU-aabb"
    assert config.graphics_mhz == 1500
    assert config.memory_mhz == 3996
    assert config.tolerance_mhz == 25
    assert config.settle_seconds == 1.5
    assert config.command_timeout_seconds == 8.0
    assert config.require_idle is False
    assert config.monitor_enabled is True
    assert config.sample_interval_ms == 20
    assert config.runtime_tolerance_mhz == 7


def test_clock_lock_config_allows_graphics_only_managed_mode() -> None:
    from scripts import run_eval as run_eval_module

    config = run_eval_module._resolve_clock_lock_config(
        _clock_lock_args(lock_clocks=True, gpu_clock_mhz=1500),
        {},
    )

    assert config.mode == "manage"
    assert config.graphics_mhz == 1500
    assert config.memory_mhz is None
    assert config.monitor_enabled is True
    assert config.sample_interval_ms == 10
    assert config.runtime_tolerance_mhz == 0


def test_clock_lock_cli_values_override_json() -> None:
    from scripts import run_eval as run_eval_module

    config = run_eval_module._resolve_clock_lock_config(
        _clock_lock_args(
            clock_lock_mode="manage",
            clock_lock_device="2",
            gpu_clock_mhz=1600,
            memory_clock_mhz=4200,
            clock_lock_require_idle=True,
        ),
        {
            "clock_lock_mode": "off",
            "clock_lock_device": "GPU-old",
            "gpu_clock_mhz": 1200,
            "memory_clock_mhz": 3000,
            "clock_lock_require_idle": False,
        },
    )

    assert config.mode == "manage"
    assert config.device_selector == "2"
    assert config.graphics_mhz == 1600
    assert config.memory_mhz == 4200
    assert config.require_idle is True


@pytest.mark.parametrize(
    ("args", "expected_mode"),
    [
        (_clock_lock_args(lock_clocks=True, gpu_clock_mhz=1500, memory_clock_mhz=3996), "manage"),
        (_clock_lock_args(require_clock_locked=True), "external"),
    ],
)
def test_clock_lock_legacy_aliases_resolve_mode(
    args: SimpleNamespace,
    expected_mode: str,
) -> None:
    from scripts import run_eval as run_eval_module

    config = run_eval_module._resolve_clock_lock_config(args, {})

    assert config.mode == expected_mode


@pytest.mark.parametrize("legacy_flag", ["clock_locked", "require_clock_locked"])
def test_managed_clock_lock_rejects_legacy_assertion_flags(
    legacy_flag: str,
) -> None:
    from scripts import run_eval as run_eval_module

    args = _clock_lock_args(
        clock_lock_mode="manage",
        gpu_clock_mhz=1500,
        memory_clock_mhz=3996,
        **{legacy_flag: True},
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        run_eval_module._resolve_clock_lock_config(args, {})


@pytest.mark.parametrize("mode", ["off", "external"])
def test_management_settings_require_manage_mode(mode: str) -> None:
    from scripts import run_eval as run_eval_module

    with pytest.raises(ValueError, match="only valid in manage mode"):
        run_eval_module._resolve_clock_lock_config(
            _clock_lock_args(clock_lock_mode=mode, gpu_clock_mhz=1500),
            {},
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("clock_lock_monitor", 1, "must be a boolean"),
        ("clock_lock_sample_interval_ms", 0, "positive integer"),
        ("clock_lock_runtime_tolerance_mhz", -1, "non-negative integer"),
    ],
)
def test_managed_monitor_config_rejects_invalid_values(
    field: str,
    value: object,
    expected_error: str,
) -> None:
    from scripts import run_eval as run_eval_module

    with pytest.raises(ValueError, match=expected_error):
        run_eval_module._resolve_clock_lock_config(
            _clock_lock_args(
                clock_lock_mode="manage",
                gpu_clock_mhz=1132,
                **{field: value},
            ),
            {},
        )


def test_worker_rejects_explicit_managed_clock_mode() -> None:
    from scripts import run_eval as run_eval_module

    with pytest.raises(ValueError, match="top-level parent"):
        run_eval_module._resolve_clock_lock_config(
            _clock_lock_args(
                worker=True,
                clock_lock_mode="manage",
                gpu_clock_mhz=1500,
                memory_clock_mhz=3996,
            ),
            {},
        )


@pytest.mark.parametrize(
    ("mode", "expected_clock_locked", "expected_required"),
    [("off", False, False), ("external", False, True)],
)
def test_run_eval_off_and_external_modes_do_not_build_manager(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_clock_locked: bool,
    expected_required: bool,
) -> None:
    from atrex_bench.eval.clock_lock import ClockLockConfig
    from scripts import run_eval as run_eval_module

    process_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        run_eval_module,
        "_run_eval_process",
        lambda *args, **kwargs: process_calls.append(kwargs) or {"ok": True},
        raising=False,
    )
    monkeypatch.setattr(
        run_eval_module,
        "_build_managed_clock_session",
        lambda **kwargs: pytest.fail("manager must not be built"),
        raising=False,
    )

    payload = run_eval_module.run_eval(
        Path("candidate.py"),
        Path("reference"),
        Path("output"),
        timestamp="20260801-000000",
        clock_lock_config=ClockLockConfig(mode=mode),
    )

    assert payload == {"ok": True}
    assert process_calls[0]["clock_locked"] is expected_clock_locked
    assert process_calls[0]["require_clock_locked"] is expected_required


def test_run_eval_propagates_validation_mode_to_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    process_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        run_eval_module,
        "_run_eval_process",
        lambda *args, **kwargs: process_calls.append(kwargs) or {"ok": True},
    )

    payload = run_eval_module.run_eval(
        Path("candidate.py"),
        Path("reference"),
        Path("output"),
        timestamp="20260804-000000",
        validation_mode="performance_only",
    )

    assert payload == {"ok": True}
    assert process_calls[0]["validation_mode"] == "performance_only"


def test_managed_mode_runs_only_in_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench.eval.clock_lock import ClockLockConfig
    from scripts import run_eval as run_eval_module

    lifecycle_events: list[str] = []
    process_calls: list[dict[str, object]] = []

    class CountingSession:
        def __init__(self) -> None:
            self.report = _managed_clock_report(verified=True, restored=True)

        def __enter__(self):
            lifecycle_events.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            lifecycle_events.append("exit")
            return False

    monkeypatch.setattr(
        run_eval_module,
        "_build_managed_clock_session",
        lambda **kwargs: CountingSession(),
        raising=False,
    )
    monkeypatch.setattr(
        run_eval_module,
        "_run_eval_process",
        lambda *args, **kwargs: (
            lifecycle_events.append("evaluate")
            or process_calls.append(kwargs)
            or {"ok": True}
        ),
        raising=False,
    )
    config = ClockLockConfig(
        mode="manage",
        device_selector="GPU-aabb",
        graphics_mhz=1500,
        memory_mhz=3996,
        settle_seconds=0,
    )

    payload = run_eval_module.run_eval(
        Path("candidate.py"),
        Path("reference"),
        Path("output"),
        timestamp="20260801-000000",
        clock_lock_config=config,
    )

    assert payload["ok"] is True
    assert payload["environment"]["clock_lock_verified"] is True
    assert lifecycle_events == ["enter", "evaluate", "exit"]
    assert len(process_calls) == 1
    assert process_calls[0]["clock_locked"] is True
    assert process_calls[0]["require_clock_locked"] is True
    assert "clock_lock_config" not in process_calls[0]


def test_torch_compile_managed_mode_owns_one_parent_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench.eval.clock_lock import ClockLockConfig
    from scripts import run_eval as run_eval_module

    lifecycle_events: list[str] = []
    process_calls: list[dict[str, object]] = []

    class CountingSession:
        def __init__(self) -> None:
            self.report = _managed_clock_report(verified=True, restored=True)

        def __enter__(self):
            lifecycle_events.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            lifecycle_events.append("exit")
            return False

    monkeypatch.setattr(
        run_eval_module,
        "_build_managed_clock_session",
        lambda **kwargs: CountingSession(),
        raising=False,
    )
    monkeypatch.setattr(
        run_eval_module,
        "_run_torch_compile_eval_process",
        lambda *args, **kwargs: (
            lifecycle_events.append("evaluate")
            or process_calls.append(kwargs)
            or {"ok": True}
        ),
        raising=False,
    )
    config = ClockLockConfig(
        mode="manage",
        graphics_mhz=1500,
        memory_mhz=3996,
        settle_seconds=0,
    )

    payload = run_eval_module.run_torch_compile_eval(
        Path("reference"),
        Path("output"),
        timestamp="20260801-000000",
        clock_lock_config=config,
    )

    assert payload["ok"] is True
    assert payload["environment"]["clock_lock_verified"] is True
    assert lifecycle_events == ["enter", "evaluate", "exit"]
    assert process_calls[0]["clock_locked"] is True
    assert process_calls[0]["require_clock_locked"] is True


@pytest.mark.parametrize(
    "worker_error",
    [KeyboardInterrupt("cancelled"), subprocess.TimeoutExpired("worker", 5)],
    ids=["sigint", "worker-timeout"],
)
def test_managed_parent_lifecycle_exits_when_worker_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    worker_error: BaseException,
) -> None:
    from atrex_bench.eval.clock_lock import ClockLockConfig
    from scripts import run_eval as run_eval_module

    lifecycle_events: list[str] = []

    class CountingSession:
        def __enter__(self):
            lifecycle_events.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            lifecycle_events.append(f"exit:{exc_type.__name__}")
            return False

    def interrupted_process(*args, **kwargs):
        lifecycle_events.append("evaluate")
        raise worker_error

    monkeypatch.setattr(
        run_eval_module,
        "_build_managed_clock_session",
        lambda **kwargs: CountingSession(),
    )
    monkeypatch.setattr(run_eval_module, "_run_eval_process", interrupted_process)
    config = ClockLockConfig(
        mode="manage",
        graphics_mhz=1500,
        memory_mhz=3996,
        settle_seconds=0,
    )

    with pytest.raises(type(worker_error)):
        run_eval_module.run_eval(
            Path("candidate.py"),
            Path("reference"),
            Path("output"),
            timestamp="20260801-000000",
            clock_lock_config=config,
        )

    assert lifecycle_events == [
        "enter",
        "evaluate",
        f"exit:{type(worker_error).__name__}",
    ]


@pytest.mark.parametrize(
    "interrupt",
    [KeyboardInterrupt("cancelled"), SystemExit(143)],
    ids=["sigint", "sigterm"],
)
def test_managed_interrupt_persists_final_clock_report_before_reraise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    from atrex_bench.eval.clock_lock import ClockLockConfig
    from scripts import run_eval as run_eval_module

    reference_dir = _build_reference_dir(
        tmp_path / "reference", name="interrupted_eval"
    )
    timestamp = "20260801-000050"
    artifact_dir = tmp_path / timestamp / "interrupted_eval"

    class InterruptedSession:
        def __init__(self) -> None:
            self.report = _managed_clock_report(
                verified=True,
                restored=False,
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.report = _managed_clock_report(
                verified=True,
                restored=True,
            )
            return False

    def interrupted_process(*args, **kwargs):
        run_eval_module._save_eval_json(
            _passing_worker_payload(), artifact_dir / "eval_result.json"
        )
        raise interrupt

    monkeypatch.setattr(
        run_eval_module,
        "_build_managed_clock_session",
        lambda **kwargs: InterruptedSession(),
    )
    monkeypatch.setattr(
        run_eval_module,
        "_run_eval_process",
        interrupted_process,
    )
    config = ClockLockConfig(
        mode="manage",
        graphics_mhz=1500,
        memory_mhz=3996,
        settle_seconds=0,
    )

    with pytest.raises(type(interrupt)) as exc_info:
        run_eval_module.run_eval(
            CANDIDATE_PATH,
            reference_dir,
            tmp_path,
            timestamp=timestamp,
            clock_lock_config=config,
        )

    assert exc_info.value is interrupt
    payload = json.loads(
        (artifact_dir / "eval_result.json").read_text(encoding="utf-8")
    )
    assert payload["correctness"] == {"sentinel": "preserve-correctness"}
    assert payload["performance"] == {"sentinel": "preserve-performance"}
    assert payload["environment"]["clock_lock_verified"] is True
    assert payload["environment"]["clock_lock"]["restored"] is True
    assert type(interrupt).__name__ in payload["error"]
    clock_report = json.loads(
        (artifact_dir / "clock_lock.json").read_text(encoding="utf-8")
    )
    assert clock_report["restored"] is True


def _managed_clock_report(
    *,
    verified: bool,
    restored: bool,
    error: str | None = None,
):
    from atrex_bench.eval.clock_lock import ClockLockReport
    from atrex_bench.eval.nvidia_clock import ClockSnapshot, NvidiaDevice

    device = NvidiaDevice("0", "0", "GPU-aabb", "NVIDIA Test GPU")
    verification = (
        ClockSnapshot(1500, 3996, "2026-08-01T00:00:00Z")
        if verified
        else None
    )
    post_restore = (
        ClockSnapshot(900, 3996, "2026-08-01T00:01:00Z")
        if restored
        else None
    )
    return ClockLockReport(
        mode="manage",
        source="atrex-managed",
        device=device,
        requested_graphics_mhz=1500,
        requested_memory_mhz=3996,
        tolerance_mhz=50,
        applied=verified,
        verified=verified,
        verification_snapshot=verification,
        restored=restored,
        post_restore_snapshot=post_restore,
        error=error,
    )


def _passing_worker_payload() -> dict[str, object]:
    return {
        "environment": {"clock_locked": True},
        "passed": {
            "compile": {"0": {"status": "passed"}},
            "correctness": {"0": {"status": "passed"}},
        },
        "correctness": {"sentinel": "preserve-correctness"},
        "performance": {"sentinel": "preserve-performance"},
        "error": None,
    }


def test_managed_session_callback_persists_clock_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench.eval.clock_lock import ClockLockConfig
    from atrex_bench.eval.nvidia_clock import NvidiaDevice
    from scripts import run_eval as run_eval_module

    captured: dict[str, object] = {}
    monitor_kwargs: dict[str, object] = {}

    class StubManagedClockLock:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class StubClockMonitor:
        def __init__(self, **kwargs) -> None:
            monitor_kwargs.update(kwargs)

    monkeypatch.setattr(run_eval_module, "ManagedClockLock", StubManagedClockLock)
    monkeypatch.setattr(run_eval_module, "NvidiaClockMonitor", StubClockMonitor)
    config = ClockLockConfig(
        mode="manage",
        graphics_mhz=1132,
        monitor_enabled=True,
        sample_interval_ms=10,
        runtime_tolerance_mhz=0,
    )

    run_eval_module._build_managed_clock_session(
        config=config,
        artifact_dir=tmp_path,
    )
    captured["report_callback"](_managed_clock_report(verified=True, restored=False))
    device = NvidiaDevice("GPU-aabb", "7", "GPU-aabb", "NVIDIA Test GPU")
    monitor = captured["monitor_factory"](device, config)

    report_path = tmp_path / "clock_lock.json"
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["verified"] is True
    assert persisted["restored"] is False
    assert isinstance(monitor, StubClockMonitor)
    assert monitor_kwargs == {
        "device_uuid": "GPU-aabb",
        "target_mhz": 1132,
        "tolerance_mhz": 0,
        "sample_interval_ms": 10,
        "trace_path": tmp_path / "clock_lock_trace.csv",
    }


def test_attach_clock_report_records_managed_provenance() -> None:
    from scripts import run_eval as run_eval_module

    original = _passing_worker_payload()
    report = _managed_clock_report(verified=True, restored=True)

    attached = run_eval_module._attach_clock_lock_report(original, report)

    assert "clock_lock" not in original["environment"]
    environment = attached["environment"]
    assert environment["clock_locked"] is True
    assert environment["clock_lock_verified"] is True
    assert environment["clock_lock_source"] == "atrex-managed"
    assert environment["clock_lock"]["verified"] is True
    assert environment["clock_lock"]["restored"] is True
    assert attached["error"] is None


def test_acquisition_failure_does_not_launch_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench.eval.clock_lock import ClockLockConfig, ClockLockError
    from scripts import run_eval as run_eval_module

    report = _managed_clock_report(
        verified=False,
        restored=False,
        error="GPU GPU-aabb is not idle",
    )

    class AcquisitionFailureSession:
        def __init__(self) -> None:
            self.report = report

        def __enter__(self):
            raise ClockLockError("GPU GPU-aabb is not idle")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        run_eval_module,
        "_build_managed_clock_session",
        lambda **kwargs: AcquisitionFailureSession(),
    )
    worker_calls: list[object] = []
    monkeypatch.setattr(
        run_eval_module,
        "_run_eval_process",
        lambda *args, **kwargs: worker_calls.append(kwargs),
    )
    monkeypatch.setattr(
        run_eval_module,
        "_build_environment",
        lambda *, clock_locked: {"clock_locked": clock_locked},
    )
    reference_dir = _build_reference_dir(tmp_path / "reference", name="acquire_fail")
    timestamp = "20260801-000100"
    config = ClockLockConfig(
        mode="manage",
        graphics_mhz=1500,
        memory_mhz=3996,
    )

    payload = run_eval_module.run_eval(
        CANDIDATE_PATH,
        reference_dir,
        tmp_path,
        timestamp=timestamp,
        clock_lock_config=config,
    )

    assert worker_calls == []
    assert payload["environment"]["clock_locked"] is False
    assert payload["environment"]["clock_lock"]["verified"] is False
    assert "not idle" in payload["error"]
    artifact_dir = tmp_path / timestamp / "acquire_fail"
    assert (artifact_dir / "clock_lock.json").is_file()
    assert (artifact_dir / "eval_result.json").is_file()


def test_reset_failure_makes_payload_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench.eval.clock_lock import ClockLockConfig, ClockLockError
    from scripts import run_eval as run_eval_module

    class ResetFailureSession:
        def __init__(self) -> None:
            self.report = _managed_clock_report(verified=True, restored=False)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.report = _managed_clock_report(
                verified=True,
                restored=False,
                error="reset_memory failed",
            )
            raise ClockLockError("Failed to restore GPU clocks: reset_memory")

    monkeypatch.setattr(
        run_eval_module,
        "_build_managed_clock_session",
        lambda **kwargs: ResetFailureSession(),
    )
    monkeypatch.setattr(
        run_eval_module,
        "_run_eval_process",
        lambda *args, **kwargs: _passing_worker_payload(),
    )
    reference_dir = _build_reference_dir(tmp_path / "reference", name="reset_fail")
    timestamp = "20260801-000200"
    config = ClockLockConfig(
        mode="manage",
        graphics_mhz=1500,
        memory_mhz=3996,
    )

    payload = run_eval_module.run_eval(
        CANDIDATE_PATH,
        reference_dir,
        tmp_path,
        timestamp=timestamp,
        clock_lock_config=config,
    )

    assert payload["correctness"] == {"sentinel": "preserve-correctness"}
    assert payload["performance"] == {"sentinel": "preserve-performance"}
    assert payload["environment"]["clock_lock"]["restored"] is False
    assert "reset_memory" in payload["error"]
    assert run_eval_module._payload_overall_passed(payload) is False


def test_main_resolves_managed_clock_cli_and_json_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "clock_lock_device": "GPU-aabb",
                "gpu_clock_mhz": 1500,
                "memory_clock_mhz": 3996,
                "clock_lock_tolerance_mhz": 25,
                "clock_lock_settle_seconds": 1.5,
                "clock_lock_command_timeout_s": 8.0,
                "clock_lock_require_idle": True,
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_run_eval(*args, **kwargs):
        calls.append(kwargs)
        return {
            "passed": {
                "compile": {"0": {"status": "passed"}},
                "correctness": {"0": {"status": "passed"}},
            }
        }

    monkeypatch.setattr(run_eval_module, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        run_eval_module,
        "get_timestamp",
        lambda timestamp=None: timestamp or "20260801-000000",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--input",
            str(CANDIDATE_PATH),
            "--reference-dir",
            str(REFERENCE_PATH.parent),
            "--output",
            str(tmp_path),
            "--config",
            str(config_path),
            "--lock-clocks",
            "--gpu-clock-mhz",
            "1600",
            "--allow-busy-gpu",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert exc_info.value.code == 0
    clock_config = calls[0]["clock_lock_config"]
    assert clock_config.mode == "manage"
    assert clock_config.device_selector == "GPU-aabb"
    assert clock_config.graphics_mhz == 1600
    assert clock_config.memory_mhz == 3996
    assert clock_config.tolerance_mhz == 25
    assert clock_config.require_idle is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clock_locked", "false"),
        ("require_clock_locked", 1),
        ("skip_kernel_attribution", "false"),
    ],
)
def test_main_rejects_non_boolean_legacy_clock_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    from scripts import run_eval as run_eval_module

    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps({field: value}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_eval_module,
        "run_eval",
        lambda *args, **kwargs: pytest.fail("invalid config reached run_eval"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--input",
            str(CANDIDATE_PATH),
            "--reference-dir",
            str(REFERENCE_PATH.parent),
            "--output",
            str(tmp_path),
            "--config",
            str(config_path),
        ],
    )

    with pytest.raises(SystemExit, match=rf"{field} must be a boolean"):
        run_eval_module.main()


def test_single_shape_worker_command_propagates_cuda_graph_mode(
    tmp_path: Path,
) -> None:
    from scripts import run_eval as run_eval_module

    command = run_eval_module._build_single_shape_worker_command(
        candidate_path=tmp_path / "candidate.py",
        reference_dir=tmp_path / "reference",
        artifact_dir=tmp_path / "artifact",
        checkpoint_root=tmp_path / "checkpoint",
        shape_id="0",
        atol=1e-2,
        rtol=0.05,
        num_correctness_cases=1,
        warmup_iters=10,
        bench_iters=20,
        shape_result_path=tmp_path / "shape.json",
        collect_kernel_events=False,
        candidate_timeout_s=60,
        perf_timeout_s=600,
        benchmark_mode="cuda_graph_replay",
        cuda_graph_cache_flush_mb=1024,
        graph_atol=1e-2,
        graph_rtol=0.05,
        graph_min_cosine=0.999,
        graph_max_rel_l2=0.01,
        trust_mode="untrusted",
        validation_mode="performance_only",
    )

    mode_index = command.index("--benchmark-mode")
    flush_index = command.index("--cuda-graph-cache-flush-mb")
    graph_atol_index = command.index("--graph-atol")
    graph_rtol_index = command.index("--graph-rtol")
    graph_min_cosine_index = command.index("--graph-min-cosine")
    graph_max_rel_l2_index = command.index("--graph-max-rel-l2")
    trust_mode_index = command.index("--trust-mode")
    assert command[mode_index + 1] == "cuda_graph_replay"
    assert command[flush_index + 1] == "1024"
    assert command[graph_atol_index + 1] == "0.01"
    assert command[graph_rtol_index + 1] == "0.05"
    assert command[graph_min_cosine_index + 1] == "0.999"
    assert command[graph_max_rel_l2_index + 1] == "0.01"
    assert command[trust_mode_index + 1] == "untrusted"
    assert "--performance-only" in command


@pytest.mark.parametrize(
    ("validation_mode", "expected_calls", "expected_correctness_status"),
    [
        ("correctness_only", {"correctness": 1, "performance": 0}, "passed"),
        ("performance_only", {"correctness": 0, "performance": 1}, "skipped"),
    ],
)
def test_single_shape_only_mode_runs_one_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_mode: str,
    expected_calls: dict[str, int],
    expected_correctness_status: str,
) -> None:
    from scripts import run_eval as run_eval_module

    calls = {"correctness": 0, "performance": 0}

    def fake_correctness(*_args, **_kwargs):
        calls["correctness"] += 1
        return run_eval_module.CorrectnessShapeResult(
            status="passed",
            reason=None,
        )

    def fake_performance(*_args, **_kwargs):
        calls["performance"] += 1
        return run_eval_module.PerformanceShapeResult(
            samples=[run_eval_module.PerformanceSample(0.1)]
        )

    monkeypatch.setattr(run_eval_module, "check_correctness", fake_correctness)
    monkeypatch.setattr(run_eval_module, "benchmark_performance", fake_performance)
    reference_dir = _build_reference_dir(tmp_path, name="validation_mode")
    candidate_path = _write_candidate_file(
        tmp_path,
        "candidate.py",
        "class Model:\n    def __call__(self, x):\n        return x\n",
    )
    artifact_dir = tmp_path / "artifact"
    checkpoint_root = tmp_path / "checkpoint"
    shape_result_path = tmp_path / "shape.json"
    artifact_dir.mkdir()
    checkpoint_root.mkdir()

    run_eval_module._run_single_shape_main(
        candidate_path=candidate_path,
        reference_dir=reference_dir,
        artifact_dir=artifact_dir,
        checkpoint_root=checkpoint_root,
        shape_id="0",
        atol=1e-2,
        rtol=0.05,
        num_correctness_cases=1,
        warmup_iters=1,
        bench_iters=1,
        shape_result_path=shape_result_path,
        collect_kernel_events=False,
        candidate_timeout_s=60,
        perf_timeout_s=600,
        trust_mode="trusted",
        validation_mode=validation_mode,
    )

    payload = json.loads(shape_result_path.read_text(encoding="utf-8"))
    assert calls == expected_calls
    assert payload["correctness"]["status"] == expected_correctness_status


@pytest.mark.parametrize(
    ("validation_mode", "correctness_status"),
    [
        ("correctness_only", "passed"),
        ("performance_only", "skipped"),
        ("full", "passed"),
    ],
)
def test_worker_records_mode_specific_performance_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_mode: str,
    correctness_status: str,
) -> None:
    from scripts import run_eval as run_eval_module

    reference_dir = _build_reference_dir(tmp_path, name="stage_verdict")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    correctness = run_eval_module.CorrectnessShapeResult(
        status=correctness_status,
        reason=(
            run_eval_module._CORRECTNESS_ONLY_SKIP_REASON
            if correctness_status == "skipped"
            else None
        ),
    )
    performance = (
        run_eval_module.PerformanceShapeResult()
        if validation_mode == "correctness_only"
        else run_eval_module.PerformanceShapeResult(
            samples=[run_eval_module.PerformanceSample(0.1)]
        )
    )
    monkeypatch.setattr(
        run_eval_module,
        "check_compilation",
        lambda _path: run_eval_module.CompileResult(status="passed"),
    )
    monkeypatch.setattr(
        run_eval_module,
        "_run_single_shape_subprocess",
        lambda **_kwargs: (correctness, performance, None),
    )

    payload = run_eval_module._run_eval_worker(
        input_path=CANDIDATE_PATH,
        reference_dir=reference_dir,
        artifact_dir=artifact_dir,
        atol=1e-2,
        rtol=0.05,
        num_correctness_cases=1,
        warmup_iters=1,
        bench_iters=1,
        checkpoint_dir=None,
        config_version="v1",
        clock_locked=False,
        require_clock_locked=False,
        collect_kernel_events=False,
        candidate_timeout_s=60,
        perf_timeout_s=600,
        validation_mode=validation_mode,
    )

    expected_performance = (
        "skipped" if validation_mode == "correctness_only" else "passed"
    )
    assert payload["runner_config"]["validation_mode"] == validation_mode
    assert payload["passed"]["correctness"]["0"]["status"] == correctness_status
    assert payload["passed"]["performance"]["0"]["status"] == expected_performance


def test_untrusted_runtime_guard_blocks_cpp_extension_load() -> None:
    import torch.utils.cpp_extension as cpp_extension

    from scripts import run_eval as run_eval_module

    original_load = cpp_extension.load
    original_load_inline = cpp_extension.load_inline
    try:
        run_eval_module._install_untrusted_runtime_guards()
        with pytest.raises(RuntimeError, match="disabled in untrusted mode"):
            cpp_extension.load(name="blocked", sources=[])
    finally:
        cpp_extension.load = original_load
        cpp_extension.load_inline = original_load_inline


def test_untrusted_worker_blocks_cpp_extension_load_during_compile_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    from scripts import run_eval as run_eval_module

    candidate_path = _write_candidate_file(
        tmp_path,
        "import_time_jit.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "import torch.utils.cpp_extension as cpp_extension",
                "",
                "cpp_extension.load(name='blocked_at_import', sources=[])",
                "",
                "class Model(nn.Module):",
                "    def forward(self, x):",
                "        return torch.relu(x)",
            ]
        ),
    )
    reference_dir = _build_reference_dir(tmp_path, name="import_time_jit")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    def _unguarded_load(*_args, **_kwargs):
        raise AssertionError("unguarded import-time load reached")

    monkeypatch.setattr(cpp_extension, "load", _unguarded_load)
    monkeypatch.setattr(cpp_extension, "load_inline", _unguarded_load)

    payload = run_eval_module._run_eval_worker(
        input_path=candidate_path,
        reference_dir=reference_dir,
        artifact_dir=artifact_dir,
        atol=1e-2,
        rtol=0.05,
        num_correctness_cases=1,
        warmup_iters=1,
        bench_iters=1,
        checkpoint_dir=None,
        config_version="v1",
        clock_locked=False,
        require_clock_locked=False,
        collect_kernel_events=False,
        candidate_timeout_s=60,
        perf_timeout_s=600,
        trust_mode="untrusted",
    )

    reason = payload["passed"]["compile"]["0"]["reason"]
    assert reason is not None
    assert "Runtime C++/CUDA extension loading is disabled in untrusted mode" in reason
    assert "unguarded import-time load reached" not in reason


def test_untrusted_integrity_detects_critical_function_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atrex_bench.eval.reward_hack import RewardHackDetected
    from scripts import run_eval as run_eval_module

    snapshot = run_eval_module._snapshot_untrusted_critical_functions()
    monkeypatch.setattr(run_eval_module, "check_correctness", lambda *args, **kwargs: None)

    with pytest.raises(RewardHackDetected, match="check_correctness"):
        run_eval_module._check_untrusted_integrity(snapshot)


def test_untrusted_integrity_detects_cuda_event_monkey_patch() -> None:
    import torch

    from atrex_bench.eval.reward_hack import RewardHackDetected
    from scripts import run_eval as run_eval_module

    original_elapsed_time = torch.cuda.Event.elapsed_time
    snapshot = run_eval_module._snapshot_untrusted_critical_functions()
    try:
        torch.cuda.Event.elapsed_time = lambda self, end_event: 0.0
        with pytest.raises(RewardHackDetected, match="elapsed_time"):
            run_eval_module._check_untrusted_integrity(snapshot)
    finally:
        torch.cuda.Event.elapsed_time = original_elapsed_time


def test_untrusted_output_guard_rejects_tensor_subclass() -> None:
    import torch

    from atrex_bench.eval.reward_hack import (
        RewardHackDetected,
        check_plain_tensor_outputs,
    )

    class LazyTensor(torch.Tensor):
        pass

    tensor = torch.ones(2)
    lazy_tensor = torch.Tensor._make_subclass(LazyTensor, tensor, False)

    with pytest.raises(RewardHackDetected, match="Lazy/proxy tensor"):
        check_plain_tensor_outputs({"out": lazy_tensor})


def test_torch_compile_overall_pass_requires_perf_samples() -> None:
    from scripts.run_eval import _payload_overall_passed

    payload = {
        "eval_mode": "torch_compile_reference",
        "passed": {
            "compile": {"0": {"status": "passed", "reason": None}},
            "correctness": {"0": {"status": "skipped", "reason": "skip"}},
        },
        "performance": {
            "shapes": {
                "0": {
                    "input_artifact": None,
                    "samples": [],
                    "error": "torch.compile failed",
                }
            }
        },
    }

    assert _payload_overall_passed(payload) is False


@pytest.mark.parametrize(
    ("validation_mode", "correctness", "performance", "expected"),
    [
        ("correctness_only", "passed", "skipped", True),
        ("performance_only", "skipped", "passed", True),
        ("full", "passed", "passed", True),
        ("full", "passed", "failed", False),
    ],
)
def test_payload_overall_passed_uses_validation_mode(
    validation_mode: str,
    correctness: str,
    performance: str,
    expected: bool,
) -> None:
    from scripts.run_eval import _payload_overall_passed

    payload = {
        "eval_mode": "candidate",
        "runner_config": {"validation_mode": validation_mode},
        "passed": {
            "compile": {"0": {"status": "passed", "reason": None}},
            "correctness": {
                "0": {"status": correctness, "reason": None},
            },
            "performance": {
                "0": {"status": performance, "reason": None},
            },
        },
        "error": None,
    }

    assert _payload_overall_passed(payload) is expected


def test_payload_overall_passed_preserves_legacy_candidate_policy() -> None:
    from scripts.run_eval import _payload_overall_passed

    payload = {
        "eval_mode": "candidate",
        "runner_config": {},
        "passed": {
            "compile": {"0": {"status": "passed", "reason": None}},
            "correctness": {"0": {"status": "passed", "reason": None}},
        },
        "error": None,
    }

    assert _payload_overall_passed(payload) is True


def test_torch_compile_mode_rejects_input_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_eval as run_eval_module

    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("class Model:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--torch-compile",
            "--input",
            str(candidate_path),
            "--reference-dir",
            str(tmp_path / "reference"),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_eval_module.main()

    assert "--input cannot be combined with --torch-compile" in str(exc_info.value)


def test_worker_subprocess_stderr_is_mirrored_live(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.run_eval import _run_subprocess_with_live_stderr

    completed = _run_subprocess_with_live_stderr(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "print('worker stdout')\n"
                "print('worker stderr', file=sys.stderr)\n"
            ),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    captured = capsys.readouterr()
    assert completed.returncode == 0
    assert completed.stdout == "worker stdout\n"
    assert completed.stderr == "worker stderr\n"
    assert "worker stderr" in captured.err


@pytest.mark.parametrize(
    ("shutdown_mode", "expected_kill_calls"),
    [("terminates", 0), ("times_out", 1), ("terminate_error", 1)],
)
def test_subprocess_interrupt_terminates_child(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    shutdown_mode: str,
    expected_kill_calls: int,
) -> None:
    from scripts import run_eval as run_eval_module

    interrupt = KeyboardInterrupt("cancel evaluation")

    class InterruptingProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO("partial stdout\n")
            self.stderr = io.StringIO("partial stderr\n")
            self.wait_calls: list[float | None] = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                raise interrupt
            if shutdown_mode == "times_out" and len(self.wait_calls) == 2:
                raise subprocess.TimeoutExpired("worker", timeout=timeout)
            return -15

        def terminate(self) -> None:
            self.terminate_calls += 1
            if shutdown_mode == "terminate_error":
                raise OSError("cannot signal worker")

        def kill(self) -> None:
            self.kill_calls += 1

    process = InterruptingProcess()
    monkeypatch.setattr(
        run_eval_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        run_eval_module._run_subprocess_with_live_stderr(
            ["fake-worker"],
            cwd="/tmp",
        )

    assert exc_info.value is interrupt
    assert process.terminate_calls == 1
    assert process.kill_calls == expected_kill_calls
    assert process.wait_calls[0] is None
    assert all(timeout is not None and timeout > 0 for timeout in process.wait_calls[1:])
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    if shutdown_mode == "terminate_error":
        assert "cannot signal worker" in capsys.readouterr().err


def test_run_eval_skips_later_stages_after_compile_failure(tmp_path: Path) -> None:
    from scripts.run_eval import run_eval

    candidate_path = _write_candidate_file(tmp_path, "broken.py", "def broken(\n")
    reference_dir = _build_reference_dir(tmp_path / "broken_case", name="broken_from_meta")
    timestamp = "20260410-103100"
    result = run_eval(
        input_path=candidate_path,
        reference_dir=reference_dir,
        output_root=tmp_path,
        timestamp=timestamp,
    )

    assert result["kernel"]["name"] == "broken_from_meta"
    compile_block = result["passed"]["compile"]
    assert compile_block, "expected per-shape compile entries"
    for shape_compile in compile_block.values():
        assert shape_compile["status"] == "failed"
        assert shape_compile["reason"] is not None
    # All shape correctness must be marked skipped due to compile failure.
    correctness_status = result["passed"]["correctness"]
    assert correctness_status, "expected at least one shape entry"
    for shape_status in correctness_status.values():
        assert shape_status["status"] == "skipped"
        assert shape_status["reason"] == "Skipped because compile stage failed."
    for shape_status in result["passed"]["performance"].values():
        assert shape_status["status"] == "skipped"
        assert shape_status["reason"] == "Skipped because compile stage failed."
    # And performance shapes should have empty samples.
    perf_shapes = result["performance"]["shapes"]
    for shape_payload in perf_shapes.values():
        assert shape_payload["samples"] == []
    assert (tmp_path / timestamp / "broken_from_meta" / "eval_result.json").exists()
    assert (tmp_path / timestamp / "broken_from_meta" / "candidate.py").exists()
    assert (tmp_path / timestamp / "broken_from_meta" / "reference.py").exists()


def test_run_eval_writes_fallback_json_on_preflight_failure(tmp_path: Path) -> None:
    from scripts.run_eval import run_eval

    reference_dir = tmp_path / "missing_metadata"
    reference_dir.mkdir(parents=True)
    shutil.copy2(REFERENCE_PATH, reference_dir / "reference.py")
    timestamp = "20260410-103150"

    result = run_eval(
        input_path=CANDIDATE_PATH,
        reference_dir=reference_dir,
        output_root=tmp_path,
        timestamp=timestamp,
    )

    result_file = tmp_path / timestamp / "missing_metadata" / "eval_result.json"
    assert result_file.exists()

    saved = json.loads(result_file.read_text(encoding="utf-8"))
    assert saved["kernel"]["name"] == "missing_metadata"
    assert result["error"] == saved["error"]
    assert "Required reference file not found" in saved["error"]
    # passed.compile is per-shape failed because pre-flight blocked the worker.
    compile_block = saved["passed"]["compile"]
    assert isinstance(compile_block, dict)
    # May have zero shapes if shapes.json was missing in preflight;
    # if populated, every shape must be failed.
    for shape_compile in compile_block.values():
        assert shape_compile["status"] == "failed"


def test_run_eval_records_failed_shape_on_subworker_signal_exit(tmp_path: Path) -> None:
    """A SIGTERM inside the candidate kills the per-shape sub-worker only.

    The parent worker must record that shape with status='failed'
    and a reason that mentions the signal, while the eval as a whole still
    runs to completion (top-level error remains null).
    """
    from scripts.run_eval import run_eval

    candidate_path = _write_candidate_file(
        tmp_path,
        "signal_exit.py",
        """
import os
import signal

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        os.kill(os.getpid(), signal.SIGTERM)
        return x


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(16, 16)]


def get_init_inputs() -> list:
    return []
""".strip(),
    )
    reference_dir = _build_reference_dir(
        tmp_path / "signal_case",
        name="signal_exit_from_candidate",
    )
    timestamp = "20260410-103250"

    result = run_eval(
        input_path=candidate_path,
        reference_dir=reference_dir,
        output_root=tmp_path,
        warmup_iters=1,
        bench_iters=1,
        num_correctness_cases=1,
        timestamp=timestamp,
    )

    result_file = tmp_path / timestamp / "signal_exit_from_candidate" / "eval_result.json"
    assert result_file.exists()

    saved = json.loads(result_file.read_text(encoding="utf-8"))
    # Eval ran to completion — top-level error stays null.
    assert saved["error"] is None
    assert result["error"] is None
    # The sub-worker crashed (SIGTERM) before producing any tensor output,
    # so per-shape compile is "failed" even though the module-level import
    # succeeded.
    assert saved["passed"]["compile"]["0"]["status"] == "failed"
    assert saved["passed"]["compile"]["0"]["reason"] is not None
    # The single shape should be marked failed with the signal reflected in reason.
    shape_status = saved["passed"]["correctness"]["0"]
    assert shape_status["status"] == "failed"
    assert shape_status["reason"] is not None
    assert "sigterm" in shape_status["reason"].lower()
    # No correctness cases were captured for the crashing shape.
    assert saved["correctness"]["shapes"]["0"]["cases"] == []
    # No performance samples for the crashing shape either.
    assert saved["performance"]["shapes"]["0"]["samples"] == []


@pytest.mark.skipif(not _gpu_available, reason="requires CUDA/HIP GPU")
def test_run_eval_continues_after_one_shape_subworker_crashes(tmp_path: Path) -> None:
    """Shape 0 crashes via SIGTERM but shape 1 still runs end-to-end.

    The per-shape sub-worker design means a fault on one shape
    must not skip the remaining shapes. We use a multi-shape reference dir
    plus a candidate that targets shape 0's specific input dimensions
    (16x16) so it kills its sub-worker only on that shape; shape 1 (8x8)
    is computed normally and produces correctness + performance samples.
    """
    from scripts.run_eval import run_eval

    candidate_path = _write_candidate_file(
        tmp_path,
        "shape0_crashes.py",
        """
import os
import signal

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kill the sub-worker only when we get shape 0's inputs (16x16);
        # smaller inputs (8x8 = shape 1) run through cleanly.
        if x.shape[0] == 16:
            os.kill(os.getpid(), signal.SIGTERM)
        return torch.relu(x)
""".strip(),
    )

    reference_dir = _build_reference_dir(
        tmp_path / "multi_case",
        name="multi_shape_one_crashes",
    )
    # Override the single-shape shapes.json with two shapes.
    multi_shapes = {
        "0": {
            "description": "Crashes the per-shape sub-worker via SIGTERM",
            "init_kwargs": None,
            "input_kwargs": {"rows": 16, "cols": 16},
        },
        "1": {
            "description": "Runs cleanly so we can verify the loop continues",
            "init_kwargs": None,
            "input_kwargs": {"rows": 8, "cols": 8},
        },
    }
    (reference_dir / "shapes.json").write_text(
        json.dumps(multi_shapes), encoding="utf-8"
    )

    timestamp = "20260513-090000"
    result = run_eval(
        input_path=candidate_path,
        reference_dir=reference_dir,
        output_root=tmp_path,
        warmup_iters=1,
        bench_iters=1,
        num_correctness_cases=1,
        timestamp=timestamp,
    )

    result_file = tmp_path / timestamp / "multi_shape_one_crashes" / "eval_result.json"
    assert result_file.exists()
    saved = json.loads(result_file.read_text(encoding="utf-8"))

    # Eval ran end-to-end; top-level error remains null.
    assert saved["error"] is None
    assert result["error"] is None

    # Per-shape compile: shape 0 crashed (SIGTERM, no output) → failed;
    # shape 1 ran to completion → passed.
    assert saved["passed"]["compile"]["0"]["status"] == "failed"
    assert saved["passed"]["compile"]["0"]["reason"] is not None
    assert saved["passed"]["compile"]["1"]["status"] == "passed"
    assert saved["passed"]["compile"]["1"]["reason"] is None

    # Shape 0 — sub-worker died, recorded as failed with signal info.
    shape0 = saved["passed"]["correctness"]["0"]
    assert shape0["status"] == "failed"
    assert shape0["reason"] is not None
    assert "sigterm" in shape0["reason"].lower()
    assert saved["correctness"]["shapes"]["0"]["cases"] == []
    assert saved["performance"]["shapes"]["0"]["samples"] == []

    # Shape 1 — completes correctness + performance normally.
    shape1 = saved["passed"]["correctness"]["1"]
    assert shape1["status"] == "passed", shape1
    assert shape1["reason"] is None
    cases1 = saved["correctness"]["shapes"]["1"]["cases"]
    assert len(cases1) == 1
    assert cases1[0]["outputs"][0]["passed"] is True
    samples1 = saved["performance"]["shapes"]["1"]["samples"]
    assert len(samples1) == 1
    assert samples1[0]["end_to_end_time_ms"] is not None

    # Aggregator overall status: not all shapes passed, so exit code path
    # would be non-zero, but the JSON is fully populated for analysis.
    from scripts.run_eval import _payload_overall_passed

    assert _payload_overall_passed(saved) is False


@pytest.mark.skipif(not _gpu_available, reason="requires CUDA/HIP GPU")
def test_run_eval_records_seed_artifacts_no_checkpoint_files(tmp_path: Path) -> None:
    from scripts.run_eval import run_eval

    timestamp = "20260410-103300"
    reference_dir = _build_reference_dir(tmp_path / "default_case", name="default_path")

    result = run_eval(
        input_path=CANDIDATE_PATH,
        reference_dir=reference_dir,
        output_root=tmp_path,
        warmup_iters=1,
        bench_iters=1,
        num_correctness_cases=1,
        timestamp=timestamp,
    )

    # Both stages record seed-only artifacts; no .pt files on disk.
    cases = result["correctness"]["shapes"]["0"]["cases"]
    assert cases[0]["input_artifact"]["format"] == "manual_seed"
    assert isinstance(cases[0]["input_artifact"]["seed"], int)
    perf_art = result["performance"]["shapes"]["0"]["input_artifact"]
    assert perf_art["format"] == "manual_seed"
    assert isinstance(perf_art["seed"], int)
    op_dir = tmp_path / timestamp / "default_path"
    assert not (op_dir / "correctness").exists()
    assert not (op_dir / "performance").exists()


def test_eval_id_is_unique_across_invocations(tmp_path: Path) -> None:
    """Two run_eval invocations must produce two distinct eval_id values.

    Plain second-precision timestamps collide when invocations land in the
    same wall-clock second; the 8-hex random suffix protects against that.
    """
    from scripts.run_eval import run_eval

    reference_dir = _build_reference_dir(tmp_path / "uniq_case", name="uniq_path")

    eval_ids: set[str] = set()
    for index in range(2):
        result = run_eval(
            input_path=CANDIDATE_PATH,
            reference_dir=reference_dir,
            output_root=tmp_path / f"out_{index}",
            warmup_iters=1,
            bench_iters=1,
            num_correctness_cases=1,
            timestamp=f"20260410-10330{index}",
        )
        assert EVAL_ID_PATTERN.match(result["eval_id"]), result["eval_id"]
        eval_ids.add(result["eval_id"])

    assert len(eval_ids) == 2, f"eval_id collided: {eval_ids}"


def test_eval_id_is_stable_within_a_single_eval(tmp_path: Path) -> None:
    """The on-disk eval_result.json eval_id must equal the in-memory return value.

    The worker persists the payload incrementally; eval_id must stay the same
    across all those incremental saves so a downstream reader sees one stable
    identity per evaluation, not a different ID after every shape.
    """
    from scripts.run_eval import run_eval

    reference_dir = _build_reference_dir(tmp_path / "stable_case", name="stable_path")

    result = run_eval(
        input_path=CANDIDATE_PATH,
        reference_dir=reference_dir,
        output_root=tmp_path,
        warmup_iters=1,
        bench_iters=1,
        num_correctness_cases=1,
        timestamp="20260410-103400",
    )

    saved_path = tmp_path / "20260410-103400" / "stable_path" / "eval_result.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["eval_id"] == result["eval_id"]
    assert "timestamp" not in saved


# ---------------------------------------------------------------------------
# Worker wall-clock timeout: the compile stage (OS-level SIGKILL, process group)
# ---------------------------------------------------------------------------


def _write_shapes(tmp_path: Path, count: int) -> Path:
    reference_dir = tmp_path / "op"
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "shapes.json").write_text(
        json.dumps({str(i): {} for i in range(count)}), encoding="utf-8"
    )
    return reference_dir


def test_derived_worker_wall_timeout_covers_compile_plus_every_shape(
    tmp_path: Path,
) -> None:
    """The worker ceiling has to cover the one stage no per-shape budget does.

    compile runs once, before any shape, and used to be bounded by nothing.
    """
    from scripts.run_eval import (
        _derived_shape_wall_timeout_s,
        _derived_worker_wall_timeout_s,
    )

    reference_dir = _write_shapes(tmp_path, 3)
    shape_ceiling = _derived_shape_wall_timeout_s(
        candidate_timeout_s=60,
        perf_timeout_s=600,
        num_correctness_cases=5,
    )

    ceiling = _derived_worker_wall_timeout_s(
        reference_dir,
        compile_timeout_s=300,
        candidate_timeout_s=60,
        perf_timeout_s=600,
        num_correctness_cases=5,
    )

    # compile(300) + 3 shapes * 1020 + worker overhead(30)
    assert ceiling == 300 + shape_ceiling * 3 + 30


def test_derived_worker_wall_timeout_is_disabled_by_a_non_positive_budget(
    tmp_path: Path,
) -> None:
    """<= 0 means "no ceiling", the convention the other timeouts already use."""
    from scripts.run_eval import _derived_worker_wall_timeout_s

    assert (
        _derived_worker_wall_timeout_s(
            _write_shapes(tmp_path, 1),
            compile_timeout_s=0,
            candidate_timeout_s=60,
            perf_timeout_s=600,
            num_correctness_cases=1,
        )
        is None
    )


def test_derived_worker_wall_timeout_survives_an_unreadable_shapes_json(
    tmp_path: Path,
) -> None:
    """A backstop must not be the thing that refuses to run."""
    from scripts.run_eval import _derived_worker_wall_timeout_s

    ceiling = _derived_worker_wall_timeout_s(
        tmp_path / "missing",
        compile_timeout_s=300,
        candidate_timeout_s=60,
        perf_timeout_s=600,
        num_correctness_cases=1,
    )

    assert ceiling is not None and ceiling > 300


def test_worker_subprocess_is_killed_when_it_outlives_its_budget() -> None:
    """A wedged worker is OS-killed rather than waited on forever."""
    from scripts.run_eval import _run_subprocess_with_live_stderr

    with pytest.raises(subprocess.TimeoutExpired):
        _run_subprocess_with_live_stderr(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout_s=0.5,
        )


def test_worker_timeout_kills_the_process_group_not_just_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The group, so a compiler the worker spawned cannot reparent and survive.

    That reparenting is what let one wedged compiler keep holding a build lock
    long after the evaluation that started it was gone.
    """
    from scripts import run_eval as run_eval_module

    killed: list[int] = []
    monkeypatch.setattr(
        run_eval_module.os, "killpg", lambda pgid, _sig: killed.append(pgid)
    )
    monkeypatch.setattr(run_eval_module.os, "getpgid", lambda pid: pid)

    with pytest.raises(subprocess.TimeoutExpired):
        run_eval_module._run_subprocess_with_live_stderr(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout_s=0.5,
        )

    assert killed, "expected the child's process group to be signalled"


# ---------------------------------------------------------------------------
# Per-shape sub-worker wall-clock timeout (OS-level SIGKILL)
# ---------------------------------------------------------------------------


def test_derived_shape_wall_timeout_formula() -> None:
    """Wall ceiling = candidate * (1 + cases) + perf_timeout + ref_overhead.

    Phase budgets per shape:
      - candidate touch (instantiate + each correctness forward) <= candidate_timeout
      - perf phase (do_bench + profiler breakdown)               <= perf_timeout
      - reference cold-start                                      <= 60s fixed
    """
    from scripts.run_eval import _derived_shape_wall_timeout_s

    # Default: candidate=60, perf=600, cases=5
    # ceiling = 60*(1+5) + 600 + 60 = 1020
    assert _derived_shape_wall_timeout_s(
        candidate_timeout_s=60,
        perf_timeout_s=600,
        num_correctness_cases=5,
    ) == 1020.0

    # Bump perf to 1200 -> +600
    assert _derived_shape_wall_timeout_s(
        candidate_timeout_s=60,
        perf_timeout_s=1200,
        num_correctness_cases=5,
    ) == 1620.0

    # Bump candidate to 120 -> +60*(1+5) = +360
    assert _derived_shape_wall_timeout_s(
        candidate_timeout_s=120,
        perf_timeout_s=600,
        num_correctness_cases=5,
    ) == 1020.0 + 360.0

    # Degenerate input (all zero) -> just reference overhead, but floor.
    assert _derived_shape_wall_timeout_s(
        candidate_timeout_s=0,
        perf_timeout_s=0,
        num_correctness_cases=0,
    ) == 60.0


def test_shape_wall_timeout_excludes_disabled_stages() -> None:
    from scripts.run_eval import _derived_shape_wall_timeout_s

    assert _derived_shape_wall_timeout_s(
        60,
        600,
        num_correctness_cases=5,
        validation_mode="correctness_only",
    ) == 420.0
    assert _derived_shape_wall_timeout_s(
        60,
        600,
        num_correctness_cases=5,
        validation_mode="performance_only",
    ) == 720.0
    assert _derived_shape_wall_timeout_s(
        60,
        600,
        num_correctness_cases=5,
        validation_mode="full",
    ) == 1020.0


def test_single_shape_subprocess_synthesizes_failed_on_wall_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the per-shape sub-worker exceeds the wall budget, subprocess.run
    raises ``TimeoutExpired``; we must catch it and synthesize a
    ``status='failed'`` ``CorrectnessShapeResult`` instead of crashing.
    """
    import subprocess

    from scripts import run_eval as run_eval_module

    def fake_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0] if args else kwargs.get("args", ["fake"]),
            timeout=kwargs.get("timeout", 180.0),
            output=b"",
            stderr=b"line1\nline2\nline3 simulated trace\n",
        )

    monkeypatch.setattr(run_eval_module.subprocess, "run", fake_subprocess_run)

    shape_results_dir = tmp_path / ".shape_results"
    shape_results_dir.mkdir()
    cor, perf, compile_succeeded = run_eval_module._run_single_shape_subprocess(
        candidate_path=tmp_path / "candidate.py",
        reference_dir=tmp_path / "ref",
        artifact_dir=tmp_path / "artifacts",
        checkpoint_root=tmp_path / "ckpt",
        shape_results_dir=shape_results_dir,
        shape_id="0",
        atol=0.01,
        rtol=0.05,
        num_correctness_cases=5,
        warmup_iters=1,
        bench_iters=1,
        collect_kernel_events=False,
        candidate_timeout_s=60.0,
        perf_timeout_s=600.0,
    )

    # Derived ceiling: candidate(60) * (1 + 5) + perf(600) + ref(60) = 1020s
    expected_ceiling = 60 * (1 + 5) + 600 + 60
    assert cor.status == "failed"
    assert cor.reason is not None
    assert f"{float(expected_ceiling)}s wall-clock budget" in cor.reason
    assert "SIGKILL" in cor.reason
    # stderr tail should be surfaced
    assert "line3 simulated trace" in cor.reason
    # perf result is empty (no samples) on wall-clock kill
    assert perf.samples == []
    # compile status unknown when sub-worker was OS-killed
    assert compile_succeeded is None


def test_performance_only_subprocess_timeout_fails_performance_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(
        run_eval_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(
                cmd=args[0] if args else kwargs.get("args", ["fake"]),
                timeout=kwargs.get("timeout", 180.0),
            )
        ),
    )

    shape_results_dir = tmp_path / ".shape_results"
    shape_results_dir.mkdir()
    correctness, performance, _ = run_eval_module._run_single_shape_subprocess(
        candidate_path=tmp_path / "candidate.py",
        reference_dir=tmp_path / "ref",
        artifact_dir=tmp_path / "artifacts",
        checkpoint_root=tmp_path / "ckpt",
        shape_results_dir=shape_results_dir,
        shape_id="0",
        atol=0.01,
        rtol=0.05,
        num_correctness_cases=5,
        warmup_iters=1,
        bench_iters=1,
        collect_kernel_events=False,
        candidate_timeout_s=60.0,
        perf_timeout_s=600.0,
        validation_mode="performance_only",
    )

    assert correctness.status == "skipped"
    assert performance.error is not None
    assert "SIGKILL" in performance.error


# ----- _ptl_state() probe -----------------------------------------------------

_AMD_SMI_SAMPLE = """\
GPU: 0
    LIMIT:
        PPT0:
            MAX_POWER_LIMIT: 650 W
            MIN_POWER_LIMIT: 0 W
            SOCKET_POWER_LIMIT: 650 W
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SHUTDOWN_VRAM_TEMPERATURE: 115 °C
        PTL_STATE: N/A
        PTL_FORMAT: N/A
"""


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess used in monkeypatch."""

    def __init__(self, *, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def test_ptl_state_parses_amd_smi_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROCm backend + valid amd-smi output -> verbatim PTL_STATE value."""
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")

    captured_cmd: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured_cmd.append(cmd)
        return _FakeCompletedProcess(stdout=_AMD_SMI_SAMPLE)

    monkeypatch.setattr(run_eval_module.subprocess, "run", fake_run)

    assert run_eval_module._ptl_state() == "N/A"
    assert captured_cmd == [["amd-smi", "static", "-l"]]


def test_ptl_state_returns_none_on_non_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-ROCm backend -> probe is skipped (no subprocess call)."""
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "cuda")

    def fake_run(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("subprocess.run must not be called on non-ROCm backend")

    monkeypatch.setattr(run_eval_module.subprocess, "run", fake_run)
    assert run_eval_module._ptl_state() is None


def test_ptl_state_returns_none_when_amd_smi_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """amd-smi binary missing -> FileNotFoundError swallowed -> None."""
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")

    def fake_run(*args, **kwargs):  # noqa: ARG001
        raise FileNotFoundError("amd-smi")

    monkeypatch.setattr(run_eval_module.subprocess, "run", fake_run)
    assert run_eval_module._ptl_state() is None


def test_ptl_state_returns_none_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """amd-smi returns non-zero -> None even with stdout text."""
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")
    monkeypatch.setattr(
        run_eval_module.subprocess,
        "run",
        lambda *a, **kw: _FakeCompletedProcess(
            stdout="PTL_STATE: upgraded\n", returncode=1
        ),
    )
    assert run_eval_module._ptl_state() is None


def test_ptl_state_returns_none_when_field_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """amd-smi succeeds but output has no PTL_STATE line -> None."""
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")
    monkeypatch.setattr(
        run_eval_module.subprocess,
        "run",
        lambda *a, **kw: _FakeCompletedProcess(stdout="GPU: 0\n    LIMIT:\n"),
    )
    assert run_eval_module._ptl_state() is None


def test_ptl_state_handles_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """subprocess.TimeoutExpired -> swallowed -> None (probe must not crash run_eval)."""
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")

    def fake_run(*args, **kwargs):  # noqa: ARG001
        raise subprocess.TimeoutExpired(cmd=["amd-smi"], timeout=10)

    monkeypatch.setattr(run_eval_module.subprocess, "run", fake_run)
    assert run_eval_module._ptl_state() is None


def test_build_environment_includes_ptl_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_build_environment exposes PTL_STATE as a top-level environment key."""
    from scripts import run_eval as run_eval_module

    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")
    monkeypatch.setattr(
        run_eval_module.subprocess,
        "run",
        lambda *a, **kw: _FakeCompletedProcess(stdout=_AMD_SMI_SAMPLE),
    )

    env = run_eval_module._build_environment(clock_locked=True)
    assert "PTL_STATE" in env
    assert env["PTL_STATE"] == "N/A"
