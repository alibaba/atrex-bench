"""Tests for end-to-end run_eval pipeline."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

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

    # passed.compile.<shape_id> + passed.correctness.<shape_id>
    assert "0" in result["passed"]["compile"]
    assert result["passed"]["compile"]["0"]["status"] == "passed"
    assert result["passed"]["compile"]["0"]["reason"] is None
    assert "0" in result["passed"]["correctness"]
    assert result["passed"]["correctness"]["0"]["status"] == "passed"
    # performance is NOT in passed
    assert "performance" not in result["passed"]

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
    assert result_file.exists()
    assert candidate_file.exists()
    assert reference_file.exists()
    assert input_file.exists()
    assert shapes_file.exists()
    assert metadata_file.exists()
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
