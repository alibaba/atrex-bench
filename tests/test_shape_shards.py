"""Shape filtering and complete report merging without a GPU or network."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from atrex_bench import SHAPE_SHARD_SCHEMA_VERSION, merge_evaluation_shards
from atrex_bench.cli import run_eval as runner
from atrex_bench.eval import (
    CompileResult,
    CorrectnessShapeResult,
    PerformanceSample,
    PerformanceShapeResult,
    check_correctness,
)
from atrex_bench.eval._runtime import deterministic_input_seed
from atrex_bench.shards import select_shape_ids

IDS = ["10", "2", "31"]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    reference = tmp_path / "operator"
    shutil.copytree(FIXTURES / "references" / "atrex_001", reference)
    shape = json.loads((reference / "shapes.json").read_text())["0"]
    (reference / "shapes.json").write_text(json.dumps({sid: shape for sid in IDS}))
    candidate = tmp_path / "candidate.py"
    shutil.copy2(FIXTURES / "generations" / "atrex_001.py", candidate)
    environment = {"gpu_name": "TestGPU", "gpu_arch": "test", "torch_version": "test"}
    monkeypatch.setattr(runner, "_build_environment", lambda **kwargs: environment.copy())
    monkeypatch.setattr(runner, "check_compilation", lambda *args: CompileResult("passed"))
    calls = []

    def evaluate_shape(**kwargs):
        sid = kwargs["shape_id"]
        calls.append(sid)
        mode = kwargs.get("validation_mode", "full")
        correctness = (
            check_correctness(
                reference / "reference.py",
                candidate,
                shape_id=sid,
                device="cpu",
                atol=kwargs.get("atol", 0.002),
                rtol=kwargs.get("rtol", 0.003),
                num_correctness_cases=kwargs.get("num_correctness_cases", 2),
            )
            if mode != "performance_only"
            else CorrectnessShapeResult("skipped")
        )
        performance = (
            PerformanceShapeResult(
                input_artifact={
                    "seed": deterministic_input_seed("performance", sid),
                    "format": "manual_seed",
                },
                samples=[PerformanceSample(float(sid))],
            )
            if mode != "correctness_only"
            else PerformanceShapeResult()
        )
        return correctness, performance, None

    monkeypatch.setattr(runner, "_run_single_shape_subprocess", evaluate_shape)
    monkeypatch.setattr(
        runner,
        "_run_single_shape_torch_compile_subprocess",
        lambda **kwargs: evaluate_shape(**kwargs, validation_mode="performance_only")[:2],
    )

    def evaluate(selected=None, mode="full", torch_compile=False):
        artifact = tmp_path / f"artifact-{len(list(tmp_path.glob('artifact-*')))}"
        artifact.mkdir()
        common = dict(
            reference_dir=reference,
            artifact_dir=artifact,
            warmup_iters=3,
            bench_iters=7,
            checkpoint_dir=None,
            config_version="v1",
            clock_locked=False,
            shape_ids=selected,
        )
        if torch_compile:
            return runner._run_torch_compile_worker(**common)
        return runner._run_eval_worker(
            **common,
            input_path=candidate,
            atol=0.002,
            rtol=0.003,
            num_correctness_cases=2,
            require_clock_locked=False,
            collect_kernel_events=True,
            candidate_timeout_s=60,
            perf_timeout_s=600,
            validation_mode=mode,
        )

    return evaluate, reference, candidate, calls


@pytest.mark.parametrize("selection", ["31,10", ["31", "10"]])
def test_selection_preserves_ids_and_full_order(selection):
    assert select_shape_ids(IDS, selection) == ["10", "31"]


@pytest.mark.parametrize("content", ['["31", "10"]', "31\n10\n"])
def test_selection_file(tmp_path, content):
    path = tmp_path / "ids.txt"
    path.write_text(content)
    assert select_shape_ids(IDS, "@" + str(path)) == ["10", "31"]


@pytest.mark.parametrize("selection", [[], "", "2,2", ["missing"], [2]])
def test_invalid_selection_rejected(selection):
    with pytest.raises(ValueError):
        select_shape_ids(IDS, selection)


@pytest.mark.parametrize("mode", ["full", "correctness_only", "performance_only"])
@pytest.mark.parametrize("torch_compile", [False, True])
def test_sharded_report_matches_serial_and_original_seeds(evaluator, mode, torch_compile):
    evaluate, reference, _, calls = evaluator
    source = (reference / "shapes.json").read_bytes()
    full = evaluate(mode=mode, torch_compile=torch_compile)
    calls.clear()
    first = evaluate(["31", "10"], mode=mode, torch_compile=torch_compile)
    assert calls == ["10", "31"]
    assert first["shape_selection"]["partial"] is True
    assert first["shape_selection"]["schema_version"] == SHAPE_SHARD_SCHEMA_VERSION
    second = evaluate(["2"], mode=mode, torch_compile=torch_compile)
    merged = merge_evaluation_shards([second, first])
    for key in ("passed", "correctness", "performance"):
        assert merged[key] == full[key]
    assert merged["shape_selection"]["input_seeds"] == full["shape_selection"]["input_seeds"]
    assert merged["shape_selection"]["partial"] is False
    assert list(merged["performance"]["shapes"]) == IDS
    assert (reference / "shapes.json").read_bytes() == source


def test_merge_files_and_output(evaluator, tmp_path):
    evaluate, _, _, _ = evaluator
    paths = []
    for index, selected in enumerate([["10", "31"], ["2"]]):
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps(evaluate(selected)))
        paths.append(path)
    output = tmp_path / "merged.json"
    merged = merge_evaluation_shards(paths, output=output)
    assert json.loads(output.read_text()) == merged


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "source",
        "config",
        "env",
        "version",
        "seed",
        "artifact_seed",
        "coverage",
    ],
)
def test_merge_rejects_incomplete_or_incompatible_shards(evaluator, mutation):
    evaluate, _, _, _ = evaluator
    first, second = evaluate(["10", "31"]), evaluate(["2"])
    if mutation == "duplicate":
        second = copy.deepcopy(first)
    elif mutation == "missing":
        with pytest.raises(ValueError, match="Missing evaluation shapes"):
            merge_evaluation_shards([first])
        return
    elif mutation == "source":
        second["shape_selection"]["source_hashes"]["candidate.py"] = "changed"
    elif mutation == "config":
        second["runner_config"]["rtol"] = 0.1
    elif mutation == "env":
        second["environment"]["torch_version"] = "changed"
    elif mutation == "version":
        second["shape_selection"]["evaluator_sha256"] = "changed"
    elif mutation == "seed":
        second["shape_selection"]["input_seeds"]["2"]["performance"] = 42
    elif mutation == "artifact_seed":
        second["performance"]["shapes"]["2"]["input_artifact"]["seed"] = 42
    else:
        second["passed"]["compile"] = {}
    with pytest.raises(ValueError):
        merge_evaluation_shards([first, second])


def test_different_devices_allowed_with_same_clock_targets(evaluator):
    evaluate, _, _, _ = evaluator
    first, second = evaluate(["10", "31"]), evaluate(["2"])
    for result, device in ((first, "GPU-A"), (second, "GPU-B")):
        result["environment"].update(
            gpu_uuid=device,
            gpu_index=device,
            clock_lock={
                "device": device,
                "requested": {"graphics_mhz": 1500},
                "measurement": {"trace": device},
                "verified": True,
            },
        )
    merged = merge_evaluation_shards([first, second])
    assert len(merged["shard_evaluations"]) == 2
    second["environment"]["clock_lock"]["requested"]["graphics_mhz"] = 1400
    with pytest.raises(ValueError, match="Incompatible"):
        merge_evaluation_shards([first, second])


@pytest.mark.parametrize(
    "versions",
    [("580.126.09", "580.95.05"), ("580.126.09", "580.126.09"), (None, "580.126.09")],
)
def test_node_local_driver_versions_are_preserved(evaluator, versions):
    evaluate, _, _, _ = evaluator
    first, second = evaluate(["10", "31"]), evaluate(["2"])
    for result, version in zip((first, second), versions):
        result["environment"]["driver_version"] = version
    merged = merge_evaluation_shards([first, second])
    assert [
        shard["environment"]["driver_version"] for shard in merged["shard_evaluations"]
    ] == list(versions)
    expected_version = versions[0] if versions[0] == versions[1] else None
    assert merged["environment"]["driver_version"] == expected_version
    assert all(row["status"] == "passed" for row in merged["passed"]["correctness"].values())


def test_failed_shape_is_retained(evaluator):
    evaluate, _, _, _ = evaluator
    first, second = evaluate(["10", "31"]), evaluate(["2"])
    second["passed"]["correctness"]["2"] = {"status": "failed", "reason": "incorrect"}
    merged = merge_evaluation_shards([first, second])
    assert merged["passed"]["correctness"]["2"]["status"] == "failed"
    assert runner._payload_overall_passed(merged) is False


def test_abba_each_leg_filters_and_merge_recomputes_global_means(evaluator, monkeypatch):
    evaluate, reference, candidate, _ = evaluator
    calls = []

    def fake_process(**kwargs):
        calls.append(kwargs["shape_ids"])
        result = evaluate(kwargs["shape_ids"])
        # Baseline and candidate legs have independent latency signals.
        factor = 2 if kwargs["input_path"].name == "baseline.py" else 1
        for shape in result["performance"]["shapes"].values():
            shape["samples"][0]["end_to_end_time_ms"] *= factor
        return result

    monkeypatch.setattr(runner, "_run_eval_process", fake_process)
    common = dict(
        baseline_path=candidate,
        candidate_path=candidate,
        reference_dir=reference,
        output_root=reference.parent / "abba",
        collect_kernel_events=True,
    )
    full = runner.run_abba_eval(**common, timestamp="full")
    first = runner.run_abba_eval(**common, timestamp="shard0", shape_ids=["10", "31"])
    second = runner.run_abba_eval(**common, timestamp="shard1", shape_ids=["2"])
    assert calls == [None] * 4 + [["10", "31"]] * 4 + [["2"]] * 4
    for shard, driver_version in ((first, "580.126.09"), (second, "580.95.05")):
        shard["environment"]["driver_version"] = driver_version
        for leg in shard["abba"]["runs"]:
            leg["result"]["environment"]["driver_version"] = driver_version
    merged = merge_evaluation_shards([first, second])
    assert merged["abba"]["comparison"] == full["abba"]["comparison"]
    assert merged["abba"]["baseline"] == full["abba"]["baseline"]
    assert merged["abba"]["candidate"] == full["abba"]["candidate"]
    assert runner._payload_overall_passed(merged)
    first["error"] = "Clock monitor detected a deviation"
    failed = merge_evaluation_shards([first, second])
    assert failed["abba"]["comparison"] is None
    assert failed["passed"]["abba"]["status"] == "failed"
    assert "Clock monitor" in failed["error"]


def test_cli_and_sdk_real_failed_compile_only_selected_ids(evaluator, tmp_path):
    _, reference, candidate, _ = evaluator
    candidate.write_text("def syntax_error(\n")
    config = {
        "input": str(candidate),
        "reference_dir": str(reference),
        "output": str(tmp_path / "real"),
        "shape_ids": ["31", "10"],
        "validation_mode": "correctness_only",
    }
    from atrex_bench import evaluate

    result = evaluate(config)
    assert list(result["passed"]["compile"]) == ["10", "31"]
    assert result["shape_selection"]["all_shape_ids"] == IDS
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "atrex_bench.cli.run_eval",
            "--config",
            str(config_path),
            "--shape-ids",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    output = completed.stdout.split("[OUTPUT] ")[-1].strip()
    cli_result = json.loads(Path(output).read_text())
    assert list(cli_result["passed"]["compile"]) == ["2"]
