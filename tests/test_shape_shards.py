"""Shape filtering and complete/partial report merging without a GPU or network."""

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


def test_gpu_name_formatting_keeps_original_shard_environments(evaluator):
    evaluate, _, _, _ = evaluator
    first, second = evaluate(["10", "31"]), evaluate(["2"])
    first["environment"]["gpu_name"] = "NVIDIA RTX PRO 5000 72GB Blackwell"
    second["environment"]["gpu_name"] = "  rtx-pro-5000 72GB blackwell  "

    merged = merge_evaluation_shards([first, second])

    assert [shard["environment"] for shard in merged["shard_evaluations"]] == [
        first["environment"], second["environment"],
    ]
    assert merged["environment"]["gpu_name"] == first["environment"]["gpu_name"]
    assert merged["performance"]["shapes"]["2"] == second["performance"]["shapes"]["2"]

    second["environment"]["gpu_name"] = "NVIDIA RTX PRO 5000 48GB Blackwell"
    with pytest.raises(ValueError, match=r"environment\.gpu_name"):
        merge_evaluation_shards([first, second])


@pytest.mark.parametrize("verified", [(True, False), (False, True), (True, True)])
def test_shard_clock_verification_is_aggregated_without_losing_evidence(evaluator, verified):
    evaluate, _, _, _ = evaluator
    first, second = evaluate(["10", "31"]), evaluate(["2"])
    for shard, status in zip((first, second), verified):
        shard["environment"].update(
            clock_locked=status,
            clock_lock_detected=status,
            clock_lock_verified=status,
            clock_lock={
                "mode": "manage",
                "source": "atrex-managed",
                "requested": {"graphics_mhz": 3090},
                "fail_on_deviation": False,
                "verified": status,
                "restored": True,
                "measurement": {"graphics_min_mhz": 3090 if status else 2580},
            },
        )

    merged = merge_evaluation_shards([first, second])
    env = merged["environment"]
    assert env["clock_lock"]["verified"] is all(verified)
    assert env["clock_locked"] is all(verified)
    assert env["clock_lock_detected"] is all(verified)
    assert env["clock_lock_verified"] is all(verified)
    assert [shard["environment"] for shard in merged["shard_evaluations"]] == [
        first["environment"], second["environment"],
    ]

    second["environment"]["clock_lock"]["restored"] = False
    with pytest.raises(ValueError, match=r"environment\.clock_lock"):
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
    assert merged["environment"]["driver_version"] is None
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


def _assert_shape_coverage(result, requested, completed, failed):
    selection = result["shape_selection"]
    assert selection["all_shape_ids"] == IDS
    assert selection["selected_shape_ids"] == requested
    assert selection["partial"] is (requested != IDS or bool(failed))
    assert result["runner_config"]["shape_ids"] == requested
    assert set(selection["input_seeds"]) == set(requested)
    assert selection.get("failed_shapes", []) == failed
    for stage in ("compile", "correctness", "performance"):
        assert list(result["passed"][stage]) == completed
    for stage in ("correctness", "performance"):
        assert list(result[stage]["shapes"]) == completed


@pytest.mark.parametrize("selection", [["31", "10"], "31,10", "@file"])
@pytest.mark.parametrize("mode", ["full", "correctness_only", "performance_only"])
def test_explicit_subset_merge_preserves_universe_and_serial_evidence(
    evaluator, tmp_path, selection, mode
):
    evaluate, reference, _, _ = evaluator
    source = (reference / "shapes.json").read_bytes()
    full = evaluate(mode=mode)
    shards = [evaluate([sid], mode=mode) for sid in ("31", "10")]
    before = copy.deepcopy(shards)
    if selection == "@file":
        path = tmp_path / "selected.json"
        path.write_text('["31", "10"]')
        selection = "@" + str(path)
    result = merge_evaluation_shards(shards, shape_ids=selection)
    requested = ["10", "31"]
    _assert_shape_coverage(result, requested, requested, [])
    for stage in ("compile", "correctness", "performance"):
        assert result["passed"][stage] == {
            sid: full["passed"][stage][sid] for sid in requested
        }
    for stage in ("correctness", "performance"):
        assert result[stage]["shapes"] == {
            sid: full[stage]["shapes"][sid] for sid in requested
        }
    assert result["shape_selection"]["input_seeds"] == {
        sid: full["shape_selection"]["input_seeds"][sid] for sid in requested
    }
    assert shards == before
    assert (reference / "shapes.json").read_bytes() == source


def test_explicit_full_selection_keeps_default_strict_merge(evaluator):
    evaluate, _, _, _ = evaluator
    shards = [evaluate(["10", "31"]), evaluate(["2"])]
    default = merge_evaluation_shards(shards)
    explicit = merge_evaluation_shards(shards, shape_ids=list(reversed(IDS)), failed_shapes=[])
    for key in ("passed", "correctness", "performance", "shape_selection", "runner_config"):
        assert explicit[key] == default[key]
    _assert_shape_coverage(explicit, IDS, IDS, [])
    with pytest.raises(ValueError, match="Missing evaluation shapes"):
        merge_evaluation_shards(shards[:1])


@pytest.mark.parametrize("mode", ["full", "correctness_only", "performance_only"])
@pytest.mark.parametrize("torch_compile", [False, True])
@pytest.mark.parametrize("requested", [IDS, ["10", "31"]])
def test_failed_shapes_account_for_missing_selection_without_fabricating_results(
    evaluator, mode, torch_compile, requested
):
    evaluate, _, _, _ = evaluator
    full = evaluate(mode=mode, torch_compile=torch_compile)
    completed = evaluate(["10"], mode=mode, torch_compile=torch_compile)
    failures = [
        {"shape_id": sid, "reason": "worker retry budget exhausted", "retries": 3}
        for sid in requested if sid != "10"
    ]
    before = copy.deepcopy(completed)
    result = merge_evaluation_shards(
        [completed], shape_ids=requested, failed_shapes=failures
    )
    _assert_shape_coverage(result, requested, ["10"], failures)
    assert result["shape_selection"]["input_seeds"] == {
        sid: full["shape_selection"]["input_seeds"][sid] for sid in requested
    }
    assert result["shard_evaluations"][0]["shape_ids"] == ["10"]
    assert completed == before
    json.dumps(result, allow_nan=False)


def test_partial_flydsl_ratio_aggregates_only_completed_shape_blocks(evaluator):
    evaluate, _, _, _ = evaluator
    shards = [evaluate([sid]) for sid in ('10', '31')]
    for shard, sid, ratio in zip(shards, ('10', '31'), (.2, .8)):
        shard['flydsl_compute_ratio'] = {
            'shapes': {sid: {'ratio': ratio, 'error': None}},
            'average': ratio, 'valid_shape_count': 1, 'total_shape_count': 1,
        }
    result = merge_evaluation_shards(
        shards, failed_shapes=[{'shape_id': '2', 'reason': 'worker lost', 'retries': 0}]
    )
    ratio = result['flydsl_compute_ratio']
    assert list(ratio['shapes']) == ['10', '31']
    assert ratio['average'] == pytest.approx(.5)
    assert ratio['valid_shape_count'] == ratio['total_shape_count'] == 2


@pytest.mark.parametrize(
    "failed",
    [
        [],
        [{"shape_id": "2", "reason": "lost", "retries": 0}],
        [{"shape_id": "31", "reason": "lost", "retries": 0}],
        [{"shape_id": "missing", "reason": "lost", "retries": 0}],
        [{"shape_id": "10", "reason": "lost", "retries": 0}],
        [{"shape_id": 31, "reason": "lost", "retries": 0}],
        [{"shape_id": "31", "retries": 0}],
        [{"reason": "lost", "retries": 0}],
        [None],
        ["31"],
        {"shape_id": "31", "reason": "lost", "retries": 0},
        [
            {"shape_id": "31", "reason": "lost", "retries": 0},
            {"shape_id": "31", "reason": "lost again", "retries": 1},
        ],
    ],
)
def test_partial_merge_rejects_missing_duplicate_or_malformed_failure_evidence(evaluator, failed):
    evaluate, _, _, _ = evaluator
    shard = evaluate(["10"])
    with pytest.raises(ValueError):
        merge_evaluation_shards([shard], shape_ids=IDS, failed_shapes=failed)


@pytest.mark.parametrize(
    "failure",
    [
        {"shape_id": "31", "reason": "", "retries": 0},
        {"shape_id": "31", "reason": "   ", "retries": 0},
        {"shape_id": "31", "reason": None, "retries": 0},
        {"shape_id": "31", "reason": 1, "retries": 0},
        {"shape_id": "31", "reason": "lost", "retries": -1},
        {"shape_id": "31", "reason": "lost", "retries": True},
        {"shape_id": "31", "reason": "lost", "retries": False},
        {"shape_id": "31", "reason": "lost", "retries": 1.0},
        {"shape_id": "31", "reason": "lost", "retries": "1"},
        {"shape_id": "31", "reason": "lost"},
    ],
)
def test_failure_fields_are_validated_even_when_ids_exactly_cover_missing(evaluator, failure):
    evaluate, _, _, _ = evaluator
    with pytest.raises(ValueError):
        merge_evaluation_shards(
            [evaluate(["10"])], shape_ids=["10", "31"], failed_shapes=[failure]
        )


@pytest.mark.parametrize("selection", [[], "", ["missing"], [2], ["10", "10"]])
def test_merge_rejects_invalid_requested_selection(evaluator, selection):
    evaluate, _, _, _ = evaluator
    with pytest.raises(ValueError):
        merge_evaluation_shards([evaluate(["10"])], shape_ids=selection)


def test_partial_merge_rejects_completed_or_failed_shapes_outside_request(evaluator):
    evaluate, _, _, _ = evaluator
    with pytest.raises(ValueError):
        merge_evaluation_shards([evaluate(["10", "2"])], shape_ids=["10"])
    with pytest.raises(ValueError):
        merge_evaluation_shards(
            [evaluate(["10"])], shape_ids=["10", "31"],
            failed_shapes=[
                {"shape_id": "31", "reason": "lost", "retries": 0},
                {"shape_id": "2", "reason": "lost", "retries": 0},
            ],
        )
    with pytest.raises(ValueError):
        merge_evaluation_shards(
            [evaluate(["10"])], shape_ids=["10"],
            failed_shapes=[{"shape_id": "10", "reason": "lost", "retries": 0}],
        )
    with pytest.raises(ValueError):
        merge_evaluation_shards(
            [evaluate(["10"]), evaluate(["10"])], shape_ids=["10"]
        )


def test_all_failed_requires_at_least_one_completed_shard():
    with pytest.raises(ValueError, match="[Aa]t least one.*shard"):
        merge_evaluation_shards(
            [], shape_ids=IDS,
            failed_shapes=[{"shape_id": sid, "reason": "lost", "retries": 0} for sid in IDS],
        )


def test_partial_merge_can_write_offline_report(evaluator, tmp_path):
    evaluate, _, _, _ = evaluator
    path = tmp_path / "completed.json"
    path.write_text(json.dumps(evaluate(["10"])))
    output = tmp_path / "partial.json"
    failures = [{"shape_id": "31", "reason": "lost", "retries": 0}]
    result = merge_evaluation_shards(
        [path], output=output, shape_ids=["31", "10"], failed_shapes=failures
    )
    assert json.loads(output.read_text()) == result
    _assert_shape_coverage(result, ["10", "31"], ["10"], failures)


@pytest.mark.parametrize(
    "requested,completed",
    [(["10", "31"], ["10", "31"]), (IDS, ["10", "31"]), (IDS, ["10"])],
)
def test_abba_partial_merge_propagates_selection_failures_and_aggregates_only_completed(
    evaluator, monkeypatch, requested, completed
):
    evaluate, reference, candidate, _ = evaluator
    baseline = candidate.parent / "baseline.py"
    baseline.write_bytes(candidate.read_bytes())

    def process(**kwargs):
        result = evaluate(kwargs["shape_ids"])
        factor = (int(kwargs["shape_ids"][0]) + 1) if kwargs["input_path"] == baseline else 1
        for shape in result["performance"]["shapes"].values():
            shape["samples"][0]["end_to_end_time_ms"] *= factor
        return result

    monkeypatch.setattr(runner, "_run_eval_process", process)
    shards = [runner.run_abba_eval(
        baseline_path=baseline, candidate_path=candidate, reference_dir=reference,
        output_root=reference.parent / "abba", timestamp="shard-" + sid,
        shape_ids=[sid], collect_kernel_events=True,
    ) for sid in completed]
    expected_runs = [
        {**step, "result": merge_evaluation_shards(
            [shard["abba"]["runs"][index]["result"] for shard in shards], shape_ids=completed
        )}
        for index, step in enumerate(runner._abba_schedule())
    ]
    expected, error = runner._aggregate_abba_runs(expected_runs, completed)
    assert error is None
    failures = [
        {"shape_id": sid, "reason": "worker lost", "retries": 2}
        for sid in requested if sid not in completed
    ]
    result = merge_evaluation_shards(
        list(reversed(shards)), shape_ids=list(reversed(requested)), failed_shapes=failures
    )
    selection = result["shape_selection"]
    assert selection["all_shape_ids"] == IDS
    assert selection["selected_shape_ids"] == requested
    assert selection["partial"] is True
    assert selection.get("failed_shapes", []) == failures
    assert result["runner_config"]["shape_ids"] == requested
    assert result["passed"]["abba"]["status"] == "passed"
    for key in ("baseline", "candidate", "comparison"):
        assert result["abba"][key] == expected[key]
    assert list(result["abba"]["comparison"]["shapes"]) == completed
    for row in result["abba"]["runs"]:
        _assert_shape_coverage(row["result"], requested, completed, failures)
        assert set(row["result"]["shape_selection"]["input_seeds"]) == set(requested)


@pytest.mark.parametrize("field", [
    "source_hashes", "evaluator_version", "evaluator_sha256", "seed_policy",
])
@pytest.mark.parametrize("partial", [False, True])
def test_merging_rejects_missing_provenance_even_when_every_shard_omits_it(
    evaluator, field, partial
):
    evaluate, _, _, _ = evaluator
    reports = [evaluate([sid]) for sid in (IDS if not partial else ["10"])]
    for report in reports:
        report["shape_selection"].pop(field)
    with pytest.raises(ValueError, match="Missing immutable shard provenance"):
        merge_evaluation_shards(reports, shape_ids=["10"] if partial else None)


@pytest.mark.parametrize("mutation", [
    "candidate_hash", "reference_hash", "evaluator_version", "evaluator_sha256",
    "extra_run", "missing_run",
])
def test_abba_merge_rejects_mixed_revisions_and_unreported_executions(
    evaluator, monkeypatch, mutation
):
    evaluate, reference, candidate, _ = evaluator
    monkeypatch.setattr(runner, "_run_eval_process",
                        lambda **kwargs: evaluate(kwargs["shape_ids"]))
    report = runner.run_abba_eval(
        baseline_path=candidate, candidate_path=candidate, reference_dir=reference,
        output_root=reference.parent / "abba", timestamp="provenance", shape_ids=["10"],
    )
    if mutation == "extra_run":
        report["abba"]["runs"].append(copy.deepcopy(report["abba"]["runs"][-1]))
        report["abba"]["runs"][-1]["result"]["error"] = "unreported execution failure"
    elif mutation == "missing_run":
        report["abba"]["runs"].pop()
    else:
        selection = report["abba"]["runs"][2]["result"]["shape_selection"]
        if mutation.endswith("_hash"):
            key = "candidate.py" if mutation == "candidate_hash" else "reference.py"
            selection["source_hashes"][key] += "different"
        else:
            selection[mutation] += "different"
    with pytest.raises(ValueError, match="Incompatible ABBA"):
        merge_evaluation_shards([report], shape_ids=["10"])
