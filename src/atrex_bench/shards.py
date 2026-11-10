"""Explicit shape selection and strict, offline evaluation report merging."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

SHAPE_SHARD_SCHEMA_VERSION = "v1"
_CLOCK_OBSERVATIONS = ("clock_locked", "clock_lock_detected", "clock_lock_verified")


@lru_cache(maxsize=1)
def _evaluator_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_shape_ids(all_shape_ids: Sequence[str], selection: object = None) -> list[str]:
    """Resolve IDs from a list, comma-separated string, or @JSON/text file.

    Results always follow the original shapes.json string ordering. IDs are
    never renumbered, and the reference bundle is never rewritten.
    """
    if selection is None:
        return list(all_shape_ids)
    if isinstance(selection, str):
        if selection.startswith("@"):
            content = Path(selection[1:]).read_text(encoding="utf-8").strip()
            selection = json.loads(content) if content.lstrip().startswith("[") else content
        if isinstance(selection, str):
            selection = selection.replace("\n", ",").split(",")
            selection = [item.strip() for item in selection]
    if not isinstance(selection, (list, tuple)) or not selection:
        raise ValueError("shape_ids must be a non-empty list, comma-separated IDs, or @file")
    if any(not isinstance(item, str) or not item for item in selection):
        raise ValueError("shape_ids must contain non-empty strings")
    if len(set(selection)) != len(selection):
        raise ValueError("shape_ids contains duplicate IDs")
    unknown = set(selection).difference(all_shape_ids)
    if unknown:
        raise ValueError("Unknown shape_ids: " + ", ".join(sorted(unknown)))
    selected = set(selection)
    return [item for item in all_shape_ids if item in selected]


def build_shape_selection(
    reference_dir: Path,
    candidate_path: Path | None,
    runner_config: Mapping[str, object],
    selected_shape_ids: Sequence[str],
    *,
    baseline_path: Path | None = None,
) -> dict[str, object]:
    """Attach immutable inputs and the existing per-shape seed policy."""
    from atrex_bench import __version__
    from atrex_bench.eval._runtime import deterministic_input_seed

    shapes_path = reference_dir / "shapes.json"
    all_ids = sorted(json.loads(shapes_path.read_text())) if shapes_path.is_file() else []
    source_paths = {
        filename: reference_dir / filename
        for filename in ("reference.py", "input.py", "shapes.json", "metadata.json")
    }
    if candidate_path is not None:
        source_paths["candidate.py"] = candidate_path
    if baseline_path is not None:
        source_paths["baseline.py"] = baseline_path
    hashes = {
        filename: hashlib.sha256(path.read_bytes()).hexdigest()
        for filename, path in source_paths.items()
        if path.is_file()
    }
    correctness_cases = int(runner_config.get("num_correctness_cases", 0))
    if correctness_cases and (reference_dir / "metadata.json").is_file():
        metadata = json.loads((reference_dir / "metadata.json").read_text())
        minimum = metadata.get("benchmark_contract", {}).get("correctness_min_cases", 1)
        if isinstance(minimum, int):
            correctness_cases = max(correctness_cases, minimum)
    validation = runner_config.get("validation_mode", "full")
    seeds = {
        shape_id: {
            "correctness": [
                deterministic_input_seed("correctness", shape_id, index)
                for index in range(correctness_cases)
            ]
            if validation != "performance_only"
            else [],
            "performance": (
                deterministic_input_seed("performance", shape_id, 0)
                if validation != "correctness_only"
                else None
            ),
        }
        for shape_id in selected_shape_ids
    }
    return {
        "schema_version": SHAPE_SHARD_SCHEMA_VERSION,
        "all_shape_ids": all_ids,
        "selected_shape_ids": list(selected_shape_ids),
        "partial": list(selected_shape_ids) != all_ids,
        "seed_policy": "sha256(stage|shape_id|case_index)-int31",
        "correctness_case_count": correctness_cases if validation != "performance_only" else 0,
        "input_seeds": seeds,
        "source_hashes": hashes,
        "evaluator_version": __version__,
        "evaluator_sha256": _evaluator_sha256(),
    }


def _compatibility(result: Mapping[str, object]) -> dict[str, object]:
    selection = result["shape_selection"]
    config = dict(result["runner_config"])
    config.pop("shape_ids", None)
    environment = _environment_identity(result["environment"])
    for key in ("driver_version", *_CLOCK_OBSERVATIONS):
        environment.pop(key, None)
    name = environment.get("gpu_name")
    if isinstance(name, str):
        # Ignore spelling differences while retaining model and capacity qualifiers.
        parts = name.casefold().replace("-", " ").split()
        if parts and parts[0] == "nvidia":
            parts = parts[1:]
        environment["gpu_name"] = " ".join(parts)
    if "clock_lock" in environment:
        environment["clock_lock"].pop("verified")
    return {
        "shape_selection": {
            key: value
            for key, value in selection.items()
            if key not in {"selected_shape_ids", "input_seeds", "partial"}
        },
        "runner_config": config,
        "environment": environment,
        "kernel": result["kernel"],
        "dsl": result["dsl"],
        "eval_mode": result["eval_mode"],
    }


def _environment_identity(environment: Mapping[str, object]) -> dict[str, object]:
    """Select aggregate environment fields without device identity or traces."""
    fields = {
        "gpu_name",
        "gpu_arch",
        "accelerator_backend",
        "PTL_STATE",
        *_CLOCK_OBSERVATIONS,
        "clock_lock_source",
    }
    stable = {
        key: value
        for key, value in environment.items()
        if key in fields or key.endswith("_version")
    }
    report = environment.get("clock_lock")
    if isinstance(report, Mapping):
        stable["clock_lock"] = {
            key: report.get(key)
            for key in ("mode", "source", "requested", "fail_on_deviation", "verified", "restored")
        }
    return stable


def _all_observed(values: Sequence[object]) -> bool | None:
    if all(value is None for value in values):
        return None
    return all(value is True for value in values)


def _merge_environments(environments: Sequence[Mapping[str, object]]) -> dict[str, object]:
    merged = _environment_identity(environments[0])
    # Individual driver versions remain in shard_evaluations.
    merged["driver_version"] = None
    for key in _CLOCK_OBSERVATIONS:
        if any(key in env for env in environments):
            merged[key] = _all_observed([env.get(key) for env in environments])
    if "clock_lock" in merged:
        merged["clock_lock"]["verified"] = _all_observed([
            env["clock_lock"].get("verified") for env in environments
        ])
    return merged


def _merge_shape_blocks(results, path, all_ids):
    shapes = {}
    for result in results:
        block = result
        for key in path:
            block = block[key]
        selected = result["shape_selection"]["selected_shape_ids"]
        if set(block) != set(selected):
            raise ValueError("Shard shape coverage differs from metadata: " + ".".join(path))
        shapes.update(block)
    return {shape_id: shapes[shape_id] for shape_id in all_ids}


def _validate_seeds(result: Mapping[str, object]) -> None:
    from atrex_bench.eval._runtime import deterministic_input_seed

    selection = result["shape_selection"]
    validation = result["runner_config"]["validation_mode"]
    for shape_id in selection["selected_shape_ids"]:
        expected = {
            "correctness": [
                deterministic_input_seed("correctness", shape_id, index)
                for index in range(selection["correctness_case_count"])
            ],
            "performance": (
                deterministic_input_seed("performance", shape_id, 0)
                if validation != "correctness_only"
                else None
            ),
        }
        if selection["input_seeds"][shape_id] != expected:
            raise ValueError(f"Incompatible input seeds for shape {shape_id!r}")
        if result["eval_mode"] == "abba":
            continue  # Each leg is validated during its merge below.
        cases = result["correctness"]["shapes"][shape_id]["cases"]
        for index, case in enumerate(cases):
            artifact = case.get("input_artifact")
            if artifact is not None and (
                index >= len(expected["correctness"])
                or artifact.get("seed") != expected["correctness"][index]
            ):
                raise ValueError(f"Correctness artifact seed mismatch for shape {shape_id!r}")
        artifact = result["performance"]["shapes"][shape_id].get("input_artifact")
        if artifact is not None and artifact.get("seed") != expected["performance"]:
            raise ValueError(f"Performance artifact seed mismatch for shape {shape_id!r}")


def merge_evaluation_shards(
    shards: Sequence[Mapping[str, object] | str | Path],
    *,
    shape_ids: object = None,
    failed_shapes: Sequence[Mapping[str, object]] | None = None,
    output: str | Path | None = None,
) -> dict[str, object]:
    """Merge disjoint shards with strict coverage of the requested shape selection.

    Missing selected shapes require explicit failure evidence; duplicate shapes
    and incompatible inputs still fail. ABBA preserves every completed leg.
    """
    if not shards:
        raise ValueError("At least one evaluation shard is required")
    results = [
        json.loads(Path(shard).read_text(encoding="utf-8"))
        if isinstance(shard, (str, Path))
        else dict(shard)
        for shard in shards
    ]
    first = results[0]
    try:
        all_ids = first["shape_selection"]["all_shape_ids"]
        if not all_ids or len(set(all_ids)) != len(all_ids):
            raise ValueError("Invalid complete shape set")
        selected_ids = select_shape_ids(all_ids, shape_ids)
        failures = [] if failed_shapes is None else list(failed_shapes)
        failed_ids = set()
        for failure in failures:
            if not isinstance(failure, Mapping):
                raise ValueError("Failed shapes must be objects")
            shape_id = failure.get("shape_id")
            if (not isinstance(shape_id, str) or shape_id not in selected_ids
                    or shape_id in failed_ids):
                raise ValueError("Invalid or duplicate failed shape ID")
            if not isinstance(failure.get("reason"), str) or not failure["reason"].strip():
                raise ValueError("Failed shapes require a non-empty reason")
            if type(failure.get("retries")) is not int or failure["retries"] < 0:
                raise ValueError("Failed shapes require a non-negative integer retry count")
            failed_ids.add(shape_id)
        identity = _compatibility(first)
        seen = set()
        for result in results:
            selection = result["shape_selection"]
            for key in ("seed_policy", "evaluator_version", "evaluator_sha256"):
                if not isinstance(selection.get(key), str) or not selection[key]:
                    raise ValueError(f"Missing immutable shard provenance: {key}")
            hashes = selection.get("source_hashes")
            if not isinstance(hashes, Mapping) or any(
                not isinstance(hashes.get(key), str) or not hashes[key]
                for key in ("reference.py", "input.py", "shapes.json")
            ):
                raise ValueError("Missing immutable shard provenance: source_hashes")
            actual = _compatibility(result)
            if (
                selection["schema_version"] != SHAPE_SHARD_SCHEMA_VERSION
                or actual != identity
            ):
                differences = []
                for section, expected in identity.items():
                    observed = actual[section]
                    if observed == expected:
                        continue
                    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
                        differences.extend(
                            f"{section}.{key}"
                            for key in sorted(set(expected) | set(observed))
                            if expected.get(key) != observed.get(key)
                        )
                    else:
                        differences.append(section)
                raise ValueError(
                    "Incompatible evaluation shards: inputs, config, environment or version"
                    + " ("
                    + ", ".join(differences or ["shape_selection.schema_version"])
                    + ")"
                )
            selected = selection["selected_shape_ids"]
            if not selected or len(set(selected)) != len(selected) or seen.intersection(selected):
                raise ValueError("Duplicate or empty shape selection in evaluation shards")
            if set(selected).difference(all_ids):
                raise ValueError("Shard declares shapes outside the complete shape set")
            if set(selection["input_seeds"]) != set(selected):
                raise ValueError("Shard seed coverage differs from selected shapes")
            _validate_seeds(result)
            seen.update(selected)
        if seen.difference(selected_ids):
            raise ValueError("Shard declares shapes outside the requested selection")
        if seen.intersection(failed_ids):
            raise ValueError("Failed shapes overlap completed evaluation shapes")
        missing = set(selected_ids).difference(seen, failed_ids)
        if missing:
            raise ValueError("Missing evaluation shapes: " + ", ".join(sorted(missing)))
        completed_ids = [shape_id for shape_id in selected_ids if shape_id in seen]

        merged = copy.deepcopy(first)
        selection = merged["shape_selection"]
        selection["selected_shape_ids"] = list(selected_ids)
        selection["partial"] = selected_ids != all_ids or bool(failures)
        if failures:
            selection["failed_shapes"] = [dict(failure) for failure in failures]
        selection["input_seeds"] = _merge_shape_blocks(
            results, ("shape_selection", "input_seeds"), completed_ids
        )
        from atrex_bench.eval._runtime import deterministic_input_seed

        for shape_id in selected_ids:
            if shape_id in failed_ids:
                selection["input_seeds"][shape_id] = {
                    "correctness": [
                        deterministic_input_seed("correctness", shape_id, index)
                        for index in range(selection["correctness_case_count"])
                    ],
                    "performance": (
                        deterministic_input_seed("performance", shape_id, 0)
                        if merged["runner_config"]["validation_mode"] != "correctness_only"
                        else None
                    ),
                }
        merged["runner_config"]["shape_ids"] = list(selected_ids)
        merged["shard_evaluations"] = [
            {
                "eval_id": result["eval_id"],
                "shape_ids": result["shape_selection"]["selected_shape_ids"],
                "environment": copy.deepcopy(result["environment"]),
            }
            for result in results
        ]
        merged["environment"] = _merge_environments([result["environment"] for result in results])
        from atrex_bench.cli.run_eval import _aggregate_abba_runs, _eval_id

        merged["eval_id"] = _eval_id()
        if merged["eval_mode"] == "abba":
            from atrex_bench.cli.run_eval import _abba_schedule

            schedule = _abba_schedule()
            if any(result["abba"]["schedule"] != schedule for result in results):
                raise ValueError("Incompatible ABBA schedules")
            for result in results:
                if len(result["abba"]["runs"]) != len(schedule):
                    raise ValueError("Incompatible ABBA run count")
                parent = result["shape_selection"]
                for index, step in enumerate(schedule):
                    child = result["abba"]["runs"][index]["result"]["shape_selection"]
                    differences = [key for key in (
                        "all_shape_ids", "selected_shape_ids", "seed_policy",
                        "evaluator_version", "evaluator_sha256",
                    ) if child.get(key) != parent[key]]
                    if differences:
                        raise ValueError(
                            "Incompatible ABBA leg provenance: " + ", ".join(differences)
                        )
                    expected_hash = parent["source_hashes"].get(step["revision"] + ".py")
                    child_hashes = child.get("source_hashes", {})
                    if (not isinstance(child_hashes, Mapping)
                            or not isinstance(expected_hash, str) or not expected_hash
                            or child_hashes.get("candidate.py") != expected_hash
                            or {k: v for k, v in child_hashes.items()
                                if k not in {"candidate.py", "baseline.py"}}
                            != {k: v for k, v in parent["source_hashes"].items()
                                if k not in {"candidate.py", "baseline.py"}}):
                        raise ValueError("Incompatible ABBA leg source hashes")
            runs = []
            for index, expected in enumerate(schedule):
                rows = [result["abba"]["runs"][index] for result in results]
                if any({key: row[key] for key in expected} != expected for row in rows):
                    raise ValueError("Incompatible ABBA run order")
                runs.append({
                    **expected,
                    "result": merge_evaluation_shards(
                        [row["result"] for row in rows],
                        shape_ids=selected_ids, failed_shapes=failures,
                    ),
                })
            errors = [str(result["error"]) for result in results if result.get("error")]
            if errors:
                aggregate, error = None, "\n".join(errors)
            else:
                aggregate, error = _aggregate_abba_runs(runs, completed_ids)
            merged["abba"].update(
                runs=runs,
                baseline=aggregate.get("baseline") if aggregate else None,
                candidate=aggregate.get("candidate") if aggregate else None,
                comparison=aggregate.get("comparison") if aggregate else None,
            )
            merged["passed"]["abba"] = {
                "status": "passed" if error is None else "failed",
                "reason": error,
            }
            merged["error"] = error
        else:
            for stage in ("compile", "correctness", "performance"):
                merged["passed"][stage] = _merge_shape_blocks(
                    results, ("passed", stage), completed_ids
                )
            for stage in ("correctness", "performance"):
                merged[stage]["shapes"] = _merge_shape_blocks(
                    results, (stage, "shapes"), completed_ids
                )
            if "flydsl_compute_ratio" in merged:
                ratio = merged["flydsl_compute_ratio"]
                ratio["shapes"] = _merge_shape_blocks(
                    results, ("flydsl_compute_ratio", "shapes"), completed_ids
                )
                valid = [
                    shape["ratio"]
                    for shape in ratio["shapes"].values()
                    if shape["error"] is None and shape["ratio"] is not None
                ]
                non_flydsl = all(
                    (shape["error"] or "").startswith("non-flydsl")
                    for shape in ratio["shapes"].values()
                )
                average = sum(valid) / len(valid) if valid else 0.0 if non_flydsl else None
                ratio.update(
                    average=average,
                    valid_shape_count=len(valid),
                    total_shape_count=len(completed_ids),
                )
            errors = [str(result["error"]) for result in results if result.get("error")]
            merged["error"] = "\n".join(errors) or None
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Invalid evaluation shard: {error}") from error
    if output is not None:
        Path(output).write_text(
            json.dumps(merged, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return merged
