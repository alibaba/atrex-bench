"""Stage 1: Correctness verification against the eager reference baseline."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

from atrex_bench.eval._runtime import (
    ShapeSpec,
    clone_model_inputs,
    deterministic_input_seed,
    flatten_outputs,
    get_device,
    instantiate_model_pair,
    load_reference_inputs,
    load_shape_call_inputs,
    load_shape_spec,
    seed_all_input_rngs,
    sync_device,
    write_input_artifact,  # kept import for backward compat; unused here
)
from atrex_bench.eval._timeout import CandidateTimeoutError, candidate_timeout

_DEFAULT_CANDIDATE_TIMEOUT_S = 60


@dataclass(frozen=True)
class OutputDiff:
    """Per-output comparison result for one correctness case.

    Fields match the data schema spec, Section 7 outputs entry exactly:
    ``name`` / ``passed`` / ``max_elementwise_abs_diff`` /
    ``max_elementwise_rel_diff`` / ``error``. dtype / shape are intentionally
    not recorded — they are derivable from metadata.json.output_dtypes and
    do not have a real consumer.
    """

    name: str
    passed: bool
    max_elementwise_abs_diff: float | None = None
    max_elementwise_rel_diff: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class CorrectnessCase:
    """One correctness case: a single random input draw.

    ``input_artifact`` is the only per-case input information persisted in
    eval_result.json — the actual random tensor values live in the .pt file.
    Everything else (init_kwargs, input_kwargs) is derivable from
    shapes.json + input.py, so it is not duplicated here.
    """

    input_artifact: dict[str, str] | None
    outputs: list[OutputDiff] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class CorrectnessShapeResult:
    """Per-shape correctness result returned by ``check_correctness``.

    ``status`` ∈ ``{"passed", "failed", "skipped"}`` is what bubbles up to
    ``eval_result.json.passed.correctness.<shape_id>``; ``cases`` populate
    ``eval_result.json.correctness.shapes.<shape_id>.cases``.
    ``check_correctness`` itself only returns ``passed`` or ``failed``;
    ``skipped`` is set by run_eval when the stage was not run because of an
    earlier-stage failure.
    """

    status: str
    reason: str | None = None
    cases: list[CorrectnessCase] = field(default_factory=list)


def _is_leaf_output(value: object) -> bool:
    """Return whether a value is a leaf in the output tree (tensor or numeric scalar)."""
    return isinstance(value, (torch.Tensor, bool, int, float))


def _describe_output_structure(value: object) -> str:
    """Return a short structural label for one output node, used in error messages."""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(value.keys())!r})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def _validate_output_structures_match(
    reference: object,
    candidate: object,
    *,
    path: str = "output",
) -> None:
    """Raise ValueError if reference / candidate output structures don't match."""
    ref_is_dict = isinstance(reference, dict)
    cand_is_dict = isinstance(candidate, dict)
    if ref_is_dict != cand_is_dict:
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}. "
            "When the reference Model.forward returns dict[str, Tensor], the "
            "candidate must also return a dict with matching keys."
        )
    if ref_is_dict:
        ref_keys = set(reference.keys())
        cand_keys = set(candidate.keys())
        if ref_keys != cand_keys:
            missing_in_candidate = sorted(ref_keys - cand_keys)
            extra_in_candidate = sorted(cand_keys - ref_keys)
            raise ValueError(
                f"Output structure mismatch at {path}: dict keys differ. "
                f"reference keys={sorted(ref_keys)}, "
                f"candidate keys={sorted(cand_keys)}, "
                f"missing in candidate={missing_in_candidate}, "
                f"extra in candidate={extra_in_candidate}."
            )
        for key in sorted(ref_keys):
            _validate_output_structures_match(
                reference[key], candidate[key], path=f"{path}.{key}"
            )
        return

    ref_is_seq = isinstance(reference, (tuple, list))
    cand_is_seq = isinstance(candidate, (tuple, list))
    if ref_is_seq != cand_is_seq:
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}."
        )
    if ref_is_seq:
        if len(reference) != len(candidate):
            raise ValueError(
                f"Output structure mismatch at {path}: sequence length differs. "
                f"reference={len(reference)}, candidate={len(candidate)}."
            )
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate)):
            _validate_output_structures_match(
                ref_item, cand_item, path=f"{path}[{index}]"
            )
        return

    if not (_is_leaf_output(reference) and _is_leaf_output(candidate)):
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}."
        )


def _flatten_output_name(prefix_path: str) -> str:
    """Convert flatten_outputs() prefix paths to the schema's tensor name.

    flatten_outputs emits "output" for single tensors, "output.<key>" for dict
    branches, "output[i]" for tuple branches. The schema records just the
    tensor name from metadata.json.output_dtypes:
      "output"        -> "out"           (single-tensor convention)
      "output.<key>"  -> "<key>"         (dict tensor name)
      "output[i]"     -> kept as-is      (legacy tuple path; not reachable
                                          under the current schema, but
                                          tolerated for backward compat)
    """
    if prefix_path == "output":
        return "out"
    if prefix_path.startswith("output."):
        return prefix_path[len("output.") :]
    return prefix_path


def _compare_output_tensors(
    reference_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> OutputDiff:
    """Compare a pair of output tensors and return the per-output diff record."""
    if reference_tensor.shape != candidate_tensor.shape:
        return OutputDiff(
            name=name,
            passed=False,
            error=(
                "Output shape mismatch: "
                f"reference={list(reference_tensor.shape)}, "
                f"candidate={list(candidate_tensor.shape)}"
            ),
        )

    if torch.is_floating_point(reference_tensor) or torch.is_floating_point(candidate_tensor):
        reference_float = reference_tensor.detach().to(torch.float64)
        candidate_float = candidate_tensor.detach().to(torch.float64)
        abs_diff = (reference_float - candidate_float).abs()
        max_elementwise_abs_diff = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
        denominator = reference_float.abs().clamp_min(max(atol, 1e-12))
        max_elementwise_rel_diff = (
            float((abs_diff / denominator).max().item()) if abs_diff.numel() else 0.0
        )
        passed = bool(torch.allclose(reference_float, candidate_float, atol=atol, rtol=rtol))
    else:
        passed = bool(torch.equal(reference_tensor, candidate_tensor))
        max_elementwise_abs_diff = 0.0 if passed else 1.0
        max_elementwise_rel_diff = 0.0 if passed else float("inf")

    return OutputDiff(
        name=name,
        passed=passed,
        max_elementwise_abs_diff=max_elementwise_abs_diff,
        max_elementwise_rel_diff=max_elementwise_rel_diff,
    )


def check_correctness(
    reference_path: Path,
    candidate_path: Path,
    *,
    shape_id: str = "0",
    atol: float = 1e-2,
    rtol: float = 0.05,
    num_correctness_cases: int = 1,
    device: str = "auto",
    artifact_dir: Path | None = None,
    artifact_root: Path | None = None,
    candidate_timeout_s: int | float | None = _DEFAULT_CANDIDATE_TIMEOUT_S,
) -> CorrectnessShapeResult:
    """Compare candidate outputs against the eager reference baseline for one shape.

    ``shape_id`` selects the entry from ``shapes.json`` next to ``reference_path``.
    Synthetic inline references without a sibling shapes.json fall back to the
    legacy ``get_inputs()`` / ``get_init_inputs()`` path; ``shape_id`` is then
    informational only.
    """
    if num_correctness_cases < 1:
        return CorrectnessShapeResult(
            status="failed",
            reason="num_correctness_cases must be at least 1",
        )

    try:
        resolved_device = get_device(device)
        loaded_models = instantiate_model_pair(
            reference_path,
            candidate_path,
            resolved_device,
            module_prefix="atrex_correctness",
            shape_id=shape_id,
            candidate_timeout_s=candidate_timeout_s,
        )
        shape: ShapeSpec | None
        if (reference_path.parent / "shapes.json").is_file():
            shape = load_shape_spec(reference_path, shape_id)
        else:
            shape = None
    except Exception:
        return CorrectnessShapeResult(
            status="failed",
            reason=traceback.format_exc(),
        )

    case_records: list[CorrectnessCase] = []
    failed_cases = 0
    # Surfaced into CorrectnessShapeResult.reason for the deterministic-failure
    # paths, so the shape-level summary (which is what bubbles up to the
    # eval_result.json passed.correctness.<id>.reason field) says *why* the
    # shape failed, not just "X/N cases failed".
    early_abort_reason: str | None = None

    def _abort_remaining_cases(after_case_index: int, short_reason: str) -> None:
        """Append 'skipped' CorrectnessCase entries for every case after this one.

        Used for *non-accuracy* failures only — timeouts, exceptions during the
        candidate call, structural / shape / count mismatches. All of these are
        deterministic w.r.t. the candidate (same code + same inputs class -> same
        failure), so re-running with a fresh random draw of inputs is pure
        wasted wall-clock. Accuracy diffs (atol/rtol failures) deliberately do
        NOT trigger early-abort: those CAN be input-dependent and we want full
        per-case coverage to surface the diff distribution.

        ``short_reason`` is ALSO bubbled up into the shape-level
        ``CorrectnessShapeResult.reason`` so the eval_result.json
        ``passed.correctness.<id>.reason`` field tells you *why* the shape
        failed instead of just "X/N cases failed".
        """
        nonlocal failed_cases, early_abort_reason
        early_abort_reason = short_reason
        skipped = num_correctness_cases - (after_case_index + 1)
        if skipped <= 0:
            return
        failed_cases += skipped
        reason = (
            f"skipped after case {after_case_index} failed deterministically: "
            f"{short_reason}"
        )
        for _ in range(skipped):
            case_records.append(
                CorrectnessCase(input_artifact=None, error=reason)
            )

    for case_index in range(num_correctness_cases):
        # Seed every RNG just before generating inputs so the random tensors
        # are reproducible from the recorded seed alone (no .pt files needed).
        seed = deterministic_input_seed("correctness", shape_id, case_index)
        seed_all_input_rngs(seed)
        if shape is not None:
            inputs = load_shape_call_inputs(loaded_models.input_module, shape, resolved_device)
        else:
            inputs = load_reference_inputs(loaded_models.input_module, resolved_device)
        artifact = {"seed": seed, "format": "manual_seed"}

        try:
            reference_call_inputs = clone_model_inputs(inputs)
            candidate_call_inputs = clone_model_inputs(inputs)
            with torch.inference_mode():
                # Reference is the golden implementation; we trust it and
                # never time it out. The candidate is the AI-generated code
                # and may hang in JIT compile or run a pathological kernel,
                # so we wrap its call (plus the trailing GPU sync so the
                # alarm fires while we are still inside the scope).
                reference_output = loaded_models.reference_model(
                    *reference_call_inputs.args,
                    **reference_call_inputs.kwargs,
                )
                try:
                    with candidate_timeout(candidate_timeout_s):
                        candidate_output = loaded_models.candidate_model(
                            *candidate_call_inputs.args,
                            **candidate_call_inputs.kwargs,
                        )
                        sync_device(resolved_device)
                except CandidateTimeoutError as timeout_error:
                    failed_cases += 1
                    case_records.append(
                        CorrectnessCase(
                            input_artifact=artifact,
                            error=str(timeout_error),
                        )
                    )
                    _abort_remaining_cases(
                        case_index, f"{candidate_timeout_s}s timeout"
                    )
                    break

            try:
                _validate_output_structures_match(reference_output, candidate_output)
            except ValueError as structure_error:
                failed_cases += 1
                case_records.append(
                    CorrectnessCase(
                        input_artifact=artifact,
                        error=str(structure_error),
                    )
                )
                _abort_remaining_cases(case_index, "output structure mismatch")
                break

            reference_outputs = flatten_outputs(reference_output)
            candidate_outputs = flatten_outputs(candidate_output)

            if len(reference_outputs) != len(candidate_outputs):
                failed_cases += 1
                case_records.append(
                    CorrectnessCase(
                        input_artifact=artifact,
                        error=(
                            "Output count mismatch: "
                            f"reference={len(reference_outputs)}, "
                            f"candidate={len(candidate_outputs)}"
                        ),
                    )
                )
                _abort_remaining_cases(case_index, "output count mismatch")
                break

            output_diffs: list[OutputDiff] = []
            case_passed = True
            has_structural_failure = False
            for (raw_name, reference_tensor), (_, candidate_tensor) in zip(
                reference_outputs,
                candidate_outputs,
            ):
                output_diff = _compare_output_tensors(
                    reference_tensor,
                    candidate_tensor,
                    name=_flatten_output_name(raw_name),
                    atol=atol,
                    rtol=rtol,
                )
                output_diffs.append(output_diff)
                if not output_diff.passed:
                    case_passed = False
                # error != None means this is a structural failure (shape
                # mismatch, compare-time exception) rather than an atol/rtol
                # diff. Structural failures are deterministic, so we treat
                # them like the other non-accuracy failures and early-abort.
                if output_diff.error is not None:
                    has_structural_failure = True

            if not case_passed:
                failed_cases += 1

            case_records.append(
                CorrectnessCase(
                    input_artifact=artifact,
                    outputs=output_diffs,
                )
            )

            if has_structural_failure:
                _abort_remaining_cases(
                    case_index, "per-tensor shape/structural mismatch"
                )
                break
        except Exception:
            # Any other exception in the case body (e.g. candidate raised an
            # OOM, kernel launch error, AttributeError on .Model, etc.) is
            # also deterministic w.r.t. the candidate. Record this case as
            # failed AND abort the remaining cases.
            failed_cases += 1
            tb = traceback.format_exc()
            case_records.append(
                CorrectnessCase(
                    input_artifact=artifact,
                    error=tb,
                )
            )
            _abort_remaining_cases(
                case_index, "candidate raised exception"
            )
            break

    if failed_cases == 0:
        status = "passed"
        reason: str | None = None
    else:
        status = "failed"
        base = f"{failed_cases}/{num_correctness_cases} correctness cases failed"
        if early_abort_reason is not None:
            reason = f"{base}: {early_abort_reason}"
        else:
            reason = base

    return CorrectnessShapeResult(
        status=status,
        reason=reason,
        cases=case_records,
    )
