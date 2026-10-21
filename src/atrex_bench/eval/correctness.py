"""Stage 1: Correctness verification against the eager reference baseline."""

from __future__ import annotations

import inspect
import json
import math
import os
import traceback
from dataclasses import dataclass, field, replace
from fractions import Fraction
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
    write_input_artifact,  # noqa: F401 - backward-compatible module export
)
from atrex_bench.eval._timeout import CandidateTimeoutError, candidate_timeout
from atrex_bench.eval.reward_hack import (
    RewardHackDetected,
    check_plain_tensor_outputs,
)

_DEFAULT_CANDIDATE_TIMEOUT_S = 60
_COMPARISON_CHUNK_ELEMENTS = 1 << 20
CORRECTNESS_MAX_REL_L2_ENV = "ATREX_CORRECTNESS_MAX_REL_L2"


@dataclass(frozen=True)
class OutputDiff:
    """Per-output comparison result for one correctness case.

    Dtype and shape are intentionally omitted because metadata owns them.
    Numerical receipts include the effective policy and mismatch counts so a
    caller can audit why a tensor passed without retaining full output values.
    """

    name: str
    passed: bool
    max_elementwise_abs_diff: float | None = None
    max_elementwise_rel_diff: float | None = None
    relative_l2: float | None = None
    error: str | None = None
    mismatched_elements: int | None = None
    total_elements: int | None = None
    allowed_mismatched_elements: int | None = None
    tolerance: dict[str, object] | None = None


@dataclass(frozen=True)
class OutputTolerance:
    """Elementwise tolerance for one logical output tensor path."""

    atol: float
    rtol: float
    max_mismatch_fraction: float = 0.0
    max_rel_l2: float | None = None


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
    mutated_inputs: list[OutputDiff] = field(default_factory=list)
    unexpected_mutations: list[OutputDiff] = field(default_factory=list)
    defined_outputs: dict[str, OutputDiff] = field(default_factory=dict)
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
    return value is None or isinstance(value, (torch.Tensor, bool, int, float))


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
    strict_types: bool = False,
) -> None:
    """Raise ValueError if reference / candidate output structures don't match."""
    if (strict_types and type(reference) is not type(candidate)) or (
        (reference is None) != (candidate is None)
    ):
        raise ValueError(
            f"Output structure mismatch at {path}: reference={type(reference).__name__}, "
            f"candidate={type(candidate).__name__}"
        )
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
                reference[key], candidate[key], path=f"{path}.{key}", strict_types=strict_types
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
                ref_item, cand_item, path=f"{path}[{index}]", strict_types=strict_types
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


def configured_max_rel_l2(explicit: float | None = None) -> float | None:
    if explicit is not None:
        value = float(explicit)
    else:
        raw = os.environ.get(CORRECTNESS_MAX_REL_L2_ENV)
        if raw is None or not raw.strip():
            return None
        value = float(raw)
    if value < 0:
        raise ValueError("correctness max_rel_l2 must be non-negative")
    return value


def _validated_global_tolerance(name: str, raw_value: object) -> float:
    """Return one process-wide allclose tolerance after fail-closed validation."""

    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"correctness {name} must be finite and non-negative; got {raw_value!r}"
        ) from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"correctness {name} must be finite and non-negative; got {raw_value!r}"
        )
    return value


def _comparison_chunks(reference, candidate, *extra):
    """Yield bounded matching views, including noncontiguous logical tensors."""
    if reference.numel() <= _COMPARISON_CHUNK_ELEMENTS:
        yield reference, candidate, *extra
        return
    axis = max(range(reference.ndim), key=lambda index: reference.shape[index])
    size = reference.shape[axis]
    half = size // 2
    for start, length in ((0, half), (half, size - half)):
        yield from _comparison_chunks(
            reference.narrow(axis, start, length),
            candidate.narrow(axis, start, length),
            *(value.narrow(axis, start, length) for value in extra),
        )


def _compare_output_tensors(
    reference_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    *,
    name: str,
    atol: float,
    rtol: float,
    max_rel_l2: float | None = None,
    strict_dtype: bool = False,
    metadata_correctness: bool = False,
    max_mismatch_fraction: float = 0.0,
    max_rel_l2_bound: float | None = None,
    defined_mask: torch.Tensor | None = None,
    mutation_before: torch.Tensor | None = None,
) -> OutputDiff:
    """Compare a pair of output tensors and return the per-output diff record."""
    if strict_dtype and reference_tensor.dtype != candidate_tensor.dtype:
        return OutputDiff(
            name=name,
            passed=False,
            error=f"Output dtype mismatch: {reference_tensor.dtype} != {candidate_tensor.dtype}",
        )
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

    if defined_mask is not None:
        if (
            not isinstance(defined_mask, torch.Tensor)
            or defined_mask.dtype != torch.bool
            or defined_mask.shape != reference_tensor.shape
            or not bool(defined_mask.any().item())
        ):
            return OutputDiff(
                name=name,
                passed=False,
                error="Output mask must be a nonempty bool mask of the full output shape",
            )
        # Validate the full physical ABI before selecting defined elements.
        if reference_tensor.dtype != candidate_tensor.dtype:
            return OutputDiff(name=name, passed=False, error="Masked output dtype mismatch")
        reference_tensor = reference_tensor[defined_mask.to(reference_tensor.device)]
        candidate_tensor = candidate_tensor[defined_mask.to(candidate_tensor.device)]

    reference_is_discrete = not (
        torch.is_floating_point(reference_tensor) or torch.is_complex(reference_tensor)
    )
    candidate_is_continuous = torch.is_floating_point(candidate_tensor) or torch.is_complex(
        candidate_tensor
    )
    if metadata_correctness and reference_is_discrete and candidate_is_continuous:
        return OutputDiff(
            name=name,
            passed=False,
            error=(
                "Metadata correctness policies keep integer and boolean reference outputs "
                "on the exact-comparison path; "
                f"candidate dtype {candidate_tensor.dtype} is floating or complex"
            ),
        )

    error: str | None = None
    compared_elements = reference_tensor.numel()
    if torch.is_floating_point(reference_tensor) or torch.is_floating_point(candidate_tensor):
        ref_is_finite = cand_is_finite = True
        reference_norm = candidate_norm = diff_l2 = 0.0
        max_elementwise_abs_diff = max_elementwise_rel_diff = 0.0
        mismatched = 0
        baseline = (
            (mutation_before,)
            if (
                isinstance(mutation_before, torch.Tensor)
                and mutation_before.shape == reference_tensor.shape
                and mutation_before.dtype == reference_tensor.dtype == candidate_tensor.dtype
            )
            else ()
        )
        for chunks in _comparison_chunks(reference_tensor, candidate_tensor, *baseline):
            ref_chunk, cand_chunk = chunks[:2]
            reference_float = ref_chunk.detach().to(torch.float64)
            candidate_float = cand_chunk.detach().to(torch.float64)
            if baseline:
                before_chunk = chunks[2]
                nonfinite = ~torch.isfinite(reference_float) | ~torch.isfinite(candidate_float)
                if bool(nonfinite.any().item()):
                    # A state pool may contain untouched NaN/Inf padding. It is
                    # accepted only when BOTH executions preserve its exact bytes.
                    def bits(value):
                        return (
                            value.detach()
                            .contiguous()
                            .reshape(-1)
                            .view(torch.uint8)
                            .reshape(value.numel(), value.element_size())
                        )

                    original = bits(before_chunk)
                    unchanged = (
                        (bits(ref_chunk) == original).all(1) & (bits(cand_chunk) == original).all(1)
                    ).reshape(ref_chunk.shape)
                    ignored = nonfinite & unchanged
                    compared_elements -= int(ignored.sum().item())
                    reference_float = reference_float.masked_fill(ignored, 0)
                    candidate_float = candidate_float.masked_fill(ignored, 0)
            ref_is_finite &= bool(torch.isfinite(reference_float).all().item())
            cand_is_finite &= bool(torch.isfinite(candidate_float).all().item())
            if not ref_is_finite or not cand_is_finite:
                continue
            reference_norm = math.hypot(
                reference_norm, torch.linalg.vector_norm(reference_float).item()
            )
            candidate_norm = math.hypot(
                candidate_norm, torch.linalg.vector_norm(candidate_float).item()
            )
            abs_diff = (reference_float - candidate_float).abs()
            diff_l2 = math.hypot(diff_l2, torch.linalg.vector_norm(abs_diff).item())
            if abs_diff.numel():
                max_elementwise_abs_diff = max(max_elementwise_abs_diff, abs_diff.max().item())
                denominator = reference_float.abs().clamp_min(max(atol, 1e-12))
                max_elementwise_rel_diff = max(
                    max_elementwise_rel_diff,
                    (abs_diff / denominator).max().item(),
                )
            # Preserve the historical torch.allclose(reference, candidate)
            # argument direction: the relative term scales with candidate.
            mismatched += int((abs_diff > atol + rtol * candidate_float.abs()).sum().item())
        if not ref_is_finite or not cand_is_finite:
            return OutputDiff(
                name=name,
                passed=False,
                max_elementwise_abs_diff=0.0,
                max_elementwise_rel_diff=0.0,
                relative_l2=0.0,
                error=(
                    "Non-finite output detected: "
                    f"reference_finite={ref_is_finite}, candidate_finite={cand_is_finite}"
                ),
            )

        candidate_is_zero = reference_norm > 0.0 and candidate_norm == 0.0
        relative_l2 = diff_l2 / max(reference_norm, 1e-12)
        fraction = Fraction(str(max_mismatch_fraction))
        allowed = compared_elements * fraction.numerator // fraction.denominator
        passed = relative_l2 <= max_rel_l2 if max_rel_l2 is not None else mismatched <= allowed
        if max_rel_l2_bound is not None:
            # Metadata max_rel_l2 is conjunctive with the elementwise rule,
            # unlike the legacy CLI/environment L2-only mode above.
            l2_passed = (
                max_elementwise_abs_diff == 0.0
                if reference_norm == 0.0
                else relative_l2 <= max_rel_l2_bound
            )
            passed = passed and l2_passed
        if candidate_is_zero and not passed:
            error = "Candidate output is all zero while reference output is non-zero"
    else:
        passed = bool(torch.equal(reference_tensor, candidate_tensor))
        max_elementwise_abs_diff = 0.0 if passed else 1.0
        max_elementwise_rel_diff = 0.0 if passed else 1.0
        relative_l2 = None
        mismatched = int((reference_tensor != candidate_tensor).sum().item())
        allowed = 0

    return OutputDiff(
        name=name,
        passed=passed,
        max_elementwise_abs_diff=max_elementwise_abs_diff,
        max_elementwise_rel_diff=max_elementwise_rel_diff,
        relative_l2=relative_l2,
        error=error,
        mismatched_elements=mismatched,
        total_elements=compared_elements,
        allowed_mismatched_elements=allowed,
        tolerance={
            "atol": atol,
            "rtol": rtol,
            "max_mismatch_fraction": max_mismatch_fraction,
            "max_rel_l2": max_rel_l2_bound,
            "relative_to": "candidate",
            "mode": (
                "exact"
                if relative_l2 is None
                else "legacy_relative_l2"
                if max_rel_l2 is not None
                else "elementwise_and_l2"
                if max_rel_l2_bound is not None
                else "elementwise"
            ),
        },
    )


def _compare_value_trees(
    reference,
    candidate,
    *,
    name,
    atol,
    rtol,
    max_rel_l2=None,
    strict_types=False,
    output_tolerances=None,
    tolerance_name=None,
    matched_tolerance_paths=None,
    output_masks=None,
    matched_mask_paths=None,
    mutation_before=None,
):
    """Compare return values, with exact types for explicit mutation contracts."""
    try:
        _validate_output_structures_match(
            reference, candidate, path=name, strict_types=strict_types
        )
    except ValueError as error:
        return [OutputDiff(name=name, passed=False, error=str(error))]
    logical_name = name if tolerance_name is None else tolerance_name
    if isinstance(reference, dict):
        return [
            diff
            for key in reference
            for diff in _compare_value_trees(
                reference[key],
                candidate[key],
                name=f"{name}.{key}",
                atol=atol,
                rtol=rtol,
                max_rel_l2=max_rel_l2,
                strict_types=strict_types,
                output_tolerances=output_tolerances,
                tolerance_name=f"{logical_name}.{key}",
                mutation_before=mutation_before.get(key)
                if isinstance(mutation_before, dict)
                else None,
                matched_tolerance_paths=matched_tolerance_paths,
                output_masks=output_masks,
                matched_mask_paths=matched_mask_paths,
            )
        ]
    if isinstance(reference, (list, tuple)):
        return [
            diff
            for index, (left, right) in enumerate(zip(reference, candidate))
            for diff in _compare_value_trees(
                left,
                right,
                name=f"{name}[{index}]",
                atol=atol,
                rtol=rtol,
                max_rel_l2=max_rel_l2,
                strict_types=strict_types,
                output_tolerances=output_tolerances,
                tolerance_name=f"{logical_name}[{index}]",
                mutation_before=(
                    mutation_before[index]
                    if isinstance(mutation_before, (list, tuple)) and index < len(mutation_before)
                    else None
                ),
                matched_tolerance_paths=matched_tolerance_paths,
                output_masks=output_masks,
                matched_mask_paths=matched_mask_paths,
            )
        ]
    if reference is None:
        return [OutputDiff(name=name, passed=True)]
    # Retain legacy scalar conversion and tolerance semantics as well as
    # list/tuple interoperability for benchmarks without an explicit contract.
    reference_tensor = flatten_outputs(reference)[0][1]
    candidate_tensor = flatten_outputs(candidate)[0][1]
    masks = output_masks or {}
    if logical_name in masks and matched_mask_paths is not None:
        matched_mask_paths.add(logical_name)
    policies = output_tolerances or {}
    tolerance = policies.get(logical_name)
    is_output_path = (
        logical_name == "output"
        or logical_name.startswith("output[")
        or logical_name.startswith("output.")
    )
    if tolerance is None and is_output_path:
        tolerance = policies.get("default")
    if logical_name in policies and matched_tolerance_paths is not None:
        matched_tolerance_paths.add(logical_name)
    return [
        _compare_output_tensors(
            reference_tensor,
            candidate_tensor,
            name=name,
            atol=tolerance.atol if tolerance is not None else atol,
            rtol=tolerance.rtol if tolerance is not None else rtol,
            max_rel_l2=max_rel_l2,
            strict_dtype=strict_types,
            defined_mask=masks.get(logical_name),
            mutation_before=mutation_before,
            metadata_correctness=bool(policies),
            max_mismatch_fraction=(
                tolerance.max_mismatch_fraction if tolerance is not None else 0.0
            ),
            max_rel_l2_bound=(tolerance.max_rel_l2 if tolerance is not None else None),
        )
    ]


def _load_benchmark_contract(reference_path: Path) -> tuple[Path, dict[str, object]]:
    """Load the evaluator-only benchmark contract next to a reference."""

    metadata_path = reference_path.parent / "metadata.json"
    if not metadata_path.is_file():
        return metadata_path, {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = payload.get("benchmark_contract") or {}
    if not isinstance(contract, dict):
        raise TypeError(f"{metadata_path}.benchmark_contract must be an object")
    return metadata_path, contract


def _load_output_tolerance_contract(reference_path: Path) -> dict[str, OutputTolerance]:
    """Load optional per-output allclose tolerances from metadata."""

    metadata_path, contract = _load_benchmark_contract(reference_path)
    raw_tolerances = contract.get("correctness_tolerances")
    if raw_tolerances is None:
        return {}
    if not isinstance(raw_tolerances, dict) or not raw_tolerances:
        raise TypeError(
            f"{metadata_path}.benchmark_contract.correctness_tolerances "
            "must be a non-empty object"
        )

    tolerances: dict[str, OutputTolerance] = {}
    for raw_path, raw_policy in raw_tolerances.items():
        is_output_path = isinstance(raw_path, str) and (
            raw_path in ("default", "output")
            or raw_path.startswith("output[")
            or raw_path.startswith("output.")
        )
        is_mutation_path = isinstance(raw_path, str) and raw_path.startswith(
            "mutated_inputs."
        )
        if not (is_output_path or is_mutation_path):
            raise ValueError(
                "correctness_tolerances keys must be logical output or "
                f"declared-mutation paths: {raw_path!r}"
            )
        if not isinstance(raw_policy, dict):
            raise TypeError(f"correctness_tolerances[{raw_path!r}] must be an object")
        unknown = sorted(
            set(raw_policy) - {"atol", "rtol", "max_mismatch_fraction", "max_rel_l2"}
        )
        if unknown:
            raise ValueError(
                f"correctness_tolerances[{raw_path!r}] has unknown fields: {unknown}"
            )
        if "atol" not in raw_policy or "rtol" not in raw_policy:
            raise ValueError(f"correctness_tolerances[{raw_path!r}] requires atol and rtol")
        tensor_atol = float(raw_policy["atol"])
        tensor_rtol = float(raw_policy["rtol"])
        if (
            not math.isfinite(tensor_atol)
            or not math.isfinite(tensor_rtol)
            or tensor_atol < 0
            or tensor_rtol < 0
        ):
            raise ValueError(
                f"correctness_tolerances[{raw_path!r}] must contain finite, "
                "non-negative atol and rtol"
            )
        mismatch_fraction = float(raw_policy.get("max_mismatch_fraction", 0.0))
        raw_l2 = raw_policy.get("max_rel_l2")
        tensor_max_rel_l2 = float(raw_l2) if raw_l2 is not None else None
        if not math.isfinite(mismatch_fraction) or not 0.0 <= mismatch_fraction <= 1.0:
            raise ValueError(
                f"correctness_tolerances[{raw_path!r}].max_mismatch_fraction "
                "must be finite and between 0 and 1"
            )
        if tensor_max_rel_l2 is not None and (
            not math.isfinite(tensor_max_rel_l2) or tensor_max_rel_l2 < 0.0
        ):
            raise ValueError(
                f"correctness_tolerances[{raw_path!r}].max_rel_l2 "
                "must be finite and non-negative"
            )
        tolerances[raw_path] = OutputTolerance(
            atol=tensor_atol,
            rtol=tensor_rtol,
            max_mismatch_fraction=mismatch_fraction,
            max_rel_l2=tensor_max_rel_l2,
        )
    return tolerances


def metadata_owns_correctness(reference_path: Path) -> bool:
    """Return whether metadata declares authoritative per-output tolerances."""

    return bool(_load_output_tolerance_contract(reference_path))


def load_minimum_correctness_cases(reference_path: Path) -> int:
    """Return the optional metadata-owned minimum random coverage per shape."""

    metadata_path, contract = _load_benchmark_contract(reference_path)
    raw_value = contract.get("correctness_min_cases", 1)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 1:
        raise ValueError(
            f"{metadata_path}.benchmark_contract.correctness_min_cases "
            "must be a positive integer"
        )
    return raw_value


def _check_unchanged(before, after, *, name):
    """Check full backing bytes (including pool tails) without float tolerances."""
    if type(before) is not type(after):
        return [OutputDiff(name=name, passed=False, error="Input type changed")]
    if isinstance(before, torch.Tensor):
        same = (
            before.dtype == after.dtype
            and before.shape == after.shape
            and before.stride() == after.stride()
            and before.storage_offset() == after.storage_offset()
        )

        def raw(tensor):
            storage = tensor.untyped_storage()
            return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )

        same = same and torch.equal(raw(before), raw(after))
        return [
            OutputDiff(
                name=name, passed=bool(same), error=None if same else "Undeclared input mutation"
            )
        ]
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return [OutputDiff(name=name, passed=False, error="Input keys changed")]
        return [
            diff
            for key in before
            for diff in _check_unchanged(before[key], after[key], name=f"{name}.{key}")
        ]
    if isinstance(before, (tuple, list)):
        if len(before) != len(after):
            return [OutputDiff(name=name, passed=False, error="Input length changed")]
        return [
            diff
            for index, (left, right) in enumerate(zip(before, after))
            for diff in _check_unchanged(left, right, name=f"{name}[{index}]")
        ]
    same = before == after
    return [OutputDiff(name=name, passed=bool(same), error=None if same else "Input value changed")]


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
    max_rel_l2: float | None = None,
    untrusted_mode: bool = False,
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
        atol = _validated_global_tolerance("atol", atol)
        rtol = _validated_global_tolerance("rtol", rtol)
        # The CLI resolves this floor before sizing worker budgets and writing
        # runner_config. Keep the same guard here for direct library callers.
        effective_num_correctness_cases = max(
            num_correctness_cases,
            load_minimum_correctness_cases(reference_path),
        )
        effective_max_rel_l2 = configured_max_rel_l2(max_rel_l2)
    except (OSError, TypeError, ValueError) as error:
        return CorrectnessShapeResult(status="failed", reason=str(error))

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
        _, contract = _load_benchmark_contract(reference_path)
        mutations = contract.get("mutates_inputs", [])
        scratch_inputs = contract.get("scratch_inputs", [])
        strict_types = "mutates_inputs" in contract
        if not isinstance(mutations, list) or not all(isinstance(x, str) for x in mutations):
            raise ValueError("benchmark_contract.mutates_inputs must be a list of input names")
        if not isinstance(scratch_inputs, list) or not all(
            isinstance(x, str) for x in scratch_inputs
        ):
            raise ValueError("benchmark_contract.scratch_inputs must be a list of input names")
        if set(mutations) & set(scratch_inputs):
            raise ValueError(
                "benchmark_contract mutates_inputs and scratch_inputs must not overlap"
            )
        output_tolerances = _load_output_tolerance_contract(reference_path)
        if output_tolerances:
            # Per-output metadata is authoritative and must not be replaced by
            # a legacy process-wide relative-L2 threshold.
            effective_max_rel_l2 = None
        mask_method = contract.get("output_mask_method")
        if mask_method is not None and (
            not isinstance(mask_method, str)
            or not mask_method.isidentifier()
            or mask_method.startswith("_")
            or not callable(getattr(loaded_models.reference_model, mask_method, None))
        ):
            raise ValueError("output_mask_method must name a public reference Model method")
        signature = inspect.signature(loaded_models.reference_model.forward)
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
        skipped = effective_num_correctness_cases - (after_case_index + 1)
        if skipped <= 0:
            return
        failed_cases += skipped
        reason = f"skipped after case {after_case_index} failed deterministically: {short_reason}"
        for _ in range(skipped):
            case_records.append(CorrectnessCase(input_artifact=None, error=reason))

    for case_index in range(effective_num_correctness_cases):
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

            def named(call):
                return signature.bind(*call.args, **call.kwargs).arguments

            original_named = named(inputs)
            if set(mutations) - original_named.keys():
                raise ValueError(
                    f"Unknown mutated input names: {set(mutations) - original_named.keys()}"
                )
            if set(scratch_inputs) - original_named.keys():
                raise ValueError(
                    f"Unknown scratch input names: {set(scratch_inputs) - original_named.keys()}"
                )
            output_masks = {}
            if mask_method is not None:
                # Only trusted reference code receives pristine input clones.
                # Compute masks before either forward; candidate output cannot select its domain.
                mask_inputs = clone_model_inputs(inputs)
                with torch.inference_mode():
                    output_masks = getattr(loaded_models.reference_model, mask_method)(
                        *mask_inputs.args, **mask_inputs.kwargs
                    )
                if not isinstance(output_masks, dict) or not output_masks:
                    raise ValueError("Output masks must be a nonempty path-to-bool-tensor mapping")
                for path, mask in output_masks.items():
                    if (
                        not isinstance(path, str)
                        or not path.startswith("output")
                        or not isinstance(mask, torch.Tensor)
                        or mask.dtype != torch.bool
                    ):
                        raise ValueError("Invalid output mask path or dtype")
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
                    _abort_remaining_cases(case_index, f"{candidate_timeout_s}s timeout")
                    break
                if untrusted_mode:
                    try:
                        check_plain_tensor_outputs(candidate_output)
                    except RewardHackDetected as reward_hack:
                        failed_cases += 1
                        case_records.append(
                            CorrectnessCase(
                                input_artifact=artifact,
                                error=str(reward_hack),
                            )
                        )
                        _abort_remaining_cases(case_index, "untrusted output guard failed")
                        break

            try:
                _validate_output_structures_match(
                    reference_output, candidate_output, strict_types=strict_types
                )
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

            matched_tolerance_paths: set[str] = set()
            matched_mask_paths: set[str] = set()
            output_diffs = _compare_value_trees(
                reference_output,
                candidate_output,
                name="output",
                atol=atol,
                rtol=rtol,
                max_rel_l2=effective_max_rel_l2,
                strict_types=strict_types,
                output_tolerances=output_tolerances,
                matched_tolerance_paths=matched_tolerance_paths,
                output_masks=output_masks,
                matched_mask_paths=matched_mask_paths,
            )
            for path in output_masks.keys() - matched_mask_paths:
                output_diffs.append(
                    OutputDiff(
                        name=path, passed=False, error="Output mask path did not match a tensor"
                    )
                )
            output_diffs = [
                replace(diff, name=_flatten_output_name(diff.name)) for diff in output_diffs
            ]
            reference_named = named(reference_call_inputs)
            candidate_named = named(candidate_call_inputs)
            mutation_diffs = []
            unexpected_diffs = []
            for key, before in original_named.items():
                if key in mutations:
                    mutation_diffs.extend(
                        _compare_value_trees(
                            reference_named[key],
                            candidate_named[key],
                            name=f"input.{key}",
                            mutation_before=before,
                            atol=atol,
                            rtol=rtol,
                            max_rel_l2=effective_max_rel_l2,
                            strict_types=True,
                            output_tolerances=output_tolerances,
                            tolerance_name=f"mutated_inputs.{key}",
                            matched_tolerance_paths=matched_tolerance_paths,
                        )
                    )
                elif key in scratch_inputs:
                    continue
                else:
                    for role, state in (
                        ("reference", reference_named),
                        ("candidate", candidate_named),
                    ):
                        unexpected_diffs.extend(
                            _check_unchanged(before, state[key], name=f"{role}.input.{key}")
                        )
            unmatched_tolerance_paths = sorted(
                set(output_tolerances) - {"default"} - matched_tolerance_paths
            )
            if unmatched_tolerance_paths:
                raise ValueError(
                    "correctness_tolerances paths did not match compared values: "
                    + ", ".join(unmatched_tolerance_paths)
                )
            all_diffs = output_diffs + mutation_diffs + unexpected_diffs
            case_passed = all(diff.passed for diff in all_diffs)
            has_structural_failure = any(diff.error is not None for diff in all_diffs)

            if not case_passed:
                failed_cases += 1

            case_records.append(
                CorrectnessCase(
                    input_artifact=artifact,
                    outputs=output_diffs,
                    defined_outputs={
                        path: diff
                        for path in output_masks
                        for diff in output_diffs
                        if diff.name == _flatten_output_name(path)
                    },
                    mutated_inputs=mutation_diffs,
                    unexpected_mutations=unexpected_diffs,
                )
            )

            if has_structural_failure:
                _abort_remaining_cases(case_index, "per-tensor shape/structural mismatch")
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
            _abort_remaining_cases(case_index, "candidate raised exception")
            break

    if failed_cases == 0:
        status = "passed"
        reason: str | None = None
    else:
        status = "failed"
        base = f"{failed_cases}/{effective_num_correctness_cases} correctness cases failed"
        if early_abort_reason is not None:
            reason = f"{base}: {early_abort_reason}"
        else:
            reason = base

    return CorrectnessShapeResult(
        status=status,
        reason=reason,
        cases=case_records,
    )
