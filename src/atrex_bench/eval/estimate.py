"""Unified estimation entrypoints for roofline-oriented metrics."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from atrex_bench.eval._runtime import (
    ModelInputs,
    ShapeSpec,
    clone_model_inputs,
    flatten_outputs,
    get_device,
    import_module_from_path,
    infer_operator_id,
    infer_target_dsl,
    load_model_init_inputs,
    load_reference_inputs,
    load_shape_call_inputs,
    load_shape_init_inputs,
    load_shape_spec,
    resolve_input_module,
    summarize_model_inputs,
    summarize_value,
    validate_reference_module,
)
from atrex_bench.eval.flops import estimate_theoretical_flops

SUPPORTED_ESTIMATE_MODES = {
    "W_theoretical",
    "Q_semantic_lower_bound",
    "Q_profiled_impl",
}


@dataclass(frozen=True)
class EstimateResult:
    """Unified result envelope for one estimate mode."""

    passed: bool
    mode: str
    operator_name: str | None = None
    module_path: str | None = None
    semantic_source_path: str | None = None
    device: str | None = None
    value: int | None = None
    units: str | None = None
    precision: dict[str, object] = field(default_factory=dict)
    components: dict[str, object] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class LoadedEstimateModule:
    """Loaded module plus one concrete forward invocation payload."""

    module_path: Path
    operator_name: str
    device: torch.device
    module: Any
    model: torch.nn.Module
    init_inputs: ModelInputs
    call_inputs: ModelInputs


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    """Yield all tensors from a nested structure."""
    if isinstance(value, ModelInputs):
        yield from _iter_tensors(value.args)
        yield from _iter_tensors(value.kwargs)
        return
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, tuple):
        for item in value:
            yield from _iter_tensors(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_tensors(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _tensor_identity_key(tensor: torch.Tensor) -> tuple[object, ...]:
    """Build a stable key for exact tensor aliases within one run."""
    return (
        str(tensor.device),
        tensor.dtype,
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(dim) for dim in tensor.stride()),
    )


def _dedup_tensor_bytes(tensors: Iterable[torch.Tensor]) -> tuple[int, int]:
    """Count logical bytes, deduplicating exact aliasing tensors."""
    total_bytes = 0
    seen: set[tuple[object, ...]] = set()
    unique_count = 0
    for tensor in tensors:
        key = _tensor_identity_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        unique_count += 1
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes, unique_count


def _model_state_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    """Return all registered parameter/buffer tensors for semantic accounting."""
    tensors: list[torch.Tensor] = []
    for parameter in model.parameters(recurse=True):
        tensors.append(parameter.detach())
    for buffer in model.buffers(recurse=True):
        tensors.append(buffer.detach())
    return tensors


def _load_estimate_module(
    module_path: Path,
    device: str = "auto",
    *,
    shape_id: str = "0",
) -> LoadedEstimateModule:
    """Load a reference-style module and prepare one concrete forward invocation."""
    resolved_device = get_device(device)
    module = import_module_from_path(module_path, "atrex_estimate_module")
    validate_reference_module(module)
    input_module = resolve_input_module(
        module_path, module, module_prefix="atrex_estimate_input"
    )

    shape: ShapeSpec | None
    if (module_path.parent / "shapes.json").is_file():
        shape = load_shape_spec(module_path, shape_id)
        init_inputs = load_shape_init_inputs(shape, resolved_device)
    else:
        shape = None
        init_inputs = load_model_init_inputs(input_module, resolved_device)
    model_init_inputs = clone_model_inputs(init_inputs)
    model = module.Model(
        *model_init_inputs.args,
        **model_init_inputs.kwargs,
    ).to(resolved_device).eval()
    if shape is not None:
        call_inputs = load_shape_call_inputs(input_module, shape, resolved_device)
    else:
        call_inputs = load_reference_inputs(input_module, resolved_device)

    return LoadedEstimateModule(
        module_path=module_path,
        operator_name=infer_operator_id(module_path),
        device=resolved_device,
        module=module,
        model=model,
        init_inputs=init_inputs,
        call_inputs=call_inputs,
    )


def _run_forward(loaded: LoadedEstimateModule) -> Any:
    """Execute one inference-mode forward pass for the loaded module."""
    call_inputs = clone_model_inputs(loaded.call_inputs)
    with torch.inference_mode():
        return loaded.model(*call_inputs.args, **call_inputs.kwargs)


def _estimate_w_theoretical(
    module_path: Path,
    *,
    semantic_source_path: Path,
    device: str,
    strict: bool,
    verbose: bool,
    shape_id: str = "0",
) -> EstimateResult:
    """Estimate the theoretical FLOP count from the semantic source module."""
    flop_result = estimate_theoretical_flops(
        semantic_source_path,
        device=device,
        strict=strict,
        shape_id=shape_id,
    )
    precision: dict[str, object] = {
        "status": "exact" if flop_result.flops_complete else "partial",
        "estimation_model": "execution_traced_aten_flop_formulas",
        "flops_complete": flop_result.flops_complete,
    }
    if flop_result.uncounted_ops:
        precision["unsupported_flop_ops"] = flop_result.uncounted_ops

    details: dict[str, object] = {}
    if verbose:
        details = {
            "inputs": flop_result.inputs,
            "environment": flop_result.environment,
            "counted_flop_ops": flop_result.counted_ops,
            "zero_flop_ops": flop_result.zero_flop_ops,
            "uncounted_flop_op_invocations": flop_result.uncounted_op_invocations,
        }

    return EstimateResult(
        passed=flop_result.passed,
        mode="W_theoretical",
        operator_name=flop_result.operator_name,
        module_path=str(module_path),
        semantic_source_path=str(semantic_source_path),
        device=flop_result.device,
        value=flop_result.total_flops,
        units="FLOPs",
        precision=precision,
        components={"flops_by_dtype": dict(flop_result.flops_by_dtype)},
        details=details,
        error=flop_result.error,
    )


def _estimate_q_semantic_lower_bound(
    module_path: Path,
    *,
    semantic_source_path: Path,
    device: str,
    verbose: bool,
    shape_id: str = "0",
) -> EstimateResult:
    """Estimate the semantic lower bound on bytes moved for one forward pass."""
    loaded = _load_estimate_module(semantic_source_path, device=device, shape_id=shape_id)
    output = _run_forward(loaded)

    input_bytes, unique_input_tensors = _dedup_tensor_bytes(_iter_tensors(loaded.call_inputs))
    state_tensors = _model_state_tensors(loaded.model)
    state_bytes, unique_state_tensors = _dedup_tensor_bytes(state_tensors)
    output_tensors = [tensor for _, tensor in flatten_outputs(output)]
    output_bytes, unique_output_tensors = _dedup_tensor_bytes(output_tensors)
    total_bytes = input_bytes + state_bytes + output_bytes

    precision: dict[str, object] = {
        "status": "exact",
        "estimation_model": "logical_inputs_plus_registered_state_plus_outputs",
    }
    if unique_state_tensors > 0:
        precision["state_accounting"] = "all_registered_parameters_and_buffers"

    details: dict[str, object] = {}
    if verbose:
        details = {
            "inputs": summarize_model_inputs(loaded.call_inputs),
            "init_inputs": summarize_model_inputs(loaded.init_inputs),
            "outputs": summarize_value(output),
            "registered_state": {
                "parameter_count": sum(1 for _ in loaded.model.parameters(recurse=True)),
                "buffer_count": sum(1 for _ in loaded.model.buffers(recurse=True)),
            },
        }

    read_bytes = input_bytes + state_bytes
    write_bytes = output_bytes
    return EstimateResult(
        passed=True,
        mode="Q_semantic_lower_bound",
        operator_name=loaded.operator_name,
        module_path=str(module_path),
        semantic_source_path=str(semantic_source_path),
        device=str(loaded.device),
        value=total_bytes,
        units="bytes",
        precision=precision,
        components={
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "input_bytes": input_bytes,
            "state_bytes": state_bytes,
            "output_bytes": output_bytes,
            "unique_input_tensors": unique_input_tensors,
            "unique_state_tensors": unique_state_tensors,
            "unique_output_tensors": unique_output_tensors,
        },
        details=details,
    )


def _estimate_q_profiled_impl_op_trace(
    module_path: Path,
    *,
    device: str,
    strict: bool,
    verbose: bool,
    shape_id: str = "0",
) -> EstimateResult:
    """Estimate implementation-side bytes from the executed visible ATen op stream."""
    target_dsl = infer_target_dsl(module_path)
    if target_dsl in {"triton", "gluon"}:
        error = (
            "Q_profiled_impl via op_trace_estimate is not reliable for opaque custom kernels "
            f"({target_dsl}). Provide a semantic source for W/Q_semantic_lower_bound, or add "
            "a hardware-profiler backend for the implementation Q path."
        )
        return EstimateResult(
            passed=False,
            mode="Q_profiled_impl",
            operator_name=infer_operator_id(module_path),
            module_path=str(module_path),
            device=str(get_device(device)),
            units="bytes",
            precision={
                "status": "unsupported",
                "backend": "op_trace_estimate",
                "target_dsl": target_dsl,
            },
            error=error,
        )

    trace_result = estimate_theoretical_flops(
        module_path,
        device=device,
        strict=False,
        shape_id=shape_id,
    )
    precision: dict[str, object] = {
        "status": "estimated",
        "backend": "op_trace_estimate",
        "estimation_model": "execution_trace_tensor_io_estimate",
        "target_dsl": target_dsl,
        "bytes_complete": trace_result.bytes_complete,
    }
    if trace_result.heuristic_byte_ops:
        precision["heuristic_byte_ops"] = trace_result.heuristic_byte_ops

    passed = trace_result.passed
    error = trace_result.error
    if strict and trace_result.heuristic_byte_ops:
        passed = False
        error = (
            "Profiled implementation Q used heuristic byte formulas for ops: "
            + ", ".join(trace_result.heuristic_byte_ops)
        )

    details: dict[str, object] = {}
    if verbose:
        details = {
            "inputs": trace_result.inputs,
            "environment": trace_result.environment,
            "counted_byte_ops": trace_result.counted_byte_ops,
            "read_byte_ops": trace_result.read_byte_ops,
            "write_byte_ops": trace_result.write_byte_ops,
            "zero_byte_ops": trace_result.zero_byte_ops,
            "heuristic_byte_op_invocations": trace_result.heuristic_byte_op_invocations,
            "counted_flop_ops": trace_result.counted_ops,
        }

    return EstimateResult(
        passed=passed,
        mode="Q_profiled_impl",
        operator_name=trace_result.operator_name,
        module_path=str(module_path),
        semantic_source_path=None,
        device=trace_result.device,
        value=trace_result.total_bytes,
        units="bytes",
        precision=precision,
        components={
            "read_bytes": trace_result.total_read_bytes,
            "write_bytes": trace_result.total_write_bytes,
            "arithmetic_intensity_flops_per_byte": trace_result.arithmetic_intensity,
        },
        details=details,
        error=error,
    )


def estimate(
    *,
    mode: str,
    module_path: Path,
    device: str = "auto",
    semantic_source_path: Path | None = None,
    profile_backend: str = "auto",
    strict: bool = False,
    verbose: bool = False,
    shape_id: str = "0",
) -> EstimateResult:
    """Run one estimate mode against a reference-style module.

    ``shape_id`` selects which entry from ``shapes.json`` (next to
    ``module_path``) is used. Defaults to ``"0"`` for backward compatibility.
    Per-operator refresh flows pass each shape id in turn.
    """
    if mode not in SUPPORTED_ESTIMATE_MODES:
        return EstimateResult(
            passed=False,
            mode=mode,
            module_path=str(module_path),
            error=(
                f"Unsupported estimate mode: {mode}. "
                f"Expected one of {sorted(SUPPORTED_ESTIMATE_MODES)}."
            ),
        )

    semantic_source = semantic_source_path or module_path
    try:
        if mode == "W_theoretical":
            return _estimate_w_theoretical(
                module_path,
                semantic_source_path=semantic_source,
                device=device,
                strict=strict,
                verbose=verbose,
                shape_id=shape_id,
            )
        if mode == "Q_semantic_lower_bound":
            return _estimate_q_semantic_lower_bound(
                module_path,
                semantic_source_path=semantic_source,
                device=device,
                verbose=verbose,
                shape_id=shape_id,
            )
        if profile_backend not in {"auto", "op_trace_estimate"}:
            return EstimateResult(
                passed=False,
                mode=mode,
                operator_name=infer_operator_id(module_path),
                module_path=str(module_path),
                device=str(get_device(device)),
                units="bytes",
                precision={
                    "status": "unsupported",
                    "backend": profile_backend,
                },
                error=(
                    "Unsupported profile backend in this environment: "
                    f"{profile_backend}. Supported backends: auto, op_trace_estimate."
                ),
            )
        return _estimate_q_profiled_impl_op_trace(
            module_path,
            device=device,
            strict=strict,
            verbose=verbose,
            shape_id=shape_id,
        )
    except Exception:
        return EstimateResult(
            passed=False,
            mode=mode,
            operator_name=infer_operator_id(module_path),
            module_path=str(module_path),
            semantic_source_path=str(semantic_source),
            error=traceback.format_exc(),
        )
