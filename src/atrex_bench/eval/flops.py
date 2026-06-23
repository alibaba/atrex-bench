"""Best-effort theoretical compute and data-movement estimation for references."""

from __future__ import annotations

import math
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from atrex_bench.eval._runtime import (
    ShapeSpec,
    clone_model_inputs,
    get_accelerator_backend,
    get_core_package_versions,
    get_device,
    get_platform_label,
    get_python_version,
    import_module_from_path,
    infer_operator_id,
    load_model_init_inputs,
    load_reference_inputs,
    load_shape_call_inputs,
    load_shape_init_inputs,
    load_shape_spec,
    resolve_input_module,
    summarize_model_inputs,
    validate_reference_module,
)


@dataclass(frozen=True)
class FlopEstimateResult:
    """Result of estimating theoretical compute cost for one reference module."""

    passed: bool
    complete: bool = False
    flops_complete: bool = False
    bytes_complete: bool = False
    total_flops: int | None = None
    flops_by_dtype: dict[str, int] = field(default_factory=dict)
    total_bytes: int | None = None
    total_read_bytes: int | None = None
    total_write_bytes: int | None = None
    arithmetic_intensity: float | None = None
    operator_name: str | None = None
    device: str | None = None
    environment: dict[str, object] = field(default_factory=dict)
    inputs: dict[str, object] = field(default_factory=dict)
    counted_ops: dict[str, int] = field(default_factory=dict)
    zero_flop_ops: dict[str, int] = field(default_factory=dict)
    uncounted_ops: list[str] = field(default_factory=list)
    uncounted_op_invocations: dict[str, int] = field(default_factory=dict)
    counted_byte_ops: dict[str, int] = field(default_factory=dict)
    read_byte_ops: dict[str, int] = field(default_factory=dict)
    write_byte_ops: dict[str, int] = field(default_factory=dict)
    zero_byte_ops: dict[str, int] = field(default_factory=dict)
    heuristic_byte_ops: list[str] = field(default_factory=list)
    heuristic_byte_op_invocations: dict[str, int] = field(default_factory=dict)
    error: str | None = None


# Map torch float dtypes to the short names used in roofline.json /
# semantic_W_flops keys (and recognized by DTYPE_PATHS in eval/roofline.py).
# Integer dtypes are intentionally absent: integer ops contribute zero FLOPs
# in the FlopCounterMode formulas and are not classified.
_FLOAT_DTYPE_SHORT_NAMES: dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float64: "fp64",
}
# fp8 dtypes only exist on torch >= 2.1; skip silently if a deployment is older.
for _name, _short in (("float8_e4m3fn", "fp8_e4m3"), ("float8_e5m2", "fp8_e5m2")):
    _candidate = getattr(torch, _name, None)
    if isinstance(_candidate, torch.dtype):
        _FLOAT_DTYPE_SHORT_NAMES[_candidate] = _short


def _resolve_primary_dtype_short_name(args: Any) -> str | None:
    """Return the short dtype name of the first floating-point tensor in args.

    Used to bucket per-op FLOPs into ``flops_by_dtype``. Walks nested
    list/tuple/dict structures and returns ``None`` if no floating-point
    tensor is found (e.g. integer-only ops, structural ops).
    """
    stack: list[Any] = [args]
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            if value.dtype.is_floating_point:
                return _FLOAT_DTYPE_SHORT_NAMES.get(value.dtype)
            continue
        if isinstance(value, (tuple, list)):
            stack.extend(reversed(value))
            continue
        if isinstance(value, dict):
            stack.extend(reversed(list(value.values())))
            continue
    return None


class _DtypeAwareFlopCounter(FlopCounterMode):
    """FlopCounterMode that also accumulates per-dtype FLOPs.

    Overrides ``_count_flops`` to tag each op invocation by its primary
    input dtype (the dtype of the first floating-point tensor in args).
    The base class continues to receive the same per-op accounting so all
    its public APIs (``get_flop_counts``, ``get_total_flops``) keep working.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.flops_by_dtype: Counter[str] = Counter()

    def _count_flops(
        self,
        func_packet: Any,
        out: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)
            for par in set(self.mod_tracker.parents):
                self.flop_counts[par][func_packet] += flop_count
            if flop_count > 0:
                dtype_name = _resolve_primary_dtype_short_name(args)
                if dtype_name is not None:
                    self.flops_by_dtype[dtype_name] += int(flop_count)
        return out


class _ExecutionRecorder(TorchDispatchMode):
    """Record executed ops and estimate per-op data movement."""

    def __init__(self) -> None:
        super().__init__()
        self.op_packets: Counter[Any] = Counter()
        self.byte_counts: Counter[Any] = Counter()
        self.read_byte_counts: Counter[Any] = Counter()
        self.write_byte_counts: Counter[Any] = Counter()
        self.zero_byte_packets: Counter[Any] = Counter()
        self.heuristic_byte_packets: Counter[Any] = Counter()
        self.byte_mapping = _build_byte_mapping()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        packet = getattr(func, "_overloadpacket", func)
        out = func(*args, **kwargs)
        self.op_packets[packet] += 1

        byte_formula = self.byte_mapping.get(packet)
        if byte_formula is None:
            self.heuristic_byte_packets[packet] += 1
            read_bytes, write_bytes = _generic_tensor_io_bytes(
                *args, out_val=out, **kwargs
            )
        else:
            read_bytes, write_bytes = byte_formula(*args, out_val=out, **kwargs)
            if read_bytes == 0 and write_bytes == 0:
                self.zero_byte_packets[packet] += 1

        self.read_byte_counts[packet] += int(read_bytes)
        self.write_byte_counts[packet] += int(write_bytes)
        self.byte_counts[packet] += int(read_bytes) + int(write_bytes)
        return out


def _shape_numel(value: Any) -> int:
    """Return the total element count for a shape-like structure."""
    if value is None:
        return 0
    if isinstance(value, torch.Size):
        return math.prod(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        if not value:
            return 1
        if all(isinstance(dim, (int, torch.SymInt)) for dim in value):
            return math.prod(int(dim) for dim in value)
        return sum(_shape_numel(item) for item in value)
    return 1


def _normalize_dim(dim: int, rank: int) -> int:
    """Normalize a possibly-negative reduction dim."""
    return dim if dim >= 0 else dim + rank


def _tensor_num_bytes(value: Any) -> int:
    """Return the total byte footprint for all tensors in a nested structure."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_num_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_num_bytes(item) for item in value.values())
    return 0


def _zero_flop(*args, out_shape=None, **kwargs) -> int:
    """Mark structural or non-floating-point ops as zero-FLOP."""
    del args, out_shape, kwargs
    return 0


def _pointwise_flop(
    *args,
    out_shape=None,
    per_element: int = 1,
    **kwargs,
) -> int:
    """Count one or more floating-point ops per output element."""
    del args, kwargs
    return per_element * _shape_numel(out_shape)


def _add_flop(a_shape, b_shape, *, alpha=1, out_shape=None, **kwargs) -> int:
    """Count add/sub style pointwise ops, including optional alpha scaling."""
    del a_shape, b_shape, kwargs
    flops = _shape_numel(out_shape)
    if alpha != 1:
        flops += _shape_numel(out_shape)
    return flops


def _sum_flop(input_shape, dim=None, keepdim=False, *, out_shape=None, **kwargs) -> int:
    """Count floating-point additions in a sum reduction."""
    del dim, keepdim, kwargs
    return max(_shape_numel(input_shape) - _shape_numel(out_shape), 0)


def _mean_flop(input_shape, dim=None, keepdim=False, *, out_shape=None, **kwargs) -> int:
    """Count floating-point adds plus one division per reduced output."""
    del dim, keepdim, kwargs
    input_numel = _shape_numel(input_shape)
    output_numel = _shape_numel(out_shape)
    return max(input_numel - output_numel, 0) + output_numel


def _softmax_flop(input_shape, dim=None, half_to_float=False, *, out_shape=None, **kwargs) -> int:
    """Approximate softmax as subtract-max, exp, reduction, and division."""
    del half_to_float, kwargs
    total_elements = _shape_numel(input_shape)
    if total_elements == 0:
        return 0

    if isinstance(input_shape, torch.Size):
        rank = len(input_shape)
        reduce_size = int(input_shape[_normalize_dim(int(dim), rank)]) if rank else 1
    elif isinstance(input_shape, tuple) and all(
        isinstance(axis, (int, torch.SymInt)) for axis in input_shape
    ):
        rank = len(input_shape)
        reduce_size = int(input_shape[_normalize_dim(int(dim), rank)]) if rank else 1
    else:
        reduce_size = total_elements

    groups = max(total_elements // max(reduce_size, 1), 1)
    subtract_flops = total_elements
    exp_flops = total_elements
    reduction_flops = max(total_elements - groups, 0)
    division_flops = _shape_numel(out_shape)
    return subtract_flops + exp_flops + reduction_flops + division_flops


def _native_layer_norm_flop(
    input_shape,
    normalized_shape=None,
    weight=None,
    bias=None,
    eps=None,
    *,
    out_shape=None,
    **kwargs,
) -> int:
    """LayerNorm FLOPs, per element of the input.

    Decomposition (per row of length N = prod(normalized_shape)):
      * mean reduction:                   ~N adds  + 1 div per row
      * (x - μ) and (x - μ)²:             N subs + N muls
      * variance reduction:               ~N adds  + 1 div per row
      * rsqrt(var + eps):                 1 add + 1 rsqrt per row
      * normalize (x - μ) * invstd:       N muls    (the sub is reused
                                                     from the variance leg)
      * gamma * x_hat (when affine):      N muls
      * + beta         (when affine):     N adds

    Per-row constant overhead (1 div, 1 rsqrt, 1 add for eps) is dropped
    because N typically dominates -- the formula returns
    ``5*N`` per row without affine and ``7*N`` with affine. Matches the
    convention used by xformers / fvcore for LayerNorm reporting.

    ``weight`` / ``bias`` arrive as torch.Size shapes when affine is on,
    or ``None`` when off; we detect affine via the not-None test.
    """
    del normalized_shape, eps, out_shape, kwargs
    total_elements = _shape_numel(input_shape)
    if total_elements == 0:
        return 0
    has_affine = weight is not None or bias is not None
    return total_elements * (7 if has_affine else 5)


def _generic_tensor_io_bytes(*args, out_val=None, **kwargs) -> tuple[int, int]:
    """Estimate (read, write) bytes as one read of all inputs plus one write of all outputs."""
    read_bytes = _tensor_num_bytes(args) + _tensor_num_bytes(kwargs)
    write_bytes = _tensor_num_bytes(out_val)
    return read_bytes, write_bytes


def _zero_byte(*args, out_val=None, **kwargs) -> tuple[int, int]:
    """Mark pure metadata/view ops as zero-byte for the current model."""
    del args, out_val, kwargs
    return 0, 0


def _write_only_bytes(*args, out_val=None, **kwargs) -> tuple[int, int]:
    """Estimate (read, write) bytes for tensor factory ops that only materialize outputs."""
    del args, kwargs
    return 0, _tensor_num_bytes(out_val)


def _empty_allocation_bytes(*args, out_val=None, **kwargs) -> tuple[int, int]:
    """Treat empty allocations as metadata-only because they do not initialize contents."""
    del args, out_val, kwargs
    return 0, 0


def _tensor_read_only_bytes(*args, out_val=None, **kwargs) -> tuple[int, int]:
    """Estimate (read, write) bytes for ops that only read tensors and return Python scalars."""
    del out_val
    return _tensor_num_bytes(args) + _tensor_num_bytes(kwargs), 0


def _copy_bytes(dst, src, *args, out_val=None, **kwargs) -> tuple[int, int]:
    """Estimate (read, write) bytes for explicit copies; dst is the destination, not a read."""
    del dst, args, kwargs
    return _tensor_num_bytes(src), _tensor_num_bytes(out_val)


def _build_custom_flop_mapping() -> dict[Any, Any]:
    """Build extra flop formulas on top of PyTorch's built-in registry."""
    pointwise_packets = {
        "add": _add_flop,
        "add_": _add_flop,
        "sub": _add_flop,
        "sub_": _add_flop,
        "mul": lambda *args, out_shape=None, **kwargs: _pointwise_flop(
            *args,
            out_shape=out_shape,
            **kwargs,
        ),
        "mul_": lambda *args, out_shape=None, **kwargs: _pointwise_flop(
            *args,
            out_shape=out_shape,
            **kwargs,
        ),
        "div": lambda *args, out_shape=None, **kwargs: _pointwise_flop(
            *args,
            out_shape=out_shape,
            **kwargs,
        ),
        "div_": lambda *args, out_shape=None, **kwargs: _pointwise_flop(
            *args,
            out_shape=out_shape,
            **kwargs,
        ),
        "silu": lambda *args, out_shape=None, **kwargs: _pointwise_flop(
            *args,
            out_shape=out_shape,
            per_element=4,
            **kwargs,
        ),
        "silu_": lambda *args, out_shape=None, **kwargs: _pointwise_flop(
            *args,
            out_shape=out_shape,
            per_element=4,
            **kwargs,
        ),
    }
    reduction_packets = {
        "sum": _sum_flop,
        "mean": _mean_flop,
        "_softmax": _softmax_flop,
        "softmax": _softmax_flop,
        # LayerNorm (F.layer_norm + nn.LayerNorm both lower to native_layer_norm).
        # Has real FLOPs that the default registry misses; see _native_layer_norm_flop
        # for the per-element accounting.
        "native_layer_norm": _native_layer_norm_flop,
    }
    zero_flop_packets = [
        "_to_copy",
        "_local_scalar_dense",
        "_unsafe_view",
        "abs",
        "abs_",
        "alias",
        "amax",
        "any",
        "as_strided",
        "clamp",
        "clamp_",
        "clone",
        "copy_",
        "detach",
        "empty",
        "empty_like",
        "empty_strided",
        "eq",
        "ge",
        "gt",
        "expand",
        "full",
        "index",
        "index_put_",
        "le",
        "lift_fresh",
        "lt",
        # ``masked_fill`` / ``masked_fill_`` write a fixed value into masked
        # positions; no floating-point arithmetic happens, only a
        # comparison + selection. By the FLOP convention (only counts FP
        # multiplies / adds / divs / transcendentals) this is genuinely 0
        # FLOPs. Listed explicitly so the counter classifies it as
        # zero_flop instead of leaving it in uncounted_ops.
        "masked_fill",
        "masked_fill_",
        "ne",
        "permute",
        "relu",
        "relu_",
        "repeat_interleave",
        "reshape",
        "select",
        "slice",
        "slice_scatter",
        "squeeze",
        "t",
        "to",
        # ``topk`` returns the K largest values + their indices via a
        # partial sort. The sort itself is comparison-only (no FP arith);
        # value selection is a copy. 0 FLOPs by convention. Memory traffic
        # is what dominates for topk-shaped ops.
        "topk",
        "transpose",
        "triu",
        "unsqueeze",
        "view",
        "where",
        "zeros",
        "zeros_like",
    ]

    mapping: dict[Any, Any] = {}
    for name, formula in pointwise_packets.items():
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None:
            mapping[packet] = formula
    for name, formula in reduction_packets.items():
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None:
            mapping[packet] = formula
    for name in zero_flop_packets:
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None:
            mapping[packet] = _zero_flop
    return mapping


def _build_byte_mapping() -> dict[Any, Any]:
    """Build byte-estimation formulas for executed aten ops."""
    generic_io_packets = [
        "_softmax",
        "_to_copy",
        "abs",
        "abs_",
        "add",
        "add_",
        "amax",
        "any",
        "clamp",
        "clamp_",
        "clone",
        "div",
        "div_",
        "eq",
        "ge",
        "gt",
        "index",
        "index_put_",
        "le",
        "lt",
        # masked_fill: read inputs (input + mask), write output of same
        # shape as input. Generic IO covers it.
        "masked_fill",
        "masked_fill_",
        "matmul",
        "mean",
        "mm",
        "mul",
        "mul_",
        # native_layer_norm: read input + (optional) weight + bias,
        # write the normalized output (mean / rstd auxiliary outputs are
        # row-shaped and bundled into out_val by the recorder).
        "native_layer_norm",
        "ne",
        "relu",
        "relu_",
        "repeat_interleave",
        "silu",
        "silu_",
        "softmax",
        "sub",
        "sub_",
        "sum",
        # topk: read full input, write top-K values + indices. Output bytes
        # are dominated by the K-shaped slices, well-modelled by the
        # generic IO formula (read all inputs once, write all outputs
        # once).
        "topk",
        "where",
        "zeros_like",
    ]
    zero_byte_packets = [
        "_unsafe_view",
        "alias",
        "as_strided",
        "detach",
        "expand",
        "lift_fresh",
        "permute",
        "reshape",
        "select",
        "slice",
        "squeeze",
        "t",
        "transpose",
        "unsqueeze",
        "view",
    ]
    write_only_packets = [
        "full",
        "zeros",
        "zeros_like",
    ]
    empty_packets = [
        "empty",
        "empty_like",
        "empty_strided",
    ]

    mapping: dict[Any, Any] = {}
    for name in generic_io_packets:
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None:
            mapping[packet] = _generic_tensor_io_bytes

    copy_packet = getattr(torch.ops.aten, "copy_", None)
    if copy_packet is not None:
        mapping[copy_packet] = _copy_bytes

    scalar_packet = getattr(torch.ops.aten, "_local_scalar_dense", None)
    if scalar_packet is not None:
        mapping[scalar_packet] = _tensor_read_only_bytes

    for name in zero_byte_packets:
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None:
            mapping[packet] = _zero_byte
    for name in write_only_packets:
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None:
            mapping[packet] = _write_only_bytes
    for name in empty_packets:
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None:
            mapping[packet] = _empty_allocation_bytes
    return mapping


def _build_environment_payload(resolved_device: torch.device) -> dict[str, object]:
    """Build a concise environment summary for FLOP estimation outputs."""
    return {
        "device": str(resolved_device),
        "accelerator_backend": get_accelerator_backend(),
        "platform": get_platform_label(),
        "python_version": get_python_version(),
        "packages": get_core_package_versions(),
    }


def _serialize_counted_ops(global_counts: dict[Any, int]) -> dict[str, int]:
    """Convert packet keyed flop counts into a stable JSON-friendly mapping."""
    return {
        str(packet): int(flops)
        for packet, flops in sorted(global_counts.items(), key=lambda item: str(item[0]))
        if int(flops) != 0
    }


def _serialize_zero_flop_ops(
    recorder_counts: Counter[Any],
    global_counts: dict[Any, int],
    supported_packets: set[Any],
) -> dict[str, int]:
    """Report executed ops that are intentionally classified as zero-FLOP."""
    zero_packets = {
        packet
        for packet, flops in global_counts.items()
        if packet in supported_packets and int(flops) == 0
    }
    return {
        str(packet): int(recorder_counts[packet])
        for packet in sorted(zero_packets, key=str)
    }


def _serialize_uncounted_op_invocations(
    recorder_counts: Counter[Any],
    supported_packets: set[Any],
) -> dict[str, int]:
    """Report invocation counts for executed ops without FLOP formulas."""
    return {
        str(packet): int(recorder_counts[packet])
        for packet in sorted(recorder_counts, key=str)
        if packet not in supported_packets
    }


def _serialize_counted_byte_ops(byte_counts: Counter[Any]) -> dict[str, int]:
    """Convert packet keyed byte counts into a stable JSON-friendly mapping."""
    return {
        str(packet): int(byte_count)
        for packet, byte_count in sorted(byte_counts.items(), key=lambda item: str(item[0]))
        if int(byte_count) != 0
    }


def _serialize_directional_byte_ops(direction_counts: Counter[Any]) -> dict[str, int]:
    """Serialize per-packet read or write byte counts, dropping zero entries."""
    return {
        str(packet): int(byte_count)
        for packet, byte_count in sorted(
            direction_counts.items(), key=lambda item: str(item[0])
        )
        if int(byte_count) != 0
    }


def _serialize_zero_byte_ops(zero_byte_packets: Counter[Any]) -> dict[str, int]:
    """Serialize the zero-byte packet invocation counters."""
    return {
        str(packet): int(count)
        for packet, count in sorted(zero_byte_packets.items(), key=lambda item: str(item[0]))
    }


def _serialize_heuristic_byte_ops(heuristic_packets: Counter[Any]) -> dict[str, int]:
    """Serialize packets that used the generic byte-estimation fallback."""
    return {
        str(packet): int(count)
        for packet, count in sorted(heuristic_packets.items(), key=lambda item: str(item[0]))
    }


def estimate_theoretical_flops(
    reference_path: Path,
    *,
    device: str = "auto",
    strict: bool = False,
    shape_id: str = "0",
) -> FlopEstimateResult:
    """Estimate theoretical FLOPs for one reference Model.forward() invocation.

    ``shape_id`` selects which entry from ``shapes.json`` (next to
    ``reference_path``) is used to construct Model and call inputs. Defaults to
    ``"0"`` for backward compatibility with single-shape callers; per-operator
    refresh flows pass each shape id in turn.
    """
    try:
        resolved_device = get_device(device)
        module = import_module_from_path(reference_path, "atrex_reference_flops")
        validate_reference_module(module)
        input_module = resolve_input_module(
            reference_path, module, module_prefix="atrex_input_flops"
        )
        shape: ShapeSpec | None
        if (reference_path.parent / "shapes.json").is_file():
            shape = load_shape_spec(reference_path, shape_id)
            init_inputs = load_shape_init_inputs(shape, resolved_device)
        else:
            shape = None
            init_inputs = load_model_init_inputs(input_module, resolved_device)
        model_inputs = clone_model_inputs(init_inputs)
        model = module.Model(*model_inputs.args, **model_inputs.kwargs).to(resolved_device).eval()
        if shape is not None:
            inputs = load_shape_call_inputs(input_module, shape, resolved_device)
        else:
            inputs = load_reference_inputs(input_module, resolved_device)
        input_summary = summarize_model_inputs(inputs)

        recorder = _ExecutionRecorder()
        counter = _DtypeAwareFlopCounter(
            display=False,
            custom_mapping=_build_custom_flop_mapping(),
        )
        call_inputs = clone_model_inputs(inputs)
        with recorder:
            with counter:
                with torch.inference_mode():
                    model(*call_inputs.args, **call_inputs.kwargs)

        global_counts = counter.get_flop_counts().get("Global", {})
        counted_ops = _serialize_counted_ops(global_counts)
        supported_packets = set(counter.flop_registry.keys())
        zero_flop_ops = _serialize_zero_flop_ops(
            recorder.op_packets,
            global_counts,
            supported_packets,
        )
        uncounted_op_invocations = _serialize_uncounted_op_invocations(
            recorder.op_packets,
            supported_packets,
        )
        uncounted_ops = sorted(
            {
                str(packet)
                for packet in recorder.op_packets
                if packet not in supported_packets
            }
        )
        total_flops = int(counter.get_total_flops())
        flops_by_dtype = {
            name: int(value)
            for name, value in sorted(counter.flops_by_dtype.items())
            if value > 0
        }
        counted_byte_ops = _serialize_counted_byte_ops(recorder.byte_counts)
        read_byte_ops = _serialize_directional_byte_ops(recorder.read_byte_counts)
        write_byte_ops = _serialize_directional_byte_ops(recorder.write_byte_counts)
        zero_byte_ops = _serialize_zero_byte_ops(recorder.zero_byte_packets)
        heuristic_byte_op_invocations = _serialize_heuristic_byte_ops(
            recorder.heuristic_byte_packets
        )
        heuristic_byte_ops = sorted(heuristic_byte_op_invocations)
        total_read_bytes = sum(read_byte_ops.values())
        total_write_bytes = sum(write_byte_ops.values())
        total_bytes = total_read_bytes + total_write_bytes
        arithmetic_intensity = (
            float(total_flops) / float(total_bytes)
            if total_bytes > 0
            else None
        )
        flops_complete = not uncounted_ops
        bytes_complete = True
        complete = flops_complete and bytes_complete
        error = None
        passed = True
        if strict and (uncounted_ops or not bytes_complete):
            passed = False
            error_parts: list[str] = []
            if uncounted_ops:
                error_parts.append(
                    "Observed ops without FLOP formulas: " + ", ".join(uncounted_ops)
                )
            if not bytes_complete:
                error_parts.append("Observed ops without byte formulas.")
            error = "\n".join(error_parts)

        return FlopEstimateResult(
            passed=passed,
            complete=complete,
            flops_complete=flops_complete,
            bytes_complete=bytes_complete,
            total_flops=total_flops,
            flops_by_dtype=flops_by_dtype,
            total_bytes=total_bytes,
            total_read_bytes=total_read_bytes,
            total_write_bytes=total_write_bytes,
            arithmetic_intensity=arithmetic_intensity,
            operator_name=infer_operator_id(reference_path),
            device=str(resolved_device),
            environment=_build_environment_payload(resolved_device),
            inputs=input_summary,
            counted_ops=counted_ops,
            zero_flop_ops=zero_flop_ops,
            uncounted_ops=uncounted_ops,
            uncounted_op_invocations=uncounted_op_invocations,
            counted_byte_ops=counted_byte_ops,
            read_byte_ops=read_byte_ops,
            write_byte_ops=write_byte_ops,
            zero_byte_ops=zero_byte_ops,
            heuristic_byte_ops=heuristic_byte_ops,
            heuristic_byte_op_invocations=heuristic_byte_op_invocations,
            error=error,
        )
    except Exception:
        return FlopEstimateResult(
            passed=False,
            complete=False,
            flops_complete=False,
            bytes_complete=False,
            operator_name=infer_operator_id(reference_path),
            error=traceback.format_exc(),
        )
