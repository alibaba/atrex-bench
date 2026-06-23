"""CLI: roofline (speed-of-light) calculator for atrex-bench operators.

Two modes:

  1) Standalone calculator — pass W/Q/dtype/SKU explicitly:
       python scripts/roofline.py \
         --w-flops 8.8e12 --q-bytes 1.6e9 --dtype bf16 \
         --hardware configs/hardware/XPU-A.yaml

  2) Per-operator — refresh roofline.json end-to-end:
       python scripts/roofline.py \
         --operator data/block_scaled_mm \
         --hardware configs/hardware/XPU-A.yaml

     By default this is the *one-shot* flow: for every sid in
     <op_dir>/shapes.json it runs estimate(W_theoretical) +
     estimate(Q_semantic_lower_bound) against reference.py, writes
     ``semantic_W_flops`` (per-dtype) / ``semantic_Q_read_bytes`` /
     ``semantic_Q_write_bytes`` into roofline.json, then computes
     ``SOL_time_ms[<hardware.name>]`` from those values. Failed shapes (OOM,
     model load errors, …) get null W/Q and are reported on stderr.

     Add ``--skip-wq`` to keep existing W/Q values and only recompute SOL
     (CPU-only path; useful when adding a new SKU yaml without re-estimating).
     Add ``--shape-ids 160,199,217-225`` to refresh only a subset.

Output formats: text (default human-readable), yaml/json structured payload.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from atrex_bench.eval.roofline import (
    DTYPE_PATHS,
    RooflineHardware,
    RooflineResult,
    compute_roofline,
    compute_roofline_hybrid,
    load_hardware,
    resolve_dtype_path,
)

# ----- shared helpers -------------------------------------------------------


def _parse_numeric(value: str, *, name: str) -> int:
    """Parse '8.8e12' or '8800000000000' into int (FLOPs / bytes are integral)."""

    try:
        f = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--{name} must be a number, got {value!r}: {exc}"
        ) from exc
    if not math.isfinite(f) or f < 0:
        raise argparse.ArgumentTypeError(
            f"--{name} must be a finite non-negative number, got {value!r}."
        )
    return int(f)


def _format_tflops(flops_per_s: float) -> str:
    return f"{flops_per_s / 1e12:.2f} TFLOPs/s"


def _format_bytes_per_s(bytes_per_s: float) -> str:
    return f"{bytes_per_s / 1e12:.2f} TB/s"


def _result_to_payload(result: RooflineResult) -> dict[str, Any]:
    payload = asdict(result)
    if math.isinf(payload["arithmetic_intensity"]):
        payload["arithmetic_intensity"] = "inf"
    return payload


def _format_text_standalone(
    *,
    hw: RooflineHardware,
    dtype: str,
    w_flops: int,
    q_bytes: int,
    result: RooflineResult,
) -> str:
    ai_label = (
        "inf"
        if math.isinf(result.arithmetic_intensity)
        else f"{result.arithmetic_intensity:.2f}"
    )
    lines = [
        f"Hardware:        {hw.sku_name} ({hw.arch})",
        f"P_peak[{dtype}]:  {_format_tflops(result.p_peak_used)}",
        f"B_peak[hbm]:     {_format_bytes_per_s(result.b_peak_used)}",
        f"Ridge AI:        {result.ridge_point_ai:.2f} FLOPs/byte",
        "",
        "Workload:",
        f"  W:             {w_flops / 1e12:.4f} TFLOP",
        f"  Q:             {q_bytes / 1e9:.4f} GB",
        f"  AI:            {ai_label} FLOPs/byte",
        "",
        "Roofline:",
        f"  P_roof:        {_format_tflops(result.p_roof_flops_per_s)}",
        f"  T_SOL:         {result.sol_time_ms:.4f} ms",
        f"  Bottleneck:    {result.bottleneck}",
    ]
    return "\n".join(lines)


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
    print(f"[OUTPUT] {output}")


def _format_payload(payload: Any, fmt: str) -> str:
    if fmt == "yaml":
        import yaml

        return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ----- standalone mode ------------------------------------------------------


def _run_standalone(args: argparse.Namespace) -> int:
    hw_path = Path(args.hardware)
    hw = load_hardware(hw_path)

    w_flops = _parse_numeric(args.w_flops, name="w-flops")
    q_bytes = _parse_numeric(args.q_bytes, name="q-bytes")

    result = compute_roofline(w_flops, q_bytes, args.dtype, hw)

    if args.format == "text":
        text = _format_text_standalone(
            hw=hw,
            dtype=args.dtype,
            w_flops=w_flops,
            q_bytes=q_bytes,
            result=result,
        )
    else:
        payload = {
            "hardware": {
                "sku_name": hw.sku_name,
                "arch": hw.arch,
                "vendor": hw.vendor,
                "stem": hw.sku_stem,
            },
            "workload": {
                "w_flops": w_flops,
                "q_bytes": q_bytes,
                "dtype": args.dtype,
                "dtype_path": resolve_dtype_path(args.dtype),
            },
            "result": _result_to_payload(result),
        }
        text = _format_payload(payload, args.format)
    _emit(text, args.output)
    return 0


# ----- per-operator mode ----------------------------------------------------


def _load_roofline_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"roofline.json not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a JSON object.")
    return raw


def _parse_shape_ids_arg(spec: str | None) -> list[str] | None:
    """Parse a ``--shape-ids 160,199,217-225`` filter into an ordered sid list.

    Returns ``None`` when the user did not pass the flag (i.e. process all).
    Ranges are expanded inclusively. Whitespace is tolerated.
    """
    if spec is None:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            if lo > hi:
                raise argparse.ArgumentTypeError(
                    f"--shape-ids range out of order: {token!r}"
                )
            for sid in range(lo, hi + 1):
                key = str(sid)
                if key not in seen:
                    out.append(key)
                    seen.add(key)
        else:
            if token not in seen:
                out.append(token)
                seen.add(token)
    return out


# Integer/index/mask dtypes have no published FLOPs/sec peak in vendor specs,
# so the roofline compute leg is skipped for these and SOL collapses to
# T_mem = Q / B_peak. Matches SOL-ExecBench Section 3.2 figure 2(d) "Mixed"
# category for integer-and-boolean-dominated kernels.
_INTEGER_DTYPES: frozenset[str] = frozenset({
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool",
})


def _expected_random_topk_active_experts(
    *,
    token_count: int,
    num_experts: int,
    top_k: int,
) -> int:
    """Expected occupied experts for uniform random top-k routing."""

    routed_slots = token_count * top_k
    if top_k >= num_experts:
        return num_experts

    # For one token, each expert has probability top_k / num_experts of being
    # selected. Across independent token routes, expected occupied experts =
    # E * (1 - P(expert never selected)).
    p_never_selected = math.exp(token_count * math.log1p(-top_k / num_experts))
    expected_active = num_experts * (1.0 - p_never_selected)
    return min(num_experts, routed_slots, math.ceil(expected_active))


def _ensure_sol_block_is_dict(shape_entry: dict[str, Any]) -> dict[str, Any]:
    """Make sure ``shape_entry["SOL_time_ms"]`` is a writable dict.

    Legacy roofline.json data from earlier schemas (or from previously failed
    runs that wrote ``null`` placeholders) may have ``SOL_time_ms = null`` or
    other non-dict values. ``dict.setdefault`` only guards against missing
    keys — when the key exists with ``None`` it returns ``None`` and the
    caller's downstream ``sol_block[hw.sku_name] = ...`` assignment crashes.
    Use this helper instead of ``setdefault`` so legacy null/non-dict values
    are silently normalized to ``{}`` and the new SKU entry can be written.
    Returns the (possibly newly-installed) dict so callers can index into it.
    """
    existing = shape_entry.get("SOL_time_ms")
    if not isinstance(existing, dict):
        existing = {}
        shape_entry["SOL_time_ms"] = existing
    return existing


def _load_metadata_dtype(op_dir: Path) -> str | None:
    """Read metadata.json[dtype] as the authoritative primary precision label.

    Per SOL-ExecBench (Section 3.2 figure 2(d)): "primary compute precision,
    defined as the dtype of the primary data tensors (not accumulation
    buffers)". This is the dtype against whose hardware peak the SOL is
    computed, regardless of what dtype the reference.py happened to dispatch
    at runtime (reference may upcast to fp32 internally for accuracy; the
    operator's nominal precision is what counts).

    Returns the declared dtype string (e.g. ``"bf16"`` / ``"fp8_e4m3"`` /
    ``"int32"``) or ``None`` when the file or field is missing — callers
    should fall back to dispatch-detected dtype with a stderr warning in
    that case so that operators added without metadata still produce
    something usable.
    """
    meta_path = op_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    dtype = meta.get("dtype") if isinstance(meta, dict) else None
    return str(dtype) if isinstance(dtype, str) and dtype else None


def _fp8_blockscale_fused_moe_wq(shape: dict[str, Any]) -> tuple[dict[str, int], int, int]:
    """Return routed-expert semantic W/Q for fp8_blockscale_fused_moe.

    The eager reference dequantizes every expert before applying ``topk_ids``.
    The production kernel receives sorted routed tokens and only touches expert
    weights/scales for experts actually present in the routed slots. The
    benchmark/input helpers generate top-k ids from random logits, but
    shapes.json does not materialize those ids. Use the expected number of
    occupied experts under uniform random top-k routing rather than the
    worst-case ``min(num_experts, token_count * top_k)``; otherwise cases such
    as ``token_count=32, top_k=8, num_experts=256`` incorrectly count every
    expert even though random routing usually covers only about 164 experts.

    The formula below intentionally mirrors the existing reference-level
    accounting for full-expert shapes:
      * per routed token: fc1 + fc2 GEMMs = 6 * H * I FLOPs
      * epilogue/reduce: silu+mul = 5 * I, weighted sum = 2 * H FLOPs
      * per active expert: fp8 weight dequant = 3 * H * I FLOPs
      * Q read: fp8 activations, fp32 activation scales, topk tensors, and
        active expert fp8 weights + fp32 block scales
      * Q write: output tensor in the reference's fp8 activation dtype
    """

    input_kwargs = shape.get("input_kwargs")
    if not isinstance(input_kwargs, dict):
        raise ValueError("fp8_blockscale_fused_moe shape input_kwargs must be a mapping.")

    def _positive_int(name: str, default: int | None = None) -> int:
        raw = input_kwargs.get(name, default)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
            raise ValueError(
                f"fp8_blockscale_fused_moe input_kwargs[{name!r}] must be a "
                f"positive integer, got {raw!r}."
            )
        return raw

    token_count = _positive_int("token_count")
    hidden_size = _positive_int("hidden_size")
    intermediate_size = _positive_int("intermediate_size")
    num_experts = _positive_int("num_experts")
    top_k = _positive_int("top_k")
    scale_block_n = _positive_int("scale_block_n", 128)
    scale_block_k = _positive_int("scale_block_k", 128)

    if hidden_size % scale_block_k != 0:
        raise ValueError(
            "fp8_blockscale_fused_moe hidden_size must be divisible by "
            f"scale_block_k, got hidden_size={hidden_size}, scale_block_k={scale_block_k}."
        )
    if intermediate_size % scale_block_n != 0:
        raise ValueError(
            "fp8_blockscale_fused_moe intermediate_size must be divisible by "
            f"scale_block_n, got intermediate_size={intermediate_size}, "
            f"scale_block_n={scale_block_n}."
        )

    routed_slots = token_count * top_k
    active_experts = _expected_random_topk_active_experts(
        token_count=token_count,
        num_experts=num_experts,
        top_k=top_k,
    )
    hidden_blocks = hidden_size // scale_block_k
    inter_blocks = intermediate_size // scale_block_n
    inter2_blocks = (2 * intermediate_size) // scale_block_n

    fp8_size = 1
    fp32_size = 4

    w1_weight_bytes_per_expert = 2 * intermediate_size * hidden_size * fp8_size
    w2_weight_bytes_per_expert = hidden_size * intermediate_size * fp8_size
    fc1_scale_bytes_per_expert = inter2_blocks * hidden_blocks * fp32_size
    fc2_scale_bytes_per_expert = hidden_blocks * inter_blocks * fp32_size
    expert_read_bytes = active_experts * (
        w1_weight_bytes_per_expert
        + w2_weight_bytes_per_expert
        + fc1_scale_bytes_per_expert
        + fc2_scale_bytes_per_expert
    )

    hidden_read_bytes = token_count * hidden_size * fp8_size
    activation_scale_bytes = token_count * hidden_blocks * fp32_size
    topk_bytes = routed_slots * (fp32_size + 4)  # topk_weights fp32 + topk_ids int32
    output_write_bytes = token_count * hidden_size * fp8_size

    gemm_flops = routed_slots * (6 * hidden_size * intermediate_size)
    epilogue_flops = routed_slots * (5 * intermediate_size + 2 * hidden_size)
    dequant_flops = active_experts * (3 * hidden_size * intermediate_size)
    w_flops = gemm_flops + epilogue_flops + dequant_flops

    q_read = hidden_read_bytes + activation_scale_bytes + topk_bytes + expert_read_bytes
    q_write = output_write_bytes
    return {"fp8_e4m3": w_flops}, q_read, q_write


def _fused_moe_q(shape: dict[str, Any]) -> tuple[int, int]:
    """Return routed-expert semantic Q for bf16 fused_moe.

    ``Q_semantic_lower_bound`` sees the full expert weight tensors as inputs,
    but production fused MoE kernels only read weights for experts represented
    in the routed token slots. This matters for sparse cases such as
    ``token_count=1, top_k=8, num_experts=128`` where only 8 experts can be
    touched, not the full 128-expert pool.
    """

    input_kwargs = shape.get("input_kwargs")
    if not isinstance(input_kwargs, dict):
        raise ValueError("fused_moe shape input_kwargs must be a mapping.")
    init_kwargs = shape.get("init_kwargs")
    if init_kwargs is not None and not isinstance(init_kwargs, dict):
        raise ValueError("fused_moe shape init_kwargs must be null or a mapping.")
    init_kwargs = init_kwargs or {}

    def _positive_int(name: str, default: int | None = None) -> int:
        raw = input_kwargs.get(name, init_kwargs.get(name, default))
        if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
            raise ValueError(
                f"fused_moe input_kwargs/init_kwargs[{name!r}] must be a "
                f"positive integer, got {raw!r}."
            )
        return raw

    token_count_raw = input_kwargs.get(
        "token_count",
        input_kwargs.get(
            "num_tokens",
            init_kwargs.get("token_count", init_kwargs.get("num_tokens")),
        ),
    )
    if (
        not isinstance(token_count_raw, int)
        or isinstance(token_count_raw, bool)
        or token_count_raw <= 0
    ):
        raise ValueError(
            "fused_moe input_kwargs/init_kwargs['token_count'/'num_tokens'] "
            f"must be a positive integer, got {token_count_raw!r}."
        )
    token_count = token_count_raw
    hidden_size = _positive_int("hidden_size")
    intermediate_size = _positive_int("intermediate_size", hidden_size * 4)
    num_experts = _positive_int("num_experts", 8)
    top_k = _positive_int("top_k", 2)

    active_experts = _expected_random_topk_active_experts(
        token_count=token_count,
        num_experts=num_experts,
        top_k=top_k,
    )

    bf16_size = 2
    fp32_size = 4
    int32_size = 4
    routed_slots = token_count * top_k

    hidden_read = token_count * hidden_size * bf16_size
    w1_read = active_experts * (2 * intermediate_size * hidden_size * bf16_size)
    w2_read = active_experts * (hidden_size * intermediate_size * bf16_size)
    topk_read = routed_slots * (fp32_size + int32_size)
    output_write = token_count * hidden_size * fp32_size

    return hidden_read + w1_read + w2_read + topk_read, output_write


def _chunk_gated_delta_rule_state_q(shape: dict[str, Any]) -> tuple[int, int]:
    """Return active-row semantic Q for chunk_gated_delta_rule_state.

    The eager reference clones and returns the full ``final_state`` tensor, but
    the production kernel receives ``initial_state_indices`` and only reads and
    writes the state rows touched by the current batch. Counting all
    ``state_count`` rows makes SOL overly conservative for trace shapes where
    ``state_count`` is the KV/state pool capacity rather than the active batch.
    """

    input_kwargs = shape.get("input_kwargs")
    if not isinstance(input_kwargs, dict):
        raise ValueError("chunk_gated_delta_rule_state shape input_kwargs must be a mapping.")
    init_kwargs = shape.get("init_kwargs")
    if init_kwargs is not None and not isinstance(init_kwargs, dict):
        raise ValueError("chunk_gated_delta_rule_state shape init_kwargs must be null or a mapping.")
    init_kwargs = init_kwargs or {}

    def _positive_int(name: str, default: int | None = None) -> int:
        raw = input_kwargs.get(name, init_kwargs.get(name, default))
        if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
            raise ValueError(
                f"chunk_gated_delta_rule_state input_kwargs/init_kwargs[{name!r}] "
                f"must be a positive integer, got {raw!r}."
            )
        return raw

    batch_size = _positive_int("batch_size", 1)
    token_count = _positive_int("token_count")
    num_k_heads = _positive_int("num_k_heads")
    num_v_heads = _positive_int("num_v_heads")
    key_dim = _positive_int("key_dim")
    value_dim = _positive_int("value_dim")
    state_count = _positive_int("state_count")
    chunk_size = _positive_int("chunk_size", 64)

    active_state_rows = min(batch_size, state_count)
    num_chunks = (token_count + chunk_size - 1) // chunk_size

    bf16_size = 2
    fp32_size = 4
    int32_size = 4

    k_read = batch_size * token_count * num_k_heads * key_dim * bf16_size
    w_read = batch_size * token_count * num_v_heads * key_dim * bf16_size
    u_read = batch_size * token_count * num_v_heads * value_dim * bf16_size
    g_read = batch_size * token_count * num_v_heads * fp32_size
    state_row_bytes = num_v_heads * key_dim * value_dim * fp32_size
    active_state_read = active_state_rows * state_row_bytes
    indices_read = batch_size * int32_size

    h_chunks_write = (
        batch_size * num_chunks * num_v_heads * key_dim * value_dim * bf16_size
    )
    v_new_write = batch_size * token_count * num_v_heads * value_dim * bf16_size
    active_state_write = active_state_rows * state_row_bytes

    q_read = k_read + w_read + u_read + g_read + active_state_read + indices_read
    q_write = h_chunks_write + v_new_write + active_state_write
    return q_read, q_write


def _operator_static_wq_override(
    op_name: str,
    shape: dict[str, Any],
) -> tuple[dict[str, int], int, int] | None:
    if op_name == "fp8_blockscale_fused_moe":
        return _fp8_blockscale_fused_moe_wq(shape)
    return None


def _operator_static_q_override(
    op_name: str,
    shape: dict[str, Any],
) -> tuple[int, int] | None:
    if op_name == "fused_moe":
        return _fused_moe_q(shape)
    if op_name == "chunk_gated_delta_rule_state":
        return _chunk_gated_delta_rule_state_q(shape)
    return None


def _refresh_wq_from_reference(
    op_dir: Path,
    raw: dict[str, Any],
    *,
    selected_shape_ids: list[str] | None,
    device: str = "auto",
) -> tuple[list[str], list[str]]:
    """Populate semantic_W_flops / semantic_Q_*_bytes for every selected sid.

    Iterates ``shapes.json`` (filtered by ``selected_shape_ids`` if set), runs
    ``estimate(W_theoretical)`` + ``estimate(Q_semantic_lower_bound)`` for each
    sid, and writes the values into ``raw['shapes'][sid]``. Orphan entries
    (in roofline.json but absent from shapes.json) are removed when no shape
    filter is active. Failed shapes get null W/Q (so the caller can retry
    later) and their sid is collected in the returned ``failed`` list.

    Importing ``estimate`` here (instead of at module top) keeps the SOL-only
    path (``--skip-wq``) free of the torch/CUDA dependency chain.
    """
    from atrex_bench.eval.estimate import estimate

    reference = op_dir / "reference.py"
    shapes_json = op_dir / "shapes.json"
    if not reference.is_file() or not shapes_json.is_file():
        raise FileNotFoundError(
            f"{op_dir}: reference.py and shapes.json are required for "
            "automatic W/Q estimation. Either add them or pass --skip-wq."
        )

    sj = json.loads(shapes_json.read_text(encoding="utf-8"))
    if not isinstance(sj, dict) or not sj:
        raise ValueError(f"{shapes_json}: must be a non-empty JSON object.")

    available_sids = sorted(sj.keys(), key=lambda s: (len(s), s))
    if selected_shape_ids is None:
        sids = available_sids
    else:
        unknown = [s for s in selected_shape_ids if s not in sj]
        if unknown:
            raise ValueError(
                f"--shape-ids references sids not in shapes.json: {unknown}"
            )
        sids = list(selected_shape_ids)

    shapes_block: dict[str, Any] = raw.setdefault("shapes", {})
    if selected_shape_ids is None:
        for orphan in [s for s in shapes_block if s not in sj]:
            del shapes_block[orphan]

    metadata_dtype = _load_metadata_dtype(op_dir)
    if metadata_dtype is None:
        print(
            f"[refresh-wq][WARN] {op_dir.name}: metadata.json[dtype] missing; "
            "falling back to dispatch-detected dtype (may misclassify reference "
            "implementations that upcast for accuracy).",
            file=sys.stderr,
        )

    refreshed: list[str] = []
    failed: list[str] = []
    for idx, sid in enumerate(sids, 1):
        print(
            f"[refresh-wq] [{idx}/{len(sids)}] {op_dir.name} sid={sid} ...",
            file=sys.stderr,
        )
        entry = shapes_block.setdefault(sid, {})
        try:
            static_wq = _operator_static_wq_override(op_dir.name, sj[sid])
            static_q = _operator_static_q_override(op_dir.name, sj[sid])
            if static_wq is not None:
                semantic_w_flops, q_read, q_write = static_wq
            else:
                w_res = estimate(
                    mode="W_theoretical",
                    module_path=reference,
                    shape_id=sid,
                    device=device,
                )
                if not w_res.passed:
                    raise RuntimeError(
                        f"W_theoretical failed: {w_res.error or 'unknown'}"
                    )

                if static_q is not None:
                    q_read, q_write = static_q
                else:
                    q_res = estimate(
                        mode="Q_semantic_lower_bound",
                        module_path=reference,
                        shape_id=sid,
                        device=device,
                    )
                    if not q_res.passed:
                        raise RuntimeError(
                            f"Q_semantic_lower_bound failed: {q_res.error or 'unknown'}"
                        )

                    q_read = int(
                        q_res.components.get("read_bytes", 0)
                        if isinstance(q_res.components, dict)
                        else 0
                    )
                    q_write = int(
                        q_res.components.get("write_bytes", 0)
                        if isinstance(q_res.components, dict)
                        else 0
                    )

                # W bucket key resolution (per SOL-ExecBench primary-precision rule):
                #   1. metadata.json[dtype] if present — authoritative
                #   2. dispatch-detected primary dtype as fallback
                #   3. integer dtypes → empty W bucket (compute leg skipped, SOL = Q/B_peak)
                total_flops = int(w_res.value or 0)
                if metadata_dtype is not None:
                    if metadata_dtype in _INTEGER_DTYPES:
                        semantic_w_flops = {}
                    else:
                        semantic_w_flops = {metadata_dtype: total_flops}
                else:
                    dispatch_buckets = (
                        w_res.components.get("flops_by_dtype")
                        if isinstance(w_res.components, dict)
                        else None
                    ) or {}
                    semantic_w_flops = (
                        {k: int(v) for k, v in dispatch_buckets.items()}
                        if dispatch_buckets
                        else {"bf16": total_flops}
                    )

            entry["semantic_W_flops"] = semantic_w_flops
            entry["semantic_Q_read_bytes"] = q_read
            entry["semantic_Q_write_bytes"] = q_write
            _ensure_sol_block_is_dict(entry)
            refreshed.append(sid)
        except Exception as exc:
            entry["semantic_W_flops"] = None
            entry["semantic_Q_read_bytes"] = None
            entry["semantic_Q_write_bytes"] = None
            _ensure_sol_block_is_dict(entry)
            print(
                f"[refresh-wq][FAIL] sid={sid}: {exc}",
                file=sys.stderr,
            )
            failed.append(sid)

    return refreshed, failed


def _shape_compute(
    *,
    shape_id: str,
    shape_block: dict[str, Any],
    hw: RooflineHardware,
    explicit_dtype: str | None,
) -> tuple[RooflineResult, dict[str, int], int, int]:
    """Compute roofline for one shape block. Returns (result, w_by_dtype, q_read, q_write).

    ``semantic_W_flops == {}`` (empty mapping) signals an integer-only / no-FP
    operator: the compute leg of the roofline is skipped and SOL collapses to
    ``T_mem = Q / B_peak`` (vendor specs do not publish integer ops/sec peaks,
    so a compute ceiling cannot be derived).
    """

    semantic_w = shape_block.get("semantic_W_flops")
    if not isinstance(semantic_w, dict):
        raise ValueError(
            f"shape {shape_id!r}: semantic_W_flops must be a mapping; "
            f"got {semantic_w!r}."
        )
    w_by_dtype_raw: dict[str, int] = {}
    for dtype, value in semantic_w.items():
        if value is None:
            raise ValueError(
                f"shape {shape_id!r}: semantic_W_flops[{dtype!r}] is null. "
                f"Fill in the theoretical FLOPs before computing roofline."
            )
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(
                f"shape {shape_id!r}: semantic_W_flops[{dtype!r}] must be a "
                f"non-negative number, got {value!r}."
            )
        w_by_dtype_raw[dtype] = int(value)

    if explicit_dtype is not None:
        if explicit_dtype not in w_by_dtype_raw:
            raise ValueError(
                f"shape {shape_id!r}: --dtype={explicit_dtype!r} is not in "
                f"semantic_W_flops keys {sorted(w_by_dtype_raw)}."
            )
        w_by_dtype: dict[str, int] = {explicit_dtype: w_by_dtype_raw[explicit_dtype]}
    else:
        w_by_dtype = w_by_dtype_raw

    q_read = shape_block.get("semantic_Q_read_bytes")
    q_write = shape_block.get("semantic_Q_write_bytes")
    if q_read is None or q_write is None:
        raise ValueError(
            f"shape {shape_id!r}: semantic_Q_read_bytes / semantic_Q_write_bytes "
            f"must be filled in before computing roofline; got "
            f"read={q_read!r}, write={q_write!r}."
        )
    if not isinstance(q_read, (int, float)) or not isinstance(q_write, (int, float)):
        raise ValueError(
            f"shape {shape_id!r}: semantic_Q_*_bytes must be numbers, got "
            f"read={q_read!r}, write={q_write!r}."
        )
    q_read_int = int(q_read)
    q_write_int = int(q_write)
    q_total = q_read_int + q_write_int

    if not w_by_dtype:
        # Integer-only / no-FP operator: no compute peak applies; SOL = Q/B_peak.
        sol_time_s = q_total / hw.b_peak_hbm if hw.b_peak_hbm > 0 else 0.0
        result = RooflineResult(
            arithmetic_intensity=0.0,
            ridge_point_ai=0.0,
            p_roof_flops_per_s=0.0,
            sol_time_s=sol_time_s,
            sol_time_ms=sol_time_s * 1000.0,
            bottleneck="memory",
            p_peak_used=0,
            b_peak_used=hw.b_peak_hbm,
        )
    elif len(w_by_dtype) == 1:
        only_dtype = next(iter(w_by_dtype))
        result = compute_roofline(w_by_dtype[only_dtype], q_total, only_dtype, hw)
    else:
        result = compute_roofline_hybrid(w_by_dtype, q_total, hw)

    return result, w_by_dtype, q_read_int, q_write_int


def _format_text_per_op(
    *,
    op_dir: Path,
    hw: RooflineHardware,
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"Operator:    {op_dir}",
        f"Hardware:    {hw.sku_name} ({hw.arch}) [stem={hw.sku_stem}]",
        "P_peak:      "
        + ", ".join(f"{p}={v / 1e12:.2f}TF" for p, v in sorted(hw.p_peak.items())),
        f"B_peak.hbm:  {_format_bytes_per_s(hw.b_peak_hbm)}",
        "",
    ]
    if not rows:
        lines.append("(no shapes)")
        return "\n".join(lines)

    header = (
        f"{'shape':>6} {'dtype':>14} {'AI(F/B)':>12} "
        f"{'P_roof':>14} {'T_SOL':>14} {'bottleneck':>12}"
    )
    sep = "-" * len(header)
    lines += [header, sep]
    for row in rows:
        ai_str = (
            "inf"
            if math.isinf(row["arithmetic_intensity"])
            else f"{row['arithmetic_intensity']:.2f}"
        )
        lines.append(
            f"{row['shape_id']:>6} "
            f"{row['dtype']:>14} "
            f"{ai_str:>12} "
            f"{row['p_roof_tflops']:>11.2f} TF "
            f"{row['sol_time_ms']:>11.4f} ms "
            f"{row['bottleneck']:>12}"
        )
    return "\n".join(lines)


def _run_per_operator(args: argparse.Namespace) -> int:
    op_dir = Path(args.operator)
    if not op_dir.is_dir():
        raise FileNotFoundError(f"Operator directory not found: {op_dir}")
    roofline_path = op_dir / "roofline.json"
    raw = _load_roofline_json(roofline_path)

    hw_path = Path(args.hardware)
    hw = load_hardware(hw_path)

    shapes_block = raw.get("shapes")
    if shapes_block is not None and not isinstance(shapes_block, dict):
        raise ValueError(
            f"{roofline_path}: top-level 'shapes' must be a JSON object."
        )

    selected_shape_ids = _parse_shape_ids_arg(args.shape_ids)
    if args.shape_id is not None:
        if selected_shape_ids is not None:
            raise ValueError(
                "--shape-id and --shape-ids are mutually exclusive."
            )
        selected_shape_ids = [args.shape_id]

    refresh_failed: list[str] = []
    if not args.skip_wq:
        try:
            _refreshed, refresh_failed = _refresh_wq_from_reference(
                op_dir,
                raw,
                selected_shape_ids=selected_shape_ids,
                device=args.estimate_device,
            )
            shapes_block = raw.get("shapes")
        except FileNotFoundError as exc:
            print(
                f"[refresh-wq][SKIP] {exc} (continuing in SOL-only mode)",
                file=sys.stderr,
            )

    if not isinstance(shapes_block, dict) or not shapes_block:
        raise ValueError(
            f"{roofline_path}: top-level 'shapes' must be a non-empty mapping. "
            "Either populate it directly, or provide reference.py + shapes.json "
            "and let the default --skip-wq=False path estimate W/Q for you."
        )

    if selected_shape_ids is not None:
        unknown = [s for s in selected_shape_ids if s not in shapes_block]
        if unknown:
            raise ValueError(
                f"{roofline_path}: shape ids not in roofline.shapes: {unknown}; "
                f"available: {sorted(shapes_block)}."
            )
        shape_ids = list(selected_shape_ids)
    else:
        shape_ids = sorted(shapes_block, key=lambda s: (len(s), s))

    rows: list[dict[str, Any]] = []
    sol_skipped: list[str] = []
    sol_unsupported: list[str] = []
    for shape_id in shape_ids:
        shape_block = shapes_block[shape_id]
        if not isinstance(shape_block, dict):
            raise ValueError(
                f"{roofline_path}: shapes[{shape_id!r}] must be a mapping."
            )
        if (
            shape_block.get("semantic_W_flops") is None
            or shape_block.get("semantic_Q_read_bytes") is None
            or shape_block.get("semantic_Q_write_bytes") is None
        ):
            sol_skipped.append(shape_id)
            continue
        try:
            result, w_by_dtype, q_read, q_write = _shape_compute(
                shape_id=shape_id,
                shape_block=shape_block,
                hw=hw,
                explicit_dtype=args.dtype,
            )
        except KeyError as exc:
            # Hardware doesn't publish a peak for the operator's primary dtype
            # (e.g. block_scaled_mm declares fp8_e4m3 but some GPUs lack fp8 tensor
            # cores). Record null SOL for this SKU so the broadcast can keep
            # going; a separate sol_unsupported summary surfaces these for
            # follow-up review. Any other raise from _shape_compute (schema
            # bug, math bug) still propagates so the user notices.
            if not args.no_write:
                sol_block = _ensure_sol_block_is_dict(shape_block)
                sol_block[hw.sku_name] = None
            sol_unsupported.append(shape_id)
            print(
                f"[SOL-skip] sid={shape_id}: {hw.sku_name} lacks dtype path "
                f"({exc})",
                file=sys.stderr,
            )
            continue
        if args.dtype is not None:
            dtype_label = args.dtype
        elif w_by_dtype:
            dtype_label = "+".join(sorted(w_by_dtype))
        else:
            # Integer-only op: no FP compute, only memory-bound SOL is meaningful.
            dtype_label = "(int/mem-only)"
        rows.append(
            {
                "shape_id": shape_id,
                "dtype": dtype_label,
                "w_flops_total": sum(w_by_dtype.values()),
                "q_read_bytes": q_read,
                "q_write_bytes": q_write,
                "arithmetic_intensity": result.arithmetic_intensity,
                "ridge_point_ai": result.ridge_point_ai,
                "p_roof_tflops": result.p_roof_flops_per_s / 1e12,
                "sol_time_ms": result.sol_time_ms,
                "bottleneck": result.bottleneck,
            }
        )

        if not args.no_write:
            sol_block = _ensure_sol_block_is_dict(shape_block)
            sol_block[hw.sku_name] = result.sol_time_ms

    if not args.no_write:
        roofline_path.write_text(
            json.dumps(raw, indent=2) + "\n", encoding="utf-8"
        )

    if args.format == "text":
        text = _format_text_per_op(op_dir=op_dir, hw=hw, rows=rows)
        if not args.no_write:
            text += f"\n\n[WROTE] {roofline_path}"
        if sol_skipped:
            text += (
                f"\n\n[SKIPPED-SOL] {len(sol_skipped)} shapes had null W/Q "
                f"(first 10: {sol_skipped[:10]})"
            )
        if sol_unsupported:
            text += (
                f"\n[UNSUPPORTED-SOL] {len(sol_unsupported)} shapes: "
                f"{hw.sku_name} lacks p_peak for the operator's primary dtype "
                f"(first 10: {sol_unsupported[:10]}). SOL_time_ms[<sku>] written "
                "as null for these."
            )
        if refresh_failed:
            text += (
                f"\n[FAILED-WQ] {len(refresh_failed)} shapes failed to "
                f"estimate (first 10: {refresh_failed[:10]})"
            )
    else:
        payload = {
            "operator": str(op_dir),
            "roofline_path": str(roofline_path),
            "hardware": {
                "sku_name": hw.sku_name,
                "arch": hw.arch,
                "vendor": hw.vendor,
                "stem": hw.sku_stem,
            },
            "shapes": rows,
            "wrote_back": not args.no_write,
            "sol_skipped_shape_ids": sol_skipped,
            "sol_unsupported_shape_ids": sol_unsupported,
            "refresh_failed_shape_ids": refresh_failed,
        }
        # JSON cannot encode inf — replace with the string "inf" for AI
        for row in payload["shapes"]:
            if math.isinf(row["arithmetic_intensity"]):
                row["arithmetic_intensity"] = "inf"
        text = _format_payload(payload, args.format)
    _emit(text, args.output)
    return 0 if (not refresh_failed and not sol_unsupported) else 1


# ----- argparse -------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute roofline (speed-of-light) upper bounds for atrex-bench "
            "operators. Two modes: standalone (--w-flops/--q-bytes/--dtype/"
            "--hardware) or per-operator (--operator/--hardware reads "
            "roofline.json and writes back SOL_time_ms[<hardware.name>])."
        )
    )
    parser.add_argument(
        "--hardware",
        required=True,
        help=(
            "Path to a SKU profile under configs/hardware/<sku>.yaml. The "
            "hardware.name value is used as the SOL_time_ms map key in "
            "per-operator mode."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="text",
        help="Output format. text (default) is human-readable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path; prints to stdout if omitted.",
    )

    # Standalone mode
    parser.add_argument(
        "--w-flops",
        default=None,
        help=(
            "Standalone mode: theoretical FLOPs (e.g. 8.8e12). Mutually "
            "exclusive with --operator."
        ),
    )
    parser.add_argument(
        "--q-bytes",
        default=None,
        help=(
            "Standalone mode: theoretical data movement bytes (e.g. 1.6e9, "
            "= read + write). Mutually exclusive with --operator."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Standalone mode: dtype name (one of "
            f"{sorted(DTYPE_PATHS)}). In per-operator mode this is optional "
            "and selects a single dtype from semantic_W_flops; omit to use "
            "all keys (hybrid roofline)."
        ),
    )

    # Per-operator mode
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Per-operator mode: path to an operator directory containing "
            "roofline.json. Mutually exclusive with --w-flops/--q-bytes."
        ),
    )
    parser.add_argument(
        "--shape-id",
        default=None,
        help=(
            "Per-operator mode: optional single shape id to compute. "
            "Mutually exclusive with --shape-ids. Defaults to all shapes."
        ),
    )
    parser.add_argument(
        "--shape-ids",
        default=None,
        help=(
            "Per-operator mode: comma-separated subset filter with optional "
            "range syntax, e.g. '160,199,217-225'. Mutually exclusive with "
            "--shape-id."
        ),
    )
    parser.add_argument(
        "--skip-wq",
        action="store_true",
        help=(
            "Per-operator mode: do NOT re-estimate semantic_W_flops / "
            "semantic_Q_*_bytes from reference.py; use whatever values are "
            "already in roofline.json. Default behaviour is to refresh W/Q "
            "for every selected shape via estimate(W_theoretical) + "
            "estimate(Q_semantic_lower_bound). Useful when adding a new SKU "
            "yaml without re-running the (CUDA-bound) estimate phase."
        ),
    )
    parser.add_argument(
        "--estimate-device",
        default="auto",
        help=(
            "Per-operator mode: device for the W/Q estimate phase "
            "(auto/cpu/cuda/cuda:0/hip/...). Default 'auto'. Ignored when "
            "--skip-wq is set."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Per-operator mode: do not write SOL_time_ms back into roofline.json."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    standalone = args.w_flops is not None or args.q_bytes is not None
    per_op = args.operator is not None

    if standalone and per_op:
        parser.error(
            "--operator cannot be combined with --w-flops/--q-bytes; pick one mode."
        )
    if not standalone and not per_op:
        parser.error(
            "must pass either --operator <dir> (per-operator) or "
            "--w-flops/--q-bytes/--dtype (standalone)."
        )

    if standalone:
        missing = [
            flag
            for flag, val in (
                ("--w-flops", args.w_flops),
                ("--q-bytes", args.q_bytes),
                ("--dtype", args.dtype),
            )
            if val is None
        ]
        if missing:
            parser.error(
                f"standalone mode requires {missing} (got "
                f"w_flops={args.w_flops!r}, q_bytes={args.q_bytes!r}, "
                f"dtype={args.dtype!r})."
            )
        return _run_standalone(args)
    return _run_per_operator(args)


if __name__ == "__main__":
    raise SystemExit(main())
