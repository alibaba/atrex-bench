#!/usr/bin/env bash
# refresh_all_roofline.sh — refresh roofline.json across every operator under
# data/ against a single hardware SKU.
#
# Best-effort policy: per-op failures (OOM, schema gaps, hw-dtype mismatches)
# do NOT abort the batch. Each op's exit code is categorized and the loop
# advances; a summary block at the end lists ok / partial / hard-fail buckets.
# Operators that completed successfully have their roofline.json fully written
# before the next op starts, so a partial batch can be re-run without losing
# already-finished work.
#
# Usage:
#
#   # Foreground (default XPU-A):
#   bash scripts/refresh_all_roofline.sh
#
#   # Override SKU:
#   bash scripts/refresh_all_roofline.sh configs/hardware/XPU-A.yaml
#
#   # Background (survives ssh disconnect, recommended for full sweeps):
#   nohup bash scripts/refresh_all_roofline.sh < /dev/null > /dev/null 2>&1 &
#   echo "PID: $!"
#   #
#   # The script's tee already mirrors stdout+stderr to a timestamped log file
#   # under logs/, so /dev/null on the nohup line is fine — you don't lose
#   # output.
#
#   # Monitor live progress:
#   tail -f logs/roofline_XPU-A_*.log
#
#   # Check final summary after run:
#   cat logs/roofline_XPU-A_*.summary
#
# SIGINT (Ctrl+C in foreground) cleanly stops the batch and prints a partial
# summary; in background, send `kill -INT <pid>` to achieve the same.
#
# Output:
#   - logs/roofline_<sku-stem>_<utc-timestamp>.log     full mirrored output
#   - logs/roofline_<sku-stem>_<utc-timestamp>.summary final per-bucket summary
#   - per-op header `===== [i/N] <op_name> ===========` for grep
#   - per-op footer `[i/N] <op_name> rc=X (Ys)`        for grep

# Note: deliberately NOT using `set -e` — failures are expected and handled
# explicitly per-op via the exit-code categorization below.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HW="${1:-configs/hardware/XPU-A.yaml}"
if [[ ! -f "$HW" ]]; then
  echo "Hardware yaml not found: $HW" >&2
  exit 2
fi

SKU_STEM="$(basename "$HW" .yaml)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/roofline_${SKU_STEM}_${STAMP}.log"
SUMMARY_FILE="$LOG_DIR/roofline_${SKU_STEM}_${STAMP}.summary"

# Force unbuffered Python output so `tail -f` shows real-time progress on
# the background-running broadcast (default buffering would batch into ~4KB
# chunks and make it look hung).
export PYTHONUNBUFFERED=1

ops=()
for op in data/*/; do
  ops+=("$op")
done
total=${#ops[@]}

ok_ops=()
partial_ops=()
hard_fail_ops=()
interrupted=0

handle_sigint() {
  interrupted=1
  echo
  echo "[refresh-all][INTERRUPT] SIGINT received; stopping after current op." | tee -a "$LOG_FILE"
}
trap handle_sigint INT TERM

write_summary() {
  local final_status="$1"
  local processed=$((${#ok_ops[@]} + ${#partial_ops[@]} + ${#hard_fail_ops[@]}))
  {
    echo "===== refresh_all_roofline.sh SUMMARY ====="
    echo "status:           $final_status"
    echo "hardware:         $HW"
    echo "log:              $LOG_FILE"
    echo "operators total:  $total"
    echo "operators run:    $processed"
    echo "  ok          (${#ok_ops[@]}):       ${ok_ops[*]:-(none)}"
    echo "  partial     (${#partial_ops[@]}):  ${partial_ops[*]:-(none)}"
    echo "  hard_fail   (${#hard_fail_ops[@]}):  ${hard_fail_ops[*]:-(none)}"
  } | tee -a "$LOG_FILE" > "$SUMMARY_FILE"
}

refresh_unified_attention_actual_wq() {
  local op_dir="$1"
  python - "$op_dir" <<'PY'
import json
import math
import sys
from pathlib import Path
from typing import Any

DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "int16": 2,
    "uint16": 2,
    "fp32": 4,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
}


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a JSON object")
    return raw


def _primary_dtype(op_dir: Path) -> str:
    metadata_path = op_dir / "metadata.json"
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        dtype = metadata.get("dtype")
        if isinstance(dtype, str) and dtype:
            return dtype
    return "bf16"


def _dtype_size(dtype: str) -> int:
    normalized = dtype.removeprefix("torch.").lower()
    if normalized not in DTYPE_BYTES:
        raise ValueError(f"unsupported unified_attention dtype: {dtype!r}")
    return DTYPE_BYTES[normalized]


def _int_list(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"input_kwargs[{name!r}] must be a non-empty list")
    return [int(item) for item in value]


def _query_seq_lens(input_kwargs: dict[str, Any], kv_seq_lens: list[int]) -> list[int]:
    for key in ("q_seq_lens", "query_seq_lens", "seq_lens_q", "q_lens"):
        value = input_kwargs.get(key)
        if isinstance(value, list):
            query_lens = _int_list(value, name=key)
            if len(query_lens) != len(kv_seq_lens):
                raise ValueError(
                    f"input_kwargs[{key!r}] length {len(query_lens)} does not "
                    f"match seq_lens length {len(kv_seq_lens)}"
                )
            return query_lens

    num_query_tokens = input_kwargs.get("num_query_tokens")
    if num_query_tokens is not None:
        query_token_count = int(num_query_tokens)
        if len(kv_seq_lens) == 1:
            return [query_token_count]
        if query_token_count == len(kv_seq_lens):
            return [1] * len(kv_seq_lens)
        raise ValueError(
            "num_query_tokens cannot be unambiguously distributed across "
            f"{len(kv_seq_lens)} sequences"
        )

    return kv_seq_lens


def _attention_pair_count(query_lens: list[int], kv_lens: list[int]) -> int:
    pair_count = 0
    for query_len, kv_len in zip(query_lens, kv_lens, strict=True):
        if query_len < 0 or kv_len < 0:
            raise ValueError("sequence lengths must be non-negative")
        if query_len == kv_len:
            pair_count += query_len * (query_len + 1) // 2
        else:
            pair_count += query_len * kv_len
    return pair_count


def _actual_wq(input_kwargs: dict[str, Any], dtype_size: int) -> tuple[int, int, int]:
    kv_seq_lens = _int_list(input_kwargs.get("seq_lens"), name="seq_lens")
    query_lens = _query_seq_lens(input_kwargs, kv_seq_lens)
    num_query_heads = int(input_kwargs["num_query_heads"])
    num_kv_heads = int(input_kwargs["num_kv_heads"])
    head_size = int(input_kwargs["head_size"])
    block_size = int(input_kwargs.get("block_size", 16))

    pair_count = _attention_pair_count(query_lens, kv_seq_lens)
    flops = 4 * pair_count * num_query_heads * head_size

    query_tokens = sum(query_lens)
    kv_tokens = sum(kv_seq_lens)
    query_bytes = query_tokens * num_query_heads * head_size * dtype_size
    kv_bytes = 2 * kv_tokens * num_kv_heads * head_size * dtype_size
    block_table_entries = sum(math.ceil(kv_len / block_size) for kv_len in kv_seq_lens)
    metadata_bytes = (len(kv_seq_lens) + 1) * 4 + len(kv_seq_lens) * 4 + block_table_entries * 4
    read_bytes = query_bytes + kv_bytes + metadata_bytes
    write_bytes = query_bytes
    return flops, read_bytes, write_bytes


def main() -> None:
    op_dir = Path(sys.argv[1])
    roofline_path = op_dir / "roofline.json"
    shapes_path = op_dir / "shapes.json"
    roofline = _load_json(roofline_path)
    shape_defs = _load_json(shapes_path)
    dtype = _primary_dtype(op_dir)
    dtype_size = _dtype_size(dtype)
    shapes_block = roofline.setdefault("shapes", {})
    if not isinstance(shapes_block, dict):
        raise ValueError(f"{roofline_path}: top-level 'shapes' must be a JSON object")

    for orphan_shape_id in [shape_id for shape_id in shapes_block if shape_id not in shape_defs]:
        del shapes_block[orphan_shape_id]

    updated = 0
    for shape_id in sorted(shape_defs, key=lambda key: (len(key), key)):
        shape_def = shape_defs[shape_id]
        if not isinstance(shape_def, dict):
            raise ValueError(f"{shapes_path}: shape {shape_id!r} must be a JSON object")
        input_kwargs = shape_def.get("input_kwargs")
        if not isinstance(input_kwargs, dict):
            raise ValueError(f"{shapes_path}: shape {shape_id!r} missing input_kwargs")
        flops, read_bytes, write_bytes = _actual_wq(input_kwargs, dtype_size)
        shape_entry = shapes_block.setdefault(shape_id, {})
        if not isinstance(shape_entry, dict):
            shape_entry = {}
            shapes_block[shape_id] = shape_entry
        shape_entry["semantic_W_flops"] = {dtype: flops}
        shape_entry["semantic_Q_read_bytes"] = read_bytes
        shape_entry["semantic_Q_write_bytes"] = write_bytes
        if not isinstance(shape_entry.get("SOL_time_ms"), dict):
            shape_entry["SOL_time_ms"] = {}
        updated += 1

    roofline_path.write_text(json.dumps(roofline, indent=2) + "\n", encoding="utf-8")
    print(
        "[refresh-wq][unified_attention] wrote actual-participating W/Q "
        f"for {updated} shapes in {roofline_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
PY
}

{
  echo "[refresh-all] hardware:    $HW"
  echo "[refresh-all] operators:   $total"
  echo "[refresh-all] log:         $LOG_FILE"
  echo "[refresh-all] summary:     $SUMMARY_FILE"
  echo "[refresh-all] policy:      best-effort (continues on per-op failures)"
  echo "[refresh-all] started:     $STAMP"
  echo
} | tee -a "$LOG_FILE"

i=0
for op in "${ops[@]}"; do
  if [[ $interrupted -eq 1 ]]; then
    break
  fi
  i=$((i + 1))
  name="$(basename "$op")"
  start_ts=$(date +%s)
  printf '===== [%d/%d] %s =====\n' "$i" "$total" "$name" | tee -a "$LOG_FILE"

  if [[ "$name" == "unified_attention" ]]; then
    refresh_unified_attention_actual_wq "$op" 2>&1 | tee -a "$LOG_FILE"
    prep_rc=${PIPESTATUS[0]}
    if [[ $prep_rc -eq 0 ]]; then
      python scripts/roofline.py \
          --operator "$op" \
          --hardware "$HW" \
          --skip-wq 2>&1 | tee -a "$LOG_FILE"
      rc=${PIPESTATUS[0]}
    else
      rc=$prep_rc
    fi
  else
    python scripts/roofline.py \
        --operator "$op" \
        --hardware "$HW" \
        --estimate-device cuda 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
  fi

  elapsed=$(( $(date +%s) - start_ts ))
  case "$rc" in
    0)
      ok_ops+=("$name")
      label="ok"
      ;;
    1)
      partial_ops+=("$name")
      label="partial"
      ;;
    *)
      hard_fail_ops+=("$name:rc=$rc")
      label="hard_fail"
      ;;
  esac
  printf '[%d/%d] %s rc=%d (%ds) [%s]\n' \
      "$i" "$total" "$name" "$rc" "$elapsed" "$label" | tee -a "$LOG_FILE"
  echo | tee -a "$LOG_FILE"
done

if [[ $interrupted -eq 1 ]]; then
  write_summary "INTERRUPTED"
  exit 130
fi

if [[ ${#hard_fail_ops[@]} -gt 0 ]]; then
  write_summary "DONE_WITH_HARD_FAILURES"
  exit 1
fi

write_summary "DONE_CLEAN"
exit 0
