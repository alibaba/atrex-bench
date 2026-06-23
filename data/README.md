# Atrex-Bench Data Layout

This directory holds the operator samples for atrex-bench (reference operators). Each operator is a self-contained subdirectory consisting of 5 files.

## Directory Layout

```text
data/
├── README.md                # this document
├── operator_importance.json # operator importance scores
├── fused_moe/               # operator directory
├── block_scaled_mm/
├── attention_forward/
└── ...                      # 30 operator directories total
```

## Operator Directory Layout

```text
data/<operator_name>/
├── reference.py     # class Model(nn.Module) — algorithm only
├── input.py         # _make_inputs(**kwargs) -> dict[str, Tensor]
├── shapes.json      # single source of truth for shapes (dict keyed by id)
├── metadata.json    # id + dtype + input_dtypes + output_dtypes + origin
└── roofline.json    # roofline measurement cache (W / Q / B_peak / SOL_time_ms)
```

| File | Single Responsibility | Agent-Visible | Release-Visible |
|---|---|:---:|:---:|
| `reference.py` | Defines `class Model(nn.Module)` | ✅ | ✅ |
| `input.py` | Exposes `_make_inputs(**input_kwargs) -> dict[str, Tensor]` | ✅ | ✅ |
| `shapes.json` | Per-shape `init_kwargs` + `input_kwargs` | ✅ | ✅ |
| `metadata.json` | Operator id, dtype, tensor-level dtypes, upstream provenance | ❌ | ✅ |
| `roofline.json` | Theoretical W / Q + per-device B_peak + budgeted SOL_time_ms | ❌ | ✅ |

## Adding a New Operator

1. Create the directory: `data/<operator_name>/`
2. Write `reference.py` with **only** `class Model(nn.Module)` — no upstream framework imports, no `__main__`, no provenance comments
3. Write `input.py` exposing **only** `def _make_inputs(**input_kwargs) -> dict[str, Tensor]`; the returned dict keys must match `Model.forward` parameter names one-to-one
4. Write `shapes.json` with at least one `"0"` entry; `init_kwargs` and `input_kwargs` must strictly match `_make_inputs` / `Model.__init__` parameter signatures
5. Write `metadata.json`: assign the next `atrex_NNN` id, set the dominant `dtype`, list `input_dtypes` / `output_dtypes`, fill in `origin`
6. Write `roofline.json` with structural placeholders; measurement fields as `null` (to be backfilled by the refresh script)

## How the Runtime Loads Operators

The evaluator loads a specific shape's `(Model instance, call_inputs dict)` pair through `src/atrex_bench/eval/_runtime.py:load_shape(reference_dir, shape_id)`. This is the single entry point for all consumers.

The workflow is:
1. Read `shapes.json[shape_id]` to obtain `init_kwargs` + `input_kwargs`
2. Import `reference.py`, instantiate `model = Model(**init_kwargs)`
3. Import `input.py`, obtain `call_inputs = _make_inputs(**input_kwargs)` as `dict[str, Tensor]`
4. Drive the forward pass with `model(**call_inputs)`; output may be a single tensor or `dict[str, Tensor]`

## Randomness and Reproducibility

`input.py` does not fix a seed — each call to `_make_inputs(...)` returns fresh random tensors. Reproducibility is guaranteed by checkpoint artifacts persisted during evaluation.
