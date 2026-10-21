# None-returning and mutating callables

Eager correctness verification accepts `None` without wrapping the return value
or changing the operator's arguments.

Declare the supplied input names written by the callable in the reference's
adjacent `metadata.json`:

```json
{"benchmark_contract": {"mutates_inputs": ["out", "residual"]}}
```

The evaluator binds actual supplied arguments to `Model.forward`, runs reference
and candidate on independent clones, and checks:

- exact return container/scalar types and tensor dtypes when `mutates_inputs`
  is explicitly present (including an empty list), plus shapes and values;
- every element of each declared mutable input, not just an active prefix;
- undeclared inputs on both sides against their original backing bytes,
  including unused storage tails, with no numerical tolerance;
- cloned strided storage offsets and aliases shared across supplied inputs.

Receipts retain `outputs` and add `mutated_inputs` and `unexpected_mutations` to
each correctness case. The latter includes successful unchanged-input checks as
well as failures. Unknown mutation names fail before invocation. A missing
mutation declaration means no supplied input may change.

Mutation declarations are storage-wide for unchanged-input checks. If `out` and
an undeclared input `x` share storage, writing even a disjoint `out` region changes
`x`'s backing bytes and fails the check. Declare every supplied alias whose
backing storage is written as mutable, or supply independent storage if the ABI
requires `x` to remain immutable. Declared mutable inputs are compared element by
element over their logical views; declaring an alias does not certify bytes
outside the union of those views.

Inputs used only as disposable workspace can be excluded from semantic output
comparison with `scratch_inputs`:

```json
{"benchmark_contract": {"scratch_inputs": ["workspace_buffer"]}}
```

## Per-output policies

An operator may own elementwise tolerances for individual return values and
declared mutations:

```json
{
  "benchmark_contract": {
    "correctness_tolerances": {
      "default": {"atol": 0.01, "rtol": 0.05, "max_mismatch_fraction": 0.0},
      "mutated_inputs.out": {
        "atol": 0.01,
        "rtol": 0.05,
        "max_mismatch_fraction": 0.001,
        "max_rel_l2": 0.01
      }
    },
    "correctness_min_cases": 6
  }
}
```

`default` applies only to floating return values under `output` without a
path-specific policy. It does not apply to declared mutations; each mutation
that needs a metadata-owned policy must use an explicit `mutated_inputs.<key>`
path. The remaining keys must match a value actually compared; stale or
misspelled paths fail evaluation. Every policy requires `atol` and `rtol`; omitted
`max_mismatch_fraction` means zero and omitted `max_rel_l2` means no additional
L2 bound. Fractions use units of one: **0.001 means 0.1%, not 0.001%**.

For each tensor the elementwise condition preserves the historical
`torch.allclose(reference, candidate)` direction:

```
bad = abs(candidate - reference) > atol + rtol * abs(candidate)
sum(bad) <= floor(max_mismatch_fraction * tensor.numel())
```

The allowed count uses exact decimal-to-rational arithmetic, without rounding
small tensors up to one permitted mismatch. Every tensor must pass; elements
are not pooled across outputs or cases. With `max_rel_l2`, also require
`norm(candidate-reference)/max(norm(reference), 1e-12) <= max_rel_l2`. A zero
reference norm requires exact zero error for this additional bound. NaN/Inf are
rejected before either metric. When metadata owns correctness, integer and
boolean reference tensors remain on the exact-comparison path regardless of the
floating policy. A floating-point or complex candidate for such a reference is
rejected; compatible discrete candidate tensors are compared exactly. This
metadata-mode rule is intentionally directional: a floating reference retains
the existing tolerant comparison behavior for an integer candidate.

Receipts include each tensor's effective tolerance, total element count,
mismatched count and allowed count, as well as the existing error metrics. The
receipt `mode` distinguishes exact comparison, elementwise comparison,
elementwise comparison with an L2 conjunct, and the legacy relative-L2 mode.
When metadata owns tolerances, legacy CLI/environment relative-L2 settings cannot
replace them. The legacy global L2-only mode is otherwise unchanged.
`correctness_min_cases` remains a floor on the CLI-requested case count.

## Scope and compatibility

CUDA Graph replay's separate output-only checker has not been extended to
certify mutable state. An eager receipt does not certify graph replay or live
service replacement. Cross-device transfers and arbitrary sparse/nested input
state are not covered by the strided-input validation.

Strided input clones now preserve internally overlapping strides instead of
densifying them. This intentionally changes the previous overlap-cloning
contract. Benchmarks without an explicit `mutates_inputs` field retain legacy
tuple/list, numeric scalar and tensor dtype comparison behavior. Explicit
mutation contracts require exact container/scalar types and tensor dtypes;
floating-point values still use the configured numerical tolerances. `None`
must match `None` in both modes, including nested leaves.

Benchmarks that omit `correctness_tolerances`, `correctness_min_cases`, and
`scratch_inputs` retain the previous global tolerance, case-count, and input
mutation behavior.

## Defined output regions

For native APIs with undefined padding, metadata may declare
`benchmark_contract.output_mask_method` as the name of a public method on the
trusted reference `Model`. The evaluator calls it with a clone of the pristine
forward inputs before either implementation runs. It must return a nonempty
mapping from output tensor paths (for example `output[1]` or `output.scales`) to
boolean tensors with the full physical output shape and at least one true bit.
Omitted output paths retain full comparison. Unknown paths and invalid masks
fail correctness.

The full output structure, shape and dtype are checked before selecting defined
elements. Finiteness, tolerances and mismatch fractions then apply only to those
elements. Input mutation checks are unchanged. The method is never called on
the candidate and is outside forward and its performance timing; it must derive
the defined region from the input contract, not from output values. Preserve the
native output layout rather than returning a compacted tensor from `forward`.

## Input device placement

`benchmark_contract.input_device_policy` defaults to `"move"`: factory tensors
are cloned and moved to the requested evaluation device. Set `"preserve"` for
factories that already construct the exact runtime placement, including GPU
activations and CPU scheduler controls. This policy still clones backing storage
and preserves layout and aliases, but does not move tensors. Such a factory must
select the process's intended GPU itself. The policy applies to forward inputs;
model initialization retains the normal device handling. Unknown policies fail.
