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
      "output[0]": {"atol": 0.01, "rtol": 0.01},
      "mutated_inputs.out": {"atol": 0.06, "rtol": 0.04}
    },
    "correctness_min_cases": 6
  }
}
```

Paths must match a value actually compared by the evaluator; stale or
misspelled paths fail evaluation. When `correctness_tolerances` is present it
is authoritative and legacy global relative-L2 or mismatch-rate options cannot
replace its elementwise checks. `correctness_min_cases` is a floor: a larger
CLI-requested case count is retained.

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
