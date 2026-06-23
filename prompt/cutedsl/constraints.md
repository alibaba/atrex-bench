Workflow and evaluation overview:
- You may use the available tools and commands in the workspace to implement, test, benchmark, and iteratively refine the candidate before finalizing it.
- The final generated code will be evaluated by a hidden evaluator that is not provided to you directly.
- That hidden evaluator mainly checks whether the candidate compiles successfully, whether its outputs match the reference implementation, and how close its performance is to the hardware limit.
- These checks are applied sequentially: only candidates that compile successfully are checked for correctness, and only candidates that pass correctness are measured for performance.
- Optimize accordingly, but keep any temporary testing, benchmarking, or debugging workflow out of the final output file.

Performance expectations:
- This is an optimization task, not only a functional translation task.
- The generated implementation will be evaluated on compile success, numerical correctness, and runtime performance.
- Among correct implementations, faster implementations are better.

Output contract:
- Write the final implementation to `generated_kernel.py` in your current working directory. This file is what will be evaluated; nothing you print to chat is read as the candidate.
- The file must contain exactly one self-contained Python module — valid Python source only, no Markdown fences, no explanatory prose, no tests/benchmarks/`__main__` block.

Implementation requirements:
- Include all necessary imports.
- Define one or more CuteDSL kernel, schedule, or program definitions if needed.
- Define `class Model(nn.Module)`.
- `Model.__init__` must accept the same initialization arguments as the reference `Model`.
- `Model.forward` must accept the same inputs as the reference `Model.forward`.
- `Model.forward` must return outputs with the same externally observable structure, shape, device, returned tensor dtype, and numerical behavior as the reference implementation.
- The main compute path must be implemented using CuteDSL kernels or compiled CuteDSL programs launched from `Model.forward`.
- You may use PyTorch only for setup or glue logic such as output allocation, reshape/view, indexing, metadata preparation, and launch orchestration.
- Internal computation precision, accumulation dtype, approximations, and intermediate layouts may differ from the PyTorch reference. You may use lower precision or mixed precision internally, including fp8/fp4/int8 paths, if the returned outputs satisfy the evaluator's numerical tolerance for every evaluated shape.
- Do not call, instantiate, or wrap the original reference `Model`.
- Do not use `torch.compile`, `torch.jit`, custom C++ extensions, or external files.

Correctness requirements:
- Preserve the reference's externally observable semantics exactly.
- Treat internal casts and accumulation choices in the PyTorch reference as defining the numerical target, not as mandatory implementation choices. The generated implementation may use different internal precision or accumulation strategies if final outputs pass correctness.
- Preserve all important masking, routing, indexing, and output-layout behavior implied by the reference code.
- If multiple kernels, schedules, or programs are needed, keep them in the same file and call them from `Model.forward`.

Evaluation contract:
- The generated file will be imported directly by the evaluator.
- The evaluator reads init kwargs and input kwargs from `shapes.json` (one entry per shape, keyed by id) sitting next to `reference.py`. Each entry has `init_kwargs` (passed to `Model(**init_kwargs)`) and `input_kwargs` (passed to `_make_inputs(**input_kwargs)` defined in `input.py`).
- `_make_inputs(**input_kwargs)` returns `dict[str, Tensor]`. The evaluator then calls `Model(**init_kwargs)(**call_inputs)`, so `Model.forward`'s parameter names must match the keys returned by `_make_inputs`.
- The generated candidate file only needs to provide `class Model`; do not redefine `_make_inputs` or read `shapes.json`.
- Outputs: if the reference returns a single tensor, your candidate also returns a single tensor with the same dtype/shape; if the reference returns multiple tensors as a `dict[str, Tensor]`, your candidate must return a dict with the same keys.

Forbidden content:
- Any operator-specific optimization hint not inferable from the reference itself.
- Any explanatory prose, benchmark code, test code, debug print, or placeholder TODO.
- Any fallback that uses the original reference `Model` as the primary execution path.
