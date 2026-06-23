Your task is to optimize the following PyTorch reference operator using Triton for the target GPU. PyTorch reference: {{REFERENCE_CODE}}.

You must follow the workflow and complete the optimization goals defined in `constraints.md`. Use `@triton.jit` kernel functions for the main compute path. Do not use third-party operator libraries as part of your implementation output.

Important: Optimize the kernel first, then output the results and evaluate.
