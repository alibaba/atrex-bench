Your task is to optimize the following PyTorch reference operator using CuteDSL for the target GPU. PyTorch reference: {{REFERENCE_CODE}}.

You must follow the workflow and complete the optimization goals defined in gpu-kernel-optimizer. Use CuteDSL kernel, schedule, or program definitions for the main compute path. Do not use third-party operator libraries as part of your implementation output.

Important: Optimize the kernel with the gpu-kernel-optimizer first, then output the results, and finally evaluate it based on the `constraints.md`.
