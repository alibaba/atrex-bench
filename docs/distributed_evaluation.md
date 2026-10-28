# Single-node collective evaluation

References declare `hardware.required_gpus` and
`benchmark_contract.distributed_evaluation` with `launcher: torchrun`,
`backend: nccl`, and a matching `world_size` (2–64). The caller reserves
that many GPUs on one node (Agate: `--num-gpus N`). Each shape launches N ranks,
binds one distinct visible GPU per local rank, and initializes NCCL before
constructing inputs. Every rank must pass correctness before eager timing.

Timings use fixed iteration counts, a barrier before each CUDA-event sample,
and the maximum rank latency per iteration. The result retains rank/device
identities, every rank's correctness cases, and raw rank timing samples under
`performance.shapes.<id>.distributed`. CUDA graph timing is currently rejected
for this contract. A missing rank, timeout, or rank failure cannot produce a pass.
ProcessGroup input handles are shared by identity when tensor inputs are cloned.
Allocation size and the contract must agree; reserving GPUs alone does not launch ranks.
