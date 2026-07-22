<div align="center">

# Atrex-Bench

**End-to-End Operator Generation Benchmark for Multiple DSLs**

Production-trace driven · Executable PyTorch references · Source-aware metadata

[![status](https://img.shields.io/badge/status-early%20stage-0F766E)](#)
[![python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB)](#)

</div>

---

Atrex-Bench measures whether an **agent platform + model** stack can turn a PyTorch reference
into a DSL kernel that **compiles**, **runs correctly**, and **approaches the achievable
hardware peak** (speed-of-light, SOL).

Every operator ships as a small, self-contained directory: an executable PyTorch `Model`, an
input generator, a shape spec, and hidden source-aware metadata. The evaluator runs a candidate
through three stages — compile → correctness → performance — and reports SOL efficiency against
a cached roofline.

<p align="center">
  <img src="assets/atrex-bench-pipeline.png" alt="Atrex-Bench pipeline from production traces to roofline scoring" width="100%">
</p>

## Highlights

- **30 operators**, mostly derived from real production traces (vLLM / SGLang / AITER / rtp-llm).
- **4 DSL backends** out of the box: Triton, Gluon, FlyDSL, CuteDSL.
- **Multi-vendor GPUs**: the same operators, references, and evaluator run on both AMD (ROCm) and NVIDIA (CUDA), with prebuilt images for each.
- **Three-stage evaluator**: compile, numerical correctness, and performance vs. roofline SOL.
- **Generation harness** that drives an LLM CLI (Claude Code / Codex) inside a scoped workspace.
- **Optional optimization agent**: [Atrex Kernel Agent](https://github.com/alibaba/atrex-kernel-agent)
  provides a profile-driven workflow for GPU kernel implementation and iterative optimization.
- **Leak-resistant by design**: a one-shot cleanup strips the checkout to the agent-visible
  surface, and provenance / roofline numbers are never staged for the agent.

## Environment

Everything runs inside a container — a **clean runtime** with PyTorch, GPU-accelerated kernels, Node 22, and the Claude Code / Codex CLIs, and **no project code**.
You `git clone` the repo *inside* the container (see [Usage](#usage)), so the code you run is
always your own checkout — no host mounts.

We provide prebuilt images for both AMD and NVIDIA GPUs:

| Platform | Image |
|----------|-------|
| AMD (ROCm) | `treinfra/atrex-bench:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.10.0` |
| NVIDIA (CUDA) | `treinfra/atrex-bench:2.10.0-cuda12.8-cudnn9-devel` |

**1 · Pull or build the image** for your GPU platform:

```bash
# AMD GPU (ROCm)
docker pull treinfra/atrex-bench:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.10.0
# or build from source:
docker build -t atrex-bench:rocm -f docker/Dockerfile.rocm .

# NVIDIA GPU (CUDA)
docker pull treinfra/atrex-bench:2.10.0-cuda12.8-cudnn9-devel
# or build from source:
docker build -t atrex-bench:cuda -f docker/Dockerfile.cuda .
```

**2 · Start the container.** Everything under [Usage](#usage) — including your API key — runs inside this shell:

```bash
# AMD GPU
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  treinfra/atrex-bench:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.10.0

# NVIDIA GPU
docker run -it --rm \
  --gpus all \
  treinfra/atrex-bench:2.10.0-cuda12.8-cudnn9-devel
```

## Usage

Everything below runs **inside the container** you started above (Python ≥ 3.10 and PyTorch
already ship in the image). The example uses the ROCm image; on the CUDA image the commands are
the same, except you skip `source /opt/venv/bin/activate` (it ships PyTorch in the system Python).

### 1 · Configure the agent CLI

Point Claude Code at your API key. Append this to `~/.bashrc` — fill in your own
`ANTHROPIC_API_KEY` (and set `ANTHROPIC_BASE_URL` only if you go through a gateway instead of
the official API):

```bash
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
export IS_SANDBOX=1
export ANTHROPIC_API_KEY="<YOUR_ANTHROPIC_API_KEY>"
# export ANTHROPIC_BASE_URL="<YOUR_API_BASE_URL>"   # optional: custom gateway
export ANTHROPIC_MODEL="<YOUR_MODEL>"
claude() {
  command claude --dangerously-skip-permissions --effort max "$@"
}
```

Reload it with `source ~/.bashrc`. (Prefer Codex? Export `OPENAI_API_KEY` instead and pass
`--cli codex` below.)

### 2 · Clone and run

Generation and evaluation run as **separate sessions** (e.g. separate pods). The generation
agent has full filesystem access, so its pod must never hold the answer files (`metadata.json`,
`roofline.json`, the evaluator) — otherwise it can just read the targets. So you clone the repo
and **strip those files in place** before generating; evaluation uses its own fresh, full clone.

#### Generate

`git clone` the repo, run the cleanup script, then generate:

```bash
git clone https://github.com/alibaba/atrex-bench.git && cd atrex-bench
source /opt/venv/bin/activate                    # activate the bundled venv (ROCm image)
python scripts/cleanup_for_generation.py        # strip the answer files in place
pip install -e . --no-deps --no-build-isolation  # deps already ship in the image

python scripts/run_generate.py \
  --operator attention_forward \
  --backend flydsl \
  --cli claude \
  --output-dir outputs/attention_forward_flydsl \
  --mirror-trace
```

`cleanup_for_generation.py` keeps only each operator's `reference.py` / `input.py` /
`shapes.json`, the `prompt/` templates, and the generation runner; it deletes
`metadata.json`, `roofline.json`, `configs/`, the evaluator, `tests/`, the README, and the
`.git` history, so provenance and SOL targets cannot leak. It rewrites the checkout **in
place** — copy out your generated kernel before evaluating, and never run cleanup in an
evaluation session.

- Backends: `triton`, `gluon`, `flydsl`, `cutedsl`. CLI: `--cli claude` (default) or `--cli codex`.
- Each run writes `generated_kernel.py` plus a `generation.json` bundle and a trace sidecar
  under the output directory.

##### Generate with Atrex Kernel Agent

[Atrex Kernel Agent](https://github.com/alibaba/atrex-kernel-agent) provides the
`gpu-kernel-optimizer` skill for profile-driven kernel implementation and optimization.
Follow its
[Interactive Skill installation guide](https://github.com/alibaba/atrex-kernel-agent#route-1-interactive-skill-skillmd)
before running the command below.

The default AKA installation base is `~/aka_kernel_opt`. The optimizer requires an explicit
target platform. Set the platform in the selected backend's optimizer prompt before generating:

```bash
TARGET_GPU="<TARGET_GPU_MODEL>"
sed -i "s/for the target GPU/for the ${TARGET_GPU} GPU/" \
  prompt/flydsl/generate_kernel_with_optimizer.md

CLAUDE_CONFIG_DIR="$HOME/aka_kernel_opt/.claude" \
python scripts/run_generate.py \
  --operator attention_forward \
  --backend flydsl \
  --template prompt/flydsl/generate_kernel_with_optimizer.md \
  --cli claude \
  --output-dir outputs/attention_forward_flydsl_agent \
  --mirror-trace \
  --skill
```

For Codex, set `CODEX_HOME="$HOME/aka_kernel_opt/.codex"` and use `--cli codex`. Ensure
that `$CODEX_HOME/config.toml` contains a top-level `model = "..."` entry.

The optimizer template explicitly invokes `gpu-kernel-optimizer`; `--skill` enables the
tools required by its workflow. Runs generated with `--skill` use a different agent workflow
and must not be compared directly with non-skill runs.

#### Evaluate

Evaluation needs the **full** repo (`metadata.json`, `roofline.json`, `configs/`, and the
evaluator), so run it in a **separate session** with a fresh clone you do _not_ strip:

```bash
git clone https://github.com/alibaba/atrex-bench.git && cd atrex-bench
source /opt/venv/bin/activate                    # activate the bundled venv (ROCm image)
pip install -e . --no-deps --no-build-isolation  # deps already ship in the image

python scripts/run_eval.py \
  --input path/to/attention_forward_flydsl/generated_kernel.py \
  --reference-dir data/attention_forward \
  --output results/attention_forward_flydsl \
  --num-correctness-cases 5 \
  --bench-iters 3
```

- `--input` — the candidate from the generate stage (a single file exposing `class Model`).
  Bring the `generated_kernel.py` you produced into this session and point `--input` at it.
- `--reference-dir` — an operator directory under `data/`.
- `--num-correctness-cases` / `--bench-iters` — correctness samples per shape (default 1) and
  timed perf iterations (default 100).
- The output directory archives every input artifact plus `eval_result.json`.

Measure the `torch.compile` baseline instead of a candidate:

```bash
python scripts/run_eval.py --torch-compile --reference-dir data/attention_forward --output results/torch_compile
```

## Repository Layout

```text
atrex-bench/
├── configs/            # hardware SKU profiles (roofline peaks)
├── data/               # one directory per operator (+ data/README.md)
├── prompt/             # generation-stage prompt templates
├── scripts/            # CLI entrypoints: generate, cleanup, eval, roofline, trace
├── src/atrex_bench/    # package: generation runner, evaluator, roofline
└── tests/              # unit + end-to-end tests
```

## Data Format

Each operator directory is self-describing:

```text
data/<operator>/
├── reference.py    # class Model(nn.Module) — definition only
├── input.py        # _make_inputs(**kwargs) -> dict[str, Tensor]
├── shapes.json     # shape spec keyed by id (init_kwargs + input_kwargs)
├── metadata.json   # id, dtype, input/output dtypes, origin    (hidden from agent)
└── roofline.json   # cached W / Q / SOL_time_ms per device     (hidden from agent)
```

See [`data/README.md`](data/README.md) for the full schema and the data-maintenance workflow.

## Operators

`data/` ships **30** operator directories, each following the same five-file contract above.
Most are trace-derived (`status: "trace_reference"`); the rest are curated.
`data/operator_importance.json` holds trace-based prioritization scores.

<details>
<summary>Full operator list</summary>
<table>
<thead>
<tr><th>Operator</th><th>id</th><th>dtype</th><th>Upstream</th><th>Status</th></tr>
</thead>
<tbody>
<tr><td><code>attention_forward</code></td><td><code>atrex_001</code></td><td><code>bf16</code></td><td><code>vllm.attention_forward_varlen</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>block_scaled_mm</code></td><td><code>atrex_002</code></td><td><code>fp8_e4m3</code></td><td><code>vllm.w8a8_triton_block_scaled_mm</code></td><td><code>curated</code></td></tr>
<tr><td><code>causal_conv1d</code></td><td><code>atrex_003</code></td><td><code>bf16</code></td><td><code>sglang.causal_conv1d_fn</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>chunk_delta_rule_output</code></td><td><code>atrex_004</code></td><td><code>bf16</code></td><td><code>sglang/vllm.chunk_fwd_o</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>chunk_gated_delta_rule_state</code></td><td><code>atrex_005</code></td><td><code>bf16</code></td><td><code>sglang/vllm.chunk_gated_delta_rule_fwd_h</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>fp8_blockscale_fused_moe</code></td><td><code>atrex_006</code></td><td><code>fp8_e4m3</code></td><td><code>aiter.fmoe_fp8_blockscale_g1u1</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>fp8_dynamic_per_token_quant</code></td><td><code>atrex_007</code></td><td><code>fp8_e4m3</code></td><td><code>rtp-llm.dynamic_per_token_scaled_quant</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>fused_add_rms_norm</code></td><td><code>atrex_008</code></td><td><code>bf16</code></td><td><code>vllm.fused_add_rms_norm</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>fused_moe</code></td><td><code>atrex_009</code></td><td><code>bf16</code></td><td><code>vllm.fused_experts</code></td><td><code>curated</code></td></tr>
<tr><td><code>fused_qk_rmsnorm</code></td><td><code>atrex_010</code></td><td><code>fp16</code></td><td><code>rtp-llm.fusedQkRmsNorm</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>fused_qkv_rope</code></td><td><code>atrex_011</code></td><td><code>fp16</code></td><td><code>rtp-llm.add_fusedQKV_bias_transpose_prefill_kernel</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>fused_rmsnorm_quant</code></td><td><code>atrex_012</code></td><td><code>fp8_e4m3</code></td><td><code>aiter.rmsnorm2d_fwd_with_add_dynamicquant</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>gated_delta_rule_update</code></td><td><code>atrex_013</code></td><td><code>bf16</code></td><td><code>sglang/vllm.fused_sigmoid_gating_delta_rule_update</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>gated_rms_norm</code></td><td><code>atrex_014</code></td><td><code>bf16</code></td><td><code>sglang.rms_norm_gated</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>l2_norm</code></td><td><code>atrex_015</code></td><td><code>bf16</code></td><td><code>vllm.l2norm_fwd</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>layer_norm</code></td><td><code>atrex_016</code></td><td><code>bf16</code></td><td><code>vllm.layer_norm</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>linear_sigmoid_mul</code></td><td><code>atrex_017</code></td><td><code>bf16</code></td><td><code>sglang.sgl_kernel.fused_linear_sigmoid_mul</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>mla_decode_attention</code></td><td><code>atrex_018</code></td><td><code>bf16</code></td><td><code>aiter.mla_decode_stage1_asm_fwd</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>moe_align_block_size</code></td><td><code>atrex_019</code></td><td><code>int32</code></td><td><code>vllm.moe_align_block_size</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>moe_count_and_sort</code></td><td><code>atrex_020</code></td><td><code>int32</code></td><td><code>vllm.moe_count_and_sort_expert_tokens</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>moe_sum_reduce</code></td><td><code>atrex_021</code></td><td><code>bf16</code></td><td><code>sglang.moe_sum_reduce_triton</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>moe_topk_gating_softmax</code></td><td><code>atrex_022</code></td><td><code>fp32</code></td><td><code>vllm.moe_topk_gating_softmax</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>mrope</code></td><td><code>atrex_023</code></td><td><code>bf16</code></td><td><code>vllm/sglang.triton_mrope</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>paged_attention_decode</code></td><td><code>atrex_024</code></td><td><code>bf16</code></td><td><code>rtp-llm.paged_attention_rocm</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>per_token_group_quant_fp8</code></td><td><code>atrex_025</code></td><td><code>fp8_e4m3</code></td><td><code>vllm.per_token_group_quant_fp8</code></td><td><code>curated</code></td></tr>
<tr><td><code>reshape_and_cache</code></td><td><code>atrex_026</code></td><td><code>bf16</code></td><td><code>vllm.reshape_and_cache_flash</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>rms_norm</code></td><td><code>atrex_027</code></td><td><code>bf16</code></td><td><code>vllm.rms_norm</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>silu_and_mul</code></td><td><code>atrex_028</code></td><td><code>bf16</code></td><td><code>vllm.vllm_silu_and_mul</code></td><td><code>trace_reference</code></td></tr>
<tr><td><code>topk_filter</code></td><td><code>atrex_029</code></td><td><code>fp32</code></td><td><code>vllm</code> / FlashInfer top-k masking</td><td><code>curated</code></td></tr>
<tr><td><code>unified_attention</code></td><td><code>atrex_030</code></td><td><code>bf16</code></td><td><code>vllm.unified_attention</code></td><td><code>curated</code></td></tr>
</tbody>
</table>
</details>

When adding or reshaping operator data, keep `shapes.json` and `roofline.json` aligned and
refresh `SOL_time_ms` through `scripts/roofline.py`.

## Citation

Please cite our [paper](https://arxiv.org/abs/2607.14541) if it is helpful to your research.

```bibtex
@misc{atrex2026,
  title         = {Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent},
  author        = {Lingyun Yang and Yuxiao Wang and Shenghao Liang and Linfeng Yang and Daocheng Ying and Chunbo You and Rui Zhang and Luping Wang and Yinghao Yu and Guodong Yang and Liping Zhang},
  year          = {2026},
  eprint        = {2607.14541},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2607.14541}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 Alibaba Group.
