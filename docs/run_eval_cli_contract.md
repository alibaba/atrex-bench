# `run_eval` CLI and Configuration Contract

## 1. Entry Points

The Gateway only needs to expose one option:

```bash
python scripts/run_eval.py --config /abs/path/run_eval.json
```

Existing public CLI options remain supported. Explicit CLI options take precedence over the config. Absolute paths are recommended for all path fields.

## 2. Configuration Fields

| Field | Type | Required when | Description |
|---|---|---|---|
| `schema_version` | string | Recommended | Currently fixed to `v1`. |
| `eval_mode` | enum | Optional | `candidate` or `torch_compile_reference`; defaults to `candidate`. |
| `validation_mode` | enum | Optional | `full`, `correctness_only`, or `performance_only`; defaults to `full`. |
| `input` | path | Required in candidate mode | Candidate Python file; must not be set in Torch compile mode. |
| `reference_dir` | path | Always | Reference directory. Must contain `reference.py`, `input.py`, `shapes.json`, and `metadata.json`. |
| `output` | path | Always | Root directory for evaluation artifacts. |
| `checkpoint_dir` | path | Optional | Root directory for correctness/performance checkpoints. |

The candidate file must expose `class Model`.

Except for mode switches and compatibility aliases, all public CLI options below can also be set in the config. Use snake_case field names without the leading `--`. For example, `--warmup-iters` maps to `warmup_iters`.

## 3. Evaluation Modes

| Mode | CLI options | Stages |
|---|---|---|
| Full | No only-mode option | Compile, correctness, and performance. |
| Correctness only | `--correctness-only` | Compile and correctness. |
| Performance only | `--performance-only` | Compile and performance. |
| Torch compile reference | `--torch-compile` | Performance of `torch.compile(reference Model)`. |

Constraints:

- `--correctness-only` and `--performance-only` are mutually exclusive.
- `--torch-compile` cannot be combined with `--input`, `--correctness-only`, or `--performance-only`.
- `eval_mode=torch_compile_reference` always runs performance evaluation. If `validation_mode` is explicitly set in the config, it must be `performance_only`.

## 4. Recommended General Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `--atol` | float | `0.01` | Absolute tolerance for correctness checks. |
| `--rtol` | float | `0.05` | Relative tolerance for correctness checks. |
| `--correctness-max-rel-l2` | float | Unset | When set, applies a global relative L2 threshold to floating-point outputs. |
| `--num-correctness-cases` | int | `1` | Number of correctness cases per shape. |
| `--warmup-iters` | int | `10` | Performance warmup budget. |
| `--bench-iters` | int | `100` | Performance benchmark budget. |
| `--candidate-timeout-s` | float | `60` | Timeout in seconds for candidate import, instantiation, and each correctness forward call; `<=0` disables it. |
| `--perf-timeout-s` | float | `600` | Timeout in seconds for the entire performance stage of each shape; `<=0` disables it. |
| `--trust-mode` | enum | `trusted` | Allowed values: `trusted`, `untrusted`. |
| `--skip-kernel-attribution` | bool flag | `false` | Skips kernel attribution and `flydsl_compute_ratio`. |

## 5. Advanced Performance Options

| Option | Type | Default | Applies when |
|---|---:|---:|---|
| `--benchmark-mode` | enum | `eager` | Allowed values: `eager`, `cuda_graph_replay`. |
| `--cuda-graph-cache-flush-mb` | int | `1024` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-atol` | float | `0.01` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-rtol` | float | `0.05` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-min-cosine` | float | Unset | `benchmark-mode=cuda_graph_replay`. When set, replaces floating-point allclose checks. |
| `--graph-max-rel-l2` | float | Unset | `benchmark-mode=cuda_graph_replay`. |

## 6. GPU Clock-Locking Options

### 6.1 Clock-Lock Modes

| Option | Values | Default | Description |
|---|---|---|---|
| `--clock-lock-mode` | `off`, `external`, `manage` | `off` | GPU clock policy. |

The config field is `clock_lock_mode`. The following CLI options are retained only as compatibility aliases:

- `--lock-clocks`: equivalent to `--clock-lock-mode manage`.
- `--require-clock-locked`: equivalent to external-marker mode.
- `--clock-locked`: records only a caller assertion; it does not lock or verify hardware clocks.

### 6.2 Managed-Mode Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `--clock-lock-device` | string | Resolved automatically | Physical GPU index or GPU UUID. Explicitly passing a UUID is recommended on multi-GPU systems. |
| `--gpu-clock-mhz` | int | None | Required target graphics clock frequency. |
| `--memory-clock-mhz` | int | Unset | Set only when the device supports runtime memory-clock locking. |
| `--clock-lock-tolerance-mhz` | int | `50` | Initial verification tolerance after setting the clocks. Must be a positive integer. |
| `--clock-lock-settle-seconds` | float | `3.0` | Time to wait for clocks to settle after applying the settings. |
| `--clock-lock-command-timeout-s` | float | `10.0` | Timeout for each `nvidia-smi` command. |
| `--clock-lock-monitor` / `--no-clock-lock-monitor` | bool | Enabled | Whether to monitor the entire evaluation window. |
| `--clock-lock-sample-interval-ms` | int | `10` | Monitoring sample interval. Must be a positive integer. |
| `--clock-lock-runtime-tolerance-mhz` | int | `0` | Clock tolerance during the evaluation window. Must be a non-negative integer. |
| `--clock-lock-fail-on-deviation` / `--no-clock-lock-fail-on-deviation` | bool | `true` | Whether a clock deviation from the target causes the evaluation to fail. |
| `--allow-busy-gpu` | bool flag | `false` | Allows existing compute processes on the target GPU; busy GPUs are rejected by default. |

Managed-mode constraints:

- Only the top-level `run_eval` process may manage clock locking.
- `--gpu-clock-mhz` must be a positive integer.
- On multi-GPU systems, `CUDA_VISIBLE_DEVICES` and `--clock-lock-device` must identify the same physical GPU.
- Managed-mode options cannot be used in `off` or `external` mode.
- `manage` cannot be combined with `--clock-locked` or `--require-clock-locked`.

## 7. Configuration Rules

The Gateway exposes only `--config`. The JSON config supports all public capabilities in Sections 2-6. Use the following canonical fields for mode switches and compatibility aliases:

| CLI capability | Config field |
|---|---|
| `--torch-compile` | `"eval_mode": "torch_compile_reference"` |
| `--correctness-only` | `"validation_mode": "correctness_only"` |
| `--performance-only` | `"validation_mode": "performance_only"` |
| `--lock-clocks` | `"clock_lock_mode": "manage"` |
| `--allow-busy-gpu` | `"clock_lock_require_idle": false` |

`config_version` defaults to `v1` and is written to `runner_config.config_version` in `eval_result.json`. It has a different meaning from the top-level `schema_version`.

Configuration precedence:

```text
Explicit CLI options > JSON config > Built-in defaults
```

Example: full candidate evaluation:

```json
{
  "schema_version": "v1",
  "eval_mode": "candidate",
  "validation_mode": "full",
  "input": "/abs/path/candidate.py",
  "reference_dir": "/abs/path/operator",
  "output": "/abs/path/results",
  "warmup_iters": 10,
  "bench_iters": 100,
  "benchmark_mode": "eager",
  "trust_mode": "trusted"
}
```

Example: performance-only candidate evaluation with managed GPU clock locking:

```json
{
  "schema_version": "v1",
  "eval_mode": "candidate",
  "validation_mode": "performance_only",
  "input": "/abs/path/candidate.py",
  "reference_dir": "/abs/path/operator",
  "output": "/abs/path/results",
  "clock_lock_mode": "manage",
  "clock_lock_device": "GPU-01234567-89ab-cdef-0123-456789abcdef",
  "gpu_clock_mhz": 2000,
  "clock_lock_monitor": true,
  "clock_lock_sample_interval_ms": 10,
  "clock_lock_runtime_tolerance_mhz": 5,
  "clock_lock_fail_on_deviation": false,
  "clock_lock_require_idle": true
}
```

## 8. Private Options: Do Not Expose

The following options are reserved for communication between `run_eval` parent and child processes:

- `--worker`
- `--torch-compile-worker`
- `--single-shape-worker`
- `--torch-compile-shape-worker`
- `--artifact-dir`
- `--checkpoint-root`
- `--shape-id`
- `--shape-result-output`

## 9. Standard Launch Commands

### 9.1 Standard Gateway Entry Point

```bash
python scripts/run_eval.py --config /abs/path/run_eval.json
```

Correctness-only, performance-only, full, Torch compile, and clock-locking modes are all controlled through JSON fields. No additional Gateway options are needed.

### 9.2 Torch Compile Configuration

```json
{
  "schema_version": "v1",
  "eval_mode": "torch_compile_reference",
  "validation_mode": "performance_only",
  "reference_dir": "/abs/path/operator",
  "output": "/abs/path/results",
  "warmup_iters": 10,
  "bench_iters": 100
}
```

### 9.3 Existing CLI Compatibility

```bash
python scripts/run_eval.py \
  --input /abs/path/candidate.py \
  --reference-dir /abs/path/operator \
  --output /abs/path/results \
  --performance-only
```

## 10. Exit Codes and Artifacts

| Exit code | Meaning |
|---:|---|
| `0` | All stages required by the selected mode passed. |
| `1` | Evaluation failure, runtime error, or configuration validation failure. |
| `2` | An argparse error involving argument syntax, required options, enum values, or mutually exclusive options. |

Standard output:

```text
[OUTPUT] <output>/<timestamp>/<operator>/eval_result.json
```

Main artifacts:

- `<output>/<timestamp>/<operator>/eval_result.json`
- `<output>/<timestamp>/<operator>/staging_manifest.json`
- `clock_lock.json` and `clock_lock_trace.csv` in managed clock-locking mode

Progress logs are written to stderr. Integrations must use both the process exit code and `eval_result.json` to determine the outcome.
