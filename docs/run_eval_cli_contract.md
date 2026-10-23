# `run_eval` CLI and Configuration Contract

## 1. Entry Points

CLI integrations only need to expose one option:

```bash
python scripts/run_eval.py --config /abs/path/run_eval.json
```

Existing public CLI options remain supported. Explicit CLI options take precedence over the config. Absolute paths are recommended for all path fields.

For both CLI and SDK calls, relative `input`, `reference_dir`, and `output` paths
are resolved against the caller's working directory before starting workers
(not against the config file or package directory). A relative `checkpoint_dir`
is instead based on the current run's artifact directory:
`<output>/<timestamp>/<kernel>/`. If omitted, that artifact directory is also the
checkpoint root. Absolute checkpoint paths are used as supplied.

Python callers can use `atrex_bench.evaluate(config)`. The SDK accepts the same fields as
`--config`, runs the evaluator in a separate subprocess, and returns the complete
`eval_result.json` object. Normal compile, correctness, or performance stage failures
are returned as results. SDK exceptions are raised only when the configuration cannot
start an evaluation, the subprocess fails abnormally, or the result artifacts are invalid.

## 2. Configuration Fields

| Field | Type | Required when | Description |
|---|---|---|---|
| `schema_version` | string | Recommended | Currently fixed to `v1`. |
| `eval_mode` | enum | Optional | `candidate`, `abba`, or `torch_compile_reference`; defaults to `candidate`, or `abba` when `baseline_input` is supplied. |
| `validation_mode` | enum | Optional | `full`, `correctness_only`, or `performance_only`; defaults to `full`. |
| `input` | path | Required in candidate and ABBA modes | Candidate Python file; must not be set in Torch compile mode. |
| `baseline_input` | path | Required in ABBA mode | Baseline A Python file. `input` is candidate B. Supplying `--baseline-input` selects ABBA mode. |
| `shape_ids` | list of strings, comma-separated string, or `@file` | Optional | Explicit subset of original `shapes.json` IDs. Omit to evaluate every shape. |
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
| ABBA comparison | `--baseline-input BASELINE --input CANDIDATE` | Four complete evaluations in A-B-B-A order. |
| Torch compile reference | `--torch-compile` | Performance of `torch.compile(reference Model)`. |

Constraints:

- `--correctness-only` and `--performance-only` are mutually exclusive.
- `--torch-compile` cannot be combined with `--input`, `--correctness-only`, or `--performance-only`.
- `eval_mode=torch_compile_reference` always runs performance evaluation. If `validation_mode` is explicitly set in the config, it must be `performance_only`.
- ABBA mode requires both `baseline_input` and `input`, and always uses `validation_mode=full`.
- All four ABBA runs execute under one top-level clock policy. Each run uses an isolated worker and artifact directory.

ABBA aggregation first takes the median of each shape's samples within one run,
then the geometric mean of the two A or B run medians. The aggregate latency is
the geometric mean across shapes. The result reports
`speedup = baseline_latency / candidate_latency` and
`improvement_pct = (speedup - 1) * 100` both per shape and overall. Any failed
run, invalid timing, incomplete schedule, or environment mismatch fails the ABBA
result; partial runs remain available as artifacts but are not aggregated.

### Explicit shape shards and offline merging

`--shape-ids 0,2` evaluates only those shapes. `--shape-ids @ids.json` reads a
JSON string array; a text file with one ID per line or comma-separated IDs also
works. The config/SDK form is `"shape_ids": ["0", "2"]`. Explicit CLI selection
overrides the config. Duplicate, unknown, or empty selections fail before
workers start. Selection follows the original string-sorted shape order, keeps
the IDs unchanged, and archives the complete, unchanged `shapes.json`.

Candidate full/correctness-only/performance-only, Torch compile, and ABBA all
support selection. Every ABBA shard executes its own complete A-B-B-A sequence
on its selected shapes. Module import/compilation still runs once per shard;
per-shape compilation, correctness, and timing run only for selected IDs. The
existing tolerance and timing settings and `sha256(stage|shape_id|case_index)`
RNG seed policy remain unchanged. Parallel shards need separate workers/GPUs;
running concurrent shards on one GPU would contaminate timing.

Every result has a `shape_selection` block containing `schema_version="v1"`,
the complete `all_shape_ids`, `selected_shape_ids`, `partial`, per-shape
`input_seeds`, the correctness case count, `seed_policy`, SHA-256 hashes of the
reference/candidate files (and ABBA baseline), `evaluator_version`, and a
fingerprint of the evaluator's Python source. Environment and runner settings
remain in the normal `environment` and `runner_config` blocks.

Merge disjoint shards with the public SDK function:

```python
from atrex_bench import evaluate, merge_evaluation_shards

common = {
    "input": "/abs/candidate.py",
    "reference_dir": "/abs/operator",
    "validation_mode": "full",
}
# Run these separately on independent GPU workers; collect the result dictionaries.
a = evaluate({**common, "output": "/abs/shard-a", "shape_ids": ["0", "2"]})
b = evaluate({**common, "output": "/abs/shard-b", "shape_ids": ["1", "3"]})
report = merge_evaluation_shards([a, b], output="/abs/merged_eval_result.json")
```

`merge_evaluation_shards` also accepts `eval_result.json` paths. It requires
exactly one result for every original shape and raises `ValueError` for missing
or duplicate shapes, mismatched source hashes, evaluator versions/fingerprints,
settings, seeds, GPU models, toolchain versions, or clock targets. GPU UUIDs,
indices, and individual clock traces may differ across workers. Failed stages
remain failed in the merged report. ABBA merging combines corresponding legs
and recomputes the overall geometric means from all shapes; it never averages
shard speedups. `shard_evaluations` preserves each source eval ID, selection,
and full environment evidence. The resulting `shape_selection.partial` is false.
Driver versions may differ across Ray nodes: each shard retains its exact
`environment.driver_version`; the merged value is always `null` because it has
no single host driver, including when all selected nodes share a version. This
keeps aggregate identity stable across independently scheduled ABBA repeats. GPU
model/architecture, evaluator, software runtime, inputs, timing configuration,
and clock targets must still match. Cross-node elapsed-time comparisons should
account for the recorded driver differences. Incompatibility errors identify
the differing report fields.

With a Gateway deployment that supports shape sharding and pins a compatible
Atrex-Bench wheel, ordinary eval automatically shards when the deployment
enables it:

```bash
agate eval --profile pre --gpu L20N --candidate candidate.py --reference-dir operator
```

Use `--shape-shards 1` for serial execution or `--shape-shards 2` to cap automatic
fanout at two shard workers. A manual partial run uses the same selection field:

```bash
agate eval --profile pre --gpu L20N --candidate candidate.py --reference-dir operator \
  --set 'shape_ids=["0","2"]' --shape-shards 1
```

For manual shards, download each native `eval_result.json` artifact and pass the
paths to `merge_evaluation_shards`. The files must cover the entire original
shape set. A partial report alone is not a full-operator result. Deployments can
check `atrex_bench.SHAPE_SHARD_SCHEMA_VERSION == "v1"` and the presence of
`merge_evaluation_shards` before accepting manual shards. Automatic Ray Data
distribution also requires the capability described below.

### Automatic Ray Data distribution

See the rendered [architecture and failure-path diagrams](ray_data_architecture.md)
for component interactions and their implementation entry points.

The installed CLI owns automatic shape splitting, GPU worker placement, and
merging. The wrapper `scripts/run_eval.py` and the unchanged `evaluate(config)`
SDK invoke this same CLI. Local execution remains the default. Install the
optional `atrex-bench[ray]` dependency in the driver and GPU worker image, and
run the driver inside an existing Ray cluster:

```bash
python scripts/run_eval.py --distribution ray_data --ray-max-workers 4 \
  --ray-shapes-per-shard 1 --input candidate.py --reference-dir operator \
  --output results/distributed
```

The equivalent SDK config is:

```python
from atrex_bench import evaluate

report = evaluate({
    "distribution": "ray_data",
    "ray_max_workers": 4,
    "ray_shapes_per_shard": 1,
    "input": "/abs/candidate.py",
    "reference_dir": "/abs/operator",
    "output": "/abs/results",
})
```

| Config field | Default | Meaning |
|---|---|---|
| `distribution` | `local` | `local` or `ray_data`. |
| `ray_max_workers` | shard count | Initial pool of one-GPU actors; an explicit positive integer caps it at the shard count. |
| `ray_shapes_per_shard` | `1` | Positive number of consecutive original shape IDs per dataset row. |
| `ray_max_shard_retries` | `3` | Worker/node failure retries per shard, integer 0–3. The CLI also accepts `--ray-max-shard-retries`. |
| `ray_resources` | `{}` | Ray custom resource requirements for GPU workers; positive numbers, excluding `CPU` and `GPU`. |
| `ray_node_selector` | `{}` | Ray `label_selector` mapping for GPU workers; the zero-GPU registry stays on the driver node. |
| `ray_namespace` | Generated | Optional standalone Ray namespace. Gateway `AGATE_RAY_NAMESPACE` overrides this field to isolate attempts. |

The driver connects with `ray.init(address="auto")` and needs no GPU. Dataset
rows carry stable shard IDs and the original string-sorted shape IDs. The
actor pool starts at `min(ray_max_workers, shard_count)`, reserves one GPU per
worker, and allows one task in flight per actor. Each row calls `evaluate` with
`distribution="local"` in fresh subprocesses. A worker can process multiple
shards sequentially; model state and evaluator subprocesses are not reused.
Candidate full/correctness-only/performance-only, Torch compile, and ABBA use
the same local harness. ABBA executes A-B-B-A within each shard and merges
corresponding legs before computing global speedups. Managed clock locking
runs on the actor's assigned GPU and covers that shard's whole local run.

Before creating workers, the driver snapshots the full reference directory,
candidate/baseline files, adjacent Python/text/CUDA sources, adjacent Python
packages, and the complete evaluator Python package. Every actor validates the
snapshot hashes and writes it into its own temporary directory. This also works
when the benchmark wheel is installed only in the driver: the actor bootstrap
is serialized by value and imports the SDK after materializing the package.
Native Python dependencies, Torch, CUDA/ROCm, and DSL toolchains must already be
installed in compatible driver/worker images. Adjacent `requirements.txt` is
transported as evidence; workers do not install it. Driver-local dependency
directories and arbitrary native site-packages are not transported. Workers replace
all inherited `PYTHONPATH` entries (including image-provided entries) with the
transported source paths; install native dependencies in the worker interpreter's
`site-packages`. Ray Data
requires `ray[data]>=2.54,<2.57`, including its pandas/PyArrow dependencies.

Automatic distribution requires the complete shape set. Explicit `shape_ids`
uses `distribution="local"` and the manual merge API. Omit
`clock_lock_device`: Ray assigns each actor's visible GPU. A relative
`checkpoint_dir` remains worker-local; absolute driver checkpoint paths are
rejected. The driver persists one final `eval_result.json` and the normal SDK
output pointer. Per-shard files and checkpoints remain temporary worker files;
their input seeds and environment evidence are retained in the native report.

Missing, duplicate, or incompatible shard results fail the distributed run.
Normal compile/correctness/performance failures remain in the merged report
and produce the usual failed verdict without retry. Ray system failures retry the
affected map task up to `ray_max_shard_retries`; user exceptions are not retried.
A CPU registry on the driver node commits each completed shard before its worker
returns, so even replay of a lost result or bundled task reuses completed reports
without running their harness again. Registry attempt counts enforce the shard
limit independently of map-task replay. Actor restarts are also finite, bounded by
shard count times the retry limit, and never increase the admitted worker pool.
Restarted workers wait for the previous subprocess cleanup grace period before
starting evaluation. A driver/registry loss or exhausted retry limit fails the run;
there is no cross-driver recovery or automatic replay of the whole evaluation.
`distribution.max_shard_retries` and `distribution.shard_attempts` retain each
started attempt, placement and committed outcome. The final native report
has full `shape_selection` coverage and a `distribution` block with `mode`,
logical `max_workers`, `shard_count`, source snapshot hash, namespace, driver
job ID, and worker actor IDs. The capability marker is
`atrex_bench.ray_data.RAY_DATA_DISTRIBUTION_SCHEMA_VERSION == "v1"`.

The same block retains `driver` process evidence: `job_id`, `node_id`, `pid`,
`assigned_gpu_ids`, `cuda_visible_devices`, and `python_executable`. Each entry
in `shards`, ordered by the stable shard ID, adds `actor_id`, `shard_id`,
`shape_ids`, `start_utc_ns`, `end_utc_ns`, and monotonic `duration_s`. These values
come from Ray and the process without importing Torch or querying a GPU. A
driver's empty Ray GPU-ID list alone does not establish its reservation or GPU
visibility; the caller must also provide the zero-GPU allocation and empty mask.
Shard intervals cover the local evaluation and result serialization, including
CPU setup. They support checking actual GPU-worker evaluation overlap and wall
time alongside the per-shape GPU samples; they are not kernel-busy telemetry or
a speedup guarantee. UTC interval comparison uses the worker clocks, while each
duration uses its own monotonic clock.

Workers and the registry belong to the driver job and are not detached. On
success or failure the driver kills and confirms registered actors before
disconnecting. Every evaluator subprocess has a POSIX lifetime supervisor:
actor death closes its pipe, then signals escalate through INT and TERM to
KILL within about four seconds; a separate watchdog also kills surviving
members of the subprocess group when the direct process exits. The report's
`worker_exit_grace_s=8` tells an external scheduler to retain GPU reservations
after Ray children become terminal. A separate janitor removes each actor's
private source/checkpoint directory six seconds after actor death, including
when a force kill bypasses Python finalizers. A hard driver stop alone is not evidence
of GPU cleanup: the scheduler must verify all namespace actors/tasks are
terminal and apply that grace interval. Detached candidate sessions and
uninterruptible kernel states are outside this process cleanup contract.

With a compatible Gateway deployment, ordinary `agate eval` can select this
mode automatically. `--shape-shards 1` selects serial execution and
`--shape-shards N` caps the worker budget. Deployment policy and runtime image
validation must enable it first; no parallel speedup is implied by local tests.

## 4. Recommended General Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `--atol` | float | `0.01` | Absolute tolerance for correctness checks. |
| `--rtol` | float | `0.05` | Relative tolerance for correctness checks. |
| `--correctness-max-rel-l2` | float | Unset | When set, applies a global relative L2 threshold to floating-point outputs. |
| `--num-correctness-cases` | int | `1` | Number of correctness cases per shape. |
| `--warmup-iters` | int | `10` | Performance warmup budget. **In `eager` mode, the unit is milliseconds, not iterations** (the `warmup` argument to Triton's `do_bench`, documented as "Warmup time (in ms)"). In `cuda_graph_replay` mode, it is the number of replays. |
| `--bench-iters` | int | `100` | Performance benchmark budget. **In `eager` mode, the unit is milliseconds, not iterations** (the `rep` argument to `do_bench`, documented as "Repetition time (in ms)"). Thus, `--bench-iters 100` requests approximately 100 ms of measurement: a fast kernel may run thousands of times, while a slow kernel may run only once. The length of `samples` in `eval_result.json` gives the recorded sample count. In `cuda_graph_replay` mode, this option is the number of replays. The option name predates this distinction and is retained for compatibility. |
| `--candidate-timeout-s` | float | `60` | Timeout in seconds for candidate import, instantiation, and each correctness forward call; `<=0` disables it. |
| `--perf-timeout-s` | float | `600` | Timeout in seconds for the entire performance stage of each shape. In `torch_compile_reference` mode, it contributes to the wall-clock limit of each shape worker. `<=0` sets this budget to zero. |
| `--compile-timeout-s` | float | `300` | Independent wall-clock limit for the candidate compilation stage. If compilation does not finish within the budget, the entire **process group** receives SIGKILL so that compiler processes spawned by the worker cannot retain locks after being reparented. This budget also contributes to the overall worker limit; in `torch_compile_reference` mode, it contributes to each shape worker's wall-clock limit. |
| `--trust-mode` | enum | `trusted` | Allowed values: `trusted`, `untrusted`. |
| `--skip-kernel-attribution` | bool flag | `false` | Skips kernel attribution and `flydsl_compute_ratio`. |

On POSIX, each worker runs in its own process group. Timeout handling kills the
whole group, including compiler descendants; the deadline also covers inherited
stdout/stderr pipes after the worker exits. A supervising worker gets up to one
second to clean up its separately grouped shape worker before SIGKILL, followed
by bounded process/pipe cleanup. This is not a sandbox: descendants that explicitly
detach into another session are outside the original process group.

### Environment Metadata

The top-level candidate and Torch compile supervisors collect only CPU-side
environment metadata when preparing fallback reports. They do not query devices,
drivers, or PTL state: these queries can initialize an extra CUDA context and
interfere with profiler counters in nested workers, including PPU ACU captures.
The provisional `gpu_name`, `gpu_arch`, `accelerator_backend`, `driver_version`,
and `PTL_STATE` fields remain `null` if no worker report is available. Package and
runtime versions, Python version, and external clock-lock markers remain available.
When a worker writes a report, its complete environment metadata is preserved,
including on failure. Managed clock locking still performs its existing device
identity and clock checks.

## 5. Advanced Performance Options

| Option | Type | Default | Applies when |
|---|---:|---:|---|
| `--benchmark-mode` | enum | `eager` | Allowed values: `eager`, `cuda_graph_replay`. |
| `--cuda-graph-cache-flush-mb` | int | `1024` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-atol` | float | `0.01` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-rtol` | float | `0.05` | `benchmark-mode=cuda_graph_replay`. |
| `--graph-min-cosine` | float | Unset | `benchmark-mode=cuda_graph_replay`. When set, replaces floating-point allclose checks. |
| `--graph-max-rel-l2` | float | Unset | `benchmark-mode=cuda_graph_replay`. |

Unless `--skip-kernel-attribution` is set, CUDA Graph mode profiles a separate
loop of replays of the captured graph. Capture, eager warmup, and cache flushing
are excluded from this loop, and profiler overhead does not affect the recorded
end-to-end samples. Kernel times are summed across all launches of each symbol
per forward (or graph replay), not averaged per kernel launch.

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
| `--baseline-input PATH` | `"eval_mode": "abba", "baseline_input": "PATH"` |
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

Example: ABBA comparison:

```json
{
  "schema_version": "v1",
  "eval_mode": "abba",
  "validation_mode": "full",
  "baseline_input": "/abs/path/baseline.py",
  "input": "/abs/path/candidate.py",
  "reference_dir": "/abs/path/operator",
  "output": "/abs/path/results",
  "warmup_iters": 10,
  "bench_iters": 100
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
- `--sdk-result-path-output`

`--sdk-result-path-output` is used only to pass the absolute path of the final
`eval_result.json` back to the SDK parent process. It is not a public option for
the Gateway or regular users.

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
  "bench_iters": 100,
  "compile_timeout_s": 300,
  "perf_timeout_s": 600
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
- `<output>/<timestamp>/<operator>/abba_runs/{00-baseline,01-candidate,02-candidate,03-baseline}/eval_result.json` in ABBA mode
- `clock_lock.json` and `clock_lock_trace.csv` in managed clock-locking mode

Progress logs are written to stderr. Integrations must use both the process exit code and `eval_result.json` to determine the outcome.
