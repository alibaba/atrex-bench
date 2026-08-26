# NVIDIA GPU Clock Locking in `run_eval`

## Modes

| Mode | Configuration | Behavior |
|---|---|---|
| Off | `--clock-lock-mode off` | Does not change GPU clocks. This is the default mode. |
| External | `--clock-lock-mode external` or `--require-clock-locked` | Requires `ATREX_BENCH_CLOCKS_LOCKED=1` or `SOL_EXECBENCH_CLOCKS_LOCKED=1`. Does not apply hardware settings or query the hardware for verification. |
| Managed | `--clock-lock-mode manage` or `--lock-clocks` | The parent process sets, verifies, holds, and restores the graphics clock of one NVIDIA GPU. It can also manage the memory clock if the device supports it. |

`--clock-locked` records only a caller assertion. It does not invoke `nvidia-smi` or establish hardware verification, and it cannot be combined with managed or external mode.

## Managed Clock Locking

First, query the GPU identity, current usage, and supported clock frequencies on the target machine:

```bash
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader
nvidia-smi -i GPU-01234567-89ab-cdef-0123-456789abcdef \
  --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader
nvidia-smi -i GPU-01234567-89ab-cdef-0123-456789abcdef \
  -q -d SUPPORTED_CLOCKS
```

`-lmc` must be a runtime memory-clock lock supported by the current driver. If the driver requires `--lock-memory-clocks-deferred` or reports `not supported`, do not set `--memory-clock-mhz`. The managed workflow still observes the memory clock, but does not claim that it is locked.

Run an evaluation with managed clock locking:

```bash
uv run python scripts/run_eval.py \
  --input candidate.py \
  --reference-dir data/example/op \
  --output artifacts \
  --lock-clocks \
  --clock-lock-device GPU-01234567-89ab-cdef-0123-456789abcdef \
  --gpu-clock-mhz 2000 \
  --clock-lock-runtime-tolerance-mhz 0
```

Managed mode requires an explicit positive integer for `--gpu-clock-mhz`. `--memory-clock-mhz` is an optional positive integer for devices that support runtime memory-clock locking. The current implementation does not infer clock frequencies from GPU names or provide default frequencies that apply across machines.

The device selector is resolved in this order:

1. `--clock-lock-device` or `clock_lock_device` in the JSON config.
2. `ATREX_BENCH_CLOCK_DEVICE`.
3. `CUDA_VISIBLE_DEVICES`, if it contains exactly one token.
4. `NVIDIA_VISIBLE_DEVICES`, if it contains exactly one token.
5. The physical index reported by `nvidia-smi`, if it reports exactly one device.

If multiple GPUs are visible and no unambiguous selector is available, the workflow fails before starting the worker.

**Warning: a numeric selector is a physical `nvidia-smi` index, not a runtime device ordinal.**
In environments that remap devices through `CUDA_VISIBLE_DEVICES` (such as containers, Ray,
and schedulers), `cuda:0` may refer to physical GPU 3. Passing `0` would instead lock
physical GPU 0, which may belong to another workload on a shared node. The resolved
device is cross-checked by UUID against the device used by the current process.
A mismatch is rejected before any `-lgc` command is issued. The recommended choices
are to **pass a GPU UUID** or **omit the selector** and let step 3 resolve it from
`CUDA_VISIBLE_DEVICES`, whose token identifies the physical device.

If the current process cannot provide a device identity (for example, CUDA is unavailable,
the runtime is ROCm, or the runtime does not expose a UUID), this cross-check is skipped.
The missing identity alone does not cause the evaluation to fail.

## Permissions and Idle-GPU Protection

A root process executes `nvidia-smi` with a fixed argument list. A non-root process executes `sudo -n nvidia-smi`, without prompting for a password. The target machine must grant the required non-interactive sudo permissions; otherwise, managed mode fails.

By default, the workflow:

- Rejects other compute processes on the target GPU before changing its clocks.
- Acquires a shared `flock` keyed by the device UUID to prevent concurrent Atrex processes on the same host from managing the same GPU.
- Rolls back only clock settings that were successfully applied.

`--allow-busy-gpu` skips only the compute-process check; it does not bypass the device-level `flock`. Use it only in a controlled environment where other processes are known not to affect the measurements or be affected by clock locking.

Managed mode requires the scheduler to guarantee exclusive GPU access. `flock` provides mutual exclusion only among processes on the same host that share the same visible lock directory. A Pod-local `/tmp` does not provide cross-Pod exclusion, and NFS locks are not used as distributed locks. Kubernetes deployments must use the device plugin, node scheduling, and resource quotas to ensure that each physical GPU is assigned to only one evaluation Pod.

## Applying, Verifying, and Restoring Clocks

The managed lifecycle follows this fixed sequence:

1. Resolve the device and acquire its device-level `flock`.
2. Check for compute processes on the target GPU.
3. Run `nvidia-smi -i <selector> -lgc <mhz>,<mhz>`.
4. If a memory frequency is configured, run `nvidia-smi -i <selector> -lmc <mhz>,<mhz>`.
5. Wait for the settling period.
6. Query `clocks.current.graphics` and `clocks.current.memory`.
7. Once all requested clocks are measured within the initial tolerance, start the `nvidia-smi --loop-ms=10` sampler.
8. Sample throughout the worker's entire lifecycle and write `clock_lock_trace.csv`.
9. After the worker exits, stop the sampler and verify the full clock window before resetting all clocks that were set.
10. Query the clocks after restoration and write the final report.

A clock-setting command is treated as failed if `nvidia-smi` reports `not supported` on stdout or stderr, even when its exit code is 0. In graphics-only mode, `requested.memory_mhz` is `null`; memory-clock snapshots are observations only.

Configurable options:

- `--clock-lock-tolerance-mhz`: defaults to `50`.
- `--clock-lock-settle-seconds`: defaults to `3.0`.
- `--clock-lock-command-timeout-s`: defaults to `10.0`.
- `--clock-lock-monitor` / `--no-clock-lock-monitor`: full-window monitoring is enabled by default in managed mode.
- `--clock-lock-sample-interval-ms`: defaults to `10`.
- `--clock-lock-runtime-tolerance-mhz`: defaults to `0`. In strict mode, any sample outside the target range causes the result to fail.
- `--clock-lock-fail-on-deviation` / `--no-clock-lock-fail-on-deviation`: defaults to failing on deviation. When disabled, an evaluation may continue only if sampling is complete and the sole deviation is graphics-clock downclocking, with no forbidden events. The result records `clock_locked=false`, `clock_lock_verified=false`, and `measurement_verified=false`.
- `--allow-busy-gpu`: disables the default rejection of existing compute processes.

SW power-cap events are diagnostic counters only when the measured graphics clock remains on target. HW slowdown, SW/HW thermal slowdown, and HW power-brake slowdown always invalidate measurement-window verification.

Normal completion, Python exceptions, worker timeouts, `SIGINT`, and `SIGTERM` all enter the restoration path. On interruption, the parent process first terminates the worker. If the bounded wait expires, it kills the worker before restoring GPU clocks.

`SIGKILL`, loss of node connectivity, host power failure, and forced destruction by the container runtime cannot be handled by this cleanup path. After such an event, an operator must consult the latest `clock_lock.json` and manually run `-rgc` for the same device. Run `-rmc` as well if the report includes a requested memory-clock setting.

## JSON Runner Configuration

```json
{
  "clock_lock_mode": "manage",
  "clock_lock_device": "GPU-01234567-89ab-cdef-0123-456789abcdef",
  "gpu_clock_mhz": 1500,
  "clock_lock_tolerance_mhz": 50,
  "clock_lock_settle_seconds": 3.0,
  "clock_lock_command_timeout_s": 10.0,
  "clock_lock_require_idle": true,
  "clock_lock_monitor": true,
  "clock_lock_sample_interval_ms": 10,
  "clock_lock_runtime_tolerance_mhz": 0,
  "clock_lock_fail_on_deviation": true
}
```

CLI options take precedence over the JSON config. Device, frequency, tolerance, settling-time, command-timeout, and busy-policy fields are allowed only in `manage` mode.

Devices that support runtime memory-clock locking can additionally set `"memory_clock_mhz": 3996`.

## External SOL-Compatible Launcher

```bash
SOL_EXECBENCH_CLOCKS_LOCKED=1 uv run python scripts/run_eval.py \
  --input candidate.py \
  --reference-dir data/example/op \
  --output artifacts \
  --require-clock-locked
```

External mode checks only that the marker is set. It does not set `environment.clock_lock_verified=true` in the result.

## Combining Clock Locking with Validation Modes

Managed clock locking supports all three candidate validation modes:

```bash
# compile + correctness
uv run python scripts/run_eval.py ... --correctness-only --lock-clocks --gpu-clock-mhz 2000

# compile + performance
uv run python scripts/run_eval.py ... --performance-only --lock-clocks --gpu-clock-mhz 2000

# compile + correctness + performance
uv run python scripts/run_eval.py ... --lock-clocks --gpu-clock-mhz 2000
```

`--correctness-only` and `--performance-only` are mutually exclusive. If neither is provided, the mode is `full`. The JSON config uses `"validation_mode": "full | correctness_only | performance_only"`. The clock-locking lifecycle covers the entire worker window for the selected mode. `correctness_only` does not produce performance samples or `flydsl_compute_ratio`; `performance_only` does not produce correctness cases.

## Artifacts and Failure States

Managed mode writes:

- `<output>/<timestamp>/<op>/clock_lock.json`: the latest request, device, measurement, restoration, and error evidence, updated after each lifecycle transition.
- `<output>/<timestamp>/<op>/clock_lock_trace.csv`: raw GPU clock, reason-bit, power, and temperature samples for the full worker window.
- `<output>/<timestamp>/<op>/eval_result.json`: the final clock-lock report is stored under `environment.clock_lock`.

Both JSON files are updated atomically using a temporary file in the same directory, `fsync`, and `os.replace`.

A failure during lock acquisition, the idle check, clock setting, initial verification, monitor startup, or restoration, or the occurrence of a forbidden event, prevents the worker from starting or causes the CLI to return a nonzero exit code. With `clock_lock_fail_on_deviation=false`, results with complete sampling and only graphics-clock downclocking retain correctness and performance data without a top-level `error`. `environment.clock_locked=false` and `measurement_verified=false` preserve the evidence.

## Automated Acceptance Tests

| Contract | Test |
|---|---|
| Validate the selector before launching a subprocess | `test_invalid_selector_is_rejected_before_subprocess` |
| Reject unsupported warnings even when the exit code is 0 | `test_zero_exit_unsupported_clock_command_is_rejected` |
| Do not modify or verify the memory clock in graphics-only mode | `test_managed_graphics_only_skips_memory_mutation_and_verification` |
| Reject busy GPUs by default and release the device lock | `test_managed_lock_rejects_busy_gpu` |
| Reset the graphics clock if memory-clock locking fails | `test_memory_lock_failure_resets_graphics` |
| Roll back both clocks when measured clocks exceed the tolerance | `test_clock_mismatch_resets_both_clocks` |
| Restore clocks after an exception in the evaluation body | `test_managed_lock_restores_after_body_exception` |
| Terminate or kill the worker when the parent is interrupted | `test_subprocess_interrupt_terminates_child` |
| Keep ownership of the managed lifecycle in the parent process | `test_managed_mode_runs_only_in_parent` |
| Do not launch the worker when lock acquisition fails | `test_acquisition_failure_does_not_launch_worker` |
| Preserve data and fail the result when restoration fails | `test_reset_failure_makes_payload_fail` |
| Verify a complete, stable trace successfully | `test_stable_samples_are_measurement_verified` |
| Fail the result on any runtime clock deviation in strict mode | `test_any_clock_deviation_invalidates_measurement` |
| Record pure downclocking and continue in non-strict mode | `test_measurement_downclock_is_recorded_without_failing_when_configured` |
| Stop the monitor before resetting clocks | `test_managed_monitor_covers_body_and_stops_before_clock_reset` |
| Reset clocks even if the monitor fails | `test_monitor_stop_failure_does_not_skip_clock_reset` |
