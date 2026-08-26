# SOL calculation and optional dispatch floor

The theoretical latency is `max(sum(W[d] / P_peak[d]), Q / B_peak)`.
An empty dtype mapping denotes memory-only work; zero compute and zero traffic
produce zero latency. Memory traffic alone is not an empty workload.

A hardware profile may explicitly set `launch_overhead_s` to a finite positive
number in seconds. Missing or `null` means no floor. For nonempty work, the
reported SOL becomes `max(theoretical_latency, launch_overhead_s)`;
`clamped_by_overhead` indicates that the floor raised the result. The bottleneck,
arithmetic intensity and roofline throughput describe the theoretical model,
not dispatch overhead. This optional latency-floor model is not a pure roofline
bound and must be identified when comparing results.

Only use a floor measured for the execution/timing mode being compared. In
particular, do not reuse an eager-dispatch measurement for CUDA Graph replay
without validation. No default floor or device-specific value is hardcoded.

The Python API and `scripts/roofline.py` use the same floor semantics, including
integer-only operators. JSON output includes the clamp flag; text labels a
clamped result. Existing `SOL_time_ms` caches are not automatically migrated.
Review with `--no-write` before explicitly recomputing an operator's cache.

The fused MoE bundle uses distinct top-k expert routes with normalized weights.
Its XPU-A production baselines are recorded AITER measurements, not measurements
performed by this migration. Reproduction requires matching hardware, framework
versions and timing conditions.
