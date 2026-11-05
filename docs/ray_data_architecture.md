# Ray Data execution and failure handling

Ray distribution is opt-in (`distribution="ray_data"`). The caller provides an
existing Ray cluster, compatible native dependencies, and a GPU worker budget.
Selected shapes become dataset rows, not independent Ray jobs. With N selected
shapes and a worker limit K, all rows are retained while at most K single-GPU
actors run them. Original IDs and the full reference bundle are preserved.
Local evaluation remains the default and is also the rollback path.

![Execution architecture: CPU driver, bounded GPU pool, result registry and process cleanup](diagrams/ray-data-execution.png)

## Shard outcomes

A normal candidate compile, correctness, or performance failure is a reportable
outcome, not an infrastructure retry. Raised user exceptions fail the distributed
run. Ray worker/node failures may replay a map task; the registry checks each
shard before running its harness, reuses committed results, caps attempts, and
rejects stale completions. Default three retries means four harness attempts.
A further transport delivery records exhaustion without running the harness.
Temporary recovery RPC failures may occur before the registry admits an attempt;
Ray transport retries are bounded separately at `2 * (max_retries + 1)` while
registry admission still permits only `max_retries + 1` harness calls. Actor restart
limits use the finite transport budget, and persistent transport loss remains fatal.
Exhausted shards produce explicit `shape_selection.failed_shapes`; successful
shards and remaining work are retained. All-shard failure or incomplete terminal
evidence still fails the run. There is no durable checkpoint or whole-driver recovery.

Completed and application-failed attempts record worker UTC endpoints and monotonic
duration. Lost workers cannot report their exact stop time: the registry records an
observed endpoint with `end_time_source="driver_observed_loss"` and
`end_time_estimated=true`. This is an upper-bound observation, not measured GPU time;
billing consumers must not sum it as exact usage.

![Shard retries, exhausted-shard partial results, fatal errors and cleanup outcomes](diagrams/ray-data-failures.png)

If recording an application failure in the registry fails, the application exception
remains primary and a separate `[atrex-ray-registry]` diagnostic is written to stderr.

When cleanup itself fails, an existing evaluation or initialization exception
remains the primary exception and a separate `[atrex-ray-cleanup]` diagnostic is
written to stderr. Without an earlier failure, cleanup failure raises instead of
returning a successful report. A hard driver kill cannot execute Python `finally`;
actor/process lifetime guards and caller-side terminal-state checks remain needed.
This is process lifecycle management, not isolation from hostile candidate code.

## Mapping the diagrams to code

| Diagram component | Implementation |
|---|---|
| Driver setup, dataset, bounded pool, cleanup scope | [`evaluate_ray_data`](../src/atrex_bench/ray_data.py) |
| Attempts, cached results, exhaustion tombstones, application failures and stale-attempt fencing | [`_ActorRegistry.register` / `begin` / `commit` / `fail`](../src/atrex_bench/ray_data.py) |
| Source bootstrap and fresh local evaluation | [`_ShardEvaluator.__init__` / `__call__`](../src/atrex_bench/ray_data.py) |
| Actor termination confirmation | [`_drain_actors`, `_terminate_registry`, `_wait_actor_terminated`](../src/atrex_bench/ray_data.py) |
| Lifetime pipe, signal propagation and descendant watchdog | [`supervised_popen` / `main`](../src/atrex_bench/_process_guard.py) |
| Local CLI and isolated subworker stages | [`run_eval.py`](../src/atrex_bench/cli/run_eval.py) |
| Requested-selection coverage with explicit failure evidence and compatible report validation | [`merge_evaluation_shards`](../src/atrex_bench/shards.py) |

The registry is unnamed and passed by actor handle; sharing a Ray namespace does
not require claiming a fixed actor name. Independent drivers still have separate
registries. This does not add support for concurrent evaluations mutating the
same Python process's Ray/DataContext state.

Workers replace **all** inherited `PYTHONPATH` entries, including image-provided
entries, with transported Python source paths. Ray cannot identify which entries
refer to driver-local directories. Native dependencies must be installed in the
worker interpreter's `site-packages`; arbitrary driver packages are not shipped.
The source snapshot does not make incompatible GPU images interchangeable.

See the [CLI/runtime contract](run_eval_cli_contract.md#automatic-ray-data-distribution)
for configuration, report fields, ABBA behavior and cleanup limits. Diagram sources
are [execution.dot](diagrams/ray-data-execution.dot) and
[failures.dot](diagrams/ray-data-failures.dot). Regenerate with Graphviz, for example:

```bash
dot -Tpng -Gdpi=110 docs/diagrams/ray-data-execution.dot -o docs/diagrams/ray-data-execution.png
dot -Tpng -Gdpi=110 docs/diagrams/ray-data-failures.dot -o docs/diagrams/ray-data-failures.png
```
