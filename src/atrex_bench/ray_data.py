"""Optional Ray Data distribution; the local evaluator remains the GPU harness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

RAY_DATA_DISTRIBUTION_SCHEMA_VERSION = "v1"
_DIRECTORY_CLEANUP_DELAY_S = 6

_DISTRIBUTION_KEYS = {
    "distribution",
    "ray_max_workers",
    "ray_shapes_per_shard",
    "ray_resources",
    "ray_node_selector",
    "ray_namespace",
}
_ADJACENT_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".cu",
    ".cuh",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
}


def validate_distribution(config):
    """Validate execution options without importing or connecting to Ray."""
    from atrex_bench import AtrexConfigError

    mode = config.get("distribution", "local")
    if mode not in {"local", "ray_data"}:
        raise AtrexConfigError("distribution must be local or ray_data")
    for key, default, maximum in (("ray_max_workers", 4, 31), ("ray_shapes_per_shard", 1, None)):
        value = config.get(key, default)
        if type(value) is not int or value < 1 or (maximum and value > maximum):
            raise AtrexConfigError(
                f"{key} must be an integer between 1 and {maximum or 'infinity'}"
            )
    resources = config.get("ray_resources", {})
    selector = config.get("ray_node_selector", {})
    if not isinstance(resources, dict) or any(
        not isinstance(k, str)
        or not k
        or isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not 0 < v < float("inf")
        or k in {"CPU", "GPU"}
        for k, v in resources.items()
    ):
        raise AtrexConfigError("ray_resources must map custom resource names to positive numbers")
    if not isinstance(selector, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in selector.items()
    ):
        raise AtrexConfigError("ray_node_selector must map label names to non-empty strings")
    namespace = config.get("ray_namespace")
    if namespace is not None and (not isinstance(namespace, str) or not namespace.strip()):
        raise AtrexConfigError("ray_namespace must be a non-empty string")
    if mode == "ray_data" and os.name != "posix":
        raise AtrexConfigError("ray_data distribution requires POSIX workers")
    if mode == "ray_data" and config.get("shape_ids") is not None:
        raise AtrexConfigError(
            "ray_data requires the full shape set; use distribution=local for subsets"
        )
    if mode == "ray_data" and config.get("clock_lock_device") is not None:
        raise AtrexConfigError("ray_data selects each Ray-assigned GPU; omit clock_lock_device")
    if mode == "ray_data" and config.get("checkpoint_dir") is not None:
        if Path(config["checkpoint_dir"]).is_absolute():
            raise AtrexConfigError("ray_data requires a relative worker-local checkpoint_dir")
    return mode


def _check_ray_version(ray):
    if tuple(int(part) for part in ray.__version__.split(".")[:2]) not in {
        (2, 54),
        (2, 55),
        (2, 56),
    }:
        raise ValueError("ray_data requires ray[data]>=2.54,<2.57 on driver and workers")


def _snapshot_sources(config):
    """Snapshot declared sources and the evaluator; do not ship native site-packages."""
    files, paths, python_dirs = {}, {}, []

    def add(source, destination):
        files[destination] = source.read_bytes()

    reference = Path(config["reference_dir"]).resolve()
    reference_dest = f"reference/{reference.name}"
    paths["reference_dir"] = reference_dest
    python_dirs.append(reference_dest)
    for source in reference.rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts:
            add(source, f"{reference_dest}/{source.relative_to(reference).as_posix()}")
    for role in ("input", "baseline_input"):
        if config.get(role) is None:
            continue
        source = Path(config[role]).resolve()
        root = role
        paths[role] = f"{root}/{source.name}"
        python_dirs.append(root)
        for adjacent in source.parent.iterdir():
            if adjacent.is_file() and adjacent.suffix in _ADJACENT_SUFFIXES:
                add(adjacent, f"{root}/{adjacent.name}")
            elif adjacent.is_dir() and (adjacent / "__init__.py").is_file():
                for item in adjacent.rglob("*"):
                    if item.is_file() and "__pycache__" not in item.parts:
                        add(item, f"{root}/{item.relative_to(source.parent).as_posix()}")
    package = Path(__file__).parent
    for source in package.rglob("*.py"):
        add(source, f"runtime/atrex_bench/{source.relative_to(package).as_posix()}")
    try:
        distribution = importlib.metadata.distribution("atrex-bench")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        for item in distribution.files or ():
            if ".dist-info/" in str(item):
                add(Path(distribution.locate_file(item)), f"runtime/{item}")
    return {
        "files": files,
        "paths": paths,
        "python_dirs": python_dirs,
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }


class _ActorRegistry:
    def __init__(self):
        self.handles = []

    def register(self, handle):
        self.handles.append(handle)

    def workers(self):
        return self.handles


def _runtime_diagnostics(ray, *, actor=False):
    """Record Ray placement and process values without entering a GPU runtime."""
    context = ray.get_runtime_context()
    result = {
        "job_id": str(context.get_job_id()),
        "node_id": str(context.get_node_id()),
        "pid": os.getpid(),
        "assigned_gpu_ids": [str(value) for value in ray.get_gpu_ids()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_executable": sys.executable,
    }
    if actor:
        result["actor_id"] = str(context.get_actor_id())
    return result


def _start_directory_janitor(root, read_fd):
    # Ray force-kills actors, so their Python finalizers cannot remove private
    # sources/checkpoints. The janitor receives only the read end of the actor
    # lifetime pipe and no GPU context. Allow group supervisors to finish first.
    code = """
import os, shutil, sys, time
while os.read(int(sys.argv[1]), 1): pass
time.sleep(float(sys.argv[3]))
shutil.rmtree(sys.argv[2], ignore_errors=True)
"""
    return subprocess.Popen(
        [sys.executable, "-c", code, str(read_fd), str(root), str(_DIRECTORY_CLEANUP_DELAY_S)],
        pass_fds=(read_fd,),
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class _ShardEvaluator:
    """One GPU per actor; every call still evaluates in fresh subprocesses."""

    def __init__(self, bundle, config, registry):
        import ray

        _check_ray_version(ray)
        ray.get(registry.register.remote(ray.get_runtime_context().current_actor))
        self.directory = tempfile.TemporaryDirectory(prefix="atrex-bench-ray-")
        self.root = Path(self.directory.name)
        self.read_fd, self.write_fd = os.pipe()
        self.janitor = _start_directory_janitor(self.root, self.read_fd)
        for name, data in bundle["files"].items():
            if hashlib.sha256(data).hexdigest() != bundle["sha256"][name]:
                raise ValueError("Ray source snapshot hash mismatch")
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        python_paths = [
            str(self.root / "runtime"),
            *[str(self.root / item) for item in bundle["python_dirs"]],
        ]
        sys.path[:0] = python_paths
        # Only transported Python paths and the worker's image environment are
        # valid here. The driver's per-job deps/PYTHONPATH paths are node-local.
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)
        self.config = {key: value for key, value in config.items() if key not in _DISTRIBUTION_KEYS}
        self.config.update({key: str(self.root / value) for key, value in bundle["paths"].items()})
        self.config["distribution"] = "local"

    def __del__(self):
        for name in ("write_fd", "read_fd"):
            descriptor = getattr(self, name, None)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, name, None)

    def __call__(self, batch):
        import ray

        from atrex_bench import evaluate, sdk
        from atrex_bench._process_guard import LIFETIME_FD_ENV, supervised_popen

        os.environ[LIFETIME_FD_ENV] = str(self.read_fd)
        original_start = sdk._start_process

        def start(command):
            return supervised_popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
            )

        sdk._start_process = start
        try:
            results, executions = [], []
            for shard_id, selected in zip(batch["shard_id"], batch["shape_ids_json"]):
                config = {
                    **self.config,
                    "shape_ids": json.loads(selected),
                    "output": str(self.root / "output" / str(shard_id)),
                }
                execution = {
                    **_runtime_diagnostics(ray, actor=True),
                    "shard_id": str(shard_id),
                    "shape_ids": config["shape_ids"],
                    "start_utc_ns": time.time_ns(),
                }
                started = time.perf_counter_ns()
                results.append(json.dumps(evaluate(config), allow_nan=False))
                execution.update(
                    end_utc_ns=time.time_ns(),
                    duration_s=(time.perf_counter_ns() - started) / 1_000_000_000,
                )
                executions.append(json.dumps(execution, allow_nan=False))
            return {
                "shard_id": batch["shard_id"],
                "result_json": results,
                "execution_json": executions,
            }
        finally:
            sdk._start_process = original_start
            os.environ.pop(LIFETIME_FD_ENV, None)


def _drain_actors(ray, registry):
    handles = ray.get(registry.workers.remote())
    actor_ids = [handle._actor_id.hex() for handle in handles]
    for handle in handles:
        ray.kill(handle, no_restart=True)
    for handle in handles:
        try:
            ray.get(handle.__ray_ready__.remote(), timeout=10)
        except ray.exceptions.RayActorError:
            pass
        else:
            raise RuntimeError("Ray Data worker did not terminate")
    return actor_ids


def _terminate_registry(ray, registry):
    ray.kill(registry, no_restart=True)
    try:
        ray.get(registry.__ray_ready__.remote(), timeout=10)
    except ray.exceptions.RayActorError:
        return
    raise RuntimeError("Ray worker registry did not terminate")


def evaluate_ray_data(config):
    """Distribute full-shape local evaluations and return a strict native report."""
    from atrex_bench import AtrexConfigError, AtrexEvaluationError, merge_evaluation_shards
    from atrex_bench._process_guard import WORKER_EXIT_GRACE_S
    from atrex_bench.shards import select_shape_ids

    validate_distribution(config)
    try:
        import ray
        import ray.data
    except ImportError as error:
        raise AtrexConfigError(
            "ray_data requires ray[data]>=2.54,<2.57 on driver and workers"
        ) from error
    try:
        _check_ray_version(ray)
    except ValueError as error:
        raise AtrexConfigError(str(error)) from error
    all_ids = sorted(json.loads((Path(config["reference_dir"]) / "shapes.json").read_text()))
    select_shape_ids(all_ids, all_ids)
    size = config.get("ray_shapes_per_shard", 1)
    shards = [all_ids[index : index + size] for index in range(0, len(all_ids), size)]
    if not shards:
        raise AtrexConfigError("ray_data requires at least one shape")
    maximum = min(config.get("ray_max_workers", 4), len(shards))
    namespace = (
        os.environ.get("AGATE_RAY_NAMESPACE")
        or config.get("ray_namespace")
        or f"atrex-bench-{uuid.uuid4().hex}"
    )
    ray.init(address="auto", namespace=namespace, log_to_driver=False)
    resources = config.get("ray_resources", {})
    placement = {"resources": resources}
    if config.get("ray_node_selector"):
        placement["label_selector"] = config["ray_node_selector"]
    registry = None
    actor_ids = []
    # The evaluator wheel may exist only in the driver's deps. Serialize the
    # small adapter by value; its actors materialize the immutable Python
    # evaluator snapshot before importing the local SDK.
    already_registered = __name__ in ray.cloudpickle.list_registry_pickle_by_value()
    if not already_registered:
        ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    try:
        driver = _runtime_diagnostics(ray)
        registry = ray.remote(num_cpus=0)(_ActorRegistry).options(**placement).remote()
        bundle = _snapshot_sources(config)
        rows = [
            {"shard_id": f"shape-{index:06d}", "shape_ids_json": json.dumps(ids)}
            for index, ids in enumerate(shards)
        ]
        dataset = ray.data.from_items(rows, override_num_blocks=len(rows)).map_batches(
            _ShardEvaluator,
            batch_size=1,
            num_gpus=1,
            compute=ray.data.ActorPoolStrategy(
                min_size=maximum, max_size=maximum, max_tasks_in_flight_per_actor=1
            ),
            fn_constructor_args=(bundle, config, registry),
            max_restarts=0,
            max_task_retries=0,
            **placement,
        )
        results = dataset.take_all()
        if {row["shard_id"] for row in results} != {row["shard_id"] for row in rows}:
            raise AtrexEvaluationError("Missing or unknown Ray shard results")
        if len(results) != len(rows):
            raise AtrexEvaluationError("Duplicate Ray shard results")
        results.sort(key=lambda row: row["shard_id"])
        merged = merge_evaluation_shards([json.loads(row["result_json"]) for row in results])
        executions = [json.loads(row["execution_json"]) for row in results]
    finally:
        try:
            if registry is not None:
                actor_ids = _drain_actors(ray, registry)
        finally:
            try:
                if registry is not None:
                    _terminate_registry(ray, registry)
            finally:
                try:
                    if not already_registered:
                        ray.cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
                finally:
                    ray.shutdown()
    merged["distribution"] = {
        "schema_version": RAY_DATA_DISTRIBUTION_SCHEMA_VERSION,
        "mode": "ray_data",
        "max_workers": maximum,
        "shard_count": len(shards),
        "worker_num_gpus": 1,
        "worker_exit_grace_s": WORKER_EXIT_GRACE_S,
        "namespace": namespace,
        "driver_job_id": driver["job_id"],
        "driver": driver,
        "shards": executions,
        "worker_actor_ids": actor_ids,
        "ray_version": ray.__version__,
        "source_snapshot_sha256": hashlib.sha256(
            json.dumps(bundle["sha256"], sort_keys=True).encode()
        ).hexdigest(),
    }
    return merged
