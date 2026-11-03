"""Ray distribution contracts with a fake cluster and real local subprocesses."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import cloudpickle
import pytest

from atrex_bench import AtrexConfigError, evaluate
from atrex_bench import ray_data as adapter
from atrex_bench._process_guard import LIFETIME_FD_ENV, supervised_popen
from atrex_bench.cli import run_eval as runner
from tests.test_shape_shards import IDS

pytest_plugins = ["tests.test_shape_shards"]

FAKE_RAY = """
import types
__version__ = "2.55.0"
initialized = None
shutdown_called = False
handles = []
current = None
class RayActorError(Exception): pass
class GetTimeoutError(Exception): pass
# Process loss cannot report an application exception.
class WorkerLost(BaseException): pass
exceptions = types.SimpleNamespace(RayActorError=RayActorError, GetTimeoutError=GetTimeoutError)
pickle_registry = set()
cloudpickle = types.SimpleNamespace(
    register_pickle_by_value=lambda m: pickle_registry.add(m.__name__),
    unregister_pickle_by_value=lambda m: pickle_registry.remove(m.__name__),
    list_registry_pickle_by_value=lambda: pickle_registry.copy())
class ID:
    def __init__(self, value): self.value = value
    def hex(self): return self.value
class Method:
    def __init__(self, handle, name): self.handle, self.name = handle, name
    def remote(self, *args, **kwargs):
        if self.handle.dead: raise RayActorError("dead")
        if self.name == "__ray_ready__": return True
        return getattr(self.handle.value, self.name)(*args, **kwargs)
class Handle:
    def __init__(self, value=None):
        self.value, self.dead = value, False
        self._actor_id = ID("actor-%d" % len(handles))
        handles.append(self)
    def __getattr__(self, name): return Method(self, name)
class Remote:
    def __init__(self, cls): self.cls = cls
    def options(self, **kwargs): return self
    def remote(self, *args): return Handle(self.cls(*args))
def remote(**kwargs): return lambda cls: Remote(cls)
def get(value, timeout=None): return value
def kill(handle, no_restart=True):
    handle.dead = True
    if hasattr(handle.value, "__del__"): handle.value.__del__()
def init(**kwargs):
    global initialized
    initialized = kwargs
def shutdown():
    global shutdown_called
    shutdown_called = True
def get_runtime_context():
    handle = current
    return types.SimpleNamespace(
        current_actor=handle, get_job_id=lambda: "job-1",
        get_node_id=lambda: "node-" + handle._actor_id.hex() if handle else "node-driver",
        get_actor_id=lambda: handle._actor_id.hex())
def get_gpu_ids(): return [0] if current else []
"""
FAKE_DATA = """
import os
import ray
last_options = None
last_rows = None
last_results = None
deliveries = {}
mutation = None
class DataContext:
    actor_task_retry_on_errors = False
    retried_map_errors = False
    max_errored_blocks = 0
    @classmethod
    def get_current(cls): return cls
class ActorPoolStrategy:
    def __init__(self, **kwargs): self.options = kwargs
class Dataset:
    def __init__(self, rows, blocks): self.rows, self.blocks = rows, blocks
    def map_batches(self, cls, **kwargs):
        global last_options
        last_options = kwargs
        self.cls, self.options = cls, kwargs
        return self
    def take_all(self):
        global last_results
        count = self.options["compute"].options["min_size"]
        actors = []
        for index in range(count):
            ray.current = ray.Handle()
            actor = self.cls(*self.options["fn_constructor_args"])
            ray.current.value = actor
            actors.append((ray.current, actor))
        results = []
        old_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            for index, row in enumerate(self.rows):
                ray.current, actor = actors[index % count]
                for attempt in range(self.options["max_task_retries"] + 1):
                    deliveries[row["shard_id"]] = deliveries.get(row["shard_id"], 0) + 1
                    try:
                        out = actor({key: [value] for key,value in row.items()})
                        if mutation == "lost_reply" and attempt == 0:
                            raise ray.WorkerLost("lost worker reply")
                        break
                    except ray.WorkerLost as error:
                        if attempt == self.options["max_task_retries"]:
                            raise ray.exceptions.RayActorError(str(error)) from error
                        actor.__del__()
                        actor = self.cls(*self.options["fn_constructor_args"])
                        ray.current.value = actor
                        actors[index % count] = ray.current, actor
                results.append({key: value[0] for key,value in out.items()})
        finally:
            ray.current = None
            if old_mask is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else: os.environ["CUDA_VISIBLE_DEVICES"] = old_mask
        if mutation == "missing": results.pop()
        if mutation == "duplicate": results.append(results[0])
        if mutation == "exception": raise RuntimeError("worker lost")
        if mutation == "empty_success": results[-1]["result_json"] = ""
        if mutation == "fake_success": results[0]["result_json"] = '{"fake": true}'
        last_results = list(reversed(results))
        return last_results
def from_items(rows, override_num_blocks):
    global last_rows
    last_rows = rows
    assert override_num_blocks == len(rows)
    return Dataset(rows, override_num_blocks)
"""


@pytest.fixture
def fake_ray(tmp_path, monkeypatch):
    package = tmp_path / "fake_cluster" / "ray"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(textwrap.dedent(FAKE_RAY))
    (package / "data.py").write_text(textwrap.dedent(FAKE_DATA))
    (package / "util").mkdir()
    (package / "util" / "__init__.py").write_text("")
    (package / "util" / "scheduling_strategies.py").write_text(
        "class NodeAffinitySchedulingStrategy:\n"
        "    def __init__(self, node_id, soft): self.node_id, self.soft = node_id, soft\n"
    )
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.setenv(
        "PYTHONPATH", str(package.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    for name in ("ray", "ray.data", "ray.util", "ray.util.scheduling_strategies"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import ray
    import ray.data

    yield ray, package.parent
    for name in ("ray", "ray.data", "ray.util", "ray.util.scheduling_strategies"):
        sys.modules.pop(name, None)


def config_for(reference, candidate, output, **overrides):
    return {
        "distribution": "ray_data",
        "input": str(candidate),
        "reference_dir": str(reference),
        "output": str(output),
        "ray_max_workers": 2,
        "ray_shapes_per_shard": 2,
        **overrides,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"distribution": "other"},
        {"ray_max_workers": True},
        {"ray_max_workers": 0},
        {"ray_shapes_per_shard": 0},
        {"ray_max_shard_retries": -1},
        {"ray_max_shard_retries": 4},
        {"ray_max_shard_retries": True},
        {"ray_resources": {"GPU": 1}},
        {"ray_resources": {"worker": float("nan")}},
        {"ray_node_selector": {"x": 1}},
        {"ray_namespace": ""},
        {"clock_lock_device": "0"},
        {"checkpoint_dir": "/driver/checkpoints"},
    ],
)
def test_invalid_distribution_config(mutation):
    with pytest.raises(AtrexConfigError):
        adapter.validate_distribution({"distribution": "ray_data", **mutation})


def test_actor_distribution_preserves_shapes_and_paths(evaluator, fake_ray, monkeypatch, tmp_path):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    calls = []

    def run(config):
        calls.append(config.copy())
        assert config["distribution"] == "local"
        assert (Path(config["reference_dir"]) / "shapes.json").read_bytes() == (
            reference / "shapes.json"
        ).read_bytes()
        assert Path(config["input"]).read_bytes() == candidate.read_bytes()
        return local(config["shape_ids"])

    monkeypatch.setattr("atrex_bench.evaluate", run)
    monkeypatch.setenv("AGATE_RAY_NAMESPACE", "gateway-attempt-1")
    result = adapter.evaluate_ray_data(
        config_for(
            reference,
            candidate,
            tmp_path / "output",
            ray_namespace="caller",
            ray_resources={"gpu_worker": 0.001},
            ray_node_selector={"kind": "worker"},
            clock_lock_mode="manage",
            gpu_clock_mhz=1000,
            checkpoint_dir="checkpoints",
        )
    )
    assert result["shape_selection"]["selected_shape_ids"] == IDS
    assert result["distribution"]["max_workers"] == 2
    assert result["distribution"]["shard_count"] == 2
    assert result["distribution"]["namespace"] == "gateway-attempt-1"
    assert result["distribution"]["worker_actor_ids"]
    assert ray.data.last_rows[0]["shape_ids_json"] == json.dumps(["10", "2"])
    assert ray.data.last_options["batch_size"] == 1
    assert ray.data.last_options["num_gpus"] == 1
    assert ray.data.last_options["compute"].options["min_size"] == 2
    assert ray.data.last_options["compute"].options["max_size"] == 2
    assert ray.data.last_options["compute"].options["max_tasks_in_flight_per_actor"] == 1
    assert ray.data.last_options["resources"] == {"gpu_worker": 0.001}
    assert calls[0]["clock_lock_mode"] == "manage"
    assert calls[0]["gpu_clock_mhz"] == 1000
    assert calls[0]["checkpoint_dir"] == "checkpoints"
    assert str(tmp_path / "output") not in calls[0]["output"]
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


@pytest.mark.parametrize("maximum,expected", [(1, 1), (2, 2), (100, 3)])
def test_initial_pool_uses_the_admitted_size_capped_by_shards(
    evaluator, fake_ray, monkeypatch, tmp_path, maximum, expected
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))
    result = adapter.evaluate_ray_data(
        config_for(
            reference, candidate, tmp_path / "out",
            ray_max_workers=maximum, ray_shapes_per_shard=1,
        )
    )
    assert ray.data.last_options["compute"].options["min_size"] == expected
    assert ray.data.last_options["compute"].options["max_size"] == expected
    assert len(result["distribution"]["worker_actor_ids"]) == expected


@pytest.mark.parametrize("shape_count", [20, 100])
def test_automatic_pool_covers_every_shape_with_one_gpu_worker(
    evaluator, fake_ray, monkeypatch, tmp_path, shape_count
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    shape = next(iter(json.loads((reference / "shapes.json").read_text()).values()))
    ids = sorted(str(i) for i in range(1, shape_count + 1))
    (reference / "shapes.json").write_text(json.dumps({sid: shape for sid in ids}))
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))
    config = config_for(reference, candidate, tmp_path / "out", ray_shapes_per_shard=1)
    config.pop("ray_max_workers")
    result = adapter.evaluate_ray_data(config)
    assert result["shape_selection"]["selected_shape_ids"] == ids
    assert result["distribution"]["max_workers"] == shape_count
    assert result["distribution"]["shard_count"] == shape_count
    assert len(set(result["distribution"]["worker_actor_ids"])) == shape_count
    assert ray.data.last_options["compute"].options["min_size"] == shape_count
    assert ray.data.last_options["num_gpus"] == 1
    assert all(len(json.loads(row["shape_ids_json"])) == 1 for row in ray.data.last_rows)
    assert all(handle.dead for handle in ray.handles)


def test_runtime_diagnostics_normalize_ray_identity_objects(fake_ray, monkeypatch):
    ray, _ = fake_ray

    class Identity:
        def __str__(self):
            return "identity-as-text"

    class Context:
        get_job_id = get_node_id = get_actor_id = lambda self: Identity()

    monkeypatch.setattr(ray, "get_runtime_context", Context)
    diagnostics = adapter._runtime_diagnostics(ray, actor=True)
    assert [diagnostics[key] for key in ("job_id", "node_id", "actor_id")] == [
        "identity-as-text"
    ] * 3
    json.dumps(diagnostics, allow_nan=False)


def test_retained_diagnostics_keep_each_shard_identity_and_monotonic_duration(
    evaluator, fake_ray, monkeypatch, tmp_path
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    utc = iter([100, 200, 300, 400, 500, 600])
    monotonic = iter([1_000_000_000, 1_250_000_000,
                      2_000_000_000, 2_500_000_000,
                      3_000_000_000, 3_750_000_000])
    monkeypatch.setattr(adapter.time, "time_ns", lambda: next(utc))
    monkeypatch.setattr(adapter.time, "perf_counter_ns", lambda: next(monotonic))
    result = adapter.evaluate_ray_data(
        config_for(reference, candidate, tmp_path / "out", ray_shapes_per_shard=1)
    )
    distribution = result["distribution"]
    driver = distribution["driver"]
    assert driver == {
        "job_id": "job-1", "node_id": "node-driver", "pid": os.getpid(),
        "assigned_gpu_ids": [], "cuda_visible_devices": "",
        "python_executable": sys.executable,
    }
    assert distribution["driver_job_id"] == driver["job_id"]
    rows = distribution["shards"]
    assert [row["shard_id"] for row in rows] == [f"shape-{i:06d}" for i in range(3)]
    assert [row["shape_ids"] for row in rows] == [[sid] for sid in IDS]
    assert [row["start_utc_ns"] for row in rows] == [100, 300, 500]
    assert [row["end_utc_ns"] for row in rows] == [200, 400, 600]
    assert [row["duration_s"] for row in rows] == [.25, .5, .75]
    assert len({row["node_id"] for row in rows}) == 2
    assert {row["actor_id"] for row in rows} == set(distribution["worker_actor_ids"])
    for row in rows:
        assert row["job_id"] == driver["job_id"]
        assert row["assigned_gpu_ids"] == ["0"]
        assert row["cuda_visible_devices"] == "0"
        assert row["pid"] == os.getpid()
        assert row["python_executable"] == sys.executable
        assert row["end_time_estimated"] is False
        assert row["end_time_source"] == "worker"
        assert set(row) == set(driver) | {
            "actor_id", "shard_id", "shape_ids", "start_utc_ns", "end_utc_ns",
            "duration_s", "attempt", "end_time_estimated", "end_time_source"
        }
    # Fake actors run serially in this test. Retained identities/times do not
    # manufacture a claim of overlapping GPU execution.
    assert all(left["end_utc_ns"] < right["start_utc_ns"]
               for left, right in zip(rows, rows[1:]))
    json.dumps(distribution, allow_nan=False)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "exception"])
def test_distribution_failure_drains_actors(evaluator, fake_ray, monkeypatch, tmp_path, failure):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))
    ray.data.mutation = failure
    with pytest.raises((RuntimeError, ValueError)):
        adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "output"))
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


def test_one_actor_reuses_lifetime_pipe_across_serial_shards(
    evaluator, fake_ray, monkeypatch, tmp_path
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    descriptors, inputs = [], []

    def run(config):
        descriptor = int(os.environ[LIFETIME_FD_ENV])
        os.fstat(descriptor)
        descriptors.append(descriptor)
        inputs.append(config["input"])
        return local(config["shape_ids"])

    monkeypatch.setattr("atrex_bench.evaluate", run)
    result = adapter.evaluate_ray_data(
        config_for(
            reference, candidate, tmp_path / "out", ray_max_workers=1, ray_shapes_per_shard=1
        )
    )
    assert len(descriptors) == 3 and len(set(descriptors)) == 1
    assert len(set(inputs)) == 1
    assert result["distribution"]["max_workers"] == 1
    assert result["distribution"]["shard_count"] == 3
    assert len(result["distribution"]["worker_actor_ids"]) == 1
    assert all(handle.dead for handle in ray.handles)


def test_snapshot_is_immutable_and_includes_adjacent_package(evaluator, tmp_path):
    _, reference, candidate, _ = evaluator
    package = candidate.parent / "helper"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n")
    (candidate.parent / "requirements.txt").write_text("custom-package==1\n")
    bundle = adapter._snapshot_sources(config_for(reference, candidate, tmp_path / "out"))
    original = bundle["files"]["input/candidate.py"]
    candidate.write_text("new revision")
    assert bundle["files"]["input/candidate.py"] == original
    assert bundle["files"]["input/helper/__init__.py"] == b"VALUE = 1\n"
    assert "input/requirements.txt" in bundle["files"]
    assert "runtime/atrex_bench/_process_guard.py" in bundle["files"]


def test_actor_bootstraps_without_installed_bench(evaluator, tmp_path):
    _, reference, candidate, _ = evaluator
    config = config_for(reference, candidate, tmp_path / "out")
    cloudpickle.register_pickle_by_value(adapter)
    try:
        serialized = cloudpickle.dumps(
            (adapter._ShardEvaluator, adapter._snapshot_sources(config), config)
        )
    finally:
        cloudpickle.unregister_pickle_by_value(adapter)
    target = tmp_path / "actor.pkl"
    target.write_bytes(serialized)
    bootstrap = """
import os, sys, types
from pathlib import Path
import cloudpickle
assert "atrex_bench" not in sys.modules
assert not any("atrex-bench/src" in p for p in sys.path)
actor_type, bundle, config = cloudpickle.loads(Path(sys.argv[1]).read_bytes())
class Register:
    def remote(self, *args): return None
class Begin:
    def remote(self, *args): return {"attempt": 1}
sys.modules["ray"] = types.SimpleNamespace(
    __version__="2.55.0", get=lambda x:x,
    get_gpu_ids=lambda: [0],
    get_runtime_context=lambda: types.SimpleNamespace(
        current_actor=None, get_job_id=lambda: "job-bootstrap",
        get_node_id=lambda: "node-bootstrap", get_actor_id=lambda: "actor-bootstrap"))
os.environ["PYTHONPATH"] = "/driver/local-deps:/image/custom-python"
actor = actor_type(bundle, config, types.SimpleNamespace(
    register=Register(), begin=Begin(), commit=Register()))
assert all(Path(p).is_relative_to(actor.root) for p in os.environ["PYTHONPATH"].split(os.pathsep))
import atrex_bench
assert callable(atrex_bench.evaluate)
atrex_bench.evaluate = lambda config: {"shape_ids": config["shape_ids"]}
assert Path(actor.config["input"]).read_bytes() == bundle["files"][bundle["paths"]["input"]]
assert Path(actor.config["reference_dir"]).joinpath("shapes.json").is_file()
row = actor({"shard_id": ["shape-000000"], "shape_ids_json": ['["10"]']})
import json
execution = json.loads(row["execution_json"][0])
assert execution["actor_id"] == "actor-bootstrap"
assert execution["assigned_gpu_ids"] == ["0"]
assert execution["python_executable"] == sys.executable
actor.__del__()
print("bootstrapped")
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(bootstrap), str(target)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip().endswith("bootstrapped")


def test_ray_version_rejected_before_connection(evaluator, fake_ray, tmp_path):
    _, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    ray.__version__ = "2.53.0"
    with pytest.raises(AtrexConfigError, match=r"ray\[data\]"):
        adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "output"))
    assert ray.initialized is None


@pytest.mark.parametrize("previous_registration", [False, True])
def test_pickle_registry_restored_after_failure(
    evaluator, fake_ray, monkeypatch, tmp_path, previous_registration
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    if previous_registration:
        ray.cloudpickle.register_pickle_by_value(adapter)
    expected = ray.cloudpickle.list_registry_pickle_by_value()
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))
    ray.data.mutation = "exception"
    with pytest.raises(RuntimeError, match="worker lost"):
        adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "output"))
    assert ray.cloudpickle.list_registry_pickle_by_value() == expected
    assert ray.shutdown_called


@pytest.mark.parametrize("stage", ["ray_init", "context", "register", "register_after_mutation"])
def test_partial_initialization_restores_state(evaluator, fake_ray, monkeypatch, tmp_path, stage):
    _, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    context = ray.data.DataContext
    context.actor_task_retry_on_errors = True
    context.retried_map_errors = ["existing error"]
    context.max_errored_blocks = 7

    def fail(*args, **kwargs):
        raise RuntimeError("setup interrupted")

    if stage == "ray_init":
        monkeypatch.setattr(ray, "init", fail)
    elif stage == "context":
        class PartialContext:
            actor_task_retry_on_errors = True
            retried_map_errors = ["existing error"]
            max_errored_blocks = 7

            def __setattr__(self, name, value):
                if name == "max_errored_blocks" and value == 0:
                    raise RuntimeError("setup interrupted")
                super().__setattr__(name, value)

        context = PartialContext()
        monkeypatch.setattr(ray.data.DataContext, "get_current", lambda: context)
    else:
        def register(module):
            if stage == "register_after_mutation":
                ray.pickle_registry.add(module.__name__)
            fail()
        monkeypatch.setattr(ray.cloudpickle, "register_pickle_by_value", register)

    with pytest.raises(AtrexConfigError, match="Ray initialization failed: setup interrupted"):
        adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "out"))
    assert context.actor_task_retry_on_errors is True
    assert context.retried_map_errors == ["existing error"]
    assert context.max_errored_blocks == 7
    assert not ray.cloudpickle.list_registry_pickle_by_value()
    assert ray.shutdown_called
    assert not ray.handles


@pytest.mark.parametrize("contents", [None, "{", "{}", '["10", "10"]', '[null]'])
def test_invalid_shape_file_is_config_error_before_connecting(
    evaluator, fake_ray, tmp_path, contents
):
    _, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    path = reference / "shapes.json"
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    with pytest.raises(AtrexConfigError, match="Invalid Ray shape set"):
        adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "out"))
    assert ray.initialized is None


def test_cleanup_failure_preserves_primary_error(
    evaluator, fake_ray, monkeypatch, tmp_path, capsys
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))
    ray.data.mutation = "exception"

    def stalled(*args):
        raise RuntimeError("registry did not terminate")

    monkeypatch.setattr(adapter, "_terminate_registry", stalled)
    with pytest.raises(RuntimeError, match="^worker lost$"):
        adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "out"))
    assert "[atrex-ray-cleanup] RuntimeError: registry did not terminate" in capsys.readouterr().err
    assert not ray.cloudpickle.list_registry_pickle_by_value()
    assert ray.shutdown_called


def test_registry_does_not_claim_shared_namespace_name(
    evaluator, fake_ray, monkeypatch, tmp_path
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    foreign_registry = ray.Handle()
    original_options = ray.Remote.options

    def options(remote, **kwargs):
        # Another driver already owns this namespace's historical fixed name.
        if kwargs.get("name") == "atrex-bench-shards":
            raise ValueError("actor name already exists")
        return original_options(remote, **kwargs)

    monkeypatch.setattr(ray.Remote, "options", options)
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))
    result = adapter.evaluate_ray_data(
        config_for(reference, candidate, tmp_path / "out", ray_namespace="shared")
    )
    assert result["shape_selection"]["selected_shape_ids"] == IDS
    assert not foreign_registry.dead
    assert all(handle.dead for handle in ray.handles[1:])


def test_registry_termination_failure_still_disconnects(evaluator, fake_ray, monkeypatch, tmp_path):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))

    def stalled(*args):
        raise RuntimeError("registry did not terminate")

    monkeypatch.setattr(adapter, "_terminate_registry", stalled)
    with pytest.raises(RuntimeError, match="registry did not terminate"):
        adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "out"))
    assert not ray.cloudpickle.list_registry_pickle_by_value()
    assert ray.shutdown_called


def test_async_kill_acknowledges_workers_and_registry_after_successful_ready_calls(
    evaluator, fake_ray, monkeypatch, tmp_path
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    original_kill, original_remote = ray.kill, ray.Method.remote

    def request_kill(handle, no_restart=True):
        handle.kill_requested = True
        handle.ready_after_kill = 0

    def delayed_remote(method, *args, **kwargs):
        if method.name == "__ray_ready__" and getattr(method.handle, "kill_requested", False):
            method.handle.ready_after_kill += 1
            if method.handle.ready_after_kill < 3:
                return True
            original_kill(method.handle)
        return original_remote(method, *args, **kwargs)

    monkeypatch.setattr(ray, "kill", request_kill)
    monkeypatch.setattr(ray.Method, "remote", delayed_remote)
    monkeypatch.setattr(adapter.time, "sleep", lambda duration: None)
    monkeypatch.setattr("atrex_bench.evaluate", lambda config: local(config["shape_ids"]))
    result = adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "out"))
    assert result["shape_selection"]["selected_shape_ids"] == IDS
    assert all(handle.dead and handle.ready_after_kill == 3 for handle in ray.handles)
    assert ray.shutdown_called


def test_actor_termination_deadline_does_not_accept_successful_ready(fake_ray, monkeypatch):
    ray, _ = fake_ray
    handle = ray.Handle()
    now = [0.0]
    monkeypatch.setattr(adapter, "_ACTOR_TERMINATION_TIMEOUT_S", 0.1)
    monkeypatch.setattr(adapter.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        adapter.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration)
    )
    monkeypatch.setattr(ray, "kill", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="worker registry did not terminate"):
        adapter._terminate_registry(ray, handle)
    assert now[0] == pytest.approx(0.1)
    assert not handle.dead


def test_actor_ready_timeout_is_not_death_confirmation(fake_ray, monkeypatch):
    ray, _ = fake_ray
    handle = ray.Handle()
    monkeypatch.setattr(ray, "kill", lambda *args, **kwargs: None)

    def timed_out(*args, **kwargs):
        raise ray.exceptions.GetTimeoutError("ready task timed out")

    monkeypatch.setattr(ray, "get", timed_out)
    with pytest.raises(RuntimeError, match="worker registry did not terminate") as failure:
        adapter._terminate_registry(ray, handle)
    assert isinstance(failure.value.__cause__, ray.exceptions.GetTimeoutError)
    assert not handle.dead


def test_abba_runs_full_schedule_in_each_shard(evaluator, fake_ray, monkeypatch, tmp_path):
    local, reference, candidate, _ = evaluator
    baseline = candidate.parent / "baseline.py"
    baseline.write_bytes(candidate.read_bytes())
    schedule = []

    def process(**kwargs):
        schedule.append((kwargs["input_path"].name, kwargs["shape_ids"]))
        result = local(kwargs["shape_ids"])
        factor = 2 if kwargs["input_path"].name == "baseline.py" else 1
        for shape in result["performance"]["shapes"].values():
            shape["samples"][0]["end_to_end_time_ms"] *= factor
        return result

    monkeypatch.setattr(runner, "_run_eval_process", process)

    def abba(config):
        return runner.run_abba_eval(
            baseline_path=Path(config["baseline_input"]),
            candidate_path=Path(config["input"]),
            reference_dir=reference,
            output_root=Path(config["output"]),
            shape_ids=config["shape_ids"],
            collect_kernel_events=True,
        )

    monkeypatch.setattr("atrex_bench.evaluate", abba)
    result = adapter.evaluate_ray_data(
        config_for(
            reference, candidate, tmp_path / "out", eval_mode="abba", baseline_input=str(baseline)
        )
    )
    assert [name for name, _ in schedule] == [
        "baseline.py",
        "candidate.py",
        "candidate.py",
        "baseline.py",
    ] * 2
    assert [ids for _, ids in schedule] == [["10", "2"]] * 4 + [["31"]] * 4
    assert result["shape_selection"]["partial"] is False
    assert len(result["distribution"]["shards"]) == 2
    for leg in result["abba"]["runs"]:
        assert leg["result"]["shape_selection"]["selected_shape_ids"] == IDS


def test_incompatible_worker_environment_fails_merge(evaluator, fake_ray, monkeypatch, tmp_path):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray

    def run(config):
        result = local(config["shape_ids"])
        result["environment"]["torch_version"] = config["shape_ids"][0]
        return result

    monkeypatch.setattr("atrex_bench.evaluate", run)
    with pytest.raises(ValueError, match=r"Incompatible.*environment\.torch_version"):
        adapter.evaluate_ray_data(config_for(reference, candidate, tmp_path / "out"))
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


@pytest.mark.parametrize('selection', [None, ['31', '10'], '31,10', '@file'])
def test_sdk_to_cli_distributed_failure_payload_and_pointer(
    evaluator, fake_ray, tmp_path, selection
):
    _, reference, candidate, _ = evaluator
    candidate.write_text("def syntax_error(\n")
    requested = IDS if selection is None else ['10', '31']
    if selection == '@file':
        path = tmp_path / 'selected.txt'
        path.write_text('31\n10\n')
        selection = '@' + str(path)
    result = evaluate(
        config_for(
            reference,
            candidate,
            tmp_path / "real",
            validation_mode="correctness_only",
            ray_shapes_per_shard=2,
            shape_ids=selection,
        )
    )
    assert list(result["passed"]["compile"]) == requested
    assert all(row["status"] == "failed" for row in result["passed"]["compile"].values())
    assert result["shape_selection"]["all_shape_ids"] == IDS
    assert result["shape_selection"]["selected_shape_ids"] == requested
    assert result["shape_selection"]["partial"] is (selection is not None)
    assert result["distribution"]["shard_count"] == (2 if selection is None else 1)
    assert result["distribution"]["driver"]["job_id"] == "job-1"
    assert result["distribution"]["driver"]["assigned_gpu_ids"] == []
    rows = result["distribution"]["shards"]
    assert [row["shape_ids"] for row in rows] == (
        [["10", "2"], ["31"]] if selection is None else [requested]
    )
    assert {row["actor_id"] for row in rows} == set(result["distribution"]["worker_actor_ids"])
    assert all(row["assigned_gpu_ids"] == ["0"] and row["duration_s"] >= 0
               and row["end_utc_ns"] >= row["start_utc_ns"] for row in rows)
    outputs = list((tmp_path / "real").rglob("eval_result.json"))
    assert len(outputs) == 1
    assert json.loads(outputs[0].read_text()) == result


def test_sdk_optional_ray_dependency_error_is_config_error(evaluator, fake_ray, tmp_path):
    _, reference, candidate, _ = evaluator
    _, fake_root = fake_ray
    (fake_root / "ray" / "data.py").write_text("raise ImportError('missing pandas')\n")
    with pytest.raises(AtrexConfigError, match=r"ray\[data\]>=2.54,<2.57"):
        evaluate(config_for(reference, candidate, tmp_path / "out"))


@pytest.mark.parametrize("returncode", [0, 23, -signal.SIGTERM, -signal.SIGKILL])
def test_guard_preserves_child_exit_status(monkeypatch, returncode):
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(LIFETIME_FD_ENV, str(read_fd))
    code = (
        f"import os; os.kill(os.getpid(), {-returncode})" if returncode < 0
        else f"import sys; sys.exit({returncode})"
    )
    process = supervised_popen(
        [sys.executable, "-c", code], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert process.wait(timeout=8) == returncode
    finally:
        os.close(write_fd)
        os.close(read_fd)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_actor_death_guard_kills_stubborn_child(tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(LIFETIME_FD_ENV, str(read_fd))
    pid_file = tmp_path / "child.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGINT,signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    process = supervised_popen(
        [sys.executable, "-c", child_code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists()
        os.close(write_fd)
        write_fd = None
        assert process.wait(timeout=8) == -9
        status = subprocess.run(
            ["ps", "-o", "stat=", "-p", pid_file.read_text()], capture_output=True, text=True
        ).stdout.strip()
        assert not status or status.startswith("Z")
    finally:
        if write_fd is not None:
            os.close(write_fd)
        os.close(read_fd)
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()


@pytest.mark.parametrize("exit_mode", ["natural", "actor_death"])
def test_guard_cleans_descendants_after_direct_child_exit(tmp_path, monkeypatch, exit_mode):
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(LIFETIME_FD_ENV, str(read_fd))
    pid_file = tmp_path / "grandchild.pid"
    descendant = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGINT,signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    direct = """
import signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
subprocess.Popen([sys.executable, "-c", sys.argv[1]])
while not Path(sys.argv[2]).exists(): time.sleep(0.02)
if sys.argv[3] == "actor_death": time.sleep(60)
"""
    process = supervised_popen(
        [sys.executable, "-c", textwrap.dedent(direct), descendant, str(pid_file), exit_mode],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists()
        if exit_mode == "actor_death":
            os.close(write_fd)
            write_fd = None
        assert process.wait(timeout=8) == 0
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = subprocess.run(
                ["ps", "-o", "stat=", "-p", pid_file.read_text()], capture_output=True, text=True
            ).stdout.strip()
            if not status or status.startswith("Z"):
                break
            time.sleep(0.02)
        assert not status or status.startswith("Z")
    finally:
        if write_fd is not None:
            os.close(write_fd)
        os.close(read_fd)
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()


def test_actor_lifetime_janitor_removes_private_sources(tmp_path):
    root = tmp_path / "actor-root"
    root.mkdir()
    (root / "candidate.py").write_text("private source")
    read_fd, write_fd = os.pipe()
    janitor = adapter._start_directory_janitor(root, read_fd)
    try:
        assert root.exists()
        os.close(write_fd)
        write_fd = None
        assert janitor.wait(timeout=8) == 0
        assert not root.exists()
    finally:
        if write_fd is not None:
            os.close(write_fd)
        os.close(read_fd)
        if janitor.poll() is None:
            janitor.kill()
            janitor.wait()


def _retry_fixture(evaluator, tmp_path, monkeypatch):
    _, reference, candidate, _ = evaluator
    shape = next(iter(json.loads((reference / 'shapes.json').read_text()).values()))
    (reference / 'shapes.json').write_text(json.dumps({str(i): shape for i in range(20)}))
    monkeypatch.setattr(adapter, '_start_directory_janitor', lambda *args: None)
    monkeypatch.setattr(adapter.time, 'sleep', lambda seconds: None)
    # Keep the real merger: a stub would miss omitted shards, failure coverage,
    # requested subsets, and synthetic successful results for exhausted workers.
    return config_for(reference, candidate, tmp_path / 'output', ray_shapes_per_shard=1)


@pytest.mark.parametrize('failures', [1, 3])
def test_only_ten_lost_workers_retry_in_twenty_shards(
    evaluator, fake_ray, monkeypatch, tmp_path, failures
):
    from collections import Counter

    local, _, _, _ = evaluator
    ray, _ = fake_ray
    config = _retry_fixture(evaluator, tmp_path, monkeypatch)
    calls = Counter()

    def run(config):
        sid = config['shape_ids'][0]
        calls[sid] += 1
        if int(sid) < 10 and calls[sid] <= failures:
            raise ray.WorkerLost('injected worker loss')
        return local(config['shape_ids'], mode='performance_only')

    monkeypatch.setattr('atrex_bench.evaluate', run)
    result = adapter.evaluate_ray_data(config)
    ids = sorted(map(str, range(20)))
    assert result['shape_selection']['all_shape_ids'] == ids
    assert result['shape_selection']['selected_shape_ids'] == ids
    assert result['shape_selection']['partial'] is False
    assert list(result['performance']['shapes']) == ids
    assert sum(calls.values()) == 20 + 10 * failures
    assert all(calls[str(i)] == (failures + 1 if i < 10 else 1) for i in range(20))
    attempts = result['distribution']['shard_attempts']
    assert sum(len(rows) for rows in attempts.values()) == 20 + 10 * failures
    lost = sum(row['status'] == 'worker_lost' for rows in attempts.values() for row in rows)
    assert lost == 10 * failures
    assert len(result['distribution']['worker_actor_ids']) == 2
    assert ray.data.last_options['num_gpus'] == 1
    assert ray.data.last_options['max_task_retries'] == 4
    assert ray.data.last_options['max_restarts'] == 80
    assert result['distribution']['max_shard_retries'] == 3
    for rows in attempts.values():
        assert [row['attempt'] for row in rows] == list(range(1, len(rows) + 1))
        for row in rows:
            assert row['end_utc_ns'] >= row['start_utc_ns']
            assert row['duration_s'] >= 0
            assert row['end_time_estimated'] is (row['status'] == 'worker_lost')
            assert row['end_time_source'] == (
                'driver_observed_loss' if row['status'] == 'worker_lost' else 'worker'
            )
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


def test_committed_shards_survive_lost_worker_replies(evaluator, fake_ray, monkeypatch, tmp_path):
    from collections import Counter

    local, _, _, _ = evaluator
    ray, _ = fake_ray
    config = _retry_fixture(evaluator, tmp_path, monkeypatch)
    calls = Counter()

    def run(config):
        sid = config['shape_ids'][0]
        calls[sid] += 1
        return local(config['shape_ids'], mode='performance_only')

    ray.data.mutation = 'lost_reply'
    monkeypatch.setattr('atrex_bench.evaluate', run)
    result = adapter.evaluate_ray_data(config)
    assert len(result['performance']['shapes']) == 20
    assert len(calls) == 20 and set(calls.values()) == {1}
    assert sum(len(rows) for rows in result['distribution']['shard_attempts'].values()) == 20
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


@pytest.mark.parametrize('retries', [None, 0, 1, 2, 3])
def test_worker_retry_exhaustion_is_bounded_and_later_shards_complete(
    evaluator, fake_ray, monkeypatch, tmp_path, retries
):
    from collections import Counter

    local, _, _, _ = evaluator
    ray, _ = fake_ray
    config = _retry_fixture(evaluator, tmp_path, monkeypatch)
    if retries is not None:
        config['ray_max_shard_retries'] = retries
    limit = 3 if retries is None else retries
    calls = Counter()
    ids = sorted(map(str, range(20)))
    failed_ids = ids[:10]
    completed_ids = ids[10:]

    def run(config):
        sid = config['shape_ids'][0]
        calls[sid] += 1
        if sid in failed_ids:
            raise ray.WorkerLost('worker unavailable')
        return local(config['shape_ids'], mode='performance_only')

    monkeypatch.setattr('atrex_bench.evaluate', run)
    result = adapter.evaluate_ray_data(config)
    selection = result['shape_selection']
    assert selection['all_shape_ids'] == ids
    assert selection['selected_shape_ids'] == ids
    assert selection['partial'] is True
    failures = selection['failed_shapes']
    assert [failure['shape_id'] for failure in failures] == failed_ids
    assert all(failure['reason'].strip() and failure['retries'] == limit for failure in failures)
    assert all(calls[sid] == limit + 1 for sid in failed_ids)
    assert all(calls[sid] == 1 for sid in completed_ids)
    assert set(selection['input_seeds']) == set(ids)
    for stage in ('compile', 'correctness', 'performance'):
        assert list(result['passed'][stage]) == completed_ids
    for stage in ('correctness', 'performance'):
        assert list(result[stage]['shapes']) == completed_ids
    assert ray.data.last_options['max_task_retries'] == limit + 1
    assert ray.data.last_options['max_restarts'] == 20 * (limit + 1)
    for index, row in enumerate(sorted(ray.data.last_results, key=lambda row: row['shard_id'])):
        sid = ids[index]
        attempts = result['distribution']['shard_attempts'][row['shard_id']]
        assert len(attempts) == (limit + 1 if sid in failed_ids else 1)
        if sid in failed_ids:
            assert row['result_json'] == ''
            assert ray.data.deliveries[row['shard_id']] == limit + 2
            assert all(attempt['status'] == 'worker_lost' for attempt in attempts)
            assert all(attempt['end_time_estimated'] is True for attempt in attempts)
            assert all(attempt['end_time_source'] == 'driver_observed_loss' for attempt in attempts)
        else:
            assert json.loads(row['result_json'])['shape_selection']['selected_shape_ids'] == [sid]
            assert attempts[0]['status'] == 'completed'
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


@pytest.mark.parametrize('retries', [0, 3])
def test_all_exhausted_shards_raise_after_accounting_for_every_shape(
    evaluator, fake_ray, monkeypatch, tmp_path, retries
):
    from collections import Counter

    ray, _ = fake_ray
    config = _retry_fixture(evaluator, tmp_path, monkeypatch)
    config['ray_max_shard_retries'] = retries
    calls = Counter()

    def run(config):
        calls[config['shape_ids'][0]] += 1
        raise ray.WorkerLost('worker unavailable')

    monkeypatch.setattr('atrex_bench.evaluate', run)
    with pytest.raises((RuntimeError, ValueError), match='All Ray evaluation shards failed'):
        adapter.evaluate_ray_data(config)
    assert len(calls) == 20 and set(calls.values()) == {retries + 1}
    assert len(ray.data.last_results) == 20
    assert all(row['result_json'] == '' for row in ray.data.last_results)
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


def test_application_error_is_not_retried_and_context_is_restored(
    evaluator, fake_ray, monkeypatch, tmp_path
):
    from atrex_bench import sdk

    ray, _ = fake_ray
    config = _retry_fixture(evaluator, tmp_path, monkeypatch)
    original_start = sdk._start_process
    calls = []
    context = ray.data.DataContext.get_current()
    context.actor_task_retry_on_errors = True
    context.retried_map_errors = True
    context.max_errored_blocks = 100

    def run(config):
        assert context.actor_task_retry_on_errors is False
        assert context.retried_map_errors is False
        assert context.max_errored_blocks == 0
        calls.append(config['shape_ids'])
        raise ValueError('candidate/config error')

    monkeypatch.setattr('atrex_bench.evaluate', run)
    with pytest.raises(ValueError, match='candidate/config error'):
        adapter.evaluate_ray_data(config)
    assert len(calls) == 1
    assert context.actor_task_retry_on_errors is True
    assert context.retried_map_errors is True
    assert context.max_errored_blocks == 100
    assert sdk._start_process is original_start
    assert LIFETIME_FD_ENV not in os.environ
    assert not ray.cloudpickle.list_registry_pickle_by_value()
    registry = next(handle.value for handle in ray.handles
                    if isinstance(handle.value, adapter._ActorRegistry))
    progress = registry.progress()
    assert progress['completed_shard_ids'] == []
    attempts = list(progress['attempts'].values())
    assert len(attempts) == 1 and len(attempts[0]) == 1
    execution = attempts[0][0]
    assert execution['status'] == 'application_failed'
    assert execution['end_utc_ns'] >= execution['start_utc_ns']
    assert execution['duration_s'] >= 0
    assert execution['end_time_estimated'] is False
    assert execution['end_time_source'] == 'worker'
    assert 'candidate/config error' in execution['reason']
    with pytest.raises(RuntimeError, match='Cannot retry an application failure'):
        registry.begin('shape-000000', {})
    with pytest.raises(RuntimeError, match='Stale Ray shard completion'):
        registry.commit('shape-000000', '{}', json.dumps({'attempt': 1}))
    assert registry.progress() == progress
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


@pytest.mark.parametrize('retries', [None, 0, 1, 2, 3])
def test_registry_bounds_retries_even_if_a_map_bundle_is_replayed(retries):
    registry = (
        adapter._ActorRegistry() if retries is None
        else adapter._ActorRegistry(max_retries=retries)
    )
    limit = 3 if retries is None else retries
    for attempt in range(1, limit + 2):
        assert registry.begin('failed', {
            'shard_id': 'failed', 'shape_ids': ['10', '31'], 'actor_id': 'actor-1',
            'start_utc_ns': time.time_ns(),
        }) == {'attempt': attempt}
    terminal = registry.begin('failed', {'shape_ids': ['10', '31']})
    assert set(terminal) == {'failed'}
    assert terminal['failed']['reason'].strip()
    assert terminal['failed']['retries'] == limit
    before = json.loads(json.dumps(registry.progress()))
    for _ in range(5):
        assert registry.begin('failed', {}) == terminal
    assert registry.progress() == before
    assert len(before['attempts']['failed']) == limit + 1
    assert all(row['status'] == 'worker_lost' for row in before['attempts']['failed'])
    for row in before['attempts']['failed']:
        assert row['end_utc_ns'] >= row['start_utc_ns']
        assert row['duration_s'] >= 0
        assert row['end_time_estimated'] is True
        assert row['end_time_source'] == 'driver_observed_loss'
    with pytest.raises(RuntimeError, match='Stale Ray shard completion'):
        registry.commit('failed', 'late', json.dumps({'attempt': limit + 1}))
    assert registry.progress() == before
    registry.begin('done', {})
    registry.commit('done', '{"passed":true}', '{"attempt":1}')
    for _ in range(5):
        assert registry.begin('done', {})['cached']['result_json'] == '{"passed":true}'
    assert len(registry.progress()['attempts']['done']) == 1


def test_old_worker_cannot_overwrite_a_newer_shard_attempt():
    registry = adapter._ActorRegistry(max_retries=1)
    registry.begin('shape', {'start_utc_ns': time.time_ns()})
    registry.begin('shape', {'start_utc_ns': time.time_ns()})
    with pytest.raises(RuntimeError, match='Stale Ray shard completion'):
        registry.commit('shape', 'old', '{"attempt":1}')
    assert registry.progress()['completed_shard_ids'] == []
    registry.commit('shape', 'new', '{"attempt":2}')
    with pytest.raises(RuntimeError, match='Stale Ray shard completion'):
        registry.commit('shape', 'old', '{"attempt":1}')
    assert registry.begin('shape', {})['cached']['result_json'] == 'new'


@pytest.mark.parametrize('version', ['2.54.0', '2.55.0'])
def test_pre_256_context_does_not_require_map_exception_retry_option(
    evaluator, fake_ray, monkeypatch, tmp_path, version
):
    local, _, _, _ = evaluator
    ray, _ = fake_ray
    ray.__version__ = version
    config = _retry_fixture(evaluator, tmp_path, monkeypatch)
    context = ray.data.DataContext.get_current()
    monkeypatch.delattr(context, 'retried_map_errors')
    monkeypatch.setattr('atrex_bench.evaluate',
                        lambda config: local(config['shape_ids'], mode='performance_only'))
    result = adapter.evaluate_ray_data(config)
    assert len(result['performance']['shapes']) == 20
    assert not hasattr(context, 'retried_map_errors')
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


@pytest.mark.parametrize('selection', [['31', '10'], '31,10', '@file'])
@pytest.mark.parametrize('shard_size,expected_workers', [(1, 2), (2, 1)])
def test_ray_subset_is_selected_before_partitioning_and_pool_admission(
    evaluator, fake_ray, monkeypatch, tmp_path, selection, shard_size, expected_workers
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    original_shapes = (reference / 'shapes.json').read_bytes()
    if selection == '@file':
        path = tmp_path / 'selection.json'
        path.write_text('["31", "10"]')
        selection = '@' + str(path)
    config = config_for(
        reference, candidate, tmp_path / 'out', shape_ids=selection,
        ray_shapes_per_shard=shard_size, ray_max_workers=100,
    )
    original_config = config.copy()
    calls = []

    def run(worker_config):
        calls.append(worker_config['shape_ids'])
        assert worker_config['distribution'] == 'local'
        assert json.loads((Path(worker_config['reference_dir']) / 'shapes.json').read_text()) == (
            json.loads(original_shapes)
        )
        return local(worker_config['shape_ids'], mode='performance_only')

    monkeypatch.setattr('atrex_bench.evaluate', run)
    result = adapter.evaluate_ray_data(config)
    requested = ['10', '31']
    assert calls == ([['10'], ['31']] if shard_size == 1 else [requested])
    assert result['shape_selection']['all_shape_ids'] == IDS
    assert result['shape_selection']['selected_shape_ids'] == requested
    assert result['shape_selection']['partial'] is True
    assert result['runner_config']['shape_ids'] == requested
    assert list(result['performance']['shapes']) == requested
    assert result['distribution']['shard_count'] == expected_workers
    assert result['distribution']['max_workers'] == expected_workers
    assert ray.data.last_options['compute'].options['min_size'] == expected_workers
    assert ray.data.last_options['compute'].options['max_size'] == expected_workers
    assert ray.data.last_options['max_restarts'] == expected_workers * 4
    assert ray.data.last_options['max_task_retries'] == 4
    assert config == original_config
    assert (reference / 'shapes.json').read_bytes() == original_shapes
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


@pytest.mark.parametrize('selection', [[], '', ['unknown'], ['10', '10'], [10], '@missing'])
def test_invalid_ray_selection_is_rejected_before_connection(
    evaluator, fake_ray, tmp_path, selection
):
    _, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    if selection == '@missing':
        selection = '@' + str(tmp_path / 'missing-ids.txt')
    with pytest.raises(AtrexConfigError):
        adapter.evaluate_ray_data(
            config_for(reference, candidate, tmp_path / 'out', shape_ids=selection)
        )
    assert ray.initialized is None
    assert not ray.handles


@pytest.mark.parametrize('selection', [None, ['10', '2', '31'], ['31', '10']])
@pytest.mark.parametrize(
    'mutation', [None, 'missing', 'duplicate', 'empty_success', 'fake_success']
)
def test_exhausted_multi_shape_shard_has_one_failure_per_shape_and_strict_transport_evidence(
    evaluator, fake_ray, monkeypatch, tmp_path, selection, mutation
):
    local, reference, candidate, _ = evaluator
    ray, _ = fake_ray
    monkeypatch.setattr(adapter, '_start_directory_janitor', lambda *args: None)
    monkeypatch.setattr(adapter.time, 'sleep', lambda seconds: None)
    context = ray.data.DataContext.get_current()
    context.actor_task_retry_on_errors = True
    context.retried_map_errors = ['prior']
    context.max_errored_blocks = 7
    calls = []

    def run(config):
        calls.append(config['shape_ids'])
        if '10' in config['shape_ids']:
            raise ray.WorkerLost('worker unavailable')
        return local(config['shape_ids'], mode='performance_only')

    monkeypatch.setattr('atrex_bench.evaluate', run)
    ray.data.mutation = mutation
    requested = IDS if selection is None else sorted(selection)
    failed_ids = [sid for sid in requested if sid != '31']
    config = config_for(
        reference, candidate, tmp_path / 'out', shape_ids=selection,
        ray_shapes_per_shard=len(failed_ids), ray_max_workers=1, ray_max_shard_retries=1,
    )
    if mutation:
        with pytest.raises((RuntimeError, ValueError), match='Missing|Duplicate|terminal evidence'):
            adapter.evaluate_ray_data(config)
    else:
        result = adapter.evaluate_ray_data(config)
        assert result['shape_selection']['all_shape_ids'] == IDS
        assert result['shape_selection']['selected_shape_ids'] == requested
        assert result['shape_selection']['partial'] is True
        failures = result['shape_selection']['failed_shapes']
        assert [failure['shape_id'] for failure in failures] == failed_ids
        assert all(failure['retries'] == 1 and failure['reason'].strip() for failure in failures)
        assert set(result['shape_selection']['input_seeds']) == set(requested)
        assert list(result['performance']['shapes']) == ['31']
        attempts = result['distribution']['shard_attempts']
        assert len(attempts['shape-000000']) == 2
        assert attempts['shape-000001'][0]['status'] == 'completed'
        tombstone = next(row for row in ray.data.last_results if row['shard_id'] == 'shape-000000')
        assert tombstone['result_json'] == ''
    assert calls == [failed_ids, failed_ids, ['31']]
    assert context.actor_task_retry_on_errors is True
    assert context.retried_map_errors == ['prior']
    assert context.max_errored_blocks == 7
    assert not ray.cloudpickle.list_registry_pickle_by_value()
    assert ray.shutdown_called and all(handle.dead for handle in ray.handles)


def test_register_closes_only_running_attempts_of_restarted_actor(fake_ray, monkeypatch):
    ray, _ = fake_ray
    registry = adapter._ActorRegistry(max_retries=1)
    handle = ray.Handle()
    assert registry.register(handle) is False
    actor_id = handle._actor_id.hex()
    for shard_id, worker in [('lost-a', actor_id), ('lost-b', actor_id), ('other', 'other-actor')]:
        registry.begin(shard_id, {
            'shard_id': shard_id, 'shape_ids': ['10'], 'actor_id': worker,
            'start_utc_ns': 1_000_000_000,
        })
    registry.begin('completed', {
        'actor_id': actor_id, 'start_utc_ns': 1_000_000_000,
    })
    registry.commit('completed', '{}', json.dumps({
        'attempt': 1, 'end_utc_ns': 2_000_000_000, 'duration_s': 1.0,
        'end_time_estimated': False, 'end_time_source': 'worker',
    }))
    replacement = ray.Handle()
    replacement._actor_id = handle._actor_id
    monkeypatch.setattr(adapter.time, 'time_ns', lambda: 3_000_000_000)
    assert registry.register(replacement) is True
    progress = registry.progress()
    for shard_id in ('lost-a', 'lost-b'):
        attempt = progress['attempts'][shard_id][0]
        assert attempt['status'] == 'worker_lost'
        assert attempt['end_utc_ns'] == 3_000_000_000
        assert attempt['duration_s'] == 2.0
        assert attempt['end_time_estimated'] is True
        assert attempt['end_time_source'] == 'driver_observed_loss'
    assert progress['attempts']['other'][0]['status'] == 'running'
    assert progress['attempts']['completed'][0]['status'] == 'completed'
    snapshot = json.loads(json.dumps(progress))
    with pytest.raises(RuntimeError, match='Stale Ray shard completion'):
        registry.commit('lost-a', '{}', '{"attempt":1}')
    with pytest.raises(RuntimeError, match='Stale Ray shard failure'):
        registry.fail('lost-b', '{"attempt":1}', 'late failure')
    assert registry.progress() == snapshot
    assert progress['attempts']['completed'][0]['end_utc_ns'] == 2_000_000_000
    assert registry.workers() == [replacement]
    closed = json.loads(json.dumps(progress['attempts']['lost-a'][0]))
    monkeypatch.setattr(adapter.time, 'time_ns', lambda: 4_000_000_000)
    assert registry.begin('lost-a', {
        'actor_id': actor_id, 'start_utc_ns': 4_000_000_000,
    }) == {'attempt': 2}
    assert registry.progress()['attempts']['lost-a'][0] == closed


@pytest.mark.parametrize('error_kind', ['value', 'ray_actor'])
def test_worker_records_application_failure_without_turning_it_into_transport_retry(
    fake_ray, monkeypatch, tmp_path, error_kind
):
    from atrex_bench import sdk

    ray, _ = fake_ray
    registry = adapter._ActorRegistry(max_retries=0)
    worker = adapter._ShardEvaluator.__new__(adapter._ShardEvaluator)
    worker.root, worker.config = tmp_path, {'distribution': 'local'}
    worker.registry = ray.Handle(registry)
    worker.read_fd, worker.write_fd = os.pipe()
    ray.current = ray.Handle(worker)
    original_start = sdk._start_process
    error = (ValueError('application error') if error_kind == 'value'
             else ray.exceptions.RayActorError('application error'))
    calls = []

    def run(config):
        calls.append(config['shape_ids'])
        raise error

    monkeypatch.setattr('atrex_bench.evaluate', run)
    utc = iter([100, 200])
    monotonic = iter([1_000_000_000, 1_500_000_000])
    monkeypatch.setattr(adapter.time, 'time_ns', lambda: next(utc))
    monkeypatch.setattr(adapter.time, 'perf_counter_ns', lambda: next(monotonic))
    try:
        with pytest.raises(type(error), match='application error') as caught:
            worker({'shard_id': ['shape'], 'shape_ids_json': ['["10", "31"]']})
        assert caught.value is error
        assert calls == [['10', '31']]
        execution = registry.progress()['attempts']['shape'][0]
        assert execution['status'] == 'application_failed'
        assert execution['start_utc_ns'] == 100
        assert execution['end_utc_ns'] == 200
        assert execution['duration_s'] == .5
        assert execution['end_time_estimated'] is False
        assert execution['end_time_source'] == 'worker'
        assert 'application error' in execution['reason']
        assert sdk._start_process is original_start
        assert LIFETIME_FD_ENV not in os.environ
    finally:
        ray.current = None
        worker.__del__()


def test_terminal_application_failure_rejects_late_commit_and_preserves_evidence():
    registry = adapter._ActorRegistry(max_retries=3)
    registry.begin('shape', {'start_utc_ns': 100, 'shape_ids': ['10']})
    registry.fail('shape', json.dumps({
        'attempt': 1, 'end_utc_ns': 200, 'duration_s': .5,
        'end_time_estimated': False, 'end_time_source': 'worker',
    }), 'ValueError: application error')
    before = json.loads(json.dumps(registry.progress()))
    with pytest.raises(RuntimeError, match='Cannot retry an application failure'):
        registry.begin('shape', {})
    with pytest.raises(RuntimeError, match='Stale Ray shard completion'):
        registry.commit('shape', 'late success', '{"attempt":1}')
    assert registry.progress() == before


@pytest.mark.parametrize('terminal', ['completed', 'application_failed', 'worker_exhausted'])
def test_late_failure_cannot_replace_terminal_registry_evidence(terminal):
    registry = adapter._ActorRegistry(max_retries=0)
    execution = {'start_utc_ns': time.time_ns(), 'shape_ids': ['10']}
    registry.begin('shape', execution)
    encoded = '{"attempt":1,"end_utc_ns":200,"duration_s":0.5}'
    if terminal == 'completed':
        registry.commit('shape', 'success', encoded)
    elif terminal == 'application_failed':
        registry.fail('shape', encoded, 'original application failure')
    else:
        registry.begin('shape', execution)
    before = json.loads(json.dumps(registry.progress()))
    with pytest.raises(RuntimeError, match='Stale Ray shard failure'):
        registry.fail('shape', encoded, 'late failure')
    assert registry.progress() == before


def test_old_worker_failure_cannot_replace_new_running_attempt():
    registry = adapter._ActorRegistry(max_retries=1)
    registry.begin('shape', {'start_utc_ns': time.time_ns()})
    registry.begin('shape', {'start_utc_ns': time.time_ns()})
    before = json.loads(json.dumps(registry.progress()))
    with pytest.raises(RuntimeError, match='Stale Ray shard failure'):
        registry.fail('shape', '{"attempt":1}', 'old worker error')
    assert registry.progress() == before


def test_begin_closes_prior_running_attempt_with_observed_loss_time(monkeypatch):
    registry = adapter._ActorRegistry(max_retries=1)
    execution = {'actor_id': 'old-worker', 'start_utc_ns': 1_000_000_000}
    registry.begin('shape', execution)
    monkeypatch.setattr(adapter.time, 'time_ns', lambda: 2_500_000_000)
    registry.begin('shape', {'actor_id': 'new-worker', 'start_utc_ns': 2_500_000_000})
    old, new = registry.progress()['attempts']['shape']
    assert old['status'] == 'worker_lost'
    assert old['end_utc_ns'] == 2_500_000_000
    assert old['duration_s'] == 1.5
    assert old['end_time_estimated'] is True
    assert old['end_time_source'] == 'driver_observed_loss'
    assert new['status'] == 'running' and new['attempt'] == 2
    assert execution == {'actor_id': 'old-worker', 'start_utc_ns': 1_000_000_000}
