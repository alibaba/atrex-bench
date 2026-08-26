"""Tests for Stage 2: performance profiling."""

import builtins
import contextlib
from pathlib import Path
from types import SimpleNamespace

import torch

from atrex_bench.eval import performance as performance_module
from atrex_bench.eval.performance import (
    PerformanceSample,
    PerformanceShapeResult,
    benchmark_performance,
    benchmark_reference_torch_compile,
)

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "references" / "atrex_001" / "reference.py"
CANDIDATE_PATH = Path(__file__).parent / "fixtures" / "generations" / "atrex_001.py"


def _write_python_file(tmp_path: Path, name: str, content: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_performance_result_records_cuda_graph_metadata() -> None:
    result = PerformanceShapeResult(
        benchmark_mode="cuda_graph_replay",
        capture_time_ms=3.5,
        cache_flush_mb=1024,
        graph_correctness={"passed": True, "outputs": []},
    )

    assert result.benchmark_mode == "cuda_graph_replay"
    assert result.capture_time_ms == 3.5
    assert result.cache_flush_mb == 1024
    assert result.graph_correctness == {"passed": True, "outputs": []}


def test_cuda_graph_mode_rejects_cpu_device() -> None:
    result = benchmark_performance(
        CANDIDATE_PATH,
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=1,
        benchmark_mode="cuda_graph_replay",
        device="cpu",
    )

    assert result.samples == []
    assert result.error is not None
    assert "requires a CUDA device" in result.error


def test_unknown_benchmark_mode_is_rejected() -> None:
    result = benchmark_performance(
        CANDIDATE_PATH,
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=1,
        benchmark_mode="unknown",
        device="cpu",
    )

    assert result.samples == []
    assert result.error is not None
    assert "unsupported benchmark_mode" in result.error


def test_measure_runner_dispatches_cuda_graph_mode(monkeypatch) -> None:
    calls: list[str] = []
    expected = performance_module._MeasurementResult(
        samples=[PerformanceSample(end_to_end_time_ms=1.25)],
    )

    monkeypatch.setattr(
        performance_module,
        "_measure_eager_samples",
        lambda **_kwargs: calls.append("eager"),
    )
    monkeypatch.setattr(
        performance_module,
        "_measure_cuda_graph_samples",
        lambda **_kwargs: calls.append("cuda_graph_replay") or expected,
    )
    monkeypatch.setattr(
        performance_module,
        "clone_model_inputs",
        lambda value: value,
    )

    result = performance_module._measure_runner_samples(
        torch.nn.Identity(),
        SimpleNamespace(args=(torch.ones(1),), kwargs={}),
        torch.device("cuda"),
        warmup_iters=1,
        bench_iters=2,
        benchmark_mode="cuda_graph_replay",
    )

    assert result is expected
    assert calls == ["cuda_graph_replay"]


def test_bench_fn_returns_the_forward_output(monkeypatch) -> None:
    """The timed closure must hand back what the forward returned.

    ``_measure_cuda_graph_samples`` captures and compares that value, so
    wrapping the forward -- as the NVTX range does -- must not swallow it.
    """
    captured: dict[str, object] = {}
    expected = performance_module._MeasurementResult(
        samples=[PerformanceSample(end_to_end_time_ms=1.25)],
    )

    def _fake_cuda_graph(**kwargs):
        captured["bench_fn"] = kwargs["bench_fn"]
        return expected

    monkeypatch.setattr(
        performance_module, "_measure_cuda_graph_samples", _fake_cuda_graph
    )
    monkeypatch.setattr(performance_module, "clone_model_inputs", lambda value: value)

    sentinel = torch.ones(3)
    performance_module._measure_runner_samples(
        torch.nn.Identity(),
        SimpleNamespace(args=(sentinel,), kwargs={}),
        torch.device("cuda"),
        warmup_iters=1,
        bench_iters=2,
        benchmark_mode="cuda_graph_replay",
    )

    assert torch.equal(captured["bench_fn"](), sentinel)


def test_eager_mode_keeps_every_iteration(monkeypatch) -> None:
    """do_bench measures each iteration; eager must not throw them away.

    The cuda_graph path already emits one sample per iteration, and
    PerformanceSample documents the list as iteration-ordered.
    """
    recorded: dict[str, object] = {}

    def _fake_do_bench(_fn, *, warmup, rep, return_mode=None):
        recorded["return_mode"] = return_mode
        return [1.0, 2.0, 3.0]

    monkeypatch.setitem(
        __import__("sys").modules,
        "triton.testing",
        SimpleNamespace(do_bench=_fake_do_bench),
    )

    result = performance_module._measure_eager_samples(
        bench_fn=lambda: None,
        device=torch.device("cpu"),
        warmup_iters=1,
        bench_iters=3,
    )

    assert recorded["return_mode"] == "all"
    assert [s.end_to_end_time_ms for s in result.samples] == [1.0, 2.0, 3.0]


def test_eager_mode_falls_back_when_triton_cannot_report_each_run(monkeypatch) -> None:
    """An older do_bench rejects return_mode; keep the reduced value, do not fail."""

    def _old_do_bench(_fn, *, warmup, rep, **kwargs):
        if kwargs:
            raise TypeError("do_bench() got an unexpected keyword argument")
        return 4.5

    monkeypatch.setitem(
        __import__("sys").modules,
        "triton.testing",
        SimpleNamespace(do_bench=_old_do_bench),
    )

    result = performance_module._measure_eager_samples(
        bench_fn=lambda: None,
        device=torch.device("cpu"),
        warmup_iters=1,
        bench_iters=3,
    )

    assert [s.end_to_end_time_ms for s in result.samples] == [4.5]


def test_forward_nvtx_is_a_noop_without_cuda(monkeypatch) -> None:
    """No CUDA means no NVTX range, and no failure either."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with performance_module._forward_nvtx():
        pass


def test_forward_nvtx_survives_a_missing_nvtx_backend(monkeypatch) -> None:
    """A build without NVTX must degrade to a no-op, not fail the measurement."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _unavailable(_name):
        raise RuntimeError("NVTX is not available in this build")

    monkeypatch.setattr(torch.cuda.nvtx, "range", _unavailable)

    with performance_module._forward_nvtx():
        pass


def test_cuda_graph_measurement_captures_once_and_replays(monkeypatch) -> None:
    calls = {"forward": 0, "replay": 0, "flush": 0}

    class FakeGraph:
        def replay(self) -> None:
            calls["replay"] += 1

    class FakeEvent:
        def __init__(self, *, enable_timing: bool):
            assert enable_timing is True

        def record(self) -> None:
            pass

        def elapsed_time(self, _other) -> float:
            return 0.5

    def bench_fn():
        calls["forward"] += 1
        return torch.ones(2)

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(
        torch.cuda,
        "graph",
        lambda _graph: contextlib.nullcontext(),
    )
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(performance_module, "sync_device", lambda _device: None)
    monkeypatch.setattr(
        performance_module,
        "_flush_cuda_cache",
        lambda _size, _device: calls.__setitem__("flush", calls["flush"] + 1),
    )

    result = performance_module._measure_cuda_graph_samples(
        bench_fn=bench_fn,
        device=torch.device("cuda"),
        warmup_iters=2,
        bench_iters=3,
        cache_flush_mb=1024,
        atol=1e-2,
        rtol=0.05,
    )

    assert calls == {
        "forward": 3,
        "replay": 5,
        "flush": 5,
    }
    assert [sample.end_to_end_time_ms for sample in result.samples] == [
        0.5,
        0.5,
        0.5,
    ]
    assert result.graph_correctness is not None
    assert result.graph_correctness["passed"] is True


def test_cuda_graph_cosine_policy_accepts_atomic_variation() -> None:
    eager = torch.ones(16384)
    graph = eager.clone()
    graph[0] = 2.0

    strict = performance_module._compare_graph_outputs(
        [("output", eager)],
        graph,
        atol=1e-2,
        rtol=0.05,
        min_cosine=None,
        max_rel_l2=None,
    )
    cosine = performance_module._compare_graph_outputs(
        [("output", eager)],
        graph,
        atol=1e-2,
        rtol=0.05,
        min_cosine=0.999,
        max_rel_l2=0.01,
    )

    assert strict["passed"] is False
    assert strict["policy"] == {
        "type": "allclose",
        "atol": 1e-2,
        "rtol": 0.05,
    }
    assert cosine["passed"] is True
    assert cosine["policy"] == {
        "type": "cosine_rel_l2",
        "min_cosine": 0.999,
        "max_rel_l2": 0.01,
    }
    assert cosine["outputs"][0]["cosine_similarity"] >= 0.999
    assert cosine["outputs"][0]["relative_l2"] <= 0.01
    assert cosine["outputs"][0]["max_elementwise_abs_diff"] == 1.0


def test_cuda_graph_cosine_policy_rejects_scale_drift() -> None:
    eager = torch.ones(4096)
    graph = eager * 2.0

    result = performance_module._compare_graph_outputs(
        [("output", eager)],
        graph,
        atol=1e-2,
        rtol=0.05,
        min_cosine=0.999,
        max_rel_l2=0.01,
    )

    assert result["outputs"][0]["cosine_similarity"] == 1.0
    assert result["outputs"][0]["relative_l2"] == 1.0
    assert result["passed"] is False


def test_eager_fallback_warms_once_when_warmup_zero(monkeypatch) -> None:
    events: list[str] = []

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "triton.testing":
            raise ModuleNotFoundError("No module named 'triton'", name="triton")
        return real_import(name, *args, **kwargs)

    def fake_perf_counter() -> float:
        events.append("clock")
        return float(len(events))

    def bench_fn() -> torch.Tensor:
        events.append("bench")
        return torch.ones(1)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(performance_module.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(performance_module, "sync_device", lambda _device: events.append("sync"))

    result = performance_module._measure_eager_samples(
        bench_fn=bench_fn,
        device=torch.device("cpu"),
        warmup_iters=0,
        bench_iters=2,
    )

    assert result.samples
    assert events == ["bench", "sync", "clock", "bench", "bench", "sync", "clock"]


def test_benchmark_returns_timing() -> None:
    result = benchmark_performance(
        CANDIDATE_PATH,
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=3,
        device="cpu",
    )
    assert result.error is None
    assert len(result.samples) >= 1
    for sample in result.samples:
        assert sample.end_to_end_time_ms is not None
        assert sample.end_to_end_time_ms > 0


def test_benchmark_reference_torch_compile_returns_timing(monkeypatch) -> None:
    compiled_models = []

    def fake_compile(model):
        compiled_models.append(model)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)

    result = benchmark_reference_torch_compile(
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=2,
        device="cpu",
    )

    assert result.error is None
    assert compiled_models
    assert len(result.samples) >= 1
    for sample in result.samples:
        assert sample.end_to_end_time_ms is not None
        assert sample.end_to_end_time_ms > 0


def test_benchmark_reference_torch_compile_writes_seed_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch, "compile", lambda model: model)

    result = benchmark_reference_torch_compile(
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=1,
        device="cpu",
    )

    # Input artifact is now seed-only (no .pt file written).
    assert result.error is None
    assert result.input_artifact is not None
    assert result.input_artifact["format"] == "manual_seed"
    assert isinstance(result.input_artifact["seed"], int)
    assert "path" not in result.input_artifact


def test_benchmark_writes_seed_artifact(tmp_path: Path) -> None:
    reference_path = _write_python_file(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self, bias):",
                "        super().__init__()",
                "        self.bias = bias",
                "",
                "    def forward(self, x):",
                "        return x + self.bias",
                "",
                "def get_inputs():",
                "    return [torch.zeros(2, 2)]",
                "",
                "def get_init_inputs():",
                "    return [1.5]",
            ]
        ),
    )
    candidate_path = _write_python_file(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self, bias):",
                "        super().__init__()",
                "        self.bias = bias",
                "",
                "    def forward(self, x):",
                "        return x + self.bias",
                "",
                "def get_inputs():",
                "    return [torch.full((2, 2), 7.0)]",
                "",
                "def get_init_inputs():",
                "    return [9.0]",
            ]
        ),
    )
    result = benchmark_performance(
        candidate_path,
        reference_path,
        warmup_iters=1,
        bench_iters=1,
        device="cpu",
    )

    # Input artifact is now seed-only (no .pt file written).
    assert result.error is None
    assert result.input_artifact is not None
    assert result.input_artifact["format"] == "manual_seed"
    assert isinstance(result.input_artifact["seed"], int)
    assert "path" not in result.input_artifact


def test_benchmark_error_handled(tmp_path: Path) -> None:
    candidate_path = _write_python_file(
        tmp_path,
        "broken_candidate.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "        raise RuntimeError('init failure')",
                "",
                "    def forward(self, x):",
                "        return x",
            ]
        ),
    )

    result = benchmark_performance(
        candidate_path,
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=1,
        device="cpu",
    )
    assert result.samples == []
    assert result.error is not None
    assert "init failure" in result.error


def test_candidate_timeout_before_input_artifact_preserves_timeout_error(
    monkeypatch,
) -> None:
    def timeout_before_inputs(*_args, **_kwargs):
        raise performance_module.CandidateTimeoutError("candidate timed out during init")

    monkeypatch.setattr(
        performance_module,
        "instantiate_model_module",
        timeout_before_inputs,
    )

    result = benchmark_performance(
        CANDIDATE_PATH,
        REFERENCE_PATH,
        warmup_iters=1,
        bench_iters=1,
        device="cpu",
    )

    assert result.input_artifact is None
    assert result.error == "candidate timed out during init"
