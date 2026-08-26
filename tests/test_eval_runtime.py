"""Tests for shared runtime device handling."""

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import device as torch_device

import atrex_bench.eval._runtime as runtime_module
from scripts import run_eval as run_eval_module


def test_clone_value_preserves_non_contiguous_strided_tensor_layout() -> None:
    source = torch.empty_strided((5, 2), (1, 8), dtype=torch.int32)
    source.copy_(torch.arange(source.numel(), dtype=source.dtype).reshape(source.shape))

    cloned = runtime_module.clone_value(source)

    assert cloned is not source
    assert cloned.shape == source.shape
    assert cloned.stride() == source.stride()
    assert torch.equal(cloned, source)


def test_clone_value_keeps_expanded_tensor_values_without_aliasing() -> None:
    source = torch.arange(2, dtype=torch.float32).reshape(1, 2).expand(5, 2)

    cloned = runtime_module.clone_value(source)

    assert cloned is not source
    assert cloned.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert torch.equal(cloned, source)


def test_clone_value_densifies_internally_overlapped_strided_tensor() -> None:
    source = torch.arange(4).as_strided((2, 2), (1, 1))

    cloned = runtime_module.clone_value(source)

    assert cloned is not source
    assert cloned.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert cloned.stride() != source.stride()
    assert torch.equal(cloned, source)


def test_clone_value_keeps_sparse_tensor_clone_support() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        source = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]).to_sparse_csr()

    cloned = runtime_module.clone_value(source)

    assert cloned is not source
    assert cloned.layout == source.layout
    assert cloned.values().untyped_storage().data_ptr() != (
        source.values().untyped_storage().data_ptr()
    )
    assert torch.equal(cloned.to_dense(), source.to_dense())


def test_clone_value_keeps_nested_tensor_clone_support() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        source = torch.nested.nested_tensor([torch.arange(2), torch.arange(3)])

    cloned = runtime_module.clone_value(source)

    assert cloned is not source
    assert cloned.is_nested
    for cloned_item, source_item in zip(cloned.unbind(), source.unbind()):
        assert cloned_item.untyped_storage().data_ptr() != (
            source_item.untyped_storage().data_ptr()
        )
        assert torch.equal(cloned_item, source_item)


def _patch_gpu_backend(
    monkeypatch,
    *,
    cuda_available: bool,
    hip_version: str | None = None,
    cuda_version: str | None = None,
) -> None:
    version = SimpleNamespace(cuda=cuda_version, hip=hip_version)
    monkeypatch.setattr(runtime_module.torch, "version", version)
    monkeypatch.setattr(runtime_module.torch.cuda, "is_available", lambda: cuda_available)


def test_get_device_accepts_hip_alias_on_rocm(monkeypatch) -> None:
    _patch_gpu_backend(monkeypatch, cuda_available=True, hip_version="6.4.1")

    assert runtime_module.get_device("hip") == torch_device("cuda")
    assert runtime_module.get_device("hip:2") == torch_device("cuda:2")
    assert runtime_module.get_device("rocm:3") == torch_device("cuda:3")


def test_get_device_rejects_hip_alias_without_rocm_build(monkeypatch) -> None:
    _patch_gpu_backend(monkeypatch, cuda_available=True, cuda_version="12.4")

    with pytest.raises(RuntimeError, match="ROCm"):
        runtime_module.get_device("hip")


def test_get_device_reports_rocm_unavailability_for_cuda_namespace(monkeypatch) -> None:
    _patch_gpu_backend(monkeypatch, cuda_available=False, hip_version="6.4.1")

    with pytest.raises(RuntimeError, match="ROCm"):
        runtime_module.get_device("cuda")


def test_build_environment_reports_rocm_backend(monkeypatch) -> None:
    _patch_gpu_backend(monkeypatch, cuda_available=True, hip_version="6.4.1")
    monkeypatch.setattr(run_eval_module, "_gpu_info", lambda: ("AMD Test GPU", "gfx000"))
    monkeypatch.setattr(run_eval_module, "_driver_version", lambda: "6.4.1.40")
    monkeypatch.setattr(run_eval_module, "get_python_version", lambda: "3.12.7")
    monkeypatch.setattr(
        run_eval_module,
        "get_core_package_versions",
        lambda: {"torch": "2.9.1+rocm", "triton": "3.2.0"},
    )
    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")

    environment = run_eval_module._build_environment(clock_locked=True)

    assert environment["accelerator_backend"] == "rocm"
    assert environment["gpu_name"] == "AMD Test GPU"
    assert environment["gpu_arch"] == "gfx000"
    assert environment["python_version"] == "3.12.7"
    assert environment["torch_version"] == "2.9.1+rocm"
    assert environment["triton_version"] == "3.2.0"
    assert environment["clock_locked"] is True
    assert environment["runtime_version"] == "6.4.1"
    assert environment["driver_version"] == "6.4.1.40"
    # The schema deliberately drops "device"/"platform"/"packages".
    assert "device" not in environment
    assert "platform" not in environment
    assert "packages" not in environment


def _patch_cuda_props(monkeypatch, *, name: str, props) -> None:
    """Replace torch.cuda.{is_available,current_device,get_device_name,get_device_properties}.

    Used by the _gpu_info() tests to simulate a specific GPU without needing
    real CUDA/ROCm hardware.
    """
    monkeypatch.setattr(run_eval_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_eval_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(run_eval_module.torch.cuda, "get_device_name", lambda idx: name)
    monkeypatch.setattr(
        run_eval_module.torch.cuda, "get_device_properties", lambda idx: props
    )


def test_gpu_info_returns_sm_arch_on_nvidia_even_when_gcn_arch_name_is_set(monkeypatch) -> None:
    """PyTorch 2.9+ may populate gcnArchName with the NVIDIA device name.

    The fix routes by accelerator_backend so the NVIDIA branch always uses
    sm_<major><minor> instead of falling for that confusingly-populated
    AMD-only attribute.
    """
    nvidia_props = SimpleNamespace(
        gcnArchName="NVIDIA Test GPU",  # PyTorch 2.9 cross-vendor compat shim
        major=0,  # Synthetic architecture used only by this mock.
        minor=0,
    )
    _patch_cuda_props(monkeypatch, name="NVIDIA Test GPU", props=nvidia_props)
    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "cuda")

    name, arch = run_eval_module._gpu_info()

    assert name == "NVIDIA Test GPU"
    assert arch == "sm_00"


def test_gpu_info_returns_gcn_arch_on_rocm(monkeypatch) -> None:
    rocm_props = SimpleNamespace(
        gcnArchName="gfx000:sramecc+:xnack-",
        major=0,
        minor=0,
    )
    _patch_cuda_props(monkeypatch, name="AMD Test GPU", props=rocm_props)
    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")

    name, arch = run_eval_module._gpu_info()

    assert name == "AMD Test GPU"
    assert arch == "gfx000:sramecc+:xnack-"


def test_gpu_info_returns_none_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(run_eval_module.torch.cuda, "is_available", lambda: False)

    assert run_eval_module._gpu_info() == (None, None)


def test_build_environment_omits_optional_dsl_versions_when_absent(monkeypatch) -> None:
    _patch_gpu_backend(monkeypatch, cuda_available=True, hip_version="6.4.1")
    monkeypatch.setattr(run_eval_module, "_gpu_info", lambda: (None, None))
    monkeypatch.setattr(run_eval_module, "_driver_version", lambda: None)
    monkeypatch.setattr(run_eval_module, "get_python_version", lambda: "3.12.7")
    monkeypatch.setattr(run_eval_module, "get_core_package_versions", lambda: {})
    monkeypatch.setattr(run_eval_module, "get_accelerator_backend", lambda: "rocm")

    environment = run_eval_module._build_environment(clock_locked=False)

    assert environment["accelerator_backend"] == "rocm"
    assert environment["clock_locked"] is False
    for dsl in ("triton", "gluon", "flydsl", "cutedsl"):
        assert f"{dsl}_version" not in environment


def _write_source(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_infer_target_dsl_detects_triton_source(tmp_path: Path) -> None:
    generated_path = _write_source(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import triton",
                "import triton.language as tl",
                "",
                "@triton.jit",
                "def kernel(x_ptr):",
                "    return",
            ]
        ),
    )

    assert runtime_module.infer_target_dsl(generated_path) == "triton"


def test_infer_target_dsl_detects_gluon_source(tmp_path: Path) -> None:
    generated_path = _write_source(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import gluon",
                "",
                "PROGRAM = gluon.compile('fake')",
            ]
        ),
    )

    assert runtime_module.infer_target_dsl(generated_path) == "gluon"


def test_infer_target_dsl_detects_flydsl_source(tmp_path: Path) -> None:
    generated_path = _write_source(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import flydsl",
                "",
                "PROGRAM = flydsl.compile('fake')",
            ]
        ),
    )

    assert runtime_module.infer_target_dsl(generated_path) == "flydsl"


def test_infer_target_dsl_detects_cutedsl_source(tmp_path: Path) -> None:
    generated_path = _write_source(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import cutedsl",
                "",
                "PROGRAM = cutedsl.compile('fake')",
            ]
        ),
    )

    assert runtime_module.infer_target_dsl(generated_path) == "cutedsl"


def test_infer_target_dsl_detects_cutedsl_by_its_real_import_path(
    tmp_path: Path,
) -> None:
    """CuteDSL is imported as ``cutlass.cute``, never as ``cutedsl``.

    Matching the bare DSL name scored nothing here, so real CuteDSL candidates
    came back as something else (observed on two of them in a live run).
    """
    generated_path = _write_source(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import torch",
                "import cutlass.cute as cute",
                "",
                "@cute.kernel",
                "def kernel(x):",
                "    return x",
            ]
        ),
    )

    assert runtime_module.infer_target_dsl(generated_path) == "cutedsl"


def test_infer_target_dsl_prefers_gluon_over_triton_on_its_own_import(
    tmp_path: Path,
) -> None:
    """Gluon lives under ``triton.experimental.gluon``.

    The bare-name match credited the enclosing Triton package and nothing to
    Gluon, so every Gluon candidate reported as Triton. Longest prefix wins.
    """
    generated_path = _write_source(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import triton.experimental.gluon as gluon",
                "",
                "@gluon.jit",
                "def kernel(x_ptr):",
                "    return",
            ]
        ),
    )

    assert runtime_module.infer_target_dsl(generated_path) == "gluon"


def test_infer_target_dsl_lets_a_kernel_definition_outweigh_a_stray_import(
    tmp_path: Path,
) -> None:
    """Defining a kernel beats merely importing another DSL.

    Both DSLs used to score nonzero and the verdict fell through to unknown.
    """
    generated_path = _write_source(
        tmp_path,
        "candidate.py",
        "\n".join(
            [
                "import triton",
                "import cutlass.cute as cute",
                "",
                "@cute.kernel",
                "def kernel(x):",
                "    return x",
            ]
        ),
    )

    assert runtime_module.infer_target_dsl(generated_path) == "cutedsl"


def test_infer_target_dsl_returns_unknown_for_plain_pytorch_fixture() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "generations" / "atrex_001.py"

    assert runtime_module.infer_target_dsl(fixture_path) == "unknown"


def test_infer_target_dsl_falls_back_to_text_scan_on_syntax_error(tmp_path: Path) -> None:
    generated_path = _write_source(
        tmp_path,
        "broken_candidate.py",
        "\n".join(
            [
                "import triton",
                "",
                "def broken(",
            ]
        ),
    )

    assert runtime_module.infer_target_dsl(generated_path) == "triton"


def _write_reference_with_input(tmp_path: Path) -> Path:
    """Write a Model-only reference.py plus an input.py next to it."""
    reference_path = _write_source(
        tmp_path,
        "reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return x",
            ]
        ),
    )
    _write_source(
        tmp_path,
        "input.py",
        "\n".join(
            [
                "import torch",
                "",
                "def _make_inputs():",
                "    return {'x': torch.zeros(2, 2)}",
            ]
        ),
    )
    return reference_path


def test_validate_reference_module_accepts_model_only(tmp_path: Path) -> None:
    """The new contract: reference.py only needs a Model class."""
    reference_path = _write_reference_with_input(tmp_path)
    module = runtime_module.import_module_from_path(reference_path, "atrex_test_ref")
    runtime_module.validate_reference_module(module)


def test_validate_reference_module_rejects_module_without_model(tmp_path: Path) -> None:
    reference_path = _write_source(
        tmp_path,
        "no_model.py",
        "VALUE = 1\n",
    )
    module = runtime_module.import_module_from_path(reference_path, "atrex_test_no_model")
    with pytest.raises(AttributeError, match="does not define 'Model'"):
        runtime_module.validate_reference_module(module)


def test_validate_input_module_requires_make_inputs(tmp_path: Path) -> None:
    incomplete_path = _write_source(
        tmp_path,
        "input.py",
        "def something_else():\n    return []\n",
    )
    module = runtime_module.import_module_from_path(incomplete_path, "atrex_test_input_bad")
    with pytest.raises(AttributeError, match="_make_inputs"):
        runtime_module.validate_input_module(module)


def test_resolve_input_module_prefers_sibling_input_py(tmp_path: Path) -> None:
    """When input.py exists next to reference.py, resolver returns it."""
    reference_path = _write_reference_with_input(tmp_path)
    reference_module = runtime_module.import_module_from_path(
        reference_path, "atrex_test_ref_resolve"
    )
    input_module = runtime_module.resolve_input_module(
        reference_path, reference_module, module_prefix="atrex_test_input_resolve"
    )
    assert input_module is not reference_module
    assert input_module._make_inputs()["x"].shape == (2, 2)


def test_resolve_input_module_falls_back_to_inline_get_inputs(tmp_path: Path) -> None:
    """Single-file references (legacy / synthetic) resolve to themselves."""
    reference_path = _write_source(
        tmp_path,
        "single_file_reference.py",
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return x",
                "",
                "def get_inputs():",
                "    return [torch.zeros(3, 3)]",
                "",
                "def get_init_inputs():",
                "    return []",
            ]
        ),
    )
    reference_module = runtime_module.import_module_from_path(
        reference_path, "atrex_test_inline_ref"
    )
    input_module = runtime_module.resolve_input_module(
        reference_path, reference_module, module_prefix="atrex_test_inline_input"
    )
    assert input_module is reference_module


def test_resolve_input_module_raises_when_no_provider(tmp_path: Path) -> None:
    reference_path = _write_source(
        tmp_path,
        "model_only.py",
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "class Model(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "    def forward(self, x):",
                "        return x",
            ]
        ),
    )
    reference_module = runtime_module.import_module_from_path(
        reference_path, "atrex_test_model_only"
    )
    with pytest.raises(FileNotFoundError, match="No input provider"):
        runtime_module.resolve_input_module(
            reference_path, reference_module, module_prefix="atrex_test_no_input"
        )
