"""Shared runtime helpers for Atrex-Bench evaluation."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import itertools
import json
import platform as platform_module
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


@dataclass(frozen=True)
class ShapeSpec:
    """A single shape entry loaded from shapes.json.

    ``init_kwargs`` is the dict (possibly empty) to pass to ``Model(**init_kwargs)``.
    ``input_kwargs`` is the dict to pass to ``_make_inputs(**input_kwargs)``.
    ``description`` is the free-form human-readable label from the shape entry.
    ``shape_id`` is the string key under which this shape lives in shapes.json.
    """

    shape_id: str
    description: str
    init_kwargs: dict[str, Any]
    input_kwargs: dict[str, Any]


@dataclass(frozen=True)
class ModelInputs:
    """Normalized positional and keyword arguments for a model call."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class LoadedModelPair:
    """Imported modules and instantiated models for one evaluation target."""

    reference_module: ModuleType
    candidate_module: ModuleType
    input_module: ModuleType
    reference_model: torch.nn.Module
    candidate_model: torch.nn.Module
    init_inputs: ModelInputs


@dataclass(frozen=True)
class LoadedModelModule:
    """Imported module and instantiated model for one evaluation target."""

    module: ModuleType
    model: torch.nn.Module
    init_inputs: ModelInputs


_MODULE_COUNTER = itertools.count()
_TARGET_DSLS = ("triton", "gluon", "flydsl", "cutedsl")
_CORE_PACKAGE_NAMES = (
    "torch",
    "triton",
    "gluon",
    "flydsl",
    "cutedsl",
    "numpy",
    "PyYAML",
    "vllm",
    "sglang",
    "flashinfer-python",
    "aiter",
)


def get_accelerator_backend() -> str | None:
    """Return the PyTorch GPU backend family, if any."""
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return None


def get_python_version() -> str:
    """Return the active Python version."""
    return platform_module.python_version()


def get_platform_label() -> str:
    """Return a compact platform identifier."""
    return f"{platform_module.system().lower()}-{platform_module.machine().lower()}"


def get_core_package_versions() -> dict[str, str]:
    """Collect versions for the core pip packages relevant to evaluation."""
    versions: dict[str, str] = {}
    for package_name in _CORE_PACKAGE_NAMES:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            continue
    return versions


def _score_dsl_mentions_from_source(source: str) -> dict[str, int]:
    """Score DSL mentions from source using AST first and raw-text fallback."""
    scores = {dsl: 0 for dsl in _TARGET_DSLS}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for dsl in _TARGET_DSLS:
                        if alias.name == dsl or alias.name.startswith(f"{dsl}."):
                            scores[dsl] += 3
            elif isinstance(node, ast.ImportFrom) and node.module:
                for dsl in _TARGET_DSLS:
                    if node.module == dsl or node.module.startswith(f"{dsl}."):
                        scores[dsl] += 3

    lowered = source.lower()
    for dsl in _TARGET_DSLS:
        if re.search(rf"(^|\n)\s*(from|import)\s+{dsl}\b", lowered, re.MULTILINE):
            scores[dsl] += 1
        if f"@{dsl}." in lowered:
            scores[dsl] += 1

    return scores


def infer_target_dsl(generated_path: Path) -> str:
    """Infer the candidate DSL from source, falling back to unknown."""
    try:
        source = generated_path.read_text(encoding="utf-8")
    except OSError:
        return "unknown"

    scores = _score_dsl_mentions_from_source(source)
    matched_dsls = [dsl for dsl, score in scores.items() if score > 0]
    if len(matched_dsls) == 1:
        return matched_dsls[0]
    return "unknown"


def _is_rocm_alias(device: str) -> bool:
    """Return whether the requested device uses a ROCm/HIP alias."""
    lowered = device.strip().lower()
    return (
        lowered == "hip"
        or lowered.startswith("hip:")
        or lowered == "rocm"
        or lowered.startswith("rocm:")
    )


def _normalize_device_request(device: str) -> str:
    """Normalize user-facing device aliases into torch.device-compatible strings."""
    lowered = device.strip().lower()
    if lowered in {"hip", "rocm"}:
        return "cuda"
    if lowered.startswith("hip:"):
        return f"cuda:{lowered.split(':', 1)[1]}"
    if lowered.startswith("rocm:"):
        return f"cuda:{lowered.split(':', 1)[1]}"
    return lowered


def get_device(device: str | None = None) -> torch.device:
    """Resolve the evaluation device."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized_device = _normalize_device_request(device)
    accelerator_backend = get_accelerator_backend()

    if _is_rocm_alias(device) and accelerator_backend != "rocm":
        raise RuntimeError(
            "HIP/ROCm was requested, but this PyTorch build is not ROCm-enabled."
        )

    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        if accelerator_backend == "rocm":
            raise RuntimeError(
                "A ROCm/HIP GPU was requested but is not available. "
                "PyTorch ROCm uses the 'cuda' device namespace internally; "
                "check torch.cuda.is_available(), ROCm runtime setup, and GPU visibility."
            )
        raise RuntimeError("CUDA was requested but is not available.")

    if device is not None:
        return torch.device(normalized_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_device(device: torch.device) -> None:
    """Synchronize the target device when required."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def import_module_from_path(module_path: Path, module_prefix: str) -> ModuleType:
    """Import a Python module from a filesystem path under a unique module name."""
    if not module_path.exists():
        raise FileNotFoundError(f"Module path does not exist: {module_path}")

    module_name = f"{module_prefix}_{next(_MODULE_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_reference_module(module: ModuleType) -> None:
    """Validate the reference.py contract: Model class only.

    Input generation lives in input.py next to reference.py (resolved via
    resolve_input_module). For ad-hoc / single-file references that still
    bundle get_inputs / get_init_inputs in the reference module, the resolver
    falls back to using the reference module itself.
    """
    if not hasattr(module, "Model"):
        raise AttributeError("Reference module does not define 'Model'")
    if not inspect.isclass(module.Model):
        raise TypeError("Reference symbol 'Model' must be a class")


def validate_input_module(module: ModuleType) -> None:
    """Validate the input.py contract: callable ``_make_inputs``.

    The legacy get_inputs / get_init_inputs wrappers were removed.
    Consumers now source init/call kwargs from shapes.json and drive
    ``_make_inputs(**input_kwargs)`` directly. Synthetic inline test
    references that still ship get_inputs/get_init_inputs go through the
    reference-module fallback path in ``resolve_input_module`` and are
    therefore not subject to this validator.
    """
    if not callable(getattr(module, "_make_inputs", None)):
        raise AttributeError(
            "Input module does not define callable '_make_inputs(**kwargs)'"
        )


def load_shape_init_inputs(shape: ShapeSpec, device: torch.device) -> ModelInputs:
    """Build init inputs (ModelInputs) for Model(**init_kwargs) from a shape spec.

    ``shape.init_kwargs`` is already a dict; this wraps it in the canonical
    ModelInputs container and moves it to the target device (in case any value
    is a tensor, which is uncommon but supported).
    """
    return prepare_model_inputs(shape.init_kwargs, device)


def load_shape_call_inputs(
    input_module: ModuleType,
    shape: ShapeSpec,
    device: torch.device,
) -> ModelInputs:
    """Build call inputs (ModelInputs) by invoking ``_make_inputs(**input_kwargs)``.

    The returned ModelInputs carries the tensor dict as kwargs (args=()),
    so callers drive forward with ``model(**inputs.kwargs)``.
    """
    if not callable(getattr(input_module, "_make_inputs", None)):
        raise AttributeError(
            "Input module does not define callable '_make_inputs(**kwargs)'"
        )
    raw = input_module._make_inputs(**shape.input_kwargs)
    if not isinstance(raw, dict):
        raise TypeError(
            f"_make_inputs must return dict[str, Tensor], got {type(raw).__name__}"
        )
    return prepare_model_inputs(raw, device)


def _shape_description_from_metadata(op_dir: Path, shape_id: str) -> str:
    """Read a shape's description from metadata.json (not agent-visible).

    Per the data schema spec, ``description`` and provenance live in
    ``metadata.json.shapes[shape_id]`` (kept out of the agent-visible
    ``shapes.json``). The eval/report side runs in the repo, where metadata.json
    is available, so the per-shape label is recovered here. Returns "" if absent.
    """
    md_path = op_dir / "metadata.json"
    if not md_path.is_file():
        return ""
    try:
        md = json.loads(md_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    shapes = md.get("shapes")
    if not isinstance(shapes, dict):
        return ""
    entry = shapes.get(shape_id)
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("description", ""))


def load_shape_spec(reference_path: Path, shape_id: str = "0") -> ShapeSpec:
    """Load one shape entry from shapes.json next to reference.py.

    shapes.json is a dict keyed by string shape id. Each entry has
    ``init_kwargs`` (dict or null) and ``input_kwargs`` (dict). The
    ``description``/provenance live in metadata.json (not agent-visible) and are
    recovered via ``_shape_description_from_metadata``. Returns a ShapeSpec that
    encodes the lookup result; ``init_kwargs`` is coerced to an empty dict when
    the source value is null (per schema §2.3).

    Raises FileNotFoundError if shapes.json is missing; KeyError if the
    shape_id is not declared; TypeError if the schema is malformed.
    """
    shapes_path = reference_path.parent / "shapes.json"
    if not shapes_path.is_file():
        raise FileNotFoundError(f"shapes.json not found next to {reference_path}")
    raw = json.loads(shapes_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"shapes.json top level must be a dict: {shapes_path}")
    if shape_id not in raw:
        raise KeyError(
            f"shape id {shape_id!r} not declared in {shapes_path} "
            f"(available: {sorted(raw.keys())})"
        )
    entry = raw[shape_id]
    if not isinstance(entry, dict):
        raise TypeError(f"shapes.json[{shape_id!r}] must be a dict: {shapes_path}")
    init_kwargs = entry.get("init_kwargs") or {}
    input_kwargs = entry.get("input_kwargs") or {}
    if not isinstance(init_kwargs, dict):
        raise TypeError(
            f"shapes.json[{shape_id!r}].init_kwargs must be a dict or null: {shapes_path}"
        )
    if not isinstance(input_kwargs, dict):
        raise TypeError(
            f"shapes.json[{shape_id!r}].input_kwargs must be a dict: {shapes_path}"
        )
    description = entry.get("description")
    if not description:
        description = _shape_description_from_metadata(shapes_path.parent, shape_id)
    return ShapeSpec(
        shape_id=shape_id,
        description=str(description or ""),
        init_kwargs=dict(init_kwargs),
        input_kwargs=dict(input_kwargs),
    )


def resolve_input_module(
    reference_path: Path,
    reference_module: ModuleType,
    *,
    module_prefix: str = "atrex_input",
) -> ModuleType:
    """Resolve the input provider module for a reference.

    Resolution order:
        1. ``input.py`` next to reference_path (the new operator-directory layout).
        2. ``reference_module`` itself if it still defines get_inputs and
           get_init_inputs (legacy single-file pattern, kept so synthetic
           test fixtures and ad-hoc estimate callers continue to work).
        3. FileNotFoundError otherwise.
    """
    input_path = reference_path.parent / "input.py"
    if input_path.is_file():
        module = import_module_from_path(input_path, module_prefix)
        validate_input_module(module)
        return module
    if callable(getattr(reference_module, "get_inputs", None)) and callable(
        getattr(reference_module, "get_init_inputs", None)
    ):
        return reference_module
    raise FileNotFoundError(
        "No input provider for reference: expected "
        f"{input_path} or get_inputs/get_init_inputs on the reference module"
    )


def validate_candidate_module(module: ModuleType) -> None:
    """Validate the required candidate module contract."""
    if not hasattr(module, "Model"):
        raise AttributeError("Candidate module does not define 'Model'")
    if not inspect.isclass(module.Model):
        raise TypeError("Candidate symbol 'Model' must be a class")


def normalize_model_inputs(raw_inputs: Any) -> ModelInputs:
    """Normalize model inputs into explicit args/kwargs containers."""
    if raw_inputs is None:
        return ModelInputs(args=(), kwargs={})
    if isinstance(raw_inputs, ModelInputs):
        return raw_inputs
    if isinstance(raw_inputs, dict):
        if "args" in raw_inputs or "kwargs" in raw_inputs:
            args = tuple(raw_inputs.get("args", ()))
            kwargs = dict(raw_inputs.get("kwargs", {}))
            return ModelInputs(args=args, kwargs=kwargs)
        return ModelInputs(args=(), kwargs=dict(raw_inputs))
    if isinstance(raw_inputs, tuple):
        return ModelInputs(args=raw_inputs, kwargs={})
    if isinstance(raw_inputs, list):
        return ModelInputs(args=tuple(raw_inputs), kwargs={})
    return ModelInputs(args=(raw_inputs,), kwargs={})


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors to the target device."""
    if isinstance(value, ModelInputs):
        return ModelInputs(
            args=move_to_device(value.args, device),
            kwargs=move_to_device(value.kwargs, device),
        )
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def clone_value(value: Any) -> Any:
    """Recursively clone tensors while preserving the overall structure."""
    if isinstance(value, ModelInputs):
        return ModelInputs(
            args=clone_value(value.args),
            kwargs=clone_value(value.kwargs),
        )
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def clone_model_inputs(inputs: ModelInputs) -> ModelInputs:
    """Clone a ModelInputs payload for a fresh model invocation."""
    return ModelInputs(
        args=tuple(clone_value(item) for item in inputs.args),
        kwargs={key: clone_value(item) for key, item in inputs.kwargs.items()},
    )


def summarize_value(value: Any) -> Any:
    """Build a JSON-serializable summary of nested inputs or outputs."""
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, list):
        return [summarize_value(item) for item in value]
    if isinstance(value, tuple):
        return [summarize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): summarize_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def summarize_model_inputs(inputs: ModelInputs) -> dict[str, Any]:
    """Build a JSON-friendly input signature."""
    return {
        "args": summarize_value(list(inputs.args)),
        "kwargs": summarize_value(inputs.kwargs),
    }


def flatten_outputs(value: Any, prefix: str = "output") -> list[tuple[str, torch.Tensor]]:
    """Flatten nested tensor outputs while preserving their logical path."""
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, (bool, int, float)):
        return [(prefix, torch.as_tensor(value))]
    if isinstance(value, tuple):
        tensors: list[tuple[str, torch.Tensor]] = []
        for index, item in enumerate(value):
            tensors.extend(flatten_outputs(item, f"{prefix}[{index}]"))
        return tensors
    if isinstance(value, list):
        tensors = []
        for index, item in enumerate(value):
            tensors.extend(flatten_outputs(item, f"{prefix}[{index}]"))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for key in sorted(value):
            tensors.extend(flatten_outputs(value[key], f"{prefix}.{key}"))
        return tensors
    raise TypeError(f"Unsupported output type: {type(value).__name__}")


def infer_operator_id(reference_path: Path) -> str:
    """Infer the operator id from the reference path."""
    if reference_path.name == "reference.py":
        return reference_path.parent.name
    return reference_path.stem


def prepare_model_inputs(raw_inputs: Any, device: torch.device) -> ModelInputs:
    """Normalize, clone, and move model inputs to the evaluation device."""
    normalized_inputs = normalize_model_inputs(raw_inputs)
    return ModelInputs(
        args=tuple(move_to_device(clone_value(item), device) for item in normalized_inputs.args),
        kwargs={
            key: move_to_device(clone_value(item), device)
            for key, item in normalized_inputs.kwargs.items()
        },
    )


def load_model_inputs(module: ModuleType, device: torch.device) -> ModelInputs:
    """Load one input case from get_inputs()."""
    return prepare_model_inputs(module.get_inputs(), device)


def load_reference_inputs(module: ModuleType, device: torch.device) -> ModelInputs:
    """Load one reference input case from get_inputs()."""
    return load_model_inputs(module, device)


def load_model_init_inputs(module: ModuleType, device: torch.device) -> ModelInputs:
    """Load model initialization inputs from get_init_inputs()."""
    return prepare_model_inputs(module.get_init_inputs(), device)


def load_init_inputs(module: ModuleType, device: torch.device) -> ModelInputs:
    """Load model initialization inputs from get_init_inputs()."""
    return load_model_init_inputs(module, device)


def _materialize_value_for_artifact(value: Any) -> Any:
    """Clone nested values for torch.save-based replay artifacts."""
    if isinstance(value, ModelInputs):
        return {
            "args": tuple(_materialize_value_for_artifact(item) for item in value.args),
            "kwargs": {
                key: _materialize_value_for_artifact(item)
                for key, item in value.kwargs.items()
            },
        }
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [_materialize_value_for_artifact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_value_for_artifact(item) for item in value)
    if isinstance(value, dict):
        return {key: _materialize_value_for_artifact(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _file_sha256(path: Path) -> str:
    """Compute the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SEED_MAX_INT32 = 0x7FFFFFFF


def deterministic_input_seed(stage: str, shape_id: str, case_index: int = 0) -> int:
    """Stable per-(stage, shape_id, case_index) seed for reproducible inputs.

    Used in place of saving the full input tensors as .pt files. Replay a
    case by setting torch's RNG to this seed and re-running
    ``_make_inputs(**input_kwargs)`` from the shape's input.py. The bench
    harness calls ``seed_all_input_rngs(seed)`` immediately before loading
    inputs so every random tensor in the input module is reproducible.

    Returns a positive int32 (keeps eval_result.json values within the
    range every downstream JSON consumer can represent).
    """
    key = f"{stage}|{shape_id}|{case_index}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") & _SEED_MAX_INT32


def seed_all_input_rngs(seed: int) -> None:
    """Seed every RNG that an input.py is likely to draw from.

    Production input modules use only ``torch.*`` RNGs (verified via repo
    scan); the random / numpy seeds are belt-and-suspenders for ad-hoc
    references that happen to also use them. Cheap to set, prevents
    accidental non-determinism if an input.py is rewritten.
    """
    import random as _random

    _random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_input_artifact(
    init_inputs: ModelInputs,
    call_inputs: ModelInputs,
    artifact_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, str]:
    """Persist exact init/call inputs for later replay.

    .. deprecated::
       The bench pipeline no longer calls this — input reproducibility is
       now handled by ``deterministic_input_seed`` + ``seed_all_input_rngs``
       (4 bytes in eval_result.json vs. multi-GB .pt files per shape). The
       function is retained for any out-of-tree callers but produces no
       output for the eval flow.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "init_inputs": _materialize_value_for_artifact(init_inputs),
            "call_inputs": _materialize_value_for_artifact(call_inputs),
        },
        artifact_path,
    )
    if artifact_root is None:
        recorded_path = str(artifact_path)
    else:
        try:
            recorded_path = str(artifact_path.relative_to(artifact_root))
        except ValueError:
            recorded_path = str(artifact_path)
    return {
        "path": recorded_path,
        "sha256": _file_sha256(artifact_path),
        "format": "torch.save",
    }


def summarize_replay_inputs(
    init_inputs: ModelInputs,
    call_inputs: ModelInputs,
    *,
    artifact: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the JSON-facing replay summary for one model invocation."""
    summary: dict[str, Any] = {
        "init": summarize_model_inputs(init_inputs),
        "call": summarize_model_inputs(call_inputs),
    }
    if artifact is not None:
        summary["artifact"] = artifact
    return summary


def instantiate_model_module(
    module_path: Path,
    device: torch.device,
    module_prefix: str,
    *,
    init_inputs: ModelInputs | None = None,
) -> LoadedModelModule:
    """Import and instantiate a single candidate-style module."""
    module = import_module_from_path(module_path, module_prefix)
    validate_candidate_module(module)
    resolved_init_inputs = (
        clone_model_inputs(init_inputs)
        if init_inputs is not None
        else load_model_init_inputs(module, device)
    )
    model_inputs = clone_model_inputs(resolved_init_inputs)
    model = module.Model(*model_inputs.args, **model_inputs.kwargs).to(device).eval()
    return LoadedModelModule(
        module=module,
        model=model,
        init_inputs=resolved_init_inputs,
    )


def instantiate_model_pair(
    reference_path: Path,
    candidate_path: Path,
    device: torch.device,
    module_prefix: str,
    *,
    shape_id: str | None = None,
    candidate_timeout_s: int | float | None = None,
) -> LoadedModelPair:
    """Import reference/candidate modules and instantiate both models.

    Auto-selects between two paths:

    * **New schema** (the data schema spec, Section 3): when a sibling ``shapes.json``
      exists, loads init_kwargs from it. Pass ``shape_id`` to pick a specific
      entry; defaults to ``"0"`` (the first shape).
    * **Legacy fallback**: when ``shapes.json`` is missing, reads init inputs
      via ``input_module.get_init_inputs()``. Kept so synthetic test
      references that bundle get_inputs/get_init_inputs in the reference
      module keep working during the migration.

    ``candidate_timeout_s`` bounds candidate-side import + instantiation
    (notably flydsl's ``@kernel`` AOT compile). The reference is never
    timed out. SIGALRM-based; for hangs inside C extensions (MLIR
    compiler, wedged GPU sync) the caller's OS-level wall budget is the
    actual catch-all.
    """
    from atrex_bench.eval._timeout import candidate_timeout

    reference_module = import_module_from_path(reference_path, f"{module_prefix}_reference")
    validate_reference_module(reference_module)
    input_module = resolve_input_module(
        reference_path,
        reference_module,
        module_prefix=f"{module_prefix}_input",
    )
    shapes_path = reference_path.parent / "shapes.json"
    if shapes_path.is_file():
        shape = load_shape_spec(reference_path, shape_id or "0")
        reference_init_inputs = load_shape_init_inputs(shape, device)
    else:
        reference_init_inputs = load_model_init_inputs(input_module, device)
    with candidate_timeout(candidate_timeout_s):
        loaded_candidate = instantiate_model_module(
            candidate_path,
            device,
            f"{module_prefix}_candidate",
            init_inputs=reference_init_inputs,
        )

    init_inputs = loaded_candidate.init_inputs
    reference_model_inputs = clone_model_inputs(init_inputs)

    reference_model = reference_module.Model(
        *reference_model_inputs.args,
        **reference_model_inputs.kwargs,
    ).to(device).eval()

    return LoadedModelPair(
        reference_module=reference_module,
        candidate_module=loaded_candidate.module,
        input_module=input_module,
        reference_model=reference_model,
        candidate_model=loaded_candidate.model,
        init_inputs=init_inputs,
    )
