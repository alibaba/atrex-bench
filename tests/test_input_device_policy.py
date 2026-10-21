"""Factories may own mixed host/device inputs without changing legacy defaults."""

import json
from types import ModuleType

import pytest
import torch

from atrex_bench.eval._runtime import ShapeSpec, load_shape_call_inputs


def test_preserve_policy_keeps_device_layout_and_aliases(tmp_path):
    module = ModuleType("input")
    module.__file__ = str(tmp_path / "input.py")
    storage = torch.arange(12)
    view = storage[1:9:2]
    module._make_inputs = lambda: {"control": view, "alias": view}
    shape = ShapeSpec(shape_id="0", description="", init_kwargs={}, input_kwargs={})
    (tmp_path / "metadata.json").write_text(
        json.dumps({"benchmark_contract": {"input_device_policy": "preserve"}})
    )
    result = load_shape_call_inputs(module, shape, torch.device("meta")).kwargs
    assert result["control"].device.type == "cpu"
    assert result["control"].stride() == (2,) and result["control"].storage_offset() == 1
    assert result["control"].untyped_storage()._cdata == result["alias"].untyped_storage()._cdata
    assert result["control"].untyped_storage().nbytes() == storage.untyped_storage().nbytes()
    assert result["control"].untyped_storage()._cdata != storage.untyped_storage()._cdata
    assert torch.equal(result["control"], view)
    (tmp_path / "metadata.json").unlink()
    assert (
        load_shape_call_inputs(module, shape, torch.device("meta")).kwargs["control"].device.type
        == "meta"
    )


def test_unknown_device_policy_is_rejected(tmp_path):
    module = ModuleType("input")
    module.__file__ = str(tmp_path / "input.py")
    module._make_inputs = lambda: {"x": torch.ones(2)}
    (tmp_path / "metadata.json").write_text(
        json.dumps({"benchmark_contract": {"input_device_policy": "typo"}})
    )
    shape = ShapeSpec(shape_id="0", description="", init_kwargs={}, input_kwargs={})
    with pytest.raises(ValueError, match="input_device_policy"):
        load_shape_call_inputs(module, shape, torch.device("cpu"))
