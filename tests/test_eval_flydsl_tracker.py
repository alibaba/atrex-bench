"""Unit tests for the runtime flydsl ``@kernel`` tracker.

These tests don't depend on flydsl being installed in CI: we install a
synthetic ``flydsl.compiler.kernel_function`` module into ``sys.modules``
that mimics the real decorator's two call shapes (bare ``@kernel`` and
parametric ``@kernel(name=...)``), then drive the tracker against it.
"""
from __future__ import annotations

import sys
import types

import pytest

from atrex_bench.eval import _flydsl_tracker as tracker
from atrex_bench.eval._flydsl_tracker import (
    install_flydsl_kernel_tracker,
    observed_kernel_symbols,
    observed_kernel_symbols_serializable,
    reset_observations,
    symbols_from_serialized,
)


def _identity_kernel():
    """Build a fake flydsl ``kernel`` that supports both decorator forms.

    Mirrors flydsl's contract:
      * ``@kernel`` / ``kernel(func)`` returns ``func`` unchanged.
      * ``@kernel(name="x", ...)`` / ``kernel(name="x")`` returns a decorator
        that, when called with ``func``, returns ``func`` unchanged.
    """

    def kernel(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]
        def _decorator(func):
            return func
        return _decorator

    return kernel


def _kernel_function_backed_kernel():
    """Build a fake flydsl kernel exposing ``KernelFunction.__call__``."""

    class FakeKernelFunction:
        def __init__(self, func, *, name=None, some_args=None, known_block_size=None):
            self._func = func
            self._name = name
            self._some_args = some_args
            self._known_block_size = known_block_size
            self._kernel_id = 0
            self._kernel_name = None

        def __call__(self, *args, **kwargs):
            del args, kwargs
            if self._name is not None:
                self._kernel_name = self._name
            else:
                self._kernel_name = f"{self._func.__name__}_{self._kernel_id}"
                self._kernel_id += 1
            return types.SimpleNamespace(launch=lambda **_kwargs: None)

    def kernel(func=None, *, name=None, some_args=None, known_block_size=None):
        if func is None:
            return lambda f: FakeKernelFunction(
                f,
                name=name,
                some_args=some_args,
                known_block_size=known_block_size,
            )
        return FakeKernelFunction(
            func,
            name=name,
            some_args=some_args,
            known_block_size=known_block_size,
        )

    return kernel, FakeKernelFunction


@pytest.fixture
def fake_flydsl(monkeypatch):
    """Install a synthetic flydsl module hierarchy under sys.modules."""
    # Reset the tracker's installed-once flag so install_... actually runs
    # against our fake module instead of returning early.
    monkeypatch.setattr(tracker, "_INSTALLED", False, raising=True)
    reset_observations()

    fake_pkg = types.ModuleType("flydsl")
    fake_compiler = types.ModuleType("flydsl.compiler")
    fake_kf = types.ModuleType("flydsl.compiler.kernel_function")

    real_kernel = _identity_kernel()
    fake_kf.kernel = real_kernel
    fake_compiler.kernel = real_kernel
    fake_compiler.kernel_function = fake_kf
    fake_pkg.compiler = fake_compiler

    monkeypatch.setitem(sys.modules, "flydsl", fake_pkg)
    monkeypatch.setitem(sys.modules, "flydsl.compiler", fake_compiler)
    monkeypatch.setitem(sys.modules, "flydsl.compiler.kernel_function", fake_kf)

    yield fake_kf
    reset_observations()


@pytest.fixture
def fake_flydsl_kernel_function(monkeypatch):
    """Install a synthetic flydsl module with a patchable KernelFunction."""
    monkeypatch.setattr(tracker, "_INSTALLED", False, raising=True)
    reset_observations()

    fake_pkg = types.ModuleType("flydsl")
    fake_compiler = types.ModuleType("flydsl.compiler")
    fake_kf = types.ModuleType("flydsl.compiler.kernel_function")

    real_kernel, fake_kernel_function = _kernel_function_backed_kernel()
    fake_kf.kernel = real_kernel
    fake_kf.KernelFunction = fake_kernel_function
    fake_compiler.kernel = real_kernel
    fake_compiler.kernel_function = fake_kf
    fake_pkg.compiler = fake_compiler

    monkeypatch.setitem(sys.modules, "flydsl", fake_pkg)
    monkeypatch.setitem(sys.modules, "flydsl.compiler", fake_compiler)
    monkeypatch.setitem(sys.modules, "flydsl.compiler.kernel_function", fake_kf)

    yield fake_kf
    reset_observations()


def test_install_returns_false_when_flydsl_missing(monkeypatch):
    """Tracker must degrade gracefully on non-flydsl candidates."""
    monkeypatch.setattr(tracker, "_INSTALLED", False, raising=True)
    monkeypatch.setitem(sys.modules, "flydsl.compiler.kernel_function", None)
    # Also ensure plain `import flydsl.compiler.kernel_function` raises:
    for name in list(sys.modules):
        if name.startswith("flydsl"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    # Block any future import attempt by setting the parent to None too.
    monkeypatch.setitem(sys.modules, "flydsl", None)
    assert install_flydsl_kernel_tracker() is False


def test_install_is_idempotent(fake_flydsl):
    assert install_flydsl_kernel_tracker() is True
    first_wrapped = fake_flydsl.kernel
    # Second call must NOT re-wrap (would double-record).
    assert install_flydsl_kernel_tracker() is True
    assert fake_flydsl.kernel is first_wrapped


def test_bare_decorator_records_func_name(fake_flydsl):
    install_flydsl_kernel_tracker()
    kernel = fake_flydsl.kernel

    @kernel
    def _moe_kernel(): pass

    exact, patterns = observed_kernel_symbols()
    assert exact == set()
    assert len(patterns) == 1
    assert patterns[0].fullmatch("_moe_kernel_0")
    assert patterns[0].fullmatch("_moe_kernel_42")
    assert not patterns[0].fullmatch("_moe_kernel")


def test_parametric_decorator_records_explicit_name(fake_flydsl):
    install_flydsl_kernel_tracker()
    kernel = fake_flydsl.kernel

    @kernel(name="gdn_gate_attn", known_block_size=[128, 1, 1])
    def whatever(): pass

    exact, patterns = observed_kernel_symbols()
    assert exact == {"gdn_gate_attn"}
    # Explicit name overrides; no <func>_<id> pattern needed.
    assert patterns == []


def test_multiple_kernels_mixed_forms(fake_flydsl):
    install_flydsl_kernel_tracker()
    kernel = fake_flydsl.kernel

    @kernel(name="explicit_one")
    def k1(): pass

    @kernel
    def k2(): pass

    @kernel
    def k3(): pass

    exact, patterns = observed_kernel_symbols()
    assert exact == {"explicit_one"}
    pattern_strs = {p.pattern for p in patterns}
    assert pattern_strs == {r"^k2_\d+$", r"^k3_\d+$"}


def test_alias_import_still_records(fake_flydsl):
    """``from flydsl.compiler.kernel_function import kernel as _kernel``."""
    install_flydsl_kernel_tracker()
    # Simulate the candidate's local rebinding under an alias.
    from flydsl.compiler.kernel_function import kernel as _kernel

    @_kernel
    def _rope_kernel(): pass

    exact, patterns = observed_kernel_symbols()
    assert exact == set()
    assert any(p.fullmatch("_rope_kernel_7") for p in patterns)


def test_string_literal_kernel_source_generator(fake_flydsl):
    """Even when @kernel is built as text + exec'd, the decorator call lands here."""
    install_flydsl_kernel_tracker()
    src = (
        "from flydsl.compiler import kernel\n"
        "@kernel(name='attn_kernel')\n"
        "def attn_kernel(q, k, v, out):\n"
        "    pass\n"
    )
    ns: dict = {}
    exec(compile(src, "<string-gen>", "exec"), ns)
    exact, patterns = observed_kernel_symbols()
    assert exact == {"attn_kernel"}
    assert patterns == []


def test_serializable_round_trip(fake_flydsl):
    install_flydsl_kernel_tracker()
    kernel = fake_flydsl.kernel

    @kernel(name="op_a")
    def _ka(): pass

    @kernel
    def _kb(): pass

    payload = observed_kernel_symbols_serializable()
    assert payload == {"exact_names": ["op_a"], "func_names": ["_kb"]}

    # And the reverse helper round-trips:
    reconstructed = symbols_from_serialized(payload)
    assert reconstructed is not None
    exact, patterns = reconstructed
    assert exact == {"op_a"}
    assert len(patterns) == 1
    assert patterns[0].fullmatch("_kb_0")


def test_runtime_call_records_dynamic_kernel_name(fake_flydsl_kernel_function):
    install_flydsl_kernel_tracker()
    kernel = fake_flydsl_kernel_function.kernel

    @kernel
    def block_scaled_mm_kernel():
        pass

    block_scaled_mm_kernel._func.__name__ = "block_scaled_mm_fp8_t32x512x4096_wpe0"
    block_scaled_mm_kernel().launch()

    payload = observed_kernel_symbols_serializable()
    assert payload["exact_names"] == ["block_scaled_mm_fp8_t32x512x4096_wpe0_0"]
    assert payload["func_names"] == [
        "block_scaled_mm_fp8_t32x512x4096_wpe0",
        "block_scaled_mm_kernel",
    ]

    reconstructed = symbols_from_serialized(payload)
    assert reconstructed is not None
    exact, patterns = reconstructed
    assert "block_scaled_mm_fp8_t32x512x4096_wpe0_0" in exact
    assert any(
        p.fullmatch("block_scaled_mm_fp8_t32x512x4096_wpe0_1")
        for p in patterns
    )


def test_symbols_from_serialized_handles_none():
    assert symbols_from_serialized(None) is None


def test_symbols_from_serialized_handles_empty():
    exact, patterns = symbols_from_serialized({"exact_names": [], "func_names": []})
    assert exact == set()
    assert patterns == []


def test_reset_clears_observations(fake_flydsl):
    install_flydsl_kernel_tracker()
    kernel = fake_flydsl.kernel

    @kernel
    def k(): pass

    assert observed_kernel_symbols_serializable() == {
        "exact_names": [],
        "func_names": ["k"],
    }
    reset_observations()
    assert observed_kernel_symbols_serializable() == {
        "exact_names": [],
        "func_names": [],
    }


def test_wrapper_preserves_flydsl_return_value(fake_flydsl):
    """The wrapped @kernel must return whatever flydsl's kernel returned.

    Otherwise the candidate breaks: ``@kernel`` is expected to return the
    KernelFunction (or in our fake: the original function), and downstream
    code uses the returned object.
    """
    install_flydsl_kernel_tracker()
    kernel = fake_flydsl.kernel

    def original_fn(x):
        return x + 1

    decorated = kernel(original_fn)
    # Our fake returns the function as-is; the wrapped tracker MUST preserve that.
    assert decorated(3) == 4

    @kernel(name="custom")
    def another_fn(y):
        return y * 2

    assert another_fn(5) == 10
