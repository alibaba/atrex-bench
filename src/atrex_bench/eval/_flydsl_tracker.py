"""Runtime tracker for flydsl ``@kernel`` decorations.

Monkey-patches ``flydsl.compiler.kernel_function.kernel`` so every call
records the decorated function's emission name (or ``<func_name>_<id>``
pattern). It also records the final emitted name when ``KernelFunction`` is
called, which catches candidates that mutate ``kernel_obj._func.__name__``
after decoration to produce shape-specialized profiler symbols. This is
strictly more reliable than source-AST parsing because:

* Different AI-generated candidates use different decoration styles
  (bare ``@kernel``, ``@kernel(name=...)``, ``from ... import kernel as _k``
  + ``@_k``, string-literal kernel-source generators, ``@flydsl.compiler.kernel``
  attribute form, ...). Source-AST detection has to enumerate all of these
  and inevitably misses one.
* The runtime patch sees what actually gets decorated, regardless of how the
  candidate spelled it. The flydsl ``kernel()`` constructor is the single
  source of truth.

Call ``install_flydsl_kernel_tracker()`` BEFORE loading the candidate module
so the wrapped decorator is in place when the candidate's ``import`` runs.
After the candidate has executed, ``observed_kernel_symbols()`` returns the
``(exact_names, func_patterns)`` tuple shaped exactly like the AST-extractor
output.
"""

from __future__ import annotations

import functools
import os
import re

_INSTALLED = False
# Per-process module state. Each per-shape sub-worker is a fresh subprocess
# so observations don't bleed between shapes/runs.
_OBSERVED_EXACT: set[str] = set()
_OBSERVED_FUNC_NAMES: set[str] = set()

# JIT compilation artifact tracking.  Counts first-time compilation
# attempts (cache miss) and how many of them produced a CompiledArtifact
# (i.e. ``JitFunction._last_compiled`` transitioned from None to non-None).
_JIT_COMPILE_ATTEMPTS: int = 0
_JIT_COMPILE_SUCCESSES: int = 0


def install_flydsl_kernel_tracker() -> bool:
    """Monkey-patch flydsl's ``@kernel`` decorator. Idempotent.

    Also disables flydsl's on-disk JIT compile cache for the current process
    (env var ``FLYDSL_RUNTIME_ENABLE_CACHE=0``). Without this, flydsl's
    JitFunction.__call__ takes the cache-hit fast path on warm runs, loading
    a pre-compiled artifact directly from disk and BYPASSING the Python
    ``@kernel`` decorator entirely -- which means our tracker observes
    nothing even though the candidate's GPU kernel runs. With cache off,
    every fresh sub-worker re-enters the Python compile path on first call,
    triggering the wrapped @kernel and giving us a complete observation.
    Cost: an extra AOT compile per @jit function per sub-worker; cheap
    compared to the eval itself.

    Returns True if the patch was installed (or was already installed),
    False if flydsl is not importable in this process.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    try:
        from flydsl.compiler import kernel_function as kf
    except ImportError:
        return False

    original = kf.kernel

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        # flydsl's @kernel supports two call shapes:
        #   1) ``@kernel`` -> ``kernel(func)`` (bare-decorator form)
        #   2) ``@kernel(name="x", ...)`` -> ``kernel(name="x", ...)`` returns a
        #      decorator factory that THEN takes the function
        # We disambiguate by checking whether the first positional arg is
        # callable; that's how every "optional-args decorator" idiom works.
        if args and callable(args[0]) and not kwargs:
            func = args[0]
            _record(func, None)
            return original(*args, **kwargs)
        # Parametric form: forward to flydsl's factory, then wrap it so we can
        # record the function name when the factory is invoked.
        inner_factory = original(*args, **kwargs)
        name = kwargs.get("name")

        @functools.wraps(inner_factory)
        def decorator(f):
            _record(f, name)
            return inner_factory(f)

        return decorator

    kf.kernel = wrapped
    _install_kernel_function_call_tracker(kf)
    # Also patch common re-export site so ``from flydsl.compiler import kernel``
    # (without going through ``.kernel_function``) sees the wrapped version.
    try:
        import flydsl.compiler as _fc

        _fc.kernel = wrapped
    except Exception:
        pass

    # Patch JitFunction.__call__ to detect whether JIT compilation actually
    # produced a CompiledArtifact.  This is the authoritative compile-pass
    # signal: ``_last_compiled`` is set AFTER ``MlirCompiler.compile()``
    # succeeds and BEFORE GPU execution begins, so its presence distinguishes
    # "compilation timed out" from "execution timed out / crashed after a
    # successful compile".
    _install_jit_compile_tracker()

    _INSTALLED = True
    return True


_ORIGINAL_JIT_CALL = None  # saved reference for uninstall


def _install_kernel_function_call_tracker(kf) -> None:
    """Record final FlyDSL GPU symbols when a ``KernelFunction`` is emitted.

    Decoration-time tracking sees the original Python function name. Some
    candidates later mutate ``kernel_obj._func.__name__`` just before calling
    the kernel inside a ``@jit`` launcher. FlyDSL chooses the profiler-visible
    ``_kernel_name`` during ``KernelFunction.__call__``, so wrapping that method
    records the same final symbol the profiler will report.
    """
    kernel_function_cls = getattr(kf, "KernelFunction", None)
    if kernel_function_cls is None:
        return
    original_call = getattr(kernel_function_cls, "__call__", None)
    if not callable(original_call):
        return
    if getattr(original_call, "_atrex_flydsl_tracker_wrapped", False):
        return

    @functools.wraps(original_call)
    def wrapped_call(self, *args, **kwargs):
        _record_kernel_function_object(self)
        launcher = original_call(self, *args, **kwargs)
        _record_emitted_kernel_name(self)
        return launcher

    wrapped_call._atrex_flydsl_tracker_wrapped = True  # type: ignore[attr-defined]
    kernel_function_cls.__call__ = wrapped_call


def _install_jit_compile_tracker() -> None:
    """Wrap ``JitFunction.__call__`` to count compile attempts / successes."""
    global _ORIGINAL_JIT_CALL
    try:
        from flydsl.compiler.jit_function import JitFunction
    except ImportError:
        return

    _ORIGINAL_JIT_CALL = JitFunction.__call__
    original_call = _ORIGINAL_JIT_CALL

    @functools.wraps(original_call)
    def _tracked_call(self, *args, **kwargs):
        global _JIT_COMPILE_ATTEMPTS, _JIT_COMPILE_SUCCESSES

        # If an artifact already exists for *any* key on this instance, this
        # call will take the cache-hit fast path — no new compilation.  Skip
        # tracking to avoid inflating the attempt counter.
        had_artifact = (
            bool(getattr(self, "_mem_cache", None))
            or getattr(self, "_last_compiled", None) is not None
        )
        if had_artifact:
            return original_call(self, *args, **kwargs)

        _JIT_COMPILE_ATTEMPTS += 1
        try:
            result = original_call(self, *args, **kwargs)
        except Exception:
            # If _last_compiled was populated before the exception propagated,
            # compilation succeeded — the error is from the *execution* phase
            # (e.g. GPU segfault, slow kernel timeout).
            if getattr(self, "_last_compiled", None) is not None:
                _JIT_COMPILE_SUCCESSES += 1
            raise
        if getattr(self, "_last_compiled", None) is not None:
            _JIT_COMPILE_SUCCESSES += 1
        return result

    JitFunction.__call__ = _tracked_call


def uninstall_jit_compile_tracker() -> None:
    """Restore the original ``JitFunction.__call__``, removing all tracking overhead.

    Call this AFTER reading ``flydsl_compile_succeeded()`` and BEFORE the
    performance benchmark so ``triton.do_bench`` measures the candidate with
    zero wrapper overhead on the hot dispatch path.
    """
    global _ORIGINAL_JIT_CALL
    if _ORIGINAL_JIT_CALL is None:
        return
    try:
        from flydsl.compiler.jit_function import JitFunction
    except ImportError:
        _ORIGINAL_JIT_CALL = None
        return
    JitFunction.__call__ = _ORIGINAL_JIT_CALL
    _ORIGINAL_JIT_CALL = None


def _record(func, name) -> None:
    if name:
        _OBSERVED_EXACT.add(name)
    else:
        _OBSERVED_FUNC_NAMES.add(func.__name__)


def _record_kernel_function_object(kernel_obj) -> None:
    explicit_name = getattr(kernel_obj, "_name", None)
    if isinstance(explicit_name, str) and explicit_name:
        _OBSERVED_EXACT.add(explicit_name)
        return
    func = getattr(kernel_obj, "_func", None)
    func_name = getattr(func, "__name__", None)
    if isinstance(func_name, str) and func_name:
        _OBSERVED_FUNC_NAMES.add(func_name)


def _record_emitted_kernel_name(kernel_obj) -> None:
    kernel_name = getattr(kernel_obj, "_kernel_name", None)
    if isinstance(kernel_name, str) and kernel_name:
        _OBSERVED_EXACT.add(kernel_name)


def observed_kernel_symbols() -> tuple[set[str], list[re.Pattern[str]]]:
    """Return ``(exact_names, func_patterns)`` -- same shape as AST extractor."""
    patterns = [re.compile(rf"^{re.escape(n)}_\d+$") for n in sorted(_OBSERVED_FUNC_NAMES)]
    return set(_OBSERVED_EXACT), patterns


def observed_kernel_symbols_serializable() -> dict[str, list[str]]:
    """JSON-friendly form for shipping across sub-worker / parent boundary.

    Stores raw func names rather than compiled regexes; the parent rebuilds
    the regex from ``func_names`` when classifying.
    """
    return {
        "exact_names": sorted(_OBSERVED_EXACT),
        "func_names": sorted(_OBSERVED_FUNC_NAMES),
    }


def flydsl_compile_succeeded() -> bool | None:
    """Check whether flydsl JIT compilation produced a ``CompiledArtifact``.

    Returns
    -------
    True
        At least one JIT compilation was attempted **and all** produced an
        artifact (``JitFunction._last_compiled`` was set).
    False
        At least one JIT compilation was attempted but some did NOT produce
        an artifact — the MLIR pass pipeline timed out or crashed before
        ``CompiledArtifact`` could be constructed.
    None
        No JIT compilation was attempted in this process (non-flydsl
        candidate, or ``model.forward()`` was never called).
    """
    if _JIT_COMPILE_ATTEMPTS == 0:
        return None
    return _JIT_COMPILE_SUCCESSES == _JIT_COMPILE_ATTEMPTS


def reset_observations() -> None:
    """Clear all recorded observations. Useful for tests."""
    global _JIT_COMPILE_ATTEMPTS, _JIT_COMPILE_SUCCESSES
    _OBSERVED_EXACT.clear()
    _OBSERVED_FUNC_NAMES.clear()
    _JIT_COMPILE_ATTEMPTS = 0
    _JIT_COMPILE_SUCCESSES = 0


def symbols_from_serialized(
    payload: dict[str, list[str]] | None,
) -> tuple[set[str], list[re.Pattern[str]]] | None:
    """Reconstruct ``(exact_names, func_patterns)`` from the serialized payload.

    Returns None if ``payload`` is None (no observations recorded). Mirrors
    the shape produced by ``observed_kernel_symbols()`` so call-sites can
    feed either source into ``compute_flydsl_compute_ratio_for_shape``.
    """
    if payload is None:
        return None
    exact = set(payload.get("exact_names") or [])
    func_names = payload.get("func_names") or []
    patterns = [re.compile(rf"^{re.escape(n)}_\d+$") for n in func_names]
    return exact, patterns
