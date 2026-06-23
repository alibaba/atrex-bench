"""Timeout primitives for guarding candidate execution.

The eval pipeline runs the *reference* (golden) implementation as well as the
AI-generated *candidate*. The candidate is the only one that can hang forever
(huge JIT compiles, infinite loops in generated code, etc.); the reference is
trusted. We therefore expose a timeout that wraps candidate-only call sites
without restricting how long the reference is allowed to take.

Implementation notes:

* Uses ``signal.SIGALRM``, which only fires on the main thread of the current
  process. The per-shape sub-worker runs the candidate on its main thread, so
  this works there. Calling the timeout on a non-main thread is a no-op (we
  return cleanly without arming the alarm) so the helper is safe to import in
  any context.
* SIGALRM is delivered asynchronously by the kernel, but Python only checks
  for pending signals between bytecode instructions or when a C call returns
  control to the interpreter. For *Python-bound* slow code (e.g. flydsl's MLIR
  tracer on huge unrolled loops) the alarm fires mid-trace and aborts
  promptly. For *GPU-bound* slow code (a kernel that blocks
  ``torch.cuda.synchronize`` for minutes) the alarm fires only once sync
  returns; the kernel is not interrupted mid-flight. In both cases the
  candidate is correctly flagged as timed out, but the GPU-bound case still
  pays the full kernel runtime before the timeout signal is observed. This is
  a known and accepted limitation: there is no portable Python API to
  interrupt a running HIP/CUDA kernel.
"""

from __future__ import annotations

import contextlib
import signal
import threading


class CandidateTimeoutError(Exception):
    """Raised when a candidate call exceeds the configured wall-clock budget."""


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


@contextlib.contextmanager
def candidate_timeout(seconds: int | float | None):
    """Context manager that raises CandidateTimeoutError after ``seconds``.

    Pass ``None``, ``0``, or a negative number to disable the timeout (the
    context manager becomes a no-op). The previous SIGALRM handler is
    restored on exit, so this is safe to nest with the rest of the runner.
    """
    if seconds is None or seconds <= 0 or not _is_main_thread():
        yield
        return

    def _handler(signum, frame):
        raise CandidateTimeoutError(
            f"candidate exceeded {seconds}s wall-clock timeout"
        )

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    # signal.alarm only accepts ints (whole seconds); round up so callers
    # asking for fractional seconds still get an at-least-N-second budget.
    arm_seconds = max(1, int(seconds + 0.999))
    signal.alarm(arm_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
