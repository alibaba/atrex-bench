"""POSIX supervisor for Ray actor-owned evaluator process groups.

Only the actor holds the lifetime pipe's write end. All evaluator supervisors
inherit its read end, so actor death wakes them independently. Each supervisor
also notices its direct parent dying, including a parent killed by a timeout.
The command shares the supervisor's process group to preserve existing killpg
semantics. A separate watchdog kills remaining group members after supervisor
exit, including when the direct command exits before its descendants. This is
process cleanup, not a sandbox for hostile daemonized code.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time

LIFETIME_FD_ENV = "ATREX_BENCH_RAY_LIFETIME_FD"
WORKER_EXIT_GRACE_S = 8
_SIGNAL_GRACE_S = 2.0


def supervised_popen(command, **kwargs):
    descriptor = int(os.environ[LIFETIME_FD_ENV])
    env = kwargs.get("env")
    if env is not None:
        env = dict(env)
        env[LIFETIME_FD_ENV] = str(descriptor)
        kwargs["env"] = env
    return subprocess.Popen(
        [sys.executable, "-m", "atrex_bench._process_guard", str(descriptor), *command],
        pass_fds=(descriptor,),
        **kwargs,
    )


def main() -> None:
    descriptor = int(sys.argv[1])
    parent = os.getppid()
    stopped = False

    # The watchdog has its own session and no stdout/stderr pipe handles. Its
    # only owner is this supervisor: EOF follows natural exit or SIGKILL. The
    # command never inherits the write end, so surviving grandchildren cannot
    # keep the watchdog asleep. The kernel records our exit status before EOF
    # wakes it, preserving the evaluator's normal verdict return code.
    cleanup_read, cleanup_write = os.pipe()
    group = os.getpgrp()
    if os.fork() == 0:
        os.setsid()
        for fd in (cleanup_write, descriptor, 0, 1, 2):
            os.close(fd)
        try:
            while os.read(cleanup_read, 1):
                pass
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os._exit(0)
    os.close(cleanup_read)

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    child = subprocess.Popen(sys.argv[2:], pass_fds=(descriptor,))
    while child.poll() is None:
        readable, _, _ = select.select([descriptor], [], [], 0.05)
        if stopped or os.getppid() != parent or (readable and not os.read(descriptor, 1)):
            # The supervisor ignores INT/TERM itself; the command and any
            # non-detached descendants in this group receive them together.
            for signum in (signal.SIGINT, signal.SIGTERM):
                os.killpg(os.getpgrp(), signum)
                deadline = time.monotonic() + _SIGNAL_GRACE_S
                while child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if child.poll() is not None:
                    break
            else:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            break
    # Keep cleanup_write open until process exit; closing it here could let
    # the watchdog kill the supervisor before its successful status is saved.
    returncode = child.wait()
    if returncode < 0:
        # Die from the same signal so Popen callers retain negative returncodes
        # and the evaluator's existing signal/OOM diagnostics remain usable.
        signum = -returncode
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    os._exit(returncode)


if __name__ == "__main__":
    main()
