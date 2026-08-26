"""Device-scoped NVIDIA clock management through ``nvidia-smi``."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_GPU_UUID_PATTERN = re.compile(r"GPU-[0-9A-Fa-f-]+")


class NvidiaClockError(RuntimeError):
    """Raised when an NVIDIA clock-management command cannot be completed."""


@dataclass(frozen=True)
class NvidiaDevice:
    """One physical NVIDIA GPU resolved from an index or UUID selector."""

    selector: str
    index: str
    uuid: str
    name: str


@dataclass(frozen=True)
class ClockSnapshot:
    """Observed graphics and memory clocks for one GPU."""

    graphics_mhz: int
    memory_mhz: int
    observed_at: str


@dataclass(frozen=True)
class ComputeProcess:
    """One compute process reported for a physical GPU."""

    pid: int
    process_name: str
    gpu_uuid: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_gpu_selector(selector: str) -> None:
    if selector.isdigit() or _GPU_UUID_PATTERN.fullmatch(selector):
        return
    raise NvidiaClockError(
        "GPU selector must be one decimal index or a physical GPU UUID "
        f"starting with 'GPU-'; got {selector!r}."
    )


def _validate_clock_mhz(mhz: int) -> None:
    if isinstance(mhz, bool) or not isinstance(mhz, int) or mhz <= 0:
        raise NvidiaClockError(
            f"Clock frequency must be a positive integer MHz value; got {mhz!r}."
        )


class NvidiaSmi:
    """Run fixed, device-scoped ``nvidia-smi`` management commands."""

    def __init__(
        self,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        geteuid: Callable[[], int] = os.geteuid,
        now: Callable[[], str] = _utc_now,
        timeout_s: float = 10.0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._run_command = run
        self._geteuid = geteuid
        self._now = now
        self._timeout_s = timeout_s

    def _command_prefix(self) -> list[str]:
        if self._geteuid() == 0:
            return ["nvidia-smi"]
        return ["sudo", "-n", "nvidia-smi"]

    def _run(self, arguments: list[str]) -> str:
        argv = [*self._command_prefix(), *arguments]
        run_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": self._timeout_s,
            "check": False,
        }
        try:
            completed = self._run_command(argv, **run_kwargs)
        except OSError as error:
            raise NvidiaClockError(f"Clock command executable was not found: {argv[0]}") from error
        except subprocess.TimeoutExpired as error:
            raise NvidiaClockError(
                f"Clock command timed out after {self._timeout_s:g}s: {' '.join(argv)}"
            ) from error
        if completed.returncode != 0:
            detail = next(
                (
                    line.strip()
                    for line in (completed.stderr or "").splitlines()
                    if line.strip()
                ),
                "no stderr",
            )
            raise NvidiaClockError(
                f"Clock command exited with code {completed.returncode}: "
                f"{' '.join(argv)}; stderr: {detail}"
            )
        unsupported_detail = next(
            (
                line.strip()
                for line in (
                    *((completed.stdout or "").splitlines()),
                    *((completed.stderr or "").splitlines()),
                )
                if "not supported" in line.lower()
            ),
            None,
        )
        if unsupported_detail is not None:
            raise NvidiaClockError(
                "Clock command reported an unsupported operation: "
                f"{' '.join(argv)}; output: {unsupported_detail}"
            )
        return completed.stdout or ""

    def list_visible_indices(self) -> tuple[str, ...]:
        """Return every GPU index visible to ``nvidia-smi``."""
        stdout = self._run(
            ["--query-gpu=index", "--format=csv,noheader,nounits"]
        )
        return tuple(line.strip() for line in stdout.splitlines() if line.strip())

    def resolve_device(self, selector: str) -> NvidiaDevice:
        """Resolve exactly one physical GPU and retain the caller's selector."""
        _validate_gpu_selector(selector)
        stdout = self._run(
            [
                "-i",
                selector,
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ]
        )
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise NvidiaClockError(
                "GPU selector must resolve to exactly one device; "
                f"selector={selector!r}, rows={len(lines)}."
            )
        parts = [part.strip() for part in lines[0].split(",", maxsplit=2)]
        if len(parts) != 3 or not all(parts):
            raise NvidiaClockError(
                f"Could not parse GPU identity for selector {selector!r}: {lines[0]!r}."
            )
        return NvidiaDevice(
            selector=selector,
            index=parts[0],
            uuid=parts[1],
            name=parts[2],
        )

    def list_compute_processes(
        self, device: NvidiaDevice
    ) -> tuple[ComputeProcess, ...]:
        """Return compute processes associated with exactly one GPU."""
        stdout = self._run(
            [
                "-i",
                device.selector,
                "--query-compute-apps=pid,process_name,gpu_uuid",
                "--format=csv,noheader,nounits",
            ]
        )
        processes: list[ComputeProcess] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = [part.strip() for part in stripped.split(",", maxsplit=2)]
            if len(parts) != 3 or not all(parts):
                raise NvidiaClockError(
                    f"Could not parse compute process row for {device.selector!r}: "
                    f"{stripped!r}."
                )
            try:
                pid = int(parts[0])
            except ValueError as error:
                raise NvidiaClockError(
                    f"Could not parse compute process PID for {device.selector!r}: "
                    f"{parts[0]!r}."
                ) from error
            processes.append(
                ComputeProcess(pid=pid, process_name=parts[1], gpu_uuid=parts[2])
            )
        return tuple(processes)

    def query_clocks(self, device: NvidiaDevice) -> ClockSnapshot:
        """Read current graphics and memory clocks for exactly one GPU."""
        stdout = self._run(
            [
                "-i",
                device.selector,
                "--query-gpu=clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ]
        )
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise NvidiaClockError(
                "Could not parse clock values for exactly one GPU; "
                f"selector={device.selector!r}, rows={len(lines)}."
            )
        parts = [part.strip() for part in lines[0].split(",")]
        if len(parts) != 2:
            raise NvidiaClockError(
                f"Could not parse clock values for {device.selector!r}: {lines[0]!r}."
            )
        try:
            graphics_mhz, memory_mhz = (int(part) for part in parts)
        except ValueError as error:
            raise NvidiaClockError(
                f"Could not parse clock values for {device.selector!r}: {lines[0]!r}."
            ) from error
        return ClockSnapshot(
            graphics_mhz=graphics_mhz,
            memory_mhz=memory_mhz,
            observed_at=self._now(),
        )

    def lock_graphics(self, device: NvidiaDevice, mhz: int) -> None:
        """Lock graphics clocks to one fixed frequency."""
        _validate_clock_mhz(mhz)
        self._run(["-i", device.selector, "-lgc", f"{mhz},{mhz}"])

    def lock_memory(self, device: NvidiaDevice, mhz: int) -> None:
        """Lock memory clocks to one fixed frequency."""
        _validate_clock_mhz(mhz)
        self._run(["-i", device.selector, "-lmc", f"{mhz},{mhz}"])

    def reset_graphics(self, device: NvidiaDevice) -> None:
        """Reset graphics clocks to the driver default."""
        self._run(["-i", device.selector, "-rgc"])

    def reset_memory(self, device: NvidiaDevice) -> None:
        """Reset memory clocks to the driver default."""
        self._run(["-i", device.selector, "-rmc"])
