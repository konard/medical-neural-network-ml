"""Detect the current machine and keep experiments inside a resource budget.

The issue asks to be careful with the development machine, so every heavy
experiment in this repository goes through :class:`ResourceBudget`: it limits
the number of threads, refuses to load a model that does not fit in memory and
stops a run that takes too long.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "BudgetExceededError",
    "GpuInfo",
    "HostInfo",
    "ResourceBudget",
    "detect_gpus",
    "detect_host",
    "parse_meminfo",
    "parse_vm_stat",
]

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class BudgetExceededError(RuntimeError):
    """Raised when an experiment would exceed its resource budget."""


@dataclass(frozen=True)
class GpuInfo:
    """One NVIDIA GPU as reported by ``nvidia-smi``."""

    name: str
    memory_total_gb: float
    memory_free_gb: float


@dataclass(frozen=True)
class HostInfo:
    """Summary of the machine running the benchmarks."""

    cpu_count: int
    usable_cpus: int
    memory_total_gb: float
    memory_available_gb: float
    cgroup_memory_limit_gb: float | None
    gpus: tuple[GpuInfo, ...] = field(default_factory=tuple)
    apple_silicon: bool = False

    @property
    def has_gpu(self) -> bool:
        """Whether at least one GPU was found."""
        return bool(self.gpus)

    def as_dict(self) -> dict[str, object]:
        """Return the information as a JSON-friendly dictionary."""
        return {
            "cpu_count": self.cpu_count,
            "usable_cpus": self.usable_cpus,
            "memory_total_gb": round(self.memory_total_gb, 2),
            "memory_available_gb": round(self.memory_available_gb, 2),
            "cgroup_memory_limit_gb": self.cgroup_memory_limit_gb,
            "gpus": [gpu.__dict__ for gpu in self.gpus],
            "apple_silicon": self.apple_silicon,
        }


def parse_meminfo(text: str) -> dict[str, float]:
    """Parse ``/proc/meminfo`` into gigabytes.

    Args:
        text: File content.

    Returns:
        Mapping such as ``{"MemTotal": 11.6, "MemAvailable": 9.1}``.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if not parts or not parts[0].isdigit():
            continue
        scale = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
        values[name.strip()] = int(parts[0]) * scale / 1e9
    return values


def parse_vm_stat(text: str) -> float:
    """Conservative reclaimable macOS memory in GB, excluding compression.

    Free, inactive and speculative pages are disjoint. Purgeable pages can
    overlap other counters, so adding them would double count memory.
    """
    page_size = re.search(r"page size of (\d+) bytes", text)
    if page_size is None:
        return 0.0
    pages = sum(
        int(match.group(1))
        for match in re.finditer(
            r"^Pages (?:free|inactive|speculative):\s+(\d+)", text, re.MULTILINE
        )
    )
    return pages * int(page_size.group(1)) / 1e9


def _macos_memory() -> tuple[float, float]:
    try:
        total = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        available = subprocess.run(
            ["/usr/bin/vm_stat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(total.stdout.strip()) / 1e9, parse_vm_stat(available.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0, 0.0


def _cgroup_memory_limit_gb(root: Path) -> float | None:
    """Return the cgroup v2 or v1 memory limit in GB, if one is set."""
    for candidate in (
        root / "sys/fs/cgroup/memory.max",
        root / "sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            raw = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw) / 1e9
    return None


def detect_gpus(nvidia_smi: str | None = None) -> tuple[GpuInfo, ...]:
    """List NVIDIA GPUs using ``nvidia-smi``.

    Args:
        nvidia_smi: Path to the binary; found on ``PATH`` when omitted.

    Returns:
        Detected GPUs, empty when there is no driver.
    """
    binary = nvidia_smi or shutil.which("nvidia-smi")
    if binary is None:
        return ()
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            try:
                # nvidia-smi reports MiB; the rest of medbench uses decimal GB.
                gpus.append(
                    GpuInfo(
                        parts[0],
                        float(parts[1]) * 1024**2 / 1e9,
                        float(parts[2]) * 1024**2 / 1e9,
                    )
                )
            except ValueError:
                continue  # Driver can report N/A for unavailable counters.
    return tuple(gpus)


def detect_host(root: Path = Path("/"), *, with_gpus: bool = True) -> HostInfo:
    """Describe the current machine.

    Args:
        root: File system root; tests pass a fake directory.
        with_gpus: Whether to call ``nvidia-smi``.

    Returns:
        Host information.
    """
    try:
        meminfo = parse_meminfo((root / "proc/meminfo").read_text(encoding="utf-8"))
    except OSError:
        meminfo = {}
    cpu_count = os.cpu_count() or 1
    try:
        affinity = getattr(os, "sched_getaffinity", None)
        usable = len(affinity(0)) if affinity else cpu_count
    except (AttributeError, OSError):
        usable = cpu_count
    is_macos = root == Path("/") and platform.system() == "Darwin"
    if is_macos:
        total, available = _macos_memory()
        meminfo = {"MemTotal": total, "MemAvailable": available}
    # macOS has no cgroups; never mix the host's Linux limits into its budget.
    cgroup_limit = None if is_macos else _cgroup_memory_limit_gb(root)
    if cgroup_limit is not None:
        for usage_path in (
            root / "sys/fs/cgroup/memory.current",
            root / "sys/fs/cgroup/memory/memory.usage_in_bytes",
        ):
            try:
                used = int(usage_path.read_text(encoding="utf-8").strip()) / 1e9
            except (OSError, ValueError):
                continue
            meminfo["MemAvailable"] = min(
                meminfo.get("MemAvailable", 0.0), max(0.0, cgroup_limit - used)
            )
            break
    return HostInfo(
        cpu_count=cpu_count,
        usable_cpus=usable,
        memory_total_gb=meminfo.get("MemTotal", 0.0),
        memory_available_gb=meminfo.get("MemAvailable", 0.0),
        cgroup_memory_limit_gb=cgroup_limit,
        gpus=detect_gpus() if with_gpus else (),
        apple_silicon=is_macos and platform.machine() == "arm64",
    )


@dataclass
class ResourceBudget:
    """Limits for one experiment.

    Attributes:
        max_threads: Upper bound for worker threads. ``None`` means "leave two
            CPUs free", which keeps a shared machine responsive.
        max_memory_fraction: Share of available memory a model may use.
        max_seconds: Wall-clock limit for the whole run.
    """

    max_threads: int | None = None
    max_memory_fraction: float = 0.7
    max_seconds: float = 600.0
    _started: float = field(default_factory=time.monotonic, repr=False)

    def threads(self, host: HostInfo) -> int:
        """Return the number of threads to use on ``host``."""
        default = max(1, host.usable_cpus - 2)
        if self.max_threads is None:
            return default
        return max(1, min(self.max_threads, host.usable_cpus))

    def thread_env(self, host: HostInfo) -> dict[str, str]:
        """Environment variables that cap BLAS/OpenMP thread pools."""
        value = str(self.threads(host))
        return dict.fromkeys(THREAD_ENV_VARS, value)

    def memory_limit_gb(self, host: HostInfo) -> float:
        """Memory the experiment may use on ``host`` (GB)."""
        available = host.memory_available_gb
        if host.cgroup_memory_limit_gb is not None:
            available = min(available, host.cgroup_memory_limit_gb)
        return available * self.max_memory_fraction

    def check_memory(self, required_gb: float, host: HostInfo) -> None:
        """Raise when ``required_gb`` does not fit in the budget.

        Raises:
            BudgetExceededError: If the model would not fit.
        """
        limit = self.memory_limit_gb(host)
        if required_gb > limit:
            message = (
                f"needs {required_gb:.1f} GB but the budget allows {limit:.1f} GB; "
                "use a smaller or more quantized model, or a GPU runner"
            )
            raise BudgetExceededError(message)

    def remaining_seconds(self) -> float:
        """Seconds left before the deadline."""
        return self.max_seconds - (time.monotonic() - self._started)

    def check_deadline(self) -> None:
        """Raise when the wall-clock budget is used up.

        Raises:
            BudgetExceededError: If the deadline has passed.
        """
        if self.remaining_seconds() <= 0:
            message = f"time budget of {self.max_seconds:.0f} s exceeded"
            raise BudgetExceededError(message)
