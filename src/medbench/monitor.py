"""Record the peak memory, GPU memory, disk and time a command really used.

Issues #6 and #9 ask for measured numbers instead of estimates: peak GPU and
host memory, free disk, driver and CUDA versions and the wall time of every
model run. :func:`run_monitored` starts a command, samples ``nvidia-smi`` and
``/proc/meminfo`` (``sysctl``/``vm_stat`` on macOS) while it runs and writes one JSON report, also when the
command fails or is interrupted. Only the standard library is used, so the
monitor works before any model stack is installed.

Example::

    medbench monitor --out resources.json -- bash scripts/medbench-model-run.sh run

Limits: host memory is the whole machine's used memory (``MemTotal`` minus
``MemAvailable``, or macOS total minus free, inactive and speculative pages),
so other processes count too; the baseline before the
command started is recorded to subtract them. ``children_max_rss_gb`` is the
largest single finished child process, not the sum of all of them. GPU
memory is sampled, so a spike shorter than ``interval`` can be missed.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

from medbench import hardware
from medbench.hardware import parse_meminfo

__all__ = [
    "GpuUsage",
    "ResourceMonitor",
    "host_memory_used_gb",
    "nvidia_versions",
    "query_gpus",
    "run_monitored",
]

_GPU_FIELDS = "index,name,memory.used,memory.total,utilization.gpu"
_NVIDIA_SMI_TIMEOUT_S = 15
_STOP_GRACE_S = 10  # below run-with-budget-warning.sh's 15 s grace
_CUDA_VERSION = re.compile(r"CUDA Version\s*:\s*([\d.]+)")
_DRIVER_VERSION = re.compile(r"Driver Version\s*:\s*([\d.]+)")


@dataclass(frozen=True)
class GpuUsage:
    """One ``nvidia-smi`` sample of one GPU (memory in MiB, as reported)."""

    index: int
    name: str
    memory_used_mib: float
    memory_total_mib: float
    utilization_pct: float | None


@dataclass
class _GpuPeak:
    """Peak use of one GPU over the run (MiB, as ``nvidia-smi`` reports)."""

    index: int
    name: str
    memory_total_mib: float
    baseline_used_mib: float
    peak_used_mib: float = 0.0
    peak_utilization_pct: float | None = None


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]" on GPUs that do not report the counter.


def _nvidia_smi(binary: str, *args: str) -> str | None:
    try:
        return subprocess.run(
            [binary, *args],
            capture_output=True,
            check=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def query_gpus(nvidia_smi: str | None) -> list[GpuUsage]:
    """Sample memory use of every NVIDIA GPU; empty without a driver."""
    if nvidia_smi is None:
        return []
    output = _nvidia_smi(
        nvidia_smi, f"--query-gpu={_GPU_FIELDS}", "--format=csv,noheader,nounits"
    )
    gpus = []
    for line in (output or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(_GPU_FIELDS.split(",")) or not parts[0].isdigit():
            continue
        used, total = _number(parts[2]), _number(parts[3])
        if used is None or total is None:
            continue
        gpus.append(GpuUsage(int(parts[0]), parts[1], used, total, _number(parts[4])))
    return gpus


def nvidia_versions(nvidia_smi: str | None) -> dict[str, str | None]:
    """Driver version and the highest CUDA version the driver supports."""
    output = _nvidia_smi(nvidia_smi, "-q") if nvidia_smi else None
    text = output or ""
    driver = _DRIVER_VERSION.search(text)
    cuda = _CUDA_VERSION.search(text)
    return {
        "driver_version": driver.group(1) if driver else None,
        "cuda_version": cuda.group(1) if cuda else None,
    }


def host_memory_used_gb(meminfo: Path) -> tuple[float, float] | None:
    """Return ``(used, total)`` host memory in decimal GB, or ``None``."""
    try:
        values = parse_meminfo(meminfo.read_text(encoding="utf-8"))
    except OSError:
        return None
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    return values["MemTotal"] - values["MemAvailable"], values["MemTotal"]


def macos_memory_used_gb() -> tuple[float, float] | None:
    """Return ``(used, total)`` from ``sysctl``/``vm_stat`` on macOS, or ``None``."""
    total, available = hardware._macos_memory()
    if total <= 0:
        return None
    return total - available, total


def _free_gb(path: Path) -> float | None:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


def _children_max_rss_gb() -> float | None:
    try:
        import resource  # Unix only
    except ImportError:  # pragma: no cover - Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Linux reports kibibytes, macOS bytes.
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9


@dataclass
class ResourceMonitor:
    """Keep the peaks of repeated samples.

    Attributes:
        nvidia_smi: Path to ``nvidia-smi``, or ``None`` on machines without it.
        meminfo: ``/proc/meminfo`` (a fixture in tests).
        disk: Directory whose file system's free space is tracked.
    """

    nvidia_smi: str | None = None
    meminfo: Path = Path("/proc/meminfo")
    disk: Path = Path()
    samples: int = 0
    gpus: dict[int, _GpuPeak] = field(default_factory=dict)
    host: dict[str, float | None] = field(default_factory=dict)
    disk_free: dict[str, float | None] = field(default_factory=dict)
    timeline: list[list[float | None]] = field(default_factory=list)

    def sample(self, elapsed_s: float) -> None:
        """Take one sample of every GPU, host memory and free disk."""
        self.samples += 1
        memory = host_memory_used_gb(self.meminfo)
        if memory is None and sys.platform == "darwin":
            memory = macos_memory_used_gb()
        if memory is not None:
            used, total = memory
            self.host.setdefault("baseline_used_gb", used)
            self.host["total_gb"] = total
            self.host["peak_used_gb"] = max(self.host.get("peak_used_gb") or 0, used)
        free = _free_gb(self.disk)
        self.disk_free.setdefault("start_gb", free)
        self.disk_free["end_gb"] = free
        if free is not None:
            low = self.disk_free.get("min_gb")
            self.disk_free["min_gb"] = free if low is None else min(low, free)
        host_used = None if memory is None else round(memory[0], 3)
        row: list[float | None] = [round(elapsed_s, 1), host_used]
        for gpu in query_gpus(self.nvidia_smi):
            peak = self.gpus.setdefault(
                gpu.index,
                _GpuPeak(
                    gpu.index, gpu.name, gpu.memory_total_mib, gpu.memory_used_mib
                ),
            )
            peak.peak_used_mib = max(peak.peak_used_mib, gpu.memory_used_mib)
            if gpu.utilization_pct is not None:
                peak.peak_utilization_pct = max(
                    peak.peak_utilization_pct or 0.0, gpu.utilization_pct
                )
            row.append(gpu.memory_used_mib)
        self.timeline.append(row)

    def as_dict(self) -> dict[str, object]:
        """Return the peaks as a JSON-friendly dictionary."""
        rounded = {key: _round(value) for key, value in self.host.items()}
        return {
            "samples": self.samples,
            "host_memory": rounded,
            "gpus": [asdict(self.gpus[index]) for index in sorted(self.gpus)],
            "disk": {
                "path": str(self.disk.resolve()),
                **{key: _round(value) for key, value in self.disk_free.items()},
            },
            "timeline_columns": ["seconds", "host_used_gb"]
            + [f"gpu{index}_used_mib" for index in sorted(self.gpus)],
            "timeline": self.timeline,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


class _InterruptedError(Exception):
    """Raised by the SIGTERM handler so the report is still written."""


def _raise_interrupted(signum: int, frame: FrameType | None) -> None:
    del frame
    raise _InterruptedError(signum)


def _stop(process: subprocess.Popen[bytes]) -> int:
    """Forward the stop to the command so its own cleanup traps run."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.wait()


def summary_markdown(report: dict[str, Any], label: str) -> str:
    """Return a one-row Markdown table of a report's peaks for a CI summary."""
    host = report.get("host_memory") or {}
    gpus = report.get("gpus") or []
    gpu = ", ".join(f"{g['peak_used_mib']:.0f}" for g in gpus) or "-"
    peak, baseline = host.get("peak_used_gb"), host.get("baseline_used_gb")
    host_text = "-" if peak is None else f"{peak:.2f} (baseline {baseline:.2f})"
    rss = report.get("children_max_rss_gb")
    disk_min = (report.get("disk") or {}).get("min_gb")
    cells = [
        label,
        f"{report['wall_seconds']:.0f}",
        host_text,
        "-" if rss is None else f"{rss:.2f}",
        gpu,
        "-" if disk_min is None else f"{disk_min:.1f}",
        str(report["exit_code"]),
    ]
    return (
        "| Step | Wall s | Host used GB, peak | Largest child RSS GB "
        "| GPU peak MiB | Min free disk GB | Exit |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"| {' | '.join(cells)} |\n"
    )


def run_monitored(
    command: Sequence[str],
    out: Path,
    *,
    interval_s: float = 5.0,
    disk: Path = Path(),
    nvidia_smi: str | None = None,
    meminfo: Path = Path("/proc/meminfo"),
    label: str | None = None,
    summary: Path | None = None,
) -> int:
    """Run ``command``, sample its resource use and write ``out`` as JSON.

    Args:
        command: Program and arguments.
        out: Report path; parent directories are created.
        interval_s: Seconds between samples.
        disk: Directory whose free space is tracked (model caches live here).
        nvidia_smi: ``nvidia-smi`` path; found on ``PATH`` when omitted.
        meminfo: Host memory source.
        label: Step name stored in the report; defaults to ``out``'s stem.
        summary: File to append a Markdown row to, such as
            ``$GITHUB_STEP_SUMMARY``.

    Returns:
        The command's exit code (``128 + signal`` when the monitor was
        stopped), so the monitor can wrap a CI step transparently.
    """
    binary = nvidia_smi or shutil.which("nvidia-smi")
    monitor = ResourceMonitor(nvidia_smi=binary, meminfo=meminfo, disk=disk)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    start = time.monotonic()
    monitor.sample(0.0)
    previous = signal.signal(signal.SIGTERM, _raise_interrupted)
    process = subprocess.Popen(list(command))
    interrupted = None
    try:
        while True:
            try:
                exit_code = process.wait(interval_s)
                break
            except subprocess.TimeoutExpired:
                monitor.sample(time.monotonic() - start)
    except (_InterruptedError, KeyboardInterrupt) as error:
        interrupted = error.args[0] if error.args else int(signal.SIGINT)
        _stop(process)
        exit_code = 128 + int(interrupted)
    finally:
        signal.signal(signal.SIGTERM, previous)
    wall = time.monotonic() - start
    monitor.sample(wall)
    report: dict[str, Any] = {
        "label": label or out.stem,
        "command": list(command),
        "exit_code": exit_code,
        "interrupted_by_signal": interrupted,
        "started_at": started_at,
        "wall_seconds": round(wall, 1),
        "interval_seconds": interval_s,
        "children_max_rss_gb": _round(_children_max_rss_gb()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        **nvidia_versions(binary),
        **monitor.as_dict(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if summary is not None:
        with summary.open("a", encoding="utf-8") as handle:
            handle.write(summary_markdown(report, report["label"]))
    return exit_code
