"""Tests for host detection and resource budgets (issue #1)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from medbench.hardware import (
    BudgetExceededError,
    GpuInfo,
    HostInfo,
    ResourceBudget,
    detect_gpus,
    detect_host,
    parse_meminfo,
)

MEMINFO = """MemTotal:       12247552 kB
MemFree:         7664128 kB
MemAvailable:   10411008 kB
HugePages_Total:       0
"""


def _host(
    cpus: int = 6, available: float = 10.0, limit: float | None = None
) -> HostInfo:
    return HostInfo(cpus, cpus, 12.0, available, limit)


def test_parse_meminfo() -> None:
    """Values in kB become GB; unit-less counters are kept."""
    values = parse_meminfo(MEMINFO + "Broken line\n")
    assert values["MemTotal"] == pytest.approx(12.54, rel=1e-3)
    assert values["HugePages_Total"] == 0
    assert "Broken line" not in values


def test_detect_host_with_fake_root(tmp_path: Path) -> None:
    """Memory and cgroup limits are read from the given root."""
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text(MEMINFO, encoding="utf-8")
    (tmp_path / "sys/fs/cgroup").mkdir(parents=True)
    (tmp_path / "sys/fs/cgroup/memory.max").write_text("8000000000\n", encoding="utf-8")
    host = detect_host(tmp_path, with_gpus=False)
    assert host.memory_available_gb == pytest.approx(10.66, rel=1e-3)
    assert host.cgroup_memory_limit_gb == 8.0
    assert not host.has_gpu
    assert host.usable_cpus >= 1
    assert host.as_dict()["gpus"] == []


def test_detect_host_without_files(tmp_path: Path) -> None:
    """Missing files give zeros instead of errors; 'max' means no limit."""
    (tmp_path / "sys/fs/cgroup").mkdir(parents=True)
    (tmp_path / "sys/fs/cgroup/memory.max").write_text("max\n", encoding="utf-8")
    host = detect_host(tmp_path, with_gpus=False)
    assert host.memory_total_gb == 0
    assert host.cgroup_memory_limit_gb is None


def test_detect_gpus_with_fake_nvidia_smi(tmp_path: Path) -> None:
    """The CSV output of nvidia-smi is parsed into GB."""
    script = tmp_path / "nvidia-smi"
    script.write_text("#!/bin/sh\necho 'NVIDIA L4, 23034, 22000'\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    gpus = detect_gpus(str(script))
    assert gpus == (GpuInfo("NVIDIA L4", 23034 / 1024, 22000 / 1024),)


def test_detect_gpus_failure(tmp_path: Path) -> None:
    """A failing or missing binary means no GPU."""
    script = tmp_path / "nvidia-smi"
    script.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    assert detect_gpus(str(script)) == ()
    assert detect_gpus(str(tmp_path / "missing")) == ()


def test_threads_leave_cpus_free() -> None:
    """By default two CPUs stay free; explicit limits are capped."""
    assert ResourceBudget().threads(_host(6)) == 4
    assert ResourceBudget().threads(_host(1)) == 1
    assert ResourceBudget(max_threads=16).threads(_host(6)) == 6
    env = ResourceBudget(max_threads=2).thread_env(_host(6))
    assert env["OMP_NUM_THREADS"] == "2"


def test_memory_budget() -> None:
    """Models larger than the budget are refused before loading."""
    budget = ResourceBudget(max_memory_fraction=0.5)
    assert budget.memory_limit_gb(_host(available=10.0)) == 5.0
    assert budget.memory_limit_gb(_host(available=10.0, limit=4.0)) == 2.0
    budget.check_memory(4.0, _host(available=10.0))
    with pytest.raises(BudgetExceededError, match="GPU runner"):
        budget.check_memory(17.0, _host(available=10.0))


def test_deadline() -> None:
    """A zero budget is exceeded immediately."""
    with pytest.raises(BudgetExceededError, match="time budget"):
        ResourceBudget(max_seconds=0).check_deadline()
    assert ResourceBudget(max_seconds=100).remaining_seconds() > 99
