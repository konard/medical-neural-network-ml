"""Tests for medbench.monitor, the peak-resource recorder (issues #6 and #9).

A shell script stands in for ``nvidia-smi`` and a file for ``/proc/meminfo``,
so the GPU path is tested without a GPU.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from medbench import hardware
from medbench.cli import main
from medbench.monitor import (
    ResourceMonitor,
    host_memory_used_gb,
    nvidia_versions,
    query_gpus,
    run_monitored,
    summary_markdown,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell scripts and signals"
)

FAKE_NVIDIA_SMI = """#!/bin/sh
# Answers like nvidia-smi 570 on one 24 GB GPU; memory.used grows on every
# call so the monitor has a peak to find.
state="$(dirname "$0")/calls"
calls=$(( $(cat "$state" 2>/dev/null || echo 0) + 1 ))
echo "$calls" > "$state"
case "$1" in
  -q)
    echo "Driver Version                            : 570.86.15"
    echo "CUDA Version                              : 12.8"
    ;;
  --query-gpu=*)
    echo "0, NVIDIA GeForce RTX 4090, $(( 1000 * calls )), 24564, [N/A]"
    echo "1, NVIDIA L4, [N/A], 23034, 7"
    ;;
esac
"""

MEMINFO = "MemTotal:       32000000 kB\nMemAvailable:   20000000 kB\n"


@pytest.fixture
def nvidia_smi(tmp_path: Path) -> str:
    """Path to the fake ``nvidia-smi``."""
    path = tmp_path / "bin" / "nvidia-smi"
    path.parent.mkdir()
    path.write_text(FAKE_NVIDIA_SMI, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def meminfo(tmp_path: Path) -> Path:
    """A ``/proc/meminfo`` with 32.768 GB total and 12.288 GB used."""
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO, encoding="utf-8")
    return path


def test_query_gpus_skips_unreported_counters(nvidia_smi: str) -> None:
    gpus = query_gpus(nvidia_smi)
    assert [(gpu.index, gpu.memory_used_mib) for gpu in gpus] == [(0, 1000.0)]
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert gpus[0].utilization_pct is None
    assert query_gpus(None) == []


def test_versions(nvidia_smi: str, tmp_path: Path) -> None:
    assert nvidia_versions(nvidia_smi) == {
        "driver_version": "570.86.15",
        "cuda_version": "12.8",
    }
    missing = str(tmp_path / "absent")
    assert nvidia_versions(missing) == {"driver_version": None, "cuda_version": None}


def test_host_memory(meminfo: Path, tmp_path: Path) -> None:
    used, total = host_memory_used_gb(meminfo) or (0, 0)
    assert used == pytest.approx(12.288)
    assert total == pytest.approx(32.768)
    assert host_memory_used_gb(tmp_path / "absent") is None


def test_macos_host_memory_comes_from_vm_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub's macOS runners have no /proc/meminfo (issue #9)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(hardware, "_macos_memory", lambda: (16.0, 6.5))
    monitor = ResourceMonitor(meminfo=tmp_path / "absent", disk=tmp_path)

    monitor.sample(0.0)

    assert monitor.as_dict()["host_memory"] == {
        "baseline_used_gb": 9.5,
        "total_gb": 16.0,
        "peak_used_gb": 9.5,
    }
    monkeypatch.setattr(hardware, "_macos_memory", lambda: (0.0, 0.0))
    monitor = ResourceMonitor(meminfo=tmp_path / "absent", disk=tmp_path)
    monitor.sample(0.0)
    assert monitor.as_dict()["host_memory"] == {}


def test_monitor_keeps_peaks(nvidia_smi: str, meminfo: Path, tmp_path: Path) -> None:
    monitor = ResourceMonitor(nvidia_smi=nvidia_smi, meminfo=meminfo, disk=tmp_path)
    for second in range(3):
        monitor.sample(float(second))
    data = monitor.as_dict()
    gpu = data["gpus"][0]  # type: ignore[index]
    assert (gpu["baseline_used_mib"], gpu["peak_used_mib"]) == (1000.0, 3000.0)
    assert gpu["memory_total_mib"] == 24564.0
    assert data["host_memory"] == {
        "baseline_used_gb": 12.288,
        "total_gb": 32.768,
        "peak_used_gb": 12.288,
    }
    disk = data["disk"]
    assert disk["start_gb"] >= disk["min_gb"] > 0  # type: ignore[index]
    assert data["timeline_columns"] == ["seconds", "host_used_gb", "gpu0_used_mib"]
    assert data["timeline"][-1] == [2.0, 12.288, 3000.0]  # type: ignore[index]


def test_run_passes_the_exit_code_through(
    nvidia_smi: str, meminfo: Path, tmp_path: Path
) -> None:
    out = tmp_path / "reports" / "resources.json"
    code = run_monitored(
        [sys.executable, "-c", "import time; time.sleep(0.3); raise SystemExit(3)"],
        out,
        interval_s=0.05,
        disk=tmp_path,
        nvidia_smi=nvidia_smi,
        meminfo=meminfo,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == report["exit_code"] == 3
    assert report["interrupted_by_signal"] is None
    assert report["samples"] >= 3
    assert report["wall_seconds"] >= 0.3
    assert report["cuda_version"] == "12.8"
    assert report["gpus"][0]["peak_used_mib"] >= 3000
    assert report["children_max_rss_gb"] > 0


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "r.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = ["monitor", "--out", str(out), "--interval", "0.05"]
    assert main([*args, "--label", "docs", "--", sys.executable, "-c", "pass"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert (report["exit_code"], report["label"]) == (0, "docs")
    lines = summary.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("| Step | Wall s | Host used GB, peak |")
    assert lines[2].startswith("| docs | 0 |")
    assert lines[2].endswith("| 0 |")
    assert main(args) == 1


def test_summary_row_marks_missing_counters() -> None:
    report = {
        "wall_seconds": 61.4,
        "exit_code": 1,
        "children_max_rss_gb": None,
        "host_memory": {"peak_used_gb": 3.256, "baseline_used_gb": 1.5},
        "gpus": [{"peak_used_mib": 20480.0}],
        "disk": {"min_gb": None},
    }

    row = summary_markdown(report, "gpu").splitlines()[2]

    assert row == "| gpu | 61 | 3.26 (baseline 1.50) | - | 20480 | - | 1 |"


def test_sigterm_stops_the_command_and_still_writes_the_report(
    tmp_path: Path,
) -> None:
    """A cancelled CI job: the command's own cleanup runs, then the report."""
    out = tmp_path / "r.json"
    marker = tmp_path / "cleaned-up"
    script = f"trap 'touch {marker}; exit 0' TERM; while :; do sleep 0.1; done"
    monitor = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "medbench.cli",
            "monitor",
            "--out",
            str(out),
            "--interval",
            "0.05",
            "--",
            "sh",
            "-c",
            script,
        ]
    )
    time.sleep(1.5)  # Python start-up, first sample, the shell's trap
    monitor.send_signal(signal.SIGTERM)
    assert monitor.wait(30) == 128 + signal.SIGTERM
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["interrupted_by_signal"] == signal.SIGTERM
    assert marker.exists()
