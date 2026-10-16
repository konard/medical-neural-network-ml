"""Collect the peak resources of every job in a set of workflow runs (issue #9).

Usage::

    python experiments/issue11_ci_resources.py OUT_DIR RUN_ID [RUN_ID ...]

For each run this saves the job list (``gh api .../jobs``) and downloads the
``resources-*`` artifacts written by ``medbench monitor``, plus the Model Tests
report artifacts that hold ``resources-*.json``. It then prints one Markdown
row per job: wall time from the API for every job, and host memory, largest
child RSS and free disk from the monitor report where the job has one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = "konard/medical-neural-network-ml"
REPORT_ARTIFACTS = ("medbench-cpu-reports", "medbench-ocr-baseline-reports")


def gh(*args: str) -> str:
    """Run the GitHub CLI and return its output."""
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout


def seconds(job: dict) -> float | None:
    """Return a job's wall time from its start and completion timestamps."""
    if not job.get("started_at") or not job.get("completed_at"):
        return None
    start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
    return (end - start).total_seconds()


def collect(out: Path, run_id: str) -> tuple[dict, list[dict], list[dict]]:
    """Save the run's jobs and resource reports under ``out/run_id``."""
    folder = out / run_id
    folder.mkdir(parents=True, exist_ok=True)
    run = json.loads(gh("api", f"repos/{REPO}/actions/runs/{run_id}"))
    jobs = json.loads(
        gh("api", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100")
    )["jobs"]
    (folder / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    patterns = ["resources-*", *REPORT_ARTIFACTS]
    for pattern in patterns:
        download = ["gh", "run", "download", run_id, "--repo", REPO]
        subprocess.run(
            [*download, "--pattern", pattern, "--dir", str(folder / "artifacts")],
            capture_output=True,
            check=False,
        )
    reports = []
    for path in sorted((folder / "artifacts").glob("**/*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "wall_seconds" in data and "host_memory" in data:
            data["file"] = str(path.relative_to(out))
            reports.append(data)
    return run, jobs, reports


def cell(value: float | None, digits: int = 2) -> str:
    """Format a number or a dash for a missing value."""
    return "-" if value is None else f"{value:.{digits}f}"


def rows(run: dict, jobs: list[dict], reports: list[dict]) -> list[str]:
    """Return one Markdown row per job, then one per monitored step."""
    lines = []
    for job in jobs:
        if job["conclusion"] == "skipped":
            continue
        lines.append(
            f"| {run['name']} | {job['name']} | {cell(seconds(job), 0)} "
            f"| {job['conclusion']} | | | | |"
        )
    for report in reports:
        host = report["host_memory"]
        disk = report.get("disk") or {}
        start, low = disk.get("start_gb"), disk.get("min_gb")
        used_disk = None if start is None or low is None else start - low
        lines.append(
            f"| {run['name']} | ↳ {report.get('label') or report['file']} "
            f"| {cell(report['wall_seconds'], 1)} | exit {report['exit_code']} "
            f"| {cell(host.get('baseline_used_gb'))} → {cell(host.get('peak_used_gb'))}"
            f" of {cell(host.get('total_gb'), 1)} "
            f"| {cell(report.get('children_max_rss_gb'), 3)} "
            f"| {cell(used_disk)} | {report['file']} |"
        )
    return lines


def main(argv: list[str]) -> int:
    """Collect the runs given on the command line and print the table."""
    out = Path(argv[0])
    print(
        "| Workflow | Job or monitored step | Wall s | Result "
        "| Host used GB (start → peak of total) | Largest child RSS GB "
        "| Disk used GB | Report |"
    )
    print("| --- | --- | ---: | --- | --- | ---: | ---: | --- |")
    for run_id in argv[1:]:
        run, jobs, reports = collect(out, run_id)
        print("\n".join(rows(run, jobs, reports)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
