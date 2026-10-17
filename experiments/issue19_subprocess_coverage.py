"""Measure how much of medbench the subprocess-launching tests cover (issue #19).

pytest-cov 7 dropped its ``.pth`` hook that measured Python subprocesses;
coverage's ``[run] patch = ["subprocess"]`` replaces it. This runs the test
file that starts ``python -m medbench.cli`` in a subprocess under pytest-cov,
with the coverage settings in pyproject.toml as they are, and prints the
covered percentage of the modules that subprocess runs.

Usage: python experiments/issue19_subprocess_coverage.py OUT.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("src/medbench/cli.py", "src/medbench/monitor.py")


def main() -> int:
    out = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "coverage.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                "tests/test_medbench_monitor.py",
                "--cov=src",
                f"--cov-report=json:{report}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        data = json.loads(report.read_text(encoding="utf-8"))
    summary = {
        "total_percent": round(data["totals"]["percent_covered"], 1),
        **{
            module: round(data["files"][module]["summary"]["percent_covered"], 1)
            for module in MODULES
        },
    }
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
