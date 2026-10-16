"""Run a command and kill it when its resident memory passes a limit (issue #9).

Usage::

    python -I experiments/issue11_rss_watchdog.py LIMIT_GB COMMAND [ARGS ...]

The training probe checks free memory before it allocates, but ``--force``
skips that check. This outer guard stops a forced run before it reaches the
container's memory limit. It exits 137 when it kills the command, otherwise
with the command's exit code. Linux only: it reads ``/proc/PID/status``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def rss_kb(pid: int) -> int:
    """Return the resident set size of a process in kB, or 0 once it exits."""
    try:
        with Path(f"/proc/{pid}/status").open(encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def main(argv: list[str]) -> int:
    """Run ``argv[1:]`` and kill it above ``argv[0]`` GB."""
    limit_kb = int(float(argv[0]) * 1e6)
    proc = subprocess.Popen(argv[1:])
    peak = 0
    while proc.poll() is None:
        rss = rss_kb(proc.pid)
        peak = max(peak, rss)
        if rss > limit_kb:
            proc.kill()
            proc.wait()
            print(f"watchdog: killed at {rss / 1e6:.2f} GB", file=sys.stderr)
            return 137
        time.sleep(0.1)
    print(f"watchdog: peak {peak / 1e6:.3f} GB", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
