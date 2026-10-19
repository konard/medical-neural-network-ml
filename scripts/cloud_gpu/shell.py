"""Run external tools (terraform, ssh, kubectl) with logging and timeouts."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO

# Set CLOUD_GPU_TRACE=1 to print the full command line and environment
# overrides of every tool call (secrets are never passed on command lines).
TRACE = os.environ.get("CLOUD_GPU_TRACE", "0") == "1"


class CommandError(RuntimeError):
    """A tool exited with a non-zero status."""

    def __init__(self, argv: Sequence[str], returncode: int, output: str = "") -> None:
        """Keep the command, its status and the end of its output."""
        self.argv = list(argv)
        self.returncode = returncode
        self.output = output
        tail = output.strip()[-2000:]
        message = f"{shlex.join(self.argv[:6])} exited with {returncode}"
        super().__init__(f"{message}\n{tail}" if tail else message)


def log(message: str) -> None:
    """Print a progress line to stderr, so stdout stays machine-readable."""
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[cloud-gpu {stamp}] {message}", file=sys.stderr, flush=True)


def require_tool(name: str) -> str:
    """Return the path of ``name`` on PATH or raise with an install hint."""
    path = shutil.which(name)
    if path is None:
        message = f"{name} is not installed or not on PATH (see docs/cloud-gpu.md)"
        raise RuntimeError(message)
    return path


def run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
    timeout: float | None = None,
    stdin: IO[bytes] | bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv``; stream its output unless ``capture`` is set.

    ``env`` is added to the current environment. ``stdin`` may be bytes or an
    open binary file. With ``capture`` the output is returned as text.
    """
    argv = [str(part) for part in argv]
    if TRACE:
        extra = " ".join(f"{key}=…" for key in sorted(env or {}))
        log(f"$ {extra + ' ' if extra else ''}{shlex.join(argv)}")
    full_env = {**os.environ, **(env or {})}
    input_bytes = stdin if isinstance(stdin, bytes) else None
    stdin_file = None if isinstance(stdin, bytes) or stdin is None else stdin
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=full_env,
        input=input_bytes,
        stdin=stdin_file if input_bytes is None else None,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
        check=False,
    )
    output = result.stdout.decode("utf-8", "replace") if capture else ""
    if check and result.returncode != 0:
        raise CommandError(argv, result.returncode, output)
    return subprocess.CompletedProcess(argv, result.returncode, output, "")


def output(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    stdin: bytes | None = None,
) -> str:
    """Run ``argv`` and return its stdout; stderr goes to our stderr."""
    argv = [str(part) for part in argv]
    if TRACE:
        log(f"$ {shlex.join(argv)}")
    result = subprocess.run(
        argv,
        cwd=cwd,
        env={**os.environ, **env} if env else None,
        input=stdin,
        stdout=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    text = result.stdout.decode("utf-8", "replace")
    if result.returncode != 0:
        raise CommandError(argv, result.returncode, text)
    return text


def wait_until(
    what: str,
    probe: Callable[[], bool],
    *,
    timeout: float,
    interval: float = 10.0,
) -> None:
    """Call ``probe`` until it returns True; raise TimeoutError after ``timeout``."""
    deadline = time.monotonic() + timeout
    log(f"waiting for {what} (up to {timeout / 60:.0f} min)")
    while True:
        if probe():
            log(f"{what}: done")
            return
        if time.monotonic() >= deadline:
            message = f"timed out after {timeout:.0f} s waiting for {what}"
            raise TimeoutError(message)
        time.sleep(interval)
