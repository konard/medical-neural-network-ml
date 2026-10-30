#!/usr/bin/env python3
"""Resolve and audit every Python dependency surface supported by the template.

The audit runs pip-audit over the resolved environment, never over pip
itself: the target environment is created ``--without-pip`` and populated
through the running interpreter's pip (issue #68). Auditing a bundled pip
reports known PYSEC advisories against the tool used to install, which the
application cannot fix and which masks advisories in real dependencies.

Fully pinned requirement files that are not installed with the package (the
UniMedVL GPU runtime of issue #6) are audited as files with ``--no-deps``.
A known advisory without a compatible fix can be accepted in
:data:`ACCEPTED_ADVISORIES` with the reason it does not apply; any advisory
not listed there fails the audit.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PIP_AUDIT_VERSION = "2.10.1"
DEPENDENCY_SURFACES = ("pyproject.toml", "docs/requirements.txt")
PINNED_SURFACES = (
    "scripts/unimedvl-requirements.txt",
    "scripts/infra-requirements.txt",
    "scripts/azure-cli-requirements.txt",
)

# Why each advisory of the pinned UniMedVL runtime is accepted, keyed by id.
# Empty since issue #21 moved the runtime to torch 2.14.1 and transformers
# 5.20.1, which fixed the 14 torch 2.8.0 and transformers 4.57.6 advisories
# accepted before. An advisory without a compatible fix goes here with the
# package and the reason it does not apply (tests/test_audit_dependencies.py).
ACCEPTED_ADVISORIES: dict[str, str] = {}

# Dependency Review (security.yml) names advisories by GHSA id and fails on
# high severity. A high-severity GHSA alias of an accepted advisory goes here,
# mapped to its accepted id, and into the ``allow-ghsas`` input of
# .github/workflows/security.yml, which must list exactly these ids
# (tests/test_audit_dependencies.py). None are needed today.
REVIEW_ALLOWED_GHSAS: dict[str, str] = {}


def run(command: list[str], *, cwd: Path, capture: bool = False) -> str:
    """Run a command, streaming output so failures stay visible.

    pip-audit prints its advisory table on stdout and then exits non-zero;
    capturing that stdout under check=True threw the table away exactly when
    it mattered (issue #68). Stream by default and capture only when the
    caller needs the output value.
    """
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    output = completed.stdout.strip() if capture else ""
    if output:
        print(output)
    return output


def python_executable(venv: Path) -> Path:
    """Return the Python executable for a virtual environment."""
    scripts_directory = "Scripts" if sys.platform == "win32" else "bin"
    executable = "python.exe" if sys.platform == "win32" else "python"
    return venv / scripts_directory / executable


def project_install_target(project_root: Path) -> str:
    """Build an install target that resolves every declared optional extra."""
    pyproject_path = project_root / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    extras = sorted(pyproject.get("project", {}).get("optional-dependencies", {}))
    return f".[{','.join(extras)}]" if extras else "."


def audit_dependencies(project_root: Path) -> None:
    """Resolve application dependencies in isolation and audit the result."""
    missing = [
        surface
        for surface in (*DEPENDENCY_SURFACES, *PINNED_SURFACES)
        if not (project_root / surface).is_file()
    ]
    if missing:
        message = f"Unmapped or missing dependency surfaces: {', '.join(missing)}"
        raise FileNotFoundError(message)

    with tempfile.TemporaryDirectory(prefix="dependency-audit-") as temporary:
        temporary_root = Path(temporary)
        target_venv = temporary_root / "target"
        audit_venv = temporary_root / "audit"
        # --without-pip keeps pip itself out of the audited environment; the
        # running interpreter's pip populates the target through --python.
        run(
            [sys.executable, "-m", "venv", "--without-pip", str(target_venv)],
            cwd=project_root,
        )
        run([sys.executable, "-m", "venv", str(audit_venv)], cwd=project_root)

        target_python = python_executable(target_venv)
        audit_python = python_executable(audit_venv)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(target_python),
                "install",
                project_install_target(project_root),
            ],
            cwd=project_root,
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(target_python),
                "install",
                "-r",
                "docs/requirements.txt",
            ],
            cwd=project_root,
        )
        run(
            [
                str(audit_python),
                "-m",
                "pip",
                "install",
                f"pip-audit=={PIP_AUDIT_VERSION}",
            ],
            cwd=project_root,
        )
        site_packages = run(
            [
                str(target_python),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            cwd=project_root,
            capture=True,
        )
        run(
            [
                str(audit_python),
                "-m",
                "pip_audit",
                "--path",
                site_packages,
                "--skip-editable",
            ],
            cwd=project_root,
        )
        for surface in PINNED_SURFACES:
            run(pinned_audit_command(audit_python, surface), cwd=project_root)


def pinned_audit_command(audit_python: Path, surface: str) -> list[str]:
    """pip-audit a fully pinned requirement file without installing it.

    ``--no-deps`` audits exactly the listed pins (every transitive package is
    pinned in the file) and ``--disable-pip`` keeps pip from resolving, so a
    multi-gigabyte CUDA stack is never downloaded just to be audited.
    """
    ignored = [
        argument
        for advisory in sorted(ACCEPTED_ADVISORIES)
        for argument in ("--ignore-vuln", advisory)
    ]
    return [
        str(audit_python),
        "-m",
        "pip_audit",
        "--requirement",
        surface,
        "--no-deps",
        "--disable-pip",
        *ignored,
    ]


if __name__ == "__main__":
    audit_dependencies(Path(__file__).resolve().parents[1])
