#!/usr/bin/env python3
"""Resolve and audit every Python dependency surface supported by the template.

The audit runs pip-audit over the resolved environment, never over pip
itself: the target environment is created ``--without-pip`` and populated
through the running interpreter's pip (issue #68). Auditing a bundled pip
reports known PYSEC advisories against the tool used to install, which the
application cannot fix and which masks advisories in real dependencies.

Fully pinned requirement files that are not installed with the package (the
UniMedVL GPU runtime of issue #6) are audited as files with ``--no-deps``.
Their known advisories that have no compatible fix are accepted one by one in
:data:`ACCEPTED_ADVISORIES` with the reason they do not apply; any advisory
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
PINNED_SURFACES = ("scripts/unimedvl-requirements.txt",)

# Why each advisory of the pinned UniMedVL runtime is accepted. torch stays
# at the newest release with a prebuilt flash-attn wheel and transformers at
# the last 4.x, which upstream's code (it imports 4.x internals) still runs
# on; see scripts/unimedvl-requirements.in.
_TORCH_OP = (
    "torch 2.8.0: crash or wrong result in an operator, profiler, "
    "torch.compile or torch.jit path that UniMedVL inference does not call"
)
_TORCH_LOAD = (
    "torch 2.8.0: pickle/pt2 loading; the adapter loads only a SHA-256-pinned "
    "safetensors file and never calls torch.load"
)
_HF_REPO = (
    "transformers 4.57.6: needs an attacker-controlled model repository; "
    "weights and configs are SHA-256-pinned, loaded from a local folder and "
    "the server runs with HF_HUB_OFFLINE=1"
)
ACCEPTED_ADVISORIES = {
    "PYSEC-2025-193": _TORCH_OP,  # rnn.unpack_sequence
    "PYSEC-2025-194": _TORCH_OP,  # torch.jit.script
    "PYSEC-2025-195": _TORCH_OP,  # lstm_cell
    "PYSEC-2025-203": _TORCH_OP,  # linalg.lu
    "PYSEC-2025-204": _TORCH_OP,  # rot90 with randn_like
    "PYSEC-2025-206": _TORCH_OP,  # nan_to_num().long()
    "PYSEC-2026-139": _TORCH_LOAD,  # pt2 loading
    "PYSEC-2026-2286": _TORCH_LOAD,  # weights_only unpickler
    "PYSEC-2025-217": (
        "transformers 4.57.6: code execution in the X-CLIP checkpoint-conversion "
        "script, which the adapter never imports"
    ),
    "PYSEC-2026-2288": "transformers 4.57.6: Trainer RNG state; no training",
    "PYSEC-2026-2289": _HF_REPO,  # _attn_implementation from config.json
    "PYSEC-2026-2290": _HF_REPO,  # LightGlue model loading
    "PYSEC-2026-4174": _HF_REPO,  # custom_generate download
    "PYSEC-2026-3929": "transformers 4.57.6: save_pretrained; nothing is saved",
}

# Dependency Review (security.yml) names advisories by GHSA id and fails on
# high severity. These are the high-severity aliases of the accepted
# advisories above, so a pull request that touches the pinned file is not
# failed for a reviewed exception; .github/workflows/security.yml allows
# exactly these ids (tests/test_audit_dependencies.py), nothing else.
REVIEW_ALLOWED_GHSAS = {
    "GHSA-29pf-2h5f-8g72": "PYSEC-2026-2289",
    "GHSA-fgcw-684q-jj6r": "PYSEC-2026-2290",
    "GHSA-x9r9-c232-4q39": "PYSEC-2026-4174",
    "GHSA-xrqw-3rrv-vx5w": "PYSEC-2026-3929",
}


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
