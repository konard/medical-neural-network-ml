#!/usr/bin/env python3
"""Verify the built wheel and sdist before they can be published (issue #8).

    python scripts/check_distribution.py dist            # inspect the files
    python scripts/check_distribution.py dist --install  # and install them

Everything is compared with pyproject.toml and the source tree, so a change
to either that the build does not reflect fails here, in the Build Package
job, before any release job runs:

- one wheel and one sdist, named for the project and version;
- core metadata 2.4+ with ``License-Expression`` and one ``License-File`` per
  ``license-files`` entry, no legacy ``License`` field or license classifier,
  and the license files themselves, byte for byte;
- the ``medbench`` console script and every other ``[project.scripts]`` entry;
- the wheel holds exactly the files under the packaged ``src/`` directories,
  each matching its RECORD hash, and the sdist holds those plus the files
  named in ``SDIST_FILES`` and the license files; no images, scans or other media (see
  THIRD_PARTY_NOTICES.md), no absolute paths, links or ``..``.

``--install`` installs the wheel into a fresh virtual environment without an
index and runs each console script plus ``medbench sources`` (which reads the
packaged data), then builds a wheel from the sdist and checks that it holds
the same files.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import re
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - the build job runs 3.13.
    tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
MINIMUM_METADATA_VERSION = (2, 4)
# Files outside src/ that the sdist must carry besides the license files:
# what hatchling needs to build the wheel again, and the changelog.
SDIST_FILES = (
    ".gitignore",
    "CHANGELOG.md",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
)
# THIRD_PARTY_NOTICES.md promises that images, scans and DDXPlus files stay in
# ignored caches and are never bundled.
MEDIA_SUFFIXES = (
    ".bmp",
    ".csv",
    ".dcm",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".zip",
)
SMOKE_COMMANDS = (("medbench", "sources"),)
COMMAND_TIMEOUT_SECONDS = 600


class DistributionError(RuntimeError):
    """A built distribution does not match the project."""


def normalize(name: str) -> str:
    """Return the file-name form of a distribution name (PEP 427/625)."""
    return re.sub(r"[-_.]+", "_", name).lower()


def load_project(root: Path = ROOT) -> dict[str, Any]:
    """Return ``[project]`` plus the wheel package list from pyproject.toml."""
    if tomllib is None:  # pragma: no cover - Python < 3.11.
        message = "check_distribution.py needs Python 3.11+ (tomllib)"
        raise DistributionError(message)
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = dict(data["project"])
    wheel = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    project["wheel-packages"] = list(wheel["packages"])
    return project


def source_files(root: Path, packages: list[str]) -> dict[str, bytes]:
    """Map each packaged source file to its content, keyed by wheel path."""
    files = {}
    for package in packages:
        base = root / package
        for path in sorted(base.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            if path.suffix in (".pyc", ".pyo"):
                continue
            relative = PurePosixPath(base.name, *path.relative_to(base).parts)
            files[str(relative)] = path.read_bytes()
    return files


def find_distributions(dist: Path, project: dict[str, Any]) -> tuple[Path, Path]:
    """Return the only wheel and the only sdist, named for this version."""
    stem = f"{normalize(project['name'])}-{project['version']}"
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    expected_wheel = dist / f"{stem}-py3-none-any.whl"
    expected_sdist = dist / f"{stem}.tar.gz"
    if wheels != [expected_wheel] or sdists != [expected_sdist]:
        found = ", ".join(path.name for path in wheels + sdists) or "nothing"
        message = (
            f"{dist} must hold exactly {expected_wheel.name} and "
            f"{expected_sdist.name}; found {found}"
        )
        raise DistributionError(message)
    return expected_wheel, expected_sdist


def check_member_names(names: list[str], where: str) -> list[str]:
    """Reject absolute paths, ``..`` and bundled media."""
    problems = []
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            problems.append(f"{where}: unsafe path {name}")
        if path.suffix.lower() in MEDIA_SUFFIXES:
            problems.append(f"{where}: bundles media or a dataset file {name}")
    return problems


def check_metadata(text: str, project: dict[str, Any], where: str) -> list[str]:
    """Compare core metadata (METADATA or PKG-INFO) with pyproject.toml."""
    metadata = Parser().parsestr(text)
    problems = []
    version = metadata.get("Metadata-Version", "0")
    parsed = tuple(int(part) for part in re.findall(r"\d+", version)[:2])
    if parsed < MINIMUM_METADATA_VERSION:
        problems.append(f"{where}: Metadata-Version {version} is older than 2.4")
    expected = {
        "Name": project["name"],
        "Version": project["version"],
        "License-Expression": project["license"],
        "Requires-Python": project["requires-python"],
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            found = metadata.get(field)
            problems.append(f"{where}: {field} is {found!r}, expected {value!r}")
    license_files = sorted(metadata.get_all("License-File") or [])
    if license_files != sorted(project["license-files"]):
        problems.append(
            f"{where}: License-File is {license_files}, expected "
            f"{sorted(project['license-files'])}"
        )
    if metadata.get("License") is not None:
        problems.append(f"{where}: legacy License field next to the expression")
    classifiers = metadata.get_all("Classifier") or []
    problems += [
        f"{where}: license classifier {item!r} next to the expression"
        for item in classifiers
        if item.startswith("License ::")
    ]
    return problems


def _record_hash(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return "sha256=" + digest.rstrip(b"=").decode()


def check_wheel(wheel: Path, project: dict[str, Any], root: Path) -> list[str]:
    """Return every way the wheel differs from the project."""
    where = wheel.name
    dist_info = f"{normalize(project['name'])}-{project['version']}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        contents = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if not info.is_dir()
        }
    problems = check_member_names(list(contents), where)
    metadata = contents.get(f"{dist_info}/METADATA")
    if metadata is None:
        return [*problems, f"{where}: {dist_info}/METADATA missing"]
    problems += check_metadata(metadata.decode(), project, where)

    for name in project["license-files"]:
        license_copy = contents.get(f"{dist_info}/licenses/{name}")
        if license_copy is None:
            problems.append(f"{where}: {dist_info}/licenses/{name} missing")
        elif license_copy != (root / name).read_bytes():
            problems.append(f"{where}: {name} differs from the repository copy")

    entry_points = configparser.ConfigParser(delimiters=("=",))
    entry_points.optionxform = str  # type: ignore[assignment,method-assign]
    entry_points.read_string(
        contents.get(f"{dist_info}/entry_points.txt", b"").decode()
    )
    scripts = (
        dict(entry_points["console_scripts"])
        if entry_points.has_section("console_scripts")
        else {}
    )
    if scripts != project.get("scripts", {}):
        problems.append(
            f"{where}: console scripts {scripts}, expected {project.get('scripts')}"
        )

    expected = source_files(root, project["wheel-packages"])
    packaged = {name for name in contents if not name.startswith(f"{dist_info}/")}
    problems += [
        f"{where}: missing {name}" for name in sorted(set(expected) - packaged)
    ]
    problems += [
        f"{where}: unexpected {name}" for name in sorted(packaged - set(expected))
    ]
    problems += [
        f"{where}: {name} differs from the source tree"
        for name in sorted(packaged & set(expected))
        if contents[name] != expected[name]
    ]

    record_name = f"{dist_info}/RECORD"
    rows = list(csv.reader(io.StringIO(contents.get(record_name, b"").decode())))
    recorded = {row[0]: row[1] for row in rows if row}
    if set(recorded) != set(contents):
        problems.append(f"{where}: RECORD does not list exactly the wheel's files")
    problems += [
        f"{where}: RECORD hash mismatch for {name}"
        for name, digest in recorded.items()
        if name != record_name
        and name in contents
        and digest != _record_hash(contents[name])
    ]
    return problems


def check_sdist(sdist: Path, project: dict[str, Any], root: Path) -> list[str]:
    """Return every way the sdist differs from the project."""
    where = sdist.name
    prefix = f"{normalize(project['name'])}-{project['version']}/"
    contents: dict[str, bytes] = {}
    problems = []
    with tarfile.open(sdist, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                problems.append(f"{where}: {member.name} is a link or device")
                continue
            if not member.name.startswith(prefix):
                problems.append(f"{where}: {member.name} is outside {prefix}")
                continue
            extracted = archive.extractfile(member)
            contents[member.name[len(prefix) :]] = (
                extracted.read() if extracted else b""
            )
    problems += check_member_names(list(contents), where)
    if "PKG-INFO" not in contents:
        return [*problems, f"{where}: PKG-INFO missing"]
    problems += check_metadata(contents["PKG-INFO"].decode(), project, where)

    in_src = {
        str(PurePosixPath(package).parent / name): content
        for package in project["wheel-packages"]
        for name, content in source_files(root, [package]).items()
    }
    expected = set(in_src) | set(SDIST_FILES) | set(project["license-files"])
    problems += [
        f"{where}: missing {name}" for name in sorted(expected - set(contents))
    ]
    problems += [
        f"{where}: unexpected {name}" for name in sorted(set(contents) - expected)
    ]
    for name in project["license-files"]:
        if name in contents and contents[name] != (root / name).read_bytes():
            problems.append(f"{where}: {name} differs from the repository copy")
    problems += [
        f"{where}: {name} differs from the source tree"
        for name, content in in_src.items()
        if name in contents and contents[name] != content
    ]
    return problems


def _run(command: list[str], cwd: Path) -> str:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        message = (
            f"{' '.join(command)} exited {completed.returncode}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
        raise DistributionError(message)
    return completed.stdout


def install_smoke_test(wheel: Path, sdist: Path, project: dict[str, Any]) -> None:
    """Install the wheel offline, run its scripts and rebuild from the sdist."""
    with tempfile.TemporaryDirectory(prefix="medbench-dist-") as temporary:
        work = Path(temporary)
        venv.create(work / "venv", with_pip=True, clear=True)
        bin_dir = work / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
        python = str(bin_dir / "python")
        pip = [python, "-I", "-m", "pip", "--disable-pip-version-check"]
        _run([*pip, "install", "--no-index", "--no-deps", str(wheel)], work)
        for script in project.get("scripts", {}):
            _run([str(bin_dir / script), "--help"], work)
        for command in SMOKE_COMMANDS:
            _run([str(bin_dir / command[0]), *command[1:]], work)
        installed = _run(
            [
                python,
                "-I",
                "-c",
                "import importlib.metadata as m; d = m.metadata("
                f"{project['name']!r}); print(d['License-Expression'])",
            ],
            work,
        ).strip()
        if installed != project["license"]:
            message = f"installed License-Expression is {installed!r}"
            raise DistributionError(message)

        rebuilt = work / "rebuilt"
        _run([*pip, "wheel", "--no-deps", "-w", str(rebuilt), str(sdist)], work)
        with zipfile.ZipFile(wheel) as original:
            want = sorted(original.namelist())
        with zipfile.ZipFile(next(rebuilt.glob("*.whl"))) as again:
            have = sorted(again.namelist())
        if want != have:
            message = (
                "the wheel built from the sdist holds different files: "
                f"{sorted(set(want) ^ set(have))}"
            )
            raise DistributionError(message)
    print("Installed the wheel, ran its scripts and rebuilt it from the sdist.")


def main(argv: list[str] | None = None) -> int:
    """Check ``dist`` and, with ``--install``, install what is in it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path, help="directory with the wheel and sdist")
    parser.add_argument("--install", action="store_true", help="smoke-test install")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        project = load_project(args.root)
        wheel, sdist = find_distributions(args.dist, project)
        problems = check_wheel(wheel, project, args.root)
        problems += check_sdist(sdist, project, args.root)
        for problem in problems:
            print(f"::error title=Distribution check::{problem}")
        if problems:
            return 1
        print(
            f"{wheel.name} and {sdist.name}: License-Expression "
            f"{project['license']}, license files "
            f"{', '.join(project['license-files'])}, console scripts "
            f"{', '.join(project.get('scripts', {}))}; manifests match src/."
        )
        if args.install:
            install_smoke_test(wheel, sdist, project)
    except (DistributionError, subprocess.TimeoutExpired) as error:
        print(f"::error title=Distribution check::{error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
