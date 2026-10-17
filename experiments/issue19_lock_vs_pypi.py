"""Compare every pin in a requirements lock with PyPI (issue #19).

For each ``name==version`` line, print the newest release on PyPI and the
newest release whose ``Requires-Python`` admits the target Python, so a pin
that is behind can be told apart from one that is held by the interpreter.

    python -I experiments/issue19_lock_vs_pypi.py scripts/unimedvl-requirements.txt 3.10
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

PIN = re.compile(r"^([A-Za-z0-9_.-]+)==(\S+)")


def releases(name: str) -> dict[str, list[dict[str, object]]]:
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["releases"]


def newest(name: str, python: str | None) -> str | None:
    best: Version | None = None
    for raw, files in releases(name).items():
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        live = [f for f in files if not f.get("yanked")]
        if version.is_prerelease or not live:
            continue
        if python is not None:
            spec = str(live[0].get("requires_python") or "")
            try:
                if spec and python not in SpecifierSet(spec):
                    continue
            except InvalidSpecifier:
                continue
        if best is None or version > best:
            best = version
    return str(best) if best else None


def main(lock: str, python: str) -> int:
    behind = 0
    for line in Path(lock).read_text(encoding="utf-8").splitlines():
        match = PIN.match(line)
        if not match:
            continue
        name, pinned = match.groups()
        latest, for_python = newest(name, None), newest(name, python)
        if pinned == latest:
            status = "latest"
        elif pinned == for_python:
            status = f"latest for Python {python}"
        else:
            status = "BEHIND"
            behind += 1
        print(f"{name}\t{pinned}\t{latest}\t{for_python}\t{status}")
    return 0 if behind == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
