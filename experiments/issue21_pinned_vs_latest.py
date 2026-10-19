#!/usr/bin/env python3
"""Table of pinned versus newest PyPI release for requirements files (issue #21).

Every ``name==version`` line (lock files) and every ``name>=version`` floor
(``pyproject.toml`` style lists given as plain requirement lines) is looked up
in the PyPI JSON API, so the tables in dev/log/issues/21 come from the registry.

Usage:
    python -I experiments/issue21_pinned_vs_latest.py FILE [FILE ...] > table.md
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

LINE = re.compile(
    r"""^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^]]*\])?\s*(==|>=)\s*([^\s;#,"']+)"""
)


def latest(name: str) -> tuple[str, str]:
    """Newest non-yanked release and its upload date."""
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{name}/json", timeout=30
    ) as response:
        data = json.load(response)
    version = data["info"]["version"]
    files = data["releases"].get(version) or [{}]
    return version, str(files[0].get("upload_time", ""))[:10]


def main(paths: list[str]) -> int:
    behind = 0
    for path in paths:
        print(f"\n### {path}\n")
        print("| Package | Pinned | Latest (PyPI) | Released | Status |")
        print("| --- | --- | --- | --- | --- |")
        for line in Path(path).read_text().splitlines():
            match = LINE.match(line.strip().strip('"').strip("',"))
            if not match:
                continue
            name, operator, pinned = match.groups()
            newest, released = latest(name)
            status = "latest" if pinned == newest else "**behind**"
            behind += pinned != newest
            print(f"| {name} | {operator}{pinned} | {newest} | {released} | {status} |")
    print(f"\n{behind} entr{'y' if behind == 1 else 'ies'} behind the newest release.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
