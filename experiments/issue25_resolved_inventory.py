"""Record every resolved Python dependency, registry release and held-pin reason.

Run with .venv/bin/python from the repository root after compiling the locks.
Only metadata is downloaded; no GPU packages are installed.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

OUT = Path("dev/log/issues/25/pulls/26/data")
SURFACES = (
    "scripts/unimedvl-requirements.txt",
    "scripts/infra-requirements.txt",
    "scripts/azure-cli-requirements.txt",
    str(OUT / "linux-python-resolution.txt"),
    str(OUT / "macos-python-resolution.txt"),
)


def read_pins(text: str) -> dict[str, str]:
    return {
        canonicalize_name(name): version
        for name, version in re.findall(r"^([\w.-]+)==([^\s;]+)", text, re.M)
    }


def metadata(key: tuple[str, str]) -> tuple[tuple[str, str], dict]:
    name, version = key
    suffix = f"/{version}" if version else ""
    url = f"https://pypi.org/pypi/{name}{suffix}/json"
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
    info = data["info"]
    return key, {
        "url": url,
        "version": info["version"],
        "requires_python": info["requires_python"],
        "requires_dist": info["requires_dist"] or [],
        "project_urls": info["project_urls"],
    }


def main() -> None:
    surfaces = {surface: read_pins(Path(surface).read_text()) for surface in SURFACES}
    old: dict[str, dict[str, str]] = {}
    for surface in SURFACES:
        result = subprocess.run(
            ["git", "show", f"HEAD:{surface}"],
            text=True,
            capture_output=True,
            check=False,
        )
        old[surface] = read_pins(result.stdout) if result.returncode == 0 else {}
    keys = {
        (name, version) for pins in surfaces.values() for name, version in pins.items()
    }
    keys |= {(name, "") for pins in surfaces.values() for name in pins}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        registry = dict(pool.map(metadata, sorted(keys)))
    raw = {
        f"{name}/{version or 'latest'}": info
        for (name, version), info in sorted(registry.items())
    }
    (OUT / "resolved-python-registry.json").write_text(json.dumps(raw, indent=2) + "\n")
    lines = [
        "# Complete resolved Python inventory",
        "",
        "Queried PyPI JSON on 2026-10-30. `unlocked` means the old installer resolved",
        "transitive dependencies at installation time. Registry metadata and parent",
        "constraints are preserved in resolved-python-registry.json. Separate environments",
        "retain upstream-required version differences; no dependency is silently omitted.",
        "",
    ]
    for surface, pins in surfaces.items():
        lines += [
            f"## {surface}",
            "",
            "| Dependency | Before | Now | Latest registry release | Reason if held |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, current in sorted(pins.items()):
            latest = registry[name, ""]["version"]
            constraints = []
            if Version(current) != Version(latest):
                for parent, version in pins.items():
                    for text in registry[parent, version]["requires_dist"]:
                        req = Requirement(text)
                        if canonicalize_name(
                            req.name
                        ) == name and not req.specifier.contains(
                            latest, prereleases=True
                        ):
                            constraints.append(f"{parent} {version}: {req.specifier}")
                if not constraints:
                    constraints.append(
                        "Inspect platform/Python wheel availability and registry metadata"
                    )
            reason = (
                "; ".join(sorted(set(constraints))).replace("|", "\\|") or "current"
            )
            before = old[surface].get(name, "unlocked/new")
            lines.append(f"| {name} | {before} | {current} | {latest} | {reason} |")
        lines.append("")
    (OUT / "resolved-python-versions.md").write_text("\n".join(lines))
    print(
        f"Recorded {len(registry)} metadata records across {len(surfaces)} environments"
    )


if __name__ == "__main__":
    main()
