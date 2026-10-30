"""Query npm for every package in pnpm's toolchain and application lock documents.

Run with .venv/bin/python from the repository root. Uses Node's bundled semver
to explain upstream constraints, rather than approximating npm's range rules.
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

OUT = Path("dev/log/issues/25/pulls/26/data")
NODE = Path(".cache/issue25/node-v26.13.0-linux-x64/bin/node").resolve()
SEMVER = NODE.parent.parent / "lib/node_modules/npm/node_modules/semver"


def fetch(name: str) -> tuple[str, dict]:
    url = "https://registry.npmjs.org/" + urllib.parse.quote(name, safe="")
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
    return name, {
        "url": url,
        "latest": data["dist-tags"]["latest"],
        "versions": {
            version: {
                key: metadata.get(key, {})
                for key in ("dependencies", "optionalDependencies", "engines")
            }
            for version, metadata in data["versions"].items()
        },
    }


def main() -> None:
    docs = list(yaml.safe_load_all(Path(".github/tools/pnpm-lock.yaml").read_text()))
    packages = {key for doc in docs for key in doc["packages"]}
    entries = [key.rsplit("@", 1) for key in sorted(packages)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        registry = dict(pool.map(fetch, sorted({name for name, _ in entries})))
    parents: dict[str, list[dict]] = {}
    for name, version in entries:
        meta = registry[name]["versions"][version]
        for child, constraint in (
            meta["dependencies"] | meta["optionalDependencies"]
        ).items():
            if child in registry:
                parents.setdefault(child, []).append(
                    {
                        "parent": f"{name}@{version}",
                        "range": constraint,
                        "latest": registry[child]["latest"],
                    }
                )
    script = (
        f"const s=require({json.dumps(str(SEMVER))});"
        "let b='';process.stdin.on('data',c=>b+=c);process.stdin.on('end',()=>{"
        "let d=JSON.parse(b);for(const [n,p] of Object.entries(d))"
        "d[n]=p.filter(x=>!s.satisfies(x.latest,x.range));"
        "process.stdout.write(JSON.stringify(d));});"
    )
    blocked = json.loads(
        subprocess.check_output(
            [str(NODE), "-e", script], input=json.dumps(parents), text=True
        )
    )
    # Keep current/latest metadata only, not the registry's full historical tree.
    current = {name: set() for name in registry}
    for name, version in entries:
        current[name].update((version, registry[name]["latest"]))
    for name, data in registry.items():
        data["versions"] = {
            version: metadata
            for version, metadata in data["versions"].items()
            if version in current[name]
        }
    (OUT / "javascript-registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    lines = [
        "# Complete JavaScript and pnpm toolchain inventory",
        "",
        "npm registry queried 2026-10-30. Before this PR only secretlint and its",
        "preset were pinned (13.0.7); all transitive packages and the package manager",
        "were resolved by npx at execution time. All pnpm platform executables are",
        "included below, even those not installed on this Linux host.",
        "",
        "| Dependency | Before | Now | Registry latest | Reason if held |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, version in entries:
        latest = registry[name]["latest"]
        reason = "—"
        if latest != version:
            reason = "; ".join(
                f"{item['parent']} requires `{item['range']}`"
                for item in blocked.get(name, [])
            )
            if not reason:
                message = f"Unexplained held version: {name}@{version}"
                raise ValueError(message)
        before = (
            "13.0.7"
            if name in ("secretlint", "@secretlint/secretlint-rule-preset-recommend")
            else "unlocked / new"
        )
        lines.append(f"| {name} | {before} | {version} | {latest} | {reason} |")
    (OUT / "javascript-versions.md").write_text("\n".join(lines) + "\n")
    print(f"Inventoried {len(entries)} locked versions from {len(registry)} packages")


if __name__ == "__main__":
    main()
