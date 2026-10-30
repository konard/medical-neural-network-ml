"""Snapshot current dependency pins and authoritative registry metadata (#25).

Run from the repository root. Saves compact raw data and reviewable tables in
``dev/log/issues/25/pulls/26/data``. Network failures stop the inventory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import tomllib
import urllib.request
from pathlib import Path


def get(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def pypi(name: str) -> tuple[str, dict]:
    try:
        data = get(f"https://pypi.org/pypi/{name}/json")
    except Exception as exc:
        message = f"PyPI query failed: {name}"
        raise RuntimeError(message) from exc
    info = data["info"]
    return name, {
        key: info.get(key)
        for key in ("version", "requires_python", "requires_dist", "project_urls")
    } | {
        "url": f"https://pypi.org/pypi/{name}/json",
        "files": data["releases"].get(info["version"], []),
    }


def snapshot_python(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    project = tomllib.loads(Path("pyproject.toml").read_text())
    pins: dict[str, list[dict]] = {}
    lines = [("pyproject.toml", req) for req in project["build-system"]["requires"]]
    for extra, reqs in project["project"]["optional-dependencies"].items():
        lines.extend((f"pyproject.toml [{extra}]", req) for req in reqs)
    for filename in [
        "docs/requirements.txt",
        "scripts/unimedvl-requirements.in",
        "scripts/unimedvl-requirements.txt",
    ]:
        lines.extend(
            (filename, line) for line in Path(filename).read_text().splitlines()
        )
    for filename in subprocess.check_output(
        ["git", "ls-files", "scripts", ".github", "infra", "ci"], text=True
    ).splitlines():
        if filename.endswith((".py", ".yml", ".yaml", ".sh")):
            content = Path(filename).read_text()
            lines.extend(
                (filename, match[0] + "==" + match[1])
                for match in re.findall(r"([a-z][a-z0-9_-]+)==([\d][\w.+-]*)", content)
            )
    for location, line in lines:
        match = re.match(r"\s*([\w.-]+)(?:\[[^]]*\])?\s*(?:==|>=)([\w.+-]+)", line)
        if match:
            name, pinned = match.groups()
            pins.setdefault(name, []).append({"location": location, "pinned": pinned})
    # Tools pinned as constants or installed by CI, rather than requirements.
    for name in [
        "uv",
        "pip",
        "pip-audit",
        "build",
        "twine",
        "zizmor",
        "ansible-core",
        "ansible-lint",
        "yamllint",
        "shellcheck-py",
        "azure-cli",
    ]:
        pins.setdefault(name, [])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        registry = dict(pool.map(pypi, sorted(pins)))
    (out / "python-registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    (out / "python-before.json").write_text(json.dumps(pins, indent=2) + "\n")
    table = [
        "| Dependency | Before | Latest on PyPI | Locations |",
        "| --- | --- | --- | --- |",
    ]
    for name, entries in sorted(pins.items()):
        table.append(
            f"| {name} | {', '.join(sorted({e['pinned'] for e in entries})) or 'tool'} | {registry[name]['version']} | {', '.join(sorted({e['location'] for e in entries}))} |"
        )
    (out / "python-before-latest.md").write_text("\n".join(table) + "\n")
    for name, entries in pins.items():
        before = sorted({entry["pinned"] for entry in entries})
        if any(v != registry[name]["version"] for v in before):
            print(f"{name}: {', '.join(before)} -> {registry[name]['version']}")


def snapshot_github(out: Path) -> None:
    repos = set(
        re.findall(
            r"uses: ([\w.-]+/[\w.-]+)",
            "\n".join(p.read_text() for p in Path(".github/workflows").glob("*.yml")),
        )
    )
    repos.update(
        re.findall(
            r"repo: https://github.com/(\S+)",
            Path(".pre-commit-config.yaml").read_text(),
        )
    )
    repos.update(
        [
            "hashicorp/terraform",
            "hashicorp/terraform-provider-aws",
            "hashicorp/terraform-provider-azurerm",
            "hashicorp/terraform-provider-google",
            "terraform-linters/tflint",
            "kubernetes/kubernetes",
            "kubernetes-sigs/kind",
            "yannh/kubeconform",
            "helm/helm",
            "actions/actions-runner-controller",
            "actions/runner",
            "NVIDIA/k8s-device-plugin",
            "NVIDIA/nvidia-container-toolkit",
            "k3s-io/k3s",
            "ggml-org/llama.cpp",
            "rhysd/actionlint",
            "aws/aws-cli",
            "python/cpython",
            "uni-medical/UniMedVL",
        ]
    )
    releases = {}
    for repo in sorted(repos):
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases/latest"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            releases[repo] = {"error": result.stderr, "response": result.stdout}
        else:
            release = json.loads(result.stdout)
            releases[repo] = {
                k: release.get(k)
                for k in ["tag_name", "published_at", "html_url", "body", "assets"]
            }
    (out / "github-releases.json").write_text(json.dumps(releases, indent=2) + "\n")
    for name, release in releases.items():
        print(f"{name}: {release.get('tag_name', release.get('error'))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot_python(args.output)
    snapshot_github(args.output)


if __name__ == "__main__":
    main()
