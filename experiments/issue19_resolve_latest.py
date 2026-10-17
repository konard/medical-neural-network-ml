#!/usr/bin/env python3
"""Resolve the newest release of every pinned dependency from its registry (issue #19).

Python packages come from the PyPI JSON API, GitHub Actions and pre-commit hook
repositories from the GitHub releases/tags API (through ``gh``), so the tables
in dev/log/issues/19 are built from the registries, not from memory.

Usage:
    python experiments/issue19_resolve_latest.py > dev/log/issues/19/pulls/20/data/latest.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request

PYPI = [
    # pyproject.toml build system, dev, imaging, mlx, docs extras
    "hatchling",
    "ruff",
    "mypy",
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    "pre-commit",
    "scriv",
    "pillow",
    "mlx-vlm",
    "mlx",
    "sphinx",
    "furo",
    "myst-parser",
    # tools installed by workflows and scripts
    "pip",
    "build",
    "twine",
    "pip-audit",
    "zizmor",
    "packaging",
    # scripts/unimedvl-requirements.txt
    "torch",
    "torchvision",
    "triton",
    "transformers",
    "accelerate",
    "safetensors",
    "tokenizers",
    "sentencepiece",
    "numpy",
    "opencv-python",
    "einops",
    "matplotlib",
    "contourpy",
    "cycler",
    "fonttools",
    "kiwisolver",
    "pyparsing",
    "python-dateutil",
    "flash-attn",
    # optional runtime imports of scripts/measure_training_memory.py
    "peft",
    "huggingface-hub",
]

ACTIONS = [
    "actions/cache",
    "actions/checkout",
    "actions/configure-pages",
    "actions/dependency-review-action",
    "actions/deploy-pages",
    "actions/download-artifact",
    "actions/setup-python",
    "actions/upload-artifact",
    "actions/upload-pages-artifact",
    "codecov/codecov-action",
    "docker/build-push-action",
    "docker/login-action",
    "docker/setup-buildx-action",
    "docker/setup-qemu-action",
    "github/codeql-action",
    "lycheeverse/lychee-action",
    "pypa/gh-action-pypi-publish",
    "zizmorcore/zizmor-action",
    "rhysd/actionlint",
    # pre-commit hook repositories
    "pre-commit/pre-commit-hooks",
    "astral-sh/ruff-pre-commit",
    "pre-commit/mirrors-mypy",
    # binaries pinned by scripts
    "ggml-org/llama.cpp",
    "Dao-AILab/flash-attention",
    "uni-medical/UniMedVL",
]


def pypi(name: str) -> dict[str, object]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as r:
        data = json.load(r)
    info = data["info"]
    version = info["version"]
    files = data["releases"].get(version, [])
    uploaded = min((f["upload_time"] for f in files), default=None)
    return {
        "latest": version,
        "requires_python": info.get("requires_python"),
        "uploaded": uploaded,
        "requires_dist": info.get("requires_dist"),
    }


def gh(path: str) -> object:
    out = subprocess.run(
        ["gh", "api", path], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)


def action(repo: str) -> dict[str, object]:
    try:
        release = gh(f"repos/{repo}/releases/latest")
        tag = release["tag_name"]  # type: ignore[index]
        published = release["published_at"]  # type: ignore[index]
    except subprocess.CalledProcessError:
        tags = gh(f"repos/{repo}/tags?per_page=1")
        tag = tags[0]["name"]  # type: ignore[index]
        published = None
    ref = gh(f"repos/{repo}/commits/{tag}")
    return {"latest": tag, "published": published, "sha": ref["sha"]}  # type: ignore[index]


def main() -> None:
    result: dict[str, dict[str, object]] = {"pypi": {}, "github": {}}
    for name in PYPI:
        try:
            result["pypi"][name] = pypi(name)
        except Exception as error:
            result["pypi"][name] = {"error": str(error)}
        print(f"pypi {name}: {result['pypi'][name].get('latest')}", file=sys.stderr)
    for repo in ACTIONS:
        try:
            result["github"][repo] = action(repo)
        except Exception as error:
            result["github"][repo] = {"error": str(error)}
        print(f"github {repo}: {result['github'][repo].get('latest')}", file=sys.stderr)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
