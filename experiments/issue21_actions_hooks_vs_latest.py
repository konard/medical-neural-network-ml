#!/usr/bin/env python3
"""Table of pinned versus newest release for actions and pre-commit hooks (issue #21).

Every ``uses: owner/repo[/path]@<sha> # vX.Y.Z`` in .github/workflows and every
``repo:``/``rev:`` pair in .pre-commit-config.yaml is compared with the newest
release tag on GitHub, read with ``git ls-remote --tags``. For actions it also
checks that the pinned commit is the commit the comment's tag points to.

"Newest" is the highest final ``vX.Y.Z`` (or ``X.Y.Z``) tag: some repositories
publish other tags as their "latest release" (codeql-action's CLI bundles).

Usage:
    python -I experiments/issue21_actions_hooks_vs_latest.py REPO_ROOT > table.md
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

USES = re.compile(r"uses: ([\w.-]+/[\w.-]+)(?:/[\w./-]+)?@([0-9a-f]{40}) # (\S+)")
HOOK = re.compile(r"- repo: https://github.com/(\S+)\n\s+rev: (\S+)")
FINAL = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


def tags(repo: str) -> dict[str, str]:
    """Tag name -> commit, with annotated tags dereferenced."""
    output = subprocess.run(
        ["git", "ls-remote", "--tags", f"https://github.com/{repo}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commits: dict[str, str] = {}
    for line in output.splitlines():
        sha, ref = line.split("\t")
        name = ref.removeprefix("refs/tags/")
        if name.endswith("^{}"):
            commits[name[:-3]] = sha
        else:
            commits.setdefault(name, sha)
    return commits


def newest(names: list[str], like: str) -> str:
    """Highest final tag written like ``like`` (with or without the ``v``)."""
    prefix = "v" if like.startswith("v") else ""
    finals = [
        (tuple(int(part) for part in match.groups()), name)
        for name in names
        if (match := FINAL.fullmatch(name)) and name.startswith(prefix)
    ]
    return max(finals)[1]


def main(root: Path) -> int:
    behind = 0
    actions: dict[str, tuple[str, str]] = {}
    for workflow in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        for repo, sha, tag in USES.findall(workflow.read_text()):
            if actions.setdefault(repo, (sha, tag)) != (sha, tag):
                print(f"{repo} is pinned twice: {actions[repo]} and {(sha, tag)}")
                behind += 1

    print("| Action | Pinned | Pinned SHA is the tag's commit | Latest | Status |")
    print("| --- | --- | --- | --- | --- |")
    for repo, (sha, tag) in sorted(actions.items()):
        commits = tags(repo)
        latest = newest(list(commits), tag)
        matches = "yes" if commits.get(tag) == sha else f"**no** ({commits.get(tag)})"
        status = "latest" if tag == latest else "**behind**"
        behind += tag != latest or commits.get(tag) != sha
        print(f"| {repo} | {tag} `{sha[:12]}` | {matches} | {latest} | {status} |")

    print("\n| Hook repository | Pinned rev | Latest | Status |")
    print("| --- | --- | --- | --- |")
    config = (root / ".pre-commit-config.yaml").read_text()
    for repo, rev in HOOK.findall(config):
        latest = newest(list(tags(repo)), rev)
        status = "latest" if rev == latest else "**behind**"
        behind += rev != latest
        print(f"| {repo} | {rev} | {latest} | {status} |")

    print(f"\n{behind} entr{'y' if behind == 1 else 'ies'} behind or mismatched.")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
