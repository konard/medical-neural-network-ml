#!/usr/bin/env python3
"""Compare the branch rulesets in .github/rulesets/ with the live repository.

The JSON files are the reviewed source of truth for the default branch
(issue #10); docs/repository-policy.md explains every rule and bypass.

    python scripts/repository_rulesets.py check            # read-only
    python scripts/repository_rulesets.py apply            # print the plan
    python scripts/repository_rulesets.py apply --yes      # write it

``check`` mirrors scripts/check-dependency-graph.sh. On pull requests (and
in forks) a missing or different ruleset is a warning: it is a repository
setting, not a defect in the change. On main and on the weekly scan it fails
with the command a repository admin has to run. ``apply`` needs an admin
token (``GH_TOKEN=$(gh auth token)``) and only writes with ``--yes``; a
ruleset PUT replaces the whole ruleset, so the files must hold every rule and
bypass actor, which is also why only rulesets named in them are touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RULESET_DIR = ROOT / ".github" / "rulesets"
REQUEST_TIMEOUT_SECONDS = 20
APPLY_COMMAND = (
    "GH_TOKEN=$(gh auth token) python scripts/repository_rulesets.py apply --yes"
)

Fetch = Callable[[str, str, "dict[str, Any] | None"], Any]


class RulesetError(RuntimeError):
    """The live rulesets could not be read or written."""


def load_desired(directory: Path = RULESET_DIR) -> list[dict[str, Any]]:
    """Read every ``*.json`` ruleset, sorted by file name."""
    rulesets = []
    for path in sorted(directory.glob("*.json")):
        ruleset = json.loads(path.read_text(encoding="utf-8"))
        if ruleset.get("name") != path.stem:
            message = f"{path.name}: name must be '{path.stem}'"
            raise RulesetError(message)
        rulesets.append(ruleset)
    return rulesets


def github_fetch(repository: str, token: str | None) -> Fetch:
    """Return a function that calls the GitHub REST API for ``repository``."""
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")

    def fetch(method: str, path: str, body: dict[str, Any] | None) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "medbench-repository-rulesets",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        url = f"{api}/repos/{repository}{path}"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read() or b"null")
        except HTTPError as error:
            detail = error.read(2000).decode(errors="replace")
            error.close()
            message = f"{method} {url} returned HTTP {error.code}: {detail}"
            raise RulesetError(message) from error
        except (OSError, URLError, ValueError) as error:
            message = f"{method} {url} failed: {error}"
            raise RulesetError(message) from error

    return fetch


def load_live(fetch: Fetch) -> dict[str, dict[str, Any]]:
    """Map each repository-level ruleset name to its full definition."""
    summaries = fetch("GET", "/rulesets?includes_parents=false&per_page=100", None)
    return {
        summary["name"]: fetch("GET", f"/rulesets/{summary['id']}", None)
        for summary in summaries
        if summary.get("source_type", "Repository") == "Repository"
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _rule_differences(
    want: list[dict[str, Any]], have: list[Any], where: str
) -> list[str]:
    """Match rules by type: exactly the rule types in ``want`` must be live."""
    live = {item.get("type"): item for item in have if isinstance(item, dict)}
    wanted = {item["type"] for item in want}
    found = [f"{where}: unexpected rule {kind}" for kind in live if kind not in wanted]
    for item in want:
        if item["type"] not in live:
            found.append(f"{where}: rule {item['type']} missing")
        else:
            found += differences(item, live[item["type"]], f"{where}[{item['type']}]")
    return found


def differences(want: Any, have: Any, where: str) -> list[str]:
    """Describe where ``have`` does not contain ``want``.

    GitHub adds defaults to what it returns (more rule parameters, ids), so
    a key missing from the file is ignored. Rules are matched by type and
    must be exactly the ones in the file; other lists compare as sets.
    """
    mismatch = [f"{where}: expected {_canonical(want)}, found {_canonical(have)}"]
    if isinstance(want, dict):
        if not isinstance(have, dict):
            return mismatch
        found = []
        for key, value in want.items():
            if key not in have:
                found.append(f"{where}.{key}: missing")
            else:
                found += differences(value, have[key], f"{where}.{key}")
        return found
    if isinstance(want, list) and not isinstance(have, list):
        return mismatch
    if (
        isinstance(want, list)
        and want
        and all(isinstance(item, dict) and "type" in item for item in want)
    ):
        return _rule_differences(want, have, where)
    if isinstance(want, list):
        same = sorted(map(_canonical, want)) == sorted(map(_canonical, have))
        return [] if same else mismatch
    return [] if want == have else mismatch


def drift(
    desired: list[dict[str, Any]], live: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Return (problems, notes) for the desired rulesets against the live ones."""
    problems: list[str] = []
    notes: list[str] = []
    for ruleset in desired:
        name = ruleset["name"]
        if name not in live:
            problems.append(f"ruleset '{name}' is not applied")
            continue
        want = dict(ruleset)
        if "bypass_actors" not in live[name]:
            # Only callers with write access see bypass actors; the job token
            # of a pull request or a read-only scan does not.
            want.pop("bypass_actors", None)
            notes.append(f"{name}: bypass actors not visible to this token")
        problems += differences(want, live[name], name)
    return problems, notes


def plan(
    desired: list[dict[str, Any]], live: dict[str, dict[str, Any]]
) -> list[tuple[str, str, dict[str, Any]]]:
    """Return the API calls that make the live rulesets match the files."""
    calls = []
    for ruleset in desired:
        current = live.get(ruleset["name"])
        if current is None:
            calls.append(("POST", "/rulesets", ruleset))
        elif differences(ruleset, current, ruleset["name"]):
            calls.append(("PUT", f"/rulesets/{current['id']}", ruleset))
    return calls


def event_mode(event_name: str | None) -> str:
    """Pull requests only report; everything else requires the rulesets."""
    if event_name in ("pull_request", "pull_request_target"):
        return "report"
    return "require"


def check(fetch: Fetch, repository: str, mode: str) -> int:
    """Print annotations for missing or different rulesets; return the exit code."""
    desired = load_desired()
    level = "error" if mode == "require" else "warning"
    try:
        live = load_live(fetch)
    except RulesetError as error:
        print(f"::{level} title=Branch rulesets unreadable::{error}")
        return 1 if mode == "require" else 0
    problems, notes = drift(desired, live)
    for note in notes:
        print(f"note: {note}")
    if not problems:
        names = ", ".join(ruleset["name"] for ruleset in desired)
        print(f"Branch rulesets match .github/rulesets/ for {repository}: {names}.")
        return 0
    for problem in problems:
        print(f"::{level} title=Branch ruleset drift::{problem}")
    print(
        f"::{level} title=Branch rulesets not applied::The default branch of "
        f"{repository} is not protected as .github/rulesets/ describes. A "
        "repository admin must read docs/repository-policy.md, then run: "
        f"{APPLY_COMMAND}"
    )
    return 1 if mode == "require" else 0


def apply(fetch: Fetch, repository: str, yes: bool) -> int:
    """Print the plan and, with ``yes``, send it."""
    desired = load_desired()
    calls = plan(desired, load_live(fetch))
    if not calls:
        print(f"{repository}: branch rulesets already match .github/rulesets/.")
        return 0
    for method, path, body in calls:
        print(f"{method} /repos/{repository}{path}  ({body['name']})")
        if yes:
            fetch(method, path, body)
    if not yes:
        print("Dry run: nothing was changed. Re-run with --yes to apply.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run ``check`` or ``apply``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "apply"))
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument(
        "--mode",
        choices=("report", "require"),
        default=event_mode(os.environ.get("GITHUB_EVENT_NAME")),
    )
    parser.add_argument("--yes", action="store_true", help="apply: really write")
    args = parser.parse_args(argv)
    if not args.repository:
        parser.error("--repository or GITHUB_REPOSITORY is required")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    fetch = github_fetch(args.repository, token)
    if args.command == "check":
        return check(fetch, args.repository, args.mode)
    try:
        return apply(fetch, args.repository, args.yes)
    except RulesetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
