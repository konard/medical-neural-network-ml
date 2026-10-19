"""GitHub REST calls for ephemeral self-hosted runners (stdlib only).

The token needs "Administration: read and write" on the repository (a
fine-grained personal access token or a GitHub App); the workflow's own
GITHUB_TOKEN cannot create runners. See docs/cloud-gpu.md.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from scripts.cloud_gpu.shell import log

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
TOKEN_VARIABLES = ("CLOUD_GPU_RUNNER_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


class GitHubError(RuntimeError):
    """The GitHub API answered with an error status."""


def token() -> str:
    """Return the first token found in the environment."""
    for name in TOKEN_VARIABLES:
        value = os.environ.get(name, "")
        if value:
            return value
    message = (
        f"set one of {', '.join(TOKEN_VARIABLES)} to a token with Administration: write"
    )
    raise GitHubError(message)


def request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    api: str | None = None,
) -> Any:
    """Call the REST API and return the decoded JSON (None for 204)."""
    url = f"{api or API}{path}"
    if not url.startswith("https://") and not url.startswith("http://127.0.0.1"):
        message = f"refusing a non-HTTPS GitHub API URL: {url}"
        raise GitHubError(message)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token()}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "medbench-cloud-gpu",
            **({"Content-Type": "application/json"} if data else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        message = f"GitHub API {method} {path}: {error.code} {detail}"
        raise GitHubError(message) from error
    return json.loads(raw) if raw else None


def generate_jit_config(
    repository: str,
    name: str,
    labels: list[str],
    *,
    runner_group_id: int = 1,
) -> tuple[int, str]:
    """Register one ephemeral runner; return its id and encoded config."""
    answer = request(
        "POST",
        f"/repos/{repository}/actions/runners/generate-jitconfig",
        {
            "name": name,
            "runner_group_id": runner_group_id,
            "labels": labels,
            "work_folder": "_work",
        },
    )
    runner_id = int(answer["runner"]["id"])
    log(f"registered ephemeral runner {name} (id {runner_id}) with {labels}")
    return runner_id, str(answer["encoded_jit_config"])


def runner_status(repository: str, runner_id: int) -> dict[str, Any] | None:
    """Return the runner, or None once GitHub has removed it."""
    try:
        answer: dict[str, Any] = request(
            "GET", f"/repos/{repository}/actions/runners/{runner_id}"
        )
    except GitHubError as error:
        if ": 404 " in str(error):
            return None
        raise
    return answer


def wait_runner(
    repository: str,
    runner_id: int,
    *,
    online_timeout: float = 1200,
    job_timeout: float = 4 * 3600,
    interval: float = 30,
) -> None:
    """Wait until the ephemeral runner has run its job and is gone.

    Raises TimeoutError if it never comes online or never finishes.
    """
    started = time.monotonic()
    seen_online = False
    while True:
        runner = runner_status(repository, runner_id)
        elapsed = time.monotonic() - started
        if runner is None:
            log(f"runner {runner_id} is gone: its job is done")
            return
        status, busy = runner.get("status"), runner.get("busy")
        if status == "online":
            seen_online = True
        if not seen_online and elapsed > online_timeout:
            message = (
                f"runner {runner_id} did not come online in {online_timeout:.0f} s"
            )
            raise TimeoutError(message)
        if elapsed > job_timeout:
            message = f"runner {runner_id} still exists after {job_timeout:.0f} s"
            raise TimeoutError(message)
        log(f"runner {runner_id}: {status}{', busy' if busy else ''}")
        time.sleep(interval)


def delete_runner(repository: str, runner_id: int) -> None:
    """Remove a runner registration; ignore one that is already gone."""
    try:
        request("DELETE", f"/repos/{repository}/actions/runners/{runner_id}")
    except GitHubError as error:
        if ": 404 " not in str(error):
            raise
    log(f"runner {runner_id} removed")


def stale_runners(repository: str, prefix: str = "medbench-") -> list[dict[str, Any]]:
    """Return offline runners whose name starts with ``prefix``."""
    found: list[dict[str, Any]] = []
    page = 1
    while True:
        answer = request(
            "GET", f"/repos/{repository}/actions/runners?per_page=100&page={page}"
        )
        runners = answer.get("runners", [])
        found += [
            runner
            for runner in runners
            if str(runner.get("name", "")).startswith(prefix)
            and runner.get("status") == "offline"
        ]
        if len(runners) < 100:
            return found
        page += 1
