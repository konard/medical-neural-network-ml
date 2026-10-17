"""Tests for scripts/check-dependency-graph.sh.

Dependency review fails every pull request when the repository's Dependency
graph setting is off. The script turns that into a warning on pull requests
and a failure with the setting to change on main. The SBOM endpoint is faked
locally, so no real GitHub API is called.
"""

from __future__ import annotations

import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.workflow_helpers import read_workflow, workflow_job_block

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check-dependency-graph.sh"
REPO = "owner/repo"


class FakeApi:
    """Answers the SBOM endpoint with a configurable status code."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.paths: list[str] = []
        self.auth: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                fake.paths.append(self.path)
                fake.auth.append(self.headers.get("Authorization", ""))
                body = b"{}"
                self.send_response(fake.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def run_check(
    tmp_path: Path, status: int, mode: str
) -> tuple[subprocess.CompletedProcess[str], str, FakeApi]:
    api = FakeApi(status)
    output = tmp_path / "github_output"
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            env={
                "PATH": "/usr/bin:/bin",
                "DEPENDENCY_GRAPH_MODE": mode,
                "GITHUB_API_URL": api.url,
                "GITHUB_SERVER_URL": "https://github.example",
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_TOKEN": "token-123",
                "GITHUB_OUTPUT": str(output),
                "DEPENDENCY_GRAPH_CURL_TIMEOUT": "5",
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        api.stop()
    written = output.read_text(encoding="utf-8") if output.exists() else ""
    return result, written, api


@pytest.mark.parametrize("mode", ["report", "require"])
def test_enabled_graph_lets_the_review_run(tmp_path: Path, mode: str) -> None:
    result, output, api = run_check(tmp_path, 200, mode)

    assert result.returncode == 0, result.stdout + result.stderr
    assert output == "enabled=true\n"
    assert api.paths == [f"/repos/{REPO}/dependency-graph/sbom"]
    assert api.auth == ["Bearer token-123"]


def test_disabled_graph_only_warns_on_pull_requests(tmp_path: Path) -> None:
    result, output, _ = run_check(tmp_path, 404, "report")

    assert result.returncode == 0, result.stdout + result.stderr
    assert output == "enabled=false\n"
    assert "::warning title=Dependency graph disabled::" in result.stdout
    assert "https://github.example/owner/repo/settings/security_analysis" in (
        result.stdout
    )


def test_disabled_graph_fails_main_with_the_setting_to_change(
    tmp_path: Path,
) -> None:
    result, output, _ = run_check(tmp_path, 404, "require")

    assert result.returncode == 1
    assert output == "enabled=false\n"
    assert "::error title=Dependency graph disabled::" in result.stdout
    assert "settings/security_analysis -> Dependency graph -> Enable" in (result.stdout)


@pytest.mark.parametrize("status", [403, 500])
def test_unclear_answer_is_not_reported_as_disabled(
    tmp_path: Path, status: int
) -> None:
    result, output, _ = run_check(tmp_path, status, "require")

    assert result.returncode == 0, result.stdout + result.stderr
    assert output == "enabled=true\n"
    assert f"HTTP {status}" in result.stdout


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    result, _, _ = run_check(tmp_path, 200, "strict")

    assert result.returncode == 2
    assert "must be report or require" in result.stdout


def test_security_workflow_wires_both_modes() -> None:
    workflow = read_workflow("security.yml")
    review = workflow_job_block(workflow, "dependency-review")
    graph = workflow_job_block(workflow, "dependency-graph")
    status = workflow_job_block(workflow, "pipeline-status")

    assert "DEPENDENCY_GRAPH_MODE: report" in review
    assert "if: steps.graph.outputs.enabled == 'true'" in review
    assert "if: github.event_name != 'pull_request'" in graph
    assert "DEPENDENCY_GRAPH_MODE: require" in graph
    assert "- dependency-graph" in status
