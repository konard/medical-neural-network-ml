"""Repository policy for issue #10: pinned actions, update PRs, branch rules.

The files are read as text, like the other workflow tests, so no YAML
library is needed.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.workflow_helpers import (
    ROOT,
    WORKFLOWS,
    job_condition,
    workflow_job_block,
    workflow_job_names,
)

DEPENDABOT = ROOT / ".github" / "dependabot.yml"


def dependabot_entries() -> dict[str, str]:
    """Map each ``package-ecosystem`` to its block in dependabot.yml."""
    text = DEPENDABOT.read_text(encoding="utf-8")
    blocks = re.split(r"^  - (?=package-ecosystem:)", text, flags=re.MULTILINE)[1:]
    entries = {}
    for block in blocks:
        ecosystem = re.match(r"package-ecosystem: ([\w-]+)", block)
        assert ecosystem, block
        entries[ecosystem.group(1)] = block
    return entries


def test_dependabot_updates_actions_and_python_with_a_cooldown() -> None:
    """Hash pins only stay current if something proposes the new hashes."""
    entries = dependabot_entries()
    assert set(entries) == {"github-actions", "pip", "pre-commit", "uv"}
    for ecosystem, block in entries.items():
        assert re.search(r"^      default-days: 7$", block, re.M), ecosystem
    assert "interval: weekly" in entries["github-actions"]
    assert "directory: /\n" in entries["github-actions"]
    assert "directory: /\n" in entries["pre-commit"]
    assert "directory: /scripts\n" in entries["uv"]
    pip = entries["pip"]
    assert "      - /\n" in pip
    assert "      - /docs\n" in pip


def test_dependabot_moves_python_and_its_hooks_together() -> None:
    """pip and pre-commit share one weekly pull request (issue #19).

    The hooks pin the dev extra's ruff, mypy and pillow
    (tests/test_project_metadata.py), so separate pull requests would each
    fail the tests. A multi-ecosystem group needs ``patterns`` on every
    member and keeps ``commit-message`` and the schedule at group level.
    """
    text = DEPENDABOT.read_text(encoding="utf-8")
    group = text.split("\nupdates:\n", 1)[0]
    assert re.search(r"^multi-ecosystem-groups:\n  python:\n", group, re.M)
    assert "interval: weekly" in group
    assert "prefix: deps" in group
    entries = dependabot_entries()
    for ecosystem in ("pip", "pre-commit"):
        block = entries[ecosystem]
        assert "multi-ecosystem-group: python\n" in block, ecosystem
        assert "    patterns:\n      - '*'\n" in block, ecosystem
        assert "commit-message" not in block, ecosystem
        assert "schedule" not in block, ecosystem


def test_dependabot_raises_the_pip_floors_with_the_hooks() -> None:
    """A new ruff or mypy moves the hook rev and the dev extra's floor at once.

    By default Dependabot treats this package as a library and only widens a
    range that already admits the new release: ``ruff>=0.17.1,<0.18`` stays
    put for 0.17.2 while the hook goes to v0.17.2, and the "same version"
    tests fail in the group's own pull request. ``increase`` raises the floor
    to the new release, which also keeps every floor at the latest
    (issue #19).
    """
    pip = dependabot_entries()["pip"]
    assert "    versioning-strategy: increase\n" in pip


def test_dependabot_updates_the_unimedvl_lock_with_uv() -> None:
    """The UniMedVL lock moves in its own uv pull request (issue #21).

    Until issue #21 torch had to match a hash-pinned flash-attn wheel, so
    the files were left out; now nothing else pins torch or transformers.
    Dependabot's pip ecosystem scans every requirements file under ``/``
    (it opened #13, #15, #16 and #17 against these), so pip still excludes
    them and only uv updates them. uv's pull request stays out of the
    ``python`` group: with pyproject.toml beside it the lock would count as
    a scripts/ source change and need a changelog fragment, which
    Dependabot cannot write.
    """
    entries = dependabot_entries()
    pip = entries["pip"]
    (pattern,) = re.findall(r"^    exclude-paths:\n      - (\S+)$", pip, re.M)
    scripts = ROOT / "scripts"
    excluded = {path.name for path in scripts.glob(pattern.removeprefix("scripts/"))}
    assert excluded == {"unimedvl-requirements.in", "unimedvl-requirements.txt"}
    uv = entries["uv"]
    assert "multi-ecosystem-group" not in uv
    assert "      interval: weekly\n" in uv
    assert "      prefix: deps\n" in uv
    audit = (ROOT / "scripts" / "audit_dependencies.py").read_text(encoding="utf-8")
    assert '"scripts/unimedvl-requirements.txt"' in audit


def test_unimedvl_lock_is_compiled_where_dependabot_compiles_it() -> None:
    """Dependabot runs ``uv pip compile`` in /scripts, the uv entry's directory.

    uv writes the input path into every ``# via -r`` line, so a lock compiled
    from the repository root would change on every line in Dependabot's next
    pull request. Dependabot keeps only ``--python-version`` from the header,
    which the lock has.
    """
    lock = (ROOT / "scripts" / "unimedvl-requirements.txt").read_text()
    header = [line for line in lock.splitlines() if line.startswith("#    uv ")]
    assert len(header) == 1
    assert re.fullmatch(
        r"#    uv pip compile unimedvl-requirements\.in -o unimedvl-requirements\.txt"
        r" --python-version [\d.]+ --python-platform x86_64-manylinux_2_28",
        header[0],
    ), header
    inputs = set(re.findall(r"-r (\S+)", lock))
    assert inputs == {"unimedvl-requirements.in"}


def test_every_workflow_action_can_be_updated_by_dependabot() -> None:
    """Dependabot's github-actions ecosystem reads every workflow in /."""
    actions = set()
    for path in WORKFLOWS.glob("*.y*ml"):
        actions |= set(
            re.findall(
                r"uses:\s+([\w.-]+/[\w.-]+)[\w/.-]*@[0-9a-f]{40}", path.read_text()
            )
        )
    assert {"actions/checkout", "pypa/gh-action-pypi-publish"} <= actions


# --- Branch rulesets (.github/rulesets/, scripts/repository_rulesets.py) ---

RULESETS = ROOT / ".github" / "rulesets"
GITHUB_ACTIONS_APP = 15368
ADMIN_ROLE = 5
spec = importlib.util.spec_from_file_location(
    "repository_rulesets", ROOT / "scripts" / "repository_rulesets.py"
)
rulesets = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules[spec.name] = rulesets  # type: ignore[union-attr]
spec.loader.exec_module(rulesets)  # type: ignore[union-attr]


def desired() -> dict[str, dict[str, Any]]:
    return {ruleset["name"]: ruleset for ruleset in rulesets.load_desired()}


def rule(ruleset: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(item for item in ruleset["rules"] if item["type"] == kind)


def workflow_triggers(text: str) -> str:
    return text.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]


def test_rulesets_protect_only_the_default_branch() -> None:
    assert set(desired()) == {"main-history", "main-checks", "main-review"}
    for ruleset in desired().values():
        assert ruleset["target"] == "branch"
        assert ruleset["enforcement"] == "active"
        assert ruleset["conditions"] == {
            "ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}
        }


def test_history_cannot_be_rewritten_or_deleted_by_anyone() -> None:
    """The release writer only fast-forwards main, so nobody needs a bypass."""
    history = desired()["main-history"]
    assert history["bypass_actors"] == []
    assert {item["type"] for item in history["rules"]} == {
        "deletion",
        "non_fast_forward",
    }


def test_bypass_is_limited_to_the_release_bot_and_admin_merges() -> None:
    """Issue #10: a narrowly scoped release-bot bypass, never an exemption.

    The GitHub Actions app pushes the version commit of manual-release.
    Admins may merge their own pull request without a second approval (one
    maintainer cannot approve their own), but only through a pull request,
    and never past the required checks: those live in a separate ruleset.
    """
    release_bot = {
        "actor_id": GITHUB_ACTIONS_APP,
        "actor_type": "Integration",
        "bypass_mode": "always",
    }
    assert desired()["main-checks"]["bypass_actors"] == [release_bot]
    assert desired()["main-review"]["bypass_actors"] == [
        release_bot,
        {
            "actor_id": ADMIN_ROLE,
            "actor_type": "RepositoryRole",
            "bypass_mode": "pull_request",
        },
    ]
    kinds = [item["type"] for r in desired().values() for item in r["rules"]]
    assert len(kinds) == len(set(kinds)), "each rule lives in exactly one ruleset"


def test_pull_requests_need_a_fresh_approval_and_resolved_threads() -> None:
    parameters = rule(desired()["main-review"], "pull_request")["parameters"]
    assert parameters["required_approving_review_count"] == 1
    assert parameters["dismiss_stale_reviews_on_push"] is True
    assert parameters["required_review_thread_resolution"] is True


def test_required_checks_are_terminal_gates_that_report_on_every_pull_request() -> None:
    """A required check that never reports blocks the pull request forever.

    Each context must be the unique name of a terminal status gate in a
    workflow that runs on every pull request (no paths or branches filter),
    and must come from GitHub Actions so nobody can post a fake status.
    """
    parameters = rule(desired()["main-checks"], "required_status_checks")["parameters"]
    checks = parameters["required_status_checks"]
    assert {check["integration_id"] for check in checks} == {GITHUB_ACTIONS_APP}
    contexts = [check["context"] for check in checks]
    assert contexts == [
        "Pipeline Status (CI/CD)",
        "Pipeline Status (Compatibility)",
        "Pipeline Status (Security)",
    ]
    names: dict[str, list[str]] = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        for name in re.findall(r"^    name: (.+)$", text, re.MULTILINE):
            names.setdefault(name, []).append(path.name)
    for context in contexts:
        assert len(names.get(context, [])) == 1, f"{context}: {names.get(context)}"
        text = (WORKFLOWS / names[context][0]).read_text(encoding="utf-8")
        gate = workflow_job_block(text, "pipeline-status")
        assert f"    name: {context}\n" in gate
        triggers = workflow_triggers(text)
        pull_request = re.search(
            r"^  pull_request:\n((?:    .*\n|  #.*\n)*)", triggers, re.M
        )
        assert pull_request, f"{names[context][0]} does not run on pull requests"
        assert not re.search(
            r"^    (paths|paths-ignore|branches)", pull_request.group(1), re.M
        )
    gates = [name for name in names if name.startswith("Pipeline Status")]
    assert all(len(names[gate]) == 1 for gate in gates), "gate names must be unique"


def test_only_reviewed_release_jobs_can_write_to_the_repository() -> None:
    """The bypass is only as narrow as the set of jobs holding a write token."""
    writers = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        for job in workflow_job_names(text):
            block = workflow_job_block(text, job)
            writers += [
                (path.name, job, scope)
                for scope in re.findall(r"^\s+([a-z-]+): write$", block, re.MULTILINE)
            ]
    assert sorted(writers) == [
        ("cloud-gpu-tests.yml", "orchestrate", "actions"),
        ("cloud-gpu-tests.yml", "orchestrate", "id-token"),
        ("cloud-gpu-tests.yml", "reap", "id-token"),
        ("docs.yml", "build", "pages"),
        ("docs.yml", "deploy", "id-token"),
        ("docs.yml", "deploy", "pages"),
        ("release.yml", "auto-release", "contents"),
        ("release.yml", "auto-release", "id-token"),
        ("release.yml", "manual-release", "contents"),
        ("release.yml", "manual-release", "id-token"),
        ("release.yml", "release-preflight", "id-token"),
        ("security.yml", "codeql", "security-events"),
        ("security.yml", "dependency-review", "pull-requests"),
    ], "a new write scope needs a review in docs/repository-policy.md"
    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main'" in (
        job_condition(release, "auto-release")
    )
    assert "github.event_name == 'workflow_dispatch'" in job_condition(
        release, "manual-release"
    )


def test_fork_pull_requests_never_run_with_secrets_or_write_tokens() -> None:
    """``pull_request`` from a fork gets a read-only token and no secrets.

    ``pull_request_target`` and ``workflow_run`` would run with the base
    repository's secrets next to fork-controlled code, so neither is used.
    Every secret a job reads is listed here and in the policy document.
    """
    secrets = set()
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        triggers = workflow_triggers(text)
        assert "pull_request_target" not in triggers, path.name
        assert "workflow_run" not in triggers, path.name
        for job in workflow_job_names(text):
            for secret in re.findall(
                r"secrets\.([A-Z_]+)", workflow_job_block(text, job)
            ):
                secrets.add((path.name, job, secret))
    assert secrets == {
        ("cloud-gpu-tests.yml", "orchestrate", "CLOUD_GPU_RUNNER_TOKEN"),
        ("cloud-gpu-tests.yml", "orchestrate", "HETZNER_ROBOT_PASSWORD"),
        ("cloud-gpu-tests.yml", "orchestrate", "HETZNER_ROBOT_USER"),
        ("cloud-gpu-tests.yml", "reap", "CLOUD_GPU_RUNNER_TOKEN"),
        ("cloud-gpu-tests.yml", "reap", "HETZNER_ROBOT_PASSWORD"),
        ("cloud-gpu-tests.yml", "reap", "HETZNER_ROBOT_USER"),
        ("links.yml", "link-checker", "GITHUB_TOKEN"),
        ("release.yml", "auto-release", "GITHUB_TOKEN"),
        ("release.yml", "manual-release", "GITHUB_TOKEN"),
        ("release.yml", "release-preflight", "DOCKERHUB_TOKEN"),
        ("release.yml", "release-preflight", "DOCKERHUB_USERNAME"),
        ("release.yml", "test", "CODECOV_TOKEN"),
        ("release.yml", "docker-publish-config", "DOCKERHUB_TOKEN"),
        ("release.yml", "docker-publish-build", "DOCKERHUB_TOKEN"),
        ("release.yml", "docker-publish", "DOCKERHUB_TOKEN"),
    }
    policy = (ROOT / "docs" / "repository-policy.md").read_text(encoding="utf-8")
    for _, _, secret in secrets:
        assert secret in policy, f"{secret} is not explained for fork pull requests"


def test_security_workflow_checks_the_live_rulesets_read_only() -> None:
    text = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
    job = workflow_job_block(text, "rulesets")
    assert "    name: Branch Rulesets" in job
    assert "contents: read" in job
    assert "write" not in job
    assert "run: python scripts/repository_rulesets.py check\n" in job + "\n"
    assert "apply" not in job


class FakeGitHub:
    """Records calls and answers like the rulesets REST API."""

    def __init__(self, live: list[dict[str, Any]] | None = None, fail: bool = False):
        self.live = live or []
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method: str, path: str, body: dict[str, Any] | None) -> Any:
        self.calls.append((method, path))
        if self.fail:
            message = "HTTP 503"
            raise rulesets.RulesetError(message)
        if path.startswith("/rulesets?"):
            return [
                {"id": item["id"], "name": item["name"], "source_type": "Repository"}
                for item in self.live
            ]
        if method == "GET":
            return next(item for item in self.live if f"/rulesets/{item['id']}" == path)
        return body


def as_live(ruleset: dict[str, Any], ruleset_id: int, bypass: bool) -> dict[str, Any]:
    """What GitHub returns: ids, extra defaults, bypass only for writers."""
    live = json.loads(json.dumps(ruleset))
    live.update(id=ruleset_id, source="konard/x", source_type="Repository")
    for item in live["rules"]:
        if item["type"] == "pull_request":
            item["parameters"]["required_reviewers"] = []
            item["parameters"]["allowed_merge_methods"].reverse()
    if not bypass:
        del live["bypass_actors"]
    return live


def all_live(bypass: bool = True) -> list[dict[str, Any]]:
    return [
        as_live(ruleset, index, bypass)
        for index, ruleset in enumerate(desired().values(), 1)
    ]


def test_missing_rulesets_fail_main_with_the_command_to_run(capsys) -> None:
    assert rulesets.check(FakeGitHub(), "konard/x", "require") == 1
    out = capsys.readouterr().out
    assert (
        "::error title=Branch ruleset drift::ruleset 'main-checks' is not applied"
        in out
    )
    assert "docs/repository-policy.md" in out
    assert rulesets.APPLY_COMMAND in out


def test_missing_rulesets_only_warn_on_pull_requests(capsys) -> None:
    assert rulesets.event_mode("pull_request") == "report"
    assert rulesets.event_mode("push") == "require"
    assert rulesets.event_mode("schedule") == "require"
    assert rulesets.check(FakeGitHub(), "konard/x", "report") == 0
    assert "::warning title=Branch rulesets not applied::" in capsys.readouterr().out


def test_unreadable_rulesets_are_not_treated_as_applied(capsys) -> None:
    assert rulesets.check(FakeGitHub(fail=True), "konard/x", "require") == 1
    assert (
        "::error title=Branch rulesets unreadable::HTTP 503" in capsys.readouterr().out
    )
    assert rulesets.check(FakeGitHub(fail=True), "konard/x", "report") == 0


def test_applied_rulesets_match_despite_github_defaults(capsys) -> None:
    assert (
        rulesets.check(FakeGitHub(all_live(bypass=False)), "konard/x", "require") == 0
    )
    out = capsys.readouterr().out
    assert "Branch rulesets match" in out
    assert "bypass actors not visible to this token" in out
    assert rulesets.check(FakeGitHub(all_live()), "konard/x", "require") == 0


def test_weakened_rulesets_are_reported() -> None:
    live = all_live()
    checks = rule(live[0], "required_status_checks")["parameters"]
    checks["required_status_checks"].pop()
    live[1]["rules"].append({"type": "update"})
    live[2]["bypass_actors"][1]["bypass_mode"] = "always"
    problems, _ = rulesets.drift(list(desired().values()), {r["name"]: r for r in live})
    assert len(problems) == 3, problems
    assert any("required_status_checks" in problem for problem in problems)
    assert any("unexpected rule update" in problem for problem in problems)
    assert any("bypass_actors" in problem for problem in problems)


def test_a_bypass_added_to_the_history_ruleset_is_reported() -> None:
    live = all_live()
    history = next(r for r in live if r["name"] == "main-history")
    history["bypass_actors"].append(
        {
            "actor_id": ADMIN_ROLE,
            "actor_type": "RepositoryRole",
            "bypass_mode": "always",
        }
    )
    problems, _ = rulesets.drift(list(desired().values()), {r["name"]: r for r in live})
    assert problems == [
        "main-history.bypass_actors: expected [], found "
        + json.dumps(history["bypass_actors"], sort_keys=True)
    ]


def test_apply_is_a_dry_run_unless_confirmed(capsys) -> None:
    github = FakeGitHub()
    assert rulesets.apply(github, "konard/x", yes=False) == 0
    assert all(method == "GET" for method, _ in github.calls)
    assert "Dry run: nothing was changed" in capsys.readouterr().out

    github = FakeGitHub()
    rulesets.apply(github, "konard/x", yes=True)
    assert [call for call in github.calls if call[0] != "GET"] == [
        ("POST", "/rulesets")
    ] * 3


def test_apply_replaces_only_drifted_rulesets() -> None:
    live = all_live()
    live[1]["enforcement"] = "evaluate"
    github = FakeGitHub(live)
    rulesets.apply(github, "konard/x", yes=True)
    assert [call for call in github.calls if call[0] != "GET"] == [
        ("PUT", f"/rulesets/{live[1]['id']}")
    ]


def test_ruleset_file_name_must_match_its_name(tmp_path: Path) -> None:
    (tmp_path / "main-x.json").write_text('{"name": "other"}')
    with pytest.raises(rulesets.RulesetError, match="name must be 'main-x'"):
        rulesets.load_desired(tmp_path)
