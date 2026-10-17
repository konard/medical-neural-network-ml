"""Tests for scripts/audit_dependencies.py (issue #68)."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "audit_dependencies.py"
)
spec = importlib.util.spec_from_file_location("audit_dependencies", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)  # type: ignore[union-attr]


def test_run_streams_output_when_the_command_fails(capfd) -> None:
    """A failing tool must not swallow its own report.

    pip-audit prints the advisory table on stdout and then exits non-zero;
    capturing stdout under check=True discarded the table exactly when it
    mattered. The default must stream, so the table survives.
    """
    try:
        module.run(
            [sys.executable, "-c", "print('advisory table'); raise SystemExit(1)"],
            cwd=Path.cwd(),
        )
    except subprocess.CalledProcessError:
        pass
    else:
        msg = "run() should propagate a non-zero exit"
        raise AssertionError(msg)

    assert "advisory table" in capfd.readouterr().out


def test_run_captured_mode_returns_output() -> None:
    output = module.run(
        [sys.executable, "-c", "print('purelib path')"],
        cwd=Path.cwd(),
        capture=True,
    )
    assert output == "purelib path"


def test_target_venv_is_created_without_pip(monkeypatch) -> None:
    """Auditing a bundled pip reports tool advisories as if they were ours."""
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, capture=False):
        commands.append([str(part) for part in command])
        return "purelib"

    monkeypatch.setattr(module, "run", fake_run)

    project_root = Path(__file__).resolve().parent.parent
    module.audit_dependencies(project_root)

    target_creation = next(
        command for command in commands if command[-1].endswith("target")
    )
    assert "--without-pip" in target_creation

    # Installs into the pip-less target go through the host interpreter's pip
    # with --python, so pip itself never lands in the audit scope. The audit
    # venv's own pip-audit install is deliberately excluded from that claim.
    target_installs = [
        command
        for command in commands
        if command[0] == sys.executable
        and "install" in command
        and "venv" not in command
    ]
    assert len(target_installs) >= 2, commands
    for command in target_installs:
        assert command[1:3] == ["-m", "pip"], command
        assert "--python" in command, command
        assert "target" in command[4], command


def test_missing_dependency_surface_fails_the_audit(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    with pytest.raises(FileNotFoundError, match=re.escape("docs/requirements.txt")):
        module.audit_dependencies(tmp_path)


def test_pinned_runtime_is_audited_without_installing_it(monkeypatch) -> None:
    """The UniMedVL pins (issue #6) are audited as a file, exactly as pinned.

    Before, the file matched no surface and torch 2.5.1 (GHSA-53q9-r3pm-6pq6)
    was only caught by Dependency Review.
    """
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, capture=False):
        commands.append([str(part) for part in command])
        return "purelib"

    monkeypatch.setattr(module, "run", fake_run)
    module.audit_dependencies(Path(__file__).resolve().parent.parent)

    pinned = [
        command
        for command in commands
        if "--requirement" in command and "scripts/unimedvl-requirements.txt" in command
    ]
    assert len(pinned) == 1, commands
    command = pinned[0]
    assert command[1:3] == ["-m", "pip_audit"]
    assert "audit" in command[0]
    assert {"--no-deps", "--disable-pip"} <= set(command)
    ignored = [
        command[index + 1]
        for index, part in enumerate(command)
        if part == "--ignore-vuln"
    ]
    assert ignored == sorted(module.ACCEPTED_ADVISORIES)


def test_every_accepted_advisory_has_a_reason() -> None:
    """An accepted advisory names its package and why it does not apply."""
    assert module.ACCEPTED_ADVISORIES
    for advisory, reason in module.ACCEPTED_ADVISORIES.items():
        assert advisory.startswith(("PYSEC-", "GHSA-")), advisory
        assert reason.split(" ", 1)[0] in {"torch", "transformers", "accelerate"}
        assert ";" in reason or ":" in reason, reason
    # torch.load RCE is fixed by the pin, never accepted.
    assert "GHSA-53q9-r3pm-6pq6" not in module.ACCEPTED_ADVISORIES


def test_dependency_review_allows_only_accepted_advisories() -> None:
    """Dependency Review skips exactly the GHSA aliases pip-audit accepts."""
    workflow = (SCRIPT_PATH.parents[1] / ".github/workflows/security.yml").read_text(
        encoding="utf-8"
    )
    allowed = re.findall(r"^\s+allow-ghsas: (.+)$", workflow, re.MULTILINE)
    assert len(allowed) == 1, allowed
    ghsas = [item.strip() for item in allowed[0].split(",")]
    assert ghsas == sorted(module.REVIEW_ALLOWED_GHSAS)
    for alias in module.REVIEW_ALLOWED_GHSAS.values():
        assert module.ACCEPTED_ADVISORIES[alias].startswith("transformers")


def test_missing_pinned_surface_fails_the_audit(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "requirements.txt").write_text("")
    with pytest.raises(FileNotFoundError, match=re.escape("unimedvl-requirements.txt")):
        module.audit_dependencies(tmp_path)
