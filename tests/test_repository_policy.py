"""Repository policy for issue #10: pinned actions, update PRs, branch rules.

The files are read as text, like the other workflow tests, so no YAML
library is needed.
"""

from __future__ import annotations

import re

from tests.workflow_helpers import ROOT, WORKFLOWS

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
    assert set(entries) == {"github-actions", "pip"}
    for ecosystem, block in entries.items():
        assert re.search(r"^      default-days: 7$", block, re.M), ecosystem
        assert "interval: weekly" in block, ecosystem
    assert "directory: /\n" in entries["github-actions"]
    pip = entries["pip"]
    assert "      - /\n" in pip
    assert "      - /docs\n" in pip


def test_dependabot_leaves_the_coupled_unimedvl_pins_alone() -> None:
    """torch must move with the hash-pinned flash-attn wheel, not on its own.

    The file is still audited (scripts/audit_dependencies.py), and a change
    under scripts/ would need a changelog fragment Dependabot cannot write.
    """
    pip = dependabot_entries()["pip"]
    assert "/scripts" not in pip
    text = DEPENDABOT.read_text(encoding="utf-8")
    assert "scripts/unimedvl-requirements.txt" in text
    audit = (ROOT / "scripts" / "audit_dependencies.py").read_text(encoding="utf-8")
    assert '"scripts/unimedvl-requirements.txt"' in audit


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
