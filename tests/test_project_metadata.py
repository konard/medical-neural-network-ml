"""Regression tests for project package metadata.

Every tool is pinned in more than one place: the ``dev`` extra installs ruff and
mypy for CI, and ``.pre-commit-config.yaml`` installs them again for the commit
hook. The Python floor appears in ``requires-python``, the classifiers, ruff's
``target-version`` and mypy's ``python_version``. Two pins that drift apart are
two different linters (issue #19), so these tests keep each pair in step.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def extra_requirement(extra: str, name: str) -> Requirement:
    """Return one requirement from an optional-dependencies extra."""
    requirements = PYPROJECT["project"]["optional-dependencies"][extra]
    parsed = (Requirement(line) for line in requirements)
    return next(requirement for requirement in parsed if requirement.name == name)


def floor(specifier: SpecifierSet) -> Version:
    """Return the version named by the ``>=`` clause of a specifier."""
    (clause,) = (spec for spec in specifier if spec.operator == ">=")
    return Version(clause.version)


def python_floor() -> Version:
    return floor(SpecifierSet(PYPROJECT["project"]["requires-python"]))


def pre_commit_rev(repo: str) -> Version:
    """Return the ``rev`` pinned for one repository in .pre-commit-config.yaml."""
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    match = re.search(rf"repo: {re.escape(repo)}\n\s+rev: v?(\S+)", config)
    assert match, repo
    return Version(match.group(1))


def test_tool_targets_match_the_python_floor() -> None:
    """ruff and mypy check against the oldest Python the package installs on."""
    minimum = python_floor()

    assert PYPROJECT["tool"]["ruff"]["target-version"] == (
        f"py{minimum.major}{minimum.minor}"
    )
    assert PYPROJECT["tool"]["mypy"]["python_version"] == (
        f"{minimum.major}.{minimum.minor}"
    )


def test_classifiers_start_at_the_python_floor() -> None:
    """Classifiers advertise exactly the minor versions from the floor upwards."""
    minimum = python_floor()
    prefix = "Programming Language :: Python :: 3."
    minors = sorted(
        int(classifier.removeprefix(prefix))
        for classifier in PYPROJECT["project"]["classifiers"]
        if classifier.startswith(prefix)
    )

    assert minors[0] == minimum.minor
    assert minors == list(range(minors[0], minors[-1] + 1))


def test_pre_commit_hooks_match_the_dev_extra() -> None:
    """The commit hook runs the same ruff and mypy that CI installs."""
    assert pre_commit_rev("https://github.com/astral-sh/ruff-pre-commit") == floor(
        extra_requirement("dev", "ruff").specifier
    )
    assert pre_commit_rev("https://github.com/pre-commit/mirrors-mypy") == floor(
        extra_requirement("dev", "mypy").specifier
    )
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    (hook_pillow,) = re.findall(r"additional_dependencies: \[(pillow==\S+)\]", config)
    assert Requirement(hook_pillow).specifier == SpecifierSet(
        f"=={floor(extra_requirement('dev', 'pillow').specifier)}"
    )


def test_extras_pin_shared_packages_identically() -> None:
    """A package listed in several extras has one version range everywhere."""
    seen: dict[str, SpecifierSet] = {}
    for requirements in PYPROJECT["project"]["optional-dependencies"].values():
        for requirement in map(Requirement, requirements):
            expected = seen.setdefault(requirement.name, requirement.specifier)
            assert requirement.specifier == expected, requirement


def test_docs_requirements_match_the_docs_extra() -> None:
    """docs.yml installs docs/requirements.txt; the extra must say the same."""
    lines = (ROOT / "docs" / "requirements.txt").read_text(encoding="utf-8")
    from_file = sorted(str(Requirement(line)) for line in lines.split())
    from_extra = sorted(PYPROJECT["project"]["optional-dependencies"]["docs"])

    assert from_file == from_extra


def test_every_advertised_python_runs_in_ci() -> None:
    """Each classifier minor is a Python that some CI job runs the suite on.

    compatibility.yml covers the older minors and the Mac, release.yml the
    newest (issue #19), so a classifier is never a claim nothing checks.
    """
    workflows = ROOT / ".github" / "workflows"
    tested = set(
        re.findall(
            r"^ +- python: '(3\.\d+)'$",
            (workflows / "compatibility.yml").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    tested |= set(
        re.findall(
            r"^ +python-version: '(3\.\d+)'$",
            (workflows / "release.yml").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    prefix = "Programming Language :: Python :: "
    advertised = {
        classifier.removeprefix(prefix)
        for classifier in PYPROJECT["project"]["classifiers"]
        if re.fullmatch(rf"{re.escape(prefix)}3\.\d+", classifier)
    }

    assert advertised == tested
