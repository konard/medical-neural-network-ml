"""Tests for scripts/check_distribution.py (issue #8).

Each test builds a small wheel and sdist by hand for a fake project, so the
checks run without a build backend or network; the Build Package job runs
the script on the real distributions.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_distribution.py"
spec = importlib.util.spec_from_file_location("check_distribution", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules[spec.name] = module  # type: ignore[union-attr]
spec.loader.exec_module(module)  # type: ignore[union-attr]

PYPROJECT = """\
[project]
name = "demo-dist"
version = "1.2.3"
license = "Unlicense"
license-files = ["LICENSE", "NOTICES.md"]
requires-python = ">=3.9"

[project.scripts]
demo = "demo.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/demo"]
"""
METADATA = """\
Metadata-Version: 2.4
Name: demo-dist
Version: 1.2.3
License-Expression: Unlicense
License-File: LICENSE
License-File: NOTICES.md
Requires-Python: >=3.9
"""
DIST_INFO = "demo_dist-1.2.3.dist-info"
SOURCES = {"demo/__init__.py": b"", "demo/cli.py": b"def main(): pass\n"}


def make_project(root: Path) -> None:
    (root / "src" / "demo").mkdir(parents=True)
    (root / "pyproject.toml").write_text(PYPROJECT)
    (root / "LICENSE").write_text("public domain\n")
    (root / "NOTICES.md").write_text("third parties\n")
    for name, content in SOURCES.items():
        (root / "src" / name).write_bytes(content)
    cache = root / "src" / "demo" / "__pycache__"
    cache.mkdir()
    (cache / "cli.cpython-313.pyc").write_bytes(b"ignored")


def wheel_files(root: Path, metadata: str = METADATA) -> dict[str, bytes]:
    files = dict(SOURCES)
    files[f"{DIST_INFO}/METADATA"] = metadata.encode()
    files[f"{DIST_INFO}/entry_points.txt"] = (
        b"[console_scripts]\ndemo = demo.cli:main\n"
    )
    for name in ("LICENSE", "NOTICES.md"):
        files[f"{DIST_INFO}/licenses/{name}"] = (root / name).read_bytes()
    return files


def write_wheel(dist: Path, files: dict[str, bytes], record: bool = True) -> Path:
    path = dist / "demo_dist-1.2.3-py3-none-any.whl"
    record_name = f"{DIST_INFO}/RECORD"
    rows = [
        f"{name},{module._record_hash(content)},{len(content)}"
        for name, content in files.items()
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        if record:
            archive.writestr(record_name, "\n".join([*rows, f"{record_name},,"]))
    return path


def sdist_files(root: Path, metadata: str = METADATA) -> dict[str, bytes]:
    files = {f"src/{name}": content for name, content in SOURCES.items()}
    for name in module.SDIST_FILES:
        files[name] = b"x"
    files["PKG-INFO"] = metadata.encode()
    for name in ("LICENSE", "NOTICES.md", "pyproject.toml"):
        files[name] = (root / name).read_bytes()
    return files


def write_sdist(dist: Path, files: dict[str, bytes]) -> Path:
    path = dist / "demo_dist-1.2.3.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"demo_dist-1.2.3/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "project"
    root.mkdir()
    make_project(root)
    dist = tmp_path / "dist"
    dist.mkdir()
    return root, dist, module.load_project(root)


def problems(root: Path, dist: Path, project_data: dict) -> list[str]:
    wheel, sdist = module.find_distributions(dist, project_data)
    return module.check_wheel(wheel, project_data, root) + module.check_sdist(
        sdist, project_data, root
    )


def test_matching_distributions_pass(project, capsys) -> None:
    root, dist, data = project
    write_wheel(dist, wheel_files(root))
    write_sdist(dist, sdist_files(root))
    assert problems(root, dist, data) == []
    assert module.main([str(dist), "--root", str(root)]) == 0
    assert "License-Expression Unlicense" in capsys.readouterr().out


def test_the_real_project_declares_what_the_check_expects() -> None:
    data = module.load_project(ROOT)
    assert data["license"] == "Unlicense"
    assert data["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    assert data["scripts"] == {"medbench": "medbench.cli:main"}
    assert "src/medbench" in data["wheel-packages"]
    files = module.source_files(ROOT, data["wheel-packages"])
    assert "medbench/data/real_samples.json" in files
    assert not any("__pycache__" in name for name in files)


def test_a_stale_or_extra_distribution_is_refused(project) -> None:
    root, dist, data = project
    write_wheel(dist, wheel_files(root))
    write_sdist(dist, sdist_files(root))
    (dist / "demo_dist-1.2.2-py3-none-any.whl").write_bytes(b"")
    with pytest.raises(module.DistributionError, match="must hold exactly"):
        module.find_distributions(dist, data)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (("License-Expression: Unlicense", "License-Expression: MIT"), "License-Exp"),
        (("License-File: NOTICES.md\n", ""), "License-File is ['LICENSE']"),
        (("Metadata-Version: 2.4", "Metadata-Version: 2.1"), "older than 2.4"),
        (("Version: 1.2.3", "Version: 1.2.4"), "Version is '1.2.4'"),
        (("Requires-Python: >=3.9\n", ""), "Requires-Python is None"),
        (
            ("Requires-Python: >=3.9\n", "Requires-Python: >=3.9\nLicense: MIT\n"),
            "legacy License field",
        ),
        (
            (
                "Requires-Python: >=3.9\n",
                "Requires-Python: >=3.9\nClassifier: License :: OSI Approved\n",
            ),
            "license classifier",
        ),
    ],
)
def test_wrong_license_metadata_is_reported(project, change, expected) -> None:
    root, dist, data = project
    metadata = METADATA.replace(*change)
    write_wheel(dist, wheel_files(root, metadata))
    write_sdist(dist, sdist_files(root, metadata))
    found = problems(root, dist, data)
    assert any(expected in problem for problem in found), found
    assert any(problem.startswith("demo_dist-1.2.3.tar.gz") for problem in found)


def test_missing_or_changed_license_files_are_reported(project) -> None:
    root, dist, data = project
    files = wheel_files(root)
    del files[f"{DIST_INFO}/licenses/NOTICES.md"]
    files[f"{DIST_INFO}/licenses/LICENSE"] = b"MIT\n"
    write_wheel(dist, files)
    write_sdist(dist, {**sdist_files(root), "NOTICES.md": b"edited\n"})
    found = problems(root, dist, data)
    assert (
        f"demo_dist-1.2.3-py3-none-any.whl: {DIST_INFO}/licenses/NOTICES.md missing"
        in found
    )
    assert (
        "demo_dist-1.2.3-py3-none-any.whl: LICENSE differs from the repository copy"
        in found
    )
    assert (
        "demo_dist-1.2.3.tar.gz: NOTICES.md differs from the repository copy" in found
    )


def test_a_missing_console_script_is_reported(project) -> None:
    root, dist, data = project
    files = wheel_files(root)
    files[f"{DIST_INFO}/entry_points.txt"] = b""
    write_wheel(dist, files)
    write_sdist(dist, sdist_files(root))
    assert any(
        "console scripts {}" in problem for problem in problems(root, dist, data)
    )


def test_manifest_differences_are_reported(project) -> None:
    root, dist, data = project
    files = wheel_files(root)
    del files["demo/cli.py"]
    files["demo/__init__.py"] = b"changed"
    files["tests/test_demo.py"] = b""
    write_wheel(dist, files)
    sdist = sdist_files(root)
    del sdist["src/demo/cli.py"]
    sdist["ci-logs/run.log"] = b"log"
    write_sdist(dist, sdist)
    found = problems(root, dist, data)
    wheel_name = "demo_dist-1.2.3-py3-none-any.whl"
    assert f"{wheel_name}: missing demo/cli.py" in found
    assert f"{wheel_name}: unexpected tests/test_demo.py" in found
    assert f"{wheel_name}: demo/__init__.py differs from the source tree" in found
    assert "demo_dist-1.2.3.tar.gz: missing src/demo/cli.py" in found
    assert "demo_dist-1.2.3.tar.gz: unexpected ci-logs/run.log" in found


def test_a_wheel_without_a_matching_record_is_reported(project) -> None:
    root, dist, data = project
    write_wheel(dist, wheel_files(root), record=False)
    write_sdist(dist, sdist_files(root))
    assert any(
        "RECORD does not list" in problem for problem in problems(root, dist, data)
    )


def test_bundled_media_and_unsafe_paths_are_reported() -> None:
    found = module.check_member_names(
        ["demo/data/xray.PNG", "/etc/passwd", "demo/../../x.py", "demo/a.json"], "w"
    )
    assert found == [
        "w: bundles media or a dataset file demo/data/xray.PNG",
        "w: unsafe path /etc/passwd",
        "w: unsafe path demo/../../x.py",
    ]


def test_links_in_the_sdist_are_reported(project) -> None:
    root, dist, data = project
    write_wheel(dist, wheel_files(root))
    path = write_sdist(dist, sdist_files(root))
    with tarfile.open(path, "w:gz") as archive:
        for name, content in sdist_files(root).items():
            info = tarfile.TarInfo(f"demo_dist-1.2.3/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        link = tarfile.TarInfo("demo_dist-1.2.3/src/demo/evil.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    found = problems(root, dist, data)
    assert (
        "demo_dist-1.2.3.tar.gz: demo_dist-1.2.3/src/demo/evil.py is a link or device"
        in found
    )


def test_main_reports_problems_as_annotations(project, capsys) -> None:
    root, dist, _data = project
    write_wheel(dist, wheel_files(root, METADATA.replace("Unlicense", "MIT")))
    write_sdist(dist, sdist_files(root))
    assert module.main([str(dist), "--root", str(root)]) == 1
    assert "::error title=Distribution check::" in capsys.readouterr().out


def test_the_install_runs_on_a_relative_dist_directory(project, monkeypatch) -> None:
    """The Build Package job passes ``dist``; pip runs in a temporary folder."""
    root, dist, data = project
    write_wheel(dist, wheel_files(root))
    write_sdist(dist, sdist_files(root))
    monkeypatch.chdir(dist.parent)
    monkeypatch.setattr(module.venv, "create", lambda *_args, **_kwargs: None)
    commands = []

    def stop(command: list[str], cwd: Path) -> str:
        commands.append(command)
        message = "stop after the first command"
        raise module.DistributionError(message)

    monkeypatch.setattr(module, "_run", stop)
    wheel, sdist = module.find_distributions(Path("dist"), data)
    with pytest.raises(module.DistributionError):
        module.install_smoke_test(wheel, sdist, data)
    installed = Path(commands[0][-1])
    assert installed.is_absolute()
    assert installed.is_file()


def test_the_build_job_checks_and_installs_the_distributions() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    build = workflow.split("\n  build:\n", 1)[1].split("\n  changelog:\n", 1)[0]
    assert "python scripts/check_distribution.py dist --install" in build
    assert build.index("check_distribution.py") < build.index("Upload artifacts")
    manual = workflow.split("\n  manual-release:\n", 1)[1]
    assert manual.index("check_distribution.py dist --install") < manual.index(
        "Publish to PyPI"
    )


def test_the_publishing_guide_matches_the_release_workflow() -> None:
    """The publisher a maintainer registers must match what PyPI will see.

    PyPI compares the OIDC token's repository, workflow file and environment
    with the publisher; a GitHub environment on a publishing job, or a
    renamed workflow or project, would make the documented settings wrong.
    """
    guide = (ROOT / "docs" / "publishing.md").read_text()
    data = module.load_project(ROOT)
    owner, repository = data["urls"]["Repository"].split("/")[-2:]
    for field, value in [
        ("PyPI Project Name", data["name"]),
        ("Owner", owner),
        ("Repository name", repository),
        ("Workflow name", "release.yml"),
    ]:
        assert f"| {field} | `{value}` |" in guide
    assert "| Environment name | *(blank)* |" in guide
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "\n    environment:" not in workflow
    publish_steps = workflow.split("pypa/gh-action-pypi-publish@")[1:]
    assert len(publish_steps) == 2
    for step in publish_steps:
        assert "password" not in step.split("\n      - name:", 1)[0]
