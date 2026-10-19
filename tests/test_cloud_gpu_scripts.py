"""Tests for the GPU host scripts of issue #23.

scripts/gpu-model-env.sh must print the settings the `gpu` job of
model-tests.yml uses, scripts/gpu-host-run.sh must run the real model script
(with UniMedVL's ``echo`` engine, so no GPU and no download) and write its
summary, and every pinned download must be the same in the shell scripts,
the Ansible playbooks, the Kubernetes values and the case-study evidence.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_medbench_model_run import linux_x86_64
from tests.workflow_helpers import read_workflow, workflow_job_block

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL_ENV = SCRIPTS / "gpu-model-env.sh"
HOST_RUN = SCRIPTS / "gpu-host-run.sh"
BOOTSTRAP = SCRIPTS / "gpu-host-bootstrap.sh"
RUNNER_START = SCRIPTS / "gpu-runner-start.sh"
ANSIBLE = ROOT / "infra" / "ansible"
EVIDENCE = ROOT / "docs" / "case-studies" / "issue-23" / "data"
SHELL_SCRIPTS = [MODEL_ENV, HOST_RUN, BOOTSTRAP, RUNNER_START]


def model_env(model: str) -> dict[str, str]:
    result = subprocess.run(
        ["bash", str(MODEL_ENV), model],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def gpu_job() -> str:
    return workflow_job_block(read_workflow("model-tests.yml"), "gpu")


def job_env(block: str) -> dict[str, str]:
    """The job-level ``env:`` mapping (six-space keys) of a job block."""
    env_text = block.split("\n    env:\n", 1)[1].split("\n    steps:\n", 1)[0]
    pairs = re.findall(r"^      ([A-Z_]+): (.+)$", env_text, re.MULTILINE)
    return {key: value.strip() for key, value in pairs if "${{" not in value}


def select_step_lines(block: str, model: str) -> dict[str, str]:
    """``echo "KEY=VALUE"`` lines of one case of the Select model step."""
    step = block.split("- name: Select model", 1)[1].split("\n      - name:", 1)[0]
    case = step.split(f"            {model})", 1)[1].split(";;", 1)[0]
    pairs = re.findall(r'echo "([A-Z_]+)=([^"$]+)"', case)
    return dict(pairs)


def test_profiles_match_the_gpu_job() -> None:
    """The cloud runs download exactly what the self-hosted GPU job does."""
    block = gpu_job()
    llama = job_env(block)
    assert set(llama) >= {"LLAMA_ARCHIVE", "LLAMA_SHA256", "CUDART_ARCHIVE"}
    for model in ("medgemma-27b-it", "medgemma-4b-it"):
        profile = model_env(model)
        for key, value in llama.items():
            assert profile.get(key) == value, (model, key)
    big = model_env("medgemma-27b-it")
    expected = select_step_lines(block, "medgemma-27b-it")
    assert set(expected) >= {"MODEL_REPO", "MODEL_SHA256", "MMPROJ_SHA256"}
    for key, value in expected.items():
        assert big[key] == value, key
    assert select_step_lines(block, "unimedvl") == {"SERVER": "unimedvl"}
    assert model_env("unimedvl") == {"SERVER": "unimedvl"}


def test_unknown_model_is_rejected() -> None:
    result = subprocess.run(
        ["bash", str(MODEL_ENV), "llama-70b"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown model" in result.stderr


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def host_run(
    tmp_path: Path, path_prefix: Path | None = None, **env: str
) -> subprocess.CompletedProcess[str]:
    """gpu-host-run.sh with this interpreter and UniMedVL's echo engine."""
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    if path_prefix is not None:
        path = f"{path_prefix}{os.pathsep}{path}"
    return subprocess.run(
        ["bash", str(HOST_RUN)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GPU_MODEL": "unimedvl",
            "MEDBENCH_PYTHON": sys.executable,
            "STEPS": "run",
            "REQUIRE_GPU": "0",
            "UNIMEDVL_ENGINE": "echo",
            "UNIMEDVL_ECHO_PYTHON": sys.executable,
            "RUN_PYTEST": "0",
            "PORT": str(free_port()),
            "CACHE_DIR": str(tmp_path / "cache"),
            "OUT_DIR": str(tmp_path / "out"),
            "TASKS": "diagnosis",
            "COUNT": "2",
            "MEDBENCH_COMMIT": "abc123",
            "PATH": path,
            **env,
        },
        cwd=tmp_path,  # the script must find the repository by itself
        timeout=180,
        check=False,
    )


@linux_x86_64
def test_host_run_runs_the_model_script_and_writes_a_summary(
    tmp_path: Path,
) -> None:
    """The entry point of every cloud flow, end to end without a GPU."""
    result = host_run(tmp_path)
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    summary = json.loads((out / "gpu-host-run.json").read_text())
    assert summary["model"] == "unimedvl"
    assert summary["exit_code"] == 0
    assert summary["commit"] == "abc123"
    assert summary["count"] == 2
    assert summary["tasks"] == ["diagnosis"]
    report = json.loads((out / "diagnosis.json").read_text())
    assert report["items_run"] == 2
    resources = json.loads((out / "resources-run.json").read_text())
    assert resources["label"] == "Run medbench"
    assert not (out / "resources-download.json").exists()


@linux_x86_64
def test_host_run_reports_a_failed_run(tmp_path: Path) -> None:
    """The summary and the exit code carry the model run's failure."""
    result = host_run(tmp_path, TIME_BUDGET="0")
    assert result.returncode == 1
    summary = json.loads((tmp_path / "out" / "gpu-host-run.json").read_text())
    assert summary["exit_code"] == 1


def fake_nvidia_smi(directory: Path, driver: str) -> Path:
    directory.mkdir()
    script = directory / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f"  *driver_version*) echo {driver} ;;\n"
        "  *) echo 'NVIDIA L4, fake' ;;\n"
        "esac\n"
    )
    script.chmod(0o755)
    return directory


@pytest.mark.parametrize(("driver", "status"), [("535.230.02", 1), ("580.95.05", 0)])
def test_host_run_checks_the_driver_branch(
    tmp_path: Path, driver: str, status: int
) -> None:
    """The CUDA 13.4 build of llama.cpp needs driver R580 or newer."""
    bin_dir = fake_nvidia_smi(tmp_path / "bin", driver)
    result = host_run(tmp_path, bin_dir, REQUIRE_GPU="1", STEPS="none")
    assert result.returncode == status, result.stderr
    if status:
        assert "older than 580" in result.stderr
    else:
        assert f"NVIDIA driver {driver}" in result.stderr


def test_host_run_without_nvidia_smi_fails_with_a_hint(tmp_path: Path) -> None:
    empty = tmp_path / "bin"
    empty.mkdir()
    for tool in ("bash", "dirname", "mkdir", "date"):
        found = shutil.which(tool)
        assert found is not None
        (empty / tool).symlink_to(found)
    result = subprocess.run(
        ["bash", str(HOST_RUN)],
        capture_output=True,
        text=True,
        env={"PATH": str(empty), "OUT_DIR": str(tmp_path / "out"), "HOME": "/"},
        cwd=tmp_path,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert "nvidia-smi not found" in result.stderr
    assert "REQUIRE_GPU=0" in result.stderr


def shell_value(path: Path, name: str) -> str:
    match = re.search(rf'^{name}="([^"$]+)"', path.read_text(), re.MULTILINE)
    assert match, f"{name} not set in {path}"
    return match.group(1)


def yaml_value(path: Path, name: str) -> str:
    match = re.search(rf"^\s+{name}: \"?([^\"\n]+)\"?$", path.read_text(), re.MULTILINE)
    assert match, f"{name} not set in {path}"
    return match.group(1).strip()


def test_driver_and_toolkit_pins_match_the_playbook() -> None:
    """The bare-host bootstrap and Ansible install the same NVIDIA stack."""
    playbook = ANSIBLE / "gpu-host.yml"
    assert shell_value(BOOTSTRAP, "DRIVER_BRANCH") == yaml_value(
        playbook, "nvidia_driver_branch"
    )
    assert shell_value(BOOTSTRAP, "TOOLKIT_VERSION") == yaml_value(
        playbook, "nvidia_container_toolkit_version"
    )
    fingerprint = shell_value(BOOTSTRAP, "TOOLKIT_KEY_FINGERPRINT")
    assert fingerprint == yaml_value(
        playbook, "nvidia_container_toolkit_key_fingerprint"
    )
    evidence = json.loads((EVIDENCE / "actions-runner-v2.338.0.json").read_text())
    toolkit = evidence["nvidia_container_toolkit"]
    assert fingerprint == toolkit["signing_key_fingerprint"]
    assert shell_value(BOOTSTRAP, "TOOLKIT_VERSION") in toolkit["versions_seen"]
    assert shell_value(BOOTSTRAP, "DRIVER_BRANCH") == shell_value(
        HOST_RUN, "MIN_DRIVER"
    ).split(":-", 1)[-1].rstrip("}")


def test_k3s_pins_match_the_playbook_and_the_evidence() -> None:
    playbook = ANSIBLE / "k3s-gpu.yml"
    evidence = json.loads((EVIDENCE / "actions-runner-v2.338.0.json").read_text())
    k3s = evidence["k3s"]
    assert shell_value(BOOTSTRAP, "K3S_VERSION") == k3s["version"]
    assert yaml_value(playbook, "k3s_version") == k3s["version"]
    assert shell_value(BOOTSTRAP, "K3S_SHA256_AMD64") == k3s["k3s_amd64_sha256"]
    assert yaml_value(playbook, "k3s_sha256") == k3s["k3s_amd64_sha256"]
    assert shell_value(BOOTSTRAP, "K3S_INSTALL_SHA256") == k3s["install_sh_sha256_at_tag"]
    assert yaml_value(playbook, "k3s_install_sha256") == k3s["install_sh_sha256_at_tag"]


def test_runner_pins_match_everywhere() -> None:
    """Every self-hosted runner flow runs the same pinned GitHub runner."""
    evidence = json.loads((EVIDENCE / "actions-runner-v2.338.0.json").read_text())
    version = shell_value(RUNNER_START, "RUNNER_VERSION")
    sha256 = shell_value(RUNNER_START, "RUNNER_SHA256")
    asset = f"actions-runner-linux-x64-{version}.tar.gz"
    assert evidence["assets"][asset] == f"sha256:{sha256}"
    playbook = ANSIBLE / "github-runner.yml"
    assert yaml_value(playbook, "runner_version") == version
    assert yaml_value(playbook, "runner_sha256") == sha256
    values = (ROOT / "infra" / "kubernetes" / "arc-gpu-runner-values.yaml").read_text()
    image = evidence["container_image"]
    assert f"{image['reference']}@{image['index_digest']}" in values


def test_runner_start_keeps_the_jit_config_off_command_lines() -> None:
    """The secret arrives on stdin and reaches the runner via its environment."""
    text = RUNNER_START.read_text()
    assert 'config="$(cat)"' in text
    assert "ACTIONS_RUNNER_INPUT_JITCONFIG=" in text
    assert "--jitconfig" not in text.split("set -euo pipefail", 1)[1]
    assert "umask 077" in text


def test_runner_start_needs_root() -> None:
    if os.geteuid() == 0:
        pytest.skip("runs as root")
    result = subprocess.run(
        ["bash", str(RUNNER_START)],
        input="x",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "run as root" in result.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="needs shellcheck")
def test_shellcheck() -> None:
    result = subprocess.run(
        ["shellcheck", *map(str, SHELL_SCRIPTS)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout
