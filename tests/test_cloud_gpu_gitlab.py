"""Tests for the GitLab CI template of the cloud GPU tests (issue #23).

ci/gitlab/cloud-gpu.gitlab-ci.yml runs the same orchestrator as the GitHub
workflow, so the tests run its shell lines against the orchestrator's real
argument parser, check that paid jobs never run for a merge request, that
the machine is deleted in after_script even when script times out, and that
the working directory (SSH key, Terraform state) is never an artifact.
ci/gitlab/cloud-login.sh runs with stub cloud CLIs.
"""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.cloud_gpu import cli

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "ci" / "gitlab" / "cloud-gpu.gitlab-ci.yml"
LOGIN = ROOT / "ci" / "gitlab" / "cloud-login.sh"
TEMPLATE: dict[str, Any] = yaml.safe_load(TEMPLATE_PATH.read_text())
JOBS = {name: job for name, job in TEMPLATE.items() if name.startswith("cloud-gpu:")}


def job(name: str) -> dict[str, Any]:
    """A job with its `extends` base merged in, as GitLab does."""
    merged: dict[str, Any] = {}
    own = JOBS[name]
    base = TEMPLATE[own["extends"]] if "extends" in own else {}
    for key in {*base, *own}:
        if isinstance(base.get(key), dict) and isinstance(own.get(key), dict):
            merged[key] = {**base[key], **own[key]}
        else:
            merged[key] = own.get(key, base.get(key))
    return merged


def flatten(lines: list[Any]) -> list[str]:
    """GitLab's nested script arrays (from YAML anchors) as one list."""
    flat: list[str] = []
    for line in lines:
        flat.extend(flatten(line) if isinstance(line, list) else [line])
    return flat


def variables(name: str) -> dict[str, str]:
    values = {**TEMPLATE["variables"], **job(name).get("variables", {})}
    return {key: str(value) for key, value in values.items()}


def run_bash(
    script: str, cwd: Path, env: dict[str, str], path: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        cwd=cwd,
        env={"PATH": path + os.environ["PATH"], **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def duration(text: str) -> int:
    """Seconds of a GitLab duration such as ``6h``, ``45m`` or ``1h30m``."""
    units = {"h": 3600, "m": 60, "s": 1}
    parts = re.findall(r"(\d+)([hms])", text)
    assert parts, text
    assert "".join(n + u for n, u in parts) == text.replace(" ", ""), text
    return sum(int(number) * units[unit] for number, unit in parts)


def test_every_job_uses_a_known_stage_and_the_python_image() -> None:
    assert set(JOBS) == {
        "cloud-gpu:check-infra",
        "cloud-gpu:plan",
        "cloud-gpu:test",
        "cloud-gpu:reap",
    }
    for name in JOBS:
        assert job(name)["stage"] in {"check", "cloud", "reap"}
        assert job(name)["image"] == "$CLOUD_GPU_IMAGE"
    assert (
        TEMPLATE["variables"]["CLOUD_GPU_IMAGE"]
        == "python:3.14-slim-trixie@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8"
    )


@pytest.mark.parametrize("name", ["cloud-gpu:test", "cloud-gpu:reap"])
def test_paid_jobs_run_only_on_protected_refs_and_never_for_a_merge_request(
    name: str,
) -> None:
    rules = job(name)["rules"]
    assert len(rules) == 1
    condition = rules[0]["if"]
    assert '$CI_COMMIT_REF_PROTECTED == "true"' in condition
    assert "merge_request_event" not in condition
    flag = "CLOUD_GPU_TEST" if name == "cloud-gpu:test" else "CLOUD_GPU_REAP"
    assert f'${flag} == "true"' in condition
    assert TEMPLATE["variables"][flag] == "false"


def test_after_script_has_time_to_delete_the_machine_before_the_job_timeout() -> None:
    test = job("cloud-gpu:test")
    values = variables("cloud-gpu:test")
    script = duration(values["RUNNER_SCRIPT_TIMEOUT"])
    after = duration(values["RUNNER_AFTER_SCRIPT_TIMEOUT"])
    # GitLab skips after_script when the job itself times out.
    assert script + after < duration(test["timeout"])
    # `down` of a GPU cluster takes up to 15 minutes; the default is 5.
    assert after >= 30 * 60
    after_script = flatten(test["after_script"])
    assert after_script[-1].startswith("python3 -m scripts.cloud_gpu down ")


def test_the_cloud_logins_use_id_tokens_for_every_cloud() -> None:
    for name in ("cloud-gpu:test", "cloud-gpu:reap"):
        tokens = job(name)["id_tokens"]
        assert set(tokens) == {"AWS_ID_TOKEN", "AZURE_ID_TOKEN", "GCP_ID_TOKEN"}
        assert tokens["GCP_ID_TOKEN"]["aud"] == (
            "https://iam.googleapis.com/$GCP_WORKLOAD_IDENTITY_PROVIDER"
        )
    test = job("cloud-gpu:test")
    login = 'source ci/gitlab/cloud-login.sh "$CLOUD_GPU_PROVIDER"'
    # after_script runs in a new shell, so it logs in again.
    assert login in flatten(test["script"])
    assert login in flatten(test["after_script"])


def test_the_working_directory_is_never_an_artifact() -> None:
    for name in JOBS:
        artifacts = job(name).get("artifacts", {})
        for path in artifacts.get("paths", []):
            assert ".cache" not in path
            assert "state.json" not in path
    assert job("cloud-gpu:test")["artifacts"]["paths"] == ["cloud-gpu-reports/"]
    assert variables("cloud-gpu:test")["CLOUD_GPU_WORKDIR"].endswith(
        "/.cache/cloud-gpu/run"
    )


def settings_args(provider: str, mode: str, **extra: str) -> list[str]:
    script = "\n".join(
        [*flatten(TEMPLATE[".cloud-gpu-settings"]), 'printf "%s\\n" "${args[@]}"']
    )
    env = {
        **variables("cloud-gpu:plan"),
        "CLOUD_GPU_PROVIDER": provider,
        "CLOUD_GPU_MODE": mode,
        "CLOUD_GPU_WORKDIR": "/builds/x/.cache/cloud-gpu/run",
        "CLOUD_GPU_REPORTS": "/builds/x/cloud-gpu-reports",
        "CI_PIPELINE_ID": "7",
        "CI_JOB_ID": "11",
        "CI_COMMIT_SHA": "0" * 40,
        "GCP_PROJECT": "medbench-ci",
        "AZURE_SUBSCRIPTION_ID": "00000000-0000-0000-0000-000000000000",
        **extra,
    }
    result = run_bash(script, ROOT, env)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


@pytest.mark.parametrize("provider", ["aws", "azure", "gcp", "hetzner"])
@pytest.mark.parametrize("mode", ["vm", "k8s"])
def test_the_settings_build_arguments_the_cli_accepts(provider: str, mode: str) -> None:
    args = settings_args(
        provider, mode, CLOUD_GPU_SPOT="true", CLOUD_GPU_REGION="eu-west-1"
    )
    parsed = cli.build_parser().parse_args(["up", *args, "--yes"])
    config = cli.config_from_args(parsed, needs_cidr=False)
    assert config.provider == provider
    assert config.mode == mode
    assert config.name == "medbench-gl-7-11"
    assert parsed.spot
    assert parsed.region == "eu-west-1"
    assert parsed.ttl_minutes == 360
    if provider == "gcp":
        assert parsed.gcp_project == "medbench-ci"


def test_the_runner_mode_is_not_offered_on_gitlab() -> None:
    comment = TEMPLATE_PATH.read_text().split("\nvariables:", 1)[0]
    assert "GitHub only" in comment
    assert TEMPLATE["variables"]["CLOUD_GPU_MODE"] in {"vm", "k8s"}


def test_the_delete_step_skips_a_run_that_created_nothing(tmp_path: Path) -> None:
    first = flatten(job("cloud-gpu:test")["after_script"])[0]
    result = run_bash(first, tmp_path, {"CLOUD_GPU_WORKDIR": str(tmp_path / "run")})
    assert result.returncode == 0
    assert result.stdout.strip() == "Nothing was created."


def test_the_checks_run_when_any_file_they_use_changes() -> None:
    patterns = [
        change
        for rule in job("cloud-gpu:check-infra")["rules"]
        for change in rule.get("changes", [])
    ]
    used = [
        *[p for p in (ROOT / "infra").rglob("*") if p.is_file()],
        *(ROOT / "scripts" / "cloud_gpu").glob("*.py"),
        *(ROOT / "scripts").glob("gpu-*.sh"),
        *(ROOT / "scripts").glob("cloud-gpu-*.sh"),
        *(ROOT / "ci" / "gitlab").iterdir(),
        ROOT / "scripts" / "check-infra.sh",
        ROOT / "examples" / "cloud_gpu_kind.sh",
    ]
    for path in used:
        relative = path.relative_to(ROOT).as_posix()
        globs = [p.replace("**/*", "*") for p in patterns]
        assert any(fnmatch.fnmatch(relative, p) for p in globs), relative


def test_every_orchestrator_call_uses_a_real_subcommand() -> None:
    text = TEMPLATE_PATH.read_text()
    subcommands = set(re.findall(r"python3 -m scripts\.cloud_gpu (\S+)", text))
    assert subcommands == {"plan", "up", "run", "down", "reap"}


@pytest.fixture
def stub_bin(tmp_path: Path) -> Path:
    """aws, az and gcloud that log their arguments and succeed."""
    directory = tmp_path / "bin"
    directory.mkdir()
    for tool in ("aws", "az", "gcloud", "python3"):
        script = directory / tool
        script.write_text(
            f'#!/usr/bin/env bash\necho "{tool} $*" >> "$STUB_LOG"\n'
            f'if [ "{tool}" = python3 ]; then printf "%s\\n" "$@" > "$STUB_ARGS"; fi\n'
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return directory


LOGIN_ENV = {
    "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/medbench-ci",
    "AWS_ID_TOKEN": "aws.jwt",
    "AZURE_CLIENT_ID": "client",
    "AZURE_TENANT_ID": "tenant",
    "AZURE_SUBSCRIPTION_ID": "subscription",
    "AZURE_ID_TOKEN": "azure.jwt",
    "GCP_WORKLOAD_IDENTITY_PROVIDER": (
        "projects/1/locations/global/workloadIdentityPools/ci/providers/gitlab"
    ),
    "GCP_SERVICE_ACCOUNT": "medbench@medbench-ci.iam.gserviceaccount.com",
    "GCP_PROJECT": "medbench-ci",
    "GCP_ID_TOKEN": "gcp.jwt",
    "HETZNER_ROBOT_USER": "robot",
    "HETZNER_ROBOT_PASSWORD": "secret",
}


def login(
    provider: str, tmp_path: Path, stub_bin: Path, env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], dict[str, str], str]:
    script = f'source "{LOGIN}" {provider}\nenv > "$TMP/env"'
    full_env = {
        **env,
        "TMP": str(tmp_path),
        "STUB_LOG": str(tmp_path / "log"),
        "CLOUD_GPU_LOGIN_DIR": str(tmp_path / "login"),
    }
    result = run_bash(script, tmp_path, full_env, path=f"{stub_bin}:")
    exported: dict[str, str] = {}
    if (tmp_path / "env").exists():
        for line in (tmp_path / "env").read_text().splitlines():
            key, _, value = line.partition("=")
            exported[key] = value
    log = (tmp_path / "log").read_text() if (tmp_path / "log").exists() else ""
    return result, exported, log


def test_aws_login_uses_a_web_identity_token_file(
    tmp_path: Path, stub_bin: Path
) -> None:
    result, exported, log = login("aws", tmp_path, stub_bin, LOGIN_ENV)
    assert result.returncode == 0, result.stderr
    token_file = Path(exported["AWS_WEB_IDENTITY_TOKEN_FILE"])
    assert token_file.read_text() == "aws.jwt"
    assert exported["AWS_ROLE_ARN"] == LOGIN_ENV["AWS_ROLE_ARN"]
    assert exported["AWS_DEFAULT_REGION"] == "us-east-1"
    assert "aws sts get-caller-identity" in log
    assert stat.S_IMODE(token_file.parent.stat().st_mode) == 0o700


def test_azure_login_gives_az_and_terraform_the_token(
    tmp_path: Path, stub_bin: Path
) -> None:
    result, exported, log = login("azure", tmp_path, stub_bin, LOGIN_ENV)
    assert result.returncode == 0, result.stderr
    assert exported["ARM_USE_OIDC"] == "true"
    assert Path(exported["ARM_OIDC_TOKEN_FILE_PATH"]).read_text() == "azure.jwt"
    assert exported["ARM_SUBSCRIPTION_ID"] == "subscription"
    assert "--federated-token azure.jwt" in log
    assert "az account set --subscription subscription" in log


def test_gcp_login_writes_an_external_account_credential(
    tmp_path: Path, stub_bin: Path
) -> None:
    result, exported, log = login("gcp", tmp_path, stub_bin, LOGIN_ENV)
    assert result.returncode == 0, result.stderr
    credentials = exported["GOOGLE_APPLICATION_CREDENTIALS"]
    assert exported["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] == credentials
    assert exported["CLOUD_GPU_GCP_PROJECT"] == "medbench-ci"
    create = next(line for line in log.splitlines() if "create-cred-config" in line)
    assert LOGIN_ENV["GCP_WORKLOAD_IDENTITY_PROVIDER"] in create
    assert f"--output-file={credentials}" in create
    source = re.findall(r"--credential-source-file=(\S+)", create)[0]
    assert Path(source).read_text() == "gcp.jwt"


@pytest.mark.parametrize("provider", ["aws", "azure", "gcp", "hetzner"])
def test_a_missing_setting_stops_the_login_and_says_what_to_add(
    provider: str, tmp_path: Path, stub_bin: Path
) -> None:
    result, _, log = login(provider, tmp_path, stub_bin, {})
    assert result.returncode == 1
    assert "this job has no" in result.stderr
    assert "docs/cloud-gpu.md" in result.stderr
    assert log == ""


def test_an_unknown_provider_is_refused(tmp_path: Path, stub_bin: Path) -> None:
    result, _, _ = login("ovh", tmp_path, stub_bin, LOGIN_ENV)
    assert result.returncode == 2


@pytest.fixture
def stub_repo(tmp_path: Path) -> Path:
    """A checkout whose install and login scripts do nothing."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "ci" / "gitlab").mkdir(parents=True)
    (repo / "scripts" / "cloud-gpu-install-tools.sh").write_text(
        'echo "install $*" >> "$STUB_LOG"\n'
    )
    (repo / "ci" / "gitlab" / "cloud-login.sh").write_text(
        'echo "login $*" >> "$STUB_LOG"\n'
    )
    return repo


def reap(stub_repo: Path, stub_bin: Path, env: dict[str, str]) -> tuple[str, str]:
    script = "\n".join(flatten(job("cloud-gpu:reap")["script"]))
    result = run_bash(
        script,
        stub_repo,
        {
            **env,
            "STUB_LOG": str(stub_repo / "log"),
            "STUB_ARGS": str(stub_repo / "args"),
        },
        path=f"{stub_bin}:",
    )
    assert result.returncode == 0, result.stderr
    log = (stub_repo / "log").read_text() if (stub_repo / "log").exists() else ""
    return result.stdout, log


def test_the_reaper_reaps_every_configured_cloud(
    stub_repo: Path, stub_bin: Path
) -> None:
    _, log = reap(
        stub_repo,
        stub_bin,
        {**LOGIN_ENV, "CLOUD_GPU_AWS_REGIONS": "us-east-1 eu-west-1"},
    )
    assert "install aws azure gcp hetzner" in log
    for provider in ("aws", "azure", "gcp", "hetzner"):
        assert f"login {provider}" in log
    command = (stub_repo / "args").read_text().splitlines()
    assert command[:2] == ["-m", "scripts.cloud_gpu"]
    parsed = cli.build_parser().parse_args(command[2:])
    assert parsed.providers == ["aws", "azure", "gcp", "hetzner"]
    assert parsed.regions == ["us-east-1", "eu-west-1"]
    assert parsed.gcp_project == "medbench-ci"
    assert parsed.yes


def test_the_reaper_passes_when_no_cloud_is_configured(
    stub_repo: Path, stub_bin: Path
) -> None:
    stdout, log = reap(stub_repo, stub_bin, {})
    assert stdout.startswith("No cloud has settings")
    assert log == ""
