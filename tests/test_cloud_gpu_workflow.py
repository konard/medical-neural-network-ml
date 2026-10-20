"""Tests for the Cloud GPU Tests workflow (issue #23).

The workflow only installs tools, logs in and calls the orchestrator, so the
tests run its shell steps against the orchestrator's real argument parser:
the arguments it builds must parse, a missing cloud setting must stop the
run before anything is created, the machine must be deleted after any
failure, and the working directory (SSH key, Terraform state) must never be
uploaded.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts.cloud_gpu import cli
from scripts.cloud_gpu import config as cfg
from tests.workflow_helpers import (
    job_condition,
    read_workflow,
    workflow_job_block,
    workflow_step_block,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = read_workflow("cloud-gpu-tests.yml")
ORCHESTRATE = workflow_job_block(WORKFLOW, "orchestrate")
REAP = workflow_job_block(WORKFLOW, "reap")


def step_script(job: str, step: str) -> str:
    """The shell script of a step's ``run: |`` block."""
    block = workflow_step_block(job, step)
    body = block.split("        run: |\n", 1)[1]
    return textwrap.dedent(body) + "\n"


def run_bash(
    script: str, tmp_path: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "RUNNER_TEMP": str(tmp_path), **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def pull_request_paths() -> list[str]:
    triggers = WORKFLOW.split("\njobs:\n", 1)[0]
    section = triggers.split("  pull_request:\n", 1)[1].split("\n  schedule:", 1)[0]
    return re.findall(r"^      - '([^']+)'$", section, re.MULTILINE)


def test_pull_requests_run_when_any_file_the_checks_use_changes() -> None:
    patterns = pull_request_paths()
    used = [
        *[p for p in (ROOT / "infra").rglob("*") if p.is_file()],
        *(ROOT / "scripts" / "cloud_gpu").glob("*.py"),
        *(ROOT / "scripts").glob("gpu-*.sh"),
        *(ROOT / "scripts").glob("cloud-gpu-*.sh"),
        *(ROOT / "ci" / "gitlab").iterdir(),
        ROOT / "scripts" / "check-infra.sh",
        ROOT / "scripts" / "run-with-budget-warning.sh",
        ROOT / "scripts" / "medbench-model-run.sh",
        ROOT / "examples" / "cloud_gpu_kind.sh",
        ROOT / "pyproject.toml",
        ROOT / ".github" / "workflows" / "cloud-gpu-tests.yml",
    ]
    for path in used:
        relative = path.relative_to(ROOT).as_posix()
        assert any(fnmatch.fnmatch(relative, p) for p in patterns), relative


def test_paid_jobs_run_only_from_main_and_never_for_a_pull_request() -> None:
    for job in ("orchestrate", "gpu-tests", "reap"):
        condition = job_condition(WORKFLOW, job)
        assert "github.ref == 'refs/heads/main'" in condition, job
        assert "pull_request" not in condition, job
    assert "github.event_name == 'workflow_dispatch'" in job_condition(
        WORKFLOW, "orchestrate"
    )
    assert "    environment: cloud-gpu\n" in ORCHESTRATE
    assert "    environment: cloud-gpu-reaper\n" in REAP
    for job in ("infra", "kind-job", "gpu-tests"):
        assert "secrets." not in workflow_job_block(WORKFLOW, job), job


def test_the_gpu_job_asks_for_the_label_of_this_runs_runner() -> None:
    run_id = re.search(r"^      RUN_ID: (.+)$", ORCHESTRATE, re.MULTILINE)
    assert run_id
    job = workflow_job_block(WORKFLOW, "gpu-tests")
    label = f"medbench-{run_id.group(1)}"
    assert f"runs-on: [self-hosted, linux, x64, '{label}']" in job
    # The longest run id GitHub can produce today is still a valid one.
    example = run_id.group(1).replace("${{ github.run_id }}", "99999999999")
    example = example.replace("${{ github.run_attempt }}", "99")
    assert cfg.check_run_id(example) == example
    config = cfg.RunConfig(
        provider="aws",
        mode="runner",
        model="medgemma-4b-it",
        run_id=example,
        workdir=ROOT / ".cache" / "cloud-gpu" / example,
        created_at=1_800_000_000,
    )
    config.fill_defaults()
    assert config.runner_labels == [
        "self-hosted",
        "linux",
        "x64",
        f"medbench-{example}",
    ]


@pytest.mark.parametrize("provider", ["aws", "azure", "gcp", "hetzner"])
@pytest.mark.parametrize("mode", ["vm", "k8s", "runner"])
def test_the_settings_step_builds_arguments_the_cli_accepts(
    tmp_path: Path, provider: str, mode: str
) -> None:
    env = {
        "PROVIDER": provider,
        "MODE": mode,
        "MODEL": "medgemma-4b-it",
        "COUNT": "2",
        "SPOT": "true",
        "TTL": "360",
        "RUN_ID": "gh-18000000000-1",
        "WORKDIR": ".cache/cloud-gpu/run",
        "REPORTS": "cloud-gpu-reports",
        "CLOUD_GPU_REGION": "",
        "GITHUB_REPOSITORY": "konard/medical-neural-network-ml",
        "GITHUB_SHA": "0123456789abcdef0123456789abcdef01234567",
        "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/medbench",
        "AZURE_CLIENT_ID": "client",
        "AZURE_TENANT_ID": "tenant",
        "AZURE_SUBSCRIPTION_ID": "00000000-0000-0000-0000-000000000000",
        "GCP_WORKLOAD_IDENTITY_PROVIDER": "projects/1/locations/global/x",
        "GCP_SERVICE_ACCOUNT": "medbench@p.iam.gserviceaccount.com",
        "GCP_PROJECT": "medbench-ci",
        "HAS_HETZNER_ROBOT_USER": "true",
        "HAS_HETZNER_ROBOT_PASSWORD": "true",
        "HAS_CLOUD_GPU_RUNNER_TOKEN": "true",
    }
    result = run_bash(
        step_script(ORCHESTRATE, "Check the cloud settings"), tmp_path, env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = (tmp_path / "cloud-gpu-args").read_text().splitlines()
    parsed = cli.build_parser().parse_args(["up", *args, "--yes"])
    assert (parsed.provider, parsed.mode, parsed.spot) == (provider, mode, True)
    assert parsed.run_id == "gh-18000000000-1"
    assert parsed.commit == env["GITHUB_SHA"]
    config = cli.config_from_args(parsed, needs_cidr=False)
    config.validate()
    assert config.name == "medbench-gh-18000000000-1"
    if provider == "gcp":
        assert config.gcp_project == "medbench-ci"


@pytest.mark.parametrize(
    ("provider", "mode", "unset", "named"),
    [
        ("aws", "vm", "AWS_ROLE_ARN", "variable AWS_ROLE_ARN"),
        ("azure", "k8s", "AZURE_TENANT_ID", "variable AZURE_TENANT_ID"),
        ("gcp", "vm", "GCP_PROJECT", "variable GCP_PROJECT"),
        ("hetzner", "vm", "HAS_HETZNER_ROBOT_USER", "secret HETZNER_ROBOT_USER"),
        (
            "aws",
            "runner",
            "HAS_CLOUD_GPU_RUNNER_TOKEN",
            "secret CLOUD_GPU_RUNNER_TOKEN",
        ),
    ],
)
def test_a_missing_setting_stops_the_run_and_says_where_to_add_it(
    tmp_path: Path, provider: str, mode: str, unset: str, named: str
) -> None:
    env = {
        "PROVIDER": provider,
        "MODE": mode,
        "AWS_ROLE_ARN": "arn",
        "AZURE_CLIENT_ID": "c",
        "AZURE_TENANT_ID": "t",
        "AZURE_SUBSCRIPTION_ID": "s",
        "GCP_WORKLOAD_IDENTITY_PROVIDER": "w",
        "GCP_SERVICE_ACCOUNT": "a",
        "GCP_PROJECT": "p",
        "HAS_HETZNER_ROBOT_USER": "true",
        "HAS_HETZNER_ROBOT_PASSWORD": "true",
        "HAS_CLOUD_GPU_RUNNER_TOKEN": "true",
    }
    env[unset] = "false" if unset.startswith("HAS_") else ""
    result = run_bash(
        step_script(ORCHESTRATE, "Check the cloud settings"), tmp_path, env
    )
    assert result.returncode == 1
    assert f"::error::The cloud-gpu environment has no {named}." in result.stdout
    assert "docs/cloud-gpu.md" in result.stdout
    assert not (tmp_path / "cloud-gpu-args").exists()


def test_secrets_are_only_seen_as_booleans_before_the_machine_is_created() -> None:
    check = workflow_step_block(ORCHESTRATE, "Check the cloud settings")
    assert re.findall(r"secrets\.[A-Z_]+ != ''", check)
    assert not re.search(r"secrets\.[A-Z_]+ }}", check)


def test_the_machine_is_deleted_after_any_failure() -> None:
    steps = re.findall(r"^      - name: (.+)$", ORCHESTRATE, re.MULTILINE)
    create = steps.index("Create the GPU machine")
    run = steps.index("Run the GPU tests")
    delete = steps.index("Delete the GPU machine")
    assert create < run < delete
    block = workflow_step_block(ORCHESTRATE, "Delete the GPU machine")
    assert "if: always() && inputs.action == 'test'" in block
    assert "python -m scripts.cloud_gpu down --workdir" in block
    assert "state.json" in block
    cancel = workflow_step_block(ORCHESTRATE, "Cancel the GPU job that no runner took")
    assert "if: failure() && inputs.mode == 'runner'" in cancel
    assert 'gh run cancel "$GITHUB_RUN_ID"' in cancel


def test_the_delete_step_skips_a_run_that_created_nothing(tmp_path: Path) -> None:
    script = step_script(ORCHESTRATE, "Delete the GPU machine")
    result = run_bash(
        script, tmp_path, {"WORKDIR": "missing", "DOWN_BUDGET_SECONDS": "5"}
    )
    assert result.returncode == 0, result.stderr
    assert "Nothing was created." in result.stdout


def test_the_working_directory_is_never_uploaded() -> None:
    uploads = re.findall(
        r"uses: actions/upload-artifact@.*\n        with:\n((?:          .*\n)+)",
        WORKFLOW + "\n",
    )
    assert len(uploads) == 3
    for upload in uploads:
        assert ".cache/cloud-gpu/run" not in upload
        assert "state.json" not in upload
    assert "path: cloud-gpu-reports\n" in ORCHESTRATE


@pytest.fixture
def stub_repo(tmp_path: Path) -> Path:
    """A directory whose budget wrapper prints the command it would run."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run-with-budget-warning.sh").write_text(
        'shift 2\nprintf "%s\\n" "$@"\n'
    )
    return tmp_path


def reap_command(stub_repo: Path, **clouds: str) -> list[str]:
    env = {
        "REAP_BUDGET_SECONDS": "2400",
        "AWS_REGIONS": "us-east-1 eu-west-1",
        "GCP_PROJECT": "medbench-ci",
        "GITHUB_REPOSITORY": "konard/medical-neural-network-ml",
    }
    for cloud in ("aws", "azure", "gcp", "hetzner", "github"):
        env[f"REAP_{cloud.upper()}"] = clouds.get(cloud, "false")
    result = run_bash(step_script(REAP, "Delete expired resources"), stub_repo, env)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_the_reaper_reaps_every_configured_cloud(stub_repo: Path) -> None:
    command = reap_command(
        stub_repo, aws="true", azure="true", gcp="true", hetzner="true", github="true"
    )
    assert command[:4] == ["python", "-m", "scripts.cloud_gpu", "reap"]
    parsed = cli.build_parser().parse_args(command[3:])
    assert parsed.providers == ["aws", "azure", "gcp", "hetzner", "github"]
    assert parsed.regions == ["us-east-1", "eu-west-1"]
    assert parsed.gcp_project == "medbench-ci"
    assert parsed.yes


def test_the_reaper_passes_when_no_cloud_is_configured(stub_repo: Path) -> None:
    command = reap_command(stub_repo)
    assert command[0].startswith("::notice::No cloud is configured")


def test_every_orchestrator_call_uses_a_real_subcommand() -> None:
    subcommands = set(re.findall(r"python -m scripts\.cloud_gpu (\S+)", WORKFLOW))
    assert subcommands == {"plan", "up", "wait-runner", "run", "down", "reap"}
    parser = cli.build_parser()
    for name in subcommands:
        with pytest.raises(SystemExit) as info:
            parser.parse_args([name, "--help"])
        assert info.value.code == 0
