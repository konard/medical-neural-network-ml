"""Tests for the cloud GPU orchestrator: config, CLI, Terraform and flows.

Nothing here talks to a cloud. Terraform and ssh are small fake programs
on PATH, and the ssh provider runs end to end against the local machine
with UniMedVL's echo engine, the way `all --provider ssh` runs on a real
GPU host.
"""

from __future__ import annotations

import io
import json
import os
import signal
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from scripts.cloud_gpu import cli, flows, hetzner, terraform
from scripts.cloud_gpu import config as cfg
from tests.test_cloud_gpu_scripts import free_port
from tests.test_medbench_model_run import linux_x86_64


def make_config(tmp_path: Path, **values: object) -> cfg.RunConfig:
    settings: dict[str, object] = {
        "provider": "aws",
        "mode": "vm",
        "model": "medgemma-4b-it",
        "run_id": "20261020120000-abcd",
        "workdir": str(tmp_path / "run"),
        "created_at": 1_800_000_000,
        "allowed_cidrs": ["203.0.113.7/32"],
        "commit": "abc123",
    }
    settings.update(values)
    config = cfg.RunConfig(**settings)  # type: ignore[arg-type]
    config.fill_defaults()
    return config


def fake_tool(directory: Path, name: str, body: str) -> Path:
    """Write an executable bash script ``name`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def on_path(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")


# Config


def test_run_ids_sort_by_time_and_are_valid_names() -> None:
    run_id = cfg.new_run_id(1_800_000_000)
    assert run_id.startswith("20270115080000-")
    assert cfg.check_run_id(run_id) == run_id
    for bad in ("", "ab", "Upper", "-lead", "a" * 32, "a_b"):
        with pytest.raises(ValueError, match="run id"):
            cfg.check_run_id(bad)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"provider": "nowhere"}, "unknown provider"),
        ({"provider": "local", "mode": "k8s"}, "supports the modes"),
        ({"provider": "kubernetes", "mode": "vm"}, "supports the modes"),
        ({"model": "gpt"}, "unknown model"),
        ({"ttl_minutes": 5}, "ttl"),
        ({"ttl_minutes": 24 * 60 + 1}, "ttl"),
        ({"provider": "gcp"}, "--gcp-project"),
        ({"provider": "ssh"}, "--ssh-host"),
        ({"mode": "runner"}, "--github-repository"),
        ({"extra_env": {"lower": "1"}}, "environment variable"),
    ],
)
def test_validate_rejects_what_cannot_work(
    tmp_path: Path, values: dict[str, object], message: str
) -> None:
    config = make_config(tmp_path)
    for key, value in values.items():
        setattr(config, key, value)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_defaults_follow_the_model(tmp_path: Path) -> None:
    small = make_config(tmp_path)
    large = make_config(tmp_path, model="unimedvl")
    assert small.region == "us-east-1"
    assert small.instance_type in cfg.HOURLY_PRICE
    assert cfg.HOURLY_PRICE[large.instance_type] > cfg.HOURLY_PRICE[small.instance_type]
    assert (small.disk_gb, large.disk_gb) == (100, 200)
    gcp = make_config(tmp_path, provider="gcp", region="europe-west4")
    assert gcp.zone == "europe-west4-a"
    assert gcp.accelerator_type == ""  # G2 machines come with their L4
    gke = make_config(tmp_path, provider="gcp", mode="k8s")
    assert gke.accelerator_type == "nvidia-l4"  # GKE node pools name it
    runner = make_config(tmp_path, mode="runner", github_repository="o/r")
    assert runner.runner_labels == [
        "self-hosted",
        "linux",
        "x64",
        "medbench-20261020120000-abcd",
    ]
    mine = make_config(tmp_path, instance_type="g6e.xlarge", disk_gb=500)
    assert (mine.instance_type, mine.disk_gb) == ("g6e.xlarge", 500)


def test_model_env_for_the_gpu_host(tmp_path: Path) -> None:
    config = make_config(tmp_path, extra_env={"PORT": "9000"}, require_gpu=False)
    assert config.model_env() == {
        "GPU_MODEL": "medgemma-4b-it",
        "COUNT": "20",
        "TIME_BUDGET": "5400",
        "STEPS": "download run",
        "REQUIRE_GPU": "0",
        "MEDBENCH_COMMIT": "abc123",
        "PORT": "9000",
    }
    assert make_config(tmp_path, tasks="ocr-history").model_env()["TASKS"] == (
        "ocr-history"
    )


def test_estimate_states_the_worst_case(tmp_path: Path) -> None:
    config = make_config(tmp_path, instance_type="g6.xlarge", ttl_minutes=60)
    price = cfg.HOURLY_PRICE["g6.xlarge"]
    assert config.estimate() == (
        f"g6.xlarge: about {price:.2f} USD/hour on demand, at most "
        f"{price:.2f} USD for the 60-minute time to live"
    )
    server = make_config(tmp_path, provider="hetzner", instance_type="GEX45")
    assert "EUR" in server.estimate()
    assert "setup fee of 209 EUR" in server.estimate()
    assert (
        "no price on record"
        in make_config(tmp_path, instance_type="x9.huge").estimate()
    )


def test_state_round_trip_is_private_and_merges_outputs(tmp_path: Path) -> None:
    config = make_config(tmp_path, extra_env={"A": "1"})
    path = cfg.save_state(config, {"host": "198.51.100.1"})
    cfg.save_state(config, {"runner_id": 7})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded, outputs = cfg.load_state(config.work)
    assert loaded == config
    assert outputs == {"host": "198.51.100.1", "runner_id": 7}
    data = json.loads(path.read_text())
    data["config"]["field_from_a_newer_version"] = 1
    path.write_text(json.dumps(data))
    assert cfg.load_state(config.work)[0] == config


# Command line


def test_paid_resources_need_yes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["up", "--allowed-cidr", "203.0.113.7/32", "--commit", "abc"]
    assert cli.main([*argv, "--workdir", str(tmp_path / "run")]) == 2
    assert "add --yes" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_reap_needs_yes_or_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["reap", "--provider", "aws"]) == 2
    assert "--dry-run" in capsys.readouterr().err
    assert (
        cli.main(["reap", "--provider", "gcp", "--dry-run", "--gcp-project", ""]) == 2
    )


def test_plan_prints_settings_and_terraform_variables(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["plan", "--provider", "azure", "--model", "medgemma-27b-it"]
    assert cli.main([*argv, "--run-id", "plan-test", "--commit", "abc"]) == 0
    out = capsys.readouterr().out
    assert '"instance_type": "Standard_NV36ads_A10_v5"' in out
    assert '"name": "medbench-plan-test"' in out
    assert '"ssh_public_key": "<per-run key>"' in out
    assert "estimate:" in out


def test_job_manifest_command(tmp_path: Path) -> None:
    target = tmp_path / "job.yaml"
    argv = ["job-manifest", "--provider", "kubernetes", "--mode", "k8s"]
    argv += ["--run-id", "manifest", "--commit", "abc", "--output", str(target)]
    assert cli.main([*argv, "--gpu-count", "0", "--env", "UNIMEDVL_ENGINE=echo"]) == 0
    text = target.read_text()
    assert 'name: "medbench-manifest"' in text
    assert (
        '            - name: COLLECT_SECONDS\n              value: "1800"\n'
        '            - name: UNIMEDVL_ENGINE\n              value: "echo"\n'
    ) in text


def test_errors_become_exit_code_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["plan", "--provider", "ssh", "--commit", "abc"]) == 1
    assert "--ssh-host is required" in capsys.readouterr().err


def test_run_and_down_need_a_workdir(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["down"]) == 2
    assert "needs --workdir" in capsys.readouterr().err


# Terraform


TERRAFORM_FAKE = """\
printf '%s|%s\\n' "${TF_IN_AUTOMATION:-}" "$*" >> "$FAKE_LOG"
case "$1" in
  output) printf '%s' '{"host":{"value":"198.51.100.9"},"ssh_user":{"value":"ubuntu"}}' ;;
  destroy) exit "${FAKE_DESTROY_STATUS:-0}" ;;
esac
"""


def test_terraform_apply_and_destroy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tool(tmp_path / "bin", "terraform", TERRAFORM_FAKE)
    on_path(monkeypatch, tmp_path / "bin")
    log = tmp_path / "terraform.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    config = make_config(tmp_path, backend_config="backend.hcl")
    terraform.destroy(config)  # nothing prepared yet: nothing to do
    assert not log.exists()
    terraform.prepare(config, "ssh-ed25519 AAAA")
    assert terraform.apply(config) == {"host": "198.51.100.9", "ssh_user": "ubuntu"}
    terraform.destroy(config)
    calls = log.read_text().splitlines()
    assert calls == [
        "1|init -input=false -lockfile=readonly -no-color -backend-config=backend.hcl",
        "1|apply -input=false -auto-approve -no-color",
        "1|output -json -no-color",
        "1|init -input=false -lockfile=readonly -no-color -backend-config=backend.hcl",
        "1|destroy -input=false -auto-approve -no-color",
    ]
    # Without a backend block Terraform ignores -backend-config.
    override = terraform.module_dir(config) / "backend_override.tf"
    assert override.read_text() == 'terraform {\n  backend "s3" {}\n}\n'


def test_backend_type_and_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    argv = ["plan", "--provider", "gcp", "--gcp-project", "p", "--commit", "abc"]
    argv += ["--allowed-cidr", "203.0.113.7/32", "--backend-config", "state.hcl"]
    args = cli.build_parser().parse_args([*argv, "--backend-type", "http"])
    config = cli.config_from_args(args, needs_cidr=True)
    assert config.backend_config == str(tmp_path / "state.hcl")
    assert config.backend_type == "http"
    config.backend_type = "http\n}"
    with pytest.raises(ValueError, match="backend type"):
        config.validate()


def test_down_runs_every_step_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tool(tmp_path / "bin", "terraform", TERRAFORM_FAKE)
    on_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "terraform.log"))
    monkeypatch.setenv("FAKE_DESTROY_STATUS", "1")
    config = make_config(tmp_path, mode="runner", github_repository="o/r")
    terraform.prepare(config, "ssh-ed25519 AAAA")
    removed: list[int] = []
    monkeypatch.setattr(
        flows.github, "delete_runner", lambda repo, number: removed.append(number)
    )
    with pytest.raises(RuntimeError, match="cleanup was incomplete") as error:
        flows.down(config, {"runner_id": 42})
    assert "terraform destroy" in str(error.value)
    assert removed == [42]
    assert "stopped_at" in cfg.load_state(config.work)[1]


def test_down_cancels_a_hetzner_server_delivered_late(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, provider="hetzner", instance_type="GEX45")
    done: list[str] = []
    monkeypatch.setattr(hetzner, "order_status", lambda _: {"server_number": 77})
    monkeypatch.setattr(hetzner, "cancel", lambda n: done.append(f"cancel {n}"))
    monkeypatch.setattr(hetzner, "delete_key", lambda f: done.append(f"key {f}"))
    flows.down(config, {"hetzner_transaction": "B20261020-1", "hetzner_key": "aa:bb"})
    assert done == ["cancel 77", "key aa:bb"]


def test_down_deletes_the_hetzner_key_when_cancelling_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, provider="hetzner", instance_type="GEX45")
    keys: list[str] = []

    def refuse(number: int) -> None:
        raise hetzner.RobotError(500, f"cannot cancel {number}")

    monkeypatch.setattr(hetzner, "cancel", refuse)
    monkeypatch.setattr(hetzner, "delete_key", keys.append)
    with pytest.raises(RuntimeError, match="cannot cancel 5"):
        flows.down(config, {"hetzner_server": 5, "hetzner_key": "aa:bb"})
    assert keys == ["aa:bb"]


# Reports


def tar_of(members: dict[str, bytes], *, link: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo("escape")
            info.type = tarfile.SYMTYPE
            info.linkname = link
            archive.addfile(info)
    return buffer.getvalue()


def test_reports_are_extracted_inside_the_target(tmp_path: Path) -> None:
    names = flows.remote.extract_reports(
        tar_of({"./a.json": b"{}", "./sub/b.json": b"[]"}), tmp_path / "reports"
    )
    assert names == ["./a.json", "./sub/b.json"]
    assert (tmp_path / "reports" / "sub" / "b.json").read_text() == "[]"


@pytest.mark.parametrize(
    "archive",
    [tar_of({"../outside.json": b"{}"}), tar_of({}, link="/etc/passwd")],
    ids=["parent", "symlink"],
)
def test_reports_cannot_escape_the_target(tmp_path: Path, archive: bytes) -> None:
    with pytest.raises(tarfile.TarError):
        flows.remote.extract_reports(archive, tmp_path / "reports")
    assert not (tmp_path / "outside.json").exists()


def test_absolute_report_names_stay_in_the_target(tmp_path: Path) -> None:
    flows.remote.extract_reports(tar_of({"/abs.json": b"{}"}), tmp_path / "reports")
    assert (tmp_path / "reports" / "abs.json").is_file()


# The ssh provider end to end, with a fake ssh that runs commands locally


SSH_FAKE = """\
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o|-p|-i) shift 2 ;;
    -t) shift ;;
    *@*) printf '%s\\n' "$1" >> "$FAKE_SSH_LOG"; shift; break ;;
    *) echo "fake ssh: unexpected $1" >&2; exit 255 ;;
  esac
done
cd "$FAKE_SSH_HOME"
exec bash -c "$*"
"""


@linux_x86_64
def test_all_runs_on_an_ssh_host_and_collects_the_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "gpu-host"
    home.mkdir()
    fake_tool(tmp_path / "bin", "ssh", SSH_FAKE)
    on_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )
    monkeypatch.setenv("FAKE_SSH_HOME", str(home))
    monkeypatch.setenv("FAKE_SSH_LOG", str(tmp_path / "ssh.log"))
    reports = tmp_path / "reports"
    status = cli.main(
        [
            "all",
            "--provider",
            "ssh",
            "--ssh-host",
            "gpu.example",
            "--ssh-user",
            "ci",
            "--model",
            "unimedvl",
            "--run-id",
            "ssh-e2e",
            "--commit",
            "abc123",
            "--workdir",
            str(tmp_path / "run"),
            "--reports",
            str(reports),
            "--steps",
            "run",
            "--no-require-gpu",
            "--tasks",
            "diagnosis",
            "--count",
            "2",
            "--env",
            f"MEDBENCH_PYTHON={sys.executable}",
            "--env",
            "UNIMEDVL_ENGINE=echo",
            "--env",
            f"UNIMEDVL_ECHO_PYTHON={sys.executable}",
            "--env",
            "RUN_PYTEST=0",
            "--env",
            f"PORT={free_port()}",
        ]
    )
    assert status == 0
    assert (home / "medbench" / "scripts" / "gpu-host-run.sh").is_file()
    assert set((tmp_path / "ssh.log").read_text().split()) == {"ci@gpu.example"}
    summary = json.loads((reports / "cloud-gpu-run.json").read_text())
    assert summary["provider"] == "ssh"
    assert summary["exit_code"] == 0
    host = json.loads((reports / "gpu-host-run.json").read_text())
    assert host["model"] == "unimedvl"
    assert json.loads((reports / "diagnosis.json").read_text())["items_run"] == 2
    _, outputs = cfg.load_state(tmp_path / "run")
    assert "stopped_at" in outputs


HANGING_SSH = """\
printf 'connected\\n' >> "$FAKE_SSH_LOG"
exec sleep 600
"""


def test_sigterm_still_runs_down(tmp_path: Path) -> None:
    """A cancelled job or an expired budget sends SIGTERM; `all` must still
    reach `down`, or the machine keeps running (and costing) until the
    reaper or its own timer finds it."""
    fake_tool(tmp_path / "bin", "ssh", HANGING_SSH)
    log = tmp_path / "ssh.log"
    env = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "FAKE_SSH_LOG": str(log),
    }
    command = [sys.executable, "-m", "scripts.cloud_gpu", "all", "--provider", "ssh"]
    command += ["--ssh-host", "gpu.example", "--model", "unimedvl"]
    command += ["--run-id", "sigterm-test", "--workdir", str(tmp_path / "run")]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 60
    while not log.exists() and process.poll() is None:
        assert time.monotonic() < deadline, "the fake ssh was never called"
        time.sleep(0.1)
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=60)
    assert process.returncode == 128 + signal.SIGTERM, output
    assert "received SIGTERM" in output
    assert "run sigterm-test cleaned up" in output
    _, outputs = cfg.load_state(tmp_path / "run")
    assert "stopped_at" in outputs


def test_ssh_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flows.remote, "require_tool", lambda name: f"/usr/bin/{name}")
    config = make_config(tmp_path, provider="gcp", gcp_project="p")
    ssh = flows.ssh_for(config, {"host": "198.51.100.3", "ssh_user": "medbench"})
    argv = ssh.argv()
    assert argv[-1] == "medbench@198.51.100.3"
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=accept-new" in argv
    assert f"UserKnownHostsFile={config.work / 'ssh' / 'known_hosts'}" in argv
    assert argv[argv.index("-i") + 1] == str(config.work / "ssh" / "id_ed25519")
    assert ssh.sudo() == "sudo -n "
    own = flows.ssh_for(make_config(tmp_path, provider="ssh", ssh_host="h"), {})
    assert own.key is None
    assert own.sudo() == ""  # root, when no user was given
