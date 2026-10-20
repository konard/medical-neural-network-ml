"""Tests for the Terraform modules and the Kubernetes Job template (issue #23).

The orchestrator writes terraform.tfvars.json for each cloud and mode; every
key must be a declared variable and every variable without a default must
be given, or `terraform apply` stops after the machine is half-created. The
rendered Job must be valid Kubernetes (kubeconform, when installed).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu import flows, kube, terraform

ROOT = Path(__file__).resolve().parents[1]
TERRAFORM = ROOT / "infra" / "terraform"
MODULES = [(cloud, kind) for cloud in cfg.CLOUDS for kind in ("vm", "k8s")]
VARIABLE = re.compile(r'^variable "([a-z_]+)" \{\n(.*?)^\}', re.MULTILINE | re.DOTALL)


def make_config(tmp_path: Path, **values: object) -> cfg.RunConfig:
    settings: dict[str, object] = {
        "provider": "aws",
        "mode": "vm",
        "model": "medgemma-27b-it",
        "run_id": "20261020120000-abcd",
        "workdir": str(tmp_path / "run"),
        "created_at": 1_800_000_000,
        "gcp_project": "medbench-test",
        "azure_subscription": "00000000-0000-0000-0000-000000000000",
        "allowed_cidrs": ["203.0.113.7/32"],
        "github_repository": "owner/repo",
        "commit": "0123456789abcdef",
    }
    settings.update(values)
    config = cfg.RunConfig(**settings)  # type: ignore[arg-type]
    config.fill_defaults()
    config.validate()
    return config


def variables(cloud: str, kind: str) -> dict[str, bool]:
    """Variable names of a module and whether each has a default."""
    text = (TERRAFORM / cloud / kind / "variables.tf").read_text()
    return {name: "default" in body for name, body in VARIABLE.findall(text)}


@pytest.mark.parametrize(("cloud", "kind"), MODULES)
@pytest.mark.parametrize("model", cfg.MODELS)
def test_tfvars_match_the_module(
    tmp_path: Path, cloud: str, kind: str, model: str
) -> None:
    mode = "k8s" if kind == "k8s" else "vm"
    config = make_config(tmp_path, provider=cloud, mode=mode, model=model)
    assert terraform.module_kind(config) == kind
    values = terraform.tfvars(config, "ssh-ed25519 AAAA test")
    declared = variables(cloud, kind)
    assert declared, f"no variables parsed for {cloud}/{kind}"
    unknown = sorted(set(values) - set(declared))
    assert not unknown, f"{cloud}/{kind} does not declare {unknown}"
    required = {name for name, has_default in declared.items() if not has_default}
    missing = sorted(required - set(values))
    assert not missing, f"the orchestrator never sets {missing} for {cloud}/{kind}"
    assert values["instance_type"] in cfg.HOURLY_PRICE


@pytest.mark.parametrize(("cloud", "kind"), MODULES)
def test_runner_mode_uses_the_vm_module(tmp_path: Path, cloud: str, kind: str) -> None:
    del kind
    config = make_config(tmp_path, provider=cloud, mode="runner")
    assert terraform.module_dir(config).name == "vm"
    assert config.runner_labels[-1] == config.name


@pytest.mark.parametrize(("cloud", "kind"), MODULES)
def test_modules_are_complete(cloud: str, kind: str) -> None:
    """Pinned providers, the reaper's tags, and the outputs flows.py reads."""
    module = TERRAFORM / cloud / kind
    for name in ("main.tf", "variables.tf", "outputs.tf", "versions.tf"):
        assert (module / name).is_file(), f"{module}/{name}"
    lock = (module / ".terraform.lock.hcl").read_text()
    assert 'provider "registry.terraform.io/hashicorp/' in lock
    assert "h1:" in lock
    versions = (module / "versions.tf").read_text()
    assert re.search(r'required_version\s*=\s*">= 1\.\d+"', versions), versions
    assert re.search(r'\n\s+version\s*=\s*"~> \d+\.\d+"', versions), versions
    assert re.search(r'version\s*=\s*"\d+\.\d+\.\d+"', lock), lock
    main = (module / "main.tf").read_text()
    assert f'managed-by = "{cfg.MANAGED_BY}"' in main
    assert "expires-at = var.expires_at" in main
    outputs = set(
        re.findall(r'^output "([a-z_]+)"', (module / "outputs.tf").read_text(), re.M)
    )
    wanted = {"kubeconfig", "needs_device_plugin"} if kind == "k8s" else {"host"}
    assert wanted <= outputs


@pytest.mark.parametrize("cloud", cfg.CLOUDS)
def test_vms_power_themselves_off(cloud: str) -> None:
    """The VM's own timer is the first safety net behind `down`."""
    main = (TERRAFORM / cloud / "vm" / "main.tf").read_text()
    assert "ttl_minutes" in main
    assert "gpu-vm.yaml.tftpl" in main
    template = (TERRAFORM / "cloud-init" / "gpu-vm.yaml.tftpl").read_text()
    assert "shutdown" in template or "poweroff" in template


def test_ssh_and_api_access_is_limited_to_given_addresses() -> None:
    for cloud in cfg.CLOUDS:
        text = (TERRAFORM / cloud / "vm" / "main.tf").read_text()
        assert "allowed_ssh_cidrs" in text
        k8s = (TERRAFORM / cloud / "k8s" / "main.tf").read_text()
        assert "allowed_api_cidrs" in k8s
        for module in ("vm", "k8s"):
            body = (TERRAFORM / cloud / module / "variables.tf").read_text()
            assert '"0.0.0.0/0"' not in body


def test_prepare_copies_the_modules_and_writes_private_variables(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, provider="gcp")
    directory = terraform.prepare(config, "ssh-ed25519 AAAA test")
    assert (directory / "main.tf").is_file()
    assert (directory.parents[1] / "cloud-init" / "gpu-vm.yaml.tftpl").is_file()
    assert not list(directory.parents[1].rglob(".terraform/*"))
    written = directory / "terraform.tfvars.json"
    assert written.stat().st_mode & 0o777 == 0o600
    values = json.loads(written.read_text())
    assert values["project"] == "medbench-test"
    assert values["zone"] == "us-central1-a"
    assert values["expires_at"] == str(1_800_000_000 + 240 * 60)


def rendered_job(tmp_path: Path, **values: object) -> str:
    config = make_config(tmp_path, provider="kubernetes", mode="k8s", **values)
    return kube.render_job(config)


def test_job_fills_every_placeholder(tmp_path: Path) -> None:
    text = rendered_job(tmp_path, tasks="diagnosis ocr-history", count=3)
    assert "@{" not in text.split("apiVersion:", 1)[1]
    assert 'name: "medbench-20261020120000-abcd"' in text
    assert 'medbench/run-id: "20261020120000-abcd"' in text
    assert 'value: "diagnosis ocr-history"' in text
    assert 'nvidia.com/gpu: "1"' in text
    assert "activeDeadlineSeconds: 14400" in text
    assert 'memory: "24Gi"' in text


def test_job_leaves_out_empty_tasks_and_a_zero_gpu_limit(tmp_path: Path) -> None:
    """Empty TASKS would replace the script's default list; a GPU limit of
    0 is how a CPU cluster (kind, the CI smoke test) runs the Job."""
    text = rendered_job(tmp_path, gpu_count=0)
    assert "name: TASKS" not in text
    assert (
        "nvidia.com/gpu" not in text.split("limits:", 1)[1].split("volumeMounts", 1)[0]
    )
    assert "ephemeral-storage" in text.split("limits:", 1)[1]


def test_job_values_are_quoted(tmp_path: Path) -> None:
    text = rendered_job(tmp_path, commit='abc"; rm -rf /')
    assert r'value: "abc\"; rm -rf /"' in text


def test_job_rejects_a_template_with_unknown_placeholders(tmp_path: Path) -> None:
    template = tmp_path / "job.yaml"
    template.write_text('name: "@{name}"\nimage: "@{image}"\n')
    config = make_config(tmp_path, provider="kubernetes", mode="k8s")
    with pytest.raises(KeyError, match="image"):
        kube.render_job(config, template)


def test_every_model_has_job_resources() -> None:
    assert set(kube.RESOURCES) == set(cfg.MODELS)


def test_job_pod_requests_can_be_set(tmp_path: Path) -> None:
    text = rendered_job(
        tmp_path, model="unimedvl", pod_cpu="500m", pod_memory="4Gi", pod_disk="10Gi"
    )
    assert 'cpu: "500m"' in text
    assert 'memory: "4Gi"' in text
    assert text.count('ephemeral-storage: "10Gi"') == 2
    with pytest.raises(ValueError, match="--pod-memory"):
        rendered_job(tmp_path, pod_memory="4 GB")


def test_kind_example_fits_a_ci_runner() -> None:
    """The kind job runs on a 4 vCPU / 16 GB runner, where the control plane
    requests about 1 CPU. UniMedVL's own requests (7 CPUs, 48Gi) left the
    pod Pending until the job was cancelled."""
    script = (ROOT / "examples" / "cloud_gpu_kind.sh").read_text()
    options = dict(re.findall(r"--pod-(cpu|memory|disk) (\S+)", script))
    assert float(options["cpu"]) <= 2
    assert options["memory"].endswith("Gi")
    assert int(options["memory"][:-2]) <= 8
    assert options["disk"].endswith("Gi")
    assert int(options["disk"][:-2]) <= 20
    cpu, memory = kube.RESOURCES["unimedvl"]
    assert float(cpu) > 4 or int(memory[:-2]) > 16


class FakeKubectl(kube.Kubectl):
    """Answers like a cluster whose only pod no node can fit."""

    def get(self, *args: str, timeout: float | None = 120) -> str:
        if args[:2] == ("get", "pods"):
            return "medbench-x-abc"
        if "jsonpath={.status.phase}" in args:
            return "Pending"
        pod = {
            "status": {
                "phase": "Pending",
                "conditions": [
                    {
                        "type": "PodScheduled",
                        "status": "False",
                        "reason": "Unschedulable",
                        "message": "0/1 nodes are available: 1 Insufficient cpu.",
                    }
                ],
            }
        }
        return json.dumps(pod)

    def raw(
        self, *args: str, stdin: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            list(args), 0, b"Events: FailedScheduling\n", b""
        )


def test_waiting_for_the_pod_says_why_it_is_pending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = tmp_path / "reports"
    with pytest.raises(TimeoutError, match="Insufficient cpu"):
        kube.wait_for_pod(FakeKubectl(), "medbench-x", reports, timeout=0)
    assert (
        "medbench-x-abc is Pending: Unschedulable: 0/1 nodes" in capsys.readouterr().err
    )
    assert (reports / "pod-describe.txt").read_text() == "Events: FailedScheduling\n"


def test_pod_waiting_reason_reports_image_pull_errors() -> None:
    class Pulling(FakeKubectl):
        def get(self, *args: str, timeout: float | None = 120) -> str:
            waiting = {"reason": "ErrImagePull", "message": "not found"}
            pod = {"status": {"containerStatuses": [{"state": {"waiting": waiting}}]}}
            return json.dumps(pod)

    assert kube.pod_waiting_reason(Pulling(), "p") == "ErrImagePull: not found"


@pytest.mark.skipif(shutil.which("kubeconform") is None, reason="needs kubeconform")
@pytest.mark.parametrize("gpu_count", [0, 1])
def test_rendered_job_is_valid_kubernetes(tmp_path: Path, gpu_count: int) -> None:
    manifest = tmp_path / "job.yaml"
    manifest.write_text(
        rendered_job(tmp_path, gpu_count=gpu_count, extra_env={"PORT": "8080"})
    )
    plugin = ROOT / "infra" / "kubernetes" / "nvidia-device-plugin.yaml"
    result = subprocess.run(
        ["kubeconform", "-strict", "-summary", str(manifest), str(plugin)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kubectl_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kube, "require_tool", lambda name: f"/usr/bin/{name}")
    local = kube.Kubectl(kubeconfig=tmp_path / "kc", context="ctx", namespace="ns")
    assert local.argv(["get", "pods"]) == [
        "/usr/bin/kubectl",
        "--namespace",
        "ns",
        "--context",
        "ctx",
        "--kubeconfig",
        str(tmp_path / "kc"),
        "get",
        "pods",
    ]


def test_kubectl_over_ssh_uses_k3s_and_sudo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(flows.remote, "require_tool", lambda name: f"/usr/bin/{name}")
    config = make_config(
        tmp_path, provider="ssh", mode="k8s", ssh_host="gpu.example", ssh_user="ci"
    )
    kubectl = flows.kubectl_for(config, {})
    argv = kubectl.argv(["get", "nodes"])
    assert argv[0] == "/usr/bin/ssh"
    assert "ci@gpu.example" in argv
    assert argv[-1] == "sudo -n k3s kubectl --namespace default get nodes"
