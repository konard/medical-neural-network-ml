"""Run the GPU tests as a Kubernetes Job (infra/kubernetes/medbench-gpu-job.yaml)."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu.remote import Ssh, extract_reports
from scripts.cloud_gpu.shell import (
    CommandError,
    log,
    output,
    require_tool,
    run,
    wait_until,
)

JOB_TEMPLATE = cfg.INFRA / "kubernetes" / "medbench-gpu-job.yaml"
DEVICE_PLUGIN = cfg.INFRA / "kubernetes" / "nvidia-device-plugin.yaml"
PLACEHOLDER = re.compile(r"@\{([a-z_]+)\}")
# CPU and memory requests per model: what the weights need in host RAM while
# loading, below the allocatable size of the default node types.
RESOURCES = {
    "medgemma-4b-it": ("3", "12Gi"),
    "medgemma-27b-it": ("6", "24Gi"),
    "unimedvl": ("7", "48Gi"),
}
COLLECT_SECONDS = 1800


def job_values(config: cfg.RunConfig) -> dict[str, str]:
    """Return the values for every placeholder of the Job template."""
    cpu, memory = RESOURCES[config.model]
    disk_gb = max(50, config.disk_gb - 30)  # leave room for the node itself
    return {
        "name": config.name,
        "namespace": config.namespace,
        "run_id": config.run_id,
        "deadline_seconds": str(config.ttl_minutes * 60),
        "gpu_model": config.model,
        "count": str(config.count),
        "tasks": config.tasks,
        "time_budget": str(config.time_budget),
        "steps": config.steps,
        "require_gpu": "1" if config.require_gpu else "0",
        "commit": config.commit,
        "collect_seconds": str(COLLECT_SECONDS),
        "cpu": cpu,
        "memory": memory,
        "disk": f"{disk_gb}Gi",
        "gpu_count": str(config.gpu_count),
    }


def render_job(config: cfg.RunConfig, template: Path = JOB_TEMPLATE) -> str:
    """Fill the Job template; JSON strings keep every value valid YAML."""
    values = job_values(config)
    text = template.read_text("utf-8")
    missing = set(PLACEHOLDER.findall(text)) - set(values)
    if missing:
        message = f"no value for {', '.join(sorted(missing))} in {template}"
        raise KeyError(message)

    def fill(match: re.Match[str]) -> str:
        value = values[match.group(1)]
        # Inside "..." the value is a quoted string already; json.dumps
        # escapes quotes and backslashes the same way YAML does.
        return json.dumps(value)[1:-1]

    rendered = PLACEHOLDER.sub(fill, text)
    if not config.tasks:
        # An empty TASKS would override the default list of the script.
        rendered = re.sub(r"\n\s+- name: TASKS\n\s+value: \"\"", "", rendered, count=1)
    if config.gpu_count == 0:
        rendered = re.sub(r"\n\s+nvidia\.com/gpu: \"0\"", "", rendered, count=1)
    if config.extra_env:
        # --env settings follow the last variable, so `job-manifest` output
        # applied by hand runs with them too.
        last = re.search(r"\n( +)- name: COLLECT_SECONDS\n +value: .*", rendered)
        if last is None:
            message = f"{template} has no COLLECT_SECONDS variable to add --env after"
            raise KeyError(message)
        indent = last.group(1)
        extra = "".join(
            f"\n{indent}- name: {key}\n{indent}  value: {json.dumps(value)}"
            for key, value in config.extra_env.items()
        )
        rendered = rendered[: last.end()] + extra + rendered[last.end() :]
    return rendered


@dataclass
class Kubectl:
    """kubectl, either local with a kubeconfig or on a host over SSH."""

    kubeconfig: Path | None = None
    context: str = ""
    namespace: str = "default"
    ssh: Ssh | None = None  # k3s on a GPU host: `sudo k3s kubectl` over SSH
    extra: list[str] = field(default_factory=list)

    def argv(self, args: Sequence[str]) -> list[str]:
        """Return the full command for kubectl ``args``."""
        options = ["--namespace", self.namespace]
        if self.context:
            options += ["--context", self.context]
        if self.ssh is not None:
            words = ["k3s", "kubectl", *options, *self.extra, *args]
            return [*self.ssh.argv(), self.ssh.sudo() + shlex.join(words)]
        if self.kubeconfig is not None:
            options += ["--kubeconfig", str(self.kubeconfig)]
        return [require_tool("kubectl"), *options, *self.extra, *args]

    def __call__(
        self,
        *args: str,
        stdin: bytes | None = None,
        check: bool = True,
        timeout: float | None = 300,
    ) -> subprocess.CompletedProcess[str]:
        """Run kubectl with its output on our output."""
        return run(self.argv(args), stdin=stdin, check=check, timeout=timeout)

    def get(self, *args: str, timeout: float | None = 120) -> str:
        """Run kubectl and return its stdout."""
        return output(self.argv(args), timeout=timeout)

    def raw(
        self, *args: str, stdin: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        """Run kubectl and return the raw result without raising."""
        return subprocess.run(
            self.argv(args),
            input=stdin,
            capture_output=True,
            check=False,
            timeout=1800,
        )


def gpu_nodes(kubectl: Kubectl) -> int:
    """Return how many nodes offer at least one nvidia.com/gpu."""
    nodes = json.loads(kubectl.get("get", "nodes", "--output", "json"))
    count = 0
    for node in nodes.get("items", []):
        allocatable = node.get("status", {}).get("allocatable", {})
        if int(allocatable.get("nvidia.com/gpu", "0") or 0) > 0:
            count += 1
    return count


def ensure_device_plugin(
    kubectl: Kubectl, *, needed: bool, timeout: float = 900
) -> None:
    """Install the NVIDIA device plugin if no node offers a GPU yet.

    GKE installs its own; EKS (AL2023 NVIDIA image) and AKS (with the GPU
    driver but no plugin) need infra/kubernetes/nvidia-device-plugin.yaml.
    """
    if gpu_nodes(kubectl) > 0:
        log("the cluster already offers nvidia.com/gpu")
        return
    if needed:
        log("applying the NVIDIA device plugin")
        plugin = Kubectl(
            kubectl.kubeconfig, kubectl.context, "kube-system", kubectl.ssh
        )
        plugin("apply", "-f", "-", stdin=DEVICE_PLUGIN.read_bytes())
    wait_until(
        "a node with nvidia.com/gpu", lambda: gpu_nodes(kubectl) > 0, timeout=timeout
    )


def job_pod(kubectl: Kubectl, job: str) -> str:
    """Return the name of the Job's pod, or "" before it exists."""
    names = kubectl.get(
        "get",
        "pods",
        "--selector",
        f"job-name={job}",
        "--output",
        "jsonpath={.items[*].metadata.name}",
    ).split()
    return names[0] if names else ""


def pod_phase(kubectl: Kubectl, pod: str) -> str:
    """Return Pending, Running, Succeeded, Failed or Unknown."""
    try:
        return kubectl.get("get", "pod", pod, "--output", "jsonpath={.status.phase}")
    except CommandError:
        return "Unknown"


def pod_exec(
    kubectl: Kubectl, pod: str, script: str, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run a shell script in the pod."""
    args = ["exec", "-i" if stdin is not None else "", pod, "--", "sh", "-c", script]
    return kubectl.raw(*[a for a in args if a], stdin=stdin)


def run_job(
    kubectl: Kubectl,
    config: cfg.RunConfig,
    archive: bytes,
    reports: Path,
    *,
    extra_env: Mapping[str, str] | None = None,
    start_timeout: float = 1800,
) -> int:
    """Apply the Job, feed it the repository, wait, collect; return its status.

    ``config.extra_env`` is part of the manifest; ``extra_env`` goes into
    /work/run.env and overrides it for this run only.
    """
    job = config.name
    kubectl("apply", "-f", "-", stdin=render_job(config).encode("utf-8"))

    pod = ""

    def running() -> bool:
        nonlocal pod
        pod = pod or job_pod(kubectl, job)
        if not pod:
            return False
        phase = pod_phase(kubectl, pod)
        if phase in {"Failed", "Succeeded"}:
            message = f"pod {pod} ended ({phase}) before the run started"
            raise RuntimeError(message)
        return phase == "Running"

    # A new GPU node can take a while: image pull, driver, device plugin.
    wait_until(f"pod of job {job}", running, timeout=start_timeout, interval=10)
    log(f"uploading the repository ({len(archive) // 1024} KiB) into {pod}")
    upload = pod_exec(
        kubectl, pod, "mkdir -p /work/repo && tar -xzf - -C /work/repo", archive
    )
    if upload.returncode != 0:
        raise CommandError(
            ["kubectl", "exec", pod, "tar"],
            upload.returncode,
            upload.stderr.decode("utf-8", "replace"),
        )
    env_lines = "".join(
        f"{key}={shlex.quote(value)}\n" for key, value in (extra_env or {}).items()
    )
    pod_exec(
        kubectl,
        pod,
        "cat > /work/run.env && touch /work/ready",
        env_lines.encode("utf-8"),
    )
    log(f"{pod} is running scripts/gpu-host-run.sh")

    def finished() -> bool:
        if pod_exec(kubectl, pod, "test -f /work/exit-code").returncode == 0:
            return True
        if pod_phase(kubectl, pod) in {"Failed", "Succeeded", "Unknown"}:
            message = f"pod {pod} ended without writing /work/exit-code"
            raise RuntimeError(message)
        return False

    try:
        wait_until(
            f"the run in {pod}", finished, timeout=config.ttl_minutes * 60, interval=30
        )
        status_text = pod_exec(kubectl, pod, "cat /work/exit-code").stdout
        status = int(status_text.decode("ascii", "replace").strip() or "1")
        data = pod_exec(kubectl, pod, "tar -cf - -C /work/reports .")
        if data.returncode == 0:
            names = extract_reports(data.stdout, reports)
            log(f"collected {len(names)} report files into {reports}")
        pod_exec(kubectl, pod, "touch /work/collected")
    finally:
        save_logs(kubectl, pod, reports / "pod.log")
    log(f"the run in {pod} exited with {status}")
    return status


def save_logs(kubectl: Kubectl, pod: str, path: Path) -> None:
    """Save the pod log next to the reports (best effort)."""
    if not pod:
        return
    result = kubectl.raw("logs", pod)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.stdout + result.stderr)


def delete_job(kubectl: Kubectl, config: cfg.RunConfig) -> None:
    """Delete the run's Job and pods; safe to repeat."""
    kubectl(
        "delete",
        "job",
        "--selector",
        f"medbench/run-id={config.run_id}",
        "--ignore-not-found",
        "--cascade=foreground",
        "--timeout=300s",
        check=False,
    )


def wait_for_api(kubectl: Kubectl, timeout: float = 600) -> None:
    """Wait until the cluster's API answers (new clusters need a moment)."""

    def ready() -> bool:
        return kubectl.raw("get", "nodes").returncode == 0

    wait_until("the Kubernetes API", ready, timeout=timeout, interval=10)
