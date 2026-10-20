"""The steps of a run per provider and mode: up, run, wait-runner, down."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu import github, hetzner, kube, remote, terraform
from scripts.cloud_gpu.shell import log, run

# Hetzner servers are ordered with nothing but a plain Ubuntu.
HETZNER_STEPS = {
    "vm": ["base", "driver"],
    "runner": ["base", "driver"],
    "k8s": ["base", "driver", "k3s"],
}


def detect_public_cidr() -> str:
    """Return this machine's public IPv4 address as a /32 for firewalls."""
    with urllib.request.urlopen(
        "https://checkip.amazonaws.com", timeout=30
    ) as response:
        address = response.read().decode("ascii").strip()
    parts = address.split(".")
    if len(parts) != 4 or not all(p.isdigit() and int(p) < 256 for p in parts):
        message = f"unexpected answer from checkip.amazonaws.com: {address!r}"
        raise RuntimeError(message)
    return f"{address}/32"


def ssh_for(config: cfg.RunConfig, outputs: dict[str, Any]) -> remote.Ssh:
    """Return the SSH destination of the run's GPU host."""
    key: Path | None
    if config.ssh_key:
        key = Path(config.ssh_key)
    elif config.provider == "ssh":
        key = None  # the user's own ssh setup (agent, ~/.ssh/config)
    else:
        key = config.work / "ssh" / "id_ed25519"
    return remote.Ssh(
        host=str(outputs.get("host") or config.ssh_host),
        user=str(outputs.get("ssh_user") or config.ssh_user or "root"),
        key=key,
        known_hosts=config.work / "ssh" / "known_hosts",
        port=config.ssh_port,
    )


def kubectl_for(config: cfg.RunConfig, outputs: dict[str, Any]) -> kube.Kubectl:
    """Return kubectl for the run's cluster."""
    if config.provider in {"hetzner", "ssh"}:
        return kube.Kubectl(namespace=config.namespace, ssh=ssh_for(config, outputs))
    kubeconfig = outputs.get("kubeconfig_path")
    if config.provider == "kubernetes":
        env_config = os.environ.get("KUBECONFIG", "")
        kubeconfig = kubeconfig or (
            env_config.split(os.pathsep)[0] if env_config else None
        )
    return kube.Kubectl(
        kubeconfig=Path(kubeconfig) if kubeconfig else None,
        context=config.kube_context,
        namespace=config.namespace,
    )


def up(config: cfg.RunConfig) -> dict[str, Any]:
    """Create the GPU host or cluster and get it ready; return the outputs."""
    outputs: dict[str, Any] = {"started_at": int(time.time())}
    cfg.save_state(config, outputs)
    if config.provider in cfg.CLOUDS:
        public_key = ""
        if config.mode != "k8s":
            _, public_key = remote.generate_key(config.work)
        terraform.prepare(config, public_key)
        outputs.update(terraform.apply(config))
        kubeconfig = outputs.pop("kubeconfig", None)
        if kubeconfig:
            path = config.work / "kubeconfig"
            path.write_text(str(kubeconfig), "utf-8")
            path.chmod(0o600)
            outputs["kubeconfig_path"] = str(path)
        cfg.save_state(config, outputs)
    elif config.provider == "hetzner":
        outputs.update(hetzner_up(config))
        cfg.save_state(config, outputs)
    if config.mode == "k8s":
        kubectl = kubectl_for(config, outputs)
        if config.provider == "hetzner":
            _prepare_host(config, outputs)
        kube.wait_for_api(kubectl)
        if config.gpu_count > 0:
            kube.ensure_device_plugin(
                kubectl, needed=bool(outputs.get("needs_device_plugin", True))
            )
        return outputs
    if config.provider in cfg.CLOUDS or config.provider == "hetzner":
        _prepare_host(config, outputs)
    elif config.mode == "runner":  # a host we keep, prepared with infra/ansible
        ssh = ssh_for(config, outputs)
        remote.wait_ready(ssh, cloud_init=False, timeout=120)
        remote.upload_repository(ssh, remote.repository_archive())
    if config.mode == "runner":
        outputs.update(start_runner(config, outputs))
        cfg.save_state(config, outputs)
    return outputs


def hetzner_up(config: cfg.RunConfig) -> dict[str, Any]:
    """Order a dedicated GPU server and return its address and ids."""
    _, public_key = remote.generate_key(config.work)
    fingerprint = hetzner.upload_key(config.name, public_key)
    result: dict[str, Any] = {"hetzner_key": fingerprint, "ssh_user": "root"}
    cfg.save_state(config, result)
    transaction = hetzner.order(config, fingerprint, test=False)
    result["hetzner_transaction"] = transaction["id"]
    cfg.save_state(config, result)
    ready = hetzner.wait_ready(
        str(transaction["id"]), timeout_minutes=config.hetzner_order_timeout
    )
    number = int(ready["server_number"])
    result.update(hetzner_server=number, host=ready["server_ip"])
    cfg.save_state(config, result)
    hetzner.rename(number, hetzner.server_name(config))
    return result


def _prepare_host(config: cfg.RunConfig, outputs: dict[str, Any]) -> None:
    """Wait for a fresh host, upload the repository, install what is missing."""
    ssh = ssh_for(config, outputs)
    remote.wait_ready(ssh, cloud_init=config.provider in cfg.CLOUDS)
    remote.upload_repository(ssh, remote.repository_archive())
    if config.provider == "hetzner":
        remote.bootstrap(ssh, HETZNER_STEPS[config.mode])
    elif config.require_gpu and config.mode != "k8s":
        remote.wait_gpu(ssh)


def start_runner(config: cfg.RunConfig, outputs: dict[str, Any]) -> dict[str, Any]:
    """Register an ephemeral runner and start it on the host."""
    runner_id, jit_config = github.generate_jit_config(
        config.github_repository, config.name, config.runner_labels
    )
    remote.start_runner(
        ssh_for(config, outputs), jit_config, poweroff=config.provider != "ssh"
    )
    return {"runner_id": runner_id}


def run_tests(config: cfg.RunConfig, outputs: dict[str, Any], reports: Path) -> int:
    """Run the GPU tests where the run's provider and mode say."""
    reports.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if config.mode == "runner":
        message = "runner mode runs the tests in the GitHub job; use wait-runner"
        raise ValueError(message)
    if config.provider == "local":
        env = {**config.model_env(), "OUT_DIR": str(reports.resolve())}
        status = run(
            ["bash", "scripts/gpu-host-run.sh"], cwd=cfg.REPO_ROOT, env=env, check=False
        ).returncode
    elif config.mode == "k8s":
        status = kube.run_job(
            kubectl_for(config, outputs),
            config,
            remote.repository_archive(),
            reports,
        )
    else:
        ssh = ssh_for(config, outputs)
        if config.provider == "ssh":
            remote.wait_ready(ssh, cloud_init=False, timeout=120)
            remote.upload_repository(ssh, remote.repository_archive())
        status = remote.run_tests(ssh, config.model_env())
        remote.fetch_reports(ssh, reports)
    write_summary(config, outputs, reports, status, time.time() - started)
    return status


def write_summary(
    config: cfg.RunConfig,
    outputs: dict[str, Any],
    reports: Path,
    status: int,
    seconds: float,
) -> Path:
    """Write cloud-gpu-run.json next to the reports."""
    summary = {
        "run_id": config.run_id,
        "provider": config.provider,
        "mode": config.mode,
        "model": config.model,
        "instance_type": config.instance_type,
        "accelerator_type": config.accelerator_type,
        "region": config.region,
        "spot": config.spot,
        "commit": config.commit,
        "exit_code": status,
        "seconds": round(seconds, 1),
        "host": outputs.get("host", ""),
        "estimate": config.estimate(),
    }
    path = reports / "cloud-gpu-run.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    return path


def wait_runner(config: cfg.RunConfig, outputs: dict[str, Any]) -> None:
    """Wait until the ephemeral runner has run its job."""
    runner_id = outputs.get("runner_id")
    if not runner_id:
        message = "no runner in the state file; run `up --mode runner` first"
        raise ValueError(message)
    github.wait_runner(
        config.github_repository, int(runner_id), job_timeout=config.ttl_minutes * 60
    )


def _cancel_hetzner(outputs: dict[str, Any]) -> None:
    """Cancel the run's server, also one delivered after `up` gave up."""
    number = outputs.get("hetzner_server")
    transaction = outputs.get("hetzner_transaction")
    if not number and transaction:
        number = hetzner.order_status(str(transaction)).get("server_number")
        if not number:
            log(
                f"order {transaction} has no server yet: the reaper cancels it a "
                f"day after the order if Hetzner delivers it later"
            )
            return
    if number:
        hetzner.cancel(int(number))


def down(config: cfg.RunConfig, outputs: dict[str, Any]) -> None:
    """Delete everything of the run; every step runs even if one fails."""
    errors: list[str] = []

    def attempt(what: str, action: Any) -> None:
        try:
            action()
        except Exception as error:
            log(f"{what} failed: {error}")
            errors.append(f"{what}: {error}")

    if config.mode == "runner" and outputs.get("runner_id"):
        attempt(
            "removing the runner",
            lambda: github.delete_runner(
                config.github_repository, int(outputs["runner_id"])
            ),
        )
    if config.mode == "k8s" and config.provider in {"kubernetes", "ssh"}:
        attempt(
            "deleting the Job",
            lambda: kube.delete_job(kubectl_for(config, outputs), config),
        )
    if config.provider in cfg.CLOUDS:
        attempt("terraform destroy", lambda: terraform.destroy(config))
    if config.provider == "hetzner":
        attempt("cancelling the server", lambda: _cancel_hetzner(outputs))
        if outputs.get("hetzner_key"):
            attempt(
                "deleting the SSH key",
                lambda: hetzner.delete_key(str(outputs["hetzner_key"])),
            )
    cfg.save_state(config, {"stopped_at": int(time.time())})
    if errors:
        message = "cleanup was incomplete:\n" + "\n".join(errors)
        raise RuntimeError(message)
    log(f"run {config.run_id} cleaned up")
