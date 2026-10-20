"""Create and delete the cloud GPU machines with Terraform or OpenTofu."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu.shell import log, output, require_tool, run


def module_kind(config: cfg.RunConfig) -> str:
    """Return ``vm`` or ``k8s``: runner mode starts a VM."""
    return "k8s" if config.mode == "k8s" else "vm"


def module_dir(config: cfg.RunConfig) -> Path:
    """Return the run's copy of the Terraform module."""
    return config.work / "terraform" / config.provider / module_kind(config)


def tfvars(config: cfg.RunConfig, ssh_public_key: str = "") -> dict[str, Any]:
    """Return the variables of the module for this run.

    The names match infra/terraform/<cloud>/<vm|k8s>/variables.tf;
    tests/test_cloud_gpu_orchestrator.py checks that every key is declared.
    """
    kind = module_kind(config)
    values: dict[str, Any] = {
        "name": config.name,
        "run_id": config.run_id,
        "expires_at": str(config.expires_at),
        "instance_type": config.instance_type,
        "spot": config.spot,
        "disk_gb": config.disk_gb,
    }
    cidrs = config.allowed_cidrs
    if kind == "vm":
        values["ttl_minutes"] = config.ttl_minutes
        values["ssh_public_key"] = ssh_public_key
        values["allowed_ssh_cidrs"] = cidrs
    else:
        values["node_count"] = config.node_count
        values["allowed_api_cidrs"] = cidrs
    if config.provider == "aws":
        values["region"] = config.region
    elif config.provider == "gcp":
        values.update(project=config.gcp_project, region=config.region)
        values["zone"] = config.zone
        if config.accelerator_type or kind == "k8s":
            values["accelerator_type"] = config.accelerator_type
    elif config.provider == "azure":
        values["location"] = config.region
        values["subscription_id"] = config.azure_subscription
    return values


def prepare(config: cfg.RunConfig, ssh_public_key: str = "") -> Path:
    """Copy the modules into the working directory and write the variables.

    The whole infra/terraform tree is copied, so the modules still find
    ../../cloud-init and the committed .terraform.lock.hcl pins the providers.
    """
    root = config.work / "terraform"
    if not root.exists():
        shutil.copytree(
            cfg.INFRA / "terraform",
            root,
            ignore=shutil.ignore_patterns(".terraform", "*.tfstate*", "*.tfplan"),
        )
    directory = module_dir(config)
    variables = directory / "terraform.tfvars.json"
    variables.write_text(json.dumps(tfvars(config, ssh_public_key), indent=2) + "\n")
    variables.chmod(0o600)
    return directory


def _tf(config: cfg.RunConfig, *args: str, capture: bool = False) -> str:
    """Run the tool in the module; with ``capture`` return its stdout only."""
    argv = [require_tool(config.tool), *args]
    env = {"TF_IN_AUTOMATION": "1", "TF_INPUT": "0"}
    if capture:
        return output(argv, cwd=module_dir(config), env=env)
    run(argv, cwd=module_dir(config), env=env)
    return ""


def init(config: cfg.RunConfig) -> None:
    """``terraform init`` with the committed provider lock file.

    The modules declare no backend, and without one Terraform ignores
    -backend-config and keeps the state local; so a remote state also gets
    an override file that declares the backend.
    """
    args = ["init", "-input=false", "-lockfile=readonly", "-no-color"]
    if config.backend_config:
        backend = config.backend_type or cfg.TERRAFORM_BACKENDS[config.provider]
        (module_dir(config) / "backend_override.tf").write_text(
            f'terraform {{\n  backend "{backend}" {{}}\n}}\n', "utf-8"
        )
        args.append(f"-backend-config={config.backend_config}")
    _tf(config, *args)


def plan(config: cfg.RunConfig) -> None:
    """Show what ``apply`` would create, without creating anything."""
    init(config)
    _tf(config, "plan", "-input=false", "-no-color")


def apply(config: cfg.RunConfig) -> dict[str, Any]:
    """Create the resources and return the module outputs."""
    init(config)
    log(f"creating {config.provider} {module_kind(config)} {config.name}")
    _tf(config, "apply", "-input=false", "-auto-approve", "-no-color")
    return outputs(config)


def outputs(config: cfg.RunConfig) -> dict[str, Any]:
    """Return ``terraform output -json`` as plain values."""
    raw = json.loads(_tf(config, "output", "-json", "-no-color", capture=True))
    return {key: item.get("value") for key, item in raw.items()}


def destroy(config: cfg.RunConfig) -> None:
    """Delete everything the run created; safe to repeat."""
    directory = module_dir(config)
    if not (directory / "terraform.tfvars.json").exists():
        log(f"nothing to destroy in {directory}")
        return
    init(config)
    log(f"deleting {config.provider} {module_kind(config)} {config.name}")
    _tf(config, "destroy", "-input=false", "-auto-approve", "-no-color")
