"""Run settings, machine defaults per model and the state file of a run."""

from __future__ import annotations

import dataclasses
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
INFRA = REPO_ROOT / "infra"
MANAGED_BY = "medbench-cloud-gpu"

PROVIDERS = ("aws", "azure", "gcp", "hetzner", "ssh", "kubernetes", "local")
CLOUDS = ("aws", "azure", "gcp")
MODES = ("vm", "k8s", "runner")
MODELS = ("medgemma-27b-it", "medgemma-4b-it", "unimedvl")

# Which modes each provider supports. A Hetzner dedicated GPU server becomes a
# one-node k3s cluster in k8s mode (scripts/gpu-host-bootstrap.sh k3s).
PROVIDER_MODES: dict[str, tuple[str, ...]] = {
    "aws": ("vm", "k8s", "runner"),
    "azure": ("vm", "k8s", "runner"),
    "gcp": ("vm", "k8s", "runner"),
    "hetzner": ("vm", "k8s", "runner"),
    "ssh": ("vm", "k8s", "runner"),
    "kubernetes": ("k8s",),
    "local": ("vm",),
}

# The cheapest machine per provider and mode that fits each model, from
# docs/case-studies/issue-23/data (prices read on 2026-10-19). Memory needs
# come from docs/case-studies/issue-1/ci-and-gpu-runner.md: MedGemma 4B Q4
# fits 16 GB, 27B Q4_K_M about 18 GB (so 24 GB), UniMedVL bf16 30 to 40 GB
# (48 GB or more, with host RAM for the 29 GB of weights).
# For GKE the accelerator is a separate setting: (machine, accelerator).
DEFAULT_MACHINES: dict[str, dict[str, dict[str, tuple[str, str]]]] = {
    "medgemma-4b-it": {
        "aws": {"vm": ("g6.xlarge", ""), "k8s": ("g6.xlarge", "")},
        "gcp": {"vm": ("g2-standard-4", ""), "k8s": ("g2-standard-4", "nvidia-l4")},
        "azure": {
            "vm": ("Standard_NC4as_T4_v3", ""),
            "k8s": ("Standard_NC4as_T4_v3", ""),
        },
        "hetzner": {"vm": ("GEX45", ""), "k8s": ("GEX45", "")},
    },
    "medgemma-27b-it": {
        "aws": {"vm": ("g6.2xlarge", ""), "k8s": ("g6.2xlarge", "")},
        "gcp": {"vm": ("g2-standard-8", ""), "k8s": ("g2-standard-8", "nvidia-l4")},
        "azure": {
            "vm": ("Standard_NV36ads_A10_v5", ""),
            "k8s": ("Standard_NC24ads_A100_v4", ""),
        },
        "hetzner": {"vm": ("GEX45", ""), "k8s": ("GEX45", "")},
    },
    "unimedvl": {
        "aws": {"vm": ("g6e.2xlarge", ""), "k8s": ("g6e.2xlarge", "")},
        "gcp": {
            "vm": ("a2-ultragpu-1g", ""),
            "k8s": ("a2-ultragpu-1g", "nvidia-a100-80gb"),
        },
        "azure": {
            "vm": ("Standard_NC24ads_A100_v4", ""),
            "k8s": ("Standard_NC24ads_A100_v4", ""),
        },
        "hetzner": {"vm": ("GEX63", ""), "k8s": ("GEX63", "")},
    },
}

DISK_GB = {"medgemma-4b-it": 100, "medgemma-27b-it": 150, "unimedvl": 200}

# On-demand price per hour of the default machines (USD for the clouds, EUR
# without VAT for Hetzner), only for the estimate `plan` prints. Sources:
# aws-gpu-prices-us-east-1.json, gcp-gpu-prices-us-central1.txt,
# azure-retail-prices-eastus*.json and hetzner-gpu-prices.json in
# docs/case-studies/issue-23/data. Hetzner also charges a setup fee per order.
HOURLY_PRICE: dict[str, float] = {
    "g6.xlarge": 0.8048,
    "g6.2xlarge": 0.9776,
    "g6e.2xlarge": 2.242,
    "g2-standard-4": 0.7068,
    "g2-standard-8": 0.8536,
    "a2-ultragpu-1g": 5.069,
    "Standard_NC4as_T4_v3": 0.526,
    "Standard_NV36ads_A10_v5": 3.20,
    "Standard_NC24ads_A100_v4": 3.673,
    "GEX45": 0.3429,
    "GEX63": 1.6009,
    "GEX131": 2.4022,
}
HETZNER_SETUP_FEE = {"GEX45": 209.0, "GEX63": 999.0, "GEX131": 1499.0}

DEFAULT_REGIONS = {
    "aws": "us-east-1",
    "gcp": "us-central1",
    "azure": "eastus",
    "hetzner": "FSN1",
}

# Hard limit for any temporary machine: the VMs power themselves off after
# ttl_minutes, and the reaper deletes anything still there after expires_at.
MAX_TTL_MINUTES = 24 * 60

# The Terraform backend used for --backend-config unless --backend-type names
# another one (for example http, GitLab's managed Terraform state).
TERRAFORM_BACKENDS = {"aws": "s3", "azure": "azurerm", "gcp": "gcs"}
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,30}$")


def new_run_id(now: float | None = None) -> str:
    """Return a run id that sorts by time and is valid in every cloud name."""
    stamp = datetime.fromtimestamp(now or time.time(), tz=UTC)
    return f"{stamp:%Y%m%d%H%M%S}-{secrets.token_hex(2)}"


def check_run_id(run_id: str) -> str:
    """Return ``run_id`` if it is a valid name part, else raise ValueError."""
    if not RUN_ID_PATTERN.match(run_id):
        message = (
            f"run id {run_id!r} must be 3 to 31 lowercase letters, digits or "
            "hyphens, starting with a letter or digit"
        )
        raise ValueError(message)
    return run_id


@dataclass
class RunConfig:
    """Everything one run needs; stored in ``<workdir>/state.json``."""

    provider: str
    mode: str
    model: str
    run_id: str
    workdir: str
    created_at: int
    ttl_minutes: int = 240
    region: str = ""
    zone: str = ""
    instance_type: str = ""
    accelerator_type: str = ""
    gpu_count: int = 1
    node_count: int = 1
    spot: bool = False
    disk_gb: int = 0
    count: int = 20
    tasks: str = ""
    time_budget: int = 5400
    steps: str = "download run"
    require_gpu: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)
    commit: str = ""
    tool: str = "terraform"
    allowed_cidrs: list[str] = field(default_factory=list)
    gcp_project: str = ""
    azure_subscription: str = ""
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_port: int = 22
    ssh_key: str = ""
    kube_context: str = ""
    namespace: str = "default"
    # Job pod requests; empty means the model's default (scripts/cloud_gpu/kube.py).
    pod_cpu: str = ""
    pod_memory: str = ""
    pod_disk: str = ""
    hetzner_location: str = ""
    hetzner_order_timeout: int = 180
    github_repository: str = ""
    runner_labels: list[str] = field(default_factory=list)
    backend_config: str = ""
    backend_type: str = ""

    @property
    def name(self) -> str:
        """Prefix of every cloud resource of this run."""
        return f"medbench-{self.run_id}"

    @property
    def expires_at(self) -> int:
        """Unix time after which the reaper deletes what is left."""
        return self.created_at + self.ttl_minutes * 60

    @property
    def work(self) -> Path:
        """The run's working directory as a path."""
        return Path(self.workdir)

    def validate(self) -> None:
        """Raise ValueError for a combination that cannot work."""
        if self.provider not in PROVIDERS:
            message = f"unknown provider {self.provider!r}"
            raise ValueError(message)
        if self.mode not in PROVIDER_MODES[self.provider]:
            message = (
                f"provider {self.provider} supports the modes "
                f"{', '.join(PROVIDER_MODES[self.provider])}, not {self.mode}"
            )
            raise ValueError(message)
        if self.model not in MODELS:
            message = f"unknown model {self.model!r}"
            raise ValueError(message)
        check_run_id(self.run_id)
        if not 10 <= self.ttl_minutes <= MAX_TTL_MINUTES:
            message = f"ttl must be 10 to {MAX_TTL_MINUTES} minutes"
            raise ValueError(message)
        if self.provider == "gcp" and not self.gcp_project:
            message = "--gcp-project (or CLOUD_GPU_GCP_PROJECT) is required"
            raise ValueError(message)
        if self.provider == "ssh" and not self.ssh_host:
            message = "--ssh-host is required for the ssh provider"
            raise ValueError(message)
        if self.mode == "runner" and not self.github_repository:
            message = "--github-repository (or GITHUB_REPOSITORY) is required"
            raise ValueError(message)
        if self.backend_type and not re.fullmatch(r"[a-z][a-z0-9]*", self.backend_type):
            message = f"invalid Terraform backend type {self.backend_type!r}"
            raise ValueError(message)
        for option, value in (
            ("--pod-cpu", self.pod_cpu),
            ("--pod-memory", self.pod_memory),
            ("--pod-disk", self.pod_disk),
        ):
            if value and not re.fullmatch(r"[0-9]+(\.[0-9]+)?(m|[KMGT]i?)?", value):
                message = f"{option} {value!r} is not a Kubernetes quantity (e.g. 2, 500m, 8Gi)"
                raise ValueError(message)
        for key in self.extra_env:
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
                message = f"invalid environment variable name {key!r}"
                raise ValueError(message)

    def fill_defaults(self) -> None:
        """Pick region, machine and disk for the model where none was given."""
        if not self.region:
            self.region = DEFAULT_REGIONS.get(self.provider, "")
        if self.provider == "gcp" and not self.zone:
            self.zone = f"{self.region}-a"
        machines = DEFAULT_MACHINES[self.model].get(self.provider)
        if machines and self.mode in ("vm", "k8s", "runner"):
            machine, accelerator = machines["k8s" if self.mode == "k8s" else "vm"]
            if not self.instance_type:
                self.instance_type = machine
                if not self.accelerator_type:
                    self.accelerator_type = accelerator
        if not self.disk_gb:
            self.disk_gb = DISK_GB[self.model]
        if self.mode == "runner" and not self.runner_labels:
            self.runner_labels = ["self-hosted", "linux", "x64", self.name]

    def model_env(self) -> dict[str, str]:
        """Environment for scripts/gpu-host-run.sh on the GPU host."""
        env = {
            "GPU_MODEL": self.model,
            "COUNT": str(self.count),
            "TIME_BUDGET": str(self.time_budget),
            "STEPS": self.steps,
            "REQUIRE_GPU": "1" if self.require_gpu else "0",
            "MEDBENCH_COMMIT": self.commit,
        }
        if self.tasks:
            env["TASKS"] = self.tasks
        env.update(self.extra_env)
        return env

    def estimate(self) -> str:
        """One line with the price per hour of the chosen machine, if known."""
        price = HOURLY_PRICE.get(self.instance_type)
        if price is None:
            return f"{self.instance_type or 'no machine'}: no price on record"
        unit = "EUR" if self.provider == "hetzner" else "USD"
        hours = self.ttl_minutes / 60
        text = (
            f"{self.instance_type}: about {price:.2f} {unit}/hour on demand, "
            f"at most {price * hours * max(1, self.node_count):.2f} {unit} "
            f"for the {self.ttl_minutes}-minute time to live"
        )
        fee = HETZNER_SETUP_FEE.get(self.instance_type)
        if self.provider == "hetzner" and fee:
            text += f", plus a one-time setup fee of {fee:.0f} {unit} per order"
        return text


def save_state(config: RunConfig, extra: dict[str, Any] | None = None) -> Path:
    """Write the config and any outputs to ``<workdir>/state.json``."""
    path = config.work / "state.json"
    data = load_raw_state(config.work) if path.exists() else {}
    data["config"] = dataclasses.asdict(config)
    data.setdefault("outputs", {})
    if extra:
        data["outputs"].update(extra)
    config.work.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", "utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)
    return path


def load_raw_state(workdir: Path) -> dict[str, Any]:
    """Return the parsed state file of a working directory."""
    data: dict[str, Any] = json.loads((workdir / "state.json").read_text("utf-8"))
    return data


def load_state(workdir: Path) -> tuple[RunConfig, dict[str, Any]]:
    """Return the config and outputs stored in a working directory."""
    data = load_raw_state(workdir)
    known = {f.name for f in dataclasses.fields(RunConfig)}
    config = RunConfig(**{k: v for k, v in data["config"].items() if k in known})
    outputs: dict[str, Any] = data.get("outputs", {})
    return config, outputs
