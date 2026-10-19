"""Delete what failed runs left behind, found by tags and deadlines.

Every resource of a run carries managed-by=medbench-cloud-gpu and
expires-at=<unix time> (tags on AWS and Azure, labels or descriptions on
Google Cloud, the server name on Hetzner). `reap` deletes those whose
deadline has passed. It is the safety net behind `down` and the VMs' own
power-off timers, and runs on a schedule (.github/workflows/cloud-gpu-tests.yml,
ci/gitlab/cloud-gpu.gitlab-ci.yml). It uses the clouds' own command line
tools (aws, az, gcloud), which the CI images already have. With
``dry_run`` it only prints what it would delete.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu import github, hetzner
from scripts.cloud_gpu.shell import CommandError, log, output, require_tool, run

EXPIRES = re.compile(r"expires-at=(\d+)")


class Reaper:
    """Find and delete expired resources of one provider."""

    def __init__(self, *, dry_run: bool, now: float | None = None) -> None:
        """``now`` is for tests; ``dry_run`` only reports."""
        self.dry_run = dry_run
        self.now = now if now is not None else time.time()
        self.deleted: list[str] = []
        self.failed: list[str] = []

    def expired(self, expires_at: object) -> bool:
        """Return True for a deadline in the past (False if unreadable)."""
        text = str(expires_at or "")
        return text.isdigit() and int(text) < self.now

    def act(self, what: str, action: Callable[..., object], *args: object) -> None:
        """Delete one resource with ``action(*args)``, or only report it."""
        if self.dry_run:
            log(f"would delete {what}")
            self.deleted.append(what)
            return
        log(f"deleting {what}")
        try:
            action(*args)
        except (CommandError, RuntimeError, OSError) as error:
            log(f"could not delete {what}: {error}")
            self.failed.append(what)
            return
        self.deleted.append(what)

    @staticmethod
    def json_of(argv: Sequence[str]) -> Any:
        """Run a cloud CLI and decode its JSON output."""
        text = output(argv).strip()
        return json.loads(text) if text else None

    # Azure: one resource group per run holds everything, including the
    # node resource group AKS creates.
    def azure(self) -> None:
        """Delete expired resource groups."""
        az = require_tool("az")
        groups = (
            self.json_of(
                [
                    az,
                    "group",
                    "list",
                    "--tag",
                    f"managed-by={cfg.MANAGED_BY}",
                    "--output",
                    "json",
                ]
            )
            or []
        )
        for group in groups:
            if self.expired((group.get("tags") or {}).get("expires-at")):
                name = group["name"]
                self.act(
                    f"Azure resource group {name}",
                    run,
                    [az, "group", "delete", "--name", name, "--yes"],
                )

    # AWS: instances and EKS clusters cost money; the VPC parts are free but
    # count against the per-region VPC quota, so they go too.
    def aws(self, region: str) -> None:
        """Delete expired instances, EKS clusters, VPCs, key pairs and roles."""
        aws = require_tool("aws")
        base = [aws, "--region", region, "--output", "json"]
        tag_filter = f"Name=tag:managed-by,Values={cfg.MANAGED_BY}"

        def tag(tags: list[dict[str, str]] | dict[str, str] | None, key: str) -> str:
            if isinstance(tags, dict):
                return tags.get(key, "")
            return next((t["Value"] for t in tags or [] if t.get("Key") == key), "")

        reservations = (
            self.json_of(
                [
                    *base,
                    "ec2",
                    "describe-instances",
                    "--filters",
                    tag_filter,
                    "Name=instance-state-name,Values=pending,running,stopping,stopped",
                ]
            )
            or {}
        )
        for reservation in reservations.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                if self.expired(tag(instance.get("Tags"), "expires-at")):
                    instance_id = instance["InstanceId"]
                    self.act(
                        f"EC2 instance {instance_id}",
                        run,
                        [
                            *base,
                            "ec2",
                            "terminate-instances",
                            "--instance-ids",
                            instance_id,
                        ],
                    )
        clusters = (self.json_of([*base, "eks", "list-clusters"]) or {}).get(
            "clusters", []
        )
        for name in clusters:
            cluster = self.json_of([*base, "eks", "describe-cluster", "--name", name])[
                "cluster"
            ]
            tags = cluster.get("tags", {})
            if tags.get("managed-by") == cfg.MANAGED_BY and self.expired(
                tags.get("expires-at")
            ):
                self.act(f"EKS cluster {name}", self._delete_eks, base, name)
        vpcs = (
            self.json_of([*base, "ec2", "describe-vpcs", "--filters", tag_filter]) or {}
        )
        for vpc in vpcs.get("Vpcs", []):
            if self.expired(tag(vpc.get("Tags"), "expires-at")):
                vpc_id = vpc["VpcId"]
                self.act(f"VPC {vpc_id}", self._delete_vpc, base, vpc_id)
        pairs = (
            self.json_of([*base, "ec2", "describe-key-pairs", "--filters", tag_filter])
            or {}
        )
        for pair in pairs.get("KeyPairs", []):
            if self.expired(tag(pair.get("Tags"), "expires-at")):
                pair_id = pair["KeyPairId"]
                self.act(
                    f"key pair {pair['KeyName']}",
                    run,
                    [*base, "ec2", "delete-key-pair", "--key-pair-id", pair_id],
                )
        roles = (
            self.json_of([aws, "--output", "json", "iam", "list-roles"]) or {}
        ).get("Roles", [])
        for role in roles:
            name = role["RoleName"]
            if not name.startswith("medbench-"):
                continue
            tags = self.json_of(
                [aws, "--output", "json", "iam", "list-role-tags", "--role-name", name]
            )
            values = tags.get("Tags", []) if tags else []
            if tag(values, "managed-by") == cfg.MANAGED_BY and self.expired(
                tag(values, "expires-at")
            ):
                self.act(f"IAM role {name}", self._delete_role, aws, name)

    def _delete_eks(self, base: list[str], name: str) -> None:
        groups = (
            self.json_of([*base, "eks", "list-nodegroups", "--cluster-name", name])
            or {}
        )
        for group in groups.get("nodegroups", []):
            run(
                [
                    *base,
                    "eks",
                    "delete-nodegroup",
                    "--cluster-name",
                    name,
                    "--nodegroup-name",
                    group,
                ]
            )
            run(
                [
                    *base,
                    "eks",
                    "wait",
                    "nodegroup-deleted",
                    "--cluster-name",
                    name,
                    "--nodegroup-name",
                    group,
                ]
            )
        run([*base, "eks", "delete-cluster", "--name", name])
        run([*base, "eks", "wait", "cluster-deleted", "--name", name])

    def _delete_vpc(self, base: list[str], vpc_id: str) -> None:
        vpc_filter = f"Name=vpc-id,Values={vpc_id}"
        gateways = (
            self.json_of(
                [
                    *base,
                    "ec2",
                    "describe-internet-gateways",
                    "--filters",
                    f"Name=attachment.vpc-id,Values={vpc_id}",
                ]
            )
            or {}
        )
        for gateway in gateways.get("InternetGateways", []):
            gateway_id = gateway["InternetGatewayId"]
            run(
                [
                    *base,
                    "ec2",
                    "detach-internet-gateway",
                    "--internet-gateway-id",
                    gateway_id,
                    "--vpc-id",
                    vpc_id,
                ]
            )
            run(
                [
                    *base,
                    "ec2",
                    "delete-internet-gateway",
                    "--internet-gateway-id",
                    gateway_id,
                ]
            )
        subnets = (
            self.json_of([*base, "ec2", "describe-subnets", "--filters", vpc_filter])
            or {}
        )
        for subnet in subnets.get("Subnets", []):
            run([*base, "ec2", "delete-subnet", "--subnet-id", subnet["SubnetId"]])
        tables = (
            self.json_of(
                [*base, "ec2", "describe-route-tables", "--filters", vpc_filter]
            )
            or {}
        )
        for table in tables.get("RouteTables", []):
            if not any(a.get("Main") for a in table.get("Associations", [])):
                run(
                    [
                        *base,
                        "ec2",
                        "delete-route-table",
                        "--route-table-id",
                        table["RouteTableId"],
                    ]
                )
        groups = (
            self.json_of(
                [*base, "ec2", "describe-security-groups", "--filters", vpc_filter]
            )
            or {}
        )
        for group in groups.get("SecurityGroups", []):
            if group.get("GroupName") != "default":
                run(
                    [
                        *base,
                        "ec2",
                        "delete-security-group",
                        "--group-id",
                        group["GroupId"],
                    ]
                )
        run([*base, "ec2", "delete-vpc", "--vpc-id", vpc_id])

    @staticmethod
    def _delete_role(aws: str, name: str) -> None:
        attached = (
            Reaper.json_of(
                [
                    aws,
                    "--output",
                    "json",
                    "iam",
                    "list-attached-role-policies",
                    "--role-name",
                    name,
                ]
            )
            or {}
        )
        for policy in attached.get("AttachedPolicies", []):
            run(
                [
                    aws,
                    "iam",
                    "detach-role-policy",
                    "--role-name",
                    name,
                    "--policy-arn",
                    policy["PolicyArn"],
                ]
            )
        run([aws, "iam", "delete-role", "--role-name", name])

    # Google Cloud: labels on instances and clusters; networks, subnets and
    # firewall rules have no labels, so their description holds the deadline.
    def gcp(self, project: str) -> None:
        """Delete expired instances, GKE clusters, firewalls, subnets, networks."""
        gcloud = require_tool("gcloud")
        base = [gcloud, "--project", project, "--quiet"]
        label = f"labels.managed-by={cfg.MANAGED_BY}"
        for instance in (
            self.json_of(
                [
                    *base,
                    "compute",
                    "instances",
                    "list",
                    f"--filter={label}",
                    "--format=json",
                ]
            )
            or []
        ):
            if self.expired(instance.get("labels", {}).get("expires-at")):
                name, zone = instance["name"], instance["zone"].rsplit("/", 1)[-1]
                self.act(
                    f"GCE instance {name}",
                    run,
                    [*base, "compute", "instances", "delete", name, f"--zone={zone}"],
                )
        clusters = (
            self.json_of(
                [
                    *base,
                    "container",
                    "clusters",
                    "list",
                    f"--filter=resourceLabels.managed-by={cfg.MANAGED_BY}",
                    "--format=json",
                ]
            )
            or []
        )
        for cluster in clusters:
            if self.expired(cluster.get("resourceLabels", {}).get("expires-at")):
                name, location = cluster["name"], cluster["location"]
                self.act(
                    f"GKE cluster {name}",
                    run,
                    [
                        *base,
                        "container",
                        "clusters",
                        "delete",
                        name,
                        f"--location={location}",
                    ],
                )
        description = "--filter=description~medbench-cloud-gpu"
        for kind, extra in (
            ("firewall-rules", []),
            ("networks subnets", ["region"]),
            ("networks", []),
        ):
            command = [
                *base,
                "compute",
                *kind.split(),
                "list",
                description,
                "--format=json",
            ]
            for item in self.json_of(command) or []:
                match = EXPIRES.search(item.get("description", ""))
                if not match or not self.expired(match.group(1)):
                    continue
                name = item["name"]
                where = (
                    [f"--region={item['region'].rsplit('/', 1)[-1]}"] if extra else []
                )
                self.act(
                    f"GCE {kind} {name}",
                    run,
                    [*base, "compute", *kind.split(), "delete", name, *where],
                )

    def hetzner(self) -> None:
        """Cancel expired dedicated servers."""
        for server in hetzner.expired_servers(self.now):
            number = int(server["server_number"])
            self.act(
                f"Hetzner server {number} ({server.get('server_name')})",
                hetzner.cancel,
                number,
            )

    def github(self, repository: str) -> None:
        """Remove offline runner registrations the runs left behind."""
        for runner in github.stale_runners(repository):
            runner_id = int(runner["id"])
            self.act(
                f"GitHub runner {runner['name']}",
                github.delete_runner,
                repository,
                runner_id,
            )
