"""Command line of ``python -m scripts.cloud_gpu`` (see docs/cloud-gpu.md).

Typical use, from the repository root:

  # what would be created, and what it costs per hour
  python -m scripts.cloud_gpu plan --provider aws --model medgemma-27b-it
  # create, test, collect reports, delete (always, also on failure)
  python -m scripts.cloud_gpu all --provider gcp --mode k8s \\
      --model medgemma-4b-it --gcp-project my-project --yes
  # the same in steps, e.g. across CI jobs
  python -m scripts.cloud_gpu up --provider azure --run-id ci-42 --yes
  python -m scripts.cloud_gpu run --workdir .cache/cloud-gpu/ci-42
  python -m scripts.cloud_gpu down --workdir .cache/cloud-gpu/ci-42
  # delete what crashed runs left behind
  python -m scripts.cloud_gpu reap --provider aws --provider gcp --yes
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu import flows, hetzner, kube, reaper, remote, terraform
from scripts.cloud_gpu.shell import CommandError, log


def _key_value(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        message = f"expected KEY=VALUE, got {text!r}"
        raise argparse.ArgumentTypeError(message)
    return key, value


def _run_options() -> argparse.ArgumentParser:
    """Return the options that describe a run (parent parser)."""
    run_options = argparse.ArgumentParser(add_help=False)
    group = run_options.add_argument_group("run settings")
    group.add_argument("--provider", choices=cfg.PROVIDERS, default="aws")
    group.add_argument(
        "--mode",
        choices=cfg.MODES,
        default="vm",
        help="vm: SSH to a GPU VM; k8s: a Job on a GPU cluster; "
        "runner: an ephemeral GitHub Actions runner on a GPU VM",
    )
    group.add_argument("--model", choices=cfg.MODELS, default="medgemma-4b-it")
    group.add_argument(
        "--run-id", default="", help="default: time stamp and random suffix"
    )
    group.add_argument(
        "--ttl",
        type=int,
        default=240,
        dest="ttl_minutes",
        help="minutes until the host powers off and the reaper may delete it",
    )
    group.add_argument(
        "--region",
        default="",
        help="region (Azure: location; Hetzner: FSN1, NBG1, HEL1)",
    )
    group.add_argument(
        "--zone", default="", help="Google Cloud zone (default: <region>-a)"
    )
    group.add_argument(
        "--instance-type",
        default="",
        help="machine type (Hetzner: product, e.g. GEX45)",
    )
    group.add_argument(
        "--accelerator-type", default="", help="Google Cloud GPU type, e.g. nvidia-l4"
    )
    group.add_argument(
        "--gpu-count", type=int, default=1, help="GPUs per Job pod (0 for a CPU test)"
    )
    group.add_argument(
        "--pod-cpu", default="", help="Job pod CPU request (default: per model)"
    )
    group.add_argument(
        "--pod-memory", default="", help="Job pod memory request, e.g. 8Gi"
    )
    group.add_argument(
        "--pod-disk", default="", help="Job pod ephemeral storage, e.g. 20Gi"
    )
    group.add_argument("--node-count", type=int, default=1)
    group.add_argument(
        "--spot",
        action="store_true",
        help="spot/preemptible capacity (cheaper, may be interrupted)",
    )
    group.add_argument("--disk-gb", type=int, default=0)
    group.add_argument("--count", type=int, default=20, help="items per task")
    group.add_argument(
        "--tasks",
        default="",
        help="space-separated medbench tasks (default: all GPU tasks)",
    )
    group.add_argument(
        "--time-budget", type=int, default=5400, help="seconds for all tasks"
    )
    group.add_argument("--steps", default="download run")
    group.add_argument(
        "--no-require-gpu",
        dest="require_gpu",
        action="store_false",
        help="allow a host without an NVIDIA driver (for tests)",
    )
    group.add_argument(
        "--env",
        action="append",
        type=_key_value,
        default=[],
        metavar="KEY=VALUE",
        help="extra variable for scripts/gpu-host-run.sh (repeatable)",
    )
    group.add_argument("--tool", choices=("terraform", "tofu"), default="terraform")
    group.add_argument(
        "--allowed-cidr",
        action="append",
        default=[],
        dest="allowed_cidrs",
        help="addresses allowed to reach SSH or the Kubernetes API (default: this machine's /32)",
    )
    group.add_argument(
        "--gcp-project", default=os.environ.get("CLOUD_GPU_GCP_PROJECT", "")
    )
    group.add_argument(
        "--azure-subscription", default=os.environ.get("ARM_SUBSCRIPTION_ID", "")
    )
    group.add_argument("--ssh-host", default="", help="ssh provider: the GPU host")
    group.add_argument("--ssh-user", default="")
    group.add_argument("--ssh-port", type=int, default=22)
    group.add_argument(
        "--ssh-key",
        default="",
        help="private key (default: per-run key, or your ssh setup)",
    )
    group.add_argument("--kube-context", default="")
    group.add_argument("--namespace", default="default")
    group.add_argument("--hetzner-location", default="")
    group.add_argument("--hetzner-order-timeout", type=int, default=180, help="minutes")
    group.add_argument(
        "--github-repository", default=os.environ.get("GITHUB_REPOSITORY", "")
    )
    group.add_argument(
        "--runner-label",
        action="append",
        default=[],
        dest="runner_labels",
        help="runner mode: labels (default: self-hosted, linux, x64, medbench-<run id>)",
    )
    group.add_argument(
        "--backend-config",
        default="",
        help="Terraform backend config file for remote state",
    )
    group.add_argument(
        "--backend-type",
        default="",
        help="Terraform backend of --backend-config (default: s3, azurerm or gcs)",
    )
    group.add_argument(
        "--commit", default="", help="recorded in the reports (default: git HEAD)"
    )
    return run_options


def _common_options() -> argparse.ArgumentParser:
    """Return the options every stateful subcommand takes (parent parser)."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="state of the run (default: .cache/cloud-gpu/<run id>)",
    )
    common.add_argument(
        "--reports", type=Path, default=None, help="default: <workdir>/reports"
    )
    common.add_argument(
        "--yes", action="store_true", help="confirm creating paid resources"
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser with every subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.cloud_gpu",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_options, common = _run_options(), _common_options()
    parents = [run_options, common]
    commands.add_parser(
        "plan",
        parents=parents,
        help="show settings, price and (with --check) terraform plan",
    ).add_argument(
        "--check",
        action="store_true",
        help="also run terraform plan, or a Hetzner test order",
    )
    commands.add_parser("up", parents=parents, help="create the GPU host or cluster")
    commands.add_parser(
        "run", parents=[common], help="run the tests on a host made by up"
    )
    commands.add_parser(
        "wait-runner", parents=[common], help="runner mode: wait for the job"
    )
    commands.add_parser("down", parents=[common], help="delete everything of the run")
    every = commands.add_parser(
        "all", parents=parents, help="up, run (or wait-runner), down"
    )
    every.add_argument(
        "--keep",
        action="store_true",
        help="skip down (debugging; you pay until you delete)",
    )
    manifest = commands.add_parser(
        "job-manifest",
        parents=[run_options],
        help="print the Kubernetes Job for kubectl apply",
    )
    manifest.add_argument("--output", type=Path, default=None)
    reap = commands.add_parser("reap", help="delete expired resources of crashed runs")
    reap.add_argument(
        "--provider",
        action="append",
        choices=(*cfg.CLOUDS, "hetzner", "github"),
        dest="providers",
        required=True,
    )
    reap.add_argument(
        "--region",
        action="append",
        default=[],
        dest="regions",
        help="AWS regions (default: us-east-1)",
    )
    reap.add_argument(
        "--gcp-project", default=os.environ.get("CLOUD_GPU_GCP_PROJECT", "")
    )
    reap.add_argument(
        "--github-repository", default=os.environ.get("GITHUB_REPOSITORY", "")
    )
    reap.add_argument("--dry-run", action="store_true")
    reap.add_argument("--yes", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace, *, needs_cidr: bool) -> cfg.RunConfig:
    """Build and check the run config from the parsed options."""
    run_id = cfg.check_run_id(args.run_id) if args.run_id else cfg.new_run_id()
    workdir = (
        getattr(args, "workdir", None)
        or cfg.REPO_ROOT / ".cache" / "cloud-gpu" / run_id
    )
    names = {f.name for f in dataclasses.fields(cfg.RunConfig)}
    values = {k: v for k, v in vars(args).items() if k in names and k != "workdir"}
    values["extra_env"] = dict(args.env)
    config = cfg.RunConfig(
        **{
            **values,
            "run_id": run_id,
            "workdir": str(Path(workdir).resolve()),
            "created_at": int(time.time()),
        }
    )
    if config.backend_config:  # terraform init runs in the module's copy
        config.backend_config = str(Path(config.backend_config).resolve())
    if not config.ssh_user and config.provider == "ssh":
        config.ssh_user = os.environ.get("USER", "root")
    if not config.commit:
        try:
            config.commit = remote.current_commit()
        except (CommandError, RuntimeError):
            config.commit = "unknown"
    config.fill_defaults()
    config.validate()
    if needs_cidr and not config.allowed_cidrs and config.provider in cfg.CLOUDS:
        config.allowed_cidrs = [flows.detect_public_cidr()]
        log(f"allowing access from {config.allowed_cidrs[0]} only")
    return config


def describe(config: cfg.RunConfig) -> None:
    """Print the settings of a run."""
    shown = {
        "run_id": config.run_id,
        "provider": config.provider,
        "mode": config.mode,
        "model": config.model,
        "region": config.region,
        "zone": config.zone,
        "instance_type": config.instance_type,
        "accelerator_type": config.accelerator_type,
        "spot": config.spot,
        "disk_gb": config.disk_gb,
        "ttl_minutes": config.ttl_minutes,
        "expires_at": config.expires_at,
        "workdir": config.workdir,
    }
    print(json.dumps(shown, indent=2))
    print(f"estimate: {config.estimate()}")


def plan(config: cfg.RunConfig, *, check: bool) -> int:
    """Show what a run would create."""
    describe(config)
    if config.provider in cfg.CLOUDS:
        print("terraform variables:")
        print(json.dumps(terraform.tfvars(config, "<per-run key>"), indent=2))
        if check:
            terraform.prepare(config, "ssh-ed25519 AAAA-plan-only")
            terraform.plan(config)
    elif config.provider == "hetzner" and check:
        try:
            item = hetzner.product(config.instance_type)
        except hetzner.RobotError as error:
            if error.status != 404:
                raise
            names = [str(p.get("id")) for p in hetzner.gpu_products()]
            log(f"no product {config.instance_type}; GPU products: {names}")
            return 1
        print(
            json.dumps(
                {
                    k: item.get(k)
                    for k in ("id", "name", "price", "price_setup", "location", "dist")
                },
                indent=2,
            )
        )
        _, public_key = remote.generate_key(config.work)
        fingerprint = hetzner.upload_key(f"{config.name}-plan", public_key)
        try:
            print(json.dumps(hetzner.order(config, fingerprint, test=True), indent=2))
        finally:
            hetzner.delete_key(fingerprint)
    return 0


def reports_dir(args: argparse.Namespace, config: cfg.RunConfig) -> Path:
    """Return where the reports go."""
    return Path(args.reports) if args.reports else config.work / "reports"


def run_all(args: argparse.Namespace, config: cfg.RunConfig) -> int:
    """up, run or wait-runner, then down, also after a failure."""
    status = 1
    try:
        outputs = flows.up(config)
        if config.mode == "runner":
            flows.wait_runner(config, outputs)
            status = 0
        else:
            status = flows.run_tests(config, outputs, reports_dir(args, config))
    finally:
        if args.keep:
            log(
                f"--keep: not deleting; run `down --workdir {config.workdir}` when done"
            )
        else:
            _, outputs = cfg.load_state(config.work)
            flows.down(config, outputs)
    return status


def reap(args: argparse.Namespace) -> int:
    """Delete expired resources of the given providers."""
    if not args.dry_run and not args.yes:
        log("reap deletes resources: add --yes, or --dry-run to only list them")
        return 2
    sweeper = reaper.Reaper(dry_run=args.dry_run)
    for provider in args.providers:
        if provider == "aws":
            for region in args.regions or ["us-east-1"]:
                sweeper.aws(region)
        elif provider == "gcp":
            if not args.gcp_project:
                log("reap --provider gcp needs --gcp-project")
                return 2
            sweeper.gcp(args.gcp_project)
        elif provider == "azure":
            sweeper.azure()
        elif provider == "hetzner":
            sweeper.hetzner()
        elif provider == "github":
            sweeper.github(args.github_repository)
    log(
        f"{len(sweeper.deleted)} expired resources {'found' if args.dry_run else 'deleted'}, "
        f"{len(sweeper.failed)} failed"
    )
    return 1 if sweeper.failed else 0


STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _stop(signum: int, frame: FrameType | None) -> None:
    """Turn the first SIGTERM or SIGHUP into SystemExit, ignore later ones.

    A cancelled CI job or an expired step budget sends SIGTERM; without this
    Python dies at once and `all` never reaches its `down`, leaving paid
    machines behind. Raising SystemExit runs the ``finally`` blocks; later
    signals are ignored so that `down` itself is not interrupted (SIGKILL
    still ends the process).
    """
    del frame
    for number in STOP_SIGNALS:
        signal.signal(number, _ignore)
    log(f"received {signal.Signals(signum).name}: stopping, deleting what was created")
    raise SystemExit(128 + signum)


def _ignore(signum: int, frame: FrameType | None) -> None:
    del frame
    log(f"received {signal.Signals(signum).name} again: still deleting")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the subcommand."""
    args = build_parser().parse_args(argv)
    previous = {}
    if threading.current_thread() is threading.main_thread():
        previous = {number: signal.signal(number, _stop) for number in STOP_SIGNALS}
    try:
        return dispatch(args)
    except (CommandError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        log(f"error: {error}")
        return 1
    except subprocess.TimeoutExpired as error:
        log(f"error: {error}")
        return 1
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def job_manifest(args: argparse.Namespace) -> int:
    """Print or write the Kubernetes Job of a run."""
    text = kube.render_job(config_from_args(args, needs_cidr=False))
    if args.output:
        args.output.write_text(text, "utf-8")
    else:
        sys.stdout.write(text)
    return 0


def resume(args: argparse.Namespace) -> int:
    """run, wait-runner or down for the run in --workdir."""
    if args.workdir is None:
        log(f"{args.command} needs --workdir of a run created with up")
        return 2
    config, outputs = cfg.load_state(args.workdir)
    if args.command == "run":
        return flows.run_tests(config, outputs, reports_dir(args, config))
    if args.command == "wait-runner":
        flows.wait_runner(config, outputs)
    else:
        flows.down(config, outputs)
    return 0


def start(args: argparse.Namespace) -> int:
    """plan, up or all for a new run."""
    command = args.command
    config = config_from_args(args, needs_cidr=command != "plan" or args.check)
    if command == "plan":
        return plan(config, check=args.check)
    if not args.yes and config.provider not in {"ssh", "kubernetes", "local"}:
        describe(config)
        log(f"{command} creates paid resources: add --yes to go ahead")
        return 2
    cfg.save_state(config)
    log(f"run {config.run_id}: state in {config.workdir}")
    if command == "up":
        flows.up(config)
        print(config.workdir)
        return 0
    return run_all(args, config)


def dispatch(args: argparse.Namespace) -> int:
    """Run one subcommand."""
    if args.command == "reap":
        return reap(args)
    if args.command == "job-manifest":
        return job_manifest(args)
    if args.command in {"run", "wait-runner", "down"}:
        return resume(args)
    return start(args)
