"""Work on a GPU host over SSH: wait, upload the repository, run, collect."""

from __future__ import annotations

import io
import shlex
import subprocess
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.cloud_gpu import config as cfg
from scripts.cloud_gpu.shell import (
    CommandError,
    log,
    output,
    require_tool,
    run,
    wait_until,
)

READY_FILE = "/var/lib/medbench-cloud-gpu-ready"
REMOTE_DIR = "medbench"  # relative to the login user's home directory
REBOOT_NEEDED = 3  # exit code of scripts/gpu-host-bootstrap.sh driver


def generate_key(workdir: Path) -> tuple[Path, str]:
    """Create the run's own SSH key pair (no passphrase) unless it exists.

    The key lives only in the working directory and in the VM; Terraform
    receives the public half.
    """
    key = workdir / "ssh" / "id_ed25519"
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        key.parent.chmod(0o700)
        run(
            [
                require_tool("ssh-keygen"),
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                f"medbench-cloud-gpu-{workdir.name}",
                "-f",
                str(key),
            ],
        )
    public = key.with_suffix(".pub").read_text("utf-8").strip()
    return key, public


@dataclass
class Ssh:
    """One SSH destination with the run's key and known_hosts file."""

    host: str
    user: str
    key: Path | None
    known_hosts: Path
    port: int = 22

    def argv(self, *, tty: bool = False) -> list[str]:
        """The ssh command up to and including the destination."""
        args = [
            require_tool("ssh"),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=6",
            "-p",
            str(self.port),
        ]
        if self.key is not None:
            args += ["-i", str(self.key), "-o", "IdentitiesOnly=yes"]
        if tty:
            args.append("-t")
        return [*args, f"{self.user}@{self.host}"]

    def sudo(self) -> str:
        """Prefix for commands that need root on the host."""
        return "" if self.user == "root" else "sudo -n "

    def run(
        self,
        command: str,
        *,
        check: bool = True,
        capture: bool = False,
        stdin: bytes | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a shell command line on the host."""
        return run(
            [*self.argv(), command],
            check=check,
            capture=capture,
            stdin=stdin,
            timeout=timeout,
        )

    def ok(self, command: str) -> bool:
        """Return True if ``command`` succeeds on the host (False if not)."""
        try:
            result = self.run(command, check=False, capture=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0


def wait_ready(ssh: Ssh, *, cloud_init: bool, timeout: float = 900) -> None:
    """Wait until SSH answers and, on a new cloud VM, cloud-init is done."""
    probe = f"test -f {READY_FILE}" if cloud_init else "true"
    wait_until(f"SSH on {ssh.host}", lambda: ssh.ok(probe), timeout=timeout)


def wait_gpu(ssh: Ssh, *, timeout: float = 900) -> None:
    """Wait until nvidia-smi works (the Google image installs the driver
    on first boot, the Azure extension right after it)."""
    wait_until(
        f"NVIDIA driver on {ssh.host}",
        lambda: ssh.ok("nvidia-smi --query-gpu=name --format=csv,noheader"),
        timeout=timeout,
        interval=20,
    )


def repository_files(root: Path) -> list[str]:
    """Return tracked and untracked, not ignored files of the work tree."""
    listing = output(
        [
            require_tool("git"),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=root,
    )
    names = sorted({name for name in listing.split("\0") if name})
    return [name for name in names if (root / name).is_file()]


def repository_archive(root: Path = cfg.REPO_ROOT) -> bytes:
    """Return a gzip tar of the work tree as it is, including local edits.

    The host needs no git credentials and tests exactly what is checked out.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in repository_files(root):
            archive.add(root / name, arcname=name, recursive=False)
    return buffer.getvalue()


def current_commit(root: Path = cfg.REPO_ROOT) -> str:
    """Return the commit of the work tree, with ``+dirty`` for local edits."""
    git = require_tool("git")
    commit = output([git, "rev-parse", "HEAD"], cwd=root).strip()
    dirty = output([git, "status", "--porcelain", "--untracked-files=no"], cwd=root)
    return f"{commit}+dirty" if dirty.strip() else commit


def upload_repository(ssh: Ssh, archive: bytes, remote_dir: str = REMOTE_DIR) -> None:
    """Replace ``remote_dir`` on the host with the archive's content."""
    log(f"uploading the repository ({len(archive) // 1024} KiB) to {ssh.host}")
    quoted = shlex.quote(remote_dir)
    ssh.run(
        f"rm -rf {quoted} && mkdir -p {quoted} && tar -xzf - -C {quoted}",
        stdin=archive,
    )


def env_assignments(env: Mapping[str, str]) -> str:
    """Return ``K=V`` words for a remote shell, each value quoted."""
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


def run_tests(
    ssh: Ssh,
    env: Mapping[str, str],
    *,
    remote_dir: str = REMOTE_DIR,
    out_dir: str = "medbench-reports",
) -> int:
    """Run scripts/gpu-host-run.sh on the host and return its exit code."""
    # The cache sits next to the checkout, so a host we keep reuses the
    # downloads although every run replaces the checkout.
    variables = env_assignments(
        {"CACHE_DIR": "../.cache/medbench", **env, "OUT_DIR": out_dir}
    )
    command = (
        f"cd {shlex.quote(remote_dir)} && env {variables} bash scripts/gpu-host-run.sh"
    )
    log(f"running scripts/gpu-host-run.sh on {ssh.host}")
    result = ssh.run(command, check=False)
    log(f"gpu-host-run.sh exited with {result.returncode}")
    return result.returncode


def extract_reports(data: bytes, local_dir: Path) -> list[str]:
    """Unpack a tar stream of reports safely and return the file names."""
    local_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        archive.extractall(local_dir, filter="data")
        return sorted(m.name for m in archive.getmembers() if m.isfile())


def fetch_reports(
    ssh: Ssh,
    local_dir: Path,
    *,
    remote_dir: str = REMOTE_DIR,
    out_dir: str = "medbench-reports",
) -> list[str]:
    """Copy the reports from the host into ``local_dir``."""
    path = shlex.quote(f"{remote_dir}/{out_dir}")
    result = subprocess.run(
        [*ssh.argv(), f"test -d {path} && tar -cf - -C {path} ."],
        stdout=subprocess.PIPE,
        check=False,
        timeout=1800,
    )
    if result.returncode != 0:
        log(f"no reports found in {remote_dir}/{out_dir} on {ssh.host}")
        return []
    names = extract_reports(result.stdout, local_dir)
    log(f"collected {len(names)} report files into {local_dir}")
    return names


def bootstrap(ssh: Ssh, steps: Sequence[str], *, remote_dir: str = REMOTE_DIR) -> None:
    """Run scripts/gpu-host-bootstrap.sh steps as root on a bare host.

    The driver step exits with 3 when a reboot is needed; then the host is
    rebooted and the step runs again.
    """
    script = f"{shlex.quote(remote_dir)}/scripts/gpu-host-bootstrap.sh"
    for step in steps:
        command = f"{ssh.sudo()}bash {script} {shlex.quote(step)}"
        result = ssh.run(command, check=False)
        if result.returncode == REBOOT_NEEDED:
            log(f"{ssh.host}: rebooting to load the NVIDIA driver")
            ssh.run(f"{ssh.sudo()}systemctl reboot", check=False)
            wait_until(
                f"{ssh.host} back after the reboot",
                lambda: ssh.ok("true"),
                timeout=1200,
                interval=15,
            )
            result = ssh.run(command, check=False)
        if result.returncode != 0:
            raise CommandError(["gpu-host-bootstrap.sh", step], result.returncode)


def start_runner(
    ssh: Ssh,
    jit_config: str,
    *,
    poweroff: bool,
    remote_dir: str = REMOTE_DIR,
) -> None:
    """Start one ephemeral GitHub Actions runner on the host.

    The just-in-time configuration travels on SSH stdin, never on a command
    line, in Terraform state or in user data (scripts/gpu-runner-start.sh).
    """
    script = f"{shlex.quote(remote_dir)}/scripts/gpu-runner-start.sh"
    flag = " --poweroff" if poweroff else ""
    log(f"starting an ephemeral GitHub Actions runner on {ssh.host}")
    ssh.run(f"{ssh.sudo()}bash {script}{flag}", stdin=jit_config.encode("utf-8"))
