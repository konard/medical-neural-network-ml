"""Tests for scripts/medbench-model-run.sh (issue #1).

Only the parts that need no download are tested: the memory preflight that
decides whether weights are copied into RAM, the usage message, the workflow
that calls the script, and (with UniMedVL's ``echo`` engine, issue #6) the
server start, the deadline and the cleanup on exit.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "medbench-model-run.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "model-tests.yml"
REVISION = "fa17ea2647ef236db91adfe1465a61ab6562cf2c"


linux_x86_64 = pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64",
    reason="the run script only runs on Linux x86_64",
)


def run_script(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=60,
        check=False,
    )


def fake_model(tmp_path: Path, size: int) -> Path:
    """A sparse model file reached through a symlink, like a shared cache."""
    store = tmp_path / "store"
    store.mkdir()
    real = store / "model.gguf"
    with real.open("wb") as handle:
        handle.truncate(size)
    model_dir = tmp_path / "cache" / "models" / REVISION[:12]
    model_dir.mkdir(parents=True)
    (model_dir / "model.gguf").symlink_to(real)
    return tmp_path / "cache"


def test_huge_model_behind_symlink_falls_back_to_mmap(tmp_path: Path) -> None:
    """The real file size is used, so a model larger than RAM is not copied.

    A regression test: measuring the symlink itself (38 bytes) chose
    ``none`` and the out-of-memory killer stopped llama-server.
    """
    cache = fake_model(tmp_path, size=1 << 40)
    result = run_script(
        "preflight", CACHE_DIR=str(cache), MODEL_FILE="model.gguf", MMPROJ_FILE=""
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "mmap"
    assert "using mmap" in result.stderr


def test_gpu_offload_uses_mmap(tmp_path: Path) -> None:
    """With GPU layers the weights live on the GPU, so RAM is not checked."""
    cache = fake_model(tmp_path, size=1024)
    result = run_script(
        "preflight",
        CACHE_DIR=str(cache),
        MODEL_FILE="model.gguf",
        MMPROJ_FILE="",
        GPU_LAYERS="999",
    )
    assert result.stdout.strip() == "mmap"


def test_explicit_load_mode_wins() -> None:
    """LOAD_MODE other than auto is used as given."""
    result = run_script("preflight", LOAD_MODE="mlock")
    assert result.stdout.strip() == "mlock"


def test_unknown_command_is_a_usage_error() -> None:
    result = run_script("wat")
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_gpu_job_only_runs_when_started_by_hand() -> None:
    """Pull requests must never reach the self-hosted GPU runner."""
    text = WORKFLOW.read_text(encoding="utf-8")
    gpu_job = text.split("\n  gpu:\n", 1)[1].split("\n  pipeline-status:\n", 1)[0]
    condition = gpu_job.split("    if: >-\n", 1)[1].split("\n    environment:", 1)[0]
    assert " ".join(condition.split()) == (
        "github.event_name == 'workflow_dispatch' && inputs.gpu && "
        "github.ref == 'refs/heads/main'"
    )
    assert "    environment: gpu\n" in gpu_job
    assert "runs-on: [self-hosted, linux, x64, gpu]" in gpu_job
    assert "secrets." not in text
    assert "persist-credentials: false" in gpu_job
    assert "contents: write" not in gpu_job


def test_gpu_job_removes_what_it_downloaded() -> None:
    """A reused GPU host must not keep untrusted weights between runs (#10)."""
    text = WORKFLOW.read_text(encoding="utf-8")
    gpu_job = text.split("\n  gpu:\n", 1)[1].split("\n  pipeline-status:\n", 1)[0]
    cleanup = gpu_job.rsplit("      - name: ", 1)[1]
    assert cleanup.startswith("Remove downloads and reports from the GPU host")
    assert "if: always()" in cleanup
    assert "rm -rf .cache/medbench medbench-reports" in cleanup


def test_only_the_gpu_job_runs_on_a_self_hosted_runner() -> None:
    """Every other job, in every workflow, stays on GitHub-hosted runners."""
    workflows = ROOT / ".github" / "workflows"
    hits = [
        (path.name, line.strip())
        for path in sorted(workflows.glob("*.y*ml"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if "self-hosted" in line and line.lstrip().startswith("runs-on:")
    ]
    assert hits == [("model-tests.yml", "runs-on: [self-hosted, linux, x64, gpu]")]


def test_workflow_pins_every_download() -> None:
    """Archives and models chosen in the workflow come with a SHA-256."""
    text = WORKFLOW.read_text(encoding="utf-8")
    for name in ("LLAMA", "CUDART", "MODEL", "MMPROJ"):
        assert f"{name}_SHA256" in text
    assert "MODEL_REVISION=" in text


def tar_folder(tmp_path: Path, folder: str, files: dict[str, str]) -> Path:
    """A release-style .tar.gz that keeps ``files`` in one top-level folder."""
    source = tmp_path / "src" / folder
    source.mkdir(parents=True)
    for name, content in files.items():
        (source / name).write_text(content)
        (source / name).chmod(0o755)
    archive = tmp_path / f"{folder}.tar.gz"
    subprocess.run(
        ["tar", "--create", "--gzip", "--file", str(archive), folder],
        cwd=source.parent,
        check=True,
    )
    return archive


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@linux_x86_64
def test_cuda_runtime_lands_next_to_the_cuda_backend(tmp_path: Path) -> None:
    """llama.cpp's libggml-cuda.so finds libcudart only in its own folder.

    Both release archives keep their files in one top-level folder, and the
    CUDA backend's RUNPATH is ``$ORIGIN``. Unpacked without
    ``--strip-components 1`` the runtime went to a subfolder, the backend
    did not load and llama-server ran on the CPU (issue #21).
    """
    llama = tar_folder(
        tmp_path,
        "llama-b1-bin-ubuntu-cuda-13.4-x64",
        {"llama-server": "#!/bin/sh\n", "libggml-cuda.so": ""},
    )
    cudart = tar_folder(
        tmp_path,
        "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64",
        {"libcudart.so.13": "", "libcublas.so.13": ""},
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    for archive in (llama, cudart):
        (cache / archive.name).write_bytes(archive.read_bytes())
    model_dir = cache / "models" / REVISION[:12]
    model_dir.mkdir(parents=True)
    (model_dir / "model.gguf").write_text("gguf")
    result = run_script(
        "download",
        CACHE_DIR=str(cache),
        LLAMA_VERSION="b1",
        LLAMA_ARCHIVE=llama.name,
        LLAMA_SHA256=sha256(llama),
        CUDART_ARCHIVE=cudart.name,
        CUDART_SHA256=sha256(cudart),
        MODEL_REVISION=REVISION,
        MODEL_FILE="model.gguf",
        MODEL_SHA256=sha256(model_dir / "model.gguf"),
        MMPROJ_FILE="",
    )
    assert result.returncode == 0, result.stderr
    llama_dir = cache / f"llama-b1-{sha256(llama)[:12]}"
    assert sorted(path.name for path in llama_dir.iterdir()) == [
        "libcublas.so.13",
        "libcudart.so.13",
        "libggml-cuda.so",
        "llama-server",
    ]


def test_pull_request_paths_cover_every_file_the_jobs_run() -> None:
    """A change to any file the model jobs execute must trigger them.

    Issue #4: the real-data test, the budget wrapper and pyproject.toml (the
    installed dependencies) were missing from the filter, so a pull request
    that broke them stayed green without running the model tests.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    trigger = text.split("\n  pull_request:\n", 1)[1].split("\n  schedule:\n", 1)[0]
    filters = set(re.findall(r"^      - '([^']+)'$", trigger, re.MULTILINE))
    jobs = text.split("\njobs:\n", 1)[1]
    used = set(re.findall(r"\b((?:scripts|tests)/[\w.-]+\.(?:sh|py))\b", jobs))
    script = SCRIPT_PATH.read_text()
    used |= set(re.findall(r"\b(tests/[\w.-]+\.py)\b", script))
    # Files the run script itself calls through $SCRIPT_DIR.
    used |= {
        f"scripts/{name}"
        for name in re.findall(r"\$SCRIPT_DIR/([\w.-]+\.(?:py|txt))", script)
    }
    used.discard("scripts/check-pipeline-status.sh")  # status gate, not a test
    used |= {"pyproject.toml", ".github/workflows/model-tests.yml"}

    missing = sorted(path for path in used if path not in filters)
    assert not missing, f"model-tests.yml pull_request paths miss {missing}"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def echo_run(
    tmp_path: Path, **env: str
) -> tuple[subprocess.CompletedProcess[str], int]:
    """``run`` with UniMedVL's echo engine: real script, server and client."""
    port = free_port()
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "run"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SERVER": "unimedvl",
            "UNIMEDVL_ENGINE": "echo",
            "UNIMEDVL_ECHO_PYTHON": sys.executable,
            "RUN_PYTEST": "0",
            "PORT": str(port),
            "CACHE_DIR": str(tmp_path / "cache"),
            "OUT_DIR": str(tmp_path / "out"),
            "TASKS": "diagnosis",
            "COUNT": "2",
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            **env,
        },
        cwd=ROOT,
        timeout=120,
        check=False,
    )
    return result, port


def server_is_down(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
    except (urllib.error.URLError, OSError):
        return True
    return False


@linux_x86_64
def test_unimedvl_echo_run_writes_reports_and_stops_the_server(
    tmp_path: Path,
) -> None:
    """Cold start, metrics and reports are saved; the server is gone after."""
    result, port = echo_run(tmp_path)
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    report = json.loads((out / "diagnosis.json").read_text())
    assert report["backend"] == "openai:unimedvl"
    assert report["items_run"] == 2
    assert report["metrics"]["error_rate"] == 0
    start = json.loads((out / "server-start.json").read_text())
    assert start["server"] == "unimedvl"
    assert start["cold_start_seconds"] >= 0
    metrics = json.loads((out / "server-metrics.json").read_text())
    assert metrics["requests"] == 2
    assert server_is_down(port)


@linux_x86_64
def test_used_up_time_budget_fails_and_still_stops_the_server(
    tmp_path: Path,
) -> None:
    """A deadline is a failure, not a green run, and frees the GPU process."""
    result, port = echo_run(tmp_path, TIME_BUDGET="0")
    assert result.returncode == 1
    assert "time budget used up" in result.stderr
    assert not (tmp_path / "out" / "diagnosis.json").exists()
    assert (tmp_path / "out" / "server-metrics.json").exists()
    assert server_is_down(port)


@linux_x86_64
def test_unknown_server_is_rejected(tmp_path: Path) -> None:
    result, _ = echo_run(tmp_path, SERVER="vllm")
    assert result.returncode == 2
    assert "unknown SERVER" in result.stderr


def test_unimedvl_pins() -> None:
    """Code commit and weight hashes are fixed; no flash-attn wheel is needed.

    Package versions are not asserted here: Dependabot updates the lock, and
    tests/test_unimedvl_upstream.py checks that upstream's code still runs
    and generates the same tokens on it (issue #21).
    """
    script = SCRIPT_PATH.read_text()
    assert 'UNIMEDVL_CODE_COMMIT="f51aa0accbd02e846e5ea20b595fb39b516dbe1f"' in script
    assert re.search(r'UNIMEDVL_MODEL_REVISION="[0-9a-f]{40}"', script)
    files = re.findall(r'^  "([\w.]+) ([0-9a-f]{64})"$', script, re.MULTILINE)
    assert {name for name, _ in files} >= {
        "ema_bf16.safetensors",
        "llm_config.json",
        "vit_config.json",
        "tokenizer.json",
    }
    # flash-attn wheels stop at torch 2.8; unimedvl_server.py provides
    # flash_attn_varlen_func on torch's own varlen_attn instead (issue #21).
    assert "FLASH_ATTN" not in script
    pins = unimedvl_pins("unimedvl-requirements.txt")
    assert "flash-attn" not in pins
    # The shim needs torch.nn.attention.varlen (torch 2.10+), and
    # restore_default_rope exists for transformers 5.
    assert Version(pins["torch"]) >= Version("2.10")
    assert Version(pins["transformers"]).major >= 5


def test_unimedvl_python_matches_everywhere() -> None:
    """The venv, the CI jobs and the lock use one Python."""
    script = SCRIPT_PATH.read_text()
    default = re.search(
        r'UNIMEDVL_PYTHON="\$\{UNIMEDVL_PYTHON:-python([\d.]+)\}"', script
    )
    jobs = re.findall(
        r"Setup Python ([\d.]+) for UniMedVL.*?python-version: '([\d.]+)'",
        WORKFLOW.read_text(),
        re.DOTALL,
    )
    source = (ROOT / "scripts" / "unimedvl-requirements.in").read_text()
    lock = (ROOT / "scripts" / "unimedvl-requirements.txt").read_text()
    compiled = re.findall(r"--python-version ([\d.]+)", source + lock)
    assert default
    assert len(jobs) == 2  # the GPU job and the unimedvl-cpu job
    assert len(compiled) == 2  # the .in's instructions and the lock's header
    versions = {default[1], *compiled, *(version for job in jobs for version in job)}
    assert len(versions) == 1, versions
    # The flash-attn wheels held this at 3.13 (issue #19); without them the
    # newest Python that torch publishes wheels for is used (issue #21).
    assert Version(versions.pop()) >= Version("3.14")


def test_unimedvl_cpu_job_runs_the_upstream_tests() -> None:
    """The unimedvl-cpu job tests the shims and upstream's code on CPU torch."""
    text = WORKFLOW.read_text(encoding="utf-8")
    job = text.split("\n  unimedvl-cpu:\n", 1)[1].split("\n  # Self-hosted", 1)[0]
    assert "UNIMEDVL_DEVICE: cpu" in job
    assert "tests/test_unimedvl_attention.py tests/test_unimedvl_upstream.py" in job
    assert "unimedvl-cpu" in text.split("\n  pipeline-status:\n", 1)[1]
    # Its pytest is installed at pyproject.toml's dev floors.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = project["project"]["optional-dependencies"]["dev"]
    for requirement in re.findall(r'"(pytest(?:-asyncio)?>=[\d.]+)"', job):
        assert requirement in dev


def unimedvl_pins(name: str) -> dict[str, str]:
    """Map each package in a UniMedVL requirements file to its exact pin."""
    pins: dict[str, str] = {}
    for line in (ROOT / "scripts" / name).read_text().splitlines():
        requirement = line.split("#", 1)[0].strip()
        if requirement:
            package, _, pinned = requirement.partition("==")
            assert pinned, line
            pins[package.strip().lower()] = pinned.strip()
    return pins


def test_unimedvl_lock_matches_its_input() -> None:
    """The .txt lock pins every package of the .in exactly as the .in does.

    The lock is compiled from the .in (with uv pip compile, see its header),
    so it also pins every transitive package; an edit to one file without the
    other would install something nobody reviewed.
    """
    wanted = unimedvl_pins("unimedvl-requirements.in")
    locked = unimedvl_pins("unimedvl-requirements.txt")
    assert wanted
    assert {name: locked.get(name) for name in wanted} == wanted
    # Everything torch, transformers and accelerate import is locked too.
    assert {"huggingface-hub", "filelock"} <= set(locked)
    assert any(name.startswith("nvidia-cudnn-cu") for name in locked)
