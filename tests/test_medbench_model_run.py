"""Tests for scripts/medbench-model-run.sh (issue #1).

Only the parts that need no download are tested: the memory preflight that
decides whether weights are copied into RAM, the usage message, the workflow
that calls the script, and (with UniMedVL's ``echo`` engine, issue #6) the
server start, the deadline and the cleanup on exit.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "medbench-model-run.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "model-tests.yml"
REVISION = "fa17ea2647ef236db91adfe1465a61ab6562cf2c"


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


linux_x86_64 = pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64",
    reason="the run script only runs on Linux x86_64",
)


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
    """Code commit, weight hashes and the flash-attn wheel hash are fixed."""
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
    assert re.search(r'FLASH_ATTN_SHA256="[0-9a-f]{64}"', script)
    pins = unimedvl_pins("unimedvl-requirements.txt")
    # The newest torch with a prebuilt flash-attn wheel (issue #19); the wheel
    # has to be built for the same torch, and for the C++11 ABI that PyPI
    # torch wheels use since 2.7.
    assert pins["torch"] == "2.8.0"
    assert "cu12torch2.8cxx11abiTRUE" in script
    # The last 4.x: upstream's code fails on transformers 5 (issue #19).
    assert pins["transformers"] == "4.57.6"


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
    assert {"huggingface-hub", "filelock", "nvidia-cudnn-cu12"} <= set(locked)
