"""Tests for scripts/medbench-model-run.sh (issue #1).

Only the parts that need no download are tested: the memory preflight that
decides whether weights are copied into RAM, the usage message and the
workflow that calls the script.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
    assert "if: github.event_name == 'workflow_dispatch' && inputs.gpu" in gpu_job
    assert "runs-on: [self-hosted, linux, x64, gpu]" in gpu_job
    assert "secrets." not in text


def test_workflow_pins_every_download() -> None:
    """Archives and models chosen in the workflow come with a SHA-256."""
    text = WORKFLOW.read_text(encoding="utf-8")
    for name in ("LLAMA", "CUDART", "MODEL", "MMPROJ"):
        assert f"{name}_SHA256" in text
    assert "MODEL_REVISION=" in text
