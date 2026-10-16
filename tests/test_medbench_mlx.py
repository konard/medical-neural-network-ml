"""MLX adapters are unit tested without importing MLX or downloading weights."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from medbench.hardware import BudgetExceededError, HostInfo, ResourceBudget
from medbench.mlx_backend import MlxBackend


def _snapshot(tmp_path: Path, size: int = 100) -> Path:
    (tmp_path / "config.json").write_text(
        json.dumps({"vision_config": {"layers": 1}, "model_type": "gemma3"})
    )
    with (tmp_path / "model.safetensors").open("wb") as handle:
        handle.truncate(size)
    return tmp_path


def _host(available: float = 12) -> HostInfo:
    return HostInfo(12, 12, 19.33, available, None, apple_silicon=True)


def test_oversized_checkpoint_is_refused_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free memory, not the Mac's total installed RAM, controls admission."""
    monkeypatch.setattr(
        "medbench.mlx_backend.detect_host", lambda **_: _host(available=2)
    )
    with pytest.raises(BudgetExceededError, match="budget allows"):
        MlxBackend(_snapshot(tmp_path, size=2_000_000_000))


def test_text_only_conversion_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLX-LM checkpoints cannot silently stand in for vision weights."""
    monkeypatch.setattr("medbench.mlx_backend.detect_host", lambda **_: _host())
    (tmp_path / "config.json").write_text('{"model_type": "gemma3_text"}')
    with pytest.raises(ValueError, match="vision_config"):
        MlxBackend(tmp_path)


def test_non_apple_host_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "medbench.mlx_backend.detect_host", lambda **_: HostInfo(4, 4, 16, 12, None)
    )
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        MlxBackend(tmp_path)


def test_mlx_templates_images_and_collects_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real adapter boundary with fake numerical dependencies."""
    seen = {}
    monkeypatch.setattr("medbench.mlx_backend.detect_host", lambda **_: _host())
    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        device_info=lambda: {"max_recommended_working_set_size": 12_000_000_000},
        set_memory_limit=lambda value: seen.update(memory=value),
        set_cache_limit=lambda value: seen.update(cache=value),
        clear_cache=lambda: seen.update(cleared=True),
    )

    def load(path: str, **kwargs: object) -> tuple[object, object]:
        seen.update(path=path, load_kwargs=kwargs)
        return object(), object()

    def template(*args: object, **kwargs: object) -> str:
        seen.update(template_args=args, template_kwargs=kwargs)
        return "formatted prompt"

    def generate(*args: object, **kwargs: object):
        seen.update(generate_args=args, generate_kwargs=kwargs)
        yield SimpleNamespace(text="Yes", prompt_tokens=300, generation_tokens=1)
        yield SimpleNamespace(text=".", prompt_tokens=300, generation_tokens=2)

    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(
        sys.modules, "mlx_vlm", SimpleNamespace(load=load, stream_generate=generate)
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        SimpleNamespace(apply_chat_template=template),
    )
    backend = MlxBackend(
        _snapshot(tmp_path), ResourceBudget(max_seconds=60), memory_limit_gb=4
    )
    image = io.BytesIO()
    Image.new("L", (8, 8)).save(image, format="PNG")
    result = backend.generate("Question", [image.getvalue()], max_tokens=8)
    assert result.text == "Yes."
    assert result.prompt_tokens == 300
    assert result.completion_tokens == 2
    assert seen["memory"] == 4_000_000_000
    assert seen["load_kwargs"] == {"trust_remote_code": False}
    assert seen["template_kwargs"] == {"num_images": 1}
    assert seen["generate_args"][2] == "formatted prompt"
    assert seen["generate_kwargs"]["temperature"] == 0
    assert seen["generate_kwargs"]["image"][0].mode == "RGB"
    assert seen["cleared"]
