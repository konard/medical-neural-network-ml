"""Optional native Apple Silicon inference with MLX-VLM.

Only local, previously downloaded snapshots are loaded. Downloads and model
license acceptance are explicit steps; importing medbench never imports MLX.
"""

from __future__ import annotations

import io
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from medbench.backends import Generation
from medbench.hardware import ResourceBudget, detect_host


class MlxBackend:
    """Run a quantized vision model inside a conservative unified RAM budget."""

    def __init__(
        self,
        model_path: Path,
        budget: ResourceBudget | None = None,
        memory_limit_gb: float = 8.0,
    ) -> None:
        """Check the host and checkpoint before loading any weights.

        The ceiling never overrides the available-memory check. Allow 1.5 GB
        beyond stored weights for a short prompt, KV cache, one image and
        runtime buffers. This is a lower bound, not a guaranteed fit.
        """
        self.budget = budget or ResourceBudget()
        host = detect_host(with_gpus=False)
        if not host.apple_silicon:
            msg = "MLX requires an Apple Silicon Mac; use the openai backend here"
            raise RuntimeError(msg)
        if memory_limit_gb <= 0:
            msg = "memory_limit_gb must be positive"
            raise ValueError(msg)
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        if not config.get("vision_config"):
            msg = "Use an MLX-VLM snapshot with vision_config, not text-only MLX-LM weights"
            raise ValueError(msg)
        weights = list(model_path.glob("*.safetensors"))
        if not weights:
            msg = f"No safetensors weights in {model_path}"
            raise FileNotFoundError(msg)
        required = sum(path.stat().st_size for path in weights) / 1e9 + 1.5
        self.budget.check_memory(required, host)
        if required > memory_limit_gb:
            msg = f"Checkpoint needs at least {required:.1f} GB; limit is {memory_limit_gb:.1f} GB"
            raise ValueError(msg)
        for key, value in self.budget.thread_env(host).items():
            os.environ[key] = value
        import mlx.core as mx
        from mlx_vlm import load

        if not mx.metal.is_available():
            msg = "MLX Metal GPU is unavailable"
            raise RuntimeError(msg)
        recommended = mx.device_info()["max_recommended_working_set_size"]
        limit = int(
            min(
                memory_limit_gb * 1e9,
                self.budget.memory_limit_gb(host) * 1e9,
                recommended,
            )
        )
        if required * 1e9 > limit:
            msg = "Checkpoint exceeds the Metal working set budget"
            raise ValueError(msg)
        mx.set_memory_limit(limit)
        mx.set_cache_limit(min(256 * 1024**2, limit // 8))
        self._mx = mx
        self.budget.check_deadline()
        self.model, self.processor = load(str(model_path), trust_remote_code=False)
        self.config = config
        self.name = f"mlx:{model_path.name}"

    def generate(
        self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 256
    ) -> Generation:
        """Template and decode greedily, checking time after each token.

        An outer process timeout must interrupt model loading or a long Metal
        operation. See ``scripts/medbench-mlx-run.py``.
        """
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        self.budget.check_deadline()
        start = time.perf_counter()
        pictures = []
        for data in images:
            with Image.open(io.BytesIO(data)) as picture:
                pictures.append(picture.convert("RGB"))
        formatted = apply_chat_template(
            self.processor, self.config, prompt, num_images=len(pictures)
        )
        chunks: list[str] = []
        last: Any = None
        stream = stream_generate(
            self.model,
            self.processor,
            formatted,
            image=pictures or None,
            max_tokens=max_tokens,
            temperature=0.0,
            prefill_step_size=256,
        )
        try:
            for result in stream:
                self.budget.check_deadline()
                chunks.append(result.text)
                last = result
        finally:
            stream.close()
            self._mx.clear_cache()
        return Generation(
            "".join(chunks),
            time.perf_counter() - start,
            last.prompt_tokens if last else None,
            last.generation_tokens if last else None,
        )
