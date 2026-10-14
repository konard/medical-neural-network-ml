"""Ways to run a model: one small interface, several implementations.

Every backend turns a text prompt (plus optional PNG images) into text.

* :class:`NaiveBayesBackend` needs nothing and answers diagnosis prompts
  with the reference Bayes ranking. It is the floor every model must beat.
* :class:`OpenAICompatibleBackend` talks HTTP to any server with the OpenAI
  ``/v1/chat/completions`` API: ``llama-server`` (llama.cpp, CPU or GPU),
  Ollama, vLLM, SGLang or Hugging Face TGI. This is how CI runs real models.
* :class:`TransformersBackend` loads a Hugging Face model in-process with
  PyTorch (GPU runner). Heavy imports happen only when it is created.
* :class:`TesseractBackend` runs the classic Tesseract OCR engine, a cheap
  OCR baseline.
* :class:`ScriptedBackend` returns canned answers for unit tests.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from medbench.synthetic_cases import rank_conditions

__all__ = [
    "Backend",
    "Generation",
    "NaiveBayesBackend",
    "OpenAICompatibleBackend",
    "ScriptedBackend",
    "TesseractBackend",
    "TransformersBackend",
]


@dataclass(frozen=True)
class Generation:
    """Output of one model call.

    Attributes:
        text: Generated text.
        latency_s: Wall-clock seconds for the call.
        prompt_tokens: Tokens read, when the backend reports it.
        completion_tokens: Tokens written, when the backend reports it.
    """

    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Backend(Protocol):
    """Anything that can answer a prompt."""

    name: str

    def generate(
        self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 256
    ) -> Generation:
        """Answer ``prompt``; ``images`` are PNG bytes."""
        ...


class ScriptedBackend:
    """Return answers from a function, for tests and dry runs."""

    def __init__(
        self, respond: Callable[[str, Sequence[bytes]], str], name: str = "scripted"
    ):
        """Create the backend.

        Args:
            respond: Function from ``(prompt, images)`` to the answer text.
            name: Name used in reports.
        """
        self.name = name
        self._respond = respond
        self.calls: list[tuple[str, int]] = []

    def generate(
        self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 256
    ) -> Generation:
        """Return the scripted answer."""
        del max_tokens
        start = time.perf_counter()
        self.calls.append((prompt, len(images)))
        text = self._respond(prompt, images)
        return Generation(text, time.perf_counter() - start)


_SYMPTOM_LIST = re.compile(r"patient reports: (.+?)\.\n", re.IGNORECASE)


class NaiveBayesBackend:
    """Reference ranking from the generator's own knowledge base.

    Only diagnosis prompts get a real answer; other prompts get ``"unknown"``.
    """

    name = "naive-bayes"

    def generate(
        self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 256
    ) -> Generation:
        """Rank conditions for the symptoms listed in ``prompt``."""
        del images, max_tokens
        start = time.perf_counter()
        match = _SYMPTOM_LIST.search(prompt)
        if not match:
            return Generation("unknown", time.perf_counter() - start)
        symptoms = [s.strip() for s in match.group(1).split(",")]
        ranked = rank_conditions(symptoms)[:5]
        text = "\n".join(f"{i}. {name}" for i, (name, _) in enumerate(ranked, 1))
        return Generation(text, time.perf_counter() - start)


class OpenAICompatibleBackend:
    """Client for an OpenAI-compatible chat completions server.

    Start a CPU server for MedGemma 4B with llama.cpp, for example::

        llama-server -m medgemma-4b-it-Q4_K_M.gguf \\
            --mmproj mmproj-F16.gguf -t 4 -c 4096 --port 8080

    and use ``OpenAICompatibleBackend("http://127.0.0.1:8080/v1")``.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        api_key: str | None = None,
        timeout_s: float = 600.0,
        temperature: float = 0.0,
    ):
        """Create the client.

        Args:
            base_url: Server URL ending in ``/v1``.
            model: Model name the server expects.
            api_key: Bearer token, if the server needs one.
            timeout_s: Seconds to wait for one answer.
            temperature: Sampling temperature; 0 makes runs repeatable.

        Raises:
            ValueError: If the URL is not HTTP(S).
        """
        if not base_url.startswith(("http://", "https://")):
            message = f"base_url must start with http:// or https://, got {base_url!r}"
            raise ValueError(message)
        self.name = f"openai:{model}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature

    def build_payload(
        self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 256
    ) -> dict[str, Any]:
        """Return the JSON body for one request (images as data URLs)."""
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(i).decode()
                },
            }
            for i in images
        ]
        content.append({"type": "text", "text": prompt})
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

    def generate(
        self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 256
    ) -> Generation:
        """Send the prompt and return the answer."""
        body = json.dumps(self.build_payload(prompt, images, max_tokens)).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(  # noqa: S310 - scheme checked in __init__
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
            data = json.loads(response.read().decode())
        latency = time.perf_counter() - start
        usage = data.get("usage") or {}
        return Generation(
            text=data["choices"][0]["message"]["content"] or "",
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class TesseractBackend:
    """Classic OCR with the ``tesseract`` command (images only)."""

    name = "tesseract"

    def __init__(self, binary: str | None = None, language: str = "eng"):
        """Create the backend.

        Args:
            binary: Path to ``tesseract``; looked up on ``PATH`` when omitted.
            language: Tesseract language pack.

        Raises:
            FileNotFoundError: If Tesseract is not installed.
        """
        found = binary or shutil.which("tesseract")
        if found is None:
            message = (
                "tesseract not found; install it with 'apt-get install tesseract-ocr'"
            )
            raise FileNotFoundError(message)
        self.binary = found
        self.language = language

    def generate(
        self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 256
    ) -> Generation:
        """OCR every image and join the texts; the prompt is ignored."""
        del prompt, max_tokens
        start = time.perf_counter()
        texts = []
        with tempfile.TemporaryDirectory() as tmp:
            for index, image in enumerate(images):
                path = Path(tmp) / f"page{index}.png"
                path.write_bytes(image)
                result = subprocess.run(  # noqa: S603
                    [self.binary, str(path), "-", "-l", self.language],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=120,
                )
                texts.append(result.stdout.strip())
        return Generation("\n".join(texts), time.perf_counter() - start)


class TransformersBackend:  # pragma: no cover - needs torch and a model download
    """Run a Hugging Face vision-language model in-process.

    Needs ``pip install torch transformers accelerate`` and, for gated
    models such as MedGemma, an accepted license plus ``HF_TOKEN``.
    """

    def __init__(
        self, model_id: str, device_map: str = "auto", dtype: str = "bfloat16"
    ):
        """Load the model and its processor.

        Args:
            model_id: Hugging Face repository id.
            device_map: Where to place the weights (``"auto"`` uses GPUs).
            dtype: Weight dtype name in ``torch``.
        """
        import torch  # noqa: PLC0415 - optional heavy dependency
        from transformers import (  # noqa: PLC0415
            AutoModelForImageTextToText,
            AutoProcessor,
        )

        self.name = f"transformers:{model_id}"
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device_map
        )

    def generate(
        self, prompt: str, images: Sequence[bytes] = (), max_tokens: int = 256
    ) -> Generation:
        """Generate greedily with the model's chat template."""
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        content: list[dict[str, Any]] = [
            {"type": "image", "image": Image.open(io.BytesIO(i)).convert("RGB")}
            for i in images
        ]
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        length = inputs["input_ids"].shape[-1]
        start = time.perf_counter()
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        latency = time.perf_counter() - start
        new_tokens = output[0][length:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return Generation(text, latency, int(length), len(new_tokens))
