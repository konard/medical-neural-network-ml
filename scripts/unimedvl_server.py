#!/usr/bin/env python3
"""OpenAI-compatible chat server for UniMedVL (issue #6).

UniMedVL (https://github.com/uni-medical/UniMedVL, Apache-2.0) is a BAGEL
model with its own Python package and its own ``model.chat`` call. Neither
llama.cpp nor ``transformers.AutoModel`` can load it: there is no GGUF, and
the checkpoint's ``auto_map`` points to files that are not in the repository.
This server loads it the same way as upstream's
``codes/interactive_vqa_inferencer.py`` and answers ``/v1/chat/completions``.
medbench can then score it with ``--backend openai``, like every other model.

The server is our own code and stays under the Unlicense. It imports the
upstream checkout at run time and does not copy it. The checkout and the
weights keep their Apache-2.0 terms (see THIRD_PARTY_NOTICES.md).

Run it with the Python of the upstream environment (Python 3.10, torch 2.6.0,
transformers 4.49.0, accelerate 1.7.0 and flash-attn; see
``scripts/medbench-model-run.sh``), from the root of a checkout at
:data:`CODE_COMMIT`::

    cd UniMedVL && python -I /path/to/unimedvl_server.py \\
        --code-dir . --model-dir /path/to/checkpoint --port 8089

Endpoints:

* ``GET /health`` - 200 once the model is loaded;
* ``GET /v1/models`` - the model name;
* ``GET /metrics`` - versions, limits, load time, peak GPU memory and speed;
* ``POST /v1/chat/completions`` - text parts and base64 ``data:`` images.

The server never downloads anything a request names. It answers one request
at a time, because the GPU holds one model.

Failures:

* CUDA out-of-memory frees the cache and answers HTTP 507 with error type
  ``out_of_memory``. The server stays up, so the next item can still run.
* Any other model error answers HTTP 500 after the same cleanup.
* Deadlines are kept by the caller: medbench's ``--request-timeout`` and
  ``--time-budget``, and the workflow's ``timeout-minutes``. On exit,
  ``medbench-model-run.sh`` stops this process, which frees its GPU memory.

``--engine echo`` replaces the model with a stub that repeats the prompt's
first line. Tests and dry runs use it to check this plumbing without a GPU.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import io
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

CODE_REPO = "https://github.com/uni-medical/UniMedVL"
CODE_COMMIT = "f51aa0accbd02e846e5ea20b595fb39b516dbe1f"
MODEL_REPO = "General-Medical-AI/UniMedVL"
MODEL_REVISION = "29918cef3ce94a13fe1670145c32a3d1198aec7d"
MODEL_NAME = "unimedvl"
# Limits of the upstream release, recorded in /metrics.
CONTEXT_TOKENS = 32768  # llm_config.json max_position_embeddings
IMAGE_LIMITS = {  # codes/data/default.yaml vlm_sft and BagelConfig
    "min_image_size": 378,
    "max_image_size": 980,
    "max_pixels": 2_007_040,
    "vit_max_num_patch_per_side": 70,
}
MAX_IMAGES = 4
MAX_BODY_BYTES = 64 * 1024 * 1024
HTTP_OUT_OF_MEMORY = 507


class RequestError(ValueError):
    """A request this server cannot answer (HTTP 400)."""


@dataclass(frozen=True)
class ChatRequest:
    """The parts of an OpenAI chat request that UniMedVL uses."""

    prompt: str
    images: tuple[bytes, ...]
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class Answer:
    """One answer and its token counts (``None`` when unknown)."""

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


class Engine(Protocol):
    """What the HTTP handler needs from a model."""

    def generate(self, request: ChatRequest) -> Answer:
        """Answer one request."""

    def cleanup(self) -> None:
        """Free cached memory after a failure."""

    def stats(self) -> dict[str, Any]:
        """Versions, limits and resource figures for /metrics."""


def decode_data_url(url: object) -> bytes:
    """Bytes of a base64 ``data:`` URL. Other URLs are refused, not fetched."""
    if not isinstance(url, str) or not url.startswith("data:"):
        message = "only base64 data: image URLs are accepted"
        raise RequestError(message)
    header, _, payload = url.partition(",")
    if not header.endswith(";base64"):
        message = "image data URLs must be base64 encoded"
        raise RequestError(message)
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        message = "image data URL is not valid base64"
        raise RequestError(message) from error


def _parts(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list) and all(isinstance(p, dict) for p in content):
        return content
    message = "message content must be a string or a list of parts"
    raise RequestError(message)


def parse_chat_request(
    body: object, default_max_tokens: int = 512, max_tokens_limit: int = 4096
) -> ChatRequest:
    """Read the text, images and limits of an OpenAI chat request.

    Text parts of all messages are joined in order with blank lines.
    UniMedVL's ``chat`` takes one prompt and puts the images before it.
    """
    if not isinstance(body, dict):
        message = "request body must be a JSON object"
        raise RequestError(message)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        message = "messages must be a non-empty list"
        raise RequestError(message)
    texts: list[str] = []
    images: list[bytes] = []
    for item in messages:
        if not isinstance(item, dict):
            message = "each message must be an object"
            raise RequestError(message)
        for part in _parts(item.get("content")):
            kind = part.get("type")
            if kind == "text":
                texts.append(str(part.get("text", "")))
            elif kind == "image_url":
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                images.append(decode_data_url(url))
            else:
                message = f"unsupported content part {kind!r}"
                raise RequestError(message)
    if len(images) > MAX_IMAGES:
        message = f"at most {MAX_IMAGES} images per request"
        raise RequestError(message)
    try:
        max_tokens = int(body.get("max_tokens") or default_max_tokens)
        temperature = float(body.get("temperature", 0.0) or 0.0)
    except (TypeError, ValueError) as error:
        message = "max_tokens and temperature must be numbers"
        raise RequestError(message) from error
    if not 1 <= max_tokens <= max_tokens_limit:
        message = f"max_tokens must be between 1 and {max_tokens_limit}"
        raise RequestError(message)
    if temperature < 0:
        message = "temperature must not be negative"
        raise RequestError(message)
    prompt = "\n\n".join(text for text in texts if text.strip())
    return ChatRequest(prompt, tuple(images), max_tokens, temperature)


def is_out_of_memory(error: BaseException) -> bool:
    """Whether ``error`` is a CUDA (or host) allocation failure.

    ``torch.cuda.OutOfMemoryError`` is matched by name so that this file can
    be imported without torch.
    """
    if isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(error).lower()


def chat_response(answer: Answer, model: str, created: int) -> dict[str, Any]:
    """OpenAI ``chat.completion`` body for ``answer``."""
    usage: dict[str, int] = {}
    if answer.prompt_tokens is not None:
        usage["prompt_tokens"] = answer.prompt_tokens
    if answer.completion_tokens is not None:
        usage["completion_tokens"] = answer.completion_tokens
    if len(usage) == 2:  # both counts known
        usage["total_tokens"] = sum(usage.values())
    return {
        "id": f"chatcmpl-unimedvl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer.text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    return {name: _package_version(name) for name in names}


def peak_host_rss_bytes() -> int:
    """Peak resident memory of this process (Linux reports KiB, macOS bytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


class EchoEngine:
    """Stand-in model that repeats the first prompt line (no GPU needed)."""

    def __init__(self) -> None:
        """Start counting requests and cleanups."""
        self.requests = 0
        self.cleanups = 0

    def generate(self, request: ChatRequest) -> Answer:
        """Answer with the prompt's first line, cut to ``max_tokens`` words."""
        self.requests += 1
        first = request.prompt.strip().splitlines()[0] if request.prompt else ""
        words = f"echo: {first}".split()[: request.max_tokens]
        return Answer(" ".join(words), len(request.prompt.split()), len(words))

    def cleanup(self) -> None:
        """Count the cleanup."""
        self.cleanups += 1

    def stats(self) -> dict[str, Any]:
        """Counters, so tests can see what happened."""
        return {
            "engine": "echo",
            "requests": self.requests,
            "cleanups": self.cleanups,
            "peak_host_rss_bytes": peak_host_rss_bytes(),
        }


class _RecordingTokenizer:
    """Pass-through tokenizer that keeps the ids ``model.chat`` decodes.

    ``chat`` returns only text. The decoded ids are the start token plus one
    id per decoding step, so their length tells whether ``max_length`` was
    reached (the answer may be cut off) or the model stopped by itself.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.last_length: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __len__(self) -> int:
        return len(self._tokenizer)

    def decode(self, ids: Any, *args: Any, **kwargs: Any) -> Any:
        self.last_length = len(ids)
        return self._tokenizer.decode(ids, *args, **kwargs)


class UniMedVLEngine:  # pragma: no cover - needs a GPU and the 29 GB checkpoint
    """UniMedVL loaded with upstream's own classes, as in its VQA inferencer."""

    def __init__(
        self, code_dir: Path, model_dir: Path, max_gpu_memory: str, seed: int
    ) -> None:
        """Build the model and move it to the GPU.

        Args:
            code_dir: Root of an upstream checkout at :data:`CODE_COMMIT`.
            model_dir: Folder with ``ema_bf16.safetensors``, the two configs
                and the tokenizer files of :data:`MODEL_REVISION`.
            max_gpu_memory: Upper bound for weights on GPU 0, e.g. ``40GiB``
                (upstream's default). Layers that do not fit stay on the CPU.
            seed: Seed for any sampling (temperature > 0).
        """
        started = time.perf_counter()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        # Everything is loaded from the verified local folder; offline mode
        # stops transformers from fetching Hub code a config might name.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # build_transform() opens codes/data/default.yaml relative to the
        # working directory, and the packages live in codes/.
        os.chdir(code_dir)
        sys.path.insert(0, str(code_dir / "codes"))

        import torch
        from accelerate import (
            dispatch_model,
            infer_auto_device_map,
            init_empty_weights,
            load_checkpoint_and_dispatch,
        )
        from data.data_utils import add_special_tokens, pil_img2rgb
        from eval.vlm.utils import build_transform
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.unimedvl import (
            Bagel,
            BagelConfig,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )

        self.torch = torch
        self.pil_img2rgb = pil_img2rgb
        torch.manual_seed(seed)
        if not torch.cuda.is_available():
            message = "UniMedVL needs a CUDA GPU (upstream uses flash-attn)"
            raise RuntimeError(message)

        llm_config = Qwen2Config.from_json_file(str(model_dir / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(
            str(model_dir / "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        config = BagelConfig(
            visual_gen=False,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vit_max_num_patch_per_side=IMAGE_LIMITS["vit_max_num_patch_per_side"],
            connector_act="gelu_pytorch_tanh",
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))
        tokenizer, self.new_token_ids, _ = add_special_tokens(tokenizer)
        self.tokenizer = _RecordingTokenizer(tokenizer)
        self.image_transform = build_transform()

        # Upstream loads the bf16 weights on the CPU first, then dispatches.
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_dir / "ema_bf16.safetensors"),
            device_map={"": "cpu"},
            offload_buffers=False,
            dtype=torch.bfloat16,
            force_hooks=False,
        )
        gc.collect()
        device_map = infer_auto_device_map(
            model,
            max_memory={0: max_gpu_memory},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        first = device_map.get("language_model.model.embed_tokens", 0)
        for name in ("language_model.model.embed_tokens", "connector", "vit_pos_embed"):
            if name in device_map:
                device_map[name] = first
        self.model = dispatch_model(model, device_map=device_map).eval()
        self.cpu_layers = sorted(
            name for name, device in device_map.items() if device in ("cpu", "disk")
        )
        self.load_seconds = time.perf_counter() - started
        self.max_gpu_memory = max_gpu_memory
        self.requests = 0
        self.generation_seconds = 0.0
        self.completion_tokens = 0

    def generate(self, request: ChatRequest) -> Answer:
        """Run ``model.chat`` on the images and prompt."""
        from PIL import Image

        images = [
            self.pil_img2rgb(Image.open(io.BytesIO(data))) for data in request.images
        ]
        started = time.perf_counter()
        text = self.model.chat(
            self.tokenizer,
            self.new_token_ids,
            self.image_transform,
            images=images,
            prompt=request.prompt,
            max_length=request.max_tokens,
            do_sample=request.temperature > 0,
            temperature=request.temperature or 1.0,
        )
        steps = self.tokenizer.last_length
        self.requests += 1
        self.generation_seconds += time.perf_counter() - started
        self.completion_tokens += steps or 0
        # Image tokens are not counted: chat() does not report them.
        prompt_tokens = len(self.tokenizer.encode(request.prompt))
        return Answer(text.strip(), prompt_tokens, steps)

    def cleanup(self) -> None:
        """Drop cached CUDA blocks after a failed request."""
        gc.collect()
        self.torch.cuda.empty_cache()
        self.torch.cuda.ipc_collect()

    def stats(self) -> dict[str, Any]:
        """Versions, limits, load time, peak memory and decoding speed."""
        cuda = self.torch.cuda
        device = cuda.current_device()
        rate = (
            self.completion_tokens / self.generation_seconds
            if self.generation_seconds
            else None
        )
        return {
            "engine": "unimedvl",
            "code": {"repo": CODE_REPO, "commit": CODE_COMMIT},
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "python": platform.python_version(),
            "packages": _package_versions(
                ("torch", "transformers", "accelerate", "flash_attn", "safetensors")
            ),
            "cuda_runtime": self.torch.version.cuda,
            "gpu": cuda.get_device_name(device),
            "gpu_memory_total_bytes": cuda.get_device_properties(device).total_memory,
            "peak_gpu_allocated_bytes": cuda.max_memory_allocated(device),
            "peak_gpu_reserved_bytes": cuda.max_memory_reserved(device),
            "peak_host_rss_bytes": peak_host_rss_bytes(),
            "max_gpu_memory": self.max_gpu_memory,
            "cpu_offloaded_modules": self.cpu_layers,
            "load_seconds": round(self.load_seconds, 2),
            "requests": self.requests,
            "completion_tokens": self.completion_tokens,
            "generation_seconds": round(self.generation_seconds, 2),
            # Prefill and decoding are one call in chat(), so this rate
            # includes the prompt and image encoding time.
            "tokens_per_second_including_prefill": rate,
            "context_tokens": CONTEXT_TOKENS,
            "image_limits": IMAGE_LIMITS,
            "max_images_per_request": MAX_IMAGES,
        }


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP handler; the server object carries the engine."""

    server: ChatServer

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[unimedvl-server] {format % args}\n")

    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, kind: str, message: str) -> None:
        self._send(status, {"error": {"type": kind, "message": message}})

    def do_GET(self) -> None:
        """Health, model list and metrics."""
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": self.server.name}]})
        elif self.path == "/metrics":
            self._send(200, self.server.engine.stats())
        else:
            self._error(404, "not_found", f"no route {self.path}")

    def do_POST(self) -> None:
        """Answer one chat completion."""
        if self.path != "/v1/chat/completions":
            self._error(404, "not_found", f"no route {self.path}")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            self._error(413, "invalid_request", "missing or too large body")
            return
        try:
            body = json.loads(self.rfile.read(length))
            request = parse_chat_request(
                body, max_tokens_limit=self.server.max_tokens_limit
            )
        except (RequestError, ValueError) as error:
            self._error(400, "invalid_request", str(error))
            return
        engine = self.server.engine
        try:
            answer = engine.generate(request)
        except Exception as error:  # report every model failure
            engine.cleanup()
            if is_out_of_memory(error):
                self._error(HTTP_OUT_OF_MEMORY, "out_of_memory", str(error))
            else:
                self._error(500, "model_error", f"{type(error).__name__}: {error}")
            return
        self._send(200, chat_response(answer, self.server.name, int(time.time())))


class ChatServer(HTTPServer):
    """One-request-at-a-time HTTP server around an :class:`Engine`."""

    def __init__(
        self,
        address: tuple[str, int],
        engine: Engine,
        name: str = MODEL_NAME,
        max_tokens_limit: int = 4096,
    ) -> None:
        """Bind ``address`` and serve ``engine``."""
        super().__init__(address, ChatHandler)
        self.engine = engine
        self.name = name
        self.max_tokens_limit = max_tokens_limit


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--engine", choices=("unimedvl", "echo"), default="unimedvl")
    parser.add_argument("--code-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--max-gpu-memory", default="40GiB")
    parser.add_argument("--max-tokens-limit", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the engine, then serve until stopped."""
    args = build_parser().parse_args(argv)
    engine: Engine
    if args.engine == "echo":
        engine = EchoEngine()
    else:  # pragma: no cover - needs a GPU
        if args.model_dir is None:
            print("--model-dir is required for --engine unimedvl", file=sys.stderr)
            return 2
        engine = UniMedVLEngine(
            args.code_dir.resolve(),
            args.model_dir.resolve(),
            args.max_gpu_memory,
            args.seed,
        )
    server = ChatServer(
        (args.host, args.port), engine, MODEL_NAME, args.max_tokens_limit
    )
    print(f"[unimedvl-server] listening on {args.host}:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
