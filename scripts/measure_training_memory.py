#!/usr/bin/env python3
"""Measure the peak memory of real fine-tuning steps (issue #9).

``medbench estimate --finetune`` adds up weights, optimizer state and
activations from a formula. This script runs a few real optimizer steps
instead and writes one JSON report. The report records every input that
changes the answer: method, batch, sequence length, image size, adapter
rank, optimizer, precision, gradient checkpointing, device and library
versions. It measures:

- the model-loading peak (host RSS and, on CUDA, allocated and reserved
  memory);
- the training peak over all steps, and the allocator's share of it
  (reserved minus allocated);
- gradient and optimizer-state bytes;
- the image tokens the processor actually inserted;
- the disk size of a resumable checkpoint (trainable weights plus optimizer
  state).

The same inputs are also run through ``estimate_finetune_memory``, so every
report includes measured/estimated ratios.

Before loading anything, the high end of the estimate is compared with the
free memory of the device. Without ``--force`` the script refuses when it
would leave less than ``--headroom`` free. ``--deadline`` stops between
steps. An out-of-memory error, a refusal or an unsupported combination still
writes a report, with ``status`` saying which.

Requires ``torch``, ``transformers`` and ``peft``. QLoRA also needs
``bitsandbytes`` and a CUDA GPU. These are not dependencies of ``medbench``.
Results from this PyTorch script say nothing about MLX training on Apple
silicon.

Example on a CUDA host::

    python scripts/measure_training_memory.py --model google/medgemma-4b-it \\
        --revision <commit> --method qlora --seq-len 2048 --image-size 896 \\
        --out medbench-reports/train-medgemma-4b-qlora.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import shutil
import struct
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GB = 1e9
METHODS = ("full", "lora", "qlora")
DTYPES = ("bf16", "fp16", "fp32")
# The projections ``estimate_finetune_memory`` assumes LoRA adapts.
LORA_TARGETS = (
    *("q_proj", "k_proj", "v_proj", "o_proj"),
    *("gate_proj", "up_proj", "down_proj"),
)
EXIT_REFUSED = 3
EXIT_UNSUPPORTED = 4
EXIT_OUT_OF_MEMORY = 5
EXIT_DEADLINE = 6
SAFETENSORS_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I8": 1, "U8": 1}


@dataclass(frozen=True)
class TrainingConfig:
    """Every input that changes the measured peak."""

    model: str
    revision: str | None
    method: str
    batch_size: int = 1
    seq_len: int = 512
    image_size: int = 0
    lora_rank: int = 16
    lora_alpha: int = 32
    optimizer: str = "torch.optim.AdamW, fp32 moments"
    learning_rate: float = 1e-5
    weights_dtype: str = "bf16"
    autocast_dtype: str | None = "bf16"
    gradient_checkpointing: bool = True
    steps: int = 3
    device: str = "auto"
    headroom: float = 0.1
    deadline_s: float = 3600.0
    # Image processor settings such as SmolVLM's ``do_image_splitting``: the
    # image token count follows them, not the input image size.
    processor_args: dict[str, Any] = field(default_factory=dict)


class MeasurementStopped(Exception):  # noqa: N818 - a status, not a bug
    """A measurement ended early with a status for the report."""

    def __init__(self, reason: str, status: str, exit_code: int) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.exit_code = exit_code


def gigabytes(size: float) -> float:
    """Convert bytes to GB, keeping kilobytes so tiny test models stay non-zero."""
    return round(size / GB, 6)


def host_peak_rss_gb() -> float:
    """Return this process's peak resident memory (high-water mark) in GB."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / GB if sys.platform == "darwin" else peak * 1024 / GB


def host_memory() -> dict[str, float | None]:
    """Return the memory this process may still use, cgroup limits included.

    ``MemAvailable`` alone overstates it inside a container: this repository's
    development VM reports 9 GB available under a 3.1 GB cgroup limit.
    """
    from medbench.hardware import detect_host

    host = detect_host(with_gpus=False)
    return {
        "host_available_gb": round(host.memory_available_gb, 3),
        "cgroup_memory_limit_gb": host.cgroup_memory_limit_gb,
    }


def directory_bytes(path: Path) -> int:
    """Return the total size of the regular files below ``path``."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def safetensors_parameters(path: Path) -> int:
    """Count the parameters in a ``.safetensors`` file from its header only."""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
    return sum(
        math.prod(tensor["shape"])
        for name, tensor in header.items()
        if name != "__metadata__"
    )


def preflight(required_gb: float, free_gb: float | None, headroom: float) -> str | None:
    """Return why the run is refused, or ``None`` when it may start.

    ``required_gb`` is the high end of the estimate; at least ``headroom``
    (a fraction) of the free memory must remain unused.
    """
    if free_gb is None:
        return "free memory of the device is unknown; pass --force to run anyway"
    usable = free_gb * (1 - headroom)
    if required_gb > usable:
        return (
            f"the estimate's high end {required_gb:.2f} GB exceeds {usable:.2f} GB "
            f"({free_gb:.2f} GB free minus {headroom:.0%} headroom)"
        )
    return None


def text_config(config: Any) -> Any:
    """Return the language-model part of a (possibly multimodal) config."""
    return getattr(config, "text_config", None) or config


def model_spec(config: Any, name: str, params: float, image: bool) -> Any:
    """Build the ``medbench`` model description that the estimator needs."""
    from medbench.models import AttentionLayout, ModelSpec

    text = text_config(config)
    heads = text.num_attention_heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // heads
    attention = AttentionLayout(
        num_layers=text.num_hidden_layers,
        num_kv_heads=getattr(text, "num_key_value_heads", None) or heads,
        head_dim=head_dim,
    )
    return ModelSpec(
        key=name,
        hf_id=name,
        total_params=params,
        active_params=params,
        attention=attention,
        modalities=("text", "image") if image else ("text",),
        license="see the model card",
        gated=False,
    )


def estimate(config: TrainingConfig, model_config: Any, params: float) -> Any:
    """Return ``estimate_finetune_memory`` for exactly these inputs."""
    from medbench.resources import estimate_finetune_memory

    spec = model_spec(model_config, config.model, params, config.image_size > 0)
    text = text_config(model_config)
    return estimate_finetune_memory(
        spec,
        config.method,
        text.hidden_size,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        lora_rank=config.lora_rank,
        gradient_checkpointing=config.gradient_checkpointing,
        vocab_size=getattr(text, "vocab_size", 0),
    )


def resolve_device(requested: str, torch: Any) -> str:
    """Pick CUDA, then Apple MPS, then CPU for ``auto``."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unsupported_reason(config: TrainingConfig, device: str) -> str | None:
    """Say why this combination cannot be measured with this script."""
    if config.method == "qlora" and device != "cuda":
        return (
            "QLoRA here means bitsandbytes NF4, which needs a CUDA GPU; "
            "MLX quantized LoRA is a different implementation"
        )
    if config.method == "qlora" and not _has_module("bitsandbytes"):
        return "QLoRA needs bitsandbytes, which is not installed"
    if config.autocast_dtype == "bf16" and device == "mps":
        return "bf16 autocast is not available on MPS; use --autocast fp16 or none"
    return None


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _gpu(torch: Any, device: str) -> dict[str, float | int | None]:
    if device != "cuda":
        return {}
    stats = torch.cuda.memory_stats()
    allocated = torch.cuda.max_memory_allocated() / GB
    reserved = torch.cuda.max_memory_reserved() / GB
    return {
        "gpu_peak_allocated_gb": round(allocated, 3),
        "gpu_peak_reserved_gb": round(reserved, 3),
        "allocator_overhead_gb": round(reserved - allocated, 3),
        "allocation_retries": stats.get("num_alloc_retries"),
        "out_of_memory_events": stats.get("num_ooms"),
    }


def _inputs(
    config: TrainingConfig, model: Any, processor: Any, torch: Any
) -> dict[str, Any]:
    """Return a batch of exactly ``seq_len`` tokens, ``image_tokens`` of them image."""
    vocab = text_config(model.config).vocab_size
    generator = torch.Generator().manual_seed(0)
    if config.image_size == 0:
        shape = (config.batch_size, config.seq_len)
        ids = torch.randint(100, vocab - 1, shape, generator=generator)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    from PIL import Image

    image = Image.new("RGB", (config.image_size, config.image_size), (90, 90, 90))
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Describe."}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(
        text=[prompt] * config.batch_size,
        images=[[image]] * config.batch_size,
        return_tensors="pt",
        **config.processor_args,
    )
    ids = batch["input_ids"]
    missing = config.seq_len - ids.shape[1]
    if missing < 0:
        message = (
            f"--seq-len {config.seq_len} is shorter than the {ids.shape[1]} "
            f"tokens of one image prompt ({image_token_count(batch, model)} of "
            "them image tokens)"
        )
        raise MeasurementStopped(message, "refused", EXIT_REFUSED)
    shape = (config.batch_size, missing)
    filler = torch.randint(100, vocab - 1, shape, generator=generator)
    batch["input_ids"] = torch.cat([ids, filler], dim=1)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    return dict(batch)


def image_token_count(batch: dict[str, Any], model: Any) -> int | None:
    """Count the image placeholder tokens in the first sequence."""
    config = model.config
    token = getattr(config, "image_token_id", None)
    token = token if token is not None else getattr(config, "image_token_index", None)
    if token is None or "pixel_values" not in batch:
        return None
    return int((batch["input_ids"][0] == token).sum())


def tensor_bytes(tensors: Any) -> int:
    """Sum the storage of an iterable of tensors (``None`` entries skipped)."""
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def optimizer_state_bytes(optimizer: Any, torch: Any) -> int:
    """Sum every tensor kept in the optimizer state (Adam moments, steps)."""
    return tensor_bytes(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _load(config: TrainingConfig, device: str, torch: Any) -> tuple[Any, Any]:
    import transformers

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    kwargs: dict[str, Any] = {
        "revision": config.revision,
        "dtype": dtype[config.weights_dtype],
        "low_cpu_mem_usage": True,
    }
    if config.method == "qlora":
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype[config.autocast_dtype or "bf16"],
        )
        kwargs["device_map"] = {"": 0}
    processor = None
    if config.image_size:
        processor = transformers.AutoProcessor.from_pretrained(
            config.model, revision=config.revision
        )
        auto = transformers.AutoModelForImageTextToText
    else:
        auto = transformers.AutoModelForCausalLM
    model = auto.from_pretrained(config.model, **kwargs)
    if config.method != "qlora":
        model.to(device)
    return model, processor


def _prepare(config: TrainingConfig, model: Any) -> Any:
    if config.method == "full":
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model
    import peft

    if config.method == "qlora":
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = peft.LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        target_modules=list(LORA_TARGETS),
        task_type="CAUSAL_LM",
    )
    return peft.get_peft_model(model, lora)


def measure(config: TrainingConfig, report: dict[str, Any]) -> None:
    """Load, train ``steps`` steps and save a checkpoint, filling ``report``."""
    import torch

    started = time.monotonic()
    device = resolve_device(config.device, torch)
    report["device"] = {"type": device, **host_memory()}
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        report["device"].update(
            name=torch.cuda.get_device_name(),
            free_gb=gigabytes(free),
            total_gb=gigabytes(total),
        )
    reason = unsupported_reason(config, device)
    if reason:
        raise MeasurementStopped(reason, "unsupported", EXIT_UNSUPPORTED)
    _check_preflight(config, report, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, processor = _load(config, device, torch)
    report["load"] = {
        "seconds": round(time.monotonic() - started, 1),
        "host_peak_rss_gb": round(host_peak_rss_gb(), 3),
        **_gpu(torch, device),
    }
    model = _prepare(config, model)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    batch = _inputs(config, model, processor, torch)
    report["inputs"] = {
        "sequence_tokens": int(batch["input_ids"].shape[1]),
        "image_tokens": image_token_count(batch, model),
    }
    batch = {key: value.to(device) for key, value in batch.items()}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step_seconds, gradients = [], 0
    autocast = config.autocast_dtype
    for _ in range(config.steps):
        if time.monotonic() - started > config.deadline_s:
            message = f"deadline of {config.deadline_s:.0f} s reached"
            raise MeasurementStopped(message, "deadline", EXIT_DEADLINE)
        tick = time.monotonic()
        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16 if autocast == "bf16" else torch.float16,
            enabled=autocast is not None,
        ):
            loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        gradients = tensor_bytes(p.grad for p in trainable)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        step_seconds.append(round(time.monotonic() - tick, 2))
    trainable_params = sum(p.numel() for p in trainable)
    report["train"] = {
        "step_seconds": step_seconds,
        "loss": round(float(loss.detach()), 4),
        "host_peak_rss_gb": round(host_peak_rss_gb(), 3),
        "trainable_params": trainable_params,
        "all_params": sum(p.numel() for p in model.parameters()),
        "weights_gb": gigabytes(tensor_bytes(model.parameters())),
        "gradients_gb": gigabytes(gradients),
        "optimizer_state_gb": gigabytes(optimizer_state_bytes(optimizer, torch)),
        **_gpu(torch, device),
    }
    report["checkpoint"] = _checkpoint(model, optimizer, torch)
    report["status"] = "measured"


def _check_preflight(
    config: TrainingConfig, report: dict[str, Any], device: str
) -> None:
    expected = report.get("estimate", {}).get("total_gb_range")
    if expected is None or report["config"].get("force"):
        return
    free = report["device"].get("free_gb" if device == "cuda" else "host_available_gb")
    reason = preflight(expected[1], free, config.headroom)
    if reason:
        raise MeasurementStopped(reason, "refused", EXIT_REFUSED)


def _checkpoint(model: Any, optimizer: Any, torch: Any) -> dict[str, float]:
    """Save trainable weights and optimizer state, then measure and delete them."""
    directory = Path(tempfile.mkdtemp(prefix="medbench-train-ckpt-"))
    try:
        model.save_pretrained(directory / "model")
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        return {
            "model_gb": gigabytes(directory_bytes(directory / "model")),
            "optimizer_gb": gigabytes((directory / "optimizer.pt").stat().st_size),
            "total_gb": gigabytes(directory_bytes(directory)),
        }
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def versions() -> dict[str, str | None]:
    """Return the versions of the libraries that change the measured peak."""
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        found[name] = _version(name)
    return found


def _version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def count_parameters(config: TrainingConfig) -> float | None:
    """Count parameters from the safetensors headers, without loading weights."""
    try:
        from huggingface_hub import snapshot_download

        local = Path(config.model)
        if not local.is_dir():
            local = Path(
                snapshot_download(
                    config.model,
                    revision=config.revision,
                    allow_patterns=["*.json", "*.safetensors"],
                )
            )
        files = sorted(local.glob("*.safetensors"))
        return float(sum(safetensors_parameters(path) for path in files)) or None
    except Exception:  # the count only feeds the preflight
        return None


def _rounded_estimate(value: Any) -> dict[str, Any]:
    data = value.as_dict()
    return {
        key: round(item, 6)
        if isinstance(item, float)
        else [round(part, 6) for part in item]
        if isinstance(item, tuple)
        else item
        for key, item in data.items()
    }


def compare(report: dict[str, Any]) -> dict[str, float] | None:
    """Return measured/estimated ratios once both sides exist."""
    expected = report.get("estimate")
    train = report.get("train")
    if not expected or not train:
        return None
    peak = train.get("gpu_peak_reserved_gb") or train["host_peak_rss_gb"]
    ratios = {"peak_to_estimate": round(peak / expected["total_gb"], 3)}
    if expected["trainable_params"]:
        ratios["trainable_params"] = round(
            train["trainable_params"] / expected["trainable_params"], 3
        )
    checkpoint = report.get("checkpoint", {}).get("total_gb")
    if checkpoint and expected["checkpoint_disk_gb"]:
        ratios["checkpoint"] = round(checkpoint / expected["checkpoint_disk_gb"], 3)
    return ratios


def run(config: TrainingConfig, out: Path, *, force: bool = False) -> int:
    """Measure ``config``, write ``out`` and return the exit code."""
    report: dict[str, Any] = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {**asdict(config), "force": force},
        "versions": versions(),
        "platform": platform.platform(),
    }
    exit_code = 0
    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            config.model, revision=config.revision
        )
        report["resolved_revision"] = getattr(model_config, "_commit_hash", None)
        params = count_parameters(config)
        if params:
            report["estimate"] = _rounded_estimate(
                estimate(config, model_config, params)
            )
        # A run killed from outside (the kernel OOM killer, a cgroup limit)
        # leaves this report behind with ``"status": "started"``.
        _write(out, report)
        measure(config, report)
    except MeasurementStopped as stop:
        report.update(status=stop.status, reason=stop.reason)
        exit_code = stop.exit_code
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        report.update(status="out-of-memory", reason=str(error).splitlines()[0])
        exit_code = EXIT_OUT_OF_MEMORY
    finally:
        report["ratios"] = compare(report)
        _write(out, report)
    return exit_code


def is_out_of_memory(error: BaseException) -> bool:
    """Recognize CUDA, MPS and CPU allocation failures."""
    text = str(error)
    return (
        isinstance(error, MemoryError)
        or type(error).__name__ == "OutOfMemoryError"
        or "can't allocate memory" in text
        or "MPS backend out of memory" in text
    )


def _write(out: Path, report: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None) -> tuple[TrainingConfig, Path, bool]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Hub id or local directory")
    parser.add_argument("--revision", help="commit to pin (recommended)")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--image-size", type=int, default=0, help="square image side, 0 = text"
    )
    parser.add_argument(
        "--processor-arg",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="image processor setting, e.g. do_image_splitting=false",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--weights-dtype",
        choices=DTYPES,
        help="default: fp32 master weights for full, bf16 for LoRA/QLoRA",
    )
    parser.add_argument("--autocast", choices=(*DTYPES[:2], "none"), default="bf16")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--headroom", type=float, default=0.1)
    parser.add_argument("--deadline", type=float, default=3600.0, help="seconds")
    parser.add_argument("--force", action="store_true", help="skip the preflight")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    weights = args.weights_dtype or ("fp32" if args.method == "full" else "bf16")
    config = TrainingConfig(
        model=args.model,
        revision=args.revision,
        method=args.method,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        weights_dtype=weights,
        autocast_dtype=None if args.autocast == "none" else args.autocast,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        steps=args.steps,
        device=args.device,
        headroom=args.headroom,
        deadline_s=args.deadline,
        processor_args=dict(_processor_arg(item) for item in args.processor_arg),
    )
    return config, args.out, args.force


def _processor_arg(item: str) -> tuple[str, Any]:
    """Split ``KEY=VALUE``; the value is JSON when it parses, else a string."""
    key, _, raw = item.partition("=")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def main(argv: list[str] | None = None) -> int:
    config, out, force = parse_args(argv)
    if os.fspath(ROOT / "src") not in sys.path and not _has_module("medbench"):
        sys.path.insert(0, os.fspath(ROOT / "src"))
    exit_code = run(config, out, force=force)
    print(out.read_text(encoding="utf-8"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
