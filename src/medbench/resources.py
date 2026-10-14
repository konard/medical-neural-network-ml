"""Back-of-the-envelope resource calculator for inference and training.

All formulas are deliberately simple so that a reader without a machine
learning background can check them by hand. They are explained, with worked
examples, in ``docs/case-studies/issue-1/resources.md``.

* Weights take ``parameters * bits_per_weight / 8`` bytes.
* The KV cache takes ``2 * layers * kv_heads * head_dim * bytes * tokens``.
* Generating one token reads every active weight once, so decoding speed is
  limited by memory bandwidth: ``tokens/s ~ bandwidth / bytes_per_token``.
* Reading the prompt (prefill) is limited by compute:
  ``tokens/s ~ FLOPS / (2 * active_params)``.
* Training costs about ``6 * parameters * training_tokens`` FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass

from medbench.models import ModelSpec

__all__ = [
    "BITS_PER_WEIGHT",
    "HARDWARE",
    "FineTuneEstimate",
    "HardwareProfile",
    "InferenceEstimate",
    "TrainingEstimate",
    "estimate_finetune_memory",
    "estimate_inference",
    "estimate_training",
    "kv_cache_bytes",
    "weights_bytes",
]

GB = 1e9

# Average bits stored per weight, including quantization scales.
# GGUF values are the llama.cpp averages for the format; real files are a bit
# larger because embeddings and norms stay at higher precision.
BITS_PER_WEIGHT: dict[str, float] = {
    "fp32": 32.0,
    "bf16": 16.0,
    "fp16": 16.0,
    "int8": 8.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.69,
    "Q4_K_M": 4.89,
    "nf4": 4.13,
    "Q3_K_M": 3.91,
    "Q2_K": 2.96,
}


@dataclass(frozen=True)
class HardwareProfile:
    """Capabilities of one machine.

    Attributes:
        key: Short name.
        description: Human readable description.
        memory_gb: Memory available for the model (RAM for CPU, VRAM for GPU).
        bandwidth_gbs: Memory bandwidth in GB/s.
        compute_tflops: Peak dense compute for the dtype normally used.
        decode_efficiency: Share of peak bandwidth reached while decoding.
        prefill_efficiency: Share of peak compute reached while reading prompts.
        is_gpu: Whether this is a GPU.
        price_per_hour: Typical rental price in USD (0 for owned hardware).
    """

    key: str
    description: str
    memory_gb: float
    bandwidth_gbs: float
    compute_tflops: float
    decode_efficiency: float = 0.6
    prefill_efficiency: float = 0.4
    is_gpu: bool = False
    price_per_hour: float = 0.0


HARDWARE: dict[str, HardwareProfile] = {
    profile.key: profile
    for profile in (
        HardwareProfile(
            key="dev-vm-6cpu",
            description=(
                "Development VM used for this study: 6 AMD EPYC vCPUs (AVX2), "
                "11 GB RAM. Efficiencies calibrated with llama-bench."
            ),
            memory_gb=11.0,
            bandwidth_gbs=20.0,
            compute_tflops=0.48,
            decode_efficiency=0.6,
            prefill_efficiency=0.25,
        ),
        HardwareProfile(
            key="github-ubuntu-4cpu",
            description="GitHub-hosted ubuntu-latest runner: 4 vCPU, 16 GB RAM.",
            memory_gb=16.0,
            bandwidth_gbs=20.0,
            compute_tflops=0.32,
            decode_efficiency=0.6,
            prefill_efficiency=0.25,
        ),
        HardwareProfile(
            key="desktop-ddr5-16c",
            description="Desktop CPU, 16 cores, dual-channel DDR5-6000, 64 GB RAM.",
            memory_gb=64.0,
            bandwidth_gbs=90.0,
            compute_tflops=2.0,
            decode_efficiency=0.65,
            prefill_efficiency=0.3,
        ),
        HardwareProfile(
            key="t4-16gb",
            description="NVIDIA T4 16 GB (GitHub GPU runner, Colab free tier).",
            memory_gb=16.0,
            bandwidth_gbs=320.0,
            compute_tflops=65.0,
            is_gpu=True,
            price_per_hour=0.35,
        ),
        HardwareProfile(
            key="l4-24gb",
            description="NVIDIA L4 24 GB.",
            memory_gb=24.0,
            bandwidth_gbs=300.0,
            compute_tflops=121.0,
            is_gpu=True,
            price_per_hour=0.8,
        ),
        HardwareProfile(
            key="rtx4090-24gb",
            description="NVIDIA GeForce RTX 4090 24 GB.",
            memory_gb=24.0,
            bandwidth_gbs=1008.0,
            compute_tflops=165.0,
            is_gpu=True,
            price_per_hour=0.4,
        ),
        HardwareProfile(
            key="a100-80gb",
            description="NVIDIA A100 80 GB SXM.",
            memory_gb=80.0,
            bandwidth_gbs=2039.0,
            compute_tflops=312.0,
            is_gpu=True,
            price_per_hour=1.5,
        ),
        HardwareProfile(
            key="h100-80gb",
            description="NVIDIA H100 80 GB SXM.",
            memory_gb=80.0,
            bandwidth_gbs=3350.0,
            compute_tflops=989.0,
            is_gpu=True,
            price_per_hour=2.5,
        ),
    )
}


def weights_bytes(params: float, quant: str) -> float:
    """Return the bytes needed to store ``params`` weights in format ``quant``.

    Args:
        params: Number of parameters.
        quant: Key of :data:`BITS_PER_WEIGHT`.

    Returns:
        Size in bytes.

    Raises:
        KeyError: If the quantization format is unknown.
    """
    if quant not in BITS_PER_WEIGHT:
        message = f"unknown quantization {quant!r}"
        raise KeyError(message)
    return params * BITS_PER_WEIGHT[quant] / 8


def kv_cache_bytes(model: ModelSpec, tokens: int, bytes_per_value: float = 2) -> float:
    """Return the KV cache size for ``tokens`` tokens of context.

    Sliding-window layers (Gemma 3 uses five local layers per global one) only
    keep the last ``sliding_window`` tokens, which is what makes long contexts
    affordable for MedGemma.

    Args:
        model: Model specification.
        tokens: Context length in tokens.
        bytes_per_value: 2 for fp16/bf16 cache, 1 for an 8-bit cache.

    Returns:
        Size in bytes.
    """
    layout = model.attention
    per_token_per_layer = 2 * layout.num_kv_heads * layout.head_dim * bytes_per_value
    if layout.sliding_window is None or layout.global_layers is None:
        return per_token_per_layer * layout.num_layers * tokens
    local_layers = layout.num_layers - layout.global_layers
    local_tokens = min(tokens, layout.sliding_window)
    cached = layout.global_layers * tokens + local_layers * local_tokens
    return per_token_per_layer * cached


@dataclass(frozen=True)
class InferenceEstimate:
    """Result of :func:`estimate_inference` (sizes in GB, speeds in tokens/s)."""

    model: str
    hardware: str
    quant: str
    weights_gb: float
    kv_cache_gb: float
    total_memory_gb: float
    fits: bool
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    seconds_per_request: float

    def as_dict(self) -> dict[str, object]:
        """Return the estimate as a JSON-friendly dictionary."""
        return dict(self.__dict__)


def estimate_inference(
    model: ModelSpec,
    hardware: HardwareProfile,
    quant: str = "bf16",
    prompt_tokens: int = 512,
    output_tokens: int = 256,
    images: int = 0,
    overhead_gb: float = 1.0,
) -> InferenceEstimate:
    """Estimate memory use and speed of one request.

    Args:
        model: Model specification.
        hardware: Machine to run on.
        quant: Weight format, a key of :data:`BITS_PER_WEIGHT`.
        prompt_tokens: Text tokens in the prompt.
        output_tokens: Tokens to generate.
        images: Number of images in the prompt.
        overhead_gb: Runtime buffers, activations and framework overhead.

    Returns:
        The estimate.
    """
    context = prompt_tokens + images * model.image_tokens + output_tokens
    weights = weights_bytes(model.total_params, quant)
    kv_cache = kv_cache_bytes(model, context)
    total = weights + kv_cache + overhead_gb * GB

    active_bytes = weights_bytes(model.active_params, quant)
    bandwidth = hardware.bandwidth_gbs * GB * hardware.decode_efficiency
    decode = bandwidth / (active_bytes + kv_cache)
    flops = hardware.compute_tflops * 1e12 * hardware.prefill_efficiency
    prefill = flops / (2 * model.active_params)
    vision_flops = images * 2 * model.vision_params * 4096
    seconds = (
        (prompt_tokens + images * model.image_tokens) / prefill
        + vision_flops / flops
        + output_tokens / decode
    )
    return InferenceEstimate(
        model=model.key,
        hardware=hardware.key,
        quant=quant,
        weights_gb=weights / GB,
        kv_cache_gb=kv_cache / GB,
        total_memory_gb=total / GB,
        fits=total / GB <= hardware.memory_gb,
        prefill_tokens_per_s=prefill,
        decode_tokens_per_s=decode,
        seconds_per_request=seconds,
    )


@dataclass(frozen=True)
class FineTuneEstimate:
    """Memory needed to fine-tune a model (GB)."""

    method: str
    weights_gb: float
    trainable_params: float
    optimizer_and_grads_gb: float
    activations_gb: float
    total_gb: float

    def as_dict(self) -> dict[str, object]:
        """Return the estimate as a JSON-friendly dictionary."""
        return dict(self.__dict__)


def _lora_params(model: ModelSpec, rank: int, hidden_size: int) -> float:
    """Approximate LoRA parameters for the attention and MLP projections.

    Seven projection matrices per layer (q, k, v, o, gate, up, down); each one
    gets two thin matrices of shape ``hidden x rank``. MLP matrices are wider
    than ``hidden``, which the factor 1.5 accounts for on average.
    """
    return model.attention.num_layers * 7 * 2 * hidden_size * rank * 1.5


def estimate_finetune_memory(
    model: ModelSpec,
    method: str,
    hidden_size: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    lora_rank: int = 16,
    gradient_checkpointing: bool = True,
) -> FineTuneEstimate:
    """Estimate GPU memory for full fine-tuning, LoRA or QLoRA.

    Full fine-tuning with AdamW in mixed precision keeps, per parameter,
    2 bytes of bf16 weights, 2 bytes of gradients, 4 bytes of fp32 master
    weights and 8 bytes of Adam moments: 16 bytes. LoRA freezes the model and
    trains small adapters; QLoRA additionally stores the frozen model in 4 bits.

    Args:
        model: Model specification.
        method: ``"full"``, ``"lora"`` or ``"qlora"``.
        hidden_size: Hidden size of the language model.
        batch_size: Sequences per step.
        seq_len: Tokens per sequence.
        lora_rank: LoRA rank.
        gradient_checkpointing: Recompute activations instead of storing them.

    Returns:
        The estimate.

    Raises:
        ValueError: If the method is unknown.
    """
    params = model.total_params
    if method == "full":
        weights = params * 2
        trainable = params
    elif method in {"lora", "qlora"}:
        weights = params * (2 if method == "lora" else BITS_PER_WEIGHT["nf4"] / 8)
        trainable = _lora_params(model, lora_rank, hidden_size)
    else:
        message = f"unknown fine-tuning method {method!r}"
        raise ValueError(message)
    optimizer = trainable * (14 if method == "full" else 16)
    tokens = batch_size * seq_len
    layers = model.attention.num_layers
    # One bf16 hidden state per layer is kept with checkpointing; without it a
    # transformer layer stores roughly 34 values per hidden unit and token.
    per_layer = 2 * hidden_size * tokens * (1 if gradient_checkpointing else 34)
    activations = per_layer * layers + 34 * 2 * hidden_size * tokens
    total = weights + optimizer + activations + 1.5 * GB
    return FineTuneEstimate(
        method=method,
        weights_gb=weights / GB,
        trainable_params=trainable,
        optimizer_and_grads_gb=optimizer / GB,
        activations_gb=activations / GB,
        total_gb=total / GB,
    )


@dataclass(frozen=True)
class TrainingEstimate:
    """Compute, time and cost of a training run."""

    flops: float
    gpu_hours: float
    wall_clock_hours: float
    cost_usd: float

    def as_dict(self) -> dict[str, object]:
        """Return the estimate as a JSON-friendly dictionary."""
        return dict(self.__dict__)


def estimate_training(
    params: float,
    tokens: float,
    hardware: HardwareProfile,
    gpus: int = 1,
    mfu: float = 0.35,
    method: str = "full",
) -> TrainingEstimate:
    """Estimate compute, time and money for training.

    Uses ``6 * N * D`` FLOPs for full training (forward 2ND, backward 4ND).
    LoRA skips the weight gradients of the frozen model: about ``4 * N * D``.

    Args:
        params: Model parameters ``N``.
        tokens: Training tokens ``D`` (epochs times dataset tokens).
        hardware: GPU type.
        gpus: Number of GPUs.
        mfu: Model FLOPs utilization, the share of peak compute achieved.
        method: ``"full"`` or ``"lora"``.

    Returns:
        The estimate.
    """
    factor = 6 if method == "full" else 4
    flops = factor * params * tokens
    per_gpu = hardware.compute_tflops * 1e12 * mfu
    gpu_hours = flops / per_gpu / 3600
    return TrainingEstimate(
        flops=flops,
        gpu_hours=gpu_hours,
        wall_clock_hours=gpu_hours / gpus,
        cost_usd=gpu_hours * hardware.price_per_hour,
    )
