"""Back-of-the-envelope resource calculator for inference and training.

All formulas are deliberately simple so that a reader without a machine
learning background can check them by hand. They are explained, with worked
examples, in ``docs/case-studies/issue-1/measurements.md``. These are
capacity estimates, not proof that a runtime supports a given model/format.

* Weights take ``parameters * bits_per_weight / 8`` bytes.
* The KV cache takes ``2 * layers * kv_heads * head_dim * bytes * tokens``.
* Generating one token reads every active weight once, so decoding speed is
  limited by memory bandwidth: ``tokens/s ~ bandwidth / bytes_per_token``.
* Reading the prompt (prefill) is limited by compute:
  ``tokens/s ~ FLOPS / (2 * active_params)``.
* Training costs about ``6 * parameters * training_tokens`` FLOPs.

Every point estimate also comes with a low/high range (see
:data:`MEMORY_RANGE` and :attr:`HardwareProfile.speed_range`). The ranges are
labelled hypotheses until a measured report narrows them; whether a runtime
can load the model at all is answered by :mod:`medbench.support`.
"""

from __future__ import annotations

from dataclasses import dataclass

from medbench.models import ModelSpec

__all__ = [
    "BITS_PER_WEIGHT",
    "FINETUNE_RANGE",
    "HARDWARE",
    "MEMORY_RANGE",
    "PRICES_CHECKED",
    "SCALING_EFFICIENCY",
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
    # MLX affine 4-bit: 4 bits plus an fp16 scale and bias per 64 weights.
    "mlx-4bit": 4.5,
}

# Multipliers for the low/high memory range around the point estimate. The
# high side covers what the simple sum ignores: allocator fragmentation,
# temporary buffers while weights load or are dequantized, and vision-encoder
# activations. They are hypotheses to be replaced by measured peaks.
MEMORY_RANGE = (0.9, 1.3)
FINETUNE_RANGE = (0.8, 1.5)

# Share of linear speed-up kept per added GPU when one job is spread over
# several GPUs of one node (point, low, high). Gradient all-reduce, sharded
# optimizer gathers (ZeRO/FSDP) and stragglers cost the rest. A hypothesis
# until a multi-GPU run of this repository measures it; across nodes without
# InfiniBand it can be much lower.
SCALING_EFFICIENCY = (0.8, 0.6, 0.95)

# Date the rental prices and their sources below were read.
PRICES_CHECKED = "2026-10-16"


@dataclass(frozen=True)
class HardwareProfile:
    """Capabilities of one machine.

    Attributes:
        key: Short name.
        description: Human readable description.
        memory_gb: Memory available for the model (RAM for CPU, VRAM for GPU).
        bandwidth_gbs: Memory bandwidth in GB/s.
        compute_tflops: Peak *dense* compute for the dtype normally used.
            NVIDIA sheets often print 2:4 structured-sparsity figures, which
            are twice as high and never reached by these models.
        sparse_tflops: The sparsity figure from the sheet, for reference only;
            no estimate uses it.
        decode_efficiency: Share of peak bandwidth reached while decoding.
        prefill_efficiency: Share of peak compute reached while reading prompts.
        vision_efficiency: Share of peak compute reached by the image encoder;
            ``None`` means the same as ``prefill_efficiency``. On CPU the
            llama.cpp image encoder is much slower than the text model.
        is_gpu: Whether this is a GPU.
        price_per_hour: On-demand rental price in USD per GPU-hour (0 for
            owned hardware), read from ``price_source`` on
            :data:`PRICES_CHECKED`. Prices change; pass your own with
            ``medbench estimate --price-per-hour``.
        price_range: Lowest and highest on-demand price seen on the same day
            across the providers listed in ``price_source``.
        price_source: Where ``price_per_hour`` comes from.
        speed_range: Multipliers for the low/high speed range. Calibrated
            CPU profiles use the spread between repeated runs; the others
            use a wide uncalibrated range.
        spec_source: Where the compute and bandwidth figures come from.
    """

    key: str
    description: str
    memory_gb: float
    bandwidth_gbs: float
    compute_tflops: float
    decode_efficiency: float = 0.6
    prefill_efficiency: float = 0.4
    vision_efficiency: float | None = None
    is_gpu: bool = False
    price_per_hour: float = 0.0
    price_range: tuple[float, float] = (0.0, 0.0)
    price_source: str = "owned hardware"
    sparse_tflops: float | None = None
    speed_range: tuple[float, float] = (0.5, 1.5)
    spec_source: str = ""


HARDWARE: dict[str, HardwareProfile] = {
    profile.key: profile
    for profile in (
        HardwareProfile(
            key="dev-vm-6cpu",
            description=(
                "Development VM used for this study: 6 AMD EPYC vCPUs (AVX2), "
                "11 GB RAM, but its cgroup caps the sandbox at 3.1 GB. "
                "Efficiencies calibrated with llama-bench (4 threads, "
                "weights in RAM: 19.4 prompt and 4.2 output tokens/s)."
            ),
            memory_gb=3.1,
            bandwidth_gbs=20.0,
            compute_tflops=0.48,
            decode_efficiency=0.6,
            prefill_efficiency=0.31,
            speed_range=(0.8, 1.2),
        ),
        HardwareProfile(
            key="github-ubuntu-4cpu",
            description=(
                "GitHub-hosted ubuntu-latest runner: 4 vCPU, 16 GB RAM. "
                "Efficiencies calibrated with the cpu-smoke job of run "
                "38550186566 (3 threads, weights in RAM: 31 prompt and 7.8 "
                "output tokens/s, 66 s to read one 896-pixel image)."
            ),
            memory_gb=16.0,
            bandwidth_gbs=36.0,
            compute_tflops=0.32,
            decode_efficiency=0.6,
            prefill_efficiency=0.83,
            vision_efficiency=0.19,
            # Runs 38550186566 and 38552481450 differed by -27% to +45%.
            speed_range=(0.7, 1.5),
        ),
        HardwareProfile(
            key="m3-pro-18gb",
            description=(
                "Apple M3 Pro, 18 GiB shared RAM, 150 GB/s bandwidth. "
                "Conservative 8 GB model ceiling leaves room for macOS/apps. "
                "Compute and efficiencies are uncalibrated assumptions; "
                "use measured MLX reports for speed."
            ),
            memory_gb=8.0,
            bandwidth_gbs=150.0,
            compute_tflops=8.0,
            decode_efficiency=0.4,
            prefill_efficiency=0.25,
            is_gpu=True,
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
            description=(
                "NVIDIA T4 16 GB (GitHub linux_4_core_gpu larger runner, "
                "Colab free tier). No bf16; 65 TFLOPS is fp16 mixed precision."
            ),
            memory_gb=16.0,
            bandwidth_gbs=320.0,
            compute_tflops=65.0,
            is_gpu=True,
            price_per_hour=3.12,
            price_range=(3.12, 3.12),
            price_source=(
                "GitHub larger runner linux_4_core_gpu, $0.052/min, only for "
                "organisations on Team/Enterprise plans "
                "(docs.github.com/en/billing/reference/actions-runner-pricing)"
            ),
            spec_source="nvidia.com/en-us/data-center/tesla-t4/",
        ),
        HardwareProfile(
            key="l4-24gb",
            description="NVIDIA L4 24 GB.",
            memory_gb=24.0,
            bandwidth_gbs=300.0,
            compute_tflops=121.0,
            sparse_tflops=242.0,
            is_gpu=True,
            price_per_hour=0.707,
            price_range=(0.20, 0.805),
            price_source=(
                "GCP g2-standard-4 us-central1 (whole VM); range: Vast.ai "
                "lowest to AWS g6.xlarge us-east-1"
            ),
            spec_source="nvidia.com/en-us/data-center/l4/ (sheet shows sparse)",
        ),
        HardwareProfile(
            key="rtx4090-24gb",
            description=(
                "NVIDIA GeForce RTX 4090 24 GB. 165 TFLOPS is bf16 with fp32 "
                "accumulation; GeForce cards run that at half the fp16 rate."
            ),
            memory_gb=24.0,
            bandwidth_gbs=1008.0,
            compute_tflops=165.0,
            sparse_tflops=330.0,
            is_gpu=True,
            price_per_hour=0.89,
            price_range=(0.13, 0.89),
            price_source=(
                "RunPod Secure Cloud; range: Vast.ai lowest to RunPod Secure"
            ),
            spec_source="NVIDIA Ada GPU architecture whitepaper",
        ),
        HardwareProfile(
            key="a100-80gb",
            description="NVIDIA A100 80 GB SXM.",
            memory_gb=80.0,
            bandwidth_gbs=2039.0,
            compute_tflops=312.0,
            sparse_tflops=624.0,
            is_gpu=True,
            price_per_hour=1.79,
            price_range=(0.43, 5.07),
            price_source=(
                "RunPod Secure Cloud SXM; range: Vast.ai lowest to GCP "
                "a2-ultragpu-1g us-central1"
            ),
            spec_source="nvidia.com/en-us/data-center/a100/",
        ),
        HardwareProfile(
            key="h100-80gb",
            description="NVIDIA H100 80 GB SXM.",
            memory_gb=80.0,
            bandwidth_gbs=3350.0,
            compute_tflops=989.0,
            sparse_tflops=1979.0,
            is_gpu=True,
            price_per_hour=3.99,
            price_range=(1.73, 11.06),
            price_source=(
                "RunPod Secure Cloud SXM; range: Vast.ai lowest to GCP "
                "a3-highgpu-8g us-central1 per GPU"
            ),
            spec_source="nvidia.com/en-us/data-center/h100/ (sheet shows sparse)",
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
    total_memory_gb_range: tuple[float, float] = (0.0, 0.0)
    capacity: str = ""
    decode_tokens_per_s_range: tuple[float, float] = (0.0, 0.0)
    seconds_per_request_range: tuple[float, float] = (0.0, 0.0)
    kv_cache_tokens: int = 0
    concurrency: int = 1
    aggregate_decode_tokens_per_s: float = 0.0

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
    allocated_context: int | None = None,
    concurrency: int = 1,
) -> InferenceEstimate:
    """Estimate memory use and speed of one request.

    By default the KV cache holds exactly the tokens a request uses. Runtimes
    that reserve the cache up front need ``allocated_context`` instead:
    ``llama-server --ctx-size 4096 --parallel 1`` (the pinned runner) keeps
    room for 4096 tokens however short the prompt is, and with ``--parallel
    N`` shares those tokens between ``N`` slots. vLLM reserves a fixed share
    of GPU memory, which this estimate does not model.

    Args:
        model: Model specification.
        hardware: Machine to run on.
        quant: Weight format, a key of :data:`BITS_PER_WEIGHT`.
        prompt_tokens: Text tokens in the prompt.
        output_tokens: Tokens to generate.
        images: Number of images in the prompt.
        overhead_gb: Runtime buffers, activations and framework overhead.
        allocated_context: Context tokens the runtime reserves for all slots
            together (``--ctx-size``), or ``None`` for the tokens used.
        concurrency: Requests decoded together (``--parallel``). Each one
            keeps its own KV cache; one decode step reads the weights once
            for all of them, so ``aggregate_decode_tokens_per_s`` is an upper
            bound that ignores the compute limit of large batches.

    Returns:
        The estimate. ``fits`` compares only the point estimate with the
        memory of ``hardware``; ``capacity`` uses the range: ``"fits"`` when
        even the high end fits, ``"does-not-fit"`` when even the low end does
        not, and ``"uncertain"`` in between. Neither says that a runtime can
        load the model (see :mod:`medbench.support`).
    """
    if concurrency < 1:
        message = "concurrency must be at least 1"
        raise ValueError(message)
    context = prompt_tokens + images * model.image_tokens + output_tokens
    per_slot = context
    if allocated_context is not None:
        per_slot = max(context, allocated_context // concurrency)
    weights = weights_bytes(model.total_params, quant)
    if quant in model.gguf_files:
        # The pinned server loads a separate unquantized vision projector.
        # Quantizing the language model does not quantize that file.
        weights = model.gguf_files[quant] + model.gguf_files.get("mmproj-F16", 0)
    kv_cache = kv_cache_bytes(model, per_slot) * concurrency
    used_kv = kv_cache_bytes(model, context) * concurrency
    total = weights + kv_cache + overhead_gb * GB

    active_bytes = weights_bytes(model.active_params, quant)
    bandwidth = hardware.bandwidth_gbs * GB * hardware.decode_efficiency
    decode = bandwidth / (active_bytes + used_kv)
    flops = hardware.compute_tflops * 1e12 * hardware.prefill_efficiency
    prefill = flops / (2 * model.active_params)
    vision_flops = images * 2 * model.vision_params * 4096
    vision_efficiency = hardware.vision_efficiency
    if vision_efficiency is None:
        vision_efficiency = hardware.prefill_efficiency
    seconds = (
        (prompt_tokens + images * model.image_tokens) / prefill
        + vision_flops / (hardware.compute_tflops * 1e12 * vision_efficiency)
        + output_tokens / decode
    )
    low_gb, high_gb = (total / GB * factor for factor in MEMORY_RANGE)
    if high_gb <= hardware.memory_gb:
        capacity = "fits"
    elif low_gb > hardware.memory_gb:
        capacity = "does-not-fit"
    else:
        capacity = "uncertain"
    slow, fast = hardware.speed_range
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
        total_memory_gb_range=(low_gb, high_gb),
        capacity=capacity,
        decode_tokens_per_s_range=(decode * slow, decode * fast),
        seconds_per_request_range=(seconds / fast, seconds / slow),
        kv_cache_tokens=per_slot * concurrency,
        concurrency=concurrency,
        aggregate_decode_tokens_per_s=decode * concurrency,
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
    total_gb_range: tuple[float, float] = (0.0, 0.0)
    checkpoint_disk_gb: float = 0.0
    optimizer: str = "AdamW, fp32 moments"

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
    # A resumable checkpoint stores the trainable weights in bf16 plus fp32
    # master weights and both Adam moments (2 + 4 + 8 bytes per parameter).
    checkpoint = trainable * 14
    return FineTuneEstimate(
        method=method,
        weights_gb=weights / GB,
        trainable_params=trainable,
        optimizer_and_grads_gb=optimizer / GB,
        activations_gb=activations / GB,
        total_gb=total / GB,
        total_gb_range=(total / GB * FINETUNE_RANGE[0], total / GB * FINETUNE_RANGE[1]),
        checkpoint_disk_gb=checkpoint / GB,
    )


@dataclass(frozen=True)
class TrainingEstimate:
    """Compute, time and cost of a training run."""

    flops: float
    gpu_hours: float
    wall_clock_hours: float
    cost_usd: float
    cost_usd_range: tuple[float, float] = (0.0, 0.0)
    price_source: str = ""
    gpus: int = 1
    scaling_efficiency: float = 1.0
    wall_clock_hours_range: tuple[float, float] = (0.0, 0.0)

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
    price_per_hour: float | None = None,
    scaling_efficiency: float | None = None,
) -> TrainingEstimate:
    """Estimate compute, time and money for training.

    Uses ``6 * N * D`` FLOPs for full training (forward 2ND, backward 4ND).
    LoRA skips the weight gradients of the frozen model: about ``4 * N * D``.

    Args:
        params: Model parameters ``N``.
        tokens: Training tokens ``D`` (epochs times dataset tokens).
        hardware: GPU type.
        gpus: Number of GPUs.
        mfu: Model FLOPs utilization of one GPU, the share of peak *dense*
            compute achieved. Sparse tensor-core FLOPS do not apply.
        method: ``"full"`` or ``"lora"``.
        price_per_hour: Your own price; default: ``hardware.price_per_hour``.
        scaling_efficiency: Share of linear speed-up kept with ``gpus`` GPUs;
            default: 1 for one GPU, :data:`SCALING_EFFICIENCY` otherwise.

    Returns:
        The estimate. ``gpu_hours`` and the cost count every GPU for the
        whole wall-clock time, so they grow when scaling is imperfect. The
        ranges combine the price range with the scaling-efficiency range.
    """
    factor = 6 if method == "full" else 4
    flops = factor * params * tokens
    per_gpu = hardware.compute_tflops * 1e12 * mfu
    ideal_hours = flops / per_gpu / 3600
    if scaling_efficiency is not None:
        efficiency = low_eff = high_eff = scaling_efficiency
    elif gpus == 1:
        efficiency = low_eff = high_eff = 1.0
    else:
        efficiency, low_eff, high_eff = SCALING_EFFICIENCY
    gpu_hours = ideal_hours / efficiency
    if price_per_hour is None:
        price = hardware.price_per_hour
        low, high = hardware.price_range
        source = f"{hardware.price_source}, checked {PRICES_CHECKED}"
    else:
        price = low = high = price_per_hour
        source = "user-supplied"
    return TrainingEstimate(
        flops=flops,
        gpu_hours=gpu_hours,
        wall_clock_hours=gpu_hours / gpus,
        cost_usd=gpu_hours * price,
        cost_usd_range=(ideal_hours / high_eff * low, ideal_hours / low_eff * high),
        price_source=source,
        gpus=gpus,
        scaling_efficiency=efficiency,
        wall_clock_hours_range=(
            ideal_hours / high_eff / gpus,
            ideal_hours / low_eff / gpus,
        ),
    )
