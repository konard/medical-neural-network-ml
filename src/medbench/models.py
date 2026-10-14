"""Catalogue of candidate models with the numbers the resource calculator needs.

Every number here was copied from the model's ``config.json`` or from the
Hugging Face Hub API (``?expand[]=safetensors``) on 2026-10-14. The case study
in ``docs/case-studies/issue-1`` explains where each value comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MODELS", "AttentionLayout", "ModelSpec", "get_model"]


@dataclass(frozen=True)
class AttentionLayout:
    """Shape of the attention layers, used to size the KV cache.

    Attributes:
        num_layers: Number of transformer layers in the language model.
        num_kv_heads: Number of key/value heads (grouped-query attention).
        head_dim: Size of one attention head.
        sliding_window: Window of the local (sliding) attention layers, or
            ``None`` when every layer attends to the whole context.
        global_layers: How many layers use full (global) attention. Only
            meaningful when ``sliding_window`` is set.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int | None = None
    global_layers: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Description of one model for resource estimation.

    Attributes:
        key: Short name used on the command line.
        hf_id: Hugging Face repository id.
        total_params: All parameters that must be stored in memory.
        active_params: Parameters read for every generated token. Smaller than
            ``total_params`` for mixture-of-experts / mixture-of-transformers
            models.
        attention: Attention layout of the language model.
        modalities: Inputs the model accepts.
        license: License name as published by the owner.
        gated: Whether downloading needs an accepted license on the Hub.
        image_tokens: Tokens one image costs in the prompt (0 for text-only).
        vision_params: Parameters of the image encoder (run once per image).
        notes: Free-form remarks.
    """

    key: str
    hf_id: str
    total_params: float
    active_params: float
    attention: AttentionLayout
    modalities: tuple[str, ...]
    license: str
    gated: bool
    image_tokens: int = 0
    vision_params: float = 0.0
    notes: str = ""
    gguf_files: dict[str, float] = field(default_factory=dict)


_GEMMA3_4B = AttentionLayout(
    num_layers=34, num_kv_heads=4, head_dim=256, sliding_window=1024, global_layers=5
)
_GEMMA3_27B = AttentionLayout(
    num_layers=62, num_kv_heads=16, head_dim=128, sliding_window=1024, global_layers=10
)
_QWEN25_7B = AttentionLayout(num_layers=28, num_kv_heads=4, head_dim=128)

MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="medgemma-4b-it",
            hf_id="google/medgemma-4b-it",
            total_params=4.300e9,
            active_params=4.300e9,
            attention=_GEMMA3_4B,
            modalities=("text", "image"),
            license="Health AI Developer Foundations terms",
            gated=True,
            image_tokens=256,
            vision_params=0.42e9,
            notes="Gemma 3 4B + MedSigLIP image encoder (896x896 input).",
            gguf_files={"Q4_K_M": 2.490e9, "Q8_0": 4.130e9, "mmproj-F16": 0.851e9},
        ),
        ModelSpec(
            key="medgemma-27b-it",
            hf_id="google/medgemma-27b-it",
            total_params=27.432e9,
            active_params=27.432e9,
            attention=_GEMMA3_27B,
            modalities=("text", "image"),
            license="Health AI Developer Foundations terms",
            gated=True,
            image_tokens=256,
            vision_params=0.42e9,
            notes="Gemma 3 27B + MedSigLIP image encoder (896x896 input).",
            gguf_files={"Q4_K_M": 16.547e9, "Q8_0": 28.708e9, "mmproj-F16": 0.858e9},
        ),
        ModelSpec(
            key="medgemma-27b-text-it",
            hf_id="google/medgemma-27b-text-it",
            total_params=27.009e9,
            active_params=27.009e9,
            attention=_GEMMA3_27B,
            modalities=("text",),
            license="Health AI Developer Foundations terms",
            gated=True,
            notes="Text-only MedGemma 27B.",
        ),
        ModelSpec(
            key="unimedvl",
            hf_id="General-Medical-AI/UniMedVL",
            total_params=14.6e9,
            active_params=7.6e9,
            attention=_QWEN25_7B,
            modalities=("text", "image"),
            license="Apache-2.0",
            gated=False,
            image_tokens=1296,
            vision_params=0.4e9,
            notes=(
                "BAGEL-style mixture-of-transformers on Qwen2.5-7B: one 7B "
                "expert for understanding, one for image generation. 29.2 GB "
                "bf16 checkpoint; no GGUF export. The SigLIP encoder uses 14 px "
                "patches at native resolution (up to 980 px), so a 504x504 "
                "image costs 36x36 = 1296 tokens."
            ),
        ),
        ModelSpec(
            key="qwen2.5-vl-7b",
            hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
            total_params=8.292e9,
            active_params=8.292e9,
            attention=_QWEN25_7B,
            modalities=("text", "image"),
            license="Apache-2.0",
            gated=False,
            image_tokens=1024,
            vision_params=0.67e9,
            notes="General-purpose baseline; image tokens depend on resolution.",
        ),
    )
}


def get_model(key: str) -> ModelSpec:
    """Return a model from the catalogue.

    Args:
        key: Short model name, e.g. ``"medgemma-27b-it"``.

    Returns:
        The matching model specification.

    Raises:
        KeyError: If the model is unknown.
    """
    try:
        return MODELS[key]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        message = f"unknown model {key!r}; known models: {known}"
        raise KeyError(message) from None
