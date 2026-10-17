#!/usr/bin/env python3
"""Run upstream UniMedVL's ``Bagel.chat`` on the CPU with a tiny random model (#19).

scripts/unimedvl-requirements.txt pins the Python 3.13 stack for the real
29 GB model, which needs a CUDA GPU and flash-attn. To check that a new set of
pins still runs upstream's code without a GPU, this script:

* replaces ``flash_attn.flash_attn_varlen_func`` with a plain-torch reference
  (per-sequence attention, bottom-right causal mask, grouped KV heads, as
  flash-attn documents it);
* shrinks the real ``llm_config.json`` / ``vit_config.json`` to two layers and
  a small hidden size (the real 152k vocabulary and tokenizer are kept);
* builds the model exactly like scripts/unimedvl_server.py does
  (``init_empty_weights``, meta patch embedding, ``load_checkpoint_and_dispatch``
  from a safetensors file, ``dispatch_model``), from seeded random weights;
* calls ``chat`` with a fixed image and prompt and prints JSON with the package
  versions and the generated token ids.

Running it in a venv with the old pins and one with the new pins and comparing
the token ids shows whether the update changed what upstream's code computes.

Usage (code_dir is a checkout of uni-medical/UniMedVL, model_dir holds the
config and tokenizer files listed in scripts/medbench-model-run.sh):

    VENV/bin/python -I experiments/issue19_unimedvl_cpu_smoke.py CODE_DIR MODEL_DIR [IMAGE_SEED]
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import tempfile
import types
import warnings
from importlib.metadata import version
from pathlib import Path


def reference_varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **_unused,
):
    """Plain-torch ``flash_attn_varlen_func`` for (total, heads, dim) inputs."""
    import torch

    del max_seqlen_q, max_seqlen_k, dropout_p
    scale = softmax_scale or q.shape[-1] ** -0.5
    groups = q.shape[1] // k.shape[1]
    outputs = []
    for index in range(len(cu_seqlens_q) - 1):
        q_start, q_end = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        k_start, k_end = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
        query = q[q_start:q_end].float().transpose(0, 1)
        key = k[k_start:k_end].float().repeat_interleave(groups, 1).transpose(0, 1)
        value = v[k_start:k_end].float().repeat_interleave(groups, 1).transpose(0, 1)
        scores = query @ key.transpose(1, 2) * scale
        if causal:
            # flash-attn aligns the causal mask to the bottom-right corner.
            rows = torch.arange(q_end - q_start).unsqueeze(1)
            cols = torch.arange(k_end - k_start).unsqueeze(0)
            offset = (k_end - k_start) - (q_end - q_start)
            scores = scores.masked_fill(cols > rows + offset, float("-inf"))
        attended = scores.softmax(-1) @ value
        outputs.append(attended.transpose(0, 1).to(q.dtype))
    return torch.cat(outputs)


def install_flash_attn_stub() -> None:
    """Make ``from flash_attn import flash_attn_varlen_func`` the reference.

    transformers probes for flash-attn with find_spec() and the installed
    distribution's version; the stub has a spec but no distribution, so
    transformers sees no flash-attn (as on a CPU host) while upstream's own
    import gets the reference.
    """
    stub = types.ModuleType("flash_attn")
    stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    stub.flash_attn_varlen_func = reference_varlen_attention  # type: ignore[attr-defined]
    sys.modules["flash_attn"] = stub


def tiny_model(model_dir: Path):
    """Build a two-layer Bagel the way scripts/unimedvl_server.py loads one."""
    import torch
    from accelerate import (
        dispatch_model,
        init_empty_weights,
        load_checkpoint_and_dispatch,
    )
    from modeling.unimedvl import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from safetensors.torch import save_file

    torch.manual_seed(0)
    llm_config = Qwen2Config.from_json_file(str(model_dir / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.hidden_size = 64
    llm_config.intermediate_size = 128
    llm_config.num_hidden_layers = 2
    llm_config.num_attention_heads = 4
    llm_config.num_key_value_heads = 2
    vit_config = SiglipVisionConfig.from_json_file(str(model_dir / "vit_config.json"))
    vit_config.rope = False
    vit_config.hidden_size = 32
    vit_config.intermediate_size = 64
    vit_config.num_hidden_layers = 2
    vit_config.num_attention_heads = 2
    config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
    )

    # Random weights, written to safetensors like the real ema_bf16 file.
    source = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
    source.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    with tempfile.TemporaryDirectory() as scratch:
        checkpoint = Path(scratch) / "tiny.safetensors"
        state = {name: t.contiguous() for name, t in source.state_dict().items()}
        save_file(state, str(checkpoint))
        with init_empty_weights():
            model = Bagel(
                Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config
            )
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(checkpoint),
            device_map={"": "cpu"},
            offload_buffers=False,
            dtype=torch.bfloat16,
            force_hooks=False,
        )
    return dispatch_model(model, device_map={"": "cpu"}).eval()


def main(code_dir: Path, model_dir: Path, image_seed: int = 0) -> dict[str, object]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    install_flash_attn_stub()
    os.chdir(code_dir)
    sys.path.insert(0, str(code_dir / "codes"))

    import numpy as np
    import torch
    from data.data_utils import add_special_tokens, pil_img2rgb
    from eval.vlm.utils import build_transform
    from modeling.qwen2 import Qwen2Tokenizer
    from PIL import Image

    model = tiny_model(model_dir)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    decoded: list[list[int]] = []
    original_decode = tokenizer.decode

    def recording_decode(ids, *args, **kwargs):
        decoded.append([int(i) for i in ids])
        return original_decode(ids, *args, **kwargs)

    tokenizer.decode = recording_decode
    rng = np.random.default_rng(image_seed)
    pixels = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    image = pil_img2rgb(Image.fromarray(pixels))
    with warnings.catch_warnings():
        # chat() asks for CUDA bf16 autocast, which torch disables (with a
        # warning) without a GPU; CPU autocast stands in for it.
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            text = model.chat(
                tokenizer,
                new_token_ids,
                build_transform(),
                images=[image],
                prompt="Describe the image.",
                max_length=12,
            )
    packages = (
        "torch",
        "torchvision",
        "transformers",
        "tokenizers",
        "accelerate",
        "safetensors",
        "numpy",
        "pillow",
        "opencv-python",
    )
    return {
        "versions": {name: version(name) for name in packages},
        "token_ids": decoded[-1],
        "text": text,
    }


if __name__ == "__main__":
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    result = main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), seed)
    print(json.dumps(result, indent=2))
