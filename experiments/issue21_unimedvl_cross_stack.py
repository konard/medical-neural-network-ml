#!/usr/bin/env python3
"""Compare upstream UniMedVL across two dependency stacks on the same weights (#21).

experiments/issue19_unimedvl_cpu_smoke.py builds its tiny random model under
whatever transformers and torch are installed, so two stacks that initialise
weights differently produce different tokens even when they compute the same
function. #19 recorded "transformers 5 with the RoPE default restored gives
different tokens" from exactly that comparison. This script removes the
confounder: ``save`` writes the tiny model's weights once, and ``run`` loads that
same safetensors file in any venv, runs ``Bagel.chat`` and records the token ids
plus the logits of every generation step.

    OLD/bin/python -I experiments/issue21_unimedvl_cross_stack.py save CODE_DIR MODEL_DIR WEIGHTS
    OLD/bin/python -I experiments/issue21_unimedvl_cross_stack.py run CODE_DIR MODEL_DIR WEIGHTS OUT.json
    NEW/bin/python -I experiments/issue21_unimedvl_cross_stack.py run CODE_DIR MODEL_DIR WEIGHTS OUT.json

``run`` puts the transformers 4.x ``"default"`` RoPE initialiser back when the
installed transformers lacks it (transformers 5 removed it), the same way
scripts/unimedvl_server.py does.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

SMOKE = runpy.run_path(str(Path(__file__).with_name("issue19_unimedvl_cpu_smoke.py")))


def prepare(code_dir: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    SMOKE["install_flash_attn_stub"]()
    os.chdir(code_dir)
    sys.path.insert(0, str(code_dir / "codes"))


def restore_default_rope() -> bool:
    """Put back the transformers 4.x ``"default"`` RoPE initialiser if missing."""
    import torch
    from transformers import modeling_rope_utils

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return False

    def default_rope_parameters(
        config: Any, device: Any = None, *_args: Any, **_kwargs: Any
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        exponents = torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim
        return 1.0 / (base**exponents), 1.0

    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters
    return True


def configs(model_dir: Path):
    from modeling.unimedvl import BagelConfig, Qwen2Config, SiglipVisionConfig

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
    return llm_config, vit_config, config


def save(model_dir: Path, weights: Path) -> None:
    import torch
    from modeling.unimedvl import Bagel, Qwen2ForCausalLM, SiglipVisionModel
    from safetensors.torch import save_file

    restore_default_rope()
    torch.manual_seed(0)
    llm_config, vit_config, config = configs(model_dir)
    source = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
    source.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    state = {name: t.contiguous() for name, t in source.state_dict().items()}
    save_file(state, str(weights))


def load(model_dir: Path, weights: Path):
    import torch
    from accelerate import dispatch_model, init_empty_weights, load_checkpoint_and_dispatch
    from modeling.unimedvl import Bagel, Qwen2ForCausalLM, SiglipVisionModel

    llm_config, vit_config, config = configs(model_dir)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(weights),
        device_map={"": "cpu"},
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=False,
    )
    return dispatch_model(model, device_map={"": "cpu"}).eval()


def run(model_dir: Path, weights: Path) -> dict[str, object]:
    import numpy as np
    import torch
    from data.data_utils import add_special_tokens, pil_img2rgb
    from eval.vlm.utils import build_transform
    from modeling.qwen2 import Qwen2Tokenizer
    from PIL import Image

    patched = restore_default_rope()
    model = load(model_dir, weights)
    inv_freq = [
        module.inv_freq.float().tolist()
        for module in model.modules()
        if hasattr(module, "inv_freq")
    ]
    step_logits: list[list[float]] = []
    model.language_model.lm_head.register_forward_hook(
        lambda _m, _i, out: step_logits.append(out[-1].float().tolist())
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    decoded: list[list[int]] = []
    original_decode = tokenizer.decode

    def recording_decode(ids, *args, **kwargs):
        decoded.append([int(i) for i in ids])
        return original_decode(ids, *args, **kwargs)

    tokenizer.decode = recording_decode
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    image = pil_img2rgb(Image.fromarray(pixels))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            model.chat(
                tokenizer,
                new_token_ids,
                build_transform(),
                images=[image],
                prompt="Describe the image.",
                max_length=12,
            )
    packages = ("torch", "torchvision", "transformers", "tokenizers", "accelerate")
    return {
        "versions": {name: version(name) for name in packages},
        "python": sys.version.split()[0],
        "rope_default_restored": patched,
        "inv_freq": inv_freq,
        "token_ids": decoded[-1],
        "step_logits": step_logits,
    }


def compare(first: Path, second: Path) -> dict[str, object]:
    """Summarise two ``run`` outputs: same tokens, largest logit difference."""
    import math

    a = json.loads(first.read_text())
    b = json.loads(second.read_text())
    steps = min(len(a["step_logits"]), len(b["step_logits"]))
    worst = []
    for index in range(steps):
        diffs = [abs(x - y) for x, y in zip(a["step_logits"][index], b["step_logits"][index])]
        worst.append(max(diffs))
    inv = max(
        abs(x - y)
        for left, right in zip(a["inv_freq"], b["inv_freq"])
        for x, y in zip(left, right)
    ) if a["inv_freq"] else math.nan
    return {
        "a": a["versions"],
        "b": b["versions"],
        "same_tokens": a["token_ids"] == b["token_ids"],
        "a_tokens": a["token_ids"],
        "b_tokens": b["token_ids"],
        "max_inv_freq_diff": inv,
        "max_logit_diff_per_step": worst,
    }


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "compare":
        print(json.dumps(compare(Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
        raise SystemExit(0)
    code, model_files, checkpoint, *rest = (Path(arg).resolve() for arg in sys.argv[2:])
    prepare(code)
    if command == "save":
        save(model_files, checkpoint)
    else:
        output = rest[0]
        output.write_text(json.dumps(run(model_files, checkpoint)))
        print(output)
