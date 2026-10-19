"""Upstream UniMedVL code on the locked packages, through unimedvl_server.py (#21).

scripts/unimedvl_server.py runs upstream's own model code, which was written
for transformers 4.x and imports flash-attn. ``prepare_upstream_imports``
provides both on the locked transformers 5 and torch (no flash-attn wheel).
This test runs ``Bagel.chat`` on a tiny model built from the real configs and
tokenizer, so a dependency update that breaks upstream's code, or changes
what it generates, fails here instead of on the GPU runner.

The weights come from generators seeded with each parameter's name, not from
transformers' initialisers, which differ between versions. GOLDEN_TOKENS and
GOLDEN_BEST_LOGITS were checked on torch 2.8.0 with transformers 4.57.6 (the
stack upstream was written for) and on torch 2.14.1 with transformers 5.20.1:
the tokens agree and the logits differ by bfloat16 rounding only
(experiments/issue21_unimedvl_golden.py).

Needs the upstream checkout and the model's configs and tokenizer:
``SERVER=unimedvl UNIMEDVL_DEVICE=cpu bash scripts/medbench-model-run.sh
download`` puts them in .cache/medbench, and the ``unimedvl-cpu`` job of
.github/workflows/model-tests.yml runs this with that venv's Python.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "unimedvl_server.py"
CODE_DIR = os.environ.get("UNIMEDVL_CODE_DIR")
MODEL_DIR = os.environ.get("UNIMEDVL_MODEL_DIR")

pytestmark = pytest.mark.skipif(
    not (CODE_DIR and MODEL_DIR),
    reason="set UNIMEDVL_CODE_DIR and UNIMEDVL_MODEL_DIR (scripts/medbench-model-run.sh download)",
)

GOLDEN_TOKENS = [
    151644, 1006, 1057, 1018, 1006, 1057, 1018, 1006, 1009, 1002, 1058, 1018,
]  # fmt: skip
GOLDEN_BEST_LOGITS = [
    1.5546875, 3.203125, 2.5, 2.96875, 2.21875, 2.4375,
    2.46875, 1.9140625, 2.421875, 2.203125, 2.5625, 2.421875,
]  # fmt: skip
# The two stacks differ by up to 1.2% (bfloat16 rounding). Attention without
# its causal mask moves some logits by 7-20%, a top-left aligned mask
# changes every token (experiments/issue21_unimedvl_golden.py).
RTOL = 0.03
# The output layer scores only these ordinary tokens. With 151,000 random
# logits the best two are usually within a bfloat16 step or two of each
# other, so the greedy choice would depend on rounding, not on the model.
OUTPUT_TOKENS = range(1000, 1064)
# Gain of the query and key norms. At 1, attention is spread almost evenly
# over the hundreds of image tokens, and even a missing causal mask moves the logits
# by no more than bfloat16 rounding; sharper attention makes masks matter.
QK_NORMS = {"q_norm", "k_norm", "q_norm_moe_gen", "k_norm_moe_gen"}
QK_GAIN = 2.0
# Chosen with experiments/issue21_unimedvl_golden.py: every step's best token
# leads the next by at least 12 bfloat16 steps on both stacks.
SEED = 2


def load_server_module() -> Any:
    spec = importlib.util.spec_from_file_location("unimedvl_server", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def tiny_bagel(model_dir: Path) -> Any:
    """UniMedVL's architecture from its configs, two layers wide enough to test."""
    from modeling.unimedvl import (  # type: ignore[import-not-found]
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )

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
    model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    seed_weights(model)
    return model.to(torch.bfloat16).eval()


def seed_weights(model: Any) -> None:
    """Fill each parameter from a generator seeded with its name.

    Seeding by name, not by position, keeps every weight the same when a
    transformers release adds, drops or reorders parameters elsewhere.
    """
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            generator = torch.Generator().manual_seed(
                zlib.crc32(f"{SEED}:{name}".encode())
            )
            noise = torch.randn(parameter.shape, generator=generator)
            if parameter.dim() == 1 and name.rsplit(".", 2)[-2] in QK_NORMS:
                parameter.copy_(QK_GAIN * (1 + 0.1 * noise))
            elif parameter.dim() == 1 and "norm" in name:
                parameter.copy_(1 + 0.1 * noise)
            elif parameter.dim() == 1:
                parameter.copy_(0.02 * noise)
            else:
                parameter.copy_(noise / math.sqrt(parameter[0].numel()))
        lm_head = model.language_model.lm_head.weight
        keep = torch.zeros(lm_head.shape[0], 1, dtype=torch.bool)
        keep[OUTPUT_TOKENS.start : OUTPUT_TOKENS.stop] = True
        lm_head.mul_(keep)


def generate(code_dir: Path, model_dir: Path) -> tuple[list[int], list[float]]:
    """Token ids of ``Bagel.chat`` on a seeded random image, and each step's best logit."""
    import numpy as np
    from data.data_utils import add_special_tokens, pil_img2rgb  # type: ignore[import-not-found]
    from eval.vlm.utils import build_transform  # type: ignore[import-not-found]
    from modeling.qwen2 import Qwen2Tokenizer  # type: ignore[import-not-found]
    from PIL import Image

    model = tiny_bagel(model_dir)
    best_logits: list[float] = []
    model.language_model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: best_logits.append(
            float(output[-1].float().max())
        )
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    decoded: list[list[int]] = []
    decode = tokenizer.decode

    def recording_decode(ids: Any, *args: Any, **kwargs: Any) -> str:
        decoded.append([int(i) for i in ids])
        return decode(ids, *args, **kwargs)

    tokenizer.decode = recording_decode
    pixels = np.random.default_rng(0).integers(
        0, 256, size=(400, 400, 3), dtype=np.uint8
    )
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model.chat(
            tokenizer,
            new_token_ids,
            build_transform(),
            images=[pil_img2rgb(Image.fromarray(pixels))],
            prompt="Describe the image.",
            max_length=12,
        )
    return decoded[-1], best_logits


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    # Both are resolved before the chdir: CI passes them relative to the
    # repository root.
    code_dir = Path(str(CODE_DIR)).resolve()
    model_dir = Path(str(MODEL_DIR)).resolve()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.chdir(code_dir)
    monkeypatch.syspath_prepend(str(code_dir / "codes"))
    load_server_module().prepare_upstream_imports()
    return code_dir, model_dir


@pytest.mark.filterwarnings("ignore:.*CUDA is not available.*")
def test_chat_generates_the_golden_tokens(upstream: tuple[Path, Path]) -> None:
    tokens, best_logits = generate(*upstream)
    assert tokens == GOLDEN_TOKENS
    # Greedy tokens survive small numeric changes; the logits show them.
    torch.testing.assert_close(
        torch.tensor(best_logits), torch.tensor(GOLDEN_BEST_LOGITS), rtol=RTOL, atol=0
    )
