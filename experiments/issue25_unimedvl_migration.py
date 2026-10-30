"""Bounded reproduction of legacy UniMedVL config warnings and CPU varlen gap.

UNIMEDVL_CODE_DIR and UNIMEDVL_MODEL_DIR point to downloads made by
scripts/medbench-model-run.sh with UNIMEDVL_DEVICE=cpu. No model weights load.
Run with that script's Python 3.14 model environment from the repository root.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import torch
import transformers
from torch.nn.attention.varlen import varlen_attn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "unimedvl_server", ROOT / "scripts/unimedvl_server.py"
)
assert SPEC
assert SPEC.loader
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)
sys.path.insert(0, str(Path(os.environ["UNIMEDVL_CODE_DIR"]) / "codes"))
server.install_flash_attn_shim()
print(f"torch={torch.__version__}; transformers={transformers.__version__}")

from modeling.unimedvl import Qwen2Config  # noqa: E402

config_path = Path(os.environ["UNIMEDVL_MODEL_DIR"]) / "llm_config.json"
with warnings.catch_warnings():
    warnings.simplefilter("error", FutureWarning)
    try:
        Qwen2Config.from_json_file(str(config_path))
    except FutureWarning as error:
        print(f"Before: {error}")
    else:
        message = "The upstream config no longer reproduces the deprecation"
        raise AssertionError(message)
    server.prepare_upstream_imports()
    config = server.load_qwen2_config(config_path)
    print(
        f"After: maintained Qwen2Config loads without FutureWarning; {config.rope_parameters}"
    )

# Only eight tokens: this cannot stress the host or load checkpoint weights.
torch.manual_seed(25)
q = torch.randn(8, 2, 16, dtype=torch.bfloat16)
cu = torch.tensor([0, 3, 8], dtype=torch.int32)
try:
    varlen_attn(q, q, q, cu, cu, 5, 5, window_size=(-1, 0))
except NotImplementedError as error:
    print(f"CPU varlen gap: {str(error).splitlines()[0]}")
else:
    message = "CPU varlen now works: remove the local fallback"
    raise AssertionError(message)
result = server.flash_attn_varlen_func(q, q, q, cu, cu, 5, 5, causal=True)
assert result.shape == q.shape
assert torch.isfinite(result).all()
print("Retained CPU fallback: finite output with the expected packed shape")
