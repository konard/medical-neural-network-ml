"""Run the UniMedVL CPU smoke test on transformers 5.x with RoPE patched (issue #19).

transformers 5 dropped ``"default"`` from ``ROPE_INIT_FUNCTIONS``, so building
upstream's Qwen2 fails with ``KeyError: 'default'``. This puts the 4.x default
back and runs experiments/issue19_unimedvl_cpu_smoke.py, to see whether that is
the only incompatibility (it is not: the generated tokens differ).

    python -I experiments/issue19_unimedvl_tf5_probe.py CODE_DIR MODEL_DIR
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import modeling_rope_utils


def default_rope_parameters(
    config: Any, device: Any = None, *_args: Any, **_kwargs: Any
) -> tuple[torch.Tensor, float]:
    """The transformers 4.x ``_compute_default_rope_parameters``."""
    base = config.rope_theta
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    exponents = torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim
    return 1.0 / (base**exponents), 1.0


if __name__ == "__main__":
    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters
    smoke = Path(__file__).with_name("issue19_unimedvl_cpu_smoke.py")
    sys.argv = [str(smoke), *sys.argv[1:]]
    runpy.run_path(str(smoke), run_name="__main__")
