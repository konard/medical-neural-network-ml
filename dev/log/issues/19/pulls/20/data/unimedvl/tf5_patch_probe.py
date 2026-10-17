import runpy, sys, torch
from transformers import modeling_rope_utils as m
def _default(config, device=None, seq_len=None, **kw):
    base = config.rope_theta
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv, 1.0
m.ROPE_INIT_FUNCTIONS["default"] = _default
sys.argv = ["x", "/tmp/unimedsrc/code", "/tmp/unimedsrc/model"]
runpy.run_path("/tmp/gh-issue-solver-1792225464198/experiments/issue19_unimedvl_cpu_smoke.py", run_name="__main__")
