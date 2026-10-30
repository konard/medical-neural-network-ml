The 8-token reproduction in this issue still raises the CPU dispatcher `NotImplementedError` on **torch 2.15.0+cpu / Python 3.14.10**. We checked it while updating https://github.com/konard/medical-neural-network-ml/pull/26; [bounded reproduction](https://github.com/konard/medical-neural-network-ml/blob/issue-25-4d7bfe8dca13/experiments/issue25_unimedvl_migration.py).

One improvement for the suggested CPU math implementation: PyTorch already provides the bottom-right mask as `torch.nn.attention.bias.causal_lower_right(q_len, k_len)`. Our per-sequence SDPA fallback now uses this instead of allocating/comparing `arange` rows/columns by hand:

```python
from torch.nn.attention.bias import causal_lower_right
from torch.nn.functional import scaled_dot_product_attention

# q_i/k_i/v_i: (heads, sequence, head_dim), one packed sequence
out = scaled_dot_product_attention(
    q_i, k_i, v_i,
    attn_mask=causal_lower_right(q_i.shape[-2], k_i.shape[-2]),
    enable_gqa=q_i.shape[0] != k_i.shape[0],
)
```

The local packed-sequence loop remains necessary until `varlen_attn` provides a CPU math backend. A native fallback could reuse this bias and SDPA; alternatively, documenting device support and raising a device-specific error would clarify the API. Current adapter tests cover packed sequences, GQA, unequal query/key lengths and the upstream chat golden.
