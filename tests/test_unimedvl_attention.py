"""Tests for the flash-attn and RoPE shims in scripts/unimedvl_server.py (#21).

They need torch (and transformers for the RoPE tests), which only the
UniMedVL environment installs, so the main test job skips them. The
``unimedvl-cpu`` job of .github/workflows/model-tests.yml runs them on the
locked UniMedVL packages (scripts/unimedvl-requirements.txt).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "unimedvl_server.py"
spec = importlib.util.spec_from_file_location("unimedvl_server", SCRIPT_PATH)
server_module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules[spec.name] = server_module  # type: ignore[union-attr]
spec.loader.exec_module(server_module)  # type: ignore[union-attr]


def reference(
    q: Any, k: Any, v: Any, lengths_q: list[int], lengths_k: list[int], *, causal: bool
) -> Any:
    """flash-attn's varlen attention written out in float32, one sequence at a time."""
    groups = q.shape[1] // k.shape[1]
    outputs = []
    q0 = k0 = 0
    for length_q, length_k in zip(lengths_q, lengths_k, strict=True):
        query = q[q0 : q0 + length_q].float().transpose(0, 1)
        key = k[k0 : k0 + length_k].float().repeat_interleave(groups, 1).transpose(0, 1)
        value = (
            v[k0 : k0 + length_k].float().repeat_interleave(groups, 1).transpose(0, 1)
        )
        scores = query @ key.transpose(1, 2) * q.shape[-1] ** -0.5
        if causal:  # bottom-right aligned, as in flash-attn 2.1+
            rows = torch.arange(length_q).unsqueeze(1)
            cols = torch.arange(length_k).unsqueeze(0)
            scores = scores.masked_fill(
                cols > rows + length_k - length_q, float("-inf")
            )
        outputs.append((scores.softmax(-1) @ value).transpose(0, 1))
        q0 += length_q
        k0 += length_k
    return torch.cat(outputs)


def packed(lengths: list[int], heads: int, dim: int = 16, seed: int = 0) -> Any:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(sum(lengths), heads, dim, generator=generator)


def cumulative(lengths: list[int]) -> Any:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )


def call(
    q: Any, k: Any, v: Any, lengths_q: list[int], lengths_k: list[int], **kwargs: Any
) -> Any:
    return server_module.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cumulative(lengths_q),
        cu_seqlens_k=cumulative(lengths_k),
        max_seqlen_q=max(lengths_q),
        max_seqlen_k=max(lengths_k),
        **kwargs,
    )


@pytest.mark.parametrize("causal", [False, True])
def test_packed_sequences_match_the_reference(causal: bool) -> None:
    """SigLIP's call: several images packed together, equal head counts."""
    lengths = [5, 1, 9]
    q, k, v = (packed(lengths, 4, seed=seed) for seed in range(3))
    result = call(q, k, v, lengths, lengths, causal=causal)
    expected = reference(q, k, v, lengths, lengths, causal=causal)
    assert result.shape == q.shape
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_cached_decoding_is_causal_from_the_bottom_right() -> None:
    """Qwen2's call with a KV cache: fewer queries than keys, 4 query heads per KV head."""
    lengths_q, lengths_k = [1, 3], [6, 3]
    q = packed(lengths_q, 8, seed=0)
    k, v = (packed(lengths_k, 2, seed=seed) for seed in (1, 2))
    result = call(q, k, v, lengths_q, lengths_k, causal=True)
    torch.testing.assert_close(
        result,
        reference(q, k, v, lengths_q, lengths_k, causal=True),
        atol=1e-5,
        rtol=1e-5,
    )
    # The single query of the first sequence sees all six cached keys.
    unmasked = reference(q[:1], k[:6], v[:6], [1], [6], causal=False)
    torch.testing.assert_close(result[:1], unmasked, atol=1e-5, rtol=1e-5)


def test_bfloat16_in_bfloat16_out() -> None:
    """Upstream casts SigLIP's inputs to bfloat16; the output keeps that dtype."""
    lengths = [7, 4]
    q, k, v = (packed(lengths, 2, seed=seed).bfloat16() for seed in range(3))
    result = call(q, k, v, lengths, lengths)
    assert result.dtype == torch.bfloat16
    expected = reference(q, k, v, lengths, lengths, causal=False)
    torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=2e-2)


def test_softmax_scale_is_used() -> None:
    lengths = [4]
    q, k, v = (packed(lengths, 2, seed=seed) for seed in range(3))
    result = call(q, k, v, lengths, lengths, softmax_scale=0.5)
    expected = reference(q * 0.5 / 16**-0.5, k, v, lengths, lengths, causal=False)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_dropout_and_unknown_arguments_are_refused() -> None:
    lengths = [3]
    q = packed(lengths, 2)
    with pytest.raises(ValueError, match="dropout"):
        call(q, q, q, lengths, lengths, dropout_p=0.1)
    with pytest.raises(TypeError, match="window_size"):
        call(q, q, q, lengths, lengths, window_size=(8, 0))


def test_varlen_attn_still_has_no_cpu_kernel() -> None:
    """Why the CPU path exists. If this fails, torch gained a CPU kernel and
    flash_attn_varlen_func can call varlen_attn on every device."""
    from torch.nn.attention.varlen import varlen_attn

    lengths = [3]
    q = packed(lengths, 2).bfloat16()
    with pytest.raises(NotImplementedError):
        varlen_attn(q, q, q, cumulative(lengths), cumulative(lengths), 3, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_uses_varlen_attn_and_matches_the_reference() -> None:  # pragma: no cover
    lengths_q, lengths_k = [1, 130, 5], [9, 130, 5]
    q = packed(lengths_q, 8).bfloat16().cuda()
    k, v = (packed(lengths_k, 2, seed=seed).bfloat16().cuda() for seed in (1, 2))
    result = server_module.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cumulative(lengths_q).cuda(),
        cu_seqlens_k=cumulative(lengths_k).cuda(),
        max_seqlen_q=130,
        max_seqlen_k=130,
        causal=True,
    )
    expected = reference(q.cpu(), k.cpu(), v.cpu(), lengths_q, lengths_k, causal=True)
    torch.testing.assert_close(result.cpu().float(), expected, atol=2e-2, rtol=2e-2)


def test_shim_is_what_upstream_imports_and_transformers_ignores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    server_module.install_flash_attn_shim()
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]

    assert flash_attn_varlen_func is server_module.flash_attn_varlen_func
    assert importlib.util.find_spec("flash_attn") is not None
    transformers_utils = pytest.importorskip("transformers.utils")
    assert not transformers_utils.is_flash_attn_2_available()


def test_default_rope_matches_transformers_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """The restored initialiser gives the frequencies transformers 5 computes itself."""
    rope_utils = pytest.importorskip("transformers.modeling_rope_utils")
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    functions = dict(rope_utils.ROPE_INIT_FUNCTIONS)
    monkeypatch.setattr(rope_utils, "ROPE_INIT_FUNCTIONS", functions)
    functions.pop("default", None)
    assert server_module.restore_default_rope() is True
    assert server_module.restore_default_rope() is False

    config = Qwen2Config(hidden_size=64, num_attention_heads=4, rope_theta=1_000_000.0)
    config.rope_theta = 1_000_000.0  # upstream's own Qwen2Config keeps this attribute
    inv_freq, factor = functions["default"](config, "cpu")
    expected, expected_factor = Qwen2RotaryEmbedding.compute_default_rope_parameters(
        config
    )
    torch.testing.assert_close(inv_freq, expected)
    assert factor == expected_factor == 1.0
