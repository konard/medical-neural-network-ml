"""Compare torch's causal_lower_right with UniMedVL's packed attention mask.

Run with the UniMedVL CPU environment. Inputs are deliberately tiny, including
KV-cache decoding (query shorter than key), GQA, and equal sequence lengths.
"""

import torch
from torch.nn.attention.bias import causal_lower_right
from torch.nn.functional import scaled_dot_product_attention


def main() -> None:
    torch.manual_seed(25)
    for query_length, key_length in [(1, 6), (3, 3), (5, 9)]:
        query = torch.randn(8, query_length, 16)
        key = torch.randn(2, key_length, 16)
        value = torch.randn_like(key)
        rows = torch.arange(query_length).unsqueeze(1)
        columns = torch.arange(key_length).unsqueeze(0)
        expected = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=columns <= rows + key_length - query_length,
            enable_gqa=True,
        )
        actual = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_lower_right(query_length, key_length),
            enable_gqa=True,
        )
        torch.testing.assert_close(actual, expected)
        print(f"query={query_length} key={key_length}: identical")


if __name__ == "__main__":
    main()
