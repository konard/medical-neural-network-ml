"""Rough single-thread memory bandwidth probe (issue #1).

Decoding speed of a language model on CPU is limited by how fast the weights
can be read from RAM, so this number calibrates the resource calculator.
Copies a 512 MB buffer a few times; bounded to ~1 GB of RAM and a few seconds.
"""

from __future__ import annotations

import time

SIZE = 512 * 1024 * 1024
REPEATS = 5


def main() -> None:
    """Print copy bandwidth in GB/s (read + write traffic)."""
    source = bytearray(b"\x01") * SIZE
    target = bytearray(b"\x02") * SIZE  # touch pages first: no page faults
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        target[:] = source
        best = min(best, time.perf_counter() - start)
    print(f"copy bandwidth: {2 * SIZE / best / 1e9:.1f} GB/s (best of {REPEATS})")


if __name__ == "__main__":
    main()
