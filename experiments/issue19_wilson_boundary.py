"""Find inputs where wilson_interval misses its exact bounds (issue #19).

With every answer right the upper Wilson bound is exactly 1, and with every
answer wrong the lower bound is exactly 0, but the closed form only reaches
them up to rounding. macOS 26 with Python 3.15 returned 0.9999999999999999 for
4 out of 4; this prints the totals and confidences that miss on this host, and
the same check with z moved by one ulp either way.

    python experiments/issue19_wilson_boundary.py
"""

from __future__ import annotations

import math
from statistics import NormalDist
from unittest import mock

from medbench.metrics import wilson_interval


def misses() -> list[tuple[int, float, float, float]]:
    found = []
    for total in range(1, 201):
        for confidence in (0.8, 0.9, 0.95, 0.99):
            low = wilson_interval(0, total, confidence)[0]
            high = wilson_interval(total, total, confidence)[1]
            if low != 0.0 or high != 1.0:
                found.append((total, confidence, low, high))
    return found


def main() -> None:
    print("this host:", misses()[:5], len(misses()))
    original = NormalDist.inv_cdf
    for direction in (-math.inf, math.inf):

        def shifted(self: NormalDist, p: float, direction: float = direction) -> float:
            return math.nextafter(original(self, p), direction)

        with mock.patch.object(NormalDist, "inv_cdf", shifted):
            found = misses()
        print(f"z one ulp toward {direction}:", found[:5], len(found))


if __name__ == "__main__":
    main()
