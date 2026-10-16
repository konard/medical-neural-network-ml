"""What the benchmark scores can and cannot show: label review, splits, sizes.

A score is only as good as its labels, its test items and their number.
This module keeps that information next to the code:

* :func:`load_label_review` - a written review of every real image and page
  (view, laterality convention, where the label comes from, open doubts and a
  source group). The review is by the maintainers, **not clinicians**;
  ``clinician_reviewed`` is false until a clinician signs it.
* :func:`split_of` / :func:`grouped_split` - deterministic train/dev/test
  splits that keep every item of one patient or source in one split, so a
  model is never tuned on a near copy of a test item.
* :func:`required_sample_size` - how many items a split needs for a 95%
  Wilson interval of a given half-width.
* :func:`class_balance` - normal versus abnormal counts per task.

See docs/evaluation-validity.md for how these are used and their limits.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from statistics import NormalDist
from typing import Any, TypeVar

from medbench.real_data import Sample

__all__ = [
    "DEFAULT_SPLITS",
    "SPLIT_SALT",
    "LabelReview",
    "ReviewEntry",
    "class_balance",
    "grouped_split",
    "load_label_review",
    "required_sample_size",
    "sample_size_targets",
    "split_of",
]

T = TypeVar("T")

# Shares of groups that go to each split. The real teaching images and pages
# are too few to split and are used for testing only (see the docs).
DEFAULT_SPLITS: tuple[tuple[str, float], ...] = (
    ("train", 0.6),
    ("dev", 0.2),
    ("test", 0.2),
)
# Changing the salt reshuffles every split; keep it fixed so reports compare.
SPLIT_SALT = "medbench-split-v1"


@dataclass(frozen=True)
class ReviewEntry:
    """The review of one sample.

    Attributes:
        sample_id: Key of the sample in ``real_samples.json``.
        view: Projection, sequence or plane (e.g. "AP", "axial T2").
        laterality: Side of the finding and the convention used to read it.
        label_source: Where the label comes from.
        source_group: Items that may share a patient or a source. Splits keep
            a group together.
        uncertainty: Doubts a reader should know about the label.
        also_present: Other true findings that an answer may name.
    """

    sample_id: str
    view: str
    laterality: str
    label_source: str
    source_group: str
    uncertainty: str
    also_present: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabelReview:
    """Who reviewed the labels, how, and the per-sample entries."""

    reviewer: str
    clinician_reviewed: bool
    method: str
    reviewed_on: str
    entries: dict[str, ReviewEntry]

    def group_of(self, sample: Sample) -> str:
        """Source group of ``sample``; unreviewed samples get their own."""
        entry = self.entries.get(sample.sample_id)
        return entry.source_group if entry else sample.sample_id


def load_label_review(text: str | None = None) -> LabelReview:
    """Read ``data/label_review.json`` (or ``text``, for tests)."""
    if text is None:
        text = (
            resources.files("medbench")
            .joinpath("data/label_review.json")
            .read_text(encoding="utf-8")
        )
    raw: dict[str, Any] = json.loads(text)
    entries = {
        item["sample_id"]: ReviewEntry(
            **{**item, "also_present": tuple(item.get("also_present", ()))}
        )
        for item in raw["entries"]
    }
    return LabelReview(
        reviewer=raw["reviewer"],
        clinician_reviewed=bool(raw["clinician_reviewed"]),
        method=raw["method"],
        reviewed_on=raw["reviewed_on"],
        entries=entries,
    )


def split_of(
    group: str,
    splits: Sequence[tuple[str, float]] = DEFAULT_SPLITS,
    salt: str = SPLIT_SALT,
) -> str:
    """Split name for ``group``, decided by a hash so it never changes.

    The same group always lands in the same split, whatever other items
    exist, so adding data does not move old items between splits.
    """
    total = sum(share for _, share in splits)
    if total <= 0 or any(share < 0 for _, share in splits):
        message = "split shares must be non-negative and sum to more than 0"
        raise ValueError(message)
    digest = hashlib.sha256(f"{salt}:{group}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64 * total
    for name, share in splits:
        point -= share
        if point < 0:
            return name
    return splits[-1][0]


def grouped_split(
    items: Iterable[T],
    group_of: Callable[[T], str],
    splits: Sequence[tuple[str, float]] = DEFAULT_SPLITS,
    salt: str = SPLIT_SALT,
) -> dict[str, list[T]]:
    """Put each item in the split of its group (see :func:`split_of`).

    Returns:
        Every split name, in the given order, with its items in input order.
    """
    result: dict[str, list[T]] = {name: [] for name, _ in splits}
    for item in items:
        result[split_of(group_of(item), splits, salt)].append(item)
    return result


def _wilson_half_width(p: float, n: int, z: float) -> float:
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def required_sample_size(
    half_width: float, expected: float = 0.5, confidence: float = 0.95
) -> int:
    """Smallest number of items whose Wilson interval is at most +-``half_width``.

    ``expected`` is the proportion you expect (0.5 is the worst case). For a
    sensitivity, count only the positive items; for a specificity, only the
    negative ones.
    """
    if not 0 < half_width < 0.5:
        message = "half_width must be between 0 and 0.5"
        raise ValueError(message)
    if not 0 <= expected <= 1:
        message = "expected must be between 0 and 1"
        raise ValueError(message)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    # The half-width shrinks as n grows: double to pass the target, then bisect.
    low, high = 0, 1
    while _wilson_half_width(expected, high, z) > half_width:
        low, high = high, high * 2
    while high - low > 1:
        middle = (low + high) // 2
        if _wilson_half_width(expected, middle, z) > half_width:
            low = middle
        else:
            high = middle
    return high


def sample_size_targets(confidence: float = 0.95) -> dict[str, dict[str, Any]]:
    """Test-split sizes the benchmark should reach before comparing models.

    The half-widths are planning choices, not clinical standards: +-5 points
    for a top-1 accuracy near 50%, +-10 points for a sensitivity or
    specificity near 80% per finding, and +-5 points for a key-term recall
    near 80%.
    """
    plans = {
        "diagnosis-ddxplus": ("top1", 0.5, 0.05, "patients"),
        "imaging": ("sensitivity and specificity", 0.8, 0.10, "items per class"),
        "ocr": ("key_term_recall", 0.8, 0.05, "key terms"),
    }
    return {
        task: {
            "metric": metric,
            "expected": expected,
            "half_width": half_width,
            "unit": unit,
            "required": required_sample_size(half_width, expected, confidence),
        }
        for task, (metric, expected, half_width, unit) in plans.items()
    }


def class_balance(samples: Iterable[Sample]) -> dict[str, dict[str, int]]:
    """Count normal and abnormal imaging samples per task."""
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        finding = sample.truth.get("finding")
        if finding is not None:
            counts[sample.task]["normal" if finding == "normal" else "abnormal"] += 1
    return {task: dict(sorted(count.items())) for task, count in counts.items()}
