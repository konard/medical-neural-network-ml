"""Scoring functions for the benchmark tasks.

All metrics are written in plain Python so they can be read and checked
without extra libraries.

* OCR: character and word error rate (CER/WER) based on edit distance.
* Diagnosis lists: top-k accuracy and mean reciprocal rank (MRR).
* Yes/no findings: sensitivity, specificity and the area under the ROC curve.
* Segmentation masks: Dice and IoU overlap.
* Confidence quality: expected calibration error (ECE).
* Uncertainty of a proportion: Wilson score interval.
* Uncertainty of any mean: bootstrap confidence interval.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

__all__ = [
    "BinaryCounts",
    "auroc",
    "bootstrap_ci",
    "cer",
    "dice",
    "expected_calibration_error",
    "iou",
    "levenshtein",
    "mean_reciprocal_rank",
    "top_k_accuracy",
    "wer",
    "wilson_interval",
]


def levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    """Minimum number of insertions, deletions and substitutions from a to b."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        current = [i]
        for j, item_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (item_a != item_b),
                )
            )
        previous = current
    return previous[-1]


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edits divided by reference length.

    Whitespace runs are collapsed first so that line wrapping does not count
    as an error. The value can exceed 1 when the hypothesis is much longer.
    """
    reference, hypothesis = _normalize_text(reference), _normalize_text(hypothesis)
    if not reference:
        return float(bool(hypothesis))
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: word-level edits divided by reference word count."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_words:
        return float(bool(hyp_words))
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def top_k_accuracy(
    truths: Sequence[str], rankings: Sequence[Sequence[str]], k: int
) -> float:
    """Share of cases whose true answer is among the first ``k`` guesses."""
    if not truths:
        return 0.0
    hits = sum(truth in ranking[:k] for truth, ranking in zip(truths, rankings))
    return hits / len(truths)


def mean_reciprocal_rank(
    truths: Sequence[str], rankings: Sequence[Sequence[str]]
) -> float:
    """Average of 1/position of the true answer (0 when it is missing)."""
    if not truths:
        return 0.0
    total = 0.0
    for truth, ranking in zip(truths, rankings):
        if truth in ranking:
            total += 1 / (list(ranking).index(truth) + 1)
    return total / len(truths)


@dataclass(frozen=True)
class BinaryCounts:
    """Confusion counts for a yes/no question."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @classmethod
    def from_predictions(
        cls, truths: Sequence[bool], predictions: Sequence[bool]
    ) -> BinaryCounts:
        """Count agreements between truths and predictions."""
        pairs = list(zip(truths, predictions))
        return cls(
            true_positive=sum(t and p for t, p in pairs),
            false_positive=sum((not t) and p for t, p in pairs),
            true_negative=sum((not t) and (not p) for t, p in pairs),
            false_negative=sum(t and (not p) for t, p in pairs),
        )

    @property
    def sensitivity(self) -> float:
        """Share of sick cases found (recall). Missing disease is costly."""
        positives = self.true_positive + self.false_negative
        return self.true_positive / positives if positives else 0.0

    @property
    def specificity(self) -> float:
        """Share of healthy cases correctly called healthy."""
        negatives = self.true_negative + self.false_positive
        return self.true_negative / negatives if negatives else 0.0

    @property
    def accuracy(self) -> float:
        """Share of all answers that are right."""
        total = (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )
        return (self.true_positive + self.true_negative) / total if total else 0.0


def auroc(truths: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the ROC curve: chance that a sick case scores above a healthy one.

    Ties count as one half. Returns 0.5 when one of the classes is missing.
    """
    positives = [s for t, s in zip(truths, scores) if t]
    negatives = [s for t, s in zip(truths, scores) if not t]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def _overlap(
    a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]
) -> tuple[int, int, int]:
    inter = size_a = size_b = 0
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            inter += bool(x and y)
            size_a += bool(x)
            size_b += bool(y)
    return inter, size_a, size_b


def dice(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> float:
    """Dice overlap of two binary masks (1 when both are empty)."""
    inter, size_a, size_b = _overlap(a, b)
    if size_a + size_b == 0:
        return 1.0
    return 2 * inter / (size_a + size_b)


def iou(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> float:
    """Intersection over union of two binary masks (1 when both are empty)."""
    inter, size_a, size_b = _overlap(a, b)
    union = size_a + size_b - inter
    return inter / union if union else 1.0


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = 10,
) -> float:
    """Gap between stated confidence and real accuracy, averaged over bins.

    A model that says "90% sure" should be right 9 times out of 10. ECE is 0
    for a perfectly calibrated model.
    """
    if not confidences:
        return 0.0
    totals = [0] * bins
    confidence_sums = [0.0] * bins
    correct_sums = [0] * bins
    for confidence, is_correct in zip(confidences, correct):
        index = min(bins - 1, int(confidence * bins))
        totals[index] += 1
        confidence_sums[index] += confidence
        correct_sums[index] += bool(is_correct)
    error = 0.0
    for total, conf_sum, corr_sum in zip(totals, confidence_sums, correct_sums):
        if total:
            error += total / len(confidences) * abs(conf_sum / total - corr_sum / total)
    return error


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval of the mean by resampling with replacement.

    With few test cases a score is very uncertain; reporting the interval
    keeps us from celebrating noise.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    low = means[int((1 - confidence) / 2 * resamples)]
    high = means[min(resamples - 1, int((1 + confidence) / 2 * resamples))]
    return (low, high)


def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval of a proportion (Wilson, 1927, JASA 22:209-212).

    Unlike the bootstrap, it does not collapse to a single point when every
    answer is right or wrong: 2 correct out of 2 gives about (0.34, 1.0), which
    is how little two items say. ``total == 0`` returns the uninformative
    interval (0, 1).
    """
    if total <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= total:
        message = f"successes must be between 0 and {total}"
        raise ValueError(message)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= denominator
    return (max(0.0, centre - half), min(1.0, centre + half))
