"""Tests for the scoring functions (issue #1)."""

from __future__ import annotations

import pytest

from medbench.metrics import (
    BinaryCounts,
    auroc,
    bootstrap_ci,
    cer,
    dice,
    expected_calibration_error,
    iou,
    levenshtein,
    mean_reciprocal_rank,
    top_k_accuracy,
    wer,
)


def test_levenshtein() -> None:
    """Classic examples."""
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("same", "same") == 0


def test_cer_and_wer() -> None:
    """Error rates are relative to the reference."""
    assert cer("500 mg", "500 mg") == 0
    assert cer("500 mg", "5OO mg") == pytest.approx(2 / 6)
    assert cer("a\n  b", "a b") == 0
    assert cer("", "") == 0
    assert cer("", "x") == 1
    assert wer("take one tablet daily", "take one tablet") == pytest.approx(0.25)
    assert wer("", "") == 0


def test_ranking_metrics() -> None:
    """Top-k and MRR."""
    truths = ["a", "b", "c"]
    rankings = [["a", "x"], ["x", "b"], ["x", "y"]]
    assert top_k_accuracy(truths, rankings, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(truths, rankings, 2) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(truths, rankings) == pytest.approx((1 + 0.5 + 0) / 3)
    assert top_k_accuracy([], [], 1) == 0
    assert mean_reciprocal_rank([], []) == 0


def test_binary_counts() -> None:
    """Sensitivity, specificity and accuracy."""
    counts = BinaryCounts.from_predictions(
        [True, True, False, False, True], [True, False, False, True, True]
    )
    assert counts == BinaryCounts(2, 1, 1, 1)
    assert counts.sensitivity == pytest.approx(2 / 3)
    assert counts.specificity == pytest.approx(0.5)
    assert counts.accuracy == pytest.approx(0.6)
    empty = BinaryCounts(0, 0, 0, 0)
    assert empty.sensitivity == empty.specificity == empty.accuracy == 0


def test_auroc() -> None:
    """Perfect, random-like and one-class inputs."""
    assert auroc([True, False], [0.9, 0.1]) == 1
    assert auroc([True, False], [0.1, 0.9]) == 0
    assert auroc([True, False], [0.5, 0.5]) == 0.5
    assert auroc([True, True], [0.5, 0.6]) == 0.5


def test_overlap() -> None:
    """Dice and IoU on small masks."""
    a = [[1, 1], [0, 0]]
    b = [[1, 0], [0, 0]]
    assert dice(a, b) == pytest.approx(2 / 3)
    assert iou(a, b) == pytest.approx(0.5)
    empty = [[0, 0]]
    assert dice(empty, empty) == 1
    assert iou(empty, empty) == 1


def test_calibration() -> None:
    """A model right as often as it claims has zero ECE."""
    assert expected_calibration_error([1.0, 1.0], [True, True]) == 0
    assert expected_calibration_error([0.9, 0.9], [False, False]) == pytest.approx(0.9)
    assert expected_calibration_error([], []) == 0


def test_bootstrap() -> None:
    """The interval contains the mean and narrows with more data."""
    small = bootstrap_ci([1.0, 0.0] * 5)
    large = bootstrap_ci([1.0, 0.0] * 200)
    assert small[0] <= 0.5 <= small[1]
    assert large[1] - large[0] < small[1] - small[0]
    assert bootstrap_ci([]) == (0.0, 0.0)
