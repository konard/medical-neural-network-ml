"""Negation, abstention, dose and OCR rules, and the Wilson interval.

The examples are hand-written by the developers and are NOT clinician
reviewed (issue #7). They pin down the behaviour of the rules, including
known limits, so that any change in scoring is visible in review.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from unittest import mock

import pytest

from medbench.clinical_text import (
    Dose,
    affirmed_mention,
    compare_doses,
    extract_doses,
    is_abstention,
    normalise_ocr,
)
from medbench.metrics import wilson_interval


@pytest.mark.parametrize(
    ("text", "term", "expected"),
    [
        ("abnormal: yes; finding: pneumothorax; side: left", "pneumothorax", True),
        ("Small left pneumothorax.", "pneumothorax", True),
        ("Tension pneumothoraces present", "pneumothora", True),
        ("no pneumothorax", "pneumothorax", False),
        ("Negative for pneumothorax.", "pneumothorax", False),
        ("There is no evidence of pleural effusion.", "effusion", False),
        ("Without effusion or consolidation", "consolidation", False),
        ("Pneumothorax is not seen.", "pneumothorax", False),
        ("Pneumothorax has been excluded", "pneumothorax", False),
        ("Pneumothorax cannot be ruled out", "pneumothorax", True),
        ("No effusion but a right pneumothorax", "pneumothorax", True),
        ("No acute process. Pneumothorax on the right.", "pneumothorax", True),
        ("No pneumothorax; large right effusion", "effusion", True),
        ("finding: none", "pneumothorax", False),
        # Stems must start a word.
        ("hemorrhagic contusion", "hemorrhag", True),
        ("gallstone in the gallbladder", "stone", False),
        # Known limit: "mass" is a stem, so "massive" also matches it.
        ("massive effusion", "mass", True),
    ],
)
def test_affirmed_mention(text: str, term: str, expected: bool) -> None:
    assert affirmed_mention(text, term) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Unknown", True),
        ("I am not sure.", True),
        ("I cannot determine this from the image.", True),
        ("Unable to read the handwriting", True),
        ("I don't know", True),
        ("uncertain", True),
        ("", False),
        ("abnormal: no; finding: none; side: none", False),
        ("1. Influenza\n2. URTI", False),
    ],
)
def test_is_abstention(text: str, expected: bool) -> None:
    assert is_abstention(text) is expected


def test_extract_doses_canonical_units() -> None:
    text = "Amoxicillin 500mg tds; 5% DNS 1 L; 0,5 mg; 10 µg; 100 units; 2,000 mg"
    assert extract_doses(text) == [
        Dose(500.0, "mg"),
        Dose(5.0, "%"),
        Dose(1.0, "l"),
        Dose(0.5, "mg"),
        Dose(10.0, "mcg"),
        Dose(100.0, "iu"),
        Dose(2000.0, "mg"),
    ]
    assert extract_doses("Dexter 60") == [Dose(60.0, "")]
    assert extract_doses("Napa") == []


def test_compare_doses_separates_unit_and_value_errors() -> None:
    result = compare_doses(
        "Drug A 500 mg, drug B 5 mg", "drug a 500 mcg, drug b 50 mg 7"
    )
    assert (result.reference, result.exact) == (2, 0)
    assert (result.unit_errors, result.value_errors) == (1, 1)
    assert (result.missing, result.extra) == (0, 1)
    perfect = compare_doses("5% DNS", "5 % dns")
    assert (perfect.exact, perfect.unit_errors, perfect.value_errors) == (1, 0, 0)
    assert compare_doses("Napa 500 mg", "Napa").missing == 1


def test_normalise_ocr() -> None:
    # Full-width "amox", the "fi" ligature and no space before the unit.
    assert normalise_ocr("\uff41\uff4d\uff4f\uff58 500mg, \ufb01ve  times.") == (
        "amox 500 mg five times"
    )
    assert normalise_ocr("2.5% dns; 1/2 tab") == "2.5 % dns 1/2 tab"
    assert normalise_ocr("Straße") == "strasse"


@pytest.mark.parametrize(
    ("successes", "total", "low", "high"),
    [
        (2, 2, 0.3424, 1.0),
        (0, 2, 0.0, 0.6576),
        (1, 2, 0.0945, 0.9055),
        (81, 263, 0.2553, 0.3662),
        (0, 0, 0.0, 1.0),
    ],
)
def test_wilson_interval(successes: int, total: int, low: float, high: float) -> None:
    assert wilson_interval(successes, total) == pytest.approx((low, high), abs=1e-4)


@pytest.mark.parametrize("ulps", [-1, 0, 1])
def test_wilson_interval_reaches_its_bounds_exactly(ulps: int) -> None:
    """All right gives an upper bound of exactly 1, all wrong a lower of 0.

    The closed form only gets there up to rounding: macOS 26 with Python 3.15
    returned 0.9999999999999999 for 4 out of 4 (issue #19), as this host does
    when z is one ulp lower, so z is nudged both ways here.
    """
    original = NormalDist.inv_cdf

    def nudged(self: NormalDist, p: float) -> float:
        z = original(self, p)
        for _ in range(abs(ulps)):
            z = math.nextafter(z, math.copysign(math.inf, ulps))
        return z

    with mock.patch.object(NormalDist, "inv_cdf", nudged):
        for total in range(1, 101):
            for confidence in (0.8, 0.9, 0.95, 0.99):
                assert wilson_interval(total, total, confidence)[1] == 1.0
                assert wilson_interval(0, total, confidence)[0] == 0.0


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        wilson_interval(3, 2)
