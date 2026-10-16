"""Ranked diagnosis parsing against all 49 DDXPlus choices (issue #7).

The offline tests use :data:`medbench.ddxplus.CONDITION_NAMES`; the network
test checks that list against the pinned ``release_conditions.json``.
"""

from __future__ import annotations

import json
import os

import pytest

from medbench.ddxplus import ALIASES, CONDITION_NAMES, ddxplus_conditions
from medbench.real_data import fetch, samples_for
from medbench.synthetic_cases import parse_ranked_diagnoses

KNOWN = ddxplus_conditions(CONDITION_NAMES)


def test_there_are_49_sorted_unique_choices() -> None:
    assert len(CONDITION_NAMES) == 49
    assert list(CONDITION_NAMES) == sorted(set(CONDITION_NAMES))
    assert set(ALIASES) <= set(CONDITION_NAMES)


@pytest.mark.parametrize("name", CONDITION_NAMES)
def test_every_choice_parses_to_itself(name: str) -> None:
    for line in (f"1. {name}", f"1) **{name}** - most likely", f"- {name.upper()}"):
        assert parse_ranked_diagnoses(line, KNOWN) == [name], line


@pytest.mark.parametrize(
    ("alias", "name"),
    [(alias, name) for name, aliases in ALIASES.items() for alias in aliases],
)
def test_every_alias_parses_to_its_choice(alias: str, name: str) -> None:
    assert parse_ranked_diagnoses(f"1. {alias}", KNOWN) == [name]


def test_full_ranked_list_keeps_order_and_drops_duplicates() -> None:
    text = (
        "Here is my differential:\n"
        "1. **Unstable angina** (most likely)\n"
        "2. Stable angina\n"
        "3. Possible NSTEMI / STEMI\n"
        "4. Unstable angina\n"
        "5. GERD - reflux symptoms\n"
    )
    assert parse_ranked_diagnoses(text, KNOWN) == [
        "Unstable angina",
        "Stable angina",
        "Possible NSTEMI / STEMI",
        "GERD",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Whole words only: these used to be read as SLE, Influenza,
        # Pericarditis, HIV and NSTEMI / STEMI.
        ("1. Measles", ["measles"]),
        ("1. Parainfluenza virus infection", ["parainfluenza virus infection"]),
        ("1. Myopericarditis", ["myopericarditis"]),
        ("1. Archived note", ["archived note"]),
        ("1. Systemic sclerosis", ["systemic sclerosis"]),
        # A line that rules a diagnosis out is not a ranked guess.
        ("1. Not pneumonia\n2. Influenza", ["Influenza"]),
        ("1. Pneumonia (ruled out)\n2. Bronchitis", ["Bronchitis"]),
        ("1. Pneumonia - unlikely\n2. Influenza", ["Influenza"]),
        ("1. Pneumonia cannot be ruled out", ["Pneumonia"]),
        ("1. Influenza - no rash, fever", ["Influenza"]),
        # Longest name first.
        ("1. Unstable angina", ["Unstable angina"]),
        ("1. Chronic sinusitis", ["Chronic rhinosinusitis"]),
        ("1. Tension pneumothorax", ["Spontaneous pneumothorax"]),
        # Unknown names stay as misses, prose without a list is unparsed.
        ("1. Atrial flutter", ["atrial flutter"]),
        ("I cannot give medical advice.", []),
        ("", []),
    ],
)
def test_adversarial_outputs(text: str, expected: list[str]) -> None:
    assert parse_ranked_diagnoses(text, KNOWN) == expected


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("MEDBENCH_NETWORK_TESTS") != "1",
    reason="set MEDBENCH_NETWORK_TESTS=1 to download the real files",
)
def test_choices_match_the_pinned_conditions_file() -> None:
    sample = next(
        s for s in samples_for("diagnosis-ddxplus") if s.truth["file"] == "conditions"
    )
    conditions = json.loads(fetch(sample).read_text(encoding="utf-8"))
    names = sorted(value["cond-name-eng"] for value in conditions.values())
    assert tuple(names) == CONDITION_NAMES
