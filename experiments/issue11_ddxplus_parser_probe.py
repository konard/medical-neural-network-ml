"""Probe parse_ranked_diagnoses with all 49 DDXPlus names and adversarial outputs.

Run: python experiments/issue11_ddxplus_parser_probe.py
Prints every case where the parser's first answer differs from the expected one.
"""

from __future__ import annotations

from medbench.ddxplus import ALIASES, CONDITION_NAMES, ddxplus_conditions
from medbench.synthetic_cases import parse_ranked_diagnoses

KNOWN = ddxplus_conditions(CONDITION_NAMES)
CASES = [(f"1. {name}", name) for name in CONDITION_NAMES]
CASES += [(f"1. {alias}", name) for name, aliases in ALIASES.items() for alias in aliases]
CASES += [
    ("1. Measles", "measles"),
    ("1. Parainfluenza virus infection", "parainfluenza virus infection"),
    ("1. Not pneumonia\n2. Influenza", "Influenza"),
    ("1. Pneumonia (ruled out)\n2. Bronchitis", "Bronchitis"),
    ("**1. GERD**", "GERD"),
    ("1. **Unstable angina**", "Unstable angina"),
    ("1. Stable angina", "Stable angina"),
    ("1. Myopericarditis", "myopericarditis"),
    ("1. Tension pneumothorax", "Spontaneous pneumothorax"),
    ("1. Muscle strain", "muscle strain"),
    ("1. Archived note", "archived note"),
    ("1. Acute bronchitis", "Bronchitis"),
    ("1. Bronchiolitis", "Bronchiolitis"),
    ("1. Atrial flutter", "atrial flutter"),
    ("1. Allergic rhinitis", "allergic rhinitis"),
    ("1. Systemic sclerosis", "systemic sclerosis"),
    ("1. Hurting throat", "hurting throat"),
    ("1. Pneumonia cannot be ruled out", "Pneumonia"),
    ("1. Pneumonia - unlikely\n2. Influenza", "Influenza"),
    ("1. Influenza - no rash, fever", "Influenza"),
    ("1. **Possible NSTEMI / STEMI** (most likely)", "Possible NSTEMI / STEMI"),
    ("1) Acute HIV infection", "HIV (initial infection)"),
    ("- Guillain-Barre syndrome", "Guillain-Barré syndrome"),
    ("1. Larygospasm", "Larygospasm"),
]
bad = 0
for text, expected in CASES:
    got = parse_ranked_diagnoses(text, KNOWN)
    first = got[0] if got else None
    if first != expected:
        bad += 1
        print(f"MISMATCH {text!r}: got {got!r}, expected {expected!r}")
print(f"{len(CASES) - bad}/{len(CASES)} as expected")
