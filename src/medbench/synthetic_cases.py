"""Synthetic patients for the "initial diagnosis from symptoms" task.

Real patient records cannot be put into a public repository, so the first
tests use made-up patients. Each patient is drawn from a small, hand-written
knowledge base: pick a condition by how common it is, then switch each symptom
on with the probability listed for that condition. Because we know exactly how
the data was made, we can compute the *best possible* ranked list of
diagnoses with Bayes' rule and compare any model against it.

The probabilities are rough teaching values, not clinical guidance.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

__all__ = [
    "CONDITIONS",
    "SYMPTOMS",
    "Condition",
    "PatientCase",
    "build_diagnosis_prompt",
    "generate_cases",
    "parse_ranked_diagnoses",
    "rank_conditions",
]


@dataclass(frozen=True)
class Condition:
    """One condition in the toy knowledge base.

    Attributes:
        name: Condition name, as the model is expected to answer it.
        prevalence: Relative frequency among patients who visit a clinic.
        symptoms: Probability of each symptom when the patient has it.
        urgent: Whether the condition needs same-day care.
        aliases: Other names a model may use.
    """

    name: str
    prevalence: float
    symptoms: dict[str, float]
    urgent: bool = False
    aliases: tuple[str, ...] = ()


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "common cold",
        0.20,
        {
            "runny nose": 0.9,
            "sore throat": 0.6,
            "cough": 0.6,
            "sneezing": 0.7,
            "fever": 0.2,
        },
        aliases=("upper respiratory infection", "viral uri", "rhinitis"),
    ),
    Condition(
        "influenza",
        0.10,
        {
            "fever": 0.9,
            "muscle aches": 0.8,
            "cough": 0.8,
            "fatigue": 0.8,
            "headache": 0.6,
            "sore throat": 0.4,
            "chills": 0.6,
        },
        aliases=("flu",),
    ),
    Condition(
        "covid-19",
        0.06,
        {
            "fever": 0.7,
            "cough": 0.7,
            "fatigue": 0.7,
            "loss of smell": 0.4,
            "shortness of breath": 0.3,
            "headache": 0.5,
            "sore throat": 0.4,
        },
        aliases=("covid", "sars-cov-2 infection", "coronavirus"),
    ),
    Condition(
        "community-acquired pneumonia",
        0.04,
        {
            "fever": 0.85,
            "cough": 0.9,
            "shortness of breath": 0.7,
            "chest pain": 0.4,
            "chills": 0.5,
            "fatigue": 0.6,
        },
        urgent=True,
        aliases=("pneumonia",),
    ),
    Condition(
        "streptococcal pharyngitis",
        0.05,
        {
            "sore throat": 0.95,
            "fever": 0.8,
            "swollen neck glands": 0.7,
            "headache": 0.3,
        },
        aliases=("strep throat", "bacterial pharyngitis"),
    ),
    Condition(
        "migraine",
        0.08,
        {
            "headache": 0.98,
            "nausea": 0.6,
            "sensitivity to light": 0.7,
            "visual aura": 0.25,
            "vomiting": 0.2,
        },
    ),
    Condition(
        "acute gastroenteritis",
        0.10,
        {
            "diarrhea": 0.9,
            "nausea": 0.7,
            "vomiting": 0.6,
            "abdominal pain": 0.7,
            "fever": 0.3,
        },
        aliases=("gastroenteritis", "stomach flu", "food poisoning"),
    ),
    Condition(
        "acute appendicitis",
        0.02,
        {
            "abdominal pain": 0.98,
            "right lower abdominal pain": 0.8,
            "nausea": 0.7,
            "loss of appetite": 0.7,
            "fever": 0.4,
            "vomiting": 0.4,
        },
        urgent=True,
        aliases=("appendicitis",),
    ),
    Condition(
        "urinary tract infection",
        0.09,
        {
            "painful urination": 0.9,
            "frequent urination": 0.8,
            "lower abdominal pain": 0.5,
            "fever": 0.15,
        },
        aliases=("uti", "cystitis", "bladder infection"),
    ),
    Condition(
        "acute coronary syndrome",
        0.02,
        {
            "chest pain": 0.9,
            "shortness of breath": 0.6,
            "sweating": 0.5,
            "nausea": 0.3,
            "pain spreading to left arm": 0.4,
            "fatigue": 0.3,
        },
        urgent=True,
        aliases=("heart attack", "myocardial infarction", "unstable angina", "acs"),
    ),
    Condition(
        "asthma exacerbation",
        0.05,
        {
            "shortness of breath": 0.9,
            "wheezing": 0.85,
            "cough": 0.7,
            "chest tightness": 0.6,
        },
        aliases=("asthma attack", "asthma"),
    ),
    Condition(
        "allergic rhinitis",
        0.12,
        {"runny nose": 0.9, "sneezing": 0.9, "itchy eyes": 0.7, "cough": 0.2},
        aliases=("hay fever", "seasonal allergies"),
    ),
    Condition(
        "type 2 diabetes mellitus",
        0.04,
        {
            "frequent urination": 0.6,
            "excessive thirst": 0.6,
            "fatigue": 0.5,
            "blurred vision": 0.3,
            "weight loss": 0.2,
        },
        aliases=("diabetes", "type 2 diabetes", "t2dm"),
    ),
    Condition(
        "iron deficiency anemia",
        0.03,
        {
            "fatigue": 0.9,
            "pale skin": 0.6,
            "shortness of breath": 0.3,
            "dizziness": 0.4,
            "headache": 0.3,
        },
        aliases=("anemia", "iron deficiency"),
    ),
)

SYMPTOMS: tuple[str, ...] = tuple(
    sorted({symptom for condition in CONDITIONS for symptom in condition.symptoms})
)

# Probability that a symptom is present for a reason unrelated to the
# condition (noise). Keeps the Bayes reference from being over-confident.
_BACKGROUND_RATE = 0.03


@dataclass(frozen=True)
class PatientCase:
    """One synthetic patient.

    Attributes:
        case_id: Stable identifier.
        age: Age in years.
        sex: ``"female"`` or ``"male"``.
        symptoms: Symptoms the patient reports.
        diagnosis: The condition the patient was generated from.
        reference_ranking: Conditions ranked by posterior probability.
        reference_probabilities: Posterior probability of each ranked condition.
    """

    case_id: str
    age: int
    sex: str
    symptoms: tuple[str, ...]
    diagnosis: str
    reference_ranking: tuple[str, ...] = field(default_factory=tuple)
    reference_probabilities: tuple[float, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        """Return the case as a JSON-friendly dictionary."""
        return {
            "case_id": self.case_id,
            "age": self.age,
            "sex": self.sex,
            "symptoms": list(self.symptoms),
            "diagnosis": self.diagnosis,
            "reference_ranking": list(self.reference_ranking),
            "reference_probabilities": [
                round(p, 4) for p in self.reference_probabilities
            ],
        }


def _symptom_probability(condition: Condition, symptom: str) -> float:
    listed = condition.symptoms.get(symptom, 0.0)
    return listed + (1 - listed) * _BACKGROUND_RATE


def rank_conditions(
    symptoms: tuple[str, ...] | list[str],
    conditions: tuple[Condition, ...] = CONDITIONS,
) -> list[tuple[str, float]]:
    """Rank conditions by posterior probability with naive Bayes.

    Both present and absent symptoms count: a missing fever makes influenza
    less likely.

    Args:
        symptoms: Reported symptoms.
        conditions: Knowledge base.

    Returns:
        ``(condition, probability)`` pairs, most likely first.
    """
    present = set(symptoms)
    all_symptoms = {s for condition in conditions for s in condition.symptoms} | present
    log_scores = []
    for condition in conditions:
        score = math.log(condition.prevalence)
        for symptom in all_symptoms:
            probability = _symptom_probability(condition, symptom)
            score += math.log(probability if symptom in present else 1 - probability)
        log_scores.append(score)
    peak = max(log_scores)
    weights = [math.exp(score - peak) for score in log_scores]
    total = sum(weights)
    return sorted(
        (
            (condition.name, weight / total)
            for condition, weight in zip(conditions, weights)
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def generate_cases(
    count: int,
    seed: int = 0,
    conditions: tuple[Condition, ...] = CONDITIONS,
) -> list[PatientCase]:
    """Generate ``count`` synthetic patients deterministically.

    Args:
        count: Number of patients.
        seed: Random seed; the same seed always gives the same patients.
        conditions: Knowledge base.

    Returns:
        The patients. Every patient has at least one symptom.
    """
    rng = random.Random(seed)
    weights = [condition.prevalence for condition in conditions]
    cases = []
    for index in range(count):
        condition = rng.choices(conditions, weights=weights, k=1)[0]
        symptoms = [s for s, p in condition.symptoms.items() if rng.random() < p]
        if not symptoms:
            symptoms = [max(condition.symptoms, key=condition.symptoms.__getitem__)]
        ranking = rank_conditions(symptoms, conditions)
        cases.append(
            PatientCase(
                case_id=f"dx-{seed}-{index:04d}",
                age=rng.randint(18, 85),
                sex=rng.choice(("female", "male")),
                symptoms=tuple(symptoms),
                diagnosis=condition.name,
                reference_ranking=tuple(name for name, _ in ranking),
                reference_probabilities=tuple(p for _, p in ranking),
            )
        )
    return cases


def build_diagnosis_prompt(case: PatientCase, top_k: int = 5) -> str:
    """Build the instruction sent to a language model.

    Args:
        case: Patient to describe.
        top_k: How many diagnoses to ask for.

    Returns:
        The prompt text.
    """
    symptoms = ", ".join(case.symptoms)
    return (
        f"A {case.age}-year-old {case.sex} patient reports: {symptoms}.\n"
        f"List the {top_k} most likely diagnoses, most likely first, one per "
        "line, as a numbered list with only the diagnosis name. This is a "
        "synthetic test case for software evaluation."
    )


_NUMBERED_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*(.+?)\s*$")


def _normalize(text: str) -> str:
    text = re.sub(r"\*\*|__|`", "", text.lower())
    text = re.split(r"\s+[-\u2013\u2014:(]\s*|\s*\(", text, maxsplit=1)[0]
    return re.sub(r"[^a-z0-9 -]", "", text).strip()


# A list line that names a diagnosis only to reject it: "Not pneumonia",
# "Pneumonia (ruled out)", "Pneumonia - excluded".
_REJECTED = re.compile(
    r"^\W*(?:not|no|without|excluded:?|ruled out:?)\s"
    r"|[(\-\u2013\u2014:,]\s*(?:ruled out|excluded|unlikely|not likely|"
    r"not (?:the )?(?:diagnosis|this))\b",
    re.IGNORECASE,
)


def _rejected(line: str) -> bool:
    """Whether a list line names a diagnosis in order to rule it out."""
    plain = re.sub(r"\*\*|__|`", "", line)
    if re.search(r"(?:cannot|can't|not) be (?:ruled out|excluded)", plain, re.I):
        return False
    return _REJECTED.search(plain) is not None


def parse_ranked_diagnoses(
    text: str,
    conditions: tuple[Condition, ...] = CONDITIONS,
) -> list[str]:
    """Turn free model output into canonical condition names.

    Lines like ``"1. **Influenza** - fever and aches"`` become
    ``"influenza"``. Unknown diagnoses are kept as normalized text so that
    metrics count them as misses instead of silently dropping them. A known
    name inside a longer line counts only as whole words ("measles" is not
    "SLE", "parainfluenza" is not "influenza"), the longest name first
    ("unstable angina" is not "stable angina"). Lines that rule a diagnosis
    out ("Not pneumonia", "Pneumonia (ruled out)") are skipped.

    Args:
        text: Model output.
        conditions: Knowledge base with aliases.

    Returns:
        Ranked, de-duplicated diagnoses.
    """
    lookup: dict[str, str] = {}
    for condition in conditions:
        for name in (condition.name, *condition.aliases):
            key = _normalize(name)
            if key:
                lookup[key] = condition.name
    by_length = [
        (re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"), alias)
        for alias in sorted(lookup, key=len, reverse=True)
    ]
    ranked: list[str] = []
    for line in text.splitlines():
        match = _NUMBERED_LINE.match(line)
        if not match or _rejected(match.group(1)):
            continue
        candidate = _normalize(match.group(1))
        if not candidate:
            continue
        matched_name = lookup.get(candidate)
        if matched_name is None:
            matched_name = next(
                (
                    lookup[alias]
                    for pattern, alias in by_length
                    if pattern.search(candidate)
                ),
                candidate,
            )
        if matched_name not in ranked:
            ranked.append(matched_name)
    return ranked
