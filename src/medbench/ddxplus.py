"""Diagnosis on DDXPlus patients: realistic interviews with a known answer.

DDXPlus (Fansi Tchango et al., NeurIPS 2022, CC BY 4.0) holds 1.3 million
patients. Each one is a set of answers to a structured medical interview
(symptoms and history), the true condition, and the ranked differential
diagnosis given by a clinician-built expert system. The patients are made by
that system, not copied from health records, so they carry no private data,
but their symptoms follow medical knowledge rather than a toy table. They
are the closest open stand-in for real triage notes.

The test split and the English question and condition files are downloaded
on first use through :mod:`medbench.real_data` (pinned SHA-256, nothing is
stored in the repository).
"""

from __future__ import annotations

import ast
import csv
import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medbench.real_data import Sample, fetch, samples_for
from medbench.synthetic_cases import Condition
from medbench.validity import split_of

__all__ = [
    "ALIASES",
    "CONDITION_NAMES",
    "TASK",
    "DdxPatient",
    "DdxplusData",
    "build_ddxplus_prompt",
    "ddxplus_conditions",
]

TASK = "diagnosis-ddxplus"

# All 49 ``cond-name-eng`` values of release_conditions.json, sorted. The
# spelling "Larygospasm" is the dataset's own. A network test checks this list
# against the pinned file, so parser tests can cover every choice offline.
CONDITION_NAMES: tuple[str, ...] = (
    "Acute COPD exacerbation / infection",
    "Acute dystonic reactions",
    "Acute laryngitis",
    "Acute otitis media",
    "Acute pulmonary edema",
    "Acute rhinosinusitis",
    "Allergic sinusitis",
    "Anaphylaxis",
    "Anemia",
    "Atrial fibrillation",
    "Boerhaave",
    "Bronchiectasis",
    "Bronchiolitis",
    "Bronchitis",
    "Bronchospasm / acute asthma exacerbation",
    "Chagas",
    "Chronic rhinosinusitis",
    "Cluster headache",
    "Croup",
    "Ebola",
    "Epiglottitis",
    "GERD",
    "Guillain-Barré syndrome",
    "HIV (initial infection)",
    "Influenza",
    "Inguinal hernia",
    "Larygospasm",
    "Localized edema",
    "Myasthenia gravis",
    "Myocarditis",
    "PSVT",
    "Pancreatic neoplasm",
    "Panic attack",
    "Pericarditis",
    "Pneumonia",
    "Possible NSTEMI / STEMI",
    "Pulmonary embolism",
    "Pulmonary neoplasm",
    "SLE",
    "Sarcoidosis",
    "Scombroid food poisoning",
    "Spontaneous pneumothorax",
    "Spontaneous rib fracture",
    "Stable angina",
    "Tuberculosis",
    "URTI",
    "Unstable angina",
    "Viral pharyngitis",
    "Whooping cough",
)

# Plain-language and abbreviation forms a model may answer with.
ALIASES: dict[str, tuple[str, ...]] = {
    "Acute COPD exacerbation / infection": ("copd exacerbation", "copd"),
    "Acute otitis media": ("otitis media",),
    "Acute pulmonary edema": ("pulmonary edema", "pulmonary oedema"),
    "Acute rhinosinusitis": ("acute sinusitis",),
    "Boerhaave": ("boerhaave syndrome", "esophageal rupture"),
    "Bronchospasm / acute asthma exacerbation": ("asthma exacerbation", "asthma"),
    "Chagas": ("chagas disease",),
    "Chronic rhinosinusitis": ("chronic sinusitis",),
    "Ebola": ("ebola virus disease",),
    "GERD": ("gastroesophageal reflux disease", "gastro-oesophageal reflux"),
    "Guillain-Barré syndrome": ("guillain-barre syndrome", "guillain barre"),
    "HIV (initial infection)": ("acute hiv infection", "acute hiv"),
    "Larygospasm": ("laryngospasm",),
    "Possible NSTEMI / STEMI": (
        "nstemi",
        "stemi",
        "myocardial infarction",
        "acute coronary syndrome",
    ),
    "PSVT": ("paroxysmal supraventricular tachycardia", "svt"),
    "Pancreatic neoplasm": ("pancreatic cancer",),
    "Pulmonary neoplasm": ("lung cancer", "pulmonary cancer"),
    "SLE": ("systemic lupus erythematosus", "lupus"),
    "Scombroid food poisoning": ("scombroid",),
    "Spontaneous pneumothorax": ("pneumothorax",),
    "Spontaneous rib fracture": ("rib fracture",),
    "URTI": ("upper respiratory tract infection", "common cold"),
    "Whooping cough": ("pertussis",),
}

_SEX = {"F": "female", "M": "male"}


@dataclass(frozen=True)
class DdxPatient:
    """One DDXPlus patient, decoded to English.

    Attributes:
        patient_id: ``ddxplus-test-<row>``.
        age: Age in years.
        sex: ``female`` or ``male``.
        chief_complaint: The first thing the patient reported.
        findings: Interview answers as ``question: answer`` lines.
        pathology: The true condition.
        differential: The expert system's ranked differential (names).
    """

    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    findings: tuple[str, ...]
    pathology: str
    differential: tuple[str, ...]


def _scale(value: str, evidence: dict[str, Any]) -> str:
    numbers = [v for v in evidence["possible-values"] if isinstance(v, int)]
    return f"{value} on a {min(numbers)}-{max(numbers)} scale" if numbers else value


@dataclass(frozen=True)
class DdxplusData:
    """The three DDXPlus files, opened from the local cache."""

    patients_zip: Path
    evidences: dict[str, Any]
    conditions: list[str]

    @classmethod
    def load(
        cls, cache_dir: Path | None = None, samples: Sequence[Sample] | None = None
    ) -> DdxplusData:
        """Fetch (once) and open the files listed for ``diagnosis-ddxplus``."""
        files = {s.truth["file"]: s for s in samples_for(TASK, samples)}
        paths = {name: fetch(sample, cache_dir) for name, sample in files.items()}
        evidences = json.loads(paths["evidences"].read_text(encoding="utf-8"))
        conditions = json.loads(paths["conditions"].read_text(encoding="utf-8"))
        names = sorted(c["cond-name-eng"] for c in conditions.values())
        return cls(paths["patients"], evidences, names)

    def decode(self, codes: Sequence[str]) -> tuple[str, ...]:
        """Turn evidence codes into ``question: answer`` lines."""
        answers: dict[str, list[str]] = {}
        for code in codes:
            name, _, value = code.partition("_@_")
            evidence = self.evidences[name]
            question = evidence["question_en"]
            if value and value == str(evidence["default_value"]):
                continue
            answer = answers.setdefault(question, [])
            if value:
                meaning = evidence["value_meaning"].get(value)
                answer.append(meaning["en"] if meaning else _scale(value, evidence))
        return tuple(
            f"{question} {', '.join(values) if values else 'yes'}"
            for question, values in answers.items()
        )

    def patients(self, count: int, split: str | None = None) -> list[DdxPatient]:
        """The first ``count`` patients of the DDXPlus test file, in file order.

        With ``split`` (``"train"``, ``"dev"`` or ``"test"``) only patients
        whose id :func:`medbench.validity.split_of` puts in that split are
        taken, so prompts can be tuned on ``dev`` and reported on ``test``
        without sharing a patient. Each DDXPlus row is a separate synthetic
        patient, so the row id is the patient group.
        """
        patients: list[DdxPatient] = []
        with (
            zipfile.ZipFile(self.patients_zip) as archive,
            archive.open("release_test_patients") as raw,
        ):
            rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            for index, row in enumerate(rows):
                if len(patients) >= count:
                    break
                if split is not None and split_of(f"ddxplus-test-{index}") != split:
                    continue
                initial = self.evidences[row["INITIAL_EVIDENCE"]]["question_en"]
                differential = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
                patients.append(
                    DdxPatient(
                        patient_id=f"ddxplus-test-{index}",
                        age=int(row["AGE"]),
                        sex=_SEX.get(row["SEX"], row["SEX"]),
                        chief_complaint=initial,
                        findings=self.decode(ast.literal_eval(row["EVIDENCES"])),
                        pathology=row["PATHOLOGY"],
                        differential=tuple(name for name, _ in differential),
                    )
                )
        return patients


def build_ddxplus_prompt(
    patient: DdxPatient, conditions: Sequence[str], top_k: int = 5
) -> str:
    """Ask for a ranked differential from the 49 DDXPlus conditions."""
    findings = "\n".join(f"- {line}" for line in patient.findings)
    return (
        f"A {patient.age}-year-old {patient.sex} patient answered a medical "
        f'interview. The visit started with a "yes" to: '
        f"{patient.chief_complaint}\n"
        f"Answers:\n{findings}\n\n"
        f"Choose the {top_k} most likely diagnoses from this list: "
        f"{'; '.join(conditions)}.\n"
        f"Answer as a numbered list, most likely first, one name per line."
    )


def ddxplus_conditions(names: Sequence[str]) -> tuple[Condition, ...]:
    """Conditions with their :data:`ALIASES`, for :func:`parse_ranked_diagnoses`."""
    return tuple(
        Condition(name, 0.0, {}, aliases=ALIASES.get(name, ())) for name in names
    )
