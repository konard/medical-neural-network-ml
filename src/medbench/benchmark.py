"""Benchmark tasks and the runner that scores a backend on them.

A task makes test items (prompt, images, truth) and scores answers. The
runner feeds every item to a backend, times it, stops at the time budget and
writes a JSON report. Three task families follow the issue:

* ``diagnosis``: ranked differential diagnosis from symptoms of generated
  patients with an exact answer;
* ``diagnosis-ddxplus``: the same on DDXPlus patients (realistic medical
  interviews, CC BY 4.0);
* ``ocr-prescription`` / ``ocr-history``: transcribe a real handwritten
  prescription or a real printed case history;
* ``imaging-xray`` / ``imaging-mri`` / ``imaging-ultrasound``: say whether
  a real image is abnormal, name the finding and its side.

The documents and images are real, openly licensed files that are
downloaded on first use (see :mod:`medbench.real_data`). Only the
``diagnosis`` cases are generated: they are short texts with an exact
answer, used to check the scoring and the naive Bayes reference.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from medbench.backends import Backend
from medbench.ddxplus import (
    DdxplusData,
    build_ddxplus_prompt,
    ddxplus_conditions,
)
from medbench.hardware import BudgetExceededError, ResourceBudget
from medbench.metrics import (
    BinaryCounts,
    bootstrap_ci,
    cer,
    mean_reciprocal_rank,
    top_k_accuracy,
    wer,
)
from medbench.real_data import Sample, fetch, prepare_image, samples_for
from medbench.synthetic_cases import (
    build_diagnosis_prompt,
    generate_cases,
    parse_ranked_diagnoses,
)

__all__ = [
    "TASKS",
    "BenchItem",
    "DdxplusTask",
    "DiagnosisTask",
    "ImagingTask",
    "ItemResult",
    "OcrTask",
    "Task",
    "TaskReport",
    "make_task",
    "parse_yes_no_side",
    "run_task",
]


@dataclass(frozen=True)
class BenchItem:
    """One question for the model."""

    item_id: str
    prompt: str
    truth: Any
    images: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class ItemResult:
    """The model's answer to one item."""

    item_id: str
    output: str
    latency_s: float
    completion_tokens: int | None = None


class Task(Protocol):
    """A benchmark task."""

    name: str

    def items(self) -> list[BenchItem]:
        """Build the test items."""
        ...

    def score(
        self, items: Sequence[BenchItem], results: Sequence[ItemResult]
    ) -> dict[str, float]:
        """Score answers; ``results[i]`` answers ``items[i]``."""
        ...


@dataclass
class DiagnosisTask:
    """Differential diagnosis from synthetic symptoms."""

    count: int = 20
    seed: int = 0
    top_k: int = 5
    name: str = "diagnosis"

    def items(self) -> list[BenchItem]:
        """One item per synthetic patient."""
        return [
            BenchItem(case.case_id, build_diagnosis_prompt(case, self.top_k), case)
            for case in generate_cases(self.count, self.seed)
        ]

    def score(
        self, items: Sequence[BenchItem], results: Sequence[ItemResult]
    ) -> dict[str, float]:
        """Top-1/3/k accuracy and MRR, next to the Bayes reference."""
        truths = [item.truth.diagnosis for item in items]
        rankings = [parse_ranked_diagnoses(result.output) for result in results]
        reference = [list(item.truth.reference_ranking) for item in items]
        hits = [float(t in r[:1]) for t, r in zip(truths, rankings)]
        low, high = bootstrap_ci(hits)
        return {
            "top1": top_k_accuracy(truths, rankings, 1),
            "top1_ci_low": low,
            "top1_ci_high": high,
            "top3": top_k_accuracy(truths, rankings, 3),
            f"top{self.top_k}": top_k_accuracy(truths, rankings, self.top_k),
            "mrr": mean_reciprocal_rank(truths, rankings),
            "reference_top1": top_k_accuracy(truths, reference, 1),
            f"reference_top{self.top_k}": top_k_accuracy(truths, reference, self.top_k),
            "unparsed_answers": float(sum(not ranking for ranking in rankings)),
        }


@dataclass
class DdxplusTask:
    """Ranked differential diagnosis on DDXPlus test patients.

    The interview answers come from :mod:`medbench.ddxplus`; the truth is
    the patient's condition, and the expert system's own differential is
    reported as the reference a model should approach.
    """

    count: int = 20
    top_k: int = 5
    cache_dir: Path | None = None
    samples: Sequence[Sample] | None = None
    name: str = "diagnosis-ddxplus"

    def items(self) -> list[BenchItem]:
        """One item per patient (downloads the data on first use)."""
        data = DdxplusData.load(self.cache_dir, self.samples)
        return [
            BenchItem(
                patient.patient_id,
                build_ddxplus_prompt(patient, data.conditions, self.top_k),
                patient,
            )
            for patient in data.patients(self.count)
        ]

    def score(
        self, items: Sequence[BenchItem], results: Sequence[ItemResult]
    ) -> dict[str, float]:
        """Top-1/3/k and MRR against the true condition, next to DDXPlus's own."""
        reference = [list(item.truth.differential) for item in items]
        truths = [item.truth.pathology for item in items]
        known = ddxplus_conditions(
            sorted({*truths, *(n for r in reference for n in r)})
        )
        rankings = [parse_ranked_diagnoses(r.output, known) for r in results]
        hits = [float(t in r[:1]) for t, r in zip(truths, rankings)]
        low, high = bootstrap_ci(hits)
        k = self.top_k
        return {
            "top1": top_k_accuracy(truths, rankings, 1),
            "top1_ci_low": low,
            "top1_ci_high": high,
            "top3": top_k_accuracy(truths, rankings, 3),
            f"top{k}": top_k_accuracy(truths, rankings, k),
            "mrr": mean_reciprocal_rank(truths, rankings),
            "reference_top1": top_k_accuracy(truths, reference, 1),
            f"reference_top{k}": top_k_accuracy(truths, reference, k),
            "unparsed_answers": float(sum(not ranking for ranking in rankings)),
        }


def _real_items(
    task: str,
    samples: Sequence[Sample] | None,
    cache_dir: Path | None,
    count: int,
    max_side: int = 1024,
) -> list[tuple[Sample, bytes]]:
    """The first ``count`` real samples of ``task`` as PNG-ready bytes."""
    chosen = samples_for(task, samples)[:count]
    items = []
    for sample in chosen:
        data = fetch(sample, cache_dir).read_bytes()
        try:
            data = prepare_image(data, max_side)
        except ImportError:  # pragma: no cover - Pillow is in the dev extra
            pass
        items.append((sample, data))
    return items


_OCR_PROMPTS = {
    "prescription": (
        "This is one handwritten word or dose cut from a real medical "
        "prescription. Transcribe it exactly as written. Output only the text."
    ),
    "history": (
        "This is a scanned page of a real printed medical case history. "
        "Transcribe all text exactly as written, line by line. Output only "
        "the text."
    ),
}


def _normalise(text: str) -> str:
    """Lower case, no punctuation; keeps decimal points, ``%`` and ``/``."""
    kept = re.sub(r"[^\w%/.]+", " ", text.lower())
    return " ".join(re.sub(r"(?<!\d)\.|\.(?!\d)", " ", kept).split())


@dataclass
class OcrTask:
    """Transcription of real scanned or photographed medical documents.

    Files come from :mod:`medbench.real_data` and are downloaded on first
    use; ``truth`` holds the human transcription and the key terms (drug
    names, doses, diagnoses) a useful reader must get right.
    """

    kind: str = "prescription"
    count: int = 10
    samples: Sequence[Sample] | None = None
    cache_dir: Path | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        """Set the task name from the document kind."""
        self.name = f"ocr-{self.kind}"

    def items(self) -> list[BenchItem]:
        """One real document per item."""
        return [
            BenchItem(sample.sample_id, _OCR_PROMPTS[self.kind], sample, (image,))
            for sample, image in _real_items(
                self.name, self.samples, self.cache_dir, self.count, max_side=2048
            )
        ]

    def score(
        self, items: Sequence[BenchItem], results: Sequence[ItemResult]
    ) -> dict[str, float]:
        """Mean CER/WER on normalised text and the share of key terms read."""
        truths = [_normalise(item.truth.truth["text"]) for item in items]
        outputs = [_normalise(result.output) for result in results]
        found = total = 0
        for item, output in zip(items, outputs):
            for term in item.truth.truth.get("key_terms", ()):
                total += 1
                found += _normalise(term) in output
        return {
            "cer": statistics.fmean(map(cer, truths, outputs)) if truths else 0.0,
            "wer": statistics.fmean(map(wer, truths, outputs)) if truths else 0.0,
            "key_term_recall": found / total if total else 0.0,
        }


_ANSWER = re.compile(r"\b(yes|no|present|absent|normal|not)\b", re.IGNORECASE)


def parse_yes_no_side(text: str) -> tuple[bool, str]:
    """Read a yes/no answer and a side (``left``/``right``/``none``).

    The first answer word decides: "Yes, in the left lung" is positive, "No
    nodule is seen" and "There is not a lesion" are negative. A line such as
    ``abnormal: yes; finding: effusion; side: left`` is read by its fields.
    """
    lowered = text.lower()
    field_match = re.search(r"abnormal\s*:\s*(yes|no)", lowered)
    if field_match:
        positive = field_match.group(1) == "yes"
    else:
        match = _ANSWER.search(lowered)
        positive = match is not None and match.group(1) in {"yes", "present"}
    side = "none"
    if positive:
        side_match = re.search(r"side\s*:\s*(left|right|both|bilateral)", lowered)
        if side_match:
            side = {"bilateral": "both"}.get(side_match.group(1), side_match.group(1))
        else:
            left, right = lowered.find("left"), lowered.find("right")
            if left >= 0 and (right < 0 or left < right):
                side = "left"
            elif right >= 0:
                side = "right"
    return positive, side


_IMAGING_PROMPT = (
    "This is a real {what}. Is there an abnormal finding? Answer on one line "
    "in this form: abnormal: yes or no; finding: <the main finding, or none>; "
    "side: <left, right, both or none, as the patient's side>."
)

_MODALITY = {
    "xray": "frontal chest X-ray",
    "mri": "brain MRI slice",
    "ultrasound": "ultrasound image",
}


def _ordered(pool: list[Sample]) -> list[Sample]:
    """Interleave abnormal and normal samples so any prefix is balanced."""
    abnormal = [s for s in pool if s.truth["finding"] != "normal"]
    normal = [s for s in pool if s.truth["finding"] == "normal"]
    mixed: list[Sample] = []
    for index in range(max(len(abnormal), len(normal))):
        mixed.extend(group[index] for group in (abnormal, normal) if index < len(group))
    return mixed


@dataclass
class ImagingTask:
    """Find and name the finding on a real X-ray, MRI or ultrasound image.

    Labels come from the radiologist or physician who published the image
    (see each sample's ``source_url``). Normal images are the negatives.
    """

    modality: str = "xray"
    count: int = 10
    samples: Sequence[Sample] | None = None
    cache_dir: Path | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        """Set the task name from the modality."""
        self.name = f"imaging-{self.modality}"

    def items(self) -> list[BenchItem]:
        """Real images, alternating abnormal and normal."""
        pool = list(samples_for(self.name, self.samples))
        pairs = _real_items(self.name, _ordered(pool), self.cache_dir, self.count)
        prompt = _IMAGING_PROMPT.format(what=_MODALITY[self.modality])
        return [
            BenchItem(sample.sample_id, prompt, sample, (image,))
            for sample, image in pairs
        ]

    def score(
        self, items: Sequence[BenchItem], results: Sequence[ItemResult]
    ) -> dict[str, float]:
        """Sensitivity, specificity, finding recall and side accuracy."""
        parsed = [parse_yes_no_side(result.output) for result in results]
        truths = [item.truth.truth for item in items]
        abnormal = [truth["finding"] != "normal" for truth in truths]
        counts = BinaryCounts.from_predictions(abnormal, [p for p, _ in parsed])
        named = [
            any(term in result.output.lower() for term in truth["terms"])
            for truth, result, sick in zip(truths, results, abnormal)
            if sick
        ]
        sides = [
            side == truth["side"]
            for truth, (positive, side) in zip(truths, parsed)
            if truth["side"] in {"left", "right", "both"} and positive
        ]
        return {
            "sensitivity": counts.sensitivity,
            "specificity": counts.specificity,
            "accuracy": counts.accuracy,
            "finding_recall": sum(named) / len(named) if named else 0.0,
            "side_accuracy_when_found": sum(sides) / len(sides) if sides else 0.0,
        }


TASKS = (
    "diagnosis",
    "diagnosis-ddxplus",
    "ocr-prescription",
    "ocr-history",
    "imaging-xray",
    "imaging-mri",
    "imaging-ultrasound",
)


def make_task(
    name: str, count: int, seed: int = 0, cache_dir: Path | None = None
) -> Task:
    """Create a task by name (see :data:`TASKS`).

    ``seed`` only changes the generated diagnosis cases; real files always
    come in manifest order. ``cache_dir`` is where real files are kept.

    Raises:
        KeyError: If the name is unknown.
    """
    if name == "diagnosis":
        return DiagnosisTask(count=count, seed=seed)
    if name == "diagnosis-ddxplus":
        return DdxplusTask(count=count, cache_dir=cache_dir)
    if name.startswith("ocr-") and name in TASKS:
        return OcrTask(kind=name.removeprefix("ocr-"), count=count, cache_dir=cache_dir)
    if name.startswith("imaging-") and name in TASKS:
        return ImagingTask(
            modality=name.removeprefix("imaging-"), count=count, cache_dir=cache_dir
        )
    message = f"unknown task {name!r}; choose from {', '.join(TASKS)}"
    raise KeyError(message)


@dataclass
class TaskReport:
    """Scores, speed and raw answers of one backend on one task."""

    task: str
    backend: str
    items_total: int
    items_run: int
    stopped_early: bool
    metrics: dict[str, float]
    latency: dict[str, float]
    results: list[ItemResult]

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-friendly dictionary."""
        return {
            "task": self.task,
            "backend": self.backend,
            "items_total": self.items_total,
            "items_run": self.items_run,
            "stopped_early": self.stopped_early,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "latency": {k: round(v, 4) for k, v in self.latency.items()},
            "results": [r.__dict__ for r in self.results],
        }

    def write(self, path: Path) -> Path:
        """Save the report as JSON and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def _latency_stats(latencies: Sequence[float]) -> dict[str, float]:
    if not latencies:
        return {"mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "total_s": 0.0}
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "mean_s": statistics.fmean(ordered),
        "p50_s": statistics.median(ordered),
        "p95_s": p95,
        "total_s": sum(ordered),
    }


def run_task(
    task: Task,
    backend: Backend,
    budget: ResourceBudget | None = None,
    max_tokens: int = 256,
) -> TaskReport:
    """Run every item of ``task`` on ``backend`` within ``budget``.

    When the time budget runs out the remaining items are skipped and the
    report says ``stopped_early``; scores cover only the items that ran.
    """
    budget = budget or ResourceBudget()
    items = task.items()
    results: list[ItemResult] = []
    stopped = False
    for item in items:
        try:
            budget.check_deadline()
        except BudgetExceededError:
            stopped = True
            break
        generation = backend.generate(item.prompt, item.images, max_tokens=max_tokens)
        results.append(
            ItemResult(
                item.item_id,
                generation.text,
                generation.latency_s,
                generation.completion_tokens,
            )
        )
    done = items[: len(results)]
    return TaskReport(
        task=task.name,
        backend=backend.name,
        items_total=len(items),
        items_run=len(results),
        stopped_early=stopped,
        metrics=task.score(done, results) if results else {},
        latency=_latency_stats([r.latency_s for r in results]),
        results=results,
    )
