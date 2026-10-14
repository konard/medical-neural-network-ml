"""Benchmark tasks and the runner that scores a backend on them.

A task makes test items (prompt, images, truth) and scores answers. The
runner feeds every item to a backend, times it, stops at the time budget and
writes a JSON report. Three task families follow the issue:

* ``diagnosis``: ranked differential diagnosis from symptoms;
* ``ocr-prescription`` / ``ocr-history``: transcribe a rendered document;
* ``imaging-xray`` / ``imaging-mri`` / ``imaging-ultrasound``: find a
  synthetic finding and its side.
"""

from __future__ import annotations

import io
import json
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from medbench.backends import Backend
from medbench.hardware import BudgetExceededError, ResourceBudget
from medbench.metrics import (
    BinaryCounts,
    bootstrap_ci,
    cer,
    mean_reciprocal_rank,
    top_k_accuracy,
    wer,
)
from medbench.synthetic_cases import (
    build_diagnosis_prompt,
    generate_cases,
    parse_ranked_diagnoses,
)
from medbench.synthetic_documents import (
    SyntheticDocument,
    generate_history,
    generate_prescription,
    render_document,
)
from medbench.synthetic_images import encode_png, generate_phantoms

__all__ = [
    "TASKS",
    "BenchItem",
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


_OCR_PROMPT = (
    "Transcribe all text in this synthetic medical document exactly as written, "
    "line by line. Output only the text."
)


def _png_bytes(document: SyntheticDocument, style: str, seed: int) -> bytes:
    buffer = io.BytesIO()
    render_document(document, style=style, seed=seed).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class OcrTask:
    """Transcription of rendered prescriptions or histories (needs Pillow)."""

    kind: str = "prescription"
    style: str = "handwritten"
    count: int = 10
    seed: int = 0
    name: str = field(init=False)

    def __post_init__(self) -> None:
        """Set the task name from the document kind."""
        self.name = f"ocr-{self.kind}"

    def items(self) -> list[BenchItem]:
        """One rendered document per item."""
        make = (
            generate_prescription if self.kind == "prescription" else generate_history
        )
        documents = [make(self.seed + i) for i in range(self.count)]
        return [
            BenchItem(
                doc.doc_id,
                _OCR_PROMPT,
                doc,
                (_png_bytes(doc, self.style, self.seed + i),),
            )
            for i, doc in enumerate(documents)
        ]

    def score(
        self, items: Sequence[BenchItem], results: Sequence[ItemResult]
    ) -> dict[str, float]:
        """Mean CER/WER and, for prescriptions, the share of drugs read right."""
        cers = [
            cer(item.truth.text, result.output) for item, result in zip(items, results)
        ]
        wers = [
            wer(item.truth.text, result.output) for item, result in zip(items, results)
        ]
        scores = {
            "cer": statistics.fmean(cers) if cers else 0.0,
            "wer": statistics.fmean(wers) if wers else 0.0,
        }
        if self.kind == "prescription":
            found = total = 0
            for item, result in zip(items, results):
                output = result.output.lower()
                for entry in item.truth.fields["items"]:
                    total += 1
                    found += (
                        entry["drug"] in output and entry["strength"].lower() in output
                    )
            scores["drug_and_strength_recall"] = found / total if total else 0.0
        return scores


_ANSWER = re.compile(r"\b(yes|no|present|absent|normal|not)\b", re.IGNORECASE)


def parse_yes_no_side(text: str) -> tuple[bool, str]:
    """Read a yes/no answer and a side (``left``/``right``/``none``).

    The first answer word decides: "Yes, in the left lung" is positive, "No
    nodule is seen" and "There is not a lesion" are negative.
    """
    lowered = text.lower()
    match = _ANSWER.search(lowered)
    positive = match is not None and match.group(1) in {"yes", "present"}
    side = "none"
    if positive:
        left, right = lowered.find("left"), lowered.find("right")
        if left >= 0 and (right < 0 or left < right):
            side = "left"
        elif right >= 0:
            side = "right"
    return positive, side


@dataclass
class ImagingTask:
    """Detect a synthetic finding and its side on a phantom image."""

    modality: str = "xray"
    count: int = 10
    seed: int = 0
    size: int = 128
    name: str = field(init=False)

    def __post_init__(self) -> None:
        """Set the task name from the modality."""
        self.name = f"imaging-{self.modality}"

    def items(self) -> list[BenchItem]:
        """Balanced phantoms: half with a finding."""
        return [
            BenchItem(p.image_id, p.question, p, (encode_png(p.pixels),))
            for p in generate_phantoms(self.modality, self.count, self.seed, self.size)
        ]

    def score(
        self, items: Sequence[BenchItem], results: Sequence[ItemResult]
    ) -> dict[str, float]:
        """Sensitivity, specificity, accuracy and side accuracy."""
        parsed = [parse_yes_no_side(result.output) for result in results]
        truths = [item.truth.has_finding for item in items]
        counts = BinaryCounts.from_predictions(truths, [p for p, _ in parsed])
        sides = [
            side == item.truth.side
            for item, (positive, side) in zip(items, parsed)
            if item.truth.has_finding and positive
        ]
        return {
            "sensitivity": counts.sensitivity,
            "specificity": counts.specificity,
            "accuracy": counts.accuracy,
            "side_accuracy_when_found": sum(sides) / len(sides) if sides else 0.0,
        }


TASKS = (
    "diagnosis",
    "ocr-prescription",
    "ocr-history",
    "imaging-xray",
    "imaging-mri",
    "imaging-ultrasound",
)


def make_task(name: str, count: int, seed: int = 0) -> Task:
    """Create a task by name (see :data:`TASKS`).

    Raises:
        KeyError: If the name is unknown.
    """
    if name == "diagnosis":
        return DiagnosisTask(count=count, seed=seed)
    if name.startswith("ocr-") and name in TASKS:
        return OcrTask(kind=name.removeprefix("ocr-"), count=count, seed=seed)
    if name.startswith("imaging-") and name in TASKS:
        return ImagingTask(
            modality=name.removeprefix("imaging-"), count=count, seed=seed
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
