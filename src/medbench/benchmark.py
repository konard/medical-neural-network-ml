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
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from medbench.backends import Backend, BackendError
from medbench.clinical_text import (
    affirmed_mention,
    compare_doses,
    is_abstention,
    normalise_ocr,
)
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
    wilson_interval,
)
from medbench.real_data import (
    Sample,
    attribution_lines,
    fetch,
    prepare_image,
    samples_for,
)
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
    "answer_quality",
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
    """The model's answer to one item.

    ``truncated`` is set when the answer used the whole ``max_tokens``
    allowance, so it may have been cut off before the model finished.
    ``error`` is set when the backend failed (``out_of_memory: ...``,
    ``timeout: ...``); the output is then empty and is scored as wrong.
    """

    item_id: str
    output: str
    latency_s: float
    completion_tokens: int | None = None
    truncated: bool = False
    error: str | None = None


def _proportion(name: str, hits: int, total: int) -> dict[str, float]:
    """A rate with its 95% Wilson interval as ``name``/``name_ci_low``/``_high``."""
    low, high = wilson_interval(hits, total)
    return {
        name: hits / total if total else 0.0,
        f"{name}_ci_low": low,
        f"{name}_ci_high": high,
    }


def answer_quality(results: Sequence[ItemResult]) -> dict[str, float]:
    """Rates that say how far the scores can be trusted, for any task.

    * ``empty_rate`` - answers with no text at all;
    * ``abstention_rate`` - answers that decline or say they are unsure;
    * ``truncation_rate`` - answers cut off at the token limit;
    * ``error_rate`` - items where the backend failed (out of memory,
      timeout, server error), so there is no answer at all.

    Each task adds its own ``invalid_rate``: answers its parser could not
    read. All of these count as wrong in the task's scores, never as skipped.
    """
    total = len(results)
    if not total:
        return {}
    return {
        "empty_rate": sum(not r.output.strip() for r in results) / total,
        "abstention_rate": sum(is_abstention(r.output) for r in results) / total,
        "truncation_rate": sum(r.truncated for r in results) / total,
        "error_rate": sum(r.error is not None for r in results) / total,
    }


def _ranking_scores(
    truths: Sequence[str],
    rankings: Sequence[Sequence[str]],
    reference: Sequence[Sequence[str]],
    top_k: int,
) -> dict[str, float]:
    """Top-1/3/k with Wilson intervals, MRR and the reference ranking's scores."""
    total = len(truths)

    def hits(k: int) -> int:
        return sum(t in r[:k] for t, r in zip(truths, rankings, strict=True))

    unparsed = sum(not ranking for ranking in rankings)
    return {
        **_proportion("top1", hits(1), total),
        "top3": top_k_accuracy(truths, rankings, 3),
        **_proportion(f"top{top_k}", hits(top_k), total),
        "mrr": mean_reciprocal_rank(truths, rankings),
        "reference_top1": top_k_accuracy(truths, reference, 1),
        f"reference_top{top_k}": top_k_accuracy(truths, reference, top_k),
        "unparsed_answers": float(unparsed),
        "invalid_rate": unparsed / total if total else 0.0,
    }


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
        return _ranking_scores(truths, rankings, reference, self.top_k)


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
    split: str | None = None
    name: str = "diagnosis-ddxplus"
    _conditions: tuple[str, ...] = field(default=(), init=False, repr=False)

    def items(self) -> list[BenchItem]:
        """One item per patient (downloads the data on first use)."""
        data = DdxplusData.load(self.cache_dir, self.samples)
        self._conditions = tuple(data.conditions)
        return [
            BenchItem(
                patient.patient_id,
                build_ddxplus_prompt(patient, data.conditions, self.top_k),
                patient,
            )
            for patient in data.patients(self.count, self.split)
        ]

    def score(
        self, items: Sequence[BenchItem], results: Sequence[ItemResult]
    ) -> dict[str, float]:
        """Top-1/3/k and MRR against the true condition, next to DDXPlus's own."""
        reference = [list(item.truth.differential) for item in items]
        truths = [item.truth.pathology for item in items]
        known = ddxplus_conditions(
            self._conditions or sorted({*truths, *(n for r in reference for n in r)})
        )
        rankings = [parse_ranked_diagnoses(r.output, known) for r in results]
        return _ranking_scores(truths, rankings, reference, self.top_k)


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
        with suppress(ImportError):
            data = prepare_image(data, max_side)
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
        """CER/WER on normalised text, key terms read and, for prescriptions, doses.

        Text goes through :func:`medbench.clinical_text.normalise_ocr` first,
        so case, full-width characters and ``500mg`` versus ``500 mg`` are
        not errors. Doses are compared value and unit apart: a model that
        reads "5 mg" as "5 mcg" has a unit error, which keyword recall hides.
        """
        truths = [normalise_ocr(item.truth.truth["text"]) for item in items]
        outputs = [normalise_ocr(result.output) for result in results]
        found = total = 0
        for item, output in zip(items, outputs, strict=True):
            for term in item.truth.truth.get("key_terms", ()):
                total += 1
                found += (
                    re.search(
                        r"(?<!\w)" + re.escape(normalise_ocr(term)) + r"(?!\w)",
                        output,
                    )
                    is not None
                )
        cers = list(map(cer, truths, outputs))
        cer_low, cer_high = bootstrap_ci(cers)
        scores = {
            "cer": statistics.fmean(cers) if cers else 0.0,
            "cer_ci_low": cer_low,
            "cer_ci_high": cer_high,
            "wer": statistics.fmean(map(wer, truths, outputs)) if truths else 0.0,
            **_proportion("key_term_recall", found, total),
        }
        if self.kind == "prescription":
            doses = [
                compare_doses(item.truth.truth["text"], result.output)
                for item, result in zip(items, results, strict=True)
            ]
            reference = sum(d.reference for d in doses)
            if reference:
                scores.update(
                    {
                        **_proportion(
                            "dose_exact", sum(d.exact for d in doses), reference
                        ),
                        "dose_unit_errors": float(sum(d.unit_errors for d in doses)),
                        "dose_value_errors": float(sum(d.value_errors for d in doses)),
                        "doses_missing": float(sum(d.missing for d in doses)),
                        "doses_extra": float(sum(d.extra for d in doses)),
                    }
                )
        return scores


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
    # Not "frontal": the pool may hold lateral or other projections.
    "xray": "chest X-ray",
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
        """Sensitivity, specificity, finding recall and side accuracy.

        Every rate has a 95% Wilson interval. A finding counts as named only
        when one of its terms is mentioned and not negated ("no
        pneumothorax" does not name a pneumothorax).
        """
        parsed = [parse_yes_no_side(result.output) for result in results]
        truths = [item.truth.truth for item in items]
        abnormal = [truth["finding"] != "normal" for truth in truths]
        valid = [
            bool(re.search(r"abnormal\s*:\s*(yes|no)\b", result.output, re.I))
            or (
                _ANSWER.search(result.output) is not None
                and not is_abstention(result.output)
            )
            for result in results
        ]
        # Unparsed answers are errors for both classes. Otherwise an empty
        # answer earns a true negative on every normal image.
        predictions = [
            p if ok else not truth
            for (p, _), ok, truth in zip(parsed, valid, abnormal, strict=True)
        ]
        counts = BinaryCounts.from_predictions(abnormal, predictions)
        named = [
            any(affirmed_mention(result.output, term) for term in truth["terms"])
            for truth, result, sick in zip(truths, results, abnormal, strict=True)
            if sick
        ]
        sides = [
            side == truth["side"]
            for truth, (positive, side) in zip(truths, parsed, strict=True)
            if truth["side"] in {"left", "right", "both"} and positive
        ]
        positives = counts.true_positive + counts.false_negative
        negatives = counts.true_negative + counts.false_positive
        unparsed = sum(not ok for ok in valid)
        return {
            **_proportion("sensitivity", counts.true_positive, positives),
            **_proportion("specificity", counts.true_negative, negatives),
            **_proportion(
                "accuracy",
                counts.true_positive + counts.true_negative,
                positives + negatives,
            ),
            **_proportion("finding_recall", sum(named), len(named)),
            "side_accuracy_when_found": sum(sides) / len(sides) if sides else 0.0,
            "unparsed_answers": float(unparsed),
            "invalid_rate": unparsed / len(valid) if valid else 0.0,
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
    name: str,
    count: int,
    seed: int = 0,
    cache_dir: Path | None = None,
    split: str | None = None,
) -> Task:
    """Create a task by name (see :data:`TASKS`).

    ``seed`` only changes the generated diagnosis cases; real files always
    come in manifest order. ``cache_dir`` is where real files are kept.
    ``split`` picks the DDXPlus patient split (see
    :meth:`medbench.ddxplus.DdxplusData.patients`); the real images and pages
    are a small test-only set and ignore it.

    Raises:
        KeyError: If the name is unknown.
    """
    if count <= 0:
        msg = "count must be positive"
        raise ValueError(msg)
    if name == "diagnosis":
        return DiagnosisTask(count=count, seed=seed)
    if name == "diagnosis-ddxplus":
        return DdxplusTask(count=count, cache_dir=cache_dir, split=split)
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
    """Scores, speed and raw answers of one backend on one task.

    ``items_requested`` is the count asked for; ``items_total`` is how many
    items the task could build (fewer when the data has fewer files) and
    ``items_run`` how many were answered before the time budget ran out.
    ``coverage`` is ``items_run / items_requested``.
    """

    task: str
    backend: str
    items_total: int
    items_run: int
    stopped_early: bool
    metrics: dict[str, float]
    latency: dict[str, float]
    results: list[ItemResult]
    sources: list[str] = field(default_factory=list)
    items_requested: int | None = None

    @property
    def coverage(self) -> float:
        """Share of the requested items that were answered."""
        requested = self.items_requested or self.items_total
        return self.items_run / requested if requested else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-friendly dictionary."""
        return {
            "task": self.task,
            "backend": self.backend,
            "items_requested": self.items_requested or self.items_total,
            "items_total": self.items_total,
            "items_run": self.items_run,
            "coverage": round(self.coverage, 4),
            "stopped_early": self.stopped_early,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "latency": {k: round(v, 4) for k, v in self.latency.items()},
            "results": [r.__dict__ for r in self.results],
            "sources": self.sources,
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
    report says ``stopped_early``; scores cover only the items that ran, and
    ``coverage`` says how many of the requested items that was. An answer
    that used all ``max_tokens`` is marked ``truncated``. A
    :class:`~medbench.backends.BackendError` (out of memory, timeout, server
    error) is recorded as an empty answer with ``error`` set, and the run
    goes on with the next item.
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
        started = time.perf_counter()
        try:
            generation = backend.generate(
                item.prompt, item.images, max_tokens=max_tokens
            )
        except BudgetExceededError:
            stopped = True
            break
        except BackendError as error:
            # Keep going: one out-of-memory or slow item must not lose the
            # rest of the run. The failure is in the report and scored wrong.
            elapsed = time.perf_counter() - started
            results.append(
                ItemResult(item.item_id, "", elapsed, error=f"{error.kind}: {error}")
            )
            continue
        results.append(
            ItemResult(
                item.item_id,
                generation.text,
                generation.latency_s,
                generation.completion_tokens,
                truncated=generation.completion_tokens is not None
                and generation.completion_tokens >= max_tokens,
            )
        )
    done = items[: len(results)]
    if budget.remaining_seconds() <= 0:
        stopped = True
    sources = [item.truth for item in done if isinstance(item.truth, Sample)]
    if isinstance(task, DdxplusTask) and done:
        sources = list(samples_for(task.name, task.samples))
    return TaskReport(
        task=task.name,
        backend=backend.name,
        items_total=len(items),
        items_run=len(results),
        stopped_early=stopped,
        metrics={**task.score(done, results), **answer_quality(results)}
        if results
        else {},
        latency=_latency_stats([r.latency_s for r in results]),
        results=results,
        sources=attribution_lines(sources),
        items_requested=getattr(task, "count", None),
    )
