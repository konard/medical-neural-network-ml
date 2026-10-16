"""Tests for the real-file manifest, downloads and real-data tasks (issue #1).

Unit tests never touch the network: downloads go through a fake opener and
the tasks read files that the test puts in the cache. The ``network`` tests
download every real file and are opt-in (``pytest -m network``).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import replace
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from medbench.backends import ScriptedBackend
from medbench.benchmark import TASKS, DdxplusTask, ImagingTask, OcrTask, run_task
from medbench.cli import main
from medbench.ddxplus import DdxplusData, build_ddxplus_prompt
from medbench.real_data import (
    ALLOWED_LICENSES,
    USER_AGENT,
    ChecksumError,
    Sample,
    attribution_lines,
    fetch,
    load_manifest,
    prepare_image,
    samples_for,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402 - after the importorskip guard

MANIFEST = load_manifest()


def _image_bytes(size: tuple[int, int] = (40, 30), fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("L", size, color=90).save(buffer, format=fmt)
    return buffer.getvalue()


def _sample(sample_id: str, data: bytes, task: str, **truth: Any) -> Sample:
    return Sample(
        sample_id=sample_id,
        task=task,
        url=f"https://example.org/{sample_id}.jpg",
        sha256=hashlib.sha256(data).hexdigest(),
        license="CC0 1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Test Author",
        source_url=f"https://example.org/{sample_id}",
        truth=truth,
    )


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _Opener:
    """Fake ``urlopen`` that serves fixed bytes and records requests."""

    def __init__(self, data: bytes):
        self.data = data
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> _Response:
        del timeout
        self.requests.append(request)
        return _Response(self.data)


class _FailingOpener(_Opener):
    """Answers with HTTP errors first, then serves the bytes."""

    def __init__(self, data: bytes, errors: Sequence[tuple[int, dict[str, str]]]):
        super().__init__(data)
        self.errors = list(errors)

    def __call__(self, request: urllib.request.Request, timeout: float) -> _Response:
        if self.errors:
            self.requests.append(request)
            code, headers = self.errors.pop(0)
            message = Message()
            for name, value in headers.items():
                message[name] = value
            raise urllib.error.HTTPError(request.full_url, code, "no", message, None)
        return super().__call__(request, timeout)


class TestManifest:
    """The packaged list of real files."""

    def test_ids_are_unique(self) -> None:
        """Every sample has its own id."""
        ids = [sample.sample_id for sample in MANIFEST]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("sample", MANIFEST, ids=lambda s: s.sample_id)
    def test_sample_is_open_and_pinned(self, sample: Sample) -> None:
        """Only open licenses, HTTPS sources and SHA-256 pins."""
        assert sample.license in ALLOWED_LICENSES
        assert not re.search(r"\b(NC|ND)\b", sample.license)
        assert sample.url.startswith("https://")
        assert sample.source_url.startswith("https://")
        assert sample.license_url.startswith("https://")
        assert re.fullmatch(r"[0-9a-f]{64}", sample.sha256)
        assert sample.author.strip()
        assert sample.task in TASKS
        if sample.archive_member:
            assert re.fullmatch(r"[0-9a-f]{64}", sample.archive_sha256)

    @pytest.mark.parametrize("sample", MANIFEST, ids=lambda s: s.sample_id)
    def test_truth_fits_the_task(self, sample: Sample) -> None:
        """Imaging samples name a finding and side; OCR samples have text."""
        truth = sample.truth
        if sample.task == "diagnosis-ddxplus":
            assert truth["file"] in {"patients", "evidences", "conditions"}
        elif sample.task.startswith("imaging-"):
            assert truth["side"] in {"left", "right", "both", "none"}
            if truth["finding"] == "normal":
                assert truth["side"] == "none"
                assert truth["terms"] == []
            else:
                assert truth["terms"]
                assert all(term == term.lower() for term in truth["terms"])
        else:
            assert truth["text"].strip()
            assert truth["key_terms"]

    @pytest.mark.parametrize("task", [t for t in TASKS if t.startswith("imaging-")])
    def test_imaging_tasks_have_both_classes(self, task: str) -> None:
        """Each imaging task has abnormal and normal images."""
        findings = {s.truth["finding"] == "normal" for s in samples_for(task)}
        assert findings == {True, False}

    @pytest.mark.parametrize("task", [t for t in TASKS if t.startswith("imaging-")])
    def test_small_runs_see_both_classes(self, task: str) -> None:
        """CI runs 2 items, so abnormal and normal images alternate."""
        first = [s.truth["finding"] == "normal" for s in samples_for(task)[:2]]
        assert first == [False, True]

    @pytest.mark.parametrize(
        "task", [t for t in TASKS if t.startswith(("ocr-", "imaging-"))]
    )
    def test_every_media_task_has_samples(self, task: str) -> None:
        """No media task falls back to generated pictures."""
        assert len(samples_for(task)) >= 2


class TestFetch:
    """Downloading, checking and caching."""

    def test_download_is_verified_and_cached(self, tmp_path: Path) -> None:
        """The first call downloads with a User-Agent, the second reuses it."""
        data = _image_bytes()
        sample = _sample("a", data, "imaging-xray")
        opener = _Opener(data)
        path = fetch(sample, tmp_path, opener=opener)
        assert path == tmp_path / "a.jpg"
        assert path.read_bytes() == data
        assert opener.requests[0].get_header("User-agent") == USER_AGENT
        assert fetch(sample, tmp_path, opener=opener) == path
        assert len(opener.requests) == 1

    def test_changed_file_is_refused(self, tmp_path: Path) -> None:
        """Bytes that do not match the pin are not stored."""
        sample = _sample("b", b"expected", "imaging-xray")
        with pytest.raises(ChecksumError, match="does not match"):
            fetch(sample, tmp_path, opener=_Opener(b"tampered"))
        assert list(tmp_path.iterdir()) == []

    def test_corrupt_cache_is_replaced(self, tmp_path: Path) -> None:
        """A damaged cached copy is downloaded again."""
        sample = _sample("c", b"good", "imaging-xray")
        (tmp_path / "c.jpg").write_bytes(b"bad")
        fetch(sample, tmp_path, opener=_Opener(b"good"))
        assert (tmp_path / "c.jpg").read_bytes() == b"good"

    def test_archive_member(self, tmp_path: Path) -> None:
        """Files published inside a zip share one checked download."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("set/P1.jpg", b"one")
            archive.writestr("set/P2.jpg", b"two")
        zipped = buffer.getvalue()
        opener = _Opener(zipped)
        for name, data in (("P1", b"one"), ("P2", b"two")):
            sample = replace(
                _sample(name, data, "ocr-prescription"),
                url="https://example.org/set.zip",
                archive_member=f"set/{name}.jpg",
                archive_sha256=hashlib.sha256(zipped).hexdigest(),
            )
            assert fetch(sample, tmp_path, opener=opener).read_bytes() == data
            assert sample.filename == f"{name}.jpg"
        assert len(opener.requests) == 1
        bad = replace(sample, sample_id="P3", archive_sha256="0" * 64)
        with pytest.raises(ChecksumError, match="archive"):
            fetch(bad, tmp_path / "other", opener=opener)

    def test_rate_limit_waits_as_asked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 429 or 503 is retried after the server's Retry-After delay."""
        waits: list[float] = []
        monkeypatch.setattr("medbench.real_data.time.sleep", waits.append)
        errors = [(429, {"Retry-After": "7"}), (503, {})]
        opener = _FailingOpener(b"ok", errors)
        sample = _sample("r", b"ok", "imaging-xray")
        assert fetch(sample, tmp_path, opener=opener, attempts=4).read_bytes() == b"ok"
        assert waits == [7.0, 10.0]
        assert len(opener.requests) == 3

    def test_long_retry_after_fails_without_an_early_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A short CI budget must not cause requests before the server allows."""
        waits: list[float] = []
        monkeypatch.setattr("medbench.real_data.time.sleep", waits.append)
        opener = _FailingOpener(b"ok", [(429, {"Retry-After": "900"})])
        with pytest.raises(urllib.error.HTTPError):
            fetch(_sample("r", b"ok", "imaging-xray"), tmp_path, opener=opener)
        assert waits == []
        assert len(opener.requests) == 1

    @pytest.mark.parametrize("code", [403, 404])
    def test_permanent_errors_fail_at_once(
        self, code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing or forbidden file is not asked for again."""
        waits: list[float] = []
        monkeypatch.setattr("medbench.real_data.time.sleep", waits.append)
        opener = _FailingOpener(b"ok", [(code, {})])
        with pytest.raises(urllib.error.HTTPError):
            fetch(_sample("p", b"ok", "imaging-xray"), tmp_path, opener=opener)
        assert waits == []
        assert len(opener.requests) == 1

    def test_offline_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MEDBENCH_OFFLINE=1 never downloads."""
        monkeypatch.setenv("MEDBENCH_OFFLINE", "1")
        with pytest.raises(FileNotFoundError, match="offline"):
            fetch(_sample("d", b"x", "imaging-xray"), tmp_path, opener=_Opener(b"x"))

    def test_prepare_image_makes_small_png(self) -> None:
        """Large JPEGs become PNGs no wider than the limit."""
        png = prepare_image(_image_bytes((3000, 1500)), max_side=1024)
        with Image.open(io.BytesIO(png)) as image:
            assert image.format == "PNG"
            assert image.size == (1024, 512)

    def test_attribution(self) -> None:
        """Credits name author, license and source once per sample."""
        sample = _sample("e", b"x", "imaging-xray")
        lines = attribution_lines([sample, sample])
        assert lines == [
            "e: Test Author, CC0 1.0 "
            "(https://creativecommons.org/publicdomain/zero/1.0/), "
            "via https://example.org/e"
        ]


def _cached(tmp_path: Path, samples: Sequence[tuple[Sample, bytes]]) -> None:
    for sample, data in samples:
        (tmp_path / sample.filename).write_bytes(data)


class TestRealTasks:
    """Imaging and OCR tasks on real-file samples (served from the cache)."""

    def _imaging(self, tmp_path: Path) -> list[Sample]:
        made = [
            ("p1", "pneumothorax", ["pneumothorax"], "right", (50, 40)),
            ("n1", "normal", [], "none", (40, 50)),
            ("p2", "pleural effusion", ["effusion"], "left", (60, 40)),
            ("n2", "normal", [], "none", (40, 60)),
        ]
        pairs = []
        for sample_id, finding, terms, side, size in made:
            data = _image_bytes(size)
            truth = {"finding": finding, "terms": terms, "side": side}
            pairs.append((_sample(sample_id, data, "imaging-xray", **truth), data))
        _cached(tmp_path, pairs)
        return [sample for sample, _ in pairs]

    def test_imaging_items_alternate(self, tmp_path: Path) -> None:
        """Any prefix mixes abnormal and normal images."""
        samples = self._imaging(tmp_path)
        ordered = [samples[0], samples[2], samples[1], samples[3]]
        task = ImagingTask(count=2, samples=ordered, cache_dir=tmp_path)
        assert [item.item_id for item in task.items()] == ["p1", "n1"]

    def test_imaging_with_oracle(self, tmp_path: Path) -> None:
        """A reader that names finding and side scores 1 everywhere."""
        samples = self._imaging(tmp_path)
        task = ImagingTask(count=4, samples=samples, cache_dir=tmp_path)
        items = task.items()
        assert all(item.images[0].startswith(b"\x89PNG") for item in items)
        answers = {
            item.item_id: (
                "abnormal: no; finding: none; side: none"
                if item.truth.truth["finding"] == "normal"
                else f"abnormal: yes; finding: {item.truth.truth['finding']}; "
                f"side: {item.truth.truth['side']}"
            )
            for item in items
        }
        backend = _by_image(items, answers)
        report = run_task(task, backend)
        expected = {
            "sensitivity": 1.0,
            "specificity": 1.0,
            "accuracy": 1.0,
            "finding_recall": 1.0,
            "side_accuracy_when_found": 1.0,
            "unparsed_answers": 0.0,
            "invalid_rate": 0.0,
            "empty_rate": 0.0,
            "abstention_rate": 0.0,
            "truncation_rate": 0.0,
        }
        assert {k: report.metrics[k] for k in expected} == expected
        # Four items cannot prove much: the interval stays wide.
        assert report.metrics["accuracy_ci_low"] == pytest.approx(0.51, abs=0.01)
        assert report.metrics["accuracy_ci_high"] == 1.0

    def test_imaging_wrong_finding(self, tmp_path: Path) -> None:
        """Calling everything abnormal with a wrong name is caught."""
        task = ImagingTask(count=4, samples=self._imaging(tmp_path), cache_dir=tmp_path)
        report = run_task(
            task,
            _constant("abnormal: yes; finding: cardiomegaly; side: both"),
        )
        assert report.metrics["sensitivity"] == 1
        assert report.metrics["specificity"] == 0
        assert report.metrics["finding_recall"] == 0
        assert report.metrics["side_accuracy_when_found"] == 0

    def test_ocr_scores(self, tmp_path: Path) -> None:
        """Perfect text scores zero errors; key terms are counted."""
        data = _image_bytes()
        truth = {
            "text": "Amoxicillin 500 mg\nthree times a day",
            "key_terms": ["amoxicillin", "500 mg"],
        }
        sample = _sample("rx", data, "ocr-prescription", **truth)
        _cached(tmp_path, [(sample, data)])
        task = OcrTask(
            kind="prescription", count=1, samples=[sample], cache_dir=tmp_path
        )
        perfect = run_task(task, _constant("AMOXICILLIN 500 mg, three times a day."))
        expected = {
            "cer": 0.0,
            "wer": 0.0,
            "key_term_recall": 1.0,
            "dose_exact": 1.0,
            "dose_unit_errors": 0.0,
            "dose_value_errors": 0.0,
        }
        assert {k: perfect.metrics[k] for k in expected} == expected
        partial = run_task(task, _constant("Amoxil 500 mg three times a day"))
        assert partial.metrics["key_term_recall"] == 0.5
        assert partial.metrics["cer"] > 0


def _constant(text: str) -> ScriptedBackend:
    return ScriptedBackend(lambda *_: text)


def _by_image(items: Sequence[Any], answers: dict[str, str]) -> ScriptedBackend:
    by_image = {item.images[0]: answers[item.item_id] for item in items}
    return ScriptedBackend(lambda _prompt, images: by_image[images[0]])


class TestCli:
    """``medbench sources`` and ``medbench fetch``."""

    def test_sources(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Every listed sample shows its license."""
        assert main(["sources", "--task", "imaging-xray"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert listed
        assert {row["task"] for row in listed} == {"imaging-xray"}
        assert all(row["license"] in ALLOWED_LICENSES for row in listed)

    def test_fetch_writes_attribution(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Cached files are verified and credited without a download."""
        sample = MANIFEST[0]
        data = b"cached"
        fake = replace(sample, sha256=hashlib.sha256(data).hexdigest())
        (tmp_path / fake.filename).write_bytes(data)
        monkeypatch.setattr("medbench.cli.load_manifest", lambda: [fake])
        monkeypatch.setenv("MEDBENCH_OFFLINE", "1")
        assert main(["fetch", "--out", str(tmp_path)]) == 0
        capsys.readouterr()
        credits = (tmp_path / "ATTRIBUTION.md").read_text(encoding="utf-8")
        assert sample.author in credits
        assert sample.license in credits


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("MEDBENCH_NETWORK_TESTS") != "1",
    reason="set MEDBENCH_NETWORK_TESTS=1 to download the real files",
)
@pytest.mark.parametrize("sample", MANIFEST, ids=lambda s: s.sample_id)
def test_real_file_downloads_and_opens(sample: Sample) -> None:
    """The pinned file is still online, unchanged, and readable."""
    path = fetch(sample)
    if sample.task == "diagnosis-ddxplus":
        assert path.stat().st_size > 1000
        return
    with Image.open(path) as image:
        image.verify()


def _ddxplus_files(tmp_path: Path) -> list[Sample]:
    """Two DDXPlus-format patients with the real file layout."""
    evidences = {
        "E_91": {
            "question_en": "Do you have a fever?",
            "default_value": 0,
            "value_meaning": {},
            "possible-values": [],
        },
        "E_56": {
            "question_en": "How intense is the pain?",
            "default_value": 0,
            "value_meaning": {},
            "possible-values": list(range(11)),
        },
        "E_55": {
            "question_en": "Do you feel pain somewhere?",
            "default_value": "V_123",
            "value_meaning": {
                "V_123": {"en": "nowhere"},
                "V_29": {"en": "upper chest"},
            },
            "possible-values": ["V_123", "V_29"],
        },
    }
    conditions = {
        "GERD": {"cond-name-eng": "GERD"},
        "Unstable angina": {"cond-name-eng": "Unstable angina"},
        "Stable angina": {"cond-name-eng": "Stable angina"},
    }
    rows = [
        "AGE,DIFFERENTIAL_DIAGNOSIS,SEX,PATHOLOGY,EVIDENCES,INITIAL_EVIDENCE",
        "49,\"[['GERD', 0.6], ['Stable angina', 0.4]]\",F,GERD,"
        "\"['E_55_@_V_29', 'E_55_@_V_123', 'E_56_@_6']\",E_55",
        "61,\"[['Stable angina', 0.5], ['Unstable angina', 0.5]]\",M,"
        "Unstable angina,\"['E_91', 'E_56_@_0']\",E_91",
    ]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("release_test_patients", "\n".join(rows) + "\n")
    files = {
        "patients": archive.getvalue(),
        "evidences": json.dumps(evidences).encode(),
        "conditions": json.dumps(conditions).encode(),
    }
    samples = []
    for role, data in files.items():
        sample = _sample(f"dx-{role}", data, "diagnosis-ddxplus", file=role)
        (tmp_path / sample.filename).write_bytes(data)
        samples.append(sample)
    return samples


class TestDdxplus:
    """DDXPlus decoding, prompts and scoring on a two-patient copy."""

    def test_patients_are_decoded(self, tmp_path: Path) -> None:
        """Codes become English answers; default values are dropped."""
        data = DdxplusData.load(tmp_path, _ddxplus_files(tmp_path))
        first, second = data.patients(5)
        assert (first.age, first.sex, first.pathology) == (49, "female", "GERD")
        assert first.findings == (
            "Do you feel pain somewhere? upper chest",
            "How intense is the pain? 6 on a 0-10 scale",
        )
        assert first.differential == ("GERD", "Stable angina")
        assert second.chief_complaint == "Do you have a fever?"
        assert data.patients(1) == [first]
        prompt = build_ddxplus_prompt(first, data.conditions, top_k=3)
        assert "49-year-old female" in prompt
        assert "GERD; Stable angina; Unstable angina." in prompt
        assert "3 most likely" in prompt

    def test_split_keeps_patients_apart(self, tmp_path: Path) -> None:
        """A split takes only its own patients and keeps their file ids."""
        data = DdxplusData.load(tmp_path, _ddxplus_files(tmp_path))
        by_split = {
            split: [p.patient_id for p in data.patients(5, split)]
            for split in ("train", "dev", "test")
        }
        assert by_split == {
            "train": ["ddxplus-test-0"],
            "dev": ["ddxplus-test-1"],
            "test": [],
        }

    def test_scores_with_aliases(self, tmp_path: Path) -> None:
        """Aliases count, and the longest name wins."""
        task = DdxplusTask(
            count=2, cache_dir=tmp_path, samples=_ddxplus_files(tmp_path)
        )
        answers = {
            "ddxplus-test-0": "1. Gastroesophageal reflux disease\n2. Stable angina",
            "ddxplus-test-1": "1. **Unstable angina pectoris** - rest pain",
        }
        backend = ScriptedBackend(
            lambda prompt, _images: answers[
                "ddxplus-test-0" if "49-year-old" in prompt else "ddxplus-test-1"
            ]
        )
        report = run_task(task, backend)
        assert report.metrics["top1"] == 1.0
        assert report.metrics["reference_top1"] == 0.5
        assert report.metrics["reference_top5"] == 1.0
        assert report.metrics["unparsed_answers"] == 0


def test_ddxplus_wrong_candidate_is_not_dropped(tmp_path: Path) -> None:
    """A condition absent from this patient's reference must keep its rank."""
    task = DdxplusTask(count=1, cache_dir=tmp_path, samples=_ddxplus_files(tmp_path))
    report = run_task(task, _constant("1. Unstable angina\n2. Stable angina\n3. GERD"))
    assert report.metrics["top1"] == 0
    assert report.metrics["mrr"] == pytest.approx(1 / 3)
    assert len(report.sources) == 3


@pytest.mark.parametrize("output", ["", "unknown", "I am not sure."])
def test_unparsed_imaging_does_not_earn_normal_credit(
    tmp_path: Path, output: str
) -> None:
    samples = TestRealTasks()._imaging(tmp_path)
    task = ImagingTask(count=4, samples=samples, cache_dir=tmp_path)
    report = run_task(task, _constant(output))
    assert report.metrics["accuracy"] == 0
    assert report.metrics["specificity"] == 0
    assert report.metrics["unparsed_answers"] == 4
    assert len(report.sources) == 4


def test_negated_finding_is_not_named(tmp_path: Path) -> None:
    """ "No pneumothorax" on a pneumothorax image does not name the finding."""
    samples = TestRealTasks()._imaging(tmp_path)
    task = ImagingTask(count=4, samples=samples, cache_dir=tmp_path)
    report = run_task(
        task, _constant("abnormal: yes; finding: no pneumothorax, no effusion")
    )
    assert report.metrics["sensitivity"] == 1
    assert report.metrics["finding_recall"] == 0


def test_xray_prompt_does_not_claim_a_frontal_view(tmp_path: Path) -> None:
    samples = TestRealTasks()._imaging(tmp_path)
    prompt = ImagingTask(count=1, samples=samples, cache_dir=tmp_path).items()[0].prompt
    assert "real chest X-ray" in prompt
    assert "frontal" not in prompt


def test_abstention_is_reported(tmp_path: Path) -> None:
    samples = TestRealTasks()._imaging(tmp_path)
    task = ImagingTask(count=4, samples=samples, cache_dir=tmp_path)
    report = run_task(task, _constant("I am not sure."))
    assert report.metrics["abstention_rate"] == 1
    assert report.metrics["invalid_rate"] == 1
    assert report.metrics["accuracy"] == 0


def test_dose_unit_error_is_counted(tmp_path: Path) -> None:
    image = _image_bytes()
    sample = _sample(
        "rx", image, "ocr-prescription", text="Digoxin 125 mcg", key_terms=["digoxin"]
    )
    _cached(tmp_path, [(sample, image)])
    task = OcrTask(count=1, samples=[sample], cache_dir=tmp_path)
    report = run_task(task, _constant("Digoxin 125 mg"))
    assert report.metrics["key_term_recall"] == 1
    assert report.metrics["dose_exact"] == 0
    assert report.metrics["dose_unit_errors"] == 1
    assert report.metrics["dose_value_errors"] == 0


def test_wrong_dose_cannot_match_substring(tmp_path: Path) -> None:
    image = _image_bytes()
    sample = _sample(
        "rx", image, "ocr-prescription", text="Amoxicillin 500 mg", key_terms=["500 mg"]
    )
    _cached(tmp_path, [(sample, image)])
    task = OcrTask(count=1, samples=[sample], cache_dir=tmp_path)
    report = run_task(task, _constant("Amoxicillin 1500 mg"))
    assert report.metrics["key_term_recall"] == 0


def test_manifest_rejects_restrictive_license(tmp_path: Path) -> None:
    manifest = tmp_path / "samples.json"
    sample = _sample("restricted", b"x", "imaging-xray")
    manifest.write_text(
        json.dumps({"samples": [{**sample.__dict__, "license": "CC BY-NC 4.0"}]})
    )
    with pytest.raises(ValueError, match="Unsupported license"):
        load_manifest(manifest)
