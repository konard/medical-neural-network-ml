"""Tests for backends, the benchmark runner and the CLI (issue #1).

The OpenAI-compatible backend is tested against a tiny local HTTP server,
so no model is needed.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from collections.abc import Iterator
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from medbench.backends import (
    BackendError,
    Generation,
    NaiveBayesBackend,
    OpenAICompatibleBackend,
    TesseractBackend,
    image_mime,
)
from medbench.benchmark import (
    DiagnosisTask,
    ItemResult,
    make_task,
    parse_yes_no_side,
    run_task,
)
from medbench.cli import _make_backend, build_parser, main
from medbench.hardware import ResourceBudget


class _FakeServer(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []  # noqa: RUF012 - shared on purpose

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body["auth"] = self.headers.get("Authorization")
        self.requests.append(body)
        reply = {
            "choices": [{"message": {"content": "1. Influenza\n2. COVID-19"}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        del args


@pytest.fixture
def server_url() -> Iterator[str]:
    """Start the fake chat completions server on a free port."""
    _FakeServer.requests.clear()
    server = HTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class TestBackends:
    """Backend implementations."""

    def test_openai_backend(self, server_url: str) -> None:
        """Prompts and images are sent in the OpenAI format."""
        backend = OpenAICompatibleBackend(server_url, model="medgemma", api_key="k")
        generation = backend.generate("hello", [b"\x89PNG"], max_tokens=9)
        assert generation.text.startswith("1. Influenza")
        assert generation.prompt_tokens == 42
        assert generation.completion_tokens == 7
        sent = _FakeServer.requests[0]
        assert sent["model"] == "medgemma"
        assert sent["max_tokens"] == 9
        assert sent["auth"] == "Bearer k"
        content = sent["messages"][0]["content"]
        assert content[0]["image_url"]["url"] == "data:image/png;base64,iVBORw=="
        assert content[1] == {"type": "text", "text": "hello"}

    def test_openai_backend_closes_error_answers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error answer is read, classified and closed (issue #19)."""
        body = io.BytesIO(b"CUDA out of memory")
        error = urllib.error.HTTPError("http://x/v1", 500, "error", Message(), body)

        def urlopen(*_args: object, **_kwargs: object) -> None:
            raise error

        monkeypatch.setattr("medbench.backends.urllib.request.urlopen", urlopen)
        backend = OpenAICompatibleBackend("http://127.0.0.1:1/v1")
        with pytest.raises(BackendError) as caught:
            backend.generate("hello")
        assert caught.value.kind == "out_of_memory"
        assert body.closed

    @pytest.mark.parametrize(
        ("data", "mime"),
        [
            (b"\xff\xd8\xff\xe0rest", "image/jpeg"),
            (b"\x89PNG\r\n\x1a\n", "image/png"),
            (b"GIF89a", "image/gif"),
            (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        ],
    )
    def test_image_mime(self, data: bytes, mime: str) -> None:
        """Real files are sent with their own type, not always PNG."""
        assert image_mime(data) == mime

    def test_openai_backend_rejects_other_schemes(self) -> None:
        """Only HTTP(S) URLs are allowed."""
        with pytest.raises(ValueError, match="http"):
            OpenAICompatibleBackend("file:///etc/passwd")

    def test_naive_bayes_backend(self) -> None:
        """The baseline answers diagnosis prompts only."""
        backend = NaiveBayesBackend()
        prompt = "A 30-year-old male patient reports: painful urination.\nList"
        assert backend.generate(prompt).text.startswith("1. urinary tract infection")
        assert backend.generate("What is on this image?").text == "unknown"

    def test_tesseract_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A clear error explains how to install Tesseract."""
        monkeypatch.setenv("PATH", "")
        with pytest.raises(FileNotFoundError, match="tesseract-ocr"):
            TesseractBackend()

    def test_tesseract_baseline_from_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--backend tesseract`` runs the binary on every image."""
        fake = tmp_path / "tesseract"
        fake.write_text("#!/bin/sh\necho 'Napa 500mg'\n", encoding="utf-8")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        backend = _make_backend(
            build_parser().parse_args(
                ["run", "--task", "ocr-prescription", "--backend", "tesseract"]
            ),
            ResourceBudget(),
        )
        assert isinstance(backend, TesseractBackend)
        out = backend.generate("ignored", [b"one", b"two"])
        assert out.text == "Napa 500mg\nNapa 500mg"


class TestParsing:
    """Yes/no answers about findings."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Yes, in the patient's left lung.", (True, "left")),
            ("Yes - right hemisphere", (True, "right")),
            ("No. The lungs are clear.", (False, "none")),
            ("There is no nodule; nothing on the left.", (False, "none")),
            ("A cyst is present on the right half.", (True, "right")),
            ("I am not sure.", (False, "none")),
            ("", (False, "none")),
            ("abnormal: yes; finding: effusion; side: left", (True, "left")),
            (
                "Abnormal: Yes; finding: right pneumothorax; side: right",
                (True, "right"),
            ),
            ("abnormal: yes; finding: edema; side: bilateral", (True, "both")),
            ("abnormal: no; finding: none; side: none", (False, "none")),
        ],
    )
    def test_parse_yes_no_side(self, text: str, expected: tuple[bool, str]) -> None:
        """The answer fields, or else the first answer word, decide."""
        assert parse_yes_no_side(text) == expected


class TestRunner:
    """Running tasks end to end."""

    def test_diagnosis_with_baseline(self) -> None:
        """The baseline matches the reference exactly."""
        report = run_task(DiagnosisTask(count=30, seed=1), NaiveBayesBackend())
        assert report.items_run == 30
        assert report.metrics["top1"] == report.metrics["reference_top1"]
        assert report.metrics["top5"] >= report.metrics["top1"]
        assert report.metrics["unparsed_answers"] == 0

    def test_time_budget_stops_early(self) -> None:
        """No items run when the budget is already used up."""
        report = run_task(
            DiagnosisTask(count=3), NaiveBayesBackend(), ResourceBudget(max_seconds=0)
        )
        assert report.stopped_early
        assert report.items_run == 0
        assert report.metrics == {}
        assert report.latency["total_s"] == 0
        assert report.as_dict()["items_requested"] == 3
        assert report.coverage == 0

    def test_truncation_and_coverage(self) -> None:
        """Answers that hit max_tokens are flagged and counted."""

        class Cut:
            name = "cut"

            def generate(
                self, prompt: str, images: object = (), max_tokens: int = 256
            ) -> Generation:
                del prompt, images
                return Generation("1. Influenza\n2.", 0.01, 50, max_tokens)

        report = run_task(DiagnosisTask(count=4), Cut(), max_tokens=8)
        data = report.as_dict()
        assert all(result["truncated"] for result in data["results"])
        assert report.metrics["truncation_rate"] == 1
        assert report.metrics["empty_rate"] == 0
        assert (data["items_requested"], data["coverage"]) == (4, 1.0)

    def test_wilson_interval_on_top1(self) -> None:
        """A perfect score on two items still has a wide interval."""
        task = DiagnosisTask(count=2, seed=1)
        items = task.items()
        results = [
            ItemResult(item.item_id, f"1. {item.truth.diagnosis}", 0.0)
            for item in items
        ]
        scores = task.score(items, results)
        assert scores["top1"] == 1
        assert scores["top1_ci_low"] == pytest.approx(0.342, abs=1e-3)
        assert scores["top1_ci_high"] == 1
        assert scores["invalid_rate"] == 0

    def test_report_json(self, tmp_path: Path) -> None:
        """Reports are written as JSON with latencies."""
        report = run_task(DiagnosisTask(count=3), NaiveBayesBackend())
        data = json.loads(report.write(tmp_path / "r.json").read_text(encoding="utf-8"))
        assert data["backend"] == "naive-bayes"
        assert len(data["results"]) == 3
        assert set(data["latency"]) == {"mean_s", "p50_s", "p95_s", "total_s"}

    def test_make_task(self, tmp_path: Path) -> None:
        """Every listed task can be created without a download."""
        assert make_task("ocr-history", 1).name == "ocr-history"
        task = make_task("imaging-mri", 1, cache_dir=tmp_path)
        assert task.name == "imaging-mri"
        assert not list(tmp_path.iterdir())
        with pytest.raises(KeyError, match="unknown task"):
            make_task("ocr-xray", 1)


class TestCli:
    """The medbench command."""

    def test_estimate(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The estimate command prints inference and fine-tuning numbers."""
        assert main(["estimate", "--model", "medgemma-4b-it", "--finetune"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert len(data["inference"]) >= 5
        assert [row["method"] for row in data["finetune"]] == ["full", "lora", "qlora"]

    def test_host(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The host command reports CPUs."""
        assert main(["host"]) == 0
        assert json.loads(capsys.readouterr().out)["cpu_count"] >= 1

    def test_generate(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Only text diagnosis cases are generated; no fake media."""
        assert main(["generate", "--out", str(tmp_path), "--count", "2"]) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["cases"] == 2
        assert [p.name for p in tmp_path.iterdir()] == ["diagnosis_cases.json"]

    def test_run_with_server(
        self, server_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The run command talks to a model server and saves the report."""
        out = tmp_path / "report.json"
        args = ["run", "--task", "diagnosis", "--backend", "openai"]
        args += ["--base-url", server_url, "--count", "2", "--out", str(out)]
        assert main(args) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["items_run"] == 2
        assert "results" not in summary
        assert len(json.loads(out.read_text(encoding="utf-8"))["results"]) == 2


@pytest.mark.parametrize("count", ["0", "-1"])
def test_cli_rejects_empty_benchmark(count: str) -> None:
    assert main(["run", "--task", "diagnosis", "--count", count]) == 1


def test_cli_incomplete_benchmark_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--task", "diagnosis", "--time-budget", "0"]) == 1
    assert json.loads(capsys.readouterr().out)["stopped_early"]
