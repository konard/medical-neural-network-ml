"""Tests for scripts/unimedvl_server.py and the client's failure handling (#6).

The real UniMedVL engine needs a CUDA GPU and a 29 GB checkpoint, so these
tests drive the HTTP layer with stand-in engines. They check the parts that
decide whether a GPU run survives a bad item: out-of-memory answers, model
errors, timeouts, cleanup, and that medbench keeps scoring the rest.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from medbench.backends import BackendError, OpenAICompatibleBackend
from medbench.benchmark import DiagnosisTask, run_task

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "unimedvl_server.py"
spec = importlib.util.spec_from_file_location("unimedvl_server", SCRIPT_PATH)
server_module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules[spec.name] = server_module  # type: ignore[union-attr]
spec.loader.exec_module(server_module)  # type: ignore[union-attr]

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class OutOfMemoryError(RuntimeError):
    """Same class name as ``torch.cuda.OutOfMemoryError``."""


class ScriptedEngine:
    """Engine that raises or answers from a list, one entry per request."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.cleanups = 0
        self.requests: list[Any] = []

    def generate(self, request: Any) -> Any:
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else "1. Influenza"
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, threading.Event):
            outcome.wait(10)
            return server_module.Answer("late", 1, 1)
        return server_module.Answer(outcome, 3, len(outcome.split()))

    def cleanup(self) -> None:
        self.cleanups += 1

    def stats(self) -> dict[str, Any]:
        return {"engine": "scripted", "cleanups": self.cleanups}


@contextmanager
def serve(engine: Any) -> Iterator[str]:
    """Run the server on a free port; yield its ``/v1`` URL."""
    server = server_module.ChatServer(("127.0.0.1", 0), engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


class TestParseChatRequest:
    """OpenAI request bodies to UniMedVL's single prompt plus images."""

    def test_text_and_image_parts(self) -> None:
        image = "data:image/png;base64," + base64.b64encode(PNG).decode()
        request = server_module.parse_chat_request(
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image}},
                            {"type": "text", "text": "Is this normal?"},
                        ],
                    },
                ],
                "max_tokens": 64,
                "temperature": 0,
            }
        )
        assert request.prompt == "Be brief.\n\nIs this normal?"
        assert request.images == (PNG,)
        assert (request.max_tokens, request.temperature) == (64, 0.0)

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ({}, "messages"),
            ({"messages": [{"content": 5}]}, "content"),
            (
                {
                    "messages": [
                        {
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.org/x.png"},
                                }
                            ]
                        }
                    ]
                },
                "data: image URLs",
            ),
            (
                {
                    "messages": [
                        {
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png,raw"},
                                }
                            ]
                        }
                    ]
                },
                "base64",
            ),
            ({"messages": [{"content": "x"}], "max_tokens": 100000}, "max_tokens"),
            ({"messages": [{"content": "x"}], "temperature": -1}, "temperature"),
            ({"messages": [{"content": [{"type": "audio"}]}]}, "unsupported"),
        ],
    )
    def test_rejects_bad_requests(self, body: Any, message: str) -> None:
        with pytest.raises(server_module.RequestError, match=message):
            server_module.parse_chat_request(body)

    def test_image_limit(self) -> None:
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        }
        body = {"messages": [{"content": [image] * (server_module.MAX_IMAGES + 1)}]}
        with pytest.raises(server_module.RequestError, match="at most"):
            server_module.parse_chat_request(body)


def test_out_of_memory_detection() -> None:
    """Torch's error is matched by class name, so the script needs no torch."""
    assert server_module.is_out_of_memory(OutOfMemoryError("CUDA"))
    assert server_module.is_out_of_memory(RuntimeError("CUDA out of memory."))
    assert server_module.is_out_of_memory(MemoryError())
    assert not server_module.is_out_of_memory(ValueError("bad shape"))


def test_recording_tokenizer_counts_decoded_steps() -> None:
    """``model.chat`` returns text only; the decoded ids give the step count."""

    class Tokenizer:
        eos_token_id = 7

        def decode(self, ids: list[int]) -> str:
            return " ".join(map(str, ids))

    recorder = server_module._RecordingTokenizer(Tokenizer())
    assert recorder.decode([1, 2, 3]) == "1 2 3"
    assert recorder.last_length == 3
    assert recorder.eos_token_id == 7


class TestServer:
    """The HTTP layer, driven through medbench's own client."""

    def test_answer_health_models_and_metrics(self) -> None:
        engine = ScriptedEngine(["1. Influenza\n2. URTI"])
        with serve(engine) as url:
            backend = OpenAICompatibleBackend(url, model="unimedvl")
            generation = backend.generate("Which diagnosis?", [PNG], max_tokens=32)
            root = url.removesuffix("/v1")
            assert get_json(f"{root}/health") == {"status": "ok"}
            assert get_json(f"{url}/models")["data"] == [{"id": "unimedvl"}]
            assert get_json(f"{root}/metrics")["engine"] == "scripted"
        assert generation.text.startswith("1. Influenza")
        assert (generation.prompt_tokens, generation.completion_tokens) == (3, 4)
        assert engine.requests[0].images == (PNG,)
        assert engine.requests[0].max_tokens == 32

    def test_out_of_memory_is_507_and_the_server_recovers(self) -> None:
        engine = ScriptedEngine([OutOfMemoryError("CUDA out of memory"), "1. Croup"])
        with serve(engine) as url:
            backend = OpenAICompatibleBackend(url)
            with pytest.raises(BackendError) as caught:
                backend.generate("first")
            assert caught.value.kind == "out_of_memory"
            assert caught.value.status == server_module.HTTP_OUT_OF_MEMORY
            assert engine.cleanups == 1
            assert backend.generate("second").text == "1. Croup"

    def test_model_error_is_500_after_cleanup(self) -> None:
        engine = ScriptedEngine([ValueError("shape mismatch")])
        with serve(engine) as url, pytest.raises(BackendError) as caught:
            OpenAICompatibleBackend(url).generate("x")
        assert (caught.value.kind, caught.value.status) == ("http", 500)
        assert "shape mismatch" in str(caught.value)
        assert engine.cleanups == 1

    def test_bad_request_is_400_without_running_the_model(self) -> None:
        engine = ScriptedEngine([])
        with serve(engine) as url:
            request = urllib.request.Request(
                f"{url}/chat/completions",
                data=b'{"messages": []}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=10)
            caught.value.close()
            assert caught.value.code == 400
        assert engine.requests == []

    def test_slow_answer_is_a_timeout(self) -> None:
        release = threading.Event()
        engine = ScriptedEngine([release])
        with serve(engine) as url:
            try:
                with pytest.raises(BackendError) as caught:
                    OpenAICompatibleBackend(url, timeout_s=0.3).generate("x")
            finally:
                release.set()
        assert caught.value.kind == "timeout"

    def test_echo_engine(self) -> None:
        with serve(server_module.EchoEngine()) as url:
            generation = OpenAICompatibleBackend(url).generate("Hello\nworld", [], 2)
        assert generation.text == "echo: Hello"
        assert generation.completion_tokens == 2


def test_unreachable_server_is_a_connection_error() -> None:
    """A server that died (e.g. killed by the kernel's OOM killer)."""
    server = server_module.ChatServer(("127.0.0.1", 0), ScriptedEngine([]))
    port = server.server_address[1]
    server.server_close()
    with pytest.raises(BackendError) as caught:
        OpenAICompatibleBackend(f"http://127.0.0.1:{port}/v1").generate("x")
    assert caught.value.kind == "connection"


def test_run_continues_after_failed_items() -> None:
    """One out-of-memory item is recorded and scored wrong; the rest still run."""
    engine = ScriptedEngine(
        ["1. Influenza", OutOfMemoryError("CUDA out of memory"), "1. Influenza"]
    )
    with serve(engine) as url:
        report = run_task(DiagnosisTask(count=3), OpenAICompatibleBackend(url))
    data = report.as_dict()
    errors = [result["error"] for result in data["results"]]
    assert errors[0] is None
    assert errors[1].startswith("out_of_memory: HTTP 507")
    assert errors[2] is None
    assert data["items_run"] == 3
    assert not data["stopped_early"]
    assert report.metrics["error_rate"] == pytest.approx(1 / 3)
    assert data["results"][1]["output"] == ""
