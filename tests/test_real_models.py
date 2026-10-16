"""Smoke tests against a real model server (issue #1).

They are skipped unless ``MEDBENCH_BASE_URL`` points at an OpenAI-compatible
server, such as the llama-server started by ``scripts/medbench-model-run.sh``.
The checks are loose on purpose: they catch a broken model, template or
image pipeline, not a slightly weaker model.
"""

from __future__ import annotations

import os
import re

import pytest

from medbench.backends import OpenAICompatibleBackend
from medbench.real_data import fetch, prepare_image, samples_for
from medbench.synthetic_cases import (
    PatientCase,
    build_diagnosis_prompt,
    parse_ranked_diagnoses,
)

BASE_URL = os.environ.get("MEDBENCH_BASE_URL", "")
VISION = os.environ.get("MEDBENCH_VISION", "0") == "1"
TIMEOUT = float(os.environ.get("MEDBENCH_REQUEST_TIMEOUT", "600"))

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not BASE_URL, reason="set MEDBENCH_BASE_URL to run"),
]


@pytest.fixture(scope="module")
def backend() -> OpenAICompatibleBackend:
    """Client for the server under test."""
    model = os.environ.get("MEDBENCH_MODEL", "default")
    return OpenAICompatibleBackend(BASE_URL, model=model, timeout_s=TIMEOUT)


def test_textbook_case_gets_a_ranked_differential(
    backend: OpenAICompatibleBackend,
) -> None:
    """A classic urinary tract infection is named in the top five."""
    case = PatientCase(
        case_id="smoke-uti",
        age=26,
        sex="female",
        symptoms=("painful urination", "frequent urination", "lower abdominal pain"),
        diagnosis="urinary tract infection",
        reference_ranking=("urinary tract infection",),
        reference_probabilities=(1.0,),
    )
    answer = backend.generate(build_diagnosis_prompt(case, top_k=5), max_tokens=120)
    ranked = parse_ranked_diagnoses(answer.text)
    assert ranked, f"no ranked list in: {answer.text!r}"
    assert "urinary tract infection" in ranked[:5], answer.text


@pytest.mark.skipif(not VISION, reason="set MEDBENCH_VISION=1 for image models")
def test_model_answers_questions_about_an_image(
    backend: OpenAICompatibleBackend,
) -> None:
    """A real chest X-ray reaches the model and it answers yes or no."""
    sample = samples_for("imaging-xray")[0]
    image = prepare_image(fetch(sample).read_bytes())
    question = (
        "This is a real chest X-ray. Is there an abnormal finding? "
        "Answer on one line: abnormal: yes or no; finding: <name>; side: <side>."
    )
    answer = backend.generate(question, [image], max_tokens=48)
    assert re.search(r"\b(yes|no)\b", answer.text.lower()), answer.text
    assert answer.prompt_tokens > 64, "the image was not turned into tokens"
