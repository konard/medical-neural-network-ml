"""Tests for the model x runtime x format x task support matrix (issue #9)."""

from __future__ import annotations

import json

import pytest

from medbench.benchmark import TASKS
from medbench.cli import main
from medbench.models import MODELS
from medbench.support import (
    MATRIX,
    RUNTIMES,
    STATUSES,
    SUPPORTED_MEASURED,
    SUPPORTED_UNMEASURED,
    TASK_NAMES,
    UNSUPPORTED,
    default_runtime,
    support_matrix,
    support_status,
)


def test_task_names_match_the_benchmark() -> None:
    assert TASK_NAMES == TASKS


def test_matrix_entries_reference_known_models_and_runtimes() -> None:
    for entry in MATRIX:
        assert entry.model in MODELS
        assert entry.runtime in RUNTIMES
        assert entry.measured <= set(TASK_NAMES)
        assert entry.how
        if entry.measured:
            assert entry.evidence


def test_measured_only_where_runs_were_published() -> None:
    status = support_status("medgemma-4b-it", "llama.cpp", "Q4_K_M", "imaging-xray")
    assert status.status == SUPPORTED_MEASURED
    assert "38714694090" in status.reason
    other = support_status("medgemma-4b-it", "llama.cpp", "Q4_K_M", "imaging-mri")
    assert other.status == SUPPORTED_UNMEASURED


@pytest.mark.parametrize("task", TASK_NAMES)
def test_large_models_are_not_marked_measured(task: str) -> None:
    """No GPU run has been published for 27B or UniMedVL (issue #6)."""
    assert (
        support_status("medgemma-27b-it", "llama.cpp", "Q4_K_M", task).status
        == SUPPORTED_UNMEASURED
    )
    assert (
        support_status("unimedvl", "unimedvl-upstream", "bf16", task).status
        == SUPPORTED_UNMEASURED
    )


@pytest.mark.parametrize("quant", ["Q4_K_M", "Q8_0", "Q2_K"])
def test_unimedvl_gguf_is_unsupported(quant: str) -> None:
    status = support_status("unimedvl", "llama.cpp", quant, "diagnosis")
    assert status.status == UNSUPPORTED
    assert "no GGUF" in status.reason


def test_text_only_model_cannot_take_image_tasks() -> None:
    status = support_status(
        "medgemma-27b-text-it", "llama.cpp", "Q4_K_M", "ocr-history"
    )
    assert status.status == UNSUPPORTED
    assert "text only" in status.reason


def test_unpinned_combination_is_unsupported_with_hint() -> None:
    status = support_status("medgemma-4b-it", "llama.cpp", "Q8_0", "diagnosis")
    assert status.status == UNSUPPORTED
    assert "llama.cpp/Q4_K_M" in status.reason


def test_unknown_keys_raise() -> None:
    with pytest.raises(KeyError, match="runtime"):
        support_status("unimedvl", "vllm", "bf16", "diagnosis")
    with pytest.raises(KeyError, match="task"):
        support_status("unimedvl", "unimedvl-upstream", "bf16", "nope")
    with pytest.raises(KeyError, match="model"):
        support_status("nope", "llama.cpp", "bf16", "diagnosis")


def test_default_runtime() -> None:
    assert default_runtime("unimedvl", "bf16") == "unimedvl-upstream"
    assert default_runtime("unimedvl", "Q4_K_M") == "llama.cpp"
    assert default_runtime("medgemma-4b-it", "mlx-4bit") == "mlx-vlm"
    assert default_runtime("qwen2.5-vl-7b", "bf16") == "transformers"


def test_matrix_covers_every_model_with_valid_statuses() -> None:
    rows = support_matrix()
    assert {row.model for row in rows} == set(MODELS)
    assert {row.status for row in rows} <= set(STATUSES)
    assert {row.status for row in rows} == set(STATUSES)


def test_cli_support_and_estimate_show_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["support", "--model", "unimedvl", "--runtime", "llama.cpp"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows
    assert {row["status"] for row in rows} == {UNSUPPORTED}

    assert (
        main(
            [
                "estimate",
                "--model",
                "medgemma-27b-it",
                "--hardware",
                "h100-80gb",
                "--quant",
                "Q4_K_M",
                "--train-tokens",
                "1e6",
                "--price-per-hour",
                "2",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["runtime"] == "llama.cpp"
    assert {row["status"] for row in result["support"]} == {SUPPORTED_UNMEASURED}
    row = result["inference"][0]
    assert row["capacity"] == "fits"
    assert len(row["total_memory_gb_range"]) == 2
    assert {row["price_source"] for row in result["training"]} == {"user-supplied"}
