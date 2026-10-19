"""Which model, runtime, weight format and task combinations actually work.

The resource calculator in :mod:`medbench.resources` only answers "would the
bytes fit?". This module answers a different question: "does this repository
have a pinned, reproducible way to run the combination, and has anybody run
it?" Every combination gets one of three statuses:

* ``supported/measured`` - a pinned runtime path exists and a run with
  published raw reports is linked as evidence;
* ``supported/unmeasured`` - a pinned runtime path exists (script, workflow
  input or backend), but no run has been published yet, so memory and speed
  are only estimates;
* ``unsupported`` - there is no pinned runtime path, the model cannot take
  the task's input, or the weight format does not exist for the model.

"Measured" means the pipeline ran on real pinned files; with two items per
task it is a transport check, not evidence of clinical accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from medbench.models import MODELS

__all__ = [
    "IMAGE_TASKS",
    "MATRIX",
    "RUNTIMES",
    "STATUSES",
    "SUPPORTED_MEASURED",
    "SUPPORTED_UNMEASURED",
    "TASK_NAMES",
    "UNSUPPORTED",
    "SupportEntry",
    "SupportStatus",
    "default_runtime",
    "support_matrix",
    "support_status",
]

SUPPORTED_MEASURED = "supported/measured"
SUPPORTED_UNMEASURED = "supported/unmeasured"
UNSUPPORTED = "unsupported"
STATUSES = (SUPPORTED_MEASURED, SUPPORTED_UNMEASURED, UNSUPPORTED)

# Kept equal to medbench.benchmark.TASKS by a test; duplicated here so that
# the calculator does not import the benchmark runner.
TASK_NAMES = (
    "diagnosis",
    "diagnosis-ddxplus",
    "ocr-prescription",
    "ocr-history",
    "imaging-xray",
    "imaging-mri",
    "imaging-ultrasound",
)
IMAGE_TASKS = frozenset(task for task in TASK_NAMES if not task.startswith("diag"))

RUNTIMES: dict[str, str] = {
    "llama.cpp": (
        "llama-server b11738 (GGUF, CPU or CUDA) started by "
        "scripts/medbench-model-run.sh and called through "
        "OpenAICompatibleBackend"
    ),
    "mlx-vlm": "mlx-vlm 0.7.9 on Apple silicon through MlxBackend",
    "unimedvl-upstream": (
        "UniMedVL's own PyTorch code at a pinned commit in its own conda "
        "environment, served by scripts/unimedvl_server.py"
    ),
    "transformers": (
        "generic Hugging Face transformers through TransformersBackend; "
        "not a model-specific inference stack"
    ),
}

_CPU_RUNS = (
    "GitHub-hosted CPU runs 38550186566, 38552481450 and 38714694090 with 2 "
    "items per task (docs/case-studies/issue-1/measurements.md sections 9-10); "
    "a pipeline check, not an accuracy result"
)


@dataclass(frozen=True)
class SupportEntry:
    """One pinned runtime path for a model in one weight format.

    Attributes:
        model: Key of :data:`medbench.models.MODELS`.
        runtime: Key of :data:`RUNTIMES`.
        quant: Weight format of the pinned artifact.
        how: The pinned files, script or workflow input that runs it.
        measured: Tasks with published raw reports from a real run.
        evidence: Where the measured reports are.
    """

    model: str
    runtime: str
    quant: str
    how: str
    measured: frozenset[str] = frozenset()
    evidence: str = ""


MATRIX: tuple[SupportEntry, ...] = (
    SupportEntry(
        model="medgemma-4b-it",
        runtime="llama.cpp",
        quant="Q4_K_M",
        how=(
            "unsloth/medgemma-4b-it-GGUF Q4_K_M + mmproj-F16, SHA-256 pinned in "
            "scripts/medbench-model-run.sh; Model Tests cpu-smoke and gpu jobs"
        ),
        measured=frozenset(
            {"diagnosis", "diagnosis-ddxplus", "ocr-prescription", "imaging-xray"}
        ),
        evidence=_CPU_RUNS,
    ),
    SupportEntry(
        model="medgemma-27b-it",
        runtime="llama.cpp",
        quant="Q4_K_M",
        how=(
            "unsloth/medgemma-27b-it-GGUF revision c56c561 Q4_K_M + mmproj-F16, "
            "SHA-256 pinned in the Model Tests gpu job (gpu-model=medgemma-27b-it)"
        ),
    ),
    SupportEntry(
        model="medgemma-4b-it",
        runtime="mlx-vlm",
        quant="mlx-4bit",
        how=(
            "mlx-community/medgemma-4b-it-4bit through MlxBackend after the "
            "free-memory preflight (docs/apple-silicon.md); the audit Mac only "
            "had room for the SmolVLM wiring model"
        ),
    ),
    SupportEntry(
        model="unimedvl",
        runtime="unimedvl-upstream",
        quant="bf16",
        how=(
            "uni-medical/UniMedVL commit f51aa0a + General-Medical-AI/UniMedVL "
            "revision 29918ce (ema_bf16.safetensors), served by "
            "scripts/unimedvl_server.py; Model Tests gpu job (gpu-model=unimedvl)"
        ),
    ),
)


@dataclass(frozen=True)
class SupportStatus:
    """Status of one model, runtime, weight format and task combination."""

    model: str
    runtime: str
    quant: str
    task: str
    status: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        """Return the status as a JSON-friendly dictionary."""
        return dict(self.__dict__)


def default_runtime(model: str, quant: str) -> str:
    """Return the runtime a weight format implies for ``model``.

    GGUF formats (``Q*``) imply llama.cpp and ``mlx-*`` formats imply mlx-vlm.
    Plain tensors use the model's own stack when this repository has one
    (UniMedVL), otherwise generic transformers.
    """
    if quant.startswith("Q"):
        return "llama.cpp"
    if quant.startswith("mlx"):
        return "mlx-vlm"
    if model == "unimedvl":
        return "unimedvl-upstream"
    return "transformers"


def support_status(model: str, runtime: str, quant: str, task: str) -> SupportStatus:
    """Classify one combination.

    Raises:
        KeyError: If the model, runtime or task is unknown.
    """
    if model not in MODELS:
        message = f"unknown model {model!r}"
        raise KeyError(message)
    if runtime not in RUNTIMES:
        message = f"unknown runtime {runtime!r}"
        raise KeyError(message)
    if task not in TASK_NAMES:
        message = f"unknown task {task!r}"
        raise KeyError(message)

    def result(status: str, reason: str) -> SupportStatus:
        return SupportStatus(model, runtime, quant, task, status, reason)

    spec = MODELS[model]
    if task in IMAGE_TASKS and "image" not in spec.modalities:
        return result(UNSUPPORTED, f"{model} accepts text only; {task} needs images")
    for entry in MATRIX:
        if (entry.model, entry.runtime, entry.quant) != (model, runtime, quant):
            continue
        if task in entry.measured:
            return result(SUPPORTED_MEASURED, entry.evidence)
        return result(
            SUPPORTED_UNMEASURED,
            f"pinned path: {entry.how}; no published run for this task yet",
        )
    if model == "unimedvl" and runtime == "llama.cpp":
        return result(
            UNSUPPORTED,
            "UniMedVL has no GGUF export and llama.cpp does not implement its "
            "mixture-of-transformers architecture; GGUF sizes are arithmetic only",
        )
    if runtime == "transformers" and model == "unimedvl":
        return result(
            UNSUPPORTED,
            "the checkpoint's auto_map points at code that is not published; "
            "use the unimedvl-upstream runtime",
        )
    pinned = sorted(
        f"{entry.runtime}/{entry.quant}" for entry in MATRIX if entry.model == model
    )
    hint = f"; pinned paths: {', '.join(pinned)}" if pinned else ""
    return result(UNSUPPORTED, f"no pinned runtime path in this repository{hint}")


def support_matrix(
    models: tuple[str, ...] | None = None,
    tasks: tuple[str, ...] = TASK_NAMES,
) -> list[SupportStatus]:
    """Return every pinned path plus the common unsupported combinations.

    Args:
        models: Models to include (default: all).
        tasks: Tasks to include.

    Returns:
        One status per model, runtime, weight format and task.
    """
    combos: list[tuple[str, str, str]] = [
        (entry.model, entry.runtime, entry.quant) for entry in MATRIX
    ]
    for key, spec in MODELS.items():
        for quant in (*sorted(spec.gguf_files.keys() - {"mmproj-F16"}), "Q4_K_M"):
            combo = (key, "llama.cpp", quant)
            if combo not in combos:
                combos.append(combo)
        if (key, default_runtime(key, "bf16"), "bf16") not in combos:
            combos.append((key, default_runtime(key, "bf16"), "bf16"))
    selected = set(models) if models is not None else set(MODELS)
    return [
        support_status(model, runtime, quant, task)
        for model, runtime, quant in combos
        if model in selected
        for task in tasks
    ]
