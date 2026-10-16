"""Tests for scripts/measure_training_memory.py (issue #9).

The real measurement needs torch, transformers and peft, which are not test
dependencies. These tests cover the preflight, the parameter count, the
estimator comparison and the report written when a run stops early; the
measured CPU reports live in docs/case-studies/issue-1/data/.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "measure_training_memory.py"
spec = importlib.util.spec_from_file_location("measure_training_memory", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules[spec.name] = module  # type: ignore[union-attr]
spec.loader.exec_module(module)  # type: ignore[union-attr]


def _write_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header = {"__metadata__": {"format": "pt"}}
    header.update(
        {name: {"dtype": "BF16", "shape": shape} for name, shape in shapes.items()}
    )
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _hf_config(**overrides: int) -> SimpleNamespace:
    values = {
        "hidden_size": 576,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "num_hidden_layers": 30,
        "vocab_size": 49152,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parameters_are_counted_from_the_header_alone(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {"a": [4, 3], "b": [5], "c": [2, 2, 2]})

    assert module.safetensors_parameters(path) == 12 + 5 + 8


@pytest.mark.skipif(
    not Path("/proc/self/clear_refs").exists(), reason="needs Linux /proc"
)
def test_the_host_peak_restarts_after_loading() -> None:
    block = bytearray(64 * 10**6)
    block[::4096] = b"x" * len(block[::4096])
    with_block = module.host_peak_rss_gb()
    del block

    assert module.reset_host_peak()
    assert module.host_peak_rss_gb() < with_block - 0.03


def test_preflight_keeps_the_headroom_free() -> None:
    assert module.preflight(8.0, 10.0, 0.1) is None
    reason = module.preflight(9.5, 10.0, 0.1)

    assert reason is not None
    assert "9.50 GB exceeds 9.00 GB" in reason
    assert "10% headroom" in reason


def test_preflight_refuses_when_free_memory_is_unknown() -> None:
    assert "--force" in module.preflight(1.0, None, 0.1)


def test_the_estimate_uses_the_config_of_the_measured_model() -> None:
    config = module.TrainingConfig(
        model="demo/tiny", revision=None, method="lora", seq_len=256, lora_rank=8
    )

    estimate = module.estimate(config, _hf_config(), 135e6)

    # 30 layers x 7 projections x 2 matrices x 576 x rank 8 x 1.5.
    assert estimate.trainable_params == pytest.approx(30 * 7 * 2 * 576 * 8 * 1.5)
    assert estimate.total_gb_range[0] < estimate.total_gb < estimate.total_gb_range[1]
    assert estimate.logits_gb == pytest.approx(256 * 49152 * 16 / 1e9)


def test_multimodal_configs_are_sized_by_their_language_model() -> None:
    config = SimpleNamespace(text_config=_hf_config(num_hidden_layers=34))

    spec = module.model_spec(config, "demo/vlm", 4e9, image=True)

    assert spec.attention.num_layers == 34
    assert spec.attention.head_dim == 64
    assert spec.modalities == ("text", "image")


@pytest.mark.parametrize(
    ("method", "device", "expected"),
    [
        ("qlora", "cpu", "CUDA"),
        ("qlora", "mps", "MLX"),
        ("lora", "cpu", None),
        ("full", "cuda", None),
    ],
)
def test_qlora_is_reported_unsupported_without_cuda(
    method: str, device: str, expected: str | None
) -> None:
    config = module.TrainingConfig(
        model="demo/tiny", revision=None, method=method, autocast_dtype=None
    )

    reason = module.unsupported_reason(config, device)

    if expected is None:
        assert reason is None
    else:
        assert expected in reason


def test_ratios_compare_the_measured_peak_with_the_estimate() -> None:
    report = {
        "estimate": {
            "total_gb": 2.0,
            "trainable_params": 1000.0,
            "checkpoint_disk_gb": 0.5,
        },
        "train": {"host_peak_rss_gb": 3.0, "trainable_params": 1100},
        "checkpoint": {"total_gb": 0.25},
    }

    assert module.compare(report) == {
        "peak_to_estimate": 1.5,
        "trainable_params": 1.1,
        "checkpoint": 0.5,
    }
    report["train"]["gpu_peak_reserved_gb"] = 1.0
    assert module.compare(report)["peak_to_estimate"] == 0.5
    assert module.compare({"estimate": report["estimate"]}) is None


def test_a_stopped_run_still_writes_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(config: object, report: dict[str, object]) -> None:
        report["device"] = {"type": "cpu"}
        stop = module.MeasurementStopped("too big", "refused", module.EXIT_REFUSED)
        raise stop

    fake_transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *_a, **_k: _hf_config())
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(module, "count_parameters", lambda _config: 135e6)
    monkeypatch.setattr(module, "measure", refuse)
    out = tmp_path / "report.json"
    config = module.TrainingConfig(model="demo/tiny", revision="abc", method="full")

    assert module.run(config, out) == module.EXIT_REFUSED
    report = json.loads(out.read_text())
    assert report["status"] == "refused"
    assert report["reason"] == "too big"
    assert report["config"]["revision"] == "abc"
    assert report["estimate"]["method"] == "full"
    assert report["ratios"] is None


def test_out_of_memory_is_a_status_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    def exhaust(config: object, report: dict[str, object]) -> None:
        message = "CUDA out of memory. Tried to allocate 2.00 GiB\nmore detail"
        raise OutOfMemoryError(message)

    fake_transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *_a, **_k: _hf_config())
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(module, "count_parameters", lambda _config: None)
    monkeypatch.setattr(module, "measure", exhaust)
    out = tmp_path / "report.json"
    config = module.TrainingConfig(model="demo/tiny", revision=None, method="lora")

    assert module.run(config, out) == module.EXIT_OUT_OF_MEMORY
    report = json.loads(out.read_text())
    assert report["status"] == "out-of-memory"
    assert report["reason"] == "CUDA out of memory. Tried to allocate 2.00 GiB"
    assert "estimate" not in report


def test_full_fine_tuning_defaults_to_fp32_master_weights(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    full, _, _ = module.parse_args(
        ["--model", "m", "--method", "full", "--out", str(out)]
    )
    lora, _, force = module.parse_args(
        [
            "--model",
            "m",
            "--method",
            "lora",
            "--lora-rank",
            "8",
            "--force",
            "--processor-arg",
            "do_image_splitting=false",
            "--processor-arg",
            "resample=bicubic",
            "--out",
            str(out),
        ]
    )

    assert full.weights_dtype == "fp32"
    assert full.autocast_dtype == "bf16"
    assert lora.weights_dtype == "bf16"
    assert lora.lora_alpha == 16
    assert lora.processor_args == {"do_image_splitting": False, "resample": "bicubic"}
    assert force


def test_cpu_lora_run_measures_a_tiny_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=512,
    )
    model_dir = tmp_path / "tiny"
    transformers.LlamaForCausalLM(config).save_pretrained(model_dir)
    out = tmp_path / "report.json"
    run_config = module.TrainingConfig(
        model=str(model_dir),
        revision=None,
        method="lora",
        seq_len=32,
        lora_rank=4,
        steps=2,
        device="cpu",
    )

    assert module.run(run_config, out, force=True) == 0
    report = json.loads(out.read_text())
    assert report["status"] == "measured"
    assert len(report["train"]["step_seconds"]) == 2
    assert 0 < report["train"]["trainable_params"] < report["train"]["all_params"]
    assert report["train"]["optimizer_state_gb"] > 0
    assert report["train"]["host_peak_includes_load"] is not module.reset_host_peak()
    assert report["checkpoint"]["total_gb"] > 0
    assert report["ratios"]["trainable_params"] > 0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MemoryError(), True),
        (RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried"), True),
        (RuntimeError("MPS backend out of memory (MPS allocated: 17 GB)"), True),
        (RuntimeError("expected scalar type BFloat16"), False),
    ],
)
def test_cpu_and_mps_allocation_failures_count_as_out_of_memory(
    error: BaseException, expected: bool
) -> None:
    assert module.is_out_of_memory(error) is expected
