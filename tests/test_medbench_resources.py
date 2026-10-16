"""Tests for the model catalogue and the resource calculator (issue #1)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from medbench.models import MODELS, get_model
from medbench.resources import (
    HARDWARE,
    estimate_finetune_memory,
    estimate_inference,
    estimate_training,
    kv_cache_bytes,
    weights_bytes,
)


class TestModels:
    """The catalogue holds the models named in the issue."""

    def test_issue_models_are_present(self) -> None:
        """MedGemma 27B and UniMedVL were requested explicitly."""
        assert "medgemma-27b-it" in MODELS
        assert "unimedvl" in MODELS

    def test_unknown_model_lists_known_ones(self) -> None:
        """The error message helps the user pick a valid name."""
        with pytest.raises(KeyError, match="medgemma-4b-it"):
            get_model("no-such-model")

    def test_gguf_sizes_match_bits_per_weight(self) -> None:
        """Published Q4_K_M files are within 15% of the formula."""
        for model in MODELS.values():
            if "Q4_K_M" in model.gguf_files:
                estimate = weights_bytes(model.total_params, "Q4_K_M")
                assert abs(estimate / model.gguf_files["Q4_K_M"] - 1) < 0.15


class TestMemory:
    """Weights and KV cache sizes."""

    def test_bf16_weights_are_two_bytes_per_parameter(self) -> None:
        """27.4 B parameters take about 55 GB in bf16."""
        model = get_model("medgemma-27b-it")
        assert weights_bytes(model.total_params, "bf16") == pytest.approx(
            54.86e9, rel=1e-3
        )

    def test_unknown_quantization(self) -> None:
        """Typos in the format are reported."""
        with pytest.raises(KeyError):
            weights_bytes(1e9, "Q4_X")

    def test_sliding_window_caps_kv_cache(self) -> None:
        """Gemma 3 local layers stop growing after 1024 tokens."""
        model = get_model("medgemma-27b-it")
        per_token = 2 * 16 * 128 * 2
        short = kv_cache_bytes(model, 1024)
        assert short == per_token * 62 * 1024
        long = kv_cache_bytes(model, 8192)
        assert long == per_token * (10 * 8192 + 52 * 1024)
        assert long < per_token * 62 * 8192 / 3

    def test_full_attention_kv_cache_is_linear(self) -> None:
        """Without a sliding window the cache grows with every token."""
        model = get_model("qwen2.5-vl-7b")
        assert kv_cache_bytes(model, 2000) == 2 * kv_cache_bytes(model, 1000)


class TestInference:
    """Speed and fit estimates."""

    def test_27b_does_not_fit_github_runner(self) -> None:
        """A 4-bit 27B model needs more than the 16 GB of a hosted runner."""
        estimate = estimate_inference(
            get_model("medgemma-27b-it"), HARDWARE["github-ubuntu-4cpu"], quant="Q4_K_M"
        )
        assert not estimate.fits
        assert estimate.weights_gb > 16

    def test_4b_needs_more_than_the_sandbox_limit(self) -> None:
        """The 3.1 GB cgroup of the dev VM is too small; a hosted runner is not."""
        model = get_model("medgemma-4b-it")
        sandbox = estimate_inference(model, HARDWARE["dev-vm-6cpu"], quant="Q4_K_M")
        runner = estimate_inference(
            model, HARDWARE["github-ubuntu-4cpu"], quant="Q4_K_M"
        )
        assert not sandbox.fits
        assert runner.fits
        assert runner.total_memory_gb < 5

    def test_dev_vm_matches_measured_speed(self) -> None:
        """llama-bench measured 19.4 prompt and 4.2 output tokens/s."""
        estimate = estimate_inference(
            get_model("medgemma-4b-it"), HARDWARE["dev-vm-6cpu"], quant="Q4_K_M"
        )
        assert estimate.prefill_tokens_per_s == pytest.approx(19.4, rel=0.25)
        assert estimate.decode_tokens_per_s == pytest.approx(4.2, rel=0.25)

    def test_github_runner_matches_measured_speed(self) -> None:
        """CI run 38550186566: 31 prompt, 7.8 output tokens/s, 66.7 s per image.

        The X-ray request had 51 text tokens, one image and a 3-token answer;
        the diagnosis request had 66 text tokens and a 36-token answer.
        """
        model = get_model("medgemma-4b-it")
        runner = HARDWARE["github-ubuntu-4cpu"]
        xray = estimate_inference(
            model,
            runner,
            quant="Q4_K_M",
            prompt_tokens=51,
            output_tokens=3,
            images=1,
        )
        diagnosis = estimate_inference(
            model, runner, quant="Q4_K_M", prompt_tokens=66, output_tokens=36
        )
        assert xray.prefill_tokens_per_s == pytest.approx(31, rel=0.1)
        assert xray.decode_tokens_per_s == pytest.approx(7.8, rel=0.1)
        assert xray.seconds_per_request == pytest.approx(66.7, rel=0.1)
        assert diagnosis.seconds_per_request == pytest.approx(6.5, rel=0.1)

    def test_vision_efficiency_defaults_to_prefill_efficiency(self) -> None:
        """Profiles without their own image-encoder number reuse the prompt one."""
        model = get_model("medgemma-4b-it")
        runner = HARDWARE["github-ubuntu-4cpu"]
        same = replace(runner, vision_efficiency=None)
        explicit = replace(runner, vision_efficiency=runner.prefill_efficiency)
        assert estimate_inference(model, same, images=1) == estimate_inference(
            model, explicit, images=1
        )
        assert (
            estimate_inference(model, runner, images=1).seconds_per_request
            > estimate_inference(model, same, images=1).seconds_per_request
        )

    def test_gpu_is_much_faster_than_cpu(self) -> None:
        """Decoding on an H100 is over 50x faster than on a CI runner."""
        model = get_model("medgemma-4b-it")
        cpu = estimate_inference(model, HARDWARE["github-ubuntu-4cpu"], quant="Q4_K_M")
        gpu = estimate_inference(model, HARDWARE["h100-80gb"], quant="Q4_K_M")
        assert gpu.decode_tokens_per_s > 50 * cpu.decode_tokens_per_s
        assert gpu.seconds_per_request < cpu.seconds_per_request

    def test_images_add_prompt_tokens(self) -> None:
        """Each image adds its tokens and encoder time."""
        model = get_model("medgemma-4b-it")
        hardware = HARDWARE["l4-24gb"]
        text_only = estimate_inference(model, hardware)
        with_image = estimate_inference(model, hardware, images=1)
        assert with_image.seconds_per_request > text_only.seconds_per_request
        assert with_image.kv_cache_gb > text_only.kv_cache_gb

    def test_moe_decodes_with_active_parameters(self) -> None:
        """UniMedVL reads only one 7B expert per token."""
        model = get_model("unimedvl")
        estimate = estimate_inference(model, HARDWARE["a100-80gb"])
        dense_bytes = weights_bytes(model.total_params, "bf16")
        assert estimate.decode_tokens_per_s > 2039e9 * 0.6 / dense_bytes
        assert "weights_gb" in estimate.as_dict()


class TestFineTune:
    """Fine-tuning memory."""

    def test_methods_are_ordered(self) -> None:
        """Full > LoRA > QLoRA, and QLoRA 27B fits a 24 GB card."""
        model = get_model("medgemma-27b-it")
        full = estimate_finetune_memory(model, "full", hidden_size=5376)
        lora = estimate_finetune_memory(model, "lora", hidden_size=5376)
        qlora = estimate_finetune_memory(model, "qlora", hidden_size=5376)
        assert full.total_gb > 400
        assert full.total_gb > lora.total_gb > qlora.total_gb
        assert qlora.total_gb < 24
        assert lora.trainable_params < 0.01 * model.total_params

    def test_checkpointing_saves_memory(self) -> None:
        """Recomputing activations uses less memory than storing them."""
        model = get_model("medgemma-4b-it")
        saved = estimate_finetune_memory(model, "lora", 2560, seq_len=4096)
        stored = estimate_finetune_memory(
            model, "lora", 2560, seq_len=4096, gradient_checkpointing=False
        )
        assert stored.activations_gb > 10 * saved.activations_gb
        assert saved.as_dict()["method"] == "lora"

    def test_unknown_method(self) -> None:
        """Only full, lora and qlora are supported."""
        with pytest.raises(ValueError, match="unknown fine-tuning method"):
            estimate_finetune_memory(get_model("medgemma-4b-it"), "dora", 2560)


class TestTraining:
    """Training compute and cost."""

    def test_six_n_d(self) -> None:
        """Full training uses 6 FLOPs per parameter and token."""
        estimate = estimate_training(1e9, 20e9, HARDWARE["h100-80gb"], mfu=0.5)
        assert estimate.flops == pytest.approx(1.2e20)
        assert estimate.gpu_hours == pytest.approx(1.2e20 / (989e12 * 0.5) / 3600)
        assert estimate.cost_usd == pytest.approx(estimate.gpu_hours * 3.99)
        low, high = estimate.cost_usd_range
        assert low < estimate.cost_usd < high
        assert "checked 2026-10-16" in estimate.price_source

    def test_user_supplied_price(self) -> None:
        """A price passed by the user replaces the dated example."""
        estimate = estimate_training(
            1e9, 20e9, HARDWARE["h100-80gb"], price_per_hour=2.0
        )
        assert estimate.cost_usd == pytest.approx(estimate.gpu_hours * 2.0)
        assert estimate.cost_usd_range == (estimate.cost_usd, estimate.cost_usd)
        assert estimate.price_source == "user-supplied"

    def test_more_gpus_shorten_wall_clock(self) -> None:
        """Eight GPUs finish eight times sooner for the same cost."""
        one = estimate_training(4e9, 1e9, HARDWARE["a100-80gb"], gpus=1, method="lora")
        eight = estimate_training(
            4e9, 1e9, HARDWARE["a100-80gb"], gpus=8, method="lora"
        )
        assert eight.wall_clock_hours == pytest.approx(one.wall_clock_hours / 8)
        assert eight.cost_usd == pytest.approx(one.cost_usd)
        assert one.as_dict()["flops"] == pytest.approx(4 * 4e9 * 1e9)


def test_separate_vision_projector_is_included() -> None:
    model = get_model("medgemma-4b-it")
    estimate = estimate_inference(model, HARDWARE["m3-pro-18gb"], quant="Q4_K_M")
    assert estimate.weights_gb == pytest.approx(3.341)
    assert estimate.total_memory_gb > 4.3
    large = estimate_inference(
        get_model("medgemma-27b-it"), HARDWARE["m3-pro-18gb"], quant="Q4_K_M"
    )
    assert not large.fits


class TestCalibration:
    """Issue #9: estimates carry sources, dense compute and ranges."""

    def test_every_rented_gpu_has_a_dated_price_source(self) -> None:
        for profile in HARDWARE.values():
            if profile.price_per_hour:
                low, high = profile.price_range
                assert low <= profile.price_per_hour <= high, profile.key
                assert profile.price_source != "owned hardware", profile.key
                assert profile.spec_source, profile.key

    def test_compute_is_dense_not_sparse(self) -> None:
        """Sheets quote 2:4 sparsity; the calculator must use half of it."""
        for profile in HARDWARE.values():
            if profile.sparse_tflops is not None:
                assert profile.compute_tflops == pytest.approx(
                    profile.sparse_tflops / 2, rel=0.01
                ), profile.key

    def test_memory_range_brackets_point_estimate(self) -> None:
        estimate = estimate_inference(
            get_model("medgemma-27b-it"), HARDWARE["rtx4090-24gb"], quant="Q4_K_M"
        )
        low, high = estimate.total_memory_gb_range
        assert low < estimate.total_memory_gb < high
        slow, fast = estimate.decode_tokens_per_s_range
        assert slow < estimate.decode_tokens_per_s < fast
        quick, long = estimate.seconds_per_request_range
        assert quick < estimate.seconds_per_request < long

    @pytest.mark.parametrize(
        ("model", "hardware", "quant", "capacity"),
        [
            ("medgemma-4b-it", "a100-80gb", "Q4_K_M", "fits"),
            ("medgemma-27b-it", "m3-pro-18gb", "Q4_K_M", "does-not-fit"),
            # 18.5 GB point estimate on a 24 GB card: the high end does not fit.
            ("medgemma-27b-it", "rtx4090-24gb", "Q4_K_M", "uncertain"),
        ],
    )
    def test_capacity_uses_the_range(
        self, model: str, hardware: str, quant: str, capacity: str
    ) -> None:
        estimate = estimate_inference(
            get_model(model), HARDWARE[hardware], quant=quant, images=1
        )
        assert estimate.capacity == capacity

    def test_finetune_reports_range_and_checkpoint_disk(self) -> None:
        model = get_model("medgemma-27b-it")
        full = estimate_finetune_memory(model, "full", 5376)
        qlora = estimate_finetune_memory(model, "qlora", 5376)
        low, high = qlora.total_gb_range
        assert low < qlora.total_gb < high
        assert full.checkpoint_disk_gb == pytest.approx(27.432e9 * 14 / 1e9)
        assert qlora.checkpoint_disk_gb == pytest.approx(
            qlora.trainable_params * 14 / 1e9
        )
