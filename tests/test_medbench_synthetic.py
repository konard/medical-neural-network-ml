"""Tests for the synthetic data generators and augmentation (issue #1)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from medbench.augment import (
    adjust_contrast,
    augment_phantom,
    flip_horizontal,
    ocr_noise,
    safe_image_augmentations,
    shift,
    synonym_swap,
)
from medbench.synthetic_cases import (
    CONDITIONS,
    build_diagnosis_prompt,
    generate_cases,
    parse_ranked_diagnoses,
    rank_conditions,
)
from medbench.synthetic_documents import (
    generate_history,
    generate_prescription,
    render_document,
)
from medbench.synthetic_images import (
    chest_xray,
    encode_png,
    generate_phantoms,
    mri_brain,
    ultrasound,
    write_png,
)


class TestCases:
    """Synthetic patients and the reference ranking."""

    def test_same_seed_same_patients(self) -> None:
        """Generation is deterministic."""
        assert generate_cases(5, seed=3) == generate_cases(5, seed=3)
        assert generate_cases(5, seed=3) != generate_cases(5, seed=4)

    def test_every_case_is_consistent(self) -> None:
        """Symptoms come from the true condition and probabilities sum to 1."""
        by_name = {condition.name: condition for condition in CONDITIONS}
        for case in generate_cases(50):
            assert case.symptoms
            assert set(case.symptoms) <= set(by_name[case.diagnosis].symptoms)
            assert sum(case.reference_probabilities) == pytest.approx(1.0)
            assert case.as_dict()["diagnosis"] == case.diagnosis

    def test_ranking_prefers_matching_condition(self) -> None:
        """Classic symptom sets rank their condition first."""
        assert rank_conditions(["painful urination", "frequent urination"])[0][0] == (
            "urinary tract infection"
        )
        assert rank_conditions(["sore throat", "fever", "swollen neck glands"])[0][
            0
        ] == ("streptococcal pharyngitis")

    def test_absent_symptoms_count(self) -> None:
        """Without fever, influenza becomes less likely."""
        with_fever = dict(rank_conditions(["cough", "muscle aches", "fever"]))
        without = dict(rank_conditions(["cough", "muscle aches"]))
        assert with_fever["influenza"] > without["influenza"]

    def test_prompt_mentions_symptoms(self) -> None:
        """The prompt carries age, sex and every symptom."""
        case = generate_cases(1)[0]
        prompt = build_diagnosis_prompt(case, top_k=3)
        assert str(case.age) in prompt
        assert all(symptom in prompt for symptom in case.symptoms)
        assert "3 most likely" in prompt


class TestParsing:
    """Free model output becomes canonical names."""

    def test_markdown_and_aliases(self) -> None:
        """Bold names, explanations and aliases are understood."""
        text = (
            "Here is my differential:\n"
            "1. **Flu** - fever and aches\n"
            "2. COVID-19 (SARS-CoV-2)\n"
            "3) Community-acquired pneumonia: consider X-ray\n"
            "- strep throat\n"
            "4. Flu again\n"
            "5. Lyme disease\n"
        )
        assert parse_ranked_diagnoses(text) == [
            "influenza",
            "covid-19",
            "community-acquired pneumonia",
            "streptococcal pharyngitis",
            "lyme disease",
        ]

    def test_no_list(self) -> None:
        """Plain prose without a list yields nothing."""
        assert parse_ranked_diagnoses("I cannot give medical advice.") == []


class TestDocuments:
    """Synthetic prescriptions and histories."""

    def test_prescription_fields_match_text(self) -> None:
        """Every drug and strength appears in the ground-truth text."""
        document = generate_prescription(11)
        assert document.kind == "prescription"
        for item in document.fields["items"]:
            assert item["drug"] in document.text
            assert item["strength"] in document.text
        assert len(document.fields["meaning"]) == len(document.fields["items"])

    def test_history_has_vitals(self) -> None:
        """Histories contain blood pressure and temperature."""
        document = generate_history(2)
        assert "Blood pressure:" in document.text
        assert document.fields["temperature"].endswith(" C")
        assert generate_history(2) == document

    @pytest.mark.parametrize("style", ["printed", "handwritten"])
    def test_render(self, style: str) -> None:
        """Rendering gives a grayscale image with dark ink."""
        pytest.importorskip("PIL")
        image = render_document(generate_prescription(1), style=style, seed=1)
        assert image.mode == "L"
        assert image.width > 200
        low, _ = image.getextrema()
        assert low < 100

    def test_render_rejects_unknown_style(self) -> None:
        """Only printed and handwritten are supported."""
        pytest.importorskip("PIL")
        with pytest.raises(ValueError, match="unknown style"):
            render_document(generate_history(1), style="typed")


class TestPhantoms:
    """Synthetic scans."""

    @pytest.mark.parametrize("make", [chest_xray, mri_brain, ultrasound])
    def test_finding_matches_mask(self, make: object) -> None:
        """Abnormal images have a mask; normal ones do not."""
        assert callable(make)
        sick = make(1, 64, abnormal=True)
        healthy = make(1, 64, abnormal=False)
        assert sick.has_finding
        assert sick.side in {"left", "right"}
        assert any(any(row) for row in sick.mask)
        assert not healthy.has_finding
        assert healthy.side == "none"
        assert not any(any(row) for row in healthy.mask)
        assert sick.size == (64, 64)
        assert all(0 <= v <= 255 for row in sick.pixels for v in row)

    def test_xray_nodule_follows_radiology_convention(self) -> None:
        """The patient's left lung is on the image right."""
        for seed in range(6):
            phantom = chest_xray(seed, 64, abnormal=True)
            columns = [x for row in phantom.mask for x, v in enumerate(row) if v]
            image_right = sum(columns) / len(columns) > 32
            assert image_right == (phantom.side == "left")

    def test_balanced_set(self) -> None:
        """Half of a generated set has a finding."""
        phantoms = generate_phantoms("mri", 6, size=48)
        assert sum(p.has_finding for p in phantoms) == 3
        with pytest.raises(KeyError, match="unknown modality"):
            generate_phantoms("pet", 1)

    def test_png_round_trip(self, tmp_path: Path) -> None:
        """The PNG writer produces a valid file with the right pixels."""
        pixels = [[0, 128, 255], [10, 20, 30]]
        data = encode_png(pixels)
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (3, 2)
        length = struct.unpack(">I", data[33:37])[0]
        raw = zlib.decompress(data[41 : 41 + length])
        assert raw == b"\x00\x00\x80\xff\x00\x0a\x14\x1e"
        path = write_png(tmp_path / "sub" / "image.png", pixels)
        assert path.read_bytes() == data

    def test_png_opens_in_pillow(self, tmp_path: Path) -> None:
        """Other tools can read the files."""
        image_module = pytest.importorskip("PIL.Image")
        phantom = ultrasound(0, 32)
        path = write_png(tmp_path / "us.png", phantom.pixels)
        with image_module.open(path) as image:
            assert image.size == (32, 32)
            assert image.getpixel((5, 20)) == phantom.pixels[20][5]


class TestAugment:
    """Label-preserving augmentation."""

    def test_flip_swaps_side(self) -> None:
        """Mirroring moves the finding and swaps the label."""
        phantom = mri_brain(4, 48, abnormal=True)
        flipped = flip_horizontal(phantom)
        assert flipped.side != phantom.side
        assert flipped.pixels[0] == phantom.pixels[0][::-1]
        assert flip_horizontal(flipped).side == phantom.side

    def test_xray_avoids_flips(self) -> None:
        """Flips are not offered for chest X-rays."""
        assert "flip_with_side_swap" not in safe_image_augmentations("xray")
        assert "flip_with_side_swap" in safe_image_augmentations("mri")

    def test_augment_keeps_label_and_mask(self) -> None:
        """Augmented images keep the finding and a non-empty mask."""
        phantom = ultrasound(2, 48, abnormal=True)
        for seed in range(4):
            augmented = augment_phantom(phantom, seed)
            assert augmented.finding == phantom.finding
            assert any(any(row) for row in augmented.mask)
            assert augmented.size == phantom.size

    def test_pixel_operations(self) -> None:
        """Contrast clips to 0-255 and shift fills empty pixels."""
        assert adjust_contrast([[0, 128, 255]], 2.0) == [[0, 128, 255]]
        assert shift([[1, 2], [3, 4]], 1, 0) == [[0, 1], [0, 3]]

    def test_ocr_noise(self) -> None:
        """Rate 0 changes nothing; rate 1 changes every confusable char."""
        text = "metformin 500 mg"
        assert ocr_noise(text, 0.0) == text
        noisy = ocr_noise(text, 1.0, seed=1)
        assert noisy != text
        assert "500" not in noisy

    def test_synonym_swap(self) -> None:
        """Symptoms get another name."""
        swapped = synonym_swap("fever and shortness of breath", seed=0)
        assert "fever" not in swapped
        assert "shortness of breath" not in swapped
