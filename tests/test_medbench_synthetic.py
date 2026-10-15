"""Tests for the synthetic diagnosis cases and augmentation (issue #1).

Only text cases are generated; images and documents are real files (see
``test_medbench_real_data.py``).
"""

from __future__ import annotations

import pytest

from medbench.augment import (
    LabeledImage,
    adjust_contrast,
    augment_image,
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


def _image(modality: str, side: str) -> LabeledImage:
    """A small gradient image; augmentation only needs pixels and labels."""
    pixels = [[(x * 16 + y) % 256 for x in range(12)] for y in range(8)]
    return LabeledImage("us-1", modality, pixels, "cyst", side)


class TestAugment:
    """Label-preserving augmentation."""

    def test_flip_swaps_side(self) -> None:
        """Mirroring moves the finding and swaps the label."""
        image = _image("mri", side="left")
        flipped = flip_horizontal(image)
        assert flipped.side == "right"
        assert flipped.pixels[0] == image.pixels[0][::-1]
        assert flip_horizontal(flipped).side == image.side
        assert flip_horizontal(_image("mri", side="both")).side == "both"

    def test_xray_avoids_flips(self) -> None:
        """Flips are not offered for chest X-rays."""
        assert "flip_with_side_swap" not in safe_image_augmentations("xray")
        assert "flip_with_side_swap" in safe_image_augmentations("mri")

    def test_augment_keeps_labels(self) -> None:
        """Augmented images keep the finding, size and a truthful side."""
        image = _image("ultrasound", side="left")
        for seed in range(6):
            augmented = augment_image(image, seed)
            assert augmented.finding == image.finding
            assert augmented.image_id.startswith("us-1")
            assert len(augmented.pixels) == len(image.pixels)
            assert len(augmented.pixels[0]) == len(image.pixels[0])
            flipped = "-flip" in augmented.image_id
            assert augmented.side == ("right" if flipped else "left")
        xray = _image("xray", side="left")
        assert all(augment_image(xray, s).side == "left" for s in range(6))

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
