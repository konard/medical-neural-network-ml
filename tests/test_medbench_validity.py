"""Label review, grouped splits and sample-size targets (issue #7)."""

from __future__ import annotations

import json

import pytest

from medbench.cli import main
from medbench.metrics import wilson_interval
from medbench.real_data import Sample, load_manifest
from medbench.validity import (
    DEFAULT_SPLITS,
    class_balance,
    grouped_split,
    load_label_review,
    required_sample_size,
    sample_size_targets,
    split_of,
)

MANIFEST = load_manifest()
REVIEW = load_label_review()
REVIEWED = [s for s in MANIFEST if s.task != "diagnosis-ddxplus"]


def test_every_real_image_and_page_is_reviewed() -> None:
    """The review covers exactly the imaging and OCR samples."""
    assert set(REVIEW.entries) == {s.sample_id for s in REVIEWED}


def test_review_states_it_is_not_clinician_reviewed() -> None:
    """Provenance is explicit: maintainers, not clinicians."""
    assert REVIEW.clinician_reviewed is False
    assert "not clinicians" in REVIEW.reviewer
    for entry in REVIEW.entries.values():
        assert entry.view
        assert entry.label_source
        assert entry.source_group
        assert entry.uncertainty


@pytest.mark.parametrize(
    "sample", [s for s in REVIEWED if "side" in s.truth], ids=lambda s: s.sample_id
)
def test_laterality_matches_the_scored_side(sample: Sample) -> None:
    """The reviewed side agrees with the side the task scores."""
    side = sample.truth["side"]
    laterality = REVIEW.entries[sample.sample_id].laterality
    if side in {"left", "right"}:
        assert laterality.startswith(side) or f"organ is the {side}" in laterality
    else:
        assert not laterality.startswith(("left", "right"))


def test_xray_views_are_not_all_frontal_pa() -> None:
    """At least two X-rays are AP, which the prompt must not hide."""
    views = [REVIEW.entries[s.sample_id].view for s in REVIEWED if "xray" in s.task]
    assert sum(view.startswith("AP") for view in views) >= 2


def test_class_balance() -> None:
    """Every imaging task has normal and abnormal items."""
    balance = class_balance(MANIFEST)
    assert set(balance) == {"imaging-xray", "imaging-mri", "imaging-ultrasound"}
    for counts in balance.values():
        assert counts["normal"] >= 3
        assert counts["abnormal"] >= 4


def test_split_is_deterministic_and_proportional() -> None:
    """The same group always gets the same split, close to the shares."""
    groups = [f"patient-{i}" for i in range(20_000)]
    first = [split_of(g) for g in groups]
    assert first == [split_of(g) for g in groups]
    for name, share in DEFAULT_SPLITS:
        assert abs(first.count(name) / len(groups) - share) < 0.02


def test_grouped_split_never_splits_a_source_group() -> None:
    """Items sharing a source or patient never land in two splits."""
    splits = grouped_split(REVIEWED, REVIEW.group_of)
    seen: dict[str, str] = {}
    for name, items in splits.items():
        for sample in items:
            group = REVIEW.group_of(sample)
            assert seen.setdefault(group, name) == name
    assert sum(len(items) for items in splits.values()) == len(REVIEWED)


def test_grouped_split_with_custom_shares() -> None:
    """Zero share means no items; a single split takes everything."""
    items = list(range(50))
    only_test = grouped_split(items, str, (("dev", 0.0), ("test", 1.0)))
    assert only_test == {"dev": [], "test": items}
    with pytest.raises(ValueError, match="shares"):
        split_of("x", (("dev", -1.0), ("test", 1.0)))


@pytest.mark.parametrize(
    ("half_width", "expected", "required"),
    [(0.05, 0.5, 381), (0.10, 0.8, 60), (0.03, 0.5, 1064)],
)
def test_required_sample_size(
    half_width: float, expected: float, required: int
) -> None:
    """The size is the smallest n whose Wilson interval is narrow enough."""
    n = required_sample_size(half_width, expected)
    assert n == required
    low, high = wilson_interval(round(expected * n), n)
    assert (high - low) / 2 <= half_width + 0.002
    low, high = wilson_interval(round(expected * (n - 1)), n - 1)
    assert (high - low) / 2 > half_width - 0.002


@pytest.mark.parametrize(("half_width", "expected"), [(0.0, 0.5), (0.1, 1.5)])
def test_required_sample_size_rejects_bad_input(
    half_width: float, expected: float
) -> None:
    """Impossible targets are refused."""
    with pytest.raises(ValueError, match="between"):
        required_sample_size(half_width, expected)


def test_targets_exceed_the_current_teaching_set() -> None:
    """The current real sets are far below the planned sizes."""
    targets = sample_size_targets()
    imaging = [s for s in MANIFEST if s.task.startswith("imaging")]
    assert targets["imaging"]["required"] > len(imaging)
    assert targets["diagnosis-ddxplus"]["required"] == 381


def test_cli_review(capsys: pytest.CaptureFixture[str]) -> None:
    """``medbench review`` prints the provenance and the targets."""
    assert main(["review", "--task", "imaging-ultrasound"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["clinician_reviewed"] is False
    assert report["class_balance"] == {
        "imaging-ultrasound": {"abnormal": 7, "normal": 3}
    }
    assert {s["sample_id"] for s in report["samples"]} == {
        s.sample_id for s in MANIFEST if s.task == "imaging-ultrasound"
    }
