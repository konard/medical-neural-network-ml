"""Synthetic medical histories and prescriptions for OCR tests.

Every document is made of fake names and values, so it can be shared freely.
The text is created first and is the ground truth; an image is then rendered
from it with Pillow (optional dependency, ``pip install .[imaging]``). The
"handwritten" style jitters, rotates and blurs every character so that OCR
models face some of the problems of real handwriting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    "DRUGS",
    "Prescription",
    "PrescriptionItem",
    "SyntheticDocument",
    "generate_history",
    "generate_prescription",
    "render_document",
]

_FIRST_NAMES = (
    "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
    "Robin", "Avery", "Quinn", "Drew", "Kim", "Lee", "Noa", "Mika",
)  # fmt: skip
_LAST_NAMES = (
    "Smith", "Garcia", "Novak", "Okafor", "Ivanova", "Tanaka", "Silva",
    "Kowalski", "Haddad", "Larsen", "Moreau", "Chen", "Patel", "Rossi",
)  # fmt: skip

# (generic name, strengths, forms) - common generic drugs with usual strengths.
DRUGS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("amoxicillin", ("250 mg", "500 mg"), "capsule"),
    ("ibuprofen", ("200 mg", "400 mg"), "tablet"),
    ("paracetamol", ("500 mg", "1 g"), "tablet"),
    ("metformin", ("500 mg", "850 mg"), "tablet"),
    ("lisinopril", ("5 mg", "10 mg", "20 mg"), "tablet"),
    ("atorvastatin", ("10 mg", "20 mg", "40 mg"), "tablet"),
    ("omeprazole", ("20 mg", "40 mg"), "capsule"),
    ("amlodipine", ("5 mg", "10 mg"), "tablet"),
    ("sertraline", ("50 mg", "100 mg"), "tablet"),
    ("levothyroxine", ("25 mcg", "50 mcg", "100 mcg"), "tablet"),
    ("azithromycin", ("250 mg", "500 mg"), "tablet"),
    ("salbutamol", ("100 mcg",), "inhaler puff"),
)

# Prescription abbreviation -> plain English meaning.
_FREQUENCIES = {
    "OD": "once daily",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "QHS": "at bedtime",
    "PRN": "as needed",
}

_COMPLAINTS = (
    "cough for 5 days", "headache since yesterday", "abdominal pain for 2 days",
    "shortness of breath on exertion", "fever and sore throat", "painful urination",
    "chest tightness at night", "fatigue for 3 weeks",
)  # fmt: skip
_HISTORY = (
    "hypertension", "type 2 diabetes", "asthma", "hypothyroidism",
    "no chronic conditions", "hyperlipidemia", "migraine",
)  # fmt: skip
_ALLERGIES = ("none known", "penicillin", "sulfonamides", "latex", "aspirin")


@dataclass(frozen=True)
class PrescriptionItem:
    """One line of a prescription."""

    drug: str
    strength: str
    form: str
    frequency: str
    days: int

    @property
    def text(self) -> str:
        """The line as written on the prescription."""
        return f"{self.drug} {self.strength} {self.form} {self.frequency} x {self.days} days"

    @property
    def meaning(self) -> str:
        """The line in plain English."""
        plain = _FREQUENCIES[self.frequency]
        return (
            f"{self.drug} {self.strength}, one {self.form} {plain} for {self.days} days"
        )


@dataclass(frozen=True)
class Prescription:
    """A synthetic prescription."""

    patient: str
    date: str
    prescriber: str
    items: tuple[PrescriptionItem, ...]

    @property
    def text(self) -> str:
        """Full text, one field per line."""
        lines = [f"Patient: {self.patient}", f"Date: {self.date}", "Rx:"]
        lines += [f"{index}. {item.text}" for index, item in enumerate(self.items, 1)]
        lines.append(f"Dr. {self.prescriber}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SyntheticDocument:
    """A document with its ground-truth text and fields.

    Attributes:
        doc_id: Stable identifier.
        kind: ``"history"`` or ``"prescription"``.
        text: Ground-truth text, exactly what the image shows.
        fields: Structured values to check after extraction.
    """

    doc_id: str
    kind: str
    text: str
    fields: dict[str, Any] = field(default_factory=dict)


def _name(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def _date(rng: random.Random, year: int = 2026) -> str:
    return f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def generate_prescription(seed: int, max_items: int = 3) -> SyntheticDocument:
    """Create a synthetic prescription.

    Args:
        seed: Random seed.
        max_items: Maximum number of drugs.

    Returns:
        The document; ``fields["items"]`` lists drug, strength and frequency.
    """
    rng = random.Random(seed)
    items = []
    for drug, strengths, form in rng.sample(DRUGS, k=rng.randint(1, max_items)):
        items.append(
            PrescriptionItem(
                drug=drug,
                strength=rng.choice(strengths),
                form=form,
                frequency=rng.choice(tuple(_FREQUENCIES)),
                days=rng.choice((3, 5, 7, 10, 14, 30)),
            )
        )
    prescription = Prescription(_name(rng), _date(rng), _name(rng), tuple(items))
    return SyntheticDocument(
        doc_id=f"rx-{seed:05d}",
        kind="prescription",
        text=prescription.text,
        fields={
            "patient": prescription.patient,
            "date": prescription.date,
            "items": [
                {"drug": i.drug, "strength": i.strength, "frequency": i.frequency}
                for i in items
            ],
            "meaning": [item.meaning for item in items],
        },
    )


def generate_history(seed: int) -> SyntheticDocument:
    """Create a synthetic medical history (clinic note).

    Args:
        seed: Random seed.

    Returns:
        The document with vitals, history and allergies as fields.
    """
    rng = random.Random(seed)
    fields: dict[str, Any] = {
        "patient": _name(rng),
        "date of birth": _date(rng, rng.randint(1940, 2005)),
        "visit date": _date(rng),
        "complaint": rng.choice(_COMPLAINTS),
        "history": rng.choice(_HISTORY),
        "allergies": rng.choice(_ALLERGIES),
        "blood pressure": f"{rng.randint(100, 165)}/{rng.randint(60, 100)} mmHg",
        "heart rate": f"{rng.randint(55, 115)} bpm",
        "temperature": f"{rng.uniform(36.1, 39.4):.1f} C",
    }
    text = "\n".join(f"{key.capitalize()}: {value}" for key, value in fields.items())
    return SyntheticDocument(f"hx-{seed:05d}", "history", text, fields)


def _load_font(size: int) -> Any:  # noqa: ANN401 - Pillow font classes vary
    from PIL import ImageFont  # noqa: PLC0415 - optional dependency

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def render_document(
    document: SyntheticDocument,
    style: str = "printed",
    seed: int = 0,
    font_size: int = 28,
) -> Image.Image:
    """Render a document to a grayscale image.

    Args:
        document: Document to draw.
        style: ``"printed"`` for clean text or ``"handwritten"`` for jittered,
            slanted, uneven characters.
        seed: Random seed for the handwriting distortions.
        font_size: Character height in pixels.

    Returns:
        A Pillow image (mode ``"L"``).

    Raises:
        ValueError: If the style is unknown.
    """
    from PIL import Image, ImageDraw, ImageFilter  # noqa: PLC0415

    if style not in {"printed", "handwritten"}:
        message = f"unknown style {style!r}"
        raise ValueError(message)
    rng = random.Random(seed)
    font = _load_font(font_size)
    lines = document.text.splitlines()
    width = max(len(line) for line in lines) * font_size * 2 // 3 + 80
    line_height = int(font_size * 1.6)
    height = len(lines) * line_height + 80
    page = Image.new("L", (width, height), color=250)
    draw = ImageDraw.Draw(page)
    if style == "printed":
        for row, line in enumerate(lines):
            draw.text((40, 40 + row * line_height), line, fill=20, font=font)
        return page

    slant = rng.uniform(-6, 6)
    for row, line in enumerate(lines):
        x = 40.0
        baseline = 40 + row * line_height + rng.uniform(-3, 3)
        for char in line:
            size = max(8, int(font_size * rng.uniform(0.85, 1.15)))
            glyph_font = _load_font(size)
            box = (size * 2, size * 2)
            glyph = Image.new("L", box, color=0)
            ImageDraw.Draw(glyph).text(
                (size // 2, size // 3), char, fill=255, font=glyph_font
            )
            glyph = glyph.rotate(
                slant + rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR
            )
            ink = rng.randint(10, 70)
            page.paste(
                ink, (int(x) - size // 2, int(baseline + rng.uniform(-2, 2))), glyph
            )
            x += size * 0.55 + rng.uniform(-1, 2)
            baseline += rng.uniform(-0.6, 0.6)
    return page.filter(ImageFilter.GaussianBlur(radius=0.6))
