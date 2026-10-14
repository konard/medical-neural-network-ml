"""Data augmentation: make many training examples from one.

Augmentation changes an example in ways that keep its label true, so a model
sees more variety without new labelling work. In medicine some "obvious"
changes are wrong: flipping a chest X-ray left-to-right moves the heart to the
other side and swaps the side of every finding. :func:`flip_horizontal`
therefore also swaps the ``side`` label, and :func:`safe_image_augmentations`
lists which changes are safe for each modality.
"""

from __future__ import annotations

import random

from medbench.synthetic_images import Phantom

__all__ = [
    "OCR_CONFUSIONS",
    "adjust_contrast",
    "add_gaussian_noise",
    "augment_phantom",
    "flip_horizontal",
    "ocr_noise",
    "safe_image_augmentations",
    "shift",
    "synonym_swap",
]

Pixels = list[list[int]]

# Characters that OCR engines often confuse, and that matter in medicine:
# "1" vs "l" turns 1 mg into l mg, "0" vs "O" changes doses.
OCR_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "0": ("O", "o"),
    "O": ("0",),
    "1": ("l", "I", "7"),
    "l": ("1", "I"),
    "5": ("S",),
    "S": ("5",),
    "8": ("B",),
    "m": ("rn", "nn"),
    "g": ("q", "9"),
    "c": ("e",),
    "e": ("c",),
    ".": (",", ""),
}

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "shortness of breath": ("difficulty breathing", "breathlessness", "dyspnea"),
    "fever": ("high temperature", "pyrexia"),
    "runny nose": ("rhinorrhea", "nasal discharge"),
    "abdominal pain": ("stomach pain", "belly pain"),
    "fatigue": ("tiredness", "lack of energy"),
    "headache": ("head pain", "cephalgia"),
    "painful urination": ("burning on urination", "dysuria"),
    "chest pain": ("chest discomfort", "pain in the chest"),
}


def _clip(value: float) -> int:
    return max(0, min(255, round(value)))


def flip_horizontal(phantom: Phantom) -> Phantom:
    """Mirror an image and swap its left/right label to keep it truthful."""
    swapped = {"left": "right", "right": "left"}.get(phantom.side, phantom.side)
    return Phantom(
        image_id=f"{phantom.image_id}-flip",
        modality=phantom.modality,
        pixels=[row[::-1] for row in phantom.pixels],
        mask=[row[::-1] for row in phantom.mask],
        finding=phantom.finding,
        side=swapped,
        question=phantom.question,
    )


def adjust_contrast(pixels: Pixels, factor: float, brightness: float = 0.0) -> Pixels:
    """Stretch contrast around mid-grey and shift brightness."""
    return [
        [_clip((v - 128) * factor + 128 + brightness) for v in row] for row in pixels
    ]


def add_gaussian_noise(pixels: Pixels, sigma: float, seed: int = 0) -> Pixels:
    """Add random noise, like a lower-dose or older scanner."""
    rng = random.Random(seed)
    return [[_clip(v + rng.gauss(0, sigma)) for v in row] for row in pixels]


def shift(pixels: Pixels, dx: int, dy: int, fill: int = 0) -> Pixels:
    """Move the image by whole pixels, filling uncovered areas with ``fill``."""
    height, width = len(pixels), len(pixels[0])
    return [
        [
            pixels[y - dy][x - dx]
            if 0 <= y - dy < height and 0 <= x - dx < width
            else fill
            for x in range(width)
        ]
        for y in range(height)
    ]


def safe_image_augmentations(modality: str) -> tuple[str, ...]:
    """Augmentations that keep labels true for ``modality``.

    Horizontal flips are only "safe" if the side label is swapped too; for
    chest X-rays many teams avoid them because the flipped anatomy (heart on
    the right) is itself abnormal.
    """
    common = ("contrast", "noise", "small_shift")
    if modality == "xray":
        return common
    return (*common, "flip_with_side_swap")


def augment_phantom(phantom: Phantom, seed: int) -> Phantom:
    """Apply a random safe augmentation chain to a phantom."""
    rng = random.Random(seed)
    options = safe_image_augmentations(phantom.modality)
    result = phantom
    if "flip_with_side_swap" in options and rng.random() < 0.5:
        result = flip_horizontal(result)
    dx, dy = rng.randint(-3, 3), rng.randint(-3, 3)
    pixels = shift(result.pixels, dx, dy)
    mask = shift(result.mask, dx, dy)
    pixels = adjust_contrast(pixels, rng.uniform(0.85, 1.15), rng.uniform(-10, 10))
    pixels = add_gaussian_noise(pixels, rng.uniform(0, 6), seed=seed)
    return Phantom(
        image_id=f"{result.image_id}-aug{seed}",
        modality=result.modality,
        pixels=pixels,
        mask=mask,
        finding=result.finding,
        side=result.side,
        question=result.question,
    )


def ocr_noise(text: str, rate: float, seed: int = 0) -> str:
    """Imitate OCR mistakes by swapping look-alike characters.

    Useful to train a post-OCR correction model, or to test how a downstream
    extractor copes with imperfect OCR.
    """
    rng = random.Random(seed)
    output = []
    for char in text:
        if char in OCR_CONFUSIONS and rng.random() < rate:
            output.append(rng.choice(OCR_CONFUSIONS[char]))
        else:
            output.append(char)
    return "".join(output)


def synonym_swap(text: str, seed: int = 0) -> str:
    """Replace symptom names with lay or medical synonyms.

    Patients say "can't catch my breath"; notes say "dyspnea". Training on
    both keeps a model from depending on one wording.
    """
    rng = random.Random(seed)
    for term, options in _SYNONYMS.items():
        if term in text:
            text = text.replace(term, rng.choice(options))
    return text
