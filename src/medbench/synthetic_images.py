"""Synthetic X-ray, MRI and ultrasound "phantoms" with known answers.

A phantom is a simple drawing that imitates the look of a scan: a chest
X-ray is two dark lungs inside a bright body with a heart on the left, an MRI
slice is an oval skull with grey and white matter, and an ultrasound frame is
a fan of grainy speckle. A finding (nodule, lesion or cyst) is added on
purpose, so we know the true answer and its exact location (the mask).

They are far easier than real scans and are only meant to check that the
pipeline works end to end and to catch gross failures, for example a model
that always answers "normal". Real accuracy must be measured on real,
licensed datasets (see the case study).

Only the standard library is used; :func:`write_png` saves a grayscale PNG
without Pillow or NumPy.
"""

from __future__ import annotations

import math
import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Phantom",
    "chest_xray",
    "encode_png",
    "generate_phantoms",
    "mri_brain",
    "ultrasound",
    "write_png",
]

Grid = list[list[float]]
Mask = list[list[int]]


@dataclass(frozen=True)
class Phantom:
    """One synthetic image with its ground truth.

    Attributes:
        image_id: Stable identifier.
        modality: ``"xray"``, ``"mri"`` or ``"ultrasound"``.
        pixels: Rows of grayscale values 0-255.
        mask: Rows of 0/1, where 1 marks the finding.
        finding: Name of the finding, or ``"none"``.
        side: ``"left"``/``"right"`` from the *patient's* point of view, or
            ``"none"``. On a frontal X-ray the patient's left is on the image
            right.
        question: Question to ask a vision-language model.
    """

    image_id: str
    modality: str
    pixels: list[list[int]]
    mask: Mask
    finding: str
    side: str
    question: str

    @property
    def has_finding(self) -> bool:
        """Whether the image contains a finding."""
        return self.finding != "none"

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return len(self.pixels[0]), len(self.pixels)


def _blank(size: int, value: float = 0.0) -> Grid:
    return [[value] * size for _ in range(size)]


def _inside_ellipse(
    x: float, y: float, cx: float, cy: float, rx: float, ry: float
) -> bool:
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def _fill_ellipse(
    grid: Grid, cx: float, cy: float, rx: float, ry: float, value: float
) -> None:
    size = len(grid)
    for y in range(max(0, int(cy - ry)), min(size, int(cy + ry) + 1)):
        for x in range(max(0, int(cx - rx)), min(size, int(cx + rx) + 1)):
            if _inside_ellipse(x, y, cx, cy, rx, ry):
                grid[y][x] = value


def _add_blob(
    grid: Grid,
    mask: Mask,
    centre: tuple[float, float],
    radius: float,
    delta: float,
) -> None:
    """Add a soft round finding and mark it in the mask."""
    cx, cy = centre
    size = len(grid)
    for y in range(max(0, int(cy - radius)), min(size, int(cy + radius) + 1)):
        for x in range(max(0, int(cx - radius)), min(size, int(cx + radius) + 1)):
            distance = math.hypot(x - cx, y - cy)
            if distance <= radius:
                grid[y][x] += delta * (1 - (distance / radius) ** 2) ** 0.5
                mask[y][x] = 1


def _finish(grid: Grid, rng: random.Random, noise: float) -> list[list[int]]:
    return [
        [max(0, min(255, round(value + rng.gauss(0, noise)))) for value in row]
        for row in grid
    ]


def chest_xray(seed: int, size: int = 128, abnormal: bool | None = None) -> Phantom:
    """Frontal chest X-ray phantom, optionally with a lung nodule.

    Args:
        seed: Random seed.
        size: Width and height in pixels.
        abnormal: Force a nodule on (True) or off (False); random when None.

    Returns:
        The phantom.
    """
    rng = random.Random(seed)
    abnormal = rng.random() < 0.5 if abnormal is None else abnormal
    grid = _blank(size, 30)
    mask: Mask = [[0] * size for _ in range(size)]
    s = size / 128
    _fill_ellipse(grid, 64 * s, 70 * s, 54 * s, 58 * s, 170)  # body
    _fill_ellipse(grid, 42 * s, 62 * s, 18 * s, 34 * s, 55)  # patient's right lung
    _fill_ellipse(grid, 86 * s, 62 * s, 18 * s, 34 * s, 55)  # patient's left lung
    _fill_ellipse(grid, 74 * s, 82 * s, 16 * s, 13 * s, 150)  # heart, patient's left
    for rib in range(6):
        y = (34 + rib * 10) * s
        for x in range(int(24 * s), int(104 * s)):
            yy = int(y + 4 * s * math.sin(x / size * math.pi))
            if 0 <= yy < size and grid[yy][x] < 100:  # noqa: PLR2004
                grid[yy][x] += 35
    side = "none"
    if abnormal:
        side = rng.choice(("left", "right"))
        cx = (86 if side == "left" else 42) * s + rng.uniform(-6, 6) * s
        cy = rng.uniform(45, 75) * s
        _add_blob(grid, mask, (cx, cy), rng.uniform(4, 8) * s, 90)
    return Phantom(
        image_id=f"xray-{seed:05d}",
        modality="xray",
        pixels=_finish(grid, rng, 6),
        mask=mask,
        finding="nodule" if abnormal else "none",
        side=side,
        question=(
            "This is a synthetic frontal chest X-ray. Is there a round lung "
            "nodule? Answer 'yes' or 'no', and if yes, say which lung "
            "(patient's left or right)."
        ),
    )


def mri_brain(seed: int, size: int = 128, abnormal: bool | None = None) -> Phantom:
    """Axial brain MRI phantom (T2-like), optionally with a bright lesion.

    Args:
        seed: Random seed.
        size: Width and height in pixels.
        abnormal: Force a lesion on or off; random when None.

    Returns:
        The phantom. Sides follow the radiological convention: the patient's
        left is on the image right.
    """
    rng = random.Random(seed)
    abnormal = rng.random() < 0.5 if abnormal is None else abnormal
    grid = _blank(size, 5)
    mask: Mask = [[0] * size for _ in range(size)]
    s = size / 128
    _fill_ellipse(grid, 64 * s, 64 * s, 48 * s, 58 * s, 200)  # scalp / CSF
    _fill_ellipse(grid, 64 * s, 64 * s, 44 * s, 54 * s, 110)  # grey matter
    _fill_ellipse(grid, 64 * s, 64 * s, 34 * s, 44 * s, 80)  # white matter
    _fill_ellipse(grid, 57 * s, 60 * s, 4 * s, 12 * s, 220)  # ventricles
    _fill_ellipse(grid, 71 * s, 60 * s, 4 * s, 12 * s, 220)
    side = "none"
    if abnormal:
        side = rng.choice(("left", "right"))
        cx = (82 if side == "left" else 46) * s + rng.uniform(-4, 4) * s
        cy = rng.uniform(40, 88) * s
        _add_blob(grid, mask, (cx, cy), rng.uniform(4, 7) * s, 120)
    return Phantom(
        image_id=f"mri-{seed:05d}",
        modality="mri",
        pixels=_finish(grid, rng, 5),
        mask=mask,
        finding="lesion" if abnormal else "none",
        side=side,
        question=(
            "This is a synthetic axial brain MRI slice. Is there a bright focal "
            "lesion? Answer 'yes' or 'no', and if yes, say which hemisphere "
            "(patient's left or right)."
        ),
    )


def ultrasound(seed: int, size: int = 128, abnormal: bool | None = None) -> Phantom:
    """Ultrasound sector phantom with speckle, optionally with a dark cyst.

    Args:
        seed: Random seed.
        size: Width and height in pixels.
        abnormal: Force a cyst on or off; random when None.

    Returns:
        The phantom. ``side`` is the image half (left/right) for ultrasound.
    """
    rng = random.Random(seed)
    abnormal = rng.random() < 0.5 if abnormal is None else abnormal
    grid = _blank(size, 0)
    mask: Mask = [[0] * size for _ in range(size)]
    apex_x, apex_y = size / 2, -size * 0.1
    max_angle, max_depth = math.radians(38), size * 1.05
    for y in range(size):
        for x in range(size):
            angle = math.atan2(x - apex_x, y - apex_y)
            depth = math.hypot(x - apex_x, y - apex_y)
            if abs(angle) <= max_angle and depth <= max_depth:
                attenuation = 1 - 0.5 * depth / max_depth
                grid[y][x] = 120 * attenuation * rng.expovariate(1.0) ** 0.5
    side = "none"
    if abnormal:
        side = rng.choice(("left", "right"))
        cx = size * (0.62 if side == "right" else 0.38)
        cy = size * rng.uniform(0.45, 0.7)
        radius = size * rng.uniform(0.06, 0.1)
        for y in range(size):
            for x in range(size):
                if math.hypot(x - cx, y - cy) <= radius:
                    grid[y][x] = rng.uniform(0, 12)
                    mask[y][x] = 1
                elif cy + radius < y < cy + 2.5 * radius and abs(x - cx) < radius:
                    grid[y][x] *= 1.5  # posterior acoustic enhancement
    return Phantom(
        image_id=f"us-{seed:05d}",
        modality="ultrasound",
        pixels=_finish(grid, rng, 2),
        mask=mask,
        finding="cyst" if abnormal else "none",
        side=side,
        question=(
            "This is a synthetic ultrasound image. Is there a dark, round, "
            "fluid-filled cyst? Answer 'yes' or 'no', and if yes, say which "
            "half of the image (left or right)."
        ),
    )


_GENERATORS = {"xray": chest_xray, "mri": mri_brain, "ultrasound": ultrasound}


def generate_phantoms(
    modality: str, count: int, seed: int = 0, size: int = 128
) -> list[Phantom]:
    """Generate a balanced set of phantoms (half with a finding).

    Args:
        modality: ``"xray"``, ``"mri"`` or ``"ultrasound"``.
        count: Number of images.
        seed: Base seed.
        size: Image size in pixels.

    Returns:
        The phantoms.

    Raises:
        KeyError: If the modality is unknown.
    """
    if modality not in _GENERATORS:
        message = f"unknown modality {modality!r}; choose from {sorted(_GENERATORS)}"
        raise KeyError(message)
    generator = _GENERATORS[modality]
    return [generator(seed + i, size, abnormal=i % 2 == 0) for i in range(count)]


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def encode_png(pixels: list[list[int]]) -> bytes:
    """Encode 8-bit grayscale rows as PNG bytes."""
    height, width = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def write_png(path: Path, pixels: list[list[int]]) -> Path:
    """Write grayscale rows to ``path`` as a PNG file and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(pixels))
    return path
