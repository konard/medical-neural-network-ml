"""Compare upstream UniMedVL's OpenCV code across OpenCV versions (issue #19).

codes/data/transforms.py is the only upstream module that calls OpenCV
(getRotationMatrix2D, warpAffine and filter2D in motion_blur_opencv; nothing
upstream calls it). Run this under each OpenCV version and compare the printed
SHA-256 digests. ``bilinear_error`` is warpAffine's largest deviation from exact
bilinear interpolation on a float32 image: OpenCV 4.x rounds sub-pixel offsets
to 1/32 pixel (about 2e-2), OpenCV 5.0 does not (below 1e-6), which is why
rotated kernels, and so the blur at angles off the axes, differ by up to one
grey level.

    python -I experiments/issue19_unimedvl_opencv_probe.py CODE_DIR
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def bilinear_error() -> float:
    """Largest gap between warpAffine and exact bilinear interpolation."""
    image = np.random.default_rng(1).random((16, 16), dtype=np.float32)
    matrix = cv2.getRotationMatrix2D((7.5, 7.5), 37, 1)
    out = cv2.warpAffine(image, matrix, (16, 16), flags=cv2.INTER_LINEAR)
    inverse = cv2.invertAffineTransform(matrix)
    padded = np.pad(image, 1)
    ys, xs = np.mgrid[0:16, 0:16]
    sx = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
    sy = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    ax, ay = sx - x0, sy - y0
    inside = (x0 >= -1) & (x0 < 16) & (y0 >= -1) & (y0 < 16)
    x0, y0 = np.clip(x0, -1, 15) + 1, np.clip(y0, -1, 15) + 1
    exact = (1 - ay) * ((1 - ax) * padded[y0, x0] + ax * padded[y0, x0 + 1]) + ay * (
        (1 - ax) * padded[y0 + 1, x0] + ax * padded[y0 + 1, x0 + 1]
    )
    return float(np.abs(np.where(inside, exact, 0) - out).max())


def main(code_dir: str) -> None:
    path = Path(code_dir) / "codes" / "data" / "transforms.py"
    spec = importlib.util.spec_from_file_location("unimedvl_transforms", path)
    if spec is None or spec.loader is None:
        message = f"cannot load {path}"
        raise SystemExit(message)
    transforms = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent.parent))
    spec.loader.exec_module(transforms)

    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 256, (64, 48, 3), dtype=np.uint8))
    gray = Image.fromarray(rng.integers(0, 256, (40, 56), dtype=np.uint8))
    digests = {}
    for name, image in (("rgb", rgb), ("gray", gray)):
        for size, angle in ((15, 0), (15, 37), (9, 90), (21, 135.5)):
            out = transforms.motion_blur_opencv(image, kernel_size=size, angle=angle)
            key = f"{name}-k{size}-a{angle}"
            digests[key] = hashlib.sha256(np.asarray(out).tobytes()).hexdigest()[:16]
    versions = {"opencv": cv2.__version__, "numpy": np.__version__}
    print(json.dumps({**versions, "bilinear_error": bilinear_error(), **digests}))


if __name__ == "__main__":
    main(sys.argv[1])
