"""Reproduce SentencePiece's released-wheel limit without downloading model weights."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="issue25-sentencepiece-") as directory:
        for version in ("3.14", "3.15"):
            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    "--python-version",
                    version,
                    "--python-platform",
                    "x86_64-manylinux_2_28",
                    "--only-binary",
                    ":all:",
                    "--output-file",
                    str(Path(directory) / f"python-{version}.txt"),
                    "-",
                ],
                input="sentencepiece==0.2.2\n",
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "RUST_LOG": "info"},
            )
            print(f"Python {version}: exit {result.returncode}")
            print(result.stdout, result.stderr)
            assert (result.returncode == 0) == (version == "3.14")
            if version == "3.15":
                assert "sentencepiece" in result.stderr
                assert "wheel" in result.stderr


if __name__ == "__main__":
    main()
