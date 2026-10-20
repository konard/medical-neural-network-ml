"""Entry point: ``python -m scripts.cloud_gpu``."""

from __future__ import annotations

import sys

from scripts.cloud_gpu.cli import main

if __name__ == "__main__":
    sys.exit(main())
