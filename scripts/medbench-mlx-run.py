#!/usr/bin/env python3
"""Run a short MLX smoke test or benchmark with a hard process deadline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def smoke() -> int:
    """Verify Metal without a model download, using at most 256 MiB."""
    import mlx.core as mx

    if not mx.metal.is_available():
        msg = "Metal GPU is unavailable"
        raise RuntimeError(msg)
    mx.set_memory_limit(256 * 1024**2)
    mx.set_cache_limit(32 * 1024**2)
    mx.set_default_device(mx.gpu)
    matrix = mx.ones((64, 64))
    result = matrix @ matrix
    mx.eval(result)
    if result[0, 0].item() != 64:
        msg = "Metal matrix check failed"
        raise RuntimeError(msg)
    print(
        json.dumps(
            {
                "mlx_version": version("mlx"),
                "mlx_vlm_version": version("mlx-vlm"),
                "device": mx.device_info(),
                "matrix_check": "passed",
                "peak_memory_bytes": mx.get_peak_memory(),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    """Launch the worker at low priority and kill it at the deadline."""
    if sys.argv[1:] == ["--worker-smoke"]:
        return smoke()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="no model or network required"
    )
    parser.add_argument(
        "--timeout", type=float, default=180, help="hard deadline, seconds"
    )
    args, extra = parser.parse_known_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.smoke and extra:
        parser.error("--smoke accepts only --timeout")
    command = [sys.executable]
    if args.smoke:
        command += [str(Path(__file__).resolve()), "--worker-smoke"]
    else:
        command += ["-m", "medbench.cli", "run", "--backend", "mlx", *extra]
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "4"
    try:
        return subprocess.run(
            ["nice", "-n", "10", *command],
            env=env,
            timeout=args.timeout,
            check=False,
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"MLX worker stopped at the {args.timeout:g}s deadline", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
