#!/usr/bin/env python3
"""Print the tokens tests/test_unimedvl_upstream.py generates on this stack (#21).

Run it with the Python of each UniMedVL environment to compare stacks and to
record GOLDEN_TOKENS and GOLDEN_BEST_LOGITS; it also prints each step's
top-1/top-2 logit margin in bfloat16 steps, so a golden sequence that a
rounding difference could flip is noticed:

    VENV/bin/python -I experiments/issue21_unimedvl_golden.py CODE_DIR MODEL_DIR [SEED ...]

With several seeds it tries each, which is how the test's SEED was chosen.
QK_GAIN in the environment overrides the test's, to try others.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import warnings
import zlib
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(code_dir: Path, model_dir: Path, seeds: list[int]) -> list[dict[str, object]]:
    spec = importlib.util.spec_from_file_location(
        "test_unimedvl_upstream", ROOT / "tests" / "test_unimedvl_upstream.py"
    )
    test = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(test)  # type: ignore[union-attr]
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.chdir(code_dir)
    sys.path.insert(0, str(code_dir / "codes"))
    test.load_server_module().prepare_upstream_imports()

    margins: list[float] = []
    parameters: list[str] = []
    build = test.tiny_bagel

    def tiny_bagel_with_margins(directory: Path):
        model = build(directory)
        parameters.extend(
            f"{name} {tuple(p.shape)}" for name, p in sorted(model.named_parameters())
        )

        def record(_module, _inputs, output):
            top = output[-1].float().topk(2).values
            # In units of the bfloat16 spacing at the best logit.
            spacing = 2.0 ** (math.floor(math.log2(abs(float(top[0])))) - 7)
            margins.append(float(top[0] - top[1]) / spacing)

        model.language_model.lm_head.register_forward_hook(record)
        return model

    test.tiny_bagel = tiny_bagel_with_margins
    test.QK_GAIN = float(os.environ.get("QK_GAIN", test.QK_GAIN))
    results = []
    for seed in seeds:
        test.SEED = seed
        margins.clear()
        parameters.clear()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
            tokens, best_logits = test.generate(code_dir, model_dir)
        results.append(
            {
                "seed": seed,
                "qk_gain": test.QK_GAIN,
                "python": sys.version.split()[0],
                "versions": {
                    name: version(name)
                    for name in ("torch", "transformers", "accelerate")
                },
                "parameters": len(parameters),
                "parameters_crc32": zlib.crc32("\n".join(parameters).encode()),
                "tokens": tokens,
                "best_logits": best_logits,
                "min_margin_bf16_steps": min(margins),
                "margins": list(margins),
            }
        )
    return results


if __name__ == "__main__":
    code, model, *seed_args = sys.argv[1:]
    seed_list = [int(seed) for seed in seed_args] or [0]
    print(
        json.dumps(
            main(Path(code).resolve(), Path(model).resolve(), seed_list), indent=1
        )
    )
