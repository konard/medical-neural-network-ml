"""medbench quick start (issue #1): real data, a baseline score and a resource estimate.

    python examples/medbench_quickstart.py [OUT_DIR]

1. Downloads the real, openly licensed X-rays, MRI and ultrasound images,
   prescriptions and case histories listed in
   ``src/medbench/data/real_samples.json`` into OUT_DIR, checks each SHA-256
   and writes ``ATTRIBUTION.md`` with the author and license of every file.
   Nothing is stored in the repository.
2. Scores the naive Bayes reference on the diagnosis task. The patients are
   made from known probabilities, so this ranking is the best possible answer
   and a real model should come close to it.
3. Prints how much memory MedGemma 27B needs and how fast it would run on a
   GitHub-hosted CPU runner and on an NVIDIA L4.
"""

from __future__ import annotations

import sys
from pathlib import Path

from medbench.backends import NaiveBayesBackend
from medbench.benchmark import make_task, run_task
from medbench.models import get_model
from medbench.real_data import attribution_lines, fetch, load_manifest
from medbench.resources import HARDWARE, estimate_inference


def download_samples(out: Path) -> None:
    """Fetch every real sample and credit its author."""
    samples = load_manifest()
    for sample in samples:
        path = fetch(sample, out)
        print(f"{path.name}: {sample.task}, {sample.license}, {sample.truth}")
    lines = attribution_lines(samples)
    credits = "# Sources and licenses\n\n" + "".join(f"- {x}\n" for x in lines)
    (out / "ATTRIBUTION.md").write_text(credits, encoding="utf-8")


def main() -> None:
    """Run the three steps."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("medbench-samples")
    download_samples(out)

    report = run_task(make_task("diagnosis", count=50), NaiveBayesBackend())
    print(
        f"naive Bayes reference: top-1 {report.metrics['top1']:.2f}, "
        f"top-5 {report.metrics['top5']:.2f} on {report.items_run} patients"
    )

    for hardware in ("github-ubuntu-4cpu", "l4-24gb"):
        est = estimate_inference(
            get_model("medgemma-27b-it"), HARDWARE[hardware], quant="Q4_K_M", images=1
        )
        print(
            f"medgemma-27b-it Q4_K_M on {hardware}: {est.total_memory_gb:.1f} GB, "
            f"fits={est.fits}, {est.decode_tokens_per_s:.1f} tokens/s, "
            f"{est.seconds_per_request:.0f} s per request"
        )


if __name__ == "__main__":
    main()
