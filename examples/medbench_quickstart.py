"""medbench quick start (issue #1): data, a baseline score and a resource estimate.

Runs in a few seconds on any laptop and downloads nothing:

    python examples/medbench_quickstart.py [OUT_DIR]

1. Writes one synthetic sample of each kind (prescription, history, X-ray,
   MRI, ultrasound) as PNG files. The pictures in
   docs/case-studies/issue-1/images were made this way.
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
from medbench.resources import HARDWARE, estimate_inference
from medbench.synthetic_documents import (
    generate_history,
    generate_prescription,
    render_document,
)
from medbench.synthetic_images import chest_xray, mri_brain, ultrasound, write_png


def write_samples(out: Path) -> None:
    """Write the five sample pictures used in the case study."""
    out.mkdir(parents=True, exist_ok=True)
    for name, make in (
        ("xray", chest_xray),
        ("mri", mri_brain),
        ("ultrasound", ultrasound),
    ):
        phantom = make(3, size=256, abnormal=True)
        write_png(out / f"phantom-{name}.png", phantom.pixels)
        print(f"phantom-{name}.png: finding on the {phantom.side} side")
    prescription = generate_prescription(7)
    render_document(prescription, "handwritten", seed=7).save(
        out / "prescription-handwritten.png"
    )
    render_document(generate_history(7), "printed").save(out / "history-printed.png")
    print("prescription text the OCR model should return:")
    print(prescription.text)


def main() -> None:
    """Run the three steps."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("medbench-samples")
    write_samples(out)

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
