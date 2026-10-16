"""Command line interface: ``medbench host|estimate|support|generate|sources|review|fetch|run``.

Examples::

    medbench host
    medbench estimate --model medgemma-27b-it --hardware a100-80gb --quant bf16
    medbench support --model unimedvl
    medbench generate --out data/synthetic --count 4
    medbench sources --task imaging-xray
    medbench review --task imaging-xray
    medbench fetch --out data/real
    medbench run --task diagnosis --backend naive-bayes --count 50
    medbench run --task imaging-xray --backend openai --base-url http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from medbench.backends import (
    Backend,
    NaiveBayesBackend,
    OpenAICompatibleBackend,
    TesseractBackend,
)
from medbench.benchmark import TASKS, make_task, run_task
from medbench.hardware import BudgetExceededError, ResourceBudget, detect_host
from medbench.models import MODELS, get_model
from medbench.real_data import (
    Sample,
    attribution_lines,
    default_cache_dir,
    fetch,
    load_manifest,
)
from medbench.resources import (
    BITS_PER_WEIGHT,
    HARDWARE,
    PRICES_CHECKED,
    estimate_finetune_memory,
    estimate_inference,
    estimate_training,
)
from medbench.support import RUNTIMES, default_runtime, support_matrix, support_status
from medbench.synthetic_cases import generate_cases
from medbench.validity import class_balance, load_label_review, sample_size_targets

__all__ = ["build_parser", "main"]

# Hidden size of the language model, needed for fine-tuning activations.
_HIDDEN_SIZE = {
    "medgemma-4b-it": 2560,
    "medgemma-27b-it": 5376,
    "medgemma-27b-text-it": 5376,
    "unimedvl": 3584,
    "qwen2.5-vl-7b": 3584,
}


def _print(data: object) -> None:
    sys.stdout.write(json.dumps(data, indent=2) + "\n")


def _cmd_host(_: argparse.Namespace) -> int:
    _print(detect_host().as_dict())
    return 0


def _rounded(row: dict[str, object]) -> dict[str, object]:
    def value(item: object) -> object:
        if isinstance(item, float):
            return round(item, 3)
        if isinstance(item, tuple):
            return [value(part) for part in item]
        return item

    return {key: value(item) for key, item in row.items()}


def _cmd_estimate(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    runtime = args.runtime or default_runtime(model.key, args.quant)
    hardware_keys = [args.hardware] if args.hardware else sorted(HARDWARE)
    rows = []
    for key in hardware_keys:
        estimate = estimate_inference(
            model,
            HARDWARE[key],
            quant=args.quant,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            images=args.images,
        )
        rows.append(_rounded(estimate.as_dict()))
    result: dict[str, object] = {
        "support": [
            {"task": status.task, "status": status.status, "reason": status.reason}
            for status in (
                support_status(model.key, runtime, args.quant, task) for task in TASKS
            )
        ],
        "runtime": runtime,
        "note": (
            "fits/capacity compare estimated bytes with device memory only; "
            "'support' says whether a pinned runtime path exists and was "
            "measured. Ranges are hypotheses until measured."
        ),
        "inference": rows,
    }
    if args.finetune:
        result["finetune"] = [
            _rounded(
                estimate_finetune_memory(
                    model, method, _HIDDEN_SIZE[model.key]
                ).as_dict()
            )
            for method in ("full", "lora", "qlora")
        ]
    if args.train_tokens:
        result["training"] = [
            {
                "hardware": key,
                "method": method,
                **_rounded(
                    estimate_training(
                        model.total_params,
                        args.train_tokens,
                        HARDWARE[key],
                        method=method,
                        price_per_hour=args.price_per_hour,
                    ).as_dict()
                ),
            }
            for key in hardware_keys
            if HARDWARE[key].is_gpu
            for method in ("full", "lora")
        ]
        result["prices_checked"] = PRICES_CHECKED
    _print(result)
    return 0


def _cmd_support(args: argparse.Namespace) -> int:
    models = (args.model,) if args.model else None
    rows = [
        status.as_dict()
        for status in support_matrix(models)
        if args.runtime is None or status.runtime == args.runtime
    ]
    _print(rows)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cases = [case.as_dict() for case in generate_cases(args.count, args.seed)]
    (out / "diagnosis_cases.json").write_text(
        json.dumps(cases, indent=2) + "\n", encoding="utf-8"
    )
    _print({"written": str(out), "cases": len(cases)})
    return 0


def _selected_samples(task: str | None) -> list[Sample]:
    return [s for s in load_manifest() if task is None or s.task == task]


def _cmd_sources(args: argparse.Namespace) -> int:
    _print(
        [
            {
                "id": sample.sample_id,
                "task": sample.task,
                "license": sample.license,
                "author": sample.author,
                "source": sample.source_url,
                "truth": sample.truth,
            }
            for sample in _selected_samples(args.task)
        ]
    )
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    review = load_label_review()
    samples = [s for s in _selected_samples(args.task) if s.sample_id in review.entries]
    _print(
        {
            "reviewer": review.reviewer,
            "clinician_reviewed": review.clinician_reviewed,
            "method": review.method,
            "reviewed_on": review.reviewed_on,
            "class_balance": class_balance(samples),
            "sample_size_targets": sample_size_targets(),
            "samples": [
                {
                    **review.entries[sample.sample_id].__dict__,
                    "finding": sample.truth.get("finding"),
                }
                for sample in samples
            ],
        }
    )
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    samples = _selected_samples(args.task)
    target: Path = args.out or default_cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        fetch(sample, target)
    credits = "".join(f"- {line}\n" for line in attribution_lines(samples))
    (target / "ATTRIBUTION.md").write_text(
        "# Sources and licenses\n\n" + credits, encoding="utf-8"
    )
    _print({"fetched": len(samples), "directory": str(target)})
    return 0


def _make_backend(args: argparse.Namespace, budget: ResourceBudget) -> Backend:
    if args.backend == "naive-bayes":
        return NaiveBayesBackend()
    if args.backend == "tesseract":
        return TesseractBackend()
    if args.backend == "mlx":
        from medbench.mlx_backend import MlxBackend

        if args.model_path is None:
            msg = "--backend mlx requires --model-path to a local snapshot"
            raise ValueError(msg)
        return MlxBackend(args.model_path, budget, args.memory_limit_gb)
    return OpenAICompatibleBackend(
        args.base_url,
        model=args.model_name,
        api_key=os.environ.get("MEDBENCH_API_KEY"),
        timeout_s=args.request_timeout,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    budget = ResourceBudget(
        max_seconds=args.time_budget, max_threads=4 if args.backend == "mlx" else None
    )
    task = make_task(
        args.task, args.count, args.seed, cache_dir=args.data_dir, split=args.split
    )
    report = run_task(
        task, _make_backend(args, budget), budget, max_tokens=args.max_tokens
    )
    if args.out:
        report.write(args.out)
    summary = report.as_dict()
    summary.pop("results")
    _print(summary)
    return int(report.stopped_early or report.items_run == 0)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="medbench", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    host = commands.add_parser("host", help="show CPUs, memory and GPUs")
    host.set_defaults(func=_cmd_host)

    estimate = commands.add_parser("estimate", help="estimate memory and speed")
    estimate.add_argument("--model", choices=sorted(MODELS), required=True)
    estimate.add_argument("--hardware", choices=sorted(HARDWARE))
    estimate.add_argument("--quant", choices=sorted(BITS_PER_WEIGHT), default="Q4_K_M")
    estimate.add_argument("--prompt-tokens", type=int, default=512)
    estimate.add_argument("--output-tokens", type=int, default=256)
    estimate.add_argument("--images", type=int, default=0)
    estimate.add_argument(
        "--runtime",
        choices=sorted(RUNTIMES),
        help="runtime to check support for (default: implied by --quant)",
    )
    estimate.add_argument(
        "--finetune", action="store_true", help="add fine-tuning memory"
    )
    estimate.add_argument(
        "--train-tokens",
        type=float,
        help="add training time and cost for this many tokens (GPUs only)",
    )
    estimate.add_argument(
        "--price-per-hour",
        type=float,
        help="your own USD per GPU-hour instead of the dated example prices",
    )
    estimate.set_defaults(func=_cmd_estimate)

    support = commands.add_parser(
        "support",
        help="model x runtime x weight format x task support matrix",
    )
    support.add_argument("--model", choices=sorted(MODELS))
    support.add_argument("--runtime", choices=sorted(RUNTIMES))
    support.set_defaults(func=_cmd_support)

    generate = commands.add_parser("generate", help="write synthetic diagnosis cases")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--count", type=int, default=10)
    generate.add_argument("--seed", type=int, default=0)
    generate.set_defaults(func=_cmd_generate)

    sources = commands.add_parser(
        "sources", help="list the real files, their licenses and answers"
    )
    sources.add_argument("--task", choices=TASKS)
    sources.set_defaults(func=_cmd_sources)

    review = commands.add_parser(
        "review",
        help="label review (not clinician-reviewed), balance and sample-size targets",
    )
    review.add_argument("--task", choices=TASKS)
    review.set_defaults(func=_cmd_review)

    fetch_cmd = commands.add_parser(
        "fetch", help="download and verify the real files, write ATTRIBUTION.md"
    )
    fetch_cmd.add_argument("--task", choices=TASKS)
    fetch_cmd.add_argument(
        "--out", type=Path, help="directory (default: $MEDBENCH_DATA_DIR or cache)"
    )
    fetch_cmd.set_defaults(func=_cmd_fetch)

    _add_run_parser(commands)
    return parser


def _add_run_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run = commands.add_parser("run", help="score a backend on a task")
    run.add_argument("--task", choices=TASKS, required=True)
    run.add_argument(
        "--backend",
        choices=("naive-bayes", "tesseract", "openai", "mlx"),
        default="naive-bayes",
        help="tesseract is the classic OCR baseline for the ocr-* tasks",
    )
    run.add_argument("--model-path", type=Path, help="local MLX-VLM snapshot")
    run.add_argument(
        "--memory-limit-gb", type=float, default=8.0, help="MLX ceiling in decimal GB"
    )
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run.add_argument("--model-name", default="default")
    run.add_argument("--count", type=int, default=10)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        help="DDXPlus patient split (tune on dev, report on test)",
    )
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--time-budget", type=float, default=600.0, help="seconds")
    run.add_argument("--request-timeout", type=float, default=600.0, help="seconds")
    run.add_argument("--out", type=Path, help="write the full JSON report here")
    run.add_argument(
        "--data-dir",
        type=Path,
        help="cache for real files (default: $MEDBENCH_DATA_DIR)",
    )
    run.set_defaults(func=_cmd_run)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (BudgetExceededError, ValueError, FileNotFoundError) as error:
        sys.stderr.write(f"medbench: {error}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
