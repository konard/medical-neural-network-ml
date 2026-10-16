"""Command line interface: ``medbench host|estimate|generate|sources|fetch|run``.

Examples::

    medbench host
    medbench estimate --model medgemma-27b-it --hardware a100-80gb --quant bf16
    medbench generate --out data/synthetic --count 4
    medbench sources --task imaging-xray
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

from medbench.backends import Backend, NaiveBayesBackend, OpenAICompatibleBackend
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
    estimate_finetune_memory,
    estimate_inference,
)
from medbench.synthetic_cases import generate_cases

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


def _cmd_estimate(args: argparse.Namespace) -> int:
    model = get_model(args.model)
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
        rows.append(
            {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in estimate.as_dict().items()
            }
        )
    result: dict[str, object] = {"inference": rows}
    if args.finetune:
        result["finetune"] = [
            {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in estimate_finetune_memory(
                    model, method, _HIDDEN_SIZE[model.key]
                )
                .as_dict()
                .items()
            }
            for method in ("full", "lora", "qlora")
        ]
    _print(result)
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
    task = make_task(args.task, args.count, args.seed, cache_dir=args.data_dir)
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
        "--finetune", action="store_true", help="add fine-tuning memory"
    )
    estimate.set_defaults(func=_cmd_estimate)

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

    fetch_cmd = commands.add_parser(
        "fetch", help="download and verify the real files, write ATTRIBUTION.md"
    )
    fetch_cmd.add_argument("--task", choices=TASKS)
    fetch_cmd.add_argument(
        "--out", type=Path, help="directory (default: $MEDBENCH_DATA_DIR or cache)"
    )
    fetch_cmd.set_defaults(func=_cmd_fetch)

    run = commands.add_parser("run", help="score a backend on a task")
    run.add_argument("--task", choices=TASKS, required=True)
    run.add_argument(
        "--backend", choices=("naive-bayes", "openai", "mlx"), default="naive-bayes"
    )
    run.add_argument("--model-path", type=Path, help="local MLX-VLM snapshot")
    run.add_argument(
        "--memory-limit-gb", type=float, default=8.0, help="MLX ceiling in decimal GB"
    )
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run.add_argument("--model-name", default="default")
    run.add_argument("--count", type=int, default=10)
    run.add_argument("--seed", type=int, default=0)
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
    return parser


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
