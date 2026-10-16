# Hardware requirements by workflow

The recommendations below are conservative planning targets, not measured
minimums. Historical raw CPU timings are in the issue #1 case study. New Mac
measurements are in [apple-silicon.md](apple-silicon.md). The hosted
workflows' peaks are measured below. The large-model workflows (MedGemma 27B,
UniMedVL, any GPU training) have not been run, so their rows remain targets.

## Hosted development workflows

| Workflow | Compute and storage | Other requirements |
| --- | --- | --- |
| `release.yml`: lint, tests, package build | Ubuntu x86_64 CPU, 4 vCPU / 16 GB RAM / 14 GB runner disk; no model weights | Python 3.13; registry access for dependencies |
| `compatibility.yml` | Runtime tests on Ubuntu/Python 3.9; full suite on arm64 `macos-15`/Python 3.12 | No model or dataset downloads; loopback HTTP for fake-server tests |
| `security.yml` | Linux CodeQL, dependency graph/review; resolved-dependency audits on Linux and arm64 macOS | Mac audit includes MLX dependencies that Linux environment markers omit; allow several GB of temporary disk for two isolated environments |
| `workflows.yml` | Ubuntu CPU; Docker for the pinned actionlint image, zizmor CLI | Read-only GitHub token for online audits |
| `links.yml` | Ubuntu CPU; network-bound | Live source access, retries, rate limits; a broken live link still fails even when an archive exists |
| `docs.yml` | Ubuntu CPU, Python/Sphinx; no model weights | Pages publishing only when repository configuration enables it |
| PyPI/GitHub/Docker publishing in `release.yml` | Package publishing needs no GPU; Docker jobs are conditional on Docker configuration | Correct PyPI trusted publisher; no Dockerfile currently supplied; never treat unconfigured credentials as a successful publish |

### Measured peaks, 2026-10-16

Every workflow ran on commit `165a710` of pull request #12. The steps under
`medbench monitor` upload a `resources-*` report and add a row to the job
summary. The table shows the largest step of each workflow. Every job, run
id and report is in `docs/case-studies/issue-1/data/ci-resources-20261016/`
in the repository.

| Workflow | Slowest job | Largest monitored step | Host used, start → peak (of total) | Largest process RSS | Disk used |
| --- | --- | --- | --- | --- | --- |
| `model-tests.yml` | CPU smoke test, 440 s | MedGemma 4B Q4_K_M tasks, 375 s | 1.85 → 5.74 GB (16.8) | 4.71 GB | model from cache |
| `compatibility.yml` | macOS / Python 3.12, 119 s | macOS tests, 98 s | 4.13 → 4.29 GB (7.5) | 0.08 GB | 0.04 GB |
| `security.yml` | macOS audit, 72 s | macOS dependency audit, 61 s | 4.32 → 4.35 GB (7.5) | 0.28 GB | 1.84 GB |
| `release.yml` | Test, 72 s | tests, 54 s | 1.10 → 1.12 GB (16.8) | 0.09 GB | 0.03 GB |
| `docs.yml` | Build docs, 28 s | Sphinx, 7 s | 1.12 → 1.20 GB (16.8) | 0.13 GB | 0.01 GB |
| `links.yml` | Check Links, 49 s | not monitored | – | – | – |
| `workflows.yml` | Audit Workflows, 15 s | not monitored | – | – | – |

These are single runs, sampled every 2–5 seconds. "Host used" counts every
process on the runner. The model download hit the cache, so a cold
3.3 GB download is not in these numbers. The publishing jobs and the
self-hosted GPU job did not run.

GitHub currently documents standard public Linux runners as 4 vCPU / 16 GB
RAM / 14 GB disk and standard arm64 macOS runners as M1 / 7 GB RAM / 14 GB
disk. A hosted Mac is not the same machine as this 18 GiB M3 Pro, and its
label does not establish that Metal model inference is available.
[GitHub runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

## Model inference

| Model/path | Initial target | Evidence and limits |
| --- | --- | --- |
| MedGemma 4B Q4_K_M, Linux CPU | 16 GB host, at least 6 GB free RAM, 8 GB free disk for setup and model | Existing measured CPU runs; 2.490 GB language file + 0.851 GB vision projector, then KV/cache/buffers |
| Small SmolVLM, native MLX | M3 Pro with at least 2.4 GB reclaimable RAM for the conservative preflight; keep one image and short output | 145 MB checkpoint; text and one real X-ray request completed locally; medical answer formats failed |
| MedGemma 4B, native MLX | 18 GiB Mac with sufficient free RAM; at least ~6.5 GB reclaimable for the 70% preflight, 8 GB model ceiling | 2.995 GB checkpoint + 1.5 GB minimum buffer allowance; not run on this busy Mac |
| MedGemma 27B Q4_K_M, llama.cpp CUDA | NVIDIA 24 GB VRAM recommended, 32 GB+ host RAM, 50 GB free disk | 16.547 GB language file + 0.858 GB projector, plus cache/buffers. GPU workflow exists, measurements outstanding |
| MedGemma 27B bf16 | 80 GB GPU is a conservative initial target | About 55 GB weights alone. A simple arithmetic fit on 64 GB does not prove runtime capacity |
| UniMedVL bf16 | Dedicated CUDA host; start with 40 GB+ VRAM, 64 GB+ host RAM, 80 GB free disk | About 29.2 GB weights. `scripts/unimedvl_server.py` serves the pinned upstream code (commit `f51aa0a`) and weights (revision `29918ce`) in an isolated Python 3.10 environment; not yet run on a GPU. 40 GB is an unverified lower target; 80 GB provides more headroom |

All model runs must record context, image resolution, output length, batch,
concurrency and runtime version. Doubling these can change capacity. The
calculator includes pinned MedGemma projector bytes, but its generic overhead
is still only a lower-order approximation. `fits` describes arithmetic
capacity, not runtime support. UniMedVL GGUF/NF4 combinations are theoretical
until a compatible implementation is demonstrated.

The manual self-hosted job uses labels `[self-hosted, linux, x64, gpu]`,
CUDA 12.8 llama.cpp binaries and a pinned runtime archive. No self-hosted
runner is registered in this repository yet. Check driver
compatibility with that CUDA release. A 16 GB T4 cannot hold the 27B Q4
working set. Never send public pull-request code directly to a persistent
GPU runner. CPU smoke runs three tasks; the manual GPU job includes all
seven, for `medgemma-4b-it`, `medgemma-27b-it` or `unimedvl`. Every download
and run is wrapped in `medbench monitor`, which writes
`medbench-reports/resources-*.json` (peak GPU and host memory, free disk,
driver and CUDA versions, wall time); the run also saves the server's cold
start time and metrics. Out-of-memory and timed-out items are recorded with
their error and the run continues. The manual GPU task budget is 90 minutes, with a 95-minute outer deadline
and a three-and-a-half-hour job limit for downloads, setup and artifact upload. This
replaces a 30-minute task budget that was shorter than the historical 27B
estimate. Actual runtime still needs measurement; incomplete coverage fails
explicitly instead of silently passing.
[Run and measure both large models](https://github.com/konard/medical-neural-network-ml/issues/6).

## Fine-tuning and training

`medbench estimate --finetune` adds up weights, trainable adapters,
gradients, AdamW state, activations and the fp32 logits of the loss. It
prints a low–high range (0.8–1.5 times the sum) for what the sum leaves out:
allocator fragmentation, loading and dequantization buffers and vision
activations. It is a formula. These are its outputs for batch 1, LoRA rank 16
and gradient checkpointing (raw output in
`docs/case-studies/issue-1/data/finetune-estimates-20261016.txt`):

| Model | Tokens | LoRA, bf16 base | QLoRA, NF4 base | Of which logits |
| --- | ---: | --- | --- | ---: |
| MedGemma 4B | 512 / 2 048 | 12.9 / 19.9 GB (10–30) | 6.5 / 13.5 GB (5.2–20) | 2.1 / 8.6 GB |
| UniMedVL | 512 / 2 048 | 32.7 / 37.1 GB (26–56) | 11.1 / 15.5 GB (8.8–23) | 1.2 / 5.0 GB |
| MedGemma 27B | 512 / 2 048 | 60.8 / 68.9 GB (49–103) | 20.1 / 28.2 GB (16–42) | 2.1 / 8.6 GB |

Full fine-tuning with AdamW is about 80 GB for 4B and 450 GB for 27B by the
same formula, so 27B needs several GPUs with sharded optimizer state. That
27B QLoRA fits one 24 GB GPU is a hypothesis: it needs short sequences or a
chunked/fused cross-entropy (Liger Kernel, Unsloth) that never holds the
logits of Gemma 3's 262 k-token vocabulary.

`scripts/measure_training_memory.py` runs real optimizer steps with the
same inputs and writes one JSON report: the loading peak, the training peak
(on CUDA also reserved minus allocated), gradient and optimizer-state bytes,
the image tokens the processor inserted, the size of a resumable checkpoint,
library versions, and measured/estimated ratios. Before loading, it compares
the high end of the estimate with free memory and refuses when less than
10% would stay free. Out-of-memory, refusal and unsupported combinations
still write a report with their `status`.

### Measured on a CPU, 2026-10-16

No GPU was available for this work, so the probe ran small models on a
6-core x86 CPU in fp32, inside a 3.1 GB memory limit, with an outer watchdog
that kills a run above 2.3 GB. Reports and commands are in
`docs/case-studies/issue-1/data/training-memory-cpu-20261016/`.

| Run | Status | Image tokens | Step peak GB | Estimate GB (range) | Peak / estimate |
| --- | --- | ---: | ---: | --- | ---: |
| SmolLM2-135M LoRA r8, 256 tokens | measured | – | 1.26 | 2.04 (1.63–3.05) | 0.62 |
| SmolLM2-135M LoRA r16, 256 tokens | measured | – | 1.30 | 2.08 (1.67–3.12) | 0.62 |
| SmolLM2-135M LoRA r64, 256 tokens | measured | – | 1.51 | 2.36 (1.89–3.54) | 0.64 |
| SmolLM2-135M LoRA r16, 1 024 tokens | measured | – | 2.00 | 2.74 (2.19–4.11) | 0.73 |
| SmolLM2-135M QLoRA | unsupported without CUDA | – | – | 1.88 (1.51–2.82) | – |
| SmolLM2-135M full, 256 tokens | killed above 2.3 GB | – | – | 3.87 (3.10–5.81) | – |
| SmolVLM-256M LoRA r16, 512 px, no splitting, 256 tokens | measured | 64 | 1.95 (load 1.97) | 2.33 (1.86–3.49) | 0.84 |
| SmolVLM-256M LoRA r16, 512 px, splitting, 1 280 tokens | killed above 2.3 GB | 1 088 (processor) | – | 3.21 (2.57–4.81) | – |

What these show:

- **The logits term is real.** From 256 to 1 024 tokens the SmolLM2 step peak
  grew by 0.68–0.70 GB in two runs, 18–19 bytes per token and vocabulary
  entry; the estimator uses 16. With Gemma 3's vocabulary that is 8.6 GB at 2 048
  tokens, more than the LoRA rank or the activations.
- **The LoRA rank matters little.** Rank 8 to 64 added 0.25 GB on SmolLM2.
- **Image tokens depend on processor settings, not only on image size.**
  SmolVLM inserted 1 088 image tokens for one 384 px image with its default
  image splitting and 64 for a 512 px image with
  `--processor-arg do_image_splitting=false`. Its processor gives the same
  counts for both sizes, because it resizes before splitting.
  Every report records the processor arguments and the measured count.
- **Measured peaks are 0.6–0.9 of the estimate.** The estimate assumes bf16
  weights plus 1.5 GB of GPU framework overhead (CUDA context, kernels);
  these CPU runs held fp32 weights and had no CUDA context. They check how
  the terms grow, not the absolute size of a GPU run.

### Still to measure on dedicated GPUs

MedGemma 4B and 27B need a CUDA host; QLoRA here means bitsandbytes NF4,
which the probe runs only on CUDA:

```bash
python scripts/measure_training_memory.py --model google/medgemma-27b-it \
    --revision <commit> --method qlora --seq-len 512 --image-size 896 \
    --out medbench-reports/train-medgemma-27b-qlora-512.json
```

Repeat at `--seq-len 2048`, with `--method lora` on a card that fits it, and
for `google/medgemma-4b-it`. UniMedVL is not a transformers model, so the
probe cannot load it; its training needs the upstream code and has only the
formula above. On the 18 GiB M3 Pro, only MedGemma 4B is worth trying, and
only when the probe's preflight accepts it (`--device mps`, no `--force`).
That is PyTorch on Metal; it says nothing about MLX LoRA.

MLX LoRA and CUDA bitsandbytes NF4 are different implementations. Do not infer
Apple support from an NVIDIA QLoRA estimate. With `--gpus` the estimator
keeps 80% of linear speed-up per job (range 60–95%) unless
`--scaling-efficiency` gives a measured value, and it assumes a fixed
utilization, so its training duration/cost is a planning calculation. GPU prices in the catalogue are on-demand list
prices read on the date in `PRICES_CHECKED`, with their source and the
lowest/highest price seen that day; pass your own with `--price-per-hour`.
No paid hardware was started by this audit.

Primary model sources: [MedGemma](https://huggingface.co/google/medgemma-27b-it),
[UniMedVL runtime](https://github.com/uni-medical/UniMedVL),
[MLX](https://ml-explore.github.io/mlx/build/html/install.html).
