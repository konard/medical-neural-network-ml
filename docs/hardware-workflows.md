# Hardware requirements by workflow

The recommendations below are conservative planning targets, not measured
minimums. Historical raw CPU timings are in the issue #1 case study. New Mac
measurements are in [apple-silicon.md](apple-silicon.md). Peak resources for
the larger workflows still require [measurement work](https://github.com/konard/medical-neural-network-ml/issues/9).

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

`medbench estimate --finetune` estimates weights, trainable adapters,
optimizer/gradients and activations. It does not allocate a real training
step, include every vision buffer, prove multi-GPU sharding, or implement a
training workflow. The historical 4B/~4.9 GB and 27B/~19.6 GB QLoRA numbers
are formula outputs, not safe minimum GPU recommendations. Start dedicated
GPU profiling with batch 1 and explicit sequence/image limits before choosing
hardware. Full Adam training stores several copies of model state; 27B can
need hundreds of GB distributed across GPUs, plus checkpoint storage.

MLX LoRA and CUDA bitsandbytes NF4 are different implementations. Do not infer
Apple support from an NVIDIA QLoRA estimate. The estimator assumes linear
speedup across GPUs and fixed utilization, so its training duration/cost is
a planning calculation. GPU prices in the catalogue are on-demand list
prices read on the date in `PRICES_CHECKED`, with their source and the
lowest/highest price seen that day; pass your own with `--price-per-hour`.
No paid hardware was started by this audit.

Primary model sources: [MedGemma](https://huggingface.co/google/medgemma-27b-it),
[UniMedVL runtime](https://github.com/uni-medical/UniMedVL),
[MLX](https://ml-explore.github.io/mlx/build/html/install.html).
