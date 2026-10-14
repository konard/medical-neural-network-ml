# Hardware, CI/CD and resource research notes (issue #1)

Research date: 2026-10-14. Scope: what hardware is needed to try
`google/medgemma-27b-it`, `google/medgemma-4b-it` (and 1.5), and
`General-Medical-AI/UniMedVL`, how fast CPU inference can be, whether it fits
in GitHub-hosted CI, and what a GPU (self-hosted runner) path costs.

Conventions:

- "Verified" facts come with a source in the "Sources" list at the end (numbers in square brackets, e.g. [S1]).
- Facts I could not confirm from a primary source are marked **(unverified)**.
- Numbers I calculated myself are marked **(estimate)** and show the working.
- No model weights were downloaded and no heavy workloads were run. Model metadata and file sizes come from the public Hugging Face API (`/api/models/<id>` and `/api/models/<id>/tree/main`), queried on 2026-10-14.

The machine this research ran on has 6 CPU cores, about 11 GiB RAM and no NVIDIA GPU (`nproc`, `free -g`, `nvidia-smi` missing). That is a useful reference for a "small dev machine".

---

## 1. GitHub-hosted runners (2026)

### 1.1 Standard runners [S1]

| Runner (label) | Public repo: vCPU / RAM / SSD | Private repo: vCPU / RAM / SSD |
|---|---|---|
| Linux x64 (`ubuntu-latest`, `ubuntu-24.04`) | 4 / 16 GB / 14 GB | 2 / 8 GB / 14 GB |
| Linux arm64 (`ubuntu-24.04-arm`) | 4 / 16 GB / 14 GB | 2 / 8 GB / 14 GB |
| Windows x64 | 4 / 16 GB / 14 GB | 2 / 8 GB / 14 GB |
| macOS arm64 (M1) | 3 / 7 GB / 14 GB | 3 / 7 GB / 14 GB |
| macOS Intel | 4 / 14 GB / 14 GB | 4 / 14 GB / 14 GB |

- The 14 GB SSD figure is what GitHub guarantees. The real free space on the image can differ **(unverified)**. Plan around 14 GB.
- There is also a 1-vCPU `ubuntu-slim` runner with a 15-minute job timeout [S1]. It is not useful for models.
- GitHub does not publish the memory bandwidth or exact CPU model of hosted runners **(unverified: they are usually AMD EPYC or Intel Xeon VMs)**. That matters, because CPU decoding speed depends on bandwidth. Measured in [run 38550186566](https://github.com/konard/medical-neural-network-ml/actions/runs/38550186566): MedGemma 4B Q4_K_M wrote 7.8 tokens/s with 3 threads, which means about 22 GB/s of useful memory bandwidth ([measurements.md §9](../measurements.md#9-a-real-run-on-a-github-hosted-cpu-runner)).

### 1.2 Limits [S2]

| Limit | Value |
|---|---|
| Job time, GitHub-hosted | 6 hours (job fails at the limit) |
| Job time, self-hosted | 5 days |
| Workflow run time | 35 days (includes waiting and approvals) |
| Concurrent standard jobs | 20 Free, 40 Pro, 60 Team, 500 Enterprise |
| Concurrent GPU jobs (larger runners) | 100 |
| Actions cache | 10 GB per repo free, 7-day retention. Since Nov 2025 you can pay to go above 10 GB [S3] |

### 1.3 Larger runners and GPU runners [S4][S5][S6]

- GPU runner spec: 1x **NVIDIA Tesla T4**, **16 GB VRAM**, 4 vCPU, 28 GB RAM, 176 GB SSD [S5].
- Larger runners (including GPU) are available only on **GitHub Team or Enterprise Cloud** organisation plans. They need a payment method, and included free minutes do not apply to them, even for public repos [S6][S4].
- Prices as of 2026 (USD per minute) [S4]:

| Runner | $/min | $/hour |
|---|---|---|
| Linux 2-core x64 (standard) | 0.006 | 0.36 |
| Linux 2-core arm64 (standard) | 0.005 | 0.30 |
| Linux 4-core x64 | 0.012 | 0.72 |
| Linux 8-core x64 | 0.022 | 1.32 |
| Linux 16-core x64 | 0.042 | 2.52 |
| Linux 32-core x64 | 0.082 | 4.92 |
| Linux 64-core x64 | 0.162 | 9.72 |
| Linux 4-core arm64 | 0.008 | 0.48 |
| Linux 64-core arm64 | 0.098 | 5.88 |
| **Linux 4-core GPU (T4)** | **0.052** | **3.12** |
| Windows 4-core GPU (T4) | 0.102 | 6.12 |

- Standard runners are free for public repositories.
- Self-hosted runner fee: GitHub announced a $0.002/min platform charge for self-hosted runners on private repos from 1 March 2026. In Dec 2025 it **postponed** that charge. It has not been cancelled [S7]. Public repos were never in scope.

Worked cost example: a 20-minute GPU test job on the hosted T4 costs 20 x $0.052 = **$1.04**. Running it on every push, 10 times a day, costs about $312/month. A self-hosted GPU runner costs nothing per minute (only power and hardware).

Important T4 limits for this project:

- T4 is Turing. It has FP16 tensor cores (65 TFLOPS FP16, 130 INT8 TOPS, 320 GB/s, 70 W) [S8], but no native bf16.
- Unsloth notes that float16 on T4, RTX 20xx and V100 can make Gemma 3 activations overflow to infinity [S9]. Run MedGemma in bf16 emulation, fp32 activations, or a GGUF quantisation through llama.cpp.
- 16 GB VRAM holds MedGemma 4B in bf16 (about 8.6 GB) or 27B at about 4 bits (about 16.5 GB for Q4_K_M plus KV cache). The 27B case is right at the limit: IQ3/Q3 quants (12-13.7 GB) fit, Q4_K_M probably does not once KV cache and mmproj are added **(estimate)**.

---

## 2. Self-hosted GitHub Actions runner on Linux with an NVIDIA GPU

### 2.1 Install the runner [S10][S11][S12]

The latest runner release on 2026-10-14 is **v2.338.0** (the linux-x64 tarball is about 228 MB) [S13].

```bash
# as a dedicated, non-root user, e.g. "ghrunner"
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64-2.338.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.338.0/actions-runner-linux-x64-2.338.0.tar.gz
# check the SHA-256 shown on the "New self-hosted runner" settings page
tar xzf ./actions-runner-linux-x64-2.338.0.tar.gz

# The token comes from Settings -> Actions -> Runners -> New self-hosted runner. It expires after 1 hour.
./config.sh --url https://github.com/<owner>/<repo> --token <TOKEN> \
  --labels gpu,cuda,nvidia --name gpu-box-1 --unattended
#   optional: --ephemeral (one job, then the runner deregisters), --no-default-labels, --disableupdate

# run as a systemd service
sudo ./svc.sh install ghrunner   # optional user argument
sudo ./svc.sh start
sudo ./svc.sh status
# sudo ./svc.sh stop ; sudo ./svc.sh uninstall
```

- Labels can only be set at configuration time with `--labels`. To change them later, use the UI or API. Labels are case-insensitive [S11].
- `--ephemeral`, `--disableupdate` and `--no-default-labels` are real `config.sh` flags (from the runner source, `CommandSettings.cs`) [S14].
- Self-hosted runners get the default labels `self-hosted`, `linux`, `x64` **(the `self-hosted` default comes from the runner and docs I know of; not re-verified in the fetched page)**.

Workflow targeting:

```yaml
jobs:
  gpu-tests:
    runs-on: [self-hosted, linux, x64, gpu]
    timeout-minutes: 120
```

### 2.2 NVIDIA driver, CUDA, container toolkit [S15][S16]

1. Install the NVIDIA driver from the distribution packages. On Ubuntu, that is `sudo ubuntu-drivers install` or a specific `nvidia-driver-5xx` package **(the Ubuntu-specific command is unverified here; NVIDIA's guide just says "use your distro package manager")**. Check it with `nvidia-smi`.
2. You do not need the CUDA toolkit on the host if you use containers (for example `nvidia/cuda:*` or `vllm/vllm-openai` images). You do need it for native builds such as llama.cpp `-DGGML_CUDA=ON`.
3. Install nvidia-container-toolkit (apt):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# add the repo list file per NVIDIA guide, then:
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1
sudo apt-get install -y nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# smoke test (from NVIDIA docs)
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### 2.3 Security guidance [S10][S17]

- GitHub: "self-hosted runners should almost never be used for public repositories", "because any user can open pull requests against the repository and compromise the environment" [S17]. The setup page recommends using self-hosted runners only with private repositories [S10].
- This repository is public (konard/medical-neural-network-ml), so the GPU workflow should:
  - trigger only on `workflow_dispatch`, `schedule`, or `push` to `main`. Do not use `pull_request` from forks, and never use `pull_request_target` together with checking out PR code.
  - optionally run on PRs only after a maintainer adds a label (`if: contains(github.event.pull_request.labels.*.name, 'run-gpu')`), together with the repo setting "Require approval for all outside collaborators".
  - set `permissions: contents: read` and pass no secrets.
  - use **ephemeral / just-in-time runners** (`--ephemeral` or the JIT REST API). They "perform at most one job before being automatically removed" [S17]. Combine them with a clean container or VM per job.
  - put the runner in its own runner group, limited to this repository [S17].
  - run the runner under a dedicated unprivileged user. Do not use root or the developer's own account, and do not run it on the developer's main desktop session.

### 2.4 actionlint and custom labels [S18]

actionlint reports unknown `runs-on` labels unless they are declared. Create `.github/actionlint.yaml`:

```yaml
self-hosted-runner:
  # Labels of self-hosted runners (glob patterns allowed)
  labels:
    - gpu
    - cuda
    - nvidia
```

---

## 3. CPU inference performance

### 3.1 The rule: decoding is memory-bandwidth-bound

At batch size 1, generating each new token reads (almost) all of the weights from RAM once. So:

```
tokens/s (decode) <= effective memory bandwidth (GB/s) / bytes of weights read per token (GB)
```

Prompt processing (prefill) is the opposite case. It is compute-bound (matrix-matrix work, about 2·N FLOPs per token), so it is usually 5-50x faster per token than generation [S19][S20].

Evidence from the llama.cpp `llama-bench` README (CPU backend, llama 7B Q4_0, 3.56 GiB) [S21]:

| threads | pp64 t/s | tg16 t/s |
|---|---|---|
| 1 | 6.17 | 4.05 |
| 4 | 23.18 | 12.22 |
| 8 | 32.29 | 16.71 |
| 16 | 33.52 | 15.32 |
| 32 | 59.00 | 16.41 |

Generation levels off at about 16 t/s after 8 threads, which shows the memory-bound behaviour. 3.56 GiB x 16.7 t/s = about 64 GB/s effective bandwidth. Prefill keeps growing with more threads, which shows the compute-bound behaviour.

A measured 2026 data point from OpenBenchmarking: Ryzen 9 9950X3D (16C/32T, 256 GB RAM, memory speed not listed), llama.cpp b8933 CPU BLAS, Mistral-7B Q8_0 (about 7.7 GB). Result: **tg128 = 5.76 t/s, pp512 = 94 t/s** [S22]. 7.7 GB x 5.76 = about 44 GB/s effective. Four DIMMs at reduced speed probably explain the low figure **(unverified)**.

### 3.2 Memory bandwidth reference

| Memory | Theoretical peak | Typical effective (about 70-85%) |
|---|---|---|
| DDR4-3200 dual channel | 51.2 GB/s | ~40 GB/s |
| DDR5-5600 dual channel | 89.6 GB/s | ~70-76 GB/s |
| DDR5-6400 dual channel | 102.4 GB/s | ~80 GB/s |
| 8-channel DDR5-4800 server (EPYC/Xeon) | 307 GB/s | ~200-250 GB/s **(estimate)** |
| Apple M-series Max/Ultra | 400-800 GB/s | (vendor numbers, unverified here) |

Peak = MT/s x 16 bytes (two 64-bit channels) [S23]. The "effective ~76 GB/s for DDR5-5600" figure comes from a community measurement [S23].

### 3.3 GGUF quantisation sizes

Bits per weight from the official llama.cpp quantize README (Llama-3.1-8B) [S24]:

| Type | bits/weight | Llama-3.1-8B size |
|---|---|---|
| Q2_K | 3.16 | 2.95 GiB |
| Q3_K_M | 4.00 | 3.74 GiB |
| Q4_K_S | 4.67 | 4.36 GiB |
| **Q4_K_M** | **4.89** | 4.58 GiB |
| Q5_K_M | 5.70 | 5.33 GiB |
| Q6_K | 6.56 | 6.14 GiB |
| **Q8_0** | **8.50** | 7.95 GiB |
| F16 | 16.0 | 14.96 GiB |

Actual MedGemma GGUF file sizes from the Hugging Face API, 2026-10-14 [S25][S26][S27]. GB = 10^9 bytes.

| File | medgemma-4b-it (unsloth) | medgemma-27b-it (unsloth) |
|---|---|---|
| UD-IQ1_S | 1.18 GB | 6.51 GB |
| Q2_K | 1.73 GB | 10.50 GB |
| Q3_K_M | 2.10 GB | 13.44 GB |
| IQ4_XS | 2.26 GB | 14.77 GB |
| Q4_0 | 2.37 GB | 15.62 GB |
| **Q4_K_M** | **2.49 GB** | **16.55 GB** |
| UD-Q4_K_XL | 2.54 GB | 16.83 GB |
| Q5_K_M | 2.83 GB | 19.27 GB |
| Q6_K | 3.19 GB | 22.17 GB |
| **Q8_0** | **4.13 GB** | **28.71 GB** |
| BF16 | 7.77 GB | (not listed as single file) |
| mmproj-F16 (vision tower + projector) | 0.85 GB | 0.86 GB |
| mmproj-F32 | 1.67 GB | 1.68 GB |

- Check: 16.55 GB x 8 / ~27.0 B LM params = **4.9 bpw** for 27B Q4_K_M, and 28.71 x 8 / 27.0 = **8.5 bpw** for Q8_0. Both match the README **(estimate: LM param count = 27.43 B total minus ~0.42 B vision)**.
- `unsloth/gemma-3-27b-it-GGUF` has byte-identical sizes for every quant, as expected for the same architecture [S27].
- `unsloth/medgemma-1.5-4b-it-GGUF` also exists: Q4_K_M 2.49 GB, Q8_0 4.13 GB, mmproj-F16 0.85 GB [S28].

Access (gating) status from the HF API, 2026-10-14:

| Repo | gated | Notes |
|---|---|---|
| `google/medgemma-27b-it` | **yes** (`auto`) | Log in and accept the Health AI Developer Foundations terms. Approval is immediate [S29]. 27.43 B BF16 params (HF metadata shows "29B" including extra tensors). |
| `google/medgemma-4b-it` | **yes** (`auto`) | 4.30 B params. |
| `google/medgemma-1.5-4b-it` | **yes** (`auto`) | Released Jan 2026. |
| `unsloth/medgemma-27b-it-GGUF` | **no** | Downloadable anonymously (the HDAI terms still apply to use). |
| `unsloth/medgemma-4b-it-GGUF` | **no** | |
| `unsloth/medgemma-27b-it` (safetensors) | no | config.json readable without a token. |
| `General-Medical-AI/UniMedVL` | **no** | Apache-2.0, "14B". Weights: `ema_bf16.safetensors` 29.2 GB + `ae.safetensors` 0.34 GB, about 29.6 GB total. LLM config is Qwen2-style (28 layers, hidden 3584, 4 KV heads), BAGEL-inspired MoT. No GGUF or CPU version is mentioned [S30]. |

So a CI job can use the unsloth GGUF mirrors without an `HF_TOKEN`. The original Google repos need a token from an account that accepted the terms. Keep that token as a secret, and never expose it to fork PRs.

### 3.4 llama.cpp and Ollama support

- llama.cpp supports multimodal input through `libmtmd` with `llama-mtmd-cli` and `llama-server`. Gemma 3 4B/12B/27B are in the official supported list (`-hf ggml-org/gemma-3-27b-it-GGUF`). With `-m`, pass the projector as `--mmproj file.gguf`. `--no-mmproj-offload` keeps the projector on the CPU [S31].
- MedGemma uses the Gemma 3 architecture, so `llama-mtmd-cli -m medgemma-4b-it-Q4_K_M.gguf --mmproj mmproj-F16.gguf --image x.png -p "..."` is expected to work **(not run here; it follows from the shared architecture and unsloth shipping mmproj files)**.
- Ollama library: `medgemma` (tags `4b`, `27b`, `4b-it-q4_*`, `4b-it-q8_0`, `4b-it-bf16`, `27b-it-q4_*`, `27b-it-q8_0`, `27b-it-bf16`; vision supported; page shows 3.3 GB for 4b and 17 GB for 27b) and `medgemma1.5` (4b) [S32].

### 3.5 Vision encoder cost on CPU (estimate)

MedGemma, like Gemma 3, uses a ~400 M-parameter SigLIP encoder. Its config: 27 layers, hidden 1152, image 896x896, patch 14. That gives (896/14)^2 = **4096 patches**, which are pooled to **256 image tokens** [S29][config.json].

- Linear layers: about 2 x 0.41 B params x 4096 patches = about 3.4 TFLOP.
- Attention: 4 x 27 layers x 4096^2 x 1152 = about 2.1 TFLOP.
- **Total: about 5.5 TFLOP per image**, before the LLM even sees it.
- Then LLM prefill of about 256 image + 100 text tokens:
  - 4B: 2 x 3.9 B x 356 = **about 2.8 TFLOP**
  - 27B: 2 x 27 B x 356 = **about 19 TFLOP**

| CPU (assumed effective FP throughput) | 4B: one image (~8.3 TFLOP) | 27B: one image (~25 TFLOP) |
|---|---|---|
| 4-vCPU CI VM (~0.15-0.3 TFLOPS) | ~30-55 s | ~85-170 s |
| 8-16 core desktop AVX2/AVX-512 (~0.5-1 TFLOPS) | ~8-17 s | ~25-50 s |

These are **(estimate)**: effective CPU FLOPS depend heavily on the kernel (F16 mmproj vs quantised) and on AVX-512/AMX support. Measure with `llama-mtmd-cli`, which prints image encode time.

### 3.6 Predicted CPU decode speed for MedGemma (estimate)

Bytes read per token is about the GGUF file size. The vision tower is not used during decoding, and KV-cache reads are small at short context.

| Model / quant | Size | DDR4-3200 2ch (~40 GB/s eff.) | DDR5-5600 2ch (~72 GB/s eff.) | 4-vCPU CI VM (assume ~20-40 GB/s, unverified) |
|---|---|---|---|---|
| 4B Q4_K_M | 2.49 GB | ~16 t/s (ceiling) | ~29 t/s | ~8-16 t/s |
| 4B Q8_0 | 4.13 GB | ~10 t/s | ~17 t/s | ~5-10 t/s |
| 4B BF16 (transformers/PyTorch) | 8.6 GB | ~4.6 t/s | ~8 t/s | ~2-4 t/s, and PyTorch CPU is usually slower than llama.cpp **(unverified)** |
| 27B Q4_K_M | 16.55 GB | ~2.4 t/s | ~4.3 t/s | does not fit in 16 GB RAM |
| 27B Q8_0 | 28.71 GB | ~1.4 t/s | ~2.5 t/s | does not fit |
| 27B BF16 | ~54.9 GB | ~0.7 t/s | ~1.3 t/s | does not fit |
| UniMedVL BF16 (14 B, ~29.6 GB) | 29.6 GB | ~1.3 t/s | ~2.4 t/s | does not fit, and no GGUF exists |

Real numbers usually land at 50-80% of the ceiling. Third-party calculators predict about 3.3 t/s for Gemma 3 27B Q4_K_M on CPU, and one report puts Mistral-Small-24B Q4_K_M at about 4 t/s on an i7-13700K with DDR5-5600 [S33]. Both agree with the table. I found no measured MedGemma-specific CPU benchmark, so we made one: on a GitHub-hosted runner MedGemma 4B Q4_K_M wrote **7.8 t/s** with 3 threads, inside the 8-16 t/s range above ([measurements.md §9](../measurements.md#9-a-real-run-on-a-github-hosted-cpu-runner)).

### 3.7 What fits in GitHub-hosted CI

RAM needed is weights + mmproj + KV cache + runtime overhead (about 0.5-1 GB).

- **Public repo `ubuntu-latest` (4 vCPU / 16 GB / 14 GB disk)**:
  - MedGemma 4B Q4_K_M + mmproj F16 needs 2.49 + 0.85 + 0.3 (KV at 4K, see 4.2) + ~1 = **~4.6 GB RAM**. This fits easily.
  - Download about 3.3 GB per run, or keep it in the 10 GB Actions cache.
  - Example smoke test: 5 image prompts x (about 40 s encode/prefill + 128 tokens / ~10 t/s ≈ 13 s), about 4-5 min. Well inside the 6 h limit **(estimate)**. Measured later: 2 text + 4 image prompts took about 5 min; each image took about 66 s to read and answers came at 7.8 t/s.
  - MedGemma 27B Q4_K_M needs 16.55 + 0.86 + KV + overhead, about 19 GB. That is **more than 16 GB RAM**, and the 14 GB disk cannot even hold it plus the toolchain. **Not feasible.**
  - IQ2/IQ1 quants of 27B (6.5-9.6 GB) would fit, but quality at 1-2 bpw is poor. It only works as a load test **(unverified quality)**.
- **Private repo (2 vCPU / 8 GB)**: 4B Q4_K_M still fits (about 4.6 GB), at roughly half the speed.
- **Conclusion**: CPU CI can run 4B smoke tests (load, one image, a few tokens, output format checks). Anything with 27B, bf16, UniMedVL or fine-tuning needs a GPU runner.

---

## 4. GPU inference

### 4.1 Weight memory

| Model | bf16 (2 B/param) | int8 (~1 B/param) | int4 (~0.5-0.6 B/param) |
|---|---|---|---|
| MedGemma 27B (27.4 B) | ~55 GB (Google: 54 GB for Gemma 3 27B) | ~29 GB (Q8_0 28.7 GB) | ~14-17 GB (Google QAT int4: 14.1 GB; Q4_K_M 16.55 GB) |
| MedGemma 4B (4.3 B) | ~8.6 GB (Google: 8 GB) | ~4.1-4.6 GB | ~2.5-2.6 GB (QAT: 2.6 GB) |
| UniMedVL (14 B) | ~29.6 GB (file size) | ~15 GB (estimate) | ~8 GB (estimate; no published quant) |

Google's Gemma 3 QAT post: 27B int4 "fits comfortably on a single desktop NVIDIA RTX 3090" (24 GB), and BF16 runs on "a single high-end GPU like the NVIDIA H100". Its figures cover **weights only, without KV cache** [S34].

### 4.2 KV-cache formula and numbers

```
KV bytes = 2 (K and V) x n_layers x n_kv_heads x head_dim x bytes_per_elem x n_tokens
```

Configs from `config.json` (unsloth mirrors) [S35]:

- MedGemma 27B: 62 layers, 16 KV heads, head_dim 128, sliding window 1024
- MedGemma 4B: 34 layers, 4 KV heads, head_dim 256, sliding window 1024

Per-token cost at fp16:

- 27B: 2 x 62 x 16 x 128 x 2 = **507,904 B ≈ 0.48 MiB/token**
- 4B: 2 x 34 x 4 x 256 x 2 = **139,264 B ≈ 0.13 MiB/token**

Gemma 3 uses 5 local (1024-token sliding window) layers per 1 global layer. If the runtime implements sliding-window (SWA) KV caching, only about 1/6 of the layers keep the full context **(estimate, assuming 10 global + 52 local layers for 27B and 5 + 29 for 4B)**:

| Context | 27B naive | 27B with SWA | 4B naive | 4B with SWA |
|---|---|---|---|---|
| 4 K | 1.94 GiB | 0.72 GiB | 0.53 GiB | 0.19 GiB |
| 8 K | 3.88 GiB | 1.03 GiB | 1.06 GiB | 0.27 GiB |
| 32 K | 15.5 GiB | 2.9 GiB | 4.25 GiB | 0.74 GiB |
| 128 K | 62 GiB | 10.4 GiB | 17 GiB | 2.6 GiB |

An 8-bit KV cache (`-ctk q8_0 -ctv q8_0` in llama.cpp, `--kv-cache-dtype fp8` in vLLM) halves these numbers.

### 4.3 Fit table (weights + 8K KV + ~1.5 GB overhead + mmproj, estimate)

| GPU (VRAM, bandwidth) | 4B bf16 (~11 GB) | 27B int4 (~19-20 GB) | 27B int8 (~32 GB) | 27B bf16 (~60 GB) |
|---|---|---|---|---|
| T4 (16 GB, 320 GB/s) [S8] | yes | no (IQ3/Q3 only) | no | no |
| L4 (24 GB, 300 GB/s) [S36] | yes | yes, tight | no | no |
| RTX 3090 (24 GB, 936 GB/s) | yes | yes | no | no |
| RTX 4090 (24 GB, 1008 GB/s) [S37] | yes | yes | no | no |
| RTX 5090 (32 GB, 1792 GB/s) [S37] | yes | yes, with long context | barely (short context) | no |
| A100 80GB (2039 GB/s) [S38] | yes | yes | yes | yes |
| H100 SXM 80GB (3.35 TB/s) [S39] | yes | yes | yes | yes |

The RTX 3090's 936 GB/s figure is NVIDIA's published spec, but I could not fetch a primary page for it **(unverified)**.

### 4.4 Expected GPU decode speed (bandwidth ceiling x ~0.6-0.7, estimate)

| GPU | 4B bf16 (8.6 GB) | 27B Q4 (16.5 GB) | 27B bf16 (55 GB) |
|---|---|---|---|
| T4 | ~25 t/s | — | — |
| L4 | ~22 t/s | ~12 t/s | — |
| RTX 3090 | ~70 t/s | ~35-40 t/s | — |
| RTX 4090 | ~75 t/s | ~38-43 t/s | — |
| RTX 5090 | ~135 t/s | ~70 t/s | — |
| A100 80GB | ~150 t/s | ~80 t/s | ~25 t/s |
| H100 SXM | ~250 t/s | ~130 t/s | ~40 t/s |

Published third-party numbers vary a lot. One post claims vLLM 18.7 t/s vs Ollama 8.1 t/s for Gemma 3 27B Q4_K_M on an RTX 4090, which is far below the bandwidth ceiling and looks like partial CPU offload [S40]. **I found no trustworthy single-stream vLLM benchmark for Gemma 3 27B on H100.** Measure locally with `llama-bench -ngl 99 -p 512 -n 128`, or `vllm bench serve` at concurrency 1. Batched serving in vLLM raises total throughput far above these single-stream numbers.

---

## 5. Cloud GPU rental prices (snapshot, 2026)

| GPU | Lambda (on-demand, 1x) [S41] | RunPod (community / secure) [S42] | Vast.ai (live API, 2026-10-14: min / median, 1x on-demand) [S43] | Hyperscalers |
|---|---|---|---|---|
| H100 SXM 80GB | $4.29/h | $2.69 / $3.49 | $2.94 / $3.14 (8 offers) | AWS p5.48xlarge $55.04/h for 8 GPUs = **$6.88/GPU-h** us-east-1 [S44] |
| H100 PCIe 80GB | $3.29/h | $1.99 / $2.89 | — | — |
| A100 80GB | (8x only: $2.79/GPU-h) | $1.39 / $1.59 (SXM) | $0.40 / $0.70 (A100 SXM4, mixed 40/80 GB) | GCP ~$3.67 A100 80GB (unverified) |
| A100 40GB | $1.99/h | — | — | GCP ~$2.93 (unverified) |
| L40S 48GB | — | $0.79 / $1.09 | — | — |
| L4 24GB | — | $0.44 / $0.49 | $0.20 / $0.33 | AWS g6.xlarge $0.805/h (4 vCPU, 16 GiB) [S45]; GCP L4 ~$0.56-0.71 (unverified) |
| T4 16GB | — | — | $0.15 / $0.16 | GCP T4 $0.35/GPU-h on-demand [S46]; AWS g4dn.xlarge $0.526/h [S47]; GitHub T4 runner $3.12/h |
| RTX 4090 24GB | — | $0.34 / $0.74 | $0.40 / $0.60 | — |
| RTX 5090 32GB | — | $0.69 / $0.99 | $0.38 / $0.74 | — |
| RTX 3090 24GB | — | $0.22 / $0.50 | $0.12 / $0.20 | — |
| A6000 48GB | $1.09/h | — | — | — |

Ranges to quote:

- H100: **~$2-4/GPU-h** on specialist clouds, **~$7/GPU-h** on AWS on-demand.
- A100 80GB: **~$1.2-2.8/GPU-h**.
- L4: **~$0.3-0.8/h**.
- T4: **~$0.15-0.55/h**.
- RTX 4090: **~$0.35-0.9/h**.

Note that the GitHub-hosted T4 ($3.12/h) costs about 6-20x more than renting a T4 elsewhere. It does make sense for occasional CI jobs that need no infrastructure. Prices change weekly. Vast.ai and RunPod community prices are marketplace prices on hardware you do not control, so they are not suitable for patient data.

---

## 6. Fine-tuning memory and compute

### 6.1 Memory per parameter

Hugging Face "model memory anatomy" for mixed-precision AdamW [S48]: weights 6 B/param (fp16 copy 2 + fp32 master 4), Adam states 8 B/param, gradients 4 B/param. That is **18 B/param plus activations**. A leaner bf16 setup (bf16 weights 2 + bf16 grads 2 + fp32 master/m/v 12) gives the often-quoted **~16 B/param**. 8-bit Adam reduces the optimiser states to 2 B/param.

| Method | MedGemma 4B (4.3 B) | MedGemma 27B (27.4 B) | Notes |
|---|---|---|---|
| Full FT, AdamW mixed precision (16-18 B/param) | ~69-77 GB + activations, so 1x H100 80GB is tight; plan for 2x | ~440-490 GB + activations, so **8x H100 80GB with FSDP/ZeRO-3** | estimate |
| LoRA, 16-bit frozen base | ~9 GB base + adapters + activations, so ~12-20 GB | Unsloth minimum **64 GB** (27B row) [S49] | bf16 base needs no bf16-capable GPU for the weights, but compute benefits from bf16 |
| QLoRA, 4-bit NF4 base (Dettmers et al. 2023) | ~3.5-6 GB (Unsloth: 3B → 3.5 GB, 7B → 5 GB) [S49] | Unsloth minimum **22 GB** [S49]; Unsloth Gemma 3 post: 27B "fits in under 22GB" [S9] | QLoRA paper: a 65B model on a single 48 GB GPU [S50] |

So QLoRA of MedGemma 27B fits on a 24 GB card (RTX 3090/4090, L4) at short sequence lengths and batch size 1-2. 32-48 GB (RTX 5090, A6000, L40S) leaves room for longer image+text sequences. QLoRA of 4B fits on a 16 GB T4 or an 8-12 GB consumer card. Unsloth requires CUDA compute capability ≥ 7.0, so T4 and V100 are supported [S49].

### 6.2 Compute: FLOPs ≈ 6·N·D

- Training compute is about 6 x N params x D tokens (forward 2N + backward 4N per token) [S51]. The Chinchilla compute-optimal data size is about **20 tokens per parameter** [S52].
- H100 SXM bf16 dense: **989 TFLOPS** (NVIDIA lists 1,979 "with sparsity") [S39][S53]. Realistic MFU is **35-45%**. Llama 3 405B pre-training reported about 38-43% bf16 MFU on 8K-16K H100s, about 380-430 TFLOPS/GPU [S53]. Small single-node fine-tunes often reach only 20-40% **(unverified)**.

Worked examples (assuming 40% MFU, so ~400 TFLOPS effective per H100, at $2.5-4/GPU-h):

1. **LoRA/QLoRA fine-tune of MedGemma 27B on 10 M tokens** (for example 20k image+text samples x ~500 tokens):
   - Compute: 6 x 27e9 x 1e7 = 1.6e18 FLOP, divided by 4e14 FLOP/s = 4,050 s ≈ **1.1 H100-hours ≈ $3-5**.
   - LoRA skips the weight-gradient work for frozen weights (about 1/3 of the backward pass), so 6ND is an upper bound. QLoRA dequantisation adds overhead back. On an RTX 4090 (~165 TFLOPS bf16 dense, ~35% MFU, so ~58 TFLOPS) the same job takes about **8 hours** **(estimate)**.
2. **LoRA of MedGemma 4B on 10 M tokens**: 6 x 4.3e9 x 1e7 = 2.6e17 FLOP. That is about 11 min on an H100, or about 1.3 h on an RTX 4090 **(estimate)**.
3. **Training a 27B model from scratch, Chinchilla-optimal** (only to show why this is out of scope):
   - D = 20 x 27e9 = 540 B tokens
   - C = 6 x 27e9 x 5.4e11 = 8.7e22 FLOP
   - Time: 8.7e22 / 4e14 = 2.2e8 s = **~61,000 H100-hours ≈ $150k-250k**

Hugging Face also notes that "training a 4B parameter model in mixed precision on a batch size of 16 requires roughly 85GB of GPU memory" [S48], which matches the 18 B/param rule plus activations.

---

## 7. Limiting resource use on the development machine

The issue asks us to "be careful with local resources". Use several layers of protection.

| Mechanism | Command / setting | What it does |
|---|---|---|
| Thread count | `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`, `torch.set_num_threads(4)`, llama.cpp `-t 4` / `--threads-batch 4` | Caps CPU parallelism. Above about the number of physical cores, decode stops getting faster anyway (see 3.1). |
| CPU priority | `nice -n 19 <cmd>` | Lowest scheduling priority. The desktop stays responsive. |
| I/O priority | `ionice -c3 <cmd>` (idle) or `-c2 -n7` | Model loading does not starve other disk users. |
| systemd scope (cgroups v2) | `systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G -p CPUQuota=400% -p IOWeight=10 <cmd>` | Hard RAM cap: the OOM killer acts inside the scope only. CPUQuota=400% = 4 CPUs' worth of time. `AllowedCPUs=0-3` pins to cores. `MemoryHigh` throttles and `MemoryMax` is the "last line of defence" [S54]. |
| Address-space ulimit | `ulimit -v 12000000` (KiB) in a subshell | Crude virtual memory cap. It can break mmap-heavy tools (llama.cpp mmaps the GGUF) and CUDA, which reserve large virtual ranges. Prefer cgroups **(general knowledge, unverified for each tool)**. |
| Docker | `docker run --cpus=4 --memory=12g --memory-swap=12g --gpus '"device=0"' ...` | `--cpus` caps CPU time. Setting `--memory-swap` equal to `--memory` disables swap [S55]. `--gpus` needs nvidia-container-toolkit [S16]. |
| GPU | `CUDA_VISIBLE_DEVICES=0`, `nvidia-smi -pl 250` (power limit, root), vLLM `--gpu-memory-utilization 0.8`, llama.cpp `-ngl N` partial offload | Leaves VRAM and headroom for the desktop. |
| Disk | `HF_HOME=/data/hf-cache`; check `df -h` before downloads; `huggingface-cli download --include "*Q4_K_M*" "mmproj-F16*"` | Avoids pulling every quant: the 27B repo is about 450 GB in total. |
| Runner isolation | Self-hosted runner as its own user, inside a systemd slice with `MemoryMax`/`CPUQuota` (add `[Service]` overrides via `systemctl edit actions.runner.*`) | CI jobs cannot take over the workstation. |

A suggested default for the 6-core / 11 GiB machine used here: `systemd-run --user --scope -p MemoryMax=8G -p CPUQuota=400% nice -n 19 ionice -c3 llama-mtmd-cli -t 4 -m medgemma-4b-it-Q4_K_M.gguf --mmproj mmproj-F16.gguf ...`. That uses about 4.6 GB for 4B Q4_K_M and leaves about 3 GB and 2 cores free. 27B Q4_K_M (~19 GB) cannot run on this machine at all **(estimate)**.

---

## 8. Summary of recommendations

1. **CPU CI (GitHub-hosted, free for public repo)**: MedGemma 4B (or 1.5-4B) Q4_K_M + mmproj-F16 from the ungated unsloth GGUF repos, run with llama.cpp. About 3.3 GB download, ~4.6 GB RAM, about 10-15 t/s decode and about 30-60 s per image on 4 vCPU **(estimate; measured later: 7.8 t/s and about 66 s per image with 3 threads)**. Keep the tests tiny (few prompts, `-n 32..128`) and cache the model with `actions/cache`.
2. **GPU tests (self-hosted runner, label `gpu`)**: 27B int4 needs at least a 24 GB GPU. 27B bf16 needs at least an 80 GB GPU. UniMedVL bf16 needs about 32-40 GB (A100/H100 or 2x24 GB) **(estimate)**. Trigger with `workflow_dispatch` / label-gated only, use an ephemeral runner, and declare labels in `.github/actionlint.yaml`.
3. **Hosted GPU alternative**: GitHub T4 runner at $0.052/min (Team/Enterprise only). It fits 4B bf16 and 27B only at ≤3.5 bpw.
4. **Fine-tuning**: QLoRA 27B needs ≥22-24 GB. QLoRA 4B needs ≥6-8 GB. Full FT of 27B needs a multi-GPU node. Budget about 1 H100-hour per 10 M training tokens for 27B LoRA.

---

## Sources

- [S1] GitHub Docs, GitHub-hosted runners reference: https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- [S2] GitHub Docs, Actions limits: https://docs.github.com/en/actions/reference/limits
- [S3] GitHub Changelog 2025-11-20, cache can exceed 10 GB: https://github.blog/changelog/2025-11-20-github-actions-cache-size-can-now-exceed-10-gb-per-repository/
- [S4] GitHub Docs, Actions runner pricing: https://docs.github.com/en/billing/reference/actions-runner-pricing
- [S5] GitHub Docs, Larger runners reference: https://docs.github.com/en/actions/reference/runners/larger-runners
- [S6] GitHub Docs, About larger runners (Team/Enterprise Cloud; no included minutes): https://docs.github.com/en/actions/concepts/runners/larger-runners and https://docs.github.com/en/enterprise-cloud@latest/actions/how-tos/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners ; GPU beta announcement: https://github.blog/2024-04-02-bringing-enterprise-level-security-and-even-more-power-to-github-hosted-runners/
- [S7] The Register, GitHub walks back self-hosted runner charge (2025-12-17): https://www.theregister.com/2025/12/17/github_charge_dev_own_hardware/ ; GitHub pricing changes page: https://resources.github.com/actions/2026-pricing-changes-for-github-actions/
- [S8] NVIDIA T4 product page: https://www.nvidia.com/en-us/data-center/tesla-t4/
- [S9] Unsloth blog, Gemma 3 (27B fits in <22 GB; float16 overflow on T4/V100): https://unsloth.ai/blog/gemma3 (summarised via https://learnopencv.com/unsloth-guide-efficient-llm-fine-tuning/ because the page returned 403 to the fetcher)
- [S10] GitHub Docs, Adding self-hosted runners: https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners
- [S11] GitHub Docs, Applying labels to self-hosted runners: https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/apply-labels
- [S12] GitHub Docs, Configuring the runner application as a service: https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application
- [S13] actions/runner latest release (v2.338.0, 2026-10-06): https://github.com/actions/runner/releases/latest
- [S14] actions/runner source, CommandSettings.cs (flags ephemeral, disableupdate, no-default-labels, labels): https://github.com/actions/runner/blob/main/src/Runner.Listener/CommandSettings.cs
- [S15] NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- [S16] NVIDIA Container Toolkit sample workload: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html
- [S17] GitHub Docs, Secure use reference (self-hosted runner hardening, JIT runners): https://docs.github.com/en/actions/reference/security/secure-use
- [S18] actionlint configuration docs: https://github.com/rhysd/actionlint/blob/main/docs/config.md
- [S19] computingforgeeks, llama.cpp benchmark (decode bandwidth-bound, prefill compute-bound): https://computingforgeeks.com/run-local-llm-llama-cpp/
- [S20] Andrew Chan, "Fast LLM inference from scratch" (bandwidth speed-of-light): https://andrewkchan.dev/posts/yalm.html
- [S21] llama.cpp llama-bench README: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- [S22] OpenBenchmarking result 2606024-NE-LLAMAALLM76 (Ryzen 9 9950X3D, llama.cpp b8933 CPU BLAS): https://openbenchmarking.org/result/2606024-NE-LLAMAALLM76
- [S23] Memory bandwidth calculator: https://west-games.com/memory-bandwidth-calculator/ ; Framework community thread (effective ~76 GB/s DDR5-5600): https://community.frame.work/t/lpcamm2-7466-100gb-s/83631
- [S24] llama.cpp quantize README (bits/weight table): https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- [S25] HF API, unsloth/medgemma-27b-it-GGUF file tree: https://huggingface.co/api/models/unsloth/medgemma-27b-it-GGUF/tree/main
- [S26] HF API, unsloth/medgemma-4b-it-GGUF file tree: https://huggingface.co/api/models/unsloth/medgemma-4b-it-GGUF/tree/main
- [S27] HF API, unsloth/gemma-3-27b-it-GGUF file tree: https://huggingface.co/api/models/unsloth/gemma-3-27b-it-GGUF/tree/main
- [S28] HF API, unsloth/medgemma-1.5-4b-it-GGUF file tree: https://huggingface.co/api/models/unsloth/medgemma-1.5-4b-it-GGUF/tree/main
- [S29] Model card google/medgemma-27b-it (gated, HAI-DEF terms, 896x896 to 256 tokens, 128K context): https://huggingface.co/google/medgemma-27b-it ; HF API gating status: https://huggingface.co/api/models/google/medgemma-27b-it , https://huggingface.co/api/models/google/medgemma-4b-it , https://huggingface.co/api/models/google/medgemma-1.5-4b-it
- [S30] Model card General-Medical-AI/UniMedVL and file tree: https://huggingface.co/General-Medical-AI/UniMedVL , https://huggingface.co/api/models/General-Medical-AI/UniMedVL/tree/main
- [S31] llama.cpp multimodal docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- [S32] Ollama library medgemma: https://ollama.com/library/medgemma , https://ollama.com/library/medgemma/tags , search: https://ollama.com/search?q=medgemma
- [S33] Third-party CPU estimates: https://llmconfigurator.com/en/can-i-run/gemma-3-on-32gb-ram and https://bestllmfor.com/guide/llm-local-cpu-sans-gpu/ (not primary measurements)
- [S34] Google Developers Blog, Gemma 3 QAT: https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/
- [S35] Model configs: https://huggingface.co/unsloth/medgemma-27b-it/blob/main/config.json , https://huggingface.co/unsloth/medgemma-4b-it/blob/main/config.json
- [S36] NVIDIA L4 product page: https://www.nvidia.com/en-us/data-center/l4/
- [S37] PCGamesN RTX 5090 specs (32 GB GDDR7, 1792 GB/s; RTX 4090 ~1 TB/s): https://www.pcgamesn.com/nvidia/geforce-rtx-5090-guide ; https://www.spheron.network/blog/nvidia-rtx-5090-specs/
- [S38] NVIDIA A100 product page: https://www.nvidia.com/en-us/data-center/a100/
- [S39] NVIDIA H100 product page: https://www.nvidia.com/en-us/data-center/h100/
- [S40] markaicode RTX 4090 tokens-per-second benchmark pages (low-reliability source; also https://markaicode.com/benchmarks/llamacpp-tokens-per-second-benchmark/): https://markaicode.com/benchmarks/cuda-gemma-4-rtx-4090-tokens-per-second-benchmark/
- [S41] Lambda pricing: https://lambda.ai/pricing
- [S42] RunPod pricing: https://www.runpod.io/pricing
- [S43] Vast.ai public offers API (queried 2026-10-14): https://console.vast.ai/api/v0/bundles/ ; pricing page: https://vast.ai/pricing
- [S44] AWS p5.48xlarge pricing (aggregators): https://calculator.holori.com/aws/ec2/p5.48xlarge/us-east-1 , https://compute.doit.com/spot/us-east-1/p5.48xlarge
- [S45] Vantage, g6.xlarge: https://instances.vantage.sh/aws/ec2/g6.xlarge
- [S46] Google Cloud GPU pricing: https://cloud.google.com/compute/gpus-pricing ; L4 third-party: https://computeprices.com/providers/google/gpus/l4
- [S47] Vantage, g4dn.xlarge: https://instances.vantage.sh/aws/ec2/g4dn.xlarge
- [S48] Hugging Face Transformers docs, model memory anatomy: https://huggingface.co/docs/transformers/main/en/model_memory_anatomy
- [S49] Unsloth requirements (VRAM table, CUDA capability ≥ 7.0): https://www.unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- [S50] Dettmers et al. 2023, QLoRA: https://arxiv.org/abs/2305.14314
- [S51] Kaplan et al. 2020, Scaling Laws for Neural Language Models (C ≈ 6ND): https://arxiv.org/abs/2001.08361
- [S52] Hoffmann et al. 2022, Training Compute-Optimal LLMs (Chinchilla): https://arxiv.org/abs/2203.15556
- [S53] Llama 3 MFU (Table 4 of "The Llama 3 Herd of Models", https://arxiv.org/abs/2407.21783), reproduced at https://zenn.dev/nariaki3551/scraps/6221fb5a306064 and https://glennklockwood.com/garden/Llama-3.1 ; H100 989 TFLOPS dense bf16 in NVIDIA NGC benchmarking: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dgxc-benchmarking/resources/llama31-405b-dgxc-benchmarking-a
- [S54] systemd.resource-control(5): https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html
- [S55] Docker resource constraints: https://docs.docker.com/engine/containers/resource_constraints/
- MedGemma GitHub repo (license: HAI-DEF for model, Apache-2.0 for code): https://github.com/google-health/medgemma
