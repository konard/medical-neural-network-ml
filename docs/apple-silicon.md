# Testing on Apple Silicon with MLX

MLX is Apple's array framework; MLX-VLM runs vision-language models on its
Metal GPU backend. Use native arm64 Python 3.11+ in a virtual environment.
The audit used Python 3.12, macOS 15.7.7, an M3 Pro with 12 CPU / 18 GPU
cores and 18 GiB unified memory. The CPU, GPU, macOS and applications share
that memory. The Neural Engine is not used by this backend.

## Install and verify the GPU

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,mlx]'
python scripts/medbench-mlx-run.py --smoke --timeout 30
medbench host
```

The smoke test allocates only a 64×64 matrix, sets a 256 MiB MLX memory ceiling
and verifies Metal execution without downloading a model. It passed on this
Mac with MLX 0.32.4 / MLX-VLM 0.7.8; peak MLX allocation was 32,772 bytes.
Some sandboxes deny Metal, `sysctl` or loopback networking; run these commands
in a normal local terminal. A zero memory reading is unknown capacity, and
model loading is refused.

If a Python.org installation cannot verify HTTPS certificates, install its
provided certificates or set `SSL_CERT_FILE` to a trusted CA bundle, for
example `export SSL_CERT_FILE="$(python -m certifi)"` when certifi is installed.
Do not turn off TLS verification.

## A small end-to-end test

This optional general-purpose model is only a wiring test. It is too small
to draw medical conclusions. Its checkpoint is about 145 MB, licensed
Apache-2.0, and kept outside Git in `.cache/`.

```bash
hf download mlx-community/SmolVLM-256M-Instruct-4bit \
  --revision 69cb5195f414ceb6398c5581254673d2c6c8d0d8 \
  --local-dir .cache/mlx-smolvlm

python scripts/medbench-mlx-run.py --timeout 120 \
  --model-path .cache/mlx-smolvlm --task diagnosis --count 1 \
  --max-tokens 32 --time-budget 90 --out medbench-reports/mlx-text.json

# Downloads one real, hash-verified image; the report includes attribution.
MEDBENCH_DATA_DIR=.cache/medbench-data \
python scripts/medbench-mlx-run.py --timeout 120 \
  --model-path .cache/mlx-smolvlm --task imaging-xray --count 1 \
  --max-tokens 48 --time-budget 90 --out medbench-reports/mlx-image.json
```

Both cases completed on this Mac: 1.34 seconds for text generation and
2.72 seconds for the image request, excluding model loading. Both failed to
produce the requested answer format. This validates inference and image
transport, not medical accuracy. Raw reports are in
[the local evidence folder](https://github.com/konard/medical-neural-network-ml/tree/main/docs/case-studies/issue-1/data/m3-pro-20261016).
The checkpoint SHA-256 is
`a283e0ef139e8a31fe5dc25bcdb0f3ee4ecf28aa6820840e49eb3519cdbec3e6`.

## MedGemma 4B

Review Google's Health AI Developer Foundations terms before obtaining the
weights. The MLX conversion retains those terms. A model that uses MLX-LM
and has no `vision_config` cannot process images; use the MLX-VLM conversion.
The following snapshot has `Gemma3ForConditionalGeneration`, a vision config
and 4-bit weights (2,995,351,814 bytes):

```bash
hf download mlx-community/medgemma-4b-it-4bit \
  --revision db10be5ae6bd569e2063c3c74b9b9dfbcbedcad3 \
  --local-dir .cache/mlx-medgemma-4b

python scripts/medbench-mlx-run.py --timeout 180 \
  --model-path .cache/mlx-medgemma-4b --task imaging-xray --count 1 \
  --max-tokens 64 --time-budget 150 --memory-limit-gb 8 \
  --out medbench-reports/mlx-medgemma-xray.json
```

At audit time the Mac had about 4.15 GB reclaimable RAM, giving the default
70% budget only 2.9 GB. Loading this checkpoint needs at least 4.5 GB with
the backend's buffer allowance, so it would be refused. No 4B performance
result is claimed. Close memory-heavy applications and rerun `medbench host`;
never increase macOS wired-memory limits to force it to fit.

The guard uses free + inactive + speculative pages, excluding compressed and
wired pages. The allocation ceiling is the smallest of 8 GB (configurable),
70% of reclaimable RAM, and Metal's recommended working set. This is an MLX
allocation limit, not a limit on every Python allocation. Large images and
prompts can still require more memory. Keep one image/request, small counts
and output limits; chunked prefill uses 256 tokens. No CPU fallback or
background server is started. The wrapper lowers priority and kills its
worker at the hard deadline. Deadline or memory failures return a nonzero
exit status.

## Larger models and training

MedGemma 27B does not fit the conservative budget on this 18 GiB Mac.
UniMedVL uses a different custom CUDA runtime; its published bf16 weights
alone exceed this machine's installed RAM. Neither is a local test target.
MLX fine-tuning is not implemented by this repository. CUDA NF4/QLoRA
estimates do not prove an equivalent MLX training workflow exists.

Sources: [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html),
[MLX-VLM](https://github.com/Blaizzy/mlx-vlm),
[MedGemma MLX model](https://huggingface.co/mlx-community/medgemma-4b-it-4bit),
[Apple M3 Pro specifications](https://support.apple.com/en-euro/117736).
