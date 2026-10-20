# Medical neural network research

Public-domain Python tools for evaluating medical models, estimating hardware
requirements, and planning training experiments. The benchmark uses real,
openly licensed medical images and documents downloaded at test time.

[![CI/CD Pipeline](https://github.com/konard/medical-neural-network-ml/actions/workflows/release.yml/badge.svg)](https://github.com/konard/medical-neural-network-ml/actions/workflows/release.yml)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://unlicense.org/)

## Features

- Seven tasks: toy differential diagnosis, DDXPlus interviews, prescription
  handwriting, printed case histories, chest X-rays, MRI and ultrasound.
- The same scoring harness for OpenAI-compatible servers and native MLX-VLM
  on Apple Silicon. Models and datasets are optional downloads.
- Pinned file hashes, source attribution in reports, and offline cache reuse.
- Memory and time budgets, plus inference and training resource estimates.
- [Research plan and evidence](docs/case-studies/issue-1/README.md), including
  alternatives, architectures, data collection and fine-tuning.
- [GPU tests on temporary cloud machines](docs/cloud-gpu.md) on AWS, Azure,
  Google Cloud or Hetzner, with or without Kubernetes. They run from a laptop,
  GitHub Actions or GitLab CI, and the machines are deleted afterwards
  ([case study](docs/case-studies/issue-23/README.md)).

These are research smoke tests. Small, selected teaching samples do not
establish clinical accuracy. MedGemma 27B and UniMedVL have not been measured
here; their resource numbers are estimates. See the
[delivery audit](docs/audit-2026-10-16.md) for remaining work.

## Quick Start

```bash
git clone https://github.com/konard/medical-neural-network-ml.git
cd medical-neural-network-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# No downloads or model required
medbench host
medbench run --task diagnosis --backend naive-bayes --count 50
pytest
```

For **M3 Pro / Apple Silicon**, use the
[native MLX instructions](docs/apple-silicon.md). They include a tiny Metal
smoke test that needs no model, then bounded text and real-image tests.
The Linux llama.cpp script downloads Linux x86_64 executables and refuses to
run those executables on macOS.

For Linux CPU inference with MedGemma 4B (about 3.3 GB of model files):

```bash
bash scripts/medbench-model-run.sh download
bash scripts/medbench-model-run.sh run
```

This downloads models subject to their own terms. Review those terms first.
Real data are downloaded into a separate cache. List sources before downloading:

```bash
medbench sources
medbench fetch --task imaging-xray --out .cache/medbench-data
MEDBENCH_DATA_DIR=.cache/medbench-data medbench run \
  --task imaging-xray --backend openai --base-url http://127.0.0.1:8080/v1 \
  --count 2 --out medbench-reports/imaging-xray.json
```

A benchmark that runs out of time or evaluates zero items exits with failure.
Reports retain completed answers and indicate whether the run stopped early.
For a hard timeout on native MLX, use `scripts/medbench-mlx-run.py`.

## Hardware and workflows

| Workflow | Hardware and resources | Validation |
| --- | --- | --- |
| Unit tests, lint, types, package/docs build | CPU, Python 3.11+ (the `docs` extra needs 3.12+, as Sphinx 9.1 does); no model | Local and hosted CI |
| Real sample verification | CPU, network, about 60 MB of data | Hash and license checks; no inference |
| MedGemma 4B CPU smoke | Linux x86_64, hosted 4 vCPU / 16 GB RAM; allow 8 GB free disk for archives, extraction and weights | Measured on GitHub CPU runners |
| M3 Pro MLX smoke | macOS, native arm64 Python 3.11+; 256 MiB allocation ceiling | Measured on this M3 Pro |
| MedGemma 4B MLX | 18 GB unified-memory Mac can be sufficient with other apps closed; approximately 3 GB checkpoint plus runtime buffers; dynamic free-memory preflight | Backend implemented; see local evidence and limits |
| MedGemma 27B Q4 GPU | Linux x86_64, NVIDIA 24 GB VRAM recommended, 32 GB+ host RAM and 50 GB free disk | Workflow available; GPU measurement outstanding |
| UniMedVL bf16 | Dedicated CUDA host; at least 40 GB VRAM as an initial capacity target, 64 GB+ RAM and 80 GB free disk | Adapter for the pinned upstream runtime implemented; GPU measurement outstanding |
| Fine-tuning / training | Dedicated GPU; depends on optimizer, context, batch, image size and sharding | Formula estimates only; measurements outstanding |

The [hardware guide](docs/hardware-workflows.md) distinguishes measured
results, capacity estimates and missing runtime support. A calculator saying
`fits: true` does not establish that a model/quantization combination is
implemented by a runtime. All sizes in calculator output use decimal GB.

```bash
medbench estimate --model medgemma-4b-it --hardware m3-pro-18gb --quant Q4_K_M
medbench estimate --model medgemma-27b-it --hardware l4-24gb --quant Q4_K_M
medbench estimate --model unimedvl --hardware a100-80gb --quant bf16 --finetune
```

The optional self-hosted NVIDIA job runs only on manual dispatch:

```bash
gh workflow run model-tests.yml -f gpu=true -f gpu-model=medgemma-27b-it -f count=20
gh workflow run model-tests.yml -f gpu=true -f gpu-model=unimedvl -f count=20
```

Each model download and run is wrapped in `medbench monitor`, which writes
`medbench-reports/resources-*.json` with peak GPU and host memory, free disk,
driver and CUDA versions and wall time, also when the run fails.

Without a GPU of your own, `python -m scripts.cloud_gpu` rents one for a
single run. It creates the machine, runs the same tests and deletes it again
([docs/cloud-gpu.md](docs/cloud-gpu.md)):

```bash
python -m scripts.cloud_gpu plan --provider aws --model medgemma-27b-it
python -m scripts.cloud_gpu all --provider gcp --model medgemma-4b-it --gcp-project my-project --yes
```

## Configuration

- `pyproject.toml`: package identity, extras, tests and changelog configuration.
- `MEDBENCH_DATA_DIR`: cache for separately licensed test files.
- `MEDBENCH_OFFLINE=1`: refuse data downloads; cached hashes are still checked.
- `MEDBENCH_API_KEY`: optional API key for a model server.
- `medbench run --help`: task, backend, token and time limits.
- [Linux runner configuration](docs/case-studies/issue-1/ci-and-gpu-runner.md).

The distribution is named `medical-neural-network-ml`; the research module and
CLI are named `medbench`. The inherited `my_package` arithmetic example is
retained for compatibility and is not the medical benchmark API.

## Development and documentation

```bash
ruff check .
ruff format --check .
mypy src/
python scripts/check_file_size.py
bash scripts/check-required-docs.sh
pip install -e '.[docs]'  # Python 3.12+ (Sphinx 9.1 dropped 3.11)
sphinx-build -W --keep-going -b html docs _site
```

API documentation is generated from `medbench`. For GitHub Pages, set
Settings → Pages → Source to GitHub Actions and the repository variable
`DEPLOY_GITHUB_PAGES=true`. Otherwise CI builds the site without deployment.
[Preview regeneration](docs/preview-regeneration.md) records inherited template
tracking; there is no browser application in this repository.

Publishing requires a PyPI trusted publisher for this repository and package
name; [Publishing to PyPI](docs/publishing.md) lists the settings a maintainer
enters and what CI checks before and after each upload. CI checks publisher
credentials and refuses a release when they are not configured.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Include a changelog fragment for code
changes. New original contributions must remain under the Unlicense. Keep
third-party weights and test media in ignored caches, with their licenses
and attribution attached to any redistributed output.

## License

Original code is released into the public domain under the
[Unlicense](LICENSE). This dedication does not relicense third-party material.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) describes dataset excerpts,
model terms, dependencies and report attribution. Downloaded data can require
attribution or share-alike terms even though this repository's code is public
domain.

The CI/CD tooling was adapted from the public-domain
[Python development pipeline template](https://github.com/link-foundation/python-ai-driven-development-pipeline-template).
