# Pinned vs latest, per ecosystem

Every "latest" below was resolved from the registry on 2026-10-19, not from
memory:

- PyPI's JSON API, through
  [experiments/issue21_pinned_vs_latest.py](../../../../../../experiments/issue21_pinned_vs_latest.py).
  Its output is in [data/python-pins-vs-latest.md](data/python-pins-vs-latest.md)
  and [data/unimedvl-lock-vs-latest.md](data/unimedvl-lock-vs-latest.md).
- GitHub tags (`git ls-remote`) for actions and hooks, through
  [experiments/issue21_actions_hooks_vs_latest.py](../../../../../../experiments/issue21_actions_hooks_vs_latest.py).
  That script also checks every pinned SHA against its tag, and its output is
  in [data/actions-hooks-vs-latest.md](data/actions-hooks-vs-latest.md).
- [data/latest.json](data/latest.json) has the GitHub releases and the PyPI
  tools.

"main" is `fff9d99`, the commit the issue was opened against. PR #20
(issue #19) had already moved most pins two days earlier, so most rows only
confirm that they are still the newest.

## 1. Python package (`pyproject.toml`, `docs/requirements.txt`)

The package is a library, so it declares floors, not a lock. Every floor is
the newest release.

| Dependency | Where | main | This branch | Latest | Note |
| --- | --- | --- | --- | --- | --- |
| Python (`requires-python`) | pyproject | `>=3.11` | `>=3.11` | 3.15.0 | Tested on 3.11-3.15 (Compatibility) |
| hatchling | build-system | `>=1.32.4` | `>=1.32.4` | 1.32.4 | |
| ruff | dev | `>=0.17.1,<0.18` | `>=0.17.1,<0.18` | 0.17.1 | The upper bound admits the newest release, so it is kept. ruff adds rules in minor releases, so a minor bump is reviewed by hand, with the hook rev (`tests/test_project_metadata.py` keeps the two equal) |
| mypy | dev | `>=2.4.0` | `>=2.4.0` | 2.4.0 | Dependabot PR #14 asked for `<3.0`. main already had a floor of 2.4.0 and no upper bound |
| pytest | dev | `>=9.1.1` | `>=9.1.1` | 9.1.1 | |
| pytest-asyncio | dev | `>=1.4.0` | `>=1.4.0` | 1.4.0 | |
| pytest-cov | dev | `>=7.1.0` | `>=7.1.0` | 7.1.0 | |
| pre-commit | dev | `>=4.7.0` | `>=4.7.0` | 4.7.0 | |
| scriv[toml] | dev | `>=1.8.0` | `>=1.8.0` | 1.8.0 | |
| pillow | dev, imaging | `>=13.0.0` | `>=13.0.0` | 13.0.0 | Same version in the mypy hook and the UniMedVL lock |
| mlx-vlm | mlx | `==0.7.8` | **`==0.7.9`** | 0.7.9 (released 2026-10-19) | Exact pin: `MlxBackend` calls its Python API. 0.7.9 reads sampling defaults from the checkpoint only for options the caller leaves unset ([analysis.md F9](analysis.md#f9-mlx-vlm-079-reads-sampling-defaults-from-the-checkpoint)) |
| sphinx | docs, docs/requirements.txt | `>=9.1.0` | `>=9.1.0` | 9.1.0 | |
| furo | docs, docs/requirements.txt | `>=2025.12.19` | `>=2025.12.19` | 2025.12.19 | |
| myst-parser | docs, docs/requirements.txt | `>=5.1.0` | `>=5.1.0` | 5.1.0 | |

The Dependabot pull requests named in the issue asked for these versions:

| PR | Asked for | Version now | Where it landed |
| --- | --- | --- | --- |
| #13 | 12 updates in the `python-minor-and-patch` group (including torch 2.6.0 → newer in the UniMedVL pins) | Every package at its newest release | main (PR #20) and this branch (UniMedVL lock) |
| #14 | mypy `>=1.13.0,<3.0` | `>=2.4.0` (no upper bound) | main (PR #20) |
| #15 | packaging 25.0 → 26.3 | 26.3 | main (PR #20) |
| #16 | transformers 4.49.0 → 5.19.0 | 5.20.1 | this branch |
| #17 | numpy 1.24.4 → 2.2.6 | 2.5.4 | main (PR #20) |

## 2. UniMedVL runtime (`scripts/unimedvl-requirements.in` → `.txt`)

This is a full lock compiled by `uv pip compile` from `scripts/`, for Python
3.14 and `x86_64-manylinux_2_28`. On main it had 55 packages. On this branch
it has 68. The CUDA libraries are now split per component (`nvidia-*`
without the `-cu12` suffix, through `cuda-toolkit`), and huggingface-hub 2
replaced requests with httpx2 and typer.

torchvision, triton and tokenizers are named in the `.in` without a version.
Each release of torch or transformers needs its own release of them, and
Dependabot updates the members of a group one at a time, so exact pins on
both sides would never move. The lock pins them, and an update of torch or
transformers carries them along
([F7](analysis.md#f7-dependabot-could-not-update-the-unimedvl-lock)).

| Package | main | This branch | Latest | Why not latest |
| --- | --- | --- | --- | --- |
| accelerate | 1.15.0 | 1.15.0 | 1.15.0 |  |
| annotated-doc | — | 0.0.5 | 0.0.5 |  |
| anyio | — | 4.15.1 | 4.15.1 |  |
| certifi | 2026.7.22 | removed | — |  |
| charset-normalizer | 3.5.2 | removed | — |  |
| click | — | 8.5.0 | 8.5.0 |  |
| contourpy | 1.4.0 | 1.4.0 | 1.4.0 |  |
| cuda-bindings | — | 13.4.4 | 13.4.4 |  |
| cuda-pathfinder | — | 1.8.3 | 1.8.3 |  |
| cuda-toolkit | — | 13.0.3.0 | 13.4.3 | torch 2.14.1 requires cuda-toolkit==13.0.3 |
| cycler | 0.12.1 | 0.12.1 | 0.12.1 |  |
| einops | 0.8.2 | 0.8.2 | 0.8.2 |  |
| filelock | 4.2.0 | 4.2.4 | 4.2.4 |  |
| fonttools | 4.67.0 | 4.67.0 | 4.67.0 |  |
| fsspec | 2026.9.0 | 2026.9.0 | 2026.9.0 |  |
| h11 | — | 0.16.0 | 0.16.0 |  |
| hf-xet | 1.7.0 | 1.7.0 | 1.7.0 |  |
| httpcore2 | — | 2.14.0 | 2.14.0 |  |
| httpx2 | — | 2.14.0 | 2.14.0 |  |
| huggingface-hub | 0.36.2 | 2.2.0 | 2.2.0 |  |
| idna | 3.20 | 3.20 | 3.20 |  |
| jinja2 | 3.1.6 | 3.1.6 | 3.1.6 |  |
| kiwisolver | 1.5.1 | 1.5.1 | 1.5.1 |  |
| markdown-it-py | — | 4.2.0 | 4.2.0 |  |
| markupsafe | 3.0.4 | 3.0.4 | 3.0.4 |  |
| matplotlib | 3.11.2 | 3.11.2 | 3.11.2 |  |
| mdurl | — | 0.1.2 | 0.1.2 |  |
| mpmath | 1.3.0 | 1.3.0 | 1.4.1 | sympy 1.14.0 (newest) requires mpmath<1.4 |
| networkx | 3.7 | 3.7 | 3.7 |  |
| numpy | 2.5.4 | 2.5.4 | 2.5.4 |  |
| nvidia-cublas | — | 13.1.1.3 | 13.8.1.7 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-cublas-cu12 | 12.8.4.1 | removed | — |  |
| nvidia-cuda-cupti | — | 13.0.85 | 13.4.92 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-cuda-cupti-cu12 | 12.8.90 | removed | — |  |
| nvidia-cuda-nvrtc | — | 13.0.88 | 13.4.92 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | removed | — |  |
| nvidia-cuda-runtime | — | 13.0.96 | 13.4.92 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-cuda-runtime-cu12 | 12.8.90 | removed | — |  |
| nvidia-cudnn-cu12 | 9.10.2.21 | removed | — |  |
| nvidia-cudnn-cu13 | — | 9.24.0.43 | 9.27.0.42 | torch 2.14.1 requires nvidia-cudnn-cu13==9.24.0.43 |
| nvidia-cufft | — | 12.0.0.61 | 12.4.0.43 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-cufft-cu12 | 11.3.3.83 | removed | — |  |
| nvidia-cufile | — | 1.15.1.6 | 1.19.1.55 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-cufile-cu12 | 1.13.1.3 | removed | — |  |
| nvidia-curand | — | 10.4.0.35 | 10.4.4.72 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-curand-cu12 | 10.3.9.90 | removed | — |  |
| nvidia-cusolver | — | 12.0.4.66 | 12.3.5.3 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-cusolver-cu12 | 11.7.3.90 | removed | — |  |
| nvidia-cusparse | — | 12.6.3.3 | 12.8.6.72 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-cusparse-cu12 | 12.5.8.93 | removed | — |  |
| nvidia-cusparselt-cu12 | 0.7.1 | removed | — |  |
| nvidia-cusparselt-cu13 | — | 0.8.1 | 0.10.0 | torch 2.14.1 requires nvidia-cusparselt-cu13==0.8.1 |
| nvidia-nccl-cu12 | 2.27.3 | removed | — |  |
| nvidia-nccl-cu13 | — | 2.30.7 | 2.32.3 | torch 2.14.1 requires nvidia-nccl-cu13==2.30.7 |
| nvidia-nvjitlink | — | 13.4.92 | 13.4.92 |  |
| nvidia-nvjitlink-cu12 | 12.8.93 | removed | — |  |
| nvidia-nvshmem-cu13 | — | 3.4.5 | 3.8.0 | torch 2.14.1 requires nvidia-nvshmem-cu13==3.4.5 |
| nvidia-nvtx | — | 13.0.85 | 13.4.92 | pinned exactly by torch 2.14.1 (through cuda-toolkit 13.0.3) |
| nvidia-nvtx-cu12 | 12.8.90 | removed | — |  |
| opencv-python | 5.0.0.93 | 5.0.0.93 | 5.0.0.93 |  |
| packaging | 26.3 | 26.3 | 26.3 |  |
| pillow | 13.0.0 | 13.0.0 | 13.0.0 |  |
| psutil | 7.2.2 | 7.2.2 | 7.2.2 |  |
| pygments | — | 2.21.0 | 2.21.0 |  |
| pyparsing | 3.3.3 | 3.3.3 | 3.3.3 |  |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 | 2.9.0.post0 |  |
| pyyaml | 6.0.3 | 6.0.3 | 6.0.3 |  |
| regex | 2026.9.29 | 2026.9.29 | 2026.9.29 |  |
| requests | 2.34.2 | removed | — |  |
| rich | — | 15.0.0 | 15.0.0 |  |
| safetensors | 0.8.0 | 0.8.0 | 0.8.0 |  |
| sentencepiece | 0.2.2 | 0.2.2 | 0.2.2 |  |
| setuptools | 84.0.0 | 84.0.0 | 84.0.0 |  |
| shellingham | — | 1.5.4 | 1.5.4 |  |
| six | 1.17.0 | 1.17.0 | 1.17.0 |  |
| sympy | 1.14.0 | 1.14.0 | 1.14.0 |  |
| tokenizers | 0.22.2 | 0.23.3 | 0.23.3 |  |
| torch | 2.8.0 | 2.14.1 | 2.14.1 |  |
| torchvision | 0.23.0 | 0.29.1 | 0.29.1 |  |
| tqdm | 4.70.1 | 4.70.1 | 4.70.1 |  |
| transformers | 4.57.6 | 5.20.1 | 5.20.1 |  |
| triton | 3.4.0 | 3.8.0 | 3.8.0 |  |
| truststore | — | 0.10.5 | 0.10.5 |  |
| typer | — | 0.27.3 | 0.27.3 |  |
| typing-extensions | 4.16.0 | 4.16.0 | 4.16.0 |  |
| urllib3 | 2.8.0 | removed | — |  |

**What held the lock back on main, and why it no longer does.** main kept
torch at 2.8.0 because it was the newest torch with a prebuilt flash-attn
wheel. It kept transformers at 4.57.6 because upstream's code broke on 5.x
(uni-medical/UniMedVL#10). This branch drops both holds:

- `scripts/unimedvl_server.py` serves upstream's `flash_attn_varlen_func`
  import from torch's own `varlen_attn`.
- The same file puts back the one transformers 4.x RoPE helper upstream
  looks up.

See [analysis.md F2-F4](analysis.md#f2-flash-attn-held-torch-at-280).

**The 16 rows still behind** are all exact pins of their newest parent:

- torch 2.14.1 pins `cuda-toolkit[...]==13.0.3`, which pins the nvidia-* CUDA
  13.0 libraries. It also pins `nvidia-cudnn-cu13==9.24.0.43`,
  `nvidia-cusparselt-cu13==0.8.1`, `nvidia-nccl-cu13==2.30.7` and
  `nvidia-nvshmem-cu13==3.4.5`
  ([data/torch-2.14.1-requires-dist.json](data/torch-2.14.1-requires-dist.json)).
- sympy 1.14.0 needs `mpmath<1.4`
  ([data/sympy-1.14.0-requires-dist.json](data/sympy-1.14.0-requires-dist.json)).

Moving any of them would mean installing a CUDA library or an mpmath that
torch or sympy was not built and tested with, against their own metadata. No
resolver accepts that without an override, and `scripts/unimedvl-requirements.in`
says so.

## 3. GitHub Actions

All actions are pinned by commit SHA, with the release in a comment. Every
SHA was checked against its tag
([data/actions-hooks-vs-latest.md](data/actions-hooks-vs-latest.md)). None
was behind, and no workflow pins an action twice. No reusable workflow or
composite `action.yml` is tracked.

| Action | main | This branch | Latest |
| --- | --- | --- | --- |
| actions/cache | v6.1.0 | v6.1.0 | v6.1.0 |
| actions/checkout | v7.0.1 | v7.0.1 | v7.0.1 |
| actions/configure-pages | v6.0.0 | v6.0.0 | v6.0.0 |
| actions/dependency-review-action | v5.0.0 | v5.0.0 | v5.0.0 |
| actions/deploy-pages | v5.0.1 | v5.0.1 | v5.0.1 |
| actions/download-artifact | v8.0.2 | v8.0.2 | v8.0.2 |
| actions/setup-python | v7.0.0 | v7.0.0 | v7.0.0 |
| actions/upload-artifact | v7.0.2 | v7.0.2 | v7.0.2 |
| actions/upload-pages-artifact | v5.0.0 | v5.0.0 | v5.0.0 |
| codecov/codecov-action | v7.1.1 | v7.1.1 | v7.1.1 |
| docker/build-push-action | v7.4.0 | v7.4.0 | v7.4.0 |
| docker/login-action | v4.6.0 | v4.6.0 | v4.6.0 |
| docker/setup-buildx-action | v4.4.1 | v4.4.1 | v4.4.1 |
| github/codeql-action | v4.39.0 | v4.39.0 | v4.39.0 |
| lycheeverse/lychee-action | v2.9.0 | v2.9.0 | v2.9.0 |
| pypa/gh-action-pypi-publish | v1.14.2 | v1.14.2 | v1.14.2 |
| zizmorcore/zizmor-action | v0.7.1 | v0.7.1 | v0.7.1 |

## 4. Toolchains, runner images and tools pinned inside workflows and scripts

| Item | main | This branch | Latest | Note |
| --- | --- | --- | --- | --- |
| Linux runners | ubuntu-26.04 | ubuntu-26.04 | ubuntu-26.04 | |
| macOS runner | macos-26 | macos-26 | macos-26 | |
| Workflow Python | 3.15 | 3.15 | 3.15.0 | Compatibility matrix 3.11-3.15 |
| UniMedVL Python (`UNIMEDVL_PYTHON`, model-tests) | 3.13 | **3.14** | 3.15.0 | torch 2.14.1 and triton 3.8.0 publish no cp315 wheels yet ([data/latest.json](data/latest.json)). 3.13 had been the limit set by flash-attn's wheels |
| llama.cpp (`LLAMA_VERSION`) | b11707 | **b11738** | b11738 (2026-10-19 16:59 UTC) | The repository's "latest release" `v0.7.0` is a nightly marker, not a build. Hashes checked against a download ([data/llama-b11738-check.txt](data/llama-b11738-check.txt)) |
| llama.cpp GPU build (model-tests `gpu`) | CUDA 12.8 | **CUDA 13.4** | CUDA 13.4 | The newest CUDA build llama.cpp publishes. Needs an R580+ driver and a Turing+ GPU, as UniMedVL's torch 2.14.1 (CUDA 13.0) does |
| flash-attn wheel (`scripts/medbench-model-run.sh`) | v2.8.3 cu12torch2.8 cp313 | **removed** | v2.8.3.post2 | Its wheels stop at torch 2.8 and cp313 ([data/flash-attn-v2.8.3.post2-assets.json](data/flash-attn-v2.8.3.post2-assets.json)). Torch's own kernel replaces it ([F2](analysis.md#f2-flash-attn-held-torch-at-280)) |
| zizmor (via zizmor-action) | 1.31.1 | 1.31.1 | 1.31.1 | |
| pip-audit (`PIP_AUDIT_VERSION`) | 2.10.1 | 2.10.1 | 2.10.1 | |
| build, twine (release.yml) | unpinned | unpinned | 1.6.1, 7.0.0 | Installed fresh on each release |
| uv (local lock compile) | 0.13.1 | 0.13.1 | 0.13.1 | Dependabot's uv ecosystem now runs the same compile |
| Docker base images | — | — | — | No Dockerfile is tracked |

## 5. pre-commit hooks (`.pre-commit-config.yaml`)

`pre-commit autoupdate` found nothing to do.

| Hook repo | main | This branch | Latest |
| --- | --- | --- | --- |
| pre-commit/pre-commit-hooks | v6.0.0 | v6.0.0 | v6.0.0 |
| astral-sh/ruff-pre-commit | v0.17.1 | v0.17.1 | v0.17.1 |
| pre-commit/mirrors-mypy | v2.4.0 | v2.4.0 | v2.4.0 |
| python-jsonschema/check-jsonschema | 0.38.2 | 0.38.2 | 0.38.2 |

## 6. Ecosystems the issue named that have no manifest

- **Shell.** `scripts/medbench-model-run.sh` downloads llama.cpp (section 4)
  and the model files, each with a SHA-256. It also installs the UniMedVL lock
  (section 2) into its own venv with `pip install -r`. On main it also
  downloaded the flash-attn wheel; that download is gone. No other shell
  script installs anything.
- **Jinja.** Only templates, with no dependencies.

## 7. Dependabot coverage (`.github/dependabot.yml`)

| Ecosystem | Directory | main | This branch |
| --- | --- | --- | --- |
| pip | `/`, `/docs` (UniMedVL files excluded) | yes | yes |
| github-actions | `/` | yes | yes |
| pre-commit | `/` | yes | yes |
| uv | `/scripts` (UniMedVL lock) | **no**: excluded on purpose, because the torch pin had to match the flash-attn wheel | **yes** ([F7](analysis.md#f7-dependabot-could-not-update-the-unimedvl-lock)) |

This covers every manifest the issue lists, plus the one it does not list:
the UniMedVL lock.

The `uv` entry groups the lock's updates. Because Dependabot still resolves
each member separately (dependabot/dependabot-core#16390), the packages
released with torch or transformers are left unpinned in the `.in` (§2).
