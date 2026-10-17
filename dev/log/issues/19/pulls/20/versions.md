# Pinned vs latest, per ecosystem

Every "latest" below was resolved from the registry on 2026-10-17, not from
memory: PyPI JSON API and GitHub releases/tags API
([data/latest.json](data/latest.json), produced by
[experiments/issue19_resolve_latest.py](../../../../../experiments/issue19_resolve_latest.py);
the UniMedVL lock by
[experiments/issue19_lock_vs_pypi.py](../../../../../experiments/issue19_lock_vs_pypi.py),
output in [data/unimedvl/lock-vs-pypi.tsv](data/unimedvl/lock-vs-pypi.tsv)).
"main" is `cc8e90b`, the commit the issue was opened against.

## 1. Python package (`pyproject.toml`, `docs/requirements.txt`)

The package is a library, so it declares floors, not a lock. Every floor is
now the newest release, so a fresh install and CI resolve the same tree.

| Dependency | Where | main | This branch | Latest | Note |
| --- | --- | --- | --- | --- | --- |
| Python (`requires-python`) | pyproject | `>=3.9` | `>=3.11` | 3.15 | 3.9 and 3.10 are end of life; tested on 3.11-3.15 |
| hatchling | build-system | unpinned | `>=1.32.4` | 1.32.4 | |
| ruff | dev | `>=0.8.0,<0.9` | `>=0.17.1,<0.18` | 0.17.1 | Upper bound kept on purpose (it admits the current release): ruff adds rules in minor releases, and `tests/test_project_metadata.py` keeps the pre-commit hook on the same version |
| mypy | dev | `>=1.13.0,<2.0` | `>=2.4.0` | 2.4.0 | Major crossed; upper bound dropped |
| pytest | dev | `>=8.3.0` | `>=9.1.1` | 9.1.1 | Major crossed; `minversion = "9.1"` |
| pytest-asyncio | dev | `>=0.24.0` | `>=1.4.0` | 1.4.0 | Major crossed |
| pytest-cov | dev | `>=6.0.0` | `>=7.1.0` | 7.1.0 | Major crossed |
| pre-commit | dev | `>=4.0.0` | `>=4.7.0` | 4.7.0 | |
| scriv[toml] | dev | `>=1.7.0` | `>=1.8.0` | 1.8.0 | |
| pillow | dev, imaging | `>=10.1` | `>=13.0.0` | 13.0.0 | Three majors crossed |
| mlx-vlm | mlx | `==0.7.8` (`python_version >= '3.10'`) | `==0.7.8` | 0.7.8 | Marker dropped: the floor is now 3.11 |
| sphinx | docs, docs/requirements.txt | `>=7.4` | `>=9.1.0` | 9.1.0 | Two majors crossed |
| furo | docs, docs/requirements.txt | `>=2024.8.6` | `>=2025.12.19` | 2025.12.19 | |
| myst-parser | docs, docs/requirements.txt | `>=3.0` | `>=5.1.0` | 5.1.0 | Two majors crossed |

## 2. UniMedVL runtime (`scripts/unimedvl-requirements.in` → `.txt`)

On main this was a hand-written list of 20 top-level pins with no
transitive pins. It is now a full lock (55 packages) compiled by
`uv pip compile` for Python 3.13 / `x86_64-manylinux_2_28`.

| Package | main | This branch | Latest | Why not latest |
| --- | --- | --- | --- | --- |
| Python | 3.10 | 3.13 | 3.15 | torch 2.8.0, triton 3.4.0 and the flash-attn cu12torch2.8 wheel are built for cp39-cp313 only |
| torch | 2.6.0 | 2.8.0 | 2.14.1 | Newest torch with a prebuilt flash-attn wheel (v2.8.3.post2 ships cu12torch2.4-2.8); 2.9+ needs flash-attn compiled from source on the GPU host |
| torchvision | 0.21.0 | 0.23.0 | 0.29.1 | Pinned to torch 2.8.0 |
| triton | 3.2.0 | 3.4.0 | 3.8.0 | Pinned to torch 2.8.0 |
| nvidia-* (15 packages) | not pinned | 12.8 series | 12.9 series | torch 2.8.0 requires them exactly (`==`) |
| transformers | 4.49.0 | 4.57.6 | 5.20.1 | Upstream's code fails on 5.x: `KeyError: 'default'` in `ROPE_INIT_FUNCTIONS`, and patched past that it generates different tokens ([analysis.md](analysis.md#f3-transformers-5-breaks-upstream-unimedvl)); reported as uni-medical/UniMedVL#10 |
| tokenizers | 0.21.1 | 0.22.2 | 0.23.3 | transformers 4.57.6 requires `tokenizers<=0.23.0,>=0.22.0`; no final 0.23.0 exists |
| huggingface-hub | not pinned | 0.36.2 | 2.2.0 | transformers 4.57.6 requires `huggingface-hub<1.0` |
| mpmath | not pinned | 1.3.0 | 1.4.1 | sympy (via torch) requires `mpmath<1.4` |
| accelerate | 1.7.0 | 1.15.0 | 1.15.0 | |
| safetensors | 0.4.5 | 0.8.0 | 0.8.0 | |
| sentencepiece | 0.2.1 | 0.2.2 | 0.2.2 | |
| pillow | 12.3.0 | 13.0.0 | 13.0.0 | Same version as the package's dev/imaging extras |
| numpy | 1.24.4 | 2.5.4 | 2.5.4 | Major crossed |
| opencv-python | 4.8.1.78 | 5.0.0.93 | 5.0.0.93 | Major crossed ([analysis.md](analysis.md#f4-opencv-5-changes-the-blur-by-one-grey-level)) |
| einops | 0.8.1 | 0.8.2 | 0.8.2 | |
| matplotlib | 3.7.0 | 3.11.2 | 3.11.2 | |
| contourpy | 1.3.2 | 1.4.0 | 1.4.0 | |
| cycler | 0.12.1 | 0.12.1 | 0.12.1 | |
| fonttools | 4.60.2 | 4.67.0 | 4.67.0 | |
| kiwisolver | 1.4.8 | 1.5.1 | 1.5.1 | |
| packaging | 25.0 | 26.3 | 26.3 | |
| pyparsing | 3.2.3 | 3.3.3 | 3.3.3 | |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 | 2.9.0.post0 | |
| flash-attn (wheel in `medbench-model-run.sh`) | 2.7.4.post1 cu12torch2.6 cxx11abiFALSE cp310 | 2.8.3.post2 cu12torch2.8 cxx11abiTRUE cp313 | 2.8.3.post2 | |
| All other transitive packages | not pinned | newest | newest | see [lock-vs-pypi.tsv](data/unimedvl/lock-vs-pypi.tsv), every row `latest` |

Pinned code and weights, all at the newest revision: UniMedVL code
`f51aa0a` (newest commit, 2026-06-05), weights `General-Medical-AI/UniMedVL`
`29918ce`, `unsloth/medgemma-4b-it-GGUF` `fa17ea2`.

## 3. GitHub Actions

All actions are pinned by commit SHA with the release in a comment; every
SHA was checked against the tag ([data/actions-check.md](data/actions-check.md)).

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
| codecov/codecov-action | v7.0.0 | **v7.1.1** | v7.1.1 |
| docker/build-push-action | v7.4.0 | v7.4.0 | v7.4.0 |
| docker/login-action | v4.6.0 | v4.6.0 | v4.6.0 |
| docker/setup-buildx-action | v4.4.1 | v4.4.1 | v4.4.1 |
| github/codeql-action | v4.39.0 | v4.39.0 | v4.39.0 (the newest *release* is the CLI tag `codeql-bundle-v2.27.2`; tag v4.39.0 dereferences to the pinned commit) |
| lycheeverse/lychee-action | v2.9.0 | v2.9.0 | v2.9.0 |
| pypa/gh-action-pypi-publish | v1.14.2 | v1.14.2 | v1.14.2 |
| zizmorcore/zizmor-action | v0.6.4 | **v0.7.1** | v0.7.1 |

## 4. Toolchains, runner images and tools pinned inside workflows

| Item | main | This branch | Latest | Note |
| --- | --- | --- | --- | --- |
| Linux runners | ubuntu-24.04, ubuntu-24.04-arm | ubuntu-26.04, ubuntu-26.04-arm | ubuntu-26.04 | actionlint 1.7.12 does not know the label yet; listed in `.github/actionlint.yaml` (rhysd/actionlint#682) |
| macOS runner | macos-15 | macos-26 | macos-26 | |
| Workflow Python | 3.13 | 3.15 | 3.15 | Compatibility matrix 3.11-3.15 |
| UniMedVL Python (model-tests) | 3.10 | 3.13 | 3.15 | See section 2 |
| zizmor (via zizmor-action) | 1.30.1 | 1.31.1 | 1.31.1 | |
| actionlint image | 1.7.12 digest | 1.7.12 digest | 1.7.12 | |
| llama.cpp | b11609 | b11707 | b11707 | The repository's "latest release" `v0.7.0` only carries `nightly-tag.txt` (b11669), older than b11707 |
| pip-audit (`PIP_AUDIT_VERSION`) | 2.10.1 | 2.10.1 | 2.10.1 | |
| build, twine (release.yml) | unpinned | unpinned | 1.6.1, 7.0.0 | Installed fresh on each release |
| uv (local lock compile) | — | 0.13.1 | 0.13.1 | Used to compile the UniMedVL lock |
| Docker base images | — | — | — | No Dockerfile is tracked |

## 5. pre-commit hooks (`.pre-commit-config.yaml`)

| Hook repo | main | This branch | Latest | Note |
| --- | --- | --- | --- | --- |
| pre-commit/pre-commit-hooks | v5.0.0 | v6.0.0 | v6.0.0 | Major crossed |
| astral-sh/ruff-pre-commit | v0.8.6 | v0.17.1 | v0.17.1 | Hook id `ruff` renamed to `ruff-check` |
| pre-commit/mirrors-mypy | v1.13.0 | v2.4.0 | v2.4.0 | `additional_dependencies: [pillow==13.0.0]`, the same pillow as the dev extra |
| python-jsonschema/check-jsonschema | — | 0.38.2 | 0.38.2 | New: validates `dependabot.yml` and the workflows |

## 6. Ecosystems the issue named that have no manifest

- Shell: `scripts/*.sh` install nothing through a package manager of their
  own; the pins they download (llama.cpp, the flash-attn wheel, model files)
  are in sections 2 and 4, each with a SHA-256.
- Jinja: only templates; no dependencies.
