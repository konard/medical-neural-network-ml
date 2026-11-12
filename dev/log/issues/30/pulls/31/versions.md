# Versions after issue #30

Every version below was looked up in its registry (PyPI JSON API, npm registry,
GitHub releases or the Terraform registry) on 2026-11-12. The raw data is in `data/`.

## Python (pyproject.toml, docs/requirements.txt, scripts/*.in/*.txt)

See `data/python-before-latest.md` for the before state and
`data/python-after-latest.md` for the after state, which lists every dependency
with its locations. Before this PR, `ruff`, `scriv`, `myst-parser`, `mlx-vlm`,
`transformers`, `sentencepiece`, `ansible-core`, `ansible-lint`, `azure-cli` and
`pytest-asyncio` (1.4.1.post1) were behind. All of them are now at the latest release.

Versions still held back, with the reason for each:

| Dependency | Locked | Latest | Reason |
| --- | --- | --- | --- |
| nvidia-* CUDA wheels, cuda-toolkit | as pinned by torch | newer | torch 2.15.0 (the latest) pins these exactly (`data/torch-2.15.0-pins.txt`) |
| mpmath | 1.3.0 | 1.4.1 | sympy 1.14.0 (the latest) requires `mpmath<1.4` |
| Lock target Python (unimedvl, infra, azure) | 3.14 | 3.15 | PyYAML has no cp315 wheel yet: yaml/pyyaml#981 |

The project itself, CI and the docs already run on Python 3.15.

## JavaScript (.github/tools, pnpm)

| Package | Before | After |
| --- | --- | --- |
| pnpm (packageManager) | 12.x | 12.18.0 |
| node engine | older | >=26.13.1 |
| semver (lock) | older | 7.8.6 |
| table (lock) | older | 6.10.1 (lodash.truncate dropped) |

`pnpm audit` reports no advisories.

## Terraform

| Item | After |
| --- | --- |
| terraform required_version | >= 1.17.2 |
| hashicorp/aws | ~> 6.73 |
| hashicorp/azurerm | ~> 5.13 |
| hashicorp/google | ~> 8.11 |

All lock files were regenerated with `terraform init -upgrade`, and `terraform validate` passes.

## GitHub Actions, tools and images

Action pins are in `data/action-pins.txt` and tool SHA-256 values in
`data/tool-checksums.txt`. The other updates:

- node 26.13.1 in release.yml
- python:3.14-slim-trixie digest refreshed
- nvidia/cuda:13.5.0-base-ubuntu24.04
- pre-commit ruff v0.17.4 (`pre-commit autoupdate` reports nothing else)

## Ruby

There are no Gemfiles or gemspecs. The only `.rb` files are archived evidence
under `dev/log/issues/25`, so there is nothing to update and no Dependabot `bundler` entry is needed.
