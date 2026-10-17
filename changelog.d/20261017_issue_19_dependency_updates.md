### Removed

- Support for Python 3.9 and 3.10, both past end of life. The package now requires Python 3.11 or newer and is tested on 3.11 through 3.15. The `tomllib` fallbacks for older Pythons are gone.

### Changed

- Updated all dependencies to their latest releases (issue #19). Development tools: ruff 0.17, mypy 2.4, pytest 9.1, pytest-asyncio 1.4, pytest-cov 7.1, pre-commit 4.7, scriv 1.8 and pillow 13. Documentation: sphinx 9.1, furo 2025.12.19 and myst-parser 5.1. Build: hatchling 1.32. Pre-commit hooks: pre-commit-hooks v6, ruff-pre-commit v0.17.1 (the hook id is now `ruff-check`) and mirrors-mypy v2.4.0.
- CI now runs on `ubuntu-26.04`, `ubuntu-26.04-arm` and `macos-26` with Python 3.15. It uses codecov-action v7.1.1, zizmor-action v0.7.1 (zizmor 1.31.1) and llama.cpp build b11707.
- Code changes required by the new ruff rules: strict `zip()` and `datetime.UTC`.
- UniMedVL runtime (`scripts/unimedvl-requirements.txt`): torch 2.6.0 → 2.8.0, transformers 4.49.0 → 4.57.6, accelerate 1.7.0 → 1.15.0, numpy 1.24.4 → 2.5.4, opencv-python 4.8 → 5.0 and every other package at its newest release. It now runs on Python 3.13 instead of 3.10, which reaches end of life this month; 3.13 is the newest Python that torch 2.8.0 and the flash-attn wheel are built for. The file is now a full lock compiled with `uv pip compile` from the new `scripts/unimedvl-requirements.in`, which records why torch and transformers stop where they do. The flash-attn wheel is v2.8.3.post2 (cu12torch2.8, C++11 ABI, CPython 3.13), which fixes GHSA-7g5w-pq96-8c5w. Thirty-three of the 47 accepted torch, transformers and accelerate advisories no longer apply and are removed from `scripts/audit_dependencies.py`.
- Dependabot now opens one weekly pull request for the Python packages and the pre-commit hooks together, because the hooks pin the same ruff, mypy and pillow as the `dev` extra. It no longer opens pull requests for the UniMedVL lock. A new `check-jsonschema` pre-commit hook validates `.github/dependabot.yml` and the workflows.

### Fixed

- `wilson_interval` now returns an upper bound of exactly 1 when every answer is right, and a lower bound of exactly 0 when every answer is wrong. The closed form reached these bounds only up to rounding: on the new macOS 26 runner with Python 3.15 it returned 0.9999999999999999 for 4 out of 4.
