# Issue #19 / PR #20: update all dependencies in all languages

- Issue: <https://github.com/konard/medical-neural-network-ml/issues/19>
- Pull request: <https://github.com/konard/medical-neural-network-ml/pull/20>

| File | Contents |
| --- | --- |
| [issue.md](issue.md) | Snapshot of the issue text |
| [requirements.md](requirements.md) | Every requirement, its status and evidence, what could not be done, assumptions |
| [versions.md](versions.md) | Per-ecosystem tables: version on `main`, on this branch, latest on the registry, reason when not latest |
| [analysis.md](analysis.md) | Findings F1-F9: symptom, root cause, fix commit, test; what was ruled out |
| [research.md](research.md) | Sources read, the migration notes for each major version, best practices applied, tools considered, upstream reports |
| [data/](data/) | Registry lookups, action SHA checks, the Dependabot alert evaluation, UniMedVL smoke-test, OpenCV and audit output |
| [ci-logs/](ci-logs/) | Log excerpt of the one CI failure the update caused (F1) |

## Outcome

- **Every ecosystem is current.**
  - Python package: every floor in `pyproject.toml` and `docs/requirements.txt`
    is the newest release.
  - GitHub Actions and the pre-commit hooks are at their newest releases.
  - Runners moved to `ubuntu-26.04` and `macos-26`, and the workflow Python to
    3.15.
  - llama.cpp is at b11707.
  - The UniMedVL runtime is now a compiled 55-package lock on Python 3.13.
- **Left behind, with a written reason.** These are torch 2.8.0 (the newest
  with a prebuilt flash-attn wheel) and transformers 4.57.6 (upstream's code
  breaks on 5.x, reported as uni-medical/UniMedVL#10). Four more follow
  from those two pins: tokenizers, huggingface-hub, mpmath and the UniMedVL
  Python. See [versions.md §2](versions.md#2-unimedvl-runtime-scriptsunimedvl-requirementsin--txt).
- **Bugs the update exposed and fixed.**
  - `wilson_interval` missed its exact bounds (F1).
  - The flash-attn wheel had the wrong C++ ABI for the new torch (F5).
- **Advisories.** The Dependabot alerts open on `main` drop from 25 to 8. The
  8 are in the two held pins, and each is accepted with a reason. `pip-audit`
  reports nothing else (F7).
- **Not verifiable here.** No GPU runner exists, so UniMedVL could not be run
  on a GPU. A CPU smoke test checked that the generated tokens are unchanged,
  and the flash-attn extension imports.
