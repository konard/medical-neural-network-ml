# Requirements of issue #19 and how PR #20 meets them

Source: [issue.md](issue.md). The issue had no comments, and the pull request
had no reviews or comments, when this was written.

| # | Requirement (from the issue text) | Status | Where |
| --- | --- | --- | --- |
| R1 | Python: update `pyproject.toml` and `docs/requirements.txt`, raise floors, drop upper bounds that exclude the current release | Done | [versions.md §1](versions.md#1-python-package-pyprojecttoml-docsrequirementstxt), 10dd4b3 |
| R2 | GitHub Actions: every `uses:` at its newest release | Done; two actions were behind (codecov v7.1.1, zizmor-action v0.7.1), every SHA checked against its tag | [versions.md §3](versions.md#3-github-actions), [data/actions-check.md](data/actions-check.md), 3fed030 |
| R3 | Infrastructure as code: `.pre-commit-config.yaml` (`pre-commit autoupdate`) | Done | [versions.md §5](versions.md#5-pre-commit-hooks-pre-commit-configyaml), 10dd4b3 |
| R4 | Shell and Jinja: check whether they pull dependencies through another ecosystem | Done: the shell scripts download llama.cpp, the flash-attn wheel and the UniMedVL runtime; all three were updated. Jinja has no dependencies | [versions.md §2, §4, §6](versions.md), b210bfd, b1b464c, dbd7ad4 |
| R5 | `dependabot.yml` covers every ecosystem and directory, kept in sync | Done: pip (`/`, `/docs`), github-actions and pre-commit are covered. The UniMedVL lock is excluded on purpose, and the Security workflow still audits it | [analysis.md F9](analysis.md#f9-dependabot-could-not-update-the-unimedvl-stack-safely), c1cb53c |
| R6 | "All" is literal: dev deps, transitive lock entries, build plugins, test runners, linters, base images, actions, toolchains | Done: hatchling, the test and lint tools, the 55-package lock, runner images, workflow Python, the UniMedVL Python and llama.cpp. No Dockerfile or base image is tracked | [versions.md](versions.md) |
| R7 | A table per ecosystem: pinned today vs released today, from the registry | Done | [versions.md](versions.md), [data/latest.json](data/latest.json), [data/unimedvl/lock-vs-pypi.tsv](data/unimedvl/lock-vs-pypi.tsv) |
| R8 | A written reason for anything left behind | Done: torch 2.8.0 (and torchvision, triton, nvidia-*), transformers 4.57.6, tokenizers, huggingface-hub, mpmath, the UniMedVL Python 3.13, and the ruff `<0.18` cap | [versions.md §1-2](versions.md), `scripts/unimedvl-requirements.in` |
| R9 | Cross majors deliberately: read changelogs, adapt code, remove shims | Done | [research.md "Major versions"](research.md#major-versions), [analysis.md F3, F4, F8, F10-F13](analysis.md) |
| R10 | Use new features; delete hand-rolled copies | Done | [research.md "New features adopted"](research.md#new-features-adopted) |
| R11 | Honest constraints; regenerate and commit every lockfile | Done: every floor in `pyproject.toml` is the newest release. The UniMedVL file went from 20 hand-written pins to a compiled 55-package lock. The package itself is a library and has no lock | [versions.md §1-2](versions.md) |
| R12 | One version per dependency across all pins | Done: pillow 13.0.0 in the dev and imaging extras, the mypy hook and the UniMedVL lock. ruff and mypy match between the hooks and the dev extra. The docs pins match between `pyproject.toml` and `docs/requirements.txt`. One Python is used for the UniMedVL venv, CI, the lock and the wheel. Tests enforce all of these | `tests/test_project_metadata.py`, `tests/test_medbench_model_run.py::test_unimedvl_python_matches_everywhere` |
| R13 | Full build, test and lint; CI green; no new deprecation warnings | Done | [research.md "Verification"](research.md#verification), [data/verification/](data/verification/), [analysis.md F14](analysis.md#f14-unclosed-http-error-answers-on-python-312), CI on the final commit |
| R14 | Audit the shipped tree; leave no advisory unresolved | Done: the alert count drops from 25 to 8. Those 8 are in pins held for R8, and each is accepted with a reason. `pip-audit` reports none unaccepted | [analysis.md F7](analysis.md#f7-dependabot-alerts-25-open-on-main-8-left), [data/dependabot-alerts-eval.txt](data/dependabot-alerts-eval.txt), [data/unimedvl/pip-audit-lock.log](data/unimedvl/pip-audit-lock.log) |
| R15 | Report the workarounds' causes upstream (search first, MRE, versions, workaround, suggested fix) and link them | Done: uni-medical/UniMedVL#10 was opened. actionlint #682, #761 and #654 already exist, as does a comment carrying this exact workaround, so nothing new was added | [research.md "Upstream reports"](research.md#upstream-reports) |
| R16 | Follow hive-mind's DEPENDENCY-UPDATE-BEST-PRACTICES.md | Done; each principle is mapped to the work | [research.md "Best practices"](research.md#best-practices) |
| R17 | Everything in this single pull request | Done: PR #20 only. Dependabot PRs #13-#18 are superseded | |
| R18 | (Solver instructions) Collect the data into `dev/log/issues/19/pulls/20`, research online, list existing tools | Done | this folder, [research.md](research.md) |
| R19 | (Solver instructions) Apply each requirement everywhere it occurs | Done: one table row per pin location. Tests stop the pins from drifting apart again (R12) | [versions.md](versions.md) |

## What could not be done

- **A GPU run of UniMedVL on the new stack.** No GPU runner is registered for
  this repository, so `Model Tests` skips the UniMedVL job. What was verified
  instead:
  - upstream's `chat()` on the CPU with a tiny random model generates the same
    tokens as on the old pins;
  - the flash-attn extension imports against torch 2.8.0;
  - `pip-audit` passes on the lock.

  See [analysis.md F2](analysis.md#f2-python-310-held-seven-unimedvl-packages-behind).
  The first GPU run after merge is the remaining check.

## Assumptions

- "Latest" means the newest final release on the registry on 2026-10-17, not
  pre-releases. For actions it means the newest release tag.
- `pyproject.toml` describes a library, so it uses floors, not exact pins.
  "Regenerate every lockfile" applies to the one real lock, the UniMedVL
  runtime.
