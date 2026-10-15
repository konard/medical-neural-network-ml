# Requirements of issue #4 and how PR #5 meets them

Source: [issue.md](issue.md) (the issue had no comments when this was written).

| # | Requirement (from the issue text) | Status | Where |
| --- | --- | --- | --- |
| R1 | Find every false positive, false negative, warning and error in CI/CD | Done | [analysis.md](analysis.md), [ci-logs/](ci-logs/) |
| R2 | Fix them all | Done, except two repository settings that only a maintainer can change (R2a, R2b) | [analysis.md](analysis.md) |
| R3 | Compare the full file tree of all workflow and CI/CD script files with the Python template | Done | [template-comparison.md](template-comparison.md) |
| R4 | Reuse every best practice from the template | Done; three practices already proposed upstream (#105, #106, #107) were adopted here | [template-comparison.md](template-comparison.md) |
| R5 | If the same problem exists in the template, report it there | Done | [template-comparison.md](template-comparison.md#reported-upstream) |
| R6 | Follow hive-mind's CI-CD-BEST-PRACTICES.md | Done; each fix names the practice it follows | [analysis.md](analysis.md), [research.md](research.md) |
| R7 | Do everything in this single pull request | Done: PR #5 only | |
| R8 | (Solver instructions) Collect the data into `dev/log/issues/4/pulls/5`, research online, list existing tools | Done | this folder, [research.md](research.md) |
| R9 | (Solver instructions) Apply each fix everywhere the problem occurs | Done; cross-workflow tests prevent regressions in any workflow | [analysis.md](analysis.md#tests) |

## R2a and R2b: what a maintainer must do

Under best practice "a check that needs a human action passes on the pull request and fails on `main` with an explicit message", both checks below now warn on pull requests and fail on `main` with the exact settings to change. No commit can make them green.

- **R2a: PyPI trusted publisher (CI/CD Pipeline, `Release Preflight`).** PyPI answers `422 invalid-publisher` because no trusted publisher exists for this repository. `pyproject.toml` still has the template name `my-package`, which is an existing PyPI project owned by someone else. Steps:
  1. Rename `[project].name` in `pyproject.toml`, for example to `medbench` (the CLI name; `https://pypi.org/pypi/medbench/json` answered 404, so the name was free on 2026-10-15).
  2. Add a pending publisher with owner `konard`, repository `medical-neural-network-ml`, workflow `release.yml` and a blank environment at <https://pypi.org/manage/account/publishing/>.
- **R2b: Dependency graph (Security, `Dependency Graph Enabled`).** Enable it at <https://github.com/konard/medical-neural-network-ml/settings/security_analysis> → Dependency graph → Enable. Until then, dependency review cannot run.

The PR deliberately does not do either step. Both are outward-facing: one publishes under a name on PyPI, the other changes a repository setting.

## Assumptions

- "False negative" means a check that is green, or never runs, although it should fail. "False positive" means a check that fails for something no commit can fix, or that reports a problem that is not there. Warnings count even when the job is green.
- Notices that the workflows print on purpose to explain an opt-in that was skipped are not defects. They tell the maintainer how to opt in. Examples: "Skipping Codecov upload because CODECOV_TOKEN is not configured" and "Skipping GitHub Pages deployment...".
- Log lines that only contain the word "warning" are not warnings either. Examples are the CodeQL schema dump (`"DEPRECATED - Whether to use buildless..."`) and `--warning=no-unknown-keyword`. They are listed in [analysis.md](analysis.md#not-defects) with the reason.
