# Requirements of issue #21 and how PR #22 meets them

Source: [issue.md](issue.md). When this was written, the issue had no
comments, and the pull request had no reviews or comments.

| # | Requirement (from the issue text) | Status | Where |
| --- | --- | --- | --- |
| R1 | Apply Dependabot PRs #13-#17 in this pull request | Done. #14, #15 and #17 were already on `main` through PR #20. #13 and #16 (the UniMedVL torch and transformers updates) are applied here, at newer versions than they asked for | [versions.md §1](versions.md#1-python-package-pyprojecttoml-docsrequirementstxt), 11dad2f |
| R2 | Fix what made them unmergeable: a failing check, a conflict, or a repository setting | Done. #13 and #16 failed because `test_unimedvl_pins` asserted exact versions, and #14 failed on the old mypy bound. All five conflicted with PR #20. The pin tests now check what the code needs, and a CPU job runs upstream's code on every lock change. Auto-merge is off in the repository settings, but that did not block these pull requests (see Assumptions) | [analysis.md F1](analysis.md#f1-why-dependabots-pull-requests-13-17-could-not-be-merged), [ci-logs/](ci-logs/), 11dad2f, f4b6f0b |
| R3 | Python: update `pyproject.toml` and `docs/requirements.txt`, raise floors, drop upper bounds that exclude the current release | Done. Every floor is the newest release. mlx-vlm moved 0.7.8 → 0.7.9. ruff's `<0.18` admits the current release and is kept, with a reason | [versions.md §1](versions.md#1-python-package-pyprojecttoml-docsrequirementstxt), b000ac3 |
| R4 | GitHub Actions: every `uses:` at its newest release | Done. All 17 actions are at their newest tag, and every SHA was checked against its tag | [versions.md §3](versions.md#3-github-actions), [data/actions-hooks-vs-latest.md](data/actions-hooks-vs-latest.md) |
| R5 | Infrastructure as code: `.pre-commit-config.yaml` | Done. Every hook is at its newest tag | [versions.md §5](versions.md#5-pre-commit-hooks-pre-commit-configyaml) |
| R6 | Shell and Jinja: check whether they pull dependencies through another ecosystem | Done. The shell scripts download llama.cpp (now b11738, CUDA 13.4) and install the UniMedVL lock. They no longer download a flash-attn wheel. Jinja is only scriv's `changelog.d/fragment_template.md.j2`, which pulls nothing (scriv is in the `dev` extra) | [versions.md §4, §6](versions.md#4-toolchains-runner-images-and-tools-pinned-inside-workflows-and-scripts), 026c803, 3f2739d |
| R7 | `dependabot.yml` covers every ecosystem and directory and stays in sync | Done. pip (`/`, `/docs`), github-actions and pre-commit were already covered. The UniMedVL lock, the one real lockfile, is now updated by Dependabot's `uv` ecosystem (`/scripts`) instead of being excluded. torchvision, triton and tokenizers are named without a version in the `.in`, because Dependabot updates group members one at a time and lockstep pins would never move | [versions.md §7](versions.md#7-dependabot-coverage-githubdependabotyml), [analysis.md F7](analysis.md#f7-dependabot-could-not-update-the-unimedvl-lock), 853e944 |
| R8 | "All" is literal: dev deps, transitive lock entries, build plugins, test runners, linters, base images, actions, toolchains | Done. This covers the 68-package lock, hatchling, the test and lint tools, runner images, the workflow Python (3.15), the UniMedVL Python (3.13 → 3.14), llama.cpp, the GPU build's CUDA (12.8 → 13.4) and the security tools. No Dockerfile or base image is tracked | [versions.md](versions.md) |
| R9 | A table per ecosystem: pinned today vs released today, from the registry | Done | [versions.md](versions.md), [data/python-pins-vs-latest.md](data/python-pins-vs-latest.md), [data/unimedvl-lock-vs-latest.md](data/unimedvl-lock-vs-latest.md), [data/latest.json](data/latest.json) |
| R10 | A written reason for anything left behind | Done. Sixteen lock rows are behind, because torch 2.14.1 pins its CUDA 13.0 libraries, cuDNN, cuSPARSELt, NCCL and NVSHMEM exactly, and sympy 1.14.0 needs `mpmath<1.4`. torch's own requirement data is saved, and the reasons are also in `scripts/unimedvl-requirements.in`. UniMedVL stays on Python 3.14 because torch and triton publish no cp315 wheels | [versions.md §2](versions.md#2-unimedvl-runtime-scriptsunimedvl-requirementsin--txt), [data/torch-2.14.1-requires-dist.json](data/torch-2.14.1-requires-dist.json), [data/sympy-1.14.0-requires-dist.json](data/sympy-1.14.0-requires-dist.json) |
| R11 | Cross majors deliberately: read changelogs, adapt code, remove shims | Done. The majors were torch 2.8 → 2.14, transformers 4 → 5, huggingface-hub 0.36 → 2, CUDA 12 → 13 and opencv 5 (already on `main`). The flash-attn wheel and its ABI check are removed | [research.md "Major versions"](research.md#major-versions), [analysis.md F2, F3, F8, F12, F13](analysis.md) |
| R12 | Use new features and delete hand-rolled copies | Done. torch's `varlen_attn` replaces the flash-attn wheel and its download, hash and ABI check. Dependabot's `uv` ecosystem replaces hand-updating the lock. The 14 accepted advisories and the `allow-ghsas` list are deleted | [research.md "New features adopted"](research.md#new-features-adopted) |
| R13 | Honest constraints; regenerate and commit every lockfile | Done. The lock is recompiled the way Dependabot compiles it, and every floor is the newest release | [versions.md §1-2](versions.md), `experiments/issue21_dependabot_uv_compile.sh` |
| R14 | One version per dependency across all pins | Done. All 80 names pinned in `pyproject.toml`, `docs/requirements.txt`, the lock and the hooks agree. The UniMedVL Python is 3.14 everywhere, and llama.cpp is b11738 in the script, the workflow and `support.py` | `tests/test_project_metadata.py`, `tests/test_medbench_model_run.py` |
| R15 | Full build, tests and lint; CI green; no new deprecation warnings | Done. This covers pytest, the UniMedVL CPU tests, pre-commit, Sphinx, the build and every CI workflow. transformers 5's one new warning comes from upstream's code and is filtered narrowly (F13) | [research.md "Verification"](research.md#verification), [data/verification/](data/verification/), CI on the final commit |
| R16 | Audit the tree afterwards; leave no advisory unresolved | Done. `pip-audit` finds none. The 8 open Dependabot alerts are fixed by the new pins, and nothing is accepted any more | [analysis.md F5](analysis.md#f5-14-accepted-advisories-and-8-open-dependabot-alerts-on-main), [data/pip-audit-unimedvl.txt](data/pip-audit-unimedvl.txt) |
| R17 | Report upstream (search first, MRE, versions, workaround, suggested fix) and link the reports | Done. A correction was posted on uni-medical/UniMedVL#10, with the transformers 5 changes and a suggested patch. The cp314 and torch ≥ 2.9 wheel gap was added to Dao-AILab/flash-attention#2995, which already tracks it. pytorch/pytorch#201457 was opened for `varlen_attn`'s missing CPU path, the reason the shim keeps its own fallback. dependabot/dependabot-core#16390 got a reproduction showing that the `uv pip compile` path also updates group members one at a time, the reason torchvision, triton and tokenizers are unpinned in the `.in` | [research.md "Upstream reports"](research.md#upstream-reports) |
| R18 | Follow hive-mind's DEPENDENCY-UPDATE-BEST-PRACTICES.md | Done. Each principle is mapped to the work | [research.md "Best practices"](research.md#best-practices) |
| R19 | Everything in this single pull request | Done: PR #22 only | |
| R20 | (Solver instructions) Collect the data into `dev/log/issues/21/pulls/22`, research online, list existing tools | Done | this folder, [research.md](research.md) |
| R21 | (Solver instructions) Apply each requirement everywhere it occurs | Done. The tables have one row per pin location, and tests keep the pins from drifting apart | [versions.md](versions.md) |

## What could not be done

- **A GPU run of UniMedVL and llama.cpp on CUDA 13.** No GPU runner is
  registered for this repository, so `Model Tests` skips the GPU jobs. What
  was verified instead:
  - Upstream's `chat()` runs on the CPU in CI on the new lock and matches
    the golden tokens and logits.
  - The attention shim's fallback matches a reference implementation, and
    the shim's call to `varlen_attn` matches torch's 2.12-2.14 signature.
  - The new llama.cpp archives pass their hash check, and `llama-server`
    reports build 11738.
  - The CUDA backend resolves its runtime beside itself.

  `test_cuda_uses_varlen_attn_and_matches_the_reference` runs on the first
  GPU run. See [analysis.md F2](analysis.md#f2-flash-attn-held-torch-at-280)
  and [F8](analysis.md#f8-cuda-13-raises-the-gpu-runners-driver-and-gpu-floor).
- **mlx-vlm 0.7.9 on Apple silicon.** No Apple silicon runner is available.
  The backend's calls were checked against 0.7.9's source
  ([data/mlx-vlm-0.7.9-api.txt](data/mlx-vlm-0.7.9-api.txt)), and the mocked
  tests pass (F9).

## Assumptions

- "Latest" means the newest final release on 2026-10-19, not pre-releases.
  For actions and hooks it means the newest release tag.
- `pyproject.toml` describes a library, so it uses floors, not exact pins.
  "Regenerate every lockfile" applies to the one real lock, the UniMedVL
  runtime.
- Turning on auto-merge (`allow_auto_merge` is `false`) is left to the owner.
  It was not why #13-#17 failed (R2). Under `docs/repository-policy.md`, Dependabot's pull requests
  need the same checks and review as any other.
- Two checks fail only on `main`, "Release Preflight (credentials)" and
  "Branch Rulesets". Each needs an admin action that its failure message
  names, and they pass on pull requests by design (F14). This pull request
  does not change them.
- Dependabot closes #13-#17 by itself when it sees their updates on `main`.
  Each one has a comment pointing to this pull request.
