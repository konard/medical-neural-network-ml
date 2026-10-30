# CI investigation

Run timestamps and commit SHAs were checked before interpreting conclusions. Old Dependabot runs are historical reproductions, not evidence that the updated branch fails. Raw logs are retained under `ci-logs/`; larger logs were inspected in chunks of at most 1,500 lines.

## Historical failures linked from issue #25

| PR / run | Timestamp | Specific failure and log lines | Resolution on current branch |
| --- | --- | --- | --- |
| #13 Compatibility [38956637946](https://github.com/konard/medical-neural-network-ml/actions/runs/38956637946) | 2026-10-16 19:41–19:42 UTC | `assert 'torch==2.6.0' in ...` fails after torch updates to 2.14.1. `dependabot-13-compatibility-38956637946.log` lines 364–378 and 842–848. | Main's #22 replaced stale exact-version assertions with lock/runtime compatibility tests. This PR updates further to 2.15.0 and verifies actual upstream code. |
| #14 Compatibility [38956645024](https://github.com/konard/medical-neural-network-ml/actions/runs/38956645024) | 2026-10-16 19:42 UTC | `assert '2.0' not in <SpecifierSet('<3.0,>=1.13.0')>` in `test_project_metadata.py:26`. `dependabot-14-compatibility-38956645024.log` lines 354–369. | Main's #20 superseded the stale mypy cap/test; current floor and hook are 2.4.1. Strict checks pass. |
| #14 CI/CD [38956645281](https://github.com/konard/medical-neural-network-ml/actions/runs/38956645281) | 2026-10-16 19:43 UTC | Same obsolete mypy assertion in Python 3.13. `dependabot-14-ci-38956645281.log` lines 916–953. | Full modern CI and supported-interpreter matrix pass. |
| #16 Compatibility [38956661623](https://github.com/konard/medical-neural-network-ml/actions/runs/38956661623) | 2026-10-16 19:41–19:46 UTC | `assert 'transformers==4.49.0' in ...` fails with Transformers 5.19.0. `dependabot-16-compatibility-38956661623.log` lines 366–380 and 846–852. | Preserve the Transformers 5 migration already on main, update to 5.22.1 and replace the obsolete rotary adapter with maintained native classes. |

PR #15's packaging 26.3 and #17's NumPy 2.2.6 had passing historical checks but conflicted with subsequent main changes. The current compiled graph carries packaging 26.3 and newer NumPy 2.5.4. Merging the stale flat-requirements branches would regress the subsequent lock/model work; the combined update is reviewed in PR #26. No repository setting or disabled auto-merge caused these observed failures.

## Initial prepared-branch failure

[Security run 41077638032](https://github.com/konard/medical-neural-network-ml/actions/runs/41077638032) was created **2026-10-30T15:06:33Z** for prepared SHA **55f57ce7ee01d4a865af5c18a4589d89384bef2c**, before the implementation commit. `data/initial-failed-run.json` preserves job conclusions and timings.

Both platform audit jobs found **setuptools 84.0.0 / GHSA-xmj3-9gf7-h52w**, fixed in 84.1.0. The exact report is in `ci-logs/initial-security-41077638032.log` at lines **1162–1165** (macOS) and **3453–3456** (Linux). The aggregate pipeline then correctly failed. This was a real dependency advisory, not a timeout or stale UI result.

The regenerated UniMedVL lock installs **84.2.1**, the current PyPI release. The same pinned-surface audit now passes without exclusions, along with both Linux and native arm64 macOS package/docs audits. No test timeout was increased to hide the error.

## Implementation verification

Implementation SHA **001794a56e5d420d75631c4e34a5fc8b420e314f** was committed/pushed before all eight new runs, created **2026-10-30T15:52:40–41Z**. `data/implementation-ci-runs.json` and `implementation-ci-checks.json` record **39 successful checks, 11 intentionally skipped jobs, zero failed/pending checks**. All eight workflow conclusions are success:

| Workflow | Run |
| --- | --- |
| CI/CD Pipeline | [41084735793](https://github.com/konard/medical-neural-network-ml/actions/runs/41084735793) |
| Security | [41084735786](https://github.com/konard/medical-neural-network-ml/actions/runs/41084735786) |
| Workflows | [41084735804](https://github.com/konard/medical-neural-network-ml/actions/runs/41084735804) |
| Cloud GPU Tests | [41084735919](https://github.com/konard/medical-neural-network-ml/actions/runs/41084735919) |
| Model Tests | [41084735959](https://github.com/konard/medical-neural-network-ml/actions/runs/41084735959) |
| Docs | [41084735961](https://github.com/konard/medical-neural-network-ml/actions/runs/41084735961) |
| Broken Link Checker | [41084736338](https://github.com/konard/medical-neural-network-ml/actions/runs/41084736338) |
| Compatibility | [41084736537](https://github.com/konard/medical-neural-network-ml/actions/runs/41084736537) |

The preserved model log verifies 40 real-sample tests (line 397), 10 upstream CPU tests with one CUDA skip (717), actual MedGemma model tasks (1203–1285), two real server tests (1310), OCR results (1904–1947) and aggregate success (2195). Real checkpoint/GPU execution is not inferred from a mock unit test.

Final evidence-only changes are followed by another current-head check before marking the PR ready. Any newly failing run must be downloaded and diagnosed before completion. Main-only release/provisioning jobs and unavailable physical GPU jobs retain their intended conditions; they are not newly skipped to make an update pass.
