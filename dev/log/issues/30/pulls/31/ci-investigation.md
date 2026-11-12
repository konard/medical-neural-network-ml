# CI on `main` after the Dependabot merges

Runs at `8ee4c93`/`c4e41dc` (2026-11-12, listed in the issue comment). Logs in [ci-logs/](ci-logs/).

| Check | Job log | Error | Root cause | Fix |
| --- | --- | --- | --- | --- |
| CI/CD `Test (Python 3.15)`, Compatibility 3.11–3.15 | `job-130991781366.log:488`, `job-130992969198.log:372` | `assert 'pytest-asyncio>=1.4.1' in [... 'pytest-asyncio>=1.4.1.post1' ...]` | #28 raised the dev floor to `1.4.1.post1` in `pyproject.toml`, but `model-tests.yml` (unimedvl-cpu) still installed `>=1.4.1`, and the test's regex `[\d.]+` cut `.post1` off | Workflow pin brought to `1.4.1.post1`; regex now `[\w.]+` and asserts both pytest requirements are found (`tests/test_medbench_model_run.py`) |
| Security `Audit dependencies` (ubuntu-26.04, macos-26) | `job-130991707854.log:536-546` | `oauthlib 4.0.0 CVE-2026-49272, fix 4.0.1` | New advisory against the Azure CLI lock's transitive oauthlib | `scripts/azure-cli-requirements.txt` recompiled with `--upgrade`: oauthlib 4.0.1 (plus Azure CLI 2.92.0) |
| Security `Branch Rulesets` | `job-130991707724.log:171-174` | `ruleset 'main-checks' is not applied ...` | Repository setting; needs an admin token. On `main` it already fails with the explicit instruction `python scripts/repository_rulesets.py apply --yes` (docs/repository-policy.md). | **Needs human action** (blocked for the solver: no admin rights). Runs on pull requests too, but `scripts/repository_rulesets.py` only warns there (lines 189-197), so the PR stays green. |
| CI/CD `Release Preflight (credentials)` | `job-130991707847.log:156` | `PyPI refused to mint an upload token (422 invalid-publisher)` | PyPI trusted publisher not configured | **Needs human action**: add the pending publisher (docs/publishing.md). Only runs on `main`, already fails with the explicit instruction. |

Dependabot #27 (python-dateutil 4.0.0) and #29 (zizmor-action 0.8.1) had only the audit failure above in common; #28 introduced the pytest-asyncio mismatch.
