# Issue #30 requirements and evidence

| # | Requirement | Solution | Evidence |
| --- | --- | --- | --- |
| 1 | Fix the main CI errors left by Dependabot #27/#28/#29 without reverting them | The test regex now accepts post-release floors. The model-tests floor is now `pytest-asyncio>=1.4.1.post1` | `ci-investigation.md`, full pytest suite passes locally |
| 2 | Apply the open Dependabot PRs #13–#17 | They are superseded: numpy 2.5.4, transformers 5.22.2, packaging 26.3, mypy>=2.4.1 | `data/python-after-latest.md` |
| 3 | Update every ecosystem to latest, including toolchains | Python, pnpm/node, Terraform and providers, actions, tools, images | `versions.md` |
| 4 | Written reason for every held version | — | `versions.md` table "Versions still held back" |
| 5 | Honest constraints, regenerated locks, the same pin everywhere | Floors raised; uv, pnpm and terraform locks regenerated; the k3s pin is the same in script, Ansible and evidence | `data/inventory-after.txt` |
| 6 | Security audits are clean | pip-audit, pnpm audit; oauthlib 4.0.1 fixes CVE-2026-49272 | `data/pip-audit-after.txt` |
| 7 | Report upstream bugs | The PyYAML cp315 wheel already has reports (yaml/pyyaml#981, #951, #994), so I linked them and did not open a duplicate | `research.md` |
| 8 | Dependabot covers every ecosystem | npm, github-actions, pip, uv, pre-commit, terraform (×6); Ruby has none | `.github/dependabot.yml` |
| 9 | Checks that need a human action | Branch rulesets and the PyPI trusted publisher need an admin. Main fails with explicit instructions; on PRs the rulesets check only warns and the preflight does not run | `ci-investigation.md` |
