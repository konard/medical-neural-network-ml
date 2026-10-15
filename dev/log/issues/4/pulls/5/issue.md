# Issue #4 (snapshot, fetched 2026-10-15)

- Title: Check for all false positives, false negatives, warnings and errors in CI/CD and fix them all
- Author: konard
- Created: 2026-10-15T11:10:13Z
- URL: https://github.com/konard/medical-neural-network-ml/issues/4
- Comments at the time of the snapshot: none

---

### Recent CI/CD runs on `main`

| Workflow | Status | Conclusion | Commit | Run |
| --- | --- | --- | --- | --- |
| Docs | completed | success | `fd7703e` | [run](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259366) |
| Broken Link Checker | completed | success | `fd7703e` | [run](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259357) |
| CI/CD Pipeline | in_progress | in_progress | `fd7703e` | [run](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259374) |
| Security | in_progress | in_progress | `fd7703e` | [run](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259420) |
| Workflows | completed | success | `fd7703e` | [run](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259341) |

Use all the best practices from CI/CD templates (check full file tree to compare for all GitHub workflow and CI/CD scripts file), if the same issue is found in template report issue also in templates:

- https://github.com/link-foundation/python-ai-driven-development-pipeline-template

We should compare all files, so we don't have more CI/CD errors in the future and reuse all the best practices from these templates.

Follow the CI/CD best practices collected in [https://github.com/link-assistant/hive-mind/blob/main/docs/CI-CD-BEST-PRACTICES.md](https://github.com/link-assistant/hive-mind/blob/main/docs/CI-CD-BEST-PRACTICES.md).

Please plan and execute everything in this single pull request, you have unlimited time and context, as context auto-compacts and you can continue indefinitely, until it is each and every requirement fully addressed, and everything is totally done.

---

<details>
<summary>Context collected by <code>/fix --ci-cd</code></summary>

- **Repository:** [konard/medical-neural-network-ml](https://github.com/konard/medical-neural-network-ml)
- **Default branch:** `main`
- **Latest commit:** `fd7703e` ([commit](https://github.com/konard/medical-neural-network-ml/commit/fd7703ea46703178ebfcc1ffb23ebbed9baf2c20)) — Merge pull request #2 from konard/issue-1-0b27055a11ec
- **CI/CD runs found:** 5 (0 not passing)

**Detected languages**

- **Python** — 91.5%
- **Shell** — 8.4%
- **Jinja** — 0.1%

**Recommended CI/CD templates**

Apply the best practices from these templates, in priority order (most-used language first):

1. **Python** — [link-foundation/python-ai-driven-development-pipeline-template](https://github.com/link-foundation/python-ai-driven-development-pipeline-template) _(detected: Python)_

Other detected languages without a dedicated template: Shell, Jinja.

</details>
