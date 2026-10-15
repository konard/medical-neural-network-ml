# Issue #4 / PR #5: false positives, false negatives, warnings and errors in CI/CD

- Issue: <https://github.com/konard/medical-neural-network-ml/issues/4>
- Pull request: <https://github.com/konard/medical-neural-network-ml/pull/5>

| File | Contents |
| --- | --- |
| [issue.md](issue.md) | Snapshot of the issue text |
| [requirements.md](requirements.md) | Every requirement, its status, the two maintainer actions, assumptions |
| [analysis.md](analysis.md) | Findings F1-F10: symptom, root cause, fix commit, test; what was ruled out |
| [research.md](research.md) | Online sources behind each fix, best practices applied, existing tools considered |
| [template-comparison.md](template-comparison.md) | File-by-file comparison with the Python template, issues reported upstream |
| [ci-logs/](ci-logs/) | Excerpts of the logs cited above (ANSI colours stripped) |

## Outcome

On `main` at `fd7703e`, two runs failed and two passed with warnings. On this branch:

- **Fixed in code (8).** mypy deprecation (F3), lychee glob warning (F4), offline zizmor warning (F5), `.sh`/`.json` change detection (F6), model-tests path filter (F7), empty-`dist/` upload (F8), cancelled pending releases (F9) and outdated action majors (F10). Each has a regression test.
- **Turned from "unknown" into an actionable failure (1).** The release preflight now says exactly which trusted publisher to add on PyPI (F1).
- **Needs a maintainer (2).** The PyPI trusted publisher with a project rename (R2a), and enabling the Dependency graph (R2b). On pull requests both are warnings; on `main` both fail with the setting to change. No commit can make them green. See [requirements.md](requirements.md#r2a-and-r2b-what-a-maintainer-must-do).
- **Reported to the template.** Five new issues (#108-#112) and two comments (#105, #107). See [template-comparison.md](template-comparison.md#reported-upstream).
