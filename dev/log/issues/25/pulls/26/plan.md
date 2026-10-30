# Issue #25 / PR #26 work plan

Objective: fully implement and verify all issue requirements in this PR; leave genuinely blocked requirements explicitly unfinished.

- [x] Confirm prepared branch and working tree; create native goal.
- [x] Read complete issue and all issue/PR conversation, inline comments and reviews (none initially); save snapshots.
- [x] Inventory every requirement, linked Dependabot PR and required sub-issue; study recent related merged PRs.
- [x] Inventory all manifests, locks, workflow actions, hooks, runtime downloads, images, toolchains, infrastructure charts/providers/modules and shell/Jinja dependencies.
- [x] Query authoritative registries/releases before edits; save raw results and complete old/latest/new tables with reasons for every held version.
- [x] Research release notes/migration guides for major changes, existing components and upstream workaround reports; follow supplied dependency-update practices.
- [x] Capture baseline failing tests/CI logs with run timestamp and SHA; add focused reproductions for discovered bugs before fixing them.
- [x] Update Python first: floors, docs, UniMedVL lock, language/toolchain and all repeated pins. Adopt upstream features that eliminate local workarounds where applicable.
- [x] Update infrastructure: Terraform providers/modules/locks/CLI, pre-commit hooks, charts, images, Kubernetes, Ansible, cloud tools and runtime downloads.
- [x] Update every GitHub Actions reference to verified newest release SHA; synchronize repeated tool pins.
- [x] Verify automated update coverage for every ecosystem/directory and add regression checks for uncovered drift.
- [x] Search upstream issues before reporting reproducible dependency bugs/missing features; link reports and retain necessary workarounds.
- [x] Run full local lint, format, mypy, pytest, package build/distribution, Sphinx, infrastructure validation and security audits. Save large logs; respect 2.92 GiB limit and run heavy work sequentially.
- [x] Commit useful atomic steps after local checks; push only issue-25-4d7bfe8dca13; preserve history.
- [x] List fresh CI runs with timestamps/SHA; download each failed run to ci-logs, read logs in <=1500-line chunks, fix actual errors and rerun checks.
- [x] Merge latest main into this branch if needed; read final PR diff for regressions, consistency and complete issue scope.
- [x] Complete requirements/evidence matrix and research logs; update PR title/body with reproduction/tests, all upstream links and Fixes #25.
- [x] Establish finalization gate: check the evidence commit against all workflows, verify a clean tree and all acceptance criteria, then mark PR #26 ready and complete the goal. The PR check list and ready state are the live finalization record.

Assumptions: latest means newest non-prerelease available from each registry at query time; compatible older platform baselines remain supported unless an upstream dependency drops them, with reasons recorded. Historical evidence in existing dev/log and case-study data is preserved rather than rewritten.
