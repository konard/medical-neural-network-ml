# Issue #25 / pull request #26 evidence

[Issue](https://github.com/konard/medical-neural-network-ml/issues/25) · [Pull request](https://github.com/konard/medical-neural-network-ml/pull/26)

This folder preserves the complete issue/PR research, authoritative registry responses, implementation decisions and verification for the dependency update. Collection began on 2026-10-30 from commit `55f57ce`; implementation is in `001794a` plus the following evidence commit. The branch remains based on current main; no obsolete Dependabot branch was merged into main.

- [Requirement inventory and possible solutions](requirements.md): every prose requirement and linked PR, with implementation and verification evidence.
- [Versions and deliberate constraints](versions.md): complete package/action inventories, infrastructure/runtime tables and written reasons for every held version.
- [Research and upstream reports](research.md): release migrations, reusable components, reproducible workarounds and upstream proposals.
- [Verification](validation.md): local commands/results, meaningful test limits and hosted CI evidence.
- [CI investigation](ci-investigation.md): historical and initial failures with timestamps, SHAs, exact errors and log line references.
- [Work checklist](plan.md), [raw evidence](data/) and [downloaded CI logs](ci-logs/).

Reusable, bounded reproductions and registry collection scripts are in `experiments/issue25_*.py`. Only metadata and tiny model inputs are used locally; the 29 GB UniMedVL checkpoint is not loaded under this container's 2.92 GiB limit. The existing real-weight CPU smoke test runs on hosted CI. Paid cloud and physical GPU jobs retain their existing trusted-main/manual access controls.

`.log` files are intentionally committed here as research evidence despite the repository-wide log ignore rule. Failed candidate installs are clearly separated from final successful validation in the documents; they are not claims about the final dependency tree.
