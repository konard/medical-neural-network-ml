# Issue #30 / pull request #31 evidence

[Issue](https://github.com/konard/medical-neural-network-ml/issues/30) · [Pull request](https://github.com/konard/medical-neural-network-ml/pull/31)

Collected on 2026-11-12 from `main` at `8ee4c93` (after Dependabot #27, #28, #29 were merged without waiting for CI).

- [requirements.md](requirements.md): every requirement of the issue, possible solutions, the chosen one and its verification.
- [versions.md](versions.md): before/latest/after tables per ecosystem, with a written reason for every held version.
- [ci-investigation.md](ci-investigation.md): the failing `main` checks, their root causes and fixes.
- [research.md](research.md): changelogs reviewed, reusable components, upstream reports.
- [data/](data/): raw registry responses (PyPI, GitHub releases), checksums, action SHAs, audits, the issue text.
- [ci-logs/](ci-logs/): downloaded logs of the failing `main` jobs.

Reproduce the collection: `python3 experiments/issue30_inventory.py --output <dir>` (PyPI + GitHub releases) and
`bash experiments/issue30_tool_checksums.sh` (SHA-256 of every pinned binary download).
