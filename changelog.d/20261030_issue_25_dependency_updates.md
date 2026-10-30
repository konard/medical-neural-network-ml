### Changed

- Update all Python, infrastructure and workflow dependencies to current releases,
  including torch 2.15, transformers 5.22, Terraform providers, GPU runtime
  downloads and cloud tools. Retain compatible Python 3.14 environments for
  UniMedVL and Ansible, and record upstream constraints in the dependency audit.
- Use transformers' maintained Qwen2 configuration and rotary embedding and
  torch's causal attention bias instead of local compatibility calculations.

### Fixed

- Cover all six Terraform roots with weekly Dependabot updates and test that
  newly added provider locks cannot silently lose update coverage.
- Lock and audit infrastructure Python tools, Azure CLI and secretlint's full dependency graph; use current Node.js and pnpm in CI.
