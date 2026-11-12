### Changed

- Update all dependencies to current releases (issue #30): ruff 0.17.4,
  scriv 1.8.1, myst-parser 5.2, mlx-vlm 0.7.12, transformers 5.22.2,
  sentencepiece 0.2.3, ansible-core 2.22, ansible-lint 26.11, Azure CLI 2.92,
  Terraform 1.17.2 with AWS 6.73, AzureRM 5.13 and Google 8.11 providers,
  TFLint 0.66, k3s v1.37.2, uv 0.13.7, AWS CLI 2.40.1, gcloud 593, CUDA 13.5
  base image, Node.js 26.13.1, pnpm 12.18 and the newest GitHub Actions pins.

### Fixed

- Compare the UniMedVL CPU job's pytest-asyncio floor with pyproject.toml
  including its post-release suffix, so the test no longer fails on `1.4.1.post1`.

### Security

- Resolve CVE-2026-49272 in the Azure CLI lock by moving oauthlib to 4.0.1.
