# Registry inventory and deliberate constraints

Registries were queried on **2026-10-30 UTC**, starting from prepared commit `55f57ce7ee01d4a865af5c18a4589d89384bef2c`. Stable releases are compared with stable releases; alpha/beta/RC builds and unreleased branches are excluded. Archive checksums and image index digests were verified before changing executable pins. Historical logs and case-study results retain their original versions.

The complete package tables are split by ecosystem so every transitive entry remains reviewable:

- [Original direct Python pins versus PyPI](data/python-before-latest.md), with original locations and [raw metadata](data/python-registry.json).
- [Every resolved Python package, before/now/latest and exact parent constraint when held](data/resolved-python-versions.md): UniMedVL, infrastructure, Azure CLI, Linux package/dev/docs and arm64 macOS package/dev/docs/MLX. [Registry metadata](data/resolved-python-registry.json) includes the installed versions' dependency bounds. New infrastructure/Azure locks replace formerly unlocked transitive graphs.
- [Every JavaScript and pnpm platform executable](data/javascript-versions.md), with [npm metadata](data/javascript-registry.json). Both pnpm 12 lock documents are included: package manager and application graph.
- [Original Actions/hooks](data/actions-hooks-before-latest.md) and [final Actions/hooks](data/actions-hooks-after-latest.md). The latter verifies each release tag's actual commit, including annotated tags, and every repeated reference. All 26 action repositories and four hook repositories are current. The actionlint container is separately listed below.

## Direct project floors

| Dependency | Before | Now | Registry latest | Notes |
| --- | --- | --- | --- | --- |
| hatchling | >=1.32.4 | >=1.33.0 | 1.33.0 | Build backend |
| ruff | >=0.17.1,<0.18 | >=0.17.3,<0.18 | 0.17.3 | Hook synchronized; current release allowed |
| mypy | >=2.4.0 | >=2.4.1 | 2.4.1 | Hook synchronized; strict checks retained |
| pytest | >=9.1.1 | >=9.2.0 | 9.2.0 | Native strict TOML configuration retained |
| pytest-asyncio | >=1.4.0 | >=1.4.1 | 1.4.1 | Function loop scope retained |
| pytest-cov | >=7.1.0 | >=7.1.0 | 7.1.0 | Native coverage subprocess patch retained |
| pre-commit | >=4.7.0 | >=4.7.0 | 4.7.0 | All hooks exercised |
| scriv[toml] | >=1.8.0 | >=1.8.0 | 1.8.0 | Changelog fragment triggers repository release flow |
| pillow | >=13.0.0 | >=13.0.0 | 13.0.0 | Imaging/dev/model surfaces agree |
| mlx-vlm | ==0.7.9 | ==0.7.11 | 0.7.11 | macOS arm64 optional extra |
| sphinx | >=9.1.0 | >=9.2.0 | 9.2.0 | pyproject and docs requirements agree |
| furo | >=2025.12.19 | >=2025.12.19 | 2025.12.19 | Current |
| myst-parser | >=5.1.0 | >=5.1.0 | 5.1.0 | Current |

The UniMedVL inputs and all their transitive versions are enumerated in the complete Python table instead of repeated here. Highlights: torch 2.14.1→2.15.0, torchvision 0.29.1→0.29.2, Transformers 5.20.1→5.22.1, triton 3.8.0→3.9.0, python-dateutil 2.9.0.post0→3.0.0 and setuptools 84.0.0→84.2.1. The requested older Dependabot versions are all superseded, including packaging 26.3 and NumPy 2.5.4, which were already current.

## Infrastructure, images and executable tools

| Dependency | Before | Now | Latest stable / source |
| --- | --- | --- | --- |
| Terraform CLI | 1.16.6; HCL floor >=1.10 | 1.17.1; floor >=1.17.1 in all six roots | [1.17.1](https://github.com/hashicorp/terraform/releases/tag/v1.17.1) |
| AWS provider, both roots | 6.69.1 / ~>6.69 | 6.71.0 / ~>6.71 | [Registry](https://registry.terraform.io/providers/hashicorp/aws/latest), `data/terraform-aws.json` |
| AzureRM provider, both roots | 5.10.0 / ~>5.10 | 5.12.0 / ~>5.12 | [Registry](https://registry.terraform.io/providers/hashicorp/azurerm/latest), `data/terraform-azurerm.json` |
| Google provider, both roots | 8.7.0 / ~>8.7 | 8.9.0 / ~>8.9 | [Registry](https://registry.terraform.io/providers/hashicorp/google/latest), `data/terraform-google.json` |
| TFLint | 0.64.0 | 0.65.0 | [0.65.0](https://github.com/terraform-linters/tflint/releases/tag/v0.65.0) |
| kubectl / upstream Kubernetes | 1.37.1 | 1.37.2 | [1.37.2](https://github.com/kubernetes/kubernetes/releases/tag/v1.37.2) |
| kubeconform | 0.8.0 | 0.8.0 | [0.8.0](https://github.com/yannh/kubeconform/releases/tag/v0.8.0) |
| Helm | 4.3.1 | 4.3.1 | [4.3.1](https://github.com/helm/helm/releases/tag/v4.3.1) |
| kind | 0.33.0 | 0.33.0 | [0.33.0](https://github.com/kubernetes-sigs/kind/releases/tag/v0.33.0) |
| k3s binary / install script | 1.37.1+k3s1 | 1.37.1+k3s1 | [Latest k3s release](https://github.com/k3s-io/k3s/releases/tag/v1.37.1%2Bk3s1); upstream Kubernetes 1.37.2 has no newer k3s build yet |
| AWS CLI | 2.37.13 | 2.39.2 | [AWS release history](https://raw.githubusercontent.com/aws/aws-cli/v2/CHANGELOG.rst), verified archive checksum |
| gcloud CLI and auth component | 588.0.0 | 591.0.0 | [Google component archive](https://dl.google.com/dl/cloudsdk/channels/rapid/components-2.json); auth component installed from this SDK |
| Azure CLI | 2.91.0, unlocked transitive graph | 2.91.0, compiled isolated graph | [PyPI](https://pypi.org/project/azure-cli/2.91.0/); every SDK pin and hold is in the Python table |
| ansible-core / ansible-lint | 2.21.5 / 26.10.1 | 2.21.5 / 26.10.1, compiled graph | [PyPI](https://pypi.org/project/ansible-core/), [PyPI](https://pypi.org/project/ansible-lint/) |
| yamllint / shellcheck-py | 1.38.0 / 0.11.0.1 | 1.38.0 / 0.11.0.1, compiled graph | [PyPI](https://pypi.org/project/yamllint/), [PyPI](https://pypi.org/project/shellcheck-py/); ShellCheck 0.11.0 |
| Actions runner archive/image | 2.338.0 | 2.338.2 | [2.338.2](https://github.com/actions/runner/releases/tag/v2.338.2); all platform archive hashes in `data/runner-pins.json` |
| ARC scale-set chart | 0.15.0 | 0.15.1 | [0.15.1](https://github.com/actions/actions-runner-controller/releases/tag/gha-runner-scale-set-0.15.1) |
| NVIDIA device-plugin manifest/image | 0.20.1 | 0.20.1 | [0.20.1](https://github.com/NVIDIA/k8s-device-plugin/releases/tag/v0.20.1); current index digest retained |
| NVIDIA container toolkit, base and libnvidia-container packages | 1.20.1-1 | 1.20.1-1 | [1.20.1](https://github.com/NVIDIA/nvidia-container-toolkit/releases/tag/v1.20.1); all four packages agree |
| NVIDIA Ubuntu server/open driver branch | R580 | R615 | Latest packaged server branch; Ubuntu 24.04 package 615.71.09-0ubuntu0.24.04.2. NVIDIA data-center 615.71.10/display 615.78.08 are different distribution channels; use signed distribution packages. `data/nvidia-driver-packages.json` and release JSON |
| llama.cpp CPU/GPU archives and cudart | b11738 | b12012 | [b12012](https://github.com/ggml-org/llama.cpp/releases/tag/b12012); CPU and GPU scripts/workflows agree; checksums verified |
| CUDA base image | 13.4.1-base-ubuntu24.04 | Same tag, verified current digest | Docker Hub `data/image-nvidia-cuda.json`; no newer release for this vendor base variant |
| Python GitLab image | 3.14-slim-trixie, floating tag | 3.14.10-slim-trixie, refreshed digest `cd89c30eda26…` | Registry manifest in `data/python-3.14-slim-trixie-manifest.json`; Ansible support hold below |
| actionlint container | 1.7.12, immutable digest | 1.7.12, same verified digest | [1.7.12](https://github.com/rhysd/actionlint/releases/tag/v1.7.12); current canonical upstream |
| zizmor CLI / action | 1.31.1 / 0.7.1 | 1.31.1 / 0.8.0 | [PyPI](https://pypi.org/project/zizmor/), [action release](https://github.com/zizmorcore/zizmor-action/releases/tag/v0.8.0); action's hash lock installs current CLI |

`data/github-releases.json` preserves release notes and asset checksums. Some GitHub “latest release” pointers belong to a different artifact stream (CodeQL bundle versus action, llama semantic release versus continuous `b` builds); the actual action tags/build stream were checked explicitly. No third-party Terraform modules, Helm dependency manifests, submodules, composite actions or reusable workflow definitions need separate regeneration.

## Language and auxiliary tooling

| Dependency | Before | Now | Latest release / reason |
| --- | --- | --- | --- |
| Main Python | CI 3.15 (previous patch 3.15.0) | 3.15.2 locally; CI 3.15 resolves current patch | [PSF 3.15.2](https://www.python.org/downloads/release/python-3152/), expedited security release; FTP registry snapshot saved |
| UniMedVL Python | 3.14 | 3.14.10 | 3.15.2 current overall; SentencePiece wheel hold below |
| Ansible control-node Python | 3.14 | 3.14.10 | 3.15.2 current overall; supported control-node matrix ends at 3.14 |
| Node.js secret scanner runtime | Runner's implicit Node | 26.13.0 | [Node release index](https://nodejs.org/dist/index.json), newest stable/LTS; 27 alpha excluded |
| pnpm package manager | npx, unlocked manager graph | 12.16.1 | [npm registry](https://registry.npmjs.org/pnpm/latest); Rust native manager and platform executables locked |
| secretlint / recommended preset | 13.0.7 / 13.0.7 | 13.1.0 / 13.1.0 | [npm registry](https://registry.npmjs.org/secretlint/latest); all transitive versions in JS table |
| uv | 0.13.1 | 0.13.6 | [0.13.6](https://github.com/astral-sh/uv/releases/tag/0.13.6), download hashes synchronized |
| pip / pip-audit | Resolved latest, no old executable pin | 26.2.2 / 2.10.1 | PyPI snapshots; isolated audit tooling |
| build / twine | Resolved latest, no old executable pin | 1.6.1 / 7.0.0 | PyPI snapshots; full build and metadata verification |

## Explicit platform and upstream holds

1. **SentencePiece 0.2.2 has no released cp315 wheel.** torch, torchvision and triton now have cp315 builds; they are no longer reasons to hold Python. `experiments/issue25_sentencepiece_wheels.py` proves binary-only 3.14 succeeds and 3.15 fails. The upstream maintainer confirms cp315 builds on master with a 0.2.3 release planned. Using unreleased source would defeat the stable-release inventory. Keep only the model environment at 3.14.10 until released wheels exist.
2. **Ansible 2.21 supports control Python 3.12–3.14.** Its [published support matrix](https://docs.ansible.com/projects/ansible/latest/reference_appendices/release_and_maintenance.html) requires the GitLab/infrastructure environment to stay on current 3.14, while project CI uses 3.15.2. Debian trixie is the current stable base for that official Python image.
3. **CUDA's linked libraries must match their parent release.** torch 2.15's exact CUDA/NVIDIA pins are recorded per row; do not override binary ABI requirements with unrelated newest library builds. CUDA Toolkit 13.2.2 in that graph and llama's CUDA 13.4 runtime are isolated, upstream-built distributions. SymPy similarly holds mpmath below 1.4; Azure CLI and the JS scanner have their own exact parent bounds. Every such row names the constraining parent. Separate environments preserve these incompatible upstream requirements.
4. **Vendor GPU image families and drivers have supported combinations.** GCP's [current supported base image](https://docs.cloud.google.com/deep-learning-vm/docs/images) remains `common-cu129-ubuntu-2404-nvidia-580`; its image family resolves the newest build. AWS uses its vendor's latest Ubuntu 24.04 DLAMI SSM alias; Azure uses Canonical's `latest` image version. Keep this common supported GPU OS family rather than substitute an unverified Ubuntu 26.x GPU image. The workload installs its isolated current Python/torch/llama distributions; the base image's preinstalled CUDA/Python is not its model environment.
5. **Azure's NVIDIA extension is a vendor-managed release channel.** `type_handler_version="1.6"` with `auto_upgrade_minor_version=true` follows Microsoft's [documented deployment schema](https://learn.microsoft.com/en-us/azure/virtual-machines/extensions/hpccompute-gpu-linux), selecting the latest compatible 1.x extension. Its per-GPU CUDA/GRID driver choice is controlled by Microsoft. The minimum R580 runtime guard remains an actual CUDA compatibility floor; forcing R615 onto vendor-managed GRID configurations would not be a safe dependency update.
6. **Managed cluster API versions and system packages use provider/distribution channels.** EKS/AKS/GKE defaults select the cloud-supported control plane; imposing upstream Kubernetes 1.37.2 would fail on providers that have not released it. Locally installed k3s is the latest k3s release. `apt-get update`/Ansible cache refresh selects current packages for each supported OS: ca-certificates, curl, git, openssh-client, unzip, xz-utils, software-properties-common, gnupg, tar, gzip, libgomp1 and tesseract-ocr. Their versions are distribution-specific, not cross-platform language-library pins; no stale exact apt version was retained. NVIDIA toolkit versions are explicit and listed separately.
7. **Supported-language floors and compatibility matrices are not stale runtime pins.** The stdlib core still supports Python >=3.11; docs require >=3.12 through Sphinx. Compatibility jobs intentionally test these older supported interpreters/runners. Raising them to 3.15 would remove existing support without an issue request. Historical evidence files intentionally retain old results.
8. **FlashAttention is an upstream optional-import compatibility problem.** Current flash-attn 2.8.3.post2 has no matching release wheels for this torch/Python combination (saved PyPI/release metadata). Keep the existing import adapter and use current native torch GPU attention or the CPU fallback. No obsolete flash-attn package is installed. This is reported with UniMedVL's hard import, and the remaining CPU fallback is reported to PyTorch.

All final language-package security audits pass with **zero advisory exclusions**. The failed npm package-manager candidate is investigation evidence, not a dependency in the resulting pnpm tree; see [research](research.md).
