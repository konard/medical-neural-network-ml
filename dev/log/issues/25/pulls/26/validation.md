# Verification and review

Validation ran in isolated environments to respect the container's **2.92 GiB shared memory limit**. Heavy jobs ran sequentially. No stress input, unbounded allocation experiment, full UniMedVL checkpoint load or paid cloud deployment was performed locally.

## Local checks

| Check | Command / environment | Result / preserved evidence |
| --- | --- | --- |
| Baseline full suite | Original prepared tree, existing dependencies | 1,114 passed, 50 skipped; `data/baseline-pytest.log` |
| Updated full suite | Python 3.15.2, editable `[dev,docs]`, `python -m pytest --cov --cov-report=term-missing` | 1,119 passed, 48 skipped, source coverage 96%; `data/final-pytest.log`. Optional/network/model test limits below. |
| Native RoPE / upstream model CPU | Python 3.14.10, torch 2.15.0+cpu, torchvision 0.29.2+cpu, current non-CUDA UniMedVL lock; pinned real upstream code/configs | 10 passed, one CUDA-only test skipped; `data/unimedvl-tests.log`. Golden tokens/logits unchanged; FutureWarning treated as error. |
| Native causal-bias equivalence | `experiments/issue25_attention_bias.py` | Identical outputs for three small prefill/decode shapes; `data/attention-bias.log` |
| Config and CPU missing-kernel reproduction | `experiments/issue25_unimedvl_migration.py` | Old warning reproduced, new config warning-free, CPU kernel gap reproduced and retained fallback finite; `data/unimedvl-migration.log` |
| Python wheel availability | `experiments/issue25_sentencepiece_wheels.py` | Released 0.2.2 resolves on 3.14 and cannot resolve binary-only on 3.15; `data/sentencepiece-wheel-reproduction.log` |
| Lint / format | `ruff check .`, `ruff format --check .` | Pass; `data/ruff.log`, `ruff-format.log` |
| Static types | `mypy src`; `mypy --strict -p scripts.cloud_gpu` | Pass; `data/mypy.log` |
| All repository hooks | `pre-commit run --all-files` | Pass, including current ruff/mypy, TOML/YAML, Dependabot/workflow schemas; `data/pre-commit.log` |
| Package artifacts | `python -m build`; `twine check dist/*`; `python scripts/check_distribution.py --install dist` | sdist/wheel build, isolated installation and both public CLIs pass; `data/package-build.log`, `twine.log`, `distribution.log` |
| Documentation | `sphinx-build -W -b html docs ...` | Pass with warnings as errors; `data/docs-build.log` |
| Repository size policy | `python scripts/check_file_size.py` | Pass; `data/file-size.log` |
| Six Terraform roots | `terraform init -upgrade`, format/validate; TFLint | All pass, refreshed authenticated registry checksums in six locks; `data/terraform-upgrade.log`, `infra-validation.log` |
| Infrastructure integration | `bash scripts/check-infra.sh` with current tool binaries and locked Python 3.14 infra environment | Production strict Ansible lint (three playbooks), rendered Kubernetes schema validation, ARC Helm rendering and ShellCheck pass; `data/infra-validation.log` |
| Workflows | Native actionlint 1.7.12 with repository's scoped config; online zizmor 1.31.1, default and pedantic high/high | No findings; `data/actionlint.log`, `zizmor.log`, `zizmor-pedantic.log` |
| Python advisories | `python scripts/audit_dependencies.py` with pip-audit 2.10.1; all three pinned tool/model locks plus package/docs; separate arm64 macOS resolution audit | Zero known advisories / zero exclusions; `data/security-audit.log`, `pinned-security-audit.json`, `macos-security-audit.json` |
| JavaScript/toolchain advisories | Node 26.13.0, pnpm 12.16.1; frozen install; `pnpm --dir .github/tools audit --json` | Zero advisories across 99 manager/application/platform entries; `data/pnpm-frozen-install.log`, `pnpm-audit.json` |
| Secrets | Locked secretlint 13.1.0 / current recommended preset | Full scan passes; `data/secretlint.log` (empty output is success) |

The final tree keeps no advisory allow-list entries. Earlier failed npm candidate audits are research evidence and have been replaced by the final pnpm audit. Registry inventories cover optional macOS and pnpm platform packages even when this Linux host does not install them.

## Reproductions before implementation

- `data/reproduction-terraform-coverage.log`: the new test fails because no Terraform Dependabot entry exists. The fix adds all six actual roots and verifies the declaration exactly matches the discovered roots.
- `data/reproduction-gke-default-pool.log`: the new test fails on deprecated `remove_default_node_pool` / temporary default nodes. The fix uses the new native zero-node setting and passes actual provider validation.
- The bounded UniMedVL experiment shows the old deprecated config setter and torch's missing CPU kernel before checking the maintained config/fallback. Existing golden/frequency/attention tests verify behavior, rather than asserting a new package's version alone.
- Original CI advisory and old Dependabot failures are preserved with exact errors and lines in [CI investigation](ci-investigation.md).

## Hosted checks and practical limits

All eight workflows pass on implementation commit `001794a`, including compatibility, strict lint/types, build/install, docs/links, online workflow checks, CodeQL, both platform audits, production infrastructure validation, real kind/container checks, all real samples, Tesseract, actual upstream UniMedVL CPU code and actual MedGemma 4B CPU model/server tests. See `data/implementation-ci-checks.json`, `implementation-ci-runs.json` and [run links/log evidence](ci-investigation.md).

The main unit environment omits optional torch/MLX libraries; those paths run in their isolated upstream/native-macOS jobs. Network tests are opt-in locally and run with real downloads in CI. GPU-only tests need CUDA hardware; cloud provisioning and write-capable release jobs require trusted main/manual invocation. Existing conditions are preserved. A passing CPU test is not a claim that the 29 GB UniMedVL checkpoint was loaded or that a physical GPU benchmark ran.

## Final review checklist

The final PR diff is reviewed for repeated pins, runtime/API behavior, retained compatibility paths, lock/input consistency, unexpected feature removal and dependency-update scope. Main is fetched again before finalization and merged only if it has advanced. The changelog fragment is the repository's release trigger; the automated release workflow owns the package version bump.

Issue comments, sub-issues and all three PR comment/review API types were refreshed after implementation: all remain empty. The requirement matrix maps every issue requirement to concrete changes and verification. Upstream reports remain linked even where the local workaround is still required. No UI was changed, so UI screenshots do not apply.
