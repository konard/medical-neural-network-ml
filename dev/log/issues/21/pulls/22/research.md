# Research: migration notes, new features, verification, upstream reports

Everything here was checked on 2026-10-19. Version tables are in
[versions.md](versions.md), and findings with root causes are in
[analysis.md](analysis.md).

## Sources

These pages and files were read for this pull request:

- PyTorch `varlen_attn`: <https://docs.pytorch.org/docs/stable/nn.attention.varlen.html>
  and the source at each release tag,
  <https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/attention/varlen.py>
  (also `v2.10.0`, `v2.11.0`, `v2.12.0` and `v2.13.0`), for
  [F12](analysis.md#f12-varlen_attn-changed-its-signature-twice).
- PyTorch release notes: <https://github.com/pytorch/pytorch/releases/tag/v2.12.0>
  and <https://github.com/pytorch/pytorch/releases/tag/v2.14.0>. torch
  2.14.1's own `Requires-Dist` is saved in
  [data/torch-2.14.1-requires-dist.json](data/torch-2.14.1-requires-dist.json).
- flash-attn: <https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2>
  (asset list in
  [data/flash-attn-v2.8.3.post2-assets.json](data/flash-attn-v2.8.3.post2-assets.json))
  and its wheel build matrix,
  <https://github.com/Dao-AILab/flash-attention/blob/main/.github/workflows/publish.yml>.
- transformers 5: <https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md>
  (RoPE moved into `rope_parameters`, and the `"default"` entry of
  `ROPE_INIT_FUNCTIONS` was removed).
- huggingface-hub 1.x/2.x: <https://huggingface.co/docs/huggingface_hub/concepts/migration>
  (`requests` replaced by `httpx`).
- CUDA: <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>
  (Table 2 and Table 3 for the driver floor, and the removal of Maxwell,
  Pascal and Volta in CUDA 13), for
  [F8](analysis.md#f8-cuda-13-raises-the-gpu-runners-driver-and-gpu-floor).
- llama.cpp: <https://github.com/ggml-org/llama.cpp/releases/tag/b11738>.
- mlx-vlm: <https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.7.9> and the
  0.7.9 wheel's source
  ([data/mlx-vlm-0.7.9-api.txt](data/mlx-vlm-0.7.9-api.txt)).
- Dependabot:
  - <https://docs.github.com/en/code-security/dependabot/ecosystems-supported-by-dependabot/supported-ecosystems-and-repositories>
    (the `uv` ecosystem);
  - dependabot-core's
    [`uv/lib/dependabot/uv/file_updater/compile_file_updater.rb`](https://github.com/dependabot/dependabot-core/blob/main/uv/lib/dependabot/uv/file_updater/compile_file_updater.rb),
    which shows how the `.in` and lock are recompiled
    ([F7](analysis.md#f7-dependabot-could-not-update-the-unimedvl-lock));
  - [dependabot/dependabot-core#16390](https://github.com/dependabot/dependabot-core/issues/16390)
    (group members are resolved one at a time).
- Process: hive-mind's
  [DEPENDENCY-UPDATE-BEST-PRACTICES.md](https://github.com/link-assistant/hive-mind/blob/main/docs/DEPENDENCY-UPDATE-BEST-PRACTICES.md).
- Registries: the PyPI JSON API and GitHub tags and releases, through
  [experiments/issue21_pinned_vs_latest.py](../../../../../../experiments/issue21_pinned_vs_latest.py)
  and
  [experiments/issue21_actions_hooks_vs_latest.py](../../../../../../experiments/issue21_actions_hooks_vs_latest.py)
  ([data/latest.json](data/latest.json)).

## Major versions

Each major bump below was checked against this repository. "Affects us?"
names the evidence, and "Change" is what this pull request did about it.

| Dependency | Breaking change | Affects us? | Change |
| --- | --- | --- | --- |
| torch 2.8 → 2.14.1 (UniMedVL) | Not a semver major, but six minors. flash-attn publishes no wheel past torch 2.8. `varlen_attn` (2.10) changed its signature in 2.11 and 2.12. The Linux wheels moved to CUDA 13.0 | **Yes.** Upstream imports `flash_attn` ([F2](analysis.md#f2-flash-attn-held-torch-at-280), [F12](analysis.md#f12-varlen_attn-changed-its-signature-twice)), and CUDA 13 raises the GPU floor ([F8](analysis.md#f8-cuda-13-raises-the-gpu-runners-driver-and-gpu-floor)) | `flash_attn_varlen_func` is served from torch's `varlen_attn`. The flash-attn wheel is deleted. The pin test requires torch ≥ 2.12 |
| transformers 4.57 → 5.20 (UniMedVL) | RoPE settings moved to `rope_parameters`, and the `"default"` RoPE initialiser was removed. `rope_config_validation` is deprecated | **Yes.** Upstream's vendored Qwen2 looks up `"default"` ([F3](analysis.md#f3-transformers-5-removed-the-default-rope-initialiser)) and calls the deprecated helper ([F13](analysis.md#f13-upstreams-call-to-a-deprecated-transformers-5-helper)) | `restore_default_rope` puts back the 4.x initialiser. The one `FutureWarning` is filtered by exact message. The tokens and logits match 4.57.6 ([F4](analysis.md#f4-issue-10s-different-tokens-on-transformers-5-was-a-test-artefact)) |
| huggingface-hub 0.36 → 2.2 (UniMedVL, through transformers) | `requests` replaced by `httpx`, so a `requests.Session` configured for the hub no longer applies | No. The only direct call is `snapshot_download(repo_id, revision=, allow_patterns=)` in `scripts/measure_training_memory.py`, and 2.2.0 has the same parameters. Upstream's `requests` uses are docstring examples only | None. The lock swaps requests, urllib3, certifi and charset-normalizer for httpx2 and its dependencies |
| CUDA 12.8 → 13.4 (llama.cpp GPU build) and 12.8 → 13.0 (torch wheels) | CUDA 13 drops Maxwell, Pascal and Volta, and needs an R580 or newer driver (R615 for 13.4's own features) | Only on the GPU runner, and none is registered | The floor is written in `docs/hardware-workflows.md` and the GPU-runner case study ([F8](analysis.md#f8-cuda-13-raises-the-gpu-runners-driver-and-gpu-floor)) |
| Python 3.13 → 3.14 (UniMedVL venv) | None for this code | No. The CPU job runs upstream's `chat()` on 3.14 | `UNIMEDVL_PYTHON`, both CI jobs and the lock header use 3.14. torch and triton publish no cp315 wheels yet |
| mlx-vlm 0.7.8 → 0.7.9 (exact pin) | Not a major, but `generate` now reads sampling defaults from the checkpoint for options the caller leaves unset | No. `MlxBackend` always passes `temperature=0.0` ([F9](analysis.md#f9-mlx-vlm-079-reads-sampling-defaults-from-the-checkpoint)) | None |

opencv-python 5, numpy 2 and the Python package's majors were crossed in
PR #20 and are recorded in
[dev/log/issues/19/pulls/20/research.md](../../../19/pulls/20/research.md).

## New features adopted

These are places where a newer release replaced something the repository did
by hand.

| New feature | Replaces | Where |
| --- | --- | --- |
| torch's `torch.nn.attention.varlen.varlen_attn` (FlashAttention-2 in torch, with `scale`, `window_size` and `enable_gqa` since 2.12) | The flash-attn wheel and its URL, SHA-256 and C++ ABI check in `scripts/medbench-model-run.sh`, and the torch and Python pins that the wheel forced | `scripts/unimedvl_server.py` (`install_flash_attn_shim`) |
| Dependabot's `uv` ecosystem for `uv pip compile` locks | Updating the UniMedVL lock by hand, and the `exclude-paths` that kept Dependabot away from it | `.github/dependabot.yml` ([F7](analysis.md#f7-dependabot-could-not-update-the-unimedvl-lock)) |
| `uv pip compile` treating the old lock's pins as preferences | Exact pins on packages that are released together with torch or transformers, which Dependabot's per-member updates could never move | `scripts/unimedvl-requirements.in` ([F7](analysis.md#f7-dependabot-could-not-update-the-unimedvl-lock)) |
| Releases with no known advisory | The 14 accepted advisories in `scripts/audit_dependencies.py` and Dependency Review's `allow-ghsas` list | [F5](analysis.md#f5-14-accepted-advisories-and-8-open-dependabot-alerts-on-main) |

Considered and not adopted:

- `varlen_attn` on the CPU. torch has no CPU kernel for it, so the shim
  keeps its `scaled_dot_product_attention` fallback. This is reported as
  pytorch/pytorch#201457, and `test_varlen_attn_still_has_no_cpu_kernel`
  fails once torch adds one.
- transformers 5's own Qwen2. Upstream ships its own modified copy (packed
  sequences, mixture of experts), so swapping it would be a rewrite of
  upstream's model, not a dependency update.

## Verification

All checks were run locally on the final code commit, then in CI.

| Check | Result | Evidence |
| --- | --- | --- |
| pytest on Python 3.15 | 951 passed, 47 skipped (opt-in network, model, GPU and multi-language tests) | [data/verification/pytest.log](data/verification/pytest.log) |
| UniMedVL tests on the lock with CPU torch 2.14.1 on Python 3.14 (what the `unimedvl-cpu` job runs, plus the training-memory tests) | 28 passed, 2 skipped (the CUDA test, and one test that needs peft) | [data/verification/pytest-unimedvl-cpu.log](data/verification/pytest-unimedvl-cpu.log) |
| Same tokens and logits on torch 2.8.0 with transformers 4.57.6 as on torch 2.14.1 with transformers 5.20.1 | Same tokens, logit difference 0.0 | [data/cross-stack/compare.json](data/cross-stack/compare.json) |
| pre-commit on every file (ruff, mypy, the Dependabot and workflow schemas, and others) | All passed | [data/verification/pre-commit.log](data/verification/pre-commit.log) |
| `sphinx-build -W --keep-going` | Build succeeded | [data/verification/sphinx.log](data/verification/sphinx.log) |
| `scripts/audit_dependencies.py` (pip-audit on the package and the UniMedVL lock) | No known vulnerabilities, none accepted | [data/verification/audit.log](data/verification/audit.log), [data/pip-audit-unimedvl.txt](data/pip-audit-unimedvl.txt) |
| Dependabot's compile replayed on the lock | Same lock as Dependabot would write | `experiments/issue21_dependabot_uv_compile.sh` |
| Dependabot's per-member group update replayed on an older torch trio | Moves only with torchvision and triton unpinned in the `.in` | [data/dependabot-lockstep.log](data/dependabot-lockstep.log) |
| llama.cpp b11738 archives | Hashes match; `llama-server` reports build 11738; the CUDA backend finds `libcudart` | [data/llama-b11738-check.txt](data/llama-b11738-check.txt) |
| CI on the pull request | All workflows green on the final commit | The pull request's checks tab |

## Upstream reports

| Blocker or workaround here | Upstream | What was done |
| --- | --- | --- |
| The `"default"` RoPE helper put back for upstream's vendored Qwen2 (F3), and the deprecated `rope_config_validation` call (F13) | [uni-medical/UniMedVL#10](https://github.com/uni-medical/UniMedVL/issues/10) | **Commented** ([comment](https://github.com/uni-medical/UniMedVL/issues/10#issuecomment-6272845731)), and the issue body was corrected. The comment withdraws the "different tokens" claim from issue #19 (F4), gives the versions and the cross-stack result, and suggests a patch: a local default RoPE function and `self.validate_rope()` |
| No flash-attn wheel for torch ≥ 2.9 or cp314, which held torch at 2.8 (F2) | [Dao-AILab/flash-attention#2995](https://github.com/Dao-AILab/flash-attention/issues/2995) | Existing issue. **Commented** ([comment](https://github.com/Dao-AILab/flash-attention/issues/2995#issuecomment-6272858359)) with the asset list for v2.8.3.post2 and the workaround used here |
| `varlen_attn` has no CPU path and fails with an error about `aten::_flash_attention_forward` (F2) | [pytorch/pytorch#201457](https://github.com/pytorch/pytorch/issues/201457) | **Opened** after searching (the varlen RFCs #171814 and #179382 do not cover CPU). It includes a reproduction, the versions, the workaround kept here, and a suggested per-sequence SDPA fallback |
| torchvision, triton and tokenizers left unpinned in the `.in`, because Dependabot updates a group's members one at a time (F7) | [dependabot/dependabot-core#16390](https://github.com/dependabot/dependabot-core/issues/16390) | Existing issue about `uv lock`. **Commented** ([comment](https://github.com/dependabot/dependabot-core/issues/16390#issuecomment-6273265798)) that the `uv pip compile` path has the same problem, with a reproduction script, the versions, this workaround and a suggested fix for `CompileFileUpdater` |

Every workaround stays in this repository, so none of these reports blocks
the pull request.

## Best practices

How each principle of hive-mind's DEPENDENCY-UPDATE-BEST-PRACTICES.md was
applied:

1. **Table before changing anything.** [versions.md](versions.md) was
   resolved from PyPI and GitHub by script
   ([data/latest.json](data/latest.json)).
2. **A written reason for anything left behind.** The 16 lock rows behind
   their newest release are exact pins of torch 2.14.1 or sympy 1.14.0. Each
   row says so, the metadata is saved, and the `.in` repeats the reason.
   ruff's `<0.18` admits the current release and has its own reason.
3. **Cross majors deliberately.** See [Major versions](#major-versions).
   No constraint was loosened to make a major pass. transformers 4 and torch
   2.8 are no longer held, and the cross-stack run shows the same outputs.
4. **Adopt new features, delete hand-rolled copies.** See
   [New features adopted](#new-features-adopted). The flash-attn download,
   hash and ABI check, the hand-updated lock and the accepted-advisory list
   are gone.
5. **Honest constraints.** Every floor is the newest release, and the lock
   is recompiled and committed. Tests check what the code needs (torch ≥
   2.12, transformers ≥ 5) instead of exact versions, which is what made
   Dependabot's #13 and #16 fail
   ([F1](analysis.md#f1-why-dependabots-pull-requests-13-17-could-not-be-merged)).
6. **One version per dependency.** The pins in `pyproject.toml`,
   `docs/requirements.txt`, the lock and the hooks agree. The UniMedVL Python
   (3.14) and llama.cpp (b11738) each have one value everywhere, and tests
   enforce both.
7. **Update the toolchain.** UniMedVL moves to Python 3.14, the GPU build to
   CUDA 13.4, and llama.cpp to b11738. The workflow Python (3.15), runner
   images, actions and hooks were already current and were checked again.
8. **Green CI, zero new deprecation warnings.** See
   [Verification](#verification). The one new warning comes from upstream's
   code, and only that exact message is filtered (F13).
9. **Audit the tree you ship.** pip-audit was run on the package and on the
   lock as committed, and the 8 open Dependabot alerts on `main` were checked
   against the new pins ([F5](analysis.md#f5-14-accepted-advisories-and-8-open-dependabot-alerts-on-main)).
10. **Report blockers upstream.** See [Upstream reports](#upstream-reports).

## Tools considered

| Tool | Used? | Why |
| --- | --- | --- |
| uv (`uv pip compile`) | Yes | Compiles the lock for Python 3.14 and `x86_64-manylinux_2_28` from any host, and runs the same compile Dependabot does |
| pip-audit | Yes | Already used by the Security workflow; run on the new lock |
| Dependabot `uv` ecosystem | Yes | Keeps the UniMedVL lock current in its own weekly pull request |
| check-jsonschema | Yes | Validates `dependabot.yml` and the workflows before push |
| `pre-commit autoupdate` | Yes | Found nothing to update |
| zizmor | Yes, unchanged | Checks that every action comment matches its SHA |
| Renovate | No | It can bump lockstep groups together, which would avoid the dependabot-core#16390 workaround. But Dependabot already runs here with cooldowns, SHA-pinned actions and the policy in `docs/repository-policy.md`, and switching tools was not asked for |
| Building flash-attn from source | No | Hours of nvcc on every cache miss, and torch ships the same kernel ([Ruled out](analysis.md#ruled-out)) |
