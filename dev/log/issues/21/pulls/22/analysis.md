# Findings while updating every dependency

Each finding gives the symptom, the root cause, the fix and the evidence.
Commit ids are on branch `issue-21-f7500eb08675`.

## F1. Why Dependabot's pull requests #13-#17 could not be merged

- **Symptom.** All five pull requests listed in the issue had a merge
  conflict, and #13, #14 and #16 also had failing checks:
  - **#13 and #16.** `tests/test_medbench_model_run.py::test_unimedvl_pins`
    failed. On #13 it asserted `'torch==2.6.0' in [...]`, and on #16
    `'transformers==4.49.0' in [...]`
    ([ci-logs/dependabot-pr13-py39-117161041309.log](ci-logs/dependabot-pr13-py39-117161041309.log),
    [ci-logs/dependabot-pr16-macos-117161120255.log](ci-logs/dependabot-pr16-macos-117161120255.log)).
  - **#14.**
    `tests/test_project_metadata.py::test_mypy_dependency_supports_configured_python_target`
    failed with `assert '2.0' not in <SpecifierSet('<3.0,>=1.13.0')>`
    ([ci-logs/dependabot-pr14-py313-117161350182.log](ci-logs/dependabot-pr14-py313-117161350182.log)).
  - **#15 and #17** passed all 31 checks.
- **Root cause.**
  - **The conflicts.** The pull requests were opened against a `main` that
    PR #20 (issue #19) then rewrote. That rewrite touched `pyproject.toml`
    and `scripts/unimedvl-requirements.txt`, the lines Dependabot had edited.
  - **The UniMedVL failures (#13, #16).** The test pinned exact versions.
    Any update of the lock therefore failed by design, even one that worked.
  - **The mypy failure (#14).** The test asserted mypy's old `<2.0` upper
    bound.
  - **Auto-merge.** The repository has `allow_auto_merge: false`, but that
    was not what stopped these pull requests. Each one had a conflict, and
    three had red checks.
- **Fix.**
  - **#14, #15 and #17.** mypy `>=2.4.0` with no upper bound, packaging 26.3
    and numpy 2.5.4 were already on `main` through PR #20. The mypy test
    there already asserts the new floor.
  - **#13 and #16 (11dad2f).** The UniMedVL lock is recompiled with torch
    2.14.1 and transformers 5.20.1, newer than both pull requests asked for.
    `test_unimedvl_pins` no longer asserts versions. It asserts what the
    code needs: torch ≥ 2.12, transformers ≥ 5 and no flash-attn (f4b6f0b).
    Whether upstream's code still runs on the lock is now tested by running
    it (F2, F3).
  - **Every pull request.** Each one got a comment saying where its update
    landed.
- **Test.** `test_unimedvl_pins` and the `unimedvl-cpu` job (F2).

## F2. flash-attn held torch at 2.8.0

- **Symptom.** On `main`, torch was 2.8.0 and transformers 4.57.6, and 14
  advisories were accepted against them (F5). torch 2.14.1 is the newest
  release.
- **Root cause.**
  - Upstream's `modeling/bagel/qwen2_navit.py` does
    `from flash_attn import flash_attn_varlen_func` at import time.
  - flash-attn's newest release, v2.8.3.post2, publishes wheels only for
    torch 2.4-2.8, CUDA 12 and CPython ≤ 3.13
    ([data/flash-attn-v2.8.3.post2-assets.json](data/flash-attn-v2.8.3.post2-assets.json)).
  - Building it from source takes hours and needs nvcc on the runner.
  - So the flash-attn wheel pinned torch, and torch pinned Python 3.13.
- **Fix (11dad2f).**
  - Since 2.10, torch ships the same FlashAttention-2 kernel as
    `torch.nn.attention.varlen.varlen_attn`. The `scale`, `window_size`
    and `enable_gqa` arguments the shim passes exist since 2.12 (86352f1).
  - `scripts/unimedvl_server.py` (`install_flash_attn_shim`) registers a
    `flash_attn` module whose `flash_attn_varlen_func` calls `varlen_attn`
    on CUDA.
  - On other devices `varlen_attn` raises `NotImplementedError`. There,
    each packed sequence goes through `scaled_dot_product_attention` with a
    bottom-right causal mask (flash-attn's convention for `q_len < k_len`).
    The missing CPU path, and the error that never mentions `varlen_attn`,
    are reported as
    [pytorch/pytorch#201457](https://github.com/pytorch/pytorch/issues/201457).
  - The flash-attn download, its SHA-256 and its C++ ABI check are deleted
    from `scripts/medbench-model-run.sh`.
  - The lock is recompiled with every package at its newest release, on
    Python 3.14.
- **Test.**
  - `tests/test_unimedvl_attention.py` compares the shim with a reference
    implementation over packed sequences, causal or not.
  - The same file checks cached decoding (bottom-right mask), bfloat16 in
    and out, the softmax scale, and that dropout and unknown arguments are
    refused.
  - `test_varlen_attn_still_has_no_cpu_kernel` fails once torch adds a CPU
    kernel, so the fallback can then be removed.
  - The CUDA path is tested by `test_cuda_uses_varlen_attn_and_matches_the_reference`.
    That test runs only on a GPU (see "Not verifiable here" in
    [README.md](README.md)).

## F3. transformers 5 removed the `"default"` RoPE initialiser

- **Symptom.** On transformers 5, upstream's `Qwen2RotaryEmbedding` fails
  with `KeyError: 'default'`.
- **Root cause.**
  - Upstream's `modeling/qwen2/modeling_qwen2.py` (a copy of transformers
    4.x code) looks up `ROPE_INIT_FUNCTIONS[self.rope_type]`, with
    `rope_type = "default"`.
  - transformers 5 moved plain RoPE into each model class and dropped the
    `"default"` key.
- **Fix (11dad2f).** `restore_default_rope` registers the 4.x function again,
  and only when the key is missing. The function is
  `inv_freq = 1 / theta^(2i/d)`, with `partial_rotary_factor`.
- **Test.**
  - `test_default_rope_matches_transformers_own` compares it with
    transformers' own computation.
  - `tests/test_unimedvl_upstream.py::test_chat_generates_the_golden_tokens`
    runs upstream's `Bagel.chat` end to end. It uses a tiny model built from
    the real `llm_config.json` and `vit_config.json`.

## F4. Issue #10's "different tokens on transformers 5" was a test artefact

- **Symptom.** uni-medical/UniMedVL#10 (opened during issue #19) said that
  upstream generated different tokens on transformers 5.
- **Root cause.** `experiments/issue19_unimedvl_cpu_smoke.py` built new
  random weights in each environment. Different torch versions draw
  different random numbers from the same seed, so the two stacks never ran
  the same model.
- **Fix (4b15ca5).**
  - `experiments/issue21_unimedvl_cross_stack.py` saves the tiny weights once.
    It then loads them under torch 2.8.0 with transformers 4.57.6, and under
    torch 2.14.1 with transformers 5.20.1.
  - Once F3 is in place, the tokens are the same and the logit difference is
    0.0 at all 12 steps
    ([data/cross-stack/compare.json](data/cross-stack/compare.json)).
  - The golden file has variants with the causal mask dropped or anchored
    top-left. Both give different tokens, so the test does catch a broken
    mask ([data/unimedvl-golden-cross-stack.json](data/unimedvl-golden-cross-stack.json)).
- **Upstream.** The correction was posted on uni-medical/UniMedVL#10, and
  the issue body was edited
  ([research.md "Upstream reports"](research.md#upstream-reports)).

## F5. 14 accepted advisories and 8 open Dependabot alerts on main

- **Symptom.** `scripts/audit_dependencies.py` on `main` accepted 14
  advisories against torch 2.8.0 and transformers 4.57.6. Dependency
  Review's `allow-ghsas` listed their high-severity aliases. Dependabot shows
  8 open alerts on `main`, all in `scripts/unimedvl-requirements.txt`: torch
  #18, #19 and #22, and transformers #14, #20, #21, #23 and #25
  ([data/dependabot-alerts-main.json](data/dependabot-alerts-main.json)).
- **Root cause.** These are the pins F2 held back.
- **Fix (099c7f2).** `pip-audit` finds no known vulnerability in the new lock
  ([data/pip-audit-unimedvl.txt](data/pip-audit-unimedvl.txt)).
  - `ACCEPTED_ADVISORIES` is now empty.
  - The `allow-ghsas` list is gone.
  - Every alert's first patched version (torch 2.13.0 at most; transformers
    5.10.0 at most, and #25 has no fix listed but covers `<= 5.8.1`) is below
    the new pins.
  - The alerts close when this merges.
- **Test.** `tests/test_audit_dependencies.py` and the Security workflow.

## F6. The CUDA runtime was unpacked where llama.cpp could not find it

- **Symptom.** In the GPU job, the `cudart-llama-*.tar.gz` archive went into
  a `cudart-llama-…/` subfolder of the llama.cpp folder. With the files
  there, `ldd libggml-cuda.so` shows `libcudart.so.13 => not found`
  ([data/llama-cuda-ldd-before-fix.txt](data/llama-cuda-ldd-before-fix.txt)).
- **Root cause.** Both release archives keep their files in one top-level
  folder. The binaries were unpacked with `--strip-components 1`, but the
  runtime was not. `libggml-cuda.so` has RUNPATH `$ORIGIN`, so it looks only
  in its own folder. llama-server would then have loaded no CUDA backend and
  run on the CPU without an error. This bug was already on `main`, at CUDA
  12.8.
- **Fix (026c803).** Both archives go through the same `unpack` function.
  With that, `ldd` resolves the runtime beside the backend, both for b11734
  ([data/llama-cuda-ldd-after-fix.txt](data/llama-cuda-ldd-after-fix.txt))
  and for b11738 ([data/llama-b11738-check.txt](data/llama-b11738-check.txt)).
- **Test.** `test_cuda_runtime_lands_next_to_the_cuda_backend` builds two
  release-style archives and runs the script's download step on them.

## F7. Dependabot could not update the UniMedVL lock

- **Symptom.** `.github/dependabot.yml` on `main` excluded
  `scripts/unimedvl-requirements.*`, so the one real lockfile was updated
  only by hand.
- **Root cause.** The torch pin had to match the hash-pinned flash-attn
  wheel, and Dependabot cannot update both together. After F2 nothing pins
  torch from outside the lock.
- **Fix (853e944).** A `uv` entry for `/scripts` opens one weekly grouped
  pull request. pip still excludes the files, so they are not proposed
  twice.
  - Dependabot reruns `uv pip compile` in the lock's own directory and keeps
    only `--python-version` from the header
    (dependabot-core `uv/file_updater/compile_file_updater.rb`).
  - A lock compiled from the repository root would have every `# via -r
    scripts/…` line rewritten in each Dependabot pull request.
  - So the lock is now compiled from `scripts/`, and the `.in` header says
    so. `experiments/issue21_dependabot_uv_compile.sh` reproduces
    Dependabot's compile and prints "Same lock as Dependabot would write."
- **Second root cause: lockstep pins.** Dependabot's `uv` ecosystem updates
  a group's members one at a time, each against the files as they are
  (dependabot/dependabot-core#16390, open). torchvision 0.28.0 requires
  `torch==2.13.0`, torchvision 0.29.0 requires `torch==2.14.0`, torch
  2.14.1 requires `triton~=3.8.0`, and transformers 5.20.1 requires
  `tokenizers<0.24`. With all of them pinned exactly in the `.in`, every
  member's compile fails, so none of them could ever move.
  `experiments/issue21_dependabot_lockstep.sh` replays Dependabot's steps
  on torch 2.13.0 / torchvision 0.28.0 / triton 3.7.1
  ([data/dependabot-lockstep.log](data/dependabot-lockstep.log)):
  - pinned: "torch -> 2.14.1: skipped", "torchvision -> 0.29.1: skipped",
    "triton -> 3.8.0: skipped", in either order;
  - torchvision and triton named without a version: "torch -> 2.14.1: lock
    now torch==2.14.1 torchvision==0.29.1 triton==3.8.0".
- **Fix (09bba63).** The `.in` names torchvision, triton and tokenizers without a
  version. The lock still pins them, and recompiling gives a byte-identical
  lock. uv treats the old lock's pins as preferences, so a torch or
  transformers update carries them along.
- **Test.** `test_dependabot_updates_the_unimedvl_lock_with_uv`,
  `test_unimedvl_lock_is_compiled_where_dependabot_compiles_it`,
  `test_unimedvl_lock_matches_its_input` (only those three packages are
  unpinned, and the lock pins them) and
  `test_unimedvl_cpu_job_runs_the_upstream_tests`. The last one checks that
  a lock pull request runs upstream's code.

## F8. CUDA 13 raises the GPU runner's driver and GPU floor

- **Symptom.** torch 2.14.1's Linux wheels pull CUDA 13.0
  (`cuda-toolkit==13.0.3`). llama.cpp's newest CUDA build is 13.4, and
  `main` used 12.8.
- **Root cause.** NVIDIA's CUDA release notes give two facts here:
  - CUDA 13.4's own driver branch is R615, but "existing CUDA 13.x
    applications run on drivers >=580 under CUDA minor version
    compatibility" (Table 2, Table 3).
  - CUDA 13 removed support for Maxwell, Pascal and Volta GPUs.
- **Fix (026c803).** llama.cpp's GPU build moved to CUDA 13.4, so both
  runtimes on the GPU runner need the same driver floor, R580 (R615 only for
  13.4's new features, which neither runtime is known to need). The requirement is
  written in `docs/hardware-workflows.md` and
  `docs/case-studies/issue-1/ci-and-gpu-runner.md`.
- **Test.** No GPU runner is registered, so this is not verifiable here.

## F9. mlx-vlm 0.7.9 reads sampling defaults from the checkpoint

- **Symptom.** mlx-vlm 0.7.9 (2026-10-19) changes `generate` to read a
  checkpoint's `generation_config.json`.
- **Root cause.** The new defaults apply only to options the caller leaves
  unset.
- **Fix (b000ac3).** `MlxBackend` always passes `temperature=0.0`, so
  decoding stays greedy and the pin moves without a code change.
  `resolve_sampling_defaults` reads the checkpoint only when `temperature`
  is `None`. The other calls it makes (`load`, `stream_generate`,
  `apply_chat_template`) keep their signatures in 0.7.9
  ([data/mlx-vlm-0.7.9-api.txt](data/mlx-vlm-0.7.9-api.txt)).
- **Test.** The MLX unit tests (`tests/test_medbench_mlx.py`) mock mlx-vlm's
  API. A real run needs Apple silicon, which is not available here.

## F10. The new layout test failed on macOS

- **Symptom.** Compatibility on `macos-26` failed in
  `test_cuda_runtime_lands_next_to_the_cuda_backend` with "This pinned
  llama.cpp build requires Linux x86_64"
  ([ci-logs/compat-macos-118187903886.log](ci-logs/compat-macos-118187903886.log)).
- **Root cause.** The run script refuses every platform except Linux x86_64,
  as it should.
- **Fix (0f1516d).** The test is marked `linux_x86_64`, the same condition
  the script checks.

## F11. The changelog gate needs a fragment for this change

- **Symptom.** "Changelog Fragment Check" failed
  ([ci-logs/ci-changelog-118187845362.log](ci-logs/ci-changelog-118187845362.log)).
- **Fix (0f1516d).**
  `changelog.d/20261019_issue_21_dependency_updates.md` describes the
  user-facing changes.

## F12. `varlen_attn` changed its signature twice

- **Symptom.** A first draft of the shim's docstring said "torch 2.10+".
- **Root cause.** The keyword arguments came in two steps, according to
  pytorch's sources at the v2.10.0-v2.14.0 tags:
  - 2.10.0 has only `(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k,
    is_causal=False, return_aux=None)`.
  - 2.11 adds `scale` and `window_size` as keyword-only arguments, and
    removes `is_causal`.
  - 2.12 adds `enable_gqa`, `seqused_k`, `block_table` and `num_splits`.
  - From 2.11, `is_causal` is computed as `window_size == (-1, 0)`, so
    `window_size` has to be a tuple. A list is silently treated as not
    causal before the kernel call.
- **Fix (86352f1, f4b6f0b).** The docstring states 2.12, the pin test
  requires torch ≥ 2.12, and the shim passes a tuple.

## F13. Upstream's call to a deprecated transformers 5 helper

- **Symptom.** `FutureWarning: rope_config_validation is deprecated` is
  raised when upstream's `Qwen2Config` is built.
- **Root cause.** It comes from upstream's code, not from this repository's.
  transformers 5 keeps `rope_config_validation` as a wrapper that still
  validates, through `validate_rope()`.
- **Fix (11dad2f).** `prepare_upstream_imports` ignores exactly that message
  and category, and nothing else. The suggested upstream change
  (`self.validate_rope()`) is in the UniMedVL#10 comment.

## F14. Two checks fail on `main` and need an admin

- **Symptom.** On `main` ([ci-logs/main-39047690239-jobs.tsv](ci-logs/main-39047690239-jobs.tsv),
  [ci-logs/main-39250329262-jobs.tsv](ci-logs/main-39250329262-jobs.tsv)),
  two checks fail:
  - "Release Preflight (credentials)": PyPI refused to mint a token,
    `422 invalid-publisher`.
  - "Branch Rulesets": rulesets `main-checks`, `main-history` and
    `main-review` are not applied.
- **Root cause.** Both need a repository admin. One has to add a PyPI
  trusted publisher, and one has to run `scripts/repository_rulesets.py
  apply`. Each failure message says exactly what to do.
- **Fix.** None in this pull request. These checks already behave as the
  repository wants: they pass on pull requests and fail on `main` with
  instructions. They existed before this pull request, and it does not
  affect them.

## F15. llama.cpp released again during the update

- **Symptom.** b11734 was the newest release at 14:08 UTC. b11738 followed
  at 16:59 UTC, the same day.
- **Fix (3f2739d).** Moved to b11738.
  - The three archive hashes were taken from GitHub's asset digests and
    matched against a fresh download.
  - `llama-server --version` reports build 11738.
  - The CUDA backend resolves its runtime
    ([data/llama-b11738-check.txt](data/llama-b11738-check.txt)).

## Ruled out

- **Building flash-attn from source** for torch 2.14 and Python 3.14. It
  would take hours of nvcc on every cache miss, and torch's own kernel (F2)
  is the same FlashAttention-2.
- **Holding transformers at 4.x.** F3 is the only incompatibility, and F4
  shows the outputs are the same.
- **Overriding torch's exact CUDA library pins** to reach the newest
  nvidia-* packages. torch 2.14.1 is built and tested against the versions
  it pins ([versions.md §2](versions.md#2-unimedvl-runtime-scriptsunimedvl-requirementsin--txt)).
- **Reporting the `window_size` list-versus-tuple behaviour to PyTorch.**
  It could not be checked on a GPU here. The flash kernel probably treats a
  `(-1, 0)` window as causal anyway, so the report would rest on a guess.
