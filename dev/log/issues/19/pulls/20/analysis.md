# Findings while updating every dependency

Each finding gives the symptom, the root cause, the fix and the evidence.
Commit ids are on branch `issue-19-5e219937bdea`.

## F1. `wilson_interval` returned 0.9999999999999999 for 4 out of 4

- **Symptom.** After the move to `macos-26` and Python 3.15, Compatibility
  run 39021135047 failed in `test_imaging_with_oracle`:
  `assert 0.9999999999999999 == 1.0`
  ([ci-logs/compat-39021135047-macos-26-py3.15.log](ci-logs/compat-39021135047-macos-26-py3.15.log)).
- **Root cause.** The Wilson closed form reaches its bounds of exactly 1
  (all right) and 0 (all wrong) only up to rounding. `z` comes from
  `NormalDist().inv_cdf`, which relies on the platform's `libm`; the macOS 26
  `libm` gives a `z` one ulp away from the Linux one, and that is enough. The
  bug is not macOS-only: on this Linux host, with `z` shifted by one ulp
  either way, 299 of 800 (total, confidence) combinations missed the exact
  bound ([experiments/issue19_wilson_boundary.py](../../../../../experiments/issue19_wilson_boundary.py)).
- **Fix (3ebdd19).** Set the bound directly when `successes == total`
  (high = 1) or `successes == 0` (low = 0).
- **Test.** `test_wilson_interval_reaches_its_bounds_exactly`, parametrized
  over a `z` nudged by -1, 0 and +1 ulp (`math.nextafter`), checks totals
  1-100 at four confidence levels. It failed before the fix and passes after.

## F2. Python 3.10 held seven UniMedVL packages behind

- **Symptom.** With the lock compiled for Python 3.10 (upstream's
  `environment.yaml` uses 3.10.18), numpy, pillow, matplotlib, contourpy,
  fonttools, networkx and filelock stopped at their last releases that still
  support 3.10 (for example numpy 2.2.6 instead of 2.5.4).
- **Root cause.** A toolchain hold, not a package hold. Python 3.10 reaches
  end of life in October 2026. The real limit is torch 2.8.0: it, triton 3.4.0
  and the flash-attn cu12torch2.8 wheel are built for cp39-cp313 only, so 3.13
  is the newest Python the stack can run on. Upstream's code uses no stdlib
  module removed after 3.10.
- **Fix (dbd7ad4).** The lock is recompiled with `--python-version 3.13`; the
  flash-attn wheel moves to its cp313 build (SHA-256 taken from the GitHub
  release asset digest and checked after download); `UNIMEDVL_PYTHON`, the
  model-tests `setup-python` step and the docs follow. All seven packages are
  now at their newest release ([data/unimedvl/lock-vs-pypi.tsv](data/unimedvl/lock-vs-pypi.tsv)).
- **Test.** `test_unimedvl_python_matches_everywhere` checks that the wheel's
  `cpXY` tag, the `UNIMEDVL_PYTHON` default, the CI step and the compile
  command name the same Python (it fails if any one is left on 3.10).
- **Evidence.** The CPU smoke test under 3.13 generates the same token ids as
  the old 3.10 pins ([smoke-python-3.13-lock.json](data/unimedvl/smoke-python-3.13-lock.json)
  vs [smoke-old-pins.json](data/unimedvl/smoke-old-pins.json)), and with image
  seed 7 the same as the 3.10 new pins ([smoke-python-3.13-image-seed-7.json](data/unimedvl/smoke-python-3.13-image-seed-7.json)
  vs [smoke-new-pins-image-seed-7.json](data/unimedvl/smoke-new-pins-image-seed-7.json));
  `flash_attn_2_cuda.cpython-313` imports against `torch 2.8.0+cu128`
  ([flash-attn-import-python-3.13.log](data/unimedvl/flash-attn-import-python-3.13.log)).

## F3. transformers 5 breaks upstream UniMedVL

- **Symptom.** With transformers 5.20.1, building the model fails:
  `KeyError: 'default'` at `ROPE_INIT_FUNCTIONS[self.rope_type]` in upstream's
  `codes/modeling/qwen2/modeling_qwen2.py`
  ([data/unimedvl/smoke-transformers-5.err](data/unimedvl/smoke-transformers-5.err)).
- **Root cause.** Upstream vendors copies of Qwen2 and SigLIP modelling code
  that import transformers 4.x internals; 5.x removed the `"default"` RoPE
  entry. Patching that entry
  ([experiments/issue19_unimedvl_tf5_probe.py](../../../../../experiments/issue19_unimedvl_tf5_probe.py))
  gets the model to run, but it then generates different tokens
  ([smoke-transformers-5-patched.json](data/unimedvl/smoke-transformers-5-patched.json)),
  so a patch here would silently change what the benchmark measures.
- **Decision.** Stay on 4.57.6, the last 4.x. Reported upstream as
  uni-medical/UniMedVL#10; related: huggingface/transformers#39753.
  Dependabot PR #16 (transformers 5.19.0) is superseded and must not be merged.

## F4. OpenCV 5 changes the blur by one grey level

- **Symptom.** `motion_blur_opencv` gives different SHA-256 digests on
  opencv-python 5.0 than on 4.8 and 4.14 for angles off the axes (37 and
  135.5 degrees); 0 and 90 degrees are identical
  ([data/unimedvl/opencv-probe.jsonl](data/unimedvl/opencv-probe.jsonl)).
- **Root cause.** OpenCV 4.x rounds `warpAffine`'s sub-pixel offsets to
  1/32 pixel (largest error against exact bilinear interpolation 0.0215);
  5.0 interpolates float images exactly (7.3e-7). The rotated kernel differs,
  so the blur differs by at most one grey level.
- **Decision.** Take 5.0: it is more accurate, and `motion_blur_opencv` is a
  training augmentation that nothing in upstream's inference path calls.

## F5. The flash-attn wheel on main had the wrong C++ ABI for the new torch

- **Root cause.** PyPI torch wheels use the C++11 ABI since 2.7. Main's wheel
  was `cu12torch2.6cxx11abiFALSE`, right for torch 2.6.0 but unusable with
  torch 2.8.0.
- **Fix.** `flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp313-cp313`,
  which also fixes GHSA-7g5w-pq96-8c5w. The import check confirms the
  extension loads (F2 evidence). A GPU forward pass could not be run: no GPU
  runner is registered for this repository.

## F6. actionlint 1.7.12 rejects the `ubuntu-26.04` labels

- **Symptom.** actionlint's runner-label check rejects `ubuntu-26.04` and
  `ubuntu-26.04-arm` as unknown labels.
- **Root cause.** 1.7.12 is the newest release and predates the image.
  Upstream: rhysd/actionlint#682 (open) and #761 (open, same request), with
  open PRs #683, #723 and #743.
  The same workaround was already posted on #682 by another user on
  2026-10-15, so no duplicate comment was added.
- **Workaround.** The two labels are listed in `.github/actionlint.yaml`
  with a comment saying when to drop them.

## F7. Dependabot alerts: 25 open on main, 8 left

[data/dependabot-alerts-eval.txt](data/dependabot-alerts-eval.txt) checks each
open alert's vulnerable range against the new pins: 17 are fixed by the
update. The other 8 are in torch 2.8.0 and transformers 4.57.6, which are held
for F3 and F5; each is accepted in `scripts/audit_dependencies.py` with the
reason it does not apply to UniMedVL inference (no `torch.load`, SHA-256-pinned
local weights, `HF_HUB_OFFLINE=1`, no training or saving). `pip-audit` on the
3.13 lock: "No known vulnerabilities found, 17 ignored"
([data/unimedvl/pip-audit-lock.log](data/unimedvl/pip-audit-lock.log)).
Alerts close only after merge to `main`.

## F8. New ruff and mypy rules

ruff 0.17 and mypy 2.4 reported real issues, fixed in code rather than
ignored: `zip(..., strict=True)` for parallel sequences, `datetime.UTC`,
`collections.abc.Callable`, stdlib imports moved to module level. Lazy imports
of the heavy optional extras stay inside functions on purpose and are listed
per file under `PLC0415`; `PLR0917` follows the existing `PLR0913` policy.
mypy overrides for `peft` and `huggingface_hub`, which `src/` never imports,
were dropped.

## F9. Dependabot could not update the UniMedVL stack safely

- **Root cause.** torch, the flash-attn wheel (SHA-256 in a shell script)
  and upstream's code must move together; Dependabot PRs #13-#18 each bumped
  one piece (#16 would have broken UniMedVL, F3).
- **Fix (c1cb53c).** `exclude-paths` drops the UniMedVL files from
  Dependabot; the Security workflow still fails on any new advisory against
  them. pip and pre-commit share one weekly multi-ecosystem group, because
  the hooks pin the same ruff, mypy and pillow as the dev extra. A
  `check-jsonschema` hook validates `dependabot.yml`, since Dependabot reports
  a broken file only in its own logs.

## Ruled out

- **llama.cpp "latest release".** `v0.7.0` is a packaging tag whose
  `nightly-tag.txt` says b11669, older than the pinned b11707 (the newest
  `bNNNNN` release on 2026-10-17).
- **CodeQL "sha differs".** The newest release is the CLI tag
  `codeql-bundle-v2.27.2`; the action tag v4.39.0 dereferences to the pinned
  commit.
