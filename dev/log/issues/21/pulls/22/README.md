# Issue #21 / PR #22: update all dependencies in all languages

- Issue: <https://github.com/konard/medical-neural-network-ml/issues/21>
- Pull request: <https://github.com/konard/medical-neural-network-ml/pull/22>

| File | Contents |
| --- | --- |
| [issue.md](issue.md) | Snapshot of the issue text |
| [requirements.md](requirements.md) | Every requirement, its status and evidence, what could not be done, assumptions |
| [versions.md](versions.md) | Per-ecosystem tables: version on `main`, on this branch, latest on the registry, reason when not latest |
| [analysis.md](analysis.md) | Findings F1-F16: symptom, root cause, fix commit, test; what was ruled out |
| [research.md](research.md) | Sources read, the major versions crossed, new features adopted, verification, upstream reports, best practices, tools considered |
| [data/](data/) | Registry lookups, action SHA checks, Dependabot alerts, torch and sympy metadata, flash-attn assets, the cross-stack comparison, llama.cpp checks, mlx-vlm's API, the Dependabot replays, and local verification logs |
| [ci-logs/](ci-logs/) | Logs of the failing Dependabot pull requests (#13, #14, #16), of the failures on this branch (F10, F11, F16), and of the main-only checks (F14) |

## Outcome

- **Every ecosystem is current.**
  - The UniMedVL runtime moves from torch 2.8.0 and transformers 4.57.6 to
    torch 2.14.1 and transformers 5.20.1, on Python 3.14. The lock grows
    from 55 to 68 packages.
  - mlx-vlm moves to 0.7.9 and llama.cpp to b11738. The GPU build moves
    from CUDA 12.8 to 13.4.
  - Every other Python floor, action and hook was already the newest
    release, and this was checked again.
- **Left behind, with a written reason.** 16 lock rows are exact pins of
  torch 2.14.1 (the CUDA 13.0 libraries, cuDNN, cuSPARSELt, NCCL and NVSHMEM)
  or sympy 1.14.0 (`mpmath<1.4`). The UniMedVL Python stays at 3.14 because
  torch and triton publish no cp315 wheels. See
  [versions.md §2](versions.md#2-unimedvl-runtime-scriptsunimedvl-requirementsin--txt).
- **Dependabot's pull requests #13-#17.** Their updates are all in `main`
  or here, at newer versions. The tests that made them fail now check what
  the code needs, not exact versions (F1).
- **What held the lock back, removed.** torch's own `varlen_attn` replaces
  the flash-attn wheel (F2), and one transformers 4.x RoPE helper is put back
  for upstream's code (F3). Upstream's `chat()` now runs on the lock in CI on
  the CPU and matches golden tokens and logits. That job failed until the test
  fixture stopped resolving the model directory relative to upstream's code
  (F16).
- **Dependabot updates the lock.** A `uv` entry for `/scripts` replaces the
  hand-updated lock (F7). Packages released together with torch or
  transformers are named in the `.in` without a version, so a grouped
  update can move them.
- **Bug found and fixed.** The GPU job unpacked llama.cpp's CUDA runtime
  where the CUDA backend could not find it (F6).
- **Advisories.** None are accepted any more. pip-audit finds none in the
  new tree, and the 8 open alerts on `main` are fixed by the new pins (F5).
- **Upstream reports.** uni-medical/UniMedVL#10,
  Dao-AILab/flash-attention#2995, pytorch/pytorch#201457 and
  dependabot/dependabot-core#16390. See
  [research.md](research.md#upstream-reports).

## Not verifiable here

- **UniMedVL and llama.cpp on a CUDA 13 GPU.** No GPU runner is
  registered, so `test_cuda_uses_varlen_attn_and_matches_the_reference` and
  the GPU job wait for the first GPU run. The GPU runner needs an R580+
  driver and a Turing or newer GPU (F8).
- **mlx-vlm 0.7.9 on Apple silicon.** The backend's calls were checked
  against 0.7.9's source, and the mocked tests pass (F9).
- **The two checks that fail on `main` only** ("Release Preflight
  (credentials)" and "Branch Rulesets") need a repository admin. Each
  failure message names the action (F14).
