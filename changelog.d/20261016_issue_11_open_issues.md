### Added

- `medbench` can serve the pinned UniMedVL checkpoint through an OpenAI-compatible adapter, keeps going when a single item fails, and records peak GPU and host memory, disk use, versions and wall time for every model run (#6, #9).
- A model × runtime × format × task support matrix with estimated ranges, Wilson intervals, negation and dose-unit scoring, label review, grouped splits and a Tesseract OCR baseline (#9).
- The default-branch rulesets live in `.github/rulesets/`, with `scripts/repository_rulesets.py` to check them or apply them after confirmation. The Security workflow warns about drift on pull requests and fails on main. `docs/repository-policy.md` documents the required checks, the release-bot bypass, fork pull requests, tokens and the self-hosted GPU runner (#10).
- Before any upload, `scripts/check_distribution.py` checks the wheel and sdist. It covers the PEP 639 license expression and license files, the manifest against `src/`, the wheel `RECORD` and the `medbench` entrypoint, then runs an offline install smoke test. `docs/publishing.md` lists the PyPI trusted-publisher settings a maintainer enters and how to make the first release (#8).
- `scripts/measure_training_memory.py` measures a real full, LoRA or QLoRA training step. It records the loading, step and checkpoint peaks, optimizer state and image tokens, checks free memory before allocating, and compares the result with the estimate. CPU runs of SmolLM2-135M and SmolVLM-256M are saved in `docs/case-studies/issue-1/data/training-memory-cpu-20261016/` (#9).
- `medbench estimate` takes `--ctx-size`, `--parallel`, `--gpus` and `--scaling-efficiency`, so the KV cache a server reserves, concurrent requests and multi-GPU speed-up are explicit inputs (#9).
- Tests, distribution checks, compatibility runs, dependency audits, the docs build, the real-sample download and the OCR baseline now run under `medbench monitor`. Each uploads a `resources-*` report and adds a row to the job summary. On macOS, `medbench monitor` reads host memory from `vm_stat`. The measured peaks of every workflow are in `docs/hardware-workflows.md` (#9).
- Dependabot opens weekly update pull requests for actions and Python dependencies, after a seven-day cooldown (#10).

### Changed

- Every GitHub Action is pinned to a commit hash with its release in a comment (#10).
- Each terminal status gate has a unique name, such as `Pipeline Status (CI/CD)`, and Compatibility runs on every pull request, so these gates can be required checks (#10).
- The sdist contains only `src/`, the changelog, the license files and the build metadata. It no longer picks up CI logs, experiments or local tool output (#8).
- The GPU job runs only when dispatched by hand from main, uses the `gpu` environment, and removes what it downloaded, even after a failure (#10).

### Fixed

- Fine-tuning estimates count the fp32 logits of the loss, which a 262 k-token Gemma 3 vocabulary makes larger than the activations. QLoRA of MedGemma 27B is 28 GB by formula at 2 048 tokens, so the earlier claim that it fits one 24 GB GPU is now a labelled hypothesis (#9).
- Free memory inside a cgroup limit counts reclaimable page cache, as the kernel does (#9).
- The distribution install smoke test resolves the `dist/` path before it changes directory (#8).
- The Web Archive fallback no longer reports links from lychee's redirect and suggestion sections as broken.

### Security

- Raised the UniMedVL runtime pins of torch, pillow, sentencepiece, opencv-python and fonttools past known high and critical advisories. Each remaining transformers, torch or accelerate advisory is accepted with a reason, and Dependency Review allows only the GHSA aliases of those accepted advisories (#6, #10).
