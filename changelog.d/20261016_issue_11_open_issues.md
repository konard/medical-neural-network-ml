### Added

- `medbench` can serve the pinned UniMedVL checkpoint through an OpenAI-compatible adapter, keeps going when a single item fails, and records peak GPU and host memory, disk use, versions and wall time for every model run (#6, #9).
- A model × runtime × format × task support matrix with estimated ranges, Wilson intervals, negation and dose-unit scoring, label review, grouped splits and a Tesseract OCR baseline (#9).
- The default-branch rulesets live in `.github/rulesets/`, with `scripts/repository_rulesets.py` to check them or apply them after confirmation. The Security workflow warns about drift on pull requests and fails on main. `docs/repository-policy.md` documents the required checks, the release-bot bypass, fork pull requests, tokens and the self-hosted GPU runner (#10).
- Before any upload, `scripts/check_distribution.py` checks the wheel and sdist. It covers the PEP 639 license expression and license files, the manifest against `src/`, the wheel `RECORD` and the `medbench` entrypoint, then runs an offline install smoke test. `docs/publishing.md` lists the PyPI trusted-publisher settings a maintainer enters and how to make the first release (#8).
- Dependabot opens weekly update pull requests for actions and Python dependencies, after a seven-day cooldown (#10).

### Changed

- Every GitHub Action is pinned to a commit hash with its release in a comment (#10).
- Each terminal status gate has a unique name, such as `Pipeline Status (CI/CD)`, and Compatibility runs on every pull request, so these gates can be required checks (#10).
- The sdist contains only `src/`, the changelog, the license files and the build metadata. It no longer picks up CI logs, experiments or local tool output (#8).
- The GPU job runs only when dispatched by hand from main, uses the `gpu` environment, and removes what it downloaded, even after a failure (#10).

### Security

- Raised the UniMedVL runtime pins of torch, pillow, sentencepiece, opencv-python and fonttools past known high and critical advisories. Each remaining transformers, torch or accelerate advisory is accepted with a reason, and Dependency Review allows only the GHSA aliases of those accepted advisories (#6, #10).
