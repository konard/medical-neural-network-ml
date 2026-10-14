### Added

- `medbench` package for issue #1: synthetic medical tasks (diagnosis from
  symptoms, prescription and history OCR, X-ray, MRI and ultrasound phantoms),
  metrics, a resource calculator and backends for OpenAI-compatible servers,
  Tesseract and a naive Bayes baseline.
- `Model Tests` workflow that runs MedGemma 4B on a hosted CPU runner and,
  when started by hand, MedGemma 4B or 27B on a self-hosted GPU runner, using
  `scripts/medbench-model-run.sh` with SHA-256 pinned downloads.
- Research case study in `docs/case-studies/issue-1`.

### Fixed

- Pull requests no longer fail Dependency Review when the repository's
  Dependency graph setting is off: `scripts/check-dependency-graph.sh` skips
  the review with a warning, and a new `Dependency Graph Enabled` job fails
  `main` and the weekly scan with the setting a maintainer has to enable.
