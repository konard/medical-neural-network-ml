### Added

- `medbench` package for issue #1: typical medical tasks (diagnosis from
  symptoms and DDXPlus interviews, handwritten prescription and printed
  history OCR, X-ray, MRI and ultrasound reading), metrics, a resource
  calculator and backends for OpenAI-compatible servers, Tesseract and a
  naive Bayes baseline.
- Real, openly licensed test files (public domain, CC0, CC BY, CC BY-SA)
  listed in `src/medbench/data/real_samples.json`. They are downloaded at test
  time, checked by SHA-256 and credited in `ATTRIBUTION.md`; none are stored
  in the repository. `medbench sources` and `medbench fetch` show and
  download them, pacing requests and honouring Wikimedia's `Retry-After`.
- `Model Tests` workflow that checks every real sample file, runs MedGemma
  4B on a hosted CPU runner and,
  when started by hand, MedGemma 4B or 27B on a self-hosted GPU runner, using
  `scripts/medbench-model-run.sh` with SHA-256 pinned downloads.
- Research case study in `docs/case-studies/issue-1`.

### Fixed

- Pull requests no longer fail Dependency Review when the repository's
  Dependency graph setting is off: `scripts/check-dependency-graph.sh` skips
  the review with a warning, and a new `Dependency Graph Enabled` job fails
  `main` and the weekly scan with the setting a maintainer has to enable.
