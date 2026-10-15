### Fixed

- Fail the release preflight with copy-paste trusted-publisher settings when
  PyPI answers `422 invalid-publisher` or `invalid-pending-publisher`, instead
  of reporting "no verdict" and stopping `main` with no instruction (#4).
- Replace mypy's deprecated `strict_concatenate` option with `extra_checks`,
  which removes the deprecation warning from every lint run.
