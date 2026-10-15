### Fixed

- Fail the release preflight with copy-paste trusted-publisher settings when
  PyPI answers `422 invalid-publisher` or `invalid-pending-publisher`, instead
  of reporting "no verdict" and stopping `main` with no instruction (#4).
- Replace mypy's deprecated `strict_concatenate` option with `extra_checks`,
  which removes the deprecation warning from every lint run.
- Give lychee the repository directory with `--extensions md,html` instead of
  a `./**/*.html` glob that matched no file and warned "No files found for this
  input source" on every link check. The same files are checked, and
  `dev/log` (collected CI logs) is excluded.
