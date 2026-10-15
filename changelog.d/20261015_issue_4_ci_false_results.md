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
- Count shell scripts (`.sh`) and JSON files (`.json`) as code in change
  detection, so a PR that changes only a pipeline script or the medbench
  sample manifest no longer skips lint, tests, build and the changelog check.
- Move every workflow to `actions/checkout@v7`, `actions/setup-python@v7` and
  `actions/cache@v6`, the current majors, and test that no workflow
  (including `model-tests.yml`) falls behind them again.
- Run the pedantic zizmor pass with the read-only job token, so it no longer
  warns "running in offline mode by default" and runs the same online audits
  as the regular pass.
- Fail the release build when `dist/` is empty (`if-no-files-found: error`)
  instead of warning and letting the publish job fail later.
- Trigger the model tests on pull requests that change
  `tests/test_medbench_real_data.py`, `scripts/run-with-budget-warning.sh` or
  `pyproject.toml`, which the jobs run or install from.
- Restore `queue: max` on the release writers, so a third release no longer
  cancels the one still waiting for the `main-write` group. GitHub documents
  `queue`; actionlint 1.7.12 does not know it yet, so only that one message is
  ignored (in `.github/actionlint.yaml`) until rhysd/actionlint#654 ships.
