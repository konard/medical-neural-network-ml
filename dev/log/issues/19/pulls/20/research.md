# Research: migration notes, new features, verification, upstream reports

Everything here was checked on 2026-10-17. Version tables are in
[versions.md](versions.md), and findings with root causes are in
[analysis.md](analysis.md).

## Sources

These are the pages read for this pull request. The changelogs were read by
a research sub-agent, and its report is summarised under
[Major versions](#major-versions). The rest were fetched directly.

- Python release status: <https://devguide.python.org/versions/> (3.9 and
  3.10 end of life, 3.15 current).
- pytest: <https://docs.pytest.org/en/stable/changelog.html>,
  <https://docs.pytest.org/en/stable/deprecations.html>,
  <https://docs.pytest.org/en/stable/reference/customize.html>. The `strict`
  option was confirmed with `pytest --help` from the installed 9.1.1,
  because the reference page was too long to fetch.
- pytest-asyncio: <https://pytest-asyncio.readthedocs.io/en/stable/reference/changelog.html>.
- pytest-cov: <https://pytest-cov.readthedocs.io/en/latest/changelog.html>.
- coverage `[run] patch`: <https://coverage.readthedocs.io/en/latest/config.html>.
- mypy: <https://mypy.readthedocs.io/en/stable/changelog.html>.
- Pillow: the release notes for 11.0.0, 12.0.0 and 13.0.0, and
  <https://pillow.readthedocs.io/en/stable/deprecations.html>.
- Sphinx: <https://www.sphinx-doc.org/en/master/changes/8.0.html>,
  [8.1](https://www.sphinx-doc.org/en/master/changes/8.1.html),
  [8.2](https://www.sphinx-doc.org/en/master/changes/8.2.html),
  [9.0](https://www.sphinx-doc.org/en/master/changes/9.0.html), and
  <https://www.sphinx-doc.org/en/master/changes/index.html> for 9.1.
- myst-parser: <https://myst-parser.readthedocs.io/en/latest/develop/_changelog.html>.
- pre-commit-hooks: <https://github.com/pre-commit/pre-commit-hooks/releases/tag/v6.0.0>.
- ruff: <https://raw.githubusercontent.com/astral-sh/ruff/main/BREAKING_CHANGES.md>
  and the v0.17.1 hook file in astral-sh/ruff-pre-commit.
- numpy: <https://numpy.org/doc/stable/release/2.5.0-notes.html> and
  <https://numpy.org/doc/stable/numpy_2_0_migration_guide.html>.
- OpenCV: <https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration>.
  The 5.0 change log still says "TBD", so the `warpAffine` change was measured
  instead ([analysis.md F4](analysis.md#f4-opencv-5-changes-the-blur-by-one-grey-level)).
- Dependabot: <https://docs.github.com/en/code-security/dependabot/working-with-dependabot/dependabot-options-reference>
  (`multi-ecosystem-groups`, `exclude-paths`, `cooldown`).
- Process: hive-mind's
  [DEPENDENCY-UPDATE-BEST-PRACTICES.md](https://github.com/link-assistant/hive-mind/blob/main/docs/DEPENDENCY-UPDATE-BEST-PRACTICES.md).
- Registries: the PyPI JSON API and the GitHub releases and tags API, through
  [experiments/issue19_resolve_latest.py](../../../../../experiments/issue19_resolve_latest.py)
  ([data/latest.json](data/latest.json)).

## Major versions

Each major bump below was checked against this repository. "Affects us?"
names the evidence, and "Change" is what this pull request did about it.

| Dependency | Breaking change | Affects us? | Change |
| --- | --- | --- | --- |
| pytest 8.3 → 9.1 | `py.path` hook arguments and Node `fspath` removed; marks on fixtures removed; `importorskip` catches only `ModuleNotFoundError`; 9.1 deprecates class-scoped fixtures as instance methods, generator `parametrize` values and `getfixturevalue` in teardown | No. There is no `conftest.py` and no such hooks, fixtures or values. The only `importorskip("PIL")` raises `ModuleNotFoundError` | Adopted the native `[tool.pytest]` table and `strict = true` (see below) |
| pytest-asyncio 0.24 → 1.4 | `event_loop` fixture removed; a start-up warning when `asyncio_default_fixture_loop_scope` is unset | **Yes, the warning.** pytest's `-W` never sees it, but `python -W error::DeprecationWarning -m pytest` stops with an INTERNALERROR ([analysis.md F10](analysis.md#f10-a-pytest-asyncio-warning-that-pytests--w-never-sees)) | `asyncio_default_fixture_loop_scope = "function"` |
| pytest-cov 6 → 7 | The `.pth` subprocess measurement was removed; coverage's `patch = ["subprocess"]` replaces it | **Yes.** `tests/test_medbench_monitor.py` runs `python -m medbench.cli` ([F11](analysis.md#f11-pytest-cov-7-stopped-measuring-subprocesses)) | `[tool.coverage.run] patch = ["subprocess"]` |
| mypy 1.13 → 2.4 | `--local-partial-types` and `--strict-bytes` are on by default; native parser; `--force-uppercase-builtins` removed | No. `mypy src/` and `mypy --strict src/` both pass, and none of the removed flags were set | Dropped two unused overrides ([F8](analysis.md#f8-new-ruff-and-mypy-rules)) |
| Pillow 10 → 13 | 11 removed `PyAccess` and changed `I;16` resampling; 12 removed `ImageMath.eval` and `BGR;*`; 13 restricts `fromarray(mode=...)` | No. `src/medbench/real_data.py` converts `I;16` to `L` before `thumbnail`, and `fromarray` is called without `mode` | None |
| Sphinx 7.4 → 9.1 | 8.0 removed the old `intersphinx_mapping` form; 8.1 removed `sphinx.util.Tee` and changed LaTeX `fvset`; 8.2 removed `MathDomain.has_equations` data and `kbd.compound`; 9.0 rewrote autodoc; **9.1 requires Python 3.12** | **Only 9.1's Python floor.** `docs/conf.py` uses the dict form of `intersphinx_mapping` and none of the removed APIs, CSS or LaTeX settings. `sphinx-build -W` is clean ([F12](analysis.md#f12-sphinx-91-needs-python-312)) | Documented that the `docs` extra needs Python 3.12+ |
| myst-parser 3 → 5.1 | 4.0 resolves footnotes later and adds new warnings; 5.0 needs Python ≥ 3.11 and Sphinx 8-9; 5.1 changes `myst_gfm_only` | No. There are no `myst_*` settings, and the `-W` build is clean | None |
| pre-commit-hooks 5 → 6 | `check-byte-order-marker` and `fix-encoding-pragma` removed | No. Neither hook is used | None |
| ruff 0.8 → 0.17 | Hook id `ruff` renamed `ruff-check`; 0.16 formats Python in Markdown; 0.17 changes ICN001 and the default rule set | **Yes**: the hook id, and Markdown formatting reaching the evidence directories ([F13](analysis.md#f13-ruff-016-formats-python-inside-markdown)) | Renamed the hook; `extend-exclude` matches pre-commit's `exclude`. Rules are selected explicitly, so the default-set change does not apply |
| numpy 1.24 → 2.5 (UniMedVL runtime) | 2.0: NEP 50 promotion, about 100 namespace removals, `copy=False` semantics, ABI break. 2.5: `row_stack` and others removed | Checked with the guide's own tool: `ruff --select NPY201` on upstream UniMedVL `f51aa0a` (29 files) and on this repository reports nothing, and no `copy=False` / `np.core` / `__array__` use ([data/unimedvl/numpy2-migration-check.txt](data/unimedvl/numpy2-migration-check.txt)). The CPU smoke test generates the same tokens as on numpy 1.24 ([F2](analysis.md#f2-python-310-held-seven-unimedvl-packages-behind)) | `NPY` rules enabled in ruff |
| opencv-python 4.8 → 5.0 (UniMedVL runtime) | `warpAffine`/`remap` interpolation is exact; ML, Haar and HOG moved to contrib | Yes, the interpolation: one grey level of difference in a training augmentation ([F4](analysis.md#f4-opencv-5-changes-the-blur-by-one-grey-level)) | Accepted and recorded in `scripts/unimedvl-requirements.in` |
| torch 2.6 → 2.8, transformers 4.49 → 4.57 (UniMedVL runtime) | flash-attn needs a wheel per torch version and C++ ABI; transformers 5 removed the `default` RoPE entry | Yes ([F3](analysis.md#f3-transformers-5-breaks-upstream-unimedvl), [F5](analysis.md#f5-the-flash-attn-wheel-on-main-had-the-wrong-c-abi-for-the-new-torch)) | New wheel; transformers held at 4.57.6, reported upstream |
| Python 3.9 → 3.11 floor; CI 3.13 → 3.15 | 3.12+ reports unclosed `HTTPError` objects as `ResourceWarning` | Yes ([F14](analysis.md#f14-unclosed-http-error-answers-on-python-312)) | Errors are closed after they are read |

## New features adopted

These are places where a newer release replaced something the repository did
by hand, or did not do at all.

| New feature | Replaces | Where |
| --- | --- | --- |
| pytest 9 native `[tool.pytest]` table with real TOML types, and `strict = true` | `[tool.pytest.ini_options]` with string values, and `--strict-markers` in `addopts`. `strict` also turns on `strict_config`, `strict_xfail` and `strict_parametrization_ids`, and future strictness options | `pyproject.toml` |
| coverage 7.10+ `[run] patch = ["subprocess"]` | pytest-cov's removed `.pth` hook | `pyproject.toml`, test `test_coverage_follows_python_subprocesses` |
| Python 3.11 stdlib `tomllib` | `try: import tomllib / except: tomllib = None` fallbacks and their error branches | `scripts/check_distribution.py`, `scripts/smoke_test_published_package.py` |
| `datetime.UTC` (3.11) | `timezone.utc` | `scripts/bump_version.py` and others ([F8](analysis.md#f8-new-ruff-and-mypy-rules)) |
| `zip(..., strict=True)` | Silent truncation of parallel sequences | `src/medbench/metrics.py`, `benchmark.py`, `synthetic_cases.py` |
| `uv pip compile` lock | 20 hand-written pins with no transitive pins | `scripts/unimedvl-requirements.in` → `.txt` |
| Dependabot `multi-ecosystem-groups`, `exclude-paths`, `patterns` | Separate pip and pre-commit PRs that fail the "same version" tests; a hand-maintained "left out" comment | `.github/dependabot.yml` ([F9](analysis.md#f9-dependabot-could-not-update-the-unimedvl-stack-safely)) |
| `check-jsonschema` hooks (`check-dependabot`, `check-github-workflows`) | Nothing: a broken `dependabot.yml` was reported only in Dependabot's own logs | `.pre-commit-config.yaml` |
| ruff `NPY` rules (NPY201 is the numpy 2 guide's migration check) | A manual read of the numpy guide for every future numpy change | `pyproject.toml` |
| ruff 0.16 Markdown formatting | Nothing. It is now applied to the documentation, and the captured evidence is excluded | `pyproject.toml` |

Considered and not adopted:

- pytest 9's experimental `subtests` fixture. It is experimental, and the
  loops it would replace already report the failing value.
- coverage's `sigterm = true`. It added no coverage over
  `patch = ["subprocess"]` in the measurement
  ([data/coverage/](data/coverage/)).

## Verification

All checks were run locally on the final code commit, then in CI.

| Check | Result | Evidence |
| --- | --- | --- |
| pytest on Python 3.15.0 and 3.11.17, run as `python -W error::DeprecationWarning -W always::ResourceWarning` with the same flags passed to pytest | 947 passed, 45 skipped (opt-in network, model and GPU tests), no warnings | [data/verification/pytest-py315.log](data/verification/pytest-py315.log), [pytest-py311.log](data/verification/pytest-py311.log) |
| ruff check, ruff format, mypy, file-size check, `sphinx-build -W`, pre-commit | All clean | [data/verification/checks.log](data/verification/checks.log) |
| UniMedVL on the new lock (CPU, tiny random model) | Same token ids as on main's pins | [analysis.md F2](analysis.md#f2-python-310-held-seven-unimedvl-packages-behind) |
| `pip-audit` on the UniMedVL lock | No known vulnerabilities, 17 accepted with reasons | [data/unimedvl/pip-audit-lock.log](data/unimedvl/pip-audit-lock.log) |
| CI on the pull request | All workflows green on the final commit | The pull request's checks tab |

## Upstream reports

| Blocker or workaround here | Upstream | What was done |
| --- | --- | --- |
| transformers held at 4.57.6, because upstream UniMedVL's vendored Qwen2 code fails on 5.x | [uni-medical/UniMedVL#10](https://github.com/uni-medical/UniMedVL/issues/10) | **Opened**. It includes the traceback, the versions, the patch that gets past the error and why that patch is not enough (different tokens), and a suggested fix: declare `transformers<5`, or replace the `"default"` lookup with a local default RoPE function and compare outputs against 4.x |
| The same RoPE change, from the transformers side | [huggingface/transformers#39753](https://github.com/huggingface/transformers/issues/39753) | Existing and open; linked from #10, nothing added |
| `ubuntu-26.04` labels listed in `.github/actionlint.yaml` | [rhysd/actionlint#682](https://github.com/rhysd/actionlint/issues/682), #761; PRs #683, #723, #743 | Existing and open. Another user posted the same workaround on #682 on 2026-10-15, so no duplicate comment was added |
| `queue:` ignored in `.github/actionlint.yaml` (from issue #4, unchanged here) | [rhysd/actionlint#654](https://github.com/rhysd/actionlint/pull/654) | Existing open PR; still needed with actionlint 1.7.12 |

No report was needed for the pytest-asyncio warning, pytest-cov, Sphinx or
ruff findings. Each is documented, intended upstream behaviour, and this
repository adapted to it.

## Best practices

How each principle of hive-mind's DEPENDENCY-UPDATE-BEST-PRACTICES.md was
applied:

1. **Table before changing anything.**
   [versions.md](versions.md) was resolved from PyPI and GitHub by script
   ([data/latest.json](data/latest.json)).
2. **A written reason for anything left behind.** Each held version has a
   "Why not latest" cell in [versions.md](versions.md), and the reasons are
   repeated as comments in `scripts/unimedvl-requirements.in`.
3. **Cross majors deliberately.** See [Major versions](#major-versions) above.
   No constraint was loosened, and no test was skipped. The one held major
   (transformers 5) has a failing reproduction and an upstream report.
4. **Adopt new features, delete hand-rolled copies.** See
   [New features adopted](#new-features-adopted). The `tomllib` fallbacks,
   the hand-pinned UniMedVL list and the `--strict-markers` flag are gone.
5. **Honest constraints.** Every floor is the newest release. The only upper
   bound left (`ruff<0.18`) admits the current release and is explained.
   The UniMedVL lock is regenerated and committed.
6. **One version per dependency.** pillow, ruff, mypy, the docs pins and the
   UniMedVL Python each have a single version, and tests enforce this
   ([requirements.md R12](requirements.md)).
7. **Update the toolchain.** `requires-python` is now `>=3.11`, CI uses
   Python 3.15 on `ubuntu-26.04` and `macos-26`, and UniMedVL runs on 3.13.
8. **Green CI, zero new deprecation warnings.** See
   [Verification](#verification). Two warnings that the update surfaced were
   fixed at the source, not filtered: the pytest-asyncio warning (F10) and the
   `ResourceWarning`s (F14).
9. **Audit the tree you ship.** `pip-audit` was run on the lock as
   committed, and the Dependabot alerts were evaluated against the new pins
   ([analysis.md F7](analysis.md#f7-dependabot-alerts-25-open-on-main-8-left)).
   The Security workflow runs this audit on a schedule.
10. **Report blockers upstream.** See [Upstream reports](#upstream-reports).

## Tools considered

| Tool | Used? | Why |
| --- | --- | --- |
| uv (`uv pip compile`) | Yes | Compiles the UniMedVL lock for a Python and platform other than the host's (`--python-version 3.13 --python-platform x86_64-manylinux_2_28`), and explains resolver conflicts (for example "sphinx==9.1.0 depends on Python>=3.12") |
| pip-audit | Yes | Already used by the Security workflow; run on the new lock |
| check-jsonschema | Yes | Validates `dependabot.yml` and the workflows before push |
| Dependabot multi-ecosystem groups | Yes | Keeps pip and pre-commit pins moving together |
| Renovate | No | It could also group across ecosystems, but Dependabot already runs here with cooldown and SHA-pinned actions, and the new groups cover the gap. Switching tools was not asked for |
| `pre-commit autoupdate` | Yes | Bumped the hook revisions |
