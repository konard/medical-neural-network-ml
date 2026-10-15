# CI/CD findings, root causes and fixes

Baseline: every workflow run on `main` at `fd7703e` (the runs listed in the issue), plus a review of every file under `.github/` and `scripts/`.

The issue's snapshot reported "0 not passing" because two of its five runs were still `in_progress` when it was filed. Both finished red:

| Workflow | Run | Result |
| --- | --- | --- |
| CI/CD Pipeline | [38725259374](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259374) | failure ([excerpt](ci-logs/main-cicd-38725259374.txt)) |
| Security | [38725259420](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259420) | failure ([excerpt](ci-logs/main-security-38725259420.txt)) |
| Workflows | [38725259341](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259341) | success with a warning ([excerpt](ci-logs/main-workflows-38725259341.txt)) |
| Broken Link Checker | [38725259357](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259357) | success with a warning ([excerpt](ci-logs/main-links-38725259357.txt)) |
| Docs | [38725259366](https://github.com/konard/medical-neural-network-ml/actions/runs/38725259366) | success (notice only) |

Every grep hit for warning, error or deprecation in those runs is in [ci-logs/main-fd7703e-warnings.txt](ci-logs/main-fd7703e-warnings.txt).

Kinds of finding used below: **E** is an error, **W** a warning, **FN** a false negative (green or skipped when it should fail) and **FP** a false positive (red for a reason no commit can fix).

## F1 (E, FP): Release preflight fails `main` with "verified nothing (1 unknown)"

- **Symptom.** `PyPI answered 422 to the mint-token probe (no verdict on trusted publishing)`, then `verified nothing (1 unknown) -- refusing to release on an unproven credential set`. The message did not say what was wrong or who could fix it.
- **Root cause.** `scripts/preflight-credentials.sh` mapped 422 to `unknown`. PyPI's mint-token endpoint (`warehouse/oidc/views.py`, `_invalid()`) returns 422 with a JSON body for *every* refusal:
  - `{"message":"Token request failed","errors":[{"code":...,"description":...}]}`
  - Codes: `invalid-payload`, `not-enabled`, `invalid-token`, `invalid-pending-publisher`, `rate-limited`, `invalid-reuse-token` and `invalid-publisher`.
  - `invalid-publisher` and `invalid-pending-publisher` are a definite answer: no trusted publisher matches this repository and workflow. That is a verdict, not an unknown (best practices 16 "Report `unknown`, never a guess" and 17 "A policy refusal is a decision").
- **Fix (dc38fd4).**
  - The preflight reads the error code from the body. The two publisher codes become `FAIL`, and the message gives the publisher fields to enter (owner, repository, workflow file, environment).
  - It probes `GET /pypi/<project>/json` so it can link the right PyPI page: a pending publisher for a new project, or the existing project's publishing settings plus a rename hint, because `my-package` belongs to someone else.
  - The project name is read from `[project].name`.
  - The other codes and a 422 without a body stay `unknown`.
- **Evidence after the fix.** The PR run ([excerpt](ci-logs/pr-cicd-38728238909.txt)) shows the real answer: `FAIL: PyPI refused to mint an upload token (422 invalid-publisher: valid token, but no corresponding publisher (Publisher with matching claims was not found))`, followed by the exact settings. It is a warning on the PR and will be an error on `main` until a maintainer acts (see [requirements.md](requirements.md#r2a-and-r2b-what-a-maintainer-must-do)).
- **Template.** Same failure on the template's own `main` ([excerpt](ci-logs/template-cicd-38041366369.txt)). Reported upstream as [template #108](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/108).

## F2 (E, needs a human): Security fails on `main` because the dependency graph is disabled

- **Symptom.** `Dependency graph is disabled for konard/medical-neural-network-ml, so dependency review cannot run. A repository admin must enable it: .../settings/security_analysis -> Dependency graph -> Enable.`
- **Analysis.** This is already the intended behaviour. On pull requests the check warns and skips dependency review; on `main` and on the schedule it fails with the setting to change. No commit can fix it, so there is no code change. It is listed as human action R2b.
- **Template.** The template runs `dependency-review-action` with no such check, so a generated repository with the graph disabled gets an unexplained failure. Reported upstream as [template #112](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/112).

## F3 (W): mypy `--strict-concatenate is deprecated`

- **Root cause.** `pyproject.toml` set `strict_concatenate = true`. mypy deprecated the option in favour of `extra_checks`, which covers it.
- **Fix (8b243ac).** `extra_checks = true`. mypy 1.20.2 reports success with no warning, and the PR's lint job shows 0 deprecation lines ([excerpt](ci-logs/pr-cicd-38728238909.txt)).
- **Template.** Same line, and the same warning in its run 38041366369. Reported upstream as [template #109](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/109).

## F4 (W): lychee `[WARN] ./**/*.html: No files found for this input source`

- **Root cause.** `links.yml` passed the globs `'./**/*.md' './**/*.html'`, and the repository has no HTML file.
- **Fix (034d5a3).**
  - lychee now gets the directory `.` with `--extensions md,html --hidden`, as the JS template does.
  - `--exclude-path dev/log` keeps collected CI logs out.
  - Checked with `--dump-inputs`: the same 47 files as before. The PR run checks the same 52 links with no warning ([excerpt](ci-logs/pr-links-38728238822.txt)).
- **Template.** Already reported as template #106; the fix follows it.

## F5 (W): zizmor `running in offline mode by default` in the pedantic pass

- **Root cause.** The second, pedantic zizmor pass runs `pipx run zizmor` without a token. Since zizmor 1.30 it then falls back to offline mode with a warning and skips the audits that query GitHub. The first pass (zizmor-action) is online, because its `token` input defaults to `github.token`.
- **Fix (131bc9c).**
  - The step gets `GH_TOKEN: ${{ github.token }}`; the job token is read-only, with `contents: read`.
  - Locally the online pass evaluates more (15 ignored instead of 12) and still reports no findings.
  - The PR run shows no warning ([excerpt](ci-logs/pr-workflows-38728238810.txt)).
- **Template.** Same step, and the same warning in its run 38041366370 ([excerpt](ci-logs/template-workflows-38041366370.txt)). Reported upstream as [template #110](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/110).

## F6 (FN): change detection ignores `.sh` and `.json`

- **Root cause.** `scripts/detect_code_changes.py` counted `.py .toml .yml .yaml` only. A PR that changed only a pipeline shell script, the medbench sample manifest `src/medbench/data/real_samples.json` or `.secretlintrc.json` skipped lint, tests, build and the changelog check.
- **Fix (7ad4901).** Add `.sh` and `.json`, with tests. `docs/`, `changelog.d/` and `dev/log/` stay excluded.
- **Template.** `.sh` is already reported as template #107. `.json` was added to that report [as a comment](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/107#issuecomment-6191229980).

## F7 (FN): model tests do not run when files they execute change

- **Root cause.** The `pull_request.paths` filter of `model-tests.yml` lacked three files: `tests/test_medbench_real_data.py` (run by `real-data`), `scripts/run-with-budget-warning.sh` (wraps every step) and `pyproject.toml` (the installed dependencies). A PR that broke any of them merged green without running the model tests (best practice 1: run on *relevant* changes, which includes every file the job executes).
- **Fix (5373cd2).** Add the three paths. A test derives the list of files from the workflow and from `scripts/medbench-model-run.sh`, so the next file the jobs start to run fails the test until it is added.
- **Template.** Not applicable: `model-tests.yml` exists only in this repository.

## F8 (W, latent FN): the dist upload only warns when `dist/` is empty

- **Root cause.** The `Upload artifacts` step in `release.yml` used the default `if-no-files-found: warn`. A build that produced nothing would pass, and the failure would surface later, in the publish job, as a download error (best practice 17, "Write logs where the upload step looks": an upload that finds nothing warns and passes). The other critical uploads (Docker digests, the docs site) already used `error`.
- **Fix (bfa20c7).**
  - `if-no-files-found: error`.
  - A test checks every `upload-artifact` step in every workflow. `ignore` is allowed only under `if: always()`, the reports uploaded after a model run that may have failed, where the failed step already explains the missing file.
- **Template.** Same step. Reported upstream as [template #111](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/111).

## F9 (latent FP): a third release cancels the one still waiting

- **Root cause.** Template issue #62 removed `queue: max` from the `main-write` concurrency group because actionlint 1.7.12 rejects the key. GitHub documents it, though: `queue: single` (the default) keeps one pending run and cancels it when the next one arrives; `max` keeps up to 100. So the push release, a manual release and a third writer can cancel a release that was due to run, and it shows as grey.
- **Fix (136a8b3, 8f67b06).**
  - `queue: max` on `auto-release` and `manual-release`, both with `cancel-in-progress: false`, because GitHub rejects `max` combined with `true`.
  - `.github/actionlint.yaml` ignores exactly the message `unexpected key "queue" for "concurrency" section` until rhysd/actionlint#654 ships, so a plain local `actionlint` agrees with CI. Another unknown concurrency key (tested with `queue-size`) still fails actionlint.
  - Tests check the key set, that `queue: max` is never combined with a `cancel-in-progress` that could be true, and that the ignore list contains only that message.
  - The CI/CD Pipeline run on the PR started and passed, so GitHub accepted the syntax.
- **Template.** Already reported as template #105. Its suggestion puts the ignore in CLI args; the config file is noted there as a simpler variant.

## F10 (W, maintenance): first-party actions one major behind

- **Root cause.** `actions/checkout@v6`, `actions/setup-python@v6` and `actions/cache@v5` while v7, v7 and v6 were current. Both v7 majors run on Node 24.
- **Fix (c4246a5).** Moved every workflow, including `model-tests.yml`, which the per-workflow version tests did not cover. A test now checks every workflow file.
- **Template.** Already reported as template #106 for checkout and setup-python.

## Not defects

| Log line | Why it is fine |
| --- | --- |
| `::notice::Skipping Codecov upload because CODECOV_TOKEN is not configured` | Deliberate opt-in notice with the fix in the text (template #27). |
| `::notice::Skipping GitHub Pages deployment. To enable publishing, ...` | Deliberate opt-in notice. |
| CodeQL `"DEPRECATED - Whether to use buildless (standalone) extraction."` and the `"Controls the level of verbosity..."` lines | A schema dump of extractor options printed by `Initialize CodeQL`, not a warning about this repository. The workflow uses `build-mode: none`. |
| `tar ... --warning=no-unknown-keyword` | A tar flag that *suppresses* warnings. |
| `SKIPPED [40] tests/test_medbench_real_data.py ... set MEDBENCH_NETWORK_TESTS=1` | Network tests are opt-in in the unit test job; `model-tests.yml` runs them with downloads. |
| `"result": "skipped"` in `Pipeline Status` | `NEEDS_JSON` of jobs that are skipped by design (release jobs on PRs). |

## Considered and left unchanged

- **Dependabot and the changeset check (template #104).** This repository has no `.github/dependabot.yml` and has never had a Dependabot PR, so the false failure cannot occur here.
- **zizmor-action v0.7.0 and zizmor 1.31.0.** Both were released on 2026-10-15, the day of this PR. The action's digest table had to be checked against the new CLI first. The existing pin is consistent and audited clean.
- **`docs/case-studies` stays excluded from lychee.** A full run over it ([results](ci-logs/lychee-case-studies.txt): 732 links, 26 errors, 4 timeouts) shows that gating on it would be a false-positive generator:
  - 15 are bot-blocking 403/406 answers (nejm.org, openai.com, hhs.gov, ...).
  - 4 are lychee extraction errors on Wikimedia file names containing `_-_` or `(PA)`, truncated at the parenthesis or dash in `ATTRIBUTION.md`, which is a generated record of a CI run.
  - 1 is an expired TLS certificate and 1 is a connection reset.
  - 3 are real 404s in research notes (torchio augmentation page, an FDA guidance URL, an NGC benchmark). They are historical notes of issue #1 and not part of the product docs.

  Keeping the template's exclusion matches best practice 17 (do not fail a commit for the state of third-party sites).
- **`pyproject.toml` metadata.** The name `my-package`, authors and URLs are still the template's placeholders. Renaming the distribution changes what gets published, so it is left to the maintainer (R2a). The preflight message now says so.

## Tests

Each fix has a regression test that failed before the fix:

| Fix | Test |
| --- | --- |
| F1 | `tests/test_preflight_credentials.py`: 9 new cases (publisher codes FAIL with the right PyPI link for a new or an existing project, the name read from `pyproject.toml`, the other codes and a body-less 422 stay unknown) |
| F4 | `tests/test_workflows.py::test_links_workflow_fails_for_every_broken_live_link`: directory input, `--extensions md,html --hidden`, `dev/log` excluded, no globs |
| F5 | `tests/test_workflows.py::test_workflow_audit_job_runs_zizmor`: the pedantic step has `GH_TOKEN` |
| F6 | `tests/test_detect_code_changes.py::test_shell_and_json_changes_activate_jobs`, `test_excluded_folders_win_over_code_extensions` |
| F7 | `tests/test_medbench_model_run.py::test_pull_request_paths_cover_every_file_the_jobs_run` |
| F8 | `tests/test_workflows.py::test_every_artifact_upload_fails_when_it_finds_no_files` |
| F9 | `tests/test_workflows.py::test_queue_max_is_never_combined_with_cancel_in_progress`, the actionlint ignore assertion, `tests/test_release_workflow.py` writer block |
| F10 | `tests/test_workflows.py::test_no_workflow_uses_a_superseded_first_party_major` |

F3 has no unit test: mypy itself is the check, and CI shows the warning gone.
