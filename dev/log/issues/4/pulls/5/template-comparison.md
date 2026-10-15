# Comparison with the Python template

Template: [link-foundation/python-ai-driven-development-pipeline-template](https://github.com/link-foundation/python-ai-driven-development-pipeline-template) at `533e0b7` (its `main` on 2026-10-15).

Method: `diff -rq` of `.github/` and `scripts/`, plus a byte comparison of `pyproject.toml`, `.pre-commit-config.yaml`, `.secretlintrc.json` and `.gitignore`, against this branch.

## File tree

| File | State | Notes |
| --- | --- | --- |
| `.github/workflows/release.yml` | differs | This PR: `if-no-files-found: error` on the dist upload (F8), `queue: max` on both release writers (F9), action majors (F10) |
| `.github/workflows/workflows.yml` | differs | This PR: online pedantic zizmor pass (F5), header comment on `queue`, action majors |
| `.github/workflows/links.yml` | differs | This PR: directory input instead of globs (F4), action majors |
| `.github/workflows/security.yml` | differs | Repository-only Dependency graph check (F2), action majors |
| `.github/workflows/docs.yml` | differs | Action majors only |
| `.github/actionlint.yaml` | repository only | Ignores the one `queue` message (F9). It also declares the self-hosted `gpu` runner label used by `model-tests.yml`. |
| `.github/workflows/model-tests.yml` | repository only | Model tests of this project. This PR fixes its `paths:` filter (F7). |
| `scripts/preflight-credentials.sh` | differs | This PR: 422 error codes (F1) |
| `scripts/detect_code_changes.py` | differs | This PR: `.sh` and `.json` (F6) |
| `scripts/check-dependency-graph.sh` | repository only | Dependency graph check behind dependency review (F2) |
| `scripts/medbench-model-run.sh` | repository only | Used by `model-tests.yml` |
| every other file under `.github/` and `scripts/` | identical | |
| `pyproject.toml` | differs | Project dependencies, plus this PR's `extra_checks` (F3). Name and URLs are still the template's placeholders (R2a). |
| `.pre-commit-config.yaml`, `.secretlintrc.json`, `.gitignore` | identical | |

No template CI file is missing here, so every template practice is in use. Besides those, three practices proposed upstream but not yet merged were adopted:

- [#105](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/105) `queue: max` (F9)
- [#106](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/106) lychee directory input and the current checkout and setup-python majors (F4, F10)
- [#107](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/107) `.sh` counts as code (F6)

[#104](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/104) (Dependabot changeset exemption) does not apply, because there is no Dependabot here (see [analysis.md](analysis.md#considered-and-left-unchanged)).

## Which findings exist in the template

| Finding | In the template? | Evidence |
| --- | --- | --- |
| F1 PyPI 422 treated as unknown | Yes | [preflight L129-L144](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/blob/533e0b7dff1587ba606867c641f149539878e355/scripts/preflight-credentials.sh#L129-L144). Its `main` run 38041366369 fails the same way ([excerpt](ci-logs/template-cicd-38041366369.txt)). |
| F2 Dependency review without a graph check | Yes | [security.yml L70-L88](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/blob/533e0b7dff1587ba606867c641f149539878e355/.github/workflows/security.yml#L70-L88). The template repository itself has the graph enabled, so its own runs are green. |
| F3 `strict_concatenate` | Yes | [pyproject.toml L116](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/blob/533e0b7dff1587ba606867c641f149539878e355/pyproject.toml#L116). The warning is in run 38041366369. |
| F4 lychee glob | Yes | Already reported as #106 |
| F5 offline pedantic zizmor | Yes | [workflows.yml L111-L116](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/blob/533e0b7dff1587ba606867c641f149539878e355/.github/workflows/workflows.yml#L111-L116). The warning is in Workflows run 38041366370 ([excerpt](ci-logs/template-workflows-38041366370.txt)). |
| F6 `.sh`/`.json` detection | Yes | `.sh` already reported as #107. `.json` was added there as a comment. |
| F7 model-tests paths | No | `model-tests.yml` exists only here |
| F8 dist upload `warn` | Yes | [release.yml L429-L433](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/blob/533e0b7dff1587ba606867c641f149539878e355/.github/workflows/release.yml#L429-L433). Run 38041366369 logs `if-no-files-found: warn`. |
| F9 `queue: max` | Yes | Already reported as #105. A simpler config-file variant was added there as a comment. |
| F10 action majors | Yes, partly | checkout and setup-python already in #106. Only this repository's `model-tests.yml` uses `actions/cache`. |

## Reported upstream

New issues, filed on 2026-10-15 after a duplicate check against all of the template's open and closed issues:

- [#108](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/108): Release preflight reports PyPI 422 as "no verdict" (F1)
- [#109](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/109): mypy `--strict-concatenate is deprecated` (F3)
- [#110](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/110): Pedantic zizmor pass runs offline (F5)
- [#111](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/111): dist upload uses `if-no-files-found: warn` (F8)
- [#112](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/112): Dependency Review fails every PR when the Dependency graph is disabled (F2)

Comments on existing issues:

- [#107 (comment)](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/107#issuecomment-6191229980): `.json` has the same gap as `.sh`
- [#105 (comment)](https://github.com/link-foundation/python-ai-driven-development-pipeline-template/issues/105#issuecomment-6191230506): put the actionlint ignore in `.github/actionlint.yaml`
