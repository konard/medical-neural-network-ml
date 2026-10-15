# Online research and existing tools

Every fact below was checked against the source linked next to it on 2026-10-15.

## Facts the fixes rely on

| Topic | Fact | Source |
| --- | --- | --- |
| PyPI mint-token errors | `POST /_/oidc/mint-token` answers every refusal with HTTP 422 and `{"message": "Token request failed", "errors": [{"code": ..., "description": ...}]}` (`_invalid()`). `invalid-publisher` ("valid token, but no corresponding publisher") is returned when no trusted or pending publisher matches the OIDC claims. | [warehouse/oidc/views.py@9d5a5ec, L75 and L322](https://github.com/pypi/warehouse/blob/9d5a5ecd97ed2e632ec68f1452f16feabf703ccb/warehouse/oidc/views.py#L75) |
| PyPI pending publishers | A trusted publisher can be registered before the project exists ("pending publisher"); the first upload creates the project. | <https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/> |
| GitHub `concurrency.queue` | `queue: single` (default) keeps at most one pending run per group and cancels it when the next one arrives. `queue: max` keeps up to 100 pending runs and starts them in order. `max` cannot be combined with `cancel-in-progress: true`. | <https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax> (`concurrency`) |
| actionlint and `queue` | actionlint 1.7.12 has no `queue` key in its built-in schema and reports `unexpected key "queue" for "concurrency" section`. Tracked upstream; until then a `paths:` → `ignore:` regex in `.github/actionlint.yaml` suppresses exactly that message. | [rhysd/actionlint#654](https://github.com/rhysd/actionlint/issues/654), [actionlint config docs](https://github.com/rhysd/actionlint/blob/main/docs/config.md) |
| zizmor operating modes | "If neither ZIZMOR_OFFLINE nor GH_TOKEN/GITHUB_TOKEN/ZIZMOR_GITHUB_TOKEN are set, then zizmor runs in offline mode" and skips the audits that need the network. The audits table marks `impostor-commit`, `ref-confusion`, `known-vulnerable-actions` and `stale-action-refs` as not working offline. The "running in offline mode by default" warning itself is from the CI log ([excerpt](ci-logs/main-workflows-38725259341.txt)). | <https://docs.zizmor.sh/usage/#operating-modes>, <https://docs.zizmor.sh/audits/> |
| zizmor-action token | The action's `token` input defaults to `${{ github.token }}`, so the action pass was already online. | <https://github.com/zizmorcore/zizmor-action#inputs> |
| `upload-artifact` | `if-no-files-found` accepts `warn` (default), `error` and `ignore`. | <https://github.com/actions/upload-artifact#inputs> |
| mypy `strict_concatenate` | Deprecated; `extra_checks` replaces it and enables the same check plus others. | <https://mypy.readthedocs.io/en/stable/config_file.html#confval-extra_checks> |
| lychee inputs | A glob that matches nothing produces `[WARN] ...: No files found for this input source`. A directory input is walked recursively and filtered with `--extensions`; `--hidden` includes dot-folders such as `.github/`. | <https://lychee.cli.rs/guides/cli/> |
| Action majors | Latest releases on 2026-10-15: `actions/checkout` v7.0.1, `actions/setup-python` v7.0.0, `actions/cache` v6.1.0. | Release pages of each repository |
| zizmor releases | zizmor 1.31.0 and zizmor-action v0.7.0 were both released on 2026-10-15. They were not adopted in this PR (see [analysis.md](analysis.md#considered-and-left-unchanged)). | <https://github.com/zizmorcore/zizmor/releases>, <https://github.com/zizmorcore/zizmor-action/releases> |
| Dependency review | `actions/dependency-review-action` fails when the dependency graph is disabled; enabling it is a repository setting. | <https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/configuring-dependency-review> |

## Hive-mind best practices applied

From <https://github.com/link-assistant/hive-mind/blob/main/docs/CI-CD-BEST-PRACTICES.md>:

| Practice | Where it applies |
| --- | --- |
| 1. Relevant file changes | F6 (`.sh` and `.json` are code), F7 (model-tests paths) |
| 10. Concurrency control | F9 (`queue: max` for release writers, `cancel-in-progress: false`) |
| 14. Lint the workflows themselves | F5 (zizmor online), F9 (actionlint ignores only the one known gap), F10 |
| 16. Prove you can publish before you build ("report unknown, never a guess") | F1 (a 422 with a publisher code is a verdict; everything else stays unknown) |
| 17. Tell a broken pipeline from a changed world | F1, F2 (fail with the setting to change), F8 (a missing artifact fails loudly), lychee case-study exclusion kept |

## Existing tools and libraries

| Tool | Used here | Role |
| --- | --- | --- |
| [actionlint](https://github.com/rhysd/actionlint) (with shellcheck, pyflakes) | Yes | Workflow schema and `run:` scripts |
| [zizmor](https://github.com/zizmorcore/zizmor) | Yes | Workflow security audit |
| [lychee](https://github.com/lycheeverse/lychee) | Yes | Link checking |
| [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) | Yes | Trusted publishing. Its own error output decodes the same 422 codes, but only at publish time, after the build; the preflight gives the answer before the build. |
| [actions/dependency-review-action](https://github.com/actions/dependency-review-action) | Yes | Dependency review, behind the graph check |
| [check-jsonschema](https://github.com/python-jsonschema/check-jsonschema) | No | Validates workflows against the SchemaStore `github-workflow` schema, which already documents `concurrency.queue` (checked 2026-10-15). It is a fallback if rhysd/actionlint#654 stalls, but it checks only the schema, not `run:` scripts, so it would sit next to actionlint rather than replace it. |
| [dorny/paths-filter](https://github.com/dorny/paths-filter) | No | Job-level path filters. `scripts/detect_code_changes.py` already does this with tests; replacing it would add an unpinned third-party dependency for no gain. |
| [step-security/harden-runner](https://github.com/step-security/harden-runner) | No | Egress monitoring; outside the scope of false results in CI. |
| [mypy](https://mypy-lang.org/), [ruff](https://docs.astral.sh/ruff/), [pytest](https://pytest.org/) | Yes | Lint, types and tests |
