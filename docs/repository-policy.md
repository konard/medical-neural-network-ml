# Repository policy

How the default branch is protected, which checks gate a merge, who may skip
which rule, and what fork pull requests, Dependabot and the self-hosted GPU
runner are allowed to do (issue #10). The rulesets are files in
`.github/rulesets/`; `tests/test_repository_policy.py` keeps this page, the
files and the workflows consistent.

**Status (2026-10-16): written, not applied.** The live repository has no
ruleset and no branch protection. Rules that block pushes change how the
release writer works, so a repository admin applies them deliberately (see
"Applying the Rulesets" below); until then the
`Branch Rulesets` job fails on main with the command to run.

## Required Checks

`.github/rulesets/main-checks.json` requires three terminal status gates,
each reported by GitHub Actions (integration 15368, so no other app or
commit status can satisfy them):

| Check | Workflow | Covers |
|---|---|---|
| `Pipeline Status (CI/CD)` | `release.yml` | lint, format, secret scan, docs validation, Python 3.15 tests, build, changelog fragment, Docker build, release preflight (report mode) |
| `Pipeline Status (Compatibility)` | `compatibility.yml` | full suite on Python 3.11–3.14 (Linux) and Python 3.15 (macOS arm64) |
| `Pipeline Status (Security)` | `security.yml` | `pip-audit` of every dependency surface, CodeQL (Python and Actions), Dependency Review, Dependency graph, branch rulesets |

A gate observes every other job of its workflow with `if: always()`, so a
job that failed, timed out or was cancelled fails the gate. Each
gate has its own name: before issue #10 all seven were called
`Pipeline Status`, and a required context with that name could be satisfied
by whichever workflow happened to finish.

Only workflows that run on **every** pull request can be required: a
workflow skipped by a `paths` filter never reports, and GitHub waits for a
missing required check forever. `compatibility.yml` therefore no longer
filters pull requests. The path-filtered gates (`Docs`, `Links`,
`Model Tests`, `Workflows`) still run when their files change and still turn
main red; they are not required.

`strict_required_status_checks_policy` is off: `pull_request` runs already
test the merge with the current base, and requiring an up-to-date branch
would force a rebase of every open pull request (Dependabot's included)
after each merge. The gates on main catch what a stale merge breaks.

## Pull Request Review

`.github/rulesets/main-review.json` requires a pull request with one
approval for every change to main. A new push dismisses the approval and
every review thread must be resolved.

Contributor pull requests need the maintainer's approval. The maintainer
cannot approve their own pull request, so the `admin` repository role
(`actor_id` 5) may bypass this ruleset **in pull-request mode only**: they
can merge their own pull request without a second reviewer, but cannot push
to main directly. The required checks and the history rules are separate
rulesets without this bypass, so an admin merge still waits for green gates.

`.github/rulesets/main-history.json` forbids force pushes and deleting main
for everyone, with no bypass at all; the release writer only fast-forwards.

## Release Bot Bypass

`manual-release` in `release.yml` pushes the version commit to main with the
job's `GITHUB_TOKEN` (`scripts/version_and_commit.py`), as
`github-actions[bot]`. A direct push has no pull request and no checks, so
the GitHub Actions app (`Integration`, `actor_id` 15368) bypasses
`main-checks` and `main-review` in `always` mode, which keeps an audit entry
for every bypass (`exempt` would not). `auto-release` publishes the version
that a merged pull request already committed and pushes nothing.

What keeps this bypass narrow:

- Fork pull requests get a read-only token; no workflow uses
  `pull_request_target` or `workflow_run`.
- Only `auto-release` and `manual-release` request `contents: write`, and
  they run only on a push to main or a manual dispatch, after lint, tests,
  build and the credential preflight succeeded.
- A collaborator could still write a workflow on their own branch that uses
  the bypass. With one maintainer that is the same trust they already have;
  before adding collaborators, replace the app bypass with a deploy key that
  only a protected `release` environment can read (actor type `DeployKey`),
  and change the `manual-release` checkout to use it.

The bypass actor id 15368 and role id 5 come from GitHub's REST schema and
other public rulesets; GitHub does not document a table of them. The first
manual release after applying the rulesets is the test (see "Test
Plan" below).

## Fork Pull Requests

A `pull_request` run from a fork has a read-only `GITHUB_TOKEN`, no OIDC
token and no secrets. Every secret a workflow reads, and why a fork run is
still safe:

| Secret | Jobs | On a fork pull request |
|---|---|---|
| `GITHUB_TOKEN` | `links.yml` link-checker; `auto-release`, `manual-release` | read-only token; the release jobs do not run |
| `CODECOV_TOKEN` | `test` | empty, so the upload step is skipped (`test_release_workflow_gates_codecov_upload_on_token`) |
| `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` | `release-preflight`, `docker-publish*` | preflight runs in `report` mode and only annotates; publishing jobs do not run |

The release preflight cannot exchange an OIDC token in a fork, and the
Dependency graph and branch ruleset checks are repository settings, so all
three report instead of failing a contributor's pull request; on main they
fail.

## Tokens and Credentials

- Every workflow defaults to `contents: read`; write scopes are granted per
  job. `tests/test_repository_policy.py` lists the ten write scopes that
  exist today, so a new one fails the tests until this section is reviewed.
- Every checkout sets `persist-credentials: false` except `manual-release`,
  which needs the token in `.git/config` for its `git push origin main`.
- `id-token: write` is held by the release jobs and the preflight (PyPI
  trusted publishing) and by the Pages deploy.

## Action Pins and Update Pull Requests

Every `uses:` names a full commit hash with the release in a comment
(`actions/checkout@3d3c42e… # v7.0.1`); zizmor's `hash-pin` policy enforces
it, and its `ref-version-mismatch` audit checks that the comment matches the
hash. `.github/dependabot.yml` opens weekly update pull requests after a
seven-day cooldown: one grouped pull request for the actions, and one for
the Python dependencies in `pyproject.toml` and `docs/requirements.txt`
together with the `.pre-commit-config.yaml` hooks (a multi-ecosystem group,
because the hooks pin the same ruff, mypy and pillow as the `dev` extra).
Those pull requests run the same required checks and need the same review:
read the release notes, check that the runner image and Python versions are
still supported, and let `pip-audit` and Dependency Review run.
`pre-commit run check-dependabot` validates the file against its schema.

`scripts/unimedvl-requirements.in` and its lock
`scripts/unimedvl-requirements.txt` get their own weekly pull request from
Dependabot's `uv` ecosystem (`directory: /scripts`), which reruns
`uv pip compile` with the lock's `--python-version`; the `pip` ecosystem
excludes them (`exclude-paths`) so they are not proposed twice. Nothing else
pins torch or transformers since issue #21: `scripts/unimedvl_server.py`
provides upstream's flash-attn and transformers 4.x imports from the
installed packages. That pull request runs the `unimedvl-cpu` job of
`model-tests.yml`, which runs upstream UniMedVL's `chat()` on the new lock
(`tests/test_unimedvl_upstream.py`, `tests/test_unimedvl_attention.py`).
To recompile by hand, run the command in the `.in` header from `scripts/`,
where Dependabot runs it. `pip-audit` checks the lock on every run
(`scripts/audit_dependencies.py`).

GitHub can also reject any workflow that uses an action by tag
(Settings → Actions → General → "Require actions to be pinned to a full-length
commit SHA", `sha_pinning_required`). It is off today; every action here is
already pinned, so a repository admin can turn it on at any time.

## Self-Hosted Runners

No self-hosted runner is attached today (0 registered on 2026-10-16). The
`gpu` job of `model-tests.yml` is the only job that may use one, and the
following holds before a GPU host is attached:

- **Trusted dispatch only.** The job runs only for a manual
  `workflow_dispatch` from main with `gpu: true`, never for a pull request,
  a schedule or a dispatch from another branch (whose workflow file anyone
  with push access could have changed). Only users with write access can
  dispatch.
- **Approval.** The job uses the `gpu` environment. Before attaching a
  runner, open Settings → Environments → `gpu`, restrict deployment branches
  to `main` and add the maintainer as a required reviewer.
- **Least privilege.** `contents: read`, no secrets, no persisted
  credentials; models and archives are pinned by revision and SHA-256.
- **Ephemeral and isolated.** Register the runner with
  `./config.sh --ephemeral` on a fresh VM or container per job (a cloud GPU
  that is destroyed afterwards), in a repository-level runner group used by
  this repository only, with the labels `self-hosted, linux, x64, gpu`. Do
  not attach a personal workstation.
- **Cleanup.** The run script stops the model server on exit or deadline
  (`tests/test_medbench_model_run.py`), and the last step removes the
  downloaded weights, the model virtual environment and the reports even
  when the job failed. Reports are uploaded as artifacts first.

## Trusted Publishing (#8) and Required Checks

PyPI trusted publishing for `medical-neural-network-ml` is not configured
yet (issue #8). The release preflight proves it with the real OIDC
exchange, so:

- On pull requests the preflight runs in `report` mode: it annotates the
  missing publisher and passes. `Pipeline Status (CI/CD)` stays green and
  contributors are not blocked by a setting they cannot change.
- On main and on a manual release it runs in `release` mode and fails with
  `release-preflight: refusing to release` and the PyPI settings to enter.
  `auto-release` and `manual-release` do not run, so no version commit is
  pushed and nothing is published.

That failure is intentional and must not be turned into a pass: no
`continue-on-error`, no skipped preflight, no bypass for it. It turns green
when a maintainer registers the publisher (owner `konard`, repository
`medical-neural-network-ml`, workflow `release.yml`, environment blank);
"Publishing to PyPI" (`docs/publishing.md`) has the steps and the first
release.

## Applying the Rulesets

A repository admin, after reading this page:

```bash
# Read-only comparison with the live repository.
python scripts/repository_rulesets.py check --repository konard/medical-neural-network-ml
# Print the API calls (dry run, nothing changes).
GH_TOKEN=$(gh auth token) python scripts/repository_rulesets.py apply --repository konard/medical-neural-network-ml
# Create or replace the three rulesets.
GH_TOKEN=$(gh auth token) python scripts/repository_rulesets.py apply --repository konard/medical-neural-network-ml --yes
```

A ruleset PUT replaces the whole ruleset, so change the JSON files, not the
settings page; the `Branch Rulesets` job of `security.yml` reports any
difference. To try the rules first, set `"enforcement": "evaluate"` in the
files: GitHub then records what would have been blocked without blocking it.

## Test Plan

After applying the rulesets:

1. **Contributor pull request.** From a fork, open a pull request that
   changes a file under `src/`. The three gates must report, the merge must
   wait for them and for one approval, and preflight, Dependency graph and
   ruleset checks must only annotate.
2. **Maintainer pull request.** The admin can merge their own green pull
   request without an approval; a red one cannot be merged.
3. **Direct push.** `git push origin main` from a maintainer clone is
   rejected, as is `git push --force`.
4. **Automatic version commit.** After #8, dispatch `CI/CD Pipeline` with a
   `patch` bump. `manual-release` must push the version commit through the
   GitHub Actions bypass (Settings → Rules → Insights shows the bypass). If
   the push is rejected, replace the bypass with a deploy key as described
   in "Release Bot Bypass"; do not remove the rules.
5. **Dependabot.** The first grouped update pull request runs the same
   gates and needs the same approval.
