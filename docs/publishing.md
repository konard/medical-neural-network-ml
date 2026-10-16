# Publishing to PyPI

What a maintainer does once so that releases reach PyPI, and what CI checks
before and after every upload (issue #8). Only the account that will own
`medical-neural-network-ml` on PyPI can do the setup; nothing in this
repository, and no CI job, changes PyPI account settings.

**Status (2026-10-16): no publisher.** `https://pypi.org/pypi/medical-neural-network-ml/json`
answers 404, so the name is unclaimed. The `Release Preflight` job fails on
main with `release-preflight: refusing to release` because PyPI refuses the
OIDC token exchange, and no release job runs. That failure is intended and
stays until the publisher exists (see "Trusted Publishing (#8) and Required
Checks" in the repository policy).

The distribution was called `my-package` until the 2026-10-16 audit; that
name belongs to an unrelated PyPI project. Do not register a publisher for it
or upload to it.

## Register the Publisher

1. Sign in to PyPI with the account that should own the project, with
   two-factor authentication enabled.
2. Check the name again: `https://pypi.org/project/medical-neural-network-ml/`
   must still be "Not Found". If someone else registered it in the meantime,
   stop: choose a new name in `pyproject.toml` (`[project] name`) in a pull
   request, and register the publisher for that name instead.
3. Open <https://pypi.org/manage/account/publishing/> and add a pending
   GitHub publisher:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `medical-neural-network-ml` |
   | Owner | `konard` |
   | Repository name | `medical-neural-network-ml` |
   | Workflow name | `release.yml` |
   | Environment name | *(blank)* |

   The environment stays blank because no job in `release.yml` uses a GitHub
   environment. A blank environment lets any job of `release.yml` in this
   repository publish; only `auto-release` and `manual-release` request
   `id-token: write` besides the preflight, which mints a token and uploads
   nothing.

A pending publisher does not reserve the name: it turns into a normal
publisher, and creates the project, on the first upload. If somebody else
registers the name first, PyPI drops the pending publisher. Publish soon
after registering it.

Once the project exists, the publisher is managed at
`https://pypi.org/manage/project/medical-neural-network-ml/settings/publishing/`.

## First Release

1. Open Actions → CI/CD Pipeline, select the latest failed run on main and
   choose **Re-run failed jobs**. Do not dispatch a new run for this: a
   dispatch runs `manual-release`, which bumps the version first.
2. `Release Preflight` exchanges a real OIDC token with PyPI. It must print
   `PASS: PyPI minted a short-lived upload token via trusted publishing`. The
   minted token is not used and expires on its own.
3. Only then does `auto-release` run. It publishes the `dist` artifact that
   the `Build Package` job of the same run built and checked, for the version
   in `pyproject.toml` (`0.1.0`, tag `v0.1.0`).

If the preflight still fails, read its annotation. `invalid-publisher` or
`invalid-pending-publisher` means a field in the table above does not match
the workflow. Correct the publisher on PyPI and re-run. Do not skip the
preflight, add `continue-on-error`, or switch to an API token to get past it.

## What CI Checks

Before uploading, in `Build Package` and again in `manual-release` (which
builds its own files after the version commit),
`scripts/check_distribution.py dist --install` checks the exact files that
will be uploaded:

- Exactly one wheel and one sdist exist, for the name and version in
  `pyproject.toml`.
- Both carry metadata 2.4 or newer with `License-Expression: Unlicense`, a
  `License-File` for `LICENSE` and `THIRD_PARTY_NOTICES.md`, and neither a
  legacy `License` field nor a `License ::` classifier (PEP 639).
- The license files are byte-identical to the repository copies.
- The wheel holds exactly the files of `src/medbench`, listed in `RECORD`
  with matching hashes. The sdist holds only `src/`, `CHANGELOG.md`,
  `pyproject.toml`, `README.md`, `.gitignore`, `PKG-INFO` and the license
  files. Neither may contain images, DICOM, PDFs or archives
  (`THIRD_PARTY_NOTICES.md` promises that media is not redistributed).
- The `medbench` console script is declared. After an offline install into
  a fresh virtual environment, `medbench --help` and `medbench sources` run,
  and the installed metadata still says `Unlicense`. A wheel built from the
  sdist has the same files as the published wheel.

After uploading, `auto-release` and `manual-release` wait until PyPI serves
the exact version, install it from PyPI in a fresh environment and import it
(`scripts/smoke_test_published_package.py`), and only then create the GitHub
release.

## After the First Release

Check by hand, once. These pages exist only after the release, so they are
written as text rather than links:

- `https://pypi.org/project/medical-neural-network-ml/0.1.0/` shows the
  `Unlicense` license expression and both files. `pypa/gh-action-pypi-publish`
  uploads attestations by default with trusted publishing, so each file's
  details should name this repository and `release.yml`.
- `https://github.com/konard/medical-neural-network-ml/releases/tag/v0.1.0`
  exists and is not a draft.
- `pip install medical-neural-network-ml==0.1.0` in a clean virtual
  environment, then `medbench sources`.

Later releases are pushed to main with a version bump in the pull request
(`auto-release`), or dispatched with a `bump_type` (`manual-release`).

## Adding a GitHub Environment

To require an approval before every upload, create a `pypi` environment
(Settings → Environments) with the maintainer as required reviewer, add
`environment: pypi` to `auto-release` and `manual-release`, and set the same
environment name on the PyPI publisher in the same change. The environment
name is part of the OIDC token PyPI checks: while the two disagree, every
upload is refused. The preflight job would then also need the environment to
prove the exchange.
