# GitHub Actions: pinned vs latest release (gh api, 2026-10-17)
# columns: action | pinned on main | pinned on this branch | latest release | SHA check
# github/codeql-action: the newest *release* is codeql-bundle-v2.27.2 (a CLI bundle tag); the newest action tag is v4.39.0, which dereferences to the pinned commit.
| `actions/cache` | v6.1.0 | v6.1.0 | v6.1.0 | sha ok |
| `actions/checkout` | v7.0.1 | v7.0.1 | v7.0.1 | sha ok |
| `actions/configure-pages` | v6.0.0 | v6.0.0 | v6.0.0 | sha ok |
| `actions/dependency-review-action` | v5.0.0 | v5.0.0 | v5.0.0 | sha ok |
| `actions/deploy-pages` | v5.0.1 | v5.0.1 | v5.0.1 | sha ok |
| `actions/download-artifact` | v8.0.2 | v8.0.2 | v8.0.2 | sha ok |
| `actions/setup-python` | v7.0.0 | v7.0.0 | v7.0.0 | sha ok |
| `actions/upload-artifact` | v7.0.2 | v7.0.2 | v7.0.2 | sha ok |
| `actions/upload-pages-artifact` | v5.0.0 | v5.0.0 | v5.0.0 | sha ok |
| `codecov/codecov-action` | v7.0.0 | v7.1.1 | v7.1.1 | sha ok |
| `docker/build-push-action` | v7.4.0 | v7.4.0 | v7.4.0 | sha ok |
| `docker/login-action` | v4.6.0 | v4.6.0 | v4.6.0 | sha ok |
| `docker/setup-buildx-action` | v4.4.1 | v4.4.1 | v4.4.1 | sha ok |
| `github/codeql-action` | v4.39.0 | v4.39.0 | codeql-bundle-v2.27.2 | sha differs |
| `lycheeverse/lychee-action` | v2.9.0 | v2.9.0 | v2.9.0 | sha ok |
| `pypa/gh-action-pypi-publish` | v1.14.2 | v1.14.2 | v1.14.2 | sha ok |
| `zizmorcore/zizmor-action` | v0.6.4 | v0.7.1 | v0.7.1 | sha ok |
