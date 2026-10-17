# Issue #19 snapshot

- URL: <https://github.com/konard/medical-neural-network-ml/issues/19>
- Title: Update all dependencies in all languages to their latest versions
- Opened: 2026-10-17T08:23:34Z by @konard
- Comments when captured (2026-10-17): 0

---

### Dependency ecosystems detected in this repository

Update **every** ecosystem below, in this order (most of the repository first). One pull request may cover them all, but each ecosystem must end up fully current — not partially.

1. **Python**
   - Detected languages: Python
   - Manifests: `docs/requirements.txt`, `pyproject.toml`
   - Update everything: `uv lock --upgrade  •  pip-compile --upgrade  •  poetry update`
   - Raise the floors in pyproject.toml as well: a `>=` floor years below the resolved version means CI and a fresh install are not testing the same tree. Drop upper bounds that exclude the current release.

2. **GitHub Actions**
   - Manifests: `.github/workflows/compatibility.yml`, `.github/workflows/docs.yml`, `.github/workflows/links.yml`, `.github/workflows/model-tests.yml`, `.github/workflows/release.yml`, `.github/workflows/security.yml`, `.github/workflows/workflows.yml`
   - Update everything: bump every `uses:` reference to the newest release (tag or pinned digest)
   - Actions pinned to a stale major (`actions/checkout@v3`) are the most common source of deprecation warnings in an otherwise green pipeline. Update reusable workflows and composite `action.yml` files too.

3. **Infrastructure as code**
   - Manifests: `.pre-commit-config.yaml`
   - Update everything: `terraform init -upgrade  •  helm dependency update  •  pre-commit autoupdate  •  git submodule update --remote`

Detected languages with no dependency manifest of their own: Shell, Jinja. Check whether they pull dependencies through one of the ecosystems above.

### Keeping them current

`.github/dependabot.yml` already exists — verify it covers every ecosystem and directory listed above, then keep it in sync with any manifest this work adds or moves.

The ecosystems found in this repository correspond to these entries:

```yaml
version: 2
updates:
  - package-ecosystem: 'pip'
    directories:
      - '/'
      - '/docs'
    schedule:
      interval: 'weekly'
  - package-ecosystem: 'github-actions'
    directory: '/'
    schedule:
      interval: 'weekly'
  - package-ecosystem: 'pre-commit'
    directory: '/'
    schedule:
      interval: 'weekly'
```

[Renovate](https://docs.renovatebot.com/) is the alternative if grouped or automerged updates are wanted instead.

Update all dependencies to their latest versions, in Python, GitHub Actions, Infrastructure as code. "All" is literal: development dependencies, transitive lockfile entries, build plugins, test runners, linters, base images, GitHub Actions, and the language/toolchain versions themselves.

For each ecosystem, produce a table of every dependency with the version pinned today and the version released today, resolved from the registry — not from memory. Anything left behind must have a written reason recorded (an upstream bug, a dropped platform, a paid tier), not silence.

Cross major versions deliberately. Read the changelog and the migration guide for every major bump, adapt the code to the new API, and remove the shims the old version needed. A constraint loosened or a test skipped to make a major "pass" is not an update.

Use the new features. Where a newer version replaces something this repository implements by hand, delete the hand-rolled copy and use the upstream feature, so there is less duplicated code and logic afterwards than before.

Make the constraints honest: raise floors that are years below what is actually installed, drop upper bounds that exclude the current release, and regenerate and commit every lockfile so a fresh install and CI resolve the same tree.

If the same dependency is pinned in more than one place — several language implementations, a Dockerfile, a workflow — bring every pin to the same version. A repository that pins one library at four different versions has four different behaviours.

Run the full build, test and lint suite of every ecosystem after updating, and make CI green. Resolve every new deprecation warning the update introduces rather than leaving it for the next reader.

Check the security advisories for the tree as it stands afterwards (`npm audit`, `cargo audit`, `pip-audit`, `bundle audit`, `dotnet list package --vulnerable`, or the ecosystem equivalent) and make sure the update leaves none unresolved.

Report to each dependency’s upstream all general logic or duplicated code that belongs in that dependency, missing features, and bugs that lead to workarounds in this repository. Search existing upstream issues first, then create or update a report with a minimal reproducible example, affected versions, the local workaround, and a suggested fix in code or feature proposal; link every report from the pull request. Keep necessary workarounds in this repository so upstream fixes do not block the pull request.

Follow the dependency-update practices collected in [https://github.com/link-assistant/hive-mind/blob/main/docs/DEPENDENCY-UPDATE-BEST-PRACTICES.md](https://github.com/link-assistant/hive-mind/blob/main/docs/DEPENDENCY-UPDATE-BEST-PRACTICES.md).

Please plan and execute everything in this single pull request, you have unlimited time and context, as context auto-compacts and you can continue indefinitely, until it is each and every requirement fully addressed, and everything is totally done.

---

<details>
<summary>Context collected by <code>/fix --update-all-dependencies</code></summary>

- **Repository:** [konard/medical-neural-network-ml](https://github.com/konard/medical-neural-network-ml)
- **Default branch:** `main`
- **Latest commit:** `cc8e90b` ([commit](https://github.com/konard/medical-neural-network-ml/commit/cc8e90b446a174cf5fff743313e441cda4a432ef)) — Merge pull request #12 from konard/issue-11-a17e793893f2
- **Ecosystems detected:** 3
- **Manifest files found:** 10

**Detected languages**

- **Python** — 93.1%
- **Shell** — 6.9%
- **Jinja** — 0.1%

</details>
