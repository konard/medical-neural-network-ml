# Issue #21 snapshot

- URL: <https://github.com/konard/medical-neural-network-ml/issues/21>
- Title: Update all dependencies in all languages to their latest versions
- Opened: 2026-10-19T15:33:20Z by @konard
- Comments when captured (2026-10-19): 0

---

### Dependabot pull requests merged before this issue was created

These Dependabot pull requests could not be merged automatically and are still open:

- [#13](https://github.com/konard/medical-neural-network-ml/pull/13) deps: bump the python-minor-and-patch group across 1 directory with 12 updates → `main` — not passing: [Compatibility / Python 3.9 on ubuntu-24.04](https://github.com/konard/medical-neural-network-ml/actions/runs/38956637946/job/117161041309) (failure), [Compatibility / Python 3.12 on macos-15](https://github.com/konard/medical-neural-network-ml/actions/runs/38956637946/job/117161041008) (failure), [Compatibility / Pipeline Status (Compatibility)](https://github.com/konard/medical-neural-network-ml/actions/runs/38956637946/job/117161833391) (failure) — skipped: merge conflict with the base branch
- [#14](https://github.com/konard/medical-neural-network-ml/pull/14) deps: update mypy requirement from <2.0,>=1.13.0 to >=1.13.0,<3.0 → `main` — not passing: [Compatibility / Python 3.12 on macos-15](https://github.com/konard/medical-neural-network-ml/actions/runs/38956645024/job/117161064859) (failure), [Compatibility / Pipeline Status (Compatibility)](https://github.com/konard/medical-neural-network-ml/actions/runs/38956645024/job/117162009605) (failure), [CI/CD Pipeline / Test (Python 3.13)](https://github.com/konard/medical-neural-network-ml/actions/runs/38956645281/job/117161350182) (failure), [CI/CD Pipeline / Pipeline Status (CI/CD)](https://github.com/konard/medical-neural-network-ml/actions/runs/38956645281/job/117162377856) (failure) — skipped: merge conflict with the base branch
- [#15](https://github.com/konard/medical-neural-network-ml/pull/15) deps: bump packaging from 25.0 to 26.3 → `main` — all 31 check(s) passing — skipped: merge conflict with the base branch
- [#16](https://github.com/konard/medical-neural-network-ml/pull/16) deps: bump transformers from 4.49.0 to 5.19.0 → `main` — not passing: [Compatibility / Python 3.9 on ubuntu-24.04](https://github.com/konard/medical-neural-network-ml/actions/runs/38956661623/job/117161120611) (failure), [Compatibility / Python 3.12 on macos-15](https://github.com/konard/medical-neural-network-ml/actions/runs/38956661623/job/117161120255) (failure), [Compatibility / Pipeline Status (Compatibility)](https://github.com/konard/medical-neural-network-ml/actions/runs/38956661623/job/117163512822) (failure) — skipped: merge conflict with the base branch
- [#17](https://github.com/konard/medical-neural-network-ml/pull/17) deps: bump numpy from 1.24.4 to 2.2.6 → `main` — all 31 check(s) passing — skipped: merge conflict with the base branch

Apply each of these updates in this issue’s pull request (Dependabot closes a pull request by itself once its update has landed), and fix the errors that made it unmergeable — a failing check, a conflict, or a repository setting such as a disabled auto-merge.

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
- **Latest commit:** `fff9d99` ([commit](https://github.com/konard/medical-neural-network-ml/commit/fff9d99fa7eff21702c8b00de4015ea46829f0fa)) — Merge pull request #20 from konard/issue-19-5e219937bdea
- **Ecosystems detected:** 3
- **Manifest files found:** 10

**Detected languages**

- **Python** — 93.3%
- **Shell** — 6.6%
- **Jinja** — 0.1%

</details>
