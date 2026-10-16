| Workflow | Job or monitored step | Wall s | Result | Host used GB (start → peak of total) | Largest child RSS GB | Disk used GB | Report |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| CI/CD Pipeline | Release Preflight (credentials) | 7 | success | | | | |
| CI/CD Pipeline | Detect Changes | 5 | success | | | | |
| CI/CD Pipeline | Validate Documentation | 5 | success | | | | |
| CI/CD Pipeline | Test (Python 3.13) | 72 | success | | | | |
| CI/CD Pipeline | Changelog Fragment Check | 7 | success | | | | |
| CI/CD Pipeline | Lint and Format Check | 29 | success | | | | |
| CI/CD Pipeline | Docker Image Build Check | 6 | success | | | | |
| CI/CD Pipeline | Build Package | 20 | success | | | | |
| CI/CD Pipeline | Pipeline Status (CI/CD) | 6 | success | | | | |
| CI/CD Pipeline | ↳ Verify distributions | 4.8 | exit 0 | 1.09 → 1.15 of 16.8 | 0.065 | 0.02 | 38950419706/artifacts/resources-build/build.json |
| CI/CD Pipeline | ↳ Test (Python 3.13) | 54.3 | exit 0 | 1.10 → 1.12 of 16.8 | 0.085 | 0.03 | 38950419706/artifacts/resources-test/test.json |
| Docs | Build docs | 28 | success | | | | |
| Docs | Pipeline Status (Docs) | 6 | success | | | | |
| Docs | ↳ Build docs | 7.1 | exit 0 | 1.12 → 1.20 of 16.8 | 0.130 | 0.01 | 38950419686/artifacts/resources-docs/docs.json |
| Security | Branch Rulesets | 8 | success | | | | |
| Security | CodeQL (actions) | 46 | success | | | | |
| Security | Dependency Review | 16 | success | | | | |
| Security | Audit dependencies (ubuntu-24.04) | 33 | success | | | | |
| Security | Audit dependencies (macos-15) | 72 | success | | | | |
| Security | CodeQL (python) | 45 | success | | | | |
| Security | Pipeline Status (Security) | 6 | success | | | | |
| Security | ↳ Dependency audit (macos-15) | 61.1 | exit 0 | 4.32 → 4.35 of 7.5 | 0.279 | 1.84 | 38950419680/artifacts/resources-dependency-audit-macos-15/dependency-audit-macos-15.json |
| Security | ↳ Dependency audit (ubuntu-24.04) | 26.4 | exit 0 | 1.33 → 1.39 of 16.8 | 0.110 | 0.36 | 38950419680/artifacts/resources-dependency-audit-ubuntu-24.04/dependency-audit-ubuntu-24.04.json |
| Compatibility | Python 3.12 on macos-15 | 119 | success | | | | |
| Compatibility | Python 3.9 on ubuntu-24.04 | 38 | success | | | | |
| Compatibility | Pipeline Status (Compatibility) | 6 | success | | | | |
| Compatibility | ↳ Compatibility (macos-15) | 97.5 | exit 0 | 4.13 → 4.29 of 7.5 | 0.077 | 0.04 | 38950419689/artifacts/resources-compatibility-macos-15/compatibility-macos-15.json |
| Compatibility | ↳ Compatibility (ubuntu-24.04) | 9.3 | exit 0 | 1.22 → 1.22 of 16.8 | 0.057 | 0.00 | 38950419689/artifacts/resources-compatibility-ubuntu-24.04/compatibility-ubuntu-24.04.json |
| Workflows | Audit Workflows | 15 | success | | | | |
| Workflows | Lint Workflows | 8 | success | | | | |
| Workflows | Pipeline Status (Workflows) | 6 | success | | | | |
| Broken Link Checker | Check Links | 49 | success | | | | |
| Broken Link Checker | Pipeline Status (Links) | 6 | success | | | | |
| Model Tests | Real sample files | 20 | success | | | | |
| Model Tests | OCR baseline (Tesseract) | 39 | success | | | | |
| Model Tests | CPU smoke test (MedGemma 4B Q4_K_M) | 440 | success | | | | |
| Model Tests | Pipeline Status (Model Tests) | 6 | success | | | | |
| Model Tests | ↳ Download model | 2.4 | exit 0 | 1.93 → 1.93 of 16.8 | 0.028 | 0.00 | 38950419674/artifacts/medbench-cpu-reports/resources-download.json |
| Model Tests | ↳ Run medbench | 375.2 | exit 0 | 1.85 → 5.74 of 16.8 | 4.714 | 0.00 | 38950419674/artifacts/medbench-cpu-reports/resources-run.json |
| Model Tests | ↳ Tesseract ocr-history | 12.2 | exit 0 | 1.24 → 1.24 of 16.8 | 0.079 | 0.00 | 38950419674/artifacts/medbench-ocr-baseline-reports/resources-tesseract-ocr-history.json |
| Model Tests | ↳ Tesseract ocr-prescription | 0.5 | exit 0 | 1.20 → 1.24 of 16.8 | 0.036 | 0.00 | 38950419674/artifacts/medbench-ocr-baseline-reports/resources-tesseract-ocr-prescription.json |
| Model Tests | ↳ Download real samples | 0.2 | exit 0 | 1.23 → 1.23 of 16.8 | 0.046 | 0.00 | 38950419674/artifacts/resources-real-data/real-data-fetch.json |
| Model Tests | ↳ Verify real samples | 0.7 | exit 0 | 1.21 → 1.23 of 16.8 | 0.066 | 0.00 | 38950419674/artifacts/resources-real-data/real-data-verify.json |
