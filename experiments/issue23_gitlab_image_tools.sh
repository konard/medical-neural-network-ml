#!/usr/bin/env bash
# Issue #23: run the GitLab template's cloud-gpu:check-infra and cloud-gpu:plan
# commands in its default image (python:3.14-slim-trixie) with Docker, as a
# GitLab runner would: install every pinned tool with
# scripts/cloud-gpu-install-tools.sh, print their versions, run
# scripts/check-infra.sh and a plan. No cloud account is used.
#
#   bash experiments/issue23_gitlab_image_tools.sh [image] > gitlab-image.log 2>&1
#
# Result on 2026-10-20:
# Terraform v1.16.6, TFLint 0.64.0, kubectl v1.37.1,
# aws-cli 2.37.13, azure-cli 2.91.0, Google Cloud SDK 588.0.0 with gke-gcloud-auth-plugin,
# Helm v4.3.1, ShellCheck 0.11.0, ansible-lint 26.10.1; "all checks passed:
# terraform tflint ansible kubernetes helm shell"; plan printed g6.xlarge.
set -euo pipefail
image="${1:-python:3.14-slim-trixie}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm -v "$repo:/src:ro" "$image" bash -euo pipefail -c '
  cp -r /src /work && cd /work
  bash scripts/cloud-gpu-install-tools.sh infra aws azure gcp
  terraform version; tflint --version; kubectl version --client; aws --version
  az version -o tsv; gcloud version | head -3
  gke-gcloud-auth-plugin --version | head -1; helm version --short
  shellcheck --version | head -2; ansible-lint --version | head -1
  bash scripts/check-infra.sh
  python3 -m scripts.cloud_gpu plan --provider aws --mode k8s \
    --model medgemma-4b-it --run-id gitlab-1 --commit abc
'
