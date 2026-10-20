#!/usr/bin/env bash
# Lint and validate the cloud GPU infrastructure code (issue #23).
#
#   bash scripts/check-infra.sh [check ...]
#
# Checks (all by default, in this order):
#   terraform   terraform fmt -check, then init -backend=false and validate
#               in every module of infra/terraform (providers come from the
#               lock files; no cloud account is needed)
#   tflint      tflint with infra/terraform/.tflint.hcl
#   ansible     ansible-lint on infra/ansible (production profile)
#   kubernetes  kubeconform -strict on the rendered Job (with and without a
#               GPU) and the NVIDIA device plugin
#   helm        helm template of the ARC runner scale set chart with
#               infra/kubernetes/arc-gpu-runner-values.yaml
#   shell       shellcheck on the GPU host scripts and examples
#
# The tools must be on PATH; versions used by the Infrastructure workflow are
# pinned in .github/workflows/cloud-gpu-tests.yml. TF=tofu uses OpenTofu.
# Runs the same on a laptop, in GitHub Actions and in GitLab CI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TF="${TF:-terraform}"
ARC_CHART="oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set"
ARC_VERSION="0.15.0"
CHECKS=("$@")
if ((${#CHECKS[@]} == 0)); then
  CHECKS=(terraform tflint ansible kubernetes helm shell)
fi

log() {
  printf '[check-infra] %s\n' "$*" >&2
}

check_terraform() {
  "$TF" fmt -check -recursive -diff infra/terraform
  local module
  for module in infra/terraform/*/vm infra/terraform/*/k8s; do
    log "$TF validate $module"
    "$TF" -chdir="$module" init -backend=false -input=false -lockfile=readonly >/dev/null
    "$TF" -chdir="$module" validate -no-color
  done
}

check_tflint() {
  tflint --chdir=infra/terraform --init --config="$PWD/infra/terraform/.tflint.hcl" >/dev/null
  tflint --chdir=infra/terraform --recursive --config="$PWD/infra/terraform/.tflint.hcl"
}

check_ansible() {
  (cd infra/ansible && ansible-lint --profile production --strict .)
}

check_kubernetes() {
  local dir
  dir="$(mktemp -d)"
  local gpus
  for gpus in 0 1; do
    python3 -m scripts.cloud_gpu job-manifest --provider kubernetes --mode k8s \
      --model medgemma-4b-it --gpu-count "$gpus" --run-id check-infra \
      --commit 0000000 --env UNIMEDVL_ENGINE=echo --output "$dir/job-$gpus.yaml"
  done
  kubeconform -strict -summary "$dir"/job-*.yaml infra/kubernetes/nvidia-device-plugin.yaml
  rm -rf "$dir"
}

check_helm() {
  helm template medbench-gpu "$ARC_CHART" --version "$ARC_VERSION" \
    --namespace arc-runners --values infra/kubernetes/arc-gpu-runner-values.yaml \
    >/dev/null
  log "helm template of gha-runner-scale-set $ARC_VERSION passed"
}

check_shell() {
  shellcheck scripts/gpu-host-*.sh scripts/gpu-model-env.sh scripts/gpu-runner-start.sh \
    scripts/check-infra.sh scripts/cloud-gpu-*.sh ci/gitlab/*.sh examples/*.sh
}

for check in "${CHECKS[@]}"; do
  case "$check" in
    terraform | tflint | ansible | kubernetes | helm | shell)
      log "== $check"
      "check_$check"
      ;;
    *)
      log "unknown check '$check'"
      exit 2
      ;;
  esac
done
log "all checks passed: ${CHECKS[*]}"
