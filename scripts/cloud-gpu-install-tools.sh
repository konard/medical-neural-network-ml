#!/usr/bin/env bash
# Install the pinned tools of the cloud GPU orchestrator (issue #23) on a
# Debian or Ubuntu image, for CI systems whose images do not have them
# (ci/gitlab/cloud-gpu.gitlab-ci.yml runs it on python:3.14-slim-trixie;
# GitHub's ubuntu-26.04 image already has the cloud CLIs).
#
#   bash scripts/cloud-gpu-install-tools.sh [set ...]
#
# Sets (several may be given; `base` is always installed):
#   base     git, ssh, curl, unzip (apt) and kubectl
#   infra    terraform, tflint, kubeconform, helm, shellcheck, ansible-lint,
#            yamllint: what scripts/check-infra.sh runs
#   aws      terraform and AWS CLI v2 (aws eks get-token, the reaper)
#   azure    terraform and Azure CLI (the reaper)
#   gcp      terraform, gcloud and gke-gcloud-auth-plugin
#   hetzner  nothing more: the Robot API is plain HTTPS from Python
#
# Every download is checked against the SHA-256 below; Python tools are
# pinned by version. The same pins as .github/workflows/cloud-gpu-tests.yml.
# PREFIX (default /usr/local) is where binaries go; needs root for apt.
set -euo pipefail

PREFIX="${PREFIX:-/usr/local}"
export DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore
TERRAFORM_VERSION=1.16.6
TERRAFORM_SHA256=38034b38d03096399a7a1264a14561a8029b2b496fd1f169452656e53968f1b7
TFLINT_VERSION=v0.64.0
TFLINT_SHA256=cca9d13e2e1d7a2c627af60ff899a3c9b74212899416aeb96ec764d2ef954537
KUBECTL_VERSION=v1.37.1
KUBECTL_SHA256=65691ff77eb6fa44c908b77a1082c9f092c3b9733b5cefabec0d1104890e21a8
KUBECONFORM_VERSION=v0.8.0
KUBECONFORM_SHA256=9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883
HELM_VERSION=v4.3.1
HELM_SHA256=0403a88ae20913d11b10a2467b50087721fea7581736da6b9d3992f5357fe54b
AWSCLI_VERSION=2.37.13
AWSCLI_SHA256=c92fde59827622889d3d38a44e2089a137894ce151a2c658a3988ecde4860caa
GCLOUD_VERSION=588.0.0
GCLOUD_SHA256=e38ceac43022bb5a4d94d5a4a9c9c51f90f28c910b59a018ae07011e220c4412
AZURE_CLI_VERSION=2.91.0
PYTHON_TOOLS=(ansible-core==2.21.5 ansible-lint==26.10.1 yamllint==1.38.0 shellcheck-py==0.11.0.1)

log() {
  printf '[cloud-gpu-install-tools] %s\n' "$*" >&2
}

# fetch URL SHA256 FILE: download and verify, or stop.
fetch() {
  curl -fsSL --retry 3 -o "$3" "$1"
  echo "$2  $3" | sha256sum -c --quiet -
}

install_base() {
  if command -v apt-get >/dev/null && ! command -v ssh >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
      ca-certificates curl git openssh-client unzip xz-utils >/dev/null
  fi
  if ! command -v kubectl >/dev/null; then
    fetch "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl" \
      "$KUBECTL_SHA256" kubectl
    install -m 0755 kubectl "$PREFIX/bin/kubectl"
  fi
}

install_terraform() {
  command -v terraform >/dev/null && return
  fetch "https://releases.hashicorp.com/terraform/$TERRAFORM_VERSION/terraform_${TERRAFORM_VERSION}_linux_amd64.zip" \
    "$TERRAFORM_SHA256" terraform.zip
  unzip -oq terraform.zip terraform
  install -m 0755 terraform "$PREFIX/bin/terraform"
}

install_infra() {
  install_terraform
  fetch "https://github.com/terraform-linters/tflint/releases/download/$TFLINT_VERSION/tflint_linux_amd64.zip" \
    "$TFLINT_SHA256" tflint.zip
  unzip -oq tflint.zip tflint
  install -m 0755 tflint "$PREFIX/bin/tflint"
  fetch "https://github.com/yannh/kubeconform/releases/download/$KUBECONFORM_VERSION/kubeconform-linux-amd64.tar.gz" \
    "$KUBECONFORM_SHA256" kubeconform.tar.gz
  tar -xzf kubeconform.tar.gz kubeconform
  install -m 0755 kubeconform "$PREFIX/bin/kubeconform"
  fetch "https://get.helm.sh/helm-$HELM_VERSION-linux-amd64.tar.gz" "$HELM_SHA256" helm.tar.gz
  tar -xzf helm.tar.gz --strip-components=1 linux-amd64/helm
  install -m 0755 helm "$PREFIX/bin/helm"
  python3 -m pip install -q --no-cache-dir "${PYTHON_TOOLS[@]}"
}

install_aws() {
  install_terraform
  command -v aws >/dev/null && return
  fetch "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-$AWSCLI_VERSION.zip" \
    "$AWSCLI_SHA256" awscli.zip
  unzip -q awscli.zip
  ./aws/install --bin-dir "$PREFIX/bin" --install-dir "$PREFIX/aws-cli" >/dev/null
}

install_azure() {
  install_terraform
  command -v az >/dev/null && return
  python3 -m venv "$PREFIX/azure-cli"
  "$PREFIX/azure-cli/bin/pip" install -q --no-cache-dir "azure-cli==$AZURE_CLI_VERSION"
  ln -sf "$PREFIX/azure-cli/bin/az" "$PREFIX/bin/az"
}

install_gcp() {
  install_terraform
  if ! command -v gcloud >/dev/null; then
    fetch "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-$GCLOUD_VERSION-linux-x86_64.tar.gz" \
      "$GCLOUD_SHA256" gcloud.tar.gz
    tar -xzf gcloud.tar.gz -C "$PREFIX"
    ln -sf "$PREFIX/google-cloud-sdk/bin/gcloud" "$PREFIX/bin/gcloud"
  fi
  if ! command -v gke-gcloud-auth-plugin >/dev/null; then
    gcloud components install gke-gcloud-auth-plugin --quiet >/dev/null
    ln -sf "$(dirname "$(readlink -f "$(command -v gcloud)")")/gke-gcloud-auth-plugin" \
      "$PREFIX/bin/gke-gcloud-auth-plugin"
  fi
}

install_hetzner() {
  :
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"
install_base
for set in "$@"; do
  case "$set" in
    base) ;;
    infra | aws | azure | gcp | hetzner)
      log "installing $set"
      "install_$set"
      ;;
    *)
      log "unknown set '$set' (base, infra, aws, azure, gcp, hetzner)"
      exit 2
      ;;
  esac
done
log "installed: base $*"
