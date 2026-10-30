#!/usr/bin/env bash
# Install the pinned tools of the cloud GPU orchestrator (issue #23) on a
# Debian or Ubuntu image, for CI systems whose images do not have them
# (ci/gitlab/cloud-gpu.gitlab-ci.yml uses python:3.14-slim-trixie because
# Ansible supports Python <=3.14;
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
TERRAFORM_VERSION=1.17.1
TERRAFORM_SHA256=4638628058e71a55212b1c563712eac03b902128f40740f3dc8174df12135061
TFLINT_VERSION=v0.65.0
TFLINT_SHA256=d4c545ed2e29d6338604b586f34f544f380f70f243aa72f7b484713835eaec61
KUBECTL_VERSION=v1.37.2
KUBECTL_SHA256=c04487523e7aa7d1493f7808174c899db93d7a9cf57531420cd76b9bd415f8f4
KUBECONFORM_VERSION=v0.8.0
KUBECONFORM_SHA256=9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883
HELM_VERSION=v4.3.1
HELM_SHA256=0403a88ae20913d11b10a2467b50087721fea7581736da6b9d3992f5357fe54b
AWSCLI_VERSION=2.39.2
AWSCLI_SHA256=bb3e675d5c9a915121f3f90b380ec041ba52517067d017e4a4ceba40e0213247
GCLOUD_VERSION=591.0.0
GCLOUD_SHA256=78c570f5fcfe58a9311bd98c06b75530595e3cae6277aad75a676a39e6ee7b63
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
  python3 -m pip install -q --no-cache-dir -r "$SCRIPT_DIR/infra-requirements.txt"
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
  "$PREFIX/azure-cli/bin/pip" install -q --no-cache-dir -r "$SCRIPT_DIR/azure-cli-requirements.txt"
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
