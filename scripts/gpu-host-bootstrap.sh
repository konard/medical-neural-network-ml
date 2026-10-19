#!/usr/bin/env bash
# Prepare a fresh Ubuntu 24.04 GPU host for scripts/gpu-host-run.sh (issue #23).
#
# The cloud images of AWS, Azure and GCP already hold an NVIDIA driver, so
# there `driver` only checks it. A Hetzner dedicated GPU server (Robot GEX)
# starts from a plain Ubuntu, and `driver` installs the R580 server driver
# with the open kernel modules that Blackwell GPUs need. The orchestrator
# (python -m scripts.cloud_gpu) runs it as root over SSH; infra/ansible holds
# the same steps as playbooks for hosts we keep.
#
# Usage (as root):
#   bash scripts/gpu-host-bootstrap.sh driver    # exit 3: reboot, run again
#   bash scripts/gpu-host-bootstrap.sh toolkit   # NVIDIA Container Toolkit
#   bash scripts/gpu-host-bootstrap.sh k3s       # one-node k3s with GPU
#
# Every version and download below is pinned; tests/test_cloud_gpu_scripts.py
# checks that infra/ansible uses the same ones.
set -euo pipefail

MIN_DRIVER="${MIN_DRIVER:-580}"
DRIVER_BRANCH="580"
DRIVER_PACKAGES=("nvidia-driver-${DRIVER_BRANCH}-server-open" "nvidia-utils-${DRIVER_BRANCH}-server")
TOOLKIT_VERSION="1.20.1-1"
TOOLKIT_KEY_FINGERPRINT="C95B321B61E88C1809C4F759DDCAE044F796ECB0"
TOOLKIT_REPO="https://nvidia.github.io/libnvidia-container/stable/deb"
K3S_VERSION="v1.37.1+k3s1"
K3S_SHA256_AMD64="92b94582eb7b34a8cf532e6006842b9ed215a6783cf56a3dd6271fd35243b2c0"
K3S_INSTALL_SHA256="ed01f89fd977bf20ac1516bbebf8370bf3ddbaa55dac8aba610956a4c78cc00b"
REBOOT_NEEDED=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log() {
  printf '[gpu-host-bootstrap] %s\n' "$*" >&2
}

apt_install() {
  DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends "$@"
}

driver_ok() {
  local version
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
  [[ "${version%%.*}" =~ ^[0-9]+$ ]] && ((${version%%.*} >= MIN_DRIVER))
}

install_driver() {
  if driver_ok; then
    log "NVIDIA driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1) is ready"
    return 0
  fi
  log "installing ${DRIVER_PACKAGES[*]} from the Ubuntu archive"
  apt-get update
  if ! apt-cache policy "${DRIVER_PACKAGES[0]}" | grep -q 'Candidate: [0-9]'; then
    # The driver lives in `restricted`, which some installer images leave out.
    apt_install software-properties-common
    add-apt-repository --yes restricted
    apt-get update
  fi
  apt_install "${DRIVER_PACKAGES[@]}"
  if modprobe nvidia 2>/dev/null && driver_ok; then
    log "NVIDIA driver loaded without a reboot"
    return 0
  fi
  log "the new driver needs a reboot; run this step again afterwards"
  return "$REBOOT_NEEDED"
}

install_toolkit() {
  local keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  local tmp
  tmp="$(mktemp -d)"
  apt_install ca-certificates curl gnupg
  curl --fail --location --retry 3 --silent --show-error \
    --output "$tmp/gpgkey" "https://nvidia.github.io/libnvidia-container/gpgkey"
  if ! gpg --show-keys --with-colons "$tmp/gpgkey" \
    | grep -q "^fpr:::::::::${TOOLKIT_KEY_FINGERPRINT}:"; then
    log "unexpected key for the NVIDIA Container Toolkit repository"
    rm -rf "$tmp"
    return 1
  fi
  gpg --batch --yes --dearmor --output "$keyring" "$tmp/gpgkey"
  rm -rf "$tmp"
  echo "deb [signed-by=$keyring] $TOOLKIT_REPO/\$(ARCH) /" \
    >/etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt_install "nvidia-container-toolkit=$TOOLKIT_VERSION" \
    "nvidia-container-toolkit-base=$TOOLKIT_VERSION" \
    "libnvidia-container-tools=$TOOLKIT_VERSION" \
    "libnvidia-container1=$TOOLKIT_VERSION"
  log "NVIDIA Container Toolkit $TOOLKIT_VERSION installed"
}

# A one-node k3s whose containerd uses the NVIDIA runtime by default (k3s
# finds nvidia-container-runtime on its own), plus the NVIDIA device plugin
# so pods can request nvidia.com/gpu. This is the Kubernetes mode for a GPU
# host without a managed cluster, such as a Hetzner GEX server.
install_k3s() {
  local tmp
  if [[ "$(uname -m)" != x86_64 ]]; then
    log "k3s is pinned for x86_64 only"
    return 1
  fi
  driver_ok || { log "install the driver first"; return 1; }
  command -v nvidia-container-runtime >/dev/null 2>&1 || install_toolkit
  tmp="$(mktemp -d)"
  curl --fail --location --retry 3 --silent --show-error --output "$tmp/k3s" \
    "https://github.com/k3s-io/k3s/releases/download/${K3S_VERSION/+/%2B}/k3s"
  curl --fail --location --retry 3 --silent --show-error --output "$tmp/install.sh" \
    "https://raw.githubusercontent.com/k3s-io/k3s/${K3S_VERSION/+/%2B}/install.sh"
  if ! printf '%s  %s\n%s  %s\n' "$K3S_SHA256_AMD64" "$tmp/k3s" \
    "$K3S_INSTALL_SHA256" "$tmp/install.sh" | sha256sum --check --status; then
    log "SHA-256 mismatch for the k3s download"
    rm -rf "$tmp"
    return 1
  fi
  install -m 0755 "$tmp/k3s" /usr/local/bin/k3s
  INSTALL_K3S_SKIP_DOWNLOAD=true INSTALL_K3S_VERSION="$K3S_VERSION" \
    INSTALL_K3S_EXEC="server --default-runtime=nvidia --disable=traefik --write-kubeconfig-mode=0600" \
    sh "$tmp/install.sh"
  rm -rf "$tmp"
  local attempt
  for attempt in $(seq 1 60); do
    if k3s kubectl get nodes >/dev/null 2>&1; then
      break
    fi
    log "waiting for k3s ($attempt/60)"
    sleep 5
  done
  k3s kubectl apply -f "$ROOT/infra/kubernetes/nvidia-device-plugin.yaml"
  k3s kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset \
    --timeout=300s
  log "k3s $K3S_VERSION with the NVIDIA device plugin is ready"
}

main() {
  if ((EUID != 0)); then
    log "run as root"
    return 1
  fi
  case "${1:-}" in
    driver) install_driver ;;
    toolkit) install_toolkit ;;
    k3s) install_k3s ;;
    *)
      log "usage: gpu-host-bootstrap.sh driver|toolkit|k3s"
      return 2
      ;;
  esac
}

main "$@"
