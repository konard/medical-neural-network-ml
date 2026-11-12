#!/usr/bin/env bash
# Resolve SHA-256 digests for the pinned tool downloads updated in issue #30.
# Uses upstream checksum files where published; downloads and hashes the
# artifact otherwise (AWS CLI, gcloud). Usage: bash experiments/issue30_tool_checksums.sh
set -euo pipefail
TF=1.17.2 TFLINT=v0.66.0 AWSCLI=2.40.1 GCLOUD=593.0.0 K3S=v1.37.2+k3s1 UV=0.13.7
k3s_enc=${K3S/+/%2B}
echo "terraform $TF $(curl -fsSL https://releases.hashicorp.com/terraform/$TF/terraform_${TF}_SHA256SUMS | awk '/linux_amd64.zip/{print $1}')"
echo "tflint $TFLINT $(curl -fsSL https://github.com/terraform-linters/tflint/releases/download/$TFLINT/checksums.txt | awk '$2=="tflint_linux_amd64.zip"{print $1}')"
echo "k3s $K3S $(curl -fsSL https://github.com/k3s-io/k3s/releases/download/$k3s_enc/sha256sum-amd64.txt | awk '$2=="k3s"{print $1}')"
echo "k3s-install $K3S $(curl -fsSL https://raw.githubusercontent.com/k3s-io/k3s/$k3s_enc/install.sh | sha256sum | cut -d' ' -f1)"
for a in x86_64 aarch64; do
  echo "uv-$a $UV $(curl -fsSL https://github.com/astral-sh/uv/releases/download/$UV/uv-$a-unknown-linux-gnu.tar.gz.sha256 | cut -d' ' -f1)"
done
echo "awscli $AWSCLI $(curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64-$AWSCLI.zip | sha256sum | cut -d' ' -f1)"
echo "gcloud $GCLOUD $(curl -fsSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-$GCLOUD-linux-x86_64.tar.gz | sha256sum | cut -d' ' -f1)"
