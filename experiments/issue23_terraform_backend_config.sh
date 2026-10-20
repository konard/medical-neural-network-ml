#!/usr/bin/env bash
# Issue #23: does `terraform init -backend-config=FILE` work for a module
# without a backend block? The cloud GPU modules declare none.
#
# Result on 2026-10-20 with Terraform v1.16.6:
#   - without a backend block, init succeeds, warns "Missing backend
#     configuration: -backend-config was used without a "backend" block",
#     and the state stays in ./terraform.tfstate: the option is ignored;
#   - with backend_override.tf declaring `backend "local" {}`, the state goes
#     to the path from the config file.
# So scripts/cloud_gpu/terraform.py writes backend_override.tf (s3, azurerm,
# gcs, or --backend-type) whenever --backend-config is given.
set -euo pipefail
tool=${TOOL:-terraform}
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
cd "$dir"
echo 'output "x" { value = 1 }' > main.tf
echo "path = \"$dir/remote.tfstate\"" > backend.hcl

"$tool" init -input=false -no-color -backend-config=backend.hcl > init1.log 2>&1
"$tool" apply -input=false -auto-approve -no-color > /dev/null
grep -q 'Missing backend configuration' init1.log && echo "without a backend block: warning, option ignored"
[ -f terraform.tfstate ] && [ ! -f remote.tfstate ] && echo "  state stayed in ./terraform.tfstate"

rm -rf .terraform .terraform.lock.hcl terraform.tfstate*
printf 'terraform {\n  backend "local" {}\n}\n' > backend_override.tf
"$tool" init -input=false -no-color -backend-config=backend.hcl > /dev/null
"$tool" apply -input=false -auto-approve -no-color > /dev/null
[ -f remote.tfstate ] && [ ! -f terraform.tfstate ] && echo "with backend_override.tf: state in the configured path"
