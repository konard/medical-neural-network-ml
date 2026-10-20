# shellcheck shell=bash
# Log in to one cloud from a GitLab CI job with its OIDC ID token
# (ci/gitlab/cloud-gpu.gitlab-ci.yml). Sourced, not run, so the exports reach
# the job's shell:
#
#   source ci/gitlab/cloud-login.sh aws|azure|gcp|hetzner
#
# The job's id_tokens give AWS_ID_TOKEN, AZURE_ID_TOKEN and GCP_ID_TOKEN.
# Each token is written to a file the cloud SDKs read again whenever they
# need new credentials, so Terraform, kubectl (aws eks get-token,
# gke-gcloud-auth-plugin) and the CLIs keep working for the whole job: GitLab
# makes the tokens valid for the job's timeout. Settings (CI/CD variables):
#   aws      AWS_ROLE_ARN (role trusting this GitLab's OIDC provider)
#   azure    AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID
#   gcp      GCP_WORKLOAD_IDENTITY_PROVIDER (projects/N/locations/global/
#            workloadIdentityPools/POOL/providers/PROVIDER), GCP_SERVICE_ACCOUNT,
#            GCP_PROJECT
#   hetzner  HETZNER_ROBOT_USER, HETZNER_ROBOT_PASSWORD (masked)
# docs/cloud-gpu.md shows the cloud-side trust for each.

cloud_gpu_login() {
  local provider="$1" missing=() name needs
  local dir="${CLOUD_GPU_LOGIN_DIR:-${CI_BUILDS_DIR:-/tmp}/cloud-gpu-login-${CI_JOB_ID:-local}}"
  # A case, not an associative array, so bash 3.2 (macOS) can source it too.
  case "$provider" in
    aws) needs="AWS_ROLE_ARN AWS_ID_TOKEN" ;;
    azure) needs="AZURE_CLIENT_ID AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID AZURE_ID_TOKEN" ;;
    gcp) needs="GCP_WORKLOAD_IDENTITY_PROVIDER GCP_SERVICE_ACCOUNT GCP_PROJECT GCP_ID_TOKEN" ;;
    hetzner) needs="HETZNER_ROBOT_USER HETZNER_ROBOT_PASSWORD" ;;
    *)
      echo "cloud-login: unknown provider '$provider' (aws, azure, gcp, hetzner)" >&2
      return 2
      ;;
  esac
  for name in $needs; do
    [ -n "${!name:-}" ] || missing+=("$name")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "cloud-login: this job has no ${missing[*]}. Add the CI/CD variables (protected) or the job's id_tokens; docs/cloud-gpu.md lists every setting." >&2
    return 1
  fi
  mkdir -p "$dir"
  chmod 700 "$dir"
  case "$provider" in
    aws)
      # The AWS SDKs, the CLI and Terraform's AWS provider assume the role
      # with this file whenever their credentials expire.
      printf '%s' "$AWS_ID_TOKEN" > "$dir/aws-token"
      export AWS_WEB_IDENTITY_TOKEN_FILE="$dir/aws-token"
      export AWS_ROLE_SESSION_NAME="medbench-gl-${CI_JOB_ID:-local}"
      export AWS_ROLE_ARN
      export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${CLOUD_GPU_REGION:-us-east-1}}"
      aws sts get-caller-identity --query Arn --output text
      ;;
    azure)
      # az for the reaper; Terraform's azurerm provider reads the token
      # file itself (ARM_USE_OIDC).
      printf '%s' "$AZURE_ID_TOKEN" > "$dir/azure-token"
      export ARM_USE_OIDC=true
      export ARM_OIDC_TOKEN_FILE_PATH="$dir/azure-token"
      export ARM_CLIENT_ID="$AZURE_CLIENT_ID"
      export ARM_TENANT_ID="$AZURE_TENANT_ID"
      export ARM_SUBSCRIPTION_ID="$AZURE_SUBSCRIPTION_ID"
      az login --service-principal --username "$AZURE_CLIENT_ID" \
        --tenant "$AZURE_TENANT_ID" --federated-token "$AZURE_ID_TOKEN" --output none
      az account set --subscription "$AZURE_SUBSCRIPTION_ID"
      ;;
    gcp)
      # An external-account credential file: Google's libraries (Terraform,
      # gcloud, gke-gcloud-auth-plugin) exchange the token file for a short
      # access token of the service account whenever they need one.
      printf '%s' "$GCP_ID_TOKEN" > "$dir/gcp-token"
      gcloud iam workload-identity-pools create-cred-config \
        "$GCP_WORKLOAD_IDENTITY_PROVIDER" \
        --service-account="$GCP_SERVICE_ACCOUNT" \
        --credential-source-file="$dir/gcp-token" \
        --output-file="$dir/gcp-credentials.json"
      export GOOGLE_APPLICATION_CREDENTIALS="$dir/gcp-credentials.json"
      export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$dir/gcp-credentials.json"
      export CLOUDSDK_CORE_PROJECT="$GCP_PROJECT"
      export CLOUD_GPU_GCP_PROJECT="$GCP_PROJECT"
      gcloud auth print-access-token >/dev/null
      ;;
    hetzner)
      # The Robot API uses HTTP basic authentication from Python.
      export HETZNER_ROBOT_USER HETZNER_ROBOT_PASSWORD
      ;;
  esac
  echo "cloud-login: logged in to $provider" >&2
}

cloud_gpu_login "$@"
