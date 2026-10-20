# GPU tests on temporary cloud machines

The GPU model tests (`scripts/gpu-host-run.sh`: download a Hugging Face
model, serve it, run the medbench GPU tasks, write `medbench-reports/`) need
an NVIDIA GPU with 16 to 80 GB of memory. GitHub-hosted runners have none,
and a GPU machine that runs all day costs hundreds of dollars a month. So the
tests rent a GPU machine for one run, from AWS, Azure, Google Cloud or
Hetzner, and delete it afterwards.

`python -m scripts.cloud_gpu` does the whole job. It uses only the Python
standard library and calls Terraform (or OpenTofu), `ssh`, `kubectl` and the
clouds' command line tools, so it runs the same on a laptop, in GitHub
Actions (`.github/workflows/cloud-gpu-tests.yml`), in GitLab CI
(`ci/gitlab/cloud-gpu.gitlab-ci.yml`) or from cron. The research behind the
choices, with prices and sources, is in the issue #23 case study
(`docs/case-studies/issue-23`).

## Modes and providers

| Mode | What runs where | Providers |
| --- | --- | --- |
| `vm` | A GPU VM (or Hetzner dedicated server) is created; the repository is copied over SSH and `scripts/gpu-host-run.sh` runs there; the reports are copied back | aws, azure, gcp, hetzner, ssh, local |
| `k8s` | A GPU Kubernetes cluster is created (EKS, AKS, GKE, or k3s on a Hetzner server); the tests run as a Job (`infra/kubernetes/medbench-gpu-job.yaml`) | aws, azure, gcp, hetzner, ssh, kubernetes |
| `runner` | A GPU VM registers a just-in-time GitHub Actions runner for this run only; a job of the same workflow runs on it, then the runner and the VM are gone | aws, azure, gcp, hetzner, ssh |

`ssh` is a GPU host you keep (prepared with `infra/ansible/gpu-host.yml`),
`kubernetes` an existing cluster (your `KUBECONFIG`), `local` this machine.

Default machines, the cheapest that fit each model (on-demand prices per
hour read on 2026-10-19 in us-east-1, us-central1, eastus and Falkenstein;
`plan` prints the settings and the worst-case cost of a run):

| Model | AWS | Google Cloud | Azure | Hetzner (EUR) |
| --- | --- | --- | --- | --- |
| medgemma-4b-it | g6.xlarge, L4 24 GB, $0.80 | g2-standard-4, L4, $0.71 | Standard_NC4as_T4_v3, T4 16 GB, $0.53 | GEX45, 0.34 |
| medgemma-27b-it | g6.2xlarge, L4, $0.98 | g2-standard-8, L4, $0.85 | Standard_NV36ads_A10_v5, A10, $3.20 (k8s: NC24ads_A100_v4, $3.67) | GEX45, 0.34 |
| unimedvl | g6e.2xlarge, L40S 48 GB, $2.24 | a2-ultragpu-1g, A100 80 GB, $5.07 | Standard_NC24ads_A100_v4, A100 80 GB, $3.67 | GEX63, 1.60 |

Hetzner Cloud has no GPU servers; the GEX line is dedicated hardware ordered
through the Robot API with a one-time setup fee (209 EUR for GEX45, 999 EUR for GEX63) and monthly billing, so it
pays off only for long or repeated runs. Ordering GEX servers through the API
was not verified; `plan --check` places a test order.

## Running it yourself

```bash
# settings and worst-case cost; no account needed
python -m scripts.cloud_gpu plan --provider aws --model medgemma-27b-it
# create, test, copy the reports, delete (also after a failure or Ctrl-C)
python -m scripts.cloud_gpu all --provider gcp --mode k8s \
    --model medgemma-4b-it --gcp-project my-project --yes
# the same in steps, for example across CI jobs
python -m scripts.cloud_gpu up --provider azure --run-id ci-42 --yes
python -m scripts.cloud_gpu run --workdir .cache/cloud-gpu/ci-42
python -m scripts.cloud_gpu down --workdir .cache/cloud-gpu/ci-42
# the Kubernetes Job for an existing cluster
python -m scripts.cloud_gpu job-manifest --model medgemma-4b-it | kubectl apply -f -
# delete what crashed runs left behind
python -m scripts.cloud_gpu reap --provider aws --region us-east-1 --dry-run
```

Useful options: `--spot` (cheaper, may be interrupted), `--region`,
`--instance-type`, `--count` (items per task), `--tasks`, `--ttl` (minutes),
`--tool tofu`, `--backend-config` (remote Terraform state; the backend
is s3, azurerm or gcs unless `--backend-type` names another, such as
`http` for GitLab-managed state),
`--allowed-cidr` (default: this machine's public address). Without `--yes`,
`up` and `all` refuse to create paid resources. Logins come from the clouds'
usual places (`aws configure`/`AWS_PROFILE`, `az login` or `ARM_*`,
`gcloud auth application-default login`, `HETZNER_ROBOT_USER` and
`HETZNER_ROBOT_PASSWORD`). Runner mode needs `CLOUD_GPU_RUNNER_TOKEN` (or
`GH_TOKEN`).

`bash scripts/cloud-gpu-install-tools.sh infra aws azure gcp` installs the
pinned tools (Terraform, TFLint, kubectl, kubeconform, Helm, ShellCheck,
ansible-lint, yamllint, AWS CLI, Azure CLI, gcloud with
gke-gcloud-auth-plugin) on a Debian or Ubuntu image, each download checked
against its SHA-256.

The working directory (`.cache/cloud-gpu/<run id>`) holds the run's SSH key,
kubeconfig and Terraform state. Keep it until `down` has finished and never
publish it; only the reports directory is meant to be shared.

## How a machine is never forgotten

1. `down` runs after every run: in `all` it is in a `finally`, SIGTERM and
   SIGHUP (a cancelled CI job) turn into a normal exit that still runs it,
   and the CI jobs run it in a step that runs after any failure.
2. Every VM runs `shutdown -P +<ttl>` from cloud-init, so it powers off by
   itself (AWS: `instance_initiated_shutdown_behavior = terminate`).
3. Every resource carries `managed-by=medbench-cloud-gpu` and
   `expires-at=<unix time>`; `reap`, scheduled daily, deletes expired ones in
   every configured cloud, and stale runners on GitHub.
4. `--ttl` is capped at 24 hours, and `plan` prints the worst-case cost
   (hourly price × TTL, plus the Hetzner setup fee).

## GitHub Actions

`.github/workflows/cloud-gpu-tests.yml`:

- **infra** (pull requests touching the infrastructure): `terraform
  validate`, TFLint, `ansible-lint`, `kubeconform`, `helm template` and
  ShellCheck through `scripts/check-infra.sh`. No cloud account.
- **kind-job**: the whole Kubernetes Job flow on a kind cluster
  (`examples/cloud_gpu_kind.sh`), with 0 GPUs and UniMedVL's echo engine.
- **orchestrate** (Actions → Cloud GPU Tests → Run workflow, from `main`
  only): choose action (`plan`, `test`, `reap`), provider, mode, model,
  items per task, spot and TTL. In runner mode the **gpu-tests** job runs on
  the just-in-time runner.
- **reap**: daily at 04:41 UTC.

Create the environments `cloud-gpu` (orchestrate) and `cloud-gpu-reaper`
(reap) under Settings → Environments, restrict both to the `main` branch,
and add only the settings of the clouds you use. A missing setting stops the
run before anything is created, with the name of what to add.

| Name | Kind | Used for |
| --- | --- | --- |
| `AWS_ROLE_ARN` | variable | AWS role assumed through OIDC |
| `CLOUD_GPU_REGION` | variable | Default region or location (otherwise us-east-1, eastus, us-central1, FSN1) |
| `CLOUD_GPU_AWS_REGIONS` | variable | Space-separated AWS regions the reaper sweeps |
| `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` | variables | Azure app registration or managed identity with a federated credential |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | variable | `projects/<number>/locations/global/workloadIdentityPools/<pool>/providers/<provider>` |
| `GCP_SERVICE_ACCOUNT`, `GCP_PROJECT` | variables | Service account to impersonate, project to create in |
| `HETZNER_ROBOT_USER`, `HETZNER_ROBOT_PASSWORD` | secrets | Robot webservice user (Hetzner has no OIDC) |
| `CLOUD_GPU_RUNNER_TOKEN` | secret | Runner mode and the runner reaper: fine-grained token for this repository with "Administration: read and write" (`GITHUB_TOKEN` cannot register runners) |

The OIDC trust on the cloud side:

- **AWS**: an IAM OIDC provider for `https://token.actions.githubusercontent.com`
  (audience `sts.amazonaws.com`) and a role whose trust policy requires
  `token.actions.githubusercontent.com:sub` to be
  `repo:konard/medical-neural-network-ml:environment:cloud-gpu` (and
  `...:environment:cloud-gpu-reaper` for the reaper). Set the role's
  maximum session duration to at least 6 hours: the workflow asks for a
  6-hour session because `down` and, in k8s mode, every `kubectl` call
  (`aws eks get-token`) use it hours after the login.
- **Azure**: a federated credential with issuer
  `https://token.actions.githubusercontent.com`, the same subjects and
  audience `api://AzureADTokenExchange`; Contributor on the subscription (or
  a resource-group scope plus the right to create resource groups).
- **Google Cloud**: a workload identity pool with a GitHub provider
  (attribute condition `assertion.repository == 'konard/medical-neural-network-ml'`),
  and `roles/iam.workloadIdentityUser` on the service account for the
  environments' subjects; the service account needs Compute Admin,
  Kubernetes Engine Admin and Service Account User in the project.

Azure and Google Cloud refresh their tokens from the runner's OIDC endpoint
by themselves. GPU quota is zero on new accounts in all three clouds;
request it first (the case study lists the quota names).

## GitLab CI

Include the template from the project's `.gitlab-ci.yml`:

```yaml
include:
  - local: ci/gitlab/cloud-gpu.gitlab-ci.yml
stages: [check, cloud, reap]
```

| Job | When | What |
| --- | --- | --- |
| `cloud-gpu:check-infra` | Merge requests that touch the infrastructure, pushes to the default branch | `scripts/check-infra.sh` after `scripts/cloud-gpu-install-tools.sh infra` |
| `cloud-gpu:plan` | Merge requests, and before a test | Settings and worst-case cost |
| `cloud-gpu:test` | Run pipeline (web or API) or schedule on a protected branch with `CLOUD_GPU_TEST=true` | `up` and `run` in `script`, `down` in `after_script` |
| `cloud-gpu:reap` | Schedule or run pipeline on a protected branch with `CLOUD_GPU_REAP=true` | `reap` for every cloud that has settings |

Choose the run with `CLOUD_GPU_PROVIDER`, `CLOUD_GPU_MODE` (`vm` or `k8s`;
runner mode registers GitHub runners and is GitHub only), `CLOUD_GPU_MODEL`,
`CLOUD_GPU_COUNT`, `CLOUD_GPU_TTL`, `CLOUD_GPU_SPOT` and `CLOUD_GPU_REGION`,
for example as variables of a pipeline schedule. The cloud settings have the
same names as in GitHub (`AWS_ROLE_ARN`, `AZURE_*`, `GCP_*`,
`HETZNER_ROBOT_*`, `CLOUD_GPU_AWS_REGIONS`); add them as protected CI/CD
variables (masked for the Hetzner password) so only protected branches see
them.

The logins (`ci/gitlab/cloud-login.sh`) use GitLab's OIDC ID tokens
(`id_tokens`): AWS through `AWS_WEB_IDENTITY_TOKEN_FILE` and `AWS_ROLE_ARN`,
Azure through `az login --federated-token` and `ARM_OIDC_TOKEN_FILE_PATH`,
Google Cloud through an external-account credential file from `gcloud iam
workload-identity-pools create-cred-config`. GitLab makes the tokens valid
for the job's timeout, and the SDKs read the token files again whenever they
need new credentials. The cloud-side trust is the same as for GitHub with
GitLab as issuer (`https://gitlab.com` or your instance), the subject
`project_path:<group>/<project>:ref_type:branch:ref:main`, and the
audiences `sts.amazonaws.com` (AWS, `CLOUD_GPU_AWS_AUDIENCE`),
`api://AzureADTokenExchange` (Azure, `CLOUD_GPU_AZURE_AUDIENCE`) and
`https://iam.googleapis.com/<GCP_WORKLOAD_IDENTITY_PROVIDER>` (Google Cloud).

GitLab runs `after_script` after a failure and after a cancellation (GitLab
17.0 and later) but not after a job timeout, so the test job stops `script`
after 5 hours (`RUNNER_SCRIPT_TIMEOUT`) and gives `after_script` 45 minutes
(`RUNNER_AFTER_SCRIPT_TIMEOUT`) within the 6-hour job timeout. The working
directory is never an artifact; only `cloud-gpu-reports/` is kept.

## Self-hosted GPU runners

For GPU machines that stay up (a monthly Hetzner GEX server, a lab
workstation):

- `infra/ansible/gpu-host.yml` installs the NVIDIA R580 driver and the
  container toolkit; `--provider ssh` then runs the tests on it.
- `infra/ansible/github-runner.yml` installs a service that registers one
  just-in-time runner at a time, runs one job as an unprivileged user and
  wipes the work directory before the next one.
- `infra/ansible/k3s-gpu.yml` makes it a one-node k3s cluster with GPU
  support, for `--provider kubernetes` or for Actions Runner Controller
  (`infra/kubernetes/arc-gpu-runner-values.yaml`), which starts an ephemeral
  runner pod per job.

Never let pull requests from forks run on a persistent GPU runner: the
workflow's GPU jobs run only on dispatch from `main`.

## Other options

These are not built in, but they fit next to the templates. The case study
for issue #23 compares them in detail.

- **Hetzner Server Auction.** These servers have no setup fee, and some have
  GPUs, for example an RTX 6000 Ada (48 GB) for about 930 EUR a month. The
  stock changes all the time, so the orchestrator does not order from it.
  To use one as a server that is kept, order it in Robot and prepare it
  with `infra/ansible/gpu-host.yml`.
- **NVIDIA GPU Operator.** It installs the driver, the toolkit and the
  device plugin on a cluster whose GPU nodes have no driver. EKS, AKS, GKE
  and the k3s playbook do not need it.
- **GitLab Runner autoscaling** (the Docker Autoscaler with the AWS, Google
  Cloud or Azure fleeting plugins) suits a project that moves to GitLab for
  good and wants GPU runners there instead of `cloud-gpu:test`. It needs a
  runner manager that runs all the time, and the Hetzner plugin has no GPUs.
- **Cloud budgets** are a useful extra alarm, not a stop. AWS Budgets
  actions can stop instances. Azure and Google Cloud budgets only alert.
  Billing data lags by hours.
- **Hugging Face Jobs** (`hf jobs run --flavor l4x1 …`) run a container on an
  L4 or A100 billed by the minute. They need only a Hugging Face token, but
  they are not one of the four providers here.

## Known limits

- No run against a real cloud account has been made yet: it needs accounts,
  GPU quota and money. The Terraform modules are validated, the orchestrator
  is unit-tested with stub tools, and the Kubernetes Job flow runs end to end
  on kind in CI.
- Hetzner: ordering GEX servers through the Robot API, the transaction
  fields the orchestrator reads and the cancellation of a fresh order are
  unverified.
- The AWS reaper may fail to delete a VPC while its instances are still
  terminating; the next run deletes it.
- A cancelled GitHub run gives `if: always()` steps limited time; the VMs'
  power-off timer and the reaper are the safety net behind it.
