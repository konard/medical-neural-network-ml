# Issue #23: requirements, plans and status

Each requirement is quoted or paraphrased from the issue text
([data/issue-23.txt](data/issue-23.txt)). For each one, this page lists:

- the solution plan
- the existing components it builds on
- what the pull request adds
- how that was verified
- what is still open

The options that were compared and rejected are in
[solution-options.md](solution-options.md). The facts and sources are in
[research-notes/cloud-gpu-landscape.md](research-notes/cloud-gpu-landscape.md),
here called "landscape".

Status values:

- **Done**: implemented and verified without a cloud account.
- **Done, cloud run open**: implemented and verified locally and in CI, but
  not yet run against a real cloud account (see
  [Blocked or unverified](#blocked-or-unverified)).
- **Partly unverified**: implemented, but it depends on a provider behavior
  that could not be checked.

## Summary

| ID | Requirement | Status |
| --- | --- | --- |
| R1 | Support Hetzner | Partly unverified |
| R2 | Support AWS | Done, cloud run open |
| R3 | Support Azure | Done, cloud run open |
| R4 | Support Google Cloud | Done, cloud run open |
| R5 | Terraform and Ansible where applicable | Done |
| R6 | Use the cloud APIs to deploy GPU servers briefly | Done, cloud run open |
| R7 | With or without a Kubernetes cluster | Done, cloud run open |
| R8 | Test that inference works and passes all tests | Done, cloud run open |
| R9 | GitHub Actions workflows | Done, cloud run open |
| R10 | Python scripts that work without GitHub Actions (for example GitLab) | Done, cloud run open |
| R11 | Tests on self-hosted GitHub Actions runners with GPUs | Done, cloud run open |
| R12 | Tests directly in the cloud providers | Done, cloud run open |
| R13 | Check the research escalation ladder against best practice and fix it | Done |
| R14 | Collect the issue's data in `docs/case-studies/issue-23` | Done |
| R15 | Deep case study with online research | Done |
| R16 | List every requirement | Done (this page) |
| R17 | Solutions and plans per requirement, including existing components and libraries | Done |
| R18 | Everything in this one pull request | Done, except the items that need cloud accounts |

## R1: Hetzner

> We should support https://www.hetzner.com

**Plan.** Hetzner Cloud has no GPU server types (landscape §2.1). Hetzner's
GPUs are the dedicated GEX servers (GEX45, GEX63, GEX131). They are ordered
through the Robot webservice, billed by the hour up to a monthly cap, and
each order has a one-time setup fee (landscape §2.2–§2.3). The plan has
three parts:

- **Order.** Order a GEX server through `POST /order/server/transaction` and
  wait until it is ready.
- **Prepare.** Install the driver over SSH (plain Ubuntu, so the same
  bootstrap script as the clouds, or the Ansible playbook for a server that
  is kept), then run the tests, a GitHub runner or k3s.
- **Cancel.** Cancel through `POST /server/{n}/cancellation`.

Because of the setup fee, the documentation recommends keeping one server
for a month as a self-hosted runner instead of ordering one per test.

**Existing components.**

- The Robot webservice API.
- `community.hrobot` (Ansible). It has no order module, so it is not used.
- The gen0sec Robot Terraform provider (v0.1.0, one star, not in the
  registry; destroying it does not cancel). Rejected, see
  [solution-options.md](solution-options.md#hetzner-ordering).
- k3s for Kubernetes.

**Done.**

- `scripts/cloud_gpu/hetzner.py`: order, wait, cancel and list, plus a
  `test=true` order for `plan --check`.
- The Hetzner flows in `scripts/cloud_gpu/flows.py` for the vm, runner and
  k8s modes (`HETZNER_STEPS`).
- `infra/ansible/gpu-host.yml`, `github-runner.yml` and `k3s-gpu.yml` for a
  server that is kept.
- The reaper cancels servers named `medbench-…` whose expiry has passed.
- The prices and setup fees in `scripts/cloud_gpu/config.py`, and the setup
  fee shown by `plan`.

**Verified.**

- `tests/test_cloud_gpu_apis.py` runs the order, wait, cancel and list calls
  against a stub Robot HTTP server.
- ansible-lint checks the playbooks (`scripts/check-infra.sh`).

**Open.** Whether GEX products can be ordered through the API, the exact
transaction fields, and how a fresh order behaves on cancellation all need
a Robot account (see [Blocked or unverified](#blocked-or-unverified)).

## R2–R4: AWS, Azure and Google Cloud

> AWS, Azure, Google cloud

**Plan.** For each cloud, provide one Terraform module for a GPU VM and one
for a small managed GPU cluster. Each module uses:

- the cloud's own GPU image and driver
- a firewall that admits only the caller's address
- `managed-by`, `run-id` and `expires-at` tags
- a self-deleting power-off after the time to live

Logins use OIDC, so CI stores no cloud keys (landscape §3–§6, §10).

**Existing components.**

- The official Terraform providers: `hashicorp/aws`, `hashicorp/azurerm`
  and `hashicorp/google`, pinned with lock files.
- The cloud GPU images:
  - AWS: the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04),
    and the `AL2023_x86_64_NVIDIA` node AMI on EKS.
  - Azure: Canonical's Ubuntu 24.04 with Microsoft's `NvidiaGpuDriverLinux`
    VM extension, and `gpu_driver = "Install"` on AKS.
  - Google Cloud: the Deep Learning VM, and `gpu_driver_installation_config`
    on GKE.
- The CI login actions: `aws-actions/configure-aws-credentials`,
  `azure/login` and `google-github-actions/auth`.

**Done.**

- `infra/terraform/{aws,azure,gcp}/{vm,k8s}`.
- `infra/terraform/cloud-init/gpu-vm.yaml.tftpl` (`shutdown -P +ttl`).
- The provider defaults in `scripts/cloud_gpu/config.py`:
  - AWS: g6.xlarge or g6.2xlarge (L4), g6e.2xlarge (L40S) for UniMedVL.
  - Google Cloud: g2-standard-4 or 8 (L4), a2-ultragpu-1g (A100 80 GB).
  - Azure: NC4as_T4_v3, NV36ads_A10_v5, NC24ads_A100_v4.
- The self-deleting power-off:
  - AWS: `instance_initiated_shutdown_behavior = "terminate"`.
  - Google Cloud: `max_run_duration` with `DELETE`.
  - Azure: a resource group per run, and `eviction_policy = "Delete"` for
    spot.
- The reaper (`scripts/cloud_gpu/reaper.py`) for all three clouds.

**Verified.**

- `terraform validate` and TFLint pass on all six modules
  (`scripts/check-infra.sh`, the `infra` CI job).
- `tests/test_cloud_gpu_infra.py` checks the tags, power-off settings,
  firewall rules and pins.
- `tests/test_cloud_gpu_orchestrator.py` runs up, run and down with stub
  `terraform`, `ssh` and cloud CLIs.

**Open.** A real run in each cloud, which needs accounts with GPU quota.

## R5: Terraform and Ansible where applicable

> with terraform/ansible where applicable

**Plan.**

- **Terraform** creates every cloud resource, because all three clouds have
  official providers with GPU support. Terraform is not used for Hetzner,
  because no official resource orders a dedicated server (landscape §2.4).
- **Ansible** prepares hosts that are kept: a Hetzner GEX server or any
  other GPU server. Temporary VMs use a shell script over SSH, so a run does
  not have to install Ansible. The playbook and the script install the same
  pinned versions.
- The orchestrator accepts `--tool tofu`, so OpenTofu can replace Terraform.

**Existing components.**

- Terraform and OpenTofu.
- ansible-core.
- The Ubuntu `nvidia-driver-580-server-open` packages and the NVIDIA
  Container Toolkit apt repository with its key fingerprint.
- The `nvidia.nvidia_driver` role was considered. The playbook installs the
  same Ubuntu server packages directly, to stay in step with the shell
  bootstrap ([solution-options.md](solution-options.md#host-preparation)).

**Done.**

- `infra/terraform/**` (six modules and the cloud-init template).
- `infra/ansible/{gpu-host,github-runner,k3s-gpu}.yml`, `inventory.example.yml`
  and `files/medbench-runner-loop.sh`.
- `scripts/gpu-host-bootstrap.sh`.
- `scripts/check-infra.sh`.

**Verified.**

- `scripts/check-infra.sh` runs terraform fmt and validate, TFLint,
  ansible-lint, kubeconform, helm lint and ShellCheck. It passes locally,
  in the `infra` CI job, and in the GitLab template's default image
  (`experiments/issue23_gitlab_image_tools.sh`).
- `tests/test_cloud_gpu_infra.py` keeps the playbook and the bootstrap
  script on the same driver and toolkit versions.

## R6: Use the cloud APIs to deploy GPU servers briefly

> uses API of cloud providers to shortly/temporary deploy servers with GPUs

**Plan.** Every run follows the same steps: `plan` (settings and worst-case
cost), `up` (create), `run` (test), and `down` (delete), with `all` doing
them in order. Four independent layers make sure no machine is forgotten:

1. `down` runs in a finally block, on SIGTERM, and in CI `always()` and
   `after_script` steps.
2. Each VM powers itself off, and the power-off deletes it.
3. Every resource has an `expires-at` tag, and a daily reaper deletes the
   expired ones.
4. A time to live of at most 24 hours (240 minutes by default).

**Existing components.**

- The cloud APIs through Terraform and the cloud CLIs (aws, az, gcloud).
- The Hetzner Robot API.
- The cloud tagging and labelling features.

**Done.**

- `scripts/cloud_gpu/` (cli, config, flows, terraform, remote, kube,
  github, hetzner, reaper, shell). It uses only the standard library and is
  checked with `mypy --strict`.
- The `reap` subcommand.
- The daily `reap` job in GitHub Actions and the `cloud-gpu:reap` job in
  GitLab.

**Verified.**

- `tests/test_cloud_gpu_orchestrator.py` checks that `down` runs after a
  failed run and on SIGTERM, and that the worst-case cost is printed.
- `tests/test_cloud_gpu_apis.py` tests the reaper's selection by tags.

## R7: With or without Kubernetes

> with or without kubernetes clusters

**Plan.** The modes:

- `--mode vm`: a GPU VM reached over SSH.
- `--mode k8s`: a small GPU cluster runs one Kubernetes Job built from
  `infra/kubernetes/medbench-gpu-job.yaml`.
  - EKS, AKS and GKE on the three clouds.
  - k3s on a Hetzner GEX server.
- `job-manifest`: writes the Job for a cluster you already have.

**Existing components.**

- EKS, AKS and GKE with their own GPU driver installation.
- k3s with its automatic `nvidia` runtime class.
- The NVIDIA k8s-device-plugin.
- kind for the CI test.

**Done.**

- `infra/terraform/{aws,azure,gcp}/k8s`.
- `infra/kubernetes/{medbench-gpu-job,nvidia-device-plugin,arc-gpu-runner-values}.yaml`.
- `infra/ansible/k3s-gpu.yml`.
- `scripts/cloud_gpu/kube.py`.
- `examples/cloud_gpu_kind.sh`.

**Verified.**

- The `kind-job` CI job runs the whole Job flow on kind: it renders the
  Job, streams the commit into the pod, installs medbench, runs it, and
  collects the reports. It uses 0 GPUs and UniMedVL's echo engine.
- kubeconform validates the manifests.
- `tests/test_cloud_gpu_orchestrator.py` tests the k8s flow with a stub
  `kubectl`.

## R8: Test whether inference works and passes all tests

> so we can test how our inference works, does it pass all the tests

**Plan.** On the GPU, run the same model command the repository already
uses for its GPU runs (`scripts/medbench-model-run.sh` through
`scripts/gpu-host-run.sh`), then copy the reports back:

- The run requires a GPU unless `--no-require-gpu` is given.
- It records `nvidia-smi` output.
- It enforces a time budget.

**Existing components.** The repository's `medbench` CLI, `medbench
monitor`, the model wrappers and `scripts/run-with-budget-warning.sh`.

**Done.**

- `scripts/gpu-host-run.sh` and `scripts/gpu-model-env.sh`.
- The reports artifact in both CI templates.

**Verified.**

- `tests/test_cloud_gpu_scripts.py` tests the host scripts with stubs.
- The kind job runs the real script inside a pod.

**Open.** A real GPU run of MedGemma and UniMedVL in a cloud.

## R9: GitHub Actions workflows

**Plan.** One workflow, `.github/workflows/cloud-gpu-tests.yml`, with these
jobs:

- `infra` and `kind-job` on pull requests.
- `orchestrate` and `gpu-tests`, started by hand from main in the
  `cloud-gpu` environment.
- A daily `reap` in the `cloud-gpu-reaper` environment.

Every action is pinned to a commit SHA, and zizmor and actionlint pass.

**Existing components.** `aws-actions/configure-aws-credentials`,
`azure/login`, `google-github-actions/auth`, `hashicorp/setup-terraform`,
`helm/kind-action`, the GitHub OIDC tokens and environments.

**Done.** The workflow; `tests/test_cloud_gpu_workflow.py`; and the setup
in [docs/cloud-gpu.md](../../cloud-gpu.md#github-actions) (environments,
variables, secrets, OIDC trust subjects, quota).

**Verified.**

- actionlint and zizmor (low confidence, regular persona) pass.
- The `infra` and `kind-job` jobs pass in the pull request's CI.
- The workflow test checks the triggers, environments, `always()` teardown
  and path filters.

## R10: Python scripts that work without GitHub Actions

> python scripts, that will work independently from GitHub Actions, for
> example if we need to migrate to GitLab

**Plan.**

- All the logic lives in `python -m scripts.cloud_gpu`. The GitHub workflow
  only installs tools, logs in and calls it.
- A GitLab CI template does the same:
  - It logs in through GitLab's OIDC ID tokens.
  - It deletes the machines in `after_script`, with
    `RUNNER_SCRIPT_TIMEOUT` set so that `after_script` still runs before the
    job's timeout.
- A tool installer prepares any Debian or Ubuntu image.

**Existing components.**

- GitLab `id_tokens`.
- `aws sts assume-role-with-web-identity`.
- `az login --federated-token`.
- gcloud with an `external_account` credentials file.
- `resource_group` (one run per cloud at a time).

**Done.**

- `scripts/cloud_gpu/`.
- `ci/gitlab/cloud-gpu.gitlab-ci.yml` and `ci/gitlab/cloud-login.sh`.
- `scripts/cloud-gpu-install-tools.sh`.
- The GitLab section of [docs/cloud-gpu.md](../../cloud-gpu.md#gitlab-ci).

**Verified.**

- `tests/test_cloud_gpu_gitlab.py` parses the template: rules, timeouts,
  `id_tokens` and `after_script`. It also runs `cloud-login.sh` against stub
  `aws`, `az` and `gcloud` commands, and runs the installer's argument
  handling.
- The `infra` and `aws`, `azure` and `gcp` tool sets were installed inside
  `python:3.14-slim-trixie` (the template's default image), and
  `check-infra.sh` and `plan` passed there
  (`experiments/issue23_gitlab_image_tools.sh`).
- The same CLI runs from a laptop.

**Open.** No real GitLab pipeline was run. The repository is on GitHub, and
the cloud logins need GitLab-side trust settings.

## R11: Self-hosted GitHub Actions runners with GPUs

> flows, where we run tests on GitHub Actions Runners with GPU access hosted
> by us

**Plan.** Two flows:

- **A temporary runner.** `--mode runner` creates a GPU VM and registers a
  just-in-time runner with a label unique to the run. The `gpu-tests` job
  runs on it, and the runner takes one job and is gone. The JIT
  configuration comes from a PAT or GitHub App token stored in the
  environment, because `GITHUB_TOKEN` cannot call that API (landscape §7.1).
- **A host that is kept.** Hetzner GEX or any GPU server, prepared with
  Ansible, runs an ephemeral runner in a loop. For a cluster that runs all
  the time, there are actions-runner-controller values for a GPU runner
  scale set.

**Existing components.** actions/runner (`--jitconfig`, `--ephemeral`), the
GitHub JIT runner REST API, and actions-runner-controller
(gha-runner-scale-set).

**Done.**

- `scripts/cloud_gpu/github.py`.
- `scripts/gpu-runner-start.sh`.
- `infra/ansible/github-runner.yml` and `files/medbench-runner-loop.sh`.
- `infra/kubernetes/arc-gpu-runner-values.yaml`.
- The `gpu-tests` job.

**Verified.**

- `tests/test_cloud_gpu_apis.py` tests the JIT config and runner removal
  against a stub GitHub API.
- `tests/test_cloud_gpu_scripts.py` tests the runner start script.
- helm lint checks the ARC values.

**Open.**

- `runner_group_id` 1 for repository runners is not documented
  (landscape §13).
- No real runner was registered.

## R12: Tests directly in the cloud providers

> or directly in these cloud providers

**Plan.** The vm and k8s modes run the tests on the cloud machine itself,
over SSH or as a Kubernetes Job, from wherever the CLI runs: GitHub, GitLab
or a laptop. Nothing has to be registered with GitHub.

**Done and verified.** Same as R6 and R7.

## R13: The research escalation ladder

> double check for all escalation ladder of research we use all industry
> leading best practices, so if you see something wrong that must be
> corrected

**Plan.** Check every claim of the ladder in `docs/case-studies/issue-1`
against a primary source. Fix the wrong, outdated and imprecise ones, and
add missing practices.

**Done.** [ladder-review.md](ladder-review.md) records each claim, its
verdict and its source. The corrections are applied in
`docs/case-studies/issue-1/{research-plan,if-models-fail,ci-and-gpu-runner}.md`
and `research-notes/training-and-data.md`. The fixes:

- 3 wrong claims
- 3 outdated claims
- 5 imprecise claims
- 4 claims missing an important detail
- 3 claims with a weak source or missing context
- 8 practices added

`docs/hardware-workflows.md` was also corrected.

**Verified.** Each verdict cites the page it was checked against. The docs
build (Sphinx with `-W`) and the link check pass.

## R14–R17: Case study

> collect data related about the issue … compile that data to
> `./docs/case-studies/issue-{id}` … deep case study analysis (also make
> sure to search online for additional facts and data), list of each and all
> requirements … propose possible solutions and solution plans for each
> requirement … check known existing components/libraries

**Done.**

- [data/](data/): price tables, quota data, registry answers, release
  lists, API notes and the issue text, each with its source and date.
- [research-notes/cloud-gpu-landscape.md](research-notes/cloud-gpu-landscape.md):
  the online research, with a source for every fact and a list of what is
  not verified.
- This page (R16) and [solution-options.md](solution-options.md) (R17).
- [README.md](README.md): the summary.

## R18: Everything in this one pull request

Everything above is in pull request
[#24](https://github.com/konard/medical-neural-network-ml/pull/24). The items
in the next section need cloud accounts, quota or money that this work did
not have. They are left open, not reported as done.

## Blocked or unverified

| Item | Why it is open | What covers it meanwhile |
| --- | --- | --- |
| A real run on AWS, Azure, Google Cloud and Hetzner in every mode | Needs accounts, GPU quota (new accounts have none) and money; the repository has no cloud settings yet | `terraform validate` and TFLint, stub-tool tests, and the kind Job run in CI |
| Ordering GEX servers through the Robot API, the transaction fields read, and cancelling a fresh order | Needs a Robot account; the API docs do not mention GPUs | `plan --check` sends a `test=true` order first; tests run against a stub Robot server |
| `runner_group_id` 1 for repository runners | Not in the REST docs | Stub API test; GitHub answers with an error if it is wrong |
| The AWS reaper deleting a VPC while instances are still terminating | AWS dependency ordering | The next daily run deletes it |
| A cancelled GitHub run gives `always()` steps only limited time | GitHub behavior | The VM power-off timer and the reaper |
| A GitLab job killed hard (runner lost) loses the local Terraform state | No remote state by default | The power-off timer and the reaper, or `--backend-config` for remote state |
| The GitLab logins on a real GitLab instance | The project is on GitHub | Stub tests of `cloud-login.sh`; the tool installer run in the default image |
| The AWS role's maximum session must be at least 6 hours | An account setting | Documented in docs/cloud-gpu.md; the workflow asks for 21600 seconds |
| kind inside this sandbox | No nested containers here | The `kind-job` CI job runs it |
| The Azure and Google Cloud default GPU quotas, the EKS default taint, and Azure auto-shutdown deallocation | Not documented (landscape §13) | The EKS Job tolerates the `nvidia.com/gpu` taint anyway; quota is a manual first step in docs/cloud-gpu.md |
