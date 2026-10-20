# Issue #23: options compared

For each part of the solution, this page lists the existing tools and
libraries that were considered and why each was chosen or rejected. The
versions, licenses and prices come from
[research-notes/cloud-gpu-landscape.md](research-notes/cloud-gpu-landscape.md)
("landscape"); its sources are linked there. The requirement IDs refer to
[requirements.md](requirements.md).

## What a solution has to do

The comparisons below use these criteria, taken from the issue and the
repository:

- **C1. Four providers.** Hetzner, AWS, Azure and Google Cloud.
- **C2. Two runtimes.** A plain VM and a Kubernetes cluster.
- **C3. Not tied to a CI system.** It works from GitHub Actions, GitLab CI
  or a laptop.
- **C4. Two ways to run the tests.** On a self-hosted GitHub runner, or
  directly on the cloud machine.
- **C5. Pay only while testing.** Nothing keeps running, and nothing is
  forgotten.
- **C6. Few new moving parts.** No extra server, SaaS or large dependency.
  Logins without stored cloud keys where the cloud allows it.
- **C7. Fits the repository.** Python, checked with mypy and tests, and the
  existing GPU run scripts.

## Orchestration (R6, R10, R12)

| Option | C1 | C2 | C3 | C5 | C6 | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| [SkyPilot](https://github.com/skypilot-org/skypilot) v0.14.0, Apache-2.0 | No Hetzner backend; a Hetzner server only as an existing SSH node | VMs and Kubernetes | Yes | Auto-down (`-i N --down`) | Large Python dependency with its own state and API server | Rejected: no Hetzner ordering, and too heavy for one test command. Picking the cheapest cloud is useful, but the repository tests fixed machines. |
| [dstack](https://github.com/dstackai/dstack) 0.22.4, MPL-2.0 | No Hetzner backend; SSH fleets only | Yes | Yes | Yes | Needs a dstack server running | Rejected: a server to run, and no Hetzner |
| [Pulumi](https://github.com/pulumi/pulumi) v3.270.0 | Cloud providers yes; no Hetzner Robot ordering | Yes | Yes | Only if the program is written for it | Needs a state backend; a new IaC language for the team | Rejected: Terraform does the same with official, widely used modules, and the issue names Terraform |
| Terraform or OpenTofu alone | No Robot ordering | Yes | Yes | Only `destroy`; no power-off, reaper or test step | Small | Used as the resource layer, not as the whole flow |
| Cloud CLIs only (aws, az, gcloud scripts) | No | Possible | Yes | Hand-written | Small, but three different scripts | Rejected for creating resources (Terraform is declarative and reviewable). Used by the reaper and for the logins. |
| **Small Python orchestrator on top of Terraform, the Robot API, kubectl and SSH** (`scripts/cloud_gpu`) | Yes | Yes | Yes | Four layers (requirements R6) | Standard library only; mypy strict; tested with stub tools | **Chosen** |

## Cloud resources (R2–R5)

| Option | Verdict |
| --- | --- |
| **Official Terraform providers** (`hashicorp/aws`, `hashicorp/azurerm`, `hashicorp/google`) with committed lock files | **Chosen.** Maintained by HashiCorp and the clouds, full GPU support, and validated by `terraform validate` and TFLint in CI. |
| Community modules (terraform-aws-modules/eks and similar) | Rejected for now: dozens of resources a one-hour test cluster does not need, and more to review. The modules here create the minimum: a VPC or network, a firewall, a VM or cluster, and the GPU pool. |
| Terraform (BUSL-1.1) vs [OpenTofu](https://github.com/opentofu/opentofu) v1.13.1 (MPL-2.0) | Both supported: `--tool terraform` (default) or `--tool tofu`. The modules use no Terraform-only feature. |
| Remote state | Optional: `--backend-config` with `--backend-type` (s3, azurerm, gcs, or http for GitLab-managed state). A local state in the run's directory is the default, because `down` runs in the same job and the reaper finds leftovers by tag. |

## Hetzner ordering

| Option | Verdict |
| --- | --- |
| Hetzner Cloud API (`hcloud` Terraform provider, `hetzner.hcloud` Ansible collection, [hcloud-github-runner](https://github.com/Cyclenerd/hcloud-github-runner), [TestFlows Hetzner runners](https://github.com/testflows/TestFlows-GitHub-Hetzner-Runners), the GitLab hcloud fleeting plugin) | Rejected: Hetzner Cloud has no GPU server types (landscape §2.1). |
| [gen0sec/terraform-provider-hetzner-robot](https://github.com/gen0sec/terraform-provider-hetzner-robot) v0.1.0 | Rejected: one star, not in the registry, and destroying the order resource does **not** cancel the server, which defeats C5. |
| Ansible `community.hrobot` 2.8.0 | Not usable for ordering: it has no order or cancel module. Its inventory plugin could list servers for a host that is kept. |
| **Robot webservice called from Python** (`POST /order/server/transaction`, `GET /order/server/transaction/{id}`, `POST /server/{n}/cancellation`) | **Chosen.** It is the only documented way to order and cancel by API. Whether GEX products can be ordered this way is not verified; `plan --check` sends a `test=true` order first. |
| Server auction (`/order/server_market/transaction`) | Not used by default: the stock changes all the time. The auction has GPU servers without a setup fee, so docs/cloud-gpu.md lists it under "Other options" for a server that is kept. |
| Keep one GEX server for a month as a self-hosted runner | **Recommended for Hetzner.** The 209 EUR or 999 EUR setup fee per order makes ordering per test more expensive than a day on AWS. `infra/ansible` prepares such a host. |

## Host preparation

| Option | Verdict |
| --- | --- |
| The clouds' GPU images (AWS DLAMI Base OSS Nvidia Driver, Google DLVM, Azure Ubuntu with the `NvidiaGpuDriverLinux` extension, the EKS NVIDIA AMI, AKS and GKE driver installation) | **Chosen** for the clouds: the driver is ready at boot, with nothing to build. |
| [`nvidia.nvidia_driver`](https://github.com/NVIDIA/ansible-role-nvidia-driver) role v2.4.0 | Considered. It installs the same Ubuntu `-server` packages from the Canonical archive. The playbook installs `nvidia-driver-580-server-open` directly instead: it is one task, it needs no Galaxy download, and a test keeps it in step with `scripts/gpu-host-bootstrap.sh`. The role is a good replacement if more distributions are needed. |
| NVIDIA's CUDA repository (`cuda-toolkit-13-2`, `nvidia-open`), as in Hetzner's MIG tutorial | Rejected: llama.cpp and PyTorch bring their own CUDA libraries, so only the driver is needed, and Ubuntu's signed packages are simpler. |
| **Ansible playbooks** for hosts that are kept, and **a shell bootstrap over SSH** for temporary ones | **Chosen.** Ansible is idempotent and readable for a server that lives for months. A temporary VM needs nothing installed on the caller's side except ssh. |
| NVIDIA Container Toolkit v1.20.1 | Installed by the playbook (with the key fingerprint checked) for container jobs and k3s. |

## Kubernetes (R7)

| Option | Verdict |
| --- | --- |
| **EKS, AKS and GKE** with a GPU node pool | **Chosen** for the clouds: managed control plane, and the clouds install the GPU driver (`AL2023_x86_64_NVIDIA`, `gpu_driver = "Install"`, `gpu_driver_installation_config`). |
| [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) v26.7.2 | Not needed on managed clusters whose nodes already have the driver. It is the documented choice for a node without a driver; docs/cloud-gpu.md lists it under "Other options". |
| [NVIDIA k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) v0.20.1 | **Used** where the platform does not add it: EKS with the NVIDIA AMI, and k3s (`infra/kubernetes/nvidia-device-plugin.yaml`). |
| **k3s** on a Hetzner GEX server | **Chosen** for Hetzner: one binary, and it adds the `nvidia` runtime class automatically when it finds the toolkit. |
| [hetzner-k3s](https://github.com/vitobotta/hetzner-k3s) v2.6.0, [kube-hetzner](https://github.com/kube-hetzner/terraform-hcloud-kube-hetzner) v3.2.1 | Rejected: both build clusters from Hetzner Cloud servers. Robot GPU servers can only join as external nodes, which is more setup than a single-node k3s for a test. |
| kind | **Used in CI** to test the whole Job flow without a GPU. |
| One Kubernetes Job, rendered from `infra/kubernetes/medbench-gpu-job.yaml` | **Chosen** over a Helm chart or Argo Workflows: one pod, one command, and the Job can be rendered for any cluster with `job-manifest`. |

## Self-hosted GitHub runners (R11)

| Option | C1 | C5 | C6 | Verdict |
| --- | --- | --- | --- | --- |
| [machulav/ec2-github-runner](https://github.com/machulav/ec2-github-runner) v2.6.1 | AWS only | Start and stop jobs | Simple | Rejected: one cloud. Its start, GPU and stop job pattern is the one `--mode runner` follows. |
| [terraform-aws-github-runner](https://github.com/github-aws-runners/terraform-aws-github-runner) v7.11.0 | AWS only | Scale to zero | Lambdas, webhooks and a GitHub App | Rejected: many moving parts, one cloud |
| [RunsOn](https://github.com/runs-on/runs-on) v3.5.0 | AWS only | Yes | License needed for commercial use | Rejected |
| [Cirun.io](https://cirun.io/pricing) | AWS, Azure, GCP, Hetzner Cloud and others | Yes | A third party holds cloud credentials | Rejected: the credentials leave the project. Its Hetzner support is Hetzner Cloud, so it has no GPUs. |
| [actions-runner-controller](https://github.com/actions/actions-runner-controller) (gha-runner-scale-set 0.15.0) | Any cluster | Only while the cluster runs | A cluster that runs all the time | **Provided as an option** (`infra/kubernetes/arc-gpu-runner-values.yaml`) for a team that already runs a GPU cluster. It is not the default, because the cluster would cost money all the time. |
| GitHub-hosted GPU runner (`linux_4_core_gpu`, T4 16 GB, about $3.12/h) | – | Yes | None | Not usable: Team or Enterprise organizations only (this repository belongs to a user account), and 16 GB is too small for MedGemma 27B and UniMedVL. |
| **GitHub JIT runner API plus a temporary VM** (`--mode runner`) | All four | One job, then gone; the VM is deleted in `always()` | Uses actions/runner; needs a PAT or App token, because `GITHUB_TOKEN` cannot call this API | **Chosen** for temporary runners |
| **Ephemeral runner loop on a host that is kept** (`infra/ansible/github-runner.yml`) | Any server, including Hetzner GEX | A fresh runner registration per job | systemd plus actions/runner | **Chosen** for the Hetzner monthly server |

## Running without GitHub Actions (R10)

| Option | Verdict |
| --- | --- |
| GitLab Runner with the Docker Autoscaler and fleeting plugins (AWS, Google Cloud, Azure; the Hetzner plugin is hcloud only) | Rejected as the main path: it needs a GitLab Runner manager that runs all the time, autoscaling groups in every cloud, and no Hetzner GPUs. It is a good choice for a team that moves to GitLab for good (docs/cloud-gpu.md, "Other options"). |
| GitLab-hosted GPU runner (`saas-linux-medium-amd64-gpu-standard`, T4 16 GB) | Rejected: Premium or Ultimate only, and too small for the models. |
| **The same Python CLI from any CI or a laptop**, with a GitLab template that only installs tools, logs in and calls it | **Chosen.** The logic is not duplicated. The logins use GitLab ID tokens with `assume-role-with-web-identity`, `az login --federated-token` and a gcloud `external_account` file. |

## Other GPU sources (for comparison)

| Option | Verdict |
| --- | --- |
| [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) (`hf jobs run --flavor l4x1`) | Not used: not one of the four providers the issue names. It is the simplest alternative ($0.80/h for an L4, $2.50/h for an A100 80 GB, billed by the minute), needing only an HF token and a container image. Listed under "Other options" in docs/cloud-gpu.md. |
| Modal, RunPod, Lambda | Not used: not among the requested providers. Their prices are in landscape §8 for comparison. |
| Spot or preemptible capacity | **Supported** with `--spot`. It is off by default, because an interruption fails the run. |

## Safety against forgotten machines (C5)

| Option | Verdict |
| --- | --- |
| Cloud budgets (AWS Budgets actions, Azure and Google Cloud budgets) | Not enough: the cost data lags, and Azure and Google Cloud budgets only alert (landscape §10). docs/cloud-gpu.md recommends them as an extra alarm. |
| Azure auto-shutdown | Rejected: it runs once a day at a fixed time, and whether it deallocates is not documented. |
| **VM power-off timer that deletes the VM** (`shutdown -P +ttl`; AWS `terminate` on shutdown; Google Cloud `max_run_duration` with `DELETE`; Azure spot `Delete`) | **Chosen** |
| **Tags plus a scheduled reaper** | **Chosen**: the only layer that also covers clusters, networks and Hetzner orders when the CI job died. |
| **Teardown in `always()` and `after_script`, and on SIGTERM** | **Chosen** |
