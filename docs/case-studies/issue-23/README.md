# Case study: issue #23, CI/CD templates for GPU inference tests

Issue: <https://github.com/konard/medical-neural-network-ml/issues/23>.
Pull request: <https://github.com/konard/medical-neural-network-ml/pull/24>.

The issue asks for a way to test the Hugging Face models (MedGemma 4B and
27B, UniMedVL) on real GPUs without keeping a GPU machine. It should rent a
GPU server on Hetzner, AWS, Azure or Google Cloud for the length of a test,
with or without Kubernetes, use Terraform and Ansible where they fit, and run
from GitHub Actions, from plain Python scripts (for GitLab or a laptop), on a
self-hosted GPU runner, or directly in a cloud. It also asks for a check of
the research escalation ladder against current best practice. The issue text
is saved in [data/issue-23.txt](data/issue-23.txt).

## Read in this order

| Page | What it answers |
| --- | --- |
| [requirements.md](requirements.md) | Every requirement of the issue, the solution plan for each, the existing components it builds on, what was done and how it was verified |
| [solution-options.md](solution-options.md) | The options compared for each requirement (existing tools and libraries) and why each was chosen or rejected |
| [research-notes/cloud-gpu-landscape.md](research-notes/cloud-gpu-landscape.md) | Online research: GPU machines, prices, quotas, images, drivers, Terraform providers, CI logins, runner tools, Kubernetes GPU support, with sources |
| [ladder-review.md](ladder-review.md) | Every claim of the research escalation ladder checked against its primary source, and the corrections |
| [data/](data/) | Raw price tables, registry answers, release lists and API notes, each with its source URL and date |
| [docs/cloud-gpu.md](../../cloud-gpu.md) | The user guide: commands, cloud settings, GitHub and GitLab setup, safety layers, limits |

## Main findings

1. **Only three of the four providers rent GPUs by the hour.** AWS, Azure
   and Google Cloud create and delete GPU VMs through their APIs and bill
   by the second or minute. Hetzner Cloud has no GPU server types. Hetzner's
   GPUs are dedicated GEX servers, ordered through the Robot API with a
   one-time setup fee (209 EUR for GEX45, 999 EUR for GEX63). A single test
   on Hetzner therefore costs more than a day on AWS. A GEX server only pays
   off when it is kept for a month as a self-hosted runner
   ([landscape §1–§2](research-notes/cloud-gpu-landscape.md#1-summary)).
2. **The cheapest machines that fit are an L4 (24 GB) for MedGemma and an
   L40S or A100 (48 to 80 GB) for UniMedVL.** On demand, that is about
   $0.71 to $0.98 per hour for MedGemma (g2-standard-4 on Google Cloud,
   g6.xlarge and g6.2xlarge on AWS) and $2.24 to $5.07 for UniMedVL. A
   90-minute test costs a few dollars
   ([landscape §3–§5](research-notes/cloud-gpu-landscape.md#3-aws-ec2)).
3. **New accounts have no GPU quota.** AWS gives 0 vCPU for G and P
   instances, and Google Cloud and Azure need a quota request first. This
   is a manual, one-time step that no template can skip.
4. **The images only need an R580 or newer NVIDIA driver**, because
   llama.cpp and PyTorch bring their own CUDA libraries. Every cloud's GPU
   image or driver installer (DLAMI, Azure's NVIDIA driver extension, DLVM,
   AKS and GKE driver installation, the EKS NVIDIA AMI)
   meets that today.
5. **No existing tool covers all four providers and both runner types.**
   SkyPilot and dstack have no Hetzner backend and add a server or a large
   dependency. machulav/ec2-github-runner and RunsOn are AWS only.
   hcloud-github-runner has no GPUs, and actions-runner-controller needs a
   cluster that runs all the time. So the PR adds a small orchestrator
   (`python -m scripts.cloud_gpu`, standard library only) on top of the
   components every cloud already supports: Terraform with the official
   providers, cloud-init, the clouds' CLIs, the GitHub just-in-time runner
   API and the Hetzner Robot API
   ([solution-options.md](solution-options.md)).
6. **A machine must never be forgotten.** Each layer covers a different
   failure:
   - `down` runs after every run, including failures and cancellations.
   - Every VM powers itself off after its time to live (`shutdown -P`;
     AWS terminates the instance, and Google Cloud's `max_run_duration`
     deletes it).
   - Every resource carries `expires-at` tags, and a daily reaper deletes
     expired ones.
   - `plan` prints the worst-case cost before anything is created.
7. **CI logins need no stored cloud keys.** GitHub and GitLab both issue
   OIDC tokens that AWS, Azure and Google Cloud accept. Hetzner Robot has no
   OIDC and needs a user and password as secrets.
8. **The escalation ladder was mostly right.** Of the claims checked, 11
   were wrong, outdated or imprecise, 4 lacked an important detail, and 3
   needed a primary source or context. Eight practices were missing. All of
   them are fixed in the issue #1 case study ([ladder-review.md](ladder-review.md)).

## What is not verified

No run against a real cloud account was made: it needs accounts, GPU quota
and money, and the repository has no cloud settings yet. What was verified
instead:

- All six Terraform modules pass `terraform validate` and TFLint.
- The Ansible playbooks pass `ansible-lint`.
- The Kubernetes manifests pass `kubeconform`.
- The scripts pass ShellCheck (`scripts/check-infra.sh`).
- The orchestrator, the reaper, the Robot and GitHub API calls, the GitLab
  logins and both CI templates are unit-tested with stub tools and stub
  HTTP servers.
- The whole Kubernetes Job flow runs on a kind cluster in CI, with UniMedVL's
  echo engine and no GPU.
- The GitLab template's tool installer and checks ran in the template's
  default image (`experiments/issue23_gitlab_image_tools.sh`).

[requirements.md](requirements.md#blocked-or-unverified) lists each item that
is still open.
