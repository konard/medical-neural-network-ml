# Cloud GPU landscape for temporary GPU test machines (issue #23)

Research notes for [issue #23](https://github.com/konard/medical-neural-network-ml/issues/23):
CI/CD templates that create a GPU machine on Hetzner, AWS, Azure or Google
Cloud, run the model tests, and delete the machine.

- **Date.** All sources were read on **2026-10-19**, unless a row says
  otherwise. Every "Source" column and link below was read on that date.
- **Raw data.** Price tables, registry responses and release lists are in
  [`../data/`](../data/). Each file names its source URL and retrieval date
  (section 14).
- **Not verified.** Things that could not be checked are marked
  **(not verified)** and are also listed in section 13.

What the repository needs (from `.github/workflows/model-tests.yml` and
[`../../issue-1/ci-and-gpu-runner.md`](../../issue-1/ci-and-gpu-runner.md)):

| Workload | Runtime | Minimum GPU |
|---|---|---|
| MedGemma 4B / 27B Q4_K_M (GGUF) | llama.cpp b11738, CUDA 13.4 build with its own `cudart` | 24 GB card for 27B (about 18 GB working set). Turing or newer. Driver R580 or newer. |
| UniMedVL bf16 | PyTorch 2.14.1 wheels with the CUDA 13.0 runtime | 40 to 80 GB. Turing or newer. Driver R580 or newer. |

Both runtimes bring their own CUDA user-space libraries. So the image only has
to supply an **NVIDIA kernel driver of the R580 branch or newer**. The CUDA
toolkit version that comes with an image does not matter.

## 1. Summary

| | Hetzner | AWS EC2 | Azure VMs | Google Compute Engine |
|---|---|---|---|---|
| GPU in the cloud API | **No.** hcloud has no GPU server types. GPUs exist only as dedicated GEX servers (Robot). | Yes | Yes | Yes |
| Cheapest fit for 24 GB | GEX45, RTX PRO 4000 Blackwell 24 GB: €0.3429/h plus **€209 setup fee** | g6.xlarge, L4: **$0.8048/h** | NC24lds_xl_RTXPRO6000BSE_v6 (1/4 RTX PRO 6000, 24 GB, vGPU): $1.13/h. Full GPU: NV36ads_A10_v5, $3.20/h | g2-standard-4, L4: **$0.7068/h** (Spot $0.4230) |
| Cheapest fit for 40 to 80 GB | GEX63, RTX PRO 6000 96 GB: €1.6009/h plus **€999 setup fee** | g6e.xlarge or g6e.2xlarge, L40S 48 GB: $1.861 or $2.242/h. 80 GB or more: g7e.2xlarge (96 GB), $3.363/h | NC72lds_xl RTX PRO (48 GB): $2.44/h. 80 GB: NC24ads_A100_v4, **$3.673/h** (Spot $0.679) | a2-highgpu-1g, A100 40 GB: $3.673/h. 48 GB: g4-standard-24, Spot $0.956/h. 80 GB: a2-ultragpu-1g, $5.069/h |
| Default GPU quota | No quota. Orders may be checked by hand. | **0 vCPU** for G/VT and for P, on-demand and Spot | 0 per GPU family. This comes from community answers; the official default was not found **(not verified)**. | Free trial: no GPUs. New projects must request a regional GPU quota and the global "GPUs (all regions)" quota. The exact default value is **not verified**. |
| GPU-ready image, driver | Ubuntu 24.04 plus driver from the Ubuntu archive (580.178.04) or the CUDA repo (595) | DLAMI Base OSS Nvidia Driver GPU Ubuntu 24.04: **595.91.07** | azhpc `ubuntu-hpc:2404`: **580.178.04**. Driver extension: 615 (CUDA path) or 595.91.07 (GRID). | DLVM `common-cu129-ubuntu-2404-nvidia-580`. COS with `cos-extensions`: **580.178.04** by default. |
| Terraform resource | `hcloud_server` (no GPUs). Robot ordering only through a community provider (`gen0sec/hetzner-robot`, v0.1.0). | `aws_instance` | `azurerm_linux_virtual_machine` | `google_compute_instance` with `guest_accelerator` (N1) or an A2/G2/G4 machine type |
| CI login without stored keys | No. API token only (Robot: user and password). | OIDC with `aws-actions/configure-aws-credentials` v6.3.0 | OIDC with `azure/login` v3.1.0 | WIF with `google-github-actions/auth` v3.0.0 |
| Managed Kubernetes with GPU | None. Use k3s (hetzner-k3s, kube-hetzner) plus a Robot node. | EKS (AL2023 NVIDIA AMI, Bottlerocket, Auto Mode, Karpenter) | AKS (`--gpu-driver`, GPU Operator) | GKE (`gpu-driver-version`, Autopilot) |

What this means for "temporary" machines:

- **AWS, Azure, GCP.** A run is create, test, delete, billed per second or per
  minute.
- **Hetzner.** A one-off GEX order costs the setup fee every time. A test
  run on GEX45 therefore costs at least about €209 + €0.34. It is cheaper to rent
  one GEX server and keep it as a long-lived, self-hosted runner with
  ephemeral jobs. The monthly cap is €214 for GEX45 and €999 for GEX63.

## 2. Hetzner

### 2.1 Hetzner Cloud (hcloud API): no GPU types

| Check | Result | Source |
|---|---|---|
| OpenAPI spec | No "gpu" or "nvidia" anywhere; server types have no GPU field | https://docs.hetzner.cloud/cloud.spec.json |
| API changelog Nov 2025 to Oct 2026 | No GPU entries; new types are only cx23–53 and cpx12–62 | https://docs.hetzner.cloud/changelog |
| Product page | Only CPU plans; the one GPU link points to dedicated servers | https://www.hetzner.com/cloud/ |
| Managed Kubernetes | None found | the spec and changelog above |

So every Hetzner Cloud tool (hcloud Terraform provider, `hetzner.hcloud`
Ansible collection, Cyclenerd/TestFlows runners, the GitLab fleeting plugin) can
only create **CPU** machines.

### 2.2 Dedicated GPU servers (GEX line, Robot)

Sources:
- Matrix and product pages: https://www.hetzner.com/dedicated-rootserver/matrix-gpu/, https://www.hetzner.com/dedicated-rootserver/gex45/, `/gex63/`, `/gex131/`.
- Public price API: `https://website-price-api.hetzner.com/api/v1/products/ROBOT_1768%2BROBOT_1266`.
- GEX45 press release, 2026-09-01: https://www.hetzner.com/pressroom/hetzner-expands-its-gpu-portfolio-with-the-gex45/

Prices exclude VAT. They include a Primary IPv4 address, which is €1.70/month
on its own.

| Model | GPU (VRAM) | CPU / RAM / disk | €/month | €/hour | Setup fee | Locations |
|---|---|---|---|---|---|---|
| GEX45-1 | RTX PRO 4000 Blackwell SFF (24 GB GDDR7 ECC) | i5-13500 / 64 GB DDR4 / 2×512 GB NVMe | 214.00 | 0.3429 | 209.00 | FSN1, HEL1 |
| GEX63-1 | RTX PRO 6000 Blackwell Max-Q (96 GB) | Core Ultra 7 265 / 128 GB DDR5 / 2×1.92 TB NVMe | 999.00 | 1.6009 | 999.00 | HEL1 |
| GEX131-1 | RTX PRO 6000 Blackwell Max-Q (96 GB) | Xeon Gold 5412U / 256 GB DDR5 ECC / 2×960 GB | 1499.00 | 2.4022 | 1499.00 | HEL1, FSN1 |
| GEX131-2 / -3 | same | 512 / 768 GB RAM | 2097.30 / 2697.30 | 3.3611 / 4.3226 | 2099 / 2699 | FSN1, HEL1, NBG1 |

- **Older models.** GEX44 (RTX 4000 SFF Ada, 20 GB) and GEX130 (RTX 6000 Ada,
  48 GB) are no longer in the matrix; `/gex44/` redirects to `/gex45/`.
- **Generation.** All of these GPUs are Ada or Blackwell, so Turing or newer is
  met.
- **Terms.** The product pages say "No minimum contract term" and
  "Cancellation period: immediately".
- **Billing** (https://docs.hetzner.com/general/billing-and-account-management/billing-at-hetzner/billing-system-hetzner/, changed 2026-10-01):
  - Hourly price; partial hours are rounded up.
  - "The total cost is never more than the monthly price."
  - Billing starts when access is given.
- **Cancellation** (https://docs.hetzner.com/general/billing-and-account-management/cancellation/cancellations-robot):
  "You can only cancel immediately. We bill by the hour and never exceed
  the monthly price cap."
- **Fixed hardware.** No RAM, disk or GPU upgrades on GEX
  (https://docs.hetzner.com/robot/dedicated-server/server-lines/gpu-server/).

**Server Auction** (https://www.hetzner.com/sb/; data from
`https://www.hetzner.com/_resources/app/data/app/live_data_sb.json`). All
auction servers have no setup fee. GPU servers seen on 2026-10-19:

| ID | GPU | €/month | €/hour | DC |
|---|---|---|---|---|
| 3088260 | RTX 4000 SFF Ada (20 GB) | 233 | 0.3734 | FSN1 |
| 3096250 | RTX 6000 Ada (48 GB) | 926 | 1.484 | NBG1 |
| 3096212 | RTX 6000 Ada (48 GB) | 943 | 1.5112 | FSN1 |

Auction stock changes all the time. A template cannot rely on one ID.

### 2.3 Robot Webservice API

Source: https://robot.hetzner.com/doc/webservice/en.html

| Topic | Fact |
|---|---|
| Base URL and auth | `https://robot-ws.your-server.de`, HTTP Basic auth with a separate web service user. 3 failed logins block the IP for 10 min. |
| Turn ordering on | Robot → Administration → Settings → Web Service Settings → Ordering |
| List products | `GET /order/server/product` (shows hourly, monthly and setup prices) |
| Order | `POST /order/server/transaction` with `product_id`, `authorized_key[]` or `password`, `location`, `dist`, `addon[]`, `test`. Limit: 20 requests/day. Status goes "in process" → "ready". Can fail with 412 "Please place your order via the web frontend". |
| Order from auction | `POST /order/server_market/transaction` |
| Cancel | `POST /server/{server-number}/cancellation` with `cancellation_date=now`. Limit: 200 requests/hour. `DELETE` withdraws a cancellation. |
| IPv4 | Servers are IPv6-only by default; add `addon[]=primary_ipv4` |
| GEX via API | **Not verified.** The docs never mention GPUs, and listing products needs credentials. |

Consequences for templates:

- **Tooling.** Ordering through the API is possible in principle. No
  official Terraform resource or Ansible module does it (sections 2.4, 2.5).
- **Provisioning time.** It is unknown, because GEX orders may be checked by hand
  **(not verified)**.
- **Cost.** Each order costs the setup fee, so "one server per run" is
  not cheap on Hetzner.
- **Recommended Hetzner flow.** Order one GEX45 (or GEX63) once. Install
  Ubuntu 24.04 and the driver with Ansible. Register an ephemeral or JIT
  GitHub runner on it, or run the tests over SSH. Cancel through the API when it
  is no longer needed.

### 2.4 Terraform and Ansible for Hetzner

| Tool | Version (date) | Can create GPU machines? | Source |
|---|---|---|---|
| `hetznercloud/hcloud` provider (`hcloud_server`, `hcloud_ssh_key`, `hcloud_firewall`) | 1.70.0 (2026-10-05) | No (Cloud only) | https://registry.terraform.io/v1/providers/hetznercloud/hcloud |
| `gen0sec/terraform-provider-hetzner-robot` | v0.1.0 (2026-07-13), MPL-2.0, 1 star, not found in the registry | Yes in principle: `server_order` (default `test=true`), `server_market_order`, `server_cancellation`. Destroying the order resource does **not** cancel the server. | https://github.com/gen0sec/terraform-provider-hetzner-robot |
| Other Robot providers (DigitecGalaxus, mwudka, sophotechlabs) | 0.0.1 / 0.0.14 / v0.2.1 | No ordering (read-only, firewall, rescue, rDNS) | GitHub API, see `hetzner-robot-terraform-providers.json` |
| Ansible `hetzner.hcloud` | 7.2.0 (2026-10-05) | No (Cloud only) | https://galaxy.ansible.com/ui/repo/published/hetzner/hcloud/ |
| Ansible `community.hrobot` | 2.8.0 (2026-10-04) | No order or cancel module. Has boot/rescue, reset, firewall, SSH keys and an inventory plugin. | https://github.com/ansible-collections/community.hrobot |

### 2.5 OS, driver, Kubernetes, auth on Hetzner

- **OS.** Ubuntu 22.04, 24.04 and 26.04 are offered in the GEX configurator.
  Reinstall with `installimage` from the Rescue System
  (https://docs.hetzner.com/robot/dedicated-server/operating-systems/installimage/).
- **Driver.** Hetzner's MIG tutorial (2026-05-28,
  https://community.hetzner.com/tutorials/gpu-mig) uses Ubuntu 24.04 with
  `cuda-toolkit-13-2` and `nvidia-open`, which installs driver 595.58.03.
  - Ubuntu 24.04's own archive has `nvidia-graphics-drivers-580-server`
    580.178.04 (Launchpad, published 2026-09-15; `ubuntu-noble-nvidia-580-server.json`).
  - Both are R580 or newer. Blackwell GPUs need the open kernel modules.
- **Kubernetes.** Hetzner has no managed Kubernetes. Options:
  - [vitobotta/hetzner-k3s](https://github.com/vitobotta/hetzner-k3s) v2.6.0 (2026-06-24), MIT. Can add Robot servers as "external" nodes, but not together with a Hetzner private network.
  - [kube-hetzner/terraform-hcloud-kube-hetzner](https://github.com/kube-hetzner/terraform-hcloud-kube-hetzner) v3.2.1 (2026-09-09), MIT. Robot nodes through hcloud-cloud-controller-manager plus a vSwitch. No NVIDIA content.
  - On a GEX node, install the NVIDIA GPU Operator (section 11). There is no Hetzner doc for this **(not verified)**.
- **Auth.**
  - hcloud uses a project API token (Read or Read & Write, `Authorization: Bearer`): https://docs.hetzner.com/cloud/api/getting-started/generating-api-token/
  - Robot uses a user and password.
  - There is no OIDC on either API, so CI must store the token as a secret.

## 3. AWS EC2

### 3.1 GPU instances and prices (us-east-1, Linux, on-demand)

Sources:
- Specs: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
- Prices: AWS public price file `https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json` (published 2026-10-15)
- Spot savings: https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json

| Instance | GPU | GPU memory | vCPU / RAM | $/h | Spot savings (advisor) |
|---|---|---|---|---|---|
| g4dn.xlarge | T4 (Turing) | 16 GiB | 4 / 16 GiB | 0.526 | 60% |
| **g6.xlarge** | L4 | 22 GiB (24 GB nominal) | 4 / 16 GiB | **0.8048** | 49% |
| g5.xlarge | A10G | 22 GiB (24 GB nominal) | 4 / 16 GiB | 1.006 | 56% |
| gr6.4xlarge | L4 | 22 GiB | 16 / 128 GiB | 1.5392 | 41% |
| g6e.xlarge | L40S | 44 GiB | 4 / 32 GiB | 1.861 | 50% |
| **g6e.2xlarge** | L40S | 44 GiB | 8 / 64 GiB | **2.24208** | 54% |
| g7.2xlarge | RTX PRO 4500 | 32 GiB | 8 / 32 GiB | 2.52 | – |
| **g7e.2xlarge** | RTX PRO Server 6000 | 96 GiB | 8 / 64 GiB | **3.36312** | 68% |
| p5.4xlarge | H100 | 80 GiB | 16 / 256 GiB | 6.88 | 9% |
| p4d.24xlarge | 8× A100 40 GB | 320 GiB | 96 / 1152 GiB | 21.96 | 56% |

- **Generation.** All of these GPUs are Turing or newer.
- **MedGemma 27B Q4.** g6.xlarge fits. Its 16 GiB of system RAM is enough
  because the weights go to VRAM.
- **UniMedVL bf16.**
  - g6e.xlarge has only 32 GiB of system RAM, which is tight for loading
    bf16 weights. g6e.2xlarge is safer.
  - g7e.2xlarge is the cheapest single GPU with 80 GB or more.
- **Spot prices.** Live spot prices need credentials
  (`aws ec2 describe-spot-price-history`) and were **not verified**.
  Vantage shows, without saying which statistic it uses: g6.xlarge $0.645,
  g6e.xlarge $1.696, g7e.2xlarge $2.664
  (https://instances.vantage.sh/aws/ec2/g6.xlarge).
- **Capacity Blocks.** They exist only for P types in us-east-1, are booked
  1–14 days or more ahead and cannot be cancelled. Not useful for short CI runs
  (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html).

### 3.2 Image and driver

| Image | Driver | SSM parameter | Source |
|---|---|---|---|
| DLAMI Base OSS Nvidia Driver GPU **Ubuntu 24.04**, release 20261006 | **595.91.07**; CUDA 12.8/12.9/13.0/13.2 (default 13.2); container toolkit 1.20.1 | `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id` | https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubaseoss-ul2404-2026-10-08.html |
| Same, Amazon Linux 2023, release 20261009 | 595.91.07 | `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id` | https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubaseoss-al2023-2026-10-12.html |
| EKS AL2023 NVIDIA AMI v20261012 | 580.178.04 by default; 595.91.07 chosen for g7 at first boot | `/aws/service/eks/optimized-ami/<k8s>/amazon-linux-2023/x86_64/nvidia/recommended/image_id` | https://github.com/awslabs/amazon-eks-ami/releases/tag/v20261012 |
| Bottlerocket NVIDIA variants v1.67.0 | R580 for K8s 1.34–1.36 (patch version **not verified**); includes device plugin | – | https://github.com/bottlerocket-os/bottlerocket/releases |

In Terraform, use `ami = "resolve:ssm:/aws/service/deeplearning/ami/..."` or a
`data "aws_ssm_parameter"` block
(https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami-parameter-store.html).

### 3.3 Quotas

Source: https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html.
The values count vCPUs.

| Quota | Code | Default |
|---|---|---|
| Running On-Demand G and VT instances | L-DB2E81BA | **0** |
| Running On-Demand P instances | L-417A185B | **0** |
| All G and VT Spot Instance Requests | L-3819A6DF | 0 |
| All P Spot Instance Requests | L-7212CCBC | 0 |

- **Increase.**
  `aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8`
  (https://docs.aws.amazon.com/cli/latest/reference/service-quotas/request-service-quota-increase.html).
- **Size.** g6e.2xlarge and g7e.2xlarge need 8 G vCPUs each, and g6.xlarge needs 4.
- **Guard.** A small quota also works as a hard spending limit.

### 3.4 Terraform and API

| Item | Value | Source |
|---|---|---|
| Provider | `hashicorp/aws` 6.69.1 (2026-10-15) | https://registry.terraform.io/v1/providers/hashicorp/aws |
| Resource | `aws_instance` (or `aws_launch_template`) | https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/instance |
| Spot | `instance_market_options { market_type = "spot" spot_options { ... } }` | same |
| Self-delete on shutdown | `instance_initiated_shutdown_behavior = "terminate"` (AWS default is `stop`) | https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html |
| CLI | `aws ec2 run-instances --instance-initiated-shutdown-behavior terminate` | https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html |

### 3.5 EKS

- **AL2023 NVIDIA AMI.** It does **not** include the NVIDIA device plugin;
  install it yourself. AWS docs state that driver 580 is needed for CUDA 13
  (https://docs.aws.amazon.com/eks/latest/userguide/ml-eks-optimized-ami.html).
- **EKS Auto Mode.** Driver and device plugin are built in.
  - The AWS example NodePool uses taint `nvidia.com/gpu:NoSchedule` and label
    `eks.amazonaws.com/instance-gpu-name: l40s`
    (https://docs.aws.amazon.com/eks/latest/userguide/auto-accelerated.html).
  - The Auto Mode driver version is **not verified**; the docs sample shows an
    old 535.
- **Karpenter.** Latest release is v1.14.2 (2026-10-19, GitHub releases).
  - It needs the device plugin DaemonSet before it counts a GPU node as ready.
  - GPU labels are `karpenter.k8s.aws/instance-gpu-name|count|memory`
    (https://karpenter.sh/docs/concepts/scheduling/).

## 4. Azure VMs

### 4.1 GPU sizes and prices (eastus, Linux)

Sources:
- Sizes: https://learn.microsoft.com/azure/virtual-machines/sizes/overview and the series pages linked below.
- Prices: Azure Retail Prices API `https://prices.azure.com/api/retail/prices` (`azure-retail-prices-eastus*.json`).

| Size | GPU | VRAM | vCPU / RAM | On-demand $/h | Spot $/h |
|---|---|---|---|---|---|
| [NC4as_T4_v3](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series) | T4 (Turing) | 16 GB | 4 / 28 GiB | 0.526 | 0.149174 |
| [NC24lds_xl_RTXPRO6000BSE_v6](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nc-rtxpro6000-bse-v6-series) | 1/4 RTX PRO 6000 Blackwell (vGPU) | 24 GB | 24 / 72 GiB | **1.13** | 0.208824 |
| [NV36ads_A10_v5](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nvadsa10v5-series) | A10, full GPU (GRID driver) | 24 GB | 36 / 440 GiB | 3.20 | 0.59136 |
| NC72lds_xl_RTXPRO6000BSE_v6 | 1/2 RTX PRO 6000 (vGPU) | 48 GB | 72 / – | **2.44** | 0.450912 |
| [NC24ads_A100_v4](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series) | A100 PCIe | 80 GB | 24 / 220 GiB | **3.673** | 0.67877 |
| NC144lds_xl_RTXPRO6000BSE_v6 | RTX PRO 6000, full GPU | 96 GB | 144 / – | 5.50 | 1.0164 |
| [NC40ads_H100_v5](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/ncadsh100v5-series) | H100 NVL | 94 GB | 40 / 320 GiB | 6.98 | 1.289904 |

- **Not usable.**
  - NCv3/NDv2 (V100, Volta) and NVv3 (M60) are pre-Turing.
  - NVv4, NGads V620 and ND MI300X are AMD.
  - NCv3 retired on 2025-09-30; NVv3 and NVv4 retire on 2026-09-30
    (https://learn.microsoft.com/azure/virtual-machines/sizes/lifecycle/retirements-and-capacity-restrictions).
- **No L4 or L40S sizes** appear in the Retail Prices API.
- **vGPU caveat.** The fractional RTX PRO 6000 sizes and the A10 sizes are
  vGPU and need the GRID driver. That CUDA 13 runs inside these vGPU profiles
  is **not verified**. The full-GPU A100 or H100 sizes avoid this question.

### 4.2 Image and driver

| Option | Driver | Source |
|---|---|---|
| NVIDIA GPU Driver Extension (`Microsoft.HpcCompute` / `NvidiaGpuDriverLinux`) | Driver map: CUDA path uses branch **615** `nvidia-open` (T4, A100, H100); GRID path uses **595.91.07** by default (A10, RTX PRO); 580.178.04 is also allowed. No Secure Boot. | https://learn.microsoft.com/azure/virtual-machines/extensions/hpccompute-gpu-linux, https://raw.githubusercontent.com/Azure/azhpc-extensions/master/NvidiaGPU/Nvidia-GPU-Linux-Resources.json |
| Azure HPC image `microsoft-dsvm:ubuntu-hpc:2404:24.04.2026092501` | **580.178.04**, CUDA 13.0.88 | https://github.com/Azure/azhpc-images/releases |
| RTX preview image `azure-hpc:ubuntu-hpc-test:2404-rtx:24.04.2026082001` | GRID 595.91.07, CUDA 13.2.86 | same |
| Manual install guide | A10 needs GRID v19.6 (580.178.04) or v20.2 (595.91.07); Secure Boot and vTPM must be off for GRID | https://learn.microsoft.com/azure/virtual-machines/linux/n-series-driver-setup |
| Data Science VM | Last release 2025-05-20, driver 535: too old for CUDA 13 | https://learn.microsoft.com/azure/machine-learning/data-science-virtual-machine/release-notes |

### 4.3 Quotas

- **Structure.** Two tiers: Total Regional vCPUs plus a per-family vCPU limit.
  Deallocated VMs still count
  (https://learn.microsoft.com/azure/virtual-machines/quotas).
- **Default.** The N-series default of 0 comes from a Microsoft Q&A answer,
  not official docs **(not verified)**
  (https://learn.microsoft.com/en-us/answers/questions/2276490/insufficient-vcpu-quota-requested-6-remaining-0-fo).
- **Family names** (confirm with `az vm list-skus`): `StandardNCASv3_T4Family`,
  `StandardNVADSA10v5Family`, `StandardNCADSA100v4Family`,
  `StandardNCadsH100v5Family`, `StandardNCRTXPRO6000BSEv6Family`.
- **Increase.**
  - Portal: Quotas → per-VM-family request
    (https://learn.microsoft.com/azure/quotas/per-vm-quota-requests).
  - CLI: `az quota update ...` (https://learn.microsoft.com/cli/azure/quota).
  - REST: `PUT .../providers/Microsoft.Quota/quotas/{family}` (api-version 2025-09-01).
- **Spot.** One shared "Total Regional Spot vCPUs" pool
  (https://learn.microsoft.com/azure/quotas/spot-quota).

### 4.4 Terraform

| Item | Value | Source |
|---|---|---|
| Provider | `hashicorp/azurerm` 5.10.0 (2026-10-15); `azure/azapi` 2.13.0 (2026-09-28) | https://registry.terraform.io/v1/providers/hashicorp/azurerm |
| VM | `azurerm_linux_virtual_machine` with `priority = "Spot"`, `eviction_policy = "Delete"`, `max_bid_price = -1` | https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/linux_virtual_machine |
| Driver | `azurerm_virtual_machine_extension` (publisher `Microsoft.HpcCompute`, type `NvidiaGpuDriverLinux`) | same registry |
| Auto-shutdown | `azurerm_dev_test_global_vm_shutdown_schedule` (daily at a fixed time, not a TTL) | same registry |
| AKS pool | `azurerm_kubernetes_cluster_node_pool` with `gpu_driver = "Install" \| "None"` | same registry |

### 4.5 AKS

Source: https://learn.microsoft.com/azure/aks/use-nvidia-gpu (ms.date 2026-07-02).

- **Driver install.** `az aks nodepool add ... --node-vm-size Standard_NC4as_T4_v3 --gpu-driver install|none`
  needs Azure CLI 2.72.2 or newer.
  - `--skip-gpu-driver-install` stopped working on 2025-08-14
    (https://github.com/Azure/AKS/issues/4925).
- **Taint.** The docs use `--node-taints sku=gpu:NoSchedule`. This is a
  convention; AKS does not add it automatically.
- **Device plugin.** With the AKS-installed driver you deploy the NVIDIA device
  plugin DaemonSet yourself.
- **GPU Operator.** It needs `--gpu-driver none`
  (https://learn.microsoft.com/azure/aks/nvidia-gpu-operator).
- **Fully managed GPU pools.** In preview: `--enable-managed-gpu`
  (https://learn.microsoft.com/azure/aks/aks-managed-gpu-nodes).
- **Node image drivers.** AgentBaker lists 580.159.04 (CUDA LTS), 580.178.04
  (GRID) and 595.91.07 (GRID v20), with device plugin 0.19.3
  (https://raw.githubusercontent.com/Azure/AgentBaker/master/parts/common/components.json,
  master branch; released node images may lag).

## 5. Google Compute Engine and GKE

### 5.1 GPU machine types and prices (us-central1)

Sources:
- Specs: https://cloud.google.com/compute/docs/gpus and https://cloud.google.com/compute/docs/accelerator-optimized-machines
- Prices: https://cloud.google.com/products/compute/pricing/accelerator-optimized, https://cloud.google.com/spot-vms/pricing

| Machine type | GPU | VRAM | vCPU / RAM | On-demand $/h | Spot $/h |
|---|---|---|---|---|---|
| n1-standard-4 + 1× `nvidia-tesla-t4` | T4 | 16 GB | 4 / 15 GB | 0.540 | 0.306 |
| **g2-standard-4** | L4 | 24 GB | 4 / 16 GB | **0.7068** | **0.4230** |
| g2-standard-8 | L4 | 24 GB | 8 / 32 GB | 0.8536 | 0.5108 |
| g4-standard-12 | 1/4 RTX PRO 6000 (vGPU) | 24 GB | 12 / 45 GB | 1.2938 | 0.4782 |
| g4-standard-24 | 1/2 RTX PRO 6000 (vGPU) | 48 GB | 24 / 90 GB | 2.5875 | **0.9564** |
| g4-standard-48 | RTX PRO 6000 | 96 GB | 48 / 180 GB | 4.4999 | 1.6635 |
| **a2-highgpu-1g** | A100 | 40 GB | 12 / 85 GB | **3.6734** | 2.2040 |
| a2-ultragpu-1g | A100 | 80 GB | 12 / 170 GB | 5.0688 | 3.0400 |
| a3-highgpu-1g | H100 | 80 GB | 26 / 234 GB | Spot or Flex-start only | 6.2739 |

- **Pre-Turing.** P4, P100 and V100 cannot be used with CUDA 13.
- **T4 end of support.** T4 support ends 2027-08-01
  (https://cloud.google.com/compute/docs/gpus).
- **G4 fractional shapes.** They need driver 580.126.09 or newer, a service
  account and a Hyperdisk Balanced boot disk (same source).
- **Spot prices** can change once a day.

### 5.2 Images and driver

| Option | Driver | Source |
|---|---|---|
| Google recommendation | R580 LTSB (support until June 2028), minimum 580.95.05; R595 is also verified | https://cloud.google.com/compute/docs/gpus/install-drivers-gpu |
| Installer script | `cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=lts` | same |
| Deep Learning VM, project `deeplearning-platform-release` | Families `common-cu129-ubuntu-2404-nvidia-580` and `pytorch-2-9-cu129-ubuntu-2404-nvidia-580`. The 570 images were deprecated on 2026-04-13. No cu13x family exists, which is fine here. Metadata `install-nvidia-driver=True`. | https://cloud.google.com/deep-learning-vm/docs/images |
| Ubuntu accelerator images, project `ubuntu-os-accelerator-images` | `ubuntu-accelerator-2404-amd64-with-nvidia-580`. Officially listed only for A4X, A4 and A3 Ultra; use on G2, G4 or A2 is **not verified**. | https://cloud.google.com/compute/docs/images/os-details |
| Container-Optimized OS | `sudo cos-extensions install gpu -- -version=latest`. COS 133: default = latest = **580.178.04**; R595 = 595.91.07. | https://cloud.google.com/container-optimized-os/docs/how-to/run-gpus, https://storage.googleapis.com/cos-tools/19999.44.126/lakitu/gpu_driver_versions.textproto |

### 5.3 Quotas

Source: https://cloud.google.com/compute/resource-usage

- **Two quotas.** You need a regional GPU quota and the global
  "GPUs (all regions)" quota.
- **Regional quota names.** `GPU_FAMILY:NVIDIA_L4` (G2), `GPU_FAMILY:NVIDIA_RTX_PRO_6000` (G4),
  `NVIDIA_A100_GPUS` / `NVIDIA_A100_80GB_GPUS` (A2), and `PREEMPTIBLE_*` for Spot.
  G2, G4 and A3 need no separate CPU quota.
- **Free trial accounts** get no GPU quota
  (https://cloud.google.com/container-optimized-os/docs/how-to/run-gpus).
- **Default value.** The exact default of the global quota is **not verified**.
  It is commonly 0, so plan for a request.
- **Increase.**
  `gcloud beta quotas preferences create --service=compute.googleapis.com --quota-id=GPUS-PER-GPU-FAMILY-per-project-region --preferred-value=1 --dimensions=region=us-central1,gpu_family=NVIDIA_L4`
  (https://cloud.google.com/sdk/gcloud/reference/beta/quotas/preferences/create).
- **Using the Spot quota.** A standard VM that is auto-deleted by
  `max_run_duration` (7 days or less) can use the preemptible quota
  (https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus#preemptible-quotas).

### 5.4 Terraform

Sources: https://registry.terraform.io/v1/providers/hashicorp/google and
https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_instance.

| Item | Value |
|---|---|
| Provider | `hashicorp/google` 8.7.0 (2026-10-13) |
| Resource | `google_compute_instance`. On N1, add a `guest_accelerator { type = "nvidia-tesla-t4", count = 1 }` block; A2, G2 and G4 machine types include the GPU. |
| Required | `scheduling { on_host_maintenance = "TERMINATE" }` |
| Spot | `provisioning_model = "SPOT"`, `preemptible = true`, `automatic_restart = false` |
| TTL | `instance_termination_action = "DELETE"` with `max_run_duration { seconds = N }`. This is **GA** in the `google` provider, not only in `google-beta`. |

### 5.5 GKE

- **Node pools** (https://cloud.google.com/kubernetes-engine/docs/how-to/gpus):
  - `gcloud container node-pools create POOL --machine-type g2-standard-4 --accelerator type=nvidia-l4,count=1,gpu-driver-version=default|latest|disabled`
  - `latest` works on COS only.
- **Driver.** R580 is the default on 1.32.8-gke.1170000+, 1.33.4-gke.1245000+ and
  1.34.0-gke.1662000+ (https://cloud.google.com/kubernetes-engine/docs/troubleshooting/gpus).
- **Taint.** GKE adds `nvidia.com/gpu=present:NoSchedule` automatically, but only when
  the cluster also has a non-GPU node pool. The matching toleration is added by the
  ExtendedResourceToleration admission controller.
- **Autopilot** (https://cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus):
  - Use `nodeSelector: cloud.google.com/gke-accelerator: nvidia-l4` (or
    `nvidia-a100-80gb`, `nvidia-rtx-pro-6000`, ...) plus
    `resources.limits: nvidia.com/gpu`.
  - The driver is automatic.

## 6. Authentication from CI

| Cloud | GitHub Actions | GitLab CI | Source |
|---|---|---|---|
| AWS | `permissions: id-token: write`; `aws-actions/configure-aws-credentials` **v6.3.0** (2026-09-15) with `role-to-assume`. IAM OIDC provider `token.actions.githubusercontent.com`, audience `sts.amazonaws.com`, trust `sub = repo:konard/medical-neural-network-ml:environment:gpu`. | `id_tokens: GITLAB_OIDC_TOKEN: aud: ...` then `aws sts assume-role-with-web-identity` | https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws, https://docs.gitlab.com/ci/cloud_services/aws/ |
| Azure | `azure/login` **v3.1.0** (2026-09-10) with `client-id`, `tenant-id`, `subscription-id`; federated credential issuer `https://token.actions.githubusercontent.com`, subject `repo:OWNER/REPO:environment:gpu`, audience `api://AzureADTokenExchange` | `id_tokens` then `az login --service-principal -u $AZURE_CLIENT_ID -t $AZURE_TENANT_ID --federated-token $GITLAB_OIDC_TOKEN`; subject `project_path:<group>/<project>:ref_type:branch:ref:<branch>` | https://github.com/Azure/login/releases, https://learn.microsoft.com/entra/workload-id/workload-identity-federation-create-trust, https://docs.gitlab.com/ci/cloud_services/azure/ |
| GCP | `google-github-actions/auth` **v3.0.0** (2025-08-28) with `workload_identity_provider` (direct WIF), optional `service_account`; restrict with `--attribute-condition` | `id_tokens` with aud = the WIF provider, then an `external_account` credentials file and `gcloud auth login --cred-file`. The GitLab built-in Google Cloud IAM integration (`identity: google_cloud`) is Beta. | https://github.com/google-github-actions/auth, https://cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines, https://docs.gitlab.com/ci/cloud_services/google_cloud/ |
| Hetzner | Secret `HCLOUD_TOKEN` (Robot: user and password secret) | CI/CD variable (masked, protected) | https://docs.hetzner.com/cloud/api/getting-started/generating-api-token/ |

The new `sub` format with numeric IDs
(`repo:<ORG>@<ORG_ID>/<REPO>@<REPO_ID>:...`) is shown only in the
configure-aws-credentials README. It was not checked against GitHub docs
**(not verified)**.

## 7. Existing tools for ephemeral CI runners

### 7.1 GitHub

| Tool | License | Latest (date) | What it does | Fit for this repo |
|---|---|---|---|---|
| [actions/runner](https://github.com/actions/runner) | MIT | v2.338.0 (2026-10-06) | Runner binary; `--ephemeral`, `--jitconfig` | Needed in every self-hosted flow |
| [actions-runner-controller](https://github.com/actions/actions-runner-controller) (gha-runner-scale-set) | Apache-2.0 | gha-runner-scale-set-0.15.0 (2026-10-01) | Runner scale sets on Kubernetes; GPU through the pod template | Needs a running cluster all the time. Heavy for manual runs. The GitHub App must be org-owned, so this repo needs a PAT. |
| [github-aws-runners/terraform-aws-github-runner](https://github.com/github-aws-runners/terraform-aws-github-runner) | MIT | v7.11.0 (2026-08-17) | Lambda + webhook autoscaling, Spot, scale to zero | AWS only; needs a GitHub App; many moving parts |
| [machulav/ec2-github-runner](https://github.com/machulav/ec2-github-runner) | MIT | v2.6.1 (2026-04-10) | Start job → GPU job → stop job; Spot, JIT | Simple and close to what we need, but AWS only |
| [RunsOn](https://github.com/runs-on/runs-on) | Source on GitHub (MIT); commercial use needs a paid license | v3.5.0 (2026-10-14) | Runners in your AWS account; GPU images (`ubuntu24-gpu-x64`), label `family=g6.xlarge` | AWS only; free for non-commercial use; paid €300–€3600/year (https://runs-on.com/pricing/) |
| [Cirun.io](https://cirun.io/pricing) | SaaS, source not public | – | `.cirun.yml` creates runners on AWS, GCP, Azure, Hetzner and others | Free for public repos and supports GPUs; adds a third party that holds cloud credentials |
| [Cyclenerd/hcloud-github-runner](https://github.com/Cyclenerd/hcloud-github-runner) | Apache-2.0 | v1.4.1 (2026-06-11) | Create → run → delete a Hetzner Cloud server inside one workflow | Good pattern; Hetzner Cloud only, so **no GPU** |
| [testflows/TestFlows-GitHub-Hetzner-Runners](https://github.com/testflows/TestFlows-GitHub-Hetzner-Runners) | Apache-2.0 | 1.10.260921.1182403 (2026-09-21) | Long-running autoscaler service for Hetzner Cloud | No GPU; needs a classic PAT |

No existing tool orders Hetzner dedicated (GEX) servers for a single job.

**GitHub-hosted GPU runner**
- Size: `linux_4_core_gpu`, 4 vCPU, 28 GB RAM, Tesla T4 16 GB.
- Price: $0.052/min, about $3.12/h.
- Only for organizations on GitHub Team or Enterprise Cloud. Larger runners
  are not free for public repos.
- Sources: https://docs.github.com/en/billing/reference/actions-runner-pricing,
  https://docs.github.com/en/actions/reference/runners/larger-runners
- Not usable here: the repo is owned by a user account, and 16 GB is below 24 GB.

**JIT runner API** (https://docs.github.com/en/rest/actions/self-hosted-runners):

| Call | Body | Response | Permission |
|---|---|---|---|
| `POST /repos/{owner}/{repo}/actions/runners/generate-jitconfig` | `name`, `runner_group_id` (int), `labels` (1–100), `work_folder` (optional) | 201 with `runner` and `encoded_jit_config`; start with `./run.sh --jitconfig <value>` | Fine-grained PAT or GitHub App: repository "Administration" write. Classic PAT: `repo`. |
| `POST /repos/{owner}/{repo}/actions/runners/registration-token` | – | `token`, `expires_at` (1 hour) | same |

- **`GITHUB_TOKEN` cannot call these.** Its permission list has no
  `administration` key
  (https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).
  So the workflow needs a PAT or GitHub App token stored in the `gpu`
  environment.
- **Runner group.** For repo-level runners, `runner_group_id` is usually 1
  (Default) **(not verified)**.

### 7.2 GitLab

| Item | Version / status | Source |
|---|---|---|
| GitLab Runner | v19.5.0 (2026-10-15), MIT | https://gitlab.com/gitlab-org/gitlab-runner/-/releases |
| Docker Autoscaler / Instance executor (use fleeting) | GA since Runner 17.1 | https://docs.gitlab.com/runner/executors/docker_autoscaler/ |
| fleeting plugin AWS (Auto Scaling group) | v1.3.0 (2026-07-22) | https://gitlab.com/gitlab-org/fleeting/plugins/aws |
| fleeting plugin Google Cloud (`googlecloud`, managed instance group) | v1.3.0 (2026-09-10) | https://gitlab.com/gitlab-org/fleeting/plugins/googlecloud |
| fleeting plugin Azure (VMSS Uniform only) | v1.2.0 (2026-01-06) | https://gitlab.com/gitlab-org/fleeting/plugins/azure |
| fleeting plugin Hetzner (hcloud, CPU only) | v1.4.6 (2026-09-16), MIT | https://gitlab.com/hetznercloud/fleeting-plugin-hetzner |
| GitLab-hosted GPU runner `saas-linux-medium-amd64-gpu-standard` | T4 16 GB, 4 vCPU, 15 GB RAM; Premium/Ultimate only; cost factor 7, about $4.20 per GPU hour at $10 per 1,000 minutes. Too small for both models. | https://docs.gitlab.com/ci/runners/hosted_runners/gpu_enabled/, https://docs.gitlab.com/ci/pipelines/compute_minutes/ |

## 8. Multi-cloud orchestration and GPU clouds

| Tool | License | Latest (date) | Clouds | Notes |
|---|---|---|---|---|
| [SkyPilot](https://github.com/skypilot-org/skypilot) | Apache-2.0 | v0.14.0 (2026-10-05) | AWS, GCP, Azure, Kubernetes, Lambda, RunPod, Vast.ai, Nebius, OCI and others. **No Hetzner**, but existing servers work through SSH Node Pools. | `sky launch --gpus L4:1`, auto-down with `-i N --down` (https://docs.skypilot.ai/en/latest/reference/auto-stop.html). Picks the cheapest cloud and handles quota fallback. Adds a large Python dependency. |
| [dstack](https://github.com/dstackai/dstack) | MPL-2.0 | 0.22.4 (2026-10-15) | AWS, Azure, GCP, Kubernetes, Lambda, RunPod, Vast.ai and others. No Hetzner backend; SSH fleets for existing servers. | Needs a dstack server |
| [Pulumi](https://github.com/pulumi/pulumi) | Apache-2.0 | v3.270.0 (2026-10-16) | All four (hcloud provider exists) | IaC in Python, but needs its own state backend |

GPU clouds, for comparison only ($/h; sources: https://modal.com/pricing,
https://www.runpod.io/pricing, https://lambda.ai/pricing):

| GPU | Modal | RunPod (community / secure) | Lambda |
|---|---|---|---|
| L4 24 GB | 0.80 | 0.44 / 0.59 | – |
| L40S 48 GB | 1.95 | – | – |
| A100 80 GB | 2.50 | 1.19–1.39 / 1.79 | 8× only |
| H100 80 GB | 3.95 | 1.99–2.69 / 2.89–3.99 | 3.29 (PCIe) |

## 9. IaC and configuration tools

Sources: GitHub releases API, PyPI and Galaxy (`tools-iac-validation-releases.json`).

| Tool | Version (date) | License | Note |
|---|---|---|---|
| Terraform | v1.16.6 (2026-10-14) | BUSL-1.1 (not open source) | https://github.com/hashicorp/terraform/releases |
| OpenTofu | v1.13.1 (2026-10-01) | MPL-2.0 | Drop-in for the providers above. https://github.com/opentofu/opentofu/releases |
| ansible-core | 2.21.5 (2026-10-05) | GPL-3.0 | https://pypi.org/project/ansible-core/ |
| ansible (community package) | 14.5.0 (2026-10-06) | GPL-3.0-or-later | https://pypi.org/project/ansible/ |
| Role `nvidia.nvidia_driver` | v2.4.0 (2026-10-05) | BSD-3-Clause | Defaults: `nvidia_driver_branch: "580"`, Ubuntu `-server` packages from the Canonical archive; supports jammy and noble, EL 8 and 9. https://github.com/NVIDIA/ansible-role-nvidia-driver |
| NVIDIA Container Toolkit | v1.20.1 (2026-09-19) | Apache-2.0 | See the install steps below |

NVIDIA Container Toolkit on Ubuntu
(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1
sudo apt-get install -y nvidia-container-toolkit=$NVIDIA_CONTAINER_TOOLKIT_VERSION \
  nvidia-container-toolkit-base=$NVIDIA_CONTAINER_TOOLKIT_VERSION \
  libnvidia-container-tools=$NVIDIA_CONTAINER_TOOLKIT_VERSION \
  libnvidia-container1=$NVIDIA_CONTAINER_TOOLKIT_VERSION
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Validation tools to run in CI:

| Tool | Version (date) | License | Command |
|---|---|---|---|
| `terraform fmt -check`, `terraform validate` (or `tofu`) | as above | – | `terraform init -backend=false && terraform validate` |
| tflint | v0.64.0 (2026-07-17) | MPL-2.0 | `tflint --recursive` |
| trivy (tfsec is merged into it) | v0.76.0 (2026-10-14) | Apache-2.0 | `trivy config infra/` |
| tfsec | v1.28.14 (2025-05-02) | MIT | README points to Trivy; do not add |
| checkov | 3.3.29 (2026-10-13) | Apache-2.0 | `checkov -d infra/` |
| ansible-lint | 26.10.1 (2026-10-13) | GPL-3.0 | `ansible-lint infra/ansible` |
| kubeconform | v0.8.0 (2026-06-04) | Apache-2.0 | `kubeconform -strict -summary infra/kubernetes` |

## 10. Cost guards

| Cloud | Guard on the VM | Budget | Source |
|---|---|---|---|
| AWS | `instance_initiated_shutdown_behavior = "terminate"` plus `shutdown -h +N` in user-data; Spot `valid_until`; small G/P quota | AWS Budgets actions (IAM/SCP policy or stop instances). Data lags, so this is a backstop. | https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html, https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-controls.html |
| Azure | Spot with `eviction_policy = "Delete"`. Auto-shutdown runs once a day at a fixed time. The Microsoft page does not say whether it deallocates **(not verified)**. Disks are billed until deleted, so delete the whole resource group. | Budgets alert only: "your consumption isn't stopped" | https://learn.microsoft.com/azure/virtual-machines/auto-shutdown-vm, https://learn.microsoft.com/azure/virtual-machines/states-billing, https://learn.microsoft.com/azure/cost-management-billing/costs/tutorial-acm-create-budgets |
| GCP | `max_run_duration` (30 s to 120 days) with `instance_termination_action = "DELETE"` | Budgets do not cap spending; spend caps do not cover Compute Engine | https://cloud.google.com/compute/docs/instances/limit-vm-runtime, https://cloud.google.com/billing/docs/how-to/budgets-spend-caps |
| Hetzner | Cancel through Robot `POST /server/{n}/cancellation`. Billing stops at the hour; the monthly price is the cap. | None (prepaid limits only) | section 2.3 |

Use these on every cloud:
- A label or tag on each machine with an `expires-at` time.
- A scheduled reaper job that deletes expired machines.
- A CI job timeout plus an `always()` teardown step.

## 11. Kubernetes

| Item | Value | Source |
|---|---|---|
| NVIDIA k8s-device-plugin | v0.20.1 (2026-09-22) | https://github.com/NVIDIA/k8s-device-plugin/releases |
| NVIDIA GPU Operator | v26.7.2 (2026-10-14); default driver **595.91.07**, toolkit v1.20.1, device plugin v0.20.1 | https://raw.githubusercontent.com/NVIDIA/gpu-operator/v26.7.2/deployments/gpu-operator/values.yaml |
| GPU request | `resources.limits: nvidia.com/gpu: 1`. GPUs go in limits; requests, if set, must equal limits. | https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/ |
| k3s | Adds the `nvidia` containerd runtime automatically when it finds the binaries; pods use `runtimeClassName: nvidia`. Latest v1.37.1+k3s1. | https://docs.k3s.io/advanced#nvidia-container-runtime-support |

Taints and selectors:

| Platform | Taint on GPU nodes | Pod side |
|---|---|---|
| GKE | `nvidia.com/gpu=present:NoSchedule`, automatic only if a non-GPU pool exists | Toleration added automatically; `nodeSelector: cloud.google.com/gke-accelerator` |
| AKS | Convention `sku=gpu:NoSchedule`, added by the user | `tolerations: [{key: sku, operator: Equal, value: gpu, effect: NoSchedule}]` |
| EKS | No default taint on managed node groups **(not verified)**. Auto Mode and Karpenter examples use `nvidia.com/gpu:NoSchedule`. | `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]` |
| k3s on Hetzner GEX | None by default | `runtimeClassName: nvidia` (or the GPU Operator) |

## 12. Hugging Face as a GPU runner

Sources: https://huggingface.co/docs/hub/jobs-pricing,
https://huggingface.co/docs/huggingface_hub/guides/jobs,
https://huggingface.co/docs/inference-endpoints/pricing,
https://huggingface.co/pricing.

| Product | Use for CI | Prices ($/h) |
|---|---|---|
| **HF Jobs** (`hf jobs run --flavor l4x1 <image> <cmd>`, `hf jobs wait`) | Closest to a CI runner. Needs a positive credit balance (PRO, Team or Enterprise credits count). Billed per minute; default timeout 30 min. | t4-small 0.40, l4x1 0.80, a10g-large 1.50, l40sx1 1.80, a100-large 2.50, rtx-pro-6000 2.75, h200 5.00 |
| Inference Endpoints | Serves one model; does not fit our script-driven runs | AWS: L4 0.80, L40S 1.80, A100 2.50. GCP: L4 0.70, A100 3.60, H100 10.00 (the /pricing page shows H100 at 4.50; this conflict was not resolved). |
| Spaces GPU | Not built for batch jobs | Same rates as Jobs; ZeroGPU free |

HF Jobs with `l4x1` (24 GB) or `a100-large` (80 GB) is a simple alternative to the
self-managed flows. It needs an HF token secret and runs our container image.

## 13. Not verified

| Item | Why |
|---|---|
| Whether GEX servers can be ordered through `POST /order/server/transaction`, and how long delivery takes | Needs Robot credentials; docs do not mention GPUs |
| Azure default GPU family quota (0) | Only community answers |
| GCP default value and exact ID of the global GPU quota | Not in docs; check with `gcloud quotas info list` |
| AWS live Spot prices | Needs credentials |
| Whether CUDA 13 works in vGPU slices (Azure RTX PRO / A10, GCP G4 fractional) | Not documented |
| EKS Auto Mode and Bottlerocket exact driver patch | Docs show old samples |
| Default taint on EKS managed node groups | Not documented |
| Whether Azure auto-shutdown deallocates | Microsoft page does not say |
| `runner_group_id` value for repo-level runners | Not stated in REST docs |
| GitHub OIDC `sub` with numeric IDs | Only in the configure-aws-credentials README |
| Ubuntu accelerator images on G2, G4 and A2 | Officially listed only for A4, A4X and A3 Ultra |

## 14. Raw data files

All files are in [`../data/`](../data/). Each one starts with its source URL and
the retrieval date (2026-10-19).

| File | Content |
|---|---|
| `hetzner-gpu-prices.json`, `hetzner-gpu-configurator.json` | GEX prices from the Hetzner price API, OS list |
| `hetzner-server-auction-gpu.json` | GPU servers in the Server Auction |
| `hetzner-hcloud-api.json`, `hetzner-hcloud-registry.json` | hcloud spec check, Terraform registry entry |
| `hetzner-robot-webservice-order-cancel.txt`, `hetzner-billing-cancellation.txt` | Robot API order and cancel, billing quotes |
| `hetzner-robot-terraform-providers.json`, `hetzner-ansible-galaxy.json`, `hetzner-k8s-tools.json` | Tool versions |
| `hetzner-gpu-docs-and-os.txt`, `ubuntu-noble-nvidia-580-server.json` | Driver install notes, Ubuntu R580 package |
| `aws-gpu-prices-us-east-1.json`, `aws-spot-indications-us-east-1.json`, `aws-gpu-instance-specs.json` | EC2 prices and specs |
| `aws-dlami-versions.txt`, `aws-eks-ami-v20261012.txt`, `aws-ec2-quotas.json`, `aws-provider-registry.json`, `aws-tool-versions.json` | AMIs, quotas, versions |
| `azure-retail-prices-eastus*.json`, `azure-retail-prices-eastus2-ncc-h100.json` | Azure Retail Prices API |
| `azure-nvidia-gpu-driver-extension-linux-resources.json`, `azure-azhpc-images-releases.json`, `azure-aks-gpu-components.json` | Driver versions |
| `azure-gpu-quota-families.json`, `azure-terraform-registry.json`, `azure-ci-tooling-versions.json` | Quota names, versions |
| `gcp-gpu-prices-us-central1.txt`, `gcp-gpu-images.txt`, `gcp-cos133-gpu-driver-versions.json` | GCE prices, images, COS drivers |
| `gcp-google-provider-registry.json`, `gcp-gitlab-runners.json` | Versions, GitLab GPU runner |
| `tools-github-releases.json`, `tools-github-runner-api.json`, `tools-ci-runner-pricing.json`, `tools-gpu-cloud-pricing.json` | Runner tools, JIT API, CI and GPU cloud prices |
| `tools-iac-validation-releases.json` | Terraform, OpenTofu, Ansible, linters, NVIDIA role, container toolkit |
| `k8s-gpu-operator-defaults.json`, `k8s-gpu-scheduling.json`, `k8s-hetzner-tools.json` | Kubernetes GPU facts |
| `hf-pricing.txt` | Hugging Face Jobs, Endpoints, Spaces prices |
