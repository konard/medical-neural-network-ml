variable "name" {
  description = "Prefix of every resource name, unique per run (medbench-<run id>)."
  type        = string
}

variable "run_id" {
  description = "Run identifier, stored as the run-id tag."
  type        = string
}

variable "expires_at" {
  description = "Unix time after which the reaper deletes the resources."
  type        = string
}

variable "ttl_minutes" {
  description = "Minutes after boot when the VM powers itself off and is terminated."
  type        = number
  default     = 240
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU instance type: g6.xlarge (1x L4 24 GB, 16 GiB RAM) fits MedGemma 4B, g6.2xlarge (32 GiB RAM) MedGemma 27B Q4, g6e.2xlarge (1x L40S 48 GB, 64 GiB RAM) UniMedVL."
  type        = string
  default     = "g6.xlarge"
}

variable "spot" {
  description = "Use a one-time spot instance (cheaper, can be interrupted)."
  type        = bool
  default     = false
}

variable "ami_ssm_parameter" {
  description = "Public SSM parameter of the image: the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) ships driver R580 or newer."
  type        = string
  default     = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
}

variable "disk_gb" {
  description = "Root disk size: weights, llama.cpp, the virtual environments and reports."
  type        = number
  default     = 150
}

variable "ssh_public_key" {
  description = "Public key of the per-run SSH key pair created by the orchestrator."
  type        = string
}

variable "allowed_ssh_cidrs" {
  description = "Addresses allowed to reach port 22, ideally only the machine that runs the orchestrator."
  type        = list(string)
}

variable "tags" {
  description = "Extra tags for every resource."
  type        = map(string)
  default     = {}
}
