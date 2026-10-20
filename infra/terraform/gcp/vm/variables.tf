variable "name" {
  description = "Prefix of every resource name, unique per run (medbench-<run id>)."
  type        = string
}

variable "run_id" {
  description = "Run identifier, stored as the run-id label."
  type        = string
}

variable "expires_at" {
  description = "Unix time after which the reaper deletes the resources."
  type        = string
}

variable "ttl_minutes" {
  description = "Minutes after which Compute Engine deletes the VM (max_run_duration)."
  type        = number
  default     = 240
}

variable "project" {
  description = "Google Cloud project id."
  type        = string
}

variable "region" {
  description = "Region of the subnet."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone of the VM; the GPU type must be offered there (gcloud compute accelerator-types list)."
  type        = string
  default     = "us-central1-a"
}

variable "instance_type" {
  description = "Machine type: g2-standard-8 (1x L4 24 GB, GPU attached) fits MedGemma 27B Q4, a2-ultragpu-1g (1x A100 80 GB) fits UniMedVL."
  type        = string
  default     = "g2-standard-8"
}

variable "accelerator_type" {
  description = "GPU to attach for N1 machine types (nvidia-tesla-t4); empty for G2, G4, A2 and A3, which come with their GPUs."
  type        = string
  default     = ""
}

variable "spot" {
  description = "Use a Spot VM (cheaper, can be preempted)."
  type        = bool
  default     = false
}

variable "image" {
  description = "Boot image: the Deep Learning VM base image ships NVIDIA driver R580."
  type        = string
  default     = "projects/deeplearning-platform-release/global/images/family/common-cu129-ubuntu-2404-nvidia-580"
}

variable "ssh_user" {
  description = "Login user created from the ssh-keys metadata."
  type        = string
  default     = "medbench"
}

variable "disk_gb" {
  description = "Boot disk size: weights, llama.cpp, the virtual environments and reports."
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

variable "labels" {
  description = "Extra labels for every resource (lowercase keys and values)."
  type        = map(string)
  default     = {}
}
