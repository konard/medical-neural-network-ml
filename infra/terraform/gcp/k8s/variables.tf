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
  description = "Zone of the zonal cluster; the GPU type must be offered there."
  type        = string
  default     = "us-central1-a"
}

variable "instance_type" {
  description = "GPU node machine type: g2-standard-8 (1x L4 24 GB) fits MedGemma 27B Q4, a2-ultragpu-1g (1x A100 80 GB) fits UniMedVL."
  type        = string
  default     = "g2-standard-8"
}

variable "accelerator_type" {
  description = "GPU of the node machine type: nvidia-l4 for G2, nvidia-a100-80gb for a2-ultragpu, nvidia-tesla-t4 for N1."
  type        = string
  default     = "nvidia-l4"
}

variable "node_count" {
  description = "GPU nodes in the node pool."
  type        = number
  default     = 1
}

variable "spot" {
  description = "Use Spot VMs for the GPU nodes."
  type        = bool
  default     = false
}

variable "disk_gb" {
  description = "Node disk size: images, weights and reports."
  type        = number
  default     = 150
}

variable "allowed_api_cidrs" {
  description = "Addresses allowed to reach the Kubernetes API, ideally only the machine that runs the orchestrator."
  type        = list(string)
}

variable "labels" {
  description = "Extra labels for every resource (lowercase keys and values)."
  type        = map(string)
  default     = {}
}
