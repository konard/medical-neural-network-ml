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

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version (empty: the EKS default)."
  type        = string
  default     = ""
}

variable "instance_type" {
  description = "GPU node type: g6.xlarge (1x L4 24 GB, 16 GiB RAM) fits MedGemma 4B, g6.2xlarge (32 GiB RAM) MedGemma 27B Q4, g6e.2xlarge (1x L40S 48 GB, 64 GiB RAM) UniMedVL."
  type        = string
  default     = "g6.xlarge"
}

variable "node_count" {
  description = "GPU nodes in the node group."
  type        = number
  default     = 1
}

variable "spot" {
  description = "Use spot capacity for the GPU nodes."
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

variable "tags" {
  description = "Extra tags for every resource."
  type        = map(string)
  default     = {}
}
