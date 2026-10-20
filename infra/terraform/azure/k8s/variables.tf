variable "name" {
  description = "Prefix of every resource name, unique per run (medbench-<run id>)."
  type        = string
}

variable "run_id" {
  description = "Run identifier, stored as the run-id tag."
  type        = string
}

variable "expires_at" {
  description = "Unix time after which the reaper deletes the resource groups."
  type        = string
}

variable "subscription_id" {
  description = "Azure subscription id (empty: ARM_SUBSCRIPTION_ID)."
  type        = string
  default     = ""
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "eastus"
}

variable "system_vm_size" {
  description = "VM size of the small CPU system node pool AKS requires."
  type        = string
  default     = "Standard_D2as_v5"
}

variable "instance_type" {
  description = "GPU node size: Standard_NC24ads_A100_v4 (1x A100 80 GB) fits every model; AKS installs the CUDA driver, so the GRID-only A10 sizes are not used here."
  type        = string
  default     = "Standard_NC24ads_A100_v4"
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

variable "tags" {
  description = "Extra tags for every resource."
  type        = map(string)
  default     = {}
}
