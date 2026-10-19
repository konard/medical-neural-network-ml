variable "name" {
  description = "Prefix of every resource name, unique per run (medbench-<run id>)."
  type        = string
}

variable "run_id" {
  description = "Run identifier, stored as the run-id tag."
  type        = string
}

variable "expires_at" {
  description = "Unix time after which the reaper deletes the resource group."
  type        = string
}

variable "ttl_minutes" {
  description = "Minutes after boot when the VM powers itself off; a stopped VM is still billed until the reaper or `down` deletes it."
  type        = number
  default     = 240
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

variable "instance_type" {
  description = "VM size: Standard_NV36ads_A10_v5 (1x A10 24 GB) fits MedGemma 27B Q4, Standard_NC24ads_A100_v4 (1x A100 80 GB) fits UniMedVL."
  type        = string
  default     = "Standard_NV36ads_A10_v5"
}

variable "spot" {
  description = "Use a Spot VM (cheaper, can be evicted)."
  type        = bool
  default     = false
}

variable "ssh_user" {
  description = "Admin user of the VM."
  type        = string
  default     = "medbench"
}

variable "disk_gb" {
  description = "OS disk size: weights, llama.cpp, the virtual environments and reports."
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
