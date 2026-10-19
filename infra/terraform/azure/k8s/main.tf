# Temporary AKS cluster with one GPU node pool (issue #23). AKS installs the
# NVIDIA driver on the GPU pool (gpu_driver = "Install"); the orchestrator
# applies infra/kubernetes/nvidia-device-plugin.yaml unless a node already
# reports nvidia.com/gpu. The cluster and its node resource group are
# deleted with the run's resource group.
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
  subscription_id = var.subscription_id == "" ? null : var.subscription_id
}

locals {
  tags = merge(var.tags, {
    managed-by = "medbench-cloud-gpu"
    run-id     = var.run_id
    expires-at = var.expires_at
  })
}

resource "azurerm_resource_group" "this" {
  name     = var.name
  location = var.location
  tags     = local.tags
}

resource "azurerm_kubernetes_cluster" "this" {
  name                = var.name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  dns_prefix          = var.name
  node_resource_group = "${var.name}-nodes"
  tags                = local.tags

  default_node_pool {
    name       = "system"
    vm_size    = var.system_vm_size
    node_count = 1
    tags       = local.tags
  }

  identity {
    type = "SystemAssigned"
  }

  # Fixed node pools only; no node auto-provisioning (Karpenter).
  node_provisioning_profile {
    mode = "Manual"
  }

  api_server_access_profile {
    authorized_ip_ranges = var.allowed_api_cidrs
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.this.id
  vm_size               = var.instance_type
  node_count            = var.node_count
  os_disk_size_gb       = var.disk_gb
  gpu_driver            = "Install"
  node_taints           = ["sku=gpu:NoSchedule"]
  priority              = var.spot ? "Spot" : "Regular"
  eviction_policy       = var.spot ? "Delete" : null
  tags                  = local.tags
}
