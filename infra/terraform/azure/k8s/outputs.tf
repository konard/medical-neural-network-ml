output "cluster_name" {
  description = "AKS cluster name."
  value       = azurerm_kubernetes_cluster.this.name
}

output "kubeconfig" {
  description = "Kubeconfig of the cluster user."
  sensitive   = true
  value       = azurerm_kubernetes_cluster.this.kube_config_raw
}

output "needs_device_plugin" {
  description = "Apply the NVIDIA device plugin if no node reports nvidia.com/gpu."
  value       = true
}

output "resource_group" {
  description = "Resource group that holds everything of this run."
  value       = azurerm_resource_group.this.name
}
