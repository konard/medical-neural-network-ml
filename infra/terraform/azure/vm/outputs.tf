output "host" {
  description = "Public address for SSH."
  value       = azurerm_public_ip.this.ip_address
}

output "ssh_user" {
  description = "Admin user of the VM."
  value       = var.ssh_user
}

output "instance_id" {
  description = "Azure VM resource id."
  value       = azurerm_linux_virtual_machine.gpu.id
}

output "resource_group" {
  description = "Resource group that holds everything of this run."
  value       = azurerm_resource_group.this.name
}
