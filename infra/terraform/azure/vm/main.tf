# Temporary GPU VM on Azure (issue #23). Everything lives in one resource
# group per run, so `destroy` or the reaper deletes it in one call. Azure
# keeps billing a VM that powered itself off, so the reaper must run.
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

resource "azurerm_virtual_network" "this" {
  name                = var.name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  address_space       = ["10.42.0.0/16"]
  tags                = local.tags
}

resource "azurerm_subnet" "this" {
  name                 = var.name
  resource_group_name  = azurerm_resource_group.this.name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = ["10.42.1.0/24"]
}

resource "azurerm_network_security_group" "ssh" {
  name                = var.name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  tags                = local.tags

  security_rule {
    name                       = "ssh-from-orchestrator"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefixes    = var.allowed_ssh_cidrs
    destination_address_prefix = "*"
  }
}

resource "azurerm_public_ip" "this" {
  name                = var.name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  allocation_method   = "Static"
  sku                 = "Standard"
  tags                = local.tags
}

resource "azurerm_network_interface" "this" {
  name                = var.name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  tags                = local.tags

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.this.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.this.id
  }
}

resource "azurerm_network_interface_security_group_association" "this" {
  network_interface_id      = azurerm_network_interface.this.id
  network_security_group_id = azurerm_network_security_group.ssh.id
}

resource "azurerm_linux_virtual_machine" "gpu" {
  name                  = var.name
  resource_group_name   = azurerm_resource_group.this.name
  location              = azurerm_resource_group.this.location
  size                  = var.instance_type
  admin_username        = var.ssh_user
  network_interface_ids = [azurerm_network_interface.this.id]
  priority              = var.spot ? "Spot" : "Regular"
  eviction_policy       = var.spot ? "Delete" : null
  tags                  = local.tags
  custom_data = base64encode(templatefile("${path.module}/../../cloud-init/gpu-vm.yaml.tftpl", {
    run_id      = var.run_id
    expires_at  = var.expires_at
    ttl_minutes = var.ttl_minutes
  }))

  admin_ssh_key {
    username   = var.ssh_user
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = var.disk_gb
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "ubuntu-24_04-lts"
    sku       = "server"
    version   = "latest"
  }

  depends_on = [azurerm_network_interface_security_group_association.this]
}

# Microsoft's driver extension picks the driver per VM family: the GRID
# driver (595 or 580) for A10 and T4, the CUDA driver for A100 and H100.
resource "azurerm_virtual_machine_extension" "nvidia" {
  name                       = "NvidiaGpuDriverLinux"
  virtual_machine_id         = azurerm_linux_virtual_machine.gpu.id
  publisher                  = "Microsoft.HpcCompute"
  type                       = "NvidiaGpuDriverLinux"
  type_handler_version       = "1.6"
  auto_upgrade_minor_version = true
  tags                       = local.tags
}
