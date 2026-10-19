# Temporary zonal GKE cluster with one GPU node pool (issue #23). GKE installs
# the NVIDIA driver (R580 for L4, A100 and H100 on the default version) and
# runs its own device plugin, so nothing else is applied.
provider "google" {
  project = var.project
  region  = var.region
  zone    = var.zone
  default_labels = merge(var.labels, {
    managed-by = "medbench-cloud-gpu"
    run-id     = var.run_id
    expires-at = var.expires_at
  })
}

resource "google_compute_network" "this" {
  name                    = var.name
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "this" {
  name          = var.name
  network       = google_compute_network.this.id
  ip_cidr_range = "10.42.0.0/20"
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.43.0.0/16"
  }
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.44.0.0/20"
  }
}

resource "google_container_cluster" "this" {
  name                     = var.name
  location                 = var.zone
  network                  = google_compute_network.this.id
  subnetwork               = google_compute_subnetwork.this.id
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false
  resource_labels = merge(var.labels, {
    managed-by = "medbench-cloud-gpu"
    run-id     = var.run_id
    expires-at = var.expires_at
  })

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.allowed_api_cidrs
      content {
        cidr_block   = cidr_blocks.value
        display_name = "orchestrator"
      }
    }
  }
}

resource "google_container_node_pool" "gpu" {
  name       = "gpu"
  cluster    = google_container_cluster.this.id
  location   = var.zone
  node_count = var.node_count

  node_config {
    machine_type = var.instance_type
    disk_size_gb = var.disk_gb
    spot         = var.spot
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.accelerator_type
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }
  }
}
