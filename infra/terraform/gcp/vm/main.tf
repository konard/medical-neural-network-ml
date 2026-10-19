# Temporary GPU VM on Google Compute Engine (issue #23). One VPC per run, so
# `destroy` leaves nothing behind; Compute Engine deletes the VM itself when
# ttl_minutes is over (scheduling.max_run_duration).
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
  ip_cidr_range = "10.42.1.0/24"
}

resource "google_compute_firewall" "ssh" {
  name          = "${var.name}-ssh"
  network       = google_compute_network.this.id
  description   = "SSH from the orchestrator only"
  direction     = "INGRESS"
  source_ranges = var.allowed_ssh_cidrs
  target_tags   = [var.name]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_instance" "gpu" {
  name         = var.name
  machine_type = var.instance_type
  tags         = [var.name]

  boot_disk {
    auto_delete = true
    initialize_params {
      image = var.image
      size  = var.disk_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.this.id
    access_config {}
  }

  dynamic "guest_accelerator" {
    for_each = var.accelerator_type == "" ? [] : [var.accelerator_type]
    content {
      type  = guest_accelerator.value
      count = 1
    }
  }

  # GPU VMs cannot live-migrate. max_run_duration deletes the VM even if
  # nobody runs `down`.
  scheduling {
    on_host_maintenance         = "TERMINATE"
    automatic_restart           = false
    preemptible                 = var.spot
    provisioning_model          = var.spot ? "SPOT" : "STANDARD"
    instance_termination_action = "DELETE"
    max_run_duration {
      seconds = var.ttl_minutes * 60
    }
  }

  metadata = {
    ssh-keys               = "${var.ssh_user}:${var.ssh_public_key}"
    block-project-ssh-keys = "true"
    enable-oslogin         = "FALSE"
    install-nvidia-driver  = "True"
    user-data = templatefile("${path.module}/../../cloud-init/gpu-vm.yaml.tftpl", {
      run_id      = var.run_id
      expires_at  = var.expires_at
      ttl_minutes = var.ttl_minutes
    })
  }

  depends_on = [google_compute_firewall.ssh]
}
