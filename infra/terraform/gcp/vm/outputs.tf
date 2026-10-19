output "host" {
  description = "Public address for SSH."
  value       = google_compute_instance.gpu.network_interface[0].access_config[0].nat_ip
}

output "ssh_user" {
  description = "Login user created from the ssh-keys metadata."
  value       = var.ssh_user
}

output "instance_id" {
  description = "Compute Engine instance id."
  value       = google_compute_instance.gpu.instance_id
}
