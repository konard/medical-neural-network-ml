output "cluster_name" {
  description = "GKE cluster name."
  value       = google_container_cluster.this.name
}

output "kubeconfig" {
  description = "Kubeconfig that gets a token from gke-gcloud-auth-plugin (needs the gcloud CLI and the same credentials)."
  sensitive   = true
  value = yamlencode({
    apiVersion      = "v1"
    kind            = "Config"
    current-context = var.name
    clusters = [{
      name = var.name
      cluster = {
        server                     = "https://${google_container_cluster.this.endpoint}"
        certificate-authority-data = google_container_cluster.this.master_auth[0].cluster_ca_certificate
      }
    }]
    contexts = [{ name = var.name, context = { cluster = var.name, user = var.name } }]
    users = [{
      name = var.name
      user = {
        exec = {
          apiVersion         = "client.authentication.k8s.io/v1beta1"
          command            = "gke-gcloud-auth-plugin"
          provideClusterInfo = true
        }
      }
    }]
  })
}

output "needs_device_plugin" {
  description = "GKE runs its own NVIDIA device plugin."
  value       = false
}
