output "cluster_name" {
  description = "EKS cluster name."
  value       = aws_eks_cluster.this.name
}

output "kubeconfig" {
  description = "Kubeconfig that gets a token from `aws eks get-token` (needs the AWS CLI and the same credentials)."
  sensitive   = true
  value = yamlencode({
    apiVersion      = "v1"
    kind            = "Config"
    current-context = var.name
    clusters = [{
      name = var.name
      cluster = {
        server                     = aws_eks_cluster.this.endpoint
        certificate-authority-data = aws_eks_cluster.this.certificate_authority[0].data
      }
    }]
    contexts = [{ name = var.name, context = { cluster = var.name, user = var.name } }]
    users = [{
      name = var.name
      user = {
        exec = {
          apiVersion = "client.authentication.k8s.io/v1beta1"
          command    = "aws"
          args       = ["eks", "get-token", "--region", var.region, "--cluster-name", aws_eks_cluster.this.name, "--output", "json"]
        }
      }
    }]
  })
}

output "needs_device_plugin" {
  description = "The AL2023 NVIDIA image has no device plugin, so the orchestrator applies it."
  value       = true
}
