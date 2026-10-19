output "host" {
  description = "Public address for SSH."
  value       = aws_instance.gpu.public_ip
}

output "ssh_user" {
  description = "Login user of the image."
  value       = "ubuntu"
}

output "instance_id" {
  description = "EC2 instance id."
  value       = aws_instance.gpu.id
}
