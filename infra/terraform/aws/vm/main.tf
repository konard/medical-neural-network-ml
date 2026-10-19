# Temporary GPU VM on AWS EC2 (issue #23). One VPC per run, so `destroy`
# leaves nothing behind; the instance terminates itself when ttl_minutes is
# over (instance_initiated_shutdown_behavior).
provider "aws" {
  region = var.region
  default_tags {
    tags = merge(var.tags, {
      managed-by = "medbench-cloud-gpu"
      run-id     = var.run_id
      expires-at = var.expires_at
    })
  }
}

data "aws_ssm_parameter" "ami" {
  name = var.ami_ssm_parameter
}

# GPU types are offered in some availability zones only.
data "aws_ec2_instance_type_offerings" "gpu" {
  location_type = "availability-zone"
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

locals {
  zone = sort(data.aws_ec2_instance_type_offerings.gpu.locations)[0]
}

resource "aws_vpc" "this" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  tags                 = { Name = var.name }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = var.name }
}

resource "aws_subnet" "this" {
  vpc_id                  = aws_vpc.this.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = local.zone
  map_public_ip_on_launch = true
  tags                    = { Name = var.name }
}

resource "aws_route_table" "this" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
  tags = { Name = var.name }
}

resource "aws_route_table_association" "this" {
  subnet_id      = aws_subnet.this.id
  route_table_id = aws_route_table.this.id
}

resource "aws_security_group" "ssh" {
  name        = var.name
  description = "SSH from the orchestrator only"
  vpc_id      = aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "ssh" {
  for_each          = toset(var.allowed_ssh_cidrs)
  security_group_id = aws_security_group.ssh.id
  description       = "SSH from the orchestrator"
  cidr_ipv4         = each.value
  ip_protocol       = "tcp"
  from_port         = 22
  to_port           = 22
}

# Model and package downloads (Hugging Face, GitHub, PyPI).
resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.ssh.id
  description       = "Downloads"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_key_pair" "this" {
  key_name   = var.name
  public_key = var.ssh_public_key
}

resource "aws_instance" "gpu" {
  ami                                  = data.aws_ssm_parameter.ami.insecure_value
  instance_type                        = var.instance_type
  subnet_id                            = aws_subnet.this.id
  vpc_security_group_ids               = [aws_security_group.ssh.id]
  key_name                             = aws_key_pair.this.key_name
  instance_initiated_shutdown_behavior = "terminate"
  user_data_replace_on_change          = true
  user_data = templatefile("${path.module}/../../cloud-init/gpu-vm.yaml.tftpl", {
    run_id      = var.run_id
    expires_at  = var.expires_at
    ttl_minutes = var.ttl_minutes
  })

  dynamic "instance_market_options" {
    for_each = var.spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  metadata_options {
    http_tokens   = "required"
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_size           = var.disk_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  tags = { Name = var.name }

  depends_on = [aws_route_table_association.this]
}
