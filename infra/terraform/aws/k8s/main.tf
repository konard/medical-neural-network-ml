# Temporary EKS cluster with one GPU node group (issue #23). The node group
# uses the EKS-optimized AL2023 NVIDIA image (driver R580 and the NVIDIA
# container toolkit); the device plugin is not part of it, so the
# orchestrator applies infra/kubernetes/nvidia-device-plugin.yaml
# (needs_device_plugin output).
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

data "aws_ec2_instance_type_offerings" "gpu" {
  location_type = "availability-zone"
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  gpu_zone = sort(data.aws_ec2_instance_type_offerings.gpu.locations)[0]
  # EKS needs subnets in two zones; the GPU nodes use the first one.
  zones = [local.gpu_zone, [for z in sort(data.aws_availability_zones.available.names) : z if z != local.gpu_zone][0]]
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
  count                   = 2
  vpc_id                  = aws_vpc.this.id
  cidr_block              = "10.42.${count.index + 1}.0/24"
  availability_zone       = local.zones[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.name}-${count.index}" }
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
  count          = 2
  subnet_id      = aws_subnet.this[count.index].id
  route_table_id = aws_route_table.this.id
}

data "aws_iam_policy_document" "cluster_assume" {
  statement {
    actions = ["sts:AssumeRole", "sts:TagSession"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "cluster" {
  name               = "${var.name}-cluster"
  assume_role_policy = data.aws_iam_policy_document.cluster_assume.json
}

resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

data "aws_iam_policy_document" "node_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name               = "${var.name}-node"
  assume_role_policy = data.aws_iam_policy_document.node_assume.json
}

resource "aws_iam_role_policy_attachment" "node" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
  ])
  role       = aws_iam_role.node.name
  policy_arn = each.value
}

resource "aws_eks_cluster" "this" {
  name     = var.name
  role_arn = aws_iam_role.cluster.arn
  version  = var.kubernetes_version == "" ? null : var.kubernetes_version

  access_config {
    authentication_mode                         = "API"
    bootstrap_cluster_creator_admin_permissions = true
  }

  vpc_config {
    subnet_ids              = aws_subnet.this[*].id
    endpoint_public_access  = true
    endpoint_private_access = true
    public_access_cidrs     = var.allowed_api_cidrs
  }

  depends_on = [aws_iam_role_policy_attachment.cluster]
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "gpu"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = [aws_subnet.this[0].id]
  ami_type        = "AL2023_x86_64_NVIDIA"
  capacity_type   = var.spot ? "SPOT" : "ON_DEMAND"
  instance_types  = [var.instance_type]
  disk_size       = var.disk_gb

  scaling_config {
    desired_size = var.node_count
    min_size     = var.node_count
    max_size     = var.node_count
  }

  depends_on = [aws_iam_role_policy_attachment.node, aws_route_table_association.this]
}
