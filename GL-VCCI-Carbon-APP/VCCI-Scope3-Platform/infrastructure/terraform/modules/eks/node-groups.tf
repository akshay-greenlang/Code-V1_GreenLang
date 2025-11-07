# EKS Node Groups Configuration

# ============================================================================
# Node Group IAM Role
# ============================================================================

resource "aws_iam_role" "node_group" {
  name = "${var.cluster_name}-node-group-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "node_group_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node_group.name
}

resource "aws_iam_role_policy_attachment" "node_group_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node_group.name
}

resource "aws_iam_role_policy_attachment" "node_group_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node_group.name
}

resource "aws_iam_role_policy_attachment" "node_group_AmazonSSMManagedInstanceCore" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  role       = aws_iam_role.node_group.name
}

# ============================================================================
# Node Security Group
# ============================================================================

resource "aws_security_group" "node_group" {
  name_prefix = "${var.cluster_name}-node-sg-"
  description = "Security group for EKS worker nodes"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = merge(
    var.tags,
    {
      Name                                        = "${var.cluster_name}-node-sg"
      "kubernetes.io/cluster/${var.cluster_name}" = "owned"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Allow nodes to communicate with each other
resource "aws_security_group_rule" "node_ingress_self" {
  description              = "Allow node to communicate with each other"
  type                     = "ingress"
  from_port                = 0
  to_port                  = 65535
  protocol                 = "-1"
  source_security_group_id = aws_security_group.node_group.id
  security_group_id        = aws_security_group.node_group.id
}

# Allow worker Kubelets and pods to receive communication from the cluster control plane
resource "aws_security_group_rule" "node_ingress_cluster" {
  description              = "Allow worker Kubelets and pods to receive communication from the cluster control plane"
  type                     = "ingress"
  from_port                = 1025
  to_port                  = 65535
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.cluster.id
  security_group_id        = aws_security_group.node_group.id
}

# Allow pods to communicate with the cluster API Server
resource "aws_security_group_rule" "cluster_ingress_node_https" {
  description              = "Allow pods to communicate with the cluster API Server"
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.node_group.id
  security_group_id        = aws_security_group.cluster.id
}

# ============================================================================
# Launch Template for Compute Nodes
# ============================================================================

resource "aws_launch_template" "compute" {
  name_prefix = "${var.cluster_name}-compute-"
  description = "Launch template for compute node group"

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      encrypted             = true
      kms_key_id            = var.kms_key_arn
      delete_on_termination = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  monitoring {
    enabled = true
  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(
      var.tags,
      {
        Name = "${var.cluster_name}-compute-node"
      }
    )
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name        = var.cluster_name
    cluster_endpoint    = aws_eks_cluster.main.endpoint
    cluster_ca          = aws_eks_cluster.main.certificate_authority[0].data
  }))

  tags = var.tags
}

# ============================================================================
# Compute Node Group (General Workloads)
# ============================================================================

resource "aws_eks_node_group" "compute" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.cluster_name}-compute"
  node_role_arn   = aws_iam_role.node_group.arn
  subnet_ids      = var.subnet_ids

  scaling_config {
    desired_size = var.compute_node_group_config.desired_size
    max_size     = var.compute_node_group_config.max_size
    min_size     = var.compute_node_group_config.min_size
  }

  instance_types = var.compute_node_group_config.instance_types
  capacity_type  = "ON_DEMAND"

  launch_template {
    id      = aws_launch_template.compute.id
    version = "$Latest"
  }

  labels = {
    role     = "compute"
    workload = "general"
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-compute-node-group"
    }
  )

  depends_on = [
    aws_iam_role_policy_attachment.node_group_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_group_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_group_AmazonEC2ContainerRegistryReadOnly,
  ]

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }
}

# ============================================================================
# Launch Template for Memory Nodes
# ============================================================================

resource "aws_launch_template" "memory" {
  name_prefix = "${var.cluster_name}-memory-"
  description = "Launch template for memory-optimized node group"

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 200
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      encrypted             = true
      kms_key_id            = var.kms_key_arn
      delete_on_termination = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  monitoring {
    enabled = true
  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(
      var.tags,
      {
        Name = "${var.cluster_name}-memory-node"
      }
    )
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name        = var.cluster_name
    cluster_endpoint    = aws_eks_cluster.main.endpoint
    cluster_ca          = aws_eks_cluster.main.certificate_authority[0].data
  }))

  tags = var.tags
}

# ============================================================================
# Memory Node Group (Database and Caching Workloads)
# ============================================================================

resource "aws_eks_node_group" "memory" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.cluster_name}-memory"
  node_role_arn   = aws_iam_role.node_group.arn
  subnet_ids      = var.subnet_ids

  scaling_config {
    desired_size = var.memory_node_group_config.desired_size
    max_size     = var.memory_node_group_config.max_size
    min_size     = var.memory_node_group_config.min_size
  }

  instance_types = var.memory_node_group_config.instance_types
  capacity_type  = "ON_DEMAND"

  launch_template {
    id      = aws_launch_template.memory.id
    version = "$Latest"
  }

  labels = {
    role     = "memory"
    workload = "database"
  }

  taint {
    key    = "workload"
    value  = "memory-intensive"
    effect = "NO_SCHEDULE"
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-memory-node-group"
    }
  )

  depends_on = [
    aws_iam_role_policy_attachment.node_group_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_group_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_group_AmazonEC2ContainerRegistryReadOnly,
  ]

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }
}

# ============================================================================
# Launch Template for GPU Nodes
# ============================================================================

resource "aws_launch_template" "gpu" {
  name_prefix = "${var.cluster_name}-gpu-"
  description = "Launch template for GPU node group"

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 150
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      encrypted             = true
      kms_key_id            = var.kms_key_arn
      delete_on_termination = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  monitoring {
    enabled = true
  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(
      var.tags,
      {
        Name = "${var.cluster_name}-gpu-node"
      }
    )
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name        = var.cluster_name
    cluster_endpoint    = aws_eks_cluster.main.endpoint
    cluster_ca          = aws_eks_cluster.main.certificate_authority[0].data
  }))

  tags = var.tags
}

# ============================================================================
# GPU Node Group (ML Workloads)
# ============================================================================

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.cluster_name}-gpu"
  node_role_arn   = aws_iam_role.node_group.arn
  subnet_ids      = var.subnet_ids

  scaling_config {
    desired_size = var.gpu_node_group_config.desired_size
    max_size     = var.gpu_node_group_config.max_size
    min_size     = var.gpu_node_group_config.min_size
  }

  instance_types = var.gpu_node_group_config.instance_types
  capacity_type  = "ON_DEMAND"
  ami_type       = "AL2_x86_64_GPU"

  launch_template {
    id      = aws_launch_template.gpu.id
    version = "$Latest"
  }

  labels = {
    role     = "gpu"
    workload = "ml"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-gpu-node-group"
    }
  )

  depends_on = [
    aws_iam_role_policy_attachment.node_group_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_group_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_group_AmazonEC2ContainerRegistryReadOnly,
  ]

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }
}
