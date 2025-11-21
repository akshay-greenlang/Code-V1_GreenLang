# ============================================================================
# GreenLang AI Agent Foundation - AWS Infrastructure
# ============================================================================
# Production-grade infrastructure with multi-AZ HA, auto-scaling, and security
# Resources: VPC, EKS, RDS, ElastiCache, S3, Secrets Manager, CloudWatch

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # Remote state storage in S3 with DynamoDB locking
  backend "s3" {
    bucket         = "greenlang-terraform-state"
    key            = "agent-foundation/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "greenlang-terraform-lock"

    # Workspaces for multi-environment
    workspace_key_prefix = "environments"
  }
}

# ============================================================================
# Provider Configuration
# ============================================================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "GreenLang-AI-Agent-Foundation"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CostCenter  = "Engineering"
      Owner       = "DevOps-Team"
    }
  }
}

# ============================================================================
# Data Sources
# ============================================================================

data "aws_availability_zones" "available" {
  state = "available"

  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# ============================================================================
# Local Variables
# ============================================================================

locals {
  cluster_name = "greenlang-${var.environment}-eks"

  azs = slice(data.aws_availability_zones.available.names, 0, 3)

  vpc_cidr = var.vpc_cidr

  private_subnets = [
    cidrsubnet(local.vpc_cidr, 4, 1),
    cidrsubnet(local.vpc_cidr, 4, 2),
    cidrsubnet(local.vpc_cidr, 4, 3),
  ]

  public_subnets = [
    cidrsubnet(local.vpc_cidr, 4, 11),
    cidrsubnet(local.vpc_cidr, 4, 12),
    cidrsubnet(local.vpc_cidr, 4, 13),
  ]

  database_subnets = [
    cidrsubnet(local.vpc_cidr, 4, 21),
    cidrsubnet(local.vpc_cidr, 4, 22),
    cidrsubnet(local.vpc_cidr, 4, 23),
  ]

  common_tags = {
    Environment = var.environment
    Application = "greenlang-agent"
    Terraform   = "true"
  }
}

# ============================================================================
# VPC Module
# ============================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "greenlang-${var.environment}-vpc"
  cidr = local.vpc_cidr

  azs              = local.azs
  private_subnets  = local.private_subnets
  public_subnets   = local.public_subnets
  database_subnets = local.database_subnets

  # NAT Gateway configuration
  enable_nat_gateway     = true
  single_nat_gateway     = var.environment == "dev" ? true : false
  one_nat_gateway_per_az = var.environment != "dev" ? true : false

  # DNS configuration
  enable_dns_hostnames = true
  enable_dns_support   = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true
  flow_log_retention_in_days           = 30

  # Subnet tags for EKS
  public_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"             = "1"
  }

  database_subnet_tags = {
    "Database" = "true"
  }

  tags = local.common_tags
}

# ============================================================================
# EKS Cluster Module
# ============================================================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster endpoint access
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks

  # Cluster encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # OIDC Provider for IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Managed Node Groups
  eks_managed_node_groups = {
    # General purpose node group
    general = {
      name = "greenlang-general"

      instance_types = var.environment == "production" ? ["m5.2xlarge"] : ["m5.xlarge"]
      capacity_type  = "ON_DEMAND"

      min_size     = var.environment == "production" ? 9 : 3
      max_size     = var.environment == "production" ? 100 : 10
      desired_size = var.environment == "production" ? 9 : 3

      # Multi-AZ distribution
      subnet_ids = module.vpc.private_subnets

      labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }

      taints = []

      update_config = {
        max_unavailable_percentage = 25
      }
    }

    # GPU node group (optional, for ML workloads)
    gpu = {
      name = "greenlang-gpu"

      instance_types = ["g4dn.xlarge"]
      capacity_type  = "ON_DEMAND"

      min_size     = 0
      max_size     = 5
      desired_size = 0

      labels = {
        Environment = var.environment
        NodeGroup   = "gpu"
        GPU         = "nvidia"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  # Cluster security group rules
  cluster_security_group_additional_rules = {
    egress_nodes_ephemeral_ports_tcp = {
      description                = "To node 1025-65535"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "egress"
      source_node_security_group = true
    }
  }

  # Node security group rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }

    ingress_cluster_all = {
      description                   = "Cluster to node all ports/protocols"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 0
      type                          = "ingress"
      source_cluster_security_group = true
    }
  }

  tags = local.common_tags
}

# ============================================================================
# RDS PostgreSQL Database
# ============================================================================

module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "greenlang-${var.environment}-db"

  # Engine configuration
  engine               = "postgres"
  engine_version       = "16.1"
  family               = "postgres16"
  major_engine_version = "16"
  instance_class       = var.db_instance_class

  # Storage configuration
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_encrypted     = true
  storage_type          = "gp3"
  iops                  = 3000

  # Database configuration
  db_name  = "greenlang"
  username = "greenlang_admin"
  port     = 5432

  # Password managed by Secrets Manager
  manage_master_user_password = true

  # Multi-AZ for production
  multi_az = var.environment == "production" ? true : false

  # Network configuration
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Backup configuration
  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  # Enhanced monitoring
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  monitoring_interval             = 60
  monitoring_role_name            = "greenlang-${var.environment}-rds-monitoring"
  create_monitoring_role          = true

  # Performance Insights
  performance_insights_enabled    = true
  performance_insights_retention_period = var.environment == "production" ? 7 : 7

  # Parameters
  parameters = [
    {
      name  = "autovacuum"
      value = "1"
    },
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
  ]

  # Deletion protection
  deletion_protection = var.environment == "production" ? true : false
  skip_final_snapshot = var.environment != "production" ? true : false

  final_snapshot_identifier_prefix = "greenlang-${var.environment}-final-"

  tags = local.common_tags
}

# ============================================================================
# ElastiCache Redis
# ============================================================================

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "greenlang-${var.environment}-redis"
  replication_group_description = "GreenLang Agent Redis Cluster"

  engine               = "redis"
  engine_version       = "7.0"
  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = aws_elasticache_parameter_group.redis.name

  # Multi-AZ configuration
  automatic_failover_enabled = var.environment == "production" ? true : false
  multi_az_enabled           = var.environment == "production" ? true : false
  num_cache_clusters         = var.environment == "production" ? 3 : 1

  # Network configuration
  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]

  # Encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled         = true

  # Backup configuration
  snapshot_retention_limit = var.environment == "production" ? 7 : 3
  snapshot_window          = "03:00-05:00"
  maintenance_window       = "sun:05:00-sun:07:00"

  # Auto minor version upgrade
  auto_minor_version_upgrade = true

  # Logging
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "slow-log"
  }

  tags = local.common_tags
}

resource "aws_elasticache_parameter_group" "redis" {
  name   = "greenlang-${var.environment}-redis-params"
  family = "redis7"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  tags = local.common_tags
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "greenlang-${var.environment}-redis-subnet"
  subnet_ids = module.vpc.database_subnets

  tags = local.common_tags
}

# ============================================================================
# S3 Bucket for Data Storage
# ============================================================================

module "s3_bucket" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 3.0"

  bucket = "greenlang-${var.environment}-data-${data.aws_caller_identity.current.account_id}"

  # Versioning
  versioning = {
    enabled = true
  }

  # Server-side encryption
  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm     = "aws:kms"
        kms_master_key_id = aws_kms_key.s3.arn
      }
      bucket_key_enabled = true
    }
  }

  # Lifecycle rules
  lifecycle_rule = [
    {
      id      = "archive-old-versions"
      enabled = true

      noncurrent_version_transition = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER"
        }
      ]

      noncurrent_version_expiration = {
        days = 365
      }
    }
  ]

  # Block public access
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true

  # Logging
  logging = {
    target_bucket = module.s3_logs_bucket.s3_bucket_id
    target_prefix = "s3-access-logs/"
  }

  tags = local.common_tags
}

# S3 bucket for logs
module "s3_logs_bucket" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 3.0"

  bucket = "greenlang-${var.environment}-logs-${data.aws_caller_identity.current.account_id}"

  # Block public access
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true

  # Lifecycle for logs
  lifecycle_rule = [
    {
      id      = "expire-old-logs"
      enabled = true

      expiration = {
        days = 90
      }
    }
  ]

  tags = local.common_tags
}

# ============================================================================
# KMS Keys
# ============================================================================

resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = merge(local.common_tags, {
    Name = "greenlang-${var.environment}-eks-key"
  })
}

resource "aws_kms_alias" "eks" {
  name          = "alias/greenlang-${var.environment}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

resource "aws_kms_key" "s3" {
  description             = "S3 Bucket Encryption Key"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = merge(local.common_tags, {
    Name = "greenlang-${var.environment}-s3-key"
  })
}

resource "aws_kms_alias" "s3" {
  name          = "alias/greenlang-${var.environment}-s3"
  target_key_id = aws_kms_key.s3.key_id
}

# ============================================================================
# Security Groups
# ============================================================================

resource "aws_security_group" "rds" {
  name_prefix = "greenlang-${var.environment}-rds-"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "PostgreSQL from EKS nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "greenlang-${var.environment}-rds-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "greenlang-${var.environment}-redis-"
  description = "Security group for ElastiCache Redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "Redis from EKS nodes"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "greenlang-${var.environment}-redis-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# ============================================================================
# CloudWatch Log Groups
# ============================================================================

resource "aws_cloudwatch_log_group" "redis_slow_log" {
  name              = "/aws/elasticache/greenlang-${var.environment}-redis/slow-log"
  retention_in_days = 30

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/greenlang-${var.environment}/application"
  retention_in_days = 30

  tags = local.common_tags
}

# ============================================================================
# Secrets Manager
# ============================================================================

resource "aws_secretsmanager_secret" "app_secrets" {
  name_prefix             = "greenlang-${var.environment}-app-"
  description             = "Application secrets for GreenLang AI Agent"
  recovery_window_in_days = var.environment == "production" ? 30 : 7

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id

  secret_string = jsonencode({
    database_url = "postgresql://${module.rds.db_instance_username}:${module.rds.db_instance_password}@${module.rds.db_instance_endpoint}/${module.rds.db_instance_name}"
    redis_url    = "redis://${aws_elasticache_replication_group.redis.primary_endpoint_address}:${aws_elasticache_replication_group.redis.port}"
  })
}

# ============================================================================
# IAM Roles for Service Accounts (IRSA)
# ============================================================================

module "irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "greenlang-${var.environment}-agent-role"

  role_policy_arns = {
    policy = aws_iam_policy.agent_policy.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["greenlang-ai:greenlang-agent-sa"]
    }
  }

  tags = local.common_tags
}

resource "aws_iam_policy" "agent_policy" {
  name_prefix = "greenlang-${var.environment}-agent-"
  description = "IAM policy for GreenLang AI Agent"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          module.s3_bucket.s3_bucket_arn,
          "${module.s3_bucket.s3_bucket_arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.app_secrets.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = [
          aws_kms_key.s3.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}
