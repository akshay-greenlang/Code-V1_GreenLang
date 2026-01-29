# main.tf - Development Environment Infrastructure
# Component: GL-FOUND-X-003 - GreenLang Unit & Reference Normalizer
# Environment: Development
# Purpose: Complete AWS infrastructure for dev environment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "s3" {
    bucket         = "greenlang-terraform-state"
    key            = "gl-normalizer/dev/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "greenlang-terraform-locks"
  }
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "GreenLang"
      Component   = "GL-FOUND-X-003"
      Environment = "dev"
      ManagedBy   = "terraform"
      Owner       = "platform-team"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = module.eks.cluster_auth_token

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# =============================================================================
# VPC Configuration
# =============================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc-${var.environment}"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway     = true
  single_nat_gateway     = true # Cost optimization for dev
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # Tags required for EKS
  public_subnet_tags = {
    "kubernetes.io/role/elb"                                      = "1"
    "kubernetes.io/cluster/${var.project_name}-${var.environment}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"                              = "1"
    "kubernetes.io/cluster/${var.project_name}-${var.environment}" = "shared"
  }

  tags = var.tags
}

# =============================================================================
# EKS Cluster
# =============================================================================

module "eks" {
  source = "../../modules/eks"

  cluster_name     = "${var.project_name}-${var.environment}"
  cluster_version  = var.eks_cluster_version
  vpc_id           = module.vpc.vpc_id
  subnet_ids       = concat(module.vpc.public_subnets, module.vpc.private_subnets)
  private_subnet_ids = module.vpc.private_subnets

  # Dev-specific settings
  enable_public_access = true
  public_access_cidrs  = var.eks_public_access_cidrs

  # Smaller node groups for dev
  general_instance_types    = ["t3.medium", "t3.large"]
  general_node_desired_size = 2
  general_node_min_size     = 1
  general_node_max_size     = 4
  node_disk_size            = 50

  # No spot nodes in dev for simplicity
  enable_spot_nodes = false

  audit_bucket_name = module.s3.audit_bucket_id

  tags = var.tags
}

# =============================================================================
# RDS PostgreSQL
# =============================================================================

module "rds" {
  source = "../../modules/rds"

  identifier  = "${var.project_name}-${var.environment}"
  environment = var.environment
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnets

  eks_node_security_group_id = module.eks.node_security_group_id

  # Dev-specific settings
  instance_class        = "db.t3.small"
  allocated_storage     = 20
  max_allocated_storage = 50
  multi_az              = false # Cost saving for dev
  backup_retention_period = 3

  # Allow deletion in dev
  skip_final_snapshot   = true
  deletion_protection   = false
  apply_immediately     = true

  # Reduced monitoring for dev
  performance_insights_enabled = false
  enhanced_monitoring_interval = 0

  tags = var.tags
}

# =============================================================================
# MSK Kafka (Simplified for Dev)
# =============================================================================

module "kafka" {
  source = "../../modules/kafka"

  cluster_name = "${var.project_name}-${var.environment}"
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.private_subnets

  eks_node_security_group_id = module.eks.node_security_group_id

  # Dev-specific settings
  kafka_version          = "3.5.1"
  number_of_broker_nodes = 2
  broker_instance_type   = "kafka.t3.small"
  broker_ebs_volume_size = 50

  # Simplified auth for dev
  enable_sasl_scram            = true
  enable_sasl_iam              = false
  enable_unauthenticated_access = true

  sasl_scram_username = "gl_normalizer_dev"
  sasl_scram_password = var.kafka_password

  # Reduced retention for dev
  log_retention_hours = 48

  # Minimal monitoring for dev
  enhanced_monitoring  = "DEFAULT"
  enable_jmx_exporter  = false
  enable_node_exporter = false
  enable_s3_logs       = false

  tags = var.tags
}

# =============================================================================
# S3 Buckets
# =============================================================================

module "s3" {
  source = "../../modules/s3"

  bucket_prefix = var.project_name
  environment   = var.environment

  gl_normalizer_role_arn = module.eks.gl_normalizer_role_arn

  # Dev-specific settings
  retention_days    = 30 # Short retention for dev
  enable_object_lock = false
  enable_access_logging = false
  enable_cross_region_replication = false

  tags = var.tags
}

# =============================================================================
# ElastiCache Redis
# =============================================================================

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-redis"
  subnet_ids = module.vpc.private_subnets

  tags = var.tags
}

resource "aws_security_group" "redis" {
  name        = "${var.project_name}-${var.environment}-redis-sg"
  description = "Security group for ElastiCache Redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-redis-sg"
  })
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project_name}-${var.environment}"
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = "cache.t3.micro" # Small for dev
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  snapshot_retention_limit = 0 # No snapshots for dev
  maintenance_window       = "sun:05:00-sun:06:00"

  tags = var.tags
}

# =============================================================================
# Secrets Manager
# =============================================================================

resource "aws_secretsmanager_secret" "app_secrets" {
  name        = "gl-normalizer/${var.environment}/app-secrets"
  description = "Application secrets for GL Normalizer ${var.environment}"

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    database_url    = "postgresql://${module.rds.master_username}:PASSWORD@${module.rds.endpoint}/${module.rds.database_name}"
    redis_url       = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:${aws_elasticache_cluster.redis.cache_nodes[0].port}"
    kafka_servers   = module.kafka.bootstrap_brokers_sasl_scram
    kafka_username  = "gl_normalizer_dev"
    api_secret_key  = random_password.api_secret.result
    jwt_secret_key  = random_password.jwt_secret.result
  })
}

resource "random_password" "api_secret" {
  length  = 64
  special = true
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

# =============================================================================
# Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.endpoint
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = "${aws_elasticache_cluster.redis.cache_nodes[0].address}:${aws_elasticache_cluster.redis.cache_nodes[0].port}"
}

output "kafka_bootstrap_servers" {
  description = "Kafka bootstrap servers"
  value       = module.kafka.bootstrap_brokers_sasl_scram
}

output "s3_audit_bucket" {
  description = "S3 audit bucket name"
  value       = module.s3.audit_bucket_id
}

output "secrets_manager_arn" {
  description = "Secrets Manager secret ARN"
  value       = aws_secretsmanager_secret.app_secrets.arn
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
