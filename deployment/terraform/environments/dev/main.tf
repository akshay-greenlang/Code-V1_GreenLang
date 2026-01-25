# GreenLang Development Environment
# Terraform configuration for the development environment

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.0"
    }
  }

  backend "s3" {
    bucket         = "greenlang-terraform-state"
    key            = "environments/dev/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "greenlang-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "dev"
      Project     = "GreenLang"
      ManagedBy   = "Terraform"
    }
  }
}

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------
locals {
  name_prefix  = "greenlang-dev"
  cluster_name = "${local.name_prefix}-eks"

  common_tags = {
    Environment = "dev"
    Project     = "GreenLang"
    CostCenter  = "development"
  }
}

# -----------------------------------------------------------------------------
# VPC Module
# -----------------------------------------------------------------------------
module "vpc" {
  source = "../../modules/vpc"

  name_prefix        = local.name_prefix
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  cluster_name       = local.cluster_name

  enable_nat_gateway   = true
  enable_flow_logs     = true
  flow_log_retention_days = 14
  enable_vpc_endpoints = true

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# EKS Module
# -----------------------------------------------------------------------------
module "eks" {
  source = "../../modules/eks"

  cluster_name    = local.cluster_name
  cluster_version = var.eks_cluster_version
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids

  enable_public_access = true
  public_access_cidrs  = var.eks_public_access_cidrs

  # System node group - smaller for dev
  system_node_instance_types = ["t3.large"]
  system_node_disk_size      = 50
  system_node_desired_size   = 2
  system_node_min_size       = 1
  system_node_max_size       = 3

  # API node group - smaller for dev
  create_api_node_group    = true
  api_node_instance_types  = ["t3.xlarge"]
  api_node_disk_size       = 50
  api_node_desired_size    = 2
  api_node_min_size        = 1
  api_node_max_size        = 4

  # Agent runtime node group - using spot for cost savings in dev
  create_agent_node_group   = true
  agent_node_instance_types = ["t3.large", "t3a.large"]
  agent_node_capacity_type  = "SPOT"
  agent_node_disk_size      = 50
  agent_node_desired_size   = 2
  agent_node_min_size       = 1
  agent_node_max_size       = 5

  enable_cluster_autoscaler       = true
  enable_load_balancer_controller = true
  enable_ebs_csi_driver           = true

  log_retention_days = 14

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# RDS Module
# -----------------------------------------------------------------------------
module "rds" {
  source = "../../modules/rds"

  identifier         = "${local.name_prefix}-postgres"
  vpc_id             = module.vpc.vpc_id
  db_subnet_group_name = module.vpc.db_subnet_group_name

  engine_version = var.rds_engine_version
  instance_class = "db.t3.medium"  # Smaller for dev

  allocated_storage     = 50
  max_allocated_storage = 200

  database_name   = "greenlang_dev"
  master_username = "greenlang_admin"

  multi_az = false  # Single AZ for dev to save costs

  backup_retention_period = 7
  deletion_protection     = false  # Allow deletion in dev

  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  monitoring_interval                   = 60

  max_connections = 200

  allowed_security_groups = [module.eks.cluster_security_group_id]
  allowed_cidr_blocks     = module.vpc.private_subnet_cidrs

  alarm_actions = []  # No alarms in dev

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# ElastiCache Module
# -----------------------------------------------------------------------------
module "elasticache" {
  source = "../../modules/elasticache"

  cluster_id        = "${local.name_prefix}-redis"
  vpc_id            = module.vpc.vpc_id
  subnet_group_name = module.vpc.elasticache_subnet_group_name

  engine_version = "7.1"
  node_type      = "cache.t3.small"  # Smaller for dev

  num_cache_clusters   = 1  # Single node for dev
  multi_az_enabled     = false

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = 3
  skip_final_snapshot      = true  # Skip in dev

  allowed_security_groups = [module.eks.cluster_security_group_id]
  allowed_cidr_blocks     = module.vpc.private_subnet_cidrs

  enable_cloudwatch_alarms = false  # No alarms in dev

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# S3 Module
# -----------------------------------------------------------------------------
module "s3" {
  source = "../../modules/s3"

  name_prefix = local.name_prefix

  create_kms_key = true

  create_artifacts_bucket                      = true
  artifacts_noncurrent_version_expiration_days = 30

  create_logs_bucket   = true
  logs_retention_days  = 30
  enable_elb_logging   = true
  elb_account_id       = var.elb_account_id

  create_backups_bucket                      = true
  backups_retention_days                     = 90
  backups_noncurrent_version_expiration_days = 30
  enable_backup_object_lock                  = false

  create_data_bucket                      = true
  data_noncurrent_version_expiration_days = 30

  enable_cors          = true
  cors_allowed_origins = ["*"]  # Allow all in dev

  create_static_assets_bucket = false  # Not needed in dev

  enable_replication   = false  # No DR in dev
  enable_access_logging = false  # Simplified for dev

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# IAM Module
# -----------------------------------------------------------------------------
module "iam" {
  source = "../../modules/iam"

  name_prefix       = local.name_prefix
  oidc_provider_arn = module.eks.oidc_provider_arn
  oidc_provider_url = module.eks.oidc_provider_url

  app_namespace = "greenlang"

  create_agent_role          = true
  agent_namespace            = "greenlang-agents"
  agent_service_account_name = "agent-runtime"
  agent_s3_write_bucket_arns = [module.s3.data_bucket_arn]

  enable_s3_access              = true
  s3_bucket_arns                = module.s3.all_bucket_arns
  enable_secrets_manager_access = true
  enable_kms_access             = true
  kms_key_arns                  = [module.s3.kms_key_arn, module.rds.kms_key_arn, module.elasticache.kms_key_arn]

  enable_sqs_access = false
  enable_sns_access = false
  enable_bedrock_access = true

  create_cicd_role          = true
  create_github_oidc_provider = var.create_github_oidc_provider
  github_oidc_provider_arn  = var.github_oidc_provider_arn
  github_org                = var.github_org
  github_repo               = var.github_repo
  ecr_repository_arns       = ["*"]
  cicd_artifact_bucket_arns = [module.s3.artifacts_bucket_arn]

  create_external_secrets_role    = true
  external_secrets_namespace      = "external-secrets"
  external_secrets_service_account = "external-secrets"

  create_monitoring_role     = true
  monitoring_namespace       = "monitoring"
  monitoring_service_account = "cloudwatch-agent"

  create_cross_account_role = false

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
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
  description = "Redis primary endpoint"
  value       = module.elasticache.primary_endpoint_address
}

output "s3_buckets" {
  description = "S3 bucket IDs"
  value       = module.s3.all_bucket_ids
}

output "app_service_account_role_arn" {
  description = "IAM role ARN for application service account"
  value       = module.iam.app_service_account_role_arn
}

output "cicd_role_arn" {
  description = "IAM role ARN for CI/CD"
  value       = module.iam.cicd_deployment_role_arn
}
