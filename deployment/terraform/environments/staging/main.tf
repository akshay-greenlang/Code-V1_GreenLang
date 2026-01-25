# GreenLang Staging Environment
# Terraform configuration for the staging environment

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
    key            = "environments/staging/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "greenlang-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "staging"
      Project     = "GreenLang"
      ManagedBy   = "Terraform"
    }
  }
}

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------
locals {
  name_prefix  = "greenlang-staging"
  cluster_name = "${local.name_prefix}-eks"

  common_tags = {
    Environment = "staging"
    Project     = "GreenLang"
    CostCenter  = "staging"
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
  flow_log_retention_days = 30
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

  # System node group - production-like sizing
  system_node_instance_types = ["m6i.large"]
  system_node_disk_size      = 100
  system_node_desired_size   = 2
  system_node_min_size       = 2
  system_node_max_size       = 4

  # API node group - production-like sizing
  create_api_node_group    = true
  api_node_instance_types  = ["c6i.xlarge"]
  api_node_disk_size       = 100
  api_node_desired_size    = 2
  api_node_min_size        = 2
  api_node_max_size        = 6

  # Agent runtime node group - mixed on-demand/spot for cost efficiency
  create_agent_node_group   = true
  agent_node_instance_types = ["c6i.large", "c6a.large", "c5.large"]
  agent_node_capacity_type  = "SPOT"
  agent_node_disk_size      = 100
  agent_node_desired_size   = 3
  agent_node_min_size       = 2
  agent_node_max_size       = 10

  enable_cluster_autoscaler       = true
  enable_load_balancer_controller = true
  enable_ebs_csi_driver           = true

  log_retention_days = 30

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
  instance_class = "db.r6g.large"

  allocated_storage     = 100
  max_allocated_storage = 500

  database_name   = "greenlang_staging"
  master_username = "greenlang_admin"

  multi_az = true  # Enable Multi-AZ for staging

  backup_retention_period = 14
  deletion_protection     = true

  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  monitoring_interval                   = 30

  max_connections = 400

  read_replica_count = 1  # One read replica for staging

  allowed_security_groups = [module.eks.cluster_security_group_id]
  allowed_cidr_blocks     = module.vpc.private_subnet_cidrs

  alarm_actions = var.alarm_sns_topic_arns

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
  node_type      = "cache.r6g.large"

  num_cache_clusters   = 2  # Primary + 1 replica
  multi_az_enabled     = true

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = 7
  skip_final_snapshot      = false

  allowed_security_groups = [module.eks.cluster_security_group_id]
  allowed_cidr_blocks     = module.vpc.private_subnet_cidrs

  enable_cloudwatch_alarms = true
  alarm_actions            = var.alarm_sns_topic_arns
  cpu_threshold            = 75
  memory_threshold         = 75

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
  artifacts_noncurrent_version_expiration_days = 60

  create_logs_bucket   = true
  logs_retention_days  = 180
  enable_elb_logging   = true
  elb_account_id       = var.elb_account_id

  create_backups_bucket                      = true
  backups_retention_days                     = 365
  backups_noncurrent_version_expiration_days = 90
  enable_backup_object_lock                  = false

  create_data_bucket                      = true
  data_noncurrent_version_expiration_days = 60

  enable_cors          = true
  cors_allowed_origins = var.cors_allowed_origins

  create_static_assets_bucket = true
  enable_static_website       = false
  static_assets_cors_origins  = var.cors_allowed_origins

  enable_replication   = false  # No cross-region replication in staging
  enable_access_logging = true

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

  enable_sqs_access     = true
  sqs_queue_arns        = var.sqs_queue_arns
  enable_sns_access     = true
  sns_topic_arns        = var.sns_topic_arns
  enable_bedrock_access = true

  create_cicd_role          = true
  create_github_oidc_provider = var.create_github_oidc_provider
  github_oidc_provider_arn  = var.github_oidc_provider_arn
  github_org                = var.github_org
  github_repo               = var.github_repo
  ecr_repository_arns       = var.ecr_repository_arns
  cicd_artifact_bucket_arns = [module.s3.artifacts_bucket_arn]
  cicd_secrets_arns         = var.cicd_secrets_arns

  create_external_secrets_role     = true
  external_secrets_namespace       = "external-secrets"
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

output "eks_cluster_certificate_authority_data" {
  description = "EKS cluster CA data"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "rds_endpoint" {
  description = "RDS primary endpoint"
  value       = module.rds.endpoint
}

output "rds_replica_endpoints" {
  description = "RDS replica endpoints"
  value       = module.rds.replica_endpoints
}

output "redis_primary_endpoint" {
  description = "Redis primary endpoint"
  value       = module.elasticache.primary_endpoint_address
}

output "redis_reader_endpoint" {
  description = "Redis reader endpoint"
  value       = module.elasticache.reader_endpoint_address
}

output "s3_buckets" {
  description = "S3 bucket IDs"
  value       = module.s3.all_bucket_ids
}

output "app_service_account_role_arn" {
  description = "IAM role ARN for application service account"
  value       = module.iam.app_service_account_role_arn
}

output "agent_service_account_role_arn" {
  description = "IAM role ARN for agent service account"
  value       = module.iam.agent_service_account_role_arn
}

output "cicd_role_arn" {
  description = "IAM role ARN for CI/CD"
  value       = module.iam.cicd_deployment_role_arn
}

output "external_secrets_role_arn" {
  description = "IAM role ARN for External Secrets Operator"
  value       = module.iam.external_secrets_role_arn
}
