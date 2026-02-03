# GreenLang Production Environment
# Terraform configuration for the production environment
#
# IMPORTANT: Production environment requires careful review before applying changes.
#
# Prerequisites:
# 1. Create S3 bucket for state: aws s3 mb s3://greenlang-terraform-state-${AWS_ACCOUNT_ID}
# 2. Create DynamoDB table for locks: aws dynamodb create-table --table-name greenlang-terraform-locks ...
# 3. Update bucket name below with your AWS account ID
# 4. Create DR region S3 bucket for replication
# 5. Create KMS key in DR region
#
# Usage:
#   terraform init
#   terraform plan -var-file="terraform.tfvars"
#   terraform apply -var-file="terraform.tfvars"

# Backend configuration - Update bucket name with your AWS account ID
terraform {
  backend "s3" {
    bucket         = "greenlang-terraform-state"
    key            = "environments/prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "greenlang-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "prod"
      Project     = "GreenLang"
      ManagedBy   = "Terraform"
    }
  }
}

# DR Region Provider
provider "aws" {
  alias  = "dr"
  region = var.dr_region

  default_tags {
    tags = {
      Environment = "prod"
      Project     = "GreenLang"
      ManagedBy   = "Terraform"
      Purpose     = "DR"
    }
  }
}

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------
locals {
  name_prefix  = "greenlang-prod"
  cluster_name = "${local.name_prefix}-eks"

  common_tags = {
    Environment  = "prod"
    Project      = "GreenLang"
    CostCenter   = "production"
    Criticality  = "high"
    DataClass    = "confidential"
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
  flow_log_retention_days = 90
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

  enable_public_access = var.enable_eks_public_access
  public_access_cidrs  = var.eks_public_access_cidrs

  # System node group - production sizing
  system_node_instance_types = ["m6i.xlarge"]
  system_node_disk_size      = 100
  system_node_desired_size   = 3
  system_node_min_size       = 2
  system_node_max_size       = 5

  # API node group - production sizing for high traffic
  create_api_node_group    = true
  api_node_instance_types  = ["c6i.2xlarge"]
  api_node_disk_size       = 100
  api_node_desired_size    = 3
  api_node_min_size        = 3
  api_node_max_size         = 10

  # Agent runtime node group - mixed on-demand for reliability
  create_agent_node_group   = true
  agent_node_instance_types = ["c6i.xlarge", "c6a.xlarge", "c5.xlarge"]
  agent_node_capacity_type  = "ON_DEMAND"  # On-demand for production reliability
  agent_node_disk_size      = 100
  agent_node_desired_size   = 5
  agent_node_min_size       = 3
  agent_node_max_size       = 25

  enable_cluster_autoscaler       = true
  enable_load_balancer_controller = true
  enable_ebs_csi_driver           = true

  log_retention_days = 90

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
  instance_class = "db.r6g.xlarge"

  allocated_storage     = 200
  max_allocated_storage = 2000
  storage_type          = "gp3"
  iops                  = 12000

  database_name   = "greenlang_prod"
  master_username = "greenlang_admin"

  multi_az = true  # Always Multi-AZ for production

  backup_retention_period = 30
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"
  deletion_protection     = true

  performance_insights_enabled          = true
  performance_insights_retention_period = 31  # Full retention
  monitoring_interval                   = 15  # 15 second granularity

  max_connections = 1000

  read_replica_count     = 2  # Two read replicas for production
  replica_instance_class = "db.r6g.large"

  iam_database_authentication_enabled = true

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
  node_type      = "cache.r6g.xlarge"

  num_cache_clusters   = 3  # Primary + 2 replicas for production
  multi_az_enabled     = true

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  # Performance tuning
  maxmemory_policy       = "volatile-lru"
  enable_aof             = false  # Disable AOF for better performance
  notify_keyspace_events = "AKE"  # Enable keyspace notifications

  snapshot_retention_limit = 14
  snapshot_window          = "02:00-03:00"
  maintenance_window       = "sun:03:00-sun:04:00"
  skip_final_snapshot      = false

  allowed_security_groups = [module.eks.cluster_security_group_id]
  allowed_cidr_blocks     = module.vpc.private_subnet_cidrs

  enable_cloudwatch_alarms  = true
  alarm_actions             = var.alarm_sns_topic_arns
  cpu_threshold             = 70
  memory_threshold          = 70
  evictions_threshold       = 500
  connections_threshold     = 10000
  replication_lag_threshold = 10
  engine_cpu_threshold      = 85

  notification_topic_arn = var.elasticache_notification_topic_arn

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
  artifacts_noncurrent_version_expiration_days = 90

  create_logs_bucket   = true
  logs_retention_days  = 365
  enable_elb_logging   = true
  elb_account_id       = var.elb_account_id

  create_backups_bucket                      = true
  backups_retention_days                     = 2555  # 7 years for compliance
  backups_noncurrent_version_expiration_days = 365
  enable_backup_object_lock                  = true
  backup_object_lock_mode                    = "GOVERNANCE"
  backup_object_lock_days                    = 90

  create_data_bucket                      = true
  data_noncurrent_version_expiration_days = 90

  enable_cors          = true
  cors_allowed_origins = var.cors_allowed_origins

  create_static_assets_bucket  = true
  enable_static_website        = false
  static_assets_cors_origins   = var.cors_allowed_origins
  cloudfront_distribution_arn  = var.cloudfront_distribution_arn

  enable_replication                  = true
  replication_destination_bucket_arn  = var.dr_data_bucket_arn
  replication_destination_kms_key_arn = var.dr_kms_key_arn
  replication_destination_region      = var.dr_region

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
  agent_secrets_arns         = var.agent_secrets_arns

  enable_s3_access              = true
  s3_bucket_arns                = module.s3.all_bucket_arns
  enable_secrets_manager_access = true
  secrets_arns                  = var.app_secrets_arns
  enable_kms_access             = true
  kms_key_arns                  = [module.s3.kms_key_arn, module.rds.kms_key_arn, module.elasticache.kms_key_arn]

  enable_sqs_access     = true
  sqs_queue_arns        = var.sqs_queue_arns
  enable_sns_access     = true
  sns_topic_arns        = var.sns_topic_arns
  enable_bedrock_access = true

  create_cicd_role            = true
  create_github_oidc_provider = var.create_github_oidc_provider
  github_oidc_provider_arn    = var.github_oidc_provider_arn
  github_org                  = var.github_org
  github_repo                 = var.github_repo
  ecr_repository_arns         = var.ecr_repository_arns
  cicd_artifact_bucket_arns   = [module.s3.artifacts_bucket_arn]
  cicd_secrets_arns           = var.cicd_secrets_arns

  create_external_secrets_role     = true
  external_secrets_namespace       = "external-secrets"
  external_secrets_service_account = "external-secrets"
  external_secrets_allowed_arns    = var.external_secrets_allowed_arns

  create_monitoring_role     = true
  monitoring_namespace       = "monitoring"
  monitoring_service_account = "cloudwatch-agent"

  # Cross-account access for DR
  create_cross_account_role     = var.enable_cross_account_access
  trusted_account_ids           = var.trusted_account_ids
  require_mfa_for_cross_account = true
  cross_account_resource_arns   = var.cross_account_resource_arns

  tags = local.common_tags
}

# Outputs are defined in outputs.tf
