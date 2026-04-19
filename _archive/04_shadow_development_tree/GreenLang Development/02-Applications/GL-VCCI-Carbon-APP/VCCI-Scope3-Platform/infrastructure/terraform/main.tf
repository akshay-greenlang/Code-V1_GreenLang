# Root Module - Main Configuration
# GL-VCCI Scope 3 Carbon Intelligence Platform

# ============================================================================
# Provider Configuration
# ============================================================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge(
      var.tags,
      {
        Environment = var.environment
        Terraform   = "true"
      }
    )
  }
}

provider "aws" {
  alias  = "secondary"
  region = var.aws_region_secondary

  default_tags {
    tags = merge(
      var.tags,
      {
        Environment = var.environment
        Terraform   = "true"
        Region      = "secondary"
      }
    )
  }
}

# ============================================================================
# Data Sources
# ============================================================================

data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

# ============================================================================
# Local Variables
# ============================================================================

locals {
  cluster_name = "${var.project_name}-${var.environment}"

  common_tags = merge(
    var.tags,
    {
      Environment = var.environment
      Region      = var.aws_region
      ManagedBy   = "Terraform"
    }
  )

  # Subnet CIDR calculations
  public_subnet_cidrs = [
    cidrsubnet(var.vpc_cidr, 8, 1),
    cidrsubnet(var.vpc_cidr, 8, 2),
    cidrsubnet(var.vpc_cidr, 8, 3),
  ]

  private_subnet_cidrs = [
    cidrsubnet(var.vpc_cidr, 8, 11),
    cidrsubnet(var.vpc_cidr, 8, 12),
    cidrsubnet(var.vpc_cidr, 8, 13),
  ]

  database_subnet_cidrs = [
    cidrsubnet(var.vpc_cidr, 8, 21),
    cidrsubnet(var.vpc_cidr, 8, 22),
    cidrsubnet(var.vpc_cidr, 8, 23),
  ]

  elasticache_subnet_cidrs = [
    cidrsubnet(var.vpc_cidr, 8, 31),
    cidrsubnet(var.vpc_cidr, 8, 32),
    cidrsubnet(var.vpc_cidr, 8, 33),
  ]
}

# ============================================================================
# KMS Keys
# ============================================================================

resource "aws_kms_key" "main" {
  description             = "KMS key for ${local.cluster_name}"
  deletion_window_in_days = var.kms_key_deletion_window
  enable_key_rotation     = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.cluster_name}-kms"
    }
  )
}

resource "aws_kms_alias" "main" {
  name          = "alias/${local.cluster_name}"
  target_key_id = aws_kms_key.main.key_id
}

# ============================================================================
# VPC Module
# ============================================================================

module "vpc" {
  source = "./modules/vpc"

  project_name             = var.project_name
  environment              = var.environment
  vpc_cidr                 = var.vpc_cidr
  availability_zones       = var.availability_zones
  public_subnet_cidrs      = local.public_subnet_cidrs
  private_subnet_cidrs     = local.private_subnet_cidrs
  database_subnet_cidrs    = local.database_subnet_cidrs
  elasticache_subnet_cidrs = local.elasticache_subnet_cidrs
  enable_nat_gateway       = var.enable_nat_gateway
  single_nat_gateway       = var.single_nat_gateway
  enable_vpc_endpoints     = var.enable_vpc_endpoints

  tags = local.common_tags
}

# ============================================================================
# IAM Module
# ============================================================================

module "iam" {
  source = "./modules/iam"

  project_name    = var.project_name
  environment     = var.environment
  cluster_name    = local.cluster_name
  oidc_issuer_url = module.eks.cluster_oidc_issuer_url

  # S3 bucket ARNs
  s3_bucket_arns = [
    module.s3.provenance_bucket_arn,
    module.s3.raw_data_bucket_arn,
    module.s3.reports_bucket_arn,
  ]

  tags = local.common_tags

  depends_on = [module.eks]
}

# ============================================================================
# EKS Module
# ============================================================================

module "eks" {
  source = "./modules/eks"

  project_name                         = var.project_name
  environment                          = var.environment
  cluster_name                         = local.cluster_name
  cluster_version                      = var.eks_cluster_version
  vpc_id                               = module.vpc.vpc_id
  subnet_ids                           = module.vpc.private_subnet_ids
  cluster_endpoint_private_access      = var.eks_cluster_endpoint_private_access
  cluster_endpoint_public_access       = var.eks_cluster_endpoint_public_access
  cluster_endpoint_public_access_cidrs = var.eks_cluster_endpoint_public_access_cidrs

  # Node group configurations
  compute_node_group_config = {
    instance_types = var.compute_node_group_instance_types
    desired_size   = var.compute_node_group_desired_size
    min_size       = var.compute_node_group_min_size
    max_size       = var.compute_node_group_max_size
  }

  memory_node_group_config = {
    instance_types = var.memory_node_group_instance_types
    desired_size   = var.memory_node_group_desired_size
    min_size       = var.memory_node_group_min_size
    max_size       = var.memory_node_group_max_size
  }

  gpu_node_group_config = {
    instance_types = var.gpu_node_group_instance_types
    desired_size   = var.gpu_node_group_desired_size
    min_size       = var.gpu_node_group_min_size
    max_size       = var.gpu_node_group_max_size
  }

  kms_key_arn = aws_kms_key.main.arn

  tags = local.common_tags

  depends_on = [module.vpc]
}

# ============================================================================
# RDS Module
# ============================================================================

module "rds" {
  source = "./modules/rds"

  project_name                          = var.project_name
  environment                           = var.environment
  cluster_identifier                    = local.cluster_name
  engine_version                        = var.rds_engine_version
  instance_class                        = var.rds_instance_class
  allocated_storage                     = var.rds_allocated_storage
  max_allocated_storage                 = var.rds_max_allocated_storage
  storage_type                          = var.rds_storage_type
  iops                                  = var.rds_iops
  storage_throughput                    = var.rds_storage_throughput
  multi_az                              = var.rds_multi_az
  database_name                         = var.rds_database_name
  master_username                       = var.rds_master_username
  backup_retention_period               = var.rds_backup_retention_period
  backup_window                         = var.rds_backup_window
  maintenance_window                    = var.rds_maintenance_window
  deletion_protection                   = var.rds_deletion_protection
  performance_insights_enabled          = var.rds_performance_insights_enabled
  performance_insights_retention_period = var.rds_performance_insights_retention_period
  read_replica_count                    = var.rds_read_replica_count
  vpc_id                                = module.vpc.vpc_id
  subnet_ids                            = module.vpc.database_subnet_ids
  allowed_security_groups               = [module.eks.cluster_security_group_id]
  kms_key_arn                           = aws_kms_key.main.arn

  tags = local.common_tags

  depends_on = [module.vpc, module.eks]
}

# ============================================================================
# ElastiCache Module
# ============================================================================

module "elasticache" {
  source = "./modules/elasticache"

  project_name                    = var.project_name
  environment                     = var.environment
  cluster_id                      = local.cluster_name
  engine_version                  = var.elasticache_engine_version
  node_type                       = var.elasticache_node_type
  num_node_groups                 = var.elasticache_num_node_groups
  replicas_per_node_group         = var.elasticache_replicas_per_node_group
  automatic_failover_enabled      = var.elasticache_automatic_failover_enabled
  multi_az_enabled                = var.elasticache_multi_az_enabled
  at_rest_encryption_enabled      = var.elasticache_at_rest_encryption_enabled
  transit_encryption_enabled      = var.elasticache_transit_encryption_enabled
  snapshot_retention_limit        = var.elasticache_snapshot_retention_limit
  snapshot_window                 = var.elasticache_snapshot_window
  maintenance_window              = var.elasticache_maintenance_window
  vpc_id                          = module.vpc.vpc_id
  subnet_ids                      = module.vpc.private_subnet_ids
  allowed_security_groups         = [module.eks.cluster_security_group_id]
  kms_key_arn                     = aws_kms_key.main.arn

  tags = local.common_tags

  depends_on = [module.vpc, module.eks]
}

# ============================================================================
# S3 Module
# ============================================================================

module "s3" {
  source = "./modules/s3"

  providers = {
    aws           = aws
    aws.secondary = aws.secondary
  }

  project_name                     = var.project_name
  environment                      = var.environment
  enable_versioning                = var.s3_enable_versioning
  enable_replication               = var.s3_enable_replication
  replication_region               = var.aws_region_secondary
  lifecycle_glacier_transition_days = var.s3_lifecycle_glacier_transition_days
  lifecycle_ia_transition_days     = var.s3_lifecycle_ia_transition_days
  lifecycle_expiration_days        = var.s3_lifecycle_expiration_days
  kms_key_arn                      = aws_kms_key.main.arn

  tags = local.common_tags
}

# ============================================================================
# Monitoring Module
# ============================================================================

module "monitoring" {
  source = "./modules/monitoring"

  project_name               = var.project_name
  environment                = var.environment
  enable_cloudwatch_logs     = var.enable_cloudwatch_logs
  log_retention_days         = var.cloudwatch_log_retention_days
  enable_cloudwatch_alarms   = var.enable_cloudwatch_alarms
  alarm_email_endpoints      = var.alarm_email_endpoints
  alarm_slack_webhook_url    = var.alarm_slack_webhook_url

  # Resource identifiers for monitoring
  eks_cluster_name           = module.eks.cluster_name
  rds_cluster_id             = module.rds.cluster_id
  elasticache_cluster_id     = module.elasticache.cluster_id

  tags = local.common_tags

  depends_on = [module.eks, module.rds, module.elasticache]
}

# ============================================================================
# Backup Module
# ============================================================================

module "backup" {
  source = "./modules/backup"

  project_name           = var.project_name
  environment            = var.environment
  enable_aws_backup      = var.enable_aws_backup
  backup_retention_days  = var.backup_retention_days
  backup_schedule        = var.backup_schedule
  kms_key_arn            = aws_kms_key.main.arn

  # Resources to backup
  rds_cluster_arn        = module.rds.cluster_arn

  tags = local.common_tags

  depends_on = [module.rds]
}
