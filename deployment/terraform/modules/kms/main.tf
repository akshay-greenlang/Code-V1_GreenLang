# =============================================================================
# GreenLang KMS Module - Centralized Key Management (SEC-003)
# =============================================================================
# Provides Customer Master Keys (CMKs) for all GreenLang services.
# Supports encryption at rest for databases, storage, caches, and applications.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"

  tags = merge(var.tags, {
    Module      = "kms"
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
    PRD         = "SEC-003"
  })
}

# -----------------------------------------------------------------------------
# Master CMK - Root key for the GreenLang platform
# -----------------------------------------------------------------------------
resource "aws_kms_key" "master" {
  description              = "GreenLang Master CMK - ${local.name_prefix}"
  deletion_window_in_days  = var.deletion_window_days
  enable_key_rotation      = var.enable_key_rotation
  multi_region             = var.enable_multi_region
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  key_usage                = "ENCRYPT_DECRYPT"

  policy = data.aws_iam_policy_document.master_key_policy.json

  tags = merge(local.tags, {
    Name    = "${local.name_prefix}-master-cmk"
    KeyType = "master"
  })
}

resource "aws_kms_alias" "master" {
  name          = "alias/${local.name_prefix}-master"
  target_key_id = aws_kms_key.master.key_id
}

# -----------------------------------------------------------------------------
# Database CMK - Aurora PostgreSQL, RDS
# -----------------------------------------------------------------------------
resource "aws_kms_key" "database" {
  count = var.create_database_key ? 1 : 0

  description              = "GreenLang Database CMK - ${local.name_prefix}"
  deletion_window_in_days  = var.deletion_window_days
  enable_key_rotation      = var.enable_key_rotation
  multi_region             = var.enable_multi_region
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  key_usage                = "ENCRYPT_DECRYPT"

  policy = data.aws_iam_policy_document.database_key_policy[0].json

  tags = merge(local.tags, {
    Name    = "${local.name_prefix}-database-cmk"
    KeyType = "database"
  })
}

resource "aws_kms_alias" "database" {
  count = var.create_database_key ? 1 : 0

  name          = "alias/${local.name_prefix}-database"
  target_key_id = aws_kms_key.database[0].key_id
}

# -----------------------------------------------------------------------------
# Storage CMK - S3, EFS
# -----------------------------------------------------------------------------
resource "aws_kms_key" "storage" {
  count = var.create_storage_key ? 1 : 0

  description              = "GreenLang Storage CMK - ${local.name_prefix}"
  deletion_window_in_days  = var.deletion_window_days
  enable_key_rotation      = var.enable_key_rotation
  multi_region             = var.enable_multi_region
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  key_usage                = "ENCRYPT_DECRYPT"

  policy = data.aws_iam_policy_document.storage_key_policy[0].json

  tags = merge(local.tags, {
    Name    = "${local.name_prefix}-storage-cmk"
    KeyType = "storage"
  })
}

resource "aws_kms_alias" "storage" {
  count = var.create_storage_key ? 1 : 0

  name          = "alias/${local.name_prefix}-storage"
  target_key_id = aws_kms_key.storage[0].key_id
}

# -----------------------------------------------------------------------------
# Cache CMK - ElastiCache Redis
# -----------------------------------------------------------------------------
resource "aws_kms_key" "cache" {
  count = var.create_cache_key ? 1 : 0

  description              = "GreenLang Cache CMK - ${local.name_prefix}"
  deletion_window_in_days  = var.deletion_window_days
  enable_key_rotation      = var.enable_key_rotation
  multi_region             = var.enable_multi_region
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  key_usage                = "ENCRYPT_DECRYPT"

  policy = data.aws_iam_policy_document.cache_key_policy[0].json

  tags = merge(local.tags, {
    Name    = "${local.name_prefix}-cache-cmk"
    KeyType = "cache"
  })
}

resource "aws_kms_alias" "cache" {
  count = var.create_cache_key ? 1 : 0

  name          = "alias/${local.name_prefix}-cache"
  target_key_id = aws_kms_key.cache[0].key_id
}

# -----------------------------------------------------------------------------
# Secrets CMK - Secrets Manager, Parameter Store
# -----------------------------------------------------------------------------
resource "aws_kms_key" "secrets" {
  count = var.create_secrets_key ? 1 : 0

  description              = "GreenLang Secrets CMK - ${local.name_prefix}"
  deletion_window_in_days  = var.deletion_window_days
  enable_key_rotation      = var.enable_key_rotation
  multi_region             = var.enable_multi_region
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  key_usage                = "ENCRYPT_DECRYPT"

  policy = data.aws_iam_policy_document.secrets_key_policy[0].json

  tags = merge(local.tags, {
    Name    = "${local.name_prefix}-secrets-cmk"
    KeyType = "secrets"
  })
}

resource "aws_kms_alias" "secrets" {
  count = var.create_secrets_key ? 1 : 0

  name          = "alias/${local.name_prefix}-secrets"
  target_key_id = aws_kms_key.secrets[0].key_id
}

# -----------------------------------------------------------------------------
# Application CMK - Application-level DEKs for envelope encryption
# -----------------------------------------------------------------------------
resource "aws_kms_key" "application" {
  count = var.create_application_key ? 1 : 0

  description              = "GreenLang Application CMK - ${local.name_prefix}"
  deletion_window_in_days  = var.deletion_window_days
  enable_key_rotation      = var.enable_key_rotation
  multi_region             = var.enable_multi_region
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  key_usage                = "ENCRYPT_DECRYPT"

  policy = data.aws_iam_policy_document.application_key_policy[0].json

  tags = merge(local.tags, {
    Name    = "${local.name_prefix}-application-cmk"
    KeyType = "application"
  })
}

resource "aws_kms_alias" "application" {
  count = var.create_application_key ? 1 : 0

  name          = "alias/${local.name_prefix}-application"
  target_key_id = aws_kms_key.application[0].key_id
}

# -----------------------------------------------------------------------------
# EKS CMK - Kubernetes secrets envelope encryption
# -----------------------------------------------------------------------------
resource "aws_kms_key" "eks" {
  count = var.create_eks_key ? 1 : 0

  description              = "GreenLang EKS CMK - ${local.name_prefix}"
  deletion_window_in_days  = var.deletion_window_days
  enable_key_rotation      = var.enable_key_rotation
  multi_region             = var.enable_multi_region
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  key_usage                = "ENCRYPT_DECRYPT"

  policy = data.aws_iam_policy_document.eks_key_policy[0].json

  tags = merge(local.tags, {
    Name    = "${local.name_prefix}-eks-cmk"
    KeyType = "eks"
  })
}

resource "aws_kms_alias" "eks" {
  count = var.create_eks_key ? 1 : 0

  name          = "alias/${local.name_prefix}-eks"
  target_key_id = aws_kms_key.eks[0].key_id
}

# -----------------------------------------------------------------------------
# Backup CMK - Backup encryption for pgBackRest, Redis RDB, etc.
# -----------------------------------------------------------------------------
resource "aws_kms_key" "backup" {
  count = var.create_backup_key ? 1 : 0

  description              = "GreenLang Backup CMK - ${local.name_prefix}"
  deletion_window_in_days  = var.deletion_window_days
  enable_key_rotation      = var.enable_key_rotation
  multi_region             = var.enable_multi_region
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  key_usage                = "ENCRYPT_DECRYPT"

  policy = data.aws_iam_policy_document.backup_key_policy[0].json

  tags = merge(local.tags, {
    Name    = "${local.name_prefix}-backup-cmk"
    KeyType = "backup"
  })
}

resource "aws_kms_alias" "backup" {
  count = var.create_backup_key ? 1 : 0

  name          = "alias/${local.name_prefix}-backup"
  target_key_id = aws_kms_key.backup[0].key_id
}

# -----------------------------------------------------------------------------
# CloudWatch Logging for KMS
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "kms_audit" {
  count = var.enable_cloudwatch_logging ? 1 : 0

  name              = "/aws/kms/${local.name_prefix}"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.master.arn

  tags = local.tags
}

# -----------------------------------------------------------------------------
# Data sources
# -----------------------------------------------------------------------------
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_partition" "current" {}
