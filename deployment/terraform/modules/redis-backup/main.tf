# ==============================================================================
# Redis Backup Infrastructure - Terraform Module
# ==============================================================================
# This module creates AWS infrastructure for Redis backup and recovery:
# - S3 bucket with lifecycle rules for backup storage
# - IAM role and policies for backup access
# - KMS key for encryption at rest
# ==============================================================================

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# ==============================================================================
# Local Variables
# ==============================================================================

locals {
  bucket_name = "${var.project_name}-redis-backups-${var.environment}"

  # Standard tags for all resources
  common_tags = merge(var.tags, {
    Project     = var.project_name
    Environment = var.environment
    Component   = "redis-backup"
    ManagedBy   = "terraform"
  })

  # S3 bucket policy for backup access
  bucket_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EnforceTLS"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.redis_backup.arn,
          "${aws_s3_bucket.redis_backup.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
      {
        Sid    = "DenyIncorrectEncryptionHeader"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.redis_backup.arn}/*"
        Condition = {
          StringNotEquals = {
            "s3:x-amz-server-side-encryption" = var.enable_kms_encryption ? "aws:kms" : "AES256"
          }
        }
      }
    ]
  })
}

# ==============================================================================
# S3 Bucket for Redis Backups
# ==============================================================================

resource "aws_s3_bucket" "redis_backup" {
  bucket        = local.bucket_name
  force_destroy = var.force_destroy_bucket

  tags = merge(local.common_tags, {
    Name = local.bucket_name
    Type = "backup-storage"
  })
}

# Bucket versioning
resource "aws_s3_bucket_versioning" "redis_backup" {
  bucket = aws_s3_bucket.redis_backup.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

# Server-side encryption configuration
resource "aws_s3_bucket_server_side_encryption_configuration" "redis_backup" {
  bucket = aws_s3_bucket.redis_backup.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.enable_kms_encryption ? "aws:kms" : "AES256"
      kms_master_key_id = var.enable_kms_encryption ? aws_kms_key.redis_backup[0].arn : null
    }
    bucket_key_enabled = var.enable_kms_encryption
  }
}

# Public access block
resource "aws_s3_bucket_public_access_block" "redis_backup" {
  bucket = aws_s3_bucket.redis_backup.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket policy
resource "aws_s3_bucket_policy" "redis_backup" {
  bucket = aws_s3_bucket.redis_backup.id
  policy = local.bucket_policy

  depends_on = [aws_s3_bucket_public_access_block.redis_backup]
}

# Lifecycle rules for backup retention
resource "aws_s3_bucket_lifecycle_configuration" "redis_backup" {
  bucket = aws_s3_bucket.redis_backup.id

  # RDB backups lifecycle
  rule {
    id     = "rdb-backup-lifecycle"
    status = "Enabled"

    filter {
      prefix = "${var.backup_prefix}/rdb/"
    }

    # Transition to Intelligent-Tiering after 7 days
    transition {
      days          = 7
      storage_class = "INTELLIGENT_TIERING"
    }

    # Transition to Glacier after 30 days
    transition {
      days          = var.rdb_glacier_transition_days
      storage_class = "GLACIER"
    }

    # Delete after retention period
    expiration {
      days = var.rdb_retention_days
    }

    # Clean up incomplete multipart uploads
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }

    # Clean up old versions
    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_retention_days
    }
  }

  # AOF backups lifecycle (shorter retention)
  rule {
    id     = "aof-backup-lifecycle"
    status = "Enabled"

    filter {
      prefix = "${var.backup_prefix}/aof/"
    }

    # Delete AOF backups after shorter retention
    expiration {
      days = var.aof_retention_days
    }

    # Clean up incomplete multipart uploads
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }

    # Clean up old versions
    noncurrent_version_expiration {
      noncurrent_days = 1
    }
  }

  # Archive long-term backups
  rule {
    id     = "archive-lifecycle"
    status = var.enable_archive_tier ? "Enabled" : "Disabled"

    filter {
      prefix = "${var.backup_prefix}/archive/"
    }

    # Transition to Deep Archive for long-term storage
    transition {
      days          = 1
      storage_class = "DEEP_ARCHIVE"
    }

    # Retain archives for compliance period
    expiration {
      days = var.archive_retention_days
    }
  }
}

# Bucket notification for backup events (optional)
resource "aws_s3_bucket_notification" "redis_backup" {
  count  = var.enable_sns_notifications ? 1 : 0
  bucket = aws_s3_bucket.redis_backup.id

  topic {
    topic_arn = var.sns_topic_arn
    events = [
      "s3:ObjectCreated:*",
      "s3:ObjectRemoved:*"
    ]
    filter_prefix = "${var.backup_prefix}/"
  }
}

# ==============================================================================
# KMS Key for Encryption
# ==============================================================================

resource "aws_kms_key" "redis_backup" {
  count = var.enable_kms_encryption ? 1 : 0

  description             = "KMS key for Redis backup encryption - ${var.project_name}-${var.environment}"
  deletion_window_in_days = var.kms_deletion_window_days
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Id      = "redis-backup-key-policy"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow Backup Role Usage"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.redis_backup.arn
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      },
      {
        Sid    = "Allow S3 Service"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey*",
          "kms:Decrypt"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-redis-backup-kms-${var.environment}"
    Type = "encryption-key"
  })
}

resource "aws_kms_alias" "redis_backup" {
  count = var.enable_kms_encryption ? 1 : 0

  name          = "alias/${var.project_name}-redis-backup-${var.environment}"
  target_key_id = aws_kms_key.redis_backup[0].key_id
}

# ==============================================================================
# IAM Role for Backup Jobs
# ==============================================================================

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# IAM role for backup jobs
resource "aws_iam_role" "redis_backup" {
  name = "${var.project_name}-redis-backup-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
      # For EKS IRSA (IAM Roles for Service Accounts)
      {
        Effect = "Allow"
        Principal = {
          Federated = var.eks_oidc_provider_arn != "" ? var.eks_oidc_provider_arn : "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = var.eks_oidc_provider_arn != "" ? {
          StringEquals = {
            "${replace(var.eks_oidc_provider_arn, "/^arn:aws:iam::\\d+:oidc-provider\\//", "")}:sub" = "system:serviceaccount:${var.kubernetes_namespace}:redis-backup-sa"
          }
        } : {}
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-redis-backup-role-${var.environment}"
    Type = "iam-role"
  })
}

# S3 access policy
resource "aws_iam_role_policy" "redis_backup_s3" {
  name = "${var.project_name}-redis-backup-s3-policy"
  role = aws_iam_role.redis_backup.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:GetBucketVersioning"
        ]
        Resource = aws_s3_bucket.redis_backup.arn
      },
      {
        Sid    = "S3ObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:DeleteObjectVersion",
          "s3:ListMultipartUploadParts",
          "s3:AbortMultipartUpload"
        ]
        Resource = "${aws_s3_bucket.redis_backup.arn}/*"
      }
    ]
  })
}

# KMS access policy
resource "aws_iam_role_policy" "redis_backup_kms" {
  count = var.enable_kms_encryption ? 1 : 0

  name = "${var.project_name}-redis-backup-kms-policy"
  role = aws_iam_role.redis_backup.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.redis_backup[0].arn
      }
    ]
  })
}

# CloudWatch Logs access for backup job logging
resource "aws_iam_role_policy" "redis_backup_cloudwatch" {
  count = var.enable_cloudwatch_logs ? 1 : 0

  name = "${var.project_name}-redis-backup-cloudwatch-policy"
  role = aws_iam_role.redis_backup.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogsAccess"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/redis-backup/*",
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/redis-backup/*:*"
        ]
      }
    ]
  })
}

# Secrets Manager access for credentials
resource "aws_iam_role_policy" "redis_backup_secrets" {
  count = var.enable_secrets_manager_access ? 1 : 0

  name = "${var.project_name}-redis-backup-secrets-policy"
  role = aws_iam_role.redis_backup.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerAccess"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/redis*",
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/redis-backup*"
        ]
      }
    ]
  })
}

# Instance profile for EC2-based backup jobs
resource "aws_iam_instance_profile" "redis_backup" {
  name = "${var.project_name}-redis-backup-profile-${var.environment}"
  role = aws_iam_role.redis_backup.name

  tags = local.common_tags
}

# ==============================================================================
# CloudWatch Log Group
# ==============================================================================

resource "aws_cloudwatch_log_group" "redis_backup" {
  count = var.enable_cloudwatch_logs ? 1 : 0

  name              = "/aws/redis-backup/${var.project_name}/${var.environment}"
  retention_in_days = var.log_retention_days

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-redis-backup-logs-${var.environment}"
    Type = "log-group"
  })
}

# ==============================================================================
# CloudWatch Alarms
# ==============================================================================

resource "aws_cloudwatch_metric_alarm" "backup_failed" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.project_name}-redis-backup-failed-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BackupFailed"
  namespace           = "RedisBackup"
  period              = 3600  # 1 hour
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Redis backup job failed"
  treat_missing_data  = "notBreaching"

  dimensions = {
    Project     = var.project_name
    Environment = var.environment
  }

  alarm_actions = var.alarm_sns_topic_arns
  ok_actions    = var.alarm_sns_topic_arns

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "backup_size_anomaly" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.project_name}-redis-backup-size-anomaly-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BackupSizeBytes"
  namespace           = "RedisBackup"
  period              = 21600  # 6 hours
  statistic           = "Average"
  threshold           = var.min_backup_size_bytes
  alarm_description   = "Redis backup size is unusually small"
  treat_missing_data  = "breaching"

  dimensions = {
    Project     = var.project_name
    Environment = var.environment
  }

  alarm_actions = var.alarm_sns_topic_arns

  tags = local.common_tags
}
