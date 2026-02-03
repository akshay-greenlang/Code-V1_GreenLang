# RDS Backup Policy for GreenLang
# INFRA-001: Automated Database Backup Configuration
# Version: 1.0.0

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "backup_retention_period" {
  description = "Number of days to retain automated backups"
  type        = number
  default     = 35
}

variable "backup_window" {
  description = "Preferred backup window (UTC)"
  type        = string
  default     = "03:00-04:00"
}

variable "maintenance_window" {
  description = "Preferred maintenance window (UTC)"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "kms_key_arn" {
  description = "KMS key ARN for backup encryption"
  type        = string
  default     = ""
}

variable "cross_region_backup_region" {
  description = "Region for cross-region backup replication"
  type        = string
  default     = "eu-west-1"
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region automated backups"
  type        = bool
  default     = true
}

variable "db_instance_identifier" {
  description = "RDS instance identifier"
  type        = string
  default     = "greenlang-postgres"
}

# =============================================================================
# KMS Key for Backup Encryption
# =============================================================================

resource "aws_kms_key" "rds_backup" {
  description             = "KMS key for GreenLang RDS backup encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  multi_region            = var.enable_cross_region_backup

  policy = jsonencode({
    Version = "2012-10-17"
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
        Sid    = "Allow RDS to use the key"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:CallerAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })

  tags = {
    Name        = "greenlang-rds-backup-key"
    Environment = var.environment
    Purpose     = "rds-backup-encryption"
    ManagedBy   = "terraform"
  }
}

resource "aws_kms_alias" "rds_backup" {
  name          = "alias/greenlang-rds-backup-${var.environment}"
  target_key_id = aws_kms_key.rds_backup.key_id
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_db_instance" "greenlang" {
  db_instance_identifier = var.db_instance_identifier
}

# =============================================================================
# RDS Instance Backup Configuration
# =============================================================================

# Note: This modifies the existing RDS instance backup settings
# In production, backup settings should be part of the main RDS module

resource "aws_db_instance" "greenlang_backup_config" {
  # Reference existing instance - this is a partial config
  # In practice, merge with your main RDS terraform config
  count = 0 # Disabled - use as reference only

  identifier = var.db_instance_identifier

  # Backup Configuration
  backup_retention_period = var.backup_retention_period
  backup_window           = var.backup_window
  maintenance_window      = var.maintenance_window

  # Encryption
  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds_backup.arn

  # Deletion Protection
  deletion_protection = true

  # Final Snapshot
  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.db_instance_identifier}-final-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  # Copy tags to snapshots
  copy_tags_to_snapshot = true

  # Enable automated backups
  backup_target = "region"

  tags = {
    Name        = "greenlang-postgres"
    Environment = var.environment
    BackupPolicy = "automated"
    ManagedBy   = "terraform"
  }
}

# =============================================================================
# Cross-Region Automated Backup Replication
# =============================================================================

resource "aws_db_instance_automated_backups_replication" "greenlang" {
  count = var.enable_cross_region_backup ? 1 : 0

  source_db_instance_arn = data.aws_db_instance.greenlang.db_instance_arn
  retention_period       = var.backup_retention_period
  kms_key_id            = aws_kms_key.rds_backup_replica[0].arn

  provider = aws.dr_region
}

# KMS Key in DR Region for encrypted backup replication
resource "aws_kms_key" "rds_backup_replica" {
  count = var.enable_cross_region_backup ? 1 : 0

  provider = aws.dr_region

  description             = "KMS key for GreenLang RDS backup replication"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Name        = "greenlang-rds-backup-replica-key"
    Environment = var.environment
    Purpose     = "rds-backup-replication"
    ManagedBy   = "terraform"
  }
}

# =============================================================================
# Manual Snapshot Management
# =============================================================================

# Lambda function for creating manual snapshots
resource "aws_lambda_function" "rds_snapshot" {
  filename         = "${path.module}/lambda/rds-snapshot.zip"
  function_name    = "greenlang-rds-manual-snapshot"
  role             = aws_iam_role.lambda_rds_snapshot.arn
  handler          = "index.handler"
  runtime          = "python3.11"
  timeout          = 300
  memory_size      = 256

  environment {
    variables = {
      DB_INSTANCE_IDENTIFIER = var.db_instance_identifier
      ENVIRONMENT           = var.environment
      RETENTION_DAYS        = "90"
      SNS_TOPIC_ARN         = aws_sns_topic.backup_notifications.arn
    }
  }

  tags = {
    Name        = "greenlang-rds-snapshot-lambda"
    Environment = var.environment
    Purpose     = "backup-automation"
  }
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_rds_snapshot" {
  name = "greenlang-rds-snapshot-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "greenlang-rds-snapshot-lambda-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy" "lambda_rds_snapshot" {
  name = "greenlang-rds-snapshot-policy"
  role = aws_iam_role.lambda_rds_snapshot.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "rds:CreateDBSnapshot",
          "rds:DeleteDBSnapshot",
          "rds:DescribeDBSnapshots",
          "rds:DescribeDBInstances",
          "rds:AddTagsToResource",
          "rds:ListTagsForResource",
          "rds:CopyDBSnapshot"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey",
          "kms:CreateGrant"
        ]
        Resource = aws_kms_key.rds_backup.arn
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.backup_notifications.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# =============================================================================
# Scheduled Snapshot Events
# =============================================================================

# Weekly snapshot schedule
resource "aws_cloudwatch_event_rule" "weekly_snapshot" {
  name                = "greenlang-rds-weekly-snapshot"
  description         = "Trigger weekly RDS snapshot"
  schedule_expression = "cron(0 5 ? * SUN *)"

  tags = {
    Name        = "greenlang-rds-weekly-snapshot"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_event_target" "weekly_snapshot" {
  rule      = aws_cloudwatch_event_rule.weekly_snapshot.name
  target_id = "RDSWeeklySnapshot"
  arn       = aws_lambda_function.rds_snapshot.arn

  input = jsonencode({
    snapshot_type = "weekly"
    retention_days = 90
  })
}

resource "aws_lambda_permission" "weekly_snapshot" {
  statement_id  = "AllowWeeklySnapshot"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rds_snapshot.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_snapshot.arn
}

# Monthly snapshot schedule
resource "aws_cloudwatch_event_rule" "monthly_snapshot" {
  name                = "greenlang-rds-monthly-snapshot"
  description         = "Trigger monthly RDS snapshot"
  schedule_expression = "cron(0 6 1 * ? *)"

  tags = {
    Name        = "greenlang-rds-monthly-snapshot"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_event_target" "monthly_snapshot" {
  rule      = aws_cloudwatch_event_rule.monthly_snapshot.name
  target_id = "RDSMonthlySnapshot"
  arn       = aws_lambda_function.rds_snapshot.arn

  input = jsonencode({
    snapshot_type = "monthly"
    retention_days = 365
  })
}

resource "aws_lambda_permission" "monthly_snapshot" {
  statement_id  = "AllowMonthlySnapshot"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rds_snapshot.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.monthly_snapshot.arn
}

# =============================================================================
# Snapshot Cleanup (Delete old snapshots)
# =============================================================================

resource "aws_lambda_function" "rds_snapshot_cleanup" {
  filename         = "${path.module}/lambda/rds-snapshot-cleanup.zip"
  function_name    = "greenlang-rds-snapshot-cleanup"
  role             = aws_iam_role.lambda_rds_snapshot.arn
  handler          = "index.handler"
  runtime          = "python3.11"
  timeout          = 300
  memory_size      = 256

  environment {
    variables = {
      DB_INSTANCE_IDENTIFIER = var.db_instance_identifier
      ENVIRONMENT           = var.environment
      WEEKLY_RETENTION_DAYS = "90"
      MONTHLY_RETENTION_DAYS = "365"
      SNS_TOPIC_ARN         = aws_sns_topic.backup_notifications.arn
    }
  }

  tags = {
    Name        = "greenlang-rds-snapshot-cleanup-lambda"
    Environment = var.environment
    Purpose     = "backup-automation"
  }
}

resource "aws_cloudwatch_event_rule" "snapshot_cleanup" {
  name                = "greenlang-rds-snapshot-cleanup"
  description         = "Cleanup old RDS snapshots"
  schedule_expression = "cron(0 7 * * ? *)"

  tags = {
    Name        = "greenlang-rds-snapshot-cleanup"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_event_target" "snapshot_cleanup" {
  rule      = aws_cloudwatch_event_rule.snapshot_cleanup.name
  target_id = "RDSSnapshotCleanup"
  arn       = aws_lambda_function.rds_snapshot_cleanup.arn
}

resource "aws_lambda_permission" "snapshot_cleanup" {
  statement_id  = "AllowSnapshotCleanup"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rds_snapshot_cleanup.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.snapshot_cleanup.arn
}

# =============================================================================
# SNS Notifications
# =============================================================================

resource "aws_sns_topic" "backup_notifications" {
  name = "greenlang-rds-backup-notifications"

  tags = {
    Name        = "greenlang-rds-backup-notifications"
    Environment = var.environment
    Purpose     = "backup-alerts"
  }
}

resource "aws_sns_topic_policy" "backup_notifications" {
  arn = aws_sns_topic.backup_notifications.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowRDSEvents"
        Effect = "Allow"
        Principal = {
          Service = "events.rds.amazonaws.com"
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.backup_notifications.arn
      },
      {
        Sid    = "AllowLambda"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.lambda_rds_snapshot.arn
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.backup_notifications.arn
      }
    ]
  })
}

# RDS Event Subscription for backup events
resource "aws_db_event_subscription" "backup_events" {
  name      = "greenlang-rds-backup-events"
  sns_topic = aws_sns_topic.backup_notifications.arn

  source_type = "db-instance"
  source_ids  = [var.db_instance_identifier]

  event_categories = [
    "backup",
    "recovery",
    "restoration",
    "notification"
  ]

  tags = {
    Name        = "greenlang-rds-backup-events"
    Environment = var.environment
  }
}

# =============================================================================
# CloudWatch Alarms
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "backup_storage" {
  alarm_name          = "greenlang-rds-backup-storage-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TotalBackupStorageBilled"
  namespace           = "AWS/RDS"
  period              = 86400
  statistic           = "Maximum"
  threshold           = 100 * 1024 * 1024 * 1024 # 100 GB

  dimensions = {
    DBInstanceIdentifier = var.db_instance_identifier
  }

  alarm_description = "RDS backup storage exceeds 100 GB"
  alarm_actions     = [aws_sns_topic.backup_notifications.arn]
  ok_actions        = [aws_sns_topic.backup_notifications.arn]

  tags = {
    Name        = "greenlang-rds-backup-storage-alarm"
    Environment = var.environment
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "kms_key_arn" {
  description = "KMS key ARN for RDS backup encryption"
  value       = aws_kms_key.rds_backup.arn
}

output "kms_key_id" {
  description = "KMS key ID for RDS backup encryption"
  value       = aws_kms_key.rds_backup.key_id
}

output "sns_topic_arn" {
  description = "SNS topic ARN for backup notifications"
  value       = aws_sns_topic.backup_notifications.arn
}

output "lambda_snapshot_arn" {
  description = "Lambda function ARN for manual snapshots"
  value       = aws_lambda_function.rds_snapshot.arn
}

output "backup_configuration" {
  description = "Summary of backup configuration"
  value = {
    retention_period       = var.backup_retention_period
    backup_window          = var.backup_window
    maintenance_window     = var.maintenance_window
    cross_region_enabled   = var.enable_cross_region_backup
    cross_region_location  = var.enable_cross_region_backup ? var.cross_region_backup_region : "disabled"
    encryption_enabled     = true
    kms_key_alias          = aws_kms_alias.rds_backup.name
  }
}
