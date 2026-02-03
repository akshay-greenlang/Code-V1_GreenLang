#------------------------------------------------------------------------------
# S3 CloudTrail Integration Module - Main Configuration
# GreenLang Infrastructure
#------------------------------------------------------------------------------

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

#------------------------------------------------------------------------------
# Data Sources
#------------------------------------------------------------------------------

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_iam_policy_document" "cloudtrail_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["cloudtrail.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

data "aws_iam_policy_document" "cloudtrail_s3_policy" {
  statement {
    sid    = "AWSCloudTrailAclCheck"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["cloudtrail.amazonaws.com"]
    }

    actions   = ["s3:GetBucketAcl"]
    resources = [aws_s3_bucket.cloudtrail_logs.arn]

    condition {
      test     = "StringEquals"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudtrail:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:trail/greenlang-s3-data-trail"]
    }
  }

  statement {
    sid    = "AWSCloudTrailWrite"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["cloudtrail.amazonaws.com"]
    }

    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.cloudtrail_logs.arn}/AWSLogs/${data.aws_caller_identity.current.account_id}/*"]

    condition {
      test     = "StringEquals"
      variable = "s3:x-amz-acl"
      values   = ["bucket-owner-full-control"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudtrail:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:trail/greenlang-s3-data-trail"]
    }
  }
}

#------------------------------------------------------------------------------
# KMS Key for CloudTrail Encryption
#------------------------------------------------------------------------------

resource "aws_kms_key" "cloudtrail" {
  description             = "KMS key for GreenLang CloudTrail log encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  multi_region            = false

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
        Sid    = "Allow CloudTrail to encrypt logs"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:SourceArn" = "arn:aws:cloudtrail:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:trail/greenlang-s3-data-trail"
          }
          StringLike = {
            "kms:EncryptionContext:aws:cloudtrail:arn" = "arn:aws:cloudtrail:*:${data.aws_caller_identity.current.account_id}:trail/*"
          }
        }
      },
      {
        Sid    = "Allow CloudWatch Logs"
        Effect = "Allow"
        Principal = {
          Service = "logs.${data.aws_region.current.name}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt*",
          "kms:Decrypt*",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:Describe*"
        ]
        Resource = "*"
        Condition = {
          ArnLike = {
            "kms:EncryptionContext:aws:logs:arn" = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/cloudtrail/*"
          }
        }
      },
      {
        Sid    = "Allow principals to decrypt"
        Effect = "Allow"
        Principal = {
          AWS = var.authorized_principals
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, {
    Name        = "greenlang-cloudtrail-kms"
    Environment = var.environment
  })
}

resource "aws_kms_alias" "cloudtrail" {
  name          = "alias/greenlang-cloudtrail"
  target_key_id = aws_kms_key.cloudtrail.key_id
}

#------------------------------------------------------------------------------
# S3 Bucket for CloudTrail Logs
#------------------------------------------------------------------------------

resource "aws_s3_bucket" "cloudtrail_logs" {
  bucket = "greenlang-cloudtrail-logs-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name        = "greenlang-cloudtrail-logs"
    Environment = var.environment
    Compliance  = "required"
  })
}

resource "aws_s3_bucket_policy" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id
  policy = data.aws_iam_policy_document.cloudtrail_s3_policy.json
}

resource "aws_s3_bucket_versioning" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.cloudtrail.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    filter {
      prefix = "AWSLogs/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 180
      storage_class = "GLACIER"
    }

    expiration {
      days = var.log_retention_days
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

resource "aws_s3_bucket_logging" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  target_bucket = aws_s3_bucket.cloudtrail_logs.id
  target_prefix = "access-logs/"
}

#------------------------------------------------------------------------------
# CloudWatch Log Group for CloudTrail
#------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "cloudtrail" {
  name              = "/aws/cloudtrail/greenlang-s3-data-events"
  retention_in_days = var.cloudwatch_log_retention_days
  kms_key_id        = aws_kms_key.cloudtrail.arn

  tags = merge(var.tags, {
    Name        = "greenlang-cloudtrail-logs"
    Environment = var.environment
  })
}

#------------------------------------------------------------------------------
# IAM Role for CloudTrail to CloudWatch Logs
#------------------------------------------------------------------------------

resource "aws_iam_role" "cloudtrail_cloudwatch" {
  name               = "greenlang-cloudtrail-cloudwatch-role"
  assume_role_policy = data.aws_iam_policy_document.cloudtrail_assume_role.json

  tags = merge(var.tags, {
    Name        = "greenlang-cloudtrail-cloudwatch-role"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "cloudtrail_cloudwatch" {
  name = "greenlang-cloudtrail-cloudwatch-policy"
  role = aws_iam_role.cloudtrail_cloudwatch.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSCloudTrailCreateLogStream"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream"
        ]
        Resource = "${aws_cloudwatch_log_group.cloudtrail.arn}:*"
      },
      {
        Sid    = "AWSCloudTrailPutLogEvents"
        Effect = "Allow"
        Action = [
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.cloudtrail.arn}:*"
      }
    ]
  })
}

#------------------------------------------------------------------------------
# SNS Topic for CloudTrail Notifications
#------------------------------------------------------------------------------

resource "aws_sns_topic" "cloudtrail_notifications" {
  name              = "greenlang-cloudtrail-notifications"
  display_name      = "GreenLang CloudTrail Notifications"
  kms_master_key_id = aws_kms_key.cloudtrail.id

  tags = merge(var.tags, {
    Name        = "greenlang-cloudtrail-notifications"
    Environment = var.environment
  })
}

resource "aws_sns_topic_policy" "cloudtrail_notifications" {
  arn = aws_sns_topic.cloudtrail_notifications.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCloudTrailPublish"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.cloudtrail_notifications.arn
        Condition = {
          StringEquals = {
            "aws:SourceArn" = "arn:aws:cloudtrail:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:trail/greenlang-s3-data-trail"
          }
        }
      }
    ]
  })
}

#------------------------------------------------------------------------------
# CloudTrail Trail for S3 Data Events
#------------------------------------------------------------------------------

resource "aws_cloudtrail" "s3_data_events" {
  name                          = "greenlang-s3-data-trail"
  s3_bucket_name                = aws_s3_bucket.cloudtrail_logs.id
  s3_key_prefix                 = "AWSLogs"
  include_global_service_events = false
  is_multi_region_trail         = var.multi_region_trail
  enable_logging                = true

  kms_key_id = aws_kms_key.cloudtrail.arn

  cloud_watch_logs_group_arn = "${aws_cloudwatch_log_group.cloudtrail.arn}:*"
  cloud_watch_logs_role_arn  = aws_iam_role.cloudtrail_cloudwatch.arn

  sns_topic_name = aws_sns_topic.cloudtrail_notifications.arn

  enable_log_file_validation = true

  # Event selectors for S3 data events
  dynamic "event_selector" {
    for_each = var.monitored_buckets
    content {
      read_write_type           = event_selector.value.read_write_type
      include_management_events = false

      data_resource {
        type   = "AWS::S3::Object"
        values = ["arn:aws:s3:::${event_selector.value.bucket_name}/"]
      }
    }
  }

  # Advanced event selector for more granular control
  dynamic "advanced_event_selector" {
    for_each = var.enable_advanced_selectors ? var.advanced_event_selectors : []
    content {
      name = advanced_event_selector.value.name

      dynamic "field_selector" {
        for_each = advanced_event_selector.value.field_selectors
        content {
          field           = field_selector.value.field
          equals          = lookup(field_selector.value, "equals", null)
          not_equals      = lookup(field_selector.value, "not_equals", null)
          starts_with     = lookup(field_selector.value, "starts_with", null)
          not_starts_with = lookup(field_selector.value, "not_starts_with", null)
          ends_with       = lookup(field_selector.value, "ends_with", null)
          not_ends_with   = lookup(field_selector.value, "not_ends_with", null)
        }
      }
    }
  }

  depends_on = [
    aws_s3_bucket_policy.cloudtrail_logs,
    aws_iam_role_policy.cloudtrail_cloudwatch
  ]

  tags = merge(var.tags, {
    Name        = "greenlang-s3-data-trail"
    Environment = var.environment
    Compliance  = "required"
  })
}

#------------------------------------------------------------------------------
# CloudTrail Insights (Optional)
#------------------------------------------------------------------------------

resource "aws_cloudtrail_event_data_store" "insights" {
  count = var.enable_insights ? 1 : 0

  name                           = "greenlang-s3-insights"
  retention_period               = var.insights_retention_days
  multi_region_enabled           = var.multi_region_trail
  organization_enabled           = false
  termination_protection_enabled = var.environment == "prod"
  kms_key_id                     = aws_kms_key.cloudtrail.arn

  advanced_event_selector {
    name = "Log all S3 data events"

    field_selector {
      field  = "eventCategory"
      equals = ["Data"]
    }

    field_selector {
      field  = "resources.type"
      equals = ["AWS::S3::Object"]
    }
  }

  tags = merge(var.tags, {
    Name        = "greenlang-s3-insights"
    Environment = var.environment
  })
}

#------------------------------------------------------------------------------
# S3 Bucket Notification for CloudTrail Log Delivery
#------------------------------------------------------------------------------

resource "aws_s3_bucket_notification" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  topic {
    topic_arn     = aws_sns_topic.cloudtrail_notifications.arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "AWSLogs/"
    filter_suffix = ".json.gz"
  }
}

#------------------------------------------------------------------------------
# Variables for CloudTrail Module
#------------------------------------------------------------------------------

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default = {
    Project   = "GreenLang"
    ManagedBy = "Terraform"
    Component = "CloudTrail"
  }
}

variable "authorized_principals" {
  description = "IAM principals authorized to decrypt CloudTrail logs"
  type        = list(string)
  default     = []
}

variable "monitored_buckets" {
  description = "List of S3 buckets to monitor with CloudTrail"
  type = list(object({
    bucket_name     = string
    read_write_type = string  # "All", "ReadOnly", or "WriteOnly"
  }))
  default = []
}

variable "log_retention_days" {
  description = "Number of days to retain CloudTrail logs in S3"
  type        = number
  default     = 2555  # 7 years for compliance
}

variable "cloudwatch_log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 90
}

variable "multi_region_trail" {
  description = "Enable multi-region trail"
  type        = bool
  default     = false
}

variable "enable_insights" {
  description = "Enable CloudTrail Insights"
  type        = bool
  default     = true
}

variable "insights_retention_days" {
  description = "Retention period for CloudTrail Insights"
  type        = number
  default     = 365
}

variable "enable_advanced_selectors" {
  description = "Enable advanced event selectors"
  type        = bool
  default     = false
}

variable "advanced_event_selectors" {
  description = "Advanced event selector configurations"
  type = list(object({
    name = string
    field_selectors = list(object({
      field           = string
      equals          = optional(list(string))
      not_equals      = optional(list(string))
      starts_with     = optional(list(string))
      not_starts_with = optional(list(string))
      ends_with       = optional(list(string))
      not_ends_with   = optional(list(string))
    }))
  }))
  default = []
}

#------------------------------------------------------------------------------
# Outputs
#------------------------------------------------------------------------------

output "cloudtrail_arn" {
  description = "ARN of the CloudTrail trail"
  value       = aws_cloudtrail.s3_data_events.arn
}

output "cloudtrail_name" {
  description = "Name of the CloudTrail trail"
  value       = aws_cloudtrail.s3_data_events.name
}

output "cloudtrail_logs_bucket_name" {
  description = "Name of the S3 bucket for CloudTrail logs"
  value       = aws_s3_bucket.cloudtrail_logs.id
}

output "cloudtrail_logs_bucket_arn" {
  description = "ARN of the S3 bucket for CloudTrail logs"
  value       = aws_s3_bucket.cloudtrail_logs.arn
}

output "cloudtrail_kms_key_arn" {
  description = "ARN of the KMS key for CloudTrail encryption"
  value       = aws_kms_key.cloudtrail.arn
}

output "cloudtrail_cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for CloudTrail"
  value       = aws_cloudwatch_log_group.cloudtrail.name
}

output "cloudtrail_sns_topic_arn" {
  description = "ARN of the SNS topic for CloudTrail notifications"
  value       = aws_sns_topic.cloudtrail_notifications.arn
}
