#############################################################################
# GreenLang S3 Compliance Module
#
# This module implements AWS Config rules for S3 compliance monitoring
# and automatic remediation of security misconfigurations.
#
# Compliance Standards Covered:
# - SOC 2 Type II
# - ISO 27001
# - PCI DSS
# - HIPAA (optional)
#
# Author: GreenLang DevOps Team
# Version: 1.0.0
#############################################################################

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

#############################################################################
# Data Sources
#############################################################################

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

#############################################################################
# AWS Config Recorder
#############################################################################

resource "aws_config_configuration_recorder" "main" {
  name     = "${var.project_name}-config-recorder"
  role_arn = aws_iam_role.config_recorder.arn

  recording_group {
    all_supported                 = false
    include_global_resource_types = false
    resource_types = [
      "AWS::S3::Bucket",
      "AWS::S3::AccountPublicAccessBlock"
    ]
  }

  recording_mode {
    recording_frequency = "CONTINUOUS"
  }
}

resource "aws_config_configuration_recorder_status" "main" {
  name       = aws_config_configuration_recorder.main.name
  is_enabled = true

  depends_on = [aws_config_delivery_channel.main]
}

#############################################################################
# Config Delivery Channel
#############################################################################

resource "aws_config_delivery_channel" "main" {
  name           = "${var.project_name}-config-delivery"
  s3_bucket_name = aws_s3_bucket.config_snapshots.id
  s3_key_prefix  = "config"
  sns_topic_arn  = var.sns_topic_arn

  snapshot_delivery_properties {
    delivery_frequency = var.config_snapshot_frequency
  }

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# S3 Bucket for Config Snapshots
#############################################################################

resource "aws_s3_bucket" "config_snapshots" {
  bucket = "${var.project_name}-config-snapshots-${data.aws_caller_identity.current.account_id}"

  force_destroy = false

  tags = merge(var.tags, {
    Name    = "${var.project_name}-config-snapshots"
    Purpose = "AWS Config Snapshots"
  })
}

resource "aws_s3_bucket_versioning" "config_snapshots" {
  bucket = aws_s3_bucket.config_snapshots.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "config_snapshots" {
  bucket = aws_s3_bucket.config_snapshots.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "config_snapshots" {
  bucket = aws_s3_bucket.config_snapshots.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "config_snapshots" {
  bucket = aws_s3_bucket.config_snapshots.id

  rule {
    id     = "config-snapshot-retention"
    status = "Enabled"

    filter {
      prefix = "config/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = var.config_snapshot_retention_days
    }
  }
}

resource "aws_s3_bucket_policy" "config_snapshots" {
  bucket = aws_s3_bucket.config_snapshots.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSConfigBucketPermissionsCheck"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.config_snapshots.arn
        Condition = {
          StringEquals = {
            "AWS:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      {
        Sid    = "AWSConfigBucketExistenceCheck"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.config_snapshots.arn
        Condition = {
          StringEquals = {
            "AWS:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      {
        Sid    = "AWSConfigBucketDelivery"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.config_snapshots.arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceAccount" = data.aws_caller_identity.current.account_id
            "s3:x-amz-acl"      = "bucket-owner-full-control"
          }
        }
      },
      {
        Sid       = "DenyNonSSLRequests"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.config_snapshots.arn,
          "${aws_s3_bucket.config_snapshots.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

#############################################################################
# IAM Role for Config Recorder
#############################################################################

resource "aws_iam_role" "config_recorder" {
  name = "${var.project_name}-config-recorder-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "config_recorder" {
  role       = aws_iam_role.config_recorder.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWS_ConfigRole"
}

resource "aws_iam_role_policy" "config_recorder_s3" {
  name = "${var.project_name}-config-recorder-s3"
  role = aws_iam_role.config_recorder.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl"
        ]
        Resource = "${aws_s3_bucket.config_snapshots.arn}/*"
        Condition = {
          StringLike = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketAcl"
        ]
        Resource = aws_s3_bucket.config_snapshots.arn
      }
    ]
  })
}

#############################################################################
# AWS Config Rules - SSL Requests Only
#############################################################################

resource "aws_config_config_rule" "s3_bucket_ssl_requests_only" {
  name        = "${var.project_name}-s3-bucket-ssl-requests-only"
  description = "Checks if S3 buckets have policies that require SSL/TLS for all requests"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_SSL_REQUESTS_ONLY"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  input_parameters = jsonencode({})

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Server Side Encryption Enabled
#############################################################################

resource "aws_config_config_rule" "s3_bucket_server_side_encryption_enabled" {
  name        = "${var.project_name}-s3-bucket-sse-enabled"
  description = "Checks if S3 buckets have server-side encryption enabled"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Public Read Prohibited
#############################################################################

resource "aws_config_config_rule" "s3_bucket_public_read_prohibited" {
  name        = "${var.project_name}-s3-bucket-public-read-prohibited"
  description = "Checks if S3 buckets do not allow public read access"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_PUBLIC_READ_PROHIBITED"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Public Write Prohibited
#############################################################################

resource "aws_config_config_rule" "s3_bucket_public_write_prohibited" {
  name        = "${var.project_name}-s3-bucket-public-write-prohibited"
  description = "Checks if S3 buckets do not allow public write access"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_PUBLIC_WRITE_PROHIBITED"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Versioning Enabled
#############################################################################

resource "aws_config_config_rule" "s3_bucket_versioning_enabled" {
  name        = "${var.project_name}-s3-bucket-versioning-enabled"
  description = "Checks if S3 buckets have versioning enabled"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_VERSIONING_ENABLED"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  input_parameters = jsonencode({
    isMfaDeleteEnabled = var.require_mfa_delete ? "true" : "false"
  })

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Logging Enabled
#############################################################################

resource "aws_config_config_rule" "s3_bucket_logging_enabled" {
  name        = "${var.project_name}-s3-bucket-logging-enabled"
  description = "Checks if S3 buckets have logging enabled"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_LOGGING_ENABLED"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  input_parameters = var.logging_target_bucket != "" ? jsonencode({
    targetBucket = var.logging_target_bucket
  }) : jsonencode({})

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Replication Enabled
#############################################################################

resource "aws_config_config_rule" "s3_bucket_replication_enabled" {
  count = var.require_replication ? 1 : 0

  name        = "${var.project_name}-s3-bucket-replication-enabled"
  description = "Checks if S3 buckets have cross-region replication enabled"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_REPLICATION_ENABLED"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Default Encryption KMS
#############################################################################

resource "aws_config_config_rule" "s3_default_encryption_kms" {
  name        = "${var.project_name}-s3-default-encryption-kms"
  description = "Checks if S3 buckets are encrypted with KMS"

  source {
    owner             = "AWS"
    source_identifier = "S3_DEFAULT_ENCRYPTION_KMS"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  input_parameters = var.required_kms_key_id != "" ? jsonencode({
    kmsKeyArns = var.required_kms_key_id
  }) : jsonencode({})

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Account Level Public Access Block
#############################################################################

resource "aws_config_config_rule" "s3_account_level_public_access_blocks" {
  name        = "${var.project_name}-s3-account-public-access-blocks"
  description = "Checks if account level public access blocks are configured"

  source {
    owner             = "AWS"
    source_identifier = "S3_ACCOUNT_LEVEL_PUBLIC_ACCESS_BLOCKS"
  }

  input_parameters = jsonencode({
    BlockPublicAcls       = "true"
    BlockPublicPolicy     = "true"
    IgnorePublicAcls      = "true"
    RestrictPublicBuckets = "true"
  })

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Bucket Level Public Access Prohibited
#############################################################################

resource "aws_config_config_rule" "s3_bucket_level_public_access_prohibited" {
  name        = "${var.project_name}-s3-bucket-level-public-access-prohibited"
  description = "Checks if S3 buckets have public access blocks configured"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_LEVEL_PUBLIC_ACCESS_PROHIBITED"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  input_parameters = jsonencode({
    excludedPublicBuckets = join(",", var.allowed_public_buckets)
  })

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# AWS Config Rules - Bucket ACL Prohibited
#############################################################################

resource "aws_config_config_rule" "s3_bucket_acl_prohibited" {
  name        = "${var.project_name}-s3-bucket-acl-prohibited"
  description = "Checks that S3 buckets do not use ACLs for access control"

  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_ACL_PROHIBITED"
  }

  scope {
    compliance_resource_types = ["AWS::S3::Bucket"]
  }

  tags = var.tags

  depends_on = [aws_config_configuration_recorder.main]
}

#############################################################################
# Remediation Configuration - Auto Enable Encryption
#############################################################################

resource "aws_config_remediation_configuration" "enable_encryption" {
  config_rule_name = aws_config_config_rule.s3_bucket_server_side_encryption_enabled.name
  target_type      = "SSM_DOCUMENT"
  target_id        = "AWS-EnableS3BucketEncryption"

  automatic                  = var.enable_auto_remediation
  maximum_automatic_attempts = 3
  retry_attempt_seconds      = 60

  parameter {
    name         = "BucketName"
    resource_value {
      value = "RESOURCE_ID"
    }
  }

  parameter {
    name         = "SSEAlgorithm"
    static_value {
      values = ["AES256"]
    }
  }

  parameter {
    name         = "AutomationAssumeRole"
    static_value {
      values = [aws_iam_role.remediation.arn]
    }
  }

  execution_controls {
    ssm_controls {
      concurrent_execution_rate_percentage = 10
      error_percentage                     = 50
    }
  }
}

#############################################################################
# Remediation Configuration - Auto Enable Versioning
#############################################################################

resource "aws_config_remediation_configuration" "enable_versioning" {
  config_rule_name = aws_config_config_rule.s3_bucket_versioning_enabled.name
  target_type      = "SSM_DOCUMENT"
  target_id        = "AWS-ConfigureS3BucketVersioning"

  automatic                  = var.enable_auto_remediation
  maximum_automatic_attempts = 3
  retry_attempt_seconds      = 60

  parameter {
    name         = "BucketName"
    resource_value {
      value = "RESOURCE_ID"
    }
  }

  parameter {
    name         = "VersioningState"
    static_value {
      values = ["Enabled"]
    }
  }

  parameter {
    name         = "AutomationAssumeRole"
    static_value {
      values = [aws_iam_role.remediation.arn]
    }
  }

  execution_controls {
    ssm_controls {
      concurrent_execution_rate_percentage = 10
      error_percentage                     = 50
    }
  }
}

#############################################################################
# Remediation Configuration - Auto Block Public Access
#############################################################################

resource "aws_config_remediation_configuration" "block_public_access" {
  config_rule_name = aws_config_config_rule.s3_bucket_public_read_prohibited.name
  target_type      = "SSM_DOCUMENT"
  target_id        = "AWS-DisableS3BucketPublicReadWrite"

  automatic                  = var.enable_auto_remediation
  maximum_automatic_attempts = 3
  retry_attempt_seconds      = 60

  parameter {
    name         = "S3BucketName"
    resource_value {
      value = "RESOURCE_ID"
    }
  }

  parameter {
    name         = "AutomationAssumeRole"
    static_value {
      values = [aws_iam_role.remediation.arn]
    }
  }

  execution_controls {
    ssm_controls {
      concurrent_execution_rate_percentage = 10
      error_percentage                     = 50
    }
  }
}

#############################################################################
# IAM Role for Remediation
#############################################################################

resource "aws_iam_role" "remediation" {
  name = "${var.project_name}-config-remediation-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ssm.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "remediation" {
  name = "${var.project_name}-config-remediation-policy"
  role = aws_iam_role.remediation.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3RemediationPermissions"
        Effect = "Allow"
        Action = [
          "s3:PutBucketVersioning",
          "s3:PutEncryptionConfiguration",
          "s3:PutBucketPublicAccessBlock",
          "s3:PutBucketAcl",
          "s3:PutBucketPolicy",
          "s3:GetBucketVersioning",
          "s3:GetEncryptionConfiguration",
          "s3:GetBucketPublicAccessBlock",
          "s3:GetBucketAcl",
          "s3:GetBucketPolicy"
        ]
        Resource = "*"
      },
      {
        Sid    = "KMSPermissions"
        Effect = "Allow"
        Action = [
          "kms:DescribeKey",
          "kms:GenerateDataKey"
        ]
        Resource = "*"
      }
    ]
  })
}

#############################################################################
# Conformance Pack for S3 Best Practices
#############################################################################

resource "aws_config_conformance_pack" "s3_best_practices" {
  name = "${var.project_name}-s3-best-practices"

  template_body = <<-EOT
    AWSTemplateFormatVersion: '2010-09-09'
    Description: Conformance pack for S3 security best practices

    Resources:
      S3BucketSSLRequestsOnly:
        Type: AWS::Config::ConfigRule
        Properties:
          ConfigRuleName: s3-bucket-ssl-requests-only-cp
          Description: Checks if S3 buckets require SSL
          Source:
            Owner: AWS
            SourceIdentifier: S3_BUCKET_SSL_REQUESTS_ONLY
          Scope:
            ComplianceResourceTypes:
              - AWS::S3::Bucket

      S3BucketServerSideEncryptionEnabled:
        Type: AWS::Config::ConfigRule
        Properties:
          ConfigRuleName: s3-bucket-sse-enabled-cp
          Description: Checks if S3 buckets have encryption enabled
          Source:
            Owner: AWS
            SourceIdentifier: S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED
          Scope:
            ComplianceResourceTypes:
              - AWS::S3::Bucket

      S3BucketPublicReadProhibited:
        Type: AWS::Config::ConfigRule
        Properties:
          ConfigRuleName: s3-bucket-public-read-prohibited-cp
          Description: Checks if S3 buckets allow public read
          Source:
            Owner: AWS
            SourceIdentifier: S3_BUCKET_PUBLIC_READ_PROHIBITED
          Scope:
            ComplianceResourceTypes:
              - AWS::S3::Bucket

      S3BucketPublicWriteProhibited:
        Type: AWS::Config::ConfigRule
        Properties:
          ConfigRuleName: s3-bucket-public-write-prohibited-cp
          Description: Checks if S3 buckets allow public write
          Source:
            Owner: AWS
            SourceIdentifier: S3_BUCKET_PUBLIC_WRITE_PROHIBITED
          Scope:
            ComplianceResourceTypes:
              - AWS::S3::Bucket

      S3BucketVersioningEnabled:
        Type: AWS::Config::ConfigRule
        Properties:
          ConfigRuleName: s3-bucket-versioning-enabled-cp
          Description: Checks if S3 buckets have versioning enabled
          Source:
            Owner: AWS
            SourceIdentifier: S3_BUCKET_VERSIONING_ENABLED
          Scope:
            ComplianceResourceTypes:
              - AWS::S3::Bucket

      S3BucketLoggingEnabled:
        Type: AWS::Config::ConfigRule
        Properties:
          ConfigRuleName: s3-bucket-logging-enabled-cp
          Description: Checks if S3 buckets have logging enabled
          Source:
            Owner: AWS
            SourceIdentifier: S3_BUCKET_LOGGING_ENABLED
          Scope:
            ComplianceResourceTypes:
              - AWS::S3::Bucket
  EOT

  depends_on = [aws_config_configuration_recorder.main]

  tags = var.tags
}

#############################################################################
# CloudWatch Event Rule for Non-Compliance
#############################################################################

resource "aws_cloudwatch_event_rule" "config_compliance_change" {
  name        = "${var.project_name}-s3-compliance-change"
  description = "Capture S3 compliance status changes"

  event_pattern = jsonencode({
    source      = ["aws.config"]
    detail-type = ["Config Rules Compliance Change"]
    detail = {
      messageType = ["ComplianceChangeNotification"]
      newEvaluationResult = {
        complianceType = ["NON_COMPLIANT"]
      }
      configRuleName = [
        { prefix = var.project_name }
      ]
    }
  })

  tags = var.tags
}

resource "aws_cloudwatch_event_target" "config_compliance_sns" {
  count = var.sns_topic_arn != "" ? 1 : 0

  rule      = aws_cloudwatch_event_rule.config_compliance_change.name
  target_id = "SendToSNS"
  arn       = var.sns_topic_arn

  input_transformer {
    input_paths = {
      configRuleName   = "$.detail.configRuleName"
      resourceType     = "$.detail.resourceType"
      resourceId       = "$.detail.resourceId"
      awsRegion        = "$.detail.awsRegion"
      awsAccountId     = "$.detail.awsAccountId"
      complianceType   = "$.detail.newEvaluationResult.complianceType"
    }
    input_template = <<-EOT
      {
        "alarm": "S3 Compliance Violation",
        "rule": "<configRuleName>",
        "resource_type": "<resourceType>",
        "resource_id": "<resourceId>",
        "region": "<awsRegion>",
        "account": "<awsAccountId>",
        "status": "<complianceType>",
        "message": "S3 bucket <resourceId> is NON_COMPLIANT with rule <configRuleName>"
      }
    EOT
  }
}

#############################################################################
# S3 Access Analyzer
#############################################################################

resource "aws_accessanalyzer_analyzer" "s3" {
  analyzer_name = "${var.project_name}-s3-analyzer"
  type          = "ACCOUNT"

  tags = var.tags
}

#############################################################################
# Account-Level Public Access Block
#############################################################################

resource "aws_s3_account_public_access_block" "main" {
  count = var.enable_account_public_access_block ? 1 : 0

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
