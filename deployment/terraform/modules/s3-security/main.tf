#############################################################################
# GreenLang S3 Security Hardening Module
#
# This module implements comprehensive S3 security controls including:
# - MFA Delete configuration for critical buckets
# - Strict bucket policies (SSL-only, encryption, VPC endpoint restrictions)
# - S3 Object Lock (GOVERNANCE and COMPLIANCE modes)
# - VPC Endpoints with restrictive policies
# - Centralized access logging
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

data "aws_vpc" "selected" {
  count = var.vpc_id != "" ? 1 : 0
  id    = var.vpc_id
}

data "aws_route_tables" "private" {
  count  = var.vpc_id != "" ? 1 : 0
  vpc_id = var.vpc_id

  filter {
    name   = "tag:Tier"
    values = ["private"]
  }
}

#############################################################################
# KMS Key for S3 Encryption
#############################################################################

resource "aws_kms_key" "s3_encryption" {
  description             = "KMS key for GreenLang S3 bucket encryption"
  deletion_window_in_days = var.kms_key_deletion_window
  enable_key_rotation     = true
  multi_region            = var.kms_multi_region

  policy = jsonencode({
    Version = "2012-10-17"
    Id      = "s3-encryption-key-policy"
    Statement = [
      {
        Sid    = "EnableRootAccountAccess"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "AllowS3ServiceAccess"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
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
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      {
        Sid    = "AllowCloudWatchLogsAccess"
        Effect = "Allow"
        Principal = {
          Service = "logs.${data.aws_region.current.name}.amazonaws.com"
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
          ArnLike = {
            "kms:EncryptionContext:aws:logs:arn" = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
          }
        }
      },
      {
        Sid    = "AllowS3AdminsWithMFA"
        Effect = "Allow"
        Principal = {
          AWS = var.s3_admin_principals
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey",
          "kms:CreateGrant"
        ]
        Resource = "*"
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.project_name}-s3-kms-key"
    Purpose = "S3 Encryption"
  })
}

resource "aws_kms_alias" "s3_encryption" {
  name          = "alias/${var.project_name}-s3-encryption"
  target_key_id = aws_kms_key.s3_encryption.key_id
}

#############################################################################
# Centralized Access Logging Bucket
#############################################################################

resource "aws_s3_bucket" "access_logs" {
  bucket = "${var.project_name}-access-logs-${data.aws_caller_identity.current.account_id}"

  # Prevent accidental deletion
  force_destroy = false

  tags = merge(var.tags, {
    Name    = "${var.project_name}-access-logs"
    Purpose = "S3 Access Logging"
  })
}

resource "aws_s3_bucket_versioning" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  versioning_configuration {
    status = "Enabled"
    # MFA Delete requires manual setup with root credentials
    # See documentation in mfa-delete-setup.md
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_encryption.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  rule {
    id     = "log-retention"
    status = "Enabled"

    filter {
      prefix = ""
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
      days = var.access_log_retention_days
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

resource "aws_s3_bucket_policy" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DenyNonSSLRequests"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.access_logs.arn,
          "${aws_s3_bucket.access_logs.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
      {
        Sid    = "AllowS3LogDelivery"
        Effect = "Allow"
        Principal = {
          Service = "logging.s3.amazonaws.com"
        }
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.access_logs.arn}/*"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      {
        Sid    = "DenyIncorrectEncryptionHeader"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.access_logs.arn}/*"
        Condition = {
          StringNotEquals = {
            "s3:x-amz-server-side-encryption" = "aws:kms"
          }
        }
      },
      {
        Sid    = "DenyUnencryptedUploads"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.access_logs.arn}/*"
        Condition = {
          Null = {
            "s3:x-amz-server-side-encryption" = "true"
          }
        }
      }
    ]
  })
}

#############################################################################
# Critical Backup Bucket with Object Lock (GOVERNANCE Mode)
#############################################################################

resource "aws_s3_bucket" "backups" {
  bucket = "${var.project_name}-backups-${data.aws_caller_identity.current.account_id}"

  # Object Lock requires this to be set at bucket creation
  object_lock_enabled = true

  force_destroy = false

  tags = merge(var.tags, {
    Name        = "${var.project_name}-backups"
    Purpose     = "Critical Backups"
    ObjectLock  = "GOVERNANCE"
    Criticality = "High"
  })
}

resource "aws_s3_bucket_object_lock_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    default_retention {
      mode = "GOVERNANCE"
      days = var.backup_retention_days
    }
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_encryption.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_logging" "backups" {
  bucket = aws_s3_bucket.backups.id

  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "backups/"
}

resource "aws_s3_bucket_policy" "backups" {
  bucket = aws_s3_bucket.backups.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DenyNonSSLRequests"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
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
        Resource = "${aws_s3_bucket.backups.arn}/*"
        Condition = {
          StringNotEquals = {
            "s3:x-amz-server-side-encryption" = "aws:kms"
          }
        }
      },
      {
        Sid    = "DenyUnencryptedUploads"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.backups.arn}/*"
        Condition = {
          Null = {
            "s3:x-amz-server-side-encryption" = "true"
          }
        }
      },
      {
        Sid    = "RestrictToVPCEndpoint"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
        Condition = {
          StringNotEquals = {
            "aws:SourceVpce" = var.vpc_id != "" ? aws_vpc_endpoint.s3[0].id : ""
          }
          StringNotLike = {
            "aws:PrincipalArn" = var.admin_role_arns
          }
        }
      },
      {
        Sid    = "DenyPublicAccess"
        Effect = "Deny"
        Principal = "*"
        Action = [
          "s3:PutBucketPolicy",
          "s3:PutBucketAcl",
          "s3:PutObjectAcl"
        ]
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = ["public-read", "public-read-write", "authenticated-read"]
          }
        }
      },
      {
        Sid    = "RequireMFAForDelete"
        Effect = "Deny"
        Principal = "*"
        Action = [
          "s3:DeleteObject",
          "s3:DeleteObjectVersion",
          "s3:PutBucketVersioning"
        ]
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "false"
          }
        }
      }
    ]
  })
}

#############################################################################
# Audit Logs Bucket with Object Lock (COMPLIANCE Mode)
#############################################################################

resource "aws_s3_bucket" "audit_logs" {
  bucket = "${var.project_name}-audit-logs-${data.aws_caller_identity.current.account_id}"

  object_lock_enabled = true
  force_destroy       = false

  tags = merge(var.tags, {
    Name        = "${var.project_name}-audit-logs"
    Purpose     = "Audit Logs"
    ObjectLock  = "COMPLIANCE"
    Criticality = "Critical"
    Compliance  = "SOC2,ISO27001"
  })
}

resource "aws_s3_bucket_object_lock_configuration" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  rule {
    default_retention {
      mode  = "COMPLIANCE"
      years = var.audit_log_retention_years
    }
  }
}

resource "aws_s3_bucket_versioning" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_encryption.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_logging" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "audit-logs/"
}

resource "aws_s3_bucket_policy" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DenyNonSSLRequests"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*"
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
        Resource = "${aws_s3_bucket.audit_logs.arn}/*"
        Condition = {
          StringNotEquals = {
            "s3:x-amz-server-side-encryption" = "aws:kms"
          }
        }
      },
      {
        Sid    = "RestrictToVPCEndpoint"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*"
        ]
        Condition = {
          StringNotEquals = {
            "aws:SourceVpce" = var.vpc_id != "" ? aws_vpc_endpoint.s3[0].id : ""
          }
          StringNotLike = {
            "aws:PrincipalArn" = concat(var.admin_role_arns, var.audit_reader_arns)
          }
        }
      },
      {
        Sid    = "DenyDeleteActions"
        Effect = "Deny"
        Principal = "*"
        Action = [
          "s3:DeleteObject",
          "s3:DeleteObjectVersion",
          "s3:DeleteBucket"
        ]
        Resource = [
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*"
        ]
      },
      {
        Sid    = "AllowCloudTrailDelivery"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action = "s3:PutObject"
        Resource = "${aws_s3_bucket.audit_logs.arn}/cloudtrail/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
            "aws:SourceArn" = "arn:aws:cloudtrail:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:trail/*"
          }
        }
      },
      {
        Sid    = "AllowCloudTrailAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.audit_logs.arn
        Condition = {
          StringEquals = {
            "aws:SourceArn" = "arn:aws:cloudtrail:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:trail/*"
          }
        }
      },
      {
        Sid    = "AllowCrossAccountAuditRead"
        Effect = "Allow"
        Principal = {
          AWS = var.cross_account_audit_principals
        }
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
        }
      }
    ]
  })
}

#############################################################################
# Application Data Bucket
#############################################################################

resource "aws_s3_bucket" "app_data" {
  bucket = "${var.project_name}-app-data-${data.aws_caller_identity.current.account_id}"

  force_destroy = false

  tags = merge(var.tags, {
    Name    = "${var.project_name}-app-data"
    Purpose = "Application Data"
  })
}

resource "aws_s3_bucket_versioning" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_encryption.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_logging" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "app-data/"
}

resource "aws_s3_bucket_lifecycle_configuration" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  rule {
    id     = "transition-old-versions"
    status = "Enabled"

    filter {
      prefix = ""
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }

  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"

    filter {
      prefix = ""
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_policy" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DenyNonSSLRequests"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
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
        Resource = "${aws_s3_bucket.app_data.arn}/*"
        Condition = {
          StringNotEquals = {
            "s3:x-amz-server-side-encryption" = "aws:kms"
          }
        }
      },
      {
        Sid    = "RestrictToVPCEndpoint"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
        ]
        Condition = {
          StringNotEquals = {
            "aws:SourceVpce" = var.vpc_id != "" ? aws_vpc_endpoint.s3[0].id : ""
          }
          StringNotLike = {
            "aws:PrincipalArn" = var.admin_role_arns
          }
        }
      },
      {
        Sid    = "AllowIPBasedAdminAccess"
        Effect = "Allow"
        Principal = {
          AWS = var.admin_role_arns
        }
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
        ]
        Condition = {
          IpAddress = {
            "aws:SourceIp" = var.admin_ip_allowlist
          }
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
        }
      }
    ]
  })
}

#############################################################################
# VPC Endpoint for S3
#############################################################################

resource "aws_vpc_endpoint" "s3" {
  count = var.vpc_id != "" ? 1 : 0

  vpc_id            = var.vpc_id
  service_name      = "com.amazonaws.${data.aws_region.current.name}.s3"
  vpc_endpoint_type = "Gateway"

  route_table_ids = var.private_route_table_ids

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "RestrictToGreenLangBuckets"
        Effect    = "Allow"
        Principal = "*"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging",
          "s3:PutObjectTagging",
          "s3:DeleteObject",
          "s3:DeleteObjectVersion"
        ]
        Resource = [
          aws_s3_bucket.access_logs.arn,
          "${aws_s3_bucket.access_logs.arn}/*",
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*",
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*",
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
        ]
        Condition = {
          StringEquals = {
            "aws:PrincipalAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      {
        Sid       = "DenyNonSSLRequests"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = "*"
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.project_name}-s3-endpoint"
  })
}

#############################################################################
# S3 Access Points for Fine-Grained Access Control
#############################################################################

resource "aws_s3_access_point" "app_data_readonly" {
  bucket = aws_s3_bucket.app_data.id
  name   = "${var.project_name}-app-data-readonly"

  vpc_configuration {
    vpc_id = var.vpc_id
  }

  public_access_block_configuration {
    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true
  }
}

resource "aws_s3_access_point" "app_data_write" {
  bucket = aws_s3_bucket.app_data.id
  name   = "${var.project_name}-app-data-write"

  vpc_configuration {
    vpc_id = var.vpc_id
  }

  public_access_block_configuration {
    block_public_acls       = true
    block_public_policy     = true
    ignore_public_acls      = true
    restrict_public_buckets = true
  }
}

resource "aws_s3control_access_point_policy" "app_data_readonly" {
  access_point_arn = aws_s3_access_point.app_data_readonly.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadOnlyAccess"
        Effect = "Allow"
        Principal = {
          AWS = var.readonly_role_arns
        }
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_access_point.app_data_readonly.arn,
          "${aws_s3_access_point.app_data_readonly.arn}/object/*"
        ]
      }
    ]
  })
}

resource "aws_s3control_access_point_policy" "app_data_write" {
  access_point_arn = aws_s3_access_point.app_data_write.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "WriteAccess"
        Effect = "Allow"
        Principal = {
          AWS = var.write_role_arns
        }
        Action = [
          "s3:PutObject",
          "s3:PutObjectTagging",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]
        Resource = [
          aws_s3_access_point.app_data_write.arn,
          "${aws_s3_access_point.app_data_write.arn}/object/*"
        ]
        Condition = {
          StringEquals = {
            "s3:x-amz-server-side-encryption" = "aws:kms"
          }
        }
      }
    ]
  })
}

#############################################################################
# S3 Inventory Configuration
#############################################################################

resource "aws_s3_bucket_inventory" "app_data" {
  bucket = aws_s3_bucket.app_data.id
  name   = "EntireBucketDaily"

  included_object_versions = "All"

  schedule {
    frequency = "Daily"
  }

  destination {
    bucket {
      format     = "Parquet"
      bucket_arn = aws_s3_bucket.access_logs.arn
      prefix     = "inventory/app-data"
      encryption {
        sse_kms {
          key_id = aws_kms_key.s3_encryption.arn
        }
      }
    }
  }

  optional_fields = [
    "Size",
    "LastModifiedDate",
    "StorageClass",
    "ETag",
    "IsMultipartUploaded",
    "ReplicationStatus",
    "EncryptionStatus",
    "ObjectLockRetainUntilDate",
    "ObjectLockMode",
    "ObjectLockLegalHoldStatus",
    "IntelligentTieringAccessTier"
  ]
}

#############################################################################
# Legal Hold Management (via null_resource for documentation)
#############################################################################

# Legal holds must be applied/removed via AWS CLI or SDK
# Example commands documented here for reference:
#
# Apply legal hold:
# aws s3api put-object-legal-hold \
#   --bucket ${aws_s3_bucket.audit_logs.id} \
#   --key <object-key> \
#   --legal-hold Status=ON
#
# Remove legal hold:
# aws s3api put-object-legal-hold \
#   --bucket ${aws_s3_bucket.audit_logs.id} \
#   --key <object-key> \
#   --legal-hold Status=OFF

#############################################################################
# MFA Delete Documentation
#############################################################################

# MFA Delete must be enabled using root account credentials.
# This cannot be done via Terraform.
#
# To enable MFA Delete:
# 1. Sign in as root user
# 2. Run the following AWS CLI command:
#
# aws s3api put-bucket-versioning \
#   --bucket ${aws_s3_bucket.backups.id} \
#   --versioning-configuration Status=Enabled,MFADelete=Enabled \
#   --mfa "arn:aws:iam::ACCOUNT_ID:mfa/root-account-mfa-device TOTP_CODE"
#
# Replace ACCOUNT_ID with your AWS account ID
# Replace TOTP_CODE with the current 6-digit code from your MFA device

#############################################################################
# CloudWatch Alarms for S3 Security Events
#############################################################################

resource "aws_cloudwatch_metric_alarm" "s3_public_access_attempt" {
  alarm_name          = "${var.project_name}-s3-public-access-attempt"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BucketPolicyDenyCount"
  namespace           = "AWS/S3"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Detected attempt to access S3 with denied policy"

  dimensions = {
    BucketName = aws_s3_bucket.app_data.id
    FilterId   = "PublicAccessDenied"
  }

  alarm_actions = var.alarm_sns_topic_arns

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "s3_unauthorized_access" {
  alarm_name          = "${var.project_name}-s3-unauthorized-access"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "4xxErrors"
  namespace           = "AWS/S3"
  period              = 300
  statistic           = "Sum"
  threshold           = 100
  alarm_description   = "High rate of 4xx errors indicating potential unauthorized access attempts"

  dimensions = {
    BucketName = aws_s3_bucket.app_data.id
  }

  alarm_actions = var.alarm_sns_topic_arns

  tags = var.tags
}
