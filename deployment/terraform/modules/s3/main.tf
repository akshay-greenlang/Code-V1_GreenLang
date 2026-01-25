# GreenLang S3 Module
# Creates S3 buckets for artifacts, logs, backups, and data with security best practices

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# -----------------------------------------------------------------------------
# KMS Key for S3 Encryption
# -----------------------------------------------------------------------------
resource "aws_kms_key" "s3" {
  count = var.create_kms_key ? 1 : 0

  description             = "KMS key for S3 bucket encryption - ${var.name_prefix}"
  deletion_window_in_days = 7
  enable_key_rotation     = true

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
        Sid    = "Allow S3 Service"
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
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-s3-kms"
  })
}

resource "aws_kms_alias" "s3" {
  count = var.create_kms_key ? 1 : 0

  name          = "alias/${var.name_prefix}-s3"
  target_key_id = aws_kms_key.s3[0].key_id
}

# -----------------------------------------------------------------------------
# Artifacts Bucket
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  count = var.create_artifacts_bucket ? 1 : 0

  bucket = "${var.name_prefix}-artifacts-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-artifacts"
    Purpose = "artifacts"
  })
}

resource "aws_s3_bucket_versioning" "artifacts" {
  count = var.create_artifacts_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifacts[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  count = var.create_artifacts_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifacts[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.create_kms_key ? "aws:kms" : "AES256"
      kms_master_key_id = var.create_kms_key ? aws_kms_key.s3[0].arn : null
    }
    bucket_key_enabled = var.create_kms_key
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  count = var.create_artifacts_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifacts[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  count = var.create_artifacts_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifacts[0].id

  rule {
    id     = "cleanup-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = var.artifacts_noncurrent_version_expiration_days
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }
  }

  rule {
    id     = "cleanup-incomplete-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# -----------------------------------------------------------------------------
# Logs Bucket
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "logs" {
  count = var.create_logs_bucket ? 1 : 0

  bucket = "${var.name_prefix}-logs-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-logs"
    Purpose = "logs"
  })
}

resource "aws_s3_bucket_versioning" "logs" {
  count = var.create_logs_bucket ? 1 : 0

  bucket = aws_s3_bucket.logs[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  count = var.create_logs_bucket ? 1 : 0

  bucket = aws_s3_bucket.logs[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.create_kms_key ? "aws:kms" : "AES256"
      kms_master_key_id = var.create_kms_key ? aws_kms_key.s3[0].arn : null
    }
    bucket_key_enabled = var.create_kms_key
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  count = var.create_logs_bucket ? 1 : 0

  bucket = aws_s3_bucket.logs[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  count = var.create_logs_bucket ? 1 : 0

  bucket = aws_s3_bucket.logs[0].id

  rule {
    id     = "log-retention"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = var.logs_retention_days
    }
  }

  rule {
    id     = "cleanup-incomplete-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Allow ALB/ELB logging
resource "aws_s3_bucket_policy" "logs" {
  count = var.create_logs_bucket && var.enable_elb_logging ? 1 : 0

  bucket = aws_s3_bucket.logs[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowELBLogging"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${var.elb_account_id}:root"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.logs[0].arn}/alb-logs/*"
      },
      {
        Sid    = "AllowELBLogDelivery"
        Effect = "Allow"
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.logs[0].arn}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      },
      {
        Sid    = "AllowELBGetBucketAcl"
        Effect = "Allow"
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.logs[0].arn
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Backups Bucket
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "backups" {
  count = var.create_backups_bucket ? 1 : 0

  bucket = "${var.name_prefix}-backups-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-backups"
    Purpose = "backups"
  })
}

resource "aws_s3_bucket_versioning" "backups" {
  count = var.create_backups_bucket ? 1 : 0

  bucket = aws_s3_bucket.backups[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  count = var.create_backups_bucket ? 1 : 0

  bucket = aws_s3_bucket.backups[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.create_kms_key ? "aws:kms" : "AES256"
      kms_master_key_id = var.create_kms_key ? aws_kms_key.s3[0].arn : null
    }
    bucket_key_enabled = var.create_kms_key
  }
}

resource "aws_s3_bucket_public_access_block" "backups" {
  count = var.create_backups_bucket ? 1 : 0

  bucket = aws_s3_bucket.backups[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  count = var.create_backups_bucket ? 1 : 0

  bucket = aws_s3_bucket.backups[0].id

  rule {
    id     = "backup-retention"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = var.backups_retention_days
    }
  }

  rule {
    id     = "noncurrent-version-cleanup"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = var.backups_noncurrent_version_expiration_days
    }
  }
}

# Object Lock for immutable backups (compliance)
resource "aws_s3_bucket_object_lock_configuration" "backups" {
  count = var.create_backups_bucket && var.enable_backup_object_lock ? 1 : 0

  bucket              = aws_s3_bucket.backups[0].id
  object_lock_enabled = "Enabled"

  rule {
    default_retention {
      mode = var.backup_object_lock_mode
      days = var.backup_object_lock_days
    }
  }
}

# -----------------------------------------------------------------------------
# Data Bucket
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "data" {
  count = var.create_data_bucket ? 1 : 0

  bucket = "${var.name_prefix}-data-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-data"
    Purpose = "data"
  })
}

resource "aws_s3_bucket_versioning" "data" {
  count = var.create_data_bucket ? 1 : 0

  bucket = aws_s3_bucket.data[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  count = var.create_data_bucket ? 1 : 0

  bucket = aws_s3_bucket.data[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.create_kms_key ? "aws:kms" : "AES256"
      kms_master_key_id = var.create_kms_key ? aws_kms_key.s3[0].arn : null
    }
    bucket_key_enabled = var.create_kms_key
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  count = var.create_data_bucket ? 1 : 0

  bucket = aws_s3_bucket.data[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  count = var.create_data_bucket ? 1 : 0

  bucket = aws_s3_bucket.data[0].id

  rule {
    id     = "intelligent-tiering"
    status = "Enabled"

    transition {
      days          = 0
      storage_class = "INTELLIGENT_TIERING"
    }
  }

  rule {
    id     = "noncurrent-version-cleanup"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = var.data_noncurrent_version_expiration_days
    }
  }

  rule {
    id     = "cleanup-incomplete-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# CORS Configuration for frontend access
resource "aws_s3_bucket_cors_configuration" "data" {
  count = var.create_data_bucket && var.enable_cors ? 1 : 0

  bucket = aws_s3_bucket.data[0].id

  cors_rule {
    allowed_headers = var.cors_allowed_headers
    allowed_methods = var.cors_allowed_methods
    allowed_origins = var.cors_allowed_origins
    expose_headers  = var.cors_expose_headers
    max_age_seconds = var.cors_max_age_seconds
  }
}

# -----------------------------------------------------------------------------
# Static Assets Bucket (Frontend)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "static_assets" {
  count = var.create_static_assets_bucket ? 1 : 0

  bucket = "${var.name_prefix}-static-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-static"
    Purpose = "static-assets"
  })
}

resource "aws_s3_bucket_versioning" "static_assets" {
  count = var.create_static_assets_bucket ? 1 : 0

  bucket = aws_s3_bucket.static_assets[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "static_assets" {
  count = var.create_static_assets_bucket ? 1 : 0

  bucket = aws_s3_bucket.static_assets[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "static_assets" {
  count = var.create_static_assets_bucket ? 1 : 0

  bucket = aws_s3_bucket.static_assets[0].id

  block_public_acls       = true
  block_public_policy     = !var.enable_static_website
  ignore_public_acls      = true
  restrict_public_buckets = !var.enable_static_website
}

# Website configuration (if needed)
resource "aws_s3_bucket_website_configuration" "static_assets" {
  count = var.create_static_assets_bucket && var.enable_static_website ? 1 : 0

  bucket = aws_s3_bucket.static_assets[0].id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "error.html"
  }
}

# CORS for static assets (CDN access)
resource "aws_s3_bucket_cors_configuration" "static_assets" {
  count = var.create_static_assets_bucket ? 1 : 0

  bucket = aws_s3_bucket.static_assets[0].id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = var.static_assets_cors_origins
    expose_headers  = ["ETag"]
    max_age_seconds = 3600
  }
}

# CloudFront Origin Access Control policy
resource "aws_s3_bucket_policy" "static_assets_cloudfront" {
  count = var.create_static_assets_bucket && var.cloudfront_distribution_arn != null ? 1 : 0

  bucket = aws_s3_bucket.static_assets[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCloudFrontServicePrincipal"
        Effect = "Allow"
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.static_assets[0].arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = var.cloudfront_distribution_arn
          }
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Replication Configuration for DR (Data Bucket)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_replication_configuration" "data" {
  count = var.create_data_bucket && var.enable_replication ? 1 : 0

  depends_on = [aws_s3_bucket_versioning.data]

  role   = aws_iam_role.replication[0].arn
  bucket = aws_s3_bucket.data[0].id

  rule {
    id     = "replicate-all"
    status = "Enabled"

    filter {
      prefix = ""
    }

    destination {
      bucket        = var.replication_destination_bucket_arn
      storage_class = "STANDARD_IA"

      encryption_configuration {
        replica_kms_key_id = var.replication_destination_kms_key_arn
      }
    }

    source_selection_criteria {
      sse_kms_encrypted_objects {
        status = "Enabled"
      }
    }

    delete_marker_replication {
      status = "Enabled"
    }
  }
}

# Replication IAM Role
resource "aws_iam_role" "replication" {
  count = var.create_data_bucket && var.enable_replication ? 1 : 0

  name = "${var.name_prefix}-s3-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "replication" {
  count = var.create_data_bucket && var.enable_replication ? 1 : 0

  name = "${var.name_prefix}-s3-replication-policy"
  role = aws_iam_role.replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = aws_s3_bucket.data[0].arn
      },
      {
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Effect   = "Allow"
        Resource = "${aws_s3_bucket.data[0].arn}/*"
      },
      {
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Effect   = "Allow"
        Resource = "${var.replication_destination_bucket_arn}/*"
      },
      {
        Action = [
          "kms:Decrypt"
        ]
        Effect = "Allow"
        Resource = var.create_kms_key ? aws_kms_key.s3[0].arn : var.kms_key_arn
        Condition = {
          StringLike = {
            "kms:ViaService"       = "s3.${data.aws_region.current.name}.amazonaws.com"
            "kms:EncryptionContext:aws:s3:arn" = "${aws_s3_bucket.data[0].arn}/*"
          }
        }
      },
      {
        Action = [
          "kms:Encrypt"
        ]
        Effect   = "Allow"
        Resource = var.replication_destination_kms_key_arn
        Condition = {
          StringLike = {
            "kms:ViaService"       = "s3.${var.replication_destination_region}.amazonaws.com"
            "kms:EncryptionContext:aws:s3:arn" = "${var.replication_destination_bucket_arn}/*"
          }
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# S3 Access Logging
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_logging" "artifacts" {
  count = var.create_artifacts_bucket && var.create_logs_bucket && var.enable_access_logging ? 1 : 0

  bucket        = aws_s3_bucket.artifacts[0].id
  target_bucket = aws_s3_bucket.logs[0].id
  target_prefix = "s3-access-logs/artifacts/"
}

resource "aws_s3_bucket_logging" "data" {
  count = var.create_data_bucket && var.create_logs_bucket && var.enable_access_logging ? 1 : 0

  bucket        = aws_s3_bucket.data[0].id
  target_bucket = aws_s3_bucket.logs[0].id
  target_prefix = "s3-access-logs/data/"
}

resource "aws_s3_bucket_logging" "backups" {
  count = var.create_backups_bucket && var.create_logs_bucket && var.enable_access_logging ? 1 : 0

  bucket        = aws_s3_bucket.backups[0].id
  target_bucket = aws_s3_bucket.logs[0].id
  target_prefix = "s3-access-logs/backups/"
}
