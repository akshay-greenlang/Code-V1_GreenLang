# =============================================================================
# GreenLang Prometheus Stack Module - S3 Storage for Thanos
# GreenLang Climate OS | OBS-001
# =============================================================================
# Creates S3 bucket for Thanos long-term metrics storage with:
#   - Versioning enabled for data protection
#   - Lifecycle rules (Intelligent Tiering at 30d, delete at 730d)
#   - Server-side encryption (AES256 or KMS)
#   - Public access blocking
#   - TLS-only bucket policy
#   - Optional access logging
# =============================================================================

# -----------------------------------------------------------------------------
# Locals
# -----------------------------------------------------------------------------

locals {
  thanos_bucket_name = var.thanos_bucket_name != "" ? var.thanos_bucket_name : "${var.project}-thanos-metrics-${var.environment}"
}

# -----------------------------------------------------------------------------
# Thanos Metrics S3 Bucket
# -----------------------------------------------------------------------------
# Stores Prometheus metrics blocks uploaded by Thanos Sidecar.
# Data is retained for 2 years (730 days) with tiered storage.

resource "aws_s3_bucket" "thanos_metrics" {
  count = var.enable_thanos ? 1 : 0

  bucket        = local.thanos_bucket_name
  force_destroy = false

  tags = merge(var.tags, {
    Name        = local.thanos_bucket_name
    Component   = "thanos"
    Purpose     = "metrics-storage"
    DataClass   = "operational"
    Retention   = "2-years"
    Environment = var.environment
  })
}

# -----------------------------------------------------------------------------
# Bucket Versioning
# -----------------------------------------------------------------------------
# Enabled to protect against accidental deletion and provide recovery options.

resource "aws_s3_bucket_versioning" "thanos_metrics" {
  count = var.enable_thanos ? 1 : 0

  bucket = aws_s3_bucket.thanos_metrics[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

# -----------------------------------------------------------------------------
# Server-Side Encryption
# -----------------------------------------------------------------------------
# Uses AES256 by default, or customer-provided KMS key if specified.

resource "aws_s3_bucket_server_side_encryption_configuration" "thanos_metrics" {
  count = var.enable_thanos ? 1 : 0

  bucket = aws_s3_bucket.thanos_metrics[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.kms_key_arn != null ? "aws:kms" : "AES256"
      kms_master_key_id = var.kms_key_arn
    }
    bucket_key_enabled = var.kms_key_arn != null ? true : false
  }
}

# -----------------------------------------------------------------------------
# Block Public Access
# -----------------------------------------------------------------------------
# All public access is blocked - metrics data should never be public.

resource "aws_s3_bucket_public_access_block" "thanos_metrics" {
  count = var.enable_thanos ? 1 : 0

  bucket = aws_s3_bucket.thanos_metrics[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -----------------------------------------------------------------------------
# Lifecycle Rules
# -----------------------------------------------------------------------------
# Transition to Intelligent Tiering after 30 days for cost optimization.
# Delete objects after 730 days (2 years) as per retention policy.

resource "aws_s3_bucket_lifecycle_configuration" "thanos_metrics" {
  count = var.enable_thanos ? 1 : 0

  bucket = aws_s3_bucket.thanos_metrics[0].id

  # Transition to Intelligent Tiering for cost optimization
  rule {
    id     = "intelligent-tiering"
    status = "Enabled"

    filter {
      prefix = ""
    }

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }

  # Delete old metric blocks after 2 years
  rule {
    id     = "delete-old-blocks"
    status = "Enabled"

    filter {
      prefix = ""
    }

    expiration {
      days = 730 # 2 years
    }

    # Clean up old versions after 30 days
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  # Clean up incomplete multipart uploads
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    filter {
      prefix = ""
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [aws_s3_bucket_versioning.thanos_metrics]
}

# -----------------------------------------------------------------------------
# Bucket Policy
# -----------------------------------------------------------------------------
# Enforces TLS-only access and allows Thanos IAM role access.

resource "aws_s3_bucket_policy" "thanos_metrics" {
  count = var.enable_thanos ? 1 : 0

  bucket = aws_s3_bucket.thanos_metrics[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyNonTLSAccess"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.thanos_metrics[0].arn,
          "${aws_s3_bucket.thanos_metrics[0].arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
      {
        Sid    = "AllowThanosAccess"
        Effect = "Allow"
        Principal = {
          AWS = var.create_iam_role ? aws_iam_role.prometheus_thanos[0].arn : var.existing_iam_role_arn
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.thanos_metrics[0].arn,
          "${aws_s3_bucket.thanos_metrics[0].arn}/*"
        ]
      }
    ]
  })

  depends_on = [
    aws_s3_bucket_public_access_block.thanos_metrics
  ]
}

# -----------------------------------------------------------------------------
# Access Logging (Optional)
# -----------------------------------------------------------------------------

resource "aws_s3_bucket_logging" "thanos_metrics" {
  count = var.enable_thanos && var.audit_logs_bucket != null ? 1 : 0

  bucket        = aws_s3_bucket.thanos_metrics[0].id
  target_bucket = var.audit_logs_bucket
  target_prefix = "s3-access-logs/thanos-metrics/"
}

# -----------------------------------------------------------------------------
# CORS Configuration
# -----------------------------------------------------------------------------
# Not needed for Thanos - only accessed by backend services.

resource "aws_s3_bucket_cors_configuration" "thanos_metrics" {
  count = var.enable_thanos ? 1 : 0

  bucket = aws_s3_bucket.thanos_metrics[0].id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}
