# =============================================================================
# GreenLang Tempo Storage Module
# GreenLang Climate OS | OBS-003
# =============================================================================
# Creates an S3 bucket for Grafana Tempo trace block storage, with:
#   - KMS server-side encryption (dedicated CMK)
#   - Public access blocking
#   - TLS-only bucket policy
#   - Lifecycle rules (STANDARD_IA after ia_transition_days, expire at retention_days)
#   - Access logging to a shared audit-logs bucket
#   - IRSA IAM role for EKS pod access
#
# Tempo manages its own block lifecycle internally; S3 lifecycle rules act as
# a safety net to ensure stale data is cleaned up even if Tempo's compactor
# fails. Versioning is disabled because Tempo writes immutable blocks.
# =============================================================================

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

# =============================================================================
# S3 Bucket - Tempo Trace Blocks
# =============================================================================
# Stores Tempo trace blocks (Parquet files). This is the sole storage bucket
# for the Tempo deployment. Write throughput scales with ingestion rate;
# reads occur during queries and compaction.
# =============================================================================

resource "aws_s3_bucket" "traces" {
  bucket        = "${var.project}-${var.environment}-tempo-traces"
  force_destroy = false

  tags = merge(var.tags, {
    Name        = "${var.project}-${var.environment}-tempo-traces"
    Project     = "GreenLang"
    Component   = "Tempo"
    Environment = var.environment
  })
}

# -----------------------------------------------------------------------------
# Versioning - Disabled (Tempo writes immutable blocks)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_versioning" "traces" {
  bucket = aws_s3_bucket.traces.id

  versioning_configuration {
    status = "Disabled"
  }
}

# -----------------------------------------------------------------------------
# Server-side encryption with KMS
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_server_side_encryption_configuration" "traces" {
  bucket = aws_s3_bucket.traces.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.tempo.arn
    }
    bucket_key_enabled = true
  }
}

# -----------------------------------------------------------------------------
# Block all public access
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_public_access_block" "traces" {
  bucket = aws_s3_bucket.traces.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -----------------------------------------------------------------------------
# Bucket policy - enforce TLS, deny HTTP
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_policy" "traces" {
  bucket = aws_s3_bucket.traces.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyNonTLSAccess"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.traces.arn,
          "${aws_s3_bucket.traces.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.traces]
}

# -----------------------------------------------------------------------------
# Lifecycle rules
# -----------------------------------------------------------------------------
# Rule 1: Transition current objects to Infrequent Access after ia_transition_days
# Rule 2: Expire current objects after retention_days
# Rule 3: Clean up incomplete multipart uploads after 7 days
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_lifecycle_configuration" "traces" {
  bucket = aws_s3_bucket.traces.id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    transition {
      days          = var.ia_transition_days
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "expire-old-traces"
    status = "Enabled"

    expiration {
      days = var.retention_days
    }
  }

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# -----------------------------------------------------------------------------
# Access logging to centralized log bucket
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_logging" "traces" {
  count = var.log_bucket_name != "" ? 1 : 0

  bucket        = aws_s3_bucket.traces.id
  target_bucket = var.log_bucket_name
  target_prefix = "s3-access-logs/tempo-traces/"
}
