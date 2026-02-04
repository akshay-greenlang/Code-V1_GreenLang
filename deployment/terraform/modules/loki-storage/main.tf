# =============================================================================
# GreenLang Loki Storage Module
# GreenLang Climate OS | INFRA-009
# =============================================================================
# Creates S3 buckets for Loki log storage (chunks and ruler), with:
#   - KMS server-side encryption
#   - Public access blocking
#   - TLS-only bucket policies
#   - Lifecycle rules (STANDARD_IA after 30d, GLACIER after 90d for chunks)
#   - Access logging to a shared audit-logs bucket
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

# -----------------------------------------------------------------------------
# KMS Key (created if kms_key_arn is not provided)
# -----------------------------------------------------------------------------
resource "aws_kms_key" "loki" {
  count = var.kms_key_arn == null ? 1 : 0

  description             = "KMS key for Loki S3 bucket encryption - ${var.project}-${var.environment}"
  deletion_window_in_days = 14
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
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
        Sid    = "AllowS3Service"
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
    Name      = "${var.project}-loki-kms-${var.environment}"
    Component = "loki"
  })
}

resource "aws_kms_alias" "loki" {
  count = var.kms_key_arn == null ? 1 : 0

  name          = "alias/${var.project}-loki-${var.environment}"
  target_key_id = aws_kms_key.loki[0].key_id
}

locals {
  kms_key_arn = var.kms_key_arn != null ? var.kms_key_arn : aws_kms_key.loki[0].arn
}

# =============================================================================
# Chunks Bucket
# =============================================================================
# Stores Loki log chunk data. This is the primary storage bucket and receives
# the highest write throughput. Lifecycle rules transition old chunks to
# cheaper storage tiers.
# =============================================================================

resource "aws_s3_bucket" "chunks" {
  bucket        = "${var.project}-loki-chunks-${var.environment}"
  force_destroy = false

  tags = merge(var.tags, {
    Name      = "${var.project}-loki-chunks-${var.environment}"
    Component = "loki"
    Purpose   = "log-chunks"
  })
}

# Versioning disabled - Loki manages its own data lifecycle
resource "aws_s3_bucket_versioning" "chunks" {
  bucket = aws_s3_bucket.chunks.id

  versioning_configuration {
    status = "Disabled"
  }
}

# Server-side encryption with KMS
resource "aws_s3_bucket_server_side_encryption_configuration" "chunks" {
  bucket = aws_s3_bucket.chunks.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = local.kms_key_arn
    }
    bucket_key_enabled = true
  }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "chunks" {
  bucket = aws_s3_bucket.chunks.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket policy - enforce TLS, deny HTTP
resource "aws_s3_bucket_policy" "chunks" {
  bucket = aws_s3_bucket.chunks.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyNonTLSAccess"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.chunks.arn,
          "${aws_s3_bucket.chunks.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.chunks]
}

# Lifecycle rules - transition to cheaper storage tiers
resource "aws_s3_bucket_lifecycle_configuration" "chunks" {
  bucket = aws_s3_bucket.chunks.id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# Access logging
resource "aws_s3_bucket_logging" "chunks" {
  count = var.audit_logs_bucket != null ? 1 : 0

  bucket        = aws_s3_bucket.chunks.id
  target_bucket = var.audit_logs_bucket
  target_prefix = "s3-access-logs/loki-chunks/"
}

# =============================================================================
# Ruler Bucket
# =============================================================================
# Stores Loki ruler configuration (alerting rules, recording rules).
# Much smaller than the chunks bucket. No lifecycle transitions needed
# since ruler configs should remain in STANDARD storage for fast access.
# =============================================================================

resource "aws_s3_bucket" "ruler" {
  bucket        = "${var.project}-loki-ruler-${var.environment}"
  force_destroy = false

  tags = merge(var.tags, {
    Name      = "${var.project}-loki-ruler-${var.environment}"
    Component = "loki"
    Purpose   = "ruler-config"
  })
}

# Versioning disabled
resource "aws_s3_bucket_versioning" "ruler" {
  bucket = aws_s3_bucket.ruler.id

  versioning_configuration {
    status = "Disabled"
  }
}

# Server-side encryption with KMS
resource "aws_s3_bucket_server_side_encryption_configuration" "ruler" {
  bucket = aws_s3_bucket.ruler.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = local.kms_key_arn
    }
    bucket_key_enabled = true
  }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "ruler" {
  bucket = aws_s3_bucket.ruler.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket policy - enforce TLS, deny HTTP
resource "aws_s3_bucket_policy" "ruler" {
  bucket = aws_s3_bucket.ruler.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyNonTLSAccess"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.ruler.arn,
          "${aws_s3_bucket.ruler.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.ruler]
}

# Access logging
resource "aws_s3_bucket_logging" "ruler" {
  count = var.audit_logs_bucket != null ? 1 : 0

  bucket        = aws_s3_bucket.ruler.id
  target_bucket = var.audit_logs_bucket
  target_prefix = "s3-access-logs/loki-ruler/"
}
