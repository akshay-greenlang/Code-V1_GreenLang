# =============================================================================
# GreenLang Grafana Module - S3
# GreenLang Climate OS | OBS-002
# =============================================================================
# S3 bucket for Grafana image rendering storage
# =============================================================================

# ---------------------------------------------------------------------------
# S3 Bucket for Rendered Images
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "grafana_images" {
  bucket = "${var.s3_bucket_prefix}-grafana-images-${var.environment}"

  tags = merge(local.default_tags, {
    Name      = "${var.s3_bucket_prefix}-grafana-images-${var.environment}"
    DataClass = "operational"
    Retention = "${var.s3_image_retention_days}-days"
  })
}

# ---------------------------------------------------------------------------
# Block Public Access
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_public_access_block" "grafana_images" {
  bucket = aws_s3_bucket.grafana_images.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# Encryption (SSE-KMS or SSE-S3)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_server_side_encryption_configuration" "grafana_images" {
  bucket = aws_s3_bucket.grafana_images.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = local.kms_key_arn != null ? "aws:kms" : "AES256"
      kms_master_key_id = local.kms_key_arn
    }
    bucket_key_enabled = local.kms_key_arn != null
  }
}

# ---------------------------------------------------------------------------
# Lifecycle Rules
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_lifecycle_configuration" "grafana_images" {
  bucket = aws_s3_bucket.grafana_images.id

  rule {
    id     = "delete-old-images"
    status = "Enabled"

    expiration {
      days = var.s3_image_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

# ---------------------------------------------------------------------------
# Versioning (Disabled for ephemeral images)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_versioning" "grafana_images" {
  bucket = aws_s3_bucket.grafana_images.id

  versioning_configuration {
    status = "Disabled"
  }
}

# ---------------------------------------------------------------------------
# Bucket Policy - HTTPS Only
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_policy" "grafana_images" {
  bucket = aws_s3_bucket.grafana_images.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "EnforceTLS"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.grafana_images.arn,
          "${aws_s3_bucket.grafana_images.arn}/*"
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
