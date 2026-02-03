# =============================================================================
# pgBackRest Terraform Module - Main Configuration
# GreenLang Database Infrastructure
# =============================================================================
# This module provisions AWS infrastructure for pgBackRest backup storage
# including S3 bucket, IAM roles/policies, and KMS encryption.
# =============================================================================

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.0"
    }
  }
}

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------
locals {
  name_prefix = "${var.project_name}-${var.environment}"

  default_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Component   = "pgbackrest"
  }

  tags = merge(local.default_tags, var.additional_tags)

  # S3 bucket name with random suffix for uniqueness
  bucket_name = var.s3_bucket_name != "" ? var.s3_bucket_name : "${local.name_prefix}-pgbackrest-${random_id.bucket_suffix.hex}"

  # Lifecycle rules based on retention policies
  lifecycle_rules = [
    {
      id      = "archive-old-backups"
      enabled = true
      prefix  = "pgbackrest/"

      transitions = [
        {
          days          = var.transition_to_ia_days
          storage_class = "STANDARD_IA"
        },
        {
          days          = var.transition_to_glacier_days
          storage_class = "GLACIER"
        }
      ]

      expiration = {
        days = var.backup_retention_days
      }

      noncurrent_version_transitions = [
        {
          noncurrent_days = 30
          storage_class   = "STANDARD_IA"
        },
        {
          noncurrent_days = 90
          storage_class   = "GLACIER"
        }
      ]

      noncurrent_version_expiration = {
        noncurrent_days = var.noncurrent_version_retention_days
      }
    }
  ]
}

# -----------------------------------------------------------------------------
# Random Suffix for S3 Bucket
# -----------------------------------------------------------------------------
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# -----------------------------------------------------------------------------
# S3 Bucket for pgBackRest Backups
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "pgbackrest" {
  bucket        = local.bucket_name
  force_destroy = var.force_destroy_bucket

  tags = merge(local.tags, {
    Name = local.bucket_name
  })
}

# Bucket versioning
resource "aws_s3_bucket_versioning" "pgbackrest" {
  bucket = aws_s3_bucket.pgbackrest.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

# Bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "pgbackrest" {
  bucket = aws_s3_bucket.pgbackrest.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.pgbackrest.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Public access block
resource "aws_s3_bucket_public_access_block" "pgbackrest" {
  bucket = aws_s3_bucket.pgbackrest.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket lifecycle configuration
resource "aws_s3_bucket_lifecycle_configuration" "pgbackrest" {
  bucket = aws_s3_bucket.pgbackrest.id

  rule {
    id     = "archive-old-backups"
    status = "Enabled"

    filter {
      prefix = "pgbackrest/"
    }

    # Transition to Infrequent Access
    transition {
      days          = var.transition_to_ia_days
      storage_class = "STANDARD_IA"
    }

    # Transition to Glacier
    transition {
      days          = var.transition_to_glacier_days
      storage_class = "GLACIER"
    }

    # Expire old backups
    expiration {
      days = var.backup_retention_days
    }

    # Noncurrent version transitions
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    # Noncurrent version expiration
    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_retention_days
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

  rule {
    id     = "intelligent-tiering"
    status = var.enable_intelligent_tiering ? "Enabled" : "Disabled"

    filter {
      prefix = "pgbackrest/"
    }

    transition {
      days          = 0
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}

# Bucket policy
resource "aws_s3_bucket_policy" "pgbackrest" {
  bucket = aws_s3_bucket.pgbackrest.id
  policy = data.aws_iam_policy_document.bucket_policy.json

  depends_on = [aws_s3_bucket_public_access_block.pgbackrest]
}

data "aws_iam_policy_document" "bucket_policy" {
  # Enforce SSL
  statement {
    sid    = "EnforceSSL"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["s3:*"]

    resources = [
      aws_s3_bucket.pgbackrest.arn,
      "${aws_s3_bucket.pgbackrest.arn}/*"
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }

  # Allow pgBackRest role access
  statement {
    sid    = "AllowPgBackRestAccess"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.pgbackrest_backup.arn, aws_iam_role.pgbackrest_restore.arn]
    }

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]

    resources = [
      aws_s3_bucket.pgbackrest.arn,
      "${aws_s3_bucket.pgbackrest.arn}/*"
    ]
  }

  # Restrict bucket access to specific VPC (optional)
  dynamic "statement" {
    for_each = var.restrict_to_vpc && var.vpc_id != "" ? [1] : []
    content {
      sid    = "RestrictToVPC"
      effect = "Deny"

      principals {
        type        = "*"
        identifiers = ["*"]
      }

      actions = ["s3:*"]

      resources = [
        aws_s3_bucket.pgbackrest.arn,
        "${aws_s3_bucket.pgbackrest.arn}/*"
      ]

      condition {
        test     = "StringNotEquals"
        variable = "aws:SourceVpc"
        values   = [var.vpc_id]
      }

      condition {
        test     = "Bool"
        variable = "aws:PrincipalIsAWSService"
        values   = ["false"]
      }
    }
  }
}

# Enable bucket logging
resource "aws_s3_bucket_logging" "pgbackrest" {
  count = var.enable_access_logging ? 1 : 0

  bucket = aws_s3_bucket.pgbackrest.id

  target_bucket = var.logging_bucket_name
  target_prefix = "pgbackrest-access-logs/"
}

# Enable bucket replication for disaster recovery
resource "aws_s3_bucket_replication_configuration" "pgbackrest" {
  count = var.enable_cross_region_replication ? 1 : 0

  bucket = aws_s3_bucket.pgbackrest.id
  role   = aws_iam_role.replication[0].arn

  rule {
    id     = "cross-region-replication"
    status = "Enabled"

    filter {
      prefix = "pgbackrest/"
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

  depends_on = [aws_s3_bucket_versioning.pgbackrest]
}

# Replication IAM role
resource "aws_iam_role" "replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  name = "${local.name_prefix}-pgbackrest-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  name = "${local.name_prefix}-pgbackrest-replication-policy"
  role = aws_iam_role.replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.pgbackrest.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = "${aws_s3_bucket.pgbackrest.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = "${var.replication_destination_bucket_arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = aws_kms_key.pgbackrest.arn
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt"
        ]
        Resource = var.replication_destination_kms_key_arn
      }
    ]
  })
}
