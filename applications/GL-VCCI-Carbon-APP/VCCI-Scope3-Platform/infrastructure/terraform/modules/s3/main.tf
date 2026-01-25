# S3 Buckets Module with Cross-Region Replication

terraform { required_providers { aws = { source = "hashicorp/aws"; configuration_aliases = [aws.secondary] } } }

locals {
  buckets = {
    provenance = "${var.project_name}-${var.environment}-provenance"
    raw_data   = "${var.project_name}-${var.environment}-raw-data"
    reports    = "${var.project_name}-${var.environment}-reports"
  }
}

# Replication IAM Role
resource "aws_iam_role" "replication" {
  count = var.enable_replication ? 1 : 0
  name  = "${var.project_name}-${var.environment}-s3-replication"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "s3.amazonaws.com" }
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy" "replication" {
  count = var.enable_replication ? 1 : 0
  role  = aws_iam_role.replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow"; Action = ["s3:GetReplicationConfiguration", "s3:ListBucket"]; Resource = [for b in aws_s3_bucket.main : b.arn] },
      { Effect = "Allow"; Action = ["s3:GetObjectVersionForReplication", "s3:GetObjectVersionAcl"]; Resource = [for b in aws_s3_bucket.main : "${b.arn}/*"] },
      { Effect = "Allow"; Action = ["s3:ReplicateObject", "s3:ReplicateDelete"]; Resource = [for b in aws_s3_bucket.replica : "${b.arn}/*"] }
    ]
  })
}

# Primary Buckets
resource "aws_s3_bucket" "main" {
  for_each = local.buckets
  bucket   = each.value
  tags     = merge(var.tags, { Name = each.value; Type = each.key })
}

resource "aws_s3_bucket_versioning" "main" {
  for_each = var.enable_versioning ? local.buckets : {}
  bucket   = aws_s3_bucket.main[each.key].id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.main[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = var.kms_key_arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  for_each                = local.buckets
  bucket                  = aws_s3_bucket.main[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "main" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.main[each.key].id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"
    transition { days = var.lifecycle_ia_transition_days; storage_class = "STANDARD_IA" }
  }

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"
    transition { days = var.lifecycle_glacier_transition_days; storage_class = "GLACIER" }
  }

  dynamic "rule" {
    for_each = var.lifecycle_expiration_days > 0 ? [1] : []
    content {
      id     = "expiration"
      status = "Enabled"
      expiration { days = var.lifecycle_expiration_days }
    }
  }
}

# Replica Buckets (Secondary Region)
resource "aws_s3_bucket" "replica" {
  for_each = var.enable_replication ? local.buckets : {}
  provider = aws.secondary
  bucket   = "${each.value}-replica"
  tags     = merge(var.tags, { Name = "${each.value}-replica"; Type = each.key; Region = "secondary" })
}

resource "aws_s3_bucket_versioning" "replica" {
  for_each = var.enable_replication ? local.buckets : {}
  provider = aws.secondary
  bucket   = aws_s3_bucket.replica[each.key].id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "replica" {
  for_each = var.enable_replication ? local.buckets : {}
  provider = aws.secondary
  bucket   = aws_s3_bucket.replica[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_replication_configuration" "main" {
  for_each = var.enable_replication ? local.buckets : {}
  depends_on = [aws_s3_bucket_versioning.main]
  role     = aws_iam_role.replication[0].arn
  bucket   = aws_s3_bucket.main[each.key].id

  rule {
    id     = "replicate-all"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.replica[each.key].arn
      storage_class = "STANDARD"
    }
  }
}
