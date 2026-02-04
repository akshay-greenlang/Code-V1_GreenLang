# =============================================================================
# GreenLang Loki Storage Module - Outputs
# GreenLang Climate OS | INFRA-009
# =============================================================================

# -----------------------------------------------------------------------------
# Chunks bucket
# -----------------------------------------------------------------------------

output "chunks_bucket_name" {
  description = "Name of the S3 bucket for Loki log chunks"
  value       = aws_s3_bucket.chunks.id
}

output "chunks_bucket_arn" {
  description = "ARN of the S3 bucket for Loki log chunks"
  value       = aws_s3_bucket.chunks.arn
}

# -----------------------------------------------------------------------------
# Ruler bucket
# -----------------------------------------------------------------------------

output "ruler_bucket_name" {
  description = "Name of the S3 bucket for Loki ruler configuration"
  value       = aws_s3_bucket.ruler.id
}

output "ruler_bucket_arn" {
  description = "ARN of the S3 bucket for Loki ruler configuration"
  value       = aws_s3_bucket.ruler.arn
}

# -----------------------------------------------------------------------------
# IAM
# -----------------------------------------------------------------------------

output "iam_role_arn" {
  description = "ARN of the IAM role for Loki IRSA (use in serviceAccount annotations)"
  value       = aws_iam_role.loki.arn
}

output "iam_role_name" {
  description = "Name of the IAM role for Loki IRSA"
  value       = aws_iam_role.loki.name
}

# -----------------------------------------------------------------------------
# KMS
# -----------------------------------------------------------------------------

output "kms_key_arn" {
  description = "ARN of the KMS key used for S3 encryption (null if external key was provided)"
  value       = var.kms_key_arn == null ? aws_kms_key.loki[0].arn : var.kms_key_arn
}
