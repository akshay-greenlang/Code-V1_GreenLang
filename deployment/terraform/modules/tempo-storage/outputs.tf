# =============================================================================
# GreenLang Tempo Storage Module - Outputs
# GreenLang Climate OS | OBS-003
# =============================================================================

# -----------------------------------------------------------------------------
# S3 Bucket
# -----------------------------------------------------------------------------

output "bucket_name" {
  description = "Name of the S3 bucket for Tempo trace blocks"
  value       = aws_s3_bucket.traces.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket for Tempo trace blocks"
  value       = aws_s3_bucket.traces.arn
}

output "bucket_domain_name" {
  description = "Regional domain name of the S3 bucket for Tempo trace blocks"
  value       = aws_s3_bucket.traces.bucket_regional_domain_name
}

# -----------------------------------------------------------------------------
# KMS
# -----------------------------------------------------------------------------

output "kms_key_arn" {
  description = "ARN of the KMS key used for Tempo S3 encryption"
  value       = aws_kms_key.tempo.arn
}

output "kms_key_id" {
  description = "ID of the KMS key used for Tempo S3 encryption"
  value       = aws_kms_key.tempo.key_id
}

# -----------------------------------------------------------------------------
# IAM / IRSA
# -----------------------------------------------------------------------------

output "irsa_role_arn" {
  description = "ARN of the IAM role for Tempo IRSA (use in serviceAccount annotations)"
  value       = aws_iam_role.tempo.arn
}

output "irsa_role_name" {
  description = "Name of the IAM role for Tempo IRSA"
  value       = aws_iam_role.tempo.name
}

output "bucket_policy_arn" {
  description = "ARN of the IAM policy granting Tempo S3 and KMS access"
  value       = aws_iam_policy.tempo_s3.arn
}
