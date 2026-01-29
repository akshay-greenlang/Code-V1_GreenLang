# outputs.tf - S3 Module Outputs

# Audit Bucket
output "audit_bucket_id" {
  description = "Audit bucket ID"
  value       = aws_s3_bucket.audit.id
}

output "audit_bucket_arn" {
  description = "Audit bucket ARN"
  value       = aws_s3_bucket.audit.arn
}

output "audit_bucket_domain_name" {
  description = "Audit bucket domain name"
  value       = aws_s3_bucket.audit.bucket_domain_name
}

output "audit_bucket_regional_domain_name" {
  description = "Audit bucket regional domain name"
  value       = aws_s3_bucket.audit.bucket_regional_domain_name
}

# Vocabulary Bucket
output "vocabulary_bucket_id" {
  description = "Vocabulary bucket ID"
  value       = aws_s3_bucket.vocabulary.id
}

output "vocabulary_bucket_arn" {
  description = "Vocabulary bucket ARN"
  value       = aws_s3_bucket.vocabulary.arn
}

# Access Logs Bucket
output "access_logs_bucket_id" {
  description = "Access logs bucket ID"
  value       = var.enable_access_logging ? aws_s3_bucket.access_logs[0].id : null
}

output "access_logs_bucket_arn" {
  description = "Access logs bucket ARN"
  value       = var.enable_access_logging ? aws_s3_bucket.access_logs[0].arn : null
}

# KMS Key
output "kms_key_arn" {
  description = "KMS key ARN used for S3 encryption"
  value       = aws_kms_key.s3.arn
}

output "kms_key_id" {
  description = "KMS key ID"
  value       = aws_kms_key.s3.key_id
}

# Replication
output "replication_role_arn" {
  description = "IAM role ARN for S3 replication"
  value       = var.enable_cross_region_replication ? aws_iam_role.replication[0].arn : null
}
