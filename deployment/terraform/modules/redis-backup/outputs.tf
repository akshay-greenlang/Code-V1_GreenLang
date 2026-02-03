# ==============================================================================
# Redis Backup Module - Outputs
# ==============================================================================

# ==============================================================================
# S3 Bucket Outputs
# ==============================================================================

output "bucket_id" {
  description = "The ID of the S3 bucket"
  value       = aws_s3_bucket.redis_backup.id
}

output "bucket_arn" {
  description = "The ARN of the S3 bucket"
  value       = aws_s3_bucket.redis_backup.arn
}

output "bucket_name" {
  description = "The name of the S3 bucket"
  value       = aws_s3_bucket.redis_backup.bucket
}

output "bucket_domain_name" {
  description = "The bucket domain name"
  value       = aws_s3_bucket.redis_backup.bucket_domain_name
}

output "bucket_regional_domain_name" {
  description = "The bucket region-specific domain name"
  value       = aws_s3_bucket.redis_backup.bucket_regional_domain_name
}

output "bucket_region" {
  description = "The AWS region of the S3 bucket"
  value       = aws_s3_bucket.redis_backup.region
}

# ==============================================================================
# KMS Key Outputs
# ==============================================================================

output "kms_key_id" {
  description = "The ID of the KMS key"
  value       = var.enable_kms_encryption ? aws_kms_key.redis_backup[0].key_id : null
}

output "kms_key_arn" {
  description = "The ARN of the KMS key"
  value       = var.enable_kms_encryption ? aws_kms_key.redis_backup[0].arn : null
}

output "kms_key_alias" {
  description = "The alias of the KMS key"
  value       = var.enable_kms_encryption ? aws_kms_alias.redis_backup[0].name : null
}

# ==============================================================================
# IAM Role Outputs
# ==============================================================================

output "iam_role_arn" {
  description = "The ARN of the IAM role for backup jobs"
  value       = aws_iam_role.redis_backup.arn
}

output "iam_role_name" {
  description = "The name of the IAM role for backup jobs"
  value       = aws_iam_role.redis_backup.name
}

output "iam_instance_profile_arn" {
  description = "The ARN of the IAM instance profile"
  value       = aws_iam_instance_profile.redis_backup.arn
}

output "iam_instance_profile_name" {
  description = "The name of the IAM instance profile"
  value       = aws_iam_instance_profile.redis_backup.name
}

# ==============================================================================
# CloudWatch Outputs
# ==============================================================================

output "cloudwatch_log_group_name" {
  description = "The name of the CloudWatch log group"
  value       = var.enable_cloudwatch_logs ? aws_cloudwatch_log_group.redis_backup[0].name : null
}

output "cloudwatch_log_group_arn" {
  description = "The ARN of the CloudWatch log group"
  value       = var.enable_cloudwatch_logs ? aws_cloudwatch_log_group.redis_backup[0].arn : null
}

# ==============================================================================
# Configuration Outputs
# ==============================================================================

output "backup_configuration" {
  description = "Backup configuration settings for Kubernetes ConfigMap"
  value = {
    s3_bucket            = aws_s3_bucket.redis_backup.bucket
    s3_region            = aws_s3_bucket.redis_backup.region
    s3_prefix            = var.backup_prefix
    encryption_enabled   = var.enable_kms_encryption
    kms_key_id           = var.enable_kms_encryption ? aws_kms_key.redis_backup[0].key_id : null
    rdb_retention_days   = var.rdb_retention_days
    aof_retention_days   = var.aof_retention_days
    iam_role_arn         = aws_iam_role.redis_backup.arn
  }
}

# ==============================================================================
# IRSA Configuration Output
# ==============================================================================

output "service_account_annotation" {
  description = "Annotation to add to Kubernetes service account for IRSA"
  value = {
    "eks.amazonaws.com/role-arn" = aws_iam_role.redis_backup.arn
  }
}

# ==============================================================================
# Backup Paths
# ==============================================================================

output "rdb_backup_path" {
  description = "S3 path for RDB backups"
  value       = "s3://${aws_s3_bucket.redis_backup.bucket}/${var.backup_prefix}/rdb/"
}

output "aof_backup_path" {
  description = "S3 path for AOF backups"
  value       = "s3://${aws_s3_bucket.redis_backup.bucket}/${var.backup_prefix}/aof/"
}

output "archive_backup_path" {
  description = "S3 path for archived backups"
  value       = "s3://${aws_s3_bucket.redis_backup.bucket}/${var.backup_prefix}/archive/"
}
