# =============================================================================
# pgBackRest Terraform Module - Outputs
# GreenLang Database Infrastructure
# =============================================================================

# -----------------------------------------------------------------------------
# S3 Bucket Outputs
# -----------------------------------------------------------------------------
output "s3_bucket_id" {
  description = "ID of the S3 bucket for pgBackRest backups"
  value       = aws_s3_bucket.pgbackrest.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for pgBackRest backups"
  value       = aws_s3_bucket.pgbackrest.arn
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for pgBackRest backups"
  value       = aws_s3_bucket.pgbackrest.bucket
}

output "s3_bucket_region" {
  description = "Region of the S3 bucket"
  value       = aws_s3_bucket.pgbackrest.region
}

output "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  value       = aws_s3_bucket.pgbackrest.bucket_domain_name
}

# -----------------------------------------------------------------------------
# KMS Key Outputs
# -----------------------------------------------------------------------------
output "kms_key_id" {
  description = "ID of the KMS key for backup encryption"
  value       = aws_kms_key.pgbackrest.key_id
}

output "kms_key_arn" {
  description = "ARN of the KMS key for backup encryption"
  value       = aws_kms_key.pgbackrest.arn
}

output "kms_key_alias" {
  description = "Alias of the KMS key"
  value       = aws_kms_alias.pgbackrest.name
}

output "secrets_kms_key_arn" {
  description = "ARN of the KMS key for secrets encryption"
  value       = var.create_secrets_kms_key ? aws_kms_key.pgbackrest_secrets[0].arn : null
}

# -----------------------------------------------------------------------------
# IAM Role Outputs
# -----------------------------------------------------------------------------
output "backup_role_arn" {
  description = "ARN of the IAM role for backup operations"
  value       = aws_iam_role.pgbackrest_backup.arn
}

output "backup_role_name" {
  description = "Name of the IAM role for backup operations"
  value       = aws_iam_role.pgbackrest_backup.name
}

output "restore_role_arn" {
  description = "ARN of the IAM role for restore operations"
  value       = aws_iam_role.pgbackrest_restore.arn
}

output "restore_role_name" {
  description = "Name of the IAM role for restore operations"
  value       = aws_iam_role.pgbackrest_restore.name
}

output "backup_instance_profile_arn" {
  description = "ARN of the instance profile for backup operations"
  value       = var.create_instance_profile ? aws_iam_instance_profile.pgbackrest_backup[0].arn : null
}

output "restore_instance_profile_arn" {
  description = "ARN of the instance profile for restore operations"
  value       = var.create_instance_profile ? aws_iam_instance_profile.pgbackrest_restore[0].arn : null
}

# -----------------------------------------------------------------------------
# IAM User Outputs (if created)
# -----------------------------------------------------------------------------
output "iam_user_arn" {
  description = "ARN of the IAM user for pgBackRest"
  value       = var.create_iam_user ? aws_iam_user.pgbackrest[0].arn : null
}

output "iam_user_name" {
  description = "Name of the IAM user for pgBackRest"
  value       = var.create_iam_user ? aws_iam_user.pgbackrest[0].name : null
}

output "credentials_secret_arn" {
  description = "ARN of the Secrets Manager secret containing S3 credentials"
  value       = var.create_iam_user ? aws_secretsmanager_secret.pgbackrest_credentials[0].arn : null
}

# -----------------------------------------------------------------------------
# CloudWatch Outputs
# -----------------------------------------------------------------------------
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = var.create_cloudwatch_log_group ? aws_cloudwatch_log_group.pgbackrest[0].name : null
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = var.create_cloudwatch_log_group ? aws_cloudwatch_log_group.pgbackrest[0].arn : null
}

# -----------------------------------------------------------------------------
# Secrets Manager Outputs
# -----------------------------------------------------------------------------
output "encryption_passphrase_secret_arn" {
  description = "ARN of the secret containing the encryption passphrase"
  value       = var.create_encryption_passphrase_secret ? aws_secretsmanager_secret.encryption_passphrase[0].arn : null
}

# -----------------------------------------------------------------------------
# pgBackRest Configuration Outputs
# -----------------------------------------------------------------------------
output "pgbackrest_config" {
  description = "pgBackRest configuration snippet for S3 repository"
  value = {
    repo_type       = "s3"
    repo_s3_bucket  = aws_s3_bucket.pgbackrest.bucket
    repo_s3_region  = aws_s3_bucket.pgbackrest.region
    repo_s3_endpoint = "s3.${aws_s3_bucket.pgbackrest.region}.amazonaws.com"
    repo_path       = "/pgbackrest/${var.project_name}"
    cipher_type     = "aes-256-cbc"
  }
}

output "kubernetes_service_account_annotations" {
  description = "Annotations for Kubernetes service accounts for IRSA"
  value = {
    backup = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.pgbackrest_backup.arn
    }
    restore = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.pgbackrest_restore.arn
    }
  }
}

# -----------------------------------------------------------------------------
# Replication Outputs
# -----------------------------------------------------------------------------
output "replication_role_arn" {
  description = "ARN of the replication IAM role"
  value       = var.enable_cross_region_replication ? aws_iam_role.replication[0].arn : null
}
