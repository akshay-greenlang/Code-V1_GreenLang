# =============================================================================
# KMS Module Outputs (SEC-003)
# =============================================================================

# -----------------------------------------------------------------------------
# Master Key
# -----------------------------------------------------------------------------

output "master_key_arn" {
  description = "ARN of the master CMK"
  value       = aws_kms_key.master.arn
}

output "master_key_id" {
  description = "ID of the master CMK"
  value       = aws_kms_key.master.key_id
}

output "master_key_alias" {
  description = "Alias of the master CMK"
  value       = aws_kms_alias.master.name
}

output "master_key_alias_arn" {
  description = "ARN of the master CMK alias"
  value       = aws_kms_alias.master.arn
}

# -----------------------------------------------------------------------------
# Database Key
# -----------------------------------------------------------------------------

output "database_key_arn" {
  description = "ARN of the database CMK for Aurora/RDS encryption"
  value       = var.create_database_key ? aws_kms_key.database[0].arn : null
}

output "database_key_id" {
  description = "ID of the database CMK"
  value       = var.create_database_key ? aws_kms_key.database[0].key_id : null
}

output "database_key_alias" {
  description = "Alias of the database CMK"
  value       = var.create_database_key ? aws_kms_alias.database[0].name : null
}

output "database_key_alias_arn" {
  description = "ARN of the database CMK alias"
  value       = var.create_database_key ? aws_kms_alias.database[0].arn : null
}

# -----------------------------------------------------------------------------
# Storage Key
# -----------------------------------------------------------------------------

output "storage_key_arn" {
  description = "ARN of the storage CMK for S3/EFS encryption"
  value       = var.create_storage_key ? aws_kms_key.storage[0].arn : null
}

output "storage_key_id" {
  description = "ID of the storage CMK"
  value       = var.create_storage_key ? aws_kms_key.storage[0].key_id : null
}

output "storage_key_alias" {
  description = "Alias of the storage CMK"
  value       = var.create_storage_key ? aws_kms_alias.storage[0].name : null
}

output "storage_key_alias_arn" {
  description = "ARN of the storage CMK alias"
  value       = var.create_storage_key ? aws_kms_alias.storage[0].arn : null
}

# -----------------------------------------------------------------------------
# Cache Key
# -----------------------------------------------------------------------------

output "cache_key_arn" {
  description = "ARN of the cache CMK for ElastiCache encryption"
  value       = var.create_cache_key ? aws_kms_key.cache[0].arn : null
}

output "cache_key_id" {
  description = "ID of the cache CMK"
  value       = var.create_cache_key ? aws_kms_key.cache[0].key_id : null
}

output "cache_key_alias" {
  description = "Alias of the cache CMK"
  value       = var.create_cache_key ? aws_kms_alias.cache[0].name : null
}

output "cache_key_alias_arn" {
  description = "ARN of the cache CMK alias"
  value       = var.create_cache_key ? aws_kms_alias.cache[0].arn : null
}

# -----------------------------------------------------------------------------
# Secrets Key
# -----------------------------------------------------------------------------

output "secrets_key_arn" {
  description = "ARN of the secrets CMK for Secrets Manager/Parameter Store"
  value       = var.create_secrets_key ? aws_kms_key.secrets[0].arn : null
}

output "secrets_key_id" {
  description = "ID of the secrets CMK"
  value       = var.create_secrets_key ? aws_kms_key.secrets[0].key_id : null
}

output "secrets_key_alias" {
  description = "Alias of the secrets CMK"
  value       = var.create_secrets_key ? aws_kms_alias.secrets[0].name : null
}

output "secrets_key_alias_arn" {
  description = "ARN of the secrets CMK alias"
  value       = var.create_secrets_key ? aws_kms_alias.secrets[0].arn : null
}

# -----------------------------------------------------------------------------
# Application Key
# -----------------------------------------------------------------------------

output "application_key_arn" {
  description = "ARN of the application CMK for envelope encryption (DEKs)"
  value       = var.create_application_key ? aws_kms_key.application[0].arn : null
}

output "application_key_id" {
  description = "ID of the application CMK"
  value       = var.create_application_key ? aws_kms_key.application[0].key_id : null
}

output "application_key_alias" {
  description = "Alias of the application CMK"
  value       = var.create_application_key ? aws_kms_alias.application[0].name : null
}

output "application_key_alias_arn" {
  description = "ARN of the application CMK alias"
  value       = var.create_application_key ? aws_kms_alias.application[0].arn : null
}

# -----------------------------------------------------------------------------
# EKS Key
# -----------------------------------------------------------------------------

output "eks_key_arn" {
  description = "ARN of the EKS CMK for Kubernetes secrets encryption"
  value       = var.create_eks_key ? aws_kms_key.eks[0].arn : null
}

output "eks_key_id" {
  description = "ID of the EKS CMK"
  value       = var.create_eks_key ? aws_kms_key.eks[0].key_id : null
}

output "eks_key_alias" {
  description = "Alias of the EKS CMK"
  value       = var.create_eks_key ? aws_kms_alias.eks[0].name : null
}

output "eks_key_alias_arn" {
  description = "ARN of the EKS CMK alias"
  value       = var.create_eks_key ? aws_kms_alias.eks[0].arn : null
}

# -----------------------------------------------------------------------------
# Backup Key
# -----------------------------------------------------------------------------

output "backup_key_arn" {
  description = "ARN of the backup CMK for pgBackRest/Redis backup encryption"
  value       = var.create_backup_key ? aws_kms_key.backup[0].arn : null
}

output "backup_key_id" {
  description = "ID of the backup CMK"
  value       = var.create_backup_key ? aws_kms_key.backup[0].key_id : null
}

output "backup_key_alias" {
  description = "Alias of the backup CMK"
  value       = var.create_backup_key ? aws_kms_alias.backup[0].name : null
}

output "backup_key_alias_arn" {
  description = "ARN of the backup CMK alias"
  value       = var.create_backup_key ? aws_kms_alias.backup[0].arn : null
}

# -----------------------------------------------------------------------------
# CloudWatch Log Group
# -----------------------------------------------------------------------------

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group for KMS audit logging"
  value       = var.enable_cloudwatch_logging ? aws_cloudwatch_log_group.kms_audit[0].arn : null
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for KMS audit logging"
  value       = var.enable_cloudwatch_logging ? aws_cloudwatch_log_group.kms_audit[0].name : null
}

# -----------------------------------------------------------------------------
# All Keys Map (for dynamic access)
# -----------------------------------------------------------------------------

output "all_keys" {
  description = "Map of all created key ARNs by type for dynamic access"
  value = {
    master      = aws_kms_key.master.arn
    database    = var.create_database_key ? aws_kms_key.database[0].arn : null
    storage     = var.create_storage_key ? aws_kms_key.storage[0].arn : null
    cache       = var.create_cache_key ? aws_kms_key.cache[0].arn : null
    secrets     = var.create_secrets_key ? aws_kms_key.secrets[0].arn : null
    application = var.create_application_key ? aws_kms_key.application[0].arn : null
    eks         = var.create_eks_key ? aws_kms_key.eks[0].arn : null
    backup      = var.create_backup_key ? aws_kms_key.backup[0].arn : null
  }
}

output "all_key_ids" {
  description = "Map of all created key IDs by type for dynamic access"
  value = {
    master      = aws_kms_key.master.key_id
    database    = var.create_database_key ? aws_kms_key.database[0].key_id : null
    storage     = var.create_storage_key ? aws_kms_key.storage[0].key_id : null
    cache       = var.create_cache_key ? aws_kms_key.cache[0].key_id : null
    secrets     = var.create_secrets_key ? aws_kms_key.secrets[0].key_id : null
    application = var.create_application_key ? aws_kms_key.application[0].key_id : null
    eks         = var.create_eks_key ? aws_kms_key.eks[0].key_id : null
    backup      = var.create_backup_key ? aws_kms_key.backup[0].key_id : null
  }
}

output "all_key_aliases" {
  description = "Map of all created key aliases by type for dynamic access"
  value = {
    master      = aws_kms_alias.master.name
    database    = var.create_database_key ? aws_kms_alias.database[0].name : null
    storage     = var.create_storage_key ? aws_kms_alias.storage[0].name : null
    cache       = var.create_cache_key ? aws_kms_alias.cache[0].name : null
    secrets     = var.create_secrets_key ? aws_kms_alias.secrets[0].name : null
    application = var.create_application_key ? aws_kms_alias.application[0].name : null
    eks         = var.create_eks_key ? aws_kms_alias.eks[0].name : null
    backup      = var.create_backup_key ? aws_kms_alias.backup[0].name : null
  }
}

# -----------------------------------------------------------------------------
# Summary (for documentation/debugging)
# -----------------------------------------------------------------------------

output "key_summary" {
  description = "Summary of all created keys with their configurations"
  value = {
    environment          = var.environment
    key_rotation_enabled = var.enable_key_rotation
    multi_region_enabled = var.enable_multi_region
    deletion_window_days = var.deletion_window_days
    keys_created = {
      master      = true
      database    = var.create_database_key
      storage     = var.create_storage_key
      cache       = var.create_cache_key
      secrets     = var.create_secrets_key
      application = var.create_application_key
      eks         = var.create_eks_key
      backup      = var.create_backup_key
    }
  }
}
