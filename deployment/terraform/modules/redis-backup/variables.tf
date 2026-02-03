# ==============================================================================
# Redis Backup Module - Variables
# ==============================================================================

# ==============================================================================
# Project Configuration
# ==============================================================================

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "greenlang"
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production", "dr"], var.environment)
    error_message = "Environment must be one of: dev, staging, production, dr."
  }
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# ==============================================================================
# S3 Bucket Configuration
# ==============================================================================

variable "backup_prefix" {
  description = "S3 key prefix for backup files"
  type        = string
  default     = "redis-backups"
}

variable "force_destroy_bucket" {
  description = "Allow bucket deletion even when not empty (use with caution)"
  type        = bool
  default     = false
}

variable "enable_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

# ==============================================================================
# Retention Configuration
# ==============================================================================

variable "rdb_retention_days" {
  description = "Number of days to retain RDB backups before deletion"
  type        = number
  default     = 90

  validation {
    condition     = var.rdb_retention_days >= 7
    error_message = "RDB retention must be at least 7 days."
  }
}

variable "rdb_glacier_transition_days" {
  description = "Number of days before transitioning RDB backups to Glacier"
  type        = number
  default     = 30

  validation {
    condition     = var.rdb_glacier_transition_days >= 30
    error_message = "Glacier transition must be at least 30 days."
  }
}

variable "aof_retention_days" {
  description = "Number of days to retain AOF backups before deletion"
  type        = number
  default     = 7

  validation {
    condition     = var.aof_retention_days >= 1
    error_message = "AOF retention must be at least 1 day."
  }
}

variable "archive_retention_days" {
  description = "Number of days to retain archived backups"
  type        = number
  default     = 2555  # ~7 years for compliance

  validation {
    condition     = var.archive_retention_days >= 365
    error_message = "Archive retention must be at least 365 days."
  }
}

variable "noncurrent_version_retention_days" {
  description = "Number of days to retain noncurrent object versions"
  type        = number
  default     = 30
}

variable "enable_archive_tier" {
  description = "Enable Deep Archive tier for long-term backup storage"
  type        = bool
  default     = true
}

# ==============================================================================
# Encryption Configuration
# ==============================================================================

variable "enable_kms_encryption" {
  description = "Enable KMS encryption for S3 bucket (SSE-KMS instead of SSE-S3)"
  type        = bool
  default     = true
}

variable "kms_deletion_window_days" {
  description = "Number of days before KMS key deletion"
  type        = number
  default     = 30

  validation {
    condition     = var.kms_deletion_window_days >= 7 && var.kms_deletion_window_days <= 30
    error_message = "KMS deletion window must be between 7 and 30 days."
  }
}

# ==============================================================================
# IAM Configuration
# ==============================================================================

variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS OIDC provider for IRSA (IAM Roles for Service Accounts)"
  type        = string
  default     = ""
}

variable "kubernetes_namespace" {
  description = "Kubernetes namespace where backup jobs run"
  type        = string
  default     = "greenlang"
}

variable "enable_secrets_manager_access" {
  description = "Enable IAM permissions for Secrets Manager access"
  type        = bool
  default     = true
}

# ==============================================================================
# Logging and Monitoring
# ==============================================================================

variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch Logs for backup job logging"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms for backup monitoring"
  type        = bool
  default     = true
}

variable "alarm_sns_topic_arns" {
  description = "List of SNS topic ARNs for CloudWatch alarm notifications"
  type        = list(string)
  default     = []
}

variable "min_backup_size_bytes" {
  description = "Minimum expected backup size in bytes (for anomaly detection)"
  type        = number
  default     = 1024  # 1 KB minimum
}

# ==============================================================================
# Notifications
# ==============================================================================

variable "enable_sns_notifications" {
  description = "Enable SNS notifications for S3 bucket events"
  type        = bool
  default     = false
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for S3 bucket notifications"
  type        = string
  default     = ""
}

# ==============================================================================
# Replication (for Disaster Recovery)
# ==============================================================================

variable "enable_cross_region_replication" {
  description = "Enable cross-region replication for disaster recovery"
  type        = bool
  default     = false
}

variable "replication_destination_bucket_arn" {
  description = "ARN of the destination bucket for cross-region replication"
  type        = string
  default     = ""
}

variable "replication_destination_region" {
  description = "AWS region for the replication destination bucket"
  type        = string
  default     = ""
}
