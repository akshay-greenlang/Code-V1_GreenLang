# =============================================================================
# KMS Module Variables (SEC-003)
# =============================================================================

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "greenlang"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,28}[a-z0-9]$", var.project_name))
    error_message = "Project name must be 4-30 characters, lowercase alphanumeric with hyphens, start with letter, end with alphanumeric"
  }
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod"
  }
}

# -----------------------------------------------------------------------------
# Key Creation Flags
# -----------------------------------------------------------------------------

variable "create_database_key" {
  description = "Create dedicated CMK for database encryption (Aurora PostgreSQL, RDS)"
  type        = bool
  default     = true
}

variable "create_storage_key" {
  description = "Create dedicated CMK for S3/EFS encryption"
  type        = bool
  default     = true
}

variable "create_cache_key" {
  description = "Create dedicated CMK for ElastiCache encryption"
  type        = bool
  default     = true
}

variable "create_secrets_key" {
  description = "Create dedicated CMK for Secrets Manager encryption"
  type        = bool
  default     = true
}

variable "create_application_key" {
  description = "Create dedicated CMK for application-level envelope encryption (DEKs)"
  type        = bool
  default     = true
}

variable "create_eks_key" {
  description = "Create dedicated CMK for EKS secrets encryption"
  type        = bool
  default     = true
}

variable "create_backup_key" {
  description = "Create dedicated CMK for backup encryption (pgBackRest, Redis RDB)"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# Key Configuration
# -----------------------------------------------------------------------------

variable "enable_key_rotation" {
  description = "Enable automatic annual key rotation (AWS-managed rotation)"
  type        = bool
  default     = true
}

variable "deletion_window_days" {
  description = "Waiting period before key deletion (7-30 days). Longer is safer for production."
  type        = number
  default     = 30

  validation {
    condition     = var.deletion_window_days >= 7 && var.deletion_window_days <= 30
    error_message = "Deletion window must be between 7 and 30 days"
  }
}

variable "enable_multi_region" {
  description = "Create multi-region keys for disaster recovery. Required for cross-region replication."
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# Access Control - Key Administrators
# -----------------------------------------------------------------------------

variable "key_administrators" {
  description = "IAM ARNs allowed to administer all keys (create, delete, update policies)"
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for arn in var.key_administrators : can(regex("^arn:aws[a-z-]*:iam::", arn))])
    error_message = "All key administrators must be valid IAM ARNs"
  }
}

# -----------------------------------------------------------------------------
# Access Control - Service Roles per Key Type
# -----------------------------------------------------------------------------

variable "database_service_roles" {
  description = "IAM role ARNs allowed to use database key for Aurora/RDS encryption"
  type        = list(string)
  default     = []
}

variable "storage_service_roles" {
  description = "IAM role ARNs allowed to use storage key for S3/EFS encryption"
  type        = list(string)
  default     = []
}

variable "cache_service_roles" {
  description = "IAM role ARNs allowed to use cache key for ElastiCache encryption"
  type        = list(string)
  default     = []
}

variable "secrets_service_roles" {
  description = "IAM role ARNs allowed to use secrets key for Secrets Manager/Parameter Store"
  type        = list(string)
  default     = []
}

variable "application_service_roles" {
  description = "IAM role ARNs allowed to use application key for envelope encryption (GenerateDataKey)"
  type        = list(string)
  default     = []
}

variable "eks_service_roles" {
  description = "IAM role ARNs allowed to use EKS key for Kubernetes secrets encryption"
  type        = list(string)
  default     = []
}

variable "backup_service_roles" {
  description = "IAM role ARNs allowed to use backup key for pgBackRest/Redis backup encryption"
  type        = list(string)
  default     = []
}

# -----------------------------------------------------------------------------
# Logging & Monitoring
# -----------------------------------------------------------------------------

variable "enable_cloudwatch_logging" {
  description = "Enable CloudWatch logging for KMS operations audit trail"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days. Must comply with data retention policies."
  type        = number
  default     = 365

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1096, 1827, 2192, 2557, 2922, 3288, 3653], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch Logs retention period"
  }
}

# -----------------------------------------------------------------------------
# Tags
# -----------------------------------------------------------------------------

variable "tags" {
  description = "Additional tags for all resources created by this module"
  type        = map(string)
  default     = {}
}
