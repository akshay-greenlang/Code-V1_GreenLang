# =============================================================================
# pgBackRest Terraform Module - Variables
# GreenLang Database Infrastructure
# =============================================================================

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "greenlang"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# -----------------------------------------------------------------------------
# S3 Bucket Configuration
# -----------------------------------------------------------------------------
variable "s3_bucket_name" {
  description = "Custom S3 bucket name. If empty, a name will be generated."
  type        = string
  default     = ""
}

variable "force_destroy_bucket" {
  description = "Whether to force destroy the bucket and all contents"
  type        = bool
  default     = false
}

variable "enable_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "enable_intelligent_tiering" {
  description = "Enable S3 Intelligent Tiering for automatic cost optimization"
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# Lifecycle Policy Configuration
# -----------------------------------------------------------------------------
variable "transition_to_ia_days" {
  description = "Days before transitioning objects to S3 Infrequent Access"
  type        = number
  default     = 30
}

variable "transition_to_glacier_days" {
  description = "Days before transitioning objects to S3 Glacier"
  type        = number
  default     = 90
}

variable "backup_retention_days" {
  description = "Days to retain backups before expiration"
  type        = number
  default     = 365
}

variable "noncurrent_version_retention_days" {
  description = "Days to retain noncurrent object versions"
  type        = number
  default     = 180
}

# -----------------------------------------------------------------------------
# Access Logging Configuration
# -----------------------------------------------------------------------------
variable "enable_access_logging" {
  description = "Enable S3 access logging"
  type        = bool
  default     = false
}

variable "logging_bucket_name" {
  description = "S3 bucket for access logs"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# VPC Configuration
# -----------------------------------------------------------------------------
variable "vpc_id" {
  description = "VPC ID for restricting bucket access"
  type        = string
  default     = ""
}

variable "restrict_to_vpc" {
  description = "Restrict S3 bucket access to specific VPC"
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# Cross-Region Replication Configuration
# -----------------------------------------------------------------------------
variable "enable_cross_region_replication" {
  description = "Enable cross-region replication for disaster recovery"
  type        = bool
  default     = false
}

variable "replication_destination_bucket_arn" {
  description = "ARN of the destination bucket for replication"
  type        = string
  default     = ""
}

variable "replication_destination_kms_key_arn" {
  description = "ARN of the KMS key in the destination region for replication"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# KMS Configuration
# -----------------------------------------------------------------------------
variable "kms_deletion_window_days" {
  description = "KMS key deletion window in days"
  type        = number
  default     = 30

  validation {
    condition     = var.kms_deletion_window_days >= 7 && var.kms_deletion_window_days <= 30
    error_message = "KMS deletion window must be between 7 and 30 days."
  }
}

variable "enable_kms_key_rotation" {
  description = "Enable automatic KMS key rotation"
  type        = bool
  default     = true
}

variable "enable_multi_region_kms" {
  description = "Enable multi-region KMS key"
  type        = bool
  default     = false
}

variable "kms_key_administrators" {
  description = "List of IAM ARNs that can administer the KMS key"
  type        = list(string)
  default     = []
}

variable "create_secrets_kms_key" {
  description = "Create a separate KMS key for secrets encryption"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# IAM Configuration
# -----------------------------------------------------------------------------
variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS OIDC provider for IRSA"
  type        = string
  default     = ""
}

variable "kubernetes_namespace" {
  description = "Kubernetes namespace for pgBackRest workloads"
  type        = string
  default     = "greenlang-database"
}

variable "create_instance_profile" {
  description = "Create IAM instance profiles for EC2 deployments"
  type        = bool
  default     = false
}

variable "create_iam_user" {
  description = "Create IAM user with access keys (not recommended for production)"
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# Notification Configuration
# -----------------------------------------------------------------------------
variable "sns_topic_arn" {
  description = "ARN of SNS topic for backup notifications"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Secrets Manager Configuration
# -----------------------------------------------------------------------------
variable "create_encryption_passphrase_secret" {
  description = "Create a secret for the encryption passphrase"
  type        = bool
  default     = true
}

variable "encryption_passphrase" {
  description = "Encryption passphrase for pgBackRest (if not auto-generated)"
  type        = string
  default     = ""
  sensitive   = true
}

# -----------------------------------------------------------------------------
# CloudWatch Configuration
# -----------------------------------------------------------------------------
variable "create_cloudwatch_log_group" {
  description = "Create CloudWatch log group for pgBackRest logs"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Days to retain CloudWatch logs"
  type        = number
  default     = 30
}

variable "create_cloudwatch_alarms" {
  description = "Create CloudWatch alarms for backup monitoring"
  type        = bool
  default     = true
}

variable "alarm_actions" {
  description = "List of ARNs to notify when alarms trigger"
  type        = list(string)
  default     = []
}

variable "ok_actions" {
  description = "List of ARNs to notify when alarms return to OK"
  type        = list(string)
  default     = []
}
