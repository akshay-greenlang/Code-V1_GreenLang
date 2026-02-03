#------------------------------------------------------------------------------
# Redis Secrets Module Variables
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Secret Naming
#------------------------------------------------------------------------------

variable "secret_name_prefix" {
  description = "Prefix for the secret name (e.g., 'greenlang' or 'greenlang/production')"
  type        = string

  validation {
    condition     = can(regex("^[a-zA-Z0-9/_-]+$", var.secret_name_prefix))
    error_message = "Secret name prefix must contain only alphanumeric characters, hyphens, underscores, and forward slashes."
  }
}

#------------------------------------------------------------------------------
# Redis Connection Parameters
#------------------------------------------------------------------------------

variable "redis_host" {
  description = "Redis host endpoint"
  type        = string
  default     = "localhost"
}

variable "redis_port" {
  description = "Redis port number"
  type        = number
  default     = 6379

  validation {
    condition     = var.redis_port > 0 && var.redis_port < 65536
    error_message = "Redis port must be between 1 and 65535."
  }
}

variable "redis_username" {
  description = "Redis username (for Redis 6.0+ ACL)"
  type        = string
  default     = "default"
}

variable "redis_ssl_enabled" {
  description = "Whether SSL/TLS is enabled for Redis connection"
  type        = bool
  default     = true
}

#------------------------------------------------------------------------------
# Password Requirements
#------------------------------------------------------------------------------

variable "password_length" {
  description = "Length of the generated password"
  type        = number
  default     = 32

  validation {
    condition     = var.password_length >= 16 && var.password_length <= 128
    error_message = "Password length must be between 16 and 128 characters."
  }
}

variable "password_special_chars" {
  description = "Include special characters in the password"
  type        = bool
  default     = true
}

variable "password_override_special" {
  description = "Override the default special characters (useful for Redis compatibility)"
  type        = string
  default     = "!#$%&*()-_=+[]{}:?"
}

variable "password_min_lower" {
  description = "Minimum number of lowercase characters"
  type        = number
  default     = 4
}

variable "password_min_upper" {
  description = "Minimum number of uppercase characters"
  type        = number
  default     = 4
}

variable "password_min_numeric" {
  description = "Minimum number of numeric characters"
  type        = number
  default     = 4
}

variable "password_min_special" {
  description = "Minimum number of special characters"
  type        = number
  default     = 2
}

#------------------------------------------------------------------------------
# Rotation Configuration
#------------------------------------------------------------------------------

variable "enable_rotation" {
  description = "Enable automatic secret rotation"
  type        = bool
  default     = false
}

variable "rotation_days" {
  description = "Number of days between automatic rotations"
  type        = number
  default     = 30

  validation {
    condition     = var.rotation_days >= 1 && var.rotation_days <= 365
    error_message = "Rotation days must be between 1 and 365."
  }
}

variable "rotation_schedule_expression" {
  description = "Cron expression for rotation schedule (overrides rotation_days if set)"
  type        = string
  default     = null
}

variable "rotation_lambda_arn" {
  description = "ARN of the Lambda function for secret rotation"
  type        = string
  default     = null
}

#------------------------------------------------------------------------------
# KMS Configuration
#------------------------------------------------------------------------------

variable "kms_key_arn" {
  description = "ARN of existing KMS key for encryption (creates new key if null)"
  type        = string
  default     = null
}

variable "kms_deletion_window_days" {
  description = "Number of days before KMS key deletion (if creating new key)"
  type        = number
  default     = 30

  validation {
    condition     = var.kms_deletion_window_days >= 7 && var.kms_deletion_window_days <= 30
    error_message = "KMS deletion window must be between 7 and 30 days."
  }
}

#------------------------------------------------------------------------------
# Secret Recovery
#------------------------------------------------------------------------------

variable "secret_recovery_window_days" {
  description = "Number of days before secret can be permanently deleted"
  type        = number
  default     = 30

  validation {
    condition     = var.secret_recovery_window_days >= 0 && var.secret_recovery_window_days <= 30
    error_message = "Secret recovery window must be between 0 and 30 days."
  }
}

#------------------------------------------------------------------------------
# Alerting Configuration
#------------------------------------------------------------------------------

variable "enable_rotation_alerts" {
  description = "Enable CloudWatch alarms for rotation failures"
  type        = bool
  default     = true
}

variable "alarm_sns_topic_arns" {
  description = "List of SNS topic ARNs for alarm notifications"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# Access Control
#------------------------------------------------------------------------------

variable "allowed_principal_arns" {
  description = "List of AWS principal ARNs allowed to access the secret (for cross-account access)"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# EKS/IRSA Configuration
#------------------------------------------------------------------------------

variable "eks_cluster_name" {
  description = "Name of the EKS cluster for IRSA configuration"
  type        = string
  default     = ""
}

variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS OIDC provider for IRSA"
  type        = string
  default     = ""
}

variable "eks_namespace" {
  description = "Kubernetes namespace for the service account"
  type        = string
  default     = "default"
}

variable "eks_service_account_name" {
  description = "Name of the Kubernetes service account"
  type        = string
  default     = "redis-secrets-sa"
}

#------------------------------------------------------------------------------
# Tags
#------------------------------------------------------------------------------

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
