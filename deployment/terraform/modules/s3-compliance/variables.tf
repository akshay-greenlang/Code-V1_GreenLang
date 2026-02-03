#############################################################################
# GreenLang S3 Compliance Module - Variables
#############################################################################

variable "project_name" {
  description = "Name of the project, used as prefix for all resources"
  type        = string
  default     = "greenlang"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}

#############################################################################
# AWS Config Configuration
#############################################################################

variable "config_snapshot_frequency" {
  description = "Frequency for AWS Config snapshot delivery"
  type        = string
  default     = "TwentyFour_Hours"

  validation {
    condition     = contains(["One_Hour", "Three_Hours", "Six_Hours", "Twelve_Hours", "TwentyFour_Hours"], var.config_snapshot_frequency)
    error_message = "Config snapshot frequency must be one of: One_Hour, Three_Hours, Six_Hours, Twelve_Hours, TwentyFour_Hours."
  }
}

variable "config_snapshot_retention_days" {
  description = "Number of days to retain Config snapshots"
  type        = number
  default     = 365

  validation {
    condition     = var.config_snapshot_retention_days >= 90
    error_message = "Config snapshot retention must be at least 90 days."
  }
}

#############################################################################
# Compliance Rule Configuration
#############################################################################

variable "require_mfa_delete" {
  description = "Require MFA delete for versioning rule compliance"
  type        = bool
  default     = true
}

variable "require_replication" {
  description = "Enable cross-region replication compliance rule"
  type        = bool
  default     = false
}

variable "logging_target_bucket" {
  description = "Target bucket name for logging compliance rule"
  type        = string
  default     = ""
}

variable "required_kms_key_id" {
  description = "Required KMS key ARN for encryption compliance"
  type        = string
  default     = ""
}

variable "allowed_public_buckets" {
  description = "List of bucket names that are allowed to be public (exceptions)"
  type        = list(string)
  default     = []
}

#############################################################################
# Remediation Configuration
#############################################################################

variable "enable_auto_remediation" {
  description = "Enable automatic remediation for non-compliant resources"
  type        = bool
  default     = false
}

variable "remediation_retry_attempts" {
  description = "Number of retry attempts for remediation actions"
  type        = number
  default     = 3

  validation {
    condition     = var.remediation_retry_attempts >= 1 && var.remediation_retry_attempts <= 10
    error_message = "Remediation retry attempts must be between 1 and 10."
  }
}

variable "remediation_retry_seconds" {
  description = "Seconds between remediation retry attempts"
  type        = number
  default     = 60

  validation {
    condition     = var.remediation_retry_seconds >= 30
    error_message = "Remediation retry seconds must be at least 30."
  }
}

#############################################################################
# Notification Configuration
#############################################################################

variable "sns_topic_arn" {
  description = "SNS topic ARN for compliance notifications"
  type        = string
  default     = ""
}

variable "enable_compliance_notifications" {
  description = "Enable CloudWatch Events for compliance change notifications"
  type        = bool
  default     = true
}

#############################################################################
# Account-Level Settings
#############################################################################

variable "enable_account_public_access_block" {
  description = "Enable account-level S3 public access block"
  type        = bool
  default     = true
}
