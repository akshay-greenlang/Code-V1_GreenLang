#############################################################################
# GreenLang S3 Security Module - Variables
#############################################################################

#############################################################################
# General Configuration
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
# VPC Configuration
#############################################################################

variable "vpc_id" {
  description = "VPC ID for VPC endpoint and access point configuration"
  type        = string
  default     = ""
}

variable "private_route_table_ids" {
  description = "List of private route table IDs for S3 VPC endpoint"
  type        = list(string)
  default     = []
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for resources requiring VPC placement"
  type        = list(string)
  default     = []
}

#############################################################################
# KMS Configuration
#############################################################################

variable "kms_key_deletion_window" {
  description = "Number of days before KMS key is permanently deleted (7-30)"
  type        = number
  default     = 30

  validation {
    condition     = var.kms_key_deletion_window >= 7 && var.kms_key_deletion_window <= 30
    error_message = "KMS key deletion window must be between 7 and 30 days."
  }
}

variable "kms_multi_region" {
  description = "Enable multi-region KMS key for cross-region replication"
  type        = bool
  default     = false
}

variable "kms_key_administrators" {
  description = "List of IAM ARNs that can administer the KMS key"
  type        = list(string)
  default     = []
}

#############################################################################
# MFA Configuration
#############################################################################

variable "enable_mfa_delete" {
  description = "Enable MFA delete for critical buckets (requires manual root account setup)"
  type        = bool
  default     = true
}

variable "mfa_device_arn" {
  description = "ARN of the MFA device for MFA delete operations (for documentation only)"
  type        = string
  default     = ""
}

#############################################################################
# Object Lock Configuration
#############################################################################

variable "backup_retention_days" {
  description = "Number of days to retain backups in GOVERNANCE mode"
  type        = number
  default     = 90

  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "Backup retention must be between 1 and 365 days."
  }
}

variable "audit_log_retention_years" {
  description = "Number of years to retain audit logs in COMPLIANCE mode"
  type        = number
  default     = 7

  validation {
    condition     = var.audit_log_retention_years >= 1 && var.audit_log_retention_years <= 100
    error_message = "Audit log retention must be between 1 and 100 years."
  }
}

variable "enable_legal_hold_by_default" {
  description = "Enable legal hold on all objects in audit logs bucket"
  type        = bool
  default     = false
}

#############################################################################
# Access Logging Configuration
#############################################################################

variable "access_log_retention_days" {
  description = "Number of days to retain access logs before deletion"
  type        = number
  default     = 365

  validation {
    condition     = var.access_log_retention_days >= 90 && var.access_log_retention_days <= 3650
    error_message = "Access log retention must be between 90 and 3650 days."
  }
}

variable "enable_access_logging" {
  description = "Enable access logging for all buckets"
  type        = bool
  default     = true
}

#############################################################################
# IP Allowlists
#############################################################################

variable "admin_ip_allowlist" {
  description = "List of IP addresses/CIDR blocks allowed for administrative access"
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for ip in var.admin_ip_allowlist : can(cidrhost(ip, 0)) || can(regex("^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$", ip))
    ])
    error_message = "All entries must be valid IP addresses or CIDR blocks."
  }
}

variable "readonly_ip_allowlist" {
  description = "List of IP addresses/CIDR blocks allowed for read-only access"
  type        = list(string)
  default     = []
}

variable "vpn_cidr_blocks" {
  description = "CIDR blocks for VPN access"
  type        = list(string)
  default     = []
}

#############################################################################
# IAM Principals
#############################################################################

variable "s3_admin_principals" {
  description = "List of IAM principal ARNs with S3 admin access"
  type        = list(string)
  default     = []
}

variable "admin_role_arns" {
  description = "List of IAM role ARNs with administrative access to S3"
  type        = list(string)
  default     = []
}

variable "readonly_role_arns" {
  description = "List of IAM role ARNs with read-only access to S3"
  type        = list(string)
  default     = []
}

variable "write_role_arns" {
  description = "List of IAM role ARNs with write access to S3"
  type        = list(string)
  default     = []
}

variable "audit_reader_arns" {
  description = "List of IAM role ARNs with access to read audit logs"
  type        = list(string)
  default     = []
}

#############################################################################
# Cross-Account Configuration
#############################################################################

variable "cross_account_principals" {
  description = "List of cross-account principal ARNs allowed access"
  type        = list(string)
  default     = []
}

variable "cross_account_audit_principals" {
  description = "List of cross-account principal ARNs allowed to read audit logs"
  type        = list(string)
  default     = []
}

variable "cross_account_ids" {
  description = "List of AWS account IDs allowed cross-account access"
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for id in var.cross_account_ids : can(regex("^\\d{12}$", id))
    ])
    error_message = "All account IDs must be valid 12-digit AWS account numbers."
  }
}

#############################################################################
# Bucket-Specific Configuration
#############################################################################

variable "additional_buckets" {
  description = "Map of additional buckets to create with their configurations"
  type = map(object({
    purpose             = string
    enable_versioning   = bool
    enable_object_lock  = bool
    object_lock_mode    = optional(string, "GOVERNANCE")
    retention_days      = optional(number, 30)
    enable_logging      = optional(bool, true)
    lifecycle_rules     = optional(list(object({
      id                  = string
      prefix              = optional(string, "")
      transition_days     = optional(number, 30)
      transition_class    = optional(string, "STANDARD_IA")
      expiration_days     = optional(number, null)
    })), [])
    allowed_principals  = optional(list(string), [])
    tags                = optional(map(string), {})
  }))
  default = {}
}

variable "enable_intelligent_tiering" {
  description = "Enable S3 Intelligent-Tiering for app-data bucket"
  type        = bool
  default     = false
}

#############################################################################
# Replication Configuration
#############################################################################

variable "enable_cross_region_replication" {
  description = "Enable cross-region replication for critical buckets"
  type        = bool
  default     = false
}

variable "replication_destination_bucket_arn" {
  description = "ARN of the destination bucket for replication"
  type        = string
  default     = ""
}

variable "replication_destination_region" {
  description = "AWS region for replication destination"
  type        = string
  default     = "us-west-2"
}

#############################################################################
# Monitoring and Alerting
#############################################################################

variable "alarm_sns_topic_arns" {
  description = "List of SNS topic ARNs for CloudWatch alarms"
  type        = list(string)
  default     = []
}

variable "enable_cloudwatch_metrics" {
  description = "Enable CloudWatch request metrics for S3 buckets"
  type        = bool
  default     = true
}

variable "metrics_filter_prefix" {
  description = "Prefix filter for CloudWatch metrics"
  type        = string
  default     = ""
}

#############################################################################
# Compliance Configuration
#############################################################################

variable "compliance_standards" {
  description = "List of compliance standards to enforce (SOC2, ISO27001, HIPAA, PCI-DSS)"
  type        = list(string)
  default     = ["SOC2", "ISO27001"]

  validation {
    condition = alltrue([
      for standard in var.compliance_standards : contains(["SOC2", "ISO27001", "HIPAA", "PCI-DSS", "GDPR"], standard)
    ])
    error_message = "Compliance standards must be one of: SOC2, ISO27001, HIPAA, PCI-DSS, GDPR."
  }
}

variable "data_classification" {
  description = "Data classification level for the buckets"
  type        = string
  default     = "confidential"

  validation {
    condition     = contains(["public", "internal", "confidential", "restricted"], var.data_classification)
    error_message = "Data classification must be one of: public, internal, confidential, restricted."
  }
}
