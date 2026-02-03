#------------------------------------------------------------------------------
# AWS EFS Module - Variables
# GreenLang Infrastructure
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# General Configuration
#------------------------------------------------------------------------------

variable "name" {
  description = "Name prefix for all EFS resources"
  type        = string
  default     = "greenlang"
}

variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}

#------------------------------------------------------------------------------
# VPC Configuration
#------------------------------------------------------------------------------

variable "vpc_id" {
  description = "VPC ID where EFS will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for EFS mount targets (one per AZ)"
  type        = list(string)
}

#------------------------------------------------------------------------------
# Security Configuration
#------------------------------------------------------------------------------

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access EFS"
  type        = list(string)
  default     = []
}

variable "allowed_security_group_ids" {
  description = "List of security group IDs allowed to access EFS"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# Performance Configuration
#------------------------------------------------------------------------------

variable "performance_mode" {
  description = "EFS performance mode (generalPurpose or maxIO)"
  type        = string
  default     = "generalPurpose"

  validation {
    condition     = contains(["generalPurpose", "maxIO"], var.performance_mode)
    error_message = "Performance mode must be either 'generalPurpose' or 'maxIO'."
  }
}

variable "throughput_mode" {
  description = "EFS throughput mode (bursting, provisioned, or elastic)"
  type        = string
  default     = "bursting"

  validation {
    condition     = contains(["bursting", "provisioned", "elastic"], var.throughput_mode)
    error_message = "Throughput mode must be 'bursting', 'provisioned', or 'elastic'."
  }
}

variable "provisioned_throughput_in_mibps" {
  description = "Provisioned throughput in MiB/s (only used when throughput_mode is 'provisioned')"
  type        = number
  default     = 256

  validation {
    condition     = var.provisioned_throughput_in_mibps >= 1 && var.provisioned_throughput_in_mibps <= 3414
    error_message = "Provisioned throughput must be between 1 and 3414 MiB/s."
  }
}

#------------------------------------------------------------------------------
# Encryption Configuration
#------------------------------------------------------------------------------

variable "encrypted" {
  description = "Whether to enable encryption at rest"
  type        = bool
  default     = true
}

variable "create_kms_key" {
  description = "Whether to create a new KMS key for EFS encryption"
  type        = bool
  default     = true
}

variable "kms_key_id" {
  description = "KMS key ARN for EFS encryption (used if create_kms_key is false)"
  type        = string
  default     = null
}

variable "kms_key_deletion_window" {
  description = "KMS key deletion window in days"
  type        = number
  default     = 30
}

variable "kms_key_enable_rotation" {
  description = "Whether to enable automatic KMS key rotation"
  type        = bool
  default     = true
}

#------------------------------------------------------------------------------
# Lifecycle Configuration
#------------------------------------------------------------------------------

variable "transition_to_ia" {
  description = "Lifecycle policy transition to Infrequent Access (AFTER_7_DAYS, AFTER_14_DAYS, AFTER_30_DAYS, AFTER_60_DAYS, AFTER_90_DAYS, AFTER_1_DAY)"
  type        = string
  default     = "AFTER_30_DAYS"

  validation {
    condition = var.transition_to_ia == null || contains([
      "AFTER_1_DAY",
      "AFTER_7_DAYS",
      "AFTER_14_DAYS",
      "AFTER_30_DAYS",
      "AFTER_60_DAYS",
      "AFTER_90_DAYS"
    ], var.transition_to_ia)
    error_message = "Invalid transition_to_ia value."
  }
}

variable "transition_to_primary_storage_class" {
  description = "Lifecycle policy transition back to primary storage (AFTER_1_ACCESS)"
  type        = string
  default     = "AFTER_1_ACCESS"

  validation {
    condition     = var.transition_to_primary_storage_class == null || var.transition_to_primary_storage_class == "AFTER_1_ACCESS"
    error_message = "transition_to_primary_storage_class must be null or 'AFTER_1_ACCESS'."
  }
}

variable "transition_to_archive" {
  description = "Lifecycle policy transition to Archive storage (AFTER_1_DAY, AFTER_7_DAYS, AFTER_14_DAYS, AFTER_30_DAYS, AFTER_60_DAYS, AFTER_90_DAYS, AFTER_180_DAYS, AFTER_270_DAYS, AFTER_365_DAYS)"
  type        = string
  default     = null
}

#------------------------------------------------------------------------------
# Access Points Configuration
#------------------------------------------------------------------------------

variable "access_points" {
  description = "Configuration for default access points"
  type = object({
    artifacts = object({
      path           = string
      uid            = number
      gid            = number
      secondary_gids = optional(list(number), [])
      permissions    = optional(string, "0755")
    })
    models = object({
      path           = string
      uid            = number
      gid            = number
      secondary_gids = optional(list(number), [])
      permissions    = optional(string, "0755")
    })
    shared = object({
      path           = string
      uid            = number
      gid            = number
      secondary_gids = optional(list(number), [])
      permissions    = optional(string, "0755")
    })
    tmp = object({
      path           = string
      uid            = number
      gid            = number
      secondary_gids = optional(list(number), [])
      permissions    = optional(string, "1777")
    })
  })

  default = {
    artifacts = {
      path           = "/artifacts"
      uid            = 1000
      gid            = 1000
      secondary_gids = []
      permissions    = "0755"
    }
    models = {
      path           = "/models"
      uid            = 1000
      gid            = 1000
      secondary_gids = []
      permissions    = "0755"
    }
    shared = {
      path           = "/shared"
      uid            = 1000
      gid            = 1000
      secondary_gids = []
      permissions    = "0755"
    }
    tmp = {
      path           = "/tmp"
      uid            = 65534
      gid            = 65534
      secondary_gids = []
      permissions    = "1777"
    }
  }
}

variable "additional_access_points" {
  description = "Map of additional access points to create"
  type = map(object({
    path           = string
    uid            = number
    gid            = number
    secondary_gids = optional(list(number), [])
    permissions    = optional(string, "0755")
  }))
  default = {}
}

#------------------------------------------------------------------------------
# Backup Configuration
#------------------------------------------------------------------------------

variable "enable_backup" {
  description = "Whether to enable AWS Backup for EFS"
  type        = bool
  default     = true
}

variable "create_backup_vault" {
  description = "Whether to create a new backup vault"
  type        = bool
  default     = true
}

variable "backup_vault_name" {
  description = "Name of existing backup vault (used if create_backup_vault is false)"
  type        = string
  default     = null
}

variable "create_backup_plan" {
  description = "Whether to create a backup plan"
  type        = bool
  default     = true
}

variable "backup_schedule" {
  description = "Cron expression for daily backup schedule"
  type        = string
  default     = "cron(0 5 ? * * *)" # Daily at 5 AM UTC
}

variable "backup_delete_after" {
  description = "Number of days after which daily backups are deleted"
  type        = number
  default     = 35
}

variable "backup_cold_storage_after" {
  description = "Number of days after which daily backups are moved to cold storage"
  type        = number
  default     = 30
}

variable "backup_weekly_schedule" {
  description = "Cron expression for weekly backup schedule"
  type        = string
  default     = "cron(0 5 ? * SUN *)" # Weekly on Sunday at 5 AM UTC
}

variable "backup_weekly_delete_after" {
  description = "Number of days after which weekly backups are deleted"
  type        = number
  default     = 365
}

variable "backup_weekly_cold_storage_after" {
  description = "Number of days after which weekly backups are moved to cold storage"
  type        = number
  default     = 90
}

variable "backup_copy_destination_vault_arn" {
  description = "ARN of destination vault for backup copies (cross-region DR)"
  type        = string
  default     = null
}

variable "backup_copy_delete_after" {
  description = "Number of days after which backup copies are deleted"
  type        = number
  default     = 35
}

variable "backup_copy_cold_storage_after" {
  description = "Number of days after which backup copies are moved to cold storage"
  type        = number
  default     = 30
}

#------------------------------------------------------------------------------
# Replication Configuration
#------------------------------------------------------------------------------

variable "enable_replication" {
  description = "Whether to enable EFS replication for DR"
  type        = bool
  default     = false
}

variable "replication_region" {
  description = "AWS region for EFS replication destination"
  type        = string
  default     = null
}

variable "replication_availability_zone" {
  description = "Availability zone for EFS replication destination (optional, for One Zone storage)"
  type        = string
  default     = null
}

variable "replication_kms_key_id" {
  description = "KMS key ARN for replication destination encryption"
  type        = string
  default     = null
}

#------------------------------------------------------------------------------
# File System Policy Configuration
#------------------------------------------------------------------------------

variable "enable_file_system_policy" {
  description = "Whether to enable EFS file system policy"
  type        = bool
  default     = true
}

variable "bypass_policy_lockout_safety_check" {
  description = "Whether to bypass the policy lockout safety check"
  type        = bool
  default     = false
}

variable "efs_access_principal_arns" {
  description = "List of IAM principal ARNs allowed to access EFS"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# CloudWatch Alarms Configuration
#------------------------------------------------------------------------------

variable "enable_cloudwatch_alarms" {
  description = "Whether to create CloudWatch alarms"
  type        = bool
  default     = true
}

variable "alarm_actions" {
  description = "List of ARNs to notify when alarm triggers"
  type        = list(string)
  default     = []
}

variable "alarm_evaluation_periods" {
  description = "Number of evaluation periods for alarms"
  type        = number
  default     = 3
}

variable "alarm_period" {
  description = "Period in seconds for alarm evaluation"
  type        = number
  default     = 300
}

variable "burst_credit_balance_threshold" {
  description = "Threshold for burst credit balance alarm (bytes)"
  type        = number
  default     = 1000000000000 # 1 TB
}

variable "percent_io_limit_threshold" {
  description = "Threshold for percent IO limit alarm (percentage)"
  type        = number
  default     = 90
}

#------------------------------------------------------------------------------
# EKS Integration
#------------------------------------------------------------------------------

variable "eks_cluster_name" {
  description = "Name of the EKS cluster for IRSA"
  type        = string
  default     = null
}

variable "eks_cluster_oidc_issuer_url" {
  description = "OIDC issuer URL for the EKS cluster"
  type        = string
  default     = null
}

variable "eks_namespace" {
  description = "Kubernetes namespace for EFS access"
  type        = string
  default     = "greenlang"
}

variable "eks_service_account_name" {
  description = "Kubernetes service account name for EFS access"
  type        = string
  default     = "efs-csi-controller-sa"
}

#------------------------------------------------------------------------------
# Cross-Account Access
#------------------------------------------------------------------------------

variable "enable_cross_account_access" {
  description = "Whether to enable cross-account access"
  type        = bool
  default     = false
}

variable "cross_account_ids" {
  description = "List of AWS account IDs for cross-account access"
  type        = list(string)
  default     = []
}
