# variables.tf - S3 Module Variables

variable "bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
  default     = "gl-normalizer"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "gl_normalizer_role_arn" {
  description = "IAM role ARN for GL Normalizer service"
  type        = string
}

# Retention
variable "retention_days" {
  description = "Default retention period in days for audit data"
  type        = number
  default     = 2555 # ~7 years
}

# Object Lock
variable "enable_object_lock" {
  description = "Enable S3 Object Lock for immutability"
  type        = bool
  default     = false
}

variable "object_lock_mode" {
  description = "Object lock retention mode (GOVERNANCE or COMPLIANCE)"
  type        = string
  default     = "GOVERNANCE"
}

variable "object_lock_retention_days" {
  description = "Object lock retention period in days"
  type        = number
  default     = 365
}

# Access Logging
variable "enable_access_logging" {
  description = "Enable S3 access logging"
  type        = bool
  default     = true
}

# Cross-Region Replication
variable "enable_cross_region_replication" {
  description = "Enable cross-region replication for DR"
  type        = bool
  default     = false
}

variable "replication_destination_bucket_arn" {
  description = "Destination bucket ARN for replication"
  type        = string
  default     = ""
}

variable "replication_destination_kms_key_arn" {
  description = "KMS key ARN in destination region"
  type        = string
  default     = ""
}

# Monitoring
variable "bucket_size_alarm_threshold_gb" {
  description = "Alarm threshold for bucket size in GB"
  type        = number
  default     = 1000
}

variable "alarm_actions" {
  description = "List of ARNs for alarm actions (SNS topics)"
  type        = list(string)
  default     = []
}

# Tags
variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
