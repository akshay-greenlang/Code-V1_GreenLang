# GreenLang S3 Module - Variables

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

# KMS Configuration
variable "create_kms_key" {
  description = "Create a KMS key for S3 encryption"
  type        = bool
  default     = true
}

variable "kms_key_arn" {
  description = "Existing KMS key ARN (if not creating new)"
  type        = string
  default     = null
}

# Artifacts Bucket Configuration
variable "create_artifacts_bucket" {
  description = "Create artifacts bucket"
  type        = bool
  default     = true
}

variable "artifacts_noncurrent_version_expiration_days" {
  description = "Days to retain noncurrent versions for artifacts"
  type        = number
  default     = 90
}

# Logs Bucket Configuration
variable "create_logs_bucket" {
  description = "Create logs bucket"
  type        = bool
  default     = true
}

variable "logs_retention_days" {
  description = "Days to retain logs before deletion"
  type        = number
  default     = 365
}

variable "enable_elb_logging" {
  description = "Enable ELB access logging to logs bucket"
  type        = bool
  default     = true
}

variable "elb_account_id" {
  description = "AWS account ID for ELB logging (varies by region)"
  type        = string
  default     = "127311923021" # us-east-1
}

# Backups Bucket Configuration
variable "create_backups_bucket" {
  description = "Create backups bucket"
  type        = bool
  default     = true
}

variable "backups_retention_days" {
  description = "Days to retain backups before deletion"
  type        = number
  default     = 2555 # ~7 years
}

variable "backups_noncurrent_version_expiration_days" {
  description = "Days to retain noncurrent versions for backups"
  type        = number
  default     = 365
}

variable "enable_backup_object_lock" {
  description = "Enable S3 Object Lock for backup immutability"
  type        = bool
  default     = false
}

variable "backup_object_lock_mode" {
  description = "Object Lock mode (COMPLIANCE or GOVERNANCE)"
  type        = string
  default     = "GOVERNANCE"
}

variable "backup_object_lock_days" {
  description = "Object Lock retention period in days"
  type        = number
  default     = 30
}

# Data Bucket Configuration
variable "create_data_bucket" {
  description = "Create data bucket"
  type        = bool
  default     = true
}

variable "data_noncurrent_version_expiration_days" {
  description = "Days to retain noncurrent versions for data"
  type        = number
  default     = 90
}

# CORS Configuration
variable "enable_cors" {
  description = "Enable CORS for data bucket"
  type        = bool
  default     = false
}

variable "cors_allowed_headers" {
  description = "CORS allowed headers"
  type        = list(string)
  default     = ["*"]
}

variable "cors_allowed_methods" {
  description = "CORS allowed methods"
  type        = list(string)
  default     = ["GET", "PUT", "POST", "DELETE", "HEAD"]
}

variable "cors_allowed_origins" {
  description = "CORS allowed origins"
  type        = list(string)
  default     = ["*"]
}

variable "cors_expose_headers" {
  description = "CORS expose headers"
  type        = list(string)
  default     = ["ETag"]
}

variable "cors_max_age_seconds" {
  description = "CORS max age in seconds"
  type        = number
  default     = 3600
}

# Static Assets Bucket Configuration
variable "create_static_assets_bucket" {
  description = "Create static assets bucket for frontend"
  type        = bool
  default     = false
}

variable "enable_static_website" {
  description = "Enable static website hosting"
  type        = bool
  default     = false
}

variable "static_assets_cors_origins" {
  description = "CORS origins for static assets"
  type        = list(string)
  default     = ["*"]
}

variable "cloudfront_distribution_arn" {
  description = "CloudFront distribution ARN for OAC policy"
  type        = string
  default     = null
}

# Replication Configuration
variable "enable_replication" {
  description = "Enable cross-region replication for data bucket"
  type        = bool
  default     = false
}

variable "replication_destination_bucket_arn" {
  description = "ARN of the destination bucket for replication"
  type        = string
  default     = null
}

variable "replication_destination_kms_key_arn" {
  description = "KMS key ARN in the destination region"
  type        = string
  default     = null
}

variable "replication_destination_region" {
  description = "AWS region of the replication destination"
  type        = string
  default     = "us-west-2"
}

# Access Logging
variable "enable_access_logging" {
  description = "Enable S3 access logging"
  type        = bool
  default     = true
}

# Tags
variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
