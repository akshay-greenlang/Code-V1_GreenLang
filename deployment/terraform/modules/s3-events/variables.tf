#------------------------------------------------------------------------------
# S3 Event Notifications Module - Variables
# GreenLang Infrastructure
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# General Configuration
#------------------------------------------------------------------------------

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "GreenLang"
    ManagedBy = "Terraform"
    Component = "S3-Events"
  }
}

#------------------------------------------------------------------------------
# Bucket Configuration
#------------------------------------------------------------------------------

variable "source_bucket_arns" {
  description = "List of S3 bucket ARNs that can publish to SNS topics"
  type        = list(string)

  validation {
    condition     = length(var.source_bucket_arns) > 0
    error_message = "At least one source bucket ARN must be provided."
  }
}

variable "notification_buckets" {
  description = "Map of buckets to configure with notifications"
  type = map(object({
    bucket_id                         = string
    bucket_arn                        = string
    enable_upload_notifications       = bool
    enable_delete_notifications       = bool
    enable_lifecycle_notifications    = bool
    enable_replication_notifications  = bool
    enable_artifact_validation        = bool
    enable_report_indexing            = bool
    upload_prefix                     = optional(string, "")
    upload_suffix                     = optional(string, "")
  }))
  default = {}
}

variable "monitored_bucket_names" {
  description = "List of bucket names to monitor via EventBridge"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# SNS Configuration
#------------------------------------------------------------------------------

variable "sns_delivery_policy" {
  description = "SNS delivery policy JSON"
  type        = string
  default     = <<EOF
{
  "http": {
    "defaultHealthyRetryPolicy": {
      "minDelayTarget": 20,
      "maxDelayTarget": 20,
      "numRetries": 3,
      "numMaxDelayRetries": 0,
      "numNoDelayRetries": 0,
      "numMinDelayRetries": 0,
      "backoffFunction": "linear"
    },
    "disableSubscriptionOverrides": false
  }
}
EOF
}

variable "enable_sns_encryption" {
  description = "Enable server-side encryption for SNS topics"
  type        = bool
  default     = true
}

#------------------------------------------------------------------------------
# SQS Configuration
#------------------------------------------------------------------------------

variable "sqs_message_retention_seconds" {
  description = "Number of seconds SQS retains a message"
  type        = number
  default     = 345600  # 4 days

  validation {
    condition     = var.sqs_message_retention_seconds >= 60 && var.sqs_message_retention_seconds <= 1209600
    error_message = "Message retention must be between 60 and 1209600 seconds."
  }
}

variable "sqs_visibility_timeout_seconds" {
  description = "Visibility timeout for SQS messages"
  type        = number
  default     = 300

  validation {
    condition     = var.sqs_visibility_timeout_seconds >= 0 && var.sqs_visibility_timeout_seconds <= 43200
    error_message = "Visibility timeout must be between 0 and 43200 seconds."
  }
}

variable "sqs_max_receive_count" {
  description = "Number of times a message can be received before moving to DLQ"
  type        = number
  default     = 3

  validation {
    condition     = var.sqs_max_receive_count >= 1 && var.sqs_max_receive_count <= 1000
    error_message = "Max receive count must be between 1 and 1000."
  }
}

variable "sqs_dlq_retention_seconds" {
  description = "Number of seconds DLQ retains a message"
  type        = number
  default     = 1209600  # 14 days
}

#------------------------------------------------------------------------------
# KMS Configuration
#------------------------------------------------------------------------------

variable "kms_key_id" {
  description = "KMS key ID for SNS/SQS encryption"
  type        = string
  default     = "alias/aws/sqs"
}

variable "kms_key_arn" {
  description = "KMS key ARN for CloudWatch Logs encryption"
  type        = string
  default     = null
}

#------------------------------------------------------------------------------
# Lambda Configuration
#------------------------------------------------------------------------------

variable "lambda_runtime" {
  description = "Lambda runtime version"
  type        = string
  default     = "python3.11"
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 300

  validation {
    condition     = var.lambda_timeout >= 1 && var.lambda_timeout <= 900
    error_message = "Lambda timeout must be between 1 and 900 seconds."
  }
}

variable "lambda_memory_size" {
  description = "Lambda function memory size in MB"
  type        = number
  default     = 512

  validation {
    condition     = var.lambda_memory_size >= 128 && var.lambda_memory_size <= 10240
    error_message = "Lambda memory must be between 128 and 10240 MB."
  }
}

variable "lambda_reserved_concurrency" {
  description = "Reserved concurrent executions for Lambda"
  type        = number
  default     = 10
}

variable "lambda_log_retention_days" {
  description = "CloudWatch log retention for Lambda functions"
  type        = number
  default     = 30

  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.lambda_log_retention_days)
    error_message = "Log retention must be a valid CloudWatch Logs retention value."
  }
}

variable "lambda_source_path" {
  description = "Path to Lambda function source code"
  type        = string
  default     = "../../../lambda/s3-events"
}

variable "artifact_validator_config" {
  description = "Configuration for artifact validator Lambda"
  type = object({
    enabled              = bool
    max_file_size_mb     = number
    allowed_extensions   = list(string)
    enable_virus_scan    = bool
    quarantine_bucket    = string
    valid_artifacts_tag  = string
  })
  default = {
    enabled              = true
    max_file_size_mb     = 100
    allowed_extensions   = [".zip", ".tar.gz", ".jar", ".whl", ".rpm", ".deb"]
    enable_virus_scan    = true
    quarantine_bucket    = ""
    valid_artifacts_tag  = "validated"
  }
}

variable "report_indexer_config" {
  description = "Configuration for report indexer Lambda"
  type = object({
    enabled                = bool
    opensearch_endpoint    = string
    opensearch_index       = string
    supported_formats      = list(string)
    extract_text           = bool
  })
  default = {
    enabled                = true
    opensearch_endpoint    = ""
    opensearch_index       = "greenlang-reports"
    supported_formats      = ["pdf", "html", "xlsx", "csv", "json"]
    extract_text           = true
  }
}

variable "audit_logger_config" {
  description = "Configuration for audit logger Lambda"
  type = object({
    enabled              = bool
    audit_bucket         = string
    audit_prefix         = string
    enable_daily_summary = bool
    compliance_mode      = string
  })
  default = {
    enabled              = true
    audit_bucket         = ""
    audit_prefix         = "audit-logs/"
    enable_daily_summary = true
    compliance_mode      = "standard"
  }
}

variable "cost_tracker_config" {
  description = "Configuration for cost tracker Lambda"
  type = object({
    enabled              = bool
    cost_allocation_tag  = string
    alert_threshold_usd  = number
    report_schedule      = string
  })
  default = {
    enabled              = true
    cost_allocation_tag  = "greenlang-storage"
    alert_threshold_usd  = 1000
    report_schedule      = "rate(1 day)"
  }
}

#------------------------------------------------------------------------------
# Event Filter Patterns
#------------------------------------------------------------------------------

variable "upload_event_filters" {
  description = "Event filter patterns for upload notifications"
  type = list(object({
    prefix = string
    suffix = string
  }))
  default = [
    { prefix = "artifacts/", suffix = "" },
    { prefix = "reports/", suffix = "" },
    { prefix = "uploads/", suffix = "" }
  ]
}

variable "delete_event_filters" {
  description = "Event filter patterns for delete notifications"
  type = list(object({
    prefix = string
    suffix = string
  }))
  default = []
}

variable "lifecycle_event_filters" {
  description = "Event filter patterns for lifecycle notifications"
  type = list(object({
    prefix = string
    suffix = string
  }))
  default = []
}

#------------------------------------------------------------------------------
# Alarm Configuration
#------------------------------------------------------------------------------

variable "alarm_sns_topic_arns" {
  description = "SNS topic ARNs for alarm notifications"
  type        = list(string)
  default     = []
}

variable "dlq_alarm_threshold" {
  description = "Number of DLQ messages to trigger alarm"
  type        = number
  default     = 10
}

variable "message_age_alarm_threshold" {
  description = "Maximum age of messages before alarm (seconds)"
  type        = number
  default     = 3600
}

#------------------------------------------------------------------------------
# CloudWatch Configuration
#------------------------------------------------------------------------------

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 90

  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch Logs retention value."
  }
}

variable "enable_detailed_metrics" {
  description = "Enable detailed CloudWatch metrics"
  type        = bool
  default     = true
}

#------------------------------------------------------------------------------
# VPC Configuration (for Lambda)
#------------------------------------------------------------------------------

variable "vpc_config" {
  description = "VPC configuration for Lambda functions"
  type = object({
    enabled            = bool
    subnet_ids         = list(string)
    security_group_ids = list(string)
  })
  default = {
    enabled            = false
    subnet_ids         = []
    security_group_ids = []
  }
}

#------------------------------------------------------------------------------
# External Service Integration
#------------------------------------------------------------------------------

variable "opensearch_config" {
  description = "OpenSearch configuration for report indexing"
  type = object({
    enabled       = bool
    endpoint      = string
    index_prefix  = string
    region        = string
  })
  default = {
    enabled       = false
    endpoint      = ""
    index_prefix  = "greenlang"
    region        = "us-east-1"
  }
}

variable "clamav_config" {
  description = "ClamAV configuration for virus scanning"
  type = object({
    enabled          = bool
    definition_bucket = string
    scan_timeout      = number
  })
  default = {
    enabled          = false
    definition_bucket = ""
    scan_timeout      = 60
  }
}
