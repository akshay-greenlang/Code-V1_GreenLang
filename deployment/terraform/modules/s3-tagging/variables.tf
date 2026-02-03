# Variables for S3 Object Tagging Module

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = "greenlang"
}

variable "monitored_bucket_names" {
  description = "List of S3 bucket names to monitor for auto-tagging"
  type        = list(string)
}

variable "batch_manifest_bucket_arn" {
  description = "ARN of the bucket for S3 Batch Operations manifests"
  type        = string
  default     = ""
}

variable "batch_report_bucket_arn" {
  description = "ARN of the bucket for S3 Batch Operations reports"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Tag Schemas per Artifact Type
# -----------------------------------------------------------------------------

variable "tag_schemas" {
  description = "Tag schemas per artifact type"
  type        = map(map(any))
  default = {
    BUILD_ARTIFACTS = {
      data_classification = "internal"
      cost_center         = "engineering"
      retention_policy    = "short-term"
      environment = {
        pattern = "builds/(dev|staging|prod)/"
        default = "unknown"
      }
      build_id = {
        pattern = "builds/[^/]+/([^/]+)/"
        default = "unknown"
      }
    }

    CALCULATION_RESULTS = {
      data_classification = "confidential"
      cost_center         = "sustainability"
      retention_policy    = "compliance"
      compliance_required = "true"
      gdpr_relevant       = "true"
      calculation_type = {
        pattern = "calculations/([^/]+)/"
        default = "general"
      }
    }

    REPORTS = {
      data_classification = "confidential"
      cost_center         = "sustainability"
      retention_policy    = "long-term"
      compliance_required = "true"
      csrd_relevant       = "true"
      report_type = {
        pattern = "reports/([^/]+)/"
        default = "general"
      }
      report_year = {
        pattern = "reports/[^/]+/(\\d{4})/"
        default = "unknown"
      }
    }

    AUDIT_LOGS = {
      data_classification = "restricted"
      cost_center         = "compliance"
      retention_policy    = "sox-compliance"
      compliance_required = "true"
      sox_relevant        = "true"
      immutable           = "true"
      audit_source = {
        pattern = "audit/([^/]+)/"
        default = "system"
      }
    }

    ML_MODELS = {
      data_classification = "internal"
      cost_center         = "data-science"
      retention_policy    = "medium-term"
      model_type = {
        pattern = "models/([^/]+)/"
        default = "unknown"
      }
      model_version = {
        pattern = "models/[^/]+/v(\\d+)/"
        default = "unknown"
      }
    }

    TEMPORARY = {
      data_classification = "internal"
      cost_center         = "operations"
      retention_policy    = "ephemeral"
      auto_delete         = "true"
    }

    UNKNOWN = {
      data_classification = "unclassified"
      cost_center         = "unknown"
      retention_policy    = "review-required"
      needs_classification = "true"
    }
  }
}

# -----------------------------------------------------------------------------
# Required Tags
# -----------------------------------------------------------------------------

variable "required_tags" {
  description = "List of tag keys that must be present on all objects"
  type        = list(string)
  default = [
    "artifact_type",
    "data_classification",
    "cost_center"
  ]
}

# -----------------------------------------------------------------------------
# Default Tags
# -----------------------------------------------------------------------------

variable "default_tags" {
  description = "Default tags to apply to all objects"
  type        = map(string)
  default = {
    managed_by  = "greenlang-auto-tagger"
    project     = "greenlang"
    environment = "production"
  }
}

# -----------------------------------------------------------------------------
# Lambda Configuration
# -----------------------------------------------------------------------------

variable "lambda_reserved_concurrency" {
  description = "Reserved concurrent executions for Lambda"
  type        = number
  default     = 10
}

variable "log_level" {
  description = "Log level for Lambda function"
  type        = string
  default     = "INFO"
  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR"], var.log_level)
    error_message = "Log level must be one of: DEBUG, INFO, WARNING, ERROR."
  }
}

variable "enable_xray_tracing" {
  description = "Enable X-Ray tracing for Lambda"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# Tag Enforcement Configuration
# -----------------------------------------------------------------------------

variable "enable_tag_enforcement" {
  description = "Enable scheduled tag enforcement via S3 Batch Operations"
  type        = bool
  default     = true
}

variable "enforcement_schedule" {
  description = "Schedule expression for tag enforcement (cron or rate)"
  type        = string
  default     = "cron(0 2 ? * SUN *)"  # Weekly on Sunday at 2 AM UTC
}

# -----------------------------------------------------------------------------
# Alerting Configuration
# -----------------------------------------------------------------------------

variable "enable_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "alarm_actions" {
  description = "List of ARNs to notify when alarms trigger"
  type        = list(string)
  default     = []
}

# -----------------------------------------------------------------------------
# Common Tags
# -----------------------------------------------------------------------------

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "GreenLang"
    Component   = "S3-Tagging"
    ManagedBy   = "Terraform"
    Environment = "production"
  }
}

# -----------------------------------------------------------------------------
# Data Classification Definitions
# -----------------------------------------------------------------------------

variable "data_classification_levels" {
  description = "Data classification levels and their descriptions"
  type        = map(object({
    level       = number
    description = string
    handling    = string
  }))
  default = {
    public = {
      level       = 1
      description = "Information that can be publicly shared"
      handling    = "No restrictions"
    }
    internal = {
      level       = 2
      description = "Internal business information"
      handling    = "Limit to employees and authorized contractors"
    }
    confidential = {
      level       = 3
      description = "Sensitive business or personal information"
      handling    = "Encrypt in transit and at rest, limit access"
    }
    restricted = {
      level       = 4
      description = "Highly sensitive information requiring strict controls"
      handling    = "Encrypt, audit all access, immutable storage"
    }
    unclassified = {
      level       = 0
      description = "Information pending classification"
      handling    = "Review and classify within 7 days"
    }
  }
}

# -----------------------------------------------------------------------------
# Retention Policy Definitions
# -----------------------------------------------------------------------------

variable "retention_policy_definitions" {
  description = "Retention policy definitions"
  type        = map(object({
    active_days  = number
    archive_days = number
    delete_days  = number
    description  = string
  }))
  default = {
    ephemeral = {
      active_days  = 7
      archive_days = 0
      delete_days  = 7
      description  = "Temporary data, delete after 7 days"
    }
    short-term = {
      active_days  = 30
      archive_days = 90
      delete_days  = 365
      description  = "Short-term retention, archive after 30 days"
    }
    medium-term = {
      active_days  = 90
      archive_days = 365
      delete_days  = 730
      description  = "Medium-term retention, 2 year total"
    }
    long-term = {
      active_days  = 365
      archive_days = 2555
      delete_days  = 0  # Never delete
      description  = "Long-term retention, 7+ years archive"
    }
    compliance = {
      active_days  = 90
      archive_days = 2555
      delete_days  = 0  # Never delete
      description  = "Compliance retention, 7 years minimum"
    }
    sox-compliance = {
      active_days  = 2555  # 7 years active (immutable)
      archive_days = 0
      delete_days  = 0  # Never delete
      description  = "SOX compliance, 7 years immutable"
    }
    review-required = {
      active_days  = 30
      archive_days = 0
      delete_days  = 0
      description  = "Pending classification review"
    }
  }
}
