# Shield and WAF Module Variables - SEC-010

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
variable "project_name" {
  description = "Name of the project (used for resource naming)"
  type        = string
  default     = "greenlang"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# -----------------------------------------------------------------------------
# AWS Shield Advanced Configuration
# -----------------------------------------------------------------------------
variable "shield_enabled" {
  description = "Whether to enable AWS Shield Advanced protection"
  type        = bool
  default     = false
}

variable "protected_resources" {
  description = "List of resource ARNs to protect with Shield Advanced"
  type        = list(string)
  default     = []
}

variable "shield_proactive_engagement" {
  description = "Enable proactive engagement with AWS Shield Response Team"
  type        = bool
  default     = false
}

variable "shield_drt_access_role_arn" {
  description = "IAM role ARN for DRT access (required for proactive engagement)"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# WAF Configuration
# -----------------------------------------------------------------------------
variable "waf_scope" {
  description = "WAF scope: REGIONAL (for ALB/API Gateway) or CLOUDFRONT"
  type        = string
  default     = "REGIONAL"
  validation {
    condition     = contains(["REGIONAL", "CLOUDFRONT"], var.waf_scope)
    error_message = "WAF scope must be REGIONAL or CLOUDFRONT."
  }
}

variable "alb_arns" {
  description = "List of ALB ARNs to associate with the WAF Web ACL"
  type        = list(string)
  default     = []
}

variable "token_domains" {
  description = "Token domains for CAPTCHA validation"
  type        = list(string)
  default     = []
}

variable "sampled_requests_enabled" {
  description = "Enable sampled requests for visibility"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# Rate Limiting Configuration
# -----------------------------------------------------------------------------
variable "rate_limit_threshold" {
  description = "Maximum requests per 5-minute window per IP (PRD default: 2000)"
  type        = number
  default     = 2000
  validation {
    condition     = var.rate_limit_threshold >= 100 && var.rate_limit_threshold <= 2000000000
    error_message = "Rate limit must be between 100 and 2,000,000,000."
  }
}

variable "login_endpoint_protection" {
  description = "Enable additional rate limiting for login endpoints"
  type        = bool
  default     = true
}

variable "login_rate_limit_threshold" {
  description = "Maximum login attempts per 5-minute window per IP"
  type        = number
  default     = 100
}

# -----------------------------------------------------------------------------
# Geographic Blocking Configuration
# -----------------------------------------------------------------------------
variable "blocked_countries" {
  description = "ISO 3166-1 alpha-2 country codes to block"
  type        = list(string)
  default     = []
  validation {
    condition = alltrue([
      for code in var.blocked_countries : length(code) == 2
    ])
    error_message = "Country codes must be 2-letter ISO 3166-1 alpha-2 codes."
  }
}

# -----------------------------------------------------------------------------
# IP Blocking Configuration
# -----------------------------------------------------------------------------
variable "blocked_ips" {
  description = "IPv4 addresses/CIDR ranges to block"
  type        = list(string)
  default     = []
}

variable "blocked_ips_v6" {
  description = "IPv6 addresses/CIDR ranges to block"
  type        = list(string)
  default     = []
}

# -----------------------------------------------------------------------------
# Bot Control Configuration
# -----------------------------------------------------------------------------
variable "bot_control_enabled" {
  description = "Enable AWS Managed Rules Bot Control"
  type        = bool
  default     = false
}

variable "bot_control_inspection_level" {
  description = "Bot control inspection level: COMMON or TARGETED"
  type        = string
  default     = "COMMON"
  validation {
    condition     = contains(["COMMON", "TARGETED"], var.bot_control_inspection_level)
    error_message = "Bot control inspection level must be COMMON or TARGETED."
  }
}

# -----------------------------------------------------------------------------
# Managed Rule Exclusions
# -----------------------------------------------------------------------------
variable "common_ruleset_excluded_rules" {
  description = "Rules to exclude from the Common Rule Set (set to COUNT)"
  type        = list(string)
  default     = []
}

# -----------------------------------------------------------------------------
# Size Constraints
# -----------------------------------------------------------------------------
variable "max_body_size_bytes" {
  description = "Maximum allowed request body size in bytes"
  type        = number
  default     = 8388608 # 8 MB
  validation {
    condition     = var.max_body_size_bytes >= 1024 && var.max_body_size_bytes <= 52428800
    error_message = "Max body size must be between 1KB and 50MB."
  }
}

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
variable "waf_logging_enabled" {
  description = "Enable WAF request logging"
  type        = bool
  default     = true
}

variable "log_destination_arns" {
  description = "List of log destination ARNs (S3, CloudWatch Logs, or Kinesis Firehose)"
  type        = list(string)
  default     = []
}

variable "log_filter_enabled" {
  description = "Enable logging filter to only log blocked/counted requests"
  type        = bool
  default     = true
}

variable "redacted_fields" {
  description = "Fields to redact from logs for privacy"
  type = list(object({
    type = string
    name = string
  }))
  default = [
    { type = "single_header", name = "authorization" },
    { type = "single_header", name = "cookie" }
  ]
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms Configuration
# -----------------------------------------------------------------------------
variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms for WAF metrics"
  type        = bool
  default     = true
}

variable "alarm_actions" {
  description = "SNS topic ARNs for alarm notifications"
  type        = list(string)
  default     = []
}

variable "blocked_requests_threshold" {
  description = "Threshold for blocked requests alarm"
  type        = number
  default     = 1000
}

variable "allowed_requests_minimum" {
  description = "Minimum allowed requests (below triggers alarm for over-blocking)"
  type        = number
  default     = 100
}
