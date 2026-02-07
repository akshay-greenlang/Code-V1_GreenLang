# =============================================================================
# GreenLang Alerting Integrations Module - Variables
# GreenLang Climate OS | OBS-004
# =============================================================================
# Input variables for configuring PagerDuty, Opsgenie, Slack, and Email
# notification integrations for the Unified Alerting & Notification Platform.
# =============================================================================

# -----------------------------------------------------------------------------
# Required Variables
# -----------------------------------------------------------------------------

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# -----------------------------------------------------------------------------
# Project Configuration
# -----------------------------------------------------------------------------

variable "project" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "gl"
}

variable "aws_region" {
  description = "AWS region for SSM and SES resources"
  type        = string
  default     = ""
}

variable "kms_key_alias" {
  description = "KMS key alias for encrypting SSM SecureString parameters. If empty, uses alias/{project}-secrets-{environment}"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# PagerDuty Configuration
# -----------------------------------------------------------------------------

variable "enable_pagerduty" {
  description = "Enable PagerDuty integration"
  type        = bool
  default     = true
}

variable "pagerduty_api_key" {
  description = "PagerDuty API token (v2) for service/escalation management"
  type        = string
  sensitive   = true
  default     = ""
}

variable "pagerduty_service_name" {
  description = "PagerDuty service name for GreenLang alerts"
  type        = string
  default     = "GreenLang Platform"
}

variable "pagerduty_service_description" {
  description = "PagerDuty service description"
  type        = string
  default     = "GreenLang Climate OS production alerts"
}

variable "pagerduty_escalation_timeout" {
  description = "Minutes before escalating to next level in PagerDuty"
  type        = number
  default     = 15

  validation {
    condition     = var.pagerduty_escalation_timeout >= 5 && var.pagerduty_escalation_timeout <= 60
    error_message = "PagerDuty escalation timeout must be between 5 and 60 minutes."
  }
}

variable "pagerduty_user_ids" {
  description = "List of PagerDuty user IDs for escalation policy levels (ordered by escalation priority)"
  type        = list(string)
  default     = []
}

variable "pagerduty_schedule_ids" {
  description = "List of PagerDuty schedule IDs for on-call rotation"
  type        = list(string)
  default     = []
}

# -----------------------------------------------------------------------------
# Opsgenie Configuration
# -----------------------------------------------------------------------------

variable "enable_opsgenie" {
  description = "Enable Opsgenie integration"
  type        = bool
  default     = true
}

variable "opsgenie_api_key" {
  description = "Opsgenie API key for alert management"
  type        = string
  sensitive   = true
  default     = ""
}

variable "opsgenie_team_name" {
  description = "Opsgenie team name for GreenLang"
  type        = string
  default     = "GreenLang Platform"
}

variable "opsgenie_team_description" {
  description = "Opsgenie team description"
  type        = string
  default     = "GreenLang Climate OS Platform Team"
}

variable "opsgenie_responders" {
  description = "List of Opsgenie responder usernames for escalation"
  type        = list(string)
  default     = []
}

variable "opsgenie_schedule_timezone" {
  description = "Timezone for Opsgenie on-call schedule"
  type        = string
  default     = "America/New_York"
}

# -----------------------------------------------------------------------------
# Slack Configuration
# -----------------------------------------------------------------------------

variable "slack_webhook_critical" {
  description = "Slack webhook URL for critical alerts"
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_webhook_warning" {
  description = "Slack webhook URL for warning alerts"
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_webhook_info" {
  description = "Slack webhook URL for informational alerts"
  type        = string
  sensitive   = true
  default     = ""
}

# -----------------------------------------------------------------------------
# Email / SES Configuration
# -----------------------------------------------------------------------------

variable "ses_from_address" {
  description = "SES verified sender email address for alert notifications"
  type        = string
  default     = "alerts@greenlang.io"
}

variable "ses_reply_to" {
  description = "Reply-to email address for alert notifications"
  type        = string
  default     = "platform-team@greenlang.io"
}

variable "ses_critical_recipients" {
  description = "Email addresses for critical alert notifications"
  type        = list(string)
  default     = []
}

variable "ses_warning_recipients" {
  description = "Email addresses for warning alert notifications"
  type        = list(string)
  default     = []
}

# -----------------------------------------------------------------------------
# Tags
# -----------------------------------------------------------------------------

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "GreenLang"
    Component = "alerting"
    ManagedBy = "terraform"
    OBS       = "004"
  }
}
