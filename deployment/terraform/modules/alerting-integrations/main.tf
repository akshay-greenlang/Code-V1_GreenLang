# =============================================================================
# GreenLang Alerting Integrations Module - Main
# GreenLang Climate OS | OBS-004
# =============================================================================
# Main module file containing:
#   - Local variables
#   - Data sources for SSM parameter references
#   - PagerDuty + Opsgenie integration orchestration
# =============================================================================

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------

data "aws_region" "current" {}

data "aws_caller_identity" "current" {}

data "aws_kms_key" "secrets" {
  key_id = var.kms_key_alias != "" ? var.kms_key_alias : "alias/${var.project}-secrets-${var.environment}"
}

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------

locals {
  # AWS Region
  aws_region = var.aws_region != "" ? var.aws_region : data.aws_region.current.name

  # Common labels for all resources
  common_labels = {
    "app.kubernetes.io/part-of"    = "greenlang"
    "app.kubernetes.io/managed-by" = "terraform"
    "greenlang.io/obs"             = "004"
    "greenlang.io/environment"     = var.environment
  }

  # SSM parameter prefix
  ssm_prefix = "/${var.project}/${var.environment}/alerting"

  # Determine which integrations to enable
  enable_pagerduty = var.pagerduty_api_key != "" && var.enable_pagerduty
  enable_opsgenie  = var.opsgenie_api_key != "" && var.enable_opsgenie

  # Slack webhook map for SSM storage
  slack_webhooks = {
    critical = var.slack_webhook_critical
    warning  = var.slack_webhook_warning
    info     = var.slack_webhook_info
  }

  # Common tags
  common_tags = merge(var.tags, {
    Project     = "GreenLang"
    Component   = "alerting"
    Environment = var.environment
    ManagedBy   = "terraform"
    OBS         = "004"
  })
}

# -----------------------------------------------------------------------------
# Integration Status Output (for validation)
# -----------------------------------------------------------------------------
# This null_resource logs integration status during apply for operator awareness.

resource "null_resource" "integration_status" {
  triggers = {
    pagerduty = local.enable_pagerduty ? "enabled" : "disabled"
    opsgenie  = local.enable_opsgenie ? "enabled" : "disabled"
    slack     = var.slack_webhook_critical != "" ? "enabled" : "disabled"
    email     = var.ses_from_address != "" ? "enabled" : "disabled"
  }
}
