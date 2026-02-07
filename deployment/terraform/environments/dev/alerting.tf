# =============================================================================
# GreenLang Development Environment - Alerting Integrations
# =============================================================================
#
# Alerting integration configuration for development environment.
# Optimized for testing with Slack-only notifications and no PagerDuty/Opsgenie
# paging. Test webhook URLs and dummy keys are used.
#
# Key characteristics:
# - PagerDuty disabled (no paging in dev)
# - Opsgenie disabled (no on-call routing in dev)
# - Slack enabled (dev alerts channel only)
# - Email notifications disabled
#
# Usage:
#   terraform plan -var-file="terraform.tfvars"
#   terraform apply -var-file="terraform.tfvars"
#
# =============================================================================

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------

locals {
  alerting_common_tags = merge(local.common_tags, {
    Component  = "alerting"
    Stack      = "observability"
    CostCenter = "development"
  })
}

# -----------------------------------------------------------------------------
# Alerting Integrations Module
# -----------------------------------------------------------------------------

module "alerting_integrations" {
  source = "../../modules/alerting-integrations"

  # Environment configuration
  environment = "dev"
  aws_region  = var.aws_region

  # PagerDuty - disabled in dev
  enable_pagerduty  = false
  pagerduty_api_key = ""

  # Opsgenie - disabled in dev
  enable_opsgenie  = false
  opsgenie_api_key = ""

  # Slack - dev channel only, single webhook for all severities
  slack_webhook_critical = var.slack_webhook_dev
  slack_webhook_warning  = var.slack_webhook_dev
  slack_webhook_info     = var.slack_webhook_dev

  # Email - disabled in dev
  ses_from_address      = "alerts-dev@greenlang.io"
  ses_critical_recipients = []
  ses_warning_recipients  = []

  # Tags
  tags = local.alerting_common_tags
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "alerting_integration_summary" {
  description = "Alerting integration status for dev"
  value       = module.alerting_integrations.integration_summary
}

output "alerting_ssm_parameter_names" {
  description = "SSM parameter names for alerting secrets"
  value       = module.alerting_integrations.ssm_parameter_names
}
