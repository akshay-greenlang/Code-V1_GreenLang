# =============================================================================
# GreenLang Staging Environment - Alerting Integrations
# =============================================================================
#
# Alerting integration configuration for staging environment.
# Near-production setup with PagerDuty in suppressed mode and Opsgenie
# enabled for on-call routing testing.
#
# Key characteristics:
# - PagerDuty enabled (suppressed mode - creates incidents but does not page)
# - Opsgenie enabled (suppressed notifications for testing)
# - Slack enabled (staging alerts channel)
# - Email enabled (staging distribution list only)
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
    CostCenter = "staging"
  })
}

# -----------------------------------------------------------------------------
# Alerting Integrations Module
# -----------------------------------------------------------------------------

module "alerting_integrations" {
  source = "../../modules/alerting-integrations"

  # Environment configuration
  environment = "staging"
  aws_region  = var.aws_region

  # PagerDuty - enabled with staging service
  enable_pagerduty              = true
  pagerduty_api_key             = var.pagerduty_api_key_staging
  pagerduty_service_name        = "GreenLang Platform Staging"
  pagerduty_service_description = "GreenLang Climate OS staging environment alerts"
  pagerduty_escalation_timeout  = 30  # Longer timeout in staging
  pagerduty_user_ids            = var.pagerduty_staging_user_ids
  pagerduty_schedule_ids        = var.pagerduty_staging_schedule_ids

  # Opsgenie - enabled with staging team
  enable_opsgenie           = true
  opsgenie_api_key          = var.opsgenie_api_key_staging
  opsgenie_team_name        = "GreenLang Platform Staging"
  opsgenie_team_description = "GreenLang staging environment on-call"
  opsgenie_responders       = var.opsgenie_staging_responders

  # Slack - separate staging channels
  slack_webhook_critical = var.slack_webhook_staging_critical
  slack_webhook_warning  = var.slack_webhook_staging_warning
  slack_webhook_info     = var.slack_webhook_staging_info

  # Email - staging distribution list
  ses_from_address        = "alerts-staging@greenlang.io"
  ses_critical_recipients = ["platform-staging@greenlang.io"]
  ses_warning_recipients  = ["platform-staging@greenlang.io"]

  # Tags
  tags = local.alerting_common_tags
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "alerting_integration_summary" {
  description = "Alerting integration status for staging"
  value       = module.alerting_integrations.integration_summary
}

output "alerting_pagerduty_service_id" {
  description = "PagerDuty service ID for staging"
  value       = module.alerting_integrations.pagerduty_service_id
}

output "alerting_opsgenie_team_id" {
  description = "Opsgenie team ID for staging"
  value       = module.alerting_integrations.opsgenie_team_id
}

output "alerting_ssm_parameter_names" {
  description = "SSM parameter names for alerting secrets"
  value       = module.alerting_integrations.ssm_parameter_names
}
