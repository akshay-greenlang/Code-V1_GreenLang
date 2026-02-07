# =============================================================================
# GreenLang Production Environment - Alerting Integrations
# =============================================================================
#
# Alerting integration configuration for production environment.
# Full integration with PagerDuty, Opsgenie, Slack, and Email for
# comprehensive alert notification and on-call management.
#
# Key characteristics:
# - PagerDuty enabled (full paging with 15-minute escalation intervals)
# - Opsgenie enabled (on-call schedule, escalation, notification policy)
# - Slack enabled (severity-separated channels)
# - Email enabled (critical/warning distribution lists)
# - All integrations use production API keys from tfvars/Vault
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
    CostCenter = "production"
    Criticality = "high"
  })
}

# -----------------------------------------------------------------------------
# Alerting Integrations Module
# -----------------------------------------------------------------------------

module "alerting_integrations" {
  source = "../../modules/alerting-integrations"

  # Environment configuration
  environment = "prod"
  aws_region  = var.aws_region

  # PagerDuty - full production integration
  enable_pagerduty              = true
  pagerduty_api_key             = var.pagerduty_api_key_prod
  pagerduty_service_name        = "GreenLang Platform"
  pagerduty_service_description = "GreenLang Climate OS production alerts - critical infrastructure"
  pagerduty_escalation_timeout  = 15  # 15-minute escalation in prod
  pagerduty_user_ids            = var.pagerduty_prod_user_ids
  pagerduty_schedule_ids        = var.pagerduty_prod_schedule_ids

  # Opsgenie - full production integration
  enable_opsgenie              = true
  opsgenie_api_key             = var.opsgenie_api_key_prod
  opsgenie_team_name           = "GreenLang Platform"
  opsgenie_team_description    = "GreenLang Climate OS Production On-Call"
  opsgenie_responders          = var.opsgenie_prod_responders
  opsgenie_schedule_timezone   = "America/New_York"

  # Slack - severity-separated production channels
  slack_webhook_critical = var.slack_webhook_prod_critical
  slack_webhook_warning  = var.slack_webhook_prod_warning
  slack_webhook_info     = var.slack_webhook_prod_info

  # Email - production distribution lists
  ses_from_address        = "alerts@greenlang.io"
  ses_reply_to            = "platform-team@greenlang.io"
  ses_critical_recipients = ["platform-oncall@greenlang.io", "engineering-leads@greenlang.io"]
  ses_warning_recipients  = ["platform-team@greenlang.io"]

  # Tags
  tags = local.alerting_common_tags
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "alerting_integration_summary" {
  description = "Alerting integration status for production"
  value       = module.alerting_integrations.integration_summary
}

output "alerting_pagerduty_service_id" {
  description = "PagerDuty service ID for production"
  value       = module.alerting_integrations.pagerduty_service_id
}

output "alerting_opsgenie_team_id" {
  description = "Opsgenie team ID for production"
  value       = module.alerting_integrations.opsgenie_team_id
}

output "alerting_ssm_parameter_arns" {
  description = "SSM parameter ARNs for alerting secrets"
  value       = module.alerting_integrations.ssm_parameter_arns
}

output "alerting_ssm_parameter_names" {
  description = "SSM parameter names for alerting secrets"
  value       = module.alerting_integrations.ssm_parameter_names
}

output "alerting_eso_policy_arn" {
  description = "IAM policy ARN for ESO access to alerting parameters"
  value       = module.alerting_integrations.eso_alerting_access_policy_arn
  sensitive   = false
}
