# =============================================================================
# GreenLang Alerting Integrations Module - Outputs
# GreenLang Climate OS | OBS-004
# =============================================================================
# Output values for integration with other modules and services.
# =============================================================================

# -----------------------------------------------------------------------------
# PagerDuty Outputs
# -----------------------------------------------------------------------------

output "pagerduty_enabled" {
  description = "Whether PagerDuty integration is enabled"
  value       = local.enable_pagerduty
}

output "pagerduty_service_id" {
  description = "PagerDuty service ID for GreenLang alerts"
  value       = local.enable_pagerduty ? pagerduty_service.greenlang[0].id : null
}

output "pagerduty_escalation_policy_id" {
  description = "PagerDuty escalation policy ID"
  value       = local.enable_pagerduty ? pagerduty_escalation_policy.greenlang[0].id : null
}

output "pagerduty_integration_key" {
  description = "PagerDuty Events API v2 integration key (routing key)"
  value       = local.enable_pagerduty ? pagerduty_service_integration.events_v2[0].integration_key : null
  sensitive   = true
}

# -----------------------------------------------------------------------------
# Opsgenie Outputs
# -----------------------------------------------------------------------------

output "opsgenie_enabled" {
  description = "Whether Opsgenie integration is enabled"
  value       = local.enable_opsgenie
}

output "opsgenie_team_id" {
  description = "Opsgenie team ID for GreenLang"
  value       = local.enable_opsgenie ? opsgenie_team.greenlang[0].id : null
}

output "opsgenie_api_integration_id" {
  description = "Opsgenie API integration ID"
  value       = local.enable_opsgenie ? opsgenie_api_integration.greenlang[0].id : null
}

output "opsgenie_api_integration_key" {
  description = "Opsgenie API integration key"
  value       = local.enable_opsgenie ? opsgenie_api_integration.greenlang[0].api_key : null
  sensitive   = true
}

# -----------------------------------------------------------------------------
# SSM Parameter Outputs
# -----------------------------------------------------------------------------

output "ssm_parameter_arns" {
  description = "ARNs of all SSM parameters created for alerting secrets"
  value = {
    pagerduty_routing_key = aws_ssm_parameter.pagerduty_routing_key.arn
    opsgenie_api_key      = aws_ssm_parameter.opsgenie_api_key.arn
    slack_critical        = aws_ssm_parameter.slack_webhook_critical.arn
    slack_warning         = aws_ssm_parameter.slack_webhook_warning.arn
    slack_info            = aws_ssm_parameter.slack_webhook_info.arn
  }
}

output "ssm_parameter_names" {
  description = "Names of all SSM parameters for alerting secrets"
  value = {
    pagerduty_routing_key = aws_ssm_parameter.pagerduty_routing_key.name
    opsgenie_api_key      = aws_ssm_parameter.opsgenie_api_key.name
    slack_critical        = aws_ssm_parameter.slack_webhook_critical.name
    slack_warning         = aws_ssm_parameter.slack_webhook_warning.name
    slack_info            = aws_ssm_parameter.slack_webhook_info.name
  }
}

# -----------------------------------------------------------------------------
# Integration URLs
# -----------------------------------------------------------------------------

output "integration_urls" {
  description = "Integration endpoint URLs for alerting service configuration"
  value = {
    pagerduty_events_url = "https://events.pagerduty.com/v2/enqueue"
    opsgenie_alerts_url  = "https://api.opsgenie.com/v2/alerts"
    ses_region           = local.aws_region
    ses_from_address     = var.ses_from_address
  }
}

# -----------------------------------------------------------------------------
# Summary Output
# -----------------------------------------------------------------------------

output "integration_summary" {
  description = "Summary of enabled integrations"
  value = {
    pagerduty = local.enable_pagerduty
    opsgenie  = local.enable_opsgenie
    slack     = var.slack_webhook_critical != ""
    email     = var.ses_from_address != ""
  }
}
