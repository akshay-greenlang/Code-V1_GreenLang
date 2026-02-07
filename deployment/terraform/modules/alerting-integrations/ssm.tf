# =============================================================================
# GreenLang Alerting Integrations - AWS SSM Parameter Store
# GreenLang Climate OS | OBS-004
# =============================================================================
# Stores alerting integration secrets in AWS SSM Parameter Store as
# SecureString parameters encrypted with KMS. These parameters are
# referenced by External Secrets Operator to inject into K8s secrets.
# =============================================================================

# -----------------------------------------------------------------------------
# PagerDuty Routing Key
# -----------------------------------------------------------------------------
# The Events API v2 routing key (integration key) used by the alerting service
# to route alerts to the correct PagerDuty service.

resource "aws_ssm_parameter" "pagerduty_routing_key" {
  name        = "${local.ssm_prefix}/pagerduty-routing-key"
  description = "PagerDuty Events API v2 routing key for GreenLang alerting"
  type        = "SecureString"
  value       = local.enable_pagerduty ? pagerduty_service_integration.events_v2[0].integration_key : "not-configured"
  key_id      = data.aws_kms_key.secrets.id
  tier        = "Standard"

  tags = merge(local.common_tags, {
    Integration = "pagerduty"
    Purpose     = "alert-routing-key"
  })
}

# -----------------------------------------------------------------------------
# Opsgenie API Key
# -----------------------------------------------------------------------------
# The Opsgenie API integration key used by the alerting service to create,
# acknowledge, and close alerts via the Opsgenie Alert API v2.

resource "aws_ssm_parameter" "opsgenie_api_key" {
  name        = "${local.ssm_prefix}/opsgenie-api-key"
  description = "Opsgenie API integration key for GreenLang alerting"
  type        = "SecureString"
  value       = local.enable_opsgenie ? opsgenie_api_integration.greenlang[0].api_key : "not-configured"
  key_id      = data.aws_kms_key.secrets.id
  tier        = "Standard"

  tags = merge(local.common_tags, {
    Integration = "opsgenie"
    Purpose     = "api-key"
  })
}

# -----------------------------------------------------------------------------
# Slack Webhook - Critical
# -----------------------------------------------------------------------------
# Slack incoming webhook URL for critical-severity alert notifications.
# Routes to the dedicated critical alerts channel.

resource "aws_ssm_parameter" "slack_webhook_critical" {
  name        = "${local.ssm_prefix}/slack-webhook-critical"
  description = "Slack webhook URL for critical GreenLang alerts"
  type        = "SecureString"
  value       = var.slack_webhook_critical != "" ? var.slack_webhook_critical : "not-configured"
  key_id      = data.aws_kms_key.secrets.id
  tier        = "Standard"

  tags = merge(local.common_tags, {
    Integration = "slack"
    Purpose     = "critical-alerts"
  })
}

# -----------------------------------------------------------------------------
# Slack Webhook - Warning
# -----------------------------------------------------------------------------
# Slack incoming webhook URL for warning-severity alert notifications.
# Routes to the warning alerts channel for non-urgent issues.

resource "aws_ssm_parameter" "slack_webhook_warning" {
  name        = "${local.ssm_prefix}/slack-webhook-warning"
  description = "Slack webhook URL for warning GreenLang alerts"
  type        = "SecureString"
  value       = var.slack_webhook_warning != "" ? var.slack_webhook_warning : "not-configured"
  key_id      = data.aws_kms_key.secrets.id
  tier        = "Standard"

  tags = merge(local.common_tags, {
    Integration = "slack"
    Purpose     = "warning-alerts"
  })
}

# -----------------------------------------------------------------------------
# Slack Webhook - Info
# -----------------------------------------------------------------------------
# Slack incoming webhook URL for informational alert notifications.
# Routes to the general alerts channel for awareness items.

resource "aws_ssm_parameter" "slack_webhook_info" {
  name        = "${local.ssm_prefix}/slack-webhook-info"
  description = "Slack webhook URL for informational GreenLang alerts"
  type        = "SecureString"
  value       = var.slack_webhook_info != "" ? var.slack_webhook_info : "not-configured"
  key_id      = data.aws_kms_key.secrets.id
  tier        = "Standard"

  tags = merge(local.common_tags, {
    Integration = "slack"
    Purpose     = "info-alerts"
  })
}

# -----------------------------------------------------------------------------
# SES Configuration Parameters
# -----------------------------------------------------------------------------
# Email notification configuration stored in SSM for the alerting service.

resource "aws_ssm_parameter" "ses_from_address" {
  name        = "${local.ssm_prefix}/ses-from-address"
  description = "SES sender email address for alert notifications"
  type        = "String"
  value       = var.ses_from_address

  tags = merge(local.common_tags, {
    Integration = "ses"
    Purpose     = "from-address"
  })
}

resource "aws_ssm_parameter" "ses_critical_recipients" {
  name        = "${local.ssm_prefix}/ses-critical-recipients"
  description = "Email recipients for critical alert notifications"
  type        = "String"
  value       = length(var.ses_critical_recipients) > 0 ? join(",", var.ses_critical_recipients) : "not-configured"

  tags = merge(local.common_tags, {
    Integration = "ses"
    Purpose     = "critical-recipients"
  })
}

resource "aws_ssm_parameter" "ses_warning_recipients" {
  name        = "${local.ssm_prefix}/ses-warning-recipients"
  description = "Email recipients for warning alert notifications"
  type        = "String"
  value       = length(var.ses_warning_recipients) > 0 ? join(",", var.ses_warning_recipients) : "not-configured"

  tags = merge(local.common_tags, {
    Integration = "ses"
    Purpose     = "warning-recipients"
  })
}

# -----------------------------------------------------------------------------
# IAM Policy for ESO access to SSM parameters
# -----------------------------------------------------------------------------
# Policy document that grants External Secrets Operator read access to
# all alerting SSM parameters for syncing into Kubernetes secrets.

data "aws_iam_policy_document" "eso_alerting_access" {
  statement {
    sid    = "AllowReadAlertingParameters"
    effect = "Allow"

    actions = [
      "ssm:GetParameter",
      "ssm:GetParameters",
      "ssm:GetParametersByPath",
    ]

    resources = [
      "arn:aws:ssm:${local.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${local.ssm_prefix}/*"
    ]
  }

  statement {
    sid    = "AllowDecryptAlertingParameters"
    effect = "Allow"

    actions = [
      "kms:Decrypt",
    ]

    resources = [
      data.aws_kms_key.secrets.arn
    ]
  }
}

resource "aws_iam_policy" "eso_alerting_access" {
  name        = "${var.project}-${var.environment}-eso-alerting-access"
  description = "Allows External Secrets Operator to read alerting SSM parameters"
  policy      = data.aws_iam_policy_document.eso_alerting_access.json

  tags = local.common_tags
}
