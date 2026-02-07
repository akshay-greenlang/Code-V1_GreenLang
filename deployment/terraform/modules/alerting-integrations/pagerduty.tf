# =============================================================================
# GreenLang Alerting Integrations - PagerDuty
# GreenLang Climate OS | OBS-004
# =============================================================================
# Configures PagerDuty service, escalation policy, and Events API v2
# integration for critical alert notification delivery.
# =============================================================================

# -----------------------------------------------------------------------------
# PagerDuty Provider
# -----------------------------------------------------------------------------
# The PagerDuty provider is configured via the pagerduty_api_key variable.
# When PagerDuty is disabled, all resources are conditionally skipped.

# -----------------------------------------------------------------------------
# PagerDuty Escalation Policy
# -----------------------------------------------------------------------------
# 3-level escalation policy with configurable timeout per level.
# Level 1: Primary on-call responder (or schedule)
# Level 2: Secondary responder + team lead
# Level 3: Engineering manager / VP on-call

resource "pagerduty_escalation_policy" "greenlang" {
  count = local.enable_pagerduty ? 1 : 0

  name        = "${var.project}-${var.environment}-escalation"
  description = "GreenLang ${var.environment} alert escalation policy"
  num_loops   = 2

  # Level 1: Primary on-call
  rule {
    escalation_delay_in_minutes = var.pagerduty_escalation_timeout

    dynamic "target" {
      for_each = length(var.pagerduty_schedule_ids) > 0 ? [var.pagerduty_schedule_ids[0]] : []
      content {
        type = "schedule_reference"
        id   = target.value
      }
    }

    dynamic "target" {
      for_each = length(var.pagerduty_user_ids) > 0 ? [var.pagerduty_user_ids[0]] : []
      content {
        type = "user_reference"
        id   = target.value
      }
    }
  }

  # Level 2: Secondary responder
  rule {
    escalation_delay_in_minutes = var.pagerduty_escalation_timeout

    dynamic "target" {
      for_each = length(var.pagerduty_schedule_ids) > 1 ? [var.pagerduty_schedule_ids[1]] : []
      content {
        type = "schedule_reference"
        id   = target.value
      }
    }

    dynamic "target" {
      for_each = length(var.pagerduty_user_ids) > 1 ? [var.pagerduty_user_ids[1]] : []
      content {
        type = "user_reference"
        id   = target.value
      }
    }
  }

  # Level 3: Management escalation
  rule {
    escalation_delay_in_minutes = var.pagerduty_escalation_timeout

    dynamic "target" {
      for_each = length(var.pagerduty_user_ids) > 2 ? [var.pagerduty_user_ids[2]] : []
      content {
        type = "user_reference"
        id   = target.value
      }
    }
  }
}

# -----------------------------------------------------------------------------
# PagerDuty Service
# -----------------------------------------------------------------------------
# Service represents GreenLang Platform in PagerDuty. All alerts are routed
# through this service and follow the escalation policy defined above.

resource "pagerduty_service" "greenlang" {
  count = local.enable_pagerduty ? 1 : 0

  name                    = "${var.pagerduty_service_name} (${var.environment})"
  description             = var.pagerduty_service_description
  auto_resolve_timeout    = 14400  # 4 hours
  acknowledgement_timeout = 1800   # 30 minutes
  escalation_policy       = pagerduty_escalation_policy.greenlang[0].id
  alert_creation          = "create_alerts_and_incidents"

  incident_urgency_rule {
    type    = "constant"
    urgency = var.environment == "prod" ? "high" : "low"
  }

  auto_pause_notifications_parameters {
    enabled = true
    timeout = 300  # 5 minutes auto-pause for transient alerts
  }
}

# -----------------------------------------------------------------------------
# PagerDuty Service Integration - Events API v2
# -----------------------------------------------------------------------------
# Events API v2 integration provides the routing key used by the alerting
# service to send alert events (trigger, acknowledge, resolve) to PagerDuty.

resource "pagerduty_service_integration" "events_v2" {
  count = local.enable_pagerduty ? 1 : 0

  name    = "GreenLang Alerting Service (Events API v2)"
  service = pagerduty_service.greenlang[0].id
  vendor  = data.pagerduty_vendor.events_v2[0].id
}

# Lookup Events API v2 vendor
data "pagerduty_vendor" "events_v2" {
  count = local.enable_pagerduty ? 1 : 0
  name  = "Events API v2"
}

# -----------------------------------------------------------------------------
# PagerDuty Service Event Rules
# -----------------------------------------------------------------------------
# Event rules for severity-based routing within the PagerDuty service.

resource "pagerduty_service_event_rule" "suppress_info" {
  count   = local.enable_pagerduty ? 1 : 0
  service = pagerduty_service.greenlang[0].id

  position = 0

  conditions {
    operator = "and"
    subconditions {
      operator = "contains"
      parameter {
        value = "info"
        path  = "payload.severity"
      }
    }
  }

  actions {
    suppress {
      value = true
    }
  }
}

resource "pagerduty_service_event_rule" "set_critical_urgency" {
  count   = local.enable_pagerduty ? 1 : 0
  service = pagerduty_service.greenlang[0].id

  position = 1

  conditions {
    operator = "and"
    subconditions {
      operator = "contains"
      parameter {
        value = "critical"
        path  = "payload.severity"
      }
    }
  }

  actions {
    urgency {
      value = "high"
    }
  }
}
