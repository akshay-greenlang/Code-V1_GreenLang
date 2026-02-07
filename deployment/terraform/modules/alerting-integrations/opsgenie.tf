# =============================================================================
# GreenLang Alerting Integrations - Opsgenie
# GreenLang Climate OS | OBS-004
# =============================================================================
# Configures Opsgenie team, API integration, escalation, and on-call schedule
# for alert management and on-call routing.
# =============================================================================

# -----------------------------------------------------------------------------
# Opsgenie Team
# -----------------------------------------------------------------------------
# Team that owns GreenLang alerts in Opsgenie. All integrations and schedules
# are associated with this team.

resource "opsgenie_team" "greenlang" {
  count = local.enable_opsgenie ? 1 : 0

  name        = "${var.opsgenie_team_name} (${var.environment})"
  description = var.opsgenie_team_description

  dynamic "member" {
    for_each = var.opsgenie_responders
    content {
      id   = member.value
      role = member.key == 0 ? "admin" : "user"
    }
  }

  delete_default_resources = false
}

# -----------------------------------------------------------------------------
# Opsgenie API Integration
# -----------------------------------------------------------------------------
# REST API integration for receiving alerts from the GreenLang alerting service.
# Produces an API key that the alerting service uses to create/acknowledge/close
# alerts via the Opsgenie Alert API v2.

resource "opsgenie_api_integration" "greenlang" {
  count = local.enable_opsgenie ? 1 : 0

  name = "${var.project}-${var.environment}-alerting"
  type = "API"

  owner_team_id          = opsgenie_team.greenlang[0].id
  allow_write_access     = true
  allow_configuration_access = false
  suppress_notifications = var.environment == "dev" ? true : false
  enabled                = true

  responders {
    type = "team"
    id   = opsgenie_team.greenlang[0].id
  }
}

# -----------------------------------------------------------------------------
# Opsgenie Escalation
# -----------------------------------------------------------------------------
# Defines escalation steps when an alert is not acknowledged within the
# specified timeframe. Escalates from team to individual responders.

resource "opsgenie_escalation" "greenlang" {
  count = local.enable_opsgenie ? 1 : 0

  name          = "${var.project}-${var.environment}-escalation"
  owner_team_id = opsgenie_team.greenlang[0].id
  description   = "GreenLang ${var.environment} alert escalation"

  rules {
    condition   = "if-not-acked"
    notify_type = "default"
    delay       = 5

    recipient {
      type = "team"
      id   = opsgenie_team.greenlang[0].id
    }
  }

  rules {
    condition   = "if-not-acked"
    notify_type = "default"
    delay       = 15

    dynamic "recipient" {
      for_each = length(var.opsgenie_responders) > 0 ? [var.opsgenie_responders[0]] : []
      content {
        type = "user"
        id   = recipient.value
      }
    }
  }

  rules {
    condition   = "if-not-acked"
    notify_type = "all"
    delay       = 30

    recipient {
      type = "team"
      id   = opsgenie_team.greenlang[0].id
    }
  }

  repeat {
    wait_interval         = 15
    count                 = 3
    reset_recipient_states = true
    close_alert_after_all = true
  }
}

# -----------------------------------------------------------------------------
# Opsgenie Schedule (Weekly Rotation)
# -----------------------------------------------------------------------------
# On-call schedule with weekly rotation for the GreenLang platform team.
# The schedule provides on-call lookup for the alerting service.

resource "opsgenie_schedule" "greenlang" {
  count = local.enable_opsgenie ? 1 : 0

  name          = "${var.project}-${var.environment}-oncall"
  owner_team_id = opsgenie_team.greenlang[0].id
  description   = "GreenLang ${var.environment} on-call rotation"
  timezone      = var.opsgenie_schedule_timezone
  enabled       = true
}

resource "opsgenie_schedule_rotation" "weekly" {
  count = local.enable_opsgenie && length(var.opsgenie_responders) > 0 ? 1 : 0

  schedule_id  = opsgenie_schedule.greenlang[0].id
  name         = "weekly-rotation"
  type         = "weekly"
  start_date   = "2026-01-01T00:00:00Z"
  time_restriction_type = null

  dynamic "participant" {
    for_each = var.opsgenie_responders
    content {
      type = "user"
      id   = participant.value
    }
  }
}

# -----------------------------------------------------------------------------
# Opsgenie Notification Policy
# -----------------------------------------------------------------------------
# Team notification policy that controls how alerts are routed to team members
# based on priority and time of day.

resource "opsgenie_notification_policy" "greenlang" {
  count = local.enable_opsgenie ? 1 : 0

  name          = "${var.project}-${var.environment}-notification-policy"
  team_id       = opsgenie_team.greenlang[0].id
  policy_description = "GreenLang ${var.environment} notification routing"
  enabled       = true

  filter {
    type = "match-all"
  }

  auto_close_action {
    duration {
      time_amount = 24
      time_unit   = "hours"
    }
  }

  auto_restart_action {
    duration {
      time_amount = 30
      time_unit   = "minutes"
    }
    max_repeat_count = 3
  }
}
