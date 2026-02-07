# =============================================================================
# GreenLang Prometheus Stack Module - Alertmanager Configuration
# GreenLang Climate OS | OBS-001
# =============================================================================
# Configures Alertmanager routing and notification channels:
#   - Slack integration for warning and default alerts
#   - PagerDuty integration for critical alerts
#   - Grouping and routing based on severity
#   - Secrets management for webhook URLs and API keys
# =============================================================================

# -----------------------------------------------------------------------------
# Alertmanager Secrets
# -----------------------------------------------------------------------------
# Kubernetes secret containing Slack and PagerDuty credentials.

resource "kubernetes_secret" "alertmanager_secrets" {
  metadata {
    name      = "alertmanager-secrets"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "alertmanager"
      "app.kubernetes.io/component"  = "secrets"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  data = {
    "slack-webhook"   = var.alertmanager_slack_webhook
    "pagerduty-key"   = var.alertmanager_pagerduty_key
  }

  type = "Opaque"

  depends_on = [kubernetes_namespace.monitoring]
}

# -----------------------------------------------------------------------------
# Alertmanager Configuration Secret
# -----------------------------------------------------------------------------
# Contains the full Alertmanager configuration with routes and receivers.

resource "kubernetes_secret" "alertmanager_config" {
  metadata {
    name      = "alertmanager-config"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "alertmanager"
      "app.kubernetes.io/component"  = "config"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  data = {
    "alertmanager.yaml" = yamlencode({
      global = {
        resolve_timeout = "5m"
        slack_api_url   = var.alertmanager_slack_webhook
      }

      # Inhibition rules - prevent duplicate alerts
      inhibit_rules = [
        {
          # If a critical alert is firing, suppress warnings for the same alertname
          source_matchers = [
            "severity = critical"
          ]
          target_matchers = [
            "severity = warning"
          ]
          equal = ["alertname", "cluster", "namespace"]
        },
        {
          # If cluster is down, suppress all other alerts
          source_matchers = [
            "alertname = KubeClusterDown"
          ]
          target_matchers = [
            "severity =~ \"warning|critical\""
          ]
          equal = ["cluster"]
        }
      ]

      # Route tree for alert routing
      route = {
        group_by        = ["alertname", "cluster", "service", "namespace"]
        group_wait      = "30s"
        group_interval  = "5m"
        repeat_interval = "4h"
        receiver        = "default-slack"

        routes = [
          # Critical alerts go to PagerDuty and Slack
          {
            match = {
              severity = "critical"
            }
            receiver = "pagerduty-critical"
            continue = true # Also send to Slack
          },
          # Critical alerts also go to Slack
          {
            match = {
              severity = "critical"
            }
            receiver = "slack-critical"
          },
          # Warning alerts go to warning Slack channel
          {
            match = {
              severity = "warning"
            }
            receiver = "slack-warnings"
          },
          # Info alerts go to default channel
          {
            match = {
              severity = "info"
            }
            receiver = "default-slack"
          },
          # Watchdog heartbeat (silent)
          {
            match = {
              alertname = "Watchdog"
            }
            receiver = "null"
          },
          # DeadMansSwitch for monitoring health
          {
            match = {
              alertname = "DeadMansSwitch"
            }
            receiver = "deadmans-switch"
          }
        ]
      }

      # Receiver definitions
      receivers = [
        # Default Slack receiver
        {
          name = "default-slack"
          slack_configs = [
            {
              channel       = var.alertmanager_slack_channel
              send_resolved = true
              title         = "[{{ .Status | toUpper }}{{ if eq .Status \"firing\" }}:{{ .Alerts.Firing | len }}{{ end }}] {{ .GroupLabels.SortedPairs.Values | join \" \" }}"
              text          = <<-EOT
                {{ range .Alerts -}}
                *Alert:* {{ .Annotations.summary }}
                *Description:* {{ .Annotations.description }}
                *Severity:* {{ .Labels.severity }}
                *Cluster:* {{ .Labels.cluster }}
                *Namespace:* {{ .Labels.namespace }}
                *Source:* {{ .GeneratorURL }}
                {{ end }}
              EOT
              actions = [
                {
                  type = "button"
                  text = "Runbook"
                  url  = "{{ (index .Alerts 0).Annotations.runbook_url }}"
                },
                {
                  type = "button"
                  text = "Source"
                  url  = "{{ (index .Alerts 0).GeneratorURL }}"
                },
                {
                  type = "button"
                  text = "Silence"
                  url  = "{{ template \"__alertmanagerURL\" . }}/#/silences/new?filter=%7B{{ range .CommonLabels.SortedPairs }}{{ .Name }}%3D%22{{ .Value }}%22%2C{{ end }}%7D"
                }
              ]
            }
          ]
        },

        # Slack receiver for critical alerts
        {
          name = "slack-critical"
          slack_configs = [
            {
              channel       = var.alertmanager_slack_channel
              send_resolved = true
              color         = "{{ if eq .Status \"firing\" }}danger{{ else }}good{{ end }}"
              title         = ":rotating_light: [CRITICAL] {{ .GroupLabels.SortedPairs.Values | join \" \" }}"
              text          = <<-EOT
                {{ range .Alerts -}}
                *Alert:* {{ .Annotations.summary }}
                *Description:* {{ .Annotations.description }}
                *Cluster:* {{ .Labels.cluster }}
                *Namespace:* {{ .Labels.namespace }}
                *Started:* {{ .StartsAt }}
                {{ end }}
              EOT
            }
          ]
        },

        # Slack receiver for warnings
        {
          name = "slack-warnings"
          slack_configs = [
            {
              channel       = var.alertmanager_slack_warning_channel
              send_resolved = true
              color         = "{{ if eq .Status \"firing\" }}warning{{ else }}good{{ end }}"
              title         = ":warning: [WARNING] {{ .GroupLabels.SortedPairs.Values | join \" \" }}"
              text          = <<-EOT
                {{ range .Alerts -}}
                *Alert:* {{ .Annotations.summary }}
                *Description:* {{ .Annotations.description }}
                *Cluster:* {{ .Labels.cluster }}
                {{ end }}
              EOT
            }
          ]
        },

        # PagerDuty receiver for critical alerts
        {
          name = "pagerduty-critical"
          pagerduty_configs = [
            {
              service_key = var.alertmanager_pagerduty_key
              severity    = "critical"
              description = "{{ .GroupLabels.alertname }}: {{ .Annotations.summary }}"
              details = {
                firing       = "{{ .Alerts.Firing | len }}"
                resolved     = "{{ .Alerts.Resolved | len }}"
                cluster      = "{{ .CommonLabels.cluster }}"
                namespace    = "{{ .CommonLabels.namespace }}"
              }
            }
          ]
        },

        # Null receiver for silenced alerts
        {
          name = "null"
        },

        # DeadMansSwitch receiver (for external health monitoring)
        {
          name = "deadmans-switch"
          webhook_configs = [
            {
              url             = "http://localhost:9095/webhook" # Placeholder - replace with actual endpoint
              send_resolved   = false
              max_alerts      = 1
            }
          ]
        }
      ]

      # Templates (optional customization)
      templates = [
        "/etc/alertmanager/config/*.tmpl"
      ]
    })
  }

  type = "Opaque"

  depends_on = [kubernetes_namespace.monitoring]
}

# -----------------------------------------------------------------------------
# Alertmanager PrometheusRule for Self-Monitoring
# -----------------------------------------------------------------------------
# Alerts for Alertmanager health and performance.

resource "kubernetes_manifest" "alertmanager_alerts" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PrometheusRule"
    metadata = {
      name      = "alertmanager-health-rules"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "alertmanager"
        "app.kubernetes.io/component"  = "alerts"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      groups = [
        {
          name = "alertmanager.rules"
          rules = [
            {
              alert = "AlertmanagerClusterDown"
              expr  = "count(up{job=\"alertmanager\"} == 1) < 2"
              for   = "5m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "Alertmanager cluster is degraded"
                description = "Less than 2 Alertmanager replicas are running. HA is compromised."
                runbook_url = "https://docs.greenlang.io/runbooks/alertmanager-cluster-down"
              }
            },
            {
              alert = "AlertmanagerConfigInconsistent"
              expr  = "count(count_values(\"config_hash\", alertmanager_config_hash)) > 1"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Alertmanager cluster has inconsistent configuration"
                description = "Alertmanager instances have different configurations. Check for failed reloads."
                runbook_url = "https://docs.greenlang.io/runbooks/alertmanager-config-inconsistent"
              }
            },
            {
              alert = "AlertmanagerNotificationsFailing"
              expr  = "rate(alertmanager_notifications_failed_total[5m]) > 0"
              for   = "5m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "Alertmanager is failing to send notifications"
                description = "Alertmanager is failing to send {{ $labels.integration }} notifications."
                runbook_url = "https://docs.greenlang.io/runbooks/alertmanager-notifications-failing"
              }
            },
            {
              alert = "AlertmanagerClusterCommunicationFailure"
              expr  = "rate(alertmanager_cluster_messages_failed_total[5m]) > 0"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Alertmanager cluster communication failing"
                description = "Alertmanager cluster members are failing to communicate."
                runbook_url = "https://docs.greenlang.io/runbooks/alertmanager-cluster-communication"
              }
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}
