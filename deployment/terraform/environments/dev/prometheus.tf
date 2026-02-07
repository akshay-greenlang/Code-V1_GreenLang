# =============================================================================
# GreenLang Development Environment - Prometheus Stack
# =============================================================================
#
# Prometheus monitoring stack configuration for development environment.
# Optimized for cost savings while maintaining observability.
#
# Key characteristics:
# - Single Prometheus replica (no HA)
# - Thanos disabled (no long-term storage needed in dev)
# - PushGateway enabled for batch job testing
# - Shorter retention periods
# - Slack notifications only (no PagerDuty)
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
  prometheus_common_tags = merge(local.common_tags, {
    Component   = "observability"
    Stack       = "prometheus"
    CostCenter  = "development"
  })
}

# -----------------------------------------------------------------------------
# Prometheus Stack Module
# -----------------------------------------------------------------------------

module "prometheus_stack" {
  source = "../../modules/prometheus-stack"

  # Environment configuration
  environment  = "dev"
  cluster_name = local.cluster_name
  aws_region   = var.aws_region

  # Prometheus configuration - minimal for dev
  prometheus_replica_count  = 1    # Single replica, no HA
  prometheus_retention_days = 2    # Short retention for dev
  prometheus_storage_size   = "10Gi"  # Smaller storage

  prometheus_resources = {
    requests = {
      cpu    = "250m"
      memory = "1Gi"
    }
    limits = {
      cpu    = "1000m"
      memory = "4Gi"
    }
  }

  # Thanos configuration - disabled for dev
  enable_thanos         = false   # No long-term storage needed
  thanos_retention_raw  = "7d"    # Would apply if enabled
  thanos_retention_5m   = "30d"
  thanos_retention_1h   = "90d"

  # PushGateway - enabled for testing batch jobs
  enable_pushgateway = true
  pushgateway_resources = {
    requests = {
      cpu    = "50m"
      memory = "64Mi"
    }
    limits = {
      cpu    = "200m"
      memory = "256Mi"
    }
  }

  # Alertmanager configuration
  alertmanager_replica_count = 1  # Single replica for dev

  alertmanager_resources = {
    requests = {
      cpu    = "50m"
      memory = "128Mi"
    }
    limits = {
      cpu    = "200m"
      memory = "256Mi"
    }
  }

  # Slack notifications only in dev (no PagerDuty)
  alertmanager_slack_webhook = var.slack_webhook_dev
  alertmanager_pagerduty_key = ""  # No PagerDuty in dev
  alertmanager_slack_channel = "#greenlang-alerts-dev"

  # Notification settings - relaxed for dev
  alert_group_wait      = "1m"    # Longer wait in dev
  alert_group_interval  = "10m"
  alert_repeat_interval = "12h"   # Less frequent repeats

  # ServiceMonitor configuration
  servicemonitor_namespaces = [
    "greenlang",
    "gl-agents",
    "gl-fuel",
    "gl-cbam",
    "gl-building"
  ]

  # Additional scrape configurations for dev-specific services
  additional_scrape_configs = [
    {
      job_name = "dev-services"
      kubernetes_sd_configs = [{
        role = "pod"
        namespaces = {
          names = ["default", "development"]
        }
      }]
      relabel_configs = [
        {
          source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"]
          action        = "keep"
          regex         = "true"
        }
      ]
    }
  ]

  # IRSA configuration for S3 access (even if Thanos disabled, may need later)
  oidc_provider_arn = module.eks.oidc_provider_arn
  oidc_provider_url = module.eks.oidc_provider_url

  # Feature flags
  enable_grafana_operator = false  # Use existing Grafana
  enable_node_exporter    = true
  enable_kube_state_metrics = true

  # Recording rules - minimal set for dev
  recording_rules_enabled = true

  # Alert rules - warning only in dev
  enable_critical_alerts = false  # Only warnings in dev

  # Tags
  tags = local.prometheus_common_tags
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "prometheus_endpoint" {
  description = "Prometheus server endpoint"
  value       = module.prometheus_stack.prometheus_endpoint
}

output "alertmanager_endpoint" {
  description = "Alertmanager endpoint"
  value       = module.prometheus_stack.alertmanager_endpoint
}

output "pushgateway_endpoint" {
  description = "PushGateway endpoint"
  value       = module.prometheus_stack.pushgateway_endpoint
}

output "prometheus_service_account_role_arn" {
  description = "IAM role ARN for Prometheus service account"
  value       = module.prometheus_stack.prometheus_role_arn
}
