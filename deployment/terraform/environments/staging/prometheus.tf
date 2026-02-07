# =============================================================================
# GreenLang Staging Environment - Prometheus Stack
# =============================================================================
#
# Prometheus monitoring stack configuration for staging environment.
# Production-like setup with some cost optimizations.
#
# Key characteristics:
# - 2 Prometheus replicas (HA)
# - Thanos enabled with 30-day S3 retention
# - PushGateway enabled
# - Slack notifications only (no PagerDuty paging)
# - Full alert rules enabled
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
    CostCenter  = "staging"
  })
}

# -----------------------------------------------------------------------------
# Prometheus Stack Module
# -----------------------------------------------------------------------------

module "prometheus_stack" {
  source = "../../modules/prometheus-stack"

  # Environment configuration
  environment  = "staging"
  cluster_name = local.cluster_name
  aws_region   = var.aws_region

  # Prometheus configuration - production-like with cost savings
  prometheus_replica_count  = 2     # HA setup
  prometheus_retention_days = 7     # 7-day local retention
  prometheus_storage_size   = "25Gi"  # Moderate storage

  prometheus_resources = {
    requests = {
      cpu    = "500m"
      memory = "2Gi"
    }
    limits = {
      cpu    = "2000m"
      memory = "8Gi"
    }
  }

  # Pod anti-affinity for HA
  prometheus_pod_anti_affinity = true

  # Thanos configuration - enabled with reduced retention
  enable_thanos         = true
  thanos_retention_raw  = "30d"    # 30 days raw data
  thanos_retention_5m   = "60d"    # 60 days 5m downsampled
  thanos_retention_1h   = "180d"   # 180 days 1h downsampled (6 months)

  thanos_query_replicas    = 2
  thanos_store_gw_replicas = 2
  thanos_compactor_replicas = 1  # Always 1 for compactor

  thanos_query_resources = {
    requests = {
      cpu    = "250m"
      memory = "512Mi"
    }
    limits = {
      cpu    = "1000m"
      memory = "2Gi"
    }
  }

  thanos_store_gw_resources = {
    requests = {
      cpu    = "250m"
      memory = "1Gi"
    }
    limits = {
      cpu    = "1000m"
      memory = "4Gi"
    }
  }

  thanos_compactor_resources = {
    requests = {
      cpu    = "250m"
      memory = "1Gi"
    }
    limits = {
      cpu    = "1000m"
      memory = "4Gi"
    }
  }

  thanos_compactor_storage_size = "50Gi"
  thanos_store_gw_storage_size  = "25Gi"

  # S3 bucket for Thanos
  thanos_s3_bucket_name = "gl-thanos-metrics-staging"

  # PushGateway - enabled
  enable_pushgateway = true
  pushgateway_replicas = 2  # HA for staging

  pushgateway_resources = {
    requests = {
      cpu    = "100m"
      memory = "128Mi"
    }
    limits = {
      cpu    = "500m"
      memory = "512Mi"
    }
  }

  # Alertmanager configuration - HA
  alertmanager_replica_count = 2

  alertmanager_resources = {
    requests = {
      cpu    = "100m"
      memory = "256Mi"
    }
    limits = {
      cpu    = "500m"
      memory = "512Mi"
    }
  }

  # Slack notifications only in staging (no PagerDuty paging)
  alertmanager_slack_webhook = var.slack_webhook_staging
  alertmanager_pagerduty_key = ""  # No PagerDuty paging in staging
  alertmanager_slack_channel = "#greenlang-alerts-staging"

  # Notification settings - tighter than dev, looser than prod
  alert_group_wait      = "30s"
  alert_group_interval  = "5m"
  alert_repeat_interval = "6h"

  # ServiceMonitor configuration
  servicemonitor_namespaces = [
    "greenlang",
    "gl-agents",
    "gl-fuel",
    "gl-cbam",
    "gl-building",
    "gl-scope3"
  ]

  # Additional scrape configurations
  additional_scrape_configs = [
    {
      job_name = "staging-services"
      kubernetes_sd_configs = [{
        role = "pod"
        namespaces = {
          names = ["staging", "qa"]
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

  # IRSA configuration for S3 access
  oidc_provider_arn = module.eks.oidc_provider_arn
  oidc_provider_url = module.eks.oidc_provider_url

  # Feature flags
  enable_grafana_operator   = false  # Use existing Grafana
  enable_node_exporter      = true
  enable_kube_state_metrics = true

  # Recording rules - full set for staging
  recording_rules_enabled = true

  # Alert rules - all enabled
  enable_critical_alerts = true
  enable_warning_alerts  = true

  # Query frontend for caching (optional in staging)
  enable_query_frontend = true
  query_frontend_replicas = 2

  query_frontend_cache_config = {
    type = "IN-MEMORY"
    config = {
      max_size       = "256MB"
      max_size_items = 500
      validity       = "5m"
    }
  }

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

output "thanos_query_endpoint" {
  description = "Thanos Query endpoint for unified queries"
  value       = module.prometheus_stack.thanos_query_endpoint
}

output "alertmanager_endpoint" {
  description = "Alertmanager endpoint"
  value       = module.prometheus_stack.alertmanager_endpoint
}

output "pushgateway_endpoint" {
  description = "PushGateway endpoint"
  value       = module.prometheus_stack.pushgateway_endpoint
}

output "thanos_bucket_name" {
  description = "S3 bucket for Thanos metrics"
  value       = module.prometheus_stack.thanos_bucket_name
}

output "prometheus_service_account_role_arn" {
  description = "IAM role ARN for Prometheus service account"
  value       = module.prometheus_stack.prometheus_role_arn
}

output "thanos_service_account_role_arn" {
  description = "IAM role ARN for Thanos service accounts"
  value       = module.prometheus_stack.thanos_role_arn
}
