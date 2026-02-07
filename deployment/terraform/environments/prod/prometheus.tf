# =============================================================================
# GreenLang Production Environment - Prometheus Stack
# =============================================================================
#
# Prometheus monitoring stack configuration for production environment.
# Full HA setup with long-term storage and complete alerting.
#
# Key characteristics:
# - 2 Prometheus replicas with pod anti-affinity (HA)
# - Thanos enabled with 2-year S3 retention
# - PushGateway HA
# - Full alerting: Slack + PagerDuty for critical alerts
# - Query frontend for caching
# - All recording rules and alert rules enabled
#
# Usage:
#   terraform plan -var-file="terraform.tfvars"
#   terraform apply -var-file="terraform.tfvars"
#
# SLA: 99.9% availability for metrics collection
#
# =============================================================================

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------

locals {
  prometheus_common_tags = merge(local.common_tags, {
    Component   = "observability"
    Stack       = "prometheus"
    CostCenter  = "production"
    Criticality = "high"
    SLA         = "99.9"
  })
}

# -----------------------------------------------------------------------------
# Prometheus Stack Module
# -----------------------------------------------------------------------------

module "prometheus_stack" {
  source = "../../modules/prometheus-stack"

  # Environment configuration
  environment  = "prod"
  cluster_name = local.cluster_name
  aws_region   = var.aws_region

  # Prometheus configuration - full HA production setup
  prometheus_replica_count  = 2     # HA with anti-affinity
  prometheus_retention_days = 7     # 7-day local retention (Thanos handles long-term)
  prometheus_storage_size   = "50Gi"  # Full storage

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

  # Pod anti-affinity for HA - required for production
  prometheus_pod_anti_affinity = true

  # Topology spread constraints for zone distribution
  prometheus_topology_spread = {
    max_skew           = 1
    topology_key       = "topology.kubernetes.io/zone"
    when_unsatisfiable = "DoNotSchedule"
  }

  # Thanos configuration - full 2-year retention
  enable_thanos         = true
  thanos_retention_raw  = "30d"    # 30 days raw data (15s resolution)
  thanos_retention_5m   = "120d"   # 120 days 5m downsampled
  thanos_retention_1h   = "730d"   # 2 years 1h downsampled

  thanos_query_replicas    = 2     # HA query layer
  thanos_store_gw_replicas = 2     # HA store gateway
  thanos_compactor_replicas = 1    # Always 1 for compactor

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

  thanos_compactor_storage_size = "100Gi"  # Large for block compaction
  thanos_store_gw_storage_size  = "50Gi"   # Cache storage

  # S3 bucket for Thanos - production
  thanos_s3_bucket_name = "gl-thanos-metrics-prod"

  # S3 lifecycle configuration
  thanos_s3_lifecycle_rules = [
    {
      id      = "intelligent-tiering"
      enabled = true
      transition = {
        days          = 30
        storage_class = "INTELLIGENT_TIERING"
      }
    },
    {
      id      = "glacier-archive"
      enabled = true
      transition = {
        days          = 365
        storage_class = "GLACIER"
      }
    },
    {
      id      = "expiration"
      enabled = true
      expiration = {
        days = 730  # 2 years
      }
    }
  ]

  # Thanos Ruler for alerting on historical data
  enable_thanos_ruler = true
  thanos_ruler_replicas = 2

  # PushGateway - HA for production
  enable_pushgateway = true
  pushgateway_replicas = 2

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

  pushgateway_persistence = {
    enabled       = true
    size          = "2Gi"
    storage_class = "gp3"
  }

  # Alertmanager configuration - full HA
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

  alertmanager_storage = {
    enabled       = true
    size          = "10Gi"
    storage_class = "gp3"
  }

  # Full alerting stack - Slack + PagerDuty
  alertmanager_slack_webhook = var.slack_webhook_prod
  alertmanager_pagerduty_key = var.pagerduty_integration_key
  alertmanager_slack_channel = "#greenlang-alerts-prod"

  # PagerDuty configuration
  pagerduty_service_key_critical = var.pagerduty_critical_key
  pagerduty_service_key_warning  = var.pagerduty_warning_key

  # Email notifications for weekly summaries
  alertmanager_email_config = {
    enabled    = true
    smtp_host  = var.smtp_host
    smtp_from  = "alerts@greenlang.io"
    smtp_to    = ["platform-team@greenlang.io", "sre@greenlang.io"]
  }

  # Notification settings - production SLA
  alert_group_wait      = "30s"   # Quick initial notification
  alert_group_interval  = "5m"    # Moderate grouping
  alert_repeat_interval = "4h"    # Regular reminders

  # Inhibition rules
  alertmanager_inhibit_rules = [
    {
      source_match = {
        severity = "critical"
      }
      target_match = {
        severity = "warning"
      }
      equal = ["alertname", "cluster", "service"]
    }
  ]

  # ServiceMonitor configuration - all production namespaces
  servicemonitor_namespaces = [
    "greenlang",
    "gl-agents",
    "gl-fuel",
    "gl-cbam",
    "gl-building",
    "gl-scope3",
    "gl-compliance",
    "gl-reporting"
  ]

  # Additional scrape configurations for production services
  additional_scrape_configs = [
    {
      job_name = "production-services"
      kubernetes_sd_configs = [{
        role = "pod"
        namespaces = {
          names = ["production", "greenlang-prod"]
        }
      }]
      relabel_configs = [
        {
          source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"]
          action        = "keep"
          regex         = "true"
        },
        {
          source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_path"]
          action        = "replace"
          target_label  = "__metrics_path__"
          regex         = "(.+)"
        }
      ]
    },
    {
      job_name = "blackbox-exporter"
      metrics_path = "/probe"
      params = {
        module = ["http_2xx"]
      }
      static_configs = [{
        targets = [
          "https://api.greenlang.io/health",
          "https://app.greenlang.io/health",
          "https://dashboard.greenlang.io/health"
        ]
      }]
      relabel_configs = [
        {
          source_labels = ["__address__"]
          target_label  = "__param_target"
        },
        {
          source_labels = ["__param_target"]
          target_label  = "instance"
        },
        {
          target_label  = "__address__"
          replacement   = "blackbox-exporter.monitoring.svc:9115"
        }
      ]
    }
  ]

  # IRSA configuration for S3 access
  oidc_provider_arn = module.eks.oidc_provider_arn
  oidc_provider_url = module.eks.oidc_provider_url

  # Feature flags - all enabled for production
  enable_grafana_operator   = false  # Use existing Grafana
  enable_node_exporter      = true
  enable_kube_state_metrics = true
  enable_blackbox_exporter  = true   # Endpoint monitoring

  # Recording rules - full set
  recording_rules_enabled = true

  # Alert rules - all enabled
  enable_critical_alerts = true
  enable_warning_alerts  = true

  # Query frontend for caching - required for production
  enable_query_frontend   = true
  query_frontend_replicas = 2

  query_frontend_cache_config = {
    type = "IN-MEMORY"
    config = {
      max_size       = "512MB"
      max_size_items = 1000
      validity       = "5m"
    }
  }

  # Query timeout configuration
  query_timeout = "2m"
  query_max_concurrency = 20

  # Security configuration
  enable_network_policies = true
  allowed_namespaces = [
    "monitoring",
    "grafana",
    "istio-system"
  ]

  # Backup configuration
  enable_backup = true
  backup_schedule = "0 2 * * *"  # Daily at 2 AM UTC
  backup_retention_days = 30

  # Tags
  tags = local.prometheus_common_tags
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms for Prometheus Stack
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "prometheus_down" {
  alarm_name          = "greenlang-prod-prometheus-down"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "up"
  namespace           = "GreenLang/Prometheus"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "Prometheus is down in production"

  dimensions = {
    Environment = "prod"
    Job         = "prometheus"
  }

  alarm_actions = [var.sns_critical_topic_arn]
  ok_actions    = [var.sns_critical_topic_arn]

  tags = local.prometheus_common_tags
}

resource "aws_cloudwatch_metric_alarm" "thanos_query_down" {
  alarm_name          = "greenlang-prod-thanos-query-down"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "up"
  namespace           = "GreenLang/Thanos"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "Thanos Query is down in production"

  dimensions = {
    Environment = "prod"
    Component   = "query"
  }

  alarm_actions = [var.sns_critical_topic_arn]
  ok_actions    = [var.sns_critical_topic_arn]

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
  description = "Thanos Query endpoint for unified queries (use this for Grafana)"
  value       = module.prometheus_stack.thanos_query_endpoint
}

output "alertmanager_endpoint" {
  description = "Alertmanager endpoint"
  value       = module.prometheus_stack.alertmanager_endpoint
}

output "pushgateway_endpoint" {
  description = "PushGateway endpoint for batch jobs"
  value       = module.prometheus_stack.pushgateway_endpoint
}

output "thanos_bucket_name" {
  description = "S3 bucket for Thanos long-term metrics storage"
  value       = module.prometheus_stack.thanos_bucket_name
}

output "thanos_bucket_arn" {
  description = "S3 bucket ARN for Thanos"
  value       = module.prometheus_stack.thanos_bucket_arn
}

output "prometheus_service_account_role_arn" {
  description = "IAM role ARN for Prometheus service account"
  value       = module.prometheus_stack.prometheus_role_arn
}

output "thanos_service_account_role_arn" {
  description = "IAM role ARN for Thanos service accounts"
  value       = module.prometheus_stack.thanos_role_arn
}

output "alertmanager_silences_endpoint" {
  description = "Alertmanager silences API endpoint"
  value       = "${module.prometheus_stack.alertmanager_endpoint}/api/v2/silences"
}
