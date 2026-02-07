# =============================================================================
# GreenLang Prometheus Stack Module - ServiceMonitors
# GreenLang Climate OS | OBS-001
# =============================================================================
# Creates ServiceMonitor and PodMonitor CRDs for Prometheus scraping:
#   - GreenLang API ServiceMonitor
#   - Agent Factory ServiceMonitor
#   - Infrastructure components (PostgreSQL, Redis, Kong)
#   - PodMonitor for annotation-based discovery
# =============================================================================

# -----------------------------------------------------------------------------
# GreenLang API ServiceMonitor
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "servicemonitor_greenlang_api" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "greenlang-api"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "greenlang-api"
        "app.kubernetes.io/component"  = "monitoring"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app" = "greenlang-api"
        }
      }
      namespaceSelector = {
        matchNames = ["greenlang"]
      }
      endpoints = [
        {
          port          = "metrics"
          interval      = "15s"
          path          = "/metrics"
          scheme        = "http"
          honorLabels   = true
          relabelings = [
            {
              sourceLabels = ["__meta_kubernetes_pod_label_app"]
              targetLabel  = "service"
            },
            {
              sourceLabels = ["__meta_kubernetes_pod_name"]
              targetLabel  = "pod"
            },
            {
              sourceLabels = ["__meta_kubernetes_namespace"]
              targetLabel  = "namespace"
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}

# -----------------------------------------------------------------------------
# Agent Factory ServiceMonitor
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "servicemonitor_agent_factory" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "agent-factory"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "agent-factory"
        "app.kubernetes.io/component"  = "monitoring"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/component" = "agent-factory"
        }
      }
      namespaceSelector = {
        matchNames = ["greenlang", "gl-agents"]
      }
      endpoints = [
        {
          port     = "metrics"
          interval = "15s"
          path     = "/metrics"
          metricRelabelings = [
            {
              sourceLabels = ["__name__"]
              regex        = "gl_agent_.*"
              action       = "keep"
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}

# -----------------------------------------------------------------------------
# PostgreSQL ServiceMonitor
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "servicemonitor_postgresql" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "postgresql"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "postgresql"
        "app.kubernetes.io/component"  = "monitoring"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app" = "postgres-exporter"
        }
      }
      namespaceSelector = {
        matchNames = ["greenlang", "database"]
      }
      endpoints = [
        {
          port     = "metrics"
          interval = "30s"
          path     = "/metrics"
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}

# -----------------------------------------------------------------------------
# Redis ServiceMonitor
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "servicemonitor_redis" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "redis"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "redis"
        "app.kubernetes.io/component"  = "monitoring"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app" = "redis-exporter"
        }
      }
      namespaceSelector = {
        matchNames = ["greenlang", "cache"]
      }
      endpoints = [
        {
          port     = "metrics"
          interval = "15s"
          path     = "/metrics"
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}

# -----------------------------------------------------------------------------
# Kong API Gateway ServiceMonitor
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "servicemonitor_kong" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "kong-gateway"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "kong"
        "app.kubernetes.io/component"  = "monitoring"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/name" = "kong"
        }
      }
      namespaceSelector = {
        matchNames = ["kong"]
      }
      endpoints = [
        {
          port     = "metrics"
          interval = "15s"
          path     = "/metrics"
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}

# -----------------------------------------------------------------------------
# PodMonitor for Annotation-Based Discovery
# -----------------------------------------------------------------------------
# Discovers pods with prometheus.io/scrape=true annotation across all
# GreenLang namespaces.

resource "kubernetes_manifest" "podmonitor_greenlang" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PodMonitor"
    metadata = {
      name      = "greenlang-pods"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "greenlang-pods"
        "app.kubernetes.io/component"  = "monitoring"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "prometheus.io/scrape" = "true"
        }
      }
      namespaceSelector = {
        matchNames = var.greenlang_namespaces
      }
      podMetricsEndpoints = [
        {
          port     = "metrics"
          interval = "15s"
          path     = "/metrics"
          relabelings = [
            {
              sourceLabels = ["__meta_kubernetes_pod_label_app"]
              targetLabel  = "app"
            },
            {
              sourceLabels = ["__meta_kubernetes_pod_label_version"]
              targetLabel  = "version"
            },
            {
              sourceLabels = ["__meta_kubernetes_namespace"]
              targetLabel  = "namespace"
            },
            {
              sourceLabels = ["__meta_kubernetes_pod_name"]
              targetLabel  = "pod"
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}

# -----------------------------------------------------------------------------
# PodMonitor for Agent Pods
# -----------------------------------------------------------------------------
# Specifically monitors agent pods across agent namespaces.

resource "kubernetes_manifest" "podmonitor_agents" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PodMonitor"
    metadata = {
      name      = "greenlang-agents"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "greenlang-agents"
        "app.kubernetes.io/component"  = "monitoring"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/part-of" = "greenlang-agents"
        }
      }
      namespaceSelector = {
        matchNames = ["gl-agents", "gl-fuel", "gl-cbam", "gl-building"]
      }
      podMetricsEndpoints = [
        {
          port     = "metrics"
          interval = "15s"
          path     = "/metrics"
          honorLabels = true
          metricRelabelings = [
            {
              sourceLabels = ["__name__"]
              regex        = "gl_.*"
              action       = "keep"
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}

# -----------------------------------------------------------------------------
# Prometheus Health Rules
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "prometheus_health_rules" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PrometheusRule"
    metadata = {
      name      = "prometheus-health-rules"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "prometheus"
        "app.kubernetes.io/component"  = "alerts"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      groups = [
        {
          name = "prometheus.rules"
          rules = [
            {
              alert = "PrometheusTargetMissing"
              expr  = "up == 0"
              for   = "5m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "Prometheus target missing (instance {{ $labels.instance }})"
                description = "A Prometheus target is down for more than 5 minutes."
                runbook_url = "https://docs.greenlang.io/runbooks/prometheus-target-down"
              }
            },
            {
              alert = "PrometheusConfigReloadFailed"
              expr  = "prometheus_config_last_reload_successful != 1"
              for   = "1m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "Prometheus config reload failed"
                description = "Prometheus configuration reload has failed. Check for syntax errors."
                runbook_url = "https://docs.greenlang.io/runbooks/prometheus-config-reload-failed"
              }
            },
            {
              alert = "PrometheusTSDBCompactionsFailed"
              expr  = "increase(prometheus_tsdb_compactions_failed_total[1h]) > 0"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Prometheus TSDB compactions failing"
                description = "Prometheus TSDB compactions are failing. This may indicate storage issues."
                runbook_url = "https://docs.greenlang.io/runbooks/prometheus-tsdb-compaction-failed"
              }
            },
            {
              alert = "PrometheusRuleEvaluationFailures"
              expr  = "increase(prometheus_rule_evaluation_failures_total[5m]) > 0"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Prometheus rule evaluation failures"
                description = "Prometheus is failing to evaluate alerting/recording rules."
                runbook_url = "https://docs.greenlang.io/runbooks/prometheus-rule-evaluation-failed"
              }
            },
            {
              alert = "PrometheusStorageAlmostFull"
              expr  = "(prometheus_tsdb_storage_blocks_bytes / (prometheus_tsdb_storage_blocks_bytes + prometheus_tsdb_head_chunks_storage_size_bytes)) > 0.8"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Prometheus storage almost full (>80%)"
                description = "Prometheus storage is more than 80% full. Consider increasing storage or reducing retention."
                runbook_url = "https://docs.greenlang.io/runbooks/prometheus-storage-full"
              }
            },
            {
              alert = "PrometheusHighMemoryUsage"
              expr  = "process_resident_memory_bytes{job=\"prometheus\"} / container_spec_memory_limit_bytes{container=\"prometheus\"} > 0.8"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Prometheus memory usage >80%"
                description = "Prometheus is using more than 80% of its memory limit."
                runbook_url = "https://docs.greenlang.io/runbooks/prometheus-high-memory"
              }
            },
            {
              alert = "PrometheusHighCardinality"
              expr  = "prometheus_tsdb_head_series > 1000000"
              for   = "15m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Prometheus high cardinality (>1M series)"
                description = "Prometheus has more than 1 million time series. This may cause performance issues."
                runbook_url = "https://docs.greenlang.io/runbooks/prometheus-high-cardinality"
              }
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}

# -----------------------------------------------------------------------------
# Thanos Health Rules
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "thanos_health_rules" {
  count = var.enable_thanos ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PrometheusRule"
    metadata = {
      name      = "thanos-health-rules"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "thanos"
        "app.kubernetes.io/component"  = "alerts"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      groups = [
        {
          name = "thanos.rules"
          rules = [
            {
              alert = "ThanosCompactorMultipleRunning"
              expr  = "sum(up{job=~\".*thanos-compactor.*\"}) > 1"
              for   = "5m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "Multiple Thanos Compactors running"
                description = "Multiple Thanos Compactor instances are running simultaneously. This can cause data corruption."
                runbook_url = "https://docs.greenlang.io/runbooks/thanos-compactor-multiple"
              }
            },
            {
              alert = "ThanosQueryHighDNSFailures"
              expr  = "rate(thanos_query_store_apis_dns_failures_total[5m]) > 0.5"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Thanos Query DNS failures"
                description = "Thanos Query is experiencing DNS failures when discovering stores."
                runbook_url = "https://docs.greenlang.io/runbooks/thanos-query-dns-failures"
              }
            },
            {
              alert = "ThanosStoreGatewayBucketOperationsFailed"
              expr  = "rate(thanos_objstore_bucket_operation_failures_total[5m]) > 0"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Thanos Store Gateway bucket operations failing"
                description = "Thanos Store Gateway is experiencing S3 bucket operation failures."
                runbook_url = "https://docs.greenlang.io/runbooks/thanos-store-gateway-bucket-failures"
              }
            },
            {
              alert = "ThanosCompactorHalted"
              expr  = "thanos_compact_halted == 1"
              for   = "5m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "Thanos Compactor has halted"
                description = "Thanos Compactor has halted due to an error. Manual intervention may be required."
                runbook_url = "https://docs.greenlang.io/runbooks/thanos-compactor-halted"
              }
            },
            {
              alert = "ThanosSidecarPrometheusDown"
              expr  = "thanos_sidecar_prometheus_up != 1"
              for   = "5m"
              labels = {
                severity = "critical"
              }
              annotations = {
                summary     = "Thanos Sidecar cannot reach Prometheus"
                description = "Thanos Sidecar cannot connect to its Prometheus instance."
                runbook_url = "https://docs.greenlang.io/runbooks/thanos-sidecar-prometheus-down"
              }
            },
            {
              alert = "ThanosQueryHighLatency"
              expr  = "histogram_quantile(0.99, rate(thanos_query_duration_seconds_bucket[5m])) > 30"
              for   = "10m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Thanos Query high latency"
                description = "Thanos Query P99 latency is above 30 seconds."
                runbook_url = "https://docs.greenlang.io/runbooks/thanos-query-high-latency"
              }
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}
