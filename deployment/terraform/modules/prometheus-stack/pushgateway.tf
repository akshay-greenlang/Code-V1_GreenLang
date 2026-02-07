# =============================================================================
# GreenLang Prometheus Stack Module - PushGateway Deployment
# GreenLang Climate OS | OBS-001
# =============================================================================
# Deploys Prometheus PushGateway for batch job metrics:
#   - HA deployment with 2+ replicas
#   - Persistent storage for metrics retention
#   - ServiceMonitor for Prometheus scraping
#   - Network policy for security
# =============================================================================

# -----------------------------------------------------------------------------
# PushGateway Helm Release
# -----------------------------------------------------------------------------

resource "helm_release" "pushgateway" {
  count = var.enable_pushgateway ? 1 : 0

  name       = "pushgateway"
  namespace  = var.namespace
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "prometheus-pushgateway"
  version    = var.pushgateway_chart_version

  create_namespace = false
  wait             = true
  timeout          = 300 # 5 minutes
  atomic           = true
  cleanup_on_fail  = true

  # -------------------------------------------------------------------------
  # Replica Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "replicaCount"
    value = var.pushgateway_replica_count
  }

  # -------------------------------------------------------------------------
  # Persistence Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "persistentVolume.enabled"
    value = "true"
  }

  set {
    name  = "persistentVolume.size"
    value = var.pushgateway_storage_size
  }

  set {
    name  = "persistentVolume.storageClass"
    value = var.prometheus_storage_class
  }

  # -------------------------------------------------------------------------
  # Service Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "service.type"
    value = "ClusterIP"
  }

  set {
    name  = "service.port"
    value = "9091"
  }

  set {
    name  = "service.targetPort"
    value = "9091"
  }

  # -------------------------------------------------------------------------
  # Resource Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "resources.requests.cpu"
    value = "50m"
  }

  set {
    name  = "resources.requests.memory"
    value = "64Mi"
  }

  set {
    name  = "resources.limits.cpu"
    value = "200m"
  }

  set {
    name  = "resources.limits.memory"
    value = "256Mi"
  }

  # -------------------------------------------------------------------------
  # Security Context
  # -------------------------------------------------------------------------
  set {
    name  = "podSecurityContext.runAsNonRoot"
    value = "true"
  }

  set {
    name  = "podSecurityContext.runAsUser"
    value = "65534"
  }

  set {
    name  = "podSecurityContext.fsGroup"
    value = "65534"
  }

  set {
    name  = "securityContext.allowPrivilegeEscalation"
    value = "false"
  }

  set {
    name  = "securityContext.readOnlyRootFilesystem"
    value = "true"
  }

  # -------------------------------------------------------------------------
  # Pod Anti-Affinity for HA
  # -------------------------------------------------------------------------
  set {
    name  = "affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].weight"
    value = "100"
  }

  set {
    name  = "affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.labelSelector.matchLabels.app"
    value = "prometheus-pushgateway"
  }

  set {
    name  = "affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey"
    value = "kubernetes.io/hostname"
  }

  # -------------------------------------------------------------------------
  # ServiceMonitor Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "serviceMonitor.enabled"
    value = "true"
  }

  set {
    name  = "serviceMonitor.namespace"
    value = var.namespace
  }

  set {
    name  = "serviceMonitor.interval"
    value = "15s"
  }

  set {
    name  = "serviceMonitor.additionalLabels.release"
    value = "${var.project}-kube-prometheus"
  }

  # Honor labels from pushed metrics
  set {
    name  = "serviceMonitor.honorLabels"
    value = "true"
  }

  # -------------------------------------------------------------------------
  # Pod Disruption Budget
  # -------------------------------------------------------------------------
  set {
    name  = "podDisruptionBudget.enabled"
    value = "true"
  }

  set {
    name  = "podDisruptionBudget.minAvailable"
    value = "1"
  }

  # -------------------------------------------------------------------------
  # Labels and Annotations
  # -------------------------------------------------------------------------
  set {
    name  = "podLabels.app\\.kubernetes\\.io/name"
    value = "pushgateway"
  }

  set {
    name  = "podLabels.app\\.kubernetes\\.io/component"
    value = "metrics-gateway"
  }

  set {
    name  = "podLabels.app\\.kubernetes\\.io/managed-by"
    value = "terraform"
  }

  depends_on = [kubernetes_namespace.monitoring]
}

# -----------------------------------------------------------------------------
# Network Policy for PushGateway
# -----------------------------------------------------------------------------
# Restricts access to PushGateway - only allow from GreenLang namespaces.

resource "kubernetes_network_policy" "pushgateway" {
  count = var.enable_pushgateway ? 1 : 0

  metadata {
    name      = "pushgateway-network-policy"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "pushgateway"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  spec {
    pod_selector {
      match_labels = {
        "app" = "prometheus-pushgateway"
      }
    }

    policy_types = ["Ingress"]

    # Allow ingress from Prometheus (for scraping)
    ingress {
      from {
        pod_selector {
          match_labels = {
            "app.kubernetes.io/name" = "prometheus"
          }
        }
      }
      ports {
        protocol = "TCP"
        port     = "9091"
      }
    }

    # Allow ingress from GreenLang namespaces (for pushing metrics)
    dynamic "ingress" {
      for_each = var.greenlang_namespaces
      content {
        from {
          namespace_selector {
            match_labels = {
              "kubernetes.io/metadata.name" = ingress.value
            }
          }
        }
        ports {
          protocol = "TCP"
          port     = "9091"
        }
      }
    }

    # Allow ingress from batch job pods with specific label
    ingress {
      from {
        pod_selector {
          match_labels = {
            "greenlang.io/batch-job" = "true"
          }
        }
        namespace_selector {}
      }
      ports {
        protocol = "TCP"
        port     = "9091"
      }
    }
  }

  depends_on = [helm_release.pushgateway]
}

# -----------------------------------------------------------------------------
# PushGateway PrometheusRule for Health Monitoring
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "pushgateway_alerts" {
  count = var.enable_pushgateway ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PrometheusRule"
    metadata = {
      name      = "pushgateway-health-rules"
      namespace = var.namespace
      labels = {
        "app.kubernetes.io/name"       = "pushgateway"
        "app.kubernetes.io/component"  = "alerts"
        "app.kubernetes.io/managed-by" = "terraform"
        "release"                      = "${var.project}-kube-prometheus"
      }
    }
    spec = {
      groups = [
        {
          name = "pushgateway.rules"
          rules = [
            {
              alert = "PushGatewayDown"
              expr  = "up{job=\"pushgateway\"} == 0"
              for   = "5m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "PushGateway is down"
                description = "PushGateway instance {{ $labels.instance }} is unreachable for more than 5 minutes."
                runbook_url = "https://docs.greenlang.io/runbooks/pushgateway-down"
              }
            },
            {
              alert = "PushGatewayHighMetricAge"
              expr  = "(time() - push_time_seconds) > 3600"
              for   = "15m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Stale metrics in PushGateway"
                description = "Metrics from job {{ $labels.job }} have not been updated for more than 1 hour."
                runbook_url = "https://docs.greenlang.io/runbooks/pushgateway-stale-metrics"
              }
            },
            {
              alert = "BatchJobFailed"
              expr  = "gl_batch_job_duration_seconds{status=\"error\"} > 0"
              for   = "1m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Batch job failed"
                description = "Batch job {{ $labels.job_name }} failed with error status."
                runbook_url = "https://docs.greenlang.io/runbooks/batch-job-failed"
              }
            },
            {
              alert = "BatchJobStale"
              expr  = "(time() - gl_batch_job_last_success_timestamp) > 86400"
              for   = "30m"
              labels = {
                severity = "warning"
              }
              annotations = {
                summary     = "Batch job has not run recently"
                description = "Batch job {{ $labels.job_name }} has not completed successfully in over 24 hours."
                runbook_url = "https://docs.greenlang.io/runbooks/batch-job-stale"
              }
            }
          ]
        }
      ]
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}
