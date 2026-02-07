# =============================================================================
# GreenLang Prometheus Stack Module - Thanos Deployment
# GreenLang Climate OS | OBS-001
# =============================================================================
# Deploys Thanos components for long-term metrics storage and HA queries:
#   - Query: Unified query interface across Prometheus replicas and S3
#   - Query Frontend: Caching layer for queries
#   - Store Gateway: Queries historical data from S3
#   - Compactor: Downsamples and compacts metric blocks
#   - Ruler: Evaluates recording and alerting rules
# =============================================================================

# -----------------------------------------------------------------------------
# Bitnami Thanos Helm Release
# -----------------------------------------------------------------------------
# Deploys all Thanos components using the Bitnami Helm chart.

resource "helm_release" "thanos" {
  count = var.enable_thanos ? 1 : 0

  name       = "thanos"
  namespace  = var.namespace
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "thanos"
  version    = var.thanos_chart_version

  create_namespace = false
  wait             = true
  timeout          = 600 # 10 minutes
  atomic           = true
  cleanup_on_fail  = true

  # -------------------------------------------------------------------------
  # Global Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "image.registry"
    value = "quay.io"
  }

  set {
    name  = "image.repository"
    value = "thanos/thanos"
  }

  set {
    name  = "image.tag"
    value = "v0.34.1"
  }

  # Object store configuration from secret
  set {
    name  = "objstoreConfig.type"
    value = "secret"
  }

  set {
    name  = "existingObjstoreSecret"
    value = kubernetes_secret.thanos_objstore[0].metadata[0].name
  }

  # -------------------------------------------------------------------------
  # Thanos Query Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "query.enabled"
    value = "true"
  }

  set {
    name  = "query.replicaCount"
    value = var.thanos_query_replicas
  }

  # DNS discovery for Prometheus sidecars
  set {
    name  = "query.dnsDiscovery.enabled"
    value = "true"
  }

  set {
    name  = "query.dnsDiscovery.sidecarsService"
    value = "prometheus-thanos-discovery"
  }

  set {
    name  = "query.dnsDiscovery.sidecarsNamespace"
    value = var.namespace
  }

  # Store endpoints (Store Gateway)
  set {
    name  = "query.stores[0]"
    value = "dnssrv+_grpc._tcp.thanos-storegateway.${var.namespace}.svc.cluster.local"
  }

  # Query resources
  set {
    name  = "query.resources.requests.cpu"
    value = "250m"
  }

  set {
    name  = "query.resources.requests.memory"
    value = "512Mi"
  }

  set {
    name  = "query.resources.limits.cpu"
    value = "1000m"
  }

  set {
    name  = "query.resources.limits.memory"
    value = "2Gi"
  }

  # Pod anti-affinity for HA
  set {
    name  = "query.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].weight"
    value = "100"
  }

  set {
    name  = "query.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.labelSelector.matchLabels.app\\.kubernetes\\.io/component"
    value = "query"
  }

  set {
    name  = "query.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey"
    value = "kubernetes.io/hostname"
  }

  # -------------------------------------------------------------------------
  # Thanos Query Frontend Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "queryFrontend.enabled"
    value = "true"
  }

  set {
    name  = "queryFrontend.replicaCount"
    value = "2"
  }

  # In-memory caching
  set {
    name  = "queryFrontend.config"
    value = <<-EOT
      type: IN-MEMORY
      config:
        max_size: 512MB
        max_size_items: 1000
        validity: 5m
    EOT
  }

  set {
    name  = "queryFrontend.resources.requests.cpu"
    value = "100m"
  }

  set {
    name  = "queryFrontend.resources.requests.memory"
    value = "256Mi"
  }

  set {
    name  = "queryFrontend.resources.limits.cpu"
    value = "500m"
  }

  set {
    name  = "queryFrontend.resources.limits.memory"
    value = "1Gi"
  }

  # -------------------------------------------------------------------------
  # Thanos Store Gateway Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "storegateway.enabled"
    value = "true"
  }

  set {
    name  = "storegateway.replicaCount"
    value = var.thanos_store_gateway_replicas
  }

  # Persistence for local cache
  set {
    name  = "storegateway.persistence.enabled"
    value = "true"
  }

  set {
    name  = "storegateway.persistence.size"
    value = var.thanos_store_gateway_storage_size
  }

  set {
    name  = "storegateway.persistence.storageClass"
    value = var.prometheus_storage_class
  }

  # Resources
  set {
    name  = "storegateway.resources.requests.cpu"
    value = "250m"
  }

  set {
    name  = "storegateway.resources.requests.memory"
    value = "1Gi"
  }

  set {
    name  = "storegateway.resources.limits.cpu"
    value = "1000m"
  }

  set {
    name  = "storegateway.resources.limits.memory"
    value = "4Gi"
  }

  # Service Account with IRSA
  set {
    name  = "storegateway.serviceAccount.create"
    value = "true"
  }

  set {
    name  = "storegateway.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = aws_iam_role.prometheus_thanos[0].arn
  }

  # -------------------------------------------------------------------------
  # Thanos Compactor Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "compactor.enabled"
    value = "true"
  }

  # Only 1 replica allowed for Compactor to avoid conflicts
  set {
    name  = "compactor.replicaCount"
    value = var.thanos_compactor_replicas
  }

  # Retention configuration
  set {
    name  = "compactor.retentionResolutionRaw"
    value = var.thanos_retention_raw
  }

  set {
    name  = "compactor.retentionResolution5m"
    value = var.thanos_retention_5m
  }

  set {
    name  = "compactor.retentionResolution1h"
    value = var.thanos_retention_1h
  }

  # Persistence for working directory
  set {
    name  = "compactor.persistence.enabled"
    value = "true"
  }

  set {
    name  = "compactor.persistence.size"
    value = var.thanos_compactor_storage_size
  }

  set {
    name  = "compactor.persistence.storageClass"
    value = var.prometheus_storage_class
  }

  # Resources
  set {
    name  = "compactor.resources.requests.cpu"
    value = "250m"
  }

  set {
    name  = "compactor.resources.requests.memory"
    value = "1Gi"
  }

  set {
    name  = "compactor.resources.limits.cpu"
    value = "1000m"
  }

  set {
    name  = "compactor.resources.limits.memory"
    value = "4Gi"
  }

  # Service Account with IRSA
  set {
    name  = "compactor.serviceAccount.create"
    value = "true"
  }

  set {
    name  = "compactor.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = aws_iam_role.prometheus_thanos[0].arn
  }

  # -------------------------------------------------------------------------
  # Thanos Ruler Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "ruler.enabled"
    value = "true"
  }

  set {
    name  = "ruler.replicaCount"
    value = var.thanos_ruler_replicas
  }

  # Alertmanager endpoints
  set {
    name  = "ruler.alertmanagers[0]"
    value = "http://${var.project}-prometheus-alertmanager.${var.namespace}.svc:9093"
  }

  # Ruler persistence
  set {
    name  = "ruler.persistence.enabled"
    value = "true"
  }

  set {
    name  = "ruler.persistence.size"
    value = "10Gi"
  }

  set {
    name  = "ruler.persistence.storageClass"
    value = var.prometheus_storage_class
  }

  # Resources
  set {
    name  = "ruler.resources.requests.cpu"
    value = "100m"
  }

  set {
    name  = "ruler.resources.requests.memory"
    value = "256Mi"
  }

  set {
    name  = "ruler.resources.limits.cpu"
    value = "500m"
  }

  set {
    name  = "ruler.resources.limits.memory"
    value = "1Gi"
  }

  # Service Account with IRSA
  set {
    name  = "ruler.serviceAccount.create"
    value = "true"
  }

  set {
    name  = "ruler.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = aws_iam_role.prometheus_thanos[0].arn
  }

  # -------------------------------------------------------------------------
  # Bucket Web (UI for S3 bucket inspection)
  # -------------------------------------------------------------------------
  set {
    name  = "bucketweb.enabled"
    value = "false"
  }

  # -------------------------------------------------------------------------
  # Receive (not used - we use Sidecar pattern)
  # -------------------------------------------------------------------------
  set {
    name  = "receive.enabled"
    value = "false"
  }

  # -------------------------------------------------------------------------
  # Metrics and ServiceMonitor
  # -------------------------------------------------------------------------
  set {
    name  = "metrics.enabled"
    value = "true"
  }

  set {
    name  = "metrics.serviceMonitor.enabled"
    value = "true"
  }

  set {
    name  = "metrics.serviceMonitor.labels.release"
    value = "${var.project}-kube-prometheus"
  }

  depends_on = [
    kubernetes_namespace.monitoring,
    kubernetes_secret.thanos_objstore,
    helm_release.kube_prometheus_stack,
    kubernetes_service.thanos_sidecar_discovery
  ]
}

# -----------------------------------------------------------------------------
# Network Policy for Thanos
# -----------------------------------------------------------------------------
# Restricts network access to Thanos components.

resource "kubernetes_network_policy" "thanos" {
  count = var.enable_thanos ? 1 : 0

  metadata {
    name      = "thanos-network-policy"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "thanos"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  spec {
    pod_selector {
      match_labels = {
        "app.kubernetes.io/name" = "thanos"
      }
    }

    policy_types = ["Ingress", "Egress"]

    # Allow ingress from monitoring namespace
    ingress {
      from {
        namespace_selector {
          match_labels = {
            "kubernetes.io/metadata.name" = var.namespace
          }
        }
      }
      ports {
        protocol = "TCP"
        port     = "10901" # gRPC
      }
      ports {
        protocol = "TCP"
        port     = "10902" # HTTP
      }
    }

    # Allow ingress from Grafana
    ingress {
      from {
        pod_selector {
          match_labels = {
            "app.kubernetes.io/name" = "grafana"
          }
        }
      }
      ports {
        protocol = "TCP"
        port     = "9090"
      }
    }

    # Allow all egress (S3, DNS, other Thanos components)
    egress {
      to {}
    }
  }

  depends_on = [helm_release.thanos]
}
