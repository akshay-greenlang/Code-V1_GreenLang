# =============================================================================
# GreenLang Prometheus Stack Module - Prometheus Deployment
# GreenLang Climate OS | OBS-001
# =============================================================================
# Deploys kube-prometheus-stack Helm chart with:
#   - HA Prometheus (2+ replicas)
#   - Thanos Sidecar for long-term storage
#   - External labels for multi-cluster identification
#   - ServiceMonitor/PodMonitor auto-discovery
#   - Custom scrape configurations for GreenLang services
# =============================================================================

# -----------------------------------------------------------------------------
# Thanos Object Store Secret
# -----------------------------------------------------------------------------
# Secret containing S3 configuration for Thanos Sidecar.

resource "kubernetes_secret" "thanos_objstore" {
  count = var.enable_thanos ? 1 : 0

  metadata {
    name      = "thanos-objstore-secret"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "thanos"
      "app.kubernetes.io/component"  = "objstore"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  data = {
    "objstore.yml" = yamlencode({
      type = "S3"
      config = {
        bucket   = local.thanos_bucket_name
        endpoint = "s3.${var.aws_region}.amazonaws.com"
        region   = var.aws_region
        # Credentials are provided via IRSA - no static keys needed
      }
    })
  }

  type = "Opaque"

  depends_on = [kubernetes_namespace.monitoring]
}

# -----------------------------------------------------------------------------
# kube-prometheus-stack Helm Release
# -----------------------------------------------------------------------------
# Main Prometheus deployment including Prometheus Operator, Prometheus,
# Alertmanager, Grafana, kube-state-metrics, and node-exporter.

resource "helm_release" "kube_prometheus_stack" {
  name       = "${var.project}-kube-prometheus"
  namespace  = var.namespace
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = var.kube_prometheus_stack_chart_version

  create_namespace = false
  wait             = true
  timeout          = 900 # 15 minutes
  atomic           = true
  cleanup_on_fail  = true

  # -------------------------------------------------------------------------
  # Global Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "fullnameOverride"
    value = "${var.project}-prometheus"
  }

  # -------------------------------------------------------------------------
  # Prometheus Server Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "prometheus.enabled"
    value = "true"
  }

  set {
    name  = "prometheus.prometheusSpec.replicas"
    value = var.prometheus_replica_count
  }

  # Retention configuration
  set {
    name  = "prometheus.prometheusSpec.retention"
    value = "${var.prometheus_retention_days}d"
  }

  set {
    name  = "prometheus.prometheusSpec.retentionSize"
    value = var.prometheus_storage_size
  }

  # Storage configuration
  set {
    name  = "prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName"
    value = var.prometheus_storage_class
  }

  set {
    name  = "prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.accessModes[0]"
    value = "ReadWriteOnce"
  }

  set {
    name  = "prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage"
    value = var.prometheus_storage_size
  }

  # Resource limits
  set {
    name  = "prometheus.prometheusSpec.resources.requests.cpu"
    value = var.prometheus_resources.requests.cpu
  }

  set {
    name  = "prometheus.prometheusSpec.resources.requests.memory"
    value = var.prometheus_resources.requests.memory
  }

  set {
    name  = "prometheus.prometheusSpec.resources.limits.cpu"
    value = var.prometheus_resources.limits.cpu
  }

  set {
    name  = "prometheus.prometheusSpec.resources.limits.memory"
    value = var.prometheus_resources.limits.memory
  }

  # External labels for multi-cluster identification
  set {
    name  = "prometheus.prometheusSpec.externalLabels.cluster"
    value = "greenlang-${var.environment}"
  }

  set {
    name  = "prometheus.prometheusSpec.externalLabels.region"
    value = var.aws_region
  }

  # ServiceMonitor/PodMonitor discovery - use all monitors regardless of release label
  set {
    name  = "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues"
    value = "false"
  }

  set {
    name  = "prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues"
    value = "false"
  }

  set {
    name  = "prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues"
    value = "false"
  }

  # Pod anti-affinity for HA
  set {
    name  = "prometheus.prometheusSpec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchLabels.app\\.kubernetes\\.io/name"
    value = "prometheus"
  }

  set {
    name  = "prometheus.prometheusSpec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey"
    value = "kubernetes.io/hostname"
  }

  # Service Account with IRSA annotation
  set {
    name  = "prometheus.serviceAccount.create"
    value = "true"
  }

  set {
    name  = "prometheus.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = var.create_iam_role ? (var.enable_thanos ? aws_iam_role.prometheus_thanos[0].arn : aws_iam_role.prometheus_only[0].arn) : var.existing_iam_role_arn
  }

  # -------------------------------------------------------------------------
  # Thanos Sidecar Configuration (conditional)
  # -------------------------------------------------------------------------
  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "prometheus.prometheusSpec.thanos.image"
      value = "quay.io/thanos/thanos:v0.34.1"
    }
  }

  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "prometheus.prometheusSpec.thanos.objectStorageConfig.existingSecret.name"
      value = kubernetes_secret.thanos_objstore[0].metadata[0].name
    }
  }

  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "prometheus.prometheusSpec.thanos.objectStorageConfig.existingSecret.key"
      value = "objstore.yml"
    }
  }

  # Thanos sidecar resources
  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "prometheus.prometheusSpec.thanos.resources.requests.cpu"
      value = "100m"
    }
  }

  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "prometheus.prometheusSpec.thanos.resources.requests.memory"
      value = "256Mi"
    }
  }

  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "prometheus.prometheusSpec.thanos.resources.limits.cpu"
      value = "500m"
    }
  }

  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "prometheus.prometheusSpec.thanos.resources.limits.memory"
      value = "1Gi"
    }
  }

  # -------------------------------------------------------------------------
  # Alertmanager Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "alertmanager.enabled"
    value = "true"
  }

  set {
    name  = "alertmanager.alertmanagerSpec.replicas"
    value = var.alertmanager_replica_count
  }

  set {
    name  = "alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.storageClassName"
    value = var.prometheus_storage_class
  }

  set {
    name  = "alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.resources.requests.storage"
    value = var.alertmanager_storage_size
  }

  set {
    name  = "alertmanager.alertmanagerSpec.resources.requests.cpu"
    value = "100m"
  }

  set {
    name  = "alertmanager.alertmanagerSpec.resources.requests.memory"
    value = "256Mi"
  }

  set {
    name  = "alertmanager.alertmanagerSpec.resources.limits.cpu"
    value = "500m"
  }

  set {
    name  = "alertmanager.alertmanagerSpec.resources.limits.memory"
    value = "512Mi"
  }

  # -------------------------------------------------------------------------
  # Grafana Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "grafana.enabled"
    value = "true"
  }

  set {
    name  = "grafana.persistence.enabled"
    value = "true"
  }

  set {
    name  = "grafana.persistence.size"
    value = "10Gi"
  }

  set {
    name  = "grafana.persistence.storageClassName"
    value = var.prometheus_storage_class
  }

  # Grafana datasources
  set {
    name  = "grafana.additionalDataSources[0].name"
    value = "Prometheus"
  }

  set {
    name  = "grafana.additionalDataSources[0].type"
    value = "prometheus"
  }

  set {
    name  = "grafana.additionalDataSources[0].url"
    value = "http://${var.project}-prometheus-prometheus.${var.namespace}.svc:9090"
  }

  set {
    name  = "grafana.additionalDataSources[0].isDefault"
    value = "true"
  }

  # Thanos datasource (conditional)
  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "grafana.additionalDataSources[1].name"
      value = "Thanos"
    }
  }

  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "grafana.additionalDataSources[1].type"
      value = "prometheus"
    }
  }

  dynamic "set" {
    for_each = var.enable_thanos ? [1] : []
    content {
      name  = "grafana.additionalDataSources[1].url"
      value = "http://thanos-query.${var.namespace}.svc:9090"
    }
  }

  # Alertmanager datasource
  set {
    name  = "grafana.additionalDataSources[2].name"
    value = "Alertmanager"
  }

  set {
    name  = "grafana.additionalDataSources[2].type"
    value = "alertmanager"
  }

  set {
    name  = "grafana.additionalDataSources[2].url"
    value = "http://${var.project}-prometheus-alertmanager.${var.namespace}.svc:9093"
  }

  # -------------------------------------------------------------------------
  # kube-state-metrics Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "kubeStateMetrics.enabled"
    value = "true"
  }

  # -------------------------------------------------------------------------
  # node-exporter Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "nodeExporter.enabled"
    value = "true"
  }

  # -------------------------------------------------------------------------
  # Prometheus Operator Configuration
  # -------------------------------------------------------------------------
  set {
    name  = "prometheusOperator.enabled"
    value = "true"
  }

  set {
    name  = "prometheusOperator.admissionWebhooks.enabled"
    value = "true"
  }

  set {
    name  = "prometheusOperator.admissionWebhooks.patch.enabled"
    value = "true"
  }

  depends_on = [
    kubernetes_namespace.monitoring,
    kubernetes_secret.thanos_objstore
  ]
}

# -----------------------------------------------------------------------------
# Thanos Discovery Service
# -----------------------------------------------------------------------------
# Headless service for Thanos Query to discover Prometheus sidecars.

resource "kubernetes_service" "thanos_sidecar_discovery" {
  count = var.enable_thanos ? 1 : 0

  metadata {
    name      = "prometheus-thanos-discovery"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "prometheus"
      "app.kubernetes.io/component"  = "thanos-sidecar"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  spec {
    type       = "ClusterIP"
    cluster_ip = "None" # Headless service

    port {
      name        = "grpc"
      port        = 10901
      target_port = 10901
      protocol    = "TCP"
    }

    selector = {
      "app.kubernetes.io/name" = "prometheus"
    }
  }

  depends_on = [helm_release.kube_prometheus_stack]
}
