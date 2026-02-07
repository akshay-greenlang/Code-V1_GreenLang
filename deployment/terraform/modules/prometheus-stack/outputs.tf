# =============================================================================
# GreenLang Prometheus Stack Module - Outputs
# GreenLang Climate OS | OBS-001
# =============================================================================
# Output values for integration with other modules and services.
# =============================================================================

# -----------------------------------------------------------------------------
# Prometheus Outputs
# -----------------------------------------------------------------------------

output "prometheus_endpoint" {
  description = "Prometheus server endpoint (ClusterIP service URL)"
  value       = "http://${var.project}-prometheus-prometheus.${var.namespace}.svc:9090"
}

output "prometheus_external_url" {
  description = "Prometheus server external URL (for ingress configuration)"
  value       = "https://prometheus.${var.environment}.greenlang.io"
}

output "prometheus_replica_count" {
  description = "Number of Prometheus replicas deployed"
  value       = var.prometheus_replica_count
}

output "prometheus_service_account" {
  description = "Prometheus ServiceAccount name"
  value       = "prometheus-${var.project}-kube-prometheus-prometheus"
}

output "prometheus_iam_role_arn" {
  description = "IAM role ARN for Prometheus ServiceAccount (IRSA)"
  value       = local.prometheus_iam_role_arn
}

# -----------------------------------------------------------------------------
# Thanos Outputs
# -----------------------------------------------------------------------------

output "thanos_query_endpoint" {
  description = "Thanos Query endpoint for unified queries"
  value       = var.enable_thanos ? "http://thanos-query.${var.namespace}.svc:9090" : null
}

output "thanos_query_frontend_endpoint" {
  description = "Thanos Query Frontend endpoint (with caching)"
  value       = var.enable_thanos ? "http://thanos-query-frontend.${var.namespace}.svc:9090" : null
}

output "thanos_store_gateway_endpoint" {
  description = "Thanos Store Gateway gRPC endpoint"
  value       = var.enable_thanos ? "dnssrv+_grpc._tcp.thanos-storegateway.${var.namespace}.svc.cluster.local" : null
}

output "thanos_bucket_name" {
  description = "S3 bucket name for Thanos metrics storage"
  value       = var.enable_thanos ? local.thanos_bucket_name : null
}

output "thanos_bucket_arn" {
  description = "S3 bucket ARN for Thanos metrics storage"
  value       = var.enable_thanos ? aws_s3_bucket.thanos_metrics[0].arn : null
}

output "thanos_enabled" {
  description = "Whether Thanos is enabled"
  value       = var.enable_thanos
}

# -----------------------------------------------------------------------------
# Alertmanager Outputs
# -----------------------------------------------------------------------------

output "alertmanager_endpoint" {
  description = "Alertmanager endpoint"
  value       = "http://${var.project}-prometheus-alertmanager.${var.namespace}.svc:9093"
}

output "alertmanager_external_url" {
  description = "Alertmanager external URL (for ingress configuration)"
  value       = "https://alertmanager.${var.environment}.greenlang.io"
}

output "alertmanager_replica_count" {
  description = "Number of Alertmanager replicas deployed"
  value       = var.alertmanager_replica_count
}

# -----------------------------------------------------------------------------
# PushGateway Outputs
# -----------------------------------------------------------------------------

output "pushgateway_endpoint" {
  description = "PushGateway endpoint for batch job metrics"
  value       = var.enable_pushgateway ? "http://pushgateway.${var.namespace}.svc:9091" : null
}

output "pushgateway_enabled" {
  description = "Whether PushGateway is enabled"
  value       = var.enable_pushgateway
}

# -----------------------------------------------------------------------------
# Grafana Outputs
# -----------------------------------------------------------------------------

output "grafana_endpoint" {
  description = "Grafana endpoint"
  value       = "http://${var.project}-prometheus-grafana.${var.namespace}.svc:80"
}

output "grafana_external_url" {
  description = "Grafana external URL (for ingress configuration)"
  value       = "https://grafana.${var.environment}.greenlang.io"
}

# -----------------------------------------------------------------------------
# Namespace and Infrastructure Outputs
# -----------------------------------------------------------------------------

output "namespace" {
  description = "Kubernetes namespace where Prometheus stack is deployed"
  value       = var.namespace
}

output "namespace_id" {
  description = "Kubernetes namespace resource ID"
  value       = kubernetes_namespace.monitoring.id
}

# -----------------------------------------------------------------------------
# Helm Release Outputs
# -----------------------------------------------------------------------------

output "prometheus_helm_release_name" {
  description = "Helm release name for kube-prometheus-stack"
  value       = helm_release.kube_prometheus_stack.name
}

output "prometheus_helm_release_version" {
  description = "Helm chart version for kube-prometheus-stack"
  value       = helm_release.kube_prometheus_stack.version
}

output "thanos_helm_release_name" {
  description = "Helm release name for Thanos"
  value       = var.enable_thanos ? helm_release.thanos[0].name : null
}

output "thanos_helm_release_version" {
  description = "Helm chart version for Thanos"
  value       = var.enable_thanos ? helm_release.thanos[0].version : null
}

output "pushgateway_helm_release_name" {
  description = "Helm release name for PushGateway"
  value       = var.enable_pushgateway ? helm_release.pushgateway[0].name : null
}

# -----------------------------------------------------------------------------
# Configuration Outputs
# -----------------------------------------------------------------------------

output "scrape_config_secret_name" {
  description = "Name of the secret containing additional scrape configs"
  value       = length(var.additional_scrape_configs) > 0 ? kubernetes_secret.additional_scrape_configs[0].metadata[0].name : null
}

output "objstore_secret_name" {
  description = "Name of the secret containing Thanos object store configuration"
  value       = var.enable_thanos ? kubernetes_secret.thanos_objstore[0].metadata[0].name : null
}

# -----------------------------------------------------------------------------
# Integration Outputs (for other modules)
# -----------------------------------------------------------------------------

output "datasource_prometheus_url" {
  description = "Prometheus datasource URL for Grafana configuration"
  value       = "http://${var.project}-prometheus-prometheus.${var.namespace}.svc:9090"
}

output "datasource_thanos_url" {
  description = "Thanos datasource URL for Grafana configuration (long-term queries)"
  value       = var.enable_thanos ? "http://thanos-query.${var.namespace}.svc:9090" : null
}

output "datasource_alertmanager_url" {
  description = "Alertmanager datasource URL for Grafana configuration"
  value       = "http://${var.project}-prometheus-alertmanager.${var.namespace}.svc:9093"
}

# -----------------------------------------------------------------------------
# Service Discovery Outputs
# -----------------------------------------------------------------------------

output "thanos_sidecar_discovery_service" {
  description = "Headless service for Thanos sidecar discovery"
  value       = var.enable_thanos ? "prometheus-thanos-discovery.${var.namespace}.svc" : null
}

output "service_monitor_labels" {
  description = "Labels to add to ServiceMonitors for Prometheus discovery"
  value = {
    "release" = "${var.project}-kube-prometheus"
  }
}
