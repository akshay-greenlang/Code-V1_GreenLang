# =============================================================================
# GreenLang Prometheus Stack Module - Main
# GreenLang Climate OS | OBS-001
# =============================================================================
# Main module file containing:
#   - Local variables
#   - Data sources for EKS and OIDC
#   - Kubernetes namespace
#   - Additional scrape configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------

# Current AWS region (if not explicitly provided)
data "aws_region" "current" {}

# Current AWS account ID
data "aws_caller_identity" "current" {}

# EKS Cluster data (for OIDC configuration)
data "aws_eks_cluster" "cluster" {
  name = var.cluster_name
}

# EKS Cluster Auth (for Kubernetes provider)
data "aws_eks_cluster_auth" "cluster" {
  name = var.cluster_name
}

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------

locals {
  # Thanos bucket name
  thanos_bucket_name = var.thanos_bucket_name != "" ? var.thanos_bucket_name : "${var.project}-thanos-metrics-${var.environment}"

  # OIDC provider (extract from EKS cluster if not provided)
  oidc_provider = var.oidc_provider != "" ? var.oidc_provider : replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")

  # OIDC provider ARN
  oidc_provider_arn = var.oidc_provider_arn != "" ? var.oidc_provider_arn : "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${local.oidc_provider}"

  # AWS Region
  aws_region = var.aws_region != "" ? var.aws_region : data.aws_region.current.name

  # Common labels for all resources
  common_labels = {
    "app.kubernetes.io/part-of"    = "greenlang"
    "app.kubernetes.io/managed-by" = "terraform"
    "greenlang.io/obs"             = "001"
    "greenlang.io/environment"     = var.environment
  }

  # IAM role ARN for service accounts
  prometheus_iam_role_arn = var.create_iam_role ? (
    var.enable_thanos ? aws_iam_role.prometheus_thanos[0].arn : aws_iam_role.prometheus_only[0].arn
  ) : var.existing_iam_role_arn
}

# -----------------------------------------------------------------------------
# Kubernetes Namespace
# -----------------------------------------------------------------------------

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = var.namespace

    labels = merge(local.common_labels, {
      "kubernetes.io/metadata.name"  = var.namespace
      "app.kubernetes.io/name"       = "monitoring"
    })

    annotations = {
      "description" = "Prometheus monitoring stack for GreenLang Climate OS"
    }
  }
}

# -----------------------------------------------------------------------------
# Additional Scrape Configs Secret
# -----------------------------------------------------------------------------
# Contains additional scrape configurations for custom endpoints.

resource "kubernetes_secret" "additional_scrape_configs" {
  count = length(var.additional_scrape_configs) > 0 ? 1 : 0

  metadata {
    name      = "prometheus-additional-scrape-configs"
    namespace = var.namespace
    labels = merge(local.common_labels, {
      "app.kubernetes.io/name"      = "prometheus"
      "app.kubernetes.io/component" = "scrape-config"
    })
  }

  data = {
    "additional-scrape-configs.yaml" = yamlencode(var.additional_scrape_configs)
  }

  type = "Opaque"

  depends_on = [kubernetes_namespace.monitoring]
}

# -----------------------------------------------------------------------------
# GreenLang Agents Scrape Config (built-in)
# -----------------------------------------------------------------------------
# Default scrape configuration for GreenLang agent pods.

resource "kubernetes_secret" "greenlang_scrape_config" {
  metadata {
    name      = "prometheus-greenlang-scrape-config"
    namespace = var.namespace
    labels = merge(local.common_labels, {
      "app.kubernetes.io/name"      = "prometheus"
      "app.kubernetes.io/component" = "scrape-config"
    })
  }

  data = {
    "greenlang-scrape-config.yaml" = yamlencode([
      {
        job_name = "greenlang-agents"
        kubernetes_sd_configs = [
          {
            role = "pod"
            namespaces = {
              names = var.greenlang_namespaces
            }
          }
        ]
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
          },
          {
            source_labels = ["__address__", "__meta_kubernetes_pod_annotation_prometheus_io_port"]
            action        = "replace"
            regex         = "([^:]+)(?::\\d+)?;(\\d+)"
            replacement   = "$1:$2"
            target_label  = "__address__"
          },
          {
            source_labels = ["__meta_kubernetes_namespace"]
            action        = "replace"
            target_label  = "namespace"
          },
          {
            source_labels = ["__meta_kubernetes_pod_name"]
            action        = "replace"
            target_label  = "pod"
          },
          {
            source_labels = ["__meta_kubernetes_pod_label_app"]
            action        = "replace"
            target_label  = "app"
          },
          {
            source_labels = ["__meta_kubernetes_pod_label_version"]
            action        = "replace"
            target_label  = "version"
          }
        ]
      },
      {
        job_name = "greenlang-services"
        kubernetes_sd_configs = [
          {
            role = "endpoints"
            namespaces = {
              names = var.greenlang_namespaces
            }
          }
        ]
        relabel_configs = [
          {
            source_labels = ["__meta_kubernetes_service_annotation_prometheus_io_scrape"]
            action        = "keep"
            regex         = "true"
          },
          {
            source_labels = ["__meta_kubernetes_service_annotation_prometheus_io_path"]
            action        = "replace"
            target_label  = "__metrics_path__"
            regex         = "(.+)"
          },
          {
            source_labels = ["__address__", "__meta_kubernetes_service_annotation_prometheus_io_port"]
            action        = "replace"
            regex         = "([^:]+)(?::\\d+)?;(\\d+)"
            replacement   = "$1:$2"
            target_label  = "__address__"
          },
          {
            source_labels = ["__meta_kubernetes_namespace"]
            action        = "replace"
            target_label  = "namespace"
          },
          {
            source_labels = ["__meta_kubernetes_service_name"]
            action        = "replace"
            target_label  = "service"
          }
        ]
      }
    ])
  }

  type = "Opaque"

  depends_on = [kubernetes_namespace.monitoring]
}

# -----------------------------------------------------------------------------
# Resource Quota for Monitoring Namespace
# -----------------------------------------------------------------------------

resource "kubernetes_resource_quota" "monitoring" {
  metadata {
    name      = "monitoring-quota"
    namespace = var.namespace
    labels    = local.common_labels
  }

  spec {
    hard = {
      "requests.cpu"    = var.environment == "prod" ? "20" : "10"
      "requests.memory" = var.environment == "prod" ? "40Gi" : "20Gi"
      "limits.cpu"      = var.environment == "prod" ? "40" : "20"
      "limits.memory"   = var.environment == "prod" ? "80Gi" : "40Gi"
      "pods"            = var.environment == "prod" ? "50" : "30"
    }
  }

  depends_on = [kubernetes_namespace.monitoring]
}

# -----------------------------------------------------------------------------
# Limit Range for Monitoring Namespace
# -----------------------------------------------------------------------------

resource "kubernetes_limit_range" "monitoring" {
  metadata {
    name      = "monitoring-limits"
    namespace = var.namespace
    labels    = local.common_labels
  }

  spec {
    limit {
      type = "Container"
      default = {
        cpu    = "500m"
        memory = "512Mi"
      }
      default_request = {
        cpu    = "100m"
        memory = "128Mi"
      }
      max = {
        cpu    = "8000m"
        memory = "16Gi"
      }
      min = {
        cpu    = "50m"
        memory = "64Mi"
      }
    }
  }

  depends_on = [kubernetes_namespace.monitoring]
}

# -----------------------------------------------------------------------------
# Network Policy for Monitoring Namespace
# -----------------------------------------------------------------------------

resource "kubernetes_network_policy" "monitoring_default" {
  metadata {
    name      = "monitoring-default"
    namespace = var.namespace
    labels    = local.common_labels
  }

  spec {
    pod_selector {}

    policy_types = ["Ingress", "Egress"]

    # Allow all ingress within the monitoring namespace
    ingress {
      from {
        namespace_selector {
          match_labels = {
            "kubernetes.io/metadata.name" = var.namespace
          }
        }
      }
    }

    # Allow ingress from GreenLang namespaces (for scraping)
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
      }
    }

    # Allow all egress (needed for S3, scraping, alerting)
    egress {
      to {}
    }
  }

  depends_on = [kubernetes_namespace.monitoring]
}
