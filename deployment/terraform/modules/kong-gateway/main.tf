# =============================================================================
# GreenLang Climate OS - Kong API Gateway Terraform Module
# =============================================================================
# PRD: INFRA-006 API Gateway (Kong)
# Purpose: Kong namespace, service accounts, IAM roles, Helm release
#
# This module provisions:
# - Kubernetes namespace with Istio injection labels
# - Service accounts for Kong Gateway and Ingress Controller (KIC)
# - AWS IAM role with IRSA for S3 config backup, CloudWatch, and Secrets Manager
# - ClusterRole and ClusterRoleBinding for Kong Ingress Controller RBAC
# - Resource quotas and limit ranges for namespace governance
# =============================================================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.0"
    }
  }
}

# =============================================================================
# KONG NAMESPACE
# =============================================================================

resource "kubernetes_namespace" "kong" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "kong"
      "app.kubernetes.io/part-of"    = "greenlang"
      "app.kubernetes.io/managed-by" = "terraform"
      "istio-injection"              = var.istio_injection ? "enabled" : "disabled"
    }
    annotations = {
      "description" = "Kong API Gateway for GreenLang Climate OS"
    }
  }
}

# =============================================================================
# SERVICE ACCOUNTS
# =============================================================================

# Service Account for Kong Gateway (data plane)
resource "kubernetes_service_account" "kong_gateway" {
  metadata {
    name      = "${var.release_name}-gateway"
    namespace = kubernetes_namespace.kong.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "kong"
      "app.kubernetes.io/component" = "gateway"
    }
    annotations = {
      "eks.amazonaws.com/role-arn" = var.create_iam_role ? aws_iam_role.kong[0].arn : var.existing_iam_role_arn
    }
  }
  automount_service_account_token = true
}

# Service Account for Kong Ingress Controller (control plane)
resource "kubernetes_service_account" "kong_controller" {
  metadata {
    name      = "${var.release_name}-controller"
    namespace = kubernetes_namespace.kong.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "kong"
      "app.kubernetes.io/component" = "controller"
    }
  }
  automount_service_account_token = true
}

# =============================================================================
# IAM ROLE FOR KONG (IRSA - IAM Roles for Service Accounts)
# =============================================================================

resource "aws_iam_role" "kong" {
  count = var.create_iam_role ? 1 : 0

  name = "${var.cluster_name}-kong-gateway"
  path = "/greenlang/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.oidc_provider}:sub" = "system:serviceaccount:${var.namespace}:${var.release_name}-gateway"
            "${var.oidc_provider}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Component = "kong-gateway"
    ManagedBy = "terraform"
  })
}

# =============================================================================
# IAM POLICY FOR KONG
# =============================================================================
# Grants permissions for:
# - S3: Configuration backup and restore
# - CloudWatch: Centralized log shipping
# - Secrets Manager: Retrieve sensitive gateway configuration

resource "aws_iam_role_policy" "kong" {
  count = var.create_iam_role ? 1 : 0

  name = "kong-gateway-policy"
  role = aws_iam_role.kong[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ConfigBackup"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.config_backup_bucket}",
          "arn:aws:s3:::${var.config_backup_bucket}/kong/*"
        ]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:*:log-group:/greenlang/kong/*"
      },
      {
        Sid    = "SecretsManagerRead"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "arn:aws:secretsmanager:${var.aws_region}:*:secret:greenlang/kong/*"
      }
    ]
  })
}

# =============================================================================
# RBAC - CLUSTERROLE FOR KONG INGRESS CONTROLLER
# =============================================================================
# The Kong Ingress Controller requires cluster-wide read access to Kubernetes
# resources (Ingress, Services, Endpoints, Secrets) and write access to
# Ingress status and Kong CRD status subresources.

resource "kubernetes_cluster_role" "kong_controller" {
  metadata {
    name = "${var.release_name}-controller"
    labels = {
      "app.kubernetes.io/name"      = "kong"
      "app.kubernetes.io/component" = "controller"
    }
  }

  # Core Kubernetes resources (read-only)
  rule {
    api_groups = [""]
    resources  = ["endpoints", "nodes", "pods", "secrets", "services", "namespaces"]
    verbs      = ["list", "watch", "get"]
  }

  # Events (create/patch for controller status reporting)
  rule {
    api_groups = [""]
    resources  = ["events"]
    verbs      = ["create", "patch"]
  }

  # Ingress resources (read + status update)
  rule {
    api_groups = ["networking.k8s.io"]
    resources  = ["ingresses", "ingressclasses"]
    verbs      = ["get", "list", "watch"]
  }
  rule {
    api_groups = ["networking.k8s.io"]
    resources  = ["ingresses/status"]
    verbs      = ["update"]
  }

  # Kong Custom Resource Definitions (read)
  rule {
    api_groups = ["configuration.konghq.com"]
    resources  = ["kongplugins", "kongclusterplugins", "kongconsumers", "kongconsumergroups", "kongingresses", "tcpingresses", "udpingresses", "kongupstreampolicies"]
    verbs      = ["get", "list", "watch"]
  }

  # Kong CRD status subresources (update)
  rule {
    api_groups = ["configuration.konghq.com"]
    resources  = ["kongplugins/status", "kongclusterplugins/status", "kongconsumers/status", "kongingresses/status", "tcpingresses/status", "udpingresses/status"]
    verbs      = ["update"]
  }

  # Leader election (required for HA controller deployments)
  rule {
    api_groups = ["coordination.k8s.io"]
    resources  = ["leases"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  # EndpointSlice discovery
  rule {
    api_groups = ["discovery.k8s.io"]
    resources  = ["endpointslices"]
    verbs      = ["get", "list", "watch"]
  }
}

# =============================================================================
# CLUSTERROLEBINDING FOR KONG INGRESS CONTROLLER
# =============================================================================

resource "kubernetes_cluster_role_binding" "kong_controller" {
  metadata {
    name = "${var.release_name}-controller"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.kong_controller.metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.kong_controller.metadata[0].name
    namespace = kubernetes_namespace.kong.metadata[0].name
  }
}

# =============================================================================
# RESOURCE QUOTA FOR KONG NAMESPACE
# =============================================================================
# Enforces aggregate resource consumption limits across all pods in the
# Kong namespace to prevent runaway resource usage.

resource "kubernetes_resource_quota" "kong" {
  metadata {
    name      = "kong-quota"
    namespace = kubernetes_namespace.kong.metadata[0].name
  }

  spec {
    hard = {
      "requests.cpu"    = var.resource_quota_cpu_requests
      "requests.memory" = var.resource_quota_memory_requests
      "limits.cpu"      = var.resource_quota_cpu_limits
      "limits.memory"   = var.resource_quota_memory_limits
      "pods"            = var.resource_quota_pods
    }
  }
}

# =============================================================================
# LIMITRANGE FOR KONG NAMESPACE
# =============================================================================
# Sets default and boundary resource values for containers that do not
# specify their own requests/limits. Prevents individual containers from
# consuming disproportionate cluster resources.

resource "kubernetes_limit_range" "kong" {
  metadata {
    name      = "kong-limits"
    namespace = kubernetes_namespace.kong.metadata[0].name
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
        cpu    = "4000m"
        memory = "4Gi"
      }
      min = {
        cpu    = "50m"
        memory = "64Mi"
      }
    }
  }
}
