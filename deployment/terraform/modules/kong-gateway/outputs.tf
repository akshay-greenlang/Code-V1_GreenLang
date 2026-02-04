# =============================================================================
# GreenLang Climate OS - Kong API Gateway Module Outputs
# =============================================================================
# PRD: INFRA-006 API Gateway (Kong)
# Production-ready outputs for downstream module consumption and CI/CD pipelines
# =============================================================================

# =============================================================================
# KUBERNETES NAMESPACE
# =============================================================================

output "namespace" {
  description = "Kubernetes namespace where Kong API Gateway resources are deployed"
  value       = kubernetes_namespace.kong.metadata[0].name
}

# =============================================================================
# SERVICE ACCOUNTS
# =============================================================================

output "service_account_name" {
  description = "Name of the Kubernetes service account for the Kong Gateway data plane"
  value       = kubernetes_service_account.kong_gateway.metadata[0].name
}

output "service_account_controller_name" {
  description = "Name of the Kubernetes service account for the Kong Ingress Controller"
  value       = kubernetes_service_account.kong_controller.metadata[0].name
}

# =============================================================================
# AWS IAM RESOURCES
# =============================================================================

output "iam_role_arn" {
  description = "ARN of the IAM role attached to the Kong Gateway service account via IRSA. Returns the existing role ARN when create_iam_role is false."
  value       = var.create_iam_role ? aws_iam_role.kong[0].arn : var.existing_iam_role_arn
}

# =============================================================================
# RBAC RESOURCES
# =============================================================================

output "cluster_role_name" {
  description = "Name of the ClusterRole granting the Kong Ingress Controller permissions to manage Ingress and Kong CRD resources"
  value       = kubernetes_cluster_role.kong_controller.metadata[0].name
}
