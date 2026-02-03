# GreenLang Vault Module - Outputs
# TASK-155: Implement API Key Management (Vault)
# Production-ready outputs for EKS deployment with AWS KMS integration

# =============================================================================
# Vault Access URLs
# =============================================================================

output "vault_url" {
  description = "Public URL for accessing Vault UI and API"
  value       = "https://${var.domain}"
}

output "vault_internal_url" {
  description = "Internal Kubernetes service URL for Vault API within the cluster"
  value       = "http://vault.${var.namespace}.svc.cluster.local:8200"
}

output "vault_active_url" {
  description = "URL for the active Vault node (for direct leader access)"
  value       = "http://vault-active.${var.namespace}.svc.cluster.local:8200"
}

output "vault_standby_url" {
  description = "URL for standby Vault nodes (for read requests)"
  value       = "http://vault-standby.${var.namespace}.svc.cluster.local:8200"
}

# =============================================================================
# AWS IAM Resources
# =============================================================================

output "vault_iam_role_arn" {
  description = "ARN of the IAM role used by Vault for AWS KMS auto-unseal and other AWS operations"
  value       = aws_iam_role.vault.arn
}

output "vault_iam_role_name" {
  description = "Name of the IAM role used by Vault"
  value       = aws_iam_role.vault.name
}

output "vault_iam_role_id" {
  description = "ID of the IAM role used by Vault"
  value       = aws_iam_role.vault.id
}

# =============================================================================
# Kubernetes Resources
# =============================================================================

output "namespace" {
  description = "Kubernetes namespace where Vault is deployed"
  value       = kubernetes_namespace.vault.metadata[0].name
}

output "service_account_name" {
  description = "Kubernetes ServiceAccount name used by Vault"
  value       = "vault"
}

output "helm_release_name" {
  description = "Name of the Vault Helm release"
  value       = helm_release.vault.name
}

output "helm_release_version" {
  description = "Version of the deployed Vault Helm chart"
  value       = helm_release.vault.version
}

output "helm_release_status" {
  description = "Status of the Vault Helm release"
  value       = helm_release.vault.status
}

# =============================================================================
# Secret Engine Paths
# =============================================================================

output "api_keys_path" {
  description = "Path to the KV v2 secrets engine for API keys"
  value       = vault_mount.api_keys.path
}

output "api_keys_full_path" {
  description = "Full path for reading API key secrets (with data/ prefix for KV v2)"
  value       = "${vault_mount.api_keys.path}/data"
}

output "database_path" {
  description = "Path to the KV v2 secrets engine for database credentials"
  value       = vault_mount.database.path
}

output "database_full_path" {
  description = "Full path for reading database secrets (with data/ prefix for KV v2)"
  value       = "${vault_mount.database.path}/data"
}

output "services_path" {
  description = "Path to the KV v2 secrets engine for service secrets"
  value       = vault_mount.services.path
}

output "services_full_path" {
  description = "Full path for reading service secrets (with data/ prefix for KV v2)"
  value       = "${vault_mount.services.path}/data"
}

# =============================================================================
# Authentication Configuration
# =============================================================================

output "kubernetes_auth_path" {
  description = "Path to the Kubernetes authentication backend"
  value       = vault_auth_backend.kubernetes.path
}

output "kubernetes_auth_accessor" {
  description = "Accessor ID for the Kubernetes authentication backend"
  value       = vault_auth_backend.kubernetes.accessor
}

# =============================================================================
# Vault Policies
# =============================================================================

output "policy_agents_read" {
  description = "Name of the read-only policy for GreenLang agents"
  value       = vault_policy.greenlang_agents_read.name
}

output "policy_api" {
  description = "Name of the read/write policy for GreenLang API"
  value       = vault_policy.greenlang_api.name
}

output "policy_rotation" {
  description = "Name of the full-access policy for secrets rotation"
  value       = vault_policy.greenlang_rotation.name
}

# =============================================================================
# Kubernetes Auth Roles
# =============================================================================

output "role_agents" {
  description = "Vault Kubernetes auth role name for GreenLang agents"
  value       = vault_kubernetes_auth_backend_role.greenlang_agents.role_name
}

output "role_api" {
  description = "Vault Kubernetes auth role name for GreenLang API"
  value       = vault_kubernetes_auth_backend_role.greenlang_api.role_name
}

output "role_rotation" {
  description = "Vault Kubernetes auth role name for secrets rotation service"
  value       = vault_kubernetes_auth_backend_role.greenlang_rotation.role_name
}

# =============================================================================
# Connection Information for Applications
# =============================================================================

output "vault_agent_config" {
  description = "Configuration snippet for Vault Agent sidecar injection annotations"
  value = {
    annotations = {
      "vault.hashicorp.com/agent-inject"             = "true"
      "vault.hashicorp.com/agent-inject-status"      = "update"
      "vault.hashicorp.com/role"                     = vault_kubernetes_auth_backend_role.greenlang_agents.role_name
      "vault.hashicorp.com/agent-pre-populate-only"  = "true"
    }
  }
}

output "vault_csi_config" {
  description = "Configuration for Vault CSI driver SecretProviderClass"
  value = {
    provider = "vault"
    parameters = {
      vaultAddress = "http://vault.${var.namespace}.svc.cluster.local:8200"
      roleName     = vault_kubernetes_auth_backend_role.greenlang_agents.role_name
    }
  }
}

# =============================================================================
# Environment Variables for Applications
# =============================================================================

output "environment_variables" {
  description = "Environment variables to configure applications to use Vault"
  value = {
    VAULT_ADDR        = "http://vault.${var.namespace}.svc.cluster.local:8200"
    VAULT_AUTH_METHOD = "kubernetes"
    VAULT_AUTH_PATH   = vault_auth_backend.kubernetes.path
    VAULT_ROLE        = vault_kubernetes_auth_backend_role.greenlang_agents.role_name
  }
}

# =============================================================================
# High Availability Information
# =============================================================================

output "ha_enabled" {
  description = "Whether Vault is running in HA mode"
  value       = var.environment == "production"
}

output "ha_replicas" {
  description = "Number of Vault server replicas"
  value       = var.environment == "production" ? 3 : 1
}

# =============================================================================
# KMS Information
# =============================================================================

output "kms_key_id" {
  description = "AWS KMS key ID used for auto-unseal"
  value       = var.kms_key_id
}

# =============================================================================
# Ingress Information
# =============================================================================

output "ingress_host" {
  description = "Hostname configured for Vault Ingress"
  value       = var.domain
}

output "tls_secret_name" {
  description = "Kubernetes secret name containing TLS certificate"
  value       = "vault-tls"
}

# =============================================================================
# AWS Account Information
# =============================================================================

output "aws_account_id" {
  description = "AWS account ID where Vault resources are deployed"
  value       = data.aws_caller_identity.current.account_id
}

# =============================================================================
# Monitoring Endpoints
# =============================================================================

output "metrics_endpoint" {
  description = "Prometheus metrics endpoint for Vault"
  value       = "http://vault.${var.namespace}.svc.cluster.local:8200/v1/sys/metrics"
}

output "health_endpoint" {
  description = "Health check endpoint for Vault"
  value       = "http://vault.${var.namespace}.svc.cluster.local:8200/v1/sys/health"
}

# =============================================================================
# CLI Configuration
# =============================================================================

output "vault_cli_config" {
  description = "Commands to configure Vault CLI for local access"
  value       = <<-EOT
    # Set Vault address
    export VAULT_ADDR="https://${var.domain}"

    # For local development with port-forward:
    # kubectl port-forward svc/vault -n ${var.namespace} 8200:8200
    # export VAULT_ADDR="http://127.0.0.1:8200"

    # Login with Kubernetes auth (from within cluster):
    # vault login -method=kubernetes role=${vault_kubernetes_auth_backend_role.greenlang_agents.role_name}
  EOT
}

# =============================================================================
# Root Token Secret (Conditional)
# =============================================================================

output "root_token_secret_arn" {
  description = "ARN of AWS Secrets Manager secret containing root token (if enabled). WARNING: Root token should be revoked after initial setup."
  value       = var.store_root_token_in_secrets_manager ? aws_secretsmanager_secret.root_token[0].arn : null
}

output "root_token_secret_name" {
  description = "Name of AWS Secrets Manager secret containing root token (if enabled)"
  value       = var.store_root_token_in_secrets_manager ? aws_secretsmanager_secret.root_token[0].name : null
}
