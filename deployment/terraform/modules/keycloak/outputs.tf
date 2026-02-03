# GreenLang Keycloak Module - Outputs
# TASK-152: Implement OAuth2/OIDC (Keycloak)
# Production-ready outputs for EKS deployment

# ==============================================================================
# KEYCLOAK SERVER OUTPUTS
# ==============================================================================

output "keycloak_url" {
  description = "Full URL to access Keycloak admin console and authentication endpoints"
  value       = "https://${var.domain}"
}

output "keycloak_admin_url" {
  description = "URL to access Keycloak admin console"
  value       = "https://${var.domain}/admin"
}

output "keycloak_internal_url" {
  description = "Internal Kubernetes service URL for Keycloak (for in-cluster communication)"
  value       = "http://keycloak.${var.namespace}.svc.cluster.local:8080"
}

output "keycloak_hostname" {
  description = "Keycloak hostname/domain"
  value       = var.domain
}

# ==============================================================================
# KUBERNETES RESOURCES
# ==============================================================================

output "namespace" {
  description = "Kubernetes namespace where Keycloak is deployed"
  value       = var.create_namespace ? kubernetes_namespace.keycloak[0].metadata[0].name : var.namespace
}

output "helm_release_name" {
  description = "Name of the Keycloak Helm release"
  value       = helm_release.keycloak.name
}

output "helm_release_version" {
  description = "Version of the deployed Keycloak Helm chart"
  value       = helm_release.keycloak.version
}

output "helm_release_status" {
  description = "Status of the Keycloak Helm release"
  value       = helm_release.keycloak.status
}

output "service_account_name" {
  description = "Name of the Kubernetes service account used by Keycloak"
  value       = var.service_account_name
}

# ==============================================================================
# SECRET REFERENCES
# ==============================================================================

output "admin_secret_name" {
  description = "Name of the Kubernetes secret containing the Keycloak admin password"
  value       = kubernetes_secret.keycloak_admin.metadata[0].name
}

output "admin_secret_namespace" {
  description = "Namespace of the Kubernetes secret containing the Keycloak admin password"
  value       = kubernetes_secret.keycloak_admin.metadata[0].namespace
}

output "admin_username" {
  description = "Keycloak admin username"
  value       = var.admin_username
}

output "tls_secret_name" {
  description = "Name of the TLS secret for HTTPS"
  value       = var.tls_secret_name
}

# ==============================================================================
# REALM CONFIGURATION
# ==============================================================================

output "realm_name" {
  description = "Name of the GreenLang Keycloak realm"
  value       = var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name
}

output "realm_id" {
  description = "ID of the GreenLang Keycloak realm"
  value       = var.create_realm ? keycloak_realm.greenlang[0].id : null
}

output "realm_display_name" {
  description = "Display name of the GreenLang realm"
  value       = var.realm_display_name
}

# ==============================================================================
# CLIENT CONFIGURATION
# ==============================================================================

output "api_client_id" {
  description = "Client ID for the GreenLang API client"
  value       = var.create_clients ? keycloak_openid_client.greenlang_api[0].client_id : "greenlang-api"
}

output "api_client_uuid" {
  description = "Internal UUID of the GreenLang API client"
  value       = var.create_clients ? keycloak_openid_client.greenlang_api[0].id : null
  sensitive   = true
}

output "webapp_client_id" {
  description = "Client ID for the GreenLang Web Application client (public client)"
  value       = var.create_clients ? keycloak_openid_client.greenlang_webapp[0].client_id : "greenlang-webapp"
}

output "webapp_client_uuid" {
  description = "Internal UUID of the GreenLang Web Application client"
  value       = var.create_clients ? keycloak_openid_client.greenlang_webapp[0].id : null
  sensitive   = true
}

# ==============================================================================
# OIDC ENDPOINTS
# ==============================================================================

output "oidc_issuer_url" {
  description = "OIDC issuer URL for token validation"
  value       = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}"
}

output "oidc_authorization_endpoint" {
  description = "OIDC authorization endpoint for OAuth2 flows"
  value       = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/auth"
}

output "oidc_token_endpoint" {
  description = "OIDC token endpoint for obtaining access tokens"
  value       = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/token"
}

output "oidc_userinfo_endpoint" {
  description = "OIDC userinfo endpoint for retrieving user profile"
  value       = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/userinfo"
}

output "oidc_jwks_uri" {
  description = "OIDC JWKS URI for token signature verification"
  value       = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/certs"
}

output "oidc_logout_endpoint" {
  description = "OIDC end session endpoint for logout"
  value       = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/logout"
}

output "oidc_well_known_url" {
  description = "OIDC discovery document URL (.well-known/openid-configuration)"
  value       = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/.well-known/openid-configuration"
}

# ==============================================================================
# ROLE IDENTIFIERS
# ==============================================================================

output "role_ids" {
  description = "Map of role names to their Keycloak IDs"
  value = var.create_roles ? {
    viewer      = keycloak_role.viewer[0].id
    operator    = keycloak_role.operator[0].id
    analyst     = keycloak_role.analyst[0].id
    manager     = keycloak_role.manager[0].id
    admin       = keycloak_role.admin[0].id
    super_admin = keycloak_role.super_admin[0].id
  } : {}
}

output "role_names" {
  description = "List of all defined role names"
  value = var.create_roles ? [
    "viewer",
    "operator",
    "analyst",
    "manager",
    "admin",
    "super_admin"
  ] : []
}

# ==============================================================================
# CLIENT SCOPE IDENTIFIERS
# ==============================================================================

output "client_scope_ids" {
  description = "Map of client scope names to their Keycloak IDs"
  value = var.create_client_scopes ? {
    emissions = keycloak_openid_client_scope.greenlang_emissions[0].id
    reports   = keycloak_openid_client_scope.greenlang_reports[0].id
    agents    = keycloak_openid_client_scope.greenlang_agents[0].id
  } : {}
}

output "client_scope_names" {
  description = "List of all defined client scope names"
  value = var.create_client_scopes ? [
    "emissions",
    "reports",
    "agents"
  ] : []
}

# ==============================================================================
# HEALTH AND MONITORING ENDPOINTS
# ==============================================================================

output "health_live_endpoint" {
  description = "Keycloak liveness probe endpoint"
  value       = "https://${var.domain}/health/live"
}

output "health_ready_endpoint" {
  description = "Keycloak readiness probe endpoint"
  value       = "https://${var.domain}/health/ready"
}

output "metrics_endpoint" {
  description = "Prometheus metrics endpoint (if enabled)"
  value       = var.enable_metrics ? "https://${var.domain}/metrics" : null
}

output "metrics_internal_endpoint" {
  description = "Internal Prometheus metrics endpoint for scraping"
  value       = var.enable_metrics ? "http://keycloak.${var.namespace}.svc.cluster.local:8080/metrics" : null
}

# ==============================================================================
# SERVICE MESH INTEGRATION
# ==============================================================================

output "service_name" {
  description = "Name of the Kubernetes service for Keycloak"
  value       = "keycloak"
}

output "service_port" {
  description = "HTTP port for the Keycloak service"
  value       = 8080
}

output "service_fqdn" {
  description = "Fully qualified domain name of the Keycloak service within the cluster"
  value       = "keycloak.${var.namespace}.svc.cluster.local"
}

# ==============================================================================
# ENVIRONMENT INFORMATION
# ==============================================================================

output "environment" {
  description = "Deployment environment"
  value       = var.environment
}

output "is_production" {
  description = "Boolean indicating if this is a production deployment"
  value       = var.environment == "production"
}

output "replicas" {
  description = "Number of Keycloak replicas deployed"
  value       = var.replicas != null ? var.replicas : (var.environment == "production" ? 3 : 1)
}

output "keycloak_version" {
  description = "Keycloak version deployed"
  value       = var.keycloak_version
}

# ==============================================================================
# INTEGRATION HELPER OUTPUTS
# ==============================================================================

output "api_client_config" {
  description = "Configuration object for integrating with the GreenLang API client"
  value = {
    client_id    = var.create_clients ? keycloak_openid_client.greenlang_api[0].client_id : "greenlang-api"
    issuer_url   = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}"
    token_url    = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/token"
    jwks_uri     = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/certs"
    access_type  = "CONFIDENTIAL"
  }
}

output "webapp_client_config" {
  description = "Configuration object for integrating with the GreenLang Web App client"
  value = {
    client_id         = var.create_clients ? keycloak_openid_client.greenlang_webapp[0].client_id : "greenlang-webapp"
    issuer_url        = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}"
    authorization_url = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/auth"
    token_url         = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/token"
    logout_url        = "https://${var.domain}/realms/${var.create_realm ? keycloak_realm.greenlang[0].realm : var.realm_name}/protocol/openid-connect/logout"
    access_type       = "PUBLIC"
    pkce_enabled      = true
  }
}

output "kubernetes_labels" {
  description = "Standard Kubernetes labels applied to all Keycloak resources"
  value = {
    "app.kubernetes.io/name"       = "keycloak"
    "app.kubernetes.io/part-of"    = "greenlang"
    "app.kubernetes.io/version"    = var.keycloak_version
    "app.kubernetes.io/managed-by" = "terraform"
    "environment"                  = var.environment
  }
}
