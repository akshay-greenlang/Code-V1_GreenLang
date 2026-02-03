# GreenLang Keycloak Module - Variables
# TASK-152: Implement OAuth2/OIDC (Keycloak)
# Production-ready input variables for EKS deployment

# ==============================================================================
# REQUIRED VARIABLES
# ==============================================================================

variable "keycloak_admin_password" {
  description = "Keycloak admin password. Must be at least 16 characters with complexity requirements for production."
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.keycloak_admin_password) >= 16
    error_message = "Keycloak admin password must be at least 16 characters long."
  }
}

variable "database_url" {
  description = "PostgreSQL database connection URL in format: jdbc:postgresql://host:port/database"
  type        = string
  sensitive   = true

  validation {
    condition     = can(regex("^(jdbc:)?postgresql://", var.database_url))
    error_message = "Database URL must be a valid PostgreSQL connection string."
  }
}

# ==============================================================================
# KUBERNETES CONFIGURATION
# ==============================================================================

variable "namespace" {
  description = "Kubernetes namespace for Keycloak deployment. Will be created if it does not exist."
  type        = string
  default     = "keycloak"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", var.namespace))
    error_message = "Namespace must be a valid Kubernetes namespace name (lowercase alphanumeric with hyphens)."
  }
}

variable "create_namespace" {
  description = "Whether to create the Kubernetes namespace. Set to false if namespace already exists."
  type        = bool
  default     = true
}

variable "service_account_name" {
  description = "Name of the Kubernetes service account for Keycloak pods"
  type        = string
  default     = "keycloak"
}

variable "enable_istio_injection" {
  description = "Enable Istio sidecar injection for the Keycloak namespace"
  type        = bool
  default     = true
}

# ==============================================================================
# DOMAIN AND NETWORKING
# ==============================================================================

variable "domain" {
  description = "Fully qualified domain name for Keycloak (e.g., auth.greenlang.io)"
  type        = string
  default     = "auth.greenlang.io"

  validation {
    condition     = can(regex("^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$", var.domain))
    error_message = "Domain must be a valid DNS hostname."
  }
}

variable "ingress_class" {
  description = "Kubernetes ingress class to use (nginx, alb, traefik)"
  type        = string
  default     = "nginx"
}

variable "tls_secret_name" {
  description = "Name of the Kubernetes TLS secret for HTTPS. Will be created by cert-manager if using cluster-issuer."
  type        = string
  default     = "keycloak-tls"
}

variable "cluster_issuer" {
  description = "Cert-manager cluster issuer name for automatic TLS certificate provisioning"
  type        = string
  default     = "letsencrypt-prod"
}

variable "enable_ssl_redirect" {
  description = "Redirect HTTP to HTTPS"
  type        = bool
  default     = true
}

# ==============================================================================
# ENVIRONMENT AND DEPLOYMENT
# ==============================================================================

variable "environment" {
  description = "Deployment environment (dev, staging, production). Affects replicas, logging, and security settings."
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "replicas" {
  description = "Number of Keycloak replicas. If null, uses environment-based defaults (production: 3, others: 1)"
  type        = number
  default     = null

  validation {
    condition     = var.replicas == null || (var.replicas >= 1 && var.replicas <= 10)
    error_message = "Replicas must be between 1 and 10."
  }
}

variable "keycloak_version" {
  description = "Keycloak Docker image version/tag"
  type        = string
  default     = "23.0"
}

variable "helm_chart_version" {
  description = "Version of the Keycloak Helm chart (codecentric/keycloakx)"
  type        = string
  default     = "2.3.0"
}

# ==============================================================================
# RESOURCE LIMITS
# ==============================================================================

variable "resources" {
  description = "Resource requests and limits for Keycloak pods"
  type = object({
    requests = object({
      memory = string
      cpu    = string
    })
    limits = object({
      memory = string
      cpu    = string
    })
  })
  default = {
    requests = {
      memory = "512Mi"
      cpu    = "250m"
    }
    limits = {
      memory = "1Gi"
      cpu    = "1000m"
    }
  }
}

variable "java_opts" {
  description = "JVM options for Keycloak. If null, uses sensible defaults based on resource limits."
  type        = string
  default     = null
}

# ==============================================================================
# HIGH AVAILABILITY
# ==============================================================================

variable "enable_pod_disruption_budget" {
  description = "Enable PodDisruptionBudget for high availability. Recommended for production."
  type        = bool
  default     = null # Defaults to true for production environment
}

variable "pdb_min_available" {
  description = "Minimum number of pods that must be available during disruptions"
  type        = number
  default     = 2
}

variable "enable_topology_spread" {
  description = "Enable topology spread constraints to distribute pods across availability zones"
  type        = bool
  default     = null # Defaults to true for production environment
}

variable "enable_pod_anti_affinity" {
  description = "Enable pod anti-affinity to prevent scheduling multiple replicas on same node"
  type        = bool
  default     = true
}

# ==============================================================================
# AUTHENTICATION AND SECURITY
# ==============================================================================

variable "admin_username" {
  description = "Keycloak admin username"
  type        = string
  default     = "admin"
}

variable "realm_name" {
  description = "Name of the GreenLang Keycloak realm"
  type        = string
  default     = "greenlang"
}

variable "realm_display_name" {
  description = "Display name for the GreenLang realm"
  type        = string
  default     = "GreenLang"
}

variable "registration_allowed" {
  description = "Allow user self-registration"
  type        = bool
  default     = false
}

variable "password_policy" {
  description = "Keycloak password policy string"
  type        = string
  default     = "length(12) and upperCase(1) and lowerCase(1) and specialChars(1) and digits(1) and notUsername"
}

variable "brute_force_protection" {
  description = "Brute force protection configuration"
  type = object({
    enabled                  = bool
    permanent_lockout        = bool
    max_login_failures       = number
    wait_increment_seconds   = number
    max_failure_wait_seconds = number
    failure_reset_time       = number
  })
  default = {
    enabled                  = true
    permanent_lockout        = false
    max_login_failures       = 5
    wait_increment_seconds   = 60
    max_failure_wait_seconds = 900
    failure_reset_time       = 43200
  }
}

# ==============================================================================
# TOKEN SETTINGS
# ==============================================================================

variable "token_settings" {
  description = "OAuth2/OIDC token lifetime settings"
  type = object({
    access_token_lifespan           = string
    sso_session_idle_timeout        = string
    sso_session_max_lifespan        = string
    offline_session_idle_timeout    = string
    offline_session_max_lifespan    = string
  })
  default = {
    access_token_lifespan           = "5m"
    sso_session_idle_timeout        = "30m"
    sso_session_max_lifespan        = "10h"
    offline_session_idle_timeout    = "720h"
    offline_session_max_lifespan    = "720h"
  }
}

# ==============================================================================
# CLIENT CONFIGURATION
# ==============================================================================

variable "api_client_redirect_uris" {
  description = "Valid redirect URIs for the GreenLang API client"
  type        = list(string)
  default     = [
    "https://api.greenlang.io/*",
    "https://app.greenlang.io/*"
  ]
}

variable "api_client_web_origins" {
  description = "Allowed web origins for the GreenLang API client (CORS)"
  type        = list(string)
  default     = [
    "https://api.greenlang.io",
    "https://app.greenlang.io"
  ]
}

variable "webapp_client_redirect_uris" {
  description = "Valid redirect URIs for the GreenLang Web App client"
  type        = list(string)
  default     = [
    "https://app.greenlang.io/*"
  ]
}

variable "webapp_client_web_origins" {
  description = "Allowed web origins for the GreenLang Web App client (CORS)"
  type        = list(string)
  default     = [
    "https://app.greenlang.io"
  ]
}

variable "enable_localhost_redirect" {
  description = "Enable localhost redirect URIs for development. Automatically disabled in production."
  type        = bool
  default     = null # Defaults based on environment
}

variable "localhost_port" {
  description = "Localhost port for development redirects"
  type        = number
  default     = 3000
}

# ==============================================================================
# OBSERVABILITY
# ==============================================================================

variable "enable_metrics" {
  description = "Enable Prometheus metrics endpoint (/metrics)"
  type        = bool
  default     = true
}

variable "enable_health_endpoints" {
  description = "Enable health check endpoints (/health/live, /health/ready)"
  type        = bool
  default     = true
}

variable "log_level" {
  description = "Keycloak log level. If null, uses DEBUG for non-production, INFO for production."
  type        = string
  default     = null

  validation {
    condition     = var.log_level == null || contains(["ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"], var.log_level)
    error_message = "Log level must be one of: ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE, WARN."
  }
}

variable "metrics_port" {
  description = "Port for metrics and management endpoints"
  type        = number
  default     = 9000
}

# ==============================================================================
# HEALTH CHECKS
# ==============================================================================

variable "liveness_probe" {
  description = "Liveness probe configuration"
  type = object({
    initial_delay_seconds = number
    period_seconds        = number
    timeout_seconds       = number
    failure_threshold     = number
  })
  default = {
    initial_delay_seconds = 60
    period_seconds        = 10
    timeout_seconds       = 5
    failure_threshold     = 3
  }
}

variable "readiness_probe" {
  description = "Readiness probe configuration"
  type = object({
    initial_delay_seconds = number
    period_seconds        = number
    timeout_seconds       = number
    failure_threshold     = number
    success_threshold     = number
  })
  default = {
    initial_delay_seconds = 30
    period_seconds        = 10
    timeout_seconds       = 5
    failure_threshold     = 3
    success_threshold     = 1
  }
}

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================

variable "database_vendor" {
  description = "Database vendor (postgres, mysql, mariadb)"
  type        = string
  default     = "postgres"

  validation {
    condition     = contains(["postgres", "mysql", "mariadb"], var.database_vendor)
    error_message = "Database vendor must be one of: postgres, mysql, mariadb."
  }
}

variable "database_schema" {
  description = "Database schema name"
  type        = string
  default     = "public"
}

# ==============================================================================
# AWS INTEGRATION (for EKS)
# ==============================================================================

variable "enable_aws_secrets_manager" {
  description = "Store Keycloak secrets in AWS Secrets Manager instead of Kubernetes secrets"
  type        = bool
  default     = false
}

variable "aws_secrets_manager_prefix" {
  description = "Prefix for secrets stored in AWS Secrets Manager"
  type        = string
  default     = "greenlang/keycloak"
}

variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts (IRSA) for AWS integration"
  type        = bool
  default     = false
}

variable "irsa_role_arn" {
  description = "ARN of the IAM role for IRSA. Required if enable_irsa is true."
  type        = string
  default     = ""
}

# ==============================================================================
# TAGGING AND METADATA
# ==============================================================================

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "labels" {
  description = "Additional labels to apply to Kubernetes resources"
  type        = map(string)
  default     = {}
}

variable "annotations" {
  description = "Additional annotations to apply to Kubernetes resources"
  type        = map(string)
  default     = {}
}

# ==============================================================================
# FEATURE FLAGS
# ==============================================================================

variable "create_realm" {
  description = "Create the GreenLang Keycloak realm and clients"
  type        = bool
  default     = true
}

variable "create_clients" {
  description = "Create the API and WebApp OIDC clients"
  type        = bool
  default     = true
}

variable "create_roles" {
  description = "Create the default GreenLang roles (viewer, operator, analyst, manager, admin, super_admin)"
  type        = bool
  default     = true
}

variable "create_client_scopes" {
  description = "Create the default client scopes (emissions, reports, agents)"
  type        = bool
  default     = true
}

# ==============================================================================
# ADVANCED SETTINGS
# ==============================================================================

variable "extra_env" {
  description = "Additional environment variables for Keycloak pods"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}

variable "extra_volumes" {
  description = "Additional volumes to mount in Keycloak pods"
  type = list(object({
    name       = string
    mount_path = string
    config_map = optional(string)
    secret     = optional(string)
  }))
  default = []
}

variable "node_selector" {
  description = "Node selector for Keycloak pods"
  type        = map(string)
  default     = {}
}

variable "tolerations" {
  description = "Tolerations for Keycloak pods"
  type = list(object({
    key      = string
    operator = string
    value    = optional(string)
    effect   = string
  }))
  default = []
}

variable "priority_class_name" {
  description = "Priority class name for Keycloak pods"
  type        = string
  default     = ""
}
