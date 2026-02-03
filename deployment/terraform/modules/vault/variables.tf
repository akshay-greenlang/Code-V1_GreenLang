# GreenLang Vault Module - Variables
# TASK-155: Implement API Key Management (Vault)
# Production-ready variables for EKS deployment with AWS KMS integration

# =============================================================================
# Required Variables
# =============================================================================

variable "eks_cluster_name" {
  description = "Name of the EKS cluster where Vault will be deployed"
  type        = string

  validation {
    condition     = length(var.eks_cluster_name) > 0
    error_message = "EKS cluster name cannot be empty."
  }
}

variable "eks_cluster_endpoint" {
  description = "EKS cluster API endpoint URL for Kubernetes auth backend configuration"
  type        = string

  validation {
    condition     = can(regex("^https://", var.eks_cluster_endpoint))
    error_message = "EKS cluster endpoint must be a valid HTTPS URL."
  }
}

variable "eks_cluster_ca_cert" {
  description = "Base64-encoded EKS cluster CA certificate for Kubernetes auth backend"
  type        = string
  sensitive   = true
}

variable "kms_key_id" {
  description = "AWS KMS key ID for Vault auto-unseal. Must be in the same region as the EKS cluster."
  type        = string

  validation {
    condition     = length(var.kms_key_id) > 0
    error_message = "KMS key ID cannot be empty."
  }
}

# =============================================================================
# Environment Configuration
# =============================================================================

variable "environment" {
  description = "Deployment environment (dev, staging, production). Affects HA configuration and resource allocation."
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "namespace" {
  description = "Kubernetes namespace for Vault deployment"
  type        = string
  default     = "vault"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", var.namespace))
    error_message = "Namespace must be a valid Kubernetes namespace name."
  }
}

variable "domain" {
  description = "Domain name for Vault UI and API access (e.g., vault.greenlang.io)"
  type        = string
  default     = "vault.greenlang.io"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$", var.domain))
    error_message = "Domain must be a valid DNS name."
  }
}

# =============================================================================
# AWS Configuration
# =============================================================================

variable "aws_region" {
  description = "AWS region for KMS auto-unseal and IAM resources"
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-\\d{1}$", var.aws_region))
    error_message = "Must be a valid AWS region (e.g., us-east-1, eu-west-1)."
  }
}

# =============================================================================
# Helm Chart Configuration
# =============================================================================

variable "vault_chart_version" {
  description = "Version of the HashiCorp Vault Helm chart"
  type        = string
  default     = "0.27.0"
}

variable "vault_image_tag" {
  description = "Vault container image tag. Leave empty to use chart default."
  type        = string
  default     = ""
}

# =============================================================================
# High Availability Configuration
# =============================================================================

variable "ha_enabled" {
  description = "Enable HA mode for Vault. Defaults to true for production, false for other environments."
  type        = bool
  default     = null  # Will be computed based on environment if not set
}

variable "ha_replicas" {
  description = "Number of Vault server replicas in HA mode. Minimum 3 for production."
  type        = number
  default     = 3

  validation {
    condition     = var.ha_replicas >= 1 && var.ha_replicas <= 5
    error_message = "HA replicas must be between 1 and 5."
  }
}

variable "injector_replicas" {
  description = "Number of Vault Agent Injector replicas"
  type        = number
  default     = 2

  validation {
    condition     = var.injector_replicas >= 1 && var.injector_replicas <= 5
    error_message = "Injector replicas must be between 1 and 5."
  }
}

# =============================================================================
# Resource Allocation
# =============================================================================

variable "server_resources" {
  description = "Resource requests and limits for Vault server pods"
  type = object({
    requests = object({
      cpu    = string
      memory = string
    })
    limits = object({
      cpu    = string
      memory = string
    })
  })
  default = {
    requests = {
      cpu    = "250m"
      memory = "256Mi"
    }
    limits = {
      cpu    = "500m"
      memory = "512Mi"
    }
  }
}

variable "injector_resources" {
  description = "Resource requests and limits for Vault Agent Injector pods"
  type = object({
    requests = object({
      cpu    = string
      memory = string
    })
    limits = object({
      cpu    = string
      memory = string
    })
  })
  default = {
    requests = {
      cpu    = "50m"
      memory = "64Mi"
    }
    limits = {
      cpu    = "100m"
      memory = "128Mi"
    }
  }
}

variable "agent_resources" {
  description = "Default resource requests and limits for injected Vault Agent sidecars"
  type = object({
    cpu_request    = string
    cpu_limit      = string
    memory_request = string
    memory_limit   = string
  })
  default = {
    cpu_request    = "250m"
    cpu_limit      = "500m"
    memory_request = "64Mi"
    memory_limit   = "128Mi"
  }
}

variable "csi_resources" {
  description = "Resource requests and limits for Vault CSI Provider pods"
  type = object({
    requests = object({
      cpu    = string
      memory = string
    })
    limits = object({
      cpu    = string
      memory = string
    })
  })
  default = {
    requests = {
      cpu    = "50m"
      memory = "64Mi"
    }
    limits = {
      cpu    = "100m"
      memory = "128Mi"
    }
  }
}

# =============================================================================
# Storage Configuration
# =============================================================================

variable "data_storage_size" {
  description = "Size of the persistent volume for Vault data storage"
  type        = string
  default     = "10Gi"
}

variable "audit_storage_size" {
  description = "Size of the persistent volume for Vault audit logs"
  type        = string
  default     = "10Gi"
}

variable "storage_class" {
  description = "Kubernetes storage class for Vault persistent volumes"
  type        = string
  default     = "gp3"
}

# =============================================================================
# Networking Configuration
# =============================================================================

variable "ingress_enabled" {
  description = "Enable Kubernetes Ingress for Vault"
  type        = bool
  default     = true
}

variable "ingress_class" {
  description = "Kubernetes Ingress class to use"
  type        = string
  default     = "nginx"
}

variable "tls_secret_name" {
  description = "Name of the Kubernetes secret containing TLS certificate for Vault Ingress"
  type        = string
  default     = "vault-tls"
}

variable "cluster_issuer" {
  description = "Cert-manager ClusterIssuer name for automatic TLS certificate provisioning"
  type        = string
  default     = "letsencrypt-prod"
}

variable "enable_ssl_redirect" {
  description = "Force redirect HTTP to HTTPS in Ingress"
  type        = bool
  default     = true
}

# =============================================================================
# UI Configuration
# =============================================================================

variable "ui_enabled" {
  description = "Enable Vault Web UI"
  type        = bool
  default     = true
}

# =============================================================================
# CSI Provider Configuration
# =============================================================================

variable "csi_enabled" {
  description = "Enable Vault CSI Provider for native secret injection via volumes"
  type        = bool
  default     = true
}

# =============================================================================
# Telemetry Configuration
# =============================================================================

variable "prometheus_retention_time" {
  description = "How long to retain metrics for Prometheus scraping"
  type        = string
  default     = "30s"
}

variable "enable_unauthenticated_metrics" {
  description = "Allow unauthenticated access to /v1/sys/metrics endpoint"
  type        = bool
  default     = true
}

# =============================================================================
# Secret Engine Paths
# =============================================================================

variable "api_keys_path" {
  description = "KV v2 secrets engine path for API keys"
  type        = string
  default     = "greenlang/api-keys"
}

variable "database_path" {
  description = "KV v2 secrets engine path for database credentials"
  type        = string
  default     = "greenlang/database"
}

variable "services_path" {
  description = "KV v2 secrets engine path for service secrets"
  type        = string
  default     = "greenlang/services"
}

# =============================================================================
# Kubernetes Auth Configuration
# =============================================================================

variable "kubernetes_auth_path" {
  description = "Path for Kubernetes authentication backend"
  type        = string
  default     = "kubernetes"
}

variable "token_ttl" {
  description = "Default TTL for tokens issued by Kubernetes auth (in seconds)"
  type        = number
  default     = 3600

  validation {
    condition     = var.token_ttl >= 300 && var.token_ttl <= 86400
    error_message = "Token TTL must be between 300 seconds (5 min) and 86400 seconds (24 hours)."
  }
}

variable "token_max_ttl" {
  description = "Maximum TTL for tokens issued by Kubernetes auth (in seconds)"
  type        = number
  default     = 86400

  validation {
    condition     = var.token_max_ttl >= 3600 && var.token_max_ttl <= 604800
    error_message = "Token max TTL must be between 3600 seconds (1 hour) and 604800 seconds (7 days)."
  }
}

# =============================================================================
# Service Account Configuration
# =============================================================================

variable "greenlang_agents_namespace" {
  description = "Kubernetes namespace where GreenLang agents are deployed"
  type        = string
  default     = "greenlang-agents"
}

variable "greenlang_agents_service_accounts" {
  description = "List of service account names allowed to access agent secrets"
  type        = list(string)
  default     = ["greenlang-agents-sa", "default"]
}

variable "greenlang_api_service_accounts" {
  description = "List of service account names allowed to access API secrets"
  type        = list(string)
  default     = ["greenlang-api-sa"]
}

variable "greenlang_rotation_service_accounts" {
  description = "List of service account names allowed to rotate secrets"
  type        = list(string)
  default     = ["greenlang-rotation-sa"]
}

# =============================================================================
# Tags and Labels
# =============================================================================

variable "tags" {
  description = "AWS tags to apply to all resources created by this module"
  type        = map(string)
  default     = {}
}

variable "labels" {
  description = "Kubernetes labels to apply to all resources created by this module"
  type        = map(string)
  default     = {}
}

# =============================================================================
# Advanced Configuration
# =============================================================================

variable "extra_helm_values" {
  description = "Additional Helm values to merge with the default configuration (YAML string)"
  type        = string
  default     = ""
}

variable "service_type" {
  description = "Kubernetes Service type for Vault (ClusterIP, LoadBalancer, NodePort)"
  type        = string
  default     = "ClusterIP"

  validation {
    condition     = contains(["ClusterIP", "LoadBalancer", "NodePort"], var.service_type)
    error_message = "Service type must be one of: ClusterIP, LoadBalancer, NodePort."
  }
}

variable "pod_disruption_budget_enabled" {
  description = "Enable PodDisruptionBudget for Vault server pods"
  type        = bool
  default     = true
}

variable "pod_disruption_budget_min_available" {
  description = "Minimum number of Vault pods that must be available during voluntary disruptions"
  type        = number
  default     = 1
}

variable "affinity_enabled" {
  description = "Enable pod anti-affinity rules to spread Vault pods across nodes"
  type        = bool
  default     = true
}

variable "priority_class_name" {
  description = "Kubernetes PriorityClass name for Vault pods"
  type        = string
  default     = ""
}

# =============================================================================
# Root Token Management
# =============================================================================

variable "store_root_token_in_secrets_manager" {
  description = "Store Vault root token in AWS Secrets Manager (for initial setup only - should be revoked after setup)"
  type        = bool
  default     = false
}

variable "root_token_secret_name" {
  description = "Name for AWS Secrets Manager secret to store root token"
  type        = string
  default     = "vault-root-token"
}

# =============================================================================
# Backup Configuration
# =============================================================================

variable "enable_snapshots" {
  description = "Enable automated Raft snapshots for backup"
  type        = bool
  default     = true
}

variable "snapshot_interval" {
  description = "Interval between automated Raft snapshots (e.g., 1h, 24h)"
  type        = string
  default     = "1h"
}
