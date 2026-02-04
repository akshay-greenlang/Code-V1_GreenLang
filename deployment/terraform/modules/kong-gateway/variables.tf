# =============================================================================
# GreenLang Climate OS - Kong API Gateway Module Variables
# =============================================================================
# PRD: INFRA-006 API Gateway (Kong)
# Production-ready input variables for EKS deployment with IRSA integration
# =============================================================================

# =============================================================================
# KUBERNETES CONFIGURATION
# =============================================================================

variable "namespace" {
  description = "Kubernetes namespace for Kong API Gateway deployment. Created by this module."
  type        = string
  default     = "kong"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", var.namespace))
    error_message = "Namespace must be a valid Kubernetes namespace name (lowercase alphanumeric with hyphens)."
  }
}

variable "release_name" {
  description = "Base name for Kong Kubernetes resources (service accounts, RBAC bindings). Used as a prefix for all named resources."
  type        = string
  default     = "kong"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", var.release_name))
    error_message = "Release name must be a valid Kubernetes resource name (lowercase alphanumeric with hyphens)."
  }
}

variable "istio_injection" {
  description = "Enable Istio sidecar injection for the Kong namespace. Recommended for service mesh observability and mTLS."
  type        = bool
  default     = true
}

# =============================================================================
# AWS / EKS CONFIGURATION
# =============================================================================

variable "cluster_name" {
  description = "Name of the EKS cluster where Kong will be deployed. Used for IAM role naming."
  type        = string

  validation {
    condition     = length(var.cluster_name) > 0
    error_message = "EKS cluster name cannot be empty."
  }
}

variable "aws_region" {
  description = "AWS region for IAM, CloudWatch, S3, and Secrets Manager resources."
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-\\d{1}$", var.aws_region))
    error_message = "Must be a valid AWS region (e.g., us-east-1, eu-west-1)."
  }
}

# =============================================================================
# IAM / IRSA CONFIGURATION
# =============================================================================

variable "create_iam_role" {
  description = "Whether to create a new IAM role for Kong via IRSA. Set to false to use an existing role."
  type        = bool
  default     = true
}

variable "existing_iam_role_arn" {
  description = "ARN of an existing IAM role to attach to the Kong gateway service account. Only used when create_iam_role is false."
  type        = string
  default     = ""
}

variable "oidc_provider_arn" {
  description = "ARN of the EKS OIDC identity provider for IRSA trust policy. Required when create_iam_role is true."
  type        = string

  validation {
    condition     = length(var.oidc_provider_arn) > 0
    error_message = "OIDC provider ARN cannot be empty."
  }
}

variable "oidc_provider" {
  description = "OIDC provider URL (without https:// prefix) for IAM trust policy conditions. Example: oidc.eks.us-east-1.amazonaws.com/id/ABCDEF1234567890"
  type        = string

  validation {
    condition     = length(var.oidc_provider) > 0
    error_message = "OIDC provider cannot be empty."
  }
}

# =============================================================================
# S3 CONFIGURATION BACKUP
# =============================================================================

variable "config_backup_bucket" {
  description = "Name of the S3 bucket used for Kong configuration backups. The IAM policy grants GetObject, PutObject, and ListBucket on the kong/* prefix."
  type        = string

  validation {
    condition     = length(var.config_backup_bucket) > 0
    error_message = "S3 config backup bucket name cannot be empty."
  }
}

# =============================================================================
# RESOURCE QUOTAS
# =============================================================================

variable "resource_quota_cpu_requests" {
  description = "Aggregate CPU requests quota for all pods in the Kong namespace."
  type        = string
  default     = "10"
}

variable "resource_quota_memory_requests" {
  description = "Aggregate memory requests quota for all pods in the Kong namespace."
  type        = string
  default     = "20Gi"
}

variable "resource_quota_cpu_limits" {
  description = "Aggregate CPU limits quota for all pods in the Kong namespace."
  type        = string
  default     = "20"
}

variable "resource_quota_memory_limits" {
  description = "Aggregate memory limits quota for all pods in the Kong namespace."
  type        = string
  default     = "40Gi"
}

variable "resource_quota_pods" {
  description = "Maximum number of pods allowed in the Kong namespace."
  type        = string
  default     = "50"
}

# =============================================================================
# TAGGING AND METADATA
# =============================================================================

variable "tags" {
  description = "AWS tags to apply to all IAM resources created by this module."
  type        = map(string)
  default     = {}
}
