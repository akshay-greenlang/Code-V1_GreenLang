# =============================================================================
# GreenLang Loki Storage Module - Variables
# GreenLang Climate OS | INFRA-009
# =============================================================================

variable "project" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "gl"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "kms_key_arn" {
  description = "Optional KMS key ARN for S3 encryption. If not provided, a new KMS key is created."
  type        = string
  default     = null
}

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "GreenLang"
    Component = "loki"
    ManagedBy = "terraform"
    INFRA     = "009"
  }
}

variable "audit_logs_bucket" {
  description = "S3 bucket name for access logging. Set to null to disable access logging."
  type        = string
  default     = null
}

# -----------------------------------------------------------------------------
# IRSA variables (used in iam.tf)
# -----------------------------------------------------------------------------

variable "eks_cluster_oidc_issuer" {
  description = "EKS cluster OIDC issuer URL (without https:// prefix)"
  type        = string
}

variable "loki_service_account_name" {
  description = "Kubernetes service account name for Loki"
  type        = string
  default     = "loki"
}

variable "loki_namespace" {
  description = "Kubernetes namespace where Loki is deployed"
  type        = string
  default     = "monitoring"
}
