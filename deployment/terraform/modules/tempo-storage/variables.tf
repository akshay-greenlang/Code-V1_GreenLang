# =============================================================================
# GreenLang Tempo Storage Module - Variables
# GreenLang Climate OS | OBS-003
# =============================================================================

# -----------------------------------------------------------------------------
# General
# -----------------------------------------------------------------------------

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

variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "eu-west-1"
}

# -----------------------------------------------------------------------------
# EKS / IRSA
# -----------------------------------------------------------------------------

variable "eks_cluster_name" {
  description = "Name of the EKS cluster (used for resource naming)"
  type        = string
}

variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS cluster OIDC provider (for IRSA trust policy)"
  type        = string
}

variable "eks_oidc_provider_url" {
  description = "URL of the EKS cluster OIDC provider (without https:// prefix)"
  type        = string
}

variable "tempo_namespace" {
  description = "Kubernetes namespace where Tempo is deployed"
  type        = string
  default     = "monitoring"
}

variable "tempo_service_account" {
  description = "Kubernetes service account name for Tempo pods"
  type        = string
  default     = "tempo"
}

# -----------------------------------------------------------------------------
# Lifecycle / Retention
# -----------------------------------------------------------------------------

variable "retention_days" {
  description = "Number of days to retain trace data before expiration"
  type        = number
  default     = 90

  validation {
    condition     = var.retention_days >= 1 && var.retention_days <= 365
    error_message = "Retention days must be between 1 and 365."
  }
}

variable "ia_transition_days" {
  description = "Number of days before transitioning trace blocks to S3 Infrequent Access"
  type        = number
  default     = 30

  validation {
    condition     = var.ia_transition_days >= 1
    error_message = "IA transition days must be at least 1."
  }
}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

variable "log_bucket_name" {
  description = "S3 bucket name for access logging. Set to empty string to disable access logging."
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Tags
# -----------------------------------------------------------------------------

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "GreenLang"
    Component = "Tempo"
    ManagedBy = "terraform"
    OBS       = "003"
  }
}
