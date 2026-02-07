# =============================================================================
# GreenLang Prometheus Stack Module - Variables
# GreenLang Climate OS | OBS-001
# =============================================================================
# Input variables for configuring the Prometheus stack deployment including
# Prometheus HA, Thanos long-term storage, Alertmanager, and PushGateway.
# =============================================================================

# -----------------------------------------------------------------------------
# Required Variables
# -----------------------------------------------------------------------------

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "cluster_name" {
  description = "EKS cluster name for IRSA configuration"
  type        = string
}

# -----------------------------------------------------------------------------
# Project Configuration
# -----------------------------------------------------------------------------

variable "project" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "gl"
}

variable "namespace" {
  description = "Kubernetes namespace for Prometheus stack"
  type        = string
  default     = "monitoring"
}

variable "aws_region" {
  description = "AWS region for S3 bucket and IAM resources"
  type        = string
  default     = "eu-west-1"
}

# -----------------------------------------------------------------------------
# Prometheus Configuration
# -----------------------------------------------------------------------------

variable "prometheus_replica_count" {
  description = "Number of Prometheus server replicas for HA"
  type        = number
  default     = 2

  validation {
    condition     = var.prometheus_replica_count >= 1 && var.prometheus_replica_count <= 10
    error_message = "Prometheus replica count must be between 1 and 10."
  }
}

variable "prometheus_retention_days" {
  description = "Local data retention period in days (Thanos handles long-term)"
  type        = number
  default     = 7

  validation {
    condition     = var.prometheus_retention_days >= 1 && var.prometheus_retention_days <= 30
    error_message = "Prometheus retention must be between 1 and 30 days."
  }
}

variable "prometheus_storage_size" {
  description = "PVC storage size for each Prometheus replica"
  type        = string
  default     = "50Gi"
}

variable "prometheus_storage_class" {
  description = "Kubernetes StorageClass for Prometheus PVCs"
  type        = string
  default     = "gp3"
}

variable "prometheus_resources" {
  description = "Resource requests and limits for Prometheus pods"
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
      cpu    = "500m"
      memory = "2Gi"
    }
    limits = {
      cpu    = "2000m"
      memory = "8Gi"
    }
  }
}

# -----------------------------------------------------------------------------
# Thanos Configuration
# -----------------------------------------------------------------------------

variable "enable_thanos" {
  description = "Enable Thanos for long-term storage and HA query"
  type        = bool
  default     = true
}

variable "thanos_retention_raw" {
  description = "Retention period for raw metrics in S3 (e.g., 30d)"
  type        = string
  default     = "30d"
}

variable "thanos_retention_5m" {
  description = "Retention period for 5-minute downsampled metrics (e.g., 120d)"
  type        = string
  default     = "120d"
}

variable "thanos_retention_1h" {
  description = "Retention period for 1-hour downsampled metrics (e.g., 730d for 2 years)"
  type        = string
  default     = "730d"
}

variable "thanos_query_replicas" {
  description = "Number of Thanos Query replicas"
  type        = number
  default     = 2
}

variable "thanos_store_gateway_replicas" {
  description = "Number of Thanos Store Gateway replicas"
  type        = number
  default     = 2
}

variable "thanos_ruler_replicas" {
  description = "Number of Thanos Ruler replicas"
  type        = number
  default     = 2
}

variable "thanos_compactor_replicas" {
  description = "Number of Thanos Compactor replicas (should be 1 for consistency)"
  type        = number
  default     = 1

  validation {
    condition     = var.thanos_compactor_replicas == 1
    error_message = "Thanos Compactor must have exactly 1 replica to avoid conflicts."
  }
}

variable "thanos_store_gateway_storage_size" {
  description = "PVC storage size for Thanos Store Gateway cache"
  type        = string
  default     = "50Gi"
}

variable "thanos_compactor_storage_size" {
  description = "PVC storage size for Thanos Compactor working directory"
  type        = string
  default     = "100Gi"
}

# -----------------------------------------------------------------------------
# Alertmanager Configuration
# -----------------------------------------------------------------------------

variable "alertmanager_replica_count" {
  description = "Number of Alertmanager replicas for HA"
  type        = number
  default     = 2
}

variable "alertmanager_storage_size" {
  description = "PVC storage size for Alertmanager"
  type        = string
  default     = "10Gi"
}

variable "alertmanager_slack_webhook" {
  description = "Slack webhook URL for alert notifications"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alertmanager_pagerduty_key" {
  description = "PagerDuty integration key for critical alerts"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alertmanager_slack_channel" {
  description = "Default Slack channel for alerts"
  type        = string
  default     = "#greenlang-alerts"
}

variable "alertmanager_slack_warning_channel" {
  description = "Slack channel for warning-level alerts"
  type        = string
  default     = "#greenlang-alerts-warning"
}

# -----------------------------------------------------------------------------
# PushGateway Configuration
# -----------------------------------------------------------------------------

variable "enable_pushgateway" {
  description = "Enable PushGateway for batch job metrics"
  type        = bool
  default     = true
}

variable "pushgateway_replica_count" {
  description = "Number of PushGateway replicas"
  type        = number
  default     = 2
}

variable "pushgateway_storage_size" {
  description = "PVC storage size for PushGateway"
  type        = string
  default     = "2Gi"
}

# -----------------------------------------------------------------------------
# Helm Chart Versions
# -----------------------------------------------------------------------------

variable "kube_prometheus_stack_chart_version" {
  description = "Version of the kube-prometheus-stack Helm chart"
  type        = string
  default     = "56.21.4"
}

variable "thanos_chart_version" {
  description = "Version of the Bitnami Thanos Helm chart"
  type        = string
  default     = "15.7.10"
}

variable "pushgateway_chart_version" {
  description = "Version of the Prometheus PushGateway Helm chart"
  type        = string
  default     = "2.8.0"
}

# -----------------------------------------------------------------------------
# IRSA Configuration
# -----------------------------------------------------------------------------

variable "create_iam_role" {
  description = "Create IAM role for IRSA. Set to false if using existing role."
  type        = bool
  default     = true
}

variable "existing_iam_role_arn" {
  description = "ARN of existing IAM role for IRSA (if create_iam_role is false)"
  type        = string
  default     = ""
}

variable "oidc_provider" {
  description = "EKS OIDC provider URL without https:// prefix"
  type        = string
  default     = ""
}

variable "oidc_provider_arn" {
  description = "EKS OIDC provider ARN"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Scrape Configuration
# -----------------------------------------------------------------------------

variable "additional_scrape_configs" {
  description = "Additional Prometheus scrape configurations"
  type        = list(any)
  default     = []
}

variable "greenlang_namespaces" {
  description = "List of GreenLang namespaces to monitor"
  type        = list(string)
  default     = ["greenlang", "gl-agents", "gl-fuel", "gl-cbam", "gl-building"]
}

# -----------------------------------------------------------------------------
# S3 Configuration
# -----------------------------------------------------------------------------

variable "thanos_bucket_name" {
  description = "Override for Thanos S3 bucket name. If empty, uses gl-thanos-metrics-{environment}"
  type        = string
  default     = ""
}

variable "kms_key_arn" {
  description = "Optional KMS key ARN for S3 encryption. If not provided, AES256 is used."
  type        = string
  default     = null
}

variable "audit_logs_bucket" {
  description = "S3 bucket name for access logging. Set to null to disable."
  type        = string
  default     = null
}

# -----------------------------------------------------------------------------
# Tags
# -----------------------------------------------------------------------------

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "GreenLang"
    Component = "prometheus"
    ManagedBy = "terraform"
    OBS       = "001"
  }
}
