# =============================================================================
# GreenLang Grafana Module - Variables
# GreenLang Climate OS | OBS-002
# =============================================================================

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "vpc_id" {
  description = "VPC ID for security group placement"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for RDS subnet group"
  type        = list(string)
}

variable "eks_node_security_group_id" {
  description = "Security group ID of EKS worker nodes (allowed to connect to RDS)"
  type        = string
}

variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS OIDC provider for IRSA"
  type        = string
}

variable "eks_oidc_provider_url" {
  description = "URL of the EKS OIDC provider (without https://)"
  type        = string
}

# ---------------------------------------------------------------------------
# RDS Configuration
# ---------------------------------------------------------------------------

variable "db_instance_class" {
  description = "RDS instance class for Grafana backend database"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Allocated storage in GB for Grafana RDS"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum storage autoscaling limit in GB"
  type        = number
  default     = 50
}

variable "db_backup_retention_period" {
  description = "Number of days to retain automated backups"
  type        = number
  default     = 7
}

variable "db_multi_az" {
  description = "Enable Multi-AZ deployment for RDS"
  type        = bool
  default     = true
}

variable "db_deletion_protection" {
  description = "Enable deletion protection for RDS"
  type        = bool
  default     = true
}

variable "db_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

# ---------------------------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------------------------

variable "s3_bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
  default     = "gl"
}

variable "s3_image_retention_days" {
  description = "Number of days to retain rendered images in S3"
  type        = number
  default     = 30
}

# ---------------------------------------------------------------------------
# KMS Configuration
# ---------------------------------------------------------------------------

variable "kms_key_arn" {
  description = "KMS key ARN for encryption (RDS, S3). Uses AWS managed key if not provided."
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Grafana Configuration
# ---------------------------------------------------------------------------

variable "grafana_namespace" {
  description = "Kubernetes namespace where Grafana is deployed"
  type        = string
  default     = "monitoring"
}

variable "grafana_service_account_name" {
  description = "Name of the Grafana Kubernetes service account"
  type        = string
  default     = "grafana"
}

# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
