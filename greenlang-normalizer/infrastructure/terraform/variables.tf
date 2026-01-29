# GreenLang Normalizer - Terraform Variables

variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "greenlang-prod"
}

variable "normalizer_replicas" {
  description = "Number of normalizer service replicas"
  type        = number
  default     = 3

  validation {
    condition     = var.normalizer_replicas >= 1 && var.normalizer_replicas <= 50
    error_message = "Replicas must be between 1 and 50."
  }
}

variable "normalizer_image" {
  description = "Docker image for normalizer service"
  type        = string
  default     = "greenlang/gl-normalizer-service:latest"
}

variable "normalizer_cpu_request" {
  description = "CPU request for normalizer pods"
  type        = string
  default     = "250m"
}

variable "normalizer_memory_request" {
  description = "Memory request for normalizer pods"
  type        = string
  default     = "512Mi"
}

variable "normalizer_cpu_limit" {
  description = "CPU limit for normalizer pods"
  type        = string
  default     = "2000m"
}

variable "normalizer_memory_limit" {
  description = "Memory limit for normalizer pods"
  type        = string
  default     = "2Gi"
}

variable "redis_enabled" {
  description = "Enable Redis for caching"
  type        = bool
  default     = true
}

variable "redis_storage_size" {
  description = "Storage size for Redis"
  type        = string
  default     = "10Gi"
}

variable "postgresql_enabled" {
  description = "Enable PostgreSQL for persistence"
  type        = bool
  default     = true
}

variable "postgresql_storage_size" {
  description = "Storage size for PostgreSQL"
  type        = string
  default     = "50Gi"
}

variable "monitoring_enabled" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "ingress_enabled" {
  description = "Enable ingress for external access"
  type        = bool
  default     = true
}

variable "ingress_host" {
  description = "Hostname for ingress"
  type        = string
  default     = "normalizer.greenlang.io"
}

variable "tls_enabled" {
  description = "Enable TLS for ingress"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
