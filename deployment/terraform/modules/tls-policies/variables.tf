# =============================================================================
# TLS Policies Module - Variables
# =============================================================================

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "greenlang"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# -----------------------------------------------------------------------------
# TLS Version Configuration
# -----------------------------------------------------------------------------
variable "min_tls_version" {
  description = "Minimum TLS version to allow"
  type        = string
  default     = "TLSv1.2"
  validation {
    condition     = contains(["TLSv1.2", "TLSv1.3"], var.min_tls_version)
    error_message = "Minimum TLS version must be TLSv1.2 or TLSv1.3."
  }
}

variable "strict_tls_13" {
  description = "Enforce TLS 1.3 only (no TLS 1.2 fallback)"
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# Cipher Configuration
# -----------------------------------------------------------------------------
variable "allowed_ciphers" {
  description = "List of allowed cipher suites (overrides default)"
  type        = list(string)
  default     = []
}

variable "disable_weak_ciphers" {
  description = "Explicitly disable known weak ciphers"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# HSTS Configuration
# -----------------------------------------------------------------------------
variable "hsts_enabled" {
  description = "Enable HTTP Strict Transport Security"
  type        = bool
  default     = true
}

variable "hsts_max_age" {
  description = "HSTS max-age in seconds (default 1 year)"
  type        = number
  default     = 31536000
}

variable "hsts_include_subdomains" {
  description = "Include subdomains in HSTS policy"
  type        = bool
  default     = true
}

variable "hsts_preload" {
  description = "Enable HSTS preload (requires registration)"
  type        = bool
  default     = false # Enable only after testing
}

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
variable "enable_tls_logging" {
  description = "Enable CloudWatch logging for TLS metrics"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# -----------------------------------------------------------------------------
# ACM Certificate Configuration
# -----------------------------------------------------------------------------
variable "create_acm_certificates" {
  description = "Whether to create ACM certificates for the domain"
  type        = bool
  default     = true
}

variable "domain_name" {
  description = "Primary domain name for certificates (e.g., greenlang.io)"
  type        = string
  default     = "greenlang.io"
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID for DNS validation"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Tags
# -----------------------------------------------------------------------------
variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
