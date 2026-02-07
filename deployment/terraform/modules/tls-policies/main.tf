# =============================================================================
# GreenLang TLS Policies Module
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# -----------------------------------------------------------------------------
# Local Values
# -----------------------------------------------------------------------------
locals {
  name_prefix = "${var.project_name}-${var.environment}"

  # TLS 1.3 + TLS 1.2 fallback cipher string for NGINX
  modern_cipher_string = "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256"

  # AWS ALB/NLB SSL Policies
  # TLS 1.3 with TLS 1.2 fallback
  alb_ssl_policy_tls13 = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  # TLS 1.3 only (strict)
  alb_ssl_policy_tls13_strict = "ELBSecurityPolicy-TLS13-1-3-2021-06"
  # TLS 1.2 minimum
  alb_ssl_policy_tls12 = "ELBSecurityPolicy-TLS-1-2-2017-01"

  # Select policy based on strict mode
  selected_alb_policy = var.strict_tls_13 ? local.alb_ssl_policy_tls13_strict : local.alb_ssl_policy_tls13

  # CloudFront minimum protocol versions
  cloudfront_tls_versions = {
    "TLSv1.2" = "TLSv1.2_2021"
    "TLSv1.3" = "TLSv1.2_2021" # CloudFront negotiates to TLS 1.3 when available
  }

  common_tags = merge(var.tags, {
    Module      = "tls-policies"
    ManagedBy   = "terraform"
    PRD         = "SEC-004"
    Environment = var.environment
  })
}

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# -----------------------------------------------------------------------------
# ALB SSL Policy Reference (for documentation/output)
# -----------------------------------------------------------------------------
# Note: AWS ALB SSL policies are predefined and cannot be created as resources.
# This module provides the policy names and cipher configurations for use
# by ALB/NLB listeners.

# -----------------------------------------------------------------------------
# CloudWatch Log Group for TLS Metrics (optional)
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "tls_metrics" {
  count = var.enable_tls_logging ? 1 : 0

  name              = "/aws/${var.project_name}/${var.environment}/tls-metrics"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# SSM Parameters for TLS Configuration
# -----------------------------------------------------------------------------
resource "aws_ssm_parameter" "alb_ssl_policy" {
  name        = "/${var.project_name}/${var.environment}/tls/alb-ssl-policy"
  type        = "String"
  value       = local.selected_alb_policy
  description = "ALB SSL policy for ${var.environment}"

  tags = local.common_tags
}

resource "aws_ssm_parameter" "cipher_string" {
  name        = "/${var.project_name}/${var.environment}/tls/cipher-string"
  type        = "String"
  value       = local.modern_cipher_string
  description = "Modern TLS cipher string for ${var.environment}"

  tags = local.common_tags
}

resource "aws_ssm_parameter" "min_tls_version" {
  name        = "/${var.project_name}/${var.environment}/tls/min-version"
  type        = "String"
  value       = var.min_tls_version
  description = "Minimum TLS version for ${var.environment}"

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# HSTS Configuration SSM Parameter
# -----------------------------------------------------------------------------
resource "aws_ssm_parameter" "hsts_config" {
  name        = "/${var.project_name}/${var.environment}/tls/hsts-config"
  type        = "String"
  value = jsonencode({
    enabled            = var.hsts_enabled
    max_age            = var.hsts_max_age
    include_subdomains = var.hsts_include_subdomains
    preload            = var.hsts_preload
    header_value       = var.hsts_enabled ? "max-age=${var.hsts_max_age}${var.hsts_include_subdomains ? "; includeSubDomains" : ""}${var.hsts_preload ? "; preload" : ""}" : ""
  })
  description = "HSTS configuration for ${var.environment}"

  tags = local.common_tags
}
