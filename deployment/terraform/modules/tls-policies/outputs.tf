# =============================================================================
# TLS Policies Module - Outputs
# =============================================================================

# -----------------------------------------------------------------------------
# ALB/NLB SSL Policy Outputs
# -----------------------------------------------------------------------------
output "alb_ssl_policy_name" {
  description = "AWS ALB SSL policy name to use for listeners"
  value       = local.selected_alb_policy
}

output "nlb_ssl_policy_name" {
  description = "AWS NLB SSL policy name to use for TLS listeners"
  value       = local.selected_alb_policy
}

output "alb_ssl_policy_tls13" {
  description = "TLS 1.3 + TLS 1.2 fallback policy name"
  value       = local.alb_ssl_policy_tls13
}

output "alb_ssl_policy_tls13_strict" {
  description = "TLS 1.3 only (strict) policy name"
  value       = local.alb_ssl_policy_tls13_strict
}

output "alb_ssl_policy_tls12" {
  description = "TLS 1.2 minimum policy name"
  value       = local.alb_ssl_policy_tls12
}

# -----------------------------------------------------------------------------
# CloudFront Outputs
# -----------------------------------------------------------------------------
output "cloudfront_min_protocol_version" {
  description = "CloudFront minimum protocol version"
  value       = local.cloudfront_tls_versions[var.min_tls_version]
}

# -----------------------------------------------------------------------------
# Cipher Configuration Outputs
# -----------------------------------------------------------------------------
output "cipher_suite_string" {
  description = "Modern cipher suite string for NGINX/Kong"
  value       = local.modern_cipher_string
}

output "cipher_suite_list" {
  description = "List of allowed cipher suites"
  value       = split(":", local.modern_cipher_string)
}

# -----------------------------------------------------------------------------
# HSTS Outputs
# -----------------------------------------------------------------------------
output "hsts_header_value" {
  description = "Complete HSTS header value"
  value       = var.hsts_enabled ? "max-age=${var.hsts_max_age}${var.hsts_include_subdomains ? "; includeSubDomains" : ""}${var.hsts_preload ? "; preload" : ""}" : ""
}

output "hsts_enabled" {
  description = "Whether HSTS is enabled"
  value       = var.hsts_enabled
}

# -----------------------------------------------------------------------------
# SSM Parameter ARNs
# -----------------------------------------------------------------------------
output "ssm_alb_ssl_policy_arn" {
  description = "SSM parameter ARN for ALB SSL policy"
  value       = aws_ssm_parameter.alb_ssl_policy.arn
}

output "ssm_cipher_string_arn" {
  description = "SSM parameter ARN for cipher string"
  value       = aws_ssm_parameter.cipher_string.arn
}

output "ssm_min_tls_version_arn" {
  description = "SSM parameter ARN for minimum TLS version"
  value       = aws_ssm_parameter.min_tls_version.arn
}

# -----------------------------------------------------------------------------
# Configuration Summary
# -----------------------------------------------------------------------------
output "tls_config_summary" {
  description = "Summary of TLS configuration"
  value = {
    min_tls_version = var.min_tls_version
    strict_tls_13   = var.strict_tls_13
    alb_policy      = local.selected_alb_policy
    hsts_enabled    = var.hsts_enabled
    hsts_max_age    = var.hsts_max_age
    hsts_preload    = var.hsts_preload
  }
}
