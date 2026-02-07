# Shield and WAF Module Outputs - SEC-010

# -----------------------------------------------------------------------------
# WAF Outputs
# -----------------------------------------------------------------------------
output "web_acl_id" {
  description = "ID of the WAF Web ACL"
  value       = aws_wafv2_web_acl.main.id
}

output "web_acl_arn" {
  description = "ARN of the WAF Web ACL"
  value       = aws_wafv2_web_acl.main.arn
}

output "web_acl_name" {
  description = "Name of the WAF Web ACL"
  value       = aws_wafv2_web_acl.main.name
}

output "web_acl_capacity" {
  description = "Capacity units used by the Web ACL"
  value       = aws_wafv2_web_acl.main.capacity
}

# -----------------------------------------------------------------------------
# IP Set Outputs
# -----------------------------------------------------------------------------
output "blocked_ips_set_arn" {
  description = "ARN of the blocked IPs IP set (IPv4)"
  value       = aws_wafv2_ip_set.blocked_ips.arn
}

output "blocked_ips_set_id" {
  description = "ID of the blocked IPs IP set (IPv4)"
  value       = aws_wafv2_ip_set.blocked_ips.id
}

output "blocked_ips_v6_set_arn" {
  description = "ARN of the blocked IPs IP set (IPv6)"
  value       = aws_wafv2_ip_set.blocked_ips_v6.arn
}

output "blocked_ips_v6_set_id" {
  description = "ID of the blocked IPs IP set (IPv6)"
  value       = aws_wafv2_ip_set.blocked_ips_v6.id
}

# -----------------------------------------------------------------------------
# Shield Outputs
# -----------------------------------------------------------------------------
output "shield_protection_ids" {
  description = "Map of resource ARNs to Shield protection IDs"
  value = {
    for k, v in aws_shield_protection.main : k => v.id
  }
}

output "shield_protection_group_id" {
  description = "ID of the Shield protection group"
  value       = try(aws_shield_protection_group.main[0].protection_group_id, null)
}

output "shield_enabled" {
  description = "Whether Shield Advanced is enabled"
  value       = var.shield_enabled
}

# -----------------------------------------------------------------------------
# Association Outputs
# -----------------------------------------------------------------------------
output "alb_associations" {
  description = "Map of ALB ARNs to WAF association IDs"
  value = {
    for k, v in aws_wafv2_web_acl_association.alb : k => v.id
  }
}

# -----------------------------------------------------------------------------
# Logging Outputs
# -----------------------------------------------------------------------------
output "logging_configuration_id" {
  description = "ID of the WAF logging configuration"
  value       = try(aws_wafv2_web_acl_logging_configuration.main[0].id, null)
}

# -----------------------------------------------------------------------------
# CloudWatch Alarm Outputs
# -----------------------------------------------------------------------------
output "cloudwatch_alarm_arns" {
  description = "ARNs of CloudWatch alarms created for WAF"
  value = compact([
    try(aws_cloudwatch_metric_alarm.waf_blocked_requests[0].arn, ""),
    try(aws_cloudwatch_metric_alarm.waf_rate_limit[0].arn, ""),
    try(aws_cloudwatch_metric_alarm.waf_allowed_requests_drop[0].arn, ""),
  ])
}

# -----------------------------------------------------------------------------
# Summary Output
# -----------------------------------------------------------------------------
output "summary" {
  description = "Summary of WAF and Shield configuration"
  value = {
    web_acl_name              = aws_wafv2_web_acl.main.name
    web_acl_arn               = aws_wafv2_web_acl.main.arn
    scope                     = var.waf_scope
    shield_enabled            = var.shield_enabled
    rate_limit_threshold      = var.rate_limit_threshold
    blocked_countries_count   = length(var.blocked_countries)
    blocked_ips_count         = length(var.blocked_ips)
    bot_control_enabled       = var.bot_control_enabled
    protected_resources_count = length(var.protected_resources)
    logging_enabled           = var.waf_logging_enabled
    alarms_enabled            = var.enable_cloudwatch_alarms
  }
}
