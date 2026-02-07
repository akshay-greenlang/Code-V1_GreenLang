# GreenLang Shield and WAF Module - SEC-010
# Provides AWS Shield Advanced and WAF v2 protection for the platform
#
# This module creates:
# - AWS Shield Advanced protection for specified resources
# - WAF v2 Web ACL with managed and custom rules
# - Association with ALB/CloudFront
# - CloudWatch logging for WAF

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
# Local Variables
# -----------------------------------------------------------------------------
locals {
  # Common tags
  common_tags = merge(var.tags, {
    Module    = "shield-waf"
    Component = "security"
    ManagedBy = "terraform"
  })

  # Web ACL name
  web_acl_name = "${var.project_name}-${var.environment}-waf"

  # IP set name for custom blocking
  ip_set_name = "${var.project_name}-${var.environment}-blocked-ips"
}

# -----------------------------------------------------------------------------
# AWS Shield Advanced Subscription (if enabled)
# Note: Shield Advanced is a paid service with monthly commitment
# -----------------------------------------------------------------------------
resource "aws_shield_protection" "main" {
  for_each = var.shield_enabled ? toset(var.protected_resources) : []

  name         = "${local.web_acl_name}-${sha256(each.value)}"
  resource_arn = each.value

  tags = local.common_tags
}

# Create a protection group for all resources
resource "aws_shield_protection_group" "main" {
  count = var.shield_enabled && length(var.protected_resources) > 0 ? 1 : 0

  protection_group_id = "${var.project_name}-${var.environment}-protection-group"
  aggregation         = "SUM"
  pattern             = "ARBITRARY"
  members             = var.protected_resources

  tags = local.common_tags

  depends_on = [aws_shield_protection.main]
}

# Enable proactive engagement if configured
resource "aws_shield_proactive_engagement" "main" {
  count = var.shield_enabled && var.shield_proactive_engagement ? 1 : 0

  enabled = true
}

# Configure DRT access role
resource "aws_shield_drt_access_role_arn_association" "main" {
  count = var.shield_enabled && var.shield_drt_access_role_arn != "" ? 1 : 0

  role_arn = var.shield_drt_access_role_arn
}

# -----------------------------------------------------------------------------
# WAF IP Set for Custom Blocking
# -----------------------------------------------------------------------------
resource "aws_wafv2_ip_set" "blocked_ips" {
  name               = local.ip_set_name
  description        = "Blocked IP addresses for ${var.project_name}"
  scope              = var.waf_scope
  ip_address_version = "IPV4"
  addresses          = var.blocked_ips

  tags = local.common_tags
}

resource "aws_wafv2_ip_set" "blocked_ips_v6" {
  name               = "${local.ip_set_name}-v6"
  description        = "Blocked IPv6 addresses for ${var.project_name}"
  scope              = var.waf_scope
  ip_address_version = "IPV6"
  addresses          = var.blocked_ips_v6

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# WAF Web ACL
# -----------------------------------------------------------------------------
resource "aws_wafv2_web_acl" "main" {
  name        = local.web_acl_name
  description = "WAF Web ACL for ${var.project_name} ${var.environment}"
  scope       = var.waf_scope

  default_action {
    allow {}
  }

  # Token domain for CAPTCHA
  dynamic "token_domains" {
    for_each = length(var.token_domains) > 0 ? [1] : []
    content {
      # Token domains are specified at the top level, not in a block
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-${var.environment}-waf"
    sampled_requests_enabled   = var.sampled_requests_enabled
  }

  # -- Rule 1: AWS Managed Rules - Common Rule Set --
  rule {
    name     = "AWS-AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"

        dynamic "rule_action_override" {
          for_each = var.common_ruleset_excluded_rules
          content {
            name = rule_action_override.value
            action_to_use {
              count {}
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesCommonRuleSet"
      sampled_requests_enabled   = var.sampled_requests_enabled
    }
  }

  # -- Rule 2: AWS Managed Rules - Known Bad Inputs --
  rule {
    name     = "AWS-AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesKnownBadInputsRuleSet"
      sampled_requests_enabled   = var.sampled_requests_enabled
    }
  }

  # -- Rule 3: AWS Managed Rules - SQL Injection --
  rule {
    name     = "AWS-AWSManagedRulesSQLiRuleSet"
    priority = 3

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesSQLiRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesSQLiRuleSet"
      sampled_requests_enabled   = var.sampled_requests_enabled
    }
  }

  # -- Rule 4: AWS Managed Rules - Linux --
  rule {
    name     = "AWS-AWSManagedRulesLinuxRuleSet"
    priority = 4

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesLinuxRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesLinuxRuleSet"
      sampled_requests_enabled   = var.sampled_requests_enabled
    }
  }

  # -- Rule 5: AWS Managed Rules - Bot Control (if enabled) --
  dynamic "rule" {
    for_each = var.bot_control_enabled ? [1] : []
    content {
      name     = "AWS-AWSManagedRulesBotControlRuleSet"
      priority = 5

      override_action {
        none {}
      }

      statement {
        managed_rule_group_statement {
          vendor_name = "AWS"
          name        = "AWSManagedRulesBotControlRuleSet"

          managed_rule_group_configs {
            aws_managed_rules_bot_control_rule_set {
              inspection_level = var.bot_control_inspection_level
            }
          }
        }
      }

      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "AWSManagedRulesBotControlRuleSet"
        sampled_requests_enabled   = var.sampled_requests_enabled
      }
    }
  }

  # -- Rule 6: Rate Limiting per IP (PRD: 2000 requests / 5 minutes) --
  rule {
    name     = "RateLimitPerIP"
    priority = 10

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = var.rate_limit_threshold
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitPerIP"
      sampled_requests_enabled   = var.sampled_requests_enabled
    }
  }

  # -- Rule 7: Geographic Blocking --
  dynamic "rule" {
    for_each = length(var.blocked_countries) > 0 ? [1] : []
    content {
      name     = "GeoBlockRule"
      priority = 20

      action {
        block {}
      }

      statement {
        geo_match_statement {
          country_codes = var.blocked_countries
        }
      }

      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "GeoBlockRule"
        sampled_requests_enabled   = var.sampled_requests_enabled
      }
    }
  }

  # -- Rule 8: IP Reputation Blocking (Blocked IPs) --
  dynamic "rule" {
    for_each = length(var.blocked_ips) > 0 ? [1] : []
    content {
      name     = "BlockedIPsRule"
      priority = 30

      action {
        block {}
      }

      statement {
        or_statement {
          statement {
            ip_set_reference_statement {
              arn = aws_wafv2_ip_set.blocked_ips.arn
            }
          }
          statement {
            ip_set_reference_statement {
              arn = aws_wafv2_ip_set.blocked_ips_v6.arn
            }
          }
        }
      }

      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "BlockedIPsRule"
        sampled_requests_enabled   = var.sampled_requests_enabled
      }
    }
  }

  # -- Rule 9: Size Constraint - Large Body Protection --
  rule {
    name     = "SizeConstraintBody"
    priority = 40

    action {
      block {}
    }

    statement {
      size_constraint_statement {
        field_to_match {
          body {
            oversize_handling = "MATCH"
          }
        }
        comparison_operator = "GT"
        size                = var.max_body_size_bytes
        text_transformation {
          priority = 0
          type     = "NONE"
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "SizeConstraintBody"
      sampled_requests_enabled   = var.sampled_requests_enabled
    }
  }

  # -- Rule 10: Login Rate Limiting (Credential Stuffing Protection) --
  dynamic "rule" {
    for_each = var.login_endpoint_protection ? [1] : []
    content {
      name     = "LoginRateLimit"
      priority = 50

      action {
        block {}
      }

      statement {
        rate_based_statement {
          limit              = var.login_rate_limit_threshold
          aggregate_key_type = "IP"

          scope_down_statement {
            byte_match_statement {
              field_to_match {
                uri_path {}
              }
              positional_constraint = "STARTS_WITH"
              search_string         = "/api/login"
              text_transformation {
                priority = 0
                type     = "LOWERCASE"
              }
            }
          }
        }
      }

      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "LoginRateLimit"
        sampled_requests_enabled   = var.sampled_requests_enabled
      }
    }
  }

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# WAF Web ACL Association with ALB
# -----------------------------------------------------------------------------
resource "aws_wafv2_web_acl_association" "alb" {
  for_each = var.waf_scope == "REGIONAL" ? toset(var.alb_arns) : []

  resource_arn = each.value
  web_acl_arn  = aws_wafv2_web_acl.main.arn
}

# -----------------------------------------------------------------------------
# WAF Logging Configuration
# -----------------------------------------------------------------------------
resource "aws_wafv2_web_acl_logging_configuration" "main" {
  count = var.waf_logging_enabled ? 1 : 0

  log_destination_configs = var.log_destination_arns
  resource_arn            = aws_wafv2_web_acl.main.arn

  dynamic "logging_filter" {
    for_each = var.log_filter_enabled ? [1] : []
    content {
      default_behavior = "DROP"

      filter {
        behavior    = "KEEP"
        requirement = "MEETS_ANY"

        condition {
          action_condition {
            action = "BLOCK"
          }
        }

        condition {
          action_condition {
            action = "COUNT"
          }
        }
      }
    }
  }

  dynamic "redacted_fields" {
    for_each = var.redacted_fields
    content {
      dynamic "single_header" {
        for_each = redacted_fields.value.type == "single_header" ? [1] : []
        content {
          name = redacted_fields.value.name
        }
      }
    }
  }
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms for WAF
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "waf_blocked_requests" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.web_acl_name}-blocked-requests-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "BlockedRequests"
  namespace           = "AWS/WAFV2"
  period              = 300
  statistic           = "Sum"
  threshold           = var.blocked_requests_threshold
  alarm_description   = "WAF blocked requests exceed threshold"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    WebACL = local.web_acl_name
    Region = var.waf_scope == "CLOUDFRONT" ? "Global" : data.aws_region.current.name
    Rule   = "ALL"
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "waf_rate_limit" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.web_acl_name}-rate-limit-triggered"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BlockedRequests"
  namespace           = "AWS/WAFV2"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Rate limiting rule triggered frequently"
  alarm_actions       = var.alarm_actions

  dimensions = {
    WebACL = local.web_acl_name
    Region = var.waf_scope == "CLOUDFRONT" ? "Global" : data.aws_region.current.name
    Rule   = "RateLimitPerIP"
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "waf_allowed_requests_drop" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.web_acl_name}-allowed-requests-drop"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  metric_name         = "AllowedRequests"
  namespace           = "AWS/WAFV2"
  period              = 300
  statistic           = "Sum"
  threshold           = var.allowed_requests_minimum
  alarm_description   = "Allowed requests dropped significantly - possible over-blocking"
  alarm_actions       = var.alarm_actions
  treat_missing_data  = "notBreaching"

  dimensions = {
    WebACL = local.web_acl_name
    Region = var.waf_scope == "CLOUDFRONT" ? "Global" : data.aws_region.current.name
    Rule   = "ALL"
  }

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------
data "aws_region" "current" {}

data "aws_caller_identity" "current" {}
