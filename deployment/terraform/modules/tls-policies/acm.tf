# =============================================================================
# AWS ACM Certificate Management
# SEC-004: TLS 1.3 Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# ACM Certificate for API Domain
# -----------------------------------------------------------------------------
resource "aws_acm_certificate" "api" {
  count = var.create_acm_certificates ? 1 : 0

  domain_name       = "api.${var.domain_name}"
  validation_method = "DNS"

  subject_alternative_names = [
    "*.api.${var.domain_name}",
  ]

  options {
    certificate_transparency_logging_preference = "ENABLED"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-api-cert"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# -----------------------------------------------------------------------------
# ACM Certificate for Main Domain (Wildcard)
# -----------------------------------------------------------------------------
resource "aws_acm_certificate" "wildcard" {
  count = var.create_acm_certificates ? 1 : 0

  domain_name       = var.domain_name
  validation_method = "DNS"

  subject_alternative_names = [
    "*.${var.domain_name}",
  ]

  options {
    certificate_transparency_logging_preference = "ENABLED"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-wildcard-cert"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# -----------------------------------------------------------------------------
# DNS Validation Records
# -----------------------------------------------------------------------------
resource "aws_route53_record" "api_validation" {
  for_each = var.create_acm_certificates ? {
    for dvo in aws_acm_certificate.api[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = var.route53_zone_id
}

resource "aws_route53_record" "wildcard_validation" {
  for_each = var.create_acm_certificates ? {
    for dvo in aws_acm_certificate.wildcard[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = var.route53_zone_id
}

# -----------------------------------------------------------------------------
# Certificate Validation
# -----------------------------------------------------------------------------
resource "aws_acm_certificate_validation" "api" {
  count = var.create_acm_certificates ? 1 : 0

  certificate_arn         = aws_acm_certificate.api[0].arn
  validation_record_fqdns = [for record in aws_route53_record.api_validation : record.fqdn]
}

resource "aws_acm_certificate_validation" "wildcard" {
  count = var.create_acm_certificates ? 1 : 0

  certificate_arn         = aws_acm_certificate.wildcard[0].arn
  validation_record_fqdns = [for record in aws_route53_record.wildcard_validation : record.fqdn]
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
output "acm_api_certificate_arn" {
  description = "ARN of the API ACM certificate"
  value       = var.create_acm_certificates ? aws_acm_certificate.api[0].arn : null
}

output "acm_wildcard_certificate_arn" {
  description = "ARN of the wildcard ACM certificate"
  value       = var.create_acm_certificates ? aws_acm_certificate.wildcard[0].arn : null
}

output "acm_api_certificate_status" {
  description = "Status of the API ACM certificate validation"
  value       = var.create_acm_certificates ? aws_acm_certificate_validation.api[0].certificate_arn : null
}
