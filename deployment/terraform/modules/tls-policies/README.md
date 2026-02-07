# TLS Policies Terraform Module

SEC-004: TLS 1.3 Configuration for All Services

This module provides centralized TLS policy configuration for GreenLang infrastructure, ensuring consistent TLS 1.3 enforcement across all AWS services and Kubernetes ingress controllers.

## Features

- AWS ALB/NLB SSL policy management (TLS 1.3 with TLS 1.2 fallback)
- Modern cipher suite configuration for NGINX and Kong
- HSTS (HTTP Strict Transport Security) configuration
- SSM Parameter Store integration for dynamic configuration
- CloudFront minimum protocol version settings
- CloudWatch logging for TLS metrics

## Usage

### Basic Usage

```hcl
module "tls_policies" {
  source = "../modules/tls-policies"

  project_name = "greenlang"
  environment  = "prod"

  tags = {
    Team = "platform"
  }
}
```

### Production Configuration (Strict TLS 1.3)

```hcl
module "tls_policies" {
  source = "../modules/tls-policies"

  project_name  = "greenlang"
  environment   = "prod"
  strict_tls_13 = true  # No TLS 1.2 fallback

  # HSTS configuration
  hsts_enabled            = true
  hsts_max_age            = 31536000  # 1 year
  hsts_include_subdomains = true
  hsts_preload            = true  # Enable after testing

  # Logging
  enable_tls_logging = true
  log_retention_days = 90

  tags = {
    Team        = "platform"
    Compliance  = "SOC2"
    Environment = "prod"
  }
}
```

### Development Configuration (TLS 1.2 Fallback)

```hcl
module "tls_policies" {
  source = "../modules/tls-policies"

  project_name  = "greenlang"
  environment   = "dev"
  strict_tls_13 = false  # Allow TLS 1.2 for compatibility

  # Relaxed HSTS for development
  hsts_enabled = false

  # Minimal logging
  enable_tls_logging = true
  log_retention_days = 7

  tags = {
    Team        = "platform"
    Environment = "dev"
  }
}
```

### Using Outputs with ALB Listener

```hcl
module "tls_policies" {
  source      = "../modules/tls-policies"
  environment = "prod"
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = module.tls_policies.alb_ssl_policy_name
  certificate_arn   = aws_acm_certificate.main.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.main.arn
  }
}
```

### Using Outputs with CloudFront Distribution

```hcl
module "tls_policies" {
  source      = "../modules/tls-policies"
  environment = "prod"
}

resource "aws_cloudfront_distribution" "main" {
  # ... other configuration ...

  viewer_certificate {
    acm_certificate_arn            = aws_acm_certificate.main.arn
    ssl_support_method             = "sni-only"
    minimum_protocol_version       = module.tls_policies.cloudfront_min_protocol_version
  }
}
```

### Using Cipher String with NGINX ConfigMap

```hcl
module "tls_policies" {
  source      = "../modules/tls-policies"
  environment = "prod"
}

resource "kubernetes_config_map" "nginx_tls" {
  metadata {
    name      = "nginx-tls-config"
    namespace = "ingress-nginx"
  }

  data = {
    ssl-protocols = "TLSv1.2 TLSv1.3"
    ssl-ciphers   = module.tls_policies.cipher_suite_string
  }
}
```

## Requirements

| Name | Version |
|------|---------|
| terraform | >= 1.0 |
| aws | >= 5.0 |

## Providers

| Name | Version |
|------|---------|
| aws | >= 5.0 |

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| project_name | Project name for resource naming | `string` | `"greenlang"` | no |
| environment | Environment name (dev, staging, prod) | `string` | n/a | yes |
| min_tls_version | Minimum TLS version to allow | `string` | `"TLSv1.2"` | no |
| strict_tls_13 | Enforce TLS 1.3 only (no TLS 1.2 fallback) | `bool` | `false` | no |
| allowed_ciphers | List of allowed cipher suites (overrides default) | `list(string)` | `[]` | no |
| disable_weak_ciphers | Explicitly disable known weak ciphers | `bool` | `true` | no |
| hsts_enabled | Enable HTTP Strict Transport Security | `bool` | `true` | no |
| hsts_max_age | HSTS max-age in seconds | `number` | `31536000` | no |
| hsts_include_subdomains | Include subdomains in HSTS policy | `bool` | `true` | no |
| hsts_preload | Enable HSTS preload | `bool` | `false` | no |
| enable_tls_logging | Enable CloudWatch logging for TLS metrics | `bool` | `true` | no |
| log_retention_days | CloudWatch log retention in days | `number` | `30` | no |
| tags | Tags to apply to all resources | `map(string)` | `{}` | no |

## Outputs

| Name | Description |
|------|-------------|
| alb_ssl_policy_name | AWS ALB SSL policy name to use for listeners |
| nlb_ssl_policy_name | AWS NLB SSL policy name to use for TLS listeners |
| alb_ssl_policy_tls13 | TLS 1.3 + TLS 1.2 fallback policy name |
| alb_ssl_policy_tls13_strict | TLS 1.3 only (strict) policy name |
| alb_ssl_policy_tls12 | TLS 1.2 minimum policy name |
| cloudfront_min_protocol_version | CloudFront minimum protocol version |
| cipher_suite_string | Modern cipher suite string for NGINX/Kong |
| cipher_suite_list | List of allowed cipher suites |
| hsts_header_value | Complete HSTS header value |
| hsts_enabled | Whether HSTS is enabled |
| ssm_alb_ssl_policy_arn | SSM parameter ARN for ALB SSL policy |
| ssm_cipher_string_arn | SSM parameter ARN for cipher string |
| ssm_min_tls_version_arn | SSM parameter ARN for minimum TLS version |
| tls_config_summary | Summary of TLS configuration |

## SSL Policies Reference

### AWS ALB/NLB Policies

| Policy Name | TLS Versions | Use Case |
|-------------|--------------|----------|
| ELBSecurityPolicy-TLS13-1-2-2021-06 | TLS 1.2, TLS 1.3 | Default (compatibility) |
| ELBSecurityPolicy-TLS13-1-3-2021-06 | TLS 1.3 only | Strict (modern clients) |
| ELBSecurityPolicy-TLS-1-2-2017-01 | TLS 1.2 only | Legacy compatibility |

### Cipher Suites

The module configures these modern cipher suites:

**TLS 1.3 Ciphers:**
- TLS_AES_256_GCM_SHA384
- TLS_CHACHA20_POLY1305_SHA256
- TLS_AES_128_GCM_SHA256

**TLS 1.2 ECDHE Ciphers (fallback):**
- ECDHE-ECDSA-AES256-GCM-SHA384
- ECDHE-RSA-AES256-GCM-SHA384
- ECDHE-ECDSA-AES128-GCM-SHA256
- ECDHE-RSA-AES128-GCM-SHA256

## Security Considerations

1. **Production environments** should use `strict_tls_13 = true` when all clients support TLS 1.3
2. **HSTS preload** requires domain registration at hstspreload.org
3. **Testing** should verify TLS configuration using tools like SSL Labs or testssl.sh
4. **Monitoring** TLS version distribution helps identify clients needing TLS 1.2 fallback

## Compliance

This module helps meet the following compliance requirements:

- **PCI DSS 4.0**: Requirement 4.2.1 - Strong cryptography for transmission
- **SOC 2**: CC6.7 - Encryption in transit
- **ISO 27001**: A.10.1.1 - Cryptographic controls
- **NIST 800-53**: SC-8 - Transmission confidentiality and integrity

## Related Documentation

- [AWS ALB SSL Policies](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/create-https-listener.html#describe-ssl-policies)
- [CloudFront Security Policy](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/secure-connections-supported-viewer-protocols-ciphers.html)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
