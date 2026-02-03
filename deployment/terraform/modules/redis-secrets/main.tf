#------------------------------------------------------------------------------
# Redis Secrets Manager Module
# Manages Redis authentication credentials with automatic rotation
#------------------------------------------------------------------------------

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

#------------------------------------------------------------------------------
# KMS Key for Secret Encryption
#------------------------------------------------------------------------------

resource "aws_kms_key" "redis_secrets" {
  count = var.kms_key_arn == null ? 1 : 0

  description             = "KMS key for Redis secrets encryption"
  deletion_window_in_days = var.kms_deletion_window_days
  enable_key_rotation     = true

  tags = merge(var.tags, {
    Name    = "${var.secret_name_prefix}-redis-kms"
    Purpose = "Redis secrets encryption"
  })
}

resource "aws_kms_alias" "redis_secrets" {
  count = var.kms_key_arn == null ? 1 : 0

  name          = "alias/${var.secret_name_prefix}-redis-secrets"
  target_key_id = aws_kms_key.redis_secrets[0].key_id
}

locals {
  kms_key_arn = var.kms_key_arn != null ? var.kms_key_arn : aws_kms_key.redis_secrets[0].arn
  kms_key_id  = var.kms_key_arn != null ? var.kms_key_arn : aws_kms_key.redis_secrets[0].key_id
}

#------------------------------------------------------------------------------
# Random Password Generation
#------------------------------------------------------------------------------

resource "random_password" "redis_auth" {
  length           = var.password_length
  special          = var.password_special_chars
  override_special = var.password_override_special
  min_lower        = var.password_min_lower
  min_upper        = var.password_min_upper
  min_numeric      = var.password_min_numeric
  min_special      = var.password_min_special

  lifecycle {
    # Password should only change during rotation, not on every apply
    ignore_changes = all
  }
}

#------------------------------------------------------------------------------
# Secrets Manager Secret
#------------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "redis_auth" {
  name        = "${var.secret_name_prefix}/redis/auth"
  description = "Redis AUTH password for ${var.secret_name_prefix}"
  kms_key_id  = local.kms_key_id

  recovery_window_in_days = var.secret_recovery_window_days

  tags = merge(var.tags, {
    Name        = "${var.secret_name_prefix}-redis-auth"
    Purpose     = "Redis authentication"
    ManagedBy   = "Terraform"
    Application = var.secret_name_prefix
  })
}

resource "aws_secretsmanager_secret_version" "redis_auth" {
  secret_id = aws_secretsmanager_secret.redis_auth.id
  secret_string = jsonencode({
    password       = random_password.redis_auth.result
    username       = var.redis_username
    host           = var.redis_host
    port           = var.redis_port
    ssl_enabled    = var.redis_ssl_enabled
    connection_url = var.redis_ssl_enabled ? "rediss://${var.redis_username}:${random_password.redis_auth.result}@${var.redis_host}:${var.redis_port}" : "redis://${var.redis_username}:${random_password.redis_auth.result}@${var.redis_host}:${var.redis_port}"
    created_at     = timestamp()
    rotated_at     = timestamp()
  })

  lifecycle {
    # Version ID changes should not trigger updates
    ignore_changes = [secret_string]
  }
}

#------------------------------------------------------------------------------
# Secret Rotation Configuration
#------------------------------------------------------------------------------

resource "aws_secretsmanager_secret_rotation" "redis_auth" {
  count = var.enable_rotation ? 1 : 0

  secret_id           = aws_secretsmanager_secret.redis_auth.id
  rotation_lambda_arn = var.rotation_lambda_arn

  rotation_rules {
    automatically_after_days = var.rotation_days
    schedule_expression      = var.rotation_schedule_expression
  }

  depends_on = [aws_secretsmanager_secret_version.redis_auth]
}

#------------------------------------------------------------------------------
# CloudWatch Alarms for Secret Rotation
#------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "rotation_failure" {
  count = var.enable_rotation && var.enable_rotation_alerts ? 1 : 0

  alarm_name          = "${var.secret_name_prefix}-redis-rotation-failure"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "RotationFailed"
  namespace           = "AWS/SecretsManager"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Redis secret rotation failed for ${var.secret_name_prefix}"

  dimensions = {
    SecretId = aws_secretsmanager_secret.redis_auth.id
  }

  alarm_actions = var.alarm_sns_topic_arns
  ok_actions    = var.alarm_sns_topic_arns

  tags = var.tags
}

#------------------------------------------------------------------------------
# Resource Policy for Cross-Account Access (Optional)
#------------------------------------------------------------------------------

resource "aws_secretsmanager_secret_policy" "redis_auth" {
  count = length(var.allowed_principal_arns) > 0 ? 1 : 0

  secret_arn = aws_secretsmanager_secret.redis_auth.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCrossAccountAccess"
        Effect = "Allow"
        Principal = {
          AWS = var.allowed_principal_arns
        }
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "secretsmanager:VersionStage" = "AWSCURRENT"
          }
        }
      }
    ]
  })
}
