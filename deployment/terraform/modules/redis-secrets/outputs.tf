#------------------------------------------------------------------------------
# Redis Secrets Module Outputs
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Secret Outputs
#------------------------------------------------------------------------------

output "secret_arn" {
  description = "ARN of the Redis authentication secret"
  value       = aws_secretsmanager_secret.redis_auth.arn
}

output "secret_id" {
  description = "ID of the Redis authentication secret"
  value       = aws_secretsmanager_secret.redis_auth.id
}

output "secret_name" {
  description = "Name of the Redis authentication secret"
  value       = aws_secretsmanager_secret.redis_auth.name
}

output "secret_version_id" {
  description = "Version ID of the current secret"
  value       = aws_secretsmanager_secret_version.redis_auth.version_id
}

#------------------------------------------------------------------------------
# KMS Key Outputs
#------------------------------------------------------------------------------

output "kms_key_arn" {
  description = "ARN of the KMS key used for encryption"
  value       = local.kms_key_arn
}

output "kms_key_id" {
  description = "ID of the KMS key used for encryption"
  value       = local.kms_key_id
}

output "kms_key_alias" {
  description = "Alias of the KMS key (if created by this module)"
  value       = var.kms_key_arn == null ? aws_kms_alias.redis_secrets[0].name : null
}

#------------------------------------------------------------------------------
# IAM Outputs
#------------------------------------------------------------------------------

output "iam_role_arn" {
  description = "ARN of the IAM role for EKS pods (IRSA)"
  value       = var.eks_oidc_provider_arn != "" ? aws_iam_role.eks_redis_secrets[0].arn : null
}

output "iam_role_name" {
  description = "Name of the IAM role for EKS pods"
  value       = var.eks_oidc_provider_arn != "" ? aws_iam_role.eks_redis_secrets[0].name : null
}

output "iam_policy_arn" {
  description = "ARN of the IAM policy for secret access"
  value       = aws_iam_policy.redis_secrets_access.arn
}

output "iam_policy_name" {
  description = "Name of the IAM policy for secret access"
  value       = aws_iam_policy.redis_secrets_access.name
}

#------------------------------------------------------------------------------
# Service Account Annotation
#------------------------------------------------------------------------------

output "service_account_annotation" {
  description = "Annotation to add to the Kubernetes service account for IRSA"
  value       = var.eks_oidc_provider_arn != "" ? "eks.amazonaws.com/role-arn: ${aws_iam_role.eks_redis_secrets[0].arn}" : null
}

#------------------------------------------------------------------------------
# Connection Info (Non-sensitive)
#------------------------------------------------------------------------------

output "redis_host" {
  description = "Redis host endpoint"
  value       = var.redis_host
}

output "redis_port" {
  description = "Redis port"
  value       = var.redis_port
}

output "redis_ssl_enabled" {
  description = "Whether SSL is enabled for Redis"
  value       = var.redis_ssl_enabled
}

#------------------------------------------------------------------------------
# Rotation Info
#------------------------------------------------------------------------------

output "rotation_enabled" {
  description = "Whether automatic rotation is enabled"
  value       = var.enable_rotation
}

output "rotation_schedule" {
  description = "Rotation schedule (days or cron expression)"
  value       = var.rotation_schedule_expression != null ? var.rotation_schedule_expression : "${var.rotation_days} days"
}

#------------------------------------------------------------------------------
# External Secrets Integration
#------------------------------------------------------------------------------

output "external_secret_store_config" {
  description = "Configuration for External Secrets ClusterSecretStore"
  value = {
    provider = "aws"
    service  = "SecretsManager"
    region   = data.aws_region.current.name
    auth = {
      jwt = {
        serviceAccountRef = {
          name      = var.eks_service_account_name
          namespace = var.eks_namespace
        }
      }
    }
  }
}

output "external_secret_config" {
  description = "Configuration for External Secrets ExternalSecret"
  value = {
    secretStoreRef = {
      kind = "ClusterSecretStore"
      name = "${var.secret_name_prefix}-redis-store"
    }
    target = {
      name = "redis-credentials"
    }
    data = [
      {
        secretKey = "password"
        remoteRef = {
          key      = aws_secretsmanager_secret.redis_auth.name
          property = "password"
        }
      },
      {
        secretKey = "connection_url"
        remoteRef = {
          key      = aws_secretsmanager_secret.redis_auth.name
          property = "connection_url"
        }
      }
    ]
  }
}

#------------------------------------------------------------------------------
# Data Sources
#------------------------------------------------------------------------------

data "aws_region" "current" {}

data "aws_caller_identity" "current" {}
