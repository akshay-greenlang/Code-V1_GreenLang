#######################################
# GreenLang Database Secrets Outputs
#######################################

#######################################
# Secret ARNs
#######################################

output "master_credentials_secret_arn" {
  description = "ARN of the master database credentials secret"
  value       = aws_secretsmanager_secret.master_credentials.arn
}

output "app_credentials_secret_arn" {
  description = "ARN of the application credentials secret"
  value       = aws_secretsmanager_secret.app_credentials.arn
}

output "replication_credentials_secret_arn" {
  description = "ARN of the replication credentials secret"
  value       = aws_secretsmanager_secret.replication_credentials.arn
}

output "pgbouncer_credentials_secret_arn" {
  description = "ARN of the PgBouncer credentials secret"
  value       = aws_secretsmanager_secret.pgbouncer_credentials.arn
}

output "pgbackrest_encryption_secret_arn" {
  description = "ARN of the pgBackRest encryption secret"
  value       = aws_secretsmanager_secret.pgbackrest_encryption.arn
}

output "all_secret_arns" {
  description = "List of all database secret ARNs"
  value = [
    aws_secretsmanager_secret.master_credentials.arn,
    aws_secretsmanager_secret.app_credentials.arn,
    aws_secretsmanager_secret.replication_credentials.arn,
    aws_secretsmanager_secret.pgbouncer_credentials.arn,
    aws_secretsmanager_secret.pgbackrest_encryption.arn,
  ]
}

#######################################
# Secret Names
#######################################

output "master_credentials_secret_name" {
  description = "Name of the master database credentials secret"
  value       = aws_secretsmanager_secret.master_credentials.name
}

output "app_credentials_secret_name" {
  description = "Name of the application credentials secret"
  value       = aws_secretsmanager_secret.app_credentials.name
}

output "replication_credentials_secret_name" {
  description = "Name of the replication credentials secret"
  value       = aws_secretsmanager_secret.replication_credentials.name
}

output "pgbouncer_credentials_secret_name" {
  description = "Name of the PgBouncer credentials secret"
  value       = aws_secretsmanager_secret.pgbouncer_credentials.name
}

output "pgbackrest_encryption_secret_name" {
  description = "Name of the pgBackRest encryption secret"
  value       = aws_secretsmanager_secret.pgbackrest_encryption.name
}

output "all_secret_names" {
  description = "Map of all database secret names"
  value = {
    master      = aws_secretsmanager_secret.master_credentials.name
    application = aws_secretsmanager_secret.app_credentials.name
    replication = aws_secretsmanager_secret.replication_credentials.name
    pgbouncer   = aws_secretsmanager_secret.pgbouncer_credentials.name
    pgbackrest  = aws_secretsmanager_secret.pgbackrest_encryption.name
  }
}

#######################################
# Secret IDs
#######################################

output "master_credentials_secret_id" {
  description = "ID of the master database credentials secret"
  value       = aws_secretsmanager_secret.master_credentials.id
}

output "app_credentials_secret_id" {
  description = "ID of the application credentials secret"
  value       = aws_secretsmanager_secret.app_credentials.id
}

output "replication_credentials_secret_id" {
  description = "ID of the replication credentials secret"
  value       = aws_secretsmanager_secret.replication_credentials.id
}

output "pgbouncer_credentials_secret_id" {
  description = "ID of the PgBouncer credentials secret"
  value       = aws_secretsmanager_secret.pgbouncer_credentials.id
}

output "pgbackrest_encryption_secret_id" {
  description = "ID of the pgBackRest encryption secret"
  value       = aws_secretsmanager_secret.pgbackrest_encryption.id
}

#######################################
# KMS Key
#######################################

output "kms_key_arn" {
  description = "ARN of the KMS key used for secrets encryption"
  value       = local.kms_key_arn
}

output "kms_key_id" {
  description = "ID of the KMS key used for secrets encryption"
  value       = var.create_kms_key ? aws_kms_key.secrets[0].key_id : ""
}

output "kms_alias_arn" {
  description = "ARN of the KMS alias"
  value       = var.create_kms_key ? aws_kms_alias.secrets[0].arn : ""
}

#######################################
# Lambda Function
#######################################

output "rotation_lambda_arn" {
  description = "ARN of the rotation Lambda function"
  value       = aws_lambda_function.rotation.arn
}

output "rotation_lambda_name" {
  description = "Name of the rotation Lambda function"
  value       = aws_lambda_function.rotation.function_name
}

output "rotation_lambda_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = aws_iam_role.rotation_lambda.arn
}

#######################################
# IAM Resources
#######################################

output "secrets_access_policy_arn" {
  description = "ARN of the IAM policy for secret access"
  value       = aws_iam_policy.secrets_access.arn
}

output "eks_secrets_role_arn" {
  description = "ARN of the IAM role for EKS pods to access secrets"
  value       = var.eks_cluster_oidc_issuer_url != "" ? aws_iam_role.eks_secrets[0].arn : ""
}

output "eks_secrets_role_name" {
  description = "Name of the IAM role for EKS pods"
  value       = var.eks_cluster_oidc_issuer_url != "" ? aws_iam_role.eks_secrets[0].name : ""
}

#######################################
# SNS Topic
#######################################

output "rotation_notifications_topic_arn" {
  description = "ARN of the SNS topic for rotation notifications"
  value       = var.create_sns_topic ? aws_sns_topic.rotation_notifications[0].arn : var.notification_sns_topic_arn
}

#######################################
# CloudWatch
#######################################

output "rotation_lambda_log_group" {
  description = "CloudWatch log group for rotation Lambda"
  value       = aws_cloudwatch_log_group.rotation_lambda.name
}

output "rotation_errors_alarm_arn" {
  description = "ARN of the CloudWatch alarm for rotation errors"
  value       = aws_cloudwatch_metric_alarm.rotation_errors.arn
}

#######################################
# External Secrets Configuration
#######################################

output "external_secrets_config" {
  description = "Configuration for External Secrets Operator"
  value = {
    aws_region = data.aws_region.current.name
    secrets = {
      master = {
        name = aws_secretsmanager_secret.master_credentials.name
        arn  = aws_secretsmanager_secret.master_credentials.arn
      }
      application = {
        name = aws_secretsmanager_secret.app_credentials.name
        arn  = aws_secretsmanager_secret.app_credentials.arn
      }
      replication = {
        name = aws_secretsmanager_secret.replication_credentials.name
        arn  = aws_secretsmanager_secret.replication_credentials.arn
      }
      pgbouncer = {
        name = aws_secretsmanager_secret.pgbouncer_credentials.name
        arn  = aws_secretsmanager_secret.pgbouncer_credentials.arn
      }
      pgbackrest = {
        name = aws_secretsmanager_secret.pgbackrest_encryption.name
        arn  = aws_secretsmanager_secret.pgbackrest_encryption.arn
      }
    }
    service_account = {
      name      = var.eks_service_account_name
      namespace = var.eks_namespace
      role_arn  = var.eks_cluster_oidc_issuer_url != "" ? aws_iam_role.eks_secrets[0].arn : ""
    }
  }
}
