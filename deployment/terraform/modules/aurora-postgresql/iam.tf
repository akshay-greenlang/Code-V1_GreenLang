################################################################################
# Aurora PostgreSQL IAM Roles
# IAM roles for enhanced monitoring, S3 export, Lambda integration, and Secrets Manager
################################################################################

################################################################################
# Enhanced Monitoring Role
################################################################################

resource "aws_iam_role" "aurora_enhanced_monitoring" {
  name = "${local.name_prefix}-aurora-enhanced-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowRDSMonitoringService"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-enhanced-monitoring"
  })
}

resource "aws_iam_role_policy_attachment" "aurora_enhanced_monitoring" {
  role       = aws_iam_role.aurora_enhanced_monitoring.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

################################################################################
# S3 Export Role
################################################################################

resource "aws_iam_role" "aurora_s3_export" {
  count = var.enable_s3_export ? 1 : 0

  name = "${local.name_prefix}-aurora-s3-export"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowRDSService"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-s3-export"
  })
}

resource "aws_iam_policy" "aurora_s3_export" {
  count = var.enable_s3_export ? 1 : 0

  name        = "${local.name_prefix}-aurora-s3-export-policy"
  description = "Policy for Aurora PostgreSQL S3 export functionality"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowS3ExportWrite"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]
        Resource = concat(
          var.s3_export_bucket_arns,
          [for bucket in var.s3_export_bucket_arns : "${bucket}/*"]
        )
      },
      {
        Sid    = "AllowKMSEncryption"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = var.create_kms_key ? aws_kms_key.aurora[0].arn : var.kms_key_arn
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-s3-export-policy"
  })
}

resource "aws_iam_role_policy_attachment" "aurora_s3_export" {
  count = var.enable_s3_export ? 1 : 0

  role       = aws_iam_role.aurora_s3_export[0].name
  policy_arn = aws_iam_policy.aurora_s3_export[0].arn
}

################################################################################
# Lambda Integration Role
################################################################################

resource "aws_iam_role" "aurora_lambda_integration" {
  name = "${local.name_prefix}-aurora-lambda-integration"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowRDSService"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-lambda-integration"
  })
}

resource "aws_iam_policy" "aurora_lambda_integration" {
  name        = "${local.name_prefix}-aurora-lambda-integration-policy"
  description = "Policy for Aurora PostgreSQL Lambda integration"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowLambdaInvoke"
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction",
          "lambda:GetFunction"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-lambda-integration-policy"
  })
}

resource "aws_iam_role_policy_attachment" "aurora_lambda_integration" {
  role       = aws_iam_role.aurora_lambda_integration.name
  policy_arn = aws_iam_policy.aurora_lambda_integration.arn
}

################################################################################
# Secrets Manager Access Role (for applications)
################################################################################

resource "aws_iam_role" "aurora_secrets_manager_access" {
  name = "${local.name_prefix}-aurora-secrets-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEC2AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
      {
        Sid    = "AllowECSAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
      {
        Sid    = "AllowLambdaAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-secrets-access"
  })
}

resource "aws_iam_policy" "aurora_secrets_manager_access" {
  name        = "${local.name_prefix}-aurora-secrets-access-policy"
  description = "Policy for accessing Aurora PostgreSQL credentials in Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowSecretsManagerRead"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = compact([
          aws_secretsmanager_secret.master_credentials.arn,
          var.create_application_credentials ? aws_secretsmanager_secret.application_credentials[0].arn : ""
        ])
      },
      {
        Sid    = "AllowKMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = var.create_kms_key ? aws_kms_key.aurora[0].arn : var.kms_key_arn
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-secrets-access-policy"
  })
}

resource "aws_iam_role_policy_attachment" "aurora_secrets_manager_access" {
  role       = aws_iam_role.aurora_secrets_manager_access.name
  policy_arn = aws_iam_policy.aurora_secrets_manager_access.arn
}

################################################################################
# IAM Database Authentication Role
################################################################################

resource "aws_iam_role" "aurora_iam_auth" {
  count = var.iam_database_authentication_enabled ? 1 : 0

  name = "${local.name_prefix}-aurora-iam-auth"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEC2AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
      {
        Sid    = "AllowECSAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
      {
        Sid    = "AllowLambdaAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-iam-auth"
  })
}

resource "aws_iam_policy" "aurora_iam_auth" {
  count = var.iam_database_authentication_enabled ? 1 : 0

  name        = "${local.name_prefix}-aurora-iam-auth-policy"
  description = "Policy for IAM database authentication to Aurora PostgreSQL"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowIAMDatabaseAuthentication"
        Effect = "Allow"
        Action = [
          "rds-db:connect"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:rds-db:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:dbuser:${aws_rds_cluster.aurora.cluster_resource_id}/*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-iam-auth-policy"
  })
}

resource "aws_iam_role_policy_attachment" "aurora_iam_auth" {
  count = var.iam_database_authentication_enabled ? 1 : 0

  role       = aws_iam_role.aurora_iam_auth[0].name
  policy_arn = aws_iam_policy.aurora_iam_auth[0].arn
}

################################################################################
# Secrets Manager Rotation Role (for automatic secret rotation)
################################################################################

resource "aws_iam_role" "aurora_secrets_rotation" {
  count = var.secret_rotation_enabled ? 1 : 0

  name = "${local.name_prefix}-aurora-secrets-rotation"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowLambdaAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-secrets-rotation"
  })
}

resource "aws_iam_policy" "aurora_secrets_rotation" {
  count = var.secret_rotation_enabled ? 1 : 0

  name        = "${local.name_prefix}-aurora-secrets-rotation-policy"
  description = "Policy for Secrets Manager rotation Lambda function"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowSecretsManagerOperations"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
          "secretsmanager:PutSecretValue",
          "secretsmanager:UpdateSecretVersionStage",
          "secretsmanager:GetRandomPassword"
        ]
        Resource = compact([
          aws_secretsmanager_secret.master_credentials.arn,
          var.create_application_credentials ? aws_secretsmanager_secret.application_credentials[0].arn : ""
        ])
      },
      {
        Sid    = "AllowKMSOperations"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = var.create_kms_key ? aws_kms_key.aurora[0].arn : var.kms_key_arn
      },
      {
        Sid    = "AllowVPCAccess"
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowCloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-secrets-rotation-policy"
  })
}

resource "aws_iam_role_policy_attachment" "aurora_secrets_rotation" {
  count = var.secret_rotation_enabled ? 1 : 0

  role       = aws_iam_role.aurora_secrets_rotation[0].name
  policy_arn = aws_iam_policy.aurora_secrets_rotation[0].arn
}

################################################################################
# Output Roles for External Use
################################################################################

output "iam_auth_role_arn" {
  description = "The ARN of the IAM authentication role (if created)"
  value       = var.iam_database_authentication_enabled ? aws_iam_role.aurora_iam_auth[0].arn : null
}

output "iam_auth_role_name" {
  description = "The name of the IAM authentication role (if created)"
  value       = var.iam_database_authentication_enabled ? aws_iam_role.aurora_iam_auth[0].name : null
}

output "secrets_access_role_arn" {
  description = "The ARN of the Secrets Manager access role"
  value       = aws_iam_role.aurora_secrets_manager_access.arn
}

output "secrets_access_role_name" {
  description = "The name of the Secrets Manager access role"
  value       = aws_iam_role.aurora_secrets_manager_access.name
}

output "secrets_rotation_role_arn" {
  description = "The ARN of the Secrets Manager rotation role (if created)"
  value       = var.secret_rotation_enabled ? aws_iam_role.aurora_secrets_rotation[0].arn : null
}

output "secrets_rotation_role_name" {
  description = "The name of the Secrets Manager rotation role (if created)"
  value       = var.secret_rotation_enabled ? aws_iam_role.aurora_secrets_rotation[0].name : null
}
