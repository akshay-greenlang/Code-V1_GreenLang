#######################################
# GreenLang Database Secrets IAM
#######################################

#######################################
# Lambda Execution Role
#######################################

resource "aws_iam_role" "rotation_lambda" {
  name = "${var.name_prefix}-secrets-rotation-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-secrets-rotation-lambda-role"
  })
}

resource "aws_iam_role_policy" "rotation_lambda" {
  name = "${var.name_prefix}-secrets-rotation-lambda-policy"
  role = aws_iam_role.rotation_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerAccess"
        Effect = "Allow"
        Action = [
          "secretsmanager:DescribeSecret",
          "secretsmanager:GetSecretValue",
          "secretsmanager:PutSecretValue",
          "secretsmanager:UpdateSecretVersionStage"
        ]
        Resource = [
          aws_secretsmanager_secret.master_credentials.arn,
          aws_secretsmanager_secret.app_credentials.arn,
          aws_secretsmanager_secret.replication_credentials.arn,
          aws_secretsmanager_secret.pgbouncer_credentials.arn,
        ]
      },
      {
        Sid    = "GenerateRandomPassword"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetRandomPassword"
        ]
        Resource = "*"
      },
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = [local.kms_key_arn]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "${aws_cloudwatch_log_group.rotation_lambda.arn}:*"
        ]
      },
      {
        Sid    = "VPCNetworkInterfaces"
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface",
          "ec2:AssignPrivateIpAddresses",
          "ec2:UnassignPrivateIpAddresses"
        ]
        Resource = "*"
      },
      {
        Sid    = "SNSPublish"
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = var.create_sns_topic ? [aws_sns_topic.rotation_notifications[0].arn] : (var.notification_sns_topic_arn != "" ? [var.notification_sns_topic_arn] : [])
      }
    ]
  })
}

#######################################
# Secrets Access Policy
#######################################

resource "aws_iam_policy" "secrets_access" {
  name        = "${var.name_prefix}-database-secrets-access"
  description = "Policy for accessing GreenLang database secrets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.master_credentials.arn,
          aws_secretsmanager_secret.app_credentials.arn,
          aws_secretsmanager_secret.replication_credentials.arn,
          aws_secretsmanager_secret.pgbouncer_credentials.arn,
          aws_secretsmanager_secret.pgbackrest_encryption.arn,
        ]
      },
      {
        Sid    = "DecryptSecrets"
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = [local.kms_key_arn]
        Condition = {
          StringEquals = {
            "kms:ViaService" = "secretsmanager.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-database-secrets-access-policy"
  })
}

#######################################
# Application Secrets Access Policy (Read-Only)
#######################################

resource "aws_iam_policy" "app_secrets_access" {
  name        = "${var.name_prefix}-app-secrets-access"
  description = "Policy for application to access only its database credentials"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadAppSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.app_credentials.arn,
        ]
      },
      {
        Sid    = "DecryptSecrets"
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = [local.kms_key_arn]
        Condition = {
          StringEquals = {
            "kms:ViaService" = "secretsmanager.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-app-secrets-access-policy"
  })
}

#######################################
# EKS Pod Role for Secrets (IRSA)
#######################################

locals {
  oidc_issuer = var.eks_cluster_oidc_issuer_url != "" ? replace(var.eks_cluster_oidc_issuer_url, "https://", "") : ""
}

data "aws_iam_policy_document" "eks_secrets_assume_role" {
  count = var.eks_cluster_oidc_issuer_url != "" ? 1 : 0

  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_issuer}:sub"
      values   = ["system:serviceaccount:${var.eks_namespace}:${var.eks_service_account_name}"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_issuer}:aud"
      values   = ["sts.amazonaws.com"]
    }

    principals {
      identifiers = ["arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${local.oidc_issuer}"]
      type        = "Federated"
    }
  }
}

resource "aws_iam_role" "eks_secrets" {
  count = var.eks_cluster_oidc_issuer_url != "" ? 1 : 0

  name               = "${var.name_prefix}-eks-secrets-role"
  assume_role_policy = data.aws_iam_policy_document.eks_secrets_assume_role[0].json

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-eks-secrets-role"
  })
}

resource "aws_iam_role_policy_attachment" "eks_secrets" {
  count = var.eks_cluster_oidc_issuer_url != "" ? 1 : 0

  role       = aws_iam_role.eks_secrets[0].name
  policy_arn = aws_iam_policy.secrets_access.arn
}

#######################################
# External Secrets Operator Role (IRSA)
#######################################

resource "aws_iam_role" "external_secrets" {
  count = var.eks_cluster_oidc_issuer_url != "" ? 1 : 0

  name = "${var.name_prefix}-external-secrets-operator"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${local.oidc_issuer}"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${local.oidc_issuer}:sub" = "system:serviceaccount:external-secrets:external-secrets"
            "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-external-secrets-operator-role"
  })
}

resource "aws_iam_role_policy_attachment" "external_secrets" {
  count = var.eks_cluster_oidc_issuer_url != "" ? 1 : 0

  role       = aws_iam_role.external_secrets[0].name
  policy_arn = aws_iam_policy.secrets_access.arn
}

#######################################
# Backup Operator Role (for pgBackRest)
#######################################

resource "aws_iam_role" "backup_operator" {
  count = var.eks_cluster_oidc_issuer_url != "" ? 1 : 0

  name = "${var.name_prefix}-backup-operator"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${local.oidc_issuer}"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${local.oidc_issuer}:sub" = "system:serviceaccount:${var.eks_namespace}:pgbackrest"
            "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-backup-operator-role"
  })
}

resource "aws_iam_role_policy" "backup_operator" {
  count = var.eks_cluster_oidc_issuer_url != "" ? 1 : 0

  name = "${var.name_prefix}-backup-operator-policy"
  role = aws_iam_role.backup_operator[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadBackupSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.pgbackrest_encryption.arn,
        ]
      },
      {
        Sid    = "DecryptSecrets"
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = [local.kms_key_arn]
      },
      {
        Sid    = "S3BackupAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = var.pgbackrest_s3_bucket != "" ? [
          "arn:aws:s3:::${var.pgbackrest_s3_bucket}",
          "arn:aws:s3:::${var.pgbackrest_s3_bucket}/*"
        ] : []
      }
    ]
  })
}

#######################################
# Output IAM Role ARNs
#######################################

output "external_secrets_role_arn" {
  description = "ARN of the External Secrets Operator IAM role"
  value       = var.eks_cluster_oidc_issuer_url != "" ? aws_iam_role.external_secrets[0].arn : ""
}

output "backup_operator_role_arn" {
  description = "ARN of the backup operator IAM role"
  value       = var.eks_cluster_oidc_issuer_url != "" ? aws_iam_role.backup_operator[0].arn : ""
}

output "app_secrets_access_policy_arn" {
  description = "ARN of the application secrets access policy"
  value       = aws_iam_policy.app_secrets_access.arn
}
