#------------------------------------------------------------------------------
# IAM Configuration for Redis Secrets Access
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IAM Policy for Secret Access
#------------------------------------------------------------------------------

resource "aws_iam_policy" "redis_secrets_access" {
  name        = "${var.secret_name_prefix}-redis-secrets-access"
  description = "Policy to access Redis secrets in AWS Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "GetRedisSecret"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
          "secretsmanager:GetResourcePolicy"
        ]
        Resource = [
          aws_secretsmanager_secret.redis_auth.arn,
          "${aws_secretsmanager_secret.redis_auth.arn}:*"
        ]
      },
      {
        Sid    = "ListSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:ListSecrets"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "secretsmanager:ResourceTag/Application" = var.secret_name_prefix
          }
        }
      },
      {
        Sid    = "DecryptSecret"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = [
          local.kms_key_arn
        ]
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.secret_name_prefix}-redis-secrets-access"
    Purpose = "Redis secret access"
  })
}

#------------------------------------------------------------------------------
# IAM Policy for ElastiCache AUTH Token
#------------------------------------------------------------------------------

resource "aws_iam_policy" "elasticache_auth" {
  name        = "${var.secret_name_prefix}-elasticache-auth"
  description = "Policy to connect to ElastiCache with IAM authentication"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ElastiCacheConnect"
        Effect = "Allow"
        Action = [
          "elasticache:Connect"
        ]
        Resource = [
          "arn:aws:elasticache:*:${data.aws_caller_identity.current.account_id}:replicationgroup:*",
          "arn:aws:elasticache:*:${data.aws_caller_identity.current.account_id}:serverlesscache:*"
        ]
        Condition = {
          StringEquals = {
            "elasticache:resourceTag/Application" = var.secret_name_prefix
          }
        }
      },
      {
        Sid    = "DescribeElastiCache"
        Effect = "Allow"
        Action = [
          "elasticache:DescribeReplicationGroups",
          "elasticache:DescribeCacheClusters",
          "elasticache:DescribeServerlessCaches"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.secret_name_prefix}-elasticache-auth"
    Purpose = "ElastiCache IAM authentication"
  })
}

#------------------------------------------------------------------------------
# IAM Role for EKS Pods (IRSA - IAM Roles for Service Accounts)
#------------------------------------------------------------------------------

resource "aws_iam_role" "eks_redis_secrets" {
  count = var.eks_oidc_provider_arn != "" ? 1 : 0

  name        = "${var.secret_name_prefix}-eks-redis-secrets"
  description = "IAM role for EKS pods to access Redis secrets via IRSA"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.eks_oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(var.eks_oidc_provider_arn, "/^arn:aws:iam::[0-9]+:oidc-provider\\//", "")}:sub" = "system:serviceaccount:${var.eks_namespace}:${var.eks_service_account_name}"
            "${replace(var.eks_oidc_provider_arn, "/^arn:aws:iam::[0-9]+:oidc-provider\\//", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name        = "${var.secret_name_prefix}-eks-redis-secrets"
    Purpose     = "IRSA for Redis secrets"
    EKSCluster  = var.eks_cluster_name
    Namespace   = var.eks_namespace
    ServiceAccount = var.eks_service_account_name
  })
}

#------------------------------------------------------------------------------
# Attach Policies to EKS Role
#------------------------------------------------------------------------------

resource "aws_iam_role_policy_attachment" "eks_redis_secrets_access" {
  count = var.eks_oidc_provider_arn != "" ? 1 : 0

  role       = aws_iam_role.eks_redis_secrets[0].name
  policy_arn = aws_iam_policy.redis_secrets_access.arn
}

resource "aws_iam_role_policy_attachment" "eks_elasticache_auth" {
  count = var.eks_oidc_provider_arn != "" ? 1 : 0

  role       = aws_iam_role.eks_redis_secrets[0].name
  policy_arn = aws_iam_policy.elasticache_auth.arn
}

#------------------------------------------------------------------------------
# IAM Role for Secret Rotation Lambda
#------------------------------------------------------------------------------

resource "aws_iam_role" "rotation_lambda" {
  count = var.enable_rotation && var.rotation_lambda_arn == null ? 1 : 0

  name        = "${var.secret_name_prefix}-redis-rotation-lambda"
  description = "IAM role for Redis secret rotation Lambda function"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.secret_name_prefix}-redis-rotation-lambda"
    Purpose = "Secret rotation Lambda"
  })
}

resource "aws_iam_role_policy" "rotation_lambda" {
  count = var.enable_rotation && var.rotation_lambda_arn == null ? 1 : 0

  name = "${var.secret_name_prefix}-redis-rotation-policy"
  role = aws_iam_role.rotation_lambda[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerAccess"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:PutSecretValue",
          "secretsmanager:UpdateSecretVersionStage",
          "secretsmanager:DescribeSecret"
        ]
        Resource = aws_secretsmanager_secret.redis_auth.arn
      },
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey"
        ]
        Resource = local.kms_key_arn
      },
      {
        Sid    = "ElastiCacheAccess"
        Effect = "Allow"
        Action = [
          "elasticache:DescribeReplicationGroups",
          "elasticache:ModifyReplicationGroup",
          "elasticache:DescribeUsers",
          "elasticache:ModifyUser"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

#------------------------------------------------------------------------------
# Additional Policy for External Secrets Operator
#------------------------------------------------------------------------------

resource "aws_iam_policy" "external_secrets_operator" {
  name        = "${var.secret_name_prefix}-external-secrets-operator"
  description = "Policy for External Secrets Operator to sync Redis secrets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "GetSecretValue"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.redis_auth.arn
        ]
      },
      {
        Sid    = "BatchGetSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:BatchGetSecretValue"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "secretsmanager:ResourceTag/Application" = var.secret_name_prefix
          }
        }
      },
      {
        Sid    = "ListSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:ListSecrets"
        ]
        Resource = "*"
      },
      {
        Sid    = "DecryptWithKMS"
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = [
          local.kms_key_arn
        ]
        Condition = {
          StringEquals = {
            "kms:ViaService" = "secretsmanager.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name    = "${var.secret_name_prefix}-external-secrets-operator"
    Purpose = "External Secrets Operator access"
  })
}

resource "aws_iam_role_policy_attachment" "eks_external_secrets" {
  count = var.eks_oidc_provider_arn != "" ? 1 : 0

  role       = aws_iam_role.eks_redis_secrets[0].name
  policy_arn = aws_iam_policy.external_secrets_operator.arn
}
