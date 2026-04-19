# GreenLang IAM Module
# Creates IAM roles and policies for EKS workloads with IRSA support

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
# Data Sources
# -----------------------------------------------------------------------------
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# -----------------------------------------------------------------------------
# Application Service Account IAM Role (IRSA)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "app_service_account" {
  name = "${var.name_prefix}-app-service-account"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Condition = {
          StringLike = {
            "${replace(var.oidc_provider_url, "https://", "")}:sub" = "system:serviceaccount:${var.app_namespace}:*"
            "${replace(var.oidc_provider_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = var.tags
}

# S3 Access Policy for Application
resource "aws_iam_role_policy" "app_s3_access" {
  count = var.enable_s3_access ? 1 : 0

  name = "${var.name_prefix}-app-s3-access"
  role = aws_iam_role.app_service_account.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BucketAccess"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Effect   = "Allow"
        Resource = concat(
          [for bucket in var.s3_bucket_arns : bucket],
          [for bucket in var.s3_bucket_arns : "${bucket}/*"]
        )
      }
    ]
  })
}

# Secrets Manager Access Policy
resource "aws_iam_role_policy" "app_secrets_access" {
  count = var.enable_secrets_manager_access ? 1 : 0

  name = "${var.name_prefix}-app-secrets-access"
  role = aws_iam_role.app_service_account.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerAccess"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Effect   = "Allow"
        Resource = var.secrets_arns
      }
    ]
  })
}

# KMS Access Policy
resource "aws_iam_role_policy" "app_kms_access" {
  count = var.enable_kms_access ? 1 : 0

  name = "${var.name_prefix}-app-kms-access"
  role = aws_iam_role.app_service_account.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "KMSAccess"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:GenerateDataKeyWithoutPlaintext",
          "kms:DescribeKey"
        ]
        Effect   = "Allow"
        Resource = var.kms_key_arns
      }
    ]
  })
}

# SQS Access Policy
resource "aws_iam_role_policy" "app_sqs_access" {
  count = var.enable_sqs_access ? 1 : 0

  name = "${var.name_prefix}-app-sqs-access"
  role = aws_iam_role.app_service_account.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SQSAccess"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:GetQueueUrl"
        ]
        Effect   = "Allow"
        Resource = var.sqs_queue_arns
      }
    ]
  })
}

# SNS Access Policy
resource "aws_iam_role_policy" "app_sns_access" {
  count = var.enable_sns_access ? 1 : 0

  name = "${var.name_prefix}-app-sns-access"
  role = aws_iam_role.app_service_account.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SNSAccess"
        Action = [
          "sns:Publish",
          "sns:Subscribe",
          "sns:Unsubscribe"
        ]
        Effect   = "Allow"
        Resource = var.sns_topic_arns
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Agent Runtime Service Account IAM Role (IRSA)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "agent_service_account" {
  count = var.create_agent_role ? 1 : 0

  name = "${var.name_prefix}-agent-service-account"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(var.oidc_provider_url, "https://", "")}:sub" = "system:serviceaccount:${var.agent_namespace}:${var.agent_service_account_name}"
            "${replace(var.oidc_provider_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = var.tags
}

# Agent S3 Access Policy
resource "aws_iam_role_policy" "agent_s3_access" {
  count = var.create_agent_role && var.enable_s3_access ? 1 : 0

  name = "${var.name_prefix}-agent-s3-access"
  role = aws_iam_role.agent_service_account[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ReadAccess"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Effect   = "Allow"
        Resource = concat(
          [for bucket in var.s3_bucket_arns : bucket],
          [for bucket in var.s3_bucket_arns : "${bucket}/*"]
        )
      },
      {
        Sid    = "S3WriteAccess"
        Action = [
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [for bucket in var.agent_s3_write_bucket_arns : "${bucket}/*"]
      }
    ]
  })
}

# Agent Secrets Manager Access
resource "aws_iam_role_policy" "agent_secrets_access" {
  count = var.create_agent_role && var.enable_secrets_manager_access ? 1 : 0

  name = "${var.name_prefix}-agent-secrets-access"
  role = aws_iam_role.agent_service_account[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerAccess"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Effect   = "Allow"
        Resource = var.agent_secrets_arns
      }
    ]
  })
}

# Agent Bedrock Access (for AI/ML agents)
resource "aws_iam_role_policy" "agent_bedrock_access" {
  count = var.create_agent_role && var.enable_bedrock_access ? 1 : 0

  name = "${var.name_prefix}-agent-bedrock-access"
  role = aws_iam_role.agent_service_account[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockModelAccess"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:bedrock:${data.aws_region.current.name}::foundation-model/*"
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# CI/CD Deployment Role
# -----------------------------------------------------------------------------
resource "aws_iam_role" "cicd_deployment" {
  count = var.create_cicd_role ? 1 : 0

  name = "${var.name_prefix}-cicd-deployment"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      # GitHub Actions OIDC Provider
      var.github_oidc_provider_arn != null ? [
        {
          Action = "sts:AssumeRoleWithWebIdentity"
          Effect = "Allow"
          Principal = {
            Federated = var.github_oidc_provider_arn
          }
          Condition = {
            StringEquals = {
              "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
            }
            StringLike = {
              "token.actions.githubusercontent.com:sub" = "repo:${var.github_org}/${var.github_repo}:*"
            }
          }
        }
      ] : [],
      # GitLab OIDC Provider
      var.gitlab_oidc_provider_arn != null ? [
        {
          Action = "sts:AssumeRoleWithWebIdentity"
          Effect = "Allow"
          Principal = {
            Federated = var.gitlab_oidc_provider_arn
          }
          Condition = {
            StringEquals = {
              "${var.gitlab_oidc_url}:aud" = var.gitlab_oidc_audience
            }
            StringLike = {
              "${var.gitlab_oidc_url}:sub" = "project_path:${var.gitlab_project_path}:*"
            }
          }
        }
      ] : [],
      # Cross-account access
      var.cicd_trusted_accounts != null && length(var.cicd_trusted_accounts) > 0 ? [
        {
          Action = "sts:AssumeRole"
          Effect = "Allow"
          Principal = {
            AWS = [for account in var.cicd_trusted_accounts : "arn:aws:iam::${account}:root"]
          }
        }
      ] : []
    )
  })

  tags = var.tags
}

# CI/CD ECR Access Policy
resource "aws_iam_role_policy" "cicd_ecr_access" {
  count = var.create_cicd_role ? 1 : 0

  name = "${var.name_prefix}-cicd-ecr-access"
  role = aws_iam_role.cicd_deployment[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Sid    = "ECRPushPull"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages"
        ]
        Effect   = "Allow"
        Resource = var.ecr_repository_arns
      }
    ]
  })
}

# CI/CD EKS Access Policy
resource "aws_iam_role_policy" "cicd_eks_access" {
  count = var.create_cicd_role ? 1 : 0

  name = "${var.name_prefix}-cicd-eks-access"
  role = aws_iam_role.cicd_deployment[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EKSDescribe"
        Action = [
          "eks:DescribeCluster",
          "eks:ListClusters"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# CI/CD S3 Access Policy (for artifacts)
resource "aws_iam_role_policy" "cicd_s3_access" {
  count = var.create_cicd_role && length(var.cicd_artifact_bucket_arns) > 0 ? 1 : 0

  name = "${var.name_prefix}-cicd-s3-access"
  role = aws_iam_role.cicd_deployment[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ArtifactAccess"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = concat(
          var.cicd_artifact_bucket_arns,
          [for bucket in var.cicd_artifact_bucket_arns : "${bucket}/*"]
        )
      }
    ]
  })
}

# CI/CD Secrets Manager Access
resource "aws_iam_role_policy" "cicd_secrets_access" {
  count = var.create_cicd_role && length(var.cicd_secrets_arns) > 0 ? 1 : 0

  name = "${var.name_prefix}-cicd-secrets-access"
  role = aws_iam_role.cicd_deployment[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsAccess"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Effect   = "Allow"
        Resource = var.cicd_secrets_arns
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# External Secrets Operator Role (IRSA)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "external_secrets" {
  count = var.create_external_secrets_role ? 1 : 0

  name = "${var.name_prefix}-external-secrets"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(var.oidc_provider_url, "https://", "")}:sub" = "system:serviceaccount:${var.external_secrets_namespace}:${var.external_secrets_service_account}"
            "${replace(var.oidc_provider_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "external_secrets" {
  count = var.create_external_secrets_role ? 1 : 0

  name = "${var.name_prefix}-external-secrets"
  role = aws_iam_role.external_secrets[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerReadOnly"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
          "secretsmanager:ListSecrets"
        ]
        Effect   = "Allow"
        Resource = var.external_secrets_allowed_arns
      },
      {
        Sid    = "SSMParameterStoreReadOnly"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter/${var.name_prefix}/*"
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Cross-Account Access Role
# -----------------------------------------------------------------------------
resource "aws_iam_role" "cross_account" {
  count = var.create_cross_account_role && length(var.trusted_account_ids) > 0 ? 1 : 0

  name = "${var.name_prefix}-cross-account-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = [for account_id in var.trusted_account_ids : "arn:aws:iam::${account_id}:root"]
        }
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = var.require_mfa_for_cross_account ? "true" : "false"
          }
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "cross_account" {
  count = var.create_cross_account_role && length(var.trusted_account_ids) > 0 ? 1 : 0

  name = "${var.name_prefix}-cross-account-policy"
  role = aws_iam_role.cross_account[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadOnlyAccess"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Effect   = "Allow"
        Resource = var.cross_account_resource_arns
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Monitoring and Logging Role (for CloudWatch Agent, Fluentd, etc.)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "monitoring" {
  count = var.create_monitoring_role ? 1 : 0

  name = "${var.name_prefix}-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(var.oidc_provider_url, "https://", "")}:sub" = "system:serviceaccount:${var.monitoring_namespace}:${var.monitoring_service_account}"
            "${replace(var.oidc_provider_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "monitoring" {
  count = var.create_monitoring_role ? 1 : 0

  name = "${var.name_prefix}-monitoring"
  role = aws_iam_role.monitoring[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchMetrics"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricData",
          "cloudwatch:ListMetrics"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogs"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/eks/${var.name_prefix}/*"
      },
      {
        Sid    = "XRayAccess"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "xray:GetSamplingRules",
          "xray:GetSamplingTargets",
          "xray:GetSamplingStatisticSummaries"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# GitHub Actions OIDC Provider (if creating from scratch)
# -----------------------------------------------------------------------------
resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1", "1c58a3a8518e8759bf075b76b750d4f2df264fcd"]

  tags = var.tags
}
