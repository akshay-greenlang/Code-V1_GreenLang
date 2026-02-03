# GitHub OIDC Authentication Module for AWS
# INFRA-001 Component
#
# This module creates the necessary AWS resources for GitHub Actions
# to authenticate using OIDC (OpenID Connect) instead of long-lived credentials.
#
# Author: GreenLang DevOps Team
# Version: 1.0.0

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

# -----------------------------------------------------------------------------
# GitHub OIDC Identity Provider
# -----------------------------------------------------------------------------
resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_oidc_provider ? 1 : 0

  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  # GitHub's OIDC provider thumbprint
  # This is the thumbprint for token.actions.githubusercontent.com
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd"
  ]

  tags = merge(var.tags, {
    Name = "github-actions-oidc"
  })
}

# -----------------------------------------------------------------------------
# IAM Role for GitHub Actions
# -----------------------------------------------------------------------------
resource "aws_iam_role" "github_actions" {
  name        = var.role_name
  description = "IAM role for GitHub Actions OIDC authentication - ${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.create_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : var.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = [
              for repo in var.github_repositories :
              "repo:${var.github_organization}/${repo}:*"
            ]
          }
        }
      }
    ]
  })

  max_session_duration = var.max_session_duration

  tags = merge(var.tags, {
    Name        = var.role_name
    Environment = var.environment
  })
}

# -----------------------------------------------------------------------------
# Terraform State Management Policy
# -----------------------------------------------------------------------------
resource "aws_iam_policy" "terraform_state" {
  name        = "${var.role_name}-terraform-state"
  description = "Policy for Terraform state management"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3StateBucket"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketVersioning"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.terraform_state_bucket}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.terraform_state_bucket}/*"
        ]
      },
      {
        Sid    = "DynamoDBLockTable"
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:DeleteItem",
          "dynamodb:DescribeTable"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:dynamodb:*:${data.aws_caller_identity.current.account_id}:table/${var.terraform_lock_table}"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "terraform_state" {
  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.terraform_state.arn
}

# -----------------------------------------------------------------------------
# EKS Management Policy
# -----------------------------------------------------------------------------
resource "aws_iam_policy" "eks_management" {
  count = var.enable_eks_management ? 1 : 0

  name        = "${var.role_name}-eks-management"
  description = "Policy for EKS cluster management"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EKSFullAccess"
        Effect = "Allow"
        Action = [
          "eks:*"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.allowed_regions
          }
        }
      },
      {
        Sid    = "EKSDescribe"
        Effect = "Allow"
        Action = [
          "eks:DescribeCluster",
          "eks:ListClusters"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "eks_management" {
  count = var.enable_eks_management ? 1 : 0

  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.eks_management[0].arn
}

# -----------------------------------------------------------------------------
# Infrastructure Management Policy (VPC, RDS, ElastiCache, etc.)
# -----------------------------------------------------------------------------
resource "aws_iam_policy" "infrastructure_management" {
  count = var.enable_infrastructure_management ? 1 : 0

  name        = "${var.role_name}-infrastructure"
  description = "Policy for infrastructure management (VPC, RDS, ElastiCache, S3)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "VPCManagement"
        Effect = "Allow"
        Action = [
          "ec2:*Vpc*",
          "ec2:*Subnet*",
          "ec2:*SecurityGroup*",
          "ec2:*RouteTable*",
          "ec2:*InternetGateway*",
          "ec2:*NatGateway*",
          "ec2:*NetworkAcl*",
          "ec2:*FlowLogs*",
          "ec2:*VpcEndpoint*",
          "ec2:*ElasticIp*",
          "ec2:CreateTags",
          "ec2:DeleteTags",
          "ec2:Describe*"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.allowed_regions
          }
        }
      },
      {
        Sid    = "RDSManagement"
        Effect = "Allow"
        Action = [
          "rds:*"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.allowed_regions
          }
        }
      },
      {
        Sid    = "ElastiCacheManagement"
        Effect = "Allow"
        Action = [
          "elasticache:*"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.allowed_regions
          }
        }
      },
      {
        Sid    = "S3BucketManagement"
        Effect = "Allow"
        Action = [
          "s3:CreateBucket",
          "s3:DeleteBucket",
          "s3:Get*",
          "s3:List*",
          "s3:Put*"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.s3_bucket_prefix}-*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.s3_bucket_prefix}-*/*"
        ]
      },
      {
        Sid    = "KMSManagement"
        Effect = "Allow"
        Action = [
          "kms:Create*",
          "kms:Describe*",
          "kms:Enable*",
          "kms:List*",
          "kms:Put*",
          "kms:Update*",
          "kms:Get*",
          "kms:TagResource",
          "kms:UntagResource",
          "kms:ScheduleKeyDeletion",
          "kms:CancelKeyDeletion"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.allowed_regions
          }
        }
      },
      {
        Sid    = "SecretsManagerManagement"
        Effect = "Allow"
        Action = [
          "secretsmanager:*"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:secretsmanager:*:${data.aws_caller_identity.current.account_id}:secret:${var.secrets_prefix}*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:DeleteLogGroup",
          "logs:CreateLogStream",
          "logs:DeleteLogStream",
          "logs:PutLogEvents",
          "logs:PutRetentionPolicy",
          "logs:TagLogGroup",
          "logs:UntagLogGroup",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchAlarms"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DeleteAlarms",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:TagResource",
          "cloudwatch:UntagResource"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "infrastructure_management" {
  count = var.enable_infrastructure_management ? 1 : 0

  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.infrastructure_management[0].arn
}

# -----------------------------------------------------------------------------
# IAM Management Policy (for creating service account roles)
# -----------------------------------------------------------------------------
resource "aws_iam_policy" "iam_management" {
  count = var.enable_iam_management ? 1 : 0

  name        = "${var.role_name}-iam-management"
  description = "Policy for IAM role and policy management"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "IAMRoleManagement"
        Effect = "Allow"
        Action = [
          "iam:CreateRole",
          "iam:DeleteRole",
          "iam:GetRole",
          "iam:UpdateRole",
          "iam:UpdateAssumeRolePolicy",
          "iam:PassRole",
          "iam:TagRole",
          "iam:UntagRole",
          "iam:ListRoleTags",
          "iam:ListAttachedRolePolicies",
          "iam:ListRolePolicies",
          "iam:AttachRolePolicy",
          "iam:DetachRolePolicy",
          "iam:PutRolePolicy",
          "iam:DeleteRolePolicy",
          "iam:GetRolePolicy"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${var.iam_role_prefix}*"
        ]
      },
      {
        Sid    = "IAMPolicyManagement"
        Effect = "Allow"
        Action = [
          "iam:CreatePolicy",
          "iam:DeletePolicy",
          "iam:GetPolicy",
          "iam:GetPolicyVersion",
          "iam:ListPolicyVersions",
          "iam:CreatePolicyVersion",
          "iam:DeletePolicyVersion",
          "iam:TagPolicy",
          "iam:UntagPolicy"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:policy/${var.iam_role_prefix}*"
        ]
      },
      {
        Sid    = "IAMServiceLinkedRoles"
        Effect = "Allow"
        Action = [
          "iam:CreateServiceLinkedRole"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "iam:AWSServiceName" = [
              "eks.amazonaws.com",
              "eks-nodegroup.amazonaws.com",
              "elasticache.amazonaws.com",
              "rds.amazonaws.com"
            ]
          }
        }
      },
      {
        Sid    = "OIDCProviderManagement"
        Effect = "Allow"
        Action = [
          "iam:CreateOpenIDConnectProvider",
          "iam:DeleteOpenIDConnectProvider",
          "iam:GetOpenIDConnectProvider",
          "iam:TagOpenIDConnectProvider",
          "iam:UntagOpenIDConnectProvider",
          "iam:UpdateOpenIDConnectProviderThumbprint"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/*"
      },
      {
        Sid    = "IAMInstanceProfile"
        Effect = "Allow"
        Action = [
          "iam:CreateInstanceProfile",
          "iam:DeleteInstanceProfile",
          "iam:GetInstanceProfile",
          "iam:AddRoleToInstanceProfile",
          "iam:RemoveRoleFromInstanceProfile",
          "iam:TagInstanceProfile",
          "iam:UntagInstanceProfile"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:instance-profile/${var.iam_role_prefix}*"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "iam_management" {
  count = var.enable_iam_management ? 1 : 0

  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.iam_management[0].arn
}

# -----------------------------------------------------------------------------
# ECR Management Policy (for container image publishing)
# -----------------------------------------------------------------------------
resource "aws_iam_policy" "ecr_management" {
  count = var.enable_ecr_management ? 1 : 0

  name        = "${var.role_name}-ecr-management"
  description = "Policy for ECR repository management"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Sid    = "ECRRepositoryManagement"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
          "ecr:CreateRepository",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DeleteRepository",
          "ecr:TagResource",
          "ecr:UntagResource",
          "ecr:SetRepositoryPolicy",
          "ecr:GetRepositoryPolicy",
          "ecr:DeleteRepositoryPolicy",
          "ecr:PutLifecyclePolicy",
          "ecr:GetLifecyclePolicy",
          "ecr:DeleteLifecyclePolicy"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:ecr:*:${data.aws_caller_identity.current.account_id}:repository/${var.ecr_repository_prefix}*"
        ]
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "ecr_management" {
  count = var.enable_ecr_management ? 1 : 0

  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.ecr_management[0].arn
}
