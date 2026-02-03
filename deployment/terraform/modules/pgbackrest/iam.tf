# =============================================================================
# pgBackRest Terraform Module - IAM Configuration
# GreenLang Database Infrastructure
# =============================================================================
# IAM roles and policies for pgBackRest backup and restore operations
# =============================================================================

# -----------------------------------------------------------------------------
# IAM Role for Backup Operations
# -----------------------------------------------------------------------------
resource "aws_iam_role" "pgbackrest_backup" {
  name        = "${local.name_prefix}-pgbackrest-backup-role"
  description = "IAM role for pgBackRest backup operations"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Allow EKS pods to assume this role (IRSA)
      {
        Effect = "Allow"
        Principal = {
          Federated = var.eks_oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(var.eks_oidc_provider_arn, "/^arn:aws:iam::[0-9]+:oidc-provider\\//", "")}:sub" = "system:serviceaccount:${var.kubernetes_namespace}:pgbackrest-backup-sa"
            "${replace(var.eks_oidc_provider_arn, "/^arn:aws:iam::[0-9]+:oidc-provider\\//", "")}:aud" = "sts.amazonaws.com"
          }
        }
      },
      # Allow EC2 instances to assume this role (for non-EKS deployments)
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  max_session_duration = 43200  # 12 hours for long-running backup operations

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-pgbackrest-backup-role"
  })
}

# Backup role policy
resource "aws_iam_role_policy" "pgbackrest_backup" {
  name = "${local.name_prefix}-pgbackrest-backup-policy"
  role = aws_iam_role.pgbackrest_backup.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 permissions for backup operations
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:GetBucketVersioning"
        ]
        Resource = aws_s3_bucket.pgbackrest.arn
      },
      {
        Sid    = "S3ObjectReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging",
          "s3:PutObject",
          "s3:PutObjectTagging",
          "s3:DeleteObject",
          "s3:DeleteObjectVersion",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]
        Resource = "${aws_s3_bucket.pgbackrest.arn}/*"
      },
      # KMS permissions for encryption
      {
        Sid    = "KMSEncryption"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.pgbackrest.arn
      },
      # Secrets Manager permissions (for retrieving credentials)
      {
        Sid    = "SecretsManagerRead"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/pgbackrest/*",
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/postgresql/*"
        ]
      },
      # CloudWatch permissions for logging
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/pgbackrest/*"
      },
      # CloudWatch metrics permissions
      {
        Sid    = "CloudWatchMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "${var.project_name}/pgbackrest"
          }
        }
      },
      # SNS permissions for notifications
      {
        Sid    = "SNSNotifications"
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = var.sns_topic_arn != "" ? var.sns_topic_arn : "arn:aws:sns:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:${var.project_name}-*"
      }
    ]
  })
}

# Instance profile for EC2-based deployments
resource "aws_iam_instance_profile" "pgbackrest_backup" {
  count = var.create_instance_profile ? 1 : 0

  name = "${local.name_prefix}-pgbackrest-backup-profile"
  role = aws_iam_role.pgbackrest_backup.name

  tags = local.tags
}

# -----------------------------------------------------------------------------
# IAM Role for Restore Operations
# -----------------------------------------------------------------------------
resource "aws_iam_role" "pgbackrest_restore" {
  name        = "${local.name_prefix}-pgbackrest-restore-role"
  description = "IAM role for pgBackRest restore operations"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Allow EKS pods to assume this role (IRSA)
      {
        Effect = "Allow"
        Principal = {
          Federated = var.eks_oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(var.eks_oidc_provider_arn, "/^arn:aws:iam::[0-9]+:oidc-provider\\//", "")}:sub" = "system:serviceaccount:${var.kubernetes_namespace}:pgbackrest-restore-sa"
            "${replace(var.eks_oidc_provider_arn, "/^arn:aws:iam::[0-9]+:oidc-provider\\//", "")}:aud" = "sts.amazonaws.com"
          }
        }
      },
      # Allow EC2 instances to assume this role
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  max_session_duration = 43200  # 12 hours for long-running restore operations

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-pgbackrest-restore-role"
  })
}

# Restore role policy
resource "aws_iam_role_policy" "pgbackrest_restore" {
  name = "${local.name_prefix}-pgbackrest-restore-policy"
  role = aws_iam_role.pgbackrest_restore.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 permissions for restore operations (read-only + delete for cleanup)
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = aws_s3_bucket.pgbackrest.arn
      },
      {
        Sid    = "S3ObjectRead"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging"
        ]
        Resource = "${aws_s3_bucket.pgbackrest.arn}/*"
      },
      # KMS permissions for decryption
      {
        Sid    = "KMSDecryption"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.pgbackrest.arn
      },
      # Secrets Manager permissions
      {
        Sid    = "SecretsManagerRead"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/pgbackrest/*",
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/postgresql/*"
        ]
      },
      # CloudWatch permissions
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/pgbackrest/*"
      },
      # SNS notifications
      {
        Sid    = "SNSNotifications"
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = var.sns_topic_arn != "" ? var.sns_topic_arn : "arn:aws:sns:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:${var.project_name}-*"
      }
    ]
  })
}

# Instance profile for EC2-based restore
resource "aws_iam_instance_profile" "pgbackrest_restore" {
  count = var.create_instance_profile ? 1 : 0

  name = "${local.name_prefix}-pgbackrest-restore-profile"
  role = aws_iam_role.pgbackrest_restore.name

  tags = local.tags
}

# -----------------------------------------------------------------------------
# IAM User for pgBackRest (Alternative to IAM Roles)
# -----------------------------------------------------------------------------
resource "aws_iam_user" "pgbackrest" {
  count = var.create_iam_user ? 1 : 0

  name = "${local.name_prefix}-pgbackrest-user"
  path = "/service-accounts/"

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-pgbackrest-user"
  })
}

resource "aws_iam_user_policy" "pgbackrest" {
  count = var.create_iam_user ? 1 : 0

  name = "${local.name_prefix}-pgbackrest-user-policy"
  user = aws_iam_user.pgbackrest[0].name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 permissions
      {
        Sid    = "S3FullAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]
        Resource = [
          aws_s3_bucket.pgbackrest.arn,
          "${aws_s3_bucket.pgbackrest.arn}/*"
        ]
      },
      # KMS permissions
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.pgbackrest.arn
      }
    ]
  })
}

# Access key for IAM user
resource "aws_iam_access_key" "pgbackrest" {
  count = var.create_iam_user ? 1 : 0

  user = aws_iam_user.pgbackrest[0].name
}

# Store access key in Secrets Manager
resource "aws_secretsmanager_secret" "pgbackrest_credentials" {
  count = var.create_iam_user ? 1 : 0

  name        = "${var.project_name}/pgbackrest/s3-credentials"
  description = "pgBackRest S3 access credentials"
  kms_key_id  = var.create_secrets_kms_key ? aws_kms_key.pgbackrest_secrets[0].arn : null

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "pgbackrest_credentials" {
  count = var.create_iam_user ? 1 : 0

  secret_id = aws_secretsmanager_secret.pgbackrest_credentials[0].id
  secret_string = jsonencode({
    access_key_id     = aws_iam_access_key.pgbackrest[0].id
    secret_access_key = aws_iam_access_key.pgbackrest[0].secret
  })
}
