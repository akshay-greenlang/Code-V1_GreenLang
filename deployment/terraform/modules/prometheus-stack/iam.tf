# =============================================================================
# GreenLang Prometheus Stack Module - IAM (IRSA)
# GreenLang Climate OS | OBS-001
# =============================================================================
# Creates IAM role for IRSA (IAM Roles for Service Accounts) to allow:
#   - Thanos Sidecar to upload blocks to S3
#   - Thanos Store Gateway to read from S3
#   - Thanos Compactor to manage blocks in S3
#   - CloudWatch access for metrics export
# =============================================================================

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# -----------------------------------------------------------------------------
# IAM Role for Prometheus/Thanos (IRSA)
# -----------------------------------------------------------------------------
# This role is assumed by Kubernetes ServiceAccounts using OIDC federation.
# Both Prometheus (with Thanos Sidecar) and Thanos components share this role.

resource "aws_iam_role" "prometheus_thanos" {
  count = var.create_iam_role && var.enable_thanos ? 1 : 0

  name = "${var.cluster_name}-prometheus-thanos"
  path = "/greenlang/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringLike = {
            "${var.oidc_provider}:sub" = [
              "system:serviceaccount:${var.namespace}:prometheus-${var.project}-kube-prometheus-prometheus",
              "system:serviceaccount:${var.namespace}:thanos-*"
            ]
            "${var.oidc_provider}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Component   = "prometheus-thanos"
    Environment = var.environment
    IRSA        = "true"
  })
}

# -----------------------------------------------------------------------------
# IAM Policy for S3 Access
# -----------------------------------------------------------------------------
# Grants permissions for Thanos to read/write metrics blocks to S3.

resource "aws_iam_role_policy" "thanos_s3_access" {
  count = var.create_iam_role && var.enable_thanos ? 1 : 0

  name = "thanos-s3-access"
  role = aws_iam_role.prometheus_thanos[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ThanosS3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = aws_s3_bucket.thanos_metrics[0].arn
      },
      {
        Sid    = "ThanosS3ObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.thanos_metrics[0].arn}/*"
      },
      {
        Sid    = "ThanosS3ListAllBuckets"
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets"
        ]
        Resource = "*"
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# IAM Policy for KMS Access (if using KMS encryption)
# -----------------------------------------------------------------------------

resource "aws_iam_role_policy" "thanos_kms_access" {
  count = var.create_iam_role && var.enable_thanos && var.kms_key_arn != null ? 1 : 0

  name = "thanos-kms-access"
  role = aws_iam_role.prometheus_thanos[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ThanosKMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = var.kms_key_arn
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# IAM Policy for CloudWatch (Optional Metrics Export)
# -----------------------------------------------------------------------------

resource "aws_iam_role_policy" "prometheus_cloudwatch" {
  count = var.create_iam_role && var.enable_thanos ? 1 : 0

  name = "prometheus-cloudwatch-access"
  role = aws_iam_role.prometheus_thanos[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchMetricsWrite"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "GreenLang/Prometheus"
          }
        }
      },
      {
        Sid    = "CloudWatchLogsWrite"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/greenlang/prometheus/*"
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# IAM Role for Prometheus (without Thanos - minimal permissions)
# -----------------------------------------------------------------------------
# Used when Thanos is disabled - Prometheus still needs basic IAM for IRSA.

resource "aws_iam_role" "prometheus_only" {
  count = var.create_iam_role && !var.enable_thanos ? 1 : 0

  name = "${var.cluster_name}-prometheus"
  path = "/greenlang/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.oidc_provider}:sub" = "system:serviceaccount:${var.namespace}:prometheus-${var.project}-kube-prometheus-prometheus"
            "${var.oidc_provider}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Component   = "prometheus"
    Environment = var.environment
    IRSA        = "true"
  })
}

# Minimal CloudWatch policy for Prometheus without Thanos
resource "aws_iam_role_policy" "prometheus_only_cloudwatch" {
  count = var.create_iam_role && !var.enable_thanos ? 1 : 0

  name = "prometheus-cloudwatch-access"
  role = aws_iam_role.prometheus_only[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogsWrite"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/greenlang/prometheus/*"
      }
    ]
  })
}
