# =============================================================================
# GreenLang Grafana Module - IAM
# GreenLang Climate OS | OBS-002
# =============================================================================
# IRSA role for Grafana service account:
#   - CloudWatch read-only access (datasource)
#   - S3 access for image rendering storage
# =============================================================================

# ---------------------------------------------------------------------------
# IAM Role for Grafana (IRSA)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "grafana" {
  name = "${local.name_prefix}-irsa"

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
            "${var.eks_oidc_provider_url}:sub" = "system:serviceaccount:${var.grafana_namespace}:${var.grafana_service_account_name}"
            "${var.eks_oidc_provider_url}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(local.default_tags, {
    Name = "${local.name_prefix}-irsa"
  })
}

# ---------------------------------------------------------------------------
# CloudWatch Read-Only Policy
# ---------------------------------------------------------------------------

resource "aws_iam_policy" "grafana_cloudwatch" {
  name        = "${local.name_prefix}-cloudwatch"
  description = "CloudWatch read-only access for Grafana datasource"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchReadOnly"
        Effect = "Allow"
        Action = [
          "cloudwatch:DescribeAlarmsForMetric",
          "cloudwatch:DescribeAlarmHistory",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:ListMetrics",
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:GetInsightRuleReport"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogsReadOnly"
        Effect = "Allow"
        Action = [
          "logs:DescribeLogGroups",
          "logs:GetLogGroupFields",
          "logs:StartQuery",
          "logs:StopQuery",
          "logs:GetQueryResults",
          "logs:GetLogEvents"
        ]
        Resource = "*"
      },
      {
        Sid    = "EC2DescribeRegions"
        Effect = "Allow"
        Action = [
          "ec2:DescribeRegions"
        ]
        Resource = "*"
      },
      {
        Sid    = "TagsReadOnly"
        Effect = "Allow"
        Action = [
          "tag:GetResources"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.default_tags
}

resource "aws_iam_role_policy_attachment" "grafana_cloudwatch" {
  role       = aws_iam_role.grafana.name
  policy_arn = aws_iam_policy.grafana_cloudwatch.arn
}

# ---------------------------------------------------------------------------
# S3 Image Storage Policy
# ---------------------------------------------------------------------------

resource "aws_iam_policy" "grafana_s3" {
  name        = "${local.name_prefix}-s3"
  description = "S3 access for Grafana image rendering storage"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ImageStorage"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.grafana_images.arn,
          "${aws_s3_bucket.grafana_images.arn}/*"
        ]
      }
    ]
  })

  tags = local.default_tags
}

resource "aws_iam_role_policy_attachment" "grafana_s3" {
  role       = aws_iam_role.grafana.name
  policy_arn = aws_iam_policy.grafana_s3.arn
}
