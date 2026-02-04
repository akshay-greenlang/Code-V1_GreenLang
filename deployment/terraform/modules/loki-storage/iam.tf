# =============================================================================
# GreenLang Loki Storage Module - IAM for IRSA
# GreenLang Climate OS | INFRA-009
# =============================================================================
# Creates an IAM role with an OIDC trust policy for EKS service accounts.
# Loki pods assume this role via IRSA to access the chunks and ruler S3
# buckets without static credentials.
# =============================================================================

# -----------------------------------------------------------------------------
# IAM Policy - S3 access for Loki
# -----------------------------------------------------------------------------
resource "aws_iam_policy" "loki_s3" {
  name        = "${var.project}-loki-s3-${var.environment}"
  description = "Allows Loki to read/write log chunks and ruler config in S3"
  path        = "/greenlang/"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowS3ObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.chunks.arn}/*",
          "${aws_s3_bucket.ruler.arn}/*"
        ]
      },
      {
        Sid    = "AllowS3BucketListing"
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.chunks.arn,
          aws_s3_bucket.ruler.arn
        ]
      },
      {
        Sid    = "AllowKMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = [
          local.kms_key_arn
        ]
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.project}-loki-s3-${var.environment}"
  })
}

# -----------------------------------------------------------------------------
# IAM Role - OIDC trust for EKS service account
# -----------------------------------------------------------------------------
resource "aws_iam_role" "loki" {
  name = "${var.project}-loki-irsa-${var.environment}"
  path = "/greenlang/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEKSServiceAccountAssume"
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${var.eks_cluster_oidc_issuer}"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.eks_cluster_oidc_issuer}:sub" = "system:serviceaccount:${var.loki_namespace}:${var.loki_service_account_name}"
            "${var.eks_cluster_oidc_issuer}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.project}-loki-irsa-${var.environment}"
  })
}

# -----------------------------------------------------------------------------
# Policy attachment
# -----------------------------------------------------------------------------
resource "aws_iam_role_policy_attachment" "loki_s3" {
  role       = aws_iam_role.loki.name
  policy_arn = aws_iam_policy.loki_s3.arn
}
