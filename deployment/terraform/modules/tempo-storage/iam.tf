# =============================================================================
# GreenLang Tempo Storage Module - IAM for IRSA
# GreenLang Climate OS | OBS-003
# =============================================================================
# Creates an IAM role with an OIDC trust policy for EKS service accounts.
# Tempo pods assume this role via IRSA to access the traces S3 bucket and
# the dedicated KMS key without static credentials.
#
# Permissions granted:
#   - s3:PutObject, s3:GetObject, s3:DeleteObject on bucket objects
#   - s3:ListBucket on the bucket itself
#   - kms:Encrypt, kms:Decrypt, kms:GenerateDataKey, kms:DescribeKey on the
#     Tempo KMS key
# =============================================================================

# -----------------------------------------------------------------------------
# IAM Policy - S3 + KMS access for Tempo
# -----------------------------------------------------------------------------
resource "aws_iam_policy" "tempo_s3" {
  name        = "${var.project}-tempo-s3-${var.environment}"
  description = "Allows Tempo to read/write trace blocks in S3 and use KMS encryption"
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
          "${aws_s3_bucket.traces.arn}/*"
        ]
      },
      {
        Sid    = "AllowS3BucketListing"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.traces.arn
        ]
      },
      {
        Sid    = "AllowKMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = [
          aws_kms_key.tempo.arn
        ]
      }
    ]
  })

  tags = merge(var.tags, {
    Name        = "${var.project}-tempo-s3-${var.environment}"
    Environment = var.environment
  })
}

# -----------------------------------------------------------------------------
# IAM Role - OIDC trust for EKS service account
# -----------------------------------------------------------------------------
resource "aws_iam_role" "tempo" {
  name = "${var.project}-tempo-irsa-${var.environment}"
  path = "/greenlang/"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEKSServiceAccountAssume"
        Effect = "Allow"
        Principal = {
          Federated = var.eks_oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.eks_oidc_provider_url}:sub" = "system:serviceaccount:${var.tempo_namespace}:${var.tempo_service_account}"
            "${var.eks_oidc_provider_url}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name        = "${var.project}-tempo-irsa-${var.environment}"
    Environment = var.environment
  })
}

# -----------------------------------------------------------------------------
# Policy attachment
# -----------------------------------------------------------------------------
resource "aws_iam_role_policy_attachment" "tempo_s3" {
  role       = aws_iam_role.tempo.name
  policy_arn = aws_iam_policy.tempo_s3.arn
}
