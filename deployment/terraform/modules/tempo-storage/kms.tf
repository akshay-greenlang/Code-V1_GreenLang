# =============================================================================
# GreenLang Tempo Storage Module - KMS Key
# GreenLang Climate OS | OBS-003
# =============================================================================
# Dedicated KMS Customer Managed Key (CMK) for encrypting Tempo trace blocks
# stored in S3. The key policy grants:
#   - Full access to the AWS account root (for administration)
#   - Encrypt/Decrypt/GenerateDataKey to the Tempo IRSA role
#   - S3 service access for bucket-key encryption
#
# Key rotation is enabled (annual automatic rotation by AWS).
# =============================================================================

# -----------------------------------------------------------------------------
# KMS Key
# -----------------------------------------------------------------------------
resource "aws_kms_key" "tempo" {
  description             = "KMS key for Tempo S3 trace block encryption - ${var.project}-${var.environment}"
  deletion_window_in_days = 14
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EnableRootAccountAccess"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "AllowTempoIRSAAccess"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.tempo.arn
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowS3Service"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, {
    Name        = "${var.project}-tempo-kms-${var.environment}"
    Project     = "GreenLang"
    Component   = "Tempo"
    Environment = var.environment
  })
}

# -----------------------------------------------------------------------------
# KMS Alias
# -----------------------------------------------------------------------------
resource "aws_kms_alias" "tempo" {
  name          = "alias/${var.project}-${var.environment}-tempo"
  target_key_id = aws_kms_key.tempo.key_id
}
