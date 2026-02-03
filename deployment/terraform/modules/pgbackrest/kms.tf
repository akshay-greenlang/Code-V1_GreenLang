# =============================================================================
# pgBackRest Terraform Module - KMS Configuration
# GreenLang Database Infrastructure
# =============================================================================
# KMS key for encrypting pgBackRest backups in S3
# =============================================================================

# -----------------------------------------------------------------------------
# KMS Key for pgBackRest Encryption
# -----------------------------------------------------------------------------
resource "aws_kms_key" "pgbackrest" {
  description             = "KMS key for pgBackRest backup encryption - ${local.name_prefix}"
  deletion_window_in_days = var.kms_deletion_window_days
  enable_key_rotation     = var.enable_kms_key_rotation
  multi_region            = var.enable_multi_region_kms

  policy = data.aws_iam_policy_document.kms_policy.json

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-pgbackrest-kms"
  })
}

# KMS key alias
resource "aws_kms_alias" "pgbackrest" {
  name          = "alias/${local.name_prefix}-pgbackrest"
  target_key_id = aws_kms_key.pgbackrest.key_id
}

# KMS key policy
data "aws_iam_policy_document" "kms_policy" {
  # Allow root account full access
  statement {
    sid    = "AllowRootAccess"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }

    actions   = ["kms:*"]
    resources = ["*"]
  }

  # Allow pgBackRest backup role to use the key
  statement {
    sid    = "AllowBackupRoleUse"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.pgbackrest_backup.arn]
    }

    actions = [
      "kms:Encrypt",
      "kms:Decrypt",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:DescribeKey"
    ]

    resources = ["*"]
  }

  # Allow pgBackRest restore role to use the key
  statement {
    sid    = "AllowRestoreRoleUse"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.pgbackrest_restore.arn]
    }

    actions = [
      "kms:Decrypt",
      "kms:DescribeKey"
    ]

    resources = ["*"]
  }

  # Allow S3 to use the key
  statement {
    sid    = "AllowS3Use"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["s3.amazonaws.com"]
    }

    actions = [
      "kms:Encrypt",
      "kms:Decrypt",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:DescribeKey"
    ]

    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }

  # Allow CloudWatch Logs to use the key (for log encryption)
  statement {
    sid    = "AllowCloudWatchLogs"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["logs.${data.aws_region.current.name}.amazonaws.com"]
    }

    actions = [
      "kms:Encrypt",
      "kms:Decrypt",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:DescribeKey"
    ]

    resources = ["*"]

    condition {
      test     = "ArnLike"
      variable = "kms:EncryptionContext:aws:logs:arn"
      values   = ["arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:*"]
    }
  }

  # Allow key administration by specific IAM users/roles
  dynamic "statement" {
    for_each = length(var.kms_key_administrators) > 0 ? [1] : []
    content {
      sid    = "AllowKeyAdministration"
      effect = "Allow"

      principals {
        type        = "AWS"
        identifiers = var.kms_key_administrators
      }

      actions = [
        "kms:Create*",
        "kms:Describe*",
        "kms:Enable*",
        "kms:List*",
        "kms:Put*",
        "kms:Update*",
        "kms:Revoke*",
        "kms:Disable*",
        "kms:Get*",
        "kms:Delete*",
        "kms:TagResource",
        "kms:UntagResource",
        "kms:ScheduleKeyDeletion",
        "kms:CancelKeyDeletion"
      ]

      resources = ["*"]
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# -----------------------------------------------------------------------------
# KMS Key for pgBackRest Configuration Encryption (Secrets)
# -----------------------------------------------------------------------------
resource "aws_kms_key" "pgbackrest_secrets" {
  count = var.create_secrets_kms_key ? 1 : 0

  description             = "KMS key for pgBackRest secrets encryption - ${local.name_prefix}"
  deletion_window_in_days = var.kms_deletion_window_days
  enable_key_rotation     = var.enable_kms_key_rotation

  policy = data.aws_iam_policy_document.secrets_kms_policy[0].json

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-pgbackrest-secrets-kms"
  })
}

resource "aws_kms_alias" "pgbackrest_secrets" {
  count = var.create_secrets_kms_key ? 1 : 0

  name          = "alias/${local.name_prefix}-pgbackrest-secrets"
  target_key_id = aws_kms_key.pgbackrest_secrets[0].key_id
}

data "aws_iam_policy_document" "secrets_kms_policy" {
  count = var.create_secrets_kms_key ? 1 : 0

  # Allow root account full access
  statement {
    sid    = "AllowRootAccess"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }

    actions   = ["kms:*"]
    resources = ["*"]
  }

  # Allow Secrets Manager to use the key
  statement {
    sid    = "AllowSecretsManagerUse"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["secretsmanager.amazonaws.com"]
    }

    actions = [
      "kms:Encrypt",
      "kms:Decrypt",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:DescribeKey"
    ]

    resources = ["*"]
  }

  # Allow pgBackRest roles to decrypt secrets
  statement {
    sid    = "AllowPgBackRestRolesDecrypt"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.pgbackrest_backup.arn, aws_iam_role.pgbackrest_restore.arn]
    }

    actions = [
      "kms:Decrypt",
      "kms:DescribeKey"
    ]

    resources = ["*"]
  }
}
