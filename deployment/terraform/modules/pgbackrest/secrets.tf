# =============================================================================
# pgBackRest Terraform Module - Secrets Manager Configuration
# GreenLang Database Infrastructure
# =============================================================================
# AWS Secrets Manager configuration for pgBackRest sensitive data
# =============================================================================

# -----------------------------------------------------------------------------
# Random Password for Encryption Passphrase
# -----------------------------------------------------------------------------
resource "random_password" "encryption_passphrase" {
  count = var.create_encryption_passphrase_secret && var.encryption_passphrase == "" ? 1 : 0

  length           = 64
  special          = true
  override_special = "!@#$%^&*()_+-=[]{}|;:,.<>?"
}

# -----------------------------------------------------------------------------
# Encryption Passphrase Secret
# -----------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "encryption_passphrase" {
  count = var.create_encryption_passphrase_secret ? 1 : 0

  name        = "${var.project_name}/pgbackrest/encryption"
  description = "Encryption passphrase for pgBackRest backups (AES-256-CBC)"
  kms_key_id  = var.create_secrets_kms_key ? aws_kms_key.pgbackrest_secrets[0].arn : aws_kms_key.pgbackrest.arn

  recovery_window_in_days = 30

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-pgbackrest-encryption-passphrase"
  })
}

resource "aws_secretsmanager_secret_version" "encryption_passphrase" {
  count = var.create_encryption_passphrase_secret ? 1 : 0

  secret_id = aws_secretsmanager_secret.encryption_passphrase[0].id
  secret_string = jsonencode({
    passphrase = var.encryption_passphrase != "" ? var.encryption_passphrase : random_password.encryption_passphrase[0].result
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# -----------------------------------------------------------------------------
# Secret Policy for Encryption Passphrase
# -----------------------------------------------------------------------------
resource "aws_secretsmanager_secret_policy" "encryption_passphrase" {
  count = var.create_encryption_passphrase_secret ? 1 : 0

  secret_arn = aws_secretsmanager_secret.encryption_passphrase[0].arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowPgBackRestRoles"
        Effect = "Allow"
        Principal = {
          AWS = [
            aws_iam_role.pgbackrest_backup.arn,
            aws_iam_role.pgbackrest_restore.arn
          ]
        }
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = "*"
      },
      {
        Sid    = "DenyDeleteWithoutMFA"
        Effect = "Deny"
        Principal = "*"
        Action = [
          "secretsmanager:DeleteSecret"
        ]
        Resource = "*"
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "false"
          }
        }
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# Rotation Configuration (Optional)
# -----------------------------------------------------------------------------
# Note: Automatic rotation for encryption passphrase is complex as it requires
# re-encrypting all backups. Manual rotation is recommended with proper planning.

# resource "aws_secretsmanager_secret_rotation" "encryption_passphrase" {
#   count = var.create_encryption_passphrase_secret && var.enable_secret_rotation ? 1 : 0
#
#   secret_id           = aws_secretsmanager_secret.encryption_passphrase[0].id
#   rotation_lambda_arn = var.rotation_lambda_arn
#
#   rotation_rules {
#     automatically_after_days = 365
#   }
# }

# -----------------------------------------------------------------------------
# PostgreSQL Credentials Secret (Reference)
# -----------------------------------------------------------------------------
# This creates a placeholder for PostgreSQL credentials that should be
# populated by the PostgreSQL module or manually.
resource "aws_secretsmanager_secret" "postgresql_credentials" {
  count = var.create_encryption_passphrase_secret ? 1 : 0

  name        = "${var.project_name}/postgresql/credentials"
  description = "PostgreSQL credentials for pgBackRest operations"
  kms_key_id  = var.create_secrets_kms_key ? aws_kms_key.pgbackrest_secrets[0].arn : aws_kms_key.pgbackrest.arn

  recovery_window_in_days = 30

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-postgresql-credentials"
  })

  lifecycle {
    # Prevent accidental deletion
    prevent_destroy = false
  }
}

# Note: The actual credentials should be populated separately
# This is just creating the secret container
resource "aws_secretsmanager_secret_version" "postgresql_credentials_placeholder" {
  count = var.create_encryption_passphrase_secret ? 1 : 0

  secret_id = aws_secretsmanager_secret.postgresql_credentials[0].id
  secret_string = jsonencode({
    username = "postgres"
    password = "PLACEHOLDER_UPDATE_MANUALLY"
    host     = "greenlang-postgresql"
    port     = "5432"
    database = "greenlang"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# -----------------------------------------------------------------------------
# Slack Webhook Secret (Optional)
# -----------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "slack_webhook" {
  count = var.create_encryption_passphrase_secret ? 1 : 0

  name        = "${var.project_name}/notifications/slack"
  description = "Slack webhook URL for pgBackRest notifications"
  kms_key_id  = var.create_secrets_kms_key ? aws_kms_key.pgbackrest_secrets[0].arn : aws_kms_key.pgbackrest.arn

  recovery_window_in_days = 7

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-slack-webhook"
  })
}

resource "aws_secretsmanager_secret_version" "slack_webhook_placeholder" {
  count = var.create_encryption_passphrase_secret ? 1 : 0

  secret_id = aws_secretsmanager_secret.slack_webhook[0].id
  secret_string = jsonencode({
    webhook_url = "https://hooks.slack.com/services/PLACEHOLDER"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}
