################################################################################
# Aurora PostgreSQL Secrets Manager Configuration
# Master credentials, application credentials, and rotation configuration
################################################################################

################################################################################
# Random Password Generation
################################################################################

resource "random_password" "master" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
  min_lower        = 1
  min_upper        = 1
  min_numeric      = 1
  min_special      = 1
}

resource "random_password" "application" {
  count = var.create_application_credentials ? 1 : 0

  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
  min_lower        = 1
  min_upper        = 1
  min_numeric      = 1
  min_special      = 1
}

################################################################################
# Master Credentials Secret
################################################################################

resource "aws_secretsmanager_secret" "master_credentials" {
  name        = "${local.name_prefix}-aurora-master-credentials"
  description = "Master credentials for Aurora PostgreSQL cluster ${local.name_prefix}"

  # KMS encryption
  kms_key_id = var.create_kms_key ? aws_kms_key.aurora[0].arn : var.kms_key_arn

  # Recovery window (days before permanent deletion)
  recovery_window_in_days = var.skip_final_snapshot ? 0 : 30

  tags = merge(local.common_tags, {
    Name        = "${local.name_prefix}-aurora-master-credentials"
    SecretType  = "master"
    AutoRotate  = var.secret_rotation_enabled ? "enabled" : "disabled"
  })
}

resource "aws_secretsmanager_secret_version" "master_credentials" {
  secret_id = aws_secretsmanager_secret.master_credentials.id

  secret_string = jsonencode({
    username             = local.master_username
    password             = random_password.master.result
    engine               = "postgres"
    host                 = aws_rds_cluster.aurora.endpoint
    port                 = aws_rds_cluster.aurora.port
    dbname               = var.database_name
    dbClusterIdentifier  = aws_rds_cluster.aurora.cluster_identifier
    # Connection strings for convenience
    connection_string    = "postgresql://${local.master_username}:${urlencode(random_password.master.result)}@${aws_rds_cluster.aurora.endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
    jdbc_connection_string = "jdbc:postgresql://${aws_rds_cluster.aurora.endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }

  depends_on = [aws_rds_cluster.aurora]
}

################################################################################
# Application Credentials Secret
################################################################################

resource "aws_secretsmanager_secret" "application_credentials" {
  count = var.create_application_credentials ? 1 : 0

  name        = "${local.name_prefix}-aurora-app-credentials"
  description = "Application credentials for Aurora PostgreSQL cluster ${local.name_prefix}"

  # KMS encryption
  kms_key_id = var.create_kms_key ? aws_kms_key.aurora[0].arn : var.kms_key_arn

  # Recovery window
  recovery_window_in_days = var.skip_final_snapshot ? 0 : 30

  tags = merge(local.common_tags, {
    Name        = "${local.name_prefix}-aurora-app-credentials"
    SecretType  = "application"
    AutoRotate  = var.secret_rotation_enabled ? "enabled" : "disabled"
  })
}

resource "aws_secretsmanager_secret_version" "application_credentials" {
  count = var.create_application_credentials ? 1 : 0

  secret_id = aws_secretsmanager_secret.application_credentials[0].id

  secret_string = jsonencode({
    username             = var.application_username
    password             = random_password.application[0].result
    engine               = "postgres"
    host                 = aws_rds_cluster.aurora.endpoint
    reader_host          = aws_rds_cluster.aurora.reader_endpoint
    port                 = aws_rds_cluster.aurora.port
    dbname               = var.database_name
    dbClusterIdentifier  = aws_rds_cluster.aurora.cluster_identifier
    # Connection strings for convenience
    writer_connection_string = "postgresql://${var.application_username}:${urlencode(random_password.application[0].result)}@${aws_rds_cluster.aurora.endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
    reader_connection_string = "postgresql://${var.application_username}:${urlencode(random_password.application[0].result)}@${aws_rds_cluster.aurora.reader_endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
    jdbc_writer_connection_string = "jdbc:postgresql://${aws_rds_cluster.aurora.endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
    jdbc_reader_connection_string = "jdbc:postgresql://${aws_rds_cluster.aurora.reader_endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }

  depends_on = [aws_rds_cluster.aurora]
}

################################################################################
# Secret Rotation Configuration (Master)
################################################################################

resource "aws_secretsmanager_secret_rotation" "master_credentials" {
  count = var.secret_rotation_enabled ? 1 : 0

  secret_id           = aws_secretsmanager_secret.master_credentials.id
  rotation_lambda_arn = aws_lambda_function.secret_rotation[0].arn

  rotation_rules {
    automatically_after_days = var.secret_rotation_days
  }

  depends_on = [
    aws_lambda_permission.secret_rotation,
    aws_secretsmanager_secret_version.master_credentials
  ]
}

################################################################################
# Secret Rotation Configuration (Application)
################################################################################

resource "aws_secretsmanager_secret_rotation" "application_credentials" {
  count = var.secret_rotation_enabled && var.create_application_credentials ? 1 : 0

  secret_id           = aws_secretsmanager_secret.application_credentials[0].id
  rotation_lambda_arn = aws_lambda_function.secret_rotation[0].arn

  rotation_rules {
    automatically_after_days = var.secret_rotation_days
  }

  depends_on = [
    aws_lambda_permission.secret_rotation,
    aws_secretsmanager_secret_version.application_credentials
  ]
}

################################################################################
# Lambda Function for Secret Rotation
################################################################################

data "aws_lambda_layer_version" "psycopg2" {
  count = var.secret_rotation_enabled ? 1 : 0

  layer_name = "psycopg2-py38"
  # Note: This is a placeholder. In production, you would create or reference
  # an actual Lambda layer containing the psycopg2 library.
}

resource "aws_lambda_function" "secret_rotation" {
  count = var.secret_rotation_enabled ? 1 : 0

  function_name = "${local.name_prefix}-aurora-secret-rotation"
  description   = "Rotates Aurora PostgreSQL credentials in Secrets Manager"
  role          = aws_iam_role.aurora_secrets_rotation[0].arn
  runtime       = "python3.11"
  handler       = "lambda_function.lambda_handler"
  timeout       = 30
  memory_size   = 128

  # Lambda code (inline placeholder - in production, use S3 or zip file)
  filename         = "${path.module}/lambda/secret_rotation.zip"
  source_code_hash = fileexists("${path.module}/lambda/secret_rotation.zip") ? filebase64sha256("${path.module}/lambda/secret_rotation.zip") : null

  vpc_config {
    subnet_ids         = var.subnet_ids
    security_group_ids = [aws_security_group.aurora.id]
  }

  environment {
    variables = {
      SECRETS_MANAGER_ENDPOINT = "https://secretsmanager.${data.aws_region.current.name}.amazonaws.com"
      EXCLUDE_CHARACTERS       = "/@\"'\\"
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-secret-rotation"
  })

  depends_on = [aws_iam_role_policy_attachment.aurora_secrets_rotation]

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_lambda_permission" "secret_rotation" {
  count = var.secret_rotation_enabled ? 1 : 0

  statement_id  = "AllowSecretsManagerInvocation"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.secret_rotation[0].function_name
  principal     = "secretsmanager.amazonaws.com"
}

################################################################################
# Secret Policy - Restrict Access
################################################################################

resource "aws_secretsmanager_secret_policy" "master_credentials" {
  secret_arn = aws_secretsmanager_secret.master_credentials.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EnableIAMUserPermissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "secretsmanager:*"
        Resource = "*"
      },
      {
        Sid    = "AllowSecretsRotation"
        Effect = "Allow"
        Principal = {
          Service = "secretsmanager.amazonaws.com"
        }
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:PutSecretValue",
          "secretsmanager:UpdateSecretVersionStage",
          "secretsmanager:DescribeSecret"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
}

resource "aws_secretsmanager_secret_policy" "application_credentials" {
  count = var.create_application_credentials ? 1 : 0

  secret_arn = aws_secretsmanager_secret.application_credentials[0].arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EnableIAMUserPermissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "secretsmanager:*"
        Resource = "*"
      },
      {
        Sid    = "AllowSecretsRotation"
        Effect = "Allow"
        Principal = {
          Service = "secretsmanager.amazonaws.com"
        }
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:PutSecretValue",
          "secretsmanager:UpdateSecretVersionStage",
          "secretsmanager:DescribeSecret"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      },
      {
        Sid    = "AllowApplicationAccess"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.aurora_secrets_manager_access.arn
        }
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = "*"
      }
    ]
  })
}

################################################################################
# Output Secret Information
################################################################################

output "master_secret_rotation_enabled" {
  description = "Whether secret rotation is enabled for master credentials"
  value       = var.secret_rotation_enabled
}

output "application_secret_rotation_enabled" {
  description = "Whether secret rotation is enabled for application credentials"
  value       = var.secret_rotation_enabled && var.create_application_credentials
}

output "secret_rotation_lambda_arn" {
  description = "ARN of the secret rotation Lambda function"
  value       = var.secret_rotation_enabled ? aws_lambda_function.secret_rotation[0].arn : null
}

output "secret_rotation_days" {
  description = "Number of days between automatic secret rotations"
  value       = var.secret_rotation_enabled ? var.secret_rotation_days : null
}
