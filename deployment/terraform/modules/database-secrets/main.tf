#######################################
# GreenLang Database Secrets Management
# AWS Secrets Manager with Rotation
#######################################

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}

#######################################
# Data Sources
#######################################

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

#######################################
# KMS Key for Secrets Encryption
#######################################

resource "aws_kms_key" "secrets" {
  count = var.create_kms_key ? 1 : 0

  description             = "KMS key for GreenLang database secrets encryption"
  deletion_window_in_days = var.kms_deletion_window_days
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow Secrets Manager"
        Effect = "Allow"
        Principal = {
          Service = "secretsmanager.amazonaws.com"
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
        Sid    = "Allow Lambda Rotation"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.rotation_lambda.arn
        }
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-secrets-kms"
  })
}

resource "aws_kms_alias" "secrets" {
  count = var.create_kms_key ? 1 : 0

  name          = "alias/${var.name_prefix}-database-secrets"
  target_key_id = aws_kms_key.secrets[0].key_id
}

locals {
  kms_key_arn = var.create_kms_key ? aws_kms_key.secrets[0].arn : var.kms_key_arn
}

#######################################
# Random Password Generation
#######################################

resource "random_password" "master_password" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "random_password" "app_password" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "random_password" "replication_password" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "random_password" "pgbouncer_password" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "random_password" "pgbackrest_encryption" {
  length  = 64
  special = false
}

#######################################
# Master Database Credentials Secret
#######################################

resource "aws_secretsmanager_secret" "master_credentials" {
  name                    = "${var.secret_name_prefix}/database/master"
  description             = "Master database credentials for GreenLang PostgreSQL cluster"
  kms_key_id              = local.kms_key_arn
  recovery_window_in_days = var.recovery_window_days

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-master-credentials"
    SecretType  = "database-master"
    Application = "greenlang"
  })
}

resource "aws_secretsmanager_secret_version" "master_credentials" {
  secret_id = aws_secretsmanager_secret.master_credentials.id
  secret_string = jsonencode({
    username             = var.master_username
    password             = random_password.master_password.result
    engine               = "postgres"
    host                 = var.database_host
    port                 = var.database_port
    dbname               = var.database_name
    dbClusterIdentifier  = var.cluster_identifier
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret_rotation" "master_credentials" {
  count = var.enable_rotation ? 1 : 0

  secret_id           = aws_secretsmanager_secret.master_credentials.id
  rotation_lambda_arn = aws_lambda_function.rotation.arn

  rotation_rules {
    automatically_after_days = var.rotation_days
    schedule_expression      = var.rotation_schedule
  }

  depends_on = [aws_lambda_permission.allow_secretsmanager]
}

#######################################
# Application User Credentials Secret
#######################################

resource "aws_secretsmanager_secret" "app_credentials" {
  name                    = "${var.secret_name_prefix}/database/application"
  description             = "Application user credentials for GreenLang PostgreSQL"
  kms_key_id              = local.kms_key_arn
  recovery_window_in_days = var.recovery_window_days

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-app-credentials"
    SecretType  = "database-application"
    Application = "greenlang"
  })
}

resource "aws_secretsmanager_secret_version" "app_credentials" {
  secret_id = aws_secretsmanager_secret.app_credentials.id
  secret_string = jsonencode({
    username            = var.app_username
    password            = random_password.app_password.result
    engine              = "postgres"
    host                = var.database_host
    port                = var.database_port
    dbname              = var.database_name
    dbClusterIdentifier = var.cluster_identifier
    connectionString    = "postgresql://${var.app_username}:${random_password.app_password.result}@${var.database_host}:${var.database_port}/${var.database_name}"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret_rotation" "app_credentials" {
  count = var.enable_rotation ? 1 : 0

  secret_id           = aws_secretsmanager_secret.app_credentials.id
  rotation_lambda_arn = aws_lambda_function.rotation.arn

  rotation_rules {
    automatically_after_days = var.rotation_days
    schedule_expression      = var.rotation_schedule
  }

  depends_on = [aws_lambda_permission.allow_secretsmanager]
}

#######################################
# Replication User Credentials Secret
#######################################

resource "aws_secretsmanager_secret" "replication_credentials" {
  name                    = "${var.secret_name_prefix}/database/replication"
  description             = "Replication user credentials for GreenLang PostgreSQL"
  kms_key_id              = local.kms_key_arn
  recovery_window_in_days = var.recovery_window_days

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-replication-credentials"
    SecretType  = "database-replication"
    Application = "greenlang"
  })
}

resource "aws_secretsmanager_secret_version" "replication_credentials" {
  secret_id = aws_secretsmanager_secret.replication_credentials.id
  secret_string = jsonencode({
    username            = var.replication_username
    password            = random_password.replication_password.result
    engine              = "postgres"
    host                = var.database_host
    port                = var.database_port
    replication_slot    = var.replication_slot_name
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret_rotation" "replication_credentials" {
  count = var.enable_rotation ? 1 : 0

  secret_id           = aws_secretsmanager_secret.replication_credentials.id
  rotation_lambda_arn = aws_lambda_function.rotation.arn

  rotation_rules {
    automatically_after_days = var.rotation_days
    schedule_expression      = var.rotation_schedule
  }

  depends_on = [aws_lambda_permission.allow_secretsmanager]
}

#######################################
# PgBouncer Auth Credentials Secret
#######################################

resource "aws_secretsmanager_secret" "pgbouncer_credentials" {
  name                    = "${var.secret_name_prefix}/database/pgbouncer"
  description             = "PgBouncer authentication credentials for GreenLang"
  kms_key_id              = local.kms_key_arn
  recovery_window_in_days = var.recovery_window_days

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-pgbouncer-credentials"
    SecretType  = "pgbouncer-auth"
    Application = "greenlang"
  })
}

resource "aws_secretsmanager_secret_version" "pgbouncer_credentials" {
  secret_id = aws_secretsmanager_secret.pgbouncer_credentials.id
  secret_string = jsonencode({
    auth_user     = var.pgbouncer_auth_user
    auth_password = random_password.pgbouncer_password.result
    userlist = join("\n", [
      "\"${var.master_username}\" \"${random_password.master_password.result}\"",
      "\"${var.app_username}\" \"${random_password.app_password.result}\"",
      "\"${var.pgbouncer_auth_user}\" \"${random_password.pgbouncer_password.result}\""
    ])
    admin_users  = var.pgbouncer_admin_users
    stats_users  = var.pgbouncer_stats_users
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret_rotation" "pgbouncer_credentials" {
  count = var.enable_rotation ? 1 : 0

  secret_id           = aws_secretsmanager_secret.pgbouncer_credentials.id
  rotation_lambda_arn = aws_lambda_function.rotation.arn

  rotation_rules {
    automatically_after_days = var.rotation_days
    schedule_expression      = var.rotation_schedule
  }

  depends_on = [aws_lambda_permission.allow_secretsmanager]
}

#######################################
# pgBackRest Encryption Key Secret
#######################################

resource "aws_secretsmanager_secret" "pgbackrest_encryption" {
  name                    = "${var.secret_name_prefix}/database/pgbackrest"
  description             = "pgBackRest encryption keys and S3 credentials for GreenLang"
  kms_key_id              = local.kms_key_arn
  recovery_window_in_days = var.recovery_window_days

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-pgbackrest-encryption"
    SecretType  = "backup-encryption"
    Application = "greenlang"
  })
}

resource "aws_secretsmanager_secret_version" "pgbackrest_encryption" {
  secret_id = aws_secretsmanager_secret.pgbackrest_encryption.id
  secret_string = jsonencode({
    repo1_cipher_pass    = random_password.pgbackrest_encryption.result
    repo1_cipher_type    = "aes-256-cbc"
    repo1_s3_key         = var.pgbackrest_s3_access_key
    repo1_s3_key_secret  = var.pgbackrest_s3_secret_key
    repo1_s3_bucket      = var.pgbackrest_s3_bucket
    repo1_s3_region      = var.pgbackrest_s3_region
    repo1_path           = var.pgbackrest_repo_path
    retention_full       = var.pgbackrest_retention_full
    retention_diff       = var.pgbackrest_retention_diff
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

#######################################
# Lambda Rotation Function
#######################################

data "archive_file" "rotation_lambda" {
  type        = "zip"
  source_dir  = "${path.module}/lambda"
  output_path = "${path.module}/lambda_rotation.zip"
}

resource "aws_lambda_function" "rotation" {
  filename         = data.archive_file.rotation_lambda.output_path
  function_name    = "${var.name_prefix}-secrets-rotation"
  role             = aws_iam_role.rotation_lambda.arn
  handler          = "rotate_credentials.lambda_handler"
  source_code_hash = data.archive_file.rotation_lambda.output_base64sha256
  runtime          = "python3.11"
  timeout          = 300
  memory_size      = 256

  environment {
    variables = {
      SECRETS_MANAGER_ENDPOINT = "https://secretsmanager.${data.aws_region.current.name}.amazonaws.com"
      DATABASE_HOST            = var.database_host
      DATABASE_PORT            = tostring(var.database_port)
      DATABASE_NAME            = var.database_name
      SNS_TOPIC_ARN            = var.notification_sns_topic_arn
      EXCLUDED_CHARACTERS      = var.excluded_password_characters
    }
  }

  vpc_config {
    subnet_ids         = var.lambda_subnet_ids
    security_group_ids = var.lambda_security_group_ids
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-secrets-rotation"
  })
}

resource "aws_lambda_permission" "allow_secretsmanager" {
  statement_id  = "AllowSecretsManagerInvocation"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rotation.function_name
  principal     = "secretsmanager.amazonaws.com"
}

#######################################
# CloudWatch Log Group for Lambda
#######################################

resource "aws_cloudwatch_log_group" "rotation_lambda" {
  name              = "/aws/lambda/${aws_lambda_function.rotation.function_name}"
  retention_in_days = var.log_retention_days
  kms_key_id        = local.kms_key_arn

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-rotation-logs"
  })
}

#######################################
# SNS Topic for Rotation Notifications
#######################################

resource "aws_sns_topic" "rotation_notifications" {
  count = var.create_sns_topic ? 1 : 0

  name              = "${var.name_prefix}-secrets-rotation-notifications"
  kms_master_key_id = local.kms_key_arn

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-rotation-notifications"
  })
}

resource "aws_sns_topic_policy" "rotation_notifications" {
  count = var.create_sns_topic ? 1 : 0

  arn = aws_sns_topic.rotation_notifications[0].arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowLambdaPublish"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.rotation_notifications[0].arn
        Condition = {
          ArnEquals = {
            "aws:SourceArn" = aws_lambda_function.rotation.arn
          }
        }
      }
    ]
  })
}

#######################################
# CloudWatch Alarms for Rotation
#######################################

resource "aws_cloudwatch_metric_alarm" "rotation_errors" {
  alarm_name          = "${var.name_prefix}-secrets-rotation-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Secrets rotation Lambda function errors"

  dimensions = {
    FunctionName = aws_lambda_function.rotation.function_name
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-rotation-errors-alarm"
  })
}
