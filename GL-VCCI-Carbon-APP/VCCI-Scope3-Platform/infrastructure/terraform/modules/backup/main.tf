# Backup Module - AWS Backup

resource "aws_backup_vault" "main" {
  count       = var.enable_aws_backup ? 1 : 0
  name        = "${var.project_name}-${var.environment}-vault"
  kms_key_arn = var.kms_key_arn
  tags        = var.tags
}

resource "aws_backup_plan" "main" {
  count = var.enable_aws_backup ? 1 : 0
  name  = "${var.project_name}-${var.environment}-plan"

  rule {
    rule_name         = "daily_backup"
    target_vault_name = aws_backup_vault.main[0].name
    schedule          = var.backup_schedule

    lifecycle {
      delete_after = var.backup_retention_days
    }
  }

  tags = var.tags
}

resource "aws_backup_selection" "rds" {
  count        = var.enable_aws_backup ? 1 : 0
  name         = "${var.project_name}-${var.environment}-rds"
  iam_role_arn = aws_iam_role.backup[0].arn
  plan_id      = aws_backup_plan.main[0].id

  resources = [var.rds_cluster_arn]
}

resource "aws_iam_role" "backup" {
  count = var.enable_aws_backup ? 1 : 0
  name  = "${var.project_name}-${var.environment}-backup"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "backup.amazonaws.com" }
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "backup" {
  count      = var.enable_aws_backup ? 1 : 0
  role       = aws_iam_role.backup[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}
