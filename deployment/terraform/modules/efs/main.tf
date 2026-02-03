#------------------------------------------------------------------------------
# AWS EFS Module - Main Configuration
# GreenLang Infrastructure
#
# This module creates a production-ready EFS file system with:
# - Multi-AZ mount targets for high availability
# - Access points for application isolation
# - Encryption at rest and in transit
# - Lifecycle policies for cost optimization
# - Backup integration with AWS Backup
# - Replication for disaster recovery
#------------------------------------------------------------------------------

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

#------------------------------------------------------------------------------
# Data Sources
#------------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

#------------------------------------------------------------------------------
# KMS Key for EFS Encryption
#------------------------------------------------------------------------------

resource "aws_kms_key" "efs" {
  count = var.create_kms_key ? 1 : 0

  description             = "KMS key for EFS encryption - ${var.name}"
  deletion_window_in_days = var.kms_key_deletion_window
  enable_key_rotation     = var.kms_key_enable_rotation
  policy                  = data.aws_iam_policy_document.kms_policy[0].json

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-efs-kms"
    }
  )
}

resource "aws_kms_alias" "efs" {
  count = var.create_kms_key ? 1 : 0

  name          = "alias/${var.name}-efs"
  target_key_id = aws_kms_key.efs[0].key_id
}

data "aws_iam_policy_document" "kms_policy" {
  count = var.create_kms_key ? 1 : 0

  # Allow root account full access
  statement {
    sid    = "AllowRootAccount"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }

    actions   = ["kms:*"]
    resources = ["*"]
  }

  # Allow EFS service to use the key
  statement {
    sid    = "AllowEFSService"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["elasticfilesystem.amazonaws.com"]
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
      variable = "kms:CallerAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }

  # Allow AWS Backup service
  statement {
    sid    = "AllowAWSBackup"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["backup.amazonaws.com"]
    }

    actions = [
      "kms:Encrypt",
      "kms:Decrypt",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:DescribeKey",
      "kms:CreateGrant"
    ]

    resources = ["*"]
  }
}

#------------------------------------------------------------------------------
# EFS File System
#------------------------------------------------------------------------------

resource "aws_efs_file_system" "main" {
  creation_token = var.name

  # Performance configuration
  performance_mode                = var.performance_mode
  throughput_mode                 = var.throughput_mode
  provisioned_throughput_in_mibps = var.throughput_mode == "provisioned" ? var.provisioned_throughput_in_mibps : null

  # Encryption configuration
  encrypted  = var.encrypted
  kms_key_id = var.encrypted ? (var.create_kms_key ? aws_kms_key.efs[0].arn : var.kms_key_id) : null

  # Lifecycle policies for cost optimization
  dynamic "lifecycle_policy" {
    for_each = var.transition_to_ia != null ? [1] : []
    content {
      transition_to_ia = var.transition_to_ia
    }
  }

  dynamic "lifecycle_policy" {
    for_each = var.transition_to_primary_storage_class != null ? [1] : []
    content {
      transition_to_primary_storage_class = var.transition_to_primary_storage_class
    }
  }

  dynamic "lifecycle_policy" {
    for_each = var.transition_to_archive != null ? [1] : []
    content {
      transition_to_archive = var.transition_to_archive
    }
  }

  tags = merge(
    var.tags,
    {
      Name = var.name
    }
  )
}

#------------------------------------------------------------------------------
# EFS Mount Targets
#------------------------------------------------------------------------------

resource "aws_efs_mount_target" "main" {
  for_each = toset(var.subnet_ids)

  file_system_id  = aws_efs_file_system.main.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

#------------------------------------------------------------------------------
# Security Group for EFS
#------------------------------------------------------------------------------

resource "aws_security_group" "efs" {
  name        = "${var.name}-efs-sg"
  description = "Security group for EFS mount targets - ${var.name}"
  vpc_id      = var.vpc_id

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-efs-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# NFS ingress rule - Allow from specified CIDR blocks
resource "aws_security_group_rule" "efs_ingress_cidr" {
  count = length(var.allowed_cidr_blocks) > 0 ? 1 : 0

  type              = "ingress"
  from_port         = 2049
  to_port           = 2049
  protocol          = "tcp"
  cidr_blocks       = var.allowed_cidr_blocks
  security_group_id = aws_security_group.efs.id
  description       = "Allow NFS traffic from CIDR blocks"
}

# NFS ingress rule - Allow from specified security groups
resource "aws_security_group_rule" "efs_ingress_sg" {
  for_each = toset(var.allowed_security_group_ids)

  type                     = "ingress"
  from_port                = 2049
  to_port                  = 2049
  protocol                 = "tcp"
  source_security_group_id = each.value
  security_group_id        = aws_security_group.efs.id
  description              = "Allow NFS traffic from security group ${each.value}"
}

# Egress rule - Allow all outbound
resource "aws_security_group_rule" "efs_egress" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.efs.id
  description       = "Allow all outbound traffic"
}

#------------------------------------------------------------------------------
# EFS Access Points
#------------------------------------------------------------------------------

# GreenLang Artifacts Access Point
resource "aws_efs_access_point" "artifacts" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    uid            = var.access_points.artifacts.uid
    gid            = var.access_points.artifacts.gid
    secondary_gids = var.access_points.artifacts.secondary_gids
  }

  root_directory {
    path = var.access_points.artifacts.path

    creation_info {
      owner_uid   = var.access_points.artifacts.uid
      owner_gid   = var.access_points.artifacts.gid
      permissions = var.access_points.artifacts.permissions
    }
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.name}-artifacts"
      Application = "greenlang-artifacts"
    }
  )
}

# GreenLang Models Access Point
resource "aws_efs_access_point" "models" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    uid            = var.access_points.models.uid
    gid            = var.access_points.models.gid
    secondary_gids = var.access_points.models.secondary_gids
  }

  root_directory {
    path = var.access_points.models.path

    creation_info {
      owner_uid   = var.access_points.models.uid
      owner_gid   = var.access_points.models.gid
      permissions = var.access_points.models.permissions
    }
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.name}-models"
      Application = "greenlang-models"
    }
  )
}

# GreenLang Shared Access Point
resource "aws_efs_access_point" "shared" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    uid            = var.access_points.shared.uid
    gid            = var.access_points.shared.gid
    secondary_gids = var.access_points.shared.secondary_gids
  }

  root_directory {
    path = var.access_points.shared.path

    creation_info {
      owner_uid   = var.access_points.shared.uid
      owner_gid   = var.access_points.shared.gid
      permissions = var.access_points.shared.permissions
    }
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.name}-shared"
      Application = "greenlang-shared"
    }
  )
}

# GreenLang Tmp Access Point
resource "aws_efs_access_point" "tmp" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    uid            = var.access_points.tmp.uid
    gid            = var.access_points.tmp.gid
    secondary_gids = var.access_points.tmp.secondary_gids
  }

  root_directory {
    path = var.access_points.tmp.path

    creation_info {
      owner_uid   = var.access_points.tmp.uid
      owner_gid   = var.access_points.tmp.gid
      permissions = var.access_points.tmp.permissions
    }
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.name}-tmp"
      Application = "greenlang-tmp"
    }
  )
}

# Dynamic additional access points
resource "aws_efs_access_point" "additional" {
  for_each = var.additional_access_points

  file_system_id = aws_efs_file_system.main.id

  posix_user {
    uid            = each.value.uid
    gid            = each.value.gid
    secondary_gids = lookup(each.value, "secondary_gids", [])
  }

  root_directory {
    path = each.value.path

    creation_info {
      owner_uid   = each.value.uid
      owner_gid   = each.value.gid
      permissions = lookup(each.value, "permissions", "0755")
    }
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.name}-${each.key}"
      Application = each.key
    }
  )
}

#------------------------------------------------------------------------------
# EFS Backup Policy
#------------------------------------------------------------------------------

resource "aws_efs_backup_policy" "main" {
  file_system_id = aws_efs_file_system.main.id

  backup_policy {
    status = var.enable_backup ? "ENABLED" : "DISABLED"
  }
}

#------------------------------------------------------------------------------
# AWS Backup Integration
#------------------------------------------------------------------------------

resource "aws_backup_vault" "efs" {
  count = var.enable_backup && var.create_backup_vault ? 1 : 0

  name        = "${var.name}-efs-backup-vault"
  kms_key_arn = var.create_kms_key ? aws_kms_key.efs[0].arn : var.kms_key_id

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-efs-backup-vault"
    }
  )
}

resource "aws_backup_plan" "efs" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  name = "${var.name}-efs-backup-plan"

  rule {
    rule_name         = "daily-backup"
    target_vault_name = var.create_backup_vault ? aws_backup_vault.efs[0].name : var.backup_vault_name
    schedule          = var.backup_schedule

    lifecycle {
      cold_storage_after = var.backup_cold_storage_after
      delete_after       = var.backup_delete_after
    }

    recovery_point_tags = merge(
      var.tags,
      {
        BackupType = "EFS-Daily"
      }
    )

    copy_action {
      lifecycle {
        cold_storage_after = var.backup_copy_cold_storage_after
        delete_after       = var.backup_copy_delete_after
      }

      destination_vault_arn = var.backup_copy_destination_vault_arn != null ? var.backup_copy_destination_vault_arn : null
    }
  }

  # Weekly backup with longer retention
  rule {
    rule_name         = "weekly-backup"
    target_vault_name = var.create_backup_vault ? aws_backup_vault.efs[0].name : var.backup_vault_name
    schedule          = var.backup_weekly_schedule

    lifecycle {
      cold_storage_after = var.backup_weekly_cold_storage_after
      delete_after       = var.backup_weekly_delete_after
    }

    recovery_point_tags = merge(
      var.tags,
      {
        BackupType = "EFS-Weekly"
      }
    )
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-efs-backup-plan"
    }
  )
}

resource "aws_backup_selection" "efs" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  name         = "${var.name}-efs-backup-selection"
  plan_id      = aws_backup_plan.efs[0].id
  iam_role_arn = aws_iam_role.backup[0].arn

  resources = [
    aws_efs_file_system.main.arn
  ]

  condition {
    string_equals {
      key   = "aws:ResourceTag/BackupEnabled"
      value = "true"
    }
  }
}

#------------------------------------------------------------------------------
# EFS Replication Configuration
#------------------------------------------------------------------------------

resource "aws_efs_replication_configuration" "main" {
  count = var.enable_replication ? 1 : 0

  source_file_system_id = aws_efs_file_system.main.id

  destination {
    availability_zone_name = var.replication_availability_zone
    kms_key_id             = var.replication_kms_key_id
    region                 = var.replication_region
  }
}

#------------------------------------------------------------------------------
# EFS File System Policy
#------------------------------------------------------------------------------

resource "aws_efs_file_system_policy" "main" {
  count = var.enable_file_system_policy ? 1 : 0

  file_system_id                     = aws_efs_file_system.main.id
  bypass_policy_lockout_safety_check = var.bypass_policy_lockout_safety_check
  policy                             = data.aws_iam_policy_document.efs_policy[0].json
}

data "aws_iam_policy_document" "efs_policy" {
  count = var.enable_file_system_policy ? 1 : 0

  # Enforce encryption in transit
  statement {
    sid    = "EnforceEncryptionInTransit"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["*"]

    resources = [aws_efs_file_system.main.arn]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }

  # Allow access from EKS service accounts via IRSA
  statement {
    sid    = "AllowEKSAccess"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = var.efs_access_principal_arns
    }

    actions = [
      "elasticfilesystem:ClientMount",
      "elasticfilesystem:ClientWrite",
      "elasticfilesystem:ClientRootAccess"
    ]

    resources = [aws_efs_file_system.main.arn]

    condition {
      test     = "Bool"
      variable = "elasticfilesystem:AccessedViaMountTarget"
      values   = ["true"]
    }
  }

  # Deny anonymous access
  statement {
    sid    = "DenyAnonymousAccess"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = [
      "elasticfilesystem:ClientMount"
    ]

    resources = [aws_efs_file_system.main.arn]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

#------------------------------------------------------------------------------
# CloudWatch Alarms
#------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "burst_credit_balance" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.name}-efs-burst-credit-balance"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "BurstCreditBalance"
  namespace           = "AWS/EFS"
  period              = var.alarm_period
  statistic           = "Average"
  threshold           = var.burst_credit_balance_threshold
  alarm_description   = "EFS burst credit balance is low"
  alarm_actions       = var.alarm_actions

  dimensions = {
    FileSystemId = aws_efs_file_system.main.id
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "percent_io_limit" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.name}-efs-percent-io-limit"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "PercentIOLimit"
  namespace           = "AWS/EFS"
  period              = var.alarm_period
  statistic           = "Average"
  threshold           = var.percent_io_limit_threshold
  alarm_description   = "EFS IO limit percentage is high"
  alarm_actions       = var.alarm_actions

  dimensions = {
    FileSystemId = aws_efs_file_system.main.id
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "throughput_exceeded" {
  count = var.enable_cloudwatch_alarms && var.throughput_mode == "provisioned" ? 1 : 0

  alarm_name          = "${var.name}-efs-throughput-exceeded"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "MeteredIOBytes"
  namespace           = "AWS/EFS"
  period              = var.alarm_period
  statistic           = "Sum"
  threshold           = var.provisioned_throughput_in_mibps * 1024 * 1024 * var.alarm_period * 0.9
  alarm_description   = "EFS throughput is approaching provisioned limit"
  alarm_actions       = var.alarm_actions

  dimensions = {
    FileSystemId = aws_efs_file_system.main.id
  }

  tags = var.tags
}
