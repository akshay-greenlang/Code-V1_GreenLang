#------------------------------------------------------------------------------
# AWS EFS Module - IAM Configuration
# GreenLang Infrastructure
#
# This file contains all IAM resources for EFS:
# - EFS access policy
# - IRSA role for EKS pods
# - Cross-account access policy
# - AWS Backup role
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# EFS Access Policy
#------------------------------------------------------------------------------

resource "aws_iam_policy" "efs_access" {
  name        = "${var.name}-efs-access-policy"
  description = "IAM policy for EFS access - ${var.name}"
  policy      = data.aws_iam_policy_document.efs_access.json

  tags = var.tags
}

data "aws_iam_policy_document" "efs_access" {
  # Allow describing EFS resources
  statement {
    sid    = "DescribeEFS"
    effect = "Allow"

    actions = [
      "elasticfilesystem:DescribeAccessPoints",
      "elasticfilesystem:DescribeFileSystems",
      "elasticfilesystem:DescribeMountTargets",
      "elasticfilesystem:DescribeMountTargetSecurityGroups",
      "elasticfilesystem:DescribeTags"
    ]

    resources = ["*"]
  }

  # Allow client operations on the EFS file system
  statement {
    sid    = "ClientAccess"
    effect = "Allow"

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

  # Allow access to specific access points
  statement {
    sid    = "AccessPointAccess"
    effect = "Allow"

    actions = [
      "elasticfilesystem:ClientMount",
      "elasticfilesystem:ClientWrite"
    ]

    resources = [
      aws_efs_access_point.artifacts.arn,
      aws_efs_access_point.models.arn,
      aws_efs_access_point.shared.arn,
      aws_efs_access_point.tmp.arn
    ]
  }

  # Allow KMS operations for encrypted EFS
  dynamic "statement" {
    for_each = var.encrypted ? [1] : []
    content {
      sid    = "KMSAccess"
      effect = "Allow"

      actions = [
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ]

      resources = [
        var.create_kms_key ? aws_kms_key.efs[0].arn : var.kms_key_id
      ]
    }
  }
}

#------------------------------------------------------------------------------
# IRSA Role for EKS Pods
#------------------------------------------------------------------------------

resource "aws_iam_role" "irsa" {
  count = var.eks_cluster_name != null ? 1 : 0

  name               = "${var.name}-efs-irsa-role"
  assume_role_policy = data.aws_iam_policy_document.irsa_assume_role[0].json

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-efs-irsa-role"
    }
  )
}

data "aws_iam_policy_document" "irsa_assume_role" {
  count = var.eks_cluster_name != null ? 1 : 0

  statement {
    sid     = "AssumeRoleWithWebIdentity"
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${replace(var.eks_cluster_oidc_issuer_url, "https://", "")}"]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(var.eks_cluster_oidc_issuer_url, "https://", "")}:sub"
      values   = ["system:serviceaccount:${var.eks_namespace}:${var.eks_service_account_name}"]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(var.eks_cluster_oidc_issuer_url, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "irsa_efs_access" {
  count = var.eks_cluster_name != null ? 1 : 0

  role       = aws_iam_role.irsa[0].name
  policy_arn = aws_iam_policy.efs_access.arn
}

# Additional policy for EFS CSI Driver
resource "aws_iam_policy" "efs_csi_driver" {
  count = var.eks_cluster_name != null ? 1 : 0

  name        = "${var.name}-efs-csi-driver-policy"
  description = "IAM policy for EFS CSI Driver - ${var.name}"
  policy      = data.aws_iam_policy_document.efs_csi_driver[0].json

  tags = var.tags
}

data "aws_iam_policy_document" "efs_csi_driver" {
  count = var.eks_cluster_name != null ? 1 : 0

  statement {
    sid    = "EFSCSIDriverAccess"
    effect = "Allow"

    actions = [
      "elasticfilesystem:DescribeAccessPoints",
      "elasticfilesystem:DescribeFileSystems",
      "elasticfilesystem:DescribeMountTargets",
      "ec2:DescribeAvailabilityZones"
    ]

    resources = ["*"]
  }

  statement {
    sid    = "CreateAccessPoint"
    effect = "Allow"

    actions = [
      "elasticfilesystem:CreateAccessPoint"
    ]

    resources = ["*"]

    condition {
      test     = "StringLike"
      variable = "aws:RequestTag/efs.csi.aws.com/cluster"
      values   = ["true"]
    }
  }

  statement {
    sid    = "TagAccessPoint"
    effect = "Allow"

    actions = [
      "elasticfilesystem:TagResource"
    ]

    resources = ["*"]

    condition {
      test     = "StringLike"
      variable = "aws:ResourceTag/efs.csi.aws.com/cluster"
      values   = ["true"]
    }
  }

  statement {
    sid    = "DeleteAccessPoint"
    effect = "Allow"

    actions = [
      "elasticfilesystem:DeleteAccessPoint"
    ]

    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/efs.csi.aws.com/cluster"
      values   = ["true"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "irsa_efs_csi_driver" {
  count = var.eks_cluster_name != null ? 1 : 0

  role       = aws_iam_role.irsa[0].name
  policy_arn = aws_iam_policy.efs_csi_driver[0].arn
}

#------------------------------------------------------------------------------
# Cross-Account Access Policy
#------------------------------------------------------------------------------

resource "aws_iam_policy" "cross_account_access" {
  count = var.enable_cross_account_access ? 1 : 0

  name        = "${var.name}-efs-cross-account-policy"
  description = "IAM policy for cross-account EFS access - ${var.name}"
  policy      = data.aws_iam_policy_document.cross_account_access[0].json

  tags = var.tags
}

data "aws_iam_policy_document" "cross_account_access" {
  count = var.enable_cross_account_access ? 1 : 0

  statement {
    sid    = "CrossAccountEFSAccess"
    effect = "Allow"

    actions = [
      "elasticfilesystem:ClientMount",
      "elasticfilesystem:ClientWrite",
      "elasticfilesystem:DescribeFileSystems",
      "elasticfilesystem:DescribeMountTargets"
    ]

    resources = [
      aws_efs_file_system.main.arn,
      "${aws_efs_file_system.main.arn}/*"
    ]

    condition {
      test     = "StringEquals"
      variable = "aws:PrincipalAccount"
      values   = var.cross_account_ids
    }
  }

  dynamic "statement" {
    for_each = var.encrypted ? [1] : []
    content {
      sid    = "CrossAccountKMSAccess"
      effect = "Allow"

      actions = [
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ]

      resources = [
        var.create_kms_key ? aws_kms_key.efs[0].arn : var.kms_key_id
      ]

      condition {
        test     = "StringEquals"
        variable = "aws:PrincipalAccount"
        values   = var.cross_account_ids
      }
    }
  }
}

#------------------------------------------------------------------------------
# AWS Backup IAM Role
#------------------------------------------------------------------------------

resource "aws_iam_role" "backup" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  name               = "${var.name}-efs-backup-role"
  assume_role_policy = data.aws_iam_policy_document.backup_assume_role[0].json

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-efs-backup-role"
    }
  )
}

data "aws_iam_policy_document" "backup_assume_role" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  statement {
    sid     = "AssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["backup.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "backup_efs" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  role       = aws_iam_role.backup[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}

resource "aws_iam_role_policy_attachment" "backup_restore" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  role       = aws_iam_role.backup[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForRestores"
}

# Custom backup policy for EFS-specific operations
resource "aws_iam_policy" "backup_custom" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  name        = "${var.name}-efs-backup-custom-policy"
  description = "Custom IAM policy for EFS backup operations - ${var.name}"
  policy      = data.aws_iam_policy_document.backup_custom[0].json

  tags = var.tags
}

data "aws_iam_policy_document" "backup_custom" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  # Allow backup operations on the EFS file system
  statement {
    sid    = "EFSBackupOperations"
    effect = "Allow"

    actions = [
      "elasticfilesystem:Backup",
      "elasticfilesystem:DescribeFileSystems"
    ]

    resources = [aws_efs_file_system.main.arn]
  }

  # Allow KMS operations for encrypted backups
  dynamic "statement" {
    for_each = var.encrypted ? [1] : []
    content {
      sid    = "KMSBackupOperations"
      effect = "Allow"

      actions = [
        "kms:Decrypt",
        "kms:Encrypt",
        "kms:GenerateDataKey",
        "kms:ReEncrypt*",
        "kms:DescribeKey",
        "kms:CreateGrant"
      ]

      resources = [
        var.create_kms_key ? aws_kms_key.efs[0].arn : var.kms_key_id
      ]

      condition {
        test     = "Bool"
        variable = "kms:GrantIsForAWSResource"
        values   = ["true"]
      }
    }
  }

  # Allow tagging backup resources
  statement {
    sid    = "BackupTagging"
    effect = "Allow"

    actions = [
      "backup:TagResource",
      "backup:UntagResource"
    ]

    resources = ["*"]
  }
}

resource "aws_iam_role_policy_attachment" "backup_custom" {
  count = var.enable_backup && var.create_backup_plan ? 1 : 0

  role       = aws_iam_role.backup[0].name
  policy_arn = aws_iam_policy.backup_custom[0].arn
}

#------------------------------------------------------------------------------
# Read-Only Access Policy
#------------------------------------------------------------------------------

resource "aws_iam_policy" "efs_read_only" {
  name        = "${var.name}-efs-read-only-policy"
  description = "IAM policy for read-only EFS access - ${var.name}"
  policy      = data.aws_iam_policy_document.efs_read_only.json

  tags = var.tags
}

data "aws_iam_policy_document" "efs_read_only" {
  statement {
    sid    = "DescribeEFS"
    effect = "Allow"

    actions = [
      "elasticfilesystem:DescribeAccessPoints",
      "elasticfilesystem:DescribeFileSystems",
      "elasticfilesystem:DescribeMountTargets",
      "elasticfilesystem:DescribeMountTargetSecurityGroups",
      "elasticfilesystem:DescribeTags",
      "elasticfilesystem:DescribeBackupPolicy",
      "elasticfilesystem:DescribeLifecycleConfiguration",
      "elasticfilesystem:DescribeFileSystemPolicy",
      "elasticfilesystem:DescribeReplicationConfigurations"
    ]

    resources = ["*"]
  }

  statement {
    sid    = "ClientMountReadOnly"
    effect = "Allow"

    actions = [
      "elasticfilesystem:ClientMount"
    ]

    resources = [aws_efs_file_system.main.arn]

    condition {
      test     = "Bool"
      variable = "elasticfilesystem:AccessedViaMountTarget"
      values   = ["true"]
    }
  }

  dynamic "statement" {
    for_each = var.encrypted ? [1] : []
    content {
      sid    = "KMSDecrypt"
      effect = "Allow"

      actions = [
        "kms:Decrypt"
      ]

      resources = [
        var.create_kms_key ? aws_kms_key.efs[0].arn : var.kms_key_id
      ]
    }
  }
}
