#############################################################################
# GreenLang S3 Security Module - IAM Policies and Roles
#############################################################################

#############################################################################
# S3 Admin Policy (with MFA Requirement)
#############################################################################

resource "aws_iam_policy" "s3_admin" {
  name        = "${var.project_name}-s3-admin-policy"
  description = "Full S3 administrative access with MFA requirement"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3FullAccessWithMFA"
        Effect = "Allow"
        Action = [
          "s3:*"
        ]
        Resource = [
          aws_s3_bucket.access_logs.arn,
          "${aws_s3_bucket.access_logs.arn}/*",
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*",
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*",
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
          NumericLessThan = {
            "aws:MultiFactorAuthAge" = "3600"
          }
        }
      },
      {
        Sid    = "S3ListBuckets"
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets",
          "s3:GetBucketLocation"
        ]
        Resource = "*"
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
        }
      },
      {
        Sid    = "KMSKeyAccess"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey",
          "kms:CreateGrant",
          "kms:ListGrants",
          "kms:RevokeGrant"
        ]
        Resource = aws_kms_key.s3_encryption.arn
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
        }
      },
      {
        Sid    = "DenyDeleteWithoutMFA"
        Effect = "Deny"
        Action = [
          "s3:DeleteBucket",
          "s3:DeleteBucketPolicy",
          "s3:DeleteObject",
          "s3:DeleteObjectVersion"
        ]
        Resource = "*"
        Condition = {
          BoolIfExists = {
            "aws:MultiFactorAuthPresent" = "false"
          }
        }
      }
    ]
  })

  tags = var.tags
}

#############################################################################
# S3 Read-Only Policy
#############################################################################

resource "aws_iam_policy" "s3_readonly" {
  name        = "${var.project_name}-s3-readonly-policy"
  description = "Read-only access to S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ReadOnlyAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging",
          "s3:GetObjectVersionTagging",
          "s3:GetObjectAcl",
          "s3:GetObjectVersionAcl",
          "s3:ListBucket",
          "s3:ListBucketVersions",
          "s3:GetBucketLocation",
          "s3:GetBucketAcl",
          "s3:GetBucketPolicy",
          "s3:GetBucketTagging",
          "s3:GetBucketVersioning",
          "s3:GetBucketLogging",
          "s3:GetEncryptionConfiguration",
          "s3:GetBucketPublicAccessBlock",
          "s3:GetBucketObjectLockConfiguration"
        ]
        Resource = [
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
        ]
      },
      {
        Sid    = "KMSDecryptAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.s3_encryption.arn
      },
      {
        Sid    = "DenyAuditLogAccess"
        Effect = "Deny"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*"
        ]
      }
    ]
  })

  tags = var.tags
}

#############################################################################
# S3 Write-Only Policy (Uploads Only)
#############################################################################

resource "aws_iam_policy" "s3_writeonly" {
  name        = "${var.project_name}-s3-writeonly-policy"
  description = "Write-only access for uploading to S3 (no read/delete)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3WriteOnlyAccess"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectTagging",
          "s3:PutObjectVersionTagging",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]
        Resource = [
          "${aws_s3_bucket.app_data.arn}/*",
          "${aws_s3_bucket.backups.arn}/*"
        ]
        Condition = {
          StringEquals = {
            "s3:x-amz-server-side-encryption" = "aws:kms"
          }
        }
      },
      {
        Sid    = "KMSEncryptAccess"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.s3_encryption.arn
      },
      {
        Sid    = "DenyReadAccess"
        Effect = "Deny"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]
        Resource = "*"
      },
      {
        Sid    = "DenyDeleteAccess"
        Effect = "Deny"
        Action = [
          "s3:DeleteObject",
          "s3:DeleteObjectVersion"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.tags
}

#############################################################################
# S3 Backup Manager Policy
#############################################################################

resource "aws_iam_policy" "s3_backup_manager" {
  name        = "${var.project_name}-s3-backup-manager-policy"
  description = "Policy for managing backups with Object Lock bypass (GOVERNANCE mode)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BackupBucketFullAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:ListBucketVersions",
          "s3:GetObjectRetention",
          "s3:GetObjectLegalHold",
          "s3:PutObjectRetention",
          "s3:BypassGovernanceRetention"
        ]
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
        }
      },
      {
        Sid    = "KMSFullAccess"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.s3_encryption.arn
      }
    ]
  })

  tags = var.tags
}

#############################################################################
# S3 Audit Reader Policy
#############################################################################

resource "aws_iam_policy" "s3_audit_reader" {
  name        = "${var.project_name}-s3-audit-reader-policy"
  description = "Read-only access to audit logs bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AuditLogsReadAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging",
          "s3:ListBucket",
          "s3:ListBucketVersions",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
        }
      },
      {
        Sid    = "AccessLogsReadAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.access_logs.arn,
          "${aws_s3_bucket.access_logs.arn}/*"
        ]
      },
      {
        Sid    = "KMSDecryptAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.s3_encryption.arn
      },
      {
        Sid    = "DenyWriteAccess"
        Effect = "Deny"
        Action = [
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:DeleteObjectVersion"
        ]
        Resource = [
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*"
        ]
      }
    ]
  })

  tags = var.tags
}

#############################################################################
# Cross-Account Assume Role
#############################################################################

resource "aws_iam_role" "s3_cross_account" {
  count = length(var.cross_account_ids) > 0 ? 1 : 0

  name        = "${var.project_name}-s3-cross-account-role"
  description = "Role for cross-account S3 access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCrossAccountAssume"
        Effect = "Allow"
        Principal = {
          AWS = [for account_id in var.cross_account_ids : "arn:aws:iam::${account_id}:root"]
        }
        Action = "sts:AssumeRole"
        Condition = {
          Bool = {
            "aws:MultiFactorAuthPresent" = "true"
          }
          StringEquals = {
            "sts:ExternalId" = "${var.project_name}-cross-account"
          }
        }
      }
    ]
  })

  max_session_duration = 3600

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "s3_cross_account_readonly" {
  count = length(var.cross_account_ids) > 0 ? 1 : 0

  role       = aws_iam_role.s3_cross_account[0].name
  policy_arn = aws_iam_policy.s3_readonly.arn
}

#############################################################################
# Service-Linked Role for S3 Storage Lens
#############################################################################

resource "aws_iam_service_linked_role" "s3_storage_lens" {
  count = var.enable_cloudwatch_metrics ? 1 : 0

  aws_service_name = "storage-lens.s3.amazonaws.com"
  description      = "Service-linked role for S3 Storage Lens"
}

#############################################################################
# S3 Replication Role
#############################################################################

resource "aws_iam_role" "s3_replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  name        = "${var.project_name}-s3-replication-role"
  description = "IAM role for S3 cross-region replication"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowS3Assume"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "s3_replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  name = "${var.project_name}-s3-replication-policy"
  role = aws_iam_role.s3_replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SourceBucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.backups.arn,
          aws_s3_bucket.app_data.arn
        ]
      },
      {
        Sid    = "SourceObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = [
          "${aws_s3_bucket.backups.arn}/*",
          "${aws_s3_bucket.app_data.arn}/*"
        ]
      },
      {
        Sid    = "DestinationBucketAccess"
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = "${var.replication_destination_bucket_arn}/*"
      },
      {
        Sid    = "KMSSourceDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = aws_kms_key.s3_encryption.arn
        Condition = {
          StringLike = {
            "kms:ViaService" = "s3.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      },
      {
        Sid    = "KMSDestinationEncrypt"
        Effect = "Allow"
        Action = [
          "kms:Encrypt"
        ]
        Resource = "*"
        Condition = {
          StringLike = {
            "kms:ViaService" = "s3.${var.replication_destination_region}.amazonaws.com"
          }
        }
      }
    ]
  })
}

#############################################################################
# Application Service Account Role
#############################################################################

resource "aws_iam_role" "s3_application" {
  name        = "${var.project_name}-s3-application-role"
  description = "IAM role for GreenLang application S3 access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEC2Assume"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
      {
        Sid    = "AllowEKSPodIdentity"
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/oidc.eks.${data.aws_region.current.name}.amazonaws.com"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "oidc.eks.${data.aws_region.current.name}.amazonaws.com:sub" = "system:serviceaccount:${var.project_name}:${var.project_name}-app"
          }
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "s3_application" {
  name = "${var.project_name}-s3-application-policy"
  role = aws_iam_role.s3_application.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AppDataReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.app_data.arn,
          "${aws_s3_bucket.app_data.arn}/*"
        ]
      },
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.s3_encryption.arn
      },
      {
        Sid    = "DenyAuditAndBackupAccess"
        Effect = "Deny"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.audit_logs.arn,
          "${aws_s3_bucket.audit_logs.arn}/*",
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
      }
    ]
  })
}

#############################################################################
# IAM Policy Outputs
#############################################################################

output "iam_policy_s3_admin_arn" {
  description = "ARN of the S3 admin policy"
  value       = aws_iam_policy.s3_admin.arn
}

output "iam_policy_s3_readonly_arn" {
  description = "ARN of the S3 read-only policy"
  value       = aws_iam_policy.s3_readonly.arn
}

output "iam_policy_s3_writeonly_arn" {
  description = "ARN of the S3 write-only policy"
  value       = aws_iam_policy.s3_writeonly.arn
}

output "iam_policy_s3_backup_manager_arn" {
  description = "ARN of the S3 backup manager policy"
  value       = aws_iam_policy.s3_backup_manager.arn
}

output "iam_policy_s3_audit_reader_arn" {
  description = "ARN of the S3 audit reader policy"
  value       = aws_iam_policy.s3_audit_reader.arn
}

output "iam_role_s3_application_arn" {
  description = "ARN of the application IAM role for S3 access"
  value       = aws_iam_role.s3_application.arn
}

output "iam_role_s3_cross_account_arn" {
  description = "ARN of the cross-account IAM role"
  value       = length(var.cross_account_ids) > 0 ? aws_iam_role.s3_cross_account[0].arn : null
}

output "iam_role_s3_replication_arn" {
  description = "ARN of the replication IAM role"
  value       = var.enable_cross_region_replication ? aws_iam_role.s3_replication[0].arn : null
}
