#############################################################################
# GreenLang S3 Security Module - Outputs
#############################################################################

#############################################################################
# KMS Key Outputs
#############################################################################

output "kms_key_id" {
  description = "ID of the KMS key used for S3 encryption"
  value       = aws_kms_key.s3_encryption.id
}

output "kms_key_arn" {
  description = "ARN of the KMS key used for S3 encryption"
  value       = aws_kms_key.s3_encryption.arn
}

output "kms_key_alias" {
  description = "Alias of the KMS key"
  value       = aws_kms_alias.s3_encryption.name
}

#############################################################################
# Bucket Outputs
#############################################################################

output "access_logs_bucket" {
  description = "Access logs bucket configuration"
  value = {
    id          = aws_s3_bucket.access_logs.id
    arn         = aws_s3_bucket.access_logs.arn
    domain_name = aws_s3_bucket.access_logs.bucket_domain_name
  }
}

output "backups_bucket" {
  description = "Backups bucket configuration"
  value = {
    id               = aws_s3_bucket.backups.id
    arn              = aws_s3_bucket.backups.arn
    domain_name      = aws_s3_bucket.backups.bucket_domain_name
    object_lock_mode = "GOVERNANCE"
    retention_days   = var.backup_retention_days
  }
}

output "audit_logs_bucket" {
  description = "Audit logs bucket configuration"
  value = {
    id               = aws_s3_bucket.audit_logs.id
    arn              = aws_s3_bucket.audit_logs.arn
    domain_name      = aws_s3_bucket.audit_logs.bucket_domain_name
    object_lock_mode = "COMPLIANCE"
    retention_years  = var.audit_log_retention_years
  }
}

output "app_data_bucket" {
  description = "Application data bucket configuration"
  value = {
    id          = aws_s3_bucket.app_data.id
    arn         = aws_s3_bucket.app_data.arn
    domain_name = aws_s3_bucket.app_data.bucket_domain_name
  }
}

output "all_bucket_arns" {
  description = "List of all bucket ARNs created by this module"
  value = [
    aws_s3_bucket.access_logs.arn,
    aws_s3_bucket.backups.arn,
    aws_s3_bucket.audit_logs.arn,
    aws_s3_bucket.app_data.arn
  ]
}

output "all_bucket_ids" {
  description = "List of all bucket IDs created by this module"
  value = [
    aws_s3_bucket.access_logs.id,
    aws_s3_bucket.backups.id,
    aws_s3_bucket.audit_logs.id,
    aws_s3_bucket.app_data.id
  ]
}

#############################################################################
# VPC Endpoint Outputs
#############################################################################

output "s3_vpc_endpoint_id" {
  description = "ID of the S3 VPC endpoint"
  value       = var.vpc_id != "" ? aws_vpc_endpoint.s3[0].id : null
}

output "s3_vpc_endpoint_dns_entry" {
  description = "DNS entry for the S3 VPC endpoint"
  value       = var.vpc_id != "" ? aws_vpc_endpoint.s3[0].dns_entry : null
}

output "s3_vpc_endpoint_policy" {
  description = "Policy document attached to the S3 VPC endpoint"
  value       = var.vpc_id != "" ? aws_vpc_endpoint.s3[0].policy : null
}

#############################################################################
# Access Point Outputs
#############################################################################

output "app_data_readonly_access_point" {
  description = "Read-only access point for app data bucket"
  value = {
    arn         = aws_s3_access_point.app_data_readonly.arn
    alias       = aws_s3_access_point.app_data_readonly.alias
    domain_name = aws_s3_access_point.app_data_readonly.domain_name
  }
}

output "app_data_write_access_point" {
  description = "Write access point for app data bucket"
  value = {
    arn         = aws_s3_access_point.app_data_write.arn
    alias       = aws_s3_access_point.app_data_write.alias
    domain_name = aws_s3_access_point.app_data_write.domain_name
  }
}

#############################################################################
# Object Lock Configuration Outputs
#############################################################################

output "backups_object_lock_config" {
  description = "Object lock configuration for backups bucket"
  value = {
    mode           = "GOVERNANCE"
    retention_days = var.backup_retention_days
  }
}

output "audit_logs_object_lock_config" {
  description = "Object lock configuration for audit logs bucket"
  value = {
    mode            = "COMPLIANCE"
    retention_years = var.audit_log_retention_years
  }
}

#############################################################################
# Policy Documents (for use in other modules)
#############################################################################

output "bucket_policy_deny_non_ssl" {
  description = "Reusable policy statement to deny non-SSL requests"
  value = jsonencode({
    Sid    = "DenyNonSSLRequests"
    Effect = "Deny"
    Principal = "*"
    Action = "s3:*"
    Resource = [
      "BUCKET_ARN",
      "BUCKET_ARN/*"
    ]
    Condition = {
      Bool = {
        "aws:SecureTransport" = "false"
      }
    }
  })
}

output "bucket_policy_deny_unencrypted" {
  description = "Reusable policy statement to deny unencrypted uploads"
  value = jsonencode({
    Sid    = "DenyUnencryptedUploads"
    Effect = "Deny"
    Principal = "*"
    Action = "s3:PutObject"
    Resource = "BUCKET_ARN/*"
    Condition = {
      Null = {
        "s3:x-amz-server-side-encryption" = "true"
      }
    }
  })
}

output "bucket_policy_vpc_endpoint_only" {
  description = "Reusable policy statement to restrict access to VPC endpoint"
  value = jsonencode({
    Sid    = "RestrictToVPCEndpoint"
    Effect = "Deny"
    Principal = "*"
    Action = "s3:*"
    Resource = [
      "BUCKET_ARN",
      "BUCKET_ARN/*"
    ]
    Condition = {
      StringNotEquals = {
        "aws:SourceVpce" = "VPC_ENDPOINT_ID"
      }
    }
  })
}

#############################################################################
# CloudWatch Alarm Outputs
#############################################################################

output "cloudwatch_alarm_arns" {
  description = "ARNs of CloudWatch alarms for S3 security monitoring"
  value = [
    aws_cloudwatch_metric_alarm.s3_public_access_attempt.arn,
    aws_cloudwatch_metric_alarm.s3_unauthorized_access.arn
  ]
}

#############################################################################
# MFA Delete Documentation Output
#############################################################################

output "mfa_delete_setup_commands" {
  description = "AWS CLI commands to enable MFA delete (must be run as root user)"
  value = {
    backups_bucket = <<-EOT
      # Enable MFA Delete for backups bucket (requires root credentials)
      aws s3api put-bucket-versioning \
        --bucket ${aws_s3_bucket.backups.id} \
        --versioning-configuration Status=Enabled,MFADelete=Enabled \
        --mfa "arn:aws:iam::${data.aws_caller_identity.current.account_id}:mfa/root-account-mfa-device TOTP_CODE"
    EOT
    audit_logs_bucket = <<-EOT
      # Enable MFA Delete for audit logs bucket (requires root credentials)
      aws s3api put-bucket-versioning \
        --bucket ${aws_s3_bucket.audit_logs.id} \
        --versioning-configuration Status=Enabled,MFADelete=Enabled \
        --mfa "arn:aws:iam::${data.aws_caller_identity.current.account_id}:mfa/root-account-mfa-device TOTP_CODE"
    EOT
  }
  sensitive = false
}
