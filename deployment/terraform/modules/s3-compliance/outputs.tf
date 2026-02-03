#############################################################################
# GreenLang S3 Compliance Module - Outputs
#############################################################################

output "config_recorder_id" {
  description = "ID of the AWS Config recorder"
  value       = aws_config_configuration_recorder.main.id
}

output "config_delivery_channel_id" {
  description = "ID of the Config delivery channel"
  value       = aws_config_delivery_channel.main.id
}

output "config_snapshots_bucket" {
  description = "S3 bucket for Config snapshots"
  value = {
    id   = aws_s3_bucket.config_snapshots.id
    arn  = aws_s3_bucket.config_snapshots.arn
  }
}

output "config_rule_arns" {
  description = "ARNs of all Config rules"
  value = {
    ssl_requests_only           = aws_config_config_rule.s3_bucket_ssl_requests_only.arn
    server_side_encryption      = aws_config_config_rule.s3_bucket_server_side_encryption_enabled.arn
    public_read_prohibited      = aws_config_config_rule.s3_bucket_public_read_prohibited.arn
    public_write_prohibited     = aws_config_config_rule.s3_bucket_public_write_prohibited.arn
    versioning_enabled          = aws_config_config_rule.s3_bucket_versioning_enabled.arn
    logging_enabled             = aws_config_config_rule.s3_bucket_logging_enabled.arn
    default_encryption_kms      = aws_config_config_rule.s3_default_encryption_kms.arn
    account_public_access       = aws_config_config_rule.s3_account_level_public_access_blocks.arn
    bucket_level_public_access  = aws_config_config_rule.s3_bucket_level_public_access_prohibited.arn
    bucket_acl_prohibited       = aws_config_config_rule.s3_bucket_acl_prohibited.arn
    replication_enabled         = var.require_replication ? aws_config_config_rule.s3_bucket_replication_enabled[0].arn : null
  }
}

output "remediation_role_arn" {
  description = "ARN of the IAM role used for remediation"
  value       = aws_iam_role.remediation.arn
}

output "conformance_pack_name" {
  description = "Name of the conformance pack"
  value       = aws_config_conformance_pack.s3_best_practices.name
}

output "access_analyzer_arn" {
  description = "ARN of the S3 access analyzer"
  value       = aws_accessanalyzer_analyzer.s3.arn
}

output "compliance_event_rule_arn" {
  description = "ARN of the CloudWatch Event rule for compliance changes"
  value       = aws_cloudwatch_event_rule.config_compliance_change.arn
}
