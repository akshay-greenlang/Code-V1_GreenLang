# GreenLang S3 Module - Outputs

# KMS Key
output "kms_key_arn" {
  description = "ARN of the KMS key for S3 encryption"
  value       = var.create_kms_key ? aws_kms_key.s3[0].arn : var.kms_key_arn
}

output "kms_key_id" {
  description = "ID of the KMS key for S3 encryption"
  value       = var.create_kms_key ? aws_kms_key.s3[0].key_id : null
}

output "kms_key_alias" {
  description = "Alias of the KMS key for S3 encryption"
  value       = var.create_kms_key ? aws_kms_alias.s3[0].name : null
}

# Artifacts Bucket
output "artifacts_bucket_id" {
  description = "ID of the artifacts bucket"
  value       = var.create_artifacts_bucket ? aws_s3_bucket.artifacts[0].id : null
}

output "artifacts_bucket_arn" {
  description = "ARN of the artifacts bucket"
  value       = var.create_artifacts_bucket ? aws_s3_bucket.artifacts[0].arn : null
}

output "artifacts_bucket_domain_name" {
  description = "Domain name of the artifacts bucket"
  value       = var.create_artifacts_bucket ? aws_s3_bucket.artifacts[0].bucket_domain_name : null
}

output "artifacts_bucket_regional_domain_name" {
  description = "Regional domain name of the artifacts bucket"
  value       = var.create_artifacts_bucket ? aws_s3_bucket.artifacts[0].bucket_regional_domain_name : null
}

# Logs Bucket
output "logs_bucket_id" {
  description = "ID of the logs bucket"
  value       = var.create_logs_bucket ? aws_s3_bucket.logs[0].id : null
}

output "logs_bucket_arn" {
  description = "ARN of the logs bucket"
  value       = var.create_logs_bucket ? aws_s3_bucket.logs[0].arn : null
}

output "logs_bucket_domain_name" {
  description = "Domain name of the logs bucket"
  value       = var.create_logs_bucket ? aws_s3_bucket.logs[0].bucket_domain_name : null
}

# Backups Bucket
output "backups_bucket_id" {
  description = "ID of the backups bucket"
  value       = var.create_backups_bucket ? aws_s3_bucket.backups[0].id : null
}

output "backups_bucket_arn" {
  description = "ARN of the backups bucket"
  value       = var.create_backups_bucket ? aws_s3_bucket.backups[0].arn : null
}

output "backups_bucket_domain_name" {
  description = "Domain name of the backups bucket"
  value       = var.create_backups_bucket ? aws_s3_bucket.backups[0].bucket_domain_name : null
}

# Data Bucket
output "data_bucket_id" {
  description = "ID of the data bucket"
  value       = var.create_data_bucket ? aws_s3_bucket.data[0].id : null
}

output "data_bucket_arn" {
  description = "ARN of the data bucket"
  value       = var.create_data_bucket ? aws_s3_bucket.data[0].arn : null
}

output "data_bucket_domain_name" {
  description = "Domain name of the data bucket"
  value       = var.create_data_bucket ? aws_s3_bucket.data[0].bucket_domain_name : null
}

output "data_bucket_regional_domain_name" {
  description = "Regional domain name of the data bucket"
  value       = var.create_data_bucket ? aws_s3_bucket.data[0].bucket_regional_domain_name : null
}

# Static Assets Bucket
output "static_assets_bucket_id" {
  description = "ID of the static assets bucket"
  value       = var.create_static_assets_bucket ? aws_s3_bucket.static_assets[0].id : null
}

output "static_assets_bucket_arn" {
  description = "ARN of the static assets bucket"
  value       = var.create_static_assets_bucket ? aws_s3_bucket.static_assets[0].arn : null
}

output "static_assets_bucket_domain_name" {
  description = "Domain name of the static assets bucket"
  value       = var.create_static_assets_bucket ? aws_s3_bucket.static_assets[0].bucket_domain_name : null
}

output "static_assets_bucket_regional_domain_name" {
  description = "Regional domain name of the static assets bucket"
  value       = var.create_static_assets_bucket ? aws_s3_bucket.static_assets[0].bucket_regional_domain_name : null
}

output "static_assets_website_endpoint" {
  description = "Website endpoint of the static assets bucket"
  value       = var.create_static_assets_bucket && var.enable_static_website ? aws_s3_bucket_website_configuration.static_assets[0].website_endpoint : null
}

# Replication Role
output "replication_role_arn" {
  description = "ARN of the S3 replication IAM role"
  value       = var.create_data_bucket && var.enable_replication ? aws_iam_role.replication[0].arn : null
}

# Reports Bucket
output "reports_bucket_id" {
  description = "ID of the reports bucket"
  value       = var.create_reports_bucket ? aws_s3_bucket.reports[0].id : null
}

output "reports_bucket_arn" {
  description = "ARN of the reports bucket"
  value       = var.create_reports_bucket ? aws_s3_bucket.reports[0].arn : null
}

output "reports_bucket_domain_name" {
  description = "Domain name of the reports bucket"
  value       = var.create_reports_bucket ? aws_s3_bucket.reports[0].bucket_domain_name : null
}

# Data Lake Raw Bucket
output "data_lake_raw_bucket_id" {
  description = "ID of the data lake raw bucket"
  value       = var.create_data_lake_buckets ? aws_s3_bucket.data_lake_raw[0].id : null
}

output "data_lake_raw_bucket_arn" {
  description = "ARN of the data lake raw bucket"
  value       = var.create_data_lake_buckets ? aws_s3_bucket.data_lake_raw[0].arn : null
}

# Data Lake Processed Bucket
output "data_lake_processed_bucket_id" {
  description = "ID of the data lake processed bucket"
  value       = var.create_data_lake_buckets ? aws_s3_bucket.data_lake_processed[0].id : null
}

output "data_lake_processed_bucket_arn" {
  description = "ARN of the data lake processed bucket"
  value       = var.create_data_lake_buckets ? aws_s3_bucket.data_lake_processed[0].arn : null
}

# All Bucket ARNs (for IAM policies)
output "all_bucket_arns" {
  description = "List of all bucket ARNs"
  value = compact([
    var.create_artifacts_bucket ? aws_s3_bucket.artifacts[0].arn : null,
    var.create_logs_bucket ? aws_s3_bucket.logs[0].arn : null,
    var.create_backups_bucket ? aws_s3_bucket.backups[0].arn : null,
    var.create_data_bucket ? aws_s3_bucket.data[0].arn : null,
    var.create_static_assets_bucket ? aws_s3_bucket.static_assets[0].arn : null,
    var.create_reports_bucket ? aws_s3_bucket.reports[0].arn : null,
    var.create_data_lake_buckets ? aws_s3_bucket.data_lake_raw[0].arn : null,
    var.create_data_lake_buckets ? aws_s3_bucket.data_lake_processed[0].arn : null
  ])
}

# All Bucket IDs
output "all_bucket_ids" {
  description = "Map of bucket purpose to bucket ID"
  value = {
    artifacts            = var.create_artifacts_bucket ? aws_s3_bucket.artifacts[0].id : null
    logs                 = var.create_logs_bucket ? aws_s3_bucket.logs[0].id : null
    backups              = var.create_backups_bucket ? aws_s3_bucket.backups[0].id : null
    data                 = var.create_data_bucket ? aws_s3_bucket.data[0].id : null
    static_assets        = var.create_static_assets_bucket ? aws_s3_bucket.static_assets[0].id : null
    reports              = var.create_reports_bucket ? aws_s3_bucket.reports[0].id : null
    data_lake_raw        = var.create_data_lake_buckets ? aws_s3_bucket.data_lake_raw[0].id : null
    data_lake_processed  = var.create_data_lake_buckets ? aws_s3_bucket.data_lake_processed[0].id : null
  }
}
