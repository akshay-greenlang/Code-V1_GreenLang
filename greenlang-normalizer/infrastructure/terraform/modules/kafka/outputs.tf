# outputs.tf - MSK Module Outputs

output "cluster_arn" {
  description = "MSK cluster ARN"
  value       = aws_msk_cluster.main.arn
}

output "cluster_name" {
  description = "MSK cluster name"
  value       = aws_msk_cluster.main.cluster_name
}

output "bootstrap_brokers" {
  description = "Plaintext connection host:port pairs"
  value       = aws_msk_cluster.main.bootstrap_brokers
}

output "bootstrap_brokers_tls" {
  description = "TLS connection host:port pairs"
  value       = aws_msk_cluster.main.bootstrap_brokers_tls
}

output "bootstrap_brokers_sasl_scram" {
  description = "SASL/SCRAM connection host:port pairs"
  value       = aws_msk_cluster.main.bootstrap_brokers_sasl_scram
}

output "bootstrap_brokers_sasl_iam" {
  description = "SASL/IAM connection host:port pairs"
  value       = aws_msk_cluster.main.bootstrap_brokers_sasl_iam
}

output "zookeeper_connect_string" {
  description = "Zookeeper connection string"
  value       = aws_msk_cluster.main.zookeeper_connect_string
}

output "zookeeper_connect_string_tls" {
  description = "Zookeeper TLS connection string"
  value       = aws_msk_cluster.main.zookeeper_connect_string_tls
}

output "current_version" {
  description = "Current version of the MSK cluster"
  value       = aws_msk_cluster.main.current_version
}

output "security_group_id" {
  description = "Security group ID for the MSK cluster"
  value       = aws_security_group.msk.id
}

output "kms_key_arn" {
  description = "KMS key ARN used for encryption"
  value       = aws_kms_key.msk.arn
}

output "configuration_arn" {
  description = "MSK configuration ARN"
  value       = aws_msk_configuration.main.arn
}

output "configuration_revision" {
  description = "MSK configuration revision"
  value       = aws_msk_configuration.main.latest_revision
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.msk.name
}

output "s3_logs_bucket" {
  description = "S3 bucket for MSK logs"
  value       = var.enable_s3_logs ? aws_s3_bucket.msk_logs[0].id : null
}

output "scram_secret_arn" {
  description = "Secrets Manager secret ARN for SASL/SCRAM credentials"
  value       = var.enable_sasl_scram ? aws_secretsmanager_secret.msk_credentials[0].arn : null
}
