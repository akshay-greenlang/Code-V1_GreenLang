# GreenLang ElastiCache Module - Outputs

output "replication_group_id" {
  description = "The ID of the ElastiCache replication group"
  value       = aws_elasticache_replication_group.main.id
}

output "replication_group_arn" {
  description = "The ARN of the ElastiCache replication group"
  value       = aws_elasticache_replication_group.main.arn
}

output "primary_endpoint_address" {
  description = "The address of the primary endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "reader_endpoint_address" {
  description = "The address of the reader endpoint"
  value       = aws_elasticache_replication_group.main.reader_endpoint_address
}

output "configuration_endpoint_address" {
  description = "The configuration endpoint address (cluster mode only)"
  value       = var.cluster_mode_enabled ? aws_elasticache_replication_group.main.configuration_endpoint_address : null
}

output "port" {
  description = "The port the cluster is listening on"
  value       = var.port
}

output "member_clusters" {
  description = "List of member cluster IDs"
  value       = aws_elasticache_replication_group.main.member_clusters
}

output "security_group_id" {
  description = "The security group ID for the ElastiCache cluster"
  value       = aws_security_group.elasticache.id
}

output "parameter_group_name" {
  description = "The name of the parameter group"
  value       = aws_elasticache_parameter_group.main.name
}

output "kms_key_arn" {
  description = "The ARN of the KMS key used for encryption"
  value       = var.at_rest_encryption_enabled ? (var.kms_key_arn != null ? var.kms_key_arn : aws_kms_key.elasticache[0].arn) : null
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret containing auth token"
  value       = var.transit_encryption_enabled ? aws_secretsmanager_secret.redis[0].arn : null
}

output "secrets_manager_secret_name" {
  description = "Name of the Secrets Manager secret"
  value       = var.transit_encryption_enabled ? aws_secretsmanager_secret.redis[0].name : null
}

# Connection URLs
output "connection_url" {
  description = "Redis connection URL (with TLS if enabled)"
  value       = var.transit_encryption_enabled ? "rediss://${aws_elasticache_replication_group.main.primary_endpoint_address}:${var.port}" : "redis://${aws_elasticache_replication_group.main.primary_endpoint_address}:${var.port}"
  sensitive   = true
}

output "reader_connection_url" {
  description = "Redis reader connection URL (with TLS if enabled)"
  value       = var.transit_encryption_enabled ? "rediss://${aws_elasticache_replication_group.main.reader_endpoint_address}:${var.port}" : "redis://${aws_elasticache_replication_group.main.reader_endpoint_address}:${var.port}"
  sensitive   = true
}

# Engine Information
output "engine_version" {
  description = "The engine version"
  value       = aws_elasticache_replication_group.main.engine_version_actual
}

output "at_rest_encryption_enabled" {
  description = "Whether at-rest encryption is enabled"
  value       = aws_elasticache_replication_group.main.at_rest_encryption_enabled
}

output "transit_encryption_enabled" {
  description = "Whether in-transit encryption is enabled"
  value       = aws_elasticache_replication_group.main.transit_encryption_enabled
}
