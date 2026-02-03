################################################################################
# Aurora PostgreSQL Module - Outputs
################################################################################

################################################################################
# Cluster Outputs
################################################################################

output "cluster_id" {
  description = "The Aurora cluster identifier"
  value       = aws_rds_cluster.aurora.id
}

output "cluster_arn" {
  description = "The ARN of the Aurora cluster"
  value       = aws_rds_cluster.aurora.arn
}

output "cluster_resource_id" {
  description = "The Resource ID of the Aurora cluster"
  value       = aws_rds_cluster.aurora.cluster_resource_id
}

################################################################################
# Endpoint Outputs
################################################################################

output "cluster_endpoint" {
  description = "The cluster endpoint (writer)"
  value       = aws_rds_cluster.aurora.endpoint
}

output "cluster_reader_endpoint" {
  description = "The reader endpoint for the cluster"
  value       = aws_rds_cluster.aurora.reader_endpoint
}

output "cluster_port" {
  description = "The port number on which the DB accepts connections"
  value       = aws_rds_cluster.aurora.port
}

output "writer_endpoint" {
  description = "The endpoint of the writer instance"
  value       = aws_rds_cluster_instance.writer.endpoint
}

output "reader_endpoints" {
  description = "List of reader instance endpoints"
  value       = aws_rds_cluster_instance.readers[*].endpoint
}

output "all_instance_endpoints" {
  description = "Map of all instance endpoints (writer and readers)"
  value = merge(
    { writer = aws_rds_cluster_instance.writer.endpoint },
    { for idx, reader in aws_rds_cluster_instance.readers : "reader-${idx + 1}" => reader.endpoint }
  )
}

output "custom_reader_endpoint" {
  description = "Custom reader endpoint (if created)"
  value       = var.create_reader_endpoint ? aws_rds_cluster_endpoint.readers[0].endpoint : null
}

################################################################################
# Instance Outputs
################################################################################

output "writer_instance_id" {
  description = "The identifier of the writer instance"
  value       = aws_rds_cluster_instance.writer.id
}

output "writer_instance_arn" {
  description = "The ARN of the writer instance"
  value       = aws_rds_cluster_instance.writer.arn
}

output "reader_instance_ids" {
  description = "List of reader instance identifiers"
  value       = aws_rds_cluster_instance.readers[*].id
}

output "reader_instance_arns" {
  description = "List of reader instance ARNs"
  value       = aws_rds_cluster_instance.readers[*].arn
}

output "all_instance_ids" {
  description = "List of all instance identifiers"
  value = concat(
    [aws_rds_cluster_instance.writer.id],
    aws_rds_cluster_instance.readers[*].id
  )
}

################################################################################
# Security Group Outputs
################################################################################

output "security_group_id" {
  description = "The ID of the Aurora security group"
  value       = aws_security_group.aurora.id
}

output "security_group_arn" {
  description = "The ARN of the Aurora security group"
  value       = aws_security_group.aurora.arn
}

output "security_group_name" {
  description = "The name of the Aurora security group"
  value       = aws_security_group.aurora.name
}

################################################################################
# Subnet Group Outputs
################################################################################

output "db_subnet_group_name" {
  description = "The name of the DB subnet group"
  value       = aws_db_subnet_group.aurora.name
}

output "db_subnet_group_arn" {
  description = "The ARN of the DB subnet group"
  value       = aws_db_subnet_group.aurora.arn
}

################################################################################
# Parameter Group Outputs
################################################################################

output "cluster_parameter_group_name" {
  description = "The name of the cluster parameter group"
  value       = aws_rds_cluster_parameter_group.aurora_timescaledb.name
}

output "cluster_parameter_group_arn" {
  description = "The ARN of the cluster parameter group"
  value       = aws_rds_cluster_parameter_group.aurora_timescaledb.arn
}

output "instance_parameter_group_name" {
  description = "The name of the instance parameter group"
  value       = aws_db_parameter_group.aurora_timescaledb.name
}

output "instance_parameter_group_arn" {
  description = "The ARN of the instance parameter group"
  value       = aws_db_parameter_group.aurora_timescaledb.arn
}

################################################################################
# Secrets Manager Outputs
################################################################################

output "master_credentials_secret_arn" {
  description = "The ARN of the master credentials secret in Secrets Manager"
  value       = aws_secretsmanager_secret.master_credentials.arn
}

output "master_credentials_secret_name" {
  description = "The name of the master credentials secret"
  value       = aws_secretsmanager_secret.master_credentials.name
}

output "application_credentials_secret_arn" {
  description = "The ARN of the application credentials secret (if created)"
  value       = var.create_application_credentials ? aws_secretsmanager_secret.application_credentials[0].arn : null
}

output "application_credentials_secret_name" {
  description = "The name of the application credentials secret (if created)"
  value       = var.create_application_credentials ? aws_secretsmanager_secret.application_credentials[0].name : null
}

################################################################################
# KMS Outputs
################################################################################

output "kms_key_arn" {
  description = "The ARN of the KMS key used for encryption"
  value       = var.create_kms_key ? aws_kms_key.aurora[0].arn : var.kms_key_arn
}

output "kms_key_id" {
  description = "The ID of the KMS key used for encryption"
  value       = var.create_kms_key ? aws_kms_key.aurora[0].key_id : null
}

output "kms_key_alias" {
  description = "The alias of the KMS key"
  value       = var.create_kms_key ? aws_kms_alias.aurora[0].name : null
}

################################################################################
# IAM Role Outputs
################################################################################

output "enhanced_monitoring_role_arn" {
  description = "The ARN of the enhanced monitoring IAM role"
  value       = aws_iam_role.aurora_enhanced_monitoring.arn
}

output "enhanced_monitoring_role_name" {
  description = "The name of the enhanced monitoring IAM role"
  value       = aws_iam_role.aurora_enhanced_monitoring.name
}

output "s3_export_role_arn" {
  description = "The ARN of the S3 export IAM role (if created)"
  value       = var.enable_s3_export ? aws_iam_role.aurora_s3_export[0].arn : null
}

output "s3_export_role_name" {
  description = "The name of the S3 export IAM role (if created)"
  value       = var.enable_s3_export ? aws_iam_role.aurora_s3_export[0].name : null
}

output "lambda_integration_role_arn" {
  description = "The ARN of the Lambda integration IAM role"
  value       = aws_iam_role.aurora_lambda_integration.arn
}

output "lambda_integration_role_name" {
  description = "The name of the Lambda integration IAM role"
  value       = aws_iam_role.aurora_lambda_integration.name
}

################################################################################
# CloudWatch Outputs
################################################################################

output "cloudwatch_log_group_name" {
  description = "The name of the CloudWatch log group for PostgreSQL logs"
  value       = "/aws/rds/cluster/${aws_rds_cluster.aurora.cluster_identifier}/postgresql"
}

output "cloudwatch_alarm_arns" {
  description = "Map of CloudWatch alarm ARNs"
  value = var.create_cloudwatch_alarms ? {
    cpu_utilization_high    = aws_cloudwatch_metric_alarm.cpu_utilization_high[0].arn
    connections_high        = aws_cloudwatch_metric_alarm.connections_high[0].arn
    freeable_memory_low     = aws_cloudwatch_metric_alarm.freeable_memory_low[0].arn
    free_storage_space_low  = aws_cloudwatch_metric_alarm.free_storage_space_low[0].arn
    read_iops_high          = aws_cloudwatch_metric_alarm.read_iops_high[0].arn
    write_iops_high         = aws_cloudwatch_metric_alarm.write_iops_high[0].arn
    replication_lag_high    = var.replica_count > 0 ? aws_cloudwatch_metric_alarm.replication_lag_high[0].arn : null
  } : {}
}

################################################################################
# Connection Strings
################################################################################

output "connection_string" {
  description = "PostgreSQL connection string for the writer endpoint"
  value       = "postgresql://${local.master_username}@${aws_rds_cluster.aurora.endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
  sensitive   = true
}

output "reader_connection_string" {
  description = "PostgreSQL connection string for the reader endpoint"
  value       = "postgresql://${local.master_username}@${aws_rds_cluster.aurora.reader_endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
  sensitive   = true
}

output "jdbc_connection_string" {
  description = "JDBC connection string for the writer endpoint"
  value       = "jdbc:postgresql://${aws_rds_cluster.aurora.endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
}

output "jdbc_reader_connection_string" {
  description = "JDBC connection string for the reader endpoint"
  value       = "jdbc:postgresql://${aws_rds_cluster.aurora.reader_endpoint}:${aws_rds_cluster.aurora.port}/${var.database_name}"
}

################################################################################
# Database Information
################################################################################

output "database_name" {
  description = "The name of the default database"
  value       = var.database_name
}

output "master_username" {
  description = "The master username"
  value       = local.master_username
  sensitive   = true
}

output "engine_version" {
  description = "The Aurora PostgreSQL engine version"
  value       = aws_rds_cluster.aurora.engine_version_actual
}

output "availability_zones" {
  description = "List of availability zones used by the cluster"
  value       = aws_rds_cluster.aurora.availability_zones
}
