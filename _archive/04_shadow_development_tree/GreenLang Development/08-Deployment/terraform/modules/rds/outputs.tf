# GreenLang RDS Module - Outputs

output "instance_id" {
  description = "The RDS instance ID"
  value       = aws_db_instance.main.id
}

output "instance_arn" {
  description = "The RDS instance ARN"
  value       = aws_db_instance.main.arn
}

output "endpoint" {
  description = "The connection endpoint"
  value       = aws_db_instance.main.endpoint
}

output "address" {
  description = "The hostname of the RDS instance"
  value       = aws_db_instance.main.address
}

output "port" {
  description = "The port the RDS instance is listening on"
  value       = aws_db_instance.main.port
}

output "database_name" {
  description = "The name of the default database"
  value       = aws_db_instance.main.db_name
}

output "username" {
  description = "The master username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

output "security_group_id" {
  description = "The security group ID for the RDS instance"
  value       = aws_security_group.rds.id
}

output "replica_endpoints" {
  description = "Endpoints for read replicas"
  value       = aws_db_instance.replica[*].endpoint
}

output "replica_addresses" {
  description = "Hostnames for read replicas"
  value       = aws_db_instance.replica[*].address
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret containing credentials"
  value       = aws_secretsmanager_secret.rds.arn
}

output "secrets_manager_secret_name" {
  description = "Name of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.rds.name
}

output "kms_key_arn" {
  description = "The ARN of the KMS key used for encryption"
  value       = var.kms_key_arn != null ? var.kms_key_arn : aws_kms_key.rds[0].arn
}

output "connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql://${aws_db_instance.main.username}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
  sensitive   = true
}
