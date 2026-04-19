# RDS Module Outputs

output "cluster_id" {
  description = "RDS instance identifier"
  value       = aws_db_instance.main.id
}

output "cluster_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.main.arn
}

output "cluster_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "cluster_reader_endpoint" {
  description = "RDS read replica endpoints"
  value       = length(aws_db_instance.replica) > 0 ? aws_db_instance.replica[0].endpoint : aws_db_instance.main.endpoint
}

output "cluster_port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "cluster_database_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "cluster_master_username" {
  description = "Master username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

output "read_replica_endpoints" {
  description = "List of read replica endpoints"
  value       = aws_db_instance.replica[*].endpoint
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.rds.id
}

output "password_secret_arn" {
  description = "ARN of the Secrets Manager secret containing the password"
  value       = aws_secretsmanager_secret.rds_password.arn
}

output "parameter_group_name" {
  description = "Parameter group name"
  value       = aws_db_parameter_group.main.name
}

output "subnet_group_name" {
  description = "DB subnet group name"
  value       = aws_db_subnet_group.main.name
}
