# ============================================================================
# GreenLang AI Agent Foundation - AWS Infrastructure Outputs
# ============================================================================

# ============================================================================
# VPC Outputs
# ============================================================================

output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr" {
  description = "The CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnets
}

output "database_subnets" {
  description = "List of database subnet IDs"
  value       = module.vpc.database_subnets
}

# ============================================================================
# EKS Outputs
# ============================================================================

output "eks_cluster_id" {
  description = "The name of the EKS cluster"
  value       = module.eks.cluster_id
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_version" {
  description = "The Kubernetes server version for the cluster"
  value       = module.eks.cluster_version
}

output "eks_cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "eks_node_security_group_id" {
  description = "Security group ID attached to the EKS nodes"
  value       = module.eks.node_security_group_id
}

output "eks_oidc_provider_arn" {
  description = "ARN of the OIDC Provider for EKS"
  value       = module.eks.oidc_provider_arn
}

# ============================================================================
# RDS Outputs
# ============================================================================

output "rds_endpoint" {
  description = "The connection endpoint for the RDS instance"
  value       = module.rds.db_instance_endpoint
}

output "rds_address" {
  description = "The address of the RDS instance"
  value       = module.rds.db_instance_address
}

output "rds_port" {
  description = "The database port"
  value       = module.rds.db_instance_port
}

output "rds_database_name" {
  description = "The database name"
  value       = module.rds.db_instance_name
}

output "rds_username" {
  description = "The master username for the database"
  value       = module.rds.db_instance_username
  sensitive   = true
}

output "rds_security_group_id" {
  description = "The security group ID of the RDS instance"
  value       = aws_security_group.rds.id
}

# ============================================================================
# ElastiCache Redis Outputs
# ============================================================================

output "redis_endpoint" {
  description = "The primary endpoint of the Redis cluster"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "redis_port" {
  description = "The port number on which the Redis cluster accepts connections"
  value       = aws_elasticache_replication_group.redis.port
}

output "redis_reader_endpoint" {
  description = "The reader endpoint for the Redis cluster"
  value       = aws_elasticache_replication_group.redis.reader_endpoint_address
}

output "redis_security_group_id" {
  description = "The security group ID of the Redis cluster"
  value       = aws_security_group.redis.id
}

# ============================================================================
# S3 Outputs
# ============================================================================

output "s3_bucket_id" {
  description = "The name of the S3 bucket for data storage"
  value       = module.s3_bucket.s3_bucket_id
}

output "s3_bucket_arn" {
  description = "The ARN of the S3 bucket"
  value       = module.s3_bucket.s3_bucket_arn
}

output "s3_logs_bucket_id" {
  description = "The name of the S3 bucket for logs"
  value       = module.s3_logs_bucket.s3_bucket_id
}

# ============================================================================
# Secrets Manager Outputs
# ============================================================================

output "secrets_manager_arn" {
  description = "ARN of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.app_secrets.arn
}

output "secrets_manager_name" {
  description = "Name of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.app_secrets.name
}

# ============================================================================
# IAM Outputs
# ============================================================================

output "irsa_role_arn" {
  description = "ARN of the IAM role for service accounts"
  value       = module.irsa_role.iam_role_arn
}

output "irsa_role_name" {
  description = "Name of the IAM role for service accounts"
  value       = module.irsa_role.iam_role_name
}

# ============================================================================
# KMS Outputs
# ============================================================================

output "kms_eks_key_id" {
  description = "The globally unique identifier for the EKS KMS key"
  value       = aws_kms_key.eks.key_id
}

output "kms_eks_key_arn" {
  description = "The Amazon Resource Name (ARN) of the EKS KMS key"
  value       = aws_kms_key.eks.arn
}

output "kms_s3_key_id" {
  description = "The globally unique identifier for the S3 KMS key"
  value       = aws_kms_key.s3.key_id
}

output "kms_s3_key_arn" {
  description = "The Amazon Resource Name (ARN) of the S3 KMS key"
  value       = aws_kms_key.s3.arn
}

# ============================================================================
# CloudWatch Outputs
# ============================================================================

output "cloudwatch_log_group_application" {
  description = "Name of the CloudWatch log group for application logs"
  value       = aws_cloudwatch_log_group.application.name
}

output "cloudwatch_log_group_redis" {
  description = "Name of the CloudWatch log group for Redis slow logs"
  value       = aws_cloudwatch_log_group.redis_slow_log.name
}

# ============================================================================
# Connection Strings (for Kubernetes secrets)
# ============================================================================

output "database_connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql://${module.rds.db_instance_username}:REPLACE_WITH_PASSWORD@${module.rds.db_instance_endpoint}/${module.rds.db_instance_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://${aws_elasticache_replication_group.redis.primary_endpoint_address}:${aws_elasticache_replication_group.redis.port}"
}

# ============================================================================
# kubectl Configuration Command
# ============================================================================

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_id}"
}
