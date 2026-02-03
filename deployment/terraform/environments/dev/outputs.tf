# GreenLang Development Environment - Outputs

# -----------------------------------------------------------------------------
# VPC Outputs
# -----------------------------------------------------------------------------
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "database_subnet_ids" {
  description = "Database subnet IDs"
  value       = module.vpc.database_subnet_ids
}

output "nat_gateway_ids" {
  description = "NAT Gateway IDs"
  value       = module.vpc.nat_gateway_ids
}

# -----------------------------------------------------------------------------
# EKS Outputs
# -----------------------------------------------------------------------------
output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "eks_cluster_certificate_authority_data" {
  description = "EKS cluster CA data"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "eks_oidc_provider_arn" {
  description = "EKS OIDC provider ARN"
  value       = module.eks.oidc_provider_arn
}

output "eks_oidc_provider_url" {
  description = "EKS OIDC provider URL"
  value       = module.eks.oidc_provider_url
}

output "eks_cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

# -----------------------------------------------------------------------------
# RDS Outputs
# -----------------------------------------------------------------------------
output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.endpoint
}

output "rds_address" {
  description = "RDS hostname"
  value       = module.rds.address
}

output "rds_port" {
  description = "RDS port"
  value       = module.rds.port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.database_name
}

output "rds_secrets_manager_arn" {
  description = "RDS Secrets Manager secret ARN"
  value       = module.rds.secrets_manager_secret_arn
}

output "rds_security_group_id" {
  description = "RDS security group ID"
  value       = module.rds.security_group_id
}

# -----------------------------------------------------------------------------
# ElastiCache Outputs
# -----------------------------------------------------------------------------
output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = module.elasticache.primary_endpoint_address
}

output "redis_port" {
  description = "Redis port"
  value       = module.elasticache.port
}

output "redis_security_group_id" {
  description = "Redis security group ID"
  value       = module.elasticache.security_group_id
}

# -----------------------------------------------------------------------------
# S3 Outputs
# -----------------------------------------------------------------------------
output "s3_buckets" {
  description = "S3 bucket IDs"
  value       = module.s3.all_bucket_ids
}

output "s3_artifacts_bucket_arn" {
  description = "Artifacts bucket ARN"
  value       = module.s3.artifacts_bucket_arn
}

output "s3_data_bucket_arn" {
  description = "Data bucket ARN"
  value       = module.s3.data_bucket_arn
}

output "s3_logs_bucket_arn" {
  description = "Logs bucket ARN"
  value       = module.s3.logs_bucket_arn
}

output "s3_backups_bucket_arn" {
  description = "Backups bucket ARN"
  value       = module.s3.backups_bucket_arn
}

output "s3_kms_key_arn" {
  description = "S3 KMS key ARN"
  value       = module.s3.kms_key_arn
}

# -----------------------------------------------------------------------------
# IAM Outputs
# -----------------------------------------------------------------------------
output "app_service_account_role_arn" {
  description = "IAM role ARN for application service account"
  value       = module.iam.app_service_account_role_arn
}

output "agent_service_account_role_arn" {
  description = "IAM role ARN for agent service account"
  value       = module.iam.agent_service_account_role_arn
}

output "cicd_role_arn" {
  description = "IAM role ARN for CI/CD"
  value       = module.iam.cicd_deployment_role_arn
}

output "external_secrets_role_arn" {
  description = "IAM role ARN for External Secrets Operator"
  value       = module.iam.external_secrets_role_arn
}

output "monitoring_role_arn" {
  description = "IAM role ARN for monitoring"
  value       = module.iam.monitoring_role_arn
}

# -----------------------------------------------------------------------------
# Service Account Annotations
# -----------------------------------------------------------------------------
output "app_service_account_annotation" {
  description = "Kubernetes annotation for app service account"
  value       = module.iam.app_service_account_annotation
}

output "agent_service_account_annotation" {
  description = "Kubernetes annotation for agent service account"
  value       = module.iam.agent_service_account_annotation
}

output "external_secrets_service_account_annotation" {
  description = "Kubernetes annotation for External Secrets service account"
  value       = module.iam.external_secrets_service_account_annotation
}

output "monitoring_service_account_annotation" {
  description = "Kubernetes annotation for monitoring service account"
  value       = module.iam.monitoring_service_account_annotation
}

# Note: For dev, static_assets_bucket is not created
# Connection strings should be retrieved from Secrets Manager in practice
