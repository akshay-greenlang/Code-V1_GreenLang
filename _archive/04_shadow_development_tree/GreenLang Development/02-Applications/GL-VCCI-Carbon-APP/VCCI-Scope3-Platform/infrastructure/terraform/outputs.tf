# Root Module Outputs
# GL-VCCI Scope 3 Carbon Intelligence Platform

# ============================================================================
# VPC Outputs
# ============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "database_subnet_ids" {
  description = "List of database subnet IDs"
  value       = module.vpc.database_subnet_ids
}

output "nat_gateway_ids" {
  description = "List of NAT Gateway IDs"
  value       = module.vpc.nat_gateway_ids
}

# ============================================================================
# EKS Outputs
# ============================================================================

output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "eks_cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "eks_cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster OIDC Issuer"
  value       = module.eks.cluster_oidc_issuer_url
}

output "eks_node_group_ids" {
  description = "EKS node group IDs"
  value       = module.eks.node_group_ids
}

output "eks_cluster_autoscaler_role_arn" {
  description = "ARN of IAM role for cluster autoscaler"
  value       = module.eks.cluster_autoscaler_role_arn
}

output "eks_load_balancer_controller_role_arn" {
  description = "ARN of IAM role for AWS Load Balancer Controller"
  value       = module.eks.load_balancer_controller_role_arn
}

# ============================================================================
# RDS Outputs
# ============================================================================

output "rds_cluster_endpoint" {
  description = "RDS cluster endpoint"
  value       = module.rds.cluster_endpoint
}

output "rds_cluster_reader_endpoint" {
  description = "RDS cluster reader endpoint"
  value       = module.rds.cluster_reader_endpoint
}

output "rds_cluster_id" {
  description = "RDS cluster identifier"
  value       = module.rds.cluster_id
}

output "rds_cluster_arn" {
  description = "RDS cluster ARN"
  value       = module.rds.cluster_arn
}

output "rds_cluster_port" {
  description = "RDS cluster port"
  value       = module.rds.cluster_port
}

output "rds_cluster_database_name" {
  description = "RDS cluster database name"
  value       = module.rds.cluster_database_name
}

output "rds_cluster_master_username" {
  description = "RDS cluster master username"
  value       = module.rds.cluster_master_username
  sensitive   = true
}

output "rds_read_replica_endpoints" {
  description = "List of RDS read replica endpoints"
  value       = module.rds.read_replica_endpoints
}

output "rds_security_group_id" {
  description = "Security group ID for RDS cluster"
  value       = module.rds.security_group_id
}

# ============================================================================
# ElastiCache Outputs
# ============================================================================

output "elasticache_cluster_id" {
  description = "ElastiCache cluster ID"
  value       = module.elasticache.cluster_id
}

output "elasticache_cluster_endpoint" {
  description = "ElastiCache cluster configuration endpoint"
  value       = module.elasticache.cluster_configuration_endpoint
}

output "elasticache_cluster_reader_endpoint" {
  description = "ElastiCache cluster reader endpoint"
  value       = module.elasticache.cluster_reader_endpoint
}

output "elasticache_cluster_arn" {
  description = "ElastiCache cluster ARN"
  value       = module.elasticache.cluster_arn
}

output "elasticache_cluster_port" {
  description = "ElastiCache cluster port"
  value       = module.elasticache.cluster_port
}

output "elasticache_security_group_id" {
  description = "Security group ID for ElastiCache cluster"
  value       = module.elasticache.security_group_id
}

# ============================================================================
# S3 Outputs
# ============================================================================

output "s3_provenance_bucket_id" {
  description = "ID of the provenance records bucket"
  value       = module.s3.provenance_bucket_id
}

output "s3_provenance_bucket_arn" {
  description = "ARN of the provenance records bucket"
  value       = module.s3.provenance_bucket_arn
}

output "s3_raw_data_bucket_id" {
  description = "ID of the raw data bucket"
  value       = module.s3.raw_data_bucket_id
}

output "s3_raw_data_bucket_arn" {
  description = "ARN of the raw data bucket"
  value       = module.s3.raw_data_bucket_arn
}

output "s3_reports_bucket_id" {
  description = "ID of the reports bucket"
  value       = module.s3.reports_bucket_id
}

output "s3_reports_bucket_arn" {
  description = "ARN of the reports bucket"
  value       = module.s3.reports_bucket_arn
}

# ============================================================================
# IAM Outputs
# ============================================================================

output "iam_eks_cluster_role_arn" {
  description = "ARN of EKS cluster IAM role"
  value       = module.iam.eks_cluster_role_arn
}

output "iam_eks_node_role_arn" {
  description = "ARN of EKS node IAM role"
  value       = module.iam.eks_node_role_arn
}

output "iam_s3_access_role_arn" {
  description = "ARN of S3 access IAM role"
  value       = module.iam.s3_access_role_arn
}

output "iam_rds_access_role_arn" {
  description = "ARN of RDS access IAM role"
  value       = module.iam.rds_access_role_arn
}

# ============================================================================
# Monitoring Outputs
# ============================================================================

output "cloudwatch_log_group_names" {
  description = "CloudWatch log group names"
  value       = module.monitoring.log_group_names
}

output "cloudwatch_alarm_arns" {
  description = "CloudWatch alarm ARNs"
  value       = module.monitoring.alarm_arns
}

output "sns_topic_arns" {
  description = "SNS topic ARNs for notifications"
  value       = module.monitoring.sns_topic_arns
}

# ============================================================================
# Backup Outputs
# ============================================================================

output "backup_vault_arn" {
  description = "ARN of the backup vault"
  value       = module.backup.vault_arn
}

output "backup_plan_id" {
  description = "ID of the backup plan"
  value       = module.backup.plan_id
}

# ============================================================================
# Connection Information
# ============================================================================

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.aws_region}"
}

output "rds_connection_string" {
  description = "RDS connection string (without password)"
  value       = "postgresql://${module.rds.cluster_master_username}@${module.rds.cluster_endpoint}:${module.rds.cluster_port}/${module.rds.cluster_database_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://${module.elasticache.cluster_configuration_endpoint}:${module.elasticache.cluster_port}"
}

# ============================================================================
# Summary Information
# ============================================================================

output "deployment_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    environment = var.environment
    region      = var.aws_region
    vpc_cidr    = module.vpc.vpc_cidr
    eks_version = var.eks_cluster_version
    rds_engine  = "PostgreSQL ${var.rds_engine_version}"
    redis_engine = "Redis ${var.elasticache_engine_version}"
  }
}
