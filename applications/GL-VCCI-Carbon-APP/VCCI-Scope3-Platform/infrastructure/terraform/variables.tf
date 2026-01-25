# Root Module Variables
# GL-VCCI Scope 3 Carbon Intelligence Platform

# ============================================================================
# General Configuration
# ============================================================================

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "vcci-scope3"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "aws_region" {
  description = "AWS region for primary deployment"
  type        = string
  default     = "us-west-2"
}

variable "aws_region_secondary" {
  description = "AWS region for disaster recovery and replication"
  type        = string
  default     = "eu-central-1"
}

variable "availability_zones" {
  description = "List of availability zones for multi-AZ deployment"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "VCCI-Scope3"
    ManagedBy   = "Terraform"
    Owner       = "Platform-Team"
    CostCenter  = "Engineering"
    Compliance  = "SOC2-GDPR"
  }
}

# ============================================================================
# VPC Configuration
# ============================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use a single NAT Gateway for all private subnets (cost optimization)"
  type        = bool
  default     = false
}

variable "enable_vpc_endpoints" {
  description = "Enable VPC endpoints for AWS services"
  type        = bool
  default     = true
}

# ============================================================================
# EKS Configuration
# ============================================================================

variable "eks_cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.27"
}

variable "eks_cluster_endpoint_private_access" {
  description = "Enable private API server endpoint"
  type        = bool
  default     = true
}

variable "eks_cluster_endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "eks_cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Compute Node Group Configuration
variable "compute_node_group_instance_types" {
  description = "Instance types for compute node group"
  type        = list(string)
  default     = ["t3.xlarge"]
}

variable "compute_node_group_desired_size" {
  description = "Desired number of compute nodes"
  type        = number
  default     = 3
}

variable "compute_node_group_min_size" {
  description = "Minimum number of compute nodes"
  type        = number
  default     = 3
}

variable "compute_node_group_max_size" {
  description = "Maximum number of compute nodes"
  type        = number
  default     = 20
}

# Memory Node Group Configuration
variable "memory_node_group_instance_types" {
  description = "Instance types for memory-optimized node group"
  type        = list(string)
  default     = ["r6g.2xlarge"]
}

variable "memory_node_group_desired_size" {
  description = "Desired number of memory nodes"
  type        = number
  default     = 2
}

variable "memory_node_group_min_size" {
  description = "Minimum number of memory nodes"
  type        = number
  default     = 2
}

variable "memory_node_group_max_size" {
  description = "Maximum number of memory nodes"
  type        = number
  default     = 10
}

# GPU Node Group Configuration
variable "gpu_node_group_instance_types" {
  description = "Instance types for GPU node group"
  type        = list(string)
  default     = ["g4dn.xlarge"]
}

variable "gpu_node_group_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_node_group_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_node_group_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 5
}

# ============================================================================
# RDS Configuration
# ============================================================================

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.3"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.2xlarge"
}

variable "rds_allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 500
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage for autoscaling in GB"
  type        = number
  default     = 1000
}

variable "rds_storage_type" {
  description = "Storage type (gp3, gp2, io1)"
  type        = string
  default     = "gp3"
}

variable "rds_iops" {
  description = "Provisioned IOPS for gp3 or io1 storage"
  type        = number
  default     = 10000
}

variable "rds_storage_throughput" {
  description = "Storage throughput in MB/s for gp3 storage"
  type        = number
  default     = 250
}

variable "rds_multi_az" {
  description = "Enable multi-AZ deployment for high availability"
  type        = bool
  default     = true
}

variable "rds_backup_retention_period" {
  description = "Number of days to retain automated backups"
  type        = number
  default     = 7
}

variable "rds_backup_window" {
  description = "Daily backup window (UTC)"
  type        = string
  default     = "03:00-04:00"
}

variable "rds_maintenance_window" {
  description = "Weekly maintenance window (UTC)"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "rds_deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = true
}

variable "rds_performance_insights_enabled" {
  description = "Enable Performance Insights"
  type        = bool
  default     = true
}

variable "rds_performance_insights_retention_period" {
  description = "Performance Insights retention period in days"
  type        = number
  default     = 7
}

variable "rds_read_replica_count" {
  description = "Number of read replicas"
  type        = number
  default     = 2

  validation {
    condition     = var.rds_read_replica_count >= 0 && var.rds_read_replica_count <= 5
    error_message = "Read replica count must be between 0 and 5."
  }
}

variable "rds_database_name" {
  description = "Name of the default database"
  type        = string
  default     = "vcci_scope3"
}

variable "rds_master_username" {
  description = "Master username for RDS"
  type        = string
  default     = "postgres"
}

# ============================================================================
# ElastiCache Configuration
# ============================================================================

variable "elasticache_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "elasticache_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "elasticache_num_node_groups" {
  description = "Number of shards (node groups)"
  type        = number
  default     = 3
}

variable "elasticache_replicas_per_node_group" {
  description = "Number of replicas per shard"
  type        = number
  default     = 2
}

variable "elasticache_automatic_failover_enabled" {
  description = "Enable automatic failover"
  type        = bool
  default     = true
}

variable "elasticache_multi_az_enabled" {
  description = "Enable multi-AZ deployment"
  type        = bool
  default     = true
}

variable "elasticache_at_rest_encryption_enabled" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "elasticache_transit_encryption_enabled" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

variable "elasticache_snapshot_retention_limit" {
  description = "Number of days to retain snapshots"
  type        = number
  default     = 5
}

variable "elasticache_snapshot_window" {
  description = "Daily snapshot window (UTC)"
  type        = string
  default     = "05:00-06:00"
}

variable "elasticache_maintenance_window" {
  description = "Weekly maintenance window (UTC)"
  type        = string
  default     = "sun:06:00-sun:07:00"
}

# ============================================================================
# S3 Configuration
# ============================================================================

variable "s3_enable_versioning" {
  description = "Enable versioning for S3 buckets"
  type        = bool
  default     = true
}

variable "s3_enable_replication" {
  description = "Enable cross-region replication"
  type        = bool
  default     = true
}

variable "s3_lifecycle_glacier_transition_days" {
  description = "Days before transitioning to Glacier"
  type        = number
  default     = 365
}

variable "s3_lifecycle_ia_transition_days" {
  description = "Days before transitioning to Infrequent Access"
  type        = number
  default     = 90
}

variable "s3_lifecycle_expiration_days" {
  description = "Days before object expiration (0 to disable)"
  type        = number
  default     = 0
}

# ============================================================================
# Monitoring Configuration
# ============================================================================

variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs"
  type        = bool
  default     = true
}

variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "alarm_email_endpoints" {
  description = "Email addresses for alarm notifications"
  type        = list(string)
  default     = []
}

variable "alarm_slack_webhook_url" {
  description = "Slack webhook URL for alarm notifications"
  type        = string
  default     = ""
  sensitive   = true
}

# ============================================================================
# Backup Configuration
# ============================================================================

variable "enable_aws_backup" {
  description = "Enable AWS Backup service"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "backup_schedule" {
  description = "Backup schedule in cron format"
  type        = string
  default     = "cron(0 3 * * ? *)" # Daily at 3 AM UTC
}

# ============================================================================
# Security Configuration
# ============================================================================

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access resources"
  type        = list(string)
  default     = []
}

variable "enable_encryption" {
  description = "Enable encryption for all supported resources"
  type        = bool
  default     = true
}

variable "kms_key_deletion_window" {
  description = "KMS key deletion window in days"
  type        = number
  default     = 30
}
