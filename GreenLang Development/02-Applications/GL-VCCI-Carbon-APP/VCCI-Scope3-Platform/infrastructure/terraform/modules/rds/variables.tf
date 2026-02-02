# RDS Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "cluster_identifier" {
  description = "Identifier for the RDS cluster"
  type        = string
}

variable "engine_version" {
  description = "PostgreSQL engine version"
  type        = string
}

variable "instance_class" {
  description = "Instance class for RDS"
  type        = string
}

variable "allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
}

variable "max_allocated_storage" {
  description = "Maximum storage for autoscaling in GB"
  type        = number
}

variable "storage_type" {
  description = "Storage type"
  type        = string
}

variable "iops" {
  description = "Provisioned IOPS"
  type        = number
  default     = null
}

variable "storage_throughput" {
  description = "Storage throughput in MB/s"
  type        = number
  default     = null
}

variable "multi_az" {
  description = "Enable multi-AZ"
  type        = bool
}

variable "database_name" {
  description = "Name of the default database"
  type        = string
}

variable "master_username" {
  description = "Master username"
  type        = string
}

variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
}

variable "backup_window" {
  description = "Backup window"
  type        = string
}

variable "maintenance_window" {
  description = "Maintenance window"
  type        = string
}

variable "deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
}

variable "performance_insights_enabled" {
  description = "Enable Performance Insights"
  type        = bool
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention period"
  type        = number
}

variable "read_replica_count" {
  description = "Number of read replicas"
  type        = number
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for DB subnet group"
  type        = list(string)
}

variable "allowed_security_groups" {
  description = "Security groups allowed to access RDS"
  type        = list(string)
}

variable "kms_key_arn" {
  description = "KMS key ARN for encryption"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
