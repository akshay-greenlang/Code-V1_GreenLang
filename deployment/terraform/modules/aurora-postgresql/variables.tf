################################################################################
# Aurora PostgreSQL Module - Variables
################################################################################

################################################################################
# General Configuration
################################################################################

variable "project_name" {
  description = "Name of the project, used for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

################################################################################
# Database Configuration
################################################################################

variable "database_name" {
  description = "Name of the default database to create"
  type        = string
  default     = "greenlang"
}

variable "master_username" {
  description = "Master username for the database (default: postgres_admin)"
  type        = string
  default     = "postgres_admin"
  sensitive   = true
}

variable "engine_version" {
  description = "Aurora PostgreSQL engine version (must be 15.x compatible)"
  type        = string
  default     = "15.4"

  validation {
    condition     = can(regex("^15\\.", var.engine_version))
    error_message = "Engine version must be PostgreSQL 15.x compatible."
  }
}

################################################################################
# Instance Configuration
################################################################################

variable "instance_class" {
  description = "Instance class for the Aurora instances (e.g., db.r6g.large, db.serverless)"
  type        = string
  default     = "db.r6g.large"
}

variable "replica_instance_class" {
  description = "Instance class for read replicas (defaults to instance_class if empty)"
  type        = string
  default     = ""
}

variable "replica_count" {
  description = "Number of read replicas to create"
  type        = number
  default     = 2

  validation {
    condition     = var.replica_count >= 0 && var.replica_count <= 15
    error_message = "Replica count must be between 0 and 15."
  }
}

variable "writer_availability_zone" {
  description = "Preferred availability zone for the writer instance (leave empty for auto-selection)"
  type        = string
  default     = ""
}

variable "availability_zone_count" {
  description = "Number of availability zones to use (minimum 2 for Multi-AZ)"
  type        = number
  default     = 3

  validation {
    condition     = var.availability_zone_count >= 2
    error_message = "At least 2 availability zones are required for Multi-AZ deployment."
  }
}

################################################################################
# Serverless v2 Configuration
################################################################################

variable "serverless_min_capacity" {
  description = "Minimum ACU capacity for serverless instances"
  type        = number
  default     = 0.5

  validation {
    condition     = var.serverless_min_capacity >= 0.5
    error_message = "Serverless minimum capacity must be at least 0.5 ACU."
  }
}

variable "serverless_max_capacity" {
  description = "Maximum ACU capacity for serverless instances"
  type        = number
  default     = 16

  validation {
    condition     = var.serverless_max_capacity <= 128
    error_message = "Serverless maximum capacity cannot exceed 128 ACU."
  }
}

################################################################################
# Storage Configuration
################################################################################

variable "storage_type" {
  description = "Storage type for Aurora cluster (aurora, aurora-iopt1)"
  type        = string
  default     = "aurora"

  validation {
    condition     = contains(["aurora", "aurora-iopt1"], var.storage_type)
    error_message = "Storage type must be 'aurora' or 'aurora-iopt1'."
  }
}

variable "allocated_storage" {
  description = "Allocated storage in GB (only for aurora-iopt1 storage type)"
  type        = number
  default     = 100
}

variable "iops" {
  description = "IOPS for aurora-iopt1 storage type"
  type        = number
  default     = 3000
}

################################################################################
# Network Configuration
################################################################################

variable "vpc_id" {
  description = "VPC ID where the Aurora cluster will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the DB subnet group"
  type        = list(string)

  validation {
    condition     = length(var.subnet_ids) >= 2
    error_message = "At least 2 subnet IDs are required for Multi-AZ deployment."
  }
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the database"
  type        = list(string)
  default     = []
}

variable "allowed_security_groups" {
  description = "List of security group IDs allowed to access the database"
  type        = list(string)
  default     = []
}

variable "publicly_accessible" {
  description = "Whether the database should be publicly accessible"
  type        = bool
  default     = false
}

################################################################################
# Encryption Configuration
################################################################################

variable "create_kms_key" {
  description = "Whether to create a new KMS key for encryption"
  type        = bool
  default     = true
}

variable "kms_key_arn" {
  description = "ARN of existing KMS key for encryption (required if create_kms_key is false)"
  type        = string
  default     = ""
}

variable "kms_key_deletion_window" {
  description = "KMS key deletion window in days"
  type        = number
  default     = 30

  validation {
    condition     = var.kms_key_deletion_window >= 7 && var.kms_key_deletion_window <= 30
    error_message = "KMS key deletion window must be between 7 and 30 days."
  }
}

################################################################################
# Backup Configuration
################################################################################

variable "backup_retention_period" {
  description = "Number of days to retain automated backups"
  type        = number
  default     = 35

  validation {
    condition     = var.backup_retention_period >= 1 && var.backup_retention_period <= 35
    error_message = "Backup retention period must be between 1 and 35 days."
  }
}

variable "preferred_backup_window" {
  description = "Preferred backup window (UTC)"
  type        = string
  default     = "03:00-04:00"
}

variable "preferred_maintenance_window" {
  description = "Preferred maintenance window (UTC)"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "skip_final_snapshot" {
  description = "Whether to skip final snapshot when deleting the cluster"
  type        = bool
  default     = false
}

################################################################################
# IAM Configuration
################################################################################

variable "iam_database_authentication_enabled" {
  description = "Whether to enable IAM database authentication"
  type        = bool
  default     = true
}

variable "enable_s3_export" {
  description = "Whether to enable S3 export functionality"
  type        = bool
  default     = true
}

variable "s3_export_bucket_arns" {
  description = "List of S3 bucket ARNs for export (required if enable_s3_export is true)"
  type        = list(string)
  default     = []
}

################################################################################
# Monitoring Configuration
################################################################################

variable "monitoring_interval" {
  description = "Enhanced monitoring interval in seconds (0 to disable, 1/5/10/15/30/60)"
  type        = number
  default     = 60

  validation {
    condition     = contains([0, 1, 5, 10, 15, 30, 60], var.monitoring_interval)
    error_message = "Monitoring interval must be 0, 1, 5, 10, 15, 30, or 60 seconds."
  }
}

variable "performance_insights_enabled" {
  description = "Whether to enable Performance Insights"
  type        = bool
  default     = true
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention period in days (7 for free tier, or 731)"
  type        = number
  default     = 7

  validation {
    condition     = contains([7, 731], var.performance_insights_retention_period)
    error_message = "Performance Insights retention must be 7 (free tier) or 731 days."
  }
}

variable "performance_insights_kms_key_id" {
  description = "KMS key ID for Performance Insights (optional, uses Aurora KMS key if not specified)"
  type        = string
  default     = ""
}

variable "enabled_cloudwatch_logs_exports" {
  description = "List of log types to export to CloudWatch"
  type        = list(string)
  default     = ["postgresql"]

  validation {
    condition     = alltrue([for log in var.enabled_cloudwatch_logs_exports : contains(["postgresql", "upgrade"], log)])
    error_message = "Valid log types are 'postgresql' and 'upgrade'."
  }
}

################################################################################
# CloudWatch Alarms Configuration
################################################################################

variable "create_cloudwatch_alarms" {
  description = "Whether to create CloudWatch alarms"
  type        = bool
  default     = true
}

variable "alarm_cpu_threshold" {
  description = "CPU utilization threshold for alarm (percentage)"
  type        = number
  default     = 80
}

variable "alarm_connections_threshold" {
  description = "Database connections threshold for alarm"
  type        = number
  default     = 500
}

variable "alarm_replication_lag_threshold" {
  description = "Replication lag threshold for alarm (milliseconds)"
  type        = number
  default     = 100
}

variable "alarm_freeable_memory_threshold" {
  description = "Freeable memory threshold for alarm (bytes)"
  type        = number
  default     = 256000000  # 256 MB
}

variable "alarm_free_storage_threshold" {
  description = "Free storage space threshold for alarm (bytes)"
  type        = number
  default     = 10737418240  # 10 GB
}

variable "alarm_read_iops_threshold" {
  description = "Read IOPS threshold for alarm"
  type        = number
  default     = 10000
}

variable "alarm_write_iops_threshold" {
  description = "Write IOPS threshold for alarm"
  type        = number
  default     = 10000
}

variable "alarm_actions" {
  description = "List of ARNs to notify when alarm triggers"
  type        = list(string)
  default     = []
}

variable "ok_actions" {
  description = "List of ARNs to notify when alarm returns to OK state"
  type        = list(string)
  default     = []
}

################################################################################
# Parameter Group Configuration
################################################################################

variable "cluster_parameters" {
  description = "Additional cluster parameter group parameters"
  type = list(object({
    name         = string
    value        = string
    apply_method = optional(string, "pending-reboot")
  }))
  default = []
}

variable "instance_parameters" {
  description = "Additional instance parameter group parameters"
  type = list(object({
    name         = string
    value        = string
    apply_method = optional(string, "pending-reboot")
  }))
  default = []
}

################################################################################
# TimescaleDB Configuration
################################################################################

variable "timescaledb_max_background_workers" {
  description = "Maximum background workers for TimescaleDB"
  type        = number
  default     = 8
}

variable "timescaledb_telemetry_level" {
  description = "TimescaleDB telemetry level (off, basic)"
  type        = string
  default     = "off"

  validation {
    condition     = contains(["off", "basic"], var.timescaledb_telemetry_level)
    error_message = "TimescaleDB telemetry level must be 'off' or 'basic'."
  }
}

################################################################################
# Connection Configuration
################################################################################

variable "max_connections" {
  description = "Maximum number of database connections"
  type        = number
  default     = 1000
}

variable "idle_in_transaction_session_timeout" {
  description = "Idle in transaction session timeout in milliseconds"
  type        = number
  default     = 300000  # 5 minutes
}

variable "statement_timeout" {
  description = "Statement timeout in milliseconds (0 for no timeout)"
  type        = number
  default     = 0
}

################################################################################
# Secrets Manager Configuration
################################################################################

variable "create_application_credentials" {
  description = "Whether to create separate application credentials in Secrets Manager"
  type        = bool
  default     = true
}

variable "application_username" {
  description = "Username for application database user"
  type        = string
  default     = "app_user"
}

variable "secret_rotation_enabled" {
  description = "Whether to enable automatic secret rotation"
  type        = bool
  default     = false
}

variable "secret_rotation_days" {
  description = "Number of days between automatic secret rotations"
  type        = number
  default     = 30
}

################################################################################
# Custom Endpoint Configuration
################################################################################

variable "create_reader_endpoint" {
  description = "Whether to create a custom reader endpoint"
  type        = bool
  default     = false
}

variable "reader_endpoint_instance_ids" {
  description = "List of instance identifiers for the custom reader endpoint"
  type        = list(string)
  default     = []
}

################################################################################
# Maintenance Configuration
################################################################################

variable "auto_minor_version_upgrade" {
  description = "Whether to enable automatic minor version upgrades"
  type        = bool
  default     = true
}

variable "apply_immediately" {
  description = "Whether to apply changes immediately or during maintenance window"
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Whether to enable deletion protection"
  type        = bool
  default     = true
}
