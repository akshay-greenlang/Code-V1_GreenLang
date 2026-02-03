#######################################
# GreenLang Database Secrets Variables
#######################################

#######################################
# General Configuration
#######################################

variable "name_prefix" {
  description = "Prefix for naming resources"
  type        = string
  default     = "greenlang"
}

variable "secret_name_prefix" {
  description = "Prefix for secret names in Secrets Manager"
  type        = string
  default     = "greenlang"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}

#######################################
# KMS Configuration
#######################################

variable "create_kms_key" {
  description = "Whether to create a new KMS key for secrets encryption"
  type        = bool
  default     = true
}

variable "kms_key_arn" {
  description = "ARN of existing KMS key to use (if create_kms_key is false)"
  type        = string
  default     = ""
}

variable "kms_deletion_window_days" {
  description = "Number of days before KMS key is deleted"
  type        = number
  default     = 30
}

#######################################
# Database Configuration
#######################################

variable "database_host" {
  description = "PostgreSQL database host"
  type        = string
}

variable "database_port" {
  description = "PostgreSQL database port"
  type        = number
  default     = 5432
}

variable "database_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "greenlang"
}

variable "cluster_identifier" {
  description = "Database cluster identifier"
  type        = string
  default     = "greenlang-postgresql"
}

#######################################
# Master Credentials
#######################################

variable "master_username" {
  description = "Master database username"
  type        = string
  default     = "postgres"
}

#######################################
# Application Credentials
#######################################

variable "app_username" {
  description = "Application database username"
  type        = string
  default     = "greenlang_app"
}

#######################################
# Replication Credentials
#######################################

variable "replication_username" {
  description = "Replication user username"
  type        = string
  default     = "replicator"
}

variable "replication_slot_name" {
  description = "PostgreSQL replication slot name"
  type        = string
  default     = "greenlang_repl_slot"
}

#######################################
# PgBouncer Configuration
#######################################

variable "pgbouncer_auth_user" {
  description = "PgBouncer auth query user"
  type        = string
  default     = "pgbouncer"
}

variable "pgbouncer_admin_users" {
  description = "Comma-separated list of PgBouncer admin users"
  type        = string
  default     = "postgres"
}

variable "pgbouncer_stats_users" {
  description = "Comma-separated list of PgBouncer stats users"
  type        = string
  default     = "pgbouncer_stats"
}

#######################################
# pgBackRest Configuration
#######################################

variable "pgbackrest_s3_access_key" {
  description = "S3 access key for pgBackRest"
  type        = string
  sensitive   = true
  default     = ""
}

variable "pgbackrest_s3_secret_key" {
  description = "S3 secret key for pgBackRest"
  type        = string
  sensitive   = true
  default     = ""
}

variable "pgbackrest_s3_bucket" {
  description = "S3 bucket for pgBackRest backups"
  type        = string
  default     = ""
}

variable "pgbackrest_s3_region" {
  description = "S3 region for pgBackRest"
  type        = string
  default     = "us-east-1"
}

variable "pgbackrest_repo_path" {
  description = "Repository path in S3 bucket"
  type        = string
  default     = "/backup"
}

variable "pgbackrest_retention_full" {
  description = "Number of full backups to retain"
  type        = number
  default     = 4
}

variable "pgbackrest_retention_diff" {
  description = "Number of differential backups to retain"
  type        = number
  default     = 14
}

#######################################
# Rotation Configuration
#######################################

variable "enable_rotation" {
  description = "Enable automatic secret rotation"
  type        = bool
  default     = true
}

variable "rotation_days" {
  description = "Number of days between automatic rotations"
  type        = number
  default     = 90
}

variable "rotation_schedule" {
  description = "Cron expression for rotation schedule (optional, overrides rotation_days)"
  type        = string
  default     = null
}

variable "recovery_window_days" {
  description = "Number of days before a deleted secret is permanently removed"
  type        = number
  default     = 30
}

variable "excluded_password_characters" {
  description = "Characters to exclude from generated passwords"
  type        = string
  default     = "/@\"'\\"
}

#######################################
# Lambda Configuration
#######################################

variable "lambda_subnet_ids" {
  description = "Subnet IDs for Lambda VPC configuration"
  type        = list(string)
  default     = []
}

variable "lambda_security_group_ids" {
  description = "Security group IDs for Lambda VPC configuration"
  type        = list(string)
  default     = []
}

#######################################
# Notifications
#######################################

variable "create_sns_topic" {
  description = "Create SNS topic for rotation notifications"
  type        = bool
  default     = true
}

variable "notification_sns_topic_arn" {
  description = "SNS topic ARN for notifications (if create_sns_topic is false)"
  type        = string
  default     = ""
}

variable "alarm_actions" {
  description = "List of ARNs for alarm actions"
  type        = list(string)
  default     = []
}

variable "ok_actions" {
  description = "List of ARNs for OK actions"
  type        = list(string)
  default     = []
}

#######################################
# Logging
#######################################

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

#######################################
# EKS Integration
#######################################

variable "eks_cluster_oidc_issuer_url" {
  description = "OIDC issuer URL for EKS cluster (for IRSA)"
  type        = string
  default     = ""
}

variable "eks_namespace" {
  description = "Kubernetes namespace for the application"
  type        = string
  default     = "greenlang"
}

variable "eks_service_account_name" {
  description = "Kubernetes service account name for secret access"
  type        = string
  default     = "greenlang-db-secrets"
}
