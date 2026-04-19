variable "project_name" { type = string }
variable "environment" { type = string }
variable "enable_aws_backup" { type = bool }
variable "backup_retention_days" { type = number }
variable "backup_schedule" { type = string }
variable "kms_key_arn" { type = string }
variable "rds_cluster_arn" { type = string }
variable "tags" { type = map(string); default = {} }
