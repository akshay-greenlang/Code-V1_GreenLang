variable "project_name" { type = string }
variable "environment" { type = string }
variable "enable_cloudwatch_logs" { type = bool }
variable "log_retention_days" { type = number }
variable "enable_cloudwatch_alarms" { type = bool }
variable "alarm_email_endpoints" { type = list(string); default = [] }
variable "alarm_slack_webhook_url" { type = string; default = ""; sensitive = true }
variable "eks_cluster_name" { type = string }
variable "rds_cluster_id" { type = string }
variable "elasticache_cluster_id" { type = string }
variable "tags" { type = map(string); default = {} }
