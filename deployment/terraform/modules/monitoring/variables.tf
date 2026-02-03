# GreenLang Monitoring Module - Variables

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

# Log Configuration
variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30
}

# SNS Configuration
variable "create_sns_topic" {
  description = "Create SNS topic for alerts"
  type        = bool
  default     = true
}

variable "alarm_actions" {
  description = "List of ARNs to notify for alarms"
  type        = list(string)
  default     = []
}

# EKS Alarms
variable "create_eks_alarms" {
  description = "Create EKS monitoring alarms"
  type        = bool
  default     = true
}

variable "eks_cpu_threshold" {
  description = "EKS CPU utilization threshold"
  type        = number
  default     = 80
}

variable "eks_memory_threshold" {
  description = "EKS memory utilization threshold"
  type        = number
  default     = 80
}

variable "pod_restart_threshold" {
  description = "Pod restart count threshold"
  type        = number
  default     = 10
}

# RDS Configuration
variable "rds_identifier" {
  description = "RDS instance identifier for dashboard"
  type        = string
  default     = null
}

# ElastiCache Configuration
variable "elasticache_cluster_id" {
  description = "ElastiCache cluster ID for dashboard"
  type        = string
  default     = null
}

# Application Alarms
variable "create_application_alarms" {
  description = "Create application monitoring alarms"
  type        = bool
  default     = true
}

variable "application_error_threshold" {
  description = "Application error count threshold"
  type        = number
  default     = 10
}

# Tags
variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
