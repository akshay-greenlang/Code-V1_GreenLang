# GreenLang ElastiCache Module - Variables

variable "cluster_id" {
  description = "Identifier for the ElastiCache cluster"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_group_name" {
  description = "Name of the ElastiCache subnet group"
  type        = string
}

# Engine Configuration
variable "engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.1"
}

variable "node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "port" {
  description = "Redis port"
  type        = number
  default     = 6379
}

# Cluster Configuration
variable "num_cache_clusters" {
  description = "Number of cache clusters (primary and replicas) for non-cluster mode"
  type        = number
  default     = 2
}

variable "cluster_mode_enabled" {
  description = "Enable cluster mode (sharding)"
  type        = bool
  default     = false
}

variable "num_node_groups" {
  description = "Number of node groups (shards) for cluster mode"
  type        = number
  default     = 1
}

variable "replicas_per_node_group" {
  description = "Number of replicas per node group for cluster mode"
  type        = number
  default     = 1
}

variable "multi_az_enabled" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = true
}

# Encryption Configuration
variable "at_rest_encryption_enabled" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "transit_encryption_enabled" {
  description = "Enable encryption in transit (TLS)"
  type        = bool
  default     = true
}

variable "kms_key_arn" {
  description = "KMS key ARN for encryption (creates new if null)"
  type        = string
  default     = null
}

variable "auth_token" {
  description = "Auth token for Redis (random if null and transit encryption enabled)"
  type        = string
  default     = null
  sensitive   = true
}

# Parameter Group Configuration
variable "maxmemory_policy" {
  description = "Eviction policy when maxmemory is reached"
  type        = string
  default     = "volatile-lru"
}

variable "enable_aof" {
  description = "Enable Append Only File persistence"
  type        = bool
  default     = false
}

variable "connection_timeout" {
  description = "Connection timeout in seconds (0 to disable)"
  type        = number
  default     = 0
}

variable "notify_keyspace_events" {
  description = "Keyspace event notifications configuration"
  type        = string
  default     = ""
}

# Maintenance Configuration
variable "maintenance_window" {
  description = "Maintenance window"
  type        = string
  default     = "sun:05:00-sun:06:00"
}

variable "snapshot_window" {
  description = "Daily snapshot window"
  type        = string
  default     = "03:00-04:00"
}

variable "snapshot_retention_limit" {
  description = "Number of days to retain snapshots"
  type        = number
  default     = 7
}

variable "skip_final_snapshot" {
  description = "Skip final snapshot on deletion"
  type        = bool
  default     = false
}

variable "auto_minor_version_upgrade" {
  description = "Enable auto minor version upgrade"
  type        = bool
  default     = true
}

variable "apply_immediately" {
  description = "Apply changes immediately or during maintenance window"
  type        = bool
  default     = false
}

# Notifications
variable "notification_topic_arn" {
  description = "ARN of SNS topic for ElastiCache events"
  type        = string
  default     = null
}

# Security Configuration
variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access ElastiCache"
  type        = list(string)
  default     = []
}

variable "allowed_security_groups" {
  description = "Security group IDs allowed to access ElastiCache"
  type        = list(string)
  default     = []
}

# CloudWatch Alarms
variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "alarm_actions" {
  description = "List of ARNs to notify for alarms"
  type        = list(string)
  default     = []
}

variable "cpu_threshold" {
  description = "CPU utilization threshold for alarm"
  type        = number
  default     = 75
}

variable "memory_threshold" {
  description = "Memory utilization threshold for alarm"
  type        = number
  default     = 75
}

variable "evictions_threshold" {
  description = "Evictions threshold for alarm"
  type        = number
  default     = 1000
}

variable "connections_threshold" {
  description = "Connections threshold for alarm"
  type        = number
  default     = 5000
}

variable "replication_lag_threshold" {
  description = "Replication lag threshold in seconds"
  type        = number
  default     = 30
}

variable "engine_cpu_threshold" {
  description = "Engine CPU utilization threshold for alarm"
  type        = number
  default     = 90
}

# Tags
variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
