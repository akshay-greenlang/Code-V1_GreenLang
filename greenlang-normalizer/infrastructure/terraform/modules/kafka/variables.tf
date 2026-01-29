# variables.tf - MSK Module Variables

variable "cluster_name" {
  description = "Name of the MSK cluster"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where MSK will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for MSK brokers (should be in different AZs)"
  type        = list(string)
}

variable "eks_node_security_group_id" {
  description = "Security group ID of EKS nodes for ingress"
  type        = string
}

# Kafka Configuration
variable "kafka_version" {
  description = "Kafka version"
  type        = string
  default     = "3.5.1"
}

variable "number_of_broker_nodes" {
  description = "Number of broker nodes (should be multiple of AZs)"
  type        = number
  default     = 3
}

variable "broker_instance_type" {
  description = "Instance type for Kafka brokers"
  type        = string
  default     = "kafka.m5.large"
}

variable "broker_ebs_volume_size" {
  description = "EBS volume size in GB for each broker"
  type        = number
  default     = 100
}

variable "enable_provisioned_throughput" {
  description = "Enable provisioned throughput for EBS"
  type        = bool
  default     = false
}

variable "provisioned_throughput_mibps" {
  description = "Provisioned throughput in MiB/s"
  type        = number
  default     = 250
}

# Topic Configuration
variable "default_partitions" {
  description = "Default number of partitions for auto-created topics"
  type        = number
  default     = 6
}

variable "log_retention_hours" {
  description = "Log retention in hours"
  type        = number
  default     = 168 # 7 days
}

variable "log_retention_bytes" {
  description = "Log retention in bytes per partition (-1 for unlimited)"
  type        = number
  default     = -1
}

# Encryption
variable "encryption_in_transit_client_broker" {
  description = "Encryption for client-broker communication (TLS, TLS_PLAINTEXT, PLAINTEXT)"
  type        = string
  default     = "TLS"
}

# Authentication
variable "enable_sasl_scram" {
  description = "Enable SASL/SCRAM authentication"
  type        = bool
  default     = true
}

variable "enable_sasl_iam" {
  description = "Enable SASL/IAM authentication"
  type        = bool
  default     = true
}

variable "enable_unauthenticated_access" {
  description = "Enable unauthenticated access"
  type        = bool
  default     = false
}

variable "sasl_scram_username" {
  description = "Username for SASL/SCRAM authentication"
  type        = string
  default     = "gl_normalizer"
  sensitive   = true
}

variable "sasl_scram_password" {
  description = "Password for SASL/SCRAM authentication"
  type        = string
  sensitive   = true
}

# Logging
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "enable_s3_logs" {
  description = "Enable S3 logging for broker logs"
  type        = bool
  default     = false
}

# Monitoring
variable "enhanced_monitoring" {
  description = "Enhanced monitoring level (DEFAULT, PER_BROKER, PER_TOPIC_PER_BROKER, PER_TOPIC_PER_PARTITION)"
  type        = string
  default     = "PER_BROKER"
}

variable "enable_jmx_exporter" {
  description = "Enable JMX Prometheus exporter"
  type        = bool
  default     = true
}

variable "enable_node_exporter" {
  description = "Enable Node Prometheus exporter"
  type        = bool
  default     = true
}

# Alarms
variable "alarm_actions" {
  description = "List of ARNs for alarm actions (SNS topics)"
  type        = list(string)
  default     = []
}

# Tags
variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
