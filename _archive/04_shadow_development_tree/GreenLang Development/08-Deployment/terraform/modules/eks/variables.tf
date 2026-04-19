# GreenLang EKS Module - Variables

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the cluster"
  type        = list(string)
}

variable "enable_public_access" {
  description = "Enable public API access"
  type        = bool
  default     = true
}

variable "public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enabled_cluster_log_types" {
  description = "List of control plane log types to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "kms_key_arn" {
  description = "KMS key ARN for secrets encryption (creates new if null)"
  type        = string
  default     = null
}

# System Node Group
variable "system_node_instance_types" {
  description = "Instance types for system node group"
  type        = list(string)
  default     = ["m6i.xlarge"]
}

variable "system_node_disk_size" {
  description = "Disk size for system nodes"
  type        = number
  default     = 100
}

variable "system_node_desired_size" {
  description = "Desired number of system nodes"
  type        = number
  default     = 2
}

variable "system_node_min_size" {
  description = "Minimum number of system nodes"
  type        = number
  default     = 2
}

variable "system_node_max_size" {
  description = "Maximum number of system nodes"
  type        = number
  default     = 4
}

# API Node Group
variable "create_api_node_group" {
  description = "Create dedicated API node group"
  type        = bool
  default     = true
}

variable "api_node_instance_types" {
  description = "Instance types for API node group"
  type        = list(string)
  default     = ["c6i.2xlarge"]
}

variable "api_node_disk_size" {
  description = "Disk size for API nodes"
  type        = number
  default     = 100
}

variable "api_node_desired_size" {
  description = "Desired number of API nodes"
  type        = number
  default     = 3
}

variable "api_node_min_size" {
  description = "Minimum number of API nodes"
  type        = number
  default     = 2
}

variable "api_node_max_size" {
  description = "Maximum number of API nodes"
  type        = number
  default     = 8
}

# Agent Runtime Node Group
variable "create_agent_node_group" {
  description = "Create dedicated agent runtime node group"
  type        = bool
  default     = true
}

variable "agent_node_instance_types" {
  description = "Instance types for agent runtime node group"
  type        = list(string)
  default     = ["c6i.xlarge", "c6a.xlarge", "c5.xlarge"]
}

variable "agent_node_capacity_type" {
  description = "Capacity type for agent nodes (ON_DEMAND or SPOT)"
  type        = string
  default     = "SPOT"
}

variable "agent_node_disk_size" {
  description = "Disk size for agent nodes"
  type        = number
  default     = 100
}

variable "agent_node_desired_size" {
  description = "Desired number of agent nodes"
  type        = number
  default     = 5
}

variable "agent_node_min_size" {
  description = "Minimum number of agent nodes"
  type        = number
  default     = 3
}

variable "agent_node_max_size" {
  description = "Maximum number of agent nodes"
  type        = number
  default     = 20
}

# Features
variable "enable_cluster_autoscaler" {
  description = "Create IAM role for cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_load_balancer_controller" {
  description = "Create IAM role for AWS Load Balancer Controller"
  type        = bool
  default     = true
}

variable "enable_ebs_csi_driver" {
  description = "Enable EBS CSI driver add-on"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
