# variables.tf - EKS Module Variables

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.29"
}

variable "vpc_id" {
  description = "VPC ID where EKS cluster will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for EKS cluster (public and private)"
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for node groups"
  type        = list(string)
}

variable "enable_public_access" {
  description = "Enable public access to EKS API endpoint"
  type        = bool
  default     = false
}

variable "public_access_cidrs" {
  description = "CIDR blocks allowed to access public EKS API endpoint"
  type        = list(string)
  default     = []
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Node Group Configuration
variable "general_instance_types" {
  description = "Instance types for general node group"
  type        = list(string)
  default     = ["m6i.large", "m6i.xlarge"]
}

variable "general_node_desired_size" {
  description = "Desired number of nodes in general node group"
  type        = number
  default     = 3
}

variable "general_node_min_size" {
  description = "Minimum number of nodes in general node group"
  type        = number
  default     = 2
}

variable "general_node_max_size" {
  description = "Maximum number of nodes in general node group"
  type        = number
  default     = 10
}

variable "enable_spot_nodes" {
  description = "Enable spot instance node group"
  type        = bool
  default     = true
}

variable "spot_instance_types" {
  description = "Instance types for spot node group"
  type        = list(string)
  default     = ["m6i.large", "m5.large", "m5a.large"]
}

variable "spot_node_desired_size" {
  description = "Desired number of nodes in spot node group"
  type        = number
  default     = 2
}

variable "spot_node_min_size" {
  description = "Minimum number of nodes in spot node group"
  type        = number
  default     = 0
}

variable "spot_node_max_size" {
  description = "Maximum number of nodes in spot node group"
  type        = number
  default     = 10
}

variable "node_disk_size" {
  description = "Disk size in GB for worker nodes"
  type        = number
  default     = 100
}

# Addon Versions
variable "vpc_cni_version" {
  description = "VPC CNI addon version"
  type        = string
  default     = "v1.16.0-eksbuild.1"
}

variable "coredns_version" {
  description = "CoreDNS addon version"
  type        = string
  default     = "v1.11.1-eksbuild.6"
}

variable "kube_proxy_version" {
  description = "Kube Proxy addon version"
  type        = string
  default     = "v1.29.0-eksbuild.1"
}

variable "ebs_csi_version" {
  description = "EBS CSI Driver addon version"
  type        = string
  default     = "v1.27.0-eksbuild.1"
}

# S3 Bucket for Audit
variable "audit_bucket_name" {
  description = "S3 bucket name for audit cold storage"
  type        = string
}

# Tags
variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
