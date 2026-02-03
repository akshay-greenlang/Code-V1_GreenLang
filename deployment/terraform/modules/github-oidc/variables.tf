# GitHub OIDC Module - Variables
# INFRA-001 Component

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "role_name" {
  description = "Name of the IAM role for GitHub Actions"
  type        = string
}

variable "create_oidc_provider" {
  description = "Whether to create the OIDC provider (set to false if it already exists)"
  type        = bool
  default     = true
}

variable "oidc_provider_arn" {
  description = "ARN of existing OIDC provider (required if create_oidc_provider is false)"
  type        = string
  default     = null
}

variable "github_organization" {
  description = "GitHub organization or user name"
  type        = string
}

variable "github_repositories" {
  description = "List of GitHub repository names allowed to assume this role"
  type        = list(string)
}

variable "max_session_duration" {
  description = "Maximum session duration in seconds (1-12 hours)"
  type        = number
  default     = 3600
}

variable "terraform_state_bucket" {
  description = "Name of the S3 bucket for Terraform state"
  type        = string
}

variable "terraform_lock_table" {
  description = "Name of the DynamoDB table for Terraform state locking"
  type        = string
}

variable "enable_eks_management" {
  description = "Enable EKS management permissions"
  type        = bool
  default     = true
}

variable "enable_infrastructure_management" {
  description = "Enable infrastructure management permissions (VPC, RDS, ElastiCache)"
  type        = bool
  default     = true
}

variable "enable_iam_management" {
  description = "Enable IAM management permissions"
  type        = bool
  default     = true
}

variable "enable_ecr_management" {
  description = "Enable ECR management permissions"
  type        = bool
  default     = true
}

variable "allowed_regions" {
  description = "List of AWS regions where resources can be managed"
  type        = list(string)
  default     = ["us-east-1"]
}

variable "s3_bucket_prefix" {
  description = "Prefix for S3 buckets that can be managed"
  type        = string
  default     = "greenlang"
}

variable "secrets_prefix" {
  description = "Prefix for Secrets Manager secrets that can be managed"
  type        = string
  default     = "greenlang"
}

variable "iam_role_prefix" {
  description = "Prefix for IAM roles that can be managed"
  type        = string
  default     = "greenlang"
}

variable "ecr_repository_prefix" {
  description = "Prefix for ECR repositories that can be managed"
  type        = string
  default     = "greenlang"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
