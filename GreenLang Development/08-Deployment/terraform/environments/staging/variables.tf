# GreenLang Staging Environment - Variables

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.1.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "eks_cluster_version" {
  description = "EKS cluster Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "eks_public_access_cidrs" {
  description = "CIDR blocks for EKS public API access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "elb_account_id" {
  description = "AWS account ID for ELB logging (varies by region)"
  type        = string
  default     = "127311923021"
}

variable "cors_allowed_origins" {
  description = "CORS allowed origins"
  type        = list(string)
  default     = ["*"]
}

variable "create_github_oidc_provider" {
  description = "Create GitHub OIDC provider"
  type        = bool
  default     = false
}

variable "github_oidc_provider_arn" {
  description = "Existing GitHub OIDC provider ARN"
  type        = string
  default     = null
}

variable "github_org" {
  description = "GitHub organization name"
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
  default     = ""
}

variable "ecr_repository_arns" {
  description = "ECR repository ARNs for CI/CD"
  type        = list(string)
  default     = ["*"]
}

variable "alarm_sns_topic_arns" {
  description = "SNS topic ARNs for CloudWatch alarms"
  type        = list(string)
  default     = []
}

variable "sqs_queue_arns" {
  description = "SQS queue ARNs for application access"
  type        = list(string)
  default     = []
}

variable "sns_topic_arns" {
  description = "SNS topic ARNs for application access"
  type        = list(string)
  default     = []
}

variable "cicd_secrets_arns" {
  description = "Secrets Manager ARNs for CI/CD"
  type        = list(string)
  default     = []
}
