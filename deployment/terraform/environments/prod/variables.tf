# GreenLang Production Environment - Variables

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "dr_region" {
  description = "DR AWS region"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.2.0.0/16"
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

variable "enable_eks_public_access" {
  description = "Enable EKS public API access"
  type        = bool
  default     = false
}

variable "eks_public_access_cidrs" {
  description = "CIDR blocks for EKS public API access"
  type        = list(string)
  default     = []
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
  default     = []
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

variable "elasticache_notification_topic_arn" {
  description = "SNS topic ARN for ElastiCache notifications"
  type        = string
  default     = null
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

variable "app_secrets_arns" {
  description = "Secrets Manager ARNs for application"
  type        = list(string)
  default     = ["*"]
}

variable "agent_secrets_arns" {
  description = "Secrets Manager ARNs for agents"
  type        = list(string)
  default     = ["*"]
}

variable "cicd_secrets_arns" {
  description = "Secrets Manager ARNs for CI/CD"
  type        = list(string)
  default     = []
}

variable "external_secrets_allowed_arns" {
  description = "Secrets Manager ARNs for External Secrets Operator"
  type        = list(string)
  default     = ["*"]
}

variable "cloudfront_distribution_arn" {
  description = "CloudFront distribution ARN for S3 OAC"
  type        = string
  default     = null
}

# DR Configuration
variable "dr_data_bucket_arn" {
  description = "DR data bucket ARN for replication"
  type        = string
  default     = null
}

variable "dr_kms_key_arn" {
  description = "DR KMS key ARN for replication"
  type        = string
  default     = null
}

# Cross-Account Access
variable "enable_cross_account_access" {
  description = "Enable cross-account access role"
  type        = bool
  default     = false
}

variable "trusted_account_ids" {
  description = "Trusted AWS account IDs for cross-account access"
  type        = list(string)
  default     = []
}

variable "cross_account_resource_arns" {
  description = "Resource ARNs for cross-account access"
  type        = list(string)
  default     = []
}
