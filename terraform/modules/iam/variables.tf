# GreenLang IAM Module - Variables

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

# OIDC Provider Configuration
variable "oidc_provider_arn" {
  description = "ARN of the EKS OIDC provider"
  type        = string
}

variable "oidc_provider_url" {
  description = "URL of the EKS OIDC provider"
  type        = string
}

# Application Service Account Configuration
variable "app_namespace" {
  description = "Kubernetes namespace for application workloads"
  type        = string
  default     = "greenlang"
}

# Agent Service Account Configuration
variable "create_agent_role" {
  description = "Create IAM role for agent service account"
  type        = bool
  default     = true
}

variable "agent_namespace" {
  description = "Kubernetes namespace for agent workloads"
  type        = string
  default     = "greenlang-agents"
}

variable "agent_service_account_name" {
  description = "Name of the agent service account"
  type        = string
  default     = "agent-runtime"
}

variable "agent_s3_write_bucket_arns" {
  description = "S3 bucket ARNs that agents can write to"
  type        = list(string)
  default     = []
}

variable "agent_secrets_arns" {
  description = "Secrets Manager ARNs that agents can access"
  type        = list(string)
  default     = ["*"]
}

# S3 Access Configuration
variable "enable_s3_access" {
  description = "Enable S3 access for service accounts"
  type        = bool
  default     = true
}

variable "s3_bucket_arns" {
  description = "S3 bucket ARNs to grant access to"
  type        = list(string)
  default     = []
}

# Secrets Manager Configuration
variable "enable_secrets_manager_access" {
  description = "Enable Secrets Manager access"
  type        = bool
  default     = true
}

variable "secrets_arns" {
  description = "Secrets Manager ARNs to grant access to"
  type        = list(string)
  default     = ["*"]
}

# KMS Configuration
variable "enable_kms_access" {
  description = "Enable KMS access"
  type        = bool
  default     = true
}

variable "kms_key_arns" {
  description = "KMS key ARNs to grant access to"
  type        = list(string)
  default     = ["*"]
}

# SQS Configuration
variable "enable_sqs_access" {
  description = "Enable SQS access"
  type        = bool
  default     = false
}

variable "sqs_queue_arns" {
  description = "SQS queue ARNs to grant access to"
  type        = list(string)
  default     = []
}

# SNS Configuration
variable "enable_sns_access" {
  description = "Enable SNS access"
  type        = bool
  default     = false
}

variable "sns_topic_arns" {
  description = "SNS topic ARNs to grant access to"
  type        = list(string)
  default     = []
}

# Bedrock Configuration
variable "enable_bedrock_access" {
  description = "Enable Amazon Bedrock access for AI agents"
  type        = bool
  default     = false
}

# CI/CD Role Configuration
variable "create_cicd_role" {
  description = "Create IAM role for CI/CD deployments"
  type        = bool
  default     = true
}

variable "github_oidc_provider_arn" {
  description = "ARN of the GitHub Actions OIDC provider"
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

variable "gitlab_oidc_provider_arn" {
  description = "ARN of the GitLab OIDC provider"
  type        = string
  default     = null
}

variable "gitlab_oidc_url" {
  description = "GitLab OIDC provider URL"
  type        = string
  default     = "gitlab.com"
}

variable "gitlab_oidc_audience" {
  description = "GitLab OIDC audience"
  type        = string
  default     = "https://gitlab.com"
}

variable "gitlab_project_path" {
  description = "GitLab project path (org/repo)"
  type        = string
  default     = ""
}

variable "cicd_trusted_accounts" {
  description = "AWS account IDs trusted for CI/CD cross-account access"
  type        = list(string)
  default     = []
}

variable "ecr_repository_arns" {
  description = "ECR repository ARNs for CI/CD access"
  type        = list(string)
  default     = ["*"]
}

variable "cicd_artifact_bucket_arns" {
  description = "S3 bucket ARNs for CI/CD artifacts"
  type        = list(string)
  default     = []
}

variable "cicd_secrets_arns" {
  description = "Secrets Manager ARNs for CI/CD"
  type        = list(string)
  default     = []
}

# External Secrets Operator Configuration
variable "create_external_secrets_role" {
  description = "Create IAM role for External Secrets Operator"
  type        = bool
  default     = true
}

variable "external_secrets_namespace" {
  description = "Namespace for External Secrets Operator"
  type        = string
  default     = "external-secrets"
}

variable "external_secrets_service_account" {
  description = "Service account name for External Secrets Operator"
  type        = string
  default     = "external-secrets"
}

variable "external_secrets_allowed_arns" {
  description = "Secrets Manager ARNs allowed for External Secrets Operator"
  type        = list(string)
  default     = ["*"]
}

# Cross-Account Access Configuration
variable "create_cross_account_role" {
  description = "Create cross-account access role"
  type        = bool
  default     = false
}

variable "trusted_account_ids" {
  description = "AWS account IDs trusted for cross-account access"
  type        = list(string)
  default     = []
}

variable "require_mfa_for_cross_account" {
  description = "Require MFA for cross-account access"
  type        = bool
  default     = true
}

variable "cross_account_resource_arns" {
  description = "Resource ARNs accessible via cross-account role"
  type        = list(string)
  default     = []
}

# Monitoring Role Configuration
variable "create_monitoring_role" {
  description = "Create IAM role for monitoring workloads"
  type        = bool
  default     = true
}

variable "monitoring_namespace" {
  description = "Namespace for monitoring workloads"
  type        = string
  default     = "monitoring"
}

variable "monitoring_service_account" {
  description = "Service account name for monitoring"
  type        = string
  default     = "cloudwatch-agent"
}

# GitHub OIDC Provider
variable "create_github_oidc_provider" {
  description = "Create GitHub Actions OIDC provider"
  type        = bool
  default     = false
}

# Tags
variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
