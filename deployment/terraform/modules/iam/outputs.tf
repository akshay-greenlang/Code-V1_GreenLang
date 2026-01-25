# GreenLang IAM Module - Outputs

# Application Service Account
output "app_service_account_role_arn" {
  description = "ARN of the application service account IAM role"
  value       = aws_iam_role.app_service_account.arn
}

output "app_service_account_role_name" {
  description = "Name of the application service account IAM role"
  value       = aws_iam_role.app_service_account.name
}

# Agent Service Account
output "agent_service_account_role_arn" {
  description = "ARN of the agent service account IAM role"
  value       = var.create_agent_role ? aws_iam_role.agent_service_account[0].arn : null
}

output "agent_service_account_role_name" {
  description = "Name of the agent service account IAM role"
  value       = var.create_agent_role ? aws_iam_role.agent_service_account[0].name : null
}

# CI/CD Deployment Role
output "cicd_deployment_role_arn" {
  description = "ARN of the CI/CD deployment IAM role"
  value       = var.create_cicd_role ? aws_iam_role.cicd_deployment[0].arn : null
}

output "cicd_deployment_role_name" {
  description = "Name of the CI/CD deployment IAM role"
  value       = var.create_cicd_role ? aws_iam_role.cicd_deployment[0].name : null
}

# External Secrets Operator
output "external_secrets_role_arn" {
  description = "ARN of the External Secrets Operator IAM role"
  value       = var.create_external_secrets_role ? aws_iam_role.external_secrets[0].arn : null
}

output "external_secrets_role_name" {
  description = "Name of the External Secrets Operator IAM role"
  value       = var.create_external_secrets_role ? aws_iam_role.external_secrets[0].name : null
}

# Cross-Account Role
output "cross_account_role_arn" {
  description = "ARN of the cross-account access IAM role"
  value       = var.create_cross_account_role && length(var.trusted_account_ids) > 0 ? aws_iam_role.cross_account[0].arn : null
}

output "cross_account_role_name" {
  description = "Name of the cross-account access IAM role"
  value       = var.create_cross_account_role && length(var.trusted_account_ids) > 0 ? aws_iam_role.cross_account[0].name : null
}

# Monitoring Role
output "monitoring_role_arn" {
  description = "ARN of the monitoring IAM role"
  value       = var.create_monitoring_role ? aws_iam_role.monitoring[0].arn : null
}

output "monitoring_role_name" {
  description = "Name of the monitoring IAM role"
  value       = var.create_monitoring_role ? aws_iam_role.monitoring[0].name : null
}

# GitHub OIDC Provider
output "github_oidc_provider_arn" {
  description = "ARN of the GitHub Actions OIDC provider"
  value       = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : null
}

# Service Account Annotations for Kubernetes
output "app_service_account_annotation" {
  description = "Kubernetes service account annotation for the app role"
  value       = "eks.amazonaws.com/role-arn: ${aws_iam_role.app_service_account.arn}"
}

output "agent_service_account_annotation" {
  description = "Kubernetes service account annotation for the agent role"
  value       = var.create_agent_role ? "eks.amazonaws.com/role-arn: ${aws_iam_role.agent_service_account[0].arn}" : null
}

output "external_secrets_service_account_annotation" {
  description = "Kubernetes service account annotation for External Secrets Operator"
  value       = var.create_external_secrets_role ? "eks.amazonaws.com/role-arn: ${aws_iam_role.external_secrets[0].arn}" : null
}

output "monitoring_service_account_annotation" {
  description = "Kubernetes service account annotation for monitoring"
  value       = var.create_monitoring_role ? "eks.amazonaws.com/role-arn: ${aws_iam_role.monitoring[0].arn}" : null
}
