# GitHub OIDC Module - Outputs
# INFRA-001 Component

output "oidc_provider_arn" {
  description = "ARN of the GitHub OIDC provider"
  value       = var.create_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : var.oidc_provider_arn
}

output "oidc_provider_url" {
  description = "URL of the GitHub OIDC provider"
  value       = "https://token.actions.githubusercontent.com"
}

output "role_arn" {
  description = "ARN of the IAM role for GitHub Actions"
  value       = aws_iam_role.github_actions.arn
}

output "role_name" {
  description = "Name of the IAM role for GitHub Actions"
  value       = aws_iam_role.github_actions.name
}

output "role_unique_id" {
  description = "Unique ID of the IAM role"
  value       = aws_iam_role.github_actions.unique_id
}

output "terraform_state_policy_arn" {
  description = "ARN of the Terraform state management policy"
  value       = aws_iam_policy.terraform_state.arn
}

output "eks_management_policy_arn" {
  description = "ARN of the EKS management policy"
  value       = var.enable_eks_management ? aws_iam_policy.eks_management[0].arn : null
}

output "infrastructure_management_policy_arn" {
  description = "ARN of the infrastructure management policy"
  value       = var.enable_infrastructure_management ? aws_iam_policy.infrastructure_management[0].arn : null
}

output "iam_management_policy_arn" {
  description = "ARN of the IAM management policy"
  value       = var.enable_iam_management ? aws_iam_policy.iam_management[0].arn : null
}

output "ecr_management_policy_arn" {
  description = "ARN of the ECR management policy"
  value       = var.enable_ecr_management ? aws_iam_policy.ecr_management[0].arn : null
}

output "github_actions_role_trust_policy" {
  description = "Trust policy for the GitHub Actions role (for reference)"
  value = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.create_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : var.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = [
              for repo in var.github_repositories :
              "repo:${var.github_organization}/${repo}:*"
            ]
          }
        }
      }
    ]
  })
}
