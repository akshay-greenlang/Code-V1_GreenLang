# GreenLang EKS Module - Outputs

output "cluster_id" {
  description = "The ID of the EKS cluster"
  value       = aws_eks_cluster.main.id
}

output "cluster_name" {
  description = "The name of the EKS cluster"
  value       = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  description = "The endpoint for the EKS cluster API server"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data for the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "cluster_version" {
  description = "The Kubernetes version of the cluster"
  value       = aws_eks_cluster.main.version
}

output "cluster_arn" {
  description = "The ARN of the EKS cluster"
  value       = aws_eks_cluster.main.arn
}

output "cluster_security_group_id" {
  description = "Security group ID for the cluster"
  value       = aws_security_group.cluster.id
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC provider for IRSA"
  value       = aws_iam_openid_connect_provider.cluster.arn
}

output "oidc_provider_url" {
  description = "The URL of the OIDC provider"
  value       = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

output "cluster_role_arn" {
  description = "The ARN of the cluster IAM role"
  value       = aws_iam_role.cluster.arn
}

output "node_role_arn" {
  description = "The ARN of the node IAM role"
  value       = aws_iam_role.node.arn
}

output "cluster_autoscaler_role_arn" {
  description = "The ARN of the cluster autoscaler IAM role"
  value       = var.enable_cluster_autoscaler ? aws_iam_role.cluster_autoscaler[0].arn : null
}

output "load_balancer_controller_role_arn" {
  description = "The ARN of the AWS Load Balancer Controller IAM role"
  value       = var.enable_load_balancer_controller ? aws_iam_role.load_balancer_controller[0].arn : null
}

output "system_node_group_name" {
  description = "The name of the system node group"
  value       = aws_eks_node_group.system.node_group_name
}

output "api_node_group_name" {
  description = "The name of the API node group"
  value       = var.create_api_node_group ? aws_eks_node_group.api[0].node_group_name : null
}

output "agent_node_group_name" {
  description = "The name of the agent runtime node group"
  value       = var.create_agent_node_group ? aws_eks_node_group.agent_runtime[0].node_group_name : null
}

output "kms_key_arn" {
  description = "The ARN of the KMS key used for secrets encryption"
  value       = var.kms_key_arn != null ? var.kms_key_arn : aws_kms_key.eks[0].arn
}

# Kubeconfig for kubectl access
output "kubeconfig" {
  description = "kubectl config to access the cluster"
  value = templatefile("${path.module}/templates/kubeconfig.tpl", {
    cluster_name     = aws_eks_cluster.main.name
    cluster_endpoint = aws_eks_cluster.main.endpoint
    cluster_ca       = aws_eks_cluster.main.certificate_authority[0].data
    region           = data.aws_region.current.name
  })
  sensitive = true
}

data "aws_region" "current" {}
