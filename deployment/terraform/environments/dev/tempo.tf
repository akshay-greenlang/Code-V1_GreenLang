# =============================================================================
# GreenLang Development Environment - Tempo Storage
# GreenLang Climate OS | OBS-003
# =============================================================================
#
# Tempo trace storage configuration for the development environment.
# Optimized for cost savings with shorter retention and earlier IA transition.
#
# Key characteristics:
# - 30-day trace retention (sufficient for debugging in dev)
# - IA transition after 14 days
# - No access logging (simplified for dev)
#
# Usage:
#   terraform plan -var-file="terraform.tfvars"
#   terraform apply -var-file="terraform.tfvars"
#
# =============================================================================

# -----------------------------------------------------------------------------
# Tempo Storage Module
# -----------------------------------------------------------------------------

module "tempo_storage" {
  source = "../../modules/tempo-storage"

  # Environment configuration
  environment   = "dev"
  aws_region    = var.aws_region
  eks_cluster_name = local.cluster_name

  # IRSA configuration
  eks_oidc_provider_arn = module.eks.oidc_provider_arn
  eks_oidc_provider_url = module.eks.oidc_provider_url

  # Shorter retention for dev - 30 days total, IA at 14 days
  retention_days     = 30
  ia_transition_days = 14

  # No access logging in dev
  log_bucket_name = ""

  # Tags
  tags = merge(local.common_tags, {
    Component  = "Tempo"
    Stack      = "observability"
    CostCenter = "development"
  })
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "tempo_bucket_name" {
  description = "Tempo trace storage S3 bucket name"
  value       = module.tempo_storage.bucket_name
}

output "tempo_irsa_role_arn" {
  description = "IAM role ARN for Tempo service account (IRSA)"
  value       = module.tempo_storage.irsa_role_arn
}

output "tempo_kms_key_arn" {
  description = "KMS key ARN for Tempo S3 encryption"
  value       = module.tempo_storage.kms_key_arn
}
