# =============================================================================
# GreenLang Production Environment - Tempo Storage
# GreenLang Climate OS | OBS-003
# =============================================================================
#
# Tempo trace storage configuration for the production environment.
# Full retention with access logging and compliance-grade encryption.
#
# Key characteristics:
# - 90-day trace retention (regulatory compliance window)
# - IA transition after 30 days (cost optimization for cold traces)
# - Access logging enabled to centralized log bucket
# - KMS encryption with automatic key rotation
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
  environment   = "prod"
  aws_region    = var.aws_region
  eks_cluster_name = local.cluster_name

  # IRSA configuration
  eks_oidc_provider_arn = module.eks.oidc_provider_arn
  eks_oidc_provider_url = module.eks.oidc_provider_url

  # Full retention for prod - 90 days total, IA at 30 days
  retention_days     = 90
  ia_transition_days = 30

  # Access logging to centralized log bucket
  log_bucket_name = module.s3.logs_bucket_name

  # Tags
  tags = merge(local.common_tags, {
    Component  = "Tempo"
    Stack      = "observability"
    CostCenter = "production"
    Compliance = "SOC2"
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
