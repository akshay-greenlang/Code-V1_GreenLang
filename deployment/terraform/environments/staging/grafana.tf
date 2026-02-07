# =============================================================================
# GreenLang Grafana - Staging Environment
# GreenLang Climate OS | OBS-002
# =============================================================================

module "grafana" {
  source = "../../modules/grafana"

  environment = "staging"

  # Network
  vpc_id                     = module.vpc.vpc_id
  private_subnet_ids         = module.vpc.private_subnet_ids
  eks_node_security_group_id = module.eks.node_security_group_id
  eks_oidc_provider_arn      = module.eks.oidc_provider_arn
  eks_oidc_provider_url      = module.eks.oidc_provider_url

  # RDS - Mid-tier for staging
  db_instance_class          = "db.t3.small"
  db_allocated_storage       = 15
  db_max_allocated_storage   = 30
  db_backup_retention_period = 3
  db_multi_az                = false
  db_deletion_protection     = false

  # S3
  s3_image_retention_days = 14

  # KMS
  kms_key_arn = module.kms.key_arn

  # Grafana
  grafana_namespace = "monitoring"

  tags = {
    Environment = "staging"
    CostCenter  = "engineering"
  }
}
