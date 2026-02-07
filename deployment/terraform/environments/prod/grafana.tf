# =============================================================================
# GreenLang Grafana - Production Environment
# GreenLang Climate OS | OBS-002
# =============================================================================

module "grafana" {
  source = "../../modules/grafana"

  environment = "prod"

  # Network
  vpc_id                     = module.vpc.vpc_id
  private_subnet_ids         = module.vpc.private_subnet_ids
  eks_node_security_group_id = module.eks.node_security_group_id
  eks_oidc_provider_arn      = module.eks.oidc_provider_arn
  eks_oidc_provider_url      = module.eks.oidc_provider_url

  # RDS - Production-grade
  db_instance_class          = "db.t3.medium"
  db_allocated_storage       = 20
  db_max_allocated_storage   = 50
  db_backup_retention_period = 7
  db_multi_az                = true
  db_deletion_protection     = true

  # S3
  s3_image_retention_days = 30

  # KMS
  kms_key_arn = module.kms.key_arn

  # Grafana
  grafana_namespace = "monitoring"

  tags = {
    Environment = "prod"
    CostCenter  = "platform"
    Compliance  = "soc2"
  }
}
