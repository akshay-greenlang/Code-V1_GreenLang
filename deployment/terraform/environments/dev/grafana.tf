# =============================================================================
# GreenLang Grafana - Dev Environment
# GreenLang Climate OS | OBS-002
# =============================================================================

module "grafana" {
  source = "../../modules/grafana"

  environment = "dev"

  # Network
  vpc_id                     = module.vpc.vpc_id
  private_subnet_ids         = module.vpc.private_subnet_ids
  eks_node_security_group_id = module.eks.node_security_group_id
  eks_oidc_provider_arn      = module.eks.oidc_provider_arn
  eks_oidc_provider_url      = module.eks.oidc_provider_url

  # RDS - Minimal for dev
  db_instance_class          = "db.t3.micro"
  db_allocated_storage       = 10
  db_max_allocated_storage   = 20
  db_backup_retention_period = 1
  db_multi_az                = false
  db_deletion_protection     = false

  # S3
  s3_image_retention_days = 7

  # Grafana
  grafana_namespace = "monitoring"

  tags = {
    Environment = "dev"
    CostCenter  = "engineering"
  }
}
