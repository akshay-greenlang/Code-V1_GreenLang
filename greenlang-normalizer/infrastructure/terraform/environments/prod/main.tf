# main.tf - Production Environment Infrastructure
# Component: GL-FOUND-X-003 - GreenLang Unit & Reference Normalizer
# Environment: Production
# Purpose: Complete AWS infrastructure for production environment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "s3" {
    bucket         = "greenlang-terraform-state"
    key            = "gl-normalizer/prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "greenlang-terraform-locks"
  }
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "GreenLang"
      Component   = "GL-FOUND-X-003"
      Environment = "prod"
      ManagedBy   = "terraform"
      Owner       = "platform-team"
      Compliance  = "required"
    }
  }
}

provider "aws" {
  alias  = "dr_region"
  region = var.dr_region
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = module.eks.cluster_auth_token
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# =============================================================================
# VPC Configuration (Production HA)
# =============================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc-${var.environment}"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  database_subnets = var.database_subnet_cidrs

  enable_nat_gateway     = true
  single_nat_gateway     = false # HA: NAT per AZ
  one_nat_gateway_per_az = true
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60

  # Tags for EKS
  public_subnet_tags = {
    "kubernetes.io/role/elb"                                      = "1"
    "kubernetes.io/cluster/${var.project_name}-${var.environment}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"                              = "1"
    "kubernetes.io/cluster/${var.project_name}-${var.environment}" = "shared"
  }

  tags = var.tags
}

# =============================================================================
# EKS Cluster (Production HA)
# =============================================================================

module "eks" {
  source = "../../modules/eks"

  cluster_name     = "${var.project_name}-${var.environment}"
  cluster_version  = var.eks_cluster_version
  vpc_id           = module.vpc.vpc_id
  subnet_ids       = concat(module.vpc.public_subnets, module.vpc.private_subnets)
  private_subnet_ids = module.vpc.private_subnets

  # Production: Private API endpoint only
  enable_public_access = false
  public_access_cidrs  = []

  # Production node groups
  general_instance_types    = ["m6i.xlarge", "m6i.2xlarge"]
  general_node_desired_size = 5
  general_node_min_size     = 3
  general_node_max_size     = 20
  node_disk_size            = 100

  # Spot nodes for batch workloads
  enable_spot_nodes      = true
  spot_instance_types    = ["m6i.xlarge", "m5.xlarge", "m5a.xlarge"]
  spot_node_desired_size = 2
  spot_node_min_size     = 0
  spot_node_max_size     = 10

  log_retention_days = 90

  audit_bucket_name = module.s3.audit_bucket_id

  tags = var.tags
}

# =============================================================================
# RDS PostgreSQL (Production HA)
# =============================================================================

module "rds" {
  source = "../../modules/rds"

  identifier  = "${var.project_name}-${var.environment}"
  environment = var.environment
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.database_subnets

  eks_node_security_group_id = module.eks.node_security_group_id

  # Production configuration
  instance_class        = "db.r6g.xlarge"
  allocated_storage     = 200
  max_allocated_storage = 2000
  multi_az              = true
  backup_retention_period = 30

  # Production safety
  skip_final_snapshot   = false
  deletion_protection   = true
  apply_immediately     = false

  # Full monitoring for production
  performance_insights_enabled = true
  enhanced_monitoring_interval = 60
  max_connections              = "500"

  # Read replica for high read throughput
  create_read_replica    = true
  replica_instance_class = "db.r6g.large"

  alarm_actions = var.alarm_actions

  tags = var.tags
}

# =============================================================================
# MSK Kafka (Production HA)
# =============================================================================

module "kafka" {
  source = "../../modules/kafka"

  cluster_name = "${var.project_name}-${var.environment}"
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.private_subnets

  eks_node_security_group_id = module.eks.node_security_group_id

  # Production configuration
  kafka_version          = "3.5.1"
  number_of_broker_nodes = 6 # 2 per AZ for HA
  broker_instance_type   = "kafka.m5.2xlarge"
  broker_ebs_volume_size = 500

  enable_provisioned_throughput = true
  provisioned_throughput_mibps  = 500

  # Secure auth for production
  enable_sasl_scram            = true
  enable_sasl_iam              = true
  enable_unauthenticated_access = false

  sasl_scram_username = "gl_normalizer_prod"
  sasl_scram_password = var.kafka_password

  # Extended retention for audit events
  log_retention_hours = 720 # 30 days
  log_retention_bytes = 1073741824000 # 1TB per partition

  # Full monitoring for production
  enhanced_monitoring  = "PER_TOPIC_PER_BROKER"
  enable_jmx_exporter  = true
  enable_node_exporter = true
  enable_s3_logs       = true
  log_retention_days   = 90

  alarm_actions = var.alarm_actions

  tags = var.tags
}

# =============================================================================
# S3 Buckets (Production with DR)
# =============================================================================

module "s3" {
  source = "../../modules/s3"

  bucket_prefix = var.project_name
  environment   = var.environment

  gl_normalizer_role_arn = module.eks.gl_normalizer_role_arn

  # Production retention and compliance
  retention_days     = 2555 # 7 years for compliance
  enable_object_lock = true
  object_lock_mode   = "GOVERNANCE"
  object_lock_retention_days = 365

  enable_access_logging = true

  # Cross-region replication for DR
  enable_cross_region_replication     = true
  replication_destination_bucket_arn  = var.dr_bucket_arn
  replication_destination_kms_key_arn = var.dr_kms_key_arn

  bucket_size_alarm_threshold_gb = 5000
  alarm_actions                  = var.alarm_actions

  tags = var.tags
}

# =============================================================================
# ElastiCache Redis (Production HA)
# =============================================================================

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-redis"
  subnet_ids = module.vpc.private_subnets

  tags = var.tags
}

resource "aws_security_group" "redis" {
  name        = "${var.project_name}-${var.environment}-redis-sg"
  description = "Security group for ElastiCache Redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-redis-sg"
  })
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "${var.project_name}-${var.environment}"
  description          = "Redis cluster for GL Normalizer production"

  engine               = "redis"
  engine_version       = "7.0"
  node_type            = "cache.r6g.large"
  port                 = 6379
  parameter_group_name = "default.redis7.cluster.on"

  # HA Configuration
  automatic_failover_enabled = true
  multi_az_enabled           = true
  num_node_groups            = 3
  replicas_per_node_group    = 1

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  # Encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  # Maintenance
  snapshot_retention_limit = 7
  snapshot_window          = "03:00-05:00"
  maintenance_window       = "sun:05:00-sun:07:00"

  # Logging
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "slow-log"
  }

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "redis" {
  name              = "/aws/elasticache/${var.project_name}-${var.environment}"
  retention_in_days = 30

  tags = var.tags
}

# =============================================================================
# Secrets Manager
# =============================================================================

resource "aws_secretsmanager_secret" "app_secrets" {
  name        = "gl-normalizer/${var.environment}/app-secrets"
  description = "Application secrets for GL Normalizer ${var.environment}"

  recovery_window_in_days = 30

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    database_url    = module.rds.connection_string
    redis_url       = "rediss://${aws_elasticache_replication_group.redis.configuration_endpoint_address}:6379"
    kafka_servers   = module.kafka.bootstrap_brokers_sasl_iam
    api_secret_key  = random_password.api_secret.result
    jwt_secret_key  = random_password.jwt_secret.result
  })
}

resource "random_password" "api_secret" {
  length  = 64
  special = true
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

# =============================================================================
# WAF for API Protection
# =============================================================================

resource "aws_wafv2_web_acl" "api" {
  name        = "${var.project_name}-${var.environment}-api-waf"
  description = "WAF for GL Normalizer API"
  scope       = "REGIONAL"

  default_action {
    allow {}
  }

  # AWS Managed Rules - Common Rule Set
  rule {
    name     = "AWS-AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesCommonRuleSet"
      sampled_requests_enabled   = true
    }
  }

  # Rate limiting
  rule {
    name     = "RateLimitRule"
    priority = 2

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-${var.environment}-api-waf"
    sampled_requests_enabled   = true
  }

  tags = var.tags
}

# =============================================================================
# CloudWatch Alarms
# =============================================================================

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-${var.environment}-alerts"

  tags = var.tags
}

# =============================================================================
# Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "rds_replica_endpoint" {
  description = "RDS read replica endpoint"
  value       = module.rds.replica_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.redis.configuration_endpoint_address
  sensitive   = true
}

output "kafka_bootstrap_servers" {
  description = "Kafka bootstrap servers (IAM)"
  value       = module.kafka.bootstrap_brokers_sasl_iam
  sensitive   = true
}

output "s3_audit_bucket" {
  description = "S3 audit bucket name"
  value       = module.s3.audit_bucket_id
}

output "secrets_manager_arn" {
  description = "Secrets Manager secret ARN"
  value       = aws_secretsmanager_secret.app_secrets.arn
}

output "waf_web_acl_arn" {
  description = "WAF Web ACL ARN"
  value       = aws_wafv2_web_acl.api.arn
}

output "sns_alerts_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = aws_sns_topic.alerts.arn
}
