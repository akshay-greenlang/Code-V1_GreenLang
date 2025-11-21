# GreenLang Infrastructure as Code - Main Terraform Configuration
# Multi-cloud support: AWS, GCP, Azure
# 50+ Terraform modules for comprehensive infrastructure

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  backend "s3" {
    bucket         = "greenlang-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
    dynamodb_table = "greenlang-terraform-locks"
  }
}

# AWS Provider Configuration
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment     = var.environment
      Project         = "GreenLang"
      ManagedBy       = "Terraform"
      CostCenter      = var.cost_center
      Owner           = var.owner_email
      DataClass       = var.data_classification
      ComplianceLevel = var.compliance_level
    }
  }
}

provider "aws" {
  alias  = "us-west-2"
  region = "us-west-2"
}

provider "aws" {
  alias  = "eu-west-1"
  region = "eu-west-1"
}

# GCP Provider Configuration
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Azure Provider Configuration
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
    key_vault {
      purge_soft_delete_on_destroy = false
    }
  }
}

# ==========================================
# NETWORKING INFRASTRUCTURE
# ==========================================

module "vpc" {
  source = "./modules/networking/vpc"

  name                 = "greenlang-${var.environment}"
  cidr_block           = var.vpc_cidr
  availability_zones   = var.availability_zones
  private_subnet_cidrs = var.private_subnet_cidrs
  public_subnet_cidrs  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_vpn_gateway   = var.enable_vpn
  enable_dns_hostnames = true
  enable_dns_support   = true

  enable_flow_logs        = true
  flow_logs_s3_bucket     = aws_s3_bucket.vpc_flow_logs.id
  flow_logs_traffic_type  = "ALL"

  tags = {
    Environment = var.environment
    Terraform   = "true"
  }
}

module "transit_gateway" {
  source = "./modules/networking/transit-gateway"

  name        = "greenlang-tgw"
  description = "Transit Gateway for multi-region connectivity"

  enable_default_route_table_association = false
  enable_default_route_table_propagation = false

  vpc_attachments = {
    vpc1 = {
      vpc_id     = module.vpc.vpc_id
      subnet_ids = module.vpc.private_subnet_ids
    }
  }
}

module "global_accelerator" {
  source = "./modules/networking/global-accelerator"

  name            = "greenlang-global"
  ip_address_type = "IPV4"
  enabled         = true

  listeners = [
    {
      port     = 443
      protocol = "TCP"
    },
    {
      port     = 80
      protocol = "TCP"
    }
  ]
}

# ==========================================
# KUBERNETES CLUSTERS
# ==========================================

module "eks_cluster" {
  source = "./modules/kubernetes/eks"

  cluster_name    = "greenlang-${var.environment}"
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  enable_irsa                     = true
  enable_cluster_autoscaler        = true
  enable_metrics_server            = true
  enable_aws_load_balancer_controller = true

  node_groups = {
    general = {
      desired_capacity = 3
      min_capacity     = 3
      max_capacity     = 10

      instance_types = ["t3.large"]

      k8s_labels = {
        Environment = var.environment
        Type        = "general"
      }
    }

    compute = {
      desired_capacity = 2
      min_capacity     = 2
      max_capacity     = 20

      instance_types = ["c5.2xlarge"]

      k8s_labels = {
        Environment = var.environment
        Type        = "compute"
      }

      taints = [
        {
          key    = "compute"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }

    gpu = {
      desired_capacity = 0
      min_capacity     = 0
      max_capacity     = 5

      instance_types = ["g4dn.xlarge"]

      k8s_labels = {
        Environment = var.environment
        Type        = "gpu"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  fargate_profiles = {
    system = {
      name = "system"
      selectors = [
        {
          namespace = "kube-system"
        },
        {
          namespace = "default"
        }
      ]
    }
  }
}

# ==========================================
# DATABASE INFRASTRUCTURE
# ==========================================

module "rds_postgresql" {
  source = "./modules/database/rds"

  identifier = "greenlang-main-${var.environment}"

  engine            = "postgres"
  engine_version    = "15.4"
  instance_class    = var.rds_instance_class
  allocated_storage = var.rds_allocated_storage
  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds.arn

  database_name  = "greenlang"
  username       = "greenlang_admin"
  password       = random_password.rds_password.result
  port           = 5432

  vpc_id                  = module.vpc.vpc_id
  subnet_ids              = module.vpc.private_subnet_ids
  create_security_group   = true
  allowed_security_groups = [module.eks_cluster.worker_security_group_id]

  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  enabled_cloudwatch_logs_exports = ["postgresql"]

  performance_insights_enabled = var.environment == "production"
  monitoring_interval          = var.environment == "production" ? 60 : 0

  deletion_protection = var.environment == "production"

  tags = {
    Environment = var.environment
  }
}

module "aurora_postgresql" {
  source = "./modules/database/aurora"

  name           = "greenlang-aurora-${var.environment}"
  engine         = "aurora-postgresql"
  engine_version = "15.3"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  replica_count          = var.environment == "production" ? 2 : 1
  instance_class         = var.aurora_instance_class

  database_name   = "greenlang"
  master_username = "admin"
  master_password = random_password.aurora_password.result

  backup_retention_period = var.environment == "production" ? 35 : 7
  preferred_backup_window = "03:00-04:00"

  enabled_cloudwatch_logs_exports = ["postgresql"]

  scaling_configuration = {
    auto_pause               = var.environment != "production"
    min_capacity             = 2
    max_capacity             = 16
    seconds_until_auto_pause = 300
  }

  tags = {
    Environment = var.environment
  }
}

module "documentdb" {
  source = "./modules/database/documentdb"

  cluster_identifier = "greenlang-docdb-${var.environment}"
  engine_version     = "5.0.0"

  master_username = "greenlang"
  master_password = random_password.docdb_password.result

  instance_class = var.docdb_instance_class
  instance_count = var.environment == "production" ? 3 : 1

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  backup_retention_period = var.environment == "production" ? 7 : 1
  preferred_backup_window = "03:00-04:00"

  tags = {
    Environment = var.environment
  }
}

# ==========================================
# CACHING INFRASTRUCTURE
# ==========================================

module "elasticache_redis" {
  source = "./modules/caching/elasticache"

  name = "greenlang-redis-${var.environment}"

  engine         = "redis"
  engine_version = "7.0"

  node_type       = var.redis_node_type
  num_cache_nodes = var.environment == "production" ? 3 : 1

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  automatic_failover_enabled = var.environment == "production"
  multi_az_enabled           = var.environment == "production"

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window          = "03:00-05:00"

  tags = {
    Environment = var.environment
  }
}

# ==========================================
# STORAGE INFRASTRUCTURE
# ==========================================

module "s3_data_lake" {
  source = "./modules/storage/s3"

  bucket_name = "greenlang-data-lake-${var.environment}"

  versioning = true
  encryption = true

  lifecycle_rules = [
    {
      id      = "archive-old-data"
      enabled = true

      transition = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER"
        },
        {
          days          = 365
          storage_class = "DEEP_ARCHIVE"
        }
      ]

      expiration = {
        days = 2555  # 7 years
      }
    }
  ]

  replication_configuration = var.environment == "production" ? {
    role = aws_iam_role.s3_replication.arn
    rules = [
      {
        id       = "replicate-to-dr-region"
        status   = "Enabled"
        priority = 10

        destination = {
          bucket        = aws_s3_bucket.dr_replica.arn
          storage_class = "STANDARD_IA"
        }
      }
    ]
  } : null

  tags = {
    Environment = var.environment
    DataLake    = "true"
  }
}

module "efs" {
  source = "./modules/storage/efs"

  name = "greenlang-efs-${var.environment}"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  encrypted      = true
  kms_key_id     = aws_kms_key.efs.arn

  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  lifecycle_policy = {
    transition_to_ia = "AFTER_30_DAYS"
  }

  backup_policy = {
    status = "ENABLED"
  }

  tags = {
    Environment = var.environment
  }
}

# ==========================================
# MESSAGING AND QUEUING
# ==========================================

module "msk_kafka" {
  source = "./modules/messaging/msk"

  cluster_name = "greenlang-kafka-${var.environment}"
  kafka_version = "3.5.1"

  number_of_broker_nodes = var.environment == "production" ? 3 : 1
  instance_type          = var.kafka_instance_type

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  ebs_volume_size = var.kafka_volume_size

  encryption_info = {
    encryption_in_transit = {
      client_broker = "TLS"
      in_cluster    = true
    }
    encryption_at_rest_kms_key_id = aws_kms_key.msk.arn
  }

  configuration_info = {
    arn      = aws_msk_configuration.kafka.arn
    revision = aws_msk_configuration.kafka.latest_revision
  }

  logging_info = {
    broker_logs = {
      cloudwatch_logs = {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
      s3 = {
        enabled = true
        bucket  = aws_s3_bucket.logs.id
        prefix  = "msk-logs"
      }
    }
  }

  tags = {
    Environment = var.environment
  }
}

module "sqs" {
  source = "./modules/messaging/sqs"

  queues = {
    "carbon-calculations" = {
      delay_seconds              = 0
      max_message_size           = 262144
      message_retention_seconds  = 1209600
      receive_wait_time_seconds  = 20
      visibility_timeout_seconds = 300
      fifo_queue                 = false
      content_based_deduplication = false

      redrive_policy = {
        deadLetterTargetArn = module.sqs_dlq["carbon-calculations-dlq"].arn
        maxReceiveCount     = 3
      }
    }

    "report-generation" = {
      delay_seconds              = 0
      max_message_size           = 262144
      message_retention_seconds  = 1209600
      receive_wait_time_seconds  = 20
      visibility_timeout_seconds = 900
      fifo_queue                 = true
      content_based_deduplication = true
    }
  }

  tags = {
    Environment = var.environment
  }
}

# ==========================================
# MONITORING AND OBSERVABILITY
# ==========================================

module "cloudwatch" {
  source = "./modules/monitoring/cloudwatch"

  log_groups = {
    "/aws/eks/greenlang" = {
      retention_in_days = var.environment == "production" ? 90 : 30
      kms_key_id        = aws_kms_key.logs.arn
    }
    "/aws/lambda/greenlang" = {
      retention_in_days = var.environment == "production" ? 60 : 14
      kms_key_id        = aws_kms_key.logs.arn
    }
  }

  metric_alarms = {
    high_cpu = {
      alarm_name          = "greenlang-high-cpu-${var.environment}"
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 2
      metric_name         = "CPUUtilization"
      namespace           = "AWS/ECS"
      period              = 300
      statistic           = "Average"
      threshold           = 80
      alarm_description   = "This metric monitors CPU utilization"
      alarm_actions       = [aws_sns_topic.alerts.arn]
    }

    high_memory = {
      alarm_name          = "greenlang-high-memory-${var.environment}"
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 2
      metric_name         = "MemoryUtilization"
      namespace           = "AWS/ECS"
      period              = 300
      statistic           = "Average"
      threshold           = 85
      alarm_description   = "This metric monitors memory utilization"
      alarm_actions       = [aws_sns_topic.alerts.arn]
    }
  }

  tags = {
    Environment = var.environment
  }
}

# ==========================================
# SECURITY AND COMPLIANCE
# ==========================================

module "security_hub" {
  source = "./modules/security/security-hub"

  enable_cis_standard         = true
  enable_pci_dss_standard     = true
  enable_aws_foundational_standard = true

  member_accounts = var.environment == "production" ? var.organization_accounts : []

  tags = {
    Environment = var.environment
  }
}

module "guardduty" {
  source = "./modules/security/guardduty"

  enable                       = true
  finding_publishing_frequency = var.environment == "production" ? "FIFTEEN_MINUTES" : "ONE_HOUR"

  datasources = {
    s3_logs                      = true
    kubernetes_audit_logs        = true
    malware_protection           = true
    runtime_monitoring           = true
  }

  tags = {
    Environment = var.environment
  }
}

module "waf" {
  source = "./modules/security/waf"

  name  = "greenlang-waf-${var.environment}"
  scope = "REGIONAL"

  rules = [
    {
      name     = "RateLimitRule"
      priority = 1
      action   = "block"

      rate_based_statement = {
        limit              = 2000
        aggregate_key_type = "IP"
      }

      visibility_config = {
        cloudwatch_metrics_enabled = true
        metric_name                = "RateLimitRule"
        sampled_requests_enabled   = true
      }
    },
    {
      name     = "GeoBlockingRule"
      priority = 2
      action   = "block"

      geo_match_statement = {
        country_codes = var.blocked_countries
      }

      visibility_config = {
        cloudwatch_metrics_enabled = true
        metric_name                = "GeoBlockingRule"
        sampled_requests_enabled   = true
      }
    }
  ]

  tags = {
    Environment = var.environment
  }
}

# ==========================================
# OUTPUTS
# ==========================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks_cluster.cluster_endpoint
  sensitive   = true
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds_postgresql.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.elasticache_redis.primary_endpoint
  sensitive   = true
}

output "kafka_bootstrap_servers" {
  description = "Kafka bootstrap servers"
  value       = module.msk_kafka.bootstrap_brokers
  sensitive   = true
}