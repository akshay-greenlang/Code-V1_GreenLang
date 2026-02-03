################################################################################
# Production Aurora PostgreSQL Configuration
# GreenLang Production Environment
################################################################################

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "greenlang-terraform-state"
    key            = "prod/aurora/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "greenlang-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "GreenLang"
      Environment = "prod"
      ManagedBy   = "terraform"
      Owner       = "platform-team"
    }
  }
}

################################################################################
# Variables
################################################################################

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID for Aurora deployment"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for Aurora"
  type        = list(string)
}

variable "app_security_group_ids" {
  description = "Application security group IDs allowed to access Aurora"
  type        = list(string)
  default     = []
}

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarms"
  type        = string
  default     = ""
}

variable "s3_export_bucket_arn" {
  description = "S3 bucket ARN for Aurora exports"
  type        = string
  default     = ""
}

################################################################################
# Data Sources
################################################################################

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# Get existing VPC info
data "aws_vpc" "main" {
  id = var.vpc_id
}

################################################################################
# Aurora PostgreSQL Module - Production Configuration
################################################################################

module "aurora_postgresql" {
  source = "../../modules/aurora-postgresql"

  # General Configuration
  project_name = "greenlang"
  environment  = "prod"

  tags = {
    CostCenter  = "engineering"
    Compliance  = "soc2"
    DataClass   = "confidential"
    BackupClass = "critical"
  }

  # Database Configuration
  database_name   = "greenlang_prod"
  master_username = "greenlang_admin"
  engine_version  = "15.4"

  # Instance Configuration - Production sized
  instance_class         = "db.r6g.xlarge"  # 4 vCPU, 32 GB RAM
  replica_instance_class = "db.r6g.large"    # 2 vCPU, 16 GB RAM for readers
  replica_count          = 2                 # 2 read replicas for HA
  availability_zone_count = 3

  # Serverless Configuration (if using db.serverless)
  # instance_class          = "db.serverless"
  # serverless_min_capacity = 2
  # serverless_max_capacity = 64

  # Storage Configuration
  storage_type      = "aurora"
  # For I/O optimized storage (high IOPS workloads):
  # storage_type      = "aurora-iopt1"
  # allocated_storage = 200
  # iops             = 5000

  # Network Configuration
  vpc_id                  = var.vpc_id
  subnet_ids              = var.private_subnet_ids
  allowed_cidr_blocks     = []  # No direct CIDR access in production
  allowed_security_groups = var.app_security_group_ids
  publicly_accessible     = false

  # Encryption Configuration
  create_kms_key          = true
  kms_key_deletion_window = 30

  # Backup Configuration
  backup_retention_period      = 35  # Maximum retention
  preferred_backup_window      = "03:00-04:00"  # 3 AM - 4 AM UTC
  preferred_maintenance_window = "sun:04:00-sun:05:00"  # Sunday 4 AM - 5 AM UTC
  skip_final_snapshot          = false

  # IAM Configuration
  iam_database_authentication_enabled = true
  enable_s3_export                    = true
  s3_export_bucket_arns               = var.s3_export_bucket_arn != "" ? [var.s3_export_bucket_arn] : []

  # Monitoring Configuration
  monitoring_interval                   = 60  # Enhanced monitoring every 60 seconds
  performance_insights_enabled          = true
  performance_insights_retention_period = 731  # 2 years (paid tier)
  enabled_cloudwatch_logs_exports       = ["postgresql"]

  # CloudWatch Alarms Configuration
  create_cloudwatch_alarms      = true
  alarm_cpu_threshold           = 80
  alarm_connections_threshold   = 800  # 80% of max_connections
  alarm_replication_lag_threshold = 100  # 100ms
  alarm_freeable_memory_threshold = 536870912  # 512 MB
  alarm_free_storage_threshold   = 21474836480  # 20 GB
  alarm_read_iops_threshold      = 15000
  alarm_write_iops_threshold     = 10000
  alarm_actions                  = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []
  ok_actions                     = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  # TimescaleDB Configuration
  timescaledb_max_background_workers = 16
  timescaledb_telemetry_level        = "off"

  # Connection Configuration
  max_connections                     = 1000
  idle_in_transaction_session_timeout = 300000  # 5 minutes
  statement_timeout                   = 0        # No timeout (application should handle)

  # Additional Cluster Parameters for Production
  cluster_parameters = [
    {
      name         = "log_min_duration_statement"
      value        = "500"  # Log queries over 500ms
      apply_method = "immediate"
    },
    {
      name         = "log_temp_files"
      value        = "1024"  # Log temp files > 1MB
      apply_method = "immediate"
    },
    {
      name         = "track_io_timing"
      value        = "on"
      apply_method = "immediate"
    },
    {
      name         = "track_functions"
      value        = "all"
      apply_method = "immediate"
    }
  ]

  # Additional Instance Parameters
  instance_parameters = []

  # Secrets Manager Configuration
  create_application_credentials = true
  application_username          = "greenlang_app"
  secret_rotation_enabled       = false  # Enable when Lambda is deployed
  secret_rotation_days          = 30

  # Maintenance Configuration
  auto_minor_version_upgrade = true
  apply_immediately          = false  # Apply during maintenance window
  deletion_protection        = true   # Prevent accidental deletion
}

################################################################################
# Outputs
################################################################################

output "aurora_cluster_id" {
  description = "The Aurora cluster identifier"
  value       = module.aurora_postgresql.cluster_id
}

output "aurora_cluster_arn" {
  description = "The ARN of the Aurora cluster"
  value       = module.aurora_postgresql.cluster_arn
}

output "aurora_cluster_endpoint" {
  description = "The cluster endpoint (writer)"
  value       = module.aurora_postgresql.cluster_endpoint
}

output "aurora_reader_endpoint" {
  description = "The reader endpoint for the cluster"
  value       = module.aurora_postgresql.cluster_reader_endpoint
}

output "aurora_security_group_id" {
  description = "The ID of the Aurora security group"
  value       = module.aurora_postgresql.security_group_id
}

output "aurora_master_secret_arn" {
  description = "The ARN of the master credentials secret"
  value       = module.aurora_postgresql.master_credentials_secret_arn
}

output "aurora_app_secret_arn" {
  description = "The ARN of the application credentials secret"
  value       = module.aurora_postgresql.application_credentials_secret_arn
}

output "aurora_kms_key_arn" {
  description = "The ARN of the KMS key used for encryption"
  value       = module.aurora_postgresql.kms_key_arn
}

output "aurora_enhanced_monitoring_role_arn" {
  description = "The ARN of the enhanced monitoring IAM role"
  value       = module.aurora_postgresql.enhanced_monitoring_role_arn
}

output "aurora_secrets_access_role_arn" {
  description = "The ARN of the Secrets Manager access IAM role"
  value       = module.aurora_postgresql.secrets_access_role_arn
}

output "aurora_iam_auth_role_arn" {
  description = "The ARN of the IAM database authentication role"
  value       = module.aurora_postgresql.iam_auth_role_arn
}

output "aurora_connection_string" {
  description = "PostgreSQL connection string (password stored in Secrets Manager)"
  value       = module.aurora_postgresql.connection_string
  sensitive   = true
}

output "aurora_reader_connection_string" {
  description = "PostgreSQL reader connection string"
  value       = module.aurora_postgresql.reader_connection_string
  sensitive   = true
}

output "aurora_cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL for Aurora monitoring"
  value       = "https://${data.aws_region.current.name}.console.aws.amazon.com/cloudwatch/home?region=${data.aws_region.current.name}#dashboards:name=greenlang-prod-aurora-dashboard"
}

################################################################################
# Additional Production Resources
################################################################################

# SNS Topic for Aurora Alarms (if not provided)
resource "aws_sns_topic" "aurora_alarms" {
  count = var.alarm_sns_topic_arn == "" ? 1 : 0

  name = "greenlang-prod-aurora-alarms"

  tags = {
    Name = "greenlang-prod-aurora-alarms"
  }
}

# CloudWatch Log Group for PostgreSQL logs with retention
resource "aws_cloudwatch_log_group" "aurora_postgresql" {
  name              = "/aws/rds/cluster/${module.aurora_postgresql.cluster_id}/postgresql"
  retention_in_days = 90  # 90 days retention for production

  tags = {
    Name = "greenlang-prod-aurora-postgresql-logs"
  }
}

# Route53 CNAME records (optional - uncomment if using Route53)
# resource "aws_route53_record" "aurora_writer" {
#   zone_id = var.route53_zone_id
#   name    = "db.greenlang.internal"
#   type    = "CNAME"
#   ttl     = 300
#   records = [module.aurora_postgresql.cluster_endpoint]
# }

# resource "aws_route53_record" "aurora_reader" {
#   zone_id = var.route53_zone_id
#   name    = "db-reader.greenlang.internal"
#   type    = "CNAME"
#   ttl     = 300
#   records = [module.aurora_postgresql.cluster_reader_endpoint]
# }
