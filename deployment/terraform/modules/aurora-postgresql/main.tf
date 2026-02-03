################################################################################
# Aurora PostgreSQL Cluster with TimescaleDB Support
# This module creates a production-ready Aurora PostgreSQL cluster with:
# - Multi-AZ deployment
# - Configurable read replicas
# - TimescaleDB extension support
# - KMS encryption
# - IAM authentication
# - Performance Insights
# - Enhanced monitoring
################################################################################

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.0"
    }
  }
}

################################################################################
# Data Sources
################################################################################

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_partition" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

################################################################################
# Locals
################################################################################

locals {
  name_prefix = "${var.project_name}-${var.environment}"

  # Calculate the number of AZs to use (minimum 2 for Multi-AZ)
  az_count = min(length(data.aws_availability_zones.available.names), var.availability_zone_count)

  # Tags to apply to all resources
  common_tags = merge(var.tags, {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Module      = "aurora-postgresql"
  })

  # Database port
  db_port = 5432

  # Master username (sanitized)
  master_username = var.master_username != "" ? var.master_username : "postgres_admin"
}

################################################################################
# KMS Key for Encryption
################################################################################

resource "aws_kms_key" "aurora" {
  count = var.create_kms_key ? 1 : 0

  description             = "KMS key for Aurora PostgreSQL cluster ${local.name_prefix}"
  deletion_window_in_days = var.kms_key_deletion_window
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow RDS to use the key"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:CallerAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-kms-key"
  })
}

resource "aws_kms_alias" "aurora" {
  count = var.create_kms_key ? 1 : 0

  name          = "alias/${local.name_prefix}-aurora"
  target_key_id = aws_kms_key.aurora[0].key_id
}

################################################################################
# DB Subnet Group
################################################################################

resource "aws_db_subnet_group" "aurora" {
  name        = "${local.name_prefix}-aurora-subnet-group"
  description = "Subnet group for Aurora PostgreSQL cluster ${local.name_prefix}"
  subnet_ids  = var.subnet_ids

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-subnet-group"
  })
}

################################################################################
# Security Group
################################################################################

resource "aws_security_group" "aurora" {
  name        = "${local.name_prefix}-aurora-sg"
  description = "Security group for Aurora PostgreSQL cluster ${local.name_prefix}"
  vpc_id      = var.vpc_id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Ingress rule for PostgreSQL port from allowed CIDR blocks
resource "aws_security_group_rule" "aurora_ingress_cidr" {
  count = length(var.allowed_cidr_blocks) > 0 ? 1 : 0

  type              = "ingress"
  from_port         = local.db_port
  to_port           = local.db_port
  protocol          = "tcp"
  cidr_blocks       = var.allowed_cidr_blocks
  security_group_id = aws_security_group.aurora.id
  description       = "PostgreSQL access from allowed CIDR blocks"
}

# Ingress rule for PostgreSQL port from allowed security groups
resource "aws_security_group_rule" "aurora_ingress_sg" {
  count = length(var.allowed_security_groups)

  type                     = "ingress"
  from_port                = local.db_port
  to_port                  = local.db_port
  protocol                 = "tcp"
  source_security_group_id = var.allowed_security_groups[count.index]
  security_group_id        = aws_security_group.aurora.id
  description              = "PostgreSQL access from security group ${var.allowed_security_groups[count.index]}"
}

# Egress rule - allow all outbound (required for Enhanced Monitoring, etc.)
resource "aws_security_group_rule" "aurora_egress" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.aurora.id
  description       = "Allow all outbound traffic"
}

################################################################################
# Aurora PostgreSQL Cluster
################################################################################

resource "aws_rds_cluster" "aurora" {
  cluster_identifier = "${local.name_prefix}-aurora-cluster"

  # Engine configuration
  engine                      = "aurora-postgresql"
  engine_version              = var.engine_version
  engine_mode                 = "provisioned"

  # Database configuration
  database_name               = var.database_name
  master_username             = local.master_username
  master_password             = aws_secretsmanager_secret_version.master_credentials.secret_string != "" ? jsondecode(aws_secretsmanager_secret_version.master_credentials.secret_string)["password"] : null
  manage_master_user_password = false
  port                        = local.db_port

  # Network configuration
  db_subnet_group_name        = aws_db_subnet_group.aurora.name
  vpc_security_group_ids      = [aws_security_group.aurora.id]

  # Parameter groups
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.aurora_timescaledb.name

  # Storage configuration
  storage_encrypted           = true
  kms_key_id                  = var.create_kms_key ? aws_kms_key.aurora[0].arn : var.kms_key_arn
  storage_type                = var.storage_type
  allocated_storage           = var.storage_type == "aurora-iopt1" ? var.allocated_storage : null
  iops                        = var.storage_type == "aurora-iopt1" ? var.iops : null

  # Backup configuration
  backup_retention_period     = var.backup_retention_period
  preferred_backup_window     = var.preferred_backup_window
  preferred_maintenance_window = var.preferred_maintenance_window
  copy_tags_to_snapshot       = true
  skip_final_snapshot         = var.skip_final_snapshot
  final_snapshot_identifier   = var.skip_final_snapshot ? null : "${local.name_prefix}-aurora-final-snapshot"

  # Availability and replication
  availability_zones          = slice(data.aws_availability_zones.available.names, 0, local.az_count)

  # IAM authentication
  iam_database_authentication_enabled = var.iam_database_authentication_enabled

  # IAM roles for S3 export and other integrations
  dynamic "iam_roles" {
    for_each = var.enable_s3_export ? [1] : []
    content {
      role_arn     = aws_iam_role.aurora_s3_export[0].arn
      feature_name = "s3Export"
    }
  }

  # CloudWatch Logs export
  enabled_cloudwatch_logs_exports = var.enabled_cloudwatch_logs_exports

  # Deletion protection
  deletion_protection = var.deletion_protection

  # Apply changes immediately or during maintenance window
  apply_immediately = var.apply_immediately

  # Serverless v2 scaling configuration (if using serverless instances)
  dynamic "serverlessv2_scaling_configuration" {
    for_each = var.instance_class == "db.serverless" ? [1] : []
    content {
      min_capacity = var.serverless_min_capacity
      max_capacity = var.serverless_max_capacity
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-cluster"
  })

  depends_on = [
    aws_secretsmanager_secret_version.master_credentials,
    aws_iam_role.aurora_enhanced_monitoring
  ]

  lifecycle {
    ignore_changes = [
      availability_zones,
      master_password
    ]
  }
}

################################################################################
# Aurora PostgreSQL Cluster Instances
################################################################################

# Writer instance
resource "aws_rds_cluster_instance" "writer" {
  identifier                   = "${local.name_prefix}-aurora-writer"
  cluster_identifier           = aws_rds_cluster.aurora.id

  # Engine configuration
  engine                       = aws_rds_cluster.aurora.engine
  engine_version               = aws_rds_cluster.aurora.engine_version

  # Instance configuration
  instance_class               = var.instance_class
  db_parameter_group_name      = aws_db_parameter_group.aurora_timescaledb.name

  # Network configuration
  db_subnet_group_name         = aws_db_subnet_group.aurora.name
  publicly_accessible          = var.publicly_accessible

  # Availability
  availability_zone            = var.writer_availability_zone != "" ? var.writer_availability_zone : null

  # Performance Insights
  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_retention_period = var.performance_insights_enabled ? var.performance_insights_retention_period : null
  performance_insights_kms_key_id       = var.performance_insights_enabled && var.create_kms_key ? aws_kms_key.aurora[0].arn : var.performance_insights_kms_key_id

  # Enhanced monitoring
  monitoring_interval          = var.monitoring_interval
  monitoring_role_arn          = var.monitoring_interval > 0 ? aws_iam_role.aurora_enhanced_monitoring.arn : null

  # Auto minor version upgrade
  auto_minor_version_upgrade   = var.auto_minor_version_upgrade

  # Apply changes
  apply_immediately            = var.apply_immediately

  # Promotion tier (lower = higher priority for failover)
  promotion_tier               = 0

  # Copy tags
  copy_tags_to_snapshot        = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-writer"
    Role = "writer"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Reader instances
resource "aws_rds_cluster_instance" "readers" {
  count = var.replica_count

  identifier                   = "${local.name_prefix}-aurora-reader-${count.index + 1}"
  cluster_identifier           = aws_rds_cluster.aurora.id

  # Engine configuration
  engine                       = aws_rds_cluster.aurora.engine
  engine_version               = aws_rds_cluster.aurora.engine_version

  # Instance configuration
  instance_class               = var.replica_instance_class != "" ? var.replica_instance_class : var.instance_class
  db_parameter_group_name      = aws_db_parameter_group.aurora_timescaledb.name

  # Network configuration
  db_subnet_group_name         = aws_db_subnet_group.aurora.name
  publicly_accessible          = var.publicly_accessible

  # Distribute readers across AZs
  availability_zone            = element(data.aws_availability_zones.available.names, count.index % local.az_count)

  # Performance Insights
  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_retention_period = var.performance_insights_enabled ? var.performance_insights_retention_period : null
  performance_insights_kms_key_id       = var.performance_insights_enabled && var.create_kms_key ? aws_kms_key.aurora[0].arn : var.performance_insights_kms_key_id

  # Enhanced monitoring
  monitoring_interval          = var.monitoring_interval
  monitoring_role_arn          = var.monitoring_interval > 0 ? aws_iam_role.aurora_enhanced_monitoring.arn : null

  # Auto minor version upgrade
  auto_minor_version_upgrade   = var.auto_minor_version_upgrade

  # Apply changes
  apply_immediately            = var.apply_immediately

  # Promotion tier (higher = lower priority for failover)
  promotion_tier               = count.index + 1

  # Copy tags
  copy_tags_to_snapshot        = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-reader-${count.index + 1}"
    Role = "reader"
  })

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [aws_rds_cluster_instance.writer]
}

################################################################################
# Aurora Cluster Endpoint (Custom)
################################################################################

resource "aws_rds_cluster_endpoint" "readers" {
  count = var.create_reader_endpoint ? 1 : 0

  cluster_identifier          = aws_rds_cluster.aurora.id
  cluster_endpoint_identifier = "${local.name_prefix}-readers"
  custom_endpoint_type        = "READER"

  static_members = var.reader_endpoint_instance_ids

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-aurora-readers-endpoint"
  })

  depends_on = [aws_rds_cluster_instance.readers]
}
