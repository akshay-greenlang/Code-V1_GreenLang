# =============================================================================
# GreenLang Grafana Module - Main
# GreenLang Climate OS | OBS-002
# =============================================================================
# Provisions AWS resources for Grafana backend:
#   - RDS PostgreSQL instance (Grafana metadata store)
#   - Secrets Manager secret (DB credentials)
#   - Kubernetes secret (for Grafana to consume)
# =============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.23"
    }
  }
}

# ---------------------------------------------------------------------------
# Local Values
# ---------------------------------------------------------------------------

locals {
  name_prefix = "gl-grafana-${var.environment}"

  default_tags = merge(var.tags, {
    Component   = "grafana"
    Environment = var.environment
    ManagedBy   = "terraform"
    Module      = "OBS-002"
    Service     = "observability"
  })

  kms_key_arn = var.kms_key_arn != "" ? var.kms_key_arn : null
}

# ---------------------------------------------------------------------------
# Random Password for Grafana DB
# ---------------------------------------------------------------------------

resource "random_password" "grafana_db" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

# ---------------------------------------------------------------------------
# RDS Subnet Group
# ---------------------------------------------------------------------------

resource "aws_db_subnet_group" "grafana" {
  name        = "${local.name_prefix}-subnet-group"
  description = "Subnet group for Grafana RDS instance"
  subnet_ids  = var.private_subnet_ids

  tags = merge(local.default_tags, {
    Name = "${local.name_prefix}-subnet-group"
  })
}

# ---------------------------------------------------------------------------
# RDS Parameter Group
# ---------------------------------------------------------------------------

resource "aws_db_parameter_group" "grafana" {
  name        = "${local.name_prefix}-pg15"
  family      = "postgres15"
  description = "Parameter group for Grafana PostgreSQL 15"

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  parameter {
    name  = "pg_stat_statements.track"
    value = "all"
  }

  parameter {
    name  = "max_connections"
    value = "100"
  }

  parameter {
    name  = "idle_in_transaction_session_timeout"
    value = "30000"
  }

  tags = merge(local.default_tags, {
    Name = "${local.name_prefix}-pg15"
  })
}

# ---------------------------------------------------------------------------
# RDS PostgreSQL Instance
# ---------------------------------------------------------------------------

resource "aws_db_instance" "grafana" {
  identifier = "${local.name_prefix}-db"

  engine               = "postgres"
  engine_version       = var.db_engine_version
  instance_class       = var.db_instance_class
  allocated_storage    = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type         = "gp3"
  storage_encrypted    = true
  kms_key_id           = local.kms_key_arn

  db_name  = "grafana"
  username = "grafana_admin"
  password = random_password.grafana_db.result

  multi_az               = var.db_multi_az
  db_subnet_group_name   = aws_db_subnet_group.grafana.name
  vpc_security_group_ids = [aws_security_group.grafana_db.id]
  parameter_group_name   = aws_db_parameter_group.grafana.name

  backup_retention_period = var.db_backup_retention_period
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  deletion_protection       = var.db_deletion_protection
  skip_final_snapshot       = var.environment != "prod"
  final_snapshot_identifier = var.environment == "prod" ? "${local.name_prefix}-final-snapshot" : null

  auto_minor_version_upgrade  = true
  copy_tags_to_snapshot       = true
  performance_insights_enabled = var.environment == "prod"

  monitoring_interval = var.environment == "prod" ? 60 : 0
  monitoring_role_arn = var.environment == "prod" ? aws_iam_role.rds_monitoring[0].arn : null

  tags = merge(local.default_tags, {
    Name = "${local.name_prefix}-db"
  })
}

# ---------------------------------------------------------------------------
# RDS Enhanced Monitoring Role (prod only)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "rds_monitoring" {
  count = var.environment == "prod" ? 1 : 0

  name = "${local.name_prefix}-rds-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.default_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  count = var.environment == "prod" ? 1 : 0

  role       = aws_iam_role.rds_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ---------------------------------------------------------------------------
# Secrets Manager - Grafana DB Credentials
# ---------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "grafana_db" {
  name        = "${local.name_prefix}/db-credentials"
  description = "Grafana PostgreSQL database credentials"
  kms_key_id  = local.kms_key_arn

  tags = merge(local.default_tags, {
    Name = "${local.name_prefix}-db-credentials"
  })
}

resource "aws_secretsmanager_secret_version" "grafana_db" {
  secret_id = aws_secretsmanager_secret.grafana_db.id
  secret_string = jsonencode({
    host     = aws_db_instance.grafana.address
    port     = aws_db_instance.grafana.port
    dbname   = aws_db_instance.grafana.db_name
    username = aws_db_instance.grafana.username
    password = random_password.grafana_db.result
    engine   = "postgres"
    ssl_mode = "verify-full"
  })
}

# ---------------------------------------------------------------------------
# Kubernetes Secret for Grafana to consume
# ---------------------------------------------------------------------------

resource "kubernetes_secret" "grafana_db" {
  metadata {
    name      = "grafana-db-credentials"
    namespace = var.grafana_namespace

    labels = {
      "app.kubernetes.io/name"      = "grafana"
      "app.kubernetes.io/component" = "database"
      "app.kubernetes.io/part-of"   = "observability"
    }
  }

  data = {
    GF_DATABASE_HOST     = aws_db_instance.grafana.address
    GF_DATABASE_PORT     = tostring(aws_db_instance.grafana.port)
    GF_DATABASE_NAME     = aws_db_instance.grafana.db_name
    GF_DATABASE_USER     = aws_db_instance.grafana.username
    GF_DATABASE_PASSWORD = random_password.grafana_db.result
    GF_DATABASE_SSL_MODE = "verify-full"
    DATABASE_URL         = "postgres://${aws_db_instance.grafana.username}:${random_password.grafana_db.result}@${aws_db_instance.grafana.address}:${aws_db_instance.grafana.port}/${aws_db_instance.grafana.db_name}?sslmode=verify-full"
  }

  type = "Opaque"
}
