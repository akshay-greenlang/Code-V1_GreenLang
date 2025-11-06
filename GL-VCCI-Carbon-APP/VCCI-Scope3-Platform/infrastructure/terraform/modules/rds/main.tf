# RDS Module - PostgreSQL Multi-AZ with Read Replicas

# ============================================================================
# DB Subnet Group
# ============================================================================

resource "aws_db_subnet_group" "main" {
  name       = "${var.cluster_identifier}-db-subnet-group"
  subnet_ids = var.subnet_ids

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_identifier}-db-subnet-group"
    }
  )
}

# ============================================================================
# Security Group
# ============================================================================

resource "aws_security_group" "rds" {
  name_prefix = "${var.cluster_identifier}-rds-sg-"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = var.allowed_security_groups
    description     = "Allow PostgreSQL from EKS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_identifier}-rds-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# ============================================================================
# Parameter Group
# ============================================================================

resource "aws_db_parameter_group" "main" {
  name_prefix = "${var.cluster_identifier}-pg-"
  family      = "postgres15"
  description = "Custom parameter group for ${var.cluster_identifier}"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  parameter {
    name  = "max_connections"
    value = "500"
  }

  parameter {
    name  = "work_mem"
    value = "16384"
  }

  parameter {
    name  = "maintenance_work_mem"
    value = "2097152"
  }

  parameter {
    name  = "effective_cache_size"
    value = "49152000"
  }

  parameter {
    name  = "random_page_cost"
    value = "1.1"
  }

  tags = var.tags

  lifecycle {
    create_before_destroy = true
  }
}

# ============================================================================
# Random Password
# ============================================================================

resource "random_password" "master" {
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "rds_password" {
  name                    = "${var.cluster_identifier}-master-password"
  recovery_window_in_days = 7
  kms_key_id              = var.kms_key_arn

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  secret_id = aws_secretsmanager_secret.rds_password.id
  secret_string = jsonencode({
    username = var.master_username
    password = random_password.master.result
    engine   = "postgres"
    host     = aws_db_instance.main.address
    port     = aws_db_instance.main.port
    dbname   = var.database_name
  })
}

# ============================================================================
# RDS Instance (Primary)
# ============================================================================

resource "aws_db_instance" "main" {
  identifier     = var.cluster_identifier
  engine         = "postgres"
  engine_version = var.engine_version
  instance_class = var.instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = var.storage_type
  storage_encrypted     = true
  kms_key_id            = var.kms_key_arn
  iops                  = var.storage_type == "gp3" || var.storage_type == "io1" ? var.iops : null
  storage_throughput    = var.storage_type == "gp3" ? var.storage_throughput : null

  db_name  = var.database_name
  username = var.master_username
  password = random_password.master.result

  multi_az               = var.multi_az
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = aws_db_parameter_group.main.name
  publicly_accessible    = false

  backup_retention_period = var.backup_retention_period
  backup_window           = var.backup_window
  maintenance_window      = var.maintenance_window
  copy_tags_to_snapshot   = true
  skip_final_snapshot     = false
  final_snapshot_identifier = "${var.cluster_identifier}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  deletion_protection = var.deletion_protection

  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_retention_period = var.performance_insights_retention_period
  performance_insights_kms_key_id       = var.kms_key_arn

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  auto_minor_version_upgrade = true
  apply_immediately          = false

  tags = merge(
    var.tags,
    {
      Name = var.cluster_identifier
      Role = "primary"
    }
  )

  lifecycle {
    ignore_changes = [password]
  }
}

# ============================================================================
# Read Replicas
# ============================================================================

resource "aws_db_instance" "replica" {
  count = var.read_replica_count

  identifier             = "${var.cluster_identifier}-replica-${count.index + 1}"
  replicate_source_db    = aws_db_instance.main.identifier
  instance_class         = var.instance_class
  storage_encrypted      = true
  kms_key_id             = var.kms_key_arn
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = aws_db_parameter_group.main.name

  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_retention_period = var.performance_insights_retention_period
  performance_insights_kms_key_id       = var.kms_key_arn

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  auto_minor_version_upgrade = true
  apply_immediately          = false

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_identifier}-replica-${count.index + 1}"
      Role = "replica"
    }
  )
}

# ============================================================================
# CloudWatch Alarms
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name          = "${var.cluster_identifier}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors RDS CPU utilization"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "database_memory" {
  alarm_name          = "${var.cluster_identifier}-low-memory"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "FreeableMemory"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "1000000000" # 1GB
  alarm_description   = "This metric monitors RDS freeable memory"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "database_storage" {
  alarm_name          = "${var.cluster_identifier}-low-storage"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "10000000000" # 10GB
  alarm_description   = "This metric monitors RDS free storage space"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "database_connections" {
  alarm_name          = "${var.cluster_identifier}-high-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "400" # 80% of max_connections (500)
  alarm_description   = "This metric monitors RDS database connections"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }

  tags = var.tags
}
