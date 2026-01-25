# GreenLang ElastiCache Module
# Creates a production-ready Redis cluster for caching and session management

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

# -----------------------------------------------------------------------------
# Random Auth Token for Redis
# -----------------------------------------------------------------------------
resource "random_password" "auth_token" {
  count = var.transit_encryption_enabled && var.auth_token == null ? 1 : 0

  length           = 64
  special          = false
  override_special = "!&#$^<>-"
}

# -----------------------------------------------------------------------------
# KMS Key for ElastiCache Encryption
# -----------------------------------------------------------------------------
resource "aws_kms_key" "elasticache" {
  count = var.at_rest_encryption_enabled && var.kms_key_arn == null ? 1 : 0

  description             = "KMS key for ElastiCache encryption - ${var.cluster_id}"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = merge(var.tags, {
    Name = "${var.cluster_id}-elasticache-kms"
  })
}

resource "aws_kms_alias" "elasticache" {
  count = var.at_rest_encryption_enabled && var.kms_key_arn == null ? 1 : 0

  name          = "alias/${var.cluster_id}-elasticache"
  target_key_id = aws_kms_key.elasticache[0].key_id
}

# -----------------------------------------------------------------------------
# Security Group
# -----------------------------------------------------------------------------
resource "aws_security_group" "elasticache" {
  name        = "${var.cluster_id}-elasticache-sg"
  description = "Security group for ElastiCache Redis cluster"
  vpc_id      = var.vpc_id

  tags = merge(var.tags, {
    Name = "${var.cluster_id}-elasticache-sg"
  })
}

resource "aws_security_group_rule" "ingress_cidr" {
  count = length(var.allowed_cidr_blocks) > 0 ? 1 : 0

  type              = "ingress"
  from_port         = var.port
  to_port           = var.port
  protocol          = "tcp"
  cidr_blocks       = var.allowed_cidr_blocks
  security_group_id = aws_security_group.elasticache.id
}

resource "aws_security_group_rule" "ingress_from_sg" {
  count = length(var.allowed_security_groups)

  type                     = "ingress"
  from_port                = var.port
  to_port                  = var.port
  protocol                 = "tcp"
  source_security_group_id = var.allowed_security_groups[count.index]
  security_group_id        = aws_security_group.elasticache.id
}

resource "aws_security_group_rule" "egress" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.elasticache.id
}

# -----------------------------------------------------------------------------
# Parameter Group
# -----------------------------------------------------------------------------
resource "aws_elasticache_parameter_group" "main" {
  name        = "${var.cluster_id}-params"
  family      = "redis7"
  description = "Parameter group for ${var.cluster_id}"

  # Memory management
  parameter {
    name  = "maxmemory-policy"
    value = var.maxmemory_policy
  }

  # Persistence configuration
  parameter {
    name  = "appendonly"
    value = var.enable_aof ? "yes" : "no"
  }

  # Client output buffer limits for pub/sub
  parameter {
    name  = "client-output-buffer-limit-pubsub-hard-limit"
    value = "33554432"
  }

  parameter {
    name  = "client-output-buffer-limit-pubsub-soft-limit"
    value = "8388608"
  }

  parameter {
    name  = "client-output-buffer-limit-pubsub-soft-seconds"
    value = "60"
  }

  # Connection timeout
  parameter {
    name  = "timeout"
    value = var.connection_timeout
  }

  # TCP keepalive
  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  # Notify keyspace events for cache invalidation
  parameter {
    name  = "notify-keyspace-events"
    value = var.notify_keyspace_events
  }

  tags = var.tags
}

# -----------------------------------------------------------------------------
# Replication Group (Redis Cluster Mode)
# -----------------------------------------------------------------------------
resource "aws_elasticache_replication_group" "main" {
  replication_group_id = var.cluster_id
  description          = "Redis replication group for ${var.cluster_id}"

  # Engine configuration
  engine               = "redis"
  engine_version       = var.engine_version
  node_type            = var.node_type
  port                 = var.port
  parameter_group_name = aws_elasticache_parameter_group.main.name

  # Cluster configuration
  num_cache_clusters         = var.cluster_mode_enabled ? null : var.num_cache_clusters
  num_node_groups            = var.cluster_mode_enabled ? var.num_node_groups : null
  replicas_per_node_group    = var.cluster_mode_enabled ? var.replicas_per_node_group : null
  automatic_failover_enabled = var.num_cache_clusters > 1 || var.cluster_mode_enabled

  # Network configuration
  subnet_group_name  = var.subnet_group_name
  security_group_ids = [aws_security_group.elasticache.id]

  # Multi-AZ configuration
  multi_az_enabled = var.multi_az_enabled

  # Encryption at rest
  at_rest_encryption_enabled = var.at_rest_encryption_enabled
  kms_key_id                 = var.at_rest_encryption_enabled ? (var.kms_key_arn != null ? var.kms_key_arn : aws_kms_key.elasticache[0].arn) : null

  # Encryption in transit
  transit_encryption_enabled = var.transit_encryption_enabled
  auth_token                 = var.transit_encryption_enabled ? (var.auth_token != null ? var.auth_token : random_password.auth_token[0].result) : null

  # Maintenance and backup
  maintenance_window       = var.maintenance_window
  snapshot_window          = var.snapshot_window
  snapshot_retention_limit = var.snapshot_retention_limit
  final_snapshot_identifier = var.skip_final_snapshot ? null : "${var.cluster_id}-final-snapshot"

  # Auto minor version upgrade
  auto_minor_version_upgrade = var.auto_minor_version_upgrade

  # Notifications
  notification_topic_arn = var.notification_topic_arn

  # Apply changes immediately or during maintenance window
  apply_immediately = var.apply_immediately

  tags = merge(var.tags, {
    Name = var.cluster_id
  })

  lifecycle {
    ignore_changes = [
      num_cache_clusters
    ]
  }
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms
# -----------------------------------------------------------------------------

# CPU Utilization Alarm
resource "aws_cloudwatch_metric_alarm" "cpu" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.cluster_id}-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = var.cpu_threshold
  alarm_description   = "ElastiCache CPU utilization exceeds ${var.cpu_threshold}%"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.member_clusters[0]
  }

  tags = var.tags
}

# Memory Utilization Alarm
resource "aws_cloudwatch_metric_alarm" "memory" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.cluster_id}-memory-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = var.memory_threshold
  alarm_description   = "ElastiCache memory utilization exceeds ${var.memory_threshold}%"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.member_clusters[0]
  }

  tags = var.tags
}

# Evictions Alarm
resource "aws_cloudwatch_metric_alarm" "evictions" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.cluster_id}-evictions"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "Evictions"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Sum"
  threshold           = var.evictions_threshold
  alarm_description   = "ElastiCache evictions exceed ${var.evictions_threshold}"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.member_clusters[0]
  }

  tags = var.tags
}

# Current Connections Alarm
resource "aws_cloudwatch_metric_alarm" "connections" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.cluster_id}-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CurrConnections"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = var.connections_threshold
  alarm_description   = "ElastiCache connections exceed ${var.connections_threshold}"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.member_clusters[0]
  }

  tags = var.tags
}

# Replication Lag Alarm
resource "aws_cloudwatch_metric_alarm" "replication_lag" {
  count = var.enable_cloudwatch_alarms && var.num_cache_clusters > 1 ? 1 : 0

  alarm_name          = "${var.cluster_id}-replication-lag"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ReplicationLag"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = var.replication_lag_threshold
  alarm_description   = "ElastiCache replication lag exceeds ${var.replication_lag_threshold} seconds"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.member_clusters[0]
  }

  tags = var.tags
}

# Engine CPU Utilization Alarm
resource "aws_cloudwatch_metric_alarm" "engine_cpu" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.cluster_id}-engine-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "EngineCPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = var.engine_cpu_threshold
  alarm_description   = "ElastiCache engine CPU utilization exceeds ${var.engine_cpu_threshold}%"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main.member_clusters[0]
  }

  tags = var.tags
}

# -----------------------------------------------------------------------------
# Store auth token in Secrets Manager
# -----------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "redis" {
  count = var.transit_encryption_enabled ? 1 : 0

  name        = "${var.cluster_id}/redis-auth"
  description = "Redis auth token for ${var.cluster_id}"

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "redis" {
  count = var.transit_encryption_enabled ? 1 : 0

  secret_id = aws_secretsmanager_secret.redis[0].id
  secret_string = jsonencode({
    auth_token       = var.auth_token != null ? var.auth_token : random_password.auth_token[0].result
    primary_endpoint = aws_elasticache_replication_group.main.primary_endpoint_address
    reader_endpoint  = aws_elasticache_replication_group.main.reader_endpoint_address
    port             = var.port
    connection_url   = "rediss://:${var.auth_token != null ? var.auth_token : random_password.auth_token[0].result}@${aws_elasticache_replication_group.main.primary_endpoint_address}:${var.port}"
  })
}

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------
data "aws_region" "current" {}
