# main.tf - AWS MSK (Managed Streaming for Apache Kafka) Module
# Component: GL-FOUND-X-003 - Unit & Reference Normalizer
# Purpose: Provision MSK cluster for audit event streaming

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# =============================================================================
# KMS Key for MSK Encryption
# =============================================================================

resource "aws_kms_key" "msk" {
  description             = "KMS key for MSK encryption - ${var.cluster_name}"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = merge(var.tags, {
    Name = "${var.cluster_name}-msk-kms"
  })
}

resource "aws_kms_alias" "msk" {
  name          = "alias/${var.cluster_name}-msk"
  target_key_id = aws_kms_key.msk.key_id
}

# =============================================================================
# Security Group for MSK
# =============================================================================

resource "aws_security_group" "msk" {
  name        = "${var.cluster_name}-msk-sg"
  description = "Security group for MSK cluster"
  vpc_id      = var.vpc_id

  tags = merge(var.tags, {
    Name = "${var.cluster_name}-msk-sg"
  })
}

resource "aws_security_group_rule" "msk_ingress_eks" {
  type                     = "ingress"
  description              = "Kafka access from EKS nodes"
  from_port                = 9092
  to_port                  = 9098
  protocol                 = "tcp"
  source_security_group_id = var.eks_node_security_group_id
  security_group_id        = aws_security_group.msk.id
}

resource "aws_security_group_rule" "msk_ingress_zookeeper" {
  type                     = "ingress"
  description              = "Zookeeper access from EKS nodes"
  from_port                = 2181
  to_port                  = 2181
  protocol                 = "tcp"
  source_security_group_id = var.eks_node_security_group_id
  security_group_id        = aws_security_group.msk.id
}

resource "aws_security_group_rule" "msk_ingress_self" {
  type              = "ingress"
  description       = "Inter-broker communication"
  from_port         = 0
  to_port           = 65535
  protocol          = "tcp"
  self              = true
  security_group_id = aws_security_group.msk.id
}

resource "aws_security_group_rule" "msk_egress" {
  type              = "egress"
  description       = "Allow all outbound"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.msk.id
}

# =============================================================================
# CloudWatch Log Group for MSK
# =============================================================================

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${var.cluster_name}"
  retention_in_days = var.log_retention_days

  tags = var.tags
}

# =============================================================================
# S3 Bucket for MSK Logs (Optional)
# =============================================================================

resource "aws_s3_bucket" "msk_logs" {
  count  = var.enable_s3_logs ? 1 : 0
  bucket = "${var.cluster_name}-msk-logs-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name = "${var.cluster_name}-msk-logs"
  })
}

resource "aws_s3_bucket_versioning" "msk_logs" {
  count  = var.enable_s3_logs ? 1 : 0
  bucket = aws_s3_bucket.msk_logs[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "msk_logs" {
  count  = var.enable_s3_logs ? 1 : 0
  bucket = aws_s3_bucket.msk_logs[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.msk.arn
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "msk_logs" {
  count  = var.enable_s3_logs ? 1 : 0
  bucket = aws_s3_bucket.msk_logs[0].id

  rule {
    id     = "log-retention"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# =============================================================================
# MSK Configuration
# =============================================================================

resource "aws_msk_configuration" "main" {
  name              = "${var.cluster_name}-config"
  kafka_versions    = [var.kafka_version]
  description       = "MSK configuration for GL Normalizer audit events"

  server_properties = <<PROPERTIES
auto.create.topics.enable=true
default.replication.factor=${var.number_of_broker_nodes >= 3 ? 3 : var.number_of_broker_nodes}
min.insync.replicas=${var.number_of_broker_nodes >= 3 ? 2 : 1}
num.io.threads=8
num.network.threads=5
num.partitions=${var.default_partitions}
num.replica.fetchers=2
replica.lag.time.max.ms=30000
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600
socket.send.buffer.bytes=102400
unclean.leader.election.enable=false
zookeeper.session.timeout.ms=18000
log.retention.hours=${var.log_retention_hours}
log.retention.bytes=${var.log_retention_bytes}
log.segment.bytes=1073741824
log.cleanup.policy=delete
message.max.bytes=10485760
compression.type=producer
PROPERTIES

  lifecycle {
    create_before_destroy = true
  }
}

# =============================================================================
# MSK Cluster
# =============================================================================

resource "aws_msk_cluster" "main" {
  cluster_name           = var.cluster_name
  kafka_version          = var.kafka_version
  number_of_broker_nodes = var.number_of_broker_nodes

  broker_node_group_info {
    instance_type   = var.broker_instance_type
    client_subnets  = var.subnet_ids
    security_groups = [aws_security_group.msk.id]

    storage_info {
      ebs_storage_info {
        volume_size = var.broker_ebs_volume_size
        provisioned_throughput {
          enabled           = var.enable_provisioned_throughput
          volume_throughput = var.enable_provisioned_throughput ? var.provisioned_throughput_mibps : null
        }
      }
    }

    connectivity_info {
      public_access {
        type = "DISABLED"
      }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }

  encryption_info {
    encryption_at_rest_kms_key_arn = aws_kms_key.msk.arn

    encryption_in_transit {
      client_broker = var.encryption_in_transit_client_broker
      in_cluster    = true
    }
  }

  client_authentication {
    sasl {
      scram = var.enable_sasl_scram
      iam   = var.enable_sasl_iam
    }

    unauthenticated = var.enable_unauthenticated_access
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }

      dynamic "s3" {
        for_each = var.enable_s3_logs ? [1] : []
        content {
          enabled = true
          bucket  = aws_s3_bucket.msk_logs[0].id
          prefix  = "msk-logs/"
        }
      }
    }
  }

  open_monitoring {
    prometheus {
      jmx_exporter {
        enabled_in_broker = var.enable_jmx_exporter
      }
      node_exporter {
        enabled_in_broker = var.enable_node_exporter
      }
    }
  }

  enhanced_monitoring = var.enhanced_monitoring

  tags = merge(var.tags, {
    Name = var.cluster_name
  })
}

# =============================================================================
# MSK SCRAM Secrets (if SASL/SCRAM enabled)
# =============================================================================

resource "aws_secretsmanager_secret" "msk_credentials" {
  count = var.enable_sasl_scram ? 1 : 0

  name        = "AmazonMSK_${var.cluster_name}_credentials"
  description = "SASL/SCRAM credentials for MSK cluster ${var.cluster_name}"
  kms_key_id  = aws_kms_key.msk.arn

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "msk_credentials" {
  count = var.enable_sasl_scram ? 1 : 0

  secret_id = aws_secretsmanager_secret.msk_credentials[0].id
  secret_string = jsonencode({
    username = var.sasl_scram_username
    password = var.sasl_scram_password
  })
}

resource "aws_msk_scram_secret_association" "main" {
  count = var.enable_sasl_scram ? 1 : 0

  cluster_arn     = aws_msk_cluster.main.arn
  secret_arn_list = [aws_secretsmanager_secret.msk_credentials[0].arn]

  depends_on = [aws_secretsmanager_secret_version.msk_credentials]
}

# =============================================================================
# CloudWatch Alarms
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${var.cluster_name}-msk-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CpuSystem"
  namespace           = "AWS/Kafka"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "MSK broker CPU utilization is high"
  alarm_actions       = var.alarm_actions

  dimensions = {
    "Cluster Name" = aws_msk_cluster.main.cluster_name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "disk_usage_high" {
  alarm_name          = "${var.cluster_name}-msk-disk-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "KafkaDataLogsDiskUsed"
  namespace           = "AWS/Kafka"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "MSK broker disk usage is high"
  alarm_actions       = var.alarm_actions

  dimensions = {
    "Cluster Name" = aws_msk_cluster.main.cluster_name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "offline_partitions" {
  alarm_name          = "${var.cluster_name}-msk-offline-partitions"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "OfflinePartitionsCount"
  namespace           = "AWS/Kafka"
  period              = 60
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "MSK has offline partitions"
  alarm_actions       = var.alarm_actions

  dimensions = {
    "Cluster Name" = aws_msk_cluster.main.cluster_name
  }

  tags = var.tags
}
