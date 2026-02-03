################################################################################
# Aurora PostgreSQL CloudWatch Alarms
# Monitoring alarms for CPU, connections, replication, storage, and IOPS
################################################################################

################################################################################
# CPU Utilization Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "cpu_utilization_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-cpu-utilization-high"
  alarm_description   = "Aurora PostgreSQL cluster CPU utilization is above ${var.alarm_cpu_threshold}%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_cpu_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-cpu-utilization-high"
    Severity = "warning"
  })
}

################################################################################
# Database Connections Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "connections_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-connections-high"
  alarm_description   = "Aurora PostgreSQL cluster connection count is above ${var.alarm_connections_threshold}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_connections_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-connections-high"
    Severity = "warning"
  })
}

################################################################################
# Replication Lag Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "replication_lag_high" {
  count = var.create_cloudwatch_alarms && var.replica_count > 0 ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-replication-lag-high"
  alarm_description   = "Aurora PostgreSQL replication lag is above ${var.alarm_replication_lag_threshold}ms"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "AuroraReplicaLag"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Maximum"
  threshold           = var.alarm_replication_lag_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-replication-lag-high"
    Severity = "warning"
  })
}

################################################################################
# Freeable Memory Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "freeable_memory_low" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-freeable-memory-low"
  alarm_description   = "Aurora PostgreSQL freeable memory is below ${var.alarm_freeable_memory_threshold / 1000000} MB"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  metric_name         = "FreeableMemory"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_freeable_memory_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-freeable-memory-low"
    Severity = "critical"
  })
}

################################################################################
# Free Storage Space Alarm (Aurora Local Storage)
################################################################################

resource "aws_cloudwatch_metric_alarm" "free_storage_space_low" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-free-storage-low"
  alarm_description   = "Aurora PostgreSQL free local storage is below ${var.alarm_free_storage_threshold / 1073741824} GB"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  metric_name         = "FreeLocalStorage"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_free_storage_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-free-storage-low"
    Severity = "critical"
  })
}

################################################################################
# Read IOPS Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "read_iops_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-read-iops-high"
  alarm_description   = "Aurora PostgreSQL read IOPS is above ${var.alarm_read_iops_threshold}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ReadIOPS"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_read_iops_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-read-iops-high"
    Severity = "warning"
  })
}

################################################################################
# Write IOPS Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "write_iops_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-write-iops-high"
  alarm_description   = "Aurora PostgreSQL write IOPS is above ${var.alarm_write_iops_threshold}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "WriteIOPS"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_write_iops_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-write-iops-high"
    Severity = "warning"
  })
}

################################################################################
# Deadlock Detection Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "deadlocks" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-deadlocks"
  alarm_description   = "Aurora PostgreSQL is experiencing deadlocks"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Deadlocks"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-deadlocks"
    Severity = "critical"
  })
}

################################################################################
# Buffer Cache Hit Ratio Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "buffer_cache_hit_ratio_low" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-buffer-cache-hit-ratio-low"
  alarm_description   = "Aurora PostgreSQL buffer cache hit ratio is below 95%"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  metric_name         = "BufferCacheHitRatio"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 95
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-buffer-cache-hit-ratio-low"
    Severity = "warning"
  })
}

################################################################################
# Instance-Level CPU Alarms (Writer)
################################################################################

resource "aws_cloudwatch_metric_alarm" "writer_cpu_utilization_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-writer-cpu-high"
  alarm_description   = "Aurora PostgreSQL writer instance CPU utilization is above ${var.alarm_cpu_threshold}%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_cpu_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_rds_cluster_instance.writer.identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-writer-cpu-high"
    Role     = "writer"
    Severity = "warning"
  })
}

################################################################################
# Instance-Level CPU Alarms (Readers)
################################################################################

resource "aws_cloudwatch_metric_alarm" "reader_cpu_utilization_high" {
  count = var.create_cloudwatch_alarms ? var.replica_count : 0

  alarm_name          = "${local.name_prefix}-aurora-reader-${count.index + 1}-cpu-high"
  alarm_description   = "Aurora PostgreSQL reader-${count.index + 1} CPU utilization is above ${var.alarm_cpu_threshold}%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_cpu_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_rds_cluster_instance.readers[count.index].identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-reader-${count.index + 1}-cpu-high"
    Role     = "reader"
    Severity = "warning"
  })
}

################################################################################
# Commit Latency Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "commit_latency_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-commit-latency-high"
  alarm_description   = "Aurora PostgreSQL commit latency is above 20ms"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CommitLatency"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 20  # 20 milliseconds
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-commit-latency-high"
    Severity = "warning"
  })
}

################################################################################
# Aurora Volume Bytes Used Alarm
################################################################################

resource "aws_cloudwatch_metric_alarm" "volume_bytes_used_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-volume-bytes-used-high"
  alarm_description   = "Aurora PostgreSQL storage volume is over 80% utilized"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "VolumeBytesUsed"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 107374182400  # 100 GB threshold (adjust as needed)
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-volume-bytes-used-high"
    Severity = "warning"
  })
}

################################################################################
# Network Throughput Alarms
################################################################################

resource "aws_cloudwatch_metric_alarm" "network_receive_throughput_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-network-receive-high"
  alarm_description   = "Aurora PostgreSQL network receive throughput is unusually high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "NetworkReceiveThroughput"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 100000000  # 100 MB/s
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-network-receive-high"
    Severity = "info"
  })
}

resource "aws_cloudwatch_metric_alarm" "network_transmit_throughput_high" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-aurora-network-transmit-high"
  alarm_description   = "Aurora PostgreSQL network transmit throughput is unusually high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "NetworkTransmitThroughput"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 100000000  # 100 MB/s
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.aurora.cluster_identifier
  }

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-network-transmit-high"
    Severity = "info"
  })
}

################################################################################
# Composite Alarm - Critical Database Health
################################################################################

resource "aws_cloudwatch_composite_alarm" "database_critical" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name        = "${local.name_prefix}-aurora-critical-health"
  alarm_description = "Composite alarm for critical Aurora PostgreSQL health issues"

  alarm_rule = <<-EOF
    ALARM(${aws_cloudwatch_metric_alarm.freeable_memory_low[0].alarm_name}) OR
    ALARM(${aws_cloudwatch_metric_alarm.free_storage_space_low[0].alarm_name}) OR
    ALARM(${aws_cloudwatch_metric_alarm.deadlocks[0].alarm_name})
  EOF

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = merge(local.common_tags, {
    Name     = "${local.name_prefix}-aurora-critical-health"
    Severity = "critical"
  })

  depends_on = [
    aws_cloudwatch_metric_alarm.freeable_memory_low,
    aws_cloudwatch_metric_alarm.free_storage_space_low,
    aws_cloudwatch_metric_alarm.deadlocks
  ]
}

################################################################################
# CloudWatch Dashboard
################################################################################

resource "aws_cloudwatch_dashboard" "aurora" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  dashboard_name = "${local.name_prefix}-aurora-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 1
        properties = {
          markdown = "# Aurora PostgreSQL - ${local.name_prefix} Dashboard"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 1
        width  = 8
        height = 6
        properties = {
          title   = "CPU Utilization"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier, { label = "Cluster Average" }]
          ]
          period = 300
          stat   = "Average"
          yAxis = {
            left = {
              min = 0
              max = 100
            }
          }
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 1
        width  = 8
        height = 6
        properties = {
          title   = "Database Connections"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "DatabaseConnections", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier]
          ]
          period = 300
          stat   = "Average"
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 1
        width  = 8
        height = 6
        properties = {
          title   = "Freeable Memory"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "FreeableMemory", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier]
          ]
          period = 300
          stat   = "Average"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 7
        width  = 12
        height = 6
        properties = {
          title   = "Read/Write IOPS"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "ReadIOPS", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier, { label = "Read IOPS" }],
            ["AWS/RDS", "WriteIOPS", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier, { label = "Write IOPS" }]
          ]
          period = 300
          stat   = "Average"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 7
        width  = 12
        height = 6
        properties = {
          title   = "Read/Write Latency"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "ReadLatency", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier, { label = "Read Latency" }],
            ["AWS/RDS", "WriteLatency", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier, { label = "Write Latency" }],
            ["AWS/RDS", "CommitLatency", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier, { label = "Commit Latency" }]
          ]
          period = 300
          stat   = "Average"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 13
        width  = 8
        height = 6
        properties = {
          title   = "Aurora Replica Lag"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "AuroraReplicaLag", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier]
          ]
          period = 300
          stat   = "Maximum"
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 13
        width  = 8
        height = 6
        properties = {
          title   = "Buffer Cache Hit Ratio"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "BufferCacheHitRatio", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier]
          ]
          period = 300
          stat   = "Average"
          yAxis = {
            left = {
              min = 0
              max = 100
            }
          }
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 13
        width  = 8
        height = 6
        properties = {
          title   = "Volume Bytes Used"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "VolumeBytesUsed", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier]
          ]
          period = 300
          stat   = "Average"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 19
        width  = 12
        height = 6
        properties = {
          title   = "Network Throughput"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "NetworkReceiveThroughput", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier, { label = "Receive" }],
            ["AWS/RDS", "NetworkTransmitThroughput", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier, { label = "Transmit" }]
          ]
          period = 300
          stat   = "Average"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 19
        width  = 12
        height = 6
        properties = {
          title   = "Deadlocks"
          region  = data.aws_region.current.name
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/RDS", "Deadlocks", "DBClusterIdentifier", aws_rds_cluster.aurora.cluster_identifier]
          ]
          period = 300
          stat   = "Sum"
        }
      }
    ]
  })
}
