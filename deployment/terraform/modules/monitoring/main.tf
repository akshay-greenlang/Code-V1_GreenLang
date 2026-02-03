# GreenLang Monitoring Module
# Creates CloudWatch dashboards, alarms, and log groups for infrastructure monitoring

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# -----------------------------------------------------------------------------
# Local Variables
# -----------------------------------------------------------------------------
locals {
  dashboard_name = "${var.name_prefix}-infrastructure"
}

# -----------------------------------------------------------------------------
# CloudWatch Log Groups
# -----------------------------------------------------------------------------

# Application Logs
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/${var.cluster_name}/application"
  retention_in_days = var.log_retention_days

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-application-logs"
  })
}

# API Gateway Logs
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/eks/${var.cluster_name}/api-gateway"
  retention_in_days = var.log_retention_days

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-api-gateway-logs"
  })
}

# Agent Runtime Logs
resource "aws_cloudwatch_log_group" "agent_runtime" {
  name              = "/aws/eks/${var.cluster_name}/agent-runtime"
  retention_in_days = var.log_retention_days

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-agent-runtime-logs"
  })
}

# -----------------------------------------------------------------------------
# SNS Topic for Alarms
# -----------------------------------------------------------------------------
resource "aws_sns_topic" "alerts" {
  count = var.create_sns_topic ? 1 : 0

  name = "${var.name_prefix}-infrastructure-alerts"

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-infrastructure-alerts"
  })
}

resource "aws_sns_topic_policy" "alerts" {
  count = var.create_sns_topic ? 1 : 0

  arn = aws_sns_topic.alerts[0].arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCloudWatchAlarms"
        Effect = "Allow"
        Principal = {
          Service = "cloudwatch.amazonaws.com"
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.alerts[0].arn
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms - EKS
# -----------------------------------------------------------------------------

# EKS Cluster CPU Utilization
resource "aws_cloudwatch_metric_alarm" "eks_cluster_cpu" {
  count = var.create_eks_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-eks-cluster-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "cluster_failed_node_count"
  namespace           = "ContainerInsights"
  period              = 300
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "EKS cluster has failed nodes"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    ClusterName = var.cluster_name
  }

  tags = var.tags
}

# EKS Node Group CPU Reservation
resource "aws_cloudwatch_metric_alarm" "eks_node_cpu_reservation" {
  count = var.create_eks_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-eks-node-cpu-reservation"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "node_cpu_reserved_capacity"
  namespace           = "ContainerInsights"
  period              = 300
  statistic           = "Average"
  threshold           = var.eks_cpu_threshold
  alarm_description   = "EKS nodes CPU reservation exceeds ${var.eks_cpu_threshold}%"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    ClusterName = var.cluster_name
  }

  tags = var.tags
}

# EKS Node Group Memory Reservation
resource "aws_cloudwatch_metric_alarm" "eks_node_memory_reservation" {
  count = var.create_eks_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-eks-node-memory-reservation"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "node_memory_reserved_capacity"
  namespace           = "ContainerInsights"
  period              = 300
  statistic           = "Average"
  threshold           = var.eks_memory_threshold
  alarm_description   = "EKS nodes memory reservation exceeds ${var.eks_memory_threshold}%"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    ClusterName = var.cluster_name
  }

  tags = var.tags
}

# EKS Pod Restart Count
resource "aws_cloudwatch_metric_alarm" "eks_pod_restarts" {
  count = var.create_eks_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-eks-pod-restarts"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "pod_number_of_container_restarts"
  namespace           = "ContainerInsights"
  period              = 300
  statistic           = "Sum"
  threshold           = var.pod_restart_threshold
  alarm_description   = "EKS pod restarts exceed ${var.pod_restart_threshold} in 5 minutes"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  dimensions = {
    ClusterName = var.cluster_name
  }

  tags = var.tags
}

# -----------------------------------------------------------------------------
# CloudWatch Dashboard
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_dashboard" "infrastructure" {
  dashboard_name = local.dashboard_name

  dashboard_body = jsonencode({
    widgets = concat(
      # EKS Widgets
      [
        {
          type   = "metric"
          x      = 0
          y      = 0
          width  = 12
          height = 6
          properties = {
            title  = "EKS Cluster CPU Utilization"
            region = var.aws_region
            metrics = [
              ["ContainerInsights", "node_cpu_utilization", "ClusterName", var.cluster_name, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        },
        {
          type   = "metric"
          x      = 12
          y      = 0
          width  = 12
          height = 6
          properties = {
            title  = "EKS Cluster Memory Utilization"
            region = var.aws_region
            metrics = [
              ["ContainerInsights", "node_memory_utilization", "ClusterName", var.cluster_name, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        },
        {
          type   = "metric"
          x      = 0
          y      = 6
          width  = 12
          height = 6
          properties = {
            title  = "EKS Pod Count"
            region = var.aws_region
            metrics = [
              ["ContainerInsights", "pod_number_of_running_pods", "ClusterName", var.cluster_name, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        },
        {
          type   = "metric"
          x      = 12
          y      = 6
          width  = 12
          height = 6
          properties = {
            title  = "EKS Network (Bytes)"
            region = var.aws_region
            metrics = [
              ["ContainerInsights", "node_network_total_bytes", "ClusterName", var.cluster_name, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        }
      ],
      # RDS Widgets (if RDS identifier is provided)
      var.rds_identifier != null ? [
        {
          type   = "metric"
          x      = 0
          y      = 12
          width  = 8
          height = 6
          properties = {
            title  = "RDS CPU Utilization"
            region = var.aws_region
            metrics = [
              ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", var.rds_identifier, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        },
        {
          type   = "metric"
          x      = 8
          y      = 12
          width  = 8
          height = 6
          properties = {
            title  = "RDS Database Connections"
            region = var.aws_region
            metrics = [
              ["AWS/RDS", "DatabaseConnections", "DBInstanceIdentifier", var.rds_identifier, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        },
        {
          type   = "metric"
          x      = 16
          y      = 12
          width  = 8
          height = 6
          properties = {
            title  = "RDS Free Storage"
            region = var.aws_region
            metrics = [
              ["AWS/RDS", "FreeStorageSpace", "DBInstanceIdentifier", var.rds_identifier, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        }
      ] : [],
      # ElastiCache Widgets (if cluster ID is provided)
      var.elasticache_cluster_id != null ? [
        {
          type   = "metric"
          x      = 0
          y      = 18
          width  = 8
          height = 6
          properties = {
            title  = "ElastiCache CPU Utilization"
            region = var.aws_region
            metrics = [
              ["AWS/ElastiCache", "CPUUtilization", "CacheClusterId", var.elasticache_cluster_id, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        },
        {
          type   = "metric"
          x      = 8
          y      = 18
          width  = 8
          height = 6
          properties = {
            title  = "ElastiCache Memory Usage"
            region = var.aws_region
            metrics = [
              ["AWS/ElastiCache", "DatabaseMemoryUsagePercentage", "CacheClusterId", var.elasticache_cluster_id, { stat = "Average" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        },
        {
          type   = "metric"
          x      = 16
          y      = 18
          width  = 8
          height = 6
          properties = {
            title  = "ElastiCache Cache Hits/Misses"
            region = var.aws_region
            metrics = [
              ["AWS/ElastiCache", "CacheHits", "CacheClusterId", var.elasticache_cluster_id, { stat = "Sum" }],
              ["AWS/ElastiCache", "CacheMisses", "CacheClusterId", var.elasticache_cluster_id, { stat = "Sum" }]
            ]
            period = 300
            view   = "timeSeries"
          }
        }
      ] : []
    )
  })
}

# -----------------------------------------------------------------------------
# CloudWatch Metric Filters for Application Logs
# -----------------------------------------------------------------------------

# Error Log Filter
resource "aws_cloudwatch_log_metric_filter" "application_errors" {
  name           = "${var.name_prefix}-application-errors"
  pattern        = "[timestamp, level=\"ERROR\", ...]"
  log_group_name = aws_cloudwatch_log_group.application.name

  metric_transformation {
    name          = "ApplicationErrors"
    namespace     = "${var.name_prefix}/Application"
    value         = "1"
    default_value = "0"
  }
}

# Warning Log Filter
resource "aws_cloudwatch_log_metric_filter" "application_warnings" {
  name           = "${var.name_prefix}-application-warnings"
  pattern        = "[timestamp, level=\"WARN*\", ...]"
  log_group_name = aws_cloudwatch_log_group.application.name

  metric_transformation {
    name          = "ApplicationWarnings"
    namespace     = "${var.name_prefix}/Application"
    value         = "1"
    default_value = "0"
  }
}

# Application Error Alarm
resource "aws_cloudwatch_metric_alarm" "application_errors" {
  count = var.create_application_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-application-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ApplicationErrors"
  namespace           = "${var.name_prefix}/Application"
  period              = 300
  statistic           = "Sum"
  threshold           = var.application_error_threshold
  alarm_description   = "Application errors exceed ${var.application_error_threshold} in 5 minutes"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions

  treat_missing_data = "notBreaching"

  tags = var.tags
}
