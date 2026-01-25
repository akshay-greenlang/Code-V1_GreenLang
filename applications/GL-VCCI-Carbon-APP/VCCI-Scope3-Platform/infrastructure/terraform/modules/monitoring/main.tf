# Monitoring Module - CloudWatch and SNS

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "eks" {
  count             = var.enable_cloudwatch_logs ? 1 : 0
  name              = "/aws/eks/${var.eks_cluster_name}/application"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "rds" {
  count             = var.enable_cloudwatch_logs ? 1 : 0
  name              = "/aws/rds/${var.rds_cluster_id}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

# SNS Topics for Alerts
resource "aws_sns_topic" "critical_alerts" {
  count = var.enable_cloudwatch_alarms ? 1 : 0
  name  = "${var.project_name}-${var.environment}-critical-alerts"
  tags  = var.tags
}

resource "aws_sns_topic" "warning_alerts" {
  count = var.enable_cloudwatch_alarms ? 1 : 0
  name  = "${var.project_name}-${var.environment}-warning-alerts"
  tags  = var.tags
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.enable_cloudwatch_alarms && length(var.alarm_email_endpoints) > 0 ? length(var.alarm_email_endpoints) : 0
  topic_arn = aws_sns_topic.critical_alerts[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email_endpoints[count.index]
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  count          = var.enable_cloudwatch_alarms ? 1 : 0
  dashboard_name = "${var.project_name}-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", { stat = "Average" }],
            ["AWS/ElastiCache", "CPUUtilization", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = data.aws_region.current.name
          title  = "CPU Utilization"
        }
      }
    ]
  })
}

data "aws_region" "current" {}
