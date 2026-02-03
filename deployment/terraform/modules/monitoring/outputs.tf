# GreenLang Monitoring Module - Outputs

# Log Groups
output "application_log_group_name" {
  description = "Application log group name"
  value       = aws_cloudwatch_log_group.application.name
}

output "application_log_group_arn" {
  description = "Application log group ARN"
  value       = aws_cloudwatch_log_group.application.arn
}

output "api_gateway_log_group_name" {
  description = "API Gateway log group name"
  value       = aws_cloudwatch_log_group.api_gateway.name
}

output "api_gateway_log_group_arn" {
  description = "API Gateway log group ARN"
  value       = aws_cloudwatch_log_group.api_gateway.arn
}

output "agent_runtime_log_group_name" {
  description = "Agent runtime log group name"
  value       = aws_cloudwatch_log_group.agent_runtime.name
}

output "agent_runtime_log_group_arn" {
  description = "Agent runtime log group ARN"
  value       = aws_cloudwatch_log_group.agent_runtime.arn
}

# SNS Topic
output "alerts_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = var.create_sns_topic ? aws_sns_topic.alerts[0].arn : null
}

output "alerts_topic_name" {
  description = "SNS topic name for alerts"
  value       = var.create_sns_topic ? aws_sns_topic.alerts[0].name : null
}

# Dashboard
output "dashboard_name" {
  description = "CloudWatch dashboard name"
  value       = aws_cloudwatch_dashboard.infrastructure.dashboard_name
}

output "dashboard_arn" {
  description = "CloudWatch dashboard ARN"
  value       = aws_cloudwatch_dashboard.infrastructure.dashboard_arn
}

# Metric Namespace
output "application_metric_namespace" {
  description = "CloudWatch metric namespace for application metrics"
  value       = "${var.name_prefix}/Application"
}
