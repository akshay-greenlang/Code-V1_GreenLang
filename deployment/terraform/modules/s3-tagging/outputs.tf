# Outputs for S3 Object Tagging Module

# -----------------------------------------------------------------------------
# Lambda Function Outputs
# -----------------------------------------------------------------------------

output "auto_tagger_lambda_arn" {
  description = "ARN of the auto-tagger Lambda function"
  value       = aws_lambda_function.auto_tagger.arn
}

output "auto_tagger_lambda_name" {
  description = "Name of the auto-tagger Lambda function"
  value       = aws_lambda_function.auto_tagger.function_name
}

output "auto_tagger_lambda_invoke_arn" {
  description = "Invoke ARN of the auto-tagger Lambda function"
  value       = aws_lambda_function.auto_tagger.invoke_arn
}

output "auto_tagger_lambda_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = aws_iam_role.lambda_role.arn
}

output "tag_enforcement_lambda_arn" {
  description = "ARN of the tag enforcement Lambda function"
  value       = var.enable_tag_enforcement ? aws_lambda_function.tag_enforcement[0].arn : null
}

output "tag_enforcement_lambda_name" {
  description = "Name of the tag enforcement Lambda function"
  value       = var.enable_tag_enforcement ? aws_lambda_function.tag_enforcement[0].function_name : null
}

# -----------------------------------------------------------------------------
# EventBridge Outputs
# -----------------------------------------------------------------------------

output "eventbridge_rule_arn" {
  description = "ARN of the EventBridge rule for S3 object creation"
  value       = aws_cloudwatch_event_rule.s3_object_created.arn
}

output "eventbridge_rule_name" {
  description = "Name of the EventBridge rule for S3 object creation"
  value       = aws_cloudwatch_event_rule.s3_object_created.name
}

output "enforcement_schedule_rule_arn" {
  description = "ARN of the EventBridge rule for scheduled tag enforcement"
  value       = var.enable_tag_enforcement ? aws_cloudwatch_event_rule.tag_enforcement_schedule[0].arn : null
}

# -----------------------------------------------------------------------------
# S3 Batch Operations Outputs
# -----------------------------------------------------------------------------

output "batch_operations_role_arn" {
  description = "ARN of the S3 Batch Operations IAM role"
  value       = aws_iam_role.batch_operations_role.arn
}

output "batch_operations_role_name" {
  description = "Name of the S3 Batch Operations IAM role"
  value       = aws_iam_role.batch_operations_role.name
}

# -----------------------------------------------------------------------------
# Configuration Outputs
# -----------------------------------------------------------------------------

output "monitored_buckets" {
  description = "List of monitored S3 buckets"
  value       = var.monitored_bucket_names
}

output "tag_schemas" {
  description = "Tag schemas per artifact type"
  value       = var.tag_schemas
}

output "required_tags" {
  description = "List of required tags"
  value       = var.required_tags
}

output "default_tags" {
  description = "Default tags applied to all objects"
  value       = var.default_tags
}

output "data_classification_levels" {
  description = "Data classification level definitions"
  value       = var.data_classification_levels
}

output "retention_policy_definitions" {
  description = "Retention policy definitions"
  value       = var.retention_policy_definitions
}

# -----------------------------------------------------------------------------
# CloudWatch Alarm Outputs
# -----------------------------------------------------------------------------

output "lambda_error_alarm_arn" {
  description = "ARN of the Lambda errors CloudWatch alarm"
  value       = var.enable_alarms ? aws_cloudwatch_metric_alarm.lambda_errors[0].arn : null
}

output "lambda_throttle_alarm_arn" {
  description = "ARN of the Lambda throttles CloudWatch alarm"
  value       = var.enable_alarms ? aws_cloudwatch_metric_alarm.lambda_throttles[0].arn : null
}

# -----------------------------------------------------------------------------
# Integration Outputs
# -----------------------------------------------------------------------------

output "integration_summary" {
  description = "Summary of integration points for other modules"
  value = {
    auto_tagger = {
      lambda_arn       = aws_lambda_function.auto_tagger.arn
      lambda_name      = aws_lambda_function.auto_tagger.function_name
      eventbridge_rule = aws_cloudwatch_event_rule.s3_object_created.arn
    }
    batch_operations = {
      role_arn = aws_iam_role.batch_operations_role.arn
    }
    enforcement = var.enable_tag_enforcement ? {
      lambda_arn     = aws_lambda_function.tag_enforcement[0].arn
      schedule_rule  = aws_cloudwatch_event_rule.tag_enforcement_schedule[0].arn
      schedule_expr  = var.enforcement_schedule
    } : null
    monitoring = var.enable_alarms ? {
      error_alarm    = aws_cloudwatch_metric_alarm.lambda_errors[0].arn
      throttle_alarm = aws_cloudwatch_metric_alarm.lambda_throttles[0].arn
    } : null
  }
}

# -----------------------------------------------------------------------------
# Artifact Type Mapping Output
# -----------------------------------------------------------------------------

output "artifact_type_patterns" {
  description = "Regex patterns for artifact type detection"
  value = {
    BUILD_ARTIFACTS     = ["^builds/", "^packages/", "^binaries/"]
    CALCULATION_RESULTS = ["^calculations/", "^emissions/", "^analysis/"]
    REPORTS             = ["^reports/"]
    AUDIT_LOGS          = ["^audit/", "^logs/audit/"]
    ML_MODELS           = ["^models/", "^ml/"]
    TEMPORARY           = ["^tmp/", "^temp/", "^scratch/"]
  }
}
