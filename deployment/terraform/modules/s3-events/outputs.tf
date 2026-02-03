#------------------------------------------------------------------------------
# S3 Event Notifications Module - Outputs
# GreenLang Infrastructure
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SNS Topic Outputs
#------------------------------------------------------------------------------

output "sns_topic_uploads_arn" {
  description = "ARN of the SNS topic for S3 upload notifications"
  value       = aws_sns_topic.s3_uploads.arn
}

output "sns_topic_uploads_name" {
  description = "Name of the SNS topic for S3 upload notifications"
  value       = aws_sns_topic.s3_uploads.name
}

output "sns_topic_deletes_arn" {
  description = "ARN of the SNS topic for S3 delete notifications"
  value       = aws_sns_topic.s3_deletes.arn
}

output "sns_topic_deletes_name" {
  description = "Name of the SNS topic for S3 delete notifications"
  value       = aws_sns_topic.s3_deletes.name
}

output "sns_topic_lifecycle_arn" {
  description = "ARN of the SNS topic for S3 lifecycle notifications"
  value       = aws_sns_topic.s3_lifecycle.arn
}

output "sns_topic_lifecycle_name" {
  description = "Name of the SNS topic for S3 lifecycle notifications"
  value       = aws_sns_topic.s3_lifecycle.name
}

output "sns_topic_replication_arn" {
  description = "ARN of the SNS topic for S3 replication notifications"
  value       = aws_sns_topic.s3_replication.arn
}

output "sns_topic_replication_name" {
  description = "Name of the SNS topic for S3 replication notifications"
  value       = aws_sns_topic.s3_replication.name
}

output "sns_topics" {
  description = "Map of all SNS topics with their ARNs and names"
  value = {
    uploads = {
      arn  = aws_sns_topic.s3_uploads.arn
      name = aws_sns_topic.s3_uploads.name
    }
    deletes = {
      arn  = aws_sns_topic.s3_deletes.arn
      name = aws_sns_topic.s3_deletes.name
    }
    lifecycle = {
      arn  = aws_sns_topic.s3_lifecycle.arn
      name = aws_sns_topic.s3_lifecycle.name
    }
    replication = {
      arn  = aws_sns_topic.s3_replication.arn
      name = aws_sns_topic.s3_replication.name
    }
  }
}

#------------------------------------------------------------------------------
# SQS Queue Outputs
#------------------------------------------------------------------------------

output "sqs_queue_artifact_processing_url" {
  description = "URL of the SQS queue for artifact processing"
  value       = aws_sqs_queue.artifact_processing.url
}

output "sqs_queue_artifact_processing_arn" {
  description = "ARN of the SQS queue for artifact processing"
  value       = aws_sqs_queue.artifact_processing.arn
}

output "sqs_queue_report_indexing_url" {
  description = "URL of the SQS queue for report indexing"
  value       = aws_sqs_queue.report_indexing.url
}

output "sqs_queue_report_indexing_arn" {
  description = "ARN of the SQS queue for report indexing"
  value       = aws_sqs_queue.report_indexing.arn
}

output "sqs_queue_audit_events_url" {
  description = "URL of the SQS queue for audit events"
  value       = aws_sqs_queue.audit_events.url
}

output "sqs_queue_audit_events_arn" {
  description = "ARN of the SQS queue for audit events"
  value       = aws_sqs_queue.audit_events.arn
}

output "sqs_queues" {
  description = "Map of all SQS queues with their URLs and ARNs"
  value = {
    artifact_processing = {
      url  = aws_sqs_queue.artifact_processing.url
      arn  = aws_sqs_queue.artifact_processing.arn
      name = aws_sqs_queue.artifact_processing.name
    }
    report_indexing = {
      url  = aws_sqs_queue.report_indexing.url
      arn  = aws_sqs_queue.report_indexing.arn
      name = aws_sqs_queue.report_indexing.name
    }
    audit_events = {
      url  = aws_sqs_queue.audit_events.url
      arn  = aws_sqs_queue.audit_events.arn
      name = aws_sqs_queue.audit_events.name
    }
  }
}

output "sqs_dlq_urls" {
  description = "Map of all DLQ URLs"
  value = {
    artifact_processing = aws_sqs_queue.artifact_processing_dlq.url
    report_indexing     = aws_sqs_queue.report_indexing_dlq.url
    audit_events        = aws_sqs_queue.audit_events_dlq.url
  }
}

#------------------------------------------------------------------------------
# Lambda Function Outputs
#------------------------------------------------------------------------------

output "lambda_artifact_validator_arn" {
  description = "ARN of the artifact validator Lambda function"
  value       = aws_lambda_function.artifact_validator.arn
}

output "lambda_artifact_validator_name" {
  description = "Name of the artifact validator Lambda function"
  value       = aws_lambda_function.artifact_validator.function_name
}

output "lambda_report_indexer_arn" {
  description = "ARN of the report indexer Lambda function"
  value       = aws_lambda_function.report_indexer.arn
}

output "lambda_report_indexer_name" {
  description = "Name of the report indexer Lambda function"
  value       = aws_lambda_function.report_indexer.function_name
}

output "lambda_audit_logger_arn" {
  description = "ARN of the audit logger Lambda function"
  value       = aws_lambda_function.audit_logger.arn
}

output "lambda_audit_logger_name" {
  description = "Name of the audit logger Lambda function"
  value       = aws_lambda_function.audit_logger.function_name
}

output "lambda_cost_tracker_arn" {
  description = "ARN of the cost tracker Lambda function"
  value       = aws_lambda_function.cost_tracker.arn
}

output "lambda_cost_tracker_name" {
  description = "Name of the cost tracker Lambda function"
  value       = aws_lambda_function.cost_tracker.function_name
}

output "lambda_functions" {
  description = "Map of all Lambda functions with their ARNs and names"
  value = {
    artifact_validator = {
      arn           = aws_lambda_function.artifact_validator.arn
      name          = aws_lambda_function.artifact_validator.function_name
      invoke_arn    = aws_lambda_function.artifact_validator.invoke_arn
      qualified_arn = aws_lambda_function.artifact_validator.qualified_arn
    }
    report_indexer = {
      arn           = aws_lambda_function.report_indexer.arn
      name          = aws_lambda_function.report_indexer.function_name
      invoke_arn    = aws_lambda_function.report_indexer.invoke_arn
      qualified_arn = aws_lambda_function.report_indexer.qualified_arn
    }
    audit_logger = {
      arn           = aws_lambda_function.audit_logger.arn
      name          = aws_lambda_function.audit_logger.function_name
      invoke_arn    = aws_lambda_function.audit_logger.invoke_arn
      qualified_arn = aws_lambda_function.audit_logger.qualified_arn
    }
    cost_tracker = {
      arn           = aws_lambda_function.cost_tracker.arn
      name          = aws_lambda_function.cost_tracker.function_name
      invoke_arn    = aws_lambda_function.cost_tracker.invoke_arn
      qualified_arn = aws_lambda_function.cost_tracker.qualified_arn
    }
  }
}

#------------------------------------------------------------------------------
# Event Configuration Outputs
#------------------------------------------------------------------------------

output "bucket_notification_ids" {
  description = "Map of bucket notification configuration IDs"
  value       = { for k, v in aws_s3_bucket_notification.bucket_notifications : k => v.id }
}

output "eventbridge_rule_arn" {
  description = "ARN of the EventBridge rule for S3 events"
  value       = aws_cloudwatch_event_rule.s3_events.arn
}

output "eventbridge_rule_name" {
  description = "Name of the EventBridge rule for S3 events"
  value       = aws_cloudwatch_event_rule.s3_events.name
}

#------------------------------------------------------------------------------
# CloudWatch Outputs
#------------------------------------------------------------------------------

output "cloudwatch_log_groups" {
  description = "Map of CloudWatch log group names"
  value = {
    lifecycle_events   = aws_cloudwatch_log_group.s3_lifecycle_events.name
    replication_events = aws_cloudwatch_log_group.s3_replication_events.name
    artifact_validator = aws_cloudwatch_log_group.artifact_validator.name
    report_indexer     = aws_cloudwatch_log_group.report_indexer.name
    audit_logger       = aws_cloudwatch_log_group.audit_logger.name
    cost_tracker       = aws_cloudwatch_log_group.cost_tracker.name
  }
}

output "cloudwatch_alarm_arns" {
  description = "Map of CloudWatch alarm ARNs"
  value = {
    artifact_dlq_depth  = aws_cloudwatch_metric_alarm.artifact_processing_dlq_depth.arn
    indexing_dlq_depth  = aws_cloudwatch_metric_alarm.report_indexing_dlq_depth.arn
    audit_dlq_depth     = aws_cloudwatch_metric_alarm.audit_events_dlq_depth.arn
    artifact_queue_age  = aws_cloudwatch_metric_alarm.artifact_queue_age.arn
  }
}

#------------------------------------------------------------------------------
# IAM Role Outputs
#------------------------------------------------------------------------------

output "lambda_execution_role_arns" {
  description = "Map of Lambda execution role ARNs"
  value = {
    artifact_validator = aws_iam_role.artifact_validator_role.arn
    report_indexer     = aws_iam_role.report_indexer_role.arn
    audit_logger       = aws_iam_role.audit_logger_role.arn
    cost_tracker       = aws_iam_role.cost_tracker_role.arn
  }
}
