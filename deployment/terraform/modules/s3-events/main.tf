#------------------------------------------------------------------------------
# S3 Event Notifications Module - Main Configuration
# GreenLang Infrastructure
#------------------------------------------------------------------------------

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
}

#------------------------------------------------------------------------------
# Data Sources
#------------------------------------------------------------------------------

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_iam_policy_document" "sns_policy" {
  statement {
    sid    = "AllowS3Publish"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["s3.amazonaws.com"]
    }

    actions   = ["sns:Publish"]
    resources = ["arn:aws:sns:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:greenlang-s3-*"]

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = var.source_bucket_arns
    }
  }
}

data "aws_iam_policy_document" "sqs_policy" {
  statement {
    sid    = "AllowSNSMessages"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["sns.amazonaws.com"]
    }

    actions   = ["sqs:SendMessage"]
    resources = ["arn:aws:sqs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:greenlang-*"]

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values = [
        aws_sns_topic.s3_uploads.arn,
        aws_sns_topic.s3_deletes.arn,
        aws_sns_topic.s3_lifecycle.arn,
        aws_sns_topic.s3_replication.arn
      ]
    }
  }
}

#------------------------------------------------------------------------------
# SNS Topics for S3 Events
#------------------------------------------------------------------------------

# SNS Topic for new object uploads
resource "aws_sns_topic" "s3_uploads" {
  name              = "greenlang-s3-uploads"
  display_name      = "GreenLang S3 Upload Notifications"
  kms_master_key_id = var.kms_key_id

  tags = merge(var.tags, {
    Name        = "greenlang-s3-uploads"
    EventType   = "upload"
    Environment = var.environment
  })
}

resource "aws_sns_topic_policy" "s3_uploads" {
  arn    = aws_sns_topic.s3_uploads.arn
  policy = data.aws_iam_policy_document.sns_policy.json
}

# SNS Topic for object deletions
resource "aws_sns_topic" "s3_deletes" {
  name              = "greenlang-s3-deletes"
  display_name      = "GreenLang S3 Delete Notifications"
  kms_master_key_id = var.kms_key_id

  tags = merge(var.tags, {
    Name        = "greenlang-s3-deletes"
    EventType   = "delete"
    Environment = var.environment
  })
}

resource "aws_sns_topic_policy" "s3_deletes" {
  arn    = aws_sns_topic.s3_deletes.arn
  policy = data.aws_iam_policy_document.sns_policy.json
}

# SNS Topic for lifecycle transitions
resource "aws_sns_topic" "s3_lifecycle" {
  name              = "greenlang-s3-lifecycle"
  display_name      = "GreenLang S3 Lifecycle Notifications"
  kms_master_key_id = var.kms_key_id

  tags = merge(var.tags, {
    Name        = "greenlang-s3-lifecycle"
    EventType   = "lifecycle"
    Environment = var.environment
  })
}

resource "aws_sns_topic_policy" "s3_lifecycle" {
  arn    = aws_sns_topic.s3_lifecycle.arn
  policy = data.aws_iam_policy_document.sns_policy.json
}

# SNS Topic for replication status
resource "aws_sns_topic" "s3_replication" {
  name              = "greenlang-s3-replication"
  display_name      = "GreenLang S3 Replication Notifications"
  kms_master_key_id = var.kms_key_id

  tags = merge(var.tags, {
    Name        = "greenlang-s3-replication"
    EventType   = "replication"
    Environment = var.environment
  })
}

resource "aws_sns_topic_policy" "s3_replication" {
  arn    = aws_sns_topic.s3_replication.arn
  policy = data.aws_iam_policy_document.sns_policy.json
}

#------------------------------------------------------------------------------
# SQS Queues for Event Processing
#------------------------------------------------------------------------------

# Dead Letter Queue for failed artifact processing
resource "aws_sqs_queue" "artifact_processing_dlq" {
  name                      = "greenlang-artifact-processing-dlq"
  message_retention_seconds = 1209600  # 14 days
  kms_master_key_id         = var.kms_key_id

  tags = merge(var.tags, {
    Name        = "greenlang-artifact-processing-dlq"
    QueueType   = "dlq"
    Environment = var.environment
  })
}

# SQS Queue for artifact validation
resource "aws_sqs_queue" "artifact_processing" {
  name                       = "greenlang-artifact-processing"
  delay_seconds              = 0
  max_message_size           = 262144
  message_retention_seconds  = 345600  # 4 days
  receive_wait_time_seconds  = 20
  visibility_timeout_seconds = 300
  kms_master_key_id          = var.kms_key_id

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.artifact_processing_dlq.arn
    maxReceiveCount     = 3
  })

  tags = merge(var.tags, {
    Name        = "greenlang-artifact-processing"
    Purpose     = "artifact-validation"
    Environment = var.environment
  })
}

resource "aws_sqs_queue_policy" "artifact_processing" {
  queue_url = aws_sqs_queue.artifact_processing.id
  policy    = data.aws_iam_policy_document.sqs_policy.json
}

# Dead Letter Queue for failed report indexing
resource "aws_sqs_queue" "report_indexing_dlq" {
  name                      = "greenlang-report-indexing-dlq"
  message_retention_seconds = 1209600  # 14 days
  kms_master_key_id         = var.kms_key_id

  tags = merge(var.tags, {
    Name        = "greenlang-report-indexing-dlq"
    QueueType   = "dlq"
    Environment = var.environment
  })
}

# SQS Queue for report search indexing
resource "aws_sqs_queue" "report_indexing" {
  name                       = "greenlang-report-indexing"
  delay_seconds              = 0
  max_message_size           = 262144
  message_retention_seconds  = 345600  # 4 days
  receive_wait_time_seconds  = 20
  visibility_timeout_seconds = 600
  kms_master_key_id          = var.kms_key_id

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.report_indexing_dlq.arn
    maxReceiveCount     = 3
  })

  tags = merge(var.tags, {
    Name        = "greenlang-report-indexing"
    Purpose     = "search-indexing"
    Environment = var.environment
  })
}

resource "aws_sqs_queue_policy" "report_indexing" {
  queue_url = aws_sqs_queue.report_indexing.id
  policy    = data.aws_iam_policy_document.sqs_policy.json
}

# Dead Letter Queue for audit events
resource "aws_sqs_queue" "audit_events_dlq" {
  name                      = "greenlang-audit-events-dlq"
  message_retention_seconds = 1209600  # 14 days
  kms_master_key_id         = var.kms_key_id

  tags = merge(var.tags, {
    Name        = "greenlang-audit-events-dlq"
    QueueType   = "dlq"
    Environment = var.environment
  })
}

# SQS Queue for compliance tracking
resource "aws_sqs_queue" "audit_events" {
  name                       = "greenlang-audit-events"
  delay_seconds              = 0
  max_message_size           = 262144
  message_retention_seconds  = 1209600  # 14 days for compliance
  receive_wait_time_seconds  = 20
  visibility_timeout_seconds = 180
  kms_master_key_id          = var.kms_key_id

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.audit_events_dlq.arn
    maxReceiveCount     = 5
  })

  tags = merge(var.tags, {
    Name        = "greenlang-audit-events"
    Purpose     = "compliance-tracking"
    Compliance  = "required"
    Environment = var.environment
  })
}

resource "aws_sqs_queue_policy" "audit_events" {
  queue_url = aws_sqs_queue.audit_events.id
  policy    = data.aws_iam_policy_document.sqs_policy.json
}

#------------------------------------------------------------------------------
# SNS to SQS Subscriptions - Fanout Pattern
#------------------------------------------------------------------------------

# Upload events fanout to artifact processing queue
resource "aws_sns_topic_subscription" "uploads_to_artifact" {
  topic_arn            = aws_sns_topic.s3_uploads.arn
  protocol             = "sqs"
  endpoint             = aws_sqs_queue.artifact_processing.arn
  raw_message_delivery = true

  filter_policy = jsonencode({
    eventName = [{ "prefix" = "ObjectCreated:" }]
    s3 = {
      object = {
        key = [
          { "prefix" = "artifacts/" },
          { "prefix" = "uploads/" }
        ]
      }
    }
  })
}

# Upload events fanout to report indexing queue
resource "aws_sns_topic_subscription" "uploads_to_indexing" {
  topic_arn            = aws_sns_topic.s3_uploads.arn
  protocol             = "sqs"
  endpoint             = aws_sqs_queue.report_indexing.arn
  raw_message_delivery = true

  filter_policy = jsonencode({
    eventName = [{ "prefix" = "ObjectCreated:" }]
    s3 = {
      object = {
        key = [
          { "prefix" = "reports/" },
          { "suffix" = ".pdf" },
          { "suffix" = ".html" },
          { "suffix" = ".xlsx" }
        ]
      }
    }
  })
}

# Upload events fanout to audit queue
resource "aws_sns_topic_subscription" "uploads_to_audit" {
  topic_arn            = aws_sns_topic.s3_uploads.arn
  protocol             = "sqs"
  endpoint             = aws_sqs_queue.audit_events.arn
  raw_message_delivery = true
}

# Delete events to audit queue only
resource "aws_sns_topic_subscription" "deletes_to_audit" {
  topic_arn            = aws_sns_topic.s3_deletes.arn
  protocol             = "sqs"
  endpoint             = aws_sqs_queue.audit_events.arn
  raw_message_delivery = true
}

# Lifecycle events to audit queue
resource "aws_sns_topic_subscription" "lifecycle_to_audit" {
  topic_arn            = aws_sns_topic.s3_lifecycle.arn
  protocol             = "sqs"
  endpoint             = aws_sqs_queue.audit_events.arn
  raw_message_delivery = true
}

# Replication events to audit queue
resource "aws_sns_topic_subscription" "replication_to_audit" {
  topic_arn            = aws_sns_topic.s3_replication.arn
  protocol             = "sqs"
  endpoint             = aws_sqs_queue.audit_events.arn
  raw_message_delivery = true
}

#------------------------------------------------------------------------------
# CloudWatch Log Groups for Event Processing
#------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "s3_lifecycle_events" {
  name              = "/aws/s3/greenlang/lifecycle-events"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn

  tags = merge(var.tags, {
    Name        = "greenlang-s3-lifecycle-logs"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_log_group" "s3_replication_events" {
  name              = "/aws/s3/greenlang/replication-events"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn

  tags = merge(var.tags, {
    Name        = "greenlang-s3-replication-logs"
    Environment = var.environment
  })
}

#------------------------------------------------------------------------------
# S3 Bucket Notification Configuration
#------------------------------------------------------------------------------

resource "aws_s3_bucket_notification" "bucket_notifications" {
  for_each = var.notification_buckets

  bucket = each.value.bucket_id

  # PUT events to SNS upload topic
  dynamic "topic" {
    for_each = each.value.enable_upload_notifications ? [1] : []
    content {
      topic_arn     = aws_sns_topic.s3_uploads.arn
      events        = ["s3:ObjectCreated:*"]
      filter_prefix = each.value.upload_prefix
      filter_suffix = each.value.upload_suffix
    }
  }

  # DELETE events to SNS delete topic
  dynamic "topic" {
    for_each = each.value.enable_delete_notifications ? [1] : []
    content {
      topic_arn = aws_sns_topic.s3_deletes.arn
      events = [
        "s3:ObjectRemoved:*",
        "s3:ObjectRemoved:Delete",
        "s3:ObjectRemoved:DeleteMarkerCreated"
      ]
    }
  }

  # Lifecycle events
  dynamic "topic" {
    for_each = each.value.enable_lifecycle_notifications ? [1] : []
    content {
      topic_arn = aws_sns_topic.s3_lifecycle.arn
      events = [
        "s3:LifecycleExpiration:*",
        "s3:LifecycleTransition"
      ]
    }
  }

  # Replication events
  dynamic "topic" {
    for_each = each.value.enable_replication_notifications ? [1] : []
    content {
      topic_arn = aws_sns_topic.s3_replication.arn
      events = [
        "s3:Replication:*",
        "s3:Replication:OperationFailedReplication",
        "s3:Replication:OperationNotTracked",
        "s3:Replication:OperationMissedThreshold"
      ]
    }
  }

  # Direct Lambda invocation for artifact validation
  dynamic "lambda_function" {
    for_each = each.value.enable_artifact_validation ? [1] : []
    content {
      lambda_function_arn = aws_lambda_function.artifact_validator.arn
      events              = ["s3:ObjectCreated:*"]
      filter_prefix       = "artifacts/"
    }
  }

  # Direct Lambda invocation for report indexing
  dynamic "lambda_function" {
    for_each = each.value.enable_report_indexing ? [1] : []
    content {
      lambda_function_arn = aws_lambda_function.report_indexer.arn
      events              = ["s3:ObjectCreated:*"]
      filter_prefix       = "reports/"
    }
  }

  depends_on = [
    aws_sns_topic_policy.s3_uploads,
    aws_sns_topic_policy.s3_deletes,
    aws_sns_topic_policy.s3_lifecycle,
    aws_sns_topic_policy.s3_replication,
    aws_lambda_permission.artifact_validator,
    aws_lambda_permission.report_indexer
  ]
}

#------------------------------------------------------------------------------
# CloudWatch Metrics and Alarms
#------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "artifact_processing_dlq_depth" {
  alarm_name          = "greenlang-artifact-processing-dlq-depth"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Artifact processing DLQ has messages - investigate failures"
  alarm_actions       = var.alarm_sns_topic_arns
  ok_actions          = var.alarm_sns_topic_arns

  dimensions = {
    QueueName = aws_sqs_queue.artifact_processing_dlq.name
  }

  tags = merge(var.tags, {
    Name        = "greenlang-artifact-dlq-alarm"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_metric_alarm" "report_indexing_dlq_depth" {
  alarm_name          = "greenlang-report-indexing-dlq-depth"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Report indexing DLQ has messages - investigate failures"
  alarm_actions       = var.alarm_sns_topic_arns
  ok_actions          = var.alarm_sns_topic_arns

  dimensions = {
    QueueName = aws_sqs_queue.report_indexing_dlq.name
  }

  tags = merge(var.tags, {
    Name        = "greenlang-indexing-dlq-alarm"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_metric_alarm" "audit_events_dlq_depth" {
  alarm_name          = "greenlang-audit-events-dlq-depth"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Audit events DLQ has messages - CRITICAL for compliance"
  alarm_actions       = var.alarm_sns_topic_arns
  ok_actions          = var.alarm_sns_topic_arns
  treat_missing_data  = "notBreaching"

  dimensions = {
    QueueName = aws_sqs_queue.audit_events_dlq.name
  }

  tags = merge(var.tags, {
    Name        = "greenlang-audit-dlq-alarm"
    Compliance  = "required"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_metric_alarm" "artifact_queue_age" {
  alarm_name          = "greenlang-artifact-processing-message-age"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ApproximateAgeOfOldestMessage"
  namespace           = "AWS/SQS"
  period              = 300
  statistic           = "Maximum"
  threshold           = 3600  # 1 hour
  alarm_description   = "Artifact processing queue has old messages - processing may be stuck"
  alarm_actions       = var.alarm_sns_topic_arns

  dimensions = {
    QueueName = aws_sqs_queue.artifact_processing.name
  }

  tags = merge(var.tags, {
    Name        = "greenlang-artifact-age-alarm"
    Environment = var.environment
  })
}

#------------------------------------------------------------------------------
# Event Bridge Rules for Additional Event Processing
#------------------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "s3_events" {
  name        = "greenlang-s3-events-rule"
  description = "Capture S3 events from CloudTrail for additional processing"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["AWS API Call via CloudTrail"]
    detail = {
      eventSource = ["s3.amazonaws.com"]
      eventName = [
        "PutObject",
        "DeleteObject",
        "CopyObject",
        "RestoreObject"
      ]
      requestParameters = {
        bucketName = var.monitored_bucket_names
      }
    }
  })

  tags = merge(var.tags, {
    Name        = "greenlang-s3-events-rule"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_event_target" "s3_events_to_audit" {
  rule      = aws_cloudwatch_event_rule.s3_events.name
  target_id = "send-to-audit-lambda"
  arn       = aws_lambda_function.audit_logger.arn

  retry_policy {
    maximum_event_age_in_seconds = 3600
    maximum_retry_attempts       = 3
  }

  dead_letter_config {
    arn = aws_sqs_queue.audit_events_dlq.arn
  }
}

resource "aws_lambda_permission" "eventbridge_audit" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.audit_logger.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.s3_events.arn
}
