#------------------------------------------------------------------------------
# S3 Event Notifications Module - Lambda Functions
# GreenLang Infrastructure
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Lambda Deployment Packages
#------------------------------------------------------------------------------

data "archive_file" "artifact_validator" {
  type        = "zip"
  source_dir  = "${var.lambda_source_path}/artifact-validator"
  output_path = "${path.module}/builds/artifact-validator.zip"
}

data "archive_file" "report_indexer" {
  type        = "zip"
  source_dir  = "${var.lambda_source_path}/report-indexer"
  output_path = "${path.module}/builds/report-indexer.zip"
}

data "archive_file" "audit_logger" {
  type        = "zip"
  source_dir  = "${var.lambda_source_path}/audit-logger"
  output_path = "${path.module}/builds/audit-logger.zip"
}

data "archive_file" "cost_tracker" {
  type        = "zip"
  source_dir  = "${var.lambda_source_path}/cost-tracker"
  output_path = "${path.module}/builds/cost-tracker.zip"
}

#------------------------------------------------------------------------------
# CloudWatch Log Groups for Lambda
#------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "artifact_validator" {
  name              = "/aws/lambda/greenlang-artifact-validator"
  retention_in_days = var.lambda_log_retention_days
  kms_key_id        = var.kms_key_arn

  tags = merge(var.tags, {
    Name        = "greenlang-artifact-validator-logs"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_log_group" "report_indexer" {
  name              = "/aws/lambda/greenlang-report-indexer"
  retention_in_days = var.lambda_log_retention_days
  kms_key_id        = var.kms_key_arn

  tags = merge(var.tags, {
    Name        = "greenlang-report-indexer-logs"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_log_group" "audit_logger" {
  name              = "/aws/lambda/greenlang-audit-logger"
  retention_in_days = var.lambda_log_retention_days
  kms_key_id        = var.kms_key_arn

  tags = merge(var.tags, {
    Name        = "greenlang-audit-logger-logs"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_log_group" "cost_tracker" {
  name              = "/aws/lambda/greenlang-cost-tracker"
  retention_in_days = var.lambda_log_retention_days
  kms_key_id        = var.kms_key_arn

  tags = merge(var.tags, {
    Name        = "greenlang-cost-tracker-logs"
    Environment = var.environment
  })
}

#------------------------------------------------------------------------------
# IAM Roles for Lambda Functions
#------------------------------------------------------------------------------

# Artifact Validator Role
resource "aws_iam_role" "artifact_validator_role" {
  name = "greenlang-artifact-validator-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = merge(var.tags, {
    Name        = "greenlang-artifact-validator-role"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "artifact_validator_policy" {
  name = "greenlang-artifact-validator-policy"
  role = aws_iam_role.artifact_validator_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.artifact_validator.arn}:*"
      },
      {
        Sid    = "S3ReadAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging",
          "s3:HeadObject"
        ]
        Resource = [for arn in var.source_bucket_arns : "${arn}/*"]
      },
      {
        Sid    = "S3WriteAccess"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectTagging",
          "s3:DeleteObject",
          "s3:CopyObject"
        ]
        Resource = [for arn in var.source_bucket_arns : "${arn}/*"]
      },
      {
        Sid    = "SQSAccess"
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:SendMessage"
        ]
        Resource = [
          aws_sqs_queue.artifact_processing.arn,
          aws_sqs_queue.artifact_processing_dlq.arn
        ]
      },
      {
        Sid    = "KMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = var.kms_key_arn != null ? [var.kms_key_arn] : ["*"]
      }
    ]
  })
}

# Report Indexer Role
resource "aws_iam_role" "report_indexer_role" {
  name = "greenlang-report-indexer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = merge(var.tags, {
    Name        = "greenlang-report-indexer-role"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "report_indexer_policy" {
  name = "greenlang-report-indexer-policy"
  role = aws_iam_role.report_indexer_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.report_indexer.arn}:*"
      },
      {
        Sid    = "S3ReadAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:HeadObject"
        ]
        Resource = [for arn in var.source_bucket_arns : "${arn}/*"]
      },
      {
        Sid    = "SQSAccess"
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:SendMessage"
        ]
        Resource = [
          aws_sqs_queue.report_indexing.arn,
          aws_sqs_queue.report_indexing_dlq.arn
        ]
      },
      {
        Sid    = "OpenSearchAccess"
        Effect = "Allow"
        Action = [
          "es:ESHttpPost",
          "es:ESHttpPut",
          "es:ESHttpGet"
        ]
        Resource = var.opensearch_config.enabled ? ["${var.opensearch_config.endpoint}/*"] : ["*"]
      },
      {
        Sid    = "KMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = var.kms_key_arn != null ? [var.kms_key_arn] : ["*"]
      }
    ]
  })
}

# Audit Logger Role
resource "aws_iam_role" "audit_logger_role" {
  name = "greenlang-audit-logger-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = merge(var.tags, {
    Name        = "greenlang-audit-logger-role"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "audit_logger_policy" {
  name = "greenlang-audit-logger-policy"
  role = aws_iam_role.audit_logger_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.audit_logger.arn}:*"
      },
      {
        Sid    = "S3AuditWrite"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectTagging"
        ]
        Resource = var.audit_logger_config.audit_bucket != "" ? ["arn:aws:s3:::${var.audit_logger_config.audit_bucket}/*"] : ["*"]
      },
      {
        Sid    = "SQSAccess"
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:SendMessage"
        ]
        Resource = [
          aws_sqs_queue.audit_events.arn,
          aws_sqs_queue.audit_events_dlq.arn
        ]
      },
      {
        Sid    = "STSGetCallerIdentity"
        Effect = "Allow"
        Action = [
          "sts:GetCallerIdentity"
        ]
        Resource = "*"
      },
      {
        Sid    = "KMSEncrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = var.kms_key_arn != null ? [var.kms_key_arn] : ["*"]
      }
    ]
  })
}

# Cost Tracker Role
resource "aws_iam_role" "cost_tracker_role" {
  name = "greenlang-cost-tracker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = merge(var.tags, {
    Name        = "greenlang-cost-tracker-role"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "cost_tracker_policy" {
  name = "greenlang-cost-tracker-policy"
  role = aws_iam_role.cost_tracker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.cost_tracker.arn}:*"
      },
      {
        Sid    = "S3ListAndGetMetrics"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:GetMetricsConfiguration",
          "s3:ListBucketVersions"
        ]
        Resource = var.source_bucket_arns
      },
      {
        Sid    = "CloudWatchMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics"
        ]
        Resource = "*"
      },
      {
        Sid    = "CostExplorerAccess"
        Effect = "Allow"
        Action = [
          "ce:GetCostAndUsage",
          "ce:GetCostForecast"
        ]
        Resource = "*"
      },
      {
        Sid    = "SNSPublish"
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = var.alarm_sns_topic_arns
      }
    ]
  })
}

#------------------------------------------------------------------------------
# Lambda Functions
#------------------------------------------------------------------------------

# Artifact Validator Lambda
resource "aws_lambda_function" "artifact_validator" {
  function_name    = "greenlang-artifact-validator"
  description      = "Validates uploaded artifacts for integrity and security"
  role             = aws_iam_role.artifact_validator_role.arn
  handler          = "handler.lambda_handler"
  runtime          = var.lambda_runtime
  timeout          = var.lambda_timeout
  memory_size      = var.lambda_memory_size

  filename         = data.archive_file.artifact_validator.output_path
  source_code_hash = data.archive_file.artifact_validator.output_base64sha256

  reserved_concurrent_executions = var.lambda_reserved_concurrency

  environment {
    variables = {
      ENVIRONMENT          = var.environment
      LOG_LEVEL            = var.environment == "prod" ? "INFO" : "DEBUG"
      MAX_FILE_SIZE_MB     = tostring(var.artifact_validator_config.max_file_size_mb)
      ALLOWED_EXTENSIONS   = join(",", var.artifact_validator_config.allowed_extensions)
      ENABLE_VIRUS_SCAN    = tostring(var.artifact_validator_config.enable_virus_scan)
      QUARANTINE_BUCKET    = var.artifact_validator_config.quarantine_bucket
      VALID_ARTIFACT_TAG   = var.artifact_validator_config.valid_artifacts_tag
      DLQ_URL              = aws_sqs_queue.artifact_processing_dlq.url
    }
  }

  dead_letter_config {
    target_arn = aws_sqs_queue.artifact_processing_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  dynamic "vpc_config" {
    for_each = var.vpc_config.enabled ? [1] : []
    content {
      subnet_ids         = var.vpc_config.subnet_ids
      security_group_ids = var.vpc_config.security_group_ids
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.artifact_validator,
    aws_iam_role_policy.artifact_validator_policy
  ]

  tags = merge(var.tags, {
    Name        = "greenlang-artifact-validator"
    Environment = var.environment
  })
}

# S3 permission for artifact validator
resource "aws_lambda_permission" "artifact_validator" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.artifact_validator.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = var.source_bucket_arns[0]
}

# Report Indexer Lambda
resource "aws_lambda_function" "report_indexer" {
  function_name    = "greenlang-report-indexer"
  description      = "Indexes reports for search functionality"
  role             = aws_iam_role.report_indexer_role.arn
  handler          = "handler.lambda_handler"
  runtime          = var.lambda_runtime
  timeout          = 600  # Reports can be large
  memory_size      = 1024 # More memory for PDF processing

  filename         = data.archive_file.report_indexer.output_path
  source_code_hash = data.archive_file.report_indexer.output_base64sha256

  reserved_concurrent_executions = var.lambda_reserved_concurrency

  environment {
    variables = {
      ENVIRONMENT          = var.environment
      LOG_LEVEL            = var.environment == "prod" ? "INFO" : "DEBUG"
      OPENSEARCH_ENDPOINT  = var.report_indexer_config.opensearch_endpoint
      OPENSEARCH_INDEX     = var.report_indexer_config.opensearch_index
      SUPPORTED_FORMATS    = join(",", var.report_indexer_config.supported_formats)
      EXTRACT_TEXT         = tostring(var.report_indexer_config.extract_text)
      DLQ_URL              = aws_sqs_queue.report_indexing_dlq.url
    }
  }

  dead_letter_config {
    target_arn = aws_sqs_queue.report_indexing_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  dynamic "vpc_config" {
    for_each = var.vpc_config.enabled ? [1] : []
    content {
      subnet_ids         = var.vpc_config.subnet_ids
      security_group_ids = var.vpc_config.security_group_ids
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.report_indexer,
    aws_iam_role_policy.report_indexer_policy
  ]

  tags = merge(var.tags, {
    Name        = "greenlang-report-indexer"
    Environment = var.environment
  })
}

# S3 permission for report indexer
resource "aws_lambda_permission" "report_indexer" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.report_indexer.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = var.source_bucket_arns[0]
}

# Audit Logger Lambda
resource "aws_lambda_function" "audit_logger" {
  function_name    = "greenlang-audit-logger"
  description      = "Records S3 events for compliance auditing"
  role             = aws_iam_role.audit_logger_role.arn
  handler          = "handler.lambda_handler"
  runtime          = var.lambda_runtime
  timeout          = 180
  memory_size      = 256

  filename         = data.archive_file.audit_logger.output_path
  source_code_hash = data.archive_file.audit_logger.output_base64sha256

  reserved_concurrent_executions = var.lambda_reserved_concurrency

  environment {
    variables = {
      ENVIRONMENT          = var.environment
      LOG_LEVEL            = var.environment == "prod" ? "INFO" : "DEBUG"
      AUDIT_BUCKET         = var.audit_logger_config.audit_bucket
      AUDIT_PREFIX         = var.audit_logger_config.audit_prefix
      ENABLE_DAILY_SUMMARY = tostring(var.audit_logger_config.enable_daily_summary)
      COMPLIANCE_MODE      = var.audit_logger_config.compliance_mode
      DLQ_URL              = aws_sqs_queue.audit_events_dlq.url
    }
  }

  dead_letter_config {
    target_arn = aws_sqs_queue.audit_events_dlq.arn
  }

  tracing_config {
    mode = "Active"
  }

  dynamic "vpc_config" {
    for_each = var.vpc_config.enabled ? [1] : []
    content {
      subnet_ids         = var.vpc_config.subnet_ids
      security_group_ids = var.vpc_config.security_group_ids
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.audit_logger,
    aws_iam_role_policy.audit_logger_policy
  ]

  tags = merge(var.tags, {
    Name        = "greenlang-audit-logger"
    Compliance  = "required"
    Environment = var.environment
  })
}

# Cost Tracker Lambda
resource "aws_lambda_function" "cost_tracker" {
  function_name    = "greenlang-cost-tracker"
  description      = "Tracks and reports S3 storage costs"
  role             = aws_iam_role.cost_tracker_role.arn
  handler          = "handler.lambda_handler"
  runtime          = var.lambda_runtime
  timeout          = 300
  memory_size      = 256

  filename         = data.archive_file.cost_tracker.output_path
  source_code_hash = data.archive_file.cost_tracker.output_base64sha256

  reserved_concurrent_executions = 1  # Only one at a time needed

  environment {
    variables = {
      ENVIRONMENT          = var.environment
      LOG_LEVEL            = var.environment == "prod" ? "INFO" : "DEBUG"
      COST_ALLOCATION_TAG  = var.cost_tracker_config.cost_allocation_tag
      ALERT_THRESHOLD_USD  = tostring(var.cost_tracker_config.alert_threshold_usd)
      ALERT_SNS_TOPIC_ARN  = length(var.alarm_sns_topic_arns) > 0 ? var.alarm_sns_topic_arns[0] : ""
      MONITORED_BUCKETS    = join(",", var.monitored_bucket_names)
    }
  }

  tracing_config {
    mode = "Active"
  }

  depends_on = [
    aws_cloudwatch_log_group.cost_tracker,
    aws_iam_role_policy.cost_tracker_policy
  ]

  tags = merge(var.tags, {
    Name        = "greenlang-cost-tracker"
    Environment = var.environment
  })
}

#------------------------------------------------------------------------------
# EventBridge Rule for Cost Tracker (Scheduled)
#------------------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "cost_tracker_schedule" {
  name                = "greenlang-cost-tracker-schedule"
  description         = "Trigger cost tracker Lambda daily"
  schedule_expression = var.cost_tracker_config.report_schedule

  tags = merge(var.tags, {
    Name        = "greenlang-cost-tracker-schedule"
    Environment = var.environment
  })
}

resource "aws_cloudwatch_event_target" "cost_tracker" {
  rule      = aws_cloudwatch_event_rule.cost_tracker_schedule.name
  target_id = "trigger-cost-tracker"
  arn       = aws_lambda_function.cost_tracker.arn
}

resource "aws_lambda_permission" "cost_tracker_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cost_tracker.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cost_tracker_schedule.arn
}

#------------------------------------------------------------------------------
# SQS Event Source Mappings
#------------------------------------------------------------------------------

resource "aws_lambda_event_source_mapping" "artifact_validator_sqs" {
  event_source_arn = aws_sqs_queue.artifact_processing.arn
  function_name    = aws_lambda_function.artifact_validator.arn
  batch_size       = 10
  enabled          = true

  function_response_types = ["ReportBatchItemFailures"]
}

resource "aws_lambda_event_source_mapping" "report_indexer_sqs" {
  event_source_arn = aws_sqs_queue.report_indexing.arn
  function_name    = aws_lambda_function.report_indexer.arn
  batch_size       = 5
  enabled          = true

  function_response_types = ["ReportBatchItemFailures"]
}

resource "aws_lambda_event_source_mapping" "audit_logger_sqs" {
  event_source_arn = aws_sqs_queue.audit_events.arn
  function_name    = aws_lambda_function.audit_logger.arn
  batch_size       = 25
  enabled          = true

  function_response_types = ["ReportBatchItemFailures"]
}
