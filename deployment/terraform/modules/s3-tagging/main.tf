# S3 Object Tagging Module for GreenLang
# Provides automated tagging via Lambda, EventBridge, and S3 Batch Operations
#
# This module creates:
# - Lambda function for auto-tagging new objects
# - EventBridge rule to trigger on S3 object creation
# - S3 Batch Operations IAM role for tag enforcement
# - Tag enforcement configuration

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.0"
    }
  }
}

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Get monitored buckets for EventBridge
data "aws_s3_bucket" "monitored_buckets" {
  for_each = toset(var.monitored_bucket_names)
  bucket   = each.value
}

# -----------------------------------------------------------------------------
# Lambda Function for Auto-Tagging
# -----------------------------------------------------------------------------

# Lambda function code
resource "local_file" "lambda_code" {
  filename = "${path.module}/lambda/auto_tagger.py"
  content  = <<-EOF
"""
S3 Object Auto-Tagger Lambda Function

This Lambda function automatically tags S3 objects based on their prefix
and artifact type. It's triggered by EventBridge when objects are created.

Author: GreenLang DevOps Team
Version: 1.0.0
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

# Tag schemas loaded from environment
TAG_SCHEMAS = json.loads(os.environ.get('TAG_SCHEMAS', '{}'))
REQUIRED_TAGS = json.loads(os.environ.get('REQUIRED_TAGS', '[]'))
DEFAULT_TAGS = json.loads(os.environ.get('DEFAULT_TAGS', '{}'))

# Artifact type patterns
ARTIFACT_PATTERNS = {
    'BUILD_ARTIFACTS': [r'^builds/', r'^packages/', r'^binaries/'],
    'CALCULATION_RESULTS': [r'^calculations/', r'^emissions/', r'^analysis/'],
    'REPORTS': [r'^reports/'],
    'AUDIT_LOGS': [r'^audit/', r'^logs/audit/'],
    'ML_MODELS': [r'^models/', r'^ml/'],
    'TEMPORARY': [r'^tmp/', r'^temp/', r'^scratch/']
}


def detect_artifact_type(key: str) -> str:
    """Detect artifact type from object key."""
    for artifact_type, patterns in ARTIFACT_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, key):
                return artifact_type
    return 'UNKNOWN'


def get_tags_for_artifact_type(artifact_type: str, key: str,
                                bucket: str) -> Dict[str, str]:
    """Get tags to apply based on artifact type."""
    tags = DEFAULT_TAGS.copy()

    # Add artifact type
    tags['artifact_type'] = artifact_type

    # Add timestamp
    tags['tagged_at'] = datetime.now(timezone.utc).isoformat()
    tags['tagged_by'] = 'auto-tagger-lambda'

    # Add bucket and key info
    tags['source_bucket'] = bucket

    # Get schema-specific tags
    schema = TAG_SCHEMAS.get(artifact_type, {})
    for tag_key, tag_config in schema.items():
        if isinstance(tag_config, dict):
            # Dynamic tag based on key patterns
            pattern = tag_config.get('pattern')
            if pattern:
                match = re.search(pattern, key)
                if match:
                    tags[tag_key] = match.group(1) if match.groups() else match.group(0)
            elif 'default' in tag_config:
                tags[tag_key] = tag_config['default']
        else:
            tags[tag_key] = str(tag_config)

    return tags


def get_existing_tags(bucket: str, key: str) -> Dict[str, str]:
    """Get existing tags from object."""
    try:
        response = s3_client.get_object_tagging(Bucket=bucket, Key=key)
        return {tag['Key']: tag['Value'] for tag in response.get('TagSet', [])}
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return {}
        logger.warning(f"Failed to get existing tags: {e}")
        return {}


def merge_tags(existing: Dict[str, str], new: Dict[str, str],
               overwrite: bool = False) -> Dict[str, str]:
    """Merge existing and new tags."""
    if overwrite:
        merged = existing.copy()
        merged.update(new)
        return merged
    else:
        # Only add new tags, don't overwrite existing
        merged = existing.copy()
        for key, value in new.items():
            if key not in merged:
                merged[key] = value
        return merged


def validate_required_tags(tags: Dict[str, str]) -> List[str]:
    """Validate that all required tags are present."""
    missing = []
    for required_tag in REQUIRED_TAGS:
        if required_tag not in tags:
            missing.append(required_tag)
    return missing


def apply_tags(bucket: str, key: str, tags: Dict[str, str]) -> bool:
    """Apply tags to S3 object."""
    try:
        tag_set = [{'Key': k, 'Value': str(v)[:256]} for k, v in tags.items()]

        # S3 has a limit of 10 tags per object
        if len(tag_set) > 10:
            logger.warning(f"Tag count ({len(tag_set)}) exceeds limit of 10, truncating")
            tag_set = tag_set[:10]

        s3_client.put_object_tagging(
            Bucket=bucket,
            Key=key,
            Tagging={'TagSet': tag_set}
        )
        logger.info(f"Applied {len(tag_set)} tags to s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logger.error(f"Failed to apply tags: {e}")
        return False


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for S3 object auto-tagging.

    Triggered by EventBridge when S3 objects are created.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    results = {
        'processed': 0,
        'tagged': 0,
        'skipped': 0,
        'failed': 0,
        'details': []
    }

    # Handle EventBridge event format
    if 'detail' in event:
        # S3 event via EventBridge
        detail = event['detail']
        bucket = detail.get('bucket', {}).get('name')
        key = detail.get('object', {}).get('key')

        if bucket and key:
            events_to_process = [{'bucket': bucket, 'key': key}]
        else:
            logger.warning("Missing bucket or key in event detail")
            return results
    elif 'Records' in event:
        # Direct S3 event
        events_to_process = [
            {
                'bucket': record['s3']['bucket']['name'],
                'key': record['s3']['object']['key']
            }
            for record in event.get('Records', [])
            if 's3' in record
        ]
    else:
        # Manual invocation with bucket/key
        bucket = event.get('bucket')
        key = event.get('key')
        if bucket and key:
            events_to_process = [{'bucket': bucket, 'key': key}]
        else:
            logger.error("Unable to parse event format")
            return results

    for item in events_to_process:
        bucket = item['bucket']
        key = item['key']
        results['processed'] += 1

        try:
            # Skip if key is a folder marker
            if key.endswith('/'):
                results['skipped'] += 1
                continue

            # Detect artifact type
            artifact_type = detect_artifact_type(key)

            # Get tags to apply
            new_tags = get_tags_for_artifact_type(artifact_type, key, bucket)

            # Get existing tags
            existing_tags = get_existing_tags(bucket, key)

            # Merge tags
            final_tags = merge_tags(existing_tags, new_tags, overwrite=False)

            # Validate required tags
            missing = validate_required_tags(final_tags)
            if missing:
                logger.warning(f"Missing required tags for {key}: {missing}")

            # Apply tags
            if apply_tags(bucket, key, final_tags):
                results['tagged'] += 1
                results['details'].append({
                    'bucket': bucket,
                    'key': key,
                    'artifact_type': artifact_type,
                    'tags_applied': len(final_tags),
                    'status': 'success'
                })
            else:
                results['failed'] += 1
                results['details'].append({
                    'bucket': bucket,
                    'key': key,
                    'status': 'failed',
                    'reason': 'Failed to apply tags'
                })

        except Exception as e:
            logger.error(f"Error processing {bucket}/{key}: {e}")
            results['failed'] += 1
            results['details'].append({
                'bucket': bucket,
                'key': key,
                'status': 'error',
                'reason': str(e)
            })

    logger.info(f"Processing complete: {json.dumps(results)}")
    return results
EOF
}

# Create Lambda deployment package
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = local_file.lambda_code.filename
  output_path = "${path.module}/lambda/auto_tagger.zip"

  depends_on = [local_file.lambda_code]
}

# Lambda execution role
resource "aws_iam_role" "lambda_role" {
  name = "${var.name_prefix}-auto-tagger-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-auto-tagger-role"
  })
}

# Lambda basic execution policy
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lambda S3 tagging policy
resource "aws_iam_role_policy" "lambda_s3_policy" {
  name = "${var.name_prefix}-s3-tagging-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3GetObjectTagging"
        Effect = "Allow"
        Action = [
          "s3:GetObjectTagging",
          "s3:GetObjectVersionTagging"
        ]
        Resource = [
          for bucket in var.monitored_bucket_names :
          "arn:aws:s3:::${bucket}/*"
        ]
      },
      {
        Sid    = "S3PutObjectTagging"
        Effect = "Allow"
        Action = [
          "s3:PutObjectTagging",
          "s3:PutObjectVersionTagging"
        ]
        Resource = [
          for bucket in var.monitored_bucket_names :
          "arn:aws:s3:::${bucket}/*"
        ]
      }
    ]
  })
}

# Lambda function
resource "aws_lambda_function" "auto_tagger" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.name_prefix}-auto-tagger"
  role             = aws_iam_role.lambda_role.arn
  handler          = "auto_tagger.lambda_handler"
  runtime          = "python3.11"
  timeout          = 60
  memory_size      = 256
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      TAG_SCHEMAS   = jsonencode(var.tag_schemas)
      REQUIRED_TAGS = jsonencode(var.required_tags)
      DEFAULT_TAGS  = jsonencode(var.default_tags)
      LOG_LEVEL     = var.log_level
    }
  }

  reserved_concurrent_executions = var.lambda_reserved_concurrency

  tracing_config {
    mode = var.enable_xray_tracing ? "Active" : "PassThrough"
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-auto-tagger"
  })
}

# Lambda permission for EventBridge
resource "aws_lambda_permission" "eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.auto_tagger.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.s3_object_created.arn
}

# -----------------------------------------------------------------------------
# EventBridge Rule for S3 Object Creation
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "s3_object_created" {
  name        = "${var.name_prefix}-s3-object-created"
  description = "Trigger auto-tagging Lambda when S3 objects are created"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = {
        name = var.monitored_bucket_names
      }
    }
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-s3-object-created"
  })
}

resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.s3_object_created.name
  target_id = "AutoTaggerLambda"
  arn       = aws_lambda_function.auto_tagger.arn
}

# Enable S3 Event Notifications to EventBridge for each bucket
resource "aws_s3_bucket_notification" "eventbridge" {
  for_each = toset(var.monitored_bucket_names)

  bucket      = each.value
  eventbridge = true
}

# -----------------------------------------------------------------------------
# S3 Batch Operations IAM Role
# -----------------------------------------------------------------------------

resource "aws_iam_role" "batch_operations_role" {
  name = "${var.name_prefix}-batch-operations-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "batchoperations.s3.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-batch-operations-role"
  })
}

resource "aws_iam_role_policy" "batch_operations_policy" {
  name = "${var.name_prefix}-batch-operations-policy"
  role = aws_iam_role.batch_operations_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging",
          "s3:GetObjectVersionTagging",
          "s3:PutObject",
          "s3:PutObjectTagging",
          "s3:PutObjectVersionTagging"
        ]
        Resource = flatten([
          for bucket in var.monitored_bucket_names :
          ["arn:aws:s3:::${bucket}/*"]
        ])
      },
      {
        Sid    = "S3ManifestAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]
        Resource = var.batch_manifest_bucket_arn != "" ? [
          "${var.batch_manifest_bucket_arn}/*"
        ] : ["arn:aws:s3:::${var.monitored_bucket_names[0]}/*"]
      },
      {
        Sid    = "S3ReportAccess"
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = var.batch_report_bucket_arn != "" ? [
          "${var.batch_report_bucket_arn}/*"
        ] : ["arn:aws:s3:::${var.monitored_bucket_names[0]}/*"]
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  count = var.enable_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-auto-tagger-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Auto-tagger Lambda function errors"
  alarm_actions       = var.alarm_actions

  dimensions = {
    FunctionName = aws_lambda_function.auto_tagger.function_name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  count = var.enable_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-auto-tagger-throttles"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Throttles"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Auto-tagger Lambda function throttles"
  alarm_actions       = var.alarm_actions

  dimensions = {
    FunctionName = aws_lambda_function.auto_tagger.function_name
  }

  tags = var.tags
}

# -----------------------------------------------------------------------------
# Lambda for Tag Enforcement (Batch Job Trigger)
# -----------------------------------------------------------------------------

resource "aws_lambda_function" "tag_enforcement" {
  count = var.enable_tag_enforcement ? 1 : 0

  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.name_prefix}-tag-enforcement"
  role             = aws_iam_role.tag_enforcement_role[0].arn
  handler          = "auto_tagger.lambda_handler"
  runtime          = "python3.11"
  timeout          = 300
  memory_size      = 512
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      TAG_SCHEMAS           = jsonencode(var.tag_schemas)
      REQUIRED_TAGS         = jsonencode(var.required_tags)
      DEFAULT_TAGS          = jsonencode(var.default_tags)
      ENFORCEMENT_MODE      = "batch"
      BATCH_ROLE_ARN        = aws_iam_role.batch_operations_role.arn
      MANIFEST_BUCKET       = var.batch_manifest_bucket_arn != "" ? split(":", var.batch_manifest_bucket_arn)[5] : var.monitored_bucket_names[0]
      REPORT_BUCKET         = var.batch_report_bucket_arn != "" ? split(":", var.batch_report_bucket_arn)[5] : var.monitored_bucket_names[0]
    }
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-tag-enforcement"
  })
}

resource "aws_iam_role" "tag_enforcement_role" {
  count = var.enable_tag_enforcement ? 1 : 0

  name = "${var.name_prefix}-tag-enforcement-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "tag_enforcement_basic" {
  count = var.enable_tag_enforcement ? 1 : 0

  role       = aws_iam_role.tag_enforcement_role[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "tag_enforcement_s3_batch" {
  count = var.enable_tag_enforcement ? 1 : 0

  name = "${var.name_prefix}-s3-batch-policy"
  role = aws_iam_role.tag_enforcement_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BatchOperations"
        Effect = "Allow"
        Action = [
          "s3:CreateJob",
          "s3:DescribeJob",
          "s3:GetJobTagging",
          "s3:PutJobTagging",
          "s3:UpdateJobPriority",
          "s3:UpdateJobStatus"
        ]
        Resource = "*"
      },
      {
        Sid    = "PassRole"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = aws_iam_role.batch_operations_role.arn
      }
    ]
  })
}

# Scheduled enforcement (weekly)
resource "aws_cloudwatch_event_rule" "tag_enforcement_schedule" {
  count = var.enable_tag_enforcement ? 1 : 0

  name                = "${var.name_prefix}-tag-enforcement-schedule"
  description         = "Weekly tag enforcement job"
  schedule_expression = var.enforcement_schedule

  tags = var.tags
}

resource "aws_cloudwatch_event_target" "tag_enforcement_target" {
  count = var.enable_tag_enforcement ? 1 : 0

  rule      = aws_cloudwatch_event_rule.tag_enforcement_schedule[0].name
  target_id = "TagEnforcementLambda"
  arn       = aws_lambda_function.tag_enforcement[0].arn
}

resource "aws_lambda_permission" "tag_enforcement_eventbridge" {
  count = var.enable_tag_enforcement ? 1 : 0

  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.tag_enforcement[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.tag_enforcement_schedule[0].arn
}
