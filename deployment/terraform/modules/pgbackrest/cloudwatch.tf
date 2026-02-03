# =============================================================================
# pgBackRest Terraform Module - CloudWatch Configuration
# GreenLang Database Infrastructure
# =============================================================================
# CloudWatch log groups, metrics, and alarms for pgBackRest monitoring
# =============================================================================

# -----------------------------------------------------------------------------
# CloudWatch Log Group
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "pgbackrest" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name              = "/aws/pgbackrest/${local.name_prefix}"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.pgbackrest.arn

  tags = merge(local.tags, {
    Name = "${local.name_prefix}-pgbackrest-logs"
  })
}

# Backup log stream
resource "aws_cloudwatch_log_stream" "backup" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name           = "backup"
  log_group_name = aws_cloudwatch_log_group.pgbackrest[0].name
}

# Restore log stream
resource "aws_cloudwatch_log_stream" "restore" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name           = "restore"
  log_group_name = aws_cloudwatch_log_group.pgbackrest[0].name
}

# Verification log stream
resource "aws_cloudwatch_log_stream" "verification" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name           = "verification"
  log_group_name = aws_cloudwatch_log_group.pgbackrest[0].name
}

# -----------------------------------------------------------------------------
# CloudWatch Metric Filters
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_metric_filter" "backup_success" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name           = "${local.name_prefix}-pgbackrest-backup-success"
  pattern        = "backup completed successfully"
  log_group_name = aws_cloudwatch_log_group.pgbackrest[0].name

  metric_transformation {
    name          = "BackupSuccess"
    namespace     = "${var.project_name}/pgbackrest"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "backup_failure" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name           = "${local.name_prefix}-pgbackrest-backup-failure"
  pattern        = "backup FAILED"
  log_group_name = aws_cloudwatch_log_group.pgbackrest[0].name

  metric_transformation {
    name          = "BackupFailure"
    namespace     = "${var.project_name}/pgbackrest"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "restore_success" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name           = "${local.name_prefix}-pgbackrest-restore-success"
  pattern        = "restore completed successfully"
  log_group_name = aws_cloudwatch_log_group.pgbackrest[0].name

  metric_transformation {
    name          = "RestoreSuccess"
    namespace     = "${var.project_name}/pgbackrest"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "restore_failure" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name           = "${local.name_prefix}-pgbackrest-restore-failure"
  pattern        = "restore FAILED"
  log_group_name = aws_cloudwatch_log_group.pgbackrest[0].name

  metric_transformation {
    name          = "RestoreFailure"
    namespace     = "${var.project_name}/pgbackrest"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "verification_failure" {
  count = var.create_cloudwatch_log_group ? 1 : 0

  name           = "${local.name_prefix}-pgbackrest-verification-failure"
  pattern        = "verification FAILED"
  log_group_name = aws_cloudwatch_log_group.pgbackrest[0].name

  metric_transformation {
    name          = "VerificationFailure"
    namespace     = "${var.project_name}/pgbackrest"
    value         = "1"
    default_value = "0"
  }
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "backup_failure" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-pgbackrest-backup-failure"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BackupFailure"
  namespace           = "${var.project_name}/pgbackrest"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "pgBackRest backup has failed"
  treat_missing_data  = "notBreaching"

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  dimensions = {
    Stanza = var.project_name
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "no_recent_backup" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-pgbackrest-no-recent-backup"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BackupSuccess"
  namespace           = "${var.project_name}/pgbackrest"
  period              = 86400  # 24 hours
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "No successful pgBackRest backup in the last 24 hours"
  treat_missing_data  = "breaching"

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "restore_failure" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-pgbackrest-restore-failure"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "RestoreFailure"
  namespace           = "${var.project_name}/pgbackrest"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "pgBackRest restore has failed"
  treat_missing_data  = "notBreaching"

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "verification_failure" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-pgbackrest-verification-failure"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "VerificationFailure"
  namespace           = "${var.project_name}/pgbackrest"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "pgBackRest backup verification has failed"
  treat_missing_data  = "notBreaching"

  alarm_actions = var.alarm_actions
  ok_actions    = var.ok_actions

  tags = local.tags
}

# S3 bucket size alarm
resource "aws_cloudwatch_metric_alarm" "bucket_size" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-pgbackrest-bucket-size"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BucketSizeBytes"
  namespace           = "AWS/S3"
  period              = 86400  # 1 day
  statistic           = "Average"
  threshold           = 1099511627776  # 1 TB
  alarm_description   = "pgBackRest S3 bucket size exceeds 1TB"
  treat_missing_data  = "notBreaching"

  alarm_actions = var.alarm_actions

  dimensions = {
    BucketName  = aws_s3_bucket.pgbackrest.id
    StorageType = "StandardStorage"
  }

  tags = local.tags
}

# -----------------------------------------------------------------------------
# CloudWatch Dashboard
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_dashboard" "pgbackrest" {
  count = var.create_cloudwatch_alarms ? 1 : 0

  dashboard_name = "${local.name_prefix}-pgbackrest"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Backup Status"
          region = data.aws_region.current.name
          metrics = [
            ["${var.project_name}/pgbackrest", "BackupSuccess", { label = "Success", color = "#2ca02c" }],
            [".", "BackupFailure", { label = "Failure", color = "#d62728" }]
          ]
          period = 86400
          stat   = "Sum"
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
          title  = "Restore and Verification Status"
          region = data.aws_region.current.name
          metrics = [
            ["${var.project_name}/pgbackrest", "RestoreSuccess", { label = "Restore Success", color = "#2ca02c" }],
            [".", "RestoreFailure", { label = "Restore Failure", color = "#d62728" }],
            [".", "VerificationFailure", { label = "Verification Failure", color = "#ff7f0e" }]
          ]
          period = 86400
          stat   = "Sum"
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
          title  = "S3 Bucket Size"
          region = data.aws_region.current.name
          metrics = [
            ["AWS/S3", "BucketSizeBytes", "BucketName", aws_s3_bucket.pgbackrest.id, "StorageType", "StandardStorage"]
          ]
          period = 86400
          stat   = "Average"
          view   = "timeSeries"
          yAxis = {
            left = {
              label     = "Bytes"
              showUnits = false
            }
          }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "S3 Object Count"
          region = data.aws_region.current.name
          metrics = [
            ["AWS/S3", "NumberOfObjects", "BucketName", aws_s3_bucket.pgbackrest.id, "StorageType", "AllStorageTypes"]
          ]
          period = 86400
          stat   = "Average"
          view   = "timeSeries"
        }
      },
      {
        type   = "alarm"
        x      = 0
        y      = 12
        width  = 24
        height = 4
        properties = {
          title  = "Backup Alarms"
          alarms = var.create_cloudwatch_alarms ? [
            aws_cloudwatch_metric_alarm.backup_failure[0].arn,
            aws_cloudwatch_metric_alarm.no_recent_backup[0].arn,
            aws_cloudwatch_metric_alarm.restore_failure[0].arn,
            aws_cloudwatch_metric_alarm.verification_failure[0].arn
          ] : []
        }
      }
    ]
  })
}
