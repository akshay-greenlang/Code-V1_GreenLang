#------------------------------------------------------------------------------
# S3 CloudTrail Integration - CloudWatch Alerts
# GreenLang Infrastructure
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Metric Filters for Security Events
#------------------------------------------------------------------------------

# Metric Filter: Unauthorized API Calls
resource "aws_cloudwatch_log_metric_filter" "unauthorized_api_calls" {
  name           = "greenlang-s3-unauthorized-api-calls"
  log_group_name = aws_cloudwatch_log_group.cloudtrail.name
  pattern        = "{ ($.errorCode = \"*UnauthorizedAccess*\") || ($.errorCode = \"AccessDenied*\") }"

  metric_transformation {
    name          = "UnauthorizedAPICalls"
    namespace     = "GreenLang/CloudTrail/S3"
    value         = "1"
    default_value = "0"
  }
}

# Metric Filter: S3 Bucket Policy Changes
resource "aws_cloudwatch_log_metric_filter" "bucket_policy_changes" {
  name           = "greenlang-s3-bucket-policy-changes"
  log_group_name = aws_cloudwatch_log_group.cloudtrail.name
  pattern        = "{ ($.eventSource = \"s3.amazonaws.com\") && (($.eventName = \"PutBucketPolicy\") || ($.eventName = \"DeleteBucketPolicy\") || ($.eventName = \"PutBucketAcl\")) }"

  metric_transformation {
    name          = "BucketPolicyChanges"
    namespace     = "GreenLang/CloudTrail/S3"
    value         = "1"
    default_value = "0"
  }
}

# Metric Filter: S3 Public Access Block Disabled
resource "aws_cloudwatch_log_metric_filter" "public_access_block_disabled" {
  name           = "greenlang-s3-public-access-disabled"
  log_group_name = aws_cloudwatch_log_group.cloudtrail.name
  pattern        = "{ ($.eventSource = \"s3.amazonaws.com\") && ($.eventName = \"DeleteBucketPublicAccessBlock\") }"

  metric_transformation {
    name          = "PublicAccessBlockDisabled"
    namespace     = "GreenLang/CloudTrail/S3"
    value         = "1"
    default_value = "0"
  }
}

# Metric Filter: S3 Object Deletions
resource "aws_cloudwatch_log_metric_filter" "object_deletions" {
  name           = "greenlang-s3-object-deletions"
  log_group_name = aws_cloudwatch_log_group.cloudtrail.name
  pattern        = "{ ($.eventSource = \"s3.amazonaws.com\") && (($.eventName = \"DeleteObject\") || ($.eventName = \"DeleteObjects\")) }"

  metric_transformation {
    name          = "ObjectDeletions"
    namespace     = "GreenLang/CloudTrail/S3"
    value         = "1"
    default_value = "0"
  }
}

# Metric Filter: S3 Encryption Disabled
resource "aws_cloudwatch_log_metric_filter" "encryption_disabled" {
  name           = "greenlang-s3-encryption-disabled"
  log_group_name = aws_cloudwatch_log_group.cloudtrail.name
  pattern        = "{ ($.eventSource = \"s3.amazonaws.com\") && ($.eventName = \"DeleteBucketEncryption\") }"

  metric_transformation {
    name          = "EncryptionDisabled"
    namespace     = "GreenLang/CloudTrail/S3"
    value         = "1"
    default_value = "0"
  }
}

# Metric Filter: Cross-Account Access
resource "aws_cloudwatch_log_metric_filter" "cross_account_access" {
  name           = "greenlang-s3-cross-account-access"
  log_group_name = aws_cloudwatch_log_group.cloudtrail.name
  pattern        = "{ ($.eventSource = \"s3.amazonaws.com\") && ($.userIdentity.accountId != \"${data.aws_caller_identity.current.account_id}\") }"

  metric_transformation {
    name          = "CrossAccountAccess"
    namespace     = "GreenLang/CloudTrail/S3"
    value         = "1"
    default_value = "0"
  }
}

# Metric Filter: Failed Authentication
resource "aws_cloudwatch_log_metric_filter" "failed_authentication" {
  name           = "greenlang-s3-failed-auth"
  log_group_name = aws_cloudwatch_log_group.cloudtrail.name
  pattern        = "{ ($.eventSource = \"s3.amazonaws.com\") && ($.errorCode = \"InvalidAccessKeyId\" || $.errorCode = \"SignatureDoesNotMatch\") }"

  metric_transformation {
    name          = "FailedAuthentication"
    namespace     = "GreenLang/CloudTrail/S3"
    value         = "1"
    default_value = "0"
  }
}

#------------------------------------------------------------------------------
# CloudWatch Alarms for Security Events
#------------------------------------------------------------------------------

# SNS Topic for Security Alerts
resource "aws_sns_topic" "security_alerts" {
  name              = "greenlang-s3-security-alerts"
  display_name      = "GreenLang S3 Security Alerts"
  kms_master_key_id = aws_kms_key.cloudtrail.id

  tags = merge(var.tags, {
    Name        = "greenlang-s3-security-alerts"
    Environment = var.environment
  })
}

# Alarm: Unauthorized API Calls
resource "aws_cloudwatch_metric_alarm" "unauthorized_api_calls" {
  alarm_name          = "greenlang-s3-unauthorized-api-calls"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "UnauthorizedAPICalls"
  namespace           = "GreenLang/CloudTrail/S3"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "CRITICAL: Multiple unauthorized S3 API calls detected"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
  ok_actions          = [aws_sns_topic.security_alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = merge(var.tags, {
    Name     = "greenlang-s3-unauthorized-alarm"
    Severity = "critical"
  })
}

# Alarm: Bucket Policy Changes
resource "aws_cloudwatch_metric_alarm" "bucket_policy_changes" {
  alarm_name          = "greenlang-s3-bucket-policy-changes"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BucketPolicyChanges"
  namespace           = "GreenLang/CloudTrail/S3"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "WARNING: S3 bucket policy or ACL was modified"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
  ok_actions          = [aws_sns_topic.security_alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = merge(var.tags, {
    Name     = "greenlang-s3-policy-change-alarm"
    Severity = "warning"
  })
}

# Alarm: Public Access Block Disabled
resource "aws_cloudwatch_metric_alarm" "public_access_block_disabled" {
  alarm_name          = "greenlang-s3-public-access-disabled"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "PublicAccessBlockDisabled"
  namespace           = "GreenLang/CloudTrail/S3"
  period              = 60
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "CRITICAL: S3 public access block was disabled - potential data exposure"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = merge(var.tags, {
    Name     = "greenlang-s3-public-access-alarm"
    Severity = "critical"
  })
}

# Alarm: High Volume Object Deletions
resource "aws_cloudwatch_metric_alarm" "high_volume_deletions" {
  alarm_name          = "greenlang-s3-high-volume-deletions"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ObjectDeletions"
  namespace           = "GreenLang/CloudTrail/S3"
  period              = 300
  statistic           = "Sum"
  threshold           = 100
  alarm_description   = "WARNING: High volume of S3 object deletions detected"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = merge(var.tags, {
    Name     = "greenlang-s3-deletion-alarm"
    Severity = "warning"
  })
}

# Alarm: Encryption Disabled
resource "aws_cloudwatch_metric_alarm" "encryption_disabled" {
  alarm_name          = "greenlang-s3-encryption-disabled"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EncryptionDisabled"
  namespace           = "GreenLang/CloudTrail/S3"
  period              = 60
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "CRITICAL: S3 bucket encryption was disabled - compliance violation"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = merge(var.tags, {
    Name     = "greenlang-s3-encryption-alarm"
    Severity = "critical"
  })
}

# Alarm: Unusual Cross-Account Access
resource "aws_cloudwatch_metric_alarm" "cross_account_access" {
  alarm_name          = "greenlang-s3-cross-account-access"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CrossAccountAccess"
  namespace           = "GreenLang/CloudTrail/S3"
  period              = 300
  statistic           = "Sum"
  threshold           = 50
  alarm_description   = "WARNING: High volume of cross-account S3 access detected"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = merge(var.tags, {
    Name     = "greenlang-s3-cross-account-alarm"
    Severity = "warning"
  })
}

# Alarm: Failed Authentication Attempts
resource "aws_cloudwatch_metric_alarm" "failed_authentication" {
  alarm_name          = "greenlang-s3-failed-authentication"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "FailedAuthentication"
  namespace           = "GreenLang/CloudTrail/S3"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "CRITICAL: Multiple failed S3 authentication attempts - possible attack"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = merge(var.tags, {
    Name     = "greenlang-s3-auth-failure-alarm"
    Severity = "critical"
  })
}

#------------------------------------------------------------------------------
# Composite Alarms for Complex Security Scenarios
#------------------------------------------------------------------------------

resource "aws_cloudwatch_composite_alarm" "security_incident" {
  alarm_description = "CRITICAL: Multiple security indicators triggered - potential security incident"
  alarm_name        = "greenlang-s3-security-incident"

  alarm_rule = <<-EOT
    ALARM(${aws_cloudwatch_metric_alarm.unauthorized_api_calls.alarm_name})
    OR
    (
      ALARM(${aws_cloudwatch_metric_alarm.bucket_policy_changes.alarm_name})
      AND
      ALARM(${aws_cloudwatch_metric_alarm.public_access_block_disabled.alarm_name})
    )
    OR
    (
      ALARM(${aws_cloudwatch_metric_alarm.encryption_disabled.alarm_name})
      AND
      ALARM(${aws_cloudwatch_metric_alarm.high_volume_deletions.alarm_name})
    )
  EOT

  alarm_actions = [aws_sns_topic.security_alerts.arn]

  tags = merge(var.tags, {
    Name     = "greenlang-s3-security-incident"
    Severity = "critical"
  })
}

#------------------------------------------------------------------------------
# CloudWatch Dashboard for S3 Security
#------------------------------------------------------------------------------

resource "aws_cloudwatch_dashboard" "s3_security" {
  dashboard_name = "GreenLang-S3-Security"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Unauthorized API Calls"
          region = data.aws_region.current.name
          metrics = [
            ["GreenLang/CloudTrail/S3", "UnauthorizedAPICalls", { "stat" = "Sum", "period" = 300 }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Cross-Account Access"
          region = data.aws_region.current.name
          metrics = [
            ["GreenLang/CloudTrail/S3", "CrossAccountAccess", { "stat" = "Sum", "period" = 300 }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 8
        height = 6
        properties = {
          title  = "Policy Changes"
          region = data.aws_region.current.name
          metrics = [
            ["GreenLang/CloudTrail/S3", "BucketPolicyChanges", { "stat" = "Sum", "period" = 300 }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 6
        width  = 8
        height = 6
        properties = {
          title  = "Object Deletions"
          region = data.aws_region.current.name
          metrics = [
            ["GreenLang/CloudTrail/S3", "ObjectDeletions", { "stat" = "Sum", "period" = 300 }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 6
        width  = 8
        height = 6
        properties = {
          title  = "Failed Authentication"
          region = data.aws_region.current.name
          metrics = [
            ["GreenLang/CloudTrail/S3", "FailedAuthentication", { "stat" = "Sum", "period" = 300 }]
          ]
          view = "timeSeries"
        }
      },
      {
        type   = "alarm"
        x      = 0
        y      = 12
        width  = 24
        height = 4
        properties = {
          title = "Security Alarms Status"
          alarms = [
            aws_cloudwatch_metric_alarm.unauthorized_api_calls.arn,
            aws_cloudwatch_metric_alarm.bucket_policy_changes.arn,
            aws_cloudwatch_metric_alarm.public_access_block_disabled.arn,
            aws_cloudwatch_metric_alarm.encryption_disabled.arn,
            aws_cloudwatch_metric_alarm.failed_authentication.arn
          ]
        }
      }
    ]
  })
}

#------------------------------------------------------------------------------
# Outputs
#------------------------------------------------------------------------------

output "security_alerts_topic_arn" {
  description = "ARN of the SNS topic for security alerts"
  value       = aws_sns_topic.security_alerts.arn
}

output "security_dashboard_name" {
  description = "Name of the CloudWatch security dashboard"
  value       = aws_cloudwatch_dashboard.s3_security.dashboard_name
}

output "alarm_arns" {
  description = "Map of CloudWatch alarm ARNs"
  value = {
    unauthorized_api_calls     = aws_cloudwatch_metric_alarm.unauthorized_api_calls.arn
    bucket_policy_changes      = aws_cloudwatch_metric_alarm.bucket_policy_changes.arn
    public_access_disabled     = aws_cloudwatch_metric_alarm.public_access_block_disabled.arn
    high_volume_deletions      = aws_cloudwatch_metric_alarm.high_volume_deletions.arn
    encryption_disabled        = aws_cloudwatch_metric_alarm.encryption_disabled.arn
    cross_account_access       = aws_cloudwatch_metric_alarm.cross_account_access.arn
    failed_authentication      = aws_cloudwatch_metric_alarm.failed_authentication.arn
    security_incident_composite = aws_cloudwatch_composite_alarm.security_incident.arn
  }
}
