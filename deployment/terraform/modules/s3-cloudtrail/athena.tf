#------------------------------------------------------------------------------
# S3 CloudTrail Integration - Athena Configuration
# GreenLang Infrastructure
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Athena Workgroup
#------------------------------------------------------------------------------

resource "aws_athena_workgroup" "cloudtrail" {
  name        = "greenlang-cloudtrail-analysis"
  description = "Workgroup for CloudTrail log analysis"
  state       = "ENABLED"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true
    bytes_scanned_cutoff_per_query     = 107374182400  # 100 GB

    result_configuration {
      output_location = "s3://${aws_s3_bucket.athena_results.id}/query-results/"

      encryption_configuration {
        encryption_option = "SSE_KMS"
        kms_key_arn       = aws_kms_key.cloudtrail.arn
      }
    }

    engine_version {
      selected_engine_version = "Athena engine version 3"
    }
  }

  tags = merge(var.tags, {
    Name        = "greenlang-cloudtrail-workgroup"
    Environment = var.environment
  })
}

#------------------------------------------------------------------------------
# S3 Bucket for Athena Query Results
#------------------------------------------------------------------------------

resource "aws_s3_bucket" "athena_results" {
  bucket = "greenlang-athena-results-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name        = "greenlang-athena-results"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.cloudtrail.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id

  rule {
    id     = "cleanup-query-results"
    status = "Enabled"

    filter {
      prefix = "query-results/"
    }

    expiration {
      days = 30
    }
  }
}

#------------------------------------------------------------------------------
# Glue Database for CloudTrail
#------------------------------------------------------------------------------

resource "aws_glue_catalog_database" "cloudtrail" {
  name        = "greenlang_cloudtrail_logs"
  description = "Database for GreenLang CloudTrail log analysis"

  create_table_default_permission {
    permissions = ["ALL"]

    principal {
      data_lake_principal_identifier = "IAM_ALLOWED_PRINCIPALS"
    }
  }
}

#------------------------------------------------------------------------------
# Athena Table for CloudTrail Logs with Partition Projection
#------------------------------------------------------------------------------

resource "aws_glue_catalog_table" "cloudtrail_logs" {
  database_name = aws_glue_catalog_database.cloudtrail.name
  name          = "cloudtrail_logs"
  description   = "CloudTrail logs with partition projection for efficient querying"
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    "classification"                         = "cloudtrail"
    "projection.enabled"                     = "true"
    "projection.region.type"                 = "enum"
    "projection.region.values"               = join(",", var.projection_regions)
    "projection.date.type"                   = "date"
    "projection.date.range"                  = "2024/01/01,NOW"
    "projection.date.format"                 = "yyyy/MM/dd"
    "projection.date.interval"               = "1"
    "projection.date.interval.unit"          = "DAYS"
    "storage.location.template"              = "s3://${aws_s3_bucket.cloudtrail_logs.id}/AWSLogs/${data.aws_caller_identity.current.account_id}/CloudTrail/$${region}/$${date}"
    "EXTERNAL"                               = "TRUE"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.cloudtrail_logs.id}/AWSLogs/${data.aws_caller_identity.current.account_id}/CloudTrail/"
    input_format  = "com.amazon.emr.cloudtrail.CloudTrailInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      name                  = "CloudTrailSerDe"
      serialization_library = "org.apache.hive.hcatalog.data.JsonSerDe"

      parameters = {
        "serialization.format" = "1"
      }
    }

    columns {
      name = "eventversion"
      type = "string"
    }

    columns {
      name = "useridentity"
      type = "struct<type:string,principalid:string,arn:string,accountid:string,invokedby:string,accesskeyid:string,username:string,sessioncontext:struct<attributes:struct<mfaauthenticated:string,creationdate:string>,sessionissuer:struct<type:string,principalid:string,arn:string,accountid:string,username:string>>>"
    }

    columns {
      name = "eventtime"
      type = "string"
    }

    columns {
      name = "eventsource"
      type = "string"
    }

    columns {
      name = "eventname"
      type = "string"
    }

    columns {
      name = "awsregion"
      type = "string"
    }

    columns {
      name = "sourceipaddress"
      type = "string"
    }

    columns {
      name = "useragent"
      type = "string"
    }

    columns {
      name = "errorcode"
      type = "string"
    }

    columns {
      name = "errormessage"
      type = "string"
    }

    columns {
      name = "requestparameters"
      type = "string"
    }

    columns {
      name = "responseelements"
      type = "string"
    }

    columns {
      name = "additionaleventdata"
      type = "string"
    }

    columns {
      name = "requestid"
      type = "string"
    }

    columns {
      name = "eventid"
      type = "string"
    }

    columns {
      name = "resources"
      type = "array<struct<arn:string,accountid:string,type:string>>"
    }

    columns {
      name = "eventtype"
      type = "string"
    }

    columns {
      name = "apiversion"
      type = "string"
    }

    columns {
      name = "readonly"
      type = "string"
    }

    columns {
      name = "recipientaccountid"
      type = "string"
    }

    columns {
      name = "serviceeventdetails"
      type = "string"
    }

    columns {
      name = "sharedeventid"
      type = "string"
    }

    columns {
      name = "vpcendpointid"
      type = "string"
    }

    columns {
      name = "tlsdetails"
      type = "struct<tlsversion:string,ciphersuite:string,clientprovidedhostheader:string>"
    }
  }

  partition_keys {
    name = "region"
    type = "string"
  }

  partition_keys {
    name = "date"
    type = "string"
  }
}

#------------------------------------------------------------------------------
# Named Queries for Common Audit Scenarios
#------------------------------------------------------------------------------

# Query: Who accessed sensitive data?
resource "aws_athena_named_query" "sensitive_data_access" {
  name        = "sensitive-data-access"
  description = "Find who accessed sensitive data in monitored S3 buckets"
  database    = aws_glue_catalog_database.cloudtrail.name
  workgroup   = aws_athena_workgroup.cloudtrail.id

  query = <<-EOT
    -- Sensitive Data Access Query
    -- Finds all users who accessed objects in sensitive S3 buckets

    SELECT
      eventtime,
      useridentity.arn AS user_arn,
      useridentity.username AS username,
      useridentity.principalid AS principal_id,
      sourceipaddress,
      eventname,
      json_extract_scalar(requestparameters, '$.bucketName') AS bucket_name,
      json_extract_scalar(requestparameters, '$.key') AS object_key,
      errorcode,
      useragent
    FROM
      ${aws_glue_catalog_database.cloudtrail.name}.${aws_glue_catalog_table.cloudtrail_logs.name}
    WHERE
      eventsource = 's3.amazonaws.com'
      AND eventname IN ('GetObject', 'GetObjectVersion', 'HeadObject')
      AND json_extract_scalar(requestparameters, '$.bucketName') IN (${join(",", [for b in var.sensitive_buckets : "'${b}'"])})
      AND date >= date_format(current_date - interval '7' day, '%Y/%m/%d')
    ORDER BY
      eventtime DESC
    LIMIT 1000;
  EOT
}

# Query: Failed access attempts
resource "aws_athena_named_query" "failed_access_attempts" {
  name        = "failed-access-attempts"
  description = "Find failed S3 access attempts (access denied, not found)"
  database    = aws_glue_catalog_database.cloudtrail.name
  workgroup   = aws_athena_workgroup.cloudtrail.id

  query = <<-EOT
    -- Failed Access Attempts Query
    -- Identifies unauthorized access attempts to S3 buckets

    SELECT
      eventtime,
      useridentity.arn AS user_arn,
      useridentity.accountid AS account_id,
      sourceipaddress,
      eventname,
      errorcode,
      errormessage,
      json_extract_scalar(requestparameters, '$.bucketName') AS bucket_name,
      json_extract_scalar(requestparameters, '$.key') AS object_key,
      useragent
    FROM
      ${aws_glue_catalog_database.cloudtrail.name}.${aws_glue_catalog_table.cloudtrail_logs.name}
    WHERE
      eventsource = 's3.amazonaws.com'
      AND errorcode IS NOT NULL
      AND errorcode IN ('AccessDenied', 'NoSuchBucket', 'NoSuchKey', 'AllAccessDisabled')
      AND date >= date_format(current_date - interval '7' day, '%Y/%m/%d')
    ORDER BY
      eventtime DESC
    LIMIT 1000;
  EOT
}

# Query: Data deletion events
resource "aws_athena_named_query" "data_deletion_events" {
  name        = "data-deletion-events"
  description = "Track all object deletion events in S3"
  database    = aws_glue_catalog_database.cloudtrail.name
  workgroup   = aws_athena_workgroup.cloudtrail.id

  query = <<-EOT
    -- Data Deletion Events Query
    -- Tracks all S3 object deletions for compliance and auditing

    SELECT
      eventtime,
      useridentity.arn AS user_arn,
      useridentity.username AS username,
      useridentity.sessioncontext.sessionissuer.arn AS assumed_role,
      sourceipaddress,
      eventname,
      json_extract_scalar(requestparameters, '$.bucketName') AS bucket_name,
      json_extract_scalar(requestparameters, '$.key') AS object_key,
      json_extract_scalar(requestparameters, '$.versionId') AS version_id,
      errorcode,
      CASE
        WHEN eventname = 'DeleteObject' THEN 'Single Delete'
        WHEN eventname = 'DeleteObjects' THEN 'Bulk Delete'
        WHEN eventname = 'PutLifecycleConfiguration' THEN 'Lifecycle Policy'
        ELSE eventname
      END AS deletion_type
    FROM
      ${aws_glue_catalog_database.cloudtrail.name}.${aws_glue_catalog_table.cloudtrail_logs.name}
    WHERE
      eventsource = 's3.amazonaws.com'
      AND eventname IN ('DeleteObject', 'DeleteObjects', 'DeleteBucket')
      AND date >= date_format(current_date - interval '30' day, '%Y/%m/%d')
    ORDER BY
      eventtime DESC
    LIMIT 2000;
  EOT
}

# Query: Cross-account access
resource "aws_athena_named_query" "cross_account_access" {
  name        = "cross-account-access"
  description = "Find S3 access from external AWS accounts"
  database    = aws_glue_catalog_database.cloudtrail.name
  workgroup   = aws_athena_workgroup.cloudtrail.id

  query = <<-EOT
    -- Cross-Account Access Query
    -- Identifies S3 access from accounts other than the bucket owner

    SELECT
      eventtime,
      useridentity.accountid AS source_account,
      useridentity.arn AS user_arn,
      useridentity.type AS identity_type,
      sourceipaddress,
      eventname,
      json_extract_scalar(requestparameters, '$.bucketName') AS bucket_name,
      json_extract_scalar(requestparameters, '$.key') AS object_key,
      recipientaccountid AS target_account,
      errorcode,
      useragent
    FROM
      ${aws_glue_catalog_database.cloudtrail.name}.${aws_glue_catalog_table.cloudtrail_logs.name}
    WHERE
      eventsource = 's3.amazonaws.com'
      AND useridentity.accountid != '${data.aws_caller_identity.current.account_id}'
      AND date >= date_format(current_date - interval '7' day, '%Y/%m/%d')
    ORDER BY
      eventtime DESC
    LIMIT 1000;
  EOT
}

# Query: High volume access patterns
resource "aws_athena_named_query" "high_volume_access" {
  name        = "high-volume-access-patterns"
  description = "Identify users with unusually high S3 access volumes"
  database    = aws_glue_catalog_database.cloudtrail.name
  workgroup   = aws_athena_workgroup.cloudtrail.id

  query = <<-EOT
    -- High Volume Access Patterns Query
    -- Identifies users with potentially suspicious high-volume access

    SELECT
      useridentity.arn AS user_arn,
      sourceipaddress,
      COUNT(*) AS total_requests,
      COUNT(DISTINCT json_extract_scalar(requestparameters, '$.bucketName')) AS unique_buckets,
      COUNT(DISTINCT json_extract_scalar(requestparameters, '$.key')) AS unique_objects,
      SUM(CASE WHEN eventname LIKE 'Get%' THEN 1 ELSE 0 END) AS read_operations,
      SUM(CASE WHEN eventname LIKE 'Put%' THEN 1 ELSE 0 END) AS write_operations,
      SUM(CASE WHEN eventname LIKE 'Delete%' THEN 1 ELSE 0 END) AS delete_operations,
      SUM(CASE WHEN errorcode IS NOT NULL THEN 1 ELSE 0 END) AS failed_operations
    FROM
      ${aws_glue_catalog_database.cloudtrail.name}.${aws_glue_catalog_table.cloudtrail_logs.name}
    WHERE
      eventsource = 's3.amazonaws.com'
      AND date >= date_format(current_date - interval '1' day, '%Y/%m/%d')
    GROUP BY
      useridentity.arn,
      sourceipaddress
    HAVING
      COUNT(*) > 1000
    ORDER BY
      total_requests DESC
    LIMIT 100;
  EOT
}

# Query: Bucket policy changes
resource "aws_athena_named_query" "bucket_policy_changes" {
  name        = "bucket-policy-changes"
  description = "Track changes to S3 bucket policies and ACLs"
  database    = aws_glue_catalog_database.cloudtrail.name
  workgroup   = aws_athena_workgroup.cloudtrail.id

  query = <<-EOT
    -- Bucket Policy Changes Query
    -- Monitors security-sensitive bucket configuration changes

    SELECT
      eventtime,
      useridentity.arn AS user_arn,
      useridentity.username AS username,
      sourceipaddress,
      eventname,
      json_extract_scalar(requestparameters, '$.bucketName') AS bucket_name,
      requestparameters,
      responseelements,
      errorcode
    FROM
      ${aws_glue_catalog_database.cloudtrail.name}.${aws_glue_catalog_table.cloudtrail_logs.name}
    WHERE
      eventsource = 's3.amazonaws.com'
      AND eventname IN (
        'PutBucketPolicy',
        'DeleteBucketPolicy',
        'PutBucketAcl',
        'PutBucketPublicAccessBlock',
        'DeleteBucketPublicAccessBlock',
        'PutBucketEncryption',
        'DeleteBucketEncryption',
        'PutBucketVersioning',
        'PutBucketLogging',
        'PutBucketReplication'
      )
      AND date >= date_format(current_date - interval '30' day, '%Y/%m/%d')
    ORDER BY
      eventtime DESC
    LIMIT 500;
  EOT
}

#------------------------------------------------------------------------------
# Variables for Athena Configuration
#------------------------------------------------------------------------------

variable "projection_regions" {
  description = "AWS regions for partition projection"
  type        = list(string)
  default     = ["us-east-1", "us-west-2", "eu-west-1"]
}

variable "sensitive_buckets" {
  description = "List of sensitive bucket names to monitor"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# Outputs
#------------------------------------------------------------------------------

output "athena_workgroup_name" {
  description = "Name of the Athena workgroup"
  value       = aws_athena_workgroup.cloudtrail.name
}

output "athena_database_name" {
  description = "Name of the Glue database for CloudTrail"
  value       = aws_glue_catalog_database.cloudtrail.name
}

output "athena_table_name" {
  description = "Name of the CloudTrail logs table"
  value       = aws_glue_catalog_table.cloudtrail_logs.name
}

output "athena_results_bucket" {
  description = "S3 bucket for Athena query results"
  value       = aws_s3_bucket.athena_results.id
}

output "named_queries" {
  description = "Map of named query IDs"
  value = {
    sensitive_data_access  = aws_athena_named_query.sensitive_data_access.id
    failed_access_attempts = aws_athena_named_query.failed_access_attempts.id
    data_deletion_events   = aws_athena_named_query.data_deletion_events.id
    cross_account_access   = aws_athena_named_query.cross_account_access.id
    high_volume_access     = aws_athena_named_query.high_volume_access.id
    bucket_policy_changes  = aws_athena_named_query.bucket_policy_changes.id
  }
}
