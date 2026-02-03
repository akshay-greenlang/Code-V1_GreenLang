#------------------------------------------------------------------------------
# GreenLang AWS Data Lake Infrastructure
# Terraform Module: data-lake - Athena Resources
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Athena Workgroup - Primary
#------------------------------------------------------------------------------

resource "aws_athena_workgroup" "primary" {
  name        = "${local.name_prefix}-primary"
  description = "Primary workgroup for general Athena queries"
  state       = "ENABLED"

  configuration {
    enforce_workgroup_configuration    = var.athena_workgroup_config.primary.enforce_workgroup_configuration
    publish_cloudwatch_metrics_enabled = var.athena_workgroup_config.primary.publish_cloudwatch_metrics
    requester_pays_enabled             = var.athena_workgroup_config.primary.requester_pays

    bytes_scanned_cutoff_per_query = var.athena_workgroup_config.primary.bytes_scanned_cutoff

    result_configuration {
      output_location = "s3://${var.athena_results_bucket_name}/query-results/primary/"

      encryption_configuration {
        encryption_option = var.athena_query_result_encryption
        kms_key_arn       = var.athena_query_result_encryption != "SSE_S3" ? var.kms_key_arn : null
      }

      acl_configuration {
        s3_acl_option = "BUCKET_OWNER_FULL_CONTROL"
      }
    }

    engine_version {
      selected_engine_version = "Athena engine version 3"
    }
  }

  tags = merge(local.common_tags, {
    WorkgroupType = "primary"
  })
}

#------------------------------------------------------------------------------
# Athena Workgroup - Analytics
#------------------------------------------------------------------------------

resource "aws_athena_workgroup" "analytics" {
  name        = "${local.name_prefix}-analytics"
  description = "Analytics workgroup for data science and ML queries"
  state       = "ENABLED"

  configuration {
    enforce_workgroup_configuration    = var.athena_workgroup_config.analytics.enforce_workgroup_configuration
    publish_cloudwatch_metrics_enabled = var.athena_workgroup_config.analytics.publish_cloudwatch_metrics
    requester_pays_enabled             = var.athena_workgroup_config.analytics.requester_pays

    bytes_scanned_cutoff_per_query = var.athena_workgroup_config.analytics.bytes_scanned_cutoff

    result_configuration {
      output_location = "s3://${var.athena_results_bucket_name}/query-results/analytics/"

      encryption_configuration {
        encryption_option = var.athena_query_result_encryption
        kms_key_arn       = var.athena_query_result_encryption != "SSE_S3" ? var.kms_key_arn : null
      }

      acl_configuration {
        s3_acl_option = "BUCKET_OWNER_FULL_CONTROL"
      }
    }

    engine_version {
      selected_engine_version = "Athena engine version 3"
    }
  }

  tags = merge(local.common_tags, {
    WorkgroupType = "analytics"
  })
}

#------------------------------------------------------------------------------
# Athena Workgroup - Reporting
#------------------------------------------------------------------------------

resource "aws_athena_workgroup" "reporting" {
  name        = "${local.name_prefix}-reporting"
  description = "Reporting workgroup for BI dashboards and scheduled reports"
  state       = "ENABLED"

  configuration {
    enforce_workgroup_configuration    = var.athena_workgroup_config.reporting.enforce_workgroup_configuration
    publish_cloudwatch_metrics_enabled = var.athena_workgroup_config.reporting.publish_cloudwatch_metrics
    requester_pays_enabled             = var.athena_workgroup_config.reporting.requester_pays

    bytes_scanned_cutoff_per_query = var.athena_workgroup_config.reporting.bytes_scanned_cutoff

    result_configuration {
      output_location = "s3://${var.athena_results_bucket_name}/query-results/reporting/"

      encryption_configuration {
        encryption_option = var.athena_query_result_encryption
        kms_key_arn       = var.athena_query_result_encryption != "SSE_S3" ? var.kms_key_arn : null
      }

      acl_configuration {
        s3_acl_option = "BUCKET_OWNER_FULL_CONTROL"
      }
    }

    engine_version {
      selected_engine_version = "Athena engine version 3"
    }
  }

  tags = merge(local.common_tags, {
    WorkgroupType = "reporting"
  })
}

#------------------------------------------------------------------------------
# Athena Named Queries - Common Operations
#------------------------------------------------------------------------------

resource "aws_athena_named_query" "show_databases" {
  name        = "${local.name_prefix}-show-databases"
  description = "List all data lake databases"
  workgroup   = aws_athena_workgroup.primary.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- List all databases in the data lake
    SHOW DATABASES;
  EOT
}

resource "aws_athena_named_query" "show_tables_gold" {
  name        = "${local.name_prefix}-show-tables-gold"
  description = "List all tables in the gold zone"
  workgroup   = aws_athena_workgroup.primary.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- List all tables in the gold (aggregated) data zone
    SHOW TABLES IN ${aws_glue_catalog_database.gold.name};
  EOT
}

resource "aws_athena_named_query" "table_partitions" {
  name        = "${local.name_prefix}-show-partitions"
  description = "Show partitions for a table (replace TABLE_NAME)"
  workgroup   = aws_athena_workgroup.analytics.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- Show partitions for a specific table
    -- Replace TABLE_NAME with the actual table name
    SHOW PARTITIONS ${aws_glue_catalog_database.gold.name}.TABLE_NAME;
  EOT
}

resource "aws_athena_named_query" "table_statistics" {
  name        = "${local.name_prefix}-table-statistics"
  description = "Generate table statistics for query optimization"
  workgroup   = aws_athena_workgroup.analytics.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- Generate table statistics for better query planning
    -- Replace TABLE_NAME with the actual table name
    ANALYZE ${aws_glue_catalog_database.gold.name}.TABLE_NAME COMPUTE STATISTICS;
  EOT
}

resource "aws_athena_named_query" "repair_table" {
  name        = "${local.name_prefix}-repair-table"
  description = "Repair table partitions after data changes"
  workgroup   = aws_athena_workgroup.primary.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- Repair table to add new partitions discovered in S3
    -- Replace TABLE_NAME with the actual table name
    MSCK REPAIR TABLE ${aws_glue_catalog_database.gold.name}.TABLE_NAME;
  EOT
}

resource "aws_athena_named_query" "data_quality_check" {
  name        = "${local.name_prefix}-data-quality-check"
  description = "Basic data quality checks"
  workgroup   = aws_athena_workgroup.analytics.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- Data quality checks template
    -- Replace TABLE_NAME and adjust columns as needed

    SELECT
      'Total Records' as metric,
      COUNT(*) as value
    FROM ${aws_glue_catalog_database.gold.name}.TABLE_NAME

    UNION ALL

    SELECT
      'Null Primary Keys' as metric,
      COUNT(*) as value
    FROM ${aws_glue_catalog_database.gold.name}.TABLE_NAME
    WHERE id IS NULL

    UNION ALL

    SELECT
      'Duplicate Records' as metric,
      COUNT(*) - COUNT(DISTINCT id) as value
    FROM ${aws_glue_catalog_database.gold.name}.TABLE_NAME;
  EOT
}

resource "aws_athena_named_query" "partition_projection_example" {
  name        = "${local.name_prefix}-partition-projection-ddl"
  description = "Example DDL with partition projection for performance"
  workgroup   = aws_athena_workgroup.analytics.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- Example: Create table with partition projection for optimized queries
    -- This eliminates the need for MSCK REPAIR TABLE

    CREATE EXTERNAL TABLE IF NOT EXISTS ${aws_glue_catalog_database.gold.name}.events_partitioned (
      event_id STRING,
      user_id STRING,
      event_type STRING,
      event_data STRING,
      created_at TIMESTAMP
    )
    PARTITIONED BY (
      year STRING,
      month STRING,
      day STRING
    )
    STORED AS PARQUET
    LOCATION 's3://${var.gold_bucket_name}/events/'
    TBLPROPERTIES (
      'projection.enabled' = 'true',
      'projection.year.type' = 'integer',
      'projection.year.range' = '2020,2030',
      'projection.month.type' = 'integer',
      'projection.month.range' = '1,12',
      'projection.month.digits' = '2',
      'projection.day.type' = 'integer',
      'projection.day.range' = '1,31',
      'projection.day.digits' = '2',
      'storage.location.template' = 's3://${var.gold_bucket_name}/events/year=$${year}/month=$${month}/day=$${day}/',
      'parquet.compression' = 'SNAPPY'
    );
  EOT
}

resource "aws_athena_named_query" "ctas_optimized_table" {
  name        = "${local.name_prefix}-ctas-optimized"
  description = "Create optimized table using CTAS"
  workgroup   = aws_athena_workgroup.analytics.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- Create optimized table using CTAS (Create Table As Select)
    -- This creates a new Parquet table with partitioning and compression

    CREATE TABLE ${aws_glue_catalog_database.gold.name}.optimized_data
    WITH (
      format = 'PARQUET',
      parquet_compression = 'SNAPPY',
      partitioned_by = ARRAY['year', 'month'],
      external_location = 's3://${var.gold_bucket_name}/optimized_data/',
      bucketed_by = ARRAY['user_id'],
      bucket_count = 10
    ) AS
    SELECT
      *,
      year(created_at) as year,
      month(created_at) as month
    FROM ${aws_glue_catalog_database.silver.name}.source_data
    WHERE created_at >= DATE '2024-01-01';
  EOT
}

resource "aws_athena_named_query" "daily_summary_report" {
  name        = "${local.name_prefix}-daily-summary"
  description = "Daily summary aggregation query"
  workgroup   = aws_athena_workgroup.reporting.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- Daily summary report template
    -- Adjust the date filter and columns based on your data

    WITH daily_metrics AS (
      SELECT
        DATE(created_at) as report_date,
        COUNT(*) as total_events,
        COUNT(DISTINCT user_id) as unique_users,
        AVG(duration) as avg_duration
      FROM ${aws_glue_catalog_database.gold.name}.TABLE_NAME
      WHERE DATE(created_at) = CURRENT_DATE - INTERVAL '1' DAY
      GROUP BY DATE(created_at)
    )
    SELECT
      report_date,
      total_events,
      unique_users,
      ROUND(avg_duration, 2) as avg_duration,
      ROUND(total_events * 1.0 / unique_users, 2) as events_per_user
    FROM daily_metrics;
  EOT
}

resource "aws_athena_named_query" "cost_estimation" {
  name        = "${local.name_prefix}-cost-estimation"
  description = "Estimate query cost by analyzing data scanned"
  workgroup   = aws_athena_workgroup.analytics.name
  database    = aws_glue_catalog_database.gold.name

  query = <<-EOT
    -- Estimate data scanned for a query (use EXPLAIN before running expensive queries)
    -- Athena pricing: $5 per TB scanned (as of 2024)

    EXPLAIN ANALYZE
    SELECT COUNT(*)
    FROM ${aws_glue_catalog_database.gold.name}.TABLE_NAME
    WHERE year = '2024';
  EOT
}
