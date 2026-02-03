#------------------------------------------------------------------------------
# GreenLang AWS Data Lake Infrastructure
# Terraform Module: data-lake - Glue ETL Jobs
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Glue ETL Job: Raw to Bronze Transformation
#------------------------------------------------------------------------------

resource "aws_glue_job" "raw_to_bronze" {
  name     = "${local.name_prefix}-raw-to-bronze-etl"
  role_arn = aws_iam_role.glue_etl.arn

  description = "ETL job to transform raw data to bronze zone with schema validation and cleansing"

  glue_version      = var.glue_version
  worker_type       = var.worker_type
  number_of_workers = var.number_of_workers
  max_retries       = var.max_retries
  timeout           = var.job_timeout

  command {
    script_location = "s3://${var.etl_scripts_bucket_name}/scripts/raw_to_bronze.py"
    python_version  = "3"
    name            = "glueetl"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--job-bookmark-option"              = var.enable_job_bookmarks ? "job-bookmark-enable" : "job-bookmark-disable"
    "--TempDir"                          = "s3://${var.etl_temp_bucket_name}/temp/raw-to-bronze/"
    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = tostring(var.enable_continuous_logging)
    "--enable-spark-ui"                  = tostring(var.enable_spark_ui)
    "--spark-event-logs-path"            = var.enable_spark_ui ? "s3://${var.spark_ui_bucket != "" ? var.spark_ui_bucket : var.etl_temp_bucket_name}/spark-ui/raw-to-bronze/" : ""

    # Custom job arguments
    "--source_database"         = aws_glue_catalog_database.raw.name
    "--target_database"         = aws_glue_catalog_database.bronze.name
    "--source_bucket"           = var.raw_bucket_name
    "--target_bucket"           = var.bronze_bucket_name
    "--output_format"           = var.output_format
    "--compression"             = var.compression_codec
    "--enable_partition_pruning" = "true"
    "--additional-python-modules" = "pyarrow==14.0.0,pandas==2.0.3"
  }

  execution_property {
    max_concurrent_runs = 1
  }

  notification_property {
    notify_delay_after = 60
  }

  tags = merge(local.common_tags, {
    ETLJob       = "raw-to-bronze"
    SourceZone   = "raw"
    TargetZone   = "bronze"
  })
}

#------------------------------------------------------------------------------
# Glue ETL Job: Bronze to Silver Transformation
#------------------------------------------------------------------------------

resource "aws_glue_job" "bronze_to_silver" {
  name     = "${local.name_prefix}-bronze-to-silver-etl"
  role_arn = aws_iam_role.glue_etl.arn

  description = "ETL job to transform bronze data to silver zone with conformance and enrichment"

  glue_version      = var.glue_version
  worker_type       = var.worker_type
  number_of_workers = var.number_of_workers
  max_retries       = var.max_retries
  timeout           = var.job_timeout

  command {
    script_location = "s3://${var.etl_scripts_bucket_name}/scripts/bronze_to_silver.py"
    python_version  = "3"
    name            = "glueetl"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--job-bookmark-option"              = var.enable_job_bookmarks ? "job-bookmark-enable" : "job-bookmark-disable"
    "--TempDir"                          = "s3://${var.etl_temp_bucket_name}/temp/bronze-to-silver/"
    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = tostring(var.enable_continuous_logging)
    "--enable-spark-ui"                  = tostring(var.enable_spark_ui)
    "--spark-event-logs-path"            = var.enable_spark_ui ? "s3://${var.spark_ui_bucket != "" ? var.spark_ui_bucket : var.etl_temp_bucket_name}/spark-ui/bronze-to-silver/" : ""

    # Custom job arguments
    "--source_database"         = aws_glue_catalog_database.bronze.name
    "--target_database"         = aws_glue_catalog_database.silver.name
    "--source_bucket"           = var.bronze_bucket_name
    "--target_bucket"           = var.silver_bucket_name
    "--output_format"           = var.output_format
    "--compression"             = var.compression_codec
    "--enable_partition_pruning" = "true"
    "--enable_deduplication"    = "true"
    "--additional-python-modules" = "pyarrow==14.0.0,pandas==2.0.3,great-expectations==0.17.0"
  }

  execution_property {
    max_concurrent_runs = 1
  }

  notification_property {
    notify_delay_after = 60
  }

  tags = merge(local.common_tags, {
    ETLJob       = "bronze-to-silver"
    SourceZone   = "bronze"
    TargetZone   = "silver"
  })
}

#------------------------------------------------------------------------------
# Glue ETL Job: Silver to Gold Aggregation
#------------------------------------------------------------------------------

resource "aws_glue_job" "silver_to_gold" {
  name     = "${local.name_prefix}-silver-to-gold-etl"
  role_arn = aws_iam_role.glue_etl.arn

  description = "ETL job to aggregate silver data into gold zone for analytics and reporting"

  glue_version      = var.glue_version
  worker_type       = var.worker_type
  number_of_workers = var.number_of_workers + 1  # Additional worker for aggregations
  max_retries       = var.max_retries
  timeout           = var.job_timeout

  command {
    script_location = "s3://${var.etl_scripts_bucket_name}/scripts/silver_to_gold.py"
    python_version  = "3"
    name            = "glueetl"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--job-bookmark-option"              = var.enable_job_bookmarks ? "job-bookmark-enable" : "job-bookmark-disable"
    "--TempDir"                          = "s3://${var.etl_temp_bucket_name}/temp/silver-to-gold/"
    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = tostring(var.enable_continuous_logging)
    "--enable-spark-ui"                  = tostring(var.enable_spark_ui)
    "--spark-event-logs-path"            = var.enable_spark_ui ? "s3://${var.spark_ui_bucket != "" ? var.spark_ui_bucket : var.etl_temp_bucket_name}/spark-ui/silver-to-gold/" : ""

    # Custom job arguments
    "--source_database"         = aws_glue_catalog_database.silver.name
    "--target_database"         = aws_glue_catalog_database.gold.name
    "--source_bucket"           = var.silver_bucket_name
    "--target_bucket"           = var.gold_bucket_name
    "--output_format"           = var.output_format
    "--compression"             = var.compression_codec
    "--enable_partition_pruning" = "true"
    "--enable_aggregations"     = "true"
    "--additional-python-modules" = "pyarrow==14.0.0,pandas==2.0.3"
  }

  execution_property {
    max_concurrent_runs = 1
  }

  notification_property {
    notify_delay_after = 60
  }

  tags = merge(local.common_tags, {
    ETLJob       = "silver-to-gold"
    SourceZone   = "silver"
    TargetZone   = "gold"
  })
}

#------------------------------------------------------------------------------
# Glue Workflow for Full Pipeline
#------------------------------------------------------------------------------

resource "aws_glue_workflow" "data_pipeline" {
  name        = "${local.name_prefix}-data-pipeline"
  description = "Complete data lake ETL pipeline workflow"

  default_run_properties = {
    environment = var.environment
    project     = var.project_name
  }

  tags = local.common_tags
}

#------------------------------------------------------------------------------
# Glue Triggers - Scheduled
#------------------------------------------------------------------------------

resource "aws_glue_trigger" "raw_to_bronze_scheduled" {
  name          = "${local.name_prefix}-raw-to-bronze-trigger"
  type          = "SCHEDULED"
  schedule      = "cron(0 4 * * ? *)"  # Daily at 4 AM UTC
  workflow_name = aws_glue_workflow.data_pipeline.name

  actions {
    job_name = aws_glue_job.raw_to_bronze.name
    arguments = {
      "--run_date" = "$${CURRENT_DATE}"
    }
  }

  tags = local.common_tags
}

resource "aws_glue_trigger" "bronze_to_silver_conditional" {
  name          = "${local.name_prefix}-bronze-to-silver-trigger"
  type          = "CONDITIONAL"
  workflow_name = aws_glue_workflow.data_pipeline.name

  predicate {
    conditions {
      job_name = aws_glue_job.raw_to_bronze.name
      state    = "SUCCEEDED"
    }
  }

  actions {
    job_name = aws_glue_job.bronze_to_silver.name
    arguments = {
      "--run_date" = "$${CURRENT_DATE}"
    }
  }

  tags = local.common_tags
}

resource "aws_glue_trigger" "silver_to_gold_conditional" {
  name          = "${local.name_prefix}-silver-to-gold-trigger"
  type          = "CONDITIONAL"
  workflow_name = aws_glue_workflow.data_pipeline.name

  predicate {
    conditions {
      job_name = aws_glue_job.bronze_to_silver.name
      state    = "SUCCEEDED"
    }
  }

  actions {
    job_name = aws_glue_job.silver_to_gold.name
    arguments = {
      "--run_date" = "$${CURRENT_DATE}"
    }
  }

  tags = local.common_tags
}

#------------------------------------------------------------------------------
# Glue Triggers - On-Demand
#------------------------------------------------------------------------------

resource "aws_glue_trigger" "raw_to_bronze_on_demand" {
  name = "${local.name_prefix}-raw-to-bronze-on-demand"
  type = "ON_DEMAND"

  actions {
    job_name = aws_glue_job.raw_to_bronze.name
  }

  tags = local.common_tags
}

resource "aws_glue_trigger" "bronze_to_silver_on_demand" {
  name = "${local.name_prefix}-bronze-to-silver-on-demand"
  type = "ON_DEMAND"

  actions {
    job_name = aws_glue_job.bronze_to_silver.name
  }

  tags = local.common_tags
}

resource "aws_glue_trigger" "silver_to_gold_on_demand" {
  name = "${local.name_prefix}-silver-to-gold-on-demand"
  type = "ON_DEMAND"

  actions {
    job_name = aws_glue_job.silver_to_gold.name
  }

  tags = local.common_tags
}

#------------------------------------------------------------------------------
# Glue Trigger - Crawler After ETL
#------------------------------------------------------------------------------

resource "aws_glue_trigger" "gold_crawler_after_etl" {
  name          = "${local.name_prefix}-gold-crawler-trigger"
  type          = "CONDITIONAL"
  workflow_name = aws_glue_workflow.data_pipeline.name

  predicate {
    conditions {
      job_name = aws_glue_job.silver_to_gold.name
      state    = "SUCCEEDED"
    }
  }

  actions {
    crawler_name = aws_glue_crawler.gold.name
  }

  tags = local.common_tags
}
