#------------------------------------------------------------------------------
# GreenLang AWS Data Lake Infrastructure
# Terraform Module: data-lake - Outputs
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Glue Data Catalog Database Outputs
#------------------------------------------------------------------------------

output "glue_database_raw_name" {
  description = "Name of the raw data zone Glue database"
  value       = aws_glue_catalog_database.raw.name
}

output "glue_database_raw_arn" {
  description = "ARN of the raw data zone Glue database"
  value       = aws_glue_catalog_database.raw.arn
}

output "glue_database_bronze_name" {
  description = "Name of the bronze data zone Glue database"
  value       = aws_glue_catalog_database.bronze.name
}

output "glue_database_bronze_arn" {
  description = "ARN of the bronze data zone Glue database"
  value       = aws_glue_catalog_database.bronze.arn
}

output "glue_database_silver_name" {
  description = "Name of the silver data zone Glue database"
  value       = aws_glue_catalog_database.silver.name
}

output "glue_database_silver_arn" {
  description = "ARN of the silver data zone Glue database"
  value       = aws_glue_catalog_database.silver.arn
}

output "glue_database_gold_name" {
  description = "Name of the gold data zone Glue database"
  value       = aws_glue_catalog_database.gold.name
}

output "glue_database_gold_arn" {
  description = "ARN of the gold data zone Glue database"
  value       = aws_glue_catalog_database.gold.arn
}

output "glue_databases" {
  description = "Map of all Glue database names by data zone"
  value = {
    raw    = aws_glue_catalog_database.raw.name
    bronze = aws_glue_catalog_database.bronze.name
    silver = aws_glue_catalog_database.silver.name
    gold   = aws_glue_catalog_database.gold.name
  }
}

#------------------------------------------------------------------------------
# Glue Crawler Outputs
#------------------------------------------------------------------------------

output "crawler_raw_name" {
  description = "Name of the raw data zone crawler"
  value       = aws_glue_crawler.raw.name
}

output "crawler_raw_arn" {
  description = "ARN of the raw data zone crawler"
  value       = aws_glue_crawler.raw.arn
}

output "crawler_bronze_name" {
  description = "Name of the bronze data zone crawler"
  value       = aws_glue_crawler.bronze.name
}

output "crawler_bronze_arn" {
  description = "ARN of the bronze data zone crawler"
  value       = aws_glue_crawler.bronze.arn
}

output "crawler_silver_name" {
  description = "Name of the silver data zone crawler"
  value       = aws_glue_crawler.silver.name
}

output "crawler_silver_arn" {
  description = "ARN of the silver data zone crawler"
  value       = aws_glue_crawler.silver.arn
}

output "crawler_gold_name" {
  description = "Name of the gold data zone crawler"
  value       = aws_glue_crawler.gold.name
}

output "crawler_gold_arn" {
  description = "ARN of the gold data zone crawler"
  value       = aws_glue_crawler.gold.arn
}

output "crawlers" {
  description = "Map of all crawler names by data zone"
  value = {
    raw    = aws_glue_crawler.raw.name
    bronze = aws_glue_crawler.bronze.name
    silver = aws_glue_crawler.silver.name
    gold   = aws_glue_crawler.gold.name
  }
}

#------------------------------------------------------------------------------
# Glue ETL Job Outputs
#------------------------------------------------------------------------------

output "etl_job_raw_to_bronze_name" {
  description = "Name of the raw to bronze ETL job"
  value       = aws_glue_job.raw_to_bronze.name
}

output "etl_job_raw_to_bronze_arn" {
  description = "ARN of the raw to bronze ETL job"
  value       = aws_glue_job.raw_to_bronze.arn
}

output "etl_job_bronze_to_silver_name" {
  description = "Name of the bronze to silver ETL job"
  value       = aws_glue_job.bronze_to_silver.name
}

output "etl_job_bronze_to_silver_arn" {
  description = "ARN of the bronze to silver ETL job"
  value       = aws_glue_job.bronze_to_silver.arn
}

output "etl_job_silver_to_gold_name" {
  description = "Name of the silver to gold ETL job"
  value       = aws_glue_job.silver_to_gold.name
}

output "etl_job_silver_to_gold_arn" {
  description = "ARN of the silver to gold ETL job"
  value       = aws_glue_job.silver_to_gold.arn
}

output "etl_jobs" {
  description = "Map of all ETL job names by transformation"
  value = {
    raw_to_bronze    = aws_glue_job.raw_to_bronze.name
    bronze_to_silver = aws_glue_job.bronze_to_silver.name
    silver_to_gold   = aws_glue_job.silver_to_gold.name
  }
}

#------------------------------------------------------------------------------
# Athena Workgroup Outputs
#------------------------------------------------------------------------------

output "athena_workgroup_primary_name" {
  description = "Name of the primary Athena workgroup"
  value       = aws_athena_workgroup.primary.name
}

output "athena_workgroup_primary_arn" {
  description = "ARN of the primary Athena workgroup"
  value       = aws_athena_workgroup.primary.arn
}

output "athena_workgroup_analytics_name" {
  description = "Name of the analytics Athena workgroup"
  value       = aws_athena_workgroup.analytics.name
}

output "athena_workgroup_analytics_arn" {
  description = "ARN of the analytics Athena workgroup"
  value       = aws_athena_workgroup.analytics.arn
}

output "athena_workgroup_reporting_name" {
  description = "Name of the reporting Athena workgroup"
  value       = aws_athena_workgroup.reporting.name
}

output "athena_workgroup_reporting_arn" {
  description = "ARN of the reporting Athena workgroup"
  value       = aws_athena_workgroup.reporting.arn
}

output "athena_workgroups" {
  description = "Map of all Athena workgroup names"
  value = {
    primary   = aws_athena_workgroup.primary.name
    analytics = aws_athena_workgroup.analytics.name
    reporting = aws_athena_workgroup.reporting.name
  }
}

output "athena_query_results_location" {
  description = "S3 location for Athena query results"
  value       = "s3://${var.athena_results_bucket_name}/query-results/"
}

#------------------------------------------------------------------------------
# Lake Formation Outputs
#------------------------------------------------------------------------------

output "lake_formation_admins" {
  description = "List of Lake Formation administrator principals"
  value       = var.lake_formation_admins
}

output "lake_formation_role_arn" {
  description = "ARN of the Lake Formation IAM role"
  value       = aws_iam_role.lake_formation.arn
}

output "lake_formation_registered_locations" {
  description = "Map of registered S3 locations with Lake Formation"
  value = {
    raw    = aws_lakeformation_resource.raw.arn
    bronze = aws_lakeformation_resource.bronze.arn
    silver = aws_lakeformation_resource.silver.arn
    gold   = aws_lakeformation_resource.gold.arn
  }
}

#------------------------------------------------------------------------------
# IAM Role Outputs
#------------------------------------------------------------------------------

output "glue_crawler_role_arn" {
  description = "ARN of the IAM role for Glue crawlers"
  value       = aws_iam_role.glue_crawler.arn
}

output "glue_crawler_role_name" {
  description = "Name of the IAM role for Glue crawlers"
  value       = aws_iam_role.glue_crawler.name
}

output "glue_etl_role_arn" {
  description = "ARN of the IAM role for Glue ETL jobs"
  value       = aws_iam_role.glue_etl.arn
}

output "glue_etl_role_name" {
  description = "Name of the IAM role for Glue ETL jobs"
  value       = aws_iam_role.glue_etl.name
}

output "athena_execution_role_arn" {
  description = "ARN of the IAM role for Athena query execution"
  value       = aws_iam_role.athena_execution.arn
}

output "athena_execution_role_name" {
  description = "Name of the IAM role for Athena query execution"
  value       = aws_iam_role.athena_execution.name
}

output "eks_athena_role_arn" {
  description = "ARN of the IRSA role for EKS pods to access Athena"
  value       = var.enable_eks_irsa ? aws_iam_role.eks_athena[0].arn : null
}

#------------------------------------------------------------------------------
# Connection Information
#------------------------------------------------------------------------------

output "data_lake_connection_info" {
  description = "Connection information for data lake consumers"
  value = {
    catalog_id        = data.aws_caller_identity.current.account_id
    region            = data.aws_region.current.name
    databases         = {
      raw    = aws_glue_catalog_database.raw.name
      bronze = aws_glue_catalog_database.bronze.name
      silver = aws_glue_catalog_database.silver.name
      gold   = aws_glue_catalog_database.gold.name
    }
    athena_workgroup  = aws_athena_workgroup.primary.name
    results_location  = "s3://${var.athena_results_bucket_name}/query-results/"
  }
}
