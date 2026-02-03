#------------------------------------------------------------------------------
# GreenLang AWS Data Lake Infrastructure
# Terraform Module: data-lake - Variables
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# General Configuration
#------------------------------------------------------------------------------

variable "project_name" {
  description = "Name of the project, used for resource naming"
  type        = string
  default     = "greenlang"
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

#------------------------------------------------------------------------------
# S3 Bucket Configuration - Data Zones
#------------------------------------------------------------------------------

variable "raw_bucket_name" {
  description = "Name of the S3 bucket for raw data zone (landing area)"
  type        = string
}

variable "bronze_bucket_name" {
  description = "Name of the S3 bucket for bronze data zone (cleansed data)"
  type        = string
}

variable "silver_bucket_name" {
  description = "Name of the S3 bucket for silver data zone (conformed data)"
  type        = string
}

variable "gold_bucket_name" {
  description = "Name of the S3 bucket for gold data zone (aggregated data)"
  type        = string
}

variable "raw_data_prefix" {
  description = "S3 prefix for raw data within the raw bucket"
  type        = string
  default     = "data/"
}

variable "bronze_data_prefix" {
  description = "S3 prefix for bronze data within the bronze bucket"
  type        = string
  default     = "data/"
}

variable "silver_data_prefix" {
  description = "S3 prefix for silver data within the silver bucket"
  type        = string
  default     = "data/"
}

variable "gold_data_prefix" {
  description = "S3 prefix for gold data within the gold bucket"
  type        = string
  default     = "data/"
}

variable "etl_scripts_bucket_name" {
  description = "Name of the S3 bucket containing Glue ETL scripts"
  type        = string
}

variable "etl_temp_bucket_name" {
  description = "Name of the S3 bucket for Glue ETL temporary files"
  type        = string
}

variable "athena_results_bucket_name" {
  description = "Name of the S3 bucket for Athena query results"
  type        = string
}

#------------------------------------------------------------------------------
# KMS Encryption Configuration
#------------------------------------------------------------------------------

variable "kms_key_arn" {
  description = "ARN of the KMS key for encrypting data lake resources"
  type        = string
}

variable "encrypt_catalog_passwords" {
  description = "Whether to encrypt Glue connection passwords"
  type        = bool
  default     = true
}

variable "catalog_encryption_mode" {
  description = "Encryption mode for Glue Data Catalog (DISABLED, SSE-KMS, SSE-KMS-WITH-SERVICE-ROLE)"
  type        = string
  default     = "SSE-KMS"

  validation {
    condition     = contains(["DISABLED", "SSE-KMS", "SSE-KMS-WITH-SERVICE-ROLE"], var.catalog_encryption_mode)
    error_message = "Catalog encryption mode must be one of: DISABLED, SSE-KMS, SSE-KMS-WITH-SERVICE-ROLE."
  }
}

#------------------------------------------------------------------------------
# Glue Crawler Configuration
#------------------------------------------------------------------------------

variable "crawler_schedules" {
  description = "Cron schedules for Glue crawlers by data zone"
  type = object({
    raw    = string
    bronze = string
    silver = string
    gold   = string
  })
  default = {
    raw    = "cron(0 */6 * * ? *)"   # Every 6 hours
    bronze = "cron(0 1 * * ? *)"      # Daily at 1 AM
    silver = "cron(0 2 * * ? *)"      # Daily at 2 AM
    gold   = "cron(0 3 * * ? *)"      # Daily at 3 AM
  }
}

variable "crawler_exclusion_patterns" {
  description = "List of glob patterns to exclude from crawling"
  type        = list(string)
  default = [
    "**/_temporary/**",
    "**/_spark_metadata/**",
    "**/.spark-staging-*/**",
    "**/checkpoint/**"
  ]
}

variable "crawler_schema_change_policy" {
  description = "Schema change policy for crawlers"
  type = object({
    delete_behavior = string
    update_behavior = string
  })
  default = {
    delete_behavior = "LOG"
    update_behavior = "UPDATE_IN_DATABASE"
  }
}

variable "crawler_recrawl_behavior" {
  description = "Recrawl behavior for crawlers (CRAWL_EVERYTHING, CRAWL_NEW_FOLDERS_ONLY, CRAWL_EVENT_MODE)"
  type        = string
  default     = "CRAWL_NEW_FOLDERS_ONLY"
}

variable "crawler_table_grouping_policy" {
  description = "Table grouping policy for crawler (CombineCompatibleSchemas)"
  type        = string
  default     = "CombineCompatibleSchemas"
}

variable "glue_classifiers" {
  description = "List of custom Glue classifier names to use for crawlers"
  type        = list(string)
  default     = []
}

variable "use_lake_formation_credentials" {
  description = "Whether crawlers should use Lake Formation credentials"
  type        = bool
  default     = true
}

#------------------------------------------------------------------------------
# Glue ETL Job Configuration
#------------------------------------------------------------------------------

variable "glue_version" {
  description = "Glue version for ETL jobs"
  type        = string
  default     = "4.0"
}

variable "worker_type" {
  description = "Worker type for Glue jobs (G.1X, G.2X, G.025X)"
  type        = string
  default     = "G.1X"
}

variable "number_of_workers" {
  description = "Number of workers for Glue jobs"
  type        = number
  default     = 2
}

variable "max_retries" {
  description = "Maximum number of retries for Glue jobs"
  type        = number
  default     = 1
}

variable "job_timeout" {
  description = "Timeout in minutes for Glue jobs"
  type        = number
  default     = 60
}

variable "enable_job_bookmarks" {
  description = "Enable job bookmarks for incremental processing"
  type        = bool
  default     = true
}

variable "enable_spark_ui" {
  description = "Enable Spark UI for Glue jobs"
  type        = bool
  default     = true
}

variable "spark_ui_bucket" {
  description = "S3 bucket for Spark UI logs"
  type        = string
  default     = ""
}

variable "enable_continuous_logging" {
  description = "Enable continuous CloudWatch logging for Glue jobs"
  type        = bool
  default     = true
}

variable "output_format" {
  description = "Output format for transformed data (parquet, orc)"
  type        = string
  default     = "parquet"

  validation {
    condition     = contains(["parquet", "orc"], var.output_format)
    error_message = "Output format must be one of: parquet, orc."
  }
}

variable "compression_codec" {
  description = "Compression codec for output files (snappy, gzip, zstd, lzo)"
  type        = string
  default     = "snappy"
}

#------------------------------------------------------------------------------
# Athena Configuration
#------------------------------------------------------------------------------

variable "athena_workgroup_config" {
  description = "Configuration for Athena workgroups"
  type = object({
    primary = object({
      bytes_scanned_cutoff           = number
      enforce_workgroup_configuration = bool
      publish_cloudwatch_metrics     = bool
      requester_pays                 = bool
    })
    analytics = object({
      bytes_scanned_cutoff           = number
      enforce_workgroup_configuration = bool
      publish_cloudwatch_metrics     = bool
      requester_pays                 = bool
    })
    reporting = object({
      bytes_scanned_cutoff           = number
      enforce_workgroup_configuration = bool
      publish_cloudwatch_metrics     = bool
      requester_pays                 = bool
    })
  })
  default = {
    primary = {
      bytes_scanned_cutoff           = 10737418240  # 10 GB
      enforce_workgroup_configuration = true
      publish_cloudwatch_metrics     = true
      requester_pays                 = false
    }
    analytics = {
      bytes_scanned_cutoff           = 107374182400  # 100 GB
      enforce_workgroup_configuration = true
      publish_cloudwatch_metrics     = true
      requester_pays                 = false
    }
    reporting = {
      bytes_scanned_cutoff           = 53687091200  # 50 GB
      enforce_workgroup_configuration = true
      publish_cloudwatch_metrics     = true
      requester_pays                 = false
    }
  }
}

variable "athena_query_result_encryption" {
  description = "Encryption option for Athena query results (SSE_S3, SSE_KMS, CSE_KMS)"
  type        = string
  default     = "SSE_KMS"
}

variable "athena_query_result_cache_ttl" {
  description = "Query result cache TTL in minutes (0 to disable)"
  type        = number
  default     = 60
}

#------------------------------------------------------------------------------
# Lake Formation Configuration
#------------------------------------------------------------------------------

variable "lake_formation_admins" {
  description = "List of IAM principals to be Lake Formation admins"
  type        = list(string)
  default     = []
}

variable "lake_formation_trusted_resource_owners" {
  description = "List of trusted resource owners for Lake Formation"
  type        = list(string)
  default     = []
}

variable "allow_external_data_filtering" {
  description = "Allow external data filtering in Lake Formation"
  type        = bool
  default     = false
}

variable "authorized_session_tags" {
  description = "List of authorized session tag values for Lake Formation"
  type        = list(string)
  default     = []
}

variable "data_engineer_principals" {
  description = "List of IAM principals for data engineers (full access to all zones)"
  type        = list(string)
  default     = []
}

variable "data_analyst_principals" {
  description = "List of IAM principals for data analysts (read access to gold zone)"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# Cross-Account Access Configuration
#------------------------------------------------------------------------------

variable "enable_cross_account_access" {
  description = "Enable cross-account access to the data lake"
  type        = bool
  default     = false
}

variable "cross_account_principals" {
  description = "List of AWS account ARNs or principal ARNs for cross-account access"
  type        = list(string)
  default     = []
}

variable "cross_account_permissions" {
  description = "List of permissions to grant for cross-account access"
  type        = list(string)
  default     = ["DESCRIBE", "SELECT"]
}

variable "cross_account_permissions_grantable" {
  description = "Whether cross-account principals can grant permissions to others"
  type        = bool
  default     = false
}

#------------------------------------------------------------------------------
# EKS IRSA Configuration
#------------------------------------------------------------------------------

variable "enable_eks_irsa" {
  description = "Enable IRSA for EKS pods to access Athena"
  type        = bool
  default     = false
}

variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS OIDC provider for IRSA"
  type        = string
  default     = ""
}

variable "eks_oidc_provider_url" {
  description = "URL of the EKS OIDC provider (without https://)"
  type        = string
  default     = ""
}

variable "eks_namespace" {
  description = "Kubernetes namespace for data lake access"
  type        = string
  default     = "data-analytics"
}

variable "eks_service_account_name" {
  description = "Kubernetes service account name for data lake access"
  type        = string
  default     = "athena-query-sa"
}

#------------------------------------------------------------------------------
# Monitoring Configuration
#------------------------------------------------------------------------------

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms for data lake components"
  type        = bool
  default     = true
}

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications"
  type        = string
  default     = ""
}

variable "crawler_failure_threshold" {
  description = "Number of crawler failures before alerting"
  type        = number
  default     = 2
}

variable "job_failure_threshold" {
  description = "Number of job failures before alerting"
  type        = number
  default     = 2
}
