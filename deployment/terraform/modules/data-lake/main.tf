#------------------------------------------------------------------------------
# GreenLang AWS Data Lake Infrastructure
# Terraform Module: data-lake
#
# This module provisions a comprehensive AWS Data Lake architecture with:
# - AWS Glue Data Catalog databases for data zones (raw, bronze, silver, gold)
# - Glue crawlers for automatic schema discovery
# - Glue ETL jobs for data transformation pipelines
# - Amazon Athena workgroups for analytics
# - AWS Lake Formation for centralized governance
#------------------------------------------------------------------------------

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

#------------------------------------------------------------------------------
# Local Variables
#------------------------------------------------------------------------------

locals {
  name_prefix = "${var.project_name}-${var.environment}"

  data_zones = {
    raw    = "raw"
    bronze = "bronze"
    silver = "silver"
    gold   = "gold"
  }

  common_tags = merge(var.tags, {
    Module      = "data-lake"
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  })

  # Glue catalog encryption configuration
  glue_catalog_encryption_settings = {
    connection_password_encryption = {
      aws_kms_key_id                       = var.kms_key_arn
      return_connection_password_encrypted = true
    }
    encryption_at_rest = {
      catalog_encryption_mode = "SSE-KMS"
      sse_aws_kms_key_id      = var.kms_key_arn
    }
  }
}

#------------------------------------------------------------------------------
# AWS Glue Data Catalog - Security Settings
#------------------------------------------------------------------------------

resource "aws_glue_data_catalog_encryption_settings" "main" {
  data_catalog_encryption_settings {
    connection_password_encryption {
      aws_kms_key_id                       = var.kms_key_arn
      return_connection_password_encrypted = var.encrypt_catalog_passwords
    }

    encryption_at_rest {
      catalog_encryption_mode = var.catalog_encryption_mode
      sse_aws_kms_key_id      = var.catalog_encryption_mode == "SSE-KMS" ? var.kms_key_arn : null
    }
  }
}

resource "aws_glue_resource_policy" "main" {
  count = var.enable_cross_account_access ? 1 : 0

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCrossAccountCatalogAccess"
        Effect = "Allow"
        Principal = {
          AWS = var.cross_account_principals
        }
        Action = [
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetTable",
          "glue:GetTables",
          "glue:GetPartition",
          "glue:GetPartitions",
          "glue:BatchGetPartition"
        ]
        Resource = [
          "arn:aws:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:aws:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:database/*",
          "arn:aws:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:table/*"
        ]
      }
    ]
  })
}

#------------------------------------------------------------------------------
# AWS Glue Data Catalog Databases - Data Zones
#------------------------------------------------------------------------------

# Raw Data Zone - Landing zone for ingested data
resource "aws_glue_catalog_database" "raw" {
  name         = "${local.name_prefix}_raw"
  description  = "Raw data zone - Landing area for ingested data before any transformation"
  catalog_id   = data.aws_caller_identity.current.account_id
  location_uri = "s3://${var.raw_bucket_name}/"

  create_table_default_permission {
    permissions = ["ALL"]

    principal {
      data_lake_principal_identifier = "IAM_ALLOWED_PRINCIPALS"
    }
  }

  tags = merge(local.common_tags, {
    DataZone = "raw"
    DataTier = "landing"
  })
}

# Bronze Data Zone - Cleansed and validated data
resource "aws_glue_catalog_database" "bronze" {
  name         = "${local.name_prefix}_bronze"
  description  = "Bronze data zone - Cleansed and validated data with schema enforcement"
  catalog_id   = data.aws_caller_identity.current.account_id
  location_uri = "s3://${var.bronze_bucket_name}/"

  create_table_default_permission {
    permissions = ["ALL"]

    principal {
      data_lake_principal_identifier = "IAM_ALLOWED_PRINCIPALS"
    }
  }

  tags = merge(local.common_tags, {
    DataZone = "bronze"
    DataTier = "cleansed"
  })
}

# Silver Data Zone - Conformed and enriched data
resource "aws_glue_catalog_database" "silver" {
  name         = "${local.name_prefix}_silver"
  description  = "Silver data zone - Conformed, enriched, and business-ready data"
  catalog_id   = data.aws_caller_identity.current.account_id
  location_uri = "s3://${var.silver_bucket_name}/"

  create_table_default_permission {
    permissions = ["ALL"]

    principal {
      data_lake_principal_identifier = "IAM_ALLOWED_PRINCIPALS"
    }
  }

  tags = merge(local.common_tags, {
    DataZone = "silver"
    DataTier = "conformed"
  })
}

# Gold Data Zone - Aggregated and analytics-ready data
resource "aws_glue_catalog_database" "gold" {
  name         = "${local.name_prefix}_gold"
  description  = "Gold data zone - Aggregated, curated data for analytics and reporting"
  catalog_id   = data.aws_caller_identity.current.account_id
  location_uri = "s3://${var.gold_bucket_name}/"

  create_table_default_permission {
    permissions = ["ALL"]

    principal {
      data_lake_principal_identifier = "IAM_ALLOWED_PRINCIPALS"
    }
  }

  tags = merge(local.common_tags, {
    DataZone = "gold"
    DataTier = "aggregated"
  })
}

#------------------------------------------------------------------------------
# AWS Glue Crawlers - Automatic Schema Discovery
#------------------------------------------------------------------------------

# Raw Zone Crawler
resource "aws_glue_crawler" "raw" {
  name          = "${local.name_prefix}-raw-crawler"
  database_name = aws_glue_catalog_database.raw.name
  role          = aws_iam_role.glue_crawler.arn
  description   = "Crawler for raw data zone - discovers schema from ingested files"

  schedule = var.crawler_schedules.raw

  s3_target {
    path       = "s3://${var.raw_bucket_name}/${var.raw_data_prefix}"
    exclusions = var.crawler_exclusion_patterns
  }

  schema_change_policy {
    delete_behavior = var.crawler_schema_change_policy.delete_behavior
    update_behavior = var.crawler_schema_change_policy.update_behavior
  }

  recrawl_policy {
    recrawl_behavior = var.crawler_recrawl_behavior
  }

  configuration = jsonencode({
    Version = 1.0
    Grouping = {
      TableGroupingPolicy = var.crawler_table_grouping_policy
    }
    CrawlerOutput = {
      Partitions = {
        AddOrUpdateBehavior = "InheritFromTable"
      }
    }
  })

  classifiers = var.glue_classifiers

  lake_formation_configuration {
    account_id                     = data.aws_caller_identity.current.account_id
    use_lake_formation_credentials = var.use_lake_formation_credentials
  }

  tags = merge(local.common_tags, {
    DataZone = "raw"
  })
}

# Bronze Zone Crawler
resource "aws_glue_crawler" "bronze" {
  name          = "${local.name_prefix}-bronze-crawler"
  database_name = aws_glue_catalog_database.bronze.name
  role          = aws_iam_role.glue_crawler.arn
  description   = "Crawler for bronze data zone - discovers schema from cleansed data"

  schedule = var.crawler_schedules.bronze

  s3_target {
    path       = "s3://${var.bronze_bucket_name}/${var.bronze_data_prefix}"
    exclusions = var.crawler_exclusion_patterns
  }

  schema_change_policy {
    delete_behavior = var.crawler_schema_change_policy.delete_behavior
    update_behavior = var.crawler_schema_change_policy.update_behavior
  }

  recrawl_policy {
    recrawl_behavior = var.crawler_recrawl_behavior
  }

  configuration = jsonencode({
    Version = 1.0
    Grouping = {
      TableGroupingPolicy = var.crawler_table_grouping_policy
    }
    CrawlerOutput = {
      Partitions = {
        AddOrUpdateBehavior = "InheritFromTable"
      }
    }
  })

  classifiers = var.glue_classifiers

  lake_formation_configuration {
    account_id                     = data.aws_caller_identity.current.account_id
    use_lake_formation_credentials = var.use_lake_formation_credentials
  }

  tags = merge(local.common_tags, {
    DataZone = "bronze"
  })
}

# Silver Zone Crawler
resource "aws_glue_crawler" "silver" {
  name          = "${local.name_prefix}-silver-crawler"
  database_name = aws_glue_catalog_database.silver.name
  role          = aws_iam_role.glue_crawler.arn
  description   = "Crawler for silver data zone - discovers schema from conformed data"

  schedule = var.crawler_schedules.silver

  s3_target {
    path       = "s3://${var.silver_bucket_name}/${var.silver_data_prefix}"
    exclusions = var.crawler_exclusion_patterns
  }

  schema_change_policy {
    delete_behavior = var.crawler_schema_change_policy.delete_behavior
    update_behavior = var.crawler_schema_change_policy.update_behavior
  }

  recrawl_policy {
    recrawl_behavior = var.crawler_recrawl_behavior
  }

  configuration = jsonencode({
    Version = 1.0
    Grouping = {
      TableGroupingPolicy = var.crawler_table_grouping_policy
    }
    CrawlerOutput = {
      Partitions = {
        AddOrUpdateBehavior = "InheritFromTable"
      }
    }
  })

  classifiers = var.glue_classifiers

  lake_formation_configuration {
    account_id                     = data.aws_caller_identity.current.account_id
    use_lake_formation_credentials = var.use_lake_formation_credentials
  }

  tags = merge(local.common_tags, {
    DataZone = "silver"
  })
}

# Gold Zone Crawler
resource "aws_glue_crawler" "gold" {
  name          = "${local.name_prefix}-gold-crawler"
  database_name = aws_glue_catalog_database.gold.name
  role          = aws_iam_role.glue_crawler.arn
  description   = "Crawler for gold data zone - discovers schema from aggregated data"

  schedule = var.crawler_schedules.gold

  s3_target {
    path       = "s3://${var.gold_bucket_name}/${var.gold_data_prefix}"
    exclusions = var.crawler_exclusion_patterns
  }

  schema_change_policy {
    delete_behavior = var.crawler_schema_change_policy.delete_behavior
    update_behavior = var.crawler_schema_change_policy.update_behavior
  }

  recrawl_policy {
    recrawl_behavior = var.crawler_recrawl_behavior
  }

  configuration = jsonencode({
    Version = 1.0
    Grouping = {
      TableGroupingPolicy = var.crawler_table_grouping_policy
    }
    CrawlerOutput = {
      Partitions = {
        AddOrUpdateBehavior = "InheritFromTable"
      }
    }
  })

  classifiers = var.glue_classifiers

  lake_formation_configuration {
    account_id                     = data.aws_caller_identity.current.account_id
    use_lake_formation_credentials = var.use_lake_formation_credentials
  }

  tags = merge(local.common_tags, {
    DataZone = "gold"
  })
}

#------------------------------------------------------------------------------
# AWS Lake Formation - Data Lake Settings
#------------------------------------------------------------------------------

resource "aws_lakeformation_data_lake_settings" "main" {
  admins = var.lake_formation_admins

  create_database_default_permissions {
    permissions = ["ALL"]
    principal   = "IAM_ALLOWED_PRINCIPALS"
  }

  create_table_default_permissions {
    permissions = ["ALL"]
    principal   = "IAM_ALLOWED_PRINCIPALS"
  }

  # External data filtering settings
  trusted_resource_owners = var.lake_formation_trusted_resource_owners

  # Allow external data filtering
  allow_external_data_filtering = var.allow_external_data_filtering

  # Authorized session tag value list
  authorized_session_tag_value_list = var.authorized_session_tags
}

#------------------------------------------------------------------------------
# AWS Lake Formation - Data Locations Registration
#------------------------------------------------------------------------------

resource "aws_lakeformation_resource" "raw" {
  arn      = "arn:aws:s3:::${var.raw_bucket_name}"
  role_arn = aws_iam_role.lake_formation.arn
}

resource "aws_lakeformation_resource" "bronze" {
  arn      = "arn:aws:s3:::${var.bronze_bucket_name}"
  role_arn = aws_iam_role.lake_formation.arn
}

resource "aws_lakeformation_resource" "silver" {
  arn      = "arn:aws:s3:::${var.silver_bucket_name}"
  role_arn = aws_iam_role.lake_formation.arn
}

resource "aws_lakeformation_resource" "gold" {
  arn      = "arn:aws:s3:::${var.gold_bucket_name}"
  role_arn = aws_iam_role.lake_formation.arn
}

#------------------------------------------------------------------------------
# AWS Lake Formation - Permissions
#------------------------------------------------------------------------------

# Database permissions for data engineers
resource "aws_lakeformation_permissions" "data_engineer_raw" {
  count = length(var.data_engineer_principals) > 0 ? 1 : 0

  principal   = var.data_engineer_principals[0]
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.raw.name
  }
}

resource "aws_lakeformation_permissions" "data_engineer_bronze" {
  count = length(var.data_engineer_principals) > 0 ? 1 : 0

  principal   = var.data_engineer_principals[0]
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.bronze.name
  }
}

resource "aws_lakeformation_permissions" "data_engineer_silver" {
  count = length(var.data_engineer_principals) > 0 ? 1 : 0

  principal   = var.data_engineer_principals[0]
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.silver.name
  }
}

resource "aws_lakeformation_permissions" "data_engineer_gold" {
  count = length(var.data_engineer_principals) > 0 ? 1 : 0

  principal   = var.data_engineer_principals[0]
  permissions = ["ALL"]

  database {
    name = aws_glue_catalog_database.gold.name
  }
}

# Read-only permissions for data analysts (gold zone only)
resource "aws_lakeformation_permissions" "data_analyst_gold" {
  count = length(var.data_analyst_principals) > 0 ? 1 : 0

  principal   = var.data_analyst_principals[0]
  permissions = ["DESCRIBE"]

  database {
    name = aws_glue_catalog_database.gold.name
  }
}

resource "aws_lakeformation_permissions" "data_analyst_gold_tables" {
  count = length(var.data_analyst_principals) > 0 ? 1 : 0

  principal   = var.data_analyst_principals[0]
  permissions = ["SELECT", "DESCRIBE"]

  table {
    database_name = aws_glue_catalog_database.gold.name
    wildcard      = true
  }
}

# Cross-account permissions
resource "aws_lakeformation_permissions" "cross_account" {
  for_each = var.enable_cross_account_access ? toset(var.cross_account_principals) : []

  principal                     = each.value
  permissions                   = var.cross_account_permissions
  permissions_with_grant_option = var.cross_account_permissions_grantable ? var.cross_account_permissions : []

  database {
    name = aws_glue_catalog_database.gold.name
  }
}

#------------------------------------------------------------------------------
# Data Sources
#------------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_partition" "current" {}
