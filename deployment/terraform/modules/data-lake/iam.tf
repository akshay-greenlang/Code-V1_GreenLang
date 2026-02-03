#------------------------------------------------------------------------------
# GreenLang AWS Data Lake Infrastructure
# Terraform Module: data-lake - IAM Resources
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IAM Role for Glue Crawlers
#------------------------------------------------------------------------------

resource "aws_iam_role" "glue_crawler" {
  name = "${local.name_prefix}-glue-crawler-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_crawler_service" {
  role       = aws_iam_role.glue_crawler.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_policy" "glue_crawler_s3" {
  name        = "${local.name_prefix}-glue-crawler-s3-policy"
  description = "S3 access policy for Glue crawlers"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ReadAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.raw_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.raw_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.bronze_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.bronze_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.silver_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.silver_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}/*"
        ]
      },
      {
        Sid    = "KMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = var.kms_key_arn
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_crawler_s3" {
  role       = aws_iam_role.glue_crawler.name
  policy_arn = aws_iam_policy.glue_crawler_s3.arn
}

resource "aws_iam_policy" "glue_crawler_lake_formation" {
  name        = "${local.name_prefix}-glue-crawler-lf-policy"
  description = "Lake Formation access policy for Glue crawlers"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "LakeFormationAccess"
        Effect = "Allow"
        Action = [
          "lakeformation:GetDataAccess"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_crawler_lake_formation" {
  role       = aws_iam_role.glue_crawler.name
  policy_arn = aws_iam_policy.glue_crawler_lake_formation.arn
}

#------------------------------------------------------------------------------
# IAM Role for Glue ETL Jobs
#------------------------------------------------------------------------------

resource "aws_iam_role" "glue_etl" {
  name = "${local.name_prefix}-glue-etl-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_etl_service" {
  role       = aws_iam_role.glue_etl.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_policy" "glue_etl_s3" {
  name        = "${local.name_prefix}-glue-etl-s3-policy"
  description = "S3 access policy for Glue ETL jobs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ReadWriteDataZones"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.raw_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.raw_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.bronze_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.bronze_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.silver_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.silver_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}/*"
        ]
      },
      {
        Sid    = "S3ReadScripts"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.etl_scripts_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.etl_scripts_bucket_name}/*"
        ]
      },
      {
        Sid    = "S3TempBucket"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.etl_temp_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.etl_temp_bucket_name}/*"
        ]
      },
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey",
          "kms:ReEncrypt*",
          "kms:DescribeKey"
        ]
        Resource = var.kms_key_arn
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_etl_s3" {
  role       = aws_iam_role.glue_etl.name
  policy_arn = aws_iam_policy.glue_etl_s3.arn
}

resource "aws_iam_policy" "glue_etl_cloudwatch" {
  name        = "${local.name_prefix}-glue-etl-cloudwatch-policy"
  description = "CloudWatch access policy for Glue ETL jobs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws-glue/*"
        ]
      },
      {
        Sid    = "CloudWatchMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "AWS/Glue"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_etl_cloudwatch" {
  role       = aws_iam_role.glue_etl.name
  policy_arn = aws_iam_policy.glue_etl_cloudwatch.arn
}

resource "aws_iam_policy" "glue_etl_lake_formation" {
  name        = "${local.name_prefix}-glue-etl-lf-policy"
  description = "Lake Formation access policy for Glue ETL jobs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "LakeFormationAccess"
        Effect = "Allow"
        Action = [
          "lakeformation:GetDataAccess"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_etl_lake_formation" {
  role       = aws_iam_role.glue_etl.name
  policy_arn = aws_iam_policy.glue_etl_lake_formation.arn
}

#------------------------------------------------------------------------------
# IAM Role for Athena Query Execution
#------------------------------------------------------------------------------

resource "aws_iam_role" "athena_execution" {
  name = "${local.name_prefix}-athena-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "athena.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "sts:ExternalId" = var.project_name
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_policy" "athena_execution" {
  name        = "${local.name_prefix}-athena-execution-policy"
  description = "Policy for Athena query execution"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AthenaAccess"
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:StopQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "athena:GetWorkGroup",
          "athena:BatchGetQueryExecution",
          "athena:ListQueryExecutions"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:athena:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:workgroup/${local.name_prefix}-*"
        ]
      },
      {
        Sid    = "GlueCatalogAccess"
        Effect = "Allow"
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
          "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:database/${local.name_prefix}_*",
          "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:table/${local.name_prefix}_*/*"
        ]
      },
      {
        Sid    = "S3ReadDataZones"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.raw_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.raw_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.bronze_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.bronze_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.silver_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.silver_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}/*"
        ]
      },
      {
        Sid    = "S3WriteQueryResults"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.athena_results_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.athena_results_bucket_name}/*"
        ]
      },
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey"
        ]
        Resource = var.kms_key_arn
      },
      {
        Sid    = "LakeFormationAccess"
        Effect = "Allow"
        Action = [
          "lakeformation:GetDataAccess"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "athena_execution" {
  role       = aws_iam_role.athena_execution.name
  policy_arn = aws_iam_policy.athena_execution.arn
}

#------------------------------------------------------------------------------
# IAM Role for Lake Formation
#------------------------------------------------------------------------------

resource "aws_iam_role" "lake_formation" {
  name = "${local.name_prefix}-lake-formation-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lakeformation.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_policy" "lake_formation" {
  name        = "${local.name_prefix}-lake-formation-policy"
  description = "Policy for Lake Formation data access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3Access"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.raw_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.raw_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.bronze_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.bronze_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.silver_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.silver_bucket_name}/*",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}/*"
        ]
      },
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey",
          "kms:ReEncrypt*",
          "kms:DescribeKey"
        ]
        Resource = var.kms_key_arn
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lake_formation" {
  role       = aws_iam_role.lake_formation.name
  policy_arn = aws_iam_policy.lake_formation.arn
}

#------------------------------------------------------------------------------
# IAM Role for EKS IRSA - Athena Access
#------------------------------------------------------------------------------

resource "aws_iam_role" "eks_athena" {
  count = var.enable_eks_irsa ? 1 : 0

  name = "${local.name_prefix}-eks-athena-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.eks_oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.eks_oidc_provider_url}:sub" = "system:serviceaccount:${var.eks_namespace}:${var.eks_service_account_name}"
            "${var.eks_oidc_provider_url}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_policy" "eks_athena" {
  count = var.enable_eks_irsa ? 1 : 0

  name        = "${local.name_prefix}-eks-athena-policy"
  description = "Policy for EKS pods to access Athena and Glue"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AthenaAccess"
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:StopQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "athena:GetWorkGroup",
          "athena:BatchGetQueryExecution",
          "athena:ListQueryExecutions",
          "athena:GetNamedQuery",
          "athena:ListNamedQueries"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:athena:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:workgroup/${local.name_prefix}-*"
        ]
      },
      {
        Sid    = "GlueCatalogAccess"
        Effect = "Allow"
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
          "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:database/${local.name_prefix}_*",
          "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:table/${local.name_prefix}_*/*"
        ]
      },
      {
        Sid    = "S3ReadGoldZone"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.gold_bucket_name}/*"
        ]
      },
      {
        Sid    = "S3WriteQueryResults"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::${var.athena_results_bucket_name}",
          "arn:${data.aws_partition.current.partition}:s3:::${var.athena_results_bucket_name}/*"
        ]
      },
      {
        Sid    = "KMSAccess"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey"
        ]
        Resource = var.kms_key_arn
      },
      {
        Sid    = "LakeFormationAccess"
        Effect = "Allow"
        Action = [
          "lakeformation:GetDataAccess"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_athena" {
  count = var.enable_eks_irsa ? 1 : 0

  role       = aws_iam_role.eks_athena[0].name
  policy_arn = aws_iam_policy.eks_athena[0].arn
}
