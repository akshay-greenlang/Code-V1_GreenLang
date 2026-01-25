# GreenLang Staging Environment Variables

aws_region = "us-east-1"

vpc_cidr = "10.1.0.0/16"

availability_zones = [
  "us-east-1a",
  "us-east-1b",
  "us-east-1c"
]

eks_cluster_version = "1.28"

eks_public_access_cidrs = ["0.0.0.0/0"]

rds_engine_version = "15.4"

# ELB Account ID for us-east-1
elb_account_id = "127311923021"

# CORS Configuration
cors_allowed_origins = [
  "https://staging.greenlang.io",
  "https://staging-app.greenlang.io"
]

# GitHub OIDC Configuration
create_github_oidc_provider = false
github_oidc_provider_arn    = null
github_org                  = "greenlang"
github_repo                 = "greenlang-app"

# ECR Repositories
ecr_repository_arns = [
  "arn:aws:ecr:us-east-1:*:repository/greenlang/*"
]

# SNS/SQS Configuration
alarm_sns_topic_arns = []
sqs_queue_arns       = []
sns_topic_arns       = []

# CI/CD Secrets
cicd_secrets_arns = []
