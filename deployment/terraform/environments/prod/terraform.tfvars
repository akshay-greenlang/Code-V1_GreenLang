# GreenLang Production Environment Variables

aws_region = "us-east-1"
dr_region  = "us-west-2"

vpc_cidr = "10.2.0.0/16"

availability_zones = [
  "us-east-1a",
  "us-east-1b",
  "us-east-1c"
]

eks_cluster_version = "1.28"

# Restrict EKS public access in production
enable_eks_public_access = false
eks_public_access_cidrs  = []  # No public access - use VPN/bastion

rds_engine_version = "15.4"

# ELB Account ID for us-east-1
elb_account_id = "127311923021"

# CORS Configuration - only allow production domains
cors_allowed_origins = [
  "https://greenlang.io",
  "https://www.greenlang.io",
  "https://app.greenlang.io",
  "https://api.greenlang.io"
]

# GitHub OIDC Configuration
create_github_oidc_provider = false
github_oidc_provider_arn    = "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
github_org                  = "greenlang"
github_repo                 = "greenlang-app"

# ECR Repositories
ecr_repository_arns = [
  "arn:aws:ecr:us-east-1:ACCOUNT_ID:repository/greenlang/*"
]

# SNS/SQS Configuration
alarm_sns_topic_arns = [
  "arn:aws:sns:us-east-1:ACCOUNT_ID:greenlang-prod-alerts"
]
elasticache_notification_topic_arn = "arn:aws:sns:us-east-1:ACCOUNT_ID:greenlang-prod-alerts"

sqs_queue_arns = [
  "arn:aws:sqs:us-east-1:ACCOUNT_ID:greenlang-prod-*"
]
sns_topic_arns = [
  "arn:aws:sns:us-east-1:ACCOUNT_ID:greenlang-prod-*"
]

# Secrets Configuration
app_secrets_arns = [
  "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:greenlang-prod/*"
]
agent_secrets_arns = [
  "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:greenlang-prod/agents/*"
]
cicd_secrets_arns = [
  "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:greenlang-prod/cicd/*"
]
external_secrets_allowed_arns = [
  "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:greenlang-prod/*"
]

# CloudFront Distribution
cloudfront_distribution_arn = "arn:aws:cloudfront::ACCOUNT_ID:distribution/DISTRIBUTION_ID"

# DR Configuration
dr_data_bucket_arn = "arn:aws:s3:::greenlang-prod-data-dr-ACCOUNT_ID"
dr_kms_key_arn     = "arn:aws:kms:us-west-2:ACCOUNT_ID:key/KEY_ID"

# Cross-Account Access
enable_cross_account_access = false
trusted_account_ids         = []
cross_account_resource_arns = []
