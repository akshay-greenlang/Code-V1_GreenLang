# GreenLang Development Environment Variables

aws_region = "us-east-1"

vpc_cidr = "10.0.0.0/16"

availability_zones = [
  "us-east-1a",
  "us-east-1b"
]

eks_cluster_version = "1.28"

eks_public_access_cidrs = ["0.0.0.0/0"]

rds_engine_version = "15.4"

# ELB Account ID for us-east-1
elb_account_id = "127311923021"

# GitHub OIDC Configuration (update with your values)
create_github_oidc_provider = true
github_oidc_provider_arn    = null  # Will be created
github_org                  = "greenlang"
github_repo                 = "greenlang-app"
