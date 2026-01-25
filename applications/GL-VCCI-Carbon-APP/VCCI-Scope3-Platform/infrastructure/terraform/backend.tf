# Terraform Backend Configuration
# This configures remote state storage in S3 with DynamoDB state locking

terraform {
  backend "s3" {
    # S3 bucket for state storage
    bucket = "vcci-scope3-terraform-state"

    # State file key (environment-specific)
    key = "vcci-scope3/terraform.tfstate"

    # AWS region for S3 bucket
    region = "us-west-2"

    # DynamoDB table for state locking
    dynamodb_table = "vcci-scope3-terraform-locks"

    # Enable encryption at rest
    encrypt = true

    # KMS key for encryption (optional, uses AWS managed key if not specified)
    # kms_key_id = "arn:aws:kms:us-west-2:ACCOUNT_ID:key/KEY_ID"

    # Workspace key prefix
    workspace_key_prefix = "workspaces"

    # Enable versioning
    versioning = true
  }
}

# Note: Before using this backend, you must create the S3 bucket and DynamoDB table
# Run: terraform init -backend=false
# Then: ./scripts/init.sh to create the backend resources
# Finally: terraform init -reconfigure to migrate state to remote backend
