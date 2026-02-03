# GreenLang Terraform Backend Bootstrap Variables

aws_region = "us-east-1"

# The bucket name will have the AWS account ID appended automatically
bucket_name = "greenlang-terraform-state"

dynamodb_table_name = "greenlang-terraform-locks"

enable_versioning  = true
enable_encryption  = true

# Retain old state versions for 90 days
noncurrent_version_expiration_days = 90
