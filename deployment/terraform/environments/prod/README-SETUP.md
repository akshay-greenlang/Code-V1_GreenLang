# GreenLang Production Terraform Configuration Setup

This document explains how to configure the Terraform variables for the GreenLang production environment.

## Prerequisites

Before configuring the Terraform variables, ensure you have:

1. AWS CLI installed and configured with appropriate credentials
2. Terraform >= 1.0 installed
3. Access to the AWS account where infrastructure will be deployed
4. Required permissions to create and manage AWS resources

## Quick Start

### Step 1: Copy the Template

```bash
cp terraform.tfvars.template terraform.tfvars
```

### Step 2: Gather Required Values

You will need the following values from your AWS account:

| Placeholder | Description | How to Find |
|-------------|-------------|-------------|
| `${AWS_ACCOUNT_ID}` | Your 12-digit AWS account ID | `aws sts get-caller-identity --query Account --output text` |
| `${CLOUDFRONT_DISTRIBUTION_ID}` | CloudFront distribution ID | AWS Console > CloudFront > Distributions |
| `${DR_KMS_KEY_ID}` | KMS key ID in DR region (us-west-2) | AWS Console > KMS > Customer managed keys |

### Step 3: Replace Placeholders

Using your preferred editor or sed, replace all placeholders:

```bash
# Replace AWS Account ID (example: 123456789012)
sed -i 's/\${AWS_ACCOUNT_ID}/123456789012/g' terraform.tfvars

# Replace CloudFront Distribution ID (example: E1ABCDEF123456)
sed -i 's/\${CLOUDFRONT_DISTRIBUTION_ID}/E1ABCDEF123456/g' terraform.tfvars

# Replace DR KMS Key ID (example: 12345678-1234-1234-1234-123456789012)
sed -i 's/\${DR_KMS_KEY_ID}/12345678-1234-1234-1234-123456789012/g' terraform.tfvars
```

### Step 4: Validate Configuration

```bash
terraform init
terraform validate
terraform plan
```

## Placeholder Details

### AWS_ACCOUNT_ID

**Format:** 12-digit number (e.g., `123456789012`)

**Description:** Your AWS account identifier used in ARN construction for IAM policies, ECR repositories, SNS topics, SQS queues, and Secrets Manager.

**How to find:**
```bash
aws sts get-caller-identity --query Account --output text
```

**Used in:**
- GitHub OIDC provider ARN
- ECR repository ARNs
- SNS/SQS ARNs
- Secrets Manager ARNs
- CloudFront distribution ARN
- DR S3 bucket name
- DR KMS key ARN

### CLOUDFRONT_DISTRIBUTION_ID

**Format:** 14 alphanumeric characters (e.g., `E1ABCDEF123456`)

**Description:** The unique identifier for your CloudFront distribution that serves static assets and provides CDN capabilities.

**How to find:**
1. AWS Console > CloudFront > Distributions
2. Look for the "ID" column
3. Or use AWS CLI:
```bash
aws cloudfront list-distributions --query 'DistributionList.Items[*].[Id,DomainName]' --output table
```

**Note:** If you have not created the CloudFront distribution yet, you can:
1. Leave this as a placeholder and update later
2. Create the distribution first, then update this value
3. Use Terraform to create the distribution in a separate module

### DR_KMS_KEY_ID

**Format:** UUID (e.g., `12345678-1234-1234-1234-123456789012`)

**Description:** The KMS key ID in the disaster recovery region (us-west-2) used for encrypting replicated data.

**How to find:**
1. AWS Console > KMS > Customer managed keys (ensure you are in us-west-2)
2. Look for the "Key ID" column
3. Or use AWS CLI:
```bash
aws kms list-keys --region us-west-2 --query 'Keys[*].KeyId' --output table
```

**Note:** If the DR KMS key does not exist yet:
1. Create it in the DR region (us-west-2) first
2. Ensure it has appropriate key policy for cross-region replication
3. Update the terraform.tfvars with the key ID

## Security Best Practices

### Never Commit Sensitive Values

Add `terraform.tfvars` to your `.gitignore`:

```gitignore
# Terraform
*.tfvars
!*.tfvars.template
*.tfstate
*.tfstate.*
.terraform/
```

### Use Environment Variables (Alternative)

Instead of hardcoding values in terraform.tfvars, you can use environment variables:

```bash
export TF_VAR_aws_account_id="123456789012"
export TF_VAR_cloudfront_distribution_id="E1ABCDEF123456"
export TF_VAR_dr_kms_key_id="12345678-1234-1234-1234-123456789012"
```

Then reference them in your Terraform configuration:

```hcl
variable "aws_account_id" {
  description = "AWS Account ID"
  type        = string
}
```

### Use AWS Secrets Manager or Parameter Store

For production environments, consider storing sensitive configuration in:
- AWS Secrets Manager
- AWS Systems Manager Parameter Store
- HashiCorp Vault

## Validation Checklist

Before running `terraform apply`, verify:

- [ ] AWS_ACCOUNT_ID is replaced with your 12-digit account ID
- [ ] CLOUDFRONT_DISTRIBUTION_ID is replaced with valid distribution ID
- [ ] DR_KMS_KEY_ID is replaced with valid KMS key ID in us-west-2
- [ ] All ARNs are syntactically correct
- [ ] terraform.tfvars is NOT committed to version control
- [ ] `terraform validate` passes without errors
- [ ] `terraform plan` shows expected changes

## Troubleshooting

### Invalid ARN Format

If you see errors about invalid ARN format:
1. Verify the AWS account ID is exactly 12 digits
2. Check that region names are correct (us-east-1, us-west-2)
3. Ensure resource names do not contain invalid characters

### KMS Key Not Found

If the DR KMS key is not found:
1. Verify you are looking in the correct region (us-west-2)
2. Check the key policy allows the required operations
3. Ensure the key is enabled and not pending deletion

### CloudFront Distribution Not Found

If the CloudFront distribution is not found:
1. Verify the distribution ID is correct (14 characters)
2. Check the distribution is not disabled or pending
3. Ensure the distribution belongs to the correct AWS account

## Support

For additional help:
- Review the main Terraform documentation in `/deployment/terraform/README.md`
- Check AWS documentation for ARN formats
- Contact the DevOps team for access issues
