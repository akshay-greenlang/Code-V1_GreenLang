# INFRA-001: GitHub Environments Setup Guide

This document describes the required GitHub Environments configuration for the INFRA-001 CI/CD pipeline.

## Overview

The INFRA-001 deployment pipeline uses GitHub Environments to:
- Manage environment-specific secrets
- Implement approval gates for production deployments
- Control who can deploy to each environment
- Enable OIDC authentication with AWS (no long-lived credentials)

## Required Environments

Create the following environments in your GitHub repository settings:

### 1. Development Environment (`dev`)

**Purpose:** Automated deployments for development testing

**Configuration:**
- **Name:** `dev`
- **Protection Rules:** None (auto-deploy enabled)
- **Deployment Branches:** `main`, `master`, `develop`

**Secrets Required:**
| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCOUNT_ID` | AWS Account ID for dev (e.g., `123456789012`) |
| `KUBECONFIG` | Base64-encoded kubeconfig for dev EKS cluster |

### 2. Development Plan Environment (`dev-plan`)

**Purpose:** Terraform plan execution (no approval required)

**Configuration:**
- **Name:** `dev-plan`
- **Protection Rules:** None
- **Deployment Branches:** All branches

### 3. Staging Environment (`staging`)

**Purpose:** Pre-production validation and testing

**Configuration:**
- **Name:** `staging`
- **Protection Rules:**
  - Required reviewers: 1 (tech lead or senior engineer)
  - Wait timer: 5 minutes (optional, for visibility)
- **Deployment Branches:** `main`, `master`

**Secrets Required:**
| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCOUNT_ID` | AWS Account ID for staging |
| `KUBECONFIG` | Base64-encoded kubeconfig for staging EKS cluster |

### 4. Staging Plan Environment (`staging-plan`)

**Purpose:** Terraform plan execution for staging

**Configuration:**
- **Name:** `staging-plan`
- **Protection Rules:** None
- **Deployment Branches:** `main`, `master`

### 5. Production Environment (`prod`)

**Purpose:** Production deployments with strict controls

**Configuration:**
- **Name:** `prod`
- **Protection Rules:**
  - Required reviewers: 2 (must include infrastructure owner)
  - Wait timer: 15 minutes
  - Prevent self-approval: Enabled
- **Deployment Branches:** `main`, `master` only

**Secrets Required:**
| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCOUNT_ID` | AWS Account ID for production |
| `KUBECONFIG` | Base64-encoded kubeconfig for prod EKS cluster |

### 6. Production Plan Environment (`prod-plan`)

**Purpose:** Terraform plan execution for production

**Configuration:**
- **Name:** `prod-plan`
- **Protection Rules:**
  - Required reviewers: 1 (for visibility into production changes)
- **Deployment Branches:** `main`, `master`

### 7. Production Destroy Environment (`prod-destroy`)

**Purpose:** Infrastructure destruction (emergency only)

**Configuration:**
- **Name:** `prod-destroy`
- **Protection Rules:**
  - Required reviewers: 3 (must include CTO/VP Engineering)
  - Wait timer: 60 minutes
  - Prevent self-approval: Enabled
- **Deployment Branches:** None (manual dispatch only)

## Repository Secrets (Global)

Configure these secrets at the repository level:

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `SLACK_WEBHOOK_URL` | Slack webhook for deployment notifications | Optional |
| `INFRACOST_API_KEY` | Infracost API key for cost estimation | Optional |
| `SNYK_TOKEN` | Snyk token for security scanning | Optional |

## OIDC Configuration for AWS

### Step 1: Create IAM OIDC Provider

In each AWS account (dev, staging, prod), create an OIDC identity provider:

```hcl
# Terraform code to create OIDC provider
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}
```

### Step 2: Create IAM Role for GitHub Actions

Create an IAM role in each AWS account with the following trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:*"
        }
      }
    }
  ]
}
```

### Step 3: Attach Required Policies

Attach these managed policies to the role:
- `AmazonEKSClusterPolicy`
- `AmazonEKS_CNI_Policy`

Plus a custom policy for Terraform operations:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::greenlang-terraform-state",
        "arn:aws:s3:::greenlang-terraform-state/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:DeleteItem"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/greenlang-terraform-locks"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "eks:*",
        "rds:*",
        "elasticache:*",
        "s3:*",
        "iam:*",
        "secretsmanager:*",
        "kms:*",
        "logs:*",
        "cloudwatch:*"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        }
      }
    }
  ]
}
```

## Role ARN Format

Update the workflow with your actual role ARNs:

| Environment | Role ARN Pattern |
|-------------|-----------------|
| dev | `arn:aws:iam::DEV_ACCOUNT_ID:role/greenlang-dev-github-actions` |
| staging | `arn:aws:iam::STAGING_ACCOUNT_ID:role/greenlang-staging-github-actions` |
| prod | `arn:aws:iam::PROD_ACCOUNT_ID:role/greenlang-prod-github-actions` |

## Workflow Permissions

The workflows require these GitHub permissions:

```yaml
permissions:
  id-token: write      # Required for OIDC token
  contents: read       # Required for checkout
  pull-requests: write # Required for PR comments
  security-events: write # Required for SARIF uploads
  actions: read        # Required for workflow status
```

## Verification Checklist

Before using the workflows, verify:

- [ ] All 7 environments are created in GitHub
- [ ] Protection rules are configured correctly
- [ ] Required reviewers are assigned
- [ ] OIDC provider exists in each AWS account
- [ ] IAM roles are created with correct trust policies
- [ ] S3 state bucket exists (`greenlang-terraform-state`)
- [ ] DynamoDB lock table exists (`greenlang-terraform-locks`)
- [ ] Kubeconfig secrets are set for each environment
- [ ] Slack webhook is configured (optional)

## Troubleshooting

### OIDC Authentication Fails

1. Check the OIDC provider thumbprint is correct
2. Verify the trust policy `sub` claim matches your repo
3. Ensure the role has `sts:AssumeRoleWithWebIdentity` permission

### Terraform State Lock

If deployments fail with lock errors:
```bash
# Force unlock (use with caution)
terraform force-unlock LOCK_ID
```

### Environment Approval Timeout

If approvals expire:
1. Re-run the workflow
2. Consider increasing wait times for production

## Support

For issues with the INFRA-001 pipeline:
1. Check workflow logs in GitHub Actions
2. Review the security gate results
3. Contact the DevOps team via #infra-support Slack channel
