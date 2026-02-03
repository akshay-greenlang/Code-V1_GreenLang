# GreenLang Terraform Infrastructure

Production-grade AWS infrastructure for the GreenLang platform using Terraform.

## Architecture Overview

```
                                    +-------------------+
                                    |   CloudFront      |
                                    |   (CDN)           |
                                    +--------+----------+
                                             |
                                    +--------v----------+
                                    |   Application     |
                                    |   Load Balancer   |
                                    +--------+----------+
                                             |
         +-----------------------------------+-----------------------------------+
         |                                   |                                   |
+--------v----------+            +-----------v-----------+            +----------v---------+
|  Public Subnets   |            |   Private Subnets     |            | Database Subnets   |
|  (NAT Gateways)   |            |   (EKS Nodes)         |            | (RDS, ElastiCache) |
+-------------------+            +-----------------------+            +--------------------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
           +--------v--------+      +--------v--------+      +--------v--------+
           |  System Nodes   |      |   API Nodes     |      | Agent Runtime   |
           |  (CoreDNS, etc) |      |   (API Gateway) |      | Nodes (Agents)  |
           +-----------------+      +-----------------+      +-----------------+
```

## Directory Structure

```
deployment/terraform/
├── backend-bootstrap/           # S3 and DynamoDB for state management
│   ├── main.tf
│   └── terraform.tfvars
├── modules/                     # Reusable Terraform modules
│   ├── vpc/                     # VPC, subnets, NAT, VPC endpoints
│   ├── eks/                     # EKS cluster, node groups, add-ons
│   ├── rds/                     # PostgreSQL RDS with replicas
│   ├── elasticache/             # Redis cluster with replication
│   ├── s3/                      # S3 buckets with lifecycle policies
│   ├── iam/                     # IAM roles for IRSA, CI/CD
│   ├── monitoring/              # CloudWatch dashboards and alarms
│   ├── keycloak/                # Keycloak for authentication
│   └── vault/                   # HashiCorp Vault for secrets
└── environments/                # Environment-specific configurations
    ├── dev/
    ├── staging/
    └── prod/
```

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.0 installed
3. **kubectl** for Kubernetes management
4. **AWS IAM permissions** for creating all required resources

## Initial Setup

### 1. Bootstrap Backend (First Time Only)

Create the S3 bucket and DynamoDB table for Terraform state:

```bash
cd deployment/terraform/backend-bootstrap
terraform init
terraform plan
terraform apply
```

### 2. Update Backend Configuration

After bootstrap, note the output values and update the backend configuration in each environment's `main.tf` if the bucket name differs.

### 3. Deploy an Environment

```bash
# Development
cd deployment/terraform/environments/dev
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars"

# Staging
cd deployment/terraform/environments/staging
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars"

# Production
cd deployment/terraform/environments/prod
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars"
```

## Modules

### VPC Module

Creates a production-ready VPC with:
- Public subnets for NAT Gateways and load balancers
- Private subnets for EKS nodes
- Database subnets for RDS and ElastiCache
- VPC Flow Logs for security monitoring
- VPC Endpoints for AWS services (S3, ECR, SSM)

### EKS Module

Creates an EKS cluster with:
- Managed node groups (system, API, agent runtime)
- OIDC provider for IRSA
- Cluster autoscaler IAM role
- AWS Load Balancer Controller IAM role
- EBS CSI driver for persistent storage
- KMS encryption for secrets

### RDS Module

Creates a PostgreSQL database with:
- Multi-AZ deployment for high availability
- Read replicas for scaling
- Automated backups with point-in-time recovery
- Performance Insights for monitoring
- Secrets Manager integration for credentials

### ElastiCache Module

Creates a Redis cluster with:
- Replication for high availability
- Encryption at rest and in transit
- Parameter tuning for performance
- CloudWatch alarms for monitoring

### S3 Module

Creates S3 buckets with:
- Artifacts bucket for CI/CD
- Logs bucket for access logs
- Backups bucket with object lock
- Data bucket with intelligent tiering
- Cross-region replication for DR

### IAM Module

Creates IAM roles for:
- Application service accounts (IRSA)
- Agent runtime service accounts
- CI/CD deployment (GitHub Actions/GitLab)
- External Secrets Operator
- Monitoring and observability

### Monitoring Module

Creates CloudWatch resources:
- Log groups for applications
- Dashboards for infrastructure
- Alarms for critical metrics
- SNS topics for notifications

## Environment Differences

| Feature | Dev | Staging | Production |
|---------|-----|---------|------------|
| Availability Zones | 2 | 3 | 3 |
| RDS Multi-AZ | No | Yes | Yes |
| RDS Read Replicas | 0 | 1 | 2 |
| ElastiCache Nodes | 1 | 2 | 3 |
| Node Capacity Type | Spot | Spot | On-Demand |
| VPC Flow Logs Retention | 14 days | 30 days | 90 days |
| Backup Retention | 7 days | 14 days | 30 days |
| Deletion Protection | No | Yes | Yes |
| Cross-Region Replication | No | No | Yes |

## Configuration Variables

### Required Variables

Update `terraform.tfvars` in each environment with:

```hcl
# AWS Configuration
aws_region = "us-east-1"

# GitHub OIDC for CI/CD (if using GitHub Actions)
github_org  = "your-org"
github_repo = "your-repo"

# SNS Topics for Alarms
alarm_sns_topic_arns = ["arn:aws:sns:us-east-1:ACCOUNT_ID:alerts"]
```

### Production-Specific Variables

For production, also configure:

```hcl
# DR Configuration
dr_region          = "us-west-2"
dr_data_bucket_arn = "arn:aws:s3:::greenlang-prod-data-dr-ACCOUNT_ID"
dr_kms_key_arn     = "arn:aws:kms:us-west-2:ACCOUNT_ID:key/KEY_ID"

# CORS Origins
cors_allowed_origins = [
  "https://greenlang.io",
  "https://app.greenlang.io"
]
```

## Accessing the EKS Cluster

After deployment, configure kubectl:

```bash
aws eks update-kubeconfig \
  --region us-east-1 \
  --name greenlang-<env>-eks
```

## Security Considerations

1. **Encryption**: All data is encrypted at rest using KMS
2. **Network**: Private subnets for all workloads
3. **IAM**: Least privilege with IRSA for pod-level permissions
4. **Secrets**: Secrets Manager for sensitive configuration
5. **Logging**: VPC Flow Logs and CloudTrail enabled
6. **Backup**: Automated backups with retention policies

## Cost Optimization

1. **Dev Environment**: Uses spot instances and minimal resources
2. **Staging Environment**: Production-like but smaller scale
3. **Production Environment**: Full redundancy and capacity

## Troubleshooting

### State Lock Issues

If terraform state is locked, check DynamoDB:

```bash
aws dynamodb scan \
  --table-name greenlang-terraform-locks \
  --filter-expression "attribute_exists(LockID)"
```

### EKS Access Issues

Ensure your IAM user/role has the appropriate aws-auth ConfigMap entry:

```bash
kubectl get configmap aws-auth -n kube-system -o yaml
```

### Module Updates

When updating modules, use targeted applies:

```bash
terraform apply -target=module.eks
terraform apply -target=module.rds
```

## Contributing

1. Create feature branch
2. Update module and test in dev
3. Update documentation
4. Submit pull request

## License

Proprietary - GreenLang Inc.
