# GL-VCCI Scope 3 Carbon Intelligence Platform - AWS Infrastructure

This repository contains the Terraform Infrastructure as Code (IaC) for deploying the GL-VCCI Scope 3 Carbon Intelligence Platform on AWS.

## Architecture Overview

The infrastructure consists of:

- **EKS Cluster**: Multi-AZ Kubernetes cluster with 3 node groups (compute, memory, GPU)
- **RDS PostgreSQL**: Multi-AZ PostgreSQL 15.3 with read replicas
- **ElastiCache Redis**: Cluster mode with multi-AZ failover
- **S3 Buckets**: Provenance records, raw data, and reports with cross-region replication
- **VPC**: Multi-AZ VPC with public/private subnets and NAT gateways
- **IAM**: Service roles and IRSA for pod-level permissions
- **Monitoring**: CloudWatch logs, metrics, and alarms
- **Backup**: AWS Backup for disaster recovery

## Prerequisites

- Terraform >= 1.5.0
- AWS CLI configured with appropriate credentials
- kubectl for EKS cluster management
- helm (optional, for Kubernetes add-ons)

## Directory Structure

```
.
├── README.md                    # This file
├── main.tf                      # Root module orchestration
├── variables.tf                 # Root module variables
├── outputs.tf                   # Root module outputs
├── terraform.tfvars.example     # Example variable values
├── backend.tf                   # Remote state configuration
├── versions.tf                  # Terraform and provider versions
├── modules/                     # Reusable Terraform modules
│   ├── vpc/                     # VPC and networking
│   ├── eks/                     # EKS cluster and node groups
│   ├── rds/                     # RDS PostgreSQL
│   ├── elasticache/             # Redis cluster
│   ├── s3/                      # S3 buckets and policies
│   ├── iam/                     # IAM roles and policies
│   ├── monitoring/              # CloudWatch and SNS
│   └── backup/                  # AWS Backup configuration
├── environments/                # Environment-specific configurations
│   ├── dev/                     # Development environment
│   ├── staging/                 # Staging environment
│   └── production/              # Production environment
└── scripts/                     # Helper scripts
    ├── init.sh                  # Initialize Terraform
    ├── plan.sh                  # Plan changes
    ├── apply.sh                 # Apply changes
    └── destroy.sh               # Destroy infrastructure
```

## Quick Start

### 1. Initialize Backend

First, create the S3 bucket and DynamoDB table for remote state:

```bash
cd environments/production
./../../scripts/init.sh
```

### 2. Configure Variables

Copy and customize the example tfvars:

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 3. Plan Infrastructure

Review the infrastructure changes:

```bash
./../../scripts/plan.sh
```

### 4. Apply Infrastructure

Deploy the infrastructure:

```bash
./../../scripts/apply.sh
```

## Environment Configuration

### Development

Optimized for cost with minimal redundancy:

- Single-AZ deployment
- Smaller instance types
- No read replicas
- 1-day backup retention

```bash
cd environments/dev
terraform init
terraform plan
terraform apply
```

### Staging

Production-like environment for testing:

- Multi-AZ deployment
- Production instance types (smaller scale)
- 1 read replica
- 3-day backup retention

```bash
cd environments/staging
terraform init
terraform plan
terraform apply
```

### Production

Full production configuration:

- Multi-AZ deployment (us-west-2a, us-west-2b, us-west-2c)
- Production instance types
- 2 read replicas
- 7-day backup retention
- Cross-region replication
- Enhanced monitoring

```bash
cd environments/production
terraform init
terraform plan
terraform apply
```

## Resource Details

### EKS Cluster

- **Version**: Kubernetes 1.27
- **Node Groups**:
  - Compute: t3.xlarge (3-20 nodes) - General workloads
  - Memory: r6g.2xlarge (2-10 nodes) - Databases and caching
  - GPU: g4dn.xlarge (1-5 nodes) - ML workloads
- **Add-ons**:
  - VPC CNI
  - CoreDNS
  - kube-proxy
  - Cluster Autoscaler
  - AWS Load Balancer Controller

### RDS PostgreSQL

- **Engine**: PostgreSQL 15.3
- **Instance**: db.r6g.2xlarge (8 vCPU, 64GB RAM)
- **Storage**: 500GB gp3 (10,000 IOPS)
- **Backups**: Daily at 03:00 UTC, 7-day retention
- **Features**:
  - Multi-AZ deployment
  - 2 read replicas
  - Performance Insights
  - Encryption at rest (KMS)
  - Automated backups

### ElastiCache Redis

- **Engine**: Redis 7.0
- **Node Type**: cache.r6g.large (2 vCPU, 13.07GB RAM)
- **Cluster**: 3 shards × 2 replicas = 6 nodes
- **Features**:
  - Cluster mode enabled
  - Multi-AZ with automatic failover
  - Encryption at rest and in transit
  - Automatic backups

### S3 Buckets

1. **Provenance Records**: 1TB estimated
2. **Raw Data**: 5TB estimated
3. **Reports**: 100GB estimated

**Features**:
- Versioning enabled
- Cross-region replication (us-west-2 → eu-central-1)
- Lifecycle policies (IA after 90 days, Glacier after 1 year)
- Server-side encryption (AES-256)
- Access logging

### VPC

- **CIDR**: 10.0.0.0/16
- **Subnets**:
  - Public: 10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24
  - Private: 10.0.11.0/24, 10.0.12.0/24, 10.0.13.0/24
  - Database: 10.0.21.0/24, 10.0.22.0/24, 10.0.23.0/24
- **NAT Gateways**: One per AZ for HA
- **VPC Endpoints**: S3, RDS, ElastiCache

## Cost Estimates

### Production Environment (Monthly)

| Service | Cost |
|---------|------|
| EKS Control Plane | $200 |
| Compute Nodes (12 × t3.xlarge) | $1,500 |
| Memory Nodes (5 × r6g.2xlarge) | $2,000 |
| GPU Nodes (2 × g4dn.xlarge) | $500 |
| RDS PostgreSQL (primary + 2 replicas) | $800 |
| ElastiCache Redis (6 nodes) | $400 |
| S3 Storage (6TB + transfer) | $200 |
| Data Transfer | $300 |
| **Total** | **~$5,900/month** |

### Development Environment (Monthly)

| Service | Cost |
|---------|------|
| EKS Control Plane | $200 |
| Compute Nodes (3 × t3.medium) | $150 |
| RDS PostgreSQL (single-AZ) | $150 |
| ElastiCache Redis (2 nodes) | $100 |
| S3 Storage | $50 |
| **Total** | **~$650/month** |

## Security

### IAM Roles

- **EKS Cluster Role**: Managed by AWS
- **EKS Node Role**: EC2 instance profile for nodes
- **IRSA Roles**: Pod-level IAM roles for fine-grained permissions
- **Service Roles**: S3 access, RDS access, CloudWatch logs

### Network Security

- Private subnets for compute resources
- Security groups with least privilege
- NACLs for network-level filtering
- VPC endpoints for AWS service access without internet

### Encryption

- RDS: Encryption at rest with KMS
- ElastiCache: Encryption at rest and in transit
- S3: Server-side encryption (AES-256)
- EKS: Secrets encrypted with KMS

## Monitoring and Alerts

### CloudWatch Metrics

- EKS cluster metrics
- RDS performance metrics
- ElastiCache metrics
- Application logs

### Alarms

- High CPU utilization (> 80%)
- High memory utilization (> 85%)
- RDS connection count (> 80% of max)
- ElastiCache evictions (> 1000/min)
- S3 bucket size thresholds

### SNS Topics

- Critical alerts (PagerDuty integration)
- Warning alerts (Email notifications)
- Info alerts (Slack integration)

## Backup and Disaster Recovery

### RDS Backups

- Automated daily backups at 03:00 UTC
- 7-day retention (production)
- Manual snapshots for major changes
- Cross-region snapshot copies

### S3 Replication

- Cross-region replication to eu-central-1
- 15-minute RTO for bucket recovery
- Versioning enabled for point-in-time recovery

### AWS Backup

- Centralized backup management
- Backup plans for RDS and EBS volumes
- Retention policies by environment

### Disaster Recovery Procedures

1. **RDS Failure**: Promote read replica to primary (5-10 min RTO)
2. **AZ Failure**: Multi-AZ automatically fails over (< 2 min RTO)
3. **Region Failure**: Restore from cross-region backups (1-2 hour RTO)
4. **Data Corruption**: Point-in-time recovery from snapshots

## Terraform Commands

### Initialize

```bash
terraform init
```

### Validate

```bash
terraform validate
```

### Plan

```bash
terraform plan -out=tfplan
```

### Apply

```bash
terraform apply tfplan
```

### Destroy

```bash
terraform destroy
```

### Format

```bash
terraform fmt -recursive
```

### State Management

```bash
# List resources
terraform state list

# Show resource details
terraform state show <resource>

# Import existing resource
terraform import <resource> <id>

# Remove resource from state
terraform state rm <resource>
```

## Troubleshooting

### EKS Access Issues

If you can't access the EKS cluster:

```bash
aws eks update-kubeconfig --name vcci-scope3-production --region us-west-2
```

### State Lock Issues

If Terraform state is locked:

```bash
# Unlock state (use lock ID from error message)
terraform force-unlock <lock-id>
```

### RDS Connection Issues

Check security groups and network ACLs:

```bash
# Test connection from EKS pod
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h <rds-endpoint> -U postgres -d vcci_scope3
```

### S3 Access Issues

Verify IAM policies and bucket policies:

```bash
# Test S3 access
aws s3 ls s3://vcci-scope3-provenance-production/
```

## Maintenance

### Updating Kubernetes Version

1. Update `eks_cluster_version` in `variables.tf`
2. Plan and apply changes
3. Update node group AMIs
4. Drain and replace nodes

### Scaling Resources

Update auto-scaling configurations in environment tfvars:

```hcl
compute_node_group_desired_size = 15
compute_node_group_max_size     = 25
```

### Updating RDS

1. Update `engine_version` in `variables.tf`
2. Plan changes (will show maintenance window)
3. Apply during maintenance window
4. Monitor Performance Insights

## Support

For issues or questions:

- Platform Team: platform@greenlang.com
- DevOps Team: devops@greenlang.com
- On-call: Use PagerDuty for critical issues

## License

Proprietary - GreenLang Inc. All rights reserved.
