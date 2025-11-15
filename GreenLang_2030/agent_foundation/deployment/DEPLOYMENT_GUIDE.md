# GreenLang AI Agent Foundation - Production Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Infrastructure Setup](#infrastructure-setup)
5. [Application Deployment](#application-deployment)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security & Compliance](#security--compliance)
8. [Disaster Recovery](#disaster-recovery)
9. [Scaling Guide](#scaling-guide)
10. [Troubleshooting](#troubleshooting)
11. [Cost Optimization](#cost-optimization)

---

## Overview

This guide provides comprehensive instructions for deploying the GreenLang AI Agent Foundation to production environments with high availability, security, and scalability.

### Key Features

- **Multi-AZ High Availability**: 9 pods distributed across 3 availability zones
- **Auto-Scaling**: Horizontal pod autoscaling (9-100 pods) based on CPU/memory
- **Zero-Downtime Deployments**: Rolling updates with maxUnavailable=0
- **Security Hardened**: Non-root containers, network policies, encrypted secrets
- **Full Observability**: Prometheus, Grafana, Jaeger, ELK stack
- **Infrastructure as Code**: Terraform for AWS/GCP/Azure
- **CI/CD Automation**: GitHub Actions and GitLab CI pipelines

---

## Prerequisites

### Required Tools

```bash
# Install required CLI tools
brew install kubectl helm terraform aws-cli # macOS
# or
choco install kubernetes-cli kubernetes-helm terraform awscli # Windows

# Verify installations
kubectl version --client
helm version
terraform version
aws --version
```

### Required Versions

- Kubernetes: >= 1.28
- Helm: >= 3.13
- Terraform: >= 1.6
- Docker: >= 24.0
- Python: 3.11+

### AWS Account Requirements

- IAM permissions for EKS, VPC, RDS, ElastiCache, S3, Secrets Manager
- VPC with at least 3 availability zones
- Route53 hosted zone (for DNS)
- ACM certificate (for HTTPS)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Internet Gateway                         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Network Load Balancer                       │
│         (Cross-zone, TLS termination, Session affinity)      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Kubernetes Ingress                        │
│              (NGINX, Rate limiting, CORS)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
    ┌──────▼──────┬──────▼──────┬──────▼──────┐
    │  AZ-1a      │  AZ-1b      │  AZ-1c      │
    │  3 Pods     │  3 Pods     │  3 Pods     │
    │             │             │             │
    │  Agent      │  Agent      │  Agent      │
    │  Instances  │  Instances  │  Instances  │
    └──────┬──────┴──────┬──────┴──────┬──────┘
           │             │             │
           └─────────────┼─────────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
    ┌──────▼──────┐ ┌────▼─────┐ ┌────▼─────┐
    │  RDS        │ │ Redis    │ │   S3     │
    │  PostgreSQL │ │ Cluster  │ │  Bucket  │
    │  Multi-AZ   │ │ Multi-AZ │ │          │
    └─────────────┘ └──────────┘ └──────────┘
```

### Component Breakdown

| Component | Purpose | HA Configuration |
|-----------|---------|------------------|
| **EKS Cluster** | Kubernetes orchestration | Multi-AZ control plane |
| **Agent Pods** | AI agent instances | 9 pods (3 per AZ) |
| **RDS PostgreSQL** | Primary database | Multi-AZ with automatic failover |
| **ElastiCache Redis** | Caching & sessions | Multi-AZ replication group |
| **S3** | Object storage | Multi-region replication |
| **Network Load Balancer** | Layer 4 load balancing | Cross-zone enabled |
| **Prometheus** | Metrics collection | HA with Thanos |
| **Grafana** | Dashboards | HA with external DB |

---

## Infrastructure Setup

### Step 1: Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Verify access
aws sts get-caller-identity
```

### Step 2: Initialize Terraform

```bash
cd infrastructure/terraform/aws

# Copy example tfvars
cp terraform.tfvars.example terraform.tfvars

# Edit variables for your environment
vim terraform.tfvars

# Initialize Terraform
terraform init

# Create workspace for environment
terraform workspace new production
# or
terraform workspace select production
```

### Step 3: Review and Apply Infrastructure

```bash
# Plan infrastructure changes
terraform plan -out=tfplan

# Review the plan carefully
# Apply infrastructure
terraform apply tfplan

# This will create:
# - VPC with 3 AZs
# - EKS cluster
# - RDS PostgreSQL (Multi-AZ)
# - ElastiCache Redis (Multi-AZ)
# - S3 buckets
# - IAM roles & policies
# - Security groups
# - Secrets Manager secrets
```

**Estimated Time**: 20-30 minutes

**Estimated Cost** (per month):
- Development: ~$500-800
- Staging: ~$1,200-1,800
- Production: ~$3,000-5,000

### Step 4: Configure kubectl

```bash
# Get the kubectl configuration command from Terraform output
terraform output configure_kubectl

# Example output:
# aws eks update-kubeconfig --region us-east-1 --name greenlang-production-eks

# Execute the command
aws eks update-kubeconfig --region us-east-1 --name greenlang-production-eks

# Verify connection
kubectl get nodes
```

---

## Application Deployment

### Step 1: Create Kubernetes Secrets

```bash
# Get database credentials from AWS Secrets Manager
export DB_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id greenlang-production-app-secrets \
  --query SecretString --output text | jq -r '.database_password')

# Create Kubernetes secret
kubectl create namespace greenlang-ai

kubectl create secret generic greenlang-secrets \
  --from-literal=database-url="postgresql://greenlang_admin:${DB_PASSWORD}@<RDS_ENDPOINT>:5432/greenlang" \
  --from-literal=redis-url="redis://<REDIS_ENDPOINT>:6379/0" \
  -n greenlang-ai

# Create registry secret for private Docker registry
kubectl create secret docker-registry greenlang-registry-secret \
  --docker-server=ghcr.io \
  --docker-username=<GITHUB_USERNAME> \
  --docker-password=<GITHUB_TOKEN> \
  -n greenlang-ai
```

### Step 2: Deploy with Helm

```bash
cd deployment/helm

# Add dependencies
helm dependency update greenlang-agent

# Install/Upgrade the release
helm upgrade --install greenlang-agent greenlang-agent/ \
  --namespace greenlang-ai \
  --create-namespace \
  --values greenlang-agent/values-production.yaml \
  --set image.tag=v1.0.0 \
  --wait \
  --timeout 10m

# Verify deployment
kubectl get pods -n greenlang-ai -o wide
kubectl get hpa -n greenlang-ai
kubectl get pdb -n greenlang-ai
```

### Step 3: Verify Pod Distribution

```bash
# Verify pods are distributed across availability zones
kubectl get pods -n greenlang-ai \
  -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,ZONE:.metadata.labels.topology\\.kubernetes\\.io/zone

# Expected output:
# NAME                              NODE                                          ZONE
# greenlang-agent-xxx-1             ip-10-0-1-100.ec2.internal                   us-east-1a
# greenlang-agent-xxx-2             ip-10-0-1-101.ec2.internal                   us-east-1a
# greenlang-agent-xxx-3             ip-10-0-1-102.ec2.internal                   us-east-1a
# greenlang-agent-xxx-4             ip-10-0-2-100.ec2.internal                   us-east-1b
# greenlang-agent-xxx-5             ip-10-0-2-101.ec2.internal                   us-east-1b
# greenlang-agent-xxx-6             ip-10-0-2-102.ec2.internal                   us-east-1b
# greenlang-agent-xxx-7             ip-10-0-3-100.ec2.internal                   us-east-1c
# greenlang-agent-xxx-8             ip-10-0-3-101.ec2.internal                   us-east-1c
# greenlang-agent-xxx-9             ip-10-0-3-102.ec2.internal                   us-east-1c
```

### Step 4: Configure DNS

```bash
# Get Load Balancer DNS name
export LB_DNS=$(kubectl get svc greenlang-agent-lb \
  -n greenlang-ai \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "Load Balancer DNS: $LB_DNS"

# Create Route53 record
aws route53 change-resource-record-sets \
  --hosted-zone-id <HOSTED_ZONE_ID> \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.greenlang.io",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [{"Value": "'$LB_DNS'"}]
      }
    }]
  }'
```

---

## Monitoring & Observability

### Deploy Monitoring Stack

```bash
# Deploy Prometheus Operator
kubectl create namespace monitoring

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values deployment/monitoring/prometheus-values.yaml

# Deploy Grafana dashboards
kubectl apply -f deployment/monitoring/grafana_dashboards/ -n monitoring

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Login: admin / prom-operator
# Navigate to: http://localhost:3000
```

### Key Dashboards

1. **Executive Dashboard**: High-level business metrics
2. **Operations Dashboard**: Infrastructure and pod health
3. **Agents Dashboard**: AI agent performance and tasks
4. **Quality Dashboard**: Error rates and latencies
5. **Financial Dashboard**: Cost metrics and resource usage

### Alerting

Alerts are configured for:
- Pod crashloop backoff
- High CPU/memory usage
- API latency > 1s
- Error rate > 5%
- Database connection failures
- Redis unavailability

Notifications sent to:
- Slack (#alerts channel)
- PagerDuty (critical only)
- Email (ops team)

---

## Security & Compliance

### Security Best Practices

1. **Network Security**
   - Network policies restrict pod-to-pod communication
   - Private subnets for application tier
   - Security groups with least privilege

2. **Container Security**
   - Non-root containers (UID 1000)
   - Read-only root filesystem
   - Security scanning with Trivy and Snyk
   - No privileged containers

3. **Secrets Management**
   - AWS Secrets Manager for sensitive data
   - Kubernetes secrets encrypted at rest (KMS)
   - Secrets rotation every 90 days
   - No secrets in environment variables

4. **Access Control**
   - RBAC with least privilege
   - IAM roles for service accounts (IRSA)
   - MFA required for production access
   - Audit logging enabled

### Compliance

- **SOC 2 Type II**: Ready (encryption, access controls, logging)
- **GDPR**: Compliant (data encryption, audit trails)
- **HIPAA**: Can be enabled (BAA required)
- **PCI DSS**: Level 1 ready

---

## Disaster Recovery

### Backup Strategy

| Resource | Backup Frequency | Retention | Recovery Time |
|----------|------------------|-----------|---------------|
| RDS PostgreSQL | Every 6 hours | 30 days | < 15 minutes |
| Redis (AOF) | Continuous | 7 days | < 5 minutes |
| S3 Data | Versioned | 365 days | Immediate |
| Kubernetes Manifests | Git commits | Forever | < 10 minutes |

### Recovery Procedures

#### Database Recovery

```bash
# List available snapshots
aws rds describe-db-snapshots \
  --db-instance-identifier greenlang-production-db

# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier greenlang-production-db-restored \
  --db-snapshot-identifier <SNAPSHOT_ID> \
  --db-subnet-group-name greenlang-production-db-subnet

# Update Kubernetes secret with new endpoint
kubectl edit secret greenlang-secrets -n greenlang-ai
```

#### Application Rollback

```bash
# View deployment history
kubectl rollout history deployment/greenlang-agent -n greenlang-ai

# Rollback to previous version
kubectl rollout undo deployment/greenlang-agent -n greenlang-ai

# Rollback to specific revision
kubectl rollout undo deployment/greenlang-agent \
  -n greenlang-ai \
  --to-revision=3
```

---

## Scaling Guide

### Horizontal Scaling

Scaling is automatic via HPA, but can be manually adjusted:

```bash
# Manual scaling
kubectl scale deployment greenlang-agent \
  --replicas=27 \
  -n greenlang-ai

# Adjust HPA limits
kubectl patch hpa greenlang-agent-hpa \
  -n greenlang-ai \
  --patch '{"spec":{"maxReplicas":200}}'
```

### Vertical Scaling

To increase pod resources:

```bash
# Edit deployment
kubectl edit deployment greenlang-agent -n greenlang-ai

# Update resources:
# resources:
#   requests:
#     memory: "4Gi"
#     cpu: "2000m"
#   limits:
#     memory: "8Gi"
#     cpu: "4000m"
```

### Database Scaling

```bash
# Modify RDS instance class
aws rds modify-db-instance \
  --db-instance-identifier greenlang-production-db \
  --db-instance-class db.r6g.2xlarge \
  --apply-immediately
```

---

## Troubleshooting

### Common Issues

#### Pods stuck in Pending

```bash
# Check pod events
kubectl describe pod <POD_NAME> -n greenlang-ai

# Common causes:
# 1. Insufficient resources
kubectl get nodes
kubectl top nodes

# 2. Pod anti-affinity cannot be satisfied
kubectl get pods -n greenlang-ai -o wide

# 3. PVC not bound
kubectl get pvc -n greenlang-ai
```

#### Database connection failures

```bash
# Verify database endpoint
kubectl get secret greenlang-secrets -n greenlang-ai -o jsonpath='{.data.database-url}' | base64 -d

# Test connection from pod
kubectl run -it --rm debug \
  --image=postgres:16-alpine \
  --restart=Never \
  -- psql <DATABASE_URL>

# Check security groups
aws ec2 describe-security-groups \
  --group-ids <RDS_SG_ID>
```

#### High latency

```bash
# Check pod metrics
kubectl top pods -n greenlang-ai

# Check HPA status
kubectl get hpa -n greenlang-ai

# View Prometheus metrics
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Query: http_request_duration_seconds{quantile="0.99"}
```

---

## Cost Optimization

### Estimated Monthly Costs

#### Development Environment
- EKS Cluster: $72
- EC2 Instances (3x m5.xlarge): $360
- RDS (db.t3.medium): $85
- ElastiCache (cache.t3.medium): $50
- Data Transfer: $30
- **Total: ~$600/month**

#### Production Environment
- EKS Cluster: $72
- EC2 Instances (9x m5.2xlarge): $2,160
- RDS Multi-AZ (db.r6g.xlarge): $450
- ElastiCache Multi-AZ (3x cache.r6g.large): $540
- S3 Storage (1TB): $23
- Data Transfer: $200
- CloudWatch Logs: $50
- **Total: ~$3,500/month**

### Cost Reduction Strategies

1. **Use Spot Instances for non-production**
   ```hcl
   # In Terraform
   capacity_type = "SPOT"
   ```
   Savings: ~70% on compute

2. **Right-size resources**
   - Enable VPA to recommend optimal sizes
   - Review and adjust based on actual usage

3. **Reserved Instances**
   - 1-year RDS Reserved Instance: 40% savings
   - 3-year Reserved Instance: 60% savings

4. **S3 Lifecycle Policies**
   - Move to Glacier after 90 days
   - Delete old versions after 1 year

5. **Auto-scaling**
   - Scale down during off-peak hours
   - Use KEDA for event-driven scaling

---

## Support

- Documentation: https://docs.greenlang.io
- Issues: https://github.com/greenlang/agent-foundation/issues
- Slack: #greenlang-ops
- Email: devops@greenlang.io

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0
**Maintainer**: GreenLang DevOps Team
