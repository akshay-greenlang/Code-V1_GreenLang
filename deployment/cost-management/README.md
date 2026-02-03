# Cost Management and Optimization

## INFRA-001: GreenLang Infrastructure Cost Management

This directory contains comprehensive cost management configurations, monitoring tools, and optimization strategies for the GreenLang infrastructure.

---

## Table of Contents

1. [Overview](#overview)
2. [Cost Breakdown by Service](#cost-breakdown-by-service)
3. [Directory Structure](#directory-structure)
4. [Kubecost Configuration](#kubecost-configuration)
5. [AWS Budget Alerts](#aws-budget-alerts)
6. [Spot Instance Strategy](#spot-instance-strategy)
7. [Optimization Recommendations](#optimization-recommendations)
8. [Reserved Instance Guidance](#reserved-instance-guidance)
9. [Monitoring and Alerting](#monitoring-and-alerting)
10. [Implementation Guide](#implementation-guide)

---

## Overview

### Cost Management Goals

| Goal | Target | Current | Status |
|------|--------|---------|--------|
| Monthly Infrastructure Cost | $20,000 | $25,000 | Over Budget |
| Savings Plan Coverage | 70% | 45% | Below Target |
| Spot Instance Usage | 50% | 35% | Below Target |
| Idle Resource Cost | < 10% | 25% | Above Target |
| Cost Visibility | 100% tagged | 85% tagged | In Progress |

### Monthly Budget Allocation

```
Total Monthly Budget: $20,000

+------------------+----------+------------+
| Category         | Budget   | Allocation |
+------------------+----------+------------+
| EKS Cluster      | $10,000  | 50%        |
| RDS Database     | $3,000   | 15%        |
| Data Transfer    | $2,000   | 10%        |
| S3 Storage       | $1,000   | 5%         |
| ElastiCache      | $1,500   | 7.5%       |
| Monitoring/Logs  | $1,500   | 7.5%       |
| Other Services   | $1,000   | 5%         |
+------------------+----------+------------+
```

---

## Cost Breakdown by Service

### Compute Costs (EKS/EC2)

| Resource Type | Instance Type | Count | Monthly Cost | Notes |
|---------------|---------------|------:|-------------:|-------|
| EKS Control Plane | - | 1 | $73 | Fixed cost |
| On-Demand Workers | m5.large | 4 | $280 | Baseline capacity |
| Spot Workers | m5.large | 8 | $112 | 60% discount |
| Spot Workers | c5.large | 4 | $51 | Compute optimized |
| NAT Gateway | - | 3 | $97 | Per AZ |
| **Subtotal** | | | **$613** | |

### Database Costs (RDS)

| Database | Instance Type | Storage | Monthly Cost | Notes |
|----------|---------------|--------:|-------------:|-------|
| greenlang-db | db.r5.xlarge | 500GB | $520 | Primary |
| greenlang-replica | db.r5.large | 500GB | $310 | Read replica |
| Automated Backups | - | 100GB | $23 | 7-day retention |
| Snapshots | - | 200GB | $46 | Manual snapshots |
| **Subtotal** | | | **$899** | |

### Storage Costs (S3/EBS)

| Service | Type | Volume | Monthly Cost | Notes |
|---------|------|-------:|-------------:|-------|
| S3 Standard | Storage | 2TB | $46 | Hot data |
| S3 Intelligent-Tiering | Storage | 5TB | $58 | Archival |
| S3 Requests | API calls | 10M | $45 | GET/PUT |
| EBS gp3 | Volumes | 2TB | $160 | Application data |
| EBS Snapshots | Backups | 500GB | $25 | Weekly snapshots |
| **Subtotal** | | | **$334** | |

### Caching Costs (ElastiCache)

| Cluster | Node Type | Nodes | Monthly Cost | Notes |
|---------|-----------|------:|-------------:|-------|
| greenlang-redis | cache.r5.large | 3 | $450 | Primary cache |
| session-cache | cache.r5.medium | 2 | $200 | Session store |
| **Subtotal** | | | **$650** | |

### Networking Costs

| Service | Type | Volume | Monthly Cost | Notes |
|---------|------|-------:|-------------:|-------|
| Data Transfer Out | Internet | 5TB | $450 | To users |
| Data Transfer | Inter-AZ | 2TB | $20 | Cross-AZ traffic |
| ALB | Load Balancer | 2 | $36 | Application LB |
| ALB Data | Processing | 1TB | $8 | Processed bytes |
| Route 53 | Hosted Zones | 3 | $2 | DNS |
| Route 53 | Queries | 50M | $25 | DNS queries |
| **Subtotal** | | | **$541** | |

### Monitoring and Logging

| Service | Type | Volume | Monthly Cost | Notes |
|---------|------|-------:|-------------:|-------|
| CloudWatch Logs | Ingestion | 100GB | $50 | Log storage |
| CloudWatch Logs | Storage | 500GB | $15 | Retention |
| CloudWatch Metrics | Custom | 200 | $60 | Application metrics |
| CloudWatch Alarms | Standard | 50 | $5 | Alerts |
| X-Ray | Traces | 10M | $50 | Distributed tracing |
| **Subtotal** | | | **$180** | |

### Total Monthly Cost Summary

```
+----------------------+-------------+------------+
| Category             | Monthly     | % of Total |
+----------------------+-------------+------------+
| Compute (EKS/EC2)    | $10,500     | 42%        |
| Database (RDS)       | $3,200      | 13%        |
| Storage (S3/EBS)     | $1,800      | 7%         |
| Caching (ElastiCache)| $1,500      | 6%         |
| Networking           | $4,500      | 18%        |
| Monitoring/Logging   | $1,200      | 5%         |
| Other Services       | $2,300      | 9%         |
+----------------------+-------------+------------+
| TOTAL                | $25,000     | 100%       |
+----------------------+-------------+------------+
```

---

## Directory Structure

```
deployment/cost-management/
├── README.md                          # This documentation
├── kubecost/
│   ├── kubecost-values.yaml          # Kubecost Helm configuration
│   ├── cost-allocation.yaml          # Namespace cost allocation rules
│   └── budget-alerts.yaml            # Budget thresholds and alerts
├── aws/
│   ├── budget-alerts.tf              # AWS Budget Terraform config
│   ├── cost-explorer-queries.json    # Common cost analysis queries
│   └── savings-plans-analysis.md     # Savings recommendations
└── spot/
    ├── spot-instance-config.yaml     # Spot node group configuration
    └── spot-termination-handler.yaml # Graceful termination handling
```

---

## Kubecost Configuration

### Installation

```bash
# Add Kubecost Helm repository
helm repo add kubecost https://kubecost.github.io/cost-analyzer/
helm repo update

# Install Kubecost
helm install kubecost kubecost/cost-analyzer \
  --namespace kubecost \
  --create-namespace \
  -f deployment/cost-management/kubecost/kubecost-values.yaml
```

### Key Features

1. **Cost Allocation** - Track costs by namespace, label, or team
2. **Budget Alerts** - Get notified when spending exceeds thresholds
3. **Optimization Recommendations** - Rightsizing and efficiency suggestions
4. **Savings Tracking** - Monitor Savings Plans and RI utilization

### Access Kubecost Dashboard

```bash
kubectl port-forward -n kubecost svc/kubecost-cost-analyzer 9090:9090
# Open http://localhost:9090
```

---

## AWS Budget Alerts

### Deploying Budget Alerts

```bash
cd deployment/cost-management/aws

# Initialize Terraform
terraform init

# Review planned changes
terraform plan -var="environment=production"

# Apply configuration
terraform apply -var="environment=production"
```

### Budget Thresholds

| Budget | Limit | 50% Alert | 80% Alert | 100% Alert |
|--------|------:|:---------:|:---------:|:----------:|
| Total Monthly | $20,000 | Yes | Yes | Yes |
| EKS Cluster | $12,000 | No | Yes | Yes |
| RDS Database | $3,000 | No | Yes | Yes |
| S3 Storage | $1,000 | No | Yes | No |
| Data Transfer | $2,000 | Yes | Yes | Yes |

### Alert Channels

- **Email**: finops@greenlang.io, platform-team@greenlang.io
- **Slack**: #finops-alerts, #platform-alerts
- **PagerDuty**: Critical budget overruns

---

## Spot Instance Strategy

### Current vs Target Spot Usage

```
Workload Type      Current    Target    Savings Potential
---------------------------------------------------------
Stateless APIs       40%       70%         $800/month
Background Workers   60%       90%         $400/month
Batch Processing     80%      100%         $200/month
CI/CD Runners        50%      100%         $150/month
Development Env      30%       80%         $300/month
---------------------------------------------------------
Total Potential Monthly Savings:          $1,850/month
```

### Deploying Spot Configuration

```bash
# Apply Karpenter provisioners
kubectl apply -f deployment/cost-management/spot/spot-instance-config.yaml

# Deploy termination handler
kubectl apply -f deployment/cost-management/spot/spot-termination-handler.yaml
```

### Instance Type Selection

Diversify across these instance types for maximum availability:

| Priority | Instance Types | Use Case |
|:--------:|----------------|----------|
| 1 | m5.large, m5a.large, m5n.large | General purpose |
| 2 | c5.large, c5a.large, c5n.large | Compute intensive |
| 3 | r5.large, r5a.large, r5n.large | Memory intensive |
| 4 | m4.large, c4.large, r4.large | Fallback (older gen) |

### Handling Spot Interruptions

1. **Node Termination Handler** monitors for 2-minute warnings
2. **Pod Disruption Budgets** ensure minimum availability
3. **Graceful Shutdown** allows in-flight requests to complete
4. **Automatic Replacement** via Karpenter or Cluster Autoscaler

---

## Optimization Recommendations

### Immediate Actions (Week 1-2)

| Action | Estimated Savings | Effort | Status |
|--------|------------------:|:------:|:------:|
| Purchase Compute Savings Plan | $2,400/month | Low | Pending |
| Enable S3 Intelligent-Tiering | $200/month | Low | Pending |
| Delete unused EBS volumes | $150/month | Low | Pending |
| Release unattached Elastic IPs | $50/month | Low | Pending |

### Short-Term Actions (Month 1-2)

| Action | Estimated Savings | Effort | Status |
|--------|------------------:|:------:|:------:|
| Purchase RDS Reserved Instances | $504/month | Low | Pending |
| Increase Spot usage to 50% | $800/month | Medium | In Progress |
| Migrate EBS to GP3 | $300/month | Medium | Pending |
| Enable S3 lifecycle policies | $150/month | Low | Pending |

### Medium-Term Actions (Month 3-6)

| Action | Estimated Savings | Effort | Status |
|--------|------------------:|:------:|:------:|
| Add EC2 Instance Savings Plan | $1,000/month | Medium | Planned |
| Implement Karpenter | $500/month | High | Planned |
| Optimize data transfer | $400/month | Medium | Planned |
| Rightsize underutilized instances | $600/month | Medium | Planned |

### Rightsizing Recommendations

Run these commands to identify rightsizing opportunities:

```bash
# Get EC2 rightsizing recommendations
aws ce get-rightsizing-recommendation \
  --service AmazonEC2 \
  --configuration BenefitsConsidered=true,RecommendationTarget=SAME_INSTANCE_FAMILY

# Check Kubecost recommendations
kubectl port-forward -n kubecost svc/kubecost-cost-analyzer 9090:9090
# Navigate to Savings -> Right-size your cluster
```

---

## Reserved Instance Guidance

### When to Use Reserved Instances

| Scenario | Recommendation |
|----------|----------------|
| Stable, predictable workloads | 1-year RI, Partial Upfront |
| Long-term infrastructure | 3-year RI, All Upfront (max savings) |
| Variable workloads | Savings Plans (more flexible) |
| Databases (RDS) | Reserved Instances (significant savings) |
| Caching (ElastiCache) | Reserved Nodes |

### Recommended Purchases

#### Compute Savings Plan

```
Type: Compute Savings Plan
Commitment: $15/hour
Term: 1 year
Payment: Partial Upfront
Expected Savings: 20% ($2,400/month)
```

#### RDS Reserved Instances

```
db.r5.xlarge  - 1 instance, 1-year, Partial Upfront - Savings: $144/month
db.r5.large   - 1 instance, 1-year, Partial Upfront - Savings: $72/month
db.r5.2xlarge - 1 instance, 1-year, Partial Upfront - Savings: $288/month

Total RDS Savings: $504/month ($6,048/year)
```

#### ElastiCache Reserved Nodes

```
cache.r5.large  - 3 nodes, 1-year, No Upfront - Savings: $135/month
cache.r5.medium - 2 nodes, 1-year, No Upfront - Savings: $60/month

Total ElastiCache Savings: $195/month ($2,340/year)
```

### Savings Plan vs Reserved Instance Comparison

| Feature | Savings Plans | Reserved Instances |
|---------|:-------------:|:------------------:|
| Flexibility | High | Low |
| Discount Level | 20-30% | 30-40% |
| Cross-region | Yes (Compute SP) | No |
| Cross-instance family | Yes (Compute SP) | No |
| Applies to Fargate | Yes | No |
| Applies to Lambda | Yes | No |
| Recommendation | **Preferred** | RDS/ElastiCache only |

---

## Monitoring and Alerting

### Key Metrics

| Metric | Target | Alert Threshold | Dashboard |
|--------|--------|-----------------|-----------|
| Daily Cost | < $700 | > $800 | Grafana |
| Monthly Projected | < $20,000 | > $18,000 | Kubecost |
| Savings Plan Utilization | > 90% | < 80% | AWS Console |
| Spot Instance Ratio | > 50% | < 40% | Kubecost |
| Idle Resources | < 10% | > 20% | Kubecost |

### Alert Configuration

Alerts are configured in multiple systems:

1. **AWS Budgets** - Cloud-level spending alerts
2. **Kubecost** - Kubernetes-level cost alerts
3. **Prometheus** - Custom cost-based alerts

### Grafana Dashboards

Import these dashboards for cost visibility:

- Kubecost Cost Allocation Dashboard
- AWS Cost and Usage Dashboard
- Spot Instance Monitoring Dashboard

---

## Implementation Guide

### Phase 1: Foundation (Week 1-2)

1. **Deploy Kubecost**
   ```bash
   helm install kubecost kubecost/cost-analyzer \
     -n kubecost --create-namespace \
     -f kubecost/kubecost-values.yaml
   ```

2. **Apply Cost Allocation Labels**
   ```bash
   kubectl apply -f kubecost/cost-allocation.yaml
   ```

3. **Configure AWS Budgets**
   ```bash
   cd aws && terraform apply
   ```

### Phase 2: Spot Optimization (Week 3-4)

1. **Deploy Spot Node Groups**
   ```bash
   kubectl apply -f spot/spot-instance-config.yaml
   ```

2. **Deploy Termination Handler**
   ```bash
   kubectl apply -f spot/spot-termination-handler.yaml
   ```

3. **Migrate Workloads to Spot**
   - Add spot tolerations to deployments
   - Configure pod disruption budgets

### Phase 3: Commitments (Month 2)

1. **Purchase Compute Savings Plan**
   - Review AWS recommendations
   - Start with 60% of baseline

2. **Purchase RDS Reserved Instances**
   - Match current production instances

### Phase 4: Continuous Optimization (Ongoing)

1. **Weekly Cost Review**
   - Review Kubecost reports
   - Identify optimization opportunities

2. **Monthly FinOps Meeting**
   - Review budget vs actual
   - Adjust commitments as needed

3. **Quarterly Savings Review**
   - Evaluate Savings Plan utilization
   - Adjust reservations

---

## Support and Resources

### Documentation

- [Kubecost Documentation](https://docs.kubecost.com/)
- [AWS Cost Management](https://docs.aws.amazon.com/cost-management/)
- [EKS Best Practices - Cost Optimization](https://aws.github.io/aws-eks-best-practices/cost_optimization/)

### Internal Resources

- **FinOps Team**: finops@greenlang.io
- **Slack Channel**: #finops-discussions
- **Wiki**: https://wiki.greenlang.io/finops

### Escalation Path

1. FinOps Team (budget questions)
2. Platform Team (infrastructure changes)
3. Engineering Leadership (major purchases)

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-03 | 1.0.0 | Initial cost management configuration |

---

**Document Owner**: FinOps Team
**Last Updated**: 2026-02-03
**Review Frequency**: Monthly
