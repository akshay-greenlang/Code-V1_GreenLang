# AWS Savings Plans Analysis and Recommendations

## INFRA-001: Cost Management and Optimization

**Document Version:** 1.0.0
**Last Updated:** 2026-02-03
**Owner:** FinOps Team

---

## Executive Summary

This document provides a comprehensive analysis of AWS Savings Plans and Reserved Instances for GreenLang infrastructure, with recommendations for optimizing cloud costs while maintaining operational flexibility.

### Current State Overview

| Metric | Current Value | Target | Status |
|--------|---------------|--------|--------|
| Monthly AWS Spend | $25,000 | $20,000 | Over Budget |
| Savings Plan Coverage | 45% | 70% | Below Target |
| RI Coverage | 30% | 40% | Below Target |
| Spot Instance Usage | 35% | 50% | Below Target |
| On-Demand Usage | 55% | 20% | Above Target |

### Projected Savings

| Optimization | Annual Savings | Implementation Effort |
|--------------|----------------|----------------------|
| Compute Savings Plans | $36,000 | Low |
| EC2 Instance Savings Plans | $24,000 | Medium |
| RDS Reserved Instances | $12,000 | Low |
| Increased Spot Usage | $18,000 | Medium |
| **Total Potential Savings** | **$90,000** | - |

---

## 1. Savings Plans Analysis

### 1.1 Compute Savings Plans (Recommended)

Compute Savings Plans provide the most flexibility, applying to EC2, Fargate, and Lambda usage across any region, instance family, OS, or tenancy.

#### Recommended Commitment

```
Hourly Commitment: $20/hour
Term: 1-year
Payment Option: Partial Upfront
Estimated Monthly Cost: $14,400
Estimated Monthly Savings: $3,000 (20%)
```

#### Coverage Analysis

| Service | Current On-Demand Spend | Covered by SP | Remaining |
|---------|------------------------|---------------|-----------|
| EC2 | $12,000/month | $9,600 | $2,400 |
| EKS (Fargate) | $3,000/month | $2,400 | $600 |
| Lambda | $500/month | $400 | $100 |

#### Pros and Cons

**Pros:**
- Maximum flexibility across compute services
- Automatic application to most cost-effective resources
- No modification needed when changing instance types
- Covers Fargate and Lambda

**Cons:**
- Slightly lower discount than EC2 Instance Savings Plans
- Requires commitment regardless of usage changes

### 1.2 EC2 Instance Savings Plans

EC2 Instance Savings Plans offer deeper discounts but are limited to specific instance families within a region.

#### Recommended Commitment

```
Instance Family: m5, c5, r5
Region: us-east-1
Hourly Commitment: $8/hour
Term: 1-year
Payment Option: All Upfront
Estimated Monthly Cost: $5,760
Estimated Monthly Savings: $1,440 (25%)
```

#### Instance Family Analysis

| Instance Family | Monthly Usage Hours | On-Demand Cost | SP Cost | Savings |
|-----------------|--------------------:|---------------:|--------:|--------:|
| m5 (General) | 2,160 | $4,320 | $3,240 | $1,080 |
| c5 (Compute) | 1,440 | $2,880 | $2,160 | $720 |
| r5 (Memory) | 720 | $1,800 | $1,350 | $450 |

### 1.3 Savings Plans Recommendations Summary

#### Phase 1: Immediate (Month 1)

1. **Purchase Compute Savings Plan**
   - Amount: $15/hour commitment
   - Term: 1-year, Partial Upfront
   - Estimated Savings: $2,400/month

2. **Enable Savings Plans Auto-Renewal**
   - Prevent coverage gaps
   - Review quarterly for adjustments

#### Phase 2: Optimization (Month 3)

1. **Add EC2 Instance Savings Plan**
   - Instance Family: m5 (most stable usage)
   - Amount: $5/hour commitment
   - Term: 1-year, All Upfront
   - Estimated Savings: $1,000/month

2. **Review and Adjust Coverage**
   - Analyze utilization reports
   - Adjust commitments based on actual usage

---

## 2. Reserved Instances Analysis

### 2.1 RDS Reserved Instances

#### Current RDS Usage

| Instance | Type | Current Cost | RI Cost (1-yr) | Savings |
|----------|------|-------------:|--------------:|--------:|
| greenlang-db | db.r5.xlarge | $480/month | $336/month | $144/month |
| greenlang-replica | db.r5.large | $240/month | $168/month | $72/month |
| analytics-db | db.r5.2xlarge | $960/month | $672/month | $288/month |

#### Recommendation

```
Purchase RDS Reserved Instances:
- db.r5.xlarge: 1 instance, 1-year, Partial Upfront
- db.r5.large: 1 instance, 1-year, Partial Upfront
- db.r5.2xlarge: 1 instance, 1-year, Partial Upfront

Total Monthly Savings: $504/month ($6,048/year)
```

### 2.2 ElastiCache Reserved Nodes

#### Current ElastiCache Usage

| Cluster | Node Type | Nodes | Monthly Cost | RI Cost | Savings |
|---------|-----------|------:|-------------:|--------:|--------:|
| greenlang-redis | cache.r5.large | 3 | $450/month | $315/month | $135/month |
| session-cache | cache.r5.medium | 2 | $200/month | $140/month | $60/month |

#### Recommendation

```
Purchase ElastiCache Reserved Nodes:
- cache.r5.large: 3 nodes, 1-year, No Upfront
- cache.r5.medium: 2 nodes, 1-year, No Upfront

Total Monthly Savings: $195/month ($2,340/year)
```

---

## 3. Spot Instance Strategy

### 3.1 Current Spot Usage

| Workload | Current Spot % | Target Spot % | Monthly Savings Potential |
|----------|---------------:|---------------:|-------------------------:|
| EKS Worker Nodes | 40% | 70% | $1,200 |
| Batch Processing | 60% | 90% | $600 |
| Development/Staging | 20% | 80% | $400 |
| CI/CD Runners | 0% | 100% | $300 |

### 3.2 Spot Instance Recommendations

#### Production EKS Cluster

```yaml
# Recommended node group configuration
managedNodeGroups:
  - name: spot-workers
    instanceTypes:
      - m5.large
      - m5a.large
      - m5n.large
      - m4.large
    capacityType: SPOT
    desiredCapacity: 6
    minSize: 4
    maxSize: 12

  - name: on-demand-baseline
    instanceTypes:
      - m5.large
    capacityType: ON_DEMAND
    desiredCapacity: 2
    minSize: 2
    maxSize: 4
```

#### Spot Best Practices

1. **Diversify Instance Types**
   - Use multiple instance types per node group
   - Spread across availability zones
   - Enable capacity-optimized allocation strategy

2. **Implement Graceful Termination**
   - Use AWS Node Termination Handler
   - Configure pod disruption budgets
   - Enable spot instance draining

3. **Monitor Spot Interruptions**
   - Track interruption frequency by instance type
   - Adjust instance selection based on interruption rates
   - Set up alerts for high interruption periods

### 3.3 Workload Classification for Spot

| Workload Type | Spot Suitability | Recommended Spot % |
|---------------|------------------|-------------------:|
| Stateless API Services | High | 70-80% |
| Background Workers | Very High | 90-100% |
| Batch Processing | Very High | 100% |
| Databases | Not Suitable | 0% |
| Caching Layers | Medium | 40-50% |
| CI/CD Pipelines | Very High | 100% |

---

## 4. Cost Optimization Recommendations

### 4.1 Immediate Actions (Week 1-2)

| Action | Estimated Savings | Effort |
|--------|------------------:|--------|
| Purchase Compute Savings Plan ($15/hr) | $2,400/month | Low |
| Enable S3 Intelligent-Tiering | $200/month | Low |
| Delete unused EBS volumes | $150/month | Low |
| Remove unattached Elastic IPs | $50/month | Low |

### 4.2 Short-Term Actions (Month 1-2)

| Action | Estimated Savings | Effort |
|--------|------------------:|--------|
| Purchase RDS Reserved Instances | $504/month | Low |
| Increase Spot instance usage to 50% | $800/month | Medium |
| Implement GP3 EBS migration | $300/month | Medium |
| Enable S3 lifecycle policies | $150/month | Low |

### 4.3 Medium-Term Actions (Month 3-6)

| Action | Estimated Savings | Effort |
|--------|------------------:|--------|
| Add EC2 Instance Savings Plan | $1,000/month | Medium |
| Implement Karpenter for EKS | $500/month | High |
| Optimize data transfer costs | $400/month | Medium |
| Right-size underutilized instances | $600/month | Medium |

### 4.4 Long-Term Actions (Month 6-12)

| Action | Estimated Savings | Effort |
|--------|------------------:|--------|
| Multi-region optimization | $800/month | High |
| Graviton2 migration (ARM) | $700/month | High |
| Implement FinOps automation | $500/month | Medium |

---

## 5. Monitoring and Governance

### 5.1 Key Metrics to Track

```yaml
metrics:
  coverage:
    - name: "Savings Plan Coverage"
      target: ">= 70%"
      alert_threshold: "< 60%"
    - name: "Reserved Instance Coverage"
      target: ">= 40%"
      alert_threshold: "< 30%"
    - name: "Spot Instance Ratio"
      target: ">= 50%"
      alert_threshold: "< 40%"

  utilization:
    - name: "Savings Plan Utilization"
      target: ">= 90%"
      alert_threshold: "< 80%"
    - name: "RI Utilization"
      target: ">= 90%"
      alert_threshold: "< 80%"

  cost:
    - name: "Monthly Total Cost"
      budget: "$20,000"
      alert_thresholds: [80%, 90%, 100%]
    - name: "Cost per Request"
      baseline: "$0.0001"
      alert_threshold: "> $0.00015"
```

### 5.2 Review Cadence

| Review Type | Frequency | Participants | Actions |
|-------------|-----------|--------------|---------|
| Daily Cost Check | Daily | FinOps | Anomaly detection |
| Weekly Cost Review | Weekly | FinOps, SRE | Trend analysis |
| Monthly FinOps Review | Monthly | All Teams | Budget review |
| Quarterly SP/RI Review | Quarterly | FinOps, Leadership | Commitment adjustments |

### 5.3 Automation Recommendations

1. **Cost Anomaly Detection**
   - Enable AWS Cost Anomaly Detection
   - Set up automated alerts for > 20% variance
   - Integrate with Slack/PagerDuty

2. **Automated Reporting**
   - Daily cost summary to Slack
   - Weekly detailed report via email
   - Monthly executive summary

3. **Resource Cleanup Automation**
   - Auto-delete unused EBS volumes after 7 days
   - Auto-release unattached EIPs after 24 hours
   - Auto-stop idle development instances

---

## 6. Implementation Timeline

```
Month 1:
├── Week 1: Purchase Compute Savings Plan ($15/hr)
├── Week 2: Enable S3 Intelligent-Tiering, delete unused resources
├── Week 3: Purchase RDS Reserved Instances
└── Week 4: Review and validate savings

Month 2:
├── Week 1-2: Increase Spot usage to 50%
├── Week 3: Implement GP3 EBS migration
└── Week 4: Enable S3 lifecycle policies

Month 3:
├── Week 1: Add EC2 Instance Savings Plan
├── Week 2-3: Implement Karpenter
└── Week 4: Quarterly review and adjustments
```

---

## 7. Risk Mitigation

### 7.1 Commitment Risks

| Risk | Mitigation |
|------|------------|
| Overcommitment | Start with 60% of baseline usage |
| Usage decline | Use convertible options where available |
| Service migration | Prefer Compute SP over EC2 SP |
| Price changes | Monitor AWS pricing announcements |

### 7.2 Spot Instance Risks

| Risk | Mitigation |
|------|------------|
| Capacity unavailability | Diversify instance types and AZs |
| Interruptions | Implement graceful termination handling |
| Workload impact | Maintain on-demand baseline (30%) |
| Cost spikes | Set Spot max price caps |

---

## Appendix A: AWS CLI Commands

### Check Current Savings Plans

```bash
# List all Savings Plans
aws savingsplans describe-savings-plans

# Get Savings Plans utilization
aws ce get-savings-plans-utilization \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY

# Get Savings Plans coverage
aws ce get-savings-plans-coverage \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY
```

### Check Reserved Instances

```bash
# List EC2 Reserved Instances
aws ec2 describe-reserved-instances

# List RDS Reserved Instances
aws rds describe-reserved-db-instances

# Get RI utilization
aws ce get-reservation-utilization \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY
```

### Get Recommendations

```bash
# Savings Plans recommendations
aws ce get-savings-plans-purchase-recommendation \
  --savings-plans-type COMPUTE_SP \
  --term-in-years ONE_YEAR \
  --payment-option PARTIAL_UPFRONT \
  --lookback-period-in-days SIXTY_DAYS

# RI recommendations
aws ce get-reservation-purchase-recommendation \
  --service "Amazon Elastic Compute Cloud - Compute" \
  --lookback-period-in-days SIXTY_DAYS
```

---

## Appendix B: Cost Allocation Tags

Ensure all resources have the following tags for accurate cost tracking:

```yaml
required_tags:
  - Key: Project
    Value: GreenLang
  - Key: Environment
    Value: production|staging|development
  - Key: CostCenter
    Value: CC-XXXX
  - Key: Team
    Value: platform|data|ml|sre
  - Key: Owner
    Value: team-email@greenlang.io

optional_tags:
  - Key: Application
    Value: api|worker|scheduler
  - Key: Component
    Value: frontend|backend|database
```

---

## Document Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| FinOps Lead | | | |
| Engineering Director | | | |
| CFO | | | |
