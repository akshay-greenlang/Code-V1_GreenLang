# GreenLang Platform - Cost Estimation Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-08
**Currency:** USD

---

## Cost Summary by Deployment Size

| Component | Small | Medium | Large | Enterprise |
|-----------|-------|--------|-------|------------|
| **Organizations** | 1-10 | 10-100 | 100-1000 | 1000+ |
| **Users** | 1-100 | 100-1000 | 1000-10K | 10K+ |
| **Transactions/Day** | 10K | 100K | 1M | 10M+ |
| **Total Monthly Cost** | **$626** | **$4,629** | **$26,917** | **Custom** |

---

## Detailed Cost Breakdown

### Small Deployment (1-10 orgs, 100 users)

#### AWS (us-east-1) - Monthly Costs

| Service | Specification | Hours | Unit Cost | Monthly Cost |
|---------|---------------|-------|-----------|--------------|
| **EC2 (Application Servers)** |
| App Server 1 | t3.large (2 vCPU, 8 GB) | 730 | $0.0832/hr | $60.74 |
| App Server 2 | t3.large (2 vCPU, 8 GB) | 730 | $0.0832/hr | $60.74 |
| **Database** |
| RDS PostgreSQL | db.t3.medium (Multi-AZ) | 730 | $0.204/hr | $148.92 |
| **Cache** |
| ElastiCache Redis | cache.t3.medium | 730 | $0.109/hr | $79.57 |
| **Storage** |
| S3 Storage | 500 GB | - | $0.023/GB | $11.50 |
| S3 Data Transfer | 1 TB | - | $0.09/GB | $90.00 |
| EBS Volumes | 200 GB (gp3) | - | $0.08/GB | $16.00 |
| **Networking** |
| Application Load Balancer | 1 ALB | 730 | $0.0225/hr | $16.43 |
| ALB LCU | ~100 LCUs | - | $0.008/LCU | $8.00 |
| **DNS** |
| Route 53 | 1 hosted zone | - | $0.50/zone | $0.50 |
| **Monitoring** |
| CloudWatch Logs | 10 GB ingested | - | $0.50/GB | $5.00 |
| CloudWatch Metrics | 100 metrics | - | $0.30/metric | $30.00 |
| **Subtotal Infrastructure** | | | | **$527.40** |
| **LLM API Costs** |
| OpenAI GPT-4 | ~5K requests/day | - | ~$0.01/req | $150.00 |
| Anthropic Claude | ~5K requests/day | - | ~$0.01/req | $50.00 |
| **Subtotal LLM** | | | | **$200.00** |
| **TOTAL MONTHLY** | | | | **$727.40** |

#### Azure (East US) - Monthly Costs

| Service | Specification | Monthly Cost |
|---------|---------------|--------------|
| Virtual Machines | 2x B2s (2 vCPU, 4 GB) | $62.00 |
| Azure Database for PostgreSQL | General Purpose (4 vCPU) | $340.00 |
| Azure Cache for Redis | Basic C1 (1 GB) | $54.74 |
| Blob Storage | 500 GB + 1 TB transfer | $85.00 |
| Load Balancer | Standard | $22.00 |
| DNS Zone | 1 zone | $0.90 |
| Monitor | Logs + Metrics | $50.00 |
| **Subtotal Infrastructure** | | **$614.64** |
| **LLM API Costs** | | **$200.00** |
| **TOTAL MONTHLY** | | **$814.64** |

#### GCP (us-central1) - Monthly Costs

| Service | Specification | Monthly Cost |
|---------|---------------|--------------|
| Compute Engine | 2x n1-standard-2 | $97.09 |
| Cloud SQL PostgreSQL | db-n1-standard-2 (HA) | $236.98 |
| Memorystore Redis | Basic M1 (1 GB) | $46.21 |
| Cloud Storage | 500 GB + 1 TB transfer | $83.50 |
| Cloud Load Balancing | Standard | $18.00 |
| Cloud DNS | 1 zone | $0.20 |
| Cloud Monitoring | | $45.00 |
| **Subtotal Infrastructure** | | **$526.98** |
| **LLM API Costs** | | **$200.00** |
| **TOTAL MONTHLY** | | **$726.98** |

---

### Medium Deployment (10-100 orgs, 1000 users)

#### AWS (us-east-1) - Monthly Costs

| Service | Specification | Monthly Cost |
|---------|---------------|--------------|
| **Compute** |
| ECS Fargate (Web) | 4 vCPUs, 16 GB RAM | $576.00 |
| ECS Fargate (Workers) | 8 vCPUs, 32 GB RAM | $1,152.00 |
| **Database** |
| RDS PostgreSQL | db.m5.xlarge (Multi-AZ) | $600.00 |
| RDS Read Replica | db.m5.large | $300.00 |
| **Cache** |
| ElastiCache Redis | cache.m5.large (cluster) | $400.00 |
| **Vector Database** |
| EC2 for Weaviate | 2x m5.2xlarge | $576.00 |
| **Storage** |
| S3 Storage | 5 TB | $115.00 |
| S3 Data Transfer | 10 TB | $900.00 |
| EBS Volumes | 1 TB (gp3) | $80.00 |
| **Networking** |
| Application Load Balancer | 2 ALBs | $50.00 |
| **Monitoring** |
| CloudWatch | | $100.00 |
| **Subtotal Infrastructure** | | **$4,849.00** |
| **LLM API Costs** |
| OpenAI (100K req/day) | | $1,000.00 |
| Anthropic (50K req/day) | | $500.00 |
| **Subtotal LLM** | | **$1,500.00** |
| **TOTAL MONTHLY** | | **$6,349.00** |

**Cost Optimization Opportunities:**
- Reserved Instances (1-year): Save ~30% ($1,450/month)
- Spot Instances for workers: Save ~50% on worker costs ($576/month)
- S3 Intelligent Tiering: Save ~20% on storage ($20/month)
- **Optimized Total: ~$4,303/month**

---

### Large Deployment (100-1000 orgs, 10K users)

#### AWS (us-east-1) - Monthly Costs

| Service | Specification | Monthly Cost |
|---------|---------------|--------------|
| **Compute** |
| ECS Fargate (Web) | 32 vCPUs, 128 GB RAM | $4,608.00 |
| ECS Fargate (Workers) | 64 vCPUs, 256 GB RAM | $9,216.00 |
| **Database** |
| RDS PostgreSQL | db.r5.4xlarge (Multi-AZ) | $3,600.00 |
| RDS Read Replicas | 2x db.r5.2xlarge | $3,600.00 |
| **Cache** |
| ElastiCache Redis | cache.r5.2xlarge (cluster) | $1,800.00 |
| **Vector Database** |
| EC2 for Weaviate | 4x r5.4xlarge | $4,608.00 |
| **Storage** |
| S3 Storage | 50 TB | $1,150.00 |
| S3 Data Transfer | 100 TB | $9,000.00 |
| EBS Volumes | 5 TB (gp3) | $400.00 |
| **Networking** |
| Application Load Balancer | 4 ALBs | $100.00 |
| **Monitoring** |
| CloudWatch | | $500.00 |
| Sentry (SaaS) | | $200.00 |
| **Subtotal Infrastructure** | | **$38,782.00** |
| **LLM API Costs** |
| OpenAI (500K req/day) | | $5,000.00 |
| Anthropic (500K req/day) | | $5,000.00 |
| **Subtotal LLM** | | **$10,000.00** |
| **TOTAL MONTHLY** | | **$48,782.00** |

**Cost Optimization Opportunities:**
- Reserved Instances (1-year): Save ~35% ($13,574/month)
- Reserved Instances (3-year): Save ~50% ($19,391/month)
- Spot Instances for workers: Save ~60% on worker costs ($5,530/month)
- S3 Intelligent Tiering + Glacier: Save ~30% on storage ($300/month)
- LLM Response Caching (80% hit rate): Save ~80% on LLM ($8,000/month)
- **Optimized Total (1-year RI + optimizations): ~$21,987/month (55% savings)**

---

## LLM API Cost Deep Dive

### Request Volume Assumptions

| App | Requests/Day | % GPT-4 | % GPT-3.5 | % Claude Opus | % Claude Sonnet |
|-----|--------------|---------|-----------|---------------|-----------------|
| **CBAM** | 0 | 0% | 0% | 0% | 0% |
| **CSRD** | 70% | 20% | 30% | 30% | 20% |
| **VCCI** | 30% | 10% | 10% | 10% | 10% |

### Pricing (as of 2025-01)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Avg Cost/Request |
|-------|----------------------|------------------------|------------------|
| GPT-4 Turbo | $10.00 | $30.00 | $0.05 |
| GPT-3.5 Turbo | $0.50 | $1.50 | $0.002 |
| Claude Opus | $15.00 | $75.00 | $0.08 |
| Claude Sonnet | $3.00 | $15.00 | $0.015 |
| Claude Haiku | $0.25 | $1.25 | $0.001 |

### Cost Calculation (Medium Deployment, 100K req/day)

```
GPT-4 (20K req/day):      20,000 × $0.05 × 30 = $30,000/month
GPT-3.5 (30K req/day):    30,000 × $0.002 × 30 = $1,800/month
Claude Opus (30K req/day): 30,000 × $0.08 × 30 = $72,000/month
Claude Sonnet (20K req/day): 20,000 × $0.015 × 30 = $9,000/month

Total without optimization: $112,800/month
```

**Cost Reduction Strategies:**

1. **Caching (80% hit rate)**
   - Effective requests: 20K/day instead of 100K/day
   - Cost after caching: $22,560/month
   - **Savings: $90,240/month (80%)**

2. **Model Selection (use cheaper models where appropriate)**
   - Use Claude Haiku for simple text processing
   - Use GPT-3.5 for classification tasks
   - Reserve GPT-4/Opus for complex analysis
   - **Additional savings: ~30% → $15,792/month total**

3. **Batch Processing**
   - Batch 10 requests into 1 (where applicable)
   - **Additional savings: ~10% → $14,213/month total**

**Optimized LLM Cost: ~$1,500/month (87% reduction)**

---

## Cost Breakdown by Application

### GL-CBAM-APP (Standalone)

| Resource | Small | Medium | Large |
|----------|-------|--------|-------|
| Compute | $60 | $150 | $600 |
| Storage | $10 | $30 | $100 |
| LLM | $0 | $0 | $0 |
| **Total** | **$70/mo** | **$180/mo** | **$700/mo** |

*Note: CBAM uses zero hallucination architecture (no LLM)*

### GL-CSRD-APP

| Resource | Small | Medium | Large |
|----------|-------|--------|-------|
| Compute | $120 | $600 | $3,000 |
| Database | $150 | $450 | $2,000 |
| Cache | $40 | $200 | $900 |
| Storage | $30 | $100 | $400 |
| LLM | $150 | $1,000 | $5,000 |
| **Total** | **$490/mo** | **$2,350/mo** | **$11,300/mo** |

### GL-VCCI-APP

| Resource | Small | Medium | Large |
|----------|-------|--------|-------|
| Compute | $180 | $1,200 | $8,000 |
| Database | $180 | $800 | $5,000 |
| Cache | $40 | $200 | $900 |
| Vector DB | $60 | $600 | $4,600 |
| Storage | $40 | $150 | $500 |
| LLM | $50 | $500 | $5,000 |
| **Total** | **$550/mo** | **$3,450/mo** | **$24,000/mo** |

---

## Annual Cost Projections

### 3-Year Total Cost of Ownership (TCO)

**Medium Deployment:**

| Year | Infrastructure | LLM API | Licenses | Support | Training | Total |
|------|----------------|---------|----------|---------|----------|-------|
| Year 1 | $50,000 | $18,000 | $0 | $12,000 | $8,000 | $88,000 |
| Year 2 | $55,000 | $15,000 | $0 | $15,000 | $3,000 | $88,000 |
| Year 3 | $60,000 | $12,000 | $0 | $18,000 | $2,000 | $92,000 |
| **Total** | **$165,000** | **$45,000** | **$0** | **$45,000** | **$13,000** | **$268,000** |

**Large Deployment:**

| Year | Infrastructure | LLM API | Licenses | Support | Training | Total |
|------|----------------|---------|----------|---------|----------|-------|
| Year 1 | $280,000 | $120,000 | $0 | $50,000 | $25,000 | $475,000 |
| Year 2 | $300,000 | $90,000 | $0 | $60,000 | $10,000 | $460,000 |
| Year 3 | $320,000 | $72,000 | $0 | $70,000 | $8,000 | $470,000 |
| **Total** | **$900,000** | **$282,000** | **$0** | **$180,000** | **$43,000** | **$1,405,000** |

---

## Cost Optimization Strategies

### 1. Reserved Instances / Savings Plans

**Impact: 30-60% savings on compute**

| Commitment | EC2 Savings | RDS Savings | Total Savings (Medium) |
|------------|-------------|-------------|------------------------|
| None | 0% | 0% | $0 |
| 1-Year | 30% | 35% | ~$1,450/month |
| 3-Year | 50% | 55% | ~$2,600/month |

### 2. Spot Instances for Workers

**Impact: 50-70% savings on worker compute**

```
Medium Deployment Worker Cost: $1,152/month
With Spot Instances (70% off): $346/month
Monthly Savings: $806
```

**Considerations:**
- Requires graceful interruption handling
- Not suitable for stateful workloads
- Best for: Celery workers, batch jobs

### 3. Auto-Scaling

**Impact: 20-40% savings during off-hours**

```
Assumptions:
- Peak hours: 8am-6pm (10 hours/day)
- Off-hours: 6pm-8am + weekends (62% of time)
- Scale down 50% during off-hours

Compute savings: 50% × 62% = 31% reduction
Medium Deployment Savings: ~$450/month
```

### 4. S3 Intelligent Tiering + Glacier

**Impact: 30-60% savings on storage**

| Tier | Access Frequency | Cost/GB | % of Data |
|------|------------------|---------|-----------|
| Frequent | Daily | $0.023 | 20% |
| Infrequent | Monthly | $0.0125 | 30% |
| Archive | Rarely | $0.004 | 50% |

```
5 TB Data Example:
Standard: 5,000 × $0.023 = $115/month
Intelligent: (1000×$0.023) + (1500×$0.0125) + (2500×$0.004) = $42/month
Savings: $73/month (63%)
```

### 5. LLM Cost Reduction

**Impact: 80-90% savings on LLM API**

| Strategy | Implementation | Savings |
|----------|----------------|---------|
| Aggressive Caching | Redis cache (24hr TTL) | 80% |
| Model Selection | Use GPT-3.5/Haiku for simple tasks | 30% of remaining |
| Prompt Optimization | Reduce token count | 20% of remaining |
| Batch Processing | Combine requests | 10% of remaining |

```
Before: $10,000/month
After caching (80%): $2,000/month
After model selection (30%): $1,400/month
After optimization (20%): $1,120/month
After batching (10%): $1,008/month

Total savings: ~90% ($8,992/month)
```

---

## ROI Analysis

### Cost Savings from Automation (vs. Manual Processes)

**CBAM Reporting:**
- Manual time: 40 hours/quarter per 1000 shipments
- Labor cost: $50/hour
- Manual cost: $2,000/quarter = $8,000/year
- Platform cost: $700/year (Small deployment, CBAM only)
- **Net savings: $7,300/year (91% reduction)**

**CSRD Reporting:**
- Manual time: 160 hours/year for report preparation
- Consultant cost: $200/hour
- Manual cost: $32,000/year
- Platform cost: $2,350/year (Medium deployment, CSRD only)
- **Net savings: $29,650/year (93% reduction)**

**VCCI Scope 3 Calculation:**
- Manual time: 320 hours/year for 10,000 transactions
- Analyst cost: $75/hour
- Manual cost: $24,000/year
- Platform cost: $3,450/year (Medium deployment, VCCI only)
- **Net savings: $20,550/year (86% reduction)**

**Total ROI (Medium Deployment, All Apps):**
- Manual cost: $64,000/year
- Platform cost: $6,349/year
- **Net savings: $57,651/year (90% reduction)**
- **Payback period: < 1 month**

---

## Cost Comparison: Cloud Providers

| Deployment Size | AWS | Azure | GCP |
|----------------|-----|-------|-----|
| **Small** | $727 | $815 | $727 |
| **Medium** | $6,349 | $6,800 | $6,200 |
| **Large** | $48,782 | $51,000 | $47,500 |

**Recommendation:**
- **Small/Medium:** GCP (slightly cheaper, simpler pricing)
- **Large:** GCP or AWS (depends on existing infrastructure)
- **Enterprise:** Negotiate custom pricing with all providers

---

## Budget Planning Template

### Year 1 Budget (Medium Deployment)

| Category | Q1 | Q2 | Q3 | Q4 | Total |
|----------|----|----|----|----|-------|
| Infrastructure | $12,000 | $13,000 | $14,000 | $15,000 | $54,000 |
| LLM API | $3,000 | $4,000 | $5,000 | $6,000 | $18,000 |
| Licenses | $0 | $0 | $0 | $0 | $0 |
| Support | $3,000 | $3,000 | $3,000 | $3,000 | $12,000 |
| Training | $5,000 | $2,000 | $500 | $500 | $8,000 |
| Contingency (10%) | $2,300 | $2,200 | $2,250 | $2,450 | $9,200 |
| **Total** | **$25,300** | **$24,200** | **$24,750** | **$26,950** | **$101,200** |

---

## Summary

### Key Takeaways

1. **Small Deployment: $700-800/month**
   - Suitable for 1-10 organizations
   - Minimal LLM usage
   - Single-region deployment

2. **Medium Deployment: $4,300-6,300/month**
   - Suitable for 10-100 organizations
   - Moderate LLM usage (with caching)
   - Multi-AZ deployment

3. **Large Deployment: $22,000-49,000/month**
   - Suitable for 100-1000 organizations
   - Heavy LLM usage (optimized)
   - Multi-region deployment

4. **Cost Optimization Can Reduce Total Cost by 40-60%**
   - Reserved Instances: 30-50% savings
   - Spot Instances: 50-70% savings on workers
   - LLM Caching: 80% savings on API calls
   - Auto-scaling: 20-40% savings

5. **ROI is Achieved in < 3 Months**
   - Platform reduces manual effort by 85-95%
   - Payback period: 1-3 months
   - 3-year TCO: $268K (medium), $1.4M (large)

---

**Document Owner:** Finance & Platform Teams
**Last Updated:** 2025-11-08
**Next Review:** Quarterly (after each billing cycle)
