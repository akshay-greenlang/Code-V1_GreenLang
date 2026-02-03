# GreenLang Artifact Storage Cost Optimization Guide

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | Platform Engineering / Finance |
| Classification | Internal |

---

## Table of Contents

1. [Overview](#overview)
2. [Storage Class Selection Guide](#storage-class-selection-guide)
3. [Lifecycle Policy Optimization](#lifecycle-policy-optimization)
4. [Intelligent Tiering Benefits](#intelligent-tiering-benefits)
5. [Request Cost Reduction](#request-cost-reduction)
6. [Data Transfer Optimization](#data-transfer-optimization)
7. [Reserved Capacity Planning](#reserved-capacity-planning)
8. [Cost Allocation Tagging](#cost-allocation-tagging)
9. [Monthly Cost Review Checklist](#monthly-cost-review-checklist)

---

## Overview

### Cost Optimization Goals

| Goal | Target | Measurement |
|------|--------|-------------|
| Storage cost per GB | < $0.015/GB/month | Average across all tiers |
| Request cost reduction | 20% YoY | Compared to baseline |
| Data transfer costs | < 5% of total | Of total S3 costs |
| Wasted storage elimination | < 1% | Unused/orphaned data |

### Current Cost Breakdown

```
                              S3 Cost Distribution
+--------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Storage Costs (65%)                   Request Costs (25%)           Transfer Costs (10%)          |
|  +-------------------------+           +--------------------+        +--------------------+        |
|  |                         |           |                    |        |                    |        |
|  |  Standard:     40%      |           |  PUT/POST:   15%   |        |  Inter-region: 6%  |        |
|  |  Standard-IA:  15%      |           |  GET:        8%    |        |  Internet:     3%  |        |
|  |  Glacier IR:   8%       |           |  LIST:       2%    |        |  CloudFront:   1%  |        |
|  |  Glacier:      2%       |           |                    |        |                    |        |
|  |                         |           |                    |        |                    |        |
|  +-------------------------+           +--------------------+        +--------------------+        |
|                                                                                                    |
+--------------------------------------------------------------------------------------------------+
```

---

## Storage Class Selection Guide

### Storage Class Comparison

| Storage Class | Cost/GB/Month | Min Duration | Retrieval Time | Best For |
|---------------|---------------|--------------|----------------|----------|
| S3 Standard | $0.023 | None | Immediate | Frequent access |
| S3 Standard-IA | $0.0125 | 30 days | Immediate | Monthly access |
| S3 One Zone-IA | $0.01 | 30 days | Immediate | Reproducible data |
| Glacier Instant Retrieval | $0.004 | 90 days | Immediate | Quarterly access |
| Glacier Flexible | $0.0036 | 90 days | 1-12 hours | Annual access |
| Glacier Deep Archive | $0.00099 | 180 days | 12-48 hours | Compliance archive |

### Decision Matrix

```
                              Storage Class Selection
+--------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Access Frequency            |  Data Size  |  Retrieval Need  |  Recommended Class               |
|  ----------------------------|-------------|------------------|--------------------------------  |
|  Daily or more               |  Any        |  Immediate       |  S3 Standard                     |
|  Weekly                      |  > 128 KB   |  Immediate       |  S3 Standard-IA                  |
|  Monthly                     |  > 128 KB   |  Immediate       |  S3 Standard-IA                  |
|  Quarterly                   |  > 128 KB   |  Immediate       |  Glacier Instant Retrieval       |
|  Annually                    |  > 128 KB   |  Hours OK        |  Glacier Flexible Retrieval      |
|  Rarely (compliance)         |  Any        |  12+ hours OK    |  Glacier Deep Archive            |
|  Unknown/Variable            |  > 128 KB   |  Immediate       |  Intelligent Tiering             |
|                                                                                                    |
+--------------------------------------------------------------------------------------------------+
```

### GreenLang Data Type Recommendations

| Data Type | Recommended Class | Rationale |
|-----------|-------------------|-----------|
| Landing Zone | Standard | Temporary, frequent access |
| Bronze Zone (recent) | Standard | Active processing |
| Bronze Zone (> 30 days) | Standard-IA | Occasional reprocessing |
| Silver Zone (recent) | Standard | Active analytics |
| Silver Zone (> 90 days) | Glacier IR | Historical queries |
| Gold Zone | Standard | Dashboard queries |
| Generated Reports (recent) | Standard | Active downloads |
| Generated Reports (> 90 days) | Glacier IR | Archived reports |
| ML Models (active) | Standard | Inference serving |
| ML Models (archived) | Standard-IA | Model versioning |
| Audit Logs | Glacier Deep Archive | Compliance only |

---

## Lifecycle Policy Optimization

### Recommended Lifecycle Policies

#### Data Lake Bucket

```json
{
  "Rules": [
    {
      "ID": "landing-cleanup",
      "Status": "Enabled",
      "Filter": {"Prefix": "landing/"},
      "Expiration": {"Days": 1},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 1}
    },
    {
      "ID": "bronze-tiering",
      "Status": "Enabled",
      "Filter": {"Prefix": "bronze/"},
      "Transitions": [
        {"Days": 30, "StorageClass": "STANDARD_IA"},
        {"Days": 90, "StorageClass": "GLACIER_IR"},
        {"Days": 365, "StorageClass": "DEEP_ARCHIVE"}
      ],
      "NoncurrentVersionTransitions": [
        {"NoncurrentDays": 30, "StorageClass": "GLACIER_IR"}
      ],
      "NoncurrentVersionExpiration": {"NoncurrentDays": 365}
    },
    {
      "ID": "silver-tiering",
      "Status": "Enabled",
      "Filter": {"Prefix": "silver/"},
      "Transitions": [
        {"Days": 90, "StorageClass": "GLACIER_IR"},
        {"Days": 365, "StorageClass": "DEEP_ARCHIVE"}
      ]
    },
    {
      "ID": "gold-retain-standard",
      "Status": "Enabled",
      "Filter": {"Prefix": "gold/"},
      "Transitions": [
        {"Days": 365, "StorageClass": "STANDARD_IA"}
      ]
    },
    {
      "ID": "abort-incomplete-uploads",
      "Status": "Enabled",
      "Filter": {},
      "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
    },
    {
      "ID": "expire-delete-markers",
      "Status": "Enabled",
      "Filter": {},
      "Expiration": {"ExpiredObjectDeleteMarker": true}
    }
  ]
}
```

#### Reports Bucket

```json
{
  "Rules": [
    {
      "ID": "reports-tiering",
      "Status": "Enabled",
      "Filter": {"Prefix": "reports/"},
      "Transitions": [
        {"Days": 90, "StorageClass": "STANDARD_IA"},
        {"Days": 365, "StorageClass": "GLACIER_IR"},
        {"Days": 2555, "StorageClass": "DEEP_ARCHIVE"}
      ]
    },
    {
      "ID": "draft-reports-cleanup",
      "Status": "Enabled",
      "Filter": {"Prefix": "drafts/"},
      "Expiration": {"Days": 30}
    }
  ]
}
```

### Cost Savings Calculation

| Transition | Before | After | Savings |
|------------|--------|-------|---------|
| Standard to Standard-IA (30 days) | $0.023/GB | $0.0125/GB | 46% |
| Standard-IA to Glacier IR (90 days) | $0.0125/GB | $0.004/GB | 68% |
| Glacier IR to Deep Archive (365 days) | $0.004/GB | $0.00099/GB | 75% |

**Example: 10 TB Data Lake over 3 years**

| Scenario | Year 1 | Year 2 | Year 3 | Total |
|----------|--------|--------|--------|-------|
| All Standard | $2,760 | $2,760 | $2,760 | $8,280 |
| With Lifecycle | $1,800 | $600 | $200 | $2,600 |
| **Savings** | | | | **$5,680 (69%)** |

---

## Intelligent Tiering Benefits

### When to Use Intelligent Tiering

**Ideal for:**
- Data with unpredictable access patterns
- Mixed access frequency within prefix
- Large datasets where manual tiering is impractical

**Avoid for:**
- Small objects (< 128 KB) - monitoring fee not cost-effective
- Predictable access patterns - manual lifecycle is cheaper
- Objects accessed less than once per month - Standard-IA is cheaper

### Intelligent Tiering Configuration

```json
{
  "IntelligentTieringConfiguration": {
    "Id": "greenlang-data-lake-tiering",
    "Status": "Enabled",
    "Filter": {
      "Prefix": "silver/",
      "Tag": {"Key": "tiering", "Value": "intelligent"}
    },
    "Tierings": [
      {
        "Days": 90,
        "AccessTier": "ARCHIVE_ACCESS"
      },
      {
        "Days": 180,
        "AccessTier": "DEEP_ARCHIVE_ACCESS"
      }
    ]
  }
}
```

### Cost Analysis

| Component | Cost |
|-----------|------|
| Frequent Access Tier | $0.023/GB |
| Infrequent Access Tier | $0.0125/GB |
| Archive Access Tier | $0.004/GB |
| Deep Archive Access Tier | $0.00099/GB |
| Monitoring Fee | $0.0025/1000 objects |

**Break-even analysis:**
- Object > 128 KB: Intelligent Tiering worthwhile
- Object < 128 KB: Monitoring fee exceeds savings

---

## Request Cost Reduction

### Request Pricing

| Operation | Cost per 1,000 requests |
|-----------|-------------------------|
| PUT, COPY, POST, LIST | $0.005 |
| GET, SELECT | $0.0004 |
| Lifecycle Transition | $0.01 |
| Data Retrieval (Glacier) | $0.03 |

### Optimization Strategies

#### 1. Batch Small Files

Instead of uploading many small files:

```python
# Inefficient: Many small files
for item in items:
    s3.put_object(Bucket=bucket, Key=f"data/{item['id']}.json", Body=json.dumps(item))

# Efficient: Batch into larger files
batch = [json.dumps(item) for item in items]
s3.put_object(Bucket=bucket, Key="data/batch_001.jsonl", Body="\n".join(batch))
```

**Savings:** 90%+ reduction in PUT requests

#### 2. Use S3 Select for Filtering

```python
# Inefficient: Download and filter client-side
response = s3.get_object(Bucket=bucket, Key=key)
data = json.loads(response['Body'].read())
filtered = [r for r in data if r['year'] == 2026]

# Efficient: Filter server-side with S3 Select
response = s3.select_object_content(
    Bucket=bucket,
    Key=key,
    Expression="SELECT * FROM s3object WHERE year = 2026",
    ExpressionType='SQL',
    InputSerialization={'JSON': {'Type': 'LINES'}},
    OutputSerialization={'JSON': {}}
)
```

**Savings:** 60-90% reduction in data transfer

#### 3. Optimize LIST Operations

```python
# Inefficient: List all objects repeatedly
def check_exists(bucket, key):
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=key)
    return 'Contents' in objects

# Efficient: Use HEAD request
def check_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
```

**Savings:** 10x cheaper per check

#### 4. Cache Frequently Accessed Data

```python
import redis
from functools import lru_cache

redis_client = redis.Redis()

def get_emission_factor(factor_id: str) -> dict:
    # Check Redis cache first
    cached = redis_client.get(f"ef:{factor_id}")
    if cached:
        return json.loads(cached)

    # Fetch from S3
    response = s3.get_object(
        Bucket=CACHE_BUCKET,
        Key=f"emission-factors/{factor_id}.json"
    )
    data = json.loads(response['Body'].read())

    # Cache for 24 hours
    redis_client.setex(f"ef:{factor_id}", 86400, json.dumps(data))

    return data
```

**Savings:** 99%+ reduction in repeated GET requests

---

## Data Transfer Optimization

### Data Transfer Costs

| Transfer Type | Cost |
|---------------|------|
| Same region (within AWS) | Free |
| Cross-region | $0.02/GB |
| Internet outbound | $0.09/GB (first 10 TB) |
| CloudFront to internet | $0.085/GB |

### Optimization Strategies

#### 1. Use VPC Endpoints

```hcl
# Gateway endpoint for S3 (free)
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.eu-west-1.s3"
}
```

**Savings:** Eliminates NAT Gateway costs for S3 traffic

#### 2. Enable S3 Transfer Acceleration Selectively

```python
# Only use for distant uploads
def upload_with_acceleration(file_path, bucket, key, client_region):
    if client_region in ['ap-southeast-1', 'us-west-2']:
        # Distant region - use acceleration
        s3_accel = boto3.client('s3', config=Config(s3={'use_accelerate_endpoint': True}))
        s3_accel.upload_file(file_path, bucket, key)
    else:
        # Close region - standard upload
        s3.upload_file(file_path, bucket, key)
```

**Cost:** $0.04/GB (but faster for distant regions)

#### 3. Use CloudFront for Downloads

```python
# Generate CloudFront signed URL instead of S3 presigned
from botocore.signers import CloudFrontSigner

def generate_cloudfront_url(key: str, expiration: int = 3600) -> str:
    cloudfront_signer = CloudFrontSigner(KEY_ID, rsa_signer)
    url = cloudfront_signer.generate_presigned_url(
        f"https://cdn.greenlang.io/{key}",
        date_less_than=datetime.utcnow() + timedelta(seconds=expiration)
    )
    return url
```

**Savings:** 5-10% on data transfer costs

#### 4. Compress Before Transfer

```python
import gzip

# Compress data before upload
def upload_compressed(data: bytes, bucket: str, key: str):
    compressed = gzip.compress(data)
    s3.put_object(
        Bucket=bucket,
        Key=f"{key}.gz",
        Body=compressed,
        ContentEncoding='gzip'
    )
```

**Savings:** 60-80% reduction in transfer size

---

## Reserved Capacity Planning

### S3 Storage Lens Analysis

Use S3 Storage Lens to analyze storage patterns:

```bash
# Create Storage Lens configuration
aws s3control put-storage-lens-configuration \
  --account-id 123456789012 \
  --config-id greenlang-storage-analysis \
  --storage-lens-configuration '{
    "Id": "greenlang-storage-analysis",
    "AccountLevel": {
      "BucketLevel": {
        "ActivityMetrics": {"IsEnabled": true},
        "PrefixLevel": {
          "StorageMetrics": {
            "IsEnabled": true,
            "SelectionCriteria": {
              "MaxDepth": 3,
              "MinStorageBytesPercentage": 1.0
            }
          }
        }
      }
    },
    "IsEnabled": true
  }'
```

### Capacity Planning Template

| Metric | Current | 6 Month Forecast | 12 Month Forecast |
|--------|---------|------------------|-------------------|
| Total Storage | 5 TB | 8 TB | 15 TB |
| Monthly Growth | 15% | 12% | 10% |
| Standard Class | 3 TB | 4 TB | 6 TB |
| Infrequent Access | 1.5 TB | 3 TB | 6 TB |
| Glacier | 0.5 TB | 1 TB | 3 TB |
| Monthly Cost | $150 | $200 | $250 |

---

## Cost Allocation Tagging

### Required Tags

All S3 objects and buckets must have:

| Tag Key | Description | Example Values |
|---------|-------------|----------------|
| greenlang:application | Application owner | cbam, csrd, sf6, platform |
| greenlang:environment | Environment | dev, staging, prod |
| greenlang:cost-center | Finance cost center | eng-100, sales-200 |
| greenlang:data-classification | Data sensitivity | public, internal, confidential |
| greenlang:tenant | Tenant ID (if applicable) | tenant-123 |

### Tagging Implementation

```python
def upload_with_tags(file_path: str, bucket: str, key: str, tenant_id: str, app: str):
    tags = urllib.parse.urlencode({
        'greenlang:application': app,
        'greenlang:environment': 'prod',
        'greenlang:cost-center': 'eng-100',
        'greenlang:data-classification': 'confidential',
        'greenlang:tenant': tenant_id
    })

    s3.upload_file(
        file_path,
        bucket,
        key,
        ExtraArgs={'Tagging': tags}
    )
```

### Cost Allocation Report

```bash
# Enable cost allocation tags
aws ce update-cost-allocation-tags-status \
  --cost-allocation-tags-status \
    TagKey=greenlang:application,Status=Active \
    TagKey=greenlang:environment,Status=Active \
    TagKey=greenlang:cost-center,Status=Active

# Get cost by application
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=TAG,Key=greenlang:application \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Simple Storage Service"]}}'
```

---

## Monthly Cost Review Checklist

### Week 1: Data Collection

- [ ] Generate Cost Explorer report for S3
- [ ] Export S3 Storage Lens metrics
- [ ] Collect S3 Inventory report
- [ ] Review CloudWatch metrics for request patterns

### Week 2: Analysis

- [ ] Compare costs to previous month
- [ ] Identify top 10 cost drivers
- [ ] Analyze storage class distribution
- [ ] Review lifecycle policy effectiveness
- [ ] Check for unused/orphaned data

### Week 3: Optimization

- [ ] Adjust lifecycle policies if needed
- [ ] Identify candidates for Intelligent Tiering
- [ ] Review and optimize request patterns
- [ ] Clean up incomplete multipart uploads
- [ ] Delete expired/unnecessary data

### Week 4: Reporting

- [ ] Prepare cost summary for leadership
- [ ] Update capacity forecasts
- [ ] Document optimization actions taken
- [ ] Plan next month's optimization targets

### Monthly Review Template

```markdown
## S3 Cost Review - [Month Year]

### Summary
- Total S3 Cost: $X,XXX
- Change from Previous Month: +/-X%
- Cost per GB: $X.XX

### Storage Breakdown
| Class | Size (TB) | Cost | % of Total |
|-------|-----------|------|------------|
| Standard | X.X | $XXX | XX% |
| Standard-IA | X.X | $XXX | XX% |
| Glacier IR | X.X | $XXX | XX% |
| Deep Archive | X.X | $XXX | XX% |

### Request Costs
- PUT/POST requests: $XXX
- GET requests: $XXX
- LIST requests: $XXX

### Data Transfer
- Cross-region: $XXX
- Internet: $XXX

### Optimization Actions
1. [Action taken]
2. [Action taken]

### Next Month Targets
1. [Target 1]
2. [Target 2]
```

---

## Related Documents

- [Architecture Guide](architecture-guide.md)
- [Operations Runbook](operations-runbook.md)
- [Naming Conventions](naming-conventions.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Platform Engineering | Initial release |
