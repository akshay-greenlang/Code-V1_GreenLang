# GreenLang Artifact Storage Operations Runbook

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | Platform Operations |
| Classification | Internal |

---

## Table of Contents

1. [Daily Operations Checklist](#daily-operations-checklist)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Incident Response Procedures](#incident-response-procedures)
4. [Disaster Recovery Steps](#disaster-recovery-steps)
5. [Backup and Restore Procedures](#backup-and-restore-procedures)
6. [Cost Management](#cost-management)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## Daily Operations Checklist

### Morning Health Check (09:00 UTC)

| Task | Command/Action | Expected Result |
|------|----------------|-----------------|
| Check bucket accessibility | `aws s3 ls s3://greenlang-prod-*` | All buckets listed |
| Verify replication status | CloudWatch > CRR metrics | Replication lag < 15 min |
| Review failed requests | CloudWatch > 4xx/5xx errors | < 0.1% error rate |
| Check storage growth | S3 > Bucket metrics | Within budget projections |
| Verify lifecycle transitions | S3 > Metrics > Lifecycle | No failed transitions |
| Check multipart upload cleanup | AWS CLI (see below) | No stale uploads > 7 days |

**Check for stale multipart uploads:**
```bash
# List incomplete multipart uploads older than 7 days
aws s3api list-multipart-uploads \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --query 'Uploads[?Initiated<=`2026-01-27`]'
```

### Weekly Tasks (Monday)

| Task | Description | Owner |
|------|-------------|-------|
| Storage cost review | Review Cost Explorer report | Platform Engineering |
| Access audit review | Review CloudTrail for unusual access | Security Team |
| Replication health check | Verify cross-region replication integrity | Platform Engineering |
| Lifecycle policy review | Verify policies are working correctly | Platform Engineering |
| Capacity planning | Review growth trends | Platform Engineering |

### Monthly Tasks (First Monday)

| Task | Description | Owner |
|------|-------------|-------|
| Full access audit | Review all IAM policies and bucket policies | Security Team |
| Disaster recovery test | Run DR drill (non-production) | Platform Engineering |
| Cost optimization review | Identify cost savings opportunities | Finance + Platform |
| Compliance review | Verify retention policies | Compliance Team |
| Documentation review | Update runbooks if needed | Platform Engineering |

---

## Monitoring and Alerting

### CloudWatch Metrics

#### Key Metrics to Monitor

| Metric | Namespace | Threshold | Severity |
|--------|-----------|-----------|----------|
| BucketSizeBytes | AWS/S3 | > 10 TB | Warning |
| NumberOfObjects | AWS/S3 | > 100M | Warning |
| 4xxErrors | AWS/S3 | > 100/min | Warning |
| 5xxErrors | AWS/S3 | > 10/min | Critical |
| FirstByteLatency | AWS/S3 | > 200ms | Warning |
| TotalRequestLatency | AWS/S3 | > 500ms | Warning |
| ReplicationLatency | AWS/S3 | > 15 min | Warning |
| OperationsPendingReplication | AWS/S3 | > 1000 | Critical |

#### CloudWatch Dashboard Setup

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "title": "S3 Request Errors",
        "metrics": [
          ["AWS/S3", "4xxErrors", "BucketName", "greenlang-prod-eu-west-1-data-lake-confidential"],
          [".", "5xxErrors", ".", "."]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "eu-west-1"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Replication Latency",
        "metrics": [
          ["AWS/S3", "ReplicationLatency", "BucketName", "greenlang-prod-eu-west-1-data-lake-confidential", "DestinationBucket", "greenlang-prod-eu-central-1-data-lake-confidential-replica"]
        ],
        "period": 300,
        "stat": "Average",
        "region": "eu-west-1"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Storage Size Trend",
        "metrics": [
          ["AWS/S3", "BucketSizeBytes", "BucketName", "greenlang-prod-eu-west-1-data-lake-confidential", "StorageType", "StandardStorage"]
        ],
        "period": 86400,
        "stat": "Average",
        "region": "eu-west-1"
      }
    }
  ]
}
```

### CloudWatch Alarms

#### Critical Alarms

```yaml
# S3 5xx Errors Alarm
alarm_name: greenlang-s3-5xx-errors-critical
metric_name: 5xxErrors
namespace: AWS/S3
dimensions:
  - BucketName: greenlang-prod-eu-west-1-data-lake-confidential
statistic: Sum
period: 300
evaluation_periods: 2
threshold: 10
comparison_operator: GreaterThanThreshold
alarm_actions:
  - arn:aws:sns:eu-west-1:123456789012:platform-critical-alerts

# Replication Pending Alarm
alarm_name: greenlang-s3-replication-pending-critical
metric_name: OperationsPendingReplication
namespace: AWS/S3
dimensions:
  - BucketName: greenlang-prod-eu-west-1-data-lake-confidential
statistic: Sum
period: 300
evaluation_periods: 3
threshold: 1000
comparison_operator: GreaterThanThreshold
alarm_actions:
  - arn:aws:sns:eu-west-1:123456789012:platform-critical-alerts
```

### Alert Response Matrix

| Alert | Severity | Response Time | On-Call Action |
|-------|----------|---------------|----------------|
| S3 5xx errors > 10/min | Critical | 5 minutes | Investigate immediately |
| S3 4xx errors > 100/min | Warning | 30 minutes | Review access patterns |
| Replication lag > 15 min | Warning | 30 minutes | Check replication status |
| Replication pending > 1000 | Critical | 15 minutes | Investigate replication |
| Storage > 10 TB | Warning | 24 hours | Review growth, optimize |
| Lifecycle failures | Warning | 24 hours | Review lifecycle rules |

---

## Incident Response Procedures

### Incident Classification

| Severity | Definition | Examples |
|----------|------------|----------|
| P1 - Critical | Complete service outage | S3 unavailable, data loss |
| P2 - High | Major functionality impaired | Replication failed, slow uploads |
| P3 - Medium | Minor functionality impaired | Increased latency, 4xx errors |
| P4 - Low | Minimal impact | Documentation issues, cosmetic |

### P1 - Critical Incident Response

**Trigger Conditions:**
- S3 service unavailable
- Data loss detected
- Complete replication failure
- Security breach suspected

**Response Steps:**

1. **Acknowledge Incident (0-5 min)**
   ```bash
   # Post to incident channel
   slack-cli post #platform-incidents "P1 INCIDENT: S3 storage issue detected. Investigating."
   ```

2. **Assess Impact (5-15 min)**
   ```bash
   # Check S3 service status
   aws s3 ls s3://greenlang-prod-eu-west-1-data-lake-confidential

   # Check for errors in CloudWatch
   aws cloudwatch get-metric-statistics \
     --namespace AWS/S3 \
     --metric-name 5xxErrors \
     --dimensions Name=BucketName,Value=greenlang-prod-eu-west-1-data-lake-confidential \
     --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
     --period 60 \
     --statistics Sum
   ```

3. **Engage Stakeholders (15-30 min)**
   - Page on-call engineer
   - Notify engineering leadership
   - Update status page

4. **Implement Mitigation**
   - If primary region down: Failover to DR region
   - If data corruption: Restore from backup
   - If security breach: Revoke access, enable forensics

5. **Communicate Updates**
   - Post updates every 15 minutes
   - Update status page
   - Prepare customer communications

6. **Post-Incident**
   - Write incident report within 24 hours
   - Schedule postmortem within 72 hours
   - Implement preventive measures

### P2 - High Incident Response

**Trigger Conditions:**
- Replication lag > 1 hour
- Upload failures > 5%
- Access denied errors spike

**Response Steps:**

1. **Acknowledge and Investigate (0-15 min)**
   ```bash
   # Check replication status
   aws s3api get-bucket-replication \
     --bucket greenlang-prod-eu-west-1-data-lake-confidential

   # Check replication metrics
   aws cloudwatch get-metric-statistics \
     --namespace AWS/S3 \
     --metric-name ReplicationLatency \
     --dimensions Name=BucketName,Value=greenlang-prod-eu-west-1-data-lake-confidential \
     --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
     --period 60 \
     --statistics Average
   ```

2. **Identify Root Cause (15-60 min)**
   - Check AWS Health Dashboard
   - Review CloudTrail logs
   - Check IAM permissions

3. **Implement Fix**
   - Address specific issue
   - Monitor for improvement

4. **Document and Close**
   - Update incident ticket
   - Notify stakeholders
   - Schedule follow-up review

---

## Disaster Recovery Steps

### DR Activation Criteria

Activate DR when:
- Primary region S3 unavailable > 30 minutes
- Data corruption detected in primary
- Primary region declared unavailable by AWS

### Failover Procedure

**Pre-Failover Checks:**
```bash
# 1. Verify DR region bucket accessibility
aws s3 ls s3://greenlang-prod-eu-central-1-data-lake-confidential-replica \
  --region eu-central-1

# 2. Check replication status before cutover
aws s3api head-object \
  --bucket greenlang-prod-eu-central-1-data-lake-confidential-replica \
  --key silver/cbam/latest/manifest.json \
  --region eu-central-1

# 3. Verify data integrity (sample check)
aws s3api head-object \
  --bucket greenlang-prod-eu-central-1-data-lake-confidential-replica \
  --key gold/sustainability/emissions_summary.parquet \
  --region eu-central-1
```

**Failover Steps:**

1. **Notify Stakeholders**
   ```bash
   slack-cli post #platform-incidents "INITIATING DR FAILOVER to eu-central-1"
   ```

2. **Update DNS/Configuration**
   ```bash
   # Update Route53 to point to DR region endpoints
   aws route53 change-resource-record-sets \
     --hosted-zone-id Z1234567890ABC \
     --change-batch file://dr-failover-records.json
   ```

3. **Update Application Configuration**
   ```bash
   # Update ConfigMap with DR bucket names
   kubectl apply -f k8s/dr/storage-configmap-dr.yaml

   # Restart affected services
   kubectl rollout restart deployment/storage-service -n greenlang
   ```

4. **Verify Failover**
   ```bash
   # Test upload to DR bucket
   echo "DR test $(date)" > /tmp/dr-test.txt
   aws s3 cp /tmp/dr-test.txt s3://greenlang-prod-eu-central-1-data-lake-confidential-replica/dr-test/

   # Verify application connectivity
   curl -s https://api.greenlang.io/v1/storage/health | jq '.status'
   ```

5. **Monitor DR Operations**
   - Enable DR-specific CloudWatch dashboards
   - Set up alerts for DR region

### Failback Procedure

**Failback Prerequisites:**
- Primary region confirmed stable for 24+ hours
- All data synchronized to primary
- Stakeholder approval received

**Failback Steps:**

1. **Verify Primary Region Health**
   ```bash
   aws s3 ls s3://greenlang-prod-eu-west-1-data-lake-confidential \
     --region eu-west-1
   ```

2. **Sync Data Back to Primary**
   ```bash
   # Sync any new data created during DR
   aws s3 sync \
     s3://greenlang-prod-eu-central-1-data-lake-confidential-replica/ \
     s3://greenlang-prod-eu-west-1-data-lake-confidential/ \
     --region eu-west-1 \
     --source-region eu-central-1
   ```

3. **Update DNS/Configuration**
   ```bash
   # Revert Route53 to primary region
   aws route53 change-resource-record-sets \
     --hosted-zone-id Z1234567890ABC \
     --change-batch file://primary-failback-records.json
   ```

4. **Update Application Configuration**
   ```bash
   kubectl apply -f k8s/storage-configmap.yaml
   kubectl rollout restart deployment/storage-service -n greenlang
   ```

5. **Re-enable Replication**
   - Verify replication rules are active
   - Monitor replication catch-up

---

## Backup and Restore Procedures

### Backup Strategy

| Data Type | Backup Method | Frequency | Retention |
|-----------|---------------|-----------|-----------|
| Data Lake (Bronze) | S3 Versioning + CRR | Continuous | 7 years |
| Data Lake (Silver/Gold) | S3 Versioning + CRR | Continuous | 7 years |
| Reports | S3 Versioning + CRR | Continuous | 10 years |
| ML Models | S3 Versioning | On change | 2 years |
| Configuration | Git + S3 | On change | Forever |

### Restore from Versioning

```bash
# List object versions
aws s3api list-object-versions \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --prefix silver/cbam/emissions/ \
  --max-keys 50

# Restore specific version
aws s3api copy-object \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --copy-source greenlang-prod-eu-west-1-data-lake-confidential/silver/cbam/emissions/data.parquet?versionId=abc123 \
  --key silver/cbam/emissions/data.parquet

# Restore deleted object (remove delete marker)
aws s3api delete-object \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --key silver/cbam/emissions/data.parquet \
  --version-id DELETE_MARKER_VERSION_ID
```

### Restore from Glacier

```bash
# Check current storage class
aws s3api head-object \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --key archive/2024/compliance_report.pdf

# Initiate restore (Expedited: 1-5 min, Standard: 3-5 hrs, Bulk: 5-12 hrs)
aws s3api restore-object \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --key archive/2024/compliance_report.pdf \
  --restore-request '{"Days":7,"GlacierJobParameters":{"Tier":"Standard"}}'

# Check restore status
aws s3api head-object \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --key archive/2024/compliance_report.pdf \
  --query 'Restore'
```

### Point-in-Time Recovery Script

```python
#!/usr/bin/env python3
"""
Point-in-Time Recovery for S3 Objects
Restores objects to a specific timestamp using versioning.
"""

import boto3
from datetime import datetime, timezone
import argparse

def restore_to_point_in_time(bucket: str, prefix: str, target_time: datetime):
    """Restore objects under prefix to their state at target_time."""
    s3_client = boto3.client('s3')

    paginator = s3_client.get_paginator('list_object_versions')

    restored_count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for version in page.get('Versions', []):
            key = version['Key']
            version_id = version['VersionId']
            last_modified = version['LastModified']

            # Find the version that was current at target_time
            if last_modified <= target_time:
                if not version['IsLatest']:
                    # Restore this version
                    print(f"Restoring {key} to version {version_id}")

                    s3_client.copy_object(
                        Bucket=bucket,
                        CopySource=f"{bucket}/{key}?versionId={version_id}",
                        Key=key
                    )
                    restored_count += 1
                break

    print(f"Restored {restored_count} objects to {target_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--timestamp', required=True, help='ISO format: 2026-02-01T12:00:00Z')

    args = parser.parse_args()
    target_time = datetime.fromisoformat(args.timestamp.replace('Z', '+00:00'))

    restore_to_point_in_time(args.bucket, args.prefix, target_time)
```

---

## Cost Management

### Daily Cost Monitoring

```bash
# Get daily S3 costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Simple Storage Service"]}}' \
  --group-by Type=TAG,Key=greenlang:application
```

### Cost Optimization Actions

| Issue | Detection | Action |
|-------|-----------|--------|
| High storage costs | Cost Explorer trend | Review lifecycle policies |
| High request costs | Cost Explorer | Implement caching |
| Incomplete uploads | S3 metrics | Enable abort policy |
| Large objects in wrong class | S3 Inventory | Adjust lifecycle rules |
| Unused data | S3 Inventory | Archive or delete |

### Monthly Cost Review Process

1. **Generate Cost Report**
   ```bash
   aws ce get-cost-and-usage \
     --time-period Start=$(date -d '1 month ago' +%Y-%m-01),End=$(date +%Y-%m-01) \
     --granularity MONTHLY \
     --metrics BlendedCost UnblendedCost \
     --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Simple Storage Service"]}}' \
     --group-by Type=DIMENSION,Key=USAGE_TYPE
   ```

2. **Review Storage Classes**
   - Check Intelligent Tiering effectiveness
   - Verify lifecycle transitions are occurring
   - Identify candidates for Glacier

3. **Review Request Patterns**
   - Identify high-frequency access patterns
   - Consider caching or CDN for frequent reads
   - Optimize batch operations

4. **Update Budget Alerts**
   - Adjust thresholds based on trends
   - Set up anomaly detection

---

## Performance Tuning

### Upload Performance Optimization

| File Size | Optimization | Configuration |
|-----------|--------------|---------------|
| < 5 MB | Single PUT | Default |
| 5-100 MB | Multipart (5 MB parts) | `multipart_threshold=5MB` |
| 100 MB - 1 GB | Multipart (10 MB parts) | `multipart_chunksize=10MB` |
| > 1 GB | Multipart (100 MB parts) + concurrency | `max_concurrency=20` |

**Boto3 Transfer Configuration:**
```python
from boto3.s3.transfer import TransferConfig

config = TransferConfig(
    multipart_threshold=5 * 1024 * 1024,  # 5 MB
    multipart_chunksize=10 * 1024 * 1024,  # 10 MB
    max_concurrency=20,
    use_threads=True
)
```

### Download Performance Optimization

| Scenario | Optimization |
|----------|--------------|
| Large files | Range requests for parallel download |
| Many small files | Batch with S3 Batch Operations |
| Repeated access | CloudFront caching |
| Regional access | S3 Transfer Acceleration |

### Prefix Distribution

For high-throughput workloads, distribute keys across prefixes:

```python
import hashlib

def generate_distributed_key(base_key: str) -> str:
    """Generate key with hash prefix for even distribution."""
    hash_prefix = hashlib.md5(base_key.encode()).hexdigest()[:2]
    return f"{hash_prefix}/{base_key}"
```

---

## Troubleshooting Guide

### Access Denied Errors

**Symptoms:**
- `AccessDenied` errors in application logs
- 403 responses from S3 API

**Diagnostic Steps:**

1. **Check IAM Permissions**
   ```bash
   # Simulate IAM policy for specific action
   aws iam simulate-principal-policy \
     --policy-source-arn arn:aws:iam::123456789012:role/greenlang-cbam-role \
     --action-names s3:GetObject s3:PutObject \
     --resource-arns arn:aws:s3:::greenlang-prod-eu-west-1-data-lake-confidential/*
   ```

2. **Check Bucket Policy**
   ```bash
   aws s3api get-bucket-policy \
     --bucket greenlang-prod-eu-west-1-data-lake-confidential
   ```

3. **Check Object ACL**
   ```bash
   aws s3api get-object-acl \
     --bucket greenlang-prod-eu-west-1-data-lake-confidential \
     --key path/to/object
   ```

4. **Check VPC Endpoint Policy** (if applicable)
   ```bash
   aws ec2 describe-vpc-endpoints \
     --filters Name=service-name,Values=com.amazonaws.eu-west-1.s3 \
     --query 'VpcEndpoints[].PolicyDocument'
   ```

**Common Fixes:**
- Add missing IAM permissions
- Update bucket policy to allow principal
- Check for explicit denies
- Verify KMS key permissions for encrypted objects

### Slow Uploads

**Symptoms:**
- Upload times exceeding SLA
- Timeouts during upload

**Diagnostic Steps:**

1. **Check Network Path**
   ```bash
   # Test latency to S3 endpoint
   ping s3.eu-west-1.amazonaws.com

   # Check for packet loss
   mtr -c 100 s3.eu-west-1.amazonaws.com
   ```

2. **Check S3 Request Latency**
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace AWS/S3 \
     --metric-name TotalRequestLatency \
     --dimensions Name=BucketName,Value=greenlang-prod-eu-west-1-data-lake-confidential \
     --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
     --period 300 \
     --statistics Average
   ```

3. **Check for Throttling**
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace AWS/S3 \
     --metric-name ThrottledRequests \
     --dimensions Name=BucketName,Value=greenlang-prod-eu-west-1-data-lake-confidential \
     --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
     --period 300 \
     --statistics Sum
   ```

**Common Fixes:**
- Enable Transfer Acceleration for distant clients
- Use multipart upload for large files
- Distribute keys across prefixes
- Increase connection pool size

### Replication Lag

**Symptoms:**
- Objects not appearing in replica bucket
- Stale data in DR region

**Diagnostic Steps:**

1. **Check Replication Status**
   ```bash
   # Check replication configuration
   aws s3api get-bucket-replication \
     --bucket greenlang-prod-eu-west-1-data-lake-confidential

   # Check replication metrics
   aws cloudwatch get-metric-statistics \
     --namespace AWS/S3 \
     --metric-name ReplicationLatency \
     --dimensions Name=BucketName,Value=greenlang-prod-eu-west-1-data-lake-confidential \
       Name=DestinationBucket,Value=greenlang-prod-eu-central-1-data-lake-confidential-replica \
     --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
     --period 300 \
     --statistics Average
   ```

2. **Check Individual Object Replication**
   ```bash
   aws s3api head-object \
     --bucket greenlang-prod-eu-west-1-data-lake-confidential \
     --key path/to/object \
     --query 'ReplicationStatus'
   ```

3. **Check Failed Replication**
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace AWS/S3 \
     --metric-name OperationsFailedReplication \
     --dimensions Name=BucketName,Value=greenlang-prod-eu-west-1-data-lake-confidential \
     --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
     --period 300 \
     --statistics Sum
   ```

**Common Fixes:**
- Verify replication role has necessary permissions
- Check destination bucket exists and is accessible
- Verify KMS key permissions for encrypted objects
- Contact AWS Support for persistent issues

### Lifecycle Issues

**Symptoms:**
- Objects not transitioning to expected storage class
- Objects not expiring as expected

**Diagnostic Steps:**

1. **Check Lifecycle Configuration**
   ```bash
   aws s3api get-bucket-lifecycle-configuration \
     --bucket greenlang-prod-eu-west-1-data-lake-confidential
   ```

2. **Check Object Metadata**
   ```bash
   aws s3api head-object \
     --bucket greenlang-prod-eu-west-1-data-lake-confidential \
     --key path/to/object
   ```

3. **Check Object Tags** (if lifecycle uses tags)
   ```bash
   aws s3api get-object-tagging \
     --bucket greenlang-prod-eu-west-1-data-lake-confidential \
     --key path/to/object
   ```

**Common Fixes:**
- Ensure object size > 128KB for IA transition
- Check lifecycle rule filters match object
- Verify minimum storage duration requirements
- Check for lifecycle rule conflicts

---

## Contact Information

### Escalation Path

| Level | Team | Contact | Response Time |
|-------|------|---------|---------------|
| L1 | Platform Operations | platform-ops@greenlang.io | 15 min |
| L2 | Platform Engineering | platform-eng@greenlang.io | 30 min |
| L3 | AWS Support | AWS Console | Per support plan |
| Management | Engineering Leadership | eng-leadership@greenlang.io | 1 hour |

### On-Call Schedule

On-call schedule maintained in PagerDuty:
- Primary: Platform Operations rotation
- Secondary: Platform Engineering rotation
- Escalation: Engineering Leadership

---

## Related Documents

- [Architecture Guide](architecture-guide.md)
- [Disaster Recovery Plan](disaster-recovery.md)
- [Cost Optimization Guide](cost-optimization.md)
- [Developer Guide](developer-guide.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Platform Operations | Initial release |
