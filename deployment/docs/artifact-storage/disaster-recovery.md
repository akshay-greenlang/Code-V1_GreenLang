# GreenLang Artifact Storage Disaster Recovery Plan

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | Platform Engineering |
| Classification | Confidential |
| Review Cycle | Quarterly |

---

## Table of Contents

1. [Overview](#overview)
2. [DR Architecture](#dr-architecture)
3. [RTO/RPO Targets](#rtorpo-targets)
4. [Cross-Region Replication Configuration](#cross-region-replication-configuration)
5. [Failover Procedures](#failover-procedures)
6. [Failback Procedures](#failback-procedures)
7. [Testing Schedule](#testing-schedule)
8. [Contact Escalation](#contact-escalation)

---

## Overview

### Purpose

This document defines the disaster recovery (DR) strategy and procedures for GreenLang's artifact storage infrastructure. It ensures business continuity in the event of regional AWS outages, data corruption, or other catastrophic failures.

### Scope

This DR plan covers:
- Amazon S3 buckets in the primary region (eu-west-1)
- Cross-region replication to DR region (eu-central-1)
- Data recovery procedures
- Failover and failback operations

### DR Objectives

| Objective | Target |
|-----------|--------|
| Recovery Time Objective (RTO) | 1 hour for critical data |
| Recovery Point Objective (RPO) | 15 minutes for critical data |
| Failover Decision Time | 30 minutes |
| Annual DR Test Frequency | 4 times (quarterly) |

---

## DR Architecture

### High-Level Architecture

```
                              GreenLang DR Architecture
+--------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Primary Region (eu-west-1)                        DR Region (eu-central-1)                        |
|  +-----------------------------------+            +-----------------------------------+            |
|  |                                   |            |                                   |            |
|  |  greenlang-prod-eu-west-1-        |   CRR     |  greenlang-prod-eu-central-1-     |            |
|  |  data-lake-confidential           |---------->|  data-lake-confidential-replica    |            |
|  |                                   |            |                                   |            |
|  |  - Landing Zone                   |            |  - Landing Zone (replicated)      |            |
|  |  - Bronze Zone                    |            |  - Bronze Zone (replicated)       |            |
|  |  - Silver Zone                    |            |  - Silver Zone (replicated)       |            |
|  |  - Gold Zone                      |            |  - Gold Zone (replicated)         |            |
|  |                                   |            |                                   |            |
|  +-----------------------------------+            +-----------------------------------+            |
|                                                                                                    |
|  +-----------------------------------+            +-----------------------------------+            |
|  |                                   |   CRR     |                                   |            |
|  |  greenlang-prod-eu-west-1-        |---------->|  greenlang-prod-eu-central-1-     |            |
|  |  reports-confidential             |            |  reports-confidential-replica      |            |
|  |                                   |            |                                   |            |
|  +-----------------------------------+            +-----------------------------------+            |
|                                                                                                    |
|  +-----------------------------------+            +-----------------------------------+            |
|  |                                   |   CRR     |                                   |            |
|  |  greenlang-prod-eu-west-1-        |---------->|  greenlang-prod-eu-central-1-     |            |
|  |  models-internal                  |            |  models-internal-replica           |            |
|  |                                   |            |                                   |            |
|  +-----------------------------------+            +-----------------------------------+            |
|                                                                                                    |
|  Supporting Services (Primary)                     Supporting Services (DR)                        |
|  +-----------------------------------+            +-----------------------------------+            |
|  |  - Route53 (Global)               |            |  - Storage Service (Standby)      |            |
|  |  - CloudFront (Global)            |            |  - API Gateway (Standby)          |            |
|  |  - KMS Key (Primary)              |            |  - KMS Key (DR Region)            |            |
|  +-----------------------------------+            +-----------------------------------+            |
|                                                                                                    |
+--------------------------------------------------------------------------------------------------+
```

### Component Redundancy

| Component | Primary Region | DR Region | Sync Method |
|-----------|----------------|-----------|-------------|
| Data Lake Bucket | eu-west-1 | eu-central-1 | Cross-Region Replication |
| Reports Bucket | eu-west-1 | eu-central-1 | Cross-Region Replication |
| Models Bucket | eu-west-1 | eu-central-1 | Cross-Region Replication |
| Audit Logs Bucket | eu-west-1 | N/A | No replication (compliance) |
| Temp Bucket | eu-west-1 | N/A | No replication (ephemeral) |
| KMS Keys | eu-west-1 | eu-central-1 | Multi-region keys |
| IAM Roles | Global | Global | Inherent |

### Data Classification for DR

| Data Type | Criticality | Replication | RTO | RPO |
|-----------|-------------|-------------|-----|-----|
| Calculation Results | Critical | Yes | 1 hour | 15 min |
| Generated Reports | Critical | Yes | 1 hour | 15 min |
| ML Models | High | Yes | 4 hours | 1 hour |
| Audit Logs | High | No* | 24 hours | N/A |
| Temporary Uploads | Low | No | N/A | N/A |
| Cache Data | Low | No | N/A | N/A |

*Audit logs are stored with versioning and Object Lock for compliance but do not require DR replication.

---

## RTO/RPO Targets

### Recovery Time Objective (RTO)

```
                              RTO Timeline
+--------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  T+0        T+15min      T+30min      T+45min      T+60min      T+90min      T+120min            |
|   |            |            |            |            |            |            |                 |
|   v            v            v            v            v            v            v                 |
|  +------------+------------+------------+------------+------------+------------+                  |
|  | Detection  | Assessment | Decision   | Failover   | Validation | Normal Ops |                 |
|  | & Alert    | & Triage   | Point      | Execution  | & Testing  | Restored   |                 |
|  +------------+------------+------------+------------+------------+------------+                  |
|                                                                                                    |
|  Critical Data Target: RTO = 60 minutes                                                           |
|  Non-Critical Data Target: RTO = 4 hours                                                          |
|                                                                                                    |
+--------------------------------------------------------------------------------------------------+
```

### RTO by Data Type

| Data Type | RTO Target | Rationale |
|-----------|------------|-----------|
| Active Calculation Results | 30 min | Required for ongoing operations |
| Generated Reports | 60 min | Customer-facing deliverables |
| Silver/Gold Zone Data | 60 min | Analytics and dashboards |
| Bronze Zone Data | 2 hours | Can be regenerated from sources |
| ML Models | 4 hours | Can use cached versions temporarily |
| Historical Reports | 24 hours | Not time-sensitive |

### Recovery Point Objective (RPO)

| Data Type | RPO Target | Replication Lag SLA |
|-----------|------------|---------------------|
| Active Calculation Results | 15 min | 99% within 15 min |
| Generated Reports | 15 min | 99% within 15 min |
| Silver/Gold Zone Data | 15 min | 99% within 15 min |
| Bronze Zone Data | 1 hour | 99% within 1 hour |
| ML Models | 1 hour | 99% within 1 hour |

### Monitoring RPO Compliance

```bash
# CloudWatch alarm for replication lag > 15 minutes
aws cloudwatch put-metric-alarm \
  --alarm-name "S3-Replication-Lag-Critical" \
  --alarm-description "S3 replication lag exceeds 15 minutes" \
  --metric-name ReplicationLatency \
  --namespace AWS/S3 \
  --dimensions Name=BucketName,Value=greenlang-prod-eu-west-1-data-lake-confidential \
    Name=DestinationBucket,Value=greenlang-prod-eu-central-1-data-lake-confidential-replica \
  --statistic Average \
  --period 300 \
  --evaluation-periods 3 \
  --threshold 900 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:eu-west-1:123456789012:platform-critical-alerts
```

---

## Cross-Region Replication Configuration

### Replication Rules

```json
{
  "Role": "arn:aws:iam::123456789012:role/greenlang-s3-replication-role",
  "Rules": [
    {
      "ID": "replicate-critical-data",
      "Status": "Enabled",
      "Priority": 1,
      "Filter": {
        "And": {
          "Prefix": "",
          "Tags": [
            {"Key": "replication", "Value": "enabled"}
          ]
        }
      },
      "Destination": {
        "Bucket": "arn:aws:s3:::greenlang-prod-eu-central-1-data-lake-confidential-replica",
        "Account": "123456789012",
        "StorageClass": "STANDARD",
        "ReplicationTime": {
          "Status": "Enabled",
          "Time": {"Minutes": 15}
        },
        "Metrics": {
          "Status": "Enabled",
          "EventThreshold": {"Minutes": 15}
        },
        "EncryptionConfiguration": {
          "ReplicaKmsKeyID": "arn:aws:kms:eu-central-1:123456789012:key/dr-key-id"
        }
      },
      "DeleteMarkerReplication": {"Status": "Enabled"},
      "SourceSelectionCriteria": {
        "SseKmsEncryptedObjects": {"Status": "Enabled"},
        "ReplicaModifications": {"Status": "Enabled"}
      }
    },
    {
      "ID": "replicate-bronze-zone",
      "Status": "Enabled",
      "Priority": 2,
      "Filter": {"Prefix": "bronze/"},
      "Destination": {
        "Bucket": "arn:aws:s3:::greenlang-prod-eu-central-1-data-lake-confidential-replica",
        "StorageClass": "STANDARD_IA"
      },
      "DeleteMarkerReplication": {"Status": "Enabled"}
    }
  ]
}
```

### Replication IAM Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetReplicationConfiguration",
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::greenlang-prod-eu-west-1-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObjectVersionForReplication",
        "s3:GetObjectVersionAcl",
        "s3:GetObjectVersionTagging"
      ],
      "Resource": "arn:aws:s3:::greenlang-prod-eu-west-1-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ReplicateObject",
        "s3:ReplicateDelete",
        "s3:ReplicateTags"
      ],
      "Resource": "arn:aws:s3:::greenlang-prod-eu-central-1-*-replica/*"
    },
    {
      "Effect": "Allow",
      "Action": ["kms:Decrypt"],
      "Resource": "arn:aws:kms:eu-west-1:123456789012:key/primary-key-id",
      "Condition": {
        "StringEquals": {"kms:ViaService": "s3.eu-west-1.amazonaws.com"}
      }
    },
    {
      "Effect": "Allow",
      "Action": ["kms:Encrypt"],
      "Resource": "arn:aws:kms:eu-central-1:123456789012:key/dr-key-id",
      "Condition": {
        "StringEquals": {"kms:ViaService": "s3.eu-central-1.amazonaws.com"}
      }
    }
  ]
}
```

---

## Failover Procedures

### Failover Decision Criteria

Initiate failover when ANY of the following conditions are met:

| Condition | Threshold | Verification |
|-----------|-----------|--------------|
| S3 API unavailable | > 30 minutes | AWS Health Dashboard + manual test |
| AWS region declared down | Any duration | AWS Health Dashboard |
| Data corruption detected | Any critical data | Integrity checks |
| Security breach confirmed | Immediate | Security team confirmation |

### Pre-Failover Checklist

Before initiating failover, verify:

- [ ] Primary region confirmed unavailable or compromised
- [ ] DR region S3 accessible
- [ ] Replication status verified (last sync time)
- [ ] Engineering leadership approval obtained
- [ ] Customer communication prepared
- [ ] Incident channel established

### Failover Procedure

#### Phase 1: Preparation (0-15 minutes)

```bash
#!/bin/bash
# Phase 1: Failover Preparation

# 1. Verify DR region accessibility
echo "Verifying DR region..."
aws s3 ls s3://greenlang-prod-eu-central-1-data-lake-confidential-replica \
  --region eu-central-1 || exit 1

# 2. Check replication status
echo "Checking replication status..."
aws s3api head-object \
  --bucket greenlang-prod-eu-central-1-data-lake-confidential-replica \
  --key silver/manifest.json \
  --region eu-central-1

# 3. Document last known good state
echo "Documenting replication state..."
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name ReplicationLatency \
  --dimensions Name=BucketName,Value=greenlang-prod-eu-west-1-data-lake-confidential \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average \
  --region eu-west-1 > /tmp/replication-status.json

# 4. Notify stakeholders
echo "Notifying stakeholders..."
slack-cli post #platform-incidents "DR FAILOVER INITIATED - Phase 1 Complete"
```

#### Phase 2: Failover Execution (15-45 minutes)

```bash
#!/bin/bash
# Phase 2: Failover Execution

# 1. Update Route53 health checks to fail primary
echo "Updating Route53 health checks..."
aws route53 update-health-check \
  --health-check-id HC-PRIMARY-S3 \
  --disabled

# 2. Apply DR ConfigMap to Kubernetes
echo "Applying DR configuration..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: storage-config
  namespace: greenlang
data:
  DATA_LAKE_BUCKET: "greenlang-prod-eu-central-1-data-lake-confidential-replica"
  REPORTS_BUCKET: "greenlang-prod-eu-central-1-reports-confidential-replica"
  MODELS_BUCKET: "greenlang-prod-eu-central-1-models-internal-replica"
  AWS_REGION: "eu-central-1"
  DR_MODE: "true"
EOF

# 3. Restart storage services
echo "Restarting storage services..."
kubectl rollout restart deployment/storage-service -n greenlang
kubectl rollout restart deployment/api-gateway -n greenlang

# 4. Wait for rollout
echo "Waiting for rollout..."
kubectl rollout status deployment/storage-service -n greenlang --timeout=300s
kubectl rollout status deployment/api-gateway -n greenlang --timeout=300s

# 5. Update API Gateway stage
echo "Updating API Gateway..."
aws apigateway update-stage \
  --rest-api-id abc123 \
  --stage-name prod \
  --patch-operations op=replace,path=/variables/S3_REGION,value=eu-central-1 \
  --region eu-central-1

echo "Phase 2 Complete"
```

#### Phase 3: Validation (45-60 minutes)

```bash
#!/bin/bash
# Phase 3: Failover Validation

# 1. Test upload to DR region
echo "Testing upload..."
echo "DR test $(date)" > /tmp/dr-test.txt
aws s3 cp /tmp/dr-test.txt \
  s3://greenlang-prod-eu-central-1-data-lake-confidential-replica/dr-validation/test.txt \
  --region eu-central-1 || exit 1

# 2. Test download from DR region
echo "Testing download..."
aws s3 cp \
  s3://greenlang-prod-eu-central-1-data-lake-confidential-replica/dr-validation/test.txt \
  /tmp/dr-download.txt \
  --region eu-central-1 || exit 1

# 3. Test presigned URL generation
echo "Testing presigned URLs..."
aws s3 presign \
  s3://greenlang-prod-eu-central-1-data-lake-confidential-replica/dr-validation/test.txt \
  --region eu-central-1 \
  --expires-in 300

# 4. Verify application health
echo "Checking application health..."
curl -s https://api.greenlang.io/v1/storage/health | jq '.status'

# 5. Run integration tests
echo "Running integration tests..."
pytest tests/integration/storage/ --dr-mode

# 6. Update status page
echo "Updating status page..."
statuspage-cli incident update --status resolved --message "Failover to DR region complete"

echo "FAILOVER COMPLETE - Operating in DR Mode"
```

---

## Failback Procedures

### Failback Prerequisites

Before initiating failback:

- [ ] Primary region confirmed stable for 24+ hours
- [ ] No ongoing AWS incidents in primary region
- [ ] All DR data synchronized back to primary
- [ ] Change approval obtained
- [ ] Maintenance window scheduled
- [ ] Customer communication sent

### Failback Procedure

#### Phase 1: Data Synchronization

```bash
#!/bin/bash
# Phase 1: Sync data from DR to Primary

# 1. Verify primary region accessibility
echo "Verifying primary region..."
aws s3 ls s3://greenlang-prod-eu-west-1-data-lake-confidential \
  --region eu-west-1 || exit 1

# 2. Sync new data created during DR
echo "Syncing data from DR to Primary..."

# Data Lake
aws s3 sync \
  s3://greenlang-prod-eu-central-1-data-lake-confidential-replica/ \
  s3://greenlang-prod-eu-west-1-data-lake-confidential/ \
  --source-region eu-central-1 \
  --region eu-west-1 \
  --exclude "landing/*" \
  --only-show-errors

# Reports
aws s3 sync \
  s3://greenlang-prod-eu-central-1-reports-confidential-replica/ \
  s3://greenlang-prod-eu-west-1-reports-confidential/ \
  --source-region eu-central-1 \
  --region eu-west-1 \
  --only-show-errors

# Models
aws s3 sync \
  s3://greenlang-prod-eu-central-1-models-internal-replica/ \
  s3://greenlang-prod-eu-west-1-models-internal/ \
  --source-region eu-central-1 \
  --region eu-west-1 \
  --only-show-errors

# 3. Verify sync completion
echo "Verifying sync..."
aws s3api list-objects-v2 \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --query 'Contents | length(@)' \
  --region eu-west-1
```

#### Phase 2: Failback Execution

```bash
#!/bin/bash
# Phase 2: Failback Execution

# 1. Re-enable Route53 health checks for primary
echo "Re-enabling primary health checks..."
aws route53 update-health-check \
  --health-check-id HC-PRIMARY-S3 \
  --no-disabled

# 2. Apply Primary ConfigMap to Kubernetes
echo "Applying primary configuration..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: storage-config
  namespace: greenlang
data:
  DATA_LAKE_BUCKET: "greenlang-prod-eu-west-1-data-lake-confidential"
  REPORTS_BUCKET: "greenlang-prod-eu-west-1-reports-confidential"
  MODELS_BUCKET: "greenlang-prod-eu-west-1-models-internal"
  AWS_REGION: "eu-west-1"
  DR_MODE: "false"
EOF

# 3. Restart services
echo "Restarting services..."
kubectl rollout restart deployment/storage-service -n greenlang
kubectl rollout restart deployment/api-gateway -n greenlang

# 4. Wait for rollout
kubectl rollout status deployment/storage-service -n greenlang --timeout=300s
kubectl rollout status deployment/api-gateway -n greenlang --timeout=300s

echo "Phase 2 Complete"
```

#### Phase 3: Validation and Cleanup

```bash
#!/bin/bash
# Phase 3: Validation and Cleanup

# 1. Verify services are using primary region
echo "Verifying primary region usage..."
curl -s https://api.greenlang.io/v1/storage/health | jq '.region'

# 2. Test operations
echo "Testing operations..."
pytest tests/integration/storage/

# 3. Verify replication is active
echo "Verifying replication..."
aws s3api get-bucket-replication \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --region eu-west-1

# 4. Clean up DR validation objects
echo "Cleaning up..."
aws s3 rm s3://greenlang-prod-eu-central-1-data-lake-confidential-replica/dr-validation/ \
  --recursive \
  --region eu-central-1

# 5. Update monitoring
echo "Resetting monitoring to primary region..."
kubectl apply -f k8s/monitoring/cloudwatch-config.yaml

echo "FAILBACK COMPLETE - Operating in Primary Region"
```

---

## Testing Schedule

### Annual DR Test Calendar

| Quarter | Test Type | Scope | Duration | Participants |
|---------|-----------|-------|----------|--------------|
| Q1 (March) | Full Failover | Production | 4 hours | Full team |
| Q2 (June) | Tabletop Exercise | All scenarios | 2 hours | Leadership + Ops |
| Q3 (September) | Partial Failover | Staging | 2 hours | Ops team |
| Q4 (December) | Full Failover | Production | 4 hours | Full team |

### Test Procedures

#### Full Failover Test

1. **Pre-Test (1 day before)**
   - Notify stakeholders
   - Schedule maintenance window
   - Prepare test scripts
   - Brief test participants

2. **Test Execution (4 hours)**
   - Execute failover procedure
   - Run validation tests
   - Document observations
   - Execute failback procedure

3. **Post-Test (1 day after)**
   - Compile test report
   - Identify improvements
   - Update procedures
   - Schedule follow-up items

#### Tabletop Exercise

1. **Scenario Presentation** (30 min)
   - Present disaster scenario
   - Explain constraints

2. **Response Discussion** (60 min)
   - Walk through procedures
   - Identify gaps
   - Discuss alternatives

3. **Action Items** (30 min)
   - Document improvements
   - Assign owners
   - Set deadlines

### Test Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Failover Time | < 60 min | Start to validation complete |
| Data Loss | < 15 min | RPO achievement |
| Application Uptime | > 99% | During failover |
| Team Response | < 15 min | First responder engagement |

---

## Contact Escalation

### Escalation Matrix

| Time | Level | Contact | Method |
|------|-------|---------|--------|
| 0-15 min | L1: On-Call Engineer | PagerDuty | Auto-page |
| 15-30 min | L2: Platform Lead | PagerDuty | Manual page |
| 30-60 min | L3: Engineering Director | Phone | Direct call |
| 60+ min | L4: VP Engineering | Phone | Direct call |

### Contact Directory

| Role | Name | Phone | Email |
|------|------|-------|-------|
| Platform On-Call | Rotation | PagerDuty | platform-oncall@greenlang.io |
| Platform Lead | [Name] | +1-XXX-XXX-XXXX | platform-lead@greenlang.io |
| Engineering Director | [Name] | +1-XXX-XXX-XXXX | eng-director@greenlang.io |
| VP Engineering | [Name] | +1-XXX-XXX-XXXX | vp-eng@greenlang.io |
| AWS TAM | [Name] | AWS Console | aws-tam@greenlang.io |

### Communication Channels

| Channel | Purpose | Audience |
|---------|---------|----------|
| #platform-incidents | Real-time updates | Engineering |
| #incidents-exec | Executive updates | Leadership |
| Status Page | Customer updates | External |
| Email | Formal notifications | All stakeholders |

### Communication Templates

#### Initial Notification

```
Subject: [P1] DR Activation - GreenLang Storage

Severity: P1 - Critical
Status: DR Failover Initiated
Time: [TIMESTAMP]

Summary:
Primary region [eu-west-1] storage is unavailable. Initiating failover to DR region [eu-central-1].

Impact:
- Expected service interruption: 30-60 minutes
- Data loss potential: Up to 15 minutes (RPO)

Next Update: [TIME + 30 min]

Incident Commander: [NAME]
```

#### Resolution Notification

```
Subject: [RESOLVED] DR Activation - GreenLang Storage

Severity: P1 - Critical
Status: Resolved
Duration: [X hours Y minutes]

Summary:
Services have been restored. Operating in [PRIMARY/DR] region.

Impact Summary:
- Total downtime: [X minutes]
- Data loss: [None / X minutes]
- Affected users: [Count]

Post-Incident:
- Postmortem scheduled: [DATE]
- Report due: [DATE]

Questions: Contact platform-ops@greenlang.io
```

---

## Related Documents

- [Architecture Guide](architecture-guide.md)
- [Operations Runbook](operations-runbook.md)
- [Access Procedures](access-procedures.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Platform Engineering | Initial release |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Platform Lead | | | |
| Security Lead | | | |
| Engineering Director | | | |
| VP Engineering | | | |
