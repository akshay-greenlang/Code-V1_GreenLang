# GreenLang Artifact Storage Compliance Guide

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | Compliance / Platform Engineering |
| Classification | Confidential |
| Review Cycle | Annually |

---

## Table of Contents

1. [Overview](#overview)
2. [Regulatory Requirements Mapping](#regulatory-requirements-mapping)
3. [Encryption Requirements](#encryption-requirements)
4. [Access Control Requirements](#access-control-requirements)
5. [Audit Logging Requirements](#audit-logging-requirements)
6. [Data Classification](#data-classification)
7. [Retention Policy Matrix](#retention-policy-matrix)

---

## Overview

### Purpose

This document defines the compliance requirements for GreenLang's artifact storage infrastructure, ensuring alignment with regulatory frameworks including CSRD/ESRS, CBAM, SOX, and GDPR.

### Scope

This compliance guide covers:
- Data retention requirements by regulation
- Encryption and access control standards
- Audit logging and monitoring
- Data classification framework
- Compliance verification procedures

### Compliance Framework

```
                              GreenLang Compliance Framework
+--------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Regulatory Requirements                 Technical Controls              Verification              |
|  +------------------------+              +---------------------+         +-------------------+     |
|  |                        |              |                     |         |                   |     |
|  |  CSRD/ESRS            |------------->|  Data Retention     |-------->|  Internal Audit   |     |
|  |  - Sustainability data |              |  - 10 year retention|         |  - Quarterly      |     |
|  |  - Audit trail         |              |  - Versioning       |         |                   |     |
|  |                        |              |                     |         |                   |     |
|  +------------------------+              +---------------------+         +-------------------+     |
|                                                                                                    |
|  +------------------------+              +---------------------+         +-------------------+     |
|  |                        |              |                     |         |                   |     |
|  |  CBAM                  |------------->|  Document Archive   |-------->|  EU Verification  |     |
|  |  - Import declarations |              |  - 7 year retention |         |  - Annual         |     |
|  |  - Emission calcs      |              |  - Immutability     |         |                   |     |
|  |                        |              |                     |         |                   |     |
|  +------------------------+              +---------------------+         +-------------------+     |
|                                                                                                    |
|  +------------------------+              +---------------------+         +-------------------+     |
|  |                        |              |                     |         |                   |     |
|  |  SOX                   |------------->|  Access Controls    |-------->|  External Audit   |     |
|  |  - Financial reporting |              |  - IAM policies     |         |  - Annual         |     |
|  |  - Change management   |              |  - MFA required     |         |                   |     |
|  |                        |              |                     |         |                   |     |
|  +------------------------+              +---------------------+         +-------------------+     |
|                                                                                                    |
|  +------------------------+              +---------------------+         +-------------------+     |
|  |                        |              |                     |         |                   |     |
|  |  GDPR                  |------------->|  Data Protection    |-------->|  DPA Review       |     |
|  |  - Personal data       |              |  - Encryption       |         |  - On request     |     |
|  |  - Data subject rights |              |  - Access logging   |         |                   |     |
|  |                        |              |                     |         |                   |     |
|  +------------------------+              +---------------------+         +-------------------+     |
|                                                                                                    |
+--------------------------------------------------------------------------------------------------+
```

---

## Regulatory Requirements Mapping

### CSRD/ESRS Requirements

The Corporate Sustainability Reporting Directive (CSRD) and European Sustainability Reporting Standards (ESRS) require:

| Requirement | ESRS Reference | Implementation |
|-------------|----------------|----------------|
| Sustainability data retention | ESRS 1 | 10-year retention in S3 with versioning |
| Audit trail of calculations | ESRS 1, Annex II | Immutable logs in CloudTrail + S3 |
| Traceability to source data | ESRS 2 | Data lineage tracking with metadata |
| Disclosure documentation | ESRS 1 | Versioned reports in Reports bucket |
| Double materiality evidence | ESRS 1 | Supporting documents with retention |

**Storage Implementation:**

```yaml
csrd_compliance:
  buckets:
    - name: greenlang-prod-eu-west-1-data-lake-confidential
      prefixes:
        - silver/csrd/
        - gold/csrd/
      retention: 10 years
      versioning: enabled
      object_lock: governance

    - name: greenlang-prod-eu-west-1-reports-confidential
      prefixes:
        - reports/csrd/
      retention: 10 years
      versioning: enabled
      object_lock: governance
```

### CBAM Requirements

The Carbon Border Adjustment Mechanism requires:

| Requirement | CBAM Reference | Implementation |
|-------------|----------------|----------------|
| Import declaration records | Article 6 | Archived in Reports bucket, 7 years |
| Emission calculation data | Article 7 | Stored in Data Lake, 7 years |
| Verification documents | Article 8 | Immutable storage with Object Lock |
| Supplier communications | Article 10 | Encrypted storage, 7 years |
| Payment records | Article 22 | Integration with financial systems |

**Storage Implementation:**

```yaml
cbam_compliance:
  buckets:
    - name: greenlang-prod-eu-west-1-data-lake-confidential
      prefixes:
        - bronze/cbam/
        - silver/cbam/
      retention: 7 years
      versioning: enabled

    - name: greenlang-prod-eu-west-1-reports-confidential
      prefixes:
        - reports/cbam/declarations/
        - reports/cbam/calculations/
      retention: 7 years
      versioning: enabled
      object_lock: compliance
```

### SOX Audit Requirements

Sarbanes-Oxley compliance for financial reporting requires:

| Requirement | SOX Section | Implementation |
|-------------|-------------|----------------|
| Access controls | Section 404 | IAM policies, MFA, role-based access |
| Change management | Section 404 | Versioning, CloudTrail logging |
| Segregation of duties | Section 404 | Separate read/write roles |
| Audit trail | Section 302 | Immutable CloudTrail logs |
| Data integrity | Section 404 | Checksums, versioning, encryption |

**Access Control Implementation:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SOXReadOnlyAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::greenlang-prod-*-reports-confidential",
        "arn:aws:s3:::greenlang-prod-*-reports-confidential/*"
      ],
      "Condition": {
        "Bool": {"aws:MultiFactorAuthPresent": "true"}
      }
    },
    {
      "Sid": "SOXWriteAccess",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::greenlang-prod-*-reports-confidential/reports/*",
      "Condition": {
        "Bool": {"aws:MultiFactorAuthPresent": "true"},
        "StringEquals": {
          "aws:PrincipalTag/role": "report-generator"
        }
      }
    }
  ]
}
```

### GDPR Requirements

General Data Protection Regulation requirements for personal data:

| Requirement | GDPR Article | Implementation |
|-------------|--------------|----------------|
| Lawful basis for processing | Article 6 | Documented in metadata |
| Data minimization | Article 5 | Retention policies, auto-deletion |
| Right to erasure | Article 17 | Deletion procedures, audit trail |
| Right to access | Article 15 | Export functionality |
| Data portability | Article 20 | Standard export formats |
| Encryption | Article 32 | SSE-KMS, TLS 1.2+ |
| Breach notification | Article 33 | Monitoring, alerting procedures |

**Personal Data Handling:**

```python
# Pseudonymization for personal data
def store_personal_data(data: dict, tenant_id: str) -> str:
    """Store personal data with pseudonymization."""

    # Extract and pseudonymize personal identifiers
    personal_fields = ['name', 'email', 'phone']
    pseudonymized = {}

    for field in personal_fields:
        if field in data:
            # Generate pseudonym
            pseudonym = hashlib.sha256(
                f"{data[field]}:{PEPPER}".encode()
            ).hexdigest()[:16]

            pseudonymized[field] = pseudonym

            # Store mapping separately with stricter access
            store_pseudonym_mapping(tenant_id, pseudonym, data[field])

    # Store pseudonymized data
    data_to_store = {**data, **pseudonymized}

    key = f"silver/personal/{tenant_id}/{uuid.uuid4()}.json"
    s3.put_object(
        Bucket=DATA_LAKE_BUCKET,
        Key=key,
        Body=json.dumps(data_to_store),
        Metadata={
            'gdpr-lawful-basis': 'contract',
            'data-subject-type': 'employee',
            'retention-period': '7-years'
        },
        ServerSideEncryption='aws:kms',
        SSEKMSKeyId=PERSONAL_DATA_KMS_KEY
    )

    return key
```

---

## Encryption Requirements

### Encryption Standards

| Data State | Standard | Implementation |
|------------|----------|----------------|
| At Rest | AES-256 | SSE-S3 or SSE-KMS |
| In Transit | TLS 1.2+ | HTTPS enforced |
| Key Management | AWS KMS | CMK with rotation |

### Encryption Configuration by Data Type

| Data Type | Encryption | KMS Key | Rationale |
|-----------|------------|---------|-----------|
| Public data | SSE-S3 | AWS managed | Cost-effective |
| Internal data | SSE-S3 | AWS managed | Standard protection |
| Confidential data | SSE-KMS | Customer managed | Audit trail |
| Personal data (GDPR) | SSE-KMS | Dedicated CMK | Compliance |
| Financial data (SOX) | SSE-KMS | Dedicated CMK | Audit requirements |

### KMS Key Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Enable IAM policies",
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::123456789012:root"},
      "Action": "kms:*",
      "Resource": "*"
    },
    {
      "Sid": "Allow S3 service",
      "Effect": "Allow",
      "Principal": {"Service": "s3.amazonaws.com"},
      "Action": ["kms:GenerateDataKey", "kms:Decrypt"],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "s3.eu-west-1.amazonaws.com",
          "kms:CallerAccount": "123456789012"
        }
      }
    },
    {
      "Sid": "Allow authorized roles",
      "Effect": "Allow",
      "Principal": {"AWS": [
        "arn:aws:iam::123456789012:role/greenlang-storage-service",
        "arn:aws:iam::123456789012:role/greenlang-audit-role"
      ]},
      "Action": [
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ],
      "Resource": "*"
    }
  ]
}
```

### Bucket Policy Enforcing Encryption

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyUnencryptedUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::greenlang-prod-*-confidential/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": ["aws:kms", "AES256"]
        }
      }
    },
    {
      "Sid": "DenyUnencryptedTransport",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::greenlang-prod-*",
        "arn:aws:s3:::greenlang-prod-*/*"
      ],
      "Condition": {
        "Bool": {"aws:SecureTransport": "false"}
      }
    }
  ]
}
```

---

## Access Control Requirements

### Role-Based Access Control (RBAC)

| Role | Read | Write | Delete | Admin |
|------|------|-------|--------|-------|
| Application Service | Yes (own data) | Yes (own data) | No | No |
| Data Engineer | Yes (all) | Yes (non-prod) | No | No |
| Platform Admin | Yes (all) | Yes (all) | Yes (with approval) | Yes |
| Auditor | Yes (all) | No | No | No |
| Compliance Officer | Yes (all) | No | No | No |

### IAM Policy Templates

**Application Service Role:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadOwnTenantData",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:GetObjectVersion"],
      "Resource": "arn:aws:s3:::greenlang-prod-*-confidential/*",
      "Condition": {
        "StringEquals": {
          "s3:ExistingObjectTag/tenant-id": "${aws:PrincipalTag/tenant-id}"
        }
      }
    },
    {
      "Sid": "WriteOwnTenantData",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:PutObjectTagging"],
      "Resource": "arn:aws:s3:::greenlang-prod-*-confidential/*",
      "Condition": {
        "StringEquals": {
          "s3:RequestObjectTag/tenant-id": "${aws:PrincipalTag/tenant-id}"
        }
      }
    }
  ]
}
```

**Auditor Role:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadOnlyAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetObjectTagging",
        "s3:ListBucket",
        "s3:ListBucketVersions"
      ],
      "Resource": [
        "arn:aws:s3:::greenlang-prod-*",
        "arn:aws:s3:::greenlang-prod-*/*"
      ],
      "Condition": {
        "Bool": {"aws:MultiFactorAuthPresent": "true"}
      }
    },
    {
      "Sid": "DenyWrite",
      "Effect": "Deny",
      "Action": [
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:PutBucketPolicy"
      ],
      "Resource": "*"
    }
  ]
}
```

### Multi-Factor Authentication Requirements

| Action | MFA Required |
|--------|--------------|
| Read confidential data | Yes (production) |
| Write to any production bucket | Yes |
| Delete any object | Yes |
| Modify bucket policies | Yes |
| Access audit logs | Yes |

---

## Audit Logging Requirements

### CloudTrail Configuration

```json
{
  "Name": "greenlang-s3-audit-trail",
  "S3BucketName": "greenlang-prod-eu-west-1-audit-logs-confidential",
  "S3KeyPrefix": "cloudtrail/s3",
  "IncludeGlobalServiceEvents": false,
  "IsMultiRegionTrail": true,
  "EnableLogFileValidation": true,
  "EventSelectors": [
    {
      "ReadWriteType": "All",
      "IncludeManagementEvents": true,
      "DataResources": [
        {
          "Type": "AWS::S3::Object",
          "Values": [
            "arn:aws:s3:::greenlang-prod-eu-west-1-data-lake-confidential/",
            "arn:aws:s3:::greenlang-prod-eu-west-1-reports-confidential/"
          ]
        }
      ]
    }
  ]
}
```

### S3 Server Access Logging

```bash
# Enable server access logging
aws s3api put-bucket-logging \
  --bucket greenlang-prod-eu-west-1-data-lake-confidential \
  --bucket-logging-status '{
    "LoggingEnabled": {
      "TargetBucket": "greenlang-prod-eu-west-1-audit-logs-confidential",
      "TargetPrefix": "s3-access-logs/data-lake/"
    }
  }'
```

### Log Retention

| Log Type | Retention | Storage Class |
|----------|-----------|---------------|
| CloudTrail (management) | 7 years | Glacier Deep Archive |
| CloudTrail (data events) | 7 years | Glacier Deep Archive |
| S3 Access Logs | 7 years | Glacier Deep Archive |
| Application Logs | 2 years | Glacier IR |

### Log Integrity Verification

```bash
# Verify CloudTrail log integrity
aws cloudtrail validate-logs \
  --trail-arn arn:aws:cloudtrail:eu-west-1:123456789012:trail/greenlang-s3-audit-trail \
  --start-time 2026-01-01T00:00:00Z \
  --end-time 2026-02-01T00:00:00Z
```

---

## Data Classification

### Classification Levels

| Level | Description | Examples | Controls |
|-------|-------------|----------|----------|
| Public | No restrictions | Marketing materials | SSE-S3 |
| Internal | Business use only | Documentation | SSE-S3, IAM |
| Confidential | Restricted access | Customer data, financials | SSE-KMS, MFA, logging |
| Highly Confidential | Need-to-know only | PII, credentials | SSE-KMS (dedicated), MFA, enhanced logging |

### Classification Tagging

```python
def classify_and_tag_object(bucket: str, key: str, classification: str):
    """Apply classification tag to S3 object."""

    valid_classifications = ['public', 'internal', 'confidential', 'highly-confidential']

    if classification not in valid_classifications:
        raise ValueError(f"Invalid classification: {classification}")

    s3.put_object_tagging(
        Bucket=bucket,
        Key=key,
        Tagging={
            'TagSet': [
                {'Key': 'data-classification', 'Value': classification},
                {'Key': 'classified-at', 'Value': datetime.utcnow().isoformat()},
                {'Key': 'classified-by', 'Value': 'auto-classifier'}
            ]
        }
    )
```

### Classification by Data Type

| Data Type | Classification | Rationale |
|-----------|----------------|-----------|
| Emission factors (public) | Public | Publicly available data |
| Calculation methodologies | Internal | Proprietary methods |
| Customer emission data | Confidential | Business sensitive |
| Customer contact info | Highly Confidential | PII under GDPR |
| API credentials | Highly Confidential | Security sensitive |
| Audit reports | Confidential | Regulatory sensitive |

---

## Retention Policy Matrix

### Retention by Regulation

| Regulation | Data Type | Minimum Retention | Our Policy |
|------------|-----------|-------------------|------------|
| CSRD/ESRS | Sustainability data | 10 years | 10 years |
| CBAM | Import declarations | 7 years | 7 years |
| CBAM | Emission calculations | 7 years | 7 years |
| SOX | Financial records | 7 years | 7 years |
| GDPR | Personal data | Minimize | As needed + 30 days |
| Tax (EU) | Tax-relevant records | 10 years | 10 years |

### Retention Implementation

```json
{
  "Rules": [
    {
      "ID": "csrd-10-year-retention",
      "Status": "Enabled",
      "Filter": {
        "And": {
          "Prefix": "silver/csrd/",
          "Tags": [{"Key": "regulation", "Value": "csrd"}]
        }
      },
      "Transitions": [
        {"Days": 365, "StorageClass": "GLACIER_IR"},
        {"Days": 1825, "StorageClass": "DEEP_ARCHIVE"}
      ],
      "Expiration": {"Days": 3650}
    },
    {
      "ID": "cbam-7-year-retention",
      "Status": "Enabled",
      "Filter": {"Prefix": "reports/cbam/"},
      "Transitions": [
        {"Days": 365, "StorageClass": "GLACIER_IR"},
        {"Days": 1095, "StorageClass": "DEEP_ARCHIVE"}
      ],
      "Expiration": {"Days": 2555}
    },
    {
      "ID": "gdpr-minimize-pii",
      "Status": "Enabled",
      "Filter": {
        "Tag": {"Key": "contains-pii", "Value": "true"}
      },
      "Expiration": {"Days": 30},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 7}
    }
  ]
}
```

### Retention Monitoring

```python
def check_retention_compliance():
    """Check objects against retention policies."""

    s3 = boto3.client('s3')
    issues = []

    # Check for objects past retention
    inventory = get_s3_inventory(BUCKET)

    for obj in inventory:
        tags = s3.get_object_tagging(Bucket=BUCKET, Key=obj['Key'])
        retention_tag = next(
            (t for t in tags['TagSet'] if t['Key'] == 'retention-until'),
            None
        )

        if retention_tag:
            retention_date = datetime.fromisoformat(retention_tag['Value'])
            if datetime.utcnow() > retention_date:
                issues.append({
                    'key': obj['Key'],
                    'issue': 'past_retention',
                    'retention_date': retention_date.isoformat()
                })

    return issues
```

---

## Compliance Verification

### Quarterly Compliance Checklist

- [ ] Verify encryption on all confidential buckets
- [ ] Review access logs for unauthorized access attempts
- [ ] Validate CloudTrail log integrity
- [ ] Check retention policy compliance
- [ ] Review and update data classification
- [ ] Verify MFA enforcement
- [ ] Test data recovery procedures
- [ ] Update compliance documentation

### Compliance Reporting

```bash
# Generate compliance report
aws s3api get-bucket-encryption --bucket greenlang-prod-eu-west-1-data-lake-confidential
aws s3api get-bucket-versioning --bucket greenlang-prod-eu-west-1-data-lake-confidential
aws s3api get-bucket-logging --bucket greenlang-prod-eu-west-1-data-lake-confidential
aws s3api get-bucket-lifecycle-configuration --bucket greenlang-prod-eu-west-1-data-lake-confidential
```

---

## Related Documents

- [Architecture Guide](architecture-guide.md)
- [Access Procedures](access-procedures.md)
- [Operations Runbook](operations-runbook.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Compliance / Platform Engineering | Initial release |
