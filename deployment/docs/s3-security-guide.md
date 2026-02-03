# GreenLang S3 Security Guide

## Table of Contents

1. [Overview](#overview)
2. [Security Architecture](#security-architecture)
3. [Encryption Requirements](#encryption-requirements)
4. [Access Control Best Practices](#access-control-best-practices)
5. [Bucket Security Configuration](#bucket-security-configuration)
6. [VPC Endpoint Security](#vpc-endpoint-security)
7. [Object Lock and Retention](#object-lock-and-retention)
8. [Compliance Requirements](#compliance-requirements)
9. [Monitoring and Alerting](#monitoring-and-alerting)
10. [Incident Response Procedures](#incident-response-procedures)
11. [Security Audit Checklist](#security-audit-checklist)
12. [Troubleshooting](#troubleshooting)

---

## Overview

This document provides comprehensive security guidelines for managing Amazon S3 storage within the GreenLang platform. It covers security architecture, encryption requirements, access controls, compliance standards, and incident response procedures.

### Scope

This guide applies to all GreenLang S3 buckets:
- `greenlang-app-data-*` - Application data storage
- `greenlang-backups-*` - Critical backup storage with Object Lock
- `greenlang-audit-logs-*` - Compliance audit logs with retention
- `greenlang-access-logs-*` - S3 access logging

### Security Principles

1. **Defense in Depth**: Multiple layers of security controls
2. **Least Privilege**: Minimal permissions required for each role
3. **Encryption Everywhere**: All data encrypted at rest and in transit
4. **Zero Trust**: Verify every access request
5. **Audit Everything**: Comprehensive logging and monitoring

---

## Security Architecture

### Architecture Diagram

```
                           +------------------+
                           |   AWS WAF        |
                           +--------+---------+
                                    |
                           +--------v---------+
                           | CloudFront       |
                           | (Public Assets)  |
                           +--------+---------+
                                    |
     +------------------------------+------------------------------+
     |                              |                              |
+----v----+                  +------v------+                +------v------+
| VPC     |                  | VPC         |                | VPC         |
| Endpoint|                  | Endpoint    |                | Endpoint    |
| (S3)    |                  | (KMS)       |                | (STS)       |
+---------+                  +-------------+                +-------------+
     |                              |                              |
     +------------------------------+------------------------------+
                                    |
                    +---------------v---------------+
                    |        S3 Buckets             |
                    |  +-------------------------+  |
                    |  | Block Public Access     |  |
                    |  | Bucket Policies         |  |
                    |  | ACLs Disabled           |  |
                    |  | Encryption (KMS)        |  |
                    |  | Versioning              |  |
                    |  | Object Lock             |  |
                    |  +-------------------------+  |
                    +-------------------------------+
                                    |
                    +---------------v---------------+
                    |     Access Logging            |
                    |     CloudTrail                |
                    |     CloudWatch Metrics        |
                    +-------------------------------+
```

### Security Layers

| Layer | Control | Purpose |
|-------|---------|---------|
| Network | VPC Endpoints | Restrict traffic to AWS private network |
| Network | Security Groups | Control network access |
| Identity | IAM Policies | Define who can access what |
| Identity | IRSA | Kubernetes pod identity |
| Data | KMS Encryption | Protect data at rest |
| Data | TLS | Protect data in transit |
| Application | Bucket Policies | Fine-grained access control |
| Audit | CloudTrail | API activity logging |
| Audit | Access Logs | Request-level logging |

---

## Encryption Requirements

### Encryption at Rest

All S3 buckets MUST use server-side encryption with AWS KMS (SSE-KMS).

**Configuration Requirements:**
- KMS key rotation: Enabled (annual)
- Bucket key: Enabled (cost optimization)
- Default encryption: `aws:kms`

```hcl
# Terraform configuration
resource "aws_s3_bucket_server_side_encryption_configuration" "example" {
  bucket = aws_s3_bucket.example.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_encryption.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}
```

### Encryption in Transit

All S3 requests MUST use HTTPS (TLS 1.2+).

**Bucket Policy to Enforce:**
```json
{
  "Sid": "DenyNonSSLRequests",
  "Effect": "Deny",
  "Principal": "*",
  "Action": "s3:*",
  "Resource": [
    "arn:aws:s3:::bucket-name",
    "arn:aws:s3:::bucket-name/*"
  ],
  "Condition": {
    "Bool": {
      "aws:SecureTransport": "false"
    }
  }
}
```

### Key Management

| Key Type | Purpose | Rotation | Access |
|----------|---------|----------|--------|
| CMK (Customer Managed) | S3 encryption | Annual (automatic) | Restricted to S3 service + admins |
| Bucket Key | Per-bucket encryption | Per-request | Derived from CMK |

---

## Access Control Best Practices

### IAM Policy Design

1. **Use Resource-Based Policies**: Bucket policies for cross-account access
2. **Use Identity-Based Policies**: IAM policies for same-account access
3. **Avoid Wildcards**: Never use `*` for principals in Allow statements
4. **Condition Keys**: Always add conditions to restrict access

**Example IAM Policy for Application Access:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AppDataReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::greenlang-app-data-*",
        "arn:aws:s3:::greenlang-app-data-*/*"
      ],
      "Condition": {
        "StringEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms"
        }
      }
    }
  ]
}
```

### Service Account Configuration (Kubernetes)

Use IAM Roles for Service Accounts (IRSA):

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: greenlang-app
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/greenlang-s3-app-role
```

### MFA Requirements

| Operation | MFA Required |
|-----------|--------------|
| Delete objects in backup bucket | Yes |
| Modify bucket policies | Yes |
| Access audit logs | Yes |
| Bypass Object Lock (GOVERNANCE) | Yes |

---

## Bucket Security Configuration

### Public Access Block (MANDATORY)

All buckets MUST have public access blocked at both bucket and account level:

```hcl
resource "aws_s3_bucket_public_access_block" "example" {
  bucket = aws_s3_bucket.example.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

### Bucket Policy Template

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyNonSSLRequests",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": ["arn:aws:s3:::BUCKET", "arn:aws:s3:::BUCKET/*"],
      "Condition": {
        "Bool": {"aws:SecureTransport": "false"}
      }
    },
    {
      "Sid": "DenyUnencryptedUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::BUCKET/*",
      "Condition": {
        "Null": {"s3:x-amz-server-side-encryption": "true"}
      }
    },
    {
      "Sid": "RestrictToVPCEndpoint",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": ["arn:aws:s3:::BUCKET", "arn:aws:s3:::BUCKET/*"],
      "Condition": {
        "StringNotEquals": {"aws:SourceVpce": "vpce-xxxxxx"}
      }
    }
  ]
}
```

### Versioning Configuration

| Bucket Type | Versioning | MFA Delete |
|-------------|------------|------------|
| App Data | Enabled | Optional |
| Backups | Enabled | Required |
| Audit Logs | Enabled | Required |
| Access Logs | Enabled | Optional |

---

## VPC Endpoint Security

### Gateway Endpoint Configuration

S3 access should only be allowed through VPC Gateway Endpoints:

```hcl
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = var.vpc_id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = var.private_route_table_ids

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "RestrictToGreenLangBuckets"
        Effect    = "Allow"
        Principal = "*"
        Action    = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource  = [
          "arn:aws:s3:::greenlang-*",
          "arn:aws:s3:::greenlang-*/*"
        ]
      }
    ]
  })
}
```

### Endpoint Policy Best Practices

1. Restrict to specific buckets (not `*`)
2. Limit allowed actions
3. Add account conditions
4. Log endpoint usage via CloudTrail

---

## Object Lock and Retention

### GOVERNANCE Mode (Backups)

- Retention period: 90 days
- Can be bypassed by users with `s3:BypassGovernanceRetention` permission
- Requires MFA for bypass

```hcl
resource "aws_s3_bucket_object_lock_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    default_retention {
      mode = "GOVERNANCE"
      days = 90
    }
  }
}
```

### COMPLIANCE Mode (Audit Logs)

- Retention period: 7 years (2555 days)
- Cannot be bypassed by ANY user, including root
- Cannot be shortened once set

```hcl
resource "aws_s3_bucket_object_lock_configuration" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  rule {
    default_retention {
      mode  = "COMPLIANCE"
      years = 7
    }
  }
}
```

### Legal Hold

Apply legal hold for litigation or investigation:

```bash
# Apply legal hold
aws s3api put-object-legal-hold \
  --bucket greenlang-audit-logs-ACCOUNT_ID \
  --key path/to/object \
  --legal-hold Status=ON

# Remove legal hold
aws s3api put-object-legal-hold \
  --bucket greenlang-audit-logs-ACCOUNT_ID \
  --key path/to/object \
  --legal-hold Status=OFF
```

---

## Compliance Requirements

### SOC 2 Type II

| Control | Implementation |
|---------|----------------|
| CC6.1 - Logical Access | IAM policies, bucket policies |
| CC6.6 - Encryption | KMS encryption at rest, TLS in transit |
| CC6.7 - Transmission Security | HTTPS enforcement |
| CC7.2 - Monitoring | CloudTrail, access logs |

### ISO 27001

| Control | Implementation |
|---------|----------------|
| A.9.2.3 - Access Rights | Least privilege IAM policies |
| A.10.1.1 - Encryption Policy | KMS encryption |
| A.12.4.1 - Event Logging | CloudTrail, access logs |
| A.13.2.1 - Data Transfer | VPC endpoints, TLS |

### PCI DSS (if applicable)

| Requirement | Implementation |
|-------------|----------------|
| 3.4 - Render PAN Unreadable | KMS encryption |
| 7.1 - Restrict Access | IAM policies |
| 10.2 - Audit Trails | CloudTrail |
| 10.5 - Secure Audit Trails | Object Lock on audit logs |

### HIPAA (if applicable)

| Safeguard | Implementation |
|-----------|----------------|
| Access Control | IAM, bucket policies |
| Encryption | KMS SSE |
| Audit Controls | CloudTrail, access logs |
| Integrity Controls | Versioning, Object Lock |

---

## Monitoring and Alerting

### CloudWatch Metrics

Monitor these S3 metrics:

| Metric | Threshold | Action |
|--------|-----------|--------|
| 4xxErrors | > 100/5min | Investigate unauthorized access |
| 5xxErrors | > 10/5min | Check S3 service health |
| AllRequests | Anomaly detection | Unusual activity |
| BytesDownloaded | > 1TB/hour | Data exfiltration risk |

### CloudTrail Events

Critical events to monitor:

```json
{
  "eventNames": [
    "DeleteBucket",
    "DeleteBucketPolicy",
    "PutBucketAcl",
    "PutBucketPolicy",
    "PutBucketPublicAccessBlock",
    "DeleteObject",
    "PutObjectAcl"
  ]
}
```

### Alert Configuration

```yaml
# Prometheus alert rule
groups:
  - name: s3-security
    rules:
      - alert: S3UnauthorizedAccess
        expr: aws_s3_4xx_errors_total > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of S3 unauthorized access attempts"
          description: "{{ $value }} 4xx errors in the last 5 minutes"

      - alert: S3DataExfiltration
        expr: rate(aws_s3_bytes_downloaded_total[1h]) > 1e12
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Potential data exfiltration detected"
          description: "More than 1TB downloaded in the last hour"
```

---

## Incident Response Procedures

### Incident Classification

| Severity | Description | Response Time |
|----------|-------------|---------------|
| Critical | Data breach, public exposure | Immediate |
| High | Unauthorized access attempt | < 1 hour |
| Medium | Policy violation | < 4 hours |
| Low | Configuration drift | < 24 hours |

### Response Procedures

#### 1. Data Breach Response

```bash
# 1. Block public access immediately
aws s3api put-public-access-block \
  --bucket BUCKET_NAME \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# 2. Revoke compromised credentials
aws iam update-access-key --access-key-id AKIAXXXXXXX --status Inactive --user-name USERNAME

# 3. Enable bucket versioning if not enabled
aws s3api put-bucket-versioning --bucket BUCKET_NAME --versioning-configuration Status=Enabled

# 4. Collect evidence from CloudTrail
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceName,AttributeValue=BUCKET_NAME \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z
```

#### 2. Unauthorized Access Response

1. Identify the source IP/principal from CloudTrail
2. Block the IP in WAF/Security Groups
3. Rotate affected credentials
4. Review and update bucket policies
5. Document incident and remediation

#### 3. Policy Violation Response

1. Identify the non-compliant configuration
2. Apply corrective action via Terraform
3. Run compliance audit
4. Update monitoring rules

### Post-Incident Actions

1. Root cause analysis
2. Update security controls
3. Security awareness training
4. Update documentation
5. Review with compliance team

---

## Security Audit Checklist

### Daily Checks

- [ ] Review CloudTrail alerts
- [ ] Check access log anomalies
- [ ] Verify backup job completion
- [ ] Monitor error rates

### Weekly Checks

- [ ] Run bucket permission audit script
- [ ] Review IAM access analyzer findings
- [ ] Check for unused credentials
- [ ] Verify encryption on new objects

### Monthly Checks

- [ ] Full compliance audit
- [ ] Review and rotate access keys
- [ ] Update security policies
- [ ] Test disaster recovery

### Quarterly Checks

- [ ] Penetration testing
- [ ] Third-party security audit
- [ ] Review compliance certifications
- [ ] Update security documentation

### Audit Commands

```bash
# Run full bucket audit
python audit-bucket-permissions.py --all-buckets --output audit-report.json

# Verify encryption compliance
python verify-encryption.py --bucket greenlang-app-data --output encryption-report.json

# Analyze access logs
python check-access-logs.py \
  --log-bucket greenlang-access-logs \
  --hours 168 \
  --output access-analysis.json
```

---

## Troubleshooting

### Common Issues

#### Access Denied Errors

1. Check IAM policy permissions
2. Verify bucket policy conditions
3. Check VPC endpoint policy
4. Verify KMS key access

```bash
# Debug access issues
aws sts get-caller-identity
aws s3api get-bucket-policy --bucket BUCKET_NAME
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::ACCOUNT:role/ROLE_NAME \
  --action-names s3:GetObject \
  --resource-arns arn:aws:s3:::BUCKET_NAME/*
```

#### Encryption Errors

1. Verify KMS key exists and is enabled
2. Check KMS key policy permissions
3. Ensure bucket default encryption is configured

```bash
# Check encryption configuration
aws s3api get-bucket-encryption --bucket BUCKET_NAME
aws kms describe-key --key-id KEY_ID
```

#### VPC Endpoint Issues

1. Verify endpoint route in route tables
2. Check endpoint policy
3. Verify DNS resolution

```bash
# Test endpoint connectivity
nslookup s3.amazonaws.com
aws s3 ls s3://BUCKET_NAME --debug
```

### Support Contacts

| Issue Type | Contact |
|------------|---------|
| Security Incident | security@greenlang.io |
| Infrastructure | devops@greenlang.io |
| Compliance | compliance@greenlang.io |
| AWS Support | AWS Support Console |

---

## Appendix

### Reference Documentation

- [AWS S3 Security Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
- [AWS KMS Developer Guide](https://docs.aws.amazon.com/kms/latest/developerguide/)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [CIS AWS Foundations Benchmark](https://www.cisecurity.org/benchmark/amazon_web_services)

### Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2024-01-15 | DevOps Team | Initial release |

### Document Approval

| Role | Name | Date |
|------|------|------|
| Security Lead | | |
| DevOps Lead | | |
| Compliance Officer | | |
