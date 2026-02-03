# GreenLang Artifact Storage Access Procedures

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | Platform Engineering / Security |
| Classification | Internal |

---

## Table of Contents

1. [Overview](#overview)
2. [Access Request Workflow](#access-request-workflow)
3. [Approval Requirements](#approval-requirements)
4. [IAM Policy Templates](#iam-policy-templates)
5. [IRSA Configuration](#irsa-configuration)
6. [Cross-Account Access](#cross-account-access)
7. [Emergency Access Procedures](#emergency-access-procedures)
8. [Access Review Schedule](#access-review-schedule)

---

## Overview

### Purpose

This document defines the procedures for requesting, granting, and managing access to GreenLang's artifact storage infrastructure. All access must follow the principle of least privilege and be documented for audit purposes.

### Scope

These procedures apply to:
- Human users (developers, operators, auditors)
- Service accounts (applications, CI/CD pipelines)
- Cross-account access (partner integrations)
- Emergency access scenarios

### Access Principles

1. **Least Privilege**: Grant minimum permissions required
2. **Need to Know**: Access based on business justification
3. **Time-Bound**: Temporary access where possible
4. **Auditable**: All access logged and reviewable
5. **Segregation**: Separate roles for different functions

---

## Access Request Workflow

### Standard Access Request Process

```
                              Access Request Workflow
+--------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Requestor           Approver              Security           Platform             System          |
|     |                   |                     |                   |                   |            |
|     |  1. Submit        |                     |                   |                   |            |
|     |     Request       |                     |                   |                   |            |
|     |------------------>|                     |                   |                   |            |
|     |                   |                     |                   |                   |            |
|     |                   |  2. Business        |                   |                   |            |
|     |                   |     Review          |                   |                   |            |
|     |                   |-------------------->|                   |                   |            |
|     |                   |                     |                   |                   |            |
|     |                   |                     |  3. Security      |                   |            |
|     |                   |                     |     Review        |                   |            |
|     |                   |                     |------------------>|                   |            |
|     |                   |                     |                   |                   |            |
|     |                   |                     |                   |  4. Implement     |            |
|     |                   |                     |                   |     Access        |            |
|     |                   |                     |                   |------------------>|            |
|     |                   |                     |                   |                   |            |
|     |                   |                     |                   |                   |  5. Access |
|     |<------------------|-------------------- |-------------------|-------------------|   Granted  |
|     |                   |                     |                   |                   |            |
|                                                                                                    |
+--------------------------------------------------------------------------------------------------+
```

### Request Submission

**Submit via**: ServiceNow / Jira (INFRA project)

**Required Information**:

```yaml
access_request:
  requestor:
    name: "[Full Name]"
    email: "[Email]"
    team: "[Team Name]"
    manager: "[Manager Name]"

  access_details:
    type: "[read_only | read_write | admin]"
    buckets:
      - "[Bucket Name]"
    prefixes:
      - "[Prefix Pattern]"
    duration: "[permanent | temporary]"
    expiry_date: "[If temporary]"

  justification:
    business_reason: "[Why is access needed?]"
    data_types: "[What data will be accessed?]"
    use_case: "[How will access be used?]"

  compliance:
    contains_pii: "[yes | no]"
    contains_financial: "[yes | no]"
    regulatory_requirements: "[CSRD | CBAM | SOX | GDPR | None]"
```

### SLA for Access Requests

| Request Type | Target SLA | Escalation |
|--------------|------------|------------|
| Read-only (non-production) | 4 hours | Platform Lead |
| Read-only (production) | 24 hours | Security Team |
| Read-write (any) | 48 hours | Engineering Manager |
| Admin access | 72 hours | VP Engineering |
| Emergency access | 1 hour | Security On-Call |

---

## Approval Requirements

### Approval Matrix

| Access Type | Data Classification | Approvers Required |
|-------------|---------------------|-------------------|
| Read-only | Public/Internal | Team Lead |
| Read-only | Confidential | Team Lead + Security |
| Read-write | Public/Internal | Engineering Manager |
| Read-write | Confidential | Engineering Manager + Security |
| Admin | Any | VP Engineering + Security Lead |
| Cross-account | Any | Security Lead + Platform Lead |

### Approval Workflow by Role

**Developer Access**:
1. Submit request with business justification
2. Team Lead approval (within team scope)
3. Security review (for confidential data)
4. Platform team implementation
5. Requestor notification

**Service Account Access**:
1. Submit request with architecture review
2. Engineering Manager approval
3. Security review
4. Platform team implementation
5. Secret rotation setup

**Auditor Access**:
1. Submit request with audit scope
2. Compliance Officer approval
3. Security review
4. Time-bound access implementation
5. Access logging enabled

---

## IAM Policy Templates

### Read-Only Access Template

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::${BUCKET_NAME}",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["${PREFIX}/*"]
        }
      }
    },
    {
      "Sid": "ReadObjects",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetObjectTagging"
      ],
      "Resource": "arn:aws:s3:::${BUCKET_NAME}/${PREFIX}/*"
    }
  ]
}
```

### Read-Write Access Template

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::${BUCKET_NAME}",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["${PREFIX}/*"]
        }
      }
    },
    {
      "Sid": "ReadWriteObjects",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetObjectTagging",
        "s3:PutObject",
        "s3:PutObjectTagging"
      ],
      "Resource": "arn:aws:s3:::${BUCKET_NAME}/${PREFIX}/*"
    }
  ]
}
```

### Application Service Account Template

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ApplicationReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::greenlang-prod-eu-west-1-data-lake-confidential",
        "arn:aws:s3:::greenlang-prod-eu-west-1-data-lake-confidential/${aws:PrincipalTag/application}/*"
      ],
      "Condition": {
        "StringEquals": {
          "aws:PrincipalTag/environment": "prod"
        }
      }
    },
    {
      "Sid": "KMSDecrypt",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ],
      "Resource": "arn:aws:kms:eu-west-1:123456789012:key/${KMS_KEY_ID}",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "s3.eu-west-1.amazonaws.com"
        }
      }
    }
  ]
}
```

### Auditor Read-Only Template

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AuditReadAll",
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
        "Bool": {"aws:MultiFactorAuthPresent": "true"},
        "IpAddress": {"aws:SourceIp": ["10.0.0.0/8", "192.168.0.0/16"]}
      }
    },
    {
      "Sid": "DenyAllWrite",
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

### Tenant-Scoped Access Template

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "TenantScopedAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::greenlang-prod-eu-west-1-data-lake-confidential",
        "arn:aws:s3:::greenlang-prod-eu-west-1-data-lake-confidential/*"
      ],
      "Condition": {
        "StringEquals": {
          "s3:ExistingObjectTag/tenant-id": "${aws:PrincipalTag/tenant-id}"
        },
        "ForAllValues:StringEquals": {
          "s3:RequestObjectTagKeys": ["tenant-id"]
        },
        "StringEquals": {
          "s3:RequestObjectTag/tenant-id": "${aws:PrincipalTag/tenant-id}"
        }
      }
    }
  ]
}
```

---

## IRSA Configuration

### Overview

IAM Roles for Service Accounts (IRSA) enables Kubernetes pods to assume IAM roles without managing credentials.

### Setup Steps

#### 1. Create IAM OIDC Provider

```bash
# Get OIDC provider URL
OIDC_PROVIDER=$(aws eks describe-cluster \
  --name greenlang-prod \
  --query "cluster.identity.oidc.issuer" \
  --output text | sed 's|https://||')

# Create IAM OIDC provider
aws iam create-open-id-connect-provider \
  --url https://${OIDC_PROVIDER} \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list $(openssl s_client -connect ${OIDC_PROVIDER}:443 2>/dev/null | \
    openssl x509 -fingerprint -noout | cut -d= -f2 | tr -d :)
```

#### 2. Create IAM Role with Trust Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/${OIDC_PROVIDER}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "${OIDC_PROVIDER}:aud": "sts.amazonaws.com",
          "${OIDC_PROVIDER}:sub": "system:serviceaccount:greenlang:cbam-service"
        }
      }
    }
  ]
}
```

#### 3. Create Kubernetes Service Account

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cbam-service
  namespace: greenlang
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/greenlang-cbam-storage-role
```

#### 4. Deploy Pod with Service Account

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cbam-service
  namespace: greenlang
spec:
  template:
    spec:
      serviceAccountName: cbam-service
      containers:
        - name: cbam-service
          image: greenlang/cbam-service:latest
          env:
            - name: AWS_REGION
              value: eu-west-1
```

### IRSA Verification

```python
import boto3

def verify_irsa_credentials():
    """Verify IRSA is providing credentials."""
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()

    print(f"Account: {identity['Account']}")
    print(f"Arn: {identity['Arn']}")
    print(f"UserId: {identity['UserId']}")

    # Verify S3 access
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(
        Bucket='greenlang-prod-eu-west-1-data-lake-confidential',
        Prefix='silver/cbam/',
        MaxKeys=1
    )
    print(f"S3 access verified: {len(response.get('Contents', []))} objects found")
```

---

## Cross-Account Access

### Partner Integration Access

For partner integrations requiring S3 access:

#### 1. Create Cross-Account Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::PARTNER_ACCOUNT_ID:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "${EXTERNAL_ID}"
        },
        "IpAddress": {
          "aws:SourceIp": ["PARTNER_IP_RANGE"]
        }
      }
    }
  ]
}
```

#### 2. Bucket Policy for Cross-Account

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PartnerReadAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:role/partner-integration-role"
      },
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::greenlang-prod-eu-west-1-data-lake-confidential",
        "arn:aws:s3:::greenlang-prod-eu-west-1-data-lake-confidential/shared/partner-${PARTNER_ID}/*"
      ]
    }
  ]
}
```

### Cross-Account Request Process

1. Partner submits integration request
2. Security review of partner AWS account
3. External ID generation
4. Role and bucket policy creation
5. Partner verification testing
6. Access documentation and monitoring setup

---

## Emergency Access Procedures

### Emergency Access Criteria

Emergency access may be granted when:
- Production incident affecting customers
- Security breach investigation
- Regulatory audit with tight deadline
- Disaster recovery operations

### Emergency Access Process

```
                              Emergency Access Process
+--------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  Time        Action                                    Responsible                                 |
|  -----       ------                                    -----------                                 |
|  T+0         Incident declared                         On-call engineer                            |
|  T+5 min     Emergency access requested                Requestor                                   |
|  T+10 min    Security on-call paged                    PagerDuty                                   |
|  T+15 min    Verbal approval (documented)              Security on-call                            |
|  T+20 min    Temporary credentials issued              Platform on-call                            |
|  T+4 hours   Access reviewed                           Security team                               |
|  T+24 hours  Access revoked (unless extended)          Automated                                   |
|  T+48 hours  Post-incident review                      Security + Platform                         |
|                                                                                                    |
+--------------------------------------------------------------------------------------------------+
```

### Emergency Access Implementation

```bash
#!/bin/bash
# emergency-access.sh - Grant temporary emergency access

REQUESTOR=$1
BUCKET=$2
DURATION_HOURS=${3:-4}

# Validate inputs
if [ -z "$REQUESTOR" ] || [ -z "$BUCKET" ]; then
    echo "Usage: emergency-access.sh <requestor-arn> <bucket-name> [duration-hours]"
    exit 1
fi

# Generate unique session name
SESSION_NAME="emergency-$(date +%Y%m%d%H%M%S)"

# Create temporary policy
POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EmergencyAccess",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${BUCKET}", "arn:aws:s3:::${BUCKET}/*"]
    }
  ]
}
EOF
)

# Attach policy with expiration
aws iam put-user-policy \
  --user-name "$REQUESTOR" \
  --policy-name "emergency-${SESSION_NAME}" \
  --policy-document "$POLICY"

# Schedule revocation
echo "aws iam delete-user-policy --user-name $REQUESTOR --policy-name emergency-${SESSION_NAME}" | \
  at now + ${DURATION_HOURS} hours

# Log access grant
aws cloudwatch put-metric-data \
  --namespace GreenLang/Security \
  --metric-name EmergencyAccessGrant \
  --value 1 \
  --dimensions Requestor=${REQUESTOR},Bucket=${BUCKET}

echo "Emergency access granted to $REQUESTOR for $BUCKET"
echo "Access will be revoked in $DURATION_HOURS hours"
```

### Post-Emergency Review

After emergency access:

1. **Document the incident**
   - Reason for emergency access
   - Data accessed
   - Actions taken

2. **Review access logs**
   - Verify only necessary data accessed
   - Check for any anomalies

3. **Update procedures if needed**
   - Identify if permanent access should be granted
   - Update runbooks based on learnings

---

## Access Review Schedule

### Quarterly Access Review

| Review Item | Frequency | Owner | Deliverable |
|-------------|-----------|-------|-------------|
| User access inventory | Quarterly | Security | Access report |
| Service account review | Quarterly | Platform | Service account audit |
| Cross-account access review | Quarterly | Security | Partner access report |
| Emergency access audit | Monthly | Security | Emergency access log |
| Unused access cleanup | Monthly | Platform | Cleanup report |

### Access Review Process

```python
#!/usr/bin/env python3
"""
Quarterly Access Review Script
Generates report of all S3 access for review.
"""

import boto3
from datetime import datetime, timedelta

def generate_access_review_report():
    """Generate quarterly access review report."""

    iam = boto3.client('iam')
    s3 = boto3.client('s3')

    report = {
        'generated_at': datetime.utcnow().isoformat(),
        'review_period': 'Q1 2026',
        'users': [],
        'roles': [],
        'policies': []
    }

    # List all IAM users with S3 access
    paginator = iam.get_paginator('list_users')
    for page in paginator.paginate():
        for user in page['Users']:
            user_policies = iam.list_user_policies(UserName=user['UserName'])
            attached_policies = iam.list_attached_user_policies(UserName=user['UserName'])

            # Check for S3 permissions
            has_s3_access = check_s3_permissions(
                user_policies['PolicyNames'],
                attached_policies['AttachedPolicies']
            )

            if has_s3_access:
                # Get last activity
                last_used = get_last_s3_activity(user['UserName'])

                report['users'].append({
                    'username': user['UserName'],
                    'created': user['CreateDate'].isoformat(),
                    'last_s3_activity': last_used,
                    'policies': user_policies['PolicyNames']
                })

    # List all roles with S3 access
    paginator = iam.get_paginator('list_roles')
    for page in paginator.paginate():
        for role in page['Roles']:
            if 'greenlang' in role['RoleName'].lower():
                role_policies = iam.list_role_policies(RoleName=role['RoleName'])

                report['roles'].append({
                    'role_name': role['RoleName'],
                    'created': role['CreateDate'].isoformat(),
                    'policies': role_policies['PolicyNames']
                })

    return report

def check_s3_permissions(inline_policies, attached_policies):
    """Check if policies include S3 permissions."""
    # Implementation to parse policies
    return True  # Simplified

def get_last_s3_activity(username):
    """Get last S3 activity from CloudTrail."""
    # Query CloudTrail for S3 events by user
    return datetime.utcnow().isoformat()  # Simplified

if __name__ == '__main__':
    report = generate_access_review_report()
    print(json.dumps(report, indent=2))
```

### Cleanup Procedures

```bash
#!/bin/bash
# cleanup-unused-access.sh - Remove access not used in 90 days

# Get users with S3 access
USERS=$(aws iam list-users --query 'Users[].UserName' --output text)

for USER in $USERS; do
    # Check last activity
    LAST_USED=$(aws iam get-user \
        --user-name "$USER" \
        --query 'User.PasswordLastUsed' \
        --output text)

    # If not used in 90 days, flag for review
    if [ "$LAST_USED" != "None" ]; then
        DAYS_AGO=$(( ($(date +%s) - $(date -d "$LAST_USED" +%s)) / 86400 ))

        if [ $DAYS_AGO -gt 90 ]; then
            echo "REVIEW: $USER - Last used $DAYS_AGO days ago"

            # Optionally disable access
            # aws iam delete-user-policy --user-name "$USER" --policy-name "s3-access"
        fi
    fi
done
```

---

## Related Documents

- [Architecture Guide](architecture-guide.md)
- [Compliance Guide](compliance-guide.md)
- [Operations Runbook](operations-runbook.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Platform Engineering / Security | Initial release |
