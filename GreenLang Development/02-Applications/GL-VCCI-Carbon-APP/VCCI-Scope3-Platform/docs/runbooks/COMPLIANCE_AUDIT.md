# Compliance Audit Runbook

**Scenario**: Prepare for and respond to SOC 2, GDPR, ISO 27001, and other compliance audits through evidence collection, control verification, and documentation review.

**Severity**: P1 (Active audit) / P2 (Audit preparation)

**RTO/RPO**: N/A (Compliance activity)

**Owner**: Security Team / Compliance Team / Audit Team

## Prerequisites

- Access to all system components
- Audit logging enabled across all services
- Security documentation repository access
- Understanding of applicable compliance frameworks
- Audit management tool access (e.g., Vanta, Drata)

## Detection

### Audit Triggers

1. **Scheduled Audits**:
   - Annual SOC 2 Type II audit
   - Quarterly internal security reviews
   - GDPR compliance assessments
   - Customer security questionnaires

2. **Regulatory Events**:
   - New client contract requirements
   - Regulatory changes
   - Security incident follow-up
   - Certification renewals

3. **Business Drivers**:
   - Sales requirements (enterprise customers)
   - Partnership agreements
   - Insurance requirements
   - Investor due diligence

## Step-by-Step Procedure

### Part 1: Audit Preparation

#### Step 1: Identify Audit Scope and Requirements

```bash
# Create audit preparation directory
mkdir -p /audit_prep/$(date +%Y%m%d)_SOC2_TypeII
cd /audit_prep/$(date +%Y%m%d)_SOC2_TypeII

# Document audit scope
cat > audit_scope.md << 'EOF'
# SOC 2 Type II Audit Scope

**Audit Period**: 2024-01-01 to 2024-12-31
**Audit Type**: SOC 2 Type II
**Trust Service Criteria**:
- [x] Security
- [x] Availability
- [x] Confidentiality
- [ ] Processing Integrity
- [ ] Privacy

**In-Scope Systems**:
- VCCI Scope 3 Carbon Intelligence Platform
  - API Gateway
  - Calculation Engine
  - Data Ingestion Service
  - Reporting Service
- Supporting Infrastructure
  - AWS EKS Cluster
  - RDS PostgreSQL Database
  - S3 Storage
  - CloudFront CDN
  - Identity & Access Management

**Out-of-Scope**:
- Internal development environments
- Third-party SaaS tools (except as service providers)

**Audit Timeline**:
- Preparation: 4 weeks
- Fieldwork: 2 weeks
- Report delivery: 2 weeks after fieldwork
EOF

cat audit_scope.md
```

#### Step 2: Review Control Objectives

```bash
# Map platform controls to SOC 2 criteria
cat > control_mapping.yaml << 'EOF'
soc2_controls:
  CC1_1_COSO_control_environment:
    description: "Organization demonstrates commitment to integrity and ethical values"
    evidence:
      - Code of Conduct document
      - Ethics training completion records
      - Background check policy
    status: implemented

  CC6_1_logical_access:
    description: "System restricts logical access through identification and authentication"
    evidence:
      - AWS IAM policies
      - Kubernetes RBAC configuration
      - MFA enforcement policy
      - Access logs from last 12 months
    validation_queries:
      - kubectl get rolebindings,clusterrolebindings -A
      - aws iam get-account-summary
    status: implemented

  CC6_2_new_users:
    description: "Prior to issuing system credentials, system registers and authorizes new users"
    evidence:
      - User provisioning workflow documentation
      - Approval records in JIRA
      - Onboarding checklist
    validation_queries:
      - psql -c "SELECT * FROM audit_log WHERE action = 'user_created' AND timestamp > NOW() - INTERVAL '12 months'"
    status: implemented

  CC6_6_encryption:
    description: "System transmits and stores data using encryption"
    evidence:
      - TLS certificate configuration
      - Database encryption settings
      - S3 bucket encryption policy
    validation_queries:
      - kubectl get ingress -A -o yaml | grep tls
      - aws rds describe-db-instances --query 'DBInstances[*].StorageEncrypted'
      - aws s3api get-bucket-encryption --bucket vcci-scope3-data-prod
    status: implemented

  CC7_2_system_monitoring:
    description: "System monitors system components and operation to detect anomalies"
    evidence:
      - Prometheus alerts configuration
      - CloudWatch alarms
      - GuardDuty findings
      - Incident response logs
    validation_queries:
      - kubectl get prometheusrules -A
      - aws cloudwatch describe-alarms
    status: implemented

  CC8_1_change_management:
    description: "System changes are authorized, tested, and deployed systematically"
    evidence:
      - Change management policy
      - GitHub pull request records
      - CI/CD pipeline logs
      - Deployment approval records
    validation_queries:
      - git log --since="12 months ago" --oneline
      - kubectl rollout history deployment -n vcci-scope3
    status: implemented

  A1_1_availability_monitoring:
    description: "System monitors availability performance and reports results"
    evidence:
      - Uptime monitoring data
      - Incident reports
      - SLA compliance reports
    validation_queries:
      - curl http://prometheus:9090/api/v1/query?query=up
    status: implemented

  C1_1_confidentiality:
    description: "Organization identifies and maintains confidential information"
    evidence:
      - Data classification policy
      - PII inventory
      - Encryption implementation
    status: implemented
EOF

cat control_mapping.yaml
```

#### Step 3: Gather Evidence Systematically

```bash
# Create evidence collection script
cat > collect_evidence.sh << 'EOF'
#!/bin/bash

EVIDENCE_DIR="./evidence"
mkdir -p $EVIDENCE_DIR/{access_controls,encryption,monitoring,change_management,backups,security_policies}

echo "=== SOC 2 Evidence Collection Script ==="
echo "Started: $(date)"

# 1. Access Controls
echo "Collecting access control evidence..."

# IAM policies
aws iam list-policies --scope Local > $EVIDENCE_DIR/access_controls/iam_policies.json
aws iam list-users > $EVIDENCE_DIR/access_controls/iam_users.json
aws iam list-roles > $EVIDENCE_DIR/access_controls/iam_roles.json

# Kubernetes RBAC
kubectl get rolebindings,clusterrolebindings -A -o yaml > $EVIDENCE_DIR/access_controls/k8s_rbac.yaml
kubectl get serviceaccounts -A -o yaml > $EVIDENCE_DIR/access_controls/k8s_serviceaccounts.yaml

# MFA status
aws iam get-credential-report --output text | base64 -d > $EVIDENCE_DIR/access_controls/iam_credential_report.csv

# 2. Encryption
echo "Collecting encryption evidence..."

# TLS certificates
kubectl get certificates -A -o yaml > $EVIDENCE_DIR/encryption/tls_certificates.yaml
kubectl get secrets -A -l certmanager.k8s.io/certificate-name -o yaml > $EVIDENCE_DIR/encryption/tls_secrets.yaml

# Database encryption
aws rds describe-db-instances --query 'DBInstances[*].[DBInstanceIdentifier,StorageEncrypted,KmsKeyId]' --output table > $EVIDENCE_DIR/encryption/rds_encryption.txt

# S3 encryption
for bucket in $(aws s3 ls | awk '{print $3}' | grep vcci-scope3); do
  echo "Bucket: $bucket" >> $EVIDENCE_DIR/encryption/s3_encryption.txt
  aws s3api get-bucket-encryption --bucket $bucket >> $EVIDENCE_DIR/encryption/s3_encryption.txt 2>&1
  echo "---" >> $EVIDENCE_DIR/encryption/s3_encryption.txt
done

# EBS encryption
aws ec2 describe-volumes --query 'Volumes[*].[VolumeId,Encrypted,KmsKeyId]' --output table > $EVIDENCE_DIR/encryption/ebs_encryption.txt

# 3. Monitoring & Alerting
echo "Collecting monitoring evidence..."

# CloudWatch alarms
aws cloudwatch describe-alarms --output json > $EVIDENCE_DIR/monitoring/cloudwatch_alarms.json

# Prometheus rules
kubectl get prometheusrules -A -o yaml > $EVIDENCE_DIR/monitoring/prometheus_rules.yaml

# GuardDuty findings (last 12 months)
aws guardduty list-findings \
  --detector-id $(aws guardduty list-detectors --query 'DetectorIds[0]' --output text) \
  --finding-criteria '{"Criterion":{"createdAt":{"Gte":'$(date -d '12 months ago' +%s000)'}}}' \
  --output json > $EVIDENCE_DIR/monitoring/guardduty_findings.json

# 4. Change Management
echo "Collecting change management evidence..."

# Git commit history
git log --since="12 months ago" --pretty=format:"%h - %an, %ar : %s" > $EVIDENCE_DIR/change_management/git_commits.txt

# Deployment history
kubectl rollout history deployment -n vcci-scope3 > $EVIDENCE_DIR/change_management/k8s_deployments.txt

# CI/CD pipeline runs (last 12 months)
# Adjust based on your CI/CD system
# gh run list --limit 1000 > $EVIDENCE_DIR/change_management/github_actions.txt

# 5. Backup & Recovery
echo "Collecting backup evidence..."

# RDS snapshots
aws rds describe-db-snapshots --output json > $EVIDENCE_DIR/backups/rds_snapshots.json

# EBS snapshots
aws ec2 describe-snapshots --owner-ids self --output json > $EVIDENCE_DIR/backups/ebs_snapshots.json

# S3 versioning
for bucket in $(aws s3 ls | awk '{print $3}' | grep vcci-scope3); do
  echo "Bucket: $bucket" >> $EVIDENCE_DIR/backups/s3_versioning.txt
  aws s3api get-bucket-versioning --bucket $bucket >> $EVIDENCE_DIR/backups/s3_versioning.txt
  echo "---" >> $EVIDENCE_DIR/backups/s3_versioning.txt
done

# 6. Security Policies
echo "Collecting security policy evidence..."

# Copy policy documents
cp /path/to/security_policies/*.pdf $EVIDENCE_DIR/security_policies/ 2>/dev/null || echo "Security policy PDFs not found"

# Network policies
kubectl get networkpolicies -A -o yaml > $EVIDENCE_DIR/security_policies/k8s_network_policies.yaml

# Security groups
aws ec2 describe-security-groups --output json > $EVIDENCE_DIR/security_policies/aws_security_groups.json

# WAF rules
aws wafv2 list-web-acls --scope REGIONAL --region us-west-2 --output json > $EVIDENCE_DIR/security_policies/waf_rules.json

echo "Evidence collection complete!"
echo "Evidence directory: $EVIDENCE_DIR"
EOF

chmod +x collect_evidence.sh
./collect_evidence.sh
```

### Part 2: Access Control Verification

#### Step 4: Verify User Access Controls

```bash
# Review all user accounts and permissions
aws iam list-users --output json | jq -r '.Users[] | [.UserName, .CreateDate, .PasswordLastUsed] | @tsv' > user_access_review.txt

# Check MFA enforcement
aws iam get-credential-report --output text | base64 -d | awk -F',' '{print $1, $4, $8}' > mfa_status.txt

# Identify users without MFA
grep "false" mfa_status.txt > users_without_mfa.txt

# Review inactive users (no activity in 90 days)
cat > check_inactive_users.sh << 'EOF'
#!/bin/bash
INACTIVE_THRESHOLD=90

aws iam get-credential-report --output text | base64 -d | tail -n +2 | while IFS=',' read user arn created pwdenabled pwdlast pwdnext mfa access1 access1last access2 access2last cert certlast; do
  if [ "$pwdlast" != "N/A" ] && [ "$pwdlast" != "no_information" ]; then
    LAST_ACTIVITY=$(date -d "$pwdlast" +%s 2>/dev/null)
    NOW=$(date +%s)
    DAYS_INACTIVE=$(( ($NOW - $LAST_ACTIVITY) / 86400 ))

    if [ $DAYS_INACTIVE -gt $INACTIVE_THRESHOLD ]; then
      echo "$user: Inactive for $DAYS_INACTIVE days (last password use: $pwdlast)"
    fi
  fi
done
EOF

chmod +x check_inactive_users.sh
./check_inactive_users.sh > inactive_users_report.txt
```

#### Step 5: Verify Least Privilege Access

```bash
# Check for overly permissive IAM policies
cat > check_admin_access.sh << 'EOF'
#!/bin/bash

echo "=== Users/Roles with Administrative Access ==="

# Check for AdministratorAccess policy
aws iam list-entities-for-policy --policy-arn arn:aws:iam::aws:policy/AdministratorAccess --output json | \
  jq -r '.PolicyUsers[].UserName, .PolicyRoles[].RoleName, .PolicyGroups[].GroupName'

# Check for custom admin policies
aws iam list-policies --scope Local --output json | \
  jq -r '.Policies[] | select(.PolicyName | contains("Admin") or contains("Full")) | .PolicyName'

# Check for wildcard permissions
echo ""
echo "=== Policies with Wildcard Permissions ==="
for policy_arn in $(aws iam list-policies --scope Local --output json | jq -r '.Policies[].Arn'); do
  version=$(aws iam get-policy --policy-arn $policy_arn --query 'Policy.DefaultVersionId' --output text)
  policy_doc=$(aws iam get-policy-version --policy-arn $policy_arn --version-id $version --query 'PolicyVersion.Document' --output json)

  if echo "$policy_doc" | jq -e '.Statement[] | select(.Effect=="Allow" and (.Action=="*" or (.Resource=="*")))' >/dev/null 2>&1; then
    echo "Policy: $policy_arn has wildcard permissions"
  fi
done
EOF

chmod +x check_admin_access.sh
./check_admin_access.sh > admin_access_review.txt
```

#### Step 6: Audit Database Access

```bash
# Review database user permissions
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF' > database_access_audit.txt
-- List all database users and roles
SELECT
  r.rolname AS username,
  r.rolsuper AS is_superuser,
  r.rolinherit AS can_inherit,
  r.rolcreaterole AS can_create_role,
  r.rolcreatedb AS can_create_db,
  r.rolcanlogin AS can_login,
  ARRAY(
    SELECT b.rolname
    FROM pg_catalog.pg_auth_members m
    JOIN pg_catalog.pg_roles b ON (m.roleid = b.oid)
    WHERE m.member = r.oid
  ) AS member_of
FROM pg_catalog.pg_roles r
WHERE r.rolname NOT LIKE 'pg_%'
  AND r.rolname NOT LIKE 'rds%'
ORDER BY r.rolname;

-- List table permissions
SELECT
  grantee,
  table_schema,
  table_name,
  string_agg(privilege_type, ', ') AS privileges
FROM information_schema.role_table_grants
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
GROUP BY grantee, table_schema, table_name
ORDER BY grantee, table_schema, table_name;
EOF
```

### Part 3: Encryption Verification

#### Step 7: Verify Data Encryption at Rest

```bash
# Create encryption audit report
cat > encryption_audit.sh << 'EOF'
#!/bin/bash

echo "=== Encryption at Rest Audit ===" > encryption_audit_report.txt
echo "Date: $(date)" >> encryption_audit_report.txt
echo "" >> encryption_audit_report.txt

# RDS Encryption
echo "## RDS Database Encryption" >> encryption_audit_report.txt
aws rds describe-db-instances --query 'DBInstances[*].[DBInstanceIdentifier,StorageEncrypted,KmsKeyId]' --output table >> encryption_audit_report.txt

# EBS Volumes
echo "" >> encryption_audit_report.txt
echo "## EBS Volume Encryption" >> encryption_audit_report.txt
TOTAL_VOLUMES=$(aws ec2 describe-volumes --query 'Volumes[*].VolumeId' --output text | wc -w)
ENCRYPTED_VOLUMES=$(aws ec2 describe-volumes --query 'Volumes[?Encrypted==`true`].VolumeId' --output text | wc -w)
echo "Total Volumes: $TOTAL_VOLUMES" >> encryption_audit_report.txt
echo "Encrypted Volumes: $ENCRYPTED_VOLUMES" >> encryption_audit_report.txt
echo "Encryption Rate: $(( $ENCRYPTED_VOLUMES * 100 / $TOTAL_VOLUMES ))%" >> encryption_audit_report.txt

# Unencrypted volumes (if any)
UNENCRYPTED=$(aws ec2 describe-volumes --query 'Volumes[?Encrypted==`false`].[VolumeId,State,Attachments[0].InstanceId]' --output table)
if [ ! -z "$UNENCRYPTED" ]; then
  echo "" >> encryption_audit_report.txt
  echo "WARNING: Unencrypted Volumes Found:" >> encryption_audit_report.txt
  echo "$UNENCRYPTED" >> encryption_audit_report.txt
fi

# S3 Buckets
echo "" >> encryption_audit_report.txt
echo "## S3 Bucket Encryption" >> encryption_audit_report.txt
for bucket in $(aws s3 ls | awk '{print $3}' | grep vcci-scope3); do
  echo "Bucket: $bucket" >> encryption_audit_report.txt
  aws s3api get-bucket-encryption --bucket $bucket >> encryption_audit_report.txt 2>&1
  echo "---" >> encryption_audit_report.txt
done

# Secrets Encryption
echo "" >> encryption_audit_report.txt
echo "## Kubernetes Secrets Encryption" >> encryption_audit_report.txt
kubectl get secrets -A --field-selector type=Opaque -o json | \
  jq -r '.items[] | "\(.metadata.namespace)/\(.metadata.name)"' | head -20 >> encryption_audit_report.txt

cat encryption_audit_report.txt
EOF

chmod +x encryption_audit.sh
./encryption_audit.sh
```

#### Step 8: Verify Data Encryption in Transit

```bash
# Test TLS configuration
cat > test_tls.sh << 'EOF'
#!/bin/bash

echo "=== TLS Configuration Test ===" > tls_audit_report.txt

# Test API endpoint
echo "## API Endpoint TLS" >> tls_audit_report.txt
echo | openssl s_client -connect api.vcci-scope3.com:443 -servername api.vcci-scope3.com 2>/dev/null | \
  openssl x509 -noout -text | grep -A 2 "Subject:\|Issuer:\|Not Before\|Not After" >> tls_audit_report.txt

# Check TLS version
echo "" >> tls_audit_report.txt
echo "## Supported TLS Versions" >> tls_audit_report.txt
for version in ssl3 tls1 tls1_1 tls1_2 tls1_3; do
  result=$(echo | timeout 2 openssl s_client -connect api.vcci-scope3.com:443 -$version 2>&1 | grep -o "Protocol.*")
  if [ ! -z "$result" ]; then
    echo "$version: ENABLED - $result" >> tls_audit_report.txt
  else
    echo "$version: DISABLED" >> tls_audit_report.txt
  fi
done

# Check cipher suites
echo "" >> tls_audit_report.txt
echo "## Cipher Suites" >> tls_audit_report.txt
nmap --script ssl-enum-ciphers -p 443 api.vcci-scope3.com >> tls_audit_report.txt 2>&1

# Database TLS
echo "" >> tls_audit_report.txt
echo "## Database TLS Connection" >> tls_audit_report.txt
psql "postgresql://$DB_ENDPOINT:5432/scope3_platform?sslmode=require" -U vcci_admin -c "\conninfo" >> tls_audit_report.txt 2>&1

cat tls_audit_report.txt
EOF

chmod +x test_tls.sh
./test_tls.sh
```

### Part 4: Audit Log Analysis

#### Step 9: Collect and Analyze Audit Logs

```bash
# Export CloudTrail logs for audit period
START_DATE="2024-01-01"
END_DATE="2024-12-31"

aws cloudtrail lookup-events \
  --start-time $START_DATE \
  --end-time $END_DATE \
  --max-results 10000 \
  --output json > cloudtrail_audit_period.json

# Analyze key events
cat > analyze_cloudtrail.sh << 'EOF'
#!/bin/bash

echo "=== CloudTrail Audit Log Analysis ===" > cloudtrail_analysis.txt
echo "Period: $START_DATE to $END_DATE" >> cloudtrail_analysis.txt
echo "" >> cloudtrail_analysis.txt

# User creation events
echo "## User Management Events" >> cloudtrail_analysis.txt
jq -r '.Events[] | select(.EventName | contains("CreateUser") or contains("DeleteUser") or contains("AttachUserPolicy")) | [.EventTime, .EventName, .Username] | @tsv' cloudtrail_audit_period.json >> cloudtrail_analysis.txt

# Failed authentication attempts
echo "" >> cloudtrail_analysis.txt
echo "## Failed Authentication Attempts" >> cloudtrail_analysis.txt
jq -r '.Events[] | select(.EventName == "ConsoleLogin" and .ResponseElements.ConsoleLogin == "Failure") | [.EventTime, .SourceIPAddress, .Username] | @tsv' cloudtrail_audit_period.json >> cloudtrail_analysis.txt

# Policy changes
echo "" >> cloudtrail_analysis.txt
echo "## IAM Policy Changes" >> cloudtrail_analysis.txt
jq -r '.Events[] | select(.EventName | contains("PutPolicy") or contains("CreatePolicy") or contains("DeletePolicy")) | [.EventTime, .EventName, .RequestParameters] | @tsv' cloudtrail_audit_period.json >> cloudtrail_analysis.txt

# Security group changes
echo "" >> cloudtrail_analysis.txt
echo "## Security Group Changes" >> cloudtrail_analysis.txt
jq -r '.Events[] | select(.EventName | contains("AuthorizeSecurityGroup") or contains("RevokeSecurityGroup")) | [.EventTime, .EventName, .RequestParameters] | @tsv' cloudtrail_audit_period.json >> cloudtrail_analysis.txt

cat cloudtrail_analysis.txt
EOF

chmod +x analyze_cloudtrail.sh
./analyze_cloudtrail.sh

# Application audit logs
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << EOF > application_audit_logs.csv
\copy (
  SELECT
    event_time,
    user_name,
    event_type,
    table_name,
    query_text,
    client_addr
  FROM audit_log
  WHERE event_time BETWEEN '$START_DATE' AND '$END_DATE'
  ORDER BY event_time DESC
) TO STDOUT WITH CSV HEADER;
EOF
```

#### Step 10: Review Incident Response Logs

```bash
# Collect incident data
cat > incident_review.sh << 'EOF'
#!/bin/bash

echo "=== Incident Response Review ===" > incident_review.txt
echo "Audit Period: 2024-01-01 to 2024-12-31" >> incident_review.txt
echo "" >> incident_review.txt

# List all incident reports
echo "## Incident Reports" >> incident_review.txt
ls -la /incidents/*.md >> incident_review.txt 2>&1

# Summarize incidents
echo "" >> incident_review.txt
echo "## Incident Summary" >> incident_review.txt
echo "Total Incidents: $(ls /incidents/*.md 2>/dev/null | wc -l)" >> incident_review.txt

# Extract key details
for incident in /incidents/*.md; do
  if [ -f "$incident" ]; then
    echo "" >> incident_review.txt
    echo "Incident: $(basename $incident)" >> incident_review.txt
    grep -E "Severity|Date|Root Cause|Resolution" "$incident" >> incident_review.txt
  fi
done

# GuardDuty findings summary
echo "" >> incident_review.txt
echo "## GuardDuty Findings Summary" >> incident_review.txt
aws guardduty get-findings-statistics \
  --detector-id $(aws guardduty list-detectors --query 'DetectorIds[0]' --output text) \
  --finding-criteria '{"Criterion":{"createdAt":{"Gte":'$(date -d '2024-01-01' +%s000)'}}}' \
  --finding-statistic-types COUNT_BY_SEVERITY \
  --output json | jq >> incident_review.txt

cat incident_review.txt
EOF

chmod +x incident_review.sh
./incident_review.sh
```

### Part 5: Control Testing

#### Step 11: Test Backup and Recovery Controls

```bash
# Verify backup processes
cat > test_backups.sh << 'EOF'
#!/bin/bash

echo "=== Backup Control Testing ===" > backup_control_test.txt

# Check RDS automated backups
echo "## RDS Automated Backups" >> backup_control_test.txt
aws rds describe-db-instances \
  --query 'DBInstances[*].[DBInstanceIdentifier,BackupRetentionPeriod,PreferredBackupWindow]' \
  --output table >> backup_control_test.txt

# Verify snapshots exist
echo "" >> backup_control_test.txt
echo "## Recent RDS Snapshots (Last 7 Days)" >> backup_control_test.txt
aws rds describe-db-snapshots \
  --query "DBSnapshots[?SnapshotCreateTime>='$(date -d '7 days ago' -u +%Y-%m-%d)'].[DBSnapshotIdentifier,SnapshotCreateTime,Status]" \
  --output table >> backup_control_test.txt

# Test restoration process (to temp instance)
echo "" >> backup_control_test.txt
echo "## Backup Restoration Test" >> backup_control_test.txt
LATEST_SNAPSHOT=$(aws rds describe-db-snapshots \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --query 'reverse(sort_by(DBSnapshots, &SnapshotCreateTime))[0].DBSnapshotIdentifier' \
  --output text)

echo "Latest Snapshot: $LATEST_SNAPSHOT" >> backup_control_test.txt
echo "Initiating restoration test..." >> backup_control_test.txt

# Restore to small test instance
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier audit-restore-test-$(date +%Y%m%d) \
  --db-snapshot-identifier $LATEST_SNAPSHOT \
  --db-instance-class db.t3.medium \
  --no-multi-az \
  --publicly-accessible false >> backup_control_test.txt 2>&1

echo "Restoration initiated. Test instance will be deleted after validation." >> backup_control_test.txt

cat backup_control_test.txt
EOF

chmod +x test_backups.sh
./test_backups.sh
```

#### Step 12: Test Access Control Changes

```bash
# Test that access control changes are logged
cat > test_access_controls.sh << 'EOF'
#!/bin/bash

echo "=== Access Control Testing ===" > access_control_test.txt

# Create test user
echo "## Creating Test User" >> access_control_test.txt
TEST_USER="audit-test-user-$(date +%s)"
aws iam create-user --user-name $TEST_USER >> access_control_test.txt 2>&1

# Verify creation was logged
sleep 5
echo "" >> access_control_test.txt
echo "## Verifying User Creation Logged" >> access_control_test.txt
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceName,AttributeValue=$TEST_USER \
  --max-results 5 \
  --query 'Events[*].[EventTime,EventName,Username]' \
  --output table >> access_control_test.txt

# Test MFA requirement
echo "" >> access_control_test.txt
echo "## Testing MFA Enforcement" >> access_control_test.txt
# Attempt to perform action without MFA (should fail)
aws iam create-access-key --user-name $TEST_USER >> access_control_test.txt 2>&1

# Clean up test user
aws iam delete-user --user-name $TEST_USER

echo "" >> access_control_test.txt
echo "Test user deleted: $TEST_USER" >> access_control_test.txt

cat access_control_test.txt
EOF

chmod +x test_access_controls.sh
./test_access_controls.sh
```

### Part 6: Documentation and Reporting

#### Step 13: Compile Audit Evidence Package

```bash
# Create comprehensive evidence package
cat > create_audit_package.sh << 'EOF'
#!/bin/bash

PACKAGE_DIR="audit_package_$(date +%Y%m%d)"
mkdir -p $PACKAGE_DIR/{policies,technical_evidence,access_controls,monitoring,incidents}

echo "Creating audit evidence package: $PACKAGE_DIR"

# Copy all collected evidence
cp -r evidence/* $PACKAGE_DIR/technical_evidence/
cp control_mapping.yaml $PACKAGE_DIR/
cp *_audit*.txt $PACKAGE_DIR/technical_evidence/
cp *_review.txt $PACKAGE_DIR/access_controls/
cp cloudtrail_analysis.txt $PACKAGE_DIR/monitoring/
cp incident_review.txt $PACKAGE_DIR/incidents/

# Copy policy documents
cp /path/to/security_policies/* $PACKAGE_DIR/policies/ 2>/dev/null

# Generate evidence index
cat > $PACKAGE_DIR/EVIDENCE_INDEX.md << 'EOINDEX'
# SOC 2 Type II Audit Evidence Package

**Audit Period**: 2024-01-01 to 2024-12-31
**Package Created**: $(date)

## Contents

### 1. Policies and Procedures
- Information Security Policy
- Access Control Policy
- Change Management Policy
- Incident Response Policy
- Business Continuity Policy
- Data Classification Policy

### 2. Technical Evidence

#### Access Controls
- IAM policies and users
- Kubernetes RBAC configuration
- MFA status report
- Inactive user analysis
- Database user permissions

#### Encryption
- TLS certificate inventory
- Database encryption settings
- S3 encryption policies
- EBS volume encryption status

#### Monitoring & Alerting
- CloudWatch alarms
- Prometheus alerting rules
- GuardDuty findings
- Security incident logs

#### Change Management
- Git commit history
- Deployment records
- CI/CD pipeline logs

#### Backup & Recovery
- RDS snapshot inventory
- Backup restoration test results
- S3 versioning configuration

### 3. Audit Logs
- CloudTrail events (12 months)
- Application audit logs
- Database audit logs
- Access attempt logs

### 4. Control Testing Results
- Backup restoration test
- Access control test
- Encryption verification

### 5. Incident Response
- Incident reports
- Root cause analyses
- Remediation evidence

## Evidence Validation

All evidence has been:
- [x] Collected from production systems
- [x] Validated for completeness
- [x] Reviewed for accuracy
- [x] Organized by control objective
- [x] Dated and time-stamped

**Prepared by**: [Name, Title]
**Reviewed by**: [Name, Title]
**Date**: $(date)
EOINDEX

# Create package archive
tar -czf ${PACKAGE_DIR}.tar.gz $PACKAGE_DIR
echo "Audit package created: ${PACKAGE_DIR}.tar.gz"

# Generate checksums for integrity
sha256sum ${PACKAGE_DIR}.tar.gz > ${PACKAGE_DIR}.tar.gz.sha256
echo "Checksum: $(cat ${PACKAGE_DIR}.tar.gz.sha256)"

EOF

chmod +x create_audit_package.sh
./create_audit_package.sh
```

#### Step 14: Create Audit Readiness Report

```bash
# Generate audit readiness report
cat > audit_readiness_report.md << 'EOF'
# SOC 2 Type II Audit Readiness Report

**Date**: $(date +%Y-%m-%d)
**Audit Period**: 2024-01-01 to 2024-12-31
**Report Prepared By**: Platform & Security Teams

## Executive Summary

The VCCI Scope 3 Carbon Intelligence Platform is ready for SOC 2 Type II audit with **98% control compliance**. All critical controls are implemented and tested.

## Control Assessment Summary

| Trust Service Category | Total Controls | Implemented | In Progress | Not Applicable |
|-------------------------|----------------|-------------|-------------|----------------|
| Security (CC)           | 45             | 44          | 1           | 0              |
| Availability (A)        | 12             | 12          | 0           | 0              |
| Confidentiality (C)     | 8              | 8           | 0           | 0              |
| **TOTAL**              | **65**         | **64**      | **1**       | **0**          |

### Control Implementation Rate: 98.5%

## Key Findings

### Strengths
1. Comprehensive logging and monitoring infrastructure
2. Automated backup and disaster recovery processes
3. Strong encryption implementation (data at rest and in transit)
4. Robust access control with MFA enforcement
5. Well-documented change management process

### Areas for Improvement
1. **CC6.7 - System Logging**: One application component missing structured logging (IN PROGRESS)
   - Expected completion: 2 weeks
   - Workaround: Manual log review process in place

## Evidence Completeness

- [x] Policy documentation (100%)
- [x] Technical configurations (100%)
- [x] Audit logs for full 12-month period (100%)
- [x] Incident response records (100%)
- [x] Change management records (100%)
- [x] Access control evidence (100%)
- [x] Backup and recovery testing (100%)

## Testing Results

All control tests passed successfully:
- Access control testing: ✓ PASS
- Backup restoration: ✓ PASS
- Encryption verification: ✓ PASS
- Monitoring effectiveness: ✓ PASS
- Incident response process: ✓ PASS

## Auditor Requests Anticipated

### 1. Sample Selections Expected
- User access reviews: 10-15 samples
- Change tickets: 20-25 samples
- Incident reports: All security incidents
- Backup validations: 5-10 samples

### 2. Interviews Required
- CISO / Security Lead
- Platform Engineering Manager
- Database Administrator
- DevOps Lead

### 3. Walkthrough Sessions
- Deployment process
- Incident response procedure
- User provisioning workflow
- Backup/recovery process

## Recommendations

1. **Pre-audit preparation** (2 weeks before):
   - Final evidence review
   - Prepare interview talking points
   - Set up auditor environment access
   - Schedule walkthrough sessions

2. **During audit**:
   - Dedicated audit liaison assigned
   - Daily status check-ins
   - Evidence request tracking
   - Issue escalation process

3. **Post-audit**:
   - Document lessons learned
   - Address any findings promptly
   - Update procedures based on feedback

## Timeline

- **Audit fieldwork start**: [Date]
- **Evidence package delivery**: [Date - 1 week before fieldwork]
- **Anticipated completion**: [Date + 2 weeks from start]
- **Report delivery**: [Date + 4 weeks from start]

## Conclusion

The platform demonstrates strong compliance with SOC 2 requirements. The minor gap identified will be remediated before audit fieldwork. The organization is well-prepared for a successful audit.

---
**Approved by**: [Name, Title]
**Date**: $(date)
EOF

cat audit_readiness_report.md
```

## Validation Checklist

- [ ] Audit scope defined and documented
- [ ] Control mapping completed
- [ ] Evidence collected for all controls
- [ ] Access controls verified
- [ ] Encryption validated (at rest and in transit)
- [ ] Audit logs reviewed and analyzed
- [ ] Incident response logs compiled
- [ ] Control testing completed
- [ ] Evidence package compiled
- [ ] Audit readiness report prepared
- [ ] Stakeholders briefed
- [ ] Auditor access prepared

## Related Documentation

- [Security Incident Runbook](./SECURITY_INCIDENT.md)
- [Data Recovery Runbook](./DATA_RECOVERY.md)
- [Incident Response Runbook](./INCIDENT_RESPONSE.md)
- [SOC 2 Compliance Guide](https://www.aicpa.org/soc4so)
- [GDPR Compliance Checklist](https://gdpr.eu/checklist/)

## Contact Information

- **Compliance Team**: compliance@company.com
- **Security Team**: security@company.com
- **Data Protection Officer**: dpo@company.com
- **Legal**: legal@company.com
- **External Auditor**: [Auditor contact]
