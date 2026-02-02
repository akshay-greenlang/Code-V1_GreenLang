# Security Incident Response Runbook

**Scenario**: Respond to security breaches, unauthorized access, data exfiltration, malware, or suspicious activity with immediate containment, investigation, and remediation.

**Severity**: P0 (Active breach) / P1 (Confirmed compromise) / P2 (Suspicious activity)

**RTO/RPO**: Immediate response required / Evidence preservation critical

**Owner**: Security Team / Incident Response Team

## Prerequisites

- AWS Console access with security permissions
- kubectl access to EKS cluster
- Access to security tools (GuardDuty, CloudTrail, Security Hub)
- Incident response contact list
- Legal/compliance team contact information
- Communication channels established

## Detection

### Security Incident Indicators

1. **Automated Alerts**:
   - AWS GuardDuty findings (Critical/High)
   - CloudWatch security alarms
   - WAF rule violations
   - Failed authentication attempts > 50/hour
   - Unusual API activity patterns

2. **System Anomalies**:
   - Unexpected outbound network connections
   - Privilege escalation attempts
   - Modified system files or configurations
   - Cryptocurrency mining activity
   - Data exfiltration to unknown destinations

3. **User Reports**:
   - Unauthorized access to accounts
   - Data visible to wrong users
   - Suspicious emails or phishing attempts
   - Unexpected system behavior

### Initial Security Check

```bash
# Check GuardDuty findings
aws guardduty list-findings \
  --detector-id $(aws guardduty list-detectors --query 'DetectorIds[0]' --output text) \
  --finding-criteria '{"Criterion":{"severity":{"Gte":7}}}' \
  --query 'FindingIds' \
  --output json

# Get finding details
aws guardduty get-findings \
  --detector-id $(aws guardduty list-detectors --query 'DetectorIds[0]' --output text) \
  --finding-ids <finding-id> \
  --query 'Findings[0].{Title:Title,Severity:Severity,Type:Type,Description:Description}' \
  --output table

# Check recent CloudTrail events
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=ConsoleLogin \
  --max-results 50 \
  --query 'Events[*].[EventTime,Username,EventName,SourceIPAddress]' \
  --output table

# Check failed kubectl auth attempts
kubectl logs -n kube-system deployment/aws-auth --tail=100 | grep -i "denied\|failed\|unauthorized"
```

## Step-by-Step Procedure

### Phase 1: Immediate Containment (First 15 Minutes)

#### Step 1: Declare Security Incident

```bash
# Activate incident response team
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "🚨 SECURITY INCIDENT DECLARED - P0",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*SECURITY INCIDENT*\n*Severity*: P0 - Active Breach\n*Time*: '$(date -u)'\n*IR Commander*: @oncall\n*Status*: CONTAINMENT PHASE"
        }
      }
    ]
  }'

# Create incident ticket
INCIDENT_ID="SEC-$(date +%Y%m%d-%H%M%S)"
echo "Incident ID: $INCIDENT_ID"

# Start incident timeline
cat > /tmp/incident_timeline_$INCIDENT_ID.txt << EOF
Security Incident Timeline - $INCIDENT_ID
==========================================
$(date -u): Incident detected
$(date -u): Incident declared
$(date -u): Containment initiated

EOF
```

#### Step 2: Isolate Affected Resources

```bash
# If compromised pod identified, isolate immediately
COMPROMISED_POD="api-gateway-7d9f8b6c5d-abc12"

# Label pod for quarantine
kubectl label pod $COMPROMISED_POD -n vcci-scope3 security-status=quarantined

# Apply network policy to isolate pod
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: quarantine-policy
  namespace: vcci-scope3
spec:
  podSelector:
    matchLabels:
      security-status: quarantined
  policyTypes:
  - Ingress
  - Egress
  ingress: []  # Deny all ingress
  egress: []   # Deny all egress
EOF

# If EC2 instance compromised
INSTANCE_ID="i-1234567890abcdef0"

# Isolate instance (remove from security groups, keep forensics)
aws ec2 modify-instance-attribute \
  --instance-id $INSTANCE_ID \
  --groups sg-forensics-isolation

# Prevent instance termination
aws ec2 modify-instance-attribute \
  --instance-id $INSTANCE_ID \
  --disable-api-termination

# If IAM user compromised, disable access immediately
COMPROMISED_USER="john.doe@company.com"

# Disable console access
aws iam delete-login-profile --user-name $COMPROMISED_USER 2>/dev/null || true

# Deactivate access keys
aws iam list-access-keys --user-name $COMPROMISED_USER --query 'AccessKeyMetadata[].AccessKeyId' --output text | \
while read key_id; do
  aws iam update-access-key --user-name $COMPROMISED_USER --access-key-id $key_id --status Inactive
  echo "$(date -u): Deactivated access key $key_id" >> /tmp/incident_timeline_$INCIDENT_ID.txt
done
```

#### Step 3: Preserve Evidence

```bash
# Snapshot compromised RDS database
aws rds create-db-snapshot \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --db-snapshot-identifier forensics-$INCIDENT_ID-$(date +%Y%m%d-%H%M%S) \
  --tags Key=IncidentID,Value=$INCIDENT_ID Key=Purpose,Value=Forensics

# Snapshot EC2 volumes
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].BlockDeviceMappings[*].Ebs.VolumeId' \
  --output text | \
while read volume_id; do
  aws ec2 create-snapshot \
    --volume-id $volume_id \
    --description "Forensics snapshot for $INCIDENT_ID" \
    --tag-specifications "ResourceType=snapshot,Tags=[{Key=IncidentID,Value=$INCIDENT_ID},{Key=Purpose,Value=Forensics}]"
done

# Capture pod logs before deletion
kubectl logs $COMPROMISED_POD -n vcci-scope3 --all-containers=true > /tmp/forensics_pod_logs_$INCIDENT_ID.txt

# Capture pod manifest
kubectl get pod $COMPROMISED_POD -n vcci-scope3 -o yaml > /tmp/forensics_pod_manifest_$INCIDENT_ID.yaml

# Export to S3 for long-term retention
aws s3 cp /tmp/forensics_pod_logs_$INCIDENT_ID.txt \
  s3://vcci-security-forensics/$INCIDENT_ID/ \
  --server-side-encryption AES256

aws s3 cp /tmp/forensics_pod_manifest_$INCIDENT_ID.yaml \
  s3://vcci-security-forensics/$INCIDENT_ID/ \
  --server-side-encryption AES256

# Enable CloudTrail log file validation if not already enabled
aws cloudtrail update-trail \
  --name vcci-scope3-trail \
  --enable-log-file-validation

# Export CloudTrail logs for incident timeframe
aws cloudtrail lookup-events \
  --start-time "2024-01-15T00:00:00Z" \
  --end-time "2024-01-15T23:59:59Z" \
  --max-results 10000 > /tmp/forensics_cloudtrail_$INCIDENT_ID.json
```

#### Step 4: Assess Scope of Compromise

```bash
# Check which resources the compromised user/role accessed
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=Username,AttributeValue=$COMPROMISED_USER \
  --start-time "$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S)" \
  --query 'Events[*].[EventTime,EventName,Resources[0].ResourceName]' \
  --output table

# Check API calls from compromised pod's service account
kubectl get serviceaccount -n vcci-scope3 -o yaml | grep -A 10 "name: $POD_SERVICE_ACCOUNT"

# Review recent database queries (if audit logging enabled)
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  event_time,
  user_name,
  database_name,
  client_addr,
  query_text
FROM audit_log
WHERE event_time > NOW() - INTERVAL '24 hours'
  AND (
    query_text ILIKE '%DROP%'
    OR query_text ILIKE '%DELETE%'
    OR query_text ILIKE '%GRANT%'
    OR query_text ILIKE '%CREATE USER%'
  )
ORDER BY event_time DESC
LIMIT 100;
EOF

# Check for data exfiltration (unusual data transfer)
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name NetworkOut \
  --dimensions Name=InstanceId,Value=$INSTANCE_ID \
  --start-time $(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 3600 \
  --statistics Sum \
  --output table

# Check S3 access logs for unusual downloads
aws s3api get-bucket-logging --bucket vcci-scope3-data-prod
# Review logs in logging bucket
aws s3 ls s3://vcci-logs/s3-access/ --recursive | tail -50
```

### Phase 2: Investigation (30 Minutes - 2 Hours)

#### Step 5: Analyze Attack Vector

```bash
# Check WAF logs for attack patterns
aws wafv2 list-web-acls \
  --scope REGIONAL \
  --region us-west-2

# Get sampled requests
aws wafv2 get-sampled-requests \
  --web-acl-arn <web-acl-arn> \
  --rule-metric-name <rule-name> \
  --scope REGIONAL \
  --time-window StartTime=$(date -u -d '6 hours ago' +%s),EndTime=$(date -u +%s) \
  --max-items 500

# Check application logs for suspicious patterns
kubectl logs -n vcci-scope3 -l app=api-gateway --since=24h | \
  grep -E "SQL injection|../|<script>|exec\(|eval\(|base64_decode" > /tmp/suspicious_requests_$INCIDENT_ID.txt

# Analyze authentication patterns
kubectl logs -n vcci-scope3 -l app=api-gateway --since=24h | \
  grep "authentication" | \
  awk '{print $1, $2, $3}' | \
  sort | uniq -c | sort -rn

# Check for privilege escalation attempts
kubectl logs -n kube-system --since=24h | \
  grep -i "forbidden\|denied\|escalate\|privilege"

# Review GuardDuty findings for IOCs
aws guardduty get-findings \
  --detector-id $(aws guardduty list-detectors --query 'DetectorIds[0]' --output text) \
  --finding-ids $(aws guardduty list-findings \
    --detector-id $(aws guardduty list-detectors --query 'DetectorIds[0]' --output text) \
    --query 'FindingIds[0]' --output text) \
  --query 'Findings[0].Service.Action' \
  --output json
```

#### Step 6: Identify Compromised Data

```bash
# Check which tables were accessed
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  schemaname,
  tablename,
  COUNT(*) as access_count,
  MAX(event_time) as last_access
FROM audit_log
WHERE event_time > NOW() - INTERVAL '24 hours'
  AND user_name = 'compromised_user'
GROUP BY schemaname, tablename
ORDER BY access_count DESC;
EOF

# Check S3 object access logs
aws s3api get-object \
  --bucket vcci-logs \
  --key s3-access/$(date +%Y-%m-%d)-*.log \
  /tmp/s3_access_logs.txt

grep "REST.GET.OBJECT" /tmp/s3_access_logs.txt | \
  awk '{print $8, $11}' | \
  sort | uniq -c | sort -rn > /tmp/s3_accessed_objects_$INCIDENT_ID.txt

# List potentially exfiltrated data
cat /tmp/s3_accessed_objects_$INCIDENT_ID.txt | head -50

# Check for database dumps
kubectl logs -n vcci-scope3 --since=24h | grep -i "pg_dump\|mysqldump\|export\|backup"
```

#### Step 7: Document Indicators of Compromise (IOCs)

```bash
# Create IOC list
cat > /tmp/ioc_list_$INCIDENT_ID.txt << EOF
Indicators of Compromise - $INCIDENT_ID
========================================
Date: $(date -u)

IP Addresses:
EOF

# Extract suspicious IPs from logs
kubectl logs -n vcci-scope3 -l app=api-gateway --since=24h | \
  grep -E "401|403|500" | \
  awk '{print $1}' | sort | uniq -c | sort -rn | head -20 >> /tmp/ioc_list_$INCIDENT_ID.txt

cat >> /tmp/ioc_list_$INCIDENT_ID.txt << EOF

Compromised Accounts:
- $COMPROMISED_USER

Affected Resources:
- Pod: $COMPROMISED_POD
- Instance: $INSTANCE_ID

Malicious Files (if any):
EOF

# Check for suspicious files in containers
kubectl exec -n vcci-scope3 $COMPROMISED_POD -- find / -type f -mtime -1 2>/dev/null | \
  grep -v "^/proc\|^/sys" >> /tmp/ioc_list_$INCIDENT_ID.txt

# Upload IOC list
aws s3 cp /tmp/ioc_list_$INCIDENT_ID.txt \
  s3://vcci-security-forensics/$INCIDENT_ID/
```

### Phase 3: Eradication (2-4 Hours)

#### Step 8: Remove Malicious Access

```bash
# Rotate all credentials that may have been compromised
# 1. Rotate database passwords
NEW_DB_PASSWORD=$(openssl rand -base64 32)

aws rds modify-db-instance \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --master-user-password "$NEW_DB_PASSWORD" \
  --apply-immediately

# Update Kubernetes secret
kubectl create secret generic database-credentials \
  --from-literal=DB_PASSWORD="$NEW_DB_PASSWORD" \
  -n vcci-scope3 \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Rotate API keys
kubectl delete secret api-keys -n vcci-scope3
# Regenerate via application admin panel or script

# 3. Rotate IAM role credentials
aws iam delete-access-key \
  --user-name $COMPROMISED_USER \
  --access-key-id $COMPROMISED_ACCESS_KEY

# Force session revocation
aws iam put-user-policy \
  --user-name $COMPROMISED_USER \
  --policy-name DenyAll \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*"
    }]
  }'

# 4. Rotate service account tokens
kubectl delete secret -n vcci-scope3 -l app=api-gateway
# New tokens auto-generated

# 5. Invalidate all user sessions
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli FLUSHDB
```

#### Step 9: Patch Vulnerabilities

```bash
# If vulnerability in container image
# Build new image with patch
docker build -t vcci-scope3-api-gateway:patched-$(date +%Y%m%d) .
docker push vcci-scope3-api-gateway:patched-$(date +%Y%m%d)

# Update deployment
kubectl set image deployment/api-gateway \
  api-gateway=vcci-scope3-api-gateway:patched-$(date +%Y%m%d) \
  -n vcci-scope3

# If OS-level vulnerability
# Update node group with patched AMI
aws eks update-nodegroup-version \
  --cluster-name vcci-scope3-prod \
  --nodegroup-name vcci-scope3-workers-prod \
  --force

# If application vulnerability, deploy code fix
git checkout -b security-patch-$INCIDENT_ID
# Apply fix
git commit -m "Security patch for $INCIDENT_ID"
git push origin security-patch-$INCIDENT_ID
# Deploy via CI/CD

# Update WAF rules to block exploit
aws wafv2 update-web-acl \
  --id <web-acl-id> \
  --scope REGIONAL \
  --region us-west-2 \
  --default-action Block={} \
  --rules file://updated-waf-rules.json
```

#### Step 10: Remove Backdoors/Persistence Mechanisms

```bash
# Check for unauthorized scheduled tasks
kubectl get cronjobs -A | grep -v "vcci-scope3\|kube-system"

# Check for unauthorized deployments
kubectl get deployments -A --sort-by=.metadata.creationTimestamp | tail -20

# Check for modified RBAC
kubectl get rolebindings,clusterrolebindings -A --sort-by=.metadata.creationTimestamp | tail -30

# Remove unauthorized resources
kubectl delete cronjob suspicious-cron -n default
kubectl delete deployment backdoor-deployment -n kube-system

# Check for unauthorized IAM roles/policies
aws iam list-roles --query 'Roles[?contains(RoleName, `backdoor`) || contains(RoleName, `temp`)]'

# Check for unauthorized security group rules
aws ec2 describe-security-groups \
  --filters Name=ip-permission.cidr,Values=0.0.0.0/0 \
  --query 'SecurityGroups[*].[GroupId,GroupName,IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]]' \
  --output table

# Review and remove suspicious rules
aws ec2 revoke-security-group-ingress \
  --group-id sg-suspicious \
  --ip-permissions IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges='[{CidrIp=0.0.0.0/0}]'
```

### Phase 4: Recovery (4-8 Hours)

#### Step 11: Restore from Clean Backups

```bash
# Restore database to point before compromise (if needed)
# See DATA_RECOVERY.md for detailed procedures

# Verify backup integrity
aws rds describe-db-snapshots \
  --db-instance-identifier vcci-scope3-prod-postgres \
  --snapshot-type manual \
  --query 'DBSnapshots[?SnapshotCreateTime<`2024-01-15T00:00:00Z`] | [0]' \
  --output table

# Restore from snapshot
CLEAN_SNAPSHOT="rds:vcci-scope3-prod-postgres-2024-01-14-06-00"

aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier vcci-scope3-prod-postgres-recovered \
  --db-snapshot-identifier $CLEAN_SNAPSHOT \
  --db-instance-class db.r6g.2xlarge \
  --multi-az

# Wait for restore
aws rds wait db-instance-available \
  --db-instance-identifier vcci-scope3-prod-postgres-recovered
```

#### Step 12: Redeploy Applications

```bash
# Delete potentially compromised pods
kubectl delete pods -n vcci-scope3 -l app=api-gateway
kubectl delete pods -n vcci-scope3 -l app=data-ingestion

# Deploy from known-good images
kubectl set image deployment/api-gateway \
  api-gateway=vcci-scope3-api-gateway:v1.2.3-verified \
  -n vcci-scope3

# Verify deployments
kubectl get pods -n vcci-scope3 -o wide
kubectl rollout status deployment/api-gateway -n vcci-scope3
```

#### Step 13: Implement Enhanced Security Controls

```bash
# Enable MFA enforcement
aws iam create-virtual-mfa-device --virtual-mfa-device-name vcci-mfa-$(date +%s)

# Update IAM policy to require MFA
cat > /tmp/mfa-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "DenyAllExceptListedIfNoMFA",
    "Effect": "Deny",
    "NotAction": [
      "iam:CreateVirtualMFADevice",
      "iam:EnableMFADevice",
      "iam:GetUser",
      "iam:ListMFADevices",
      "iam:ListVirtualMFADevices",
      "iam:ResyncMFADevice",
      "sts:GetSessionToken"
    ],
    "Resource": "*",
    "Condition": {
      "BoolIfExists": {"aws:MultiFactorAuthPresent": "false"}
    }
  }]
}
EOF

aws iam put-group-policy \
  --group-name VCCIDevelopers \
  --policy-name RequireMFA \
  --policy-document file:///tmp/mfa-policy.json

# Enable AWS Config rules
aws configservice put-config-rule --config-rule file://security-config-rules.json

# Enable VPC Flow Logs
aws ec2 create-flow-logs \
  --resource-type VPC \
  --resource-ids vpc-xxxxxxxx \
  --traffic-type ALL \
  --log-destination-type s3 \
  --log-destination arn:aws:s3:::vcci-vpc-flow-logs

# Implement pod security policies
kubectl apply -f - <<EOF
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  readOnlyRootFilesystem: true
EOF
```

### Phase 5: Notification and Compliance

#### Step 14: Determine Notification Requirements

```bash
# GDPR: Must notify within 72 hours of discovery if personal data breach
# Calculate time remaining
DISCOVERY_TIME="2024-01-15T10:00:00Z"
NOTIFICATION_DEADLINE=$(date -u -d "$DISCOVERY_TIME + 72 hours" +"%Y-%m-%d %H:%M:%S UTC")
echo "GDPR Notification Deadline: $NOTIFICATION_DEADLINE"

# Assess data breach impact
cat > /tmp/breach_assessment_$INCIDENT_ID.txt << EOF
Data Breach Assessment - $INCIDENT_ID
======================================

Personal Data Affected: [YES/NO]
Number of Individuals: [COUNT]
Data Categories: [List: names, emails, financial, health, etc.]
Geographic Scope: [Countries/regions affected]

Risk to Individuals:
- [ ] Identity theft
- [ ] Financial loss
- [ ] Reputational damage
- [ ] Discrimination
- [ ] Other: _______________

Likelihood of Risk: [LOW/MEDIUM/HIGH]
Severity of Risk: [LOW/MEDIUM/HIGH]

Notification Required: [YES/NO]
Reason: [Explanation]
EOF
```

#### Step 15: Notify Affected Parties

```bash
# Notify Data Protection Authority (if required by GDPR)
# Template notification letter
cat > /tmp/dpa_notification_$INCIDENT_ID.txt << EOF
To: [Data Protection Authority]
From: Data Protection Officer, [Company Name]
Date: $(date)
Subject: Personal Data Breach Notification - $INCIDENT_ID

Dear Sir/Madam,

In accordance with Article 33 of the GDPR, we are writing to notify you of a personal data breach.

1. Nature of the breach:
   [Description]

2. Categories and approximate number of data subjects affected:
   [Details]

3. Categories and approximate number of personal data records concerned:
   [Details]

4. Contact point for more information:
   DPO: dpo@company.com

5. Likely consequences of the breach:
   [Assessment]

6. Measures taken to address the breach:
   [List of actions]

7. Measures taken to mitigate possible adverse effects:
   [List of mitigations]

Sincerely,
Data Protection Officer
EOF

# Notify affected users (if required)
# Use templated email system
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF' > /tmp/affected_users.csv
\copy (
  SELECT DISTINCT email, first_name, last_name
  FROM users
  WHERE user_id IN (
    SELECT user_id FROM audit_log
    WHERE event_time BETWEEN '2024-01-15 00:00:00' AND '2024-01-15 23:59:59'
  )
) TO STDOUT WITH CSV HEADER;
EOF

# Send notification emails
# Use email service provider to send templated notification
```

#### Step 16: Report to Leadership and Board

```bash
# Create executive summary
cat > /tmp/executive_summary_$INCIDENT_ID.md << EOF
# Security Incident Executive Summary
**Incident ID**: $INCIDENT_ID
**Date**: $(date)
**Severity**: P0 - Data Breach
**Status**: RESOLVED

## Incident Overview
[Brief description of what happened]

## Impact
- **Data Compromised**: [Type and volume]
- **Users Affected**: [Number]
- **Financial Impact**: [Estimated cost]
- **Regulatory Impact**: [GDPR, SOC 2, etc.]

## Timeline
- **Detection**: [Time]
- **Containment**: [Time]
- **Eradication**: [Time]
- **Recovery**: [Time]
- **Total Duration**: [Hours]

## Root Cause
[Technical explanation]

## Actions Taken
1. [Action 1]
2. [Action 2]
3. [Action 3]

## Preventive Measures
1. [Measure 1]
2. [Measure 2]
3. [Measure 3]

## Lessons Learned
- [Lesson 1]
- [Lesson 2]

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]

Prepared by: Security Team
Contact: security@company.com
EOF

cat /tmp/executive_summary_$INCIDENT_ID.md
```

### Phase 6: Post-Incident Activities

#### Step 17: Conduct Post-Incident Review

```bash
# Schedule post-incident review meeting (within 5 business days)
# Agenda:
# 1. Timeline review
# 2. What went well
# 3. What could be improved
# 4. Action items

# Create post-incident report
cat > /tmp/post_incident_report_$INCIDENT_ID.md << 'EOF'
# Post-Incident Review Report

## Incident Details
- **Incident ID**: $INCIDENT_ID
- **Date Occurred**: [Date]
- **Date Resolved**: [Date]
- **Duration**: [Hours]

## Attendees
- [Name, Role]
- [Name, Role]

## Incident Summary
[Detailed description]

## Timeline
[Detailed timeline with all actions]

## What Went Well
1. [Item 1]
2. [Item 2]

## What Could Be Improved
1. [Item 1]
2. [Item 2]

## Root Cause Analysis
[5 Whys or other RCA method]

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [Action 1] | [Owner] | [Date] | Open |
| [Action 2] | [Owner] | [Date] | Open |

## Updated Procedures
[List of runbook/procedure updates needed]

## Training Needs
[List of training required]
EOF
```

#### Step 18: Implement Security Improvements

```bash
# Based on lessons learned, implement improvements

# Example: Enhanced monitoring
cat > /tmp/enhanced_monitoring.yaml << 'EOF'
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: security-alerts-enhanced
  namespace: monitoring
spec:
  groups:
  - name: security
    interval: 30s
    rules:
    - alert: SuspiciousAPIActivity
      expr: rate(http_requests_total{status=~"401|403"}[5m]) > 10
      labels:
        severity: warning
      annotations:
        summary: "High rate of unauthorized API access attempts"

    - alert: UnusualDataExfiltration
      expr: rate(s3_bytes_downloaded[5m]) > 1000000000  # > 1GB/5min
      labels:
        severity: critical
      annotations:
        summary: "Unusual volume of data download detected"

    - alert: PrivilegeEscalation
      expr: increase(k8s_audit_events{verb="escalate"}[5m]) > 0
      labels:
        severity: critical
      annotations:
        summary: "Privilege escalation attempt detected"
EOF

kubectl apply -f /tmp/enhanced_monitoring.yaml

# Implement automated response
cat > /tmp/automated_response.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-automation
  namespace: vcci-scope3
data:
  auto_block_suspicious_ip.sh: |
    #!/bin/bash
    # Automatically block IPs with > 100 failed auth attempts
    # Triggered by CloudWatch Events
    IP=$1
    aws ec2 authorize-security-group-ingress \
      --group-id sg-waf \
      --ip-permissions IpProtocol=-1,IpRanges="[{CidrIp=$IP/32,Description='Auto-blocked suspicious IP'}]"
EOF
```

## Validation Checklist

- [ ] All compromised credentials rotated
- [ ] Malicious access removed
- [ ] Vulnerabilities patched
- [ ] Backdoors/persistence mechanisms eliminated
- [ ] Enhanced security controls implemented
- [ ] Forensic evidence preserved
- [ ] Affected parties notified (if required)
- [ ] Regulatory notifications submitted (if required)
- [ ] Executive summary prepared
- [ ] Post-incident review conducted
- [ ] Action items assigned and tracked
- [ ] Documentation updated

## Troubleshooting

### Issue 1: Cannot Isolate Compromised Resource

**Resolution**: If technical isolation fails, escalate to physical/network isolation

### Issue 2: Forensic Evidence Corrupted

**Resolution**: Rely on immutable logs (CloudTrail, S3 access logs with object lock)

### Issue 3: Notification Deadline Approaching

**Resolution**: Submit preliminary notification, follow up with full details

## Related Documentation

- [Incident Response Runbook](./INCIDENT_RESPONSE.md)
- [Data Recovery Runbook](./DATA_RECOVERY.md)
- [Compliance Audit Runbook](./COMPLIANCE_AUDIT.md)
- [GDPR Breach Notification Guide](https://gdpr.eu/data-breach-notification/)
- [AWS Security Incident Response](https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/welcome.html)

## Contact Information

- **Security Team**: security@company.com
- **Incident Response**: ir@company.com / PagerDuty
- **Data Protection Officer**: dpo@company.com
- **Legal**: legal@company.com
- **AWS Support**: Premium Support (security incidents)
- **Law Enforcement**: [Local cybercrime unit contact]
