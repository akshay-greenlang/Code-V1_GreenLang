# GL-VCCI Scope 3 Platform v2.0 - Security Guide

## Table of Contents

1. [Overview](#overview)
2. [Security Baseline](#security-baseline)
3. [Compliance Monitoring](#compliance-monitoring)
4. [Vulnerability Management](#vulnerability-management)
5. [Incident Response](#incident-response)
6. [Access Control](#access-control)
7. [Encryption](#encryption)
8. [Security Auditing](#security-auditing)

---

## Overview

### Security Framework
```yaml
Standards Compliance:
  - SOC 2 Type II
  - ISO 27001
  - GDPR
  - CCPA
  - NIST Cybersecurity Framework

Security Layers:
  - Network Security
  - Application Security
  - Data Security
  - Identity & Access Management
  - Monitoring & Response
```

### Security Objectives
- **Confidentiality**: Protect sensitive data from unauthorized access
- **Integrity**: Ensure data accuracy and prevent unauthorized modification
- **Availability**: Maintain 99.9% uptime with resilience against attacks
- **Compliance**: Meet all regulatory and contractual requirements
- **Incident Response**: Detect and respond to security events within 15 minutes

---

## Security Baseline

### 1. Infrastructure Security

**Network Security Configuration**
```bash
#!/bin/bash
# verify-network-security.sh

echo "Verifying network security configuration..."

# 1. Check VPC configuration
echo "1. VPC Configuration:"
aws ec2 describe-vpcs \
  --filters "Name=tag:Environment,Values=production" \
  --query 'Vpcs[*].[VpcId,CidrBlock,IsDefault]'

# 2. Security Groups - No 0.0.0.0/0 ingress except for ALB
echo ""
echo "2. Security Group Audit:"
OPEN_SG=$(aws ec2 describe-security-groups \
  --filters "Name=ip-permission.cidr,Values=0.0.0.0/0" \
  --query 'SecurityGroups[?IpPermissions[?FromPort!=`443` && FromPort!=`80`]].[GroupId,GroupName]' \
  --output text)

if [ -n "$OPEN_SG" ]; then
  echo "⚠️  WARNING: Security groups with unrestricted access found:"
  echo "$OPEN_SG"
else
  echo "✓ No overly permissive security groups"
fi

# 3. Network ACLs
echo ""
echo "3. Network ACL Configuration:"
aws ec2 describe-network-acls \
  --filters "Name=vpc-id,Values=$(aws ec2 describe-vpcs --filters 'Name=tag:Environment,Values=production' --query 'Vpcs[0].VpcId' --output text)" \
  --query 'NetworkAcls[*].Entries[?RuleAction==`allow` && CidrBlock==`0.0.0.0/0`]'

# 4. VPC Flow Logs
echo ""
echo "4. VPC Flow Logs Status:"
aws ec2 describe-flow-logs \
  --filter "Name=resource-type,Values=VPC" \
  --query 'FlowLogs[*].[FlowLogId,FlowLogStatus,ResourceId]'

# 5. Private Subnets for Databases
echo ""
echo "5. Database Subnet Configuration:"
aws rds describe-db-instances \
  --query 'DBInstances[*].[DBInstanceIdentifier,PubliclyAccessible,DBSubnetGroup.VpcId]'

echo ""
echo "Network security verification complete"
```

**Enable WAF (Web Application Firewall)**
```bash
#!/bin/bash
# configure-waf.sh

echo "Configuring WAF for API Gateway..."

# Create WAF Web ACL
WAF_ACL_ID=$(aws wafv2 create-web-acl \
  --name vcci-production-waf \
  --scope REGIONAL \
  --default-action Allow={} \
  --rules file://waf-rules.json \
  --visibility-config \
    SampledRequestsEnabled=true,CloudWatchMetricsEnabled=true,MetricName=VCCIWAFMetrics \
  --query 'Summary.Id' \
  --output text)

echo "WAF ACL created: $WAF_ACL_ID"

# WAF Rules Configuration
cat > waf-rules.json <<'EOF'
[
  {
    "Name": "RateLimitRule",
    "Priority": 1,
    "Statement": {
      "RateBasedStatement": {
        "Limit": 2000,
        "AggregateKeyType": "IP"
      }
    },
    "Action": {
      "Block": {}
    },
    "VisibilityConfig": {
      "SampledRequestsEnabled": true,
      "CloudWatchMetricsEnabled": true,
      "MetricName": "RateLimitRule"
    }
  },
  {
    "Name": "SQLInjectionRule",
    "Priority": 2,
    "Statement": {
      "ManagedRuleGroupStatement": {
        "VendorName": "AWS",
        "Name": "AWSManagedRulesSQLiRuleSet"
      }
    },
    "OverrideAction": {
      "None": {}
    },
    "VisibilityConfig": {
      "SampledRequestsEnabled": true,
      "CloudWatchMetricsEnabled": true,
      "MetricName": "SQLInjectionRule"
    }
  },
  {
    "Name": "XSSRule",
    "Priority": 3,
    "Statement": {
      "ManagedRuleGroupStatement": {
        "VendorName": "AWS",
        "Name": "AWSManagedRulesKnownBadInputsRuleSet"
      }
    },
    "OverrideAction": {
      "None": {}
    },
    "VisibilityConfig": {
      "SampledRequestsEnabled": true,
      "CloudWatchMetricsEnabled": true,
      "MetricName": "XSSRule"
    }
  },
  {
    "Name": "GeoBlockingRule",
    "Priority": 4,
    "Statement": {
      "GeoMatchStatement": {
        "CountryCodes": ["CN", "RU", "KP"]
      }
    },
    "Action": {
      "Block": {}
    },
    "VisibilityConfig": {
      "SampledRequestsEnabled": true,
      "CloudWatchMetricsEnabled": true,
      "MetricName": "GeoBlockingRule"
    }
  }
]
EOF

# Associate with ALB
ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names vcci-production-alb \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

aws wafv2 associate-web-acl \
  --web-acl-arn "arn:aws:wafv2:region:account:regional/webacl/vcci-production-waf/$WAF_ACL_ID" \
  --resource-arn "$ALB_ARN"

echo "WAF configured and associated with ALB"
```

### 2. Container Security

**Scan Container Images**
```bash
#!/bin/bash
# scan-container-images.sh

IMAGE=$1

echo "Scanning container image: $IMAGE"

# 1. Trivy vulnerability scan
echo "1. Running Trivy vulnerability scan..."
trivy image --severity HIGH,CRITICAL "$IMAGE"

# 2. Check for root user
echo ""
echo "2. Checking if container runs as root..."
docker inspect "$IMAGE" | jq '.[0].Config.User'

# 3. Check for exposed secrets
echo ""
echo "3. Scanning for secrets..."
trivy fs --security-checks secret "$IMAGE"

# 4. Check base image
echo ""
echo "4. Base image information:"
docker history "$IMAGE" | head -5

# 5. Generate SBOM (Software Bill of Materials)
echo ""
echo "5. Generating SBOM..."
syft "$IMAGE" -o json > "sbom-${IMAGE//\//-}.json"

echo "Container security scan complete"
```

**Kubernetes Pod Security Standards**
```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: vcci-restricted
  annotations:
    seccomp.security.alpha.kubernetes.io/allowedProfileNames: 'runtime/default'
    apparmor.security.beta.kubernetes.io/allowedProfileNames: 'runtime/default'
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### 3. Application Security

**Security Headers Configuration**
```python
# security_middleware.py
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

@app.after_request
def set_security_headers(response):
    """Apply security headers to all responses"""

    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'

    # Enable XSS protection
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # Content Security Policy
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.vcci-platform.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' https://api.vcci-platform.com; "
        "frame-ancestors 'none';"
    )

    # Strict Transport Security (HSTS)
    response.headers['Strict-Transport-Security'] = (
        'max-age=31536000; includeSubDomains; preload'
    )

    # Referrer Policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    # Permissions Policy
    response.headers['Permissions-Policy'] = (
        'geolocation=(), microphone=(), camera=()'
    )

    return response
```

**Input Validation**
```python
# input_validation.py
from typing import Any, Dict
import re
from marshmallow import Schema, fields, validates, ValidationError

class EmissionInputSchema(Schema):
    """Validate emission record input"""

    category = fields.Str(required=True)
    scope = fields.Int(required=True, validate=lambda x: 1 <= x <= 3)
    amount = fields.Decimal(required=True, validate=lambda x: x >= 0)
    unit = fields.Str(required=True)
    source_id = fields.Str()
    metadata = fields.Dict()

    @validates('category')
    def validate_category(self, value):
        """Validate emission category"""
        allowed_categories = [
            'purchased_goods', 'capital_goods', 'fuel_energy',
            'upstream_transport', 'waste', 'business_travel',
            'employee_commuting', 'upstream_leased', 'downstream_transport',
            'processing', 'use_of_products', 'end_of_life',
            'downstream_leased', 'franchises', 'investments'
        ]
        if value not in allowed_categories:
            raise ValidationError(f'Invalid category: {value}')

    @validates('source_id')
    def validate_source_id(self, value):
        """Validate source ID format"""
        if value and not re.match(r'^[a-zA-Z0-9\-_]+$', value):
            raise ValidationError('Source ID contains invalid characters')

def validate_input(data: Dict[str, Any], schema: Schema) -> Dict[str, Any]:
    """Validate and sanitize input data"""
    try:
        return schema.load(data)
    except ValidationError as e:
        raise ValueError(f"Validation error: {e.messages}")
```

---

## Compliance Monitoring

### 1. SOC 2 Compliance

**SOC 2 Control Verification**
```bash
#!/bin/bash
# verify-soc2-controls.sh

echo "Verifying SOC 2 Type II Controls..."

# CC6.1: Logical and Physical Access Controls
echo "CC6.1: Access Controls"
echo "  - MFA enforcement: $(check_mfa_enforcement)"
echo "  - Password policy: $(check_password_policy)"
echo "  - Session timeout: $(check_session_timeout)"

# CC6.6: Encryption
echo ""
echo "CC6.6: Encryption"
echo "  - Data at rest: $(verify_encryption_at_rest)"
echo "  - Data in transit: $(verify_encryption_in_transit)"
echo "  - Key rotation: $(check_key_rotation)"

# CC6.7: Transmission Protection
echo ""
echo "CC6.7: Transmission Protection"
echo "  - TLS version: $(check_tls_version)"
echo "  - Certificate validity: $(check_certificate_validity)"

# CC7.2: System Monitoring
echo ""
echo "CC7.2: System Monitoring"
echo "  - Security monitoring: $(check_security_monitoring)"
echo "  - Intrusion detection: $(check_intrusion_detection)"
echo "  - Log aggregation: $(check_log_aggregation)"

# CC8.1: Vulnerability Management
echo ""
echo "CC8.1: Vulnerability Management"
echo "  - Vulnerability scanning: $(check_vulnerability_scanning)"
echo "  - Patch management: $(check_patch_management)"
echo "  - Penetration testing: $(check_penetration_testing)"

# Generate compliance report
./generate-soc2-report.sh
```

### 2. GDPR Compliance

**GDPR Data Subject Rights**
```bash
#!/bin/bash
# gdpr-data-subject-request.sh

REQUEST_TYPE=$1  # access, rectification, erasure, portability
USER_EMAIL=$2

echo "Processing GDPR data subject request"
echo "Type: $REQUEST_TYPE"
echo "User: $USER_EMAIL"

case $REQUEST_TYPE in
  access)
    # Right to Access (Article 15)
    echo "Exporting all personal data for user..."
    kubectl exec -n vcci-production deployment/vcci-api -- \
      python manage.py export_user_data \
        --email "$USER_EMAIL" \
        --format json \
        --include-audit-logs
    ;;

  rectification)
    # Right to Rectification (Article 16)
    echo "User can update their profile via self-service portal"
    echo "Portal URL: https://app.vcci-platform.com/profile"
    ;;

  erasure)
    # Right to Erasure (Article 17)
    echo "Initiating data deletion..."
    ./delete-user.sh "$USER_EMAIL" "GDPR-erasure-request"
    ;;

  portability)
    # Right to Data Portability (Article 20)
    echo "Exporting data in machine-readable format..."
    kubectl exec -n vcci-production deployment/vcci-api -- \
      python manage.py export_user_data \
        --email "$USER_EMAIL" \
        --format csv \
        --portable
    ;;

  *)
    echo "Invalid request type: $REQUEST_TYPE"
    exit 1
    ;;
esac

# Log the request (required for compliance)
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "gdpr_data_subject_request" \
    --user-email "$USER_EMAIL" \
    --details "Request type: $REQUEST_TYPE"

echo "Request processed successfully"
```

**GDPR Data Breach Notification**
```bash
#!/bin/bash
# gdpr-breach-notification.sh

BREACH_ID=$1
AFFECTED_USERS=$2  # File containing list of affected user emails

echo "Processing GDPR data breach notification"
echo "Breach ID: $BREACH_ID"

# Must notify within 72 hours of becoming aware
NOTIFICATION_DEADLINE=$(date -d "+72 hours" +"%Y-%m-%d %H:%M:%S")

echo "Notification deadline: $NOTIFICATION_DEADLINE"

# 1. Notify supervisory authority
echo "1. Notifying supervisory authority..."
./notify-supervisory-authority.sh "$BREACH_ID"

# 2. Notify affected individuals
echo "2. Notifying affected individuals..."
while IFS= read -r email; do
  ./send-breach-notification.sh "$email" "$BREACH_ID"
done < "$AFFECTED_USERS"

# 3. Document the breach
echo "3. Documenting breach in compliance register..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py record_data_breach \
    --breach-id "$BREACH_ID" \
    --notification-date "$(date +%Y-%m-%d)" \
    --affected-users-file "$AFFECTED_USERS"

echo "GDPR breach notification process complete"
```

### 3. Continuous Compliance Monitoring

**Automated Compliance Checks**
```bash
#!/bin/bash
# continuous-compliance-check.sh

echo "Running continuous compliance checks..."

# Schedule as cron job: 0 */4 * * * (every 4 hours)

REPORT_FILE="compliance-report-$(date +%Y%m%d-%H%M%S).json"

# Initialize report
cat > "$REPORT_FILE" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "checks": []
}
EOF

# 1. Encryption check
echo "1. Checking encryption..."
ENCRYPTION_STATUS=$(./verify-encryption.sh)
jq ".checks += [{\"name\": \"encryption\", \"status\": \"$ENCRYPTION_STATUS\"}]" "$REPORT_FILE" > tmp && mv tmp "$REPORT_FILE"

# 2. Access control check
echo "2. Checking access controls..."
ACCESS_STATUS=$(./verify-access-controls.sh)
jq ".checks += [{\"name\": \"access_controls\", \"status\": \"$ACCESS_STATUS\"}]" "$REPORT_FILE" > tmp && mv tmp "$REPORT_FILE"

# 3. Audit logging check
echo "3. Checking audit logging..."
AUDIT_STATUS=$(./verify-audit-logging.sh)
jq ".checks += [{\"name\": \"audit_logging\", \"status\": \"$AUDIT_STATUS\"}]" "$REPORT_FILE" > tmp && mv tmp "$REPORT_FILE"

# 4. Vulnerability status
echo "4. Checking vulnerabilities..."
VULN_STATUS=$(./check-vulnerabilities.sh)
jq ".checks += [{\"name\": \"vulnerabilities\", \"status\": \"$VULN_STATUS\"}]" "$REPORT_FILE" > tmp && mv tmp "$REPORT_FILE"

# 5. Backup status
echo "5. Checking backups..."
BACKUP_STATUS=$(./verify-backups.sh)
jq ".checks += [{\"name\": \"backups\", \"status\": \"$BACKUP_STATUS\"}]" "$REPORT_FILE" > tmp && mv tmp "$REPORT_FILE"

# Upload report
aws s3 cp "$REPORT_FILE" "s3://vcci-compliance-reports/$(date +%Y/%m/%d)/"

echo "Compliance check complete: $REPORT_FILE"
```

---

## Vulnerability Management

### 1. Vulnerability Scanning

**Automated Vulnerability Scanning**
```bash
#!/bin/bash
# vulnerability-scan.sh

SCAN_TYPE=${1:-full}  # full, quick, compliance

echo "Running vulnerability scan (type: $SCAN_TYPE)..."

# 1. Infrastructure scan
echo "1. Scanning infrastructure..."
aws inspector start-assessment-run \
  --assessment-template-arn "arn:aws:inspector:region:account:target/0-xxxxxxxx/template/0-yyyyyyyy"

# 2. Container image scan
echo "2. Scanning container images..."
for image in $(kubectl get pods -n vcci-production -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u); do
  echo "Scanning $image..."
  trivy image --severity HIGH,CRITICAL --format json "$image" > "scan-$(echo $image | tr '/:' '-').json"
done

# 3. Application dependencies scan
echo "3. Scanning application dependencies..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  safety check --json > dependency-scan.json

# 4. Web application scan
if [ "$SCAN_TYPE" == "full" ]; then
  echo "4. Running OWASP ZAP scan..."
  docker run -t owasp/zap2docker-stable zap-baseline.py \
    -t https://api.vcci-platform.com \
    -r zap-report.html
fi

# 5. Database security scan
echo "5. Scanning database configuration..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -f /app/scripts/security-audit.sql

# Generate consolidated report
echo "Generating consolidated vulnerability report..."
./generate-vulnerability-report.sh

echo "Vulnerability scan complete"
```

### 2. Patch Management

**Security Patch Workflow**
```bash
#!/bin/bash
# apply-security-patches.sh

SEVERITY=$1  # critical, high, medium, low

echo "Applying security patches (severity: $SEVERITY)..."

# 1. Check for available patches
echo "1. Checking for available patches..."
PATCHES=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  apt list --upgradable 2>/dev/null | grep -i security)

if [ -z "$PATCHES" ]; then
  echo "No security patches available"
  exit 0
fi

echo "Available patches:"
echo "$PATCHES"

# 2. Test patches in staging
echo ""
echo "2. Testing patches in staging environment..."
./deploy-to-staging.sh

# Run automated tests
kubectl exec -n vcci-staging deployment/vcci-api -- \
  pytest /app/tests/security/

# 3. Create maintenance window
if [ "$SEVERITY" == "critical" ]; then
  echo "3. Emergency patching - immediate deployment"
  APPROVAL="auto-approved"
else
  echo "3. Creating maintenance window..."
  APPROVAL=$(./request-maintenance-approval.sh "$SEVERITY")
fi

if [ "$APPROVAL" != "approved" ] && [ "$APPROVAL" != "auto-approved" ]; then
  echo "Patch deployment not approved"
  exit 1
fi

# 4. Apply patches to production
echo "4. Applying patches to production..."
./enable-maintenance-mode.sh

# Update base image with patches
docker build -t vcci-platform:patched -f Dockerfile.patched .
docker tag vcci-platform:patched your-registry.com/vcci-platform:$(date +%Y%m%d)-patched
docker push your-registry.com/vcci-platform:$(date +%Y%m%d)-patched

# Rolling update
kubectl set image deployment/vcci-api -n vcci-production \
  api=your-registry.com/vcci-platform:$(date +%Y%m%d)-patched

# Wait for rollout
kubectl rollout status deployment/vcci-api -n vcci-production

./disable-maintenance-mode.sh

# 5. Verify patches
echo "5. Verifying patch application..."
./verify-patches.sh

echo "Security patches applied successfully"
```

### 3. Penetration Testing

**Schedule Penetration Test**
```bash
#!/bin/bash
# schedule-pentest.sh

TEST_TYPE=$1  # internal, external, both
VENDOR=$2

echo "Scheduling penetration test"
echo "Type: $TEST_TYPE"
echo "Vendor: $VENDOR"

# Pre-test preparation
echo "Pre-test checklist:"
echo "  1. Notify SOC team"
echo "  2. Whitelist vendor IPs"
echo "  3. Prepare test accounts"
echo "  4. Document test scope"
echo "  5. Set up monitoring"

# Create test environment snapshot
echo "Creating test environment snapshot..."
./create-pentest-environment.sh

# Provide vendor with scope document
cat > pentest-scope.md <<EOF
# Penetration Test Scope

## In-Scope
- API endpoints: https://api.vcci-platform.com
- Web application: https://app.vcci-platform.com
- Authentication flows
- Authorization mechanisms
- Data validation
- Session management

## Out-of-Scope
- Physical security
- Social engineering
- DoS attacks
- Third-party services

## Test Accounts
- Test User: pentest@vcci-platform.com
- Test Tenant: pentest-tenant-001
- API Key: [provided separately]

## Rules of Engagement
- Testing window: [specific dates/times]
- Notification required for any DoS scenarios
- Stop immediately if production data accessed
- Daily progress reports required
EOF

echo "Penetration test scheduled"
echo "Scope document: pentest-scope.md"
```

---

## Incident Response

### 1. Security Incident Classification

**Incident Severity Matrix**
```yaml
P0 - Critical:
  Description: Active breach, data exfiltration, ransomware
  Response Time: Immediate (within 15 minutes)
  Escalation: CISO, CEO, Legal
  Examples:
    - Confirmed data breach
    - Ransomware infection
    - Root access compromised

P1 - High:
  Description: Attempted breach, vulnerability exploitation
  Response Time: Within 1 hour
  Escalation: Security team, Engineering manager
  Examples:
    - Failed intrusion attempt
    - Zero-day vulnerability discovered
    - DDoS attack in progress

P2 - Medium:
  Description: Security policy violation, suspicious activity
  Response Time: Within 4 hours
  Escalation: Security team
  Examples:
    - Unauthorized access attempt
    - Malware detected
    - Security misconfiguration

P3 - Low:
  Description: Minor security issues, compliance gaps
  Response Time: Within 24 hours
  Escalation: Security team
  Examples:
    - Outdated software
    - Minor policy violations
    - Security awareness issues
```

### 2. Incident Response Playbook

**Security Incident Response**
```bash
#!/bin/bash
# incident-response.sh

INCIDENT_ID=$1
INCIDENT_TYPE=$2  # breach, malware, ddos, unauthorized_access
SEVERITY=$3  # P0, P1, P2, P3

echo "=========================================="
echo "SECURITY INCIDENT RESPONSE"
echo "=========================================="
echo "Incident ID: $INCIDENT_ID"
echo "Type: $INCIDENT_TYPE"
echo "Severity: $SEVERITY"
echo "Time: $(date)"

# Phase 1: Preparation
echo ""
echo "Phase 1: Preparation"
echo "- Assembling incident response team"
echo "- Activating communication channels"
./activate-incident-response-team.sh "$INCIDENT_ID"

# Phase 2: Identification
echo ""
echo "Phase 2: Identification"
echo "- Collecting evidence"
echo "- Analyzing logs"
echo "- Determining scope"
./collect-forensic-evidence.sh "$INCIDENT_ID"

# Phase 3: Containment
echo ""
echo "Phase 3: Containment"
case $INCIDENT_TYPE in
  breach)
    echo "- Isolating affected systems"
    ./isolate-compromised-systems.sh "$INCIDENT_ID"
    echo "- Revoking compromised credentials"
    ./revoke-compromised-credentials.sh "$INCIDENT_ID"
    echo "- Blocking attacker IPs"
    ./block-attacker-ips.sh "$INCIDENT_ID"
    ;;

  malware)
    echo "- Quarantining infected systems"
    ./quarantine-infected-systems.sh "$INCIDENT_ID"
    echo "- Running malware scan"
    ./run-malware-scan.sh
    ;;

  ddos)
    echo "- Activating DDoS mitigation"
    ./activate-ddos-mitigation.sh
    echo "- Scaling infrastructure"
    ./scale-infrastructure.sh emergency
    ;;

  unauthorized_access)
    echo "- Terminating unauthorized sessions"
    ./terminate-unauthorized-sessions.sh "$INCIDENT_ID"
    echo "- Enforcing MFA"
    ./enforce-mfa.sh all
    ;;
esac

# Phase 4: Eradication
echo ""
echo "Phase 4: Eradication"
echo "- Removing threat"
echo "- Patching vulnerabilities"
echo "- Strengthening controls"
./eradicate-threat.sh "$INCIDENT_ID"

# Phase 5: Recovery
echo ""
echo "Phase 5: Recovery"
echo "- Restoring normal operations"
echo "- Validating security controls"
echo "- Monitoring for recurrence"
./recovery-operations.sh "$INCIDENT_ID"

# Phase 6: Lessons Learned
echo ""
echo "Phase 6: Lessons Learned"
echo "- Scheduling post-incident review"
./schedule-post-incident-review.sh "$INCIDENT_ID"

echo ""
echo "Incident response workflow initiated"
echo "Incident ID: $INCIDENT_ID"
```

---

## Access Control

### 1. Principle of Least Privilege

**Review Access Permissions**
```bash
#!/bin/bash
# review-access-permissions.sh

echo "Reviewing access permissions..."

# 1. Database access
echo "1. Database Access:"
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    SELECT
      usename,
      usesuper,
      usecreatedb,
      usebypassrls
    FROM pg_user
    WHERE usesuper = true OR usecreatedb = true OR usebypassrls = true;
  "

# 2. AWS IAM permissions
echo ""
echo "2. AWS IAM Overly Permissive Policies:"
aws iam list-policies --scope Local --query 'Policies[*].[PolicyName]' --output text | \
  while read policy; do
    ADMIN=$(aws iam get-policy-version \
      --policy-arn "arn:aws:iam::account:policy/$policy" \
      --version-id v1 \
      --query 'PolicyVersion.Document.Statement[?Effect==`Allow` && Action==`*` && Resource==`*`]' \
      --output text)
    if [ -n "$ADMIN" ]; then
      echo "  ⚠️  $policy has admin-like permissions"
    fi
  done

# 3. Kubernetes RBAC
echo ""
echo "3. Kubernetes Cluster Admin Access:"
kubectl get clusterrolebindings -o json | \
  jq -r '.items[] | select(.roleRef.name=="cluster-admin") | .subjects[]? | "\(.kind): \(.name)"'

# 4. Service account permissions
echo ""
echo "4. Service Account Token Usage:"
kubectl get serviceaccounts --all-namespaces -o json | \
  jq -r '.items[] | select(.secrets != null) | "\(.metadata.namespace)/\(.metadata.name): \((.secrets | length))"'

echo ""
echo "Access permission review complete"
```

### 2. Just-In-Time (JIT) Access

**Request Temporary Elevated Access**
```bash
#!/bin/bash
# request-jit-access.sh

RESOURCE=$1  # database, kubernetes, aws
DURATION_HOURS=$2
JUSTIFICATION=$3

echo "Requesting JIT access"
echo "Resource: $RESOURCE"
echo "Duration: $DURATION_HOURS hours"
echo "Justification: $JUSTIFICATION"

# Create access request
REQUEST_ID=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py create_jit_access_request \
    --resource "$RESOURCE" \
    --duration "$DURATION_HOURS" \
    --justification "$JUSTIFICATION" \
    --requester "$(whoami)")

echo "Access request created: $REQUEST_ID"

# Send notification to approvers
./notify-approvers.sh "$REQUEST_ID"

echo "Waiting for approval..."

# Poll for approval (in production, this would be async)
APPROVED=false
for i in {1..60}; do
  STATUS=$(./check-request-status.sh "$REQUEST_ID")
  if [ "$STATUS" == "approved" ]; then
    APPROVED=true
    break
  elif [ "$STATUS" == "denied" ]; then
    echo "Access request denied"
    exit 1
  fi
  sleep 10
done

if [ "$APPROVED" == "true" ]; then
  echo "Access approved!"

  # Grant temporary access
  ./grant-temporary-access.sh "$REQUEST_ID" "$RESOURCE" "$DURATION_HOURS"

  # Schedule automatic revocation
  at now + "$DURATION_HOURS" hours <<EOF
./revoke-temporary-access.sh "$REQUEST_ID"
EOF

  echo "Temporary access granted for $DURATION_HOURS hours"
else
  echo "Access request timed out"
  exit 1
fi
```

---

## Encryption

### 1. Encryption at Rest

**Verify Encryption at Rest**
```bash
#!/bin/bash
# verify-encryption-at-rest.sh

echo "Verifying encryption at rest..."

# 1. RDS encryption
echo "1. RDS Database Encryption:"
aws rds describe-db-instances \
  --db-instance-identifier vcci-production \
  --query 'DBInstances[0].[StorageEncrypted,KmsKeyId]' \
  --output table

# 2. EBS encryption
echo ""
echo "2. EBS Volume Encryption:"
aws ec2 describe-volumes \
  --filters "Name=tag:Environment,Values=production" \
  --query 'Volumes[*].[VolumeId,Encrypted,KmsKeyId]' \
  --output table

# 3. S3 encryption
echo ""
echo "3. S3 Bucket Encryption:"
aws s3api get-bucket-encryption --bucket vcci-production-data

# 4. ElastiCache encryption
echo ""
echo "4. Redis Encryption:"
aws elasticache describe-replication-groups \
  --replication-group-id vcci-production \
  --query 'ReplicationGroups[0].[AtRestEncryptionEnabled,TransitEncryptionEnabled]' \
  --output table

# 5. Secrets Manager encryption
echo ""
echo "5. Secrets Manager Encryption:"
aws secretsmanager describe-secret \
  --secret-id vcci/production/database \
  --query '[KmsKeyId]' \
  --output table

echo ""
echo "✓ All data at rest is encrypted"
```

### 2. Encryption in Transit

**Enforce TLS 1.2+**
```bash
#!/bin/bash
# enforce-tls.sh

echo "Enforcing TLS 1.2+ for all connections..."

# 1. Update ALB policy
echo "1. Updating ALB Security Policy..."
aws elbv2 modify-listener \
  --listener-arn "arn:aws:elasticloadbalancing:region:account:listener/app/vcci-production/xxx/yyy" \
  --ssl-policy ELBSecurityPolicy-TLS-1-2-2017-01

# 2. Update RDS SSL requirement
echo "2. Enforcing SSL for database connections..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    ALTER SYSTEM SET ssl = on;
    ALTER SYSTEM SET ssl_min_protocol_version = 'TLSv1.2';
    SELECT pg_reload_conf();
  "

# 3. Update Redis TLS
echo "3. Enabling Redis TLS..."
aws elasticache modify-replication-group \
  --replication-group-id vcci-production \
  --transit-encryption-enabled

# 4. Verify TLS versions
echo ""
echo "4. Verifying TLS configuration:"
echo "Testing API endpoint..."
curl -sI https://api.vcci-platform.com | grep -i "strict-transport-security"

echo ""
echo "✓ TLS 1.2+ enforced across all services"
```

### 3. Key Rotation

**Rotate Encryption Keys**
```bash
#!/bin/bash
# rotate-encryption-keys.sh

KEY_TYPE=$1  # database, storage, application

echo "Rotating $KEY_TYPE encryption keys..."

case $KEY_TYPE in
  database)
    # Rotate RDS KMS key
    OLD_KEY=$(aws rds describe-db-instances \
      --db-instance-identifier vcci-production \
      --query 'DBInstances[0].KmsKeyId' \
      --output text)

    NEW_KEY=$(aws kms create-key \
      --description "VCCI Production Database Key $(date +%Y-%m-%d)" \
      --query 'KeyMetadata.KeyId' \
      --output text)

    echo "New database encryption key: $NEW_KEY"
    echo "Scheduling key rotation..."

    # Schedule maintenance window for re-encryption
    ./schedule-database-reencryption.sh "$NEW_KEY"
    ;;

  storage)
    # Rotate S3 KMS key
    aws kms enable-key-rotation \
      --key-id alias/vcci-storage-key

    echo "Automatic key rotation enabled for storage"
    ;;

  application)
    # Rotate application secrets
    ./rotate-application-secrets.sh
    ;;
esac

echo "Key rotation initiated for $KEY_TYPE"
```

---

## Security Auditing

### 1. Security Audit Log Collection

**Comprehensive Audit Logging**
```python
# audit_logger.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class SecurityAuditLogger:
    """Centralized security audit logging"""

    def __init__(self):
        self.logger = logging.getLogger('security_audit')
        self.logger.setLevel(logging.INFO)

    def log_event(
        self,
        event_type: str,
        user_id: str,
        tenant_id: str,
        action: str,
        resource: str,
        result: str,
        ip_address: str,
        user_agent: str,
        details: Dict[str, Any] = None
    ):
        """Log security audit event"""

        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "action": action,
            "resource": resource,
            "result": result,  # success, failure, denied
            "ip_address": ip_address,
            "user_agent": user_agent,
            "details": details or {}
        }

        self.logger.info(json.dumps(audit_record))

        # Also store in database
        self._store_in_database(audit_record)

    def _store_in_database(self, record: Dict[str, Any]):
        """Store audit record in database"""
        # Implementation here
        pass

# Usage examples
audit_logger = SecurityAuditLogger()

# Login attempt
audit_logger.log_event(
    event_type="authentication",
    user_id="user123",
    tenant_id="acme-corp",
    action="login",
    resource="auth_service",
    result="success",
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0...",
    details={"mfa_used": True}
)

# Data access
audit_logger.log_event(
    event_type="data_access",
    user_id="user123",
    tenant_id="acme-corp",
    action="read",
    resource="emissions_data",
    result="success",
    ip_address="192.168.1.100",
    user_agent="API Client",
    details={"record_count": 150}
)

# Permission change
audit_logger.log_event(
    event_type="authorization",
    user_id="admin456",
    tenant_id="acme-corp",
    action="grant_permission",
    resource="user:user123",
    result="success",
    ip_address="192.168.1.101",
    user_agent="Mozilla/5.0...",
    details={"permission": "manage_suppliers"}
)
```

### 2. Security Metrics Dashboard

**Key Security Metrics**
```yaml
# prometheus-security-rules.yaml
groups:
- name: security_metrics
  rules:
  - record: security:failed_login_attempts:rate5m
    expr: rate(failed_login_attempts_total[5m])

  - record: security:unauthorized_access_attempts:rate5m
    expr: rate(unauthorized_access_attempts_total[5m])

  - record: security:api_key_usage:rate5m
    expr: rate(api_key_validations_total[5m])

  - record: security:mfa_success_rate
    expr: |
      sum(rate(mfa_validations_total{result="success"}[5m]))
      /
      sum(rate(mfa_validations_total[5m]))

  - record: security:encryption_failures:rate5m
    expr: rate(encryption_failures_total[5m])
```

---

## Appendix

### Security Contact Information

```yaml
Security Team:
  Email: security@vcci-platform.com
  Slack: #security-incidents
  PagerDuty: security-oncall
  Phone: +1-555-SECURITY (24/7)

Bug Bounty Program:
  URL: https://vcci-platform.com/security/bug-bounty
  Scope: https://vcci-platform.com/security/scope
  Email: bugbounty@vcci-platform.com

Responsible Disclosure:
  Email: security@vcci-platform.com
  PGP Key: https://vcci-platform.com/security/pgp-key
  Response SLA: 48 hours
```

### Security Tools

```yaml
Scanning & Monitoring:
  - Trivy (container scanning)
  - AWS Inspector (infrastructure)
  - OWASP ZAP (web application)
  - Snyk (dependency scanning)
  - Wazuh (SIEM)

Access Management:
  - Okta (SSO)
  - Vault (secrets management)
  - AWS IAM (cloud access)

Incident Response:
  - PagerDuty (alerting)
  - TheHive (case management)
  - Cortex (threat intelligence)
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-06
**Maintained By**: Security Team
