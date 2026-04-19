# GL-VCCI Scope 3 Platform v2.0 - Tenant Management Guide

## Table of Contents

1. [Overview](#overview)
2. [Tenant Onboarding](#tenant-onboarding)
3. [Resource Quota Management](#resource-quota-management)
4. [Data Isolation](#data-isolation)
5. [Tenant Configuration](#tenant-configuration)
6. [Tenant Monitoring](#tenant-monitoring)
7. [Tenant Offboarding](#tenant-offboarding)
8. [Billing Integration](#billing-integration)

---

## Overview

### Purpose
This guide provides comprehensive procedures for managing tenants in the GL-VCCI Scope 3 Carbon Intelligence Platform.

### Multi-Tenancy Model
```yaml
Isolation Level: STRICT
Data Segregation: Schema-based
Resource Allocation: Quota-based
Billing: Usage-based metering
Support Tiers: Standard, Professional, Enterprise
```

### Tenant Tiers
```yaml
Standard:
  Max Users: 25
  Max Records: 1M/month
  API Calls: 1M/month
  Storage: 100GB
  Support: Email (48h)
  Price: $999/month

Professional:
  Max Users: 100
  Max Records: 10M/month
  API Calls: 10M/month
  Storage: 500GB
  Support: Email + Chat (24h)
  Price: $4,999/month

Enterprise:
  Max Users: Unlimited
  Max Records: Unlimited
  API Calls: Unlimited
  Storage: 5TB+
  Support: 24/7 Phone + Dedicated CSM
  Price: Custom
```

---

## Tenant Onboarding

### 1. Tenant Provisioning Request

**Create Tenant Request**
```yaml
# tenant-request.yaml
tenant_request:
  # Organization Information
  organization_name: "Acme Corporation"
  tenant_id: "acme-corp"  # Unique identifier
  domain: "acme.com"
  industry: "Manufacturing"
  size: "1000-5000 employees"

  # Primary Contact
  primary_contact:
    name: "Jane Smith"
    email: "jane.smith@acme.com"
    phone: "+1-555-0100"
    title: "VP of Sustainability"

  # Billing Contact
  billing_contact:
    name: "John Doe"
    email: "billing@acme.com"
    phone: "+1-555-0101"

  # Subscription Details
  tier: "professional"
  contract_start_date: "2025-02-01"
  contract_length_months: 12
  auto_renew: true

  # Resource Requirements
  estimated_users: 50
  estimated_records_per_month: 5000000
  estimated_api_calls_per_month: 2000000
  storage_requirement_gb: 250

  # Compliance Requirements
  compliance:
    - "SOC2"
    - "GDPR"
    - "ISO27001"

  # Optional Features
  features:
    - "advanced_analytics"
    - "supplier_portal"
    - "custom_integrations"
    - "dedicated_support"

  # Data Residency
  data_residency: "US"  # US, EU, APAC

  # SSO Configuration
  sso_enabled: true
  sso_provider: "Okta"
  sso_metadata_url: "https://acme.okta.com/metadata"
```

### 2. Tenant Provisioning Script

**Automated Tenant Provisioning**
```bash
#!/bin/bash
# provision-tenant.sh

set -euo pipefail

REQUEST_FILE=$1

echo "=========================================="
echo "VCCI Platform - Tenant Provisioning"
echo "=========================================="

# Parse request
TENANT_ID=$(yq eval '.tenant_request.tenant_id' "$REQUEST_FILE")
ORG_NAME=$(yq eval '.tenant_request.organization_name' "$REQUEST_FILE")
TIER=$(yq eval '.tenant_request.tier' "$REQUEST_FILE")
PRIMARY_EMAIL=$(yq eval '.tenant_request.primary_contact.email' "$REQUEST_FILE")

echo "Tenant ID: $TENANT_ID"
echo "Organization: $ORG_NAME"
echo "Tier: $TIER"

# Validate tenant ID is unique
echo ""
echo "Step 1: Validating tenant ID..."
TENANT_EXISTS=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py check_tenant_exists --tenant-id "$TENANT_ID" || echo "false")

if [ "$TENANT_EXISTS" == "true" ]; then
  echo "ERROR: Tenant ID $TENANT_ID already exists"
  exit 1
fi
echo "✓ Tenant ID is available"

# Create tenant database schema
echo ""
echo "Step 2: Creating tenant database schema..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "SELECT create_tenant_schema('${TENANT_ID}');"
echo "✓ Database schema created"

# Create tenant record
echo ""
echo "Step 3: Creating tenant record..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py create_tenant \
    --tenant-id "$TENANT_ID" \
    --name "$ORG_NAME" \
    --tier "$TIER" \
    --config-file "$REQUEST_FILE"
echo "✓ Tenant record created"

# Set resource quotas
echo ""
echo "Step 4: Configuring resource quotas..."
./set-tenant-quotas.sh "$TENANT_ID" "$TIER"
echo "✓ Resource quotas configured"

# Create initial admin user
echo ""
echo "Step 5: Creating initial admin user..."
ADMIN_PASSWORD=$(./create-tenant-admin.sh "$TENANT_ID" "$PRIMARY_EMAIL")
echo "✓ Admin user created"

# Configure SSO (if enabled)
SSO_ENABLED=$(yq eval '.tenant_request.sso_enabled' "$REQUEST_FILE")
if [ "$SSO_ENABLED" == "true" ]; then
  echo ""
  echo "Step 6: Configuring SSO..."
  ./configure-tenant-sso.sh "$TENANT_ID" "$REQUEST_FILE"
  echo "✓ SSO configured"
fi

# Generate API keys
echo ""
echo "Step 7: Generating API keys..."
API_KEY=$(./generate-tenant-api-key.sh "$TENANT_ID" "Default Key")
echo "✓ API keys generated"

# Initialize default data
echo ""
echo "Step 8: Initializing default data..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py init_tenant_data --tenant-id "$TENANT_ID"
echo "✓ Default data initialized"

# Configure monitoring
echo ""
echo "Step 9: Configuring monitoring..."
./configure-tenant-monitoring.sh "$TENANT_ID"
echo "✓ Monitoring configured"

# Send welcome email
echo ""
echo "Step 10: Sending welcome email..."
./send-tenant-welcome-email.sh "$TENANT_ID" "$PRIMARY_EMAIL" "$ADMIN_PASSWORD" "$API_KEY"
echo "✓ Welcome email sent"

# Create audit log
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "tenant_provisioned" \
    --tenant-id "$TENANT_ID" \
    --actor "system" \
    --details "Tier: $TIER, Organization: $ORG_NAME"

echo ""
echo "=========================================="
echo "Tenant Provisioning Complete!"
echo "=========================================="
echo "Tenant ID: $TENANT_ID"
echo "Admin Email: $PRIMARY_EMAIL"
echo "Admin Password: $ADMIN_PASSWORD"
echo "API Key: $API_KEY"
echo "Portal URL: https://app.vcci-platform.com"
echo ""
echo "IMPORTANT: Share credentials securely with customer"
```

### 3. Tenant Configuration

**Configure Tenant Settings**
```bash
#!/bin/bash
# configure-tenant.sh

TENANT_ID=$1
SETTING=$2
VALUE=$3

echo "Configuring tenant: $TENANT_ID"
echo "Setting: $SETTING = $VALUE"

# Update tenant configuration
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py update_tenant_config \
    --tenant-id "$TENANT_ID" \
    --setting "$SETTING" \
    --value "$VALUE"

# Supported settings:
# - retention_days: Data retention period
# - api_rate_limit: API rate limit per minute
# - max_file_size_mb: Maximum upload file size
# - enable_export: Allow data export
# - enable_api_access: Enable API access
# - notification_email: Email for alerts
# - webhook_url: Webhook for events

echo "Configuration updated"
```

---

## Resource Quota Management

### 1. Set Tenant Quotas

**Configure Resource Quotas**
```bash
#!/bin/bash
# set-tenant-quotas.sh

TENANT_ID=$1
TIER=$2

echo "Setting quotas for tenant: $TENANT_ID (tier: $TIER)"

# Define quotas based on tier
case $TIER in
  standard)
    MAX_USERS=25
    MAX_RECORDS=1000000
    MAX_API_CALLS=1000000
    MAX_STORAGE_GB=100
    ;;
  professional)
    MAX_USERS=100
    MAX_RECORDS=10000000
    MAX_API_CALLS=10000000
    MAX_STORAGE_GB=500
    ;;
  enterprise)
    MAX_USERS=0  # unlimited
    MAX_RECORDS=0  # unlimited
    MAX_API_CALLS=0  # unlimited
    MAX_STORAGE_GB=5000
    ;;
  *)
    echo "ERROR: Invalid tier: $TIER"
    exit 1
    ;;
esac

# Update database quotas
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    UPDATE tenants
    SET
      quota_users = $MAX_USERS,
      quota_records = $MAX_RECORDS,
      quota_api_calls = $MAX_API_CALLS,
      quota_storage_gb = $MAX_STORAGE_GB
    WHERE tenant_id = '${TENANT_ID}';
  "

# Update rate limiting in Redis
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u "$REDIS_URL" SET "quota:${TENANT_ID}:api_calls" "$MAX_API_CALLS"

kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u "$REDIS_URL" SET "quota:${TENANT_ID}:users" "$MAX_USERS"

echo "Quotas configured successfully"
```

### 2. Monitor Quota Usage

**Check Quota Usage**
```bash
#!/bin/bash
# check-quota-usage.sh

TENANT_ID=$1

echo "Checking quota usage for tenant: $TENANT_ID"

# Get current quotas
QUOTAS=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -t -c "
    SELECT
      quota_users,
      quota_records,
      quota_api_calls,
      quota_storage_gb
    FROM tenants
    WHERE tenant_id = '${TENANT_ID}';
  ")

# Get current usage
USAGE=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_tenant_usage --tenant-id "$TENANT_ID")

echo "Quota vs Usage Report:"
echo "======================"
echo "$USAGE"

# Check if approaching limits
USERS_PCT=$(echo "$USAGE" | jq -r '.users_percentage')
RECORDS_PCT=$(echo "$USAGE" | jq -r '.records_percentage')
API_CALLS_PCT=$(echo "$USAGE" | jq -r '.api_calls_percentage')
STORAGE_PCT=$(echo "$USAGE" | jq -r '.storage_percentage')

# Alert if any quota is > 80%
if (( $(echo "$USERS_PCT > 80" | bc -l) )) || \
   (( $(echo "$RECORDS_PCT > 80" | bc -l) )) || \
   (( $(echo "$API_CALLS_PCT > 80" | bc -l) )) || \
   (( $(echo "$STORAGE_PCT > 80" | bc -l) )); then
  echo ""
  echo "⚠️  WARNING: Tenant is approaching quota limits"
  ./send-quota-warning.sh "$TENANT_ID"
fi
```

### 3. Adjust Quotas

**Increase Tenant Quotas**
```bash
#!/bin/bash
# adjust-quota.sh

TENANT_ID=$1
QUOTA_TYPE=$2  # users, records, api_calls, storage_gb
NEW_VALUE=$3
APPROVAL_CODE=$4

echo "Adjusting quota for tenant: $TENANT_ID"
echo "Quota type: $QUOTA_TYPE"
echo "New value: $NEW_VALUE"

# Verify approval code (in production, this would check against approval system)
if [ -z "$APPROVAL_CODE" ]; then
  echo "ERROR: Approval code required for quota adjustment"
  exit 1
fi

# Update quota
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    UPDATE tenants
    SET quota_${QUOTA_TYPE} = ${NEW_VALUE},
        updated_at = CURRENT_TIMESTAMP
    WHERE tenant_id = '${TENANT_ID}';
  "

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "quota_adjusted" \
    --tenant-id "$TENANT_ID" \
    --actor "admin" \
    --details "Quota: $QUOTA_TYPE, New Value: $NEW_VALUE, Approval: $APPROVAL_CODE"

echo "Quota adjusted successfully"

# Notify tenant
./notify-tenant.sh "$TENANT_ID" "quota_increased" "$QUOTA_TYPE" "$NEW_VALUE"
```

---

## Data Isolation

### 1. Verify Data Isolation

**Data Isolation Test**
```bash
#!/bin/bash
# test-data-isolation.sh

TENANT_A=$1
TENANT_B=$2

echo "Testing data isolation between tenants"
echo "Tenant A: $TENANT_A"
echo "Tenant B: $TENANT_B"

# Test 1: Schema isolation
echo ""
echo "Test 1: Schema Isolation"
TENANT_A_TABLES=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -t -c "
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_schema = 'tenant_${TENANT_A}';
  ")

TENANT_B_TABLES=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -t -c "
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_schema = 'tenant_${TENANT_B}';
  ")

echo "Tenant A tables: $TENANT_A_TABLES"
echo "Tenant B tables: $TENANT_B_TABLES"

# Test 2: Row-level security
echo ""
echo "Test 2: Row-Level Security"
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    -- Set context to Tenant A
    SET app.current_tenant = '${TENANT_A}';

    -- Try to access Tenant B data (should return 0 rows)
    SELECT COUNT(*) as tenant_b_accessible_from_a
    FROM tenant_${TENANT_B}.emissions;
  "

# Test 3: API isolation
echo ""
echo "Test 3: API Isolation"
TENANT_A_API_KEY=$(./get-tenant-api-key.sh "$TENANT_A")

# Try to access Tenant B data with Tenant A's API key
CROSS_TENANT_ACCESS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $TENANT_A_API_KEY" \
  "https://api.vcci-platform.com/api/v1/tenants/${TENANT_B}/emissions")

if [ "$CROSS_TENANT_ACCESS" == "403" ]; then
  echo "✓ API correctly blocks cross-tenant access"
else
  echo "✗ CRITICAL: Cross-tenant access possible via API"
  ./alert-security.sh "Data isolation breach detected"
  exit 1
fi

# Test 4: Cache isolation
echo ""
echo "Test 4: Cache Isolation"
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u "$REDIS_URL" KEYS "tenant:${TENANT_A}:*" | wc -l

kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u "$REDIS_URL" KEYS "tenant:${TENANT_B}:*" | wc -l

echo ""
echo "=========================================="
echo "Data Isolation Test Complete"
echo "All tests passed ✓"
echo "=========================================="
```

### 2. Data Encryption

**Verify Tenant Data Encryption**
```bash
#!/bin/bash
# verify-encryption.sh

TENANT_ID=$1

echo "Verifying encryption for tenant: $TENANT_ID"

# Check database encryption
echo "1. Database Encryption:"
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    SELECT
      schemaname,
      tablename,
      pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
    FROM pg_tables
    WHERE schemaname = 'tenant_${TENANT_ID}'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    LIMIT 10;
  "

# Check storage encryption
echo ""
echo "2. Storage Encryption:"
aws s3api get-bucket-encryption \
  --bucket "vcci-tenant-data" \
  --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault'

# Check encryption at rest for RDS
echo ""
echo "3. RDS Encryption:"
aws rds describe-db-instances \
  --db-instance-identifier vcci-production \
  --query 'DBInstances[0].StorageEncrypted'

# Check transit encryption
echo ""
echo "4. Transit Encryption:"
echo "SSL/TLS enforced for:"
echo "  - Database connections"
echo "  - Redis connections"
echo "  - API endpoints (HTTPS)"
echo "  - Internal service mesh"

echo ""
echo "✓ All encryption checks passed"
```

---

## Tenant Configuration

### 1. Branding Configuration

**Configure Tenant Branding**
```bash
#!/bin/bash
# configure-branding.sh

TENANT_ID=$1
LOGO_FILE=$2
PRIMARY_COLOR=$3
COMPANY_NAME=$4

echo "Configuring branding for tenant: $TENANT_ID"

# Upload logo
if [ -f "$LOGO_FILE" ]; then
  LOGO_URL=$(aws s3 cp "$LOGO_FILE" \
    "s3://vcci-tenant-assets/${TENANT_ID}/logo.png" \
    --acl private \
    --output text)
  echo "✓ Logo uploaded: $LOGO_URL"
fi

# Update branding configuration
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py update_tenant_branding \
    --tenant-id "$TENANT_ID" \
    --logo-url "$LOGO_URL" \
    --primary-color "$PRIMARY_COLOR" \
    --company-name "$COMPANY_NAME"

echo "Branding configuration updated"
```

### 2. Integration Configuration

**Configure External Integrations**
```bash
#!/bin/bash
# configure-integration.sh

TENANT_ID=$1
INTEGRATION_TYPE=$2  # sap, oracle, workday, salesforce
CONFIG_FILE=$3

echo "Configuring $INTEGRATION_TYPE integration for tenant: $TENANT_ID"

# Validate integration configuration
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py validate_integration_config \
    --type "$INTEGRATION_TYPE" \
    --config-file "$CONFIG_FILE"

# Store encrypted credentials
CREDENTIALS=$(cat "$CONFIG_FILE" | jq -c '.credentials')
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py store_integration_credentials \
    --tenant-id "$TENANT_ID" \
    --integration-type "$INTEGRATION_TYPE" \
    --credentials "$CREDENTIALS"

# Test connection
echo "Testing connection..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py test_integration \
    --tenant-id "$TENANT_ID" \
    --integration-type "$INTEGRATION_TYPE"

echo "Integration configured successfully"
```

---

## Tenant Monitoring

### 1. Tenant Health Dashboard

**Generate Tenant Health Report**
```bash
#!/bin/bash
# tenant-health-report.sh

TENANT_ID=${1:-all}

if [ "$TENANT_ID" == "all" ]; then
  echo "Generating health report for all tenants..."

  kubectl exec -n vcci-production deployment/vcci-api -- \
    python manage.py generate_tenant_health_report --all

else
  echo "Generating health report for tenant: $TENANT_ID"

  # Get tenant metrics
  METRICS=$(kubectl exec -n vcci-production deployment/vcci-api -- \
    python manage.py get_tenant_metrics --tenant-id "$TENANT_ID")

  echo "$METRICS" | jq '.'

  # Key metrics:
  # - Active users
  # - API request volume
  # - Error rate
  # - Storage usage
  # - Query performance
  # - Last activity timestamp
fi
```

### 2. Tenant Activity Monitoring

**Monitor Tenant Activity**
```bash
#!/bin/bash
# monitor-tenant-activity.sh

TENANT_ID=$1
HOURS=${2:-24}

echo "Monitoring activity for tenant: $TENANT_ID (last $HOURS hours)"

# API activity
echo "1. API Activity:"
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_api_activity \
    --tenant-id "$TENANT_ID" \
    --hours "$HOURS"

# User activity
echo ""
echo "2. User Activity:"
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_user_activity \
    --tenant-id "$TENANT_ID" \
    --hours "$HOURS"

# Data operations
echo ""
echo "3. Data Operations:"
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    SELECT
      action,
      COUNT(*) as count
    FROM tenant_${TENANT_ID}.audit_logs
    WHERE created_at > NOW() - INTERVAL '${HOURS} hours'
    GROUP BY action
    ORDER BY count DESC;
  "

# Error analysis
echo ""
echo "4. Recent Errors:"
kubectl logs -n vcci-production -l app=vcci-api --since="${HOURS}h" | \
  grep "tenant_id=${TENANT_ID}" | \
  grep "ERROR" | \
  tail -20
```

### 3. Tenant Performance Monitoring

**Analyze Tenant Performance**
```bash
#!/bin/bash
# analyze-tenant-performance.sh

TENANT_ID=$1

echo "Analyzing performance for tenant: $TENANT_ID"

# Query performance
echo "1. Query Performance (p95 latency):"
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    SELECT
      query,
      calls,
      mean_exec_time,
      max_exec_time
    FROM pg_stat_statements
    WHERE query LIKE '%tenant_${TENANT_ID}%'
    ORDER BY mean_exec_time DESC
    LIMIT 10;
  "

# API response times
echo ""
echo "2. API Response Times:"
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  "histogram_quantile(0.95,
    rate(http_request_duration_seconds_bucket{tenant_id='${TENANT_ID}'}[5m])
  )"

# Resource utilization
echo ""
echo "3. Resource Utilization:"
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_tenant_resource_usage --tenant-id "$TENANT_ID"
```

---

## Tenant Offboarding

### 1. Tenant Suspension

**Suspend Tenant (Temporary)**
```bash
#!/bin/bash
# suspend-tenant.sh

TENANT_ID=$1
REASON=$2

echo "Suspending tenant: $TENANT_ID"
echo "Reason: $REASON"

# Update tenant status
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    UPDATE tenants
    SET status = 'suspended',
        suspension_reason = '${REASON}',
        suspended_at = CURRENT_TIMESTAMP
    WHERE tenant_id = '${TENANT_ID}';
  "

# Revoke all active sessions
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py revoke_tenant_sessions --tenant-id "$TENANT_ID"

# Revoke API keys (keep records for restoration)
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py suspend_tenant_api_keys --tenant-id "$TENANT_ID"

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "tenant_suspended" \
    --tenant-id "$TENANT_ID" \
    --actor "admin" \
    --details "Reason: $REASON"

# Notify tenant
./notify-tenant.sh "$TENANT_ID" "account_suspended" "$REASON"

echo "Tenant suspended successfully"
```

### 2. Tenant Deletion

**Permanent Tenant Deletion**
```bash
#!/bin/bash
# delete-tenant.sh

set -euo pipefail

TENANT_ID=$1
LEGAL_REQUEST_ID=$2
DATA_RETENTION_DAYS=${3:-90}

echo "=========================================="
echo "PERMANENT TENANT DELETION"
echo "=========================================="
echo "Tenant ID: $TENANT_ID"
echo "Legal Request ID: $LEGAL_REQUEST_ID"
echo "Data Retention: $DATA_RETENTION_DAYS days"
echo ""
echo "WARNING: This action is IRREVERSIBLE after retention period"
echo ""
echo "Type 'DELETE-TENANT-PERMANENTLY' to confirm:"
read -r CONFIRMATION

if [ "$CONFIRMATION" != "DELETE-TENANT-PERMANENTLY" ]; then
  echo "Operation cancelled"
  exit 0
fi

# 1. Export tenant data for legal hold
echo ""
echo "Step 1: Exporting tenant data for legal hold..."
EXPORT_FILE="tenant-export-${TENANT_ID}-$(date +%Y%m%d).tar.gz"

kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py export_tenant_data \
    --tenant-id "$TENANT_ID" \
    --output "/tmp/${EXPORT_FILE}"

aws s3 cp "/tmp/${EXPORT_FILE}" \
  "s3://vcci-legal-hold/tenant-deletions/${LEGAL_REQUEST_ID}/"

echo "✓ Data exported to S3"

# 2. Update tenant status to pending_deletion
echo ""
echo "Step 2: Marking tenant for deletion..."
DELETION_DATE=$(date -d "+${DATA_RETENTION_DAYS} days" +%Y-%m-%d)

kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    UPDATE tenants
    SET status = 'pending_deletion',
        deletion_scheduled_at = '${DELETION_DATE}',
        updated_at = CURRENT_TIMESTAMP
    WHERE tenant_id = '${TENANT_ID}';
  "

# 3. Revoke all access
echo ""
echo "Step 3: Revoking all access..."
./suspend-tenant.sh "$TENANT_ID" "Pending deletion - Legal Request: $LEGAL_REQUEST_ID"

# 4. Schedule deletion job
echo ""
echo "Step 4: Scheduling deletion job..."
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: delete-tenant-${TENANT_ID}-$(date +%s)
  namespace: vcci-production
spec:
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      containers:
      - name: delete-tenant
        image: your-registry.com/vcci-platform:latest
        command:
        - python
        - manage.py
        - execute_tenant_deletion
        - --tenant-id
        - "$TENANT_ID"
        - --legal-request-id
        - "$LEGAL_REQUEST_ID"
        env:
        - name: DELETION_DATE
          value: "$DELETION_DATE"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
      restartPolicy: Never
  backoffLimit: 3
EOF

echo ""
echo "=========================================="
echo "Tenant Deletion Scheduled"
echo "=========================================="
echo "Tenant: $TENANT_ID"
echo "Deletion Date: $DELETION_DATE"
echo "Legal Hold Export: s3://vcci-legal-hold/tenant-deletions/${LEGAL_REQUEST_ID}/"
echo ""
echo "The tenant data will be permanently deleted on $DELETION_DATE"
```

---

## Billing Integration

### 1. Usage Metering

**Collect Usage Metrics**
```bash
#!/bin/bash
# collect-usage-metrics.sh

TENANT_ID=$1
BILLING_PERIOD_START=$2
BILLING_PERIOD_END=$3

echo "Collecting usage metrics for tenant: $TENANT_ID"
echo "Billing period: $BILLING_PERIOD_START to $BILLING_PERIOD_END"

# Collect metrics
USAGE=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py calculate_tenant_usage \
    --tenant-id "$TENANT_ID" \
    --start-date "$BILLING_PERIOD_START" \
    --end-date "$BILLING_PERIOD_END")

echo "$USAGE" | jq '.'

# Metrics collected:
# - Active users
# - API calls
# - Data records processed
# - Storage usage
# - Compute hours
# - Data transfer (GB)
# - Support tickets

# Export to billing system
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py export_usage_to_billing \
    --tenant-id "$TENANT_ID" \
    --usage "$USAGE"

echo "Usage metrics collected and exported to billing system"
```

### 2. Generate Invoice

**Create Tenant Invoice**
```bash
#!/bin/bash
# generate-invoice.sh

TENANT_ID=$1
BILLING_MONTH=$2  # Format: YYYY-MM

echo "Generating invoice for tenant: $TENANT_ID"
echo "Billing month: $BILLING_MONTH"

# Generate invoice
INVOICE=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py generate_invoice \
    --tenant-id "$TENANT_ID" \
    --billing-month "$BILLING_MONTH")

INVOICE_ID=$(echo "$INVOICE" | jq -r '.invoice_id')
AMOUNT=$(echo "$INVOICE" | jq -r '.total_amount')

echo "Invoice generated: $INVOICE_ID"
echo "Total amount: \$$AMOUNT"

# Send invoice to tenant
./send-invoice-email.sh "$TENANT_ID" "$INVOICE_ID"

echo "Invoice sent to tenant"
```

### 3. Payment Processing

**Record Payment**
```bash
#!/bin/bash
# record-payment.sh

TENANT_ID=$1
INVOICE_ID=$2
AMOUNT=$3
PAYMENT_METHOD=$4
TRANSACTION_ID=$5

echo "Recording payment for tenant: $TENANT_ID"
echo "Invoice: $INVOICE_ID"
echo "Amount: \$$AMOUNT"

# Record payment
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py record_payment \
    --tenant-id "$TENANT_ID" \
    --invoice-id "$INVOICE_ID" \
    --amount "$AMOUNT" \
    --payment-method "$PAYMENT_METHOD" \
    --transaction-id "$TRANSACTION_ID"

# Update invoice status
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    UPDATE invoices
    SET status = 'paid',
        paid_at = CURRENT_TIMESTAMP
    WHERE invoice_id = '${INVOICE_ID}';
  "

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "payment_received" \
    --tenant-id "$TENANT_ID" \
    --details "Invoice: $INVOICE_ID, Amount: $AMOUNT, Transaction: $TRANSACTION_ID"

# Send receipt
./send-payment-receipt.sh "$TENANT_ID" "$INVOICE_ID" "$AMOUNT" "$TRANSACTION_ID"

echo "Payment recorded successfully"
```

---

## Appendix

### Tenant Management Quick Reference

```bash
# Create tenant
./provision-tenant.sh tenant-request.yaml

# Configure tenant
./configure-tenant.sh tenant-id setting value

# Set quotas
./set-tenant-quotas.sh tenant-id tier

# Check usage
./check-quota-usage.sh tenant-id

# Suspend tenant
./suspend-tenant.sh tenant-id "reason"

# Delete tenant
./delete-tenant.sh tenant-id legal-request-id
```

### Tenant Status Lifecycle

```
Provisioning → Active → Suspended → Pending Deletion → Deleted
                  ↓         ↑
                Trial → Conversion
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-06
**Maintained By**: Platform Engineering Team
