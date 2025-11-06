# GL-VCCI Scope 3 Platform v2.0 - User Management Guide

## Table of Contents

1. [Overview](#overview)
2. [User Lifecycle Management](#user-lifecycle-management)
3. [Role-Based Access Control](#role-based-access-control)
4. [Multi-Tenant User Isolation](#multi-tenant-user-isolation)
5. [API Key Management](#api-key-management)
6. [Authentication and Authorization](#authentication-and-authorization)
7. [Audit Logging](#audit-logging)
8. [Self-Service Operations](#self-service-operations)

---

## Overview

### Purpose
This guide provides comprehensive procedures for managing users in the GL-VCCI Scope 3 Carbon Intelligence Platform.

### User Management Principles
- **Least Privilege**: Users receive minimum permissions needed
- **Separation of Duties**: Critical operations require multiple approvals
- **Multi-Tenant Isolation**: Users can only access their tenant's data
- **Audit Trail**: All user actions are logged
- **Self-Service**: Users can manage certain aspects themselves

### User Types
```yaml
System Administrator:
  Description: Platform-wide administration
  Access Level: Full system access
  Tenant Scope: All tenants

Tenant Administrator:
  Description: Tenant-level administration
  Access Level: Full tenant access
  Tenant Scope: Single tenant

Power User:
  Description: Advanced tenant operations
  Access Level: Read/Write tenant data
  Tenant Scope: Single tenant

Standard User:
  Description: Regular operations
  Access Level: Read tenant data, limited writes
  Tenant Scope: Single tenant

API User:
  Description: Programmatic access
  Access Level: API-specific permissions
  Tenant Scope: Single tenant

Read-Only User:
  Description: View-only access
  Access Level: Read-only
  Tenant Scope: Single tenant
```

---

## User Lifecycle Management

### 1. User Onboarding

#### Step 1: Create User Request
```bash
# Create user request form
cat > user-request.yaml <<EOF
user_request:
  # User Information
  first_name: "John"
  last_name: "Doe"
  email: "john.doe@company.com"
  employee_id: "EMP12345"
  department: "Sustainability"
  manager_email: "manager@company.com"

  # Tenant Assignment
  tenant_id: "acme-corp"
  tenant_name: "Acme Corporation"

  # Access Requirements
  role: "power_user"
  permissions:
    - "read_emissions"
    - "write_emissions"
    - "read_analytics"
    - "manage_suppliers"

  # Justification
  business_justification: "Responsible for Scope 3 emissions tracking"
  access_duration: "permanent"  # or specific end date
  requested_by: "manager@company.com"
  request_date: "2025-01-06"
EOF
```

#### Step 2: Review and Approve Request
```bash
#!/bin/bash
# review-user-request.sh

REQUEST_FILE=$1

echo "Reviewing user request..."

# Parse request
FIRST_NAME=$(yq eval '.user_request.first_name' "$REQUEST_FILE")
LAST_NAME=$(yq eval '.user_request.last_name' "$REQUEST_FILE")
EMAIL=$(yq eval '.user_request.email' "$REQUEST_FILE")
TENANT_ID=$(yq eval '.user_request.tenant_id' "$REQUEST_FILE")
ROLE=$(yq eval '.user_request.role' "$REQUEST_FILE")

echo "User: $FIRST_NAME $LAST_NAME"
echo "Email: $EMAIL"
echo "Tenant: $TENANT_ID"
echo "Role: $ROLE"

# Validate tenant exists
TENANT_EXISTS=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py check_tenant --tenant-id "$TENANT_ID" || echo "false")

if [ "$TENANT_EXISTS" != "true" ]; then
  echo "ERROR: Tenant $TENANT_ID does not exist"
  exit 1
fi

# Check for existing user
EXISTING_USER=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py check_user --email "$EMAIL" || echo "false")

if [ "$EXISTING_USER" == "true" ]; then
  echo "WARNING: User with email $EMAIL already exists"
fi

echo ""
echo "Request validation passed"
echo "Approve this request? (yes/no)"
read -r APPROVAL

if [ "$APPROVAL" != "yes" ]; then
  echo "Request rejected"
  exit 0
fi

# Create user
./create-user.sh "$REQUEST_FILE"
```

#### Step 3: Create User Account
```bash
#!/bin/bash
# create-user.sh

set -euo pipefail

REQUEST_FILE=$1

# Parse request
FIRST_NAME=$(yq eval '.user_request.first_name' "$REQUEST_FILE")
LAST_NAME=$(yq eval '.user_request.last_name' "$REQUEST_FILE")
EMAIL=$(yq eval '.user_request.email' "$REQUEST_FILE")
TENANT_ID=$(yq eval '.user_request.tenant_id' "$REQUEST_FILE")
ROLE=$(yq eval '.user_request.role' "$REQUEST_FILE")
PERMISSIONS=$(yq eval '.user_request.permissions | join(",")' "$REQUEST_FILE")

echo "Creating user account..."

# Generate temporary password
TEMP_PASSWORD=$(openssl rand -base64 16)

# Create user in database
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py create_user \
    --email "$EMAIL" \
    --first-name "$FIRST_NAME" \
    --last-name "$LAST_NAME" \
    --tenant-id "$TENANT_ID" \
    --role "$ROLE" \
    --permissions "$PERMISSIONS" \
    --password "$TEMP_PASSWORD" \
    --require-password-change

USER_ID=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_user_id --email "$EMAIL")

echo "User created successfully"
echo "User ID: $USER_ID"
echo ""

# Send welcome email
./send-welcome-email.sh "$EMAIL" "$TEMP_PASSWORD"

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "user_created" \
    --user-id "$USER_ID" \
    --actor "system" \
    --details "User onboarded via create-user.sh"

echo "Onboarding complete"
```

#### Step 4: Send Welcome Email
```bash
#!/bin/bash
# send-welcome-email.sh

EMAIL=$1
TEMP_PASSWORD=$2

cat > /tmp/welcome-email.html <<EOF
<!DOCTYPE html>
<html>
<head>
  <title>Welcome to VCCI Platform</title>
</head>
<body>
  <h1>Welcome to the VCCI Scope 3 Carbon Intelligence Platform</h1>

  <p>Your account has been created. Please use the following credentials to log in:</p>

  <ul>
    <li><strong>Email:</strong> $EMAIL</li>
    <li><strong>Temporary Password:</strong> $TEMP_PASSWORD</li>
    <li><strong>Login URL:</strong> https://app.vcci-platform.com</li>
  </ul>

  <p><strong>Important:</strong> You will be required to change your password on first login.</p>

  <h2>Getting Started</h2>
  <ul>
    <li>Review the <a href="https://docs.vcci-platform.com/user-guide">User Guide</a></li>
    <li>Watch the <a href="https://docs.vcci-platform.com/training">Training Videos</a></li>
    <li>Join our <a href="https://slack.com/vcci-users">Slack Community</a></li>
  </ul>

  <p>If you have any questions, contact support@vcci-platform.com</p>
</body>
</html>
EOF

# Send email via AWS SES
aws ses send-email \
  --from "noreply@vcci-platform.com" \
  --to "$EMAIL" \
  --subject "Welcome to VCCI Platform" \
  --html file:///tmp/welcome-email.html

echo "Welcome email sent to $EMAIL"
```

### 2. User Modification

#### Update User Profile
```bash
#!/bin/bash
# update-user.sh

USER_EMAIL=$1
FIELD=$2
VALUE=$3

echo "Updating user: $USER_EMAIL"
echo "Field: $FIELD"
echo "Value: $VALUE"

# Validate field
ALLOWED_FIELDS=("first_name" "last_name" "department" "phone")
if [[ ! " ${ALLOWED_FIELDS[@]} " =~ " ${FIELD} " ]]; then
  echo "ERROR: Invalid field. Allowed: ${ALLOWED_FIELDS[*]}"
  exit 1
fi

# Update user
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py update_user \
    --email "$USER_EMAIL" \
    --field "$FIELD" \
    --value "$VALUE"

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "user_updated" \
    --user-email "$USER_EMAIL" \
    --actor "admin" \
    --details "Updated $FIELD to $VALUE"

echo "User updated successfully"
```

#### Change User Role
```bash
#!/bin/bash
# change-user-role.sh

USER_EMAIL=$1
NEW_ROLE=$2
APPROVAL_REQUIRED=${3:-true}

echo "Changing role for user: $USER_EMAIL"
echo "New role: $NEW_ROLE"

# Get current role
CURRENT_ROLE=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_user_role --email "$USER_EMAIL")

echo "Current role: $CURRENT_ROLE"

# Validate role
VALID_ROLES=("system_admin" "tenant_admin" "power_user" "standard_user" "read_only")
if [[ ! " ${VALID_ROLES[@]} " =~ " ${NEW_ROLE} " ]]; then
  echo "ERROR: Invalid role. Valid roles: ${VALID_ROLES[*]}"
  exit 1
fi

# Check if approval required
if [ "$APPROVAL_REQUIRED" == "true" ]; then
  echo ""
  echo "This action requires manager approval."
  echo "Manager email:"
  read -r MANAGER_EMAIL

  echo "Approval reason:"
  read -r REASON

  # Send approval request
  ./send-approval-request.sh "$MANAGER_EMAIL" "$USER_EMAIL" "$NEW_ROLE" "$REASON"

  echo "Approval request sent. Waiting for approval..."
  # In production, this would wait for async approval
  echo "Approval code:"
  read -r APPROVAL_CODE

  # Validate approval code
  # ... validation logic ...
fi

# Change role
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py change_user_role \
    --email "$USER_EMAIL" \
    --role "$NEW_ROLE"

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "role_changed" \
    --user-email "$USER_EMAIL" \
    --actor "admin" \
    --details "Role changed from $CURRENT_ROLE to $NEW_ROLE"

echo "Role changed successfully"
```

### 3. User Offboarding

#### Disable User Account
```bash
#!/bin/bash
# disable-user.sh

set -euo pipefail

USER_EMAIL=$1
REASON=$2

echo "Disabling user account: $USER_EMAIL"
echo "Reason: $REASON"

# Get user details
USER_INFO=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_user_info --email "$USER_EMAIL")

echo "User info:"
echo "$USER_INFO"

echo ""
echo "Are you sure you want to disable this account? (yes/no)"
read -r CONFIRMATION

if [ "$CONFIRMATION" != "yes" ]; then
  echo "Operation cancelled"
  exit 0
fi

# 1. Disable user account
echo "Step 1: Disabling account..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py disable_user --email "$USER_EMAIL"

# 2. Revoke all active sessions
echo "Step 2: Revoking active sessions..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py revoke_user_sessions --email "$USER_EMAIL"

# 3. Revoke API keys
echo "Step 3: Revoking API keys..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py revoke_user_api_keys --email "$USER_EMAIL"

# 4. Remove from shared resources
echo "Step 4: Removing from shared resources..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py remove_user_from_groups --email "$USER_EMAIL"

# 5. Archive user data
echo "Step 5: Archiving user data..."
ARCHIVE_ID=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py archive_user_data --email "$USER_EMAIL")

echo "User data archived: $ARCHIVE_ID"

# 6. Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "user_disabled" \
    --user-email "$USER_EMAIL" \
    --actor "admin" \
    --details "Reason: $REASON. Archive ID: $ARCHIVE_ID"

echo ""
echo "User account disabled successfully"
echo "Archive ID: $ARCHIVE_ID"
```

#### Delete User Account (GDPR)
```bash
#!/bin/bash
# delete-user.sh

set -euo pipefail

USER_EMAIL=$1
LEGAL_REQUEST_ID=$2

echo "=========================================="
echo "PERMANENT USER DELETION (GDPR)"
echo "=========================================="
echo "User: $USER_EMAIL"
echo "Legal Request ID: $LEGAL_REQUEST_ID"
echo ""
echo "WARNING: This action is IRREVERSIBLE"
echo "All user data will be permanently deleted"
echo ""
echo "Type 'DELETE-USER-PERMANENTLY' to confirm:"
read -r CONFIRMATION

if [ "$CONFIRMATION" != "DELETE-USER-PERMANENTLY" ]; then
  echo "Operation cancelled"
  exit 0
fi

# 1. Create deletion audit record
echo "Step 1: Creating audit record..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "user_deletion_initiated" \
    --user-email "$USER_EMAIL" \
    --actor "admin" \
    --details "Legal Request ID: $LEGAL_REQUEST_ID"

# 2. Export user data for legal hold
echo "Step 2: Exporting user data for legal hold..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py export_user_data --email "$USER_EMAIL" \
    --output "/tmp/user-export-$LEGAL_REQUEST_ID.json"

# Upload to secure storage
aws s3 cp "/tmp/user-export-$LEGAL_REQUEST_ID.json" \
  "s3://vcci-legal-hold/user-deletions/$LEGAL_REQUEST_ID/"

# 3. Delete user data from database
echo "Step 3: Deleting user data from database..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py delete_user_data --email "$USER_EMAIL"

# 4. Delete user account
echo "Step 4: Deleting user account..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py delete_user --email "$USER_EMAIL"

# 5. Purge from cache
echo "Step 5: Purging from cache..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u "$REDIS_URL" DEL "user:$USER_EMAIL:*"

# 6. Final audit log
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "user_deleted" \
    --details "User: $USER_EMAIL, Legal Request ID: $LEGAL_REQUEST_ID" \
    --actor "admin"

echo ""
echo "User deletion complete"
echo "Legal hold export: s3://vcci-legal-hold/user-deletions/$LEGAL_REQUEST_ID/"
```

---

## Role-Based Access Control

### 1. Role Definitions

**System Roles Configuration**
```yaml
# config/roles.yaml
roles:
  system_admin:
    name: "System Administrator"
    description: "Full system access across all tenants"
    permissions:
      - "*:*:*"  # All permissions
    assignable_by:
      - system_admin
    max_users: 5

  tenant_admin:
    name: "Tenant Administrator"
    description: "Full access within tenant"
    permissions:
      - "tenant:*:*"
      - "user:read:tenant"
      - "user:create:tenant"
      - "user:update:tenant"
      - "emission:*:tenant"
      - "analytics:*:tenant"
      - "supplier:*:tenant"
      - "report:*:tenant"
    assignable_by:
      - system_admin
      - tenant_admin
    max_users: 10

  power_user:
    name: "Power User"
    description: "Advanced operations within tenant"
    permissions:
      - "emission:read:tenant"
      - "emission:create:tenant"
      - "emission:update:tenant"
      - "analytics:read:tenant"
      - "supplier:read:tenant"
      - "supplier:update:tenant"
      - "report:read:tenant"
      - "report:create:tenant"
    assignable_by:
      - system_admin
      - tenant_admin
    max_users: 50

  standard_user:
    name: "Standard User"
    description: "Regular operations"
    permissions:
      - "emission:read:tenant"
      - "emission:create:tenant"
      - "analytics:read:tenant"
      - "supplier:read:tenant"
      - "report:read:tenant"
    assignable_by:
      - system_admin
      - tenant_admin
    max_users: 200

  read_only:
    name: "Read-Only User"
    description: "View-only access"
    permissions:
      - "emission:read:tenant"
      - "analytics:read:tenant"
      - "supplier:read:tenant"
      - "report:read:tenant"
    assignable_by:
      - system_admin
      - tenant_admin
    max_users: unlimited

  api_user:
    name: "API User"
    description: "Programmatic access via API"
    permissions:
      - "api:read:tenant"
      - "api:write:tenant"
    assignable_by:
      - system_admin
      - tenant_admin
    max_users: 20
```

### 2. Permission Management

**Check User Permissions**
```bash
#!/bin/bash
# check-permissions.sh

USER_EMAIL=$1
ACTION=$2
RESOURCE=$3

echo "Checking permissions for: $USER_EMAIL"
echo "Action: $ACTION"
echo "Resource: $RESOURCE"

# Get user permissions
HAS_PERMISSION=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py check_permission \
    --email "$USER_EMAIL" \
    --action "$ACTION" \
    --resource "$RESOURCE")

if [ "$HAS_PERMISSION" == "true" ]; then
  echo "✓ User has permission"
  exit 0
else
  echo "✗ User does NOT have permission"
  exit 1
fi
```

**Grant Custom Permission**
```bash
#!/bin/bash
# grant-permission.sh

USER_EMAIL=$1
PERMISSION=$2
EXPIRATION=$3  # Optional: YYYY-MM-DD or "permanent"

echo "Granting permission: $PERMISSION"
echo "To user: $USER_EMAIL"
echo "Expiration: ${EXPIRATION:-permanent}"

# Grant permission
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py grant_permission \
    --email "$USER_EMAIL" \
    --permission "$PERMISSION" \
    --expiration "${EXPIRATION:-permanent}"

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "permission_granted" \
    --user-email "$USER_EMAIL" \
    --actor "admin" \
    --details "Permission: $PERMISSION, Expiration: ${EXPIRATION:-permanent}"

echo "Permission granted successfully"
```

### 3. Role Templates

**Create Role Template**
```bash
#!/bin/bash
# create-role-template.sh

TEMPLATE_NAME=$1

cat > "/tmp/role-template-${TEMPLATE_NAME}.yaml" <<EOF
role_template:
  name: "$TEMPLATE_NAME"
  description: "Custom role template"
  permissions:
    - "emission:read:tenant"
    - "analytics:read:tenant"
  assignable_by:
    - "system_admin"
    - "tenant_admin"
  max_users: 100
  expires_after_days: null
EOF

echo "Role template created: /tmp/role-template-${TEMPLATE_NAME}.yaml"
echo "Edit the file and then apply with: ./apply-role-template.sh"
```

---

## Multi-Tenant User Isolation

### 1. Tenant Isolation Enforcement

**Row-Level Security (RLS)**
```sql
-- Enable RLS on user tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Create RLS policy
CREATE POLICY tenant_isolation_policy ON users
  USING (tenant_id = current_setting('app.current_tenant'));

-- Create function to set tenant context
CREATE OR REPLACE FUNCTION set_tenant_context(tenant_id TEXT)
RETURNS VOID AS $$
BEGIN
  PERFORM set_config('app.current_tenant', tenant_id, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

**Verify Tenant Isolation**
```bash
#!/bin/bash
# verify-tenant-isolation.sh

USER_EMAIL=$1

echo "Verifying tenant isolation for: $USER_EMAIL"

# Get user's tenant
USER_TENANT=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_user_tenant --email "$USER_EMAIL")

echo "User tenant: $USER_TENANT"

# Test cross-tenant access
echo "Testing cross-tenant access prevention..."

# Try to access another tenant's data
OTHER_TENANT="other-tenant-id"

CAN_ACCESS=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py test_cross_tenant_access \
    --user-email "$USER_EMAIL" \
    --target-tenant "$OTHER_TENANT" || echo "false")

if [ "$CAN_ACCESS" == "false" ]; then
  echo "✓ Tenant isolation is working correctly"
else
  echo "✗ WARNING: Cross-tenant access detected!"
  # Alert security team
  ./alert-security.sh "Cross-tenant access detected for $USER_EMAIL"
fi
```

### 2. Data Segregation

**Tenant-Specific Schemas**
```bash
#!/bin/bash
# verify-data-segregation.sh

TENANT_ID=$1

echo "Verifying data segregation for tenant: $TENANT_ID"

# Check schema isolation
SCHEMA_EXISTS=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -t -c \
    "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'tenant_${TENANT_ID}');")

if [ "$SCHEMA_EXISTS" == " t" ]; then
  echo "✓ Tenant schema exists"
else
  echo "✗ Tenant schema does NOT exist"
  exit 1
fi

# Verify RLS policies
RLS_ENABLED=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -t -c \
    "SELECT rowsecurity FROM pg_tables WHERE schemaname = 'tenant_${TENANT_ID}';")

if [[ "$RLS_ENABLED" == *"t"* ]]; then
  echo "✓ Row-level security is enabled"
else
  echo "✗ Row-level security is NOT enabled"
  exit 1
fi

echo "Data segregation verification passed"
```

---

## API Key Management

### 1. Generate API Key

**Create API Key**
```bash
#!/bin/bash
# generate-api-key.sh

USER_EMAIL=$1
KEY_NAME=$2
EXPIRATION_DAYS=${3:-365}

echo "Generating API key for: $USER_EMAIL"
echo "Key name: $KEY_NAME"
echo "Expiration: $EXPIRATION_DAYS days"

# Calculate expiration date
EXPIRATION_DATE=$(date -d "+${EXPIRATION_DAYS} days" +%Y-%m-%d)

# Generate API key
API_KEY=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py generate_api_key \
    --user-email "$USER_EMAIL" \
    --name "$KEY_NAME" \
    --expiration "$EXPIRATION_DATE")

echo ""
echo "API Key generated successfully"
echo "=========================================="
echo "API Key: $API_KEY"
echo "Expires: $EXPIRATION_DATE"
echo "=========================================="
echo ""
echo "IMPORTANT: Save this key securely. It cannot be retrieved again."

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "api_key_created" \
    --user-email "$USER_EMAIL" \
    --actor "admin" \
    --details "Key name: $KEY_NAME, Expires: $EXPIRATION_DATE"
```

### 2. Revoke API Key

**Revoke API Key**
```bash
#!/bin/bash
# revoke-api-key.sh

KEY_ID=$1
REASON=$2

echo "Revoking API key: $KEY_ID"
echo "Reason: $REASON"

# Get key details
KEY_INFO=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_api_key_info --key-id "$KEY_ID")

echo "Key info:"
echo "$KEY_INFO"

# Revoke key
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py revoke_api_key --key-id "$KEY_ID"

# Purge from cache
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u "$REDIS_URL" DEL "api_key:$KEY_ID"

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "api_key_revoked" \
    --details "Key ID: $KEY_ID, Reason: $REASON" \
    --actor "admin"

echo "API key revoked successfully"
```

### 3. API Key Rotation

**Rotate API Key**
```bash
#!/bin/bash
# rotate-api-key.sh

OLD_KEY_ID=$1

echo "Rotating API key: $OLD_KEY_ID"

# Get old key details
USER_EMAIL=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_api_key_user --key-id "$OLD_KEY_ID")

KEY_NAME=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_api_key_name --key-id "$OLD_KEY_ID")

echo "User: $USER_EMAIL"
echo "Key name: $KEY_NAME"

# Generate new key
NEW_API_KEY=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py generate_api_key \
    --user-email "$USER_EMAIL" \
    --name "${KEY_NAME}-rotated" \
    --expiration "$(date -d '+365 days' +%Y-%m-%d)")

# Mark old key for deprecation (30-day grace period)
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py deprecate_api_key \
    --key-id "$OLD_KEY_ID" \
    --grace-period-days 30

echo ""
echo "API Key rotated successfully"
echo "=========================================="
echo "New API Key: $NEW_API_KEY"
echo "Old key will be revoked in 30 days"
echo "=========================================="

# Send notification email
./send-key-rotation-notification.sh "$USER_EMAIL" "$NEW_API_KEY"
```

---

## Authentication and Authorization

### 1. Multi-Factor Authentication (MFA)

**Enable MFA for User**
```bash
#!/bin/bash
# enable-mfa.sh

USER_EMAIL=$1

echo "Enabling MFA for: $USER_EMAIL"

# Generate MFA secret
MFA_SECRET=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py generate_mfa_secret --email "$USER_EMAIL")

# Generate QR code URL
QR_CODE_URL=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py generate_mfa_qr --email "$USER_EMAIL" --secret "$MFA_SECRET")

echo "MFA Secret: $MFA_SECRET"
echo "QR Code URL: $QR_CODE_URL"

# Send setup instructions
./send-mfa-setup-email.sh "$USER_EMAIL" "$QR_CODE_URL"

echo "MFA setup instructions sent to user"
```

**Enforce MFA for Role**
```bash
#!/bin/bash
# enforce-mfa-for-role.sh

ROLE=$1

echo "Enforcing MFA for role: $ROLE"

# Update role configuration
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py update_role \
    --role "$ROLE" \
    --require-mfa true

# Get users with this role
USERS=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_users_by_role --role "$ROLE")

# Notify users
while IFS= read -r user_email; do
  echo "Notifying $user_email about MFA requirement"
  ./send-mfa-requirement-email.sh "$user_email"
done <<< "$USERS"

echo "MFA enforcement enabled for role: $ROLE"
```

### 2. Session Management

**View Active Sessions**
```bash
#!/bin/bash
# view-active-sessions.sh

USER_EMAIL=${1:-all}

if [ "$USER_EMAIL" == "all" ]; then
  echo "Viewing all active sessions..."
  kubectl exec -n vcci-production deployment/vcci-api -- \
    python manage.py list_active_sessions
else
  echo "Viewing active sessions for: $USER_EMAIL"
  kubectl exec -n vcci-production deployment/vcci-api -- \
    python manage.py list_active_sessions --email "$USER_EMAIL"
fi
```

**Terminate Session**
```bash
#!/bin/bash
# terminate-session.sh

SESSION_ID=$1
REASON=$2

echo "Terminating session: $SESSION_ID"
echo "Reason: $REASON"

# Get session details
SESSION_INFO=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py get_session_info --session-id "$SESSION_ID")

echo "Session info:"
echo "$SESSION_INFO"

# Terminate session
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py terminate_session --session-id "$SESSION_ID"

# Purge from cache
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u "$REDIS_URL" DEL "session:$SESSION_ID"

# Log action
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py log_audit \
    --action "session_terminated" \
    --details "Session ID: $SESSION_ID, Reason: $REASON" \
    --actor "admin"

echo "Session terminated successfully"
```

---

## Audit Logging

### 1. Audit Log Configuration

**Audit Events**
```yaml
# config/audit-events.yaml
audit_events:
  user_events:
    - user_created
    - user_updated
    - user_disabled
    - user_deleted
    - role_changed
    - permission_granted
    - permission_revoked

  authentication_events:
    - login_success
    - login_failed
    - logout
    - password_changed
    - password_reset_requested
    - mfa_enabled
    - mfa_disabled
    - session_expired

  api_key_events:
    - api_key_created
    - api_key_revoked
    - api_key_rotated
    - api_key_expired

  data_access_events:
    - data_read
    - data_created
    - data_updated
    - data_deleted
    - data_exported

  administrative_events:
    - config_changed
    - tenant_created
    - tenant_updated
    - backup_initiated
    - restore_initiated
```

### 2. Query Audit Logs

**Search Audit Logs**
```bash
#!/bin/bash
# search-audit-logs.sh

ACTION=$1
USER_EMAIL=$2
START_DATE=$3
END_DATE=$4

echo "Searching audit logs..."
echo "Action: ${ACTION:-all}"
echo "User: ${USER_EMAIL:-all}"
echo "Date range: $START_DATE to $END_DATE"

# Build query
QUERY="SELECT * FROM audit_logs WHERE created_at BETWEEN '$START_DATE' AND '$END_DATE'"

if [ -n "$ACTION" ] && [ "$ACTION" != "all" ]; then
  QUERY="$QUERY AND action = '$ACTION'"
fi

if [ -n "$USER_EMAIL" ] && [ "$USER_EMAIL" != "all" ]; then
  QUERY="$QUERY AND user_email = '$USER_EMAIL'"
fi

QUERY="$QUERY ORDER BY created_at DESC LIMIT 1000"

# Execute query
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "$QUERY"
```

**Export Audit Logs**
```bash
#!/bin/bash
# export-audit-logs.sh

START_DATE=$1
END_DATE=$2
OUTPUT_FILE="audit-logs-${START_DATE}-to-${END_DATE}.csv"

echo "Exporting audit logs..."
echo "Date range: $START_DATE to $END_DATE"

# Export to CSV
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "
    COPY (
      SELECT
        id,
        action,
        user_email,
        actor,
        details,
        ip_address,
        user_agent,
        created_at
      FROM audit_logs
      WHERE created_at BETWEEN '$START_DATE' AND '$END_DATE'
      ORDER BY created_at DESC
    ) TO STDOUT WITH CSV HEADER
  " > "$OUTPUT_FILE"

echo "Audit logs exported to: $OUTPUT_FILE"

# Upload to secure storage
aws s3 cp "$OUTPUT_FILE" "s3://vcci-audit-logs/exports/$OUTPUT_FILE"

echo "Uploaded to S3: s3://vcci-audit-logs/exports/$OUTPUT_FILE"
```

### 3. Audit Compliance Reports

**Generate Compliance Report**
```bash
#!/bin/bash
# generate-compliance-report.sh

REPORT_TYPE=$1  # soc2, gdpr, hipaa
START_DATE=$2
END_DATE=$3

echo "Generating $REPORT_TYPE compliance report"
echo "Period: $START_DATE to $END_DATE"

kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py generate_compliance_report \
    --type "$REPORT_TYPE" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --output "/tmp/compliance-report.pdf"

# Copy report locally
kubectl cp "vcci-production/$(kubectl get pod -n vcci-production -l app=vcci-api -o jsonpath='{.items[0].metadata.name}'):/tmp/compliance-report.pdf" \
  "./compliance-report-${REPORT_TYPE}-${START_DATE}.pdf"

echo "Compliance report generated: ./compliance-report-${REPORT_TYPE}-${START_DATE}.pdf"
```

---

## Self-Service Operations

### 1. User Self-Service Portal

**Password Reset**
```bash
# User initiates password reset
POST /api/v1/auth/password-reset/request
{
  "email": "user@company.com"
}

# User receives email with reset token
# User submits new password
POST /api/v1/auth/password-reset/confirm
{
  "token": "reset-token-here",
  "new_password": "NewSecurePassword123!"
}
```

**Update Profile**
```bash
# User updates their profile
PUT /api/v1/users/me
{
  "first_name": "John",
  "last_name": "Doe",
  "phone": "+1-555-0123",
  "timezone": "America/New_York"
}
```

**Manage API Keys**
```bash
# User creates API key
POST /api/v1/users/me/api-keys
{
  "name": "Production Integration",
  "permissions": ["read_emissions", "write_emissions"],
  "expires_in_days": 90
}

# User lists their API keys
GET /api/v1/users/me/api-keys

# User revokes API key
DELETE /api/v1/users/me/api-keys/{key_id}
```

### 2. Access Request Workflow

**Request Additional Permissions**
```bash
#!/bin/bash
# request-access.sh

PERMISSION=$1
JUSTIFICATION=$2

echo "Requesting permission: $PERMISSION"
echo "Justification: $JUSTIFICATION"

# Create access request
REQUEST_ID=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py create_access_request \
    --user-email "$USER_EMAIL" \
    --permission "$PERMISSION" \
    --justification "$JUSTIFICATION")

echo "Access request created: $REQUEST_ID"
echo "Your manager will be notified for approval"

# Send notification to manager
./notify-manager.sh "$REQUEST_ID"
```

---

## Appendix

### User Management Quick Reference

**Common Commands**
```bash
# Create user
./create-user.sh user-request.yaml

# Update user
./update-user.sh user@example.com first_name "John"

# Disable user
./disable-user.sh user@example.com "Termination"

# Change role
./change-user-role.sh user@example.com power_user

# Generate API key
./generate-api-key.sh user@example.com "Integration Key" 365

# View audit logs
./search-audit-logs.sh user_created all 2025-01-01 2025-01-31
```

### Troubleshooting

**User Cannot Login**
1. Verify account is active: `check_user_status.sh user@example.com`
2. Check password attempts: `check_login_attempts.sh user@example.com`
3. Verify MFA configuration: `check_mfa_status.sh user@example.com`
4. Check audit logs: `search-audit-logs.sh login_failed user@example.com`

**Permission Denied Errors**
1. Check user role: `get_user_role.sh user@example.com`
2. Verify permissions: `check-permissions.sh user@example.com action resource`
3. Check tenant assignment: `get_user_tenant.sh user@example.com`

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-06
**Maintained By**: Platform Engineering Team
