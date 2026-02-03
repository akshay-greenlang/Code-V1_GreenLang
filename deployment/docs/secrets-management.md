# GreenLang Database Secrets Management

This document describes the secrets management infrastructure for GreenLang's PostgreSQL database cluster, including how credentials are stored, rotated, and accessed.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Secrets Inventory](#secrets-inventory)
4. [How Credentials Are Stored](#how-credentials-are-stored)
5. [Rotation Procedure](#rotation-procedure)
6. [Kubernetes Integration](#kubernetes-integration)
7. [Emergency Credential Reset](#emergency-credential-reset)
8. [Troubleshooting](#troubleshooting)
9. [Security Best Practices](#security-best-practices)

---

## Overview

GreenLang uses AWS Secrets Manager as the central secrets store for all database credentials. The External Secrets Operator synchronizes these secrets to Kubernetes, ensuring applications always have access to current credentials.

### Key Features

- **Centralized Storage**: All database credentials stored in AWS Secrets Manager
- **Automatic Rotation**: 90-day rotation cycle with Lambda-based rotation
- **KMS Encryption**: All secrets encrypted with customer-managed KMS keys
- **Kubernetes Integration**: External Secrets Operator for seamless K8s integration
- **Audit Trail**: CloudTrail logging for all secret access
- **Multi-Environment**: Separate secrets for dev, staging, and production

---

## Architecture

```
+------------------+     +----------------------+     +------------------+
|                  |     |                      |     |                  |
|  AWS Secrets     |<--->|  External Secrets    |<--->|  Kubernetes      |
|  Manager         |     |  Operator            |     |  Secrets         |
|                  |     |                      |     |                  |
+------------------+     +----------------------+     +------------------+
        ^                                                      |
        |                                                      v
+------------------+                                   +------------------+
|                  |                                   |                  |
|  Lambda          |                                   |  Application     |
|  Rotation        |                                   |  Pods            |
|                  |                                   |                  |
+------------------+                                   +------------------+
        |
        v
+------------------+
|                  |
|  PostgreSQL      |
|  Database        |
|                  |
+------------------+
```

### Components

| Component | Purpose |
|-----------|---------|
| AWS Secrets Manager | Central secrets store |
| AWS KMS | Encryption key management |
| Lambda Rotation Function | Automated password rotation |
| External Secrets Operator | Kubernetes secrets synchronization |
| SNS Topic | Rotation notifications |
| CloudWatch | Monitoring and alerting |

---

## Secrets Inventory

### Database Credentials

| Secret Name | Purpose | Rotation |
|-------------|---------|----------|
| `greenlang/database/master` | PostgreSQL superuser credentials | 90 days |
| `greenlang/database/application` | Application database user | 90 days |
| `greenlang/database/replication` | Streaming replication user | 90 days |
| `greenlang/database/pgbouncer` | PgBouncer authentication | 90 days |
| `greenlang/database/pgbackrest` | Backup encryption and S3 | No rotation |

### Secret Contents

#### Master Credentials
```json
{
  "username": "postgres",
  "password": "<generated>",
  "engine": "postgres",
  "host": "greenlang-db.cluster-xxx.region.rds.amazonaws.com",
  "port": 5432,
  "dbname": "greenlang",
  "dbClusterIdentifier": "greenlang-postgresql"
}
```

#### Application Credentials
```json
{
  "username": "greenlang_app",
  "password": "<generated>",
  "engine": "postgres",
  "host": "greenlang-db.cluster-xxx.region.rds.amazonaws.com",
  "port": 5432,
  "dbname": "greenlang",
  "connectionString": "postgresql://greenlang_app:<password>@<host>:5432/greenlang"
}
```

#### Replication Credentials
```json
{
  "username": "replicator",
  "password": "<generated>",
  "engine": "postgres",
  "host": "greenlang-db.cluster-xxx.region.rds.amazonaws.com",
  "port": 5432,
  "replication_slot": "greenlang_repl_slot"
}
```

#### PgBouncer Credentials
```json
{
  "auth_user": "pgbouncer",
  "auth_password": "<generated>",
  "userlist": "\"postgres\" \"<password>\"\n\"greenlang_app\" \"<password>\"",
  "admin_users": "postgres",
  "stats_users": "pgbouncer_stats"
}
```

#### pgBackRest Encryption
```json
{
  "repo1_cipher_pass": "<64-char-key>",
  "repo1_cipher_type": "aes-256-cbc",
  "repo1_s3_key": "<access-key>",
  "repo1_s3_key_secret": "<secret-key>",
  "repo1_s3_bucket": "greenlang-backups",
  "repo1_s3_region": "us-east-1",
  "repo1_path": "/backup",
  "retention_full": 4,
  "retention_diff": 14
}
```

---

## How Credentials Are Stored

### AWS Secrets Manager Configuration

1. **Encryption**: All secrets encrypted using customer-managed KMS key
2. **Versioning**: Secrets Manager maintains version history
3. **Staging Labels**: `AWSCURRENT`, `AWSPENDING`, `AWSPREVIOUS`
4. **Recovery Window**: 30-day recovery window before permanent deletion

### Terraform Configuration

```hcl
module "database_secrets" {
  source = "./modules/database-secrets"

  name_prefix        = "greenlang"
  secret_name_prefix = "greenlang"

  # Database configuration
  database_host = module.aurora.cluster_endpoint
  database_port = 5432
  database_name = "greenlang"

  # Rotation settings
  enable_rotation = true
  rotation_days   = 90

  # EKS integration
  eks_cluster_oidc_issuer_url = module.eks.oidc_provider_url
  eks_namespace               = "greenlang"
  eks_service_account_name    = "greenlang-db-secrets"

  # VPC for Lambda
  lambda_subnet_ids         = module.vpc.private_subnets
  lambda_security_group_ids = [module.vpc.default_security_group_id]
}
```

---

## Rotation Procedure

### Automatic Rotation (Every 90 Days)

The rotation process follows AWS Secrets Manager's four-step protocol:

#### Step 1: createSecret
- Generate new password meeting complexity requirements
- Store as new secret version with `AWSPENDING` label

#### Step 2: setSecret
- Connect to PostgreSQL using master credentials
- Execute `ALTER USER ... WITH PASSWORD ...`
- Update password in database

#### Step 3: testSecret
- Connect to PostgreSQL using new credentials
- Verify connection successful
- Execute test query

#### Step 4: finishSecret
- Move `AWSCURRENT` label to new version
- Move `AWSPREVIOUS` label to old version
- Send SNS notification

### Manual Rotation

```bash
# Rotate a specific secret immediately
aws secretsmanager rotate-secret \
  --secret-id greenlang/database/application \
  --rotation-lambda-arn arn:aws:lambda:us-east-1:123456789:function:greenlang-secrets-rotation

# Check rotation status
aws secretsmanager describe-secret \
  --secret-id greenlang/database/application \
  --query 'RotationEnabled'
```

### Rotation Monitoring

```bash
# View rotation Lambda logs
aws logs tail /aws/lambda/greenlang-secrets-rotation --follow

# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions Name=FunctionName,Value=greenlang-secrets-rotation \
  --start-time $(date -d '1 hour ago' -u +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 300 \
  --statistics Sum
```

---

## Kubernetes Integration

### External Secrets Configuration

The External Secrets Operator synchronizes AWS secrets to Kubernetes every hour.

#### ClusterSecretStore
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: greenlang-aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
            namespace: external-secrets
```

#### ExternalSecret
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: postgresql-app-credentials
  namespace: greenlang
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: greenlang-database-secrets
    kind: SecretStore
  target:
    name: postgresql-app-credentials
  data:
    - secretKey: username
      remoteRef:
        key: greenlang/database/application
        property: username
```

### Verifying Kubernetes Secrets

```bash
# Check ExternalSecret status
kubectl get externalsecret -n greenlang

# Verify secret sync
kubectl get secret postgresql-app-credentials -n greenlang -o yaml

# Check sync events
kubectl describe externalsecret postgresql-app-credentials -n greenlang
```

### Using Secrets in Pods

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: greenlang-app
  namespace: greenlang
spec:
  containers:
    - name: app
      image: greenlang/app:latest
      env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: postgresql-app-credentials
              key: DATABASE_URL
```

---

## Emergency Credential Reset

### When to Use Emergency Reset

- Suspected credential compromise
- Rotation failure with production impact
- Security incident requiring immediate credential change

### Emergency Reset Procedure

#### Step 1: Assess the Situation

```bash
# Check current secret versions
aws secretsmanager list-secret-version-ids \
  --secret-id greenlang/database/master

# Verify database connectivity
psql "postgresql://greenlang_app@greenlang-db.cluster-xxx.region.rds.amazonaws.com:5432/greenlang" \
  -c "SELECT 1"
```

#### Step 2: Generate New Credentials

```bash
# Generate a new secure password
NEW_PASSWORD=$(openssl rand -base64 32 | tr -d '/+=')

# Update the secret in Secrets Manager
aws secretsmanager update-secret \
  --secret-id greenlang/database/application \
  --secret-string "{
    \"username\": \"greenlang_app\",
    \"password\": \"$NEW_PASSWORD\",
    \"engine\": \"postgres\",
    \"host\": \"greenlang-db.cluster-xxx.region.rds.amazonaws.com\",
    \"port\": 5432,
    \"dbname\": \"greenlang\",
    \"connectionString\": \"postgresql://greenlang_app:$NEW_PASSWORD@greenlang-db.cluster-xxx.region.rds.amazonaws.com:5432/greenlang\"
  }"
```

#### Step 3: Update Database Password

```bash
# Connect as superuser
psql "postgresql://postgres@greenlang-db.cluster-xxx.region.rds.amazonaws.com:5432/greenlang" \
  -c "ALTER USER greenlang_app WITH PASSWORD '$NEW_PASSWORD'"
```

#### Step 4: Force Kubernetes Secret Refresh

```bash
# Trigger immediate sync
kubectl annotate externalsecret postgresql-app-credentials \
  -n greenlang \
  force-sync=$(date +%s) --overwrite

# Verify new secret
kubectl get secret postgresql-app-credentials -n greenlang \
  -o jsonpath='{.data.password}' | base64 -d
```

#### Step 5: Restart Affected Pods

```bash
# Rolling restart of deployments
kubectl rollout restart deployment/greenlang-api -n greenlang
kubectl rollout restart deployment/greenlang-worker -n greenlang

# Monitor rollout
kubectl rollout status deployment/greenlang-api -n greenlang
```

#### Step 6: Update PgBouncer (if applicable)

```bash
# Reload PgBouncer configuration
kubectl exec -n greenlang pgbouncer-0 -- pgbouncer -R /etc/pgbouncer/pgbouncer.ini

# Verify connections
kubectl exec -n greenlang pgbouncer-0 -- psql -p 6432 pgbouncer -c "SHOW POOLS"
```

#### Step 7: Verify Application Connectivity

```bash
# Check application logs
kubectl logs -n greenlang -l app=greenlang-api --tail=50

# Verify database connections
kubectl exec -n greenlang deploy/greenlang-api -- \
  python -c "from app.db import engine; print(engine.execute('SELECT 1').scalar())"
```

### Emergency Contact Escalation

| Level | Contact | Response Time |
|-------|---------|---------------|
| L1 | On-call DevOps | 15 minutes |
| L2 | Database Team Lead | 30 minutes |
| L3 | Security Team | Immediate for breaches |

---

## Troubleshooting

### Common Issues

#### Rotation Lambda Failure

```bash
# Check Lambda logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/greenlang-secrets-rotation \
  --filter-pattern "ERROR"

# Common causes:
# - VPC connectivity issues
# - IAM permission errors
# - Database connection timeout
```

#### External Secrets Not Syncing

```bash
# Check operator logs
kubectl logs -n external-secrets deploy/external-secrets

# Verify IRSA configuration
kubectl describe serviceaccount greenlang-db-secrets -n greenlang

# Check ExternalSecret events
kubectl describe externalsecret -n greenlang
```

#### Application Cannot Connect After Rotation

```bash
# Verify secret content matches database
aws secretsmanager get-secret-value \
  --secret-id greenlang/database/application \
  --query SecretString --output text | jq .password

# Check if K8s secret is updated
kubectl get secret postgresql-app-credentials -n greenlang \
  -o jsonpath='{.data.password}' | base64 -d

# Force pod restart if needed
kubectl delete pod -n greenlang -l app=greenlang-api
```

### Diagnostic Commands

```bash
# Full secrets health check
./scripts/check-secrets-health.sh

# Test database connectivity
./scripts/test-db-connection.sh

# Verify all rotation configurations
aws secretsmanager list-secrets \
  --filter Key=name,Values=greenlang \
  --query 'SecretList[*].[Name,RotationEnabled,LastRotatedDate]' \
  --output table
```

---

## Security Best Practices

### Access Control

1. **Least Privilege**: Only grant necessary IAM permissions
2. **IRSA**: Use IAM Roles for Service Accounts, not static credentials
3. **Network Isolation**: Lambda runs in VPC with database access only
4. **Audit Logging**: Enable CloudTrail for all Secrets Manager API calls

### Encryption

1. **KMS CMK**: Use customer-managed KMS keys
2. **Key Rotation**: Enable automatic KMS key rotation
3. **Envelope Encryption**: Secrets Manager uses envelope encryption

### Monitoring

1. **CloudWatch Alarms**: Alert on rotation failures
2. **SNS Notifications**: Notify on successful rotations
3. **Access Logging**: Monitor secret access patterns

### Compliance

1. **Secret Rotation**: Enforce 90-day maximum secret age
2. **Password Policy**: Minimum 32 characters, mixed complexity
3. **Recovery Window**: 30-day window before permanent deletion
4. **Backup**: Secrets backed up through AWS native capabilities

---

## Appendix

### IAM Policies

#### Application Secret Access
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:greenlang/database/application-*"
    },
    {
      "Effect": "Allow",
      "Action": "kms:Decrypt",
      "Resource": "arn:aws:kms:*:*:key/*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "secretsmanager.us-east-1.amazonaws.com"
        }
      }
    }
  ]
}
```

### Terraform Outputs

```bash
# Get all secret ARNs
terraform output -json all_secret_arns

# Get IAM role for EKS
terraform output eks_secrets_role_arn

# Get rotation Lambda ARN
terraform output rotation_lambda_arn
```

### Useful Scripts

```bash
# List all secrets with rotation status
./scripts/list-secrets-status.sh

# Manually trigger rotation for all secrets
./scripts/rotate-all-secrets.sh

# Backup secrets to encrypted file
./scripts/backup-secrets.sh
```

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2024-01-15 | DevOps Team | Initial documentation |
| 1.1.0 | 2024-02-01 | DevOps Team | Added emergency reset procedures |
| 1.2.0 | 2024-03-01 | DevOps Team | Added Kubernetes integration details |
