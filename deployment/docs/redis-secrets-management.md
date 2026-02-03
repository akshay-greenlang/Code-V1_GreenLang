# Redis Secrets Management Guide

This document describes how Redis credentials are managed, rotated, and integrated with GreenLang applications using AWS Secrets Manager and Kubernetes External Secrets.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Credential Storage](#credential-storage)
- [Rotation Procedure](#rotation-procedure)
- [Emergency Credential Reset](#emergency-credential-reset)
- [Application Integration](#application-integration)
- [Troubleshooting](#troubleshooting)
- [Security Best Practices](#security-best-practices)

---

## Architecture Overview

```
+-------------------+     +----------------------+     +------------------+
|  AWS Secrets      |     |  External Secrets    |     |  Kubernetes      |
|  Manager          |---->|  Operator            |---->|  Secrets         |
|                   |     |                      |     |                  |
|  - Redis password |     |  - ClusterSecretStore|     |  - redis-creds   |
|  - Connection URL |     |  - ExternalSecret    |     |  - redis-url     |
|  - Rotation config|     |  - 1h refresh        |     |                  |
+-------------------+     +----------------------+     +------------------+
        |                          |                          |
        |                          |                          v
        v                          v                   +------------------+
+-------------------+     +----------------------+     |  Application     |
|  KMS Key          |     |  IRSA                |     |  Pods            |
|  - Encryption     |     |  - IAM Role          |     |                  |
|  - Key rotation   |     |  - Service Account   |     |  - Env vars      |
+-------------------+     +----------------------+     |  - Volume mounts |
                                                       +------------------+
```

### Components

| Component | Purpose | Location |
|-----------|---------|----------|
| AWS Secrets Manager | Stores Redis credentials securely | AWS |
| KMS Key | Encrypts secrets at rest | AWS |
| External Secrets Operator | Syncs secrets from AWS to K8s | Kubernetes |
| ClusterSecretStore | Configures AWS provider for External Secrets | Kubernetes |
| ExternalSecret | Defines which secrets to sync | Kubernetes |
| IRSA Role | Provides IAM authentication for pods | AWS/EKS |

---

## Credential Storage

### Secret Structure in AWS Secrets Manager

The Redis credentials are stored in AWS Secrets Manager with the following structure:

**Secret Name:** `greenlang/redis/auth`

```json
{
  "password": "generated-32-char-password",
  "username": "default",
  "host": "greenlang-redis.xxxxx.cache.amazonaws.com",
  "port": 6379,
  "ssl_enabled": true,
  "connection_url": "rediss://default:password@host:6379",
  "created_at": "2024-01-15T10:30:00Z",
  "rotated_at": "2024-01-15T10:30:00Z"
}
```

### Encryption

- **KMS Key:** All secrets are encrypted using a dedicated KMS key
- **Key Rotation:** KMS key rotation is enabled (automatic annual rotation)
- **Access Control:** IAM policies restrict access to specific roles

### Terraform Configuration

```hcl
module "redis_secrets" {
  source = "./modules/redis-secrets"

  secret_name_prefix = "greenlang"

  # Redis connection details
  redis_host        = module.elasticache.primary_endpoint
  redis_port        = 6379
  redis_ssl_enabled = true

  # Password requirements
  password_length        = 32
  password_special_chars = true

  # Rotation (optional)
  enable_rotation     = true
  rotation_days       = 30
  rotation_lambda_arn = aws_lambda_function.rotate_redis.arn

  # EKS IRSA configuration
  eks_cluster_name         = module.eks.cluster_name
  eks_oidc_provider_arn    = module.eks.oidc_provider_arn
  eks_namespace            = "greenlang"
  eks_service_account_name = "redis-secrets-sa"

  tags = var.common_tags
}
```

---

## Rotation Procedure

### Automatic Rotation

When enabled, secrets are automatically rotated using AWS Lambda:

1. **Trigger:** CloudWatch Events triggers rotation based on schedule
2. **Create Secret:** Lambda generates new password
3. **Set Secret:** New credentials are applied to ElastiCache
4. **Test Secret:** Lambda verifies connection with new credentials
5. **Finish Secret:** Old version is marked for deletion

### Manual Rotation

To manually rotate Redis credentials:

#### Step 1: Generate New Credentials

```bash
# Using AWS CLI
aws secretsmanager rotate-secret \
  --secret-id greenlang/redis/auth \
  --rotation-lambda-arn arn:aws:lambda:us-east-1:ACCOUNT:function:redis-rotate

# Or trigger via Terraform
terraform apply -target=module.redis_secrets -var="force_rotation=true"
```

#### Step 2: Verify Rotation

```bash
# Check secret version
aws secretsmanager describe-secret \
  --secret-id greenlang/redis/auth \
  --query 'VersionIdsToStages'

# Verify new credentials work
aws secretsmanager get-secret-value \
  --secret-id greenlang/redis/auth \
  --version-stage AWSCURRENT
```

#### Step 3: Trigger Kubernetes Sync

```bash
# Force External Secrets to refresh immediately
kubectl annotate externalsecret redis-credentials \
  -n greenlang \
  force-sync=$(date +%s) --overwrite

# Verify secret was updated
kubectl get secret redis-credentials -n greenlang -o yaml
```

#### Step 4: Rolling Restart Applications

```bash
# Restart deployments to pick up new credentials
kubectl rollout restart deployment -n greenlang -l app.kubernetes.io/uses-redis=true

# Monitor rollout
kubectl rollout status deployment/greenlang-app -n greenlang
```

### Rotation Schedule

| Environment | Rotation Frequency | Schedule |
|-------------|-------------------|----------|
| Production  | Every 30 days     | `cron(0 2 1 * ? *)` |
| Staging     | Every 14 days     | `cron(0 2 15 * ? *)` |
| Development | Manual only       | N/A |

---

## Emergency Credential Reset

### When to Use

- Suspected credential compromise
- Security incident response
- Failed rotation recovery

### Emergency Reset Procedure

#### Step 1: Immediate Actions

```bash
# 1. Revoke current credentials (if compromised)
aws elasticache modify-user \
  --user-id greenlang-user \
  --passwords "TEMPORARY_EMERGENCY_PASSWORD" \
  --no-password-required false

# 2. Update secret immediately
aws secretsmanager put-secret-value \
  --secret-id greenlang/redis/auth \
  --secret-string '{
    "password": "TEMPORARY_EMERGENCY_PASSWORD",
    "username": "default",
    "host": "greenlang-redis.xxxxx.cache.amazonaws.com",
    "port": 6379,
    "ssl_enabled": true,
    "connection_url": "rediss://default:TEMPORARY_EMERGENCY_PASSWORD@greenlang-redis.xxxxx.cache.amazonaws.com:6379",
    "rotated_at": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
  }'
```

#### Step 2: Force Kubernetes Update

```bash
# Delete and recreate ExternalSecret to force immediate sync
kubectl delete externalsecret redis-credentials -n greenlang
kubectl apply -f deployment/kubernetes/database/redis-secrets/externalsecret-redis.yaml

# Alternatively, delete the K8s secret directly (External Secrets will recreate it)
kubectl delete secret redis-credentials -n greenlang
```

#### Step 3: Restart All Applications

```bash
# Immediate restart of all pods using Redis
kubectl delete pods -n greenlang -l app.kubernetes.io/uses-redis=true

# Scale down and up for clean restart
kubectl scale deployment --replicas=0 -n greenlang --all
kubectl scale deployment --replicas=3 -n greenlang --all
```

#### Step 4: Verify Recovery

```bash
# Check application logs
kubectl logs -n greenlang -l app=greenlang-app --tail=100

# Test Redis connection
kubectl exec -it deploy/greenlang-app -n greenlang -- \
  redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping
```

#### Step 5: Generate Permanent Credentials

```bash
# After emergency is resolved, generate proper random password
NEW_PASSWORD=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9!#$%&*()-_=+[]{}:?' | head -c 32)

# Update ElastiCache
aws elasticache modify-user \
  --user-id greenlang-user \
  --passwords "$NEW_PASSWORD"

# Update Secrets Manager
aws secretsmanager put-secret-value \
  --secret-id greenlang/redis/auth \
  --secret-string "{...}" # Full JSON with new password
```

### Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call DevOps | PagerDuty | Immediate |
| Security Team | security@greenlang.io | Within 15 min |
| Infrastructure Lead | infrastructure@greenlang.io | Within 30 min |

---

## Application Integration

### Environment Variables

Applications receive Redis credentials via environment variables:

```yaml
# Deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-app
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        # Option 1: Complete URL
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-connection-url
              key: REDIS_URL

        # Option 2: Individual components
        - name: REDIS_HOST
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: REDIS_HOST
        - name: REDIS_PORT
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: REDIS_PORT
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: REDIS_PASSWORD
```

### Volume Mount (Alternative)

```yaml
spec:
  containers:
  - name: app
    volumeMounts:
    - name: redis-secrets
      mountPath: /etc/secrets/redis
      readOnly: true
  volumes:
  - name: redis-secrets
    secret:
      secretName: redis-credentials
```

### Application Code Examples

#### Python (redis-py)

```python
import os
import redis

# Using URL
redis_client = redis.from_url(os.environ['REDIS_URL'])

# Using components
redis_client = redis.Redis(
    host=os.environ['REDIS_HOST'],
    port=int(os.environ['REDIS_PORT']),
    password=os.environ['REDIS_PASSWORD'],
    ssl=os.environ.get('REDIS_SSL', 'true').lower() == 'true'
)
```

#### Node.js (ioredis)

```javascript
const Redis = require('ioredis');

// Using URL
const redis = new Redis(process.env.REDIS_URL);

// Using components
const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: process.env.REDIS_PORT,
  password: process.env.REDIS_PASSWORD,
  tls: process.env.REDIS_SSL === 'true' ? {} : undefined
});
```

#### Go (go-redis)

```go
import "github.com/redis/go-redis/v9"

// Using URL
opt, _ := redis.ParseURL(os.Getenv("REDIS_URL"))
client := redis.NewClient(opt)

// Using components
client := redis.NewClient(&redis.Options{
    Addr:     fmt.Sprintf("%s:%s", os.Getenv("REDIS_HOST"), os.Getenv("REDIS_PORT")),
    Password: os.Getenv("REDIS_PASSWORD"),
})
```

### Sidecar Pattern (Optional)

For applications that need dynamic credential refresh:

```yaml
spec:
  containers:
  - name: app
    # Main application

  - name: secret-refresher
    image: external-secrets/kubernetes-external-secrets:latest
    command: ["/bin/sh", "-c"]
    args:
      - |
        while true; do
          # Check if secret has changed
          kubectl get secret redis-credentials -o jsonpath='{.metadata.resourceVersion}' > /tmp/version
          if ! diff -q /tmp/version /tmp/old-version > /dev/null 2>&1; then
            # Signal main app to reload
            kill -HUP 1
            cp /tmp/version /tmp/old-version
          fi
          sleep 60
        done
```

---

## Troubleshooting

### Common Issues

#### ExternalSecret Not Syncing

```bash
# Check ExternalSecret status
kubectl describe externalsecret redis-credentials -n greenlang

# Check External Secrets Operator logs
kubectl logs -n external-secrets -l app.kubernetes.io/name=external-secrets

# Verify ClusterSecretStore
kubectl describe clustersecretstore greenlang-redis-aws-secrets
```

#### IAM Permission Errors

```bash
# Verify IRSA is configured correctly
kubectl describe sa redis-secrets-sa -n greenlang

# Test IAM role from pod
kubectl run aws-cli --image=amazon/aws-cli -it --rm -- \
  sts get-caller-identity

# Check IAM role policies
aws iam list-attached-role-policies \
  --role-name greenlang-eks-redis-secrets
```

#### Secret Not Found in AWS

```bash
# List secrets
aws secretsmanager list-secrets \
  --filters Key=name,Values=greenlang/redis

# Check secret exists
aws secretsmanager describe-secret \
  --secret-id greenlang/redis/auth
```

#### KMS Decryption Errors

```bash
# Verify KMS key access
aws kms describe-key --key-id alias/greenlang-redis-secrets

# Test decryption
aws secretsmanager get-secret-value \
  --secret-id greenlang/redis/auth
```

### Health Check Commands

```bash
# Check all components
echo "=== External Secrets Operator ==="
kubectl get pods -n external-secrets

echo "=== ClusterSecretStore ==="
kubectl get clustersecretstore

echo "=== ExternalSecrets ==="
kubectl get externalsecret -A

echo "=== Kubernetes Secrets ==="
kubectl get secret redis-credentials -n greenlang

echo "=== AWS Secret ==="
aws secretsmanager describe-secret --secret-id greenlang/redis/auth
```

---

## Security Best Practices

### Access Control

1. **Least Privilege:** IAM roles only have access to specific secrets
2. **Namespace Isolation:** Use SecretStore instead of ClusterSecretStore for stricter isolation
3. **Audit Logging:** Enable CloudTrail for Secrets Manager API calls

### Network Security

1. **VPC Endpoints:** Use VPC endpoints for Secrets Manager access
2. **Private Subnets:** ElastiCache should only be accessible from private subnets
3. **Security Groups:** Restrict Redis port access to application pods only

### Monitoring

```yaml
# Prometheus alert for secret sync failures
- alert: ExternalSecretSyncFailed
  expr: external_secrets_sync_calls_error > 0
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "External Secret sync failed"
    description: "Secret {{ $labels.name }} failed to sync"
```

### Compliance

- **Encryption:** All secrets encrypted at rest with KMS
- **Rotation:** Automated rotation every 30 days
- **Auditing:** All access logged to CloudTrail
- **Access Reviews:** Quarterly review of IAM policies

---

## Related Documentation

- [Terraform Module README](../terraform/modules/redis-secrets/README.md)
- [External Secrets Operator Documentation](https://external-secrets.io/)
- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- [EKS IRSA Documentation](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html)
