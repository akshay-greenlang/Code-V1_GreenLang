# GreenLang AWS Secrets Manager - INFRA-001

This directory contains templates and scripts for managing secrets in AWS Secrets Manager for the GreenLang platform.

---

## Overview

The GreenLang platform requires secure management of sensitive credentials including:

- Database connection strings and passwords
- Redis/cache authentication tokens
- Third-party API keys (OpenAI, Anthropic, etc.)
- CI/CD runner tokens and registry credentials
- TLS certificates and private keys
- JWT signing keys and encryption keys

All secrets are stored in AWS Secrets Manager with KMS encryption and organized by environment (dev, staging, prod).

---

## Directory Structure

```
deployment/secrets/
├── README.md                    # This documentation
├── aws-secrets-template.json    # Secret structure templates
└── .gitignore                   # Prevents accidental secret commits

deployment/scripts/
└── create-aws-secrets.sh        # AWS Secrets Manager setup script
```

---

## Required Secrets

### 1. Database Credentials (`greenlang-{env}/database`)

PostgreSQL connection information for the primary database.

| Key | Description | Example |
|-----|-------------|---------|
| `connection_string` | Full PostgreSQL connection URI | `postgresql://user:pass@host:5432/db` |
| `host` | Database hostname | `greenlang-db.xxx.rds.amazonaws.com` |
| `port` | Database port | `5432` |
| `database` | Database name | `greenlang_prod` |
| `username` | Database username | `greenlang_admin` |
| `password` | Database password | (secure password) |
| `ssl_mode` | SSL connection mode | `require` |
| `pool_size` | Connection pool size | `20` |
| `max_overflow` | Max overflow connections | `40` |
| `read_replica_host` | Read replica hostname | `greenlang-db-reader.xxx.rds.amazonaws.com` |

**Generation Commands:**
```bash
# Generate secure password (32+ characters)
openssl rand -base64 32
```

---

### 2. Redis Credentials (`greenlang-{env}/redis`)

ElastiCache Redis authentication and connection details.

| Key | Description | Example |
|-----|-------------|---------|
| `password` | Redis AUTH password | (secure password) |
| `auth_token` | Redis AUTH token (64 chars) | (secure token) |
| `host` | Redis primary endpoint | `greenlang-redis.xxx.cache.amazonaws.com` |
| `port` | Redis port | `6379` |
| `ssl_enabled` | TLS encryption enabled | `true` |
| `cluster_mode` | Cluster mode enabled | `false` |
| `sentinel_host` | Sentinel endpoint (if used) | `greenlang-redis-sentinel.xxx.cache.amazonaws.com` |
| `database` | Redis database number | `0` |

**Generation Commands:**
```bash
# Generate Redis AUTH token (64 characters)
openssl rand -hex 32
```

---

### 3. API Keys (`greenlang-{env}/api-keys`)

Third-party service API keys for AI and external integrations.

| Key | Description | Where to Get |
|-----|-------------|--------------|
| `openai_api_key` | OpenAI API key | https://platform.openai.com/api-keys |
| `openai_org_id` | OpenAI Organization ID | https://platform.openai.com/account/org-settings |
| `anthropic_api_key` | Anthropic/Claude API key | https://console.anthropic.com/settings/keys |
| `pinecone_api_key` | Pinecone vector DB key | https://app.pinecone.io/ |
| `pinecone_environment` | Pinecone environment | `us-east-1-aws` |
| `cohere_api_key` | Cohere API key | https://dashboard.cohere.com/api-keys |
| `huggingface_api_key` | HuggingFace API token | https://huggingface.co/settings/tokens |
| `weaviate_api_key` | Weaviate Cloud key | https://console.weaviate.cloud/ |
| `google_ai_api_key` | Google AI/Gemini key | https://aistudio.google.com/apikey |
| `azure_openai_api_key` | Azure OpenAI key | Azure Portal |
| `sendgrid_api_key` | SendGrid email key | https://app.sendgrid.com/settings/api_keys |
| `stripe_api_key` | Stripe payment key | https://dashboard.stripe.com/apikeys |
| `stripe_webhook_secret` | Stripe webhook secret | Stripe Dashboard > Webhooks |

---

### 4. Runner Credentials (`greenlang-{env}/runner`)

CI/CD runner and container registry credentials.

| Key | Description | Where to Get |
|-----|-------------|--------------|
| `github_runner_token` | Self-hosted runner token | GitHub Settings > Actions > Runners |
| `github_app_id` | GitHub App ID | GitHub Developer Settings |
| `github_app_private_key` | GitHub App private key | GitHub App settings |
| `github_webhook_secret` | Webhook validation secret | Repository webhooks |
| `ecr_registry_url` | AWS ECR registry URL | AWS Console |
| `ecr_access_key_id` | ECR access key | IAM user credentials |
| `ecr_secret_access_key` | ECR secret key | IAM user credentials |
| `ghcr_username` | GitHub Container Registry user | GitHub username |
| `ghcr_token` | GHCR personal access token | GitHub Settings > Developer > PAT |
| `dockerhub_username` | Docker Hub username | Docker Hub account |
| `dockerhub_token` | Docker Hub access token | Docker Hub > Security |
| `sonarqube_token` | SonarQube analysis token | SonarQube > Security |
| `codecov_token` | Codecov upload token | Codecov settings |
| `snyk_token` | Snyk security scan token | Snyk account settings |

---

### 5. TLS Certificates (`greenlang-{env}/tls`)

TLS/SSL certificates for secure communications.

| Key | Description | Notes |
|-----|-------------|-------|
| `domain` | Primary domain | `api.greenlang.io` |
| `certificate` | Server certificate (PEM) | From CA or Let's Encrypt |
| `private_key` | Certificate private key | Keep secure! |
| `certificate_chain` | Intermediate certificates | CA-provided chain |
| `ca_certificate` | Root CA certificate | For validation |
| `wildcard_certificate` | Wildcard cert (*.greenlang.io) | Optional |
| `wildcard_private_key` | Wildcard private key | Optional |
| `internal_ca_certificate` | Internal PKI CA cert | For service mesh |
| `mtls_client_certificate` | mTLS client cert | For service-to-service |
| `mtls_client_key` | mTLS client key | For service-to-service |

**Note:** For production, prefer AWS Certificate Manager (ACM) for public certificates.

---

### 6. JWT/Auth Secrets (`greenlang-{env}/jwt`)

Authentication and JWT signing credentials.

| Key | Description | Generation |
|-----|-------------|------------|
| `jwt_secret_key` | JWT HMAC signing key (256-bit) | `openssl rand -hex 32` |
| `jwt_refresh_secret_key` | Refresh token signing key | `openssl rand -hex 32` |
| `jwt_algorithm` | JWT algorithm | `HS256` or `RS256` |
| `jwt_access_token_expire_minutes` | Access token TTL | `30` |
| `jwt_refresh_token_expire_days` | Refresh token TTL | `7` |
| `rsa_private_key` | RSA private key (for RS256) | `openssl genrsa -out private.pem 2048` |
| `rsa_public_key` | RSA public key | `openssl rsa -in private.pem -pubout` |
| `encryption_key` | Data encryption key | `openssl rand -hex 32` |
| `session_secret` | Session signing secret | `openssl rand -hex 32` |

**Generation Commands:**
```bash
# Generate 256-bit secret key
openssl rand -hex 32

# Generate RSA key pair
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -pubout -out public.pem
```

---

### 7. Monitoring Credentials (`greenlang-{env}/monitoring`)

Observability and alerting service credentials.

| Key | Description | Where to Get |
|-----|-------------|--------------|
| `grafana_admin_user` | Grafana admin username | Set during setup |
| `grafana_admin_password` | Grafana admin password | Generated |
| `prometheus_basic_auth_password` | Prometheus auth password | Generated |
| `alertmanager_slack_webhook_url` | Slack notification webhook | Slack App settings |
| `pagerduty_integration_key` | PagerDuty integration key | PagerDuty > Integrations |
| `datadog_api_key` | Datadog API key | Datadog > Organization Settings |
| `datadog_app_key` | Datadog Application key | Datadog > Organization Settings |
| `newrelic_license_key` | New Relic license key | New Relic > API Keys |
| `sentry_dsn` | Sentry DSN | Sentry > Project Settings |
| `opsgenie_api_key` | OpsGenie API key | OpsGenie > Settings |

---

### 8. Encryption Keys (`greenlang-{env}/encryption`)

Data encryption keys for at-rest encryption.

| Key | Description | Generation |
|-----|-------------|------------|
| `data_encryption_key` | Primary DEK (256-bit) | `openssl rand -hex 32` |
| `kms_key_id` | AWS KMS key ARN | AWS KMS Console |
| `backup_encryption_key` | Backup encryption key | `openssl rand -hex 32` |
| `pii_encryption_key` | PII-specific encryption | `openssl rand -hex 32` |
| `csrd_encryption_key` | CSRD data encryption | `openssl rand -hex 32` |
| `field_level_encryption_key` | Field-level encryption | `openssl rand -hex 32` |
| `key_derivation_salt` | Key derivation salt | `openssl rand -hex 16` |

---

## How to Populate Secrets

### Option 1: Using the Setup Script (Recommended)

```bash
# Navigate to scripts directory
cd deployment/scripts

# Make script executable (Linux/Mac)
chmod +x create-aws-secrets.sh

# Run in dry-run mode first to preview
./create-aws-secrets.sh dev --dry-run

# Create secrets for development
./create-aws-secrets.sh dev

# Create secrets for staging
./create-aws-secrets.sh staging

# Create secrets for production
./create-aws-secrets.sh prod
```

### Option 2: Using AWS CLI Directly

```bash
# Create a single secret
aws secretsmanager create-secret \
    --name "greenlang-prod/database" \
    --description "PostgreSQL database credentials" \
    --secret-string '{"username":"admin","password":"secure-password"}' \
    --kms-key-id "alias/greenlang-secrets-prod" \
    --region us-east-1

# Update an existing secret
aws secretsmanager update-secret \
    --secret-id "greenlang-prod/database" \
    --secret-string '{"username":"admin","password":"new-password"}'

# Retrieve a secret value
aws secretsmanager get-secret-value \
    --secret-id "greenlang-prod/database" \
    --region us-east-1
```

### Option 3: Using AWS Console

1. Navigate to AWS Secrets Manager Console
2. Click "Store a new secret"
3. Choose "Other type of secret"
4. Enter key-value pairs based on templates
5. Name: `greenlang-{environment}/{secret-name}`
6. Configure rotation if needed
7. Review and store

### Option 4: Using Terraform

```hcl
resource "aws_secretsmanager_secret" "database" {
  name        = "greenlang-prod/database"
  description = "PostgreSQL database credentials"
  kms_key_id  = aws_kms_key.secrets.id
}

resource "aws_secretsmanager_secret_version" "database" {
  secret_id = aws_secretsmanager_secret.database.id
  secret_string = jsonencode({
    username = "greenlang_admin"
    password = random_password.db.result
    # ... other fields
  })
}
```

---

## Security Best Practices

### Secret Generation

1. **Use cryptographically secure random generators**
   ```bash
   # Good - cryptographically secure
   openssl rand -hex 32

   # Bad - predictable
   echo "password123"
   ```

2. **Minimum key lengths**
   - Passwords: 32+ characters
   - API keys: Use provider defaults
   - Encryption keys: 256 bits (32 bytes)
   - JWT secrets: 256 bits minimum

3. **Character requirements**
   - Include uppercase, lowercase, numbers, special characters
   - Avoid ambiguous characters (0/O, 1/l/I)

### Access Control

1. **IAM Policies**
   - Use least-privilege principle
   - Separate read/write permissions
   - Environment-specific policies

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["secretsmanager:GetSecretValue"],
       "Resource": "arn:aws:secretsmanager:*:*:secret:greenlang-prod/*"
     }]
   }
   ```

2. **Service Accounts**
   - Use IAM roles for EC2/EKS workloads
   - Avoid long-lived access keys
   - Enable MFA for console access

3. **Network Security**
   - Access secrets only from VPC
   - Use VPC endpoints for Secrets Manager
   - Enable VPC flow logs

### Rotation

1. **Enable automatic rotation where supported**
   - Database passwords: 30 days
   - Redis tokens: 90 days
   - JWT secrets: 90 days
   - Encryption keys: 365 days

2. **Manual rotation schedule**
   - API keys: Quarterly or on compromise
   - TLS certificates: Before expiration
   - Runner tokens: After personnel changes

3. **Rotation process**
   ```bash
   # 1. Generate new secret
   NEW_SECRET=$(openssl rand -hex 32)

   # 2. Update application to accept both old and new
   # 3. Update secret in Secrets Manager
   aws secretsmanager update-secret \
       --secret-id greenlang-prod/jwt \
       --secret-string "{\"jwt_secret_key\":\"$NEW_SECRET\"}"

   # 4. Deploy application update
   # 5. Remove old secret acceptance
   ```

### Monitoring and Auditing

1. **Enable CloudTrail logging**
   ```bash
   aws cloudtrail create-trail \
       --name greenlang-secrets-audit \
       --s3-bucket-name greenlang-audit-logs
   ```

2. **Set up CloudWatch alarms**
   - Failed access attempts
   - Unusual access patterns
   - Secret deletions

3. **Regular audits**
   - Review access logs monthly
   - Audit IAM policies quarterly
   - Test rotation procedures

### Emergency Procedures

1. **Secret compromise response**
   ```bash
   # 1. Immediately rotate the compromised secret
   ./create-aws-secrets.sh prod --rotate-single database

   # 2. Review access logs
   aws cloudtrail lookup-events \
       --lookup-attributes AttributeKey=ResourceName,AttributeValue=greenlang-prod/database

   # 3. Revoke suspicious sessions
   # 4. Notify security team
   ```

2. **Backup and recovery**
   - Secrets Manager maintains version history
   - Use `--recovery-window-in-days` for deletion protection
   - Document recovery procedures

---

## Environment-Specific Notes

### Development (`greenlang-dev`)
- Rotation disabled for convenience
- Can use weaker passwords for local testing
- Separate AWS account recommended

### Staging (`greenlang-staging`)
- Mirror production configuration
- Enable rotation with longer intervals
- Use production-like secrets

### Production (`greenlang-prod`)
- Enable all security features
- Shortest rotation intervals
- Require MFA for access
- Enable deletion protection

---

## Troubleshooting

### Common Issues

1. **Access Denied**
   ```
   Error: AccessDeniedException
   ```
   - Check IAM policy permissions
   - Verify KMS key permissions
   - Ensure correct region

2. **Secret Not Found**
   ```
   Error: ResourceNotFoundException
   ```
   - Verify secret name (case-sensitive)
   - Check AWS region
   - Confirm secret was created

3. **Decryption Failed**
   ```
   Error: KMSAccessDeniedException
   ```
   - Add `kms:Decrypt` permission
   - Check KMS key policy
   - Verify VIA service condition

### Useful Commands

```bash
# List all GreenLang secrets
aws secretsmanager list-secrets \
    --filter Key=name,Values=greenlang \
    --region us-east-1

# Get secret metadata
aws secretsmanager describe-secret \
    --secret-id greenlang-prod/database

# View secret versions
aws secretsmanager list-secret-version-ids \
    --secret-id greenlang-prod/database

# Restore deleted secret
aws secretsmanager restore-secret \
    --secret-id greenlang-prod/database
```

---

## Related Documentation

- [AWS Secrets Manager Documentation](https://docs.aws.amazon.com/secretsmanager/)
- [GreenLang Deployment Guide](../DEPLOYMENT_GUIDE.md)
- [Security Audit Documentation](../security/README.md)
- [Terraform Infrastructure](../terraform/README.md)

---

## Contacts

- **Security Team**: security@greenlang.io
- **DevOps Team**: devops@greenlang.io
- **On-Call**: #platform-oncall (Slack)

---

**Last Updated**: 2026-02-03
**Version**: 1.0.0
**Classification**: INTERNAL USE ONLY
