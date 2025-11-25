# GreenLang Environment Variables Reference

This document provides a comprehensive reference for all environment variables used across the GreenLang platform and its applications.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Database Configuration](#database-configuration)
3. [Cache Configuration (Redis)](#cache-configuration-redis)
4. [Security Settings](#security-settings)
5. [External API Keys](#external-api-keys)
6. [Cloud Provider Configuration](#cloud-provider-configuration)
7. [Service Endpoints](#service-endpoints)
8. [Message Queue (RabbitMQ)](#message-queue-rabbitmq)
9. [Object Storage (S3)](#object-storage-s3)
10. [Email Configuration](#email-configuration)
11. [Logging Configuration](#logging-configuration)
12. [Monitoring and Observability](#monitoring-and-observability)
13. [Performance Settings](#performance-settings)
14. [Feature Flags](#feature-flags)
15. [GreenLang-Specific Settings](#greenlang-specific-settings)
16. [Application-Specific Variables](#application-specific-variables)
17. [Development and Testing](#development-and-testing)
18. [Deployment Configuration](#deployment-configuration)
19. [Compliance and Audit](#compliance-and-audit)
20. [Backup and Recovery](#backup-and-recovery)
21. [Sensitive Variables Handling](#sensitive-variables-handling)

---

## Quick Start

Copy the appropriate environment template for your use case:

```bash
# For development
cp .env.example .env

# For production
cp .env.template .env

# For Docker Compose
cp docker-compose.env.example docker-compose.env
```

Generate secure secrets:

```bash
# Generate JWT secret (32+ characters)
openssl rand -hex 32

# Generate encryption key
openssl rand -hex 32

# Generate Fernet key (for field-level encryption)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## Database Configuration

### PostgreSQL Primary Database

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `POSTGRES_HOST` | Yes | `localhost` | PostgreSQL host address | `db.example.com` |
| `POSTGRES_PORT` | Yes | `5432` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Yes | - | Database name | `greenlang_db` |
| `POSTGRES_USER` | Yes | - | Database username | `greenlang_user` |
| `POSTGRES_PASSWORD` | Yes | - | Database password | (secure password) |
| `POSTGRES_SSL_MODE` | No | `require` | SSL mode: disable, require, verify-ca, verify-full | `require` |
| `POSTGRES_CONNECTION_POOL_SIZE` | No | `20` | Connection pool size | `20` |
| `POSTGRES_MAX_OVERFLOW` | No | `10` | Maximum overflow connections | `10` |
| `DATABASE_URL` | No | - | Full database connection URL (overrides individual settings) | `postgresql://user:pass@host:5432/db` |

**Used by:** Core API, GL-CBAM-APP, GL-CSRD-APP, GL-VCCI-Carbon-APP

### PostgreSQL Read Replica (Optional)

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `POSTGRES_READ_REPLICA_HOST` | No | - | Read replica host | `read.db.example.com` |
| `POSTGRES_READ_REPLICA_PORT` | No | `5432` | Read replica port | `5432` |
| `DATABASE_REPLICA_HOST` | No | - | Alternative name for read replica host | `replica.db.example.com` |
| `DATABASE_REPLICA_PORT` | No | `5432` | Alternative name for read replica port | `5432` |

**Used by:** GL-VCCI-Carbon-APP

### PostgreSQL Container Settings (Docker)

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `POSTGRES_MAX_CONNECTIONS` | No | `200` | Maximum connections | `200` |
| `POSTGRES_SHARED_BUFFERS` | No | `256MB` | Shared buffers size | `256MB` |
| `POSTGRES_EFFECTIVE_CACHE_SIZE` | No | `1GB` | Effective cache size | `1GB` |
| `POSTGRES_MAINTENANCE_WORK_MEM` | No | `128MB` | Maintenance work memory | `128MB` |
| `POSTGRES_CHECKPOINT_COMPLETION_TARGET` | No | `0.9` | Checkpoint completion target | `0.9` |
| `POSTGRES_WAL_BUFFERS` | No | `16MB` | WAL buffers size | `16MB` |
| `POSTGRES_DEFAULT_STATISTICS_TARGET` | No | `100` | Statistics target | `100` |
| `POSTGRES_RANDOM_PAGE_COST` | No | `1.1` | Random page cost | `1.1` |
| `POSTGRES_EFFECTIVE_IO_CONCURRENCY` | No | `200` | I/O concurrency | `200` |

**Used by:** docker-compose.yml

### Database Performance

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `DB_POOL_SIZE` | No | `10` | Database pool size | `10` |
| `DB_MAX_OVERFLOW` | No | `20` | Maximum overflow | `20` |
| `DB_POOL_TIMEOUT` | No | `30` | Pool timeout (seconds) | `30` |
| `DB_POOL_RECYCLE` | No | `3600` | Connection recycle time (seconds) | `3600` |
| `DB_ECHO` | No | `false` | Enable SQL query logging | `false` |
| `DB_STATEMENT_TIMEOUT` | No | `30000` | Statement timeout (milliseconds) | `30000` |
| `DB_LOCK_TIMEOUT` | No | `10000` | Lock timeout (milliseconds) | `10000` |
| `DB_CONNECTION_TIMEOUT` | No | `10` | Connection timeout (seconds) | `10` |
| `DB_COMMAND_TIMEOUT` | No | `30` | Command timeout (seconds) | `30` |

**Used by:** Core API, GL-CBAM-APP

---

## Cache Configuration (Redis)

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `REDIS_HOST` | Yes | `localhost` | Redis host address | `redis.example.com` |
| `REDIS_PORT` | Yes | `6379` | Redis port | `6379` |
| `REDIS_PASSWORD` | No | - | Redis password | (secure password) |
| `REDIS_DB` | No | `0` | Redis database number | `0` |
| `REDIS_SSL` | No | `false` | Enable SSL/TLS | `true` |
| `REDIS_URL` | No | - | Full Redis URL (overrides individual settings) | `redis://:password@host:6379/0` |
| `REDIS_CONNECTION_POOL_SIZE` | No | `50` | Connection pool size | `50` |
| `REDIS_SOCKET_TIMEOUT` | No | `30` | Socket timeout (seconds) | `30` |
| `REDIS_MAX_CONNECTIONS` | No | `50` | Maximum connections | `50` |
| `REDIS_DECODE_RESPONSES` | No | `true` | Decode responses to strings | `true` |
| `REDIS_MAXMEMORY` | No | `512mb` | Maximum memory (Docker) | `512mb` |
| `REDIS_MAXMEMORY_POLICY` | No | `allkeys-lru` | Eviction policy | `allkeys-lru` |

**Used by:** Core API, greenlang/cache, GL-CBAM-APP, GL-VCCI-Carbon-APP

### Cache Settings

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `CACHE_ENABLED` | No | `true` | Enable caching | `true` |
| `CACHE_DEFAULT_TIMEOUT` | No | `300` | Default cache TTL (seconds) | `300` |
| `CACHE_KEY_PREFIX` | No | `greenlang:` | Cache key prefix | `greenlang:` |
| `CACHE_VERSION` | No | `1` | Cache version (for invalidation) | `1` |
| `CACHE_TTL_SHORT` | No | `300` | Short TTL (5 minutes) | `300` |
| `CACHE_TTL_MEDIUM` | No | `3600` | Medium TTL (1 hour) | `3600` |
| `CACHE_TTL_LONG` | No | `86400` | Long TTL (24 hours) | `86400` |
| `CACHE_TTL_SECONDS` | No | `3600` | Cache TTL for VCCI | `3600` |

**Used by:** greenlang/api, GL-CBAM-APP, GL-VCCI-Carbon-APP

---

## Security Settings

### JWT Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `JWT_SECRET_KEY` | Yes | - | JWT signing secret (min 32 chars) | `openssl rand -hex 32` |
| `JWT_REFRESH_SECRET_KEY` | No | - | JWT refresh token secret | `openssl rand -hex 32` |
| `JWT_SECRET` | No | - | Alternative name for JWT secret | (same as JWT_SECRET_KEY) |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm | `HS256` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No | `30` | Access token expiry (minutes) | `30` |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | No | `7` | Refresh token expiry (days) | `7` |
| `JWT_EXPIRATION_SECONDS` | No | `3600` | Token expiration (seconds) | `3600` |
| `JWT_ISSUER` | No | `greenlang.ai` | JWT issuer claim | `greenlang.ai` |
| `JWT_AUDIENCE` | No | `greenlang-api` | JWT audience claim | `greenlang-api` |
| `JWT_LEEWAY` | No | `0` | JWT leeway for time validation | `0` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `30` | Alternative access token expiry | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | `7` | Alternative refresh token expiry | `7` |

**Used by:** greenlang/auth, greenlang/api, GL-CBAM-APP, GL-VCCI-Carbon-APP

### Encryption Keys

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `ENCRYPTION_KEY` | Yes | - | Primary encryption key (32 bytes hex) | `openssl rand -hex 32` |
| `ENCRYPTION_SALT` | No | - | Encryption salt (16 bytes hex) | `openssl rand -hex 16` |
| `FIELD_ENCRYPTION_KEY` | No | - | Field-level encryption key | (Fernet key) |
| `PROVENANCE_SIGNING_KEY` | No | - | Provenance signing key | (secure key) |
| `CSRD_ENCRYPTION_KEY` | No | - | CSRD-specific encryption key | (Fernet key) |
| `ENCRYPT_SENSITIVE_DATA` | No | `true` | Enable sensitive data encryption | `true` |
| `ENCRYPTION_KEY_ROTATION_DAYS` | No | `90` | Key rotation period (days) | `90` |
| `ENCRYPTION_KEY_GRACE_PERIOD_DAYS` | No | `30` | Grace period for key rotation | `30` |

**Used by:** Core platform, GL-CSRD-APP, GL-VCCI-Carbon-APP

### API Security

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `SECRET_KEY` | Yes | - | Application secret key | `openssl rand -base64 32` |
| `API_RATE_LIMIT_PER_MINUTE` | No | `100` | Rate limit per minute | `100` |
| `RATE_LIMIT_PER_MINUTE` | No | `100` | Alternative rate limit name | `100` |
| `RATE_LIMIT_PER_HOUR` | No | `1000` | Rate limit per hour | `1000` |
| `RATE_LIMIT_BURST` | No | `20` | Rate limit burst | `20` |
| `API_KEY_HEADER_NAME` | No | `X-API-Key` | API key header name | `X-API-Key` |
| `API_KEY_ENABLED` | No | `false` | Enable API key authentication | `true` |
| `JWT_ENABLED` | No | `false` | Enable JWT authentication | `true` |
| `CSRF_SECRET_KEY` | No | - | CSRF protection secret | (secure key) |

**Used by:** greenlang/api, GL-CBAM-APP

### CORS Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `CORS_ALLOWED_ORIGINS` | No | `*` | Allowed origins (comma-separated) | `https://app.greenlang.ai` |
| `CORS_ORIGINS` | No | - | Alternative name for allowed origins | `http://localhost:3000` |
| `CORS_ALLOWED_METHODS` | No | `GET,POST,PUT,DELETE,OPTIONS` | Allowed HTTP methods | `GET,POST,PUT,DELETE,OPTIONS` |
| `CORS_ALLOW_METHODS` | No | - | Alternative name for allowed methods | `GET,POST,PUT,DELETE,OPTIONS,PATCH` |
| `CORS_ALLOWED_HEADERS` | No | `*` | Allowed headers | `Content-Type,Authorization` |
| `CORS_ALLOW_HEADERS` | No | - | Alternative name for allowed headers | `Content-Type,Authorization,X-Requested-With` |
| `CORS_ALLOW_CREDENTIALS` | No | `true` | Allow credentials | `true` |
| `CORS_MAX_AGE` | No | `86400` | CORS max age (seconds) | `86400` |
| `ALLOWED_HOSTS` | No | - | Allowed hosts (production) | `api.company.com,*.company.com` |

**Used by:** greenlang/api, GL-CBAM-APP, GL-VCCI-Carbon-APP

### Session Security

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `SESSION_SECRET_KEY` | No | - | Session secret key | `openssl rand -hex 32` |
| `SESSION_COOKIE_SECURE` | No | `true` | Secure cookies (HTTPS only) | `true` |
| `SESSION_COOKIE_HTTPONLY` | No | `true` | HTTP-only cookies | `true` |
| `SESSION_COOKIE_SAMESITE` | No | `strict` | SameSite cookie policy | `strict` |

**Used by:** greenlang/api, GL-CBAM-APP

### Password Security

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `PASSWORD_HASH_SCHEMES` | No | `bcrypt` | Password hashing schemes | `bcrypt` |
| `PASSWORD_MIN_LENGTH` | No | `12` | Minimum password length | `12` |

**Used by:** GL-CBAM-APP, greenlang/auth

---

## External API Keys

### LLM Providers

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `OPENAI_API_KEY` | No | - | OpenAI API key | `sk-proj-xxx...` |
| `ANTHROPIC_API_KEY` | No | - | Anthropic (Claude) API key | `sk-ant-api03-xxx...` |
| `AZURE_OPENAI_API_KEY` | No | - | Azure OpenAI API key | (Azure key) |
| `AZURE_OPENAI_ENDPOINT` | No | - | Azure OpenAI endpoint | `https://resource.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | No | - | Azure OpenAI deployment name | `gpt-4-deployment` |

**Used by:** GL-CSRD-APP, GL-VCCI-Carbon-APP, intelligence module

### Vector Database

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `PINECONE_API_KEY` | No | - | Pinecone API key | (Pinecone key) |
| `PINECONE_ENVIRONMENT` | No | - | Pinecone environment | `us-east-1-aws` |
| `WEAVIATE_URL` | No | `http://localhost:8080` | Weaviate URL | `http://weaviate:8080` |
| `WEAVIATE_API_KEY` | No | - | Weaviate API key | (Weaviate key) |

**Used by:** GL-CSRD-APP, GL-VCCI-Carbon-APP

### Entity Resolution APIs

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `GLEIF_API_KEY` | No | - | GLEIF API key | (32-char key) |
| `GLEIF_API_URL` | No | `https://api.gleif.org/api/v1` | GLEIF API URL | `https://api.gleif.org/api/v1` |
| `GLEIF_RATE_LIMIT` | No | `10` | GLEIF rate limit (req/sec) | `10` |
| `DUNS_API_KEY` | No | - | Dun & Bradstreet API key | (D&B key) |
| `DUNS_API_SECRET` | No | - | Dun & Bradstreet API secret | (D&B secret) |
| `DUNS_API_URL` | No | `https://api.dnb.com/v1` | D&B API URL | `https://api.dnb.com/v1` |
| `DUNS_API_ENDPOINT` | No | `https://plus.dnb.com/v2` | Alternative D&B endpoint | `https://plus.dnb.com/v2` |
| `DUNS_SANDBOX_MODE` | No | `true` | Use D&B sandbox | `true` |
| `OPENCORPORATES_API_KEY` | No | - | OpenCorporates API key | (OC key) |
| `OPENCORPORATES_API_URL` | No | `https://api.opencorporates.com/v0.4` | OpenCorporates API URL | `https://api.opencorporates.com/v0.4` |

**Used by:** Core platform, GL-VCCI-Carbon-APP

### Satellite Data Providers

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `PLANET_API_KEY` | No | - | Planet Labs API key | (Planet key) |
| `SENTINEL_HUB_CLIENT_ID` | No | - | Sentinel Hub client ID | (Sentinel ID) |
| `SENTINEL_HUB_CLIENT_SECRET` | No | - | Sentinel Hub client secret | (Sentinel secret) |
| `NASA_EARTHDATA_USERNAME` | No | - | NASA Earthdata username | (NASA username) |
| `NASA_EARTHDATA_PASSWORD` | No | - | NASA Earthdata password | (NASA password) |

**Used by:** EUDR module, satellite ML features

### LCA Database

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `ECOINVENT_LICENSE_KEY` | No | - | Ecoinvent LCA database license | (license key) |

**Used by:** GL-VCCI-Carbon-APP

---

## Cloud Provider Configuration

### AWS Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | No | - | AWS access key ID | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | No | - | AWS secret access key | (40-char secret) |
| `AWS_REGION` | No | `us-east-1` | AWS region | `us-east-1` |
| `AWS_KMS_KEY_ID` | No | - | AWS KMS key ARN | `arn:aws:kms:...` |
| `AWS_PROFILE` | No | - | AWS profile name | `greenlang-prod` |
| `AWS_ROLE_ARN` | No | - | AWS IAM role ARN (for IRSA) | `arn:aws:iam::...` |
| `AWS_S3_BUCKET` | No | - | S3 bucket name | `greenlang-storage` |

**Used by:** Core platform, GL-CBAM-APP, GL-CSRD-APP, GL-VCCI-Carbon-APP

### Azure Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `AZURE_TENANT_ID` | No | - | Azure tenant ID (GUID) | `12345678-1234-...` |
| `AZURE_CLIENT_ID` | No | - | Azure client ID (GUID) | `12345678-1234-...` |
| `AZURE_CLIENT_SECRET` | No | - | Azure client secret | (Azure secret) |
| `AZURE_KEY_VAULT_URL` | No | - | Azure Key Vault URL | `https://vault.vault.azure.net/` |
| `AZURE_STORAGE_CONNECTION_STRING` | No | - | Azure Storage connection string | (connection string) |
| `AZURE_CONTAINER_NAME` | No | - | Azure blob container name | `csrd-reports` |

**Used by:** Core platform, GL-CSRD-APP

### GCP Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `GCP_SERVICE_ACCOUNT_KEY` | No | - | GCP service account key (Base64) | (Base64 JSON) |
| `GCP_PROJECT_ID` | No | - | GCP project ID | `greenlang-prod` |
| `GCP_KMS_KEYRING` | No | - | GCP KMS keyring name | `greenlang-keyring` |
| `GCP_KMS_KEY` | No | - | GCP KMS key name | `greenlang-key` |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | - | Path to GCP credentials file | `/path/to/credentials.json` |
| `GOOGLE_SERVICE_ACCOUNT` | No | - | GCP service account email | (email) |

**Used by:** Core platform

---

## Service Endpoints

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `GREENLANG_API_URL` | No | `http://localhost:8000` | GreenLang API URL | `https://api.greenlang.io` |
| `GREENLANG_FRONTEND_URL` | No | `http://localhost:3000` | Frontend URL | `https://app.greenlang.ai` |
| `GREENLANG_WEBHOOK_URL` | No | - | Webhook URL | `https://api.greenlang.io/webhooks` |
| `GREENLANG_METRICS_URL` | No | `http://localhost:9090` | Metrics URL | `http://prometheus:9090` |
| `VCCI_BASE_URL` | No | `http://vcci-backend-api:8000` | VCCI backend URL | `http://vcci-backend-api:8000` |
| `API_HOST` | No | `0.0.0.0` | API host binding | `0.0.0.0` |
| `API_PORT` | No | `8000` | API port | `8000` |
| `HOST` | No | `0.0.0.0` | Server host | `0.0.0.0` |
| `PORT` | No | `8000` | Server port | `8000` |
| `WORKERS` | No | `4` | Number of workers | `4` |

**Used by:** Core API, all applications

---

## Message Queue (RabbitMQ)

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `RABBITMQ_HOST` | No | `localhost` | RabbitMQ host | `rabbitmq.example.com` |
| `RABBITMQ_PORT` | No | `5672` | RabbitMQ port | `5672` |
| `RABBITMQ_MANAGEMENT_PORT` | No | `15672` | Management UI port | `15672` |
| `RABBITMQ_USER` | No | - | RabbitMQ username | `greenlang` |
| `RABBITMQ_PASSWORD` | No | - | RabbitMQ password | (secure password) |
| `RABBITMQ_VHOST` | No | `/greenlang` | RabbitMQ virtual host | `/greenlang` |
| `RABBITMQ_URL` | No | - | Full RabbitMQ URL | `amqp://user:pass@host:5672/vhost` |
| `RABBITMQ_DEFAULT_USER` | No | - | Default user (Docker) | `greenlang` |
| `RABBITMQ_DEFAULT_PASS` | No | - | Default password (Docker) | (secure password) |
| `RABBITMQ_DEFAULT_VHOST` | No | `/greenlang` | Default vhost (Docker) | `/greenlang` |

**Used by:** Core platform, cross-application integration

---

## Object Storage (S3)

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `S3_ENDPOINT_URL` | No | `https://s3.amazonaws.com` | S3 endpoint URL | `https://s3.amazonaws.com` |
| `S3_BUCKET_NAME` | No | - | S3 bucket name | `greenlang-storage` |
| `S3_BUCKET` | No | - | Alternative bucket name | `greenlang-storage` |
| `S3_ACCESS_KEY` | No | - | S3 access key | (access key) |
| `S3_SECRET_KEY` | No | - | S3 secret key | (secret key) |
| `S3_REGION` | No | `us-east-1` | S3 region | `us-east-1` |
| `S3_USE_SSL` | No | `true` | Use SSL for S3 | `true` |

**Used by:** Core platform, all applications

---

## Email Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `SMTP_HOST` | No | `smtp.gmail.com` | SMTP server host | `smtp.gmail.com` |
| `SMTP_PORT` | No | `587` | SMTP port | `587` |
| `SMTP_USER` | No | - | SMTP username | `noreply@greenlang.ai` |
| `SMTP_USERNAME` | No | - | Alternative SMTP username | `noreply@greenlang.ai` |
| `SMTP_PASSWORD` | No | - | SMTP password | (email password) |
| `SMTP_USE_TLS` | No | `true` | Enable TLS | `true` |
| `SMTP_FROM_EMAIL` | No | - | From email address | `noreply@greenlang.ai` |
| `EMAIL_FROM_ADDRESS` | No | - | Alternative from address | `noreply@greenlang.ai` |
| `EMAIL_FROM_NAME` | No | `GreenLang Platform` | From name | `GreenLang Platform` |
| `NOTIFICATION_EMAIL` | No | - | Notification recipient | `admin@company.com` |

**Used by:** GL-CBAM-APP, GL-CSRD-APP, GL-VCCI-Carbon-APP

### Third-Party Email Providers

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `EMAIL_PROVIDER` | No | `sendgrid` | Email provider | `sendgrid` |
| `SENDGRID_API_KEY` | No | - | SendGrid API key | (SendGrid key) |
| `SENDGRID_FROM_EMAIL` | No | - | SendGrid from email | `noreply@platform.com` |
| `SENDGRID_FROM_NAME` | No | - | SendGrid from name | `Platform` |
| `MAILGUN_API_KEY` | No | - | Mailgun API key | (Mailgun key) |
| `MAILGUN_DOMAIN` | No | - | Mailgun domain | `mg.company.com` |

**Used by:** GL-VCCI-Carbon-APP

---

## Logging Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `LOG_LEVEL` | No | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL | `INFO` |
| `LOG_FORMAT` | No | `json` | Log format: json, plain, text | `json` |
| `LOG_FILE_PATH` | No | `/var/log/greenlang/app.log` | Log file path | `/var/log/greenlang/app.log` |
| `LOG_FILE_MAX_SIZE` | No | `100` | Max log file size (MB) | `100` |
| `LOG_FILE_BACKUP_COUNT` | No | `10` | Number of backup files | `10` |
| `LOG_MAX_BYTES` | No | `10485760` | Max log bytes (10MB) | `10485760` |
| `LOG_BACKUP_COUNT` | No | `5` | Backup file count | `5` |
| `LOG_TO_STDOUT` | No | `true` | Log to stdout | `true` |
| `LOG_OUTPUT` | No | `both` | Output: console, file, both | `both` |
| `LOG_DESTINATION` | No | `stdout` | Destination: stdout, file, cloudwatch | `stdout` |
| `LOG_INCLUDE_TIMESTAMP` | No | `true` | Include timestamp | `true` |
| `LOG_INCLUDE_HOSTNAME` | No | `true` | Include hostname | `true` |
| `LOG_INCLUDE_PROCESS_ID` | No | `true` | Include PID | `true` |
| `LOG_INCLUDE_THREAD_ID` | No | `false` | Include thread ID | `false` |
| `VERBOSE_LOGGING` | No | `false` | Enable verbose logging | `false` |

**Used by:** All applications

### External Logging Services

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `DATADOG_API_KEY` | No | - | Datadog API key | (Datadog key) |
| `DATADOG_APP_KEY` | No | - | Datadog application key | (Datadog app key) |
| `DATADOG_SERVICE_NAME` | No | `greenlang-api` | Datadog service name | `greenlang-api` |
| `DATADOG_ENV` | No | `production` | Datadog environment | `production` |
| `SENTRY_DSN` | No | - | Sentry DSN | `https://...@sentry.io/...` |
| `SENTRY_ENVIRONMENT` | No | `production` | Sentry environment | `production` |
| `SENTRY_TRACES_SAMPLE_RATE` | No | `0.1` | Sentry trace sample rate | `0.1` |
| `NEW_RELIC_LICENSE_KEY` | No | - | New Relic license key | (NR key) |
| `NEW_RELIC_APP_NAME` | No | `GreenLang API` | New Relic app name | `GreenLang API` |

**Used by:** All applications

---

## Monitoring and Observability

### Prometheus Metrics

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `PROMETHEUS_METRICS_PORT` | No | `9090` | Prometheus metrics port | `9090` |
| `PROMETHEUS_METRICS_PATH` | No | `/metrics` | Metrics endpoint path | `/metrics` |
| `PROMETHEUS_PORT` | No | `9090` | Prometheus server port | `9090` |
| `PROMETHEUS_RETENTION_TIME` | No | `15d` | Data retention time | `15d` |
| `PROMETHEUS_RETENTION_SIZE` | No | `10GB` | Data retention size | `10GB` |
| `PROMETHEUS_ENABLED` | No | `true` | Enable Prometheus | `true` |
| `ENABLE_METRICS` | No | `true` | Enable metrics collection | `true` |
| `METRICS_PORT` | No | `9090` | Metrics port | `9090` |

**Used by:** All applications

### StatsD

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `STATSD_HOST` | No | `localhost` | StatsD host | `localhost` |
| `STATSD_PORT` | No | `8125` | StatsD port | `8125` |
| `STATSD_PREFIX` | No | `greenlang.` | StatsD metric prefix | `greenlang.` |

**Used by:** Core platform

### Health Checks

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `HEALTH_CHECK_PATH` | No | `/health` | Health check endpoint | `/health` |
| `HEALTH_CHECK_ENABLED` | No | `true` | Enable health checks | `true` |
| `HEALTH_CHECK_INCLUDE_DB` | No | `true` | Include database in health check | `true` |
| `HEALTH_CHECK_INCLUDE_REDIS` | No | `true` | Include Redis in health check | `true` |
| `HEALTH_CHECK_INCLUDE_EXTERNAL_APIS` | No | `false` | Include external APIs | `false` |
| `HEALTH_CHECK_INTERVAL` | No | `30s` | Health check interval | `30s` |
| `HEALTH_CHECK_TIMEOUT` | No | `10s` | Health check timeout | `10s` |
| `HEALTH_CHECK_RETRIES` | No | `3` | Health check retry count | `3` |
| `HEALTH_CHECK_START_PERIOD` | No | `40s` | Start period before checks | `40s` |

**Used by:** All applications

### Distributed Tracing

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `ENABLE_TRACING` | No | `false` | Enable distributed tracing | `true` |
| `JAEGER_AGENT_HOST` | No | `localhost` | Jaeger agent host | `jaeger` |
| `JAEGER_AGENT_PORT` | No | `6831` | Jaeger agent port | `6831` |
| `JAEGER_COLLECTOR_PORT` | No | `14268` | Jaeger collector port | `14268` |
| `JAEGER_UI_PORT` | No | `16686` | Jaeger UI port | `16686` |
| `JAEGER_SERVICE_NAME` | No | `greenlang-api` | Jaeger service name | `greenlang-api` |
| `JAEGER_SAMPLING_RATE` | No | `0.01` | Jaeger sampling rate | `0.01` |
| `ZIPKIN_ENDPOINT` | No | - | Zipkin endpoint | `http://localhost:9411/api/v2/spans` |
| `ZIPKIN_SERVICE_NAME` | No | `greenlang-api` | Zipkin service name | `greenlang-api` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | `http://localhost:4317` | OpenTelemetry endpoint | `http://collector:4317` |
| `OTEL_SERVICE_NAME` | No | - | OpenTelemetry service name | `gl-cbam-api` |

**Used by:** All applications

### Grafana

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `GRAFANA_PORT` | No | `3001` | Grafana port | `3001` |
| `GRAFANA_USER` | No | `admin` | Grafana admin user | `admin` |
| `GRAFANA_PASSWORD` | No | - | Grafana admin password | (secure password) |
| `GRAFANA_ADMIN_USER` | No | `admin` | Alternative admin user | `admin` |
| `GRAFANA_ADMIN_PASSWORD` | No | - | Alternative admin password | (secure password) |
| `GRAFANA_ANONYMOUS_ENABLED` | No | `false` | Enable anonymous access | `false` |
| `GRAFANA_INSTALL_PLUGINS` | No | - | Plugins to install | `grafana-piechart-panel` |

**Used by:** Monitoring stack

---

## Performance Settings

### Application Performance

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `APP_WORKERS` | No | `4` | Number of worker processes | `4` |
| `WORKER_PROCESSES` | No | `4` | Alternative worker count | `4` |
| `MAX_WORKERS` | No | `12` | Maximum concurrent workers | `12` |
| `APP_THREADS_PER_WORKER` | No | `4` | Threads per worker | `4` |
| `APP_MAX_REQUESTS` | No | `1000` | Max requests before worker restart | `1000` |
| `MAX_REQUESTS` | No | `1000` | Alternative max requests | `1000` |
| `APP_MAX_REQUESTS_JITTER` | No | `50` | Jitter for max requests | `50` |
| `MAX_REQUESTS_JITTER` | No | `100` | Alternative jitter | `100` |
| `WORKER_TIMEOUT` | No | `120` | Worker timeout (seconds) | `120` |
| `KEEP_ALIVE` | No | `5` | Keep-alive timeout (seconds) | `5` |
| `REQUEST_TIMEOUT` | No | `300` | Request timeout (seconds) | `300` |
| `CONNECTION_POOL_SIZE` | No | `20` | Connection pool size | `20` |
| `CONNECTION_POOL_TIMEOUT` | No | `30` | Connection pool timeout | `30` |

**Used by:** All applications

### Batch Processing

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `BATCH_SIZE_DEFAULT` | No | `1000` | Default batch size | `1000` |
| `BATCH_SIZE` | No | `10000` | Batch size for VCCI | `10000` |
| `BATCH_PROCESSING_TIMEOUT` | No | `600` | Batch processing timeout (seconds) | `600` |
| `BATCH_MAX_PARALLEL_JOBS` | No | `5` | Max parallel batch jobs | `5` |

**Used by:** Core platform, GL-VCCI-Carbon-APP

### File Upload

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `MAX_UPLOAD_SIZE_MB` | No | `100` | Maximum upload size (MB) | `100` |
| `MAX_FILE_SIZE_MB` | No | `500` | Maximum file size (MB) | `500` |
| `UPLOAD_DIR` | No | `/app/uploads` | Upload directory | `/app/uploads` |
| `ALLOWED_UPLOAD_EXTENSIONS` | No | `.csv,.xlsx,.json,.yaml,.yml` | Allowed extensions | `.csv,.xlsx,.json,.yaml,.yml` |
| `FILE_PROCESSING_TIMEOUT` | No | `300` | File processing timeout (seconds) | `300` |

**Used by:** GL-CBAM-APP, GL-CSRD-APP

---

## Feature Flags

See [FEATURE_FLAGS.md](./FEATURE_FLAGS.md) for detailed feature flag documentation.

### Core Features

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEATURE_ZERO_HALLUCINATION` | No | `true` | Enable zero-hallucination mode |
| `FEATURE_PROVENANCE_TRACKING` | No | `true` | Enable provenance tracking |
| `FEATURE_ASYNC_PROCESSING` | No | `true` | Enable async processing |
| `FEATURE_BATCH_PROCESSING` | No | `true` | Enable batch processing |
| `FEATURE_REAL_TIME_VALIDATION` | No | `true` | Enable real-time validation |

### AI Features

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEATURE_LLM_CLASSIFICATION` | No | `true` | Enable LLM classification |
| `FEATURE_LLM_NARRATIVE_GEN` | No | `true` | Enable LLM narrative generation |
| `FEATURE_SATELLITE_ML` | No | `false` | Enable satellite ML |
| `FEATURE_ANOMALY_DETECTION` | No | `true` | Enable anomaly detection |

### Regulatory Features

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEATURE_CBAM_MODULE` | No | `true` | Enable CBAM module |
| `FEATURE_CSRD_MODULE` | No | `true` | Enable CSRD module |
| `FEATURE_SB253_MODULE` | No | `true` | Enable SB253 module |
| `FEATURE_EUDR_MODULE` | No | `false` | Enable EUDR module |
| `FEATURE_TAXONOMY_MODULE` | No | `true` | Enable EU Taxonomy module |

### Integration Features

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEATURE_SAP_INTEGRATION` | No | `false` | Enable SAP integration |
| `FEATURE_ORACLE_INTEGRATION` | No | `false` | Enable Oracle integration |
| `FEATURE_SALESFORCE_INTEGRATION` | No | `false` | Enable Salesforce integration |
| `FEATURE_AZURE_IOT_INTEGRATION` | No | `false` | Enable Azure IoT integration |

---

## GreenLang-Specific Settings

### Platform Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `GREENLANG_VERSION` | No | `1.0.0` | Platform version | `1.0.0` |
| `GREENLANG_TENANT_ID` | No | `default` | Tenant ID | `tenant-123` |
| `GREENLANG_WORKSPACE_ID` | No | `main` | Workspace ID | `workspace-1` |
| `GREENLANG_DEFAULT_LANGUAGE` | No | `en` | Default language | `en` |
| `GREENLANG_DEFAULT_CURRENCY` | No | `USD` | Default currency | `USD` |
| `GREENLANG_DEFAULT_TIMEZONE` | No | `UTC` | Default timezone | `UTC` |
| `GREENLANG_REGION` | No | `US` | GreenLang region | `US` |
| `GREENLANG_REPORT_FORMAT` | No | `json` | Report format: json, text, markdown | `json` |
| `GREENLANG_DATA_DIR` | No | `/tmp/greenlang/data` | Data directory | `/tmp/greenlang/data` |
| `GREENLANG_UPLOAD_DIR` | No | `/tmp/greenlang/uploads` | Upload directory | `/tmp/greenlang/uploads` |
| `GREENLANG_REPORT_DIR` | No | `/tmp/greenlang/reports` | Report directory | `/tmp/greenlang/reports` |
| `GREENLANG_API_KEY` | No | - | GreenLang API key | (API key) |
| `GREENLANG_TOKEN` | No | - | GreenLang registry token | (token) |
| `GREENLANG_DEBUG_MODE` | No | `false` | Enable debug mode | `false` |
| `GREENLANG_DEV_MODE` | No | `false` | Enable development mode | `false` |

**Used by:** Core platform, all applications

### Agent Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `AGENT_TIMEOUT_SECONDS` | No | `300` | Agent timeout (seconds) | `300` |
| `AGENT_MAX_RETRIES` | No | `3` | Maximum retries | `3` |
| `AGENT_RETRY_DELAY_SECONDS` | No | `5` | Retry delay (seconds) | `5` |
| `AGENT_PARALLEL_EXECUTION` | No | `true` | Enable parallel execution | `true` |

**Used by:** Core platform

### Formula Engine

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `FORMULA_ENGINE_CACHE_SIZE` | No | `1000` | Formula cache size | `1000` |
| `FORMULA_ENGINE_PRECISION` | No | `6` | Decimal precision | `6` |
| `FORMULA_ENGINE_ROUNDING_MODE` | No | `half_up` | Rounding mode | `half_up` |

**Used by:** Core platform

### Emission Factors

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `EMISSION_FACTORS_SOURCE` | No | `EPA` | Source: EPA, DEFRA, IPCC, custom | `EPA` |
| `EMISSION_FACTORS_VERSION` | No | `2024.1` | Version | `2024.1` |
| `EMISSION_FACTORS_UPDATE_FREQUENCY` | No | `monthly` | Update frequency | `monthly` |
| `EMISSION_FACTORS_PATH` | No | - | Path to emission factors | `/app/data/emission_factors.json` |
| `USE_DATABASE_FACTORS` | No | `false` | Use database factors | `true` |

**Used by:** Core platform, GL-CBAM-APP

### Supply Chain

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `SUPPLY_CHAIN_TIERS` | No | `5` | Number of tiers to track | `5` |
| `SUPPLY_CHAIN_DATA_QUALITY_THRESHOLD` | No | `80` | Data quality threshold (%) | `80` |
| `SUPPLY_CHAIN_RISK_SCORING_ENABLED` | No | `true` | Enable risk scoring | `true` |

**Used by:** Core platform, GL-VCCI-Carbon-APP

### CLI Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `GL_LOG_LEVEL` | No | `INFO` | CLI log level | `DEBUG` |
| `GL_LOG_DIR` | No | `/var/log/greenlang` | CLI log directory | `/var/log/greenlang` |
| `GL_CACHE_DIR` | No | `~/.greenlang/cache` | CLI cache directory | `~/.greenlang/cache` |
| `GL_STATE_DIR` | No | `~/.greenlang` | CLI state directory | `~/.greenlang` |
| `GL_BACKEND` | No | `docker` | CLI backend | `docker` |
| `GL_EXEC_MODE` | No | `live` | Execution mode | `live` |
| `GL_PROFILE` | No | - | CLI profile | `production` |
| `GL_REGION` | No | - | CLI region | `US` |
| `GL_HUB` | No | `hub.greenlang.io` | Hub URL | `hub.greenlang.io` |
| `GL_HUB_TOKEN` | No | - | Hub authentication token | (token) |
| `GL_TELEMETRY` | No | `on` | Telemetry setting | `on` |
| `GL_TELEMETRY_ENABLED` | No | `true` | Enable telemetry | `true` |
| `GL_POLICY_BUNDLE` | No | - | Policy bundle name | `enterprise-policies` |
| `GL_ALLOW_UNSIGNED` | No | - | Allow unsigned packs | `1` |
| `GL_VERIFY_SIGNATURES` | No | `true` | Verify signatures | `true` |
| `GL_SANDBOX_ENABLED` | No | `true` | Enable sandbox | `true` |
| `GL_DEBUG` | No | `false` | Enable debug mode | `true` |
| `GL_ENV` | No | `development` | Environment | `production` |
| `GL_ENVIRONMENT` | No | - | Alternative environment name | `production` |
| `GL_SECRET_KEY` | No | - | CLI secret key | (secret key) |
| `GL_SECRET_PATH` | No | - | Path to secret key file | `~/.greenlang/secret.key` |
| `GL_CA_BUNDLE` | No | - | CA bundle path | `/path/to/ca-bundle.crt` |
| `GL_ALLOWED_DOMAINS` | No | - | Allowed domains (comma-separated) | `greenlang.io,api.greenlang.io` |
| `GL_BLOCKED_DOMAINS` | No | - | Blocked domains (comma-separated) | `malicious.com` |
| `GL_ENABLE_PLUGINS` | No | - | Enable plugins | `1` |
| `GL_MODE` | No | - | Operation mode | `replay` |

**Used by:** GreenLang CLI, core platform

### Intelligence Module

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `GREENLANG_INTELLIGENCE_MODEL` | No | - | Intelligence model name | `gpt-4` |
| `GREENLANG_INTELLIGENCE_TIMEOUT` | No | - | Intelligence timeout (seconds) | `30` |
| `GREENLANG_INTELLIGENCE_MAX_RETRIES` | No | - | Max retries | `3` |

**Used by:** greenlang/intelligence

---

## Application-Specific Variables

### GL-CBAM-APP

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `APP_ENV` | No | `production` | Application environment | `production` |
| `APP_VERSION` | No | `1.0.0` | Application version | `1.0.0` |
| `API_TITLE` | No | `GL-CBAM-APP API` | API title | `GL-CBAM-APP API` |
| `API_DESCRIPTION` | No | - | API description | (description) |
| `API_VERSION` | No | `1.0.0` | API version | `1.0.0` |
| `ENABLE_API_DOCS` | No | `true` | Enable API docs | `true` |
| `ENABLE_REDOC` | No | `true` | Enable ReDoc | `true` |
| `DEFAULT_QUARTER` | No | `Q4-2024` | Default reporting quarter | `Q4-2024` |
| `CBAM_REGISTRY_URL` | No | `https://cbam-registry.ec.europa.eu` | CBAM registry URL | (URL) |
| `CN_CODES_PATH` | No | `/app/data/cn_codes.json` | CN codes path | (path) |
| `CBAM_RULES_PATH` | No | `/app/rules/cbam_rules.yaml` | CBAM rules path | (path) |
| `SUPPLIERS_PATH` | No | `/app/examples/demo_suppliers.yaml` | Suppliers path | (path) |
| `ENABLE_CACHING` | No | `true` | Enable caching | `true` |
| `ENABLE_BACKGROUND_TASKS` | No | `true` | Enable background tasks | `true` |
| `ENABLE_EXPERIMENTAL_FEATURES` | No | `false` | Enable experimental features | `false` |

**Used by:** GL-CBAM-APP

### GL-CSRD-APP

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `OUTPUT_DIR` | No | `output` | Output directory | `output` |
| `ENABLE_AI_MATERIALITY` | No | `true` | Enable AI materiality assessment | `true` |
| `ENABLE_AI_NARRATIVES` | No | `true` | Enable AI narratives | `true` |
| `ENABLE_XBRL_GENERATION` | No | `true` | Enable XBRL generation | `true` |
| `ENABLE_EMAIL_NOTIFICATIONS` | No | `false` | Enable email notifications | `true` |
| `LOG_SQL_QUERIES` | No | `false` | Log SQL queries | `false` |
| `DETAILED_ERROR_TRACES` | No | `true` | Show detailed error traces | `true` |
| `DEFAULT_COMPANY_NAME` | No | `Demo Manufacturing GmbH` | Default company name | (name) |
| `DEFAULT_LEI_CODE` | No | `529900DEMO00000000001` | Default LEI code | (LEI) |
| `DEFAULT_REPORTING_YEAR` | No | `2024` | Default reporting year | `2024` |
| `DEFAULT_COMPANY_COUNTRY` | No | `DE` | Default country code | `DE` |

**Used by:** GL-CSRD-APP

### GL-VCCI-Carbon-APP

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `VCCI_ENVIRONMENT` | No | `production` | VCCI environment | `production` |
| `VCCI_VERSION` | No | `1.0.0` | VCCI version | `1.0.0` |
| `VCCI_CONFIG_PATH` | No | `config/vcci_config.yaml` | Config path | (path) |
| `VCCI_OUTPUT_DIR` | No | `./output` | Output directory | `./output` |
| `TENANT_ID` | No | `default` | Tenant ID | `tenant-123` |
| `COMPANY_NAME` | No | - | Company name | `Your Company` |
| `REPORT_FOOTER` | No | - | Report footer | (footer text) |
| `FEATURE_ENTITY_RESOLUTION` | No | `true` | Enable entity resolution | `true` |
| `FEATURE_LLM_CATEGORIZATION` | No | `true` | Enable LLM categorization | `true` |
| `FEATURE_SUPPLIER_ENGAGEMENT` | No | `true` | Enable supplier engagement | `true` |
| `FEATURE_AUTOMATED_REPORTING` | No | `true` | Enable automated reporting | `true` |
| `FEATURE_SCENARIO_MODELING` | No | `true` | Enable scenario modeling | `true` |
| `FEATURE_REAL_TIME_MONITORING` | No | `true` | Enable real-time monitoring | `true` |
| `FEATURE_BLOCKCHAIN_PROVENANCE` | No | `false` | Enable blockchain provenance | `false` |
| `FEATURE_SATELLITE_MONITORING` | No | `false` | Enable satellite monitoring | `false` |
| `FEATURE_MOBILE_APP` | No | `false` | Enable mobile app | `false` |
| `DISABLE_AUTH` | No | `false` | Disable authentication | `false` |
| `MOCK_ERP_CONNECTIONS` | No | `false` | Mock ERP connections | `false` |
| `MOCK_LLM_RESPONSES` | No | `false` | Mock LLM responses | `false` |
| `DATA_RETENTION_YEARS` | No | `7` | Data retention (years) | `7` |
| `ENABLE_AUDIT_LOG` | No | `true` | Enable audit logging | `true` |
| `LOG_ALL_CALCULATIONS` | No | `true` | Log all calculations | `true` |
| `LOG_ALL_API_CALLS` | No | `true` | Log all API calls | `true` |
| `LOG_ALL_DATA_ACCESS` | No | `true` | Log all data access | `true` |
| `MAX_LLM_TOKENS_PER_DAY` | No | `1000000` | Max LLM tokens per day | `1000000` |
| `MAX_LLM_COST_PER_DAY_USD` | No | `100` | Max LLM cost per day (USD) | `100` |
| `LLM_ALERT_THRESHOLD` | No | `0.80` | LLM alert threshold | `0.80` |
| `VCCI_PROVENANCE_BUCKET` | No | - | Provenance bucket | `vcci-scope3-provenance` |
| `VCCI_PROVENANCE_REGION` | No | `us-west-2` | Provenance region | `us-west-2` |
| `USE_GREENLANG_INFRASTRUCTURE` | No | `true` | Use GreenLang infrastructure | `true` |

**Used by:** GL-VCCI-Carbon-APP

### ERP Integration (GL-VCCI-Carbon-APP)

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `SAP_API_ENDPOINT` | No | - | SAP S/4HANA endpoint | `https://sap.company.com/...` |
| `SAP_OAUTH_CLIENT_ID` | No | - | SAP OAuth client ID | (client ID) |
| `SAP_OAUTH_CLIENT_SECRET` | No | - | SAP OAuth client secret | (secret) |
| `SAP_OAUTH_TOKEN_URL` | No | - | SAP OAuth token URL | (URL) |
| `ORACLE_API_ENDPOINT` | No | - | Oracle ERP endpoint | `https://oracle.company.com` |
| `ORACLE_OAUTH_CLIENT_ID` | No | - | Oracle OAuth client ID | (client ID) |
| `ORACLE_OAUTH_CLIENT_SECRET` | No | - | Oracle OAuth client secret | (secret) |
| `ORACLE_OAUTH_TOKEN_URL` | No | - | Oracle OAuth token URL | (URL) |
| `WORKDAY_API_ENDPOINT` | No | - | Workday endpoint | `https://wd2-impl-services1.workday.com` |
| `WORKDAY_OAUTH_CLIENT_ID` | No | - | Workday OAuth client ID | (client ID) |
| `WORKDAY_OAUTH_CLIENT_SECRET` | No | - | Workday OAuth client secret | (secret) |
| `WORKDAY_OAUTH_TOKEN_URL` | No | - | Workday OAuth token URL | (URL) |

**Used by:** GL-VCCI-Carbon-APP

---

## Development and Testing

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `ENVIRONMENT` | Yes | `development` | Environment: development, staging, production, test | `development` |
| `DEBUG` | No | `false` | Enable debug mode | `true` |
| `TESTING` | No | `false` | Enable testing mode | `true` |
| `PROFILING` | No | `false` | Enable profiling | `false` |
| `ENABLE_PROFILING` | No | `false` | Alternative profiling flag | `false` |
| `SQL_DEBUG` | No | `false` | Enable SQL debugging | `false` |
| `RELOAD` | No | `false` | Enable hot reload | `true` |
| `HOT_RELOAD` | No | `false` | Alternative hot reload flag | `true` |
| `FORCE_COLOR` | No | `true` | Force colored output | `true` |
| `DEV_DB_SEED` | No | `true` | Seed development database | `true` |
| `DEV_DB_SAMPLE_DATA_SIZE` | No | `1000` | Sample data size | `1000` |
| `TEST_DATABASE_URL` | No | - | Test database URL | `postgresql://test:test@localhost/test_db` |
| `TEST_REDIS_URL` | No | `redis://localhost:6379/1` | Test Redis URL | `redis://localhost:6379/1` |
| `TEST_COVERAGE_THRESHOLD` | No | `85` | Test coverage threshold (%) | `85` |
| `TEST_PARALLEL_WORKERS` | No | `4` | Parallel test workers | `4` |

**Used by:** All applications

---

## Deployment Configuration

### Container Configuration

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `COMPOSE_PROJECT_NAME` | No | `greenlang` | Docker Compose project name | `greenlang` |
| `DOCKER_REGISTRY` | No | `docker.io/greenlang` | Docker registry | `docker.io/greenlang` |
| `DOCKER_IMAGE_TAG` | No | `latest` | Docker image tag | `v1.0.0` |
| `CONTAINER_PORT` | No | `8000` | Container port | `8000` |
| `CONTAINER_MEMORY_LIMIT` | No | `2048` | Memory limit (MB) | `2048` |
| `CONTAINER_CPU_LIMIT` | No | `2000` | CPU limit (millicores) | `2000` |
| `BUILD_DATE` | No | - | Build date (set by CI/CD) | `2024-01-15T10:30:00Z` |

**Used by:** Docker, CI/CD

### Kubernetes

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `K8S_NAMESPACE` | No | `greenlang-prod` | Kubernetes namespace | `greenlang-prod` |
| `K8S_SERVICE_ACCOUNT` | No | `greenlang-api` | Service account name | `greenlang-api` |
| `K8S_CONFIG_MAP_NAME` | No | `greenlang-config` | ConfigMap name | `greenlang-config` |
| `K8S_SECRET_NAME` | No | `greenlang-secrets` | Secret name | `greenlang-secrets` |
| `KUBERNETES_SERVICE_HOST` | No | - | Kubernetes service host (auto-set) | (auto) |

**Used by:** Kubernetes deployments

### Cloud Deployment

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `CLOUD_PROVIDER` | No | `aws` | Cloud provider: aws, azure, gcp | `aws` |
| `CLOUD_REGION` | No | `us-east-1` | Cloud region | `us-east-1` |
| `CLOUD_ZONE` | No | `us-east-1a` | Cloud zone | `us-east-1a` |
| `CLOUD_RESOURCE_GROUP` | No | - | Resource group name | `greenlang-prod` |

**Used by:** Cloud deployments

### Networking

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `DOCKER_NETWORK` | No | `greenlang-network` | Docker network name | `greenlang-network` |
| `DOCKER_NETWORK_DRIVER` | No | `bridge` | Network driver | `bridge` |
| `SERVICE_DISCOVERY_ENABLED` | No | `true` | Enable service discovery | `true` |
| `DNS_RESOLVER` | No | `127.0.0.11` | DNS resolver | `127.0.0.11` |

**Used by:** Docker Compose

### Timezone

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `TZ` | No | `UTC` | Timezone | `UTC` |
| `LOCALE` | No | `en_US.UTF-8` | Locale | `en_US.UTF-8` |

**Used by:** All applications

---

## Compliance and Audit

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `AUDIT_LOG_ENABLED` | No | `true` | Enable audit logging | `true` |
| `AUDIT_LOG_TABLE` | No | `audit_logs` | Audit log table name | `audit_logs` |
| `AUDIT_LOG_RETENTION_DAYS` | No | `2555` | Audit log retention (7 years) | `2555` |
| `AUDIT_SENSITIVE_FIELDS` | No | `password,ssn,credit_card` | Sensitive fields to mask | `password,ssn` |
| `COMPLIANCE_MODE` | No | `strict` | Compliance mode: strict, standard, relaxed | `strict` |
| `GDPR_ENABLED` | No | `true` | Enable GDPR compliance | `true` |
| `GDPR_DATA_RETENTION_DAYS` | No | `1095` | GDPR data retention (3 years) | `1095` |
| `SOC2_ENABLED` | No | `true` | Enable SOC2 compliance | `true` |
| `ISO27001_ENABLED` | No | `false` | Enable ISO 27001 compliance | `true` |
| `DATA_RESIDENCY_REGION` | No | `EU` | Data residency region | `EU` |
| `DATA_ENCRYPTION_AT_REST` | No | `true` | Enable encryption at rest | `true` |
| `DATA_ENCRYPTION_IN_TRANSIT` | No | `true` | Enable encryption in transit | `true` |

**Used by:** Core platform, all applications

---

## Backup and Recovery

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `BACKUP_ENABLED` | No | `true` | Enable backups | `true` |
| `ENABLE_DB_BACKUP` | No | `true` | Enable database backups | `true` |
| `BACKUP_SCHEDULE` | No | `0 2 * * *` | Backup schedule (cron) | `0 2 * * *` |
| `DB_BACKUP_SCHEDULE` | No | `0 2 * * *` | DB backup schedule | `0 2 * * *` |
| `BACKUP_RETENTION_DAYS` | No | `30` | Backup retention (days) | `30` |
| `DB_BACKUP_RETENTION_DAYS` | No | `30` | DB backup retention | `30` |
| `BACKUP_STORAGE_PATH` | No | `s3://greenlang-backups` | Backup storage path | `s3://greenlang-backups` |
| `BACKUP_ENCRYPTION_KEY` | No | - | Backup encryption key | (encryption key) |
| `BACKUP_S3_BUCKET` | No | - | Backup S3 bucket | `gl-cbam-backups` |
| `BACKUP_S3_PREFIX` | No | `database/` | Backup S3 prefix | `database/` |
| `RECOVERY_POINT_OBJECTIVE` | No | `4` | RPO (hours) | `4` |
| `RECOVERY_TIME_OBJECTIVE` | No | `8` | RTO (hours) | `8` |
| `DISASTER_RECOVERY_REGION` | No | `us-west-2` | DR region | `us-west-2` |

**Used by:** Core platform, all applications

---

## Sensitive Variables Handling

The following variables contain sensitive information and require special handling:

### Secrets (Must be securely managed)

| Variable | Category | Notes |
|----------|----------|-------|
| `POSTGRES_PASSWORD` | Database | Use secrets manager |
| `REDIS_PASSWORD` | Cache | Use secrets manager |
| `JWT_SECRET_KEY` | Security | Minimum 32 characters, rotate regularly |
| `JWT_REFRESH_SECRET_KEY` | Security | Minimum 32 characters |
| `ENCRYPTION_KEY` | Security | 32 bytes hex, critical for data encryption |
| `FIELD_ENCRYPTION_KEY` | Security | Fernet key for field-level encryption |
| `SECRET_KEY` | Security | Application secret |
| `SESSION_SECRET_KEY` | Security | Session encryption |
| `CSRF_SECRET_KEY` | Security | CSRF protection |

### API Keys (Sensitive, restrict access)

| Variable | Category | Notes |
|----------|----------|-------|
| `OPENAI_API_KEY` | External API | Rate-limited, monitor usage |
| `ANTHROPIC_API_KEY` | External API | Rate-limited, monitor usage |
| `AWS_ACCESS_KEY_ID` | Cloud | Use IAM roles when possible |
| `AWS_SECRET_ACCESS_KEY` | Cloud | Never commit to repository |
| `AZURE_CLIENT_SECRET` | Cloud | Use managed identity when possible |
| `GCP_SERVICE_ACCOUNT_KEY` | Cloud | Base64-encoded, use workload identity |
| `GLEIF_API_KEY` | External API | Commercial license |
| `DUNS_API_KEY` | External API | Commercial license |
| `DUNS_API_SECRET` | External API | Commercial license |

### Email Credentials

| Variable | Category | Notes |
|----------|----------|-------|
| `SMTP_PASSWORD` | Email | Use app-specific passwords |
| `SENDGRID_API_KEY` | Email | Restrict permissions |

### Monitoring Credentials

| Variable | Category | Notes |
|----------|----------|-------|
| `DATADOG_API_KEY` | Monitoring | Read/write access |
| `SENTRY_DSN` | Monitoring | Contains project identifier |
| `NEW_RELIC_LICENSE_KEY` | Monitoring | Account-level access |

### Best Practices

1. **Never commit secrets to version control**
   - Use `.gitignore` to exclude `.env` files
   - Use git-secrets or similar tools

2. **Use secrets managers in production**
   - AWS Secrets Manager
   - Azure Key Vault
   - HashiCorp Vault
   - Google Secret Manager

3. **Rotate secrets regularly**
   - JWT secrets: Every 90 days
   - API keys: Every 180 days
   - Database passwords: Every 90 days

4. **Use environment-specific secrets**
   - Separate secrets for dev, staging, production
   - Never reuse production secrets in development

5. **Audit secret access**
   - Enable logging for secret access
   - Review access patterns regularly

6. **Minimum privilege**
   - Create API keys with minimum required permissions
   - Use read-only tokens where possible

---

## Total Variables Documented

| Category | Count |
|----------|-------|
| Database Configuration | 28 |
| Cache Configuration | 17 |
| Security Settings | 35 |
| External API Keys | 23 |
| Cloud Providers | 17 |
| Service Endpoints | 10 |
| Message Queue | 10 |
| Object Storage | 7 |
| Email Configuration | 15 |
| Logging | 25 |
| Monitoring | 35 |
| Performance | 20 |
| Feature Flags | 25 |
| GreenLang-Specific | 50 |
| Application-Specific | 45 |
| Development | 15 |
| Deployment | 18 |
| Compliance | 12 |
| Backup | 12 |
| **Total** | **~420** |

---

## See Also

- [FEATURE_FLAGS.md](./FEATURE_FLAGS.md) - Detailed feature flag documentation
- [.env.template](../.env.template) - Full environment template
- [.env.example](../.env.example) - Environment example file
- [docker-compose.env.example](../docker-compose.env.example) - Docker Compose environment
- [config/env_validator.py](../config/env_validator.py) - Environment validation tool
