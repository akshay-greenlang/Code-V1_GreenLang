# GreenLang Production Deployment Guide

**Comprehensive deployment documentation for GreenLang Climate Operating System**

Version: 1.0.0
Last Updated: November 2025

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Environment Variables Reference](#environment-variables-reference)
4. [Local Development Deployment](#local-development-deployment)
5. [Docker Single Container](#docker-single-container)
6. [Docker Compose Multi-Service](#docker-compose-multi-service)
7. [Kubernetes Production Deployment](#kubernetes-production-deployment)
8. [Cloud Platform Deployments](#cloud-platform-deployments)
9. [Database Setup](#database-setup)
10. [Secrets Management](#secrets-management)
11. [Monitoring and Observability](#monitoring-and-observability)
12. [Load Balancer Configuration](#load-balancer-configuration)
13. [SSL/TLS Certificates](#ssltls-certificates)
14. [Scaling and Autoscaling](#scaling-and-autoscaling)
15. [Backup and Recovery](#backup-and-recovery)
16. [Rollback Procedures](#rollback-procedures)
17. [Health Checks](#health-checks)
18. [Security Hardening](#security-hardening)
19. [Troubleshooting](#troubleshooting)
20. [Performance Tuning](#performance-tuning)

---

## Prerequisites

### System Requirements

**Minimum Requirements (Development):**
- CPU: 2 cores
- RAM: 4 GB
- Storage: 20 GB SSD
- OS: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ (with WSL2)

**Recommended Requirements (Production):**
- CPU: 8+ cores
- RAM: 16 GB+
- Storage: 100 GB SSD (NVMe preferred)
- OS: Linux (Ubuntu 22.04 LTS or RHEL 8+)

### Software Dependencies

**Required:**
- Python 3.10 or higher
- pip 23.0+
- Git 2.30+

**Optional (for containerized deployments):**
- Docker 24.0+
- Docker Compose 2.20+
- Kubernetes 1.28+ (for production)
- kubectl 1.28+
- Helm 3.12+ (for Kubernetes deployments)

**Production Infrastructure:**
- PostgreSQL 14+ (database)
- Redis 7.0+ (caching, task queue)
- Nginx 1.24+ or HAProxy 2.8+ (load balancer)
- Prometheus 2.45+ (monitoring)
- Grafana 10.0+ (visualization)

### Credentials and Access

Before deployment, ensure you have:

1. **Cloud Provider Credentials** (if deploying to cloud):
   - AWS: Access Key ID, Secret Access Key, Region
   - Azure: Subscription ID, Tenant ID, Client ID, Client Secret
   - GCP: Service Account JSON key

2. **Container Registry Access** (for Docker deployments):
   - Docker Hub credentials
   - Private registry credentials (if using custom registry)

3. **Database Credentials**:
   - PostgreSQL connection string
   - Redis connection string

4. **API Keys** (for optional features):
   - OpenAI API key (for LLM integration)
   - Anthropic API key (for Claude integration)

5. **SSL/TLS Certificates**:
   - Domain name (for Let's Encrypt)
   - Or custom certificate files (cert.pem, key.pem)

---

## Deployment Options

GreenLang supports multiple deployment architectures based on your needs:

| Deployment Type | Use Case | Complexity | HA Support | Recommended For |
|----------------|----------|------------|------------|-----------------|
| **Local Development** | Development, testing | Low | No | Individual developers |
| **Docker Single Container** | Demos, POCs | Low | No | Quick evaluations |
| **Docker Compose** | Small deployments, staging | Medium | Limited | Small teams, staging environments |
| **Kubernetes** | Production, high availability | High | Yes | Enterprise production |
| **AWS ECS/Fargate** | AWS-native, serverless containers | Medium | Yes | AWS-centric organizations |
| **Azure Container Apps** | Azure-native, serverless | Medium | Yes | Azure-centric organizations |
| **GCP Cloud Run** | GCP-native, serverless | Medium | Yes | GCP-centric organizations |

**Decision Matrix:**

```
Start Here: What's your deployment goal?

├─ Testing/Development?
│  └─> Local Development (Section 4)
│
├─ Quick Demo/POC?
│  └─> Docker Single Container (Section 5)
│
├─ Staging Environment?
│  └─> Docker Compose (Section 6)
│
├─ Production with HA?
│  ├─ Cloud Provider Preference?
│  │  ├─ AWS → ECS/Fargate (Section 8.1)
│  │  ├─ Azure → Container Apps (Section 8.2)
│  │  ├─ GCP → Cloud Run (Section 8.3)
│  │  └─ Multi-cloud/On-prem → Kubernetes (Section 7)
│  └─> Kubernetes (Section 7)
```

---

## Environment Variables Reference

Complete reference for all environment variables supported by GreenLang.

### Core Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GL_VERSION` | No | `0.3.0` | GreenLang version |
| `GL_ENVIRONMENT` | No | `development` | Environment: `development`, `staging`, `production` |
| `GL_LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `GL_LOG_DIR` | No | `/var/log/greenlang` | Log file directory |
| `GL_CACHE_DIR` | No | `/var/cache/greenlang` | Cache directory |
| `GL_HOME` | No | `/var/lib/greenlang` | GreenLang home directory |

### Database Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes* | - | PostgreSQL connection string (format: `postgresql://user:pass@host:port/db`) |
| `DATABASE_HOST` | Yes* | `localhost` | PostgreSQL host |
| `DATABASE_PORT` | Yes* | `5432` | PostgreSQL port |
| `DATABASE_NAME` | Yes* | `greenlang` | Database name |
| `DATABASE_USER` | Yes* | `greenlang` | Database user |
| `DATABASE_PASSWORD` | Yes* | - | Database password (store in secrets) |
| `DATABASE_SSL_MODE` | No | `prefer` | SSL mode: `disable`, `allow`, `prefer`, `require`, `verify-ca`, `verify-full` |
| `DATABASE_POOL_SIZE` | No | `10` | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | No | `20` | Maximum overflow connections |

*Either `DATABASE_URL` OR all individual `DATABASE_*` variables required.

### Redis Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | Yes* | - | Redis connection string (format: `redis://host:port/db`) |
| `REDIS_HOST` | Yes* | `localhost` | Redis host |
| `REDIS_PORT` | Yes* | `6379` | Redis port |
| `REDIS_DB` | No | `0` | Redis database number |
| `REDIS_PASSWORD` | No | - | Redis password (if authentication enabled) |
| `REDIS_SSL` | No | `false` | Enable SSL/TLS for Redis |
| `REDIS_SENTINEL` | No | `false` | Use Redis Sentinel for HA |
| `REDIS_SENTINEL_HOSTS` | No | - | Comma-separated sentinel hosts (if using Sentinel) |
| `REDIS_CACHE_TTL` | No | `3600` | Default cache TTL in seconds |

### Authentication & Security

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET_KEY` | Yes | - | JWT signing key (MUST be secret, 256-bit random) |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm: `HS256`, `RS256`, `ES256` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No | `60` | Access token expiration (minutes) |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | No | `7` | Refresh token expiration (days) |
| `OAUTH2_ENABLED` | No | `false` | Enable OAuth2 authentication |
| `OAUTH2_PROVIDER` | No | - | OAuth2 provider: `google`, `azure`, `okta`, `auth0` |
| `OAUTH2_CLIENT_ID` | No | - | OAuth2 client ID |
| `OAUTH2_CLIENT_SECRET` | No | - | OAuth2 client secret |
| `ENCRYPTION_KEY` | Yes | - | AES-256 encryption key (32 bytes, base64 encoded) |
| `ENABLE_CORS` | No | `true` | Enable CORS for API |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated) |

### API Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_HOST` | No | `0.0.0.0` | API server bind host |
| `API_PORT` | No | `8000` | API server port |
| `API_WORKERS` | No | `4` | Number of API workers (Gunicorn/Uvicorn) |
| `API_WORKER_TIMEOUT` | No | `300` | Worker timeout in seconds |
| `API_MAX_REQUESTS` | No | `10000` | Max requests per worker before restart |
| `API_RATE_LIMIT` | No | `100/minute` | Rate limit (format: `<count>/<period>`) |
| `API_BASE_PATH` | No | `/api/v1` | API base path prefix |

### Monitoring & Observability

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENABLE_METRICS` | No | `true` | Enable Prometheus metrics |
| `METRICS_PORT` | No | `9090` | Metrics endpoint port |
| `ENABLE_TRACING` | No | `false` | Enable distributed tracing |
| `TRACING_BACKEND` | No | `jaeger` | Tracing backend: `jaeger`, `zipkin`, `otlp` |
| `JAEGER_AGENT_HOST` | No | `localhost` | Jaeger agent host |
| `JAEGER_AGENT_PORT` | No | `6831` | Jaeger agent port |
| `SENTRY_DSN` | No | - | Sentry DSN for error tracking |
| `SENTRY_ENVIRONMENT` | No | - | Sentry environment tag |

### LLM Integration (Optional)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No | - | OpenAI API key (for GPT models) |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key (for Claude models) |
| `LLM_PROVIDER` | No | `openai` | Default LLM provider: `openai`, `anthropic`, `azure` |
| `LLM_MODEL` | No | `gpt-4` | Default LLM model |
| `LLM_TEMPERATURE` | No | `0.0` | LLM temperature (0.0 = deterministic) |
| `LLM_MAX_TOKENS` | No | `4096` | Maximum tokens per LLM request |

### Feature Flags

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEATURE_AI_AGENTS` | No | `false` | Enable AI-powered agents (anomaly detection, forecasting) |
| `FEATURE_RAG_SYSTEM` | No | `false` | Enable RAG (Retrieval-Augmented Generation) system |
| `FEATURE_MULTI_TENANCY` | No | `false` | Enable multi-tenancy support |
| `FEATURE_WEBHOOKS` | No | `true` | Enable webhook notifications |
| `FEATURE_AUDIT_LOGGING` | No | `true` | Enable detailed audit logging |

### Example .env File

```bash
# Core Configuration
GL_ENVIRONMENT=production
GL_LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://greenlang:SecurePassword123@postgres.example.com:5432/greenlang_prod
DATABASE_SSL_MODE=require
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://:RedisPassword456@redis.example.com:6379/0
REDIS_SSL=true
REDIS_CACHE_TTL=3600

# Security
JWT_SECRET_KEY=your-256-bit-secret-key-here-change-this-in-production
ENCRYPTION_KEY=your-base64-encoded-32-byte-encryption-key-here
CORS_ORIGINS=https://app.example.com,https://dashboard.example.com

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=8
API_RATE_LIMIT=100/minute

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
SENTRY_ENVIRONMENT=production

# Feature Flags
FEATURE_AI_AGENTS=true
FEATURE_AUDIT_LOGGING=true
```

---

## Local Development Deployment

For development, testing, and debugging on your local machine.

### Step 1: Clone Repository

```bash
git clone https://github.com/greenlang/greenlang.git
cd greenlang
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

**Core Platform:**
```bash
pip install --upgrade pip
pip install -e .
```

**Full Platform (recommended for development):**
```bash
pip install -e ".[dev]"
```

**All Features:**
```bash
pip install -e ".[all]"
```

### Step 4: Configure Environment

Create `.env` file in project root:

```bash
# .env for local development
GL_ENVIRONMENT=development
GL_LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/greenlang_dev
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=dev-secret-key-not-for-production
ENCRYPTION_KEY=dev-encryption-key-base64-encoded
```

### Step 5: Start Local Services

**Option A: Use Docker Compose for services only**

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: greenlang_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

Start services:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

**Option B: Install services locally**

**PostgreSQL (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb greenlang_dev
```

**Redis (Ubuntu/Debian):**
```bash
sudo apt install redis-server
sudo systemctl start redis-server
```

### Step 6: Initialize Database

```bash
# Run database migrations
gl db migrate

# Verify database connection
gl db check
```

### Step 7: Run Application

**CLI Usage:**
```bash
gl --version
gl --help
```

**Start API Server (if using full platform):**
```bash
# Development server with auto-reload
uvicorn greenlang.api.main:app --reload --host 0.0.0.0 --port 8000

# Or using the CLI
gl server start --reload
```

**Run Tests:**
```bash
pytest
```

**Check Code Quality:**
```bash
ruff check .
mypy greenlang/
black --check .
```

### Step 8: Access Application

- API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- Metrics (Prometheus): http://localhost:9090/metrics

### Development Workflow

1. **Make code changes** in your editor
2. **Run tests**: `pytest tests/` or `pytest -v tests/test_specific.py`
3. **Check code quality**: `ruff check .` and `mypy greenlang/`
4. **Format code**: `black greenlang/ tests/`
5. **Commit changes**: `git add . && git commit -m "Your message"`

---

## Docker Single Container

Quick deployment using a single Docker container (suitable for demos and testing).

### Option 1: Pull from Docker Hub

**Core Platform (CLI only):**
```bash
docker pull greenlang/greenlang:0.3.0

# Run CLI
docker run -it greenlang/greenlang:0.3.0 gl --version
```

**Full Platform (with API server):**
```bash
docker pull greenlang/greenlang:0.3.0-full

# Run API server
docker run -d \
  --name greenlang-api \
  -p 8000:8000 \
  -p 9090:9090 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e REDIS_URL=redis://host:6379/0 \
  greenlang/greenlang:0.3.0-full
```

### Option 2: Build from Source

**Build Core Image:**
```bash
docker build -t greenlang:local -f Dockerfile .
```

**Build Full Platform Image:**
```bash
docker build -t greenlang:local-full -f Dockerfile.full .
```

### Run Container with Volumes

```bash
docker run -d \
  --name greenlang \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/data:/workspace/data \
  -v greenlang-cache:/var/cache/greenlang \
  -v greenlang-logs:/var/log/greenlang \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e REDIS_URL=redis://host:6379/0 \
  -e JWT_SECRET_KEY=your-secret-key \
  --restart unless-stopped \
  greenlang:local-full
```

### Container Health Check

```bash
# Check container status
docker ps

# Check logs
docker logs greenlang

# Check health endpoint
curl http://localhost:8000/health

# Execute commands inside container
docker exec -it greenlang gl --version
```

### Stop and Remove Container

```bash
docker stop greenlang
docker rm greenlang
```

---

## Docker Compose Multi-Service

Production-like environment with multiple services (API, database, cache, monitoring).

### Complete docker-compose.yml

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:16-alpine
    container_name: greenlang-postgres
    environment:
      POSTGRES_DB: greenlang
      POSTGRES_USER: greenlang
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U greenlang"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - greenlang
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: greenlang-redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - greenlang
    restart: unless-stopped

  # GreenLang API Server
  api:
    build:
      context: .
      dockerfile: Dockerfile.full
    image: greenlang/greenlang:0.3.0-full
    container_name: greenlang-api
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      GL_ENVIRONMENT: production
      GL_LOG_LEVEL: INFO
      DATABASE_URL: postgresql://greenlang:${DATABASE_PASSWORD}@postgres:5432/greenlang
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
      ENCRYPTION_KEY: ${ENCRYPTION_KEY}
      API_WORKERS: 4
      ENABLE_METRICS: "true"
    volumes:
      - greenlang_cache:/var/cache/greenlang
      - greenlang_logs:/var/log/greenlang
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - greenlang
    restart: unless-stopped

  # Nginx Reverse Proxy
  nginx:
    image: nginx:1.25-alpine
    container_name: greenlang-nginx
    depends_on:
      - api
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    networks:
      - greenlang
    restart: unless-stopped

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: greenlang-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - greenlang
    restart: unless-stopped

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:10.2.2
    container_name: greenlang-grafana
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
      - grafana_data:/var/lib/grafana
    networks:
      - greenlang
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  greenlang_cache:
  greenlang_logs:
  nginx_logs:
  prometheus_data:
  grafana_data:

networks:
  greenlang:
    driver: bridge
```

### .env File for Docker Compose

Create `.env` file in the same directory as `docker-compose.yml`:

```bash
# Database
DATABASE_PASSWORD=ChangeThisToSecurePassword123

# Redis
REDIS_PASSWORD=ChangeThisToSecureRedisPassword456

# Security
JWT_SECRET_KEY=your-256-bit-secret-key-change-in-production
ENCRYPTION_KEY=your-base64-encoded-32-byte-key

# Monitoring
GRAFANA_PASSWORD=admin123
```

### Deploy Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps

# View specific service logs
docker-compose logs -f api

# Scale API workers
docker-compose up -d --scale api=3

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Nginx**: http://localhost:80
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Metrics**: http://localhost:9090/metrics

### Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Check Prometheus metrics
curl http://localhost:9090/metrics

# Run test calculation
curl -X POST http://localhost:8000/api/v1/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "activity_type": "fuel_combustion",
    "fuel_type": "natural_gas",
    "quantity": 1000,
    "unit": "kWh"
  }'
```

---

## Kubernetes Production Deployment

Enterprise-grade production deployment with high availability, autoscaling, and fault tolerance.

### Prerequisites

1. **Kubernetes cluster** (v1.28+):
   - Managed: AWS EKS, Azure AKS, GCP GKE
   - Self-managed: kubeadm, kops, Rancher

2. **kubectl** configured to access your cluster
3. **Helm** 3.12+ installed
4. **Ingress controller** (nginx-ingress, traefik, or cloud-native)
5. **Cert-manager** (for automatic SSL/TLS certificates)
6. **Container registry** (Docker Hub, AWS ECR, GCR, Azure ACR)

### Step 1: Create Namespace

```bash
kubectl create namespace greenlang
kubectl label namespace greenlang name=greenlang
kubectl label namespace greenlang environment=production
```

**Namespace with Resource Quotas:**

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: greenlang
  labels:
    name: greenlang
    environment: production
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: greenlang-quota
  namespace: greenlang
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"
    limits.cpu: "200"
    limits.memory: "400Gi"
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
```

Apply:
```bash
kubectl apply -f namespace.yaml
```

### Step 2: Create Secrets

**Database Secret:**

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: greenlang-secrets
  namespace: greenlang
type: Opaque
stringData:
  database-url: "postgresql://greenlang:SecurePassword@postgres.example.com:5432/greenlang_prod"
  redis-url: "redis://:RedisPassword@redis.example.com:6379/0"
  jwt-secret-key: "your-256-bit-secret-key-change-this"
  encryption-key: "your-base64-encoded-32-byte-encryption-key"
  openai-api-key: "sk-your-openai-api-key"
  sentry-dsn: "https://your-sentry-dsn@sentry.io/project"
```

**Apply secrets:**
```bash
kubectl apply -f secrets.yaml
```

**Using Sealed Secrets (Recommended for GitOps):**

```bash
# Install sealed-secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Create sealed secret
kubeseal --format=yaml < secrets.yaml > sealed-secrets.yaml

# Apply sealed secret
kubectl apply -f sealed-secrets.yaml
```

### Step 3: Create ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: greenlang-config
  namespace: greenlang
data:
  GL_ENVIRONMENT: "production"
  GL_LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "8"
  API_RATE_LIMIT: "100/minute"
  ENABLE_METRICS: "true"
  METRICS_PORT: "9090"
  CORS_ORIGINS: "https://app.example.com"
  FEATURE_AI_AGENTS: "true"
  FEATURE_AUDIT_LOGGING: "true"
```

Apply:
```bash
kubectl apply -f configmap.yaml
```

### Step 4: Deploy PostgreSQL (StatefulSet)

```yaml
# postgres-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: greenlang
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: greenlang
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:16-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: greenlang_prod
        - name: POSTGRES_USER
          value: greenlang
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: greenlang-secrets
              key: database-password
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

Apply:
```bash
kubectl apply -f postgres-statefulset.yaml
```

### Step 5: Deploy Redis (StatefulSet with Sentinel for HA)

```yaml
# redis-sentinel.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-sentinel-config
  namespace: greenlang
data:
  redis.conf: |
    bind 0.0.0.0
    port 6379
    dir /data
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
  sentinel.conf: |
    bind 0.0.0.0
    port 26379
    sentinel monitor mymaster redis-0.redis.greenlang.svc.cluster.local 6379 2
    sentinel down-after-milliseconds mymaster 5000
    sentinel parallel-syncs mymaster 1
    sentinel failover-timeout mymaster 10000
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: greenlang
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    name: redis
  - port: 26379
    name: sentinel
  clusterIP: None
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: greenlang
spec:
  serviceName: redis
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "/etc/redis/redis.conf"]
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: config
          mountPath: /etc/redis
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      - name: sentinel
        image: redis:7-alpine
        command: ["redis-sentinel", "/etc/redis/sentinel.conf"]
        ports:
        - containerPort: 26379
        volumeMounts:
        - name: config
          mountPath: /etc/redis
      volumes:
      - name: config
        configMap:
          name: redis-sentinel-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 20Gi
```

Apply:
```bash
kubectl apply -f redis-sentinel.yaml
```

### Step 6: Deploy GreenLang API (Deployment with HPA)

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-api
  namespace: greenlang
  labels:
    app: greenlang-api
    version: v0.3.0
spec:
  replicas: 3
  revisionHistoryLimit: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: greenlang-api
  template:
    metadata:
      labels:
        app: greenlang-api
        version: v0.3.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: greenlang-api-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: greenlang-api
            topologyKey: topology.kubernetes.io/zone
      containers:
      - name: api
        image: greenlang/greenlang:0.3.0-full
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
        - name: metrics
          containerPort: 9090
        envFrom:
        - configMapRef:
            name: greenlang-config
        - secretRef:
            name: greenlang-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: cache
          mountPath: /var/cache/greenlang
        - name: logs
          mountPath: /var/log/greenlang
      volumes:
      - name: cache
        emptyDir:
          sizeLimit: 2Gi
      - name: logs
        emptyDir:
          sizeLimit: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  name: greenlang-api
  namespace: greenlang
spec:
  selector:
    app: greenlang-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: greenlang-api-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: greenlang-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

Apply:
```bash
kubectl apply -f api-deployment.yaml
```

### Step 7: Create Ingress with SSL/TLS

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: greenlang-ingress
  namespace: greenlang
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/limit-rps: "10"
spec:
  tls:
  - hosts:
    - api.greenlang.example.com
    secretName: greenlang-tls
  rules:
  - host: api.greenlang.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: greenlang-api
            port:
              number: 80
```

Apply:
```bash
kubectl apply -f ingress.yaml
```

### Step 8: Verify Deployment

```bash
# Check all resources
kubectl get all -n greenlang

# Check pod status
kubectl get pods -n greenlang

# View logs
kubectl logs -f deployment/greenlang-api -n greenlang

# Check HPA status
kubectl get hpa -n greenlang

# Test API endpoint
curl https://api.greenlang.example.com/health
```

### Step 9: Monitoring Setup

**Deploy Prometheus ServiceMonitor:**

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: greenlang-api
  namespace: greenlang
spec:
  selector:
    matchLabels:
      app: greenlang-api
  endpoints:
  - port: metrics
    interval: 30s
```

Apply:
```bash
kubectl apply -f servicemonitor.yaml
```

### Complete Deployment Script

```bash
#!/bin/bash
# deploy-k8s.sh

set -e

NAMESPACE="greenlang"
CHART_VERSION="0.3.0"

echo "Deploying GreenLang to Kubernetes..."

# Create namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply secrets (assuming you have sealed-secrets)
kubectl apply -f sealed-secrets.yaml

# Apply configmap
kubectl apply -f configmap.yaml

# Deploy PostgreSQL
kubectl apply -f postgres-statefulset.yaml

# Deploy Redis Sentinel
kubectl apply -f redis-sentinel.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

# Deploy API
kubectl apply -f api-deployment.yaml

# Wait for API to be ready
kubectl wait --for=condition=available deployment/greenlang-api -n $NAMESPACE --timeout=300s

# Deploy Ingress
kubectl apply -f ingress.yaml

# Deploy monitoring
kubectl apply -f servicemonitor.yaml

echo "Deployment complete!"
echo "Check status: kubectl get all -n $NAMESPACE"
echo "API endpoint: https://api.greenlang.example.com"
```

Make executable and run:
```bash
chmod +x deploy-k8s.sh
./deploy-k8s.sh
```

---

## Cloud Platform Deployments

### 8.1 AWS ECS/Fargate

**Prerequisites:**
- AWS CLI configured
- ECR repository created
- VPC with public/private subnets
- RDS PostgreSQL instance
- ElastiCache Redis cluster

**Step 1: Push Image to ECR**

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Build and tag image
docker build -t greenlang:0.3.0 -f Dockerfile.full .
docker tag greenlang:0.3.0 123456789012.dkr.ecr.us-east-1.amazonaws.com/greenlang:0.3.0

# Push to ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/greenlang:0.3.0
```

**Step 2: Create ECS Task Definition**

```json
{
  "family": "greenlang-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/greenlangTaskRole",
  "containerDefinitions": [
    {
      "name": "greenlang-api",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/greenlang:0.3.0",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        },
        {
          "containerPort": 9090,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GL_ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "GL_LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:greenlang/database-url"
        },
        {
          "name": "REDIS_URL",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:greenlang/redis-url"
        },
        {
          "name": "JWT_SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:greenlang/jwt-secret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/greenlang-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

Save as `task-definition.json` and register:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

**Step 3: Create ECS Service**

```bash
aws ecs create-service \
  --cluster greenlang-cluster \
  --service-name greenlang-api \
  --task-definition greenlang-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/greenlang-api/abc123,containerName=greenlang-api,containerPort=8000" \
  --health-check-grace-period-seconds 60
```

**Step 4: Configure Auto Scaling**

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/greenlang-cluster/greenlang-api \
  --min-capacity 3 \
  --max-capacity 20

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/greenlang-cluster/greenlang-api \
  --policy-name greenlang-cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

**scaling-policy.json:**
```json
{
  "TargetValue": 70.0,
  "PredefinedMetricSpecification": {
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  },
  "ScaleInCooldown": 300,
  "ScaleOutCooldown": 60
}
```

### 8.2 Azure Container Apps

**Prerequisites:**
- Azure CLI installed
- Resource group created
- Azure Container Registry (ACR)
- Azure Database for PostgreSQL
- Azure Cache for Redis

**Step 1: Push Image to ACR**

```bash
# Login to ACR
az acr login --name greenlangregistry

# Build and push
docker build -t greenlang:0.3.0 -f Dockerfile.full .
docker tag greenlang:0.3.0 greenlangregistry.azurecr.io/greenlang:0.3.0
docker push greenlangregistry.azurecr.io/greenlang:0.3.0
```

**Step 2: Create Container App Environment**

```bash
az containerapp env create \
  --name greenlang-env \
  --resource-group greenlang-rg \
  --location eastus \
  --logs-workspace-id <workspace-id> \
  --logs-workspace-key <workspace-key>
```

**Step 3: Deploy Container App**

```bash
az containerapp create \
  --name greenlang-api \
  --resource-group greenlang-rg \
  --environment greenlang-env \
  --image greenlangregistry.azurecr.io/greenlang:0.3.0 \
  --target-port 8000 \
  --ingress external \
  --cpu 2 \
  --memory 4Gi \
  --min-replicas 3 \
  --max-replicas 20 \
  --scale-rule-name cpu-scaling \
  --scale-rule-type cpu \
  --scale-rule-metadata concurrentRequests=100 \
  --env-vars \
    GL_ENVIRONMENT=production \
    GL_LOG_LEVEL=INFO \
  --secrets \
    database-url=secretref:database-url \
    redis-url=secretref:redis-url \
    jwt-secret=secretref:jwt-secret
```

### 8.3 GCP Cloud Run

**Prerequisites:**
- gcloud CLI configured
- GCP project created
- Artifact Registry repository
- Cloud SQL PostgreSQL instance
- Memorystore Redis instance

**Step 1: Push Image to Artifact Registry**

```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and push
docker build -t greenlang:0.3.0 -f Dockerfile.full .
docker tag greenlang:0.3.0 us-central1-docker.pkg.dev/greenlang-project/images/greenlang:0.3.0
docker push us-central1-docker.pkg.dev/greenlang-project/images/greenlang:0.3.0
```

**Step 2: Deploy to Cloud Run**

```bash
gcloud run deploy greenlang-api \
  --image us-central1-docker.pkg.dev/greenlang-project/images/greenlang:0.3.0 \
  --platform managed \
  --region us-central1 \
  --port 8000 \
  --cpu 2 \
  --memory 4Gi \
  --min-instances 3 \
  --max-instances 20 \
  --concurrency 80 \
  --timeout 300 \
  --set-env-vars GL_ENVIRONMENT=production,GL_LOG_LEVEL=INFO \
  --set-secrets DATABASE_URL=database-url:latest,REDIS_URL=redis-url:latest,JWT_SECRET_KEY=jwt-secret:latest \
  --allow-unauthenticated \
  --vpc-connector greenlang-vpc-connector
```

---

## Database Setup

### PostgreSQL Initialization

**Step 1: Create Database**

```bash
# Connect to PostgreSQL
psql -U postgres -h localhost

# Create database and user
CREATE DATABASE greenlang_prod;
CREATE USER greenlang WITH PASSWORD 'SecurePassword123';
GRANT ALL PRIVILEGES ON DATABASE greenlang_prod TO greenlang;

# Enable required extensions
\c greenlang_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
```

**Step 2: Run Migrations**

Using GreenLang CLI:
```bash
# Set database connection
export DATABASE_URL="postgresql://greenlang:SecurePassword123@localhost:5432/greenlang_prod"

# Run migrations
gl db migrate

# Verify migration status
gl db status
```

Manual migration script:
```bash
# Download migration scripts
wget https://github.com/greenlang/greenlang/releases/download/v0.3.0/migrations.tar.gz
tar -xzf migrations.tar.gz

# Run migrations using Alembic
alembic upgrade head
```

### Database Schema

GreenLang creates the following tables:

- `emission_factors` - Emission factor library
- `calculations` - Calculation history
- `agents` - Agent registry
- `workflows` - Workflow definitions
- `jobs` - Asynchronous job tracking
- `users` - User accounts (if auth enabled)
- `audit_logs` - Audit trail
- `cache_entries` - Application-level cache

### Database Backup

**Automated Backup (PostgreSQL)**

```bash
#!/bin/bash
# backup-db.sh

BACKUP_DIR="/var/backups/greenlang"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/greenlang_$TIMESTAMP.sql.gz"

mkdir -p $BACKUP_DIR

# Backup database
pg_dump -h localhost -U greenlang greenlang_prod | gzip > $BACKUP_FILE

# Retain last 7 days
find $BACKUP_DIR -name "greenlang_*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
```

Schedule with cron:
```bash
# Run daily at 2 AM
0 2 * * * /usr/local/bin/backup-db.sh
```

**Point-in-Time Recovery (AWS RDS)**

```bash
# Restore to specific time
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier greenlang-prod \
  --target-db-instance-identifier greenlang-restored \
  --restore-time 2025-11-21T10:00:00Z
```

---

## Secrets Management

### HashiCorp Vault

**Step 1: Install Vault**

```bash
# Install Vault
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# Initialize Vault
vault server -dev -dev-root-token-id="root"
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN="root"
```

**Step 2: Store Secrets**

```bash
# Enable KV secrets engine
vault secrets enable -path=greenlang kv-v2

# Store database credentials
vault kv put greenlang/database \
  url="postgresql://greenlang:SecurePassword@postgres:5432/greenlang" \
  password="SecurePassword"

# Store JWT secret
vault kv put greenlang/auth \
  jwt_secret="your-256-bit-secret-key" \
  encryption_key="your-base64-encoded-key"

# Store API keys
vault kv put greenlang/api-keys \
  openai="sk-your-openai-key" \
  anthropic="your-anthropic-key"
```

**Step 3: Retrieve Secrets in Application**

```python
# greenlang/config/vault.py
import hvac

client = hvac.Client(url='http://vault:8200', token='root')

# Read database secret
secret = client.secrets.kv.v2.read_secret_version(path='database', mount_point='greenlang')
database_url = secret['data']['data']['url']
```

**Kubernetes Integration (Vault Agent Injector)**

```yaml
# deployment-with-vault.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-api
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "greenlang"
        vault.hashicorp.com/agent-inject-secret-database: "greenlang/data/database"
        vault.hashicorp.com/agent-inject-template-database: |
          {{- with secret "greenlang/data/database" -}}
          export DATABASE_URL="{{ .Data.data.url }}"
          {{- end }}
    spec:
      serviceAccountName: greenlang-vault
      containers:
      - name: api
        image: greenlang/greenlang:0.3.0-full
        command: ["/bin/sh", "-c"]
        args:
        - source /vault/secrets/database && /app/entrypoint.sh
```

### AWS Secrets Manager

**Store Secrets:**

```bash
# Create secret
aws secretsmanager create-secret \
  --name greenlang/database \
  --description "GreenLang database credentials" \
  --secret-string '{"url":"postgresql://user:pass@host:5432/db","password":"SecurePassword"}'

# Update secret
aws secretsmanager update-secret \
  --secret-id greenlang/database \
  --secret-string '{"url":"postgresql://user:newpass@host:5432/db"}'
```

**Retrieve in Application:**

```python
import boto3
import json

client = boto3.client('secretsmanager', region_name='us-east-1')

def get_secret(secret_name):
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
db_secret = get_secret('greenlang/database')
database_url = db_secret['url']
```

### Azure Key Vault

**Store Secrets:**

```bash
# Create Key Vault
az keyvault create \
  --name greenlang-vault \
  --resource-group greenlang-rg \
  --location eastus

# Add secrets
az keyvault secret set \
  --vault-name greenlang-vault \
  --name database-url \
  --value "postgresql://user:pass@host:5432/db"

az keyvault secret set \
  --vault-name greenlang-vault \
  --name jwt-secret \
  --value "your-secret-key"
```

**Retrieve in Application:**

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://greenlang-vault.vault.azure.net/", credential=credential)

# Retrieve secret
database_url = client.get_secret("database-url").value
```

---

## Monitoring and Observability

### Prometheus Configuration

**prometheus.yml:**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'greenlang-prod'
    environment: 'production'

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

rule_files:
  - "/etc/prometheus/alerts/*.yml"

scrape_configs:
  # GreenLang API
  - job_name: 'greenlang-api'
    static_configs:
    - targets: ['api:9090']
    metrics_path: '/metrics'

  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
    - targets: ['postgres-exporter:9187']

  # Redis
  - job_name: 'redis'
    static_configs:
    - targets: ['redis-exporter:9121']

  # Node Exporter
  - job_name: 'node'
    static_configs:
    - targets: ['node-exporter:9100']

  # Kubernetes (if deployed on k8s)
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__
```

### Alert Rules

**alerts/greenlang-alerts.yml:**

```yaml
groups:
- name: greenlang_alerts
  interval: 30s
  rules:

  # API Health
  - alert: APIDown
    expr: up{job="greenlang-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "GreenLang API is down"
      description: "API instance {{ $labels.instance }} has been down for more than 1 minute."

  # High Error Rate
  - alert: HighErrorRate
    expr: rate(http_requests_total{job="greenlang-api",status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors/sec on {{ $labels.instance }}."

  # High Response Time
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="greenlang-api"}[5m])) > 1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High API response time"
      description: "95th percentile response time is {{ $value }}s on {{ $labels.instance }}."

  # Database Connection Pool Exhaustion
  - alert: DatabasePoolExhausted
    expr: database_pool_size - database_pool_available < 2
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Database connection pool nearly exhausted"
      description: "Only {{ $value }} connections available in pool."

  # High Memory Usage
  - alert: HighMemoryUsage
    expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}."

  # High CPU Usage
  - alert: HighCPUUsage
    expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}."
```

### Grafana Dashboards

**GreenLang API Dashboard:**

```json
{
  "dashboard": {
    "title": "GreenLang API Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"greenlang-api\"}[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"greenlang-api\",status=~\"5..\"}[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"greenlang-api\"}[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Active Connections",
        "targets": [
          {
            "expr": "database_pool_size - database_pool_available"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

Import dashboard:
```bash
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @greenlang-dashboard.json
```

---

## Load Balancer Configuration

### Nginx Configuration

**nginx.conf:**

```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 50M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript
               application/json application/javascript application/xml+rss;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_status 429;

    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=addr:10m;

    # Upstream backend
    upstream greenlang_backend {
        least_conn;
        server api-1:8000 max_fails=3 fail_timeout=30s;
        server api-2:8000 max_fails=3 fail_timeout=30s;
        server api-3:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # HTTP -> HTTPS redirect
    server {
        listen 80;
        server_name api.greenlang.example.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name api.greenlang.example.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        ssl_stapling on;
        ssl_stapling_verify on;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            limit_conn addr 10;

            proxy_pass http://greenlang_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 300s;

            # Buffering
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
            proxy_busy_buffers_size 8k;
        }

        # Health check endpoint (no rate limit)
        location /health {
            proxy_pass http://greenlang_backend;
            access_log off;
        }

        # Metrics endpoint (internal only)
        location /metrics {
            allow 10.0.0.0/8;
            deny all;
            proxy_pass http://greenlang_backend;
        }
    }
}
```

Deploy Nginx:
```bash
# Using Docker
docker run -d \
  --name nginx \
  -p 80:80 \
  -p 443:443 \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v $(pwd)/ssl:/etc/nginx/ssl:ro \
  --restart unless-stopped \
  nginx:1.25-alpine
```

### HAProxy Configuration

**haproxy.cfg:**

```haproxy
global
    log stdout local0
    maxconn 4096
    daemon

defaults
    log global
    mode http
    option httplog
    option dontlognull
    option http-server-close
    option forwardfor except 127.0.0.0/8
    option redispatch
    retries 3
    timeout connect 5s
    timeout client 50s
    timeout server 50s

frontend http_front
    bind *:80
    redirect scheme https code 301 if !{ ssl_fc }

frontend https_front
    bind *:443 ssl crt /etc/haproxy/ssl/cert.pem alpn h2,http/1.1

    # ACLs
    acl is_api path_beg /api/
    acl is_health path /health
    acl is_metrics path /metrics

    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny deny_status 429 if { sc_http_req_rate(0) gt 100 }

    # Routing
    use_backend greenlang_api if is_api
    use_backend greenlang_health if is_health
    use_backend greenlang_metrics if is_metrics

backend greenlang_api
    balance leastconn
    option httpchk GET /health
    http-check expect status 200
    server api-1 api-1:8000 check inter 5s rise 2 fall 3 maxconn 100
    server api-2 api-2:8000 check inter 5s rise 2 fall 3 maxconn 100
    server api-3 api-3:8000 check inter 5s rise 2 fall 3 maxconn 100

backend greenlang_health
    option httpchk GET /health
    server api-1 api-1:8000 check

backend greenlang_metrics
    acl internal_network src 10.0.0.0/8
    http-request deny unless internal_network
    server api-1 api-1:9090
```

---

## SSL/TLS Certificates

### Let's Encrypt (Automatic)

**Using Certbot:**

```bash
# Install Certbot
sudo apt update
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.greenlang.example.com

# Auto-renewal (already configured by certbot)
sudo certbot renew --dry-run
```

**Using cert-manager (Kubernetes):**

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@greenlang.example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

Ingress will automatically request certificate (see Kubernetes section).

### Custom Certificates

**Generate self-signed certificate (development only):**

```bash
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=api.greenlang.example.com"
```

**Use commercial certificate:**

```bash
# Combine certificate chain
cat domain.crt intermediate.crt root.crt > cert.pem

# Use private key as-is
cp domain.key key.pem

# Store in Kubernetes secret
kubectl create secret tls greenlang-tls \
  --cert=cert.pem \
  --key=key.pem \
  -n greenlang
```

---

## Scaling and Autoscaling

### Horizontal Pod Autoscaler (HPA)

See Kubernetes section (Step 6) for HPA configuration.

**Monitor HPA:**

```bash
# Watch HPA status
kubectl get hpa -n greenlang -w

# Describe HPA
kubectl describe hpa greenlang-api-hpa -n greenlang

# Get current metrics
kubectl top pods -n greenlang
```

### Vertical Pod Autoscaler (VPA)

**Install VPA:**

```bash
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh
```

**Create VPA:**

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: greenlang-api-vpa
  namespace: greenlang
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: greenlang-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      minAllowed:
        cpu: 500m
        memory: 1Gi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
```

### Manual Scaling

```bash
# Kubernetes
kubectl scale deployment greenlang-api --replicas=10 -n greenlang

# Docker Compose
docker-compose up -d --scale api=5

# AWS ECS
aws ecs update-service \
  --cluster greenlang-cluster \
  --service greenlang-api \
  --desired-count 10
```

---

## Backup and Recovery

### Database Backup Strategy

**Full Backup (Daily):**

```bash
#!/bin/bash
# /usr/local/bin/backup-full.sh

BACKUP_DIR="/var/backups/greenlang/full"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

mkdir -p $BACKUP_DIR

# Full database dump
pg_dump -h localhost -U greenlang greenlang_prod \
  --format=custom \
  --file="$BACKUP_DIR/greenlang_full_$TIMESTAMP.dump"

# Compress
gzip "$BACKUP_DIR/greenlang_full_$TIMESTAMP.dump"

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR/greenlang_full_$TIMESTAMP.dump.gz" \
  s3://greenlang-backups/database/full/

# Cleanup old backups
find $BACKUP_DIR -name "greenlang_full_*.dump.gz" -mtime +$RETENTION_DAYS -delete

echo "Full backup completed: $TIMESTAMP"
```

**Incremental Backup (Hourly):**

```bash
#!/bin/bash
# /usr/local/bin/backup-incremental.sh

BACKUP_DIR="/var/backups/greenlang/wal"
RETENTION_HOURS=72

mkdir -p $BACKUP_DIR

# Archive WAL files
pg_receivewal -h localhost -U greenlang -D $BACKUP_DIR

# Cleanup old WAL files
find $BACKUP_DIR -name "*.wal" -mmin +$((RETENTION_HOURS * 60)) -delete
```

**Schedule Backups:**

```bash
# /etc/cron.d/greenlang-backup

# Full backup daily at 2 AM
0 2 * * * /usr/local/bin/backup-full.sh

# Incremental backup every hour
0 * * * * /usr/local/bin/backup-incremental.sh
```

### Restore Procedures

**Restore from Full Backup:**

```bash
# Stop application
kubectl scale deployment greenlang-api --replicas=0 -n greenlang

# Download backup from S3
aws s3 cp s3://greenlang-backups/database/full/greenlang_full_20251121_020000.dump.gz .

# Decompress
gunzip greenlang_full_20251121_020000.dump.gz

# Drop existing database (WARNING: destructive)
psql -U postgres -c "DROP DATABASE greenlang_prod;"
psql -U postgres -c "CREATE DATABASE greenlang_prod OWNER greenlang;"

# Restore
pg_restore -h localhost -U greenlang -d greenlang_prod \
  greenlang_full_20251121_020000.dump

# Restart application
kubectl scale deployment greenlang-api --replicas=3 -n greenlang
```

**Point-in-Time Recovery (PITR):**

```bash
# Restore base backup
pg_restore -d greenlang_prod greenlang_full_20251121_020000.dump

# Create recovery.conf
cat > /var/lib/postgresql/data/recovery.conf <<EOF
restore_command = 'cp /var/backups/greenlang/wal/%f %p'
recovery_target_time = '2025-11-21 10:00:00 UTC'
recovery_target_action = 'promote'
EOF

# Start PostgreSQL (will replay WAL logs up to target time)
systemctl start postgresql
```

### Disaster Recovery Plan

**RTO (Recovery Time Objective):** 1 hour
**RPO (Recovery Point Objective):** 5 minutes

**Recovery Steps:**

1. **Assess Impact:** Determine scope of failure (database, application, infrastructure)
2. **Notify Stakeholders:** Alert team via PagerDuty/Slack
3. **Activate Backup Site:** Switch to DR region/cluster (if multi-region)
4. **Restore Data:** Restore database from latest backup
5. **Redeploy Application:** Deploy to backup infrastructure
6. **Validate:** Run smoke tests, verify data integrity
7. **Update DNS:** Point domain to backup infrastructure
8. **Monitor:** Watch for anomalies, errors
9. **Post-Mortem:** Document incident, improve procedures

**DR Checklist:**

- [ ] Database backups tested monthly
- [ ] Application deployment tested in DR environment
- [ ] DNS failover procedure documented
- [ ] RTO/RPO targets measured quarterly
- [ ] Team trained on DR procedures

---

## Rollback Procedures

### Kubernetes Rollback

**Automatic Rollback (if deployment fails):**

Kubernetes automatically rolls back if:
- Readiness probe fails for new pods
- `progressDeadlineSeconds` exceeded (default: 600s)

**Manual Rollback:**

```bash
# View rollout history
kubectl rollout history deployment/greenlang-api -n greenlang

# Rollback to previous version
kubectl rollout undo deployment/greenlang-api -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/greenlang-api --to-revision=3 -n greenlang

# Check rollout status
kubectl rollout status deployment/greenlang-api -n greenlang
```

### Docker Compose Rollback

```bash
# Stop current version
docker-compose down

# Checkout previous version
git checkout v0.2.0

# Rebuild and start
docker-compose up -d --build
```

### Database Migration Rollback

```bash
# Rollback last migration
gl db downgrade -1

# Rollback to specific version
gl db downgrade abc123

# Or using Alembic directly
alembic downgrade -1
```

---

## Health Checks

### Application Health Endpoints

GreenLang exposes three health check endpoints:

**1. Startup Probe (`/startup`):**

Used to determine if application has finished starting.

```bash
curl http://localhost:8000/startup
```

Response:
```json
{
  "status": "ok",
  "timestamp": "2025-11-21T10:00:00Z"
}
```

**2. Liveness Probe (`/health` or `/healthz`):**

Used to determine if application is running correctly.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.3.0",
  "uptime": 3600,
  "timestamp": "2025-11-21T10:00:00Z"
}
```

**3. Readiness Probe (`/ready`):**

Used to determine if application can accept traffic.

```bash
curl http://localhost:8000/ready
```

Response (healthy):
```json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "redis": "ok",
    "disk_space": "ok"
  },
  "timestamp": "2025-11-21T10:00:00Z"
}
```

Response (unhealthy):
```json
{
  "status": "not_ready",
  "checks": {
    "database": "ok",
    "redis": "failed",
    "disk_space": "ok"
  },
  "timestamp": "2025-11-21T10:00:00Z"
}
```

### Custom Health Checks

Add custom health checks by implementing the health check interface:

```python
# greenlang/health/custom.py
from greenlang.health import HealthCheck

class CustomHealthCheck(HealthCheck):
    def check(self) -> bool:
        # Your custom logic
        return True

    def name(self) -> str:
        return "custom_check"
```

Register in configuration:

```python
# config.py
HEALTH_CHECKS = [
    "greenlang.health.database.DatabaseHealthCheck",
    "greenlang.health.redis.RedisHealthCheck",
    "greenlang.health.custom.CustomHealthCheck",
]
```

---

## Security Hardening

### Production Security Checklist

**Application Security:**

- [ ] Change all default passwords
- [ ] Generate strong JWT secret (256-bit random)
- [ ] Generate strong encryption key (32 bytes, base64)
- [ ] Enable HTTPS/TLS everywhere
- [ ] Use TLS 1.2+ only (disable TLS 1.0, 1.1)
- [ ] Enable CORS with specific origins (not `*`)
- [ ] Set secure session cookies (`HttpOnly`, `Secure`, `SameSite`)
- [ ] Enable rate limiting (100 req/min recommended)
- [ ] Enable request size limits (50MB max recommended)
- [ ] Validate all inputs (use Pydantic models)
- [ ] Sanitize all outputs (prevent XSS)
- [ ] Use parameterized queries (prevent SQL injection)
- [ ] Disable debug mode (`GL_ENVIRONMENT=production`)
- [ ] Enable audit logging
- [ ] Configure Sentry for error tracking

**Infrastructure Security:**

- [ ] Run containers as non-root user (UID 10001)
- [ ] Use read-only root filesystem
- [ ] Drop all capabilities (`cap_drop: ALL`)
- [ ] Enable `no-new-privileges` security option
- [ ] Use network policies (Kubernetes)
- [ ] Isolate database in private subnet
- [ ] Use VPC/VNet for cloud deployments
- [ ] Configure firewall rules (allow only necessary ports)
- [ ] Enable DDoS protection (CloudFlare, AWS Shield)
- [ ] Use WAF (Web Application Firewall)
- [ ] Enable container scanning (Trivy, Snyk)
- [ ] Enable vulnerability scanning (daily)
- [ ] Rotate secrets regularly (90 days)
- [ ] Use managed secrets (Vault, AWS Secrets Manager)

**Database Security:**

- [ ] Use strong database password (16+ chars, random)
- [ ] Enable SSL/TLS for database connections
- [ ] Use `verify-full` SSL mode (verify certificate)
- [ ] Restrict database access by IP (firewall)
- [ ] Enable database audit logging
- [ ] Encrypt database at rest (LUKS, AWS KMS)
- [ ] Rotate database password quarterly
- [ ] Use least-privilege database user
- [ ] Disable unnecessary database extensions
- [ ] Enable pg_stat_statements for monitoring

**Monitoring & Alerting:**

- [ ] Enable Prometheus metrics
- [ ] Configure Grafana dashboards
- [ ] Set up Alertmanager
- [ ] Configure PagerDuty/Opsgenie
- [ ] Enable Sentry error tracking
- [ ] Set up log aggregation (ELK, Splunk)
- [ ] Monitor disk space (alert at 80%)
- [ ] Monitor memory usage (alert at 85%)
- [ ] Monitor CPU usage (alert at 80%)
- [ ] Monitor database connections (alert near pool limit)
- [ ] Monitor API error rate (alert > 1%)
- [ ] Monitor API response time (alert p95 > 1s)

### Security Scanning

**Container Scanning (Trivy):**

```bash
# Scan Docker image
trivy image greenlang/greenlang:0.3.0

# Scan with severity filter
trivy image --severity HIGH,CRITICAL greenlang/greenlang:0.3.0

# Generate report
trivy image --format json --output trivy-report.json greenlang/greenlang:0.3.0
```

**Dependency Scanning:**

```bash
# Python dependencies
pip-audit

# Or using Safety
safety check --file requirements.txt
```

**Secret Scanning (TruffleHog):**

```bash
# Scan Git history for secrets
trufflehog git file://. --json > secrets-scan.json

# Scan filesystem
trufflehog filesystem . --json
```

---

## Troubleshooting

### Common Issues

**Issue: API returns 503 Service Unavailable**

**Cause:** Database connection pool exhausted

**Solution:**
```bash
# Check database connections
kubectl logs deployment/greenlang-api -n greenlang | grep "pool"

# Increase pool size
kubectl set env deployment/greenlang-api DATABASE_POOL_SIZE=20 -n greenlang
```

**Issue: High memory usage, pods getting OOMKilled**

**Cause:** Memory leak or insufficient resources

**Solution:**
```bash
# Check memory usage
kubectl top pods -n greenlang

# Increase memory limit
kubectl set resources deployment greenlang-api --limits=memory=8Gi -n greenlang

# Or use VPA for automatic adjustment (see Scaling section)
```

**Issue: SSL certificate expired**

**Cause:** Let's Encrypt certificate not renewed

**Solution:**
```bash
# Renew certificate manually
certbot renew

# Check cert-manager (Kubernetes)
kubectl get certificate -n greenlang
kubectl describe certificate greenlang-tls -n greenlang

# Force renewal
kubectl delete secret greenlang-tls -n greenlang
kubectl delete certificate greenlang-tls -n greenlang
kubectl apply -f ingress.yaml
```

**Issue: Slow database queries**

**Cause:** Missing indexes, unoptimized queries

**Solution:**
```bash
# Enable slow query logging (PostgreSQL)
psql -U postgres -c "ALTER SYSTEM SET log_min_duration_statement = 1000;"
psql -U postgres -c "SELECT pg_reload_conf();"

# View slow queries
psql -U greenlang -d greenlang_prod -c "
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
"

# Add missing indexes (example)
psql -U greenlang -d greenlang_prod -c "
CREATE INDEX idx_calculations_created_at ON calculations(created_at);
"
```

### Debug Mode

Enable debug logging:

```bash
# Kubernetes
kubectl set env deployment/greenlang-api GL_LOG_LEVEL=DEBUG -n greenlang

# Docker Compose
# Edit docker-compose.yml, set GL_LOG_LEVEL=DEBUG, then:
docker-compose up -d

# Local development
export GL_LOG_LEVEL=DEBUG
gl server start
```

View detailed logs:

```bash
# Kubernetes
kubectl logs -f deployment/greenlang-api -n greenlang

# Docker
docker logs -f greenlang-api

# Follow logs with grep
kubectl logs -f deployment/greenlang-api -n greenlang | grep ERROR
```

---

## Performance Tuning

### Database Optimization

**PostgreSQL Configuration:**

```bash
# postgresql.conf optimizations for production

# Memory
shared_buffers = 4GB                # 25% of system RAM
effective_cache_size = 12GB         # 75% of system RAM
work_mem = 32MB                     # For sorting/hashing
maintenance_work_mem = 512MB        # For VACUUM, CREATE INDEX

# Checkpoints
checkpoint_timeout = 15min
checkpoint_completion_target = 0.9
max_wal_size = 4GB

# Query planner
random_page_cost = 1.1              # For SSD
effective_io_concurrency = 200      # For SSD

# Connections
max_connections = 200

# WAL
wal_buffers = 16MB
wal_compression = on

# Logging
log_min_duration_statement = 1000   # Log slow queries (>1s)
```

Apply and restart:
```bash
psql -U postgres -c "SELECT pg_reload_conf();"
# Or restart for some settings
systemctl restart postgresql
```

### API Performance

**Gunicorn/Uvicorn Workers:**

```bash
# Calculate optimal workers: (2 x CPU cores) + 1
# For 8-core server: 17 workers

kubectl set env deployment/greenlang-api API_WORKERS=17 -n greenlang
```

**Connection Pooling:**

```bash
# Increase database pool size
kubectl set env deployment/greenlang-api \
  DATABASE_POOL_SIZE=20 \
  DATABASE_MAX_OVERFLOW=40 \
  -n greenlang
```

**Redis Caching:**

```bash
# Increase cache TTL for stable data
kubectl set env deployment/greenlang-api REDIS_CACHE_TTL=7200 -n greenlang
```

### Load Testing

**Using Apache Bench:**

```bash
# 10,000 requests, 100 concurrent
ab -n 10000 -c 100 https://api.greenlang.example.com/health
```

**Using k6:**

```javascript
// load-test.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 200 },
    { duration: '5m', target: 200 },
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  let res = http.get('https://api.greenlang.example.com/health');
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
```

Run:
```bash
k6 run load-test.js
```

---

## Summary

This comprehensive deployment guide covers all aspects of deploying GreenLang in production:

1. **Prerequisites** - System requirements, dependencies, credentials
2. **Deployment Options** - Local, Docker, Kubernetes, Cloud
3. **Environment Variables** - Complete reference table
4. **Database Setup** - PostgreSQL initialization, migrations, backups
5. **Secrets Management** - Vault, AWS Secrets Manager, Azure Key Vault
6. **Monitoring** - Prometheus, Grafana, alerts
7. **Load Balancing** - Nginx, HAProxy configurations
8. **SSL/TLS** - Let's Encrypt, cert-manager, custom certificates
9. **Scaling** - HPA, VPA, manual scaling
10. **Backup & Recovery** - Full backups, PITR, disaster recovery
11. **Rollback** - Kubernetes, Docker Compose, database migrations
12. **Health Checks** - Startup, liveness, readiness probes
13. **Security Hardening** - Production security checklist
14. **Troubleshooting** - Common issues and solutions
15. **Performance Tuning** - Database, API, caching optimizations

**Next Steps:**

1. Choose deployment option based on your requirements
2. Follow step-by-step instructions for your chosen platform
3. Configure monitoring and alerting
4. Set up backups and disaster recovery
5. Run load tests to validate performance
6. Enable security hardening checklist items
7. Document your specific deployment configuration
8. Train your team on operational procedures

**Support:**

- Documentation: https://greenlang.io/docs
- GitHub Issues: https://github.com/greenlang/greenlang/issues
- Email: support@greenlang.io

---

**GreenLang Deployment Guide v1.0.0**
Last Updated: November 2025
