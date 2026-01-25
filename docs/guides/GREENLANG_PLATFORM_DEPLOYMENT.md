# GreenLang Carbon Intelligence Platform - Master Deployment Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-08
**Status:** Production Ready

---

## Executive Summary

This master deployment guide provides comprehensive instructions for deploying the complete **GreenLang Carbon Intelligence Platform**, consisting of three integrated applications:

1. **GL-CBAM-APP** - CBAM Importer Copilot (EU Carbon Border Adjustment Mechanism)
2. **GL-CSRD-APP** - CSRD Reporting Platform (Corporate Sustainability Reporting Directive)
3. **GL-VCCI-APP** - VCCI Scope 3 Platform (Value Chain Carbon Intelligence)

### Platform Overview

The GreenLang Platform is an enterprise-grade carbon accounting and compliance solution that enables organizations to:

- **Comply** with EU CBAM regulations for imported goods
- **Report** under CSRD/ESRS sustainability frameworks
- **Measure** and manage Scope 3 value chain emissions
- **Integrate** carbon data across the entire supply chain

### Key Features

- **Unified Architecture**: Shared infrastructure, databases, and monitoring
- **Microservices Design**: Each application is independently deployable
- **Scalable**: Supports small businesses to large enterprises
- **Cloud-Native**: Optimized for AWS, Azure, and GCP
- **Secure**: Enterprise-grade security and compliance
- **Observable**: Comprehensive monitoring and alerting

---

## Table of Contents

1. [Platform Architecture](#platform-architecture)
2. [Prerequisites](#prerequisites)
3. [Deployment Order](#deployment-order)
4. [Infrastructure Setup](#infrastructure-setup)
5. [Application Deployment](#application-deployment)
6. [Integration Configuration](#integration-configuration)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security & Compliance](#security--compliance)
9. [Disaster Recovery](#disaster-recovery)
10. [Scaling & Performance](#scaling--performance)
11. [Cost Optimization](#cost-optimization)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

---

## Platform Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GreenLang Carbon Platform                      │
│                       Load Balancer / CDN                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway / NGINX                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  GL-CBAM-APP │      │ GL-CSRD-APP  │      │ GL-VCCI-APP  │
│   Port 8001  │      │  Port 8002   │      │  Port 8000   │
├──────────────┤      ├──────────────┤      ├──────────────┤
│ CBAM Copilot │      │ CSRD/ESRS    │      │ Scope 3      │
│ - Intake     │      │ - LLM Agent  │      │ - 5 Agents   │
│ - Calculator │      │ - XBRL Gen   │      │ - Entity MDM │
│ - Reporter   │      │ - Reporting  │      │ - PCF Exch   │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SHARED INFRASTRUCTURE                        │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL Cluster     │  Redis Cluster   │  Weaviate Vector  │
│  (Primary + Replicas)   │  (Cache + Queue) │  Database         │
├─────────────────────────────────────────────────────────────────┤
│  Celery Workers         │  Message Queue   │  Object Storage   │
│  (Async Tasks)          │  (RabbitMQ/SQS)  │  (S3/Blob/GCS)    │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring Stack                                                │
│  Prometheus | Grafana | Loki | Jaeger | AlertManager           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXTERNAL INTEGRATIONS                           │
├─────────────────────────────────────────────────────────────────┤
│  ERP Systems    │  LLM Providers  │  Identity Providers         │
│  (SAP, Oracle)  │  (OpenAI, Claude)│  (Okta, Azure AD)          │
└─────────────────────────────────────────────────────────────────┘
```

### Application Dependencies

```
Deployment Order (Critical Path):
1. Infrastructure Layer
   ├── PostgreSQL Cluster
   ├── Redis Cluster
   ├── Weaviate (Vector DB)
   └── Object Storage

2. Platform Services
   ├── Message Queue (RabbitMQ/SQS)
   ├── Monitoring Stack
   └── API Gateway

3. Applications (Can be deployed in parallel)
   ├── GL-CBAM-APP (Standalone - minimal dependencies)
   ├── GL-CSRD-APP (Depends on PostgreSQL, Redis, LLM APIs)
   └── GL-VCCI-APP (Depends on All infrastructure + Weaviate)

4. Integration Layer
   ├── ERP Connectors
   ├── Cross-App Data Sync
   └── Unified Dashboard
```

---

## Prerequisites

### Technical Requirements

#### Minimum Infrastructure
- **CPU**: 16 vCPUs (total)
- **RAM**: 64 GB
- **Storage**: 500 GB SSD
- **Network**: 1 Gbps
- **OS**: Ubuntu 22.04 LTS or Amazon Linux 2023

#### Recommended Infrastructure (Production)
- **CPU**: 64 vCPUs (distributed)
- **RAM**: 256 GB
- **Storage**: 2 TB NVMe SSD
- **Network**: 10 Gbps
- **OS**: Ubuntu 22.04 LTS

### Software Dependencies

```yaml
Core Dependencies:
  - Docker: 24.0+
  - Docker Compose: 2.20+
  - Kubernetes: 1.28+ (optional, for k8s deployment)
  - Python: 3.11+
  - Node.js: 20 LTS
  - PostgreSQL: 15+
  - Redis: 7+
  - NGINX: 1.25+

Cloud Provider CLIs (choose one):
  - AWS CLI: 2.13+ (for AWS deployment)
  - Azure CLI: 2.50+ (for Azure deployment)
  - gcloud CLI: 440+ (for GCP deployment)

Monitoring:
  - Prometheus: 2.45+
  - Grafana: 10.0+
  - Loki: 2.8+
  - Jaeger: 1.48+
```

### Access & Credentials

Required API Keys and Credentials:

1. **LLM Provider APIs**
   - OpenAI API Key (GPT-4)
   - Anthropic API Key (Claude)

2. **Cloud Provider**
   - AWS Access Key + Secret (or IAM Role)
   - Azure Service Principal
   - GCP Service Account

3. **External Services**
   - SendGrid API Key (email notifications)
   - Sentry DSN (error tracking)
   - GitHub OAuth App (optional)

4. **Certificates**
   - SSL/TLS certificates for domains
   - Code signing certificate (optional)

### Network Requirements

```yaml
Inbound Ports:
  - 80/TCP    # HTTP (redirect to HTTPS)
  - 443/TCP   # HTTPS (all applications)
  - 8080/TCP  # Health checks (internal)
  - 9090/TCP  # Prometheus (internal)
  - 3000/TCP  # Grafana (internal)

Outbound Access:
  - https://api.openai.com (LLM)
  - https://api.anthropic.com (LLM)
  - Package registries (pip, npm, apt)
  - Cloud provider APIs
  - External data sources (GLEIF, D&B, etc.)

Internal Network:
  - PostgreSQL: 5432/TCP
  - Redis: 6379/TCP
  - Weaviate: 8080/TCP
  - RabbitMQ: 5672/TCP, 15672/TCP (management)
```

---

## Deployment Order

### Phase 1: Infrastructure Foundation (Day 0)

**Estimated Time:** 4-6 hours

1. **Network Setup**
   - Create VPC/VNet
   - Configure subnets (public, private, database)
   - Set up security groups/NSGs
   - Configure NAT Gateway
   - Set up VPN/Bastion host

2. **Storage Setup**
   - Create S3 buckets / Azure Blob / GCS buckets
   - Configure lifecycle policies
   - Set up backup storage
   - Create database volumes

3. **Database Cluster**
   - Deploy PostgreSQL cluster (primary + 2 replicas)
   - Configure replication
   - Create databases: `cbam_db`, `csrd_db`, `vcci_db`, `shared_db`
   - Run initial migrations
   - Set up connection pooling (PgBouncer)

4. **Cache & Queue**
   - Deploy Redis cluster (3 nodes)
   - Configure persistence (AOF)
   - Deploy message queue (RabbitMQ or use cloud-native SQS/Service Bus)

5. **Vector Database**
   - Deploy Weaviate cluster
   - Create schemas for Entity MDM
   - Configure backups

### Phase 2: Platform Services (Day 1)

**Estimated Time:** 2-4 hours

1. **API Gateway**
   - Deploy NGINX reverse proxy
   - Configure SSL/TLS certificates
   - Set up rate limiting
   - Configure routing rules

2. **Monitoring Stack**
   - Deploy Prometheus server
   - Deploy Grafana
   - Deploy Loki (log aggregation)
   - Deploy Jaeger (distributed tracing)
   - Configure AlertManager
   - Set up dashboards

3. **Secrets Management**
   - Deploy HashiCorp Vault or use cloud-native (AWS Secrets Manager, Azure Key Vault)
   - Store all credentials
   - Configure access policies

### Phase 3: Application Deployment (Day 2)

**Estimated Time:** 4-6 hours

Deploy applications in this order:

#### Step 1: GL-CBAM-APP (Simplest, minimal dependencies)

```bash
cd GL-CBAM-APP/CBAM-Importer-Copilot

# Build Docker image
docker build -t greenlang/cbam-app:1.0.0 .

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Verify
curl http://localhost:8001/health
```

**Resources Required:**
- 2 vCPUs
- 4 GB RAM
- 10 GB storage

#### Step 2: GL-CSRD-APP (Moderate dependencies)

```bash
cd GL-CSRD-APP/CSRD-Reporting-Platform

# Build Docker image
docker build -t greenlang/csrd-app:1.0.0 .

# Run database migrations
docker-compose run --rm web alembic upgrade head

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Verify
curl http://localhost:8002/health
```

**Resources Required:**
- 4 vCPUs
- 16 GB RAM
- 50 GB storage

#### Step 3: GL-VCCI-APP (Most complex, full dependencies)

```bash
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform

# Build Docker images
docker build -t greenlang/vcci-backend:2.0.0 -f backend/Dockerfile .
docker build -t greenlang/vcci-worker:2.0.0 -f worker/Dockerfile .
docker build -t greenlang/vcci-frontend:2.0.0 -f frontend/Dockerfile .

# Initialize Weaviate schemas
python scripts/init-weaviate-schema.py

# Run database migrations
docker-compose run --rm backend-api alembic upgrade head

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Verify
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

**Resources Required:**
- 8 vCPUs
- 32 GB RAM
- 200 GB storage

### Phase 4: Integration & Testing (Day 3)

**Estimated Time:** 4-6 hours

1. **Cross-Application Integration**
   - Configure shared authentication (JWT)
   - Set up data synchronization
   - Configure cross-app APIs
   - Test end-to-end workflows

2. **Load Balancer Configuration**
   - Configure health checks
   - Set up SSL termination
   - Configure sticky sessions (if needed)
   - Test failover scenarios

3. **Smoke Tests**
   - Run health checks on all services
   - Test basic workflows in each app
   - Verify monitoring data collection
   - Test alerting

### Phase 5: Go-Live Preparation (Day 4)

**Estimated Time:** 4-6 hours

1. **Performance Testing**
   - Load test each application
   - Stress test shared infrastructure
   - Verify auto-scaling (if configured)

2. **Security Hardening**
   - Run vulnerability scans
   - Apply security patches
   - Configure firewalls
   - Enable audit logging

3. **Documentation**
   - Update runbooks
   - Document credentials (in vault)
   - Create incident response plan
   - Train operations team

4. **Backup & DR**
   - Test database backups
   - Test disaster recovery procedures
   - Verify backup retention policies

---

## Infrastructure Setup

### Option 1: Docker Compose (Development / Small Production)

**Best for:** Small deployments, development, testing

**Pros:**
- Simple setup
- Low cost
- Easy to manage
- Fast deployment

**Cons:**
- Limited scalability
- Single point of failure
- Manual scaling

See: `deployment/docker-compose-unified.yml`

### Option 2: Kubernetes (Production / Enterprise)

**Best for:** Large deployments, high availability, auto-scaling

**Pros:**
- Highly scalable
- Self-healing
- Zero-downtime deployments
- Advanced networking

**Cons:**
- Complex setup
- Higher cost
- Steep learning curve

See: `deployment/kubernetes/`

### Option 3: Cloud-Managed Services (Recommended for Production)

#### AWS Architecture

```yaml
Compute:
  - ECS Fargate (for containers)
  - EC2 (for heavy workloads)
  - Lambda (for serverless functions)

Database:
  - RDS PostgreSQL (Multi-AZ)
  - ElastiCache Redis (cluster mode)
  - OpenSearch (for Weaviate alternative)

Storage:
  - S3 (object storage)
  - EFS (shared file system)
  - EBS (block storage)

Networking:
  - ALB (Application Load Balancer)
  - Route 53 (DNS)
  - VPC (network isolation)
  - CloudFront (CDN)

Monitoring:
  - CloudWatch (metrics & logs)
  - X-Ray (distributed tracing)

Security:
  - WAF (web application firewall)
  - Secrets Manager
  - IAM (identity & access)
  - GuardDuty (threat detection)
```

#### Azure Architecture

```yaml
Compute:
  - Azure Container Instances
  - Azure Kubernetes Service (AKS)
  - Azure Functions

Database:
  - Azure Database for PostgreSQL (HA)
  - Azure Cache for Redis
  - Azure Cosmos DB (vector search)

Storage:
  - Azure Blob Storage
  - Azure Files
  - Azure Disk Storage

Networking:
  - Azure Load Balancer
  - Azure Front Door (CDN)
  - Azure Virtual Network

Monitoring:
  - Azure Monitor
  - Application Insights
  - Log Analytics

Security:
  - Azure Firewall
  - Azure Key Vault
  - Azure AD
  - Microsoft Defender
```

#### GCP Architecture

```yaml
Compute:
  - Cloud Run (serverless containers)
  - GKE (Kubernetes)
  - Cloud Functions

Database:
  - Cloud SQL PostgreSQL (HA)
  - Memorystore Redis
  - Vertex AI Vector Search

Storage:
  - Cloud Storage (buckets)
  - Filestore
  - Persistent Disk

Networking:
  - Cloud Load Balancing
  - Cloud CDN
  - VPC

Monitoring:
  - Cloud Monitoring
  - Cloud Logging
  - Cloud Trace

Security:
  - Cloud Armor (WAF)
  - Secret Manager
  - IAM
  - Security Command Center
```

---

## Application Deployment

### GL-CBAM-APP Deployment

**Purpose:** EU CBAM Transitional Registry reporting

**Components:**
- Python application (FastAPI)
- No database required (can use shared DB for audit logs)
- Minimal dependencies

**Environment Variables:**

```bash
# Application
APP_ENV=production
LOG_LEVEL=INFO
WORKERS=4

# Optional: Database for audit logs
DATABASE_URL=postgresql://cbam_user:password@postgres:5432/cbam_db

# Optional: LLM (not used in core calculations)
OPENAI_API_KEY=sk-...
```

**Deployment Steps:**

1. Build image:
   ```bash
   docker build -t greenlang/cbam-app:1.0.0 \
     -f GL-CBAM-APP/CBAM-Importer-Copilot/Dockerfile \
     GL-CBAM-APP/CBAM-Importer-Copilot
   ```

2. Run migrations (if using database):
   ```bash
   docker run --rm \
     -e DATABASE_URL=postgresql://... \
     greenlang/cbam-app:1.0.0 \
     alembic upgrade head
   ```

3. Deploy:
   ```bash
   docker run -d \
     --name cbam-app \
     -p 8001:8000 \
     -e APP_ENV=production \
     greenlang/cbam-app:1.0.0
   ```

4. Verify:
   ```bash
   curl http://localhost:8001/health
   # Expected: {"status": "healthy", "version": "1.0.0"}
   ```

**Health Checks:**
- Endpoint: `/health`
- Expected: HTTP 200
- Timeout: 5s
- Interval: 30s

**Resource Limits:**
```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

---

### GL-CSRD-APP Deployment

**Purpose:** CSRD/ESRS digital reporting and XBRL generation

**Components:**
- Web application (FastAPI + Celery)
- PostgreSQL database
- Redis cache
- LLM integration (OpenAI, Anthropic)
- XBRL processor (Arelle)

**Environment Variables:**

```bash
# Application
APP_ENV=production
LOG_LEVEL=INFO
WORKERS=4

# Database
DATABASE_URL=postgresql://csrd_user:password@postgres:5432/csrd_db

# Cache
REDIS_URL=redis://:password@redis:6379/0

# Celery
CELERY_BROKER_URL=redis://:password@redis:6379/1
CELERY_RESULT_BACKEND=redis://:password@redis:6379/2

# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Vector Database (optional)
PINECONE_API_KEY=...
WEAVIATE_URL=http://weaviate:8080

# Security
JWT_SECRET=your-secret-key-min-32-chars
ALLOWED_ORIGINS=https://csrd.yourdomain.com

# Storage
S3_BUCKET=csrd-reports
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

**Deployment Steps:**

1. Build images:
   ```bash
   docker build -t greenlang/csrd-app:1.0.0 \
     -f GL-CSRD-APP/CSRD-Reporting-Platform/Dockerfile \
     GL-CSRD-APP/CSRD-Reporting-Platform
   ```

2. Initialize database:
   ```bash
   # Create database
   psql -h postgres -U postgres -c "CREATE DATABASE csrd_db;"
   psql -h postgres -U postgres -c "CREATE USER csrd_user WITH PASSWORD 'password';"
   psql -h postgres -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE csrd_db TO csrd_user;"

   # Run migrations
   docker run --rm \
     -e DATABASE_URL=postgresql://csrd_user:password@postgres:5432/csrd_db \
     greenlang/csrd-app:1.0.0 \
     alembic upgrade head
   ```

3. Deploy web application:
   ```bash
   docker run -d \
     --name csrd-web \
     -p 8002:8000 \
     -e DATABASE_URL=postgresql://... \
     -e REDIS_URL=redis://... \
     -e ANTHROPIC_API_KEY=... \
     greenlang/csrd-app:1.0.0
   ```

4. Deploy Celery workers:
   ```bash
   docker run -d \
     --name csrd-worker \
     -e DATABASE_URL=postgresql://... \
     -e REDIS_URL=redis://... \
     -e ANTHROPIC_API_KEY=... \
     greenlang/csrd-app:1.0.0 \
     celery -A tasks worker --loglevel=info --concurrency=4
   ```

5. Deploy Celery beat (scheduler):
   ```bash
   docker run -d \
     --name csrd-beat \
     -e DATABASE_URL=postgresql://... \
     -e REDIS_URL=redis://... \
     greenlang/csrd-app:1.0.0 \
     celery -A tasks beat --loglevel=info
   ```

6. Verify:
   ```bash
   curl http://localhost:8002/health
   curl http://localhost:8002/api/v1/status
   ```

**Health Checks:**
- Web: `/health`
- Database: `/health/db`
- Redis: `/health/redis`
- LLM: `/health/llm`

**Resource Limits:**
```yaml
Web Service:
  resources:
    requests:
      cpu: 1000m
      memory: 4Gi
    limits:
      cpu: 4000m
      memory: 16Gi

Worker Service:
  resources:
    requests:
      cpu: 2000m
      memory: 8Gi
    limits:
      cpu: 8000m
      memory: 32Gi
```

---

### GL-VCCI-APP Deployment

**Purpose:** Value Chain Carbon Intelligence (Scope 3 emissions)

**Components:**
- Backend API (FastAPI)
- Frontend (React)
- 5 Agent Services (Intake, Calculator, Hotspot, Engagement, Reporting)
- Celery workers
- PostgreSQL database
- Redis cache
- Weaviate vector database
- ERP connectors (SAP, Oracle, Workday)

**Environment Variables:**

```bash
# Application
APP_ENV=production
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://vcci:password@postgres:5432/vcci
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=10

# Cache
REDIS_URL=redis://:password@redis:6379/0

# Celery
CELERY_BROKER_URL=redis://:password@redis:6379/1
CELERY_RESULT_BACKEND=redis://:password@redis:6379/2

# Vector Database (Weaviate)
WEAVIATE_URL=http://weaviate:8080
WEAVIATE_API_KEY=...

# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Security
JWT_SECRET=your-secret-key-min-32-chars
JWT_EXPIRY_HOURS=24
ALLOWED_ORIGINS=https://vcci.yourdomain.com

# Storage
S3_BUCKET=vcci-data
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# ERP Connectors
SAP_BASE_URL=https://sap.company.com
SAP_CLIENT_ID=...
SAP_CLIENT_SECRET=...

ORACLE_CONNECTION_STRING=...
WORKDAY_API_URL=...
WORKDAY_TENANT=...

# Entity MDM
GLEIF_API_KEY=...
DNB_API_KEY=...

# Policy Engine
OPA_URL=http://opa:8181

# Monitoring
SENTRY_DSN=https://...@sentry.io/...
```

**Deployment Steps:**

1. Build images:
   ```bash
   # Backend API
   docker build -t greenlang/vcci-backend:2.0.0 \
     -f GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/Dockerfile \
     GL-VCCI-Carbon-APP/VCCI-Scope3-Platform

   # Worker
   docker build -t greenlang/vcci-worker:2.0.0 \
     -f GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/worker/Dockerfile \
     GL-VCCI-Carbon-APP/VCCI-Scope3-Platform

   # Frontend
   docker build -t greenlang/vcci-frontend:2.0.0 \
     -f GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/frontend/Dockerfile \
     GL-VCCI-Carbon-APP/VCCI-Scope3-Platform
   ```

2. Initialize infrastructure:
   ```bash
   # Create PostgreSQL database
   psql -h postgres -U postgres -c "CREATE DATABASE vcci;"
   psql -h postgres -U postgres -c "CREATE USER vcci WITH PASSWORD 'password';"
   psql -h postgres -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE vcci TO vcci;"

   # Initialize Weaviate schemas
   docker run --rm \
     -e WEAVIATE_URL=http://weaviate:8080 \
     greenlang/vcci-backend:2.0.0 \
     python scripts/init-weaviate-schema.py

   # Run database migrations
   docker run --rm \
     -e DATABASE_URL=postgresql://vcci:password@postgres:5432/vcci \
     greenlang/vcci-backend:2.0.0 \
     alembic upgrade head
   ```

3. Deploy backend API:
   ```bash
   docker run -d \
     --name vcci-backend-api \
     -p 8000:8000 \
     -e DATABASE_URL=postgresql://... \
     -e REDIS_URL=redis://... \
     -e WEAVIATE_URL=http://weaviate:8080 \
     -e ANTHROPIC_API_KEY=... \
     greenlang/vcci-backend:2.0.0 \
     uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

4. Deploy Celery workers:
   ```bash
   docker run -d \
     --name vcci-worker \
     -e DATABASE_URL=postgresql://... \
     -e REDIS_URL=redis://... \
     -e CELERY_BROKER_URL=redis://... \
     -e ANTHROPIC_API_KEY=... \
     greenlang/vcci-worker:2.0.0 \
     celery -A celery_app worker --loglevel=info --concurrency=4
   ```

5. Deploy Celery beat:
   ```bash
   docker run -d \
     --name vcci-beat \
     -e CELERY_BROKER_URL=redis://... \
     greenlang/vcci-worker:2.0.0 \
     celery -A celery_app beat --loglevel=info
   ```

6. Deploy frontend:
   ```bash
   docker run -d \
     --name vcci-frontend \
     -p 3000:3000 \
     -e REACT_APP_API_URL=http://localhost:8000 \
     greenlang/vcci-frontend:2.0.0
   ```

7. Verify:
   ```bash
   # Backend health
   curl http://localhost:8000/health/live
   curl http://localhost:8000/health/ready

   # Check Weaviate connection
   curl http://localhost:8000/health/weaviate

   # Check database
   curl http://localhost:8000/health/database

   # Frontend
   curl http://localhost:3000
   ```

**Health Checks:**
- Backend: `/health/live`, `/health/ready`
- Database: `/health/database`
- Redis: `/health/redis`
- Weaviate: `/health/weaviate`
- Workers: Celery flower dashboard at `:5555`

**Resource Limits:**
```yaml
Backend API:
  resources:
    requests:
      cpu: 2000m
      memory: 8Gi
    limits:
      cpu: 8000m
      memory: 32Gi

Worker:
  resources:
    requests:
      cpu: 4000m
      memory: 16Gi
    limits:
      cpu: 16000m
      memory: 64Gi

Frontend:
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi
```

---

## Integration Configuration

### Shared Authentication (JWT)

All three applications use unified JWT authentication:

```yaml
JWT Configuration:
  Secret: shared-secret-key-min-32-characters
  Algorithm: HS256
  Expiry: 24 hours
  Refresh: 7 days

Token Structure:
  {
    "sub": "user_id",
    "email": "user@example.com",
    "org_id": "org_123",
    "roles": ["admin", "analyst"],
    "apps": ["cbam", "csrd", "vcci"],
    "exp": 1699999999,
    "iat": 1699913599
  }
```

**Implementation:**

1. Deploy centralized auth service:
   ```bash
   docker run -d \
     --name greenlang-auth \
     -p 8080:8080 \
     -e JWT_SECRET=... \
     -e DATABASE_URL=... \
     greenlang/auth-service:1.0.0
   ```

2. Configure each app to validate JWT:
   ```python
   # In each app's middleware
   from jose import jwt, JWTError

   def verify_token(token: str):
       try:
           payload = jwt.decode(
               token,
               JWT_SECRET,
               algorithms=["HS256"]
           )
           return payload
       except JWTError:
           raise HTTPException(401, "Invalid token")
   ```

### Cross-Application Data Sync

**Scenario:** VCCI calculates Scope 3 emissions → CSRD uses for E1 reporting

**Implementation:**

1. **Webhook-based sync:**
   ```python
   # In VCCI app - after emissions calculation
   async def sync_to_csrd(emissions_data):
       async with httpx.AsyncClient() as client:
           response = await client.post(
               "http://csrd-app:8002/api/v1/sync/vcci-emissions",
               json=emissions_data,
               headers={"Authorization": f"Bearer {service_token}"}
           )
   ```

2. **Shared database tables:**
   ```sql
   -- In shared_db
   CREATE TABLE cross_app_emissions (
       id UUID PRIMARY KEY,
       source_app VARCHAR(50) NOT NULL,  -- 'vcci', 'cbam'
       target_app VARCHAR(50) NOT NULL,  -- 'csrd'
       org_id VARCHAR(255) NOT NULL,
       data_type VARCHAR(100) NOT NULL,  -- 'scope3_emissions', 'cbam_imports'
       payload JSONB NOT NULL,
       created_at TIMESTAMP DEFAULT NOW(),
       synced_at TIMESTAMP,
       INDEX idx_target_app (target_app, org_id)
   );
   ```

3. **Message queue-based sync (recommended for production):**
   ```yaml
   # In each app's config
   Message Queue:
     Type: RabbitMQ / AWS SQS
     Exchanges:
       - greenlang.emissions.calculated
       - greenlang.report.generated
       - greenlang.cbam.submitted

   VCCI → CSRD:
     Exchange: greenlang.emissions.calculated
     Routing Key: vcci.scope3.{org_id}
     Consumer: CSRD worker

   CBAM → CSRD:
     Exchange: greenlang.emissions.calculated
     Routing Key: cbam.imports.{org_id}
     Consumer: CSRD worker
   ```

### Unified API Gateway

**NGINX Configuration:**

```nginx
# /etc/nginx/conf.d/greenlang-platform.conf

upstream cbam_backend {
    server cbam-app:8001 max_fails=3 fail_timeout=30s;
}

upstream csrd_backend {
    server csrd-web:8002 max_fails=3 fail_timeout=30s;
}

upstream vcci_backend {
    server vcci-backend-api:8000 max_fails=3 fail_timeout=30s;
}

upstream vcci_frontend {
    server vcci-frontend:3000 max_fails=3 fail_timeout=30s;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=10r/s;

server {
    listen 80;
    server_name greenlang.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name greenlang.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000" always;

    # CBAM API
    location /api/cbam/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://cbam_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # CSRD API
    location /api/csrd/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://csrd_backend/api/v1/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;  # LLM calls can be slow
        proxy_read_timeout 300s;
    }

    # VCCI API
    location /api/vcci/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://vcci_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # VCCI Frontend (React app)
    location / {
        proxy_pass http://vcci_frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

---

## Monitoring & Observability

See: `monitoring/unified-dashboard.json` for complete Grafana dashboard configuration.

### Metrics Collection

**Prometheus Configuration:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'greenlang-platform'
    environment: 'production'

scrape_configs:
  # CBAM App
  - job_name: 'cbam-app'
    static_configs:
      - targets: ['cbam-app:8001']
    metrics_path: '/metrics'

  # CSRD App
  - job_name: 'csrd-app'
    static_configs:
      - targets: ['csrd-web:8002', 'csrd-worker:9090']
    metrics_path: '/metrics'

  # VCCI App
  - job_name: 'vcci-backend'
    static_configs:
      - targets: ['vcci-backend-api:8000']
    metrics_path: '/metrics'

  - job_name: 'vcci-worker'
    static_configs:
      - targets: ['vcci-worker:9090']
    metrics_path: '/metrics'

  # Infrastructure
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Node exporters
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

# Alerting rules
rule_files:
  - 'alerts/*.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Key Metrics to Monitor

```yaml
Application Metrics:
  # Request metrics
  - http_requests_total (counter)
  - http_request_duration_seconds (histogram)
  - http_requests_in_flight (gauge)

  # Business metrics
  - cbam_reports_generated_total (counter)
  - csrd_reports_generated_total (counter)
  - vcci_emissions_calculated_total (counter)
  - llm_api_calls_total (counter)
  - llm_api_cost_usd (counter)

  # Error metrics
  - http_errors_total (counter)
  - llm_api_errors_total (counter)
  - database_errors_total (counter)

Infrastructure Metrics:
  # Database
  - pg_stat_database_tup_fetched
  - pg_stat_database_tup_inserted
  - pg_stat_database_conflicts
  - pg_stat_database_deadlocks

  # Redis
  - redis_connected_clients
  - redis_memory_used_bytes
  - redis_commands_processed_total

  # System
  - node_cpu_seconds_total
  - node_memory_MemAvailable_bytes
  - node_disk_io_time_seconds_total
  - node_network_receive_bytes_total
```

### Alerting Rules

```yaml
# alerts/platform.yml
groups:
  - name: platform_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(http_errors_total[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "{{ $labels.app }} has error rate above 5% (current: {{ $value }})"

      # Database connection pool exhausted
      - alert: DatabasePoolExhausted
        expr: |
          pg_stat_database_numbackends / pg_settings_max_connections > 0.9
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool nearly exhausted"

      # LLM API high cost
      - alert: HighLLMCost
        expr: |
          rate(llm_api_cost_usd[1h]) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM API cost exceeding $100/hour"

      # Disk space low
      - alert: DiskSpaceLow
        expr: |
          node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Disk space below 10%"
```

### Log Aggregation (Loki)

```yaml
# promtail-config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Docker logs
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'
      - source_labels: ['__meta_docker_container_label_com_docker_compose_service']
        target_label: 'service'
    pipeline_stages:
      - json:
          expressions:
            level: level
            timestamp: timestamp
            message: message
            app: app
      - labels:
          level:
          app:
```

---

## Security & Compliance

### SSL/TLS Configuration

**Generate certificates:**

```bash
# Using Let's Encrypt (recommended for production)
certbot certonly --standalone \
  -d greenlang.yourdomain.com \
  -d cbam.greenlang.yourdomain.com \
  -d csrd.greenlang.yourdomain.com \
  -d vcci.greenlang.yourdomain.com \
  --email admin@yourdomain.com \
  --agree-tos

# Or use wildcard certificate
certbot certonly --dns-cloudflare \
  --dns-cloudflare-credentials ~/.secrets/cloudflare.ini \
  -d '*.greenlang.yourdomain.com' \
  -d greenlang.yourdomain.com
```

**Configure auto-renewal:**

```bash
# Add to crontab
0 0 * * * certbot renew --quiet && docker exec nginx nginx -s reload
```

### Secrets Management

**Using HashiCorp Vault:**

```bash
# Initialize Vault
docker run -d \
  --name vault \
  -p 8200:8200 \
  -e VAULT_DEV_ROOT_TOKEN_ID=root \
  vault:latest

# Store secrets
vault kv put secret/greenlang/jwt \
  secret="your-jwt-secret-min-32-chars"

vault kv put secret/greenlang/postgres \
  url="postgresql://user:pass@host:5432/db"

vault kv put secret/greenlang/anthropic \
  api_key="sk-ant-..."

vault kv put secret/greenlang/openai \
  api_key="sk-..."

# Create policy
vault policy write greenlang-app - <<EOF
path "secret/data/greenlang/*" {
  capabilities = ["read"]
}
EOF

# Create app token
vault token create -policy=greenlang-app
```

**Using AWS Secrets Manager:**

```bash
# Store secrets
aws secretsmanager create-secret \
  --name greenlang/jwt-secret \
  --secret-string "your-jwt-secret"

aws secretsmanager create-secret \
  --name greenlang/database-url \
  --secret-string "postgresql://..."

aws secretsmanager create-secret \
  --name greenlang/anthropic-api-key \
  --secret-string "sk-ant-..."

# Retrieve in application
import boto3

client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='greenlang/jwt-secret')
JWT_SECRET = response['SecretString']
```

### Access Control

**Role-Based Access Control (RBAC):**

```yaml
Roles:
  super_admin:
    description: Full platform access
    permissions:
      - platform:*
    apps: [cbam, csrd, vcci]

  org_admin:
    description: Organization administrator
    permissions:
      - org:{org_id}:read
      - org:{org_id}:write
      - users:{org_id}:manage
    apps: [cbam, csrd, vcci]

  compliance_officer:
    description: Compliance and reporting
    permissions:
      - org:{org_id}:read
      - reports:{org_id}:generate
      - reports:{org_id}:submit
    apps: [cbam, csrd]

  analyst:
    description: Data analysis
    permissions:
      - org:{org_id}:read
      - emissions:{org_id}:calculate
      - dashboards:{org_id}:view
    apps: [vcci]

  viewer:
    description: Read-only access
    permissions:
      - org:{org_id}:read
    apps: [cbam, csrd, vcci]
```

### Data Encryption

**At Rest:**

```yaml
PostgreSQL:
  - Enable transparent data encryption (TDE)
  - Encrypt backups with AES-256
  - Use encrypted EBS volumes (AWS) or encrypted disks (Azure/GCP)

Redis:
  - Enable encryption at rest (available in Redis 6+)
  - Use encrypted storage volumes

Object Storage:
  - S3: Enable server-side encryption (SSE-S3 or SSE-KMS)
  - Azure Blob: Enable encryption with platform-managed keys
  - GCS: Enable default encryption

Application Data:
  - Use Fernet encryption for sensitive fields
  - Encrypt PII data before storing
```

**In Transit:**

```yaml
External:
  - TLS 1.3 for all HTTPS connections
  - Certificate pinning for mobile apps
  - HSTS headers enabled

Internal:
  - mTLS between services (in Kubernetes with Istio/Linkerd)
  - Encrypted Redis connections
  - SSL-enabled PostgreSQL connections
```

### Vulnerability Scanning

```bash
# Docker image scanning
docker scan greenlang/cbam-app:1.0.0
docker scan greenlang/csrd-app:1.0.0
docker scan greenlang/vcci-backend:2.0.0

# Dependency scanning
# Python
pip-audit

# Node.js
npm audit

# Container runtime scanning (Trivy)
trivy image greenlang/vcci-backend:2.0.0

# Infrastructure scanning (Checkov)
checkov -d deployment/
```

### Compliance Certifications

```yaml
Target Certifications:
  - SOC 2 Type II
  - ISO 27001
  - GDPR compliant
  - CCPA compliant

Audit Logging:
  - All API requests logged
  - User actions logged
  - Data access logged
  - Log retention: 7 years
  - Immutable audit logs

Data Residency:
  - EU data stored in EU regions only
  - US data in US regions
  - Support for data sovereignty requirements
```

---

## Disaster Recovery

See: `deployment/platform-disaster-recovery.md` for detailed DR procedures.

### Backup Strategy

**PostgreSQL:**

```bash
# Automated daily backups
# Using pg_dump
0 2 * * * docker exec postgres \
  pg_dump -U postgres -F c vcci_db > /backups/vcci_db_$(date +\%Y\%m\%d).dump

# Using WAL archiving (continuous backup)
# In postgresql.conf
archive_mode = on
archive_command = 'aws s3 cp %p s3://greenlang-backups/wal/%f'

# Point-in-time recovery (PITR) capable
# Retention: 30 days
```

**Redis:**

```bash
# RDB snapshots every 5 minutes
save 300 1

# AOF for durability
appendonly yes
appendfsync everysec

# Backup AOF/RDB files to S3
0 * * * * aws s3 cp /data/appendonly.aof s3://greenlang-backups/redis/
```

**Weaviate:**

```bash
# Backup using Weaviate API
curl -X POST http://weaviate:8080/v1/backups/greenlang \
  -H "Content-Type: application/json" \
  -d '{
    "id": "greenlang-backup-'$(date +%Y%m%d-%H%M%S)'",
    "include": ["Entity", "Supplier", "Product"]
  }'

# Copy to S3
aws s3 sync /var/lib/weaviate/backups s3://greenlang-backups/weaviate/
```

**Application Data:**

```bash
# Backup configuration files
tar -czf /backups/config_$(date +%Y%m%d).tar.gz \
  /opt/greenlang/config \
  /opt/greenlang/.env \
  /opt/greenlang/deployment

# Backup to S3
aws s3 sync /backups s3://greenlang-backups/config/
```

### Recovery Procedures

**RTO (Recovery Time Objective): 4 hours**
**RPO (Recovery Point Objective): 1 hour**

**Disaster Scenarios:**

1. **Single Database Failure**
   - Failover to read replica (automatic)
   - RTO: 5 minutes
   - RPO: 0 (synchronous replication)

2. **Complete Database Cluster Failure**
   - Restore from latest backup
   - Replay WAL logs
   - RTO: 2 hours
   - RPO: 1 hour

3. **Application Server Failure**
   - Auto-scaling launches new instances
   - RTO: 10 minutes
   - RPO: 0

4. **Complete Region Failure**
   - Failover to DR region
   - Restore from S3 backups
   - RTO: 4 hours
   - RPO: 1 hour

**Recovery Steps:**

```bash
# 1. Restore PostgreSQL from backup
aws s3 cp s3://greenlang-backups/postgres/vcci_db_20251108.dump /tmp/
docker exec -i postgres pg_restore -U postgres -d vcci_db < /tmp/vcci_db_20251108.dump

# 2. Restore Redis from AOF
aws s3 cp s3://greenlang-backups/redis/appendonly.aof /data/
docker restart redis

# 3. Restore Weaviate
curl -X POST http://weaviate:8080/v1/backups/greenlang/restore \
  -H "Content-Type: application/json" \
  -d '{"id": "greenlang-backup-20251108-020000"}'

# 4. Redeploy applications
docker-compose -f docker-compose.prod.yml up -d

# 5. Verify health
./scripts/health-check-all.sh

# 6. Notify stakeholders
./scripts/send-dr-notification.sh
```

### High Availability Configuration

**PostgreSQL HA (using Patroni):**

```yaml
# 3-node PostgreSQL cluster
Cluster:
  Primary: postgres-1 (writes)
  Replica 1: postgres-2 (reads)
  Replica 2: postgres-3 (reads)
  Synchronous Replication: yes
  Automatic Failover: yes (via Patroni + etcd)
  Load Balancing: PgBouncer + HAProxy
```

**Redis Cluster:**

```yaml
# 6-node Redis cluster (3 masters, 3 replicas)
Cluster:
  Master 1: redis-1 (writes)
  Replica 1: redis-2 (reads)
  Master 2: redis-3 (writes)
  Replica 2: redis-4 (reads)
  Master 3: redis-5 (writes)
  Replica 3: redis-6 (reads)
  Automatic Failover: yes (Redis Sentinel)
```

**Application HA:**

```yaml
Each application:
  Min Instances: 2
  Max Instances: 10
  Health Check: HTTP /health
  Auto-scaling:
    - CPU > 70% for 5 min → scale up
    - CPU < 30% for 10 min → scale down
  Load Balancer: ALB / Azure LB / GCP LB
  Session Affinity: Cookie-based (if needed)
```

---

## Scaling & Performance

### Horizontal Scaling

**Application Tier:**

```yaml
CBAM App:
  Development: 1 instance
  Production Small: 2 instances
  Production Medium: 4 instances
  Production Large: 8 instances
  Max: 20 instances

CSRD App:
  Development: 1 web + 1 worker
  Production Small: 2 web + 2 workers
  Production Medium: 4 web + 8 workers
  Production Large: 8 web + 16 workers
  Max: 20 web + 50 workers

VCCI App:
  Development: 1 backend + 1 worker + 1 frontend
  Production Small: 2 backend + 4 workers + 2 frontend
  Production Medium: 4 backend + 8 workers + 4 frontend
  Production Large: 8 backend + 16 workers + 8 frontend
  Max: 20 backend + 100 workers + 10 frontend
```

**Database Tier:**

```yaml
PostgreSQL:
  Read Replicas: 0-10 (based on read load)
  Connection Pooling: PgBouncer (pool size: 100-1000)
  Sharding: By organization (if > 1000 orgs)

Redis:
  Cluster Nodes: 3-12 (based on memory requirements)
  Memory per Node: 8-64 GB

Weaviate:
  Cluster Nodes: 1-10 (based on vector data volume)
  Storage per Node: 100 GB - 2 TB
```

### Vertical Scaling

**Sizing Guide:**

| Deployment Size | Organizations | Users | Requests/Day | Infrastructure |
|----------------|---------------|-------|--------------|----------------|
| **Small** | 1-10 | 1-100 | 10K | 16 vCPUs, 64 GB RAM |
| **Medium** | 10-100 | 100-1000 | 100K | 64 vCPUs, 256 GB RAM |
| **Large** | 100-1000 | 1000-10000 | 1M | 256 vCPUs, 1 TB RAM |
| **Enterprise** | 1000+ | 10000+ | 10M+ | Custom |

### Performance Optimization

**Caching Strategy:**

```yaml
Redis Cache Layers:
  L1 - API Response Cache:
    TTL: 5 minutes
    Keys: api:{endpoint}:{params_hash}

  L2 - Database Query Cache:
    TTL: 1 hour
    Keys: db:{table}:{query_hash}

  L3 - LLM Response Cache:
    TTL: 24 hours
    Keys: llm:{model}:{prompt_hash}

  L4 - Computed Metrics:
    TTL: 6 hours
    Keys: metrics:{org_id}:{metric_type}

CDN Caching:
  Static Assets: 1 year
  API Responses (public): 1 hour
  User-specific: No cache
```

**Database Optimization:**

```sql
-- Create indexes for common queries
CREATE INDEX idx_emissions_org_date ON emissions(org_id, calculation_date);
CREATE INDEX idx_shipments_cbam ON shipments(cn_code, origin_country);
CREATE INDEX idx_csrd_reports_org ON csrd_reports(org_id, reporting_period);

-- Partition large tables
CREATE TABLE emissions_2025_q1 PARTITION OF emissions
  FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

-- Materialized views for dashboards
CREATE MATERIALIZED VIEW mv_org_emissions_summary AS
  SELECT
    org_id,
    DATE_TRUNC('month', calculation_date) AS month,
    SUM(total_emissions_tco2) AS total_emissions
  FROM emissions
  GROUP BY org_id, month;

REFRESH MATERIALIZED VIEW CONCURRENTLY mv_org_emissions_summary;
```

**LLM Cost Optimization:**

```yaml
Strategies:
  1. Aggressive Caching:
     - Cache identical prompts (exact match)
     - Cache similar prompts (semantic similarity > 95%)

  2. Model Selection:
     - Use GPT-3.5 for simple tasks
     - Use GPT-4 only for complex analysis
     - Use Claude Haiku for text processing
     - Use Claude Opus for critical compliance

  3. Prompt Optimization:
     - Minimize token count
     - Use few-shot learning
     - Batch requests where possible

  4. Rate Limiting:
     - Max 100 LLM calls per user per hour
     - Max 10,000 LLM calls per org per day

  5. Fallback Strategies:
     - Use rule-based logic when possible
     - Degrade gracefully if LLM unavailable

Expected Cost Reduction: 60-80%
```

---

## Cost Optimization

See: `deployment/cost-estimation.md` for detailed cost breakdown.

### Infrastructure Costs (AWS us-east-1)

**Small Deployment (1-10 orgs, 100 users):**

| Service | Specification | Monthly Cost |
|---------|---------------|--------------|
| EC2 (App Servers) | 2x t3.large (2 vCPU, 8 GB) | $120 |
| RDS PostgreSQL | db.t3.medium (Multi-AZ) | $150 |
| ElastiCache Redis | cache.t3.medium | $80 |
| S3 | 500 GB storage + 1000 GB transfer | $30 |
| ALB | 1 load balancer | $25 |
| Route 53 | 1 hosted zone | $1 |
| CloudWatch | Logs + Metrics | $20 |
| **Total Infrastructure** | | **$426/month** |
| LLM API Costs | ~10K requests/day | $200 |
| **Grand Total** | | **$626/month** |

**Medium Deployment (10-100 orgs, 1000 users):**

| Service | Specification | Monthly Cost |
|---------|---------------|--------------|
| EC2 (App Servers) | 8x m5.xlarge (4 vCPU, 16 GB) | $1,152 |
| RDS PostgreSQL | db.m5.xlarge (Multi-AZ) | $600 |
| ElastiCache Redis | cache.m5.large (cluster) | $400 |
| S3 | 5 TB storage + 10 TB transfer | $250 |
| ALB | 2 load balancers | $50 |
| Route 53 | 1 hosted zone | $1 |
| CloudWatch | Logs + Metrics | $100 |
| Weaviate (EC2) | 2x m5.2xlarge | $576 |
| **Total Infrastructure** | | **$3,129/month** |
| LLM API Costs | ~100K requests/day | $1,500 |
| **Grand Total** | | **$4,629/month** |

**Large Deployment (100-1000 orgs, 10K users):**

| Service | Specification | Monthly Cost |
|---------|---------------|--------------|
| ECS Fargate | 64 vCPUs, 256 GB RAM | $4,608 |
| RDS PostgreSQL | db.r5.4xlarge (Multi-AZ) | $3,600 |
| ElastiCache Redis | cache.r5.xlarge (cluster) | $1,500 |
| S3 | 50 TB storage + 100 TB transfer | $2,000 |
| ALB | 4 load balancers | $100 |
| Route 53 | 1 hosted zone | $1 |
| CloudWatch | Logs + Metrics | $500 |
| Weaviate (EC2) | 4x r5.4xlarge | $4,608 |
| **Total Infrastructure** | | **$16,917/month** |
| LLM API Costs | ~1M requests/day | $10,000 |
| **Grand Total** | | **$26,917/month** |

### Cost Optimization Strategies

1. **Reserved Instances / Savings Plans**
   - Save 30-60% on EC2/RDS with 1-year commitment
   - Target: $300-500/month savings

2. **Spot Instances for Workers**
   - Use spot instances for Celery workers (70% savings)
   - Graceful handling of interruptions
   - Target: $200-400/month savings

3. **S3 Lifecycle Policies**
   - Move old data to S3 Glacier (90% cheaper)
   - Delete temp files after 7 days
   - Target: $50-100/month savings

4. **LLM API Cost Reduction**
   - Aggressive caching (80% cache hit rate)
   - Use cheaper models where possible
   - Batch requests
   - Target: $1,000-5,000/month savings

5. **Auto-Scaling**
   - Scale down during off-hours (nights, weekends)
   - Target: $200-500/month savings

**Total Potential Savings: 30-40%**

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

**Symptoms:**
- `FATAL: remaining connection slots are reserved for non-replication superuser connections`
- `psycopg2.OperationalError: FATAL: too many connections`

**Solution:**
```bash
# Increase max_connections in PostgreSQL
psql -U postgres -c "ALTER SYSTEM SET max_connections = 200;"
docker restart postgres

# Or use connection pooling (PgBouncer)
docker run -d \
  --name pgbouncer \
  -e DB_HOST=postgres \
  -e DB_PORT=5432 \
  -e POOL_MODE=transaction \
  -e MAX_CLIENT_CONN=1000 \
  -e DEFAULT_POOL_SIZE=25 \
  pgbouncer/pgbouncer
```

#### 2. Redis Memory Issues

**Symptoms:**
- `OOM command not allowed when used memory > 'maxmemory'`
- Redis evicting keys unexpectedly

**Solution:**
```bash
# Increase maxmemory
redis-cli CONFIG SET maxmemory 8gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Or upgrade to larger instance
# AWS: cache.m5.large → cache.m5.xlarge
```

#### 3. LLM API Timeout

**Symptoms:**
- `httpx.ReadTimeout: timed out`
- LLM requests hanging

**Solution:**
```python
# Increase timeout
import httpx

client = httpx.AsyncClient(timeout=300.0)  # 5 minutes

# Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def call_llm(prompt):
    response = await client.post(...)
    return response
```

#### 4. Weaviate Out of Memory

**Symptoms:**
- `WARN: Memory usage at 95%`
- Slow vector searches

**Solution:**
```bash
# Scale up Weaviate instance
# Or increase LIMIT_RESOURCES in config
LIMIT_RESOURCES=false docker-compose restart weaviate

# Or add more Weaviate nodes (horizontal scaling)
```

#### 5. High CPU Usage

**Symptoms:**
- Application slow
- CPU consistently >80%

**Solution:**
```bash
# Identify CPU-intensive processes
docker stats

# Scale horizontally (add more instances)
docker-compose up -d --scale vcci-worker=8

# Or optimize code (use profiling)
python -m cProfile -o profile.out main.py
snakeviz profile.out
```

### Health Check Endpoints

```bash
# Platform-wide health check
curl http://localhost/health

# Individual apps
curl http://localhost:8001/health  # CBAM
curl http://localhost:8002/health  # CSRD
curl http://localhost:8000/health/live  # VCCI
curl http://localhost:8000/health/ready  # VCCI (includes dependencies)

# Infrastructure
curl http://localhost:5432  # PostgreSQL (should connect)
redis-cli ping  # Redis (should return PONG)
curl http://localhost:8080/v1/.well-known/ready  # Weaviate

# Monitoring
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health  # Grafana
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
docker-compose -f docker-compose.prod.yml up

# Access logs
docker logs -f cbam-app
docker logs -f csrd-web
docker logs -f vcci-backend-api

# PostgreSQL logs
docker exec postgres tail -f /var/log/postgresql/postgresql.log

# Redis logs
docker logs -f redis
```

---

## References

### Documentation Links

- [GL-CBAM-APP Documentation](GL-CBAM-APP/CBAM-Importer-Copilot/README.md)
- [GL-CSRD-APP Documentation](GL-CSRD-APP/CSRD-Reporting-Platform/README.md)
- [GL-VCCI-APP Documentation](GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/README.md)
- [Platform Architecture](architecture/platform-architecture.md)
- [Shared Infrastructure Setup](deployment/shared-infrastructure.md)
- [Cross-Application Integration](deployment/cross-application-integration.md)
- [Environment Sizing Guide](deployment/environment-sizing-guide.md)
- [Cost Estimation](deployment/cost-estimation.md)
- [Disaster Recovery Strategy](deployment/platform-disaster-recovery.md)
- [Production Go-Live Checklist](PRODUCTION_GO_LIVE_CHECKLIST.md)

### External Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [Azure Documentation](https://docs.microsoft.com/en-us/azure/)
- [GCP Documentation](https://cloud.google.com/docs)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

### Support Contacts

```yaml
Technical Support:
  Email: support@greenlang.io
  Slack: greenlang-platform.slack.com
  Response Time: 4 hours (business hours)

Emergency (Production Down):
  Phone: +1-xxx-xxx-xxxx
  On-Call: Available 24/7
  Response Time: 30 minutes

Sales & Licensing:
  Email: sales@greenlang.io

Community:
  GitHub: github.com/akshay-greenlang/Code-V1_GreenLang
  Discussions: github.com/akshay-greenlang/Code-V1_GreenLang/discussions
```

---

## Changelog

### Version 1.0.0 (2025-11-08)

- Initial release
- Complete deployment guide for all 3 applications
- Unified infrastructure architecture
- Monitoring and observability setup
- Disaster recovery procedures
- Security and compliance guidelines
- Cost optimization strategies

---

## License

Copyright (c) 2025 GreenLang. All rights reserved.

This documentation is proprietary and confidential.

---

**Document Owner:** Platform Engineering Team
**Last Reviewed:** 2025-11-08
**Next Review:** 2025-12-08
