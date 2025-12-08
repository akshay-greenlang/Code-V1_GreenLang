# GreenLang Administrator Training

**Document Version:** 1.0
**Last Updated:** December 2025
**Audience:** System Administrators, DevOps Engineers, Platform Engineers
**Prerequisites:** Completed [01_getting_started.md](01_getting_started.md), Linux/Infrastructure experience

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Installation and Deployment](#installation-and-deployment)
4. [Configuration Management](#configuration-management)
5. [Security Administration](#security-administration)
6. [User and Access Management](#user-and-access-management)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Backup and Recovery](#backup-and-recovery)
9. [Performance Tuning](#performance-tuning)
10. [High Availability](#high-availability)
11. [Upgrade Procedures](#upgrade-procedures)
12. [Compliance and Auditing](#compliance-and-auditing)
13. [Incident Response](#incident-response)

---

## Introduction

This training module prepares administrators to deploy, configure, monitor, and maintain GreenLang installations. Upon completion, you will be able to:

- Deploy GreenLang in production environments
- Configure security and access controls
- Monitor system health and performance
- Perform backup and recovery operations
- Manage high availability configurations
- Respond to incidents effectively

---

## System Architecture

### Deployment Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Load Balancer                   │
                    │         (NGINX / AWS ALB / GCP LB)          │
                    └─────────────────┬───────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
┌───────▼───────┐           ┌─────────▼────────┐          ┌────────▼───────┐
│  API Server   │           │   API Server     │          │  API Server    │
│  (Instance 1) │           │   (Instance 2)   │          │  (Instance 3)  │
└───────┬───────┘           └─────────┬────────┘          └────────┬───────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
┌───────▼───────┐           ┌─────────▼────────┐          ┌────────▼───────┐
│   Worker      │           │    Worker        │          │   Worker       │
│   (Pod 1)     │           │    (Pod 2)       │          │   (Pod 3)      │
└───────────────┘           └──────────────────┘          └────────────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
        ┌──────────────┬──────────────┴───────────┬─────────────────┐
        │              │                          │                 │
┌───────▼───────┐ ┌────▼────┐           ┌─────────▼─────┐  ┌───────▼───────┐
│  PostgreSQL   │ │  Redis  │           │  Message Queue │  │  Object Store │
│   Primary     │ │ Cluster │           │   (RabbitMQ)   │  │     (S3)      │
└───────┬───────┘ └─────────┘           └───────────────┘  └───────────────┘
        │
┌───────▼───────┐
│  PostgreSQL   │
│   Replica     │
└───────────────┘
```

### Component Requirements

| Component | Minimum | Recommended | High Availability |
|-----------|---------|-------------|-------------------|
| **API Servers** | 1 | 3 | 3+ across zones |
| **Workers** | 2 | 4 | 6+ across zones |
| **PostgreSQL** | 1 (8GB) | Primary + Replica | Multi-AZ |
| **Redis** | 1 (2GB) | 3-node cluster | Sentinel/Cluster |
| **Message Queue** | 1 | 3-node cluster | Mirrored queues |

### Port Reference

| Port | Service | Protocol | Description |
|------|---------|----------|-------------|
| 80 | HTTP | TCP | Redirect to HTTPS |
| 443 | HTTPS | TCP | API and Web UI |
| 5432 | PostgreSQL | TCP | Database |
| 6379 | Redis | TCP | Cache |
| 5672 | RabbitMQ | TCP | Message queue |
| 15672 | RabbitMQ Management | TCP | Admin UI |
| 9090 | Prometheus | TCP | Metrics |
| 3000 | Grafana | TCP | Dashboards |

---

## Installation and Deployment

### Prerequisites Checklist

```bash
# Verify prerequisites
echo "Checking prerequisites..."

# Docker
docker --version  # >= 20.10

# Kubernetes (if using K8s)
kubectl version --client  # >= 1.25

# Helm (if using Helm)
helm version  # >= 3.10

# PostgreSQL client
psql --version  # >= 14

# Required ports available
netstat -tlnp | grep -E '(443|5432|6379)'
```

### Option 1: Docker Compose (Development/Small Deployments)

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  api:
    image: greenlang/greenlang:${VERSION:-latest}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://greenlang:${DB_PASSWORD}@db:5432/greenlang
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
    depends_on:
      - db
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    image: greenlang/greenlang-worker:${VERSION:-latest}
    deploy:
      replicas: 4
    environment:
      - DATABASE_URL=postgresql://greenlang:${DB_PASSWORD}@db:5432/greenlang
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://rabbit:5672

  db:
    image: postgres:14
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=greenlang
      - POSTGRES_PASSWORD=${DB_PASSWORD}

  redis:
    image: redis:6-alpine
    volumes:
      - redisdata:/data

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - api

volumes:
  pgdata:
  redisdata:
```

### Option 2: Kubernetes (Production)

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: greenlang
  labels:
    name: greenlang

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: greenlang-config
  namespace: greenlang
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  WORKERS_PER_POD: "4"

---
# secrets.yaml (use sealed-secrets or external-secrets in production)
apiVersion: v1
kind: Secret
metadata:
  name: greenlang-secrets
  namespace: greenlang
type: Opaque
stringData:
  DATABASE_URL: "postgresql://greenlang:password@postgres:5432/greenlang"
  SECRET_KEY: "your-secret-key-here"
  REDIS_URL: "redis://redis:6379"

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-api
  namespace: greenlang
spec:
  replicas: 3
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
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: greenlang-api
              topologyKey: "topology.kubernetes.io/zone"
      containers:
      - name: api
        image: greenlang/greenlang:1.0.0
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: greenlang-config
        - secretRef:
            name: greenlang-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          runAsNonRoot: true
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: greenlang-api
  namespace: greenlang
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: greenlang-api

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: greenlang-ingress
  namespace: greenlang
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.greenlang.io
    secretName: greenlang-tls
  rules:
  - host: api.greenlang.io
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

### Installation Commands

```bash
# Apply Kubernetes manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Verify deployment
kubectl get pods -n greenlang
kubectl get services -n greenlang

# Check logs
kubectl logs -n greenlang -l app=greenlang-api --tail=100
```

---

## Configuration Management

### Configuration Hierarchy

```
1. Environment variables (highest priority)
2. Configuration files (greenlang.yaml)
3. Default values (lowest priority)
```

### Main Configuration File

```yaml
# /etc/greenlang/greenlang.yaml
# GreenLang Production Configuration

# General settings
general:
  environment: production
  debug: false
  log_level: INFO
  timezone: UTC

# Database configuration
database:
  url: ${DATABASE_URL}
  pool_size: 20
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600
  ssl_mode: require

# Redis configuration
redis:
  url: ${REDIS_URL}
  max_connections: 100
  socket_timeout: 5
  socket_connect_timeout: 5

# API configuration
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 60
  max_request_size: 10485760  # 10MB
  rate_limit:
    enabled: true
    requests_per_minute: 100

# Security
security:
  secret_key: ${SECRET_KEY}
  jwt_expiry_hours: 24
  password_min_length: 12
  mfa_enabled: true
  session_timeout_minutes: 60

# Emission factors
emissions:
  default_region: US
  emission_factors_source: EPA
  update_frequency_days: 30
  cache_ttl_hours: 24

# Monitoring
monitoring:
  metrics_enabled: true
  metrics_port: 9090
  tracing_enabled: true
  tracing_sample_rate: 0.1

# Logging
logging:
  format: json
  output: stdout
  level: INFO
  include_request_id: true
  sensitive_fields:
    - password
    - api_key
    - secret
    - token
```

### Environment-Specific Overrides

```bash
# /etc/greenlang/env.d/production.env

# Database
DATABASE_URL=postgresql://greenlang:password@db-primary.internal:5432/greenlang?sslmode=require

# Redis
REDIS_URL=redis://redis-cluster.internal:6379

# Security
SECRET_KEY=<generated-secret-key>

# Feature flags
FEATURE_NEW_DASHBOARD=false
FEATURE_BETA_ML=false
```

### Configuration Validation

```bash
# Validate configuration before deployment
greenlang config validate --config /etc/greenlang/greenlang.yaml

# Check for sensitive data exposure
greenlang config audit --config /etc/greenlang/greenlang.yaml

# Generate configuration documentation
greenlang config docs --output config-reference.md
```

---

## Security Administration

### SSL/TLS Configuration

```nginx
# /etc/nginx/conf.d/greenlang.conf
server {
    listen 443 ssl http2;
    server_name api.greenlang.io;

    ssl_certificate /etc/nginx/certs/greenlang.crt;
    ssl_certificate_key /etc/nginx/certs/greenlang.key;

    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass http://greenlang-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Firewall Rules

```bash
# UFW example
ufw default deny incoming
ufw default allow outgoing

# Allow SSH
ufw allow 22/tcp

# Allow HTTPS
ufw allow 443/tcp

# Allow internal services (adjust subnet)
ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL
ufw allow from 10.0.0.0/8 to any port 6379  # Redis

ufw enable
```

### Secrets Management

```bash
# Using HashiCorp Vault
vault kv put secret/greenlang \
    database_password="secure-password" \
    secret_key="$(openssl rand -base64 32)" \
    api_key="$(openssl rand -hex 32)"

# Kubernetes external-secrets
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: greenlang-secrets
  namespace: greenlang
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: ClusterSecretStore
    name: vault-backend
  target:
    name: greenlang-secrets
  data:
  - secretKey: DATABASE_PASSWORD
    remoteRef:
      key: secret/greenlang
      property: database_password
```

### Security Hardening Checklist

```
[ ] TLS 1.2+ enforced
[ ] Strong cipher suites configured
[ ] HSTS enabled
[ ] Security headers configured
[ ] Database connections encrypted
[ ] Secrets stored in vault
[ ] API keys rotated regularly
[ ] MFA enabled for admin accounts
[ ] Audit logging enabled
[ ] Network segmentation configured
[ ] Container security scanning
[ ] Dependency vulnerability scanning
```

---

## User and Access Management

### Role-Based Access Control

```yaml
# roles.yaml
roles:
  - name: admin
    description: Full system access
    permissions:
      - "*:*"

  - name: operator
    description: Day-to-day operations
    permissions:
      - calculations:create
      - calculations:read
      - calculations:update
      - reports:create
      - reports:read
      - reports:export
      - alarms:acknowledge
      - alarms:read

  - name: developer
    description: API integration access
    permissions:
      - calculations:*
      - agents:read
      - pipelines:read
      - webhooks:*

  - name: auditor
    description: Read-only audit access
    permissions:
      - calculations:read
      - reports:read
      - audit:read
      - provenance:read

  - name: viewer
    description: Dashboard read-only
    permissions:
      - calculations:read
      - reports:read
      - dashboards:read
```

### User Management Commands

```bash
# Create user
greenlang user create \
    --username jsmith \
    --email john.smith@company.com \
    --role operator \
    --require-mfa

# List users
greenlang user list --format table

# Modify user role
greenlang user modify jsmith --role admin

# Disable user
greenlang user disable jsmith

# Reset password
greenlang user reset-password jsmith

# Enable MFA
greenlang user enable-mfa jsmith
```

### API Key Management

```bash
# Generate API key
greenlang api-key create \
    --name "Integration Key" \
    --scope calculations:read,calculations:create \
    --expires-in 365d

# List API keys
greenlang api-key list

# Revoke API key
greenlang api-key revoke KEY_ID

# Rotate API key
greenlang api-key rotate KEY_ID
```

---

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'greenlang-api'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - greenlang
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

  - job_name: 'greenlang-workers'
    static_configs:
      - targets: ['worker-1:9090', 'worker-2:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - /etc/prometheus/alerts/*.yml
```

### Alert Rules

```yaml
# alerts/greenlang.yml
groups:
  - name: greenlang
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) /
          sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          description: Error rate is {{ $value | humanizePercentage }}

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High API latency
          description: P99 latency is {{ $value | humanizeDuration }}

      - alert: DatabaseConnectionPoolExhausted
        expr: greenlang_db_pool_available < 5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: Database connection pool nearly exhausted

      - alert: QueueBacklogHigh
        expr: greenlang_queue_depth > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: Message queue backlog is high
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "GreenLang Operations",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (status)"
          }
        ]
      },
      {
        "title": "Latency P99",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Active Calculations",
        "type": "gauge",
        "targets": [
          {
            "expr": "greenlang_calculations_active"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

```yaml
# fluent-bit.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Log_Level     info
        Parsers_File  parsers.conf

    [INPUT]
        Name              tail
        Path              /var/log/containers/greenlang*.log
        Parser            json
        Tag               greenlang.*

    [FILTER]
        Name              kubernetes
        Match             greenlang.*
        Merge_Log         On

    [OUTPUT]
        Name              elasticsearch
        Match             *
        Host              elasticsearch
        Port              9200
        Index             greenlang-logs
```

---

## Backup and Recovery

### Backup Strategy

| Component | Method | Frequency | Retention |
|-----------|--------|-----------|-----------|
| PostgreSQL | pg_dump + WAL | Hourly/Continuous | 30 days |
| Redis | RDB + AOF | Every 15 min | 7 days |
| Configurations | Git + S3 | On change | Forever |
| Secrets | Vault backup | Daily | 90 days |

### Database Backup Script

```bash
#!/bin/bash
# /opt/greenlang/scripts/backup-db.sh

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/greenlang"
S3_BUCKET="s3://greenlang-backups/database"

# Create backup
pg_dump \
    --host=db-primary.internal \
    --username=greenlang \
    --format=custom \
    --file="${BACKUP_DIR}/greenlang_${TIMESTAMP}.dump"

# Compress
gzip "${BACKUP_DIR}/greenlang_${TIMESTAMP}.dump"

# Upload to S3
aws s3 cp \
    "${BACKUP_DIR}/greenlang_${TIMESTAMP}.dump.gz" \
    "${S3_BUCKET}/greenlang_${TIMESTAMP}.dump.gz"

# Clean old local backups (keep 7 days)
find "${BACKUP_DIR}" -name "*.dump.gz" -mtime +7 -delete

echo "Backup completed: greenlang_${TIMESTAMP}.dump.gz"
```

### Point-in-Time Recovery

```bash
# Stop application
kubectl scale deployment greenlang-api --replicas=0 -n greenlang

# Restore from backup
pg_restore \
    --host=db-primary.internal \
    --username=greenlang \
    --dbname=greenlang \
    --clean \
    /path/to/backup.dump

# Apply WAL logs to point in time
# (Requires PostgreSQL configured for WAL archiving)

# Restart application
kubectl scale deployment greenlang-api --replicas=3 -n greenlang
```

### Disaster Recovery Runbook

```markdown
## Disaster Recovery Procedure

### 1. Assess Situation
- Identify affected components
- Estimate data loss window
- Notify stakeholders

### 2. Failover to DR Site (if applicable)
- Update DNS to DR load balancer
- Verify DR database is current
- Scale up DR infrastructure

### 3. Restore from Backup
- Identify most recent clean backup
- Restore database
- Restore configurations
- Restore secrets from vault

### 4. Verify System
- Run health checks
- Verify data integrity
- Run smoke tests
- Check recent calculations

### 5. Resume Operations
- Notify users
- Monitor closely for 24 hours
- Document incident
```

---

## Performance Tuning

### Database Tuning

```sql
-- postgresql.conf optimizations
-- Memory
shared_buffers = 4GB                # 25% of RAM
effective_cache_size = 12GB         # 75% of RAM
work_mem = 64MB
maintenance_work_mem = 512MB

-- Checkpoints
checkpoint_completion_target = 0.9
wal_buffers = 64MB
max_wal_size = 4GB

-- Query planning
random_page_cost = 1.1              # For SSD
effective_io_concurrency = 200       # For SSD

-- Connections
max_connections = 200
```

### Redis Tuning

```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
activedefrag yes
```

### Application Tuning

```yaml
# Worker and connection pool tuning
api:
  workers: 8                    # 2 * CPU cores
  worker_connections: 1000
  keepalive_timeout: 65

database:
  pool_size: 20                 # Connections per worker
  max_overflow: 10
  pool_pre_ping: true

redis:
  max_connections: 100
  socket_timeout: 5

calculation:
  batch_size: 1000
  parallel_workers: 4
```

---

## High Availability

### Multi-Region Setup

```
┌────────────────────────────────────────────────────────────────────┐
│                     Global Load Balancer                           │
│                   (Route53 / CloudFlare)                          │
└───────────────────────────┬────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
    ┌───────▼───────┐               ┌───────▼───────┐
    │   US-EAST-1   │               │   EU-WEST-1   │
    │   (Primary)   │               │  (Secondary)  │
    ├───────────────┤               ├───────────────┤
    │ API Servers   │               │ API Servers   │
    │ Workers       │               │ Workers       │
    │ DB Primary    │──────────────▶│ DB Replica    │
    │ Redis         │               │ Redis         │
    └───────────────┘               └───────────────┘
```

### Failover Configuration

```yaml
# HAProxy configuration
global
    maxconn 10000

defaults
    mode http
    timeout connect 5s
    timeout client 30s
    timeout server 30s
    option httpchk GET /health

frontend api
    bind *:443 ssl crt /etc/haproxy/certs/greenlang.pem
    default_backend api_servers

backend api_servers
    balance roundrobin
    option httpchk GET /health
    server api1 10.0.1.10:8000 check inter 5s fall 3 rise 2
    server api2 10.0.1.11:8000 check inter 5s fall 3 rise 2
    server api3 10.0.1.12:8000 check inter 5s fall 3 rise 2
    server api_dr 10.1.1.10:8000 check inter 5s fall 3 rise 2 backup
```

---

## Upgrade Procedures

### Pre-Upgrade Checklist

```
[ ] Review release notes
[ ] Test upgrade in staging
[ ] Backup database
[ ] Backup configurations
[ ] Notify users of maintenance window
[ ] Prepare rollback plan
```

### Rolling Upgrade (Kubernetes)

```bash
# Update image version
kubectl set image deployment/greenlang-api \
    api=greenlang/greenlang:1.1.0 \
    -n greenlang

# Monitor rollout
kubectl rollout status deployment/greenlang-api -n greenlang

# If issues, rollback
kubectl rollout undo deployment/greenlang-api -n greenlang
```

### Database Migration

```bash
# Apply migrations
greenlang db migrate --target 1.1.0

# Verify migration
greenlang db status

# Rollback if needed
greenlang db rollback --target 1.0.0
```

---

## Compliance and Auditing

### Audit Log Configuration

```yaml
audit:
  enabled: true
  storage: elasticsearch
  retention_days: 365
  events:
    - user.login
    - user.logout
    - calculation.created
    - calculation.modified
    - report.generated
    - settings.changed
    - api_key.created
    - api_key.revoked
```

### Compliance Reports

```bash
# Generate SOC 2 compliance report
greenlang compliance report --standard soc2 --period 2025-Q4

# Generate audit trail export
greenlang audit export \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --format csv \
    --output audit-2025.csv
```

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| SEV1 | System down | Immediate | Management + On-call |
| SEV2 | Major degradation | 15 minutes | On-call team |
| SEV3 | Minor issue | 1 hour | Support team |
| SEV4 | Low impact | Next business day | Ticket system |

### Incident Response Procedure

```markdown
## Incident Response Steps

### 1. Detection and Triage (5 min)
- Acknowledge alert
- Assess severity
- Create incident ticket
- Notify appropriate team

### 2. Investigation (15 min)
- Check dashboards and metrics
- Review recent changes
- Check error logs
- Identify root cause

### 3. Mitigation (30 min)
- Apply immediate fix
- Rollback if necessary
- Scale resources if needed
- Communicate status

### 4. Resolution
- Implement permanent fix
- Verify system stable
- Update documentation
- Close incident

### 5. Post-Incident Review
- Root cause analysis
- Timeline documentation
- Action items
- Prevention measures
```

---

## Certification Checklist

### Knowledge Assessment

```
[ ] Understand deployment architecture
[ ] Know configuration management
[ ] Understand security controls
[ ] Know monitoring setup
[ ] Understand backup procedures
[ ] Know HA configuration
```

### Practical Skills

```
[ ] Deploy GreenLang cluster
[ ] Configure SSL/TLS
[ ] Set up monitoring and alerting
[ ] Perform backup and restore
[ ] Execute rolling upgrade
[ ] Respond to simulated incident
```

---

**Training Complete!** You are now ready to administer GreenLang systems.
