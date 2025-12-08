# GreenLang Administrator Training

## Module Overview

**Duration:** 12 hours
**Prerequisites:** [Fundamentals](./fundamentals.md)
**Level:** Advanced

This module prepares system administrators to install, configure, secure, and maintain GreenLang Process Heat Agents in production environments.

---

## Part 1: Installation and Configuration

### 1.1 System Requirements

#### Hardware Requirements

| Component | Minimum | Recommended | Enterprise |
|-----------|---------|-------------|------------|
| CPU | 4 cores | 8 cores | 16+ cores |
| RAM | 8 GB | 16 GB | 32+ GB |
| Storage | 50 GB SSD | 200 GB SSD | 500+ GB NVMe |
| Network | 100 Mbps | 1 Gbps | 10 Gbps |

#### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Operating System | Ubuntu 20.04+, RHEL 8+, Windows Server 2019+ | Linux preferred for production |
| Docker | 20.10+ | Required for containerized deployment |
| Kubernetes | 1.25+ | Optional, for orchestrated deployment |
| Python | 3.9+ | Required for agent runtime |
| PostgreSQL | 14+ | Primary database |
| Redis | 7.0+ | Caching and message queue |
| TimescaleDB | 2.10+ | Time-series data storage |

### 1.2 Installation Methods

#### Method 1: Docker Compose (Development/Small Production)

```bash
# Download GreenLang
git clone https://github.com/greenlang/greenlang.git
cd greenlang

# Configure environment
cp .env.example .env
nano .env  # Edit configuration

# Start services
docker-compose up -d

# Verify installation
docker-compose ps
curl http://localhost:8000/health
```

**docker-compose.yml Overview:**

```yaml
version: '3.8'
services:
  greenlang-api:
    image: greenlang/api:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

  greenlang-agent:
    image: greenlang/agent:latest
    environment:
      - API_URL=http://greenlang-api:8000
    depends_on:
      - greenlang-api

  greenlang-ml:
    image: greenlang/ml-engine:latest
    environment:
      - API_URL=http://greenlang-api:8000
    depends_on:
      - greenlang-api

  postgres:
    image: timescale/timescaledb:latest-pg14
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=greenlang
      - POSTGRES_USER=greenlang
      - POSTGRES_PASSWORD=${DB_PASSWORD}

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

#### Method 2: Kubernetes (Production)

```bash
# Add Helm repository
helm repo add greenlang https://charts.greenlang.io
helm repo update

# Create namespace
kubectl create namespace greenlang

# Create secrets
kubectl create secret generic greenlang-secrets \
  --namespace greenlang \
  --from-literal=db-password='your-secure-password' \
  --from-literal=jwt-secret='your-jwt-secret'

# Install GreenLang
helm install greenlang greenlang/greenlang \
  --namespace greenlang \
  --values values.yaml

# Verify installation
kubectl get pods -n greenlang
```

**values.yaml Example:**

```yaml
global:
  environment: production
  domain: greenlang.yourcompany.com

api:
  replicas: 3
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi

agent:
  replicas: 2
  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 4000m
      memory: 8Gi

mlEngine:
  replicas: 2
  gpu:
    enabled: true
    count: 1
  resources:
    requests:
      cpu: 2000m
      memory: 8Gi
    limits:
      cpu: 8000m
      memory: 32Gi

postgresql:
  enabled: true
  primary:
    persistence:
      size: 100Gi

redis:
  enabled: true
  replica:
    replicaCount: 2

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: greenlang.yourcompany.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: greenlang-tls
      hosts:
        - greenlang.yourcompany.com
```

#### Method 3: Air-Gapped Installation

For networks without internet access:

```bash
# On connected system: Download packages
greenlang-cli package download --output greenlang-bundle.tar.gz

# Transfer to air-gapped system
# (USB drive, secure file transfer, etc.)

# On air-gapped system: Import packages
greenlang-cli package import --input greenlang-bundle.tar.gz

# Install
greenlang-cli install --offline
```

### 1.3 Initial Configuration

#### Configuration File Structure

```
/etc/greenlang/
  config.yaml           # Main configuration
  agents/               # Agent configurations
    furnace_zone_1.yaml
    furnace_zone_2.yaml
  integrations/         # Integration configurations
    opc_ua.yaml
    modbus.yaml
  security/             # Security configurations
    auth.yaml
    tls/
      server.crt
      server.key
```

#### Main Configuration (config.yaml)

```yaml
# GreenLang Configuration
version: "1.0"

# Server Settings
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  debug: false

# Database Configuration
database:
  type: postgresql
  host: localhost
  port: 5432
  name: greenlang
  user: greenlang
  password: ${DB_PASSWORD}  # From environment
  pool_size: 20
  max_overflow: 10
  ssl_mode: require

# Cache Configuration
cache:
  type: redis
  host: localhost
  port: 6379
  password: ${REDIS_PASSWORD}
  db: 0
  ssl: true

# Time-Series Database
timeseries:
  type: timescaledb
  host: localhost
  port: 5432
  name: greenlang_ts
  retention_days: 365
  chunk_interval: 1d

# ML Engine Configuration
ml_engine:
  enabled: true
  model_path: /var/lib/greenlang/models
  inference_workers: 4
  training_workers: 2
  gpu_enabled: true
  auto_retrain:
    enabled: true
    schedule: "0 2 * * 0"  # Weekly at 2 AM Sunday

# Logging Configuration
logging:
  level: INFO
  format: json
  output:
    - type: console
    - type: file
      path: /var/log/greenlang/greenlang.log
      max_size: 100MB
      max_files: 10
    - type: syslog
      host: syslog.local
      port: 514

# Monitoring Configuration
monitoring:
  metrics:
    enabled: true
    port: 9090
    path: /metrics
  tracing:
    enabled: true
    type: jaeger
    endpoint: http://jaeger:14268/api/traces
  health_check:
    enabled: true
    path: /health

# Security Configuration
security:
  auth:
    type: jwt
    issuer: greenlang
    audience: greenlang-api
    access_token_expire: 3600
    refresh_token_expire: 604800
  tls:
    enabled: true
    cert_file: /etc/greenlang/security/tls/server.crt
    key_file: /etc/greenlang/security/tls/server.key
    min_version: TLS1.2
  cors:
    enabled: true
    origins:
      - https://dashboard.yourcompany.com
```

#### Agent Configuration Example

```yaml
# agents/furnace_zone_1.yaml
name: furnace_zone_1
type: process_heat_agent
enabled: true

# Data Sources
data_sources:
  - name: temperature
    type: opc_ua
    node_id: "ns=2;s=Zone1.Temperature"
    polling_interval: 1000

  - name: fuel_flow
    type: modbus
    address: 40001
    type: holding_register
    scale: 0.01
    polling_interval: 1000

# Processing Configuration
processing:
  buffer_size: 1000
  aggregation_interval: 60s
  anomaly_detection:
    enabled: true
    model: isolation_forest
    threshold: 0.7

# Alarm Configuration
alarms:
  - name: high_temperature
    condition: "temperature > 900"
    priority: high
    message: "Zone 1 temperature exceeds 900C"

  - name: fuel_flow_deviation
    condition: "abs(fuel_flow - setpoint) > 10"
    priority: medium
    message: "Fuel flow deviation from setpoint"

# ML Configuration
ml:
  prediction:
    enabled: true
    model: lstm_temperature
    horizon: 3600  # 1 hour
    update_interval: 60

  optimization:
    enabled: true
    model: bayesian_optimizer
    constraints:
      temperature_min: 800
      temperature_max: 950

# Output Configuration
outputs:
  - type: dashboard
    update_interval: 1000

  - type: historian
    database: timescaledb
    retention: 365d

  - type: alert
    channels:
      - email
      - sms
      - dashboard
```

### 1.4 Integration Configuration

#### OPC-UA Integration

```yaml
# integrations/opc_ua.yaml
connections:
  - name: plc_main
    endpoint: opc.tcp://192.168.1.100:4840
    security_mode: SignAndEncrypt
    security_policy: Basic256Sha256
    certificate: /etc/greenlang/security/opc_ua/client.crt
    private_key: /etc/greenlang/security/opc_ua/client.key
    username: opcua_client
    password: ${OPCUA_PASSWORD}

    subscriptions:
      - node_id: "ns=2;s=Zone1"
        sampling_interval: 100
        publishing_interval: 1000
        queue_size: 10

      - node_id: "ns=2;s=Zone2"
        sampling_interval: 100
        publishing_interval: 1000
        queue_size: 10
```

#### Modbus Integration

```yaml
# integrations/modbus.yaml
connections:
  - name: furnace_plc
    type: tcp
    host: 192.168.1.101
    port: 502
    unit_id: 1
    timeout: 5s
    retry_count: 3

    registers:
      - name: zone1_temp
        address: 40001
        type: holding
        data_type: float32
        byte_order: big
        word_order: big
        scale: 1.0
        offset: 0.0

      - name: zone1_setpoint
        address: 40003
        type: holding
        data_type: float32
        byte_order: big
        word_order: big
```

---

## Part 2: User Management and Security

### 2.1 Authentication Configuration

#### JWT Configuration

```yaml
# security/auth.yaml
authentication:
  type: jwt
  issuer: greenlang
  audience: greenlang-api

  # Token Settings
  access_token:
    expire_seconds: 3600
    algorithm: RS256
    private_key_file: /etc/greenlang/security/jwt/private.pem
    public_key_file: /etc/greenlang/security/jwt/public.pem

  refresh_token:
    expire_seconds: 604800
    rotate: true

  # Password Policy
  password_policy:
    min_length: 12
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_special: true
    max_age_days: 90
    history_count: 12

  # Session Settings
  session:
    max_concurrent: 3
    idle_timeout: 1800
    absolute_timeout: 28800

  # MFA Configuration
  mfa:
    enabled: true
    methods:
      - totp
      - sms
    grace_period_seconds: 300

  # Lockout Policy
  lockout:
    enabled: true
    max_attempts: 5
    lockout_duration: 900
    reset_after: 3600
```

#### LDAP/Active Directory Integration

```yaml
# security/ldap.yaml
ldap:
  enabled: true
  type: active_directory

  connection:
    url: ldaps://ldap.yourcompany.com:636
    bind_dn: cn=greenlang,ou=services,dc=yourcompany,dc=com
    bind_password: ${LDAP_BIND_PASSWORD}
    base_dn: dc=yourcompany,dc=com

  user_search:
    base: ou=users,dc=yourcompany,dc=com
    filter: "(sAMAccountName={username})"
    attributes:
      username: sAMAccountName
      email: mail
      first_name: givenName
      last_name: sn

  group_search:
    base: ou=groups,dc=yourcompany,dc=com
    filter: "(member={user_dn})"

  group_mapping:
    "CN=GreenLang Admins,OU=Groups,DC=yourcompany,DC=com": admin
    "CN=GreenLang Operators,OU=Groups,DC=yourcompany,DC=com": operator
    "CN=GreenLang Viewers,OU=Groups,DC=yourcompany,DC=com": viewer
```

### 2.2 Role-Based Access Control (RBAC)

#### Default Roles

| Role | Description | Permissions |
|------|-------------|-------------|
| admin | Full system access | All permissions |
| operator | Operational access | Read, write, acknowledge alarms |
| engineer | Engineering access | Read, write, configure agents |
| viewer | Read-only access | Read only |
| auditor | Audit access | Read, export logs |

#### Role Definitions

```yaml
# security/roles.yaml
roles:
  - name: admin
    description: Full system administrator
    permissions:
      - "*"  # All permissions

  - name: operator
    description: Process operator
    permissions:
      - "dashboard:read"
      - "dashboard:write"
      - "alarms:read"
      - "alarms:acknowledge"
      - "agents:read"
      - "reports:read"
      - "reports:create"

  - name: engineer
    description: Process engineer
    permissions:
      - "dashboard:read"
      - "dashboard:write"
      - "alarms:*"
      - "agents:read"
      - "agents:write"
      - "agents:configure"
      - "ml:read"
      - "ml:train"
      - "reports:*"

  - name: viewer
    description: Read-only viewer
    permissions:
      - "dashboard:read"
      - "alarms:read"
      - "agents:read"
      - "reports:read"

  - name: auditor
    description: Compliance auditor
    permissions:
      - "dashboard:read"
      - "alarms:read"
      - "logs:read"
      - "logs:export"
      - "reports:read"
      - "compliance:read"
```

#### User Management Commands

```bash
# Create user
greenlang-cli user create \
  --username jsmith \
  --email jsmith@company.com \
  --role operator \
  --password-prompt

# List users
greenlang-cli user list

# Modify user role
greenlang-cli user modify jsmith --role engineer

# Disable user
greenlang-cli user disable jsmith

# Reset password
greenlang-cli user reset-password jsmith --password-prompt

# Enable MFA
greenlang-cli user enable-mfa jsmith --method totp
```

### 2.3 Network Security

#### Firewall Configuration

```bash
# Required ports
# 8000  - API (HTTPS)
# 8443  - Dashboard (HTTPS)
# 9090  - Metrics (Prometheus)
# 5432  - PostgreSQL (internal only)
# 6379  - Redis (internal only)

# UFW Example
ufw default deny incoming
ufw default allow outgoing
ufw allow from 10.0.0.0/8 to any port 8000
ufw allow from 10.0.0.0/8 to any port 8443
ufw allow from 10.0.1.0/24 to any port 9090  # Monitoring only
ufw enable
```

#### TLS Configuration

```yaml
# TLS settings for all services
tls:
  enabled: true
  min_version: TLS1.2
  max_version: TLS1.3

  cipher_suites:
    - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
    - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
    - TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
    - TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256

  certificate:
    cert_file: /etc/greenlang/security/tls/server.crt
    key_file: /etc/greenlang/security/tls/server.key
    ca_file: /etc/greenlang/security/tls/ca.crt

  client_auth:
    enabled: false  # Enable for mutual TLS
    ca_file: /etc/greenlang/security/tls/client-ca.crt
```

### 2.4 Audit Logging

#### Audit Configuration

```yaml
# security/audit.yaml
audit:
  enabled: true

  # What to audit
  events:
    - authentication
    - authorization
    - configuration_change
    - data_access
    - alarm_response
    - user_management

  # Audit output
  output:
    - type: database
      table: audit_log
      retention_days: 2555  # 7 years

    - type: syslog
      host: siem.yourcompany.com
      port: 514
      protocol: tcp
      format: cef

    - type: file
      path: /var/log/greenlang/audit.log
      format: json
      rotation:
        max_size: 100MB
        max_files: 100
        compress: true

  # Tamper protection
  integrity:
    enabled: true
    algorithm: sha256
    chain: true  # Blockchain-style chaining
```

#### Audit Log Format

```json
{
  "timestamp": "2025-12-07T14:32:15.123Z",
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "configuration_change",
  "severity": "info",
  "user": {
    "id": "user_123",
    "username": "jsmith",
    "role": "engineer",
    "ip_address": "10.0.1.50"
  },
  "action": {
    "type": "update",
    "resource": "agent",
    "resource_id": "furnace_zone_1",
    "changes": {
      "temperature_threshold": {
        "old": 900,
        "new": 920
      }
    }
  },
  "result": "success",
  "hash": "abc123...",
  "previous_hash": "xyz789..."
}
```

---

## Part 3: Backup and Recovery

### 3.1 Backup Strategy

#### Backup Components

| Component | Type | Frequency | Retention |
|-----------|------|-----------|-----------|
| Configuration | Full | Daily | 90 days |
| PostgreSQL | Full | Daily | 30 days |
| PostgreSQL | Incremental | Hourly | 7 days |
| TimescaleDB | Continuous | Real-time | Per policy |
| ML Models | Full | Weekly | 12 versions |
| Logs | Archive | Daily | 7 years |

#### Backup Configuration

```yaml
# backup/backup.yaml
backup:
  enabled: true

  # Backup destination
  destination:
    type: s3
    bucket: greenlang-backups
    prefix: production
    region: us-east-1
    encryption: AES256

  # Backup schedules
  schedules:
    - name: config_backup
      type: configuration
      schedule: "0 0 * * *"  # Daily at midnight
      retention_days: 90

    - name: database_full
      type: postgresql
      mode: full
      schedule: "0 1 * * *"  # Daily at 1 AM
      retention_days: 30

    - name: database_incremental
      type: postgresql
      mode: incremental
      schedule: "0 * * * *"  # Hourly
      retention_days: 7

    - name: timeseries_backup
      type: timescaledb
      mode: continuous
      schedule: "*/15 * * * *"  # Every 15 minutes

    - name: ml_models
      type: models
      schedule: "0 2 * * 0"  # Weekly Sunday at 2 AM
      retention_count: 12

  # Verification
  verification:
    enabled: true
    schedule: "0 6 * * *"  # Daily at 6 AM
    restore_test: weekly

  # Notifications
  notifications:
    success: false
    failure: true
    channels:
      - email:ops@company.com
      - slack:#greenlang-alerts
```

#### Backup Commands

```bash
# Manual full backup
greenlang-cli backup create --type full --name manual_backup

# Backup specific component
greenlang-cli backup create --type postgresql --name db_backup

# List backups
greenlang-cli backup list

# Verify backup integrity
greenlang-cli backup verify --name backup_20251207_010000

# Download backup
greenlang-cli backup download --name backup_20251207_010000 --output /tmp/
```

### 3.2 Disaster Recovery

#### Recovery Procedures

##### Procedure 1: Database Recovery

```bash
# Stop services
greenlang-cli service stop --all

# Restore database
greenlang-cli backup restore \
  --name backup_20251207_010000 \
  --type postgresql \
  --target-time "2025-12-07T10:00:00Z"  # Optional point-in-time

# Verify database
greenlang-cli db verify

# Start services
greenlang-cli service start --all

# Verify application
greenlang-cli health check
```

##### Procedure 2: Full System Recovery

```bash
# On new system, install GreenLang base
greenlang-cli install --minimal

# Configure backup access
greenlang-cli backup configure \
  --type s3 \
  --bucket greenlang-backups \
  --region us-east-1

# List available backups
greenlang-cli backup list --type full

# Restore full system
greenlang-cli backup restore \
  --name full_backup_20251207_010000 \
  --type full

# Restore configuration
greenlang-cli backup restore \
  --name config_backup_20251207_000000 \
  --type configuration

# Restore database
greenlang-cli backup restore \
  --name db_backup_20251207_010000 \
  --type postgresql

# Restore ML models
greenlang-cli backup restore \
  --name ml_backup_20251201_020000 \
  --type models

# Start and verify
greenlang-cli service start --all
greenlang-cli health check --detailed
```

##### Procedure 3: Configuration Recovery

```bash
# If only configuration is corrupted
greenlang-cli service stop --all

# Restore configuration
greenlang-cli backup restore \
  --name config_backup_20251207_000000 \
  --type configuration

# Validate configuration
greenlang-cli config validate

# Start services
greenlang-cli service start --all
```

#### Recovery Time Objectives

| Scenario | RTO | RPO | Procedure |
|----------|-----|-----|-----------|
| Database corruption | 1 hour | 1 hour | Database recovery |
| Server failure | 4 hours | 1 hour | Full system recovery |
| Configuration error | 30 min | 24 hours | Configuration recovery |
| Complete site loss | 24 hours | 1 hour | Disaster recovery |

### 3.3 High Availability

#### HA Architecture

```
                    +------------------+
                    |   Load Balancer  |
                    +--------+---------+
                             |
           +-----------------+------------------+
           |                 |                  |
    +------v------+   +------v------+    +------v------+
    |   API-1     |   |   API-2     |    |   API-3     |
    +------+------+   +------+------+    +------+------+
           |                 |                  |
           +-----------------+------------------+
                             |
                    +--------v---------+
                    |   Redis Cluster  |
                    +--------+---------+
                             |
           +-----------------+------------------+
           |                 |                  |
    +------v------+   +------v------+    +------v------+
    |  Agent-1    |   |  Agent-2    |    |  Agent-3    |
    +------+------+   +------+------+    +------+------+
           |                 |                  |
           +-----------------+------------------+
                             |
                    +--------v---------+
                    | PostgreSQL HA    |
                    | (Primary/Replica)|
                    +------------------+
```

#### HA Configuration

```yaml
# ha/haproxy.cfg
global
    maxconn 4096
    log stdout format raw local0

defaults
    mode http
    timeout connect 5s
    timeout client 50s
    timeout server 50s
    option httplog

frontend greenlang_frontend
    bind *:443 ssl crt /etc/haproxy/certs/greenlang.pem
    default_backend greenlang_api

backend greenlang_api
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    server api1 10.0.1.10:8000 check inter 5s fall 3 rise 2
    server api2 10.0.1.11:8000 check inter 5s fall 3 rise 2
    server api3 10.0.1.12:8000 check inter 5s fall 3 rise 2
```

---

## Part 4: Performance Tuning

### 4.1 System Monitoring

#### Key Metrics

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| CPU Usage | > 70% | > 90% | Scale horizontally |
| Memory Usage | > 75% | > 90% | Increase memory, tune GC |
| Disk Usage | > 70% | > 85% | Add storage, clean up |
| API Latency P95 | > 500ms | > 1s | Tune queries, add caching |
| Agent Processing Lag | > 10s | > 60s | Scale agents |
| ML Inference Time | > 100ms | > 500ms | Optimize models |

#### Monitoring Setup

```yaml
# monitoring/prometheus.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'greenlang-api'
    static_configs:
      - targets: ['api:9090']
    metrics_path: /metrics

  - job_name: 'greenlang-agents'
    static_configs:
      - targets: ['agent1:9090', 'agent2:9090']
    metrics_path: /metrics

  - job_name: 'greenlang-ml'
    static_configs:
      - targets: ['ml:9090']
    metrics_path: /metrics

  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - /etc/prometheus/rules/*.yaml
```

#### Alert Rules

```yaml
# monitoring/rules/greenlang.yaml
groups:
  - name: greenlang
    rules:
      - alert: HighCPUUsage
        expr: avg(rate(process_cpu_seconds_total{job="greenlang-api"}[5m])) > 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High CPU usage on GreenLang API

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes{job="greenlang-api"} / node_memory_MemTotal_bytes > 0.75
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High memory usage on GreenLang API

      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: API latency exceeds 500ms (P95)

      - alert: AgentProcessingLag
        expr: greenlang_agent_processing_lag_seconds > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: Agent processing lag exceeds 10 seconds

      - alert: MLInferenceSlowdown
        expr: greenlang_ml_inference_duration_seconds > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: ML inference time exceeds 100ms
```

### 4.2 Database Tuning

#### PostgreSQL Configuration

```ini
# postgresql.conf - Performance tuning

# Memory Settings
shared_buffers = 4GB                    # 25% of RAM
effective_cache_size = 12GB             # 75% of RAM
work_mem = 256MB                        # Per query
maintenance_work_mem = 1GB              # For VACUUM, CREATE INDEX

# Write Performance
wal_buffers = 64MB
checkpoint_completion_target = 0.9
checkpoint_timeout = 15min
max_wal_size = 4GB

# Query Performance
random_page_cost = 1.1                  # For SSD
effective_io_concurrency = 200          # For SSD
default_statistics_target = 100

# Connection Settings
max_connections = 200
superuser_reserved_connections = 5

# Parallelism
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4

# Logging
log_min_duration_statement = 1000       # Log queries > 1s
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
```

#### TimescaleDB Optimization

```sql
-- Optimize chunk interval based on data volume
SELECT set_chunk_time_interval('process_data', INTERVAL '1 day');

-- Enable compression for older data
ALTER TABLE process_data SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'agent_id',
  timescaledb.compress_orderby = 'time DESC'
);

-- Add compression policy
SELECT add_compression_policy('process_data', INTERVAL '7 days');

-- Add retention policy
SELECT add_retention_policy('process_data', INTERVAL '365 days');

-- Create continuous aggregates for reporting
CREATE MATERIALIZED VIEW process_data_hourly
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', time) AS bucket,
  agent_id,
  avg(temperature) AS avg_temperature,
  max(temperature) AS max_temperature,
  min(temperature) AS min_temperature,
  avg(energy_consumption) AS avg_energy
FROM process_data
GROUP BY bucket, agent_id;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('process_data_hourly',
  start_offset => INTERVAL '3 hours',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour');
```

### 4.3 Application Tuning

#### API Performance

```yaml
# config.yaml - API tuning
server:
  workers: 8                      # CPU cores
  worker_connections: 1000
  keepalive_timeout: 65

  # Request limits
  max_request_size: 10MB
  request_timeout: 60s

  # Connection pooling
  pool:
    size: 20
    max_overflow: 10
    pool_recycle: 3600

# Caching configuration
cache:
  enabled: true
  default_ttl: 300

  policies:
    - pattern: "/api/v1/dashboard/*"
      ttl: 5
    - pattern: "/api/v1/reports/*"
      ttl: 3600
    - pattern: "/api/v1/config/*"
      ttl: 300
```

#### Agent Performance

```yaml
# Agent tuning
agent:
  # Processing configuration
  processing:
    batch_size: 1000
    buffer_size: 10000
    flush_interval: 1s

  # Threading
  workers:
    data_collection: 4
    processing: 8
    ml_inference: 4

  # Memory management
  memory:
    max_heap: 4GB
    gc_threshold: 0.8

  # Performance optimizations
  optimizations:
    vectorized_operations: true
    caching:
      enabled: true
      size: 1000
      ttl: 60
```

### 4.4 ML Engine Tuning

```yaml
# ML Engine tuning
ml_engine:
  # Inference optimization
  inference:
    batch_size: 32
    num_threads: 4
    use_gpu: true
    mixed_precision: true

  # Model caching
  model_cache:
    enabled: true
    max_models: 10
    preload:
      - temperature_predictor
      - anomaly_detector

  # Training optimization
  training:
    num_workers: 8
    gpu_memory_fraction: 0.8
    early_stopping:
      enabled: true
      patience: 10

  # Auto-scaling
  auto_scaling:
    enabled: true
    min_replicas: 2
    max_replicas: 10
    target_latency: 50ms
```

---

## Summary

In this module, you learned:

1. **Installation:** Multiple deployment methods (Docker, Kubernetes, air-gapped)
2. **Configuration:** System, agent, and integration configuration
3. **Security:** Authentication, RBAC, network security, audit logging
4. **Backup/Recovery:** Backup strategies, disaster recovery, high availability
5. **Performance:** Monitoring, database tuning, application optimization

---

## Knowledge Check

### Questions

1. What are the minimum hardware requirements for a production GreenLang deployment?
2. Explain the difference between authentication and authorization in GreenLang.
3. What is the recommended backup strategy for a production environment?
4. How would you troubleshoot high API latency?
5. What PostgreSQL settings should be tuned for a time-series workload?

### Practical Exercises

1. **Installation:** Deploy GreenLang using Docker Compose
2. **Security:** Configure LDAP authentication and RBAC
3. **Backup:** Set up automated backups and perform a test restore
4. **Monitoring:** Configure Prometheus and create alert rules
5. **Tuning:** Optimize PostgreSQL for your workload

---

## Next Steps

After completing this module:

- **Deploy** GreenLang in a test environment
- **Configure** security according to your organization's policies
- **Set up** monitoring and alerting
- **Practice** disaster recovery procedures
- **Prepare** for administrator certification

---

*Module Version: 1.0.0*
*Last Updated: December 2025*
