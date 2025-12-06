# GreenLang Process Heat Platform - Administrator Guide

**Document Version:** 1.0.0
**Last Updated:** 2025-12-06
**Classification:** Administrative Documentation
**Intended Audience:** System Administrators, IT Operations, DevOps Engineers

---

## Table of Contents

1. [Installation and Deployment](#1-installation-and-deployment)
2. [Configuration Management](#2-configuration-management)
3. [User Management and RBAC](#3-user-management-and-rbac)
4. [Backup and Recovery](#4-backup-and-recovery)
5. [Performance Tuning](#5-performance-tuning)
6. [Security Hardening](#6-security-hardening)
7. [Upgrade Procedures](#7-upgrade-procedures)
8. [Troubleshooting Common Issues](#8-troubleshooting-common-issues)
9. [Monitoring and Observability](#9-monitoring-and-observability)
10. [Appendix: Configuration Reference](#appendix-configuration-reference)

---

## 1. Installation and Deployment

### 1.1 System Requirements

#### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| CPU | 8 cores (x86_64) |
| RAM | 32 GB |
| Storage | 500 GB SSD |
| Network | 1 Gbps |
| OS | Ubuntu 22.04 LTS, RHEL 8+, Windows Server 2022 |

#### Recommended Production Requirements

| Component | Requirement |
|-----------|-------------|
| CPU | 16+ cores (x86_64) |
| RAM | 64 GB |
| Storage | 1 TB NVMe SSD |
| Network | 10 Gbps |
| OS | Ubuntu 22.04 LTS or RHEL 9 |

#### Software Dependencies

- Python 3.11+
- PostgreSQL 15+ (or TimescaleDB)
- Redis 7.0+
- Docker 24.0+ (for containerized deployment)
- Kubernetes 1.28+ (for orchestrated deployment)

### 1.2 Deployment Options

#### Option A: Docker Compose (Development/Small Scale)

**Step 1:** Clone the repository

```bash
git clone https://github.com/greenlang/process-heat.git
cd process-heat
```

**Step 2:** Configure environment variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

**Step 3:** Start services

```bash
docker-compose up -d
```

**Step 4:** Verify deployment

```bash
docker-compose ps
curl http://localhost:8000/health
```

#### Option B: Kubernetes (Production)

**Step 1:** Add Helm repository

```bash
helm repo add greenlang https://charts.greenlang.io
helm repo update
```

**Step 2:** Create namespace

```bash
kubectl create namespace greenlang-process-heat
```

**Step 3:** Configure values

```yaml
# values.yaml
global:
  environment: production
  domain: processheat.yourcompany.com

orchestrator:
  replicas: 3
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"

agents:
  boilerOptimizer:
    enabled: true
    replicas: 2
  emissionsGuardian:
    enabled: true
    replicas: 2
  predictiveMaintenance:
    enabled: true
    replicas: 1

database:
  enabled: true
  persistence:
    size: 100Gi
    storageClass: fast-ssd

redis:
  enabled: true
  replicas: 3

ingress:
  enabled: true
  className: nginx
  tls:
    enabled: true
    secretName: processheat-tls
```

**Step 4:** Install

```bash
helm install process-heat greenlang/process-heat \
  --namespace greenlang-process-heat \
  --values values.yaml
```

**Step 5:** Verify deployment

```bash
kubectl get pods -n greenlang-process-heat
kubectl get svc -n greenlang-process-heat
```

#### Option C: On-Premises Installation

**Step 1:** Install system dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip postgresql-15 redis-server

# RHEL/CentOS
sudo dnf install -y python3.11 postgresql15-server redis
```

**Step 2:** Create application user

```bash
sudo useradd -r -m -s /bin/bash greenlang
sudo mkdir -p /opt/greenlang/process-heat
sudo chown greenlang:greenlang /opt/greenlang/process-heat
```

**Step 3:** Install application

```bash
sudo -u greenlang bash
cd /opt/greenlang/process-heat
python3.11 -m venv venv
source venv/bin/activate
pip install greenlang-process-heat
```

**Step 4:** Configure systemd services

```ini
# /etc/systemd/system/greenlang-orchestrator.service
[Unit]
Description=GreenLang Process Heat Orchestrator
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=greenlang
Group=greenlang
WorkingDirectory=/opt/greenlang/process-heat
Environment=GREENLANG_ENV=production
ExecStart=/opt/greenlang/process-heat/venv/bin/greenlang-orchestrator
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Step 5:** Start services

```bash
sudo systemctl daemon-reload
sudo systemctl enable greenlang-orchestrator
sudo systemctl start greenlang-orchestrator
```

### 1.3 Post-Installation Verification

Run the following checks after installation:

```bash
# Check service health
curl -s http://localhost:8000/api/v1/health | jq

# Expected output:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "components": {
#     "database": "healthy",
#     "redis": "healthy",
#     "orchestrator": "running",
#     "agents": 18
#   }
# }

# Check agent status
curl -s http://localhost:8000/api/v1/agents | jq

# Run diagnostic tests
greenlang-cli diagnose --all
```

---

## 2. Configuration Management

### 2.1 Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `config.yaml` | Main configuration | `/etc/greenlang/config.yaml` |
| `agents.yaml` | Agent configuration | `/etc/greenlang/agents.yaml` |
| `safety.yaml` | Safety parameters | `/etc/greenlang/safety.yaml` |
| `logging.yaml` | Logging configuration | `/etc/greenlang/logging.yaml` |
| `secrets.yaml` | Encrypted secrets | `/etc/greenlang/secrets.yaml` |

### 2.2 Main Configuration (config.yaml)

```yaml
# /etc/greenlang/config.yaml
environment: production
debug: false

# Server configuration
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 30

# Database configuration
database:
  type: postgresql
  host: localhost
  port: 5432
  name: greenlang_processheat
  user: greenlang
  # Password from secrets.yaml or environment variable
  pool_size: 20
  max_overflow: 10

# Redis configuration
redis:
  host: localhost
  port: 6379
  db: 0
  # Password from secrets.yaml

# Orchestrator configuration
orchestrator:
  id: GL-001-PROD
  name: ThermalCommand-Production
  heartbeat_interval_ms: 1000
  watchdog_timeout_ms: 5000
  max_concurrent_workflows: 100

# Integration configuration
integration:
  opcua:
    enabled: true
    server_url: opc.tcp://dcs-server:4840
    security_mode: SignAndEncrypt
  mqtt:
    enabled: true
    broker_url: mqtt://message-broker:1883
    topic_prefix: greenlang/processheat
  kafka:
    enabled: true
    bootstrap_servers: kafka-1:9092,kafka-2:9092
    topic_prefix: greenlang.processheat

# Metrics configuration
metrics:
  enabled: true
  prefix: greenlang_process_heat
  prometheus:
    enabled: true
    port: 9090
  push_gateway_url: http://prometheus-pushgateway:9091
```

### 2.3 Agent Configuration (agents.yaml)

```yaml
# /etc/greenlang/agents.yaml
defaults:
  safety_level: SIL_2
  heartbeat_interval_ms: 1000
  watchdog_timeout_ms: 5000
  max_consecutive_failures: 3
  provenance_tracking: true
  audit_enabled: true

agents:
  - id: GL-002-B001
    type: GL-002
    name: BoilerOptimizer-B001
    enabled: true
    config:
      boiler_id: B-001
      fuel_type: natural_gas
      rated_capacity_lb_hr: 100000
      guarantee_efficiency_pct: 82.0
      safety:
        sil_level: 2
        alarm_thresholds:
          high_pressure_psig: 150
          high_temperature_f: 700
          low_water_level_in: -2
      combustion:
        target_o2_pct: 2.5
        max_co_ppm: 100
        max_flue_temp_f: 450
      economizer:
        enabled: true
        design_effectiveness: 0.85

  - id: GL-010-PLANT
    type: GL-010
    name: EmissionsGuardian-Plant
    enabled: true
    config:
      source_id: PLANT-001
      permit_limits:
        co2_lb_hr: 5000
        nox_lb_hr: 0.5
        so2_lb_hr: 0.1
      reporting:
        interval_minutes: 15
        retention_days: 365
```

### 2.4 Safety Configuration (safety.yaml)

```yaml
# /etc/greenlang/safety.yaml
global:
  sil_level: SIL_2
  emergency_shutdown_enabled: true
  fail_safe_mode: safe_state  # Options: safe_state, last_known, shutdown

interlocks:
  - id: HIGH_PRESSURE
    description: High pressure emergency shutdown
    condition: steam_pressure_psig > 160
    action: emergency_shutdown
    delay_ms: 0

  - id: LOW_WATER
    description: Low water level shutdown
    condition: drum_level_in < -4
    action: emergency_shutdown
    delay_ms: 0

  - id: HIGH_TEMP_WARNING
    description: High temperature warning
    condition: flue_gas_temp_f > 500
    action: alarm
    severity: high
    delay_ms: 5000

alarm_thresholds:
  efficiency_warning_pct: 80
  efficiency_critical_pct: 75
  o2_high_warning_pct: 4.0
  o2_high_critical_pct: 5.0
  co_warning_ppm: 100
  co_critical_ppm: 200

esd_integration:
  type: modbus_tcp
  host: 192.168.1.100
  port: 502
  unit_id: 1
  coils:
    esd_trigger: 0
    esd_status: 1
    esd_reset: 2
```

### 2.5 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GREENLANG_ENV` | Environment (development/staging/production) | development |
| `GREENLANG_CONFIG_PATH` | Path to configuration directory | /etc/greenlang |
| `GREENLANG_LOG_LEVEL` | Logging level | INFO |
| `GREENLANG_DB_PASSWORD` | Database password | (required) |
| `GREENLANG_REDIS_PASSWORD` | Redis password | (optional) |
| `GREENLANG_SECRET_KEY` | Application secret key | (required) |
| `GREENLANG_OPCUA_USER` | OPC-UA username | (optional) |
| `GREENLANG_OPCUA_PASSWORD` | OPC-UA password | (optional) |

### 2.6 Configuration Validation

Validate configuration before applying:

```bash
greenlang-cli config validate --all

# Expected output:
# Validating config.yaml... OK
# Validating agents.yaml... OK
# Validating safety.yaml... OK
# Validating logging.yaml... OK
# All configurations valid.
```

---

## 3. User Management and RBAC

### 3.1 Role-Based Access Control Model

```
+------------------------------------------------------------------+
|                     RBAC Hierarchy                                 |
+------------------------------------------------------------------+
|                                                                    |
|  +------------+     +------------+     +------------+              |
|  | Super Admin|---->| Org Admin  |---->| Plant Admin|              |
|  +------------+     +------------+     +------------+              |
|                            |                  |                    |
|                            v                  v                    |
|                     +------------+     +------------+              |
|                     | Engineer   |     | Supervisor |              |
|                     +------------+     +------------+              |
|                            |                  |                    |
|                            v                  v                    |
|                     +------------+     +------------+              |
|                     | Technician |     | Operator   |              |
|                     +------------+     +------------+              |
|                                               |                    |
|                                               v                    |
|                                        +------------+              |
|                                        | Viewer     |              |
|                                        +------------+              |
+------------------------------------------------------------------+
```

### 3.2 Default Roles and Permissions

| Role | Description | Key Permissions |
|------|-------------|-----------------|
| **Super Admin** | System-wide administration | All permissions |
| **Org Admin** | Organization administration | User management, configuration |
| **Plant Admin** | Plant-level administration | Plant config, agent management |
| **Engineer** | Engineering functions | Full agent control, overrides |
| **Supervisor** | Shift supervision | Override approval, reporting |
| **Operator** | Day-to-day operations | Agent control, alarms |
| **Technician** | Technical support | Read-only, diagnostics |
| **Viewer** | Read-only access | Dashboard viewing only |

### 3.3 Managing Users

#### Creating a User (CLI)

```bash
greenlang-cli user create \
  --username "jsmith" \
  --email "jsmith@company.com" \
  --first-name "John" \
  --last-name "Smith" \
  --role "operator" \
  --plant "PLANT-001" \
  --send-invite
```

#### Creating a User (API)

```bash
curl -X POST "https://api.greenlang.io/v1/users" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jsmith",
    "email": "jsmith@company.com",
    "first_name": "John",
    "last_name": "Smith",
    "role": "operator",
    "plant_id": "PLANT-001"
  }'
```

#### Creating a User (Admin Console)

1. Navigate to **Administration > User Management**
2. Click **Create User**
3. Fill in user details:

```
+------------------------------------------+
|  CREATE NEW USER                          |
+------------------------------------------+
|  Username: [jsmith                    ]   |
|  Email: [jsmith@company.com           ]   |
|  First Name: [John                    ]   |
|  Last Name: [Smith                    ]   |
|                                          |
|  Role: [Operator              v]          |
|  Plant: [PLANT-001            v]          |
|                                          |
|  [ ] Send welcome email                   |
|  [ ] Require password change              |
|                                          |
|         [Cancel]  [Create User]          |
+------------------------------------------+
```

[Screenshot placeholder: User creation form]

### 3.4 Managing Roles

#### Viewing Role Permissions

```bash
greenlang-cli role show operator

# Output:
# Role: operator
# Description: Day-to-day operations
# Permissions:
#   - dashboard:read
#   - agents:read
#   - agents:start
#   - agents:stop
#   - alarms:read
#   - alarms:acknowledge
#   - overrides:create_temporary
#   - reports:read
#   - reports:create
```

#### Creating a Custom Role

```bash
greenlang-cli role create \
  --name "senior_operator" \
  --description "Senior operator with extended permissions" \
  --copy-from "operator" \
  --add-permission "overrides:create_extended" \
  --add-permission "agents:restart"
```

### 3.5 Authentication Integration

#### LDAP/Active Directory

```yaml
# /etc/greenlang/auth.yaml
authentication:
  type: ldap
  ldap:
    server_url: ldaps://ldap.company.com:636
    base_dn: dc=company,dc=com
    user_search_base: ou=users
    user_search_filter: (sAMAccountName={username})
    group_search_base: ou=groups
    bind_dn: cn=greenlang-svc,ou=service-accounts,dc=company,dc=com
    # bind_password from secrets

    role_mapping:
      cn=ProcessHeat-Admins: plant_admin
      cn=ProcessHeat-Engineers: engineer
      cn=ProcessHeat-Operators: operator
      cn=ProcessHeat-Viewers: viewer
```

#### SAML/SSO

```yaml
authentication:
  type: saml
  saml:
    idp_metadata_url: https://idp.company.com/metadata
    sp_entity_id: greenlang-processheat
    assertion_consumer_service_url: https://processheat.company.com/saml/acs
    name_id_format: emailAddress

    attribute_mapping:
      email: http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress
      first_name: http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname
      last_name: http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname
      groups: http://schemas.xmlsoap.org/claims/Group
```

### 3.6 API Key Management

```bash
# Create API key for integration
greenlang-cli apikey create \
  --name "DCS-Integration" \
  --role "integration" \
  --expires "2026-12-31" \
  --rate-limit 1000

# Output:
# API Key created:
# Key ID: ak_prod_abc123xyz
# Secret: glsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#
# IMPORTANT: Save this secret securely. It will not be shown again.
```

---

## 4. Backup and Recovery

### 4.1 Backup Strategy

| Data Type | Frequency | Retention | Method |
|-----------|-----------|-----------|--------|
| Configuration | Daily + on change | 90 days | File backup |
| Database | Hourly | 30 days | pg_dump |
| Provenance records | Daily | 7 years | Archive |
| Audit logs | Daily | 7 years | Archive |
| Time series data | Daily | 1 year | TimescaleDB |

### 4.2 Automated Backup Configuration

```yaml
# /etc/greenlang/backup.yaml
backup:
  enabled: true
  schedule: "0 */6 * * *"  # Every 6 hours

  targets:
    - type: database
      enabled: true
      retention_days: 30
      compression: true

    - type: configuration
      enabled: true
      retention_days: 90
      include:
        - /etc/greenlang/*.yaml
        - /etc/greenlang/certs/*
      exclude:
        - /etc/greenlang/secrets.yaml

    - type: provenance
      enabled: true
      retention_years: 7
      archive_after_days: 30

  storage:
    type: s3  # Options: local, s3, azure, gcs
    s3:
      bucket: greenlang-backups
      region: us-east-1
      prefix: process-heat/
      storage_class: STANDARD_IA

  encryption:
    enabled: true
    key_id: alias/greenlang-backup-key

  notifications:
    email: backups@company.com
    slack_webhook: https://hooks.slack.com/...
```

### 4.3 Manual Backup Procedures

#### Database Backup

```bash
# Full database backup
greenlang-cli backup database \
  --output /backup/db_$(date +%Y%m%d_%H%M%S).sql.gz \
  --compress

# Backup specific tables
greenlang-cli backup database \
  --tables provenance,audit_log \
  --output /backup/audit_$(date +%Y%m%d).sql.gz
```

#### Configuration Backup

```bash
# Backup all configuration
greenlang-cli backup config \
  --output /backup/config_$(date +%Y%m%d).tar.gz

# Include secrets (encrypted)
greenlang-cli backup config \
  --include-secrets \
  --encryption-key /path/to/backup.key \
  --output /backup/config_full_$(date +%Y%m%d).tar.gz.enc
```

### 4.4 Recovery Procedures

#### Database Recovery

**Step 1:** Stop all services

```bash
sudo systemctl stop greenlang-orchestrator
sudo systemctl stop greenlang-agents
```

**Step 2:** Restore database

```bash
# Drop and recreate database
sudo -u postgres psql -c "DROP DATABASE greenlang_processheat;"
sudo -u postgres psql -c "CREATE DATABASE greenlang_processheat OWNER greenlang;"

# Restore from backup
gunzip -c /backup/db_20251206_120000.sql.gz | \
  sudo -u postgres psql greenlang_processheat
```

**Step 3:** Verify data integrity

```bash
greenlang-cli database verify
```

**Step 4:** Restart services

```bash
sudo systemctl start greenlang-orchestrator
sudo systemctl start greenlang-agents
```

#### Configuration Recovery

```bash
# Restore configuration
greenlang-cli backup restore config \
  --input /backup/config_20251206.tar.gz \
  --target /etc/greenlang

# Validate restored configuration
greenlang-cli config validate --all

# Restart to apply
sudo systemctl restart greenlang-orchestrator
```

### 4.5 Disaster Recovery

#### Recovery Time Objectives

| Scenario | RTO | RPO |
|----------|-----|-----|
| Single agent failure | < 5 minutes | 0 |
| Orchestrator failure | < 15 minutes | 0 |
| Database failure | < 1 hour | 1 hour |
| Full system failure | < 4 hours | 6 hours |
| Site disaster | < 24 hours | 6 hours |

#### High Availability Setup

```yaml
# HA configuration
high_availability:
  enabled: true
  mode: active-passive  # or active-active

  database:
    type: postgresql_cluster
    primary: db-primary.company.com
    replicas:
      - db-replica-1.company.com
      - db-replica-2.company.com
    failover_timeout_s: 30

  redis:
    type: sentinel
    sentinels:
      - redis-sentinel-1:26379
      - redis-sentinel-2:26379
      - redis-sentinel-3:26379
    master_name: greenlang-master

  orchestrator:
    replicas: 3
    leader_election: true
    lease_duration_s: 15
```

---

## 5. Performance Tuning

### 5.1 Database Optimization

#### PostgreSQL Configuration

```ini
# /etc/postgresql/15/main/postgresql.conf

# Memory
shared_buffers = 8GB              # 25% of RAM
effective_cache_size = 24GB       # 75% of RAM
work_mem = 256MB
maintenance_work_mem = 2GB

# Connections
max_connections = 200

# Write Performance
wal_buffers = 64MB
checkpoint_completion_target = 0.9
checkpoint_timeout = 10min

# Query Performance
random_page_cost = 1.1            # SSD storage
effective_io_concurrency = 200    # SSD storage

# Parallel Queries
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
```

#### Index Optimization

```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_provenance_timestamp
ON provenance_records (timestamp DESC);

CREATE INDEX CONCURRENTLY idx_provenance_agent
ON provenance_records (agent_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_audit_category_time
ON audit_log (category, timestamp DESC);

-- Analyze tables
ANALYZE provenance_records;
ANALYZE audit_log;
```

### 5.2 Application Tuning

```yaml
# Performance configuration
performance:
  # Worker processes
  workers: 8                    # CPU cores
  threads_per_worker: 4

  # Connection pools
  database_pool_size: 20
  database_max_overflow: 10
  redis_pool_size: 50

  # Caching
  cache:
    enabled: true
    default_ttl_s: 300
    max_size_mb: 512

  # Batch processing
  batch:
    calculation_batch_size: 100
    provenance_flush_interval_s: 10
    audit_flush_interval_s: 5

  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 1000
    burst_size: 100
```

### 5.3 Agent Performance

```yaml
# Agent performance tuning
agents:
  defaults:
    # Processing
    max_concurrent_tasks: 10
    task_timeout_s: 30
    task_queue_size: 100

    # Memory
    max_memory_mb: 2048
    gc_interval_s: 60

    # Metrics collection
    metrics_interval_s: 15
    metrics_buffer_size: 1000
```

### 5.4 Monitoring Performance

Key metrics to monitor:

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| API latency (p95) | < 100ms | > 200ms | > 500ms |
| Agent processing time | < 1s | > 2s | > 5s |
| Database query time | < 50ms | > 100ms | > 500ms |
| Memory usage | < 70% | > 80% | > 90% |
| CPU usage | < 60% | > 75% | > 90% |
| Error rate | < 0.1% | > 1% | > 5% |

---

## 6. Security Hardening

### 6.1 Network Security

#### Firewall Configuration

```bash
# Required ports
# Inbound
- 443/tcp    # HTTPS API
- 8443/tcp   # Admin console
- 9090/tcp   # Prometheus metrics (internal only)

# Outbound
- 5432/tcp   # PostgreSQL
- 6379/tcp   # Redis
- 4840/tcp   # OPC-UA
- 1883/tcp   # MQTT
- 9092/tcp   # Kafka

# Example iptables rules
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 8443 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 9090 -s 10.0.0.0/8 -j ACCEPT
```

#### TLS Configuration

```yaml
# /etc/greenlang/tls.yaml
tls:
  enabled: true
  min_version: TLSv1.2

  certificates:
    server:
      cert: /etc/greenlang/certs/server.crt
      key: /etc/greenlang/certs/server.key
      chain: /etc/greenlang/certs/chain.crt

    client_ca: /etc/greenlang/certs/ca.crt

  cipher_suites:
    - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
    - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
    - TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384

  mutual_tls:
    enabled: true
    required_for:
      - api
      - opcua
```

### 6.2 Secrets Management

#### Using HashiCorp Vault

```yaml
# /etc/greenlang/vault.yaml
secrets:
  provider: vault
  vault:
    address: https://vault.company.com:8200
    auth_method: kubernetes  # or token, approle
    role: greenlang-processheat
    secret_path: secret/data/greenlang/processheat

    # Secrets mapping
    secrets:
      database_password: db_password
      redis_password: redis_password
      api_secret_key: api_secret
      opcua_password: opcua_password
```

#### Encrypting Secrets at Rest

```bash
# Encrypt secrets file
greenlang-cli secrets encrypt \
  --input secrets.yaml \
  --output secrets.yaml.enc \
  --key-id alias/greenlang-secrets-key

# Decrypt for viewing (requires access)
greenlang-cli secrets decrypt \
  --input secrets.yaml.enc \
  --key-id alias/greenlang-secrets-key
```

### 6.3 Audit Logging

```yaml
# /etc/greenlang/audit.yaml
audit:
  enabled: true

  # What to audit
  events:
    authentication: true
    authorization: true
    configuration_changes: true
    agent_control: true
    safety_events: true
    data_access: true

  # Where to send
  destinations:
    - type: database
      enabled: true
      retention_days: 365

    - type: syslog
      enabled: true
      facility: local0
      host: syslog.company.com
      port: 514

    - type: splunk
      enabled: true
      hec_url: https://splunk.company.com:8088/services/collector
      # hec_token from secrets
      index: greenlang_audit

  # Tamper protection
  integrity:
    enabled: true
    algorithm: SHA-256
    chain_validation: true
```

### 6.4 Security Checklist

- [ ] TLS 1.2+ enabled for all connections
- [ ] Mutual TLS for OPC-UA and inter-service communication
- [ ] Secrets stored in vault or encrypted at rest
- [ ] Database connections encrypted
- [ ] API authentication required for all endpoints
- [ ] Rate limiting enabled
- [ ] Audit logging enabled and tamper-proof
- [ ] Firewall rules configured
- [ ] Security headers configured (HSTS, CSP, etc.)
- [ ] Regular security scanning scheduled
- [ ] Vulnerability patching process in place

---

## 7. Upgrade Procedures

### 7.1 Version Compatibility

| From Version | To Version | Upgrade Path | Notes |
|--------------|------------|--------------|-------|
| 1.0.x | 1.0.y | Direct upgrade | Patch release |
| 1.0.x | 1.1.x | Direct upgrade | Minor release |
| 1.x.x | 2.0.x | Migration required | Major release |

### 7.2 Pre-Upgrade Checklist

- [ ] Review release notes for breaking changes
- [ ] Backup database and configuration
- [ ] Test upgrade in staging environment
- [ ] Schedule maintenance window
- [ ] Notify operators of planned downtime
- [ ] Verify rollback procedure
- [ ] Ensure sufficient disk space

### 7.3 Upgrade Procedure

#### Docker Compose

```bash
# Pull new images
docker-compose pull

# Backup current state
docker-compose exec db pg_dump -U greenlang > backup.sql

# Stop services
docker-compose down

# Update images and start
docker-compose up -d

# Run migrations
docker-compose exec orchestrator greenlang-cli migrate

# Verify
docker-compose exec orchestrator greenlang-cli health
```

#### Kubernetes (Helm)

```bash
# Update Helm repo
helm repo update

# View changes
helm diff upgrade process-heat greenlang/process-heat \
  --namespace greenlang-process-heat \
  --values values.yaml

# Backup database
kubectl exec -n greenlang-process-heat db-0 -- pg_dump -U greenlang > backup.sql

# Perform upgrade
helm upgrade process-heat greenlang/process-heat \
  --namespace greenlang-process-heat \
  --values values.yaml \
  --wait \
  --timeout 10m

# Monitor rollout
kubectl rollout status deployment/orchestrator -n greenlang-process-heat

# Verify
kubectl exec -n greenlang-process-heat deployment/orchestrator -- greenlang-cli health
```

#### On-Premises

```bash
# Stop services
sudo systemctl stop greenlang-orchestrator
sudo systemctl stop greenlang-agents

# Backup
sudo -u greenlang pg_dump greenlang_processheat > /backup/pre-upgrade.sql
sudo tar -czf /backup/config-pre-upgrade.tar.gz /etc/greenlang

# Upgrade package
sudo -u greenlang bash
source /opt/greenlang/process-heat/venv/bin/activate
pip install --upgrade greenlang-process-heat

# Run migrations
greenlang-cli migrate

# Start services
sudo systemctl start greenlang-orchestrator
sudo systemctl start greenlang-agents

# Verify
greenlang-cli health
```

### 7.4 Rollback Procedure

```bash
# Stop services
sudo systemctl stop greenlang-orchestrator

# Restore database
sudo -u postgres psql -c "DROP DATABASE greenlang_processheat;"
sudo -u postgres psql -c "CREATE DATABASE greenlang_processheat OWNER greenlang;"
sudo -u postgres psql greenlang_processheat < /backup/pre-upgrade.sql

# Restore configuration
sudo tar -xzf /backup/config-pre-upgrade.tar.gz -C /

# Downgrade package
sudo -u greenlang bash
source /opt/greenlang/process-heat/venv/bin/activate
pip install greenlang-process-heat==1.0.0  # Previous version

# Start services
sudo systemctl start greenlang-orchestrator
```

---

## 8. Troubleshooting Common Issues

### 8.1 Service Won't Start

**Symptoms:** Service fails to start, shows "failed" status

**Diagnosis:**

```bash
# Check service status
sudo systemctl status greenlang-orchestrator

# View logs
sudo journalctl -u greenlang-orchestrator -n 100 --no-pager

# Check configuration
greenlang-cli config validate
```

**Common Causes:**

| Issue | Solution |
|-------|----------|
| Invalid configuration | Run `greenlang-cli config validate` and fix errors |
| Database connection failed | Verify database is running and credentials are correct |
| Port already in use | Check for conflicting services on port 8000 |
| Permission denied | Ensure greenlang user owns /opt/greenlang directory |
| Missing dependencies | Run `pip install -r requirements.txt` |

### 8.2 Agent Not Responding

**Symptoms:** Agent shows as UNHEALTHY or OFFLINE

**Diagnosis:**

```bash
# Check agent status
greenlang-cli agent status GL-002-B001

# View agent logs
greenlang-cli agent logs GL-002-B001 --tail 100

# Check heartbeat
greenlang-cli agent heartbeat GL-002-B001
```

**Common Causes:**

| Issue | Solution |
|-------|----------|
| Network connectivity | Verify network path to orchestrator |
| Resource exhaustion | Check memory/CPU usage |
| Configuration error | Validate agent configuration |
| Dependency service down | Check OPC-UA, database connections |

### 8.3 High Memory Usage

**Symptoms:** Memory usage > 80%, OOM errors

**Diagnosis:**

```bash
# Check memory usage
free -h
greenlang-cli metrics memory

# Check for memory leaks
greenlang-cli diagnose memory --agent GL-002-B001
```

**Solutions:**

1. Increase available memory
2. Reduce batch sizes in configuration
3. Reduce number of concurrent agents
4. Check for memory leaks and report to support
5. Restart affected agents

### 8.4 Database Performance Issues

**Symptoms:** Slow queries, timeouts, high latency

**Diagnosis:**

```bash
# Check slow queries
greenlang-cli diagnose database --slow-queries

# Check table sizes
greenlang-cli diagnose database --table-sizes

# Check connection pool
greenlang-cli diagnose database --connections
```

**Solutions:**

1. Run `VACUUM ANALYZE` on affected tables
2. Add missing indexes
3. Increase connection pool size
4. Archive old data
5. Upgrade database resources

### 8.5 Communication Failures

**Symptoms:** OPC-UA/MQTT/Kafka connection errors

**Diagnosis:**

```bash
# Test OPC-UA connection
greenlang-cli diagnose opcua --server opc.tcp://dcs:4840

# Test MQTT connection
greenlang-cli diagnose mqtt --broker mqtt://broker:1883

# Test Kafka connection
greenlang-cli diagnose kafka --bootstrap-servers kafka:9092
```

**Common Causes:**

| Issue | Solution |
|-------|----------|
| Network blocked | Check firewall rules |
| Certificate expired | Renew TLS certificates |
| Authentication failed | Verify credentials |
| Server overloaded | Check server resources |

---

## 9. Monitoring and Observability

### 9.1 Prometheus Metrics

**Key Metrics:**

```
# Agent metrics
greenlang_process_heat_agent_processing_duration_seconds
greenlang_process_heat_agent_processing_total
greenlang_process_heat_agent_errors_total
greenlang_process_heat_agent_health_score

# Workflow metrics
greenlang_process_heat_workflow_duration_seconds
greenlang_process_heat_workflow_total
greenlang_process_heat_workflow_failed_total

# System metrics
greenlang_process_heat_active_agents
greenlang_process_heat_active_workflows
greenlang_process_heat_api_requests_total
greenlang_process_heat_api_latency_seconds
```

### 9.2 Grafana Dashboards

Import pre-built dashboards:

1. Navigate to Grafana > Dashboards > Import
2. Enter dashboard ID: `18745` (GreenLang Process Heat Overview)
3. Select Prometheus data source
4. Click Import

Available dashboards:
- **18745**: System Overview
- **18746**: Agent Performance
- **18747**: Workflow Analysis
- **18748**: Safety Monitoring
- **18749**: Emissions Tracking

### 9.3 Alert Configuration

```yaml
# /etc/greenlang/alerts.yaml
alerts:
  - name: AgentUnhealthy
    expression: greenlang_process_heat_agent_health_score < 0.8
    duration: 5m
    severity: warning
    annotations:
      summary: "Agent {{ $labels.agent_id }} is unhealthy"

  - name: HighErrorRate
    expression: rate(greenlang_process_heat_agent_errors_total[5m]) > 0.1
    duration: 2m
    severity: critical
    annotations:
      summary: "High error rate on agent {{ $labels.agent_id }}"

  - name: DatabaseConnectionHigh
    expression: greenlang_process_heat_db_connections_active > 150
    duration: 5m
    severity: warning
    annotations:
      summary: "Database connection pool near capacity"
```

### 9.4 Log Management

```yaml
# /etc/greenlang/logging.yaml
logging:
  level: INFO
  format: json

  handlers:
    - type: console
      level: INFO

    - type: file
      level: DEBUG
      path: /var/log/greenlang/process-heat.log
      max_size_mb: 100
      backup_count: 10

    - type: elasticsearch
      level: INFO
      hosts:
        - https://elasticsearch:9200
      index_prefix: greenlang-processheat

  loggers:
    greenlang.agents: DEBUG
    greenlang.safety: INFO
    greenlang.api: INFO
```

---

## Appendix: Configuration Reference

### Environment Variables Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GREENLANG_ENV` | string | development | Environment name |
| `GREENLANG_DEBUG` | bool | false | Enable debug mode |
| `GREENLANG_CONFIG_PATH` | path | /etc/greenlang | Configuration directory |
| `GREENLANG_LOG_LEVEL` | string | INFO | Log level |
| `GREENLANG_LOG_FORMAT` | string | json | Log format (json/text) |
| `GREENLANG_DB_HOST` | string | localhost | Database host |
| `GREENLANG_DB_PORT` | int | 5432 | Database port |
| `GREENLANG_DB_NAME` | string | greenlang | Database name |
| `GREENLANG_DB_USER` | string | greenlang | Database user |
| `GREENLANG_DB_PASSWORD` | string | - | Database password |
| `GREENLANG_DB_SSL_MODE` | string | prefer | Database SSL mode |
| `GREENLANG_REDIS_HOST` | string | localhost | Redis host |
| `GREENLANG_REDIS_PORT` | int | 6379 | Redis port |
| `GREENLANG_REDIS_PASSWORD` | string | - | Redis password |
| `GREENLANG_SECRET_KEY` | string | - | Application secret key |
| `GREENLANG_API_PORT` | int | 8000 | API server port |
| `GREENLANG_METRICS_PORT` | int | 9090 | Metrics server port |

### CLI Command Reference

```bash
# General
greenlang-cli --help
greenlang-cli version
greenlang-cli health

# Configuration
greenlang-cli config validate [--all]
greenlang-cli config show [--section SECTION]

# User management
greenlang-cli user list
greenlang-cli user create --username USER --email EMAIL --role ROLE
greenlang-cli user delete --username USER

# Agent management
greenlang-cli agent list
greenlang-cli agent status AGENT_ID
greenlang-cli agent start AGENT_ID
greenlang-cli agent stop AGENT_ID
greenlang-cli agent restart AGENT_ID
greenlang-cli agent logs AGENT_ID [--tail N]

# Database
greenlang-cli migrate
greenlang-cli migrate --rollback
greenlang-cli database verify

# Backup
greenlang-cli backup database --output FILE
greenlang-cli backup config --output FILE
greenlang-cli backup restore --type TYPE --input FILE

# Diagnostics
greenlang-cli diagnose --all
greenlang-cli diagnose database
greenlang-cli diagnose network
greenlang-cli diagnose memory
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-06 | GL-TechWriter | Initial release |

---

*For technical support, contact support@greenlang.io or visit https://docs.greenlang.io*
