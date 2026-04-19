# Shared Infrastructure Setup Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-08

---

## Overview

This guide provides step-by-step instructions for setting up the shared infrastructure components required by all three GreenLang applications (CBAM, CSRD, VCCI).

### Shared Components

1. PostgreSQL Cluster (Primary + Replicas)
2. Redis Cluster (Cache + Queue)
3. Weaviate Vector Database
4. Message Queue (RabbitMQ)
5. Object Storage (S3 / Azure Blob / GCS)
6. Monitoring Stack (Prometheus, Grafana, Loki)
7. API Gateway (NGINX)

---

## 1. PostgreSQL Cluster Setup

### 1.1 AWS RDS PostgreSQL (Recommended for Production)

```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier greenlang-postgres \
  --db-instance-class db.m5.xlarge \
  --engine postgres \
  --engine-version 15.4 \
  --master-username postgres \
  --master-user-password <your-password> \
  --allocated-storage 500 \
  --storage-type gp3 \
  --storage-encrypted \
  --multi-az \
  --vpc-security-group-ids sg-xxxxxxxxx \
  --db-subnet-group-name greenlang-db-subnet \
  --backup-retention-period 30 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "sun:04:00-sun:05:00" \
  --enable-performance-insights \
  --publicly-accessible false \
  --tags Key=Project,Value=GreenLang Key=Environment,Value=production

# Create read replicas
aws rds create-db-instance-read-replica \
  --db-instance-identifier greenlang-postgres-replica-1 \
  --source-db-instance-identifier greenlang-postgres \
  --db-instance-class db.m5.large \
  --publicly-accessible false

aws rds create-db-instance-read-replica \
  --db-instance-identifier greenlang-postgres-replica-2 \
  --source-db-instance-identifier greenlang-postgres \
  --db-instance-class db.m5.large \
  --publicly-accessible false
```

### 1.2 Self-Hosted PostgreSQL (Docker)

```yaml
# docker-compose-postgres.yml
version: '3.8'

services:
  postgres-primary:
    image: postgres:15-alpine
    container_name: greenlang-postgres-primary
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres-replica-1:
    image: postgres:15-alpine
    container_name: greenlang-postgres-replica-1
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_MASTER_HOST: postgres-primary
      POSTGRES_MASTER_PORT: 5432
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
    volumes:
      - postgres_replica1_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    depends_on:
      - postgres-primary

volumes:
  postgres_primary_data:
  postgres_replica1_data:
```

### 1.3 Create Databases

```bash
# Connect to PostgreSQL
psql -h greenlang-postgres.xxxxxx.us-east-1.rds.amazonaws.com -U postgres

# Create databases
CREATE DATABASE cbam_db;
CREATE DATABASE csrd_db;
CREATE DATABASE vcci_db;
CREATE DATABASE shared_db;

# Create users
CREATE USER cbam_user WITH PASSWORD 'secure_password_1';
CREATE USER csrd_user WITH PASSWORD 'secure_password_2';
CREATE USER vcci_user WITH PASSWORD 'secure_password_3';
CREATE USER shared_user WITH PASSWORD 'secure_password_4';

# Grant permissions
GRANT ALL PRIVILEGES ON DATABASE cbam_db TO cbam_user;
GRANT ALL PRIVILEGES ON DATABASE csrd_db TO csrd_user;
GRANT ALL PRIVILEGES ON DATABASE vcci_db TO vcci_user;
GRANT ALL PRIVILEGES ON DATABASE shared_db TO shared_user;

# Grant read access to replicas
GRANT SELECT ON ALL TABLES IN SCHEMA public TO replica_user;
```

### 1.4 PostgreSQL Configuration (postgresql.conf)

```ini
# postgresql.conf (for production)

# Connection Settings
max_connections = 200
superuser_reserved_connections = 3

# Memory Settings
shared_buffers = 8GB                    # 25% of RAM
effective_cache_size = 24GB             # 75% of RAM
maintenance_work_mem = 2GB
work_mem = 64MB

# WAL Settings (for replication & PITR)
wal_level = replica
archive_mode = on
archive_command = 'aws s3 cp %p s3://greenlang-backups/wal/%f'
max_wal_senders = 10
wal_keep_size = 1GB

# Replication Settings
hot_standby = on
hot_standby_feedback = on

# Query Tuning
random_page_cost = 1.1                  # For SSD storage
effective_io_concurrency = 200

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d.log'
log_statement = 'ddl'
log_duration = on
log_line_prefix = '%t [%p]: user=%u,db=%d,app=%a,client=%h '
log_min_duration_statement = 1000       # Log queries > 1 second
```

---

## 2. Redis Cluster Setup

### 2.1 AWS ElastiCache Redis (Recommended)

```bash
# Create Redis cluster (cluster mode enabled)
aws elasticache create-replication-group \
  --replication-group-id greenlang-redis \
  --replication-group-description "GreenLang Redis Cluster" \
  --engine redis \
  --engine-version 7.0 \
  --cache-node-type cache.m5.large \
  --num-node-groups 3 \
  --replicas-per-node-group 1 \
  --cache-parameter-group-name default.redis7.cluster.on \
  --cache-subnet-group-name greenlang-redis-subnet \
  --security-group-ids sg-xxxxxxxxx \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --auth-token <your-auth-token> \
  --snapshot-retention-limit 7 \
  --snapshot-window "03:00-05:00" \
  --auto-minor-version-upgrade \
  --multi-az-enabled \
  --tags Key=Project,Value=GreenLang Key=Environment,Value=production
```

### 2.2 Self-Hosted Redis Cluster (Docker)

```yaml
# docker-compose-redis.yml
version: '3.8'

services:
  redis-node-1:
    image: redis:7-alpine
    container_name: redis-node-1
    command: >
      redis-server
      --port 6379
      --cluster-enabled yes
      --cluster-config-file nodes.conf
      --cluster-node-timeout 5000
      --appendonly yes
      --appendfilename "appendonly.aof"
      --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_node1_data:/data
    ports:
      - "6379:6379"

  redis-node-2:
    image: redis:7-alpine
    container_name: redis-node-2
    command: >
      redis-server
      --port 6380
      --cluster-enabled yes
      --cluster-config-file nodes.conf
      --cluster-node-timeout 5000
      --appendonly yes
      --appendfilename "appendonly.aof"
      --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_node2_data:/data
    ports:
      - "6380:6380"

  redis-node-3:
    image: redis:7-alpine
    container_name: redis-node-3
    command: >
      redis-server
      --port 6381
      --cluster-enabled yes
      --cluster-config-file nodes.conf
      --cluster-node-timeout 5000
      --appendonly yes
      --appendfilename "appendonly.aof"
      --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_node3_data:/data
    ports:
      - "6381:6381"

volumes:
  redis_node1_data:
  redis_node2_data:
  redis_node3_data:
```

```bash
# Initialize Redis cluster
docker exec -it redis-node-1 redis-cli \
  --cluster create \
  127.0.0.1:6379 127.0.0.1:6380 127.0.0.1:6381 \
  --cluster-replicas 0 \
  --cluster-yes
```

### 2.3 Redis Database Allocation

```
DB 0: API response cache (all apps)
DB 1: Celery broker (CSRD, VCCI)
DB 2: Celery results (CSRD, VCCI)
DB 3: Session store (all apps)
DB 4: Rate limiting (all apps)
DB 5: LLM response cache (CSRD, VCCI)
```

---

## 3. Weaviate Vector Database

### 3.1 Deploy Weaviate

```yaml
# docker-compose-weaviate.yml
version: '3.8'

services:
  weaviate:
    image: semitechnologies/weaviate:1.23.0
    container_name: greenlang-weaviate
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'false'
      AUTHENTICATION_APIKEY_ENABLED: 'true'
      AUTHENTICATION_APIKEY_ALLOWED_KEYS: '${WEAVIATE_API_KEY}'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: 'text2vec-openai,text2vec-huggingface'
      CLUSTER_HOSTNAME: 'weaviate-node-1'
      CLUSTER_GOSSIP_BIND_PORT: '7100'
      CLUSTER_DATA_BIND_PORT: '7101'
    volumes:
      - weaviate_data:/var/lib/weaviate
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/.well-known/ready"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  weaviate_data:
```

### 3.2 Initialize Weaviate Schemas

```python
# scripts/init-weaviate-schema.py
import weaviate
import os

client = weaviate.Client(
    url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
    auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
)

# Entity schema (for Entity MDM)
entity_schema = {
    "class": "Entity",
    "description": "Company/supplier entity for resolution",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "vectorizeClassName": False
        }
    },
    "properties": [
        {"name": "name", "dataType": ["text"], "description": "Entity name"},
        {"name": "legal_name", "dataType": ["text"]},
        {"name": "country", "dataType": ["string"]},
        {"name": "lei", "dataType": ["string"]},
        {"name": "duns", "dataType": ["string"]},
        {"name": "address", "dataType": ["text"]},
        {"name": "industry", "dataType": ["string"]},
        {"name": "metadata", "dataType": ["object"]}
    ]
}

# Product schema (for PCF data)
product_schema = {
    "class": "Product",
    "description": "Product carbon footprints",
    "vectorizer": "text2vec-openai",
    "properties": [
        {"name": "name", "dataType": ["text"]},
        {"name": "description", "dataType": ["text"]},
        {"name": "pcf_value", "dataType": ["number"]},
        {"name": "pcf_unit", "dataType": ["string"]},
        {"name": "supplier_id", "dataType": ["string"]},
        {"name": "vintage", "dataType": ["int"]},
        {"name": "methodology", "dataType": ["string"]}
    ]
}

# Create schemas
client.schema.create_class(entity_schema)
client.schema.create_class(product_schema)
print("Weaviate schemas created successfully")
```

---

## 4. Message Queue (RabbitMQ)

### 4.1 Deploy RabbitMQ

```yaml
# docker-compose-rabbitmq.yml
version: '3.8'

services:
  rabbitmq:
    image: rabbitmq:3.12-management-alpine
    container_name: greenlang-rabbitmq
    hostname: rabbitmq
    ports:
      - "5672:5672"     # AMQP port
      - "15672:15672"   # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: ${RABBITMQ_USER}
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
      RABBITMQ_DEFAULT_VHOST: greenlang
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
      - ./rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  rabbitmq_data:
```

### 4.2 Configure Exchanges and Queues

```bash
# After RabbitMQ is running
docker exec rabbitmq rabbitmqadmin declare exchange \
  name=greenlang.emissions.calculated \
  type=topic \
  durable=true

docker exec rabbitmq rabbitmqadmin declare exchange \
  name=greenlang.report.generated \
  type=topic \
  durable=true

docker exec rabbitmq rabbitmqadmin declare exchange \
  name=greenlang.cbam.submitted \
  type=topic \
  durable=true

# Create queues
docker exec rabbitmq rabbitmqadmin declare queue \
  name=csrd.emissions.sync \
  durable=true

docker exec rabbitmq rabbitmqadmin declare queue \
  name=vcci.pcf.exchange \
  durable=true

# Bind queues to exchanges
docker exec rabbitmq rabbitmqadmin declare binding \
  source=greenlang.emissions.calculated \
  destination=csrd.emissions.sync \
  routing_key="vcci.scope3.#"
```

---

## 5. Object Storage

### 5.1 AWS S3

```bash
# Create S3 buckets
aws s3 mb s3://greenlang-data --region us-east-1
aws s3 mb s3://greenlang-reports --region us-east-1
aws s3 mb s3://greenlang-backups --region us-east-1
aws s3 mb s3://greenlang-logs --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket greenlang-backups \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket greenlang-data \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Set lifecycle policies
aws s3api put-bucket-lifecycle-configuration \
  --bucket greenlang-logs \
  --lifecycle-configuration file://lifecycle-policy.json
```

```json
// lifecycle-policy.json
{
  "Rules": [
    {
      "Id": "Delete old logs",
      "Status": "Enabled",
      "Expiration": {
        "Days": 90
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 60,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

### 5.2 Azure Blob Storage

```bash
# Create storage account
az storage account create \
  --name greenlangdata \
  --resource-group greenlang-rg \
  --location eastus \
  --sku Standard_LRS \
  --encryption-services blob

# Create containers
az storage container create --name data --account-name greenlangdata
az storage container create --name reports --account-name greenlangdata
az storage container create --name backups --account-name greenlangdata
az storage container create --name logs --account-name greenlangdata
```

---

## 6. Monitoring Stack

### 6.1 Deploy Monitoring Stack

```yaml
# docker-compose-monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: greenlang-prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: greenlang-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=https://grafana.yourdomain.com
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    restart: unless-stopped
    depends_on:
      - prometheus

  loki:
    image: grafana/loki:latest
    container_name: greenlang-loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    restart: unless-stopped

  promtail:
    image: grafana/promtail:latest
    container_name: greenlang-promtail
    volumes:
      - /var/log:/var/log
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    container_name: greenlang-alertmanager
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
    ports:
      - "9093:9093"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
  loki_data:
  alertmanager_data:
```

See `monitoring/unified-dashboard.json` for Grafana dashboard configuration.

---

## 7. API Gateway (NGINX)

### 7.1 Deploy NGINX

```yaml
# docker-compose-nginx.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    container_name: greenlang-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    restart: unless-stopped
    depends_on:
      - cbam-app
      - csrd-web
      - vcci-backend-api
```

### 7.2 NGINX Configuration

See `GREENLANG_PLATFORM_DEPLOYMENT.md` section "Unified API Gateway" for complete NGINX configuration.

---

## Verification

After setting up all components, verify the infrastructure:

```bash
# PostgreSQL
psql -h <postgres-host> -U postgres -c "SELECT version();"

# Redis
redis-cli -h <redis-host> -a <password> ping

# Weaviate
curl http://<weaviate-host>:8080/v1/.well-known/ready

# RabbitMQ
curl -u <user>:<password> http://<rabbitmq-host>:15672/api/overview

# S3 (AWS)
aws s3 ls

# Prometheus
curl http://<prometheus-host>:9090/-/healthy

# Grafana
curl http://<grafana-host>:3000/api/health
```

---

## Next Steps

1. Deploy applications (see `GREENLANG_PLATFORM_DEPLOYMENT.md`)
2. Configure monitoring dashboards
3. Set up backups and disaster recovery
4. Perform load testing
5. Review security configuration

---

**Document Maintained By:** Infrastructure Team
**Last Updated:** 2025-11-08
