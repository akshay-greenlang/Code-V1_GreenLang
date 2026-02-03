# GreenLang Platform - Operations Guide

**Document ID:** INFRA-001-OPS
**Version:** 1.0.0
**Last Updated:** 2026-02-03
**Status:** Production

---

## Table of Contents

1. [Day-to-Day Operations](#day-to-day-operations)
2. [Scaling Procedures](#scaling-procedures)
3. [Backup and Restore Procedures](#backup-and-restore-procedures)
4. [Incident Response](#incident-response)
5. [Maintenance Procedures](#maintenance-procedures)
6. [Runbooks](#runbooks)

---

## Day-to-Day Operations

### Daily Checklist

Perform these checks at the start of each operational day:

```
[ ] Platform Health Check
    - Run: ./deploy.sh health
    - Verify all services show "healthy"

[ ] Monitor Dashboard Review
    - Access Grafana: http://localhost:3000
    - Check "GreenLang Platform - Unified Dashboard"
    - Review error rates (should be < 1%)
    - Review response times (p99 < 5s)

[ ] Log Review
    - Check for ERROR or CRITICAL entries
    - Run: docker-compose -f docker-compose-unified.yml logs --since 24h | grep -E "ERROR|CRITICAL"

[ ] Database Health
    - Check connection count (< 250)
    - Check disk space (> 20% free)
    - Run: docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -c "SELECT count(*) FROM pg_stat_activity;"

[ ] Queue Health
    - Check RabbitMQ queues (no stale messages)
    - Access: http://localhost:15672
    - Verify all queues have < 1000 pending messages

[ ] Backup Verification
    - Confirm last backup completed successfully
    - Check S3 bucket for latest backup files
```

### Common Operational Commands

#### Service Management

```bash
# Start all services
./deploy.sh start

# Stop all services
./deploy.sh stop

# Restart all services
./deploy.sh restart

# Check service status
./deploy.sh status

# View logs (all services)
./deploy.sh logs

# View logs (specific service)
docker-compose -f docker-compose-unified.yml logs -f cbam-api
docker-compose -f docker-compose-unified.yml logs -f csrd-web
docker-compose -f docker-compose-unified.yml logs -f vcci-backend

# View logs (last N lines)
docker-compose -f docker-compose-unified.yml logs --tail=100 cbam-api

# View logs (since timestamp)
docker-compose -f docker-compose-unified.yml logs --since 1h cbam-api
```

#### Health Checks

```bash
# Platform-wide health check
./deploy.sh health

# Individual service health checks
curl http://localhost:8001/health  # CBAM
curl http://localhost:8002/health  # CSRD
curl http://localhost:8000/health/live  # VCCI

# PostgreSQL health
docker-compose -f docker-compose-unified.yml exec postgres pg_isready -U greenlang_admin

# Redis health
docker-compose -f docker-compose-unified.yml exec redis redis-cli ping

# RabbitMQ health
docker-compose -f docker-compose-unified.yml exec rabbitmq rabbitmq-diagnostics ping
```

#### Resource Monitoring

```bash
# Container resource usage
docker stats

# Disk usage by volume
docker system df -v

# PostgreSQL database size
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
SELECT
    schemaname,
    pg_size_pretty(sum(pg_total_relation_size(schemaname||'.'||tablename))) as size
FROM pg_tables
WHERE schemaname IN ('cbam', 'csrd', 'vcci', 'public', 'shared')
GROUP BY schemaname
ORDER BY sum(pg_total_relation_size(schemaname||'.'||tablename)) DESC;
"

# Redis memory usage
docker-compose -f docker-compose-unified.yml exec redis redis-cli INFO memory | grep used_memory_human
```

### Service Restart Procedures

#### Restart Single Application (Zero Downtime)

```bash
# 1. Check current status
docker-compose -f docker-compose-unified.yml ps cbam-api

# 2. Restart the service
docker-compose -f docker-compose-unified.yml restart cbam-api

# 3. Wait for health check (30 seconds)
sleep 30

# 4. Verify health
curl http://localhost:8001/health
```

#### Restart Infrastructure Service

**WARNING:** Restarting infrastructure affects all applications.

```bash
# PostgreSQL Restart
# 1. Notify users of potential brief interruption
# 2. Restart
docker-compose -f docker-compose-unified.yml restart postgres

# 3. Wait for health (60 seconds)
sleep 60

# 4. Verify and restart applications if needed
docker-compose -f docker-compose-unified.yml exec postgres pg_isready -U greenlang_admin
docker-compose -f docker-compose-unified.yml restart cbam-api csrd-web vcci-backend vcci-worker
```

---

## Scaling Procedures

### Horizontal Scaling (Docker Compose)

#### Scale Application Instances

```bash
# Scale CBAM API to 3 instances
docker-compose -f docker-compose-unified.yml up -d --scale cbam-api=3

# Scale CSRD Web to 3 instances
docker-compose -f docker-compose-unified.yml up -d --scale csrd-web=3

# Scale VCCI Backend to 3 instances
docker-compose -f docker-compose-unified.yml up -d --scale vcci-backend=3

# Scale VCCI Workers to 5 instances
docker-compose -f docker-compose-unified.yml up -d --scale vcci-worker=5
```

**Note:** When scaling applications, you need a load balancer (nginx, Traefik) to distribute traffic.

#### Add Load Balancer (nginx)

Create `nginx.conf`:

```nginx
upstream cbam_backend {
    least_conn;
    server cbam-api-1:8000;
    server cbam-api-2:8000;
    server cbam-api-3:8000;
}

upstream csrd_backend {
    least_conn;
    server csrd-web-1:8000;
    server csrd-web-2:8000;
    server csrd-web-3:8000;
}

upstream vcci_backend {
    least_conn;
    server vcci-backend-1:8000;
    server vcci-backend-2:8000;
    server vcci-backend-3:8000;
}

server {
    listen 80;

    location /cbam/ {
        proxy_pass http://cbam_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /csrd/ {
        proxy_pass http://csrd_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /vcci/ {
        proxy_pass http://vcci_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Vertical Scaling (Resource Limits)

#### Update Container Resources

Edit `docker-compose-unified.yml`:

```yaml
cbam-api:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
```

Apply changes:

```bash
docker-compose -f docker-compose-unified.yml up -d cbam-api
```

### Scaling Guidelines

| Metric | Current | Scale Up Trigger | Scale Down Trigger |
|--------|---------|------------------|-------------------|
| CPU Usage | - | > 70% for 5 min | < 30% for 15 min |
| Memory Usage | - | > 80% for 5 min | < 40% for 15 min |
| Request Latency (p99) | < 1s | > 3s for 5 min | < 500ms for 15 min |
| Queue Depth | < 100 | > 1000 for 5 min | < 50 for 15 min |
| Error Rate | < 0.1% | > 1% for 2 min | < 0.01% for 15 min |

### Kubernetes Horizontal Pod Autoscaler

For Kubernetes deployments, use the HPA configuration:

```yaml
# deployment/kubernetes/manifests/worker-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vcci-worker-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vcci-worker
  minReplicas: 2
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
```

---

## Backup and Restore Procedures

### Automated Backup Schedule

| Component | Frequency | Retention | Location |
|-----------|-----------|-----------|----------|
| PostgreSQL Full | Daily 2 AM | 30 days | S3 |
| PostgreSQL WAL | Continuous | 7 days | S3 |
| Redis RDB | Every 6 hours | 7 days | S3 |
| Weaviate | Daily 3 AM | 30 days | S3 |
| Configuration | Daily 1 AM | 90 days | S3 |

### Manual Backup Procedures

#### PostgreSQL Full Backup

```bash
#!/bin/bash
# backup-postgres.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
BACKUP_FILE="greenlang_platform_${TIMESTAMP}.sql.gz"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Perform backup
docker-compose -f docker-compose-unified.yml exec -T postgres \
  pg_dump -U greenlang_admin greenlang_platform | gzip > ${BACKUP_DIR}/${BACKUP_FILE}

# Verify backup
if [ -f "${BACKUP_DIR}/${BACKUP_FILE}" ] && [ -s "${BACKUP_DIR}/${BACKUP_FILE}" ]; then
    echo "Backup successful: ${BACKUP_FILE}"
    echo "Size: $(du -h ${BACKUP_DIR}/${BACKUP_FILE} | cut -f1)"
else
    echo "Backup FAILED!"
    exit 1
fi

# Optional: Upload to S3
# aws s3 cp ${BACKUP_DIR}/${BACKUP_FILE} s3://greenlang-backups/postgres/
```

#### PostgreSQL Schema-Specific Backup

```bash
# Backup CBAM schema only
docker-compose -f docker-compose-unified.yml exec -T postgres \
  pg_dump -U greenlang_admin -n cbam greenlang_platform > backup_cbam_$(date +%Y%m%d).sql

# Backup CSRD schema only
docker-compose -f docker-compose-unified.yml exec -T postgres \
  pg_dump -U greenlang_admin -n csrd greenlang_platform > backup_csrd_$(date +%Y%m%d).sql

# Backup VCCI schema only
docker-compose -f docker-compose-unified.yml exec -T postgres \
  pg_dump -U greenlang_admin -n vcci greenlang_platform > backup_vcci_$(date +%Y%m%d).sql
```

#### Redis Backup

```bash
#!/bin/bash
# backup-redis.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/redis"

mkdir -p ${BACKUP_DIR}

# Trigger RDB snapshot
docker-compose -f docker-compose-unified.yml exec redis redis-cli BGSAVE

# Wait for background save to complete
sleep 10

# Copy RDB file
docker cp greenlang-redis:/data/dump.rdb ${BACKUP_DIR}/dump_${TIMESTAMP}.rdb

echo "Redis backup complete: dump_${TIMESTAMP}.rdb"
```

### Restore Procedures

#### PostgreSQL Full Restore

**WARNING:** This will overwrite all existing data.

```bash
#!/bin/bash
# restore-postgres.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    exit 1
fi

# 1. Stop applications
echo "Stopping applications..."
docker-compose -f docker-compose-unified.yml stop cbam-api csrd-web vcci-backend vcci-worker

# 2. Wait for connections to close
sleep 10

# 3. Drop and recreate database
echo "Recreating database..."
docker-compose -f docker-compose-unified.yml exec -T postgres psql -U greenlang_admin -c "
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'greenlang_platform' AND pid <> pg_backend_pid();
"
docker-compose -f docker-compose-unified.yml exec -T postgres dropdb -U greenlang_admin greenlang_platform
docker-compose -f docker-compose-unified.yml exec -T postgres createdb -U greenlang_admin greenlang_platform

# 4. Restore from backup
echo "Restoring from backup..."
gunzip -c ${BACKUP_FILE} | docker-compose -f docker-compose-unified.yml exec -T postgres psql -U greenlang_admin greenlang_platform

# 5. Verify restoration
echo "Verifying restoration..."
docker-compose -f docker-compose-unified.yml exec -T postgres psql -U greenlang_admin -d greenlang_platform -c "\dn"

# 6. Restart applications
echo "Starting applications..."
docker-compose -f docker-compose-unified.yml start cbam-api csrd-web vcci-backend vcci-worker

# 7. Health check
sleep 30
./deploy.sh health

echo "Restore complete!"
```

#### Redis Restore

```bash
#!/bin/bash
# restore-redis.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <dump.rdb>"
    exit 1
fi

# 1. Stop Redis
docker-compose -f docker-compose-unified.yml stop redis

# 2. Copy backup file
docker cp ${BACKUP_FILE} greenlang-redis:/data/dump.rdb

# 3. Start Redis (will load from RDB)
docker-compose -f docker-compose-unified.yml start redis

# 4. Verify
sleep 5
docker-compose -f docker-compose-unified.yml exec redis redis-cli INFO keyspace

echo "Redis restore complete!"
```

#### Point-in-Time Recovery (PostgreSQL)

For recovering to a specific point in time (requires WAL archiving enabled):

```bash
#!/bin/bash
# restore-pitr.sh

TARGET_TIME="2026-02-03 10:30:00"  # Restore to this point

# 1. Stop PostgreSQL
docker-compose -f docker-compose-unified.yml stop postgres

# 2. Backup current data
mv postgres-data postgres-data.old

# 3. Restore base backup
tar -xzf base_backup.tar.gz -C postgres-data

# 4. Create recovery.signal
echo "restore_command = 'cp /wal_archive/%f %p'
recovery_target_time = '${TARGET_TIME}'
recovery_target_action = 'promote'" > postgres-data/recovery.signal

# 5. Start PostgreSQL
docker-compose -f docker-compose-unified.yml start postgres

# 6. Monitor recovery
docker-compose -f docker-compose-unified.yml logs -f postgres
```

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **P1 - Critical** | Complete service outage | 15 minutes | All apps down, database failure |
| **P2 - High** | Major functionality impaired | 1 hour | One app down, high error rate |
| **P3 - Medium** | Minor functionality impaired | 4 hours | Slow performance, partial feature loss |
| **P4 - Low** | Cosmetic or minimal impact | Next business day | UI glitch, non-critical warnings |

### Incident Response Workflow

```
+=============================================================================+
|                        INCIDENT RESPONSE WORKFLOW                            |
+=============================================================================+

    [Incident Detected]
           |
           v
    +------------------+
    | 1. ACKNOWLEDGE   |  (within 5 minutes)
    | - Confirm alert  |
    | - Assign owner   |
    | - Create ticket  |
    +------------------+
           |
           v
    +------------------+
    | 2. ASSESS        |  (within 15 minutes)
    | - Determine      |
    |   severity       |
    | - Identify scope |
    | - Notify team    |
    +------------------+
           |
           v
    +------------------+
    | 3. COMMUNICATE   |  (ongoing)
    | - Status page    |
    | - Stakeholder    |
    |   updates        |
    | - Slack channel  |
    +------------------+
           |
           v
    +------------------+
    | 4. MITIGATE      |  (ASAP)
    | - Stop bleeding  |
    | - Restore service|
    | - Apply workaround|
    +------------------+
           |
           v
    +------------------+
    | 5. RESOLVE       |
    | - Root cause fix |
    | - Verify normal  |
    |   operation      |
    +------------------+
           |
           v
    +------------------+
    | 6. POST-MORTEM   |  (within 48 hours)
    | - Timeline       |
    | - Root cause     |
    | - Action items   |
    +------------------+
```

### Emergency Contacts

```
On-Call Engineer:     platform-oncall@greenlang.io
Engineering Lead:     engineering-lead@greenlang.io
Database Team:        dba-oncall@greenlang.io
Security Team:        security@greenlang.io

Escalation:
  L1 -> On-Call Engineer (PagerDuty)
  L2 -> Engineering Lead (15 min no response)
  L3 -> CTO (P1 incidents)
```

### Quick Recovery Actions

#### Application Not Responding

```bash
# 1. Check container status
docker-compose -f docker-compose-unified.yml ps cbam-api

# 2. Check logs for errors
docker-compose -f docker-compose-unified.yml logs --tail=100 cbam-api

# 3. Restart the service
docker-compose -f docker-compose-unified.yml restart cbam-api

# 4. If restart fails, recreate container
docker-compose -f docker-compose-unified.yml up -d --force-recreate cbam-api

# 5. Verify health
curl http://localhost:8001/health
```

#### Database Connection Issues

```bash
# 1. Check PostgreSQL status
docker-compose -f docker-compose-unified.yml exec postgres pg_isready -U greenlang_admin

# 2. Check connection count
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -c \
  "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 3. Kill idle connections if necessary
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - interval '1 hour';"

# 4. Restart applications to reset connection pools
docker-compose -f docker-compose-unified.yml restart cbam-api csrd-web vcci-backend
```

#### High Memory Usage

```bash
# 1. Check memory usage
docker stats --no-stream

# 2. Identify memory hogs
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}" | sort -k2 -h -r

# 3. Restart the affected service
docker-compose -f docker-compose-unified.yml restart <service-name>

# 4. Clear Redis cache if needed
docker-compose -f docker-compose-unified.yml exec redis redis-cli FLUSHALL

# 5. Consider increasing memory limits in docker-compose
```

---

## Maintenance Procedures

### Planned Maintenance Window

**Recommended Time:** Sundays 2:00 AM - 6:00 AM (local time)

#### Pre-Maintenance Checklist

```
[ ] Notify users 48 hours in advance
[ ] Create full backup
[ ] Verify backup integrity
[ ] Document rollback procedure
[ ] Confirm maintenance window
[ ] Prepare maintenance scripts
```

#### Maintenance Procedure Template

```bash
#!/bin/bash
# maintenance.sh

echo "=== Starting Maintenance ==="
date

# 1. Create backup
echo "Creating backup..."
./scripts/backup-postgres.sh
./scripts/backup-redis.sh

# 2. Stop applications
echo "Stopping applications..."
docker-compose -f docker-compose-unified.yml stop cbam-api csrd-web vcci-backend vcci-worker

# 3. Perform maintenance tasks
echo "Performing maintenance..."
# Add your maintenance tasks here

# 4. Start applications
echo "Starting applications..."
docker-compose -f docker-compose-unified.yml start cbam-api csrd-web vcci-backend vcci-worker

# 5. Verify health
echo "Verifying health..."
sleep 60
./deploy.sh health

echo "=== Maintenance Complete ==="
date
```

### Database Maintenance

#### Vacuum and Analyze

Run weekly to maintain database performance:

```bash
# Full vacuum (locks tables - run during maintenance window)
docker-compose -f docker-compose-unified.yml exec postgres vacuumdb -U greenlang_admin -d greenlang_platform --full --analyze

# Regular vacuum (no locks - can run anytime)
docker-compose -f docker-compose-unified.yml exec postgres vacuumdb -U greenlang_admin -d greenlang_platform --analyze
```

#### Index Maintenance

```bash
# Check index usage
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
SELECT
    schemaname,
    relname as table,
    indexrelname as index,
    idx_scan as scans,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC
LIMIT 20;
"

# Reindex a specific table
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
REINDEX TABLE cbam.declarations;
"
```

### Container Updates

#### Update Application Images

```bash
# 1. Build new images
./deploy.sh build

# 2. Update one service at a time (rolling update)
docker-compose -f docker-compose-unified.yml up -d --no-deps cbam-api
sleep 30
./deploy.sh health

docker-compose -f docker-compose-unified.yml up -d --no-deps csrd-web
sleep 30
./deploy.sh health

docker-compose -f docker-compose-unified.yml up -d --no-deps vcci-backend vcci-worker
sleep 30
./deploy.sh health
```

#### Update Infrastructure Images

```bash
# 1. Create backups first
./scripts/backup-postgres.sh
./scripts/backup-redis.sh

# 2. Pull new images
docker-compose -f docker-compose-unified.yml pull postgres redis rabbitmq weaviate

# 3. Update during maintenance window
docker-compose -f docker-compose-unified.yml up -d postgres
sleep 60
docker-compose -f docker-compose-unified.yml exec postgres pg_isready -U greenlang_admin

docker-compose -f docker-compose-unified.yml up -d redis rabbitmq weaviate
sleep 30

# 4. Verify all services
./deploy.sh health
```

### Log Rotation

Docker handles log rotation automatically, but verify settings:

```json
// /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  }
}
```

### Cleanup Procedures

```bash
# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove unused volumes (CAUTION: verify no important data)
docker volume prune -f

# Remove unused networks
docker network prune -f

# Full cleanup
docker system prune -a -f --volumes
```

---

## Runbooks

### Runbook: Application Deployment

```yaml
name: Deploy Application Update
trigger: Manual or CI/CD
duration: 15-30 minutes

steps:
  - name: Pre-deployment checks
    commands:
      - ./deploy.sh health
      - docker-compose -f docker-compose-unified.yml exec postgres pg_isready

  - name: Create backup
    commands:
      - ./scripts/backup-postgres.sh

  - name: Pull/build new image
    commands:
      - ./deploy.sh build

  - name: Deploy with rolling update
    commands:
      - docker-compose -f docker-compose-unified.yml up -d --no-deps cbam-api
      - sleep 30
      - curl http://localhost:8001/health

  - name: Post-deployment verification
    commands:
      - ./deploy.sh health
      - docker-compose -f docker-compose-unified.yml logs --tail=50 cbam-api

rollback:
  - docker-compose -f docker-compose-unified.yml stop cbam-api
  - docker tag greenlang/cbam-app:previous greenlang/cbam-app:latest
  - docker-compose -f docker-compose-unified.yml up -d cbam-api
```

### Runbook: Database Migration

```yaml
name: Execute Database Migration
trigger: Application deployment with schema changes
duration: 10-60 minutes (depends on data size)

steps:
  - name: Backup database
    commands:
      - ./scripts/backup-postgres.sh

  - name: Review migration
    commands:
      - cat migrations/XXXX_migration_name.sql

  - name: Test migration on backup
    commands:
      - createdb migration_test
      - psql migration_test < backup.sql
      - psql migration_test < migrations/XXXX_migration_name.sql
      - dropdb migration_test

  - name: Execute migration
    commands:
      - docker-compose -f docker-compose-unified.yml exec -T postgres psql -U greenlang_admin -d greenlang_platform < migrations/XXXX_migration_name.sql

  - name: Verify migration
    commands:
      - docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "\dt cbam.*"

rollback:
  - ./scripts/restore-postgres.sh backup_before_migration.sql.gz
```

### Runbook: Certificate Renewal

```yaml
name: Renew SSL/TLS Certificates
trigger: Certificate expiry (30 days before)
duration: 15 minutes

steps:
  - name: Check current certificate
    commands:
      - openssl s_client -connect api.greenlang.io:443 -servername api.greenlang.io 2>/dev/null | openssl x509 -noout -dates

  - name: Request new certificate (Let's Encrypt)
    commands:
      - certbot renew --dry-run
      - certbot renew

  - name: Update certificate in deployment
    commands:
      - kubectl create secret tls greenlang-tls --cert=fullchain.pem --key=privkey.pem --dry-run=client -o yaml | kubectl apply -f -

  - name: Reload ingress/load balancer
    commands:
      - kubectl rollout restart deployment/nginx-ingress-controller

verification:
  - curl -vI https://api.greenlang.io 2>&1 | grep "expire date"
```

---

## Related Documentation

- [Architecture Guide](./ARCHITECTURE.md) - System architecture details
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common issues and solutions
- [Disaster Recovery](../platform-disaster-recovery.md) - Recovery procedures
- [Security Guide](../security/README.md) - Security procedures

---

**Document Owner:** Platform Operations Team
**Review Cycle:** Monthly
**Next Review:** 2026-03-03
