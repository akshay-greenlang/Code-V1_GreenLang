# Platform-Wide Disaster Recovery Strategy

**Version:** 1.0.0
**Last Updated:** 2025-11-08

---

## Executive Summary

This document outlines the disaster recovery (DR) strategy for the complete GreenLang Carbon Intelligence Platform, covering all three applications (CBAM, CSRD, VCCI) and shared infrastructure.

### Recovery Objectives

- **RTO (Recovery Time Objective):** 4 hours
- **RPO (Recovery Point Objective):** 1 hour
- **Data Loss Tolerance:** < 1 hour of data
- **Availability Target:** 99.9% (8.76 hours downtime/year)

---

## Table of Contents

1. [Disaster Scenarios](#disaster-scenarios)
2. [Backup Strategy](#backup-strategy)
3. [Recovery Procedures](#recovery-procedures)
4. [High Availability Architecture](#high-availability-architecture)
5. [Testing & Validation](#testing--validation)
6. [Incident Response](#incident-response)

---

## Disaster Scenarios

### Scenario 1: Database Failure (PostgreSQL)

**Impact:** High
**Probability:** Low
**RTO:** 1 hour
**RPO:** 0 (with replication)

**Causes:**
- Hardware failure
- Corruption
- Accidental deletion
- Ransomware attack

**Mitigation:**
- Multi-AZ deployment
- Read replicas
- Continuous WAL archiving
- Point-in-time recovery (PITR)

### Scenario 2: Application Server Failure

**Impact:** Medium
**Probability:** Medium
**RTO:** 15 minutes
**RPO:** 0 (stateless)

**Causes:**
- Instance failure
- Out of memory
- Network issues
- Deployment errors

**Mitigation:**
- Auto-scaling groups (min 2 instances)
- Health checks + auto-replacement
- Blue-green deployments
- Rollback capability

### Scenario 3: Cache Failure (Redis)

**Impact:** Low-Medium
**Probability:** Low
**RTO:** 30 minutes
**RPO:** Acceptable (cache can be rebuilt)

**Causes:**
- Instance failure
- Memory exhaustion
- Network partition

**Mitigation:**
- Redis cluster with replication
- Sentinel for automatic failover
- AOF + RDB persistence
- Applications degrade gracefully without cache

### Scenario 4: Complete Region Failure

**Impact:** Critical
**Probability:** Very Low
**RTO:** 4 hours
**RPO:** 1 hour

**Causes:**
- Natural disaster
- Major AWS/Azure/GCP outage
- Network partition
- Terrorism/war

**Mitigation:**
- Multi-region deployment (for large/enterprise)
- Cross-region replication
- Automated failover
- Regular DR drills

### Scenario 5: Data Corruption / Ransomware

**Impact:** Critical
**Probability:** Low
**RTO:** 8 hours
**RPO:** 24 hours

**Causes:**
- Ransomware attack
- Malicious insider
- Software bug
- Cascading failure

**Mitigation:**
- Immutable backups
- Offline backups (air-gapped)
- Version control for data
- Rapid malware detection
- Zero-trust security

---

## Backup Strategy

### PostgreSQL Database Backups

#### Continuous Backups (WAL Archiving)

```bash
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'aws s3 cp %p s3://greenlang-backups/wal/%f'
archive_timeout = 300  # Force WAL switch every 5 minutes

# Enable PITR
backup_label
recovery.conf (for restoration)
```

**Retention:** 30 days
**Storage:** S3 with versioning
**Frequency:** Continuous (every 5 minutes)

#### Full Database Backups

```bash
#!/bin/bash
# scripts/backup-postgres.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
S3_BUCKET="s3://greenlang-backups/postgres"

# Full backup (all databases)
pg_dumpall -U postgres -f ${BACKUP_DIR}/full_backup_${TIMESTAMP}.sql

# Compress
gzip ${BACKUP_DIR}/full_backup_${TIMESTAMP}.sql

# Upload to S3
aws s3 cp ${BACKUP_DIR}/full_backup_${TIMESTAMP}.sql.gz ${S3_BUCKET}/

# Clean up old local backups (keep last 3 days)
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +3 -delete

# Verify backup integrity
gunzip -t ${BACKUP_DIR}/full_backup_${TIMESTAMP}.sql.gz || \
  echo "Backup verification failed!" | mail -s "Backup Alert" ops@greenlang.io
```

**Retention:**
- Daily backups: 30 days
- Weekly backups: 90 days
- Monthly backups: 1 year

**Schedule (cron):**
```cron
# Daily full backup at 2 AM
0 2 * * * /opt/greenlang/scripts/backup-postgres.sh

# Weekly backup (Sundays at 3 AM)
0 3 * * 0 /opt/greenlang/scripts/backup-postgres-weekly.sh

# Monthly backup (1st of month at 4 AM)
0 4 1 * * /opt/greenlang/scripts/backup-postgres-monthly.sh
```

#### Per-Database Backups

```bash
# Backup individual databases
pg_dump -U postgres -d cbam_db -F c -f cbam_db_${TIMESTAMP}.dump
pg_dump -U postgres -d csrd_db -F c -f csrd_db_${TIMESTAMP}.dump
pg_dump -U postgres -d vcci_db -F c -f vcci_db_${TIMESTAMP}.dump
pg_dump -U postgres -d shared_db -F c -f shared_db_${TIMESTAMP}.dump
```

### Redis Backups

#### RDB Snapshots

```conf
# redis.conf
save 900 1      # Save if 1 key changed in 15 minutes
save 300 10     # Save if 10 keys changed in 5 minutes
save 60 10000   # Save if 10,000 keys changed in 1 minute

dbfilename dump.rdb
dir /data
```

#### AOF (Append-Only File)

```conf
# redis.conf
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec  # fsync every second (good balance)
```

#### Backup Script

```bash
#!/bin/bash
# scripts/backup-redis.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/redis"
S3_BUCKET="s3://greenlang-backups/redis"

# Trigger RDB snapshot
redis-cli BGSAVE

# Wait for completion
while [ $(redis-cli LASTSAVE) == $LASTSAVE ]; do
  sleep 1
done

# Copy RDB and AOF
cp /data/dump.rdb ${BACKUP_DIR}/dump_${TIMESTAMP}.rdb
cp /data/appendonly.aof ${BACKUP_DIR}/appendonly_${TIMESTAMP}.aof

# Upload to S3
aws s3 cp ${BACKUP_DIR}/dump_${TIMESTAMP}.rdb ${S3_BUCKET}/
aws s3 cp ${BACKUP_DIR}/appendonly_${TIMESTAMP}.aof ${S3_BUCKET}/

# Clean up old backups (keep last 7 days)
find ${BACKUP_DIR} -name "*.rdb" -mtime +7 -delete
find ${BACKUP_DIR} -name "*.aof" -mtime +7 -delete
```

**Schedule:** Every 6 hours

### Weaviate Backups

```bash
#!/bin/bash
# scripts/backup-weaviate.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_ID="greenlang-backup-${TIMESTAMP}"

# Create Weaviate backup via API
curl -X POST "http://weaviate:8080/v1/backups/greenlang" \
  -H "Content-Type: application/json" \
  -d "{
    \"id\": \"${BACKUP_ID}\",
    \"include\": [\"Entity\", \"Product\", \"Supplier\"]
  }"

# Wait for backup completion (poll status)
while true; do
  STATUS=$(curl -s "http://weaviate:8080/v1/backups/greenlang/${BACKUP_ID}" | jq -r '.status')
  if [ "$STATUS" == "SUCCESS" ]; then
    break
  elif [ "$STATUS" == "FAILED" ]; then
    echo "Weaviate backup failed!"
    exit 1
  fi
  sleep 10
done

# Copy backup to S3
aws s3 sync /var/lib/weaviate/backups/${BACKUP_ID} \
  s3://greenlang-backups/weaviate/${BACKUP_ID}/
```

**Schedule:** Daily (3 AM)
**Retention:** 30 days

### Application Configuration Backups

```bash
#!/bin/bash
# scripts/backup-config.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="config_${TIMESTAMP}.tar.gz"

# Backup configuration files
tar -czf /backups/${BACKUP_FILE} \
  /opt/greenlang/config/ \
  /opt/greenlang/.env \
  /opt/greenlang/deployment/ \
  /etc/nginx/nginx.conf \
  /etc/prometheus/prometheus.yml \
  /etc/grafana/

# Upload to S3
aws s3 cp /backups/${BACKUP_FILE} s3://greenlang-backups/config/

# Encrypt sensitive configs
gpg --encrypt --recipient ops@greenlang.io /backups/${BACKUP_FILE}
aws s3 cp /backups/${BACKUP_FILE}.gpg s3://greenlang-backups/config/encrypted/
```

**Schedule:** Daily (1 AM)

### S3 Data Backups

**S3 Versioning:** Enabled on all buckets
**Cross-Region Replication:** Enabled for critical buckets

```bash
# Enable versioning
aws s3api put-bucket-versioning \
  --bucket greenlang-data \
  --versioning-configuration Status=Enabled

# Enable cross-region replication
aws s3api put-bucket-replication \
  --bucket greenlang-data \
  --replication-configuration file://replication-config.json
```

---

## Recovery Procedures

### Procedure 1: Restore PostgreSQL from Backup

#### Full Restore (from pg_dumpall)

```bash
#!/bin/bash
# scripts/restore-postgres-full.sh

# 1. Stop all applications
docker-compose -f docker-compose.prod.yml stop cbam-app csrd-web vcci-backend-api

# 2. Download latest backup from S3
LATEST_BACKUP=$(aws s3 ls s3://greenlang-backups/postgres/ --recursive | sort | tail -n 1 | awk '{print $4}')
aws s3 cp s3://greenlang-backups/${LATEST_BACKUP} /tmp/postgres_backup.sql.gz

# 3. Decompress
gunzip /tmp/postgres_backup.sql.gz

# 4. Drop existing databases (CAUTION!)
psql -U postgres -c "DROP DATABASE IF EXISTS cbam_db;"
psql -U postgres -c "DROP DATABASE IF EXISTS csrd_db;"
psql -U postgres -c "DROP DATABASE IF EXISTS vcci_db;"
psql -U postgres -c "DROP DATABASE IF EXISTS shared_db;"

# 5. Restore
psql -U postgres -f /tmp/postgres_backup.sql

# 6. Verify restoration
psql -U postgres -c "\l"  # List databases

# 7. Start applications
docker-compose -f docker-compose.prod.yml start cbam-app csrd-web vcci-backend-api

# 8. Verify application health
./scripts/health-check-all.sh
```

**Estimated Time:** 1-2 hours (depends on data size)

#### Point-in-Time Recovery (PITR)

```bash
#!/bin/bash
# scripts/restore-postgres-pitr.sh

TARGET_TIME="2025-11-08 10:30:00"  # Restore to this point

# 1. Stop PostgreSQL
systemctl stop postgresql

# 2. Backup current data (just in case)
mv /var/lib/postgresql/data /var/lib/postgresql/data.old

# 3. Restore base backup
aws s3 cp s3://greenlang-backups/postgres/base_backup_20251108.tar.gz /tmp/
tar -xzf /tmp/base_backup_20251108.tar.gz -C /var/lib/postgresql/data

# 4. Create recovery.conf
cat > /var/lib/postgresql/data/recovery.conf <<EOF
restore_command = 'aws s3 cp s3://greenlang-backups/wal/%f %p'
recovery_target_time = '${TARGET_TIME}'
recovery_target_action = 'promote'
EOF

# 5. Start PostgreSQL (will replay WAL logs until target time)
systemctl start postgresql

# 6. Monitor recovery
tail -f /var/log/postgresql/postgresql.log

# 7. After recovery completes, verify
psql -U postgres -c "SELECT NOW();"
```

**Estimated Time:** 2-4 hours (depends on WAL size)

### Procedure 2: Restore Redis from Backup

```bash
#!/bin/bash
# scripts/restore-redis.sh

# 1. Stop Redis
systemctl stop redis

# 2. Download latest backup
LATEST_RDB=$(aws s3 ls s3://greenlang-backups/redis/ --recursive | grep ".rdb" | sort | tail -n 1 | awk '{print $4}')
LATEST_AOF=$(aws s3 ls s3://greenlang-backups/redis/ --recursive | grep ".aof" | sort | tail -n 1 | awk '{print $4}')

aws s3 cp s3://greenlang-backups/${LATEST_RDB} /data/dump.rdb
aws s3 cp s3://greenlang-backups/${LATEST_AOF} /data/appendonly.aof

# 3. Set permissions
chown redis:redis /data/dump.rdb
chown redis:redis /data/appendonly.aof

# 4. Start Redis
systemctl start redis

# 5. Verify
redis-cli INFO keyspace
```

**Estimated Time:** 15-30 minutes

### Procedure 3: Restore Weaviate from Backup

```bash
#!/bin/bash
# scripts/restore-weaviate.sh

BACKUP_ID="greenlang-backup-20251108-020000"

# 1. Stop Weaviate
docker-compose stop weaviate

# 2. Download backup from S3
aws s3 sync s3://greenlang-backups/weaviate/${BACKUP_ID}/ \
  /var/lib/weaviate/backups/${BACKUP_ID}/

# 3. Start Weaviate
docker-compose start weaviate

# 4. Wait for Weaviate to be ready
until curl -s http://weaviate:8080/v1/.well-known/ready | grep -q "true"; do
  sleep 5
done

# 5. Restore via API
curl -X POST "http://weaviate:8080/v1/backups/greenlang/${BACKUP_ID}/restore" \
  -H "Content-Type: application/json" \
  -d "{}"

# 6. Monitor restoration status
while true; do
  STATUS=$(curl -s "http://weaviate:8080/v1/backups/greenlang/${BACKUP_ID}/restore" | jq -r '.status')
  echo "Restore status: $STATUS"
  if [ "$STATUS" == "SUCCESS" ]; then
    break
  elif [ "$STATUS" == "FAILED" ]; then
    echo "Weaviate restore failed!"
    exit 1
  fi
  sleep 10
done

# 7. Verify data
curl "http://weaviate:8080/v1/objects?class=Entity&limit=10"
```

**Estimated Time:** 30-60 minutes

### Procedure 4: Complete Platform Recovery (Worst Case)

```bash
#!/bin/bash
# scripts/disaster-recovery-full.sh

echo "===== STARTING FULL DISASTER RECOVERY ====="
echo "Estimated completion: 4 hours"

# Phase 1: Infrastructure (30 minutes)
echo "Phase 1: Restoring infrastructure..."
./scripts/deploy-infrastructure.sh

# Phase 2: Database (2 hours)
echo "Phase 2: Restoring PostgreSQL..."
./scripts/restore-postgres-pitr.sh

# Phase 3: Cache & Vector DB (1 hour)
echo "Phase 3: Restoring Redis and Weaviate..."
./scripts/restore-redis.sh &
./scripts/restore-weaviate.sh &
wait

# Phase 4: Applications (30 minutes)
echo "Phase 4: Deploying applications..."
docker-compose -f docker-compose.prod.yml up -d

# Phase 5: Verification (30 minutes)
echo "Phase 5: Verifying system health..."
sleep 60  # Wait for services to stabilize

./scripts/health-check-all.sh

if [ $? -eq 0 ]; then
  echo "===== DISASTER RECOVERY COMPLETE ====="
  echo "All systems operational."
else
  echo "===== DISASTER RECOVERY FAILED ====="
  echo "Manual intervention required."
  exit 1
fi
```

---

## High Availability Architecture

### Multi-AZ Deployment (Production)

```
Region: us-east-1

AZ 1 (us-east-1a):
  - Application Servers (2)
  - PostgreSQL Primary
  - Redis Master 1
  - Weaviate Node 1
  - NAT Gateway

AZ 2 (us-east-1b):
  - Application Servers (2)
  - PostgreSQL Replica 1
  - Redis Master 2
  - Weaviate Node 2
  - NAT Gateway

AZ 3 (us-east-1c):
  - Application Servers (2)
  - PostgreSQL Replica 2 (backup)
  - Redis Master 3
  - Weaviate Node 3
  - NAT Gateway

Load Balancer:
  - Health checks every 10 seconds
  - Automatic failover
  - Distributes traffic across all AZs
```

### Multi-Region Deployment (Enterprise)

```
Primary Region (us-east-1):
  - Full infrastructure
  - Read + Write operations
  - 70% of traffic

DR Region (eu-west-1):
  - Full infrastructure (standby)
  - Read-only (until failover)
  - 25% of traffic (read operations)

Failover Region (ap-southeast-1):
  - Minimal infrastructure
  - Read-only replicas
  - 5% of traffic

Failover Process:
  1. Detect primary region failure
  2. Promote DR region PostgreSQL replica to primary
  3. Update DNS to point to DR region
  4. Resume write operations in DR region
  5. Estimated failover time: 15 minutes (automated)
```

---

## Testing & Validation

### DR Drill Schedule

```yaml
Monthly (First Saturday):
  Test: Database backup restoration
  Duration: 2 hours
  Success Criteria: Restore < 1 hour, data integrity 100%

Quarterly (First Saturday):
  Test: Complete application recovery
  Duration: 4 hours
  Success Criteria: RTO < 4 hours, RPO < 1 hour

Annually (Scheduled maintenance window):
  Test: Multi-region failover
  Duration: 8 hours
  Success Criteria: Complete region failover < 1 hour, zero data loss
```

### DR Drill Checklist

```markdown
# DR Drill Checklist - [Date]

## Pre-Drill
- [ ] Notify team (1 week advance)
- [ ] Schedule maintenance window
- [ ] Prepare test environment
- [ ] Backup current production state

## During Drill
- [ ] Simulate failure scenario
- [ ] Execute recovery procedures
- [ ] Document timeline and issues
- [ ] Verify data integrity
- [ ] Test application functionality
- [ ] Measure RTO/RPO

## Post-Drill
- [ ] Restore normal operations
- [ ] Conduct retrospective
- [ ] Update procedures based on learnings
- [ ] File incident report
- [ ] Schedule follow-up actions
```

---

## Incident Response

### Incident Severity Levels

| Severity | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| **P1 (Critical)** | Complete service outage | 15 minutes | Region failure, database down |
| **P2 (High)** | Major functionality impaired | 1 hour | One app down, high error rate |
| **P3 (Medium)** | Minor functionality impaired | 4 hours | Slow performance, cache miss |
| **P4 (Low)** | Cosmetic or minimal impact | Next business day | UI glitch, log warnings |

### Incident Response Team

```yaml
Incident Commander:
  - Primary: Platform Engineering Lead
  - Backup: Senior DevOps Engineer

Technical Lead:
  - Primary: Senior Backend Engineer
  - Backup: Database Administrator

Communications Lead:
  - Primary: Product Manager
  - Backup: Customer Success Manager

On-Call Rotation:
  - 24/7 coverage
  - Weekly rotation
  - PagerDuty integration
```

### Incident Response Process

```
1. Detect (0-5 min)
   - Monitoring alerts
   - Customer reports
   - Health check failures

2. Assess (5-15 min)
   - Determine severity
   - Identify impact
   - Assign Incident Commander

3. Respond (15 min - 4 hours)
   - Assemble response team
   - Execute recovery procedures
   - Communicate status updates

4. Recover (varies by severity)
   - Restore service
   - Verify functionality
   - Monitor for recurrence

5. Post-Incident (24 hours after)
   - Conduct post-mortem
   - Document root cause
   - Implement preventive measures
   - Update runbooks
```

---

## Contact Information

```yaml
Emergency Contacts:
  Platform Team:
    - Primary: +1-xxx-xxx-xxxx
    - Email: platform-oncall@greenlang.io
    - Slack: #platform-incidents

  Database Team:
    - Primary: +1-xxx-xxx-xxxx
    - Email: dba-oncall@greenlang.io

  Security Team:
    - Primary: +1-xxx-xxx-xxxx
    - Email: security@greenlang.io

Escalation Path:
  L1: On-Call Engineer (PagerDuty)
  L2: Engineering Lead
  L3: CTO
  L4: CEO (P1 only)

External Vendors:
  AWS Support: Enterprise support plan
  Anthropic Support: support@anthropic.com
  OpenAI Support: support@openai.com
```

---

**Document Owner:** Platform Engineering & SRE Teams
**Last Updated:** 2025-11-08
**Next Review:** Quarterly (after each DR drill)
**Last DR Drill:** [To be scheduled]
**Next DR Drill:** [To be scheduled]
