# GL-001 ThermalCommand - Disaster Recovery Plan

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-001 |
| Agent Name | ThermalCommand |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Classification | Internal - Operations |
| Owner | GreenLang Process Heat Team |
| Review Cycle | Quarterly |

---

## 1. Purpose and Scope

### 1.1 Purpose

This Disaster Recovery Plan (DRP) provides comprehensive procedures for recovering GL-001 ThermalCommand services following a disaster event. The plan ensures business continuity for process heat operations, protecting $20B in value at stake.

### 1.2 Scope

This plan covers:
- Complete regional failure
- Data center loss
- Ransomware/security incidents
- Extended outages (> 1 hour)
- Data corruption scenarios

### 1.3 Recovery Objectives

| Objective | Target | Maximum Tolerable |
|-----------|--------|-------------------|
| Recovery Time Objective (RTO) | 5 minutes | 30 minutes |
| Recovery Point Objective (RPO) | 1 minute | 5 minutes |
| Maximum Tolerable Downtime (MTD) | 30 minutes | 4 hours |

---

## 2. Backup Procedures and Schedules

### 2.1 Backup Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Primary Region  |     |  Secondary Region |     |   Offline Storage |
|   (US-EAST-1)     |     |   (US-WEST-2)     |     |   (Glacier Deep)  |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
+-------------------+     +-------------------+     +-------------------+
| Real-time Sync    |     | Near-real-time    |     | Daily Archives    |
| - WAL streaming   |     | - 1-min delay     |     | - 90-day retention|
| - Redis sync      |     | - Cross-region    |     | - Encrypted       |
| - Kafka mirror    |     |   replication     |     | - Immutable       |
+-------------------+     +-------------------+     +-------------------+
```

### 2.2 Backup Schedule Matrix

| Data Type | Method | Frequency | Retention | Location |
|-----------|--------|-----------|-----------|----------|
| PostgreSQL - WAL | Streaming | Continuous | 7 days | S3 + Cross-region |
| PostgreSQL - Full | pg_basebackup | Daily 02:00 UTC | 30 days | S3 Glacier |
| PostgreSQL - Incremental | pgBackRest | Every 6 hours | 14 days | S3 Standard |
| Redis RDB | Snapshot | Every 5 minutes | 7 days | S3 Standard |
| Redis AOF | Append-only | Continuous | 24 hours | Local + S3 |
| Kafka Topics | MirrorMaker2 | Continuous | 7 days | Cross-region |
| Audit Logs | Immutable append | Continuous | 7 years | S3 Glacier Deep |
| Configuration | GitOps | On change | Unlimited | GitHub |
| Secrets | Vault replication | Continuous | 90 days | Vault cluster |
| ML Models | S3 versioning | On deploy | 180 days | S3 Standard |

### 2.3 Backup Verification

```bash
#!/bin/bash
# Daily backup verification - runs at 06:00 UTC

# Verify PostgreSQL backup
BACKUP_DATE=$(date -d "yesterday" +%Y-%m-%d)
BACKUP_FILE="s3://greenlang-backups/gl-001/postgresql/full/${BACKUP_DATE}.tar.gz"

# Check backup exists
aws s3 ls "$BACKUP_FILE" || alert "PostgreSQL backup missing"

# Verify backup integrity
aws s3 cp "$BACKUP_FILE" - | tar -tzf - > /dev/null || alert "PostgreSQL backup corrupted"

# Test restore to verification cluster
kubectl apply -f verification-cluster.yaml
./restore-to-verification.sh "$BACKUP_FILE"

# Run integrity checks
kubectl exec -it verification-pg-0 -- psql -c "SELECT count(*) FROM heat_plans;"
kubectl exec -it verification-pg-0 -- psql -c "SELECT count(*) FROM audit_log;"

# Cleanup
kubectl delete -f verification-cluster.yaml

# Log success
echo "$(date): Backup verification completed successfully" >> /var/log/backup-verification.log
```

### 2.4 Automated Backup Script

```bash
#!/bin/bash
# /scripts/dr/backup.sh
# Automated backup orchestration for GL-001 ThermalCommand

set -euo pipefail

# Configuration
BACKUP_ROOT="s3://greenlang-backups/gl-001"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/gl-001-backup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# PostgreSQL Full Backup
backup_postgresql() {
    log "Starting PostgreSQL backup..."

    kubectl exec -it postgresql-0 -n greenlang -- \
        pg_basebackup -D /tmp/backup -Ft -z -P \
        -h localhost -U replicator

    kubectl cp greenlang/postgresql-0:/tmp/backup/base.tar.gz \
        /tmp/postgresql_${TIMESTAMP}.tar.gz

    aws s3 cp /tmp/postgresql_${TIMESTAMP}.tar.gz \
        ${BACKUP_ROOT}/postgresql/full/

    # Encrypt and upload to cross-region
    gpg --encrypt --recipient backup@greenlang.ai \
        /tmp/postgresql_${TIMESTAMP}.tar.gz

    aws s3 cp /tmp/postgresql_${TIMESTAMP}.tar.gz.gpg \
        s3://greenlang-backups-dr/gl-001/postgresql/full/ \
        --region us-west-2

    log "PostgreSQL backup completed: ${TIMESTAMP}"
}

# Redis Backup
backup_redis() {
    log "Starting Redis backup..."

    kubectl exec -it redis-master-0 -n greenlang -- redis-cli BGSAVE
    sleep 10  # Wait for RDB dump

    kubectl cp greenlang/redis-master-0:/data/dump.rdb \
        /tmp/redis_${TIMESTAMP}.rdb

    aws s3 cp /tmp/redis_${TIMESTAMP}.rdb \
        ${BACKUP_ROOT}/redis/rdb/

    log "Redis backup completed: ${TIMESTAMP}"
}

# Kafka Topic Backup
backup_kafka_topics() {
    log "Starting Kafka topic backup..."

    TOPICS=(
        "gl001.telemetry.normalized"
        "gl001.plan.dispatch"
        "gl001.safety.events"
        "gl001.audit.log"
    )

    for topic in "${TOPICS[@]}"; do
        kafka-console-consumer \
            --bootstrap-server kafka-0.greenlang.svc:9092 \
            --topic "$topic" \
            --from-beginning \
            --timeout-ms 30000 \
            > /tmp/kafka_${topic}_${TIMESTAMP}.json

        gzip /tmp/kafka_${topic}_${TIMESTAMP}.json
        aws s3 cp /tmp/kafka_${topic}_${TIMESTAMP}.json.gz \
            ${BACKUP_ROOT}/kafka/topics/
    done

    log "Kafka topic backup completed: ${TIMESTAMP}"
}

# Configuration Backup
backup_configuration() {
    log "Starting configuration backup..."

    # Export all ConfigMaps
    kubectl get configmaps -n greenlang -o yaml > /tmp/configmaps_${TIMESTAMP}.yaml

    # Export all Secrets (encrypted)
    kubectl get secrets -n greenlang -o yaml | \
        sops --encrypt --age age1... > /tmp/secrets_${TIMESTAMP}.yaml.enc

    aws s3 cp /tmp/configmaps_${TIMESTAMP}.yaml ${BACKUP_ROOT}/config/
    aws s3 cp /tmp/secrets_${TIMESTAMP}.yaml.enc ${BACKUP_ROOT}/config/

    log "Configuration backup completed: ${TIMESTAMP}"
}

# Audit Log Archival
archive_audit_logs() {
    log "Starting audit log archival..."

    # Export audit logs older than 30 days
    kubectl exec -it postgresql-0 -n greenlang -- psql -c "
        COPY (
            SELECT * FROM audit_log
            WHERE created_at < NOW() - INTERVAL '30 days'
        ) TO STDOUT WITH CSV HEADER
    " > /tmp/audit_archive_${TIMESTAMP}.csv

    # Compress and encrypt
    gzip /tmp/audit_archive_${TIMESTAMP}.csv
    gpg --encrypt --recipient compliance@greenlang.ai \
        /tmp/audit_archive_${TIMESTAMP}.csv.gz

    # Upload to Glacier Deep Archive
    aws s3 cp /tmp/audit_archive_${TIMESTAMP}.csv.gz.gpg \
        s3://greenlang-archives/gl-001/audit/ \
        --storage-class DEEP_ARCHIVE

    # Verify upload and delete from primary
    if aws s3 ls s3://greenlang-archives/gl-001/audit/audit_archive_${TIMESTAMP}.csv.gz.gpg; then
        kubectl exec -it postgresql-0 -n greenlang -- psql -c "
            DELETE FROM audit_log
            WHERE created_at < NOW() - INTERVAL '30 days'
        "
    fi

    log "Audit log archival completed: ${TIMESTAMP}"
}

# Main execution
main() {
    log "========================================="
    log "GL-001 ThermalCommand Backup Starting"
    log "========================================="

    backup_postgresql
    backup_redis
    backup_kafka_topics
    backup_configuration

    # Weekly audit archival
    if [ "$(date +%u)" = "7" ]; then
        archive_audit_logs
    fi

    log "========================================="
    log "GL-001 ThermalCommand Backup Complete"
    log "========================================="

    # Send success notification
    curl -X POST "${SLACK_WEBHOOK}" \
        -H 'Content-type: application/json' \
        --data '{"text":"GL-001 Backup completed successfully at '"${TIMESTAMP}"'"}'
}

main "$@"
```

---

## 3. Recovery Procedures

### 3.1 Disaster Classification

| Level | Description | Example | Response Team |
|-------|-------------|---------|---------------|
| Level 1 | Single component failure | Pod crash, replica failure | On-call SRE |
| Level 2 | Multiple component failure | Database primary + replica | SRE Team |
| Level 3 | Availability Zone failure | Complete AZ outage | SRE + Platform |
| Level 4 | Regional failure | Region unavailable | Full DR Team |
| Level 5 | Multi-region / Data loss | Ransomware, corruption | Executive + DR Team |

### 3.2 Level 4: Regional Failover Procedure

#### Step 1: Declare Disaster (0-5 minutes)

```bash
#!/bin/bash
# Incident Commander executes this

# 1. Verify primary region is unavailable
./scripts/dr/verify-region.sh us-east-1
if [ $? -eq 0 ]; then
    echo "Primary region responding. Investigate before failover."
    exit 1
fi

# 2. Open incident
./scripts/incident/create-incident.sh \
    --severity P1 \
    --title "GL-001 Regional Failover - US-EAST-1 Unavailable" \
    --commander "$ONCALL_IC"

# 3. Notify stakeholders
./scripts/notify/send-alert.sh \
    --channel "#gl-001-ops" \
    --message "DISASTER DECLARED: Initiating regional failover to US-WEST-2"
```

#### Step 2: Activate DR Site (5-10 minutes)

```bash
#!/bin/bash
# /scripts/dr/activate-dr-site.sh

# Switch kubectl context
export KUBECONFIG=/etc/kubernetes/dr-cluster-config
kubectl config use-context eks-us-west-2-dr

# 1. Scale up DR application replicas
kubectl scale deployment gl-001-thermalcommand \
    --replicas=5 \
    -n greenlang

# 2. Verify pod health
kubectl wait --for=condition=ready pod \
    -l app=gl-001-thermalcommand \
    -n greenlang \
    --timeout=120s

# 3. Promote PostgreSQL replica to primary
kubectl exec -it postgresql-0 -n greenlang -- \
    patronictl switchover --master postgresql-0 --candidate postgresql-replica-0 --force

# 4. Verify database is writable
kubectl exec -it postgresql-replica-0 -n greenlang -- \
    psql -c "INSERT INTO health_check (checked_at) VALUES (NOW());"
```

#### Step 3: Update DNS and Traffic (10-15 minutes)

```bash
#!/bin/bash
# /scripts/dr/update-dns.sh

# 1. Update Route53 health check
aws route53 update-health-check \
    --health-check-id HC123456789 \
    --disabled

# 2. Update DNS to point to DR region
cat > /tmp/dns-failover.json << EOF
{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "gl-001.greenlang.io",
      "Type": "A",
      "AliasTarget": {
        "HostedZoneId": "Z2FDTNDATAQYW2",
        "DNSName": "us-west-2-alb.greenlang.io",
        "EvaluateTargetHealth": true
      }
    }
  }]
}
EOF

aws route53 change-resource-record-sets \
    --hosted-zone-id Z123456789 \
    --change-batch file:///tmp/dns-failover.json

# 3. Verify DNS propagation
for i in {1..10}; do
    dig +short gl-001.greenlang.io
    sleep 30
done
```

#### Step 4: Verify Services (15-20 minutes)

```bash
#!/bin/bash
# /scripts/dr/verify-services.sh

echo "=== GL-001 DR Verification ==="

# 1. API Health Check
curl -f https://gl-001.greenlang.io/api/v1/health || exit 1

# 2. Deep Health Check
curl -f https://gl-001.greenlang.io/api/v1/health/deep || exit 1

# 3. Database Connectivity
kubectl exec -it gl-001-thermalcommand-0 -n greenlang -- \
    python -c "from core.db import engine; engine.connect()"

# 4. Kafka Connectivity
kubectl exec -it gl-001-thermalcommand-0 -n greenlang -- \
    python -c "from streaming.kafka_streaming import producer; producer.list_topics()"

# 5. Redis Connectivity
kubectl exec -it gl-001-thermalcommand-0 -n greenlang -- \
    python -c "from core.cache import redis_client; redis_client.ping()"

# 6. End-to-end Test
./scripts/e2e/smoke-test.sh

echo "=== All services verified ==="
```

#### Step 5: Communicate and Document (20-30 minutes)

```bash
#!/bin/bash
# /scripts/dr/post-failover.sh

# 1. Send success notification
./scripts/notify/send-alert.sh \
    --channel "#gl-001-ops" \
    --channel "#incident-management" \
    --message "GL-001 Regional Failover COMPLETE. Service restored in US-WEST-2."

# 2. Update status page
./scripts/statuspage/update.sh \
    --component "gl-001-thermalcommand" \
    --status "operational" \
    --message "Service restored following regional failover"

# 3. Create post-incident tasks
./scripts/incident/create-tasks.sh \
    --incident "$INCIDENT_ID" \
    --tasks "root-cause-analysis,primary-region-recovery,failback-planning"
```

### 3.3 Full Restore Script

```bash
#!/bin/bash
# /scripts/dr/restore.sh
# Complete restoration from backup

set -euo pipefail

BACKUP_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}
BACKUP_ROOT="s3://greenlang-backups/gl-001"
RESTORE_NAMESPACE="greenlang"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

restore_postgresql() {
    log "Restoring PostgreSQL from backup: ${BACKUP_DATE}..."

    # 1. Download backup
    aws s3 cp "${BACKUP_ROOT}/postgresql/full/${BACKUP_DATE}.tar.gz" /tmp/

    # 2. Stop application
    kubectl scale deployment gl-001-thermalcommand --replicas=0 -n ${RESTORE_NAMESPACE}

    # 3. Delete existing PVC (if corrupted)
    # kubectl delete pvc postgresql-data-postgresql-0 -n ${RESTORE_NAMESPACE}

    # 4. Create new PostgreSQL with restore
    kubectl apply -f - << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: pg-restore-script
  namespace: ${RESTORE_NAMESPACE}
data:
  restore.sh: |
    #!/bin/bash
    pg_restore -d greenlang -c -j 4 /backup/backup.tar
EOF

    # 5. Run restore job
    kubectl create job pg-restore-${BACKUP_DATE} \
        --image=postgres:14 \
        -n ${RESTORE_NAMESPACE} \
        -- /scripts/restore.sh

    # 6. Wait for completion
    kubectl wait --for=condition=complete job/pg-restore-${BACKUP_DATE} \
        -n ${RESTORE_NAMESPACE} --timeout=1h

    # 7. Verify data
    kubectl exec -it postgresql-0 -n ${RESTORE_NAMESPACE} -- \
        psql -c "SELECT COUNT(*) FROM heat_plans;"

    log "PostgreSQL restore completed"
}

restore_redis() {
    log "Restoring Redis from backup: ${BACKUP_DATE}..."

    # 1. Download RDB file
    aws s3 cp "${BACKUP_ROOT}/redis/rdb/redis_${BACKUP_DATE}.rdb" /tmp/dump.rdb

    # 2. Stop Redis
    kubectl scale statefulset redis-master --replicas=0 -n ${RESTORE_NAMESPACE}

    # 3. Copy RDB to PVC
    kubectl cp /tmp/dump.rdb ${RESTORE_NAMESPACE}/redis-restore-pod:/data/dump.rdb

    # 4. Restart Redis
    kubectl scale statefulset redis-master --replicas=1 -n ${RESTORE_NAMESPACE}

    # 5. Verify
    kubectl exec -it redis-master-0 -n ${RESTORE_NAMESPACE} -- redis-cli DBSIZE

    log "Redis restore completed"
}

restore_kafka() {
    log "Restoring Kafka topics from backup: ${BACKUP_DATE}..."

    TOPICS=(
        "gl001.telemetry.normalized"
        "gl001.plan.dispatch"
        "gl001.safety.events"
        "gl001.audit.log"
    )

    for topic in "${TOPICS[@]}"; do
        # Download backup
        aws s3 cp "${BACKUP_ROOT}/kafka/topics/kafka_${topic}_${BACKUP_DATE}.json.gz" /tmp/
        gunzip /tmp/kafka_${topic}_${BACKUP_DATE}.json.gz

        # Recreate topic if needed
        kafka-topics --create --if-not-exists \
            --bootstrap-server kafka-0.${RESTORE_NAMESPACE}.svc:9092 \
            --topic "$topic" \
            --partitions 8 \
            --replication-factor 3

        # Restore messages
        kafka-console-producer \
            --bootstrap-server kafka-0.${RESTORE_NAMESPACE}.svc:9092 \
            --topic "$topic" \
            < /tmp/kafka_${topic}_${BACKUP_DATE}.json
    done

    log "Kafka restore completed"
}

restore_configuration() {
    log "Restoring configuration from backup: ${BACKUP_DATE}..."

    # Download configuration
    aws s3 cp "${BACKUP_ROOT}/config/configmaps_${BACKUP_DATE}.yaml" /tmp/
    aws s3 cp "${BACKUP_ROOT}/config/secrets_${BACKUP_DATE}.yaml.enc" /tmp/

    # Decrypt secrets
    sops --decrypt /tmp/secrets_${BACKUP_DATE}.yaml.enc > /tmp/secrets.yaml

    # Apply configuration
    kubectl apply -f /tmp/configmaps_${BACKUP_DATE}.yaml -n ${RESTORE_NAMESPACE}
    kubectl apply -f /tmp/secrets.yaml -n ${RESTORE_NAMESPACE}

    log "Configuration restore completed"
}

main() {
    log "========================================="
    log "GL-001 ThermalCommand Restore Starting"
    log "Backup Date: ${BACKUP_DATE}"
    log "========================================="

    restore_postgresql
    restore_redis
    restore_kafka
    restore_configuration

    # Restart application
    kubectl scale deployment gl-001-thermalcommand --replicas=3 -n ${RESTORE_NAMESPACE}

    # Wait for ready
    kubectl wait --for=condition=ready pod \
        -l app=gl-001-thermalcommand \
        -n ${RESTORE_NAMESPACE} \
        --timeout=300s

    log "========================================="
    log "GL-001 ThermalCommand Restore Complete"
    log "========================================="
}

main "$@"
```

---

## 4. Data Restoration Verification

### 4.1 Verification Checklist

```bash
#!/bin/bash
# /scripts/dr/verify-backup.sh

set -euo pipefail

NAMESPACE="greenlang"
VERIFICATION_REPORT="/tmp/verification_$(date +%Y%m%d_%H%M%S).json"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

verify_postgresql() {
    log "Verifying PostgreSQL integrity..."

    # 1. Check table existence
    TABLES=$(kubectl exec -it postgresql-0 -n ${NAMESPACE} -- \
        psql -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")

    # 2. Check row counts
    HEAT_PLANS=$(kubectl exec -it postgresql-0 -n ${NAMESPACE} -- \
        psql -t -c "SELECT COUNT(*) FROM heat_plans;")

    AUDIT_LOGS=$(kubectl exec -it postgresql-0 -n ${NAMESPACE} -- \
        psql -t -c "SELECT COUNT(*) FROM audit_log;")

    # 3. Check data integrity (checksums)
    CHECKSUM=$(kubectl exec -it postgresql-0 -n ${NAMESPACE} -- \
        psql -t -c "SELECT md5(string_agg(id::text, '')) FROM heat_plans ORDER BY id;")

    # 4. Verify foreign key constraints
    FK_VIOLATIONS=$(kubectl exec -it postgresql-0 -n ${NAMESPACE} -- \
        psql -t -c "SELECT COUNT(*) FROM heat_plan_actions WHERE plan_id NOT IN (SELECT id FROM heat_plans);")

    echo "{\"postgresql\": {\"tables\": ${TABLES}, \"heat_plans\": ${HEAT_PLANS}, \"audit_logs\": ${AUDIT_LOGS}, \"checksum\": \"${CHECKSUM}\", \"fk_violations\": ${FK_VIOLATIONS}}}" >> ${VERIFICATION_REPORT}

    if [ "${FK_VIOLATIONS}" -gt 0 ]; then
        log "WARNING: Foreign key violations detected!"
        return 1
    fi

    log "PostgreSQL verification passed"
}

verify_redis() {
    log "Verifying Redis integrity..."

    # 1. Check key count
    KEY_COUNT=$(kubectl exec -it redis-master-0 -n ${NAMESPACE} -- redis-cli DBSIZE)

    # 2. Check memory usage
    MEMORY=$(kubectl exec -it redis-master-0 -n ${NAMESPACE} -- redis-cli INFO memory | grep used_memory_human)

    # 3. Check replication status
    REPL_STATUS=$(kubectl exec -it redis-master-0 -n ${NAMESPACE} -- redis-cli INFO replication | grep connected_slaves)

    echo "{\"redis\": {\"keys\": \"${KEY_COUNT}\", \"memory\": \"${MEMORY}\", \"replication\": \"${REPL_STATUS}\"}}" >> ${VERIFICATION_REPORT}

    log "Redis verification passed"
}

verify_kafka() {
    log "Verifying Kafka integrity..."

    TOPICS=(
        "gl001.telemetry.normalized"
        "gl001.plan.dispatch"
        "gl001.safety.events"
        "gl001.audit.log"
    )

    for topic in "${TOPICS[@]}"; do
        # Check topic exists
        EXISTS=$(kafka-topics --list --bootstrap-server kafka-0.${NAMESPACE}.svc:9092 | grep -c "$topic" || true)

        # Check partition count
        PARTITIONS=$(kafka-topics --describe --topic "$topic" --bootstrap-server kafka-0.${NAMESPACE}.svc:9092 | grep -c "Partition:" || true)

        # Check ISR status
        ISR=$(kafka-topics --describe --topic "$topic" --bootstrap-server kafka-0.${NAMESPACE}.svc:9092 | grep "Isr:" | head -1)

        echo "{\"topic\": \"${topic}\", \"exists\": ${EXISTS}, \"partitions\": ${PARTITIONS}, \"isr\": \"${ISR}\"}" >> ${VERIFICATION_REPORT}
    done

    log "Kafka verification passed"
}

verify_application() {
    log "Verifying application health..."

    # 1. Health check
    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" https://gl-001.greenlang.io/api/v1/health)

    # 2. Deep health check
    DEEP_HEALTH=$(curl -s https://gl-001.greenlang.io/api/v1/health/deep)

    # 3. Sample API call
    API_TEST=$(curl -s -o /dev/null -w "%{http_code}" https://gl-001.greenlang.io/api/v1/heat-plans)

    echo "{\"application\": {\"health\": ${HEALTH}, \"deep_health\": ${DEEP_HEALTH}, \"api_test\": ${API_TEST}}}" >> ${VERIFICATION_REPORT}

    if [ "${HEALTH}" -ne 200 ]; then
        log "ERROR: Application health check failed!"
        return 1
    fi

    log "Application verification passed"
}

main() {
    log "========================================="
    log "GL-001 Backup Verification Starting"
    log "========================================="

    echo "[]" > ${VERIFICATION_REPORT}

    verify_postgresql
    verify_redis
    verify_kafka
    verify_application

    log "========================================="
    log "Verification Complete"
    log "Report: ${VERIFICATION_REPORT}"
    log "========================================="

    cat ${VERIFICATION_REPORT}
}

main "$@"
```

### 4.2 Data Integrity Validation

| Check | Query/Command | Expected Result |
|-------|---------------|-----------------|
| Table count | `SELECT COUNT(*) FROM information_schema.tables` | >= 15 |
| Heat plans exist | `SELECT COUNT(*) FROM heat_plans` | > 0 |
| Audit integrity | `SELECT COUNT(*) FROM audit_log WHERE hash IS NULL` | 0 |
| Redis connectivity | `redis-cli PING` | PONG |
| Kafka topics | `kafka-topics --list` | All 7 topics |
| API health | `GET /api/v1/health` | 200 OK |

---

## 5. Communication Plan

### 5.1 Escalation Matrix

| Time Elapsed | Action | Contacts |
|--------------|--------|----------|
| 0-5 min | Acknowledge incident | On-call SRE |
| 5-15 min | Escalate to team lead | SRE Lead, Platform Lead |
| 15-30 min | Escalate to management | Director of Engineering |
| 30-60 min | Executive notification | VP Engineering, CTO |
| 60+ min | Customer communication | Customer Success, Legal |

### 5.2 Communication Templates

#### Internal Incident Declaration
```
SUBJECT: [P1] GL-001 ThermalCommand - Disaster Recovery Activated

SEVERITY: P1 - Critical
COMPONENT: GL-001 ThermalCommand
STATUS: DR Activated
INCIDENT COMMANDER: [Name]

SUMMARY:
Primary region (US-EAST-1) is unavailable. Initiating failover to DR site (US-WEST-2).

IMPACT:
- Process heat optimization temporarily unavailable
- Estimated customer impact: [X] sites

ACTIONS:
1. [TIME] DR site activation initiated
2. [TIME] Database failover in progress
3. [TIME] DNS update pending

NEXT UPDATE: [TIME + 15 min]

Join bridge: [Bridge URL]
```

#### Customer Notification
```
SUBJECT: Service Disruption Notice - GL-001 ThermalCommand

Dear Customer,

We are currently experiencing a service disruption affecting GL-001 ThermalCommand.

CURRENT STATUS: Investigating
IMPACT: Process heat optimization recommendations may be delayed
ESTIMATED RESOLUTION: [TIME]

Our team is actively working to restore full service. We will provide updates every 30 minutes.

For urgent matters, please contact: support@greenlang.io

We apologize for any inconvenience.

GreenLang Operations Team
```

### 5.3 Post-Incident Communication

```
SUBJECT: [RESOLVED] GL-001 ThermalCommand - Service Restored

INCIDENT DURATION: [START] - [END] ([DURATION])
ROOT CAUSE: [Brief description]

TIMELINE:
- [TIME] Issue detected by monitoring
- [TIME] DR activation initiated
- [TIME] Failover completed
- [TIME] Service verified operational
- [TIME] All-clear declared

IMPACT:
- Total downtime: [X] minutes
- Customers affected: [X]
- Data loss: None (RPO met)

NEXT STEPS:
1. Root cause analysis (due: [DATE])
2. Preventive measures implementation
3. Post-incident review meeting: [DATE/TIME]

Full incident report will be shared within 48 hours.
```

---

## 6. DR Testing Schedule

### 6.1 Test Calendar

| Test Type | Frequency | Next Scheduled | Duration |
|-----------|-----------|----------------|----------|
| Backup Verification | Daily | Automated | 15 min |
| Component Failover | Weekly | Sunday 02:00 UTC | 30 min |
| Database Recovery | Monthly | 1st Saturday | 2 hours |
| Regional Failover | Quarterly | Q1: Mar 15 | 4 hours |
| Full DR Test | Semi-annually | June 15 | 8 hours |
| Tabletop Exercise | Quarterly | With regional | 2 hours |

### 6.2 Test Scenarios

| Scenario | Test Steps | Success Criteria |
|----------|------------|------------------|
| Pod failure | Kill random pod | Recovery < 30s |
| Node failure | Drain node | Recovery < 2min |
| DB primary failure | Stop PostgreSQL primary | Failover < 30s |
| Redis master failure | Kill Redis master | Sentinel failover < 10s |
| AZ failure | Simulate AZ outage | Service continues |
| Region failure | Full DR activation | RTO < 30 min |
| Data corruption | Restore from backup | Data integrity verified |

---

## 7. Appendices

### 7.1 Recovery Team Contacts

| Role | Primary | Backup | Contact |
|------|---------|--------|---------|
| Incident Commander | Jane Smith | John Doe | ic@greenlang.ai |
| SRE Lead | Alex Chen | Sam Wilson | sre-lead@greenlang.ai |
| DBA | Maria Garcia | Tom Brown | dba@greenlang.ai |
| Platform Lead | Chris Lee | Pat Johnson | platform@greenlang.ai |
| Security | Robin Taylor | Jordan Kim | security@greenlang.ai |

### 7.2 External Contacts

| Vendor | Service | Contact | SLA |
|--------|---------|---------|-----|
| AWS | Cloud Infrastructure | aws-support | 15 min response |
| Confluent | Kafka Support | confluent-support | 1 hour response |
| PagerDuty | Alerting | pagerduty-support | 24/7 |
| Cloudflare | DNS/CDN | cloudflare-support | 30 min response |

### 7.3 Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL DevOps | Initial release |
