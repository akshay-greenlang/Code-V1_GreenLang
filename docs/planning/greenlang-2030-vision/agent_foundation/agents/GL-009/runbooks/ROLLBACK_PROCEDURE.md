# GL-009 THERMALIQ Rollback Procedure

**Agent**: GL-009 THERMALIQ ThermalEfficiencyCalculator
**Version**: 1.0.0
**Last Updated**: 2025-11-26
**Owner**: GreenLang SRE Team

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Rollback Checklist](#pre-rollback-checklist)
3. [Rollback Scenarios](#rollback-scenarios)
4. [Deployment Rollback](#deployment-rollback)
5. [Database Migration Rollback](#database-migration-rollback)
6. [Configuration Rollback](#configuration-rollback)
7. [Cache Invalidation](#cache-invalidation)
8. [Post-Rollback Verification](#post-rollback-verification)
9. [Communication Plan](#communication-plan)
10. [Rollback Decision Matrix](#rollback-decision-matrix)

---

## Overview

This runbook provides procedures for rolling back GL-009 THERMALIQ deployments, database migrations, configuration changes, and other critical updates when issues are detected in production.

### When to Rollback

**Immediate Rollback** (within 5 minutes of detection):
- Complete service outage
- Data corruption detected
- Security vulnerability introduced
- Error rate > 50%
- Critical functionality broken

**Planned Rollback** (within 30 minutes):
- Performance regression > 50%
- Error rate > 10%
- Significant functionality broken
- Incompatibility with dependent services

**Delayed Rollback** (within 2 hours):
- Minor performance regression
- Error rate 5-10%
- Non-critical functionality broken
- Workarounds available

**No Rollback** (fix forward):
- Error rate < 5%
- Cosmetic issues
- Documentation errors
- Configuration tuning needed

---

## Pre-Rollback Checklist

Before initiating any rollback, complete this checklist:

### 1. Verify Issue Severity

```bash
# Check error rate
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_calculation_errors_total[5m])/rate(thermaliq_calculation_requests_total[5m])' | \
  jq '.data.result[0].value[1]'

# Check service availability
curl https://api.greenlang.io/v1/thermaliq/health

# Check recent deployments
kubectl rollout history deployment/thermaliq -n gl-009-production | tail -5

# Check metrics dashboard
# https://grafana.greenlang.io/d/thermaliq-production
```

**Severity Assessment**:
- [ ] Error rate measured
- [ ] User impact assessed
- [ ] Business impact estimated
- [ ] Rollback urgency determined

### 2. Identify Rollback Target

```bash
# Get current version
kubectl get deployment thermaliq -n gl-009-production -o jsonpath='{.spec.template.spec.containers[0].image}'

# Get previous version
kubectl rollout history deployment/thermaliq -n gl-009-production

# Verify previous version was stable
kubectl logs -n gl-009-production -l app=thermaliq,version=v1.2.2 --tail=500 | \
  grep "ERROR" | wc -l

# Should be low error count in previous version
```

**Version Verification**:
- [ ] Current version identified
- [ ] Target rollback version identified
- [ ] Target version stability confirmed
- [ ] Database migration compatibility checked

### 3. Check Dependencies

```bash
# Check dependent services
kubectl get deployments -n gl-009-production -l depends-on=thermaliq

# Check API consumers
curl https://api-gateway.greenlang.io/v1/routing/consumers/thermaliq | jq .

# Check integration status
kubectl logs -n gl-009-production -l app=erp-connector --tail=100 | \
  grep "thermaliq"
```

**Dependency Assessment**:
- [ ] Dependent services identified
- [ ] API compatibility verified
- [ ] Integration impacts assessed
- [ ] Downstream systems notified (if needed)

### 4. Backup Current State

```bash
# Backup database
kubectl exec -it postgres-0 -n gl-009-production -- \
  pg_dump -U thermaliq -d thermaliq_production -F c -f /tmp/backup_$(date +%Y%m%d_%H%M%S).dump

kubectl cp gl-009-production/postgres-0:/tmp/backup_$(date +%Y%m%d_%H%M%S).dump \
  ./backups/

# Backup configuration
kubectl get configmap thermaliq-config -n gl-009-production -o yaml > \
  backups/config_$(date +%Y%m%d_%H%M%S).yaml

kubectl get secret thermaliq-secrets -n gl-009-production -o yaml > \
  backups/secrets_$(date +%Y%m%d_%H%M%S).yaml

# Backup current deployment spec
kubectl get deployment thermaliq -n gl-009-production -o yaml > \
  backups/deployment_$(date +%Y%m%d_%H%M%S).yaml

# Tag current state
git tag rollback-point-$(date +%Y%m%d-%H%M%S)
git push origin --tags
```

**Backup Verification**:
- [ ] Database backup created
- [ ] Configuration backup created
- [ ] Deployment spec backed up
- [ ] Git tag created
- [ ] Backup integrity verified

### 5. Notify Stakeholders

```bash
# Create rollback notification
cat > rollback_notification.md <<EOF
# Rollback Notification

**Service**: GL-009 THERMALIQ
**Severity**: SEV2
**Reason**: High error rate after v1.2.3 deployment
**Rollback Version**: v1.2.2
**Estimated Duration**: 15 minutes
**Expected Downtime**: None (rolling rollback)

## Impact
- Brief increase in latency during rollback
- No data loss expected
- No user action required

## Timeline
- **$(date -u +%Y-%m-%dT%H:%M:%SZ)**: Rollback initiated
- **$(date -u -d '+15 minutes' +%Y-%m-%dT%H:%M:%SZ)**: Rollback expected complete

## Communication Channel
Slack: #incident-gl009-rollback-$(date +%Y%m%d)
EOF

# Post to Slack
slack-notify "#incidents" -f rollback_notification.md

# Update status page
curl -X POST https://api.statuspage.io/v1/incidents \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "name": "GL-009: Rolling back deployment",
      "status": "investigating",
      "impact_override": "minor"
    }
  }'
```

**Notification Checklist**:
- [ ] Incident channel created
- [ ] Engineering team notified
- [ ] On-call engineer notified
- [ ] Status page updated
- [ ] Customer success notified (if customer-facing)

---

## Rollback Scenarios

### Scenario 1: Failed Deployment

**Symptoms**:
- Pods not starting
- CrashLoopBackOff
- ImagePullBackOff
- Deployment timeout

**Rollback Procedure**:
```bash
# Check deployment status
kubectl rollout status deployment/thermaliq -n gl-009-production --timeout=60s

# If failed, rollback immediately
kubectl rollout undo deployment/thermaliq -n gl-009-production

# Wait for rollback completion
kubectl rollout status deployment/thermaliq -n gl-009-production

# Verify pods running
kubectl get pods -n gl-009-production -l app=thermaliq

# Check health
curl https://api.greenlang.io/v1/thermaliq/health
```

**Duration**: 2-5 minutes
**Downtime**: None (old version still running)

---

### Scenario 2: Performance Regression

**Symptoms**:
- Increased latency (p95 > 30s)
- High CPU/memory usage
- Timeout rate increased
- Slow queries

**Rollback Procedure**:
```bash
# Confirm performance regression
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[5m])' | \
  jq '.data.result[0].value[1]'

# Compare with baseline (should be < 15s)
# If > 30s, initiate rollback

# Rollback deployment
kubectl rollout undo deployment/thermaliq -n gl-009-production

# Monitor latency during rollback
watch -n 10 'curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode "query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[1m])" | \
  jq ".data.result[0].value[1]"'

# Verify performance restored (p95 < 15s)
```

**Duration**: 5-10 minutes
**Downtime**: None

---

### Scenario 3: Data Corruption

**Symptoms**:
- Incorrect calculation results
- Database constraint violations
- Data integrity errors
- Missing or duplicate records

**Rollback Procedure**:

**CRITICAL**: This requires both deployment AND database rollback

```bash
# 1. IMMEDIATELY scale down to prevent more corruption
kubectl scale deployment/thermaliq -n gl-009-production --replicas=0

# 2. Assess corruption scope
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT COUNT(*) AS corrupted_records
   FROM calculations
   WHERE created_at >= '$(date -u -d '-1 hour' +%Y-%m-%d %H:%M:%S)'
     AND (efficiency < 0 OR efficiency > 100 OR energy_balance_percent > 10);"

# 3. If corruption detected, rollback database
# See "Database Migration Rollback" section

# 4. Restore from last good backup if needed
kubectl exec -it postgres-0 -n gl-009-production -- \
  pg_restore -U thermaliq -d thermaliq_production -c /backups/last_good_backup.dump

# 5. Rollback deployment
kubectl rollout undo deployment/thermaliq -n gl-009-production

# 6. Scale back up
kubectl scale deployment/thermaliq -n gl-009-production --replicas=4

# 7. Verify data integrity
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT COUNT(*) FROM calculations WHERE created_at >= NOW() - INTERVAL '5 minutes';"

# New calculations should have valid data
```

**Duration**: 30-60 minutes
**Downtime**: 5-15 minutes (during scale-down and database restore)

---

### Scenario 4: Security Vulnerability

**Symptoms**:
- CVE published affecting dependencies
- Security scan failure
- Unauthorized access detected
- Data leak detected

**Rollback Procedure**:

**CRITICAL**: Immediate rollback required

```bash
# 1. IMMEDIATELY rollback
kubectl rollout undo deployment/thermaliq -n gl-009-production

# 2. Verify vulnerable version no longer running
kubectl get pods -n gl-009-production -l app=thermaliq -o json | \
  jq '.items[].spec.containers[0].image'

# None should be running vulnerable version

# 3. Check for compromise
kubectl logs -n gl-009-production -l app=thermaliq --tail=10000 --since=1h | \
  grep -i "unauthorized\|attack\|exploit\|breach"

# 4. Audit database access
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT * FROM pg_stat_activity WHERE usename != 'thermaliq';"

# 5. Rotate credentials
kubectl create secret generic thermaliq-secrets \
  -n gl-009-production \
  --from-literal=database-password=$(openssl rand -base64 32) \
  --from-literal=api-key=$(openssl rand -base64 32) \
  --dry-run=client -o yaml | kubectl apply -f -

# 6. Notify security team
slack-notify "#security-incidents" "GL-009 rolled back due to security vulnerability. Investigation required."

# 7. Verify no ongoing compromise
# Monitor for 24 hours
```

**Duration**: 5 minutes (rollback) + 24 hours (monitoring)
**Downtime**: None

---

## Deployment Rollback

### Kubernetes Rolling Rollback

**Standard rollback procedure** (zero downtime):

```bash
# Step 1: Verify current deployment
kubectl get deployment thermaliq -n gl-009-production

# Step 2: Check rollout history
kubectl rollout history deployment/thermaliq -n gl-009-production

# Output:
# REVISION  CHANGE-CAUSE
# 1         Initial deployment
# 2         Update to v1.2.2
# 3         Update to v1.2.3  <- Current (problematic)

# Step 3: Rollback to previous revision
kubectl rollout undo deployment/thermaliq -n gl-009-production

# Step 4: Watch rollout progress
kubectl rollout status deployment/thermaliq -n gl-009-production

# Output:
# Waiting for deployment "thermaliq" rollout to finish: 1 out of 4 new replicas have been updated...
# Waiting for deployment "thermaliq" rollout to finish: 2 out of 4 new replicas have been updated...
# Waiting for deployment "thermaliq" rollout to finish: 3 out of 4 new replicas have been updated...
# deployment "thermaliq" successfully rolled out

# Step 5: Verify pods running correct version
kubectl get pods -n gl-009-production -l app=thermaliq -o json | \
  jq '.items[] | {name: .metadata.name, image: .spec.containers[0].image, ready: .status.conditions[] | select(.type=="Ready") | .status}'

# All should show v1.2.2 image and Ready: "True"

# Step 6: Verify health
curl https://api.greenlang.io/v1/thermaliq/health

# Expected: {"status": "healthy", "version": "1.2.2"}

# Step 7: Test functionality
curl -X POST https://api.greenlang.io/v1/thermaliq/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_id": "FAC-TEST-001",
    "time_period": {
      "start": "2025-11-01T00:00:00Z",
      "end": "2025-11-01T01:00:00Z"
    }
  }'

# Should return successful calculation
```

**Duration**: 5-10 minutes (depending on replica count)
**Downtime**: None (rolling update maintains availability)

---

### Rollback to Specific Revision

If you need to rollback to a specific revision (not just previous):

```bash
# List all revisions
kubectl rollout history deployment/thermaliq -n gl-009-production

# REVISION  CHANGE-CAUSE
# 1         Initial deployment v1.0.0
# 2         Update to v1.2.0
# 3         Update to v1.2.1
# 4         Update to v1.2.2
# 5         Update to v1.2.3  <- Current (problematic)

# View details of specific revision
kubectl rollout history deployment/thermaliq -n gl-009-production --revision=4

# Rollback to revision 4 (v1.2.2)
kubectl rollout undo deployment/thermaliq -n gl-009-production --to-revision=4

# Verify rollback
kubectl rollout status deployment/thermaliq -n gl-009-production
```

---

### Emergency Rollback (Fast)

If you need the **fastest possible rollback** (accepts brief downtime):

```bash
# Step 1: Set image directly to last known good version
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.2.2

# Step 2: Force rollout restart
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Step 3: Wait for completion
kubectl rollout status deployment/thermaliq -n gl-009-production

# This is faster than `rollout undo` but may cause brief unavailability
```

**Duration**: 2-3 minutes
**Downtime**: 30-60 seconds possible

---

### Canary Rollback

If you deployed using canary and need to rollback:

```bash
# Step 1: Check canary status
kubectl get rollout thermaliq -n gl-009-production -o json | \
  jq '.status.canary'

# Step 2: Abort canary rollout
kubectl argo rollouts abort thermaliq -n gl-009-production

# Step 3: Rollback to stable
kubectl argo rollouts undo thermaliq -n gl-009-production

# Step 4: Verify all traffic to stable version
kubectl argo rollouts get rollout thermaliq -n gl-009-production
```

---

## Database Migration Rollback

### Check Migration Status

```bash
# Connect to database
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production

# Check migration version
thermaliq_production=# SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 5;

#    version
# --------------
#  20251126_003  <- Current (problematic?)
#  20251125_002
#  20251124_001
#  20251120_005
#  20251119_004

# Check migration details
thermaliq_production=# \d+ calculations;

# Look for recently added columns or changes
```

---

### Alembic Migration Rollback

If using Alembic for database migrations:

```bash
# Step 1: Check current migration
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  alembic current

# Output: 20251126_003 (head)

# Step 2: Check migration history
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  alembic history

# Step 3: Downgrade to previous migration
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  alembic downgrade -1

# Or to specific revision
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  alembic downgrade 20251125_002

# Step 4: Verify rollback
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  alembic current

# Output: 20251125_002
```

---

### Manual Migration Rollback

If migration doesn't have a down migration or you need manual rollback:

```bash
# Step 1: Create rollback SQL script
cat > migration_rollback.sql <<EOF
-- Rollback migration 20251126_003

BEGIN;

-- Drop new column
ALTER TABLE calculations DROP COLUMN IF EXISTS new_column;

-- Restore old column
ALTER TABLE calculations ADD COLUMN old_column VARCHAR(255);

-- Revert data transformations
UPDATE calculations SET old_format = new_format WHERE ...;

-- Drop new index
DROP INDEX IF EXISTS idx_new_index;

-- Recreate old index
CREATE INDEX idx_old_index ON calculations(old_column);

-- Update migration version
DELETE FROM schema_migrations WHERE version = '20251126_003';

COMMIT;
EOF

# Step 2: Review rollback script
cat migration_rollback.sql

# Step 3: Take database backup
kubectl exec -it postgres-0 -n gl-009-production -- \
  pg_dump -U thermaliq -d thermaliq_production -F c -f /tmp/pre_rollback_backup.dump

# Step 4: Execute rollback
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -f /path/to/migration_rollback.sql

# Step 5: Verify rollback
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "\d+ calculations"

# Check that schema matches expected state
```

---

### Data Migration Rollback

If migration involved data transformations:

```bash
# Step 1: Check if backup exists
kubectl exec -it postgres-0 -n gl-009-production -- \
  ls -lh /backups/ | grep pre_migration

# Step 2: Restore from backup
kubectl exec -it postgres-0 -n gl-009-production -- \
  pg_restore -U thermaliq -d thermaliq_production -c --if-exists \
  /backups/pre_migration_20251126.dump

# Step 3: Verify data restored
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT COUNT(*) FROM calculations WHERE created_at >= '2025-11-26';"

# Compare with expected count

# Step 4: Rebuild indexes
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "REINDEX DATABASE thermaliq_production;"

# Step 5: Update statistics
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "ANALYZE;"
```

---

### Migration Rollback with Zero Downtime

For critical systems that cannot tolerate downtime:

```bash
# Step 1: Create rollback migration (additive)
# Instead of dropping column, mark as deprecated
# Instead of removing data, keep both old and new

# Step 2: Deploy rollback migration
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  alembic upgrade head

# Step 3: Update application to use old schema
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.2.2

# Step 4: Wait for rollout
kubectl rollout status deployment/thermaliq -n gl-009-production

# Step 5: Clean up new schema (after verification)
# Schedule for maintenance window
```

---

## Configuration Rollback

### ConfigMap Rollback

```bash
# Step 1: Backup current config
kubectl get configmap thermaliq-config -n gl-009-production -o yaml > \
  backups/config_current.yaml

# Step 2: Restore previous config
kubectl apply -f backups/config_20251125.yaml

# Step 3: Verify config updated
kubectl get configmap thermaliq-config -n gl-009-production -o yaml | \
  diff - backups/config_20251125.yaml

# Should show no differences

# Step 4: Reload configuration in pods
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Or if hot reload supported:
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  kill -HUP 1  # Send SIGHUP to PID 1 to reload config

# Step 5: Verify config applied
kubectl logs -n gl-009-production -l app=thermaliq --tail=50 | \
  grep "Configuration loaded"
```

---

### Environment Variable Rollback

```bash
# Step 1: Check current environment variables
kubectl get deployment thermaliq -n gl-009-production -o yaml | \
  grep -A 50 "env:"

# Step 2: Remove problematic environment variable
kubectl set env deployment/thermaliq -n gl-009-production \
  PROBLEMATIC_VAR-

# Note the trailing "-" to remove the variable

# Step 3: Or set back to previous value
kubectl set env deployment/thermaliq -n gl-009-production \
  CONFIG_VALUE=previous_value

# Step 4: Wait for rollout
kubectl rollout status deployment/thermaliq -n gl-009-production

# Step 5: Verify environment variables
kubectl exec -it deployment/thermaliq -n gl-009-production -- env | \
  grep THERMALIQ
```

---

### Secret Rollback

```bash
# Step 1: Backup current secret
kubectl get secret thermaliq-secrets -n gl-009-production -o yaml > \
  backups/secrets_current.yaml

# Step 2: Restore previous secret
kubectl apply -f backups/secrets_20251125.yaml

# Step 3: Restart pods to pick up new secret
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Step 4: Verify pods using new secret
kubectl logs -n gl-009-production -l app=thermaliq --tail=50 | \
  grep "Authentication\|Connection"

# Should show successful connections with restored credentials
```

---

## Cache Invalidation

After rollback, cache may contain data from new version:

### Redis Cache Flush

```bash
# Step 1: Flush all cached data
kubectl exec -it redis-0 -n gl-009-production -- redis-cli FLUSHDB

# Step 2: Verify cache cleared
kubectl exec -it redis-0 -n gl-009-production -- redis-cli DBSIZE

# Should return: (integer) 0

# Step 3: Warm cache with critical data
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/warm_cache.py --facilities FAC-001,FAC-002,FAC-003

# Step 4: Verify cache warming
kubectl logs -n gl-009-production -l app=thermaliq --tail=100 | \
  grep "Cache warming"
```

---

### Selective Cache Invalidation

If you only need to invalidate specific keys:

```bash
# Step 1: Identify keys to invalidate
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli KEYS "calculation:*:v1.2.3:*"

# Step 2: Delete matching keys
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli --scan --pattern "calculation:*:v1.2.3:*" | \
  xargs kubectl exec -it redis-0 -n gl-009-production -- redis-cli DEL

# Step 3: Verify deletion
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli KEYS "calculation:*:v1.2.3:*"

# Should return: (empty array)
```

---

### Application-Level Cache Invalidation

```bash
# If application has in-memory cache:

# Step 1: Call cache invalidation endpoint
curl -X POST https://api.greenlang.io/v1/thermaliq/admin/cache/invalidate \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"pattern": "calculation:*"}'

# Step 2: Or restart pods to clear in-memory cache
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Step 3: Verify cache cleared
curl https://api.greenlang.io/v1/thermaliq/admin/cache/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN" | \
  jq .

# Should show: {"size": 0, "hits": 0, "misses": 0}
```

---

## Post-Rollback Verification

### Verification Checklist

Complete this checklist after every rollback:

**Deployment Verification**:
```bash
# 1. All pods running and ready
kubectl get pods -n gl-009-production -l app=thermaliq

# All should show READY 1/1

# 2. Correct version deployed
kubectl get deployment thermaliq -n gl-009-production -o jsonpath='{.spec.template.spec.containers[0].image}'

# Should show: ghcr.io/greenlang/thermaliq:v1.2.2

# 3. No pods restarting
kubectl get pods -n gl-009-production -l app=thermaliq -o json | \
  jq '.items[] | {pod: .metadata.name, restarts: .status.containerStatuses[0].restartCount}'

# Restart count should not be increasing

# 4. Health check passing
curl https://api.greenlang.io/v1/thermaliq/health

# Expected: {"status": "healthy"}
```

**Functionality Verification**:
```bash
# 1. Submit test calculation
curl -X POST https://api.greenlang.io/v1/thermaliq/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @test_calculation.json | jq .

# Should return: {"calculation_id": "CALC-XXX", "status": "completed"}

# 2. Retrieve calculation result
calc_id=$(curl -X POST https://api.greenlang.io/v1/thermaliq/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @test_calculation.json | jq -r '.calculation_id')

curl https://api.greenlang.io/v1/thermaliq/calculations/$calc_id \
  -H "Authorization: Bearer $TOKEN" | jq .

# Should return complete calculation with efficiency value

# 3. Test Sankey generation
curl https://api.greenlang.io/v1/thermaliq/calculations/$calc_id/sankey \
  -H "Authorization: Bearer $TOKEN" > sankey_test.svg

file sankey_test.svg
# Should return: SVG image data

# 4. Test benchmark comparison
curl https://api.greenlang.io/v1/thermaliq/calculations/$calc_id/benchmark \
  -H "Authorization: Bearer $TOKEN" | jq .

# Should return benchmark comparison data
```

**Performance Verification**:
```bash
# 1. Check error rate (should be < 1%)
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_calculation_errors_total[5m])/rate(thermaliq_calculation_requests_total[5m])' | \
  jq '.data.result[0].value[1]'

# 2. Check latency (p95 should be < 15s)
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[5m])' | \
  jq '.data.result[0].value[1]'

# 3. Check throughput (should be normal)
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_calculation_requests_total[5m])' | \
  jq '.data.result[0].value[1]'

# 4. Check resource usage (CPU/memory should be normal)
kubectl top pods -n gl-009-production -l app=thermaliq
```

**Database Verification**:
```bash
# 1. Check database connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  psql $DATABASE_URL -c "SELECT 1"

# 2. Check migration version
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"

# Should show expected version (not the rolled-back version)

# 3. Verify data integrity
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT COUNT(*) AS total,
          COUNT(CASE WHEN efficiency < 0 OR efficiency > 100 THEN 1 END) AS invalid
   FROM calculations
   WHERE created_at >= NOW() - INTERVAL '1 hour';"

# invalid count should be 0

# 4. Check database performance
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT query, mean_exec_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 5;"

# Mean exec time should be < 1000ms
```

**Integration Verification**:
```bash
# 1. Check energy meter connectivity
kubectl logs -n gl-009-production -l app=thermaliq --tail=100 | \
  grep "EnergyMeter" | \
  grep -i "connected\|authenticated"

# Should show successful connections

# 2. Check historian connectivity
kubectl logs -n gl-009-production -l app=thermaliq --tail=100 | \
  grep "Historian" | \
  grep -i "connected\|query.*success"

# Should show successful queries

# 3. Check ERP connector status
curl https://api.greenlang.io/v1/erp-connector/status | jq .

# Should show: {"thermaliq_integration": "healthy"}

# 4. Test end-to-end flow
# Submit calculation -> Check ERP export -> Verify data in ERP system
```

**Monitoring Verification**:
```bash
# 1. Check Prometheus scraping
curl -s http://prometheus:9090/api/v1/targets | \
  jq '.data.activeTargets[] | select(.labels.job=="thermaliq") | {health: .health, lastScrape: .lastScrape}'

# All targets should show health: "up"

# 2. Check Grafana dashboards
curl -s https://grafana.greenlang.io/api/dashboards/uid/thermaliq-prod | jq .

# Dashboard should be accessible

# 3. Check alerting rules
curl -s http://prometheus:9090/api/v1/rules | \
  jq '.data.groups[] | select(.name=="thermaliq") | .rules[] | {alert: .name, state: .state}'

# No alerts should be firing

# 4. Check logs aggregation
curl -s http://loki:3100/loki/api/v1/query \
  --data-urlencode 'query={app="thermaliq"}' | \
  jq '.data.result | length'

# Should return > 0 (logs being collected)
```

---

## Communication Plan

### During Rollback

**Initial Notification** (T+0 minutes):
```markdown
## Rollback Initiated

**Service**: GL-009 THERMALIQ
**Time**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Reason**: [Brief reason]
**Target Version**: v1.2.2
**Expected Duration**: 15 minutes

**Status**: In progress
**Impact**: [Brief user impact]

Updates will be provided every 5 minutes.
```

**Progress Updates** (Every 5 minutes):
```markdown
## Rollback Update - T+5min

**Completed Steps**:
- [x] Backup created
- [x] Deployment rolled back
- [ ] Verification in progress

**Current Status**: Pods restarting with previous version
**Issues**: None
**Next Steps**: Complete verification, resume normal operations
```

**Completion Notification** (T+15 minutes):
```markdown
## Rollback Complete

**Service**: GL-009 THERMALIQ
**Completed**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Duration**: 15 minutes
**Version**: v1.2.2

**Verification Results**:
- ✓ All pods running
- ✓ Health checks passing
- ✓ Error rate < 1%
- ✓ Performance restored
- ✓ Integrations working

**Root Cause**: [Brief explanation]
**Next Steps**: Post-incident review scheduled for [date/time]

Service is operating normally. No user action required.
```

---

### Post-Rollback Communication

**Internal Post-Mortem Email**:
```
Subject: [Post-Mortem] GL-009 THERMALIQ Rollback - [Date]

Team,

We performed a rollback of GL-009 THERMALIQ from v1.2.3 to v1.2.2 on [date] due to [reason].

TIMELINE:
- [Time]: Deployment of v1.2.3 completed
- [Time]: High error rate detected
- [Time]: Rollback initiated
- [Time]: Rollback completed
- [Time]: Service verified healthy

ROOT CAUSE:
[Detailed explanation]

IMPACT:
- Duration: 15 minutes
- Error rate: 25% during incident
- Affected calculations: ~150
- User complaints: 3

ACTION ITEMS:
1. [Action 1] - Owner: [Name] - Due: [Date]
2. [Action 2] - Owner: [Name] - Due: [Date]
3. [Action 3] - Owner: [Name] - Due: [Date]

LESSONS LEARNED:
- [Lesson 1]
- [Lesson 2]
- [Lesson 3]

Post-incident review meeting: [Date/Time]
```

**Customer Communication** (if customer-facing):
```
Subject: Service Disruption - GL-009 THERMALIQ - [Date]

Dear Valued Customer,

We experienced a brief service disruption affecting the THERMALIQ Thermal Efficiency Calculator on [date] from [time] to [time] UTC.

WHAT HAPPENED:
[Customer-friendly explanation]

IMPACT TO YOUR ACCOUNT:
- Calculation requests may have failed during the incident
- Approximately [X] calculations were affected
- All data has been preserved and no action is required on your part

RESOLUTION:
We rolled back to a previous stable version and have restored normal service.

PREVENTION:
[What we're doing to prevent this in the future]

We sincerely apologize for any inconvenience. If you have questions, please contact support@greenlang.io.

Thank you for your patience.

The GreenLang Team
```

---

## Rollback Decision Matrix

Use this matrix to decide whether to rollback:

| Error Rate | Latency | Data Corruption | Security Issue | Decision |
|------------|---------|-----------------|----------------|----------|
| > 50% | Any | Any | Any | **Immediate Rollback** |
| 10-50% | > 2x baseline | No | No | **Rollback** (within 15 min) |
| 5-10% | 1.5-2x baseline | No | No | **Evaluate** (30 min window) |
| 1-5% | < 1.5x baseline | No | No | **Monitor** (fix forward) |
| < 1% | Normal | No | No | **No Action** |
| Any | Any | **Yes** | Any | **Immediate Rollback** |
| Any | Any | Any | **Yes** | **Immediate Rollback** |

**Decision Factors**:
- Error rate
- Latency increase
- Data corruption
- Security vulnerability
- Business impact
- Availability of fix
- Time to fix forward vs rollback

**When to Fix Forward** (instead of rollback):
- Error rate < 5%
- Simple configuration fix available
- Rollback would cause more disruption
- Database migration cannot be easily rolled back
- Issue only affects non-critical functionality

**When to Rollback**:
- Error rate > 10%
- Unknown root cause
- Data integrity at risk
- Security vulnerability
- No quick fix available
- Customer-facing functionality broken

---

## Appendix: Rollback Scripts

### Automated Rollback Script

```bash
#!/bin/bash
# rollback.sh - Automated rollback script for GL-009 THERMALIQ

set -euo pipefail

# Configuration
NAMESPACE="gl-009-production"
DEPLOYMENT="thermaliq"
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Functions
log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1"
}

notify_slack() {
    curl -X POST "$SLACK_WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$1\"}"
}

# Main rollback procedure
main() {
    log "Starting rollback procedure..."
    notify_slack "🔄 GL-009 THERMALIQ rollback initiated"

    # Step 1: Backup current state
    log "Creating backup..."
    kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o yaml > \
        "backups/deployment_$(date +%Y%m%d_%H%M%S).yaml"

    # Step 2: Rollback deployment
    log "Rolling back deployment..."
    kubectl rollout undo deployment/"$DEPLOYMENT" -n "$NAMESPACE"

    # Step 3: Wait for rollout
    log "Waiting for rollout to complete..."
    kubectl rollout status deployment/"$DEPLOYMENT" -n "$NAMESPACE" --timeout=5m

    # Step 4: Verify health
    log "Verifying health..."
    sleep 10
    health_status=$(curl -s https://api.greenlang.io/v1/thermaliq/health | jq -r '.status')

    if [ "$health_status" != "healthy" ]; then
        log "ERROR: Health check failed after rollback!"
        notify_slack "❌ GL-009 THERMALIQ rollback failed - health check not passing"
        exit 1
    fi

    # Step 5: Verify functionality
    log "Testing functionality..."
    calc_response=$(curl -s -X POST https://api.greenlang.io/v1/thermaliq/calculate \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d @test_calculation.json)

    calc_id=$(echo "$calc_response" | jq -r '.calculation_id')

    if [ -z "$calc_id" ] || [ "$calc_id" == "null" ]; then
        log "ERROR: Test calculation failed!"
        notify_slack "❌ GL-009 THERMALIQ rollback completed but functionality test failed"
        exit 1
    fi

    # Success
    log "Rollback completed successfully!"
    notify_slack "✅ GL-009 THERMALIQ rollback completed successfully"

    # Display version
    current_version=$(kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')
    log "Current version: $current_version"
}

# Run main procedure
main "$@"
```

Usage:
```bash
chmod +x rollback.sh
./rollback.sh
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-26
**Next Review**: 2025-12-26
**Owner**: GreenLang SRE Team
