# GL-008 SteamTrapInspector - Rollback Procedure

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Owner:** Platform Operations Team
**Critical Process:** Follow exactly as documented

---

## Table of Contents

1. [Rollback Decision Criteria](#rollback-decision-criteria)
2. [Pre-Rollback Checklist](#pre-rollback-checklist)
3. [Rollback Scenarios](#rollback-scenarios)
4. [Application Rollback](#application-rollback)
5. [Database Rollback](#database-rollback)
6. [ML Model Rollback](#ml-model-rollback)
7. [Configuration Rollback](#configuration-rollback)
8. [Verification Procedures](#verification-procedures)
9. [Post-Rollback Actions](#post-rollback-actions)
10. [Communication Templates](#communication-templates)

---

## Rollback Decision Criteria

### When to Rollback

Execute rollback when ANY of the following occur:

**Severity 1 - Immediate Rollback (No Approval Required):**
- Complete service outage (all inspections failing)
- Data corruption detected
- Security vulnerability actively exploited
- Error rate >50% of requests
- Database connection failures >80%

**Severity 2 - Rollback After Assessment (Engineering Manager Approval):**
- Error rate 20-50% of requests
- Significant performance degradation (>3x baseline latency)
- False positive rate >40%
- Multiple customer escalations
- SLA breach imminent

**Severity 3 - Consider Rollback (On-Call Engineer Decision):**
- Error rate 10-20% of requests
- Performance degradation 2-3x baseline
- Feature not working as expected but non-critical
- Single customer escalation

### When NOT to Rollback

**Do NOT rollback if:**
- Issue affects <5% of requests
- Workaround available and documented
- Forward fix available within 30 minutes
- Rollback would cause more disruption than current issue
- Database migrations cannot be safely reversed

**Alternative Actions:**
- Feature flag disable
- Traffic routing to specific pods
- Rate limiting problematic endpoints
- Temporary configuration changes

---

## Pre-Rollback Checklist

### Critical Questions (Answer ALL before proceeding)

```
[ ] What is the exact issue? ______________________________________
[ ] When did the issue start? ______________________________________
[ ] What deployment/change triggered it? ___________________________
[ ] How many customers/sites affected? _____________________________
[ ] What is current error rate? ____________________________________
[ ] Are there database schema changes? [ ] Yes [ ] No
[ ] Are there data migrations? [ ] Yes [ ] No
[ ] Is ML model rollback required? [ ] Yes [ ] No
[ ] What is the target rollback version? ___________________________
[ ] Who authorized this rollback? __________________________________
[ ] Incident ticket number: ________________________________________
```

### Pre-Rollback Commands

```bash
# 1. Document current state
echo "Capturing current state at $(date)"

# Current deployment version
kubectl get deployment steam-trap-inspector -n greenlang-gl008 -o yaml > current-deployment-$(date +%Y%m%d-%H%M%S).yaml

# Current replica count
kubectl get deployment steam-trap-inspector -n greenlang-gl008 -o jsonpath='{.spec.replicas}' > current-replicas.txt

# Current ConfigMaps
kubectl get configmap -n greenlang-gl008 -o yaml > current-configmaps-$(date +%Y%m%d-%H%M%S).yaml

# Current database schema version
psql $DB_URL -c "SELECT version, applied_at FROM schema_migrations ORDER BY applied_at DESC LIMIT 5;" > current-schema.txt

# Current ML model version
kubectl get configmap ml-model-config -n greenlang-gl008 -o yaml | grep ML_MODEL_VERSION > current-model-version.txt

# 2. Create backup snapshot (if not already exists)
./scripts/create-backup-snapshot.sh --label="pre-rollback-$(date +%Y%m%d-%H%M%S)"

# 3. Verify target version exists
TARGET_VERSION="v2.4.1"  # Replace with actual target version
docker pull greenlang/steam-trap-inspector:$TARGET_VERSION

# 4. Alert stakeholders
./scripts/send-rollback-notification.sh \
  --severity=P1 \
  --target-version=$TARGET_VERSION \
  --reason="[Brief description]"

# 5. Put service in maintenance mode (if required for database rollback)
# kubectl scale deployment/steam-trap-inspector -n greenlang-gl008 --replicas=0
```

### Rollback Team Assembly

**Minimum Required Roles:**
- Incident Commander (IC): ______________________
- Platform Engineer: ____________________________
- Database Administrator (if DB changes): ________
- ML Engineer (if model changes): _______________

**Communication Channels:**
- Incident Slack channel: #incident-gl008-______
- Video call: _________________________________
- Status page: https://status.greenlang.io

---

## Rollback Scenarios

### Scenario 1: Fast Rollback (5 minutes)

**When to Use:**
- Application-only changes
- No database schema changes
- No data migrations
- Configuration changes only

**Conditions:**
- Same database schema version
- Same ML model version (or model is backward compatible)
- No breaking API changes

**Execution:**

```bash
#!/bin/bash
# Quick rollback script

set -e  # Exit on error

IC_NAME="[Incident Commander Name]"
TARGET_VERSION="v2.4.1"  # Replace with target version
NAMESPACE="greenlang-gl008"

echo "=== FAST ROLLBACK INITIATED ==="
echo "Incident Commander: $IC_NAME"
echo "Target Version: $TARGET_VERSION"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# Step 1: Verify target version exists (30 seconds)
echo "[Step 1/5] Verifying target version..."
if ! docker pull greenlang/steam-trap-inspector:$TARGET_VERSION 2>/dev/null; then
  echo "ERROR: Target version $TARGET_VERSION not found in registry"
  exit 1
fi
echo "✓ Target version verified"

# Step 2: Set image to target version (30 seconds)
echo "[Step 2/5] Rolling back deployment..."
kubectl set image deployment/steam-trap-inspector \
  steam-trap-inspector=greenlang/steam-trap-inspector:$TARGET_VERSION \
  -n $NAMESPACE

# Step 3: Wait for rollout (3 minutes)
echo "[Step 3/5] Waiting for rollout to complete..."
kubectl rollout status deployment/steam-trap-inspector -n $NAMESPACE --timeout=180s

# Step 4: Quick health check (30 seconds)
echo "[Step 4/5] Running health check..."
sleep 10  # Give pods time to stabilize

HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.greenlang.io/v1/steam-trap/health)
if [ "$HEALTH_STATUS" -eq 200 ]; then
  echo "✓ Health check passed"
else
  echo "⚠ Health check returned status: $HEALTH_STATUS"
fi

# Step 5: Verify error rate decreased (30 seconds)
echo "[Step 5/5] Checking error rate..."
ERROR_COUNT=$(kubectl logs -n $NAMESPACE -l app=steam-trap-inspector --since=2m | grep ERROR | wc -l)
echo "Errors in last 2 minutes: $ERROR_COUNT"

echo ""
echo "=== FAST ROLLBACK COMPLETE ==="
echo "Completion Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""
echo "Next steps:"
echo "1. Monitor for 15 minutes: ./scripts/post-rollback-monitor.sh"
echo "2. Update incident ticket with rollback confirmation"
echo "3. Schedule post-incident review"
```

**Expected Duration:** 5 minutes

**Verification:**

```bash
# Verify rollback successful
kubectl get pods -n greenlang-gl008 -l app=steam-trap-inspector

# Check all pods are running
kubectl get pods -n greenlang-gl008 -l app=steam-trap-inspector --field-selector=status.phase=Running | wc -l

# Test API endpoint
curl https://api.greenlang.io/v1/steam-trap/health | jq '.'

# Check recent errors
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --tail=50 | grep -E "(ERROR|FATAL)"
```

---

### Scenario 2: Standard Rollback (15 minutes)

**When to Use:**
- ConfigMap or Secret changes included
- Feature flags need to be disabled
- Cache needs to be cleared
- Multiple components affected

**Conditions:**
- No database schema changes
- May include ML model rollback
- Configuration changes need reverting

**Execution:**

```bash
#!/bin/bash
# Standard rollback script

set -e

IC_NAME="[Incident Commander Name]"
TARGET_VERSION="v2.4.1"
NAMESPACE="greenlang-gl008"
ROLLBACK_REASON="[Brief reason]"

echo "=== STANDARD ROLLBACK INITIATED ==="
echo "Incident Commander: $IC_NAME"
echo "Target Version: $TARGET_VERSION"
echo "Reason: $ROLLBACK_REASON"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# Step 1: Backup current state (1 minute)
echo "[Step 1/10] Backing up current state..."
kubectl get deployment steam-trap-inspector -n $NAMESPACE -o yaml > rollback-backup-deployment-$(date +%Y%m%d-%H%M%S).yaml
kubectl get configmap -n $NAMESPACE -o yaml > rollback-backup-configmaps-$(date +%Y%m%d-%H%M%S).yaml
echo "✓ Backup created"

# Step 2: Identify previous ConfigMap versions (1 minute)
echo "[Step 2/10] Identifying previous configurations..."
PREV_CONFIG_VERSION=$(kubectl rollout history deployment/steam-trap-inspector -n $NAMESPACE | tail -2 | head -1 | awk '{print $1}')
echo "Previous config version: $PREV_CONFIG_VERSION"

# Step 3: Disable feature flags (1 minute)
echo "[Step 3/10] Disabling new feature flags..."
kubectl patch configmap feature-flags -n $NAMESPACE --type merge -p '{
  "data": {
    "ENABLE_NEW_ACOUSTIC_ALGORITHM": "false",
    "ENABLE_ENHANCED_THERMAL_DETECTION": "false",
    "ENABLE_EXPERIMENTAL_FEATURES": "false"
  }
}'
echo "✓ Feature flags disabled"

# Step 4: Rollback ML model if needed (2 minutes)
echo "[Step 4/10] Checking ML model version..."
CURRENT_MODEL=$(kubectl get configmap ml-model-config -n $NAMESPACE -o jsonpath='{.data.ML_MODEL_VERSION}')
TARGET_MODEL="v2.4.0"  # Update as needed

if [ "$CURRENT_MODEL" != "$TARGET_MODEL" ]; then
  echo "Rolling back ML model from $CURRENT_MODEL to $TARGET_MODEL..."
  kubectl patch configmap ml-model-config -n $NAMESPACE --type merge -p "{
    \"data\": {
      \"ML_MODEL_VERSION\": \"$TARGET_MODEL\"
    }
  }"
  echo "✓ ML model version updated"
else
  echo "ML model version unchanged"
fi

# Step 5: Clear cache (1 minute)
echo "[Step 5/10] Clearing cache..."
kubectl exec -n $NAMESPACE deployment/redis -- redis-cli FLUSHDB
echo "✓ Cache cleared"

# Step 6: Rollback application deployment (3 minutes)
echo "[Step 6/10] Rolling back application deployment..."
kubectl set image deployment/steam-trap-inspector \
  steam-trap-inspector=greenlang/steam-trap-inspector:$TARGET_VERSION \
  -n $NAMESPACE

# Step 7: Wait for rollout (3 minutes)
echo "[Step 7/10] Waiting for rollout..."
kubectl rollout status deployment/steam-trap-inspector -n $NAMESPACE --timeout=180s
echo "✓ Rollout complete"

# Step 8: Restart dependent services (2 minutes)
echo "[Step 8/10] Restarting dependent services..."
kubectl rollout restart deployment/api-gateway -n $NAMESPACE
kubectl rollout restart deployment/worker -n $NAMESPACE
kubectl rollout status deployment/api-gateway -n $NAMESPACE --timeout=120s
kubectl rollout status deployment/worker -n $NAMESPACE --timeout=120s
echo "✓ Dependent services restarted"

# Step 9: Health check (1 minute)
echo "[Step 9/10] Running health checks..."
sleep 15  # Allow time for stabilization

HEALTH_CHECK=$(curl -s https://api.greenlang.io/v1/steam-trap/health)
if echo "$HEALTH_CHECK" | jq -e '.status == "healthy"' > /dev/null; then
  echo "✓ Health check passed"
else
  echo "⚠ Health check WARNING: $HEALTH_CHECK"
fi

# Step 10: Smoke tests (1 minute)
echo "[Step 10/10] Running smoke tests..."
./scripts/smoke-test.sh --quick
echo "✓ Smoke tests complete"

echo ""
echo "=== STANDARD ROLLBACK COMPLETE ==="
echo "Completion Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""
echo "Post-rollback monitoring initiated (30 minutes)"
./scripts/post-rollback-monitor.sh --duration=30m &
```

**Expected Duration:** 15 minutes

---

### Scenario 3: Full Rollback with Database (1 hour)

**When to Use:**
- Database schema changes present
- Data migrations executed
- Breaking changes to data model
- Multiple interconnected changes

**DANGER:** Database rollbacks are HIGH RISK. Ensure DBA approval.

**Pre-Execution Requirements:**

```bash
# STOP: Answer these questions before proceeding
echo "DATABASE ROLLBACK CHECKLIST:"
echo "[ ] DBA approval obtained? __________"
echo "[ ] Database backup verified? __________"
echo "[ ] Point-in-time recovery tested? __________"
echo "[ ] Data loss acceptable? __________"
echo "[ ] Downtime window approved? __________"
echo "[ ] Customer notification sent? __________"
```

**Execution:**

```bash
#!/bin/bash
# Full rollback with database changes

set -e

IC_NAME="[Incident Commander Name]"
DBA_NAME="[Database Administrator Name]"
TARGET_VERSION="v2.4.1"
TARGET_SCHEMA_VERSION="20251115_1430"  # Update with actual version
NAMESPACE="greenlang-gl008"

echo "=== FULL ROLLBACK WITH DATABASE INITIATED ==="
echo "Incident Commander: $IC_NAME"
echo "Database Administrator: $DBA_NAME"
echo "Target Application Version: $TARGET_VERSION"
echo "Target Schema Version: $TARGET_SCHEMA_VERSION"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""
echo "⚠️  WARNING: This procedure includes database rollback"
echo "⚠️  Estimated downtime: 15-30 minutes"
echo ""
read -p "Type 'CONFIRM ROLLBACK' to proceed: " CONFIRMATION

if [ "$CONFIRMATION" != "CONFIRM ROLLBACK" ]; then
  echo "Rollback cancelled"
  exit 1
fi

# Phase 1: Pre-Rollback (10 minutes)
echo ""
echo "=== PHASE 1: PRE-ROLLBACK ==="

# Step 1: Create database backup (5 minutes)
echo "[Step 1/15] Creating database backup..."
BACKUP_NAME="pre-rollback-$(date +%Y%m%d-%H%M%S)"
pg_dump $DB_URL | gzip > backups/${BACKUP_NAME}.sql.gz
BACKUP_SIZE=$(ls -lh backups/${BACKUP_NAME}.sql.gz | awk '{print $5}')
echo "✓ Backup created: ${BACKUP_NAME}.sql.gz ($BACKUP_SIZE)"

# Step 2: Verify backup integrity (1 minute)
echo "[Step 2/15] Verifying backup integrity..."
gunzip -t backups/${BACKUP_NAME}.sql.gz
echo "✓ Backup integrity verified"

# Step 3: Put service in maintenance mode (2 minutes)
echo "[Step 3/15] Entering maintenance mode..."
kubectl scale deployment/steam-trap-inspector -n $NAMESPACE --replicas=0
kubectl scale deployment/api-gateway -n $NAMESPACE --replicas=0
kubectl scale deployment/worker -n $NAMESPACE --replicas=0

# Wait for pods to terminate
kubectl wait --for=delete pod -l app=steam-trap-inspector -n $NAMESPACE --timeout=120s
echo "✓ Service in maintenance mode"

# Step 4: Update status page (1 minute)
echo "[Step 4/15] Updating status page..."
curl -X POST https://api.statuspage.io/v1/pages/$STATUSPAGE_ID/incidents \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "name": "GL-008 Maintenance - Rollback in Progress",
      "status": "investigating",
      "impact_override": "maintenance",
      "body": "We are performing a rollback of GL-008 Steam Trap Inspector. Service will be unavailable for approximately 30 minutes."
    }
  }'
echo "✓ Status page updated"

# Step 5: Snapshot current database state (1 minute)
echo "[Step 5/15] Capturing current schema version..."
CURRENT_SCHEMA=$(psql $DB_URL -t -c "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1;")
echo "Current schema version: $CURRENT_SCHEMA"

# Phase 2: Database Rollback (20 minutes)
echo ""
echo "=== PHASE 2: DATABASE ROLLBACK ==="

# Step 6: Check for reversible migrations (2 minutes)
echo "[Step 6/15] Checking migration reversibility..."
./scripts/check-migration-reversibility.sh --from=$CURRENT_SCHEMA --to=$TARGET_SCHEMA_VERSION

# Step 7: Run down migrations (10 minutes)
echo "[Step 7/15] Running down migrations..."
echo "This may take several minutes for large datasets..."

# Calculate migrations to rollback
MIGRATIONS_TO_ROLLBACK=$(psql $DB_URL -t -c "
  SELECT version
  FROM schema_migrations
  WHERE applied_at > (
    SELECT applied_at FROM schema_migrations WHERE version = '$TARGET_SCHEMA_VERSION'
  )
  ORDER BY applied_at DESC;
")

echo "Migrations to rollback:"
echo "$MIGRATIONS_TO_ROLLBACK"

# Execute down migrations
for migration in $MIGRATIONS_TO_ROLLBACK; do
  echo "Rolling back migration: $migration"
  ./scripts/migrate.sh down --version=$migration
done

echo "✓ Database migrations rolled back"

# Step 8: Verify database schema (2 minutes)
echo "[Step 8/15] Verifying database schema..."
ACTUAL_SCHEMA=$(psql $DB_URL -t -c "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1;")

if [ "$ACTUAL_SCHEMA" == "$TARGET_SCHEMA_VERSION" ]; then
  echo "✓ Schema version verified: $TARGET_SCHEMA_VERSION"
else
  echo "❌ ERROR: Schema mismatch. Expected $TARGET_SCHEMA_VERSION, got $ACTUAL_SCHEMA"
  echo "ROLLBACK FAILED - Manual intervention required"
  exit 1
fi

# Step 9: Restore data if needed (5 minutes)
echo "[Step 9/15] Checking for data restoration requirements..."
# This step is conditional based on migration type
# Add data restoration logic here if needed

# Step 10: Database health check (1 minute)
echo "[Step 10/15] Running database health check..."
psql $DB_URL -c "
  SELECT COUNT(*) as trap_count FROM traps;
  SELECT COUNT(*) as inspection_count FROM trap_inspections;
  SELECT COUNT(*) as sensor_count FROM sensors;
"
echo "✓ Database health check passed"

# Phase 3: Application Rollback (15 minutes)
echo ""
echo "=== PHASE 3: APPLICATION ROLLBACK ==="

# Step 11: Rollback ConfigMaps (2 minutes)
echo "[Step 11/15] Rolling back ConfigMaps..."
kubectl apply -f rollback-configs/configmaps-$TARGET_VERSION.yaml -n $NAMESPACE
echo "✓ ConfigMaps rolled back"

# Step 12: Rollback application deployment (3 minutes)
echo "[Step 12/15] Rolling back application..."
kubectl set image deployment/steam-trap-inspector \
  steam-trap-inspector=greenlang/steam-trap-inspector:$TARGET_VERSION \
  -n $NAMESPACE

# Scale back up to normal replica count
NORMAL_REPLICAS=$(cat current-replicas.txt)
kubectl scale deployment/steam-trap-inspector -n $NAMESPACE --replicas=$NORMAL_REPLICAS
kubectl scale deployment/api-gateway -n $NAMESPACE --replicas=3
kubectl scale deployment/worker -n $NAMESPACE --replicas=5

echo "✓ Application deployment rolled back"

# Step 13: Wait for pods to be ready (5 minutes)
echo "[Step 13/15] Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=steam-trap-inspector -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=api-gateway -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=worker -n $NAMESPACE --timeout=300s
echo "✓ All pods ready"

# Phase 4: Verification (15 minutes)
echo ""
echo "=== PHASE 4: VERIFICATION ==="

# Step 14: Comprehensive health check (5 minutes)
echo "[Step 14/15] Running comprehensive health checks..."
./scripts/health-check.sh --comprehensive

# Step 15: Smoke tests (10 minutes)
echo "[Step 15/15] Running full smoke test suite..."
./scripts/smoke-test.sh --full

echo ""
echo "=== FULL ROLLBACK COMPLETE ==="
echo "Completion Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""
echo "Service restored. Monitoring for 60 minutes..."
./scripts/post-rollback-monitor.sh --duration=60m &

# Update status page
curl -X PATCH https://api.statuspage.io/v1/pages/$STATUSPAGE_ID/incidents/$INCIDENT_ID \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "status": "resolved",
      "body": "Rollback completed successfully. Service has been restored."
    }
  }'
```

**Expected Duration:** 60 minutes (includes 15-30 min downtime)

---

## Application Rollback

### Kubernetes Deployment Rollback

```bash
# Method 1: Rollback to previous revision
kubectl rollout undo deployment/steam-trap-inspector -n greenlang-gl008

# Method 2: Rollback to specific revision
kubectl rollout history deployment/steam-trap-inspector -n greenlang-gl008
kubectl rollout undo deployment/steam-trap-inspector -n greenlang-gl008 --to-revision=42

# Method 3: Set specific image version
kubectl set image deployment/steam-trap-inspector \
  steam-trap-inspector=greenlang/steam-trap-inspector:v2.4.1 \
  -n greenlang-gl008

# Monitor rollback progress
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008

# Verify pod versions
kubectl get pods -n greenlang-gl008 -l app=steam-trap-inspector \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}'
```

### Multi-Component Rollback

```bash
# Rollback all GL-008 components together
COMPONENTS=("steam-trap-inspector" "api-gateway" "worker" "sensor-gateway")
TARGET_VERSION="v2.4.1"

for component in "${COMPONENTS[@]}"; do
  echo "Rolling back $component to $TARGET_VERSION..."
  kubectl set image deployment/$component \
    $component=greenlang/$component:$TARGET_VERSION \
    -n greenlang-gl008
done

# Wait for all rollouts
for component in "${COMPONENTS[@]}"; do
  kubectl rollout status deployment/$component -n greenlang-gl008 --timeout=300s
done
```

---

## Database Rollback

### Schema Migration Rollback

```bash
# Check current migration version
psql $DB_URL -c "
  SELECT version, applied_at, description
  FROM schema_migrations
  ORDER BY applied_at DESC
  LIMIT 5;
"

# Rollback last migration
./scripts/migrate.sh down

# Rollback to specific version
./scripts/migrate.sh down --target-version=20251115_1430

# Verify schema version
psql $DB_URL -c "
  SELECT version FROM schema_migrations
  ORDER BY applied_at DESC LIMIT 1;
"
```

### Data Restoration

```bash
# Restore from latest backup
BACKUP_FILE="backups/greenlang-gl008-2025-11-26-10-30.sql.gz"

# Verify backup file exists
ls -lh $BACKUP_FILE

# Restore database (requires downtime)
# Step 1: Terminate all connections
psql $DB_URL -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE datname = 'greenlang'
    AND pid <> pg_backend_pid();
"

# Step 2: Restore from backup
gunzip -c $BACKUP_FILE | psql $DB_URL

# Step 3: Verify restoration
psql $DB_URL -c "
  SELECT COUNT(*) FROM traps;
  SELECT COUNT(*) FROM trap_inspections;
  SELECT MAX(detected_at) FROM trap_inspections;
"
```

### Point-in-Time Recovery (PITR)

```bash
# Recover database to specific point in time
TARGET_TIME="2025-11-26 10:30:00 UTC"

# Stop PostgreSQL
kubectl scale statefulset/postgresql -n database --replicas=0

# Restore base backup
./scripts/restore-base-backup.sh --backup-id=base-2025-11-26

# Replay WAL logs until target time
./scripts/pitr-recovery.sh --target-time="$TARGET_TIME"

# Start PostgreSQL
kubectl scale statefulset/postgresql -n database --replicas=1

# Verify recovery
psql $DB_URL -c "SELECT NOW(), pg_last_wal_replay_lsn();"
```

---

## ML Model Rollback

### Model Version Rollback

```bash
# Check current model version
CURRENT_MODEL=$(kubectl get configmap ml-model-config -n greenlang-gl008 \
  -o jsonpath='{.data.ML_MODEL_VERSION}')
echo "Current ML model version: $CURRENT_MODEL"

# List available model versions
aws s3 ls s3://greenlang-ml-models/steam-trap-inspector/

# Rollback to previous model version
TARGET_MODEL="v2.4.0"

kubectl patch configmap ml-model-config -n greenlang-gl008 --type merge -p "{
  \"data\": {
    \"ML_MODEL_VERSION\": \"$TARGET_MODEL\"
  }
}"

# Restart pods to load new model
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008

# Verify model loaded
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector | grep "ML model loaded" | tail -5
```

### Model Performance Verification

```bash
# Test model with validation dataset
python scripts/test_ml_model.py \
  --model-version=$TARGET_MODEL \
  --test-data=validation_datasets/steam_trap_validation_v1.csv \
  --min-accuracy=0.90

# Compare model performance
python scripts/compare_model_performance.py \
  --model-a=$CURRENT_MODEL \
  --model-b=$TARGET_MODEL \
  --metrics=accuracy,precision,recall,f1
```

---

## Configuration Rollback

### ConfigMap Rollback

```bash
# Backup current ConfigMaps
kubectl get configmap -n greenlang-gl008 -o yaml > current-configmaps-backup.yaml

# Restore ConfigMaps from previous version
kubectl apply -f rollback-configs/configmaps-v2.4.1.yaml -n greenlang-gl008

# Verify ConfigMap values
kubectl get configmap ml-model-config -n greenlang-gl008 -o yaml
kubectl get configmap feature-flags -n greenlang-gl008 -o yaml
```

### Feature Flag Rollback

```bash
# Disable all new feature flags
kubectl patch configmap feature-flags -n greenlang-gl008 --type merge -p '{
  "data": {
    "ENABLE_NEW_ACOUSTIC_ALGORITHM": "false",
    "ENABLE_ENHANCED_THERMAL_DETECTION": "false",
    "ENABLE_BATCH_INFERENCE": "false",
    "ENABLE_HIERARCHICAL_PREDICTION": "false",
    "ENABLE_EXPERIMENTAL_FEATURES": "false"
  }
}'

# Restart to apply changes
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

### Secret Rollback

```bash
# Rotate back to previous credentials (if credential rotation caused issue)
kubectl create secret generic steam-trap-db-credentials \
  --from-literal=username=$PREVIOUS_DB_USER \
  --from-literal=password=$PREVIOUS_DB_PASS \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to use new credentials
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

---

## Verification Procedures

### Post-Rollback Health Check

```bash
#!/bin/bash
# Comprehensive post-rollback verification

echo "=== POST-ROLLBACK VERIFICATION ==="
echo ""

PASSED=0
FAILED=0

# Test 1: Pod Status
echo "[Test 1/10] Checking pod status..."
RUNNING_PODS=$(kubectl get pods -n greenlang-gl008 -l app=steam-trap-inspector --field-selector=status.phase=Running -o name | wc -l)
EXPECTED_PODS=$(kubectl get deployment steam-trap-inspector -n greenlang-gl008 -o jsonpath='{.spec.replicas}')

if [ "$RUNNING_PODS" -eq "$EXPECTED_PODS" ]; then
  echo "✓ All pods running ($RUNNING_PODS/$EXPECTED_PODS)"
  ((PASSED++))
else
  echo "✗ Pod count mismatch ($RUNNING_PODS/$EXPECTED_PODS)"
  ((FAILED++))
fi

# Test 2: API Health
echo "[Test 2/10] Checking API health..."
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.greenlang.io/v1/steam-trap/health)
if [ "$API_STATUS" -eq 200 ]; then
  echo "✓ API health check passed"
  ((PASSED++))
else
  echo "✗ API health check failed (status: $API_STATUS)"
  ((FAILED++))
fi

# Test 3: Database Connectivity
echo "[Test 3/10] Checking database connectivity..."
if psql $DB_URL -c "SELECT 1;" > /dev/null 2>&1; then
  echo "✓ Database connection successful"
  ((PASSED++))
else
  echo "✗ Database connection failed"
  ((FAILED++))
fi

# Test 4: Sensor Connectivity
echo "[Test 4/10] Checking sensor connectivity..."
ONLINE_SENSORS=$(curl -s https://api.greenlang.io/v1/steam-trap/sensors/status | jq -r '.online_sensors_count')
if [ "$ONLINE_SENSORS" -gt 0 ]; then
  echo "✓ Sensors online: $ONLINE_SENSORS"
  ((PASSED++))
else
  echo "✗ No sensors online"
  ((FAILED++))
fi

# Test 5: Inspection Workflow
echo "[Test 5/10] Testing inspection workflow..."
INSPECTION_RESULT=$(curl -s -X POST https://api.greenlang.io/v1/steam-trap/inspection \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @test-data/sample-inspection.json)

if echo "$INSPECTION_RESULT" | jq -e '.job_id' > /dev/null; then
  echo "✓ Inspection job submitted successfully"
  ((PASSED++))
else
  echo "✗ Inspection job submission failed"
  ((FAILED++))
fi

# Test 6: ML Model Loading
echo "[Test 6/10] Verifying ML model loaded..."
MODEL_STATUS=$(kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --tail=100 | grep "ML model loaded" | wc -l)
if [ "$MODEL_STATUS" -gt 0 ]; then
  echo "✓ ML model loaded successfully"
  ((PASSED++))
else
  echo "✗ ML model loading verification failed"
  ((FAILED++))
fi

# Test 7: Error Rate
echo "[Test 7/10] Checking error rate..."
ERROR_COUNT=$(kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --since=5m | grep ERROR | wc -l)
if [ "$ERROR_COUNT" -lt 10 ]; then
  echo "✓ Error rate acceptable ($ERROR_COUNT errors in last 5 minutes)"
  ((PASSED++))
else
  echo "⚠ High error rate ($ERROR_COUNT errors in last 5 minutes)"
  ((FAILED++))
fi

# Test 8: Database Schema Version
echo "[Test 8/10] Verifying database schema version..."
SCHEMA_VERSION=$(psql $DB_URL -t -c "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1;")
echo "Current schema version: $SCHEMA_VERSION"
((PASSED++))

# Test 9: Recent Inspections
echo "[Test 9/10] Checking recent inspections..."
RECENT_INSPECTIONS=$(psql $DB_URL -t -c "SELECT COUNT(*) FROM trap_inspections WHERE detected_at > NOW() - INTERVAL '10 minutes';")
echo "Inspections in last 10 minutes: $RECENT_INSPECTIONS"
((PASSED++))

# Test 10: Performance Metrics
echo "[Test 10/10] Checking performance metrics..."
LATENCY_P95=$(curl -s https://api.greenlang.io/v1/steam-trap/metrics/latency | jq -r '.p95_latency_ms')
if [ "$LATENCY_P95" -lt 2000 ]; then
  echo "✓ Latency acceptable (P95: ${LATENCY_P95}ms)"
  ((PASSED++))
else
  echo "⚠ High latency (P95: ${LATENCY_P95}ms)"
  ((FAILED++))
fi

echo ""
echo "=== VERIFICATION SUMMARY ==="
echo "Passed: $PASSED/10"
echo "Failed: $FAILED/10"
echo ""

if [ "$FAILED" -eq 0 ]; then
  echo "✓ All verification tests passed"
  echo "Rollback successful"
  exit 0
else
  echo "⚠ Some verification tests failed"
  echo "Manual investigation required"
  exit 1
fi
```

### Continuous Monitoring

```bash
#!/bin/bash
# Post-rollback monitoring script

DURATION=${1:-30}  # Default 30 minutes
INTERVAL=60  # Check every 60 seconds
END_TIME=$(($(date +%s) + $DURATION * 60))

echo "Monitoring for $DURATION minutes..."
echo "Press Ctrl+C to stop"
echo ""

while [ $(date +%s) -lt $END_TIME ]; do
  TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

  # Check error rate
  ERROR_COUNT=$(kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --since=1m | grep ERROR | wc -l)

  # Check pod status
  RUNNING_PODS=$(kubectl get pods -n greenlang-gl008 -l app=steam-trap-inspector --field-selector=status.phase=Running -o name | wc -l)

  # Check API latency
  LATENCY=$(curl -s https://api.greenlang.io/v1/steam-trap/metrics/latency | jq -r '.p95_latency_ms')

  # Check inspection success rate
  SUCCESS_RATE=$(curl -s https://api.greenlang.io/v1/steam-trap/metrics/success-rate | jq -r '.rate')

  # Display status
  echo "[$TIMESTAMP] Pods: $RUNNING_PODS | Errors (1m): $ERROR_COUNT | Latency P95: ${LATENCY}ms | Success Rate: ${SUCCESS_RATE}%"

  # Alert if thresholds exceeded
  if [ "$ERROR_COUNT" -gt 20 ]; then
    echo "⚠️  ALERT: High error rate detected!"
  fi

  if [ "$LATENCY" -gt 3000 ]; then
    echo "⚠️  ALERT: High latency detected!"
  fi

  sleep $INTERVAL
done

echo ""
echo "Monitoring complete"
```

---

## Post-Rollback Actions

### Immediate Actions (Within 1 hour)

```bash
# 1. Update incident ticket
./scripts/update-incident.sh \
  --incident-id=$INCIDENT_ID \
  --status="Resolved - Rolled back to v2.4.1" \
  --resolution-time="$(date -u)"

# 2. Notify stakeholders
./scripts/send-notification.sh \
  --template=rollback_complete \
  --target-version=v2.4.1 \
  --channels=slack,email

# 3. Update status page
curl -X PATCH https://api.statuspage.io/v1/pages/$STATUSPAGE_ID/incidents/$INCIDENT_ID \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "status": "resolved",
      "body": "Service has been rolled back and fully restored."
    }
  }'

# 4. Document rollback in runbook
cat >> rollback-log.md <<EOF
## Rollback: $(date +"%Y-%m-%d %H:%M:%S UTC")

- **From Version:** v2.5.0
- **To Version:** v2.4.1
- **Reason:** [Brief description]
- **Duration:** [X minutes]
- **Incident Commander:** [Name]
- **Outcome:** [Success/Issues encountered]

EOF
```

### Follow-Up Actions (Within 24 hours)

1. **Schedule Post-Incident Review**
   - Book meeting within 48 hours
   - Invite all involved parties
   - Prepare timeline and impact analysis

2. **Root Cause Analysis**
   - Investigate what caused the issue
   - Identify gaps in testing
   - Document lessons learned

3. **Update Deployment Process**
   - Add pre-deployment checks to prevent similar issues
   - Update testing procedures
   - Improve monitoring/alerting

4. **Customer Communication**
   - Send detailed post-mortem to affected customers
   - Offer credit/compensation if SLA breached
   - Schedule customer sync calls if needed

---

## Communication Templates

### Rollback Initiation Notification

```
Subject: [ACTION REQUIRED] GL-008 Rollback Initiated - Incident #[ID]

Team,

We are initiating a rollback of GL-008 Steam Trap Inspector due to [brief issue description].

**Rollback Details:**
- Current Version: v2.5.0
- Target Version: v2.4.1
- Rollback Type: [Fast/Standard/Full with Database]
- Estimated Duration: [X minutes]
- Expected Downtime: [None / X minutes]
- Incident Commander: @[name]
- DBA: @[name] (if database changes)

**Impact:**
- [Number] sites affected
- [Functionality] unavailable during rollback

**Communication Channels:**
- Incident Channel: #incident-gl008-[id]
- Status Page: https://status.greenlang.io/incidents/[id]

**Next Update:** In [15/30] minutes

DO NOT make any GL-008 changes without IC approval.

[Incident Commander Name]
```

### Rollback Completion Notification

```
Subject: [RESOLVED] GL-008 Rollback Complete - Incident #[ID]

Team,

The rollback of GL-008 Steam Trap Inspector has been completed successfully.

**Rollback Summary:**
- Rolled back from v2.5.0 to v2.4.1
- Duration: [X minutes]
- Downtime: [X minutes / None]
- All verification tests passed

**Current Status:**
✓ All pods healthy
✓ API responding normally
✓ Database connections stable
✓ Sensors online
✓ Inspections completing successfully

**Monitoring:**
We will continue monitoring for the next [30/60] minutes to ensure stability.

**Next Steps:**
1. Post-incident review scheduled for [Date/Time]
2. Root cause analysis in progress
3. Customer communication sent to affected sites

**Service Restored:** [Timestamp]

Thank you to everyone involved in the rapid response.

[Incident Commander Name]
```

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-11-26
**Next Review:** 2026-02-26
**Maintained By:** Platform Operations Team

**CRITICAL:** This is a safety-critical procedure. Any modifications must be reviewed by Engineering Manager and tested in staging environment.
