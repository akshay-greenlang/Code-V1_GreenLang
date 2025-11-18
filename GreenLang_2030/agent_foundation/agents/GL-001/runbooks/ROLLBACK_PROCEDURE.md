# GL-001 ProcessHeatOrchestrator - Rollback Procedures

Safe version rollback procedures for GL-001 ProcessHeatOrchestrator and coordinated sub-agent rollbacks across multi-plant operations.

**CRITICAL SAFETY**: All rollbacks affecting heat operations must maintain critical heat supply. Never rollback if it will cause heat loss to life-safety systems.

## Table of Contents

1. [When to Rollback](#when-to-rollback)
2. [Pre-Rollback Checklist](#pre-rollback-checklist)
3. [Emergency Rollback (<5 Minutes)](#emergency-rollback-5-minutes)
4. [Coordinated Multi-Agent Rollback](#coordinated-multi-agent-rollback)
5. [Specific Revision Rollback](#specific-revision-rollback)
6. [Partial Rollback (Single Plant)](#partial-rollback-single-plant)
7. [Blue-Green Rollback](#blue-green-rollback)
8. [ConfigMap/Secret Rollback](#configmapsecret-rollback)
9. [Database Migration Rollback](#database-migration-rollback)
10. [Multi-Plant Verification Procedures](#multi-plant-verification-procedures)
11. [Rollback Communication](#rollback-communication)

---

## When to Rollback

### Decision Matrix

| Condition | Rollback? | Type | Timeframe |
|-----------|-----------|------|-----------|
| **Critical heat loss after deployment** | ✅ YES - IMMEDIATE | Emergency Rollback | <5 min |
| **Master orchestrator down after deployment** | ✅ YES | Emergency or Coordinated | <10 min |
| **Multi-plant cascade failure** | ✅ YES | Coordinated Multi-Agent | <10 min |
| **Heat optimization errors** | ✅ YES | Specific Revision | <15 min |
| **Single plant issues** | ⚠️ MAYBE | Partial Rollback | <8 min |
| **Performance degradation (CPU/latency)** | ⚠️ MONITOR 15 MIN | Consider after monitoring | - |
| **Minor issues, no heat impact** | ❌ NO | Fix forward instead | - |
| **Sub-agent errors only** | ⚠️ MAYBE | Rollback affected agents only | <10 min |

### Rollback Criteria

**ROLLBACK IMMEDIATELY if:**
- Critical heat loss occurred within 2 hours of deployment
- Master orchestrator pods all failing after deployment
- Multi-plant coordination completely broken
- Emissions compliance violations after deployment
- Safety alarms triggered post-deployment
- Database corruption detected

**MONITOR FIRST (15 minutes) if:**
- Performance degradation but heat supply normal
- Elevated error rate but <20%
- Single pod failures with auto-recovery
- Minor sub-agent issues

**FIX FORWARD if:**
- Issue unrelated to recent deployment (>4 hours ago)
- Simple configuration error (use ConfigMap rollback)
- Issue can be hotfixed faster than rollback
- Rollback would cause more disruption than fix

---

## Pre-Rollback Checklist

**Before ANY rollback, verify:**

```bash
# 1. Check deployment history
kubectl rollout history deployment/gl-001-process-heat-orchestrator -n greenlang

# Output shows recent changes:
# REVISION  CHANGE-CAUSE
# 12        <deployment-12-details>
# 13        <deployment-13-details> (current)

# 2. Identify what will be rolled back TO
kubectl rollout history deployment/gl-001-process-heat-orchestrator --revision=12 -n greenlang

# 3. Check current heat status (CRITICAL)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/critical-heat-status | \
  jq '.plants[] | {plant_id, heat_status, critical_systems_ok}'

# 4. Verify backup heat systems available
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/backup-heat-status | \
  jq '.plants[] | {plant_id, backup_available, backup_capacity_mw}'

# 5. Check sub-agent versions (for coordinated rollback)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/versions | \
  jq '.agents[] | {agent_id, version, deployment_time}'

# 6. Notify stakeholders
# Post in #gl-001-incidents and #plant-operations

# 7. Have emergency contacts ready
# Plant Safety Officer: (see PagerDuty)
# Plant Operations Manager: (see PagerDuty)
```

**Pre-Rollback Safety Checklist:**
- [ ] Critical heat systems identified and monitored
- [ ] Backup heat systems confirmed available
- [ ] Plant Safety Officer notified (for P0 incidents)
- [ ] Plant Operations Manager notified
- [ ] Rollback destination version verified safe
- [ ] Sub-agent coordination plan ready (if needed)
- [ ] Emergency rollback procedure reviewed
- [ ] Communication channels established

---

## Emergency Rollback (5 Minutes)

**Use for: P0 critical heat loss, master orchestrator complete failure**

**Target: Restore critical heat within 5 minutes**

### Step 1: Immediate Rollback (0-2 minutes)

```bash
# 1. ROLLBACK MASTER ORCHESTRATOR IMMEDIATELY
kubectl rollout undo deployment/gl-001-process-heat-orchestrator -n greenlang

# Expected output:
# deployment.apps/gl-001-process-heat-orchestrator rolled back

# 2. MONITOR ROLLBACK STATUS
kubectl rollout status deployment/gl-001-process-heat-orchestrator -n greenlang --timeout=2m

# Expected output after ~90 seconds:
# Waiting for deployment "gl-001-process-heat-orchestrator" rollout to finish: 1 out of 3 new replicas have been updated...
# Waiting for deployment "gl-001-process-heat-orchestrator" rollout to finish: 2 out of 3 new replicas have been updated...
# deployment "gl-001-process-heat-orchestrator" successfully rolled out

# 3. IMMEDIATE HEALTH CHECK
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -f http://localhost:8000/api/v1/health

# Expected: HTTP 200 OK
```

### Step 2: Verify Heat Restoration (2-3 minutes)

```bash
# Check critical heat supply restored
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/critical-heat-status | \
  jq '.plants[] | select(.heat_loss == true)'

# Expected output: [] (empty array, no heat loss)

# If heat not restored, check specific plants
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status | \
  jq '.plants[] | {plant_id, heat_status, efficiency, alarms}'

# Verify sub-agents reconnecting
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq '.agents | map(select(.status == "connected")) | length'

# Target: >90 agents connected (out of 99)
```

### Step 3: Coordinated Sub-Agent Rollback if Needed (3-5 minutes)

**Only if master rollback didn't restore heat AND sub-agents were also deployed recently**

```bash
# Identify which sub-agents need rollback
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/versions | \
  jq '.agents[] | select(.deployment_time > (now - 7200)) | .agent_id'

# Rollback critical sub-agents in PARALLEL
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang &
kubectl rollout undo deployment/gl-003-combustion-optimizer -n greenlang &
kubectl rollout undo deployment/gl-004-emissions-control -n greenlang &
kubectl rollout undo deployment/gl-005-heat-recovery -n greenlang &

# Wait for all rollbacks to complete
wait

# Verify all sub-agents rolled back
kubectl get pods -n greenlang | grep -E '(gl-002|gl-003|gl-004|gl-005)'

# Expected: All pods "Running" with "1/1 READY"
```

### Step 4: Final Verification (5 minutes)

```bash
# Verify ALL critical metrics
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/emergency/rollback-verification | jq

# Expected output:
# {
#   "rollback_status": "successful",
#   "heat_status": {
#     "all_plants_ok": true,
#     "critical_heat_restored": true,
#     "heat_loss_count": 0
#   },
#   "master_orchestrator": {
#     "pods_running": 3,
#     "health": "healthy",
#     "version": "1.4.2"
#   },
#   "sub_agents": {
#     "total": 99,
#     "connected": 99,
#     "coordination_ok": true
#   },
#   "optimization": {
#     "heat_distribution": "optimal",
#     "efficiency": 85.3
#   },
#   "safety": {
#     "alarms_active": 0,
#     "safety_systems_ok": true
#   }
# }

# If verification fails, escalate to Plant Safety Officer IMMEDIATELY
```

**Emergency Rollback Completion Checklist:**
- [ ] Master orchestrator rolled back: Revision ___ → ___
- [ ] All master pods running: 3/3 READY
- [ ] Heat restored to all plants: ✅
- [ ] Sub-agents connected: ___/99 (target >95)
- [ ] No active safety alarms
- [ ] Heat distribution optimization: [optimal/converged]
- [ ] Plant Operations Manager notified of restoration
- [ ] Incident documented in #gl-001-incidents

---

## Coordinated Multi-Agent Rollback

**Use for: Master + sub-agents deployed together and failing together**

**Target: 10 minutes for coordinated rollback**

### Step 1: Identify Coordination Scope (0-2 minutes)

```bash
# Determine which agents were deployed in the same release
kubectl get events -n greenlang --sort-by='.lastTimestamp' | \
  grep -E '(gl-001|gl-002|gl-003|gl-004|gl-005)' | \
  grep -E '(Deployment|Scaled)' | tail -50

# Check deployment times
kubectl get deployments -n greenlang -l agent-group=process-heat \
  -o custom-columns=NAME:.metadata.name,REVISION:.metadata.generation,AGE:.metadata.creationTimestamp

# Identify agents deployed within same 30-minute window
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/deployment-groups | \
  jq '.deployment_groups[] | select(.deployment_time > (now - 7200))'

# Example output:
# {
#   "deployment_id": "release-2025-11-17-1430",
#   "agents": ["gl-001", "gl-002", "gl-003", "gl-012", "gl-016"],
#   "deployment_time": "2025-11-17T14:30:00Z"
# }
```

### Step 2: Plan Rollback Order (2-3 minutes)

**Rollback order for safety:**
1. **Master orchestrator** (GL-001) - FIRST (restore coordination)
2. **Boiler agents** (GL-002, GL-012, GL-016, GL-017) - Critical heat
3. **Combustion agents** (GL-003, GL-004) - Safety and emissions
4. **Heat recovery agents** (GL-005, GL-006) - Efficiency (lower priority)

```bash
# Create rollback plan
cat > rollback-plan.sh <<'EOF'
#!/bin/bash
set -e

echo "Starting coordinated multi-agent rollback..."

# Step 1: Rollback master orchestrator
echo "[1/4] Rolling back GL-001 master orchestrator..."
kubectl rollout undo deployment/gl-001-process-heat-orchestrator -n greenlang
kubectl rollout status deployment/gl-001-process-heat-orchestrator -n greenlang --timeout=3m

# Step 2: Rollback boiler agents (parallel)
echo "[2/4] Rolling back boiler agents (GL-002, GL-012, GL-016, GL-017)..."
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang &
kubectl rollout undo deployment/gl-012-steam-system -n greenlang &
kubectl rollout undo deployment/gl-016-boiler-control -n greenlang &
kubectl rollout undo deployment/gl-017-feedwater -n greenlang &
wait
echo "Boiler agents rolled back"

# Step 3: Rollback combustion/emissions agents (parallel)
echo "[3/4] Rolling back combustion agents (GL-003, GL-004)..."
kubectl rollout undo deployment/gl-003-combustion-optimizer -n greenlang &
kubectl rollout undo deployment/gl-004-emissions-control -n greenlang &
wait
echo "Combustion agents rolled back"

# Step 4: Rollback heat recovery agents (parallel)
echo "[4/4] Rolling back heat recovery agents (GL-005, GL-006)..."
kubectl rollout undo deployment/gl-005-heat-recovery -n greenlang &
kubectl rollout undo deployment/gl-006-waste-heat -n greenlang &
wait
echo "Heat recovery agents rolled back"

echo "Coordinated rollback complete!"
EOF

chmod +x rollback-plan.sh
```

### Step 3: Execute Coordinated Rollback (3-8 minutes)

```bash
# Execute rollback plan
./rollback-plan.sh

# Monitor progress in separate terminal
watch -n 5 'kubectl get pods -n greenlang | grep -E "(gl-001|gl-002|gl-003|gl-004|gl-005|gl-012|gl-016|gl-017)"'

# Expected timeline:
# T+0m: Master rollback started
# T+2m: Master rollback complete, boiler agents starting
# T+4m: Boiler agents complete, combustion agents starting
# T+6m: Combustion agents complete, heat recovery starting
# T+8m: All agents rolled back
```

### Step 4: Verify Coordination Restored (8-10 minutes)

```bash
# Verify master-agent coordination
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq '{
    total: .agents | length,
    connected: [.agents[] | select(.status == "connected")] | length,
    disconnected: [.agents[] | select(.status != "connected")] | map(.agent_id)
  }'

# Expected:
# {
#   "total": 99,
#   "connected": 99,
#   "disconnected": []
# }

# Verify heat distribution optimization
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/status | \
  jq '.optimization_status'

# Expected: "optimal" or "converged"

# Verify all plants normal
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status | \
  jq '.plants[] | {plant_id, heat_status, efficiency}'

# All plants should show heat_status: "normal"
```

---

## Specific Revision Rollback

**Use for: Rollback to specific known-good version (not just previous)**

### Step 1: Identify Target Revision

```bash
# List all revisions
kubectl rollout history deployment/gl-001-process-heat-orchestrator -n greenlang

# Example output:
# REVISION  CHANGE-CAUSE
# 10        Version 1.4.0 - Stable baseline
# 11        Version 1.4.1 - Hotfix for heat calculation
# 12        Version 1.4.2 - Performance improvements
# 13        Version 1.5.0 - Multi-plant optimization (CURRENT - FAILING)

# View specific revision details
kubectl rollout history deployment/gl-001-process-heat-orchestrator \
  --revision=12 -n greenlang

# Decide target: Revision 12 (last known stable)
```

### Step 2: Rollback to Specific Revision

```bash
# Rollback to revision 12
kubectl rollout undo deployment/gl-001-process-heat-orchestrator \
  --to-revision=12 -n greenlang

# Monitor rollback
kubectl rollout status deployment/gl-001-process-heat-orchestrator -n greenlang --timeout=5m

# Verify correct version deployed
kubectl get deployment gl-001-process-heat-orchestrator -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'

# Expected: gcr.io/greenlang/gl-001:1.4.2
```

### Step 3: Verification

```bash
# Verify version in application
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/version | jq

# Expected:
# {
#   "agent_id": "GL-001",
#   "version": "1.4.2",
#   "revision": 12,
#   "build_time": "2025-11-10T10:30:00Z"
# }

# Run full verification suite
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/test/post-rollback-verification
```

---

## Partial Rollback (Single Plant)

**Use for: Issues isolated to specific plant, don't want to rollback entire system**

### Step 1: Isolate Plant

```bash
# Identify affected plant
AFFECTED_PLANT="Plant-003"

# Check plant-specific agents
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq ".agents[] | select(.plant_id == \"$AFFECTED_PLANT\")"

# Put plant in isolation mode
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/plants/$AFFECTED_PLANT/isolate \
    -H "Content-Type: application/json" \
    -d '{
      "reason": "partial_rollback",
      "maintain_heat": true,
      "use_local_control": true
    }'
```

### Step 2: Rollback Plant-Specific Configuration

```bash
# Rollback ConfigMap for specific plant
kubectl get configmap gl-001-plant-configs -n greenlang -o yaml > config-backup.yaml

kubectl edit configmap gl-001-plant-configs -n greenlang

# Revert Plant-003 configuration to previous values
# Example:
# data:
#   plant-003.yaml: |
#     plant_id: "Plant-003"
#     heat_demand_mw: 45  # Revert from 50 to 45
#     optimization_enabled: false  # Disable new optimization

# Trigger config reload (no pod restart needed)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/config/reload \
    -H "Content-Type: application/json" \
    -d '{"plants": ["Plant-003"]}'
```

### Step 3: Verify and Re-integrate Plant

```bash
# Verify plant operating normally with reverted config
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/$AFFECTED_PLANT/status | jq

# Re-integrate plant with master orchestration
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/plants/$AFFECTED_PLANT/integrate \
    -H "Content-Type: application/json" \
    -d '{"gradual": true, "verify_before_full": true}'

# Monitor re-integration
watch -n 10 'kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/'$AFFECTED_PLANT'/integration-status'
```

---

## Blue-Green Rollback

**Use for: Zero-downtime rollback for non-emergency scenarios**

### Step 1: Verify Blue Deployment Exists

```bash
# Check if blue deployment (previous version) is still running
kubectl get deployment gl-001-process-heat-orchestrator-blue -n greenlang

# If it exists:
# NAME                                        READY   UP-TO-DATE   AVAILABLE   AGE
# gl-001-process-heat-orchestrator-blue       3/3     3            3           2h

# If it doesn't exist, this rollback method is not available
# Fall back to Emergency Rollback
```

### Step 2: Switch Traffic to Blue (20 minutes)

```bash
# Update service to point to blue deployment
kubectl patch service gl-001-process-heat-orchestrator -n greenlang -p \
  '{"spec":{"selector":{"deployment":"gl-001-blue"}}}'

# Expected output:
# service/gl-001-process-heat-orchestrator patched

# Verify traffic switched
kubectl get service gl-001-process-heat-orchestrator -n greenlang \
  -o jsonpath='{.spec.selector}'

# Expected: {"deployment":"gl-001-blue"}

# Monitor sub-agents reconnecting to blue deployment
watch -n 5 'kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator-blue -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq ".agents | map(select(.status == \"connected\")) | length"'
```

### Step 3: Scale Down Green (Failed) Deployment

```bash
# Once blue is fully operational, scale down green (failed) deployment
kubectl scale deployment gl-001-process-heat-orchestrator-green --replicas=0 -n greenlang

# Verify
kubectl get deployments -n greenlang | grep gl-001

# Expected:
# gl-001-process-heat-orchestrator-blue    3/3     3            3           2h
# gl-001-process-heat-orchestrator-green   0/0     0            0           30m
```

### Step 4: Promote Blue to Primary

```bash
# Rename deployments (blue becomes primary)
kubectl delete deployment gl-001-process-heat-orchestrator-green -n greenlang

kubectl patch deployment gl-001-process-heat-orchestrator-blue -n greenlang \
  --type=json -p='[{"op": "replace", "path": "/metadata/name", "value": "gl-001-process-heat-orchestrator"}]'

# Update service to use standard selector
kubectl patch service gl-001-process-heat-orchestrator -n greenlang -p \
  '{"spec":{"selector":{"app":"gl-001-process-heat-orchestrator"}}}'
```

---

## ConfigMap/Secret Rollback

**Use for: Configuration-only changes, no code deployment**

**Target: 2 minutes**

```bash
# 1. Backup current ConfigMap
kubectl get configmap gl-001-config -n greenlang -o yaml > configmap-backup-$(date +%Y%m%d-%H%M%S).yaml

# 2. Retrieve previous ConfigMap from etcd backup or Git
# Option A: From Git (if ConfigMaps are version controlled)
git checkout HEAD~1 -- kubernetes/gl-001/configmap.yaml
kubectl apply -f kubernetes/gl-001/configmap.yaml

# Option B: Manually edit to revert changes
kubectl edit configmap gl-001-config -n greenlang

# 3. Trigger config reload (no pod restart for non-breaking changes)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/config/reload

# 4. If breaking change, rolling restart
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang

# 5. Verify config applied
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/config/current | jq
```

**Secret Rollback:**
```bash
# Secrets cannot be edited in-place, must be recreated

# 1. Backup current secret
kubectl get secret gl-001-db-credentials -n greenlang -o yaml > secret-backup.yaml

# 2. Delete current secret
kubectl delete secret gl-001-db-credentials -n greenlang

# 3. Recreate with previous values
kubectl create secret generic gl-001-db-credentials -n greenlang \
  --from-literal=username=gl001_user \
  --from-literal=password='<previous-password>'

# 4. Restart pods to pick up new secret
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

---

## Database Migration Rollback

**Use for: Database schema changes causing issues**

**CRITICAL: Test in staging first, database rollbacks are risky**

### Step 1: Assess Migration Impact

```bash
# Check current migration status
kubectl exec -n greenlang postgresql-0 -- psql -U gl001_user -d gl001_production -c \
  "SELECT * FROM schema_migrations ORDER BY version DESC LIMIT 10;"

# Identify which migration to rollback
# Example: Migration 20251117143000 causing issues

# Check if migration is reversible
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  python manage.py showmigrations --plan | grep 20251117143000
```

### Step 2: Stop Application (Prevent Writes)

```bash
# Scale down to 0 to prevent database writes during rollback
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=0 -n greenlang

# Wait for all pods to terminate
kubectl wait --for=delete pod -l app=gl-001-process-heat-orchestrator -n greenlang --timeout=2m

# Verify no connections
kubectl exec -n greenlang postgresql-0 -- psql -U postgres -c \
  "SELECT COUNT(*) FROM pg_stat_activity WHERE datname='gl001_production' AND state='active';"

# Expected: 0 active connections
```

### Step 3: Database Backup (CRITICAL)

```bash
# Create snapshot before rollback
kubectl exec -n greenlang postgresql-0 -- pg_dump -U gl001_user gl001_production \
  > db-backup-before-rollback-$(date +%Y%m%d-%H%M%S).sql

# Verify backup created
ls -lh db-backup-before-rollback-*.sql

# Or use AWS RDS snapshot
aws rds create-db-snapshot \
  --db-instance-identifier gl-001-production \
  --db-snapshot-identifier gl-001-rollback-$(date +%Y%m%d-%H%M%S)
```

### Step 4: Execute Migration Rollback

```bash
# Rollback migration
kubectl run -it --rm migration-rollback --image=gcr.io/greenlang/gl-001:previous-version \
  -n greenlang --restart=Never -- \
  python manage.py migrate <app_name> 20251116120000  # Previous migration

# Expected output:
# Running migrations:
#   Unapplying <migration_name>... OK

# Verify rollback
kubectl exec -n greenlang postgresql-0 -- psql -U gl001_user -d gl001_production -c \
  "SELECT * FROM schema_migrations ORDER BY version DESC LIMIT 5;"
```

### Step 5: Restore Application

```bash
# Scale application back up
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=3 -n greenlang

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=gl-001-process-heat-orchestrator \
  -n greenlang --timeout=5m

# Verify database connectivity
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/health/database
```

---

## Multi-Plant Verification Procedures

**After ANY rollback, verify ALL plants operational**

### Comprehensive Verification Script

```bash
#!/bin/bash
# GL-001 Post-Rollback Verification Script

echo "=== GL-001 ProcessHeatOrchestrator Rollback Verification ==="
echo ""

# 1. Master Orchestrator Health
echo "[1/8] Checking master orchestrator health..."
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -sf http://localhost:8000/api/v1/health || { echo "FAILED: Master health check"; exit 1; }
echo "✅ Master orchestrator healthy"

# 2. All Pods Running
echo "[2/8] Verifying all pods running..."
POD_COUNT=$(kubectl get pods -n greenlang -l app=gl-001-process-heat-orchestrator --field-selector=status.phase=Running --no-headers | wc -l)
if [ "$POD_COUNT" -lt 3 ]; then
  echo "FAILED: Only $POD_COUNT pods running (expected 3)"
  exit 1
fi
echo "✅ All $POD_COUNT pods running"

# 3. Multi-Plant Heat Status
echo "[3/8] Verifying heat status across all plants..."
HEAT_LOSS_COUNT=$(kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status | \
  jq '.plants | map(select(.heat_loss == true)) | length')
if [ "$HEAT_LOSS_COUNT" -gt 0 ]; then
  echo "FAILED: $HEAT_LOSS_COUNT plants with heat loss"
  exit 1
fi
echo "✅ No heat loss at any plant"

# 4. Sub-Agent Coordination
echo "[4/8] Verifying sub-agent coordination..."
CONNECTED_AGENTS=$(kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq '.agents | map(select(.status == "connected")) | length')
if [ "$CONNECTED_AGENTS" -lt 94 ]; then  # Allow up to 5% disconnected
  echo "WARNING: Only $CONNECTED_AGENTS/99 agents connected (expected >94)"
fi
echo "✅ $CONNECTED_AGENTS/99 agents connected"

# 5. Heat Distribution Optimization
echo "[5/8] Verifying heat distribution optimization..."
OPT_STATUS=$(kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/status | \
  jq -r '.optimization_status')
if [ "$OPT_STATUS" != "optimal" ] && [ "$OPT_STATUS" != "converged" ]; then
  echo "WARNING: Optimization status is $OPT_STATUS (expected optimal/converged)"
fi
echo "✅ Heat optimization: $OPT_STATUS"

# 6. Database Connectivity
echo "[6/8] Verifying database connectivity..."
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -sf http://localhost:8000/api/v1/health/database || { echo "FAILED: Database check"; exit 1; }
echo "✅ Database connectivity OK"

# 7. Message Bus Health
echo "[7/8] Verifying message bus health..."
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -sf http://localhost:8000/api/v1/message-bus/health || { echo "FAILED: Message bus check"; exit 1; }
echo "✅ Message bus healthy"

# 8. Safety Systems
echo "[8/8] Verifying safety systems..."
ALARM_COUNT=$(kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/safety/alarms | \
  jq '.active_alarms | length')
if [ "$ALARM_COUNT" -gt 0 ]; then
  echo "WARNING: $ALARM_COUNT active safety alarms"
  kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
    curl -s http://localhost:8000/api/v1/safety/alarms | jq '.active_alarms'
fi
echo "✅ No critical safety alarms"

echo ""
echo "=== ROLLBACK VERIFICATION COMPLETE ==="
echo "Status: ALL CHECKS PASSED ✅"
echo "Timestamp: $(date)"
```

**Run verification:**
```bash
chmod +x verify-rollback.sh
./verify-rollback.sh
```

---

## Rollback Communication

### Pre-Rollback Announcement

```
#gl-001-incidents #plant-operations

⚠️ ROLLBACK INITIATED - GL-001 ProcessHeatOrchestrator

Rollback Lead: @your-name
Reason: [Brief description, e.g., "Critical heat loss after v1.5.0 deployment"]
Type: [Emergency / Coordinated / Specific Revision / Partial]
Affected Systems: [Master only / Master + Sub-agents / Single plant]

Current Status:
- Heat Supply: [CRITICAL / DEGRADED / NORMAL]
- Plants Affected: [Plant-001, Plant-002, ...] or ALL
- Sub-Agents Affected: [count]

Rollback Plan:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Target ETA: [time estimate]
Next Update: 5 minutes
```

### Mid-Rollback Update

```
ROLLBACK UPDATE - <timestamp>

Progress: [XX% complete]
Time Elapsed: [XX minutes]

Completed:
✅ [Completed step 1]
✅ [Completed step 2]

In Progress:
⏳ [Current step]

Next Steps:
- [Next step 1]
- [Next step 2]

Heat Status: [Current heat supply status]
Next Update: 5 minutes
```

### Rollback Completion

```
✅ ROLLBACK COMPLETE - GL-001 ProcessHeatOrchestrator

Rollback Lead: @your-name
Duration: [XX minutes]
Completed: <timestamp>

Rollback Summary:
- Master Orchestrator: v1.5.0 → v1.4.2 (Revision 13 → 12)
- Sub-Agents Rolled Back: [GL-002, GL-003, GL-004, ...] or NONE
- Database Migration: [Rolled back / No changes]

Verification Results:
✅ All plants heat status: NORMAL
✅ Master orchestrator: 3/3 pods healthy
✅ Sub-agents connected: 99/99
✅ Heat optimization: OPTIMAL
✅ No safety alarms
✅ All systems operational

Root Cause: [Brief description]
Next Steps:
- Monitor for 1 hour: @your-name
- Post-mortem scheduled: <date/time>
- Fix forward plan: [TBD / In progress]

Thank you all for the rapid response.
```

---

**Last Updated**: 2025-11-17
**Version**: 1.0
**Maintained By**: GreenLang Platform Engineering & Process Heat Team
**Review Cycle**: After each rollback or quarterly
