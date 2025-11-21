# GL-002 BoilerEfficiencyOptimizer - Rollback Procedure

Comprehensive rollback guide for GL-002 BoilerEfficiencyOptimizer production deployments. This runbook provides step-by-step procedures for safe version rollbacks.

## Table of Contents

1. [When to Rollback](#when-to-rollback)
2. [Pre-Rollback Checklist](#pre-rollback-checklist)
3. [Rollback Methods](#rollback-methods)
4. [Verification Procedures](#verification-procedures)
5. [Post-Rollback Validation](#post-rollback-validation)
6. [Database Migration Rollback](#database-migration-rollback)
7. [Communication Plan](#communication-plan)
8. [Emergency Rollback (5 Minutes)](#emergency-rollback-5-minutes)

---

## When to Rollback

### Immediate Rollback Required (P0)

Execute immediate rollback if any of these conditions occur within 2 hours of deployment:

- **Complete Service Outage**: All pods down, service unavailable
- **Data Corruption**: Database inconsistencies detected
- **Security Vulnerability**: Critical security issue discovered
- **Safety System Failure**: Boiler safety monitoring compromised
- **Error Rate >50%**: More than half of requests failing
- **Critical Integration Failure**: Unable to connect to SCADA systems

### Rollback Recommended (P1)

Consider rollback if these conditions persist for >15 minutes after deployment:

- **High Error Rate (20-50%)**: Significant request failures
- **Performance Degradation**: Response time >10 seconds (p95)
- **Memory Leaks**: OOMKilled events occurring
- **Pod Crash Loops**: Multiple pods repeatedly crashing
- **Integration Degradation**: Intermittent SCADA connection failures
- **Determinism Failures**: Inconsistent calculation results

### Monitor and Fix (P2)

These conditions may NOT require rollback - consider hotfix instead:

- **Low Error Rate (<5%)**: Isolated failures
- **Minor Performance Issues**: Response time 3-5 seconds
- **Non-Critical Features**: Reporting API failures
- **Cache Issues**: Redis connectivity problems (can failover to database)
- **Cosmetic Issues**: UI rendering problems

---

## Pre-Rollback Checklist

Before initiating rollback, verify the following:

### 1. Confirm Rollback Decision

```bash
# Check current deployment version
kubectl get deployment gl-002-boiler-efficiency -n greenlang -o jsonpath='{.spec.template.spec.containers[0].image}'

# Check rollout history
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# Check error rate
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E 'error_rate|http_requests_total'

# Check recent logs
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=100 --all-containers=true
```

**Checklist**:
- [ ] Confirmed error rate >20% OR critical failure
- [ ] Issue started after recent deployment (<2 hours)
- [ ] No alternative quick fix available
- [ ] Rollback approved by on-call engineer or incident commander
- [ ] Communication sent to #gl-002-incidents channel

### 2. Identify Rollback Target

```bash
# List recent revisions
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# Example output:
# REVISION  CHANGE-CAUSE
# 10        Deploy v1.2.3 - Stable
# 11        Deploy v1.2.4 - Performance improvements
# 12        Deploy v1.2.5 - Database migration (CURRENT - FAILING)

# View specific revision details
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang --revision=11
```

**Checklist**:
- [ ] Identified target revision (typically previous revision)
- [ ] Verified target revision was stable
- [ ] Checked if database migrations need rollback
- [ ] Confirmed target revision image exists in registry

### 3. Backup Current State

```bash
# Backup current deployment spec
kubectl get deployment gl-002-boiler-efficiency -n greenlang -o yaml > \
  /tmp/gl-002-deployment-backup-$(date +%Y%m%d-%H%M%S).yaml

# Backup current ConfigMap
kubectl get configmap gl-002-config -n greenlang -o yaml > \
  /tmp/gl-002-config-backup-$(date +%Y%m%d-%H%M%S).yaml

# Backup current Secrets (metadata only, not values)
kubectl get secret gl-002-secrets -n greenlang -o yaml | \
  grep -v 'data:' > /tmp/gl-002-secrets-backup-$(date +%Y%m%d-%H%M%S).yaml

# Export database snapshot (if applicable)
# For AWS RDS:
aws rds create-db-snapshot \
  --db-instance-identifier gl-002-prod \
  --db-snapshot-identifier gl-002-before-rollback-$(date +%Y%m%d-%H%M%S)
```

**Checklist**:
- [ ] Deployment spec backed up
- [ ] ConfigMap backed up
- [ ] Database snapshot created (if migrations involved)
- [ ] Backup files stored safely

### 4. Notify Stakeholders

```bash
# Post in Slack #gl-002-incidents
```

**Slack Message Template**:
```
🔄 ROLLBACK INITIATED: GL-002 BoilerEfficiencyOptimizer
Initiator: @your-name
Current Version: v1.2.5 (revision 12)
Target Version: v1.2.4 (revision 11)
Reason: High error rate (35%), database connection failures
ETA: 5-10 minutes
Status: https://status.greenlang.io
```

**Checklist**:
- [ ] Posted in #gl-002-incidents
- [ ] Updated status page (if P0/P1)
- [ ] Notified incident commander (if active incident)
- [ ] Documented rollback decision in incident ticket

---

## Rollback Methods

### Method 1: Quick Rollback (Recommended - 5 minutes)

Use for immediate rollback to previous revision.

```bash
# Step 1: Initiate rollback
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang

# Step 2: Monitor rollout progress
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang --timeout=5m

# Step 3: Watch pods during rollback
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency -w
```

**Expected Output**:
```
Waiting for deployment "gl-002-boiler-efficiency" rollout to finish: 1 out of 3 new replicas have been updated...
Waiting for deployment "gl-002-boiler-efficiency" rollout to finish: 1 old replicas are pending termination...
Waiting for deployment "gl-002-boiler-efficiency" rollout to finish: 2 of 3 updated replicas are available...
deployment "gl-002-boiler-efficiency" successfully rolled out
```

**Rollback Timeline**:
- 0:00 - Command executed
- 0:30 - First pod with old version starts
- 1:00 - First pod passes health checks, added to service
- 2:00 - Second pod with old version ready
- 3:00 - Third pod with old version ready
- 3:30 - Old pods terminating
- 5:00 - Rollback complete

### Method 2: Rollback to Specific Revision (10 minutes)

Use when you need to rollback to a version older than the immediate previous.

```bash
# Step 1: List all revisions
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# Step 2: View specific revision details
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang --revision=10

# Step 3: Rollback to specific revision
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang --to-revision=10

# Step 4: Monitor rollout
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang --timeout=10m
```

### Method 3: Blue-Green Rollback (15 minutes)

Use for zero-downtime rollback with instant switchback capability.

```bash
# Step 1: Deploy "green" environment with old version
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-002-boiler-efficiency-green
  namespace: greenlang
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gl-002-boiler-efficiency
      slot: green
  template:
    metadata:
      labels:
        app: gl-002-boiler-efficiency
        slot: green
    spec:
      containers:
      - name: boiler-optimizer
        image: ghcr.io/greenlang/gl-002:v1.2.4  # Previous stable version
        # ... rest of container spec
EOF

# Step 2: Wait for green deployment to be ready
kubectl rollout status deployment/gl-002-boiler-efficiency-green -n greenlang

# Step 3: Verify green deployment health
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency-green -- \
  curl -f http://localhost:8000/api/v1/health

# Step 4: Run smoke tests on green
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency-green -- \
  curl -X POST http://localhost:8000/api/v1/test/smoke \
    -H "Content-Type: application/json" \
    -d '{"test_suite": "critical"}'

# Step 5: Switch service to green deployment
kubectl patch service gl-002-boiler-efficiency -n greenlang -p '{"spec":{"selector":{"slot":"green"}}}'

# Step 6: Verify traffic switched
kubectl get endpoints gl-002-boiler-efficiency -n greenlang

# Step 7: Monitor green for 5 minutes
kubectl logs -f -n greenlang deployment/gl-002-boiler-efficiency-green

# Step 8: If stable, delete blue (failed) deployment
kubectl delete deployment gl-002-boiler-efficiency -n greenlang

# Step 9: Rename green to primary
kubectl patch deployment gl-002-boiler-efficiency-green -n greenlang \
  --type='json' -p='[{"op": "remove", "path": "/spec/selector/matchLabels/slot"}]'

kubectl patch deployment gl-002-boiler-efficiency-green -n greenlang \
  --type='json' -p='[{"op": "remove", "path": "/spec/template/metadata/labels/slot"}]'

kubectl patch service gl-002-boiler-efficiency -n greenlang \
  --type='json' -p='[{"op": "remove", "path": "/spec/selector/slot"}]'
```

### Method 4: ConfigMap/Secret Rollback (2 minutes)

Use when issue is caused by configuration changes, not code.

```bash
# Step 1: List ConfigMap revisions (if using version control)
kubectl get configmap -n greenlang | grep gl-002-config

# Step 2: Restore from backup
kubectl apply -f /tmp/gl-002-config-backup-20251117-143000.yaml

# Step 3: Restart pods to pick up old config
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# Step 4: Monitor rollout
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang
```

---

## Verification Procedures

After rollback completes, verify system health:

### Step 1: Pod Health Check (1 minute)

```bash
# Check all pods are running
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency

# Expected output: All pods in "Running" state with READY 1/1
# NAME                                       READY   STATUS    RESTARTS   AGE
# gl-002-boiler-efficiency-7b8c9d5f4-abc12   1/1     Running   0          2m
# gl-002-boiler-efficiency-7b8c9d5f4-def34   1/1     Running   0          3m
# gl-002-boiler-efficiency-7b8c9d5f4-ghi56   1/1     Running   0          4m

# Check for CrashLoopBackOff or Error states
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].ready}{"\n"}{end}'
```

**Verification Checklist**:
- [ ] All 3 pods in "Running" state
- [ ] READY status shows 1/1 for all pods
- [ ] No restarts in last 5 minutes
- [ ] No CrashLoopBackOff or Error states

### Step 2: Health Endpoint Check (1 minute)

```bash
# Check health endpoint
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/health

# Expected output:
# {
#   "status": "healthy",
#   "version": "1.2.4",
#   "components": {
#     "database": "healthy",
#     "cache": "healthy",
#     "integrations": {
#       "scada": "healthy",
#       "erp": "healthy",
#       "emissions": "healthy"
#     }
#   },
#   "uptime_seconds": 120
# }

# Check readiness endpoint
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/ready

# Expected: {"ready": true, "checks": {...}}
```

**Verification Checklist**:
- [ ] Health endpoint returns 200 OK
- [ ] All components show "healthy"
- [ ] Version matches target rollback version
- [ ] Database connectivity confirmed
- [ ] Cache connectivity confirmed
- [ ] All integrations healthy

### Step 3: Error Rate Check (2 minutes)

```bash
# Check error rate from metrics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E 'http_requests_total|error'

# Calculate error rate (last 5 minutes)
# Prometheus query (if available):
curl -s "http://prometheus:9090/api/v1/query?query=rate(gl_002_http_requests_total{status=~'5..'}[5m])" | jq

# Check recent logs for errors
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=200 --all-containers=true | grep -i error | wc -l

# Expected: <10 errors in last 200 log lines
```

**Verification Checklist**:
- [ ] Error rate <1% (target: <0.5%)
- [ ] No critical errors in recent logs
- [ ] HTTP 5xx responses <1%
- [ ] No database connection errors
- [ ] No integration timeout errors

### Step 4: Performance Check (2 minutes)

```bash
# Check response time metrics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep duration

# Run performance test
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/test/performance \
    -H "Content-Type: application/json" \
    -d '{"iterations": 100}'

# Expected output:
# {
#   "p50": 0.35,  # seconds
#   "p95": 1.2,
#   "p99": 2.1,
#   "avg": 0.52
# }

# Check CPU and memory usage
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency
```

**Verification Checklist**:
- [ ] Response time p95 <2 seconds (target: <1.5s)
- [ ] Response time p99 <5 seconds
- [ ] CPU usage <70%
- [ ] Memory usage <80%
- [ ] No memory leak indicators

### Step 5: Integration Test (3 minutes)

```bash
# Run integration smoke tests
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/test/integrations \
    -H "Content-Type: application/json" \
    -d '{
      "tests": [
        "scada_connectivity",
        "erp_connectivity",
        "emissions_api",
        "database_query",
        "cache_read_write"
      ]
    }'

# Expected output:
# {
#   "results": {
#     "scada_connectivity": "PASSED",
#     "erp_connectivity": "PASSED",
#     "emissions_api": "PASSED",
#     "database_query": "PASSED",
#     "cache_read_write": "PASSED"
#   },
#   "passed": 5,
#   "failed": 0
# }

# Test SCADA integration specifically
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/boiler/optimize \
    -H "Content-Type: application/json" \
    -d '{
      "boiler_id": "test-boiler-001",
      "test_mode": true
    }'
```

**Verification Checklist**:
- [ ] All integration tests passing
- [ ] SCADA connectivity confirmed
- [ ] ERP connectivity confirmed
- [ ] Emissions API connectivity confirmed
- [ ] Database queries executing successfully
- [ ] Cache read/write operations working

### Step 6: Functional Test (5 minutes)

```bash
# Run full smoke test suite
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/test/smoke \
    -H "Content-Type: application/json" \
    -d '{
      "test_suite": "full",
      "timeout_seconds": 300
    }'

# Expected: All tests pass

# Test optimization workflow end-to-end
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  python -c "
import requests
import json

# Submit optimization request
resp = requests.post('http://localhost:8000/api/v1/boiler/optimize',
    json={'boiler_id': 'test-001', 'test_mode': True})
print(f'Optimization request: {resp.status_code}')
assert resp.status_code == 200

# Get optimization results
job_id = resp.json()['job_id']
resp = requests.get(f'http://localhost:8000/api/v1/boiler/results/{job_id}')
print(f'Results fetch: {resp.status_code}')
assert resp.status_code == 200

print('✅ End-to-end test PASSED')
"
```

**Verification Checklist**:
- [ ] Full smoke test suite passes
- [ ] Optimization workflow completes successfully
- [ ] Results retrieval works
- [ ] Determinism maintained (same inputs = same outputs)
- [ ] No data corruption detected

---

## Post-Rollback Validation

### Step 1: Monitor for 30 Minutes

```bash
# Watch logs continuously
kubectl logs -f -n greenlang deployment/gl-002-boiler-efficiency --all-containers=true

# Monitor error rate every minute
watch -n 60 'kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate'

# Monitor resource usage
watch -n 60 'kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency'

# Check for pod restarts
watch -n 60 'kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency'
```

**Monitoring Checklist (30-minute observation)**:
- [ ] Error rate remains <1%
- [ ] No pod restarts
- [ ] Memory usage stable (not increasing)
- [ ] CPU usage stable
- [ ] No new error patterns in logs
- [ ] Response times stable

### Step 2: Load Test (Optional but Recommended)

```bash
# Run load test to verify stability under load
kubectl run -it --rm load-test --image=williamyeh/hey --restart=Never -n greenlang -- \
  /hey -n 10000 -c 50 -q 10 http://gl-002-boiler-efficiency.greenlang.svc.cluster.local/api/v1/health

# Expected output shows:
# - Success rate >99%
# - Response time p95 <2s
# - No errors

# Monitor during load test
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency
```

### Step 3: Verify HPA (Auto-Scaling)

```bash
# Check HPA status
kubectl get hpa gl-002-boiler-efficiency-hpa -n greenlang

# Expected: Current replicas = desired replicas (e.g., 3/3)

# If HPA disabled during rollback, re-enable
kubectl apply -f deployment/hpa.yaml -n greenlang

# Verify HPA is working
kubectl describe hpa gl-002-boiler-efficiency-hpa -n greenlang
```

---

## Database Migration Rollback

If the deployment included database migrations, rollback may require additional steps.

### Check if Migrations Were Applied

```bash
# Connect to database
kubectl exec -it -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL

# Check migration history
SELECT * FROM alembic_version ORDER BY version_num DESC LIMIT 5;

# Example output:
#  version_num
# -------------
#  abc123def456  <-- Current (from failed deployment v1.2.5)
#  xyz789ghi012  <-- Previous (stable v1.2.4)
```

### Rollback Database Migration

```bash
# Step 1: Backup database before migration rollback
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  pg_dump $DATABASE_URL > /tmp/gl-002-db-before-migration-rollback-$(date +%Y%m%d-%H%M%S).sql

# Or use AWS RDS snapshot
aws rds create-db-snapshot \
  --db-instance-identifier gl-002-prod \
  --db-snapshot-identifier gl-002-before-migration-rollback-$(date +%Y%m%d-%H%M%S)

# Step 2: Rollback migration using Alembic
kubectl exec -it -n greenlang deployment/gl-002-boiler-efficiency -- \
  alembic downgrade -1

# Or rollback to specific revision
kubectl exec -it -n greenlang deployment/gl-002-boiler-efficiency -- \
  alembic downgrade xyz789ghi012

# Step 3: Verify migration rollback
kubectl exec -it -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "SELECT * FROM alembic_version;"

# Expected: version_num should match target version (xyz789ghi012)

# Step 4: Run database integrity check
kubectl exec -it -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "SELECT COUNT(*) FROM boiler_configurations;"

kubectl exec -it -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "SELECT COUNT(*) FROM optimization_results;"

# Verify counts match expected values
```

**Database Rollback Checklist**:
- [ ] Database backed up before migration rollback
- [ ] Migration downgrade executed successfully
- [ ] Migration version matches target
- [ ] Data integrity checks passed
- [ ] No data loss detected
- [ ] Foreign key constraints valid

---

## Communication Plan

### During Rollback

**Slack Update (Start of Rollback)**:
```
🔄 ROLLBACK IN PROGRESS: GL-002 v1.2.5 → v1.2.4
Status: Rolling back deployment
ETA: 5 minutes
Current: 0/3 pods updated
Reason: High error rate (35%), database connection failures
Next update: 5 minutes
```

**Slack Update (Rollback Complete)**:
```
✅ ROLLBACK COMPLETE: GL-002 v1.2.4 Restored
Status: Monitoring
Duration: 6 minutes
Result: All pods healthy (3/3)
Error rate: <0.5% (normal)
Monitoring: Will monitor for 30 minutes before resolving incident
```

### Status Page Updates

**Investigating → Identified → Monitoring → Resolved**

**Status Page Update (During Rollback)**:
```
Title: GL-002 Service Issues - Rolling Back
Status: Identified

We have identified a deployment issue causing errors in the GL-002
service. We are rolling back to the previous stable version (v1.2.4).
Expected resolution time: 10 minutes.

Updated: 2025-11-17 14:45 UTC
```

**Status Page Update (Rollback Complete)**:
```
Title: GL-002 Service Issues - Monitoring
Status: Monitoring

The rollback to v1.2.4 has been completed successfully. All systems
are operating normally. We are monitoring the service to ensure
stability before closing this incident.

Updated: 2025-11-17 14:51 UTC
```

**Status Page Update (Resolved)**:
```
Title: GL-002 Service Issues - Resolved
Status: Resolved

This incident has been resolved. The service has been stable for 30
minutes following the rollback. We will conduct a post-mortem to
prevent future occurrences.

Updated: 2025-11-17 15:21 UTC
```

---

## Emergency Rollback (5 Minutes)

For P0 critical incidents, use this streamlined emergency procedure:

```bash
# EMERGENCY ROLLBACK - Execute immediately for P0 incidents

# 1. Post emergency rollback notice (10 seconds)
# Slack: @here P0 EMERGENCY ROLLBACK INITIATED - GL-002

# 2. Initiate rollback (10 seconds)
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang

# 3. Monitor rollout (3-5 minutes)
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang --timeout=5m

# 4. Quick health check (30 seconds)
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/health

# 5. Verify error rate (30 seconds)
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate

# 6. Post resolution (30 seconds)
# Slack: ✅ EMERGENCY ROLLBACK COMPLETE - GL-002 v1.2.4 restored, monitoring

# Total time: ~6 minutes
```

---

## Rollback Failure Recovery

If rollback itself fails:

### Scenario 1: Rollback Stuck

```bash
# Check rollout status
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang

# If stuck, force restart
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# If still stuck, manually scale down and up
kubectl scale deployment gl-002-boiler-efficiency --replicas=0 -n greenlang
sleep 30
kubectl scale deployment gl-002-boiler-efficiency --replicas=3 -n greenlang
```

### Scenario 2: Old Version Image Missing

```bash
# Check image availability
docker pull ghcr.io/greenlang/gl-002:v1.2.4

# If image missing, manually set to known good version
kubectl set image deployment/gl-002-boiler-efficiency -n greenlang \
  boiler-optimizer=ghcr.io/greenlang/gl-002:v1.2.3
```

### Scenario 3: Database Migration Prevents Rollback

```bash
# Emergency: Deploy old code with NEW schema (temporary)
kubectl set image deployment/gl-002-boiler-efficiency -n greenlang \
  boiler-optimizer=ghcr.io/greenlang/gl-002:v1.2.4

# Set compatibility flag
kubectl set env deployment/gl-002-boiler-efficiency -n greenlang \
  DATABASE_SCHEMA_COMPATIBILITY=forward

# Then schedule migration rollback during maintenance window
```

---

## Rollback Metrics

Track rollback metrics for continuous improvement:

```bash
# Rollback frequency
# Target: <1 rollback per 10 deployments

# Rollback duration
# Target: <10 minutes (P0), <15 minutes (P1)

# Rollback success rate
# Target: 100% (rollback should always work)

# Time to detect issue requiring rollback
# Target: <10 minutes
```

---

## Post-Rollback Actions

### Immediate (Within 1 Hour)

- [ ] Update incident ticket with rollback details
- [ ] Document root cause analysis (preliminary)
- [ ] Create JIRA ticket for fix
- [ ] Update status page to "Resolved"
- [ ] Send all-clear message to stakeholders

### Short-term (Within 24 Hours)

- [ ] Schedule post-mortem meeting
- [ ] Analyze what caused the failed deployment
- [ ] Identify what could have prevented the issue
- [ ] Update CI/CD pipeline with additional checks
- [ ] Update runbook with new scenario (if novel)

### Long-term (Within 1 Week)

- [ ] Implement preventative measures
- [ ] Add tests to catch the issue in CI
- [ ] Update deployment checklist
- [ ] Conduct blameless post-mortem
- [ ] Share learnings with engineering team

---

## Additional Resources

- **Incident Response Guide**: `INCIDENT_RESPONSE.md`
- **Troubleshooting Guide**: `TROUBLESHOOTING.md`
- **Scaling Guide**: `SCALING_GUIDE.md`
- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002\deployment\README.md`
- **Blue-Green Deployment**: https://docs.greenlang.io/deployment/blue-green
- **Database Migrations**: https://docs.greenlang.io/database/migrations
- **Post-Mortem Template**: https://docs.greenlang.io/incident-management/post-mortem-template
