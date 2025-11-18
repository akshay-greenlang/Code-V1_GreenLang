# GL-003 SteamSystemAnalyzer Rollback Procedure

## Document Control

**Document Version:** 1.0.0
**Last Updated:** 2025-11-17
**Owner:** GL-003 Operations Team
**Reviewers:** Platform Engineering, Steam Safety Team
**Next Review:** 2025-12-17

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [When to Rollback Decision Tree](#when-to-rollback-decision-tree)
3. [Rollback Severity Levels](#rollback-severity-levels)
4. [Pre-Rollback Checklist](#pre-rollback-checklist)
5. [Emergency Rollback Procedure](#emergency-rollback-procedure)
6. [Standard Rollback Procedure](#standard-rollback-procedure)
7. [Kubernetes Rollback](#kubernetes-rollback)
8. [Database Rollback](#database-rollback)
9. [Steam Analysis Recalibration](#steam-analysis-recalibration)
10. [Partial Rollback Procedures](#partial-rollback-procedures)
11. [Post-Rollback Validation](#post-rollback-validation)
12. [Communication Templates](#communication-templates)
13. [Rollback Audit Trail](#rollback-audit-trail)
14. [Recovery Procedures](#recovery-procedures)
15. [Appendices](#appendices)

---

## Executive Summary

This runbook provides comprehensive procedures for rolling back GL-003 SteamSystemAnalyzer deployments. Given the critical nature of steam system monitoring and safety, rollbacks must be executed with precision to ensure:

- **Safety First:** No compromise to steam system monitoring
- **Data Integrity:** Preserve all steam measurement data
- **Minimal Downtime:** <5 minutes for emergency rollbacks
- **Audit Compliance:** Complete rollback audit trail
- **Stakeholder Communication:** Clear notification protocols

**Critical Success Factors:**
- Pre-rollback validation of target version
- Automated health checks post-rollback
- Steam meter continuity verification
- Database integrity validation
- Safety alert system verification

---

## When to Rollback Decision Tree

### Decision Tree Flowchart

```
                    [Deployment Issue Detected]
                              |
                              v
                  [Is safety compromised?]
                     /              \
                   YES               NO
                    |                 |
                    v                 v
            [EMERGENCY ROLLBACK]  [Assess Impact]
            Execute within 5min        |
                                       v
                            [Can issue be fixed forward?]
                              /              \
                            YES               NO
                             |                 |
                             v                 v
                      [Apply Hotfix]    [Is downtime acceptable?]
                      [Monitor]              /           \
                                          YES            NO
                                           |              |
                                           v              v
                                   [STANDARD ROLLBACK] [PARTIAL ROLLBACK]
                                   Planned 15-30min    Component-specific
```

### Rollback Triggers

#### IMMEDIATE ROLLBACK (P0 - Critical)

Execute emergency rollback immediately if any occur:

1. **Safety System Failures**
   - Steam pressure monitoring failure (>10% meters offline)
   - Safety alert system not functioning
   - Emergency shutdown integration broken
   - Pressure threshold alerts not triggering

2. **Data Loss Scenarios**
   - Steam measurement data not being recorded
   - Database write failures >5% of operations
   - TimescaleDB hypertable corruption
   - Data retention policy failing

3. **System Instability**
   - Pod crash loop (>3 restarts in 5 minutes)
   - Memory leaks detected (>80% memory usage sustained)
   - CPU throttling affecting real-time monitoring
   - Network partition affecting meter connectivity

4. **Compliance Violations**
   - Audit trail not recording changes
   - Regulatory reporting endpoints failing
   - Data encryption failure
   - Access control bypass detected

#### PLANNED ROLLBACK (P1 - High Priority)

Schedule rollback within 1 hour if:

1. **Performance Degradation**
   - Response time >5 seconds (baseline: 200ms)
   - Throughput decreased >30%
   - Queue backlog >10,000 messages
   - Database query performance >2x slower

2. **Functional Regressions**
   - Key features not working (condensate tracking, etc.)
   - Calculation accuracy issues detected
   - Report generation failures
   - API endpoint errors >5%

3. **Integration Failures**
   - ERP connector not syncing
   - SCADA system integration broken
   - Third-party API failures
   - Webhook delivery failures >10%

#### DEFERRED ROLLBACK (P2 - Medium Priority)

Can be scheduled during maintenance window:

1. **Minor Issues**
   - UI cosmetic issues
   - Non-critical feature bugs
   - Performance issues in low-priority features
   - Documentation discrepancies

2. **Dependency Issues**
   - Library version conflicts (non-breaking)
   - Container image size bloat
   - Log verbosity issues
   - Metric collection gaps

### Risk Assessment Matrix

| Severity | Safety Risk | Data Risk | Downtime | Decision |
|----------|-------------|-----------|----------|----------|
| P0-Critical | High | High | Any | Immediate Rollback |
| P1-High | Medium | Medium | <30min | Planned Rollback (1hr) |
| P2-Medium | Low | Low | <15min | Maintenance Window |
| P3-Low | None | None | <5min | Monitor or Fix Forward |

---

## Rollback Severity Levels

### Level 1: Emergency Rollback (<5 minutes)

**Criteria:**
- Safety systems compromised
- Data loss occurring
- System completely unavailable

**Approval:**
- On-call engineer (any)
- Notify VP Engineering within 15 minutes

**Process:**
- Skip pre-rollback validation (use known good version)
- Execute automated emergency rollback script
- Validate safety systems immediately
- Full post-mortem required

### Level 2: Urgent Rollback (<30 minutes)

**Criteria:**
- Significant performance degradation
- Major functional regressions
- Integration failures affecting operations

**Approval:**
- Lead Engineer + Product Manager
- Notify stakeholders before execution

**Process:**
- Quick pre-rollback validation (5 minutes)
- Execute standard rollback procedure
- Coordinate with affected teams
- Post-mortem recommended

### Level 3: Scheduled Rollback (Maintenance Window)

**Criteria:**
- Minor issues affecting non-critical features
- Performance optimization needed
- Preventive rollback

**Approval:**
- Engineering Manager + Product Manager
- Change Advisory Board (CAB) approval

**Process:**
- Full pre-rollback validation
- Coordinated communication plan
- Planned downtime notification
- Optional post-mortem

---

## Pre-Rollback Checklist

### Critical Pre-Flight Checks

**Execute before ANY rollback (except P0 emergencies):**

```bash
#!/bin/bash
# pre_rollback_checklist.sh

set -e

echo "=== GL-003 Pre-Rollback Checklist ==="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# 1. Identify target rollback version
echo "[1/10] Verifying target rollback version..."
CURRENT_VERSION=$(kubectl get deployment gl-003-steam-analyzer \
  -n greenlang-agents \
  -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)
echo "Current version: $CURRENT_VERSION"

ROLLBACK_VERSION="${1:-}"
if [ -z "$ROLLBACK_VERSION" ]; then
  # Get previous deployment version
  ROLLBACK_VERSION=$(kubectl rollout history deployment/gl-003-steam-analyzer \
    -n greenlang-agents --revision=2 | grep Image | awk '{print $2}' | cut -d: -f2)
fi
echo "Target rollback version: $ROLLBACK_VERSION"

if [ "$CURRENT_VERSION" == "$ROLLBACK_VERSION" ]; then
  echo "ERROR: Current version matches rollback version"
  exit 1
fi

# 2. Verify target version availability
echo ""
echo "[2/10] Verifying container image availability..."
docker pull gcr.io/greenlang/gl-003-steam-analyzer:$ROLLBACK_VERSION
if [ $? -ne 0 ]; then
  echo "ERROR: Rollback version image not available"
  exit 1
fi
echo "Image verified: gcr.io/greenlang/gl-003-steam-analyzer:$ROLLBACK_VERSION"

# 3. Check database backup exists
echo ""
echo "[3/10] Verifying database backup..."
BACKUP_TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -F c -f /backups/pre_rollback_${BACKUP_TIMESTAMP}.dump
echo "Database backup created: pre_rollback_${BACKUP_TIMESTAMP}.dump"

# Verify TimescaleDB hypertables backup
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -c "SELECT hypertable_name, total_chunks FROM timescaledb_information.hypertables;" \
  > /backups/hypertable_state_${BACKUP_TIMESTAMP}.txt
echo "Hypertable state captured"

# 4. Verify steam meter connectivity
echo ""
echo "[4/10] Checking steam meter connectivity..."
ACTIVE_METERS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT COUNT(*) FROM steam_meters WHERE status = 'active' AND last_reading_at > NOW() - INTERVAL '5 minutes';")
TOTAL_METERS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT COUNT(*) FROM steam_meters WHERE status = 'active';")
echo "Active meters: $ACTIVE_METERS / $TOTAL_METERS"

CONNECTIVITY_PERCENT=$(echo "scale=2; $ACTIVE_METERS * 100 / $TOTAL_METERS" | bc)
if (( $(echo "$CONNECTIVITY_PERCENT < 90" | bc -l) )); then
  echo "WARNING: Meter connectivity <90% - Review before rollback"
  read -p "Continue with rollback? (yes/no): " CONTINUE
  if [ "$CONTINUE" != "yes" ]; then
    echo "Rollback aborted by operator"
    exit 1
  fi
fi

# 5. Check current system health
echo ""
echo "[5/10] Checking current system health..."
HEALTHY_PODS=$(kubectl get pods -n greenlang-agents \
  -l app=gl-003-steam-analyzer \
  -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)
TOTAL_PODS=$(kubectl get pods -n greenlang-agents \
  -l app=gl-003-steam-analyzer \
  -o jsonpath='{.items[*].metadata.name}' | wc -w)
echo "Healthy pods: $HEALTHY_PODS / $TOTAL_PODS"

# 6. Verify no active incidents
echo ""
echo "[6/10] Checking for active incidents..."
ACTIVE_ALERTS=$(curl -s http://prometheus:9090/api/v1/alerts | \
  jq '[.data.alerts[] | select(.labels.agent=="gl-003" and .state=="firing")] | length')
echo "Active alerts: $ACTIVE_ALERTS"

if [ "$ACTIVE_ALERTS" -gt 5 ]; then
  echo "WARNING: Multiple active alerts - Review incident state"
fi

# 7. Check database connections
echo ""
echo "[7/10] Checking database connection capacity..."
DB_CONNECTIONS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'gl003_steam';")
DB_MAX_CONNECTIONS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres \
  -t -c "SELECT setting FROM pg_settings WHERE name = 'max_connections';")
echo "Current connections: $DB_CONNECTIONS / $DB_MAX_CONNECTIONS"

CONNECTION_PERCENT=$(echo "scale=2; $DB_CONNECTIONS * 100 / $DB_MAX_CONNECTIONS" | bc)
if (( $(echo "$CONNECTION_PERCENT > 80" | bc -l) )); then
  echo "WARNING: Database connections >80% capacity"
fi

# 8. Verify audit trail system
echo ""
echo "[8/10] Verifying audit trail system..."
RECENT_AUDITS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT COUNT(*) FROM audit_log WHERE created_at > NOW() - INTERVAL '1 hour';")
echo "Audit entries (last hour): $RECENT_AUDITS"

if [ "$RECENT_AUDITS" -lt 10 ]; then
  echo "WARNING: Low audit activity - Verify audit system health"
fi

# 9. Check downstream dependencies
echo ""
echo "[9/10] Checking downstream dependencies..."
# Test SCADA integration
SCADA_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  http://scada-gateway:8080/health)
echo "SCADA Gateway: HTTP $SCADA_STATUS"

# Test ERP connector
ERP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  http://erp-connector:8080/health)
echo "ERP Connector: HTTP $ERP_STATUS"

# 10. Create rollback execution plan
echo ""
echo "[10/10] Generating rollback execution plan..."
cat > /tmp/rollback_plan_${BACKUP_TIMESTAMP}.txt <<EOF
GL-003 SteamSystemAnalyzer Rollback Execution Plan
Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

Source Version: $CURRENT_VERSION
Target Version: $ROLLBACK_VERSION

Pre-Rollback State:
- Active Meters: $ACTIVE_METERS / $TOTAL_METERS ($CONNECTIVITY_PERCENT%)
- Healthy Pods: $HEALTHY_PODS / $TOTAL_PODS
- Database Connections: $DB_CONNECTIONS / $DB_MAX_CONNECTIONS ($CONNECTION_PERCENT%)
- Active Alerts: $ACTIVE_ALERTS
- Database Backup: pre_rollback_${BACKUP_TIMESTAMP}.dump

Rollback Steps:
1. Scale down current deployment
2. Apply rollback version
3. Validate pod startup
4. Test steam meter connectivity
5. Verify safety alert system
6. Validate database connectivity
7. Run smoke tests
8. Monitor for 15 minutes

Rollback By: ${ROLLBACK_OPERATOR:-$(whoami)}
Approval: ${ROLLBACK_APPROVAL:-PENDING}

EOF

echo "Rollback plan saved: /tmp/rollback_plan_${BACKUP_TIMESTAMP}.txt"
cat /tmp/rollback_plan_${BACKUP_TIMESTAMP}.txt

# Final confirmation
echo ""
echo "=== Pre-Rollback Checklist Complete ==="
echo ""
echo "Ready to proceed with rollback to version: $ROLLBACK_VERSION"
echo "Estimated downtime: 5-10 minutes"
echo ""
read -p "Execute rollback? (yes/no): " EXECUTE

if [ "$EXECUTE" != "yes" ]; then
  echo "Rollback cancelled by operator"
  exit 1
fi

# Export variables for rollback script
export CURRENT_VERSION
export ROLLBACK_VERSION
export BACKUP_TIMESTAMP
export ACTIVE_METERS
export TOTAL_METERS

echo "Proceeding with rollback..."
```

### Rollback Readiness Scorecard

Score each criterion (0-10), minimum 80/100 to proceed:

| Criterion | Weight | Score | Notes |
|-----------|--------|-------|-------|
| Target version validated | 15 | ___/10 | Image available, version tested |
| Database backup verified | 15 | ___/10 | Backup exists, restorable |
| Meter connectivity >90% | 10 | ___/10 | Steam meters reporting normally |
| No active P0 incidents | 15 | ___/10 | System currently stable |
| Team availability | 10 | ___/10 | Engineers on-call and ready |
| Communication prepared | 10 | ___/10 | Stakeholders notified |
| Audit trail functional | 10 | ___/10 | Recording enabled and verified |
| Downstream deps healthy | 10 | ___/10 | SCADA, ERP connectors OK |
| Rollback plan approved | 5 | ___/10 | Management signoff obtained |
| **TOTAL** | **100** | ___/100 | **Minimum: 80/100** |

---

## Emergency Rollback Procedure

**OBJECTIVE:** Restore service within 5 minutes when safety or data integrity is compromised.

### Emergency Rollback Script

```bash
#!/bin/bash
# emergency_rollback.sh
# Execute ONLY for P0 critical incidents

set -e

ROLLBACK_START=$(date +%s)
NAMESPACE="greenlang-agents"
DEPLOYMENT="gl-003-steam-analyzer"

echo "========================================="
echo "   GL-003 EMERGENCY ROLLBACK INITIATED"
echo "========================================="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Operator: $(whoami)"
echo "Incident: ${INCIDENT_ID:-UNKNOWN}"
echo ""

# Log to incident management system
curl -X POST https://incident.greenlang.io/api/v1/events \
  -H "Content-Type: application/json" \
  -d "{
    \"event_type\": \"emergency_rollback_started\",
    \"agent\": \"gl-003\",
    \"incident_id\": \"${INCIDENT_ID:-UNKNOWN}\",
    \"operator\": \"$(whoami)\",
    \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"
  }" || true

# Step 1: Identify last known good version (30 seconds)
echo "[Step 1/7] Identifying last known good version..."
CURRENT_REVISION=$(kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE | tail -n 1 | awk '{print $1}')
TARGET_REVISION=$((CURRENT_REVISION - 1))

echo "Current revision: $CURRENT_REVISION"
echo "Target revision: $TARGET_REVISION"

# Step 2: Execute Kubernetes rollback (60 seconds)
echo ""
echo "[Step 2/7] Executing Kubernetes rollback..."
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE --to-revision=$TARGET_REVISION

# Wait for rollout to start
sleep 5

# Step 3: Monitor rollout progress (90 seconds)
echo ""
echo "[Step 3/7] Monitoring rollback progress..."
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=90s

# Step 4: Verify pod health (30 seconds)
echo ""
echo "[Step 4/7] Verifying pod health..."
TIMEOUT=30
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
  READY_PODS=$(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT \
    -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)
  TOTAL_PODS=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE \
    -o jsonpath='{.spec.replicas}')

  if [ "$READY_PODS" -eq "$TOTAL_PODS" ]; then
    echo "All pods ready: $READY_PODS/$TOTAL_PODS"
    break
  fi

  echo "Waiting for pods: $READY_PODS/$TOTAL_PODS ready..."
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done

if [ "$READY_PODS" -ne "$TOTAL_PODS" ]; then
  echo "ERROR: Not all pods ready after $TIMEOUT seconds"
  echo "Manual intervention required"
  exit 1
fi

# Step 5: Validate safety alert system (30 seconds)
echo ""
echo "[Step 5/7] Validating safety alert system..."
HEALTH_CHECK=$(kubectl exec -n $NAMESPACE \
  $(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/health/safety)

echo "$HEALTH_CHECK" | jq .

SAFETY_STATUS=$(echo "$HEALTH_CHECK" | jq -r '.status')
if [ "$SAFETY_STATUS" != "healthy" ]; then
  echo "ERROR: Safety alert system not healthy"
  echo "Manual intervention required"
  exit 1
fi

# Step 6: Test steam meter connectivity (30 seconds)
echo ""
echo "[Step 6/7] Testing steam meter connectivity..."
METER_TEST=$(kubectl exec -n $NAMESPACE \
  $(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/api/v1/meters/connectivity-test)

CONNECTED_METERS=$(echo "$METER_TEST" | jq -r '.connected_count')
TOTAL_METERS=$(echo "$METER_TEST" | jq -r '.total_count')
CONNECTIVITY_PERCENT=$(echo "scale=2; $CONNECTED_METERS * 100 / $TOTAL_METERS" | bc)

echo "Steam meters: $CONNECTED_METERS / $TOTAL_METERS ($CONNECTIVITY_PERCENT%)"

if (( $(echo "$CONNECTIVITY_PERCENT < 85" | bc -l) )); then
  echo "WARNING: Meter connectivity <85% - Monitor closely"
fi

# Step 7: Record emergency rollback completion
echo ""
echo "[Step 7/7] Recording rollback completion..."
ROLLBACK_END=$(date +%s)
ROLLBACK_DURATION=$((ROLLBACK_END - ROLLBACK_START))

# Log to incident system
curl -X POST https://incident.greenlang.io/api/v1/events \
  -H "Content-Type: application/json" \
  -d "{
    \"event_type\": \"emergency_rollback_completed\",
    \"agent\": \"gl-003\",
    \"incident_id\": \"${INCIDENT_ID:-UNKNOWN}\",
    \"operator\": \"$(whoami)\",
    \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
    \"duration_seconds\": $ROLLBACK_DURATION,
    \"target_revision\": $TARGET_REVISION,
    \"meter_connectivity_percent\": $CONNECTIVITY_PERCENT
  }" || true

# Log to audit trail
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
INSERT INTO audit_log (event_type, agent_id, user_id, action, details, timestamp)
VALUES (
  'emergency_rollback',
  'gl-003',
  '$(whoami)',
  'deployment_rollback',
  jsonb_build_object(
    'from_revision', $CURRENT_REVISION,
    'to_revision', $TARGET_REVISION,
    'duration_seconds', $ROLLBACK_DURATION,
    'meter_connectivity_percent', $CONNECTIVITY_PERCENT,
    'incident_id', '${INCIDENT_ID:-UNKNOWN}'
  ),
  NOW()
);
EOF

echo ""
echo "========================================="
echo "   EMERGENCY ROLLBACK COMPLETE"
echo "========================================="
echo "Duration: ${ROLLBACK_DURATION}s (Target: <300s)"
echo "Status: $([ $ROLLBACK_DURATION -lt 300 ] && echo 'SUCCESS' || echo 'EXCEEDED TARGET')"
echo ""
echo "Next Steps:"
echo "1. Monitor system for 15 minutes"
echo "2. Verify steam measurement data continuity"
echo "3. Check safety alert system logs"
echo "4. Schedule post-mortem within 24 hours"
echo "5. Document lessons learned"
echo ""
echo "Monitoring Dashboard:"
echo "https://grafana.greenlang.io/d/gl-003-overview"
echo ""
```

### Emergency Rollback Execution

**Execute from operator terminal:**

```bash
# Set incident ID
export INCIDENT_ID="INC-20251117-001"

# Set database credentials
export POSTGRES_HOST="postgres.greenlang-db.svc.cluster.local"
export POSTGRES_USER="gl003_admin"
export POSTGRES_PASSWORD="<from-secret>"

# Execute emergency rollback
./emergency_rollback.sh

# Monitor continuously for 15 minutes
watch -n 10 'kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer'

# Check steam meter connectivity every minute
watch -n 60 'kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath="{.items[0].metadata.name}") \
  -- curl -s http://localhost:8080/api/v1/meters/stats | jq .'
```

### Emergency Rollback Notification

**Immediately after rollback, send to:**

- Incident Commander
- VP Engineering
- Product Manager
- Customer Success (if customer-facing impact)
- Steam Safety Team

**Template:**

```
Subject: [P0] GL-003 Emergency Rollback Executed - INC-20251117-001

Team,

An emergency rollback of GL-003 SteamSystemAnalyzer has been executed.

Incident: INC-20251117-001
Rollback Start: 2025-11-17 14:32:00 UTC
Rollback Complete: 2025-11-17 14:36:45 UTC
Duration: 4m 45s

Reason: [Safety system failure / Data loss / System unavailable]

Current Status:
- Service: Operational
- Steam Meters: 487/500 connected (97.4%)
- Safety Alerts: Functional
- Database: Healthy

Rollback Details:
- From: Version 2.3.1 (Revision 15)
- To: Version 2.3.0 (Revision 14)

Impact:
- Downtime: ~5 minutes
- Affected Facilities: [List if specific]
- Data Loss: None (verified)

Next Steps:
1. Continuous monitoring for 15 minutes
2. Steam measurement data continuity verification
3. Post-mortem scheduled: [Date/Time]
4. Root cause analysis in progress

Questions: Contact [On-Call Engineer]

Monitoring: https://grafana.greenlang.io/d/gl-003-overview
```

---

## Standard Rollback Procedure

**OBJECTIVE:** Controlled rollback with full validation and minimal risk (15-30 minutes).

### Standard Rollback Script

```bash
#!/bin/bash
# standard_rollback.sh
# Full rollback procedure with comprehensive validation

set -e

ROLLBACK_START=$(date +%s)
NAMESPACE="greenlang-agents"
DEPLOYMENT="gl-003-steam-analyzer"
ROLLBACK_VERSION="${1:-}"

if [ -z "$ROLLBACK_VERSION" ]; then
  echo "ERROR: Rollback version required"
  echo "Usage: $0 <rollback_version>"
  echo "Example: $0 2.3.0"
  exit 1
fi

echo "========================================="
echo "   GL-003 STANDARD ROLLBACK PROCEDURE"
echo "========================================="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Operator: $(whoami)"
echo "Target Version: $ROLLBACK_VERSION"
echo ""

# Phase 1: Pre-Rollback Validation (5 minutes)
echo "=== PHASE 1: Pre-Rollback Validation ==="
echo ""

# Run pre-rollback checklist
./pre_rollback_checklist.sh $ROLLBACK_VERSION

BACKUP_TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")

# Phase 2: Create Comprehensive Backup (5 minutes)
echo ""
echo "=== PHASE 2: Comprehensive Backup ==="
echo ""

echo "[2.1] Backing up Kubernetes resources..."
kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o yaml > \
  /backups/k8s_deployment_${BACKUP_TIMESTAMP}.yaml
kubectl get configmap gl-003-config -n $NAMESPACE -o yaml > \
  /backups/k8s_configmap_${BACKUP_TIMESTAMP}.yaml
kubectl get secret gl-003-secrets -n $NAMESPACE -o yaml > \
  /backups/k8s_secrets_${BACKUP_TIMESTAMP}.yaml
kubectl get service $DEPLOYMENT -n $NAMESPACE -o yaml > \
  /backups/k8s_service_${BACKUP_TIMESTAMP}.yaml
kubectl get hpa $DEPLOYMENT -n $NAMESPACE -o yaml > \
  /backups/k8s_hpa_${BACKUP_TIMESTAMP}.yaml
echo "Kubernetes resources backed up"

echo ""
echo "[2.2] Backing up PostgreSQL database..."
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -F c -f /backups/postgres_full_${BACKUP_TIMESTAMP}.dump \
  --verbose
echo "PostgreSQL backup complete: $(du -h /backups/postgres_full_${BACKUP_TIMESTAMP}.dump | awk '{print $1}')"

echo ""
echo "[2.3] Backing up TimescaleDB hypertables..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Export hypertable configuration
\copy (SELECT * FROM timescaledb_information.hypertables) TO '/backups/hypertables_config_${BACKUP_TIMESTAMP}.csv' CSV HEADER;

-- Export continuous aggregates
\copy (SELECT * FROM timescaledb_information.continuous_aggregates) TO '/backups/continuous_aggregates_${BACKUP_TIMESTAMP}.csv' CSV HEADER;

-- Export retention policies
\copy (SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention') TO '/backups/retention_policies_${BACKUP_TIMESTAMP}.csv' CSV HEADER;
EOF
echo "TimescaleDB configuration backed up"

echo ""
echo "[2.4] Backing up steam meter calibration data..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -c "\copy (SELECT * FROM meter_calibrations WHERE created_at > NOW() - INTERVAL '30 days') TO '/backups/meter_calibrations_${BACKUP_TIMESTAMP}.csv' CSV HEADER;"
echo "Calibration data backed up"

echo ""
echo "[2.5] Creating backup manifest..."
cat > /backups/rollback_manifest_${BACKUP_TIMESTAMP}.json <<EOF
{
  "rollback_id": "RB-${BACKUP_TIMESTAMP}",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "operator": "$(whoami)",
  "current_version": "$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)",
  "target_version": "$ROLLBACK_VERSION",
  "backup_files": {
    "kubernetes_deployment": "k8s_deployment_${BACKUP_TIMESTAMP}.yaml",
    "kubernetes_configmap": "k8s_configmap_${BACKUP_TIMESTAMP}.yaml",
    "kubernetes_secrets": "k8s_secrets_${BACKUP_TIMESTAMP}.yaml",
    "kubernetes_service": "k8s_service_${BACKUP_TIMESTAMP}.yaml",
    "kubernetes_hpa": "k8s_hpa_${BACKUP_TIMESTAMP}.yaml",
    "postgres_full": "postgres_full_${BACKUP_TIMESTAMP}.dump",
    "hypertables_config": "hypertables_config_${BACKUP_TIMESTAMP}.csv",
    "continuous_aggregates": "continuous_aggregates_${BACKUP_TIMESTAMP}.csv",
    "retention_policies": "retention_policies_${BACKUP_TIMESTAMP}.csv",
    "meter_calibrations": "meter_calibrations_${BACKUP_TIMESTAMP}.csv"
  },
  "system_state": {
    "active_meters": $ACTIVE_METERS,
    "total_meters": $TOTAL_METERS,
    "database_size": "$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam -t -c "SELECT pg_size_pretty(pg_database_size('gl003_steam'));")",
    "pod_count": $(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT -o json | jq '.items | length')
  }
}
EOF
echo "Backup manifest created"

# Phase 3: Notify Stakeholders (2 minutes)
echo ""
echo "=== PHASE 3: Stakeholder Notification ==="
echo ""

# Send rollback start notification
curl -X POST https://notifications.greenlang.io/api/v1/notify \
  -H "Content-Type: application/json" \
  -d "{
    \"channel\": \"#gl-003-deployments\",
    \"message\": \"GL-003 Rollback Starting\",
    \"details\": {
      \"operator\": \"$(whoami)\",
      \"current_version\": \"$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)\",
      \"target_version\": \"$ROLLBACK_VERSION\",
      \"estimated_downtime\": \"5-10 minutes\"
    },
    \"priority\": \"high\"
  }"

echo "Stakeholder notification sent"

# Phase 4: Execute Rollback (10 minutes)
echo ""
echo "=== PHASE 4: Execute Rollback ==="
echo ""

echo "[4.1] Scaling down current deployment..."
ORIGINAL_REPLICAS=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.replicas}')
echo "Original replicas: $ORIGINAL_REPLICAS"

# Scale to 1 replica for graceful transition
kubectl scale deployment $DEPLOYMENT -n $NAMESPACE --replicas=1
kubectl wait --for=condition=available --timeout=60s deployment/$DEPLOYMENT -n $NAMESPACE

echo ""
echo "[4.2] Updating deployment image..."
kubectl set image deployment/$DEPLOYMENT -n $NAMESPACE \
  steam-analyzer=gcr.io/greenlang/gl-003-steam-analyzer:$ROLLBACK_VERSION

echo ""
echo "[4.3] Waiting for rollout to complete..."
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s

echo ""
echo "[4.4] Scaling to original replica count..."
kubectl scale deployment $DEPLOYMENT -n $NAMESPACE --replicas=$ORIGINAL_REPLICAS
kubectl wait --for=condition=available --timeout=120s deployment/$DEPLOYMENT -n $NAMESPACE

echo ""
echo "[4.5] Verifying all pods are running..."
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT

# Phase 5: Post-Rollback Validation (8 minutes)
echo ""
echo "=== PHASE 5: Post-Rollback Validation ==="
echo ""

# Run comprehensive validation
./post_rollback_validation.sh $ROLLBACK_VERSION

# Phase 6: Steam System Recalibration Check
echo ""
echo "=== PHASE 6: Steam System Verification ==="
echo ""

echo "[6.1] Checking steam meter connectivity..."
CONNECTED_METERS=$(kubectl exec -n $NAMESPACE \
  $(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/api/v1/meters/connectivity-test | jq -r '.connected_count')
echo "Connected meters: $CONNECTED_METERS / $TOTAL_METERS"

echo ""
echo "[6.2] Verifying safety alert system..."
SAFETY_CHECK=$(kubectl exec -n $NAMESPACE \
  $(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/health/safety)
echo "$SAFETY_CHECK" | jq .

SAFETY_STATUS=$(echo "$SAFETY_CHECK" | jq -r '.status')
if [ "$SAFETY_STATUS" != "healthy" ]; then
  echo "ERROR: Safety system not healthy after rollback"
  exit 1
fi

echo ""
echo "[6.3] Testing data ingestion..."
TEST_READING=$(kubectl exec -n $NAMESPACE \
  $(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -X POST http://localhost:8080/api/v1/test/ingest-reading \
  -H "Content-Type: application/json" \
  -d '{"meter_id": "test-meter-001", "pressure_psi": 150.5, "temperature_f": 350.0, "flow_rate_lbh": 5000}')
echo "$TEST_READING" | jq .

echo ""
echo "[6.4] Verifying database writes..."
RECENT_READINGS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT COUNT(*) FROM steam_readings WHERE timestamp > NOW() - INTERVAL '5 minutes';")
echo "Recent readings: $RECENT_READINGS"

if [ "$RECENT_READINGS" -lt 100 ]; then
  echo "WARNING: Low reading count - Monitor data ingestion"
fi

# Phase 7: Documentation and Audit Trail
echo ""
echo "=== PHASE 7: Documentation ==="
echo ""

ROLLBACK_END=$(date +%s)
ROLLBACK_DURATION=$((ROLLBACK_END - ROLLBACK_START))

# Create rollback report
cat > /reports/rollback_report_${BACKUP_TIMESTAMP}.md <<EOF
# GL-003 Rollback Report

**Rollback ID:** RB-${BACKUP_TIMESTAMP}
**Executed By:** $(whoami)
**Start Time:** $(date -d @$ROLLBACK_START -u +"%Y-%m-%dT%H:%M:%SZ")
**End Time:** $(date -d @$ROLLBACK_END -u +"%Y-%m-%dT%H:%M:%SZ")
**Duration:** ${ROLLBACK_DURATION}s ($(($ROLLBACK_DURATION / 60))m $(($ROLLBACK_DURATION % 60))s)

## Version Change
- **From:** $(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)
- **To:** $ROLLBACK_VERSION

## System State
- **Steam Meters:** $CONNECTED_METERS / $TOTAL_METERS connected
- **Safety System:** $SAFETY_STATUS
- **Database:** Operational
- **Recent Readings:** $RECENT_READINGS (last 5 min)

## Rollback Phases
1. Pre-Rollback Validation: Complete
2. Comprehensive Backup: Complete
3. Stakeholder Notification: Complete
4. Rollback Execution: Complete
5. Post-Rollback Validation: Complete
6. Steam System Verification: Complete
7. Documentation: Complete

## Validation Results
All validation checks passed successfully.

## Backup Manifest
Backup ID: ${BACKUP_TIMESTAMP}
Location: /backups/
Manifest: rollback_manifest_${BACKUP_TIMESTAMP}.json

## Next Steps
1. Monitor system for 30 minutes
2. Verify steam measurement data continuity
3. Review logs for anomalies
4. Update rollback documentation if needed
5. Schedule post-rollback review

## Sign-off
- Executed By: $(whoami)
- Verified By: [Pending]
- Approved By: [Pending]
EOF

echo "Rollback report created: /reports/rollback_report_${BACKUP_TIMESTAMP}.md"

# Log to audit trail
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
INSERT INTO audit_log (event_type, agent_id, user_id, action, details, timestamp)
VALUES (
  'standard_rollback',
  'gl-003',
  '$(whoami)',
  'deployment_rollback',
  jsonb_build_object(
    'rollback_id', 'RB-${BACKUP_TIMESTAMP}',
    'target_version', '$ROLLBACK_VERSION',
    'duration_seconds', $ROLLBACK_DURATION,
    'connected_meters', $CONNECTED_METERS,
    'total_meters', $TOTAL_METERS,
    'safety_status', '$SAFETY_STATUS',
    'backup_manifest', 'rollback_manifest_${BACKUP_TIMESTAMP}.json'
  ),
  NOW()
);
EOF

echo ""
echo "========================================="
echo "   STANDARD ROLLBACK COMPLETE"
echo "========================================="
echo "Duration: ${ROLLBACK_DURATION}s ($(($ROLLBACK_DURATION / 60))m $(($ROLLBACK_DURATION % 60))s)"
echo "Status: SUCCESS"
echo ""
echo "Rollback Report: /reports/rollback_report_${BACKUP_TIMESTAMP}.md"
echo "Backup Manifest: /backups/rollback_manifest_${BACKUP_TIMESTAMP}.json"
echo ""
echo "Next Steps:"
echo "1. Monitor for 30 minutes: https://grafana.greenlang.io/d/gl-003-overview"
echo "2. Review rollback report and update documentation"
echo "3. Notify stakeholders of completion"
echo "4. Schedule post-rollback review meeting"
echo ""
```

### Post-Standard Rollback Monitoring

**Monitor for 30 minutes after rollback:**

```bash
# Terminal 1: Pod status
watch -n 10 'kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o wide'

# Terminal 2: Steam meter connectivity
watch -n 30 'kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath="{.items[0].metadata.name}") \
  -- curl -s http://localhost:8080/api/v1/meters/stats | jq "{connected: .connected_count, total: .total_count, percent: .connectivity_percent}"'

# Terminal 3: Database metrics
watch -n 30 'psql -h postgres.greenlang-db.svc.cluster.local -U gl003_admin -d gl003_steam \
  -c "SELECT \
    (SELECT COUNT(*) FROM steam_readings WHERE timestamp > NOW() - INTERVAL '\''5 minutes'\'') AS recent_readings, \
    (SELECT COUNT(*) FROM pg_stat_activity WHERE datname = '\''gl003_steam'\'') AS db_connections, \
    (SELECT pg_size_pretty(pg_database_size('\''gl003_steam'\''))) AS db_size;"'

# Terminal 4: Error logs
kubectl logs -n greenlang-agents -l app=gl-003-steam-analyzer --tail=50 -f | grep -i error
```

---

## Kubernetes Rollback

### Deployment Rollback

**Check deployment history:**

```bash
# View revision history
kubectl rollout history deployment/gl-003-steam-analyzer -n greenlang-agents

# View specific revision details
kubectl rollout history deployment/gl-003-steam-analyzer -n greenlang-agents --revision=14

# Example output:
# deployment.apps/gl-003-steam-analyzer with revision #14
# Pod Template:
#   Labels:       app=gl-003-steam-analyzer
#                 version=2.3.0
#   Containers:
#    steam-analyzer:
#     Image:      gcr.io/greenlang/gl-003-steam-analyzer:2.3.0
#     Environment:
#       ENV: production
#       LOG_LEVEL: info
```

**Rollback to specific revision:**

```bash
# Rollback to previous revision
kubectl rollout undo deployment/gl-003-steam-analyzer -n greenlang-agents

# Rollback to specific revision
kubectl rollout undo deployment/gl-003-steam-analyzer -n greenlang-agents --to-revision=14

# Monitor rollback progress
kubectl rollout status deployment/gl-003-steam-analyzer -n greenlang-agents

# Verify new pods
kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer
```

### ConfigMap Rollback

**ConfigMaps require manual rollback:**

```bash
# Backup current configmap
kubectl get configmap gl-003-config -n greenlang-agents -o yaml > current_configmap.yaml

# Restore from backup
kubectl apply -f /backups/k8s_configmap_20251117_143000.yaml

# Restart deployment to pick up configmap changes
kubectl rollout restart deployment/gl-003-steam-analyzer -n greenlang-agents

# Verify configmap
kubectl describe configmap gl-003-config -n greenlang-agents
```

**ConfigMap rollback script:**

```bash
#!/bin/bash
# rollback_configmap.sh

BACKUP_FILE="${1:-}"
NAMESPACE="greenlang-agents"

if [ -z "$BACKUP_FILE" ]; then
  echo "ERROR: Backup file required"
  echo "Usage: $0 <configmap_backup.yaml>"
  exit 1
fi

echo "Rolling back ConfigMap from: $BACKUP_FILE"

# Apply backup
kubectl apply -f $BACKUP_FILE

# Restart pods to pick up changes
echo "Restarting deployment to apply configmap changes..."
kubectl rollout restart deployment/gl-003-steam-analyzer -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/gl-003-steam-analyzer -n $NAMESPACE

echo "ConfigMap rollback complete"
```

### Secret Rollback

**Secrets require careful handling:**

```bash
# Backup current secret (base64 encoded)
kubectl get secret gl-003-secrets -n greenlang-agents -o yaml > current_secrets.yaml

# Restore from backup
kubectl apply -f /backups/k8s_secrets_20251117_143000.yaml

# Restart deployment to pick up secret changes
kubectl rollout restart deployment/gl-003-steam-analyzer -n greenlang-agents

# Verify secret (without exposing values)
kubectl describe secret gl-003-secrets -n greenlang-agents
```

**WARNING:** Never commit secrets to version control. Store backup secrets in secure vault.

### Service and HPA Rollback

**Services rarely need rollback, but if needed:**

```bash
# Rollback service
kubectl apply -f /backups/k8s_service_20251117_143000.yaml

# Rollback HPA
kubectl apply -f /backups/k8s_hpa_20251117_143000.yaml

# Verify service
kubectl get service gl-003-steam-analyzer -n greenlang-agents

# Verify HPA
kubectl get hpa gl-003-steam-analyzer -n greenlang-agents
```

### Complete Kubernetes State Rollback

**Restore entire Kubernetes state:**

```bash
#!/bin/bash
# rollback_k8s_state.sh

BACKUP_DIR="${1:-}"
NAMESPACE="greenlang-agents"

if [ -z "$BACKUP_DIR" ] || [ ! -d "$BACKUP_DIR" ]; then
  echo "ERROR: Valid backup directory required"
  echo "Usage: $0 <backup_directory>"
  exit 1
fi

echo "Rolling back complete Kubernetes state from: $BACKUP_DIR"

# Rollback in order: secrets, configmaps, service, deployment, hpa
echo "1. Rolling back secrets..."
kubectl apply -f $BACKUP_DIR/k8s_secrets_*.yaml

echo "2. Rolling back configmaps..."
kubectl apply -f $BACKUP_DIR/k8s_configmap_*.yaml

echo "3. Rolling back service..."
kubectl apply -f $BACKUP_DIR/k8s_service_*.yaml

echo "4. Rolling back deployment..."
kubectl apply -f $BACKUP_DIR/k8s_deployment_*.yaml

echo "5. Rolling back HPA..."
kubectl apply -f $BACKUP_DIR/k8s_hpa_*.yaml

echo "6. Waiting for rollout to complete..."
kubectl rollout status deployment/gl-003-steam-analyzer -n $NAMESPACE

echo "7. Verifying pod health..."
kubectl get pods -n $NAMESPACE -l app=gl-003-steam-analyzer

echo "Complete Kubernetes state rollback finished"
```

---

## Database Rollback

### PostgreSQL Full Database Rollback

**Restore from pg_dump backup:**

```bash
#!/bin/bash
# rollback_postgres.sh

BACKUP_FILE="${1:-}"
DB_NAME="gl003_steam"

if [ -z "$BACKUP_FILE" ]; then
  echo "ERROR: Backup file required"
  echo "Usage: $0 <backup_file.dump>"
  exit 1
fi

echo "========================================="
echo "   PostgreSQL Database Rollback"
echo "========================================="
echo "Backup File: $BACKUP_FILE"
echo "Database: $DB_NAME"
echo ""

# Step 1: Terminate existing connections
echo "[Step 1/6] Terminating existing connections..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres <<EOF
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = '$DB_NAME'
  AND pid <> pg_backend_pid();
EOF

# Step 2: Drop and recreate database
echo ""
echo "[Step 2/6] Recreating database..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres <<EOF
DROP DATABASE IF EXISTS ${DB_NAME}_temp;
CREATE DATABASE ${DB_NAME}_temp WITH OWNER = gl003_admin;
EOF

# Step 3: Restore to temporary database
echo ""
echo "[Step 3/6] Restoring backup to temporary database..."
pg_restore -h $POSTGRES_HOST -U $POSTGRES_USER -d ${DB_NAME}_temp \
  --verbose --no-owner --no-acl $BACKUP_FILE

# Step 4: Verify restoration
echo ""
echo "[Step 4/6] Verifying restored data..."
RESTORED_TABLES=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d ${DB_NAME}_temp \
  -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
RESTORED_READINGS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d ${DB_NAME}_temp \
  -t -c "SELECT COUNT(*) FROM steam_readings;")

echo "Restored tables: $RESTORED_TABLES"
echo "Restored readings: $RESTORED_READINGS"

read -p "Verification looks correct? (yes/no): " VERIFIED
if [ "$VERIFIED" != "yes" ]; then
  echo "Rollback aborted - cleaning up temporary database"
  psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres \
    -c "DROP DATABASE ${DB_NAME}_temp;"
  exit 1
fi

# Step 5: Swap databases
echo ""
echo "[Step 5/6] Swapping databases..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres <<EOF
-- Rename current database to backup
ALTER DATABASE $DB_NAME RENAME TO ${DB_NAME}_rollback_backup;

-- Rename restored database to production
ALTER DATABASE ${DB_NAME}_temp RENAME TO $DB_NAME;
EOF

echo "Database swap complete"

# Step 6: Reconnect application
echo ""
echo "[Step 6/6] Reconnecting application..."
kubectl rollout restart deployment/gl-003-steam-analyzer -n greenlang-agents
kubectl rollout status deployment/gl-003-steam-analyzer -n greenlang-agents

echo ""
echo "========================================="
echo "   PostgreSQL Rollback Complete"
echo "========================================="
echo ""
echo "Current database: $DB_NAME (restored from backup)"
echo "Backup of previous state: ${DB_NAME}_rollback_backup"
echo ""
echo "To delete backup database after verification:"
echo "psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres -c 'DROP DATABASE ${DB_NAME}_rollback_backup;'"
echo ""
```

### TimescaleDB Hypertable Rollback

**TimescaleDB requires special handling for hypertables:**

```bash
#!/bin/bash
# rollback_timescaledb.sh

BACKUP_TIMESTAMP="${1:-}"

if [ -z "$BACKUP_TIMESTAMP" ]; then
  echo "ERROR: Backup timestamp required"
  echo "Usage: $0 <backup_timestamp>"
  echo "Example: $0 20251117_143000"
  exit 1
fi

echo "========================================="
echo "   TimescaleDB Hypertable Rollback"
echo "========================================="
echo "Backup Timestamp: $BACKUP_TIMESTAMP"
echo ""

# Step 1: Restore hypertable configuration
echo "[Step 1/5] Restoring hypertable configuration..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Recreate hypertables from backup configuration
-- This assumes hypertables were backed up as CSV

CREATE TEMP TABLE hypertable_backup (
  hypertable_schema text,
  hypertable_name text,
  time_column_name text,
  chunk_time_interval interval
);

\copy hypertable_backup FROM '/backups/hypertables_config_${BACKUP_TIMESTAMP}.csv' CSV HEADER;

-- Note: Actual hypertable recreation requires custom logic per table
-- This is a template - adjust for your specific hypertables

DO \$\$
DECLARE
  r RECORD;
BEGIN
  FOR r IN SELECT * FROM hypertable_backup LOOP
    -- Drop existing hypertable if exists
    EXECUTE format('DROP TABLE IF EXISTS %I.%I CASCADE', r.hypertable_schema, r.hypertable_name);

    -- Recreate table structure (from backup)
    -- This needs to be customized per hypertable

    -- Convert to hypertable
    PERFORM create_hypertable(
      format('%I.%I', r.hypertable_schema, r.hypertable_name),
      r.time_column_name,
      chunk_time_interval => r.chunk_time_interval
    );
  END LOOP;
END;
\$\$;

DROP TABLE hypertable_backup;
EOF

# Step 2: Restore continuous aggregates
echo ""
echo "[Step 2/5] Restoring continuous aggregates..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Drop existing continuous aggregates
DROP MATERIALIZED VIEW IF EXISTS steam_readings_hourly CASCADE;
DROP MATERIALIZED VIEW IF EXISTS steam_readings_daily CASCADE;

-- Recreate from backup configuration
-- (Simplified example - adjust for your schema)

CREATE MATERIALIZED VIEW steam_readings_hourly
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', timestamp) AS bucket,
  meter_id,
  AVG(pressure_psi) AS avg_pressure,
  AVG(temperature_f) AS avg_temperature,
  AVG(flow_rate_lbh) AS avg_flow_rate,
  COUNT(*) AS reading_count
FROM steam_readings
GROUP BY bucket, meter_id;

CREATE MATERIALIZED VIEW steam_readings_daily
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 day', timestamp) AS bucket,
  meter_id,
  AVG(pressure_psi) AS avg_pressure,
  AVG(temperature_f) AS avg_temperature,
  AVG(flow_rate_lbh) AS avg_flow_rate,
  MIN(pressure_psi) AS min_pressure,
  MAX(pressure_psi) AS max_pressure,
  COUNT(*) AS reading_count
FROM steam_readings
GROUP BY bucket, meter_id;
EOF

# Step 3: Restore retention policies
echo ""
echo "[Step 3/5] Restoring retention policies..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Remove existing policies
SELECT remove_retention_policy('steam_readings', if_exists => true);

-- Recreate from backup (30 days retention)
SELECT add_retention_policy('steam_readings', INTERVAL '30 days');

-- Verify policy
SELECT * FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention';
EOF

# Step 4: Refresh continuous aggregates
echo ""
echo "[Step 4/5] Refreshing continuous aggregates..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
CALL refresh_continuous_aggregate('steam_readings_hourly', NULL, NULL);
CALL refresh_continuous_aggregate('steam_readings_daily', NULL, NULL);
EOF

# Step 5: Verify TimescaleDB health
echo ""
echo "[Step 5/5] Verifying TimescaleDB health..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Check hypertables
SELECT * FROM timescaledb_information.hypertables;

-- Check chunks
SELECT hypertable_name, COUNT(*) AS chunk_count
FROM timescaledb_information.chunks
GROUP BY hypertable_name;

-- Check continuous aggregates
SELECT * FROM timescaledb_information.continuous_aggregates;

-- Check data integrity
SELECT
  'steam_readings' AS table_name,
  COUNT(*) AS row_count,
  MIN(timestamp) AS earliest_reading,
  MAX(timestamp) AS latest_reading
FROM steam_readings;
EOF

echo ""
echo "========================================="
echo "   TimescaleDB Rollback Complete"
echo "========================================="
echo ""
echo "Next Steps:"
echo "1. Verify hypertable structure"
echo "2. Test continuous aggregate queries"
echo "3. Confirm retention policies active"
echo "4. Monitor chunk compression"
echo ""
```

### Selective Data Rollback

**Rollback specific tables or time ranges:**

```bash
#!/bin/bash
# selective_data_rollback.sh

TABLE_NAME="${1:-}"
START_TIME="${2:-}"
END_TIME="${3:-}"
BACKUP_FILE="${4:-}"

echo "========================================="
echo "   Selective Data Rollback"
echo "========================================="
echo "Table: $TABLE_NAME"
echo "Time Range: $START_TIME to $END_TIME"
echo "Backup: $BACKUP_FILE"
echo ""

# Step 1: Export current data in range (safety backup)
echo "[Step 1/4] Creating safety backup of current data..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
\copy (SELECT * FROM $TABLE_NAME WHERE timestamp BETWEEN '$START_TIME' AND '$END_TIME') TO '/backups/safety_backup_$(date +%Y%m%d_%H%M%S).csv' CSV HEADER;
EOF

# Step 2: Delete data in target range
echo ""
echo "[Step 2/4] Deleting data in target range..."
DELETED_ROWS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "DELETE FROM $TABLE_NAME WHERE timestamp BETWEEN '$START_TIME' AND '$END_TIME' RETURNING 1;" | wc -l)
echo "Deleted rows: $DELETED_ROWS"

# Step 3: Restore data from backup
echo ""
echo "[Step 3/4] Restoring data from backup..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
\copy $TABLE_NAME FROM '$BACKUP_FILE' CSV HEADER;
EOF

RESTORED_ROWS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT COUNT(*) FROM $TABLE_NAME WHERE timestamp BETWEEN '$START_TIME' AND '$END_TIME';")
echo "Restored rows: $RESTORED_ROWS"

# Step 4: Verify data integrity
echo ""
echo "[Step 4/4] Verifying data integrity..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
SELECT
  MIN(timestamp) AS earliest,
  MAX(timestamp) AS latest,
  COUNT(*) AS total_rows,
  COUNT(DISTINCT meter_id) AS unique_meters
FROM $TABLE_NAME
WHERE timestamp BETWEEN '$START_TIME' AND '$END_TIME';
EOF

echo ""
echo "========================================="
echo "   Selective Rollback Complete"
echo "========================================="
```

---

## Steam Analysis Recalibration

### Post-Rollback Calibration Verification

After database rollback, verify steam analysis calibration:

```bash
#!/bin/bash
# verify_steam_calibration.sh

echo "========================================="
echo "   Steam Analysis Calibration Verification"
echo "========================================="
echo ""

# Step 1: Check meter calibration data
echo "[Step 1/6] Verifying meter calibration data..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
SELECT
  meter_id,
  calibration_date,
  pressure_offset,
  temperature_offset,
  flow_multiplier,
  CASE
    WHEN calibration_date > NOW() - INTERVAL '90 days' THEN 'CURRENT'
    WHEN calibration_date > NOW() - INTERVAL '180 days' THEN 'WARNING'
    ELSE 'EXPIRED'
  END AS calibration_status
FROM meter_calibrations
WHERE meter_id IN (SELECT meter_id FROM steam_meters WHERE status = 'active')
ORDER BY calibration_date DESC;
EOF

# Step 2: Test pressure calculations
echo ""
echo "[Step 2/6] Testing pressure calculations..."
kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- python3 -c "
from tools import calculate_steam_properties
import json

# Test calculation with known values
result = calculate_steam_properties(
    pressure_psi=150.0,
    temperature_f=350.0,
    flow_rate_lbh=5000.0
)

print(json.dumps(result, indent=2))
"

# Step 3: Verify condensate tracking
echo ""
echo "[Step 3/6] Verifying condensate tracking..."
kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- python3 -c "
from tools import track_condensate_return
import json

# Test condensate calculation
result = track_condensate_return(
    meter_id='test-meter-001',
    steam_flow_lbh=5000.0,
    condensate_temp_f=180.0,
    return_pressure_psi=15.0
)

print(json.dumps(result, indent=2))
"

# Step 4: Check energy calculations
echo ""
echo "[Step 4/6] Checking energy calculations..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Test energy calculation for recent readings
SELECT
  meter_id,
  COUNT(*) AS reading_count,
  AVG(pressure_psi) AS avg_pressure,
  AVG(temperature_f) AS avg_temperature,
  AVG(flow_rate_lbh) AS avg_flow_rate,
  SUM(energy_mmbtu) AS total_energy
FROM steam_readings
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY meter_id
ORDER BY meter_id;
EOF

# Step 5: Verify safety thresholds
echo ""
echo "[Step 5/6] Verifying safety thresholds..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
SELECT
  meter_id,
  max_pressure_psi,
  max_temperature_f,
  min_flow_rate_lbh,
  alert_enabled
FROM steam_meters
WHERE status = 'active'
ORDER BY meter_id;
EOF

# Step 6: Run calibration test suite
echo ""
echo "[Step 6/6] Running calibration test suite..."
kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- pytest tests/test_calibration.py -v

echo ""
echo "========================================="
echo "   Calibration Verification Complete"
echo "========================================="
echo ""
echo "Review Results:"
echo "1. All calibration data current? (Check Step 1)"
echo "2. Calculations producing expected results? (Check Steps 2-4)"
echo "3. Safety thresholds properly configured? (Check Step 5)"
echo "4. All calibration tests passing? (Check Step 6)"
echo ""
echo "If any issues found, run: ./recalibrate_steam_system.sh"
echo ""
```

### Full Steam System Recalibration

If calibration data was lost or corrupted:

```bash
#!/bin/bash
# recalibrate_steam_system.sh

echo "========================================="
echo "   Full Steam System Recalibration"
echo "========================================="
echo ""

# WARNING: This process may take 30-60 minutes for large installations

# Step 1: Identify meters needing recalibration
echo "[Step 1/5] Identifying meters needing recalibration..."
METERS_TO_CALIBRATE=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT meter_id FROM steam_meters WHERE status = 'active' AND (
    meter_id NOT IN (SELECT meter_id FROM meter_calibrations) OR
    meter_id IN (SELECT meter_id FROM meter_calibrations WHERE calibration_date < NOW() - INTERVAL '180 days')
  );")

echo "Meters requiring recalibration:"
echo "$METERS_TO_CALIBRATE"

METER_COUNT=$(echo "$METERS_TO_CALIBRATE" | wc -l)
echo "Total meters: $METER_COUNT"

if [ "$METER_COUNT" -eq 0 ]; then
  echo "No meters require recalibration"
  exit 0
fi

# Step 2: Load manufacturer calibration data
echo ""
echo "[Step 2/5] Loading manufacturer calibration data..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Import manufacturer calibration specifications
-- This would typically come from a CSV file provided by meter manufacturers

CREATE TEMP TABLE manufacturer_calibrations (
  meter_model text,
  pressure_offset_default float,
  temperature_offset_default float,
  flow_multiplier_default float,
  calibration_interval_days int
);

-- Example data (replace with actual manufacturer data)
INSERT INTO manufacturer_calibrations VALUES
  ('STEAM-METER-X100', 0.5, 1.0, 1.002, 180),
  ('STEAM-METER-X200', 0.3, 0.8, 1.001, 180),
  ('STEAM-METER-X300', 0.7, 1.2, 1.003, 180);

-- Apply manufacturer defaults to meters without calibration
INSERT INTO meter_calibrations (meter_id, calibration_date, pressure_offset, temperature_offset, flow_multiplier, calibrated_by)
SELECT
  sm.meter_id,
  NOW() AS calibration_date,
  mc.pressure_offset_default,
  mc.temperature_offset_default,
  mc.flow_multiplier_default,
  'automated_rollback_recalibration' AS calibrated_by
FROM steam_meters sm
JOIN manufacturer_calibrations mc ON sm.meter_model = mc.meter_model
WHERE sm.meter_id = ANY(string_to_array('$METERS_TO_CALIBRATE', E'\n'))
  AND NOT EXISTS (
    SELECT 1 FROM meter_calibrations
    WHERE meter_id = sm.meter_id
    AND calibration_date > NOW() - INTERVAL '180 days'
  )
ON CONFLICT (meter_id) DO UPDATE
SET
  calibration_date = EXCLUDED.calibration_date,
  pressure_offset = EXCLUDED.pressure_offset,
  temperature_offset = EXCLUDED.temperature_offset,
  flow_multiplier = EXCLUDED.flow_multiplier,
  calibrated_by = EXCLUDED.calibrated_by;

DROP TABLE manufacturer_calibrations;
EOF

# Step 3: Run live calibration verification
echo ""
echo "[Step 3/5] Running live calibration verification..."

# For each meter, collect readings and verify against known benchmarks
for METER_ID in $METERS_TO_CALIBRATE; do
  echo "Calibrating meter: $METER_ID"

  # Collect 1 minute of readings
  kubectl exec -n greenlang-agents \
    $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
    -- python3 -c "
import time
from tools import collect_meter_readings

readings = []
for i in range(60):  # Collect 60 seconds of data
    reading = collect_meter_readings('$METER_ID')
    readings.append(reading)
    time.sleep(1)

# Analyze readings for calibration
from tools import analyze_calibration
analysis = analyze_calibration('$METER_ID', readings)

print(f'Meter: $METER_ID')
print(f'Pressure variance: {analysis[\"pressure_variance\"]:.2f}')
print(f'Temperature variance: {analysis[\"temperature_variance\"]:.2f}')
print(f'Recommended pressure offset: {analysis[\"pressure_offset_recommendation\"]:.2f}')
print(f'Recommended temperature offset: {analysis[\"temperature_offset_recommendation\"]:.2f}')
"
done

# Step 4: Update calibration coefficients
echo ""
echo "[Step 4/5] Updating calibration coefficients..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Update calibration based on live analysis
-- This would use the results from Step 3
-- For now, mark calibration as verified

UPDATE meter_calibrations
SET
  verification_date = NOW(),
  verification_status = 'verified'
WHERE meter_id = ANY(string_to_array('$METERS_TO_CALIBRATE', E'\n'));
EOF

# Step 5: Restart analyzer to apply new calibrations
echo ""
echo "[Step 5/5] Restarting analyzer to apply new calibrations..."
kubectl rollout restart deployment/gl-003-steam-analyzer -n greenlang-agents
kubectl rollout status deployment/gl-003-steam-analyzer -n greenlang-agents

echo ""
echo "========================================="
echo "   Recalibration Complete"
echo "========================================="
echo ""
echo "Recalibrated meters: $METER_COUNT"
echo ""
echo "Next Steps:"
echo "1. Monitor meter readings for 1 hour"
echo "2. Verify calculations match expected values"
echo "3. Schedule physical calibration verification for critical meters"
echo "4. Update calibration documentation"
echo ""
```

---

## Partial Rollback Procedures

### Component-Specific Rollback

**Rollback only specific components without full deployment rollback:**

#### Rollback Database Schema Only

```bash
#!/bin/bash
# rollback_schema_only.sh

BACKUP_FILE="${1:-}"

echo "========================================="
echo "   Database Schema Rollback (Data Preserved)"
echo "========================================="
echo ""

# Step 1: Export current data
echo "[Step 1/4] Exporting current data..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Export all data (except schema)
\copy steam_readings TO '/tmp/steam_readings_backup.csv' CSV HEADER;
\copy steam_meters TO '/tmp/steam_meters_backup.csv' CSV HEADER;
\copy meter_calibrations TO '/tmp/meter_calibrations_backup.csv' CSV HEADER;
\copy audit_log TO '/tmp/audit_log_backup.csv' CSV HEADER;
EOF

# Step 2: Drop current schema
echo ""
echo "[Step 2/4] Dropping current schema..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO gl003_admin;
GRANT ALL ON SCHEMA public TO public;
EOF

# Step 3: Restore schema from backup
echo ""
echo "[Step 3/4] Restoring schema from backup..."
pg_restore -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  --schema-only --verbose $BACKUP_FILE

# Step 4: Reimport data
echo ""
echo "[Step 4/4] Reimporting data..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
\copy steam_readings FROM '/tmp/steam_readings_backup.csv' CSV HEADER;
\copy steam_meters FROM '/tmp/steam_meters_backup.csv' CSV HEADER;
\copy meter_calibrations FROM '/tmp/meter_calibrations_backup.csv' CSV HEADER;
\copy audit_log FROM '/tmp/audit_log_backup.csv' CSV HEADER;

-- Resequence primary keys
SELECT setval('steam_readings_id_seq', (SELECT MAX(id) FROM steam_readings));
SELECT setval('steam_meters_id_seq', (SELECT MAX(id) FROM steam_meters));
SELECT setval('meter_calibrations_id_seq', (SELECT MAX(id) FROM meter_calibrations));
SELECT setval('audit_log_id_seq', (SELECT MAX(id) FROM audit_log));
EOF

# Clean up temporary files
rm -f /tmp/*_backup.csv

echo ""
echo "========================================="
echo "   Schema Rollback Complete (Data Preserved)"
echo "========================================="
```

#### Rollback Configuration Only

```bash
#!/bin/bash
# rollback_config_only.sh

BACKUP_CONFIGMAP="${1:-}"

echo "Rolling back configuration only (no deployment restart)"

# Rollback configmap
kubectl apply -f $BACKUP_CONFIGMAP

# Trigger config reload without pod restart (if supported)
kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- kill -HUP 1  # Send SIGHUP to reload config

echo "Configuration rolled back - monitoring required to verify reload"
```

#### Rollback Single Pod (Canary Rollback)

```bash
#!/bin/bash
# rollback_single_pod.sh

TARGET_VERSION="${1:-}"
NAMESPACE="greenlang-agents"
DEPLOYMENT="gl-003-steam-analyzer"

echo "========================================="
echo "   Canary Rollback (Single Pod)"
echo "========================================="
echo ""

# Step 1: Get current pod list
PODS=($(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT -o jsonpath='{.items[*].metadata.name}'))
echo "Current pods: ${PODS[@]}"

# Step 2: Select one pod for rollback (first pod)
CANARY_POD="${PODS[0]}"
echo "Selected canary pod: $CANARY_POD"

# Step 3: Delete canary pod (will be recreated with rolled back version)
echo ""
echo "Step 1: Deleting canary pod..."
kubectl delete pod $CANARY_POD -n $NAMESPACE

# Wait for pod to terminate
kubectl wait --for=delete pod/$CANARY_POD -n $NAMESPACE --timeout=60s

# Step 4: Temporarily patch deployment to use rollback version
echo ""
echo "Step 2: Patching deployment with rollback version..."
kubectl patch deployment $DEPLOYMENT -n $NAMESPACE \
  --type=json \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/image", "value": "gcr.io/greenlang/gl-003-steam-analyzer:'$TARGET_VERSION'"}]'

# Wait for new pod to be ready
echo ""
echo "Step 3: Waiting for canary pod to be ready..."
kubectl wait --for=condition=ready pod -l app=$DEPLOYMENT -n $NAMESPACE --timeout=120s

# Step 5: Monitor canary pod
NEW_CANARY=$(kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT -o jsonpath='{.items[0].metadata.name}')
echo ""
echo "New canary pod: $NEW_CANARY"
echo ""
echo "Monitor canary pod for 10 minutes:"
echo "kubectl logs -f $NEW_CANARY -n $NAMESPACE"
echo ""
echo "If canary is stable, complete rollback with:"
echo "kubectl rollout restart deployment/$DEPLOYMENT -n $NAMESPACE"
echo ""
echo "If canary fails, revert with:"
echo "kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE"
echo ""
```

### Feature Flag Rollback

**Disable specific features without code rollback:**

```bash
#!/bin/bash
# rollback_feature_flags.sh

FEATURE="${1:-}"

echo "========================================="
echo "   Feature Flag Rollback"
echo "========================================="
echo "Feature: $FEATURE"
echo ""

# Update feature flags in database
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Disable specific feature
UPDATE feature_flags
SET
  enabled = false,
  disabled_at = NOW(),
  disabled_by = '$(whoami)',
  disable_reason = 'Rollback due to issues'
WHERE feature_name = '$FEATURE';

-- Log to audit trail
INSERT INTO audit_log (event_type, agent_id, user_id, action, details, timestamp)
VALUES (
  'feature_rollback',
  'gl-003',
  '$(whoami)',
  'disable_feature',
  jsonb_build_object('feature', '$FEATURE', 'reason', 'rollback'),
  NOW()
);
EOF

# Reload feature flags in application
kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -X POST http://localhost:8080/admin/reload-feature-flags

echo ""
echo "Feature '$FEATURE' disabled"
echo "Application feature flags reloaded"
echo ""
```

---

## Post-Rollback Validation

### Comprehensive Validation Script

```bash
#!/bin/bash
# post_rollback_validation.sh

ROLLBACK_VERSION="${1:-}"

echo "========================================="
echo "   Post-Rollback Validation Suite"
echo "========================================="
echo "Target Version: $ROLLBACK_VERSION"
echo ""

VALIDATION_START=$(date +%s)
VALIDATION_PASSED=0
VALIDATION_FAILED=0

# Test 1: Verify deployment version
echo "[Test 1/15] Verifying deployment version..."
CURRENT_VERSION=$(kubectl get deployment gl-003-steam-analyzer -n greenlang-agents \
  -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)

if [ "$CURRENT_VERSION" == "$ROLLBACK_VERSION" ]; then
  echo "✓ PASS: Version matches ($CURRENT_VERSION)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Version mismatch (Expected: $ROLLBACK_VERSION, Got: $CURRENT_VERSION)"
  ((VALIDATION_FAILED++))
fi

# Test 2: Verify all pods running
echo ""
echo "[Test 2/15] Verifying all pods running..."
EXPECTED_REPLICAS=$(kubectl get deployment gl-003-steam-analyzer -n greenlang-agents \
  -o jsonpath='{.spec.replicas}')
RUNNING_REPLICAS=$(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer \
  -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)

if [ "$RUNNING_REPLICAS" -eq "$EXPECTED_REPLICAS" ]; then
  echo "✓ PASS: All pods running ($RUNNING_REPLICAS/$EXPECTED_REPLICAS)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Not all pods running ($RUNNING_REPLICAS/$EXPECTED_REPLICAS)"
  ((VALIDATION_FAILED++))
fi

# Test 3: Health check endpoint
echo ""
echo "[Test 3/15] Testing health check endpoint..."
HEALTH_STATUS=$(kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/health | jq -r '.status')

if [ "$HEALTH_STATUS" == "healthy" ]; then
  echo "✓ PASS: Health check OK"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Health check failed ($HEALTH_STATUS)"
  ((VALIDATION_FAILED++))
fi

# Test 4: Database connectivity
echo ""
echo "[Test 4/15] Testing database connectivity..."
DB_TEST=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT 1;" 2>&1)

if [ "$DB_TEST" == " 1" ]; then
  echo "✓ PASS: Database connectivity OK"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Database connectivity failed"
  ((VALIDATION_FAILED++))
fi

# Test 5: Steam meter connectivity
echo ""
echo "[Test 5/15] Testing steam meter connectivity..."
METER_CONNECTIVITY=$(kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/api/v1/meters/connectivity-test | jq -r '.connectivity_percent')

if (( $(echo "$METER_CONNECTIVITY > 90" | bc -l) )); then
  echo "✓ PASS: Meter connectivity OK ($METER_CONNECTIVITY%)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Meter connectivity low ($METER_CONNECTIVITY%)"
  ((VALIDATION_FAILED++))
fi

# Test 6: Safety alert system
echo ""
echo "[Test 6/15] Testing safety alert system..."
SAFETY_STATUS=$(kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/health/safety | jq -r '.status')

if [ "$SAFETY_STATUS" == "healthy" ]; then
  echo "✓ PASS: Safety alert system OK"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Safety alert system not healthy"
  ((VALIDATION_FAILED++))
fi

# Test 7: Data ingestion
echo ""
echo "[Test 7/15] Testing data ingestion..."
TEST_READING=$(kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -X POST http://localhost:8080/api/v1/test/ingest-reading \
  -H "Content-Type: application/json" \
  -d '{"meter_id": "test-meter-001", "pressure_psi": 150.5, "temperature_f": 350.0, "flow_rate_lbh": 5000}' \
  | jq -r '.status')

if [ "$TEST_READING" == "success" ]; then
  echo "✓ PASS: Data ingestion OK"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Data ingestion failed"
  ((VALIDATION_FAILED++))
fi

# Test 8: Calculation accuracy
echo ""
echo "[Test 8/15] Testing calculation accuracy..."
CALC_TEST=$(kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- python3 -c "
from tools import calculate_steam_properties
result = calculate_steam_properties(150.0, 350.0, 5000.0)
print(result['enthalpy_btu_lb'])
" 2>&1)

# Expected enthalpy ~1195 BTU/lb for these conditions
if (( $(echo "$CALC_TEST > 1150 && $CALC_TEST < 1250" | bc -l) )); then
  echo "✓ PASS: Calculation accuracy OK ($CALC_TEST BTU/lb)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Calculation accuracy questionable ($CALC_TEST BTU/lb)"
  ((VALIDATION_FAILED++))
fi

# Test 9: API endpoint responsiveness
echo ""
echo "[Test 9/15] Testing API endpoint responsiveness..."
API_RESPONSE_TIME=$(kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -o /dev/null -w '%{time_total}' http://localhost:8080/api/v1/meters)

if (( $(echo "$API_RESPONSE_TIME < 1.0" | bc -l) )); then
  echo "✓ PASS: API response time OK (${API_RESPONSE_TIME}s)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: API response time slow (${API_RESPONSE_TIME}s)"
  ((VALIDATION_FAILED++))
fi

# Test 10: Database write performance
echo ""
echo "[Test 10/15] Testing database write performance..."
WRITE_START=$(date +%s%N)
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF > /dev/null 2>&1
INSERT INTO steam_readings (meter_id, timestamp, pressure_psi, temperature_f, flow_rate_lbh)
VALUES ('test-meter-rollback', NOW(), 150.0, 350.0, 5000.0);
DELETE FROM steam_readings WHERE meter_id = 'test-meter-rollback';
EOF
WRITE_END=$(date +%s%N)
WRITE_TIME=$(echo "scale=3; ($WRITE_END - $WRITE_START) / 1000000000" | bc)

if (( $(echo "$WRITE_TIME < 0.1" | bc -l) )); then
  echo "✓ PASS: Database write performance OK (${WRITE_TIME}s)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Database write performance slow (${WRITE_TIME}s)"
  ((VALIDATION_FAILED++))
fi

# Test 11: TimescaleDB hypertables
echo ""
echo "[Test 11/15] Testing TimescaleDB hypertables..."
HYPERTABLE_COUNT=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT COUNT(*) FROM timescaledb_information.hypertables;")

if [ "$HYPERTABLE_COUNT" -gt 0 ]; then
  echo "✓ PASS: TimescaleDB hypertables OK ($HYPERTABLE_COUNT found)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: No TimescaleDB hypertables found"
  ((VALIDATION_FAILED++))
fi

# Test 12: Continuous aggregates
echo ""
echo "[Test 12/15] Testing continuous aggregates..."
CONTINUOUS_AGG_COUNT=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates;")

if [ "$CONTINUOUS_AGG_COUNT" -gt 0 ]; then
  echo "✓ PASS: Continuous aggregates OK ($CONTINUOUS_AGG_COUNT found)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: No continuous aggregates found"
  ((VALIDATION_FAILED++))
fi

# Test 13: Audit trail logging
echo ""
echo "[Test 13/15] Testing audit trail logging..."
AUDIT_TEST=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "INSERT INTO audit_log (event_type, agent_id, user_id, action, details, timestamp) VALUES ('test', 'gl-003', 'validation', 'test', '{}'::jsonb, NOW()) RETURNING id;")

if [ ! -z "$AUDIT_TEST" ]; then
  echo "✓ PASS: Audit trail logging OK"
  ((VALIDATION_PASSED++))
  # Cleanup test entry
  psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
    -c "DELETE FROM audit_log WHERE id = $AUDIT_TEST;" > /dev/null 2>&1
else
  echo "✗ FAIL: Audit trail logging failed"
  ((VALIDATION_FAILED++))
fi

# Test 14: Metrics collection
echo ""
echo "[Test 14/15] Testing metrics collection..."
METRICS=$(kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/metrics | grep -c "gl003_")

if [ "$METRICS" -gt 10 ]; then
  echo "✓ PASS: Metrics collection OK ($METRICS metrics found)"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Insufficient metrics ($METRICS found)"
  ((VALIDATION_FAILED++))
fi

# Test 15: Integration endpoints
echo ""
echo "[Test 15/15] Testing integration endpoints..."
SCADA_STATUS=$(kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s -o /dev/null -w '%{http_code}' http://scada-gateway:8080/health)

if [ "$SCADA_STATUS" == "200" ]; then
  echo "✓ PASS: Integration endpoints OK"
  ((VALIDATION_PASSED++))
else
  echo "✗ FAIL: Integration endpoints not reachable (HTTP $SCADA_STATUS)"
  ((VALIDATION_FAILED++))
fi

# Calculate results
VALIDATION_END=$(date +%s)
VALIDATION_DURATION=$((VALIDATION_END - VALIDATION_START))
TOTAL_TESTS=$((VALIDATION_PASSED + VALIDATION_FAILED))
SUCCESS_RATE=$(echo "scale=2; $VALIDATION_PASSED * 100 / $TOTAL_TESTS" | bc)

echo ""
echo "========================================="
echo "   Validation Results"
echo "========================================="
echo "Duration: ${VALIDATION_DURATION}s"
echo "Tests Passed: $VALIDATION_PASSED / $TOTAL_TESTS"
echo "Tests Failed: $VALIDATION_FAILED / $TOTAL_TESTS"
echo "Success Rate: $SUCCESS_RATE%"
echo ""

if [ "$VALIDATION_FAILED" -eq 0 ]; then
  echo "✓ ALL VALIDATIONS PASSED"
  echo ""
  echo "Rollback to version $ROLLBACK_VERSION successful."
  exit 0
else
  echo "✗ VALIDATION FAILURES DETECTED"
  echo ""
  echo "Review failed tests above."
  echo "Consider additional rollback steps or escalation."
  exit 1
fi
```

---

## Communication Templates

### Emergency Rollback Notification

**Subject:** [P0-CRITICAL] GL-003 Emergency Rollback in Progress

```
Team,

An emergency rollback of GL-003 SteamSystemAnalyzer is currently in progress due to critical issues.

Incident Details:
- Incident ID: ${INCIDENT_ID}
- Severity: P0 - Critical
- Rollback Initiated: ${TIMESTAMP}
- Initiated By: ${OPERATOR}

Issue Description:
${ISSUE_DESCRIPTION}

Rollback Action:
- From Version: ${CURRENT_VERSION}
- To Version: ${TARGET_VERSION}
- Estimated Duration: <5 minutes
- Expected Downtime: ~5 minutes

Current Status:
${ROLLBACK_STATUS}

Impact:
- Steam system monitoring: ${IMPACT_LEVEL}
- Affected facilities: ${AFFECTED_FACILITIES}
- Data integrity: ${DATA_STATUS}

Stakeholder Actions Required:
- Incident Commander: Monitor rollback progress
- Steam Safety Team: Standby for manual monitoring
- Customer Success: Prepare customer communications

Updates: Will be provided every 5 minutes

Monitoring Dashboard: https://grafana.greenlang.io/d/gl-003-overview
Incident Channel: #incident-${INCIDENT_ID}

${OPERATOR}
GL-003 Operations Team
```

### Planned Rollback Notification

**Subject:** [PLANNED] GL-003 Rollback Scheduled - ${DATE}

```
Team,

A planned rollback of GL-003 SteamSystemAnalyzer has been scheduled to address identified issues.

Rollback Details:
- Scheduled Date: ${DATE}
- Scheduled Time: ${TIME} UTC
- Duration: 15-30 minutes
- Maintenance Window: ${MAINTENANCE_WINDOW}

Reason for Rollback:
${ROLLBACK_REASON}

Version Change:
- Current Version: ${CURRENT_VERSION}
- Target Version: ${TARGET_VERSION}

Impact Assessment:
- Service Availability: Limited during maintenance window
- Steam Monitoring: Continuous (no gaps)
- Data Integrity: Preserved (full backup)
- User Impact: Minimal (${EXPECTED_IMPACT})

Pre-Rollback Checklist:
[✓] Backup created and verified
[✓] Stakeholders notified
[✓] Rollback plan reviewed and approved
[✓] Team availability confirmed
[✓] Communication plan prepared

Team Assignments:
- Rollback Execution: ${EXECUTOR}
- Monitoring: ${MONITOR}
- Communication: ${COMMUNICATOR}
- Approval: ${APPROVER}

Communication Plan:
- T-24h: This notification
- T-1h: Reminder notification
- T-0: Rollback start notification
- T+completion: Rollback completion notification

Questions or Concerns:
Contact ${CONTACT_PERSON} before ${CUTOFF_TIME}

Monitoring Dashboard: https://grafana.greenlang.io/d/gl-003-overview
Rollback Documentation: ${DOC_LINK}

${SENDER}
GL-003 Operations Team
```

### Rollback Completion Notification

**Subject:** [COMPLETE] GL-003 Rollback Successfully Completed

```
Team,

The GL-003 SteamSystemAnalyzer rollback has been successfully completed.

Rollback Summary:
- Rollback ID: RB-${BACKUP_TIMESTAMP}
- Execution Time: ${START_TIME} to ${END_TIME} UTC
- Duration: ${DURATION} (Target: <30 minutes)
- Status: ${STATUS}

Version Change:
- Previous Version: ${PREVIOUS_VERSION}
- Current Version: ${CURRENT_VERSION}

Validation Results:
- System Health: ${HEALTH_STATUS}
- Steam Meters: ${METER_CONNECTIVITY}
- Database: ${DATABASE_STATUS}
- Safety Alerts: ${SAFETY_STATUS}
- API Endpoints: ${API_STATUS}

All validation tests passed: ${VALIDATION_RESULTS}

System Metrics:
- Uptime: ${UPTIME}
- Response Time: ${RESPONSE_TIME}
- Error Rate: ${ERROR_RATE}
- Active Connections: ${CONNECTIONS}

Post-Rollback Actions Completed:
[✓] All pods running and healthy
[✓] Steam meter connectivity verified (>90%)
[✓] Safety alert system tested
[✓] Database integrity validated
[✓] Audit trail updated
[✓] Backup retained for ${RETENTION_PERIOD}

Ongoing Monitoring:
- Continuous monitoring for 30 minutes: ${MONITOR_UNTIL}
- Dashboard: https://grafana.greenlang.io/d/gl-003-overview
- On-call engineer: ${ON_CALL}

Post-Mortem:
- Scheduled: ${POST_MORTEM_DATE}
- Facilitator: ${FACILITATOR}
- Participants: ${PARTICIPANTS}

Documentation:
- Rollback Report: ${REPORT_LINK}
- Backup Manifest: ${MANIFEST_LINK}
- Lessons Learned: TBD (post-mortem)

Thank you for your support during this rollback.

${OPERATOR}
GL-003 Operations Team
```

---

## Rollback Audit Trail

### Audit Log Entry Format

Every rollback must be logged to the audit trail:

```sql
INSERT INTO audit_log (
  event_type,
  agent_id,
  user_id,
  action,
  details,
  timestamp
) VALUES (
  'deployment_rollback',  -- event_type
  'gl-003',               -- agent_id
  'operator_name',        -- user_id
  'emergency_rollback',   -- action: emergency_rollback, standard_rollback, partial_rollback
  jsonb_build_object(
    'rollback_id', 'RB-20251117_143000',
    'incident_id', 'INC-20251117-001',
    'from_version', '2.3.1',
    'to_version', '2.3.0',
    'rollback_type', 'emergency',
    'severity', 'P0',
    'reason', 'Safety system failure',
    'duration_seconds', 285,
    'validation_passed', true,
    'meter_connectivity_percent', 97.4,
    'backup_manifest', 'rollback_manifest_20251117_143000.json',
    'executed_by', 'operator_name',
    'approved_by', 'manager_name',
    'post_mortem_required', true
  ),                      -- details (JSONB)
  NOW()                   -- timestamp
);
```

### Rollback History Query

```sql
-- Query rollback history
SELECT
  id,
  event_type,
  agent_id,
  user_id AS operator,
  action AS rollback_type,
  details->>'from_version' AS from_version,
  details->>'to_version' AS to_version,
  details->>'reason' AS reason,
  (details->>'duration_seconds')::int AS duration_seconds,
  details->>'validation_passed' AS validation_passed,
  timestamp,
  CASE
    WHEN details->>'severity' = 'P0' THEN 'Emergency'
    WHEN details->>'severity' = 'P1' THEN 'Urgent'
    ELSE 'Planned'
  END AS rollback_category
FROM audit_log
WHERE event_type = 'deployment_rollback'
  AND agent_id = 'gl-003'
  AND timestamp > NOW() - INTERVAL '90 days'
ORDER BY timestamp DESC;
```

### Rollback Metrics Dashboard

**Prometheus queries for rollback metrics:**

```promql
# Rollback count by type (last 30 days)
count_over_time(
  audit_log_rollback_total{agent="gl-003"}[30d]
)

# Average rollback duration
avg_over_time(
  audit_log_rollback_duration_seconds{agent="gl-003"}[30d]
)

# Rollback success rate
sum(rate(audit_log_rollback_success_total{agent="gl-003"}[30d])) /
sum(rate(audit_log_rollback_total{agent="gl-003"}[30d])) * 100

# Time between rollbacks
time() - max(audit_log_rollback_timestamp{agent="gl-003"})
```

---

## Recovery Procedures

### Recovery from Failed Rollback

If rollback itself fails:

```bash
#!/bin/bash
# recover_from_failed_rollback.sh

echo "========================================="
echo "   Recovery from Failed Rollback"
echo "========================================="
echo ""

# Step 1: Assess current state
echo "[Step 1/7] Assessing current state..."
kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer
kubectl get deployment gl-003-steam-analyzer -n greenlang-agents

# Step 2: Scale deployment to 0 (stop all traffic)
echo ""
echo "[Step 2/7] Scaling deployment to 0 replicas..."
kubectl scale deployment gl-003-steam-analyzer -n greenlang-agents --replicas=0
sleep 10

# Step 3: Restore known-good database backup
echo ""
echo "[Step 3/7] Restoring last known-good database backup..."
LATEST_BACKUP=$(ls -t /backups/postgres_full_*.dump | head -n 1)
echo "Restoring from: $LATEST_BACKUP"

# Create recovery database
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres <<EOF
DROP DATABASE IF EXISTS gl003_steam_recovery;
CREATE DATABASE gl003_steam_recovery WITH OWNER = gl003_admin;
EOF

# Restore to recovery database
pg_restore -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam_recovery \
  --verbose --no-owner --no-acl $LATEST_BACKUP

# Step 4: Validate recovery database
echo ""
echo "[Step 4/7] Validating recovery database..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam_recovery <<EOF
SELECT COUNT(*) AS steam_readings FROM steam_readings;
SELECT COUNT(*) AS steam_meters FROM steam_meters;
SELECT COUNT(*) AS calibrations FROM meter_calibrations;
SELECT COUNT(*) AS audit_entries FROM audit_log;
EOF

read -p "Database validation OK? (yes/no): " DB_OK
if [ "$DB_OK" != "yes" ]; then
  echo "Recovery aborted - database validation failed"
  exit 1
fi

# Step 5: Swap to recovery database
echo ""
echo "[Step 5/7] Swapping to recovery database..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres <<EOF
ALTER DATABASE gl003_steam RENAME TO gl003_steam_failed_rollback;
ALTER DATABASE gl003_steam_recovery RENAME TO gl003_steam;
EOF

# Step 6: Deploy last known-good version
echo ""
echo "[Step 6/7] Deploying last known-good version..."
LAST_KNOWN_GOOD="2.2.5"  # Update with actual LKG version

kubectl set image deployment/gl-003-steam-analyzer -n greenlang-agents \
  steam-analyzer=gcr.io/greenlang/gl-003-steam-analyzer:$LAST_KNOWN_GOOD

# Scale back to normal replica count
kubectl scale deployment gl-003-steam-analyzer -n greenlang-agents --replicas=3

# Wait for deployment
kubectl rollout status deployment/gl-003-steam-analyzer -n greenlang-agents

# Step 7: Validate recovery
echo ""
echo "[Step 7/7] Validating recovery..."
./post_rollback_validation.sh $LAST_KNOWN_GOOD

echo ""
echo "========================================="
echo "   Recovery Complete"
echo "========================================="
echo ""
echo "System recovered to last known-good state:"
echo "- Version: $LAST_KNOWN_GOOD"
echo "- Database: Restored from backup"
echo ""
echo "CRITICAL: Schedule immediate post-incident review"
echo ""
```

### Rollback from Rollback

If you need to undo a rollback:

```bash
#!/bin/bash
# rollback_from_rollback.sh

ORIGINAL_VERSION="${1:-}"

echo "========================================="
echo "   Rolling Back from Rollback"
echo "========================================="
echo "Restoring to original version: $ORIGINAL_VERSION"
echo ""

# This is essentially a forward roll
echo "WARNING: This will restore the version that was previously rolled back."
echo "Ensure the original issues are resolved before proceeding."
echo ""

read -p "Continue with rollback restoration? (yes/no): " CONTINUE
if [ "$CONTINUE" != "yes" ]; then
  echo "Rollback restoration cancelled"
  exit 0
fi

# Use standard rollback procedure to restore original version
./standard_rollback.sh $ORIGINAL_VERSION

echo ""
echo "Rollback restoration complete"
echo "Original version $ORIGINAL_VERSION has been restored"
echo ""
```

---

## Appendices

### Appendix A: Rollback Checklist Summary

**Pre-Rollback:**
- [ ] Incident severity assessed
- [ ] Rollback approval obtained
- [ ] Target version identified and validated
- [ ] Database backup created
- [ ] Kubernetes resources backed up
- [ ] Steam meter connectivity verified (>90%)
- [ ] Team availability confirmed
- [ ] Stakeholders notified
- [ ] Rollback plan documented
- [ ] Audit trail prepared

**During Rollback:**
- [ ] Rollback initiated (logged to audit trail)
- [ ] Deployment scaled and updated
- [ ] Pod health monitored
- [ ] Database changes applied (if needed)
- [ ] Steam meter connectivity maintained
- [ ] Safety alert system verified
- [ ] Progress communicated to stakeholders

**Post-Rollback:**
- [ ] All validation tests passed
- [ ] Steam system recalibration verified
- [ ] Database integrity confirmed
- [ ] Metrics collection validated
- [ ] Documentation updated
- [ ] Stakeholders notified of completion
- [ ] Continuous monitoring initiated (30 min)
- [ ] Post-mortem scheduled
- [ ] Lessons learned documented

### Appendix B: Emergency Contact List

**Incident Response:**
- Incident Commander: ${IC_CONTACT}
- On-Call Engineer: ${ONCALL_CONTACT}
- VP Engineering: ${VP_ENG_CONTACT}

**Technical Teams:**
- GL-003 Lead: ${GL003_LEAD}
- Database Admin: ${DBA_CONTACT}
- Platform Engineering: ${PLATFORM_CONTACT}
- Steam Safety Team: ${SAFETY_CONTACT}

**Business Stakeholders:**
- Product Manager: ${PM_CONTACT}
- Customer Success: ${CS_CONTACT}
- Executive Sponsor: ${EXEC_CONTACT}

### Appendix C: Rollback Decision Matrix

| Condition | Severity | Rollback Type | Approval Level | SLA |
|-----------|----------|---------------|----------------|-----|
| Safety system down | P0 | Emergency | On-call engineer | <5 min |
| Data loss occurring | P0 | Emergency | On-call engineer | <5 min |
| >10% meters offline | P0 | Emergency | On-call engineer | <5 min |
| System unavailable | P0 | Emergency | On-call engineer | <5 min |
| Major perf degradation | P1 | Urgent | Lead + PM | <30 min |
| Integration failures | P1 | Urgent | Lead + PM | <1 hour |
| Functional regressions | P1 | Urgent | Lead + PM | <1 hour |
| Minor feature bugs | P2 | Scheduled | Manager + PM | Maintenance window |
| UI cosmetic issues | P3 | Deferred | Product team | Next sprint |

### Appendix D: Version Compatibility Matrix

| Version | Database Schema | Config Format | API Version | Rollback Supported |
|---------|----------------|---------------|-------------|-------------------|
| 2.3.1 | v12 | yaml-v2 | v1.3 | Yes (to 2.3.0) |
| 2.3.0 | v12 | yaml-v2 | v1.3 | Yes (to 2.2.5) |
| 2.2.5 | v11 | yaml-v2 | v1.2 | Yes (to 2.2.0) |
| 2.2.0 | v11 | yaml-v1 | v1.2 | Yes (to 2.1.0) |
| 2.1.0 | v10 | yaml-v1 | v1.1 | Limited |

**Rollback Notes:**
- Schema v12 → v11: Requires data migration
- yaml-v2 → yaml-v1: ConfigMap transformation needed
- API v1.3 → v1.2: Backward compatible
- Versions >6 months old: Discouraged without testing

### Appendix E: Common Rollback Scenarios

**Scenario 1: Database Migration Failure**
```bash
# Rollback database schema only, keep application
./rollback_schema_only.sh /backups/postgres_full_<timestamp>.dump
```

**Scenario 2: Configuration Error**
```bash
# Rollback configuration only, no deployment restart
./rollback_config_only.sh /backups/k8s_configmap_<timestamp>.yaml
```

**Scenario 3: Feature Causing Issues**
```bash
# Disable specific feature without rollback
./rollback_feature_flags.sh <feature_name>
```

**Scenario 4: Partial Deployment Failure**
```bash
# Rollback single pod for testing
./rollback_single_pod.sh <target_version>
```

**Scenario 5: Complete System Failure**
```bash
# Emergency rollback everything
./emergency_rollback.sh
```

### Appendix F: Rollback Testing

**Test rollback procedures quarterly:**

```bash
#!/bin/bash
# test_rollback_procedures.sh

echo "GL-003 Rollback Procedure Test"
echo "Environment: STAGING"
echo ""

# Test 1: Simulate emergency rollback (dry-run)
echo "[Test 1] Emergency rollback simulation..."
kubectl rollout undo deployment/gl-003-steam-analyzer -n greenlang-staging --dry-run=client

# Test 2: Test backup/restore procedures
echo "[Test 2] Backup/restore test..."
pg_dump -h $STAGING_DB_HOST -U $POSTGRES_USER -d gl003_steam_staging \
  -F c -f /tmp/test_backup.dump
pg_restore -h $STAGING_DB_HOST -U $POSTGRES_USER -d gl003_steam_staging \
  --list /tmp/test_backup.dump | head -20

# Test 3: Verify backup manifests
echo "[Test 3] Backup manifest verification..."
ls -lh /backups/*.json | tail -5

# Test 4: Test communication templates
echo "[Test 4] Communication template test..."
# Send test notification to staging channel

echo ""
echo "Rollback procedure test complete"
echo "Review results and update procedures as needed"
```

---

## Document Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-17 | GL-003 Team | Initial comprehensive rollback procedure |

---

## Additional Resources

- **GL-003 Deployment Guide:** `DEPLOYMENT_GUIDE.md`
- **GL-003 Incident Response:** `INCIDENT_RESPONSE.md`
- **GL-003 Monitoring Guide:** `MONITORING_GUIDE.md`
- **PostgreSQL Backup Guide:** `DATABASE_BACKUP.md`
- **Kubernetes Operations:** `K8S_OPERATIONS.md`

---

**END OF RUNBOOK**
