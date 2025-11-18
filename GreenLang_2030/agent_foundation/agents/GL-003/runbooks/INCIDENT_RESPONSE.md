# GL-003 SteamSystemAnalyzer - Incident Response Guide

Emergency procedures and incident response playbook for GL-003 SteamSystemAnalyzer production operations.

## Table of Contents

1. [Incident Severity Definitions](#incident-severity-definitions)
2. [Response Procedures by Severity](#response-procedures-by-severity)
3. [Steam System Specific Incidents](#steam-system-specific-incidents)
4. [Escalation Paths](#escalation-paths)
5. [Communication Templates](#communication-templates)
6. [Emergency Rollback Procedures](#emergency-rollback-procedures)
7. [Post-Incident Review](#post-incident-review)

---

## Incident Severity Definitions

### P0 - Critical (Production Down)

**Definition**: Complete service outage affecting all users and critical steam system monitoring operations.

**Examples**:
- All GL-003 pods are down (0/3 running)
- Database completely inaccessible
- **Critical steam leak detected but not reported**
- **All steam meter connectivity lost**
- **Steam trap monitoring completely offline**
- Data loss or corruption
- Security breach detected

**Response Time**: Immediate (5 minutes)

**Escalation**: Automatic to Director of Engineering + CTO + Plant Safety Officer (for steam safety issues)

**Communication**: Every 15 minutes until resolved

**Safety Note**: If incident involves potential steam safety issues (undetected leaks, failed emergency alerts), immediately notify Plant Safety Officer before technical remediation.

---

### P1 - High (Major Degradation)

**Definition**: Significant service degradation affecting majority of users or critical steam analysis features.

**Examples**:
- 2/3 pods down (33% capacity)
- Error rate >20%
- Response time >10 seconds (p95)
- **Steam leak detection accuracy <80% (normal: >95%)**
- **Steam trap classification failing (>10% misclassification)**
- **Distribution efficiency calculation errors**
- Integration failures with critical systems (SCADA, steam meters)
- Memory leak causing OOMKilled
- **SCADA connection intermittent (>50% packet loss)**
- **Condensate return optimization not working**

**Response Time**: 15 minutes

**Escalation**: Senior Engineer + Engineering Manager within 1 hour if not resolved

**Communication**: Every 30 minutes until resolved

---

### P2 - Medium (Minor Degradation)

**Definition**: Service degradation affecting some users or non-critical features.

**Examples**:
- 1/3 pods down (66% capacity)
- Error rate 5-20%
- Response time 5-10 seconds (p95)
- **Leak detection accuracy degraded (80-95%)**
- **Steam trap monitoring delayed (>5 min lag)**
- **Pressure drop calculation warnings**
- Non-critical integration failures (reporting APIs)
- Cache failures (Redis down, falling back to database)
- High CPU/memory usage (>80%)
- **Single steam meter offline**
- **Individual trap sensor failure**

**Response Time**: 1 hour

**Escalation**: Team Lead within 4 hours if not resolved

**Communication**: Hourly updates until resolved

---

### P3 - Low (Monitoring Alert)

**Definition**: Potential issues detected by monitoring, no current user impact.

**Examples**:
- Resource usage approaching limits (>70% CPU/memory)
- Slow database queries (>5 seconds)
- Cache miss rate increasing
- Certificate expiring within 7 days
- Disk space >70%
- **Steam meter communication latency increasing**
- **Condensate return rate slightly below target (70-75%)**

**Response Time**: 4 hours

**Escalation**: On-call engineer handles during business hours

**Communication**: Update in Slack, no status page update

---

### P4 - Informational

**Definition**: Information-only alerts, no action required.

**Examples**:
- Scheduled maintenance completed
- Deployment successful
- Auto-scaling events
- Performance improvements detected
- **Routine steam trap test cycle completed**
- **Leak detection algorithm recalibration success**

**Response Time**: None

**Escalation**: None

**Communication**: Logged in monitoring system only

---

## Response Procedures by Severity

### P0 - Critical Incident Response

#### Step 1: Acknowledge (0-5 minutes)

```bash
# 1. Acknowledge in PagerDuty
# Click "Acknowledge" in PagerDuty notification

# 2. Join incident war room
# Slack: /join #incident-<timestamp>

# 3. Announce in Slack
# Post in #gl-003-incidents
```

**Slack Message:**
```
@here P0 INCIDENT: GL-003 SteamSystemAnalyzer DOWN
Incident Commander: @your-name
Status: Investigating
Started: <timestamp>
Link: https://status.greenlang.io/incidents/<id>

SAFETY NOTE: If steam leak detection is affected, Plant Safety Officer notified: YES / NO
```

**CRITICAL**: If incident affects steam leak detection or emergency alerting:
```bash
# Immediately notify Plant Safety Officer
# Phone: +1-555-SAFETY
# Slack: @plant-safety-officer

# Message: "GL-003 steam leak detection offline. Manual monitoring required until system restored."
```

#### Step 2: Assess (5-10 minutes)

```bash
# Check service status
kubectl get pods -n greenlang | grep gl-003

# Check recent deployments
kubectl rollout history deployment/gl-003-steam-system -n greenlang

# Check error logs
kubectl logs -n greenlang deployment/gl-003-steam-system --tail=200 --all-containers=true

# Check metrics
curl -s "http://prometheus:9090/api/v1/query?query=up{job='gl-003'}" | jq

# Check dependencies
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/integrations/health

# Check steam system critical functions
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/analysis/leaks/status

kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/analysis/traps/status

kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/integrations/meters/health
```

**Assessment Checklist**:
- [ ] All pods down or all pods in CrashLoopBackOff?
- [ ] Recent deployment (last 2 hours)?
- [ ] Database accessible?
- [ ] Redis accessible?
- [ ] SCADA systems accessible?
- [ ] Steam meters responding?
- [ ] Leak detection operational?
- [ ] Steam trap monitoring active?
- [ ] Network policies blocking traffic?
- [ ] Resource exhaustion on nodes?

#### Step 3: Immediate Mitigation (10-20 minutes)

**If recent deployment (< 2 hours ago):**
```bash
# ROLLBACK IMMEDIATELY - See ROLLBACK_PROCEDURE.md
kubectl rollout undo deployment/gl-003-steam-system -n greenlang
kubectl rollout status deployment/gl-003-steam-system -n greenlang --timeout=5m

# Verify rollback
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/health
```

**If database down:**
```bash
# Check database status
kubectl get pods -n greenlang | grep postgres

# Check TimescaleDB extension
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';"

# Restart database (if in cluster)
kubectl rollout restart statefulset/postgresql -n greenlang

# Or restore from backup (AWS RDS)
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier gl-003-db-restored \
  --db-snapshot-identifier gl-003-snapshot-latest
```

**If all pods OOMKilled:**
```bash
# Immediately increase memory limits
kubectl set resources deployment gl-003-steam-system -n greenlang \
  --limits=memory=4Gi

# Verify pods start
kubectl get pods -n greenlang -w
```

**If SCADA/Steam Meter connection failed:**
```bash
# Test SCADA connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  telnet scada-server 4840

# Test steam meter network
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  ping -c 5 steam-meter-gateway.local

# Check network policies
kubectl get networkpolicy -n greenlang
kubectl describe networkpolicy gl-003-network-policy -n greenlang

# Temporarily disable network policy (emergency only!)
kubectl annotate networkpolicy gl-003-network-policy -n greenlang \
  'kubectl.kubernetes.io/last-applied-configuration-'
```

**If resource exhaustion:**
```bash
# Add more nodes (AWS EKS)
eksctl scale nodegroup --cluster=greenlang-cluster --nodes=8 --name=standard-workers

# Or reduce replica count temporarily
kubectl scale deployment gl-003-steam-system --replicas=1 -n greenlang
```

#### Step 4: Verify Recovery (20-30 minutes)

```bash
# Check pod health
kubectl get pods -n greenlang | grep gl-003
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/health

# Check error rate
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate

# Verify steam system functions
# 1. Leak detection
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/leaks/test \
    -H "Content-Type: application/json" \
    -d '{"test_mode": true}'

# 2. Steam trap monitoring
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/traps/test \
    -H "Content-Type: application/json" \
    -d '{"test_mode": true}'

# 3. Steam meter connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/status | jq

# Run smoke tests
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/test/smoke \
    -H "Content-Type: application/json" \
    -d '{"test_suite": "critical"}'

# Monitor for 15 minutes
kubectl logs -f -n greenlang deployment/gl-003-steam-system
```

**Recovery Checklist**:
- [ ] All pods running (3/3)
- [ ] Health endpoint returns 200 OK
- [ ] Error rate <1%
- [ ] Response time <2 seconds (p95)
- [ ] Leak detection operational and accurate
- [ ] Steam trap monitoring active
- [ ] Steam meter connectivity >99%
- [ ] SCADA integration healthy
- [ ] Integration tests passing
- [ ] No errors in logs

#### Step 5: Communicate Resolution

**Slack:**
```
@here P0 INCIDENT RESOLVED: GL-003 SteamSystemAnalyzer RESTORED
Incident Commander: @your-name
Status: Resolved
Duration: <XX minutes>
Root Cause: <brief description>
Mitigation: <action taken>
Steam System Status: All monitoring functions restored and operational
Post-Mortem: Scheduled for <date/time>
```

**Status Page:**
```
RESOLVED: GL-003 Steam System Analyzer service has been restored.
All systems are operating normally. Steam leak detection, trap monitoring,
and distribution efficiency analysis are fully functional.
We will conduct a post-incident review to prevent future occurrences.
```

**Plant Safety Officer Notification (if applicable):**
```
RESOLVED: GL-003 steam leak detection and monitoring systems restored.
Duration of manual monitoring required: <XX minutes>
All automated safety alerts now operational.
No safety incidents occurred during outage.
Post-incident report will be provided within 24 hours.
```

---

### P1 - High Severity Incident Response

#### Step 1: Acknowledge (0-15 minutes)

```bash
# Acknowledge in PagerDuty
# Post in #gl-003-incidents
```

**Slack Message:**
```
P1 INCIDENT: GL-003 Major Degradation
Owner: @your-name
Status: Investigating
Impact: <describe impact - e.g., "Leak detection accuracy degraded to 75%">
Started: <timestamp>
```

#### Step 2: Investigate (15-30 minutes)

```bash
# Check pod health
kubectl get pods -n greenlang | grep gl-003
kubectl describe pod -n greenlang <pod-name>

# Check logs
kubectl logs -n greenlang deployment/gl-003-steam-system --tail=500

# Check metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(error|latency|cpu|memory|leak|trap)'

# Check recent changes
kubectl rollout history deployment/gl-003-steam-system -n greenlang

# Check steam system specific metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(leak_detection_accuracy|trap_classification_accuracy|meter_connectivity|distribution_efficiency)'
```

#### Step 3: Mitigate (30-60 minutes)

**If pods down:**
```bash
# Restart affected pods
kubectl delete pod -n greenlang <pod-name>

# Or scale up
kubectl scale deployment gl-003-steam-system --replicas=5 -n greenlang
```

**If high error rate:**
```bash
# Check integration health
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/health | jq

# Disable failing integration temporarily
kubectl edit configmap gl-003-config -n greenlang
# Set: SCADA_INTEGRATION_ENABLED: "false"

# Restart
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

**If leak detection accuracy degraded:**
```bash
# Check leak detection calibration
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/calibration | jq

# Recalibrate leak detection algorithm
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/leaks/recalibrate \
    -H "Content-Type: application/json" \
    -d '{"use_golden_dataset": true}'

# Verify accuracy improved
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep leak_detection_accuracy
```

**If steam trap classification failing:**
```bash
# Check trap sensor connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/traps/connectivity | jq

# Reset trap classification model
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/traps/reset-model

# Restart pods to reload model
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

**If performance degradation:**
```bash
# Enable auto-scaling
kubectl autoscale deployment gl-003-steam-system -n greenlang \
  --min=3 --max=10 --cpu-percent=70

# Clear cache if corrupted
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  redis-cli -h redis-host FLUSHDB

# Restart pods
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

#### Step 4: Monitor (60+ minutes)

```bash
# Monitor recovery
kubectl get pods -n greenlang -w
kubectl top pods -n greenlang | grep gl-003

# Check error rate every 5 minutes
watch -n 300 'kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate'

# Monitor steam system metrics
watch -n 60 'kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E "(leak_detection|trap_classification|meter_connectivity)"'
```

#### Step 5: Communicate Resolution

**Slack:**
```
P1 INCIDENT RESOLVED: GL-003 Major Degradation
Owner: @your-name
Duration: <XX minutes>
Root Cause: <brief description>
Action Taken: <mitigation steps>
Steam System Status: <e.g., "Leak detection accuracy restored to 96%">
```

---

### P2 - Medium Severity Incident Response

#### Quick Response (1 hour)

```bash
# Assess impact
kubectl get pods -n greenlang | grep gl-003
kubectl logs -n greenlang deployment/gl-003-steam-system --tail=200

# Apply quick fix
# Example: Restart single pod
kubectl delete pod -n greenlang <pod-name>

# Monitor
kubectl get pods -n greenlang -w
```

**Slack Update:**
```
P2 INCIDENT: GL-003 Minor Degradation
Owner: @your-name
Impact: <describe>
Action: <what you're doing>
ETA: <estimated resolution time>
```

---

## Steam System Specific Incidents

### Critical Steam Leak Detected but Not Reported

**Symptoms**: Plant reports steam leak, but GL-003 did not generate alert

**Immediate Actions**:
```bash
# 1. Verify leak detection status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/status

# 2. Check recent leak detection results
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/recent | jq

# 3. Check sensor data from leak location
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/location/<location_id> | jq

# 4. Force manual leak scan
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/leaks/scan \
    -H "Content-Type: application/json" \
    -d '{"location_id": "<location_id>", "priority": "immediate"}'

# 5. Review leak detection thresholds
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/config/leak-detection | jq
```

**Root Cause Analysis**:
- Sensor malfunction or offline
- Leak detection threshold too high
- Algorithm false negative
- Network connectivity issue
- Data processing delay

**Follow-up**:
- Notify Plant Safety Officer
- Review and adjust leak detection sensitivity
- Schedule sensor calibration
- Create post-incident report for compliance

---

### Steam Trap Mass Failure Event

**Symptoms**: >20% of steam traps showing as failed simultaneously

**Immediate Actions**:
```bash
# 1. Verify trap sensor connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/traps/connectivity | jq

# 2. Check for sensor network failure
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/traps/network-status | jq

# 3. Review trap classification algorithm logs
kubectl logs -n greenlang deployment/gl-003-steam-system | grep "trap_classification"

# 4. Check if false positive due to environmental factors
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/traps/environmental-factors | jq

# 5. Manual verification of sample traps
# (Coordinate with facility maintenance team for physical inspection)
```

**Possible Causes**:
- Sensor network failure
- Environmental conditions (extreme cold, ambient temperature change)
- Algorithm false positive spike
- Actual mass failure event (rare but possible)
- Power outage affecting sensors

---

### Distribution Efficiency Sudden Drop

**Symptoms**: Distribution efficiency drops from >90% to <80% within minutes

**Immediate Actions**:
```bash
# 1. Check for large leaks
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/active | jq

# 2. Check steam generation vs consumption balance
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/balance | jq

# 3. Review pressure drop across distribution network
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/pressure-drop | jq

# 4. Check meter accuracy
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/accuracy | jq
```

**Likely Causes**:
- Major steam leak
- Meter calibration drift
- Calculation error in efficiency algorithm
- Sudden load increase not captured
- Condensate return failure

---

### Pressure Anomaly Alerts

**Symptoms**: Pressure measurements outside expected range

**Immediate Actions**:
```bash
# 1. Verify pressure sensor readings
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/pressure | jq

# 2. Check historical pressure trends
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/pressure-trends | jq

# 3. Compare with boiler system (GL-002)
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/boiler/steam-pressure | jq

# 4. Check for rapid pressure changes (safety concern)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/pressure-rate-of-change | jq
```

**Safety Note**: Rapid pressure changes can indicate safety issues. Notify Plant Safety Officer if rate of change >10 bar/minute.

---

### SCADA Connection Loss

**Symptoms**: Loss of connectivity to SCADA system for >5 minutes

**Immediate Actions**:
```bash
# 1. Test SCADA connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  telnet scada-server 4840

# 2. Check OPC UA connection status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/scada/status | jq

# 3. Review SCADA connection logs
kubectl logs -n greenlang deployment/gl-003-steam-system | grep "SCADA"

# 4. Verify SCADA credentials
kubectl get secret gl-003-secrets -n greenlang -o jsonpath='{.data.SCADA_USERNAME}' | base64 -d

# 5. Check network policies
kubectl describe networkpolicy gl-003-network-policy -n greenlang

# 6. Attempt reconnection
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/integrations/scada/reconnect
```

**Fallback**: Switch to direct steam meter polling if SCADA unavailable

---

### Database Timeseries Query Failures

**Symptoms**: TimescaleDB queries timing out or failing

**Immediate Actions**:
```bash
# 1. Check database connection
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT version();"

# 2. Check TimescaleDB extension status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.hypertables;"

# 3. Check for bloated chunks
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.chunks ORDER BY total_bytes DESC LIMIT 10;"

# 4. Check compression status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.compression_settings;"

# 5. Force chunk compression (if needed)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT compress_chunk(i) FROM show_chunks('steam_measurements', older_than => INTERVAL '7 days') i;"

# 6. Vacuum and analyze
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "VACUUM ANALYZE;"
```

---

## Escalation Paths

### P0 Critical Escalation

```
Incident Detected
    |
    v
PagerDuty Alert → On-Call Engineer (Primary)
    |
    v (No ACK in 5 min)
PagerDuty Alert → On-Call Engineer (Secondary)
    |
    v (No ACK in 5 min)
PagerDuty Alert → Engineering Manager
    |
    v (Automatically)
Slack Alert → #incidents channel
    |
    v (Automatically for steam safety issues)
Slack/Phone Alert → Plant Safety Officer
    |
    v (Automatically)
Slack Alert → @director-engineering + @cto
    |
    v (If not resolved in 1 hour)
Email → CEO + Board (for critical safety issues)
```

### P1 High Escalation

```
Incident Detected
    |
    v
PagerDuty Alert → On-Call Engineer
    |
    v (No resolution in 1 hour)
Slack Alert → Engineering Manager
    |
    v (For steam system issues)
Slack Alert → Steam System SME
    |
    v (No resolution in 4 hours)
Email → Director of Engineering
```

### P2 Medium Escalation

```
Incident Detected
    |
    v
PagerDuty Alert → On-Call Engineer
    |
    v (No resolution in 4 hours)
Slack Alert → Team Lead
    |
    v (No resolution in 8 hours)
Email → Engineering Manager
```

---

## Communication Templates

### P0 Initial Notification

**Subject:** [P0 CRITICAL] GL-003 SteamSystemAnalyzer Production Outage

**Body:**
```
INCIDENT: P0 Critical - GL-003 Production Outage
STATUS: Investigating
STARTED: 2025-11-17 14:32 UTC
IMPACT: All users unable to access GL-003 services
STEAM SYSTEM IMPACT: Leak detection offline, manual monitoring required
INCIDENT COMMANDER: @engineer-name
WAR ROOM: #incident-2025-11-17-1432

SAFETY NOTIFICATIONS:
- Plant Safety Officer notified: YES
- Manual leak monitoring activated: YES
- Emergency procedures initiated: YES

CURRENT ACTIONS:
- Investigating root cause
- Checking recent deployments
- Verifying database connectivity
- Testing SCADA connectivity
- Preparing rollback if needed

NEXT UPDATE: 15 minutes (14:47 UTC)

STATUS PAGE: https://status.greenlang.io/incidents/2025-11-17-1432
GRAFANA: https://grafana.greenlang.io/d/gl-003/incident-2025-11-17-1432
```

### P0 Update (Every 15 minutes)

**Subject:** [P0 CRITICAL] GL-003 Update #2 - Root Cause Identified

**Body:**
```
INCIDENT: P0 Critical - GL-003 Production Outage
STATUS: Mitigating
DURATION: 23 minutes
IMPACT: All users unable to access GL-003 services

ROOT CAUSE IDENTIFIED:
Database migration in v1.3.5 deployment failed, causing TimescaleDB
hypertable corruption. All pods unable to query steam measurement data.

MITIGATION IN PROGRESS:
1. Rolling back deployment to v1.3.4 (ETA: 5 minutes)
2. Restoring database from last known good snapshot
3. Verifying leak detection functionality post-rollback
4. Running steam system validation tests

SAFETY STATUS:
- Manual leak monitoring continues
- No safety incidents reported
- Steam trap monitoring will resume after rollback

NEXT UPDATE: 15 minutes (14:57 UTC)

STATUS PAGE: https://status.greenlang.io/incidents/2025-11-17-1432
```

### P0 Resolution

**Subject:** [P0 RESOLVED] GL-003 SteamSystemAnalyzer Restored

**Body:**
```
INCIDENT: P0 Critical - GL-003 Production Outage
STATUS: RESOLVED
STARTED: 2025-11-17 14:32 UTC
RESOLVED: 2025-11-17 14:58 UTC
DURATION: 26 minutes
INCIDENT COMMANDER: @engineer-name

ROOT CAUSE:
TimescaleDB hypertable migration in v1.3.5 deployment failed due to
incorrect chunk interval configuration, causing all time-series queries
to fail and preventing steam measurement data access.

MITIGATION TAKEN:
1. Rolled back deployment to v1.3.4
2. Restored database from snapshot (14:25 UTC)
3. Verified all steam system functions operational
4. Ran full validation test suite
5. Monitored for 15 minutes - all systems normal

CURRENT STATUS:
✅ All pods running (3/3)
✅ Health checks passing
✅ Error rate <0.1%
✅ Response time <1.5s (p95)
✅ Leak detection operational (accuracy: 97.2%)
✅ Steam trap monitoring active (418/420 traps online)
✅ Steam meter connectivity 99.8% (496/497 meters online)
✅ Distribution efficiency calculation accurate
✅ All integrations healthy

SAFETY STATUS:
✅ Leak detection fully restored
✅ Manual monitoring concluded
✅ No safety incidents during outage
✅ All emergency alerts operational

PREVENTION:
- Added TimescaleDB chunk interval validation to CI pipeline
- Updated deployment procedure to require database migration dry-run
- Added automated rollback trigger for database query failures
- Scheduled post-mortem for 2025-11-18 10:00 UTC

POST-MORTEM: https://docs.greenlang.io/postmortems/2025-11-17-gl-003-outage
STATUS PAGE: https://status.greenlang.io/incidents/2025-11-17-1432
```

### P1 Notification

**Subject:** [P1 HIGH] GL-003 Steam Leak Detection Degraded

**Body:**
```
INCIDENT: P1 High - GL-003 Leak Detection Accuracy Degraded
STATUS: Investigating
STARTED: 2025-11-17 15:15 UTC
IMPACT: Leak detection accuracy degraded to 78% (normal: >95%)
OWNER: @engineer-name

SYMPTOMS:
- Leak detection false positive rate: 18% (normal: <2%)
- Steam meter data quality score: 72% (normal: >90%)
- Acoustic sensor connectivity: 85% (normal: >98%)

ACTIONS:
- Investigating sensor network issues
- Checking acoustic sensor calibration
- Reviewing leak detection algorithm parameters
- Preparing to recalibrate algorithm if needed

SAFETY MITIGATION:
- Enhanced manual monitoring initiated
- Leak detection threshold temporarily lowered
- Additional sensor checks scheduled

NEXT UPDATE: 30 minutes (15:45 UTC)
```

### Status Page Update Templates

**Investigating:**
```
Title: GL-003 Service Degradation - Steam Leak Detection
Status: Investigating

We are investigating reports of degraded accuracy in steam leak
detection. The system is operational but may show increased false
positives. Manual verification of leak alerts is recommended during
this time. Our team is actively working to identify and resolve the issue.

Updated: 2025-11-17 14:32 UTC
```

**Identified:**
```
Title: GL-003 Service Degradation - Steam Leak Detection
Status: Identified

We have identified the root cause as acoustic sensor network
connectivity issues affecting leak detection accuracy. We are
recalibrating the leak detection algorithm and restoring sensor
connectivity. Expected resolution within 30 minutes.

Updated: 2025-11-17 14:45 UTC
```

**Monitoring:**
```
Title: GL-003 Service Degradation - Steam Leak Detection
Status: Monitoring

Sensor connectivity has been restored and leak detection algorithm
has been recalibrated. We are monitoring the system to ensure accuracy
has returned to normal levels (>95%) before marking this incident
as resolved.

Updated: 2025-11-17 14:55 UTC
```

**Resolved:**
```
Title: GL-003 Service Degradation - Steam Leak Detection
Status: Resolved

This incident has been resolved. Leak detection accuracy has been
restored to 97.2% (target: >95%). All steam system monitoring
functions are operating normally. We apologize for any inconvenience.

Updated: 2025-11-17 14:58 UTC
```

---

## Emergency Rollback Procedures

### Quick Rollback (5 minutes)

```bash
# 1. Rollback to previous version
kubectl rollout undo deployment/gl-003-steam-system -n greenlang

# 2. Wait for rollout to complete
kubectl rollout status deployment/gl-003-steam-system -n greenlang --timeout=5m

# 3. Verify health
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/health

# 4. Verify steam system functions
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/status

kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/traps/status

# 5. Check logs for errors
kubectl logs -n greenlang deployment/gl-003-steam-system --tail=50

# 6. Monitor error rate
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate
```

### Rollback to Specific Version

```bash
# 1. List revision history
kubectl rollout history deployment/gl-003-steam-system -n greenlang

# 2. Rollback to specific revision
kubectl rollout undo deployment/gl-003-steam-system -n greenlang --to-revision=5

# 3. Verify
kubectl rollout status deployment/gl-003-steam-system -n greenlang
```

For detailed rollback procedures, see: `ROLLBACK_PROCEDURE.md`

---

## Post-Incident Review

### Within 24 Hours of Resolution

Schedule a post-incident review (post-mortem) meeting with:
- Incident Commander
- On-call engineer(s)
- Engineering Manager
- Affected team members
- Steam System SME (for steam-specific incidents)
- Optional: Product Manager, Customer Success (if customer-facing impact)
- **Required for safety incidents**: Plant Safety Officer

### Post-Mortem Template

**1. Incident Summary**
- Incident ID: INC-2025-11-17-001
- Severity: P0 Critical
- Duration: 26 minutes
- Impact: All users unable to access GL-003
- Steam System Impact: Leak detection offline, manual monitoring required

**2. Timeline**
- 14:32 UTC: Deployment v1.3.5 completed
- 14:33 UTC: Monitoring alerts triggered (all pods down)
- 14:34 UTC: Plant Safety Officer notified, manual monitoring activated
- 14:35 UTC: PagerDuty alert sent to on-call
- 14:37 UTC: Incident acknowledged, investigation started
- 14:45 UTC: Root cause identified (TimescaleDB migration failure)
- 14:47 UTC: Rollback initiated
- 14:52 UTC: Rollback completed, pods healthy
- 14:55 UTC: Leak detection verified operational
- 14:58 UTC: Incident resolved after monitoring period

**3. Root Cause Analysis**
- TimescaleDB hypertable chunk interval misconfigured in migration script
- Migration script set chunk_time_interval to 1 hour instead of 1 day
- This caused excessive chunk creation and query planner failures
- All time-series queries (steam measurements) failed
- Pods unable to retrieve steam meter data for analysis
- Leak detection and trap monitoring became non-operational

**4. What Went Well**
- Monitoring detected issue within 1 minute of deployment
- On-call responded quickly (2 minutes to acknowledge)
- Plant Safety Officer immediately notified (safety protocol followed)
- Manual monitoring activated promptly
- Rollback procedure worked as expected (5 minutes)
- Communication was clear and frequent
- No safety incidents occurred during outage

**5. What Went Wrong**
- TimescaleDB chunk interval not validated in CI pipeline
- No dry-run of database migration before production deployment
- No automated rollback trigger for database query failures
- Insufficient testing of time-series query performance under migration

**6. Action Items**
- [ ] Add TimescaleDB chunk interval validation to CI pipeline (Owner: @dev1, Due: 2025-11-20)
- [ ] Implement database migration dry-run requirement (Owner: @dba1, Due: 2025-11-22)
- [ ] Add automated rollback trigger for database query failures (Owner: @sre1, Due: 2025-11-25)
- [ ] Add time-series query performance tests to test suite (Owner: @dev2, Due: 2025-11-23)
- [ ] Update runbook with TimescaleDB-specific scenarios (Owner: @tech-writer, Due: 2025-11-18)
- [ ] Schedule TimescaleDB best practices training (Owner: @eng-manager, Due: 2025-11-30)

---

## Incident Response Checklist

### P0 Critical Checklist

- [ ] Acknowledge incident in PagerDuty (within 5 min)
- [ ] Notify Plant Safety Officer if steam monitoring affected
- [ ] Activate manual monitoring procedures if needed
- [ ] Post initial message in #gl-003-incidents
- [ ] Update status page (Investigating)
- [ ] Join war room (create #incident-<timestamp>)
- [ ] Assess impact and root cause (within 10 min)
- [ ] Implement mitigation (rollback or fix)
- [ ] Verify recovery (health checks, smoke tests)
- [ ] Verify steam system functions (leak detection, traps, meters)
- [ ] Update status page (Resolved)
- [ ] Post resolution message in Slack
- [ ] Notify Plant Safety Officer of resolution
- [ ] Deactivate manual monitoring procedures
- [ ] Schedule post-mortem within 24 hours
- [ ] Document lessons learned
- [ ] Update runbooks with new scenarios

### P1 High Checklist

- [ ] Acknowledge incident in PagerDuty (within 15 min)
- [ ] Post message in #gl-003-incidents
- [ ] Assess steam system impact
- [ ] Investigate root cause
- [ ] Implement mitigation
- [ ] Monitor recovery
- [ ] Post resolution message
- [ ] Create ticket for follow-up actions
- [ ] Update runbook if needed

### P2 Medium Checklist

- [ ] Acknowledge incident (within 1 hour)
- [ ] Post message in #gl-003-incidents
- [ ] Investigate and fix
- [ ] Monitor
- [ ] Post update when resolved
- [ ] Create ticket for root cause fix

---

## Contact Information

### On-Call Rotation

**Primary On-Call**: PagerDuty schedule "GL-003 Primary"
- Phone: See PagerDuty
- Slack: @gl-003-oncall-primary

**Secondary On-Call**: PagerDuty schedule "GL-003 Secondary"
- Phone: See PagerDuty
- Slack: @gl-003-oncall-secondary

**Steam System SME**: On-call schedule "Steam-SME"
- Phone: See PagerDuty
- Slack: @steam-sme
- Availability: Business hours + on-call rotation

### Escalation Contacts

**Engineering Manager**: Jane Smith
- Phone: +1-555-0101
- Slack: @jane-smith
- Email: jane.smith@greenlang.io

**Director of Engineering**: John Doe
- Phone: +1-555-0102
- Slack: @john-doe
- Email: john.doe@greenlang.io

**CTO**: Sarah Johnson
- Phone: +1-555-0103
- Slack: @sarah-johnson
- Email: sarah.johnson@greenlang.io

**Plant Safety Officer**: Mike Davis (for steam safety incidents)
- Phone: +1-555-SAFETY (24/7)
- Slack: @plant-safety-officer
- Email: safety@greenlang.io

### External Contacts

**AWS Support**:
- Phone: 1-800-123-4567
- Support Portal: https://console.aws.amazon.com/support
- Severity: Business-critical production system down

**Database Vendor (TimescaleDB)**:
- Support: support@timescale.com
- Phone: 1-855-TIMESCALE
- SLA: 1-hour response for P0

**SCADA Vendor**:
- Phone: +1-555-0200
- Email: support@scada-vendor.com
- SLA: 2-hour response for P0

**Steam Meter Vendor**:
- Phone: +1-555-0201
- Email: support@meter-vendor.com
- SLA: 4-hour response for P1

**Acoustic Leak Detection Vendor**:
- Phone: +1-555-0202
- Email: support@acoustic-vendor.com
- SLA: 4-hour response for P1

---

## Additional Resources

- **Troubleshooting Guide**: `TROUBLESHOOTING.md`
- **Rollback Procedure**: `ROLLBACK_PROCEDURE.md`
- **Scaling Guide**: `SCALING_GUIDE.md`
- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-003\deployment\README.md`
- **Status Page**: https://status.greenlang.io
- **PagerDuty**: https://greenlang.pagerduty.com
- **Grafana Dashboards**: https://grafana.greenlang.io/d/gl-003
- **Post-Mortems Archive**: https://docs.greenlang.io/postmortems
- **Steam System Safety Procedures**: https://docs.greenlang.io/safety/steam-systems
