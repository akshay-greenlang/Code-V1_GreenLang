# GL-001 ProcessHeatOrchestrator - Incident Response Guide

Emergency procedures and incident response playbook for GL-001 ProcessHeatOrchestrator production operations managing multi-plant industrial process heat.

**CRITICAL SAFETY NOTE**: GL-001 manages life-safety critical heat operations. ALL incidents must consider safety implications. For P0 heat loss incidents, immediately escalate to Plant Safety Officer.

## Table of Contents

1. [Incident Severity Definitions](#incident-severity-definitions)
2. [Response Procedures by Severity](#response-procedures-by-severity)
3. [Process Heat Specific Incidents](#process-heat-specific-incidents)
4. [Safety Protocols](#safety-protocols)
5. [Escalation Paths](#escalation-paths)
6. [Communication Templates](#communication-templates)
7. [Emergency Rollback Procedures](#emergency-rollback-procedures)
8. [Post-Incident Review](#post-incident-review)

---

## Incident Severity Definitions

### P0 - Critical (Life Safety / Total Heat Loss)

**Definition**: Complete heat loss, multi-plant failure, or safety hazard requiring immediate action.

**Examples**:
- **CRITICAL HEAT LOSS**: Total facility heat loss affecting safety systems or production
- **MULTI-PLANT CASCADE**: 3+ plants experiencing heat supply failures simultaneously
- **MASTER ORCHESTRATOR DOWN**: All GL-001 pods down, no coordination possible
- **SAFETY HAZARD**: Boiler pressure alarm, combustion instability, chemical process risk
- **SUB-AGENT CASCADE**: 10+ sub-agents (GL-002 through GL-100) failing simultaneously
- **SCADA TOTAL OUTAGE**: Loss of all real-time plant data across multiple facilities
- **DATABASE CATASTROPHIC FAILURE**: Complete data loss or corruption affecting all plants
- **SECURITY BREACH**: Unauthorized access to critical control systems

**Response Time**: **IMMEDIATE** (acknowledge within 5 minutes, action within 10 minutes)

**Escalation**: **AUTOMATIC** to:
- Plant Safety Officer (immediate)
- Plant Operations Manager (immediate)
- Engineering Manager (within 15 minutes)
- VP Operations (within 30 minutes if unresolved)

**Communication**: Every 15 minutes until resolved, status page update required

**Safety Priority**: Heat restoration to life-safety systems within 10 minutes

---

### P1 - High (Single Plant Heat Loss / Major Degradation)

**Definition**: Single plant heat loss or significant service degradation affecting critical operations.

**Examples**:
- **SINGLE PLANT HEAT LOSS**: One plant experiencing complete heat supply failure
- **MAJOR SUB-AGENT CASCADE**: 5-10 sub-agents failing, impacting multiple plants
- **SCADA PARTIAL OUTAGE**: Single plant or critical SCADA feed loss
- **HEAT OPTIMIZATION FAILURE**: Heat distribution LP solver failures across facilities
- **MASTER ORCHESTRATOR DEGRADED**: 2/3 master pods down (33% capacity)
- **ERP INTEGRATION FAILURE**: Production scheduling data unavailable
- **DATABASE PERFORMANCE**: Severe degradation (>10s query latency) affecting operations
- **MEMORY LEAK**: Master orchestrator OOMKilled repeatedly
- **CRITICAL EMISSIONS VIOLATION**: Exceeding regulatory limits requiring immediate reporting

**Response Time**: 15 minutes (acknowledge), 30 minutes (mitigation started)

**Escalation**: Plant Operations Manager within 30 minutes if heat not restored

**Communication**: Every 30 minutes until resolved, status page update for external impact

**Safety Priority**: Heat restoration to affected plant within 30 minutes

---

### P2 - Medium (Performance Degradation / Minor Issues)

**Definition**: Service degradation affecting efficiency but not critical heat supply.

**Examples**:
- **HEAT DISTRIBUTION INEFFICIENCY**: >20% from optimal distribution, no heat loss
- **MINOR SUB-AGENT DEGRADATION**: 3-5 sub-agents experiencing errors
- **ORCHESTRATION LATENCY**: Master coordination >5 seconds (p95)
- **SINGLE POD FAILURE**: 1/3 master pods down (66% capacity)
- **ERROR RATE ELEVATED**: 5-20% error rate on non-critical operations
- **CACHE FAILURES**: Redis down, falling back to database (performance impact)
- **HIGH RESOURCE USAGE**: CPU/Memory >80% on master orchestrator
- **NON-CRITICAL INTEGRATION**: Reporting APIs or non-essential systems down
- **EMISSIONS NEAR LIMITS**: Approaching but not exceeding regulatory limits

**Response Time**: 1 hour (acknowledge and investigate)

**Escalation**: Engineering Manager within 4 hours if not resolved

**Communication**: Hourly updates in Slack, no status page update unless customer-facing

**Safety Priority**: Monitor for escalation, maintain heat supply

---

### P3 - Low (Monitoring Alert / Potential Issues)

**Definition**: Potential issues detected by monitoring, no current operational impact.

**Examples**:
- Resource usage approaching limits (>70% CPU/memory)
- Slow heat optimization (>3 seconds, but <5 second threshold)
- Cache miss rate increasing
- Certificate expiring within 7 days
- Disk space >70%
- Minor database query slowness (>2 seconds)
- Non-critical sub-agent warning alerts
- Heat efficiency below target but within acceptable range

**Response Time**: 4 hours (business hours)

**Escalation**: On-call engineer handles, no automatic escalation

**Communication**: Update in Slack #gl-001-ops, no broadcast

**Safety Priority**: None, monitoring only

---

### P4 - Informational

**Definition**: Information-only alerts, no action required.

**Examples**:
- Scheduled maintenance completed successfully
- Deployment successful (canary or full)
- Auto-scaling events (HPA triggered)
- Performance improvements detected
- Heat efficiency improvements
- Successful sub-agent coordination
- Cache hit rate improvements

**Response Time**: None

**Escalation**: None

**Communication**: Logged in monitoring system only

---

## Response Procedures by Severity

### P0 - Critical: Total Heat Loss Response

**SCENARIO: Critical heat loss affecting entire facility or multiple plants**

#### Step 1: Immediate Acknowledgement (0-5 minutes)

```bash
# 1. ACKNOWLEDGE IN PAGERDUTY (Within 5 minutes)
# Click "Acknowledge" immediately

# 2. JOIN INCIDENT WAR ROOM
# Slack: /join #incident-heat-loss-<timestamp>

# 3. ANNOUNCE IN SLACK
# Post in #gl-001-incidents AND #plant-operations
```

**Slack Emergency Message:**
```
@channel @plant-safety-officer @plant-operations-manager

🚨🚨🚨 P0 CRITICAL: TOTAL HEAT LOSS - IMMEDIATE ACTION REQUIRED 🚨🚨🚨

Incident Commander: @your-name
Status: EMERGENCY RESPONSE IN PROGRESS
Affected Plants: [Plant-001, Plant-002, ...] or ALL
Heat Loss Confirmed: YES
Safety Systems Affected: [YES/NO]
Started: <timestamp>
War Room: #incident-heat-loss-<timestamp>

IMMEDIATE ACTIONS:
1. Verifying backup heat systems
2. Checking master orchestrator status
3. Assessing sub-agent cascade
4. Plant Safety Officer notified

Next Update: 10 minutes
```

#### Step 2: Emergency Assessment (5-10 minutes)

**PRIORITY 1: Verify Heat Supply Status**
```bash
# Check critical heat status across all plants
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/critical-heat-status | \
  jq '.plants[] | select(.heat_loss == true)'

# Expected output:
# {
#   "plant_id": "Plant-001",
#   "heat_loss": true,
#   "affected_systems": ["boiler-1", "boiler-2"],
#   "backup_heat_available": false,
#   "time_to_critical_temperature": "15 minutes"
# }
```

**PRIORITY 2: Check Master Orchestrator Status**
```bash
# Master pods status
kubectl get pods -n greenlang | grep gl-001

# If all pods down:
# NAME                                            READY   STATUS             RESTARTS   AGE
# gl-001-process-heat-orchestrator-5d8c7f-abc12   0/1     CrashLoopBackOff   5          10m
# gl-001-process-heat-orchestrator-5d8c7f-def34   0/1     CrashLoopBackOff   5          10m
# gl-001-process-heat-orchestrator-5d8c7f-ghi56   0/1     Error              3          10m

# Check recent deployments
kubectl rollout history deployment/gl-001-process-heat-orchestrator -n greenlang

# Check error logs
kubectl logs -n greenlang deployment/gl-001-process-heat-orchestrator \
  --tail=200 --all-containers=true | grep -E '(ERROR|FATAL|heat_loss|cascade_failure)'
```

**PRIORITY 3: Check Sub-Agent Cascade Status**
```bash
# Check sub-agent health
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/health | \
  jq '.agents[] | select(.status != "healthy") | {agent_id, status, plant_id}'

# Count failed agents
kubectl get pods -n greenlang | grep -E '(gl-002|gl-003|gl-004|gl-005)' | grep -v Running | wc -l

# If >10 agents down, this is a cascade failure
```

**Emergency Assessment Checklist**:
- [ ] Heat loss confirmed at which plants? (List all)
- [ ] Backup heat systems available? (YES/NO)
- [ ] Time to critical temperature? (<15 min = EXTREME URGENCY)
- [ ] Master orchestrator status? (All down / Degraded / Running)
- [ ] Sub-agent cascade? (Count of failed agents: ___)
- [ ] Recent deployment in last 2 hours? (YES/NO)
- [ ] Database accessible? (YES/NO)
- [ ] SCADA systems accessible? (YES/NO)
- [ ] Safety alarms active? (List any critical alarms)

#### Step 3: Emergency Heat Restoration (10-20 minutes)

**OPTION A: If Recent Deployment (<2 hours)**

```bash
# IMMEDIATE EMERGENCY ROLLBACK
# See ROLLBACK_PROCEDURE.md for detailed steps

# 1. Rollback master orchestrator
kubectl rollout undo deployment/gl-001-process-heat-orchestrator -n greenlang

# 2. Monitor rollback (should complete in <5 minutes)
kubectl rollout status deployment/gl-001-process-heat-orchestrator -n greenlang --timeout=5m

# 3. Verify heat coordination restored
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -f http://localhost:8000/api/v1/plants/critical-heat-status

# 4. If sub-agents also need rollback (coordinated rollback)
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang
kubectl rollout undo deployment/gl-003-combustion-optimizer -n greenlang
# Rollback other affected sub-agents as needed

# 5. Verify heat supply restoring
watch -n 5 'kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status'
```

**OPTION B: If Master Orchestrator Down (No Recent Deployment)**

```bash
# 1. Restart master orchestrator
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang

# 2. Scale up aggressively for immediate capacity
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=10 -n greenlang

# 3. Monitor pod startup
kubectl get pods -n greenlang -w | grep gl-001

# 4. If pods still failing, check resource constraints
kubectl describe nodes | grep -E '(cpu|memory)' | grep -E '[0-9]+%'

# 5. If resource exhaustion, add nodes immediately
eksctl scale nodegroup --cluster=greenlang-cluster --nodes=12 --name=standard-workers

# 6. Check database connectivity
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -f http://localhost:8000/api/v1/health/database
```

**OPTION C: If Database Failure**

```bash
# 1. Check database status
kubectl get pods -n greenlang | grep postgres
kubectl logs -n greenlang statefulset/postgresql --tail=100

# 2. Restore from latest backup (AWS RDS)
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier gl-001-db-emergency \
  --db-snapshot-identifier gl-001-snapshot-$(date +%Y%m%d) \
  --db-instance-class db.r5.2xlarge \
  --publicly-accessible false

# 3. Update connection string (use ConfigMap)
kubectl edit configmap gl-001-config -n greenlang
# Update DB_HOST to new endpoint

# 4. Restart master orchestrator to connect to new DB
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

**OPTION D: If SCADA Total Outage**

```bash
# 1. Switch to MANUAL HEAT CONTROL mode
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/emergency/manual-mode \
    -H "Content-Type: application/json" \
    -d '{"mode": "manual", "reason": "SCADA_outage", "authorized_by": "your-name"}'

# 2. Use last known good heat distribution
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/emergency/use-last-good-config

# 3. Notify plant operators to use local controls
# Send to plant-operators@greenlang.io and post in #plant-operations

# 4. Contact SCADA vendor for emergency support
# See README.md for vendor contact (Siemens/Honeywell 24/7 hotline)
```

#### Step 4: Verify Heat Restoration (20-30 minutes)

```bash
# 1. Check critical heat supply restored at ALL plants
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status | \
  jq '.plants[] | {plant_id, heat_status, efficiency, critical_systems_ok}'

# Expected output (ALL plants should show heat_status: "normal"):
# {
#   "plant_id": "Plant-001",
#   "heat_status": "normal",
#   "efficiency": 82.5,
#   "critical_systems_ok": true
# }

# 2. Verify master orchestrator healthy
kubectl get pods -n greenlang | grep gl-001
# All pods should be "Running" with "3/3 READY"

kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -f http://localhost:8000/api/v1/health

# 3. Verify sub-agent coordination
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq '.agents[] | select(.status != "connected")'

# Should return empty (all agents connected)

# 4. Check heat distribution optimization
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/status | \
  jq '.optimization_status'

# Should return "optimal" or "converged"

# 5. Monitor for 15 minutes to ensure stability
watch -n 30 'kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | \
  grep -E "(heat_loss_incidents|orchestration_errors|agent_failures)"'
```

**Heat Restoration Checklist**:
- [ ] ALL plants showing heat_status: "normal"
- [ ] Critical safety systems receiving heat
- [ ] Master orchestrator pods: 3/3 Running
- [ ] Sub-agents connected: ___/99 (target: >95)
- [ ] Heat distribution optimization: "optimal"
- [ ] No errors in logs for 15 minutes
- [ ] Safety alarms cleared
- [ ] Plant operators confirm heat supply normal

#### Step 5: Communicate Resolution

**Slack Resolution Message:**
```
@channel @plant-safety-officer @plant-operations-manager

✅ P0 INCIDENT RESOLVED: HEAT SUPPLY RESTORED TO ALL PLANTS ✅

Incident Commander: @your-name
Status: RESOLVED
Duration: <XX minutes>
Affected Plants: [Plant-001, Plant-002, ...] or ALL

ROOT CAUSE:
<Brief description, e.g., "Master orchestrator deployment regression causing cascade failure">

MITIGATION ACTIONS:
1. Emergency rollback of GL-001 deployment
2. Coordinated rollback of affected sub-agents (GL-002, GL-003)
3. Heat distribution optimization restored
4. All 99 sub-agents reconnected and coordinating

VERIFICATION:
✅ All plants heat_status: "normal"
✅ Critical safety systems operational
✅ Master orchestrator 3/3 pods healthy
✅ 99/99 sub-agents connected
✅ Heat optimization running at 87% efficiency
✅ No safety alarms active

POST-INCIDENT ACTIONS:
- Post-mortem scheduled: <date/time>
- Safety review required: YES (Plant Safety Officer)
- Root cause analysis: In progress
- Preventive measures: To be determined

War Room: #incident-heat-loss-<timestamp> (archived)
```

**Status Page Update:**
```
RESOLVED: Process Heat Operations Restored

The GL-001 Process Heat Orchestrator service has been fully restored.
All industrial facilities are receiving normal heat supply, and all
safety systems are operating correctly.

Incident Duration: XX minutes
Affected Facilities: [List]
Root Cause: [Brief description]
Preventive Measures: We are conducting a comprehensive safety review
and implementing additional safeguards to prevent recurrence.

Next Update: Post-incident review report within 48 hours
```

---

### P0 - Critical: Master Orchestrator Failure

**SCENARIO: All master orchestrator pods failing, cannot coordinate sub-agents**

#### Step 1: Acknowledge (0-5 minutes)

```
@here P0 CRITICAL: GL-001 MASTER ORCHESTRATOR DOWN

Incident Commander: @your-name
Status: INVESTIGATING
Impact: Cannot coordinate 99 sub-agents, heat optimization disabled
Plants Affected: Coordination lost for ALL plants
Started: <timestamp>

Immediate Priority: Restore master orchestration to prevent heat loss
```

#### Step 2: Rapid Assessment (5-10 minutes)

```bash
# Check all pods
kubectl get pods -n greenlang | grep gl-001

# Check pod failure reasons
kubectl describe pods -n greenlang -l app=gl-001-process-heat-orchestrator | \
  grep -A 10 -E '(Events|Conditions)'

# Common failure patterns:
# - CrashLoopBackOff: Application crash, check logs
# - ImagePullBackOff: Image registry issue
# - OOMKilled: Memory limit exceeded
# - Pending: Resource constraints or scheduling issues

# Check logs from all failed pods
for pod in $(kubectl get pods -n greenlang -l app=gl-001-process-heat-orchestrator -o name); do
  echo "=== $pod ==="
  kubectl logs -n greenlang $pod --tail=50 --previous 2>/dev/null || \
  kubectl logs -n greenlang $pod --tail=50
done

# Check recent deployments
kubectl rollout history deployment/gl-001-process-heat-orchestrator -n greenlang
```

#### Step 3: Emergency Recovery (10-20 minutes)

**If Recent Deployment:**
```bash
# Emergency rollback
kubectl rollout undo deployment/gl-001-process-heat-orchestrator -n greenlang
kubectl rollout status deployment/gl-001-process-heat-orchestrator -n greenlang --timeout=5m
```

**If OOMKilled:**
```bash
# Increase memory limits immediately
kubectl set resources deployment gl-001-process-heat-orchestrator -n greenlang \
  --limits=memory=8Gi --requests=memory=4Gi

# Reduce replica count temporarily to free resources
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=1 -n greenlang

# Wait for single pod to start
kubectl wait --for=condition=ready pod -l app=gl-001-process-heat-orchestrator -n greenlang --timeout=5m

# Scale back up gradually
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=3 -n greenlang
```

**If Image Pull Issues:**
```bash
# Check image pull secrets
kubectl get secrets -n greenlang | grep docker

# Recreate pull secret if needed
kubectl create secret docker-registry greenlang-registry \
  --docker-server=gcr.io \
  --docker-username=_json_key \
  --docker-password="$(cat gcr-key.json)" \
  -n greenlang

# Force image pull with patch
kubectl patch deployment gl-001-process-heat-orchestrator -n greenlang \
  --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/imagePullPolicy", "value":"Always"}]'

kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

**If Resource Constraints:**
```bash
# Add nodes immediately
eksctl scale nodegroup --cluster=greenlang-cluster --nodes=10 --name=standard-workers

# Or evict lower priority workloads
kubectl delete pod -n greenlang -l priority=low

# Verify pod scheduling
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep gl-001
```

---

### P1 - High: Single Plant Heat Loss

**SCENARIO: One plant experiencing heat supply failure**

#### Response Procedure (15-30 minutes)

```bash
# 1. Identify affected plant
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status | \
  jq '.plants[] | select(.heat_loss == true)'

# 2. Check sub-agents for that plant
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq '.agents[] | select(.plant_id == "Plant-001")'

# 3. Restart affected sub-agents
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
kubectl rollout restart deployment/gl-003-combustion-optimizer -n greenlang

# 4. Verify heat restoration to plant
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status | \
  jq '.plants[] | select(.plant_id == "Plant-001")'

# 5. Check SCADA connection for that plant
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/scada/status | \
  jq '.scada_feeds[] | select(.plant_id == "Plant-001")'
```

---

### P1 - High: Sub-Agent Cascade Failure Recovery

**SCENARIO: 5-10 sub-agents failing simultaneously**

```bash
# 1. Identify failing agents
kubectl get pods -n greenlang | grep -E '(gl-002|gl-003|gl-004|gl-005)' | grep -v Running

# 2. Group by failure type
kubectl get pods -n greenlang -o json | \
  jq '.items[] | select(.status.phase != "Running") | {name: .metadata.name, reason: .status.containerStatuses[0].state}'

# 3. If same failure across agents (e.g., all OOMKilled)
# Increase resources for entire agent group
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang --limits=memory=4Gi
kubectl set resources deployment gl-003-combustion-optimizer -n greenlang --limits=memory=4Gi
# Repeat for all affected agents

# 4. If message bus issue (agents can't connect to master)
kubectl rollout restart deployment/kafka -n greenlang
# Or RabbitMQ:
kubectl rollout restart deployment/rabbitmq -n greenlang

# 5. Verify agents reconnecting
watch -n 5 'kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | jq ".agents | length"'
```

---

## Process Heat Specific Incidents

### Critical: Total Facility Heat Loss

**Emergency Heat Restoration Procedures (Target: <10 minutes)**

```bash
# STEP 1: Verify heat loss scope
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/emergency/heat-loss-assessment \
    -H "Content-Type: application/json" \
    -d '{
      "assessment_type": "immediate",
      "include_backup_systems": true,
      "priority": "life_safety"
    }'

# STEP 2: Activate backup heat systems
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/emergency/activate-backup-heat \
    -H "Content-Type: application/json" \
    -d '{
      "plants": ["all"],
      "backup_systems": ["emergency_boilers", "standby_heaters"],
      "priority": "life_safety_first",
      "authorized_by": "your-name"
    }'

# STEP 3: Switch to degraded operation mode (maintain critical heat only)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/emergency/degraded-mode \
    -H "Content-Type: application/json" \
    -d '{
      "mode": "critical_heat_only",
      "non_critical_processes": "suspend",
      "reason": "total_heat_loss_recovery"
    }'

# STEP 4: Manual heat distribution (bypass optimization)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/emergency/manual-heat-distribution \
    -H "Content-Type: application/json" \
    -d '{
      "distribution_strategy": "last_known_good",
      "override_optimization": true,
      "target": "restore_critical_heat_asap"
    }'

# STEP 5: Verify critical heat restored
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/emergency/critical-heat-status | \
  jq '.critical_systems[] | {system, heat_ok, temperature_c}'
```

**Critical Heat Priority Levels:**
1. **Life Safety (0-10 min)**: Hospital HVAC, personnel safety systems
2. **Product Safety (10-30 min)**: Food processing, pharmaceutical reactors
3. **Equipment Protection (30-60 min)**: Freeze protection, condensate systems
4. **Production (1-4 hours)**: Normal manufacturing processes

---

### Heat Distribution Optimization Failure

**SCENARIO: LP solver failing, cannot optimize heat distribution**

```bash
# 1. Check optimization status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/status

# 2. Switch to heuristic algorithm (faster, sub-optimal but functional)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/optimization/heat-distribution/algorithm \
    -H "Content-Type: application/json" \
    -d '{
      "algorithm": "greedy_heuristic",
      "reason": "LP_solver_failure",
      "fallback_mode": true
    }'

# 3. Check logs for LP solver errors
kubectl logs -n greenlang deployment/gl-001-process-heat-orchestrator \
  --tail=200 | grep -E '(LP_solver|optimization_failed|CPLEX|Gurobi)'

# 4. If license issue (Gurobi/CPLEX)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/optimization/license-check

# 5. If constraint violation (infeasible problem)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/constraints | \
  jq '.violated_constraints'

# Temporarily relax constraints
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/optimization/relax-constraints \
    -H "Content-Type: application/json" \
    -d '{
      "constraints": ["min_efficiency"],
      "relaxation_factor": 0.9,
      "temporary": true,
      "duration_minutes": 60
    }'
```

---

### Multi-Plant Coordination Failure

**SCENARIO: Master cannot coordinate across plants**

```bash
# 1. Check inter-plant network connectivity
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/network/plant-connectivity | \
  jq '.plants[] | {plant_id, reachable, latency_ms}'

# 2. Check message bus health (Kafka/RabbitMQ)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/message-bus/health | \
  jq '.status'

# 3. If message bus down, restart
kubectl rollout restart deployment/kafka -n greenlang
kubectl wait --for=condition=available deployment/kafka -n greenlang --timeout=5m

# 4. Switch to plant-local control temporarily
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/emergency/plant-local-control \
    -H "Content-Type: application/json" \
    -d '{
      "mode": "autonomous",
      "plants": ["all"],
      "reason": "multi_plant_coordination_failure",
      "revert_when_restored": true
    }'

# 5. Verify each plant operating independently
for plant in Plant-001 Plant-002 Plant-003; do
  kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
    curl -s http://localhost:8000/api/v1/plants/$plant/autonomous-status
done
```

---

## Safety Protocols

### Immediate Safety Actions

**ALWAYS prioritize safety over system restoration**

1. **If boiler pressure alarm**:
   ```bash
   # Immediately notify Plant Safety Officer
   # Trigger emergency shutdown if pressure >95% of safety valve setpoint
   kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
     curl -X POST http://localhost:8000/api/v1/emergency/safety-shutdown \
       -H "Content-Type: application/json" \
       -d '{
         "reason": "boiler_pressure_alarm",
         "equipment": ["boiler-001"],
         "authorized_by": "your-name",
         "notify_safety_officer": true
       }'
   ```

2. **If combustion instability** (risk of explosion):
   ```bash
   # Immediate fuel cutoff
   kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
     curl -X POST http://localhost:8000/api/v1/emergency/fuel-cutoff \
       -H "Content-Type: application/json" \
       -d '{
         "equipment": ["furnace-001"],
         "reason": "combustion_instability",
         "immediate": true
       }'
   ```

3. **If toxic gas detection**:
   ```bash
   # Notify safety officer and emergency services
   # Trigger area evacuation alarms
   # Log incident for regulatory reporting
   ```

### Safety Officer Escalation

**ALWAYS escalate to Plant Safety Officer for:**
- Any P0 heat loss incident
- Boiler or pressure vessel alarms
- Combustion instability or fire risk
- Toxic gas or chemical exposure
- Equipment failure with injury risk
- Emissions violations requiring regulatory reporting

**Contact:**
- **24/7 Hotline**: (see PagerDuty emergency contacts)
- **Email**: safety@greenlang.io
- **Slack**: @plant-safety-officer

---

## Escalation Paths

### Escalation Chain

```
Level 1: Primary On-Call Engineer (5 min response)
   ↓ (15 min if no progress)
Level 2: Secondary On-Call Engineer (10 min response)
   ↓ (30 min if no progress OR P0 heat loss)
Level 3: Plant Operations Manager (immediate for P0)
   ↓ (simultaneously for P0 heat loss)
Level 4: Plant Safety Officer (immediate for P0 safety)
   ↓ (1 hour if unresolved P0)
Level 5: Engineering Manager
   ↓ (2 hours if unresolved P0)
Level 6: VP Operations
```

### Automatic Escalation Triggers

**Immediate (P0)**:
- Total heat loss (any duration)
- Multi-plant cascade (3+ plants)
- Safety alarm (pressure, fire, toxic gas)
→ Escalate to Plant Operations Manager AND Plant Safety Officer immediately

**15 Minutes (P1)**:
- Single plant heat loss not restored
- Master orchestrator down with no recovery
→ Escalate to Plant Operations Manager

**30 Minutes (P1)**:
- Sub-agent cascade affecting >10 agents
- Critical optimization failures
→ Escalate to Engineering Manager

**1 Hour (P0)**:
- Any P0 incident not resolved
→ Escalate to Engineering Manager AND VP Operations

---

## Communication Templates

### P0 - Initial Alert

```
@channel @plant-safety-officer @plant-operations-manager

🚨 P0 CRITICAL INCIDENT 🚨

Incident Type: [Total Heat Loss / Master Orchestrator Down / Safety Hazard]
Incident Commander: @your-name
Status: [Investigating / Mitigating / Restoring]
Started: <timestamp>

Affected Systems:
- Plants: [Plant-001, Plant-002, ...] or ALL
- Heat Loss: [YES/NO]
- Safety Systems Affected: [YES/NO]
- Sub-Agents Down: [count]

Immediate Actions Taken:
1. [Action 1]
2. [Action 2]
3. [Action 3]

Current Priority: [Restore heat / Restart master / Safety shutdown]
ETA to Restoration: [time estimate]

Next Update: 15 minutes
War Room: #incident-<timestamp>
```

### P0 - Update (Every 15 Minutes)

```
🚨 P0 UPDATE - <timestamp>

Status: [In Progress / Improving / Degrading]
Time Elapsed: <XX minutes>

Progress:
✅ [Completed action 1]
✅ [Completed action 2]
⏳ [In progress action 3]
❌ [Blocked action 4 - reason]

Heat Status:
- Plant-001: [Normal / Degraded / Lost]
- Plant-002: [Normal / Degraded / Lost]
[... for all plants]

Master Orchestrator: [Healthy / Degraded / Down]
Sub-Agents: [XX/99 connected]

Next Actions:
1. [Next action 1]
2. [Next action 2]

ETA: [updated estimate]
Next Update: 15 minutes
```

### P0 - Resolution

```
✅ P0 INCIDENT RESOLVED ✅

Incident: [Description]
Duration: <XX minutes>
Resolved: <timestamp>

Final Status:
✅ Heat supply restored to all plants
✅ Master orchestrator: 3/3 pods healthy
✅ Sub-agents: 99/99 connected
✅ Safety systems: All normal
✅ No active alarms

Root Cause: [Brief description]

Mitigation Actions:
1. [Action 1]
2. [Action 2]
3. [Action 3]

Post-Incident Actions:
- Post-mortem: Scheduled for <date/time>
- Safety review: [Required/Not Required]
- Root cause analysis: [Assigned to]
- Preventive measures: [TBD]

Thank you to everyone involved in the rapid response.
```

---

## Emergency Rollback Procedures

### Emergency Rollback (<5 Minutes)

**For critical heat loss scenarios requiring immediate rollback**

```bash
# 1. ROLLBACK MASTER IMMEDIATELY
kubectl rollout undo deployment/gl-001-process-heat-orchestrator -n greenlang

# 2. MONITOR ROLLBACK
kubectl rollout status deployment/gl-001-process-heat-orchestrator -n greenlang --timeout=5m

# 3. VERIFY HEAT COORDINATION RESTORED
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status | \
  jq '.plants[] | select(.heat_loss == true)'

# Expected: No plants with heat_loss == true

# 4. IF SUB-AGENTS ALSO AFFECTED (Coordinated Rollback)
# Rollback critical sub-agents in parallel
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang &
kubectl rollout undo deployment/gl-003-combustion-optimizer -n greenlang &
kubectl rollout undo deployment/gl-004-emissions-control -n greenlang &
wait

# 5. VERIFY ALL AGENTS CONNECTED
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq '.agents | map(select(.status != "connected")) | length'

# Expected: 0 (all agents connected)
```

**See ROLLBACK_PROCEDURE.md for complete rollback procedures**

---

## Post-Incident Review

### Post-Incident Review Template

**Incident ID**: INC-<YYYYMMDD>-<NNN>
**Date**: <date>
**Severity**: P0 / P1 / P2
**Duration**: <total duration>
**Incident Commander**: <name>

#### 1. Incident Summary

**What Happened**:
<Brief description of the incident>

**Impact**:
- Plants Affected: [list]
- Heat Loss Duration: <duration>
- Sub-Agents Affected: <count>
- Customer Impact: [Yes/No, description]
- Safety Impact: [Yes/No, description]

#### 2. Timeline

| Time | Event |
|------|-------|
| T+0m | Incident detected |
| T+5m | On-call acknowledged |
| T+10m | Assessment complete |
| T+15m | Mitigation started |
| T+30m | Heat restored to critical systems |
| T+45m | Full resolution |

#### 3. Root Cause Analysis

**Primary Root Cause**:
<Detailed description>

**Contributing Factors**:
1. <Factor 1>
2. <Factor 2>

**Why It Wasn't Caught**:
<Gaps in monitoring, testing, etc.>

#### 4. What Went Well

- <Positive aspect 1>
- <Positive aspect 2>
- <Positive aspect 3>

#### 5. What Went Poorly

- <Issue 1>
- <Issue 2>
- <Issue 3>

#### 6. Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| <Action 1> | <Name> | <Date> | Open |
| <Action 2> | <Name> | <Date> | Open |
| <Action 3> | <Name> | <Date> | Open |

#### 7. Preventive Measures

**Short-term** (Implement within 1 week):
1. <Measure 1>
2. <Measure 2>

**Long-term** (Implement within 1 month):
1. <Measure 1>
2. <Measure 2>

#### 8. Runbook Updates

**Updates Required**:
- [ ] TROUBLESHOOTING.md: Add scenario for <specific issue>
- [ ] INCIDENT_RESPONSE.md: Update <procedure>
- [ ] ROLLBACK_PROCEDURE.md: Clarify <step>
- [ ] SCALING_GUIDE.md: Add capacity planning for <scenario>

#### 9. Approvals

- Engineering Manager: <signature/approval>
- Plant Safety Officer: <signature/approval> (if safety incident)
- VP Operations: <signature/approval> (if P0)

---

**Last Updated**: 2025-11-17
**Version**: 1.0
**Maintained By**: GreenLang Platform Engineering & Process Heat Operations
**Review Cycle**: After each P0/P1 incident or quarterly
