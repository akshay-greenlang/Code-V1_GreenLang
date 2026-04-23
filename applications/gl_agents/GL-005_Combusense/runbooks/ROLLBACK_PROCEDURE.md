# GL-005 COMBUSTIONCONTROLAGENT - ROLLBACK PROCEDURE

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Owner:** GreenLang Industrial Control Operations Team
**Classification:** OPERATIONAL CRITICAL - SAFETY SYSTEM

---

## PURPOSE

This runbook defines rollback procedures for the GL-005 CombustionControlAgent, ensuring rapid recovery from failed deployments, configuration errors, or system issues while maintaining combustion safety, production continuity, and SIL-2 safety integrity.

**CRITICAL NOTE:** Combustion control is a **safety-critical system**. All rollbacks must preserve safety interlocks and emergency shutdown capability at all times.

---

## WHEN TO EXECUTE ROLLBACK

Initiate rollback if:
- ✗ Control calculations producing unsafe results (safety risk)
- ✗ Safety interlock logic failures or false trips
- ✗ PID controller instability causing equipment stress
- ✗ DCS/PLC integration failures preventing control
- ✗ Performance degradation >20% after deployment
- ✗ Critical bugs discovered in control algorithms
- ✗ Database migration failures affecting control state
- ✗ Pod crash loops (>5 restarts in 10 minutes)
- ✗ Flame stability issues after update
- ✗ Emissions compliance violations after deployment

**DO NOT ROLLBACK IF:**
- ✓ Safety systems are functioning correctly
- ✓ Issue is cosmetic (dashboards, logging, reporting)
- ✓ Alternative mitigation is safer than rollback

---

## PRE-ROLLBACK CHECKLIST

### Safety Verification
- [ ] Verify safety interlocks are active and functioning
- [ ] Confirm flame scanners operational
- [ ] Check emergency shutdown capability
- [ ] Review current combustion parameters (within safe limits)
- [ ] Notify control room operators of impending rollback

### System State Documentation
- [ ] Document current system state and error symptoms
- [ ] Capture logs from failing pods (last 2 hours)
- [ ] Export recent control event audit trails (compliance critical)
- [ ] Record current ConfigMap and Secret versions
- [ ] Identify last known good deployment version
- [ ] Verify rollback target availability in container registry
- [ ] Check DCS/PLC connection status

### Approvals & Communication
- [ ] Get approval for P0/P1 rollbacks (VP Engineering + Safety Officer)
- [ ] Notify stakeholders (Operations Manager, Plant Safety, Production)
- [ ] Prepare rollback communication plan
- [ ] Schedule rollback during low-load period if possible
- [ ] Ensure backup control operator on standby

### Data Preservation
- [ ] Ensure database backup is recent (<1 hour old for production)
- [ ] Backup current control state (PID tuning, setpoints)
- [ ] Export control performance data for post-mortem

---

## ROLLBACK TYPES

### 1. Configuration Rollback (Fastest - 5 minutes)

**When to Use:** ConfigMap/Secret change caused issues, code is stable

**Impact:** Minimal - Configuration only, no code changes

**Safety Impact:** Low (if safety config unchanged)

#### Procedure

```bash
# 1. Identify current ConfigMap version
kubectl get configmap gl-005-config -n greenlang -o yaml > /tmp/current-config.yaml
cat /tmp/current-config.yaml | grep "resourceVersion:"

# 2. List ConfigMap history (if using GitOps)
git log --oneline k8s/configmap.yaml | head -10

# 3. Review changes to identify problematic config
git diff HEAD~1 k8s/configmap.yaml

# 4. Restore from backup (if GitOps not used)
kubectl apply -f backup/configmap-$(date -d "yesterday" +%Y%m%d).yaml

# OR restore from Git
git checkout HEAD~1 k8s/configmap.yaml
kubectl apply -f k8s/configmap.yaml

# 5. Restart pods to apply new config (rolling restart)
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang

# 6. Monitor rollout (ensure smooth transition)
kubectl rollout status deployment/gl-005-combustion-control -n greenlang -w

# 7. Verify configuration applied
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  env | grep -E "PID_|SAFETY_|CONTROL_MODE"

# 8. Test with read-only data acquisition (no control commands)
curl -X POST http://gl-005-combustion-control:8000/test/data-acquisition \
  -H "Content-Type: application/json" \
  -d '{"unit_id": "BOILER001", "duration_sec": 30, "mode": "passive"}'

# 9. Verify safety interlocks
curl http://gl-005-combustion-control:8000/safety/status | jq '.safety_status'
# Expected: "OK" or "WARNING" (not "CRITICAL" or "EMERGENCY_STOP")
```

#### Verification

```bash
# Check health
curl http://gl-005-combustion-control:8000/health | jq '.status'
# Expected: "healthy"

# Verify control calculations match baseline
python tests/verify_control_calculations.py \
  --baseline tests/baselines/control_baseline.json \
  --tolerance 0.01

# Verify DCS/PLC connectivity
curl http://gl-005-combustion-control:8000/integrations/status | \
  jq '.integrations[] | select(.status != "connected")'
# Expected: No output (all integrations connected)

# Monitor error rates for 10 minutes
watch -n 30 'curl -s http://gl-005-combustion-control:8001/metrics | grep gl005_errors_total'
```

---

### 2. Application Rollback (Medium - 10 minutes)

**When to Use:** Code deployment caused issues (control logic bugs, integration failures)

**Impact:** Moderate - Application code changes reverted

**Safety Impact:** Medium (requires safety validation after rollback)

#### Procedure

```bash
# 1. Check rollout history
kubectl rollout history deployment/gl-005-combustion-control -n greenlang

# Output example:
# REVISION  CHANGE-CAUSE
# 1         Initial deployment
# 2         Update PID tuning parameters
# 3         Add feedforward compensation
# 4         Current (BROKEN - PID instability)

# 2. Review specific revision
kubectl rollout history deployment/gl-005-combustion-control -n greenlang --revision=3

# 3. Put system in safe state before rollback
# Option A: Graceful handoff to manual control
curl -X POST http://gl-005-combustion-control:8000/control/handoff \
  -H "Content-Type: application/json" \
  -d '{"mode": "manual", "notify_operators": true}'

# Option B: Reduce to monitoring-only mode (no control commands)
kubectl set env deployment/gl-005-combustion-control \
  CONTROL_MODE=monitoring_only \
  -n greenlang

# Wait 30 seconds for mode transition
sleep 30

# 4. Rollback to previous version (revision N-1)
kubectl rollout undo deployment/gl-005-combustion-control -n greenlang

# OR rollback to specific revision
kubectl rollout undo deployment/gl-005-combustion-control -n greenlang --to-revision=3

# 5. Monitor rollback progress
kubectl rollout status deployment/gl-005-combustion-control -n greenlang -w

# 6. Verify all pods running
kubectl get pods -n greenlang -l app=gl-005-combustion-control -o wide

# Expected: All pods in Running state, READY 1/1

# 7. Check deployed image version
kubectl get deployment gl-005-combustion-control -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'

# Should show previous image tag

# 8. Verify application version
curl http://gl-005-combustion-control:8000/version
```

#### Post-Rollback Validation

```bash
# 1. Run safety interlock validation
python tests/test_safety_interlocks.py \
  --env production \
  --unit BOILER001 \
  --validate-all

# Expected: All safety tests passing

# 2. Run control loop stability test (passive monitoring)
curl -X POST http://gl-005-combustion-control:8000/test/control-stability \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": "BOILER001",
    "duration_sec": 300,
    "mode": "passive",
    "setpoint_mw": 50.0
  }'

# 3. Verify calculations deterministic
python tests/verify_determinism.py \
  --test-cases tests/determinism/control_test_cases.json \
  --runs 10 \
  --tolerance 1e-10

# 4. Check all agents healthy
curl http://gl-005-combustion-control:8000/health/agents | \
  jq '.agents[] | select(.status != "healthy")'

# Expected: No output (all 5 agents healthy)

# 5. Monitor control loop latency
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_control_loop_duration_seconds{quantile="0.95"}'

# Expected: <0.1 (100ms P95 latency)

# 6. Resume automatic control (if safe)
curl -X POST http://gl-005-combustion-control:8000/control/resume \
  -H "Content-Type: application/json" \
  -d '{"mode": "automatic", "ramp_rate_mw_per_min": 5.0}'
```

---

### 3. Full System Rollback (Comprehensive - 30 minutes)

**When to Use:** Major issues affecting multiple components, database schema changes, complete system failure, safety system compromise

**Impact:** High - Full application, configuration, and potentially database rollback

**Safety Impact:** High (requires comprehensive safety validation)

#### Procedure

```bash
# ============================================================================
# PHASE 1: EMERGENCY SAFE STATE
# ============================================================================

# 1. Transfer control to backup system or manual operation
curl -X POST http://gl-005-combustion-control:8000/emergency/transfer-control \
  -H "Content-Type: application/json" \
  -d '{
    "target": "backup_plc",
    "reason": "Full system rollback required",
    "notify_operators": true,
    "emergency_mode": true
  }'

# Verify transfer complete
curl http://gl-005-combustion-control:8000/emergency/status | jq '.control_authority'
# Expected: "backup_plc" or "manual"

# 2. Scale deployment to 0 (stop all processing)
kubectl scale deployment/gl-005-combustion-control --replicas=0 -n greenlang

# 3. Verify all pods terminated
kubectl get pods -n greenlang -l app=gl-005-combustion-control

# Expected: No pods running

# 4. Document current state
kubectl get all -n greenlang -l app=gl-005-combustion-control -o yaml > \
  /tmp/current_state_$(date +%Y%m%d_%H%M%S).yaml

# ============================================================================
# PHASE 2: DATABASE ROLLBACK (IF NEEDED)
# ============================================================================

# Only perform if database schema changed or control state corrupted

# 1. Check recent database backups
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  ls -lh /backups/ | tail -10

# 2. Identify backup to restore (within last 4 hours)
BACKUP_FILE="gl005_db_$(date -d '2 hours ago' +%Y%m%d_%H)0000.sql"

# 3. Stop database connections (already done - deployment scaled to 0)

# 4. Restore database
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang < /backups/$BACKUP_FILE

# 5. Verify database restore
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "SELECT COUNT(*) FROM control_events;"

kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "SELECT MAX(timestamp) FROM combustion_data;"

# 6. Check database version/schema
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c \
  "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"

# ============================================================================
# PHASE 3: RESTORE KUBERNETES RESOURCES
# ============================================================================

# 1. Restore ConfigMap
kubectl apply -f backup/configmap-$(date -d "yesterday" +%Y%m%d).yaml

# 2. Restore Secrets (if changed)
kubectl apply -f backup/secrets-$(date -d "yesterday" +%Y%m%d).yaml

# 3. Restore Deployment (specific image version)
kubectl apply -f backup/deployment-$(date -d "yesterday" +%Y%m%d).yaml

# OR manually set image version
kubectl set image deployment/gl-005-combustion-control \
  gl-005-combustion-control=greenlang/gl-005-combustion-control:v1.2.3 \
  -n greenlang

# 4. Restore Service
kubectl apply -f backup/service-$(date -d "yesterday" +%Y%m%d).yaml

# 5. Restore ServiceMonitor (Prometheus)
kubectl apply -f backup/servicemonitor-$(date -d "yesterday" +%Y%m%d).yaml

# 6. Restore HPA
kubectl apply -f backup/hpa-$(date -d "yesterday" +%Y%m%d).yaml

# ============================================================================
# PHASE 4: SCALE UP AND VERIFY
# ============================================================================

# 1. Scale deployment to normal replica count
kubectl scale deployment/gl-005-combustion-control --replicas=3 -n greenlang

# 2. Monitor pod startup
kubectl get pods -n greenlang -l app=gl-005-combustion-control -w

# Wait for all pods to be Running and READY 1/1

# 3. Check pod logs for errors
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=50

# 4. Verify all endpoints
curl http://gl-005-combustion-control:8000/health
curl http://gl-005-combustion-control:8000/health/agents
curl http://gl-005-combustion-control:8000/health/database
curl http://gl-005-combustion-control:8000/safety/status

# ============================================================================
# PHASE 5: SAFETY VALIDATION
# ============================================================================

# 1. Verify safety interlocks
python tests/test_safety_interlocks.py \
  --env production \
  --comprehensive \
  --output /tmp/safety_validation_$(date +%Y%m%d_%H%M%S).json

# 2. Test emergency shutdown
curl -X POST http://gl-005-combustion-control:8000/test/emergency-shutdown \
  -H "Content-Type: application/json" \
  -d '{"unit_id": "BOILER001", "test_mode": true, "dry_run": true}'

# 3. Verify sensor connectivity
curl http://gl-005-combustion-control:8000/integrations/status | \
  jq '.integrations[] | {name: .name, status: .status, latency_ms: .latency_ms}'

# All should show "connected" with latency <100ms

# ============================================================================
# PHASE 6: FUNCTIONAL VALIDATION
# ============================================================================

# 1. Test data acquisition (passive monitoring)
curl -X POST http://gl-005-combustion-control:8000/control/start \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": "BOILER001",
    "control_mode": "monitoring_only",
    "duration_sec": 300
  }'

# 2. Verify control calculations (passive)
python tests/verify_control_calculations.py \
  --baseline tests/baselines/control_baseline.json \
  --live-test \
  --duration 300 \
  --tolerance 0.01

# 3. Test PID controller (simulation mode)
curl -X POST http://gl-005-combustion-control:8000/test/pid-controller \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": "BOILER001",
    "setpoint_mw": 50.0,
    "simulation_mode": true,
    "duration_sec": 600
  }'

# 4. Verify emissions calculations
python tests/test_emissions_calculations.py \
  --fuel-type natural_gas \
  --load-range 10-100 \
  --validate-compliance

# 5. Check compliance validation
curl http://gl-005-combustion-control:8000/compliance/status | \
  jq '.compliance_status'

# Expected: "compliant"

# ============================================================================
# PHASE 7: CONTROLLED RESUME
# ============================================================================

# 1. Transfer control back from backup (gradual handoff)
curl -X POST http://gl-005-combustion-control:8000/control/transfer \
  -H "Content-Type: application/json" \
  -d '{
    "from": "backup_plc",
    "to": "gl005_agent",
    "mode": "gradual",
    "ramp_duration_sec": 600,
    "monitor_performance": true
  }'

# 2. Monitor transfer progress
watch -n 5 'curl -s http://gl-005-combustion-control:8000/control/transfer/status | jq'

# 3. Verify performance post-transfer
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_control_performance'
```

---

## DATA PRESERVATION DURING ROLLBACK

### Critical Data to Preserve

```bash
# 1. Export control event audit trails (COMPLIANCE CRITICAL)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from backend.app import export_all_control_events
export_all_control_events('/exports/control_events_$(date +%Y%m%d_%H%M%S).json')
"

# 2. Backup current PID tuning parameters
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from backend.app import export_pid_tuning
export_pid_tuning('/exports/pid_tuning_$(date +%Y%m%d_%H%M%S).json')
"

# 3. Export safety configuration
kubectl get configmap gl-005-safety-config -n greenlang -o yaml > \
  /backup/safety_config_$(date +%Y%m%d_%H%M%S).yaml

# 4. Backup current metrics
curl http://gl-005-combustion-control:8001/metrics > \
  /backup/metrics_$(date +%Y%m%d_%H%M%S).txt

# 5. Export recent combustion data (last 24 hours)
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c \
  "COPY (SELECT * FROM combustion_data
   WHERE timestamp > NOW() - INTERVAL '24 hours')
   TO STDOUT CSV HEADER" > /backup/combustion_data_$(date +%Y%m%d).csv

# 6. Export control performance data
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c \
  "COPY (SELECT * FROM control_performance
   WHERE timestamp > NOW() - INTERVAL '7 days')
   TO STDOUT CSV HEADER" > /backup/control_performance_$(date +%Y%m%d).csv

# 7. Export safety event log (compliance)
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c \
  "COPY (SELECT * FROM safety_events
   WHERE timestamp > NOW() - INTERVAL '90 days')
   TO STDOUT CSV HEADER" > /backup/safety_events_$(date +%Y%m%d).csv

# 8. Export emission factor database
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  cat /app/data/emission_factors.py > /backup/emission_factors_$(date +%Y%m%d).py
```

---

## ROLLBACK COMMUNICATION

### Internal Notification Template

```
ROLLBACK EXECUTED: GL-005 CombustionControlAgent

Time: {{timestamp}}
Severity: P{{0-2}}
Reason: {{brief description of issue}}

Rollback Details:
- Type: Configuration / Application / Full System
- From Version: {{current_version/revision}}
- To Version: {{rollback_version/revision}}
- Duration: {{minutes}}
- Control Transfer: {{manual/backup_plc/none}}

Impact:
- Combustion Control Interrupted: {{yes/no}}
- Production Impact: {{none/minimal/moderate/severe}}
- Safety Systems: {{fully operational/degraded/offline}}
- Data Integrity: {{preserved/requires validation}}
- Emissions Compliance Risk: {{none/low/medium/high}}

Validation Status:
- Health Checks: {{Pass/Fail}}
- Safety Interlocks: {{Pass/Fail}}
- Control Calculations: {{Pass/Fail}}
- DCS/PLC Integration: {{Pass/Fail}}
- Emissions Compliance: {{Pass/Fail}}

Current Status: {{Monitoring Only/Manual Control/Automatic Control Resumed}}

Next Steps:
1. {{action item}}
2. {{action item}}

Incident Response Team:
- Lead: {{name}}
- Safety Officer: {{name}}
- Control Room: {{contact}}
```

### Plant Operations Notification Template

```
GL-005 COMBUSTION CONTROL SYSTEM UPDATE

Dear Operations Team,

We have rolled back the GL-005 CombustionControlAgent to a previous stable version due to {{brief reason}}.

Impact Assessment:
- Current Control Mode: {{manual/backup/automatic}}
- Production Impact: {{description}}
- Safety Systems: Fully operational
- Performance: {{normal/degraded by X%}}

Actions Required:
{{none / monitor closely / manual control recommended}}

Current Status: System is {{stable/under observation}} and operating {{normally/in degraded mode}}

Safety & Compliance: All safety interlocks functional. Emissions monitoring active.

Timeline:
- Issue Detected: {{timestamp}}
- Control Transferred: {{timestamp}}
- Rollback Completed: {{timestamp}}
- System Verified: {{timestamp}}
- Automatic Control Resumed: {{timestamp or "pending validation"}}

Please contact the control room if you observe any anomalies.

Contact: combustion-control@greenlang.io | Control Room: x5555
```

---

## ROLLBACK FAILURE SCENARIOS

### If Rollback Fails

1. **Pods Still Failing After Rollback:**
```bash
# Collect diagnostics
kubectl describe pod -n greenlang <failing-pod>
kubectl logs -n greenlang <failing-pod> --previous

# Check node resources
kubectl describe node <node-name>

# Check for resource exhaustion
kubectl top pods -n greenlang

# Try manual pod deletion
kubectl delete pod -n greenlang <failing-pod>

# If persistent, check safety of emergency fallback
# Transfer to backup control system
curl -X POST http://backup-control:8000/assume-control \
  -d '{"reason": "primary_system_failure", "emergency": true}'
```

2. **Database Restore Fails:**
```bash
# Use older backup
OLDER_BACKUP="gl005_db_$(date -d '4 hours ago' +%Y%m%d_%H)0000.sql"
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang < /backups/$OLDER_BACKUP

# If all backups fail, use disaster recovery backup (offsite)
# Contact DBA for disaster recovery procedure

# Rebuild from control event log if necessary
python scripts/rebuild_database_from_audit_log.py \
  --audit-log /backup/control_events_latest.json \
  --start-date $(date -d '7 days ago' +%Y-%m-%d)
```

3. **Safety System Compromise:**
```bash
# IMMEDIATE: Transfer all control to manual
curl -X POST http://gl-005-combustion-control:8000/emergency/manual-override \
  -d '{"reason": "safety_system_failure", "notify_all": true}'

# Verify safety interlocks bypass (requires dual authorization)
# Only authorized safety personnel can execute

# Run comprehensive safety diagnostics
python scripts/safety_diagnostics.py \
  --comprehensive \
  --generate-report \
  --output /tmp/safety_diagnostics_$(date +%Y%m%d_%H%M%S).pdf

# Do NOT resume automatic control until safety validated by Safety Officer
```

4. **Complete System Failure:**
```bash
# Nuclear option: Redeploy from scratch (use only if absolutely necessary)

# 1. Transfer control to backup system FIRST
curl -X POST http://backup-control:8000/assume-full-control

# 2. Delete namespace
kubectl delete namespace greenlang

# 3. Recreate namespace
kubectl create namespace greenlang

# 4. Restore from GitOps repository
kubectl apply -k k8s/overlays/production/

# 5. Restore database from disaster recovery backup
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang < /disaster-recovery/gl005_db_dr_latest.sql

# 6. Restore safety configuration (CRITICAL)
kubectl apply -f /disaster-recovery/safety_config_validated.yaml

# 7. Run comprehensive validation
python tests/comprehensive_validation.py --production

# 8. Transfer control back (gradual, monitored)
# Only after full validation by Safety Officer
```

### Emergency Manual Mode

If automated system cannot be restored safely:

```bash
# 1. Document current combustion parameters
python scripts/emergency_export_state.py \
  --dcs-endpoint "modbus://10.0.0.10:502" \
  --output /emergency/current_state_$(date +%Y%m%d_%H%M%S).json

# 2. Provide operators with manual control guidelines
python scripts/generate_manual_control_guide.py \
  --unit BOILER001 \
  --current-load-mw 50.0 \
  --fuel-type natural_gas \
  --output /emergency/manual_control_guide_$(date +%Y%m%d).pdf

# 3. Setup manual monitoring dashboards
python scripts/setup_emergency_dashboards.py \
  --grafana-url http://grafana:3000 \
  --unit BOILER001

# 4. Continuous safety monitoring (external system)
python scripts/external_safety_monitor.py \
  --dcs-endpoint "modbus://10.0.0.10:502" \
  --alert-email safety@greenlang.io \
  --alert-threshold critical
```

---

## POST-ROLLBACK ACTIONS

### Immediate (0-4 hours)

- [ ] Monitor system stability (all health checks green)
- [ ] Verify safety interlocks functioning correctly
- [ ] Review all error logs for recurring issues
- [ ] Verify control calculations match baselines
- [ ] Check emissions compliance
- [ ] Monitor control loop performance (latency, stability)
- [ ] Verify DCS/PLC integration health
- [ ] Update incident tracker
- [ ] Notify all stakeholders of resolution
- [ ] Safety Officer sign-off on system safety

### Short-term (24 hours)

- [ ] Root cause analysis completed
- [ ] Document lessons learned
- [ ] Update runbooks with new findings
- [ ] Plan permanent fix for original issue
- [ ] Schedule post-mortem meeting (P0/P1 only)
- [ ] Review and update backup procedures
- [ ] Verify all control performance data intact
- [ ] Analyze production impact
- [ ] Review emissions compliance for rollback period

### Long-term (1 week)

- [ ] Implement preventive measures
- [ ] Enhance testing procedures (prevent recurrence)
- [ ] Update deployment checklist
- [ ] Review change management process
- [ ] Train team on lessons learned
- [ ] Update monitoring and alerts
- [ ] Conduct rollback drill (practice scenario)
- [ ] Update safety documentation
- [ ] Review with Safety Committee

---

## ROLLBACK TESTING

### Quarterly Rollback Drill

**Purpose:** Ensure team proficiency and procedure accuracy

**Procedure:**
1. Schedule drill during planned maintenance window
2. Deploy intentionally degraded version to staging
3. Execute full rollback procedure
4. Measure time to recovery
5. Validate safety system integrity post-rollback
6. Document issues encountered
7. Update runbook based on findings
8. Safety Officer review and approval

**Success Criteria:**
- Configuration rollback: <5 minutes
- Application rollback: <10 minutes
- Full system rollback: <30 minutes
- All safety tests passing post-rollback
- Zero safety incidents during rollback
- All control calculations deterministic post-rollback

---

## RELATED RUNBOOKS

- INCIDENT_RESPONSE.md - For incident triage and response
- TROUBLESHOOTING.md - For diagnosing issues
- MAINTENANCE.md - For routine operations
- SCALING_GUIDE.md - For performance scaling

---

## APPENDIX: Quick Reference Commands

```bash
# Check current deployment version
kubectl get deployment gl-005-combustion-control -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'

# List rollout history
kubectl rollout history deployment/gl-005-combustion-control -n greenlang

# Rollback to previous
kubectl rollout undo deployment/gl-005-combustion-control -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/gl-005-combustion-control -n greenlang --to-revision=3

# Monitor rollback
kubectl rollout status deployment/gl-005-combustion-control -n greenlang -w

# Scale replicas
kubectl scale deployment/gl-005-combustion-control --replicas=3 -n greenlang

# Restart deployment
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang

# Check pod health
kubectl get pods -n greenlang -l app=gl-005-combustion-control

# View logs
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100

# Test health endpoint
curl http://gl-005-combustion-control:8000/health

# Test safety status
curl http://gl-005-combustion-control:8000/safety/status

# Emergency manual override
curl -X POST http://gl-005-combustion-control:8000/emergency/manual-override
```

---

**Document Control:**
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Next Review:** 2026-02-18 (Quarterly)
- **Owner:** GL-005 Operations Team
- **Approvers:** VP Engineering, Safety Officer, SRE Lead
- **Safety Classification:** SIL-2 Safety System

---

*This rollback procedure must be practiced quarterly and reviewed by the Safety Committee. Any changes require Safety Officer approval.*
