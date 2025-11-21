# GL-004 BurnerOptimizationAgent - Rollback Procedure

## When to Execute Rollback

Initiate rollback if:
- ✗ Emissions exceed limits for >15 minutes
- ✗ Safety interlocks triggering repeatedly (>3 in 1 hour)
- ✗ Burner instability or flame loss events
- ✗ Optimization causing damage risk
- ✗ Agent crashes (>5 restarts in 10 minutes)
- ✗ Critical bugs discovered in production

## Pre-Rollback Checklist

- [ ] Document current system state and errors
- [ ] Capture current configuration (ConfigMap, Secrets)
- [ ] Export optimization history
- [ ] Record error logs and metrics
- [ ] Identify last known good configuration/version
- [ ] Verify rollback target version availability
- [ ] Notify operations and management teams
- [ ] Get stakeholder approval (for P0/P1)
- [ ] Prepare rollback communication

## Rollback Types

### 1. Configuration Rollback (Fastest - 5 minutes)

**When to Use:** Configuration change caused issues, code is stable

```bash
# 1. Identify last good configuration
kubectl get configmap gl-004-config -n greenlang -o yaml > current-config.yaml

# 2. Restore from backup
kubectl apply -f backup/configmap-<timestamp>.yaml

# 3. Restart pods to apply
kubectl rollout restart deployment/gl-004 -n greenlang

# 4. Monitor rollout
kubectl rollout status deployment/gl-004 -n greenlang

# 5. Verify health
curl http://gl-004.greenlang.svc.cluster.local:8000/health
```

### 2. Application Rollback (Medium - 10 minutes)

**When to Use:** Code change caused issues

```bash
# 1. Check rollout history
kubectl rollout history deployment/gl-004 -n greenlang

# 2. Rollback to previous version
kubectl rollout undo deployment/gl-004 -n greenlang

# 3. OR rollback to specific revision
kubectl rollout undo deployment/gl-004 --to-revision=<number> -n greenlang

# 4. Monitor status
kubectl rollout status deployment/gl-004 -n greenlang
kubectl get pods -n greenlang -l app=gl-004

# 5. Verify version
kubectl describe deployment gl-004 -n greenlang | grep Image
```

### 3. Full System Rollback (Comprehensive - 30 minutes)

**When to Use:** Major issues affecting multiple components

```bash
# 1. Stop current deployment
kubectl scale deployment/gl-004 --replicas=0 -n greenlang

# 2. Restore database from backup
pg_restore -d greenlang -c backup/gl004_<timestamp>.dump

# 3. Restore configurations
kubectl apply -f backup/configmap-<timestamp>.yaml
kubectl apply -f backup/secrets-<timestamp>.yaml
kubectl apply -f backup/deployment-<timestamp>.yaml

# 4. Restore service and other resources
kubectl apply -f backup/service-<timestamp>.yaml

# 5. Scale up deployment
kubectl scale deployment/gl-004 --replicas=3 -n greenlang

# 6. Verify all pods running
kubectl get pods -n greenlang -l app=gl-004 -w
```

## Post-Rollback Validation

### 1. Health Checks
```bash
# Agent health
curl http://gl-004:8000/health

# Readiness
curl http://gl-004:8000/readiness

# Detailed status
curl http://gl-004:8000/status | jq '.'
```

### 2. Functional Validation
```bash
# Test burner state collection
curl http://gl-004:8000/burner/state | jq '.o2_level, .burner_load'

# Trigger test optimization
curl -X POST http://gl-004:8000/optimize

# Check optimization history
curl http://gl-004:8000/optimization/history?limit=5
```

### 3. Emissions Compliance
```bash
# Check current emissions
curl http://gl-004:8000/burner/state | jq '.nox_level, .co_level'

# Verify within limits
# NOx should be <50 ppm
# CO should be <100 ppm
```

### 4. Performance Validation
- API latency P95 <200ms
- Optimization cycles <60s
- Memory usage stable
- No error spikes in logs

### 5. Monitoring Validation
```bash
# Check Prometheus metrics
curl http://gl-004:8001/metrics | grep gl004_combustion_efficiency

# Verify Grafana dashboards loading
# Check alert status in Prometheus
```

## Rollback Communication

### Internal Notification Template
```
ROLLBACK EXECUTED: GL-004 BurnerOptimizationAgent

Time: <timestamp>
Severity: P<0-2>
Reason: <brief description of issue>

Rollback Details:
- Type: <Configuration/Application/Full>
- From Version: <version/revision>
- To Version: <version/revision>
- Duration: <minutes>

Impact:
- Systems Affected: <burners, plants>
- Downtime: <minutes>
- Data Loss: <None/Minimal/Describe>

Validation Status:
- Health Checks: <Pass/Fail>
- Functional Tests: <Pass/Fail>
- Emissions Compliance: <Pass/Fail>
- Performance: <Normal/Degraded>

Current Status: <Stable/Monitoring/Issues>

Next Steps:
1. <action item>
2. <action item>

Incident Response Team:
- Lead: <name>
- Contact: <email/phone>
```

### External Stakeholder Template
```
GL-004 Burner Optimization System Update

Dear Stakeholders,

We have rolled back the GL-004 BurnerOptimizationAgent to a previous stable version due to <brief reason>.

Impact: <minimal/moderate/significant>
Current Status: System is stable and operating normally
Emissions Compliance: Maintained throughout rollback
Safety: All safety systems remained operational

We are investigating the root cause and will provide updates.

Contact: greenlang-support@example.com
```

## Data Preservation During Rollback

### Backup Current Data
```bash
# Export optimization history
kubectl exec -n greenlang <pod> -- curl \
  http://localhost:8000/optimization/history?limit=1000 \
  > backup/opt-history-<timestamp>.json

# Backup current metrics
curl http://gl-004:8001/metrics > backup/metrics-<timestamp>.txt

# Export burner states
kubectl exec -n greenlang <pod> -- \
  psql $DATABASE_URL -c \
  "COPY burner_states TO STDOUT CSV HEADER" \
  > backup/burner-states-<timestamp>.csv
```

## Rollback Failure Scenarios

### If Rollback Fails
1. Scale deployment to 0 replicas immediately
2. Switch to manual burner control
3. Contact vendor support
4. Prepare for emergency maintenance window
5. Consider deploying from scratch

### Emergency Manual Control
```bash
# Disable GL-004 completely
kubectl delete deployment gl-004 -n greenlang

# Document manual control setpoints
# Return to DCS/SCADA manual operation
# Monitor emissions manually
```

## Post-Rollback Actions

### Immediate (0-4 hours)
- [ ] Monitor system stability
- [ ] Review all error logs
- [ ] Verify emissions compliance
- [ ] Check optimization effectiveness
- [ ] Update incident tracker

### Short-term (24 hours)
- [ ] Root cause analysis
- [ ] Document lessons learned
- [ ] Update runbooks if needed
- [ ] Plan fix for original issue
- [ ] Schedule post-mortem meeting

### Long-term (1 week)
- [ ] Implement preventive measures
- [ ] Enhance testing procedures
- [ ] Update deployment checklist
- [ ] Review change management process
- [ ] Train team on learnings

## Related Runbooks
- INCIDENT_RESPONSE.md
- TROUBLESHOOTING.md
- MAINTENANCE.md
- DEPLOYMENT_GUIDE.md
