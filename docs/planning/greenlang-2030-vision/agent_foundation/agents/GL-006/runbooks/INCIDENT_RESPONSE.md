# GL-006 HeatRecoveryMaximizer Incident Response Runbook

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-006 |
| Codename | HEATRECLAIM |
| Version | 1.0.0 |
| Last Updated | 2024-11-26 |
| Owner | Platform Team |
| On-Call | platform-oncall@greenlang.io |

---

## 1. Overview

This runbook provides step-by-step procedures for responding to incidents involving the GL-006 HeatRecoveryMaximizer agent. It covers incident classification, response procedures, escalation paths, and post-incident activities.

---

## 2. Incident Classification

### 2.1 Severity Levels

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **SEV1** | Critical - Complete service outage | 15 minutes | All pods down, database unreachable |
| **SEV2** | High - Major functionality impaired | 30 minutes | Calculation failures >50%, high error rate |
| **SEV3** | Medium - Partial degradation | 2 hours | Slow response times, single pod failure |
| **SEV4** | Low - Minor issues | 8 hours | Non-critical errors, cosmetic issues |

### 2.2 Incident Types

1. **Service Outage** - Agent completely unavailable
2. **Performance Degradation** - Slow response times or high latency
3. **Calculation Errors** - Incorrect or failed calculations
4. **Integration Failures** - SCADA/Historian connection issues
5. **Data Quality Issues** - Invalid or missing input data
6. **Security Incidents** - Unauthorized access or data breaches

---

## 3. Initial Response Checklist

### 3.1 Acknowledge and Assess

- [ ] Acknowledge the alert within SLA timeframe
- [ ] Join the incident Slack channel: `#incident-gl006`
- [ ] Assess initial impact and severity
- [ ] Update incident status page

### 3.2 Gather Information

```bash
# Check pod status
kubectl get pods -n greenlang -l app=gl-006-heatreclaim

# Check recent events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep gl-006

# View pod logs
kubectl logs -n greenlang -l app=gl-006-heatreclaim --tail=100

# Check metrics
kubectl port-forward svc/gl-006-heatreclaim-metrics 9090:9090 -n greenlang
# Then visit http://localhost:9090/metrics
```

### 3.3 Initial Communication

Send initial update to stakeholders:

```
INCIDENT DETECTED - GL-006 HeatRecoveryMaximizer

Time: [TIMESTAMP]
Severity: [SEV1/SEV2/SEV3/SEV4]
Status: Investigating
Impact: [Description of user impact]
Next Update: [Time + 30 minutes]

Incident Commander: [Your Name]
```

---

## 4. Response Procedures

### 4.1 SEV1 - Complete Service Outage

**Symptoms:**
- All pods in CrashLoopBackOff or Failed state
- No response from health endpoints
- 100% error rate on requests

**Response Steps:**

1. **Check Kubernetes cluster health**
   ```bash
   kubectl get nodes
   kubectl top nodes
   kubectl get cs
   ```

2. **Check deployment status**
   ```bash
   kubectl describe deployment gl-006-heatreclaim -n greenlang
   kubectl get rs -n greenlang -l app=gl-006-heatreclaim
   ```

3. **Check pod status and logs**
   ```bash
   kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o wide
   kubectl describe pod <pod-name> -n greenlang
   kubectl logs <pod-name> -n greenlang --previous
   ```

4. **Check dependencies**
   ```bash
   # PostgreSQL
   kubectl get pods -n greenlang -l app=postgresql

   # Redis
   kubectl get pods -n greenlang -l app=redis
   ```

5. **Attempt recovery**
   ```bash
   # Restart deployment
   kubectl rollout restart deployment gl-006-heatreclaim -n greenlang

   # If needed, scale down and up
   kubectl scale deployment gl-006-heatreclaim --replicas=0 -n greenlang
   kubectl scale deployment gl-006-heatreclaim --replicas=3 -n greenlang
   ```

6. **If restart fails, consider rollback**
   ```bash
   kubectl rollout undo deployment gl-006-heatreclaim -n greenlang
   ```

### 4.2 SEV2 - High Error Rate

**Symptoms:**
- Error rate > 5%
- Calculation failures > 50%
- Partial functionality

**Response Steps:**

1. **Identify error patterns**
   ```bash
   # Check error metrics
   curl -s localhost:9090/metrics | grep gl006_errors_total

   # Check logs for errors
   kubectl logs -n greenlang -l app=gl-006-heatreclaim | grep -i error | tail -50
   ```

2. **Check resource utilization**
   ```bash
   kubectl top pods -n greenlang -l app=gl-006-heatreclaim
   ```

3. **Check HPA status**
   ```bash
   kubectl get hpa gl-006-heatreclaim-hpa -n greenlang
   ```

4. **Scale if needed**
   ```bash
   kubectl scale deployment gl-006-heatreclaim --replicas=5 -n greenlang
   ```

### 4.3 SEV3 - Performance Degradation

**Symptoms:**
- P99 latency > 5 seconds
- Increased response times
- Some requests timing out

**Response Steps:**

1. **Check latency metrics**
   ```bash
   curl -s localhost:9090/metrics | grep gl006_http_request_duration
   ```

2. **Analyze slow queries**
   ```bash
   # Check database performance
   kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
   ```

3. **Check cache hit rate**
   ```bash
   kubectl exec -it redis-0 -n greenlang -- redis-cli INFO stats
   ```

4. **Consider scaling**
   ```bash
   kubectl scale deployment gl-006-heatreclaim --replicas=5 -n greenlang
   ```

### 4.4 Integration Failures

**Symptoms:**
- SCADA connection errors
- Historian data missing
- Integration timeout errors

**Response Steps:**

1. **Check integration metrics**
   ```bash
   curl -s localhost:9090/metrics | grep gl006_integration
   ```

2. **Verify connectivity**
   ```bash
   kubectl exec -it <pod-name> -n greenlang -- nc -zv scada-server 4840
   kubectl exec -it <pod-name> -n greenlang -- nc -zv historian 4840
   ```

3. **Check secrets**
   ```bash
   kubectl get secret gl-006-heatreclaim-secrets -n greenlang -o yaml
   ```

4. **Restart affected integrations**
   - Contact integration team if external systems are down
   - Enable fallback mode if available

---

## 5. Escalation Matrix

| Severity | Initial Responder | 15 min | 30 min | 1 hour |
|----------|-------------------|--------|--------|--------|
| SEV1 | On-Call Engineer | Team Lead | Engineering Manager | VP Engineering |
| SEV2 | On-Call Engineer | Team Lead | Engineering Manager | - |
| SEV3 | On-Call Engineer | Team Lead | - | - |
| SEV4 | On-Call Engineer | - | - | - |

### Escalation Contacts

| Role | Name | Phone | Slack |
|------|------|-------|-------|
| Primary On-Call | Rotating | PagerDuty | @platform-oncall |
| Team Lead | [Name] | [Phone] | @[handle] |
| Engineering Manager | [Name] | [Phone] | @[handle] |
| VP Engineering | [Name] | [Phone] | @[handle] |

---

## 6. Communication Templates

### 6.1 Status Update Template

```
GL-006 INCIDENT UPDATE

Time: [TIMESTAMP]
Status: [Investigating/Identified/Monitoring/Resolved]
Severity: [SEV1/SEV2/SEV3/SEV4]

Current State:
- [Description of current state]

Actions Taken:
- [Action 1]
- [Action 2]

Next Steps:
- [Next action]

ETA: [Estimated time to resolution]
Next Update: [Time]

Incident Commander: [Name]
```

### 6.2 Resolution Template

```
GL-006 INCIDENT RESOLVED

Time: [TIMESTAMP]
Duration: [Total incident duration]
Severity: [SEV1/SEV2/SEV3/SEV4]

Root Cause:
[Brief description of root cause]

Resolution:
[How the incident was resolved]

Impact:
- [Number of affected users/requests]
- [Duration of impact]

Follow-up Actions:
- [ ] Post-incident review scheduled
- [ ] RCA document created
- [ ] Prevention measures identified

Incident Commander: [Name]
```

---

## 7. Post-Incident Activities

### 7.1 Immediate (Within 24 hours)

- [ ] Update incident ticket with full timeline
- [ ] Collect all relevant logs and metrics
- [ ] Document workarounds applied
- [ ] Schedule post-incident review

### 7.2 Post-Incident Review (Within 5 business days)

- [ ] Conduct blameless post-mortem
- [ ] Document root cause analysis
- [ ] Identify prevention measures
- [ ] Create action items with owners and deadlines

### 7.3 Follow-up (Within 30 days)

- [ ] Complete all action items
- [ ] Update runbooks if needed
- [ ] Implement monitoring improvements
- [ ] Verify prevention measures are effective

---

## 8. Useful Commands Reference

```bash
# Quick health check
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o wide

# View logs
kubectl logs -f -n greenlang -l app=gl-006-heatreclaim --all-containers

# Check metrics
curl -s http://localhost:9090/metrics | grep gl006

# Restart deployment
kubectl rollout restart deployment gl-006-heatreclaim -n greenlang

# Rollback
kubectl rollout undo deployment gl-006-heatreclaim -n greenlang

# Scale
kubectl scale deployment gl-006-heatreclaim --replicas=N -n greenlang

# Check HPA
kubectl get hpa gl-006-heatreclaim-hpa -n greenlang -w

# Check PDB
kubectl get pdb gl-006-heatreclaim-pdb -n greenlang

# Port forward for debugging
kubectl port-forward svc/gl-006-heatreclaim 8000:80 -n greenlang
```

---

## 9. Related Documents

- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- [ROLLBACK_PROCEDURE.md](./ROLLBACK_PROCEDURE.md)
- [SCALING_GUIDE.md](./SCALING_GUIDE.md)
- [MAINTENANCE.md](./MAINTENANCE.md)

---

*This runbook is maintained by the Platform Team. For updates, contact platform-team@greenlang.io*
