# GL-002 BoilerEfficiencyOptimizer - Incident Response Guide

Emergency procedures and incident response playbook for GL-002 BoilerEfficiencyOptimizer production operations.

## Table of Contents

1. [Incident Severity Definitions](#incident-severity-definitions)
2. [Response Procedures by Severity](#response-procedures-by-severity)
3. [Escalation Paths](#escalation-paths)
4. [Communication Templates](#communication-templates)
5. [Emergency Rollback Procedures](#emergency-rollback-procedures)
6. [Post-Incident Review](#post-incident-review)

---

## Incident Severity Definitions

### P0 - Critical (Production Down)

**Definition**: Complete service outage affecting all users and critical business operations.

**Examples**:
- All GL-002 pods are down (0/3 running)
- Database completely inaccessible
- Critical safety system failure
- Data loss or corruption
- Security breach detected

**Response Time**: Immediate (5 minutes)

**Escalation**: Automatic to Director of Engineering + CTO

**Communication**: Every 15 minutes until resolved

---

### P1 - High (Major Degradation)

**Definition**: Significant service degradation affecting majority of users or critical features.

**Examples**:
- 2/3 pods down (33% capacity)
- Error rate >20%
- Response time >10 seconds (p95)
- Integration failures with critical systems (SCADA)
- Memory leak causing OOMKilled
- Determinism failures in production calculations

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
- Non-critical integration failures (reporting APIs)
- Cache failures (Redis down, falling back to database)
- High CPU/memory usage (>80%)

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
# Post in #gl-002-incidents
```

**Slack Message:**
```
@here P0 INCIDENT: GL-002 BoilerEfficiencyOptimizer DOWN
Incident Commander: @your-name
Status: Investigating
Started: <timestamp>
Link: https://status.greenlang.io/incidents/<id>
```

#### Step 2: Assess (5-10 minutes)

```bash
# Check service status
kubectl get pods -n greenlang | grep gl-002

# Check recent deployments
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# Check error logs
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=200 --all-containers=true

# Check metrics
curl -s "http://prometheus:9090/api/v1/query?query=up{job='gl-002'}" | jq

# Check dependencies
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/integrations/health
```

**Assessment Checklist**:
- [ ] All pods down or all pods in CrashLoopBackOff?
- [ ] Recent deployment (last 2 hours)?
- [ ] Database accessible?
- [ ] Redis accessible?
- [ ] SCADA systems accessible?
- [ ] Network policies blocking traffic?
- [ ] Resource exhaustion on nodes?

#### Step 3: Immediate Mitigation (10-20 minutes)

**If recent deployment (< 2 hours ago):**
```bash
# ROLLBACK IMMEDIATELY - See ROLLBACK_PROCEDURE.md
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang --timeout=5m

# Verify rollback
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/health
```

**If database down:**
```bash
# Check database status
kubectl get pods -n greenlang | grep postgres

# Restart database (if in cluster)
kubectl rollout restart statefulset/postgresql -n greenlang

# Or restore from backup (AWS RDS)
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier gl-002-db-restored \
  --db-snapshot-identifier gl-002-snapshot-latest
```

**If all pods OOMKilled:**
```bash
# Immediately increase memory limits
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --limits=memory=4Gi

# Verify pods start
kubectl get pods -n greenlang -w
```

**If resource exhaustion:**
```bash
# Add more nodes (AWS EKS)
eksctl scale nodegroup --cluster=greenlang-cluster --nodes=8 --name=standard-workers

# Or reduce replica count temporarily
kubectl scale deployment gl-002-boiler-efficiency --replicas=1 -n greenlang
```

#### Step 4: Verify Recovery (20-30 minutes)

```bash
# Check pod health
kubectl get pods -n greenlang | grep gl-002
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/health

# Check error rate
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate

# Run smoke tests
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/test/smoke \
    -H "Content-Type: application/json" \
    -d '{"test_suite": "critical"}'

# Monitor for 15 minutes
kubectl logs -f -n greenlang deployment/gl-002-boiler-efficiency
```

**Recovery Checklist**:
- [ ] All pods running (3/3)
- [ ] Health endpoint returns 200 OK
- [ ] Error rate <1%
- [ ] Response time <2 seconds (p95)
- [ ] Integration tests passing
- [ ] No errors in logs

#### Step 5: Communicate Resolution

**Slack:**
```
@here P0 INCIDENT RESOLVED: GL-002 BoilerEfficiencyOptimizer RESTORED
Incident Commander: @your-name
Status: Resolved
Duration: <XX minutes>
Root Cause: <brief description>
Mitigation: <action taken>
Post-Mortem: Scheduled for <date/time>
```

**Status Page:**
```
RESOLVED: GL-002 Boiler Efficiency Optimizer service has been restored.
All systems are operating normally. We will conduct a post-incident
review to prevent future occurrences.
```

---

### P1 - High Severity Incident Response

#### Step 1: Acknowledge (0-15 minutes)

```bash
# Acknowledge in PagerDuty
# Post in #gl-002-incidents
```

**Slack Message:**
```
P1 INCIDENT: GL-002 Major Degradation
Owner: @your-name
Status: Investigating
Impact: <describe impact>
Started: <timestamp>
```

#### Step 2: Investigate (15-30 minutes)

```bash
# Check pod health
kubectl get pods -n greenlang | grep gl-002
kubectl describe pod -n greenlang <pod-name>

# Check logs
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=500

# Check metrics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(error|latency|cpu|memory)'

# Check recent changes
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang
```

#### Step 3: Mitigate (30-60 minutes)

**If pods down:**
```bash
# Restart affected pods
kubectl delete pod -n greenlang <pod-name>

# Or scale up
kubectl scale deployment gl-002-boiler-efficiency --replicas=5 -n greenlang
```

**If high error rate:**
```bash
# Check integration health
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/integrations/health | jq

# Disable failing integration temporarily
kubectl edit configmap gl-002-config -n greenlang
# Set: SCADA_INTEGRATION_ENABLED: "false"

# Restart
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

**If performance degradation:**
```bash
# Enable auto-scaling
kubectl autoscale deployment gl-002-boiler-efficiency -n greenlang \
  --min=3 --max=10 --cpu-percent=70

# Clear cache if corrupted
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  redis-cli -h redis-host FLUSHDB

# Restart pods
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

#### Step 4: Monitor (60+ minutes)

```bash
# Monitor recovery
kubectl get pods -n greenlang -w
kubectl top pods -n greenlang | grep gl-002

# Check error rate every 5 minutes
watch -n 300 'kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate'
```

#### Step 5: Communicate Resolution

**Slack:**
```
P1 INCIDENT RESOLVED: GL-002 Major Degradation
Owner: @your-name
Duration: <XX minutes>
Root Cause: <brief description>
Action Taken: <mitigation steps>
```

---

### P2 - Medium Severity Incident Response

#### Quick Response (1 hour)

```bash
# Assess impact
kubectl get pods -n greenlang | grep gl-002
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=200

# Apply quick fix
# Example: Restart single pod
kubectl delete pod -n greenlang <pod-name>

# Monitor
kubectl get pods -n greenlang -w
```

**Slack Update:**
```
P2 INCIDENT: GL-002 Minor Degradation
Owner: @your-name
Impact: <describe>
Action: <what you're doing>
ETA: <estimated resolution time>
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

**Subject:** [P0 CRITICAL] GL-002 BoilerEfficiencyOptimizer Production Outage

**Body:**
```
INCIDENT: P0 Critical - GL-002 Production Outage
STATUS: Investigating
STARTED: 2025-11-17 14:32 UTC
IMPACT: All users unable to access GL-002 services
INCIDENT COMMANDER: @engineer-name
WAR ROOM: #incident-2025-11-17-1432

CURRENT ACTIONS:
- Investigating root cause
- Checking recent deployments
- Verifying database connectivity
- Preparing rollback if needed

NEXT UPDATE: 15 minutes (14:47 UTC)

STATUS PAGE: https://status.greenlang.io/incidents/2025-11-17-1432
GRAFANA: https://grafana.greenlang.io/d/gl-002/incident-2025-11-17-1432
```

### P0 Update (Every 15 minutes)

**Subject:** [P0 CRITICAL] GL-002 Update #2 - Root Cause Identified

**Body:**
```
INCIDENT: P0 Critical - GL-002 Production Outage
STATUS: Mitigating
DURATION: 23 minutes
IMPACT: All users unable to access GL-002 services

ROOT CAUSE IDENTIFIED:
Recent deployment (v1.2.5) introduced database migration issue causing
connection failures. All pods unable to connect to PostgreSQL.

MITIGATION IN PROGRESS:
1. Rolling back deployment to v1.2.4 (ETA: 5 minutes)
2. Verifying database connectivity post-rollback
3. Running smoke tests

NEXT UPDATE: 15 minutes (14:57 UTC)

STATUS PAGE: https://status.greenlang.io/incidents/2025-11-17-1432
```

### P0 Resolution

**Subject:** [P0 RESOLVED] GL-002 BoilerEfficiencyOptimizer Restored

**Body:**
```
INCIDENT: P0 Critical - GL-002 Production Outage
STATUS: RESOLVED
STARTED: 2025-11-17 14:32 UTC
RESOLVED: 2025-11-17 14:58 UTC
DURATION: 26 minutes
INCIDENT COMMANDER: @engineer-name

ROOT CAUSE:
Database migration in v1.2.5 deployment failed, causing all pods to be
unable to connect to PostgreSQL database. Migration script had incorrect
timeout settings.

MITIGATION TAKEN:
1. Rolled back deployment to v1.2.4
2. Verified database connectivity
3. Ran full smoke test suite
4. Monitored for 15 minutes - all systems normal

CURRENT STATUS:
✅ All pods running (3/3)
✅ Health checks passing
✅ Error rate <0.1%
✅ Response time <1.5s (p95)
✅ All integrations healthy

PREVENTION:
- Added database migration timeout tests to CI pipeline
- Updated deployment procedure to require migration dry-run
- Scheduled post-mortem for 2025-11-18 10:00 UTC

POST-MORTEM: https://docs.greenlang.io/postmortems/2025-11-17-gl-002-outage
STATUS PAGE: https://status.greenlang.io/incidents/2025-11-17-1432
```

### P1 Notification

**Subject:** [P1 HIGH] GL-002 Major Degradation

**Body:**
```
INCIDENT: P1 High - GL-002 Major Degradation
STATUS: Investigating
STARTED: 2025-11-17 15:15 UTC
IMPACT: 2/3 pods down, service degraded for 33% of users
OWNER: @engineer-name

SYMPTOMS:
- 2 of 3 pods in CrashLoopBackOff
- Error rate: 28%
- Response time: 12s (p95)

ACTIONS:
- Investigating pod crash logs
- Checking resource constraints
- Preparing to scale up replicas

NEXT UPDATE: 30 minutes (15:45 UTC)
```

### Status Page Update Templates

**Investigating:**
```
Title: GL-002 Service Degradation
Status: Investigating

We are investigating reports of degraded performance in the GL-002
Boiler Efficiency Optimizer service. Some users may experience slow
response times or intermittent errors. Our team is actively working
to identify and resolve the issue.

Updated: 2025-11-17 14:32 UTC
```

**Identified:**
```
Title: GL-002 Service Degradation
Status: Identified

We have identified the root cause as a database connection issue
caused by a recent deployment. We are rolling back the deployment
and expect service to be restored within 15 minutes.

Updated: 2025-11-17 14:45 UTC
```

**Monitoring:**
```
Title: GL-002 Service Degradation
Status: Monitoring

The rollback has been completed and service is being restored. We are
monitoring the system to ensure stability before marking this incident
as resolved.

Updated: 2025-11-17 14:55 UTC
```

**Resolved:**
```
Title: GL-002 Service Degradation
Status: Resolved

This incident has been resolved. All systems are operating normally.
We apologize for any inconvenience caused.

Updated: 2025-11-17 14:58 UTC
```

---

## Emergency Rollback Procedures

### Quick Rollback (5 minutes)

```bash
# 1. Rollback to previous version
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang

# 2. Wait for rollout to complete
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang --timeout=5m

# 3. Verify health
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/health

# 4. Check logs for errors
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=50

# 5. Monitor error rate
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate
```

### Rollback to Specific Version

```bash
# 1. List revision history
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# 2. Rollback to specific revision
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang --to-revision=5

# 3. Verify
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang
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
- Optional: Product Manager, Customer Success (if customer-facing impact)

### Post-Mortem Template

**1. Incident Summary**
- Incident ID: INC-2025-11-17-001
- Severity: P0 Critical
- Duration: 26 minutes
- Impact: All users unable to access GL-002

**2. Timeline**
- 14:32 UTC: Deployment v1.2.5 completed
- 14:33 UTC: Monitoring alerts triggered (all pods down)
- 14:35 UTC: PagerDuty alert sent to on-call
- 14:37 UTC: Incident acknowledged, investigation started
- 14:45 UTC: Root cause identified (database migration failure)
- 14:47 UTC: Rollback initiated
- 14:52 UTC: Rollback completed, pods healthy
- 14:58 UTC: Incident resolved after monitoring period

**3. Root Cause Analysis**
- Database migration script in v1.2.5 had incorrect timeout (10s instead of 60s)
- Migration failed during deployment, leaving database in inconsistent state
- All new pods unable to connect to database
- Liveness probes failed, pods entered CrashLoopBackOff

**4. What Went Well**
- Monitoring detected issue within 1 minute of deployment
- On-call responded quickly (2 minutes to acknowledge)
- Rollback procedure worked as expected (5 minutes)
- Communication was clear and frequent

**5. What Went Wrong**
- Database migration timeout not tested in CI pipeline
- No smoke test after deployment (before routing traffic)
- No automated rollback trigger

**6. Action Items**
- [ ] Add database migration timeout tests to CI pipeline (Owner: @dev1, Due: 2025-11-20)
- [ ] Implement automated smoke tests post-deployment (Owner: @dev2, Due: 2025-11-22)
- [ ] Add automated rollback trigger for failed deployments (Owner: @sre1, Due: 2025-11-25)
- [ ] Update runbook with this scenario (Owner: @tech-writer, Due: 2025-11-18)

---

## Incident Response Checklist

### P0 Critical Checklist

- [ ] Acknowledge incident in PagerDuty (within 5 min)
- [ ] Post initial message in #gl-002-incidents
- [ ] Update status page (Investigating)
- [ ] Join war room (create #incident-<timestamp>)
- [ ] Assess impact and root cause (within 10 min)
- [ ] Implement mitigation (rollback or fix)
- [ ] Verify recovery (health checks, smoke tests)
- [ ] Update status page (Resolved)
- [ ] Post resolution message in Slack
- [ ] Schedule post-mortem within 24 hours
- [ ] Document lessons learned
- [ ] Update runbooks with new scenarios

### P1 High Checklist

- [ ] Acknowledge incident in PagerDuty (within 15 min)
- [ ] Post message in #gl-002-incidents
- [ ] Investigate root cause
- [ ] Implement mitigation
- [ ] Monitor recovery
- [ ] Post resolution message
- [ ] Create ticket for follow-up actions
- [ ] Update runbook if needed

### P2 Medium Checklist

- [ ] Acknowledge incident (within 1 hour)
- [ ] Post message in #gl-002-incidents
- [ ] Investigate and fix
- [ ] Monitor
- [ ] Post update when resolved
- [ ] Create ticket for root cause fix

---

## Contact Information

### On-Call Rotation

**Primary On-Call**: PagerDuty schedule "GL-002 Primary"
- Phone: See PagerDuty
- Slack: @gl-002-oncall-primary

**Secondary On-Call**: PagerDuty schedule "GL-002 Secondary"
- Phone: See PagerDuty
- Slack: @gl-002-oncall-secondary

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

### External Contacts

**AWS Support**:
- Phone: 1-800-123-4567
- Support Portal: https://console.aws.amazon.com/support
- Severity: Business-critical production system down

**Database Vendor (PostgreSQL)**:
- Support: support@postgresql-vendor.com
- SLA: 1-hour response for P0

**SCADA Vendor**:
- Phone: +1-555-0200
- Email: support@scada-vendor.com
- SLA: 2-hour response for P0

---

## Additional Resources

- **Troubleshooting Guide**: `TROUBLESHOOTING.md`
- **Rollback Procedure**: `ROLLBACK_PROCEDURE.md`
- **Scaling Guide**: `SCALING_GUIDE.md`
- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002\deployment\README.md`
- **Status Page**: https://status.greenlang.io
- **PagerDuty**: https://greenlang.pagerduty.com
- **Grafana Dashboards**: https://grafana.greenlang.io/d/gl-002
- **Post-Mortems Archive**: https://docs.greenlang.io/postmortems
