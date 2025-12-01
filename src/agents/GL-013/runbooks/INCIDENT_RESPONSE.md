# GL-013 PREDICTMAINT - Incident Response Runbook

```
================================================================================
                    INCIDENT RESPONSE RUNBOOK - GL-013 PREDICTMAINT
                         Standard Incident Handling Procedures
================================================================================
```

**Version:** 1.0.0
**Last Updated:** 2024-12-01
**Owner:** Site Reliability Engineering (SRE) Team
**On-Call Use:** Primary

---

## Table of Contents

1. [Overview](#overview)
2. [Incident Classification](#incident-classification)
3. [Initial Response](#initial-response)
4. [Incident Procedures](#incident-procedures)
5. [Communication Templates](#communication-templates)
6. [Post-Incident Process](#post-incident-process)

---

## Overview

This runbook provides procedures for responding to incidents affecting GL-013 PREDICTMAINT. Follow these steps in order during any incident.

### Scope

This runbook covers:
- Service outages
- Performance degradation
- Data quality issues
- Integration failures
- Security incidents

### Prerequisites

- Kubernetes cluster access (kubectl)
- Grafana dashboard access
- PagerDuty access
- Slack access (#gl-013-incidents)

---

## Incident Classification

### Severity Levels

| Severity | Impact | Response Time | Example |
|----------|--------|---------------|---------|
| **SEV1** | Complete service outage | 15 minutes | API returning 5xx for all requests |
| **SEV2** | Major functionality broken | 30 minutes | RUL predictions failing |
| **SEV3** | Minor functionality impacted | 4 hours | Single integration down |
| **SEV4** | Minimal impact | Next business day | UI rendering issues |

### Classification Criteria

**SEV1 - Critical:**
- API completely unavailable
- All predictions failing
- Data corruption detected
- Security breach confirmed

**SEV2 - High:**
- Prediction latency >10x normal
- >50% of requests failing
- Major integration down (SAP PM, Maximo)
- Authentication system failing

**SEV3 - Medium:**
- Single calculator failing
- Minor integration issues
- Performance degradation <50%
- Non-critical alerts failing

**SEV4 - Low:**
- Documentation errors
- UI cosmetic issues
- Non-blocking bugs
- Enhancement requests

---

## Initial Response

### Step 1: Acknowledge the Incident

```bash
# Acknowledge in PagerDuty within 5 minutes
# Then post in Slack
```

**Slack Message:**
```
@here GL-013 INCIDENT DECLARED
Severity: [SEV1/SEV2/SEV3/SEV4]
Summary: [Brief description]
Impact: [User impact description]
Status: Investigating
Incident Commander: [Your name]
```

### Step 2: Initial Assessment (5 minutes)

Run these commands to assess the situation:

```bash
# Check pod status
kubectl get pods -n greenlang -l app=gl-013-predictmaint

# Check pod events
kubectl describe pods -n greenlang -l app=gl-013-predictmaint | grep -A 20 Events

# Check recent logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100 --since=10m

# Check API health
curl -s https://api.greenlang.io/v1/gl-013/health | jq .

# Check metrics
curl -s http://gl-013-metrics:9090/metrics | grep -E "^predictmaint_(errors|latency)"
```

### Step 3: Check Dashboards

Open these Grafana dashboards:

1. **GL-013 Overview** - Overall service health
2. **GL-013 Performance** - Latency and throughput
3. **GL-013 Errors** - Error rates and types
4. **GL-013 Integrations** - Connector status

### Step 4: Identify Impact

Determine:
- Number of affected users
- Affected functionality
- Duration of impact
- Business impact

---

## Incident Procedures

### Procedure: Service Completely Down

**Symptoms:**
- Health endpoint returning non-200
- All API requests failing
- Pods in CrashLoopBackOff

**Steps:**

1. **Check Pod Status**
```bash
kubectl get pods -n greenlang -l app=gl-013-predictmaint -o wide
```

2. **Check Pod Logs**
```bash
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=200
```

3. **Check Resource Usage**
```bash
kubectl top pods -n greenlang -l app=gl-013-predictmaint
```

4. **Restart Pods (if needed)**
```bash
kubectl rollout restart deployment/gl-013-predictmaint -n greenlang
```

5. **Verify Recovery**
```bash
kubectl rollout status deployment/gl-013-predictmaint -n greenlang
curl -s https://api.greenlang.io/v1/gl-013/health
```

### Procedure: High Prediction Latency

**Symptoms:**
- P99 latency >2000ms
- Users reporting slow responses
- Timeout errors in logs

**Steps:**

1. **Check Current Latency**
```bash
curl -s http://gl-013-metrics:9090/metrics | grep prediction_latency
```

2. **Check Cache Hit Ratio**
```bash
curl -s http://gl-013-metrics:9090/metrics | grep cache_hit
```

3. **Check Redis Connection**
```bash
redis-cli -h redis.greenlang.io ping
redis-cli -h redis.greenlang.io info memory
```

4. **Check Database**
```bash
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT count(*) FROM pg_stat_activity;"
```

5. **Scale Up (if needed)**
```bash
kubectl scale deployment/gl-013-predictmaint -n greenlang --replicas=5
```

### Procedure: Integration Failure

**Symptoms:**
- CMMS connector errors
- Work orders not syncing
- Authentication failures

**Steps:**

1. **Check Integration Status**
```bash
curl -s http://gl-013-metrics:9090/metrics | grep integration_status
```

2. **Check Connector Logs**
```bash
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100 | grep -i "cmms\|maximo\|sap"
```

3. **Verify External Service**
```bash
curl -v https://sap.company.com/api/health
```

4. **Check Credentials**
```bash
kubectl get secret gl-013-predictmaint-secrets -n greenlang -o yaml
```

5. **Restart Connector**
```bash
# Connectors restart with pod
kubectl rollout restart deployment/gl-013-predictmaint -n greenlang
```

### Procedure: Database Issues

**Symptoms:**
- Database connection errors
- Query timeouts
- Data inconsistency

**Steps:**

1. **Check Database Connectivity**
```bash
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT 1;"
```

2. **Check Active Connections**
```bash
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"
```

3. **Check Long-Running Queries**
```bash
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT pid, now() - query_start AS duration, query FROM pg_stat_activity WHERE state = 'active' ORDER BY duration DESC LIMIT 5;"
```

4. **Check Replication Lag (if replica)**
```bash
psql -h postgres.greenlang.io -U readonly -d predictmaint -c "SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;"
```

---

## Communication Templates

### Initial Notification

```
INCIDENT: GL-013 PREDICTMAINT

Status: INVESTIGATING
Severity: [SEV1/SEV2/SEV3]
Start Time: [HH:MM UTC]

Summary:
[Brief description of the issue]

Impact:
[User impact description]

Current Actions:
- Investigating root cause
- Monitoring service metrics

Next Update: [Time + 30 minutes]
```

### Status Update

```
INCIDENT UPDATE: GL-013 PREDICTMAINT

Status: [INVESTIGATING/IDENTIFIED/MONITORING/RESOLVED]
Severity: [SEV1/SEV2/SEV3]
Duration: [X hours Y minutes]

Update:
[What has changed since last update]

Impact:
[Current user impact]

Actions Taken:
- [Action 1]
- [Action 2]

Next Steps:
- [Planned action 1]
- [Planned action 2]

Next Update: [Time]
```

### Resolution Notification

```
INCIDENT RESOLVED: GL-013 PREDICTMAINT

Status: RESOLVED
Severity: [SEV1/SEV2/SEV3]
Duration: [X hours Y minutes]

Resolution:
[What fixed the issue]

Root Cause:
[Brief root cause description]

Impact Summary:
- Duration: [X hours Y minutes]
- Users Affected: [Number/percentage]
- Requests Failed: [Number]

Follow-up:
- Post-mortem scheduled for [Date/Time]
- Action items will be tracked in [Ticket system]
```

---

## Post-Incident Process

### Immediate Actions (Within 24 Hours)

1. **Create Post-Mortem Document**
   - Use the post-mortem template
   - Include timeline, impact, root cause

2. **Update Incident Ticket**
   - Add resolution details
   - Link related tickets
   - Update severity if changed

3. **Notify Stakeholders**
   - Send resolution notification
   - Schedule post-mortem review

### Post-Mortem Meeting

**Agenda:**
1. Timeline review (10 min)
2. Root cause analysis (15 min)
3. What went well (5 min)
4. What could be improved (10 min)
5. Action items (10 min)

**Required Attendees:**
- Incident Commander
- Engineers who responded
- Service owner
- Product manager (SEV1/SEV2)

### Action Item Tracking

All action items must:
- Have an owner assigned
- Have a due date
- Be tracked in Jira
- Be reviewed weekly until closed

---

## Quick Reference

### Emergency Contacts

| Role | Contact | Method |
|------|---------|--------|
| On-Call Engineer | PagerDuty | Auto-page |
| Platform Lead | @platform-lead | Slack DM |
| Engineering Manager | @eng-manager | Slack DM |
| VP Engineering | @vp-eng | Phone (SEV1 only) |

### Useful Commands

```bash
# Get all pods status
kubectl get pods -n greenlang -l app=gl-013-predictmaint -o wide

# Get pod logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100

# Restart deployment
kubectl rollout restart deployment/gl-013-predictmaint -n greenlang

# Scale deployment
kubectl scale deployment/gl-013-predictmaint -n greenlang --replicas=5

# Check events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | tail -20

# Port forward for debugging
kubectl port-forward svc/gl-013-predictmaint -n greenlang 8000:8000
```

### Dashboard Links

- [GL-013 Overview](https://grafana.greenlang.io/d/gl-013-overview)
- [GL-013 Performance](https://grafana.greenlang.io/d/gl-013-performance)
- [GL-013 Errors](https://grafana.greenlang.io/d/gl-013-errors)
- [GL-013 Integrations](https://grafana.greenlang.io/d/gl-013-integrations)

---

```
================================================================================
                    GL-013 PREDICTMAINT - Incident Response Runbook
                         GreenLang Inc. - SRE Team
================================================================================
```
