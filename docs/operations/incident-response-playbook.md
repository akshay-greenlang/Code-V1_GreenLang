# GreenLang Incident Response Playbook

**Version:** 1.0
**Last Updated:** 2025-11-07
**Document Classification:** CRITICAL - Operations
**Review Cycle:** Quarterly
**Next Review:** 2026-02-07

---

## Executive Summary

This playbook provides standardized procedures for responding to production incidents in the GreenLang platform. It defines incident severity levels, response procedures, escalation paths, and post-incident review processes to ensure rapid and effective incident resolution.

**Key Objectives:**
- Minimize service disruption and customer impact
- Provide clear, actionable response procedures
- Ensure appropriate escalation and communication
- Learn from incidents through structured post-mortems

**Response Time Targets:**

| Severity | Initial Response | Status Update | Resolution Target |
|----------|------------------|---------------|-------------------|
| P0 - Critical | < 5 minutes | Every 15 min | < 4 hours |
| P1 - High | < 15 minutes | Every 30 min | < 8 hours |
| P2 - Medium | < 1 hour | Every 2 hours | < 24 hours |
| P3 - Low | < 4 hours | Daily | < 1 week |

---

## Table of Contents

1. [Incident Severity Levels](#incident-severity-levels)
2. [Incident Response Process](#incident-response-process)
3. [Common Incident Scenarios](#common-incident-scenarios)
4. [Escalation Procedures](#escalation-procedures)
5. [Communication Templates](#communication-templates)
6. [On-Call Procedures](#on-call-procedures)
7. [Post-Incident Review](#post-incident-review)

---

## Incident Severity Levels

### P0 - Critical

**Definition:** Complete service outage or critical security breach

**Impact:**
- Service completely unavailable to all users
- Data loss or corruption
- Critical security vulnerability being actively exploited
- Major data breach confirmed

**Examples:**
- API returning 100% errors
- Database completely unavailable
- All authentication failing
- Active security attack causing damage

**Response:**
- **Initial Response:** < 5 minutes
- **Page:** All hands on deck
- **Escalate To:** Incident Commander immediately, CTO within 30 minutes
- **Communication:** Status page update immediately, customer email within 1 hour
- **War Room:** Required

**SLA Impact:** Critical SLA breach, automatic customer credits

---

### P1 - High

**Definition:** Severe service degradation affecting majority of users

**Impact:**
- Service available but severely degraded (>20% error rate)
- Critical feature unavailable
- Performance severely degraded (p95 >5s)
- Security vulnerability identified (not actively exploited)

**Examples:**
- Error rate sustained >20%
- Database primary failed, running on replica
- Agent execution failing for 50%+ of requests
- Critical API endpoint unavailable

**Response:**
- **Initial Response:** < 15 minutes
- **Page:** On-call engineer + incident commander
- **Escalate To:** Engineering lead within 1 hour if not resolved
- **Communication:** Status page update within 30 minutes
- **War Room:** Recommended

**SLA Impact:** Major SLA breach, possible customer credits

---

### P2 - Medium

**Definition:** Partial service degradation affecting some users

**Impact:**
- Service available with moderate degradation (5-20% error rate)
- Non-critical feature unavailable
- Performance degraded (p95 1-5s)
- Minor security issue

**Examples:**
- Error rate 5-20%
- Specific agent type failing
- Monitoring dashboards unavailable
- Documentation site down

**Response:**
- **Initial Response:** < 1 hour
- **Page:** On-call engineer only
- **Escalate To:** Engineering lead if not resolved within 4 hours
- **Communication:** Status page update within 2 hours
- **War Room:** Not required

**SLA Impact:** Minor SLA impact

---

### P3 - Low

**Definition:** Minor issue with no immediate user impact

**Impact:**
- Service fully operational
- Minor inconvenience to users
- Cosmetic issues
- Future risk identified

**Examples:**
- Non-critical logs showing warnings
- Monitoring gaps identified
- Minor UI bug
- Deprecated dependency identified

**Response:**
- **Initial Response:** < 4 hours
- **Page:** Not required (ticket assigned)
- **Escalate To:** Not required
- **Communication:** No customer communication needed
- **War Room:** Not required

**SLA Impact:** No impact

---

## Incident Response Process

### Phase 1: Detection

**How Incidents Are Detected:**

1. **Automated Monitoring Alerts**
   ```bash
   # Alert fired from Prometheus/AlertManager
   # PagerDuty notification sent
   # Slack #alerts channel notification
   ```

2. **Customer Reports**
   ```bash
   # Support ticket created
   # Email to support@greenlang.io
   # Direct escalation from customer success
   ```

3. **Manual Discovery**
   ```bash
   # Engineer notices issue during normal work
   # Internal user reports problem
   ```

**Initial Actions (within 2 minutes):**
```bash
# Acknowledge alert
# PagerDuty: Press ACK button
# Slack: React with :eyes: emoji

# Create incident ticket
/incident create "Brief description"

# Start incident log
# Document: Detection time, alert source, initial symptoms
```

---

### Phase 2: Assessment

**Objective:** Determine severity and scope within 5 minutes

**Assessment Checklist:**

1. **Verify Issue is Real**
   ```bash
   # Check monitoring dashboards
   https://grafana.greenlang.io/d/system-overview

   # Test service manually
   curl -i https://api.greenlang.io/health
   curl -i https://api.greenlang.io/v1/agents

   # Check from multiple locations
   # Use external monitoring (Pingdom, etc.)
   ```

2. **Determine Scope**
   - [ ] What services are affected?
   - [ ] How many users impacted? (Check metrics)
   - [ ] What functionality is broken?
   - [ ] Is issue getting worse?

3. **Check Recent Changes**
   ```bash
   # Recent deployments
   kubectl rollout history deployment/greenlang-api

   # Recent configuration changes
   git log --since="2 hours ago" --oneline

   # Recent incidents
   grep "incident" /var/log/incidents/$(date +%Y-%m).log
   ```

4. **Assign Severity**
   - Use severity definitions above
   - Document reasoning
   - Can upgrade/downgrade as more information available

**Output:**
- Severity level assigned: P0/P1/P2/P3
- Affected services identified
- User impact estimated
- Initial hypothesis of root cause

---

### Phase 3: Response

**P0/P1 Incident Response:**

1. **Activate Incident Response Team** (within 5 minutes)
   ```bash
   # Page incident commander
   /page incident-commander "P0: [Brief description]"

   # Create war room
   # Zoom: https://greenlang.zoom.us/war-room
   # Slack: #incident-war-room

   # Post incident declaration
   /incident declare "P0: Complete service outage"
   ```

2. **Assign Roles**
   - **Incident Commander (IC):** Coordinates response, makes decisions
   - **Technical Lead:** Investigates root cause, implements fixes
   - **Communications Lead:** Handles customer communications
   - **Scribe:** Documents timeline and actions

3. **Initial Customer Communication** (within 15 minutes for P0, 30 min for P1)
   ```bash
   # Update status page
   curl -X POST https://status.greenlang.io/api/incidents \
     -H "Authorization: Bearer $STATUS_API_KEY" \
     -d '{
       "status": "investigating",
       "message": "We are investigating a service disruption affecting [description]",
       "impact": "major_outage"
     }'

   # Post to customer Slack channel
   # Send email to high-priority customers (Communications Lead)
   ```

4. **Investigate and Resolve**
   - Follow relevant scenario procedure (see Common Incident Scenarios below)
   - Technical Lead coordinates investigation
   - IC ensures work is progressing
   - Scribe documents all actions and findings

5. **Regular Status Updates**
   - **P0:** Every 15 minutes
   - **P1:** Every 30 minutes
   - **P2:** Every 2 hours

   ```bash
   # Update template
   Incident Update - [HH:MM]
   Status: [Investigating / Identified / Monitoring / Resolved]
   Current Actions: [What we're doing now]
   Next Steps: [What's coming next]
   ETA: [When we expect resolution or next update]
   ```

---

### Phase 4: Resolution

**When to Declare Resolved:**
- Root cause identified and fixed
- Service metrics returned to normal
- Validation tests passing
- No recurring symptoms for 30 minutes (P0) or 60 minutes (P1)

**Resolution Steps:**

1. **Verify Resolution** (10-30 minutes)
   ```bash
   # Run smoke tests
   pytest tests/smoke/ --env=production -v

   # Check key metrics
   # - Error rate < 1%
   # - p95 latency < 500ms
   # - Success rate > 99%

   # Monitor for regression
   watch curl https://api.greenlang.io/health
   ```

2. **Update Status Page** (within 5 minutes of resolution)
   ```bash
   curl -X PATCH https://status.greenlang.io/api/incidents/$INCIDENT_ID \
     -H "Authorization: Bearer $STATUS_API_KEY" \
     -d '{
       "status": "resolved",
       "message": "The issue has been resolved. All services are operating normally."
     }'
   ```

3. **Customer Communication**
   ```
   Subject: [RESOLVED] GreenLang Service Disruption

   Dear GreenLang Customer,

   The service disruption that began at [TIME] has been resolved as of [TIME].

   Issue: [Brief description]
   Duration: [X hours Y minutes]
   Impact: [What was affected]
   Root Cause: [Brief explanation]

   All services are now operating normally. We apologize for the inconvenience.

   A detailed post-incident report will be published within 5 business days.

   Best regards,
   GreenLang Operations Team
   ```

4. **Close Incident Ticket**
   ```bash
   /incident close "Resolved: [Brief description of fix]"
   ```

---

### Phase 5: Post-Incident Review

**Timeline:** Within 5 business days of resolution

**Required for:** All P0 and P1 incidents, optional for P2

See [Post-Incident Review](#post-incident-review) section below for full process.

---

## Common Incident Scenarios

### INC-001: High Error Rate (>1%)

**Symptoms:**
- Error rate alert firing
- Increased 5xx responses
- Customer reports of failures

**Initial Triage (5 minutes):**
```bash
# Check error rate
# Grafana: https://grafana.greenlang.io/d/errors
rate(gl_errors_total[5m]) / rate(gl_requests_total[5m])

# Check which endpoints affected
kubectl logs -l app=greenlang-api --tail=100 | grep ERROR

# Check recent deployments
kubectl rollout history deployment/greenlang-api
```

**Common Causes and Fixes:**

1. **Recent Bad Deployment**
   ```bash
   # Rollback immediately
   kubectl rollout undo deployment/greenlang-api

   # Monitor for improvement
   watch -n 5 'curl -s https://api.greenlang.io/metrics | grep gl_error_rate'

   # If improved: incident resolved
   # If not: continue investigation
   ```

2. **Database Connection Issues**
   ```bash
   # Check database health
   psql -h db.greenlang.io -c "SELECT 1;"

   # Check connection pool
   psql -h db.greenlang.io -c "SELECT count(*) FROM pg_stat_activity WHERE application_name='greenlang';"

   # If pool exhausted: scale up connection pool
   kubectl set env deployment/greenlang-api DB_POOL_SIZE=50

   # Or: scale up database replicas
   ```

3. **Downstream Service Failure**
   ```bash
   # Check external API status
   # - LLM providers: status.openai.com, status.anthropic.com
   # - Other integrations

   # If provider down: enable circuit breaker or switch provider
   kubectl set env deployment/greenlang-api ENABLE_CIRCUIT_BREAKER=true
   ```

4. **Resource Exhaustion**
   ```bash
   # Check resource usage
   kubectl top pods -l app=greenlang-api

   # If CPU/memory high: scale horizontally
   kubectl scale deployment/greenlang-api --replicas=10

   # If persistent: scale vertically (increase resource limits)
   ```

**Validation:**
```bash
# Error rate should drop to < 1%
# Monitor for 15 minutes to confirm stability
```

---

### INC-002: Performance Degradation (p95 >1s)

**Symptoms:**
- Latency alert firing
- Slow response times reported
- Timeout errors

**Initial Triage (5 minutes):**
```bash
# Check latency metrics
# Grafana: https://grafana.greenlang.io/d/performance
histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))

# Check which endpoints slow
kubectl logs -l app=greenlang-api --tail=100 | grep "duration_ms" | sort -k3 -n -r

# Check resource usage
kubectl top pods -l app=greenlang-api
kubectl top nodes
```

**Common Causes and Fixes:**

1. **Database Slow Queries**
   ```bash
   # Identify slow queries
   psql -h db.greenlang.io -c "
     SELECT pid, now() - query_start AS duration, query
     FROM pg_stat_activity
     WHERE state = 'active' AND now() - query_start > interval '5 seconds'
     ORDER BY duration DESC;
   "

   # Kill long-running queries if safe
   psql -h db.greenlang.io -c "SELECT pg_terminate_backend(PID);"

   # Add missing indexes if needed (non-production first)
   # Schedule full query optimization review
   ```

2. **LLM Provider Latency**
   ```bash
   # Check provider latency metrics
   # Grafana: LLM provider dashboard

   # If one provider slow: route to faster provider
   kubectl set env deployment/greenlang-api PREFER_PROVIDER=anthropic

   # Or: reduce timeout
   kubectl set env deployment/greenlang-api LLM_TIMEOUT_MS=15000
   ```

3. **Cold Start Issues**
   ```bash
   # Check if pods recently started
   kubectl get pods -l app=greenlang-api -o wide

   # Increase pod count to reduce load per pod
   kubectl scale deployment/greenlang-api --replicas=8

   # Warm up caches
   for i in {1..100}; do
     curl -s https://api.greenlang.io/v1/agents > /dev/null
   done
   ```

4. **Network Issues**
   ```bash
   # Check inter-service latency
   kubectl exec -it deploy/greenlang-api -- ping db.greenlang.io

   # Check external network
   kubectl exec -it deploy/greenlang-api -- curl -w "@curl-format.txt" -o /dev/null https://api.openai.com
   ```

**Validation:**
```bash
# p95 latency should drop to < 500ms
# p99 latency should be < 2s
# Monitor for 30 minutes to confirm
```

---

### INC-003: Resource Exhaustion

**Symptoms:**
- CPU usage >80%
- Memory usage >85%
- Disk space >90%
- OOMKilled pods

**Severity Assessment:**
- CPU >95% or Memory >95% = P0
- CPU >80% or Memory >85% = P1
- Disk >90% = P1

**Immediate Actions:**

1. **CPU Exhaustion**
   ```bash
   # Check CPU usage
   kubectl top pods --sort-by=cpu

   # Scale horizontally immediately
   kubectl scale deployment/greenlang-api --replicas=10

   # Identify CPU-heavy processes
   kubectl exec -it deploy/greenlang-api -- top

   # If single pod misbehaving: delete it
   kubectl delete pod greenlang-api-xxxxx

   # If widespread: may be caused by:
   # - Traffic spike (check request rate)
   # - Infinite loop (check logs for repeating patterns)
   # - Crypto mining malware (check processes)
   ```

2. **Memory Exhaustion**
   ```bash
   # Check memory usage
   kubectl top pods --sort-by=memory

   # Look for memory leaks
   kubectl logs -l app=greenlang-api | grep -i "memory\|oom"

   # Immediate mitigation: restart pods
   kubectl rollout restart deployment/greenlang-api

   # Scale up memory limits (temporary)
   kubectl set resources deployment/greenlang-api \
     --limits=memory=4Gi --requests=memory=2Gi

   # Long-term: investigate and fix memory leak
   ```

3. **Disk Space Exhaustion**
   ```bash
   # Check disk usage on nodes
   kubectl get nodes -o wide
   ssh node1 "df -h"

   # Common causes:
   # - Log files
   # - Docker images
   # - Temp files

   # Clean up logs
   ssh node1 "find /var/log -name '*.log' -mtime +7 -delete"

   # Clean up Docker images
   ssh node1 "docker system prune -af --volumes"

   # If database disk full:
   psql -h db.greenlang.io -c "VACUUM FULL;"

   # Emergency: expand disk
   aws ec2 modify-volume --volume-id vol-xxx --size 500
   ssh node1 "sudo resize2fs /dev/xvda1"
   ```

**Validation:**
```bash
# CPU usage < 70%
# Memory usage < 75%
# Disk usage < 80%
# No OOMKilled events
```

---

### INC-004: Agent Execution Failures

**Symptoms:**
- Agent execution error rate high
- Specific agent type failing
- Timeout errors

**Initial Triage:**
```bash
# Check agent metrics
# Grafana: https://grafana.greenlang.io/d/agents

# Which agents failing?
kubectl logs -l app=greenlang-agent --tail=200 | grep ERROR

# Recent agent pack updates?
git log --oneline --since="24 hours ago" packs/
```

**Common Causes and Fixes:**

1. **Agent Pack Configuration Error**
   ```bash
   # Validate agent pack
   gl agent validate packs/failing-agent.yml

   # Rollback to previous version
   git revert HEAD
   kubectl rollout restart deployment/greenlang-agent

   # Or: disable failing agent
   kubectl set env deployment/greenlang-agent DISABLE_AGENTS=failing-agent
   ```

2. **LLM API Errors**
   ```bash
   # Check LLM provider status
   curl https://status.openai.com/api/v2/status.json

   # Check API key validity
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

   # Rotate API key if compromised
   # Update secret
   kubectl create secret generic llm-credentials \
     --from-literal=openai-key=$NEW_KEY --dry-run=client -o yaml | kubectl apply -f -

   # Restart pods to pick up new key
   kubectl rollout restart deployment/greenlang-agent
   ```

3. **Agent Timeout Issues**
   ```bash
   # Check agent execution duration
   kubectl logs -l app=greenlang-agent | grep "duration_ms" | sort -k3 -n -r

   # Increase timeout
   kubectl set env deployment/greenlang-agent AGENT_TIMEOUT_MS=60000

   # Or: optimize agent logic
   ```

**Validation:**
```bash
# Agent success rate > 99%
# No timeout errors
# All agent types functioning
```

---

### INC-005: Database Issues

**Symptoms:**
- Database connection errors
- Slow queries
- Replication lag
- Disk full

**Initial Triage:**
```bash
# Check database health
psql -h db.greenlang.io -U admin -c "SELECT 1;"

# Check connections
psql -h db.greenlang.io -U admin -c "
  SELECT count(*), state FROM pg_stat_activity GROUP BY state;
"

# Check replication lag
psql -h db.greenlang.io -U admin -c "
  SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;
"
```

**Common Causes and Fixes:**

1. **Connection Pool Exhausted**
   ```bash
   # Check active connections
   psql -c "SELECT count(*) FROM pg_stat_activity;"

   # Increase connection limit (requires restart)
   psql -c "ALTER SYSTEM SET max_connections = 500;"
   systemctl restart postgresql

   # Or: kill idle connections
   psql -c "
     SELECT pg_terminate_backend(pid)
     FROM pg_stat_activity
     WHERE state = 'idle' AND state_change < now() - interval '5 minutes';
   "
   ```

2. **Replication Lag**
   ```bash
   # Check lag in seconds
   psql -h replica -c "
     SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;
   "

   # If lag > 300 seconds: investigate
   # - Network issues between primary and replica
   # - Heavy write load on primary
   # - Replica under-provisioned

   # Temporary: route reads to primary
   kubectl set env deployment/greenlang-api DB_READ_HOST=db-primary.greenlang.io

   # Long-term: scale replica or reduce write load
   ```

3. **Disk Full**
   ```bash
   # Check disk usage
   psql -c "
     SELECT pg_size_pretty(pg_database_size('greenlang')) AS db_size;
   "

   # Free up space
   psql -c "VACUUM FULL;"
   psql -c "REINDEX DATABASE greenlang;"

   # Delete old data (if applicable)
   psql -c "DELETE FROM logs WHERE created_at < now() - interval '90 days';"

   # Expand disk (emergency)
   aws rds modify-db-instance --db-instance-identifier greenlang-prod \
     --allocated-storage 1000 --apply-immediately
   ```

**Validation:**
```bash
# All connections succeeding
# Replication lag < 10 seconds
# Disk usage < 80%
# Query performance normal
```

---

### INC-006: External API Failures

**Symptoms:**
- LLM API calls failing
- Third-party integration errors
- Circuit breaker open

**Initial Triage:**
```bash
# Check external API status pages
curl https://status.openai.com/api/v2/status.json
curl https://status.anthropic.com/api/v2/status.json

# Check circuit breaker state
curl https://api.greenlang.io/internal/circuit-breakers

# Check error logs
kubectl logs -l app=greenlang-api | grep "external_api_error"
```

**Response:**

1. **Provider Outage**
   ```bash
   # If OpenAI down, switch to Anthropic
   kubectl set env deployment/greenlang-api PRIMARY_LLM_PROVIDER=anthropic

   # Enable circuit breaker for fast-fail
   kubectl set env deployment/greenlang-api CIRCUIT_BREAKER_ENABLED=true

   # Monitor provider status
   # Wait for recovery
   ```

2. **Rate Limiting**
   ```bash
   # Check rate limit headers in logs
   kubectl logs -l app=greenlang-api | grep "rate_limit"

   # Reduce request rate
   kubectl set env deployment/greenlang-api MAX_LLM_REQUESTS_PER_MINUTE=100

   # Or: upgrade API tier with provider
   ```

3. **API Key Issues**
   ```bash
   # Validate API key
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

   # If invalid: rotate key
   # See INC-004 for key rotation procedure
   ```

**Validation:**
```bash
# External API success rate > 99%
# No circuit breakers open
# Rate limits not exceeded
```

---

### INC-007: Security Incident

**Severity:** Always at least P1, likely P0

**Symptoms:**
- Unauthorized access detected
- Suspicious activity in logs
- Security scan findings
- Customer report of potential breach

**STOP: This is a security incident**
- Do NOT continue normal troubleshooting
- Do NOT modify systems without Security Lead approval
- Preserve evidence

**Immediate Actions:**

1. **Isolate Affected Systems** (within 5 minutes)
   ```bash
   # Isolate compromised pods (do not delete - preserve evidence)
   kubectl cordon node-with-compromised-pod
   kubectl taint node node-with-compromised-pod compromised=true:NoSchedule

   # Block suspicious IPs
   kubectl apply -f security/ip-blocklist.yaml

   # Revoke suspicious sessions
   # (Method depends on auth system)
   ```

2. **Activate Security Response Team**
   ```bash
   # Page Security Lead
   /page security-lead "P0 Security Incident: [Brief description]"

   # Page Legal (if data breach suspected)
   /page legal "Potential data breach"

   # Create separate secure Slack channel
   # #security-incident-YYYYMMDD
   ```

3. **Preserve Evidence**
   ```bash
   # Copy logs before they rotate
   kubectl logs compromised-pod > evidence/pod-logs-$(date +%s).log

   # Take disk snapshots
   aws ec2 create-snapshot --volume-id vol-xxx \
     --description "Evidence: Security incident $(date)"

   # Copy database state
   pg_dump -h db.greenlang.io -U admin greenlang \
     > evidence/db-snapshot-$(date +%s).sql
   ```

4. **Assess Impact**
   - What data accessed?
   - What systems compromised?
   - How did attacker gain access?
   - Are they still in the system?

5. **Containment**
   - Security Lead coordinates
   - May require taking systems offline
   - Do NOT notify attacker (no customer comms until contained)

6. **Forensics and Recovery**
   - Follow security incident runbook (separate doc)
   - May require external security firm
   - Rotate ALL credentials
   - Rebuild compromised systems from clean backups

**Customer Communication:**
- Delayed until Security Lead approves
- May require legal review
- Must comply with GDPR (72-hour notification)

**Validation:**
- Attacker access revoked
- All vulnerabilities patched
- Systems rebuilt and verified clean
- Credentials rotated

---

## Escalation Procedures

### Escalation Matrix

| Time Since Initial Response | P0 | P1 | P2 |
|-----------------------------|----|----|-----|
| **0 minutes** | On-call engineer + IC | On-call engineer | On-call engineer |
| **15 minutes** | Add: Engineering Lead | | |
| **30 minutes** | Add: CTO | Add: IC | |
| **60 minutes** | Add: CEO (if customer-facing) | Add: Engineering Lead | |
| **4 hours** | | Add: CTO | Add: Engineering Lead |

### Escalation Triggers

**Escalate to Incident Commander when:**
- Incident reaches P0 or P1 severity
- Root cause unknown after 30 minutes
- Multiple services affected
- Need to coordinate multiple teams

**Escalate to Engineering Lead when:**
- Not resolved within time targets
- Need architectural decisions
- Requires significant resources

**Escalate to CTO when:**
- P0 incident exceeds 1 hour
- Major customer impact
- Significant financial impact
- Media attention likely

**Escalate to CEO when:**
- Potential existential threat
- Major PR crisis
- Regulatory investigation
- Strategic decision needed

### How to Escalate

```bash
# PagerDuty escalation
/pd escalate incident-123 "Reason for escalation"

# Slack escalation
@engineering-lead Need immediate assistance on incident-123: [reason]

# Phone escalation (use published rotation)
# Call Engineering Lead: +1-XXX-XXX-XXXX
```

---

## Communication Templates

### Status Page Update Templates

**Investigating:**
```
We are investigating a service disruption affecting [component/feature].
Current impact: [description of what's not working]
We will provide an update within [timeframe].
```

**Identified:**
```
We have identified the root cause of the service disruption.
Issue: [brief technical description]
Current status: [what's working, what's not]
Next steps: [what we're doing to fix it]
ETA: [estimated resolution time]
```

**Monitoring:**
```
We have implemented a fix and are monitoring the service.
Issue: [what was wrong]
Fix: [what we did]
Current status: Service appears to be operating normally.
We will continue monitoring for [timeframe] before marking as resolved.
```

**Resolved:**
```
The service disruption has been resolved.
Issue: [what was wrong]
Resolution: [what we did]
Duration: [X hours Y minutes]
Impact: [summary of what was affected]
A detailed post-incident report will be published within 5 business days.
```

### Internal Update Template

```
Incident Update #X - [HH:MM UTC]

Status: [Investigating / Identified / Monitoring / Resolved]

Summary: [1-2 sentence current state]

Actions Completed:
- [Action 1]
- [Action 2]

Current Actions:
- [What we're doing right now]

Next Steps:
- [What's coming next]

Blockers:
- [None / List any blockers]

Metrics:
- Error Rate: [X%]
- Latency p95: [Xms]
- Success Rate: [X%]

ETA: [Next update or resolution time]

IC: [Name]
Tech Lead: [Name]
```

### Customer Email Templates

**Initial Notification:**
```
Subject: [ACTION REQUIRED] GreenLang Service Disruption

Dear GreenLang Customer,

We are currently experiencing a service disruption that may affect your use of GreenLang.

What's happening: [Brief description]
Impact: [What features are affected]
Started: [Time]
Current status: [What we're doing]

What you should do:
- [Any workarounds available]
- [What to avoid]

We are working urgently to resolve this issue. You can track real-time updates at:
https://status.greenlang.io

Next update: [Time]

We sincerely apologize for the inconvenience.

Best regards,
GreenLang Operations Team
```

**Resolution Notification:**
```
Subject: [RESOLVED] GreenLang Service Disruption

Dear GreenLang Customer,

The service disruption has been resolved and all systems are operating normally.

Summary:
- Issue: [What went wrong]
- Started: [Time]
- Resolved: [Time]
- Duration: [X hours Y minutes]
- Impact: [What was affected]

Root Cause: [Brief technical explanation]

What we're doing to prevent this:
- [Improvement 1]
- [Improvement 2]

A detailed post-incident report will be shared within 5 business days.

If you are still experiencing issues, please contact support@greenlang.io

We sincerely apologize for the disruption to your operations.

Best regards,
GreenLang Operations Team
```

---

## On-Call Procedures

### On-Call Rotation

**Schedule:** Weekly rotation, Monday 9am - Monday 9am

**Responsibilities:**
- Respond to all production alerts
- Initial incident triage
- Escalate as needed
- Document all incidents
- Handoff summary to next on-call

### On-Call Preparation

**Before Your Shift:**
- [ ] Test PagerDuty notifications
- [ ] Verify VPN access
- [ ] Test kubectl access to production
- [ ] Test database access
- [ ] Review recent incidents
- [ ] Check calendar for conflicts
- [ ] Identify backup coverage if needed

### During Your Shift

**When Alert Fires:**

1. **Acknowledge Immediately** (< 2 minutes)
   - PagerDuty or Slack
   - Prevents multiple people responding

2. **Initial Assessment** (< 5 minutes)
   - Is it a real incident?
   - What's the severity?
   - Can I handle it alone?

3. **Escalate if Needed**
   - Don't hesitate to call for help
   - Better to over-escalate than under-escalate

4. **Document Everything**
   - Incident log is critical
   - Future you will thank you

5. **Communicate Regularly**
   - Status updates per severity schedule
   - No surprises for stakeholders

### On-Call Handoff

**End of Shift:**
```
On-Call Handoff - [Date]
From: [Your Name]
To: [Next On-Call Name]

Open Incidents:
- [None / List with current status]

Ongoing Issues:
- [None / List with monitoring recommendations]

Recent Incidents:
- [Summary of past week]

Upcoming:
- [Any scheduled maintenance or known issues]

Notes:
- [Any other relevant information]
```

### On-Call Support

**Escalation Contacts:**
- Incident Commander: [Phone]
- Engineering Lead: [Phone]
- DBA: [Phone]
- Security Lead: [Phone]

**Resources:**
- Runbooks: docs/operations/
- Dashboards: https://grafana.greenlang.io
- Logs: https://grafana.greenlang.io/explore
- Status Pages: https://status.openai.com, etc.

---

## Post-Incident Review

**Required for:** All P0 and P1 incidents
**Optional for:** P2 incidents with lessons learned
**Timeline:** Within 5 business days of resolution

### Post-Incident Review Process

**Objective:** Learn from incidents and prevent recurrence

**NOT A BLAME SESSION:** Focus on systems, not people

#### 1. Schedule Review Meeting

**Attendees:**
- Incident Commander
- Technical responders
- Engineering Lead
- Any other stakeholders

**Duration:** 60 minutes

**Timing:** 2-5 days after incident (allows for reflection)

#### 2. Prepare Incident Timeline

```markdown
## Incident Timeline

| Time (UTC) | Event | Action Taken | By |
|------------|-------|--------------|-----|
| 14:32 | Error rate alert fired | Acknowledged | Alice |
| 14:35 | Identified database connection issue | Started investigation | Alice |
| 14:40 | Paged Incident Commander | Escalated to P1 | Alice |
| 14:45 | Updated status page | Customer communication | Bob |
| ... | ... | ... | ... |
```

#### 3. Root Cause Analysis

Use "5 Whys" technique:

```
Problem: API error rate reached 35%

Why? Database connections were being refused
Why? Connection pool was exhausted
Why? Sudden spike in traffic from one customer
Why? Customer deployed misconfigured integration
Why? We don't have per-customer rate limiting

Root Cause: Lack of per-customer rate limiting allowed
one customer to exhaust shared resources.
```

#### 4. Impact Analysis

```markdown
## Impact

**Duration:** 2 hours 15 minutes (14:32 - 16:47 UTC)

**Affected Services:**
- API: 35% error rate
- Agent executions: 25% failure rate
- Dashboard: Unavailable for 15 minutes

**Customer Impact:**
- Estimated 1,200 users affected
- Approximately 5,000 failed requests
- 3 customers contacted support

**Business Impact:**
- Revenue loss: ~$1,500 (estimated)
- SLA credits: $4,000
- Support burden: 8 hours

**Reputation Impact:**
- 15 negative tweets
- 2 news mentions
- Trust impact: Moderate
```

#### 5. What Went Well

```markdown
## What Went Well

- Alert fired immediately when issue started
- On-call engineer responded within 3 minutes
- Escalation process worked smoothly
- Communication was clear and timely
- Rollback procedure was effective
- Team collaboration was excellent
```

#### 6. What Went Wrong

```markdown
## What Went Wrong

- Lack of per-customer rate limiting
- Database connection pool too small for spike
- No automated mitigation for this scenario
- Delayed customer communication (should have been at 15 min, was at 25 min)
- Runbook for this scenario was outdated
```

#### 7. Action Items

```markdown
## Action Items

| Action | Owner | Priority | Due Date | Status |
|--------|-------|----------|----------|--------|
| Implement per-customer rate limiting | Alice | P0 | 2025-11-14 | In Progress |
| Increase DB connection pool size | Bob | P0 | 2025-11-10 | Done |
| Create automated mitigation for this scenario | Charlie | P1 | 2025-11-21 | Not Started |
| Update runbook with learnings | Alice | P2 | 2025-11-17 | Not Started |
| Review communication process | Comms | P2 | 2025-11-17 | Not Started |
```

#### 8. Publish Report

**Internal Report:** Full details, shared with all engineering

**Customer Report:** Executive summary, published to status page

**Example Customer Report:**
```markdown
# Post-Incident Report: Service Disruption on 2025-11-07

## Summary

On November 7, 2025, GreenLang experienced a service disruption lasting 2 hours and 15 minutes.
Approximately 1,200 users were affected, experiencing elevated error rates.

## Timeline

- **14:32 UTC:** Monitoring detected elevated error rate
- **14:35 UTC:** Incident response initiated
- **14:45 UTC:** Status page updated
- **15:30 UTC:** Root cause identified
- **16:15 UTC:** Fix implemented
- **16:47 UTC:** Service fully restored

## Root Cause

A sudden traffic spike from a single customer exhausted the shared database connection pool,
causing requests from all customers to fail. This was exacerbated by lack of per-customer
rate limiting.

## Resolution

We increased the database connection pool size and implemented emergency rate limiting.
The fix was validated and service was restored.

## Prevention

We are implementing the following improvements:
1. Per-customer rate limiting to prevent resource exhaustion
2. Increased database connection pool capacity
3. Automated detection and mitigation for similar scenarios
4. Enhanced monitoring for per-customer resource usage

## Affected Customers

If you were affected by this incident, SLA credits will be automatically applied to your account
within 5 business days.

We sincerely apologize for this disruption and the impact to your operations.
We are committed to preventing similar incidents in the future.

If you have any questions, please contact support@greenlang.io
```

---

## Appendix A: Quick Reference

### Severity Decision Tree

```
Is service completely unavailable to all users? → YES → P0
  ↓ NO
Is there a critical security breach? → YES → P0
  ↓ NO
Is error rate > 20% or critical feature down? → YES → P1
  ↓ NO
Is error rate 5-20% or moderate degradation? → YES → P2
  ↓ NO
Minor issue, no immediate user impact? → YES → P3
```

### Response Time Checklist

**P0 Incident:**
- [ ] Acknowledged within 5 minutes
- [ ] Incident Commander paged within 5 minutes
- [ ] Status page updated within 15 minutes
- [ ] Customer email sent within 60 minutes
- [ ] Updates every 15 minutes

**P1 Incident:**
- [ ] Acknowledged within 15 minutes
- [ ] Status page updated within 30 minutes
- [ ] Updates every 30 minutes

---

## Appendix B: Contact Information

### Internal Contacts

| Role | Primary | Phone | Email |
|------|---------|-------|-------|
| On-Call Engineer | [Rotation] | PagerDuty | oncall@greenlang.io |
| Incident Commander | [Name] | +1-XXX-XXX-XXXX | ic@greenlang.io |
| Engineering Lead | [Name] | +1-XXX-XXX-XXXX | eng-lead@greenlang.io |
| CTO | [Name] | +1-XXX-XXX-XXXX | cto@greenlang.io |
| Security Lead | [Name] | +1-XXX-XXX-XXXX | security@greenlang.io |
| DBA | [Name] | +1-XXX-XXX-XXXX | dba@greenlang.io |

### External Contacts

| Service | Contact | Account ID |
|---------|---------|------------|
| AWS Support | +1-XXX-XXX-XXXX | [Account] |
| OpenAI Support | support@openai.com | [Org ID] |
| Anthropic Support | support@anthropic.com | [Org ID] |
| PagerDuty Support | support@pagerduty.com | [Account] |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-07 | Operations Team | Initial version |

**Next Review Date:** 2026-02-07
**Approved By:** [CTO], [Operations Lead], [Security Lead]

---

**Remember:**
- Stay calm
- Follow procedures
- Communicate clearly
- Ask for help early
- Document everything
- Learn from every incident

**You've got this!**
