# Incident Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to incidents in the VCCI Scope 3 Platform.

**On-Call**: #vcci-oncall | PagerDuty: vcci-production
**Escalation**: Platform Team → Engineering Manager → CTO

---

## Severity Classification

### P0 - Critical (Production Down)
- **Impact**: Complete service outage, data breach, security incident
- **Response Time**: Immediate (< 15 minutes)
- **Communication**: Hourly updates to leadership + customers
- **Examples**:
  - API returning 500 errors for all requests
  - Database unavailable
  - Active security breach
  - Data loss incident

### P1 - High (Major Degradation)
- **Impact**: Significant functionality unavailable, major performance degradation
- **Response Time**: 30 minutes
- **Communication**: Updates every 2 hours
- **Examples**:
  - Single tenant completely unable to access system
  - Critical feature (emissions calculation) not working
  - Database performance severely degraded

### P2 - Medium (Partial Degradation)
- **Impact**: Non-critical feature unavailable, minor performance issues
- **Response Time**: 2 hours
- **Communication**: Daily updates
- **Examples**:
  - Report generation failing
  - Email notifications delayed
  - Slow API response times (> 2s)

### P3 - Low (Minor Issue)
- **Impact**: Cosmetic issues, individual user problems
- **Response Time**: Next business day
- **Communication**: As needed
- **Examples**:
  - UI display issues
  - Single user cannot login
  - Documentation errors

---

## Incident Response Workflow

### Phase 1: Detection & Triage (0-5 minutes)

**1.1 Acknowledge the Alert**
```bash
# Acknowledge in PagerDuty
# Respond in Slack #vcci-incidents

# Check incident dashboard
kubectl get pods -n vcci-production
kubectl get nodes
kubectl top nodes
```

**1.2 Initial Assessment**
```bash
# Quick health check
./scripts/quick-health-check.sh

# Check recent deployments
kubectl rollout history deployment/vcci-api -n vcci-production

# Check error rates
kubectl logs -n vcci-production -l app=vcci-api --tail=100 | grep ERROR
```

**1.3 Determine Severity**
```
Questions to ask:
- Is the service completely down? → P0
- Are multiple tenants affected? → P0/P1
- Is this a security incident? → P0
- Is data at risk? → P0
- Is a single tenant affected? → P1/P2
- Is a non-critical feature down? → P2/P3
```

**1.4 Create Incident Record**
```bash
# Create incident in tracking system
./scripts/create-incident.sh \
  --severity P1 \
  --title "API returning 500 errors" \
  --oncall "$(whoami)"

# Incident ID will be returned (e.g., INC-20250106-001)
export INCIDENT_ID="INC-20250106-001"
```

### Phase 2: Communication (5-10 minutes)

**2.1 Notify Stakeholders**

**For P0/P1 Incidents:**
```bash
# Slack notification
/incident announce ${INCIDENT_ID} \
  "P1 Incident: API degradation affecting multiple tenants. \
   Investigating root cause. ETA for update: 30 minutes."

# Email notification template
To: platform-team@company.com, engineering@company.com
Cc: leadership@company.com
Subject: [P1] Production Incident - API Degradation

Incident ID: ${INCIDENT_ID}
Severity: P1
Status: Investigating
Affected: Multiple tenants experiencing slow API responses

Current Actions:
- Investigating database performance
- Checking for resource constraints
- Reviewing recent changes

Next Update: 30 minutes

Incident Commander: [Your Name]
```

**For P2/P3 Incidents:**
```bash
# Slack notification only
Post in #vcci-incidents channel
```

**2.2 Customer Communication (P0/P1 only)**
```bash
# Update status page
./scripts/update-status-page.sh \
  --status "Investigating" \
  --message "We are investigating API performance issues affecting some customers. \
             Our team is actively working on a resolution."
```

### Phase 3: Investigation (10-30 minutes)

**3.1 Gather Data**

**Check System Health**
```bash
# Node health
kubectl get nodes
kubectl describe nodes | grep -A 5 "Conditions"

# Pod health
kubectl get pods -n vcci-production
kubectl describe pods -n vcci-production | grep -A 10 "Events"

# Resource usage
kubectl top nodes
kubectl top pods -n vcci-production

# Check for OOMKilled pods
kubectl get pods -n vcci-production -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}' | grep OOMKilled
```

**Check Application Logs**
```bash
# Recent errors
kubectl logs -n vcci-production -l app=vcci-api --tail=500 --timestamps | grep ERROR

# Specific pod logs
kubectl logs -n vcci-production vcci-api-7d8f9c6b5d-abcde --tail=200

# Previous container logs (if pod restarted)
kubectl logs -n vcci-production vcci-api-7d8f9c6b5d-abcde --previous
```

**Check Database**
```bash
# Connection count
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"

# Long-running queries
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT pid, now() - pg_stat_activity.query_start AS duration, query
    FROM pg_stat_activity
    WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
    AND state = 'active';
  "

# Database locks
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT blocked_locks.pid AS blocked_pid,
           blocking_locks.pid AS blocking_pid,
           blocked_activity.query AS blocked_statement
    FROM pg_catalog.pg_locks blocked_locks
    JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
    JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
    JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
    WHERE NOT blocked_locks.granted;
  "

# Database disk space
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT pg_size_pretty(pg_database_size('vcci_production'));"
```

**Check Redis**
```bash
# Redis health
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u $REDIS_URL ping

# Redis memory
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u $REDIS_URL INFO memory

# Redis slow log
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u $REDIS_URL SLOWLOG GET 10
```

**Check External Dependencies**
```bash
# Check RabbitMQ
kubectl exec -n vcci-production deployment/vcci-api -- \
  curl -u guest:guest http://rabbitmq:15672/api/overview

# Check S3 access
aws s3 ls s3://vcci-production-data/ --region us-east-1

# Check external APIs (if applicable)
curl -I https://external-api.example.com/health
```

**3.2 Analyze Metrics**

```bash
# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Check dashboards:
# - Error rate (last 1 hour)
# - Response time (p95, p99)
# - CPU usage
# - Memory usage
# - Database connections
# - Cache hit rate

# Query Prometheus directly
kubectl exec -n monitoring prometheus-0 -- promtool query instant \
  'rate(http_requests_total{status=~"5.."}[5m])'
```

**3.3 Check Recent Changes**

```bash
# Recent deployments
kubectl rollout history deployment/vcci-api -n vcci-production

# Recent configuration changes
git log --since="24 hours ago" --oneline k8s/

# Recent database migrations
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py showmigrations --plan | tail -10

# Recent infrastructure changes
cd terraform/environments/production
git log --since="24 hours ago" --oneline
```

**3.4 Common Issue Decision Tree**

```
Start Here: What are you seeing?

├─ All API requests returning 500
│  ├─ Database connection refused → Check database health
│  ├─ Redis connection refused → Check Redis health
│  └─ Application error → Check application logs
│
├─ Some API requests slow (> 2s)
│  ├─ Database queries slow → Check slow queries, indexes
│  ├─ External API timeout → Check external dependencies
│  └─ High CPU/memory → Check resource usage
│
├─ Pods crashing (CrashLoopBackOff)
│  ├─ OOMKilled → Increase memory limits
│  ├─ Error in logs → Fix application bug
│  └─ Configuration error → Check ConfigMaps/Secrets
│
├─ Specific tenant affected
│  ├─ Data isolation issue → Check tenant schema
│  ├─ Resource quota exceeded → Check tenant limits
│  └─ Configuration issue → Check tenant settings
│
└─ Intermittent errors
   ├─ Network issues → Check security groups, DNS
   ├─ Resource constraints → Check autoscaling
   └─ Race conditions → Check application logic
```

### Phase 4: Mitigation (Variable)

**4.1 Emergency Actions**

**Rollback Recent Deployment**
```bash
# Check deployment history
kubectl rollout history deployment/vcci-api -n vcci-production

# Rollback to previous version
kubectl rollout undo deployment/vcci-api -n vcci-production

# Rollback to specific revision
kubectl rollout undo deployment/vcci-api -n vcci-production --to-revision=5

# Monitor rollback
kubectl rollout status deployment/vcci-api -n vcci-production

# Verify health
./scripts/health-check.sh
```

**Scale Up Resources**
```bash
# Scale up pods
kubectl scale deployment/vcci-api -n vcci-production --replicas=12

# Increase resource limits
kubectl set resources deployment/vcci-api -n vcci-production \
  --limits=cpu=4,memory=8Gi \
  --requests=cpu=2,memory=4Gi

# Scale up database
aws rds modify-db-instance \
  --db-instance-identifier vcci-production \
  --db-instance-class db.r5.4xlarge \
  --apply-immediately
```

**Restart Components**
```bash
# Restart application pods
kubectl rollout restart deployment/vcci-api -n vcci-production

# Delete problematic pod
kubectl delete pod -n vcci-production vcci-api-7d8f9c6b5d-abcde

# Restart all pods (rolling)
kubectl delete pods -n vcci-production -l app=vcci-api
```

**Clear Cache**
```bash
# Clear Redis cache
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u $REDIS_URL FLUSHDB

# Clear specific cache keys
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli -u $REDIS_URL DEL "cache:emissions:*"
```

**Database Emergency Actions**
```bash
# Kill long-running queries
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE state = 'active'
    AND (now() - pg_stat_activity.query_start) > interval '10 minutes';
  "

# Increase connection limit (temporary)
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "ALTER SYSTEM SET max_connections = 500;"

# Vacuum analyze (if bloat suspected)
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "VACUUM ANALYZE;"
```

**Traffic Management**
```bash
# Enable rate limiting
kubectl patch ingress vcci-api-ingress -n vcci-production --type=json -p='[
  {"op": "add", "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1limit-rps", "value": "50"}
]'

# Block specific IP
kubectl patch ingress vcci-api-ingress -n vcci-production --type=json -p='[
  {"op": "add", "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1whitelist-source-range", "value": "0.0.0.0/0"}
]'

# Enable maintenance mode
./scripts/enable-maintenance-mode.sh
```

**4.2 Common Incident Resolutions**

**Scenario 1: Database Connection Pool Exhausted**
```bash
# Symptoms: "Too many connections" errors

# Immediate fix: Increase pool size
kubectl set env deployment/vcci-api -n vcci-production \
  DATABASE_POOL_SIZE=50

# Kill idle connections
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE state = 'idle'
    AND (now() - state_change) > interval '5 minutes';
  "

# Long-term fix: Implement connection pooling with pgbouncer
```

**Scenario 2: Memory Leak**
```bash
# Symptoms: Pods being OOMKilled repeatedly

# Immediate fix: Increase memory limits
kubectl set resources deployment/vcci-api -n vcci-production \
  --limits=memory=16Gi --requests=memory=8Gi

# Restart pods to clear memory
kubectl rollout restart deployment/vcci-api -n vcci-production

# Enable memory profiling
kubectl set env deployment/vcci-api -n vcci-production \
  MEMORY_PROFILING_ENABLED=true

# Long-term fix: Identify and fix memory leak in code
```

**Scenario 3: Disk Space Full**
```bash
# Symptoms: "No space left on device" errors

# Check disk usage
kubectl exec -n vcci-production deployment/vcci-api -- df -h

# Clean up old logs
kubectl exec -n vcci-production deployment/vcci-api -- \
  find /var/log -type f -name "*.log" -mtime +7 -delete

# Clean up temporary files
kubectl exec -n vcci-production deployment/vcci-api -- \
  rm -rf /tmp/*

# Expand volume (if on cloud)
aws ec2 modify-volume --volume-id vol-xxxxx --size 200
```

**Scenario 4: Slow Database Queries**
```bash
# Symptoms: High database latency, slow API responses

# Identify slow queries
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT query, mean_exec_time, calls
    FROM pg_stat_statements
    ORDER BY mean_exec_time DESC
    LIMIT 10;
  "

# Create missing indexes (example)
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    CREATE INDEX CONCURRENTLY idx_emissions_tenant_date
    ON emissions(tenant_id, calculated_at);
  "

# Analyze tables
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "ANALYZE VERBOSE;"
```

**Scenario 5: External API Down**
```bash
# Symptoms: Timeouts calling external services

# Enable circuit breaker
kubectl set env deployment/vcci-api -n vcci-production \
  CIRCUIT_BREAKER_ENABLED=true

# Use cached data
kubectl set env deployment/vcci-api -n vcci-production \
  USE_CACHED_DATA=true

# Notify customers about limited functionality
./scripts/update-status-page.sh \
  --status "Degraded" \
  --message "External data provider unavailable. Some features limited."
```

### Phase 5: Monitoring & Updates (Ongoing)

**5.1 Continue Monitoring**
```bash
# Watch pod status
watch kubectl get pods -n vcci-production

# Monitor logs
kubectl logs -n vcci-production -l app=vcci-api -f | grep -i error

# Monitor metrics
# Keep Grafana dashboard open
# Watch error rates, response times, resource usage
```

**5.2 Provide Updates**

**Every 30 minutes for P0/P1:**
```bash
# Update incident record
./scripts/update-incident.sh ${INCIDENT_ID} \
  --status "Investigating" \
  --update "Identified root cause as database connection pool exhaustion. \
            Increased pool size. Monitoring for stability."

# Post to Slack
/incident update ${INCIDENT_ID} \
  "Update: Increased database connection pool. Error rate decreasing. \
   Continuing to monitor."

# Update status page (if customer-facing)
./scripts/update-status-page.sh \
  --status "Monitoring" \
  --message "Fix implemented. Monitoring for stability."
```

### Phase 6: Resolution (When stable for 30+ minutes)

**6.1 Verify Resolution**
```bash
# Verify all pods healthy
kubectl get pods -n vcci-production | grep -v Running

# Verify error rates back to normal
kubectl exec -n monitoring prometheus-0 -- promtool query instant \
  'rate(http_requests_total{status=~"5.."}[5m])'

# Verify response times normal
kubectl exec -n monitoring prometheus-0 -- promtool query instant \
  'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'

# Run smoke tests
./scripts/smoke-tests.sh
```

**6.2 Close Incident**
```bash
# Close incident record
./scripts/close-incident.sh ${INCIDENT_ID} \
  --resolution "Database connection pool exhausted due to traffic spike. \
                Increased pool size from 20 to 50. \
                Monitoring shows stability restored."

# Post final update
/incident resolve ${INCIDENT_ID} \
  "Resolved: Database connection pool increased. All systems operational."

# Update status page
./scripts/update-status-page.sh \
  --status "Operational" \
  --message "All systems operational. Incident resolved."

# Thank the team
Post in #vcci-incidents: "Thanks everyone for the quick response! 🎉"
```

### Phase 7: Post-Incident Review (Within 48 hours)

**7.1 Schedule Review Meeting**
```bash
# Create calendar invite
# Attendees: Incident responders, engineering manager, relevant stakeholders
# Duration: 60 minutes
# Agenda: Timeline, root cause, action items
```

**7.2 Document Incident**

**Post-Incident Report Template:**
```markdown
# Post-Incident Report: ${INCIDENT_ID}

## Incident Summary
- **Date**: 2025-01-06
- **Duration**: 45 minutes (14:30 - 15:15 UTC)
- **Severity**: P1
- **Impact**: ~30% of API requests failing, affecting 15 tenants
- **Root Cause**: Database connection pool exhaustion

## Timeline
- 14:30: Alert triggered - high error rate
- 14:32: Incident acknowledged, initial investigation
- 14:40: Root cause identified - connection pool exhausted
- 14:45: Mitigation applied - increased pool size to 50
- 14:55: Error rate returning to normal
- 15:15: Incident resolved - system stable

## Root Cause Analysis
### What Happened
- Traffic spike caused by marketing campaign
- Database connection pool (size: 20) exhausted
- New requests failed with "too many connections" error

### Why It Happened
- Connection pool size too small for traffic volume
- No alerting on connection pool utilization
- No autoscaling configured for connection pool

### Contributing Factors
- Marketing campaign not communicated to engineering
- Load testing scenarios didn't include connection pool stress
- Monitoring gaps - no connection pool metrics

## Impact Analysis
- **Customer Impact**: 15 tenants experienced intermittent errors (30% failure rate)
- **Revenue Impact**: Estimated $500 in SLA credits
- **Reputation Impact**: 3 support tickets filed

## What Went Well
- ✓ Quick detection (alert triggered within 1 minute)
- ✓ Fast initial response (acknowledged in 2 minutes)
- ✓ Effective communication (regular updates)
- ✓ Quick identification of root cause (10 minutes)
- ✓ Simple mitigation available

## What Didn't Go Well
- ✗ No proactive monitoring of connection pool
- ✗ No early warning of traffic spike
- ✗ Manual intervention required (not self-healing)
- ✗ No runbook for this scenario
- ✗ Communication gap with marketing team

## Action Items
1. **[P0] Add connection pool monitoring** (Owner: @alice, Due: 2025-01-08)
   - Add Prometheus metrics for connection pool utilization
   - Create alert when utilization > 80%

2. **[P0] Implement connection pool autoscaling** (Owner: @bob, Due: 2025-01-15)
   - Dynamically adjust pool size based on load
   - Set max pool size to 100

3. **[P1] Create runbook for connection pool issues** (Owner: @charlie, Due: 2025-01-10)
   - Document troubleshooting steps
   - Add to incident response runbook

4. **[P1] Improve cross-team communication** (Owner: @manager, Due: 2025-01-12)
   - Marketing must notify engineering of campaigns
   - Implement change advisory process

5. **[P2] Enhance load testing** (Owner: @bob, Due: 2025-01-31)
   - Include connection pool stress scenarios
   - Test with 3x expected traffic

## Lessons Learned
- Always monitor resource pools (connections, threads, memory)
- Cross-team communication is critical for reliability
- Autoscaling should extend to all resources, not just compute
- Load testing must cover all bottleneck scenarios

## Related Incidents
- INC-20241215-003: Similar issue with Redis connection pool
```

---

## Communication Templates

### Initial Notification (P0/P1)
```
🚨 [P1 INCIDENT] API Degradation

Incident ID: ${INCIDENT_ID}
Status: Investigating
Started: 14:30 UTC
Impact: Multiple customers experiencing errors

We are investigating reports of API errors affecting multiple customers.
Our team is actively working to identify and resolve the issue.

Next update: 30 minutes
Incident Commander: @oncall-engineer
```

### Update (Every 30 min)
```
📊 [P1 INCIDENT] Update #2

Incident ID: ${INCIDENT_ID}
Status: Mitigating
Elapsed: 30 minutes
Impact: Error rate decreasing

Update: We have identified the root cause as database connection pool exhaustion.
We have increased the connection pool size and are seeing error rates decrease.
Continuing to monitor for stability.

Next update: 30 minutes
```

### Resolution
```
✅ [P1 INCIDENT] Resolved

Incident ID: ${INCIDENT_ID}
Status: Resolved
Duration: 45 minutes
Resolution: Database connection pool increased

The incident has been resolved. All systems are operating normally.
We will conduct a post-incident review to prevent recurrence.

Thank you for your patience.
```

---

## Escalation Procedures

### When to Escalate

**Escalate to Engineering Manager if:**
- Incident is P0 and lasting > 30 minutes
- Incident is P1 and lasting > 2 hours
- Customer escalation received
- Need additional resources
- Unsure how to proceed

**Escalate to CTO if:**
- Incident is P0 and lasting > 1 hour
- Data breach or security incident
- Legal/compliance implications
- Executive customer impacted

### How to Escalate

```bash
# Page engineering manager
./scripts/page-manager.sh ${INCIDENT_ID}

# Call emergency hotline
Phone: +1-555-0123

# Slack escalation
@engineering-manager Need immediate assistance with ${INCIDENT_ID}

# PagerDuty escalation
Escalate incident in PagerDuty UI
```

---

## Quick Reference

### Emergency Contacts
```
Platform Team: #vcci-platform
Oncall: #vcci-oncall
Security: #security-incidents
Leadership: #exec-incidents

PagerDuty: vcci-production-oncall
Phone Hotline: +1-555-0123 (24/7)
```

### Common Commands
```bash
# Health check
./scripts/health-check.sh

# View logs
kubectl logs -n vcci-production -l app=vcci-api --tail=100

# Rollback deployment
kubectl rollout undo deployment/vcci-api -n vcci-production

# Scale up
kubectl scale deployment/vcci-api -n vcci-production --replicas=12

# Restart
kubectl rollout restart deployment/vcci-api -n vcci-production

# Enable maintenance mode
./scripts/enable-maintenance-mode.sh
```

### Useful Links
- Grafana: http://grafana.vcci-platform.com
- Kibana: http://kibana.vcci-platform.com
- PagerDuty: https://company.pagerduty.com
- Status Page: https://status.vcci-platform.com
- Runbooks: https://docs.vcci-platform.com/runbooks

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-06
**Maintained By**: Platform Engineering Team
