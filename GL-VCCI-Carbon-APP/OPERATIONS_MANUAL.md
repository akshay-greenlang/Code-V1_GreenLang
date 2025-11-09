# GL-VCCI Scope 3 Platform - Operations Manual

**Version**: 2.0.0
**Last Updated**: 2025-11-09
**Maintained By**: Platform Operations Team
**Classification**: Internal Use Only

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Daily Operations](#daily-operations)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Incident Response](#incident-response)
6. [Backup and Recovery](#backup-and-recovery)
7. [Performance Management](#performance-management)
8. [Security Operations](#security-operations)
9. [Capacity Planning](#capacity-planning)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Runbooks](#runbooks)
12. [Escalation Procedures](#escalation-procedures)
13. [Appendices](#appendices)

---

## Executive Summary

### Purpose
This Operations Manual provides comprehensive procedures for operating, maintaining, and troubleshooting the GL-VCCI Scope 3 Carbon Intelligence Platform in production environments.

### Key Metrics
```yaml
Platform Version: 2.0.0
Availability SLO: 99.9% (43.8 min/month downtime)
Performance SLO: p95 < 500ms, p99 < 1000ms
Error Rate SLO: < 0.1%
Recovery Time Objective (RTO): 4 hours
Recovery Point Objective (RPO): 15 minutes
```

### Critical Contacts
```yaml
Platform Team Lead: platform-lead@company.com | +1-555-0100
On-Call Engineer: Use PagerDuty (https://company.pagerduty.com)
Security Team: security@company.com | +1-555-0101
Engineering Manager: eng-manager@company.com | +1-555-0102
CTO (Escalation): cto@company.com | +1-555-0103
```

### On-Call Schedule
- **Primary On-Call**: 24/7 rotation (1-week shifts)
- **Secondary On-Call**: Backup coverage
- **Response SLA**: P0 (immediate), P1 (15 min), P2 (1 hour), P3 (next business day)
- **Handoff Time**: Mondays 9:00 AM EST

---

## System Overview

### Architecture Components

#### Application Layer
```yaml
API Servers:
  - Deployment: vcci-api
  - Replicas: 6-20 (auto-scaling)
  - Resource: 2-4 CPU, 4-8GB RAM per pod
  - Technology: FastAPI, Python 3.11

Worker Processes:
  - Deployment: vcci-worker
  - Replicas: 4-12 (auto-scaling)
  - Queues: emissions, reports, supplier_engagement, data_import
  - Technology: Celery, RabbitMQ

Agent Services:
  - Intake Agent: Data ingestion and validation
  - Calculator Agents: Category 1, 4, 6, 11, 15 emissions
  - Reporting Agent: Multi-standard report generation
  - Hotspot Agent: AI-powered analysis
```

#### Data Layer
```yaml
PostgreSQL:
  - Instance: db.r5.2xlarge (8 vCPU, 64GB RAM)
  - Read Replicas: 1
  - Storage: 500GB (auto-scaling to 2TB)
  - Backup: Daily automated + PITR (7 days)

Redis:
  - Instance: cache.r5.xlarge (4 vCPU, 26GB RAM)
  - Mode: Cluster (3 nodes)
  - Persistence: AOF + RDB snapshots
  - Usage: Session, cache, rate limiting

RabbitMQ:
  - Instance: mq.m5.large
  - Queues: 4 primary queues
  - Persistence: Disk-backed
```

#### Infrastructure
```yaml
Kubernetes:
  - Platform: AWS EKS 1.28
  - Nodes: 6 m5.2xlarge (8 vCPU, 32GB RAM each)
  - Node Pools: application, ml-workloads
  - Networking: VPC CNI, Calico network policies

Storage:
  - S3: vcci-production-data (versioned, encrypted)
  - EBS: Persistent volumes for databases
  - Lifecycle: 90 days → IA, 180 days → Glacier

Monitoring:
  - Prometheus: Metrics collection
  - Grafana: 7+ dashboards
  - ELK Stack: Log aggregation
  - PagerDuty: Alert routing
```

### Service Topology
```
Users (Web/API)
    ↓
NGINX Ingress (TLS Termination)
    ↓
API Servers (6-20 pods)
    ↓ ↓ ↓
    ├─> PostgreSQL (primary + read replica)
    ├─> Redis (cluster)
    ├─> RabbitMQ (message queue)
    ├─> S3 (file storage)
    └─> External APIs (Factor databases, LLMs)
```

---

## Daily Operations

### Morning Health Check (9:00 AM EST)

**Automated Daily Health Check Script**
Location: `/opt/vcci/scripts/daily-health-check.sh`

```bash
#!/bin/bash
# Daily Health Check - Run at 9:00 AM EST
set -euo pipefail

NAMESPACE="vcci-production"
API_URL="https://api.vcci-platform.com"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"

echo "=========================================="
echo "VCCI Platform Daily Health Check"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 1. Cluster Health
echo "1. Checking Kubernetes cluster..."
kubectl cluster-info
kubectl get nodes
kubectl top nodes

# 2. Pod Health
echo "2. Checking pod status..."
kubectl get pods -n $NAMESPACE
kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running

# 3. API Health
echo "3. Checking API endpoints..."
curl -f ${API_URL}/health/live || echo "❌ Liveness check failed"
curl -f ${API_URL}/health/ready || echo "❌ Readiness check failed"
curl -f ${API_URL}/health/metrics || echo "❌ Metrics endpoint failed"

# 4. Database Health
echo "4. Checking database..."
kubectl exec -n $NAMESPACE deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT COUNT(*) FROM pg_stat_activity;"

# 5. Redis Health
echo "5. Checking Redis..."
kubectl exec -n $NAMESPACE deployment/vcci-api -- \
  redis-cli -u $REDIS_URL PING

# 6. Queue Health
echo "6. Checking RabbitMQ..."
kubectl exec -n $NAMESPACE deployment/vcci-api -- \
  python -c "import pika; conn = pika.BlockingConnection(); print('✅ RabbitMQ OK')"

# 7. Storage Health
echo "7. Checking S3..."
aws s3 ls s3://vcci-production-data/ --summarize || echo "❌ S3 access failed"

# 8. Certificate Expiry
echo "8. Checking certificates..."
kubectl get certificate -n $NAMESPACE
openssl s_client -connect api.vcci-platform.com:443 -servername api.vcci-platform.com 2>/dev/null | \
  openssl x509 -noout -dates

# 9. Recent Errors
echo "9. Checking recent errors (last hour)..."
kubectl logs -n $NAMESPACE -l app=vcci-api --since=1h | grep -i error | tail -20

# 10. Summary
echo "=========================================="
echo "Health Check Complete"
echo "Time: $(date '+%H:%M:%S')"
echo "=========================================="
```

**Checklist**:
- [ ] All nodes are in Ready state
- [ ] All pods are Running (0 CrashLoopBackOff)
- [ ] API health endpoints return 200 OK
- [ ] Database connections < 80% of max
- [ ] Redis memory usage < 85%
- [ ] RabbitMQ queue depths normal (<1000 per queue)
- [ ] S3 accessible and within quota
- [ ] Certificates valid for >30 days
- [ ] No critical errors in last hour
- [ ] Prometheus targets all UP

### Weekly Operations (Every Monday)

**Weekly Tasks**
```bash
#!/bin/bash
# Weekly Operations - Run Monday 10:00 AM EST

# 1. Review metrics from past week
echo "1. Generating weekly metrics report..."
python /opt/vcci/scripts/weekly-report.py \
  --start-date "$(date -d '7 days ago' +%Y-%m-%d)" \
  --end-date "$(date +%Y-%m-%d)" \
  --output /tmp/weekly-report.pdf

# 2. Check database table sizes
echo "2. Analyzing database growth..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT
      schemaname,
      tablename,
      pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
    FROM pg_tables
    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    LIMIT 20;"

# 3. Review slow queries
echo "3. Analyzing slow queries..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT query, calls, mean_exec_time, max_exec_time
    FROM pg_stat_statements
    ORDER BY mean_exec_time DESC
    LIMIT 10;"

# 4. Clean up old logs
echo "4. Cleaning up old logs..."
find /var/log/vcci -name "*.log" -mtime +30 -delete

# 5. Update security patches
echo "5. Checking for security updates..."
kubectl get pods -n vcci-production -o json | \
  jq -r '.items[].spec.containers[].image' | \
  sort -u | \
  xargs -I {} trivy image {}
```

**Checklist**:
- [ ] Weekly metrics report generated and reviewed
- [ ] Database growth within expected range (<10% per week)
- [ ] Slow query report reviewed, optimizations planned
- [ ] Old logs cleaned up (>30 days removed)
- [ ] Security scans completed, critical vulns patched
- [ ] On-call handoff completed
- [ ] Incident post-mortems from past week reviewed
- [ ] Capacity planning updated if needed

### Monthly Operations (First Monday of Month)

**Monthly Tasks**:
- [ ] **Disaster Recovery Drill**: Test full backup restoration (estimated 2 hours)
- [ ] **Security Review**: Review access logs, user permissions, API keys
- [ ] **Performance Review**: Analyze trends, identify optimization opportunities
- [ ] **Cost Review**: Analyze cloud spend, identify cost optimization
- [ ] **Compliance Check**: Review audit logs, ensure compliance with policies
- [ ] **Documentation Review**: Update runbooks, operational procedures
- [ ] **Dependency Updates**: Review and plan updates for dependencies
- [ ] **Certificate Rotation**: Rotate long-lived certificates if needed

---

## Monitoring and Alerting

### Grafana Dashboards

**Dashboard 1: Platform Overview**
URL: `https://grafana.vcci-platform.com/d/platform-overview`

Key Metrics:
- Total Requests/sec
- Error Rate (%)
- API Response Time (p50, p95, p99)
- Active Connections
- Queue Depths
- Database Connections

**Dashboard 2: Application Performance**
URL: `https://grafana.vcci-platform.com/d/app-performance`

Key Metrics:
- Request Rate by Endpoint
- Response Time Heatmap
- Error Rate by Endpoint
- Cache Hit Rate
- Circuit Breaker Status
- Agent Performance (emissions calculations/sec)

**Dashboard 3: Infrastructure Health**
URL: `https://grafana.vcci-platform.com/d/infrastructure`

Key Metrics:
- Node CPU/Memory Utilization
- Pod CPU/Memory Usage
- Network Traffic (ingress/egress)
- Disk I/O
- Database IOPS
- Redis Memory Usage

**Dashboard 4: Business Metrics**
URL: `https://grafana.vcci-platform.com/d/business-metrics`

Key Metrics:
- Active Tenants
- Total Emissions Calculated
- Reports Generated
- Supplier Engagements
- API Calls by Tenant
- Data Quality Score Trends

**Dashboard 5: Security Monitoring**
URL: `https://grafana.vcci-platform.com/d/security`

Key Metrics:
- Failed Authentication Attempts
- API Rate Limit Violations
- Suspicious Activity Alerts
- Token Expiry Events
- WAF Block Rate
- Audit Log Volume

**Dashboard 6: SLO Tracking**
URL: `https://grafana.vcci-platform.com/d/slo-tracking`

Key Metrics:
- Availability (rolling 30-day)
- Error Budget Remaining
- SLO Compliance by Service
- Incident Impact on SLO
- Latency SLO Compliance

**Dashboard 7: Database Performance**
URL: `https://grafana.vcci-platform.com/d/database`

Key Metrics:
- Active Connections
- Query Performance (slow queries)
- Replication Lag
- Lock Wait Time
- Disk Usage
- Cache Hit Ratio

### Alert Rules

**Critical Alerts (P0 - Page Immediately)**

```yaml
- alert: APIHighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
  for: 2m
  annotations:
    summary: "API error rate above 1%"
    description: "Error rate is {{ $value }}% for {{ $labels.instance }}"
    runbook: "https://runbooks.vcci-platform.com/high-error-rate"
  labels:
    severity: critical
    team: platform

- alert: DatabaseDown
  expr: up{job="postgres"} == 0
  for: 1m
  annotations:
    summary: "PostgreSQL database is down"
    description: "Database {{ $labels.instance }} is unreachable"
    runbook: "https://runbooks.vcci-platform.com/database-down"
  labels:
    severity: critical
    team: platform

- alert: APIHighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
  for: 5m
  annotations:
    summary: "API p95 latency > 1 second"
    description: "p95 latency is {{ $value }}s for {{ $labels.endpoint }}"
    runbook: "https://runbooks.vcci-platform.com/high-latency"
  labels:
    severity: critical
    team: platform

- alert: PodCrashLooping
  expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
  for: 5m
  annotations:
    summary: "Pod is crash looping"
    description: "Pod {{ $labels.pod }} in {{ $labels.namespace }} is restarting"
    runbook: "https://runbooks.vcci-platform.com/pod-crash-loop"
  labels:
    severity: critical
    team: platform
```

**Warning Alerts (P1 - Investigate within 15 min)**

```yaml
- alert: HighMemoryUsage
  expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.85
  for: 10m
  annotations:
    summary: "Container memory usage above 85%"
    description: "{{ $labels.pod }} memory usage is {{ $value }}%"
  labels:
    severity: warning
    team: platform

- alert: DiskSpaceRunningLow
  expr: node_filesystem_avail_bytes / node_filesystem_size_bytes < 0.15
  for: 10m
  annotations:
    summary: "Disk space below 15%"
    description: "Node {{ $labels.node }} has {{ $value }}% free disk"
  labels:
    severity: warning
    team: platform

- alert: CacheHitRateLow
  expr: redis_cache_hit_rate < 0.80
  for: 15m
  annotations:
    summary: "Cache hit rate below 80%"
    description: "Redis cache hit rate is {{ $value }}%"
  labels:
    severity: warning
    team: platform

- alert: CircuitBreakerOpen
  expr: circuit_breaker_state{state="open"} > 0
  for: 5m
  annotations:
    summary: "Circuit breaker is open"
    description: "Circuit {{ $labels.circuit }} is open, fallback tier active"
  labels:
    severity: warning
    team: platform
```

**Info Alerts (P2 - Review within 1 hour)**

```yaml
- alert: HighQueueDepth
  expr: rabbitmq_queue_messages_ready > 1000
  for: 30m
  annotations:
    summary: "Queue depth above 1000"
    description: "Queue {{ $labels.queue }} has {{ $value }} messages"
  labels:
    severity: info
    team: platform

- alert: CertificateExpiringSoon
  expr: (cert_expiry_timestamp - time()) / 86400 < 30
  annotations:
    summary: "Certificate expiring in < 30 days"
    description: "Certificate {{ $labels.cn }} expires in {{ $value }} days"
  labels:
    severity: info
    team: platform
```

### Alert Response Procedures

**P0 - Critical (Immediate Response)**
1. Acknowledge alert in PagerDuty within 5 minutes
2. Join incident war room: Slack #vcci-incidents
3. Assess impact: Check dashboards, error rates, affected users
4. Engage secondary on-call if needed
5. Begin troubleshooting using runbooks
6. Update status page: https://status.vcci-platform.com
7. Escalate to manager if unresolved after 30 minutes
8. Document timeline and actions in incident report

**P1 - Warning (15-minute Response)**
1. Acknowledge alert in PagerDuty
2. Review alert details and dashboards
3. Determine if escalation to P0 needed
4. Investigate root cause
5. Apply fix or workaround
6. Monitor for recurrence
7. Create ticket for permanent fix if needed
8. Update runbooks if new pattern discovered

**P2 - Info (1-hour Response)**
1. Acknowledge alert
2. Review during business hours
3. Assess if trending toward P1/P0
4. Schedule fix during maintenance window
5. Update capacity planning if needed

---

## Incident Response

### Incident Severity Levels

```yaml
P0 - Critical:
  Definition: "Complete service outage or data loss"
  Examples:
    - API completely down (all endpoints return 5xx)
    - Database corruption or loss
    - Security breach detected
    - Data privacy violation
  Response: Immediate (page on-call)
  Communication: Update every 15 minutes
  Status Page: Update immediately

P1 - High:
  Definition: "Major functionality impaired, workaround exists"
  Examples:
    - Single agent service down (other agents working)
    - Elevated error rate (1-5%)
    - Performance degradation (p95 > 1s)
    - Authentication issues affecting subset of users
  Response: 15 minutes
  Communication: Update every 30 minutes
  Status Page: Update if >5% users affected

P2 - Medium:
  Definition: "Minor functionality impaired, low user impact"
  Examples:
    - Single feature not working
    - Slow performance for non-critical endpoints
    - Intermittent errors (<1%)
  Response: 1 hour
  Communication: Update every 2 hours
  Status Page: Optional

P3 - Low:
  Definition: "Cosmetic issue or enhancement request"
  Examples:
    - UI display issue
    - Documentation error
    - Feature request
  Response: Next business day
  Communication: Not required
  Status Page: Not applicable
```

### Incident Response Workflow

**1. Detection (Automated or Reported)**
- Alert fires in PagerDuty
- OR user reports issue via support
- OR monitoring detects anomaly

**2. Acknowledgment (< 5 min for P0)**
```bash
# Acknowledge in PagerDuty
# Post in Slack #vcci-incidents
Incident: [P0/P1/P2] <Short Description>
Assigned To: @on-call-engineer
Status: Investigating
Time Detected: <timestamp>
```

**3. Assessment (< 10 min for P0)**
- Check Grafana dashboards: https://grafana.vcci-platform.com
- Review recent deployments: `kubectl rollout history -n vcci-production`
- Check logs: `kubectl logs -n vcci-production -l app=vcci-api --tail=200`
- Assess user impact: Query metrics for affected endpoints/tenants
- Determine severity level

**4. Communication**
```bash
# Update status page
curl -X POST https://api.statuspage.io/v1/incidents \
  -d "name=API Performance Degradation" \
  -d "status=investigating" \
  -d "impact=major"

# Update Slack
Incident Update:
Status: Investigating
Impact: <describe>
ETA: <estimated resolution time>
Next Update: <15/30/60 min>
```

**5. Mitigation**
- Apply immediate fix using runbook procedures
- If unclear, rollback recent changes
- Scale resources if capacity issue
- Enable circuit breakers if external dependency failing
- Implement workaround to restore service

**6. Resolution**
- Verify fix: Monitor dashboards for 15 minutes
- Confirm user reports resolved
- Update status page to "Resolved"
- Document timeline and resolution

**7. Post-Incident Review (within 48 hours)**
```markdown
## Post-Incident Review: <Incident Name>

### Incident Summary
- Date/Time: <datetime>
- Duration: <total time>
- Severity: P0/P1/P2
- Impact: <users affected, revenue impact>

### Timeline
- HH:MM - Incident began
- HH:MM - Alert fired
- HH:MM - Engineer acknowledged
- HH:MM - Root cause identified
- HH:MM - Fix applied
- HH:MM - Incident resolved

### Root Cause
<Detailed explanation>

### Resolution
<What fixed it>

### Action Items
- [ ] Prevent recurrence: <action>
- [ ] Improve detection: <action>
- [ ] Update runbooks: <action>
- [ ] Training needed: <action>

### What Went Well
- <positive observations>

### What Could Be Improved
- <areas for improvement>
```

---

## Backup and Recovery

### Backup Strategy

**Database Backups**
```yaml
PostgreSQL:
  Automated Daily Backup:
    Schedule: "03:00 AM EST daily"
    Retention: 30 days
    Location: s3://vcci-backups/postgres/
    Type: Full backup

  Point-in-Time Recovery (PITR):
    Enabled: Yes
    Retention: 7 days
    WAL Archiving: s3://vcci-backups/postgres-wal/

  Manual Backup:
    Command: |
      kubectl exec -n vcci-production deployment/vcci-api -- \
        pg_dump $DATABASE_URL | \
        gzip > backup-$(date +%Y%m%d-%H%M%S).sql.gz
```

**Redis Backups**
```yaml
Redis:
  Snapshot Backups:
    Schedule: Every 6 hours
    Retention: 7 days
    Location: /data/redis/backups/

  AOF (Append-Only File):
    Enabled: Yes
    Fsync: everysec
    Rewrite: automatic
```

**Application State**
```yaml
Kubernetes:
  Manifests:
    Repository: git@github.com:company/vcci-k8s-manifests.git
    Backup: Continuous (Git history)

  Secrets:
    Backup: AWS Secrets Manager (encrypted)
    Rotation: 90 days

  ConfigMaps:
    Backup: Git repository (version controlled)
```

**File Storage**
```yaml
S3 Backups:
  Versioning: Enabled
  Retention:
    - Current versions: Indefinite
    - Non-current versions: 30 days → Glacier
  Cross-Region Replication: us-west-2 (DR region)
  Backup: Continuous (S3 versioning)
```

### Recovery Procedures

**Database Recovery (RTO: 4 hours, RPO: 15 minutes)**

**Scenario 1: Restore to Specific Point in Time**
```bash
#!/bin/bash
# Restore database to specific timestamp

TARGET_TIME="2025-11-09 14:30:00+00"
BACKUP_IDENTIFIER="vcci-production-$(date +%Y%m%d-%H%M%S)"

# 1. Create restore point
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier vcci-production \
  --target-db-instance-identifier vcci-production-restore \
  --restore-time "$TARGET_TIME"

# 2. Wait for restore to complete (15-30 min)
aws rds wait db-instance-available \
  --db-instance-identifier vcci-production-restore

# 3. Get new endpoint
NEW_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier vcci-production-restore \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

# 4. Update Kubernetes secret
kubectl create secret generic database-credentials \
  --from-literal=host=$NEW_ENDPOINT \
  --from-literal=port=5432 \
  --from-literal=database=vcci_production \
  --from-literal=username=vcci_admin \
  --from-literal=password=$DB_PASSWORD \
  --dry-run=client -o yaml | \
  kubectl apply -f -

# 5. Restart application pods
kubectl rollout restart deployment/vcci-api -n vcci-production
kubectl rollout status deployment/vcci-api -n vcci-production

# 6. Verify recovery
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT COUNT(*) FROM pg_stat_activity;"

echo "Recovery complete. New endpoint: $NEW_ENDPOINT"
```

**Scenario 2: Restore from Backup File**
```bash
#!/bin/bash
# Restore from specific backup file

BACKUP_FILE=$1
DB_HOST=$(kubectl get secret database-credentials -n vcci-production -o jsonpath='{.data.host}' | base64 -d)

# 1. Download backup from S3
aws s3 cp s3://vcci-backups/postgres/$BACKUP_FILE backup.sql.gz

# 2. Decompress
gunzip backup.sql.gz

# 3. Stop application (prevent writes)
kubectl scale deployment/vcci-api -n vcci-production --replicas=0

# 4. Restore database
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL < backup.sql

# 5. Verify restoration
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT MAX(created_at) FROM tenants;"

# 6. Restart application
kubectl scale deployment/vcci-api -n vcci-production --replicas=6

echo "Restoration complete from $BACKUP_FILE"
```

**Application Recovery (RTO: 30 minutes)**

**Scenario: Rollback to Previous Version**
```bash
#!/bin/bash
# Rollback application to previous stable version

PREVIOUS_VERSION=${1:-v1.9.5}

# 1. Get current version
CURRENT_VERSION=$(kubectl get deployment/vcci-api -n vcci-production \
  -o jsonpath='{.spec.template.spec.containers[0].image}' | \
  cut -d: -f2)

echo "Current version: $CURRENT_VERSION"
echo "Rolling back to: $PREVIOUS_VERSION"

# 2. Update deployment
kubectl set image deployment/vcci-api -n vcci-production \
  api=ghcr.io/company/vcci-platform:$PREVIOUS_VERSION

# 3. Monitor rollout
kubectl rollout status deployment/vcci-api -n vcci-production

# 4. Verify health
sleep 30
kubectl exec -n vcci-production deployment/vcci-api -- \
  curl -f http://localhost:8000/health/ready

# 5. Run smoke tests
./scripts/smoke-tests.sh

echo "Rollback to $PREVIOUS_VERSION complete"
```

**Disaster Recovery (Full Region Failure)**

**DR Activation Checklist**:
```markdown
## Disaster Recovery Activation

### Prerequisites
- [ ] Regional outage confirmed (>1 hour expected)
- [ ] DR region (us-west-2) verified healthy
- [ ] Executive approval obtained
- [ ] All stakeholders notified

### DNS Failover (Estimated: 15 minutes)
- [ ] Update Route53 to point to DR region
- [ ] Verify DNS propagation
- [ ] Test access from multiple locations

### Database Failover (Estimated: 30 minutes)
- [ ] Promote read replica in DR region to primary
- [ ] Verify replication lag is acceptable (<5 min)
- [ ] Update connection strings

### Application Activation (Estimated: 45 minutes)
- [ ] Scale up DR environment to production capacity
- [ ] Verify all services healthy
- [ ] Run full smoke test suite
- [ ] Monitor error rates and latency

### Validation (Estimated: 30 minutes)
- [ ] End-to-end test successful
- [ ] Customer-facing validation
- [ ] Monitor for 30 minutes for stability

### Communication
- [ ] Status page updated
- [ ] Customer email sent
- [ ] Team notification sent
- [ ] Incident channel active

### Total RTO: 2 hours
```

---

## Performance Management

### Performance Baselines

```yaml
Expected Performance (Normal Load):
  Requests/Second: 2,000-3,000 req/s
  API Latency:
    p50: <200ms
    p95: <500ms
    p99: <1000ms
  Cache Hit Rate: >85%
  Database Query Time: p95 <100ms
  Error Rate: <0.1%

Peak Performance (High Load):
  Requests/Second: 5,000-5,200 req/s
  API Latency:
    p50: <250ms
    p95: <600ms
    p99: <1200ms
  Cache Hit Rate: >80%
  Database Query Time: p95 <150ms
  Error Rate: <0.5%

Resource Utilization:
  CPU: 50-70% (normal), 70-85% (peak)
  Memory: 60-75% (normal), 75-85% (peak)
  Disk I/O: <70% capacity
  Network: <60% capacity
```

### Performance Tuning

**Database Optimization**
```sql
-- Check slow queries (run weekly)
SELECT
  query,
  calls,
  total_exec_time,
  mean_exec_time,
  max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 100  -- queries >100ms
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Check missing indexes
SELECT
  schemaname,
  tablename,
  attname,
  n_distinct,
  correlation
FROM pg_stats
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  AND n_distinct > 100
  AND correlation < 0.1
ORDER BY n_distinct DESC;

-- Vacuum and analyze (automated, but can run manually)
VACUUM ANALYZE;
```

**Cache Optimization**
```bash
# Check cache hit rates
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli INFO stats | grep keyspace_hits

# Analyze cache key distribution
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli --bigkeys

# Check memory usage
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli INFO memory
```

**Application Tuning**
```yaml
# Recommended settings for high performance

API Server:
  workers: 4 per pod
  worker_connections: 1000
  keepalive_timeout: 65
  request_timeout: 300
  max_request_size: 100MB

Database Connection Pool:
  pool_size: 20
  max_overflow: 10
  pool_recycle: 3600
  pool_pre_ping: true

Redis:
  max_connections: 50
  socket_timeout: 5
  socket_keepalive: true

Celery Workers:
  concurrency: 4
  max_tasks_per_child: 1000
  task_acks_late: true
  worker_prefetch_multiplier: 2
```

---

## Security Operations

### Security Monitoring

**Daily Security Checks**
```bash
#!/bin/bash
# Daily security audit

# 1. Check failed login attempts (last 24h)
kubectl logs -n vcci-production -l app=vcci-api --since=24h | \
  grep "authentication failed" | wc -l

# 2. Check rate limit violations
kubectl logs -n vcci-production -l app=vcci-api --since=24h | \
  grep "rate limit exceeded" | wc -l

# 3. Check suspicious API access patterns
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT
      ip_address,
      COUNT(*) as request_count,
      COUNT(DISTINCT endpoint) as endpoints_accessed
    FROM audit_logs
    WHERE created_at > NOW() - INTERVAL '24 hours'
    GROUP BY ip_address
    HAVING COUNT(*) > 10000
    ORDER BY request_count DESC;"

# 4. Check for security header violations
curl -I https://api.vcci-platform.com | \
  grep -E "(Strict-Transport-Security|X-Frame-Options|X-Content-Type-Options)"

# 5. Check for expiring certificates
kubectl get certificates -n vcci-production -o json | \
  jq '.items[] | select(.status.notAfter | fromdateiso8601 < (now + (30*86400)))'
```

### Audit Log Review

**Weekly Audit Log Analysis**
```sql
-- Unusual access patterns
SELECT
  user_email,
  action,
  resource,
  COUNT(*) as frequency
FROM audit_logs
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY user_email, action, resource
HAVING COUNT(*) > 1000
ORDER BY frequency DESC;

-- Access outside business hours
SELECT
  user_email,
  action,
  resource,
  created_at
FROM audit_logs
WHERE created_at > NOW() - INTERVAL '7 days'
  AND EXTRACT(hour FROM created_at) NOT BETWEEN 8 AND 18
  AND action IN ('DELETE', 'UPDATE', 'EXPORT')
ORDER BY created_at DESC;

-- Data export activities
SELECT
  user_email,
  resource,
  metadata->>'record_count' as records_exported,
  created_at
FROM audit_logs
WHERE action = 'EXPORT'
  AND created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
```

### Security Incident Response

**Suspected Security Breach**
```markdown
## Security Incident Response Checklist

### Immediate Actions (< 15 min)
- [ ] Assess scope: What data/systems affected?
- [ ] Contain: Isolate affected systems
- [ ] Notify: Security team, legal, executive team
- [ ] Preserve: Take snapshots, collect logs
- [ ] Document: Start incident timeline

### Investigation (< 2 hours)
- [ ] Review audit logs for suspicious activity
- [ ] Check for unauthorized access
- [ ] Identify compromised accounts
- [ ] Determine data accessed/exfiltrated
- [ ] Collect forensic evidence

### Remediation
- [ ] Revoke compromised credentials
- [ ] Patch vulnerabilities
- [ ] Reset all API keys if needed
- [ ] Force password resets for affected accounts
- [ ] Review and update security policies

### Communication
- [ ] Internal: Notify all stakeholders
- [ ] External: Customer notification if data affected
- [ ] Regulatory: Report to authorities if required (GDPR, etc.)
- [ ] Public: Press release if necessary

### Post-Incident
- [ ] Root cause analysis
- [ ] Update security procedures
- [ ] Additional training
- [ ] Enhanced monitoring
```

---

## Capacity Planning

### Growth Tracking

**Monthly Capacity Review**
```yaml
Track Monthly:
  - Total tenants (target growth: +15% MoM)
  - Total API requests (track trend)
  - Total emissions calculated (track trend)
  - Database size (should be <80% of allocated)
  - Storage usage (should be <70% of allocated)
  - Average response time (should remain <500ms p95)
```

**Scaling Triggers**
```yaml
Scale Up When:
  CPU Utilization: >70% sustained for >15 min
  Memory Utilization: >80% sustained for >15 min
  API Latency: p95 >500ms for >10 min
  Database Connections: >80% of max for >5 min
  Queue Depth: >1000 messages for >30 min

Scale Down When:
  CPU Utilization: <30% sustained for >1 hour
  Memory Utilization: <40% sustained for >1 hour
  During: Off-peak hours (12 AM - 6 AM EST)
```

### Capacity Forecasting

```python
# capacity-forecast.py
# Run monthly to project capacity needs

import pandas as pd
from prophet import Prophet

# Load historical metrics
df = pd.read_csv('/opt/vcci/metrics/historical-usage.csv')
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['api_requests_per_day']

# Forecast next 90 days
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Determine if scale-up needed
current_capacity = 5000  # req/s
forecasted_peak = forecast['yhat'].max() / 86400  # convert to req/s

if forecasted_peak > current_capacity * 0.8:
    print(f"⚠️ Capacity increase needed in 90 days")
    print(f"Current: {current_capacity} req/s")
    print(f"Forecasted: {forecasted_peak:.0f} req/s")
    print(f"Recommended: {forecasted_peak * 1.25:.0f} req/s (25% buffer)")
```

---

## Troubleshooting Guide

### Common Issues

**Issue 1: High API Latency**

Symptoms:
- p95 latency > 500ms
- Users reporting slow responses
- Grafana alerts firing

Diagnosis:
```bash
# 1. Check current latency
kubectl exec -n vcci-production deployment/vcci-api -- \
  curl http://localhost:8000/health/metrics | grep http_request_duration

# 2. Check for slow database queries
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT * FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 seconds';"

# 3. Check cache hit rate
kubectl exec -n vcci-production deployment/vcci-api -- \
  redis-cli INFO stats | grep keyspace_hits

# 4. Check for resource constraints
kubectl top pods -n vcci-production
kubectl describe nodes | grep -A5 "Allocated resources"
```

Resolution:
1. **If database slow**: Add indexes, optimize queries, scale read replicas
2. **If cache misses high**: Increase Redis memory, adjust TTLs, pre-warm cache
3. **If CPU high**: Scale horizontally (add pods)
4. **If memory high**: Increase pod memory limits
5. **If external API slow**: Check circuit breakers, fallback to cached data

**Issue 2: Pod CrashLoopBackOff**

Diagnosis:
```bash
# 1. Check pod status
kubectl get pods -n vcci-production

# 2. Check pod logs
kubectl logs -n vcci-production <pod-name>
kubectl logs -n vcci-production <pod-name> --previous

# 3. Check pod events
kubectl describe pod -n vcci-production <pod-name>

# 4. Check resource limits
kubectl get pod -n vcci-production <pod-name> -o yaml | grep -A10 resources
```

Common Causes:
- **Out of Memory**: Increase memory limits
- **Failed Health Checks**: Fix application bug or increase timeout
- **Missing Dependencies**: Verify database, Redis, external APIs accessible
- **Configuration Error**: Check ConfigMaps and Secrets

**Issue 3: Database Connection Exhaustion**

Diagnosis:
```bash
# Check active connections
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT COUNT(*) FROM pg_stat_activity;"

# Check connection pool status
kubectl logs -n vcci-production -l app=vcci-api | grep "connection pool"
```

Resolution:
```yaml
# Increase connection pool size
# Update ConfigMap: app-config

DATABASE_POOL_SIZE: "30"  # was 20
DATABASE_POOL_MAX_OVERFLOW: "15"  # was 10

# Apply changes
kubectl rollout restart deployment/vcci-api -n vcci-production
```

---

## Runbooks

### Runbook 1: Deploy New Version

**Objective**: Deploy new application version with zero downtime

**Prerequisites**:
- [ ] New Docker image built and pushed to registry
- [ ] Smoke tests passed in staging
- [ ] Change request approved
- [ ] Rollback plan ready

**Steps**:
```bash
#!/bin/bash
# deploy-new-version.sh

NEW_VERSION=$1

# 1. Verify image exists
docker manifest inspect ghcr.io/company/vcci-platform:$NEW_VERSION

# 2. Update deployment
kubectl set image deployment/vcci-api -n vcci-production \
  api=ghcr.io/company/vcci-platform:$NEW_VERSION

# 3. Monitor rollout
kubectl rollout status deployment/vcci-api -n vcci-production --timeout=10m

# 4. Verify health
sleep 30
kubectl exec -n vcci-production deployment/vcci-api -- \
  curl -f http://localhost:8000/health/ready

# 5. Run smoke tests
./scripts/smoke-tests.sh

# 6. Monitor for 15 minutes
echo "Deployment complete. Monitoring for 15 minutes..."
sleep 900

# 7. Check error rates
echo "Checking error rates..."
kubectl logs -n vcci-production -l app=vcci-api --since=15m | grep -i error | wc -l

echo "Deployment successful!"
```

**Rollback Procedure** (if issues detected):
```bash
# Immediate rollback
kubectl rollout undo deployment/vcci-api -n vcci-production

# Monitor rollback
kubectl rollout status deployment/vcci-api -n vcci-production
```

### Runbook 2: Scale Application

**Objective**: Scale application to handle increased load

**Scenarios**:

**Scenario A: Horizontal Scaling (Add Pods)**
```bash
# Scale API servers
kubectl scale deployment/vcci-api -n vcci-production --replicas=12

# Scale workers
kubectl scale deployment/vcci-worker -n vcci-production --replicas=8

# Verify
kubectl get pods -n vcci-production
kubectl top pods -n vcci-production
```

**Scenario B: Vertical Scaling (Increase Resources)**
```yaml
# Update deployment resources
kubectl edit deployment/vcci-api -n vcci-production

# Change:
resources:
  requests:
    cpu: 3000m      # was 2000m
    memory: 6Gi     # was 4Gi
  limits:
    cpu: 6000m      # was 4000m
    memory: 12Gi    # was 8Gi
```

**Scenario C: Auto-Scaling Adjustment**
```yaml
# Update HPA
kubectl edit hpa/vcci-api-hpa -n vcci-production

# Change:
spec:
  minReplicas: 8    # was 6
  maxReplicas: 30   # was 20
```

### Runbook 3: Database Maintenance

**Objective**: Perform routine database maintenance

**Weekly Maintenance (Non-disruptive)**:
```bash
# Run during off-peak hours (2-4 AM EST)

# 1. Analyze tables for query optimization
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "ANALYZE VERBOSE;"

# 2. Reindex if needed (check for bloat)
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "REINDEX DATABASE vcci_production CONCURRENTLY;"

# 3. Clean up old data (if applicable)
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "DELETE FROM audit_logs WHERE created_at < NOW() - INTERVAL '90 days';"

# 4. Check table sizes
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "
    SELECT
      schemaname,
      tablename,
      pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
    FROM pg_tables
    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    LIMIT 10;"
```

**Monthly Maintenance (May cause brief performance impact)**:
```bash
# Requires maintenance window (Saturday 2-4 AM EST)

# 1. Full vacuum
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "VACUUM FULL VERBOSE;"

# 2. Update statistics
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "ANALYZE;"
```

---

## Escalation Procedures

### Escalation Path

```
Level 1: On-Call Engineer (24/7)
  ↓ (if unresolved after 30 min for P0, 2 hours for P1)

Level 2: Secondary On-Call / Engineering Manager
  ↓ (if unresolved after 1 hour for P0)

Level 3: Director of Engineering / CTO
  ↓ (for major incidents, data breaches, or escalation needed)

Level 4: Executive Team (CEO, COO)
```

### Contact Information

```yaml
Level 1 - On-Call Engineer:
  PagerDuty: https://company.pagerduty.com
  Slack: #vcci-oncall
  Response: Immediate for P0, 15 min for P1

Level 2 - Engineering Manager:
  Email: eng-manager@company.com
  Phone: +1-555-0102
  Slack: @eng-manager
  Response: 30 min for escalated P0

Level 3 - Director / CTO:
  Email: cto@company.com
  Phone: +1-555-0103 (24/7 emergency)
  Slack: @cto
  Response: 1 hour for escalated P0

Support Channels:
  Team Slack: #vcci-platform-team
  Incidents: #vcci-incidents
  Security: #vcci-security
  Customer Impact: #vcci-customer-support
```

---

## Appendices

### Appendix A: Useful Commands

**Kubernetes**
```bash
# Get pod status
kubectl get pods -n vcci-production

# Check pod logs
kubectl logs -n vcci-production -l app=vcci-api --tail=100 --follow

# Execute command in pod
kubectl exec -n vcci-production deployment/vcci-api -- <command>

# Port forward for debugging
kubectl port-forward -n vcci-production service/vcci-api 8000:80

# Scale deployment
kubectl scale deployment/vcci-api -n vcci-production --replicas=10

# Restart deployment
kubectl rollout restart deployment/vcci-api -n vcci-production

# Check deployment history
kubectl rollout history deployment/vcci-api -n vcci-production

# Rollback deployment
kubectl rollout undo deployment/vcci-api -n vcci-production

# Check resource usage
kubectl top pods -n vcci-production
kubectl top nodes
```

**Database**
```bash
# Connect to database
kubectl exec -it -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL

# Check active queries
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT pid, usename, query, state FROM pg_stat_activity WHERE state = 'active';"

# Kill long-running query
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT pg_terminate_backend(<pid>);"

# Check database size
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT pg_size_pretty(pg_database_size('vcci_production'));"
```

**Redis**
```bash
# Connect to Redis
kubectl exec -it -n vcci-production deployment/vcci-api -- redis-cli -u $REDIS_URL

# Get info
kubectl exec -n vcci-production deployment/vcci-api -- redis-cli -u $REDIS_URL INFO

# Check memory usage
kubectl exec -n vcci-production deployment/vcci-api -- redis-cli -u $REDIS_URL INFO memory

# Flush cache (CAUTION)
kubectl exec -n vcci-production deployment/vcci-api -- redis-cli -u $REDIS_URL FLUSHDB
```

### Appendix B: Monitoring URLs

```yaml
Dashboards:
  Grafana: https://grafana.vcci-platform.com
  Prometheus: https://prometheus.vcci-platform.com
  Kibana: https://kibana.vcci-platform.com

Status:
  Status Page: https://status.vcci-platform.com
  Health Check: https://api.vcci-platform.com/health/ready

Alerts:
  PagerDuty: https://company.pagerduty.com
  Slack: #vcci-alerts

Documentation:
  Runbooks: https://runbooks.vcci-platform.com
  API Docs: https://docs.vcci-platform.com
  Wiki: https://wiki.company.com/vcci
```

### Appendix C: Change Management

**Change Request Process**:
1. Create change request in ServiceNow/Jira
2. Describe change, risk level, rollback plan
3. Get approval from Engineering Manager (P2/P3) or Director (P0/P1 changes)
4. Schedule during approved maintenance window or off-peak hours
5. Execute change following runbook
6. Verify success, monitor for 30 minutes
7. Update change request with results
8. Close change request

**Approved Maintenance Windows**:
- **Standard**: Saturday 2-4 AM EST (weekly)
- **Emergency**: Any time with manager approval for P0 incidents
- **No-Change Periods**: Black Friday, Cyber Monday, End-of-Quarter, End-of-Year

---

**Document Classification**: Internal Use Only
**Version**: 2.0.0
**Last Review Date**: 2025-11-09
**Next Review Date**: 2025-12-09
**Document Owner**: Platform Engineering Team
