# GL-VCCI Scope 3 Platform v2.0 - Operations Guide

## Table of Contents

1. [Overview](#overview)
2. [Daily Operations](#daily-operations)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Log Analysis](#log-analysis)
5. [Performance Tuning](#performance-tuning)
6. [Capacity Planning](#capacity-planning)
7. [Backup and Restore](#backup-and-restore)
8. [Maintenance Windows](#maintenance-windows)

---

## Overview

### Purpose
This guide provides comprehensive operational procedures for maintaining the GL-VCCI Scope 3 Carbon Intelligence Platform in production environments.

### Operational Responsibilities
- **Platform Team**: Infrastructure, deployment, monitoring
- **Development Team**: Application bugs, feature support
- **DevOps Team**: CI/CD, automation, tooling
- **Security Team**: Security monitoring, incident response

### Service Level Objectives (SLOs)
```yaml
Availability: 99.9% uptime (43.8 minutes downtime/month)
API Response Time: p95 < 500ms, p99 < 1000ms
Error Rate: < 0.1% of all requests
Database Query Time: p95 < 100ms
Recovery Time Objective (RTO): 4 hours
Recovery Point Objective (RPO): 15 minutes
```

### On-Call Schedule
- **Primary**: 24/7 rotation (1 week shifts)
- **Secondary**: Backup on-call engineer
- **Escalation**: Engineering manager, CTO
- **Response Time**: P0 (immediate), P1 (15 min), P2 (1 hour), P3 (next business day)

---

## Daily Operations

### 1. Morning Health Check (09:00 AM)

**Daily Checklist Script**
```bash
#!/bin/bash
# daily-health-check.sh

set -euo pipefail

NAMESPACE="vcci-production"
API_URL="https://api.vcci-platform.com"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"

echo "=========================================="
echo "VCCI Platform Daily Health Check"
echo "Date: $(date)"
echo "=========================================="

# Function to send Slack notification
send_slack() {
  local message=$1
  local color=${2:-"good"}
  curl -X POST "$SLACK_WEBHOOK" \
    -H 'Content-Type: application/json' \
    -d "{\"attachments\":[{\"color\":\"${color}\",\"text\":\"${message}\"}]}"
}

# 1. Check Kubernetes Cluster Health
echo "1. Checking Kubernetes cluster health..."
CLUSTER_STATUS=$(kubectl cluster-info | head -n 1)
NODE_STATUS=$(kubectl get nodes --no-headers | awk '{print $2}' | grep -c "Ready" || echo 0)
TOTAL_NODES=$(kubectl get nodes --no-headers | wc -l)

if [ "$NODE_STATUS" -eq "$TOTAL_NODES" ]; then
  echo "✓ All $TOTAL_NODES nodes are Ready"
else
  echo "✗ Only $NODE_STATUS/$TOTAL_NODES nodes are Ready"
  send_slack "⚠️ Node health issue: Only $NODE_STATUS/$TOTAL_NODES nodes are Ready" "warning"
fi

# 2. Check Pod Health
echo "2. Checking pod health..."
NOT_RUNNING=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
CRASH_LOOPING=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -c "CrashLoopBackOff" || echo 0)

if [ "$NOT_RUNNING" -gt 0 ] || [ "$CRASH_LOOPING" -gt 0 ]; then
  echo "✗ Pod issues detected: $NOT_RUNNING not running, $CRASH_LOOPING crash looping"
  send_slack "⚠️ Pod health issue: $NOT_RUNNING not running, $CRASH_LOOPING crash looping" "danger"
else
  echo "✓ All pods are healthy"
fi

# 3. Check API Health
echo "3. Checking API health..."
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health/ready")

if [ "$API_STATUS" -eq 200 ]; then
  echo "✓ API is healthy (HTTP $API_STATUS)"
else
  echo "✗ API health check failed (HTTP $API_STATUS)"
  send_slack "⚠️ API health check failed: HTTP $API_STATUS" "danger"
fi

# 4. Check Database Connections
echo "4. Checking database connections..."
DB_CONNECTIONS=$(kubectl exec -n "$NAMESPACE" deployment/vcci-api -- \
  psql "$DATABASE_URL" -t -c "SELECT count(*) FROM pg_stat_activity WHERE datname='vcci_production';")
DB_MAX_CONNECTIONS=$(kubectl exec -n "$NAMESPACE" deployment/vcci-api -- \
  psql "$DATABASE_URL" -t -c "SHOW max_connections;")

DB_USAGE=$((DB_CONNECTIONS * 100 / DB_MAX_CONNECTIONS))
echo "  Database connections: $DB_CONNECTIONS / $DB_MAX_CONNECTIONS ($DB_USAGE%)"

if [ "$DB_USAGE" -gt 80 ]; then
  echo "✗ Database connection usage is high: $DB_USAGE%"
  send_slack "⚠️ Database connection usage is high: $DB_USAGE%" "warning"
else
  echo "✓ Database connections are healthy"
fi

# 5. Check Redis Health
echo "5. Checking Redis health..."
REDIS_PING=$(kubectl exec -n "$NAMESPACE" deployment/vcci-api -- \
  redis-cli -u "$REDIS_URL" ping 2>/dev/null || echo "FAILED")

if [ "$REDIS_PING" == "PONG" ]; then
  echo "✓ Redis is healthy"
else
  echo "✗ Redis health check failed"
  send_slack "⚠️ Redis health check failed" "danger"
fi

# 6. Check Disk Usage
echo "6. Checking disk usage..."
DISK_USAGE=$(kubectl exec -n "$NAMESPACE" deployment/vcci-api -- df -h / | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$DISK_USAGE" -gt 80 ]; then
  echo "✗ Disk usage is high: $DISK_USAGE%"
  send_slack "⚠️ Disk usage is high: $DISK_USAGE%" "warning"
else
  echo "✓ Disk usage is healthy: $DISK_USAGE%"
fi

# 7. Check Certificate Expiration
echo "7. Checking certificate expiration..."
CERT_EXPIRY=$(kubectl get certificate -n "$NAMESPACE" vcci-api-tls -o jsonpath='{.status.notAfter}')
CERT_EXPIRY_EPOCH=$(date -d "$CERT_EXPIRY" +%s)
CURRENT_EPOCH=$(date +%s)
DAYS_UNTIL_EXPIRY=$(( (CERT_EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))

if [ "$DAYS_UNTIL_EXPIRY" -lt 30 ]; then
  echo "✗ Certificate expires in $DAYS_UNTIL_EXPIRY days"
  send_slack "⚠️ Certificate expires in $DAYS_UNTIL_EXPIRY days" "warning"
else
  echo "✓ Certificate is valid for $DAYS_UNTIL_EXPIRY days"
fi

# 8. Check Recent Errors
echo "8. Checking recent errors..."
ERROR_COUNT=$(kubectl logs -n "$NAMESPACE" -l app=vcci-api --since=24h | grep -c "ERROR" || echo 0)
echo "  Errors in last 24h: $ERROR_COUNT"

if [ "$ERROR_COUNT" -gt 1000 ]; then
  echo "✗ High error rate detected: $ERROR_COUNT errors in 24h"
  send_slack "⚠️ High error rate: $ERROR_COUNT errors in last 24h" "warning"
else
  echo "✓ Error rate is acceptable"
fi

# 9. Check Backup Status
echo "9. Checking backup status..."
LAST_BACKUP=$(aws s3 ls s3://vcci-backups/production/database/ | tail -1 | awk '{print $1, $2}')
LAST_BACKUP_EPOCH=$(date -d "$LAST_BACKUP" +%s)
HOURS_SINCE_BACKUP=$(( (CURRENT_EPOCH - LAST_BACKUP_EPOCH) / 3600 ))

if [ "$HOURS_SINCE_BACKUP" -gt 24 ]; then
  echo "✗ Last backup was $HOURS_SINCE_BACKUP hours ago"
  send_slack "⚠️ Last backup was $HOURS_SINCE_BACKUP hours ago" "danger"
else
  echo "✓ Last backup was $HOURS_SINCE_BACKUP hours ago"
fi

# 10. Summary
echo ""
echo "=========================================="
echo "Daily Health Check Complete"
echo "=========================================="

# Send summary to Slack
send_slack "✅ Daily health check completed - All systems operational" "good"
```

**Run Daily Health Check**
```bash
chmod +x daily-health-check.sh
./daily-health-check.sh
```

### 2. Resource Usage Monitoring

**Check Resource Usage**
```bash
#!/bin/bash
# check-resources.sh

NAMESPACE="vcci-production"

echo "=========================================="
echo "Resource Usage Report"
echo "=========================================="

# Pod resource usage
echo "1. Pod Resource Usage:"
kubectl top pods -n "$NAMESPACE" --sort-by=memory | head -20

echo ""
echo "2. Node Resource Usage:"
kubectl top nodes

echo ""
echo "3. Persistent Volume Usage:"
kubectl get pv -o custom-columns=\
NAME:.metadata.name,\
CAPACITY:.spec.capacity.storage,\
USED:.status.capacity.storage,\
STATUS:.status.phase

echo ""
echo "4. HPA Status:"
kubectl get hpa -n "$NAMESPACE"

echo ""
echo "5. Resource Quotas:"
kubectl get resourcequota -n "$NAMESPACE"
```

### 3. Log Rotation and Cleanup

**Cleanup Old Logs**
```bash
#!/bin/bash
# cleanup-logs.sh

NAMESPACE="vcci-production"
RETENTION_DAYS=90

echo "Cleaning up logs older than $RETENTION_DAYS days..."

# Cleanup Elasticsearch indices
curl -X DELETE "http://elasticsearch:9200/vcci-logs-$(date -d "$RETENTION_DAYS days ago" +%Y.%m.%d)"

# Cleanup S3 logs
aws s3 ls s3://vcci-logs/production/ --recursive | \
  while read -r line; do
    createDate=$(echo "$line" | awk {'print $1" "$2'})
    createDate=$(date -d "$createDate" +%s)
    olderThan=$(date -d "$RETENTION_DAYS days ago" +%s)
    if [[ $createDate -lt $olderThan ]]; then
      fileName=$(echo "$line" | awk {'print $4'})
      if [[ $fileName != "" ]]; then
        aws s3 rm "s3://vcci-logs/production/$fileName"
      fi
    fi
  done

echo "Log cleanup complete"
```

---

## Monitoring and Alerting

### 1. Grafana Dashboards

**Main Platform Dashboard**
```json
{
  "dashboard": {
    "title": "VCCI Platform Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{namespace=\"vcci-production\"}[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{namespace=\"vcci-production\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{namespace=\"vcci-production\"}[5m]))"
          }
        ]
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{namespace=\"vcci-production\"}[5m])) by (le))"
          }
        ]
      },
      {
        "title": "Active Database Connections",
        "targets": [
          {
            "expr": "pg_stat_activity_count{datname=\"vcci_production\"}"
          }
        ]
      },
      {
        "title": "Redis Memory Usage",
        "targets": [
          {
            "expr": "redis_memory_used_bytes / redis_memory_max_bytes * 100"
          }
        ]
      },
      {
        "title": "Pod CPU Usage",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total{namespace=\"vcci-production\"}[5m])) by (pod)"
          }
        ]
      },
      {
        "title": "Pod Memory Usage",
        "targets": [
          {
            "expr": "sum(container_memory_working_set_bytes{namespace=\"vcci-production\"}) by (pod)"
          }
        ]
      },
      {
        "title": "Tenant Activity",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{namespace=\"vcci-production\"}[5m])) by (tenant_id)"
          }
        ]
      }
    ]
  }
}
```

**Access Dashboards**
```bash
# Port-forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Open in browser
open http://localhost:3000

# Default credentials
Username: admin
Password: <retrieve from secrets>
```

### 2. Prometheus Alerts

**Alert Rules Configuration**
```yaml
# prometheus-alerts.yaml
groups:
- name: vcci-platform-alerts
  interval: 30s
  rules:

  # High Error Rate
  - alert: HighErrorRate
    expr: |
      sum(rate(http_requests_total{namespace="vcci-production",status=~"5.."}[5m]))
      / sum(rate(http_requests_total{namespace="vcci-production"}[5m])) > 0.01
    for: 5m
    labels:
      severity: critical
      team: platform
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} (threshold: 1%)"

  # High Response Time
  - alert: HighResponseTime
    expr: |
      histogram_quantile(0.95,
        sum(rate(http_request_duration_seconds_bucket{namespace="vcci-production"}[5m])) by (le)
      ) > 0.5
    for: 10m
    labels:
      severity: warning
      team: platform
    annotations:
      summary: "High API response time"
      description: "p95 response time is {{ $value }}s (threshold: 0.5s)"

  # Pod Crash Looping
  - alert: PodCrashLooping
    expr: rate(kube_pod_container_status_restarts_total{namespace="vcci-production"}[15m]) > 0
    for: 5m
    labels:
      severity: critical
      team: platform
    annotations:
      summary: "Pod {{ $labels.pod }} is crash looping"
      description: "Pod {{ $labels.pod }} has restarted {{ $value }} times in the last 15 minutes"

  # High Database Connections
  - alert: HighDatabaseConnections
    expr: |
      pg_stat_activity_count{datname="vcci_production"}
      / pg_settings_max_connections > 0.8
    for: 5m
    labels:
      severity: warning
      team: platform
    annotations:
      summary: "High database connection usage"
      description: "Database connections at {{ $value | humanizePercentage }} of max"

  # High Redis Memory
  - alert: HighRedisMemory
    expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
    for: 5m
    labels:
      severity: warning
      team: platform
    annotations:
      summary: "Redis memory usage high"
      description: "Redis memory usage at {{ $value | humanizePercentage }}"

  # Disk Space Low
  - alert: DiskSpaceLow
    expr: |
      (node_filesystem_avail_bytes{mountpoint="/"}
      / node_filesystem_size_bytes{mountpoint="/"}) < 0.2
    for: 5m
    labels:
      severity: warning
      team: platform
    annotations:
      summary: "Low disk space on {{ $labels.instance }}"
      description: "Disk space is {{ $value | humanizePercentage }} full"

  # Certificate Expiring Soon
  - alert: CertificateExpiringSoon
    expr: (certmanager_certificate_expiration_timestamp_seconds - time()) / 86400 < 30
    for: 1h
    labels:
      severity: warning
      team: platform
    annotations:
      summary: "Certificate {{ $labels.name }} expiring soon"
      description: "Certificate expires in {{ $value }} days"

  # Backup Failed
  - alert: BackupFailed
    expr: time() - vcci_last_successful_backup_timestamp > 86400
    for: 1h
    labels:
      severity: critical
      team: platform
    annotations:
      summary: "Backup has not completed successfully"
      description: "Last successful backup was {{ $value | humanizeDuration }} ago"

  # Pod Memory Usage High
  - alert: PodMemoryHigh
    expr: |
      container_memory_working_set_bytes{namespace="vcci-production"}
      / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: warning
      team: platform
    annotations:
      summary: "Pod {{ $labels.pod }} memory usage high"
      description: "Memory usage at {{ $value | humanizePercentage }}"

  # Pod CPU Throttling
  - alert: PodCPUThrottling
    expr: |
      rate(container_cpu_cfs_throttled_seconds_total{namespace="vcci-production"}[5m])
      / rate(container_cpu_cfs_periods_total{namespace="vcci-production"}[5m]) > 0.5
    for: 10m
    labels:
      severity: warning
      team: platform
    annotations:
      summary: "Pod {{ $labels.pod }} CPU throttling"
      description: "CPU throttling at {{ $value | humanizePercentage }}"
```

**Apply Alert Rules**
```bash
kubectl apply -f prometheus-alerts.yaml -n monitoring

# Verify alerts
kubectl exec -n monitoring prometheus-0 -- promtool check rules /etc/prometheus/rules/prometheus-alerts.yaml
```

### 3. PagerDuty Integration

**Configure PagerDuty**
```yaml
# alertmanager-config.yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'pagerduty-critical'

  routes:
  - match:
      severity: critical
    receiver: 'pagerduty-critical'

  - match:
      severity: warning
    receiver: 'slack-warnings'

receivers:
- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: '<PAGERDUTY_SERVICE_KEY>'
    description: '{{ .GroupLabels.alertname }}'
    severity: 'critical'
    details:
      summary: '{{ .CommonAnnotations.summary }}'
      description: '{{ .CommonAnnotations.description }}'

- name: 'slack-warnings'
  slack_configs:
  - api_url: '<SLACK_WEBHOOK_URL>'
    channel: '#vcci-platform-alerts'
    title: '{{ .GroupLabels.alertname }}'
    text: '{{ .CommonAnnotations.description }}'
    color: 'warning'
```

---

## Log Analysis

### 1. Centralized Logging with ELK

**Access Kibana**
```bash
# Port-forward Kibana
kubectl port-forward -n logging svc/kibana 5601:5601

# Open in browser
open http://localhost:5601
```

**Common Log Queries**

**Find API Errors (Last 24h)**
```
namespace: "vcci-production" AND
level: "ERROR" AND
@timestamp: [now-24h TO now]
```

**Find Slow Database Queries**
```
namespace: "vcci-production" AND
message: "slow query" AND
duration: >1000
```

**Find Authentication Failures**
```
namespace: "vcci-production" AND
message: "authentication failed" OR message: "invalid credentials"
```

**Find Tenant-Specific Issues**
```
namespace: "vcci-production" AND
tenant_id: "acme-corp" AND
level: "ERROR"
```

### 2. Log Analysis Scripts

**Analyze Error Patterns**
```bash
#!/bin/bash
# analyze-errors.sh

NAMESPACE="vcci-production"
HOURS_AGO=24

echo "Analyzing errors from last $HOURS_AGO hours..."

# Get error logs
kubectl logs -n "$NAMESPACE" -l app=vcci-api --since="${HOURS_AGO}h" | \
  grep "ERROR" > /tmp/errors.log

# Count by error type
echo "Top 10 error types:"
cat /tmp/errors.log | \
  grep -oP '"error_type":"\K[^"]+' | \
  sort | uniq -c | sort -rn | head -10

# Count by tenant
echo ""
echo "Top 10 tenants with errors:"
cat /tmp/errors.log | \
  grep -oP '"tenant_id":"\K[^"]+' | \
  sort | uniq -c | sort -rn | head -10

# Count by endpoint
echo ""
echo "Top 10 endpoints with errors:"
cat /tmp/errors.log | \
  grep -oP '"endpoint":"\K[^"]+' | \
  sort | uniq -c | sort -rn | head -10
```

**Export Logs for Analysis**
```bash
#!/bin/bash
# export-logs.sh

NAMESPACE="vcci-production"
START_DATE=$1
END_DATE=$2
OUTPUT_FILE="logs-${START_DATE}-to-${END_DATE}.json"

# Export logs from Elasticsearch
curl -X GET "http://elasticsearch:9200/vcci-logs-*/_search" \
  -H 'Content-Type: application/json' \
  -d "{
    \"query\": {
      \"range\": {
        \"@timestamp\": {
          \"gte\": \"${START_DATE}\",
          \"lte\": \"${END_DATE}\"
        }
      }
    },
    \"size\": 10000,
    \"sort\": [{\"@timestamp\": \"desc\"}]
  }" > "$OUTPUT_FILE"

echo "Logs exported to $OUTPUT_FILE"
```

### 3. Log Retention Policy

**Configure Log Retention**
```bash
# Elasticsearch ILM policy
curl -X PUT "http://elasticsearch:9200/_ilm/policy/vcci-logs-policy" \
  -H 'Content-Type: application/json' \
  -d '{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": {
            "max_size": "50GB",
            "max_age": "7d"
          }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "allocate": {
            "number_of_replicas": 1
          },
          "forcemerge": {
            "max_num_segments": 1
          },
          "shrink": {
            "number_of_shards": 1
          }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "allocate": {
            "number_of_replicas": 0
          },
          "freeze": {}
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}'
```

---

## Performance Tuning

### 1. Database Performance Tuning

**Analyze Slow Queries**
```sql
-- Enable pg_stat_statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find slow queries
SELECT
  query,
  calls,
  total_exec_time,
  mean_exec_time,
  max_exec_time,
  stddev_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Find queries with high I/O
SELECT
  query,
  calls,
  shared_blks_hit,
  shared_blks_read,
  shared_blks_hit::float / (shared_blks_hit + shared_blks_read) as cache_hit_ratio
FROM pg_stat_statements
WHERE shared_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 20;
```

**Index Optimization**
```sql
-- Find missing indexes
SELECT
  schemaname,
  tablename,
  seq_scan,
  seq_tup_read,
  idx_scan,
  seq_tup_read / seq_scan as avg_seq_scan
FROM pg_stat_user_tables
WHERE seq_scan > 0
  AND seq_tup_read / seq_scan > 1000
ORDER BY seq_tup_read DESC
LIMIT 20;

-- Find unused indexes
SELECT
  schemaname,
  tablename,
  indexname,
  idx_scan,
  pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Create recommended indexes
-- Review and create based on analysis
CREATE INDEX CONCURRENTLY idx_emissions_tenant_date
  ON tenant_acme.emissions(tenant_id, calculated_at);
```

**Vacuum and Analyze**
```bash
# Manual vacuum
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "VACUUM ANALYZE;"

# Schedule automatic vacuum
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-vacuum
  namespace: vcci-production
spec:
  schedule: "0 2 * * 0"  # Weekly on Sunday at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: vacuum
            image: postgres:14-alpine
            command:
            - psql
            - \$(DATABASE_URL)
            - -c
            - "VACUUM ANALYZE;"
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: database-credentials
                  key: url
          restartPolicy: OnFailure
EOF
```

### 2. Redis Cache Optimization

**Monitor Cache Hit Rate**
```bash
#!/bin/bash
# check-redis-stats.sh

REDIS_URL=$1

redis-cli -u "$REDIS_URL" INFO stats | grep -E "keyspace_hits|keyspace_misses"

# Calculate hit rate
HITS=$(redis-cli -u "$REDIS_URL" INFO stats | grep keyspace_hits | cut -d: -f2 | tr -d '\r')
MISSES=$(redis-cli -u "$REDIS_URL" INFO stats | grep keyspace_misses | cut -d: -f2 | tr -d '\r')
TOTAL=$((HITS + MISSES))
HIT_RATE=$(echo "scale=2; $HITS * 100 / $TOTAL" | bc)

echo "Cache hit rate: $HIT_RATE%"
```

**Optimize Cache Configuration**
```bash
# Update Redis configuration
redis-cli -u "$REDIS_URL" CONFIG SET maxmemory-policy allkeys-lru
redis-cli -u "$REDIS_URL" CONFIG SET maxmemory-samples 10
redis-cli -u "$REDIS_URL" CONFIG SET lazyfree-lazy-eviction yes
redis-cli -u "$REDIS_URL" CONFIG SET lazyfree-lazy-expire yes

# Save configuration
redis-cli -u "$REDIS_URL" CONFIG REWRITE
```

### 3. Application Performance Tuning

**Profile API Performance**
```python
# Add profiling middleware
from werkzeug.middleware.profiler import ProfilerMiddleware

app.wsgi_app = ProfilerMiddleware(
    app.wsgi_app,
    restrictions=[30],
    profile_dir='/tmp/profiler'
)
```

**Optimize Database Queries**
```python
# Use query caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_emission_factors(category: str):
    return db.query(EmissionFactor).filter_by(category=category).all()

# Use connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Use eager loading
emissions = db.query(Emission)\
    .options(joinedload(Emission.source))\
    .options(joinedload(Emission.category))\
    .all()
```

---

## Capacity Planning

### 1. Resource Forecasting

**Historical Usage Analysis**
```bash
#!/bin/bash
# analyze-capacity.sh

NAMESPACE="vcci-production"
DAYS=30

echo "Analyzing capacity usage for last $DAYS days..."

# CPU usage trend
kubectl top pods -n "$NAMESPACE" --containers | \
  awk '{sum+=$2} END {print "Average CPU:", sum/NR, "cores"}'

# Memory usage trend
kubectl top pods -n "$NAMESPACE" --containers | \
  awk '{sum+=$3} END {print "Average Memory:", sum/NR, "MB"}'

# Database size growth
kubectl exec -n "$NAMESPACE" deployment/vcci-api -- \
  psql "$DATABASE_URL" -t -c "
    SELECT pg_size_pretty(pg_database_size('vcci_production'));
  "

# Request rate trend
echo "Request rate (requests/min):"
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  "rate(http_requests_total{namespace='vcci-production'}[24h]) * 60"
```

**Growth Projections**
```python
# capacity-forecast.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load historical data
df = pd.read_csv('historical-metrics.csv')

# Prepare data
X = np.array(df['days_since_start']).reshape(-1, 1)
y = df['total_requests']

# Train model
model = LinearRegression()
model.fit(X, y)

# Forecast next 90 days
future_days = np.array(range(len(df), len(df) + 90)).reshape(-1, 1)
forecast = model.predict(future_days)

# Calculate required capacity
current_capacity = 10000000  # requests/day
forecasted_peak = forecast.max()
required_capacity = forecasted_peak * 1.3  # 30% buffer

print(f"Current capacity: {current_capacity:,} req/day")
print(f"Forecasted peak: {forecasted_peak:,.0f} req/day")
print(f"Required capacity: {required_capacity:,.0f} req/day")
print(f"Scale factor: {required_capacity / current_capacity:.2f}x")
```

### 2. Scaling Decisions

**When to Scale Up**
```yaml
Scale Application:
  - CPU usage > 70% sustained for 10+ minutes
  - Memory usage > 80% sustained
  - Request queue length > 100
  - Response time p95 > 500ms

Scale Database:
  - Connection pool > 80% utilized
  - Query time p95 > 100ms
  - Disk I/O wait > 20%
  - Database size approaching 80% of volume

Scale Redis:
  - Memory usage > 80%
  - Cache hit rate < 70%
  - Connection pool > 80% utilized
```

**Scaling Checklist**
```bash
# 1. Review current metrics
./analyze-capacity.sh

# 2. Calculate required capacity
# Use forecasting scripts

# 3. Update HPA settings
kubectl patch hpa vcci-api-hpa -n vcci-production -p '
{
  "spec": {
    "maxReplicas": 30
  }
}'

# 4. Update node group (if needed)
aws eks update-nodegroup-config \
  --cluster-name vcci-production \
  --nodegroup-name application \
  --scaling-config minSize=6,maxSize=20,desiredSize=12

# 5. Validate scaling
kubectl get hpa -n vcci-production -w
```

---

## Backup and Restore

### 1. Automated Backup Configuration

**Database Backup CronJob**
```yaml
# k8s/cronjobs/database-backup.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: vcci-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  successfulJobsHistoryLimit: 7
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: vcci-backup
          containers:
          - name: backup
            image: postgres:14-alpine
            command:
            - /bin/sh
            - -c
            - |
              BACKUP_FILE="vcci-production-$(date +%Y%m%d-%H%M%S).sql.gz"

              # Create backup
              pg_dump "$DATABASE_URL" | gzip > "/tmp/$BACKUP_FILE"

              # Upload to S3
              aws s3 cp "/tmp/$BACKUP_FILE" "s3://vcci-backups/production/database/$BACKUP_FILE"

              # Verify backup
              aws s3 ls "s3://vcci-backups/production/database/$BACKUP_FILE"

              # Update metrics
              echo "vcci_last_successful_backup_timestamp $(date +%s)" | \
                curl --data-binary @- http://prometheus-pushgateway:9091/metrics/job/database_backup

              echo "Backup completed: $BACKUP_FILE"
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: database-credentials
                  key: url
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key-id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-access-key
          restartPolicy: OnFailure
```

**Apply Backup CronJob**
```bash
kubectl apply -f k8s/cronjobs/database-backup.yaml

# Verify cron job
kubectl get cronjobs -n vcci-production

# Manual trigger
kubectl create job --from=cronjob/database-backup manual-backup-$(date +%s) -n vcci-production
```

### 2. Backup Verification

**Test Backup Integrity**
```bash
#!/bin/bash
# verify-backup.sh

BACKUP_FILE=$1

echo "Verifying backup: $BACKUP_FILE"

# Download backup
aws s3 cp "s3://vcci-backups/production/database/$BACKUP_FILE" /tmp/

# Test decompression
gunzip -t "/tmp/$BACKUP_FILE" && echo "✓ Backup file is valid"

# Test restore to temporary database (optional)
# createdb test_restore
# gunzip -c "/tmp/$BACKUP_FILE" | psql test_restore
# dropdb test_restore

echo "Backup verification complete"
```

### 3. Restore Procedures

**Full Database Restore**
```bash
#!/bin/bash
# restore-database.sh

set -euo pipefail

BACKUP_FILE=$1
CONFIRMATION=$2

if [ "$CONFIRMATION" != "YES-RESTORE-DATABASE" ]; then
  echo "ERROR: You must confirm restore by passing 'YES-RESTORE-DATABASE'"
  echo "Usage: $0 <backup-file> YES-RESTORE-DATABASE"
  exit 1
fi

echo "=========================================="
echo "DATABASE RESTORE - DESTRUCTIVE OPERATION"
echo "Backup: $BACKUP_FILE"
echo "=========================================="

# 1. Scale down application
echo "Step 1: Scaling down application..."
kubectl scale deployment/vcci-api -n vcci-production --replicas=0

# 2. Create pre-restore backup
echo "Step 2: Creating pre-restore backup..."
PREBACKUP="pre-restore-$(date +%Y%m%d-%H%M%S).sql.gz"
kubectl exec -n vcci-production deployment/vcci-api -- \
  pg_dump "$DATABASE_URL" | gzip > "/tmp/$PREBACKUP"
aws s3 cp "/tmp/$PREBACKUP" "s3://vcci-backups/production/database/$PREBACKUP"

# 3. Download backup
echo "Step 3: Downloading backup..."
aws s3 cp "s3://vcci-backups/production/database/$BACKUP_FILE" /tmp/

# 4. Restore database
echo "Step 4: Restoring database..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  sh -c "gunzip -c | psql \$DATABASE_URL" < "/tmp/$BACKUP_FILE"

# 5. Verify restore
echo "Step 5: Verifying restore..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM public.tenants;"

# 6. Scale up application
echo "Step 6: Scaling up application..."
kubectl scale deployment/vcci-api -n vcci-production --replicas=6

echo ""
echo "=========================================="
echo "Database Restore Complete"
echo "=========================================="
```

---

## Maintenance Windows

### 1. Scheduled Maintenance

**Maintenance Window Schedule**
```
Weekly Maintenance: Sunday 02:00-04:00 UTC
Monthly Maintenance: First Sunday 02:00-06:00 UTC
Emergency Maintenance: As needed (requires approval)
```

**Maintenance Checklist**
```markdown
Pre-Maintenance (1 week before):
- [ ] Review change requests
- [ ] Schedule maintenance window
- [ ] Notify stakeholders (email, Slack)
- [ ] Prepare rollback plan
- [ ] Update status page

Pre-Maintenance (1 day before):
- [ ] Final stakeholder notification
- [ ] Backup all critical data
- [ ] Test rollback procedures
- [ ] Prepare monitoring dashboards

During Maintenance:
- [ ] Enable maintenance mode
- [ ] Execute changes
- [ ] Validate each step
- [ ] Monitor for issues
- [ ] Document deviations

Post-Maintenance:
- [ ] Disable maintenance mode
- [ ] Verify all services operational
- [ ] Update status page
- [ ] Post-mortem (if issues occurred)
- [ ] Update documentation
```

### 2. Maintenance Mode

**Enable Maintenance Mode**
```bash
#!/bin/bash
# enable-maintenance-mode.sh

NAMESPACE="vcci-production"

echo "Enabling maintenance mode..."

# Deploy maintenance page
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: maintenance-page
  namespace: $NAMESPACE
data:
  index.html: |
    <!DOCTYPE html>
    <html>
    <head>
      <title>Maintenance</title>
    </head>
    <body>
      <h1>Scheduled Maintenance</h1>
      <p>We're currently performing scheduled maintenance.</p>
      <p>Expected completion: 04:00 UTC</p>
    </body>
    </html>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: maintenance-page
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: maintenance-page
  template:
    metadata:
      labels:
        app: maintenance-page
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        volumeMounts:
        - name: content
          mountPath: /usr/share/nginx/html
      volumes:
      - name: content
        configMap:
          name: maintenance-page
---
apiVersion: v1
kind: Service
metadata:
  name: maintenance-page
  namespace: $NAMESPACE
spec:
  selector:
    app: maintenance-page
  ports:
  - port: 80
    targetPort: 80
EOF

# Update ingress to point to maintenance page
kubectl patch ingress vcci-api-ingress -n "$NAMESPACE" --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/rules/0/http/paths/0/backend/service/name",
    "value": "maintenance-page"
  }
]'

echo "Maintenance mode enabled"
```

**Disable Maintenance Mode**
```bash
#!/bin/bash
# disable-maintenance-mode.sh

NAMESPACE="vcci-production"

echo "Disabling maintenance mode..."

# Restore ingress
kubectl patch ingress vcci-api-ingress -n "$NAMESPACE" --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/rules/0/http/paths/0/backend/service/name",
    "value": "vcci-api"
  }
]'

# Remove maintenance page
kubectl delete deployment maintenance-page -n "$NAMESPACE"
kubectl delete service maintenance-page -n "$NAMESPACE"
kubectl delete configmap maintenance-page -n "$NAMESPACE"

echo "Maintenance mode disabled"
```

---

## Appendix

### Quick Reference Commands

**Kubernetes**
```bash
# Get pod logs
kubectl logs -n vcci-production -l app=vcci-api --tail=100 -f

# Execute command in pod
kubectl exec -n vcci-production -it deployment/vcci-api -- /bin/bash

# Port forward
kubectl port-forward -n vcci-production svc/vcci-api 8000:80

# Restart deployment
kubectl rollout restart deployment/vcci-api -n vcci-production

# Scale deployment
kubectl scale deployment/vcci-api -n vcci-production --replicas=10
```

**Database**
```bash
# Connect to database
kubectl exec -n vcci-production -it deployment/vcci-api -- psql $DATABASE_URL

# Run query
kubectl exec -n vcci-production deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT COUNT(*) FROM public.tenants;"

# Export data
kubectl exec -n vcci-production deployment/vcci-api -- \
  pg_dump $DATABASE_URL --table=public.emissions > emissions.sql
```

**Monitoring**
```bash
# Check metrics
kubectl top pods -n vcci-production
kubectl top nodes

# View HPA status
kubectl get hpa -n vcci-production

# Check events
kubectl get events -n vcci-production --sort-by='.lastTimestamp'
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-06
**Maintained By**: Platform Engineering Team
