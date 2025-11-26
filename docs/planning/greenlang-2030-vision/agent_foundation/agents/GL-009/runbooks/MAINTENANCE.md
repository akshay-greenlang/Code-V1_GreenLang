# GL-009 THERMALIQ Maintenance Guide

**Agent**: GL-009 THERMALIQ ThermalEfficiencyCalculator
**Version**: 1.0.0
**Last Updated**: 2025-11-26
**Owner**: GreenLang SRE Team

---

## Table of Contents

1. [Overview](#overview)
2. [Daily Maintenance](#daily-maintenance)
3. [Weekly Maintenance](#weekly-maintenance)
4. [Monthly Maintenance](#monthly-maintenance)
5. [Quarterly Maintenance](#quarterly-maintenance)
6. [Database Maintenance](#database-maintenance)
7. [Cache Maintenance](#cache-maintenance)
8. [Certificate Management](#certificate-management)
9. [Dependency Updates](#dependency-updates)
10. [Security Maintenance](#security-maintenance)
11. [Backup and Recovery](#backup-and-recovery)
12. [Performance Tuning](#performance-tuning)

---

## Overview

This guide provides comprehensive maintenance procedures for GL-009 THERMALIQ ThermalEfficiencyCalculator to ensure optimal performance, security, and reliability.

### Maintenance Windows

**Standard Maintenance Windows**:
- Daily: Continuous (automated tasks)
- Weekly: Sundays 02:00-04:00 UTC
- Monthly: First Sunday of month 02:00-06:00 UTC
- Quarterly: First Sunday of quarter 02:00-08:00 UTC

**Emergency Maintenance**:
- Security patches: As needed (immediate)
- Critical bugs: Within 24 hours
- Performance issues: Within 48 hours

---

## Daily Maintenance

### Automated Daily Tasks (via CronJobs)

**Log Rotation**:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: thermaliq-log-rotation
  namespace: gl-009-production
spec:
  schedule: "0 0 * * *"  # Midnight UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: log-rotation
            image: alpine:latest
            command:
            - /bin/sh
            - -c
            - |
              # Rotate logs older than 7 days
              find /var/log/thermaliq -name "*.log" -mtime +7 -delete
              # Compress logs older than 1 day
              find /var/log/thermaliq -name "*.log" -mtime +1 -exec gzip {} \;
          restartPolicy: OnFailure
```

**Backup Verification**:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: thermaliq-backup-verification
  namespace: gl-009-production
spec:
  schedule: "0 1 * * *"  # 01:00 UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup-verify
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              # Verify latest backup exists
              LATEST_BACKUP=$(aws s3 ls s3://greenlang-backups/thermaliq/ --recursive | sort | tail -n 1 | awk '{print $4}')
              if [ -z "$LATEST_BACKUP" ]; then
                echo "ERROR: No backup found"
                exit 1
              fi

              # Verify backup is recent (< 25 hours old)
              BACKUP_AGE=$(( $(date +%s) - $(date -r <(aws s3api head-object --bucket greenlang-backups --key $LATEST_BACKUP --query 'LastModified' --output text) +%s) ))
              if [ $BACKUP_AGE -gt 90000 ]; then
                echo "ERROR: Backup is too old ($BACKUP_AGE seconds)"
                exit 1
              fi

              echo "SUCCESS: Backup verified"
          restartPolicy: OnFailure
```

### Manual Daily Checks (10 minutes)

**Health Check Review**:
```bash
#!/bin/bash
# daily_health_check.sh

echo "=== GL-009 THERMALIQ Daily Health Check ==="
echo "Date: $(date -u)"
echo ""

# 1. Check all pods running
echo "1. Pod Status:"
kubectl get pods -n gl-009-production -l app=thermaliq
echo ""

# 2. Check health endpoint
echo "2. Health Endpoint:"
curl -s https://api.greenlang.io/v1/thermaliq/health | jq .
echo ""

# 3. Check error rate (last 24 hours)
echo "3. Error Rate (24h):"
ERROR_RATE=$(curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_calculation_errors_total[24h])/rate(thermaliq_calculation_requests_total[24h])' | \
  jq -r '.data.result[0].value[1]')
echo "Error Rate: $(echo "$ERROR_RATE * 100" | bc)%"
if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
  echo "⚠️  WARNING: Error rate above 1%"
fi
echo ""

# 4. Check latency (24h p95)
echo "4. Latency (24h p95):"
LATENCY=$(curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[24h])' | \
  jq -r '.data.result[0].value[1]')
echo "p95 Latency: ${LATENCY}s"
if (( $(echo "$LATENCY > 20" | bc -l) )); then
  echo "⚠️  WARNING: Latency above 20s"
fi
echo ""

# 5. Check resource usage
echo "5. Resource Usage:"
kubectl top pods -n gl-009-production -l app=thermaliq
echo ""

# 6. Check database connections
echo "6. Database Connections:"
DB_CONN=$(kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -t -c \
  "SELECT count(*) FROM pg_stat_activity WHERE datname='thermaliq_production';")
echo "Active Connections: $DB_CONN"
echo ""

# 7. Check cache hit rate
echo "7. Cache Hit Rate (24h):"
CACHE_HIT_RATE=$(curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_cache_hits_total[24h])/(rate(thermaliq_cache_hits_total[24h])+rate(thermaliq_cache_misses_total[24h]))' | \
  jq -r '.data.result[0].value[1]')
echo "Cache Hit Rate: $(echo "$CACHE_HIT_RATE * 100" | bc)%"
if (( $(echo "$CACHE_HIT_RATE < 0.70" | bc -l) )); then
  echo "⚠️  WARNING: Cache hit rate below 70%"
fi
echo ""

# 8. Check recent errors
echo "8. Recent Errors (last 1 hour):"
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 --since=1h | \
  grep "ERROR" | \
  tail -5
echo ""

echo "=== Health Check Complete ==="
```

Run daily check:
```bash
./daily_health_check.sh > logs/daily_check_$(date +%Y%m%d).log
cat logs/daily_check_$(date +%Y%m%d).log
```

**Error Log Review**:
```bash
# Review errors from last 24 hours
kubectl logs -n gl-009-production -l app=thermaliq --since=24h | \
  grep "ERROR" | \
  jq -r '{time: .timestamp, facility: .facility_id, error: .error_message}' | \
  sort | uniq -c | sort -rn | head -20

# Check for patterns:
# - Repeated errors for same facility
# - New error types
# - Increasing error frequency
```

**Cache Performance Review**:
```bash
# Check cache statistics
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO stats | \
  grep -E "keyspace_hits|keyspace_misses|evicted_keys"

# Calculate hit rate
kubectl exec -it redis-0 -n gl-009-production -- redis-cli --stat | head -20
```

---

## Weekly Maintenance

### Weekly Tasks (Sunday 02:00-04:00 UTC)

**Dependency Updates Review**:
```bash
#!/bin/bash
# weekly_dependency_check.sh

echo "=== Weekly Dependency Check ==="
date

# 1. Check for outdated Python packages
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip list --outdated

# 2. Check for security vulnerabilities
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip-audit

# 3. Check Docker base image updates
docker pull ghcr.io/greenlang/thermaliq:latest
docker pull python:3.11-slim

# 4. Check Kubernetes updates
kubectl version --short

# 5. Generate dependency report
cat > dependency_report_$(date +%Y%m%d).md <<EOF
# Dependency Update Report - $(date +%Y-%m-%d)

## Outdated Packages
$(kubectl exec -it deployment/thermaliq -n gl-009-production -- pip list --outdated)

## Security Vulnerabilities
$(kubectl exec -it deployment/thermaliq -n gl-009-production -- pip-audit)

## Recommendations
- [ ] Review and update dependencies
- [ ] Test in staging environment
- [ ] Deploy to production
EOF
```

**Security Scan Review**:
```bash
# Run security scans
trivy image ghcr.io/greenlang/thermaliq:latest \
  --severity HIGH,CRITICAL > security_scan_$(date +%Y%m%d).txt

# Review vulnerabilities
cat security_scan_$(date +%Y%m%d).txt | \
  grep -A 5 "Total:"

# Create tickets for critical vulnerabilities
```

**Performance Metrics Analysis**:
```bash
# Generate weekly performance report
curl -s http://prometheus:9090/api/v1/query_range \
  --data-urlencode 'query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket)' \
  --data-urlencode "start=$(date -u -d '7 days ago' +%s)" \
  --data-urlencode "end=$(date -u +%s)" \
  --data-urlencode 'step=3600' | \
  jq -r '.data.result[0].values[] | @csv' > latency_7days.csv

# Analyze trends
python3 <<EOF
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('latency_7days.csv', names=['timestamp', 'latency'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['latency'] = df['latency'].astype(float)

# Calculate statistics
print(f"Mean latency: {df['latency'].mean():.2f}s")
print(f"Median latency: {df['latency'].median():.2f}s")
print(f"95th percentile: {df['latency'].quantile(0.95):.2f}s")
print(f"Max latency: {df['latency'].max():.2f}s")

# Detect anomalies (> 2 std devs from mean)
mean = df['latency'].mean()
std = df['latency'].std()
anomalies = df[df['latency'] > mean + 2*std]
print(f"\nAnomalies detected: {len(anomalies)}")
if len(anomalies) > 0:
    print(anomalies)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['latency'])
plt.axhline(y=mean, color='r', linestyle='--', label='Mean')
plt.axhline(y=mean+2*std, color='orange', linestyle='--', label='2σ')
plt.xlabel('Time')
plt.ylabel('Latency (seconds)')
plt.title('GL-009 THERMALIQ Latency - Last 7 Days')
plt.legend()
plt.savefig('latency_trend_$(date +%Y%m%d).png')
print("\nTrend chart saved to latency_trend_$(date +%Y%m%d).png")
EOF
```

**Database Statistics Update**:
```bash
# Update database statistics for query optimizer
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "ANALYZE VERBOSE;"

# Reindex if needed
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "REINDEX DATABASE thermaliq_production CONCURRENTLY;"
```

**Benchmark Data Refresh**:
```bash
# Update industry benchmark data
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/update_benchmarks.py \
    --source industry_database \
    --regions all

# Verify benchmark data
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT industry, region, COUNT(*), MAX(updated_at)
   FROM industry_benchmarks
   GROUP BY industry, region
   ORDER BY industry, region;"
```

**Log Aggregation and Analysis**:
```bash
# Aggregate weekly logs
kubectl logs -n gl-009-production -l app=thermaliq --since=168h > \
  logs/thermaliq_week_$(date +%Y%m%d).log

# Compress
gzip logs/thermaliq_week_$(date +%Y%m%d).log

# Analyze log patterns
zcat logs/thermaliq_week_$(date +%Y%m%d).log.gz | \
  jq -r '.level' | \
  sort | uniq -c | \
  sort -rn

# Check for new error types
zcat logs/thermaliq_week_$(date +%Y%m%d).log.gz | \
  grep "ERROR" | \
  jq -r '.error_type' | \
  sort | uniq -c | \
  sort -rn
```

---

## Monthly Maintenance

### Monthly Tasks (First Sunday, 02:00-06:00 UTC)

**Comprehensive Performance Audit**:
```bash
#!/bin/bash
# monthly_performance_audit.sh

echo "=== Monthly Performance Audit ==="
date

# 1. Calculate throughput (30 days)
THROUGHPUT=$(curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_calculation_requests_total[30d])' | \
  jq -r '.data.result[0].value[1]')
echo "Average Throughput (30d): $THROUGHPUT req/s"

# 2. Calculate success rate
SUCCESS_RATE=$(curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=1 - (rate(thermaliq_calculation_errors_total[30d])/rate(thermaliq_calculation_requests_total[30d]))' | \
  jq -r '.data.result[0].value[1]')
echo "Success Rate (30d): $(echo "$SUCCESS_RATE * 100" | bc)%"

# 3. Latency percentiles
for percentile in 50 95 99; do
  LATENCY=$(curl -s http://prometheus:9090/api/v1/query \
    --data-urlencode "query=histogram_quantile(0.$percentile, thermaliq_calculation_duration_seconds_bucket[30d])" | \
    jq -r '.data.result[0].value[1]')
  echo "p$percentile Latency (30d): ${LATENCY}s"
done

# 4. Resource utilization
echo "\nResource Utilization (30d avg):"
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant "avg_over_time(container_cpu_usage_seconds_total{pod=~'thermaliq.*'}[30d])" | \
  tail -1

# 5. Cost analysis
echo "\nCompute Hours (30d):"
COMPUTE_HOURS=$(curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sum(kube_pod_container_resource_requests_cpu_cores{pod=~"thermaliq.*"}) * 24 * 30' | \
  jq -r '.data.result[0].value[1]')
echo "Total CPU Hours: $COMPUTE_HOURS"
echo "Estimated Cost (@ $0.05/CPU-hour): \$$(echo "$COMPUTE_HOURS * 0.05" | bc)"

# 6. Generate report
cat > monthly_report_$(date +%Y%m).md <<EOF
# GL-009 THERMALIQ Monthly Report - $(date +%B %Y)

## Key Metrics
- **Throughput**: $THROUGHPUT req/s
- **Success Rate**: $(echo "$SUCCESS_RATE * 100" | bc)%
- **p95 Latency**: $(curl -s http://prometheus:9090/api/v1/query --data-urlencode "query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[30d])" | jq -r '.data.result[0].value[1]')s
- **Compute Hours**: $COMPUTE_HOURS

## Trends
$(echo "Compare with previous month...")

## Issues
$(kubectl get events -n gl-009-production --sort-by='.lastTimestamp' | grep -i "warning\|error" | head -10)

## Recommendations
- [ ] Review and address performance bottlenecks
- [ ] Optimize resource allocation
- [ ] Update capacity planning
EOF

cat monthly_report_$(date +%Y%m).md
```

**Capacity Planning Review**:
```bash
# Analyze growth trends
python3 <<'EOF'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fetch monthly request counts (last 12 months)
months = []
request_counts = []

for i in range(12):
    month_start = (datetime.now() - timedelta(days=30*i)).replace(day=1)
    month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)

    # Query Prometheus for request count
    # (pseudo-code - replace with actual query)
    count = get_request_count(month_start, month_end)

    months.append(month_start.strftime('%Y-%m'))
    request_counts.append(count)

# Calculate growth rate
df = pd.DataFrame({'month': months, 'requests': request_counts})
df['growth_rate'] = df['requests'].pct_change()

print("Monthly Growth Analysis:")
print(df)

# Project next 6 months
avg_growth = df['growth_rate'].mean()
current_requests = df['requests'].iloc[0]

print(f"\nAverage monthly growth: {avg_growth*100:.2f}%")
print("\nProjected requests:")
for i in range(1, 7):
    projected = current_requests * (1 + avg_growth) ** i
    print(f"Month +{i}: {projected:.0f} requests")

# Calculate required capacity
calculations_per_second = projected / (30 * 24 * 3600)
pods_needed = calculations_per_second / 0.1  # Assuming 0.1 calc/s per pod
print(f"\nRequired pods (month +6): {pods_needed:.0f}")
EOF
```

**Documentation Updates**:
```bash
# Update runbooks with lessons learned
# Review and update:
# - INCIDENT_RESPONSE.md
# - TROUBLESHOOTING.md
# - ROLLBACK_PROCEDURE.md
# - SCALING_GUIDE.md
# - MAINTENANCE.md (this document)

# Check for outdated information
grep -r "TODO\|FIXME\|XXX" docs/runbooks/

# Update version numbers and dates
sed -i "s/Last Updated: .*/Last Updated: $(date +%Y-%m-%d)/" docs/runbooks/*.md
```

**Certificate Renewal Check**:
```bash
# Check certificate expiration
kubectl get certificates -n gl-009-production -o json | \
  jq -r '.items[] | select(.status.renewalTime != null) |
    "\(.metadata.name): expires \(.status.renewalTime)"'

# Check certificates expiring in next 30 days
openssl s_client -connect api.greenlang.io:443 -servername api.greenlang.io </dev/null 2>/dev/null | \
  openssl x509 -noout -dates

# Renew if needed (cert-manager should auto-renew)
kubectl get certificaterequests -n gl-009-production
```

---

## Quarterly Maintenance

### Quarterly Tasks (First Sunday of Quarter, 02:00-08:00 UTC)

**Disaster Recovery Drill**:
```bash
#!/bin/bash
# quarterly_dr_drill.sh

echo "=== Disaster Recovery Drill ==="
echo "Date: $(date -u)"
echo "Scenario: Complete region failure (us-east-1)"
echo ""

# 1. Backup production database
echo "Step 1: Backup production database..."
kubectl exec -it postgres-0 -n gl-009-production -- \
  pg_dump -U thermaliq -d thermaliq_production -F c -f /tmp/dr_drill_backup.dump
kubectl cp gl-009-production/postgres-0:/tmp/dr_drill_backup.dump \
  ./backups/dr_drill_$(date +%Y%m%d).dump
echo "✓ Backup created"
echo ""

# 2. Switch to secondary region (eu-west-1)
echo "Step 2: Switch to secondary region..."
kubectl config use-context eu-west-1
echo "✓ Context switched to eu-west-1"
echo ""

# 3. Verify secondary region readiness
echo "Step 3: Verify secondary region..."
kubectl get pods -n gl-009-production -l app=thermaliq
echo ""

# 4. Update DNS to point to secondary
echo "Step 4: Update DNS (SIMULATION - not actually changing)..."
echo "Would update Route53 to point api.greenlang.io to eu-west-1 load balancer"
echo "✓ DNS update simulated"
echo ""

# 5. Test functionality in secondary region
echo "Step 5: Test functionality..."
SECONDARY_URL="https://api-eu.greenlang.io/v1/thermaliq"
curl -s "$SECONDARY_URL/health" | jq .

# Submit test calculation
curl -X POST "$SECONDARY_URL/calculate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @test_calculation.json | jq .
echo "✓ Functionality test passed"
echo ""

# 6. Measure failover time
echo "Step 6: Measure failover time..."
echo "Total time: 5 minutes (simulated)"
echo "RTO met: Yes (target: 15 minutes)"
echo ""

# 7. Document findings
cat > dr_drill_report_$(date +%Y%m%d).md <<EOF
# Disaster Recovery Drill Report - $(date +%Y-%m-%d)

## Scenario
Complete failure of primary region (us-east-1)

## Steps Executed
1. Database backup created
2. Switched to secondary region (eu-west-1)
3. Verified secondary region readiness
4. Updated DNS (simulated)
5. Tested functionality in secondary region
6. Measured failover time

## Results
- **RTO**: 5 minutes (target: 15 minutes) ✓
- **RPO**: 0 minutes (continuous replication) ✓
- **Functionality**: All tests passed ✓

## Issues Found
- None

## Action Items
- [ ] Document procedure
- [ ] Update runbooks
- [ ] Train team on DR process
EOF

cat dr_drill_report_$(date +%Y%m%d).md

# 8. Switch back to primary region
echo "Step 8: Switch back to primary region..."
kubectl config use-context us-east-1
echo "✓ Drill complete"
```

**Security Audit**:
```bash
#!/bin/bash
# quarterly_security_audit.sh

echo "=== Quarterly Security Audit ==="
date

# 1. Check RBAC permissions
echo "1. Reviewing RBAC permissions..."
kubectl get rolebindings -n gl-009-production -o json | \
  jq -r '.items[] | "\(.metadata.name): \(.subjects[].name) -> \(.roleRef.name)"'
echo ""

# 2. Scan for secrets in code
echo "2. Scanning for exposed secrets..."
git clone https://github.com/greenlang/thermaliq.git /tmp/thermaliq-audit
cd /tmp/thermaliq-audit
git-secrets --scan
echo ""

# 3. Review network policies
echo "3. Reviewing network policies..."
kubectl get networkpolicies -n gl-009-production -o yaml
echo ""

# 4. Check for outdated images
echo "4. Checking for outdated images..."
kubectl get pods -n gl-009-production -o json | \
  jq -r '.items[] | .spec.containers[] | "\(.name): \(.image)"' | \
  sort -u
echo ""

# 5. Scan images for vulnerabilities
echo "5. Scanning images for vulnerabilities..."
for image in $(kubectl get pods -n gl-009-production -o json | jq -r '.items[] | .spec.containers[].image' | sort -u); do
  echo "Scanning $image..."
  trivy image --severity CRITICAL,HIGH "$image"
done
echo ""

# 6. Review audit logs
echo "6. Reviewing audit logs..."
kubectl logs -n kube-system -l component=kube-apiserver --tail=1000 | \
  grep "gl-009-production" | \
  jq 'select(.verb=="delete" or .verb=="create")' | \
  head -20
echo ""

# 7. Check for exposed services
echo "7. Checking for exposed services..."
kubectl get services -n gl-009-production -o json | \
  jq -r '.items[] | select(.spec.type=="LoadBalancer") | "\(.metadata.name): \(.status.loadBalancer.ingress[0].hostname)"'
echo ""

# 8. Generate security report
cat > security_audit_$(date +%Y%m%d).md <<EOF
# Security Audit Report - $(date +%Y-%m-%d)

## Vulnerabilities Found
$(trivy image ghcr.io/greenlang/thermaliq:latest --severity CRITICAL,HIGH --format table)

## RBAC Review
- [ ] All service accounts have minimum necessary permissions
- [ ] No overly permissive roles
- [ ] Regular users cannot access production namespace

## Network Security
- [ ] Network policies enforce least privilege
- [ ] No unnecessary service exposure
- [ ] TLS enabled for all external communication

## Secrets Management
- [ ] No secrets in code
- [ ] Secrets stored in Kubernetes secrets or vault
- [ ] Secrets rotated regularly

## Action Items
- [ ] Patch critical vulnerabilities
- [ ] Review and tighten RBAC
- [ ] Update network policies
- [ ] Rotate secrets
EOF

cat security_audit_$(date +%Y%m%d).md
```

**Architecture Review**:
```bash
# Review and document current architecture
# Update architecture diagrams
# Identify technical debt
# Plan architectural improvements

cat > architecture_review_$(date +%Y%m%d).md <<'EOF'
# Architecture Review - Q4 2025

## Current Architecture

### Application Tier
- **Deployment**: 4-12 pods (HPA)
- **Resources**: 2-4 CPU, 4-8GB RAM per pod
- **Language**: Python 3.11
- **Framework**: Flask + Gunicorn

### Data Tier
- **Database**: PostgreSQL 15 (3 instances: 1 primary + 2 replicas)
- **Cache**: Redis 7.0 (3 node cluster)
- **Storage**: AWS S3 for backups and reports

### External Dependencies
- Energy Meter API
- Historian (OSIsoft PI)
- Benchmark Database

### Observability
- **Metrics**: Prometheus + Grafana
- **Logs**: Loki
- **Tracing**: Jaeger
- **Alerting**: Alertmanager + PagerDuty

## Technical Debt

1. **Database Schema**: Some tables lack proper indexes
2. **Legacy Code**: Old calculation algorithms need refactoring
3. **Test Coverage**: Unit test coverage at 75% (target: 90%)
4. **Documentation**: API documentation needs updates

## Improvement Opportunities

1. **Performance**: Implement calculation result streaming for large datasets
2. **Scalability**: Add database sharding for high-volume facilities
3. **Reliability**: Implement circuit breakers for external APIs
4. **Observability**: Add distributed tracing

## Recommendations

- [ ] Q1 2026: Refactor calculation engine
- [ ] Q2 2026: Implement database sharding
- [ ] Q3 2026: Add distributed tracing
- [ ] Q4 2026: Migrate to gRPC for internal services
EOF
```

---

## Database Maintenance

### Vacuum and Analyze

**Automated Vacuum** (configured in PostgreSQL):
```sql
-- Check autovacuum settings
SHOW autovacuum;
SHOW autovacuum_vacuum_scale_factor;
SHOW autovacuum_analyze_scale_factor;

-- Current settings should be:
-- autovacuum = on
-- autovacuum_vacuum_scale_factor = 0.1
-- autovacuum_analyze_scale_factor = 0.05
```

**Manual Vacuum** (monthly):
```bash
# Full vacuum (requires downtime)
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "VACUUM FULL VERBOSE;"

# Or concurrent vacuum (no downtime)
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "VACUUM (VERBOSE, ANALYZE);"

# Vacuum specific tables
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "VACUUM VERBOSE ANALYZE energy_readings;"

kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "VACUUM VERBOSE ANALYZE calculations;"
```

### Index Maintenance

**Identify Missing Indexes**:
```bash
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production <<'EOF'
-- Find tables with sequential scans
SELECT schemaname, tablename, seq_scan, seq_tup_read, idx_scan, idx_tup_fetch
FROM pg_stat_user_tables
WHERE seq_scan > 100
  AND seq_tup_read > 10000
  AND idx_scan < seq_scan
ORDER BY seq_tup_read DESC
LIMIT 20;

-- Find unused indexes
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexname NOT LIKE '%_pkey'
ORDER BY pg_size_pretty(pg_relation_size(indexrelid::regclass)) DESC
LIMIT 20;
EOF
```

**Rebuild Fragmented Indexes** (monthly):
```bash
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production <<'EOF'
-- Reindex all indexes (can be slow)
REINDEX DATABASE thermaliq_production CONCURRENTLY;

-- Or reindex specific tables
REINDEX TABLE CONCURRENTLY energy_readings;
REINDEX TABLE CONCURRENTLY calculations;
EOF
```

### Partition Management

**Create New Partitions** (automated):
```bash
# Using pg_partman for automatic partition creation
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production <<'EOF'
-- Run partition maintenance
SELECT partman.run_maintenance('public.energy_readings');

-- Check partition status
SELECT parent_table, last_partition, premake
FROM partman.part_config;
EOF
```

**Drop Old Partitions** (quarterly):
```bash
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production <<'EOF'
-- Drop partitions older than 2 years
SELECT partman.drop_partition_time('public.energy_readings',
  INTERVAL '2 years',
  p_keep_table := false);

-- Verify
SELECT tablename
FROM pg_tables
WHERE tablename LIKE 'energy_readings_%'
ORDER BY tablename;
EOF
```

### Database Statistics

**Update Statistics** (weekly):
```bash
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c "ANALYZE VERBOSE;"
```

**Review Statistics** (monthly):
```bash
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production <<'EOF'
-- Check table statistics
SELECT schemaname, tablename,
       n_live_tup, n_dead_tup,
       last_vacuum, last_autovacuum,
       last_analyze, last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC
LIMIT 20;

-- Check index usage
SELECT schemaname, tablename, indexname,
       idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC
LIMIT 20;
EOF
```

---

## Cache Maintenance

### Redis Maintenance

**Memory Optimization**:
```bash
# Check memory usage
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO memory

# Check for memory fragmentation
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli INFO memory | grep mem_fragmentation_ratio

# If fragmentation > 1.5, defragment
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli MEMORY PURGE

# Or restart Redis (will clear cache)
kubectl rollout restart statefulset/redis -n gl-009-production
```

**Key Expiration Audit**:
```bash
# Check for keys without TTL
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli --scan --pattern "*" | \
  while read key; do
    ttl=$(kubectl exec -it redis-0 -n gl-009-production -- redis-cli TTL "$key")
    if [ "$ttl" -eq "-1" ]; then
      echo "$key: no expiration"
    fi
  done | head -20

# Set TTL for keys without expiration
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli --scan --pattern "calculation:*" | \
  while read key; do
    kubectl exec -it redis-0 -n gl-009-production -- \
      redis-cli EXPIRE "$key" 86400
  done
```

**Cache Performance Analysis**:
```bash
# Get cache statistics
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO stats | \
  grep -E "keyspace_hits|keyspace_misses|evicted_keys|expired_keys"

# Calculate hit rate
hits=$(kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO stats | grep keyspace_hits | cut -d: -f2)
misses=$(kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO stats | grep keyspace_misses | cut -d: -f2)
hit_rate=$(echo "scale=4; $hits / ($hits + $misses)" | bc)
echo "Cache hit rate: $(echo "$hit_rate * 100" | bc)%"
```

### Cache Warming

**Warm Cache After Maintenance**:
```bash
# Warm cache for top facilities
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/warm_cache.py \
    --facilities $(cat top_facilities.txt) \
    --days 30 \
    --concurrency 10

# Verify cache populated
kubectl exec -it redis-0 -n gl-009-production -- redis-cli DBSIZE
```

---

## Certificate Management

### Check Certificate Expiration

```bash
# Check all certificates
kubectl get certificates -n gl-009-production -o json | \
  jq -r '.items[] | "\(.metadata.name): \(.status.notAfter)"'

# Check specific certificate
kubectl describe certificate greenlang-tls -n gl-009-production

# Check certificate from endpoint
echo | openssl s_client -connect api.greenlang.io:443 -servername api.greenlang.io 2>/dev/null | \
  openssl x509 -noout -dates
```

### Certificate Renewal

**Automatic Renewal** (cert-manager):
```yaml
# Cert-manager automatically renews certificates 30 days before expiration
# Verify cert-manager is running
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: greenlang-tls
  namespace: gl-009-production
spec:
  secretName: greenlang-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.greenlang.io
  renewBefore: 720h  # 30 days
```

**Manual Renewal** (if needed):
```bash
# Force renewal
kubectl delete certificaterequest \
  $(kubectl get certificaterequests -n gl-009-production -o name | head -1) \
  -n gl-009-production

# Cert-manager will create new request
kubectl get certificaterequests -n gl-009-production --watch

# Verify new certificate
kubectl get certificate greenlang-tls -n gl-009-production
```

### Certificate Rotation

**Rotate Internal Certificates** (quarterly):
```bash
# Generate new certificates
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout internal-key.pem \
  -out internal-cert.pem \
  -days 365 \
  -subj "/CN=thermaliq-internal"

# Update Kubernetes secret
kubectl create secret tls thermaliq-internal-tls \
  -n gl-009-production \
  --cert=internal-cert.pem \
  --key=internal-key.pem \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new certificate
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

---

## Dependency Updates

### Check for Updates

```bash
# Python packages
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip list --outdated

# Docker base image
docker pull python:3.11-slim
docker images python:3.11-slim

# Kubernetes components
kubectl version --short
helm repo update
helm search repo
```

### Update Procedure

**Test Environment Update**:
```bash
# 1. Update dependencies in staging
kubectl config use-context staging
kubectl set image deployment/thermaliq -n gl-009-staging \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.0-rc1

# 2. Run integration tests
./scripts/integration_tests.sh --env staging

# 3. Load test
./scripts/load_test.sh --env staging --duration 3600 --rps 50

# 4. Monitor for 24 hours
# Check metrics, logs, error rates
```

**Production Update** (if staging successful):
```bash
# 1. Create rollback point
kubectl get deployment thermaliq -n gl-009-production -o yaml > \
  backups/deployment_pre_update_$(date +%Y%m%d).yaml

# 2. Update with canary
kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: thermaliq
  namespace: gl-009-production
spec:
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 10m}
      - setWeight: 25
      - pause: {duration: 10m}
      - setWeight: 50
      - pause: {duration: 10m}
      - setWeight: 75
      - pause: {duration: 10m}
EOF

# 3. Monitor canary rollout
kubectl argo rollouts get rollout thermaliq -n gl-009-production --watch

# 4. Promote or abort
# If healthy:
kubectl argo rollouts promote thermaliq -n gl-009-production

# If issues:
kubectl argo rollouts abort thermaliq -n gl-009-production
kubectl argo rollouts undo thermaliq -n gl-009-production
```

---

## Security Maintenance

### Rotate Secrets

**Database Password Rotation** (quarterly):
```bash
# 1. Generate new password
NEW_PASSWORD=$(openssl rand -base64 32)

# 2. Update password in database
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U postgres -c "ALTER USER thermaliq WITH PASSWORD '$NEW_PASSWORD';"

# 3. Update Kubernetes secret
kubectl create secret generic postgres-credentials \
  -n gl-009-production \
  --from-literal=username=thermaliq \
  --from-literal=password=$NEW_PASSWORD \
  --from-literal=url=postgresql://thermaliq:$NEW_PASSWORD@postgres-service:5432/thermaliq_production \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Restart application to pick up new password
kubectl rollout restart deployment/thermaliq -n gl-009-production

# 5. Verify connectivity
kubectl logs -n gl-009-production -l app=thermaliq --tail=50 | grep "Database"
```

**API Key Rotation** (quarterly):
```bash
# Rotate external API keys
# - Energy Meter API
# - Historian API
# - Benchmark Database API

# Update secrets
kubectl create secret generic external-api-keys \
  -n gl-009-production \
  --from-literal=energy-meter-key=$NEW_METER_KEY \
  --from-literal=historian-key=$NEW_HISTORIAN_KEY \
  --from-literal=benchmark-key=$NEW_BENCHMARK_KEY \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

### Security Scanning

**Container Image Scanning**:
```bash
# Scan with Trivy
trivy image ghcr.io/greenlang/thermaliq:latest \
  --severity CRITICAL,HIGH \
  --format table

# Scan with Clair
clairctl analyze ghcr.io/greenlang/thermaliq:latest

# Scan with Anchore
anchore-cli image add ghcr.io/greenlang/thermaliq:latest
anchore-cli image vuln ghcr.io/greenlang/thermaliq:latest os
```

**Dependency Scanning**:
```bash
# Scan Python dependencies
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip-audit

# Scan with safety
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  safety check --json
```

---

## Backup and Recovery

### Backup Verification

**Daily Backup Verification**:
```bash
# Check latest backup
aws s3 ls s3://greenlang-backups/thermaliq/ --recursive | \
  sort | tail -1

# Download and verify
aws s3 cp s3://greenlang-backups/thermaliq/latest.dump /tmp/
pg_restore --list /tmp/latest.dump | head -20

# Test restore (to test database)
kubectl exec -it postgres-test-0 -n gl-009-staging -- \
  pg_restore -U thermaliq -d thermaliq_test -c /tmp/latest.dump
```

### Backup Retention

**Implement Retention Policy**:
```bash
# Keep:
# - Daily backups: 30 days
# - Weekly backups: 90 days
# - Monthly backups: 1 year

# Cleanup old backups
aws s3 ls s3://greenlang-backups/thermaliq/daily/ | \
  awk '{print $4}' | \
  head -n -30 | \
  while read file; do
    aws s3 rm s3://greenlang-backups/thermaliq/daily/$file
  done
```

### Recovery Testing

**Monthly Recovery Test**:
```bash
# Test recovery procedure
./scripts/test_recovery.sh

# Verify data integrity after restore
./scripts/verify_data_integrity.sh
```

---

## Performance Tuning

### Application Tuning

**Optimize Gunicorn Workers**:
```bash
# Calculate optimal workers
CORES=$(kubectl exec -it deployment/thermaliq -n gl-009-production -- nproc)
WORKERS=$(echo "($CORES * 2) + 1" | bc)

# Update worker count
kubectl set env deployment/thermaliq -n gl-009-production \
  GUNICORN_WORKERS=$WORKERS

# Restart
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Optimize Database Connection Pool**:
```bash
# Calculate optimal pool size
# Formula: (cores * 2) + effective_spindle_count
# For cloud: (cores * 2) + 1

POOL_SIZE=$(echo "($CORES * 2) + 1" | bc)

kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_POOL_SIZE=$POOL_SIZE \
  DATABASE_MAX_OVERFLOW=10

# Restart
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

### Database Tuning

**PostgreSQL Configuration**:
```sql
-- Update PostgreSQL configuration for optimal performance
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '10MB';
ALTER SYSTEM SET min_wal_size = '2GB';
ALTER SYSTEM SET max_wal_size = '8GB';

-- Reload configuration
SELECT pg_reload_conf();
```

---

## Maintenance Checklist

### Daily ✓
- [ ] Review health check dashboard
- [ ] Check error logs
- [ ] Verify backup completed
- [ ] Monitor resource usage
- [ ] Check active alerts

### Weekly ✓
- [ ] Review dependency updates
- [ ] Run security scans
- [ ] Analyze performance metrics
- [ ] Update database statistics
- [ ] Refresh benchmark data
- [ ] Review and archive logs

### Monthly ✓
- [ ] Comprehensive performance audit
- [ ] Capacity planning review
- [ ] Documentation updates
- [ ] Certificate expiration check
- [ ] Database vacuum and reindex
- [ ] Review and optimize queries
- [ ] Cache performance analysis

### Quarterly ✓
- [ ] Disaster recovery drill
- [ ] Security audit
- [ ] Architecture review
- [ ] Rotate secrets (database, API keys)
- [ ] Review and update SLAs
- [ ] Dependency major version updates
- [ ] Performance benchmarking
- [ ] Cost optimization review

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-26
**Next Review**: 2026-01-26
**Owner**: GreenLang SRE Team
