# GL-008 SteamTrapInspector - Maintenance Runbook

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Owner:** Platform Operations Team

---

## Table of Contents

1. [Maintenance Overview](#maintenance-overview)
2. [Daily Checks](#daily-checks)
3. [Weekly Maintenance](#weekly-maintenance)
4. [Monthly Tasks](#monthly-tasks)
5. [Quarterly Reviews](#quarterly-reviews)
6. [Sensor Calibration](#sensor-calibration)
7. [ML Model Retraining](#ml-model-retraining)
8. [Database Maintenance](#database-maintenance)
9. [Security Updates](#security-updates)
10. [Backup & Recovery](#backup--recovery)
11. [Performance Optimization](#performance-optimization)
12. [Maintenance Windows](#maintenance-windows)

---

## Maintenance Overview

### Maintenance Philosophy

**Proactive > Reactive**
- Regular preventive maintenance reduces emergency incidents
- Scheduled maintenance windows minimize customer impact
- Automated checks catch issues before they become critical

### Maintenance Schedule Summary

| Frequency | Duration | Window | Tasks |
|-----------|----------|--------|-------|
| **Daily** | 15 min | Automated | Health checks, log review, metrics monitoring |
| **Weekly** | 1 hour | Sunday 2-3 AM UTC | Cache clearing, log rotation, sensor validation |
| **Monthly** | 2 hours | First Sunday 2-4 AM UTC | DB optimization, model evaluation, dependency updates |
| **Quarterly** | 4 hours | TBD with customers | Full system review, capacity planning, DR testing |

### Maintenance Team Roster

| Role | Primary | Backup | Escalation |
|------|---------|---------|------------|
| **Daily Checks** | On-call engineer | SRE team | Engineering Manager |
| **Weekly Tasks** | Platform Engineer | DevOps Engineer | VP Engineering |
| **Monthly Tasks** | Platform Team Lead | Senior Engineer | VP Engineering |
| **Quarterly Review** | Engineering Manager | CTO | Executive Team |

---

## Daily Checks

### Automated Daily Health Check

```bash
#!/bin/bash
# File: scripts/daily-health-check.sh
# Runs automatically at 9 AM UTC via CronJob

set -e

REPORT_DATE=$(date +"%Y-%m-%d")
REPORT_FILE="reports/daily-health-${REPORT_DATE}.txt"

exec > >(tee -a "$REPORT_FILE")
exec 2>&1

echo "=============================================="
echo "GL-008 Daily Health Check"
echo "Date: $REPORT_DATE"
echo "Time: $(date -u +"%H:%M:%S UTC")"
echo "=============================================="
echo ""

PASSED=0
WARNINGS=0
FAILED=0

# Check 1: Kubernetes Pods
echo "[1/12] Kubernetes Pod Status"
echo "----------------------------------------------"
EXPECTED_PODS=17
RUNNING_PODS=$(kubectl get pods -n greenlang-gl008 --field-selector=status.phase=Running -o name | wc -l)

if [ "$RUNNING_PODS" -eq "$EXPECTED_PODS" ]; then
  echo "✓ All pods running ($RUNNING_PODS/$EXPECTED_PODS)"
  ((PASSED++))
elif [ "$RUNNING_PODS" -ge $((EXPECTED_PODS - 2)) ]; then
  echo "⚠ Some pods not running ($RUNNING_PODS/$EXPECTED_PODS)"
  kubectl get pods -n greenlang-gl008 | grep -v Running || true
  ((WARNINGS++))
else
  echo "✗ Multiple pods not running ($RUNNING_PODS/$EXPECTED_PODS)"
  kubectl get pods -n greenlang-gl008
  ((FAILED++))
fi
echo ""

# Check 2: API Health Endpoint
echo "[2/12] API Health Endpoint"
echo "----------------------------------------------"
API_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" https://api.greenlang.io/v1/steam-trap/health)
if [ "$API_RESPONSE" -eq 200 ]; then
  echo "✓ API health check passed (HTTP 200)"
  ((PASSED++))
else
  echo "✗ API health check failed (HTTP $API_RESPONSE)"
  ((FAILED++))
fi
echo ""

# Check 3: Database Connectivity
echo "[3/12] Database Connectivity"
echo "----------------------------------------------"
if psql $DB_URL -c "SELECT 1;" > /dev/null 2>&1; then
  TRAP_COUNT=$(psql $DB_URL -t -c "SELECT COUNT(*) FROM traps;")
  INSPECTION_COUNT_24H=$(psql $DB_URL -t -c "SELECT COUNT(*) FROM trap_inspections WHERE detected_at > NOW() - INTERVAL '24 hours';")
  echo "✓ Database connection successful"
  echo "  Total traps: $TRAP_COUNT"
  echo "  Inspections (24h): $INSPECTION_COUNT_24H"
  ((PASSED++))
else
  echo "✗ Database connection failed"
  ((FAILED++))
fi
echo ""

# Check 4: Sensor Status
echo "[4/12] Sensor Status"
echo "----------------------------------------------"
SENSOR_STATUS=$(curl -s https://api.greenlang.io/v1/steam-trap/sensors/status)
ONLINE_SENSORS=$(echo "$SENSOR_STATUS" | jq -r '.online_sensors_count')
OFFLINE_SENSORS=$(echo "$SENSOR_STATUS" | jq -r '.offline_sensors_count')
TOTAL_SENSORS=$((ONLINE_SENSORS + OFFLINE_SENSORS))
OFFLINE_PCT=$(echo "scale=2; 100 * $OFFLINE_SENSORS / $TOTAL_SENSORS" | bc)

echo "Sensors online: $ONLINE_SENSORS"
echo "Sensors offline: $OFFLINE_SENSORS ($OFFLINE_PCT%)"

if (( $(echo "$OFFLINE_PCT < 5" | bc -l) )); then
  echo "✓ Sensor status healthy (<5% offline)"
  ((PASSED++))
elif (( $(echo "$OFFLINE_PCT < 15" | bc -l) )); then
  echo "⚠ Elevated sensor offline rate ($OFFLINE_PCT%)"
  ((WARNINGS++))
else
  echo "✗ High sensor offline rate ($OFFLINE_PCT%)"
  ((FAILED++))
fi
echo ""

# Check 5: Error Rate
echo "[5/12] Error Rate (Last 24 Hours)"
echo "----------------------------------------------"
ERROR_COUNT=$(kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --since=24h | grep ERROR | wc -l)
ERROR_RATE_PER_HOUR=$((ERROR_COUNT / 24))

echo "Total errors (24h): $ERROR_COUNT"
echo "Error rate: $ERROR_RATE_PER_HOUR errors/hour"

if [ "$ERROR_RATE_PER_HOUR" -lt 10 ]; then
  echo "✓ Error rate acceptable (<10/hour)"
  ((PASSED++))
elif [ "$ERROR_RATE_PER_HOUR" -lt 50 ]; then
  echo "⚠ Elevated error rate ($ERROR_RATE_PER_HOUR/hour)"
  ((WARNINGS++))
else
  echo "✗ High error rate ($ERROR_RATE_PER_HOUR/hour)"
  ((FAILED++))
fi
echo ""

# Check 6: API Latency
echo "[6/12] API Latency"
echo "----------------------------------------------"
LATENCY_METRICS=$(curl -s https://api.greenlang.io/v1/steam-trap/metrics/latency)
LATENCY_P50=$(echo "$LATENCY_METRICS" | jq -r '.p50_latency_ms')
LATENCY_P95=$(echo "$LATENCY_METRICS" | jq -r '.p95_latency_ms')
LATENCY_P99=$(echo "$LATENCY_METRICS" | jq -r '.p99_latency_ms')

echo "Latency P50: ${LATENCY_P50}ms"
echo "Latency P95: ${LATENCY_P95}ms"
echo "Latency P99: ${LATENCY_P99}ms"

if [ "$LATENCY_P95" -lt 1000 ]; then
  echo "✓ Latency within SLA (P95 < 1s)"
  ((PASSED++))
elif [ "$LATENCY_P95" -lt 2000 ]; then
  echo "⚠ Latency elevated (P95: ${LATENCY_P95}ms)"
  ((WARNINGS++))
else
  echo "✗ Latency exceeds SLA (P95: ${LATENCY_P95}ms)"
  ((FAILED++))
fi
echo ""

# Check 7: Inspection Success Rate
echo "[7/12] Inspection Success Rate"
echo "----------------------------------------------"
SUCCESS_RATE=$(curl -s https://api.greenlang.io/v1/steam-trap/metrics/success-rate | jq -r '.rate')
echo "Success rate: ${SUCCESS_RATE}%"

if (( $(echo "$SUCCESS_RATE > 95" | bc -l) )); then
  echo "✓ Success rate excellent (>95%)"
  ((PASSED++))
elif (( $(echo "$SUCCESS_RATE > 90" | bc -l) )); then
  echo "⚠ Success rate acceptable (90-95%)"
  ((WARNINGS++))
else
  echo "✗ Success rate below target (<90%)"
  ((FAILED++))
fi
echo ""

# Check 8: Database Size & Growth
echo "[8/12] Database Size & Growth"
echo "----------------------------------------------"
DB_SIZE=$(psql $DB_URL -t -c "SELECT pg_size_pretty(pg_database_size('greenlang'));")
DB_SIZE_BYTES=$(psql $DB_URL -t -c "SELECT pg_database_size('greenlang');")
echo "Current database size: $DB_SIZE"

# Compare to yesterday (if data available)
if [ -f "reports/daily-health-$(date -d "yesterday" +"%Y-%m-%d").txt" ]; then
  PREV_SIZE=$(grep "Current database size:" "reports/daily-health-$(date -d "yesterday" +"%Y-%m-%d").txt" | awk '{print $4}')
  echo "Previous size: $PREV_SIZE"
  echo "✓ Database size tracked"
  ((PASSED++))
else
  echo "ℹ No previous data for comparison"
  ((PASSED++))
fi
echo ""

# Check 9: Backup Status
echo "[9/12] Backup Status"
echo "----------------------------------------------"
LATEST_BACKUP=$(aws s3 ls s3://greenlang-backups/gl008/ --recursive | sort | tail -1)
if [ -n "$LATEST_BACKUP" ]; then
  BACKUP_DATE=$(echo "$LATEST_BACKUP" | awk '{print $1, $2}')
  BACKUP_AGE_HOURS=$(( ($(date +%s) - $(date -d "$BACKUP_DATE" +%s)) / 3600 ))
  echo "Latest backup: $BACKUP_DATE"
  echo "Backup age: ${BACKUP_AGE_HOURS} hours"

  if [ "$BACKUP_AGE_HOURS" -lt 26 ]; then
    echo "✓ Backup is recent (<26 hours old)"
    ((PASSED++))
  else
    echo "⚠ Backup is stale (>26 hours old)"
    ((WARNINGS++))
  fi
else
  echo "✗ No recent backup found"
  ((FAILED++))
fi
echo ""

# Check 10: Certificate Expiry
echo "[10/12] SSL Certificate Expiry"
echo "----------------------------------------------"
CERT_EXPIRY=$(echo | openssl s_client -servername api.greenlang.io -connect api.greenlang.io:443 2>/dev/null | openssl x509 -noout -enddate | cut -d= -f2)
CERT_EXPIRY_EPOCH=$(date -d "$CERT_EXPIRY" +%s)
DAYS_UNTIL_EXPIRY=$(( ($CERT_EXPIRY_EPOCH - $(date +%s)) / 86400 ))

echo "Certificate expires: $CERT_EXPIRY"
echo "Days until expiry: $DAYS_UNTIL_EXPIRY"

if [ "$DAYS_UNTIL_EXPIRY" -gt 30 ]; then
  echo "✓ Certificate valid for >30 days"
  ((PASSED++))
elif [ "$DAYS_UNTIL_EXPIRY" -gt 7 ]; then
  echo "⚠ Certificate expires in <30 days"
  ((WARNINGS++))
else
  echo "✗ Certificate expires soon (<7 days)"
  ((FAILED++))
fi
echo ""

# Check 11: Disk Space
echo "[11/12] Disk Space"
echo "----------------------------------------------"
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- df -h | head -2
DISK_USAGE=$(kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- df -h / | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$DISK_USAGE" -lt 70 ]; then
  echo "✓ Disk usage healthy (<70%)"
  ((PASSED++))
elif [ "$DISK_USAGE" -lt 85 ]; then
  echo "⚠ Disk usage elevated (${DISK_USAGE}%)"
  ((WARNINGS++))
else
  echo "✗ Disk usage critical (${DISK_USAGE}%)"
  ((FAILED++))
fi
echo ""

# Check 12: Security Vulnerabilities
echo "[12/12] Security Vulnerability Scan"
echo "----------------------------------------------"
# Run Trivy scan on latest image
VULN_COUNT=$(trivy image --severity HIGH,CRITICAL --format json greenlang/steam-trap-inspector:latest 2>/dev/null | jq '.Results[].Vulnerabilities | length' | awk '{s+=$1} END {print s}')

if [ "$VULN_COUNT" -lt 5 ]; then
  echo "✓ Low vulnerability count ($VULN_COUNT critical/high)"
  ((PASSED++))
elif [ "$VULN_COUNT" -lt 20 ]; then
  echo "⚠ Moderate vulnerabilities ($VULN_COUNT critical/high)"
  ((WARNINGS++))
else
  echo "✗ High vulnerability count ($VULN_COUNT critical/high)"
  ((FAILED++))
fi
echo ""

# Summary
echo "=============================================="
echo "DAILY HEALTH CHECK SUMMARY"
echo "=============================================="
echo "Passed:   $PASSED/12"
echo "Warnings: $WARNINGS/12"
echo "Failed:   $FAILED/12"
echo ""

if [ "$FAILED" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
  echo "✓ All checks passed - System healthy"
  EXIT_CODE=0
elif [ "$FAILED" -eq 0 ]; then
  echo "⚠ Some warnings - Review recommended"
  EXIT_CODE=1
else
  echo "✗ Critical issues detected - Immediate action required"
  EXIT_CODE=2
fi

echo ""
echo "Report saved to: $REPORT_FILE"
echo "=============================================="

# Send report via email/Slack if warnings or failures
if [ "$EXIT_CODE" -gt 0 ]; then
  ./scripts/send-health-report.sh "$REPORT_FILE" "$EXIT_CODE"
fi

exit $EXIT_CODE
```

### Manual Daily Review (5 minutes)

```bash
# Quick manual checks by on-call engineer

# 1. Review Grafana dashboards
open https://grafana.greenlang.io/d/gl008-overview

# 2. Check PagerDuty for any alerts
open https://greenlang.pagerduty.com/incidents

# 3. Review customer support tickets
open https://greenlang.zendesk.com/agent/filters/gl008

# 4. Scan Slack channels for issues
# #gl008-alerts, #gl008-support

# 5. Quick sanity test
curl https://api.greenlang.io/v1/steam-trap/health | jq '.'
```

---

## Weekly Maintenance

### Weekly Maintenance Script

```bash
#!/bin/bash
# File: scripts/weekly-maintenance.sh
# Runs every Sunday at 2 AM UTC

set -e

echo "=============================================="
echo "GL-008 Weekly Maintenance"
echo "Date: $(date +"%Y-%m-%d")"
echo "Time: $(date -u +"%H:%M:%S UTC")"
echo "=============================================="
echo ""

# Task 1: Log Rotation & Archive
echo "[1/8] Log Rotation & Archive"
echo "----------------------------------------------"
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --since=168h > logs/steam-trap-inspector-$(date +%Y%m%d).log
gzip logs/steam-trap-inspector-$(date +%Y%m%d).log

# Upload to S3
aws s3 cp logs/steam-trap-inspector-$(date +%Y%m%d).log.gz s3://greenlang-logs/gl008/

# Delete logs older than 90 days
find logs/ -name "*.log.gz" -mtime +90 -delete

echo "✓ Logs rotated and archived"
echo ""

# Task 2: Clear Application Cache
echo "[2/8] Clear Application Cache"
echo "----------------------------------------------"
kubectl exec -n greenlang-gl008 deployment/redis -- redis-cli --scan --pattern "steam-trap:*" | \
  xargs -L 100 kubectl exec -n greenlang-gl008 deployment/redis -- redis-cli DEL

CACHE_SIZE=$(kubectl exec -n greenlang-gl008 deployment/redis -- redis-cli DBSIZE)
echo "✓ Cache cleared. Current size: $CACHE_SIZE keys"
echo ""

# Task 3: Database Vacuum & Analyze
echo "[3/8] Database Vacuum & Analyze"
echo "----------------------------------------------"
psql $DB_URL -c "VACUUM ANALYZE trap_inspections;"
psql $DB_URL -c "VACUUM ANALYZE sensor_readings;"
psql $DB_URL -c "VACUUM ANALYZE traps;"

echo "✓ Database vacuumed and analyzed"
echo ""

# Task 4: Sensor Health Validation
echo "[4/8] Sensor Health Validation"
echo "----------------------------------------------"
SENSORS_NEEDING_CALIBRATION=$(psql $DB_URL -t -c "
  SELECT COUNT(*)
  FROM sensors
  WHERE last_calibrated_at < NOW() - INTERVAL '90 days'
    OR calibration_drift_score > 0.15;
")

echo "Sensors needing calibration: $SENSORS_NEEDING_CALIBRATION"

if [ "$SENSORS_NEEDING_CALIBRATION" -gt 0 ]; then
  echo "Creating calibration tickets..."
  psql $DB_URL -c "
    INSERT INTO maintenance_tickets (sensor_id, ticket_type, priority, created_at)
    SELECT
      sensor_id,
      'CALIBRATION_REQUIRED',
      CASE
        WHEN calibration_drift_score > 0.20 THEN 'HIGH'
        ELSE 'MEDIUM'
      END,
      NOW()
    FROM sensors
    WHERE last_calibrated_at < NOW() - INTERVAL '90 days'
      OR calibration_drift_score > 0.15;
  "
  echo "✓ Calibration tickets created"
else
  echo "✓ All sensors within calibration parameters"
fi
echo ""

# Task 5: Review ML Model Performance
echo "[5/8] ML Model Performance Review"
echo "----------------------------------------------"
python scripts/evaluate_model_performance.py --lookback-days=7

# Check if model accuracy has degraded
CURRENT_ACCURACY=$(curl -s https://api.greenlang.io/v1/steam-trap/ml/metrics | jq -r '.accuracy')
echo "Current model accuracy: $CURRENT_ACCURACY"

if (( $(echo "$CURRENT_ACCURACY < 0.90" | bc -l) )); then
  echo "⚠ Model accuracy below threshold. Scheduling retraining."
  # Trigger retraining job
  kubectl create job retrain-model-$(date +%s) --from=cronjob/ml-model-retraining -n greenlang-gl008
else
  echo "✓ Model performance within acceptable range"
fi
echo ""

# Task 6: Update Dependencies
echo "[6/8] Check for Dependency Updates"
echo "----------------------------------------------"
# Check for outdated npm packages
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- npm outdated || true

# Check for Python package updates
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- pip list --outdated || true

echo "ℹ Review dependency updates for security patches"
echo ""

# Task 7: Clean Up Old Data
echo "[7/8] Archive Old Inspection Data"
echo "----------------------------------------------"
# Archive inspections older than 2 years
ARCHIVED_COUNT=$(psql $DB_URL -t -c "
  WITH archived AS (
    INSERT INTO trap_inspections_archive
    SELECT *
    FROM trap_inspections
    WHERE detected_at < NOW() - INTERVAL '2 years'
    RETURNING id
  )
  SELECT COUNT(*) FROM archived;
")

if [ "$ARCHIVED_COUNT" -gt 0 ]; then
  psql $DB_URL -c "
    DELETE FROM trap_inspections
    WHERE detected_at < NOW() - INTERVAL '2 years';
  "
  echo "✓ Archived $ARCHIVED_COUNT old inspection records"
else
  echo "✓ No old records to archive"
fi
echo ""

# Task 8: Restart Pods to Clear Memory Leaks
echo "[8/8] Rolling Restart for Memory Management"
echo "----------------------------------------------"
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
kubectl rollout restart deployment/worker -n greenlang-gl008

echo "Waiting for rollout to complete..."
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008
kubectl rollout status deployment/worker -n greenlang-gl008

echo "✓ Rolling restart complete"
echo ""

echo "=============================================="
echo "Weekly Maintenance Complete"
echo "=============================================="
```

### Weekly Report Generation

```bash
#!/bin/bash
# Generate weekly operational report

WEEK_START=$(date -d "7 days ago" +"%Y-%m-%d")
WEEK_END=$(date +"%Y-%m-%d")

cat > reports/weekly-report-${WEEK_END}.md <<EOF
# GL-008 Weekly Operational Report
**Week Ending:** $WEEK_END

## Summary Metrics

### Inspections
\`\`\`sql
$(psql $DB_URL -c "
  SELECT
    COUNT(*) as total_inspections,
    COUNT(CASE WHEN status = 'COMPLETED' THEN 1 END) as completed,
    COUNT(CASE WHEN status = 'FAILED' THEN 1 END) as failed,
    ROUND(100.0 * COUNT(CASE WHEN status = 'COMPLETED' THEN 1 END) / COUNT(*), 2) as success_rate
  FROM trap_inspections
  WHERE detected_at BETWEEN '$WEEK_START' AND '$WEEK_END';
")
\`\`\`

### System Health
\`\`\`sql
$(psql $DB_URL -c "
  SELECT
    DATE(detected_at) as date,
    COUNT(*) as inspections,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration_sec
  FROM trap_inspections
  WHERE detected_at BETWEEN '$WEEK_START' AND '$WEEK_END'
    AND status = 'COMPLETED'
  GROUP BY DATE(detected_at)
  ORDER BY date;
")
\`\`\`

### Top Issues
- [ ] Issue 1: [Description]
- [ ] Issue 2: [Description]

### Action Items
- [ ] Action 1: [Owner] - [Due Date]
- [ ] Action 2: [Owner] - [Due Date]

EOF

echo "Weekly report generated: reports/weekly-report-${WEEK_END}.md"
```

---

## Monthly Tasks

### Monthly Maintenance Procedure

```bash
#!/bin/bash
# File: scripts/monthly-maintenance.sh
# Runs first Sunday of each month at 2 AM UTC

set -e

echo "=============================================="
echo "GL-008 Monthly Maintenance"
echo "Date: $(date +"%Y-%m-%d")"
echo "=============================================="
echo ""

# Task 1: Database Index Optimization
echo "[1/10] Database Index Optimization"
echo "----------------------------------------------"

# Identify unused indexes
psql $DB_URL -c "
  SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
  FROM pg_stat_user_indexes
  WHERE idx_scan = 0
    AND schemaname = 'public'
  ORDER BY pg_relation_size(indexrelid) DESC;
"

# Rebuild bloated indexes
psql $DB_URL -c "REINDEX DATABASE greenlang;"

echo "✓ Database indexes optimized"
echo ""

# Task 2: Full ML Model Evaluation
echo "[2/10] ML Model Evaluation"
echo "----------------------------------------------"

python scripts/comprehensive_model_eval.py \
  --lookback-days=30 \
  --test-sets=validation,production_sample \
  --output-report=reports/ml-model-eval-$(date +%Y%m).html

# Check if retraining is needed
if [ -f "reports/ml-model-eval-$(date +%Y%m).html" ]; then
  echo "✓ Model evaluation complete"
  echo "Report: reports/ml-model-eval-$(date +%Y%m).html"
else
  echo "⚠ Model evaluation failed"
fi
echo ""

# Task 3: Security Audit
echo "[3/10] Security Audit"
echo "----------------------------------------------"

# Scan all container images
for image in steam-trap-inspector api-gateway worker sensor-gateway ml-service; do
  echo "Scanning $image..."
  trivy image --severity HIGH,CRITICAL greenlang/$image:latest
done

# Check for expiring secrets/credentials
echo "Checking credential expiry..."
# Add credential expiry check logic

echo "✓ Security audit complete"
echo ""

# Task 4: Capacity Review
echo "[4/10] Capacity Review"
echo "----------------------------------------------"

# Generate capacity report
python scripts/capacity_analysis.py --output=reports/capacity-$(date +%Y%m).json

# Database growth analysis
psql $DB_URL -c "
  SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as bytes
  FROM pg_tables
  WHERE schemaname = 'public'
  ORDER BY bytes DESC
  LIMIT 10;
"

echo "✓ Capacity review complete"
echo ""

# Task 5: Dependency Updates
echo "[5/10] Apply Security Patches & Dependency Updates"
echo "----------------------------------------------"

# Update base images
docker pull python:3.11-slim
docker pull node:18-alpine
docker pull postgres:15

# Rebuild and deploy (to staging first)
echo "ℹ Manual review required before production deployment"
echo ""

# Task 6: Backup Verification
echo "[6/10] Backup Verification & Test Restore"
echo "----------------------------------------------"

# Find latest backup
LATEST_BACKUP=$(aws s3 ls s3://greenlang-backups/gl008/ --recursive | sort | tail -1 | awk '{print $4}')

echo "Latest backup: $LATEST_BACKUP"

# Test restore to staging environment
echo "Testing restore to staging..."
./scripts/test-restore.sh --backup="$LATEST_BACKUP" --target=staging

echo "✓ Backup verification complete"
echo ""

# Task 7: Configuration Drift Detection
echo "[7/10] Configuration Drift Detection"
echo "----------------------------------------------"

# Compare production config to IaC definitions
kubectl get configmap -n greenlang-gl008 -o yaml > /tmp/current-config.yaml
diff -u terraform/configmaps.yaml /tmp/current-config.yaml || echo "⚠ Configuration drift detected"

echo "✓ Configuration drift check complete"
echo ""

# Task 8: Performance Benchmarking
echo "[8/10] Performance Benchmarking"
echo "----------------------------------------------"

# Run performance benchmarks
./scripts/performance-benchmark.sh --duration=300 --output=reports/perf-$(date +%Y%m).json

echo "✓ Performance benchmarking complete"
echo ""

# Task 9: Log Analysis
echo "[9/10] Monthly Log Analysis"
echo "----------------------------------------------"

# Analyze error patterns
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --since=720h | \
  grep ERROR | \
  awk '{print $NF}' | \
  sort | \
  uniq -c | \
  sort -rn | \
  head -20 > reports/error-patterns-$(date +%Y%m).txt

echo "✓ Log analysis complete"
echo "Top errors saved to: reports/error-patterns-$(date +%Y%m).txt"
echo ""

# Task 10: Documentation Review
echo "[10/10] Documentation Review Reminder"
echo "----------------------------------------------"
echo "ℹ Monthly reminder: Review and update documentation"
echo "  - API documentation"
echo "  - Runbooks"
echo "  - Architecture diagrams"
echo "  - Onboarding guides"
echo ""

echo "=============================================="
echo "Monthly Maintenance Complete"
echo "=============================================="
```

### Monthly Metrics Report

```sql
-- File: scripts/monthly_metrics_report.sql
-- Generate comprehensive monthly metrics

\echo '=== GL-008 Monthly Metrics Report ==='
\echo ''

-- Inspection Statistics
\echo '--- Inspection Statistics ---'
SELECT
  COUNT(*) as total_inspections,
  COUNT(CASE WHEN status = 'COMPLETED' THEN 1 END) as completed,
  COUNT(CASE WHEN status = 'FAILED' THEN 1 END) as failed,
  ROUND(100.0 * COUNT(CASE WHEN status = 'COMPLETED' THEN 1 END) / COUNT(*), 2) as success_rate_pct,
  ROUND(AVG(EXTRACT(EPOCH FROM (updated_at - created_at)))) as avg_duration_sec
FROM trap_inspections
WHERE detected_at > DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  AND detected_at < DATE_TRUNC('month', CURRENT_DATE);

-- Daily Inspection Volume
\echo ''
\echo '--- Daily Inspection Volume ---'
SELECT
  DATE(detected_at) as date,
  COUNT(*) as inspections,
  COUNT(DISTINCT site_id) as active_sites
FROM trap_inspections
WHERE detected_at > DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  AND detected_at < DATE_TRUNC('month', CURRENT_DATE)
GROUP BY DATE(detected_at)
ORDER BY date;

-- False Positive Analysis
\echo ''
\echo '--- False Positive Analysis ---'
SELECT
  DATE_TRUNC('week', detected_at) as week,
  COUNT(*) as total_failures,
  SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
  ROUND(100.0 * SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) / COUNT(*), 2) as fp_rate_pct
FROM trap_inspections
WHERE detected_at > DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  AND detected_at < DATE_TRUNC('month', CURRENT_DATE)
  AND status = 'FAILED'
  AND verified_status IS NOT NULL
GROUP BY DATE_TRUNC('week', detected_at)
ORDER BY week;

-- Top Sites by Activity
\echo ''
\echo '--- Top 10 Sites by Inspection Volume ---'
SELECT
  s.site_name,
  s.site_id,
  COUNT(*) as inspections,
  COUNT(DISTINCT t.trap_id) as unique_traps
FROM trap_inspections ti
JOIN traps t ON ti.trap_id = t.id
JOIN sites s ON t.site_id = s.id
WHERE ti.detected_at > DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  AND ti.detected_at < DATE_TRUNC('month', CURRENT_DATE)
GROUP BY s.site_name, s.site_id
ORDER BY inspections DESC
LIMIT 10;

-- Energy Savings Calculated
\echo ''
\echo '--- Energy Savings Summary ---'
SELECT
  COUNT(DISTINCT ti.trap_id) as traps_with_failures,
  ROUND(SUM(ec.energy_loss_kwh)) as total_energy_loss_kwh,
  ROUND(SUM(ec.energy_loss_kwh) * 0.12) as estimated_cost_savings_usd
FROM trap_inspections ti
JOIN energy_calculations ec ON ti.id = ec.inspection_id
WHERE ti.detected_at > DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  AND ti.detected_at < DATE_TRUNC('month', CURRENT_DATE)
  AND ti.status = 'FAILED'
  AND ti.verified_status != 'FALSE_POSITIVE';

-- Sensor Health
\echo ''
\echo '--- Sensor Health Summary ---'
SELECT
  sensor_type,
  COUNT(*) as total_sensors,
  SUM(CASE WHEN status = 'ONLINE' THEN 1 ELSE 0 END) as online,
  SUM(CASE WHEN status = 'OFFLINE' THEN 1 ELSE 0 END) as offline,
  SUM(CASE WHEN status = 'DEGRADED' THEN 1 ELSE 0 END) as degraded,
  ROUND(100.0 * SUM(CASE WHEN status = 'ONLINE' THEN 1 ELSE 0 END) / COUNT(*), 2) as uptime_pct
FROM sensors
GROUP BY sensor_type;
```

---

## Quarterly Reviews

### Quarterly Business Review Template

```markdown
# GL-008 Quarterly Business Review - Q[X] [YEAR]

**Review Date:** [Date]
**Attendees:** [List]
**Period:** [Start Date] - [End Date]

## Executive Summary

### Key Highlights
- [Highlight 1]
- [Highlight 2]
- [Highlight 3]

### Metrics Summary
| Metric | Q[X] | Q[X-1] | Change |
|--------|------|--------|--------|
| Total Inspections | [N] | [N] | [+/-N%] |
| Sites Deployed | [N] | [N] | [+/-N%] |
| System Uptime | [N%] | [N%] | [+/-N%] |
| Avg Response Time | [Nms] | [Nms] | [+/-N%] |
| Customer Satisfaction | [N/10] | [N/10] | [+/-N] |

## Operational Performance

### Reliability
- **System Uptime:** [99.X%]
- **Incident Count:** [N] (P0: [N], P1: [N], P2: [N])
- **MTTR:** [N] minutes
- **MTBF:** [N] days

### Performance
- **Inspection Throughput:** [N]/day
- **API Latency P95:** [N]ms
- **Database Performance:** [Description]

### Cost Efficiency
- **Infrastructure Cost:** $[N]/month
- **Cost per Inspection:** $[N]
- **Cost Trend:** [Up/Down/Stable] [N%]

## Technical Improvements

### Completed Initiatives
1. [Initiative 1] - [Impact]
2. [Initiative 2] - [Impact]
3. [Initiative 3] - [Impact]

### Challenges & Resolutions
1. **Challenge:** [Description]
   **Resolution:** [Description]
   **Outcome:** [Description]

## Customer Impact

### New Deployments
- [Customer 1]: [N] sites, [N] traps
- [Customer 2]: [N] sites, [N] traps

### Customer Feedback
- **Positive:** [Summary]
- **Areas for Improvement:** [Summary]

### Support Metrics
- **Total Tickets:** [N]
- **Avg Resolution Time:** [N] hours
- **Customer Satisfaction:** [N/10]

## Roadmap Progress

### Completed Features
- [Feature 1]
- [Feature 2]

### In Progress
- [Feature 1] - [X%] complete
- [Feature 2] - [X%] complete

### Planned for Next Quarter
- [Feature 1]
- [Feature 2]

## Risk Assessment

### Current Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| [Risk 1] | [H/M/L] | [H/M/L] | [Strategy] |
| [Risk 2] | [H/M/L] | [H/M/L] | [Strategy] |

## Action Items

| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| [Action 1] | [Name] | [Date] | [H/M/L] |
| [Action 2] | [Name] | [Date] | [H/M/L] |

## Appendix

### Detailed Metrics
[Attach detailed reports]

### Architecture Changes
[Document any significant architecture changes]
```

### Disaster Recovery Test

```bash
#!/bin/bash
# Quarterly DR test procedure

echo "=== GL-008 Disaster Recovery Test ==="
echo "⚠️  This is a TEST. No production impact."
echo ""

# Scenario: Complete datacenter failure

# Step 1: Verify secondary region readiness
echo "[1/6] Verifying secondary region infrastructure..."
kubectl get nodes --context=us-west-2 | grep Ready

# Step 2: Restore database to secondary region
echo "[2/6] Testing database restore..."
./scripts/restore-database.sh \
  --backup=latest \
  --target=dr-test-db \
  --region=us-west-2

# Step 3: Deploy application to secondary region
echo "[3/6] Deploying application to secondary region..."
kubectl apply -f k8s/deployment.yaml --context=us-west-2

# Step 4: Run smoke tests
echo "[4/6] Running smoke tests..."
./scripts/smoke-test.sh --target=https://dr-test.greenlang.io

# Step 5: Measure RTO/RPO
echo "[5/6] Measuring Recovery Objectives..."
echo "RTO (Recovery Time Objective): [Calculate]"
echo "RPO (Recovery Point Objective): [Calculate]"

# Step 6: Cleanup
echo "[6/6] Cleaning up DR test resources..."
kubectl delete -f k8s/deployment.yaml --context=us-west-2
./scripts/delete-test-database.sh --db=dr-test-db

echo ""
echo "✓ DR test complete"
echo "Document results in: reports/dr-test-Q[X]-[YEAR].md"
```

---

## Sensor Calibration

### Automated Calibration Scheduling

```python
#!/usr/bin/env python3
# File: scripts/schedule_sensor_calibration.py

import psycopg2
from datetime import datetime, timedelta

def schedule_calibration_for_overdue_sensors():
    """
    Identify sensors needing calibration and create work orders
    """
    conn = psycopg2.connect(os.environ['DB_URL'])
    cur = conn.cursor()

    # Find sensors needing calibration
    cur.execute("""
        SELECT
            sensor_id,
            sensor_type,
            site_id,
            last_calibrated_at,
            calibration_drift_score,
            CASE
                WHEN calibration_drift_score > 0.25 OR
                     last_calibrated_at < NOW() - INTERVAL '120 days' THEN 'URGENT'
                WHEN calibration_drift_score > 0.15 OR
                     last_calibrated_at < NOW() - INTERVAL '90 days' THEN 'HIGH'
                ELSE 'MEDIUM'
            END as priority
        FROM sensors
        WHERE last_calibrated_at < NOW() - INTERVAL '90 days'
           OR calibration_drift_score > 0.15
        ORDER BY priority, calibration_drift_score DESC;
    """)

    sensors = cur.fetchall()

    print(f"Found {len(sensors)} sensors needing calibration")

    # Create calibration work orders
    for sensor in sensors:
        sensor_id, sensor_type, site_id, last_cal, drift, priority = sensor

        cur.execute("""
            INSERT INTO calibration_work_orders
            (sensor_id, sensor_type, site_id, priority, scheduled_date, status, created_at)
            VALUES (%s, %s, %s, %s, %s, 'SCHEDULED', NOW())
            ON CONFLICT (sensor_id, status) WHERE status = 'SCHEDULED'
            DO NOTHING
            RETURNING id;
        """, (
            sensor_id,
            sensor_type,
            site_id,
            priority,
            datetime.now() + timedelta(days=7 if priority == 'URGENT' else 14)
        ))

        if cur.rowcount > 0:
            work_order_id = cur.fetchone()[0]
            print(f"Created work order {work_order_id} for sensor {sensor_id} (Priority: {priority})")

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    schedule_calibration_for_overdue_sensors()
```

### On-Site Calibration Procedure

```markdown
# On-Site Sensor Calibration Procedure

## Pre-Calibration Checklist
- [ ] Work order created in system
- [ ] Calibration equipment prepared:
  - [ ] Reference acoustic source (1250 Hz ± 5 Hz)
  - [ ] Reference thermal camera (NIST-traceable)
  - [ ] Pressure gauge (±1% accuracy)
- [ ] Site safety approval obtained
- [ ] Baseline readings recorded

## Acoustic Sensor Calibration

### Step 1: Position Reference Source
1. Place reference acoustic source 12 inches from sensor
2. Set frequency to 1250 Hz
3. Set amplitude to 85 dB

### Step 2: Record Baseline
```bash
curl -X GET https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/reading
# Record: frequency_measured, amplitude_measured
```

### Step 3: Calculate Offset
```python
offset_freq = 1250 - frequency_measured
offset_amp = 85 - amplitude_measured
```

### Step 4: Apply Calibration
```bash
curl -X POST https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/calibrate \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "frequency_offset": '$offset_freq',
    "amplitude_offset": '$offset_amp',
    "calibrated_by": "Technician Name",
    "reference_equipment": "Acoustic Calibrator Model XYZ"
  }'
```

### Step 5: Verify Calibration
```bash
# Wait 60 seconds
sleep 60

# Take new reading
curl -X GET https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/reading

# Verify: frequency_measured ≈ 1250 Hz (±2%), amplitude_measured ≈ 85 dB (±1 dB)
```

## Thermal Camera Calibration

[Similar procedure for thermal sensors]

## Post-Calibration
- [ ] Update calibration record in system
- [ ] Attach calibration certificate
- [ ] Schedule next calibration (90 days)
```

---

## ML Model Retraining

### Automated Retraining Schedule

```yaml
# File: k8s/cronjob-ml-retraining.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: ml-model-retraining
  namespace: greenlang-gl008
spec:
  schedule: "0 2 1 * *"  # First day of each month at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: model-retraining
            image: greenlang/ml-trainer:latest
            env:
            - name: DB_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
            - name: S3_MODEL_BUCKET
              value: "s3://greenlang-ml-models/steam-trap-inspector/"
            - name: MLFLOW_TRACKING_URI
              value: "https://mlflow.greenlang.io"
            command:
            - /bin/bash
            - -c
            - |
              set -e

              echo "Starting ML model retraining..."

              # Extract training data
              python scripts/extract_training_data.py \
                --lookback-days=180 \
                --min-confidence=0.8 \
                --include-verified-only \
                --output=/data/training_data.csv

              # Train model
              python scripts/train_model.py \
                --input=/data/training_data.csv \
                --model-type=random_forest \
                --hyperparameter-tuning=bayesian \
                --cross-validation=5 \
                --output=/models/steam_trap_$(date +%Y%m%d).pkl

              # Evaluate model
              python scripts/evaluate_model.py \
                --model=/models/steam_trap_$(date +%Y%m%d).pkl \
                --test-data=/data/validation_data.csv \
                --min-accuracy=0.92 \
                --min-precision=0.90 \
                --min-recall=0.90

              # If evaluation passes, upload to S3
              aws s3 cp /models/steam_trap_$(date +%Y%m%d).pkl \
                $S3_MODEL_BUCKET/steam_trap_$(date +%Y%m%d).pkl

              # Register in MLflow
              python scripts/register_model.py \
                --model-path=/models/steam_trap_$(date +%Y%m%d).pkl \
                --model-name=steam-trap-inspector \
                --stage=staging

              echo "Model retraining complete"
          restartPolicy: OnFailure
```

### Model Deployment Approval Process

```bash
#!/bin/bash
# Manual approval and deployment of new ML model

MODEL_VERSION="20251126"
MODEL_PATH="s3://greenlang-ml-models/steam-trap-inspector/steam_trap_${MODEL_VERSION}.pkl"

echo "=== ML Model Deployment Approval ==="
echo "Model Version: $MODEL_VERSION"
echo ""

# Step 1: Review model metrics
echo "[1/5] Review Model Metrics"
python scripts/model_report.py --model-version=$MODEL_VERSION

echo ""
read -p "Metrics acceptable? (yes/no): " METRICS_OK
if [ "$METRICS_OK" != "yes" ]; then
  echo "Deployment cancelled"
  exit 1
fi

# Step 2: Deploy to staging
echo "[2/5] Deploying to Staging"
kubectl set env deployment/steam-trap-inspector-staging \
  ML_MODEL_VERSION=$MODEL_VERSION \
  -n greenlang-staging

kubectl rollout status deployment/steam-trap-inspector-staging -n greenlang-staging

# Step 3: Run staging tests
echo "[3/5] Running Staging Tests (24 hours)"
echo "Monitor: https://grafana.greenlang.io/d/gl008-ml-staging"
echo ""
read -p "Press ENTER when ready to proceed to production..."

# Step 4: Production deployment
echo "[4/5] Deploying to Production"
kubectl set env deployment/steam-trap-inspector \
  ML_MODEL_VERSION=$MODEL_VERSION \
  -n greenlang-gl008

kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008

# Step 5: Monitor production
echo "[5/5] Monitor production for 2 hours"
./scripts/monitor-model-performance.sh --duration=2h --alert-threshold=0.90

echo ""
echo "✓ Model deployment complete"
```

---

## Database Maintenance

### Weekly Database Maintenance

```sql
-- File: scripts/weekly_db_maintenance.sql

-- 1. Update table statistics
ANALYZE VERBOSE trap_inspections;
ANALYZE VERBOSE sensor_readings;
ANALYZE VERBOSE traps;
ANALYZE VERBOSE sites;
ANALYZE VERBOSE energy_calculations;

-- 2. Vacuum to reclaim space
VACUUM (VERBOSE, ANALYZE) trap_inspections;
VACUUM (VERBOSE, ANALYZE) sensor_readings;

-- 3. Check for table bloat
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
  n_live_tup,
  n_dead_tup,
  ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_tuple_pct
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND n_live_tup > 0
ORDER BY dead_tuple_pct DESC;

-- 4. Reindex if necessary (high bloat)
-- REINDEX TABLE CONCURRENTLY trap_inspections;

-- 5. Check for long-running queries
SELECT
  pid,
  now() - query_start as duration,
  state,
  query
FROM pg_stat_activity
WHERE state != 'idle'
  AND query NOT LIKE '%pg_stat_activity%'
  AND now() - query_start > interval '5 minutes'
ORDER BY duration DESC;

-- 6. Check connection usage
SELECT
  count(*),
  state
FROM pg_stat_activity
GROUP BY state;

-- 7. Database size tracking
SELECT
  pg_size_pretty(pg_database_size('greenlang')) as total_size,
  (SELECT pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))::bigint)
   FROM pg_tables
   WHERE schemaname = 'public') as public_schema_size;
```

### Database Performance Tuning

```sql
-- Optimize PostgreSQL configuration for GL-008 workload

-- Increase shared buffers (25% of RAM)
ALTER SYSTEM SET shared_buffers = '8GB';

-- Optimize work memory for complex queries
ALTER SYSTEM SET work_mem = '64MB';

-- Increase maintenance work memory for VACUUM/INDEX operations
ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- Optimize checkpoint settings
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;

-- Query planner
ALTER SYSTEM SET random_page_cost = 1.1;  # For SSD storage
ALTER SYSTEM SET effective_cache_size = '24GB';

-- Reload configuration
SELECT pg_reload_conf();
```

---

## Security Updates

### Security Patch Procedure

```bash
#!/bin/bash
# Apply security patches to GL-008

echo "=== GL-008 Security Patch Procedure ==="

# Step 1: Scan for vulnerabilities
echo "[1/5] Scanning for vulnerabilities..."
trivy image --severity HIGH,CRITICAL greenlang/steam-trap-inspector:latest > vuln-report.txt

# Step 2: Review vulnerability report
cat vuln-report.txt

echo ""
read -p "Continue with patching? (yes/no): " CONTINUE
if [ "$CONTINUE" != "yes" ]; then
  exit 0
fi

# Step 3: Update dependencies
echo "[2/5] Updating dependencies..."
cd docker/steam-trap-inspector
npm audit fix
pip install --upgrade -r requirements.txt

# Step 4: Build new image
echo "[3/5] Building updated image..."
VERSION_NEW="v2.4.3-security-patch-$(date +%Y%m%d)"
docker build -t greenlang/steam-trap-inspector:$VERSION_NEW .

# Step 5: Test updated image
echo "[4/5] Testing updated image..."
docker run --rm greenlang/steam-trap-inspector:$VERSION_NEW pytest

# Step 6: Deploy to staging first
echo "[5/5] Deploying to staging..."
kubectl set image deployment/steam-trap-inspector-staging \
  steam-trap-inspector=greenlang/steam-trap-inspector:$VERSION_NEW \
  -n greenlang-staging

echo ""
echo "✓ Security patch applied to staging"
echo "Monitor staging for 24 hours before production deployment"
```

---

## Backup & Recovery

### Automated Backup Schedule

```yaml
# File: k8s/cronjob-database-backup.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: greenlang-gl008
spec:
  schedule: "0 3 * * *"  # Daily at 3 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: pg-dump
            image: postgres:15
            env:
            - name: PGHOST
              value: "postgresql.database.svc.cluster.local"
            - name: PGDATABASE
              value: "greenlang"
            - name: PGUSER
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: username
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: password
            command:
            - /bin/bash
            - -c
            - |
              BACKUP_FILE="greenlang-gl008-$(date +%Y%m%d-%H%M%S).sql.gz"

              echo "Creating backup: $BACKUP_FILE"

              pg_dump | gzip > /backups/$BACKUP_FILE

              echo "Uploading to S3..."
              aws s3 cp /backups/$BACKUP_FILE s3://greenlang-backups/gl008/$BACKUP_FILE

              echo "Backup complete: $BACKUP_FILE"

              # Cleanup backups older than 30 days
              aws s3 ls s3://greenlang-backups/gl008/ | \
                awk '{print $4}' | \
                while read file; do
                  if [ $(( ($(date +%s) - $(date -d "$(echo $file | cut -d'-' -f3 | cut -d'.' -f1)" +%s)) / 86400 )) -gt 30 ]; then
                    echo "Deleting old backup: $file"
                    aws s3 rm s3://greenlang-backups/gl008/$file
                  fi
                done
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
```

---

## Performance Optimization

### Query Optimization

```bash
# Identify and optimize slow queries monthly

psql $DB_URL <<EOF
-- Find slow queries
SELECT
  substring(query, 1, 100) as query_snippet,
  calls,
  mean_exec_time,
  max_exec_time,
  total_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Add missing indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_inspections_site_date
ON trap_inspections(site_id, DATE(detected_at));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensors_last_reading
ON sensors(last_reading_at DESC) WHERE status = 'ONLINE';
EOF
```

---

## Maintenance Windows

### Scheduled Maintenance Communication

```
Subject: Scheduled Maintenance - GL-008 Steam Trap Inspector

Dear Valued Customer,

We will be performing scheduled maintenance on the GL-008 Steam Trap Inspector platform.

**Maintenance Window:**
- Date: [Date]
- Time: 2:00 AM - 4:00 AM UTC ([Local Time])
- Expected Duration: 2 hours

**Impact:**
- Steam trap inspections will be unavailable during the maintenance window
- Historical data and reports will remain accessible
- Scheduled inspections will resume automatically after maintenance

**Work Being Performed:**
- Database optimization
- Security patches
- Performance improvements

No action is required on your part. If you have any questions, please contact support@greenlang.io.

Thank you for your patience.

GreenLang Operations Team
```

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-11-26
**Next Review:** 2026-02-26
**Maintained By:** Platform Operations Team
