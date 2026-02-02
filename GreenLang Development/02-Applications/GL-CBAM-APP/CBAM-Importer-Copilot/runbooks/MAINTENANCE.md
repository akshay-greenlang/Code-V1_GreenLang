# CBAM IMPORTER COPILOT - MAINTENANCE GUIDE

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Owner:** GreenLang CBAM Operations Team

---

## PURPOSE

This guide defines routine maintenance procedures for the CBAM Importer Copilot to ensure system reliability, data accuracy, regulatory compliance, and optimal performance for EU CBAM quarterly reporting.

---

## DAILY MAINTENANCE

### 1. Health Monitoring (10 minutes)

**Schedule:** Every business day, 09:00 CET

```bash
# 1. Check system health
curl http://cbam-importer:8000/health | jq '.status'
# Expected: "healthy"

# 2. Check all agents healthy
curl http://cbam-importer:8000/health/agents | jq '.agents[] | select(.status != "healthy")'
# Expected: No output (all agents healthy)

# 3. Check pod status
kubectl get pods -n greenlang -l app=cbam-importer
# Expected: All pods Running, READY 1/1

# 4. Review error logs (last 24 hours)
kubectl logs -n greenlang deployment/cbam-importer --since=24h | grep -i error | wc -l
# Expected: <10 errors per day

# 5. Check processing success rate
curl http://cbam-importer:8001/metrics | grep cbam_pipeline_success_rate
# Expected: >95%
```

**Action Items:**
- If health check fails → Follow INCIDENT_RESPONSE.md (P1)
- If error count >10 → Review error logs, identify patterns
- If success rate <95% → Investigate with TROUBLESHOOTING.md

### 2. Data Quality Monitoring (10 minutes)

```bash
# 1. Check validation error rate
curl http://cbam-importer:8001/metrics | grep cbam_validation_error_rate
# Expected: <5%

# 2. Review recent validation errors
curl http://cbam-importer:8000/reports/validation-errors?since=24h | jq '.top_errors[0:5]'

# 3. Check emissions calculation consistency
python scripts/daily_calculation_audit.py \
  --date $(date -d "yesterday" +%Y-%m-%d) \
  --output /reports/daily_audit_$(date +%Y%m%d).json

# Expected: All calculations match expected values

# 4. Monitor supplier data quality scores
curl http://cbam-importer:8000/reports/supplier-quality | \
  jq '.suppliers[] | select(.data_quality != "high")'
```

**Action Items:**
- Validation error rate >5% → Review input data quality with customers
- Calculation inconsistencies → Follow INCIDENT_RESPONSE.md (P0 - compliance risk)
- Low supplier data quality → Contact suppliers for data updates

### 3. Compliance Monitoring (5 minutes)

```bash
# 1. Check CBAM regulation compliance
curl http://cbam-importer:8001/metrics | grep cbam_compliance_validation_failures
# Expected: 0

# 2. Verify emission factor database current
grep "vintage: $(date +%Y)" data/emission_factors.py | wc -l
# Expected: >0 (factors from current year available)

# 3. Check CN code database version
jq '.metadata.version' data/cn_codes.json
# Expected: Current year version

# 4. Review upcoming reporting deadlines
python scripts/check_deadlines.py --days-ahead 30
```

**Action Items:**
- Compliance failures → Immediate investigation (P0)
- Emission factors outdated → Schedule update
- CN codes outdated → Update from EU source
- Deadline within 7 days → Notify customers, scale infrastructure

---

## WEEKLY MAINTENANCE

### 1. Performance Review (30 minutes)

**Schedule:** Every Monday, 10:00 CET

```bash
# 1. Review Grafana dashboards
# - CBAM Operations Dashboard
# - CBAM Performance Dashboard
# - CBAM Data Quality Dashboard

# 2. Check throughput trends
curl http://prometheus:9090/api/v1/query?query=rate(cbam_shipments_processed_total[7d])

# 3. Check latency percentiles
curl http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95, cbam_pipeline_duration_seconds)

# 4. Check error rate trends
curl http://prometheus:9090/api/v1/query?query=rate(cbam_errors_total[7d])

# 5. Resource utilization
kubectl top pods -n greenlang -l app=cbam-importer --containers

# 6. Database performance
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SELECT query, calls, mean_exec_time, total_exec_time
           FROM pg_stat_statements
           ORDER BY total_exec_time DESC LIMIT 10;"
```

**Action Items:**
- Throughput declining → Investigate with TROUBLESHOOTING.md
- Latency increasing → Review SCALING_GUIDE.md
- High error rate → Root cause analysis
- Resource saturation → Plan scaling

### 2. Database Maintenance (30 minutes)

**Schedule:** Every Sunday, 02:00 CET (maintenance window)

```bash
# 1. Backup database
kubectl exec -n greenlang deployment/postgres -- \
  pg_dump greenlang | gzip > /backups/cbam_db_$(date +%Y%m%d).sql.gz

# Verify backup
gunzip -t /backups/cbam_db_$(date +%Y%m%d).sql.gz
echo "Backup size: $(du -h /backups/cbam_db_$(date +%Y%m%d).sql.gz)"

# 2. Vacuum and analyze
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "VACUUM ANALYZE shipments;"

kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "VACUUM ANALYZE reports;"

# 3. Check database size and growth
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS external_size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# 4. Archive old data (>90 days)
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "
DELETE FROM shipments
WHERE import_date < NOW() - INTERVAL '90 days'
AND shipment_id IN (
  SELECT s.shipment_id FROM shipments s
  INNER JOIN reports r ON s.report_id = r.report_id
  WHERE r.submitted_to_eu = TRUE
);
"

# 5. Reindex if needed
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "REINDEX DATABASE greenlang;"

# 6. Update statistics
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "ANALYZE;"
```

**Action Items:**
- Backup failed → Retry immediately, investigate
- Database size growing rapidly → Review archival policy
- Slow queries detected → Add indexes, optimize

### 3. Log Review (20 minutes)

```bash
# 1. Aggregate weekly errors
kubectl logs -n greenlang deployment/cbam-importer --since=168h | \
  grep ERROR | \
  awk '{print $NF}' | \
  sort | uniq -c | sort -rn > /tmp/weekly_errors.txt

# 2. Review top errors
head -20 /tmp/weekly_errors.txt

# 3. Check for new error patterns
diff /reports/last_week_errors.txt /tmp/weekly_errors.txt

# 4. Archive logs
kubectl logs -n greenlang deployment/cbam-importer --since=168h > \
  /archives/logs/cbam_$(date +%Y%m%d).log

gzip /archives/logs/cbam_$(date +%Y%m%d).log
```

**Action Items:**
- New error patterns → Investigate root cause
- Recurring errors → Create permanent fix
- High error frequency → Escalate to engineering

---

## MONTHLY MAINTENANCE

### 1. Security Updates (2 hours)

**Schedule:** First Saturday of month, 10:00 CET

```bash
# 1. Update Python dependencies
cd /app
pip list --outdated

# Review each outdated package for security advisories
# Update requirements.txt
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Run tests
pytest tests/ -v

# If tests pass, commit and deploy
git add requirements.txt
git commit -m "chore: update dependencies $(date +%Y-%m)"
# Deploy via CI/CD pipeline

# 2. Scan for vulnerabilities
safety check --json > /tmp/safety_report.json

# Review findings
cat /tmp/safety_report.json | jq '.vulnerabilities'

# 3. Update base Docker image
# Check for security patches
docker pull python:3.11-slim

# Rebuild image
docker build -t greenlang/cbam-importer:latest .

# Test image
docker run --rm greenlang/cbam-importer:latest python -c "import sys; print(sys.version)"

# Push to registry
docker push greenlang/cbam-importer:latest
```

**Action Items:**
- Critical vulnerabilities → Patch immediately (P0)
- High vulnerabilities → Patch within 7 days (P1)
- Medium/Low → Schedule for next release

### 2. Emission Factor Updates (1 hour)

**Schedule:** 15th of each month

```bash
# 1. Check for updated emission factors from authoritative sources
# - IEA Cement Technology Roadmap
# - IPCC Guidelines
# - World Steel Association LCA
# - International Aluminium Institute GHG Protocol

# 2. Download updates
python scripts/update_emission_factors.py \
  --sources IEA,IPCC,WSA,IAI \
  --output data/emission_factors_$(date +%Y%m%d).py

# 3. Compare with current
diff data/emission_factors.py data/emission_factors_$(date +%Y%m%d).py > /tmp/ef_changes.txt

# 4. Review changes
cat /tmp/ef_changes.txt

# 5. If changes significant, update documentation
# Update EMISSION_FACTORS_SOURCES.md

# 6. Test with baseline calculations
python tests/test_emission_factors.py \
  --factors data/emission_factors_$(date +%Y%m%d).py \
  --baseline tests/baselines/emissions_baseline.json

# 7. If tests pass, deploy
cp data/emission_factors_$(date +%Y%m%d).py data/emission_factors.py
git add data/emission_factors.py data/EMISSION_FACTORS_SOURCES.md
git commit -m "data: update emission factors $(date +%Y-%m)"
# Deploy via CI/CD
```

**Action Items:**
- Emission factors changed → Communicate to customers
- New product types added → Update documentation
- Sources changed → Update citations

### 3. CN Code Updates (1 hour)

**Schedule:** 1st of each month

```bash
# 1. Check for EU Combined Nomenclature updates
# https://ec.europa.eu/taxation_customs/dds2/taric/

# 2. Download latest
python scripts/update_cn_codes.py \
  --source https://ec.europa.eu/taxation_customs/dds2/taric/ \
  --filter-cbam \
  --output data/cn_codes_$(date +%Y%m%d).json

# 3. Compare
diff data/cn_codes.json data/cn_codes_$(date +%Y%m%d).json

# 4. Validate
python scripts/validate_cn_codes.py \
  --cn-codes data/cn_codes_$(date +%Y%m%d).json

# 5. Test with recent shipments
python scripts/test_cn_enrichment.py \
  --cn-codes data/cn_codes_$(date +%Y%m%d).json \
  --shipments /data/recent_shipments_sample.csv

# 6. Deploy if tests pass
cp data/cn_codes_$(date +%Y%m%d).json data/cn_codes.json
git add data/cn_codes.json
git commit -m "data: update CN codes $(date +%Y-%m)"
```

### 4. Capacity Planning Review (1 hour)

```bash
# 1. Analyze last month usage
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "
SELECT
  DATE_TRUNC('day', created_at) as date,
  COUNT(*) as shipments_processed,
  AVG(processing_duration_ms) as avg_duration_ms
FROM pipeline_runs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date;
"

# 2. Check resource trends
# Review Grafana "CBAM Capacity Planning" dashboard

# 3. Forecast next quarter
python scripts/capacity_forecast.py \
  --historical-months 6 \
  --forecast-months 3 \
  --output /reports/capacity_forecast_$(date +%Y%m).json

# 4. Review and plan scaling
cat /reports/capacity_forecast_$(date +%Y%m).json | jq '.recommendations'
```

**Action Items:**
- Growth trending up → Plan infrastructure scaling
- Seasonal patterns identified → Pre-scale before peaks
- Resource waste detected → Right-size deployments

### 5. Certificate Renewal Check (10 minutes)

```bash
# 1. Check TLS certificate expiration
openssl x509 -in /etc/ssl/certs/cbam-importer.crt -noout -dates

# 2. List all certificates
kubectl get certificates -n greenlang

# 3. Check cert-manager renewals
kubectl describe certificate cbam-importer-tls -n greenlang

# 4. Verify automatic renewal configured
kubectl get certificate cbam-importer-tls -n greenlang -o yaml | grep renewBefore
```

**Action Items:**
- Certificate expiring <30 days → Renew immediately
- Auto-renewal not configured → Configure cert-manager
- Renewal failures → Troubleshoot cert-manager

---

## QUARTERLY MAINTENANCE

### 1. Comprehensive Audit (4 hours)

**Schedule:** Last week of each quarter (aligned with CBAM reporting)

```bash
# 1. Run full compliance audit
python scripts/quarterly_compliance_audit.py \
  --quarter Q4-2025 \
  --output /reports/compliance_audit_Q4_2025.pdf

# 2. Review all submitted reports
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "
SELECT report_id, quarter, submitted_at, total_emissions_tco2, validation_status
FROM reports
WHERE quarter = 'Q4-2025'
ORDER BY submitted_at DESC;
"

# 3. Verify calculation accuracy
python scripts/verify_quarterly_calculations.py \
  --quarter Q4-2025 \
  --sample-size 1000 \
  --output /reports/calculation_verification_Q4_2025.json

# 4. Audit supplier data quality
python scripts/supplier_data_audit.py \
  --quarter Q4-2025 \
  --output /reports/supplier_audit_Q4_2025.json

# 5. Review CBAM regulation changes
# Check for updates to EU CBAM Regulation 2023/956
# Update rules/cbam_rules.yaml if needed
```

### 2. Performance Optimization (3 hours)

```bash
# 1. Identify performance bottlenecks
python scripts/performance_analysis.py \
  --period-days 90 \
  --output /reports/perf_analysis_$(date +%Y%m).json

# 2. Review slow queries
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "
SELECT query, calls, mean_exec_time, stddev_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC LIMIT 20;
"

# 3. Optimize indexes
# Based on slow query analysis, add/remove indexes

# 4. Update application configuration
# Tune connection pools, caching, batch sizes

# 5. Benchmark improvements
python scripts/benchmark_pipeline.py \
  --shipment-counts 1000,5000,10000,50000 \
  --output /reports/benchmark_$(date +%Y%m).json
```

### 3. Disaster Recovery Drill (2 hours)

```bash
# 1. Simulate complete system failure
# Create test namespace
kubectl create namespace cbam-dr-test

# 2. Restore from backups
# Use ROLLBACK_PROCEDURE.md Full System Rollback

# 3. Verify data integrity
python scripts/verify_backup_integrity.py \
  --backup /backups/cbam_db_latest.sql.gz \
  --output /reports/dr_drill_$(date +%Y%m%d).json

# 4. Measure recovery time
# Target: <30 minutes to full operation

# 5. Document findings
# Update disaster recovery procedures

# 6. Cleanup
kubectl delete namespace cbam-dr-test
```

### 4. Documentation Review (1 hour)

```bash
# 1. Review all runbooks for accuracy
# - INCIDENT_RESPONSE.md
# - TROUBLESHOOTING.md
# - ROLLBACK_PROCEDURE.md
# - SCALING_GUIDE.md
# - MAINTENANCE.md (this document)

# 2. Update based on lessons learned
# Add new troubleshooting entries
# Update procedures based on actual incidents

# 3. Review API documentation
# Ensure all endpoints documented
# Update examples

# 4. Update README
# Reflect current features and capabilities
# Update performance benchmarks
```

---

## ANNUAL MAINTENANCE

### 1. Infrastructure Refresh (1 day)

**Schedule:** January (after year-end reporting complete)

- Update Kubernetes cluster version
- Update Docker base images
- Review and update all dependencies
- Major version upgrades (Python, PostgreSQL, etc.)
- Security hardening review

### 2. Compliance Certification (2 days)

- Full CBAM regulation compliance audit
- Document all calculation methodologies
- Verify audit trail completeness
- Update compliance documentation
- External audit preparation (if required)

### 3. Capacity Planning (4 hours)

- Review full year capacity usage
- Plan infrastructure for next year
- Budget forecasting
- Identify optimization opportunities

---

## BACKUP PROCEDURES

### Automated Daily Backups

**Schedule:** Every day at 02:00 CET

```bash
# Script: /scripts/backup.sh
#!/bin/bash
BACKUP_DIR=/backups
DATE=$(date +%Y%m%d)

# Database backup
kubectl exec -n greenlang deployment/postgres -- \
  pg_dump greenlang | gzip > $BACKUP_DIR/cbam_db_$DATE.sql.gz

# Configuration backup
kubectl get configmap cbam-config -n greenlang -o yaml > $BACKUP_DIR/configmap_$DATE.yaml
kubectl get secret cbam-secrets -n greenlang -o yaml > $BACKUP_DIR/secrets_$DATE.yaml

# Application manifests
kubectl get all -n greenlang -l app=cbam-importer -o yaml > $BACKUP_DIR/app_$DATE.yaml

# Data files
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/

# Retention: Keep 30 days
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.yaml" -mtime +30 -delete
```

### Backup Verification

**Weekly:** Verify backup integrity

```bash
# Test database backup restore
gunzip -t /backups/cbam_db_$(date +%Y%m%d).sql.gz

# Verify backup size reasonable
BACKUP_SIZE=$(du -sh /backups/cbam_db_$(date +%Y%m%d).sql.gz | cut -f1)
echo "Backup size: $BACKUP_SIZE"
# Alert if size < 10MB or >10GB (unusual)
```

---

## MONITORING MAINTENANCE

### Alert Rule Review (Monthly)

```bash
# 1. Review fired alerts last month
curl http://prometheus:9090/api/v1/alerts | \
  jq '.data.alerts[] | select(.state == "firing")' > /tmp/fired_alerts.json

# 2. Identify noisy alerts (false positives)
python scripts/analyze_alerts.py \
  --period-days 30 \
  --output /reports/alert_analysis_$(date +%Y%m).json

# 3. Tune thresholds
# Edit monitoring/alerts.yml

# 4. Test alert rules
promtool test rules monitoring/alerts.yml

# 5. Deploy updated alerts
kubectl apply -f monitoring/alerts.yml
```

### Dashboard Maintenance (Quarterly)

- Review Grafana dashboards for relevance
- Add new metrics if needed
- Remove obsolete panels
- Update thresholds and goals
- Ensure dashboards load <2 seconds

---

## RELATED RUNBOOKS

- INCIDENT_RESPONSE.md - For production incidents
- TROUBLESHOOTING.md - For diagnosing issues
- ROLLBACK_PROCEDURE.md - For deployment rollbacks
- SCALING_GUIDE.md - For performance scaling

---

## MAINTENANCE CALENDAR

| Frequency | Day/Time | Duration | Task |
|-----------|----------|----------|------|
| **Daily** | 09:00 CET | 25 min | Health + Data Quality + Compliance |
| **Weekly** | Mon 10:00 CET | 30 min | Performance Review |
| **Weekly** | Sun 02:00 CET | 30 min | Database Maintenance |
| **Weekly** | Fri 15:00 CET | 20 min | Log Review |
| **Monthly** | 1st Sat 10:00 CET | 2 hr | Security Updates |
| **Monthly** | 15th | 1 hr | Emission Factor Updates |
| **Monthly** | 1st | 1 hr | CN Code Updates |
| **Monthly** | Last Fri | 1 hr | Capacity Planning |
| **Monthly** | 10th | 10 min | Certificate Check |
| **Quarterly** | End of Quarter | 4 hr | Comprehensive Audit |
| **Quarterly** | End of Quarter | 3 hr | Performance Optimization |
| **Quarterly** | End of Quarter | 2 hr | DR Drill |
| **Quarterly** | End of Quarter | 1 hr | Documentation Review |
| **Annual** | January | 1 day | Infrastructure Refresh |
| **Annual** | January | 2 days | Compliance Certification |
| **Annual** | January | 4 hr | Annual Capacity Planning |

---

**Document Control:**
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Next Review:** 2025-12-18
- **Owner:** CBAM Operations Team

---

*This maintenance guide should be reviewed monthly and updated based on operational experience.*
