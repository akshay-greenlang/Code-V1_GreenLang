# GL-005 COMBUSTIONCONTROLAGENT - MAINTENANCE GUIDE

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Owner:** GreenLang Industrial Control Operations Team

---

## PURPOSE

This guide defines routine maintenance procedures for the GL-005 CombustionControlAgent to ensure system reliability, control performance, safety integrity (SIL-2), regulatory compliance, and optimal combustion efficiency.

**SAFETY NOTE:** All maintenance activities must preserve safety interlock functionality. Never disable safety systems during maintenance.

---

## DAILY MAINTENANCE

### 1. Health Monitoring (10 minutes)

**Schedule:** Every business day, 08:00 local time

```bash
# 1. Check system health
curl http://gl-005-combustion-control:8000/health | jq '.status'
# Expected: "healthy"

# 2. Check all 5 agents healthy
curl http://gl-005-combustion-control:8000/health/agents | \
  jq '.agents[] | select(.status != "healthy")'
# Expected: No output (all agents healthy)

# 3. Check pod status
kubectl get pods -n greenlang -l app=gl-005-combustion-control
# Expected: All pods Running, READY 1/1

# 4. Check safety interlocks
curl http://gl-005-combustion-control:8000/safety/status | jq '.safety_status'
# Expected: "OK" (not "WARNING", "CRITICAL", or "EMERGENCY_STOP")

# 5. Review error logs (last 24 hours)
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=24h | \
  grep -i error | wc -l
# Expected: <10 errors per day

# 6. Check control loop performance
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_control_loop_duration_seconds{quantile="0.95"}'
# Expected: <0.1 (100ms P95 latency)
```

**Action Items:**
- If health check fails → Follow INCIDENT_RESPONSE.md (P1)
- If safety status not "OK" → Immediate investigation (P0)
- If error count >10 → Review error patterns
- If control latency >100ms → Follow TROUBLESHOOTING.md

### 2. Control Performance Monitoring (10 minutes)

```bash
# 1. Check combustion efficiency
curl http://gl-005-combustion-control:8000/reports/efficiency-summary?period=24h | \
  jq '.average_efficiency_percent'
# Expected: >85% for gas, >83% for oil

# 2. Check emissions compliance
curl http://gl-005-combustion-control:8000/compliance/status | \
  jq '.compliance_status, .violations'
# Expected: "compliant", violations: []

# 3. Monitor fuel-air ratio stability
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_fuel_air_ratio_deviation_percent{quantile="0.95"}'
# Expected: <2% deviation

# 4. Check flame stability
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_flame_stability_index'
# Expected: >0.9 (scale 0-1)

# 5. Monitor heat output accuracy
curl http://gl-005-combustion-control:8000/reports/heat-output-accuracy?period=24h | \
  jq '.average_error_percent'
# Expected: <3% error
```

**Action Items:**
- Efficiency <85% (gas) or <83% (oil) → Investigate combustion tuning
- Emissions violations → Immediate investigation (P0 - regulatory)
- Fuel-air deviation >2% → Check sensor calibration
- Flame stability <0.9 → Check burner condition
- Heat output error >3% → Verify measurement instrumentation

### 3. Integration Health Check (5 minutes)

```bash
# 1. Check DCS/PLC connectivity
curl http://gl-005-combustion-control:8000/integrations/status | \
  jq '.integrations[] | {name: .name, status: .status, latency_ms: .latency_ms}'
# Expected: All "connected", latency <50ms

# 2. Verify CEMS data quality
curl http://gl-005-combustion-control:8000/integrations/cems/status | \
  jq '.analyzers[] | select(.status != "ok")'
# Expected: No output (all analyzers ok)

# 3. Check SCADA publishing
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_scada_publish_errors_total'
# Expected: 0 errors

# 4. Monitor data acquisition rate
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_data_acquisition_rate_hz'
# Expected: 50-100 Hz (configured rate)
```

**Action Items:**
- Integration failures → Check network connectivity
- High latency (>50ms) → Investigate network or DCS/PLC performance
- CEMS analyzer failures → Check analyzer status, calibration
- SCADA publish errors → Verify SCADA server health

---

## WEEKLY MAINTENANCE

### 1. Performance Review (30 minutes)

**Schedule:** Every Monday, 10:00 local time

```bash
# 1. Review Grafana dashboards
# - GL-005 Agent Performance Dashboard
# - GL-005 Combustion Metrics Dashboard
# - GL-005 Safety Monitoring Dashboard

# 2. Check control loop latency trends (7 days)
curl "http://prometheus:9090/api/v1/query_range?query=\
gl005_control_loop_duration_seconds{quantile=\"0.95\"}&\
start=$(date -d '7 days ago' +%s)&end=$(date +%s)&step=3600"

# 3. Check combustion efficiency trends
curl "http://prometheus:9090/api/v1/query_range?query=\
gl005_combustion_efficiency_percent&\
start=$(date -d '7 days ago' +%s)&end=$(date +%s)&step=3600"

# 4. Check emissions trends (NOx, CO, CO2)
curl "http://prometheus:9090/api/v1/query_range?query=\
gl005_emissions_nox_ppm&\
start=$(date -d '7 days ago' +%s)&end=$(date +%s)&step=3600"

# 5. Resource utilization
kubectl top pods -n greenlang -l app=gl-005-combustion-control --containers

# 6. Database performance
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%combustion%' OR query LIKE '%control%'
ORDER BY total_exec_time DESC LIMIT 10;"
```

**Action Items:**
- Latency increasing → Review SCALING_GUIDE.md
- Efficiency declining → Schedule burner tuning
- Emissions increasing → Check fuel quality, burner condition
- High resource usage → Plan scaling or optimization
- Slow queries → Add indexes, optimize

### 2. Sensor Calibration Verification (20 minutes)

**Schedule:** Every Friday, 14:00 local time

```bash
# 1. Check sensor calibration status
curl http://gl-005-combustion-control:8000/sensors/calibration-status | \
  jq '.sensors[] | select(.days_since_calibration > 30)'
# Flag sensors >30 days since calibration

# 2. Verify sensor health scores
curl http://gl-005-combustion-control:8000/sensors/health | \
  jq '.sensors[] | select(.health_score < 0.9)'
# Flag sensors with health score <0.9

# 3. Check for sensor drift
curl http://gl-005-combustion-control:8000/sensors/drift-analysis | \
  jq '.sensors[] | select(.drift_percent > 5)'
# Flag sensors with >5% drift

# 4. Review redundant sensor agreement
curl http://gl-005-combustion-control:8000/sensors/redundancy-check | \
  jq '.sensor_groups[] | select(.disagreement_percent > 10)'
# Flag sensor groups with >10% disagreement

# 5. Generate calibration schedule
python scripts/generate_calibration_schedule.py \
  --weeks-ahead 4 \
  --output /reports/calibration_schedule_$(date +%Y%m%d).pdf
```

**Action Items:**
- Sensors >30 days since calibration → Schedule calibration
- Health score <0.9 → Investigate sensor issues
- Drift >5% → Recalibrate immediately
- Redundant sensor disagreement >10% → Check for faulty sensor

### 3. Database Maintenance (30 minutes)

**Schedule:** Every Sunday, 02:00 local time (maintenance window)

```bash
# 1. Backup database
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  pg_dump greenlang | gzip > /backups/gl005_db_$(date +%Y%m%d).sql.gz

# Verify backup
gunzip -t /backups/gl005_db_$(date +%Y%m%d).sql.gz
echo "Backup size: $(du -h /backups/gl005_db_$(date +%Y%m%d).sql.gz)"

# 2. Vacuum and analyze (TimescaleDB-aware)
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "VACUUM ANALYZE combustion_data;"

kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "VACUUM ANALYZE control_events;"

kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "VACUUM ANALYZE safety_events;"

# 3. Check database size and growth
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT
  hypertable_name,
  pg_size_pretty(hypertable_size(format('%I.%I', hypertable_schema, hypertable_name))) AS size
FROM timescaledb_information.hypertables
ORDER BY hypertable_size(format('%I.%I', hypertable_schema, hypertable_name)) DESC;
"

# 4. Compress old data (>7 days)
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT compress_chunk(chunk, if_not_compressed => true)
FROM timescaledb_information.chunks
WHERE hypertable_name = 'combustion_data'
  AND range_end < NOW() - INTERVAL '7 days';
"

# 5. Drop very old data (>90 days) - retention policy
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT drop_chunks('combustion_data', INTERVAL '90 days');
SELECT drop_chunks('control_events', INTERVAL '365 days');  -- Keep 1 year for audit
"

# 6. Update statistics
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "ANALYZE;"

# 7. Check for bloat
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) -
                      pg_relation_size(schemaname||'.'||tablename)) AS external_size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema', '_timescaledb_internal')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
"
```

**Action Items:**
- Backup failed → Retry immediately, investigate
- Database size growing rapidly → Review retention policies
- High bloat → Schedule VACUUM FULL during maintenance window
- Compression not working → Check TimescaleDB compression policies

### 4. Log Review and Analysis (20 minutes)

```bash
# 1. Aggregate weekly errors
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=168h | \
  grep ERROR | \
  awk '{print $NF}' | \
  sort | uniq -c | sort -rn > /tmp/weekly_errors.txt

# 2. Review top errors
head -20 /tmp/weekly_errors.txt

# 3. Check for new error patterns
diff /reports/last_week_errors.txt /tmp/weekly_errors.txt

# 4. Review safety event logs
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT event_type, COUNT(*) as count, MAX(timestamp) as last_occurrence
FROM safety_events
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY event_type
ORDER BY count DESC;
"

# 5. Archive logs
kubectl logs -n greenlang deployment/gl-005-combustion-control --since=168h > \
  /archives/logs/gl005_$(date +%Y%m%d).log

gzip /archives/logs/gl005_$(date +%Y%m%d).log

# 6. Save for next week comparison
cp /tmp/weekly_errors.txt /reports/last_week_errors.txt
```

**Action Items:**
- New error patterns → Investigate root cause
- Recurring errors → Create permanent fix
- High error frequency → Escalate to engineering
- Safety events increasing → Safety review required

---

## MONTHLY MAINTENANCE

### 1. Security Updates and Patching (2 hours)

**Schedule:** First Saturday of month, 10:00 local time

```bash
# 1. Review security advisories
# Check for vulnerabilities in:
# - Python packages
# - Base Docker images
# - Kubernetes components
# - DCS/PLC integration libraries

# 2. Update Python dependencies
cd /app
pip list --outdated

# Review each outdated package for security advisories
# Update requirements.txt
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# 3. Run tests
pytest tests/ -v --cov --cov-report=term-missing

# If tests pass, commit and deploy
git add requirements.txt
git commit -m "chore: update dependencies $(date +%Y-%m)"
# Deploy via CI/CD pipeline

# 4. Scan for vulnerabilities
safety check --json > /tmp/safety_report.json
bandit -r agents/ calculators/ integrations/ -ll -i

# Review findings
cat /tmp/safety_report.json | jq '.vulnerabilities'

# 5. Update base Docker image
docker pull python:3.11-slim

# Rebuild image
docker build -t greenlang/gl-005-combustion-control:latest .

# Test image
docker run --rm greenlang/gl-005-combustion-control:latest python --version

# Scan with Trivy
trivy image greenlang/gl-005-combustion-control:latest

# Push to registry if scan passes
docker push greenlang/gl-005-combustion-control:latest

# 6. Update Kubernetes manifests
kubectl apply -f k8s/
```

**Action Items:**
- Critical vulnerabilities → Patch immediately (P0)
- High vulnerabilities → Patch within 7 days (P1)
- Medium/Low → Schedule for next release

### 2. PID Controller Tuning Review (1.5 hours)

**Schedule:** 15th of each month

```bash
# 1. Analyze control performance (last 30 days)
python scripts/analyze_pid_performance.py \
  --period-days 30 \
  --output /reports/pid_analysis_$(date +%Y%m).json

# 2. Check for oscillations
curl "http://prometheus:9090/api/v1/query?query=\
stddev_over_time(gl005_heat_output_mw[1h]) > 2"

# 3. Check for steady-state error
curl "http://prometheus:9090/api/v1/query?query=\
abs(gl005_heat_output_mw - gl005_heat_setpoint_mw) > 1"

# 4. Review integral windup events
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_pid_integral_windup_events_total'

# 5. Generate tuning recommendations
python scripts/recommend_pid_tuning.py \
  --analysis /reports/pid_analysis_$(date +%Y%m).json \
  --output /reports/pid_tuning_recommendations_$(date +%Y%m).pdf

# 6. Apply tuning updates (if needed, requires approval)
# Update ConfigMap with new PID parameters
kubectl edit configmap gl-005-config -n greenlang

# Restart pods to apply new tuning
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang

# 7. Monitor for 48 hours after tuning changes
# Verify improved performance
```

**Action Items:**
- Oscillations detected → Reduce gain (Kp) or derivative (Kd)
- Steady-state error → Increase integral gain (Ki)
- Integral windup → Improve anti-windup protection
- Poor load following → Adjust feedforward compensation

### 3. Emission Factor Database Updates (1 hour)

**Schedule:** 20th of each month

```bash
# 1. Check for updated emission factors from authoritative sources
# - EPA AP-42 (Compilation of Air Pollutant Emission Factors)
# - IPCC Guidelines
# - Local regulatory authorities

# 2. Download updates
python scripts/update_emission_factors.py \
  --sources EPA,IPCC,LOCAL \
  --output data/emission_factors_$(date +%Y%m%d).py

# 3. Compare with current
diff data/emission_factors.py data/emission_factors_$(date +%Y%m%d).py > \
  /tmp/ef_changes.txt

# 4. Review changes
cat /tmp/ef_changes.txt

# 5. If changes significant, update documentation
# Update docs/EMISSION_FACTORS_SOURCES.md

# 6. Test with baseline calculations
python tests/test_emission_calculations.py \
  --factors data/emission_factors_$(date +%Y%m%d).py \
  --baseline tests/baselines/emissions_baseline.json

# 7. If tests pass, deploy
cp data/emission_factors_$(date +%Y%m%d).py data/emission_factors.py
git add data/emission_factors.py docs/EMISSION_FACTORS_SOURCES.md
git commit -m "data: update emission factors $(date +%Y-%m)"
# Deploy via CI/CD
```

**Action Items:**
- Emission factors changed → Communicate to compliance team
- New fuel types added → Update burner configuration
- Sources changed → Update citations

### 4. Capacity Planning Review (1 hour)

```bash
# 1. Analyze last month usage
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT
  DATE_TRUNC('day', timestamp) as date,
  COUNT(DISTINCT unit_id) as active_units,
  AVG(control_frequency_hz) as avg_frequency,
  MAX(burner_count) as max_burners
FROM control_performance
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', timestamp)
ORDER BY date;
"

# 2. Check resource trends
# Review Grafana "GL-005 Capacity Planning" dashboard

# 3. Forecast next quarter
python scripts/capacity_forecast.py \
  --historical-months 6 \
  --forecast-months 3 \
  --output /reports/capacity_forecast_$(date +%Y%m).json

# 4. Review and plan scaling
cat /reports/capacity_forecast_$(date +%Y%m).json | jq '.recommendations'

# 5. Check node capacity
kubectl describe nodes | grep -A 5 "Allocated resources"

# 6. Plan infrastructure changes
# If growth >20% expected, plan node additions
# If new facilities commissioning, plan dedicated nodes
```

**Action Items:**
- Growth trending up → Plan infrastructure scaling
- Seasonal patterns identified → Pre-scale before peaks
- Resource waste detected → Right-size deployments
- New facilities planned → Prepare dedicated infrastructure

### 5. Certificate and Credential Management (15 minutes)

```bash
# 1. Check TLS certificate expiration
openssl x509 -in /etc/ssl/certs/gl-005.crt -noout -dates

# 2. List all certificates
kubectl get certificates -n greenlang

# 3. Check cert-manager renewals
kubectl describe certificate gl-005-tls -n greenlang

# 4. Verify automatic renewal configured
kubectl get certificate gl-005-tls -n greenlang -o yaml | grep renewBefore

# 5. Check DCS/PLC credentials expiration
# Review integration configuration
kubectl get secret gl-005-dcs-credentials -n greenlang -o json | \
  jq -r '.data.expires_at' | base64 -d

# 6. Rotate API keys (if expiring <30 days)
python scripts/rotate_api_keys.py \
  --expiring-within-days 30 \
  --backup /backups/api_keys_$(date +%Y%m%d).enc
```

**Action Items:**
- Certificate expiring <30 days → Renew immediately
- Auto-renewal not configured → Configure cert-manager
- Credentials expiring → Coordinate with IT for rotation
- API keys expiring → Rotate and update secrets

---

## QUARTERLY MAINTENANCE

### 1. Comprehensive Safety Audit (4 hours)

**Schedule:** Last week of each quarter

```bash
# 1. Run full safety validation suite
python tests/test_safety_interlocks.py \
  --comprehensive \
  --include-sil2-validation \
  --output /reports/safety_audit_Q$(date +%q)_$(date +%Y).pdf

# 2. Review all safety event logs
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT event_type, severity, COUNT(*) as count,
       MAX(timestamp) as last_occurrence,
       AVG(response_time_ms) as avg_response_ms
FROM safety_events
WHERE timestamp > NOW() - INTERVAL '90 days'
GROUP BY event_type, severity
ORDER BY severity DESC, count DESC;
"

# 3. Verify safety interlock response times
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_safety_interlock_latency_seconds'
# All should be <100ms

# 4. Test emergency shutdown
# Perform dry-run emergency shutdown test
curl -X POST http://gl-005-combustion-control:8000/test/emergency-shutdown \
  -H "Content-Type: application/json" \
  -d '{"test_mode": true, "dry_run": true, "unit_id": "BOILER001"}'

# 5. Review sensor redundancy health
curl http://gl-005-combustion-control:8000/sensors/redundancy-health | \
  jq '.redundancy_groups[] | select(.health_score < 0.95)'

# 6. Verify SIL-2 compliance
python scripts/verify_sil2_compliance.py \
  --period-days 90 \
  --output /reports/sil2_compliance_Q$(date +%q)_$(date +%Y).pdf

# 7. Safety Officer review and sign-off
# Generate report for Safety Officer
python scripts/generate_safety_officer_report.py \
  --quarter Q$(date +%q)-$(date +%Y) \
  --output /reports/safety_officer_report_Q$(date +%q)_$(date +%Y).pdf
```

**Action Items:**
- Safety test failures → Immediate investigation (P0)
- Response times >100ms → Performance optimization required
- Redundancy health <0.95 → Check sensor maintenance
- SIL-2 non-compliance → Corrective action plan required

### 2. Performance Optimization (3 hours)

```bash
# 1. Identify performance bottlenecks
python scripts/performance_analysis.py \
  --period-days 90 \
  --output /reports/perf_analysis_$(date +%Y%m).json

# 2. Review control loop latency distribution
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT
  percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
  percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99,
  MAX(latency_ms) AS max
FROM control_performance
WHERE timestamp > NOW() - INTERVAL '90 days';
"

# 3. Review slow queries
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
SELECT query, calls, mean_exec_time, stddev_exec_time, max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC LIMIT 20;
"

# 4. Optimize indexes
# Based on slow query analysis, add/remove indexes
# Review query plans
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "EXPLAIN ANALYZE SELECT ..."

# 5. Update application configuration
# Tune connection pools, caching, batch sizes based on analysis

# 6. Benchmark improvements
python scripts/benchmark_control_loop.py \
  --burner-counts 10,25,50,75,100 \
  --frequencies 1,5,10 \
  --output /reports/benchmark_Q$(date +%q)_$(date +%Y).json

# 7. Compare with baseline
python scripts/compare_performance.py \
  --current /reports/benchmark_Q$(date +%q)_$(date +%Y).json \
  --baseline /reports/baseline_performance.json
```

**Action Items:**
- Latency regression → Investigate and optimize
- Slow queries → Add indexes or rewrite queries
- Resource saturation → Plan scaling or optimization
- Benchmark regressions → Root cause analysis required

### 3. Disaster Recovery Drill (2 hours)

```bash
# 1. Simulate complete system failure
# Create test namespace
kubectl create namespace gl005-dr-test

# 2. Restore from backups
# Use ROLLBACK_PROCEDURE.md Full System Rollback

# 3. Verify data integrity
python scripts/verify_backup_integrity.py \
  --backup /backups/gl005_db_latest.sql.gz \
  --output /reports/dr_drill_$(date +%Y%m%d).json

# 4. Test safety systems post-restore
python tests/test_safety_interlocks.py \
  --namespace gl005-dr-test \
  --comprehensive

# 5. Measure recovery time
# Target: <30 minutes to full operation with safety validated

# 6. Test failover to backup control system
curl -X POST http://backup-control:8000/assume-control \
  -d '{"reason": "dr_drill", "test_mode": true}'

# 7. Document findings
# Update disaster recovery procedures based on drill

# 8. Cleanup
kubectl delete namespace gl005-dr-test
```

**Action Items:**
- Recovery time >30 minutes → Optimize DR procedures
- Data integrity issues → Review backup procedures
- Safety validation failures → Critical - fix immediately
- Failover issues → Test backup control system

### 4. Documentation Review and Update (1.5 hours)

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
# Incorporate findings from quarterly drills

# 3. Review API documentation
# Ensure all endpoints documented
# Update examples
# Verify accuracy of integration guides

# 4. Update README and architecture documentation
# Reflect current features and capabilities
# Update performance benchmarks
# Update deployment guides

# 5. Review and update safety documentation
# Safety procedures
# SIL-2 validation documentation
# Emergency response procedures

# 6. Generate quarterly report
python scripts/generate_quarterly_report.py \
  --quarter Q$(date +%q)-$(date +%Y) \
  --output /reports/quarterly_report_Q$(date +%q)_$(date +%Y).pdf
```

**Action Items:**
- Documentation gaps → Fill immediately
- Outdated procedures → Update and test
- Inconsistencies → Resolve and validate
- Safety documentation → Safety Officer review required

---

## ANNUAL MAINTENANCE

### 1. Infrastructure Refresh (2 days)

**Schedule:** January (after year-end production complete)

**Day 1: System Updates**
- Update Kubernetes cluster version
- Update Docker base images to latest LTS
- Review and update all dependencies (major versions)
- Major version upgrades (Python, PostgreSQL, Redis, TimescaleDB)
- Security hardening review
- Update monitoring and alerting infrastructure

**Day 2: Hardware and Network**
- Review DCS/PLC network infrastructure
- Update network switches and firewalls
- Verify sensor calibration (all sensors)
- Test emergency backup systems
- Review physical security controls

### 2. Comprehensive Compliance Certification (3 days)

**Day 1: Regulatory Compliance**
- Full emissions compliance audit (EPA, local regulations)
- Document all calculation methodologies
- Verify audit trail completeness
- Update compliance documentation
- External audit preparation (if required)

**Day 2: Safety Compliance**
- SIL-2 certification review
- Safety system validation
- Emergency response procedure validation
- Safety training for operations staff
- Safety documentation review and update

**Day 3: Industry Standards**
- ASME PTC 4.1 compliance verification
- NFPA 85 compliance review
- IEC 61508 functional safety review
- Update standards documentation

### 3. Capacity Planning and Forecasting (1 day)

**Morning: Analysis**
- Review full year capacity usage
- Analyze growth trends
- Identify seasonal patterns
- Review resource utilization efficiency

**Afternoon: Planning**
- Plan infrastructure for next year
- Budget forecasting
- Identify optimization opportunities
- Plan for new facility integrations

### 4. Team Training and Knowledge Transfer (1 day)

- Operations team training on system updates
- Safety procedure refresher
- Emergency response drills
- Knowledge transfer sessions
- Update training materials

---

## BACKUP PROCEDURES

### Automated Daily Backups

**Schedule:** Every day at 02:00 local time

```bash
# Script: /scripts/backup.sh
#!/bin/bash
BACKUP_DIR=/backups
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  pg_dump greenlang | gzip > $BACKUP_DIR/gl005_db_$DATE.sql.gz

# Configuration backup
kubectl get configmap gl-005-config -n greenlang -o yaml > \
  $BACKUP_DIR/configmap_$DATE.yaml
kubectl get secret gl-005-secrets -n greenlang -o yaml > \
  $BACKUP_DIR/secrets_$DATE.yaml

# Safety configuration backup (CRITICAL)
kubectl get configmap gl-005-safety-config -n greenlang -o yaml > \
  $BACKUP_DIR/safety_config_$DATE.yaml

# Application manifests
kubectl get all -n greenlang -l app=gl-005-combustion-control -o yaml > \
  $BACKUP_DIR/app_$DATE.yaml

# PID tuning parameters backup
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  cat /app/config/pid_tuning.yaml > $BACKUP_DIR/pid_tuning_$DATE.yaml

# Emission factors backup
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  cat /app/data/emission_factors.py > $BACKUP_DIR/emission_factors_$DATE.py

# Retention policies
# Keep daily backups: 30 days
# Keep weekly backups: 90 days
# Keep monthly backups: 1 year
# Keep quarterly backups: 3 years (compliance)

find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.yaml" -mtime +30 -delete
```

### Backup Verification

**Weekly:** Verify backup integrity

```bash
# Test database backup restore
gunzip -t /backups/gl005_db_latest.sql.gz

# Verify backup size reasonable
BACKUP_SIZE=$(du -sh /backups/gl005_db_latest.sql.gz | cut -f1)
echo "Backup size: $BACKUP_SIZE"
# Alert if size < 100MB or >100GB (unusual)

# Monthly: Test restore in isolated environment
kubectl create namespace gl005-backup-test
# Restore and validate
kubectl delete namespace gl005-backup-test
```

---

## MONITORING MAINTENANCE

### Alert Rule Review (Monthly)

```bash
# 1. Review fired alerts last month
curl "http://prometheus:9090/api/v1/alerts" | \
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
- Verify all links to runbooks functional

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
| **Daily** | 08:00 | 25 min | Health + Control Performance + Integration Check |
| **Weekly** | Mon 10:00 | 30 min | Performance Review |
| **Weekly** | Fri 14:00 | 20 min | Sensor Calibration Verification |
| **Weekly** | Sun 02:00 | 30 min | Database Maintenance |
| **Weekly** | Fri 16:00 | 20 min | Log Review and Analysis |
| **Monthly** | 1st Sat 10:00 | 2 hr | Security Updates and Patching |
| **Monthly** | 15th | 1.5 hr | PID Controller Tuning Review |
| **Monthly** | 20th | 1 hr | Emission Factor Updates |
| **Monthly** | Last Fri | 1 hr | Capacity Planning Review |
| **Monthly** | 10th | 15 min | Certificate & Credential Management |
| **Quarterly** | End of Quarter | 4 hr | Comprehensive Safety Audit |
| **Quarterly** | End of Quarter | 3 hr | Performance Optimization |
| **Quarterly** | End of Quarter | 2 hr | Disaster Recovery Drill |
| **Quarterly** | End of Quarter | 1.5 hr | Documentation Review |
| **Annual** | January | 2 days | Infrastructure Refresh |
| **Annual** | January | 3 days | Compliance Certification |
| **Annual** | January | 1 day | Capacity Planning & Forecasting |
| **Annual** | January | 1 day | Team Training |

---

**Document Control:**
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Next Review:** 2025-12-18 (Monthly)
- **Owner:** GL-005 Operations Team
- **Safety Classification:** SIL-2 Safety System

---

*This maintenance guide should be reviewed monthly and updated based on operational experience. All safety-related maintenance requires Safety Officer approval.*
