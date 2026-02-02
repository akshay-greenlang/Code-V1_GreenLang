# CBAM IMPORTER COPILOT - INCIDENT RESPONSE RUNBOOK

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Owner:** GreenLang CBAM Operations Team
**Classification:** OPERATIONAL CRITICAL

---

## PURPOSE

This runbook defines incident response procedures for the CBAM Importer Copilot application, ensuring rapid detection, triage, mitigation, and resolution of production incidents affecting EU CBAM quarterly reporting compliance.

---

## INCIDENT SEVERITY LEVELS

### P0 - CRITICAL (Production System Down / Compliance Risk)

**Definition:**
- Complete system failure preventing CBAM report generation
- Emissions calculations producing invalid results (compliance risk)
- EU reporting deadline at risk (<48 hours to quarterly submission)
- Data loss or corruption affecting submitted reports
- Security breach exposing sensitive supplier data

**Response Time:** Immediate (15 minutes to acknowledgment)
**Resolution Target:** 4 hours
**Escalation:** Immediate to VP Engineering and Compliance Officer
**Communication:** Every 30 minutes to stakeholders

**Examples:**
- CBAM pipeline complete failure (all 3 agents down)
- Emissions calculation formula error discovered after report submission
- Database corruption affecting shipment or emissions data
- CBAM Transitional Registry submission API unavailable 24 hours before deadline
- Ransomware attack encrypting shipment or supplier data

### P1 - HIGH (Major Feature Broken / Data Quality Risk)

**Definition:**
- Single agent failure (1 of 3 agents down)
- Significant performance degradation (>50% slowdown)
- Data quality degradation affecting >20% of shipments
- CN code enrichment failures affecting product classification
- Supplier actual emissions data unavailable (falling back to defaults)

**Response Time:** 1 hour to acknowledgment
**Resolution Target:** 24 hours
**Escalation:** After 8 hours to Engineering Manager
**Communication:** Every 2 hours during business hours

**Examples:**
- EmissionsCalculatorAgent crashing on specific product groups
- ShipmentIntakeAgent validation rejecting >20% of valid shipments
- 10x performance degradation (processing 10K shipments takes 100 minutes instead of 10)
- Missing emission factors for newly added CN codes
- Supplier profile database synchronization failures

### P2 - MEDIUM (Non-Critical Feature Broken / Minor Impact)

**Definition:**
- Performance degradation (<50% slowdown)
- Non-blocking validation warnings affecting <10% of shipments
- Monitoring or alerting system degradation
- Intermittent API failures with successful retries
- Minor data quality issues (missing optional fields)

**Response Time:** 4 hours to acknowledgment
**Resolution Target:** 5 business days
**Escalation:** After 3 days to Engineering Manager
**Communication:** Daily updates

**Examples:**
- Markdown summary generation failing (JSON report still generated)
- Grafana dashboards not loading
- Prometheus metrics collection delays
- Supplier linking failures (manual override possible)
- Log aggregation system issues

### P3 - LOW (Minor Bug / Cosmetic Issue)

**Definition:**
- Minor UI/UX issues in reports
- Non-critical documentation errors
- Low-impact performance optimizations
- Cosmetic logging or formatting issues

**Response Time:** 1 business day to acknowledgment
**Resolution Target:** 2 weeks
**Escalation:** None
**Communication:** Weekly updates in sprint review

### P4 - TRIVIAL (Enhancement Request)

**Definition:**
- Feature requests
- Performance optimizations with minimal impact
- Documentation improvements
- Code refactoring without functional changes

**Response Time:** Best effort
**Resolution Target:** Future sprint
**Escalation:** None
**Communication:** Backlog review

---

## INCIDENT RESPONSE WORKFLOW

### 1. DETECTION

#### Automated Detection (Preferred)

**Prometheus Alerts:**
```bash
# Check active alerts
curl http://prometheus:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'

# Common CBAM alerts
- CBAMPipelineFailureRate > 5%
- CBAMEmissionsCalculationError > 1%
- CBAMValidationErrorRate > 10%
- CBAMAPILatencyP95 > 5s
- CBAMReportDeadlineRisk (72h to submission)
```

**Grafana Dashboards:**
- CBAM Operations Dashboard: http://grafana:3000/d/cbam-ops
- CBAM Data Quality Dashboard: http://grafana:3000/d/cbam-quality
- CBAM Compliance Dashboard: http://grafana:3000/d/cbam-compliance

**Health Check Endpoints:**
```bash
# Application health
curl http://cbam-importer:8000/health

# Agent health
curl http://cbam-importer:8000/health/agents

# Database health
curl http://cbam-importer:8000/health/database

# Dependency health
curl http://cbam-importer:8000/health/dependencies
```

#### Manual Detection

**User Reports:**
- Customer support tickets
- Email to cbam-support@greenlang.io
- Slack channel: #cbam-incidents
- EU Importer hotline: +31-20-xxx-xxxx

### 2. TRIAGE

**Initial Assessment (within 5 minutes):**

```bash
# 1. Check system status
kubectl get pods -n greenlang -l app=cbam-importer

# 2. Check recent logs
kubectl logs -n greenlang deployment/cbam-importer --tail=100 --timestamps

# 3. Check error rates
curl http://cbam-importer:8001/metrics | grep cbam_errors_total

# 4. Check active processing
curl http://cbam-importer:8000/status | jq '.active_pipelines'
```

**Severity Classification Checklist:**

| Check | P0 | P1 | P2 | P3+ |
|-------|----|----|----|----|
| System completely down? | ✓ | | | |
| Compliance risk (wrong calculations)? | ✓ | | | |
| Deadline at risk (<48h)? | ✓ | | | |
| Single agent failure? | | ✓ | | |
| Performance >50% degraded? | | ✓ | | |
| Data quality >20% affected? | | ✓ | | |
| Performance <50% degraded? | | | ✓ | |
| Monitoring issue? | | | ✓ | |
| Cosmetic issue? | | | | ✓ |

### 3. NOTIFICATION

**P0 Notification (Immediate):**
```bash
# Send PagerDuty alert
curl -X POST https://events.pagerduty.com/v2/incidents \
  -H "Authorization: Token token=XXX" \
  -d '{
    "incident": {
      "type": "incident",
      "title": "P0: CBAM Pipeline Complete Failure",
      "service": {"id": "CBAM_SERVICE_ID"},
      "urgency": "high",
      "body": {"type": "incident_body", "details": "..."}
    }
  }'

# Notify stakeholders
# - VP Engineering (SMS + Email + Call)
# - Compliance Officer (Email + Call)
# - Customer Success (Email)
# - Affected EU Importers (Email)
```

**Communication Template:**
```
INCIDENT: [P0/P1/P2] CBAM Importer Copilot - [Brief Description]

Severity: P0/P1/P2
Detected: [timestamp]
Status: Investigating/Mitigating/Resolved
Impact: [description of user/business impact]
ETA to Resolution: [X hours]

Details:
[Detailed description]

Workaround (if available):
[Steps for manual processing or alternative approach]

Updates will be provided every [30min/2hr/daily].

Contact: [on-call engineer name] | cbam-support@greenlang.io
```

### 4. MITIGATION (Immediate Actions to Reduce Impact)

#### P0: Complete System Failure

**Quick Health Check:**
```bash
# Check all pods
kubectl get pods -n greenlang -l app=cbam-importer -o wide

# Check events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep cbam

# Check resource usage
kubectl top pods -n greenlang -l app=cbam-importer
```

**Immediate Actions:**
1. Scale up replicas if some pods are running:
```bash
kubectl scale deployment/cbam-importer --replicas=5 -n greenlang
```

2. Restart all pods if all are failing:
```bash
kubectl rollout restart deployment/cbam-importer -n greenlang
```

3. Check for recent deployments:
```bash
kubectl rollout history deployment/cbam-importer -n greenlang
```

4. Rollback if issue started after recent deployment:
```bash
kubectl rollout undo deployment/cbam-importer -n greenlang
```

5. Enable manual CBAM processing mode:
```bash
# Switch to manual CSV export for emergency submission
python scripts/emergency_manual_export.py \
  --input /data/shipments \
  --output /emergency/cbam_manual_export.csv
```

#### P0: Emissions Calculation Error (Compliance Risk)

**CRITICAL: This is the highest-risk scenario (regulatory non-compliance)**

**Immediate Actions:**
1. Stop all active pipeline runs:
```bash
curl -X POST http://cbam-importer:8000/admin/stop-all-pipelines
```

2. Identify affected shipments:
```bash
# Query database for recent calculations
kubectl exec -n greenlang deployment/cbam-importer -- \
  psql $DATABASE_URL -c \
  "SELECT shipment_id, cn_code, calculated_emissions_tco2, calculation_timestamp
   FROM shipments WHERE calculation_timestamp > NOW() - INTERVAL '24 hours'
   ORDER BY calculation_timestamp DESC LIMIT 100;"
```

3. Preserve calculation audit trail:
```bash
# Export all calculation data for forensic analysis
kubectl exec -n greenlang deployment/cbam-importer -- \
  python -c "
from backend.app import export_calculation_audit_trail
export_calculation_audit_trail(
    output_path='/exports/calculation_audit_$(date +%Y%m%d_%H%M%S).json'
)
"
```

4. Notify EU Compliance immediately:
   - Email: compliance@greenlang.io
   - CC: legal@greenlang.io
   - Subject: "URGENT: CBAM Emissions Calculation Error Detected"

5. Activate recalculation protocol:
```bash
# Use verified backup emission factors
python scripts/recalculate_with_verified_factors.py \
  --affected-shipments /exports/affected_shipments.json \
  --verified-factors data/emission_factors_backup_verified.py \
  --output /recalculated/corrected_emissions.json
```

#### P1: Single Agent Failure

**Example: EmissionsCalculatorAgent Crashing**

```bash
# Check agent-specific logs
kubectl logs -n greenlang deployment/cbam-importer -c app | grep EmissionsCalculatorAgent

# Check agent health endpoint
curl http://cbam-importer:8000/health/agents | jq '.emissions_calculator'

# Restart only affected agent (if isolation supported)
curl -X POST http://cbam-importer:8000/admin/restart-agent \
  -d '{"agent": "emissions-calculator"}'
```

**Workaround:**
- Route shipments to backup calculation service
- Enable fallback to default emission factors only
- Process manually for urgent reports

#### P1: Severe Performance Degradation

```bash
# Check database performance
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Check slow queries
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SELECT query, state, now() - query_start AS duration
           FROM pg_stat_activity
           WHERE state = 'active' AND now() - query_start > interval '30 seconds';"

# Check connection pool
curl http://cbam-importer:8001/metrics | grep db_pool

# Increase workers if CPU is underutilized
kubectl scale deployment/cbam-importer --replicas=8 -n greenlang

# Increase database connections
kubectl set env deployment/cbam-importer DB_POOL_SIZE=20 DB_MAX_OVERFLOW=40 -n greenlang
```

#### P2: Data Quality Issues

```bash
# Run data quality checks
python scripts/data_quality_check.py \
  --input /data/shipments \
  --output /reports/quality_$(date +%Y%m%d).json

# Identify validation failures
curl http://cbam-importer:8000/reports/validation-errors | jq '.top_errors'

# Review supplier data
kubectl exec -n greenlang deployment/cbam-importer -- \
  cat examples/demo_suppliers.yaml | grep -A 10 "data_quality: low"
```

### 5. INVESTIGATION

**Collect Diagnostic Data:**

```bash
# 1. Application logs (last 24 hours)
kubectl logs -n greenlang deployment/cbam-importer --since=24h > /tmp/cbam_logs.txt

# 2. Prometheus metrics snapshot
curl http://prometheus:9090/api/v1/query?query=cbam_pipeline_runs_total > /tmp/metrics_snapshot.json

# 3. Database state
kubectl exec -n greenlang deployment/postgres -- \
  pg_dump greenlang > /tmp/cbam_db_$(date +%Y%m%d_%H%M%S).sql

# 4. Recent events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | tail -100 > /tmp/k8s_events.txt

# 5. Resource usage
kubectl top pods -n greenlang -l app=cbam-importer --containers > /tmp/resource_usage.txt
```

**Common Root Causes:**

| Symptom | Likely Root Cause | Investigation |
|---------|------------------|---------------|
| All pods crashing | OOM / Configuration error | Check `kubectl describe pod` for OOM kills |
| Emissions calc errors | Emission factor database update | Check `data/emission_factors.py` recent changes |
| Validation failures spike | CN code database out of date | Check `data/cn_codes.json` version |
| Slow performance | Database query inefficiency | Check PostgreSQL slow query log |
| Supplier linking failures | Supplier YAML syntax error | Validate YAML with `yamllint` |
| Intermittent API errors | Network/DNS issues | Check network policies, DNS resolution |

### 6. RESOLUTION

**Resolution Checklist:**

- [ ] Root cause identified and documented
- [ ] Fix implemented and tested in staging
- [ ] Fix deployed to production
- [ ] Monitoring confirms issue resolved
- [ ] Affected data identified and corrected
- [ ] EU Compliance notified if calculations were affected
- [ ] Post-incident report drafted

**Verification Steps:**

```bash
# 1. Run end-to-end test
python cbam_pipeline.py \
  --input tests/fixtures/test_shipments.csv \
  --output /tmp/test_report.json \
  --importer-name "Test Co" \
  --importer-country NL \
  --importer-eori NL000000000000 \
  --declarant-name "Test User" \
  --declarant-position "Tester"

# 2. Verify calculations against known baseline
python tests/verify_calculations.py \
  --report /tmp/test_report.json \
  --baseline tests/baselines/known_good_calculations.json

# 3. Check all health endpoints
curl http://cbam-importer:8000/health
curl http://cbam-importer:8000/health/agents
curl http://cbam-importer:8000/health/database

# 4. Monitor for 1 hour
watch -n 60 'curl -s http://cbam-importer:8001/metrics | grep cbam_errors_total'
```

### 7. POST-INCIDENT REVIEW

**Timeline: Within 48 hours of resolution for P0/P1, 1 week for P2**

**Post-Incident Report Template:**

```markdown
# CBAM Incident Post-Mortem

**Incident ID:** INC-CBAM-YYYYMMDD-NNN
**Severity:** P0/P1/P2
**Date:** YYYY-MM-DD
**Duration:** X hours Y minutes
**Resolved By:** [Name]

## Summary
[One-paragraph summary of what happened]

## Timeline
- **YYYY-MM-DD HH:MM** - Incident detected
- **YYYY-MM-DD HH:MM** - Severity classified as PX
- **YYYY-MM-DD HH:MM** - Mitigation started
- **YYYY-MM-DD HH:MM** - Root cause identified
- **YYYY-MM-DD HH:MM** - Fix deployed
- **YYYY-MM-DD HH:MM** - Incident resolved

## Root Cause
[Detailed technical explanation]

## Impact
- **Shipments Affected:** X shipments
- **Reports Delayed:** X reports
- **Data Correction Required:** Yes/No
- **Compliance Risk:** Yes/No
- **Financial Impact:** €X estimated

## Resolution
[What was done to resolve]

## Prevention
- [ ] Action Item 1 (Owner: [Name], Due: [Date])
- [ ] Action Item 2
- [ ] Action Item 3

## Lessons Learned
- What went well
- What could be improved
- Changes to runbooks/procedures
```

---

## CBAM-SPECIFIC INCIDENT SCENARIOS

### Scenario 1: Emissions Calculation Formula Error

**Situation:** Error in emission factor lookup causing 10% underreporting of emissions

**Impact:** P0 - Compliance violation, potential EU fines

**Response:**
1. Immediately stop all calculations
2. Identify all affected reports (query database by timestamp)
3. Notify EU Compliance Officer and legal team
4. Prepare corrected calculations using verified formula
5. Submit amendment to EU CBAM Transitional Registry
6. Notify affected EU importers
7. Document incident for regulatory audit trail

**Prevention:**
- Implement calculation verification against test baselines
- Add automated tests for all emission factor lookups
- Require peer review for any emission factor database changes
- Implement calculation provenance tracking (SHA-256 hashes)

### Scenario 2: EU Reporting Deadline at Risk

**Situation:** Pipeline failures 36 hours before quarterly CBAM submission deadline

**Impact:** P0 - Regulatory deadline miss, potential penalties

**Response:**
1. Activate emergency manual processing protocol
2. Assign dedicated engineer team (all hands on deck)
3. Process shipments in batches if full pipeline unavailable
4. Use backup calculation scripts if agent unavailable
5. Prepare manual submission to EU Registry if automation fails
6. Request deadline extension from EU authorities (if permitted)
7. Communicate status hourly to EU importers

**Manual Submission Procedure:**
```bash
# Emergency manual CBAM report generation
python scripts/emergency_cbam_export.py \
  --shipments-csv /data/Q4_2025_shipments.csv \
  --suppliers-yaml config/suppliers.yaml \
  --importer-info config/importer_declaration.yaml \
  --output /emergency/CBAM_Q4_2025_MANUAL.json \
  --verify-calculations \
  --generate-audit-trail
```

### Scenario 3: Supplier Data Unavailable

**Situation:** Supplier database synchronization failure, all shipments falling back to EU default emission factors

**Impact:** P1 - Data quality degradation, suboptimal reporting (higher reported emissions)

**Response:**
1. Assess criticality: Is supplier actual data available for high-volume products?
2. Contact supplier data providers to restore feed
3. If <48h to deadline: Proceed with defaults + document in report
4. If >48h to deadline: Delay processing until supplier data restored
5. Generate comparison report showing emissions difference (supplier actual vs defaults)
6. Communicate to EU importers the data quality situation

**Default vs Actual Comparison:**
```bash
# Generate comparison report
python scripts/compare_emissions_methods.py \
  --shipments /data/shipments.json \
  --output-comparison /reports/default_vs_actual_comparison.json
```

### Scenario 4: CN Code Database Out of Date

**Situation:** EU updates CBAM Annex I with new CN codes, validation rejecting valid shipments

**Impact:** P1 - Valid shipments rejected, reporting delays

**Response:**
1. Obtain updated EU Combined Nomenclature from official source
2. Update `data/cn_codes.json` with new codes
3. Re-validate rejected shipments
4. Re-process affected batches
5. Deploy updated CN code database to production
6. Monitor for additional new codes

**CN Code Update Procedure:**
```bash
# Update CN codes from official EU source
python scripts/update_cn_codes.py \
  --source https://ec.europa.eu/taxation_customs/dds2/taric/taric_consultation.jsp \
  --output data/cn_codes_$(date +%Y%m%d).json \
  --verify-cbam-annex-i

# Test against current shipments
python scripts/validate_cn_codes.py \
  --shipments /data/current_shipments.csv \
  --cn-codes data/cn_codes_$(date +%Y%m%d).json
```

### Scenario 5: High Volume Processing Overload

**Situation:** Year-end quarter with 50,000 shipments (5x normal volume) causing timeout failures

**Impact:** P1 - Processing delays, deadline risk

**Response:**
1. Scale horizontally: Increase pods from 3 to 10
```bash
kubectl scale deployment/cbam-importer --replicas=10 -n greenlang
```

2. Increase database resources:
```bash
kubectl scale statefulset/postgres --replicas=3 -n greenlang
```

3. Enable batch processing mode:
```bash
# Process in smaller batches
python cbam_pipeline.py \
  --input large_shipments.csv \
  --batch-size 5000 \
  --output-dir /batched_reports/ \
  --merge-final
```

4. Optimize performance:
```bash
# Disable non-critical features
export CBAM_SKIP_SUMMARY_MARKDOWN=true
export CBAM_SKIP_INTERMEDIATE_OUTPUTS=true
export CBAM_PARALLEL_WORKERS=8
```

---

## EMERGENCY CONTACTS

### Internal Escalation

| Role | Contact | Phone | Availability |
|------|---------|-------|--------------|
| On-Call Engineer | Slack @oncall-cbam | +31-20-xxx-1001 | 24/7 |
| Engineering Manager | john.smith@greenlang.io | +31-20-xxx-1002 | Business hours |
| VP Engineering | sarah.jones@greenlang.io | +31-20-xxx-1003 | P0/P1 only |
| Compliance Officer | compliance@greenlang.io | +31-20-xxx-1004 | P0 only |
| CTO | cto@greenlang.io | +31-20-xxx-1005 | P0 critical |

### External Escalation

| Entity | Contact | Use Case |
|--------|---------|----------|
| EU CBAM Helpdesk | taxud-cbam@ec.europa.eu | Regulatory questions, deadline extensions |
| EU Importer Hotline | +31-20-xxx-2000 | User impact communication |
| Cloud Provider Support | aws-support-cbam | Infrastructure issues |
| Database Vendor | postgres-enterprise-support | Database-specific issues |

---

## INCIDENT TRACKING

**Incident Management System:** JIRA (Project: CBAM-OPS)

**Create Incident:**
```bash
# JIRA CLI
jira create \
  --project CBAM-OPS \
  --type Incident \
  --severity P0/P1/P2 \
  --summary "Brief description" \
  --description "Full details" \
  --assignee @oncall-cbam
```

**Update Incident:**
```bash
jira update CBAM-OPS-123 \
  --status "Investigating" \
  --comment "Update message"
```

---

## APPENDIX

### A. Health Check Commands Quick Reference

```bash
# System health
curl http://cbam-importer:8000/health

# Agent health
curl http://cbam-importer:8000/health/agents | jq '.'

# Database health
kubectl exec -n greenlang deployment/cbam-importer -- \
  psql $DATABASE_URL -c "SELECT 1;"

# Metrics
curl http://cbam-importer:8001/metrics | grep cbam_
```

### B. Common kubectl Commands

```bash
# Get pods
kubectl get pods -n greenlang -l app=cbam-importer

# Logs
kubectl logs -n greenlang deployment/cbam-importer --tail=100

# Describe pod
kubectl describe pod -n greenlang <pod-name>

# Events
kubectl get events -n greenlang --sort-by='.lastTimestamp'

# Scale
kubectl scale deployment/cbam-importer --replicas=5 -n greenlang

# Restart
kubectl rollout restart deployment/cbam-importer -n greenlang

# Rollback
kubectl rollout undo deployment/cbam-importer -n greenlang
```

### C. Emergency Manual Processing Scripts

Located in: `scripts/emergency/`

- `emergency_manual_export.py` - Export shipments to CSV for manual processing
- `recalculate_with_verified_factors.py` - Recalculate with verified emission factors
- `emergency_cbam_export.py` - Generate CBAM report without full pipeline
- `compare_emissions_methods.py` - Compare default vs actual emissions
- `update_cn_codes.py` - Update CN code database from EU source
- `validate_cn_codes.py` - Validate CN codes against shipments

---

**Document Control:**
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Next Review:** 2025-12-18
- **Owner:** CBAM Operations Team
- **Approvers:** VP Engineering, Compliance Officer

---

*This runbook is a living document and should be updated after each incident to incorporate lessons learned.*
