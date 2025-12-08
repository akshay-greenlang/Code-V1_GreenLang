# GreenLang Troubleshooting Workshop

**Document Version:** 1.0
**Last Updated:** December 2025
**Audience:** All Roles (Operators, Developers, Administrators)
**Prerequisites:** Role-specific training completed

---

## Table of Contents

1. [Introduction](#introduction)
2. [Troubleshooting Methodology](#troubleshooting-methodology)
3. [Common Issues and Solutions](#common-issues-and-solutions)
4. [Diagnostic Tools](#diagnostic-tools)
5. [Log Analysis](#log-analysis)
6. [Performance Troubleshooting](#performance-troubleshooting)
7. [Data Quality Issues](#data-quality-issues)
8. [Integration Problems](#integration-problems)
9. [Calculation Errors](#calculation-errors)
10. [Hands-On Exercises](#hands-on-exercises)
11. [Escalation Procedures](#escalation-procedures)

---

## Introduction

This workshop provides practical troubleshooting skills for GreenLang. You will learn systematic approaches to identify, diagnose, and resolve common issues encountered in production environments.

### Learning Objectives

Upon completion, you will be able to:

- Apply systematic troubleshooting methodology
- Use diagnostic tools effectively
- Analyze logs to identify root causes
- Resolve common operational issues
- Know when and how to escalate

---

## Troubleshooting Methodology

### The SOLVED Framework

```
S - Symptoms: What is happening?
O - Onset: When did it start?
L - Location: Where is it occurring?
V - Variables: What changed recently?
E - Evidence: What do logs/metrics show?
D - Diagnosis: What is the root cause?
```

### Step-by-Step Process

```
1. GATHER INFORMATION
   ├── Collect error messages
   ├── Note timestamps
   ├── Identify affected components
   └── Check recent changes

2. REPRODUCE THE ISSUE
   ├── Confirm symptoms
   ├── Isolate variables
   └── Document steps

3. ANALYZE ROOT CAUSE
   ├── Review logs
   ├── Check metrics
   ├── Trace request flow
   └── Test hypotheses

4. IMPLEMENT FIX
   ├── Apply solution
   ├── Verify resolution
   └── Monitor for recurrence

5. DOCUMENT & PREVENT
   ├── Update runbook
   ├── Add monitoring
   └── Implement prevention
```

---

## Common Issues and Solutions

### Issue 1: API Returns 500 Errors

**Symptoms:**
- API requests return HTTP 500
- Error rate spike in monitoring
- Users report failed operations

**Diagnosis Steps:**

```bash
# Check API server logs
kubectl logs -n greenlang -l app=greenlang-api --tail=100 | grep -i error

# Check pod status
kubectl get pods -n greenlang

# Check resource usage
kubectl top pods -n greenlang

# Check database connectivity
kubectl exec -n greenlang deployment/greenlang-api -- greenlang db check
```

**Common Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Database connection exhausted | Increase pool size or fix connection leak |
| Memory exhaustion | Increase memory limits or fix memory leak |
| Unhandled exception | Check logs, deploy fix |
| External service timeout | Check dependent services |

**Example Resolution:**

```bash
# If database pool exhausted
kubectl exec -n greenlang deployment/greenlang-api -- \
    greenlang db connections --show

# Restart pods to clear connections
kubectl rollout restart deployment/greenlang-api -n greenlang
```

---

### Issue 2: Slow Calculation Performance

**Symptoms:**
- Calculations take > 5 seconds
- Timeout errors
- Queue backlog increasing

**Diagnosis Steps:**

```python
# Enable query logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

# Check slow queries
greenlang db slow-queries --threshold 100ms

# Check worker status
greenlang worker status
```

**Common Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Missing database index | Add appropriate index |
| Large dataset processing | Implement pagination/batching |
| External API latency | Add caching layer |
| Insufficient workers | Scale horizontally |

**Performance Analysis:**

```bash
# Profile a calculation
greenlang calc profile --id calc_abc123

# Check database explain plan
greenlang db explain "SELECT * FROM calculations WHERE facility_id = 'FAC-001'"

# Monitor queue depth
greenlang queue stats
```

---

### Issue 3: Incorrect Emission Calculations

**Symptoms:**
- Emission values don't match expected
- Audit flags discrepancies
- Quality scores dropped

**Diagnosis Steps:**

```python
# Verify emission factor
from greenlang.factors import EmissionFactorRegistry

registry = EmissionFactorRegistry()
factor = registry.get("diesel", region="US", date="2025-12-07")
print(f"Factor: {factor.value} {factor.unit}")
print(f"Source: {factor.source}")
print(f"Version: {factor.version}")

# Check calculation provenance
from greenlang import GreenLang

client = GreenLang()
calc = client.get_calculation("calc_abc123")
print(f"Provenance: {calc.provenance}")
print(f"Inputs: {calc.inputs}")
print(f"Emission Factor Used: {calc.emission_factor}")
```

**Validation Checklist:**

```
[ ] Correct fuel type selected
[ ] Correct unit conversion applied
[ ] Emission factor version is current
[ ] Region mapping is correct
[ ] No data entry errors
[ ] Methodology matches requirements
```

---

### Issue 4: Data Sync Failures

**Symptoms:**
- ERP data not appearing
- "Sync failed" alerts
- Stale data in reports

**Diagnosis Steps:**

```bash
# Check connector status
greenlang connector status --name erp_sap

# View sync logs
greenlang connector logs --name erp_sap --tail 50

# Test connection
greenlang connector test --name erp_sap

# Check last successful sync
greenlang connector last-sync --name erp_sap
```

**Common Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Network connectivity | Check firewall, VPN status |
| Authentication expired | Refresh credentials |
| Schema change | Update connector mapping |
| Rate limiting | Adjust sync frequency |

**Manual Sync Recovery:**

```bash
# Retry failed sync
greenlang connector sync --name erp_sap --force

# Sync specific date range
greenlang connector sync --name erp_sap \
    --start 2025-12-01 \
    --end 2025-12-07 \
    --force
```

---

### Issue 5: Authentication Failures

**Symptoms:**
- Users cannot log in
- API returns 401/403
- Session timeouts

**Diagnosis Steps:**

```bash
# Check auth service
greenlang auth status

# Verify user exists
greenlang user get --username jsmith

# Check user permissions
greenlang user permissions --username jsmith

# Verify API key
greenlang api-key verify --key "gl_..."
```

**Common Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Password expired | Reset password |
| Account locked | Unlock account |
| MFA issue | Reset MFA |
| Token expired | Re-authenticate |
| Permission changed | Review RBAC |

---

### Issue 6: Alarm Storm

**Symptoms:**
- Excessive alarms firing
- Alarm dashboard overwhelmed
- Operator fatigue

**Diagnosis Steps:**

```bash
# Get alarm statistics
greenlang alarm stats --period 1h

# Find top alarm sources
greenlang alarm top-sources --limit 10

# Check alarm configuration
greenlang alarm config show --name high_emissions
```

**Resolution Steps:**

```python
# Suppress nuisance alarms temporarily
from greenlang.alarms import AlarmManager

am = AlarmManager()

# Shelve alarm for investigation
am.shelve(
    alarm_pattern="EMISSIONS_HIGH_*",
    duration_minutes=60,
    reason="Investigating root cause"
)

# Adjust alarm thresholds if too sensitive
am.update_threshold(
    alarm_name="high_emissions",
    new_threshold=3000,  # Was 2500
    reason="Reduced sensitivity per SOP-123"
)
```

---

## Diagnostic Tools

### CLI Diagnostics

```bash
# System health check
greenlang health check --verbose

# Component status
greenlang status --all

# Configuration validation
greenlang config validate

# Database health
greenlang db health

# Cache status
greenlang cache stats

# Queue status
greenlang queue status
```

### API Diagnostics

```bash
# Test API endpoint
curl -v https://api.greenlang.io/health

# Check response headers
curl -I https://api.greenlang.io/api/v1/calculations

# Time request
time curl -s https://api.greenlang.io/api/v1/calculations/calc_123

# Debug request
curl -v -X POST https://api.greenlang.io/api/v1/calculations \
    -H "Content-Type: application/json" \
    -d '{"fuel_type": "diesel", "quantity": 100}'
```

### Database Diagnostics

```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE datname = 'greenlang';

-- Find long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC
LIMIT 10;

-- Check table sizes
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;

-- Check index usage
SELECT indexrelname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC
LIMIT 10;
```

### Network Diagnostics

```bash
# Test database connectivity
nc -zv db-primary.internal 5432

# Test Redis connectivity
redis-cli -h redis.internal ping

# Test external API
curl -v https://external-api.example.com/health

# DNS resolution
dig api.greenlang.io

# Trace route
traceroute api.greenlang.io
```

---

## Log Analysis

### Log Locations

| Component | Location | Format |
|-----------|----------|--------|
| API Server | stdout / /var/log/greenlang/api.log | JSON |
| Workers | stdout / /var/log/greenlang/worker.log | JSON |
| Database | /var/log/postgresql/postgresql.log | Text |
| Nginx | /var/log/nginx/access.log, error.log | Combined |

### Log Analysis Commands

```bash
# Find errors in last hour
greenlang logs search --level ERROR --since 1h

# Follow logs in real-time
greenlang logs tail --follow

# Search for specific request
greenlang logs search --request-id "req_abc123"

# Find all logs for calculation
greenlang logs search --calculation-id "calc_xyz789"

# Export logs for analysis
greenlang logs export \
    --start "2025-12-07T00:00:00" \
    --end "2025-12-07T23:59:59" \
    --output logs_2025-12-07.json
```

### Log Patterns to Watch

```json
// Connection errors
{"level": "ERROR", "message": "Database connection failed", "error": "timeout"}

// Calculation failures
{"level": "ERROR", "message": "Calculation failed", "calculation_id": "calc_123", "error": "Invalid fuel type"}

// Performance warnings
{"level": "WARN", "message": "Slow query", "duration_ms": 5000, "query": "..."}

// Security events
{"level": "WARN", "message": "Authentication failed", "username": "admin", "ip": "1.2.3.4"}
```

### Using ELK/Splunk

```
# Elasticsearch query for errors
GET /greenlang-logs-*/_search
{
  "query": {
    "bool": {
      "must": [
        {"match": {"level": "ERROR"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  },
  "sort": [{"@timestamp": "desc"}],
  "size": 100
}
```

---

## Performance Troubleshooting

### Identifying Bottlenecks

```
1. Check API response times
2. Check database query times
3. Check external API latencies
4. Check queue processing times
5. Check cache hit rates
```

### Metrics to Monitor

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| API Latency P99 | < 200ms | < 1s | > 5s |
| Error Rate | < 0.1% | < 1% | > 5% |
| DB Connections | < 50% | < 80% | > 90% |
| CPU Usage | < 60% | < 80% | > 90% |
| Memory Usage | < 70% | < 85% | > 95% |
| Queue Depth | < 100 | < 500 | > 1000 |

### Performance Profiling

```python
# Profile calculation
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run calculation
result = agent.execute(input_data)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Quick Performance Fixes

```bash
# Clear cache if stale
greenlang cache clear --pattern "ef_*"

# Restart workers if stuck
kubectl rollout restart deployment/greenlang-worker -n greenlang

# Scale up temporarily
kubectl scale deployment/greenlang-api --replicas=5 -n greenlang
```

---

## Data Quality Issues

### Quality Score Analysis

```python
from greenlang.quality import QualityAnalyzer

analyzer = QualityAnalyzer()

# Analyze recent data
report = analyzer.analyze(
    date_range=("2025-12-01", "2025-12-07"),
    facility_id="FAC-001"
)

print(f"Overall Score: {report.overall_score}")
print(f"Completeness: {report.completeness}")
print(f"Accuracy: {report.accuracy}")
print(f"Consistency: {report.consistency}")

# Show issues
for issue in report.issues:
    print(f"- {issue.type}: {issue.description}")
```

### Common Data Issues

| Issue | Detection | Resolution |
|-------|-----------|------------|
| Missing values | Completeness check | Fill from source or estimate |
| Outliers | Statistical analysis | Verify or correct |
| Duplicates | Hash comparison | Remove duplicates |
| Wrong units | Unit validation | Convert units |
| Stale data | Timestamp check | Refresh from source |

### Data Correction Workflow

```python
# Identify issues
issues = client.data.find_issues(
    facility_id="FAC-001",
    date_range=("2025-12-01", "2025-12-07")
)

# Review each issue
for issue in issues:
    print(f"Record: {issue.record_id}")
    print(f"Field: {issue.field}")
    print(f"Current: {issue.current_value}")
    print(f"Expected: {issue.expected_range}")

    # Correct if needed
    if confirm_correction(issue):
        client.data.correct(
            record_id=issue.record_id,
            field=issue.field,
            new_value=correct_value,
            reason="Data entry error correction"
        )
```

---

## Integration Problems

### ERP Integration Issues

```bash
# Check ERP connector status
greenlang connector status --name sap_erp

# Test ERP connection
greenlang connector test --name sap_erp --verbose

# View recent sync errors
greenlang connector errors --name sap_erp --since 24h

# Check mapping configuration
greenlang connector mapping --name sap_erp --show
```

### API Integration Issues

```bash
# Test webhook delivery
greenlang webhook test --id wh_123

# View webhook delivery history
greenlang webhook history --id wh_123 --limit 10

# Check failed deliveries
greenlang webhook failures --since 24h

# Retry failed webhook
greenlang webhook retry --delivery-id del_456
```

### Authentication Integration

```bash
# Test SSO/SAML
greenlang auth test-sso --provider okta

# Verify JWT validation
greenlang auth verify-token --token "eyJ..."

# Check OAuth configuration
greenlang auth oauth-config --show
```

---

## Calculation Errors

### Error Categories

| Error Type | Example | Resolution |
|------------|---------|------------|
| Validation Error | "Invalid fuel type" | Check input data |
| Factor Not Found | "No emission factor for X" | Add factor or check mapping |
| Calculation Error | "Division by zero" | Check input values |
| Provenance Error | "Hash mismatch" | Verify reproducibility |
| Timeout Error | "Calculation timeout" | Optimize or increase timeout |

### Debugging Calculations

```python
# Enable debug mode
from greenlang.debug import CalculationDebugger

debugger = CalculationDebugger()

# Run with debugging
result = debugger.run(
    input_data={
        "fuel_type": "diesel",
        "quantity": 1000,
        "unit": "liters"
    },
    trace=True
)

# View execution trace
for step in result.trace:
    print(f"{step.stage}: {step.input} -> {step.output}")
    print(f"  Duration: {step.duration_ms}ms")
```

### Reproducing Calculation Issues

```python
# Get original calculation details
calc = client.get_calculation("calc_abc123")

# Reproduce with same inputs
reproduced = client.calculate(
    **calc.original_inputs,
    debug=True
)

# Compare results
if calc.result != reproduced.result:
    print("Results differ!")
    print(f"Original: {calc.result}")
    print(f"Reproduced: {reproduced.result}")
    print(f"Possible causes: emission factor update, code change")
```

---

## Hands-On Exercises

### Exercise 1: Diagnose API Error

**Scenario:** Users report that calculations are failing with "Internal Server Error"

**Steps:**
1. Check API logs for errors
2. Identify the error pattern
3. Find the root cause
4. Implement fix
5. Verify resolution

```bash
# Your commands here
greenlang logs search --level ERROR --since 1h
# ...
```

**Expected Outcome:** Identify database connection timeout and fix by increasing pool size

---

### Exercise 2: Performance Investigation

**Scenario:** Calculations that normally take 100ms are taking 10+ seconds

**Steps:**
1. Check metrics dashboard
2. Profile slow calculations
3. Identify bottleneck
4. Apply optimization
5. Verify improvement

```bash
# Your commands here
greenlang calc profile --id calc_slow123
greenlang db slow-queries --threshold 100ms
# ...
```

**Expected Outcome:** Find missing index and add it

---

### Exercise 3: Data Quality Fix

**Scenario:** Monthly report shows unexpected emission spike

**Steps:**
1. Identify affected records
2. Analyze data quality
3. Find data entry errors
4. Correct records
5. Regenerate report

```python
# Your code here
from greenlang.quality import QualityAnalyzer
# ...
```

**Expected Outcome:** Find duplicate fuel entries and remove them

---

### Exercise 4: Integration Recovery

**Scenario:** ERP sync has been failing for 3 days

**Steps:**
1. Check connector status
2. Review error logs
3. Identify failure cause
4. Fix configuration
5. Backfill missing data

```bash
# Your commands here
greenlang connector status --name erp_sap
greenlang connector logs --name erp_sap --since 72h
# ...
```

**Expected Outcome:** API credential expired, renew and resync

---

### Exercise 5: Alarm Flood

**Scenario:** 500 alarms fired in the last hour, overwhelming operators

**Steps:**
1. Get alarm statistics
2. Identify nuisance alarms
3. Shelve during investigation
4. Adjust thresholds
5. Document changes

```bash
# Your commands here
greenlang alarm stats --period 1h
greenlang alarm top-sources --limit 5
# ...
```

**Expected Outcome:** Alarm threshold too sensitive, adjust and prevent future floods

---

## Escalation Procedures

### When to Escalate

| Situation | Escalate To |
|-----------|-------------|
| Data loss risk | On-call admin + Manager |
| Security incident | Security team + CISO |
| Compliance violation | Compliance officer |
| Extended outage (>30 min) | Management |
| Customer impact | Support manager |

### Escalation Template

```markdown
## Escalation Report

**Date/Time:** YYYY-MM-DD HH:MM UTC
**Reporter:** Your Name
**Severity:** SEV1/SEV2/SEV3

### Issue Summary
Brief description of the issue

### Impact
- Number of users affected
- Business processes impacted
- Data at risk

### Timeline
- HH:MM - Issue first detected
- HH:MM - Investigation started
- HH:MM - Escalation triggered

### Actions Taken
1. Action 1
2. Action 2

### Current Status
In progress / Mitigated / Resolved

### Next Steps
1. Required action
2. Required action

### Resources Needed
- Additional personnel
- Vendor support
- Management decision
```

### Contact List

```yaml
escalation_contacts:
  on_call:
    primary: "+1-555-0100"
    secondary: "+1-555-0101"

  management:
    engineering_manager: "eng-manager@company.com"
    director: "director@company.com"

  security:
    security_team: "security@company.com"
    ciso: "ciso@company.com"

  vendor:
    greenlang_support: "support@greenlang.io"
    priority_line: "+1-555-GREENLANG"
```

---

## Quick Reference Card

### Common Commands

```bash
# Health check
greenlang health check

# View logs
greenlang logs tail --follow --level ERROR

# Check database
greenlang db health

# Restart service
kubectl rollout restart deployment/greenlang-api -n greenlang

# Clear cache
greenlang cache clear

# Check queues
greenlang queue status

# Test connectivity
greenlang connector test --name erp_sap
```

### Useful Queries

```sql
-- Recent errors
SELECT * FROM calculation_errors
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Slow calculations
SELECT * FROM calculations
WHERE processing_time_ms > 5000
ORDER BY created_at DESC LIMIT 10;

-- Active sessions
SELECT count(*) FROM user_sessions
WHERE expires_at > NOW();
```

---

## Training Complete

You have completed the GreenLang Troubleshooting Workshop. Remember:

1. **Follow the methodology** - SOLVED framework
2. **Gather evidence** - Logs, metrics, timestamps
3. **Document everything** - For future reference
4. **Know when to escalate** - Don't go it alone
5. **Learn from incidents** - Improve processes

---

**Good luck with your troubleshooting!**
