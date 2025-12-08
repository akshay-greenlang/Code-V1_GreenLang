# Exercise 07: Troubleshooting Scenarios

**Difficulty:** Intermediate
**Duration:** 60 minutes
**Target Audience:** All roles

## Learning Objectives

- Apply systematic troubleshooting methodology
- Analyze logs and metrics
- Identify root causes
- Implement solutions

---

## Scenario 1: Calculation Failures

### Situation

Users report that emission calculations started failing around 10:00 AM. The error rate has increased from 0.1% to 15%.

### Available Information

**Error Log Excerpt:**
```json
{"timestamp": "2025-12-07T10:05:23Z", "level": "ERROR", "message": "Calculation failed", "calculation_id": "calc_789", "error": "KeyError: 'emission_factor'"}
{"timestamp": "2025-12-07T10:05:24Z", "level": "ERROR", "message": "Calculation failed", "calculation_id": "calc_790", "error": "KeyError: 'emission_factor'"}
{"timestamp": "2025-12-07T10:05:25Z", "level": "ERROR", "message": "Calculation failed", "calculation_id": "calc_791", "error": "KeyError: 'emission_factor'"}
```

**Recent Changes:**
- 09:55 AM: Deployed version 1.2.3 of emission factor service
- 09:30 AM: Database maintenance completed
- 08:00 AM: No changes

**Metrics:**
- API Latency P99: Normal (150ms)
- Database Connections: Normal (45/100)
- Redis Hit Rate: Dropped from 95% to 0%

### Tasks

1. What is the most likely root cause?
2. What diagnostic commands would you run?
3. How would you resolve the issue?
4. What preventive measures would you implement?

### Your Answers

```
1. Root Cause:
   [Your answer here]

2. Diagnostic Commands:
   [Your commands here]

3. Resolution:
   [Your steps here]

4. Prevention:
   [Your recommendations here]
```

### Solution

<details>
<summary>Click to reveal solution</summary>

**Root Cause:** The emission factor service deployment at 09:55 AM likely cleared or corrupted the Redis cache. The service is looking for cached emission factors that no longer exist.

**Diagnostic Commands:**
```bash
# Check Redis status
redis-cli ping
redis-cli info

# Check if emission factors exist in cache
greenlang cache get "ef_diesel_US"
greenlang cache stats

# Check emission factor service logs
greenlang logs search --service emission_factor --since 10:00

# Verify database has emission factors
greenlang db query "SELECT COUNT(*) FROM emission_factors"
```

**Resolution:**
```bash
# Option 1: Warm up the cache
greenlang cache warm --type emission_factors

# Option 2: Rollback deployment
kubectl rollout undo deployment/emission-factor-service

# Option 3: Fix cache key format (if changed)
# Review code changes in version 1.2.3
```

**Prevention:**
- Add cache warming to deployment pipeline
- Implement fallback to database when cache miss
- Add monitoring alert for cache hit rate drops
- Require integration tests before deployment

</details>

---

## Scenario 2: Slow Performance

### Situation

The monthly report generation is taking 45 minutes instead of the usual 5 minutes. This is blocking the compliance team.

### Available Information

**System Metrics:**
```
CPU Usage: 95%
Memory Usage: 72%
Database Connections: 98/100 (near limit)
Active Queries: 47
Longest Running Query: 12 minutes
```

**Application Logs:**
```
2025-12-07 14:30:00 WARN: Connection pool nearly exhausted (98/100)
2025-12-07 14:30:05 WARN: Query timeout warning: SELECT * FROM calculations WHERE...
2025-12-07 14:30:10 WARN: Connection pool nearly exhausted (99/100)
```

**Database Slow Query Log:**
```sql
-- Running for 12 minutes
SELECT c.*, f.name as facility_name, e.value as emission_factor
FROM calculations c
JOIN facilities f ON c.facility_id = f.id
JOIN emission_factors e ON c.fuel_type = e.fuel_type
WHERE c.created_at BETWEEN '2025-01-01' AND '2025-12-31'
ORDER BY c.created_at;
-- Rows examined: 15,000,000
```

### Tasks

1. Identify the bottleneck
2. What is the root cause of the slow query?
3. Provide immediate mitigation steps
4. Propose long-term solution

### Your Answers

```
1. Bottleneck:
   [Your answer here]

2. Root Cause:
   [Your analysis here]

3. Immediate Mitigation:
   [Your steps here]

4. Long-term Solution:
   [Your recommendations here]
```

### Solution

<details>
<summary>Click to reveal solution</summary>

**Bottleneck:** Database query performance - one query is examining 15 million rows and holding connections.

**Root Cause:**
- Missing index on `calculations.created_at` column
- Full table scan for year's worth of data
- No pagination or batching
- Connection pool being exhausted by long-running query

**Immediate Mitigation:**
```bash
# 1. Kill the long-running query
greenlang db kill-query --pid 12345

# 2. Increase connection pool temporarily
kubectl set env deployment/greenlang-api DATABASE_POOL_SIZE=150

# 3. Run report in batches (monthly)
for month in $(seq 1 12); do
    greenlang report generate --period 2025-$month --type monthly
done
```

**Long-term Solution:**
```sql
-- Add index
CREATE INDEX CONCURRENTLY idx_calculations_created_at
ON calculations(created_at);

-- Add composite index for common query patterns
CREATE INDEX CONCURRENTLY idx_calculations_facility_date
ON calculations(facility_id, created_at);
```

```python
# Implement pagination in report generation
def generate_annual_report(year):
    for month in range(1, 13):
        batch = get_calculations(year, month, batch_size=10000)
        process_batch(batch)
```

</details>

---

## Scenario 3: Data Discrepancy

### Situation

The audit team found that Q4 2025 emissions report shows 15% lower emissions than expected based on fuel purchase records.

### Available Information

**Report Summary:**
- Q4 Expected (from fuel purchases): 125,000 kg CO2e
- Q4 Reported (from GreenLang): 106,250 kg CO2e
- Difference: 18,750 kg CO2e (15%)

**Data Analysis:**
```
Fuel Records Comparison:
- Diesel purchases: 50,000 liters
- Diesel in GreenLang: 42,500 liters (15% less)

- Natural Gas purchases: 30,000 m3
- Natural Gas in GreenLang: 30,000 m3 (matches)

- Gasoline purchases: 10,000 liters
- Gasoline in GreenLang: 10,000 liters (matches)
```

**Sync Logs:**
```
2025-10-15 03:00:00 INFO: ERP sync completed - 150 records
2025-10-16 03:00:00 ERROR: ERP sync failed - connection timeout
2025-10-17 03:00:00 ERROR: ERP sync failed - connection timeout
2025-10-18 03:00:00 INFO: ERP sync completed - 145 records
...
```

### Tasks

1. What is the likely cause of the discrepancy?
2. How would you verify your hypothesis?
3. How would you correct the data?
4. How would you prevent this in the future?

### Your Answers

```
1. Likely Cause:
   [Your answer here]

2. Verification Steps:
   [Your steps here]

3. Data Correction:
   [Your procedure here]

4. Prevention:
   [Your recommendations here]
```

### Solution

<details>
<summary>Click to reveal solution</summary>

**Likely Cause:** The ERP sync failures on October 16-17 resulted in missing diesel fuel records. The 7,500 missing liters (15% of 50,000) correspond to roughly 2 days of diesel consumption.

**Verification Steps:**
```bash
# 1. Check sync history for gaps
greenlang connector history --name erp_sap --period 2025-10

# 2. Find missing dates
greenlang data gaps --type fuel --period 2025-Q4

# 3. Compare specific dates
greenlang data compare \
    --source erp_purchases \
    --target greenlang_fuel \
    --dates 2025-10-16,2025-10-17
```

**Data Correction:**
```bash
# 1. Export missing data from ERP
greenlang connector export --name erp_sap \
    --start 2025-10-16 --end 2025-10-17 \
    --output missing_fuel.csv

# 2. Review and validate
cat missing_fuel.csv

# 3. Import with audit trail
greenlang data import \
    --file missing_fuel.csv \
    --type fuel \
    --reason "Backfill for sync failure Oct 16-17" \
    --ticket INC-2025-1234

# 4. Regenerate affected reports
greenlang report regenerate --period 2025-Q4 --type emissions
```

**Prevention:**
- Add alert for consecutive sync failures
- Implement automatic retry with exponential backoff
- Add weekly data reconciliation check
- Create dashboard showing sync health
- Require signoff on reports with data gaps

</details>

---

## Scenario 4: Authentication Issues

### Situation

Multiple users from the European office report they cannot log in. US users are unaffected.

### Available Information

**Error Messages (from users):**
- "Authentication failed: Invalid credentials"
- "Unable to connect to authentication server"

**System Status:**
- US Auth Server: Healthy
- EU Auth Server: Unknown
- SSO Provider (Okta): Healthy

**Logs:**
```
2025-12-07 09:00:00 INFO: [EU] Login attempt: user@eu.company.com
2025-12-07 09:00:01 ERROR: [EU] LDAP connection failed: timeout
2025-12-07 09:00:02 INFO: [EU] Fallback to SSO
2025-12-07 09:00:03 ERROR: [EU] SSO callback failed: SSL certificate expired
```

### Tasks

1. What are the two issues affecting EU authentication?
2. Provide step-by-step resolution
3. How would you communicate with affected users?

### Your Answers

```
1. Issues:
   [Your answer here]

2. Resolution Steps:
   [Your steps here]

3. Communication:
   [Your draft message here]
```

### Solution

<details>
<summary>Click to reveal solution</summary>

**Issues:**
1. EU LDAP server is timing out (network issue or server problem)
2. SSL certificate for SSO callback has expired

**Resolution Steps:**
```bash
# 1. Check EU LDAP server
ping ldap.eu.company.internal
nc -zv ldap.eu.company.internal 389

# 2. Check SSL certificate
openssl s_client -connect auth-eu.greenlang.io:443 2>/dev/null | \
    openssl x509 -noout -dates

# 3. If certificate expired, renew immediately
certbot renew --cert-name auth-eu.greenlang.io

# 4. If LDAP server down, check with EU IT team
# Meanwhile, enable SSO-only mode for EU
kubectl set env deployment/auth-eu LDAP_ENABLED=false

# 5. Verify fix
greenlang auth test --region EU
```

**Communication:**
```
Subject: [RESOLVED] EU Authentication Issue - December 7, 2025

Hi Team,

We identified and resolved an authentication issue affecting EU users
between 09:00 and 10:30 UTC.

Root Cause: An expired SSL certificate on our EU authentication server
prevented SSO login. A secondary LDAP connectivity issue prevented the
fallback authentication.

Resolution: We have renewed the SSL certificate and restored LDAP
connectivity. All authentication services are now fully operational.

Impact: EU users were unable to log in during the affected period.
No data was lost or compromised.

Next Steps: We are implementing automated certificate renewal alerts
to prevent similar issues.

Please contact support@greenlang.io if you experience any ongoing issues.

Best regards,
GreenLang Operations Team
```

</details>

---

## Scenario 5: Alarm Flood

### Situation

At 2:00 PM, 500 alarms fired within 10 minutes. Operators are overwhelmed and cannot identify real issues.

### Available Information

**Alarm Statistics:**
```
Alarm Type              | Count | Priority
------------------------|-------|----------
EMISSIONS_HIGH          | 342   | HIGH
DATA_QUALITY_LOW        | 98    | MEDIUM
SYNC_DELAYED            | 45    | LOW
CALCULATION_FAILED      | 15    | HIGH
```

**Timeline:**
- 13:55 - Normal operations
- 14:00 - First EMISSIONS_HIGH alarms
- 14:02 - Alarm flood begins
- 14:10 - 500 alarms active

**Recent Events:**
- 13:58 - Emission factor database updated (monthly update)
- 14:00 - New emission factors took effect

### Tasks

1. What is causing the alarm flood?
2. How would you stabilize the situation immediately?
3. What is the root cause?
4. How would you prevent future alarm floods?

### Your Answers

```
1. Cause:
   [Your answer here]

2. Immediate Stabilization:
   [Your steps here]

3. Root Cause:
   [Your analysis here]

4. Prevention:
   [Your recommendations here]
```

### Solution

<details>
<summary>Click to reveal solution</summary>

**Cause:** The monthly emission factor update at 13:58 increased some factors, which caused many facilities to exceed their alarm thresholds simultaneously.

**Immediate Stabilization:**
```bash
# 1. Shelve non-critical alarms temporarily
greenlang alarm shelve \
    --pattern "EMISSIONS_HIGH*" \
    --duration 2h \
    --reason "Investigating emission factor update impact"

# 2. Suppress new alarms from same source
greenlang alarm suppress \
    --source emission_factor_change \
    --duration 2h

# 3. Notify operators
greenlang notify \
    --team operators \
    --message "Alarm flood contained. Investigating root cause."
```

**Root Cause:**
- Emission factors for diesel increased by 5% in the monthly update
- Alarm thresholds were set based on old factors
- All facilities using diesel simultaneously exceeded thresholds
- No alarm suppression for expected factor changes

**Prevention:**
```yaml
# 1. Add alarm suppression during factor updates
emission_factor_update:
  pre_actions:
    - suppress_alarms:
        pattern: "EMISSIONS_*"
        duration: "1h"
  post_actions:
    - review_thresholds:
        auto_adjust: true
        notify: compliance_team

# 2. Implement alarm rate limiting
alarm_settings:
  flood_protection:
    enabled: true
    threshold: 50  # alarms per 5 minutes
    action: suppress_low_priority

# 3. Add change management for factor updates
emission_factor_changes:
  require_review: true
  notify_before: 24h
  test_impact: true
```

</details>

---

## Evaluation Checklist

After completing all scenarios, evaluate yourself:

| Skill | Demonstrated |
|-------|-------------|
| Applied SOLVED methodology | [ ] |
| Identified correct root causes | [ ] |
| Proposed appropriate diagnostic steps | [ ] |
| Provided effective resolutions | [ ] |
| Suggested preventive measures | [ ] |
| Considered communication needs | [ ] |

## Next Steps

1. Review any scenarios you found challenging
2. Practice with the troubleshooting tools in a test environment
3. Create your own scenarios based on real incidents
4. Share learnings with your team

---

**Exercise Complete!**
