# Assumption Validation Failures

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `AssumptionsValidationFailureSpike` | Warning | Validation failure rate exceeds 10% of total operations for 10 minutes |
| `AssumptionsValidationRuleError` | Warning | Custom validator execution errors detected |
| `AssumptionsValidationBypassed` | Critical | Validation bypass detected in strict mode (possible misconfiguration) |

**Thresholds:**

```promql
# AssumptionsValidationFailureSpike
# Validation failures exceeding 10% of total assumption write operations
rate(gl_assumptions_validation_failures_total[10m]) /
  rate(gl_assumptions_operations_total{operation="write"}[10m]) > 0.1
# sustained for 10 minutes

# AssumptionsValidationRuleError
# Custom validator execution errors (not validation failures, but validator crashes)
increase(gl_assumptions_validation_rule_errors_total[5m]) > 0

# AssumptionsValidationBypassed
# Validation bypass in strict mode (should never happen)
increase(gl_assumptions_validation_bypassed_total[5m]) > 0
```

---

## Description

These alerts fire when the validation engine of the Assumptions Registry service (AGENT-FOUND-004) detects an abnormal rate of validation failures or validator execution errors. The validation engine is a critical component of the zero-hallucination guarantee -- it ensures that every assumption value persisted in the registry conforms to defined constraints before it is available for use in compliance calculations.

### How Assumption Validation Works

Every assumption write operation (create, update, import) passes through the validation engine before persistence:

1. **Type Validation** -- The assumption value is checked against its declared data type (numeric, string, boolean, date, enum)
2. **Range Validation** -- Numeric values are checked against optional `min_value` and `max_value` constraints
3. **Allowed Values Validation** -- Enum-type assumptions are checked against a whitelist of permitted values
4. **Regex Pattern Validation** -- String values are checked against optional regex patterns (e.g., ISO date formats, unit codes)
5. **Cross-Reference Validation** -- Values that reference other assumptions (e.g., a factor that depends on a base rate) are verified to reference existing, active assumptions
6. **Custom Validator Execution** -- If a custom validator function is registered for the assumption category, it is invoked with the value and context. Custom validators can enforce domain-specific business rules (e.g., "emission factor for coal must be between 1.5 and 3.0 tCO2/t")
7. **Provenance Validation** -- The change reason, user ID, and source citation are verified to be non-empty (required for audit trail)
8. **Scenario Constraint Validation** -- If the value is a scenario override, it is checked against the scenario's constraint rules (e.g., optimistic scenario values must be <= baseline)

### What Validation Failures Mean

1. **Legitimate Data Quality Issues**: The most common cause -- users or integrations are submitting assumption values that genuinely violate the defined constraints. This may indicate upstream data quality problems in the source systems feeding assumption values.

2. **Overly Strict Validation Rules**: Validation rules may have been defined too narrowly for the actual range of valid values. For example, a min/max range based on historical data may not accommodate new regulatory requirements or updated emission factors.

3. **Custom Validator Bugs**: A custom validator function may contain logic errors that incorrectly reject valid values or crash during execution.

4. **Schema Evolution Mismatch**: The assumption schema has been updated (e.g., new required fields, changed data types) but existing integrations have not been updated to comply with the new schema.

5. **Bulk Import Issues**: A bulk import of assumptions from a CSV or external system may contain formatting errors, unit mismatches, or values outside expected ranges.

6. **Scenario Configuration Errors**: Scenario override values may violate scenario-specific constraints (e.g., a conservative scenario value that is lower than the baseline when it should be higher).

### Normal vs. Abnormal Validation Failure Rates

| Condition | Status | Typical Cause |
|-----------|--------|---------------|
| <1% validation failure rate | Normal | Occasional user input errors, caught and corrected |
| 1-5% failure rate | Elevated | New data source with formatting issues |
| 5-10% failure rate | Warning | Systematic issue with an integration or import |
| >10% failure rate | Alert | Validation rules mismatch, bulk import error, or validator bug |
| Custom validator errors | Alert | Validator code bug, missing dependencies, or runtime error |
| Validation bypass in strict mode | Critical | Configuration error or security concern |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Users see "validation failed" errors when creating or updating assumptions; work is blocked until values are corrected or rules are adjusted |
| **Data Impact** | Low-Medium | Failed validations prevent bad data from entering the registry (this is working as intended); however, legitimate values may be rejected if rules are too strict |
| **SLA Impact** | Low | Service remains available; only write operations with invalid data are rejected |
| **Revenue Impact** | Low-Medium | Customer workflows may be delayed if assumption updates are blocked by overly strict rules |
| **Compliance Impact** | Medium | If validation rules are too strict, legitimate regulatory updates may be delayed; if too lenient, incorrect values could enter the registry |
| **Downstream Impact** | Medium | Calculations depending on updated assumption values cannot proceed until the values pass validation |

---

## Symptoms

### Validation Failure Spike

- `AssumptionsValidationFailureSpike` alert firing
- `gl_assumptions_validation_failures_total` counter incrementing rapidly
- Users reporting "validation failed" errors when creating or updating assumptions
- Bulk import operations returning high failure counts
- `gl_assumptions_operations_total{operation="write", status="validation_failed"}` increasing

### Custom Validator Errors

- `AssumptionsValidationRuleError` alert firing
- `gl_assumptions_validation_rule_errors_total` counter incrementing
- Logs showing "custom validator execution error" or "validator timeout"
- Specific assumption categories consistently failing validation

### Validation Bypass

- `AssumptionsValidationBypassed` alert firing
- `gl_assumptions_validation_bypassed_total` counter incrementing (should be 0 in strict mode)
- Audit log showing writes without validation records

---

## Diagnostic Steps

### Step 1: Check Validation Failure Metrics

```bash
# Port-forward to the assumptions service
kubectl port-forward -n greenlang svc/assumptions-service 8080:8080

# Get current validation metrics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
total_ops = data.get('operations_total', 0)
write_ops = data.get('operations_write_total', 0)
val_failures = data.get('validation_failures_total', 0)
val_errors = data.get('validation_rule_errors_total', 0)
val_bypassed = data.get('validation_bypassed_total', 0)
failure_rate = (val_failures / write_ops * 100) if write_ops > 0 else 0
print(f'Total operations: {total_ops}')
print(f'Write operations: {write_ops}')
print(f'Validation failures: {val_failures} ({failure_rate:.1f}%)')
print(f'Validator errors: {val_errors}')
print(f'Validation bypassed: {val_bypassed}')
"
```

```promql
# Validation failure rate over time
rate(gl_assumptions_validation_failures_total[5m]) /
  rate(gl_assumptions_operations_total{operation="write"}[5m])

# Failure breakdown by validation rule type
sum by (rule_type) (rate(gl_assumptions_validation_failures_total[5m]))

# Failure breakdown by assumption category
sum by (category) (rate(gl_assumptions_validation_failures_total[5m]))

# Custom validator error rate
rate(gl_assumptions_validation_rule_errors_total[5m])

# Validation bypass rate (should be 0)
rate(gl_assumptions_validation_bypassed_total[5m])
```

### Step 2: Query Recent Validation Failures from the Change Log

```bash
# Get the most recent validation failures from the database
kubectl run pg-failures --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT assumption_key, attempted_value, validation_rule, failure_reason,
          user_id, created_at
   FROM assumptions_change_log
   WHERE status = 'validation_failed'
   ORDER BY created_at DESC
   LIMIT 30;"

# Identify which validation rules are failing most often
kubectl run pg-rule-stats --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT validation_rule, failure_reason, count(*) as failure_count,
          count(DISTINCT assumption_key) as affected_assumptions,
          count(DISTINCT user_id) as affected_users,
          min(created_at) as first_failure, max(created_at) as last_failure
   FROM assumptions_change_log
   WHERE status = 'validation_failed'
     AND created_at > NOW() - INTERVAL '24 hours'
   GROUP BY validation_rule, failure_reason
   ORDER BY failure_count DESC
   LIMIT 20;"

# Identify which assumption categories have the most failures
kubectl run pg-cat-stats --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT ar.category, cl.validation_rule, count(*) as failure_count
   FROM assumptions_change_log cl
   JOIN assumptions_registry ar ON cl.assumption_key = ar.key
   WHERE cl.status = 'validation_failed'
     AND cl.created_at > NOW() - INTERVAL '24 hours'
   GROUP BY ar.category, cl.validation_rule
   ORDER BY failure_count DESC
   LIMIT 20;"
```

### Step 3: Check Validation Rule Definitions

```bash
# List all active validation rules
kubectl run pg-rules --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT vr.id, vr.assumption_key, vr.rule_type, vr.rule_config,
          vr.is_active, vr.created_at, vr.updated_at
   FROM assumptions_validation_rules vr
   WHERE vr.is_active = true
   ORDER BY vr.assumption_key
   LIMIT 50;"

# Check rules for a specific failing assumption
kubectl run pg-specific-rule --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT vr.rule_type, vr.rule_config, ar.value as current_value,
          ar.category, ar.unit
   FROM assumptions_validation_rules vr
   JOIN assumptions_registry ar ON vr.assumption_key = ar.key
   WHERE vr.assumption_key = '<failing_assumption_key>'
     AND vr.is_active = true;"

# Check for rules with very narrow ranges
kubectl run pg-narrow-rules --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT vr.assumption_key, vr.rule_config::json->>'min_value' as min_val,
          vr.rule_config::json->>'max_value' as max_val,
          ar.value as current_value, ar.unit
   FROM assumptions_validation_rules vr
   JOIN assumptions_registry ar ON vr.assumption_key = ar.key
   WHERE vr.rule_type = 'range'
     AND vr.is_active = true
     AND (vr.rule_config::json->>'max_value')::float -
         (vr.rule_config::json->>'min_value')::float <
         (vr.rule_config::json->>'min_value')::float * 0.1
   ORDER BY vr.assumption_key;"
```

### Step 4: Check Custom Validators

```bash
# Check logs for custom validator errors
kubectl logs -n greenlang -l app=assumptions-service --tail=500 \
  | grep -i "custom.*valid\|validator.*error\|validator.*fail\|validator.*timeout"

# List custom validators registered in the system
kubectl run pg-validators --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT vr.assumption_key, vr.rule_config::json->>'validator_name' as validator,
          vr.rule_config::json->>'timeout_ms' as timeout_ms,
          vr.is_active, vr.updated_at
   FROM assumptions_validation_rules vr
   WHERE vr.rule_type = 'custom'
     AND vr.is_active = true
   ORDER BY vr.updated_at DESC;"

# Check if any custom validators are timing out
kubectl logs -n greenlang -l app=assumptions-service --tail=1000 \
  | grep -i "validator.*timeout\|custom.*exceed\|validator.*slow"
```

### Step 5: Check for Recent Bulk Imports

```bash
# Check if a bulk import is the source of failures
kubectl logs -n greenlang -l app=assumptions-service --tail=500 \
  | grep -i "import\|bulk\|batch.*write\|csv\|upload"

# Check recent import job status
kubectl run pg-imports --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT cl.batch_id, count(*) as total_records,
          count(*) FILTER (WHERE cl.status = 'validation_failed') as failed,
          count(*) FILTER (WHERE cl.status = 'success') as succeeded,
          min(cl.created_at) as started, max(cl.created_at) as ended,
          cl.user_id
   FROM assumptions_change_log cl
   WHERE cl.batch_id IS NOT NULL
     AND cl.created_at > NOW() - INTERVAL '24 hours'
   GROUP BY cl.batch_id, cl.user_id
   ORDER BY started DESC
   LIMIT 10;"
```

### Step 6: Check Scenario Constraint Violations

```bash
# Check if scenario overrides are failing validation
kubectl run pg-scenario-fail --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT cl.assumption_key, cl.attempted_value, cl.validation_rule,
          cl.failure_reason, s.name as scenario_name
   FROM assumptions_change_log cl
   JOIN assumptions_scenarios s ON cl.scenario_id = s.id
   WHERE cl.status = 'validation_failed'
     AND cl.scenario_id IS NOT NULL
     AND cl.created_at > NOW() - INTERVAL '24 hours'
   ORDER BY cl.created_at DESC
   LIMIT 20;"
```

### Step 7: Check Validation Mode Configuration

```bash
# Verify the validation mode is set correctly
kubectl get configmap assumptions-service-config -n greenlang -o yaml \
  | grep -i "validation"

# Check environment variables
kubectl get deployment assumptions-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env}' | python3 -c "
import sys, json
envs = json.loads(sys.stdin.read())
for env in envs:
    name = env.get('name', '')
    if any(k in name.upper() for k in ['VALIDATION', 'STRICT', 'RULE', 'CONSTRAINT']):
        print(f'{name}={env.get(\"value\", \"<from-secret>\")}')"
```

---

## Resolution Steps

### Option 1: Update Validation Rules That Are Too Strict

If the validation rules have ranges or constraints that are too narrow for legitimate new values:

```bash
# Step 1: Identify the overly strict rules (from Step 3 above)

# Step 2: Update the rule via API
curl -X PATCH http://localhost:8080/v1/assumptions/admin/validation-rules/<rule_id> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "rule_config": {
      "min_value": 0.5,
      "max_value": 5.0
    },
    "change_reason": "Expanded range to accommodate updated regulatory emission factors"
  }' | python3 -m json.tool

# Step 3: Verify the rule change
curl -s http://localhost:8080/v1/assumptions/admin/validation-rules/<rule_id> \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -m json.tool
```

**Important:** Rule changes are themselves audited. Always provide a `change_reason` documenting why the rule was modified.

### Option 2: Fix Incorrect Assumption Values

If the values being submitted are genuinely incorrect:

```bash
# Step 1: Identify the failing values from the change log (Step 2 above)

# Step 2: Communicate to the submitting user/integration about the correct values
# Check who is submitting the invalid values
kubectl run pg-users --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT user_id, count(*) as failure_count,
          array_agg(DISTINCT assumption_key) as affected_keys
   FROM assumptions_change_log
   WHERE status = 'validation_failed'
     AND created_at > NOW() - INTERVAL '24 hours'
   GROUP BY user_id
   ORDER BY failure_count DESC;"

# Step 3: If the values need to be corrected in the source system,
# contact the data source owner
```

### Option 3: Fix Custom Validator Errors

If a custom validator is crashing or producing incorrect results:

```bash
# Step 1: Identify the failing validator (from Step 4 above)

# Step 2: Temporarily disable the failing custom validator
curl -X PATCH http://localhost:8080/v1/assumptions/admin/validation-rules/<rule_id> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "is_active": false,
    "change_reason": "Temporarily disabled due to validator execution errors. Ticket: PLAT-XXXX"
  }' | python3 -m json.tool

# Step 3: File a bug report for the custom validator
# Include the error logs from Step 4

# Step 4: After the validator is fixed, re-enable it
curl -X PATCH http://localhost:8080/v1/assumptions/admin/validation-rules/<rule_id> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "is_active": true,
    "change_reason": "Re-enabled after validator fix. Ticket: PLAT-XXXX"
  }' | python3 -m json.tool
```

### Option 4: Fix Bulk Import Data Quality

If a bulk import is the source of validation failures:

```bash
# Step 1: Export the failed import records for review
kubectl run pg-export-fail --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "COPY (
    SELECT assumption_key, attempted_value, validation_rule, failure_reason, created_at
    FROM assumptions_change_log
    WHERE status = 'validation_failed'
      AND batch_id = '<batch_id>'
    ORDER BY created_at
  ) TO STDOUT WITH CSV HEADER"

# Step 2: Fix the import file and re-submit
# Common fixes:
# - Correct numeric formatting (comma vs period decimal separators)
# - Fix unit mismatches (ensure values match the expected unit)
# - Remove out-of-range values
# - Fix date formatting (ISO 8601 required)

# Step 3: Re-import the corrected file
curl -X POST http://localhost:8080/v1/assumptions/import \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "file_url": "https://example.com/corrected_assumptions.csv",
    "format": "CSV",
    "validation_mode": "strict",
    "change_reason": "Re-import with corrected values after validation failure"
  }' | python3 -m json.tool
```

### Option 5: Add New Validation Rules for New Use Cases

If new assumption types need validation rules that do not yet exist:

```bash
# Add a range validation rule for a new assumption
curl -X POST http://localhost:8080/v1/assumptions/admin/validation-rules \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "assumption_key": "emission_factor_coal_combustion",
    "rule_type": "range",
    "rule_config": {
      "min_value": 1.5,
      "max_value": 3.5,
      "unit": "tCO2/t"
    },
    "is_active": true,
    "change_reason": "Added range constraint based on IPCC AR6 Table 2.2 values"
  }' | python3 -m json.tool

# Add an allowed values validation rule
curl -X POST http://localhost:8080/v1/assumptions/admin/validation-rules \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "assumption_key": "gwp_source",
    "rule_type": "allowed_values",
    "rule_config": {
      "allowed": ["AR4", "AR5", "AR6"]
    },
    "is_active": true,
    "change_reason": "Restricted GWP source to supported IPCC Assessment Report versions"
  }' | python3 -m json.tool

# Add a regex pattern validation rule
curl -X POST http://localhost:8080/v1/assumptions/admin/validation-rules \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "assumption_key": "reporting_period",
    "rule_type": "regex",
    "rule_config": {
      "pattern": "^\\d{4}-(Q[1-4]|H[12]|FY)$"
    },
    "is_active": true,
    "change_reason": "Enforced reporting period format (e.g., 2026-Q1, 2026-H1, 2026-FY)"
  }' | python3 -m json.tool
```

### Option 6: Update Scenario Constraints

If scenario overrides are failing because scenario constraints are misconfigured:

```bash
# Check current scenario constraint configuration
curl -s http://localhost:8080/v1/assumptions/scenarios \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('scenarios', []):
    print(f\"Scenario: {s['name']}\")
    for k, v in s.get('constraints', {}).items():
        print(f\"  {k}: {v}\")
"

# Update scenario constraints
curl -X PATCH http://localhost:8080/v1/assumptions/scenarios/<scenario_id> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "constraints": {
      "override_must_be_within_pct_of_baseline": 50,
      "allow_negative_values": false
    },
    "change_reason": "Relaxed override range from 20% to 50% of baseline to accommodate sensitivity analysis"
  }' | python3 -m json.tool
```

---

## Post-Resolution Verification

```promql
# 1. Validation failure rate should be below 5%
rate(gl_assumptions_validation_failures_total[5m]) /
  rate(gl_assumptions_operations_total{operation="write"}[5m]) < 0.05

# 2. Custom validator errors should be 0
rate(gl_assumptions_validation_rule_errors_total[5m]) == 0

# 3. Validation bypass count should be 0
gl_assumptions_validation_bypassed_total == 0

# 4. Write operations should be succeeding
rate(gl_assumptions_operations_total{operation="write", status="success"}[5m])

# 5. Failure rate by rule type should be decreasing
sum by (rule_type) (rate(gl_assumptions_validation_failures_total[5m]))
```

```bash
# 6. Test a write operation that was previously failing
curl -X POST http://localhost:8080/v1/assumptions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "key": "<previously_failing_key>",
    "value": <corrected_value>,
    "unit": "<unit>",
    "category": "<category>",
    "source": "<source>",
    "effective_date": "2026-01-01",
    "change_reason": "Post-resolution verification test"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('error'):
    print(f'STILL FAILING: {data.get(\"message\")}')
    print(f'Rule: {data.get(\"validation_rule\")}')
else:
    print(f'SUCCESS: key={data.get(\"key\")}, version={data.get(\"version\")}')
"

# 7. Verify the change log shows the validation pass
kubectl run pg-verify --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT assumption_key, status, validation_rule, created_at
   FROM assumptions_change_log
   WHERE created_at > NOW() - INTERVAL '10 minutes'
   ORDER BY created_at DESC
   LIMIT 10;"
```

---

## Prevention

### Validation Rule Governance

1. **Define validation rules based on authoritative sources** -- Use IPCC tables, regulatory limits, and industry standards as the basis for min/max ranges, not arbitrary guesses
2. **Include headroom in range constraints** -- Set min/max ranges to at least 2x the expected range to accommodate future updates without rule changes
3. **Version-pin validation rules to reporting periods** -- Rules active during a reporting period should not be modified mid-period; create new rules for new periods
4. **Require change reasons for all rule modifications** -- Every rule change must include a documented reason and source citation
5. **Review custom validators quarterly** -- Ensure custom validator logic is tested, maintained, and has adequate error handling

### Import Data Quality

1. **Validate import files before submission** -- Provide a dry-run/preview endpoint that validates without persisting
2. **Use the import template** -- Provide downloadable CSV/Excel templates with the correct column headers, data types, and example values
3. **Set import size limits** -- Limit bulk imports to 10,000 records per batch to make error identification manageable
4. **Include validation summary in import results** -- Return a detailed report of passed/failed records with specific failure reasons

### Monitoring

- **Dashboard:** Assumptions Registry Health (`/d/assumptions-service-health`) -- validation panels
- **Dashboard:** Assumptions Operations Overview (`/d/assumptions-operations-overview`)
- **Alert:** `AssumptionsValidationFailureSpike` (this alert)
- **Alert:** `AssumptionsValidationRuleError` (this alert)
- **Alert:** `AssumptionsValidationBypassed` (this alert)
- **Key metrics to watch:**
  - `gl_assumptions_validation_failures_total` rate and breakdown by `rule_type` and `category`
  - `gl_assumptions_validation_rule_errors_total` (should be 0)
  - `gl_assumptions_validation_bypassed_total` (should be 0 in strict mode)
  - `gl_assumptions_operations_total{operation="write", status="success"}` rate (should be high relative to total writes)
  - `gl_assumptions_import_records_total` and `gl_assumptions_import_failures_total` (import health)

### Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Validation failure rate elevated (<25%), limited to a single category | On-call engineer | Within 30 minutes |
| L2 | Validation failure rate >25% or affecting multiple categories | Platform team lead + data quality team | 15 minutes |
| L3 | Custom validator crashing, affecting compliance calculations | Platform team + compliance team | Immediate |
| L4 | Validation bypass detected in strict mode | Security team + platform team + incident commander | Immediate |

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team + Data Quality Team
- **Review cadence:** Quarterly or after any validation-related incident
- **Related alerts:** `AssumptionsServiceDown`, `AssumptionsAuditGap`
- **Related dashboards:** Assumptions Registry Health, Assumptions Operations Overview
- **Related runbooks:** [Assumptions Service Down](./assumptions-service-down.md), [Scenario Drift Detection](./scenario-drift-detection.md), [Assumptions Audit Compliance](./assumptions-audit-compliance.md)
