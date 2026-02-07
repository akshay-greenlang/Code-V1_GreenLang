# High Validation Error Rate

## Alert

**Alert Name:** `SchemaHighValidationErrorRate`

**Severity:** Warning

**Threshold:** `rate(glschema_validations_failed[5m]) / rate(glschema_validations_total[5m]) > 0.25` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when more than 25% of payload validation requests fail over a 5-minute window. A sustained high validation error rate indicates a systemic issue -- either the schemas have changed in a way that breaks existing data producers, the upstream data format has changed without corresponding schema updates, or a schema regression has been deployed.

### How Validation Works

The GreenLang Schema Compiler & Validator (AGENT-FOUND-002) validates payloads through multiple phases:

1. **Parse** -- Parse YAML/JSON payload into structured data
2. **Compile** -- Compile the referenced schema to IR (cached after first compilation)
3. **Structural** -- Validate types, required fields, unknown fields (GLSCHEMA-E1xx errors)
4. **Constraints** -- Validate ranges, patterns, enums, lengths (GLSCHEMA-E2xx errors)
5. **Units** -- Validate unit dimensions and compatibility (GLSCHEMA-E3xx errors)
6. **Rules** -- Evaluate cross-field validation rules (GLSCHEMA-E4xx errors)
7. **Lint** -- Check for typos, deprecated fields, naming conventions (GLSCHEMA-W6xx/W7xx warnings)

Each validation produces a report with findings categorized by error code (GLSCHEMA-E100 through GLSCHEMA-E809). Understanding the distribution of error codes is key to diagnosing the root cause.

### Normal vs. Abnormal Error Rates

| Error Rate | Status | Typical Cause |
|------------|--------|---------------|
| <5% | Normal | Occasional data entry errors, known edge cases |
| 5-15% | Elevated | New data source with formatting issues, minor schema change |
| 15-25% | High | Schema version mismatch, upstream format change |
| >25% | Alert threshold | Systemic issue requiring immediate investigation |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Data submissions are being rejected; users cannot complete compliance workflows |
| **Data Impact** | Medium | Valid data may be incorrectly rejected (false positives) or invalid data may indicate upstream quality issues |
| **SLA Impact** | Medium | Data processing throughput is reduced; regulatory submission timelines at risk |
| **Revenue Impact** | Medium | Customer experience degraded; repeated submission failures cause frustration |
| **Compliance Impact** | Medium | If validation is too strict, valid reports cannot be generated; if too lenient, invalid data may be accepted |
| **Downstream Impact** | High | Orchestrator DAG executions waiting for validated data are blocked or receive error results |

---

## Symptoms

- `SchemaHighValidationErrorRate` alert firing
- `glschema_validations_failed` counter increasing rapidly relative to `glschema_validations_total`
- Users reporting "validation failed" errors when submitting data
- Orchestrator executions failing at validation nodes
- Specific GLSCHEMA error codes appearing in high volume
- Recent schema deployment or registry update in the change log
- Upstream data source reporting format changes

---

## Diagnostic Steps

### Step 1: Check Error Distribution by Error Code

The most important diagnostic step is understanding which error codes are occurring most frequently.

```bash
# Port-forward to the schema service
kubectl port-forward -n greenlang svc/schema-service 8080:8080

# Check current metrics for error breakdown
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
total = data.get('validations_total', 0)
failed = data.get('validations_failed', 0)
success = data.get('validations_success', 0)
rate = (failed / total * 100) if total > 0 else 0
print(f'Total validations: {total}')
print(f'Successful: {success}')
print(f'Failed: {failed}')
print(f'Error rate: {rate:.1f}%')
"
```

```promql
# Error rate over time
rate(glschema_validations_failed[5m]) / rate(glschema_validations_total[5m])

# Validation error breakdown by error_code (if labeled)
topk(10, sum(rate(glschema_validation_errors_total[5m])) by (error_code))

# Validation error breakdown by schema_id
topk(10, sum(rate(glschema_validation_errors_total[5m])) by (schema_id))

# Compare current error rate to historical baseline
rate(glschema_validations_failed[5m]) / rate(glschema_validations_total[5m])
  vs
avg_over_time((rate(glschema_validations_failed[5m]) / rate(glschema_validations_total[5m]))[24h:5m])
```

### Step 2: Check Validation Logs for Error Patterns

```bash
# Get recent validation failure logs
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "validation.*failed\|valid=False\|errors="

# Look for specific error code patterns
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "GLSCHEMA-E"

# Look for schema resolution failures
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "schema not found\|version not found\|ref resolution"
```

### Step 3: Check Recent Schema Changes

```bash
# Check the schema registry for recent commits
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry log --oneline -20 --since="24 hours ago"

# Check if a specific schema was recently updated
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry log --oneline -10 -- schemas/emissions/

# Compare the latest version with the previous version
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry diff HEAD~1 HEAD -- schemas/
```

### Step 4: Check Error Distribution by Error Category

Map the most common errors to their categories to narrow down the root cause:

| Error Code Range | Category | Likely Root Cause |
|------------------|----------|-------------------|
| GLSCHEMA-E1xx | Structural | Schema added required fields, changed types, or tightened unknown field policy |
| GLSCHEMA-E2xx | Constraint | Schema changed ranges, patterns, or enum values |
| GLSCHEMA-E3xx | Unit | Unit dimensions changed or new unit requirements added |
| GLSCHEMA-E4xx | Rule | Cross-field rules changed or new conditional requirements |
| GLSCHEMA-E5xx | Schema | Schema itself is invalid, $ref resolution failing, or version mismatch |
| GLSCHEMA-E8xx | Limit | Payloads exceeding size/complexity limits |

```bash
# Count errors by category in recent logs
kubectl logs -n greenlang -l app=schema-service --tail=2000 \
  | grep -oP 'GLSCHEMA-[EW]\d+' | sort | uniq -c | sort -rn | head -20
```

### Step 5: Test Specific Failing Payloads

```bash
# Get a sample failing payload from logs or upstream service
# Then test it directly against the schema service

curl -X POST http://localhost:8080/v1/schema/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{
    "schema_ref": {
      "schema_id": "emissions/activity",
      "version": "1.3.0"
    },
    "payload": {
      "facility_id": "FAC-001",
      "year": 2025,
      "scope": "scope_1",
      "emissions_value": 1234.56
    }
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Valid: {data.get('valid')}\")
print(f\"Errors: {data.get('summary', {}).get('error_count', 0)}\")
print(f\"Warnings: {data.get('summary', {}).get('warning_count', 0)}\")
for finding in data.get('findings', []):
    print(f\"  [{finding.get('severity')}] {finding.get('code')}: {finding.get('message')}\")
"
```

### Step 6: Compare Schema Versions

If a schema was recently updated, compare the old and new versions to identify breaking changes.

```bash
# Get the current schema version
curl -s http://localhost:8080/v1/schema/emissions/activity/versions \
  -H "X-API-Key: $SCHEMA_API_KEY" | python3 -m json.tool

# Get the current schema content
curl -s http://localhost:8080/v1/schema/emissions/activity/1.3.0 \
  -H "X-API-Key: $SCHEMA_API_KEY" | python3 -m json.tool

# Get the previous schema content for comparison
curl -s http://localhost:8080/v1/schema/emissions/activity/1.2.0 \
  -H "X-API-Key: $SCHEMA_API_KEY" | python3 -m json.tool
```

### Step 7: Check Upstream Data Sources

```bash
# Check if upstream ERP connectors or intake agents are sending different data
kubectl logs -n greenlang -l app.kubernetes.io/component=intake-agent --tail=200 \
  | grep -i "format\|changed\|version\|schema"

# Check if a specific data source is generating most errors
kubectl logs -n greenlang -l app=schema-service --tail=1000 \
  | grep "valid=False" | grep -oP 'schema=\S+' | sort | uniq -c | sort -rn
```

---

## Resolution Steps

### Option 1: Rollback Schema Version

If a recent schema change is causing the errors, rollback to the previous version.

```bash
# Identify the breaking schema commit
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry log --oneline -10

# Revert the problematic commit
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry revert <commit-hash> --no-edit

# Invalidate the IR cache for the affected schema to force recompilation
curl -X POST http://localhost:8080/v1/schema/cache/invalidate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_id": "emissions/activity"}'

# Verify error rate drops
# Watch: rate(glschema_validations_failed[5m]) / rate(glschema_validations_total[5m])
```

### Option 2: Notify Upstream Data Producers

If the upstream data format has changed (not the schema), notify the data producers.

```bash
# Identify which data sources are affected
kubectl logs -n greenlang -l app=schema-service --tail=2000 \
  | grep "valid=False" | grep -oP 'trace_id=\S+' | head -20

# Create a report of the most common errors for the upstream team
kubectl logs -n greenlang -l app=schema-service --tail=5000 \
  | grep "valid=False" \
  | grep -oP 'GLSCHEMA-[EW]\d+' | sort | uniq -c | sort -rn > /tmp/error_report.txt
```

Communicate to `#data-ops` with the error breakdown and affected schemas.

### Option 3: Adjust Validation Profile

If the errors are caused by overly strict validation (e.g., after switching from "standard" to "strict" profile), temporarily switch to a more lenient profile.

```bash
# Switch to standard profile (from strict)
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_DEFAULT_PROFILE=standard

# Restart to apply
kubectl rollout restart deployment/schema-service -n greenlang
```

**Caution:** Do not switch to "permissive" profile in production unless absolutely necessary. Permissive mode disables unit validation and deprecation checks, which may allow non-compliant data through.

### Option 4: Add Coercion Rules for Known Format Issues

If the errors are caused by minor format differences (e.g., string "42" instead of integer 42), enable safe coercion.

```bash
# Enable safe coercion (string-to-number, etc.)
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_COERCION_POLICY=safe

# Restart to apply
kubectl rollout restart deployment/schema-service -n greenlang
```

### Option 5: Fix the Schema (Forward Fix)

If the schema change was intentional but introduced a backward-incompatible change, update the schema to be backward compatible.

Common backward-compatible schema changes:
- Make new required fields optional (with defaults)
- Expand enum values (do not remove existing values)
- Widen numeric ranges (do not tighten)
- Add new properties (do not remove existing ones)
- Deprecate fields with `$deprecated` instead of removing them

```bash
# Deploy the fixed schema to the registry
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry add . && \
  git -C /data/schema-registry commit -m "Fix backward compatibility for emissions/activity v1.3.0"

# Invalidate cache
curl -X POST http://localhost:8080/v1/schema/cache/invalidate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_id": "emissions/activity"}'
```

---

## Post-Resolution Verification

```bash
# 1. Verify error rate has dropped below threshold
# Watch for 10 minutes to confirm sustained improvement
```

```promql
# Error rate should drop below 25% (alert threshold) and ideally below 5%
rate(glschema_validations_failed[5m]) / rate(glschema_validations_total[5m])

# Verify validation throughput has recovered
rate(glschema_validations_total[5m])

# Check that no specific error code is dominating
topk(5, sum(rate(glschema_validation_errors_total[5m])) by (error_code))
```

```bash
# 2. Test a known-good payload to confirm validation is working
curl -X POST http://localhost:8080/v1/schema/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{
    "schema_ref": {
      "schema_id": "emissions/activity",
      "version": "1.3.0"
    },
    "payload": <known_good_payload>
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert data.get('valid') == True, f'Validation failed: {data.get(\"findings\")}'
print('Validation PASSED for known-good payload')
"
```

---

## Prevention

### Schema Versioning Best Practices

1. **Use semantic versioning** for all schemas: MAJOR.MINOR.PATCH
   - MAJOR: Backward-incompatible changes (new required fields, removed fields, tightened constraints)
   - MINOR: Backward-compatible additions (new optional fields, expanded enums)
   - PATCH: Bug fixes and documentation
2. **Never remove fields** -- use `$deprecated` with a migration path
3. **Never tighten constraints** without a MAJOR version bump
4. **Deploy schema changes in staging first** and validate against representative data

### Backward Compatibility Tests

Before deploying a schema change, run the existing test suite against the new schema:

```bash
# Run the schema compatibility check (should be part of CI/CD)
glschema validate --schema-ref emissions/activity:1.3.0 \
  --test-data tests/fixtures/emissions/valid_payloads/*.json
```

### Canary Deployment for Schema Changes

1. Deploy the new schema version alongside the old version
2. Route a small percentage of traffic to the new version
3. Monitor validation error rates for the new version
4. If error rate is acceptable, gradually increase traffic
5. Once fully migrated, mark the old version as deprecated

### Monitoring

- **Dashboard:** Schema Validation Overview (`/d/schema-validation-overview`)
- **Alert:** `SchemaHighValidationErrorRate` (this alert)
- **Key metrics to watch:**
  - `glschema_validations_failed / glschema_validations_total` (error rate by schema)
  - `glschema_validation_errors_total` by error_code (error distribution)
  - Correlation between schema registry commits and error rate spikes
  - Per-schema error rates to isolate affected schemas

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any validation error rate incident
- **Related alerts:** `SchemaServiceDown`, `SchemaCompilationFailure`
- **Related dashboards:** Schema Validation Overview, Schema Service Health
- **Related runbooks:** [Schema Service Down](./schema-service-down.md), [Schema Cache Corruption](./schema-cache-corruption.md), [Compilation Timeout](./compilation-timeout.md)
