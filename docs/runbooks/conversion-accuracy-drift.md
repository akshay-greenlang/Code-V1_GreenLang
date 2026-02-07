# Conversion Accuracy Drift

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `NormalizerConversionAccuracyDrift` | Warning | Conversion factor discrepancies detected against reference values |
| `NormalizerGWPVersionMismatch` | Critical | GWP version inconsistency between services or tenants |
| `NormalizerProvenanceChainBroken` | Critical | Provenance hash verification failure on conversion results |

**Thresholds:**

```promql
# NormalizerConversionAccuracyDrift
# Detected via periodic factor validation job comparing active factors to reference sources
increase(glnorm_factor_validation_failures_total[15m]) > 0

# NormalizerGWPVersionMismatch
# Multiple GWP versions in use within the same reporting period
count(count by (gwp_source) (glnorm_ghg_conversions_total)) > 1
# sustained for 5 minutes (indicates mixed AR5/AR6 in same pipeline)

# NormalizerProvenanceChainBroken
# Provenance hash verification failures
increase(glnorm_provenance_verification_failures_total[5m]) > 0
```

---

## Description

These alerts fire when the Unit & Reference Normalizer service (AGENT-FOUND-003) produces conversion results that deviate from expected reference values, or when provenance integrity is compromised. Conversion accuracy is the foundational guarantee of the normalizer -- every conversion must be deterministic, traceable, and mathematically correct.

### How Conversions Work

The normalizer performs unit conversions through a deterministic pipeline:

1. **Unit Normalization** -- Input unit names are normalized to canonical form (lowercase, alias resolution)
2. **Dimensional Analysis** -- Source and target units are verified to belong to the same physical dimension
3. **Factor Lookup** -- Conversion factors are retrieved from the dimension-specific lookup table (base unit factors)
4. **Decimal Arithmetic** -- The conversion is calculated using Python `Decimal` with `ROUND_HALF_UP` rounding: `result = value * (from_factor / to_factor)`
5. **Precision Application** -- Result is quantized to the requested precision (default: 6 decimal places)
6. **Provenance Hashing** -- SHA-256 hash is computed over `(input, output, factor, source)` for the audit trail

For GHG conversions, an additional GWP multiplication step occurs between the factor lookup and arithmetic steps, applying the appropriate Global Warming Potential from the selected IPCC Assessment Report (AR4, AR5, or AR6).

### What Accuracy Drift Means

1. **GWP Version Mismatch**: Different parts of the pipeline use different GWP tables (e.g., one service uses AR5 CH4 GWP of 28, another uses AR6 GWP of 29.8). This produces inconsistent CO2-equivalent values for the same source data.

2. **Stale Custom Factors**: Tenant-specific conversion factors have not been updated to reflect new regulatory requirements or source data corrections. The custom factor overrides the standard factor, producing incorrect results.

3. **Unit Alias Conflicts**: Two different canonical units share an alias, causing the normalizer to resolve to the wrong unit. For example, "ton" could mean metric tonne (1000 kg), US short ton (907.18 kg), or UK long ton (1016.05 kg).

4. **Float vs. Decimal Drift**: Upstream or downstream services use IEEE 754 floating-point arithmetic instead of Decimal, causing accumulated rounding errors. The normalizer uses Decimal internally, but values passed through float-based intermediaries may drift.

5. **Factor Source Discrepancy**: A conversion factor in the database does not match the authoritative source (e.g., SI definition, IPCC table). This can happen after a manual database edit, a failed migration, or a data import error.

### Normal vs. Abnormal Accuracy Levels

| Condition | Status | Typical Cause |
|-----------|--------|---------------|
| All factors match reference sources | Normal | Standard operation |
| Single custom factor out of date | Minor | Tenant forgot to update seasonal factor |
| GWP version inconsistency across services | Serious | Configuration drift after deployment |
| Multiple standard factors differ from SI/IPCC | Critical | Database corruption or bad migration |
| Provenance hash verification failures | Critical | Data tampering or serialization bug |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Users may not notice accuracy issues until audit; silent data quality degradation |
| **Data Impact** | Critical | Incorrect conversion factors produce wrong emissions values; all downstream calculations are affected |
| **SLA Impact** | Medium | Service remains available but produces incorrect results |
| **Revenue Impact** | High | Inaccurate reports may lead to regulatory penalties for customers |
| **Compliance Impact** | Critical | Incorrect GWP values or conversion factors directly violate CSRD, CBAM, and other regulatory requirements; audit trail integrity at risk |
| **Downstream Impact** | Critical | All calculations, reports, and submissions based on incorrect conversions are invalid |

---

## Symptoms

### GWP Version Mismatch

- Different CO2e values for the same CH4 or N2O input across different API calls
- Audit log shows mixed `gwp_source` values (e.g., both "AR5" and "AR6") within the same reporting period
- Cross-system reconciliation failures between GreenLang and external systems
- `glnorm_ghg_conversions_total` metric shows multiple `gwp_source` label values active simultaneously

### Stale Custom Factors

- Tenant-specific conversions produce results that differ from expected values
- Custom factor `last_updated` timestamps are older than the current reporting period
- Entity resolution results reference outdated factor versions

### Unit Alias Conflicts

- The same input unit name produces different results at different times
- Logs show "ambiguous unit resolution" warnings
- Conversion results differ from manual calculation using the expected factor

### Float vs. Decimal Drift

- Small but systematic rounding differences (typically in the 6th-8th decimal place)
- Aggregated totals do not match sum of individual conversions
- Provenance hash mismatches when re-computing from stored float values

### Provenance Chain Broken

- `NormalizerProvenanceChainBroken` alert firing
- Audit verification returns "hash mismatch" for specific conversion records
- `glnorm_provenance_verification_failures_total` incrementing

---

## Diagnostic Steps

### Step 1: Check GWP Version Consistency

```bash
# Port-forward to the normalizer service
kubectl port-forward -n greenlang svc/normalizer-service 8080:8080

# Check which GWP source is configured as default
kubectl get configmap normalizer-service-config -n greenlang -o yaml \
  | grep -i "gwp"

# Check environment variable
kubectl get deployment normalizer-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env}' | python3 -c "
import sys, json
envs = json.loads(sys.stdin.read())
for env in envs:
    if 'GWP' in env.get('name', '').upper():
        print(f'{env[\"name\"]}={env.get(\"value\", \"<from-secret>\")}')"
```

```promql
# Check which GWP versions are in active use
count by (gwp_source) (increase(glnorm_ghg_conversions_total[1h]))

# Check if GWP version distribution has changed recently
sum by (gwp_source) (rate(glnorm_ghg_conversions_total[5m]))
```

### Step 2: Verify Standard Conversion Factors Against Reference

```bash
# Test known reference conversions
# Mass: 1 tonne = 1000 kg (exact)
curl -s -X POST http://localhost:8080/v1/normalizer/convert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"value": 1, "from_unit": "tonne", "to_unit": "kg", "precision": 10}' \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
expected = 1000.0
actual = data['converted_value']
diff = abs(actual - expected)
status = 'PASS' if diff == 0 else 'FAIL'
print(f'[{status}] 1 tonne -> kg: expected={expected}, actual={actual}, diff={diff}')
"

# Energy: 1 MWh = 3600000000 J (exact)
curl -s -X POST http://localhost:8080/v1/normalizer/convert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"value": 1, "from_unit": "MWh", "to_unit": "j", "precision": 10}' \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
expected = 3600000000.0
actual = data['converted_value']
diff = abs(actual - expected)
status = 'PASS' if diff == 0 else 'FAIL'
print(f'[{status}] 1 MWh -> J: expected={expected}, actual={actual}, diff={diff}')
"

# GHG: 1000 kgCH4 -> tCO2e (AR6 GWP = 29.8)
curl -s -X POST http://localhost:8080/v1/normalizer/ghg \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"value": 1000, "from_unit": "kgCH4", "to_unit": "tCO2e", "gwp_source": "AR6"}' \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
expected = 29.8
actual = data['converted_value']
diff = abs(actual - expected)
status = 'PASS' if diff < 0.001 else 'FAIL'
print(f'[{status}] 1000 kgCH4 -> tCO2e (AR6): expected={expected}, actual={actual}, diff={diff}')
print(f'  GWP applied: {data.get(\"gwp_applied\")}')
print(f'  GWP source: {data.get(\"gwp_source\")}')
"
```

### Step 3: Check Custom Factor Tables

```bash
# List all active custom conversion factors
kubectl run pg-factors --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT tenant_id, dimension, unit_name, to_base_factor, source, updated_at
   FROM normalizer_custom_factors
   WHERE active = true
   ORDER BY updated_at DESC
   LIMIT 50;"

# Check for factors that have not been updated in the current reporting period
kubectl run pg-stale --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT tenant_id, dimension, unit_name, to_base_factor, source, updated_at
   FROM normalizer_custom_factors
   WHERE active = true AND updated_at < NOW() - INTERVAL '90 days'
   ORDER BY updated_at ASC;"

# Check for factors that differ from standard values
kubectl run pg-diff --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT cf.tenant_id, cf.dimension, cf.unit_name, cf.to_base_factor as custom_factor,
          ud.to_base_factor as standard_factor,
          abs(cf.to_base_factor - ud.to_base_factor) / ud.to_base_factor * 100 as pct_diff
   FROM normalizer_custom_factors cf
   JOIN normalizer_unit_definitions ud ON cf.unit_name = ud.unit_name AND cf.dimension = ud.dimension
   WHERE cf.active = true
   ORDER BY pct_diff DESC
   LIMIT 20;"
```

### Step 4: Check Provenance Hash Integrity

```bash
# Check for recent provenance verification failures
kubectl logs -n greenlang -l app=normalizer-service --tail=500 \
  | grep -i "provenance\|hash.*mismatch\|integrity\|verification.*fail"

# Run a provenance verification for a specific conversion
curl -s -X POST http://localhost:8080/v1/normalizer/verify-provenance \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "conversion_id": "<conversion_id>",
    "expected_hash": "<stored_hash>"
  }' | python3 -m json.tool
```

```promql
# Check provenance verification failure rate
rate(glnorm_provenance_verification_failures_total[5m])

# Check total provenance records
glnorm_provenance_records_total
```

### Step 5: Check for Float vs. Decimal Drift

```bash
# Compare Decimal-based conversion with float-based conversion
curl -s -X POST http://localhost:8080/v1/normalizer/convert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"value": 0.1, "from_unit": "tonne", "to_unit": "kg", "precision": 15}' \
  | python3 -c "
import sys, json
from decimal import Decimal, ROUND_HALF_UP
data = json.load(sys.stdin)
actual = data['converted_value']
# Expected Decimal result: Decimal('0.1') * Decimal('1000') = Decimal('100.0')
# Float result: 0.1 * 1000 = 100.00000000000001
expected_decimal = float(Decimal('0.1') * Decimal('1000'))
expected_float = 0.1 * 1000
print(f'API result:     {actual}')
print(f'Decimal expect: {expected_decimal}')
print(f'Float expect:   {expected_float}')
print(f'API uses Decimal: {actual == expected_decimal}')
"
```

### Step 6: Check Unit Alias Resolution

```bash
# Test ambiguous unit aliases
for unit in "ton" "t" "tonne" "mt" "metric_ton" "short_ton" "long_ton"; do
  result=$(curl -s -X POST http://localhost:8080/v1/normalizer/convert \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -d "{\"value\": 1, \"from_unit\": \"$unit\", \"to_unit\": \"kg\", \"precision\": 6}")
  converted=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('converted_value', 'ERROR'))")
  echo "$unit -> kg = $converted"
done
```

---

## Resolution Steps

### Option 1: Pin GWP Version Across All Services

If the issue is GWP version inconsistency, enforce a single GWP version across the entire deployment.

```bash
# Set the GWP version explicitly in the normalizer service
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_GWP_DEFAULT_SOURCE=AR6 \
  GL_NORM_GWP_ENFORCE_CONSISTENCY=true

# Restart to apply
kubectl rollout restart deployment/normalizer-service -n greenlang

# Verify the GWP version is correct
curl -s http://localhost:8080/v1/normalizer/config \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
config = json.load(sys.stdin)
print(f'GWP source: {config.get(\"gwp_default_source\")}')
print(f'Enforce consistency: {config.get(\"gwp_enforce_consistency\")}')
"
```

**Important:** When changing GWP versions, all previously converted values in the current reporting period must be reconverted. This is a data migration, not just a configuration change.

### Option 2: Audit and Fix Custom Factors

If stale or incorrect custom factors are the cause:

```bash
# Deactivate stale custom factors (older than 90 days with no review)
kubectl run pg-fix --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "UPDATE normalizer_custom_factors
   SET active = false, deactivation_reason = 'stale_factor_audit'
   WHERE active = true AND updated_at < NOW() - INTERVAL '90 days'
   RETURNING tenant_id, dimension, unit_name, updated_at;"

# Clear the vocabulary cache to force reload
curl -X POST http://localhost:8080/v1/normalizer/cache/clear \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Restart to apply
kubectl rollout restart deployment/normalizer-service -n greenlang
```

Notify affected tenants to review and re-submit their custom factors.

### Option 3: Fix Unit Alias Conflicts

If ambiguous unit aliases are causing incorrect resolution:

```bash
# Check for duplicate alias entries in the database
kubectl run pg-aliases --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT alias, count(*), array_agg(canonical_unit || ' (' || dimension || ')') as targets
   FROM normalizer_unit_aliases
   GROUP BY alias
   HAVING count(*) > 1
   ORDER BY count(*) DESC;"

# Remove or disambiguate conflicting aliases
kubectl run pg-fix-alias --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "-- Example: ensure 'ton' maps to US short ton (not metric tonne)
   -- 'tonne' and 't' already map to metric tonne
   UPDATE normalizer_unit_aliases
   SET canonical_unit = 'short_ton', dimension = 'mass'
   WHERE alias = 'ton';"
```

### Option 4: Fix Provenance Hash Mismatches

If provenance verification is failing:

```bash
# Identify the affected conversion records
kubectl logs -n greenlang -l app=normalizer-service --tail=1000 \
  | grep "provenance.*fail\|hash.*mismatch" \
  | head -20

# Check if the issue is a serialization format change
# (e.g., JSON key ordering changed between versions)
kubectl logs -n greenlang -l app=normalizer-service --tail=500 \
  | grep -i "serializ\|json\|sort_keys\|canonical"

# If the issue is a serialization change, recompute provenance hashes
# for affected records (requires maintenance window)
curl -X POST http://localhost:8080/v1/normalizer/admin/recompute-provenance \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "start_date": "2026-01-01",
    "end_date": "2026-02-07",
    "dry_run": true
  }' | python3 -m json.tool
```

### Option 5: Reconvert Affected Data

If conversion factors were incorrect for a period of time and data needs to be corrected:

```bash
# Identify the affected time range and factor
# Step 1: Determine when the incorrect factor was active
kubectl run pg-history --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT * FROM normalizer_factor_audit_log
   WHERE dimension = '<dimension>' AND unit_name = '<unit>'
   ORDER BY changed_at DESC
   LIMIT 10;"

# Step 2: Queue a reconversion job for the affected data
curl -X POST http://localhost:8080/v1/normalizer/admin/reconvert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "affected_dimension": "emissions",
    "affected_unit": "tCO2e",
    "start_date": "2026-01-15",
    "end_date": "2026-02-07",
    "new_gwp_source": "AR6",
    "dry_run": true
  }' | python3 -m json.tool
```

---

## Post-Resolution Verification

```bash
# 1. Run the full reference factor validation suite
curl -X POST http://localhost:8080/v1/normalizer/admin/validate-factors \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Factors validated: {data.get(\"total_factors\", 0)}')
print(f'Factors correct: {data.get(\"correct\", 0)}')
print(f'Factors with issues: {data.get(\"issues\", 0)}')
for issue in data.get('issue_details', []):
    print(f'  ISSUE: {issue[\"dimension\"]}/{issue[\"unit\"]}: '
          f'expected={issue[\"expected\"]}, actual={issue[\"actual\"]}')
"
```

```promql
# 2. Verify factor validation failures have dropped to 0
rate(glnorm_factor_validation_failures_total[5m]) == 0

# 3. Verify GWP version consistency
count(count by (gwp_source) (glnorm_ghg_conversions_total)) == 1

# 4. Verify provenance verification is passing
rate(glnorm_provenance_verification_failures_total[5m]) == 0

# 5. Verify conversion accuracy metric
glnorm_conversion_accuracy_score == 1.0
```

```bash
# 6. Run reference conversion tests
tests=(
  '{"value":1,"from_unit":"tonne","to_unit":"kg","precision":10}|1000.0'
  '{"value":1,"from_unit":"MWh","to_unit":"kWh","precision":10}|1000.0'
  '{"value":1,"from_unit":"gallon","to_unit":"l","precision":6}|3.785412'
  '{"value":1,"from_unit":"mile","to_unit":"km","precision":6}|1.609344'
)
pass=0; fail=0
for test in "${tests[@]}"; do
  input="${test%|*}"
  expected="${test#*|}"
  actual=$(curl -s -X POST http://localhost:8080/v1/normalizer/convert \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -d "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('converted_value','ERROR'))")
  if [ "$actual" = "$expected" ]; then
    echo "PASS: $input -> $actual"
    ((pass++))
  else
    echo "FAIL: $input -> expected=$expected, actual=$actual"
    ((fail++))
  fi
done
echo "Results: $pass passed, $fail failed"
```

---

## Prevention

### Factor Validation Pipeline

1. **Automated daily validation job** compares all active conversion factors against authoritative reference sources (SI definitions, IPCC AR6 tables, EPA emission factor hub)
2. **CI/CD validation** runs reference conversion tests on every deployment
3. **Custom factor review process**: Tenant-submitted custom factors must be reviewed and approved before activation
4. **Version-pinned vocabularies**: Each reporting period pins a specific vocabulary version; mid-period changes require explicit migration

### GWP Version Management

1. **Pin GWP version per reporting period** -- All conversions within a reporting period must use the same GWP table
2. **Configure `GL_NORM_GWP_ENFORCE_CONSISTENCY=true`** to reject requests with mismatched GWP versions
3. **Include GWP version in provenance hash** so that version changes are detectable in the audit trail
4. **Document GWP version in regulatory submissions** -- Include the IPCC AR version used in all compliance reports

### Custom Factor Governance

1. **Require source citations** for all custom factors (regulatory reference, lab measurement, vendor datasheet)
2. **Set expiration dates** on custom factors -- factors must be re-validated periodically
3. **Maintain factor change audit log** -- Every factor change records who, when, why, and the previous value
4. **Limit custom factor deviation** -- Custom factors that deviate more than 20% from standard values require L2 approval

### Monitoring

- **Dashboard:** Normalizer Service Health (`/d/normalizer-service-health`) -- accuracy panels
- **Dashboard:** Unit Conversion Overview (`/d/normalizer-conversion-overview`)
- **Alert:** `NormalizerConversionAccuracyDrift` (this alert)
- **Alert:** `NormalizerGWPVersionMismatch` (this alert)
- **Alert:** `NormalizerProvenanceChainBroken` (this alert)
- **Key metrics to watch:**
  - `glnorm_factor_validation_failures_total` (should be 0 in steady state)
  - `glnorm_ghg_conversions_total` by `gwp_source` (should show only one version per period)
  - `glnorm_provenance_verification_failures_total` (should be 0)
  - `glnorm_custom_factors_active` count and age distribution
  - `glnorm_conversion_accuracy_score` (should be 1.0)

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any accuracy-related incident
- **Related alerts:** `NormalizerServiceDown`, `NormalizerEntityResolutionLowConfidence`
- **Related dashboards:** Normalizer Service Health, Unit Conversion Overview
- **Related runbooks:** [Normalizer Service Down](./normalizer-service-down.md), [Entity Resolution Low Confidence](./entity-resolution-low-confidence.md), [Normalizer High Latency](./normalizer-high-latency.md)
