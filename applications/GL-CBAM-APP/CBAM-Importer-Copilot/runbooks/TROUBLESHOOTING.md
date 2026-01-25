# CBAM IMPORTER COPILOT - TROUBLESHOOTING GUIDE

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Owner:** GreenLang CBAM Engineering Team

---

## PURPOSE

This guide provides step-by-step troubleshooting procedures for common issues encountered when operating the CBAM Importer Copilot. Each issue includes symptoms, diagnostic steps, root causes, and resolution procedures.

---

## TABLE OF CONTENTS

1. [Emissions Calculation Discrepancies](#1-emissions-calculation-discrepancies)
2. [Shipment Validation Failures](#2-shipment-validation-failures)
3. [CN Code Enrichment Errors](#3-cn-code-enrichment-errors)
4. [Supplier Data Linking Issues](#4-supplier-data-linking-issues)
5. [Performance Degradation](#5-performance-degradation)
6. [Pipeline Timeout Failures](#6-pipeline-timeout-failures)
7. [Database Connection Errors](#7-database-connection-errors)
8. [Report Generation Failures](#8-report-generation-failures)
9. [High Memory Usage / OOM Kills](#9-high-memory-usage--oom-kills)
10. [CBAM Compliance Validation Errors](#10-cbam-compliance-validation-errors)

---

## 1. EMISSIONS CALCULATION DISCREPANCIES

### Symptoms
- Calculated emissions don't match expected values
- Different results for same shipment on re-processing
- Emissions significantly higher/lower than industry benchmarks
- Audit trail shows unexpected emission factor selection

### Diagnostic Steps

```bash
# 1. Check emission factor database version
grep "VERSION" data/emission_factors.py

# 2. Check calculation audit trail for specific shipment
kubectl exec -n greenlang deployment/cbam-importer -- \
  python -c "
from agents.emissions_calculator_agent_v2 import EmissionsCalculatorAgent
import json
# Load shipment
with open('/tmp/problem_shipment.json') as f:
    shipment = json.load(f)
# Get calculation details
agent = EmissionsCalculatorAgent()
result = agent.calculate_with_audit_trail(shipment)
print(json.dumps(result, indent=2))
"

# 3. Compare supplier actual vs default emissions
python scripts/compare_emissions_methods.py \
  --shipment-id SHP-12345 \
  --output /tmp/comparison.json

# 4. Verify emission factor lookup
python -c "
from data.emission_factors import EMISSION_FACTORS_DB
cn_code = '72031000'
if cn_code in EMISSION_FACTORS_DB:
    print(EMISSION_FACTORS_DB[cn_code])
else:
    print(f'CN code {cn_code} not found in emission factors database')
"
```

### Common Root Causes

#### A. Incorrect Emission Factor Selection

**Cause:** Supplier actual data not being selected (falling back to defaults incorrectly)

**Solution:**
1. Check supplier profile has `has_actual_emissions: true`
```bash
cat examples/demo_suppliers.yaml | grep -A 20 "supplier_id: SUPPLIER-ID"
```

2. Verify actual emissions data structure:
```yaml
actual_emissions:
  - cn_code: "72031000"
    direct_tco2_per_ton: 1.85
    indirect_tco2_per_ton: 0.42
    total_tco2_per_ton: 2.27
    vintage: 2024
    certification: ISO 14064-1
```

3. Check agent priority logic:
```python
# agents/emissions_calculator_agent_v2.py
# Should prioritize: supplier actual > EU default > error
```

#### B. Unit Conversion Errors

**Cause:** Mass not being converted correctly (kg vs tons)

**Solution:**
1. Verify shipment mass units:
```bash
# Shipments should be in kilograms
jq '.shipments[0].net_mass_kg' validated_shipments.json
```

2. Check calculation:
```python
# Correct calculation
emissions_tco2 = (net_mass_kg / 1000) * emission_factor_tco2_per_ton
```

3. Add test case for unit conversion:
```python
def test_unit_conversion():
    # 10,000 kg steel @ 2.0 tCO2/ton = 20 tCO2
    assert calculate_emissions(10000, 2.0) == 20.0
```

#### C. Rounding Differences

**Cause:** Inconsistent rounding across calculation steps

**Solution:**
1. Check rounding configuration:
```python
# config.py
EMISSIONS_ROUNDING_DECIMALS = 4  # tCO2 to 4 decimal places
```

2. Ensure consistent rounding:
```python
# Always round at the final step, not intermediate steps
emissions_tco2 = round(
    (net_mass_kg / 1000.0) * emission_factor,
    EMISSIONS_ROUNDING_DECIMALS
)
```

3. Document rounding policy:
```markdown
# CBAM Rounding Policy
- Mass: 2 decimal places (kg)
- Emission factors: 4 decimal places (tCO2/ton)
- Final emissions: 4 decimal places (tCO2)
- Percentages: 2 decimal places (%)
```

#### D. Emission Factor Database Out of Date

**Cause:** Using old emission factors

**Solution:**
1. Check emission factor vintage:
```bash
grep "vintage" data/emission_factors.py | sort | uniq
```

2. Update emission factors from authoritative sources:
```bash
# IEA Cement Technology Roadmap 2024
# IPCC Guidelines 2023 revision
# World Steel Association LCA 2024
# IAI GHG Protocol 2024
```

3. Implement version control:
```bash
git log --oneline data/emission_factors.py
```

---

## 2. SHIPMENT VALIDATION FAILURES

### Symptoms
- High percentage of shipments rejected (>10%)
- Specific validation rules failing frequently
- Valid-looking shipments being rejected
- Validation error messages unclear

### Diagnostic Steps

```bash
# 1. Get validation error summary
curl http://cbam-importer:8000/reports/validation-errors | jq '.error_summary'

# 2. Check validation rules
cat rules/cbam_rules.yaml | grep -A 5 "rule_id: E002"

# 3. Analyze specific rejection
kubectl logs -n greenlang deployment/cbam-importer | grep "shipment_id: SHP-12345"

# 4. Test single shipment validation
python -c "
from agents.shipment_intake_agent_v2 import ShipmentIntakeAgent
import json
shipment = {
    'shipment_id': 'TEST-001',
    'cn_code': '72031000',
    'origin_country': 'CN',
    'net_mass_kg': 25000,
    'import_date': '2025-10-01'
}
agent = ShipmentIntakeAgent()
result = agent.validate_single_shipment(shipment)
print(json.dumps(result, indent=2))
"
```

### Common Root Causes

#### A. Invalid CN Code Format

**Error:** `E002: Invalid CN code - must be 8 digits`

**Solution:**
1. Check CN code format in input data:
```bash
# CN codes must be exactly 8 digits, can include spaces/dashes
# Valid: 72031000, 7203 10 00, 7203-10-00
# Invalid: 7203, 720310, 72.03.10.00
```

2. Implement CN code normalization:
```python
def normalize_cn_code(cn_code: str) -> str:
    """Remove spaces and dashes, ensure 8 digits"""
    normalized = cn_code.replace(' ', '').replace('-', '')
    if len(normalized) != 8 or not normalized.isdigit():
        raise ValueError(f"Invalid CN code format: {cn_code}")
    return normalized
```

#### B. Missing Required Fields

**Error:** `E001: Missing required field`

**Solution:**
1. Check all required fields are present:
```python
REQUIRED_FIELDS = [
    'shipment_id',
    'cn_code',
    'origin_country',
    'net_mass_kg',
    'import_date'
]
```

2. Validate input CSV/JSON schema:
```bash
# Use JSON Schema validator
python -c "
import jsonschema
import json
with open('schemas/shipment.schema.json') as f:
    schema = json.load(f)
with open('/tmp/problem_shipments.json') as f:
    data = json.load(f)
jsonschema.validate(data, schema)
"
```

#### C. Date Format Issues

**Error:** `E003: Invalid date format`

**Solution:**
1. Ensure dates are in YYYY-MM-DD format:
```bash
# Valid: 2025-10-01
# Invalid: 10/01/2025, 01-10-2025, 2025.10.01
```

2. Implement date parsing with multiple formats:
```python
from datetime import datetime
def parse_date(date_str: str) -> datetime:
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")
```

#### D. Country Code Validation Failure

**Error:** `E009: Country not in EU`

**Cause:** Trying to validate importer country against EU list when it's origin country

**Solution:**
1. Clarify validation logic:
```python
# Origin country: Any valid ISO 2-letter code (for exports TO EU)
# Importer country: Must be EU27 member state
EU27 = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE"}
```

2. Add validation context to error messages:
```python
# Clear error message
if origin_country not in valid_countries:
    raise ValidationError(
        f"Origin country '{origin_country}' not recognized. "
        f"Must be valid ISO 3166-1 alpha-2 code."
    )
```

---

## 3. CN CODE ENRICHMENT ERRORS

### Symptoms
- Product descriptions not being added
- Product group classification incorrect
- "CN code not found" errors for valid CBAM products
- Missing metadata in output

### Diagnostic Steps

```bash
# 1. Check CN code database
cat data/cn_codes.json | jq 'keys | length'

# 2. Search for specific CN code
cat data/cn_codes.json | jq '."72031000"'

# 3. List all CBAM product groups
cat data/cn_codes.json | jq '[.[] | .product_group] | unique'

# 4. Test CN code lookup
python -c "
import json
with open('data/cn_codes.json') as f:
    cn_codes = json.load(f)
cn_code = '72031000'
if cn_code in cn_codes:
    print(json.dumps(cn_codes[cn_code], indent=2))
else:
    print(f'CN code {cn_code} not found')
"
```

### Common Root Causes

#### A. CN Code Not in CBAM Annex I

**Cause:** Product not covered by CBAM regulation

**Solution:**
1. Verify CN code is CBAM-covered:
```bash
# CBAM Annex I product groups:
# - Cement (2523*)
# - Iron & Steel (72*, 73*)
# - Aluminum (76*)
# - Fertilizers (31*)
# - Hydrogen (2804 10 00)
```

2. Add warning for non-CBAM products:
```python
if not is_cbam_covered(cn_code):
    warnings.append({
        "code": "W005",
        "message": f"CN code {cn_code} may not be covered by CBAM",
        "suggestion": "Verify product scope with EU CBAM Annex I"
    })
```

#### B. CN Code Database Version Mismatch

**Cause:** Using outdated CN code database

**Solution:**
1. Check CN code database version:
```bash
head -1 data/cn_codes.json | jq '.metadata.version'
```

2. Update from EU source:
```bash
python scripts/update_cn_codes.py \
  --source https://ec.europa.eu/taxation_customs/dds2/taric/ \
  --filter-cbam-only \
  --output data/cn_codes_$(date +%Y%m%d).json
```

3. Deploy updated database:
```bash
kubectl create configmap cbam-cn-codes \
  --from-file=data/cn_codes_$(date +%Y%m%d).json \
  -n greenlang --dry-run=client -o yaml | kubectl apply -f -
```

---

## 4. SUPPLIER DATA LINKING ISSUES

### Symptoms
- Shipments not linking to supplier profiles
- All shipments using default emission factors
- Supplier actual emissions not being applied
- `supplier_found: false` in output

### Diagnostic Steps

```bash
# 1. Check supplier database
cat examples/demo_suppliers.yaml | grep "supplier_id:"

# 2. Test supplier lookup
python -c "
import yaml
with open('examples/demo_suppliers.yaml') as f:
    data = yaml.safe_load(f)
suppliers = {s['supplier_id']: s for s in data['suppliers']}
supplier_id = 'BAOSTEEL-CN-001'
if supplier_id in suppliers:
    print(suppliers[supplier_id])
else:
    print(f'Supplier {supplier_id} not found')
"

# 3. Check shipment supplier_id field
jq '.shipments[0].supplier_id' validated_shipments.json
```

### Common Root Causes

#### A. Missing supplier_id in Shipment Data

**Cause:** Shipment CSV doesn't include supplier_id column

**Solution:**
1. Add supplier_id to input data:
```csv
shipment_id,cn_code,origin_country,net_mass_kg,import_date,supplier_id
SHP-001,72031000,CN,25000,2025-10-01,BAOSTEEL-CN-001
```

2. Make supplier_id optional but warn if missing:
```python
if 'supplier_id' not in shipment or not shipment['supplier_id']:
    warnings.append({
        "code": "W010",
        "message": "Supplier ID not provided, using default emission factors",
        "impact": "May result in higher reported emissions"
    })
```

#### B. Supplier ID Mismatch

**Cause:** supplier_id in shipment doesn't match supplier profiles

**Solution:**
1. Implement fuzzy matching:
```python
from difflib import get_close_matches
def find_supplier(supplier_id: str, suppliers: dict) -> Optional[dict]:
    if supplier_id in suppliers:
        return suppliers[supplier_id]
    # Try fuzzy match
    matches = get_close_matches(supplier_id, suppliers.keys(), n=1, cutoff=0.8)
    if matches:
        logger.warning(f"Supplier ID '{supplier_id}' fuzzy matched to '{matches[0]}'")
        return suppliers[matches[0]]
    return None
```

2. Generate supplier mapping report:
```bash
python scripts/analyze_supplier_mapping.py \
  --shipments data/shipments.csv \
  --suppliers examples/demo_suppliers.yaml \
  --output /reports/supplier_mapping.json
```

---

## 5. PERFORMANCE DEGRADATION

### Symptoms
- Processing time significantly increased (>2x normal)
- API latency high (P95 >5s)
- Database query slow log warnings
- High CPU/memory usage

### Diagnostic Steps

```bash
# 1. Check current performance metrics
curl http://cbam-importer:8001/metrics | grep -E "cbam_pipeline_duration|cbam_agent_duration"

# 2. Check database performance
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# 3. Check resource usage
kubectl top pods -n greenlang -l app=cbam-importer --containers

# 4. Check active connections
curl http://cbam-importer:8001/metrics | grep db_pool

# 5. Profile specific pipeline run
python cbam_pipeline.py \
  --input examples/demo_shipments.csv \
  --output /tmp/test.json \
  --profile \
  --importer-name "Test" --importer-country NL --importer-eori NL000 \
  --declarant-name "Test" --declarant-position "Test"
```

### Common Root Causes

#### A. Database Connection Pool Exhaustion

**Symptoms:** `TimeoutError: QueuePool limit exceeded`

**Solution:**
1. Increase connection pool size:
```bash
kubectl set env deployment/cbam-importer \
  DB_POOL_SIZE=20 \
  DB_MAX_OVERFLOW=40 \
  -n greenlang
```

2. Check for connection leaks:
```python
# Ensure connections are returned to pool
from contextlib import contextmanager
@contextmanager
def get_db_connection():
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        conn.close()  # Return to pool
```

#### B. Large Shipment Volume

**Symptoms:** Memory spikes with >10,000 shipments

**Solution:**
1. Enable batch processing:
```bash
python cbam_pipeline.py \
  --input large_shipments.csv \
  --batch-size 5000 \
  --output-dir /batched/ \
  --merge-final
```

2. Optimize DataFrame operations:
```python
# Don't load entire CSV into memory
for chunk in pd.read_csv('shipments.csv', chunksize=1000):
    process_chunk(chunk)
```

#### C. Slow Emission Factor Lookups

**Symptoms:** Agent 2 (EmissionsCalculatorAgent) taking >50% of total time

**Solution:**
1. Implement in-memory caching:
```python
from functools import lru_cache
@lru_cache(maxsize=128)
def get_emission_factor(cn_code: str) -> float:
    return EMISSION_FACTORS_DB.get(cn_code)
```

2. Pre-load emission factors:
```python
# Load once at startup
EMISSION_FACTORS_CACHE = {k: v for k, v in EMISSION_FACTORS_DB.items()}
```

---

## 6. PIPELINE TIMEOUT FAILURES

### Symptoms
- Pipeline runs timeout before completion
- "TimeoutError" in logs
- Partial results generated
- Kubernetes pod restarts mid-processing

### Diagnostic Steps

```bash
# 1. Check timeout configuration
kubectl get deployment cbam-importer -n greenlang -o yaml | grep -A 3 timeout

# 2. Check pod restart history
kubectl get pods -n greenlang -l app=cbam-importer -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}'

# 3. Check pipeline duration
curl http://cbam-importer:8001/metrics | grep cbam_pipeline_duration_seconds

# 4. Check liveness/readiness probes
kubectl describe pod -n greenlang <pod-name> | grep -A 10 "Liveness\|Readiness"
```

### Solutions

#### A. Increase Timeout Values

```yaml
# deployment.yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 6  # 3 minutes before kill

readinessProbe:
  httpGet:
    path: /readiness
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

#### B. Optimize Long-Running Operations

```python
# Add progress reporting for long runs
def process_large_batch(shipments):
    total = len(shipments)
    for i, shipment in enumerate(shipments):
        process_shipment(shipment)
        if i % 1000 == 0:
            logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
            # Update health endpoint to prevent probe failure
            update_health_status(f"Processing {i}/{total}")
```

---

## 7. DATABASE CONNECTION ERRORS

### Symptoms
- `psycopg2.OperationalError: could not connect to server`
- Intermittent connection failures
- "Too many connections" errors
- "Connection reset by peer"

### Diagnostic Steps

```bash
# 1. Check database pod status
kubectl get pods -n greenlang -l app=postgres

# 2. Check database logs
kubectl logs -n greenlang deployment/postgres --tail=100

# 3. Check connection count
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SELECT count(*) FROM pg_stat_activity;"

# 4. Check max connections
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SHOW max_connections;"

# 5. Test connection from app pod
kubectl exec -n greenlang deployment/cbam-importer -- \
  psql $DATABASE_URL -c "SELECT 1;"
```

### Solutions

#### A. Database Not Ready

```bash
# Wait for database to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n greenlang --timeout=300s
```

#### B. Connection String Issues

```bash
# Verify connection string format
echo $DATABASE_URL
# Should be: postgresql://user:password@postgres:5432/greenlang

# Test connection string
python -c "
import os
import psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
print('Connection successful')
conn.close()
"
```

#### C. Network Policy Blocking

```bash
# Check network policies
kubectl get networkpolicies -n greenlang

# Test network connectivity
kubectl exec -n greenlang deployment/cbam-importer -- \
  nc -zv postgres 5432
```

---

## 8. REPORT GENERATION FAILURES

### Symptoms
- JSON report not generated
- Markdown summary missing or incomplete
- Validation errors in final report
- Report structure incorrect

### Diagnostic Steps

```bash
# 1. Check ReportingPackagerAgent logs
kubectl logs -n greenlang deployment/cbam-importer | grep ReportingPackagerAgent

# 2. Validate report schema
python -c "
import json
import jsonschema
with open('schemas/registry_output.schema.json') as f:
    schema = json.load(f)
with open('/tmp/cbam_report.json') as f:
    report = json.load(f)
try:
    jsonschema.validate(report, schema)
    print('Report is valid')
except jsonschema.ValidationError as e:
    print(f'Validation error: {e.message}')
"

# 3. Check intermediate outputs
ls -la /intermediate/
```

### Solutions

#### A. Missing Importer Information

**Error:** `KeyError: 'importer_name'`

**Solution:**
```bash
# Ensure all importer fields provided
python cbam_pipeline.py \
  --input shipments.csv \
  --importer-name "Full Legal Name" \
  --importer-country NL \
  --importer-eori NL123456789012 \
  --declarant-name "John Doe" \
  --declarant-position "Compliance Officer" \
  --output report.json
```

#### B. Aggregation Failures

**Error:** Emissions totals don't match detail records

**Solution:**
```python
# Add validation step
def validate_report_totals(report):
    detail_sum = sum(r['embedded_emissions_tco2'] for r in report['detailed_goods'])
    summary_total = report['emissions_summary']['total_embedded_emissions_tco2']
    assert abs(detail_sum - summary_total) < 0.01, \
        f"Total mismatch: detail={detail_sum}, summary={summary_total}"
```

---

## 9. HIGH MEMORY USAGE / OOM KILLS

### Symptoms
- Pods being killed by Kubernetes (OOMKilled)
- Memory usage steadily increasing
- Slow performance before crash
- `MemoryError` exceptions

### Diagnostic Steps

```bash
# 1. Check current memory usage
kubectl top pods -n greenlang -l app=cbam-importer

# 2. Check memory limits
kubectl get deployment cbam-importer -n greenlang -o yaml | grep -A 3 resources

# 3. Check OOM kills
kubectl describe pod -n greenlang <pod-name> | grep -i oom

# 4. Profile memory usage
python -m memory_profiler cbam_pipeline.py [args]
```

### Solutions

#### A. Increase Memory Limits

```yaml
# deployment.yaml
resources:
  requests:
    memory: "2Gi"
  limits:
    memory: "4Gi"
```

#### B. Optimize Memory Usage

```python
# Process in chunks instead of loading all data
def process_large_file(file_path):
    for chunk in pd.read_csv(file_path, chunksize=1000):
        yield process_chunk(chunk)
        # Chunk goes out of scope and is garbage collected

# Clear large objects after use
import gc
del large_dataframe
gc.collect()
```

---

## 10. CBAM COMPLIANCE VALIDATION ERRORS

### Symptoms
- `is_valid: false` in final report
- CBAM rule violations flagged
- Cannot submit to EU Registry
- Compliance warnings for valid data

### Diagnostic Steps

```bash
# 1. Check validation results
jq '.validation_results' cbam_report.json

# 2. List all errors
jq '.validation_results.errors' cbam_report.json

# 3. Check specific rule
cat rules/cbam_rules.yaml | grep -A 10 "rule_id: R025"
```

### Common Validation Errors

#### A. Complex Goods 20% Cap Violation

**Error:** `R025: Complex goods exceed 20% of total emissions`

**Solution:**
```python
# Calculate complex goods percentage
complex_emissions = sum(e for e in emissions if is_complex_good(e))
total_emissions = sum(emissions)
if complex_emissions / total_emissions > 0.20:
    # Need to provide more specific embedded emissions data
    # or justify to EU authorities
```

#### B. Missing Actual Emissions Data

**Error:** `R030: High-volume shipments should have actual emissions data`

**Solution:**
1. Contact suppliers for certified emissions data
2. Document why actual data unavailable
3. Use most accurate available defaults

---

## QUICK REFERENCE COMMANDS

```bash
# Health check
curl http://cbam-importer:8000/health

# View logs
kubectl logs -n greenlang deployment/cbam-importer --tail=100 -f

# Restart deployment
kubectl rollout restart deployment/cbam-importer -n greenlang

# Scale up for performance
kubectl scale deployment/cbam-importer --replicas=5 -n greenlang

# Check metrics
curl http://cbam-importer:8001/metrics | grep cbam_

# Run test pipeline
python cbam_pipeline.py \
  --input examples/demo_shipments.csv \
  --output /tmp/test.json \
  --importer-name "Test" --importer-country NL --importer-eori NL000 \
  --declarant-name "Test" --declarant-position "Tester"

# Validate report
python scripts/validate_cbam_report.py --report /tmp/test.json
```

---

## RELATED RUNBOOKS

- INCIDENT_RESPONSE.md - For production incidents
- ROLLBACK_PROCEDURE.md - For deployment rollbacks
- SCALING_GUIDE.md - For performance scaling
- MAINTENANCE.md - For routine maintenance

---

**Document Control:**
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Next Review:** 2025-12-18
- **Owner:** CBAM Engineering Team

---

*This troubleshooting guide will be updated as new issues are discovered and resolved.*
