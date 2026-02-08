# ERP Connector Sync Failures

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `ERPHighSyncFailureRateWarning` | Warning | >10% sync job failure rate for 10 minutes |
| `ERPHighSyncFailureRateCritical` | Critical | >25% sync job failure rate for 5 minutes |
| `ERPConnectionFailure` | Critical | No active ERP connections for 5 minutes |
| `ERPCurrencyConversionFailure` | Warning | >10 currency conversion failures in 5 minutes |
| `ERPEmissionCalculationErrors` | Warning | >15% emission calculation error rate for 15 minutes |
| `ERPVendorMappingGaps` | Warning | >30% of spend records have unmapped vendors for 30 minutes |
| `ERPHighSyncLatencyWarning` | Warning | p99 sync latency above 300s for 10 minutes |
| `ERPHighSyncLatencyCritical` | Critical | p99 sync latency above 900s for 5 minutes |
| `ERPSyncTimeout` | Warning | Sync timeout rate above 0.05/sec for 15 minutes |
| `ERPSyncQueueBacklog` | Warning | >20 pending sync jobs for 15 minutes |
| `ERPConnectionError` | Warning | >5 ERP connection errors in 5 minutes |

**Thresholds:**

```promql
# ERPHighSyncFailureRateWarning
# More than 10% sync failure rate for 10 minutes
(sum(rate(gl_erp_connector_sync_jobs_total{status="failed"}[5m]))
 / sum(rate(gl_erp_connector_sync_jobs_total[5m]))) > 0.10

# ERPHighSyncFailureRateCritical
# More than 25% sync failure rate for 5 minutes
(sum(rate(gl_erp_connector_sync_jobs_total{status="failed"}[5m]))
 / sum(rate(gl_erp_connector_sync_jobs_total[5m]))) > 0.25

# ERPConnectionFailure
# No active ERP connections for 5 minutes
gl_erp_connector_active_connections == 0

# ERPCurrencyConversionFailure
# More than 10 currency conversion failures in 5 minutes
increase(gl_erp_connector_currency_conversions_total{status="failed"}[5m]) > 10

# ERPEmissionCalculationErrors
# More than 15% emission calculation error rate for 15 minutes
(sum(rate(gl_erp_connector_emission_calculation_errors_total[5m]))
 / sum(rate(gl_erp_connector_emission_calculations_total[5m]))) > 0.15

# ERPVendorMappingGaps
# More than 30% of spend records unmapped for 30 minutes
(sum(rate(gl_erp_connector_unmapped_vendors_total[5m]))
 / sum(rate(gl_erp_connector_spend_records_total[5m]))) > 0.30

# ERPHighSyncLatencyWarning
# p99 sync latency above 300s for 10 minutes
histogram_quantile(0.99,
  sum(rate(gl_erp_connector_sync_duration_seconds_bucket[5m])) by (le)) > 300

# ERPSyncTimeout
# Sync timeout rate above 0.05/sec for 15 minutes
sum(rate(gl_erp_connector_sync_jobs_total{status="timeout"}[5m])) > 0.05

# ERPSyncQueueBacklog
# More than 20 pending sync jobs for 15 minutes
gl_erp_connector_pending_sync_jobs > 20
```

---

## Description

These alerts fire when the ERP/Finance Connector (AGENT-DATA-003) encounters sync failures, connection errors, emission calculation issues, or vendor mapping gaps. Sync failures directly impact the availability and accuracy of Scope 3 procurement emission data extracted from enterprise ERP systems.

### How ERP Synchronization Works

The ERP Connector uses a multi-system sync pipeline:

1. **Connection Management** -- Connections to ERP systems (SAP, Oracle, Dynamics, NetSuite, etc.) are maintained with health monitoring. Each connection stores credentials, host, port, company code, and sync history.

2. **Sync Job Scheduling** -- Sync jobs run on a configurable schedule (or on-demand) with support for full, incremental, delta, manual, and scheduled modes. Each job targets a specific query type (spend, PO, inventory, vendors, materials).

3. **Data Extraction** -- The connector queries the ERP system API (OData, BAPI, REST, SOAP) to extract records in configurable batch sizes (default 1000). Records are validated and deduplicated during extraction.

4. **Currency Conversion** -- Non-USD amounts are converted to USD using exchange rates from configurable providers (ECB, OpenExchangeRates, Fixer, manual). Rates are cached for performance.

5. **Vendor/Material Classification** -- Vendors and materials are mapped to GreenLang spend categories and assigned emission factors. Auto-classification uses EEIO sector codes. Manual overrides are supported.

6. **Scope 3 Emission Calculation** -- Emissions are calculated using spend-based (amount * EF/USD), activity-based (quantity * EF/unit), or hybrid methodology per GHG Protocol.

7. **Provenance Tracking** -- Every synced record and calculation receives a SHA-256 provenance hash for tamper detection and audit compliance.

### Common Failure Modes

| Failure Mode | Typical Cause | Impact |
|--------------|---------------|--------|
| ERP connection lost | Network failure, firewall change, credential expiration | All sync jobs for that connection fail |
| API rate limiting | Too many concurrent requests to ERP system | Sync jobs throttled or fail with 429 |
| Authentication failure | Expired token, changed password, revoked access | Connection enters error state |
| Data format change | ERP system upgrade, schema migration, API version change | Records parsed incorrectly or rejected |
| Currency conversion failure | Exchange rate API down, invalid currency code | Amounts cannot be converted to USD |
| Vendor mapping gap | New vendor without classification | Emission factor defaults to 0 or EEIO average |
| Sync timeout | Very large data volume, slow ERP API response | Job fails; records not synced |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Scope 3 emission data may be stale or incomplete |
| **Data Impact** | High | Procurement-based emission calculations use synced spend data; sync failures = stale calculations |
| **SLA Impact** | Medium | ERP sync processing SLA degraded but not fully blocked |
| **Revenue Impact** | Medium | Enterprise customers require timely procurement data for reporting deadlines |
| **Compliance Impact** | High | CSRD/GHG Protocol require verifiable Scope 3 supply chain data |
| **Downstream Impact** | High | Emission reporting agents receive stale or incomplete procurement data |

---

## Symptoms

- `gl_erp_connector_sync_jobs_total{status="failed"}` rate is elevated
- `gl_erp_connector_sync_jobs_total{status="timeout"}` rate is elevated
- `gl_erp_connector_active_connections` is lower than expected or 0
- `gl_erp_connector_connection_errors_total` counter is incrementing rapidly
- `gl_erp_connector_currency_conversions_total{status="failed"}` is elevated
- `gl_erp_connector_unmapped_vendors_total` rate is high relative to spend records
- `gl_erp_connector_emission_calculation_errors_total` rate is elevated
- Grafana ERP Connector dashboard shows sync failures or connection errors
- Users report stale Scope 3 emission data or missing procurement records
- Sync jobs fail with ERP API error messages in logs

---

## Diagnostic Steps

### Step 1: Identify the Scope of Sync Failures

```bash
# Check sync job metrics by status and query type
kubectl port-forward -n greenlang svc/erp-connector-service 8080:8080
curl -s http://localhost:8080/metrics | grep gl_erp_connector_sync

# Check which ERP connections are in error state
curl -s "http://localhost:8080/v1/connections?status=error&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check recent failed sync jobs
curl -s "http://localhost:8080/v1/sync-jobs?status=failed&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 2: Identify the Failure Pattern

```bash
# Check if failures are concentrated on a specific ERP system
curl -s http://localhost:8080/metrics | grep gl_erp_connector_sync_jobs_total

# Check if failures are concentrated on a specific query type
curl -s "http://localhost:8080/v1/sync-jobs?status=failed&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('items', data if isinstance(data, list) else [])
for job in items:
    print(f\"Job: {job.get('job_id', '?')[:8]}  Type: {job.get('query_type', '?')}  Mode: {job.get('sync_mode', '?')}  Error: {str(job.get('errors', '?'))[:120]}\")
"

# Check connection health for each ERP system
curl -s "http://localhost:8080/v1/connections" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('items', data if isinstance(data, list) else [])
for conn in items:
    print(f\"Connection: {conn.get('connection_id', '?')[:8]}  System: {conn.get('erp_system', '?')}  Status: {conn.get('status', '?')}  Last Sync: {conn.get('last_sync', 'never')}  Errors: {conn.get('error_count', 0)}\")
"
```

### Step 3: Check ERP System Connectivity

```bash
# Check logs for ERP connectivity errors
kubectl logs -n greenlang -l app=erp-connector-service --tail=500 \
  | grep -i "connection\|auth\|sap\|oracle\|dynamics\|netsuite\|workday\|timeout"

# Check for authentication errors
kubectl logs -n greenlang -l app=erp-connector-service --tail=500 \
  | grep -i "auth\|credential\|unauthorized\|403\|forbidden\|expired\|token"

# Check for rate limiting
kubectl logs -n greenlang -l app=erp-connector-service --tail=500 \
  | grep -i "rate limit\|throttl\|429\|too many\|quota"

# Check for network errors
kubectl logs -n greenlang -l app=erp-connector-service --tail=500 \
  | grep -i "timeout\|timed out\|refused\|unreachable\|dns\|ssl\|tls"
```

### Step 4: Check Currency Conversion

```bash
# Check currency conversion metrics
curl -s http://localhost:8080/metrics | grep gl_erp_connector_currency

# Check exchange rate provider connectivity
kubectl logs -n greenlang -l app=erp-connector-service --tail=200 \
  | grep -i "exchange\|currency\|ecb\|fixer\|rate"

# Verify exchange rate API key validity
kubectl get secret erp-connector-service-secrets -n greenlang -o yaml | head -20
```

### Step 5: Check Vendor/Material Mapping Gaps

```bash
# Check unmapped vendor metrics
curl -s http://localhost:8080/metrics | grep gl_erp_connector_unmapped

# List recently synced vendors without mappings
curl -s "http://localhost:8080/v1/vendors/unmapped?limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check auto-classification status
curl -s "http://localhost:8080/v1/config" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Auto-classify vendors: {data.get('auto_classify_vendors', 'unknown')}\")
print(f\"Auto-classify materials: {data.get('auto_classify_materials', 'unknown')}\")
print(f\"Default EF database: {data.get('default_ef_database', 'unknown')}\")
"
```

### Step 6: Check Emission Calculation Errors

```bash
# Check emission calculation metrics
curl -s http://localhost:8080/metrics | grep gl_erp_connector_emission

# Check recent calculation errors
curl -s "http://localhost:8080/v1/emission-calculations?status=error&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Look for missing emission factors
kubectl logs -n greenlang -l app=erp-connector-service --tail=500 \
  | grep -i "emission factor\|missing factor\|zero factor\|eeio"
```

---

## Resolution Steps

### Scenario 1: ERP System Connection Lost

**Symptoms:** Connection errors for a specific ERP system. Sync jobs fail with connectivity errors.

**Resolution:**

1. Verify ERP system is reachable:
```bash
kubectl run erp-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv <erp-host> 443'
```

2. Check and rotate credentials if expired:
```bash
# Check secret values (redacted)
kubectl get secret erp-connector-service-secrets -n greenlang -o yaml | head -20

# Update credentials
kubectl create secret generic erp-connector-service-secrets \
  --from-literal=SAP_CLIENT_SECRET=<new-secret> \
  --dry-run=client -o yaml | kubectl apply -f -
```

3. Restart the connector to pick up new credentials:
```bash
kubectl rollout restart deployment/erp-connector-service -n greenlang
kubectl rollout status deployment/erp-connector-service -n greenlang
```

4. Reprocess failed sync jobs:
```bash
curl -s -X POST "http://localhost:8080/v1/sync-jobs/reprocess" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "failed", "connection_id": "<connection_id>"}' \
  | python3 -m json.tool
```

### Scenario 2: Currency Conversion Failures

**Symptoms:** Spend amounts cannot be converted to USD. Exchange rate API errors in logs.

**Resolution:**

1. Switch to fallback exchange rate provider:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_EXCHANGE_RATE_PROVIDER=openexchangerates
```

2. If all providers are down, enable manual rates:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_EXCHANGE_RATE_PROVIDER=manual
```

3. Once provider is restored, switch back:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_EXCHANGE_RATE_PROVIDER=ecb
```

### Scenario 3: Vendor Mapping Gaps (Missing Emission Factors)

**Symptoms:** High percentage of spend records without vendor emission factor mappings.

**Resolution:**

1. Check auto-classification status and enable if disabled:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_AUTO_CLASSIFY_VENDORS=true
```

2. Bulk-import vendor mappings:
```bash
curl -s -X POST "http://localhost:8080/v1/vendors/bulk-map" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"emission_factor_database": "EEIO", "overwrite": false}' \
  | python3 -m json.tool
```

3. Manually map high-spend unmapped vendors:
```bash
curl -s -X POST "http://localhost:8080/v1/vendors/map" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "vendor_id": "<vendor_id>",
    "spend_category": "office_supplies",
    "emission_factor_kgco2e_per_dollar": 0.42,
    "emission_factor_source": "EEIO"
  }' | python3 -m json.tool
```

### Scenario 4: Sync Timeouts (Large Data Volumes)

**Symptoms:** Timeout rate increasing, primarily affecting full sync or large date ranges.

**Resolution:**

1. Reduce batch size:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_SYNC_BATCH_SIZE=500
```

2. Increase timeout for large syncs:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_SYNC_TIMEOUT_SECONDS=7200
```

3. Switch to incremental sync mode:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_DEFAULT_SYNC_MODE=incremental
```

### Scenario 5: ERP API Rate Limiting

**Symptoms:** Intermittent failures, 429 status codes in logs.

**Resolution:**

1. Reduce concurrent sync jobs:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_MAX_CONCURRENT_SYNCS=1
```

2. Reduce batch size to lower API call rate:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_SYNC_BATCH_SIZE=250
```

3. Increase retry delay:
```bash
kubectl set env deployment/erp-connector-service -n greenlang \
  GL_ERP_CONNECTOR_SYNC_RETRY_DELAY_SECONDS=60
```

4. Contact ERP system administrator to increase rate limits if needed.

---

## Post-Incident Steps

### Step 1: Verify Sync Is Restored

```bash
# Check that sync success rate is back to normal
curl -s http://localhost:8080/metrics | grep gl_erp_connector_sync_jobs_total

# Check that connection health is restored
curl -s http://localhost:8080/metrics | grep gl_erp_connector_active_connections

# Check that error rate is back to normal
curl -s http://localhost:8080/metrics | grep gl_erp_connector_connection_errors
```

### Step 2: Reprocess Failed Sync Jobs

```bash
# Reprocess all sync jobs that failed during the incident
curl -s -X POST "http://localhost:8080/v1/sync-jobs/reprocess" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "failed", "since": "<incident_start_time>"}' \
  | python3 -m json.tool
```

### Step 3: Verify Emission Calculations Are Current

```bash
# Check that emission calculations are running
curl -s http://localhost:8080/metrics | grep gl_erp_connector_emission_calculations

# Trigger recalculation for the incident period
curl -s -X POST "http://localhost:8080/v1/emission-calculations/recalculate" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"since": "<incident_start_time>"}' \
  | python3 -m json.tool
```

### Step 4: Verify Audit Trail Integrity

```promql
# Verify audit events are being recorded
increase(gl_erp_connector_audit_events_total[5m])

# Check for audit gaps
(
  sum(rate(gl_erp_connector_sync_jobs_total[5m]))
  - sum(rate(gl_erp_connector_audit_events_total[5m]))
)
```

---

## Interim Mitigation

While sync issues are being resolved:

1. **Switch to incremental sync.** If full syncs are failing, switch to incremental mode to sync only changed records since the last successful sync.

2. **Reduce concurrency and batch size.** Lower the load on ERP systems by reducing concurrent syncs and batch sizes.

3. **Use manual vendor mappings.** For high-spend vendors without auto-classification, manually assign emission factors using the vendor mapping API.

4. **Enable manual data entry bypass.** For time-critical emission reports, allow manual entry of procurement spend data.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-data` -- engineering response
   - `#data-pipeline-ops` -- data pipeline impact
   - `#compliance-ops` -- compliance impact if Scope 3 data is stale

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Sync failure rate elevated, investigation in progress | On-call engineer | 15 minutes |
| L2 | ERP connection lost, sync jobs failing at >10% | Platform team lead + #platform-data | Immediate (<5 min) |
| L3 | All ERP connections down, procurement data ingestion blocked | Platform team + data team + CTO notification | Immediate |
| L4 | Systemic failure affecting Scope 3 emission calculations downstream | All-hands engineering + incident commander + executive notification | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** ERP Connector Health (`/d/erp-connector-service`)
- **Alerts:** `ERPHighSyncFailureRateCritical`, `ERPConnectionFailure`, `ERPCurrencyConversionFailure`, `ERPEmissionCalculationErrors`
- **Key metrics to watch:**
  - `gl_erp_connector_active_connections` (should match expected count)
  - `gl_erp_connector_connection_errors_total` (should be 0 or near-zero)
  - `gl_erp_connector_sync_jobs_total{status="failed"}` rate (should be < 10%)
  - `gl_erp_connector_sync_jobs_total{status="timeout"}` rate (should be near 0)
  - `gl_erp_connector_currency_conversions_total{status="failed"}` (should be near 0)
  - `gl_erp_connector_unmapped_vendors_total` rate (should be < 30% of spend records)
  - `gl_erp_connector_emission_calculation_errors_total` rate (should be < 15%)
  - `gl_erp_connector_sync_duration_seconds` p99 (should be < 300s)

### Best Practices

1. **Rotate ERP credentials proactively** before expiration
2. **Use incremental sync mode** for routine operations, full sync only for initial load
3. **Maintain vendor/material mapping coverage** above 70% to ensure emission factor accuracy
4. **Set appropriate batch sizes** per ERP system (SAP may handle 5000, QuickBooks may limit to 100)
5. **Monitor ERP system API quotas** and adjust sync frequency accordingly
6. **Test sync configurations** in staging before deploying to production
7. **Review Scope 3 category mappings weekly** for accuracy
8. **Set up exchange rate API key expiration alerts** to prevent currency conversion outages

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `ERPHighSyncFailureRateCritical` | Critical | >25% sync failure rate |
| `ERPConnectionFailure` | Critical | No active ERP connections |
| `ERPHighSyncLatencyCritical` | Critical | p99 sync latency above 900s |
| `ERPHighSyncFailureRateWarning` | Warning | >10% sync failure rate |
| `ERPCurrencyConversionFailure` | Warning | Currency conversion errors |
| `ERPEmissionCalculationErrors` | Warning | >15% emission calculation errors |
| `ERPVendorMappingGaps` | Warning | >30% unmapped vendors |
| `ERPSyncTimeout` | Warning | Sync timeout rate elevated |
| `ERPSyncQueueBacklog` | Warning | Pending sync queue growing |
| `ERPConnectionError` | Warning | ERP connection errors detected |
| `ERPConnectorServiceDown` | Critical | No healthy pods running |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Data Team
- **Review cadence:** Quarterly or after any P1 ERP sync incident
- **Related runbooks:** [ERP Connector Service Down](./erp-connector-service-down.md)
