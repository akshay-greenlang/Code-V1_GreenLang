# Excel Normalizer Service Down

## Alert

**Alert Name:** `ExcelNormalizerServiceDown`

**Severity:** Critical

**Threshold:** `up{job="excel-normalizer-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Excel & CSV Normalizer (AGENT-DATA-002) are running. The Excel Normalizer is the tabular data ingestion and normalization service for all GreenLang Climate OS spreadsheet-based data sources. It is responsible for:

1. **Spreadsheet upload and registration** -- Accepting Excel (XLSX, XLS), CSV, TSV, and ODS files with SHA-256 file hash deduplication and metadata capture
2. **Sheet enumeration and header detection** -- Automatically detecting sheet structure, header rows, column names, data types, encodings (UTF-8, ISO-8859-1, Windows-1252, etc.), and delimiters (comma, tab, semicolon, pipe)
3. **Column-to-canonical mapping** -- Mapping source column headers to canonical GreenLang fields using exact match, synonym matching, fuzzy matching (Levenshtein), ML classification, template-based, and regex strategies
4. **Data normalization** -- Converting raw spreadsheet data into canonical format with type casting, unit conversion, date parsing, string cleaning, and quality scoring
5. **Data quality assessment** -- Scoring data across completeness, accuracy, and consistency dimensions with per-column quality scores and issue identification
6. **Schema validation** -- Validating normalized data against configurable rules including type checks, range validation, required fields, format patterns, and cross-column consistency
7. **Transform operations** -- Applying data transformations (type cast, unit convert, date parse, string clean, merge columns, split column, fill nulls, deduplicate, filter, aggregate)
8. **Mapping template management** -- Maintaining reusable column mapping templates for consistent normalization of recurring spreadsheet formats
9. **Provenance hash chains** -- Maintaining SHA-256 hash chains across all normalized records and audit events for tamper detection and compliance
10. **Emitting Prometheus metrics** (12+ metrics under the `gl_excel_normalizer_*` prefix) for monitoring normalization rates, mapping confidence, quality scores, and service health

When the Excel Normalizer is down:
- **Tabular data ingestion stops** and no new spreadsheets can be uploaded or normalized
- **Emission calculation data feed is interrupted** from spreadsheet-based sources (utility bills, fuel logs, energy audits)
- **Normalization queue will grow** and files will accumulate without processing
- **Data quality assessment is unavailable** and normalized data quality cannot be verified
- **Audit trail has a gap** and compliance requirements for traceable data ingestion are violated

**Note:** All spreadsheet files, sheet metadata, column mappings, normalization jobs, normalized records, mapping templates, data quality reports, validation results, transform operations, and audit events are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full state will be immediately available. Pending normalization jobs will need to be reprocessed.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | No new spreadsheets can be uploaded or processed for normalization |
| **Data Impact** | High | Spreadsheet-based emission data feed interrupted; normalization queue growing |
| **SLA Impact** | High | Spreadsheet processing SLA violated (all normalization jobs fail) |
| **Revenue Impact** | Medium | Compliance-sensitive customers require timely spreadsheet processing |
| **Compliance Impact** | High | CSRD, CBAM require traceable, verifiable source data processing |
| **Downstream Impact** | High | Emission calculation agents waiting for normalized utility/fuel/energy data |

---

## Symptoms

- `up{job="excel-normalizer-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=excel-normalizer-service`
- `gl_excel_normalizer_jobs_total` counter stops incrementing
- `gl_excel_normalizer_files_processed_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Spreadsheet upload endpoints return errors
- Grafana Excel Normalizer dashboard shows "No Data" or stale timestamps
- Normalization job queue backlog grows without being processed

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List Excel normalizer service pods
kubectl get pods -n greenlang -l app=excel-normalizer-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=excel-normalizer-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to Excel normalizer service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=excel-normalizer-service | tail -30

# Check deployment status
kubectl describe deployment excel-normalizer-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment excel-normalizer-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=excel-normalizer-service

# Check for rollout issues
kubectl rollout status deployment/excel-normalizer-service -n greenlang

# Check HPA status (scales 2-8 replicas)
kubectl get hpa -n greenlang -l app=excel-normalizer-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=excel-normalizer-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=excel-normalizer-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for normalization-specific errors
kubectl logs -n greenlang -l app=excel-normalizer-service --tail=500 \
  | grep -i "encoding\|delimiter\|mapping\|normalize\|parse\|column\|template"

# Look for database connection errors
kubectl logs -n greenlang -l app=excel-normalizer-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of Excel normalizer service pods
kubectl top pods -n greenlang -l app=excel-normalizer-service

# Check if pods were OOMKilled (large spreadsheet processing is memory-intensive)
kubectl get pods -n greenlang -l app=excel-normalizer-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes

# Check upload staging volume usage
kubectl exec -n greenlang <pod-name> -- df -h /app/uploads /tmp
```

### Step 5: Check Database Connectivity

```bash
# Verify PostgreSQL connectivity
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check if the excel_normalizer_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='excel_normalizer_service'
   ORDER BY table_name;"
```

### Step 6: Check ConfigMap and Secrets

```bash
# Verify the Excel normalizer service ConfigMap exists and is valid
kubectl get configmap excel-normalizer-service-config -n greenlang
kubectl get configmap excel-normalizer-service-config -n greenlang -o yaml | head -50

# Verify secrets exist
kubectl get secret excel-normalizer-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment excel-normalizer-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 7: Check Network Policies

```bash
# Check network policies affecting the Excel normalizer service
kubectl get networkpolicy -n greenlang | grep excel-normalizer

# Verify the Excel normalizer service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify egress to S3 (HTTPS) for spreadsheet storage
kubectl run net-test-s3 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv s3.amazonaws.com 443'
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137. Common with large spreadsheets or high row counts.

**Resolution:**

1. Confirm the OOM cause:
```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. Increase memory limits (large spreadsheet processing requires more memory):
```bash
kubectl patch deployment excel-normalizer-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "excel-normalizer-service",
            "resources": {
              "limits": {
                "cpu": "1",
                "memory": "1Gi"
              },
              "requests": {
                "cpu": "500m",
                "memory": "512Mi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

3. Verify pods restart successfully:
```bash
kubectl rollout status deployment/excel-normalizer-service -n greenlang
kubectl get pods -n greenlang -l app=excel-normalizer-service
```

### Scenario 2: CrashLoopBackOff -- Database Migration Failure

**Symptoms:** Pod status shows CrashLoopBackOff, init container logs show migration errors.

**Resolution:**

1. Check init container logs:
```bash
kubectl logs -n greenlang <pod-name> -c check-db-migration --tail=100
```

2. Verify database schema:
```bash
kubectl run pg-migration --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description, success FROM flyway_schema_history
   ORDER BY installed_rank DESC LIMIT 5;"
```

3. Restart the deployment after fixing:
```bash
kubectl rollout restart deployment/excel-normalizer-service -n greenlang
kubectl rollout status deployment/excel-normalizer-service -n greenlang
```

### Scenario 3: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/excel-normalizer-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/excel-normalizer-service -n greenlang
kubectl rollout status deployment/excel-normalizer-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=excel-normalizer-service
kubectl port-forward -n greenlang svc/excel-normalizer-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=excel-normalizer-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/excel-normalizer-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the Excel normalizer service is being scraped
up{job="excel-normalizer-service"} == 1

# Verify normalization job count metric is populated
gl_excel_normalizer_jobs_total > 0

# Verify file processing metrics are incrementing
increase(gl_excel_normalizer_files_processed_total[5m])
```

### Step 3: Reprocess Pending Jobs

```bash
# Check for stuck or failed jobs during the outage
curl -s "http://localhost:8080/v1/jobs?status=pending&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Reprocess failed jobs
curl -s -X POST "http://localhost:8080/v1/jobs/reprocess" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "failed", "since": "2024-01-01T00:00:00Z"}' \
  | python3 -m json.tool
```

---

## Interim Mitigation

While the Excel Normalizer is being restored:

1. **Normalization data is safe.** All spreadsheet files, sheet metadata, column mappings, normalization jobs, normalized records, templates, quality reports, validation results, transform operations, and audit events are stored in PostgreSQL with TimescaleDB. The database persists independently.

2. **Spreadsheets will queue up.** Uploaded files will be stored but not processed. Once the service recovers, the normalization queue will need to be drained.

3. **Emission calculations from spreadsheet sources are delayed.** Utility bill, fuel log, and energy audit data will not be available for emission calculations until spreadsheets are processed.

4. **Manual data entry may be needed.** For time-critical emission reports, manual data entry can bypass the spreadsheet normalization pipeline temporarily.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-data` -- engineering response
   - `#platform-oncall` -- on-call engineer
   - `#data-pipeline-ops` -- data pipeline impact notification

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Excel normalizer down, normalization queue growing | On-call engineer | Immediate (<5 min) |
| L2 | Excel normalizer down > 15 minutes, emission data feed interrupted | Platform team lead + #platform-data | 15 minutes |
| L3 | Excel normalizer down > 30 minutes, compliance reporting blocked | Platform team + compliance team + CTO notification | Immediate |
| L4 | Excel normalizer down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Excel Normalizer Health (`/d/excel-normalizer-service`)
- **Alert:** `ExcelNormalizerServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="excel-normalizer-service"}` (should always be >= 2)
  - `gl_excel_normalizer_jobs_total` rate (should be non-zero during business hours)
  - `gl_excel_normalizer_files_processed_total` rate (should be non-zero)
  - `gl_excel_normalizer_mapping_confidence` avg (should stay above 0.7)
  - `gl_excel_normalizer_mapping_errors_total` (should be 0 or near-zero)
  - `gl_excel_normalizer_job_duration_seconds` p99 (should stay below 30s)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 8 replicas** based on CPU and memory utilization
4. **Upload staging volume** sized at 2Gi for spreadsheet buffering
5. **Database connection pool** is sized for expected concurrency (default: min 2, max 10)

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `ExcelNormalizerServiceDown` | Critical | This alert -- no Excel normalizer pods running |
| `ExcelHighNormalizationFailureRateCritical` | Critical | >25% of normalization jobs are failing |
| `ExcelColumnMappingFailure` | Critical | Column mapping errors exceeding threshold |
| `ExcelAuditGap` | Critical | Normalization operations without audit entries |
| `ExcelDatabaseConnectionFailure` | Critical | Database connection errors |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Data Team
- **Review cadence:** Quarterly or after any P1 Excel normalizer incident
- **Related runbooks:** [Excel Normalizer Parsing Failures](./excel-normalizer-parsing-failures.md)
