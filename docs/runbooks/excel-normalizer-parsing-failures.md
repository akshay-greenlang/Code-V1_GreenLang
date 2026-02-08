# Excel Normalizer Parsing Failures

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `ExcelLowMappingConfidence` | Warning | Average column mapping confidence below 60% for 15 minutes |
| `ExcelColumnMappingFailure` | Critical | >10 column mapping errors in 5 minutes |
| `ExcelValidationErrorSpike` | Warning | >20% of field validations producing errors for 15 minutes |
| `ExcelHighNormalizationFailureRateWarning` | Warning | >10% normalization job failure rate for 10 minutes |
| `ExcelHighNormalizationFailureRateCritical` | Critical | >25% normalization job failure rate for 5 minutes |
| `ExcelNormalizationTimeout` | Warning | Normalization timeouts exceeding 0.1/sec for 15 minutes |

**Thresholds:**

```promql
# ExcelLowMappingConfidence
# Average column mapping confidence below 0.60 for 15 minutes
avg(gl_excel_normalizer_mapping_confidence) < 0.6

# ExcelColumnMappingFailure
# More than 10 column mapping errors in 5 minutes
increase(gl_excel_normalizer_mapping_errors_total[5m]) > 10

# ExcelValidationErrorSpike
# More than 20% validation errors for 15 minutes
(sum(rate(gl_excel_normalizer_validation_results_total{severity="error"}[5m]))
 / sum(rate(gl_excel_normalizer_validation_results_total[5m]))) > 0.20

# ExcelHighNormalizationFailureRateWarning
# More than 10% normalization failure rate for 10 minutes
(sum(rate(gl_excel_normalizer_jobs_total{status="failed"}[5m]))
 / sum(rate(gl_excel_normalizer_jobs_total[5m]))) > 0.10

# ExcelHighNormalizationFailureRateCritical
# More than 25% normalization failure rate for 5 minutes
(sum(rate(gl_excel_normalizer_jobs_total{status="failed"}[5m]))
 / sum(rate(gl_excel_normalizer_jobs_total[5m]))) > 0.25

# ExcelNormalizationTimeout
# Normalization timeout rate above 0.1/sec for 15 minutes
sum(rate(gl_excel_normalizer_jobs_total{status="timeout"}[5m])) > 0.1
```

---

## Description

These alerts fire when the Excel & CSV Normalizer (AGENT-DATA-002) encounters parsing failures, encoding issues, column mapping errors, or elevated validation error rates. Parsing failures directly impact the quality and availability of tabular data (utility bills, fuel logs, energy audits, emissions reports) that feed into GreenLang's emission calculations.

### How Spreadsheet Normalization Works

The Excel Normalizer uses a multi-stage normalization pipeline:

1. **File Upload** -- Spreadsheets (XLSX, XLS, CSV, TSV, ODS) are uploaded and stored with SHA-256 file hash for deduplication.

2. **Format Detection** -- The file format is detected and the appropriate parser is selected. For CSV/TSV files, the encoding and delimiter are automatically detected.

3. **Sheet Enumeration** -- Multi-sheet workbooks are enumerated. Each sheet is processed independently with header detection and structure analysis.

4. **Header Detection** -- Header rows are automatically identified using heuristics (row position, data type uniformity, non-null ratio). The header row index is recorded.

5. **Encoding Detection** -- For CSV/TSV files, the character encoding is detected using statistical analysis. Supported encodings:
   - `utf-8` -- Default, most common
   - `utf-16` -- Unicode with BOM
   - `ascii` -- 7-bit ASCII
   - `iso-8859-1` / `latin-1` -- Western European
   - `windows-1252` -- Windows Western European
   - `shift_jis` -- Japanese
   - `gb2312` / `gbk` -- Chinese Simplified
   - `big5` -- Chinese Traditional
   - `euc-kr` -- Korean

6. **Delimiter Detection** -- For CSV/TSV files, the delimiter is detected from candidates (comma, tab, semicolon, pipe).

7. **Column Mapping** -- Source column headers are mapped to canonical GreenLang fields using configurable strategies:
   - `exact` -- Exact string match against canonical field names
   - `synonym` -- Match against a curated synonym dictionary (e.g., "Electric Usage" -> "electricity_consumption_kwh")
   - `fuzzy` -- Levenshtein distance-based fuzzy matching
   - `ml` -- ML-based column name classification (optional)
   - `template` -- Template-based mapping from pre-configured templates
   - `regex` -- Regular expression pattern matching
   - `manual` -- User-specified explicit mappings

8. **Data Type Inference** -- For each column, the data type is inferred (string, integer, float, decimal, boolean, date, datetime, currency, percentage, etc.).

9. **Normalization** -- Data is normalized with type casting, unit conversion, date parsing, string cleaning, null handling, and quality scoring.

10. **Validation** -- Normalized data is validated against configurable rules (type checks, range validation, required fields, format patterns, cross-column consistency).

### Common Failure Modes

| Failure Mode | Typical Cause | Impact |
|--------------|---------------|--------|
| Encoding error | Wrong encoding detected, binary data in CSV | File cannot be parsed; entire sheet fails |
| Corrupt file | Truncated XLSX, invalid zip structure | File cannot be opened |
| Oversized file | >50MB, >1M rows, >1000 columns | Processing timeout or OOM |
| Header detection failure | No clear header row, merged cells in Excel | Columns mapped incorrectly |
| Column mapping failure | Non-standard headers, foreign language columns | Fields normalized to wrong canonical names |
| Delimiter mismatch | Wrong delimiter detected for CSV | Columns parsed incorrectly |
| Date format ambiguity | MM/DD/YYYY vs DD/MM/YYYY | Dates parsed incorrectly |
| Type mismatch | Numbers stored as text, mixed types in column | Type casting errors |
| Duplicate columns | Same header name in multiple columns | Mapping ambiguity |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Normalized data may be incorrect or unavailable |
| **Data Impact** | High | Emission calculations use normalized tabular data; bad parsing = bad calculations |
| **SLA Impact** | Medium | Spreadsheet processing SLA degraded but not fully blocked |
| **Revenue Impact** | Medium | Compliance-sensitive customers require accurate data normalization |
| **Compliance Impact** | High | CSRD/CBAM require verifiable source data from spreadsheets |
| **Downstream Impact** | High | Emission calculation agents receive degraded or missing input data |

---

## Symptoms

- `gl_excel_normalizer_mapping_confidence` average is below 0.7
- `gl_excel_normalizer_mapping_errors_total` counter is incrementing rapidly
- `gl_excel_normalizer_jobs_total{status="failed"}` rate is elevated
- `gl_excel_normalizer_jobs_total{status="timeout"}` rate is elevated
- `gl_excel_normalizer_validation_results_total{severity="error"}` rate is elevated
- `gl_excel_normalizer_encoding_errors_total` counter is incrementing
- Grafana Excel Normalizer dashboard shows confidence drop or failure spike
- Users report incorrect normalization results or missing columns
- Normalization jobs fail with parsing-related error messages in logs

---

## Diagnostic Steps

### Step 1: Identify the Scope of Parsing Failures

```bash
# Check mapping confidence metrics
kubectl port-forward -n greenlang svc/excel-normalizer-service 8080:8080
curl -s http://localhost:8080/metrics | grep gl_excel_normalizer_mapping

# Check which mapping strategies are failing
curl -s http://localhost:8080/metrics | grep gl_excel_normalizer_mapping_errors

# Check recent failed normalization jobs
curl -s "http://localhost:8080/v1/jobs?status=failed&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 2: Identify the Failure Pattern

```bash
# Check if failures are concentrated on a specific file format
curl -s http://localhost:8080/metrics | grep gl_excel_normalizer_jobs_total

# Check if failures correlate with file format
curl -s "http://localhost:8080/v1/jobs?status=failed&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('items', data if isinstance(data, list) else [])
for job in items:
    print(f\"Job: {job.get('job_id', '?')[:8]}  Format: {job.get('file_format', '?')}  Rows: {job.get('rows_processed', '?')}  Error: {str(job.get('errors', '?'))[:100]}\")
"

# Check for encoding-related failures
kubectl logs -n greenlang -l app=excel-normalizer-service --tail=500 \
  | grep -i "encoding\|codec\|decode\|utf\|ascii\|chardet\|charset"
```

### Step 3: Check File Parsing Health

```bash
# Check for encoding errors
curl -s http://localhost:8080/metrics | grep gl_excel_normalizer_encoding_errors

# Check for delimiter detection issues
kubectl logs -n greenlang -l app=excel-normalizer-service --tail=500 \
  | grep -i "delimiter\|separator\|csv\|tsv\|parse"

# Check for corrupt file errors
kubectl logs -n greenlang -l app=excel-normalizer-service --tail=500 \
  | grep -i "corrupt\|invalid\|truncated\|zip\|openpyxl\|xlrd"
```

### Step 4: Examine Specific File Failures

```bash
# Get details for a specific failed job
curl -s "http://localhost:8080/v1/jobs/<job_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Get sheet metadata for a file
curl -s "http://localhost:8080/v1/files/<file_id>/sheets" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Get column mappings for a sheet
curl -s "http://localhost:8080/v1/sheets/<sheet_id>/mappings" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Get validation results for a file
curl -s "http://localhost:8080/v1/files/<file_id>/validations" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 5: Check Memory and Resource Pressure

```bash
# Check memory usage (large spreadsheets load entirely into memory)
kubectl top pods -n greenlang -l app=excel-normalizer-service

# Check for OOMKilled events
kubectl get pods -n greenlang -l app=excel-normalizer-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check upload staging volume
kubectl exec -n greenlang <pod-name> -- df -h /app/uploads /tmp
```

### Step 6: Check Data Quality Trends

```bash
# Check recent quality reports with low scores
curl -s "http://localhost:8080/v1/quality-reports?sort=overall_score_asc&limit=10" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check per-column quality for a specific file
curl -s "http://localhost:8080/v1/files/<file_id>/quality" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Overall: {data.get('overall_score', 0):.2f}  Completeness: {data.get('completeness_score', 0):.2f}  Accuracy: {data.get('accuracy_score', 0):.2f}  Consistency: {data.get('consistency_score', 0):.2f}\")
for col, score in data.get('column_scores', {}).items():
    print(f\"  Column '{col}': {score:.2f}\")
"
```

---

## Resolution Steps

### Scenario 1: Encoding Detection Failure

**Symptoms:** Encoding errors concentrated on CSV/TSV files. XLSX files process normally. Logs show codec or decode errors.

**Resolution:**

1. Identify the problematic encoding:
```bash
kubectl logs -n greenlang -l app=excel-normalizer-service --tail=500 \
  | grep -i "encoding\|codec\|decode"
```

2. Force a specific encoding for the problematic files:
```bash
curl -s -X POST "http://localhost:8080/v1/normalize" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "<file_id>",
    "encoding": "windows-1252",
    "delimiter": ","
  }' | python3 -m json.tool
```

3. If a specific encoding is consistently misdetected, adjust the detection threshold:
```bash
kubectl set env deployment/excel-normalizer-service -n greenlang \
  GL_EXCEL_NORMALIZER_ENCODING_CONFIDENCE_THRESHOLD=0.5
```

4. Reprocess failed files with explicit encoding:
```bash
curl -s -X POST "http://localhost:8080/v1/jobs/reprocess" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "failed", "override_encoding": "windows-1252"}' \
  | python3 -m json.tool
```

### Scenario 2: Corrupt or Malformed Files

**Symptoms:** Specific files consistently fail. Logs show zip, openpyxl, or xlrd errors. Other files process normally.

**Resolution:**

1. Identify the corrupt files:
```bash
curl -s "http://localhost:8080/v1/jobs?status=failed&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. Validate file integrity:
```bash
# Download the problematic file and check locally
curl -s "http://localhost:8080/v1/files/<file_id>/download" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -o /tmp/test_file.xlsx

# Attempt to open with Python
python3 -c "
import openpyxl
try:
    wb = openpyxl.load_workbook('/tmp/test_file.xlsx')
    print(f'Sheets: {wb.sheetnames}')
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        print(f'  {sheet}: {ws.max_row} rows x {ws.max_column} cols')
except Exception as e:
    print(f'Error: {e}')
"
```

3. If files are confirmed corrupt:
   - Notify the file submitter to re-upload uncorrupted versions
   - Mark the failed jobs as permanently failed

4. For partially corrupt XLSX files, attempt CSV export:
   - Users can re-save as CSV from Excel and re-upload

### Scenario 3: Oversized Files Causing Timeouts or OOM

**Symptoms:** Timeout or OOMKilled errors. Files are very large (>50MB, >1M rows).

**Resolution:**

1. Check file sizes causing issues:
```bash
curl -s "http://localhost:8080/v1/jobs?status=timeout&limit=10" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

curl -s "http://localhost:8080/v1/files?sort=file_size_desc&limit=10" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. Increase timeout for large files:
```bash
kubectl set env deployment/excel-normalizer-service -n greenlang \
  GL_EXCEL_NORMALIZER_JOB_TIMEOUT_SECONDS=1200
```

3. Increase memory limits if OOMKilled:
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

4. If files are excessively large:
   - Split spreadsheets into smaller chunks (e.g., by sheet or by row ranges) before upload
   - Reduce max_rows_per_sheet configuration
   - Increase batch_size for chunked processing

### Scenario 4: Column Mapping Failures (Non-Standard Headers)

**Symptoms:** Low mapping confidence, mapping errors increasing. Files parse successfully but columns map to wrong canonical fields.

**Resolution:**

1. Check which columns are failing to map:
```bash
curl -s "http://localhost:8080/v1/sheets/<sheet_id>/mappings" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('items', data if isinstance(data, list) else [])
for m in items:
    conf = m.get('confidence', 0)
    marker = '***' if conf < 0.6 else ''
    print(f\"{marker} '{m.get('source_column', '?')}' -> '{m.get('canonical_field', '?')}' (confidence={conf:.2f}, strategy={m.get('mapping_strategy', '?')})\")
"
```

2. Update the synonym dictionary to include the missing terms:
```bash
kubectl edit configmap excel-normalizer-service-config -n greenlang
# Add new synonyms to the synonyms.json configuration
```

3. Create a mapping template for the recurring file format:
```bash
curl -s -X POST "http://localhost:8080/v1/templates" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template_name": "custom_utility_bill_v2",
    "source_type": "utility_bill",
    "column_mappings": {
      "Electric Usage (kWh)": "electricity_consumption_kwh",
      "Gas Usage (therms)": "gas_consumption_therms",
      "Billing Period Start": "period_start_date",
      "Billing Period End": "period_end_date"
    }
  }' | python3 -m json.tool
```

4. Reprocess files using the new template.

### Scenario 5: Date Format Ambiguity

**Symptoms:** Dates parsed incorrectly (month/day swapped). Validation errors on date fields.

**Resolution:**

1. Identify which files have date issues:
```bash
curl -s "http://localhost:8080/v1/files/<file_id>/validations?rule_name=date_format" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. Specify explicit date format for the file:
```bash
curl -s -X POST "http://localhost:8080/v1/normalize" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "<file_id>",
    "date_format": "DD/MM/YYYY",
    "date_columns": ["invoice_date", "service_period_start", "service_period_end"]
  }' | python3 -m json.tool
```

3. Update the template to include date format hints for recurring formats.

### Scenario 6: Memory Pressure from Concurrent Jobs

**Symptoms:** Multiple large files being processed simultaneously, OOMKilled events.

**Resolution:**

1. Reduce concurrent jobs to lower memory pressure:
```bash
kubectl set env deployment/excel-normalizer-service -n greenlang \
  GL_EXCEL_NORMALIZER_MAX_CONCURRENT_JOBS=2
```

2. Reduce batch size for row processing:
```bash
kubectl set env deployment/excel-normalizer-service -n greenlang \
  GL_EXCEL_NORMALIZER_BATCH_SIZE=500
```

3. Scale up to more replicas to distribute the load:
```bash
kubectl scale deployment/excel-normalizer-service -n greenlang --replicas=4
```

---

## Post-Incident Steps

### Step 1: Verify Normalization Quality Is Restored

```bash
# Check that mapping confidence is back to normal
curl -s http://localhost:8080/metrics | grep gl_excel_normalizer_mapping_confidence

# Check that error rate is back to normal
curl -s http://localhost:8080/metrics | grep gl_excel_normalizer_mapping_errors

# Check that normalization success rate is back to normal
curl -s http://localhost:8080/metrics | grep gl_excel_normalizer_jobs_total
```

### Step 2: Reprocess Failed Files

```bash
# Reprocess all files that failed during the incident
curl -s -X POST "http://localhost:8080/v1/jobs/reprocess" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "failed", "since": "<incident_start_time>"}' \
  | python3 -m json.tool
```

### Step 3: Verify Downstream Data Quality

```bash
# Check that normalized records are being produced
curl -s "http://localhost:8080/v1/records?limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check data quality reports
curl -s "http://localhost:8080/v1/quality-reports?limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Verify Audit Trail Integrity

```promql
# Verify audit events are being recorded
increase(gl_excel_normalizer_audit_events_total[5m])

# Check for audit gaps
(
  sum(rate(gl_excel_normalizer_jobs_total[5m]))
  - sum(rate(gl_excel_normalizer_audit_events_total[5m]))
)
```

---

## Interim Mitigation

While parsing issues are being resolved:

1. **Try alternative file formats.** If CSV files fail with encoding issues, ask users to save as XLSX. If XLSX fails, try CSV with explicit UTF-8 encoding.

2. **Use manual column mappings.** For files with non-standard headers, provide explicit column mappings via the API instead of relying on auto-detection.

3. **Lower quality threshold temporarily.** Accept lower-quality normalizations with manual review flags rather than blocking all processing.

4. **Enable manual data entry bypass.** For time-critical emission reports, allow manual entry of utility bill and fuel data.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-data` -- engineering response
   - `#data-pipeline-ops` -- data pipeline impact
   - `#compliance-ops` -- compliance impact if normalization quality is degraded

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Low confidence or validation errors, investigation in progress | On-call engineer | 15 minutes |
| L2 | Column mapping failure, normalization jobs failing at >10% | Platform team lead + #platform-data | Immediate (<5 min) |
| L3 | All normalization failing, spreadsheet ingestion blocked | Platform team + data team + CTO notification | Immediate |
| L4 | Systemic failure affecting emission calculations downstream | All-hands engineering + incident commander + executive notification | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Excel Normalizer Health (`/d/excel-normalizer-service`)
- **Alerts:** `ExcelColumnMappingFailure`, `ExcelLowMappingConfidence`, `ExcelValidationErrorSpike`, `ExcelHighNormalizationFailureRateCritical`
- **Key metrics to watch:**
  - `gl_excel_normalizer_mapping_confidence` avg (should be > 0.7)
  - `gl_excel_normalizer_mapping_errors_total` (should be 0 or near-zero)
  - `gl_excel_normalizer_jobs_total{status="failed"}` rate (should be < 10%)
  - `gl_excel_normalizer_jobs_total{status="timeout"}` rate (should be near 0)
  - `gl_excel_normalizer_validation_results_total{severity="error"}` rate (should be < 20%)
  - `gl_excel_normalizer_encoding_errors_total` (should be near 0)
  - `gl_excel_normalizer_job_duration_seconds` p99 (should be < 30s)

### Best Practices

1. **Maintain comprehensive synonym dictionaries** with common variations of column names across industries and languages
2. **Create mapping templates** for each recurring spreadsheet format from data providers
3. **Set appropriate file size limits** and row count limits per the service capacity
4. **Monitor encoding detection accuracy** and update confidence thresholds as needed
5. **Test template changes** against a golden set of spreadsheets before deploying
6. **Review normalization quality weekly** and update templates and synonyms as needed
7. **Document expected column headers** for each data source type (utility bills, fuel logs, etc.)
8. **Validate date formats** per data source region (US vs EU vs Asia formatting)

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `ExcelColumnMappingFailure` | Critical | Column mapping errors >10 in 5 minutes |
| `ExcelLowMappingConfidence` | Warning | Average mapping confidence below 60% |
| `ExcelValidationErrorSpike` | Warning | >20% of validations are errors |
| `ExcelHighNormalizationFailureRateWarning` | Warning | >10% normalization failure rate |
| `ExcelHighNormalizationFailureRateCritical` | Critical | >25% normalization failure rate |
| `ExcelNormalizationTimeout` | Warning | Normalization timeout rate elevated |
| `ExcelNormalizerServiceDown` | Critical | No healthy pods running |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Data Team
- **Review cadence:** Quarterly or after any P1 normalization parsing incident
- **Related runbooks:** [Excel Normalizer Service Down](./excel-normalizer-service-down.md)
