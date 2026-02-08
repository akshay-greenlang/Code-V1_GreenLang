# PDF Extractor Service Down

## Alert

**Alert Name:** `PDFExtractorServiceDown`

**Severity:** Critical

**Threshold:** `up{job="pdf-extractor-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang PDF & Invoice Extractor (AGENT-DATA-001) are running. The PDF Extractor is the document ingestion service for all GreenLang Climate OS document-based data sources. It is responsible for:

1. **Document upload and registration** -- Accepting PDF, PNG, JPG, TIFF, and BMP documents with SHA-256 file hash deduplication and metadata capture
2. **Multi-engine OCR extraction** -- Extracting raw text from document pages using configurable OCR engines (Tesseract, AWS Textract, Azure Form Recognizer, Google Document AI, EasyOCR, PaddleOCR, native PDF text)
3. **Structured field extraction** -- Extracting typed fields (text, number, currency, date, address, etc.) from document text using template matching, regex patterns, ML models, and rule-based methods
4. **Invoice extraction** -- Parsing structured invoice data including vendor, line items, totals, dates, PO numbers, and tax amounts with cross-field validation
5. **Utility bill extraction** -- Parsing utility bill data including provider, account, consumption (kWh, therms, gallons), rates, and meter readings for emission calculations
6. **Shipping manifest extraction** -- Parsing manifest data including shipper, consignee, cargo details, weight, and route for supply chain emission tracking
7. **Template management** -- Maintaining configurable extraction templates with field patterns and validation rules for repeatable extraction
8. **Field validation** -- Validating extracted fields against configurable rules with severity classification (error, warning, info)
9. **Provenance hash chains** -- Maintaining SHA-256 hash chains across all extracted data and audit events for tamper detection and compliance
10. **Emitting Prometheus metrics** (12+ metrics under the `gl_pdf_extractor_*` prefix) for monitoring extraction rates, OCR confidence, validation results, and service health

When the PDF Extractor is down:
- **Document ingestion stops** and no new invoices, utility bills, or manifests can be processed
- **Emission calculation data feed is interrupted** from document-based sources (utility bills, fuel invoices)
- **OCR extraction queue will grow** and documents will accumulate without processing
- **Field validation is unavailable** and extracted data quality cannot be verified
- **Audit trail has a gap** and compliance requirements for traceable data ingestion are violated

**Note:** All documents, pages, extraction jobs, extracted fields, invoice/manifest/utility bill extractions, templates, validation results, and audit events are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full state will be immediately available. Pending extraction jobs will need to be reprocessed.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | No new documents can be uploaded or processed for extraction |
| **Data Impact** | High | Document-based emission data feed interrupted; extraction queue growing |
| **SLA Impact** | High | Document processing SLA violated (all extraction jobs fail) |
| **Revenue Impact** | Medium | Compliance-sensitive customers require timely document processing |
| **Compliance Impact** | High | CSRD, CBAM require traceable, verifiable source document processing |
| **Downstream Impact** | High | Emission calculation agents waiting for extracted utility/invoice data |

---

## Symptoms

- `up{job="pdf-extractor-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=pdf-extractor-service`
- `gl_pdf_extractor_jobs_total` counter stops incrementing
- `gl_pdf_extractor_documents_uploaded_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Document upload endpoints return errors
- Grafana PDF Extractor dashboard shows "No Data" or stale timestamps
- Extraction job queue backlog grows without being processed

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List PDF extractor service pods
kubectl get pods -n greenlang -l app=pdf-extractor-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=pdf-extractor-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to PDF extractor service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=pdf-extractor-service | tail -30

# Check deployment status
kubectl describe deployment pdf-extractor-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment pdf-extractor-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=pdf-extractor-service

# Check for rollout issues
kubectl rollout status deployment/pdf-extractor-service -n greenlang

# Check HPA status (scales 2-8 replicas)
kubectl get hpa -n greenlang -l app=pdf-extractor-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for OCR-specific errors
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=500 \
  | grep -i "ocr\|tesseract\|textract\|extraction\|confidence\|template"

# Look for database connection errors
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of PDF extractor service pods
kubectl top pods -n greenlang -l app=pdf-extractor-service

# Check if pods were OOMKilled (OCR processing is memory-intensive)
kubectl get pods -n greenlang -l app=pdf-extractor-service \
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

# Check if the pdf_extractor_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='pdf_extractor_service'
   ORDER BY table_name;"
```

### Step 6: Check ConfigMap and Secrets

```bash
# Verify the PDF extractor service ConfigMap exists and is valid
kubectl get configmap pdf-extractor-service-config -n greenlang
kubectl get configmap pdf-extractor-service-config -n greenlang -o yaml | head -50

# Verify secrets exist (including OCR provider credentials)
kubectl get secret pdf-extractor-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment pdf-extractor-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 7: Check Network Policies

```bash
# Check network policies affecting the PDF extractor service
kubectl get networkpolicy -n greenlang | grep pdf-extractor

# Verify the PDF extractor service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify egress to S3 (HTTPS) for document storage
kubectl run net-test-s3 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv s3.amazonaws.com 443'
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137. Common with large PDF files or high-resolution images.

**Resolution:**

1. Confirm the OOM cause:
```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. Increase memory limits (OCR processing requires more memory for large documents):
```bash
kubectl patch deployment pdf-extractor-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "pdf-extractor-service",
            "resources": {
              "limits": {
                "cpu": "2",
                "memory": "2Gi"
              },
              "requests": {
                "cpu": "500m",
                "memory": "1Gi"
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
kubectl rollout status deployment/pdf-extractor-service -n greenlang
kubectl get pods -n greenlang -l app=pdf-extractor-service
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
kubectl rollout restart deployment/pdf-extractor-service -n greenlang
kubectl rollout status deployment/pdf-extractor-service -n greenlang
```

### Scenario 3: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/pdf-extractor-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/pdf-extractor-service -n greenlang
kubectl rollout status deployment/pdf-extractor-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=pdf-extractor-service
kubectl port-forward -n greenlang svc/pdf-extractor-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=pdf-extractor-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/pdf-extractor-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the PDF extractor service is being scraped
up{job="pdf-extractor-service"} == 1

# Verify extraction job count metric is populated
gl_pdf_extractor_jobs_total > 0

# Verify document upload metrics are incrementing
increase(gl_pdf_extractor_documents_uploaded_total[5m])
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

While the PDF Extractor is being restored:

1. **Extraction data is safe.** All documents, pages, extraction jobs, extracted fields, structured extractions, templates, validation results, and audit events are stored in PostgreSQL with TimescaleDB. The database persists independently.

2. **Documents will queue up.** Uploaded documents will be stored but not processed. Once the service recovers, the extraction queue will need to be drained.

3. **Emission calculations from document sources are delayed.** Utility bill and invoice data will not be available for emission calculations until documents are processed.

4. **Manual data entry may be needed.** For time-critical emission reports, manual data entry can bypass the PDF extraction pipeline temporarily.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-data` -- engineering response
   - `#platform-oncall` -- on-call engineer
   - `#data-pipeline-ops` -- data pipeline impact notification

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | PDF extractor down, document queue growing | On-call engineer | Immediate (<5 min) |
| L2 | PDF extractor down > 15 minutes, emission data feed interrupted | Platform team lead + #platform-data | 15 minutes |
| L3 | PDF extractor down > 30 minutes, compliance reporting blocked | Platform team + compliance team + CTO notification | Immediate |
| L4 | PDF extractor down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** PDF Extractor Health (`/d/pdf-extractor-service`)
- **Alert:** `PDFExtractorServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="pdf-extractor-service"}` (should always be >= 2)
  - `gl_pdf_extractor_jobs_total` rate (should be non-zero during business hours)
  - `gl_pdf_extractor_documents_uploaded_total` rate (should be non-zero)
  - `gl_pdf_extractor_ocr_confidence` avg (should stay above 0.7)
  - `gl_pdf_extractor_ocr_errors_total` (should be 0 or near-zero)
  - `gl_pdf_extractor_job_duration_seconds` p99 (should stay below 30s)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 8 replicas** based on CPU and memory utilization
4. **Upload staging volume** sized at 2Gi for document buffering
5. **Database connection pool** is sized for expected concurrency (default: min 2, max 10)

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `PDFExtractorServiceDown` | Critical | This alert -- no PDF extractor pods running |
| `PDFHighExtractionFailureRateCritical` | Critical | >25% of extraction jobs are failing |
| `PDFOCREngineFailure` | Critical | OCR engine errors exceeding threshold |
| `PDFAuditGap` | Critical | Extraction operations without audit entries |
| `PDFDatabaseConnectionFailure` | Critical | Database connection errors |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Data Team
- **Review cadence:** Quarterly or after any P1 PDF extractor incident
- **Related runbooks:** [PDF Extractor OCR Failures](./pdf-extractor-ocr-failures.md)
