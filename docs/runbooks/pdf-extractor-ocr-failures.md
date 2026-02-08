# PDF Extractor OCR Failures

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `PDFLowOCRConfidence` | Warning | Average OCR confidence below 60% for 15 minutes |
| `PDFOCREngineFailure` | Critical | >10 OCR engine errors in 5 minutes |
| `PDFValidationErrorSpike` | Warning | >20% of field validations producing errors for 15 minutes |
| `PDFHighExtractionFailureRateWarning` | Warning | >10% extraction job failure rate for 10 minutes |
| `PDFHighExtractionFailureRateCritical` | Critical | >25% extraction job failure rate for 5 minutes |
| `PDFExtractionTimeout` | Warning | Extraction timeouts exceeding 0.1/sec for 15 minutes |

**Thresholds:**

```promql
# PDFLowOCRConfidence
# Average OCR confidence below 0.60 for 15 minutes
avg(gl_pdf_extractor_ocr_confidence) < 0.6

# PDFOCREngineFailure
# More than 10 OCR engine errors in 5 minutes
increase(gl_pdf_extractor_ocr_errors_total[5m]) > 10

# PDFValidationErrorSpike
# More than 20% validation errors for 15 minutes
(sum(rate(gl_pdf_extractor_validation_results_total{severity="error"}[5m]))
 / sum(rate(gl_pdf_extractor_validation_results_total[5m]))) > 0.20

# PDFHighExtractionFailureRateWarning
# More than 10% extraction failure rate for 10 minutes
(sum(rate(gl_pdf_extractor_jobs_total{status="failed"}[5m]))
 / sum(rate(gl_pdf_extractor_jobs_total[5m]))) > 0.10

# PDFHighExtractionFailureRateCritical
# More than 25% extraction failure rate for 5 minutes
(sum(rate(gl_pdf_extractor_jobs_total{status="failed"}[5m]))
 / sum(rate(gl_pdf_extractor_jobs_total[5m]))) > 0.25

# PDFExtractionTimeout
# Extraction timeout rate above 0.1/sec for 15 minutes
sum(rate(gl_pdf_extractor_jobs_total{status="timeout"}[5m])) > 0.1
```

---

## Description

These alerts fire when the PDF & Invoice Extractor (AGENT-DATA-001) encounters OCR extraction failures, low confidence scores, or elevated validation error rates. OCR failures directly impact the quality and availability of data extracted from documents (invoices, utility bills, shipping manifests) that feed into GreenLang's emission calculations.

### How OCR Extraction Works

The PDF Extractor uses a multi-engine OCR pipeline:

1. **Document Upload** -- Documents (PDF, PNG, JPG, TIFF, BMP) are uploaded and stored with SHA-256 file hash for deduplication.

2. **Page Processing** -- Multi-page documents are split into individual pages. Each page is processed independently with the configured OCR engine.

3. **OCR Engine Selection** -- The default engine is Tesseract (local), with fallback engines available:
   - `tesseract` -- Local, open-source, fast, good for clean documents
   - `textract` -- AWS Textract, cloud-based, excellent for complex layouts
   - `azure_form_recognizer` -- Azure, cloud-based, strong for invoices/receipts
   - `google_document_ai` -- Google Cloud, strong for handwritten text
   - `easyocr` -- Local, ML-based, good multi-language support
   - `paddleocr` -- Local, ML-based, strong for structured documents
   - `native_pdf` -- Direct text extraction from digital PDFs (no OCR needed)

4. **Confidence Scoring** -- Each extracted page and field receives a confidence score (0.0 to 1.0). Fields below the confidence threshold (default 0.7) are flagged for review.

5. **Field Extraction** -- Structured fields are extracted from OCR text using templates (regex patterns), ML models, or rule-based methods.

6. **Validation** -- Extracted fields are validated against configurable rules (type checks, range validation, cross-field consistency, format verification).

### Common Failure Modes

| Failure Mode | Typical Cause | Impact |
|--------------|---------------|--------|
| Low confidence | Poor scan quality, skewed documents, unusual fonts | Fields may be incorrect; manual review needed |
| OCR engine down | API key expired, rate limit, network failure | All extraction for that engine fails |
| Template mismatch | Document layout changed, wrong template applied | Fields extracted incorrectly or missed |
| Timeout | Very large documents (100+ pages), complex layouts | Job fails; document not processed |
| Validation errors | Extracted values fail business rules | Data quality degraded |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Extracted data may be incorrect or unavailable |
| **Data Impact** | High | Emission calculations use extracted utility/invoice data; bad OCR = bad calculations |
| **SLA Impact** | Medium | Document processing SLA degraded but not fully blocked |
| **Revenue Impact** | Medium | Compliance-sensitive customers require accurate document extraction |
| **Compliance Impact** | High | CSRD/CBAM require verifiable source data from documents |
| **Downstream Impact** | High | Emission calculation agents receive degraded or missing input data |

---

## Symptoms

- `gl_pdf_extractor_ocr_confidence` average is below 0.7
- `gl_pdf_extractor_ocr_errors_total` counter is incrementing rapidly
- `gl_pdf_extractor_jobs_total{status="failed"}` rate is elevated
- `gl_pdf_extractor_jobs_total{status="timeout"}` rate is elevated
- `gl_pdf_extractor_validation_results_total{severity="error"}` rate is elevated
- `gl_pdf_extractor_low_confidence_extractions_total` counter is incrementing
- Grafana PDF Extractor dashboard shows confidence drop or failure spike
- Users report incorrect extraction results or missing fields
- Extraction jobs fail with OCR-related error messages in logs

---

## Diagnostic Steps

### Step 1: Identify the Scope of OCR Failures

```bash
# Check OCR confidence metrics by engine
kubectl port-forward -n greenlang svc/pdf-extractor-service 8080:8080
curl -s http://localhost:8080/metrics | grep gl_pdf_extractor_ocr

# Check which OCR engines are failing
curl -s http://localhost:8080/metrics | grep gl_pdf_extractor_ocr_errors

# Check recent failed extraction jobs
curl -s "http://localhost:8080/v1/jobs?status=failed&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 2: Identify the Failure Pattern

```bash
# Check if failures are concentrated on a specific document type
curl -s http://localhost:8080/metrics | grep gl_pdf_extractor_jobs_total

# Check if failures are concentrated on a specific OCR engine
curl -s "http://localhost:8080/v1/jobs?status=failed&ocr_engine=tesseract&limit=10" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check if failures correlate with document format
curl -s "http://localhost:8080/v1/jobs?status=failed&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('items', data if isinstance(data, list) else [])
for job in items:
    print(f\"Job: {job.get('job_id', '?')[:8]}  Type: {job.get('document_type', '?')}  Engine: {job.get('ocr_engine', '?')}  Error: {str(job.get('error_message', '?'))[:100]}\")
"
```

### Step 3: Check OCR Engine Health

```bash
# Check Tesseract binary availability
kubectl exec -n greenlang <pod-name> -- which tesseract
kubectl exec -n greenlang <pod-name> -- tesseract --version

# Check AWS Textract connectivity (if enabled)
kubectl exec -n greenlang <pod-name> -- python3 -c "
import boto3
client = boto3.client('textract', region_name='us-east-1')
print('Textract client created successfully')
"

# Check Azure Form Recognizer connectivity (if enabled)
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=200 \
  | grep -i "azure\|form_recognizer\|cognitive"

# Check Google Document AI connectivity (if enabled)
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=200 \
  | grep -i "google\|document_ai\|processor"
```

### Step 4: Examine Specific Document Failures

```bash
# Get details for a specific failed job
curl -s "http://localhost:8080/v1/jobs/<job_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Get extracted pages with confidence scores
curl -s "http://localhost:8080/v1/documents/<document_id>/pages" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Get validation errors for a document
curl -s "http://localhost:8080/v1/documents/<document_id>/validations" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 5: Check Cloud Provider Rate Limits

```bash
# Check for rate limit errors in logs
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=500 \
  | grep -i "rate limit\|throttl\|429\|quota\|too many"

# Check for authentication errors
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=500 \
  | grep -i "auth\|credential\|unauthorized\|403\|forbidden\|expired"

# Check for timeout errors
kubectl logs -n greenlang -l app=pdf-extractor-service --tail=500 \
  | grep -i "timeout\|timed out\|deadline\|context canceled"
```

### Step 6: Check Document Quality

```bash
# Check recent documents with low confidence
curl -s "http://localhost:8080/v1/documents?sort=confidence_asc&limit=10" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check page-level confidence for a specific document
curl -s "http://localhost:8080/v1/documents/<document_id>/pages" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
items = data.get('items', data if isinstance(data, list) else [])
for page in items:
    print(f\"Page {page.get('page_number')}: confidence={page.get('confidence', 0):.2f} words={page.get('word_count', 0)} engine={page.get('ocr_engine_used', '?')}\")
"
```

---

## Resolution Steps

### Scenario 1: OCR Engine Connectivity Failure (Cloud Providers)

**Symptoms:** OCR errors concentrated on cloud engines (Textract, Azure, Google). Local Tesseract still works.

**Resolution:**

1. Verify API credentials are valid:
```bash
kubectl get secret pdf-extractor-service-secrets -n greenlang -o yaml | head -20
```

2. Switch to fallback engine temporarily:
```bash
kubectl set env deployment/pdf-extractor-service -n greenlang \
  GL_PDF_EXTRACTOR_DEFAULT_OCR_ENGINE=tesseract
```

3. Once cloud provider is restored, switch back:
```bash
kubectl set env deployment/pdf-extractor-service -n greenlang \
  GL_PDF_EXTRACTOR_DEFAULT_OCR_ENGINE=textract
```

4. Reprocess failed jobs that used the unavailable engine:
```bash
curl -s -X POST "http://localhost:8080/v1/jobs/reprocess" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "failed", "ocr_engine": "textract"}' \
  | python3 -m json.tool
```

### Scenario 2: Low OCR Confidence (Document Quality)

**Symptoms:** Average confidence dropping across all engines. Not engine-specific.

**Resolution:**

1. Check if a batch of low-quality documents was uploaded:
```bash
curl -s "http://localhost:8080/v1/documents?sort=upload_timestamp_desc&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. If documents are low quality (blurry scans, poor contrast):
   - Notify the document submitter to re-scan at higher resolution (300+ DPI)
   - Apply image preprocessing (deskew, contrast enhancement) if available

3. If a specific document type consistently has low confidence:
   - Create or update extraction templates for that document type
   - Consider switching to a more suitable OCR engine for that type

4. Temporarily lower the confidence threshold if needed:
```bash
kubectl set env deployment/pdf-extractor-service -n greenlang \
  GL_PDF_EXTRACTOR_CONFIDENCE_THRESHOLD=0.5
```

### Scenario 3: Template Mismatch (Wrong Fields Extracted)

**Symptoms:** High validation error rate, fields extracted with wrong names or values.

**Resolution:**

1. Check which templates are being applied:
```bash
curl -s "http://localhost:8080/v1/templates?limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. Update the template with corrected field patterns:
```bash
curl -s -X PATCH "http://localhost:8080/v1/templates/<template_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "field_patterns": { ... },
    "validation_rules": { ... }
  }' | python3 -m json.tool
```

3. Reprocess documents that used the incorrect template.

### Scenario 4: Extraction Timeouts (Large Documents)

**Symptoms:** Timeout rate increasing, primarily affecting large documents (100+ pages).

**Resolution:**

1. Check document sizes causing timeouts:
```bash
curl -s "http://localhost:8080/v1/jobs?status=timeout&limit=10" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

2. Increase timeout for large documents:
```bash
kubectl set env deployment/pdf-extractor-service -n greenlang \
  GL_PDF_EXTRACTOR_EXTRACTION_TIMEOUT_SECONDS=600
```

3. If documents are excessively large:
   - Split documents into smaller chunks before upload
   - Reduce max_pages_per_document configuration
   - Scale up to more replicas with higher CPU/memory

### Scenario 5: Rate Limiting by Cloud Provider

**Symptoms:** Intermittent failures on cloud OCR engines, 429 status codes in logs.

**Resolution:**

1. Reduce concurrent jobs to stay within rate limits:
```bash
kubectl set env deployment/pdf-extractor-service -n greenlang \
  GL_PDF_EXTRACTOR_MAX_CONCURRENT_JOBS=2
```

2. Enable rate limiting in the service configuration:
```bash
kubectl edit configmap pdf-extractor-service-config -n greenlang
# Add or update rate limiting settings in the config YAML
```

3. Contact the cloud provider to increase rate limits if needed.

4. Distribute load across multiple OCR engines.

---

## Post-Incident Steps

### Step 1: Verify OCR Quality Is Restored

```bash
# Check that OCR confidence is back to normal
curl -s http://localhost:8080/metrics | grep gl_pdf_extractor_ocr_confidence

# Check that error rate is back to normal
curl -s http://localhost:8080/metrics | grep gl_pdf_extractor_ocr_errors

# Check that extraction success rate is back to normal
curl -s http://localhost:8080/metrics | grep gl_pdf_extractor_jobs_total
```

### Step 2: Reprocess Failed Documents

```bash
# Reprocess all documents that failed during the incident
curl -s -X POST "http://localhost:8080/v1/jobs/reprocess" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "failed", "since": "<incident_start_time>"}' \
  | python3 -m json.tool
```

### Step 3: Verify Downstream Data Quality

```bash
# Check that extraction results are being consumed by downstream agents
curl -s "http://localhost:8080/v1/extractions/invoices?limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check utility bill extractions
curl -s "http://localhost:8080/v1/extractions/utility-bills?limit=5" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Verify Audit Trail Integrity

```promql
# Verify audit events are being recorded
increase(gl_pdf_extractor_audit_events_total[5m])

# Check for audit gaps
(
  sum(rate(gl_pdf_extractor_jobs_total[5m]))
  - sum(rate(gl_pdf_extractor_audit_events_total[5m]))
)
```

---

## Interim Mitigation

While OCR issues are being resolved:

1. **Switch to fallback OCR engine.** If one engine is down, switch to Tesseract (always available locally) or another cloud provider.

2. **Lower confidence threshold temporarily.** Accept lower-confidence extractions with manual review flags rather than blocking all processing.

3. **Enable manual data entry bypass.** For time-critical emission reports, allow manual entry of utility bill and invoice data.

4. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-data` -- engineering response
   - `#data-pipeline-ops` -- data pipeline impact
   - `#compliance-ops` -- compliance impact if extraction quality is degraded

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Low confidence or validation errors, investigation in progress | On-call engineer | 15 minutes |
| L2 | OCR engine failure, extraction jobs failing at >10% | Platform team lead + #platform-data | Immediate (<5 min) |
| L3 | All OCR engines failing, document ingestion blocked | Platform team + data team + CTO notification | Immediate |
| L4 | Systemic failure affecting emission calculations downstream | All-hands engineering + incident commander + executive notification | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** PDF Extractor Health (`/d/pdf-extractor-service`)
- **Alerts:** `PDFOCREngineFailure`, `PDFLowOCRConfidence`, `PDFValidationErrorSpike`, `PDFHighExtractionFailureRateCritical`
- **Key metrics to watch:**
  - `gl_pdf_extractor_ocr_confidence` avg (should be > 0.7)
  - `gl_pdf_extractor_ocr_errors_total` (should be 0 or near-zero)
  - `gl_pdf_extractor_jobs_total{status="failed"}` rate (should be < 10%)
  - `gl_pdf_extractor_jobs_total{status="timeout"}` rate (should be near 0)
  - `gl_pdf_extractor_validation_results_total{severity="error"}` rate (should be < 20%)
  - `gl_pdf_extractor_low_confidence_extractions_total` (should be near 0)
  - `gl_pdf_extractor_job_duration_seconds` p99 (should be < 30s)

### Best Practices

1. **Maintain multiple OCR engines** as fallbacks for each other
2. **Set appropriate confidence thresholds** per document type (invoices may need higher than manifests)
3. **Create and maintain extraction templates** for each recurring document format
4. **Monitor document quality trends** and provide guidance to uploaders
5. **Test template changes** against a golden set of documents before deploying
6. **Set up alerts for API key expiration** for cloud OCR providers
7. **Review extraction accuracy weekly** and update templates as needed
8. **Maintain rate limit headroom** with cloud providers for burst processing

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `PDFOCREngineFailure` | Critical | OCR engine errors >10 in 5 minutes |
| `PDFLowOCRConfidence` | Warning | Average OCR confidence below 60% |
| `PDFValidationErrorSpike` | Warning | >20% of validations are errors |
| `PDFHighExtractionFailureRateWarning` | Warning | >10% extraction failure rate |
| `PDFHighExtractionFailureRateCritical` | Critical | >25% extraction failure rate |
| `PDFExtractionTimeout` | Warning | Extraction timeout rate elevated |
| `PDFExtractorServiceDown` | Critical | No healthy pods running |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Data Team
- **Review cadence:** Quarterly or after any P1 OCR extraction incident
- **Related runbooks:** [PDF Extractor Service Down](./pdf-extractor-service-down.md)
