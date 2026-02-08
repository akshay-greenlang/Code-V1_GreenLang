# PRD: AGENT-DATA-001 - PDF & Invoice Extractor

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-001 |
| **Agent ID** | GL-DATA-X-001 |
| **Component** | Document Ingestion & OCR Agent (PDF Parsing, Invoice Extraction, Manifest Processing) |
| **Category** | Data Intake Agent |
| **Priority** | P0 - Critical (primary data ingestion gateway for all document-based inputs) |
| **Status** | Layer 1 Complete (~847 lines), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires structured data from diverse document sources for emission
calculations, supply chain tracking, and compliance reporting. Organizations submit PDFs,
invoices, shipping manifests, utility bills, weight tickets, purchase orders, and receipts
that must be accurately digitized into structured fields. Without a production-grade
document ingestion agent:

- **No automated PDF parsing**: Manual data entry from thousands of documents
- **No invoice field extraction**: Vendor, amounts, line items not automatically captured
- **No manifest processing**: Shipping data (weights, origins, carriers) entered manually
- **No utility bill extraction**: Energy consumption data not auto-captured for Scope 2
- **No OCR integration**: Scanned documents remain unprocessable
- **No confidence scoring**: Extracted fields lack quality indicators
- **No document classification**: Documents must be manually categorized
- **No cross-field validation**: Inconsistencies (total != subtotal + tax) go undetected
- **No audit trail**: Document processing operations not tracked for compliance

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/data/document_ingestion_agent.py` (~847 lines)
- `DocumentIngestionAgent` (BaseAgent subclass, AGENT_ID: GL-DATA-X-001)
- 3 enums: DocumentType(9: pdf/invoice/manifest/bill_of_lading/weight_ticket/utility_bill/receipt/purchase_order/unknown), ExtractionStatus(4: success/partial/failed/low_confidence), OCREngine(5: tesseract/azure_vision/aws_textract/google_vision/simulated)
- 8 Pydantic models: BoundingBox, ExtractedField, LineItem, InvoiceData, ManifestData, UtilityBillData, DocumentIngestionInput, DocumentIngestionOutput
- Regex-based field extraction patterns for invoices (6 fields), manifests (4 fields), utility bills (3 fields)
- Document classification via keyword scoring
- Line item extraction via regex
- Invoice total validation (subtotal + tax = total, line items sum = subtotal)
- Simulated OCR for testing
- SHA-256 provenance hashing
- Convenience methods: ingest_document, extract_invoice_data, extract_manifest_data, classify_document_type
- In-memory processing (no database persistence)

### 3.2 Layer 1 Tests
None found.

## 4. Identified Gaps

### Gap 1: No Integration Module
No `greenlang/pdf_extractor/` package providing a clean SDK.

### Gap 2: No Prometheus Metrics
No `greenlang/pdf_extractor/metrics.py` following the standard 12-metric pattern.

### Gap 3: No Service Setup Facade
No `configure_pdf_extractor(app)` / `get_pdf_extractor(app)` pattern.

### Gap 4: No Real OCR Integration
Layer 1 only has simulated OCR; no production OCR engine adapters.

### Gap 5: No REST API Router
No `greenlang/pdf_extractor/api/router.py` with FastAPI endpoints.

### Gap 6: No K8s Deployment Manifests
No `deployment/kubernetes/pdf-extractor-service/` manifests.

### Gap 7: No Database Migration
No `V031__pdf_extractor_service.sql` for persistent document/extraction storage.

### Gap 8: No Monitoring
No Grafana dashboard or alert rules.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/pdf-extractor-ci.yml`.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for PDF extractor operations.

### Gap 11: No Template System
No configurable extraction templates for different invoice/document formats.

### Gap 12: No Batch Processing
No bulk document ingestion capability.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/pdf_extractor/
  __init__.py                  # Public API exports
  config.py                    # PDFExtractorConfig with GL_PDF_EXTRACTOR_ env prefix
  models.py                    # Pydantic v2 models (re-export + enhance from Layer 1)
  document_parser.py           # DocumentParser: PDF parsing, page extraction, text extraction
  ocr_engine.py                # OCREngineAdapter: Tesseract, AWS Textract, Azure Vision, Google Vision
  field_extractor.py           # FieldExtractor: pattern-based extraction with confidence scoring
  invoice_processor.py         # InvoiceProcessor: invoice-specific extraction and validation
  manifest_processor.py        # ManifestProcessor: shipping manifest/BOL extraction
  document_classifier.py       # DocumentClassifier: keyword/pattern-based classification
  validation_engine.py         # ValidationEngine: cross-field validation, total verification
  provenance.py                # ProvenanceTracker: SHA-256 hash chain for document audit
  metrics.py                   # 12 Prometheus self-monitoring metrics
  setup.py                     # PDFExtractorService facade, configure/get
  api/
    __init__.py
    router.py                  # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V031)
```sql
CREATE SCHEMA pdf_extractor_service;
-- documents (document registry with metadata and classification)
-- document_pages (individual page content and OCR results)
-- extraction_jobs (hypertable - extraction job tracking)
-- extracted_fields (individual field extraction results with confidence)
-- invoice_extractions (structured invoice data)
-- manifest_extractions (structured manifest data)
-- utility_bill_extractions (structured utility bill data)
-- extraction_templates (configurable extraction templates)
-- validation_results (cross-field validation results)
-- pdf_audit_log (hypertable - document processing audit trail)
```

### 5.3 Prometheus Self-Monitoring Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_pdf_documents_processed_total` | Counter | Total documents processed by type |
| 2 | `gl_pdf_processing_duration_seconds` | Histogram | Document processing latency |
| 3 | `gl_pdf_pages_extracted_total` | Counter | Total pages extracted |
| 4 | `gl_pdf_fields_extracted_total` | Counter | Total fields extracted by status |
| 5 | `gl_pdf_extraction_confidence` | Histogram | Extraction confidence distribution |
| 6 | `gl_pdf_ocr_operations_total` | Counter | OCR operations by engine |
| 7 | `gl_pdf_validation_errors_total` | Counter | Validation errors detected |
| 8 | `gl_pdf_classification_total` | Counter | Document classification by type |
| 9 | `gl_pdf_line_items_extracted_total` | Counter | Total line items extracted |
| 10 | `gl_pdf_batch_jobs_total` | Counter | Batch processing jobs |
| 11 | `gl_pdf_active_jobs` | Gauge | Currently active extraction jobs |
| 12 | `gl_pdf_queue_size` | Gauge | Documents waiting in queue |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/documents/ingest` | Ingest a single document |
| POST | `/v1/documents/batch` | Batch ingest multiple documents |
| GET | `/v1/documents` | List documents (with filters) |
| GET | `/v1/documents/{document_id}` | Get document details |
| GET | `/v1/documents/{document_id}/fields` | Get extracted fields |
| GET | `/v1/documents/{document_id}/pages` | Get document pages |
| POST | `/v1/documents/{document_id}/reprocess` | Reprocess a document |
| POST | `/v1/documents/classify` | Classify document type |
| POST | `/v1/invoices/extract` | Extract invoice data |
| GET | `/v1/invoices/{document_id}` | Get invoice extraction result |
| POST | `/v1/manifests/extract` | Extract manifest data |
| GET | `/v1/manifests/{document_id}` | Get manifest extraction result |
| POST | `/v1/utility-bills/extract` | Extract utility bill data |
| GET | `/v1/utility-bills/{document_id}` | Get utility bill extraction |
| POST | `/v1/templates` | Create/update extraction template |
| GET | `/v1/templates` | List extraction templates |
| POST | `/v1/validate/{document_id}` | Run validation on extracted data |
| GET | `/v1/jobs` | List extraction jobs |
| GET | `/v1/statistics` | Get extraction statistics |
| GET | `/health` | Service health check |

### 5.5 Key Design Principles
1. **Zero-hallucination extraction**: All field extraction uses deterministic regex/pattern matching, NO LLM for numeric values
2. **Multi-engine OCR**: Pluggable OCR backends (Tesseract, AWS Textract, Azure Vision, Google Vision) with fallback
3. **Confidence scoring**: Every extracted field has a computed confidence score (0.0-1.0)
4. **Cross-field validation**: Automatic verification (totals match, dates consistent, etc.)
5. **Document classification**: Keyword/pattern-based classification with scoring
6. **Template system**: Configurable extraction templates for different document formats
7. **Batch processing**: Bulk document ingestion with job tracking
8. **Bounding box tracking**: OCR region coordinates for field provenance
9. **Multi-format support**: PDF, scanned images, digital documents
10. **Complete audit trail**: Every document operation logged with SHA-256 provenance chain

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/pdf_extractor/__init__.py` - Public API exports
2. Create `greenlang/pdf_extractor/config.py` - PDFExtractorConfig with GL_PDF_EXTRACTOR_ env prefix
3. Create `greenlang/pdf_extractor/models.py` - Pydantic v2 models
4. Create `greenlang/pdf_extractor/document_parser.py` - DocumentParser with PDF text extraction
5. Create `greenlang/pdf_extractor/ocr_engine.py` - OCREngineAdapter with multi-engine support
6. Create `greenlang/pdf_extractor/field_extractor.py` - FieldExtractor with pattern matching
7. Create `greenlang/pdf_extractor/invoice_processor.py` - InvoiceProcessor with line item extraction
8. Create `greenlang/pdf_extractor/manifest_processor.py` - ManifestProcessor with BOL fields
9. Create `greenlang/pdf_extractor/document_classifier.py` - DocumentClassifier with scoring
10. Create `greenlang/pdf_extractor/validation_engine.py` - ValidationEngine with cross-field checks
11. Create `greenlang/pdf_extractor/provenance.py` - ProvenanceTracker
12. Create `greenlang/pdf_extractor/metrics.py` - 12 Prometheus metrics
13. Create `greenlang/pdf_extractor/api/router.py` - FastAPI router with 20 endpoints
14. Create `greenlang/pdf_extractor/setup.py` - PDFExtractorService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V031__pdf_extractor_service.sql`
2. Create K8s manifests in `deployment/kubernetes/pdf-extractor-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1-16. Create unit, integration, and load tests (500+ tests target)

## 7. Success Criteria
- Integration module provides clean SDK for all document ingestion operations
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V031 database migration for persistent document storage
- 20 REST API endpoints operational
- 500+ tests passing
- Invoice extraction with >85% field accuracy on standard formats
- Manifest/BOL extraction operational
- Utility bill extraction operational
- Document classification with >90% accuracy
- Cross-field validation catching 100% of total mismatches
- Complete audit trail for every document operation

## 8. Integration Points

### 8.1 Upstream Dependencies
- **AGENT-FOUND-003 Unit Normalizer**: Normalize extracted units (kg, lbs, kWh, etc.)
- **AGENT-FOUND-005 Citations**: Track document sources
- **AGENT-FOUND-006 Access Guard**: Authorization for document access
- **AGENT-FOUND-010 Observability**: Metrics, tracing, logging

### 8.2 Downstream Consumers
- **Scope 1/2/3 Calculation Agents**: Consume extracted utility bills, invoices
- **Supply Chain Agents**: Consume extracted manifests, BOLs
- **Compliance Agents**: Consume extracted regulatory documents
- **CSRD/CBAM Reporting**: Document evidence packages
- **Admin Dashboard**: Document processing status visualization

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent document and extraction storage (V031 migration)
- **Redis**: Document queue, extraction result caching
- **S3**: Raw document file storage (via INFRA-004)
- **Prometheus**: 12 self-monitoring metrics
- **Grafana**: PDF extractor service dashboard
- **Alertmanager**: 15 alert rules
- **K8s**: Standard deployment with HPA
