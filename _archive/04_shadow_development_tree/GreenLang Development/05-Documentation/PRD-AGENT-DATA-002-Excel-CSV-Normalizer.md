# PRD: AGENT-DATA-002 - Excel/CSV Normalizer

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-002 |
| **Agent ID** | GL-DATA-X-016 |
| **Component** | Excel & CSV Data Normalizer Agent (Spreadsheet Parsing, Column Mapping, Data Type Normalization, Schema Validation) |
| **Category** | Data Intake Agent |
| **Priority** | P0 - Critical (primary structured data ingestion gateway for spreadsheet-based inputs) |
| **Status** | No Layer 1 Foundation - Full SDK Build Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires structured emissions data, activity data, and procurement
records from organizations. The dominant submission format is spreadsheets (Excel, CSV).
Without a production-grade Excel/CSV normalizer agent:

- **No Excel parsing**: .xlsx/.xls workbooks with multiple sheets remain unprocessable
- **No CSV ingestion**: Varied delimiters, encodings, and quote styles not handled
- **No column mapping**: Organization-specific headers not mapped to canonical GreenLang fields
- **No data type detection**: Numeric, date, currency, percentage, and unit fields not auto-typed
- **No unit normalization**: Mixed units (kg vs tonnes, kWh vs MWh) not standardized
- **No date normalization**: Varied date formats (US/EU/ISO) not canonicalized
- **No missing value handling**: Empty cells, N/A, nulls not consistently processed
- **No data quality scoring**: Imported data lacks completeness/accuracy indicators
- **No schema validation**: Spreadsheet data not validated against GreenLang schemas
- **No deduplication**: Duplicate rows across files not detected
- **No pivot/unpivot**: Wide-format data not reshaped to long-format for time series
- **No template system**: No reusable column mappings for recurring data sources
- **No batch processing**: Bulk file normalization not supported
- **No audit trail**: Data transformation operations not tracked for compliance

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**No Layer 1 agent exists.** There is no `greenlang/agents/data/excel_csv_normalizer_agent.py`.
This SDK will be built as a complete standalone package.

### 3.2 Related Agents
- **GL-DATA-X-001 (PDF Extractor)**: Handles document ingestion; Excel/CSV is complementary
- **GL-FOUND-X-003 (Unit Normalizer)**: Provides unit conversion capabilities; upstream dependency
- **GL-FOUND-X-002 (Schema Compiler)**: Provides schema validation; upstream dependency

### 3.3 Layer 1 Tests
None (no Layer 1 exists).

## 4. Identified Gaps

### Gap 1: No SDK Package
No `greenlang/excel_normalizer/` package for spreadsheet processing.

### Gap 2: No Prometheus Metrics
No `greenlang/excel_normalizer/metrics.py` following the standard 12-metric pattern.

### Gap 3: No Service Setup Facade
No `configure_excel_normalizer(app)` / `get_excel_normalizer(app)` pattern.

### Gap 4: No Excel Parser
No production Excel (.xlsx/.xls) parsing with multi-sheet support.

### Gap 5: No CSV Parser
No CSV parsing with delimiter/encoding auto-detection.

### Gap 6: No Column Mapper
No intelligent column header mapping to canonical GreenLang field names.

### Gap 7: No Data Type Detector
No automatic detection and conversion of data types.

### Gap 8: No REST API Router
No `greenlang/excel_normalizer/api/router.py` with FastAPI endpoints.

### Gap 9: No K8s Deployment Manifests
No `deployment/kubernetes/excel-normalizer-service/` manifests.

### Gap 10: No Database Migration
No `V032__excel_normalizer_service.sql` for persistent processing storage.

### Gap 11: No Monitoring
No Grafana dashboard or alert rules.

### Gap 12: No CI/CD Pipeline
No `.github/workflows/excel-normalizer-ci.yml`.

### Gap 13: No Operational Runbooks
No `docs/runbooks/` for Excel normalizer operations.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/excel_normalizer/
  __init__.py                  # Public API exports
  config.py                    # ExcelNormalizerConfig with GL_EXCEL_NORMALIZER_ env prefix
  models.py                    # Pydantic v2 models (enums + SDK models)
  excel_parser.py              # ExcelParser: .xlsx/.xls workbook parsing, multi-sheet support
  csv_parser.py                # CSVParser: delimiter/encoding auto-detection, streaming
  column_mapper.py             # ColumnMapper: header-to-canonical field mapping with fuzzy match
  data_type_detector.py        # DataTypeDetector: numeric/date/currency/percentage/unit detection
  schema_validator.py          # SchemaValidator: validate against GreenLang canonical schemas
  data_quality_scorer.py       # DataQualityScorer: completeness, accuracy, consistency scoring
  transform_engine.py          # TransformEngine: pivot/unpivot, dedup, merge, filter, aggregate
  provenance.py                # ProvenanceTracker: SHA-256 hash chain for transformation audit
  metrics.py                   # 12 Prometheus self-monitoring metrics
  setup.py                     # ExcelNormalizerService facade, configure/get
  api/
    __init__.py
    router.py                  # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V032)
```sql
CREATE SCHEMA excel_normalizer_service;
-- spreadsheet_files (file registry with metadata, format, sheet count)
-- sheet_metadata (per-sheet info: name, row/col counts, header row)
-- column_mappings (header-to-canonical field mappings with confidence)
-- normalization_jobs (hypertable - job tracking with status lifecycle)
-- normalized_records (transformed output records with quality scores)
-- mapping_templates (reusable column mapping templates per data source)
-- data_quality_reports (per-file quality assessment results)
-- validation_results (schema validation findings)
-- transform_operations (transformation operation log)
-- excel_audit_log (hypertable - processing audit trail)
```

### 5.3 Prometheus Self-Monitoring Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_excel_files_processed_total` | Counter | Total files processed by format (xlsx/xls/csv) |
| 2 | `gl_excel_processing_duration_seconds` | Histogram | File processing latency |
| 3 | `gl_excel_rows_normalized_total` | Counter | Total rows normalized |
| 4 | `gl_excel_columns_mapped_total` | Counter | Total columns mapped by match type |
| 5 | `gl_excel_mapping_confidence` | Histogram | Column mapping confidence distribution |
| 6 | `gl_excel_type_detections_total` | Counter | Data type detections by detected type |
| 7 | `gl_excel_validation_errors_total` | Counter | Schema validation errors |
| 8 | `gl_excel_quality_score` | Histogram | Data quality score distribution |
| 9 | `gl_excel_transforms_total` | Counter | Transform operations by type |
| 10 | `gl_excel_batch_jobs_total` | Counter | Batch processing jobs by status |
| 11 | `gl_excel_active_jobs` | Gauge | Currently active normalization jobs |
| 12 | `gl_excel_queue_size` | Gauge | Files waiting in queue |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/files/upload` | Upload and normalize a single file |
| POST | `/v1/files/batch` | Batch upload multiple files |
| GET | `/v1/files` | List processed files (with filters) |
| GET | `/v1/files/{file_id}` | Get file details and metadata |
| GET | `/v1/files/{file_id}/sheets` | Get sheet metadata for a workbook |
| GET | `/v1/files/{file_id}/preview` | Preview first N rows of normalized data |
| POST | `/v1/files/{file_id}/reprocess` | Reprocess a file with different settings |
| POST | `/v1/normalize` | Normalize raw data payload (inline) |
| POST | `/v1/columns/map` | Map columns to canonical fields |
| GET | `/v1/columns/canonical` | List available canonical field names |
| POST | `/v1/columns/detect-types` | Detect data types for columns |
| POST | `/v1/validate` | Validate data against a GreenLang schema |
| POST | `/v1/transform` | Apply transform operations (pivot, dedup, etc.) |
| GET | `/v1/quality/{file_id}` | Get data quality report for a file |
| POST | `/v1/templates` | Create/update mapping template |
| GET | `/v1/templates` | List mapping templates |
| GET | `/v1/templates/{template_id}` | Get template details |
| GET | `/v1/jobs` | List normalization jobs |
| GET | `/v1/statistics` | Get service statistics |
| GET | `/health` | Service health check |

### 5.5 Key Design Principles
1. **Zero-hallucination normalization**: All type detection uses deterministic heuristics (regex, statistical sampling), NO LLM for data values
2. **Multi-format support**: Excel (.xlsx, .xls), CSV (comma, semicolon, tab, pipe), TSV
3. **Encoding auto-detection**: UTF-8, UTF-16, Latin-1, Windows-1252, Shift-JIS via chardet/cchardet
4. **Fuzzy column mapping**: Levenshtein distance + synonym dictionaries for header matching
5. **Canonical field registry**: 200+ GreenLang canonical fields for emissions, energy, transport, waste
6. **Unit-aware normalization**: Delegates to AGENT-FOUND-003 Unit Normalizer for unit conversions
7. **Date format detection**: ISO 8601, US (MM/DD/YYYY), EU (DD/MM/YYYY), with ambiguity resolution
8. **Data quality scoring**: Completeness (% non-null), accuracy (type conformance), consistency (cross-field)
9. **Transform pipeline**: Composable operations (pivot, unpivot, dedup, merge, filter, aggregate, rename)
10. **Complete audit trail**: Every transformation logged with SHA-256 provenance chain

### 5.6 Canonical Field Categories
The column mapper maintains a registry of canonical GreenLang fields organized by category:

| Category | Example Fields | Count |
|----------|---------------|-------|
| Energy | electricity_kwh, natural_gas_therms, steam_mmbtu, chilled_water_kwh | 25+ |
| Transport | distance_km, fuel_liters, vehicle_type, trip_count | 20+ |
| Waste | waste_kg, waste_type, disposal_method, recycled_pct | 15+ |
| Water | water_m3, water_source, discharge_m3 | 10+ |
| Emissions | scope1_tco2e, scope2_tco2e, scope3_tco2e, emission_factor | 20+ |
| Procurement | spend_usd, supplier_name, material_type, quantity | 25+ |
| Facility | facility_name, facility_id, country, region, floor_area_m2 | 20+ |
| Temporal | reporting_year, reporting_month, start_date, end_date | 15+ |
| Organization | org_name, business_unit, department, cost_center | 15+ |
| Meta | data_source, data_quality, notes, reference_id | 15+ |

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/excel_normalizer/__init__.py` - Public API exports
2. Create `greenlang/excel_normalizer/config.py` - ExcelNormalizerConfig with GL_EXCEL_NORMALIZER_ env prefix
3. Create `greenlang/excel_normalizer/models.py` - Pydantic v2 models (all new, no Layer 1)
4. Create `greenlang/excel_normalizer/excel_parser.py` - ExcelParser with openpyxl/xlrd support
5. Create `greenlang/excel_normalizer/csv_parser.py` - CSVParser with auto-detection
6. Create `greenlang/excel_normalizer/column_mapper.py` - ColumnMapper with fuzzy matching
7. Create `greenlang/excel_normalizer/data_type_detector.py` - DataTypeDetector with heuristics
8. Create `greenlang/excel_normalizer/schema_validator.py` - SchemaValidator against GreenLang schemas
9. Create `greenlang/excel_normalizer/data_quality_scorer.py` - DataQualityScorer with metrics
10. Create `greenlang/excel_normalizer/transform_engine.py` - TransformEngine with operations
11. Create `greenlang/excel_normalizer/provenance.py` - ProvenanceTracker
12. Create `greenlang/excel_normalizer/metrics.py` - 12 Prometheus metrics
13. Create `greenlang/excel_normalizer/api/router.py` - FastAPI router with 20 endpoints
14. Create `greenlang/excel_normalizer/setup.py` - ExcelNormalizerService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V032__excel_normalizer_service.sql`
2. Create K8s manifests in `deployment/kubernetes/excel-normalizer-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1-16. Create unit, integration, and load tests (550+ tests target)

## 7. Success Criteria
- SDK module provides clean API for all spreadsheet normalization operations
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V032 database migration for persistent processing storage
- 20 REST API endpoints operational
- 550+ tests passing
- Excel (.xlsx/.xls) parsing with multi-sheet support
- CSV parsing with delimiter/encoding auto-detection
- Column mapping with >85% accuracy on common emission data headers
- Data type detection with >95% accuracy
- Date format normalization handling US/EU/ISO formats
- Data quality scoring (completeness, accuracy, consistency)
- Transform pipeline (pivot, unpivot, dedup, merge, filter, aggregate)
- Mapping template system for recurring data sources
- Complete audit trail for every transformation operation

## 8. Integration Points

### 8.1 Upstream Dependencies
- **AGENT-FOUND-003 Unit Normalizer**: Unit conversion (kg->tonnes, kWh->MWh, etc.)
- **AGENT-FOUND-002 Schema Compiler**: Schema validation for GreenLang canonical schemas
- **AGENT-FOUND-005 Citations**: Track data source provenance
- **AGENT-FOUND-006 Access Guard**: Authorization for data access
- **AGENT-FOUND-010 Observability**: Metrics, tracing, logging

### 8.2 Downstream Consumers
- **AGENT-DATA-001 PDF Extractor**: Complementary ingestion path (documents vs spreadsheets)
- **Scope 1/2/3 Calculation Agents**: Consume normalized activity data
- **Supply Chain Agents**: Consume normalized procurement/supplier data
- **Compliance Agents**: Consume normalized regulatory data
- **CSRD/CBAM Reporting**: Structured emissions data
- **Admin Dashboard**: File processing status visualization

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent file and normalization storage (V032 migration)
- **Redis**: Processing queue, mapping cache, template caching
- **S3**: Raw file storage (via INFRA-004)
- **Prometheus**: 12 self-monitoring metrics
- **Grafana**: Excel normalizer service dashboard
- **Alertmanager**: 15 alert rules
- **K8s**: Standard deployment with HPA
