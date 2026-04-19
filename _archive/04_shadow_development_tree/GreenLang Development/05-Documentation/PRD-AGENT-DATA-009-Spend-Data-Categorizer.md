# PRD: AGENT-DATA-009 - Spend Data Categorizer

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-009 |
| **Agent ID** | GL-DATA-SUP-002 |
| **Component** | Spend Data Categorizer Agent (Spend Ingestion, Taxonomy Classification, Scope 3 Mapping, Emission Factor Lookup, Spend Analytics, Category Rules, Reporting) |
| **Category** | Data Intake Agent (Procurement Analytics / Scope 3 Category Mapping) |
| **Priority** | P0 - Critical (required for Scope 3 spend-based emissions, SB 253, CSRD/ESRS E1, GHG Protocol Corporate Value Chain Standard) |
| **Status** | Layer 1 Partial (~3 files in agents/mrv/, agents/data/, agents/procurement/), SDK Build Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires automated spend data categorization for Scope 3 emissions calculation, procurement carbon footprint analysis, and regulatory compliance. Without a production-grade Spend Data Categorizer:

- **No multi-source spend ingestion**: Organizations cannot ingest spend data from diverse sources (ERP extracts, CSV/Excel files, API feeds, procurement platforms) into a unified pipeline
- **No taxonomy classification**: Spend line items lack mapping to standard classification systems (UNSPSC, NAICS, eCl@ss, ISIC, SIC, CPV, HS/CN codes)
- **No automated Scope 3 mapping**: Manual mapping of thousands of spend categories to GHG Protocol 15 Scope 3 categories is error-prone and inconsistent
- **No emission factor resolution**: Cannot automatically match spend categories to appropriate EEIO/EXIOBASE/DEFRA emission factors based on taxonomy codes
- **No spend analytics**: No hotspot identification, Pareto analysis, category concentration metrics, or year-over-year trending
- **No category rule management**: No configurable rule engine for organization-specific spend-to-category mappings or overrides
- **No multi-format reporting**: Cannot generate spend categorization reports for auditors, procurement teams, or sustainability officers
- **No confidence scoring**: Categorization quality is opaque with no transparency on mapping certainty
- **No deduplication/normalization**: Spend records from multiple sources contain duplicates, inconsistent vendor names, and currency mismatches
- **No audit trail**: Categorization decisions not tracked for GHG Protocol verification and regulatory compliance

## 3. Existing Implementation

### 3.1 Layer 1: Scope 3 Category Mapper
**File**: `greenlang/agents/mrv/scope3_category_mapper.py` (~652 lines)

- GL-MRV-X-005 agent with deterministic category mapping
- Scope3Category enum (15 GHG Protocol categories)
- DataSourceType enum (SPEND_DATA, PURCHASE_ORDER, BOM, SUPPLIER_DATA, TRAVEL_DATA, WASTE_DATA)
- CalculationApproach enum (SUPPLIER_SPECIFIC, HYBRID, AVERAGE_DATA, SPEND_BASED)
- NAICS_TO_CATEGORY mapping (35 NAICS prefixes -> Scope 3 categories)
- SPEND_KEYWORDS_TO_CATEGORY mapping (35 keywords -> Scope 3 categories)
- SpendRecord, PurchaseOrder, BOMItem models
- CategoryMappingResult with confidence scoring
- 3-tier mapping hierarchy: NAICS code (0.9) -> spend category (0.8) -> keyword (0.5)
- SHA-256 provenance hashing
- In-memory only, no persistence

### 3.2 Layer 1: ERP Connector Agent
**File**: `greenlang/agents/data/erp_connector_agent.py` (~837 lines)

- GL-DATA-X-004 agent with 10 ERP system support
- SpendCategory enum (11 high-level categories)
- SPEND_TO_SCOPE3_MAPPING (11 spend categories -> Scope 3 categories)
- DEFAULT_EMISSION_FACTORS (11 categories, kg CO2e/USD, EPA EEIO/EXIOBASE)
- Vendor/material mapping registries
- Spend-based emissions calculation
- Scope 3 summary aggregation
- Simulated ERP data for testing

### 3.3 Layer 1: Procurement Carbon Footprint Agent
**File**: `greenlang/agents/procurement/procurement_carbon_footprint.py` (~200+ lines)

- GL-PROC-X-004 agent with procurement footprint calculation
- SpendCategory enum (8 categories)
- SPEND_BASED_FACTORS (8 categories with EEIO/IEA sources)
- ProcurementItem model with supplier-specific emission factors
- Hybrid calculation: supplier-specific (quality 0.9) vs spend-based (quality 0.5)
- Emissions intensity (kgCO2e/USD) calculation
- Category and method breakdown summaries

## 4. Identified Gaps

| # | Gap | Impact | Severity |
|---|-----|--------|----------|
| 1 | No UNSPSC classification (83,000+ codes, 5-level hierarchy) | Cannot map to industry-standard procurement taxonomy | CRITICAL |
| 2 | No eCl@ss classification (45,000+ codes, EU standard) | No EU procurement taxonomy support | HIGH |
| 3 | No ISIC/SIC code mapping for international spend | Limited to NAICS (US-centric) | HIGH |
| 4 | No multi-source ingestion pipeline with deduplication | Cannot handle real-world multi-ERP environments | CRITICAL |
| 5 | No EEIO factor database (400+ sectors) | Only 11 coarse-grain emission factors available | CRITICAL |
| 6 | No EXIOBASE factor integration (163 products x 49 regions) | No regional emission factor resolution | HIGH |
| 7 | No currency normalization (multi-currency spend) | Inaccurate spend aggregation for global orgs | HIGH |
| 8 | No vendor name normalization/deduplication | Same vendor appears under multiple names | MEDIUM |
| 9 | No configurable rule engine for custom mappings | Organizations cannot override default categorizations | HIGH |
| 10 | No Pareto/hotspot analysis of spend emissions | Cannot identify highest-impact spend categories | HIGH |
| 11 | No year-over-year trending and variance analysis | Cannot track improvement or regression | MEDIUM |
| 12 | No multi-format reporting (audit, procurement, executive) | Manual report assembly required | MEDIUM |

## 5. Architecture

### 5.1 Engine Architecture

```
┌─────────────────────────────────────────────────────┐
│              SpendCategorizerService                  │
│                   (Facade)                            │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Engine 1: SpendIngestionEngine                       │
│  ├── Multi-source intake (ERP/CSV/Excel/API/Manual)  │
│  ├── Currency normalization (150+ currencies)         │
│  ├── Vendor name normalization                        │
│  ├── Deduplication (fuzzy + exact matching)           │
│  └── Batch processing with progress tracking          │
│                                                       │
│  Engine 2: TaxonomyClassifierEngine                   │
│  ├── UNSPSC classification (5-level hierarchy)        │
│  ├── NAICS mapping (2-6 digit codes)                  │
│  ├── eCl@ss mapping (EU standard)                     │
│  ├── ISIC Rev 4 mapping (UN standard)                 │
│  ├── SIC mapping (US legacy)                          │
│  ├── CPV mapping (EU public procurement)              │
│  ├── HS/CN code mapping (trade/customs)               │
│  └── Cross-taxonomy code translation                  │
│                                                       │
│  Engine 3: Scope3MapperEngine                         │
│  ├── GHG Protocol 15-category classification          │
│  ├── 3-tier confidence (NAICS>taxonomy>keyword)       │
│  ├── Capital vs operating expense detection            │
│  ├── Multi-category split allocation                   │
│  └── Upstream/downstream boundary assignment           │
│                                                       │
│  Engine 4: EmissionFactorEngine                       │
│  ├── EPA EEIO factor database (400+ sectors)          │
│  ├── EXIOBASE 3 factors (163 products x 49 regions)  │
│  ├── DEFRA conversion factors (UK standard)           │
│  ├── Factor selection hierarchy (specific>regional>global) │
│  └── Factor vintage management (year, source, version)│
│                                                       │
│  Engine 5: SpendAnalyticsEngine                       │
│  ├── Spend aggregation by category/vendor/period      │
│  ├── Pareto/ABC analysis (80/20 rule)                 │
│  ├── Emissions hotspot identification                  │
│  ├── YoY trending with variance analysis               │
│  ├── Category concentration (HHI index)                │
│  └── Supplier carbon intensity ranking                 │
│                                                       │
│  Engine 6: CategoryRuleEngine                         │
│  ├── Custom categorization rule CRUD                   │
│  ├── Priority-based rule evaluation                    │
│  ├── Regex + exact + fuzzy pattern matching            │
│  ├── Override management (org-specific mappings)       │
│  └── Rule effectiveness scoring                        │
│                                                       │
│  Engine 7: ReportingEngine                            │
│  ├── Categorization summary reports                    │
│  ├── Emissions by Scope 3 category                     │
│  ├── Audit trail reports                               │
│  ├── Procurement team reports                          │
│  └── Export (JSON, CSV, Markdown, HTML)                │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### 5.2 Data Flow

```
ERP/CSV/Excel/API → SpendIngestionEngine → normalized records
  → TaxonomyClassifierEngine → UNSPSC/NAICS/eCl@ss codes
  → Scope3MapperEngine → GHG Protocol Scope 3 categories
  → EmissionFactorEngine → emission estimates (tCO2e)
  → SpendAnalyticsEngine → aggregations/hotspots/trends
  → ReportingEngine → reports in multiple formats
```

### 5.3 Database Schema (V039)

**Schema**: `spend_categorizer_service`

| Table | Type | Description |
|-------|------|-------------|
| spend_records | regular | Ingested spend records with normalization |
| taxonomy_mappings | regular | UNSPSC/NAICS/eCl@ss/ISIC code mappings |
| scope3_classifications | regular | Scope 3 category assignments per record |
| emission_factors | regular | EEIO/EXIOBASE/DEFRA factor database |
| emission_calculations | regular | Per-record emission calculation results |
| category_rules | regular | Custom categorization rules |
| analytics_snapshots | regular | Periodic analytics aggregations |
| ingestion_batches | regular | Batch ingestion tracking |
| categorization_events | hypertable | Time-series categorization events (7d chunks) |
| emission_calculations_ts | hypertable | Time-series emission calc events (7d chunks) |
| analytics_events | hypertable | Time-series analytics events (7d chunks) |
| hourly_categorization_stats | continuous_agg | Hourly categorization statistics |
| hourly_emission_stats | continuous_agg | Hourly emission calculation statistics |

### 5.4 Prometheus Metrics (12)

1. `gl_spend_cat_records_ingested_total` (Counter) - Records ingested by source
2. `gl_spend_cat_records_classified_total` (Counter) - Records classified by taxonomy
3. `gl_spend_cat_scope3_mapped_total` (Counter) - Records mapped to Scope 3 categories
4. `gl_spend_cat_emissions_calculated_total` (Counter) - Emission calculations performed
5. `gl_spend_cat_rules_evaluated_total` (Counter) - Category rules evaluated
6. `gl_spend_cat_reports_generated_total` (Counter) - Reports generated by format
7. `gl_spend_cat_classification_confidence` (Histogram) - Classification confidence distribution
8. `gl_spend_cat_processing_duration_seconds` (Histogram) - Processing duration by operation
9. `gl_spend_cat_active_batches` (Gauge) - Active ingestion batches
10. `gl_spend_cat_total_spend_usd` (Gauge) - Total spend amount tracked
11. `gl_spend_cat_processing_errors_total` (Counter) - Processing errors by type
12. `gl_spend_cat_emission_factor_lookups_total` (Counter) - Emission factor lookups by source

### 5.5 REST API (20 endpoints)

| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/ingest | Ingest spend records (batch) |
| POST | /v1/ingest/file | Ingest from CSV/Excel file |
| GET | /v1/records | List spend records with filters |
| GET | /v1/records/{id} | Get single spend record |
| POST | /v1/classify | Classify spend records (taxonomy) |
| POST | /v1/classify/batch | Batch taxonomy classification |
| POST | /v1/map-scope3 | Map records to Scope 3 categories |
| POST | /v1/map-scope3/batch | Batch Scope 3 mapping |
| POST | /v1/calculate-emissions | Calculate emissions for records |
| POST | /v1/calculate-emissions/batch | Batch emissions calculation |
| GET | /v1/emission-factors | List available emission factors |
| GET | /v1/emission-factors/{taxonomy_code} | Get factor for taxonomy code |
| POST | /v1/rules | Create categorization rule |
| GET | /v1/rules | List categorization rules |
| PUT | /v1/rules/{id} | Update categorization rule |
| DELETE | /v1/rules/{id} | Delete categorization rule |
| GET | /v1/analytics | Get spend analytics (aggregations) |
| GET | /v1/analytics/hotspots | Get emission hotspots |
| POST | /v1/reports | Generate categorization report |
| GET | /health | Health check endpoint |

### 5.6 Configuration

**Prefix**: `GL_SPEND_CAT_`

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| GL_SPEND_CAT_DATABASE_URL | str | "" | PostgreSQL connection URL |
| GL_SPEND_CAT_REDIS_URL | str | "" | Redis connection URL |
| GL_SPEND_CAT_LOG_LEVEL | str | "INFO" | Logging level |
| GL_SPEND_CAT_DEFAULT_CURRENCY | str | "USD" | Default currency for normalization |
| GL_SPEND_CAT_DEFAULT_TAXONOMY | str | "unspsc" | Default taxonomy system |
| GL_SPEND_CAT_EEIO_VERSION | str | "2024" | EPA EEIO factor year |
| GL_SPEND_CAT_EXIOBASE_VERSION | str | "3.8.2" | EXIOBASE version |
| GL_SPEND_CAT_DEFRA_VERSION | str | "2025" | DEFRA factor year |
| GL_SPEND_CAT_MIN_CONFIDENCE | float | 0.3 | Minimum classification confidence |
| GL_SPEND_CAT_BATCH_SIZE | int | 1000 | Ingestion batch size |
| GL_SPEND_CAT_MAX_RECORDS | int | 100000 | Maximum records per batch |
| GL_SPEND_CAT_DEDUP_THRESHOLD | float | 0.85 | Fuzzy dedup similarity threshold |
| GL_SPEND_CAT_VENDOR_NORMALIZATION | bool | True | Enable vendor name normalization |
| GL_SPEND_CAT_CACHE_TTL | int | 3600 | Emission factor cache TTL (seconds) |
| GL_SPEND_CAT_ENABLE_EXIOBASE | bool | True | Enable EXIOBASE regional factors |
| GL_SPEND_CAT_ENABLE_DEFRA | bool | True | Enable DEFRA factors |
| GL_SPEND_CAT_RATE_LIMIT_RPM | int | 120 | Rate limit (requests per minute) |

## 6. Completion Plan

### Phase 1: Foundation (config.py, models.py)
- Thread-safe configuration with GL_SPEND_CAT_ env prefix
- Pydantic v2 models with Layer 1 re-exports
- 10+ new enums, 15+ SDK models, 7+ request models

### Phase 2: Core Engines (7 engines)
- SpendIngestionEngine: multi-source intake, normalization, deduplication
- TaxonomyClassifierEngine: UNSPSC/NAICS/eCl@ss/ISIC classification
- Scope3MapperEngine: GHG Protocol 15-category mapping with confidence
- EmissionFactorEngine: EEIO/EXIOBASE/DEFRA factor database and lookup
- SpendAnalyticsEngine: aggregation, Pareto, hotspot, trending
- CategoryRuleEngine: custom rules, priority evaluation, overrides
- ReportingEngine: multi-format report generation

### Phase 3: Infrastructure (provenance, metrics, setup, API)
- SHA-256 chain-hashed provenance tracking
- 12 Prometheus metrics with graceful fallback
- SpendCategorizerService facade
- FastAPI router with 20 endpoints

### Phase 4: Database & Deployment
- V039 migration (10 tables, 3 hypertables, 2 continuous aggregates)
- 10 K8s manifests with security hardening
- GitHub Actions CI/CD (7-job pipeline)

### Phase 5: Testing
- 15 test files, 1000+ test functions
- Config, models, provenance, metrics, all 7 engines, setup, router
- Thread safety, edge cases, error handling

## 7. Success Criteria

| Criteria | Target |
|----------|--------|
| Taxonomy systems supported | >= 7 (UNSPSC, NAICS, eCl@ss, ISIC, SIC, CPV, HS) |
| EEIO emission factors | >= 400 sectors |
| EXIOBASE regional factors | >= 100 products x 20 regions |
| DEFRA conversion factors | >= 50 categories |
| Classification confidence | >= 0.3 minimum, >= 0.7 average |
| Scope 3 categories covered | 15/15 GHG Protocol categories |
| Test coverage | >= 85% line coverage |
| Test functions | >= 1000 |
| Prometheus metrics | 12 |
| API endpoints | 20 |
| Database tables | 10 + 3 hypertables + 2 continuous aggregates |
| SHA-256 provenance | On every mutation |

## 8. Integration Points

### Layer 1 Re-exports
- `scope3_category_mapper.py`: Scope3Category, DataSourceType, CalculationApproach, SpendRecord, CategoryMappingResult, NAICS_TO_CATEGORY, SPEND_KEYWORDS_TO_CATEGORY
- `erp_connector_agent.py`: ERPSystem, SpendCategory, TransactionType, SPEND_TO_SCOPE3_MAPPING, DEFAULT_EMISSION_FACTORS
- `procurement_carbon_footprint.py`: CalculationMethod, ProcurementItem, EmissionCalculation

### Upstream Dependencies
- AGENT-DATA-003 (ERP Connector): Provides raw spend data from ERP systems
- AGENT-DATA-002 (Excel/CSV Normalizer): Provides normalized file-based spend data
- AGENT-DATA-004 (API Gateway): Routes queries to spend categorizer

### Downstream Consumers
- Scope 3 calculation engines (Category 1, 2, 4, 5, 6, etc.)
- CSRD/ESRS E1 climate reporting
- SB 253 disclosure pipeline
- CDP Climate Change reporting
- Procurement sustainability dashboards
- CSDDD due diligence data
