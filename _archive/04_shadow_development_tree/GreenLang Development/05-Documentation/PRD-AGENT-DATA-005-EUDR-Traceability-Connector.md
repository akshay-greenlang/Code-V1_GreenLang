# PRD: AGENT-DATA-005 - EUDR Traceability Connector

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-005 |
| **Agent ID** | GL-DATA-EUDR-001 |
| **Component** | EUDR Traceability Connector Agent (Plot Registry, Chain of Custody, Due Diligence, Risk Assessment, Commodity Classification, Compliance Verification, EU System Integration) |
| **Category** | Data Intake Agent |
| **Priority** | P0 - Critical / TIER 1 EXTREME URGENCY (EU Deforestation Regulation deadline December 30, 2025) |
| **Status** | Layer 1 Complete (~1,180 lines), Full Production Build Required |
| **Regulation** | Regulation (EU) 2023/1115 - EU Deforestation Regulation (EUDR) |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) requires operators and traders
placing seven covered commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood) and
their derived products on the EU market to demonstrate that products are:

1. **Deforestation-free** (no deforestation after December 31, 2020 cutoff)
2. **Legally produced** (compliant with producing country laws)
3. **Covered by a due diligence statement** submitted to the EU Information System

Without a production-grade EUDR Traceability Connector Agent:

- **No plot-level traceability**: Production plot geolocations not systematically registered
- **No chain of custody**: Product movement through supply chain not tracked
- **No due diligence statements**: Article 4 DDS not auto-generated for EU submission
- **No risk assessment engine**: Country/commodity/supplier risk not scored per Article 10
- **No commodity classification**: CN/HS codes not mapped to EUDR-covered products
- **No compliance verification**: Article 3/9/10/11 requirements not systematically checked
- **No EU Information System integration**: No automated submission to EU regulatory system
- **No batch/lot traceability**: Mass balance and segregation not tracked
- **No supplier declaration management**: Supplier compliance declarations not collected
- **No deforestation monitoring**: Satellite-derived forest change data not integrated
- **No audit trail**: Traceability operations not tracked for regulatory inspection

## 3. Regulatory Requirements

### 3.1 EUDR Articles Addressed
| Article | Requirement | Implementation |
|---------|-------------|----------------|
| **Art. 3** | Prohibition on non-compliant products | ComplianceVerifier engine |
| **Art. 4** | Due diligence obligation | DueDiligenceEngine |
| **Art. 9** | Geolocation of production plots | PlotRegistryEngine |
| **Art. 10** | Risk assessment & mitigation | RiskAssessmentEngine |
| **Art. 11** | Risk mitigation measures | RiskAssessmentEngine |
| **Art. 12** | EU Information System submission | EUSystemConnector |
| **Art. 29** | Country benchmarking (risk levels) | RiskAssessmentEngine |
| **Art. 31** | Record keeping (5 years) | Database persistence |

### 3.2 Covered Commodities (Article 1)
| Commodity | Example Derived Products | Key CN Codes |
|-----------|-------------------------|--------------|
| Cattle | Beef, leather, gelatin | 0102, 0201-0202, 4101-4115 |
| Cocoa | Chocolate, cocoa butter, powder | 1801-1806 |
| Coffee | Roasted, instant, extracts | 0901, 2101 |
| Oil Palm | Palm oil, kernel oil, oleochemicals | 1511, 1513, 3823 |
| Rubber | Natural rubber, tyres, latex | 4001, 4005-4017 |
| Soya | Soybean oil, meal, flour, lecithin | 1201, 1507, 2304 |
| Wood | Timber, furniture, paper, charcoal | 4401-4421, 4701-4813, 9401-9403 |

### 3.3 Key Dates
| Date | Milestone |
|------|-----------|
| June 29, 2023 | Regulation entered into force |
| December 31, 2020 | Deforestation cutoff date |
| December 30, 2024 | Enforcement for large operators (extended to Dec 30, 2025) |
| June 30, 2025 | Enforcement for SMEs (extended to Jun 30, 2026) |
| December 30, 2025 | **CURRENT DEADLINE** for large operators |

## 4. Existing Implementation

### 4.1 Layer 1: Foundation Code
**File**: `greenlang/data/supply_chain/eudr/traceability.py` (~1,180 lines)
- `EUDRTraceabilityManager` class with in-memory storage
- 5 enums: `EUDRCommodity`(19 values), `RiskLevel`(4), `ComplianceStatus`(5), `LandUseType`(6)
- 3 dataclasses: `PlotRecord`, `ChainOfCustodyRecord`, `DueDiligenceStatement`
- Operations: register_plot, record_transfer, generate_due_diligence_statement
- Risk assessment (country-based), compliance summary, EU system export
- SHA-256 integrity hashing per record
- In-memory storage only (no database persistence)

**File**: `greenlang/extensions/regulations/eudr/models.py` (~513 lines)
- `GeolocationData`, `ProductionPlot`, `EUDRProduct`, `RiskAssessment`
- `SupplierDeclaration`, `DueDiligenceStatement` (Article 4 format)
- Zero-hallucination risk scoring with deterministic weights

**File**: `greenlang/extensions/regulations/eudr/risk_engine.py`
- Country risk scoring
- Commodity risk assessment

**File**: `greenlang/extensions/regulations/eudr/validators.py`
- Geolocation validation
- Coordinate precision checks

### 4.2 Related GL-Agent-Factory Agents
- `gl_eudr_001_supply_chain_mapper` - Supply chain mapping agent
- `gl_eudr_002_geolocation_collector` - Geolocation data collection agent

### 4.3 Golden Tests
- `tests/golden/eudr_compliance/` - 5 golden test fixtures

## 5. Identified Gaps

### Gap 1: No Integration Module
No `greenlang/eudr_traceability/` package providing a clean SDK.

### Gap 2: No Prometheus Metrics
No `greenlang/eudr_traceability/metrics.py` following the standard 12-metric pattern.

### Gap 3: No Service Setup Facade
No `configure_eudr_traceability(app)` / `get_eudr_traceability(app)` pattern.

### Gap 4: No Production Plot Registry
Layer 1 has basic plot registration but no persistent registry with polygon support,
satellite data integration, and deforestation monitoring links.

### Gap 5: No Chain of Custody Engine
Layer 1 has basic custody records but no mass balance tracking, batch splitting/merging,
segregation models, or multi-tier supply chain graph.

### Gap 6: No Due Diligence Engine
Layer 1 generates basic DDS but no EU Information System format compliance, digital
signing, XML/JSON export per EU specification, or submission workflow.

### Gap 7: No Commodity Classification Engine
No systematic CN/HS code classifier with derived product mapping and TARIC integration.

### Gap 8: No REST API Router
No `greenlang/eudr_traceability/api/router.py` with FastAPI endpoints.

### Gap 9: No K8s Deployment Manifests
No `deployment/kubernetes/eudr-traceability-service/` manifests.

### Gap 10: No Database Migration
No `V034__eudr_traceability_service.sql` for persistent traceability storage.

### Gap 11: No Monitoring
No Grafana dashboard or alert rules for EUDR traceability operations.

### Gap 12: No CI/CD Pipeline
No `.github/workflows/eudr-traceability-ci.yml`.

### Gap 13: No Supplier Declaration Management
No systematic collection and validation of supplier compliance declarations.

### Gap 14: No Batch/Lot Traceability
No mass balance, segregation, or identity-preserved tracking models.

## 6. Architecture (Final State)

### 6.1 Integration Module
```
greenlang/eudr_traceability/
  __init__.py                       # Public API exports
  config.py                         # EUDRTraceabilityConfig with GL_EUDR_TRACEABILITY_ env prefix
  models.py                         # Pydantic v2 models (re-export + enhance from Layer 1)
  plot_registry.py                  # PlotRegistryEngine: plot geolocation, polygon mgmt, monitoring
  chain_of_custody.py               # ChainOfCustodyEngine: custody tracking, mass balance, segregation
  due_diligence.py                  # DueDiligenceEngine: DDS generation, EU format, signing
  risk_assessment.py                # RiskAssessmentEngine: country/commodity/supplier scoring
  commodity_classifier.py           # CommodityClassifier: CN/HS code mapping, derived products
  compliance_verifier.py            # ComplianceVerifier: Article 3/9/10/11 systematic checks
  eu_system_connector.py            # EUSystemConnector: EU Information System integration
  provenance.py                     # ProvenanceTracker: SHA-256 hash chain for audit
  metrics.py                        # 12 Prometheus self-monitoring metrics
  setup.py                          # EUDRTraceabilityService facade, configure/get
  api/
    __init__.py
    router.py                       # FastAPI router (20 endpoints)
```

### 6.2 Database Schema (V034)
```sql
CREATE SCHEMA eudr_traceability_service;
-- production_plots (plot registry with geolocation and polygon data)
-- plot_monitoring (hypertable - satellite monitoring and deforestation checks)
-- chain_of_custody (custody transfer records with batch/lot tracking)
-- custody_batches (batch splitting, merging, and mass balance)
-- due_diligence_statements (DDS with EU Information System format)
-- dds_submissions (hypertable - EU system submission tracking)
-- risk_assessments (country/commodity/supplier risk scores)
-- commodity_classifications (CN/HS code mappings)
-- supplier_declarations (supplier compliance declarations)
-- compliance_checks (hypertable - Article compliance verification results)
```

### 6.3 Prometheus Self-Monitoring Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_plots_registered_total` | Counter | Total production plots registered |
| 2 | `gl_eudr_custody_transfers_total` | Counter | Total chain of custody transfers recorded |
| 3 | `gl_eudr_dds_generated_total` | Counter | Due diligence statements generated |
| 4 | `gl_eudr_risk_assessments_total` | Counter | Risk assessments performed by risk level |
| 5 | `gl_eudr_commodity_classifications_total` | Counter | Commodity classifications by type |
| 6 | `gl_eudr_compliance_checks_total` | Counter | Compliance checks by article and result |
| 7 | `gl_eudr_eu_submissions_total` | Counter | EU Information System submissions by status |
| 8 | `gl_eudr_supplier_declarations_total` | Counter | Supplier declarations processed |
| 9 | `gl_eudr_processing_duration_seconds` | Histogram | Processing operation latency |
| 10 | `gl_eudr_errors_total` | Counter | Errors by operation type |
| 11 | `gl_eudr_active_plots` | Gauge | Currently active/monitored production plots |
| 12 | `gl_eudr_pending_verifications` | Gauge | Plots/products pending compliance verification |

### 6.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/plots` | Register a production plot with geolocation |
| GET | `/v1/plots` | List production plots (with filters) |
| GET | `/v1/plots/{plot_id}` | Get plot details with monitoring history |
| PUT | `/v1/plots/{plot_id}/compliance` | Update plot compliance status |
| DELETE | `/v1/plots/{plot_id}` | Archive a production plot |
| POST | `/v1/custody/transfers` | Record a chain of custody transfer |
| GET | `/v1/custody/transfers` | Query custody transfer records |
| GET | `/v1/custody/trace/{batch_id}` | Trace product batch to origin plots |
| POST | `/v1/custody/batches/split` | Split a batch for downstream processing |
| POST | `/v1/dds` | Generate a due diligence statement |
| GET | `/v1/dds` | List due diligence statements |
| GET | `/v1/dds/{dds_id}` | Get DDS details |
| POST | `/v1/dds/{dds_id}/submit` | Submit DDS to EU Information System |
| POST | `/v1/risk/assess` | Perform risk assessment for product/plot |
| GET | `/v1/risk/countries` | Get country risk classifications |
| POST | `/v1/commodities/classify` | Classify product by CN/HS code |
| POST | `/v1/suppliers/declarations` | Register supplier compliance declaration |
| GET | `/v1/suppliers/declarations` | List supplier declarations |
| GET | `/v1/compliance/summary` | Get compliance summary report |
| GET | `/health` | Service health check |

### 6.5 Key Design Principles
1. **Zero-hallucination**: All risk scoring uses deterministic weights and rules, NO LLM for compliance determinations
2. **Regulation-faithful**: Every field and check maps to a specific EUDR Article requirement
3. **Deforestation cutoff**: Hard-coded December 31, 2020 cutoff date per EUDR Article 3
4. **7 commodities + derived**: Full coverage of all EUDR-covered commodities and derived products
5. **Geolocation precision**: WGS84 coordinates with polygon support for plots > 4 hectares per Article 9
6. **Country benchmarking**: Risk levels per Article 29 (Low/Standard/High) with 18+ high-risk countries
7. **Mass balance tracking**: Batch splitting, merging, and segregation for complex supply chains
8. **EU System format**: DDS export in EU Information System compatible format
9. **5-year record retention**: Article 31 compliance with persistent storage
10. **Complete audit trail**: Every traceability operation logged with SHA-256 provenance chains

### 6.6 Engine Specifications

#### Engine 1: PlotRegistryEngine
- Register production plots with WGS84 geolocation
- Support point coordinates (plots <= 4 ha) and polygon boundaries (plots > 4 ha)
- Index plots by commodity, country, producer, risk level
- Track deforestation-free status with cutoff date validation
- Link to satellite monitoring data for deforestation detection
- Store certification information (FSC, RSPO, Rainforest Alliance, etc.)
- Validate coordinate ranges and polygon topology
- Bulk import from CSV/GeoJSON

#### Engine 2: ChainOfCustodyEngine
- Record custody transfers between supply chain actors
- Track batch/lot numbers with splitting and merging
- Support three custody models: Identity Preserved, Segregated, Mass Balance
- Maintain quantity reconciliation (input vs output mass balance)
- Link every transfer back to origin production plots
- Track transport documents, customs declarations, CN codes
- Generate traceability graphs for product origin visualization
- Verify custody chain completeness

#### Engine 3: DueDiligenceEngine
- Generate DDS per Article 4 with all required fields
- Populate operator information, product details, geolocation references
- Aggregate risk assessment results
- Include deforestation-free and legal compliance declarations
- Format output for EU Information System submission
- Support digital signing of statements
- Track DDS lifecycle (draft -> submitted -> verified -> expired)
- Generate DDS for import, export, and domestic placement

#### Engine 4: RiskAssessmentEngine
- Score country risk per Article 29 benchmarking (Low/Standard/High)
- Score commodity risk based on deforestation association
- Score supplier risk based on compliance history and certifications
- Score traceability risk based on data completeness
- Calculate overall risk with deterministic weighted formula
- Recommend risk mitigation measures per Article 11
- Maintain country risk classification database
- Support enhanced due diligence triggers for high-risk assessments

#### Engine 5: CommodityClassifier
- Map products to EUDR-covered commodities (7 primary + derived)
- Classify by CN codes (Combined Nomenclature) per EU regulation
- Classify by HS codes (Harmonized System) for international trade
- Determine if product is primary commodity or derived product
- Map derived products to their primary commodity
- Validate CN/HS codes against EUDR Annex I
- Support TARIC integration for EU customs classification
- Track product composition for mixed-commodity products

#### Engine 6: ComplianceVerifier
- Systematically verify Article 3 requirements (deforestation-free + legal)
- Verify Article 9 geolocation completeness (coordinates, polygons)
- Verify Article 10 due diligence completeness (risk assessment performed)
- Verify Article 11 risk mitigation (measures applied for high-risk)
- Generate compliance score per product/operator
- Identify compliance gaps with specific remediation guidance
- Track compliance status changes over time
- Support batch compliance verification

#### Engine 7: EUSystemConnector
- Format data for EU Information System API
- Handle submission workflow (prepare -> validate -> submit -> confirm)
- Parse EU system responses and reference numbers
- Track submission status and retry failed submissions
- Support bulk DDS submission
- Handle EU system authentication and certificates
- Maintain submission audit log
- Support sandbox/test and production EU system environments

## 7. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/eudr_traceability/__init__.py` - Public API exports
2. Create `greenlang/eudr_traceability/config.py` - EUDRTraceabilityConfig
3. Create `greenlang/eudr_traceability/models.py` - Pydantic v2 models
4. Create `greenlang/eudr_traceability/plot_registry.py` - PlotRegistryEngine
5. Create `greenlang/eudr_traceability/chain_of_custody.py` - ChainOfCustodyEngine
6. Create `greenlang/eudr_traceability/due_diligence.py` - DueDiligenceEngine
7. Create `greenlang/eudr_traceability/risk_assessment.py` - RiskAssessmentEngine
8. Create `greenlang/eudr_traceability/commodity_classifier.py` - CommodityClassifier
9. Create `greenlang/eudr_traceability/compliance_verifier.py` - ComplianceVerifier
10. Create `greenlang/eudr_traceability/eu_system_connector.py` - EUSystemConnector
11. Create `greenlang/eudr_traceability/provenance.py` - ProvenanceTracker
12. Create `greenlang/eudr_traceability/metrics.py` - 12 Prometheus metrics
13. Create `greenlang/eudr_traceability/api/router.py` - FastAPI router (20 endpoints)
14. Create `greenlang/eudr_traceability/setup.py` - EUDRTraceabilityService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V034__eudr_traceability_service.sql`
2. Create K8s manifests in `deployment/kubernetes/eudr-traceability-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1. Create tests for PlotRegistryEngine (80+ tests)
2. Create tests for ChainOfCustodyEngine (80+ tests)
3. Create tests for DueDiligenceEngine (70+ tests)
4. Create tests for RiskAssessmentEngine (70+ tests)
5. Create tests for CommodityClassifier (60+ tests)
6. Create tests for ComplianceVerifier (70+ tests)
7. Create tests for EUSystemConnector (50+ tests)
8. Create tests for API router (40+ tests)
9. Create tests for models and config (30+ tests)
10. Create tests for provenance tracking (30+ tests)
11. Create integration tests (20+ tests)
12. Create load/performance tests (10+ tests)

## 8. Success Criteria
- Integration module provides clean SDK for all EUDR traceability operations
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V034 database migration for persistent traceability storage
- 20 REST API endpoints operational
- 600+ tests passing
- Plot registry with WGS84 geolocation and polygon support
- Chain of custody with mass balance and segregation models
- DDS generation in EU Information System format
- Risk assessment with deterministic scoring per Article 29
- CN/HS code commodity classification
- Systematic Article 3/9/10/11 compliance verification
- 5-year record retention per Article 31
- Complete audit trail for every traceability operation

## 9. Integration Points

### 9.1 Upstream Dependencies
- **AGENT-FOUND-002 Schema Compiler**: Validate EUDR data structures
- **AGENT-FOUND-003 Unit Normalizer**: Normalize quantities and coordinates
- **AGENT-FOUND-005 Citations**: Track regulatory source references
- **AGENT-FOUND-006 Access Guard**: Authorization for traceability data
- **AGENT-FOUND-008 Reproducibility**: Deterministic risk scoring verification
- **AGENT-FOUND-010 Observability**: Metrics, tracing, logging
- **AGENT-DATA-001 PDF Extractor**: Process supplier declarations and certificates
- **AGENT-DATA-003 ERP Connector**: Procurement data for supply chain mapping

### 9.2 Downstream Consumers
- **GL-EUDR-APP**: Full EUDR compliance application
- **GL-EUDR-001 Supply Chain Mapper**: Supply chain mapping agent
- **GL-EUDR-002 Geolocation Collector**: Geolocation data collection
- **GL-EUDR-003 Commodity Traceability Agent**: Product traceability
- **GL-EUDR-014 Traceability Audit Agent**: Audit trail verification
- **GL-CSDDD-APP**: Supply chain due diligence application
- **Satellite ML agents**: Deforestation monitoring integration

### 9.3 Infrastructure Integration
- **PostgreSQL**: Persistent plot, custody, DDS, risk storage (V034 migration)
- **TimescaleDB**: Time-series monitoring data and submission tracking
- **Redis**: Plot lookup cache, risk score cache, classification cache
- **S3**: Polygon GeoJSON storage, certificate documents, DDS exports
- **Prometheus**: 12 self-monitoring metrics
- **Grafana**: EUDR traceability service dashboard
- **Alertmanager**: 15 alert rules (compliance gaps, submission failures, risk escalations)
- **K8s**: Standard deployment with HPA

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| EU Information System API not finalized | High | Build adapter pattern for easy API changes |
| Country benchmarking list changes | Medium | Database-driven country risk, hot-reload |
| Regulation amendments | Medium | Configuration-driven compliance rules |
| Polygon data volume | Medium | PostGIS extension, spatial indexing |
| Supply chain complexity | High | Support 3 custody models (IP/SEG/MB) |
| Cross-commodity mixing | Medium | Composition tracking for mixed products |
