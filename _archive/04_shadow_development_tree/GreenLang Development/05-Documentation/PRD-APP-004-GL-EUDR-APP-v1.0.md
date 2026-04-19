# PRD: APP-004 -- GL-EUDR-APP v1.0

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-APP-004 |
| Application | GL-EUDR-APP v1.0 |
| Priority | P0 (CRITICAL -- TIER 1 EXTREME URGENCY) |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence + GL-EUDR-PM |
| Date | 2026-02-26 |
| Deadline | December 30, 2025 (ALREADY PAST - retroactive compliance build) |
| Base | Existing AGENT-DATA-005 (EUDR Traceability) + AGENT-DATA-007 (Deforestation Satellite) |

---

## 1. Overview

### 1.1 Purpose
Build the GL-EUDR-APP v1.0 as a full-stack EUDR compliance platform leveraging 2 existing production-ready backend agents and adding:
1. **5-Agent Pipeline Orchestration** -- Wire AGENT-DATA-005 + AGENT-DATA-007 into a 5-stage pipeline with 3 new agents (Supplier Intake, Document Verification, DDS Reporting)
2. **React Frontend Dashboard** -- Full web UI for supplier management, plot registry, risk monitoring, DDS tracking, compliance dashboard
3. **Backend API Layer** -- Unified FastAPI application integrating all 5 agents with comprehensive REST API
4. **Database Layer** -- Application-specific tables for DDS lifecycle, document verification, pipeline orchestration
5. **Configuration & Setup** -- Application config, service facade, pipeline orchestrator

### 1.2 Regulatory Context -- EU Regulation 2023/1115
- **Scope**: Prohibits placing/making available products linked to post-Dec 2020 deforestation
- **Commodities**: 7 regulated (cattle, cocoa, coffee, palm oil, rubber, soy, wood) + derived products
- **Obligation**: Operators must submit Due Diligence Statements (DDS) via EU Information System
- **Penalties**: Up to 4% of annual EU turnover, market access denial, public naming
- **Companies affected**: 100,000+ across EU importers and exporters

### 1.3 Current State Assessment

| Component | Status | What Exists |
|-----------|--------|-------------|
| AGENT-DATA-005 (EUDR Traceability) | BUILT | 15 files, 8,075 lines, 7,309 test lines |
| AGENT-DATA-007 (Deforestation Satellite) | BUILT | 15 files, 9,195 lines, 9,757 test lines |
| Regulatory Module | BUILT | 4 files, 1,320 lines (extensions/regulations/eudr/) |
| Supply Chain Module | BUILT | 2 files, 1,208 lines (data/supply_chain/eudr/) |
| DB Migrations | BUILT | V034 (1,349 lines) + V037 (1,335 lines) |
| K8s Manifests | BUILT | 20 files (dev/staging/prod) |
| CI/CD Workflows | BUILT | 2 GitHub Actions workflows |
| **Frontend Dashboard** | **NOT BUILT** | No React/TypeScript code |
| **Application Pipeline** | **NOT BUILT** | No orchestration layer |
| **Supplier Intake Agent** | **NOT BUILT** | Only planned in docs |
| **Document Verification Agent** | **NOT BUILT** | Only planned in docs |
| **DDS Reporting Agent** | **NOT BUILT** | Only planned in docs |
| **Unified API Layer** | **NOT BUILT** | Individual agent APIs exist |
| **Application Config** | **NOT BUILT** | No app-level config |

### 1.4 Technical Context
- **Existing Backend**: AGENT-DATA-005 (FastAPI, 20 endpoints) + AGENT-DATA-007 (FastAPI, 20 endpoints)
- **Frontend Stack**: React 18 + TypeScript + Material-UI 5 + Redux Toolkit + Recharts + Vite + Leaflet (maps)
- **Backend Stack**: FastAPI + Python 3.11+ + PostGIS + NumPy + Pydantic v2
- **Spatial**: Shapely, GDAL, GeoJSON, WGS84
- **ML**: Satellite imagery classification, NDVI change detection

---

## 2. Application Components

### 2.1 Backend: Pipeline Orchestrator
Wires all 5 agents into a DAG-based processing pipeline.

#### 2.1.1 Pipeline Engine
- 5-stage pipeline: Intake → GeoValidation → DeforestationRisk → DocVerification → DDSReporting
- Async execution with status tracking
- Error handling with retry logic per stage
- Pipeline state persistence in PostgreSQL
- Parallel processing for independent stages
- Event-driven with callback hooks

#### 2.1.2 Supplier Intake Engine
- ERP data normalization (SAP, Oracle, CSV/Excel)
- Supplier master data management
- Procurement record ingestion
- Commodity classification integration (7 EUDR commodities)
- Data quality scoring
- Batch import with validation

#### 2.1.3 Document Verification Engine
- Document type classification (certificates, permits, land titles, invoices, transport docs)
- OCR text extraction (simulated for v1.0)
- Compliance rule matching against EUDR articles
- Document authenticity scoring
- Evidence chain linking to DDS
- Gap analysis for missing documents

#### 2.1.4 DDS Reporting Engine
- Due Diligence Statement generation per EU format
- Reference number generation (EUDR-{country}-{year}-{seq})
- DDS lifecycle management (draft → review → submitted → accepted/rejected)
- Bulk DDS generation for multiple shipments
- EU Information System submission simulation
- Amendment/correction workflow
- Annual reporting aggregation

### 2.2 Frontend: React Dashboard

#### 2.2.1 Dashboard Page
- KPI cards: Total suppliers, Compliant %, DDS submitted, High-risk plots
- Compliance trend chart (monthly)
- Risk distribution pie chart
- Recent DDS status timeline
- Alert feed for compliance issues

#### 2.2.2 Supplier Management Page
- Supplier table with search/filter/sort
- Supplier detail view (profile, commodities, plots, compliance status)
- Add/edit supplier form
- Bulk import via CSV/Excel
- Supplier risk scoring display
- Compliance status badges (compliant/pending/non-compliant/under-review)

#### 2.2.3 Plot Registry Page
- Interactive map (Leaflet/OpenStreetMap) with plot polygons
- Plot table with coordinates, area, commodity, risk level
- Add plot with polygon drawing tool
- Validate plot coordinates
- Overlap detection visualization
- Satellite imagery overlay (NDVI heatmap)
- Deforestation alert markers

#### 2.2.4 Risk Assessment Page
- Risk heatmap (country × commodity matrix)
- Plot-level risk detail cards
- Satellite evidence viewer (before/after comparison)
- Risk trend timeline
- Deforestation alert feed
- Risk mitigation recommendations

#### 2.2.5 DDS Management Page
- DDS table with lifecycle status (draft/review/submitted/accepted/rejected)
- DDS detail view with all sections
- Generate new DDS wizard (multi-step form)
- DDS validation results
- Submit to EU system button
- Download DDS (PDF/XML)
- Amendment workflow

#### 2.2.6 Document Library Page
- Document upload with drag-and-drop
- Document classification auto-detect
- Verification status indicators
- Link documents to suppliers/plots/DDS
- Document search and filter

#### 2.2.7 Pipeline Monitor Page
- Pipeline execution status (5-stage progress bar)
- Active/completed/failed pipeline runs
- Per-stage metrics and timing
- Error log viewer
- Manual re-run triggers

### 2.3 Backend: Unified API Layer

#### 2.3.1 Application API Endpoints
- **Suppliers** (8 endpoints): CRUD, search, bulk import, compliance status, risk summary
- **Plots** (7 endpoints): CRUD, validate, overlaps, bulk import, satellite status
- **DDS** (9 endpoints): CRUD, generate, validate, submit, download, amend, list, bulk generate
- **Documents** (6 endpoints): CRUD, upload, verify, link, search
- **Pipeline** (5 endpoints): start, status, history, retry, cancel
- **Risk** (5 endpoints): assess, heatmap, alerts, trends, mitigations
- **Dashboard** (3 endpoints): metrics, trends, alerts
- **Settings** (3 endpoints): get, update, defaults

### 2.4 Database Migration
- V078: GL-EUDR-APP application tables (suppliers app-level, DDS lifecycle, documents, pipeline runs, settings)
- Extends existing V034 (traceability) and V037 (satellite) with application-specific joins

---

## 3. File Structure

```
applications/GL-EUDR-APP/EUDR-Compliance-Platform/
    config/
        eudr_config.yaml                    (~150 lines)
    services/
        __init__.py                          (~50 lines)
        config.py                            (~200 lines)
        models.py                            (~800 lines)
        pipeline_orchestrator.py             (~600 lines)
        supplier_intake_engine.py            (~500 lines)
        document_verification_engine.py      (~500 lines)
        dds_reporting_engine.py              (~700 lines)
        risk_aggregator.py                   (~400 lines)
        setup.py                             (~300 lines)
        api/
            __init__.py                      (~30 lines)
            supplier_routes.py               (~500 lines)
            plot_routes.py                   (~400 lines)
            dds_routes.py                    (~600 lines)
            document_routes.py               (~400 lines)
            pipeline_routes.py               (~300 lines)
            risk_routes.py                   (~350 lines)
            dashboard_routes.py              (~250 lines)
            settings_routes.py               (~200 lines)
    frontend/
        package.json                         (~40 lines)
        tsconfig.json                        (~25 lines)
        vite.config.ts                       (~20 lines)
        index.html                           (~15 lines)
        src/
            main.tsx                          (~20 lines)
            App.tsx                           (~80 lines)
            types/index.ts                    (~300 lines)
            services/api.ts                   (~400 lines)
            store/
                index.ts                      (~30 lines)
                hooks.ts                      (~10 lines)
                slices/
                    dashboardSlice.ts         (~150 lines)
                    supplierSlice.ts          (~250 lines)
                    plotSlice.ts              (~200 lines)
                    ddsSlice.ts               (~250 lines)
                    documentSlice.ts          (~200 lines)
                    pipelineSlice.ts          (~150 lines)
                    riskSlice.ts              (~150 lines)
            components/
                layout/
                    Sidebar.tsx               (~150 lines)
                    Header.tsx                (~80 lines)
                    Layout.tsx                (~50 lines)
                common/
                    StatCard.tsx              (~60 lines)
                    StatusBadge.tsx           (~50 lines)
                    DataTable.tsx             (~200 lines)
                    LoadingSpinner.tsx        (~30 lines)
                    ConfirmDialog.tsx         (~60 lines)
                maps/
                    PlotMap.tsx               (~400 lines)
                    RiskHeatmap.tsx           (~300 lines)
                    SatelliteOverlay.tsx      (~250 lines)
                suppliers/
                    SupplierTable.tsx         (~300 lines)
                    SupplierDetail.tsx        (~350 lines)
                    SupplierForm.tsx          (~300 lines)
                dds/
                    DDSTable.tsx              (~250 lines)
                    DDSDetail.tsx             (~350 lines)
                    DDSWizard.tsx             (~500 lines)
                    DDSValidation.tsx         (~200 lines)
                risk/
                    RiskMatrix.tsx            (~300 lines)
                    RiskTimeline.tsx          (~200 lines)
                    EvidenceViewer.tsx        (~250 lines)
                pipeline/
                    PipelineProgress.tsx      (~200 lines)
                    PipelineHistory.tsx       (~200 lines)
            pages/
                Dashboard.tsx                 (~250 lines)
                SupplierManagement.tsx        (~200 lines)
                PlotRegistry.tsx              (~250 lines)
                RiskAssessment.tsx            (~200 lines)
                DDSManagement.tsx             (~250 lines)
                DocumentLibrary.tsx           (~200 lines)
                PipelineMonitor.tsx           (~200 lines)
            utils/
                formatters.ts                 (~100 lines)
                validators.ts                 (~80 lines)
    tests/
        __init__.py                          (~5 lines)
        test_models.py                       (~400 lines)
        test_pipeline_orchestrator.py        (~500 lines)
        test_supplier_intake.py              (~400 lines)
        test_document_verification.py        (~400 lines)
        test_dds_reporting.py                (~500 lines)
        test_risk_aggregator.py              (~300 lines)
        test_supplier_routes.py              (~400 lines)
        test_dds_routes.py                   (~500 lines)
        test_pipeline_routes.py              (~300 lines)
```

---

## 4. Development Tasks (Parallel Build Plan)

### Task Group A: Backend Core (Agent 1)
- A1: Build config.py, models.py (domain models, enums, Pydantic schemas)
- A2: Build pipeline_orchestrator.py (5-stage DAG, async execution)
- A3: Build supplier_intake_engine.py (ERP normalization, batch import)
- A4: Build document_verification_engine.py (classification, OCR stub, compliance matching)
- A5: Build dds_reporting_engine.py (DDS generation, lifecycle, EU submission sim)
- A6: Build risk_aggregator.py (combine plot/supplier/country risk)
- A7: Build setup.py (service facade)
- A8: Build eudr_config.yaml

### Task Group B: Backend API Layer (Agent 2)
- B1: Build api/__init__.py, supplier_routes.py (8 endpoints)
- B2: Build plot_routes.py (7 endpoints)
- B3: Build dds_routes.py (9 endpoints)
- B4: Build document_routes.py (6 endpoints)
- B5: Build pipeline_routes.py (5 endpoints)
- B6: Build risk_routes.py, dashboard_routes.py, settings_routes.py

### Task Group C: Frontend Core + Layout + Store (Agent 3)
- C1: Build package.json, tsconfig.json, vite.config.ts, index.html, main.tsx
- C2: Build types/index.ts (TypeScript interfaces)
- C3: Build services/api.ts (API client)
- C4: Build store (index, hooks, 7 slices)
- C5: Build layout components (Sidebar, Header, Layout)
- C6: Build common components (StatCard, StatusBadge, DataTable, LoadingSpinner, ConfirmDialog)
- C7: Build App.tsx with routing

### Task Group D: Frontend Pages + Domain Components (Agent 4)
- D1: Build maps/ components (PlotMap, RiskHeatmap, SatelliteOverlay)
- D2: Build suppliers/ components (SupplierTable, SupplierDetail, SupplierForm)
- D3: Build dds/ components (DDSTable, DDSDetail, DDSWizard, DDSValidation)
- D4: Build risk/ components (RiskMatrix, RiskTimeline, EvidenceViewer)
- D5: Build pipeline/ components (PipelineProgress, PipelineHistory)
- D6: Build all 7 pages
- D7: Build utils (formatters, validators)

### Task Group E: Tests + DB Migration (Agent 5)
- E1: Build all 10 backend test files
- E2: Build V078 database migration SQL

---

## 5. Acceptance Criteria

1. 5-stage pipeline orchestrator with async execution and status tracking
2. Supplier intake with ERP normalization and batch import
3. Document verification with classification and compliance matching
4. DDS generation with EU format, lifecycle management, and bulk processing
5. Risk aggregation combining satellite, country, and supplier risk
6. Full React dashboard with 7 pages
7. Interactive map with plot polygons, satellite overlay, risk markers
8. DDS wizard with multi-step form and validation
9. 46+ REST API endpoints across 8 route modules
10. Comprehensive test coverage (10 test files, 300+ tests)
11. Database migration for application tables
12. Integration with existing AGENT-DATA-005 and AGENT-DATA-007
