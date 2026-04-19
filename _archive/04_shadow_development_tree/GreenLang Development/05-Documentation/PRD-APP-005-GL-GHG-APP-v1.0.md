# PRD: APP-005 -- GL-GHG-APP v1.0

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-APP-005 |
| Application | GL-GHG-APP v1.0 (Beta) |
| Priority | P1 (High) |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-02-26 |
| Standard | GHG Protocol Corporate Accounting and Reporting Standard (Revised Edition, 2015) |
| Base | 28 MRV Agents (662K+ lines), Scope 3 Category Mapper, Audit Trail & Lineage |

---

## 1. Overview

### 1.1 Purpose
Build GL-GHG-APP v1.0 as the definitive GHG Protocol Corporate Standard reporting platform. This application orchestrates all 28 existing MRV agents into a unified corporate GHG inventory management system covering:
1. **Inventory Management** -- Organizational/operational boundary, consolidation approach, base year
2. **Scope 1 Aggregation** -- 8 source categories (stationary, mobile, process, fugitive, refrigerants, land use, waste treatment, agricultural)
3. **Scope 2 Dual Reporting** -- Location-based + Market-based with reconciliation
4. **Scope 3 Full Coverage** -- All 15 categories with materiality screening
5. **Corporate Reporting** -- GHG inventory report, intensity metrics, trend analysis, verification workflow
6. **Dashboard & Analytics** -- Executive dashboard, scope breakdown, trends, benchmarking, target tracking
7. **Multi-Format Export** -- PDF report, Excel workbook, JSON, CSV, XBRL

### 1.2 GHG Protocol Requirements Mapping

| GHG Protocol Chapter | Requirement | GL-GHG-APP Feature |
|----------------------|-------------|---------------------|
| Ch 3 - GHG Accounting Principles | Relevance, completeness, consistency, transparency, accuracy | Quality scoring engine, completeness checker |
| Ch 4 - Organizational Boundaries | Equity share, financial control, operational control | Boundary manager with 3 consolidation approaches |
| Ch 5 - Operational Boundaries | Scope 1, 2, 3 classification | Scope classifier + 28 MRV agents |
| Ch 6 - Tracking Over Time | Base year, recalculation policy, structural changes | Base year manager with recalculation triggers |
| Ch 7 - Identifying/Calculating | 6 GHGs (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3) | Multi-gas tracking in all MRV agents |
| Ch 8 - Collecting Data | Activity data quality, emission factors | Data quality scoring (5-dim DQI) |
| Ch 9 - Scope 2 | Location + market-based, dual reporting | MRV-009 through MRV-013 |
| Ch 10 - Scope 3 | 15 upstream/downstream categories | MRV-014 through MRV-028 + Category Mapper |
| Ch 11 - Uncertainty | Quantitative assessment | Monte Carlo aggregation engine |
| Ch 12 - Reporting | Mandatory + optional disclosures | Report generator with compliance checker |

### 1.3 Technical Context
- **Backend**: FastAPI + Python 3.11+ + Pydantic v2
- **Frontend**: React 18 + TypeScript + Material-UI 5 + Redux Toolkit + Recharts + Vite
- **MRV Agents**: 28 production-ready agents at `greenlang/` with individual FastAPI routers
- **Support Agents**: Scope 3 Category Mapper, Audit Trail & Lineage, Consolidation Rollup, Inventory Boundary
- **Database**: PostgreSQL + TimescaleDB (V051-V081 already deployed)

---

## 2. Application Components

### 2.1 Backend: Inventory Management Engine
- Organizational boundary definition (entities, facilities, operations)
- Consolidation approach selection (operational control / financial control / equity share)
- Equity share percentage tracking per entity
- Operational boundary classification (Scope 1, 2, 3)
- Reporting period management (calendar year, fiscal year)
- Multi-entity hierarchy (parent → subsidiary → facility → source)

### 2.2 Backend: Base Year Manager
- Base year selection with justification
- Structural change detection (mergers, acquisitions, divestitures)
- Organic growth vs. structural change classification
- Significance threshold (configurable, default 5%)
- Automatic recalculation triggers
- Base year emissions lock with audit trail

### 2.3 Backend: Scope Aggregation Engine
- Aggregates results from all 28 MRV agents
- Scope 1 rollup: 8 source categories → total direct emissions
- Scope 2 dual: location-based total + market-based total
- Scope 3 rollup: 15 categories → total indirect emissions
- Grand total: Scope 1 + Scope 2 (location OR market) + Scope 3
- Per-gas breakdown (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
- Per-facility breakdown
- Per-country/region breakdown
- Biogenic CO2 separate reporting (GHG Protocol requirement)

### 2.4 Backend: Intensity Calculator
- Revenue intensity (tCO2e per $M revenue)
- Employee intensity (tCO2e per FTE)
- Production intensity (tCO2e per unit produced)
- Floor area intensity (tCO2e per m²)
- Custom intensity metrics
- Year-over-year comparison
- Peer benchmarking data

### 2.5 Backend: Uncertainty Engine
- Aggregated Monte Carlo across all scopes (propagation of uncertainty)
- Per-scope confidence intervals (90%, 95%, 99%)
- Data quality contribution to uncertainty
- Sensitivity ranking across all parameters
- Convergence assessment

### 2.6 Backend: Completeness Checker
- Scope 1 source completeness (all categories covered?)
- Scope 2 dual reporting present?
- Scope 3 materiality screening (which categories relevant?)
- Data gap identification with severity
- Exclusion justifications required
- GHG Protocol mandatory disclosure checklist

### 2.7 Backend: Report Generator
- GHG Protocol mandatory disclosures (15 items)
- Optional disclosures (10 items)
- Scope 3 supplementary reporting
- Executive summary with key metrics
- Trend analysis (5-year)
- Export: PDF, Excel workbook (multi-sheet), JSON, CSV

### 2.8 Backend: Verification Workflow
- Internal review stages (data owner → reviewer → approver)
- External verifier assignment
- Materiality threshold for assurance
- Finding tracking (material, immaterial, observations)
- Verification statement generation
- Limited vs. reasonable assurance levels

### 2.9 Frontend: Dashboard
- Executive KPI cards (Total emissions, Scope 1/2/3 split, YoY change, intensity)
- Scope breakdown donut chart
- Monthly/quarterly trend line chart
- Scope 3 category waterfall chart
- Geographic emissions heatmap
- Target progress gauge
- Data quality scorecard

### 2.10 Frontend: Inventory Setup
- Boundary wizard (org structure, consolidation approach, scopes)
- Entity/facility hierarchy builder
- Base year configuration
- Reporting period selector
- Emission source mapping

### 2.11 Frontend: Scope 1 Page
- Source category breakdown (stationary, mobile, process, fugitive, etc.)
- Per-facility emissions table
- Per-gas breakdown chart
- Monthly trend
- Data entry / import for each category

### 2.12 Frontend: Scope 2 Page
- Dual reporting comparison (location vs. market)
- Instrument tracking (RECs, PPAs, green tariffs)
- Grid emission factor sources
- Residual mix analysis
- Reconciliation waterfall

### 2.13 Frontend: Scope 3 Page
- 15-category overview with materiality indicators
- Category detail panels
- Calculation method per category
- Data quality per category
- Completeness percentage

### 2.14 Frontend: Reports Page
- Report builder with section selection
- Preview mode
- Export (PDF, Excel, JSON)
- Report history
- Verification status

### 2.15 Frontend: Targets Page
- Science-Based Targets (SBTi) alignment
- Absolute vs. intensity targets
- Progress tracking with forecast
- Gap-to-target analysis

---

## 3. File Structure

```
applications/GL-GHG-APP/GHG-Corporate-Platform/
    config/
        ghg_config.yaml                      (~200 lines)
    services/
        __init__.py                           (~60 lines)
        config.py                             (~250 lines)
        models.py                             (~900 lines)
        inventory_manager.py                  (~700 lines)
        base_year_manager.py                  (~500 lines)
        scope_aggregator.py                   (~800 lines)
        intensity_calculator.py               (~400 lines)
        uncertainty_engine.py                 (~500 lines)
        completeness_checker.py               (~500 lines)
        report_generator.py                   (~700 lines)
        verification_workflow.py              (~500 lines)
        target_tracker.py                     (~400 lines)
        setup.py                              (~350 lines)
        api/
            __init__.py                       (~30 lines)
            inventory_routes.py               (~500 lines)
            scope1_routes.py                  (~400 lines)
            scope2_routes.py                  (~400 lines)
            scope3_routes.py                  (~500 lines)
            reporting_routes.py               (~400 lines)
            dashboard_routes.py               (~300 lines)
            verification_routes.py            (~350 lines)
            target_routes.py                  (~300 lines)
            settings_routes.py                (~200 lines)
    frontend/
        package.json, tsconfig.json, vite.config.ts, index.html
        src/
            main.tsx, App.tsx
            types/index.ts                    (~400 lines)
            services/api.ts                   (~500 lines)
            store/ (index, hooks, 8 slices)
            components/
                layout/ (Sidebar, Header, Layout)
                common/ (StatCard, StatusBadge, DataTable, LoadingSpinner)
                dashboard/ (ScopeDonut, TrendChart, WaterfallChart, IntensityCard, QualityScore, TargetGauge)
                inventory/ (BoundaryWizard, EntityTree, BaseYearConfig)
                scope1/ (SourceBreakdown, FacilityTable, GasBreakdown)
                scope2/ (DualReportingComparison, InstrumentTracker, ReconciliationWaterfall)
                scope3/ (CategoryOverview, CategoryDetail, MaterialityMatrix)
                reports/ (ReportBuilder, ReportPreview, ExportDialog)
                targets/ (TargetProgress, GapAnalysis, SBTiAlignment)
                verification/ (VerificationTimeline, FindingTracker)
            pages/
                Dashboard.tsx
                InventorySetup.tsx
                Scope1.tsx
                Scope2.tsx
                Scope3.tsx
                Reports.tsx
                Targets.tsx
                Verification.tsx
            utils/ (formatters, validators)
    tests/
        test_models.py
        test_inventory_manager.py
        test_scope_aggregator.py
        test_intensity_calculator.py
        test_uncertainty_engine.py
        test_completeness_checker.py
        test_report_generator.py
        test_verification_workflow.py
        test_api_routes.py
        test_target_tracker.py
```

---

## 4. Development Tasks (Parallel Build Plan)

### Task Group A: Backend Core Engines (Agent 1)
- A1: config.py, models.py (enums, domain models, request/response schemas)
- A2: inventory_manager.py (boundary, consolidation, entity hierarchy)
- A3: base_year_manager.py (selection, recalculation, structural changes)
- A4: scope_aggregator.py (Scope 1/2/3 rollup from 28 MRV agents)
- A5: intensity_calculator.py (revenue, employee, production, custom)
- A6: uncertainty_engine.py (Monte Carlo propagation across scopes)
- A7: completeness_checker.py (GHG Protocol mandatory disclosures)
- A8: report_generator.py (multi-format export)
- A9: verification_workflow.py (internal review, external assurance)
- A10: target_tracker.py (SBTi, absolute, intensity targets)
- A11: setup.py + __init__.py + ghg_config.yaml

### Task Group B: Backend API Layer (Agent 2)
- B1: api/__init__.py, inventory_routes.py (8 endpoints)
- B2: scope1_routes.py, scope2_routes.py, scope3_routes.py
- B3: reporting_routes.py, dashboard_routes.py
- B4: verification_routes.py, target_routes.py, settings_routes.py

### Task Group C: Frontend Core + Layout + Store (Agent 3)
- C1: Config files, entry points, types, API client
- C2: Redux store (8 slices)
- C3: Layout + common components
- C4: App.tsx with routing

### Task Group D: Frontend Pages + Domain Components (Agent 4)
- D1: Dashboard components (ScopeDonut, TrendChart, Waterfall, etc.)
- D2: Inventory components (BoundaryWizard, EntityTree, BaseYearConfig)
- D3: Scope 1/2/3 components
- D4: Reports + Targets + Verification components
- D5: All 8 pages

### Task Group E: Tests + DB Migration (Agent 5)
- E1: 10 backend test files (300+ tests)
- E2: V083 database migration

---

## 5. Acceptance Criteria

1. Organizational boundary with 3 consolidation approaches
2. Base year management with recalculation triggers
3. Scope 1 aggregation from 8 source categories
4. Scope 2 dual reporting (location + market) with reconciliation
5. Scope 3 all 15 categories with materiality screening
6. GHG intensity metrics (4+ denominators)
7. Monte Carlo uncertainty propagation across all scopes
8. GHG Protocol completeness check (15 mandatory disclosures)
9. Multi-format report export (PDF, Excel, JSON, CSV)
10. Verification workflow (internal + external)
11. Target tracking with SBTi alignment
12. Full React dashboard with 8 pages
13. 50+ REST API endpoints
14. 300+ backend tests
15. Database migration for inventory management
