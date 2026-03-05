# PRD: APP-006 -- GL-ISO14064-APP v1.0

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-APP-006 |
| Application | GL-ISO14064-APP v1.0 (Beta) |
| Priority | P1 (High) |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-03-02 |
| Standard | ISO 14064-1:2018 (Greenhouse gases -- Part 1: Specification with guidance at the organization level for quantification and reporting of greenhouse gas emissions and removals) |
| Base | 28 MRV Agents (662K+ lines), Scope 3 Category Mapper, Audit Trail & Lineage, GL-GHG-APP v1.0 |
| Ralphy Task ID | APP-006 |

---

## 1. Overview

### 1.1 Purpose
Build GL-ISO14064-APP v1.0 as a comprehensive ISO 14064-1:2018 compliance platform for organization-level GHG quantification and reporting. This application orchestrates all 28 existing MRV agents mapped to ISO 14064-1's 6 emission/removal categories, adds GHG removals tracking, significance assessment, quality management planning, and ISO-specific reporting.

**Key differentiator from GL-GHG-APP**: While GL-GHG-APP implements the GHG Protocol Corporate Standard (Scope 1/2/3), GL-ISO14064-APP implements ISO 14064-1:2018 with its 6-category system, explicit removals tracking, significance assessment, quality management plan, and GHG management plan -- plus a cross-walk module mapping between both frameworks.

Core capabilities:
1. **GHG Inventory Design** -- Organizational/operational boundary per Clause 5
2. **6-Category Quantification** -- Direct + 5 indirect categories per Clause 6
3. **GHG Removals Tracking** -- Carbon sinks, CCS, nature-based removals (ISO 14064-1 specific)
4. **Significance Assessment** -- Materiality evaluation for indirect categories
5. **Uncertainty Assessment** -- Quantitative uncertainty per Clause 6.4
6. **Quality Management** -- QM plan per Clause 7
7. **GHG Reporting** -- ISO 14064-1 compliant report per Clause 8
8. **GHG Management Plan** -- Improvement actions per Clause 8.3.5
9. **Verification Support** -- Preparation for ISO 14064-3 verification per Clause 9
10. **GHG Protocol Cross-Walk** -- Automated mapping between ISO 14064-1 categories and GHG Protocol scopes
11. **Dashboard & Analytics** -- Executive dashboard with category breakdowns, trends, quality scores
12. **Multi-Format Export** -- PDF, Excel, JSON, CSV

### 1.2 ISO 14064-1:2018 Requirements Mapping

| ISO 14064-1 Clause | Requirement | GL-ISO14064-APP Feature |
|---------------------|-------------|--------------------------|
| Clause 4 - Principles | Relevance, completeness, consistency, accuracy, transparency | Quality scoring engine, completeness checker |
| Clause 5.1 - Organizational boundary | Control/equity share approach | Boundary manager with control/equity selection |
| Clause 5.2 - Operational boundary | Direct + indirect categories | 6-category classifier + significance assessment |
| Clause 5.3 - Reporting period | Minimum 12-month period | Period manager (calendar/fiscal/custom) |
| Clause 6.1 - Quantification | Activity data x emission factor, direct measurement, mass balance | Multi-method quantification engine |
| Clause 6.2 - Category 1 (Direct) | Direct emissions AND removals | 8 MRV agents + removals tracker |
| Clause 6.3 - Categories 2-6 (Indirect) | 5 indirect categories with significance | Category aggregator + significance engine |
| Clause 6.4 - Uncertainty | Quantitative uncertainty assessment | Monte Carlo + analytical methods |
| Clause 7.1 - QM Plan | Information management, document control | Quality management plan engine |
| Clause 7.2 - Data quality | Activity data + EF quality assessment | 5-dimension data quality index |
| Clause 7.3 - Recalculation | Base year recalculation triggers | Base year manager with recalculation policy |
| Clause 8.1 - GHG Report | Required report contents (14 items) | Report generator with compliance checker |
| Clause 8.2 - Report content | Detailed content requirements | Section-by-section report builder |
| Clause 8.3 - Additional reporting | Management plan, removals, biogenic | Management plan engine, removals module |
| Clause 9 - Verification | Organization's role in verification | Verification workflow with ISO 14064-3 prep |

### 1.3 ISO 14064-1 Category-to-MRV Agent Mapping

| ISO Category | Description | MRV Agents |
|-------------|-------------|------------|
| **Category 1** | Direct GHG emissions and removals | MRV-001 (Stationary), MRV-003 (Mobile), MRV-004 (Process), MRV-005 (Fugitive), MRV-002 (Refrigerants), MRV-006 (Land Use), MRV-007 (Waste Treatment), MRV-008 (Agricultural) + NEW Removals Engine |
| **Category 2** | Indirect from imported energy | MRV-009 (Location), MRV-010 (Market), MRV-011 (Steam/Heat), MRV-012 (Cooling), MRV-013 (Dual Reporting) |
| **Category 3** | Indirect from transportation | MRV-017 (Upstream Transport), MRV-022 (Downstream Transport), MRV-019 (Business Travel), MRV-020 (Employee Commuting) |
| **Category 4** | Indirect from products used by org | MRV-014 (Purchased Goods), MRV-015 (Capital Goods), MRV-016 (Fuel & Energy), MRV-018 (Waste Generated) |
| **Category 5** | Indirect from use of products from org | MRV-023 (Processing Sold), MRV-024 (Use of Sold), MRV-025 (End-of-Life) |
| **Category 6** | Indirect from other sources | MRV-021 (Upstream Leased), MRV-026 (Downstream Leased), MRV-027 (Franchises), MRV-028 (Investments) |

### 1.4 Technical Context
- **Backend**: FastAPI + Python 3.11+ + Pydantic v2
- **Frontend**: React 18 + TypeScript + Material-UI 5 + Redux Toolkit + Recharts + Vite
- **MRV Agents**: 28 production-ready agents at `greenlang/` with individual FastAPI routers
- **Support Agents**: Scope 3 Category Mapper, Audit Trail & Lineage
- **Database**: PostgreSQL + TimescaleDB (V051-V083 already deployed)
- **Existing**: GL-GHG-APP v1.0 patterns for reference

---

## 2. Application Components

### 2.1 Backend: Inventory Boundary Manager
- Organizational boundary definition per Clause 5.1
- Control approach selection (operational control / financial control)
- Equity share approach with percentage tracking
- Multi-entity hierarchy (parent -> subsidiary -> facility -> source)
- Facility registry with geographic metadata (country, region, coordinates)
- Operational boundary: 6 ISO categories with inclusion/exclusion tracking
- Reporting period management (minimum 12 months per Clause 5.3)
- Historical boundary comparison

### 2.2 Backend: GHG Quantification Engine
- Three quantification approaches per Clause 6.1:
  - Calculation-based: activity data x emission factor x GWP
  - Direct measurement: CEMS, flow meters
  - Mass balance: input-output material balance
- Multi-gas tracking: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3
- GWP conversion using AR5/AR6 values (configurable)
- Emission factor source tracking (IPCC, DEFRA, EPA, custom)
- Data quality scoring per source (5-dimension DQI)
- Aggregation across facilities and categories

### 2.3 Backend: Removals Tracker (ISO 14064-1 Specific)
- Carbon sink quantification (forestry, soil carbon, ocean)
- Carbon capture and storage (CCS) tracking
- Nature-based removal methods (afforestation, reforestation, wetland restoration)
- Technological removal methods (direct air capture, BECCS)
- Removal permanence assessment (temporary vs. permanent)
- Net emissions calculation (gross emissions - verified removals)
- Removal verification documentation
- Biogenic CO2 separate tracking per Clause 8.3.2

### 2.4 Backend: Category Aggregation Engine
- Aggregates results from all 28 MRV agents into 6 ISO categories
- Category 1 rollup: 8 direct source types + removals -> net direct
- Category 2 rollup: imported electricity, heat, steam, cooling
- Category 3 rollup: upstream + downstream transport, travel, commuting
- Category 4 rollup: purchased goods, capital goods, fuel & energy, waste
- Category 5 rollup: processing sold, use of sold, end-of-life
- Category 6 rollup: leased assets, franchises, investments
- Grand total: Category 1 (net) + Categories 2-6 (significant)
- Per-gas breakdown across all categories
- Per-facility breakdown
- Cross-walk to GHG Protocol Scope 1/2/3

### 2.5 Backend: Significance Assessment Engine
- Significance evaluation for indirect categories (2-6) per Clause 5.2
- Multi-criteria assessment:
  - Magnitude (absolute emissions relative to total)
  - Influence (organization's ability to reduce)
  - Risk (exposure to regulatory/market risks)
  - Stakeholder concern (external expectations)
  - Data availability (feasibility of quantification)
- Significance threshold (configurable, default 1% of total)
- Automatic recommendation: include/exclude/monitor
- Justification documentation for exclusions
- Year-over-year significance tracking

### 2.6 Backend: Uncertainty Assessment Engine
- Quantitative uncertainty per Clause 6.4
- Monte Carlo simulation (N=10,000 default)
- Analytical method (error propagation, IPCC Approach 1)
- Combined uncertainty across categories
- Per-source uncertainty contribution ranking
- Confidence intervals (90%, 95%, 99%)
- Sensitivity analysis (parameter ranking)
- Data quality contribution to uncertainty
- Convergence assessment

### 2.7 Backend: Quality Management Plan Engine
- QM plan per Clause 7.1:
  - Information management procedures
  - Document control and retention
  - Roles and responsibilities
  - Internal audit schedule
  - Corrective action tracking
- Data quality assessment per Clause 7.2:
  - Activity data quality scoring
  - Emission factor quality scoring
  - Composite data quality indicator
- GHG information management system
- Version control for emission factors
- Calibration records for measurement equipment
- Training records management

### 2.8 Backend: Base Year Manager
- Base year selection per Clause 7.3
- Fixed vs. rolling base year support
- Recalculation trigger detection:
  - Structural changes (mergers, acquisitions, divestitures)
  - Methodology changes
  - Discovery of significant errors
  - Changes in quantification approach
- Significance threshold for recalculation (default 5%)
- Historical recalculation audit trail
- Base year emissions lock mechanism

### 2.9 Backend: Report Generator
- ISO 14064-1 compliant report per Clause 8
- 14 mandatory report elements:
  1. Reporting organization description
  2. Contact person responsible
  3. Reporting period
  4. Organizational boundary documentation
  5. Operational boundary (categories included)
  6. Direct GHG emissions (Category 1)
  7. GHG removals (Category 1)
  8. Indirect emissions (Categories 2-6 -- significant)
  9. Base year and recalculation information
  10. Quantification methodologies
  11. GHG emission/removal factors used
  12. Uncertainty assessment results
  13. Changes from previous report
  14. Statement on whether report has been verified
- Additional reporting per Clause 8.3:
  - GHG management plan
  - Biogenic CO2 emissions/removals
  - Per-gas disaggregation
  - Exclusions and justifications
- Export formats: PDF, Excel (multi-sheet), JSON, CSV

### 2.10 Backend: GHG Management Plan Engine
- Improvement actions tracking per Clause 8.3.5
- Action categories: emission reduction, removal enhancement, data improvement
- Target setting (absolute and intensity)
- Implementation timeline and milestones
- Cost-benefit analysis per action
- Progress tracking against plan
- Annual plan review cycle
- Integration with ISO 14001 EMS

### 2.11 Backend: Verification Workflow
- Preparation for ISO 14064-3 verification per Clause 9
- Internal review stages (data owner -> reviewer -> approver)
- Evidence package assembly
- Verification scope definition
- Materiality assessment (5% of total)
- Finding tracking (material, immaterial, observations)
- Corrective action management
- Verification statement tracking
- Limited vs. reasonable assurance levels

### 2.12 Backend: GHG Protocol Cross-Walk
- Automated mapping between ISO 14064-1 categories and GHG Protocol scopes
- Category 1 -> Scope 1
- Category 2 -> Scope 2
- Categories 3-6 -> Scope 3 (with category-level mapping)
- Reconciliation report between frameworks
- Dual-standard compliance checking
- Gap analysis between frameworks

### 2.13 Frontend: Dashboard
- Executive KPI cards (Total emissions, Net emissions, Removals, Categories 1-6, YoY change)
- 6-category stacked bar/donut chart
- Emissions vs. Removals balance chart
- Monthly/quarterly trend line chart
- Facility geographic heatmap
- Data quality scorecard
- Significance assessment matrix
- Quality management plan status

### 2.14 Frontend: Inventory Setup
- Boundary wizard (org structure, control approach, categories)
- Entity/facility hierarchy builder with drag-and-drop
- Category selection with significance pre-screening
- Reporting period selector
- Base year configuration
- GWP source selection (AR5/AR6)

### 2.15 Frontend: Category Pages (6 Pages)
- **Category 1 (Direct)**: Source breakdown (8 types), per-facility table, gas breakdown, removals panel
- **Category 2 (Imported Energy)**: Electricity/heat/steam/cooling, grid factors, contractual instruments
- **Category 3 (Transportation)**: Upstream/downstream transport, travel, commuting, mode split
- **Category 4 (Products Used)**: Purchased goods, capital goods, fuel & energy, waste
- **Category 5 (Products From Org)**: Processing sold, use of sold, end-of-life
- **Category 6 (Other)**: Leased assets, franchises, investments

### 2.16 Frontend: Removals Page
- Removal sources list with type/method/quantity
- Permanence assessment indicators
- Net emissions calculator (gross - removals)
- Biogenic CO2 tracking panel
- Removal verification status

### 2.17 Frontend: Reports Page
- Report builder with 14 mandatory elements checklist
- Section-by-section preview
- Compliance checker (ISO 14064-1 vs GHG Protocol)
- Cross-walk viewer (ISO categories <-> GHG Protocol scopes)
- Export (PDF, Excel, JSON)
- Report history

### 2.18 Frontend: Quality Management Page
- QM plan dashboard (procedures, audits, actions)
- Data quality matrix (per source, per category)
- Corrective action tracker
- Document control log

### 2.19 Frontend: Management Plan Page
- Improvement actions list with status
- Target progress tracking
- Timeline/Gantt view
- Cost-benefit summary

---

## 3. File Structure

```
applications/GL-ISO14064-APP/ISO14064-Compliance-Platform/
    config/
        iso14064_config.yaml                     (~250 lines)
    services/
        __init__.py                              (~80 lines)
        config.py                                (~400 lines)
        models.py                                (~1,100 lines)
        boundary_manager.py                      (~700 lines)
        quantification_engine.py                 (~800 lines)
        removals_tracker.py                      (~600 lines)
        category_aggregator.py                   (~900 lines)
        significance_engine.py                   (~500 lines)
        uncertainty_engine.py                    (~600 lines)
        quality_management.py                    (~500 lines)
        base_year_manager.py                     (~500 lines)
        report_generator.py                      (~800 lines)
        management_plan.py                       (~500 lines)
        verification_workflow.py                 (~600 lines)
        crosswalk_engine.py                      (~400 lines)
        setup.py                                 (~450 lines)
        api/
            __init__.py                          (~40 lines)
            inventory_routes.py                  (~550 lines)
            category_routes.py                   (~600 lines)
            removals_routes.py                   (~400 lines)
            reporting_routes.py                  (~500 lines)
            dashboard_routes.py                  (~350 lines)
            verification_routes.py               (~400 lines)
            management_plan_routes.py            (~350 lines)
            quality_routes.py                    (~350 lines)
            crosswalk_routes.py                  (~300 lines)
            settings_routes.py                   (~250 lines)
    frontend/
        package.json, tsconfig.json, vite.config.ts, index.html
        src/
            main.tsx, App.tsx
            types/index.ts                       (~500 lines)
            services/api.ts                      (~600 lines)
            store/ (index, hooks, 10 slices)
            components/
                layout/ (Sidebar, Header, Layout)
                common/ (StatCard, StatusBadge, DataTable, LoadingSpinner)
                dashboard/ (CategoryDonut, TrendChart, EmissionsRemovalsChart, QualityScore, SignificanceMatrix, FacilityMap)
                inventory/ (BoundaryWizard, EntityTree, BaseYearConfig, CategorySelector)
                categories/ (CategoryBreakdown, SourceTable, GasBreakdown, FacilityBreakdown)
                removals/ (RemovalsList, PermanenceIndicator, NetEmissionsCalc, BiogenicPanel)
                reports/ (ReportBuilder, ReportPreview, ComplianceChecker, CrossWalkViewer, ExportDialog)
                quality/ (QMPlanDashboard, DataQualityMatrix, CorrectiveActions)
                management/ (ActionsList, TargetProgress, TimelineView)
                verification/ (VerificationTimeline, FindingTracker, EvidencePackage)
            pages/
                Dashboard.tsx
                InventorySetup.tsx
                Category1Direct.tsx
                Category2Energy.tsx
                Category3Transport.tsx
                Category4ProductsUsed.tsx
                Category5ProductsFromOrg.tsx
                Category6Other.tsx
                Removals.tsx
                Reports.tsx
                QualityManagement.tsx
                ManagementPlan.tsx
                Verification.tsx
            utils/ (formatters, validators)
    tests/
        __init__.py
        test_models.py
        test_boundary_manager.py
        test_quantification_engine.py
        test_removals_tracker.py
        test_category_aggregator.py
        test_significance_engine.py
        test_uncertainty_engine.py
        test_quality_management.py
        test_report_generator.py
        test_management_plan.py
        test_verification_workflow.py
        test_crosswalk_engine.py
        test_api_routes.py
```

---

## 4. Database Migration: V084

```sql
-- V084__iso14064_app_service.sql
-- Tables: ~15 tables + 3 hypertables + 2 continuous aggregates

-- Core tables:
-- gl_iso_organizations, gl_iso_entities, gl_iso_boundaries
-- gl_iso_inventories, gl_iso_emission_sources
-- gl_iso_removals, gl_iso_removal_methods
-- gl_iso_category_results, gl_iso_significance_assessments
-- gl_iso_uncertainty_results
-- gl_iso_quality_plans, gl_iso_corrective_actions
-- gl_iso_management_plans, gl_iso_improvement_actions
-- gl_iso_verification_records, gl_iso_findings
-- gl_iso_reports, gl_iso_crosswalk_mappings

-- Hypertables: gl_iso_emission_sources, gl_iso_removals, gl_iso_category_results
-- Continuous aggregates: monthly category totals, quarterly net emissions
```

---

## 5. Development Tasks (Parallel Build Plan)

### Task Group A: Backend Core Engines (Agent 1 - gl-backend-developer)
- A1: config.py, models.py (enums, domain models, request/response schemas)
- A2: boundary_manager.py (organizational/operational boundary, entity hierarchy)
- A3: quantification_engine.py (calculation, measurement, mass balance methods)
- A4: removals_tracker.py (carbon sinks, CCS, nature-based, permanence)
- A5: category_aggregator.py (6-category rollup from 28 MRV agents)
- A6: significance_engine.py (multi-criteria significance assessment)
- A7: uncertainty_engine.py (Monte Carlo + analytical uncertainty)
- A8: quality_management.py (QM plan, data quality, corrective actions)
- A9: base_year_manager.py (selection, recalculation, structural changes)
- A10: report_generator.py (14 mandatory elements, multi-format export)
- A11: management_plan.py (improvement actions, targets, cost-benefit)
- A12: verification_workflow.py (internal review, evidence, findings)
- A13: crosswalk_engine.py (ISO 14064-1 <-> GHG Protocol mapping)
- A14: setup.py + __init__.py + iso14064_config.yaml

### Task Group B: Backend API Layer (Agent 2 - gl-api-developer)
- B1: api/__init__.py, inventory_routes.py (8+ endpoints)
- B2: category_routes.py (6 categories, aggregation endpoints)
- B3: removals_routes.py (CRUD, net emissions, verification)
- B4: reporting_routes.py, dashboard_routes.py
- B5: verification_routes.py, management_plan_routes.py
- B6: quality_routes.py, crosswalk_routes.py, settings_routes.py

### Task Group C: Frontend Core + Layout + Store (Agent 3 - gl-frontend-developer)
- C1: Config files, entry points, types, API client
- C2: Redux store (10 slices)
- C3: Layout + common components
- C4: App.tsx with routing

### Task Group D: Frontend Pages + Domain Components (Agent 4 - gl-frontend-developer)
- D1: Dashboard components (CategoryDonut, TrendChart, EmissionsRemovalsChart, etc.)
- D2: Inventory components (BoundaryWizard, EntityTree, CategorySelector)
- D3: Category components (CategoryBreakdown, SourceTable, GasBreakdown)
- D4: Removals components (RemovalsList, PermanenceIndicator, NetEmissionsCalc)
- D5: Reports + Quality + Management + Verification components
- D6: All 13 pages

### Task Group E: Tests + DB Migration (Agent 5 - gl-test-engineer)
- E1: 13 backend test files (350+ tests)
- E2: V084 database migration

---

## 6. Acceptance Criteria

1. Organizational boundary with control/equity share approaches (Clause 5.1)
2. Operational boundary with 6 ISO categories (Clause 5.2)
3. Category 1 quantification with 8 source types + removals (Clause 6.2)
4. Categories 2-6 quantification via MRV agents (Clause 6.3)
5. GHG removals tracking with permanence assessment (Clause 6.2)
6. Significance assessment for indirect categories (Clause 5.2)
7. Quantitative uncertainty assessment (Clause 6.4)
8. Quality management plan with data quality scoring (Clause 7)
9. Base year management with recalculation triggers (Clause 7.3)
10. ISO 14064-1 compliant report with 14 mandatory elements (Clause 8)
11. GHG management plan with improvement actions (Clause 8.3.5)
12. Verification workflow with ISO 14064-3 preparation (Clause 9)
13. GHG Protocol cross-walk (ISO categories <-> Scopes)
14. Full React dashboard with 13 pages
15. 55+ REST API endpoints
16. 350+ backend tests
17. Database migration with TimescaleDB hypertables
