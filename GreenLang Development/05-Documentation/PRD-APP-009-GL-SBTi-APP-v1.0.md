# PRD: APP-009 -- GL-SBTi-APP v1.0

| Field         | Value                                                        |
|---------------|--------------------------------------------------------------|
| **PRD ID**    | PRD-APP-009                                                  |
| **Application** | GL-SBTi-APP v1.0 Beta (SBTi Target Validation)            |
| **Priority**  | P1                                                           |
| **Version**   | 1.0.0                                                        |
| **Status**    | In Progress                                                  |
| **Author**    | GreenLang Platform Team                                      |
| **Date**      | 2026-03-03                                                   |
| **Standard**  | SBTi Corporate Near-Term Criteria V5.3, Net-Zero Standard V1.3, FINZ V1.0, FLAG Guidance |
| **Base**      | INFRA-001..010, SEC-001..011, OBS-001..005, AGENT-FOUND-001..010, AGENT-MRV-001..030 |
| **Ralphy Task ID** | APP-009                                                 |

---

## 1. Overview

### 1.1 Purpose

GL-SBTi-APP v1.0 is a comprehensive **Science Based Targets initiative (SBTi) Target Validation Platform** that enables organizations to set, validate, track, and manage science-based GHG emission reduction targets aligned with the Paris Agreement. The platform implements SBTi Corporate Near-Term Criteria V5.3, Net-Zero Standard V1.3, Financial Institutions Net-Zero Standard (FINZ) V1.0, FLAG Guidance, and all sector-specific pathways.

The application differentiates itself by providing **automated target validation** against all SBTi Criteria Assessment Indicators, **pathway calculation engines** for both Absolute Contraction Approach (ACA) and Sectoral Decarbonization Approach (SDA), and **deep integration with all 30 MRV agents** for real-time emissions tracking against target pathways. It also provides **temperature scoring** at company and portfolio levels, **five-year review management**, and **cross-framework alignment** with CDP, TCFD, CSRD, GHG Protocol, and ISO 14064.

The platform serves both **corporate users** (setting and tracking their own targets) and **financial institutions** (portfolio-level target setting via FINZ, PCAF integration, portfolio coverage tracking). It covers all 12+ SBTi-recognized sectors with sector-specific intensity pathways.

Core capabilities:

1. Near-term target configuration and validation (5-10 year, 1.5C/WB2C pathways)
2. Long-term target configuration (2050, >=90% reduction)
3. Net-zero target management (4-pillar: near-term + long-term + residual neutralization + BVCM)
4. Absolute Contraction Approach (ACA) pathway calculator (4.2%/yr for 1.5C)
5. Sectoral Decarbonization Approach (SDA) pathway calculator (12+ sectors)
6. FLAG target assessment (11 commodity pathways + sector pathway, deforestation commitment)
7. Scope 3 screening (40% trigger, 67%/90% coverage, category hotspot analysis)
8. Automated criteria validation against SBTi Criteria Assessment Indicators V1.5
9. Temperature scoring (company-level and portfolio-level alignment)
10. Progress tracking with annual pathway variance analysis
11. Base year recalculation management (5% threshold, revalidation triggers)
12. Five-year mandatory review workflow with deadline tracking
13. Financial institutions module (FINZ, portfolio coverage, PCAF integration)
14. Cross-framework alignment (CDP C4, TCFD MT, CSRD ESRS E1, GHG Protocol, ISO 14064)
15. Sector-specific pathway libraries (power, cement, steel, buildings, maritime, aviation, chemicals, land transport, apparel)

### 1.2 SBTi Framework Mapping

#### 1.2.1 Target Types

| Target Type | Timeframe | Scope 1+2 Coverage | Scope 3 Coverage | Ambition | Method |
|-------------|-----------|---------------------|-------------------|----------|--------|
| **Near-Term** | 5-10 years | >= 95% | >= 67% (if S3 >= 40%) | 1.5C (S1+2), WB2C (S3) | ACA, SDA |
| **Long-Term** | By 2050 | >= 95% | >= 90% | 1.5C | ACA, SDA |
| **Net-Zero** | By 2050 | >= 95% | >= 90% | 1.5C + residual neutralization | ACA, SDA + CDR |
| **FLAG** | 5-10 years / 2050 | N/A | FLAG emissions | 1.5C | Commodity/Sector pathway |

#### 1.2.2 Target-Setting Methods

| Method | Description | Annual Rate (1.5C) | Applicable Scopes |
|--------|-------------|--------------------|--------------------|
| **ACA** | Absolute Contraction Approach | 4.2% linear | S1, S2, S3 |
| **SDA** | Sectoral Decarbonization Approach | Sector-specific intensity | S1, S2 (sector-dependent) |
| **Economic Intensity** | CO2e per revenue/value added | Varies | S3 only |
| **Physical Intensity** | CO2e per physical output | Sector-specific | S1, S2, S3 |
| **Supplier Engagement** | % suppliers with SBTi targets | N/A | S3 |

#### 1.2.3 Sector-Specific Pathways

| Sector | Intensity Metric | Pathway | Reference |
|--------|-----------------|---------|-----------|
| **Power** | tCO2/MWh | 1.5C | IEA NZE |
| **Cement** | tCO2e/tonne cement | 1.5C | SBTi Cement V1.0 |
| **Steel** | tCO2e/tonne steel | 1.5C | SBTi Steel V1.0 |
| **Buildings** | kgCO2e/m2 | 1.5C | CRREM |
| **Maritime** | gCO2/dwt-nm | 1.5C | IMO pathways |
| **Aviation** | gCO2/RPK, gCO2/RTK | WB2C+interim 1.5C | ICAO |
| **Land Transport** | gCO2/vkm | 1.5C | IEA NZE |
| **Chemicals** | Sector-specific | 1.5C | SBTi Chemicals V1.0 |
| **Apparel** | tCO2e per unit | 1.5C | SBTi Apparel V1.3 |
| **FLAG** | Commodity-specific | 1.5C | SBTi FLAG |
| **Financial Institutions** | Portfolio coverage/temperature | 1.5C | FINZ V1.0 |

#### 1.2.4 Criteria Assessment Indicators (V1.5)

| Criterion | Description | Validation Check |
|-----------|-------------|------------------|
| **C1** | Organizational boundary | Parent/group-level, all subsidiaries |
| **C2** | Greenhouse gases | All 7 GHGs per GHG Protocol |
| **C3** | Scope 1+2 | Company-wide coverage >= 95% |
| **C4** | Base year | No earlier than 2015 |
| **C5** | Target timeframe | 5-10 years from submission |
| **C6** | Target ambition S1+2 | 1.5C-aligned (>= 4.2%/yr ACA) |
| **C7** | Target ambition S3 | WB2C-aligned minimum |
| **C8** | Scope 3 trigger | Required if S3 >= 40% of total |
| **C9** | Scope 3 coverage | >= 67% of S3 emissions |
| **C10** | Bioenergy | Land-related emissions included |
| **C11** | Carbon credits | Not counted toward reductions |
| **C12** | Avoided emissions | Separate accounting |
| **C13** | Target timeframe | Within 5-10 year window |
| **C14-C28** | Additional criteria | Sector-specific, FLAG, FI requirements |

#### 1.2.5 FLAG Target Requirements

| Requirement | Threshold | Detail |
|-------------|-----------|--------|
| **FLAG trigger** | FLAG > 20% of total | Mandatory FLAG target |
| **Sector pathway** | 3.03%/yr reduction | Absolute reduction |
| **Commodity pathways** | 11 commodities | Intensity convergence |
| **Deforestation** | By 2025 | No-deforestation commitment |
| **Long-term** | >= 72% by 2050 | FLAG-specific pathway |

### 1.3 Technical Context

- **Backend**: Python 3.12, FastAPI, Pydantic v2, async PostgreSQL
- **Frontend**: React 18, TypeScript 5.3, Redux Toolkit, Material-UI, Recharts, Vite
- **MRV Agents**: All 30 MRV agents (001-030) for real-time emissions data
- **Support Agents**: AGENT-FOUND-001 (Orchestrator), AGENT-DATA-010 (Data Quality Profiler)
- **Database**: PostgreSQL + TimescaleDB (V087 migration)
- **Existing Apps**: GL-GHG-APP (emissions inventory), GL-CDP-APP (CDP alignment), GL-TCFD-APP (TCFD metrics), GL-ISO14064-APP (verification)

---

## 2. Application Components

### 2.1 Backend: Target Configuration Engine (~1,200 lines)
- CRUD for near-term, long-term, and net-zero targets
- Target scope definition (S1, S2, S1+2, S3, S1+2+3, FLAG)
- Coverage percentage validation (95% S1+2, 67%/90% S3)
- Base year selection and validation (>= 2015)
- Target year computation (5-10 year window from submission)
- Multi-target management per organization (separate FLAG, FI targets)
- Target status lifecycle (committed, submitted, validated, published, expired, withdrawn)
- Submission form data collection aligned with SBTi booking system
- Recalculation trigger detection and workflow

### 2.2 Backend: Pathway Calculator Engine (~1,400 lines)
- Absolute Contraction Approach (ACA): 4.2%/yr (1.5C), 2.5%/yr (WB2C) linear reduction
- ACA formula: `Target = Base * (1 - rate)^years`
- Sectoral Decarbonization Approach (SDA) for 12+ sectors
- SDA intensity pathway convergence calculations per IEA/sector budgets
- Economic intensity pathway calculator (CO2e per revenue)
- Physical intensity pathway calculator (CO2e per physical unit)
- Supplier engagement pathway calculator (% suppliers with targets)
- FLAG commodity intensity pathways (11 commodities)
- FLAG sector pathway (3.03%/yr absolute reduction)
- Multi-pathway comparison and optimization
- Interim milestone generation (annual pathway points)
- Uncertainty bands (+/- confidence intervals on pathways)

### 2.3 Backend: Validation Engine (~1,500 lines)
- Automated assessment against all SBTi Criteria Assessment Indicators (C1-C28+)
- Net-Zero criteria validation (NZ-C1 through NZ-C14+)
- Boundary completeness checker (organizational, operational, equity share)
- Coverage threshold validation (95% S1+2, 67%/90% S3)
- Ambition level assessment (1.5C vs WB2C alignment verification)
- Base year requirements validation (>= 2015, consistent S1+S2 base year)
- Target timeframe validation (5-10 year window)
- Bioenergy inclusion check (C10)
- Carbon credit exclusion verification (C11)
- Avoided emissions separation check (C12)
- Sector-specific criteria validation
- FLAG-specific criteria validation
- Financial institution-specific criteria validation (FINZ)
- Validation result scoring (pass/fail per criterion, overall readiness %)
- Pre-submission readiness checklist

### 2.4 Backend: Scope 3 Screening Engine (~900 lines)
- 40% trigger assessment (S3 / total emissions)
- 15-category Scope 3 breakdown analysis
- Category-level hotspot identification (top contributors)
- 67% coverage calculator (near-term) and 90% coverage calculator (long-term)
- Minimum boundary assessment per GHG Protocol category
- Data availability assessment per category
- Recommended target categories selection
- Spend-based vs supplier-specific vs average-data method recommendations
- Integration with all 15 Scope 3 MRV agents (MRV-014 through MRV-028)

### 2.5 Backend: FLAG Assessment Engine (~1,000 lines)
- 20% FLAG trigger assessment (FLAG emissions / total)
- FLAG sector classification (agriculture, food processing, retail, forestry, tobacco)
- 11 commodity pathway calculator (beef, chicken, dairy, leather, maize, palm oil, pork, rice, soy, wheat, timber)
- Sector pathway calculator (3.03%/yr absolute reduction)
- Deforestation commitment tracker
- FLAG removals eligibility assessment (storage, monitoring, traceability requirements)
- Long-term FLAG target validation (>= 72% by 2050)
- FLAG emissions separation from non-FLAG Scope 1+2+3
- Integration with MRV-006 (Land Use) and MRV-008 (Agricultural)

### 2.6 Backend: Sector Pathway Engine (~1,300 lines)
- Sector intensity pathway library for 12+ SBTi sectors
- Power sector: tCO2/MWh convergence pathway (IEA NZE)
- Cement sector: tCO2e/tonne pathway with CCS considerations
- Steel sector: tCO2e/tonne with iron-ore vs scrap differentiation
- Buildings sector: kgCO2e/m2 via CRREM regional pathways
- Maritime sector: gCO2/dwt-nm pathway
- Aviation sector: gCO2/RPK and gCO2/RTK pathways
- Land transport: gCO2/vkm with ICE phase-out timelines
- Chemicals sector: sector-specific pathways (V1.0)
- Apparel sector: tCO2e per unit with FLAG integration
- Custom sector pathway definition
- Sector classification auto-detection from ISIC/NACE/NAICS codes
- Multi-sector companies: weighted pathway blending

### 2.7 Backend: Progress Tracking Engine (~1,100 lines)
- Annual emissions tracking against target pathway
- Pathway variance analysis (actual vs expected, % deviation)
- On-track / off-track determination with RAG status
- Cumulative reduction calculation from base year
- Projected target achievement analysis
- Acceleration/deceleration rate calculation
- Year-over-year emissions trend analysis
- Scope-level progress breakdown (S1, S2, S3 separately)
- Category-level Scope 3 progress tracking
- MRV agent integration for real-time emissions data
- Progress dashboard data generation

### 2.8 Backend: Temperature Scoring Engine (~1,000 lines)
- Company-level temperature score calculation (0-4C range)
- Target-to-temperature mapping using SBTi/CDP methodology
- Short-term vs long-term temperature scores
- Overall company temperature rating (weighted S1+2+3)
- Portfolio-level temperature scoring for financial institutions
- Temperature score time series tracking
- Peer comparison temperature rankings
- Temperature alignment visualization data
- Implied Temperature Rise (ITR) methodology
- Engagement temperature scoring (based on supplier targets)

### 2.9 Backend: Recalculation Engine (~800 lines)
- 5% threshold monitoring on base year emissions
- Trigger detection (acquisition, divestment, merger, methodology change, structural change)
- Pre/post recalculation comparison
- Recalculated base year emissions computation
- Target revalidation assessment post-recalculation
- Recalculation audit trail
- M&A impact modeling on targets
- Organic vs inorganic growth separation

### 2.10 Backend: Five-Year Review Engine (~700 lines)
- Trigger date calculation (5 years from initial validation)
- Review deadline tracking (trigger date + 12 months)
- Review readiness assessment
- Updated criteria compliance check
- Progress-to-date summary for review submission
- Notification scheduling (12-month, 6-month, 3-month, 1-month alerts)
- Review outcome recording (renewed, updated, expired)
- Historical review tracking

### 2.11 Backend: Financial Institutions Module (~1,200 lines)
- FINZ V1.0 standard implementation
- Portfolio coverage approach (linear path to 100% by 2040)
- Temperature scoring at portfolio level
- SDA for financial institution sector targets (Segment B)
- PCAF data quality integration (DQ 1-5)
- Asset class-specific calculations (listed equity, bonds, private equity, CRE, mortgages)
- Financed emissions attribution (EVIC, revenue, balance sheet)
- WACI (Weighted Average Carbon Intensity) calculation
- Portfolio alignment assessment
- Investee engagement tracking
- SBTi-FI to NZBA/GFANZ alignment

### 2.12 Backend: Framework Integration Engine (~900 lines)
- CDP C4 (Targets & Performance) alignment mapping
- TCFD Metrics & Targets (MT-c) cross-reference
- CSRD ESRS E1 (Climate Change) transition plan alignment
- GHG Protocol base year and inventory alignment
- ISO 14064 verification linkage
- SB 253 (California) reporting alignment
- Cross-framework gap identification
- Unified target reporting across frameworks

### 2.13 Backend: Reporting Engine (~1,100 lines)
- SBTi Target Submission Form auto-population
- Target progress reports (annual/quarterly)
- Validation readiness report
- Temperature alignment report
- Portfolio coverage report (for FIs)
- Export formats: PDF, Excel, JSON, XML
- Comparison reports (current vs pathway, peer vs peer)
- Executive summary generation
- Board-ready presentation data

### 2.14 Backend: Gap Analysis Engine (~800 lines)
- Criteria-by-criteria gap assessment (C1-C28+)
- Data gap identification (missing emissions categories, coverage shortfalls)
- Ambition gap analysis (current reduction rate vs required)
- Process gap identification (governance, reporting, verification)
- Prioritized action plan generation
- Estimated effort and timeline per gap
- Peer benchmarking on readiness scores

### 2.15 Frontend: Dashboard Page
- Overall SBTi readiness score (0-100%)
- Target status summary cards (near-term, long-term, net-zero, FLAG)
- Pathway chart with actual emissions overlay
- Temperature alignment gauge (company-level)
- Five-year review countdown
- Key milestone timeline
- Quick-start wizard for new targets

### 2.16 Frontend: Target Configuration Page
- Target type selector (near-term, long-term, net-zero)
- Scope selector with coverage calculator
- Method selector (ACA, SDA, intensity, engagement)
- Base year and target year configuration
- Pathway visualization with reduction milestones
- FLAG target configuration sub-section
- Multi-target management table

### 2.17 Frontend: Pathway Calculator Page
- Interactive pathway builder with parameter sliders
- ACA vs SDA comparison charts
- Sector pathway selection with intensity metric display
- Multi-scenario pathway overlay
- Uncertainty band visualization
- Milestone markers on pathway curves

### 2.18 Frontend: Validation Checker Page
- Criteria checklist (C1-C28+) with pass/fail indicators
- Coverage dashboard (S1+2 %, S3 %, FLAG %)
- Ambition level indicator (1.5C, WB2C)
- Pre-submission readiness score
- Issue list with resolution guidance
- Export validation report

### 2.19 Frontend: Progress Tracking Page
- Annual emissions vs pathway chart
- On-track / off-track indicator with trend arrows
- Scope breakdown progress bars
- Category-level Scope 3 progress table
- Year-over-year reduction rate chart
- Projected achievement date

### 2.20 Frontend: Temperature Scoring Page
- Company temperature gauge (0-4C dial)
- Short-term vs long-term temperature comparison
- Scope-level temperature breakdown
- Peer temperature ranking table
- Portfolio temperature dashboard (for FIs)

### 2.21 Frontend: Scope 3 Screening Page
- 40% trigger assessment result
- 15-category waterfall chart
- Category hotspot heatmap
- Coverage calculator with category selector
- Data quality indicators per category

### 2.22 Frontend: FLAG Assessment Page
- 20% trigger result
- Commodity pathway selector
- Deforestation commitment tracker
- FLAG vs non-FLAG emissions split
- Commodity-level intensity charts

### 2.23 Frontend: Financial Institutions Page
- Portfolio coverage progress chart (path to 100% by 2040)
- Financed emissions breakdown by asset class
- PCAF data quality distribution
- Investee engagement tracker
- WACI trend chart

### 2.24 Frontend: Recalculation & Review Page
- Base year change monitor
- Recalculation trigger alerts
- Five-year review timeline
- Review readiness checklist
- Historical review log

### 2.25 Frontend: Reports Page
- Report builder with template selection
- Preview panel
- Export dialog (PDF, Excel, JSON, XML)
- Submission form preview
- Validation package assembly

### 2.26 Frontend: Settings Page
- Organization profile configuration
- Sector classification (ISIC/NACE/NAICS)
- Framework integration preferences
- MRV agent connection settings
- Notification preferences
- User role management

---

## 3. File Structure

```
applications/GL-SBTi-APP/SBTi-Target-Platform/
    config/
        sbti_config.yaml                         (~500 lines)
    services/
        __init__.py                              (~80 lines)
        config.py                                (~1,500 lines)
        models.py                                (~2,000 lines)
        setup.py                                 (~550 lines)
        target_configuration_engine.py           (~1,200 lines)
        pathway_calculator_engine.py             (~1,400 lines)
        validation_engine.py                     (~1,500 lines)
        scope3_screening_engine.py               (~900 lines)
        flag_assessment_engine.py                (~1,000 lines)
        sector_pathway_engine.py                 (~1,300 lines)
        progress_tracking_engine.py              (~1,100 lines)
        temperature_scoring_engine.py            (~1,000 lines)
        recalculation_engine.py                  (~800 lines)
        five_year_review_engine.py               (~700 lines)
        financial_institutions_engine.py         (~1,200 lines)
        framework_integration_engine.py          (~900 lines)
        reporting_engine.py                      (~1,100 lines)
        gap_analysis_engine.py                   (~800 lines)
        api/
            __init__.py                          (~60 lines)
            target_routes.py                     (~800 lines)
            pathway_routes.py                    (~600 lines)
            validation_routes.py                 (~700 lines)
            scope3_routes.py                     (~500 lines)
            flag_routes.py                       (~500 lines)
            sector_routes.py                     (~500 lines)
            progress_routes.py                   (~600 lines)
            temperature_routes.py                (~500 lines)
            recalculation_routes.py              (~400 lines)
            review_routes.py                     (~400 lines)
            fi_routes.py                         (~600 lines)
            framework_routes.py                  (~400 lines)
            reporting_routes.py                  (~600 lines)
            dashboard_routes.py                  (~400 lines)
            gap_routes.py                        (~500 lines)
            settings_routes.py                   (~600 lines)
    frontend/
        index.html
        package.json
        tsconfig.json
        vite.config.ts
        public/
        src/
            App.tsx
            main.tsx
            components/
                common/                          # DataTable, LoadingSpinner, ScoreGauge, etc.
                layout/                          # Header, Sidebar, Layout
                dashboard/                       # ReadinessScore, TargetCards, PathwayPreview, etc.
                targets/                         # TargetForm, ScopeSelector, MethodPicker, etc.
                pathways/                        # PathwayChart, ACAvsSDASider, MilestoneMarker, etc.
                validation/                      # CriteriaChecklist, CoverageGauge, ReadinessBar, etc.
                progress/                        # EmissionsVsPathway, RAGIndicator, TrendArrows, etc.
                temperature/                     # TempGauge, ScopeBreakdown, PeerRanking, etc.
                scope3/                          # TriggerAssessment, CategoryWaterfall, HotspotMap, etc.
                flag/                            # FLAGTrigger, CommoditySelector, DeforestationTracker, etc.
                fi/                              # PortfolioCoverage, FinancedEmissions, PCAFQuality, etc.
                recalculation/                   # ChangeMonitor, TriggerAlerts, ReviewTimeline, etc.
                reports/                         # ReportBuilder, PreviewPanel, ExportDialog, etc.
                frameworks/                      # FrameworkAlignment, CrossRef, GapIndicator, etc.
            pages/                               # 14 pages
                Dashboard.tsx
                TargetConfiguration.tsx
                PathwayCalculator.tsx
                ValidationChecker.tsx
                ProgressTracking.tsx
                TemperatureScoring.tsx
                Scope3Screening.tsx
                FLAGAssessment.tsx
                FinancialInstitutions.tsx
                RecalculationReview.tsx
                Reports.tsx
                FrameworkAlignment.tsx
                GapAnalysis.tsx
                Settings.tsx
            store/
                index.ts
                hooks.ts
                slices/                          # 14 Redux slices
            services/
                api.ts
            types/
                index.ts
            utils/
                formatters.ts
                pathwayHelpers.ts
                validators.ts
```

---

## 4. Database Migration

```sql
-- V087__sbti_app_service.sql
-- GL-SBTi-APP v1.0: Science Based Targets Target Validation Platform
-- Tables: 25 tables + 3 hypertables + 2 continuous aggregates

-- Core tables:
-- gl_sbti_organizations         -- Organization profiles with sector classification
-- gl_sbti_emissions_inventories -- Base year and annual emissions data
-- gl_sbti_targets               -- Target definitions (near-term, long-term, net-zero)
-- gl_sbti_target_scopes         -- Per-scope target configuration
-- gl_sbti_pathways              -- Calculated reduction pathways
-- gl_sbti_pathway_milestones    -- Annual milestone points on pathways
-- gl_sbti_validation_results    -- Criteria validation results (hypertable)
-- gl_sbti_criteria_checks       -- Individual criterion pass/fail results
-- gl_sbti_scope3_screenings     -- Scope 3 category screening results
-- gl_sbti_scope3_categories     -- 15-category emissions breakdown
-- gl_sbti_flag_assessments      -- FLAG target assessments
-- gl_sbti_flag_commodities      -- Commodity-level FLAG data
-- gl_sbti_sector_pathways       -- Sector-specific intensity pathways
-- gl_sbti_sector_benchmarks     -- Sector pathway benchmark data
-- gl_sbti_progress_records      -- Annual progress tracking (hypertable)
-- gl_sbti_temperature_scores    -- Temperature alignment scores (hypertable)
-- gl_sbti_recalculations        -- Base year recalculation events
-- gl_sbti_five_year_reviews     -- Five-year review records
-- gl_sbti_fi_portfolios         -- Financial institution portfolios
-- gl_sbti_fi_portfolio_holdings -- Portfolio holdings/investees
-- gl_sbti_fi_engagement         -- Investee engagement tracking
-- gl_sbti_framework_mappings    -- Cross-framework alignment records
-- gl_sbti_reports               -- Generated reports
-- gl_sbti_gap_assessments       -- Gap analysis results
-- gl_sbti_gap_items             -- Individual gap items

-- Hypertables: gl_sbti_validation_results, gl_sbti_progress_records, gl_sbti_temperature_scores
-- Continuous aggregates: sbti_app.annual_progress_summary, sbti_app.quarterly_temperature_trends
```

---

## 5. Development Tasks

### Task Group A: Backend Core Engines (Agent 1 -- Opus)
- A1: `config.py` -- Enumerations, SBTi criteria, sector pathways, pathway rates, configuration
- A2: `models.py` -- Pydantic v2 domain models for all SBTi entities
- A3: `target_configuration_engine.py` -- Target CRUD, lifecycle, scope management
- A4: `pathway_calculator_engine.py` -- ACA, SDA, intensity, engagement pathway calculators
- A5: `validation_engine.py` -- Automated criteria validation (C1-C28+, NZ-C1-C14+)
- A6: `scope3_screening_engine.py` -- 40% trigger, coverage, hotspot analysis
- A7: `flag_assessment_engine.py` -- FLAG trigger, commodity/sector pathways, deforestation
- A8: `sector_pathway_engine.py` -- 12+ sector intensity pathway library
- A9: `progress_tracking_engine.py` -- Annual tracking, variance, on-track analysis
- A10: `temperature_scoring_engine.py` -- Company/portfolio temperature scores
- A11: `recalculation_engine.py` -- 5% threshold, recalculation workflow
- A12: `five_year_review_engine.py` -- Review lifecycle, deadlines, notifications
- A13: `financial_institutions_engine.py` -- FINZ, portfolio coverage, PCAF
- A14: `framework_integration_engine.py` -- CDP, TCFD, CSRD cross-references
- A15: `reporting_engine.py` -- Report generation, export formats
- A16: `gap_analysis_engine.py` -- Criteria gap assessment, action planning
- A17: `setup.py` -- Platform facade, FastAPI app factory
- A18: `__init__.py` -- Package exports

### Task Group B: Backend API Layer (Agent 2 -- Opus)
- B1: `api/__init__.py` -- Router exports
- B2: `api/target_routes.py` -- Target CRUD endpoints
- B3: `api/pathway_routes.py` -- Pathway calculation endpoints
- B4: `api/validation_routes.py` -- Validation check endpoints
- B5: `api/scope3_routes.py` -- Scope 3 screening endpoints
- B6: `api/flag_routes.py` -- FLAG assessment endpoints
- B7: `api/sector_routes.py` -- Sector pathway endpoints
- B8: `api/progress_routes.py` -- Progress tracking endpoints
- B9: `api/temperature_routes.py` -- Temperature scoring endpoints
- B10: `api/recalculation_routes.py` -- Recalculation endpoints
- B11: `api/review_routes.py` -- Five-year review endpoints
- B12: `api/fi_routes.py` -- Financial institutions endpoints
- B13: `api/framework_routes.py` -- Framework alignment endpoints
- B14: `api/reporting_routes.py` -- Reporting endpoints
- B15: `api/dashboard_routes.py` -- Dashboard endpoints
- B16: `api/gap_routes.py` -- Gap analysis endpoints
- B17: `api/settings_routes.py` -- Settings endpoints

### Task Group C: Frontend Core + Layout + Store (Agent 3)
- C1: `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`
- C2: `src/types/index.ts` -- TypeScript type definitions
- C3: `src/services/api.ts` -- Axios API client
- C4: `src/store/index.ts`, `hooks.ts` -- Redux store setup
- C5: `src/store/slices/` -- 14 feature slices
- C6: `src/components/layout/` -- Header, Sidebar, Layout
- C7: `src/components/common/` -- Shared UI components
- C8: `src/utils/` -- Formatters, pathway helpers, validators
- C9: `src/App.tsx`, `src/main.tsx` -- App shell and entry

### Task Group D: Frontend Pages + Domain Components (Agent 4)
- D1: `src/pages/Dashboard.tsx` + `src/components/dashboard/`
- D2: `src/pages/TargetConfiguration.tsx` + `src/components/targets/`
- D3: `src/pages/PathwayCalculator.tsx` + `src/components/pathways/`
- D4: `src/pages/ValidationChecker.tsx` + `src/components/validation/`
- D5: `src/pages/ProgressTracking.tsx` + `src/components/progress/`
- D6: `src/pages/TemperatureScoring.tsx` + `src/components/temperature/`
- D7: `src/pages/Scope3Screening.tsx` + `src/components/scope3/`
- D8: `src/pages/FLAGAssessment.tsx` + `src/components/flag/`
- D9: `src/pages/FinancialInstitutions.tsx` + `src/components/fi/`
- D10: `src/pages/RecalculationReview.tsx` + `src/components/recalculation/`
- D11: `src/pages/Reports.tsx` + `src/components/reports/`
- D12: `src/pages/FrameworkAlignment.tsx` + `src/components/frameworks/`
- D13: `src/pages/GapAnalysis.tsx`
- D14: `src/pages/Settings.tsx`

### Task Group E: Tests + DB Migration + Auth Integration (Agent 5)
- E1: `tests/unit/apps/sbti/conftest.py` -- Shared fixtures
- E2: `tests/unit/apps/sbti/test_models.py` -- Model validation tests
- E3: `tests/unit/apps/sbti/test_target_configuration.py`
- E4: `tests/unit/apps/sbti/test_pathway_calculator.py`
- E5: `tests/unit/apps/sbti/test_validation_engine.py`
- E6: `tests/unit/apps/sbti/test_scope3_screening.py`
- E7: `tests/unit/apps/sbti/test_flag_assessment.py`
- E8: `tests/unit/apps/sbti/test_sector_pathways.py`
- E9: `tests/unit/apps/sbti/test_progress_tracking.py`
- E10: `tests/unit/apps/sbti/test_temperature_scoring.py`
- E11: `tests/unit/apps/sbti/test_recalculation.py`
- E12: `tests/unit/apps/sbti/test_five_year_review.py`
- E13: `tests/unit/apps/sbti/test_fi_module.py`
- E14: `tests/unit/apps/sbti/test_framework_integration.py`
- E15: `tests/unit/apps/sbti/test_reporting.py`
- E16: `tests/unit/apps/sbti/test_gap_analysis.py`
- E17: `tests/unit/apps/sbti/test_api_routes.py`
- E18: `deployment/database/migrations/sql/V087__sbti_app_service.sql`
- E19: Auth integration (`auth_setup.py`, `route_protector.py` updates)

---

## 6. Acceptance Criteria

1. All 14 backend service engines implemented with full SBTi criteria coverage
2. ACA pathway calculator correctly computes 4.2%/yr linear reduction for 1.5C alignment
3. SDA pathway calculator supports all 12+ SBTi-recognized sectors with correct intensity metrics
4. Validation engine checks all Criteria Assessment Indicators (C1-C28+, NZ-C1-C14+)
5. Scope 3 screening correctly identifies 40% trigger and calculates 67%/90% coverage
6. FLAG assessment correctly identifies 20% trigger and supports all 11 commodity pathways
7. Progress tracking engine integrates with all 30 MRV agents for real-time emissions data
8. Temperature scoring produces company-level scores in 0-4C range using CDP/WWF methodology
9. Five-year review engine correctly calculates trigger dates and manages review lifecycle
10. Financial institutions module implements FINZ V1.0 with portfolio coverage path to 100% by 2040
11. 17 API routers expose 80+ endpoints with proper authentication and authorization
12. 14 frontend pages with 80+ React components providing full user workflow
13. V087 database migration creates 25 tables, 3 hypertables, 2 continuous aggregates
14. 17+ test files with 400+ tests achieving comprehensive coverage
15. Full auth integration with sbti-* permissions in PERMISSION_MAP
16. Cross-framework alignment with CDP, TCFD, CSRD, GHG Protocol, ISO 14064

---

## 7. Framework Alignment

| Framework | GL-SBTi-APP Alignment |
|-----------|-----------------------|
| **SBTi Corporate V5.3** | Full criteria implementation (C1-C28+), ACA/SDA methods, near-term targets |
| **SBTi Net-Zero V1.3** | 4-pillar net-zero standard (NZ-C1-C14+), residual neutralization, BVCM |
| **SBTi FINZ V1.0** | Financial institutions module with portfolio coverage, temperature scoring |
| **SBTi FLAG** | FLAG assessment with 11 commodity pathways, sector pathway, deforestation commitment |
| **CDP** | C4 (Targets & Performance) data mapping via GL-CDP-APP integration |
| **TCFD** | Metrics & Targets pillar (MT-c) alignment via GL-TCFD-APP integration |
| **CSRD ESRS E1** | Transition plan targets alignment via GL-CSRD-APP integration |
| **GHG Protocol** | Base inventory methodology, Scope definitions, 15 Scope 3 categories |
| **ISO 14064** | Verification linkage via GL-ISO14064-APP integration |
| **SB 253** | California disclosure reporting alignment |
