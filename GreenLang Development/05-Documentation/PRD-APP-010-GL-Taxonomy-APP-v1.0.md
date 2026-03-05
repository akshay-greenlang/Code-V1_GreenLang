# PRD: APP-010 -- GL-Taxonomy-APP v1.0

| Field         | Value                                                        |
|---------------|--------------------------------------------------------------|
| **PRD ID**    | PRD-APP-010                                                  |
| **Application** | GL-Taxonomy-APP v1.0 Alpha (EU Taxonomy Screening & DNSH) |
| **Priority**  | P2                                                           |
| **Version**   | 1.0.0                                                        |
| **Status**    | In Progress                                                  |
| **Author**    | GreenLang Platform Team                                      |
| **Date**      | 2026-03-05                                                   |
| **Standard**  | EU Taxonomy Regulation 2020/852, Climate DA 2021/2139, Environmental DA 2023/2486, Disclosures DA 2021/2178 |
| **Base**      | INFRA-001..010, SEC-001..011, OBS-001..005, AGENT-FOUND-001..010, AGENT-MRV-001..030 |
| **Ralphy Task ID** | APP-010                                                 |

---

## 1. Overview

### 1.1 Purpose

GL-Taxonomy-APP v1.0 is a comprehensive **EU Taxonomy Alignment & Green Investment Ratio Platform** that enables financial institutions and non-financial undertakings to perform EU Taxonomy screening, alignment assessment, DNSH validation, KPI calculation, and regulatory disclosure generation in accordance with the EU Taxonomy Regulation (2020/852) and all associated delegated acts.

The platform implements the full EU Taxonomy four-condition assessment framework: (1) **Substantial Contribution** to at least one of 6 environmental objectives, (2) **Do No Significant Harm (DNSH)** to the remaining 5 objectives, (3) **Minimum Safeguards** compliance (OECD Guidelines, UNGPs, ILO), and (4) **Technical Screening Criteria** (TSC) from the Climate and Environmental Delegated Acts.

The application covers **~240 economic activities** across 13 NACE sectors, with full Technical Screening Criteria databases for all 6 environmental objectives. It provides **deterministic KPI calculation** (Turnover, CapEx, OpEx) for non-financial undertakings, **Green Asset Ratio (GAR)** and **Banking Book Taxonomy Alignment Ratio (BTAR)** for credit institutions, and **Investment KPI** for asset managers and insurers.

Core capabilities:

1. Activity screening with NACE Rev. 2 code mapping (~240 activities, 13 sectors)
2. Taxonomy eligibility determination per delegated act and objective
3. Substantial Contribution assessment per activity per environmental objective
4. DNSH assessment matrix (activity x 5 remaining objectives)
5. Minimum Safeguards 4-topic company-level assessment (HR, anti-corruption, tax, competition)
6. KPI calculation engine (Turnover, CapEx, OpEx) with double-counting prevention
7. Green Asset Ratio (GAR) calculation (stock + flow) for credit institutions
8. Banking Book Taxonomy Alignment Ratio (BTAR) for extended FI scope
9. Article 8 disclosure template generation (EBA Pillar 3 Templates 6-10)
10. CSRD/ESRS integration and XBRL tagging support
11. Climate risk & vulnerability assessment workflow (DNSH adaptation)
12. Environmental delegated act objectives 3-6 (water, circular economy, pollution, biodiversity)
13. Enabling and transitional activity classification
14. CapEx plan management (5-year horizon)
15. De minimis threshold (10%) screening per Omnibus simplification
16. Data quality scoring and evidence management

### 1.2 EU Taxonomy Framework Mapping

#### 1.2.1 Six Environmental Objectives

| # | Environmental Objective | Delegated Act | Activities | Application Date |
|---|------------------------|---------------|------------|-----------------|
| 1 | Climate Change Mitigation | Climate DA 2021/2139 | ~101 | 1 Jan 2022 |
| 2 | Climate Change Adaptation | Climate DA 2021/2139 | ~105 | 1 Jan 2022 |
| 3 | Water & Marine Resources | Environmental DA 2023/2486 | ~6 | 1 Jan 2024 |
| 4 | Circular Economy | Environmental DA 2023/2486 | ~21 | 1 Jan 2024 |
| 5 | Pollution Prevention & Control | Environmental DA 2023/2486 | ~6 | 1 Jan 2024 |
| 6 | Biodiversity & Ecosystems | Environmental DA 2023/2486 | ~2 | 1 Jan 2024 |

#### 1.2.2 Four-Condition Alignment Test

| Condition | Scope | Assessment Level | Pass Criteria |
|-----------|-------|------------------|---------------|
| **Substantial Contribution** | Per activity, per objective | Activity-level | Meets quantitative/qualitative TSC thresholds |
| **DNSH** | Per activity, per non-SC objective | Activity-level | Passes all activity-specific DNSH criteria |
| **Minimum Safeguards** | All 4 topics (HR, corruption, tax, competition) | Company-level | Procedural + outcome test both pass |
| **TSC Compliance** | Per delegated act | Activity-level | Documented evidence of criteria met |

#### 1.2.3 KPI Framework

| KPI | Numerator | Denominator | Entities |
|-----|-----------|-------------|----------|
| **Turnover KPI** | Net turnover from aligned activities | Total net turnover (IAS 1.82a) | Non-financial undertakings |
| **CapEx KPI** | CapEx related to aligned activities | Total CapEx (IAS 16/38, IFRS 16, IAS 40) | Non-financial undertakings |
| **OpEx KPI** | OpEx related to aligned activities | Total OpEx (narrow: R&D, renovation, maintenance) | Non-financial undertakings |
| **GAR (Stock)** | Aligned on-BS covered assets | Total on-BS covered assets | Credit institutions |
| **GAR (Flow)** | New aligned originations | Total new originations | Credit institutions |
| **BTAR** | Extended aligned assets | Extended covered assets (incl. SMEs) | Credit institutions (voluntary) |
| **Investment KPI** | Aligned AUM | Total in-scope AUM | Asset managers |
| **Underwriting KPI** | Aligned gross premiums | Total non-life gross premiums | Insurance/reinsurance |

#### 1.2.4 13 Economic Sectors

| # | Sector | NACE Level 1 | Example Activities |
|---|--------|-------------|-------------------|
| 1 | Forestry | A | Afforestation, rehabilitation, forest management |
| 2 | Environmental Protection | -- | Wetland restoration |
| 3 | Manufacturing | C | Cement, aluminium, steel, hydrogen, batteries, vehicles |
| 4 | Energy | D | Solar PV, wind, hydro, geothermal, nuclear, gas, storage |
| 5 | Water/Waste | E | Water collection, sewerage, material recovery |
| 6 | Transport | H | Rail, road, maritime, cycling, infrastructure |
| 7 | Construction & Real Estate | F, L | New buildings, renovation, EPC rating |
| 8 | ICT | J | Data processing, GHG data solutions |
| 9 | Professional Services | M | R&D, energy performance |
| 10 | Financial & Insurance | K | Banking, insurance, asset management |
| 11 | Education | P | Education activities |
| 12 | Health & Social Work | Q | Health activities |
| 13 | Arts & Recreation | R | Ecotourism |

---

## 2. Backend Architecture

### 2.1 Service Engines (10)

| # | Engine | File | Responsibility |
|---|--------|------|----------------|
| 1 | **ActivityScreeningEngine** | `activity_screening_engine.py` | NACE mapping, ~240 activities, eligibility determination, sector classification, de minimis |
| 2 | **SubstantialContributionEngine** | `substantial_contribution_engine.py` | TSC threshold evaluation per activity per objective, enabling/transitional classification |
| 3 | **DNSHAssessmentEngine** | `dnsh_assessment_engine.py` | 6-objective DNSH matrix, climate risk assessment, water/circular/pollution/biodiversity checks |
| 4 | **MinimumSafeguardsEngine** | `minimum_safeguards_engine.py` | 4-topic company-level assessment (HR, anti-corruption, tax, competition), procedural + outcome tests |
| 5 | **KPICalculationEngine** | `kpi_calculation_engine.py` | Turnover/CapEx/OpEx KPI, double-counting prevention, CapEx plans, objective disaggregation |
| 6 | **GARCalculationEngine** | `gar_calculation_engine.py` | GAR stock/flow, BTAR, exposure classification, asset class mapping, EBA templates |
| 7 | **AlignmentEngine** | `alignment_engine.py` | End-to-end alignment workflow orchestration (SC + DNSH + MS = aligned), portfolio alignment |
| 8 | **ReportingEngine** | `reporting_engine.py` | Article 8 templates, EBA Pillar 3, PDF/Excel export, XBRL support, CSRD/ESRS |
| 9 | **DataQualityEngine** | `data_quality_engine.py` | Completeness/accuracy/coverage scoring, evidence management, improvement plans |
| 10 | **RegulatoryUpdateEngine** | `regulatory_update_engine.py` | DA version tracking, TSC updates, Omnibus changes, transition management |

### 2.2 Configuration (config.py)

- 25+ enumerations: EnvironmentalObjective, ActivityType, AlignmentStatus, DNSHStatus, SafeguardTopic, KPIType, GARType, ExposureType, AssetClass, EPCRating, DelegatedAct, Sector, etc.
- TAXONOMY_ACTIVITIES: ~240 activities with NACE codes, sectors, TSC references
- SUBSTANTIAL_CONTRIBUTION_CRITERIA: Thresholds per activity per objective
- DNSH_CRITERIA: Matrix of activity x non-SC objective requirements
- MINIMUM_SAFEGUARD_TOPICS: 4 topics with procedural + outcome checklists
- SECTOR_DEFINITIONS: 13 sectors with NACE mappings
- GAR_EXPOSURE_TYPES: Corporate, retail mortgage, auto loan, project finance
- REGULATORY_JURISDICTIONS: EU-wide + national implementations
- TaxonomyAppConfig(BaseSettings) with env_prefix="TAXONOMY_APP_"

### 2.3 Domain Models (models.py)

40+ Pydantic v2 domain models including:
- Core: Organization, EconomicActivity, NACECode, TaxonomyActivity
- Screening: EligibilityResult, ActivityMatch, SectorBreakdown
- SC: SubstantialContributionAssessment, TSCEvaluation, ThresholdCheck
- DNSH: DNSHAssessment, ObjectiveDNSH, ClimateRiskAssessment, EvidenceItem
- Safeguards: MinimumSafeguardAssessment, TopicAssessment, ProceduralCheck, OutcomeCheck
- KPI: TurnoverKPI, CapExKPI, OpExKPI, KPISummary, ActivityFinancials, CapExPlan
- GAR: GreenAssetRatio, BankingBookRatio, ExposureBreakdown, CoveredAssets
- Alignment: AlignmentResult, PortfolioAlignment, ActivityAlignment
- Reporting: DisclosureReport, Article8Template, EBATemplate, XBRLOutput
- Request/Response: 20+ API request/response models

### 2.4 Setup (setup.py)

- TaxonomyPlatform class composing all 10 engines
- `create_app()` FastAPI factory with 16 OpenAPI tags
- `get_router()` for auth integration at `/api/v1/taxonomy`
- `run_full_alignment()` 6-step pipeline (screen -> SC -> DNSH -> MS -> align -> report)
- Health check and platform info endpoints
- CORS, middleware, request ID

### 2.5 Config YAML (taxonomy_config.yaml)

~900 lines covering: app metadata, environmental objectives, delegated acts, NACE sectors, activity catalog (top-level), TSC reference data, DNSH matrix structure, minimum safeguard topics, KPI definitions, GAR methodology, reporting templates, regulatory jurisdictions, data quality dimensions, default thresholds.

---

## 3. API Architecture

### 3.1 Routers (16 total, ~130 endpoints)

| # | Router | Prefix | Tag | Endpoints | Description |
|---|--------|--------|-----|-----------|-------------|
| 1 | `activity_routes.py` | /activities | Activities | 8 | Activity catalog, NACE search, sector browse |
| 2 | `screening_routes.py` | /screening | Screening | 7 | Eligibility screening, batch screening, de minimis |
| 3 | `sc_routes.py` | /substantial-contribution | Substantial Contribution | 8 | SC assessment, TSC thresholds, enabling/transitional |
| 4 | `dnsh_routes.py` | /dnsh | DNSH | 9 | DNSH matrix, climate risk, water, circular, pollution, biodiversity |
| 5 | `safeguards_routes.py` | /safeguards | Minimum Safeguards | 7 | 4-topic assessment, procedural, outcome, company-level |
| 6 | `kpi_routes.py` | /kpi | KPI Calculation | 9 | Turnover/CapEx/OpEx, double-counting, CapEx plans, disaggregation |
| 7 | `gar_routes.py` | /gar | GAR/BTAR | 10 | GAR stock/flow, BTAR, exposure mapping, EBA templates |
| 8 | `alignment_routes.py` | /alignment | Alignment | 8 | Full alignment workflow, portfolio alignment, batch |
| 9 | `reporting_routes.py` | /reports | Reporting | 8 | Article 8, EBA Pillar 3, PDF/Excel, XBRL |
| 10 | `portfolio_routes.py` | /portfolios | Portfolio | 8 | Portfolio CRUD, holdings, upload, exposure management |
| 11 | `dashboard_routes.py` | /dashboard | Dashboard | 6 | Executive KPIs, alignment overview, sector breakdown |
| 12 | `data_quality_routes.py` | /data-quality | Data Quality | 6 | Quality scoring, evidence, completeness, improvement |
| 13 | `regulatory_routes.py` | /regulatory | Regulatory | 5 | DA versions, TSC updates, Omnibus tracking |
| 14 | `gap_routes.py` | /gap-analysis | Gap Analysis | 7 | Alignment gaps, DNSH gaps, safeguard gaps, action plans |
| 15 | `settings_routes.py` | /settings | Settings | 8 | Org config, reporting periods, thresholds, MRV mapping |
| 16 | `api/__init__.py` | -- | -- | -- | Router aggregation and exports |

---

## 4. Database Migration (V088)

### 4.1 Schema: `taxonomy_app`, Prefix: `gl_tax_`

| # | Table | Purpose |
|---|-------|---------|
| 1 | `gl_tax_organizations` | Organization master data |
| 2 | `gl_tax_economic_activities` | ~240 activity catalog |
| 3 | `gl_tax_nace_mappings` | NACE Rev. 2 to activity mapping |
| 4 | `gl_tax_eligibility_screenings` | Screening results per org per period |
| 5 | `gl_tax_screening_results` | Per-activity eligibility results |
| 6 | `gl_tax_sc_assessments` | Substantial contribution assessments |
| 7 | `gl_tax_tsc_evaluations` | Individual TSC threshold evaluations |
| 8 | `gl_tax_dnsh_assessments` | DNSH assessment headers |
| 9 | `gl_tax_dnsh_objective_results` | Per-objective DNSH results |
| 10 | `gl_tax_climate_risk_assessments` | Climate risk & vulnerability (DNSH adaptation) |
| 11 | `gl_tax_minimum_safeguard_assessments` | Company-level safeguard assessments |
| 12 | `gl_tax_safeguard_topic_results` | Per-topic (4) assessment results |
| 13 | `gl_tax_kpi_calculations` | KPI calculation headers |
| 14 | `gl_tax_activity_financials` | Per-activity financial data (turnover/CapEx/OpEx) |
| 15 | `gl_tax_capex_plans` | 5-year CapEx plans |
| 16 | `gl_tax_gar_calculations` | GAR/BTAR calculation results |
| 17 | `gl_tax_exposures` | Portfolio exposure data |
| 18 | `gl_tax_alignment_results` | End-to-end alignment results |
| 19 | `gl_tax_portfolio_alignments` | Portfolio-level alignment summaries |
| 20 | `gl_tax_reports` | Generated reports metadata |
| 21 | `gl_tax_evidence_items` | Evidence documents for assessments |
| 22 | `gl_tax_regulatory_versions` | Delegated act version tracking |
| 23 | `gl_tax_data_quality_scores` | Data quality assessments |
| 24 | `gl_tax_gap_assessments` | Gap analysis results |
| 25 | `gl_tax_gap_items` | Individual gap items |

- 3 hypertables: `gl_tax_sc_assessments` (assessment_date, 3-month), `gl_tax_gar_calculations` (calculation_date, 3-month), `gl_tax_alignment_results` (alignment_date, 3-month)
- 2 continuous aggregates: `gl_tax_quarterly_alignment_summary`, `gl_tax_annual_gar_trends`
- ~170 indexes (B-tree + GIN on JSONB)
- 35 security permissions
- 20 update triggers
- 3 retention policies (10-year), 3 compression policies (90-day)

---

## 5. Frontend Architecture

### 5.1 Pages (14)

| # | Page | Components | Purpose |
|---|------|-----------|---------|
| 1 | Dashboard | AlignmentOverview, KPISummaryCards, SectorBreakdownChart, EligibleVsAligned, TimelineTrend | Executive overview |
| 2 | ActivityScreening | NACESearch, ActivityCatalog, EligibilityMatrix, SectorFilter, BatchUpload | Activity eligibility screening |
| 3 | SubstantialContribution | TSCChecklist, ThresholdEvaluator, ObjectiveSelector, EvidenceUpload, SCResult | SC assessment |
| 4 | DNSHAssessment | ObjectiveMatrix, ClimateRiskWizard, WaterAssessment, CircularAssessment, PollutionCheck, BiodiversityCheck | DNSH evaluation |
| 5 | MinimumSafeguards | TopicAssessment, ProceduralChecklist, OutcomeMonitor, DueDiligenceTracker | MS evaluation |
| 6 | KPICalculator | TurnoverKPI, CapExKPI, OpExKPI, DoubleCountingGuard, ObjectiveBreakdown, CapExPlanManager | KPI calculation |
| 7 | GARCalculator | GARStock, GARFlow, BTARCalculator, ExposureClassifier, EBATemplatePreview | FI GAR/BTAR |
| 8 | AlignmentWorkflow | AlignmentStepper, ActivityStatus, PortfolioView, BatchAlignment | End-to-end alignment |
| 9 | Reporting | TemplateSelector, Article8Preview, EBAPillar3, ExportDialog, XBRLPreview | Report generation |
| 10 | PortfolioManagement | PortfolioTable, HoldingsEditor, ExposureUpload, CounterpartySearch | Portfolio CRUD |
| 11 | DataQuality | QualityScorecard, CompletenessMatrix, EvidenceTracker, ImprovementActions | Data quality |
| 12 | GapAnalysis | GapHeatmap, ActionPlan, DNSHGaps, SafeguardGaps, PriorityMatrix | Gap assessment |
| 13 | RegulatoryUpdates | DAVersionTracker, TSCChangeLog, OmnibusTimeline, TransitionPlanner | Regulation tracking |
| 14 | Settings | OrgConfig, ReportingPeriods, Thresholds, MRVMapping, Notifications | Configuration |

### 5.2 Redux Slices (14)

dashboard, activities, screening, substantialContribution, dnsh, safeguards, kpi, gar, alignment, reporting, portfolio, dataQuality, regulatory, settings

### 5.3 Shared Components

Layout (Header, Sidebar, Layout), Common (DataTable, LoadingSpinner, ScoreGauge, StatusBadge, ProgressBar, EvidenceUploader)

---

## 6. Development Tasks

### Group A: Backend Foundation (Parallel Agent 1)
- A1: config.py (~1,800 lines) -- 25+ enums, ~240 activity catalog, TSC refs, DNSH matrix, sector defs, GAR types
- A2: models.py (~2,000 lines) -- 40+ Pydantic v2 models across 10 domains

### Group B: Service Engines (Parallel Agents 2-3)
- B1: activity_screening_engine.py (~850 lines) -- NACE mapping, eligibility, sector classification
- B2: substantial_contribution_engine.py (~900 lines) -- TSC threshold evaluation, enabling/transitional
- B3: dnsh_assessment_engine.py (~950 lines) -- 6-objective DNSH matrix, evidence
- B4: minimum_safeguards_engine.py (~750 lines) -- 4-topic, procedural + outcome
- B5: kpi_calculation_engine.py (~850 lines) -- Turnover/CapEx/OpEx, double-counting
- B6: gar_calculation_engine.py (~900 lines) -- GAR stock/flow, BTAR, EBA
- B7: alignment_engine.py (~800 lines) -- Orchestration, portfolio alignment
- B8: reporting_engine.py (~850 lines) -- Article 8, EBA Pillar 3, XBRL
- B9: data_quality_engine.py (~750 lines) -- Quality scoring, evidence mgmt
- B10: regulatory_update_engine.py (~650 lines) -- DA versions, Omnibus tracking

### Group C: Setup & Config (Parallel Agent 4)
- C1: setup.py (~800 lines) -- TaxonomyPlatform facade, create_app(), get_router()
- C2: __init__.py (~80 lines) -- Package exports
- C3: taxonomy_config.yaml (~900 lines) -- Full configuration

### Group D: API Routers (Parallel Agent 5)
- D1-D16: 16 router files + api/__init__.py (~8,500 lines total, ~130 endpoints)

### Group E: Frontend (Parallel Agent 6)
- E1: Build configs, types, API service, store, utils
- E2: 14 Redux slices
- E3: Layout + common components
- E4: 60+ domain components across 14 feature areas
- E5: 14 pages + App.tsx + main.tsx

### Group F: Tests + Migration (Parallel Agent 7)
- F1: conftest.py (~1,300 lines) -- 30+ fixtures
- F2: 16 test modules (~6,000 lines total)
- F3: test_api_routes.py (~1,000 lines) -- All router tests
- F4: V088__taxonomy_app_service.sql (~1,700 lines) -- Full migration

### Group G: Auth Integration (Direct)
- G1: auth_setup.py -- Taxonomy import + router registration
- G2: route_protector.py -- ~140 taxonomy permission entries

---

## 7. Acceptance Criteria

- [ ] ~240 EU Taxonomy activities loaded with NACE codes across 6 objectives
- [ ] Substantial Contribution assessment with quantitative TSC thresholds
- [ ] DNSH matrix evaluation for all 6 objectives per activity
- [ ] Minimum Safeguards 4-topic company-level assessment
- [ ] KPI calculation (Turnover, CapEx, OpEx) with double-counting prevention
- [ ] GAR stock/flow and BTAR calculation for credit institutions
- [ ] Article 8 disclosure template generation
- [ ] EBA Pillar 3 Templates 6-10 for financial institutions
- [ ] Evidence management and audit trail
- [ ] De minimis threshold (10%) screening
- [ ] Climate risk & vulnerability assessment workflow
- [ ] Data quality scoring across 5 dimensions
- [ ] Cross-framework references (CSRD/ESRS, IFRS S2)
- [ ] 14 frontend pages with full component coverage
- [ ] 18 test files with 400+ test cases
- [ ] V088 migration with 25 tables, 3 hypertables, 2 continuous aggregates

---

## 8. Framework Alignment

| Framework | Integration Point |
|-----------|------------------|
| **CSRD/ESRS E1** | Taxonomy KPIs embedded in sustainability statement |
| **IFRS S2** | Taxonomy disclosures cross-referenced |
| **EBA Pillar 3** | GAR/BTAR Templates 6-10 |
| **GHG Protocol** | Scope 1/2/3 emissions feed into TSC evaluation |
| **TCFD** | Scenario analysis informs climate risk DNSH |
| **CDP** | Climate questionnaire references Taxonomy alignment |
| **SFDR** | Fund-level Taxonomy alignment for Article 8/9 funds |

---

## 9. Regulatory References

- EU Taxonomy Regulation (EU) 2020/852
- Climate Delegated Act (EU) 2021/2139 (amended 2023)
- Environmental Delegated Act (EU) 2023/2486
- Complementary Climate Delegated Act (EU) 2022/1214 (nuclear & gas)
- Disclosures Delegated Act (EU) 2021/2178 (Article 8 reporting)
- Simplification Delegated Act (adopted July 2025, effective 28 Jan 2026)
- EBA ITS on Pillar 3 ESG Risks Disclosure
- Platform on Sustainable Finance -- Minimum Safeguards Report (Oct 2022)
