# PRD-PACK-023: SBTi Alignment Pack

**Pack ID:** PACK-023-sbti-alignment
**Category:** Net Zero Packs
**Tier:** Standalone
**Version:** 1.0.0
**Status:** Approved
**Author:** GreenLang Product Team
**Date:** 2026-03-18
**Prerequisite:** None (standalone; enhanced with PACK-021/022 if present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Organizations pursuing Science Based Targets face significant challenges across the full SBTi lifecycle:

1. **Target Setting Complexity**: SBTi Corporate Manual V5.3 defines 28 near-term criteria (C1-C28) and the Net-Zero Standard V1.3 adds 14 net-zero criteria (NZ-C1 to NZ-C14). Manually validating compliance across all 42 criteria is error-prone and time-consuming.
2. **Pathway Selection**: Choosing between Absolute Contraction Approach (ACA) at 4.2%/yr for 1.5C and Sectoral Decarbonization Approach (SDA) for 12+ sectors requires deep technical understanding. Most companies default to ACA when SDA would be more appropriate.
3. **Scope 3 Screening**: The 40% materiality trigger and 67%/90% coverage requirements for near-term/long-term Scope 3 targets require systematic screening across all 15 categories.
4. **FLAG Integration**: Companies with >20% FLAG emissions must set separate FLAG targets covering 11 commodity categories with specific no-deforestation commitments.
5. **Financial Institution Requirements**: FIs face distinct requirements under FINZ V1.0 covering portfolio-level targets across 8+ asset classes with PCAF data quality scoring.
6. **Progress Tracking**: Annual progress against pathways requires variance analysis, recalculation triggers (M&A, divestitures), and five-year review cycles.
7. **Temperature Rating**: SBTi Temperature Rating v2.0 methodology with 6 portfolio aggregation methods (WATS/TETS/MOTS/EOTS/ECOTS/AOTS) is complex to implement correctly.
8. **Cross-Framework Alignment**: Companies must align SBTi targets with CDP (C4), TCFD (Metrics & Targets), CSRD (ESRS E1), and other frameworks simultaneously.

### 1.2 Solution Overview

PACK-023 is the **SBTi Alignment Pack** -- the third pack in the "Net Zero Packs" category. It provides a comprehensive, standalone SBTi lifecycle management solution with 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the entire journey from commitment to validated targets to ongoing progress tracking.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Capabilities

| Capability | Description |
|-----------|-------------|
| Target Setting | Near-term, long-term, and net-zero targets with ACA/SDA/FLAG pathways |
| Criteria Validation | Full 42-criterion automated assessment (C1-C28 + NZ-C1 to NZ-C14) |
| Scope 3 Screening | 15-category materiality screening with 40% trigger and coverage tracking |
| SDA Pathways | 12-sector intensity convergence with IEA NZE benchmarks |
| FLAG Assessment | 11 commodity categories with deforestation commitment validation |
| Temperature Rating | SBTi TR v2.0 with 6 portfolio aggregation methods |
| FI Module | FINZ V1.0 portfolio targets across 8 asset classes |
| Progress Tracking | Annual variance, recalculation triggers, five-year reviews |
| Framework Crosswalk | CDP/TCFD/CSRD/GHG Protocol/ISO 14064 alignment mapping |
| Submission Readiness | Pre-submission checklist and gap analysis for SBTi validation |

### 1.4 Target Users

**Primary:**
- Sustainability managers setting SBTi targets for the first time
- Climate strategy teams preparing SBTi target submissions
- Companies with committed targets needing validation readiness
- Organizations tracking progress against validated targets

**Secondary:**
- Financial institutions setting portfolio-level targets (FINZ)
- External consultants guiding clients through SBTi process
- Investor relations teams communicating target ambition
- Auditors validating SBTi-aligned climate disclosures

### 1.5 Success Metrics

| Metric | Target |
|--------|--------|
| Criteria validation time | <30 min (vs. 20+ hours manual) |
| SDA pathway accuracy | 100% match SBTi tool outputs |
| Scope 3 screening coverage | All 15 categories assessed |
| Temperature score accuracy | Within 0.1C of SBTi portal |
| FLAG assessment completeness | All 11 commodities covered |
| Submission readiness accuracy | 95% first-pass approval rate |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| SBTi Corporate Manual | V5.3 (2024) | 28 near-term criteria (C1-C28) |
| SBTi Corporate Net-Zero Standard | V1.3 (2024) | 14 net-zero criteria (NZ-C1 to NZ-C14) |
| SBTi SDA Tool | V3.0 (2024) | 12-sector intensity convergence |
| SBTi FLAG Guidance | V1.1 (2022) | 11 commodity pathways |
| SBTi FINZ | V1.0 (2024) | Financial institution targets |
| SBTi Temperature Rating | V2.0 (2024) | Company and portfolio scoring |

### 2.2 Supporting Frameworks

| Framework | Reference | Pack Relevance |
|-----------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015) | Scope 1+2 methodology basis |
| GHG Protocol Scope 3 Standard | WRI/WBCSD (2011) | Activity-based Scope 3 |
| IPCC AR6 WG1/WG3 | IPCC (2021/2022) | Carbon budgets, GWP-100 |
| Paris Agreement | UNFCCC (2015) | 1.5C temperature target |
| ISO 14064-1:2018 | ISO | GHG quantification |
| CDP Climate Change | CDP (2024) | C4 targets module alignment |
| TCFD Recommendations | FSB/TCFD (2017) | Metrics & targets |
| CSRD ESRS E1 | EU (2023) | E1-4 GHG reduction targets |
| IEA Net Zero Roadmap | IEA (2023) | Sector pathway benchmarks |
| PCAF Global Standard | PCAF (2022) | Financed emissions data quality |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | SBTi lifecycle calculation engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report and dashboard templates |
| Integrations | 12 | Agent, app, and platform bridges |
| Presets | 8 | Sector-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `target_setting_engine.py` | Near-term, long-term, and net-zero target definition with ACA/SDA pathway selection, ambition level assessment (1.5C/WB2C/2C), and target boundary validation |
| 2 | `criteria_validation_engine.py` | Full 42-criterion automated validation: C1-C28 (near-term) + NZ-C1 to NZ-C14 (net-zero), with pass/fail/warning for each criterion and detailed remediation guidance |
| 3 | `scope3_screening_engine.py` | 15-category Scope 3 materiality screening with 40% trigger assessment, 67%/90% coverage tracking, supplier engagement target validation, and category prioritization |
| 4 | `sda_sector_engine.py` | SDA intensity convergence for 12 sectors (power, cement, steel, aluminium, pulp/paper, chemicals, aviation, maritime, road transport, buildings commercial, buildings residential, food/beverage) with IEA NZE benchmarks |
| 5 | `flag_assessment_engine.py` | FLAG emissions assessment for 11 commodities (cattle, soy, palm, timber, cocoa, coffee, rubber, rice, sugarcane, maize, wheat), 20% trigger evaluation, no-deforestation commitments, and FLAG pathway (3.03%/yr) |
| 6 | `temperature_rating_engine.py` | SBTi Temperature Rating v2.0 with company-level scoring (1.0-6.0C range), portfolio aggregation (WATS/TETS/MOTS/EOTS/ECOTS/AOTS), and contribution analysis |
| 7 | `progress_tracking_engine.py` | Annual progress against validated targets with on-track/off-track assessment, gap analysis, corrective action triggers, and trajectory projection |
| 8 | `recalculation_engine.py` | Base year recalculation for M&A, divestitures, methodology changes, and structural changes with 5% significance threshold and audit trail |
| 9 | `fi_portfolio_engine.py` | Financial institution portfolio targets per FINZ V1.0 covering 8 asset classes (listed equity, corporate bonds, business loans, mortgages, commercial RE, project finance, sovereign bonds, securitized), PCAF data quality scoring |
| 10 | `submission_readiness_engine.py` | Pre-submission checklist scoring, gap analysis, documentation completeness, data quality assessment, and estimated timeline to submission-ready status |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `target_setting_workflow.py` | 5: Inventory -> Screening -> PathwaySelect -> TargetDef -> Validate | End-to-end target setting |
| 2 | `validation_workflow.py` | 4: DataCollect -> CriteriaCheck -> GapAnalysis -> Report | Full validation assessment |
| 3 | `scope3_assessment_workflow.py` | 4: CategoryScreen -> MaterialityCalc -> CoverageCheck -> TargetDesign | Scope 3 target workflow |
| 4 | `sda_pathway_workflow.py` | 4: SectorClassify -> BenchmarkLoad -> ConvergenceCalc -> Validate | SDA sector pathway |
| 5 | `flag_workflow.py` | 4: CommodityAssess -> TriggerEval -> PathwayCalc -> CommitmentValidate | FLAG assessment workflow |
| 6 | `progress_review_workflow.py` | 5: DataUpdate -> ProgressCalc -> VarianceAnalysis -> Recalc -> Report | Annual progress review |
| 7 | `fi_target_workflow.py` | 4: PortfolioMap -> AssetClassTarget -> CoverageCalc -> Validate | FI portfolio targets |
| 8 | `full_sbti_lifecycle_workflow.py` | 8: Commitment -> Inventory -> TargetSet -> Validate -> Submit -> Track -> Review -> Revalidate | Full SBTi lifecycle |

### 3.4 Templates

| # | Template | Purpose |
|---|----------|---------|
| 1 | `target_summary_report.py` | Target definitions with pathway visualization and ambition assessment |
| 2 | `validation_report.py` | 42-criterion validation results with pass/fail matrix |
| 3 | `scope3_screening_report.py` | 15-category materiality analysis with coverage dashboard |
| 4 | `sda_pathway_report.py` | Sector-specific intensity convergence with benchmarks |
| 5 | `flag_assessment_report.py` | FLAG commodity assessment with commitment status |
| 6 | `temperature_rating_report.py` | Company and portfolio temperature scores |
| 7 | `progress_dashboard_report.py` | Annual progress against targets with trajectory |
| 8 | `fi_portfolio_report.py` | FI portfolio-level target and coverage report |
| 9 | `submission_package_report.py` | Complete SBTi submission package document |
| 10 | `framework_crosswalk_report.py` | CDP/TCFD/CSRD/GHG Protocol alignment mapping |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline with conditional FLAG/FI phases |
| 2 | `sbti_app_bridge.py` | Bridge to GL-SBTi-APP (14 engines, full API) |
| 3 | `mrv_bridge.py` | Routes to 30 MRV agents for emissions data |
| 4 | `ghg_app_bridge.py` | GL-GHG-APP for inventory management |
| 5 | `pack021_bridge.py` | PACK-021 baseline and gap analysis |
| 6 | `pack022_bridge.py` | PACK-022 scenarios, SDA, temperature scoring |
| 7 | `decarb_bridge.py` | 21 DECARB-X agents for reduction planning |
| 8 | `data_bridge.py` | 20 DATA agents for data intake |
| 9 | `reporting_bridge.py` | CDP/TCFD/CSRD/ISO cross-framework reporting |
| 10 | `offset_bridge.py` | Carbon credit management for net-zero neutralization |
| 11 | `health_check.py` | 20-category system verification |
| 12 | `setup_wizard.py` | 6-step guided SBTi configuration |

### 3.6 Presets

| # | Preset | Sector | Key Characteristics |
|---|--------|--------|---------------------|
| 1 | `power_generation.yaml` | Power/Utilities | SDA mandatory, tCO2e/MWh intensity, coal phase-out |
| 2 | `heavy_industry.yaml` | Cement/Steel/Aluminium | SDA mandatory, process emissions, CCS pathway |
| 3 | `manufacturing.yaml` | General Manufacturing | ACA or SDA, mixed scopes, energy efficiency |
| 4 | `transport.yaml` | Road/Aviation/Maritime | SDA for transport, fleet electrification, SAF |
| 5 | `financial_services.yaml` | Banks/Insurance/AM | FINZ V1.0, PCAF, portfolio temperature |
| 6 | `food_agriculture.yaml` | Food/Beverage/Agriculture | FLAG mandatory (>20%), 11 commodities, land use |
| 7 | `real_estate.yaml` | Buildings/Construction | SDA buildings, CRREM alignment, Scope 3 Cat 13 |
| 8 | `technology.yaml` | Technology/Software | ACA pathway, RE100, data center PUE |

---

## 4. Engine Specifications

### 4.1 Engine 1: Target Setting Engine

**Purpose:** Define near-term, long-term, and net-zero targets with pathway selection.

**Key Features:**
- Target types: NEAR_TERM (5-10yr), LONG_TERM (>2035), NET_ZERO (2050 max)
- Pathway methods: ACA (4.2%/yr 1.5C, 2.5%/yr WB2C), SDA (12 sectors), FLAG (3.03%/yr)
- Ambition levels: 1.5C, Well-Below-2C, 2C
- Scope coverage: S1+S2 (95% min), S3 (67% near-term, 90% long-term)
- Base year validation: must be >=2015, no older than 5 years for new submissions
- Target boundary: operational or financial control, equity share

**Key Models:**
- `TargetSettingInput` - Organization profile, emissions inventory, sector, preferences
- `TargetSettingResult` - Defined targets with pathways, milestones, ambition assessment
- `TargetDefinition` - Single target with scope, boundary, base year, target year, reduction %
- `PathwayMilestone` - Annual milestone point on reduction pathway
- `AmbitionAssessment` - Temperature alignment of selected targets

### 4.2 Engine 2: Criteria Validation Engine

**Purpose:** Validate targets against all 42 SBTi criteria.

**Near-Term Criteria (C1-C28):**
- C1-C4: Organization boundary and scope coverage
- C5-C8: Base year and emissions inventory
- C9-C12: Target ambition and pathway
- C13-C16: Scope 2 methodology and renewable energy
- C17-C20: Scope 3 materiality and targets
- C21-C24: Target timeframe and review
- C25-C28: Reporting and disclosure

**Net-Zero Criteria (NZ-C1 to NZ-C14):**
- NZ-C1-C4: Net-zero target definition
- NZ-C5-C8: Long-term target requirements
- NZ-C9-C11: Residual emissions and neutralization
- NZ-C12-C14: Transition planning and governance

**Key Models:**
- `ValidationInput` - All target data and supporting evidence
- `ValidationResult` - Overall pass/fail with criterion-level details
- `CriterionCheck` - Single criterion assessment (pass/fail/warning/NA)
- `GapItem` - Specific gap identified with remediation guidance

### 4.3 Engine 3: Scope 3 Screening Engine

**Purpose:** Screen all 15 Scope 3 categories for materiality and coverage.

**Key Calculations:**
- Total Scope 3 as % of total emissions (40% trigger threshold)
- Per-category emissions and % of total Scope 3
- Coverage calculation: categories targeted / total Scope 3 emissions
- Supplier engagement target assessment (for category-specific targets)
- Data quality scoring per category (primary/secondary/proxy/spend)

### 4.4 Engine 4: SDA Sector Engine

**Purpose:** Calculate SDA intensity convergence pathways for 12 sectors.

**SDA Convergence Formula:**
```
I(t) = I_sector(t) + (I_company(base) - I_sector(base))
       * ((I_sector(target) - I_sector(t)) / (I_sector(target) - I_sector(base)))
```

**12 Sectors with 2050 Benchmarks:**
1. Power: 0.014 tCO2e/MWh
2. Cement: 0.119 tCO2e/tonne
3. Steel: 0.156 tCO2e/tonne
4. Aluminium: 1.31 tCO2e/tonne
5. Pulp & Paper: 0.175 tCO2e/tonne
6. Chemicals: sector-specific
7. Aviation: sector-specific gCO2e/RPK
8. Maritime: sector-specific gCO2e/tkm
9. Road Transport: 5.3 gCO2e/pkm
10. Buildings (Commercial): 3.1 kgCO2e/m2
11. Buildings (Residential): 2.3 kgCO2e/m2
12. Food & Beverage: sector-specific

### 4.5 Engine 5: FLAG Assessment Engine

**Purpose:** Assess FLAG emissions and commodity-specific targets.

**11 FLAG Commodities:**
cattle, soy, palm_oil, timber, cocoa, coffee, rubber, rice, sugarcane, maize, wheat

**Key Calculations:**
- FLAG emissions as % of total (20% trigger threshold)
- Per-commodity emission allocation
- FLAG pathway: 3.03%/yr linear reduction
- No-deforestation commitment validation
- Land use change emissions quantification

### 4.6 Engine 6: Temperature Rating Engine

**Purpose:** SBTi Temperature Rating v2.0 implementation.

**Temperature Mapping:** Piecewise-linear mapping from Annual Reduction Rate (ARR) to temperature:
- 7.0%/yr -> 1.20C (most ambitious)
- 4.2%/yr -> 1.50C (1.5C aligned)
- 2.5%/yr -> 1.80C (Well-Below-2C)
- 0.0%/yr -> 3.20C (default/no target)

**6 Portfolio Aggregation Methods:**
- WATS: Weighted Average Temperature Score (by revenue)
- TETS: Total Emissions Temperature Score (by emissions)
- MOTS: Market Owned Temperature Score (by market cap ownership)
- EOTS: Enterprise Owned Temperature Score (by EV ownership)
- ECOTS: Enterprise Value + Cash Temperature Score
- AOTS: All-Owned Temperature Score (by total invested capital)

### 4.7 Engine 7: Progress Tracking Engine

**Purpose:** Track annual progress against validated targets.

**Key Features:**
- On-track/off-track/critical assessment with RAG status
- Gap analysis with corrective action triggers
- Trajectory projection (linear extrapolation)
- Budget remaining calculation
- Annualized reduction rate vs. required rate

### 4.8 Engine 8: Recalculation Engine

**Purpose:** Manage base year recalculations for structural changes.

**Recalculation Triggers:**
- Acquisition (>5% emissions impact)
- Divestiture (>5% emissions impact)
- Merger
- Methodology change (emission factor updates)
- Structural change (outsourcing, insourcing)
- Organic growth (>5% threshold)

### 4.9 Engine 9: FI Portfolio Engine

**Purpose:** Financial institution portfolio targets per FINZ V1.0.

**8 Asset Classes:**
listed_equity, corporate_bonds, business_loans, mortgages, commercial_real_estate, project_finance, sovereign_bonds, securitized

**Key Features:**
- Portfolio-level target setting per asset class
- PCAF data quality scoring (1-5 scale)
- Coverage calculation (% of portfolio with targets)
- Engagement target tracking
- Temperature alignment per asset class

### 4.10 Engine 10: Submission Readiness Engine

**Purpose:** Assess readiness for SBTi target submission.

**Readiness Dimensions:**
- Data completeness (emissions inventory quality)
- Criteria compliance (42-criterion check summary)
- Documentation readiness (supporting evidence)
- Governance readiness (board approval, public commitment)
- Timeline estimation (weeks to submission-ready)

---

## 5. Agent Dependencies

### 5.1 GL-SBTi-APP (14 engines)
All GL-SBTi-APP engines via `sbti_app_bridge.py`.

### 5.2 MRV Agents (30)
All 30 AGENT-MRV agents via `mrv_bridge.py`.

### 5.3 Decarbonization Agents (21)
All 21 DECARB-X agents via `decarb_bridge.py`.

### 5.4 Application Dependencies
- GL-GHG-APP, GL-SBTi-APP, GL-CDP-APP, GL-TCFD-APP

### 5.5 Data Agents (20)
All 20 AGENT-DATA agents via `data_bridge.py`.

### 5.6 Foundation Agents (10)
All 10 AGENT-FOUND agents.

### 5.7 Optional Pack Dependencies
- PACK-021 Net Zero Starter Pack (baseline, gap analysis)
- PACK-022 Net Zero Acceleration Pack (scenarios, SDA, temperature)

---

## 6. Performance Targets

| Metric | Target |
|--------|--------|
| Full 42-criterion validation | <5 minutes |
| SDA pathway calculation | <2 minutes per sector |
| Scope 3 screening (15 categories) | <3 minutes |
| FLAG assessment (11 commodities) | <2 minutes |
| Temperature scoring (single entity) | <30 seconds |
| Temperature scoring (50-entity portfolio) | <5 minutes |
| FI portfolio targets (8 asset classes) | <10 minutes |
| Full SBTi lifecycle workflow | <2 hours |
| Submission readiness assessment | <15 minutes |
| Memory ceiling | 4096 MB |
| Max entities (FI portfolio) | 500 |

---

## 7. Security Requirements

- JWT RS256 authentication
- RBAC with 6 roles: `sbti_admin`, `sustainability_manager`, `climate_analyst`, `fi_portfolio_manager`, `external_auditor`, `viewer`
- AES-256-GCM encryption at rest
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all outputs
- Full audit trail per SEC-005

---

## 8. Testing Requirements

| Test Type | Target |
|-----------|--------|
| Unit tests per engine | 40-60 tests |
| Integration tests | 20-30 tests |
| E2E pipeline tests | 10-15 tests |
| Config/preset tests | 20-30 tests |
| Total test count | 500+ |
| Test pass rate | 100% |
| Code coverage | 85%+ |
