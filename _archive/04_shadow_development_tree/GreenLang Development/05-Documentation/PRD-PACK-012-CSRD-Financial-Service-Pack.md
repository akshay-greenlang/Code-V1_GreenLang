# PRD: PACK-012 — CSRD Financial Service Pack

**Pack ID:** PACK-012-csrd-financial-service
**Category:** EU Compliance / CSRD
**Tier:** Sector-Specific (Financial Services)
**Version:** 1.0.0
**Status:** APPROVED
**Author:** GreenLang Product Team
**Date:** 2026-03-15

---

## 1. Executive Summary

PACK-012 is a **sector-specific CSRD Solution Pack** purpose-built for financial institutions — banks, insurance companies, asset managers, and investment firms. While PACK-001/002/003 provide general CSRD compliance, financial services have unique reporting requirements that demand specialized engines:

- **Financed Emissions (PCAF)**: Scope 3 Category 15 typically represents 95-99% of a financial institution's total emissions
- **Green Asset Ratio (GAR)**: Credit institutions must disclose taxonomy alignment of covered assets
- **Banking Book Taxonomy Alignment Ratio (BTAR)**: Extended taxonomy alignment including non-CSRD counterparties
- **Insurance Underwriting Emissions**: "Follow the risk" attribution for underwritten activities
- **Climate Risk Stress Testing**: NGFS scenario analysis required by ECB/EBA
- **EBA Pillar 3 ESG Disclosures**: Binding ITS for ESG risk disclosure

PACK-012 reuses ~75-80% of existing platform components and builds 20-25% net-new financial-services-specific functionality.

---

## 2. Regulatory Context

### Primary Regulations
| Regulation | Reference | Relevance |
|------------|-----------|-----------|
| CSRD | Directive (EU) 2022/2464 | Core sustainability reporting directive |
| ESRS Set 1 | Delegated Regulation (EU) 2023/2772 | 12 sector-agnostic standards |
| EU Taxonomy | Regulation (EU) 2020/852 | Green classification system |
| Taxonomy DA for FI | Delegated Regulation (EU) 2021/2178 Art 10 | GAR/BTAR/investment KPIs |
| SFDR | Regulation (EU) 2019/2088 | Fund-level sustainability disclosure |
| CRR/CRD VI | Regulation (EU) 575/2013 (amended) | Pillar 3 ESG risk disclosures |
| Solvency II | Directive 2009/138/EC | Insurance ESG risk integration |
| EBA Pillar 3 ESG ITS | EBA/ITS/2022/01 (updated 2025) | Binding ESG disclosure standards for banks |
| PCAF Standard | Global GHG Accounting Standard 3rd Ed (Dec 2025) | Financed emissions methodology |
| SBTi-FI | Financial Sector Guidance v1.2 | Science-based targets for FIs |
| NGFS Scenarios | Network for Greening the Financial System | Climate stress testing framework |

### Omnibus I Impact (Published 2026-02-26)
- Revised thresholds: >1,000 employees AND >EUR 450M net turnover
- Wave 2 postponed to FY2027 (publication 2028) via Stop-The-Clock Regulation
- Sector-specific ESRS work paused; FIs use sector-agnostic ESRS + PCAF
- Financial holding companies exempt (subsidiaries may still be subject)

### Financial Institution Types Covered
1. **Credit Institutions** (banks): GAR, BTAR, Pillar 3, PCAF, NGFS stress testing
2. **Insurance Undertakings**: Underwriting emissions, Solvency II ESG, investment portfolio
3. **Asset Managers**: SFDR integration, portfolio carbon footprint, sustainable investment
4. **Investment Firms**: MiFID II sustainability preferences, product governance

---

## 3. Architecture

### Component Summary
| Component | Count | Description |
|-----------|-------|-------------|
| Engines | 8 | FI-specific calculation engines |
| Workflows | 8 | FI-specific compliance workflows |
| Templates | 8 | FI-specific disclosure templates |
| Integrations | 10 | Bridges to existing platform + external data |
| Presets | 6 | Bank, Insurance, Asset Manager, Investment Firm, Pension Fund, Conglomerate |
| Tests | ~17 | conftest + manifest + config + 8 engines + workflows + templates + integrations + e2e + agent |

### Dependency Chain
```
PACK-012 (Financial Services)
  ├── PACK-001/002/003 (CSRD Starter/Professional/Enterprise) -- ESRS core
  ├── PACK-010/011 (SFDR Article 8/9) -- Portfolio carbon, PAI, taxonomy
  ├── PACK-008 (EU Taxonomy Alignment) -- Taxonomy computation
  ├── AGENT-MRV-028 (Investments Agent) -- PCAF calculation stack
  ├── greenlang.agents.finance (9 agents) -- Investment screening, green bonds, stranded assets
  ├── greenlang.agents.decarbonization (transition risk) -- Transition risk assessment
  └── greenlang.agents.adaptation (physical risk) -- Physical risk screening
```

---

## 4. Engines (8)

### Engine 1: FinancedEmissionsEngine
**Purpose:** Calculate Scope 3 Category 15 financed emissions using PCAF methodology across all asset classes.

**Key Features:**
- 10 PCAF asset classes: listed equity, corporate bonds, business loans, project finance, commercial real estate, mortgages, motor vehicle loans, sovereign bonds, securitizations, sub-sovereign debt
- Attribution factor calculation: Outstanding Amount / EVIC (or alternatives per asset class)
- Data quality scoring (PCAF Score 1-5 per asset class)
- Weighted average data quality score
- Sector-level and portfolio-level aggregation
- Double-counting prevention for equity vs. debt holdings
- Multi-currency normalization
- Year-over-year emissions trajectory
- SHA-256 provenance on all results

**Key Classes:** `FinancedEmissionsEngine`, `FinancedEmissionsConfig`, `AssetClassData`, `AttributionResult`, `PortfolioEmissionsResult`, `DataQualityScore`, `PCafAssetClass`, `EmissionsByAssetClass`

### Engine 2: InsuranceUnderwritingEngine
**Purpose:** Calculate associated emissions from insurance underwriting portfolios using PCAF Part C "follow the risk" methodology.

**Key Features:**
- Premium-based attribution (premium share of policyholder, not capital flows)
- Coverage: commercial motor, personal motor, commercial property, general liability, project insurance, treaty reinsurance
- Line-of-business emission factors
- Gross vs. net (of reinsurance) calculation
- Claims-linked emissions tracking
- Physical risk overlay on underwriting portfolio
- Industry sector classification (NACE/SIC)
- Provenance hashing

**Key Classes:** `InsuranceUnderwritingEngine`, `UnderwritingConfig`, `PolicyData`, `UnderwritingEmissionsResult`, `LineOfBusinessResult`, `ReinsuranceAdjustment`, `ClaimsEmissions`

### Engine 3: GreenAssetRatioEngine
**Purpose:** Calculate the Green Asset Ratio (GAR) for credit institutions per EU Taxonomy Delegated Act.

**Key Features:**
- GAR formula: Taxonomy-Aligned Assets / Total Covered Assets
- Covered assets: loans, debt securities, equity instruments
- Excluded exposures: sovereign, central bank, trading book, interbank
- Turnover-based, CapEx-based, and OpEx-based GAR variants
- Stock vs. flow GAR (new origination alignment)
- Breakdown by environmental objective (6 objectives)
- Breakdown by counterparty type (NFC, FC, household, local government)
- Enabling/transitional activity identification
- Off-balance-sheet GAR (guarantees, AuM)
- BTAR extension for full banking book
- Provenance hashing

**Key Classes:** `GreenAssetRatioEngine`, `GARConfig`, `CoveredAssetData`, `GARResult`, `GARBreakdown`, `CounterpartyTypeBreakdown`, `OffBalanceSheetKPI`, `FlowGAR`

### Engine 4: BTARCalculatorEngine
**Purpose:** Calculate the Banking Book Taxonomy Alignment Ratio (BTAR) extending GAR to the entire banking book.

**Key Features:**
- Full banking book coverage (including SMEs, non-EU, derivatives, intangibles)
- Estimation methodology for non-CSRD counterparties
- Sector proxy alignment estimation
- Internal ESG assessment integration
- Third-party data quality tracking
- BTAR vs. GAR reconciliation
- Data coverage and estimation ratios
- Provenance hashing

**Key Classes:** `BTARCalculatorEngine`, `BTARConfig`, `BankingBookData`, `BTARResult`, `EstimationMethodology`, `SectorProxyResult`, `DataCoverageReport`, `BTARvsGARReconciliation`

### Engine 5: ClimateRiskScoringEngine
**Purpose:** Score physical and transition climate risk for FI portfolios using NGFS scenarios and EBA/ECB methodology.

**Key Features:**
- Physical risk: acute (flood, wildfire, storm) and chronic (sea level rise, heat stress, drought)
- Transition risk: policy, technology, market, reputation, legal
- NGFS 6 scenario categories: Net Zero 2050, Below 2C, Divergent Net Zero, Delayed Transition, NDCs, Current Policies
- Time horizons: short (1-3yr), medium (3-10yr), long (10-30yr)
- Sector heatmap (NACE code mapping)
- Collateral physical risk (real estate, infrastructure)
- Credit risk impact: PD uplift, LGD adjustment, expected loss
- Stranded asset identification
- Composite risk score (0-100)
- Provenance hashing

**Key Classes:** `ClimateRiskScoringEngine`, `ClimateRiskConfig`, `ExposureData`, `ClimateRiskResult`, `PhysicalRiskScore`, `TransitionRiskScore`, `NGFSScenarioResult`, `StrandedAssetExposure`, `CreditRiskImpact`

### Engine 6: FSDoubleMaterialityEngine
**Purpose:** Conduct double materiality assessment specific to financial institutions, capturing impact through financed/insured/invested activities.

**Key Features:**
- Financial materiality: credit risk from climate, stranded asset risk, physical risk to collateral, regulatory capital impact
- Impact materiality: financed emissions, insured emissions, advisory impact, product design effects
- Stakeholder mapping for FI (regulators, investors, customers, communities affected by financed projects)
- IRO (Impact, Risk, Opportunity) identification per ESRS
- Materiality scoring matrix (severity x likelihood x scope)
- FI-specific sustainability matters (financial inclusion, responsible lending, fair pricing, data protection)
- Cross-reference to ESRS datapoints (E1, E4, S1-S4, G1)
- Provenance hashing

**Key Classes:** `FSDoubleMaterialityEngine`, `FSMaterialityConfig`, `MaterialityTopicData`, `FSMaterialityResult`, `IROAssessment`, `FinancedImpactAssessment`, `StakeholderInput`, `MaterialityMatrix`, `DatapointMapping`

### Engine 7: FSTransitionPlanEngine
**Purpose:** Assess and score financial institution transition plans against ESRS E1, SBTi-FI, and NZBA/NZAOA commitments.

**Key Features:**
- Portfolio decarbonization pathway assessment
- SBTi-FI target-setting: Sectoral Decarbonization Approach, Portfolio Coverage, Temperature Rating
- NZBA/NZAOA commitment tracking
- Sector targets: power (gCO2/kWh), real estate (kgCO2/m2), oil & gas, transport, steel, cement
- CapEx alignment trajectory
- Client engagement strategy assessment
- Financing phase-out commitments (coal, oil, gas)
- Intermediate targets (2025, 2030, 2035, 2040, 2050)
- Transition plan credibility score
- Provenance hashing

**Key Classes:** `FSTransitionPlanEngine`, `TransitionPlanConfig`, `SectorTargetData`, `TransitionPlanResult`, `SBTiFIAssessment`, `NZBACommitment`, `SectorDecarbPath`, `PhaseOutCommitment`, `CredibilityScore`

### Engine 8: Pillar3ESGEngine
**Purpose:** Calculate EBA Pillar 3 ESG risk disclosures per binding ITS for credit institutions.

**Key Features:**
- Template 1: Banking book - Climate change transition risk (by sector, PD, maturity)
- Template 2: Banking book - Climate change physical risk (by geography, hazard type)
- Template 3: Real estate collateral (EPC labels, energy efficiency)
- Template 4: Top 20 carbon-intensive exposures
- Template 5: Taxonomy alignment (GAR/BTAR KPIs)
- Template 6: Other ESG risks (biodiversity, social, governance)
- Template 10: Qualitative information on ESG risk
- Transition risk exposure concentration (by NACE sector)
- Physical risk geolocation analysis
- Maturity mismatch analysis
- Risk-weighted exposure alignment
- Provenance hashing

**Key Classes:** `Pillar3ESGEngine`, `Pillar3Config`, `BankingBookExposure`, `Pillar3Result`, `TransitionRiskTemplate`, `PhysicalRiskTemplate`, `RealEstateTemplate`, `Top20CarbonExposure`, `TaxonomyAlignmentTemplate`, `QualitativeDisclosure`

---

## 5. Workflows (8)

### Workflow 1: FinancedEmissionsWorkflow (5-phase)
Phases: DataCollection → AttributionCalculation → QualityAssessment → Aggregation → Reporting

### Workflow 2: GARBTARWorkflow (4-phase)
Phases: AssetClassification → AlignmentAssessment → KPIComputation → DisclosureGeneration

### Workflow 3: InsuranceEmissionsWorkflow (4-phase)
Phases: PolicyDataIngestion → EmissionAttribution → ReinsuranceAdjustment → ReportGeneration

### Workflow 4: ClimateStressTestWorkflow (5-phase)
Phases: ExposureMapping → ScenarioSelection → RiskCalculation → ImpactQuantification → ReportGeneration

### Workflow 5: FSMaterialityWorkflow (4-phase)
Phases: StakeholderEngagement → ImpactAssessment → FinancialAssessment → MatrixGeneration

### Workflow 6: TransitionPlanWorkflow (4-phase)
Phases: BaselineAssessment → TargetSetting → PathwayModeling → CredibilityScoring

### Workflow 7: Pillar3ReportingWorkflow (4-phase)
Phases: DataExtraction → TemplatePopulation → QualityValidation → FilingPreparation

### Workflow 8: RegulatoryIntegrationWorkflow (3-phase)
Phases: RequirementMapping → CrossReferenceAlignment → GapAnalysis

---

## 6. Templates (8)

### Template 1: PCAFReportTemplate
PCAF-compliant financed emissions disclosure with asset class breakdown, data quality scores, year-over-year comparison, and sector attribution.

### Template 2: GARBTARReportTemplate
EU Taxonomy Article 8 delegated act reporting for credit institutions with GAR by environmental objective, counterparty type, flow/stock split, and BTAR reconciliation.

### Template 3: Pillar3ESGTemplate
EBA Pillar 3 ESG ITS template covering all 10+ quantitative and qualitative templates for CRR-compliant banks.

### Template 4: ClimateRiskReportTemplate
TCFD-aligned climate risk report with physical risk heatmaps, transition risk sector analysis, NGFS scenario results, and stress test outcomes.

### Template 5: FSESRSChapterTemplate
Financial-services-specific ESRS chapter generator covering E1 (with financed emissions), S1-S4 (financial inclusion, responsible lending), and G1 (board ESG governance).

### Template 6: FinancedEmissionsDashboard
Interactive portfolio emissions dashboard with WACI waterfall, asset class drill-down, data quality traffic light, and decarbonization trajectory chart.

### Template 7: InsuranceESGTemplate
Insurance-specific ESG disclosure covering underwriting emissions, Solvency II ESG risk integration, and responsible underwriting policies.

### Template 8: SBTiFIReportTemplate
SBTi for Financial Institutions progress report with sector targets, portfolio coverage, temperature rating, and NZBA/NZAOA commitment tracking.

---

## 7. Integrations (10)

| # | Integration | Description |
|---|-------------|-------------|
| 1 | **PackOrchestrator** | 11-phase FI-specific pipeline orchestrator |
| 2 | **CSRDPackBridge** | Bridge to PACK-001/002/003 (ESRS core, consolidation, quality gates) |
| 3 | **SFDRPackBridge** | Bridge to PACK-010/011 (PAI, taxonomy alignment, portfolio carbon) |
| 4 | **TaxonomyPackBridge** | Bridge to PACK-008 (EU Taxonomy computation) |
| 5 | **MRVInvestmentsBridge** | Bridge to AGENT-MRV-028 (PCAF engine stack) |
| 6 | **FinanceAgentBridge** | Bridge to greenlang.agents.finance (screening, green bonds, stranded assets) |
| 7 | **ClimateRiskBridge** | Bridge to physical/transition risk agents |
| 8 | **EBAPillar3Bridge** | EBA Pillar 3 ITS data integration and export |
| 9 | **HealthCheck** | System health monitoring for all 8 engines + bridges |
| 10 | **SetupWizard** | FI-specific guided configuration (bank vs insurance vs asset manager) |

---

## 8. Presets (6)

| Preset | FI Type | Key Features |
|--------|---------|-------------|
| `bank.yaml` | Credit Institution | GAR, BTAR, Pillar 3, PCAF, NGFS stress testing |
| `insurance.yaml` | Insurance Company | Underwriting emissions, Solvency II, investment portfolio |
| `asset_manager.yaml` | Asset Management | SFDR integration, WACI, sustainable investment |
| `investment_firm.yaml` | Investment Firm | MiFID II, product governance, portfolio carbon |
| `pension_fund.yaml` | Pension Fund | IORP II, long-term stewardship, TCFD |
| `conglomerate.yaml` | Financial Conglomerate | Multi-entity, cross-sector, consolidated reporting |

---

## 9. Agent Orchestration

### Platform Agents Used
| Layer | Agents | Count |
|-------|--------|-------|
| MRV Scope 1-3 | AGENT-MRV 001-030 | 30 |
| Data Intake | AGENT-DATA 001-020 | 20 |
| Foundation | AGENT-FOUND 001-010 | 10 |
| Finance | GL-FIN-X 001-009 | 9 |
| Climate Risk | GL-DECARB, GL-ADAPT | 2 |
| Policy | GL-POL-X-007 (CSRD) | 1 |
| **Total** | | **72** |

---

## 10. Test Plan

### Test Categories
| Category | Files | Tests (est.) |
|----------|-------|-------------|
| Manifest validation | 1 | 25 |
| Configuration | 1 | 40 |
| Demo config | 1 | 20 |
| Engine 1: Financed Emissions | 1 | 35 |
| Engine 2: Insurance Underwriting | 1 | 30 |
| Engine 3: Green Asset Ratio | 1 | 35 |
| Engine 4: BTAR Calculator | 1 | 30 |
| Engine 5: Climate Risk Scoring | 1 | 35 |
| Engine 6: FS Double Materiality | 1 | 30 |
| Engine 7: FS Transition Plan | 1 | 30 |
| Engine 8: Pillar 3 ESG | 1 | 35 |
| Workflows | 1 | 30 |
| Templates | 1 | 25 |
| Integrations | 1 | 20 |
| E2E | 1 | 15 |
| Agent Integration | 1 | 10 |
| **Total** | **17** | **~465** |

---

## 11. Key Metrics & Acceptance Criteria

| Metric | Target |
|--------|--------|
| Test pass rate | 100% |
| Engine count | 8 |
| Workflow count | 8 |
| Template count | 8 |
| Integration count | 10 |
| Preset count | 6 |
| PCAF asset classes covered | 10 |
| Pillar 3 templates covered | 6+ |
| NGFS scenarios supported | 6 |
| Data quality score range | 1-5 (PCAF) |
| GAR/BTAR computation | Full Article 8 DA |
| File count target | ~73 |
| Total size target | ~2 MB |

---

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ESRS sector-specific FI standards paused | HIGH | Use sector-agnostic ESRS + PCAF + EBA ITS as foundation |
| Omnibus I threshold changes | MEDIUM | Config-driven thresholds, easy to update |
| PCAF 3rd edition methodology changes | MEDIUM | Engine parameterized by PCAF version |
| EBA Pillar 3 ITS updates | MEDIUM | Template-driven, version-controlled |
| SFDR 2.0 reclassification | LOW | Cross-pack bridge abstracts SFDR version |

---

## 13. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Research & PRD | Complete | This document |
| Package structure | Immediate | Directory + __init__.py files |
| Configuration + pack.yaml | Build | Config, presets, demo |
| Engines (8) | Build | All 8 calculation engines |
| Workflows (8) | Build | All 8 workflow orchestrators |
| Templates (8) | Build | All 8 disclosure templates |
| Integrations (10) | Build | All 10 bridges |
| Unit tests | Build | 17 test files, ~465 tests |
| Verification | Run | 100% pass rate |

---

## 14. Dependencies

### Required Packs (for cross-pack bridges)
- PACK-001/002/003: CSRD Starter/Professional/Enterprise
- PACK-008: EU Taxonomy Alignment
- PACK-010: SFDR Article 8
- PACK-011: SFDR Article 9

### Required GreenLang Agents
- AGENT-MRV-028: Investments Agent (PCAF)
- GL-FIN-X-001 through 009: Finance agents
- GL-DECARB-X-021: Transition risk
- GL-ADAPT-X-001: Physical risk
- GL-POL-X-007: CSRD compliance
- AGENT-DATA-020: Climate hazard connector
