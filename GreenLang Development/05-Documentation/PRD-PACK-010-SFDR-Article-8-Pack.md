# PRD-PACK-010: SFDR Article 8 Pack

**Document ID**: PRD-PACK-010
**Version**: 1.0.0
**Status**: Approved
**Author**: GreenLang Platform Team
**Date**: 2026-03-15
**Category**: EU Compliance > Solution Packs > Sustainable Finance

---

## 1. Executive Summary

PACK-010 delivers a comprehensive **SFDR Article 8 Pack** for financial market participants
(FMPs) managing financial products that promote environmental or social characteristics
under the EU Sustainable Finance Disclosure Regulation (SFDR). Article 8 products
("light green" funds) must demonstrate how they promote E/S characteristics, disclose
the proportion of sustainable investments, report Principal Adverse Impact (PAI)
indicators, and provide EU Taxonomy alignment ratios.

This pack bundles 8 calculation engines, 8 compliance workflows, 8 report templates,
and 10 integration bridges to deliver end-to-end SFDR Article 8 compliance. It
leverages existing GreenLang agents (30 MRV agents for emissions data, 10 data
agents, 10 foundation agents) and integrates tightly with PACK-008 (EU Taxonomy
Alignment) for taxonomy ratio calculations.

### Key Capabilities

- **18 mandatory PAI indicators** (Table 1 of Annex I RTS) with full calculation engines
- **Additional opt-in PAI indicators** from Table 2 (environment) and Table 3 (social)
- **Pre-contractual disclosures** (Annex II template) with product-level ESG strategy
- **Periodic reporting** (Annex IV template) with actual vs. commitment tracking
- **Website disclosures** (Annex III template) with ongoing update management
- **EU Taxonomy alignment ratio** calculation for Article 8 products
- **DNSH assessment** under SFDR RTS (distinct from Taxonomy DNSH)
- **Good governance checks** (Article 2(17)) for investee companies
- **Sustainable investment percentage** calculation and monitoring
- **European ESG Template (EET)** data management and export
- **Portfolio carbon footprint** (WACI) and GHG intensity metrics
- **Cross-framework alignment** with CSRD/ESRS, EU Taxonomy, TCFD, CDP

### Target Users

| User Type | Use Case |
|-----------|----------|
| Asset Managers | Fund-level Article 8 compliance for UCITS/AIF products |
| Insurance Companies | Unit-linked and pension product Article 8 disclosures |
| Banks | Structured product and portfolio mandate Article 8 compliance |
| Pension Funds | IORP Article 8 fund classification and reporting |
| Wealth Managers | Discretionary portfolio Article 8 alignment |
| Compliance Officers | PAI monitoring, disclosure review, regulatory filing |

---

## 2. Background & Regulatory Context

### 2.1 SFDR Overview

The **Sustainable Finance Disclosure Regulation** (Regulation (EU) 2019/2088) establishes
transparency rules for financial market participants regarding sustainability risks and
adverse impacts. It introduces a three-tier classification system:

| Classification | Description | Disclosure Level |
|---------------|-------------|------------------|
| Article 6 | Products without ESG integration | Basic sustainability risk disclosure |
| **Article 8** | **Products promoting E/S characteristics** | **Enhanced disclosure (Annexes II-IV)** |
| Article 9 | Products with sustainable investment objective | Maximum disclosure (Annexes II-V) |

### 2.2 Article 8 Specific Requirements

Article 8 products must meet the following core requirements under the SFDR RTS
(Commission Delegated Regulation (EU) 2022/1288):

1. **Pre-contractual Disclosure (Annex II)**: Product-level information on:
   - Environmental/social characteristics promoted
   - Investment strategy and binding elements
   - Proportion of sustainable investments (if applicable)
   - EU Taxonomy alignment percentage
   - DNSH assessment methodology
   - Good governance verification approach

2. **Website Disclosure (Annex III)**: Ongoing information on:
   - Summary of E/S characteristics promoted
   - Investment strategy details
   - Proportion of investments by category
   - Monitoring methodology
   - Data sources and limitations
   - Engagement policies

3. **Periodic Reporting (Annex IV)**: Annual/semi-annual report on:
   - Extent to which E/S characteristics were attained
   - Actual proportion of sustainable investments
   - Actual EU Taxonomy alignment
   - Top investments list
   - Sector/geographic allocation
   - PAI indicators considered

### 2.3 Principal Adverse Impact (PAI) Indicators

The 18 mandatory PAI indicators from Table 1 of Annex I RTS:

| # | Indicator | Category | Metric |
|---|-----------|----------|--------|
| 1 | GHG emissions | Climate | Scope 1, 2, 3 + Total (tCO2eq) |
| 2 | Carbon footprint | Climate | tCO2eq per EUR million invested |
| 3 | GHG intensity | Climate | tCO2eq per EUR million revenue |
| 4 | Fossil fuel exposure | Climate | % of investee revenue from fossil fuels |
| 5 | Non-renewable energy | Climate | % share non-renewable consumption/production |
| 6 | Energy intensity | Climate | GWh per EUR million revenue, per NACE sector |
| 7 | Biodiversity impact | Environment | Activities near biodiversity-sensitive areas |
| 8 | Water emissions | Environment | Tonnes of water pollutants discharged |
| 9 | Hazardous waste | Environment | Tonnes of hazardous/radioactive waste |
| 10 | UNGC/OECD violations | Social | Violations of UNGC principles or OECD Guidelines |
| 11 | UNGC/OECD monitoring | Social | Lack of compliance monitoring processes |
| 12 | Gender pay gap | Social | Average unadjusted gender pay gap |
| 13 | Board gender diversity | Social | Average female-to-male board ratio |
| 14 | Controversial weapons | Social | Exposure to controversial weapons (% of portfolio) |
| 15 | GHG intensity (sovereigns) | Sovereign | tCO2eq per EUR million GDP |
| 16 | Investee countries with social violations | Sovereign | Countries subject to social violations |
| 17 | Fossil fuel exposure (real estate) | Real Estate | Exposure through inefficient real estate |
| 18 | Energy efficiency (real estate) | Real Estate | Proportion of energy-inefficient assets |

### 2.4 EU Taxonomy Alignment in Article 8

Article 8 products that make sustainable investments must disclose:
- **Minimum taxonomy alignment commitment** (pre-contractual)
- **Actual taxonomy alignment** (periodic reporting)
- **Breakdown by environmental objective** (CCM, CCA, WTR, CE, PPC, BIO)
- **Taxonomy-eligible but not aligned** proportion
- **Pie chart visualization** (mandatory per RTS template)

### 2.5 Regulatory Timeline

| Date | Event |
|------|-------|
| Mar 2021 | SFDR Level 1 effective (entity-level disclosures) |
| Jan 2023 | SFDR Level 2 / RTS effective (product-level disclosures) |
| Jun 2023 | First periodic reports due under RTS |
| Apr 2024 | SFDR targeted consultation by European Commission |
| Nov 2025 | Expected SFDR Level 1 review / SFDR 2.0 proposal |
| H1 2027 | Expected SFDR 2.0 application (product categorization reform) |

### 2.6 Existing GreenLang Components

| Component | Relevance to SFDR Article 8 |
|-----------|---------------------------|
| PACK-008 (EU Taxonomy) | Taxonomy alignment ratios, DNSH, KPI engines |
| GL-Taxonomy-APP (APP-010) | Taxonomy alignment core engine |
| `green_investment_screener.py` | `SFDRClassification` enum, Article 8 criteria |
| AGENT-MRV-001 to 030 | GHG emissions for PAI indicators 1-3 |
| AGENT-DATA-001 to 020 | Data intake/quality for portfolio data |
| AGENT-FOUND-001 to 010 | Foundation agents (orchestration, evidence, etc.) |
| PACK-009 Bundle Bridge | Cross-framework mapping to SFDR |

---

## 3. Goals & Non-Goals

### Goals

1. Deliver complete SFDR Article 8 pre-contractual, website, and periodic disclosures
2. Calculate all 18 mandatory PAI indicators with auditable methodology
3. Integrate EU Taxonomy alignment reporting within Article 8 disclosures
4. Provide automated DNSH assessment under SFDR RTS methodology
5. Verify good governance for investee companies (Article 2(17))
6. Calculate sustainable investment percentage with sustainable objective tracking
7. Generate EET (European ESG Template) compliant data exports
8. Support portfolio-level carbon footprint (WACI) calculations
9. Enable cross-framework data flow with CSRD, EU Taxonomy, TCFD
10. Provide 250+ unit tests with 100% component coverage

### Non-Goals

1. Article 9 product disclosures (separate pack: PACK-011)
2. Entity-level PAI statements (separate from product-level)
3. Real-time market data integration (uses periodic data feeds)
4. Custom ESG scoring models (uses regulatory-defined indicators)
5. MiFID II suitability assessment integration
6. SFDR 2.0 product categorization (pending final regulation text)

---

## 4. Architecture

### 4.1 Component Overview

```
PACK-010-sfdr-article-8/
  pack.yaml                    # Pack manifest with agent references
  __init__.py
  config/
    pack_config.py             # SFDRArticle8Config (Pydantic v2)
    presets/                   # 5 presets (asset_manager, insurance, bank, pension, wealth)
    demo/demo_config.yaml      # Demo configuration
  engines/                     # 8 calculation engines
  workflows/                   # 8 compliance workflows
  templates/                   # 8 report templates
  integrations/                # 10 integration bridges
  tests/                       # 17 test files
```

### 4.2 Agent Dependencies

| Agent Layer | Count | Purpose |
|-------------|-------|---------|
| MRV Agents (001-030) | 30 | GHG emissions for PAI 1-6, carbon footprint |
| Data Agents (001-020) | 10 | Portfolio data intake, quality, validation |
| Foundation Agents (001-010) | 10 | Orchestration, evidence, reproducibility |
| PACK-008 Engines | 10 | EU Taxonomy alignment, DNSH, KPIs |
| **Pack-specific** | **34** | **8 engines + 8 workflows + 8 templates + 10 integrations** |
| **Total** | **~84** | |

---

## 5. Engines (8)

### 5.1 PAI Indicator Calculator Engine
- **File**: `engines/pai_indicator_calculator.py`
- **Purpose**: Calculate all 18 mandatory PAI indicators + optional indicators
- **Features**:
  - Portfolio-weighted PAI calculations (enterprise value/AUM basis)
  - Coverage ratio tracking per indicator
  - Data quality scoring (estimated vs. reported vs. verified)
  - Year-over-year comparison for periodic reporting
  - Sovereign PAI indicators (15-16) for government bond holdings
  - Real estate PAI indicators (17-18) for property exposures
- **Key Methods**: `calculate_all_pai()`, `calculate_single_pai(indicator_id)`, `get_coverage_ratios()`, `compare_periods(current, previous)`
- **Calculation**: `PAI = SUM(investee_metric * portfolio_weight) / total_invested`

### 5.2 Taxonomy Alignment Ratio Engine
- **File**: `engines/taxonomy_alignment_ratio.py`
- **Purpose**: Calculate EU Taxonomy alignment percentages for Article 8 products
- **Features**:
  - Fund-level taxonomy alignment ratio (by revenue/CapEx/OpEx)
  - Breakdown by environmental objective (CCM, CCA, WTR, CE, PPC, BIO)
  - Eligible vs. aligned vs. non-eligible breakdown
  - Minimum commitment tracking (pre-contractual vs. actual)
  - Double-counting prevention across objectives
  - Transitional and enabling activity handling
- **Key Methods**: `calculate_alignment_ratio()`, `breakdown_by_objective()`, `check_commitment_adherence()`, `generate_pie_chart_data()`

### 5.3 DNSH Assessment Engine (SFDR)
- **File**: `engines/sfdr_dnsh_engine.py`
- **Purpose**: SFDR-specific DNSH assessment distinct from Taxonomy Regulation DNSH
- **Features**:
  - PAI indicator-based DNSH screening (mandatory PAI consideration)
  - Environmental and social minimum safeguards
  - Positive contribution vs. no significant harm threshold
  - Investment-level DNSH pass/fail determination
  - Aggregate portfolio DNSH compliance scoring
- **Key Methods**: `assess_dnsh(investment)`, `assess_portfolio_dnsh()`, `get_dnsh_criteria()`, `generate_dnsh_report()`

### 5.4 Good Governance Engine
- **File**: `engines/good_governance_engine.py`
- **Purpose**: Verify investee companies meet Article 2(17) good governance requirements
- **Features**:
  - Sound management structures assessment
  - Employee relations evaluation
  - Remuneration compliance verification
  - Tax compliance assessment
  - UNGC/OECD Guidelines adherence check
  - Anti-corruption and anti-bribery evaluation
  - Governance scoring with pass/fail threshold
- **Key Methods**: `assess_governance(company)`, `assess_portfolio_governance()`, `get_violation_report()`, `governance_score()`

### 5.5 ESG Characteristics Engine
- **File**: `engines/esg_characteristics_engine.py`
- **Purpose**: Define, track, and report on promoted E/S characteristics
- **Features**:
  - Environmental characteristic definition (climate, water, biodiversity, pollution, CE)
  - Social characteristic definition (labor, human rights, diversity, community)
  - Binding element tracking (minimum commitments)
  - Characteristic attainment measurement
  - Benchmark comparison (designated reference benchmark)
  - Sustainability indicator definition and monitoring
- **Key Methods**: `define_characteristics()`, `measure_attainment()`, `compare_to_benchmark()`, `get_binding_elements()`

### 5.6 Sustainable Investment Calculator Engine
- **File**: `engines/sustainable_investment_calculator.py`
- **Purpose**: Calculate and classify sustainable investments per SFDR Article 2(17)
- **Features**:
  - Sustainable investment identification (environmental + social objectives)
  - Proportion calculation (% of total fund NAV)
  - Minimum commitment adherence tracking
  - EU Taxonomy-aligned sustainable investments subset
  - Non-taxonomy sustainable investments (other environmental/social)
  - Minimum proportion maintenance monitoring
- **Key Methods**: `classify_investments()`, `calculate_proportion()`, `check_minimum_commitment()`, `breakdown_sustainable()`

### 5.7 Portfolio Carbon Footprint Engine
- **File**: `engines/portfolio_carbon_footprint.py`
- **Purpose**: Calculate portfolio-level carbon metrics for PAI reporting
- **Features**:
  - Weighted Average Carbon Intensity (WACI) calculation
  - Portfolio carbon footprint (tCO2eq per EUR million invested)
  - Attribution by sector, geography, and holding
  - Financed emissions calculation (PCAF methodology)
  - Scope 1+2 and Scope 1+2+3 breakdowns
  - Temperature alignment assessment (optional)
- **Key Methods**: `calculate_waci()`, `calculate_carbon_footprint()`, `calculate_financed_emissions()`, `attribution_analysis()`

### 5.8 EET Data Engine
- **File**: `engines/eet_data_engine.py`
- **Purpose**: Manage European ESG Template (EET) data input/output
- **Features**:
  - EET v1.1.1 field mapping (250+ fields)
  - SFDR-related EET fields extraction (pre-contractual, periodic)
  - EU Taxonomy EET fields mapping
  - PAI EET fields population
  - Data validation against EET specification
  - Export in EET-compatible CSV/XML format
- **Key Methods**: `populate_eet_fields()`, `validate_eet_data()`, `export_eet()`, `import_eet()`, `get_sfdr_fields()`

---

## 6. Workflows (8)

### 6.1 Pre-contractual Disclosure Workflow
- **File**: `workflows/precontractual_disclosure.py`
- **Phases**: 5
  1. Product Classification - Verify Article 8 eligibility, define E/S characteristics
  2. Investment Strategy - Define binding elements, asset allocation, derivatives policy
  3. Sustainability Assessment - Taxonomy alignment %, sustainable investment %, DNSH
  4. Template Population - Annex II template generation with all required sections
  5. Review & Approval - Compliance review, legal sign-off, versioning

### 6.2 Periodic Reporting Workflow
- **File**: `workflows/periodic_reporting.py`
- **Phases**: 5
  1. Data Collection - Portfolio holdings, emissions data, governance data
  2. Performance Assessment - Actual vs. committed E/S characteristics attainment
  3. PAI Calculation - All applicable PAI indicators with YoY comparison
  4. Template Generation - Annex IV template with actual figures and comparisons
  5. Filing Package - Assemble filing-ready package with supporting evidence

### 6.3 Website Disclosure Workflow
- **File**: `workflows/website_disclosure.py`
- **Phases**: 4
  1. Content Assembly - Gather all Annex III required disclosures
  2. Template Generation - Structured HTML/Markdown/JSON output
  3. Update Tracking - Version control, change history, last-updated timestamps
  4. Publication - Generate publication-ready content with regulatory references

### 6.4 PAI Statement Workflow
- **File**: `workflows/pai_statement.py`
- **Phases**: 4
  1. Data Sourcing - Collect investee-level data for all 18 mandatory PAI indicators
  2. Calculation - Portfolio-weighted PAI calculations with coverage tracking
  3. Reporting - Generate PAI statement with narrative explanations
  4. Action Planning - Engagement actions, exclusion decisions, monitoring updates

### 6.5 Portfolio Screening Workflow
- **File**: `workflows/portfolio_screening.py`
- **Phases**: 4
  1. Universe Definition - Define investment universe and screening criteria
  2. Negative Screening - Apply exclusion criteria (controversial weapons, fossil fuel thresholds)
  3. Positive Screening - Apply E/S promotion criteria (ESG ratings, sustainability indicators)
  4. Compliance Check - Verify portfolio meets Article 8 binding elements

### 6.6 Taxonomy Alignment Workflow
- **File**: `workflows/taxonomy_alignment.py`
- **Phases**: 4
  1. Holdings Analysis - Map portfolio holdings to taxonomy-eligible activities
  2. Alignment Assessment - Calculate alignment ratios per holding and objective
  3. Aggregation - Portfolio-level alignment ratio with double-counting prevention
  4. Commitment Tracking - Compare actual alignment to pre-contractual commitments

### 6.7 Compliance Review Workflow
- **File**: `workflows/compliance_review.py`
- **Phases**: 4
  1. Disclosure Completeness - Verify all mandatory disclosures are current
  2. Data Quality Check - Assess data coverage, estimation rates, data age
  3. Commitment Adherence - Check all binding elements are satisfied
  4. Action Items - Generate remediation tasks for any compliance gaps

### 6.8 Regulatory Update Workflow
- **File**: `workflows/regulatory_update.py`
- **Phases**: 3
  1. Change Detection - Monitor SFDR amendments, ESMA guidance, Q&A updates
  2. Impact Assessment - Evaluate impact on current disclosures and processes
  3. Migration Planning - Plan disclosure updates and timeline for compliance

---

## 7. Templates (8)

### 7.1 Pre-contractual Disclosure Template (Annex II)
- **File**: `templates/annex_ii_precontractual.py`
- **Sections**: Product name, E/S characteristics, investment strategy, asset allocation,
  sustainable investment %, taxonomy alignment %, DNSH, data sources, limitations,
  due diligence, engagement policies, designated reference benchmark
- **Outputs**: Markdown, HTML, JSON

### 7.2 Periodic Report Template (Annex IV)
- **File**: `templates/annex_iv_periodic.py`
- **Sections**: E/S characteristics attainment, top 15 investments, proportion breakdown
  (sustainable/taxonomy/other), PAI indicators considered, actions taken,
  comparison to previous period, designated benchmark performance
- **Outputs**: Markdown, HTML, JSON

### 7.3 Website Disclosure Template (Annex III)
- **File**: `templates/annex_iii_website.py`
- **Sections**: Summary, no sustainable investment objective statement, E/S characteristics,
  investment strategy, proportion of investments, monitoring of E/S characteristics,
  methodologies, data sources, limitations, due diligence, engagement policies,
  designated reference benchmark
- **Outputs**: Markdown, HTML, JSON

### 7.4 PAI Statement Template
- **File**: `templates/pai_statement_template.py`
- **Sections**: All 18 mandatory indicators with metrics, YoY comparison, explanation
  of actions taken, engagement outcomes, exclusion policy impact
- **Outputs**: Markdown, HTML, JSON

### 7.5 Portfolio ESG Dashboard
- **File**: `templates/portfolio_esg_dashboard.py`
- **Sections**: Fund overview, ESG scores, taxonomy alignment gauge, carbon metrics,
  sector allocation, PAI indicator summary, commitment tracking, alerts
- **Outputs**: Markdown, HTML, JSON

### 7.6 Taxonomy Alignment Report
- **File**: `templates/taxonomy_alignment_report.py`
- **Sections**: Alignment ratio summary, objective breakdown, pie chart data,
  eligible vs. aligned, commitment adherence, top aligned holdings
- **Outputs**: Markdown, HTML, JSON

### 7.7 Executive Summary Report
- **File**: `templates/executive_summary.py`
- **Sections**: Fund classification, key metrics, compliance status, risk flags,
  strategic recommendations, regulatory outlook
- **Outputs**: Markdown, HTML, JSON

### 7.8 Audit Trail Report
- **File**: `templates/audit_trail_report.py`
- **Sections**: Data lineage, calculation provenance (SHA-256), methodology references,
  data source inventory, estimation methodology, assumptions log
- **Outputs**: Markdown, HTML, JSON

---

## 8. Integrations (10)

### 8.1 Pack Orchestrator
- **File**: `integrations/pack_orchestrator.py`
- **Pipeline**: 10-phase master orchestration
  1. Health Check
  2. Configuration Init
  3. Portfolio Data Loading
  4. PAI Data Collection
  5. Taxonomy Alignment Assessment
  6. DNSH & Governance Screening
  7. ESG Characteristics Assessment
  8. Disclosure Generation
  9. Compliance Verification
  10. Audit Trail

### 8.2 Taxonomy Pack Bridge
- **File**: `integrations/taxonomy_pack_bridge.py`
- **Purpose**: Connects PACK-008 engines for taxonomy alignment calculations
- **Mappings**: Taxonomy alignment ratio, eligible/aligned breakdown, objective split

### 8.3 MRV Emissions Bridge
- **File**: `integrations/mrv_emissions_bridge.py`
- **Purpose**: Routes emissions data from 30 MRV agents to PAI indicators 1-6
- **Mappings**: Scope 1/2/3 -> PAI 1, Carbon footprint -> PAI 2, GHG intensity -> PAI 3

### 8.4 Investment Screener Bridge
- **File**: `integrations/investment_screener_bridge.py`
- **Purpose**: Connects to `green_investment_screener.py` for SFDR classification
- **Mappings**: SFDRClassification enum, exclusion screening, ESG rating thresholds

### 8.5 Portfolio Data Bridge
- **File**: `integrations/portfolio_data_bridge.py`
- **Purpose**: Ingests portfolio holdings, NAV data, sector/geographic classification
- **Data**: Holdings, weights, enterprise values, sectors, countries

### 8.6 EET Data Bridge
- **File**: `integrations/eet_data_bridge.py`
- **Purpose**: Import/export European ESG Template data
- **Format**: CSV/XML EET v1.1.1 compatible

### 8.7 Regulatory Tracking Bridge
- **File**: `integrations/regulatory_tracking_bridge.py`
- **Purpose**: Monitor SFDR regulatory updates, ESMA Q&A, level 2 amendments

### 8.8 Data Quality Bridge
- **File**: `integrations/data_quality_bridge.py`
- **Purpose**: Enforce data quality thresholds for PAI calculations
- **Features**: Coverage ratio enforcement, estimation flagging, data age checks

### 8.9 Health Check
- **File**: `integrations/health_check.py`
- **Purpose**: 20-category system verification (engines, workflows, config, data)

### 8.10 Setup Wizard
- **File**: `integrations/setup_wizard.py`
- **Purpose**: 8-step guided configuration for SFDR Article 8 product setup
- **Steps**: Product type, E/S characteristics, binding elements, taxonomy commitment,
  PAI selection, data sources, reporting schedule, validation

---

## 9. Presets (5)

| Preset | Description | Focus |
|--------|-------------|-------|
| `asset_manager` | UCITS/AIF fund managers | Fund-level Article 8, full PAI |
| `insurance` | Insurance undertakings | Unit-linked products, IORP |
| `bank` | Credit institutions | Structured products, portfolio mandates |
| `pension_fund` | Occupational pension schemes | IORP Article 8, member reporting |
| `wealth_manager` | Discretionary portfolio managers | Portfolio-level compliance |

---

## 10. Testing Strategy

### 10.1 Unit Tests (Target: 250+)

| Test File | Component | Tests |
|-----------|-----------|-------|
| `test_manifest.py` | pack.yaml validation | 20 |
| `test_config.py` | Configuration models | 30 |
| `test_pai_calculator.py` | PAI indicator engine | 25 |
| `test_taxonomy_alignment.py` | Taxonomy alignment engine | 20 |
| `test_dnsh.py` | SFDR DNSH engine | 18 |
| `test_good_governance.py` | Good governance engine | 15 |
| `test_esg_characteristics.py` | ESG characteristics engine | 15 |
| `test_sustainable_investment.py` | Sustainable investment engine | 15 |
| `test_carbon_footprint.py` | Portfolio carbon footprint | 15 |
| `test_eet_data.py` | EET data engine | 15 |
| `test_workflows.py` | All 8 workflows | 12 |
| `test_templates.py` | All 8 templates | 12 |
| `test_integrations.py` | All 10 integrations | 15 |
| `test_demo.py` | Demo/smoke tests | 8 |
| `test_e2e.py` | End-to-end flows | 12 |
| `test_agent_integration.py` | Agent integration | 15 |

### 10.2 Integration Tests

- All integration tests marked with `@pytest.mark.integration`
- Cross-pack tests updated in `tests/pack_integration/test_cross_pack_integration.py`
- Pack runner updated in `tests/run_pack_integration.py`

---

## 11. PAI Calculation Methodology

### 11.1 General Formula

For corporate PAI indicators:
```
PAI_indicator = SUM(current_value_of_investment_i / enterprise_value_of_investee_i * adverse_impact_of_investee_i) / total_current_value_of_all_investments
```

### 11.2 Specific Formulas

**PAI 1 - GHG Emissions:**
```
Scope_x_emissions = SUM(
  (current_value_of_investment_i / enterprise_value_of_company_i) *
  scope_x_GHG_emissions_of_company_i
)
```

**PAI 2 - Carbon Footprint:**
```
Carbon_footprint = SUM(
  (current_value_of_investment_i / enterprise_value_of_company_i) *
  total_GHG_emissions_of_company_i
) / current_value_of_all_investments (EUR million)
```

**PAI 3 - GHG Intensity:**
```
GHG_intensity = SUM(
  (current_value_of_investment_i / current_value_of_all_investments) *
  (GHG_emissions_of_company_i / revenue_of_company_i)
)
```

**PAI 4 - Fossil Fuel Exposure:**
```
Fossil_fuel_exposure = (
  current_value_invested_in_fossil_fuel_companies /
  current_value_of_all_investments
) * 100
```

---

## 12. Regulatory References

| Reference | Document |
|-----------|----------|
| SFDR Level 1 | Regulation (EU) 2019/2088 |
| SFDR RTS | Commission Delegated Regulation (EU) 2022/1288 |
| Taxonomy Regulation | Regulation (EU) 2020/852 |
| Taxonomy Disclosures DA | Delegated Regulation (EU) 2021/2178 |
| SFDR Amendments | Delegated Regulation (EU) 2023/363 |
| ESMA Q&A | ESMA34-45-1218 |
| ESMA Guidelines (fund names) | ESMA34-472-440 |
| EET Standard | FinDatEx European ESG Template v1.1.1 |
| PCAF Standard | Global GHG Accounting & Reporting Standard |

---

## 13. Acceptance Criteria

1. All 18 mandatory PAI indicators calculate correctly with deterministic results
2. Pre-contractual (Annex II), website (Annex III), and periodic (Annex IV) templates generate complete output
3. EU Taxonomy alignment ratio integrates correctly with PACK-008
4. DNSH assessment under SFDR RTS produces pass/fail per investment
5. Good governance checks cover all Article 2(17) requirements
6. EET data export passes EET v1.1.1 validation
7. Portfolio carbon footprint matches PCAF methodology
8. All 250+ unit tests pass with 0 failures
9. Cross-pack integration tests pass
10. SHA-256 provenance hashing on all engine outputs

---

## 14. Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Phase 1: Config & Structure | Day 1 | pack.yaml, pack_config.py, presets, directory structure |
| Phase 2: Engines | Day 1 | 8 calculation engines |
| Phase 3: Workflows & Templates | Day 1 | 8 workflows + 8 templates |
| Phase 4: Integrations | Day 1 | 10 integration bridges |
| Phase 5: Tests | Day 1 | 17 test files, 250+ tests |
| Phase 6: Integration | Day 1 | Cross-pack tests, pack runner update |
