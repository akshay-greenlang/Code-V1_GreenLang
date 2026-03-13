# PRD: AGENT-EUDR-018 -- Commodity Risk Analyzer

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-018 |
| **Agent ID** | GL-EUDR-CRA-018 |
| **Component** | Commodity Risk Analyzer Agent |
| **Category** | EUDR Regulatory Agent -- Commodity Intelligence |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-09 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-09 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **Location** | `greenlang/agents/eudr/commodity_risk_analyzer/` |
| **DB Prefix** | `gl_eudr_cra_` |
| **Metrics Prefix** | `gl_eudr_cra_` |
| **Env Prefix** | `GL_EUDR_CRA_` |
| **Migration** | V106 |
| **API Prefix** | `/api/v1/eudr-cra` |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) regulates seven commodity groups -- cattle, cocoa, coffee, oil palm, rubber, soya, and wood -- plus hundreds of derived products listed in Annex I. Each commodity carries a unique and distinct risk profile shaped by its production geography, processing chain complexity, price volatility, substitution dynamics, and regulatory treatment under EUDR. Operators placing these commodities (or products derived from them) on the EU market must perform commodity-specific due diligence (Articles 4, 9, 10) that accounts for these unique risk characteristics.

Today, most compliance systems treat all seven EUDR commodities as interchangeable from a risk analysis perspective, applying a single generic risk model regardless of whether the operator imports raw rubber latex or finished chocolate bars. This creates critical blind spots:

- **No commodity-specific risk profiling**: A generic "commodity risk" score does not capture the unique deforestation drivers, supply chain structures, and processing chain risks that differ radically between, for example, oil palm (plantation-based, mill-level aggregation, RSPO certification landscape) and cattle (pasture-based, animal movement, indirect land-use change).
- **No derived product traceability risk**: When cocoa beans become chocolate, or cattle hides become leather, or palm oil becomes biodiesel, the risk transformation through processing stages is not modeled. A chocolate bar inherits risk from its cocoa origin, but that risk is amplified or attenuated depending on the processing chain's transparency.
- **No price and market risk integration**: Commodity price volatility and market disruptions directly drive deforestation risk. When palm oil prices spike, deforestation pressure intensifies in frontier regions. No existing EUDR tool integrates price signals into commodity risk scoring.
- **No production forecasting**: Climate change, pest outbreaks, and policy shifts alter commodity production patterns. Without forward-looking production forecasts, operators cannot anticipate where deforestation pressure will intensify.
- **No substitution risk detection**: When operators or suppliers switch from one EUDR commodity to another (e.g., replacing palm oil with soya oil), or shift sourcing regions to avoid scrutiny, the risk implications are invisible without a dedicated substitution risk analyzer.
- **No commodity-level regulatory mapping**: EUDR Articles impose different requirements depending on the commodity type (e.g., geolocation requirements differ for cattle vs. crops, Annex I product classifications vary by commodity). No tool maps these article-specific requirements at the commodity level.
- **No portfolio-level commodity risk**: Operators handling multiple EUDR commodities cannot assess their aggregate commodity risk exposure, concentration risk, or diversification benefits.

**Distinction from EUDR-016 Country Risk Evaluator**: AGENT-EUDR-016's `commodity_risk_analyzer` engine performs country-commodity cross-analysis -- it answers "What is the risk of importing cocoa FROM Brazil?" by crossing country risk data with commodity data. AGENT-EUDR-018, by contrast, is a standalone deep-dive commodity intelligence agent that answers "What is the full risk profile OF cocoa as a commodity?" including its derived products, price dynamics, production forecasts, substitution patterns, regulatory requirements, due diligence workflows, and portfolio-level aggregation. EUDR-016 looks at the country axis with commodity as a filter; EUDR-018 looks at the commodity axis with depth and breadth that no country-level agent can provide.

Without a dedicated Commodity Risk Analyzer, EU operators face regulatory penalties of up to 4% of annual EU turnover, confiscation of non-compliant goods, and exclusion from public procurement -- driven by commodity-specific compliance failures that a generic risk model cannot detect or prevent.

### 1.2 Solution Overview

AGENT-EUDR-018: Commodity Risk Analyzer is a specialized agent that provides deep, commodity-specific risk intelligence for all seven EUDR-regulated commodities and their derived products. It operates as an 8-engine analytical platform that profiles each commodity's unique risk characteristics, traces risk through derived product processing chains, integrates price and market signals, forecasts production patterns, detects commodity substitution risks, maps commodity-specific regulatory requirements, orchestrates commodity-level due diligence workflows, and aggregates risk across multi-commodity portfolios.

Core capabilities:

1. **Commodity-specific deep risk profiling** -- Comprehensive per-commodity risk profiles covering deforestation drivers, supply chain archetypes, processing chain structures, certification landscapes, and historical compliance patterns for all 7 EUDR commodities.
2. **Derived product traceability risk** -- Maps the full commodity-to-product transformation chain (e.g., cocoa beans to cocoa butter to chocolate to confectionery) with risk attenuation/amplification modeling at each processing stage.
3. **Price volatility and market risk integration** -- Tracks commodity price indices, calculates historical and implied volatility, detects market disruptions, and correlates price movements with deforestation pressure signals.
4. **Production forecasting** -- Models yield per commodity per region, assesses climate impact on production, identifies seasonal risk patterns, and forecasts where deforestation pressure will intensify.
5. **Substitution risk detection** -- Identifies commodity switching patterns (palm oil to soya, natural rubber to synthetic), cross-commodity substitution risks, and potential greenwashing through commodity misclassification.
6. **Commodity-specific regulatory compliance** -- Maps EUDR Articles to commodity-specific requirements, tracks Annex I derived product classifications, assesses penalty risk per commodity, and monitors regulatory evolution.
7. **Commodity-level due diligence workflows** -- Provides commodity-specific DD checklists, evidence requirements, verification protocols, and documentation templates tailored to each commodity's unique characteristics.
8. **Cross-commodity portfolio risk aggregation** -- Aggregates risk across multiple commodity positions, calculates concentration risk, diversification scoring, and portfolio-level compliance readiness.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Commodity coverage | All 7 EUDR commodities + 100% of Annex I derived products | Commodity/product coverage matrix |
| Risk profile completeness | 50+ risk factors per commodity profile | Profile attribute count audit |
| Derived product chain depth | Average 4+ processing stages modeled per commodity | Chain depth statistics |
| Price data freshness | Daily price index updates for all 7 commodities | Data staleness monitoring |
| Production forecast accuracy | Within 15% of actual production for 12-month horizon | Forecast vs. actuals comparison |
| Substitution detection rate | 90%+ of commodity switching events detected | Precision/recall against audit |
| Regulatory mapping completeness | 100% of EUDR Articles mapped to commodity-specific requirements | Article coverage matrix |
| Portfolio risk calculation | < 2 seconds for 50-commodity portfolio analysis | p99 latency benchmark |
| Due diligence workflow coverage | All 7 commodities with tailored DD workflows | Workflow completeness audit |
| Zero-hallucination compliance | 100% deterministic, reproducible risk calculations | Bit-perfect reproducibility tests |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with commodity risk analytics representing a 1-2 billion EUR technology sub-segment.
- **SAM (Serviceable Addressable Market)**: 80,000+ EU importers handling multiple EUDR commodities who require commodity-specific risk intelligence, estimated at 600M-900M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 400+ enterprise customers in Year 1 requiring deep commodity analytics, representing 30M-50M EUR in commodity risk module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers handling 3+ EUDR commodity groups simultaneously
- Multinational food and beverage companies with multi-commodity portfolios (cocoa + palm oil + soya)
- Commodity trading houses dealing in all 7 regulated commodities
- Retailers with private-label products containing multiple EUDR commodities

**Secondary:**
- Commodity procurement teams needing price-risk-deforestation correlation intelligence
- Risk management departments requiring portfolio-level EUDR exposure analysis
- Certification bodies validating commodity-specific compliance claims
- Financial institutions assessing EUDR commodity exposure in trade finance portfolios
- SME importers preparing for June 30, 2026 enforcement

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Generic EUDR compliance tools | Broad coverage; simple UI | No commodity depth; single risk model for all commodities | Deep per-commodity profiling with 50+ risk factors |
| Commodity trading platforms (Refinitiv, Bloomberg) | Price data; market intelligence | Not EUDR-specific; no deforestation linkage; no DD workflows | EUDR-native with deforestation-price correlation |
| Sustainability certification tools (RSPO, FSC) | Commodity expertise | Single-commodity only; no cross-commodity portfolio view | All 7 commodities; portfolio aggregation; substitution detection |
| Country risk benchmarking (EUDR-016 type) | Country-level risk; EC benchmark integration | No commodity depth; treats commodities generically | Commodity-first analysis; derived product chains; production forecasting |
| In-house commodity risk models | Tailored to org | 6-12 month build; no regulatory updates; limited to known commodities | Ready now; all 7 commodities; continuous regulatory updates |

### 2.4 Differentiation Strategy

1. **Commodity-first architecture** -- Every engine is purpose-built for commodity-specific analysis, not a generic risk tool with commodity as a filter.
2. **Derived product chain modeling** -- No competitor traces risk through commodity-to-product transformation chains with processing stage risk modeling.
3. **Price-deforestation correlation** -- Unique integration of commodity price signals with deforestation risk, enabling predictive risk intelligence.
4. **Substitution detection** -- First-to-market capability for detecting commodity switching and greenwashing through misclassification.
5. **Portfolio aggregation** -- Cross-commodity risk view that no single-commodity tool can provide.
6. **GreenLang ecosystem integration** -- Deep integration with EUDR-016 (country risk), EUDR-001 (supply chain), and 20+ AGENT-DATA connectors.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to perform commodity-specific EUDR due diligence | 100% of customers pass commodity-level audit requirements | Q2 2026 |
| BG-2 | Reduce commodity risk assessment time from days to minutes | 95% reduction in per-commodity risk analysis time | Q2 2026 |
| BG-3 | Become the reference commodity risk platform for EUDR compliance | 400+ enterprise customers using commodity analytics | Q4 2026 |
| BG-4 | Detect and prevent commodity substitution-driven non-compliance | Zero substitution-related penalties for active customers | Ongoing |
| BG-5 | Provide forward-looking commodity risk intelligence | Production forecast reports adopted by 200+ customers | Q3 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Deep commodity profiling | Comprehensive risk profiles for all 7 EUDR commodities with 50+ risk factors each |
| PG-2 | Derived product traceability | Map commodity-to-product transformation chains for all Annex I products |
| PG-3 | Price-risk correlation | Integrate commodity price data with deforestation risk for predictive analytics |
| PG-4 | Production forecasting | Model yield, climate impact, and seasonal risk patterns per commodity per region |
| PG-5 | Substitution detection | Identify commodity switching, source shifting, and misclassification patterns |
| PG-6 | Regulatory per-commodity mapping | Map EUDR Articles to commodity-specific requirements and documentation needs |
| PG-7 | Commodity DD workflows | Provide tailored due diligence workflows for each commodity type |
| PG-8 | Portfolio risk aggregation | Cross-commodity risk view with concentration and diversification analysis |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Risk calculation performance | < 500ms p99 for single commodity risk scoring |
| TG-2 | Portfolio aggregation performance | < 2 seconds for 50-commodity portfolio analysis |
| TG-3 | Price data ingestion throughput | 10,000 price records per minute |
| TG-4 | API response time | < 200ms p95 for standard queries |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility for all risk calculations |
| TG-7 | Derived product chain resolution | < 1 second for full chain trace through 10+ processing stages |

---

## 4. User Personas

### Persona 1: Commodity Risk Manager -- Erik (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Commodity Risk at a large EU food conglomerate |
| **Company** | 12,000 employees, importing cocoa, palm oil, soya, and coffee |
| **EUDR Pressure** | Must assess and manage risk across 4 EUDR commodities simultaneously |
| **Pain Points** | Uses separate tools for each commodity; no unified risk view; cannot correlate price volatility with deforestation pressure; no derived product risk tracking for 200+ SKUs |
| **Goals** | Single platform for all commodity risk analytics; portfolio-level risk dashboard; price-deforestation correlation alerts; commodity-specific DD workflows |
| **Technical Skill** | High -- comfortable with analytics platforms, dashboards, and data exports |

### Persona 2: Compliance Officer -- Isabelle (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | EUDR Compliance Lead at a European chocolate manufacturer |
| **Company** | 3,000 employees, sourcing cocoa from 8 countries |
| **EUDR Pressure** | Must demonstrate commodity-specific due diligence for cocoa and all derived products (cocoa butter, cocoa powder, chocolate) |
| **Pain Points** | Cannot map how risk transforms through processing chain (bean to bar); no commodity-specific DD checklists; unsure which EUDR articles apply specifically to cocoa vs. other commodities |
| **Goals** | Commodity-specific regulatory mapping; derived product risk traceability; tailored DD workflows; audit-ready documentation per commodity |
| **Technical Skill** | Moderate -- comfortable with web applications and compliance tools |

### Persona 3: Procurement Strategist -- Pieter (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Strategic Procurement Director at a palm oil refinery and biodiesel producer |
| **Company** | 5,000 employees, processing palm oil into 15+ derived products |
| **EUDR Pressure** | Must understand how commodity substitution (palm oil to soya, rapeseed) affects EUDR compliance |
| **Pain Points** | Suppliers occasionally substitute commodities without disclosure; price spikes drive sourcing shifts to higher-risk regions; no tool detects substitution patterns |
| **Goals** | Substitution risk alerts; price-risk correlation for procurement decisions; production forecast data for strategic sourcing; commodity switching pattern detection |
| **Technical Skill** | Moderate -- uses ERP, procurement platforms, and market intelligence tools |

### Persona 4: Financial Risk Analyst -- Dr. Schreiber (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | ESG Risk Analyst at a European trade finance bank |
| **Company** | Financing commodity trade for 500+ clients |
| **EUDR Pressure** | Must assess EUDR commodity exposure in trade finance portfolio |
| **Pain Points** | No portfolio-level view of commodity risk across clients; cannot assess concentration risk; no commodity-specific regulatory penalty probability |
| **Goals** | Portfolio risk aggregation across clients and commodities; concentration risk alerts; regulatory penalty probability per commodity; export analytics for risk committee |
| **Technical Skill** | High -- comfortable with financial analytics, APIs, and data modeling |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 1(1-2)** | Scope: rules for placing/making available on EU market products containing the 7 commodities | CommodityProfiler covers all 7 commodities; DerivedProductAnalyzer maps all Annex I products |
| **Art. 2(1-3)** | Definitions of "deforestation", "deforestation-free", "forest degradation" | Per-commodity deforestation risk factors in CommodityProfiler (different drivers per commodity) |
| **Art. 2(4-12)** | Definitions of the 7 commodities: cattle, cocoa, coffee, oil palm, rubber, soya, wood | Commodity profiles with HS/CN code mappings, Annex I derived product classifications |
| **Art. 2(5)** | "Relevant products" -- products listed in Annex I that contain, have been fed with, or made using the 7 commodities | DerivedProductAnalyzer maps all Annex I products to source commodities with transformation chains |
| **Art. 3** | Prohibition on non-compliant products | Per-commodity compliance scoring in RegulatoryComplianceEngine |
| **Art. 4(1-2)** | Due diligence obligation: information collection, risk assessment, risk mitigation | CommodityDueDiligenceEngine provides commodity-specific DD workflows |
| **Art. 9(1)(a-d)** | Geolocation requirements (differ by commodity -- cattle vs. crops) | RegulatoryComplianceEngine maps geolocation rules per commodity type |
| **Art. 10(1-2)** | Risk assessment criteria | CommodityProfiler provides commodity-specific risk factors for Art. 10 assessment |
| **Art. 10(2)(a)** | Complexity of relevant supply chain | Per-commodity supply chain complexity scoring (wood = very high, soya = medium-high) |
| **Art. 10(2)(b)** | Risk of circumvention | SubstitutionRiskAnalyzer detects commodity switching and misclassification |
| **Art. 10(2)(d)** | Prevalence of deforestation in area of production | PriceVolatilityEngine correlates price pressure with deforestation prevalence |
| **Art. 10(2)(e)** | Country of production risk | Integration with EUDR-016 Country Risk Evaluator for country-commodity cross-reference |
| **Art. 10(2)(f)** | Risk of mixing with unknown origin products | SubstitutionRiskAnalyzer flags commodity blending and adulteration risks |
| **Art. 11** | Risk mitigation measures | CommodityDueDiligenceEngine provides commodity-specific mitigation protocols |
| **Art. 12** | DDS submission | RegulatoryComplianceEngine validates commodity-specific DDS requirements |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) | PortfolioRiskAggregator integrates country benchmarking at commodity portfolio level |
| **Art. 31** | Record keeping for 5 years | Audit log with full commodity risk calculation provenance |
| **Annex I** | List of relevant products with CN codes | DerivedProductAnalyzer contains complete Annex I mapping with 400+ CN codes |

### 5.2 Covered Commodities and Derived Product Chains

| Commodity | Key Derived Products (Annex I) | Processing Stages | Risk Complexity |
|-----------|-------------------------------|-------------------|-----------------|
| **Cattle** | Beef, leather, hides, tallow, gelatin, pet food, bone meal | Ranch -> Feedlot -> Slaughterhouse -> Packer -> Tanner/Processor -> Manufacturer | Very High (animal movement, pasture rotation, indirect land-use) |
| **Cocoa** | Cocoa butter, cocoa powder, cocoa paste, chocolate, confectionery | Farm -> Fermentation -> Drying -> Roasting -> Grinding -> Pressing -> Conching -> Moulding | Very High (smallholder aggregation, multi-origin blending) |
| **Coffee** | Green coffee, roasted coffee, instant coffee, coffee extracts | Farm -> Wet/Dry Processing -> Export -> Roasting -> Grinding -> Extraction -> Packaging | High (altitude/origin segregation, blending for flavor profiles) |
| **Oil Palm** | Palm oil, palm kernel oil, oleochemicals, biodiesel, glycerine, soap, margarine | Plantation -> Mill -> Refinery -> Fractionation -> Oleochemical Plant -> Manufacturer | Very High (mass balance challenges, RSPO certification complexity) |
| **Rubber** | Natural rubber, latex, tires, gloves, conveyor belts, seals, hoses | Smallholder -> Collector -> Sheet/Block Processing -> Factory -> Manufacturer | High (latex aggregation, mixed origin at collector level) |
| **Soya** | Soybeans, soy meal, soy oil, soy lecithin, tofu, animal feed | Farm -> Silo -> Crusher -> Refinery -> Food/Feed Manufacturer | Medium-High (large volumes, commodity co-mingling, dual-use food/feed) |
| **Wood** | Timber, lumber, plywood, veneer, pulp, paper, furniture, charcoal, printed books | Forest -> Sawmill -> Veneer/Plywood Plant -> Pulp Mill -> Paper Mill -> Manufacturer | Very High (multi-step processing, species mixing, long chains) |

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline for all commodity-specific deforestation verification |
| June 29, 2023 | Regulation entered into force | Legal basis for all commodity compliance requirements |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Full commodity risk analysis must be operational |
| June 30, 2026 | Enforcement for SMEs | Scale for SME commodity portfolios; simplified workflows |
| Ongoing (quarterly) | EC country benchmarking updates | Commodity-country cross-risk recalculation triggered |
| Ongoing | Annex I amendments (derived product list) | DerivedProductAnalyzer must adapt to new product classifications |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 8 features below are P0 launch blockers. The agent cannot ship without all 8 features operational. Features 1-5 form the core commodity intelligence engine; Features 6-8 form the compliance, workflow, and aggregation layer.

**P0 Features 1-5: Core Commodity Intelligence Engine**

---

#### Feature 1: Commodity Profiler Engine

**User Story:**
```
As a commodity risk manager,
I want comprehensive risk profiles for each of the 7 EUDR commodities,
So that I can understand the unique risk characteristics, deforestation drivers, and supply chain structures specific to each commodity I import.
```

**Acceptance Criteria:**
- [ ] Maintains deep risk profiles for all 7 EUDR commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood)
- [ ] Each profile contains 50+ structured risk factors organized into categories: deforestation drivers, supply chain structure, processing complexity, certification landscape, historical compliance, environmental impact
- [ ] Profiles include commodity-specific deforestation driver analysis (e.g., pasture expansion for cattle, plantation clearing for palm oil)
- [ ] Profiles include supply chain archetype definition (typical depth, actor types, aggregation points)
- [ ] Profiles include certification landscape mapping (which certifications exist, coverage rates, credibility scores)
- [ ] Profiles include historical non-compliance event registry (past enforcement actions, seizures, scandals per commodity)
- [ ] Calculates composite commodity risk score (0-100) using deterministic weighted formula
- [ ] Supports commodity risk comparison (side-by-side analysis of 2+ commodities)
- [ ] Supports temporal risk trending (how commodity risk has evolved over 5+ years)
- [ ] Profiles are versioned and auditable with SHA-256 provenance hashes
- [ ] All risk calculations are deterministic with zero LLM involvement

**Risk Factor Categories (per commodity):**
```
Commodity Risk Score = sum of weighted category scores:

  Deforestation_Pressure  * W_deforestation  (0.25)
  Supply_Chain_Opacity    * W_opacity         (0.20)
  Processing_Complexity   * W_processing      (0.15)
  Certification_Coverage  * W_certification   (0.15)
  Regulatory_Exposure     * W_regulatory      (0.10)
  Market_Volatility       * W_market          (0.10)
  Historical_Compliance   * W_historical      (0.05)

Where each category score is 0-100 and weights sum to 1.0 (configurable).
```

**Non-Functional Requirements:**
- Performance: Profile retrieval < 50ms; risk score calculation < 200ms
- Completeness: 100% of Annex I commodity classifications covered
- Reproducibility: Deterministic risk scoring across runs

**Dependencies:**
- EUDR Annex I commodity definitions and CN code mappings
- Historical deforestation data per commodity (FAO, GFW)
- Certification body coverage statistics (FSC, RSPO, Rainforest Alliance, UTZ, 4C)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- New commodity added to EUDR scope (regulation amendment) -- Profile creation workflow with minimum required fields
- Commodity with zero historical compliance data -- Flag as "insufficient data" with elevated risk
- Regional commodity variants (Robusta vs. Arabica coffee) -- Sub-commodity profiling support

---

#### Feature 2: Derived Product Analyzer

**User Story:**
```
As a compliance officer,
I want to understand how risk transforms when a raw commodity is processed into derived products,
So that I can assess the EUDR compliance risk for every product in my portfolio, not just raw commodities.
```

**Acceptance Criteria:**
- [ ] Maps all Annex I derived products to their source commodity with CN/HS code linkage
- [ ] Models commodity-to-product transformation chains with defined processing stages
- [ ] Tracks risk attenuation/amplification at each processing stage based on traceability preservation
- [ ] Calculates derived product risk score that inherits from source commodity but adjusts for processing chain transparency
- [ ] Models multi-commodity derived products (e.g., chocolate = cocoa + soya lecithin + palm oil margarine)
- [ ] Supports chain-of-custody model tracking through processing stages (Identity Preserved, Segregated, Mass Balance)
- [ ] Identifies processing stages where traceability is most commonly lost (critical loss points)
- [ ] Generates derived product risk reports with full processing chain visualization
- [ ] Supports custom derived product definitions for products not in Annex I but containing EUDR commodities
- [ ] Maintains derived product database with 400+ CN code mappings

**Derived Product Risk Formula:**
```
Derived_Product_Risk = Source_Commodity_Risk
                     * Processing_Chain_Transparency_Factor  (0.5 - 1.5)
                     * Custody_Model_Factor                  (IP=0.8, Seg=1.0, MB=1.3)
                     * Blending_Complexity_Factor            (1.0 - 2.0)

Where:
- Processing_Chain_Transparency_Factor increases with each opaque processing stage
- Custody_Model_Factor reflects traceability strength of the chain-of-custody model
- Blending_Complexity_Factor increases when multiple commodity sources are blended
```

**Non-Functional Requirements:**
- Coverage: 100% of Annex I products mapped to source commodities
- Performance: Full chain trace < 1 second for 10+ processing stages
- Accuracy: Derived product risk scores validated against manual assessment for 50+ products

**Dependencies:**
- EUDR Annex I product list with CN/HS codes
- Processing chain definitions per commodity (industry reference data)
- AGENT-EUDR-001 Supply Chain Mapping Master for actual supply chain data

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 domain specialist)

---

#### Feature 3: Price Volatility Engine

**User Story:**
```
As a procurement strategist,
I want to understand how commodity price movements correlate with deforestation risk,
So that I can anticipate risk escalation during price spikes and adjust sourcing strategies proactively.
```

**Acceptance Criteria:**
- [ ] Tracks daily price indices for all 7 EUDR commodities from reference sources
- [ ] Calculates historical volatility (30-day, 90-day, 1-year rolling windows)
- [ ] Calculates implied volatility from price movement patterns
- [ ] Detects market disruption events (price spikes > 2 standard deviations, supply shocks)
- [ ] Correlates price movements with deforestation alert data (satellite-derived) with configurable lag periods
- [ ] Generates price-deforestation correlation coefficients per commodity per region
- [ ] Provides price-driven risk adjustment that modifies commodity risk score during high-volatility periods
- [ ] Tracks price differentials between certified and non-certified commodity streams
- [ ] Generates market risk alerts when price-deforestation correlation exceeds threshold
- [ ] Stores 10+ years of historical price data for trend analysis
- [ ] All calculations are deterministic -- no ML predictions in the risk scoring path

**Price-Risk Adjustment Formula:**
```
Price_Adjusted_Risk = Base_Commodity_Risk * (1 + Price_Volatility_Factor)

Price_Volatility_Factor = min(
    (Current_Volatility - Baseline_Volatility) / Baseline_Volatility * Sensitivity,
    Max_Adjustment
)

Where:
- Baseline_Volatility = 5-year average volatility for the commodity
- Sensitivity = configurable per commodity (default 0.3)
- Max_Adjustment = configurable cap (default 0.5, meaning max 50% risk increase)
```

**Non-Functional Requirements:**
- Data Freshness: Price data updated daily (T+1 at most)
- Storage: 10+ years of daily price history per commodity (approximately 25,000 records per commodity)
- Calculation Speed: Volatility calculation < 100ms; correlation analysis < 500ms
- Determinism: Same input data produces identical volatility and correlation outputs

**Dependencies:**
- Commodity price data sources (World Bank Commodity Prices, ICE Futures, CBOT)
- AGENT-DATA-007 Deforestation Satellite Connector for deforestation alert correlation
- TimescaleDB hypertable for time-series price storage

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

---

#### Feature 4: Production Forecast Engine

**User Story:**
```
As a commodity risk manager,
I want forward-looking production forecasts per commodity and region,
So that I can anticipate where supply shortages will create deforestation pressure and adjust risk assessments accordingly.
```

**Acceptance Criteria:**
- [ ] Models historical production volumes per commodity per country/region (10+ year history)
- [ ] Calculates production trend analysis using linear and polynomial regression (deterministic)
- [ ] Incorporates climate impact factors (temperature, rainfall anomalies) on yield per commodity
- [ ] Identifies seasonal production patterns (harvest cycles, planting seasons) per commodity per region
- [ ] Generates 12-month production forecasts per commodity per top-20 producing countries
- [ ] Calculates supply-demand gap indicators that signal potential deforestation pressure
- [ ] Identifies frontier expansion risk -- regions where production is growing fastest (proxy for deforestation)
- [ ] Generates production risk heatmaps by commodity and region
- [ ] Supports what-if scenarios: "What happens to palm oil risk if Indonesia production drops 10%?"
- [ ] All forecasts use deterministic statistical methods (regression, moving averages) -- no ML/LLM in the calculation path

**Production Risk Score:**
```
Production_Risk = f(
    Supply_Demand_Gap,        # Higher gap = more pressure to expand = more deforestation risk
    Frontier_Expansion_Rate,  # Faster expansion in new areas = higher risk
    Climate_Vulnerability,    # More climate-vulnerable production = more displacement risk
    Yield_Trend               # Declining yields = more land needed = higher risk
)
```

**Non-Functional Requirements:**
- Data Coverage: Top 20 producing countries per commodity (covers 90%+ of global production)
- Forecast Accuracy: Within 15% of actual production for 12-month horizon
- Update Frequency: Quarterly forecast updates with monthly data refresh
- Performance: Forecast generation < 5 seconds per commodity per country

**Dependencies:**
- FAO production statistics (FAOSTAT)
- USDA Foreign Agricultural Service production estimates
- Climate data (NOAA, ERA5 reanalysis)
- AGENT-DATA-020 Climate Hazard Connector for climate impact data

**Estimated Effort:** 4 weeks (1 backend engineer, 1 data scientist)

---

#### Feature 5: Substitution Risk Analyzer

**User Story:**
```
As a procurement strategist,
I want to detect when suppliers switch commodities or sourcing regions to circumvent EUDR requirements,
So that I can identify potential compliance violations and greenwashing before they result in regulatory action.
```

**Acceptance Criteria:**
- [ ] Detects commodity switching patterns in procurement records (e.g., palm oil volumes drop, soya volumes spike from same supplier)
- [ ] Detects source region shifting (same commodity, different origin to avoid high-risk country classification)
- [ ] Detects commodity misclassification (products declared under wrong CN code to avoid EUDR scope)
- [ ] Calculates substitution probability score based on historical patterns, price differentials, and supplier behavior
- [ ] Identifies cross-commodity substitution pairs (palm oil <-> soya, natural rubber <-> synthetic, cattle leather <-> synthetic leather)
- [ ] Flags potential circumvention: re-routing through low-risk countries (transshipment risk)
- [ ] Generates substitution risk alerts with evidence (volume changes, price deltas, timing analysis)
- [ ] Tracks substitution events over time for trend analysis
- [ ] Provides greenwashing detection: products claimed as non-EUDR that actually contain EUDR commodities
- [ ] All detection uses deterministic rule-based analysis (threshold comparison, ratio analysis) -- no ML in scoring path

**Substitution Detection Rules:**
```
Rule 1 - Volume Shift: Flag when commodity X volume drops > 20% AND commodity Y volume
         increases > 20% from same supplier within 90-day window
Rule 2 - Origin Shift: Flag when same commodity shows > 30% change in country-of-origin
         mix within 180-day window
Rule 3 - Price Arbitrage: Flag when supplier switches to significantly cheaper alternative
         commodity during high-price period (price differential > 15%)
Rule 4 - Classification Anomaly: Flag when CN code changes for same product description
         from same supplier
Rule 5 - Transshipment: Flag when commodity enters EU through low-risk country but
         production country traces to high-risk country
```

**Non-Functional Requirements:**
- Detection Rate: 90%+ of genuine substitution events detected (validated against audit)
- False Positive Rate: < 15% false positives
- Latency: Substitution analysis < 3 seconds per supplier per quarter
- Evidence Quality: Every alert includes supporting data points and confidence score

**Dependencies:**
- AGENT-DATA-003 ERP/Finance Connector for procurement volume data
- AGENT-DATA-005 EUDR Traceability Connector for chain-of-custody data
- AGENT-EUDR-016 Country Risk Evaluator for country risk classifications
- Historical procurement records (12+ months)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

**P0 Features 6-8: Compliance, Workflow, and Aggregation Layer**

> Features 6, 7, and 8 are P0 launch blockers. Without commodity-specific regulatory mapping, DD workflows, and portfolio aggregation, the core intelligence engine cannot deliver actionable compliance value. These features transform raw commodity analytics into regulatory-ready outputs.

---

#### Feature 6: Regulatory Compliance Engine

**User Story:**
```
As a compliance officer,
I want to know exactly which EUDR requirements apply to each commodity I handle,
So that I can ensure my due diligence addresses all commodity-specific regulatory obligations.
```

**Acceptance Criteria:**
- [ ] Maps all EUDR Articles to commodity-specific requirements (e.g., Art. 9 geolocation differs for cattle vs. crops)
- [ ] Maintains complete Annex I product classification matrix with CN codes per commodity
- [ ] Tracks commodity-specific documentation requirements (which documents needed for which commodity)
- [ ] Calculates penalty risk per commodity based on enforcement history and compliance gaps
- [ ] Identifies commodity-specific simplified due diligence eligibility (Art. 13 -- products from low-risk countries)
- [ ] Tracks regulatory evolution: amendments to Annex I, new derived product additions, EC guidance updates
- [ ] Generates commodity-specific compliance checklists with Article references
- [ ] Validates product CN codes against Annex I to confirm EUDR scope applicability
- [ ] Supports regulatory what-if: "What if cattle products are removed from Annex I?" impact analysis
- [ ] Generates commodity compliance readiness score (0-100) based on documentation completeness

**Non-Functional Requirements:**
- Regulatory Accuracy: 100% of EUDR Articles correctly mapped to commodity requirements
- Update Latency: Regulatory changes reflected within 48 hours of EC publication
- Coverage: All Annex I CN codes (400+) mapped and validated

**Dependencies:**
- EUDR text (Regulation (EU) 2023/1115) and implementing regulations
- EU Combined Nomenclature (CN) code database
- EC EUDR guidance documents and FAQ updates
- AGENT-FOUND-005 Citations & Evidence Agent for regulatory references

**Estimated Effort:** 2 weeks (1 backend engineer, 1 regulatory specialist)

---

#### Feature 7: Commodity Due Diligence Engine

**User Story:**
```
As a compliance officer,
I want commodity-specific due diligence workflows with tailored evidence requirements,
So that I can efficiently perform the right level of due diligence for each commodity type rather than using a one-size-fits-all approach.
```

**Acceptance Criteria:**
- [ ] Provides 7 commodity-specific DD workflow templates (one per EUDR commodity)
- [ ] Each workflow includes commodity-tailored evidence requirements (e.g., cattle: animal movement records; wood: forest concession permits; palm oil: RSPO certificates)
- [ ] Defines verification protocols per commodity (what to verify, how to verify, acceptable evidence)
- [ ] Tracks DD workflow progress per commodity per operator (% complete, outstanding items)
- [ ] Generates commodity-specific risk mitigation action plans when DD identifies elevated risk
- [ ] Supports enhanced DD escalation for high-risk commodities or high-risk country origins
- [ ] Provides document checklists per commodity with status tracking (uploaded/pending/rejected)
- [ ] Generates DD completion reports suitable for auditor review
- [ ] Supports DD workflow versioning (new versions when regulatory requirements change)
- [ ] Integrates with AGENT-DATA-008 Supplier Questionnaire Processor for supplier-side evidence collection

**Commodity-Specific DD Requirements:**
| Commodity | Unique DD Evidence | Verification Protocol |
|-----------|-------------------|----------------------|
| Cattle | Animal movement records, pasture GPS, veterinary certificates | Cross-reference movement with deforestation dates |
| Cocoa | Cooperative membership lists, farm-gate purchase receipts, fermentation records | Verify smallholder farm polygons against satellite |
| Coffee | Wet mill delivery records, altitude/grade documentation, export certificates | Verify origin claims against geographic/altitude data |
| Oil Palm | Mill GPS, RSPO/ISPO certificates, concession maps, smallholder registration | Verify mill catchment against satellite deforestation |
| Rubber | Collector aggregation records, processing plant inputs, latex quality records | Verify collector sources against forest boundary maps |
| Soya | Silo intake records, crushing plant inputs, RTRS certificates, CAR registration | Verify farm coordinates against Amazon/Cerrado boundaries |
| Wood | Forest management plans, concession permits, species declarations, FSC/PEFC certificates | Verify concession boundaries, species against CITES |

**Non-Functional Requirements:**
- Workflow Coverage: All 7 commodities with tailored workflows
- Completion Tracking: Real-time progress updates, < 1 second latency
- Audit Trail: Every DD action logged with timestamp, actor, and provenance hash

**Dependencies:**
- AGENT-DATA-008 Supplier Questionnaire Processor for evidence collection
- AGENT-DATA-005 EUDR Traceability Connector for chain-of-custody verification
- SEC-005 Centralized Audit Logging for DD audit trail

**Estimated Effort:** 3 weeks (1 backend engineer, 1 regulatory specialist)

---

#### Feature 8: Portfolio Risk Aggregator

**User Story:**
```
As a commodity risk manager,
I want a portfolio-level view of my EUDR commodity risk exposure across all commodities I handle,
So that I can identify concentration risks, assess diversification benefits, and prioritize risk mitigation investments.
```

**Acceptance Criteria:**
- [ ] Aggregates commodity risk scores across all commodities in operator's portfolio
- [ ] Calculates portfolio-level composite risk score using deterministic weighted aggregation
- [ ] Identifies concentration risk: over-reliance on a single commodity, country, or supplier
- [ ] Calculates diversification score: how well-distributed risk is across commodities and origins
- [ ] Generates Herfindahl-Hirschman Index (HHI) for commodity concentration
- [ ] Provides risk contribution analysis: which commodity drives the most portfolio risk
- [ ] Supports portfolio-level what-if analysis: "What if we add rubber imports?"
- [ ] Generates portfolio risk heatmap by commodity x country matrix
- [ ] Tracks portfolio risk trending over time (monthly/quarterly snapshots)
- [ ] Exports portfolio risk reports in PDF, CSV, and JSON for risk committee review
- [ ] All aggregation calculations are deterministic with SHA-256 provenance hashes

**Portfolio Risk Formulas:**
```
Portfolio_Risk = sum(Commodity_Risk_i * Portfolio_Weight_i) for all commodities i

Concentration_Risk (HHI) = sum(Portfolio_Share_i^2) for all commodities i
  - HHI < 0.15: Low concentration
  - 0.15 <= HHI < 0.25: Moderate concentration
  - HHI >= 0.25: High concentration

Diversification_Score = 1 - (HHI / max_possible_HHI) scaled to 0-100

Risk_Contribution_i = (Commodity_Risk_i * Portfolio_Weight_i) / Portfolio_Risk
```

**Non-Functional Requirements:**
- Performance: Portfolio aggregation < 2 seconds for 50-commodity portfolio
- Accuracy: Portfolio risk deterministic and bit-perfect reproducible
- Reporting: PDF/CSV/JSON export < 5 seconds

**Dependencies:**
- Features 1-7 (all commodity-level engines)
- AGENT-EUDR-016 Country Risk Evaluator for country risk integration
- AGENT-EUDR-001 Supply Chain Mapping Master for supply chain risk integration

**Estimated Effort:** 2 weeks (1 backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 9: Commodity Scenario Simulator
- Model impact of EUDR scope changes (new commodity added, product removed from Annex I)
- Model impact of EC country reclassification on commodity portfolio
- Simulate supplier loss scenarios per commodity
- Monte Carlo simulation for portfolio risk under multiple scenarios

#### Feature 10: Commodity Benchmark Analytics
- Benchmark operator's commodity risk against anonymized industry averages
- Rank commodities by risk-adjusted compliance cost
- Provide industry-level statistics per commodity (average risk, compliance rates)

#### Feature 11: AI-Assisted Commodity Classification
- LLM-assisted classification of products into EUDR commodity categories
- Automated CN code suggestion for new products
- Natural language product description matching to Annex I entries

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Real-time commodity trading integration (live bid/ask prices)
- Commodity futures price forecasting using ML models
- Automated commodity hedging strategy recommendations
- Blockchain-based commodity certification verification
- Direct integration with commodity exchanges (ICE, CBOT, LME)
- Commodity ESG scoring beyond EUDR scope (biodiversity, water use)
- Mobile native application (web responsive design only for v1.0)

---

## 7. Technical Requirements

### 7.1 Architecture Overview

```
                                    +---------------------------+
                                    |     GL-EUDR-APP v1.0      |
                                    |   Frontend (React/TS)     |
                                    +-------------+-------------+
                                                  |
                                    +-------------v-------------+
                                    |     Unified API Layer      |
                                    |  /api/v1/eudr-cra (FastAPI)|
                                    +-------------+-------------+
                                                  |
        +-------------------+---------+-----------+-----------+---------+-------------------+
        |                   |         |           |           |         |                   |
+-------v-------+ +--------v------+ +v-----------v+ +--------v------+ +v---------+ +------v--------+
| Engine 1      | | Engine 2      | | Engine 3    | | Engine 4      | | Engine 5 | | Engine 6      |
| Commodity     | | Derived       | | Price       | | Production    | | Substit. | | Regulatory    |
| Profiler      | | Product       | | Volatility  | | Forecast      | | Risk     | | Compliance    |
|               | | Analyzer      | |             | |               | | Analyzer | |               |
| - 7 profiles  | | - Annex I map | | - Prices    | | - Yield model | | - Switch | | - Article map |
| - 50+ factors | | - Chain trace | | - Volatility| | - Climate     | | - Detect | | - CN codes    |
| - Risk calc   | | - Multi-commod| | - Correlate | | - Seasonal    | | - Alert  | | - Penalties   |
+---------------+ +---------------+ +-------------+ +---------------+ +----------+ +---------------+
        |                   |         |           |           |         |                   |
        +-------------------+---------+-----------+-----------+---------+-------------------+
                                                  |
                        +-------------------------+-------------------------+
                        |                                                   |
              +---------v---------+                              +----------v----------+
              | Engine 7          |                              | Engine 8            |
              | Commodity DD      |                              | Portfolio Risk      |
              |                   |                              | Aggregator          |
              | - DD workflows    |                              | - HHI concentration |
              | - Evidence reqs   |                              | - Diversification   |
              | - Verification    |                              | - Risk contribution |
              +-------------------+                              +---------------------+
                        |                                                   |
        +---------------+---------------------------------------------------+
        |                                                                   |
+-------v-------+        +------------------+        +---------------------+
| EUDR-016      |        | EUDR-001         |        | AGENT-DATA-005/007  |
| Country Risk  |        | Supply Chain     |        | Traceability /      |
| Evaluator     |        | Mapping Master   |        | Satellite           |
+---------------+        +------------------+        +---------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/commodity_risk_analyzer/
    __init__.py                          # Public API exports
    config.py                            # CommodityRiskAnalyzerConfig with GL_EUDR_CRA_ env prefix
    models.py                            # Pydantic v2 models for commodities, risk scores, portfolios
    commodity_profiler.py                # Engine 1: CommodityProfiler - deep per-commodity risk profiling
    derived_product_analyzer.py          # Engine 2: DerivedProductAnalyzer - Annex I product chain tracing
    price_volatility_engine.py           # Engine 3: PriceVolatilityEngine - price tracking and correlation
    production_forecast_engine.py        # Engine 4: ProductionForecastEngine - yield modeling and forecasting
    substitution_risk_analyzer.py        # Engine 5: SubstitutionRiskAnalyzer - switching and misclassification
    regulatory_compliance_engine.py      # Engine 6: RegulatoryComplianceEngine - article-to-commodity mapping
    commodity_dd_engine.py               # Engine 7: CommodityDueDiligenceEngine - commodity-specific DD workflows
    portfolio_risk_aggregator.py         # Engine 8: PortfolioRiskAggregator - cross-commodity aggregation
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains for all calculations
    metrics.py                           # 18 Prometheus self-monitoring metrics
    setup.py                             # CommodityRiskAnalyzerService facade (wires all 8 engines)
    reference_data/
        annex_i_products.json            # Complete Annex I product list with CN codes
        commodity_profiles.json          # Base commodity profile data
        processing_chains.json           # Commodity-to-product processing chain definitions
        certification_registry.json      # Certification schemes per commodity
    api/
        __init__.py
        router.py                        # FastAPI router (30+ endpoints)
        commodity_routes.py              # Commodity profiles and risk scoring endpoints
        derived_product_routes.py        # Derived product analysis endpoints
        price_routes.py                  # Price/market analysis endpoints
        forecast_routes.py               # Production forecast endpoints
        substitution_routes.py           # Substitution risk detection endpoints
        regulatory_routes.py             # Regulatory compliance check endpoints
        dd_routes.py                     # Due diligence workflow endpoints
        portfolio_routes.py              # Portfolio aggregation endpoints
```

### 7.3 Data Models (Key Entities)

```python
# EUDR Commodity Enumeration
class EUDRCommodity(str, Enum):
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

# Risk Level Classification
class RiskLevel(str, Enum):
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"

# Chain of Custody Model
class CustodyModel(str, Enum):
    IDENTITY_PRESERVED = "identity_preserved"
    SEGREGATED = "segregated"
    MASS_BALANCE = "mass_balance"

# Commodity Profile
class CommodityProfile(BaseModel):
    profile_id: str                      # Unique identifier
    commodity: EUDRCommodity             # Which EUDR commodity
    version: int                         # Profile version
    deforestation_drivers: List[DeforestationDriver]  # Commodity-specific drivers
    supply_chain_archetype: SupplyChainArchetype       # Typical supply chain structure
    processing_chain: List[ProcessingStage]            # Raw-to-product stages
    certification_landscape: List[CertificationScheme] # Available certifications
    historical_compliance: ComplianceHistory            # Past enforcement events
    environmental_impact: EnvironmentalImpact           # Deforestation association data
    risk_factors: Dict[str, RiskFactor]                # 50+ structured risk factors
    composite_risk_score: Decimal                       # 0-100 deterministic score
    risk_level: RiskLevel
    risk_weights: Dict[str, Decimal]                   # Configurable weights
    top_producing_countries: List[ProducingCountry]     # Top 20 producers
    annex_i_cn_codes: List[str]                        # CN codes for this commodity
    provenance_hash: str                                # SHA-256
    calculated_at: datetime
    metadata: Dict[str, Any]

# Derived Product
class DerivedProduct(BaseModel):
    product_id: str
    product_name: str
    cn_code: str                         # Combined Nomenclature code
    hs_code: str                         # Harmonized System code
    source_commodities: List[EUDRCommodity]  # Can be multi-commodity
    source_commodity_shares: Dict[EUDRCommodity, Decimal]  # % composition
    processing_chain: List[ProcessingStage]
    chain_depth: int                     # Number of processing stages
    custody_model: CustodyModel
    traceability_loss_points: List[str]  # Where traceability commonly breaks
    derived_risk_score: Decimal          # Adjusted from source commodity risk
    risk_level: RiskLevel
    annex_i_reference: str               # Annex I entry reference
    provenance_hash: str

# Price Record
class PriceRecord(BaseModel):
    record_id: str
    commodity: EUDRCommodity
    price_date: date
    price_usd: Decimal                   # USD per metric ton
    price_source: str                    # Data source identifier
    currency_original: str               # Original currency if not USD
    exchange_rate: Optional[Decimal]
    volume_traded: Optional[Decimal]     # Daily trading volume

# Volatility Metrics
class VolatilityMetrics(BaseModel):
    commodity: EUDRCommodity
    calculation_date: date
    volatility_30d: Decimal              # 30-day rolling volatility
    volatility_90d: Decimal              # 90-day rolling volatility
    volatility_1y: Decimal               # 1-year rolling volatility
    baseline_volatility: Decimal         # 5-year average
    volatility_ratio: Decimal            # Current / baseline
    price_change_30d_pct: Decimal        # 30-day price change %
    market_disruption_flag: bool         # True if > 2 std dev
    deforestation_correlation: Optional[Decimal]  # Price-deforestation correlation
    provenance_hash: str

# Production Forecast
class ProductionForecast(BaseModel):
    forecast_id: str
    commodity: EUDRCommodity
    country_code: str                    # ISO 3166-1 alpha-2
    forecast_year: int
    forecast_month: Optional[int]
    production_volume_mt: Decimal        # Metric tons
    confidence_interval_lower: Decimal
    confidence_interval_upper: Decimal
    trend_direction: str                 # "increasing" / "stable" / "decreasing"
    climate_impact_factor: Decimal       # Multiplier on baseline production
    frontier_expansion_rate: Decimal     # % of new area under production
    supply_demand_gap: Decimal           # Positive = undersupply pressure
    provenance_hash: str

# Substitution Event
class SubstitutionEvent(BaseModel):
    event_id: str
    operator_id: str
    supplier_id: Optional[str]
    detection_rule: str                  # Which rule triggered
    substitution_type: str               # commodity_switch / origin_shift / misclassification / transshipment
    commodity_from: Optional[EUDRCommodity]
    commodity_to: Optional[EUDRCommodity]
    country_from: Optional[str]
    country_to: Optional[str]
    volume_change_pct: Decimal
    price_differential_pct: Optional[Decimal]
    confidence_score: Decimal            # 0-100
    evidence: Dict[str, Any]            # Supporting data points
    detected_at: datetime
    status: str                          # "open" / "investigating" / "confirmed" / "dismissed"
    provenance_hash: str

# Portfolio Analysis
class PortfolioAnalysis(BaseModel):
    analysis_id: str
    operator_id: str
    analysis_date: date
    commodities: List[PortfolioCommodityEntry]  # Commodity positions
    portfolio_risk_score: Decimal               # Weighted aggregate (0-100)
    concentration_hhi: Decimal                  # Herfindahl-Hirschman Index
    concentration_level: str                    # "low" / "moderate" / "high"
    diversification_score: Decimal              # 0-100
    risk_contributions: Dict[EUDRCommodity, Decimal]  # % contribution to portfolio risk
    top_risk_driver: EUDRCommodity             # Commodity driving most risk
    recommendations: List[str]                  # Risk mitigation recommendations
    provenance_hash: str

class PortfolioCommodityEntry(BaseModel):
    commodity: EUDRCommodity
    volume_mt: Decimal                   # Annual import volume (metric tons)
    value_eur: Decimal                   # Annual import value (EUR)
    portfolio_share_volume: Decimal      # % of total volume
    portfolio_share_value: Decimal       # % of total value
    commodity_risk_score: Decimal        # From CommodityProfiler
    risk_contribution: Decimal           # Contribution to portfolio risk
    primary_countries: List[str]         # Top origin countries
```

### 7.4 Database Schema (New Migration: V106)

```sql
-- Migration: V106__agent_eudr_commodity_risk_analyzer.sql
-- Agent: GL-EUDR-CRA-018 Commodity Risk Analyzer
-- Prefix: gl_eudr_cra_

CREATE SCHEMA IF NOT EXISTS eudr_commodity_risk;

-- 1. Commodity profiles (deep risk profiles for each EUDR commodity)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_commodity_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    deforestation_drivers JSONB NOT NULL DEFAULT '[]',
    supply_chain_archetype JSONB NOT NULL DEFAULT '{}',
    processing_chain JSONB NOT NULL DEFAULT '[]',
    certification_landscape JSONB NOT NULL DEFAULT '[]',
    historical_compliance JSONB NOT NULL DEFAULT '{}',
    environmental_impact JSONB NOT NULL DEFAULT '{}',
    risk_factors JSONB NOT NULL DEFAULT '{}',
    composite_risk_score NUMERIC(5,2) NOT NULL DEFAULT 0.0,
    risk_level VARCHAR(20) NOT NULL DEFAULT 'standard',
    risk_weights JSONB NOT NULL DEFAULT '{}',
    top_producing_countries JSONB NOT NULL DEFAULT '[]',
    annex_i_cn_codes JSONB NOT NULL DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT uq_commodity_version UNIQUE (commodity, version)
);

-- 2. Derived products (Annex I products mapped to source commodities)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_derived_products (
    product_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_name VARCHAR(500) NOT NULL,
    cn_code VARCHAR(20) NOT NULL,
    hs_code VARCHAR(20),
    source_commodities JSONB NOT NULL DEFAULT '[]',
    source_commodity_shares JSONB NOT NULL DEFAULT '{}',
    processing_chain JSONB NOT NULL DEFAULT '[]',
    chain_depth INTEGER NOT NULL DEFAULT 1,
    custody_model VARCHAR(30) DEFAULT 'segregated',
    traceability_loss_points JSONB DEFAULT '[]',
    derived_risk_score NUMERIC(5,2) DEFAULT 0.0,
    risk_level VARCHAR(20) DEFAULT 'standard',
    annex_i_reference VARCHAR(100),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- 3. Price history (hypertable -- daily commodity prices)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_price_history (
    record_id UUID DEFAULT gen_random_uuid(),
    commodity VARCHAR(50) NOT NULL,
    price_date TIMESTAMPTZ NOT NULL,
    price_usd NUMERIC(12,4) NOT NULL,
    price_source VARCHAR(100) NOT NULL,
    currency_original VARCHAR(10) DEFAULT 'USD',
    exchange_rate NUMERIC(12,6),
    volume_traded NUMERIC(18,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_commodity_risk.gl_eudr_cra_price_history', 'price_date');

-- 4. Production forecasts
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_production_forecasts (
    forecast_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity VARCHAR(50) NOT NULL,
    country_code CHAR(2) NOT NULL,
    forecast_year INTEGER NOT NULL,
    forecast_month INTEGER,
    production_volume_mt NUMERIC(18,4) NOT NULL,
    confidence_interval_lower NUMERIC(18,4),
    confidence_interval_upper NUMERIC(18,4),
    trend_direction VARCHAR(20) DEFAULT 'stable',
    climate_impact_factor NUMERIC(5,4) DEFAULT 1.0,
    frontier_expansion_rate NUMERIC(5,4) DEFAULT 0.0,
    supply_demand_gap NUMERIC(18,4) DEFAULT 0.0,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_forecast UNIQUE (commodity, country_code, forecast_year, forecast_month)
);

-- 5. Substitution events
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_substitution_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    supplier_id UUID,
    detection_rule VARCHAR(100) NOT NULL,
    substitution_type VARCHAR(50) NOT NULL,
    commodity_from VARCHAR(50),
    commodity_to VARCHAR(50),
    country_from CHAR(2),
    country_to CHAR(2),
    volume_change_pct NUMERIC(8,4),
    price_differential_pct NUMERIC(8,4),
    confidence_score NUMERIC(5,2) NOT NULL DEFAULT 0.0,
    evidence JSONB DEFAULT '{}',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(30) DEFAULT 'open',
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    resolution_notes TEXT,
    provenance_hash VARCHAR(64) NOT NULL
);

-- 6. Regulatory requirements (EUDR article-to-commodity mapping)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_regulatory_requirements (
    requirement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity VARCHAR(50) NOT NULL,
    eudr_article VARCHAR(30) NOT NULL,
    requirement_type VARCHAR(100) NOT NULL,
    requirement_description TEXT NOT NULL,
    documentation_needed JSONB DEFAULT '[]',
    penalty_risk_level VARCHAR(20) DEFAULT 'standard',
    simplified_dd_eligible BOOLEAN DEFAULT FALSE,
    effective_date DATE,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- 7. DD workflows (commodity-specific due diligence workflows)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_dd_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    workflow_version INTEGER DEFAULT 1,
    status VARCHAR(30) DEFAULT 'in_progress',
    total_steps INTEGER NOT NULL,
    completed_steps INTEGER DEFAULT 0,
    completion_pct NUMERIC(5,2) DEFAULT 0.0,
    evidence_checklist JSONB NOT NULL DEFAULT '[]',
    verification_results JSONB DEFAULT '{}',
    risk_mitigation_plan JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    assigned_to VARCHAR(200),
    provenance_hash VARCHAR(64) NOT NULL
);

-- 8. Portfolio analyses
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_portfolio_analyses (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    analysis_date DATE NOT NULL,
    commodities JSONB NOT NULL DEFAULT '[]',
    portfolio_risk_score NUMERIC(5,2) NOT NULL DEFAULT 0.0,
    concentration_hhi NUMERIC(8,6) NOT NULL DEFAULT 0.0,
    concentration_level VARCHAR(20) DEFAULT 'low',
    diversification_score NUMERIC(5,2) DEFAULT 0.0,
    risk_contributions JSONB DEFAULT '{}',
    top_risk_driver VARCHAR(50),
    recommendations JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 9. Commodity risk scores (hypertable -- time-series risk scoring)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_commodity_risk_scores (
    score_id UUID DEFAULT gen_random_uuid(),
    commodity VARCHAR(50) NOT NULL,
    operator_id UUID,
    score_date TIMESTAMPTZ NOT NULL,
    composite_risk_score NUMERIC(5,2) NOT NULL,
    deforestation_pressure_score NUMERIC(5,2),
    supply_chain_opacity_score NUMERIC(5,2),
    processing_complexity_score NUMERIC(5,2),
    certification_coverage_score NUMERIC(5,2),
    regulatory_exposure_score NUMERIC(5,2),
    market_volatility_score NUMERIC(5,2),
    historical_compliance_score NUMERIC(5,2),
    price_adjusted_risk NUMERIC(5,2),
    risk_level VARCHAR(20) NOT NULL,
    risk_weights JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL
);

SELECT create_hypertable('eudr_commodity_risk.gl_eudr_cra_commodity_risk_scores', 'score_date');

-- 10. Processing chains (commodity-to-product transformation definitions)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_processing_chains (
    chain_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity VARCHAR(50) NOT NULL,
    chain_name VARCHAR(300) NOT NULL,
    stages JSONB NOT NULL DEFAULT '[]',
    total_stages INTEGER NOT NULL,
    critical_loss_points JSONB DEFAULT '[]',
    typical_custody_model VARCHAR(30) DEFAULT 'segregated',
    transparency_factor NUMERIC(4,2) DEFAULT 1.0,
    output_products JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- 11. Market indicators (supplementary market intelligence data)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_market_indicators (
    indicator_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity VARCHAR(50) NOT NULL,
    indicator_type VARCHAR(100) NOT NULL,
    indicator_name VARCHAR(300) NOT NULL,
    indicator_value NUMERIC(18,6) NOT NULL,
    indicator_unit VARCHAR(50),
    reference_date DATE NOT NULL,
    data_source VARCHAR(200) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- 12. Audit log (hypertable -- all agent operations)
CREATE TABLE eudr_commodity_risk.gl_eudr_cra_audit_log (
    log_id UUID DEFAULT gen_random_uuid(),
    operation VARCHAR(100) NOT NULL,
    engine VARCHAR(100) NOT NULL,
    operator_id UUID,
    commodity VARCHAR(50),
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    duration_ms NUMERIC(10,2),
    status VARCHAR(30) NOT NULL DEFAULT 'success',
    error_message TEXT,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_commodity_risk.gl_eudr_cra_audit_log', 'created_at');

-- Continuous Aggregates

-- Daily price averages per commodity
CREATE MATERIALIZED VIEW eudr_commodity_risk.gl_eudr_cra_daily_price_avg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', price_date) AS bucket,
    commodity,
    AVG(price_usd) AS avg_price_usd,
    MIN(price_usd) AS min_price_usd,
    MAX(price_usd) AS max_price_usd,
    COUNT(*) AS record_count
FROM eudr_commodity_risk.gl_eudr_cra_price_history
GROUP BY bucket, commodity;

-- Weekly risk score summaries per commodity
CREATE MATERIALIZED VIEW eudr_commodity_risk.gl_eudr_cra_weekly_risk_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', score_date) AS bucket,
    commodity,
    AVG(composite_risk_score) AS avg_risk_score,
    MAX(composite_risk_score) AS max_risk_score,
    MIN(composite_risk_score) AS min_risk_score,
    COUNT(*) AS score_count,
    AVG(price_adjusted_risk) AS avg_price_adjusted_risk
FROM eudr_commodity_risk.gl_eudr_cra_commodity_risk_scores
GROUP BY bucket, commodity;

-- Indexes
CREATE INDEX idx_cra_profiles_commodity ON eudr_commodity_risk.gl_eudr_cra_commodity_profiles(commodity);
CREATE INDEX idx_cra_products_cn_code ON eudr_commodity_risk.gl_eudr_cra_derived_products(cn_code);
CREATE INDEX idx_cra_products_commodity ON eudr_commodity_risk.gl_eudr_cra_derived_products USING gin(source_commodities);
CREATE INDEX idx_cra_forecasts_commodity ON eudr_commodity_risk.gl_eudr_cra_production_forecasts(commodity, country_code);
CREATE INDEX idx_cra_substitutions_operator ON eudr_commodity_risk.gl_eudr_cra_substitution_events(operator_id);
CREATE INDEX idx_cra_substitutions_status ON eudr_commodity_risk.gl_eudr_cra_substitution_events(status);
CREATE INDEX idx_cra_reqs_commodity ON eudr_commodity_risk.gl_eudr_cra_regulatory_requirements(commodity);
CREATE INDEX idx_cra_dd_operator ON eudr_commodity_risk.gl_eudr_cra_dd_workflows(operator_id, commodity);
CREATE INDEX idx_cra_dd_status ON eudr_commodity_risk.gl_eudr_cra_dd_workflows(status);
CREATE INDEX idx_cra_portfolio_operator ON eudr_commodity_risk.gl_eudr_cra_portfolio_analyses(operator_id);
CREATE INDEX idx_cra_chains_commodity ON eudr_commodity_risk.gl_eudr_cra_processing_chains(commodity);
CREATE INDEX idx_cra_market_commodity ON eudr_commodity_risk.gl_eudr_cra_market_indicators(commodity, indicator_type);
CREATE INDEX idx_cra_audit_operation ON eudr_commodity_risk.gl_eudr_cra_audit_log(operation, engine);
CREATE INDEX idx_cra_audit_operator ON eudr_commodity_risk.gl_eudr_cra_audit_log(operator_id);
```

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **Commodity Profiles** | | |
| GET | `/api/v1/eudr-cra/commodities` | List all 7 commodity profiles with summary risk scores |
| GET | `/api/v1/eudr-cra/commodities/{commodity}` | Get deep commodity profile with all risk factors |
| POST | `/api/v1/eudr-cra/commodities/{commodity}/score` | Calculate/recalculate commodity risk score |
| GET | `/api/v1/eudr-cra/commodities/compare` | Compare 2+ commodity risk profiles side-by-side |
| GET | `/api/v1/eudr-cra/commodities/{commodity}/history` | Get commodity risk score history (time series) |
| **Derived Products** | | |
| GET | `/api/v1/eudr-cra/derived-products` | List derived products with filters (commodity, CN code) |
| GET | `/api/v1/eudr-cra/derived-products/{product_id}` | Get derived product details with processing chain |
| POST | `/api/v1/eudr-cra/derived-products/analyze` | Analyze derived product risk from commodity sources |
| GET | `/api/v1/eudr-cra/derived-products/lookup/{cn_code}` | Look up derived product by CN code |
| GET | `/api/v1/eudr-cra/derived-products/{product_id}/chain` | Get full processing chain for a derived product |
| **Price/Market Analysis** | | |
| GET | `/api/v1/eudr-cra/prices/{commodity}` | Get price history for a commodity (date range filter) |
| GET | `/api/v1/eudr-cra/prices/{commodity}/volatility` | Get volatility metrics for a commodity |
| GET | `/api/v1/eudr-cra/prices/{commodity}/correlation` | Get price-deforestation correlation analysis |
| GET | `/api/v1/eudr-cra/prices/market-alerts` | Get active market disruption alerts |
| POST | `/api/v1/eudr-cra/prices/ingest` | Ingest price data (batch upload) |
| **Production Forecasts** | | |
| GET | `/api/v1/eudr-cra/forecasts/{commodity}` | Get production forecasts for a commodity |
| GET | `/api/v1/eudr-cra/forecasts/{commodity}/{country_code}` | Get forecast for specific commodity-country |
| POST | `/api/v1/eudr-cra/forecasts/{commodity}/generate` | Generate/refresh production forecast |
| GET | `/api/v1/eudr-cra/forecasts/{commodity}/heatmap` | Get production risk heatmap data |
| **Substitution Risk** | | |
| POST | `/api/v1/eudr-cra/substitution/analyze/{operator_id}` | Run substitution analysis for an operator |
| GET | `/api/v1/eudr-cra/substitution/events` | List substitution events (with filters) |
| GET | `/api/v1/eudr-cra/substitution/events/{event_id}` | Get substitution event details with evidence |
| PUT | `/api/v1/eudr-cra/substitution/events/{event_id}/status` | Update substitution event status |
| **Regulatory Compliance** | | |
| GET | `/api/v1/eudr-cra/regulatory/{commodity}` | Get commodity-specific regulatory requirements |
| GET | `/api/v1/eudr-cra/regulatory/{commodity}/checklist` | Get commodity compliance checklist |
| POST | `/api/v1/eudr-cra/regulatory/validate-cn-code` | Validate CN code against Annex I scope |
| GET | `/api/v1/eudr-cra/regulatory/{commodity}/penalties` | Get penalty risk assessment per commodity |
| **Due Diligence Workflows** | | |
| POST | `/api/v1/eudr-cra/dd/workflows` | Create a commodity DD workflow for an operator |
| GET | `/api/v1/eudr-cra/dd/workflows` | List DD workflows (with filters: operator, commodity, status) |
| GET | `/api/v1/eudr-cra/dd/workflows/{workflow_id}` | Get DD workflow details with progress |
| PUT | `/api/v1/eudr-cra/dd/workflows/{workflow_id}/evidence` | Upload evidence for a DD workflow step |
| PUT | `/api/v1/eudr-cra/dd/workflows/{workflow_id}/complete` | Mark DD workflow as complete |
| **Portfolio Aggregation** | | |
| POST | `/api/v1/eudr-cra/portfolio/analyze` | Run portfolio risk analysis for an operator |
| GET | `/api/v1/eudr-cra/portfolio/{operator_id}` | Get latest portfolio analysis |
| GET | `/api/v1/eudr-cra/portfolio/{operator_id}/history` | Get portfolio risk history |
| GET | `/api/v1/eudr-cra/portfolio/{operator_id}/export` | Export portfolio risk report (PDF/CSV/JSON) |
| **Health** | | |
| GET | `/api/v1/eudr-cra/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_cra_profiles_accessed_total` | Counter | Commodity profile access count by commodity |
| 2 | `gl_eudr_cra_risk_scores_calculated_total` | Counter | Commodity risk score calculations by commodity |
| 3 | `gl_eudr_cra_derived_products_analyzed_total` | Counter | Derived product analyses performed |
| 4 | `gl_eudr_cra_price_records_ingested_total` | Counter | Price records ingested by commodity |
| 5 | `gl_eudr_cra_volatility_calculations_total` | Counter | Volatility metric calculations |
| 6 | `gl_eudr_cra_forecasts_generated_total` | Counter | Production forecasts generated by commodity |
| 7 | `gl_eudr_cra_substitution_analyses_total` | Counter | Substitution risk analyses performed |
| 8 | `gl_eudr_cra_substitution_events_detected_total` | Counter | Substitution events detected by type |
| 9 | `gl_eudr_cra_regulatory_checks_total` | Counter | Regulatory compliance checks performed |
| 10 | `gl_eudr_cra_dd_workflows_created_total` | Counter | DD workflows created by commodity |
| 11 | `gl_eudr_cra_dd_workflows_completed_total` | Counter | DD workflows completed by commodity |
| 12 | `gl_eudr_cra_portfolio_analyses_total` | Counter | Portfolio risk analyses performed |
| 13 | `gl_eudr_cra_processing_duration_seconds` | Histogram | Processing latency by engine and operation |
| 14 | `gl_eudr_cra_api_request_duration_seconds` | Histogram | API endpoint response latency |
| 15 | `gl_eudr_cra_errors_total` | Counter | Errors by engine and operation type |
| 16 | `gl_eudr_cra_active_dd_workflows` | Gauge | Currently active DD workflows |
| 17 | `gl_eudr_cra_market_alerts_active` | Gauge | Currently active market disruption alerts |
| 18 | `gl_eudr_cra_price_data_staleness_hours` | Gauge | Hours since last price data update per commodity |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for prices and risk scores |
| Cache | Redis | Commodity profile caching, price query caching, derived product lookup |
| Object Storage | S3 | Portfolio reports, forecast exports, reference data snapshots |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Numeric | Python Decimal | Deterministic arithmetic for all risk calculations |
| Statistics | NumPy + SciPy (deterministic only) | Volatility calculation, regression, correlation |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control per engine |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across engine calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-cra:commodities:read` | View commodity profiles and risk scores | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cra:commodities:write` | Update commodity profile data | Compliance Officer, Admin |
| `eudr-cra:derived-products:read` | View derived product analysis | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cra:derived-products:analyze` | Trigger derived product risk analysis | Analyst, Compliance Officer, Admin |
| `eudr-cra:prices:read` | View price data and volatility metrics | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cra:prices:ingest` | Upload/ingest price data | Data Engineer, Admin |
| `eudr-cra:forecasts:read` | View production forecasts | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cra:forecasts:generate` | Trigger forecast generation | Analyst, Admin |
| `eudr-cra:substitution:read` | View substitution risk events | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cra:substitution:analyze` | Trigger substitution analysis | Analyst, Compliance Officer, Admin |
| `eudr-cra:substitution:manage` | Update substitution event status | Compliance Officer, Admin |
| `eudr-cra:regulatory:read` | View regulatory requirements and checklists | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cra:dd:read` | View DD workflow status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cra:dd:manage` | Create, update, complete DD workflows | Compliance Officer, Admin |
| `eudr-cra:dd:evidence` | Upload evidence to DD workflows | Analyst, Compliance Officer, Admin |
| `eudr-cra:portfolio:read` | View portfolio risk analysis | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-cra:portfolio:analyze` | Trigger portfolio risk analysis | Analyst, Compliance Officer, Admin |
| `eudr-cra:portfolio:export` | Export portfolio risk reports | Analyst, Compliance Officer, Admin |
| `eudr-cra:audit:read` | View audit log and provenance data | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent / System | Integration | Data Flow |
|----------------|-------------|-----------|
| AGENT-EUDR-016 Country Risk Evaluator | Country risk data | Country risk classifications feed into commodity-country cross-analysis |
| AGENT-EUDR-001 Supply Chain Mapping Master | Supply chain graph data | Supply chain structure informs commodity processing chain analysis |
| AGENT-DATA-005 EUDR Traceability Connector | Chain of custody data | Custody records for derived product traceability analysis |
| AGENT-DATA-007 Deforestation Satellite Connector | Deforestation alerts | Satellite data for price-deforestation correlation |
| AGENT-DATA-003 ERP/Finance Connector | Procurement records | Volume/value data for substitution detection and portfolio analysis |
| AGENT-DATA-020 Climate Hazard Connector | Climate data | Climate impact factors for production forecasting |
| AGENT-FOUND-005 Citations & Evidence | Regulatory references | EUDR article citations for compliance engine |
| AGENT-FOUND-008 Reproducibility Agent | Determinism verification | Bit-perfect verification of risk calculations |
| External: World Bank Commodity Prices | Price data | Daily commodity price indices |
| External: FAOSTAT | Production statistics | Historical production volumes per commodity per country |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | API integration | Commodity risk data -> frontend dashboards, DD workflows |
| AGENT-EUDR-001 Supply Chain Mapping Master | Risk enrichment | Commodity risk scores enrich supply chain node risk |
| AGENT-EUDR-016 Country Risk Evaluator | Commodity dimension | Commodity risk feeds into country-commodity cross-analysis |
| AGENT-DATA-005 DueDiligenceEngine | DD commodity section | Commodity-specific DD data for DDS generation |
| GL-EUDR-APP DDS Reporting Engine | Commodity risk section | Commodity risk summary for DDS submission |
| External Auditors | Read-only API + exports | Commodity risk reports for third-party verification |
| Financial Risk Teams | Portfolio exports | Portfolio risk reports for risk committee review |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Commodity Risk Assessment (Commodity Risk Manager)

```
1. Risk manager logs in to GL-EUDR-APP
2. Navigates to "Commodity Risk Analyzer" module
3. Dashboard shows overview of all 7 commodities with risk score cards
4. Clicks on "Oil Palm" commodity -> deep profile view
   - Sees composite risk score: 78/100 (HIGH)
   - Reviews 50+ risk factors organized by category
   - Views deforestation drivers specific to oil palm (plantation clearing, peatland drainage)
   - Reviews certification landscape (RSPO coverage: 19% globally)
5. Navigates to "Derived Products" tab -> sees palm oil derivatives
   - Palm kernel oil: risk 72 (attenuation from better traceability)
   - Biodiesel from palm oil: risk 85 (amplification from complex processing)
   - Margarine: risk 68 (attenuation from mass balance + EU processing)
6. Checks "Price & Market" tab -> sees 3-month price spike of 28%
   - Price-adjusted risk now 89/100
   - Alert: "Market disruption detected -- elevated deforestation pressure expected"
7. Reviews "Production Forecast" -> Indonesia production projected -5% due to El Nino
   - Supply-demand gap widening -> frontier expansion risk elevated
8. Exports commodity risk report for board presentation
```

#### Flow 2: Derived Product Compliance (Compliance Officer)

```
1. Compliance officer needs to file DDS for chocolate product shipment
2. Opens "Derived Products" section in Commodity Risk Analyzer
3. Looks up chocolate (CN code 1806.31) -> system shows:
   - Source commodities: cocoa (92%), soya lecithin (5%), palm oil (3%)
   - Processing chain: cocoa bean -> paste -> butter/powder -> chocolate
   - Critical traceability loss points: cooperative aggregation, roasting/grinding blend
   - Derived product risk: 71/100 (driven by cocoa origin opacity)
4. System highlights: "This product contains 3 EUDR commodities -- all require DD"
5. Officer reviews commodity-specific DD checklist for each source commodity
6. Initiates DD workflows for cocoa, soya, and palm oil components
7. Each workflow provides tailored evidence requirements and verification protocols
```

#### Flow 3: Substitution Detection (Procurement Strategist)

```
1. Procurement strategist receives substitution alert notification
2. Opens "Substitution Risk" section
3. Alert shows: "Supplier XYZ -- potential commodity switch detected"
   - Palm oil volume from Supplier XYZ dropped 35% in Q4
   - Soya oil volume from same supplier increased 40% in same period
   - Price differential: soya oil 22% cheaper than palm oil at time of switch
   - Confidence: 87%
4. Strategist reviews evidence: volume comparison charts, price overlay, timeline
5. Investigates: contacts supplier for explanation
6. Outcome A: Legitimate reformulation -> marks event as "dismissed" with notes
7. Outcome B: Undisclosed substitution -> escalates to compliance team
   - System flags: "If soya oil sourcing region is high-risk, EUDR DD required"
```

#### Flow 4: Portfolio Risk Review (Financial Risk Analyst)

```
1. Risk analyst runs quarterly portfolio analysis
2. Opens "Portfolio Risk" section -> triggers new analysis
3. System aggregates across all commodity positions:
   - Total portfolio: 5 commodities (cocoa, palm oil, soya, coffee, rubber)
   - Portfolio risk score: 68/100
   - Concentration HHI: 0.28 (HIGH -- palm oil is 52% of portfolio value)
   - Diversification score: 42/100 (low)
4. Risk contribution analysis shows:
   - Palm oil: 48% of portfolio risk (driven by high volume + high commodity risk)
   - Cocoa: 28% (high commodity risk despite lower volume)
   - Others: 24% combined
5. Recommendations generated:
   - "Reduce palm oil concentration below 40% of portfolio value"
   - "Diversify cocoa sourcing to include low-risk country origins"
   - "Consider certified rubber sources to reduce rubber risk component"
6. Analyst exports portfolio report (PDF) for risk committee meeting
```

### 8.2 Key Screen Descriptions

**Commodity Dashboard:**
- Card-based overview of all 7 EUDR commodities
- Each card shows: commodity name, composite risk score (gauge chart), risk level badge, trend arrow
- Cards ordered by risk score (highest first)
- Quick filters: show only HIGH risk, show trend changes
- "Compare" button to select 2+ commodities for side-by-side view

**Commodity Deep Profile View:**
- Header: commodity name, risk score (large gauge), risk level badge
- Tab navigation: Risk Factors | Derived Products | Price & Market | Production | Substitution | Regulatory | DD Workflows
- Risk Factors tab: categorized risk factors with individual scores and weights
- Spider/radar chart showing 7 risk category scores
- Historical risk trend line chart (monthly over 5 years)

**Derived Product Explorer:**
- Searchable/filterable list of Annex I products
- Filter by source commodity, CN code prefix, risk level
- Product detail view: processing chain visualization (flow diagram)
- Multi-commodity products highlighted with stacked bar showing commodity composition

**Portfolio Risk Dashboard:**
- Portfolio risk gauge (0-100) with trend
- Concentration donut chart (commodity proportions)
- Risk contribution waterfall chart
- Diversification score meter
- Commodity x Country risk heatmap (matrix view)
- Export button (PDF, CSV, JSON)

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 8 P0 features (Features 1-8) implemented and tested
  - [ ] Feature 1: Commodity Profiler -- 7 commodity profiles with 50+ risk factors each
  - [ ] Feature 2: Derived Product Analyzer -- 400+ Annex I products mapped with processing chains
  - [ ] Feature 3: Price Volatility Engine -- daily price tracking, volatility calculation, correlation
  - [ ] Feature 4: Production Forecast Engine -- 12-month forecasts for top 20 countries per commodity
  - [ ] Feature 5: Substitution Risk Analyzer -- 5 detection rules operational with evidence
  - [ ] Feature 6: Regulatory Compliance Engine -- all EUDR Articles mapped per commodity
  - [ ] Feature 7: Commodity DD Engine -- 7 commodity-specific DD workflows
  - [ ] Feature 8: Portfolio Risk Aggregator -- HHI, diversification, risk contribution
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC integrated with 19 permissions)
- [ ] Performance targets met (< 500ms risk score p99; < 2s portfolio analysis)
- [ ] All risk calculations verified deterministic (bit-perfect reproducibility)
- [ ] Database migration V106 tested and validated (12 tables + 3 hypertables + 2 continuous aggregates)
- [ ] API documentation complete (OpenAPI spec for 30+ endpoints)
- [ ] Integration with EUDR-016, EUDR-001, DATA-005, DATA-007 verified
- [ ] 5 beta customers successfully analyzed commodity risk portfolios
- [ ] All 7 commodity profiles validated by domain experts
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ commodity risk analyses performed by customers
- All 7 commodity profiles accessed at least 50 times each
- 50+ derived product analyses completed
- Price data current (< 24 hours stale) for all 7 commodities
- < 3 support tickets per customer
- p99 API latency < 200ms

**60 Days:**
- 500+ commodity risk analyses performed
- 100+ substitution analysis runs
- 50+ portfolio risk analyses completed
- 10+ substitution events detected and investigated
- Production forecasts within 15% accuracy for available actuals
- Price-deforestation correlation coefficients validated for 3+ commodities

**90 Days:**
- 1,000+ commodity risk analyses
- 200+ portfolio analyses across 100+ operators
- 200+ DD workflows initiated, 80%+ completion rate
- Zero commodity-specific compliance failures for active customers
- Substitution detection precision >= 90% (validated by audit sample)
- NPS > 50 from commodity risk manager persona

---

## 10. Timeline and Milestones

### Phase 1: Core Commodity Intelligence (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Commodity Profiler (Engine 1): 7 profiles, 50+ risk factors, scoring formula | Senior Backend Engineer |
| 2-3 | Derived Product Analyzer (Engine 2): Annex I mapping, processing chains, risk adjustment | Senior Backend Engineer |
| 3-4 | Price Volatility Engine (Engine 3): price ingestion, volatility calculation, correlation | Backend + Data Engineer |
| 4-5 | Production Forecast Engine (Engine 4): yield modeling, climate factors, seasonal patterns | Backend + Data Engineer |
| 5-6 | Substitution Risk Analyzer (Engine 5): 5 detection rules, evidence collection, alerting | Senior Backend Engineer |

**Milestone: Core 5 engines operational with deterministic risk calculations (Week 6)**

### Phase 2: Compliance, Workflow, and Aggregation (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Regulatory Compliance Engine (Engine 6): article mapping, Annex I validation, checklists | Backend + Regulatory Specialist |
| 8-9 | Commodity DD Engine (Engine 7): 7 commodity workflows, evidence tracking, verification | Backend Engineer |
| 9-10 | Portfolio Risk Aggregator (Engine 8): HHI, diversification, risk contribution, export | Backend Engineer |

**Milestone: All 8 engines operational (Week 10)**

### Phase 3: API Layer and Integration (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | REST API Layer: 30+ endpoints, authentication, rate limiting, OpenAPI docs | Backend Engineer |
| 12-13 | Integration with EUDR-016, EUDR-001, DATA-005, DATA-007, DATA-020 | Backend Engineer |
| 13-14 | Reference data loading: Annex I products, price history, production statistics | Data Engineer |

**Milestone: Full API operational with cross-agent integration (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 900+ tests, golden tests for all 7 commodities | Test Engineer |
| 16-17 | Performance testing, security audit, load testing, determinism verification | DevOps + Security |
| 17 | Database migration V106 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers), domain expert profile validation | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 8 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Commodity scenario simulator (Feature 9)
- Commodity benchmark analytics (Feature 10)
- AI-assisted commodity classification (Feature 11)
- Additional derived product chain definitions
- Enhanced climate impact modeling

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Stable, production-ready; provides country risk data |
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable; provides supply chain graph data |
| AGENT-DATA-005 EUDR Traceability Connector | BUILT (100%) | Low | Stable; provides chain-of-custody data |
| AGENT-DATA-007 Deforestation Satellite Connector | BUILT (100%) | Low | Stable; provides deforestation alerts |
| AGENT-DATA-003 ERP/Finance Connector | BUILT (100%) | Low | Stable; provides procurement data |
| AGENT-DATA-020 Climate Hazard Connector | BUILT (100%) | Low | Stable; provides climate impact data |
| AGENT-DATA-008 Supplier Questionnaire Processor | BUILT (100%) | Low | Stable; provides supplier evidence |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration points defined |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| AGENT-FOUND-005 Citations & Evidence | BUILT (100%) | Low | Provides regulatory reference citations |
| AGENT-FOUND-008 Reproducibility Agent | BUILT (100%) | Low | Determinism verification |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR Annex I product list (CN codes) | Published, stable | Low | Static reference data; update on amendment |
| World Bank Commodity Prices | Available, API access | Medium | Cache locally; fallback to alternative sources (IMF) |
| FAOSTAT production data | Available, updated annually | Medium | Cache locally; supplement with USDA FAS estimates |
| EU Combined Nomenclature database | Published annually | Low | Static reference; annual refresh |
| EC country benchmarking list | Published, updated periodically | Medium | Database-driven; hot-reloadable |
| Climate data (NOAA, ERA5) | Available | Low | Multi-provider fallback via AGENT-DATA-020 |
| EC EUDR implementing regulations | Evolving | Medium | Configuration-driven compliance rules |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Commodity price data sources become unavailable or change API | Medium | Medium | Multi-source fallback (World Bank, IMF, ICE); local cache with 30-day buffer |
| R2 | Production statistics lagged (FAOSTAT updates annually) | High | Medium | Supplement with USDA FAS monthly estimates; use interpolation for monthly granularity |
| R3 | Annex I amended with new derived products or commodities | Medium | High | Extensible derived product framework; new products added via configuration, not code |
| R4 | Substitution detection generates excessive false positives | Medium | Medium | Tunable thresholds per operator; confidence scoring; human-in-the-loop confirmation |
| R5 | Climate impact factors inaccurate for production forecasting | Medium | Medium | Conservative confidence intervals; multiple climate data sources; quarterly model recalibration |
| R6 | Operators handle commodity variants not in standard profiles (e.g., shea butter, rubber wood) | Low | Medium | Custom derived product definition support; extensible commodity profile framework |
| R7 | Portfolio risk formula weights disputed by different customers | Medium | Low | Fully configurable weights per operator; industry-standard defaults; weight sensitivity analysis |
| R8 | Integration complexity with 10+ upstream agents | Medium | Medium | Well-defined interfaces; mock adapters; circuit breaker pattern; graceful degradation |
| R9 | EU regulation amended or new commodities added to scope | Low | High | Modular commodity profile system; new commodity profile added in days, not weeks |
| R10 | Competitive tools launch commodity-specific analytics before GreenLang | Medium | Medium | Deeper ecosystem integration; 8-engine depth no competitor matches; faster iteration |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Commodity Profiler Unit Tests | 120+ | Profile creation, risk factor scoring, composite calculation, comparison, trending |
| Derived Product Analyzer Tests | 100+ | Annex I mapping, processing chain trace, multi-commodity products, risk adjustment |
| Price Volatility Engine Tests | 90+ | Price ingestion, volatility calculation, correlation, market disruption detection |
| Production Forecast Engine Tests | 80+ | Yield modeling, climate impact, seasonal patterns, forecast accuracy |
| Substitution Risk Analyzer Tests | 80+ | All 5 detection rules, evidence quality, false positive rates, edge cases |
| Regulatory Compliance Engine Tests | 70+ | Article mapping, CN code validation, checklist generation, penalty assessment |
| Commodity DD Engine Tests | 80+ | Workflow creation, evidence tracking, verification protocols, completion |
| Portfolio Risk Aggregator Tests | 70+ | HHI calculation, diversification scoring, risk contribution, what-if, export |
| API Tests | 90+ | All 30+ endpoints, auth, error handling, pagination, rate limiting |
| Golden Tests | 56+ | All 7 commodities with 8 scenarios each (see 13.2) |
| Integration Tests | 40+ | Cross-agent integration with EUDR-016, EUDR-001, DATA-005, DATA-007 |
| Performance Tests | 25+ | Risk scoring latency, portfolio analysis, price data bulk ingestion |
| Determinism Tests | 20+ | Bit-perfect reproducibility for all calculation engines |
| **Total** | **900+** | |

### 13.2 Golden Test Scenarios per Commodity

Each of the 7 commodities will have dedicated golden test scenarios:

1. **Full profile scoring** -- Complete commodity profile with all risk factors populated, expect deterministic composite score
2. **Derived product chain** -- Full processing chain from raw commodity to final product, expect correct risk adjustment
3. **Price volatility spike** -- Inject price spike, expect market disruption detection and risk adjustment
4. **Production decline** -- Model production decline scenario, expect elevated frontier expansion risk
5. **Substitution detected** -- Inject commodity switch pattern, expect detection with correct evidence
6. **Regulatory checklist** -- Generate commodity-specific checklist, expect all applicable EUDR Articles covered
7. **DD workflow completion** -- Run full DD workflow, expect correct evidence requirements and verification
8. **Portfolio with commodity** -- Include commodity in portfolio, expect correct HHI contribution

Total: 7 commodities x 8 scenarios = 56 golden test scenarios

### 13.3 Determinism Verification Protocol

For every calculation engine (Engines 1-8), the following determinism protocol applies:

1. **Seed Test**: Run calculation with fixed input 100 times -> expect bit-perfect identical output 100/100
2. **Cross-Platform Test**: Run on Linux and Windows -> expect identical results
3. **Decimal Verification**: All monetary and risk calculations use Python Decimal, not float
4. **Provenance Hash Test**: SHA-256 hash of output must be identical across runs with same input
5. **Reproducibility Agent Integration**: AGENT-FOUND-008 independently verifies determinism

---

## 14. Security and Compliance

### 14.1 Data Security

| Requirement | Implementation |
|-------------|----------------|
| Authentication | JWT (RS256) via SEC-001 for all API endpoints |
| Authorization | RBAC via SEC-002 with 19 permissions across 6 roles |
| Encryption at Rest | AES-256-GCM via SEC-003 for all database tables |
| Encryption in Transit | TLS 1.3 via SEC-004 for all API communications |
| Audit Logging | SEC-005 integration for all operations; `gl_eudr_cra_audit_log` hypertable |
| Secrets Management | HashiCorp Vault via SEC-006 for API keys, database credentials |
| PII Detection | SEC-011 PII redaction for any supplier-identifiable data in logs |

### 14.2 Data Classification

| Data Type | Classification | Handling |
|-----------|---------------|----------|
| Commodity profiles (public risk factors) | Internal | Standard access control |
| Commodity price data | Internal | Standard access control; sourced from public indices |
| Operator procurement volumes | Confidential | RBAC-restricted; encrypted at rest |
| Supplier substitution evidence | Confidential | RBAC-restricted; encrypted at rest; audit-logged |
| DD workflow evidence documents | Confidential | RBAC-restricted; encrypted at rest; S3 with access logging |
| Portfolio risk analyses | Confidential | Operator-isolated; RBAC-restricted; encrypted |
| Production forecasts | Internal | Standard access control |

### 14.3 Regulatory Compliance

| Requirement | Implementation |
|-------------|----------------|
| EUDR Article 31 (Record keeping 5 years) | All data retained in TimescaleDB with 5-year retention policy; provenance hashes |
| EUDR Article 12 (DDS submission data) | Commodity risk data formatted for DDS integration |
| GDPR (if supplier PII involved) | PII detection via SEC-011; data minimization; purpose limitation |
| SOC 2 Type II | Audit trail via SEC-005; access controls; change management |

---

## 15. Deployment Plan

### 15.1 Infrastructure Requirements

| Resource | Specification | Justification |
|----------|--------------|---------------|
| Kubernetes Pods | 3 replicas (min), 8 replicas (max) | High availability; auto-scaling for portfolio analysis bursts |
| CPU | 2 vCPU per pod | Statistical calculations (volatility, regression, correlation) |
| Memory | 4 GB per pod | Price history caching; commodity profile storage |
| PostgreSQL Storage | 50 GB initial, auto-expand | Price history, risk scores, audit logs |
| Redis Cache | 2 GB | Commodity profiles, derived product lookups, price queries |
| S3 Storage | 10 GB | Portfolio reports, reference data, DD evidence |

### 15.2 Deployment Stages

| Stage | Environment | Duration | Gate |
|-------|-------------|----------|------|
| 1. Development | dev-eks | Weeks 1-14 | All tests pass; code review approved |
| 2. Staging | staging-eks | Weeks 15-16 | Integration tests pass; performance benchmarks met |
| 3. Canary | prod-eks (5% traffic) | Week 17 | No errors; latency within SLO; monitoring green |
| 4. Production | prod-eks (100% traffic) | Week 18 | Canary gates passed; beta customer sign-off |

### 15.3 Monitoring and Alerting

| Alert | Condition | Severity | Channel |
|-------|-----------|----------|---------|
| Risk Calculation Error | `gl_eudr_cra_errors_total` > 10/min | Critical | PagerDuty + Slack |
| Price Data Stale | `gl_eudr_cra_price_data_staleness_hours` > 48 | High | Slack + Email |
| API Latency Spike | `gl_eudr_cra_api_request_duration_seconds` p99 > 1s | High | Slack |
| Substitution Alert Backlog | Unresolved events > 50 | Medium | Email |
| DD Workflow Stale | Active workflow with no activity > 14 days | Medium | Email |
| Portfolio Analysis Failure | Failed analysis count > 3/hour | High | Slack |

### 15.4 Rollback Plan

1. **Automated rollback**: Kubernetes deployment rolls back automatically if health checks fail within 5 minutes of deployment
2. **Manual rollback**: `kubectl rollout undo deployment/eudr-cra` restores previous version
3. **Database rollback**: V106 migration has DOWN migration script for clean schema removal
4. **Feature flag**: `GL_EUDR_CRA_ENABLED` environment variable disables all endpoints without deployment

---

## 16. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **CN Code** | Combined Nomenclature -- EU product classification code |
| **HS Code** | Harmonized System -- international product classification code |
| **Annex I** | EUDR Annex listing all regulated products with CN codes, organized by commodity |
| **HHI** | Herfindahl-Hirschman Index -- measure of market/portfolio concentration |
| **Volatility** | Statistical measure of the dispersion of commodity price returns |
| **Derived Product** | A product made from, containing, or fed with one of the 7 EUDR commodities |
| **Processing Chain** | The sequence of transformation stages from raw commodity to final product |
| **Substitution Risk** | Risk that a supplier switches commodities or sourcing regions to circumvent EUDR |
| **Mass Balance** | Chain of custody model where compliant and non-compliant material may be mixed but quantities tracked |
| **Identity Preserved** | Chain of custody model where compliant material is physically separated throughout |
| **Segregated** | Chain of custody model where compliant material is kept separate from non-compliant |
| **Frontier Expansion** | Conversion of new land (often forest) for commodity production |
| **RSPO** | Roundtable on Sustainable Palm Oil -- certification scheme for palm oil |
| **FSC** | Forest Stewardship Council -- certification scheme for wood/timber |
| **4C** | Common Code for the Coffee Community -- coffee sustainability certification |
| **RTRS** | Round Table on Responsible Soy Association -- soya certification |

### Appendix B: EUDR Annex I Commodity-Product Mapping (Summary)

| Commodity | Annex I Chapter | Key CN Code Ranges | Product Count |
|-----------|----------------|--------------------|----|
| Cattle | Ex Chapter 1, 2, 41, 42, 43 | 0102, 0201-0202, 4101-4115, 4201-4206 | 60+ |
| Cocoa | Chapter 18 | 1801, 1802, 1803, 1804, 1805, 1806 | 25+ |
| Coffee | Chapter 9, 21 | 0901, 2101.11, 2101.12 | 15+ |
| Oil Palm | Chapter 15 | 1511, 1513, 1516, 3823, 3826 | 40+ |
| Rubber | Chapter 40 | 4001, 4005, 4006, 4007, 4008, 4010-4017 | 50+ |
| Soya | Chapter 12, 15, 23 | 1201, 1507, 1508, 2304 | 20+ |
| Wood | Chapters 44, 47, 48, 49, 94 | 4401-4421, 4701-4707, 4801-4823, 9401-9403 | 150+ |
| **Total** | | | **400+** |

### Appendix C: Commodity Risk Factor Categories (Detail)

**Category 1: Deforestation Pressure (Weight: 0.25)**
- Historical deforestation rate associated with commodity production
- Land-use change correlation with commodity price movements
- Frontier expansion rate in top producing regions
- Indirect land-use change (iLUC) risk specific to commodity

**Category 2: Supply Chain Opacity (Weight: 0.20)**
- Average supply chain depth (tiers) for the commodity
- Number of aggregation points where traceability commonly breaks
- Percentage of production from smallholders (harder to trace)
- Availability of digital traceability infrastructure in producing regions

**Category 3: Processing Complexity (Weight: 0.15)**
- Number of processing stages from raw commodity to final product
- Batch mixing frequency during processing
- Multi-commodity blending prevalence
- Processing stage documentation availability

**Category 4: Certification Coverage (Weight: 0.15)**
- Percentage of global production covered by credible certification
- Number of certification schemes available
- Certification scheme credibility rating
- Rate of certification growth/decline

**Category 5: Regulatory Exposure (Weight: 0.10)**
- Number of EUDR Articles with commodity-specific requirements
- Complexity of geolocation requirements for this commodity
- Documentation burden relative to other commodities
- Historical enforcement action rate for this commodity

**Category 6: Market Volatility (Weight: 0.10)**
- 5-year average price volatility
- Frequency of supply shocks
- Price-deforestation correlation strength
- Market concentration (few buyers/sellers increase manipulation risk)

**Category 7: Historical Compliance (Weight: 0.05)**
- Known non-compliance events in last 5 years
- Seizure/confiscation events for this commodity at EU borders
- Industry self-reported compliance challenges
- NGO/media deforestation investigations for this commodity

### Appendix D: Price-Deforestation Correlation Methodology

The PriceVolatilityEngine calculates the Pearson correlation coefficient between commodity price movements and deforestation alert frequency:

```
Correlation(Price, Deforestation) = Cov(P, D) / (StdDev(P) * StdDev(D))

Where:
- P = monthly price change (%) for the commodity
- D = monthly deforestation alert count in top producing regions (from AGENT-DATA-007)
- Time window: 36 months rolling
- Lag options: 0, 3, 6, 12 months (deforestation may lag price signals)
- Minimum data points: 24 months required for calculation

Interpretation:
- Correlation > 0.5: Strong positive (price increase drives deforestation) -> HIGH market risk
- Correlation 0.2-0.5: Moderate positive -> STANDARD market risk
- Correlation < 0.2: Weak or no correlation -> LOW market risk
```

This is deterministic (same input data produces same correlation coefficient) and uses no ML/LLM.

### Appendix E: Substitution Detection Rule Specifications

**Rule 1 -- Volume Shift Detection:**
```
Trigger when:
  Supplier S, Commodity A volume in period T < Commodity A volume in period T-1 * (1 - threshold)
  AND Supplier S, Commodity B volume in period T > Commodity B volume in period T-1 * (1 + threshold)
  AND Commodities A and B are both EUDR-regulated
  AND Time window: 90 days (configurable)
  AND Threshold: 20% (configurable)

Confidence = min(|volume_change_A|, |volume_change_B|) / max(|volume_change_A|, |volume_change_B|)
             * 100 (closer volumes = higher confidence)
```

**Rule 2 -- Origin Shift Detection:**
```
Trigger when:
  Operator O, Commodity C, Country X share in period T < Country X share in period T-1 - threshold
  AND Commodity C, Country Y share in period T > Country Y share in period T-1 + threshold
  AND Country Y has lower risk classification than Country X
  AND Time window: 180 days (configurable)
  AND Threshold: 30% (configurable)

Confidence = based on magnitude of shift and country risk differential
```

**Rule 3 -- Price Arbitrage Detection:**
```
Trigger when:
  Supplier S switches from Commodity A to Commodity B
  AND Price(B) < Price(A) * (1 - price_threshold) at time of switch
  AND Switch coincides with price spike in Commodity A
  AND Price threshold: 15% (configurable)

Confidence = price_differential_pct / 100 * volume_shift_pct / 100 * 10000 (normalized)
```

**Rule 4 -- Classification Anomaly Detection:**
```
Trigger when:
  Product P from Supplier S changes CN code between shipments
  AND Product description remains substantially similar (Levenshtein distance < 0.3)
  AND New CN code falls outside EUDR Annex I scope

Confidence = (1 - description_similarity_score) * 100
```

**Rule 5 -- Transshipment Detection:**
```
Trigger when:
  Commodity C enters EU through Country E (entry point)
  AND Country E is classified LOW risk
  AND Production origin traces to Country P which is HIGH risk
  AND Country E is not a significant producer of Commodity C
  AND Transit time is unusually short for the route

Confidence = based on country risk differential and production implausibility
```

### Appendix F: Portfolio Risk Calculation Methodology

**Step 1: Portfolio Weight Calculation**
```
Portfolio_Weight_i = Value_EUR_i / sum(Value_EUR_j) for all commodities j
```

**Step 2: Portfolio Risk Score**
```
Portfolio_Risk = sum(Commodity_Risk_i * Portfolio_Weight_i) for all commodities i
```

**Step 3: Concentration (HHI)**
```
HHI = sum(Portfolio_Weight_i^2) for all commodities i

Interpretation:
- HHI < 0.15: Low concentration (diversified)
- 0.15 <= HHI < 0.25: Moderate concentration
- HHI >= 0.25: High concentration

Maximum HHI = 1.0 (single commodity portfolio)
Minimum HHI = 1/N (equally distributed across N commodities)
```

**Step 4: Diversification Score**
```
Diversification = (1 - HHI) / (1 - 1/N) * 100

Where N = number of commodities in portfolio
Scaled to 0-100 where:
- 100 = perfectly diversified across all commodities
- 0 = concentrated in single commodity
```

**Step 5: Risk Contribution**
```
Risk_Contribution_i = (Commodity_Risk_i * Portfolio_Weight_i) / Portfolio_Risk * 100

Sum of all Risk_Contribution_i = 100%
```

### Appendix G: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 -- EU Deforestation Regulation
2. EU Deforestation Regulation -- Annex I: List of Relevant Products
3. EU Combined Nomenclature (CN) -- Commission Implementing Regulation
4. EUDR Technical Specifications for the EU Information System
5. RSPO Supply Chain Certification Standard (2020)
6. FSC Chain of Custody Standard (FSC-STD-40-004 V3)
7. 4C Code of Conduct (Version 4.0)
8. RTRS Standard for Responsible Soy Production (Version 3.1)
9. World Bank Commodity Markets Outlook (latest edition)
10. FAO -- FAOSTAT Production Statistics Database
11. USDA Foreign Agricultural Service -- Production, Supply and Distribution Database
12. Global Forest Watch -- Technical Documentation and API Reference
13. ISO 22095:2020 -- Chain of Custody -- General Terminology and Models
14. Herfindahl-Hirschman Index -- U.S. Department of Justice Merger Guidelines

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-09 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-09 | GL-ProductManager | Initial draft created |
| 1.0.0 | 2026-03-09 | GL-ProductManager | Finalized: all 8 P0 features confirmed, 8 engines specified (CommodityProfiler/DerivedProductAnalyzer/PriceVolatilityEngine/ProductionForecastEngine/SubstitutionRiskAnalyzer/RegulatoryComplianceEngine/CommodityDueDiligenceEngine/PortfolioRiskAggregator), V106 migration schema defined (12 tables + 3 hypertables + 2 continuous aggregates), 30+ API endpoints, 19 RBAC permissions, 18 Prometheus metrics, 900+ test target, regulatory coverage verified (Articles 1/2/3/4/9/10/11/12/29/31 + Annex I), distinction from EUDR-016 commodity engine clarified, approval granted |
