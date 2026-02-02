# Carbon Markets Regulatory Update

**Document ID:** GL-REG-CM-001
**Version:** 1.0.0
**Date:** December 5, 2025
**Author:** GL-RegulatoryIntelligence
**Status:** ACTIVE

---

## Executive Summary

Carbon markets are undergoing significant transformation with the operationalization of Article 6 of the Paris Agreement, Phase 4 of the EU ETS, and major quality initiatives in voluntary markets. This document provides comprehensive analysis of compliance and voluntary carbon market developments relevant to GreenLang agent development.

### Market Overview

| Market Type | Size (2024) | Key Regulations | GreenLang Agent |
|-------------|-------------|-----------------|-----------------|
| EU ETS | EUR 160B+ | EU ETS Directive | gl-ets-reporting-v1 |
| Article 6.2 | Emerging | Paris Agreement | gl-article6-v1 |
| Article 6.4 | Emerging | Paris Agreement | gl-article6-v1 |
| Voluntary (VCM) | $2B | ICVCM, VCMI | gl-carbon-offset-v1 |

---

## 1. EU Emissions Trading System (EU ETS) Phase 4

### 1.1 System Summary

**Official Name:** EU Emissions Trading System

**Legal Basis:** Directive 2003/87/EC (as amended)

**Current Phase:** Phase 4 (2021-2030)

**Purpose:** Cap-and-trade system for GHG emissions reduction

### 1.2 Phase 4 Key Parameters

```yaml
eu_ets_phase_4:
  cap_trajectory:
    2021_cap: "1,572 million allowances"
    annual_reduction: "Linear Reduction Factor 2.2% -> 4.3% (from 2024)"
    2030_cap: "~950 million allowances"
    reduction_target: "62% below 2005 levels by 2030"

  scope:
    stationary_installations:
      - Power generation
      - Energy-intensive industry
      - Refineries
      - Steel and iron
      - Cement
      - Chemicals
      - Glass
      - Paper and pulp

    aviation:
      - Intra-EEA flights
      - EEA departing flights (from 2024)
      - Full scope from 2027

    maritime:
      - Intra-EU voyages (100%)
      - EU port calls (50%)
      - Phased: 40% (2024), 70% (2025), 100% (2026)

  free_allocation:
    industrial_installations:
      benchmark_based: true
      carbon_leakage_list: "Updated every 5 years"
      phase_out: "2026-2034 for CBAM sectors"

    aviation:
      free_allocation_end: 2026
      auctioning: "Full from 2027"
```

### 1.3 ETS2 (Buildings, Road Transport, Additional Sectors)

```yaml
ets_2:
  start_date: "2027"
  cap: "Separate cap from ETS1"
  scope:
    - Buildings (heating fuels)
    - Road transport (fuels)
    - Additional sectors (small industry)

  price_stability:
    maximum_price_2027: "EUR 45"
    automatic_release: "At EUR 45 price"

  social_climate_fund:
    size: "EUR 65B (2026-2032)"
    purpose: "Support vulnerable households and SMEs"

  reporting:
    regulated_entity: "Fuel suppliers"
    monitoring: "Fuel sales volumes"
```

### 1.4 Compliance Timeline

| Date | Requirement | Penalty |
|------|-------------|---------|
| Feb 28 | Submit verified emissions report | EUR 100/tCO2e + surrender |
| Apr 30 | Surrender allowances | EUR 100/tCO2e penalty |
| Sep 30 | Aviation allocation applications | N/A |

### 1.5 Data Requirements

```yaml
eu_ets_data_requirements:
  monitoring_plan:
    - Installation boundaries
    - Emission sources
    - Source streams
    - Calculation methodologies
    - Tier levels

  annual_emissions_report:
    combustion_emissions:
      - Fuel type
      - Quantity consumed
      - Net calorific value
      - Emission factor
      - Oxidation factor

    process_emissions:
      - Activity data
      - Emission factor
      - Conversion factor

    monitoring_methodology:
      calculation_based:
        tier_1: "Default emission factors"
        tier_2: "Country-specific factors"
        tier_3: "Installation-specific factors"
      measurement_based:
        cems: "Continuous emission monitoring"

  verification:
    standard: "ISO 14065"
    level: "Reasonable assurance"
    frequency: "Annual"
```

### 1.6 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-ets-reporting-v1 |
| Priority | P2-MEDIUM |
| Deadline | Dec 2026 (production-ready) |
| Golden Tests | 150 |

**Key Features:**
- Monitoring plan generator
- Emissions calculator (combustion + process)
- Verification package
- Allowance surrender calculator
- Free allocation tracker
- Maritime/aviation modules

---

## 2. Article 6 Paris Agreement Operationalization

### 2.1 Article 6 Overview

**Purpose:** International carbon market mechanisms under Paris Agreement

**Components:**
- Article 6.2: Bilateral/multilateral cooperative approaches
- Article 6.4: UN-supervised crediting mechanism
- Article 6.8: Non-market approaches

### 2.2 Article 6.2 (ITMOs)

```yaml
article_6_2:
  description: "Internationally Transferred Mitigation Outcomes"

  key_features:
    bilateral_agreements: "Country-to-country cooperation"
    flexibility: "Parties define methodologies"
    corresponding_adjustments: "Required for NDC use"

  process:
    1_authorization: "Host country authorizes transfer"
    2_transfer: "ITMOs transferred between registries"
    3_adjustment: "Host makes corresponding adjustment"
    4_use: "Acquiring country uses toward NDC"

  reporting:
    annual_information: "Required under Article 6 guidance"
    btrs: "Biennial Transparency Reports include ITMO use"

  active_agreements:
    switzerland:
      partners: ["Peru", "Ghana", "Senegal", "Georgia", "Thailand"]
      itmos_contracted: "~1.5M"
    singapore:
      partners: ["Ghana", "Papua New Guinea", "Peru"]
    japan:
      jcm_countries: 29
      projects: 200+

  challenges:
    - "Corresponding adjustment complexity"
    - "Registry interoperability"
    - "Double counting prevention"
    - "Environmental integrity concerns"
```

### 2.3 Article 6.4 (Successor to CDM)

```yaml
article_6_4:
  description: "UN-supervised crediting mechanism"

  governance:
    supervisory_body: "Article 6.4 Supervisory Body"
    registry: "UN-managed (in development)"
    methodologies: "SB-approved"

  key_decisions:
    cop28_2023:
      - "CDM transition rules finalized"
      - "First methodologies adopted"
      - "Removal activities guidance"

    cop29_2024:
      - "Additional methodologies"
      - "Registry operationalization"
      - "Credit issuance begins"

  methodology_categories:
    emission_reductions:
      - Renewable energy
      - Energy efficiency
      - Fuel switching
      - Transport
      - Waste management

    removals:
      - Afforestation/reforestation
      - Enhanced rock weathering
      - Direct air capture (under consideration)
      - Soil carbon

  credit_types:
    A6.4ERs: "Emission reductions"
    A6.4_removals: "Carbon removals"

  corresponding_adjustments:
    mandatory: "For NDC use"
    optional: "For other purposes (with host authorization)"

  cdm_transition:
    eligible_projects: "CDM projects registered before Jan 1, 2013"
    cers_carryover: "Limited to 2.5% of host NDC"
```

### 2.4 GreenLang Agent Specification (Article 6)

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-article6-v1 |
| Priority | P3-STANDARD |
| Deadline | Q4 2027 (production-ready) |
| Golden Tests | 100 |

**Key Features:**
- ITMO tracking
- Corresponding adjustment calculator
- NDC contribution assessment
- Registry integration (when available)
- Bilateral agreement tracker

---

## 3. Voluntary Carbon Market Quality Initiatives

### 3.1 Integrity Council for the Voluntary Carbon Market (ICVCM)

```yaml
icvcm:
  purpose: "Define and enforce high-integrity carbon credits"

  core_carbon_principles:
    governance:
      - Effective governance
      - Tracking and transparency
      - Robust independent validation

    emissions_impact:
      - Additionality
      - Permanence
      - Robust quantification
      - No double counting

    sustainable_development:
      - Sustainable development benefits
      - Safeguarding
      - Contribution to net-zero transition

  ccp_eligible_categories:
    assessed:
      - VCS methodologies
      - Gold Standard methodologies
      - ACR methodologies
      - CAR methodologies
    labeling: "CCP-Approved"

  assessment_framework:
    program_level: "Carbon crediting program assessment"
    category_level: "Methodology/category assessment"
    timeline:
      2023: "Program assessments begin"
      2024: "Category assessments"
      2025: "Full marketplace integration"

  current_status:
    programs_assessed:
      - Verra (conditional)
      - Gold Standard (conditional)
      - ACR (conditional)
    categories_assessed: "Ongoing"
```

### 3.2 Voluntary Carbon Markets Integrity Initiative (VCMI)

```yaml
vcmi:
  purpose: "Guide high-integrity corporate use of carbon credits"

  claims_code_of_practice:
    version: "2.0 (November 2023)"

    claim_tiers:
      platinum:
        criteria:
          - "Scope 1, 2, 3 near-term SBTi target"
          - "On track to meet target"
          - "High-quality credits >= 100% residual emissions"
      gold:
        criteria:
          - "Scope 1, 2, 3 near-term SBTi target"
          - "On track to meet target"
          - "High-quality credits >= 50% residual emissions"
      silver:
        criteria:
          - "Scope 1, 2 near-term SBTi target"
          - "On track to meet target"
          - "High-quality credits >= 20% residual emissions"

    beyond_value_chain:
      concept: "Credits for climate contribution beyond own value chain"
      claims:
        - "Supports climate action beyond value chain"
        - "Contributes to global net-zero"

    credit_quality:
      required: "ICVCM CCP-eligible or equivalent"
      prohibition: "No credits for carbon neutral claims without mitigation"

  corporate_adoption:
    early_adopters: "50+ companies"
    sectors: ["Consumer goods", "Tech", "Finance"]
```

### 3.3 Carbon Credit Quality Assessment

```yaml
credit_quality_criteria:
  additionality:
    definition: "Project would not occur without carbon finance"
    tests:
      - Investment test
      - Barrier analysis
      - Common practice test
    scoring: "Pass/Fail + Confidence level"

  permanence:
    definition: "GHG benefit will not be reversed"
    mechanisms:
      buffer_pools: "Insurance against reversal (typically 10-20%)"
      tonne_year_accounting: "Discounting for temporary storage"
      insurance: "Third-party reversal insurance"
    risk_assessment:
      low: "<5% reversal risk over 100 years"
      medium: "5-20% reversal risk"
      high: ">20% reversal risk"

  quantification:
    methodology: "Conservative, peer-reviewed"
    baseline: "Dynamic or conservative static"
    leakage: "Fully accounted"
    uncertainty: "Quantified and adjusted"

  double_counting:
    types:
      - Double issuance
      - Double claiming
      - Double use
    safeguards:
      - Unique serial numbers
      - Registry interlinking
      - Corresponding adjustments (NDC use)

  co_benefits:
    sdg_alignment: "Contribution to UN SDGs"
    verification: "Third-party verified"
    examples:
      - Biodiversity protection
      - Community livelihoods
      - Health improvements
```

### 3.4 GreenLang Agent Specification (Offsets)

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-carbon-offset-v1 |
| Priority | P4-STANDARD |
| Deadline | Feb 2027 (production-ready) |
| Golden Tests | 150 |

**Key Features:**
- Registry API connectors (VCS, Gold Standard, ACR, CAR)
- ICVCM CCP assessment
- VCMI claims code alignment
- Offset quality scoring
- Additionality assessment
- Retirement tracking
- Claims substantiation for Green Claims Directive

---

## 4. Regional Carbon Markets

### 4.1 Global Carbon Market Overview

| Market | Status | Coverage | 2024 Price Range |
|--------|--------|----------|------------------|
| EU ETS | Operational | 40% EU emissions | EUR 55-90 |
| UK ETS | Operational | UK installations | GBP 35-70 |
| California Cap-and-Trade | Operational | CA economy-wide | USD 30-40 |
| China ETS | Operational | Power sector | CNY 60-100 |
| Korea ETS | Operational | Heavy industry | KRW 8,000-15,000 |
| New Zealand ETS | Operational | Economy-wide | NZD 50-80 |

### 4.2 UK Emissions Trading Scheme

```yaml
uk_ets:
  description: "Post-Brexit UK carbon market"
  start_date: "January 2021"

  scope:
    - Power generation
    - Energy-intensive industry
    - Aviation (UK domestic + to EEA)

  cap:
    2021: "155.7M allowances"
    2024: "140M allowances (tightened)"
    reduction: "5.1% annual from 2024"

  linkage:
    eu_ets: "No current linkage"
    potential: "Under discussion"

  price:
    carbon_price_floor: "Dynamic AUCPS"
    2024_range: "GBP 35-70"

  reporting:
    deadline: "March 31 (verified emissions)"
    surrender: "April 30"
```

### 4.3 California Cap-and-Trade

```yaml
california_cap_trade:
  description: "Economy-wide carbon market"
  administrator: "California Air Resources Board (CARB)"

  scope:
    covered_entities:
      - Large industrial facilities (>25,000 tCO2e/year)
      - Electricity importers
      - Fuel suppliers
    coverage: "~80% California GHG emissions"

  cap:
    2024: "~320M allowances"
    reduction: "~4% annual"
    2030_target: "40% below 1990 by 2030"

  linkage:
    quebec: "Active linkage since 2014"
    washington: "Linked 2024"

  offsets:
    usage_limit: "4% of compliance obligation"
    protocol_types:
      - US Forest Projects
      - Urban Forest Projects
      - Livestock Digesters
      - Ozone Depleting Substances
      - Mine Methane Capture
      - Rice Cultivation

  price:
    auction_reserve: "USD 23.30 (2024)"
    2024_range: "USD 28-40"
```

### 4.4 China National ETS

```yaml
china_ets:
  description: "World's largest carbon market by coverage"
  administrator: "Ministry of Ecology and Environment"
  start_date: "July 2021"

  scope:
    current: "Power sector (~2,000 entities)"
    planned_expansion:
      - Cement (2024-2025)
      - Aluminum
      - Steel
      - Petrochemicals
      - Chemicals
      - Paper
      - Aviation

  cap_mechanism:
    type: "Intensity-based (tCO2/MWh)"
    benchmarks:
      - Coal-fired power
      - Gas-fired power
      - Combined heat and power

  ccer_offsets:
    status: "Restarted 2024"
    limit: "5% of compliance obligation"
    methodologies: "CCER methodologies"

  price:
    2024_range: "CNY 60-100 (~USD 8-14)"
    trend: "Increasing"
```

---

## 5. Carbon Market Data Integration

### 5.1 Registry Connectivity

```yaml
carbon_registries:
  compliance:
    eu_ets:
      registry: "EU Transaction Log + National Registries"
      api: "Limited (national level)"
      data: "Installation emissions, surrenders"

    uk_ets:
      registry: "UK Registry"
      api: "Limited"
      data: "Account holdings, transfers"

    california:
      registry: "CITSS"
      api: "Limited public"
      data: "Compliance reports"

  voluntary:
    verra_vcs:
      registry: "Verra Registry"
      api: "Public API available"
      data: "Project details, issuances, retirements"
      endpoint: "https://registry.verra.org/api"

    gold_standard:
      registry: "Gold Standard Registry"
      api: "Public API"
      data: "Project details, credits"
      endpoint: "https://registry.goldstandard.org/api"

    acr:
      registry: "American Carbon Registry"
      api: "Available"
      data: "Project and credit data"

    car:
      registry: "Climate Action Reserve"
      api: "Available"
      data: "Projects, credits"
```

### 5.2 Price Data Sources

| Source | Markets Covered | Update Frequency |
|--------|-----------------|------------------|
| ICE | EU ETS, UK ETS | Real-time |
| CME | California, RGGI | Real-time |
| Carbon Pulse | All major markets | Daily |
| Refinitiv | Global | Real-time |
| Bloomberg NEF | Global analysis | Daily |

---

## 6. Compliance Integration

### 6.1 Carbon Market - ESG Reporting Links

| Reporting Framework | Carbon Market Data Required |
|--------------------|-----------------------------|
| CSRD ESRS E1 | ETS emissions, allowances, carbon credits |
| SB 253 | Carbon offset usage disclosure |
| TCFD | Carbon price scenarios |
| CDP | Carbon market participation |
| Green Claims Directive | Offset quality for claims |

### 6.2 GreenLang Carbon Market Suite

| Agent | Market Coverage | Priority | Status |
|-------|-----------------|----------|--------|
| gl-ets-reporting-v1 | EU ETS, UK ETS | P2 | Planned |
| gl-carbon-offset-v1 | VCM (VCS, GS, ACR, CAR) | P4 | Planned |
| gl-article6-v1 | Paris Article 6 | P3 | Planned |
| gl-cbam-importer-v1 | EU CBAM | P0 | In development |

### 6.3 Integration Architecture

```yaml
carbon_market_integration:
  data_flows:
    emissions_to_compliance:
      - Calculate facility emissions
      - Compare to free allocation
      - Determine allowance purchase/surrender

    offsets_to_claims:
      - Assess offset quality (ICVCM)
      - Verify retirement
      - Substantiate climate claims

    article6_to_ndc:
      - Track ITMO transfers
      - Calculate corresponding adjustments
      - Report in BTRs

  shared_components:
    - GHG emissions calculator
    - Registry API connectors
    - Price data feeds
    - Verification workflow
```

---

## 7. Market Outlook and Trends

### 7.1 Price Trajectories

| Market | 2024 | 2025E | 2030E |
|--------|------|-------|-------|
| EU ETS | EUR 55-90 | EUR 70-100 | EUR 100-150 |
| UK ETS | GBP 35-70 | GBP 50-80 | GBP 80-120 |
| California | USD 28-40 | USD 35-50 | USD 50-70 |
| VCM (high quality) | USD 5-30 | USD 10-40 | USD 20-60 |

### 7.2 Key Developments to Monitor

| Development | Timeline | Impact |
|-------------|----------|--------|
| ETS2 launch | 2027 | New compliance market |
| CBAM full implementation | 2027 | Import carbon pricing |
| Article 6.4 registry | 2025 | New credit supply |
| ICVCM full rollout | 2025 | VCM quality standards |
| China ETS expansion | 2025+ | Largest market grows |

---

## 8. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-05 | GL-RegulatoryIntelligence | Initial carbon markets update |

---

**END OF DOCUMENT**
