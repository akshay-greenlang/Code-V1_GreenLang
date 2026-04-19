# Sector-Specific Regulatory Requirements

**Document ID:** GL-REG-SEC-001
**Version:** 1.0.0
**Date:** December 5, 2025
**Author:** GL-RegulatoryIntelligence
**Status:** ACTIVE

---

## Executive Summary

Certain sectors face unique and stringent sustainability disclosure requirements beyond general corporate reporting frameworks. This document covers regulatory requirements for Aviation (CORSIA, EU ETS), Maritime (IMO CII, EU MRV), Real Estate (CRREM, GRESB), and Financial Services (SFDR Article 8/9).

### Sector Overview

| Sector | Key Regulations | Compliance Urgency | GreenLang Agent |
|--------|-----------------|-------------------|-----------------|
| Aviation | CORSIA, EU ETS Aviation | HIGH | gl-corsia-aviation-v1 |
| Maritime | IMO CII, EU MRV, FuelEU | HIGH | gl-maritime-mrv-v1 |
| Real Estate | CRREM, GRESB, ENERGY STAR | MEDIUM | gl-real-estate-v1 |
| Finance | SFDR, EU Taxonomy, TCFD | HIGH | gl-sfdr-finance-v1 |

---

## 1. Aviation Sector

### 1.1 CORSIA (Carbon Offsetting and Reduction Scheme for International Aviation)

#### Regulation Summary

**Official Name:** Carbon Offsetting and Reduction Scheme for International Aviation

**Authority:** International Civil Aviation Organization (ICAO)

**Purpose:** Stabilize net CO2 emissions from international aviation at 2019 levels

#### Timeline and Phases

```yaml
corsia_timeline:
  pilot_phase:
    period: "2021-2023"
    participation: "Voluntary"
    countries: 107

  first_phase:
    period: "2024-2026"
    participation: "Voluntary with expanded coverage"
    baseline: "2019 emissions"

  second_phase:
    period: "2027-2035"
    participation: "Mandatory for most states"
    exceptions:
      - Least developed countries
      - Small island developing states
      - Landlocked developing countries
      - Low aviation activity states (<0.5% RTK)
```

#### Offsetting Requirements

```yaml
corsia_offsetting:
  calculation:
    formula: "Operator emissions * Growth factor * Sectoral/Individual factor"

  growth_factor:
    2024_2026: "100% sectoral (industry average)"
    2027_2029: "Transitioning to individual"
    2030_onwards: "Individual operator responsibility increases"

  eligible_offsets:
    programs:
      - Verra VCS
      - Gold Standard
      - ACR
      - CAR
      - Architecture for REDD+ Transactions (ART)
    requirements:
      - CORSIA-eligible emission units
      - Vintage post-2016
      - No double counting

  exemptions:
    - Flights to/from non-participating states
    - Humanitarian flights
    - Medical flights
    - Military flights
```

#### Monitoring, Reporting, Verification (MRV)

```yaml
corsia_mrv:
  monitoring:
    methods:
      method_a: "Fuel consumption calculation"
      method_b: "Block-off/block-on fuel measurement"
    data_required:
      - Fuel uplift
      - Fuel density
      - Flight pairs
      - Aircraft type

  reporting:
    deadline: "March 31 (for previous calendar year)"
    content:
      - Total CO2 emissions
      - Flight routes (state pairs)
      - Fuel consumption

  verification:
    standard: "ICAO CORSIA verification standard"
    accreditation: "National accreditation body"
    frequency: "Annual"
```

#### Data Requirements

```yaml
corsia_data_requirements:
  flight_data:
    - Flight identifier
    - Departure/arrival airports
    - Aircraft type/registration
    - ICAO aircraft type designator

  fuel_data:
    - Fuel type (Jet A, Jet A-1, sustainable aviation fuel)
    - Quantity consumed (kg)
    - Fuel density
    - Emission factor

  sustainable_aviation_fuel:
    tracking:
      - SAF production pathway
      - Lifecycle emissions
      - Chain of custody
    benefit:
      - Reduced offset requirement
      - CORSIA SAF eligibility criteria
```

#### GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-corsia-aviation-v1 |
| Priority | P2-MEDIUM |
| Deadline | Dec 2026 (production-ready) |
| Golden Tests | 100 |

**Key Features:**
- Flight emissions calculator
- CORSIA offset requirement calculator
- SAF tracking and credit
- Verification package generator
- ICAO reporting format

---

### 1.2 EU ETS Aviation

#### Regulation Summary

**Official Name:** EU ETS for Aviation

**Legal Basis:** Directive 2003/87/EC (as amended)

**Scope:** Flights within EEA (expanded scope from 2024)

#### Scope Evolution

```yaml
eu_ets_aviation_scope:
  pre_2024:
    coverage: "Intra-EEA flights only"

  2024_onwards:
    intra_eea: "100% covered"
    extra_eea_departing:
      phase_in: "Flights departing EEA to third countries"
      condition: "If equivalent CORSIA measures not in place"

  2027_onwards:
    full_auctioning: "Phase out free allocation"
    saf_requirement: "Part of ETS compliance pathway"
```

#### Free Allocation Phase-Out

| Year | Free Allocation | Auctioning |
|------|-----------------|------------|
| 2024 | 75% | 25% |
| 2025 | 50% | 50% |
| 2026 | 25% | 75% |
| 2027+ | 0% | 100% |

#### GreenLang Agent Integration

**Primary Agent:** gl-ets-reporting-v1

**Aviation Module Features:**
- Intra-EEA flight tracking
- Benchmark allocation calculator
- Allowance surrender calculator
- SAF incentive tracking

---

## 2. Maritime Sector

### 2.1 IMO Carbon Intensity Indicator (CII)

#### Regulation Summary

**Official Name:** IMO Carbon Intensity Indicator

**Authority:** International Maritime Organization (MARPOL Annex VI)

**Purpose:** Measure and rate ships' operational carbon intensity

**Effective Date:** January 1, 2023

#### CII Calculation

```yaml
cii_calculation:
  formula: "AER = (CO2 emissions) / (DWT x Distance)"

  components:
    co2_emissions:
      source: "Fuel consumption x Emission factor"
      units: "gCO2"

    dwt:
      meaning: "Deadweight tonnage"
      units: "tonnes"

    distance:
      source: "AIS data or voyage records"
      units: "nautical miles"

  rating_scale:
    A: "Major superior (well below reference)"
    B: "Minor superior"
    C: "Moderate (reference level)"
    D: "Minor inferior"
    E: "Inferior (well above reference)"

  rating_thresholds:
    year_specific: true
    annual_reduction: "~2% per year"
    2030_target: "40% reduction from 2008"

  consequences:
    A_B_C: "Compliant"
    D_E:
      action: "Corrective action plan required"
      consecutive_years: "3 years D or E = enhanced PSC inspection"
```

#### Ship Types Covered

```yaml
cii_ship_types:
  covered:
    - Bulk carriers (>5,000 GT)
    - Tankers (>5,000 GT)
    - Container ships (>5,000 GT)
    - Gas carriers (>5,000 GT)
    - LNG carriers (>5,000 GT)
    - General cargo ships (>5,000 GT)
    - Refrigerated cargo carriers
    - Combination carriers
    - Cruise ships (>5,000 GT)
    - Ro-Ro cargo ships (>5,000 GT)
    - Ro-Ro passenger ships (>5,000 GT)

  threshold: "400 gross tonnage and above"
```

#### Data Requirements

```yaml
cii_data_requirements:
  ship_particulars:
    - IMO number
    - Ship type
    - Gross tonnage
    - Deadweight tonnage
    - Flag state

  annual_data:
    - Fuel consumption by type
    - Distance traveled
    - Hours underway
    - Transport work (cargo)

  reporting:
    collector: "Fuel oil data collection system"
    verifier: "Flag state or recognized organization"
    database: "IMO DCS database"
```

---

### 2.2 EU MRV Maritime

#### Regulation Summary

**Official Name:** EU Regulation on Maritime Emissions Monitoring, Reporting and Verification

**Legal Citation:** Regulation (EU) 2015/757 (as amended)

**Purpose:** Monitor, report, and verify CO2 emissions from maritime transport

#### Scope

```yaml
eu_mrv_scope:
  ships: ">5,000 gross tonnage"

  voyages:
    - Voyages between EU/EEA ports (100%)
    - Voyages from EU port to non-EU port (50%)
    - Voyages from non-EU port to EU port (50%)

  gases:
    current: [CO2]
    from_2024: [CO2, CH4, N2O]
```

#### EU ETS Maritime Integration

```yaml
eu_ets_maritime:
  phase_in:
    2024: "40% of verified emissions"
    2025: "70% of verified emissions"
    2026: "100% of verified emissions"

  scope:
    intra_eu: "100%"
    extra_eu: "50% (port calls)"

  compliance:
    monitoring: "EU MRV"
    allowance_surrender: "April 30 following year"
```

#### GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-maritime-mrv-v1 |
| Priority | P2-MEDIUM |
| Deadline | Oct 2028 (production-ready) |
| Golden Tests | 120 |

**Key Features:**
- CII calculator and rating
- EU MRV report generator
- EU ETS allowance calculator
- FuelEU maritime compliance
- Voyage emissions tracker
- Ship efficiency rating

---

### 2.3 FuelEU Maritime

#### Regulation Summary

**Official Name:** FuelEU Maritime Regulation

**Legal Citation:** Regulation (EU) 2023/1805

**Purpose:** Increase demand for renewable and low-carbon fuels in maritime

#### GHG Intensity Limits

```yaml
fueleu_intensity:
  baseline: "91.16 gCO2eq/MJ (reference value)"

  reduction_targets:
    2025: "-2%"
    2030: "-6%"
    2035: "-14.5%"
    2040: "-31%"
    2045: "-62%"
    2050: "-80%"

  calculation:
    formula: "Well-to-Wake GHG intensity of energy used"
    scope: "All energy used on board (at berth and at sea)"
```

#### Compliance Mechanisms

```yaml
fueleu_compliance:
  primary: "Meet GHG intensity limit"

  flexibility:
    banking: "Overcompliance banked for next 3 years"
    borrowing: "Limited borrowing against future compliance"
    pooling: "Ships can pool compliance"

  penalties:
    formula: "Deviation x Energy x EUR 2,400/tCO2eq"
    severity: "Substantial for non-compliance"
```

---

## 3. Real Estate Sector

### 3.1 CRREM (Carbon Risk Real Estate Monitor)

#### Framework Summary

**Official Name:** Carbon Risk Real Estate Monitor

**Type:** Science-based decarbonization pathway tool

**Purpose:** Assess stranding risk of real estate assets under climate scenarios

#### CRREM Methodology

```yaml
crrem_methodology:
  pathways:
    description: "1.5C and 2C aligned trajectories by asset type"
    granularity:
      - Country-specific
      - Asset type specific
      - Climate scenario specific

  asset_types:
    - Office
    - Retail (high street, shopping center, warehouse)
    - Hotel
    - Logistics/industrial
    - Residential (single-family, multi-family)
    - Healthcare
    - Data centers

  metrics:
    energy_intensity:
      unit: "kWh/m2/year"
      scope: "Whole building energy"

    carbon_intensity:
      unit: "kgCO2e/m2/year"
      scope: "Operational emissions (Scope 1 + 2)"

  stranding_analysis:
    definition: "Year asset exceeds pathway"
    implications:
      - Value impairment risk
      - Retrofit investment need
      - Portfolio rebalancing
```

#### Data Requirements

```yaml
crrem_data:
  asset_information:
    - Property name/ID
    - Country
    - Asset type
    - Gross floor area (m2)
    - Year built
    - Energy rating (EPC)

  energy_data:
    - Total energy consumption (kWh/year)
    - By fuel type (electricity, gas, district heating)
    - Sub-metered data where available
    - Renewable energy on-site/purchased

  occupancy_data:
    - Occupancy rate
    - Operating hours
    - Tenant energy (if separate)
```

---

### 3.2 GRESB (Global Real Estate Sustainability Benchmark)

#### Framework Summary

**Official Name:** Global Real Estate Sustainability Benchmark

**Type:** ESG benchmark for real estate and infrastructure

**Purpose:** Standardized sustainability assessment and benchmarking

#### Assessment Structure

```yaml
gresb_assessment:
  components:
    management:
      weight: "30%"
      aspects:
        - Leadership
        - Policies
        - Reporting
        - Risk management
        - Stakeholder engagement

    performance:
      weight: "70%"
      aspects:
        - Energy
        - GHG emissions
        - Water
        - Waste
        - Data coverage
        - Building certifications

  scoring:
    scale: "0-100"
    rating: "1-5 stars"
    peer_groups: "By sector and region"

  indicators:
    energy:
      - Like-for-like energy consumption
      - Energy intensity (kWh/m2)
      - Renewable energy percentage

    ghg:
      - Scope 1 emissions
      - Scope 2 emissions (location & market)
      - Scope 3 emissions (select categories)
      - Carbon intensity (kgCO2e/m2)

    water:
      - Water consumption
      - Water intensity (m3/m2)

    waste:
      - Waste generated
      - Diversion rate
```

#### Timeline

| Date | Activity |
|------|----------|
| April 1 | Assessment opens |
| July 1 | Submission deadline |
| October | Results released |

---

### 3.3 Energy Efficiency Regulations

#### EU Energy Performance of Buildings Directive (EPBD)

```yaml
epbd_recast:
  status: "Adopted 2024"

  key_requirements:
    new_buildings:
      2028: "Zero-emission (public buildings)"
      2030: "Zero-emission (all new buildings)"

    existing_buildings:
      residential:
        2030: "Average EPC E"
        2033: "Average EPC D"
      non_residential:
        2027: "Worst 16% upgraded"
        2030: "Worst 26% upgraded"

    solar:
      new_buildings: "Solar-ready or installed"
      major_renovations: "Solar consideration"

  minimum_energy_performance_standards:
    type: "National MEPS required"
    target: "Trajectory to zero-emission by 2050"
```

#### US ENERGY STAR

```yaml
energy_star_buildings:
  purpose: "Benchmark building energy performance"

  score:
    scale: "1-100"
    threshold: "75+ for certification"
    basis: "Percentile ranking vs peers"

  portfolio_manager:
    tool: "EPA Portfolio Manager"
    data:
      - Property characteristics
      - Monthly energy use
      - Weather normalization
      - Occupancy

  benchmarking_mandates:
    cities:
      - New York (LL84, LL97)
      - Los Angeles
      - San Francisco
      - Boston
      - Washington DC
      - Seattle
      - Chicago
```

#### GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-real-estate-v1 |
| Priority | P3-STANDARD |
| Deadline | Q1 2027 (production-ready) |
| Golden Tests | 150 |

**Key Features:**
- CRREM pathway analysis
- GRESB assessment preparation
- Building energy calculator
- EPC/Energy rating tracker
- Portfolio carbon footprint
- Stranding risk assessment

---

## 4. Financial Services Sector

### 4.1 SFDR (Sustainable Finance Disclosure Regulation)

#### Regulation Summary

**Official Name:** Sustainable Finance Disclosure Regulation

**Legal Citation:** Regulation (EU) 2019/2088

**Purpose:** Transparency on sustainability in financial markets

#### Product Classifications

```yaml
sfdr_classifications:
  article_6:
    description: "Non-ESG products"
    requirements:
      - Sustainability risk integration disclosure
      - PAI consideration statement

  article_8:
    description: "Products promoting E/S characteristics"
    requirements:
      - Environmental/social characteristics disclosure
      - Pre-contractual information
      - Periodic reporting
      - Website disclosure
    binding_elements:
      - Minimum sustainable investments
      - ESG screening criteria
      - Benchmark alignment (if applicable)

  article_9:
    description: "Products with sustainable investment objective"
    requirements:
      - Sustainable investment objective disclosure
      - DNSH assessment
      - Pre-contractual information
      - Periodic reporting
    investment_threshold: "100% sustainable investments (excl. cash/hedging)"
```

#### Principal Adverse Impact (PAI) Indicators

```yaml
pai_indicators:
  mandatory:
    climate:
      1: "GHG emissions (Scope 1, 2, 3)"
      2: "Carbon footprint"
      3: "GHG intensity of investee companies"
      4: "Exposure to fossil fuel sector"
      5: "Non-renewable energy share"
      6: "Energy consumption intensity"

    biodiversity:
      7: "Biodiversity-sensitive areas"

    water:
      8: "Emissions to water"

    waste:
      9: "Hazardous waste ratio"

    social:
      10: "UNGC/OECD violations"
      11: "UNGC/OECD compliance processes"
      12: "Gender pay gap"
      13: "Board gender diversity"
      14: "Controversial weapons exposure"

    sovereigns:
      15: "GHG intensity of sovereigns"
      16: "Investee countries with social violations"

  optional:
    additional_climate: "5 indicators"
    additional_environmental: "6 indicators"
    additional_social: "17 indicators"
```

#### Regulatory Technical Standards (RTS)

```yaml
sfdr_rts:
  pre_contractual:
    article_8:
      - Environmental/social characteristics
      - Investment strategy
      - Proportion of investments
      - Monitoring methodology
      - Benchmark designation (if any)

    article_9:
      - Sustainable investment objective
      - No significant harm approach
      - Investment strategy
      - Proportion (100% sustainable)
      - Benchmark (if Paris-aligned)

  periodic_reporting:
    frequency: "Annual"
    content:
      - Achievement of E/S characteristics or objective
      - Top investments
      - Sector/geography breakdown
      - PAI indicators
      - Taxonomy alignment
```

---

### 4.2 EU Taxonomy for Financial Products

#### Integration with SFDR

```yaml
taxonomy_financial:
  article_8_products:
    disclosure: "Taxonomy alignment percentage"
    minimum: "No minimum (but must disclose if 0%)"

  article_9_products:
    disclosure: "Taxonomy alignment percentage"
    minimum: "If objective is taxonomy objective"

  kpis:
    turnover_aligned: "Weighted average of investee taxonomy turnover"
    capex_aligned: "Weighted average of investee taxonomy CapEx"

  data_challenges:
    - "Investee company Taxonomy disclosures availability"
    - "Non-EU company data gaps"
    - "Estimation methodologies"
```

#### GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-sfdr-finance-v1 |
| Priority | P2-MEDIUM |
| Deadline | Q2 2026 (production-ready) |
| Golden Tests | 200 |

**Key Features:**
- Article 8/9 classification tool
- PAI indicator calculator
- Pre-contractual template generator
- Periodic report generator
- Taxonomy alignment calculator
- Portfolio emissions calculator

---

### 4.3 Additional Financial Regulations

#### UK SDR (Sustainability Disclosure Requirements)

```yaml
uk_sdr:
  status: "Final rules published November 2023"
  effective: "July 2024 (labeling)"

  labels:
    sustainability_focus:
      - "Assets with improved sustainability"
      - "70% minimum threshold"

    sustainability_improvers:
      - "Assets on improvement pathway"
      - "Measurable improvement"

    sustainability_impact:
      - "Achieve positive sustainability outcomes"
      - "Impact measurement"

    sustainability_mixed_goals:
      - "Combination of above"

  anti_greenwashing:
    rule: "Claims must be fair, clear, not misleading"
    evidence: "Robust evidence required"
```

#### SEC ESG Fund Rules (US)

```yaml
sec_esg_funds:
  status: "Adopted September 2023"

  requirements:
    esg_integrated:
      - "Disclosure of ESG factors considered"
      - "How factors affect selection"

    esg_focused:
      - "Prospectus disclosure"
      - "Annual report ESG section"

    esg_impact:
      - "Impact measurement disclosure"
      - "Progress reporting"
```

---

## 5. Sector Prioritization Matrix

### 5.1 Agent Development Priorities

| Sector | Agent | Regulation | Priority | Deadline | Effort |
|--------|-------|------------|----------|----------|--------|
| Finance | gl-sfdr-finance-v1 | SFDR | P2 | Q2 2026 | 16 weeks |
| Aviation | gl-corsia-aviation-v1 | CORSIA | P2 | Dec 2026 | 12 weeks |
| Maritime | gl-maritime-mrv-v1 | IMO CII, EU MRV | P2 | Oct 2028 | 14 weeks |
| Real Estate | gl-real-estate-v1 | CRREM, GRESB | P3 | Q1 2027 | 12 weeks |

### 5.2 Cross-Sector Dependencies

```yaml
sector_dependencies:
  finance_to_corporate:
    - CSRD data for PAI calculations
    - Taxonomy disclosures for alignment
    - SBTi targets for Article 9

  real_estate_to_csrd:
    - ESRS E1 energy data
    - ESRS E2 building emissions
    - Portfolio carbon footprint

  aviation_maritime_to_ets:
    - EU ETS compliance
    - Allowance management
    - Free allocation tracking
```

### 5.3 Data Integration Architecture

```yaml
sector_data_integration:
  shared_data_layer:
    - GHG emissions (Scope 1, 2, 3)
    - Energy consumption
    - EU Taxonomy alignment
    - Climate targets

  sector_specific:
    aviation:
      - Flight data
      - SAF consumption
      - CORSIA offsets

    maritime:
      - Voyage data
      - CII ratings
      - FuelEU compliance

    real_estate:
      - Building data
      - EPC ratings
      - CRREM pathways

    finance:
      - Portfolio holdings
      - PAI indicators
      - SFDR classifications
```

---

## 6. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-05 | GL-RegulatoryIntelligence | Initial sector requirements |

---

**END OF DOCUMENT**
