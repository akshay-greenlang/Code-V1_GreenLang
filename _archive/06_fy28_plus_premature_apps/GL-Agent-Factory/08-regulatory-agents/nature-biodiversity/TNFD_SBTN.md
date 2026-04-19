# Nature and Biodiversity Disclosure Frameworks

**Document ID:** GL-REG-NAT-001
**Version:** 1.0.0
**Date:** December 5, 2025
**Author:** GL-RegulatoryIntelligence
**Status:** ACTIVE

---

## Executive Summary

Nature-related disclosure is rapidly emerging as the next frontier in corporate sustainability reporting. This document covers the key frameworks and regulations governing nature, biodiversity, and land-use disclosures, including TNFD, SBTN, EU Biodiversity Strategy, and deforestation regulations beyond EUDR.

### Key Frameworks Overview

| Framework/Regulation | Type | Status | Key Deadline | GreenLang Agent |
|---------------------|------|--------|--------------|-----------------|
| TNFD | Voluntary Framework | Final (Sep 2023) | Dec 2027 (G20) | gl-tnfd-reporting-v1 |
| SBTN | Voluntary Standard | Validation open | Ongoing | gl-sbtn-targets-v1 |
| EU Biodiversity Strategy | Policy Framework | Active | 2030 | Multiple agents |
| EUDR | EU Regulation | In force | Dec 30, 2025 | gl-eudr-compliance-v1 |
| UK Environment Act | UK Regulation | In force | 2025+ | gl-uk-deforestation-v1 |

---

## 1. Taskforce on Nature-related Financial Disclosures (TNFD)

### 1.1 Framework Summary

**Official Name:** Taskforce on Nature-related Financial Disclosures

**Version:** Final Recommendations v1.0

**Publication Date:** September 18, 2023

**Purpose:** Enable organizations to report and act on evolving nature-related dependencies, impacts, risks, and opportunities

### 1.2 TNFD Structure

```yaml
tnfd_structure:
  pillars:
    governance:
      description: "Board and management oversight of nature-related issues"
      disclosures:
        A: "Board oversight of nature-related dependencies, impacts, risks, opportunities"
        B: "Management role in assessing and managing nature-related issues"

    strategy:
      description: "Integration of nature into business strategy"
      disclosures:
        A: "Nature-related dependencies, impacts, risks, opportunities identified"
        B: "Effect on business model, strategy, financial planning"
        C: "Resilience of strategy under different scenarios"
        D: "Nature-related targets and goals"

    risk_impact_management:
      description: "Processes for managing nature-related issues"
      disclosures:
        A: "Processes for identifying dependencies, impacts, risks, opportunities"
        B: "Processes for managing nature-related issues"
        C: "Integration with overall risk management"

    metrics_targets:
      description: "Metrics and targets for nature-related issues"
      disclosures:
        A: "Metrics on dependencies, impacts, risks, opportunities"
        B: "Metrics on scope 3 and value chain"
        C: "Targets and progress"
```

### 1.3 LEAP Approach

```yaml
leap_approach:
  description: "TNFD integrated assessment process"

  L_locate:
    purpose: "Locate interface with nature"
    activities:
      - "Identify business footprint (assets, operations, sourcing)"
      - "Map biomes and ecosystems"
      - "Identify sensitive locations"
      - "Assess completeness"
    data_sources:
      - IBAT (Integrated Biodiversity Assessment Tool)
      - WWF Risk Filter
      - Global Forest Watch
      - ENCORE database

  E_evaluate:
    purpose: "Evaluate dependencies and impacts"
    activities:
      - "Identify environmental assets and ecosystem services"
      - "Identify dependencies on nature"
      - "Identify impacts on nature"
      - "Analyze dependencies and impacts"
    frameworks:
      - ENCORE (Exploring Natural Capital Opportunities, Risks and Exposure)
      - Natural Capital Protocol

  A_assess:
    purpose: "Assess risks and opportunities"
    activities:
      - "Identify nature-related risks"
      - "Identify nature-related opportunities"
      - "Assess risk profile"
      - "Determine material risks/opportunities"
    risk_categories:
      physical:
        - Acute (ecosystem collapse, pollution events)
        - Chronic (soil degradation, water scarcity)
      transition:
        - Policy and legal
        - Market
        - Technology
        - Reputation
      systemic:
        - Ecosystem stability
        - Financial contagion

  P_prepare:
    purpose: "Prepare for reporting and response"
    activities:
      - "Develop strategy response"
      - "Set targets"
      - "Prepare disclosures"
      - "Present to stakeholders"
```

### 1.4 Core Metrics

```yaml
tnfd_core_metrics:
  global_core:
    C1:
      metric: "Total spatial footprint"
      unit: "km2"
      required: true
    C2:
      metric: "Total spatial footprint in high-importance biodiversity areas"
      unit: "km2"
      required: true
    C3:
      metric: "Extent of land/freshwater/ocean use change"
      unit: "km2/year"
      required: true

  sector_metrics:
    by_industry:
      - Agriculture and food
      - Energy
      - Mining
      - Financial services
      - Retail
      - Real estate

  additional_disclosure:
    dependency_metrics:
      - Ecosystem services relied upon
      - Critical dependencies by location
    impact_metrics:
      - Land use change
      - Freshwater use
      - Pollution
      - Resource extraction
      - Invasive species
```

### 1.5 Timeline and Adoption

| Milestone | Date | Description |
|-----------|------|-------------|
| Final framework | Sep 2023 | TNFD v1.0 released |
| Early adopters | 2024 | 320+ organizations |
| G20 recommendation | 2024 | G20 supports adoption |
| Target adoption | Dec 2027 | G20 companies expected to report |
| ISSB alignment | 2025+ | IFRS S2 harmonization |

### 1.6 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-tnfd-reporting-v1 |
| Priority | P2-MEDIUM |
| Deadline | Oct 2027 (production-ready) |
| Golden Tests | 150 |

**Key Features:**
- LEAP assessment workflow
- Location footprint mapping
- Biodiversity risk assessment
- Dependency/impact matrix generator
- TNFD-aligned report generator
- IBAT/ENCORE integration

---

## 2. Science-Based Targets for Nature (SBTN)

### 2.1 Framework Summary

**Official Name:** Science Based Targets Network - Targets for Nature

**Version:** Initial Guidance v1.0

**Publication Date:** May 2023 (Step 1-3), 2024 (Step 4-5)

**Purpose:** Enable companies to set science-based targets for nature aligned with planetary boundaries

### 2.2 5-Step Process

```yaml
sbtn_steps:
  step_1_assess:
    purpose: "Assess material impacts and dependencies"
    activities:
      - Materiality screening
      - Sector-specific assessment
      - Value chain mapping
      - Pressure identification
    output: "Priority list of issues, geographies, and value chain stages"

  step_2_interpret_prioritize:
    purpose: "Interpret and prioritize"
    activities:
      - Interpret materiality results
      - Prioritize locations
      - Prioritize value chain stages
      - Determine target scope
    output: "Prioritized issues and locations for target-setting"

  step_3_measure_set_disclose:
    purpose: "Measure baseline, set targets, disclose"
    activities:
      - Establish baseline year
      - Set quantified targets
      - Choose target type
      - Submit for validation
    target_types:
      - No conversion of natural ecosystems
      - No degradation of high integrity areas
      - Reduction targets (freshwater, land use)
      - Engagement targets (supply chain)
    output: "Validated science-based targets"

  step_4_act:
    purpose: "Take action to meet targets"
    activities:
      - Avoid (AR1)
      - Reduce (AR2)
      - Regenerate & Restore (AR3)
      - Transform (AR4)
    ar_framework:
      AR1_avoid:
        - Prevent conversion
        - Prevent degradation
        - Prevent pollution
      AR2_reduce:
        - Reduce resource use
        - Reduce land use intensity
        - Reduce pollution
      AR3_regenerate_restore:
        - Regenerate ecosystems
        - Restore ecosystems
        - Improve ecosystem condition
      AR4_transform:
        - Shift to sustainable practices
        - Industry transformation
        - Policy advocacy

  step_5_track:
    purpose: "Track progress"
    activities:
      - Monitor indicators
      - Report progress
      - Verify claims
    output: "Annual progress reports"
```

### 2.3 Target Categories

```yaml
sbtn_target_categories:
  freshwater:
    quantity:
      target_type: "Reduction"
      baseline: "High water stress basins"
      methodology: "Science-based water targets"
    quality:
      target_type: "Reduction"
      pollutants: ["Nitrogen", "Phosphorus"]
      methodology: "Nutrient loading limits"

  land:
    conversion:
      target_type: "No conversion"
      scope: "Natural ecosystems"
      base_year: 2020
    degradation:
      target_type: "No degradation"
      scope: "High integrity areas"
    landscape_engagement:
      target_type: "Engagement"
      scope: "Priority landscapes"

  ocean:
    status: "In development"
    expected: "2025"

  biodiversity:
    status: "In development"
    expected: "2025"
```

### 2.4 Validation Process

| Stage | Duration | Requirements |
|-------|----------|--------------|
| Application | 2 weeks | Submit baseline data |
| Review | 6-8 weeks | Technical assessment |
| Feedback | 2 weeks | Address comments |
| Validation | 2 weeks | Final approval |
| Publication | Immediate | Public commitment |

### 2.5 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-sbtn-targets-v1 |
| Priority | P3-STANDARD |
| Deadline | Q2 2027 (production-ready) |
| Golden Tests | 100 |

**Key Features:**
- Materiality screening tool
- Value chain mapping
- Freshwater quantity calculator
- Land use baseline assessment
- Target validation package generator
- Progress tracking dashboard

---

## 3. EU Biodiversity Strategy 2030

### 3.1 Strategy Summary

**Official Name:** EU Biodiversity Strategy for 2030

**Publication Date:** May 2020

**Purpose:** Put Europe's biodiversity on path to recovery by 2030

### 3.2 Key Targets

```yaml
eu_biodiversity_targets:
  protected_areas:
    target: "30% of EU land and sea protected"
    current: "~26% land, ~11% sea"
    strict_protection: "10% (including old-growth forests)"

  restoration:
    target: "Legally binding restoration targets"
    regulation: "Nature Restoration Law (2024)"
    scope:
      - 20% of EU land by 2030
      - 20% of EU sea by 2030
      - All ecosystems by 2050

  pollinators:
    target: "Reverse pollinator decline"
    measures:
      - 50% reduction in pesticide use
      - 50% reduction in nutrient losses
      - 20% reduction in fertilizer use

  organic_farming:
    target: "25% of agricultural land organic by 2030"
    current: "~9%"

  urban_greening:
    target: "All cities >20k inhabitants have urban greening plans"

  trees:
    target: "3 billion additional trees by 2030"
```

### 3.3 Nature Restoration Law

```yaml
nature_restoration_law:
  citation: "Regulation (EU) 2024/..."
  status: "Adopted June 2024"
  entry_into_force: "August 2024"

  key_requirements:
    article_4_terrestrial:
      - "Restoration of 30% degraded habitats by 2030"
      - "60% by 2040"
      - "90% by 2050"

    article_5_marine:
      - "Restoration of marine habitats"
      - "Coherent network of protected areas"

    article_6_urban:
      - "No net loss of urban green space by 2030"
      - "5% increase by 2050"

    article_9_agricultural:
      - "Enhance biodiversity in agricultural land"
      - "Pollinator populations increasing trend"

    article_10_forests:
      - "Enhance forest biodiversity"
      - "Standing and lying deadwood"
      - "Uneven-aged forest structure"

  national_restoration_plans:
    deadline: "2026"
    content:
      - Restoration targets by ecosystem
      - Measures and timelines
      - Financing mechanisms
      - Monitoring plans
```

### 3.4 Corporate Implications

| Sector | Key Requirements | Reporting Link |
|--------|------------------|----------------|
| Agriculture | Sustainable practices, organic | CSRD ESRS E4 |
| Forestry | Sustainable forest management | EUDR, CSRD |
| Real Estate | Urban greening, biodiversity net gain | CSRD ESRS E4 |
| Financial Services | Nature-related risk assessment | SFDR, TNFD |
| Mining/Extractives | Restoration obligations | CSRD ESRS E4 |

### 3.5 GreenLang Agent Integration

**Relevant Agents:**
- gl-csrd-reporting-v1 (ESRS E4 Biodiversity)
- gl-tnfd-reporting-v1 (Nature disclosures)
- gl-eudr-compliance-v1 (Deforestation-free)

---

## 4. Deforestation Regulations Beyond EUDR

### 4.1 UK Environment Act - Deforestation Provisions

```yaml
uk_environment_act:
  citation: "Environment Act 2021, Schedule 17"
  status: "Secondary legislation pending"
  expected_implementation: "2025"

  scope:
    - Businesses using forest-risk commodities
    - UK market threshold TBD

  commodities:
    - Cattle products
    - Cocoa
    - Coffee
    - Oil palm
    - Rubber
    - Soy
    - (Potentially wood/timber)

  requirements:
    due_diligence:
      - Establish and implement due diligence system
      - Identify and obtain information on commodities
      - Assess risk of illegal deforestation
      - Mitigate risk

    reporting:
      - Annual public reporting
      - Country of origin
      - Due diligence actions
      - Risk assessment outcomes

  enforcement:
    - Civil penalties
    - Stop notices
    - Public naming

  differences_from_eudr:
    - "Illegal deforestation" vs "all deforestation"
    - Different baseline date (expected)
    - UK-specific enforcement
```

### 4.2 US FOREST Act (Proposed)

```yaml
us_forest_act:
  citation: "Fostering Overseas Rule of Law and Environmentally Sound Trade Act"
  status: "Proposed (multiple sessions)"

  scope:
    - Importers of forest-risk commodities
    - $500M+ in imported commodities

  commodities:
    - Cattle products
    - Cocoa
    - Coffee
    - Rubber
    - Soy
    - Palm oil
    - Pulp and paper

  requirements:
    due_diligence:
      - Supply chain mapping
      - Risk assessment
      - Mitigation measures
    reporting:
      - Annual disclosure
      - Country of origin
      - Deforestation-free certification

  current_status: "Not enacted; monitor future sessions"
```

### 4.3 Japan Clean Wood Act

```yaml
japan_clean_wood_act:
  citation: "Act on Promotion of Use of Legally-Logged Wood"
  status: "In force since 2017"

  scope:
    - Wood and wood products
    - All market participants

  requirements:
    - Due diligence for legality
    - Documentation of origin
    - Registration as "Wood-related Business"

  relationship_to_eudr: "Narrower scope (legality only, timber only)"
```

### 4.4 Australia Illegal Logging Prohibition

```yaml
australia_illegal_logging:
  citation: "Illegal Logging Prohibition Act 2012"
  status: "In force"

  scope:
    - Regulated timber products
    - Importers and domestic processors

  requirements:
    - Due diligence system
    - Country/species information
    - Compliance documentation

  penalties:
    - Civil penalties up to AUD 1.11M
    - Criminal penalties for deliberate violations
```

### 4.5 Comparative Matrix

| Element | EUDR | UK Environment | US FOREST | Japan | Australia |
|---------|------|----------------|-----------|-------|-----------|
| Status | In force | Pending | Proposed | In force | In force |
| Commodities | 7 | 6+ | 7 | 1 (wood) | 1 (wood) |
| Standard | Deforestation-free | Illegal deforestation | Deforestation-free | Legal | Legal |
| Cutoff date | Dec 2020 | TBD | TBD | N/A | N/A |
| Geolocation | Required | TBD | TBD | No | No |
| Penalties | 4% turnover | TBD | TBD | Limited | AUD 1.1M |

---

## 5. GreenLang Nature & Biodiversity Agent Suite

### 5.1 Agent Portfolio

| Agent | Framework | Priority | Status | ETA |
|-------|-----------|----------|--------|-----|
| gl-eudr-compliance-v1 | EUDR | P0 | In development | Dec 2025 |
| gl-tnfd-reporting-v1 | TNFD | P2 | Planned | Oct 2027 |
| gl-sbtn-targets-v1 | SBTN | P3 | Planned | Q2 2027 |
| gl-uk-deforestation-v1 | UK Env Act | P3 | Monitor | TBD |
| gl-biodiversity-esrs-v1 | CSRD ESRS E4 | P2 | Planned | Q3 2026 |

### 5.2 Shared Components

```yaml
nature_shared_components:
  spatial_data:
    - Geolocation validation
    - Protected area overlays
    - Biodiversity hotspot mapping
    - Ecosystem classification

  data_sources:
    - IBAT (Integrated Biodiversity Assessment Tool)
    - Global Forest Watch
    - WWF Risk Filter Suite
    - ENCORE database
    - World Database on Protected Areas (WDPA)

  calculators:
    - Land use change detection
    - Deforestation assessment
    - Freshwater use
    - Biodiversity impact scores

  reporting:
    - TNFD-aligned reports
    - ESRS E4 datapoints
    - Due diligence statements
    - Target validation packages
```

### 5.3 Data Requirements Summary

| Data Type | TNFD | SBTN | EUDR | ESRS E4 |
|-----------|------|------|------|---------|
| Geolocation | Required | Required | Mandatory | By location |
| Land use | Core metric | Baseline | Historical | Material |
| Water use | Sector-specific | Target area | - | Material |
| Supply chain | Value chain | Full chain | Mandatory | Material |
| Protected areas | Priority | Priority | Check | Disclosure |
| Ecosystem services | Dependency | Dependency | - | Dependency |

---

## 6. Implementation Recommendations

### 6.1 Development Priorities

| Phase | Timeline | Focus | Agents |
|-------|----------|-------|--------|
| Phase 1 | Q4 2025 | EUDR compliance | gl-eudr-compliance-v1 |
| Phase 2 | 2026 | CSRD biodiversity | gl-biodiversity-esrs-v1 |
| Phase 3 | 2027 | TNFD/SBTN | gl-tnfd-reporting-v1, gl-sbtn-targets-v1 |
| Phase 4 | 2027+ | Multi-jurisdiction | gl-uk-deforestation-v1 |

### 6.2 Resource Requirements

| Agent | Engineers | Scientists | Duration |
|-------|-----------|------------|----------|
| gl-eudr-compliance-v1 | 4 | 2 | 12 weeks |
| gl-tnfd-reporting-v1 | 3 | 2 | 16 weeks |
| gl-sbtn-targets-v1 | 2 | 2 | 12 weeks |
| gl-biodiversity-esrs-v1 | 2 | 1 | 8 weeks |

### 6.3 Integration Points

```yaml
integration_architecture:
  csrd_integration:
    - ESRS E4 biodiversity datapoints
    - Double materiality for nature
    - Nature-related risks in ESRS E1

  csddd_integration:
    - Environmental adverse impacts
    - Deforestation in supply chain
    - Biodiversity due diligence

  taxonomy_integration:
    - DNSH biodiversity criteria
    - Nature-positive activities
    - Restoration investments

  sbti_integration:
    - FLAG targets (land use)
    - Nature-based solutions
    - Carbon sequestration
```

---

## 7. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-05 | GL-RegulatoryIntelligence | Initial nature/biodiversity frameworks |

---

**END OF DOCUMENT**
