# US Climate Disclosure Regulatory Landscape

**Document ID:** GL-REG-US-001
**Version:** 1.0.0
**Date:** December 5, 2025
**Author:** GL-RegulatoryIntelligence
**Status:** ACTIVE

---

## Executive Summary

The United States is experiencing a surge in climate disclosure requirements at the state level, led by California's landmark SB 253 and SB 261. This document provides comprehensive analysis of all major US climate disclosure regulations, their requirements, and recommendations for GreenLang agent development.

### Regulatory Landscape Overview

| Regulation | Jurisdiction | Status | First Deadline | Companies Affected |
|------------|--------------|--------|----------------|-------------------|
| SEC Climate Rule | Federal | Stayed (litigation) | TBD | ~2,800 public companies |
| CA SB 253 | California | Enacted | Jun 30, 2026 | ~5,400 companies |
| CA SB 261 | California | Enacted | Jan 1, 2026 | ~10,000 companies |
| CO SB 23-016 | Colorado | Enacted | Jan 1, 2028 | ~1,000 companies |
| NY Climate Leadership | New York | Proposed | TBD | TBD |
| IL Climate Act | Illinois | Proposed | TBD | TBD |

---

## 1. SEC Climate Disclosure Rule

### 1.1 Regulation Summary

**Official Name:** The Enhancement and Standardization of Climate-Related Disclosures for Investors

**Legal Citation:** SEC Release Nos. 33-11275; 34-99678

**Scope:** Publicly traded companies registered with the SEC

### 1.2 Current Status

| Status Element | Details |
|----------------|---------|
| Adoption Date | March 6, 2024 |
| Effective Date | Stayed pending litigation |
| Court Status | Consolidated in 8th Circuit Court of Appeals |
| Litigation | Multiple challenges (industry groups + states) |
| Expected Resolution | 2025-2026 |

### 1.3 Key Requirements (If Reinstated)

#### Disclosure Content

| Requirement | Large Accelerated Filers | Accelerated Filers | Non-Accelerated/SRCs |
|-------------|-------------------------|--------------------|-----------------------|
| Scope 1 Emissions | Required (with assurance) | Required | Required |
| Scope 2 Emissions | Required (with assurance) | Required | Required |
| Scope 3 Emissions | If material | If material | Exempt |
| Climate Risk Disclosure | Required | Required | Required |
| Transition Plans | If material | If material | If material |
| Financial Statement Impacts | Required | Required | Required |

#### Phased Implementation Timeline (If Unstayed)

| Filer Type | Fiscal Year | Disclosure Deadline |
|------------|-------------|---------------------|
| Large Accelerated | FY 2025 | 2026 |
| Accelerated | FY 2026 | 2027 |
| Non-Accelerated/SRC | FY 2027 | 2028 |

#### Assurance Requirements

| Requirement | Phase 1 | Phase 2 | Phase 3 |
|-------------|---------|---------|---------|
| Large Accelerated | Limited (FY2026) | Reasonable (FY2028) | Reasonable |
| Accelerated | Limited (FY2028) | Reasonable (FY2030) | Reasonable |

### 1.4 Data Requirements

```yaml
sec_climate_data_requirements:
  scope_1_emissions:
    - Direct GHG emissions from owned/controlled sources
    - Breakdown by GHG (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
    - Emissions methodology (GHG Protocol or equivalent)
    - Organizational boundary approach

  scope_2_emissions:
    - Location-based method (required)
    - Market-based method (if used internally)
    - Grid emission factors

  scope_3_emissions:
    - Material categories only
    - Methodology disclosure
    - Data quality indicators

  climate_risk:
    - Physical risks (acute and chronic)
    - Transition risks
    - Risk management processes
    - Board oversight
    - Strategy integration

  financial_impacts:
    - Capitalized costs for severe weather
    - Carbon offsets and RECs (>1% of costs)
    - Climate-related expenditures
```

### 1.5 GreenLang Agent Recommendation

| Agent | Priority | Dependencies |
|-------|----------|--------------|
| gl-sec-climate-v1 | P2-MEDIUM (pending litigation) | gl-sb253-disclosure-v1 |

**Note:** Development should be deferred until litigation resolved. SB 253 agent can be adapted for SEC requirements due to GHG Protocol alignment.

---

## 2. California SB 253 (Climate Corporate Data Accountability Act)

### 2.1 Regulation Summary

**Official Name:** Climate Corporate Data Accountability Act

**Legal Citation:** California Senate Bill 253, Chapter 382

**Enacted:** October 7, 2023

**Effective Date:** January 1, 2024

### 2.2 Applicability Criteria

```yaml
applicability:
  revenue_threshold: $1,000,000,000  # $1B annual revenue
  jurisdiction_nexus: "Doing business in California"
  definition_doing_business:
    - Actively engaged in transaction for profit
    - Organized under CA law
    - Has CA property/employees
    - CA sales exceed $739,600 (2025 threshold)

  estimated_companies: 5,400+

  exclusions:
    - Companies below revenue threshold
    - Not doing business in California
    - Insurance companies (separate regs)
```

### 2.3 Reporting Timeline

| Reporting Element | First Report | Frequency | Data Year |
|-------------------|--------------|-----------|-----------|
| Scope 1 Emissions | Jun 30, 2026 | Annual | 2025 |
| Scope 2 Emissions | Jun 30, 2026 | Annual | 2025 |
| Scope 3 Emissions | Jun 30, 2027 | Annual | 2026 |

### 2.4 Third-Party Assurance Requirements

| Assurance Level | Start Date | Scope Coverage |
|-----------------|------------|----------------|
| Limited Assurance | 2026 | Scope 1 & 2 |
| Reasonable Assurance | 2030 | Scope 1, 2, & 3 |

**Assurance Standards:**
- ISAE 3000/3410
- AICPA AT-C Section 105
- Accredited third-party verifier required

### 2.5 Data Requirements

```yaml
sb253_data_requirements:
  scope_1:
    description: "Direct GHG emissions from owned/controlled sources"
    methodology: "GHG Protocol Corporate Standard"
    categories:
      - stationary_combustion
      - mobile_combustion
      - process_emissions
      - fugitive_emissions
    gases: [CO2, CH4, N2O, HFCs, PFCs, SF6, NF3]

  scope_2:
    description: "Indirect emissions from purchased energy"
    methods:
      - location_based: "Required"
      - market_based: "Required if used"
    energy_types:
      - electricity
      - steam
      - heating
      - cooling

  scope_3:
    description: "Value chain emissions"
    categories:
      upstream:
        1: "Purchased goods and services"
        2: "Capital goods"
        3: "Fuel- and energy-related activities"
        4: "Upstream transportation and distribution"
        5: "Waste generated in operations"
        6: "Business travel"
        7: "Employee commuting"
        8: "Upstream leased assets"
      downstream:
        9: "Downstream transportation and distribution"
        10: "Processing of sold products"
        11: "Use of sold products"
        12: "End-of-life treatment of sold products"
        13: "Downstream leased assets"
        14: "Franchises"
        15: "Investments"

  reporting_entity:
    - Legal entity name
    - EIN/Tax ID
    - NAICS code
    - Total revenue
    - California revenue

  data_quality:
    - Data Quality Indicators (DQI) per GHG Protocol
    - Uncertainty analysis
    - Exclusions justification
```

### 2.6 Penalty Structure

| Violation Type | Maximum Penalty |
|----------------|-----------------|
| Failure to file | $500,000/year |
| Inaccurate disclosure | $500,000/year |
| Non-compliance pattern | Cumulative penalties |

**Enforcement Agency:** California Air Resources Board (CARB)

### 2.7 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-sb253-disclosure-v1 |
| Priority | P0-CRITICAL |
| Deadline | May 2026 (production-ready) |
| Golden Tests | 300 |

**Key Features:**
- EPA eGRID integration for Scope 2
- All 15 Scope 3 category calculators
- CARB portal filing generation
- Third-party assurance package
- Multi-state support (CA, CO, WA)

---

## 3. California SB 261 (Climate-Related Financial Risk Act)

### 3.1 Regulation Summary

**Official Name:** Greenhouse Gases: Climate-Related Financial Risk

**Legal Citation:** California Senate Bill 261, Chapter 383

**Enacted:** October 7, 2023

### 3.2 Applicability Criteria

```yaml
applicability:
  revenue_threshold: $500,000,000  # $500M annual revenue
  jurisdiction_nexus: "Doing business in California"
  estimated_companies: 10,000+

  exclusions:
    - Companies below revenue threshold
    - Insurance companies (CDI requirements)
```

### 3.3 Reporting Requirements

| Requirement | Description | First Report |
|-------------|-------------|--------------|
| Climate Risk Disclosure | TCFD-aligned or equivalent | Jan 1, 2026 |
| Biennial Updates | Every two years | Jan 1, 2028 |

### 3.4 Disclosure Content (TCFD Framework)

```yaml
tcfd_disclosure_requirements:
  governance:
    - Board oversight of climate risks
    - Management role in assessing risks
    - Climate expertise on board

  strategy:
    - Climate-related risks and opportunities
    - Impact on business strategy
    - Resilience of strategy under scenarios
    - Short/medium/long-term horizons

  risk_management:
    - Processes for identifying climate risks
    - Processes for managing climate risks
    - Integration with overall risk management

  metrics_and_targets:
    - Metrics used to assess risks
    - Scope 1, 2, 3 emissions (cross-reference SB 253)
    - Climate-related targets and progress

  scenario_analysis:
    - Physical risk scenarios
    - Transition risk scenarios
    - 1.5C and 2C pathways
    - Business-as-usual scenario
```

### 3.5 Penalty Structure

| Violation Type | Maximum Penalty |
|----------------|-----------------|
| Failure to disclose | $50,000/reporting cycle |
| Inadequate disclosure | Administrative action |

### 3.6 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-climate-risk-v1 |
| Priority | P1-HIGH |
| Deadline | Nov 2025 (production-ready) |
| Golden Tests | 150 |

**Key Features:**
- TCFD alignment validation
- Scenario analysis tools
- Physical/transition risk assessment
- Integration with SB 253 emissions data

---

## 4. Colorado SB 23-016

### 4.1 Regulation Summary

**Official Name:** Greenhouse Gas Emissions Reporting

**Legal Citation:** Colorado Senate Bill 23-016

**Enacted:** 2023

### 4.2 Applicability Criteria

```yaml
applicability:
  revenue_threshold: $500,000,000  # $500M annual revenue
  employee_threshold: 500  # 500+ employees in Colorado
  jurisdiction_nexus: "Operating in Colorado"
  estimated_companies: 1,000+

  applies_if:
    - Revenue >= $500M AND
    - Employees in CO >= 500
```

### 4.3 Reporting Timeline

| Milestone | Date | Requirement |
|-----------|------|-------------|
| Registration | Nov 15, 2026 | Register with CDPHE |
| First Report | Jan 1, 2028 | Scope 1, 2, 3 disclosure |
| Annual Reports | Ongoing | Annual updates |

### 4.4 Data Requirements

```yaml
colorado_data_requirements:
  scope_1:
    methodology: "GHG Protocol Corporate Standard"
    breakdown_by: [facility, source_type]

  scope_2:
    methods: [location_based, market_based]
    grid_factors: "State-specific where available"

  scope_3:
    categories: "All 15 categories"
    estimation_allowed: true
    methodology_disclosure: required

  additional:
    - Company identification
    - Colorado operations details
    - Employee count
    - Methodology documentation
```

### 4.5 Assurance Requirements

| Level | Requirement |
|-------|-------------|
| Limited Assurance | Required |
| Reasonable Assurance | Not currently required |

### 4.6 Penalty Structure

| Violation Type | Penalty |
|----------------|---------|
| Failure to file | Per CDPHE regulations |
| Inaccurate reporting | Administrative penalties |

**Enforcement Agency:** Colorado Department of Public Health and Environment (CDPHE)

### 4.7 GreenLang Agent Recommendation

**Primary Agent:** gl-sb253-disclosure-v1 (multi-state extension)

**Colorado-Specific Features:**
- CDPHE filing format
- Colorado employee verification
- State-specific grid factors

---

## 5. New York Climate Leadership and Community Protection Act (CLCPA)

### 5.1 Regulation Summary

**Official Name:** Climate Leadership and Community Protection Act

**Legal Citation:** New York S. 6599

**Enacted:** 2019

**Status:** Framework law; implementing regulations in development

### 5.2 Current Requirements

```yaml
clcpa_current_requirements:
  statewide_targets:
    - 40% GHG reduction by 2030 (from 1990 levels)
    - 85% GHG reduction by 2050
    - Net-zero by 2050

  electricity_sector:
    - 70% renewable by 2030
    - 100% zero-emission by 2040

  current_reporting:
    - Large facilities under EPA GHGRP
    - State GHG inventory (agency-level)
```

### 5.3 Proposed Corporate Requirements

**Status:** Under development by Climate Action Council

| Proposed Element | Description |
|------------------|-------------|
| Corporate Disclosure | Emissions reporting for large companies |
| Financial Alignment | Climate-aligned financial disclosures |
| Value Chain | Supply chain emissions consideration |

### 5.4 GreenLang Agent Recommendation

| Agent | Priority | Status |
|-------|----------|--------|
| gl-ny-climate-v1 | P3-STANDARD | Monitor development |

**Recommendation:** Track CAC rulemakings; adapt SB 253 agent when requirements finalized.

---

## 6. State-by-State Comparison Matrix

### 6.1 Emissions Disclosure Requirements

| State | Regulation | Revenue Threshold | Scope 1 | Scope 2 | Scope 3 | First Deadline |
|-------|------------|-------------------|---------|---------|---------|----------------|
| California | SB 253 | $1B | Required | Required | Required | Jun 2026 |
| California | SB 261 | $500M | Via SB 253 | Via SB 253 | Via SB 253 | Jan 2026 |
| Colorado | SB 23-016 | $500M + 500 employees | Required | Required | Required | Jan 2028 |
| New York | CLCPA | TBD | TBD | TBD | TBD | TBD |
| Washington | HB 1589 (proposed) | $1B | Proposed | Proposed | Proposed | TBD |
| Illinois | (proposed) | TBD | TBD | TBD | TBD | TBD |
| Massachusetts | (proposed) | TBD | TBD | TBD | TBD | TBD |

### 6.2 Assurance Requirements

| State | Regulation | Limited Assurance | Reasonable Assurance |
|-------|------------|-------------------|----------------------|
| California | SB 253 | 2026 | 2030 |
| Colorado | SB 23-016 | Required | Not required |
| SEC (Federal) | Climate Rule | 2026/2028 (stayed) | 2028/2030 (stayed) |

### 6.3 Climate Risk Disclosure

| State | Regulation | TCFD Alignment | Scenario Analysis | First Deadline |
|-------|------------|----------------|-------------------|----------------|
| California | SB 261 | Required | Required | Jan 2026 |
| SEC (Federal) | Climate Rule | Partial (stayed) | Required (stayed) | TBD |

### 6.4 Penalty Comparison

| State | Regulation | Maximum Penalty | Enforcement Agency |
|-------|------------|-----------------|---------------------|
| California | SB 253 | $500,000/year | CARB |
| California | SB 261 | $50,000/cycle | CARB |
| Colorado | SB 23-016 | Per CDPHE | CDPHE |
| SEC (Federal) | Climate Rule | SEC penalties | SEC |

---

## 7. GreenLang Agent Development Strategy

### 7.1 Multi-State Architecture

```yaml
multi_state_agent_architecture:
  base_agent: gl-sb253-disclosure-v1
  state_modules:
    california_sb253:
      filing_portal: CARB
      format: CARB XML
      assurance: ISAE 3410

    california_sb261:
      filing_portal: TBD
      format: TCFD
      cross_reference: SB 253 emissions

    colorado_sb23_016:
      filing_portal: CDPHE
      format: CDPHE schema
      employee_verification: required

    washington_hb1589:
      status: monitor
      filing_portal: Ecology

  shared_components:
    - GHG Protocol calculators
    - EPA eGRID integration
    - Scope 3 category engines
    - Assurance package generator
    - Audit trail provenance
```

### 7.2 Development Priorities

| Priority | Agent | State Coverage | Deadline |
|----------|-------|----------------|----------|
| P0 | gl-sb253-disclosure-v1 | CA, CO, WA | May 2026 |
| P1 | gl-climate-risk-v1 | CA (SB 261) | Nov 2025 |
| P2 | gl-sec-climate-v1 | Federal | TBD (litigation) |

### 7.3 Golden Test Distribution

| Test Category | SB 253 | SB 261 | Colorado | Total |
|---------------|--------|--------|----------|-------|
| Scope 1 Calculations | 60 | - | 20 | 80 |
| Scope 2 Calculations | 70 | - | 20 | 90 |
| Scope 3 Calculations | 120 | - | 40 | 160 |
| Verification | 50 | - | 20 | 70 |
| TCFD/Risk | - | 80 | - | 80 |
| Multi-State Filing | - | - | 20 | 20 |
| **Total** | **300** | **80** | **120** | **500** |

---

## 8. Compliance Checklist

### 8.1 California SB 253 Pre-Filing Checklist

- [ ] Confirm revenue threshold ($1B+)
- [ ] Verify California nexus
- [ ] Establish organizational boundary
- [ ] Collect Scope 1 data (fuel, process, fugitive)
- [ ] Collect Scope 2 data (electricity, steam, heat)
- [ ] Collect Scope 3 data (all 15 categories)
- [ ] Apply GHG Protocol methodology
- [ ] Calculate emissions with proper EFs
- [ ] Document data quality indicators
- [ ] Engage third-party verifier
- [ ] Generate CARB filing format
- [ ] Submit by June 30

### 8.2 California SB 261 Pre-Filing Checklist

- [ ] Confirm revenue threshold ($500M+)
- [ ] Verify California nexus
- [ ] Establish TCFD governance disclosures
- [ ] Document strategy and risk management
- [ ] Conduct scenario analysis
- [ ] Cross-reference SB 253 emissions
- [ ] Prepare disclosure report
- [ ] Submit by January 1 (biennial)

---

## 9. Upcoming Regulatory Developments

### 9.1 Watch List

| Development | Status | Expected Date | Impact |
|-------------|--------|---------------|--------|
| SEC Rule Litigation | 8th Circuit | 2025-2026 | May resurrect federal requirements |
| CARB Implementation | In progress | Q1 2026 | Filing formats and portals |
| CDPHE Regulations | Drafting | 2026 | Colorado-specific requirements |
| Washington HB 1589 | Legislative | 2026 session | New state requirements |
| Massachusetts Bill | Proposed | 2026 session | New state requirements |
| Illinois Climate Act | Proposed | 2026-2027 | New state requirements |

### 9.2 Federal Developments

| Development | Status | Potential Impact |
|-------------|--------|------------------|
| EPA GHGRP Expansion | Under review | Lower thresholds possible |
| Federal Buildings | Executive Order | Scope 3 for contractors |
| Inflation Reduction Act | Implementation | Incentive reporting |

---

## 10. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-05 | GL-RegulatoryIntelligence | Initial US climate disclosure analysis |

---

**END OF DOCUMENT**
