# EU Sustainability Regulatory Landscape

**Document ID:** GL-REG-EU-001
**Version:** 1.0.0
**Date:** December 5, 2025
**Author:** GL-RegulatoryIntelligence
**Status:** ACTIVE

---

## Executive Summary

The European Union has established the world's most comprehensive sustainability regulatory framework through the European Green Deal. This document provides detailed analysis of major EU regulations affecting corporate sustainability reporting, supply chain due diligence, and environmental claims.

### Key Regulations Overview

| Regulation | Status | Key Deadline | Companies Affected | GreenLang Agent |
|------------|--------|--------------|-------------------|-----------------|
| CSRD | In force | Jan 1, 2024+ | ~50,000 | gl-csrd-reporting-v1 |
| CSDDD | Adopted | Jul 26, 2027+ | ~5,500 | gl-csddd-v1 |
| EU Taxonomy | In force | Ongoing | CSRD-scope | gl-eu-taxonomy-v1 |
| CBAM | Transition | Jan 1, 2027 | All importers | gl-cbam-importer-v1 |
| EUDR | In force | Dec 30, 2025 | All operators | gl-eudr-compliance-v1 |
| Green Claims | Proposed | Sep 2026 (est) | All B2C claims | gl-green-claims-v1 |

---

## 1. Corporate Sustainability Reporting Directive (CSRD)

### 1.1 Regulation Summary

**Official Name:** Corporate Sustainability Reporting Directive

**Legal Citation:** Directive (EU) 2022/2464

**Publication Date:** December 14, 2022

**Amends:** Non-Financial Reporting Directive (NFRD)

### 1.2 Scope and Applicability

```yaml
csrd_scope:
  wave_1:
    description: "Large public-interest entities already under NFRD"
    criteria:
      - Listed companies
      - Banks
      - Insurance companies
      - Employee threshold: 500+
    reporting_start: FY 2024
    first_report_due: 2025

  wave_2:
    description: "Large companies meeting 2 of 3 criteria"
    criteria:
      option_1: "Balance sheet > EUR 25M"
      option_2: "Net turnover > EUR 50M"
      option_3: "Employees > 250"
    reporting_start: FY 2025
    first_report_due: 2026

  wave_3:
    description: "Listed SMEs, small credit institutions, captive insurance"
    criteria:
      - Listed on EU regulated market
      - Meet SME thresholds
    reporting_start: FY 2026
    first_report_due: 2027
    opt_out: "Until 2028"

  wave_4:
    description: "Non-EU companies with significant EU operations"
    criteria:
      - EU net turnover > EUR 150M
      - At least one EU subsidiary or branch
    reporting_start: FY 2028
    first_report_due: 2029

  total_companies: ~50,000
```

### 1.3 European Sustainability Reporting Standards (ESRS)

#### Standard Structure

| Standard | Name | Mandatory | Disclosure Requirements |
|----------|------|-----------|------------------------|
| ESRS 1 | General requirements | Yes | Materiality, reporting principles |
| ESRS 2 | General disclosures | Yes | Governance, strategy, impacts |
| ESRS E1 | Climate change | Conditional | GHG emissions, energy, transition |
| ESRS E2 | Pollution | Conditional | Air, water, soil pollution |
| ESRS E3 | Water and marine resources | Conditional | Water consumption, marine impacts |
| ESRS E4 | Biodiversity and ecosystems | Conditional | Biodiversity impacts, dependencies |
| ESRS E5 | Resource use and circular economy | Conditional | Material flows, circularity |
| ESRS S1 | Own workforce | Conditional | Employment, health & safety |
| ESRS S2 | Workers in value chain | Conditional | Supply chain labor |
| ESRS S3 | Affected communities | Conditional | Community impacts |
| ESRS S4 | Consumers and end-users | Conditional | Product impacts |
| ESRS G1 | Business conduct | Conditional | Ethics, corruption, lobbying |

#### ESRS E1 Climate Change (Key Datapoints)

```yaml
esrs_e1_requirements:
  transition_plan:
    - Climate transition plan existence
    - GHG reduction targets (scope 1, 2, 3)
    - Decarbonization levers
    - CapEx/OpEx for transition
    - Locked-in GHG emissions

  ghg_emissions:
    scope_1:
      - Total gross emissions (tCO2e)
      - Percentage from regulated ETS
      - Breakdown by country (if material)
    scope_2:
      - Location-based (tCO2e)
      - Market-based (tCO2e)
    scope_3:
      - All material categories
      - Categories 1-15 where applicable
    total:
      - Location-based total
      - Market-based total

  energy:
    - Total energy consumption (MWh)
    - Energy consumption from fossil sources
    - Energy consumption from nuclear
    - Energy consumption from renewables
    - Energy intensity (per net revenue)

  targets:
    - GHG reduction targets
    - Base year and target year
    - Scope coverage
    - Progress toward targets

  carbon_credits:
    - Carbon credits planned for claims
    - Type, standard, and vintage
```

### 1.4 Double Materiality Assessment

```yaml
double_materiality:
  impact_materiality:
    description: "Inside-out: Company impacts on environment and society"
    assessment_criteria:
      - Severity (scale, scope, irremediability)
      - Likelihood
      - Actual vs potential impacts
      - Short/medium/long-term

  financial_materiality:
    description: "Outside-in: Sustainability matters affecting company"
    assessment_criteria:
      - Risk magnitude
      - Probability
      - Time horizon
      - Dependencies and opportunities

  process:
    1: "Identify IROs (Impacts, Risks, Opportunities)"
    2: "Assess against thresholds"
    3: "Determine material topics"
    4: "Document assessment"
    5: "Stakeholder validation"
```

### 1.5 Reporting Format

| Element | Requirement |
|---------|-------------|
| Format | iXBRL (Inline XBRL) |
| Taxonomy | ESRS XBRL Taxonomy |
| Location | Management report |
| Assurance | Limited (mandatory), Reasonable (by 2028) |
| Filing | European Single Access Point (ESAP) |

### 1.6 Penalty Structure

| Member State | Maximum Penalty |
|--------------|-----------------|
| Germany | EUR 10M or 5% turnover |
| France | EUR 75,000 + EUR 37,500/day |
| Netherlands | EUR 900,000 |
| Italy | EUR 150,000 |
| Spain | EUR 1M |

### 1.7 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-csrd-reporting-v1 |
| Priority | P1-HIGH |
| Deadline | Jun 2026 (production-ready) |
| Golden Tests | 500 |

**Key Features:**
- Double materiality assessment engine
- All ESRS E1-E5, S1-S4, G1 calculators
- iXBRL/ESEF report generator
- ESRS gap analysis tool
- Management report integration

---

## 2. Corporate Sustainability Due Diligence Directive (CSDDD/CS3D)

### 2.1 Regulation Summary

**Official Name:** Corporate Sustainability Due Diligence Directive

**Legal Citation:** Directive (EU) 2024/1760

**Publication Date:** July 5, 2024

**Entry into Force:** July 25, 2024

### 2.2 Scope and Timeline

```yaml
csddd_scope:
  group_1:
    description: "Largest companies"
    criteria:
      - EU companies: >1,000 employees AND >EUR 450M turnover
      - Non-EU: >EUR 450M EU turnover
    compliance_date: July 26, 2027

  group_2:
    description: "Large companies"
    criteria:
      - EU companies: >500 employees AND >EUR 150M turnover
      - Non-EU: >EUR 250M EU turnover (with 40% in high-impact sectors)
    compliance_date: July 26, 2028

  group_3:
    description: "Smaller in-scope companies"
    criteria:
      - Reduced thresholds
      - Ultimate parent company obligations
    compliance_date: July 26, 2029

  total_companies: ~5,500 EU + ~900 non-EU
```

### 2.3 Due Diligence Requirements

```yaml
csddd_obligations:
  step_1_integrate:
    description: "Integrate due diligence into policies"
    requirements:
      - DD policy covering all operations
      - Business partner expectations
      - Annual policy review

  step_2_identify:
    description: "Identify adverse impacts"
    scope:
      - Own operations
      - Subsidiaries
      - Business partners (direct and indirect)
    impacts:
      human_rights:
        - ILO core conventions
        - UN Guiding Principles
        - Child labor
        - Forced labor
        - Health and safety
      environmental:
        - Climate change
        - Biodiversity
        - Pollution
        - Deforestation
        - Water use

  step_3_prevent_mitigate:
    description: "Prevent and mitigate impacts"
    actions:
      - Prevention action plan
      - Contractual assurances
      - Capacity building
      - Financial support for SME suppliers
      - Verification measures

  step_4_end_or_minimize:
    description: "End or minimize actual impacts"
    requirements:
      - Cessation measures
      - Remediation plans
      - Neutralization measures
      - Suspension as last resort

  step_5_remediation:
    description: "Provide remediation"
    mechanisms:
      - Grievance mechanism
      - Remediation process
      - Financial remediation where appropriate

  step_6_engagement:
    description: "Stakeholder engagement"
    stakeholders:
      - Workers and representatives
      - Trade unions
      - Civil society
      - Affected communities

  step_7_monitoring:
    description: "Monitoring effectiveness"
    frequency: "Annual at minimum"
    methods:
      - Audits
      - Surveys
      - Site visits
      - Third-party assessments
```

### 2.4 Climate Transition Plan

```yaml
climate_transition_plan:
  requirement: "Mandatory for Group 1 and Group 2 companies"
  content:
    - Time-bound targets for GHG reduction
    - 2030 and 2050 milestones
    - Paris Agreement alignment
    - Scope 1, 2, 3 coverage
    - Decarbonization levers
    - Investment plans
    - Board oversight

  integration:
    - CSRD ESRS E1 alignment
    - Science-based targets recommended
    - Annual progress reporting
```

### 2.5 Penalty Structure

| Penalty Type | Maximum Amount |
|--------------|----------------|
| Administrative fines | 5% worldwide net turnover |
| Civil liability | Full compensation for damages |
| Supervisory measures | Compliance orders, public statements |

### 2.6 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-csddd-v1 |
| Priority | P3-MEDIUM |
| Deadline | May 2027 (production-ready) |
| Golden Tests | 250 |

**Key Features:**
- Supplier risk assessment engine
- Value chain mapping (Tier 1-N)
- Adverse impact identifier
- Grievance tracking system
- Remediation workflow
- Climate transition plan integration

---

## 3. EU Taxonomy Regulation

### 3.1 Regulation Summary

**Official Name:** Taxonomy Regulation

**Legal Citation:** Regulation (EU) 2020/852

**Purpose:** Classification system for environmentally sustainable economic activities

### 3.2 Environmental Objectives

| Objective | Status | Delegated Acts |
|-----------|--------|----------------|
| Climate change mitigation | Active | Climate Delegated Act (2021) |
| Climate change adaptation | Active | Climate Delegated Act (2021) |
| Sustainable use of water | Active | Environmental Delegated Act (2023) |
| Transition to circular economy | Active | Environmental Delegated Act (2023) |
| Pollution prevention | Active | Environmental Delegated Act (2023) |
| Biodiversity and ecosystems | Active | Environmental Delegated Act (2023) |

### 3.3 Taxonomy Alignment Assessment

```yaml
taxonomy_alignment:
  step_1_eligibility:
    description: "Activity in scope of Taxonomy"
    check: "NACE code mapping to Taxonomy activities"

  step_2_substantial_contribution:
    description: "Meets technical screening criteria for objective"
    examples:
      electricity_generation:
        threshold: "<100g CO2e/kWh lifecycle"
      building_renovation:
        threshold: ">30% primary energy demand reduction"
      transport:
        threshold: "Zero direct emissions"

  step_3_dnsh:
    description: "Do No Significant Harm to other 5 objectives"
    assessment:
      climate_adaptation: "Climate risk assessment"
      water: "Environmental Impact Assessment"
      circular_economy: "Waste hierarchy compliance"
      pollution: "REACH, IED compliance"
      biodiversity: "EIA, protected areas check"

  step_4_minimum_safeguards:
    description: "Comply with social safeguards"
    standards:
      - OECD Guidelines for MNEs
      - UN Guiding Principles on Business and Human Rights
      - ILO Core Conventions
```

### 3.4 Reporting KPIs

```yaml
taxonomy_kpis:
  turnover:
    formula: "Aligned turnover / Total turnover x 100"
    disclosure: "By activity and objective"

  capex:
    formula: "Aligned CapEx / Total CapEx x 100"
    types:
      - Category A: "Related to Taxonomy-aligned activities"
      - Category B: "Taxonomy-aligned CapEx plan"
      - Category C: "Purchase of Taxonomy-aligned outputs"

  opex:
    formula: "Aligned OpEx / Total OpEx x 100"
    scope: "Research, renovation, short-term leases, maintenance"
```

### 3.5 Delegated Acts Updates (2024-2025)

| Update | Status | Key Changes |
|--------|--------|-------------|
| Climate DA Amendment | Effective 2024 | Nuclear and gas criteria |
| Environmental DA | Effective 2024 | Water, CE, pollution, biodiversity criteria |
| Disclosure DA Amendment | Effective 2024 | Template updates |
| Platform on Sustainable Finance recommendations | 2024-2025 | Activity scope expansion |

### 3.6 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-eu-taxonomy-v1 |
| Priority | P2-MEDIUM |
| Deadline | Jul 2026 (production-ready) |
| Golden Tests | 300 |

**Key Features:**
- NACE to Taxonomy activity mapper
- Technical screening criteria evaluator
- DNSH assessment engine
- Minimum safeguards validator
- Taxonomy KPI calculator
- CSRD/iXBRL integration

---

## 4. Carbon Border Adjustment Mechanism (CBAM)

### 4.1 Regulation Summary

**Official Name:** Carbon Border Adjustment Mechanism

**Legal Citation:** Regulation (EU) 2023/956

**Purpose:** Prevent carbon leakage by applying carbon price to imports

### 4.2 Timeline

| Phase | Period | Requirements |
|-------|--------|--------------|
| Transition | Oct 1, 2023 - Dec 31, 2025 | Quarterly reports, no payments |
| Transition (extended) | Jan 1, 2026 - Dec 31, 2026 | Quarterly reports, certificates begin |
| Permanent | Jan 1, 2027 onwards | Full compliance, certificate surrender |

### 4.3 Product Scope

```yaml
cbam_product_scope:
  phase_1:
    - cement
    - iron_and_steel
    - aluminium
    - fertilizers
    - electricity
    - hydrogen

  potential_expansion:
    - organic_chemicals
    - plastics
    - ammonia_derivatives

  cn_codes:
    cement: ["2507", "2523"]
    iron_steel: ["72", "73"]
    aluminium: ["76"]
    fertilizers: ["2808", "2814", "2834", "3102", "3105"]
    electricity: ["2716"]
    hydrogen: ["2804"]
```

### 4.4 Emissions Calculation

```yaml
cbam_emissions:
  direct_emissions:
    description: "Emissions from production process"
    methodology:
      - Actual emissions (preferred)
      - Default values (if actual unavailable)
    requirement: "Per tonne of product"

  indirect_emissions:
    description: "Emissions from electricity used"
    applies_to: ["cement", "iron_steel", "aluminium", "hydrogen", "fertilizers"]
    methodology:
      - Grid average emission factor
      - Contractual arrangements (PPAs)

  embedded_emissions:
    formula: "Direct + Indirect emissions"
    unit: "tCO2e per tonne of product"

  default_values:
    source: "EU implementing acts"
    basis: "Country of origin + average EU"
    fallback: "Highest 10% of EU producers"
```

### 4.5 CBAM Certificate Requirements

```yaml
cbam_certificates:
  price: "Weekly EU ETS price"
  purchase: "From national competent authority"
  surrender: "By May 31 for previous year"

  calculation:
    formula: "Embedded emissions x Certificates - Deductions"

  deductions:
    - Carbon price paid in country of origin
    - Free EU ETS allocation adjustment

  reporting:
    quarterly:
      - Imported quantity
      - Embedded emissions
      - Country of origin
      - Installation of production
    annual:
      - Total embedded emissions
      - Certificates surrendered
      - Verification report
```

### 4.6 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-cbam-importer-v1 |
| Priority | P0-CRITICAL |
| Deadline | Nov 2026 (production-ready) |
| Golden Tests | 200 |

**Key Features:**
- Embedded emissions calculator
- Default values database
- Certificate requirements calculator
- Quarterly report generator
- Annual declaration generator
- Verification package

---

## 5. Green Claims Directive

### 5.1 Regulation Summary

**Official Name:** Proposal for a Directive on Green Claims

**Legal Citation:** COM/2023/166

**Status:** Under negotiation (trilogue expected 2025)

**Purpose:** Substantiation and verification of environmental claims

### 5.2 Scope

```yaml
green_claims_scope:
  applies_to:
    - B2C environmental claims
    - Environmental labels
    - Carbon neutrality claims
    - Product environmental footprint claims

  excludes:
    - Regulated eco-labels (EU Ecolabel)
    - Mandatory disclosures
    - B2B communications (excluded from main text)
```

### 5.3 Substantiation Requirements

```yaml
substantiation_requirements:
  explicit_claims:
    examples:
      - "Eco-friendly"
      - "Sustainable packaging"
      - "Reduced carbon footprint"
    requirements:
      - Scientific evidence
      - PEF/OEF methodology (where applicable)
      - Whole lifecycle consideration
      - Identification of tradeoffs

  comparative_claims:
    examples:
      - "50% less emissions than competitor"
      - "Most sustainable in category"
    requirements:
      - Same methodology
      - Equivalent scope
      - Recent data
      - Statistical significance

  carbon_neutrality_claims:
    examples:
      - "Carbon neutral"
      - "Climate positive"
      - "Net zero"
    requirements:
      - Own emissions reduction first
      - High-quality offsets only
      - Transparent methodology
      - Offset details disclosure
```

### 5.4 Environmental Label Requirements

```yaml
label_requirements:
  new_labels:
    - Approval by competent authority
    - Third-party certification
    - Publicly available criteria
    - Regular review

  existing_labels:
    - Demonstrate compliance within 24 months
    - Or cease use

  prohibited:
    - Self-declared labels without verification
    - Labels based on offset-only claims
```

### 5.5 Penalty Structure (Proposed)

| Violation | Penalty Range |
|-----------|---------------|
| Unsubstantiated claims | Up to 4% annual turnover |
| False claims | Market ban, recall |
| Repeated violations | Exclusion from public procurement |

### 5.6 GreenLang Agent Specification

| Attribute | Value |
|-----------|-------|
| Agent ID | gl-green-claims-v1 |
| Priority | P2-MEDIUM |
| Deadline | Aug 2026 (production-ready) |
| Golden Tests | 200 |

**Key Features:**
- Claim text analyzer (NLP)
- PEF/OEF calculator
- Evidence quality scorer
- Comparative claim validator
- Offset registry integration
- Substantiation report generator

---

## 6. EU Regulatory Integration Matrix

### 6.1 Cross-Regulation Data Requirements

| Data Element | CSRD | CSDDD | Taxonomy | CBAM | EUDR | Green Claims |
|--------------|------|-------|----------|------|------|--------------|
| Scope 1 Emissions | Required | Linked | Linked | Direct | Related | Linked |
| Scope 2 Emissions | Required | Linked | Linked | Indirect | - | Linked |
| Scope 3 Emissions | Required | Linked | - | - | Supply chain | Linked |
| Supply Chain Data | Material | Required | - | Origin | Required | Product |
| Climate Targets | Required | Required | - | - | - | Net zero |
| Human Rights DD | Material | Required | Safeguards | - | Deforestation | - |
| Product LCA | - | - | Some | - | - | Required |

### 6.2 Timeline Coordination

```
2024 |----[CSRD Wave 1]---------------------------------------------->
2025 |--------[CSRD Wave 2]--[EUDR]---------------------------------->
2026 |------------[CBAM Transition]--[Green Claims]------------------>
2027 |----------------[CBAM Full]--[CSDDD G1]------------------------>
2028 |--------------------[CSDDD G2]--[CSRD Wave 4]------------------>
2029 |------------------------[CSDDD G3]---------------------------->
2030 |---------------------------[EU Climate Target 55%]------------->
```

### 6.3 GreenLang Integration Architecture

```yaml
integrated_platform:
  shared_data_layer:
    - Emissions inventory (all scopes)
    - Supply chain mapping
    - Product lifecycle data
    - Climate targets and progress

  agent_dependencies:
    gl-csrd-reporting-v1:
      requires: [gl-eu-taxonomy-v1, gl-csddd-v1]
      provides: [emissions_data, materiality_assessment]

    gl-csddd-v1:
      requires: [gl-eudr-compliance-v1]
      provides: [supply_chain_data, adverse_impacts]

    gl-eu-taxonomy-v1:
      requires: [gl-csrd-reporting-v1]
      provides: [alignment_kpis, activity_classification]

    gl-green-claims-v1:
      requires: [gl-csrd-reporting-v1, gl-carbon-offset-v1]
      provides: [claim_substantiation]

    gl-cbam-importer-v1:
      requires: []
      provides: [embedded_emissions, certificates]
```

---

## 7. Implementation Priorities

### 7.1 Agent Development Roadmap

| Phase | Timeline | Agents | Integration |
|-------|----------|--------|-------------|
| Phase 1 | Dec 2025 | EUDR | Standalone |
| Phase 2 | Q1-Q2 2026 | CSRD, CBAM | Shared emissions |
| Phase 3 | Q3-Q4 2026 | Taxonomy, Green Claims | Full integration |
| Phase 4 | Q1-Q2 2027 | CSDDD | Supply chain link |

### 7.2 Resource Requirements

| Agent | FTEs | Climate Scientists | Duration |
|-------|------|-------------------|----------|
| gl-csrd-reporting-v1 | 4 | 1 | 16 weeks |
| gl-csddd-v1 | 3 | 1 | 12 weeks |
| gl-eu-taxonomy-v1 | 3 | 1 | 12 weeks |
| gl-cbam-importer-v1 | 2 | 0.5 | 8 weeks |
| gl-green-claims-v1 | 2 | 1 | 10 weeks |

---

## 8. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-05 | GL-RegulatoryIntelligence | Initial EU regulatory landscape |

---

**END OF DOCUMENT**
