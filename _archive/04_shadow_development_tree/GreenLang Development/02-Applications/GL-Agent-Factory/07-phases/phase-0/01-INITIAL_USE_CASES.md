# Phase 0: Initial Use Cases - Flagship Agents

**Version:** 1.0
**Date:** 2025-12-03
**Product Manager:** GL-ProductManager
**Status:** Draft - Pending Approval

---

## Executive Summary

This document defines the first 2-3 flagship use cases for the Agent Factory. These agents will be the "wedge" that proves the factory concept, demonstrating that we can generate high-quality, certified agents from specifications rather than hand-coding them.

**Selection Criteria:**
1. High business impact (customer demand, revenue potential)
2. Regulatory urgency (deadline-driven compliance)
3. Technical feasibility (achievable within Phase 2)
4. Domain coverage (tests multiple factory capabilities)

---

## Use Case 1: Decarbonization Roadmap Engineer

### Overview

| Attribute | Value |
|-----------|-------|
| **Agent Name** | GL-DECARB-ROADMAP |
| **Target User** | Sustainability managers at industrial sites |
| **Regulatory Driver** | CSRD, Net-Zero commitments, EU Taxonomy |
| **Complexity** | High (multi-step reasoning, multiple data sources) |
| **Priority** | P0 - First agent to generate |

### Problem Statement

Industrial sites (steel mills, cement plants, chemical facilities) need customized decarbonization roadmaps that consider:
- Current emissions baseline
- Available technologies (electrification, CCS, hydrogen)
- Capex/Opex constraints
- Regulatory timelines (CSRD 2024, EU Taxonomy 2025, Net-Zero 2050)
- Supply chain impacts

**Current State:** Consultants spend 4-8 weeks creating each roadmap at $50K-$200K per site.

**Target State:** Agent generates draft roadmap in <4 hours, requiring only 2-3 hours of human review.

### Agent Specification (Preview)

```yaml
agent_id: gl-decarb-roadmap-v1
name: Decarbonization Roadmap Engineer
version: "1.0.0"
type: multi-step-reasoning

inputs:
  - name: site_profile
    type: object
    required: true
    schema:
      sector: string  # steel, cement, chemicals, etc.
      location: string
      annual_emissions_tco2e: number
      current_energy_mix: object
      capex_budget_eur: number
      timeline_years: number

  - name: regulatory_requirements
    type: array
    items:
      regulation: string
      deadline: date
      scope: string

  - name: technology_preferences
    type: object
    schema:
      include: array
      exclude: array

outputs:
  - name: roadmap
    type: object
    schema:
      phases: array
      total_abatement_tco2e: number
      total_cost_eur: number
      roi_years: number
      regulatory_compliance: object

  - name: technology_recommendations
    type: array
    items:
      technology: string
      abatement_potential_tco2e: number
      cost_eur: number
      implementation_timeline: object

  - name: risk_assessment
    type: object
    schema:
      technology_risks: array
      regulatory_risks: array
      financial_risks: array

tools:
  - emissions_calculator  # Scope 1, 2, 3 calculations
  - technology_database   # Technology options and costs
  - regulatory_tracker    # Deadline monitoring
  - financial_modeler     # Capex/Opex analysis

validation:
  - type: domain
    validators:
      - csrd_compliance_check
      - eu_taxonomy_alignment
      - science_based_targets_check

  - type: calculation
    validators:
      - emissions_arithmetic_check
      - cost_calculation_check
```

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Roadmap generation time | <4 hours | End-to-end processing time |
| Human review time | <3 hours | Time from draft to approved |
| Accuracy vs. consultant roadmap | >85% | Expert blind comparison |
| Regulatory compliance | 100% | Passes all validation checks |
| Customer satisfaction | >8/10 | Post-delivery survey |

### Acceptance Criteria

**Given** a valid site profile and regulatory requirements
**When** the agent processes the inputs
**Then**:
- [ ] Generates a phased roadmap with 3-5 technology interventions
- [ ] Calculates total abatement potential within 5% of manual calculation
- [ ] Identifies applicable regulations and compliance status
- [ ] Provides ROI analysis with payback period
- [ ] Flags high-risk technologies requiring further review
- [ ] Exports to PDF and Excel formats

### Technical Complexity Assessment

| Dimension | Complexity | Notes |
|-----------|------------|-------|
| Input parsing | Medium | Multiple data sources, varied formats |
| Reasoning steps | High | Multi-step optimization, constraint solving |
| Data dependencies | High | Technology DB, emissions factors, regulatory rules |
| Output generation | Medium | Structured report with visualizations |
| Validation | High | Domain validation, arithmetic checks |
| **Overall** | **High** | Good stress test for factory |

### Timeline

- **Week 1-2 (Phase 0):** Finalize agent spec, identify data sources
- **Week 3-8 (Phase 1):** Build SDK foundations, migrate emissions calculator
- **Week 9-16 (Phase 2):** Generate agent from spec, run evaluation
- **Week 17-20 (Phase 2):** Certification, golden tests, beta deployment

---

## Use Case 2: CSRD + CBAM Planning Copilot

### Overview

| Attribute | Value |
|-----------|-------|
| **Agent Name** | GL-CSRD-CBAM-COPILOT |
| **Target User** | Compliance officers, CFOs, sustainability teams |
| **Regulatory Driver** | CSRD (2024-2025), CBAM (2025-2026) |
| **Complexity** | Medium (document analysis, calculation, reporting) |
| **Priority** | P0 - Second agent to generate |

### Problem Statement

Companies subject to CSRD (17,000+ in EU) and CBAM (all importers of carbon-intensive goods) face overlapping compliance requirements:
- CSRD: Disclose climate risks, emissions, transition plans
- CBAM: Report embedded emissions on imports (steel, cement, aluminum, fertilizers, hydrogen)

**Current State:** Companies treat these as separate workstreams, duplicating effort and risking inconsistencies.

**Target State:** Integrated copilot that manages both CSRD and CBAM compliance in a unified workflow.

### Agent Specification (Preview)

```yaml
agent_id: gl-csrd-cbam-copilot-v1
name: CSRD + CBAM Planning Copilot
version: "1.0.0"
type: workflow-orchestrator

inputs:
  - name: company_profile
    type: object
    required: true
    schema:
      name: string
      sector: string
      eu_subsidiaries: array
      employee_count: number
      annual_revenue_eur: number

  - name: emissions_data
    type: object
    schema:
      scope_1: number
      scope_2: number
      scope_3_categories: object

  - name: import_data
    type: array
    items:
      product_category: string  # steel, cement, etc.
      origin_country: string
      quantity_tonnes: number
      supplier: string

  - name: reporting_period
    type: object
    schema:
      start_date: date
      end_date: date

outputs:
  - name: csrd_report_draft
    type: object
    schema:
      esrs_disclosures: object  # E1, E2, E3, E4, E5, S1-S4, G1-G2
      data_gaps: array
      materiality_assessment: object

  - name: cbam_report
    type: object
    schema:
      quarterly_declaration: object
      embedded_emissions: array
      supplier_data_requests: array

  - name: compliance_calendar
    type: array
    items:
      deadline: date
      requirement: string
      status: string

  - name: data_collection_plan
    type: object
    schema:
      csrd_data_needs: array
      cbam_data_needs: array
      supplier_outreach: array

tools:
  - csrd_materiality_analyzer
  - cbam_emissions_calculator
  - esrs_gap_checker
  - regulatory_calendar
  - supplier_data_collector

validation:
  - type: domain
    validators:
      - csrd_esrs_compliance_check
      - cbam_schema_validation
      - double_materiality_check

  - type: calculation
    validators:
      - emissions_consistency_check
      - cbam_cn_code_validation
```

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| CSRD gap analysis time | <2 hours | Time to identify data gaps |
| CBAM report generation | <1 hour | For 1,000 import records |
| Data consistency | 100% | Emissions match across CSRD and CBAM |
| Regulatory compliance | 100% | Passes EU schema validation |
| User task completion | >90% | % of workflows completed without help |

### Acceptance Criteria

**Given** a company profile with emissions and import data
**When** the agent processes the inputs for CSRD + CBAM compliance
**Then**:
- [ ] Generates CSRD materiality assessment per ESRS standards
- [ ] Identifies data gaps with collection recommendations
- [ ] Calculates embedded emissions for all CBAM imports
- [ ] Generates CBAM quarterly declaration JSON
- [ ] Creates unified compliance calendar with deadlines
- [ ] Flags inconsistencies between CSRD and CBAM data

### Technical Complexity Assessment

| Dimension | Complexity | Notes |
|-----------|------------|-------|
| Input parsing | Medium | Structured company and import data |
| Reasoning steps | Medium | Rule-based with some optimization |
| Data dependencies | High | ESRS standards, CBAM emission factors |
| Output generation | High | Multiple report formats (CSRD, CBAM) |
| Validation | High | EU schema compliance, cross-report consistency |
| **Overall** | **Medium-High** | Good coverage of factory capabilities |

### Timeline

- **Week 1-2 (Phase 0):** Finalize agent spec, map ESRS requirements
- **Week 3-8 (Phase 1):** Build SDK foundations, migrate CBAM calculator
- **Week 9-18 (Phase 2):** Generate agent from spec, run evaluation
- **Week 19-22 (Phase 2):** Certification, golden tests, beta deployment

---

## Use Case 3: Supply Chain Emissions Mapper (Candidate)

### Overview

| Attribute | Value |
|-----------|-------|
| **Agent Name** | GL-SCOPE3-MAPPER |
| **Target User** | Procurement teams, sustainability teams |
| **Regulatory Driver** | CSRD Scope 3, CBAM supplier data |
| **Complexity** | High (supplier engagement, data collection, calculation) |
| **Priority** | P1 - Third agent candidate |

### Problem Statement

Scope 3 emissions (supply chain) account for 70-90% of most companies' carbon footprint but are the hardest to measure. Companies need to:
- Engage suppliers for primary data
- Estimate emissions when primary data unavailable
- Track data quality and improvement over time

**Current State:** Spreadsheet-based, manual supplier outreach, inconsistent methodologies.

**Target State:** Automated supplier engagement and emissions mapping with continuous improvement.

### Agent Specification (Preview)

```yaml
agent_id: gl-scope3-mapper-v1
name: Supply Chain Emissions Mapper
version: "1.0.0"
type: data-collection-orchestrator

inputs:
  - name: supplier_list
    type: array
    items:
      supplier_name: string
      spend_eur: number
      category: string
      country: string

  - name: procurement_data
    type: array
    items:
      product: string
      quantity: number
      unit: string
      supplier: string

outputs:
  - name: scope3_emissions
    type: object
    schema:
      total_tco2e: number
      by_category: object  # GHG Protocol categories
      by_supplier: array
      data_quality_score: number

  - name: supplier_scorecards
    type: array
    items:
      supplier: string
      emissions_tco2e: number
      data_quality: string
      improvement_recommendations: array

  - name: data_collection_tasks
    type: array
    items:
      supplier: string
      data_needed: string
      due_date: date
      status: string

tools:
  - supplier_engagement_portal
  - spend_based_estimator
  - ghg_protocol_calculator
  - data_quality_scorer

validation:
  - type: domain
    validators:
      - ghg_protocol_compliance_check
      - data_quality_threshold_check
```

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Supplier response rate | >60% | % responding to data requests |
| Primary data coverage | >40% | % of emissions from primary data |
| Calculation accuracy | >90% | Compared to manual calculation |
| Time to map 100 suppliers | <1 week | End-to-end processing |

### Status

**Decision Required:** Confirm as third use case or select alternative.

**Alternatives Considered:**
- Regulatory Change Monitor (tracks regulation updates)
- Carbon Footprint Calculator (product-level LCA)
- Climate Risk Assessor (physical and transition risks)

---

## Use Case Selection Matrix

| Use Case | Business Impact | Regulatory Urgency | Technical Feasibility | Domain Coverage | Total Score |
|----------|-----------------|-------------------|----------------------|-----------------|-------------|
| Decarbonization Roadmap | 5 | 4 | 4 | 5 | **18** |
| CSRD + CBAM Copilot | 5 | 5 | 4 | 4 | **18** |
| Scope 3 Mapper | 4 | 4 | 3 | 4 | **15** |
| Regulatory Monitor | 3 | 5 | 5 | 2 | **15** |
| Carbon Footprint Calc | 4 | 3 | 4 | 3 | **14** |

**Scoring:** 5 = Highest, 1 = Lowest

### Selection Rationale

**Selected for Phase 2:**
1. **Decarbonization Roadmap Engineer** - Highest complexity, proves full factory capability
2. **CSRD + CBAM Planning Copilot** - Highest urgency, strong customer demand

**Deferred to Phase 3:**
3. **Supply Chain Emissions Mapper** - Important but requires supplier portal infrastructure

---

## Resource Requirements by Use Case

### Use Case 1: Decarbonization Roadmap Engineer

| Resource | Quantity | Weeks | Notes |
|----------|----------|-------|-------|
| AI/Agent Engineer | 2 | 8 | Agent development |
| Climate Scientist | 1 | 6 | Domain validation |
| Data Engineer | 1 | 4 | Technology database |
| ML Engineer | 1 | 4 | Optimization components |
| **Total** | 5 | - | 22 person-weeks |

### Use Case 2: CSRD + CBAM Copilot

| Resource | Quantity | Weeks | Notes |
|----------|----------|-------|-------|
| AI/Agent Engineer | 2 | 6 | Agent development |
| Climate Scientist | 1 | 4 | ESRS mapping |
| Policy Analyst | 1 | 4 | CBAM requirements |
| Data Engineer | 1 | 3 | Data pipelines |
| **Total** | 5 | - | 17 person-weeks |

---

## Dependencies and Risks

### External Dependencies

| Dependency | Use Case | Risk | Mitigation |
|------------|----------|------|------------|
| Technology cost database | UC1 | Data may be outdated | Partner with tech vendors |
| ESRS final standards | UC2 | Standards still evolving | Build for flexibility |
| CBAM emission factors | UC2 | EU publishes quarterly | Automated factor updates |
| Supplier data | UC3 | Low response rates | Build estimation fallbacks |

### Use Case Specific Risks

| Risk | Likelihood | Impact | Owner | Mitigation |
|------|------------|--------|-------|------------|
| Roadmap recommendations too generic | Medium | High | Climate Science | Industry-specific templates |
| CSRD/CBAM requirements change | Medium | Medium | Policy Team | Modular design, monitoring |
| Calculation accuracy disputed | Low | High | ML Platform | Audit trail, provenance |

---

## Validation and Certification Plan

### Per-Use-Case Certification

| Use Case | Golden Tests | Expert Reviews | Beta Customers |
|----------|-------------|----------------|----------------|
| Decarbonization Roadmap | 50 tests | 3 climate experts | 5 industrial sites |
| CSRD + CBAM Copilot | 75 tests | 2 auditors + 1 policy expert | 10 companies |
| Scope 3 Mapper | 40 tests | 2 GHG Protocol experts | 8 companies |

### Certification Criteria

**All use cases must pass:**
- [ ] 100% golden test pass rate
- [ ] Expert review sign-off (minimum 2 of 3)
- [ ] Beta customer feedback score >7/10
- [ ] Zero critical bugs in production for 2 weeks
- [ ] Regulatory schema validation (where applicable)

---

## Appendices

### Appendix A: User Personas

**Persona 1: Sustainability Manager (Industrial)**
- **Name:** Maria, Sustainability Manager at SteelCorp
- **Goals:** Create decarbonization roadmap, secure board approval for investments
- **Pain Points:** Consultant costs, long timelines, inconsistent methodologies
- **Use Cases:** Decarbonization Roadmap, Scope 3 Mapper

**Persona 2: Compliance Officer (Importer)**
- **Name:** Johan, Head of Compliance at EuroTrade GmbH
- **Goals:** Meet CBAM deadlines, integrate with CSRD reporting
- **Pain Points:** Manual data collection, spreadsheet errors, regulatory complexity
- **Use Cases:** CSRD + CBAM Copilot

**Persona 3: CFO (Mid-Size Company)**
- **Name:** Elena, CFO at GreenManufacturing
- **Goals:** Understand climate investment ROI, comply with CSRD disclosure
- **Pain Points:** Translating sustainability to financial terms
- **Use Cases:** Decarbonization Roadmap, CSRD + CBAM Copilot

### Appendix B: Competitive Landscape

| Competitor | Strengths | Gaps | GreenLang Advantage |
|------------|-----------|------|---------------------|
| Big 4 Consultants | Deep expertise | Expensive, slow | 20x faster, 10x cheaper |
| Carbon Accounting Platforms | Data collection | No planning/roadmap | End-to-end solution |
| Generic AI Assistants | Flexible | No domain validation | Zero-hallucination, certified |

### Appendix C: Data Sources Required

**Use Case 1: Decarbonization Roadmap**
- IEA technology cost projections
- IPCC emission factors
- Industry-specific benchmarks (WSA, IAI, etc.)
- National grid emission factors

**Use Case 2: CSRD + CBAM Copilot**
- ESRS disclosure requirements
- CBAM emission factors (EU published)
- Country-specific carbon prices
- CN code mappings

**Use Case 3: Scope 3 Mapper**
- GHG Protocol guidance
- Spend-based emission factors (Exiobase, EEIO)
- Supplier databases

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL-ProductManager | Initial use cases definition |

---

**Approvals:**

- Product Manager: ___________________
- Climate Science Lead: ___________________
- Engineering Lead: ___________________
