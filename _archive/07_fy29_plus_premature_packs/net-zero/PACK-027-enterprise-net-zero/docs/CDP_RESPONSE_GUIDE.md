# PACK-027 Enterprise Net Zero Pack -- CDP Response Guide

**Pack ID:** PACK-027-enterprise-net-zero
**Version:** 1.0.0
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [Introduction](#introduction)
2. [CDP Climate Change Questionnaire Overview](#cdp-climate-change-questionnaire-overview)
3. [CDP Scoring Methodology](#cdp-scoring-methodology)
4. [Module-by-Module Guidance](#module-by-module-guidance)
5. [PACK-027 Data Mapping](#pack-027-data-mapping)
6. [Auto-Population Workflow](#auto-population-workflow)
7. [Score Optimization Strategy](#score-optimization-strategy)
8. [CDP Supply Chain Integration](#cdp-supply-chain-integration)
9. [Year-over-Year Consistency](#year-over-year-consistency)
10. [Common Scoring Pitfalls](#common-scoring-pitfalls)
11. [Timeline and Deadlines](#timeline-and-deadlines)

---

## Introduction

This guide helps enterprise users maximize their CDP Climate Change questionnaire score using PACK-027. The pack auto-populates responses across all CDP modules (C0-C15) from existing GreenLang data, optimizing for A-list scoring.

### What is CDP?

CDP (formerly the Carbon Disclosure Project) runs the global disclosure system that enables companies, cities, states, and regions to measure and manage their environmental impacts. Over 23,000 companies disclosed through CDP in 2024, and CDP scores are used by investors representing over $136 trillion in assets.

### Why CDP Matters for Enterprises

- **Investor expectations**: 740+ financial institutions with $136T in assets use CDP data
- **Regulatory alignment**: CDP data feeds into CSRD, SEC, and other regulatory disclosures
- **SBTi integration**: SBTi uses CDP for target tracking and progress reporting
- **Supply chain requests**: CDP Supply Chain enables customer-driven disclosure
- **Benchmarking**: CDP scores enable peer comparison within industry sectors
- **Reputation**: A-list recognition is a significant reputational signal

### PACK-027 CDP Integration

PACK-027 auto-populates approximately 80% of CDP Climate Change questionnaire responses directly from enterprise data. The remaining 20% requires governance and strategic narrative inputs from the sustainability team.

---

## CDP Climate Change Questionnaire Overview

### Module Structure

| Module | Title | Questions | Weight | PACK-027 Auto-Population |
|--------|-------|-----------|--------|-------------------------|
| C0 | Introduction | 8 | Low | 90% (organization profile from config) |
| C1 | Governance | 4 | High | 40% (structure from config; narrative manual) |
| C2 | Risks and Opportunities | 4 | High | 60% (scenario data from engine; narrative manual) |
| C3 | Business Strategy | 5 | High | 50% (transition plan; scenario from engine) |
| C4 | Targets and Performance | 3 | Very High | 95% (SBTi targets from engine) |
| C5 | Emissions Methodology | 3 | Medium | 100% (methodology from engine config) |
| C6 | Emissions Data | 10 | Very High | 100% (Scope 1/2/3 from baseline engine) |
| C7 | Emissions Breakdown | 9 | High | 95% (breakdowns from baseline engine) |
| C8 | Energy | 2 | Medium | 90% (energy from baseline engine) |
| C9 | Additional Metrics | 2 | Medium | 80% (intensity from baseline engine) |
| C10 | Verification | 3 | High | 90% (assurance from assurance workflow) |
| C11 | Carbon Pricing | 3 | Medium | 100% (from carbon pricing engine) |
| C12 | Engagement | 4 | High | 85% (supplier data from supply chain engine) |
| C13-C14 | Other/Signoff | 2 | Low | 100% (auto-generated) |
| C15 | Biodiversity | 3 | Medium | 30% (partial from FLAG/land use data) |

---

## CDP Scoring Methodology

### Scoring Levels

| Level | Score | Criteria | Enterprise Target |
|-------|-------|----------|------------------|
| **Leadership** | A / A- | Demonstrating best practice; verified emissions; ambitious targets; engagement | **A-list target** |
| **Management** | B / B- | Taking coordinated action; targets set; risk management established | Minimum acceptable |
| **Awareness** | C / C- | Knowledge of impacts; measurement established; policies in place | Below target |
| **Disclosure** | D / D- | Providing information; basic measurement | Significantly below target |
| **Not disclosed** | F | Did not respond or insufficient information | Unacceptable |

### Scoring Criteria by Module

| Module | Disclosure Points | Awareness Points | Management Points | Leadership Points |
|--------|------------------|------------------|-------------------|-------------------|
| C1 Governance | Board oversight described | Climate roles defined | Climate in strategy | Climate-competent board |
| C2 Risks | Risks identified | Process described | Risk integrated into ERM | Scenario analysis done |
| C3 Strategy | Plans described | Low-carbon transition plan | SBTi-aligned targets | 1.5C scenario analysis |
| C4 Targets | Targets listed | Abs + intensity targets | SBTi-validated targets | Progress ahead of pathway |
| C6 Emissions | Scope 1+2 disclosed | Scope 3 disclosed | All categories covered | Verified, year-over-year decrease |
| C10 Verification | None | Plan for verification | Limited assurance | Reasonable assurance (Scope 1+2+3) |
| C11 Carbon Pricing | None | Awareness | Shadow/internal price | Price informs decisions |
| C12 Engagement | None | Policy exists | Active engagement | 67%+ suppliers engaged |

---

## Module-by-Module Guidance

### C0: Introduction

**Auto-populated fields:**
```python
from templates.cdp_climate_response import CDPClimateResponse

cdp = CDPClimateResponse()

c0 = cdp.populate_c0(
    org_name="GlobalMfg Corp",
    reporting_year=2025,
    country="US",
    sector="Manufacturing",
    ticker="GMFG",
    annual_revenue_usd=25_000_000_000,
    employees=85_000,
    # PACK-027 auto-fills: accounting year, consolidation approach, reporting boundary
)
```

| Question | PACK-027 Source | Manual Input Required |
|----------|----------------|----------------------|
| C0.1 Organization name | Config | No |
| C0.2 Reporting year | Config | No |
| C0.3 Countries of operation | Config | No |
| C0.4 Currency | Config | No |
| C0.5 Reporting boundary | Consolidation approach from config | No |
| C0.6 Activities | Manual (business description) | Yes |
| C0.7 ISIN | Manual | Yes |
| C0.8 Subsidiaries included | Entity hierarchy from orchestrator | No |

### C1: Governance

**PACK-027 contribution:** Board structure and roles from configuration. Narrative descriptions require manual input.

| Question | Scoring Target (A-list) | PACK-027 Support |
|----------|------------------------|-----------------|
| C1.1a Board oversight of climate | Board committee with climate responsibility | Template: governance structure section |
| C1.1b Management responsibility | Named officer (CSO or equivalent) | Template: management roles |
| C1.2 Incentivized targets | Climate KPIs in executive compensation | Manual input |
| C1.3 Climate in strategy | Climate integrated into business strategy | Template from scenario analysis |

**Best practice for A-list:**
- Describe specific board committee (e.g., ESG Committee, Sustainability Committee)
- Name the highest-ranking officer with climate responsibility
- Link executive compensation to quantified climate KPIs (e.g., % of bonus linked to emission reduction targets)
- Show how scenario analysis (1.5C/2C) informs strategic decisions

### C2: Risks and Opportunities

**PACK-027 contribution:** Scenario modeling engine provides quantified risk data.

```python
c2 = cdp.populate_c2(
    scenario_result=scenario_result,
    financial_result=financial_result,
)

# Auto-populated:
# - Physical risks by geography (from climate hazard data)
# - Transition risks by category (policy, technology, market, reputation)
# - Financial impact estimates (from carbon pricing engine)
# - Scenario analysis results (from scenario modeling engine)
```

| Question | Scoring Target | PACK-027 Source |
|----------|---------------|----------------|
| C2.1 Climate-related risks identified | Yes, both physical and transition | Scenario engine + climate hazard connector |
| C2.2 Risk management process | Integrated into enterprise risk management | Manual narrative |
| C2.3 Climate opportunities | Quantified opportunities by type | Financial integration engine |
| C2.4 Financial impact | Quantified financial impact of risks/opportunities | Carbon pricing engine |

### C3: Business Strategy

| Question | Scoring Target | PACK-027 Source |
|----------|---------------|----------------|
| C3.1 Transition plan | Published, SBTi-aligned transition plan | SBTi target engine + template |
| C3.2 Scenario analysis | 1.5C, 2C, and 4C scenarios analyzed | Scenario modeling engine |
| C3.3 Financial planning influenced | Climate in CapEx allocation | Carbon pricing engine (carbon-adjusted NPV) |
| C3.4 Low-carbon products | Revenue from low-carbon products/services | Avoided emissions engine |
| C3.5 Internal carbon price | Price level, scope, and how it informs decisions | Carbon pricing engine |

### C4: Targets and Performance

**This is the most critical module for A-list scoring.**

```python
c4 = cdp.populate_c4(
    targets=sbti_targets,
    baseline=baseline,
    current_year=annual_result,
)

# Auto-populated:
# C4.1: All targets (SBTi near-term + long-term + net-zero)
# C4.1a: Absolute emission targets with base year, target year, % reduction
# C4.1b: Intensity targets (if SDA pathway)
# C4.2: Progress against targets (on-track assessment)
# C4.3: SBTi validation status
```

| Question | Scoring Target | PACK-027 Source |
|----------|---------------|----------------|
| C4.1 Emission reduction targets | SBTi-validated absolute targets (1.5C) | SBTi target engine |
| C4.1a Absolute targets detail | Base year, target year, coverage, reduction % | SBTi target result |
| C4.1b Intensity targets detail | SDA pathway intensity convergence | SBTi target engine (SDA) |
| C4.2 Progress against targets | On-track vs. pathway, year-over-year | Annual inventory workflow |
| C4.3 SBTi status | Targets validated | SBTi bridge status |

### C5: Emissions Methodology

```python
c5 = cdp.populate_c5(
    config=config,
    baseline=baseline,
)

# Auto-populated:
# C5.1: GHG Protocol Corporate Standard used
# C5.2: Base year and recalculation policy
# C5.3: Gases included (all 7 Kyoto gases)
```

### C6: Emissions Data

**Fully auto-populated from enterprise baseline engine.**

```python
c6 = cdp.populate_c6(
    baseline=baseline,
)

# C6.1: Scope 1 total (tCO2e) by gas
# C6.2: Scope 1 breakdown by country
# C6.3: Scope 2 (location + market)
# C6.4: Scope 2 methodology explanation
# C6.5: Scope 3 by category (all 15)
# C6.7: Methodology per Scope 3 category
# C6.10: Emissions data for all material gases
```

| Question | Data | PACK-027 Source |
|----------|------|----------------|
| C6.1 Scope 1 gross | Total tCO2e, by gas | MRV-001 to MRV-008 |
| C6.2 Scope 1 by country | Per-country breakdown | Entity-level baseline |
| C6.3 Scope 2 location-based | Grid-average tCO2e | MRV-009 |
| C6.3 Scope 2 market-based | Contractual tCO2e | MRV-010 |
| C6.5 Scope 3 Category 1 | Purchased goods tCO2e | MRV-014 |
| C6.5 Scope 3 Category 2 | Capital goods tCO2e | MRV-015 |
| C6.5 Scope 3 Category 3 | Fuel & energy tCO2e | MRV-016 |
| C6.5 Scope 3 Category 4 | Upstream transport tCO2e | MRV-017 |
| C6.5 Scope 3 Category 5 | Waste tCO2e | MRV-018 |
| C6.5 Scope 3 Category 6 | Business travel tCO2e | MRV-019 |
| C6.5 Scope 3 Category 7 | Employee commuting tCO2e | MRV-020 |
| C6.5 Scope 3 Category 8 | Upstream leased tCO2e | MRV-021 |
| C6.5 Scope 3 Category 9 | Downstream transport tCO2e | MRV-022 |
| C6.5 Scope 3 Category 10 | Processing sold products tCO2e | MRV-023 |
| C6.5 Scope 3 Category 11 | Use of sold products tCO2e | MRV-024 |
| C6.5 Scope 3 Category 12 | End-of-life treatment tCO2e | MRV-025 |
| C6.5 Scope 3 Category 13 | Downstream leased tCO2e | MRV-026 |
| C6.5 Scope 3 Category 14 | Franchises tCO2e | MRV-027 |
| C6.5 Scope 3 Category 15 | Investments tCO2e | MRV-028 |

### C7: Emissions Breakdown

```python
c7 = cdp.populate_c7(
    baseline=baseline,
)

# C7.1: By business division / segment
# C7.2: By country / region
# C7.3: By GHG type (CO2, CH4, N2O, etc.)
# C7.5: By Scope 1 source (stationary, mobile, process, fugitive)
# C7.6: Biogenic emissions
# C7.9: Year-over-year comparison
```

### C8: Energy

```python
c8 = cdp.populate_c8(
    baseline=baseline,
)

# C8.1: Total energy consumption (MWh)
# C8.2: Renewable energy
#   - % electricity from renewables
#   - PPA capacity (if applicable)
#   - REC/GO procured
```

| Question | Data | PACK-027 Source |
|----------|------|----------------|
| C8.1 Energy consumption | Total MWh by fuel type | MRV-001, MRV-009/010 |
| C8.2a Renewable electricity | MWh from RE sources, % of total | MRV-010 (market-based) |
| C8.2b RE100 or equivalent | Target and progress | Config feature flags |

### C9: Additional Metrics

```python
c9 = cdp.populate_c9(
    baseline=baseline,
    financial_data=financial_data,
)

# C9.1: Intensity metrics
#   - tCO2e per $M revenue
#   - tCO2e per FTE
#   - tCO2e per unit of production (sector-specific)
```

### C10: Verification

```python
c10 = cdp.populate_c10(
    assurance_result=assurance_result,
)

# C10.1: Verification status
# C10.1a: Scope 1 verification details
# C10.1b: Scope 2 verification details
# C10.1c: Scope 3 verification details (if verified)
# C10.2: Verification standard (ISO 14064-3 / ISAE 3410)
```

**A-list requirement:** At minimum, limited assurance on Scope 1+2. Reasonable assurance earns additional points.

### C11: Carbon Pricing

```python
c11 = cdp.populate_c11(
    carbon_pricing_result=carbon_pricing_result,
)

# C11.1: Carbon pricing mechanisms affecting operations (ETS, carbon tax)
# C11.2: Projects generating carbon credits
# C11.3: Internal carbon pricing
#   - Type: shadow price / internal fee
#   - Price level: $/tCO2e
#   - Scope of application
#   - Revenue generated (if internal fee)
#   - How it informs capital allocation
```

| Question | Scoring Target | PACK-027 Source |
|----------|---------------|----------------|
| C11.3a Internal price type | Shadow price or internal fee | Carbon pricing engine config |
| C11.3b Price level | $50-200/tCO2e | Carbon pricing engine |
| C11.3c Scope | Scope 1+2 minimum, ideally Scope 3 | Carbon pricing engine config |
| C11.3d Impact on decisions | Quantified: X projects rejected/approved | Investment appraisal results |

### C12: Engagement

```python
c12 = cdp.populate_c12(
    supply_chain_result=supply_chain_result,
)

# C12.1: Engagement with value chain
# C12.1a: Supplier engagement strategy
# C12.1b: Engagement metrics
# C12.3: Policy engagement (trade associations, government)
# C12.4: Collaboration with non-governmental organizations
```

| Question | Scoring Target | PACK-027 Source |
|----------|---------------|----------------|
| C12.1a Suppliers engaged | 67%+ of Scope 3 emissions from engaged suppliers | Supply chain engine |
| C12.1b CDP Supply Chain | Member of CDP Supply Chain program | CDP bridge |
| C12.1c Engagement success | Suppliers with SBTi commitments | Supply chain scorecards |
| C12.3 Policy engagement | Consistent positions across lobbying and targets | Manual input |

---

## PACK-027 Data Mapping

### Complete Data Mapping Table

| CDP Question | CDP Field | PACK-027 Engine | PACK-027 Model Field | Auto |
|-------------|-----------|-----------------|---------------------|------|
| C0.1 | Organization name | Config | `config.organization_name` | Yes |
| C4.1a | Absolute target % | SBTi Target Engine | `near_term.scope12_reduction_pct` | Yes |
| C4.1a | Base year emissions | Enterprise Baseline | `baseline.scope12_base_year_tco2e` | Yes |
| C4.1a | Target year | SBTi Target Engine | `near_term.target_year` | Yes |
| C6.1 | Scope 1 total | Enterprise Baseline | `baseline.scope1_total_tco2e` | Yes |
| C6.3 | Scope 2 location | Enterprise Baseline | `baseline.scope2_location_tco2e` | Yes |
| C6.3 | Scope 2 market | Enterprise Baseline | `baseline.scope2_market_tco2e` | Yes |
| C6.5 | Scope 3 Cat 1-15 | Enterprise Baseline | `baseline.scope3_by_category[n].tco2e` | Yes |
| C8.1 | Total energy MWh | Enterprise Baseline | `baseline.total_energy_mwh` | Yes |
| C8.2a | RE percentage | Enterprise Baseline | `baseline.renewable_energy_pct` | Yes |
| C10.1 | Verification status | Assurance Workflow | `assurance.level` | Yes |
| C11.3 | Internal carbon price | Carbon Pricing Engine | `pricing.price_usd_per_tco2e` | Yes |

---

## Auto-Population Workflow

### Running CDP Auto-Population

```python
from templates.cdp_climate_response import CDPClimateResponse

cdp = CDPClimateResponse()

# Full auto-population (requires all engine results)
response = cdp.auto_populate(
    config=config,
    baseline=baseline_result,
    targets=sbti_target_result,
    scenarios=scenario_result,
    carbon_pricing=carbon_pricing_result,
    supply_chain=supply_chain_result,
    assurance=assurance_result,
    financial=financial_result,
)

# Review auto-populated fields
print(f"Total questions: {response.total_questions}")
print(f"Auto-populated: {response.auto_populated_count}")
print(f"Requires manual input: {response.manual_required_count}")
print(f"Auto-population rate: {response.auto_population_pct:.0f}%")

# Export for review
output = cdp.render(response, format="html")

# List manual input fields
for field in response.manual_fields:
    print(f"  [{field.module}] {field.question}: {field.description}")
```

### Manual Fields Requiring Input

| Module | Question | Why Manual | Guidance |
|--------|----------|-----------|----------|
| C1.1a | Board climate governance narrative | Company-specific governance | Describe board committee, meeting frequency, agenda items |
| C1.2 | Executive incentive details | Compensation structure | Describe % of bonus linked to climate KPIs |
| C2.1a | Risk management narrative | Process description | Describe how climate risks are identified and assessed |
| C2.3 | Opportunity details | Strategic judgment | Describe revenue opportunities from low-carbon transition |
| C3.1 | Transition plan narrative | Strategic document | Reference published transition plan |
| C3.3 | Financial planning influenced | Capital allocation narrative | Describe how carbon price influences CapEx decisions |
| C12.3 | Policy engagement | Trade association alignment | Describe climate positions of trade associations |
| C12.4 | Collaborations | Partnership details | List climate coalitions and initiatives |
| C15.1 | Biodiversity dependencies | Ecological assessment | Describe dependencies on ecosystem services |

---

## Score Optimization Strategy

### A-List Requirements (all must be met)

1. **Disclosure score**: >= 80% of applicable questions answered
2. **Awareness score**: >= 70% of awareness points earned
3. **Management score**: >= 70% of management points earned
4. **Leadership score**: >= 70% of leadership points earned
5. **Verification**: Scope 1+2 verified (limited or reasonable assurance)
6. **Targets**: SBTi-validated or equivalent science-based targets
7. **Year-over-year emission decrease**: Or explanation if increase is structural

### PACK-027 Score Optimization Actions

| Action | Expected Score Impact | PACK-027 Feature |
|--------|----------------------|-----------------|
| Complete all 15 Scope 3 categories | +3-5 points | Enterprise baseline (all MRV agents) |
| SBTi-validated 1.5C targets | +5-8 points | SBTi target engine |
| External verification (limited assurance) | +3-5 points | Assurance workflow |
| Internal carbon pricing | +2-3 points | Carbon pricing engine |
| Quantified scenario analysis (1.5C) | +3-4 points | Scenario modeling engine |
| Supplier engagement (67%+ Scope 3) | +3-5 points | Supply chain engine |
| Transition plan published | +2-3 points | Templates |
| Year-over-year emission reduction | +2-3 points | Annual inventory workflow |
| Renewable energy procurement | +2-3 points | Market-based Scope 2 |
| Board climate competency | +1-2 points | Governance template |

### Estimated Score Range

| Without PACK-027 | Typical Score | With PACK-027 Optimization | Target Score |
|-----------------|---------------|---------------------------|-------------|
| D- to C | 30-50% | B to A | 75-95% |

---

## CDP Supply Chain Integration

### Requesting Supplier Data via CDP

```python
from integrations.cdp_bridge import CDPBridge

cdp_bridge = CDPBridge(config=config)

# Request CDP disclosure from top suppliers
requests = cdp_bridge.send_supply_chain_requests(
    supplier_list=supply_chain_result.tier1_suppliers,
    questionnaire_year=2025,
)

print(f"Requests sent: {requests.sent_count}")
print(f"Suppliers already disclosing: {requests.already_disclosing}")
print(f"First-time responders: {requests.first_time}")
```

### Analyzing Supplier CDP Scores

```python
scores = cdp_bridge.get_supplier_scores(
    supplier_ids=supply_chain_result.all_supplier_ids,
    year=2024,
)

print(f"A-list suppliers: {scores.a_count}")
print(f"B-list suppliers: {scores.b_count}")
print(f"C-list suppliers: {scores.c_count}")
print(f"Non-disclosing suppliers: {scores.non_disclosing_count}")
print(f"Average supplier score: {scores.average_score}")
```

---

## Year-over-Year Consistency

### Ensuring Consistent Responses

PACK-027 maintains year-over-year consistency through:

1. **Base year consistency**: Same base year used across all responses
2. **Methodology consistency**: Same emission factors and calculation methodology
3. **Boundary consistency**: Same organizational boundary and consolidation approach
4. **Restatement tracking**: Any changes to prior year data documented with explanation

```python
# Compare current response with prior year
consistency = cdp.check_consistency(
    current_response=response_2025,
    prior_response=response_2024,
)

for issue in consistency.inconsistencies:
    print(f"  [{issue.severity}] {issue.module}: {issue.description}")
    print(f"    2024 value: {issue.prior_value}")
    print(f"    2025 value: {issue.current_value}")
    print(f"    Explanation needed: {issue.explanation_required}")
```

---

## Common Scoring Pitfalls

| # | Pitfall | Impact | PACK-027 Prevention |
|---|---------|--------|---------------------|
| 1 | Missing Scope 3 categories | -5 to -10 points | All 15 categories auto-calculated |
| 2 | No third-party verification | Cannot reach A-list | Assurance workflow generates workpapers |
| 3 | Targets not SBTi-validated | -5 to -8 points | SBTi submission workflow |
| 4 | No internal carbon price | -2 to -3 points | Carbon pricing engine |
| 5 | No scenario analysis | -3 to -4 points | Scenario modeling engine |
| 6 | Inconsistent year-over-year | -3 to -5 points | Consistency checker |
| 7 | No supplier engagement data | -3 to -5 points | Supply chain engine |
| 8 | Missing energy breakdown | -2 to -3 points | Energy data from baseline |
| 9 | No transition plan reference | -2 to -3 points | Generated in templates |
| 10 | Biogenic emissions omitted | -1 to -2 points | MRV-006/008 include biogenic |

---

## Timeline and Deadlines

### CDP Calendar (Typical)

| Date | Activity | PACK-027 Support |
|------|----------|-----------------|
| January | CDP questionnaire released | Template updated |
| February | Questionnaire opens for response | Auto-populate workflow ready |
| March-April | Response preparation | Run auto-population, fill manual fields |
| July | Submission deadline | Export and submit |
| August-October | CDP scoring and analysis | N/A |
| November-December | Scores published | Compare with target, identify improvements |

### PACK-027 CDP Workflow Timeline

```
Week 1-2: Run enterprise baseline for reporting year
    |
    v
Week 3: Run SBTi progress assessment
    |
    v
Week 4: Auto-populate CDP response (80% complete)
    |
    v
Week 5-6: Fill manual fields (governance, narrative, strategy)
    |
    v
Week 7: Internal review and approval
    |
    v
Week 8: Submit to CDP
    |
    Total elapsed: ~8 weeks from data refresh to submission
```

---

## Appendix: CDP Question Count by Module

| Module | Mandatory Questions | Optional Questions | Total |
|--------|--------------------|--------------------|-------|
| C0 Introduction | 8 | 0 | 8 |
| C1 Governance | 3 | 1 | 4 |
| C2 Risks | 3 | 1 | 4 |
| C3 Strategy | 4 | 1 | 5 |
| C4 Targets | 3 | 0 | 3 |
| C5 Methodology | 3 | 0 | 3 |
| C6 Emissions | 8 | 2 | 10 |
| C7 Breakdown | 7 | 2 | 9 |
| C8 Energy | 2 | 0 | 2 |
| C9 Metrics | 1 | 1 | 2 |
| C10 Verification | 3 | 0 | 3 |
| C11 Carbon Pricing | 3 | 0 | 3 |
| C12 Engagement | 3 | 1 | 4 |
| C13-C14 Other/Signoff | 2 | 0 | 2 |
| C15 Biodiversity | 2 | 1 | 3 |
| **Total** | **55** | **10** | **65** |
