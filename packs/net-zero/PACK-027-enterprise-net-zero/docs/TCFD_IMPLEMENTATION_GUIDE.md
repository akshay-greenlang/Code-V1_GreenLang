# PACK-027 Enterprise Net Zero Pack -- TCFD Implementation Guide

**Pack ID:** PACK-027-enterprise-net-zero
**Version:** 1.0.0
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [Introduction](#introduction)
2. [TCFD Framework Overview](#tcfd-framework-overview)
3. [Transition to ISSB S2](#transition-to-issb-s2)
4. [Pillar 1: Governance](#pillar-1-governance)
5. [Pillar 2: Strategy](#pillar-2-strategy)
6. [Pillar 3: Risk Management](#pillar-3-risk-management)
7. [Pillar 4: Metrics and Targets](#pillar-4-metrics-and-targets)
8. [Scenario Analysis Methodology](#scenario-analysis-methodology)
9. [Financial Impact Quantification](#financial-impact-quantification)
10. [PACK-027 TCFD Template](#pack-027-tcfd-template)
11. [Regulatory Cross-Walk](#regulatory-cross-walk)
12. [Implementation Timeline](#implementation-timeline)

---

## Introduction

This guide helps enterprise users implement the Task Force on Climate-related Financial Disclosures (TCFD) recommendations using PACK-027. While the TCFD was formally disbanded in October 2023 with responsibilities transferred to the ISSB (IFRS S2), the TCFD framework remains the foundation for climate disclosure worldwide and is explicitly referenced by:

- **ISSB S2** (IFRS Sustainability Disclosure Standard S2) -- direct successor
- **CSRD / ESRS E1** -- EU climate disclosure standard
- **SEC Climate Disclosure Rule** -- US regulatory requirement
- **California SB 261** -- California climate financial risk reporting
- **CDP Climate Change** -- Global voluntary disclosure

PACK-027 generates TCFD/ISSB S2 compliant disclosures across all four pillars using enterprise data from engines, workflows, and templates.

---

## TCFD Framework Overview

### The Four Pillars

```
+-------------------------------------------------------------------+
|                    TCFD / ISSB S2 FRAMEWORK                        |
+-------------------------------------------------------------------+
|                                                                   |
|  GOVERNANCE              STRATEGY                                 |
|  +------------------+   +------------------+                     |
|  | Board oversight  |   | Climate risks    |                     |
|  | of climate       |   | and opportunities|                     |
|  |                  |   |                  |                     |
|  | Management role  |   | Scenario analysis|                     |
|  | in climate       |   | (1.5C/2C/4C)     |                     |
|  +------------------+   |                  |                     |
|                         | Financial impact |                     |
|                         +------------------+                     |
|                                                                   |
|  RISK MANAGEMENT         METRICS & TARGETS                        |
|  +------------------+   +------------------+                     |
|  | Process for      |   | Scope 1+2+3      |                     |
|  | identifying      |   | emissions        |                     |
|  | climate risks    |   |                  |                     |
|  |                  |   | Intensity metrics|                     |
|  | Integration into |   |                  |                     |
|  | enterprise risk  |   | Targets and      |                     |
|  | management       |   | progress         |                     |
|  +------------------+   +------------------+                     |
|                                                                   |
+-------------------------------------------------------------------+
```

### 11 TCFD Recommended Disclosures

| # | Pillar | Disclosure | PACK-027 Support |
|---|--------|-----------|-----------------|
| G-a | Governance | Board oversight of climate-related risks and opportunities | Template: governance section |
| G-b | Governance | Management's role in assessing and managing climate-related risks | Template: management section |
| S-a | Strategy | Climate-related risks and opportunities identified (short/medium/long term) | Scenario engine + climate hazard connector |
| S-b | Strategy | Impact on strategy, financial planning, and business model | Carbon pricing engine + financial integration |
| S-c | Strategy | Resilience under different scenarios (including 2C or lower) | Scenario modeling engine (1.5C/2C/BAU) |
| RM-a | Risk Management | Process for identifying and assessing climate-related risks | Template: risk process section |
| RM-b | Risk Management | Process for managing climate-related risks | Template: risk management section |
| RM-c | Risk Management | Integration into overall risk management | Template: ERM integration section |
| MT-a | Metrics & Targets | Metrics used to assess climate-related risks and opportunities | Enterprise baseline engine (Scope 1/2/3) |
| MT-b | Metrics & Targets | Scope 1, Scope 2, and Scope 3 greenhouse gas emissions | All 30 MRV agents |
| MT-c | Metrics & Targets | Targets used and performance against targets | SBTi target engine |

---

## Transition to ISSB S2

### TCFD to ISSB S2 Mapping

| TCFD Recommendation | ISSB S2 Paragraph | Key Differences |
|--------------------|-------------------|-----------------|
| G-a Board oversight | S2.6 | S2 requires description of governance body skills and competencies |
| G-b Management role | S2.7 | S2 requires description of how governance body is informed |
| S-a Risks/opportunities | S2.10-12 | S2 requires classification by value chain position |
| S-b Impact on strategy | S2.13-14 | S2 requires transition plan details |
| S-c Scenario resilience | S2.22 | S2 requires quantitative scenario analysis |
| RM-a Identification | S2.25 | S2 integrates into general IFRS S1 risk management |
| RM-b Management | S2.25 | Same as above |
| RM-c ERM integration | S2.25 | Same as above |
| MT-a Metrics | S2.29 | S2 specifies cross-industry and industry-specific metrics |
| MT-b Emissions | S2.29(a) | S2 mandates Scope 3 (TCFD encouraged) |
| MT-c Targets | S2.33 | S2 requires quantitative interim and long-term targets |

PACK-027 TCFD template outputs are compatible with both TCFD and ISSB S2 requirements.

---

## Pillar 1: Governance

### G-a: Board Oversight

**Required disclosure:** Describe the board's oversight of climate-related risks and opportunities.

**PACK-027 template content:**

```python
from templates.tcfd_report import TCFDReport

tcfd = TCFDReport()

governance = tcfd.populate_governance(
    config=config,
    board_structure={
        "climate_committee": "ESG and Sustainability Committee",
        "committee_chair": "Independent Non-Executive Director",
        "meeting_frequency": "Quarterly",
        "climate_agenda_items": [
            "Review of GHG emissions performance vs. SBTi pathway",
            "Approval of annual GHG inventory",
            "Review of climate scenario analysis",
            "Approval of SBTi target submissions",
            "Review of climate risk assessments",
            "Oversight of carbon pricing impact on capital allocation",
        ],
        "climate_training": "Annual climate literacy briefing for full board",
    },
)
```

**Best practice elements (for A-list CDP and ISSB S2):**

| Element | Description | PACK-027 Template Section |
|---------|-------------|--------------------------|
| Designated committee | Named board committee with climate mandate | Board structure section |
| Meeting frequency | At least quarterly climate review | Meeting schedule |
| Agenda items | Specific climate items reviewed by board | Agenda listing |
| Competency | Climate skills assessment for board members | Skills matrix |
| Decision authority | Board sign-off on targets, disclosures, strategy | Authority matrix |
| Information flow | How management reports to board on climate | Reporting cadence |

### G-b: Management's Role

**Required disclosure:** Describe management's role in assessing and managing climate-related risks and opportunities.

| Management Level | Climate Responsibility | PACK-027 Support |
|-----------------|----------------------|-----------------|
| CEO | Overall climate strategy accountability | Governance template |
| CSO / VP Sustainability | Day-to-day climate program leadership | Config persona |
| CFO | Climate-financial integration, carbon pricing, ESRS E1-8/E1-9 | Financial integration engine |
| COO | Operational emission reduction, energy, fleet, facilities | Baseline engine |
| CRO | Climate risk integration into ERM | Scenario engine |
| Supply Chain Director | Scope 3 reduction, supplier engagement | Supply chain engine |

---

## Pillar 2: Strategy

### S-a: Climate-Related Risks and Opportunities

**PACK-027 populates physical and transition risks from scenario modeling and climate hazard data:**

#### Physical Risks

| Risk Type | Time Horizon | PACK-027 Source | Financial Impact Method |
|-----------|-------------|-----------------|----------------------|
| Extreme heat | Short-term | DATA-020 Climate Hazard Connector | Productivity loss, cooling cost increase |
| Flooding | Medium-term | DATA-020 Climate Hazard Connector | Asset damage, business interruption |
| Water stress | Medium-term | DATA-020 Climate Hazard Connector | Operational disruption, supply chain impact |
| Sea level rise | Long-term | DATA-020 Climate Hazard Connector | Asset relocation, stranded coastal assets |
| Wildfire | Short-medium | DATA-020 Climate Hazard Connector | Supply chain disruption, insurance cost |
| Drought | Medium-long | DATA-020 Climate Hazard Connector | Agricultural supply chain impact |

#### Transition Risks

| Risk Type | Time Horizon | PACK-027 Source | Financial Impact Method |
|-----------|-------------|-----------------|----------------------|
| Carbon pricing (ETS/tax) | Short-term | Carbon pricing engine | Direct cost at carbon price trajectory |
| CBAM exposure | Short-term | Carbon pricing engine | Import certificate cost |
| Stranded assets | Medium-term | Scenario modeling engine | Asset write-down under 1.5C scenario |
| Technology obsolescence | Medium-term | Scenario modeling engine | CapEx requirements for transition |
| Market shift | Medium-long | Scenario modeling engine | Revenue impact from demand changes |
| Reputation | Short-term | N/A | Manual assessment |
| Litigation | Medium-term | N/A | Manual assessment |

#### Opportunities

| Opportunity | PACK-027 Source | Financial Impact |
|-------------|----------------|-----------------|
| Renewable energy cost savings | Scenario engine | LCOE reduction trajectory |
| Energy efficiency | DECARB-X agents | CapEx/OpEx savings |
| New low-carbon products | Scope 4 avoided emissions engine | Revenue from product substitution |
| Green finance | Financial integration engine | Lower cost of capital, green bond issuance |
| Resource efficiency | DECARB-X agents | Material and waste cost savings |

### S-b: Impact on Strategy and Financial Planning

```python
strategy = tcfd.populate_strategy(
    scenario_result=scenario_result,
    carbon_pricing_result=carbon_pricing_result,
    financial_result=financial_result,
    avoided_emissions_result=avoided_emissions_result,
)

# Auto-populated sections:
# 1. How climate risks/opportunities influence corporate strategy
# 2. Capital allocation decisions affected by carbon pricing
# 3. Revenue at risk from transition (high-carbon products)
# 4. Revenue opportunity from low-carbon products
# 5. CapEx plan for transition (electrification, renewables, efficiency)
# 6. Financial planning horizon alignment with climate scenarios
```

### S-c: Resilience Under Different Scenarios

**PACK-027 generates scenario comparison directly from the scenario modeling engine:**

```python
from engines.scenario_modeling_engine import ScenarioModelingEngine

engine = ScenarioModelingEngine(config=config)
scenarios = engine.run_scenarios(
    baseline=baseline,
    scenarios=["1.5C", "2C", "BAU"],
    monte_carlo_runs=10_000,
)

# TCFD scenario disclosure content
resilience = tcfd.populate_resilience(
    scenarios=scenarios,
    carbon_pricing=carbon_pricing_result,
)

# Produces:
# - Emission trajectory fan charts (P10-P90) for each scenario
# - Investment requirements comparison
# - Stranded asset risk assessment
# - Probability of target achievement
# - Carbon budget consumption rate
# - Key sensitivity drivers (tornado chart)
```

| Scenario | Description | Key Assumptions | PACK-027 Parameters |
|----------|-------------|-----------------|---------------------|
| 1.5C (Aggressive) | Paris-aligned, rapid transition | 100% RE by 2035, high carbon price ($150+), aggressive electrification | `scenario_type="1.5C"` |
| 2C (Moderate) | Orderly transition | 80% RE by 2035, medium carbon price ($75-100), steady transition | `scenario_type="2C"` |
| BAU (Conservative) | Current policies only | No additional policy, low carbon price ($25-50), slow transition | `scenario_type="BAU"` |

---

## Pillar 3: Risk Management

### RM-a: Process for Identifying Climate Risks

**Template guidance for describing the enterprise risk identification process:**

| Element | Description | PACK-027 Support |
|---------|-------------|-----------------|
| Risk identification | Annual climate risk assessment | Scenario engine identifies top risks |
| Risk categorization | Physical vs. transition, acute vs. chronic | Template categorization framework |
| Time horizons | Short (<3yr), medium (3-10yr), long (>10yr) | Scenario engine time horizons |
| Materiality | Financial materiality threshold | Financial integration engine |
| Frequency | Annual full assessment, quarterly monitoring | Annual inventory + quarterly dashboard |

### RM-b: Process for Managing Climate Risks

| Management Process | PACK-027 Implementation |
|-------------------|------------------------|
| Risk mitigation | DECARB-X agents (500+ abatement options) |
| Risk monitoring | Data quality guardian (continuous monitoring) |
| Risk reporting | Board climate report (quarterly) |
| Risk response | Scenario-specific action plans |
| Risk appetite | Internal carbon price as risk proxy |

### RM-c: Integration into Enterprise Risk Management

```python
risk_management = tcfd.populate_risk_management(
    config=config,
    scenario_result=scenario_result,
)

# Template sections:
# 1. Climate risk register (mapped to enterprise risk register)
# 2. Risk appetite statement (carbon price as quantitative threshold)
# 3. ERM integration (climate as tier-1 enterprise risk)
# 4. Three lines model for climate data governance
# 5. Control framework for climate data quality
```

---

## Pillar 4: Metrics and Targets

### MT-a: Metrics

**PACK-027 provides all TCFD-recommended metrics automatically:**

| Metric Category | Specific Metric | PACK-027 Source |
|----------------|-----------------|-----------------|
| GHG emissions | Scope 1 total (tCO2e) | Enterprise baseline engine |
| GHG emissions | Scope 2 location-based (tCO2e) | MRV-009 |
| GHG emissions | Scope 2 market-based (tCO2e) | MRV-010 |
| GHG emissions | Scope 3 by category (tCO2e) | MRV-014 to MRV-028 |
| GHG emissions | Total (tCO2e) | Consolidated result |
| Intensity | tCO2e / $M revenue | Baseline / financial data |
| Intensity | tCO2e / FTE | Baseline / employee data |
| Energy | Total energy consumption (MWh) | Baseline energy data |
| Energy | Renewable energy share (%) | Market-based Scope 2 |
| Carbon pricing | Internal carbon price ($/tCO2e) | Carbon pricing engine |
| Carbon pricing | CBAM exposure ($) | Carbon pricing engine |
| Financial | Carbon-adjusted EBITDA | Financial integration engine |
| Financial | EU Taxonomy CapEx alignment (%) | Financial integration engine |
| Water | Water consumption (if material) | DATA-020 (if configured) |
| Supply chain | Suppliers with SBTi targets (%) | Supply chain engine |
| Supply chain | Scope 3 engagement coverage (%) | Supply chain engine |

### MT-b: Scope 1, 2, and 3 GHG Emissions

```python
metrics = tcfd.populate_metrics(
    baseline=baseline,
    trend_data=trend_data,  # Multi-year
)

# Produces comprehensive emission disclosure:
# - Scope 1 by gas (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
# - Scope 1 by source (stationary, mobile, process, fugitive)
# - Scope 2 location-based and market-based (dual reporting)
# - Scope 3 all 15 categories with methodology and DQ score
# - Total emissions with year-over-year trend
# - Intensity metrics (revenue, FTE, production-specific)
# - Base year comparison
# - Data quality assessment
```

### MT-c: Targets and Performance

```python
targets = tcfd.populate_targets(
    sbti_targets=sbti_target_result,
    annual_progress=annual_result,
)

# Produces target disclosure:
# - SBTi near-term target (absolute and/or intensity)
# - SBTi long-term target (net-zero by 2050)
# - Annual milestone pathway
# - Current year vs. pathway position
# - On-track / off-track assessment
# - Key reduction initiatives and their impact
# - Five-year interim milestones
```

---

## Scenario Analysis Methodology

### PACK-027 Scenario Framework

```
Input Parameters (per scenario):
    - Carbon price trajectory (2025-2050)
    - Grid decarbonization rate (by country)
    - Technology adoption curves (EVs, heat pumps, CCS)
    - Policy stringency (current / enhanced / accelerated)
    - Physical climate impacts (RCP 2.6 / 4.5 / 8.5)
    - Energy prices (fossil fuel, renewable)
    - Supplier engagement effectiveness
        |
        v
Monte Carlo Simulation (10,000 runs):
    - Latin Hypercube Sampling for parameter space exploration
    - Annual emissions computed for each year 2025-2050
    - Financial impacts computed for each year
        |
        v
Output Analytics:
    - Fan charts (P10, P25, P50, P75, P90 for each year)
    - Tornado chart (top 10 sensitivity drivers)
    - Sobol indices (first-order and total-order)
    - Probability of target achievement
    - Investment requirements (CapEx P50, P90)
    - Stranded asset exposure
    - Carbon budget consumption trajectory
```

### Sensitivity Analysis

The scenario engine identifies the top 10 drivers of uncertainty:

| # | Parameter | Typical Sobol Index | Interpretation |
|---|-----------|-------------------|---------------|
| 1 | Carbon price trajectory | 0.25-0.35 | Largest single driver for most enterprises |
| 2 | Grid decarbonization rate | 0.15-0.25 | Critical for Scope 2 reduction |
| 3 | Supplier engagement rate | 0.10-0.20 | Critical for Scope 3 reduction |
| 4 | Technology adoption (EVs) | 0.08-0.15 | Fleet electrification timeline |
| 5 | Energy efficiency rate | 0.05-0.12 | Compound effect over time |
| 6 | Renewable energy cost | 0.05-0.10 | PPA economics |
| 7 | Regulatory stringency | 0.05-0.10 | Policy acceleration risk |
| 8 | Physical climate impact | 0.03-0.08 | Operational disruption |
| 9 | Technology adoption (heat pumps) | 0.03-0.08 | Building decarbonization |
| 10 | Scope 3 DQ improvement | 0.02-0.05 | Data quality effect on totals |

---

## Financial Impact Quantification

### ESRS E1-9: Anticipated Financial Effects

PACK-027's financial integration engine quantifies climate financial impacts for ESRS E1-9 disclosure:

| Impact Category | Sub-Category | Quantification Method | PACK-027 Engine |
|----------------|-------------|----------------------|-----------------|
| Physical risks (acute) | Extreme weather damage | Asset exposure x probability x severity | Climate hazard + scenarios |
| Physical risks (chronic) | Temperature increase costs | Productivity loss model, cooling cost increase | Scenario engine |
| Transition risks (policy) | Carbon pricing cost | Emissions x carbon price trajectory | Carbon pricing engine |
| Transition risks (policy) | CBAM exposure | Imported goods emissions x CBAM certificate price | Carbon pricing engine |
| Transition risks (technology) | Stranded assets | Asset-level write-down under 1.5C scenario | Scenario engine |
| Transition risks (market) | Revenue from high-carbon products | Revenue at risk under accelerated transition | Scenario engine |
| Climate opportunities | Renewable energy savings | LCOE reduction x energy consumption | Scenario engine |
| Climate opportunities | Low-carbon product revenue | Product substitution x market growth | Avoided emissions engine |
| Climate opportunities | Green finance | Lower cost of capital (green bond spread) | Financial integration engine |

```python
financial_impact = tcfd.populate_financial_impact(
    carbon_pricing=carbon_pricing_result,
    scenarios=scenario_result,
    financial=financial_result,
)

# Produces ESRS E1-9 aligned financial impact disclosure:
# - Physical risk exposure by geography and type
# - Transition risk exposure by category
# - Net financial impact under each scenario
# - Time horizon (current year, 5-year, 10-year, 2050)
# - Confidence level (P10/P50/P90 range)
```

---

## PACK-027 TCFD Template

### Generating the TCFD Report

```python
from templates.tcfd_report import TCFDReport

tcfd = TCFDReport()

report = tcfd.render(
    config=config,
    baseline=baseline,
    targets=sbti_targets,
    scenarios=scenario_result,
    carbon_pricing=carbon_pricing_result,
    supply_chain=supply_chain_result,
    financial=financial_result,
    assurance=assurance_result,
    format="pdf",
)

# Report structure:
# 1. Executive Summary (1 page)
# 2. Governance (2-3 pages)
#    - G-a: Board oversight
#    - G-b: Management role
# 3. Strategy (5-8 pages)
#    - S-a: Risks and opportunities
#    - S-b: Impact on strategy
#    - S-c: Scenario resilience
# 4. Risk Management (2-3 pages)
#    - RM-a: Identification process
#    - RM-b: Management process
#    - RM-c: ERM integration
# 5. Metrics and Targets (5-8 pages)
#    - MT-a: Climate metrics
#    - MT-b: GHG emissions (Scope 1+2+3)
#    - MT-c: Targets and progress
# 6. Appendix: Data tables, methodology notes
#
# Total: 15-25 pages
```

### Output Formats

| Format | Use Case | Size |
|--------|----------|------|
| PDF | Annual report inclusion, board papers | 15-25 pages |
| HTML | Website publication, interactive charts | Web page |
| JSON | Machine-readable, regulatory submission | Structured data |
| MD | Internal review, markdown viewers | Text document |

---

## Regulatory Cross-Walk

### TCFD to Multi-Framework Mapping

| TCFD Recommendation | ISSB S2 | CSRD ESRS E1 | SEC Climate | CDP | CA SB 261 |
|--------------------|---------|-------------|------------|-----|----------|
| G-a Board oversight | S2.6 | E1-1 (GOV-1) | 1502(a) | C1.1 | Required |
| G-b Management role | S2.7 | E1-1 (GOV-1) | 1502(b) | C1.1b | Required |
| S-a Risks/opps | S2.10-12 | E1-9 | 1502(c) | C2.1 | Required |
| S-b Strategic impact | S2.13-14 | E1-1, E1-9 | 1502(d) | C3.1 | Required |
| S-c Scenario resilience | S2.22 | E1-9 | 1502(d) | C3.2 | Required |
| RM-a Identification | S2.25 | E1-1 | 1503(a) | C2.2 | Required |
| RM-b Management | S2.25 | E1-1 | 1503(b) | C2.2 | Required |
| RM-c ERM integration | S2.25 | E1-1 | 1503(c) | C2.2 | Required |
| MT-a Metrics | S2.29 | E1-5, E1-6 | 1504 | C9 | N/A |
| MT-b Emissions | S2.29(a) | E1-6 | 1504(b) | C6 | N/A |
| MT-c Targets | S2.33 | E1-4 | 1504(c) | C4 | N/A |

PACK-027 generates a single TCFD report that satisfies requirements across all frameworks.

---

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-4)

| Activity | PACK-027 Component | Output |
|----------|-------------------|--------|
| Complete GHG inventory | Enterprise baseline engine | Scope 1+2+3 emissions |
| Run scenario analysis | Scenario modeling engine | 1.5C/2C/BAU fan charts |
| Set SBTi targets | SBTi target engine | Near-term + long-term targets |

### Phase 2: Financial Integration (Weeks 5-8)

| Activity | PACK-027 Component | Output |
|----------|-------------------|--------|
| Configure internal carbon price | Carbon pricing engine | Carbon-adjusted P&L, CBAM |
| Run financial impact assessment | Financial integration engine | ESRS E1-9 data |
| Identify physical risks | Climate hazard connector | Risk register by location |

### Phase 3: Report Generation (Weeks 9-12)

| Activity | PACK-027 Component | Output |
|----------|-------------------|--------|
| Generate TCFD report | TCFD report template | 15-25 page PDF |
| Board review and approval | Board climate report template | Board paper |
| Publish disclosure | Regulatory filings template | Website, annual report |

### Phase 4: Ongoing Monitoring (Continuous)

| Activity | Frequency | PACK-027 Component |
|----------|-----------|-------------------|
| Emission tracking | Quarterly | Annual inventory workflow |
| Scenario update | Annual | Scenario analysis workflow |
| Risk assessment refresh | Annual | Climate hazard connector |
| Board reporting | Quarterly | Board climate report template |
| Regulatory filing | Annual | Regulatory filings template |

---

## Appendix: TCFD Disclosure Quality Indicators

| Indicator | Score 1 (Basic) | Score 3 (Good) | Score 5 (Best Practice) |
|-----------|----------------|----------------|----------------------|
| Board oversight | Board informed | Dedicated committee | Climate-competent committee with quantified KPIs |
| Scenario analysis | Qualitative only | 2 scenarios with ranges | 3+ scenarios with Monte Carlo, fan charts, Sobol indices |
| Financial impact | Qualitative only | Estimated ranges | Quantified P10/P50/P90 with time horizons |
| Scope 3 | Partial categories | All material categories | All 15 categories with DQ scores |
| Targets | Self-defined | SBTi-committed | SBTi-validated 1.5C with annual milestones |
| Verification | No verification | Limited assurance (S1+S2) | Reasonable assurance (S1+S2+S3) |
| Internal carbon price | None | Shadow price stated | Price with decision impact quantified |
| Supply chain engagement | Policy stated | Top suppliers engaged | 67%+ of S3 emissions, SBTi cascade |

PACK-027 enables Score 5 (Best Practice) across all indicators.
