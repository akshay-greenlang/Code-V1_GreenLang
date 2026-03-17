# PRD-PACK-016: ESRS E1 Climate Pack

**Pack ID:** PACK-016-esrs-e1-climate
**Category:** EU Compliance / ESRS Topical
**Tier:** Standalone (Cross-Sector)
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-17

---

## 1. Executive Summary

### 1.1 Problem Statement

ESRS E1 Climate Change is the most complex and data-intensive topical standard in the European Sustainability Reporting Standards (ESRS) framework. It comprises 9 disclosure requirements (E1-1 through E1-9) spanning transition planning, climate policies, actions and resources, targets, energy consumption, GHG emissions across all three scopes, carbon removals and credits, internal carbon pricing, and financial effects of climate-related risks and opportunities. Companies subject to CSRD must disclose against all E1 requirements that are material per their Double Materiality Assessment.

Current challenges facing sustainability teams:

1. **Fragmented data sources**: E1 disclosures require data from energy management systems, GHG inventories, financial planning, risk registers, strategy documents, and target-setting frameworks -- typically scattered across 8-15 internal systems with no unified data model.
2. **Calculation complexity**: E1-6 alone requires Scope 1, 2 (location-based and market-based), and Scope 3 (15 categories) GHG emissions calculated per GHG Protocol methodology with IPCC AR6 GWP values, gas-level disaggregation, and biogenic CO2 separate reporting.
3. **Cross-referencing requirements**: E1 disclosure requirements cross-reference ESRS 2 (general disclosures), EU Taxonomy (climate mitigation/adaptation), GHG Protocol, SBTi, TCFD, and Paris Agreement alignment -- requiring consistent data across all frameworks.
4. **Transition plan rigor**: E1-1 requires a granular transition plan with GHG reduction levers, locked-in emissions, CapEx/OpEx allocation, and decarbonization milestones that must be internally consistent with E1-4 targets and E1-6 emissions data.
5. **Audit readiness**: All E1 disclosures are subject to limited assurance (moving to reasonable assurance), requiring complete calculation provenance, methodology documentation, and reproducible outputs.

### 1.2 Solution Overview

PACK-016 is a **standalone ESRS E1 Climate Change Solution Pack** that provides end-to-end automation for all 9 ESRS E1 disclosure requirements. It implements 8 deterministic calculation engines, 9 disclosure-specific workflows, 9 ESRS-aligned report templates, and 8 integrations that connect to the GreenLang agent ecosystem. Every calculation is zero-hallucination (deterministic lookups and arithmetic only, no LLM in any calculation path), bit-perfect reproducible, and SHA-256 hashed for audit assurance.

The pack transforms raw operational data (energy bills, fuel consumption, fleet records, procurement data, financial plans, risk assessments) into submission-ready ESRS E1 disclosures with XBRL tagging per the EFRAG taxonomy, in a fraction of the time required by manual processes.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-016 ESRS E1 Climate Pack |
|-----------|-------------------------------|-------------------------------|
| Time to complete all 9 DRs | 200-400 hours per reporting cycle | <20 hours (10-20x faster) |
| GHG calculation accuracy | Error-prone (formula drift, copy-paste) | Deterministic, bit-perfect, AR6 GWP |
| Scope 3 coverage | Typically 3-5 categories | All 15 categories via AGENT-MRV |
| Cross-framework consistency | Manual reconciliation | Automated (GHG Protocol, SBTi, Taxonomy) |
| Audit readiness | Manual documentation | SHA-256 provenance, full lineage |
| XBRL tagging | Manual tagging or outsourced | Automated EFRAG XBRL taxonomy |
| Year-over-year tracking | New spreadsheet each year | Delta-based with trend analysis |

### 1.4 Target Users

**Primary:**
- Sustainability managers and climate officers at CSRD-subject companies
- ESG reporting teams preparing ESRS sustainability statements
- Companies with >1,000 employees AND >EUR 450M turnover (Omnibus I threshold)

**Secondary:**
- External auditors conducting limited/reasonable assurance on E1 disclosures
- Sustainability consultants preparing E1 disclosures for clients
- Investor relations teams communicating climate performance
- Board members and executive committees reviewing transition plans

### 1.5 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to complete all 9 E1 DRs | <20 hours (vs. 200-400 manual) | Time from data upload to final output |
| GHG calculation accuracy | 100% match with manual verification | Tested against 500 known emission values |
| Scope 3 category coverage | 15/15 categories | Number of categories with calculations |
| XBRL tag coverage | 100% of E1 datapoints | Mapped vs total EFRAG E1 datapoints |
| Audit finding rate | <2 findings per engagement | External auditor findings on E1 section |
| Customer satisfaction (NPS) | >55 | Net Promoter Score survey |

---

## 2. Regulatory Basis

### 2.1 Primary Regulation

| Regulation | Reference | Effective | E1 Relevance |
|------------|-----------|-----------|--------------|
| CSRD | Directive (EU) 2022/2464 | FY2025+ | Mandates sustainability reporting per ESRS; E1 applies when climate change is material |
| ESRS E1 Climate Change | Delegated Regulation (EU) 2023/2772 | With CSRD | 9 disclosure requirements (E1-1 through E1-9), 80+ datapoints |
| ESRS 1 General Requirements | Delegated Regulation (EU) 2023/2772 | With CSRD | Materiality assessment determining E1 applicability; time horizons; value chain scope |
| ESRS 2 General Disclosures | Delegated Regulation (EU) 2023/2772 | With CSRD | Cross-cutting disclosures referenced by E1 (GOV-3, SBM-3, IRO-1) |
| Omnibus I | Directive (EU) 2026/470 | 2026 | Revised thresholds; 61% datapoint reduction; E1-6 Scope 1/2 remain mandatory for all in-scope |

### 2.2 Supporting Standards and Frameworks

| Standard / Framework | Reference | E1 Relevance |
|---------------------|-----------|--------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2004, 2015 update) | E1-6: Scope 1 and 2 calculation methodology |
| GHG Protocol Scope 2 Guidance | WRI/WBCSD (2015) | E1-6: Location-based and market-based Scope 2 |
| GHG Protocol Corporate Value Chain (Scope 3) | WRI/WBCSD (2011) | E1-6: Scope 3 categories 1-15 methodology |
| IPCC AR6 (2021) | IPCC Sixth Assessment Report | E1-6: GWP values (100-year) for all greenhouse gases |
| SBTi Corporate Net-Zero Standard | SBTi v1.2 (2024) | E1-4: Science-based target validation and progress |
| SBTi FLAG Guidance | SBTi (2022) | E1-4: Targets for land use sectors |
| EU Taxonomy Regulation | Regulation (EU) 2020/852 | E1-1: Climate mitigation/adaptation alignment; E1-9: Taxonomy CapEx |
| EU Taxonomy Climate Delegated Act | Delegated Regulation (EU) 2021/2139 | E1-1: Technical screening criteria for climate mitigation/adaptation |
| Paris Agreement | UNFCCC (2015) | E1-1: 1.5C alignment for transition plans |
| TCFD Recommendations | FSB/TCFD (2017) | E1-9: Physical and transition risk framework |
| ISO 14064-1:2018 | International Organization for Standardization | E1-6: Organization-level GHG quantification |
| EFRAG IG-1 | EFRAG Implementation Guidance (2024) | Materiality assessment for E1 topics |
| EFRAG IG-2 | EFRAG Implementation Guidance (2024) | Value chain boundary for Scope 3 |

### 2.3 ESRS E1 Paragraph Reference Map

| DR | Title | ESRS E1 Paragraphs | Mandatory / Material | Omnibus Status |
|----|-------|--------------------|--------------------|----------------|
| E1-1 | Transition plan for climate change mitigation | para 14-16, AR 1-13 | Material (if E1 material) | Retained |
| E1-2 | Policies related to climate change mitigation and adaptation | para 22-24, AR 14-15 | Material (if E1 material) | Retained |
| E1-3 | Actions and resources in relation to climate change policies | para 26-28, AR 16-19 | Material (if E1 material) | Retained |
| E1-4 | Targets related to climate change mitigation and adaptation | para 30-33, AR 20-29 | Material (if E1 material) | Retained |
| E1-5 | Energy consumption and mix | para 35-39, AR 30-37 | Material (if E1 material) | Retained |
| E1-6 | Gross Scopes 1, 2, 3 and Total GHG emissions | para 44-55, AR 39-63 | Mandatory for all in-scope (Scope 1+2); Scope 3 material | Scope 1+2 mandatory |
| E1-7 | GHG removals and GHG mitigation projects financed through carbon credits | para 56-60, AR 64-68 | Material (if applicable) | Retained |
| E1-8 | Internal carbon pricing | para 62-64, AR 69-72 | Material (if applicable) | Retained |
| E1-9 | Anticipated financial effects from material physical and transition risks | para 66-69, AR 73-79 | Material (if E1 material) | Retained |

---

## 3. ESRS E1 Disclosure Requirements -- Detailed Specifications

### 3.1 E1-1: Transition Plan for Climate Change Mitigation (para 14-16)

**Objective (ESRS E1 para 14):** The undertaking shall disclose its transition plan for climate change mitigation, explaining how it plans to ensure that its business model and strategy are compatible with the transition to a sustainable economy and with the limiting of global warming to 1.5 degrees Celsius in line with the Paris Agreement.

**Required Datapoints:**
- Description of the transition plan, including key assumptions (para 14)
- GHG emission reduction targets (cross-ref E1-4) covered by the plan (para 15)
- Decarbonization levers and key actions planned (para 15(a))
- Locked-in GHG emissions from existing assets and products (para 15(b), AR 8)
- CapEx and OpEx amounts associated with the transition plan (para 15(c))
- Compatibility assessment with 1.5C / well-below 2C pathways (para 16, AR 1)
- Explanation of how the plan is embedded in the overall business strategy (para 16(a))
- Progress against prior-year plan milestones (para 16(b))
- If no transition plan exists: explanation and timeline for adoption (AR 2)

**Application Requirements (AR 1-13):**
- AR 1: Reference to recognized climate scenarios (IEA NZE, NGFS, IPCC SSP1-1.9)
- AR 3-4: Levers to achieve targets (energy efficiency, fuel switching, electrification, renewables, phase-out of fossil fuels)
- AR 5: Key performance indicators for tracking implementation
- AR 6-7: Governance and oversight of transition plan
- AR 8: Definition and calculation of locked-in emissions
- AR 9-10: CapEx breakdown by decarbonization lever
- AR 11-13: Sector-specific considerations

**Cross-References:**
- ESRS 2 SBM-3: Material impacts, risks, opportunities from climate change
- ESRS 2 GOV-3: Climate-related governance
- EU Taxonomy: Climate mitigation substantial contribution criteria
- E1-4: Emission reduction targets (must be consistent with plan)
- E1-6: Baseline and current emissions (plan must reference)

### 3.2 E1-2: Policies Related to Climate Change (para 22-24)

**Objective (ESRS E1 para 22):** The undertaking shall describe its policies adopted to manage its material impacts, risks, and opportunities related to climate change mitigation and adaptation.

**Required Datapoints:**
- Description of policies addressing climate change mitigation (para 22(a))
- Description of policies addressing climate change adaptation (para 22(b))
- Description of policies addressing energy efficiency (para 22(c))
- Description of policies addressing renewable energy deployment (para 22(d))
- How the policies relate to the transition plan (para 23)
- Scope of the policies (own operations, upstream, downstream) (para 24)

**Application Requirements (AR 14-15):**
- AR 14: Cross-reference to ESRS 2 MDR-P (Minimum Disclosure Requirements for Policies)
- AR 15: How climate policies consider the interests of affected stakeholders

### 3.3 E1-3: Actions and Resources (para 26-28)

**Objective (ESRS E1 para 26):** The undertaking shall disclose the climate change mitigation and adaptation actions and the resources allocated to their implementation.

**Required Datapoints:**
- List of key actions taken and planned (para 26(a))
- Quantified GHG emission reductions expected from each action (para 26(b))
- Amount of current and future financial resources allocated (CapEx and OpEx) (para 27)
- Timeframe for each action (para 28)
- How actions relate to targets disclosed under E1-4 (para 28(a))

**Application Requirements (AR 16-19):**
- AR 16: Cross-reference to ESRS 2 MDR-A (Minimum Disclosure Requirements for Actions)
- AR 17: Distinguish between implemented and planned actions
- AR 18: Quantification methodology for expected emission reductions
- AR 19: CapEx/OpEx breakdown per action

### 3.4 E1-4: Targets Related to Climate Change (para 30-33)

**Objective (ESRS E1 para 30):** The undertaking shall disclose the climate-related targets it has set.

**Required Datapoints:**
- GHG emission reduction targets (absolute and/or intensity) (para 30(a))
- Base year and base year emissions (para 30(b))
- Target year and target level (para 30(c))
- Scope coverage (Scope 1, 2, 3 or combination) (para 30(d))
- Whether target is science-based and validated by SBTi (para 31)
- Progress against targets (current year vs base year vs target year) (para 32)
- Methodology for target-setting (absolute contraction, sectoral decarbonization) (para 33)
- Base year recalculation policy (para 33(a), AR 25)

**Application Requirements (AR 20-29):**
- AR 20: Cross-reference to ESRS 2 MDR-T (Minimum Disclosure Requirements for Targets)
- AR 21-22: SBTi target categories (near-term, long-term, net-zero)
- AR 23: Interim milestones (5-year intervals)
- AR 24: Sectoral pathway reference (SDA, absolute contraction, FLAG)
- AR 25: Base year recalculation triggers and policy
- AR 26-27: Intensity metrics (tCO2e/EUR revenue, tCO2e/FTE, tCO2e/unit)
- AR 28: Carbon neutrality/net-zero claims and their basis
- AR 29: Target decomposition across business units

### 3.5 E1-5: Energy Consumption and Mix (para 35-39)

**Objective (ESRS E1 para 35):** The undertaking shall provide information on its energy consumption and mix.

**Required Datapoints:**
- Total energy consumption from non-renewable sources (MWh) (para 35(a))
  - Breakdown by fuel type: coal, oil, natural gas, other non-renewable (AR 30-31)
  - Energy from nuclear sources (if applicable)
- Total energy consumption from renewable sources (MWh) (para 35(b))
  - Breakdown: solar, wind, hydroelectric, biomass, geothermal, other renewable
- Total energy consumption (MWh) (para 36)
- Energy intensity per net revenue (MWh/EUR million) (para 37)
- Share of renewable energy in total energy mix (%) (para 38)
- Energy consumption from activities in high climate impact sectors (NACE A-H, L) (para 39)

**Application Requirements (AR 30-37):**
- AR 30-31: Fuel type disaggregation (minimum: coal, oil products, natural gas, other fossil, biomass, other renewable)
- AR 32: Self-generated vs purchased energy
- AR 33: Energy sold / exported (to be deducted)
- AR 34: Primary energy vs final energy
- AR 35: Conversion factors (GJ to MWh = /3.6)
- AR 36: Energy from high climate impact sectors (NACE division level)
- AR 37: Consistency with E1-6 (energy data must be consistent with emission calculations)

### 3.6 E1-6: Gross Scopes 1, 2, 3 and Total GHG Emissions (para 44-55)

**Objective (ESRS E1 para 44):** The undertaking shall disclose in metric tons of CO2 equivalent its gross Scope 1, Scope 2, Scope 3, and total GHG emissions.

**This is the most data-intensive disclosure requirement in ESRS E1.**

**Scope 1 Required Datapoints (para 44-46):**
- Total gross Scope 1 GHG emissions (tCO2e) (para 44(a))
- Percentage of Scope 1 from regulated emission trading schemes (para 44(b))
- Breakdown by country (where material) (AR 39(a))
- Breakdown by operating segment (where applicable) (AR 39(b))
- Breakdown of Scope 1 by GHG type: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3 (para 45)
- Biogenic CO2 emissions from combustion/biodegradation (separately, not in total) (para 46)
- GWP values used (must be IPCC AR6 100-year) (AR 46)

**Scope 2 Required Datapoints (para 47-49):**
- Gross location-based Scope 2 GHG emissions (tCO2e) (para 47(a))
- Gross market-based Scope 2 GHG emissions (tCO2e) (para 47(b))
- Breakdown by country (where material) (AR 39(a))
- Contractual instruments used for market-based (GOs, RECs, PPAs) (AR 50-52)
- If only one method disclosed: explanation (para 48)

**Scope 3 Required Datapoints (para 50-53):**
- Total gross Scope 3 GHG emissions (tCO2e) (para 50)
- Breakdown by Scope 3 category (15 categories) (para 51):
  - Cat 1: Purchased goods and services
  - Cat 2: Capital goods
  - Cat 3: Fuel-and-energy-related activities (not in Scope 1/2)
  - Cat 4: Upstream transportation and distribution
  - Cat 5: Waste generated in operations
  - Cat 6: Business travel
  - Cat 7: Employee commuting
  - Cat 8: Upstream leased assets
  - Cat 9: Downstream transportation and distribution
  - Cat 10: Processing of sold products
  - Cat 11: Use of sold products
  - Cat 12: End-of-life treatment of sold products
  - Cat 13: Downstream leased assets
  - Cat 14: Franchises
  - Cat 15: Investments
- Scope 3 categories excluded with justification (para 52)
- Data quality rating per Scope 3 category (AR 58-59)
- Calculation methodology per category (supplier-specific, hybrid, average, spend-based) (AR 55-57)
- Biogenic CO2 in Scope 3 (separate disclosure) (para 53)

**Total GHG Emissions (para 54-55):**
- Total GHG emissions (Scope 1 + location-based Scope 2 + Scope 3) (para 54)
- Total GHG emissions (Scope 1 + market-based Scope 2 + Scope 3) (para 54)
- GHG emissions intensity per net revenue (tCO2e/EUR million) (para 55)

**Application Requirements (AR 39-63):**
- AR 39-41: Organizational boundary (operational control, financial control, equity share)
- AR 42-45: Scope 1 calculation methodology (fuel combustion, process, fugitive, mobile)
- AR 46: IPCC AR6 GWP-100 values mandatory
- AR 47-49: Scope 2 location-based methodology (grid emission factors)
- AR 50-52: Scope 2 market-based methodology (contractual instruments hierarchy)
- AR 53-54: Scope 2 residual mix factors
- AR 55-57: Scope 3 calculation approaches per category
- AR 58-59: Scope 3 data quality assessment
- AR 60: Base year emissions for comparison
- AR 61: Significant changes in reporting boundary
- AR 62: Restatement of prior year emissions
- AR 63: GHG emission factor sources and vintages

### 3.7 E1-7: GHG Removals and Carbon Credits (para 56-60)

**Objective (ESRS E1 para 56):** The undertaking shall disclose GHG removals and storage in its own operations and the amount of GHG emission reductions or removals from climate change mitigation projects outside its value chain financed through carbon credits.

**Required Datapoints:**
- GHG removals in own operations (tCO2e) (para 56(a))
  - By removal activity: afforestation, reforestation, BECCS, DACCS, enhanced weathering, soil carbon (AR 64)
  - Permanence assessment and reversal risk (AR 65)
- Carbon credits purchased and retired (para 57-58):
  - Total credits retired (tCO2e) (para 57(a))
  - Type: avoidance/reduction credits vs removal credits (para 57(b))
  - Certification standard: VCS, Gold Standard, ACR, CAR, CDM (para 58)
  - Vintage year of credits (para 58(a))
  - Whether credits are used to offset reported emissions (para 58(b))
- Statement that carbon credits are NOT deducted from gross GHG emissions in E1-6 (para 59)
- Crediting towards net-zero targets (if applicable) (para 60)

**Application Requirements (AR 64-68):**
- AR 64: Categories of removals (nature-based vs technological)
- AR 65: Permanence criteria and monitoring approach
- AR 66: Additionality requirements for credits
- AR 67: Third-party verification of removal activities
- AR 68: Distinction between neutralization (own removals) and compensation (credits)

### 3.8 E1-8: Internal Carbon Pricing (para 62-64)

**Objective (ESRS E1 para 62):** The undertaking shall disclose whether it applies internal carbon pricing schemes and how they support its decision-making and incentivise the implementation of climate-related policies and targets.

**Required Datapoints:**
- Whether internal carbon pricing is applied (yes/no) (para 62)
- Type of scheme: shadow price, internal carbon fee, implicit price (para 62(a))
- Carbon price level (EUR/tCO2e) (para 63)
- Scope of application (which emissions, which decisions) (para 63(a))
- How the carbon price is used in investment decisions (para 63(b))
- How the carbon price supports transition plan implementation (para 64)
- Methodology for setting the price level (para 64(a))
- Evolution of the price over time (current, planned increases) (AR 69-70)

**Application Requirements (AR 69-72):**
- AR 69: Shadow price vs internal fee distinction
- AR 70: Price trajectory (current year, 2030, 2040, 2050)
- AR 71: Revenue generated by internal carbon fees and how it is used
- AR 72: Link between internal carbon price and external carbon markets (EU ETS, CBAM)

### 3.9 E1-9: Anticipated Financial Effects from Climate Risks (para 66-69)

**Objective (ESRS E1 para 66):** The undertaking shall disclose the anticipated financial effects of material physical risks and material transition risks and the potential climate-related opportunities.

**Required Datapoints:**
- Monetary amount of assets at material physical risk (para 66(a))
  - Breakdown: chronic risks (sea level rise, temperature increase, water stress) and acute risks (floods, storms, wildfires) (AR 73)
- Monetary amount of assets at material transition risk (para 66(b))
  - Breakdown: policy/legal, technology, market, reputation (AR 74)
- Proportion of assets at risk vs total assets (%) (para 67)
- Monetary amount of net revenue at risk from physical and transition risks (para 67(a))
- Potential financial effects of climate opportunities (para 68):
  - Green revenue (products/services supporting climate mitigation/adaptation)
  - Cost savings from energy efficiency and renewables
  - Access to green finance (lower cost of capital)
- CapEx related to assets at material risk (para 69)
- Alignment with EU Taxonomy CapEx (cross-reference) (AR 79)

**Application Requirements (AR 73-79):**
- AR 73: Physical risk categories (acute and chronic) with geographic mapping
- AR 74: Transition risk categories per TCFD framework
- AR 75: Time horizons for risk materialization (short, medium, long term)
- AR 76: Scenario analysis approach (qualitative or quantitative)
- AR 77: Sensitivity analysis of key assumptions
- AR 78: Reconciliation with financial statements
- AR 79: Consistency with EU Taxonomy reporting

---

## 4. Pack Architecture

### 4.1 Directory Structure

```
PACK-016-esrs-e1-climate/
├── __init__.py
├── pack.yaml
├── config/
│   ├── __init__.py
│   ├── pack_config.py
│   ├── presets/
│   │   ├── __init__.py
│   │   ├── power_generation.yaml         # Power plants, utilities, IPPs
│   │   ├── manufacturing.yaml           # Heavy/light industry, process emissions
│   │   ├── transport.yaml               # Fleet, logistics, aviation, maritime
│   │   ├── financial_services.yaml      # Financed emissions, PCAF, transition risk
│   │   ├── real_estate.yaml             # Building emissions, CRREM pathways
│   │   └── multi_sector.yaml            # Diversified groups, conglomerates
│   └── demo/
│       ├── __init__.py
│       └── demo_config.yaml
├── engines/
│   ├── __init__.py
│   ├── ghg_inventory_engine.py          # Engine 1: Scope 1/2/3 GHG (E1-6)
│   ├── energy_mix_engine.py             # Engine 2: Energy consumption (E1-5)
│   ├── transition_plan_engine.py        # Engine 3: Transition plan (E1-1)
│   ├── climate_target_engine.py         # Engine 4: Targets (E1-4)
│   ├── climate_action_engine.py         # Engine 5: Actions + Policies (E1-2, E1-3)
│   ├── carbon_credit_engine.py          # Engine 6: Removals & Credits (E1-7)
│   ├── carbon_pricing_engine.py         # Engine 7: Internal pricing (E1-8)
│   └── climate_risk_engine.py           # Engine 8: Financial effects (E1-9)
├── workflows/
│   ├── __init__.py
│   ├── ghg_inventory_workflow.py              # Workflow 1: GHG inventory (E1-6)
│   ├── energy_assessment_workflow.py          # Workflow 2: Energy assessment (E1-5)
│   ├── transition_plan_workflow.py            # Workflow 3: Transition plan (E1-1)
│   ├── target_setting_workflow.py             # Workflow 4: Climate targets (E1-4)
│   ├── climate_actions_workflow.py            # Workflow 5: Actions & policies (E1-2, E1-3)
│   ├── carbon_credits_workflow.py             # Workflow 6: Carbon credits (E1-7)
│   ├── carbon_pricing_workflow.py             # Workflow 7: Carbon pricing (E1-8)
│   ├── climate_risk_workflow.py               # Workflow 8: Climate risk (E1-9)
│   └── full_e1_workflow.py                    # Workflow 9: End-to-end E1 disclosure
├── templates/
│   ├── __init__.py
│   ├── ghg_emissions_report.py                # Template 1: E1-6 GHG emissions
│   ├── energy_mix_report.py                   # Template 2: E1-5 energy consumption
│   ├── transition_plan_report.py              # Template 3: E1-1 transition plan
│   ├── climate_policy_report.py               # Template 4: E1-2 climate policies
│   ├── climate_actions_report.py              # Template 5: E1-3 actions & resources
│   ├── climate_targets_report.py              # Template 6: E1-4 climate targets
│   ├── carbon_credits_report.py               # Template 7: E1-7 carbon credits
│   ├── carbon_pricing_report.py               # Template 8: E1-8 carbon pricing
│   └── climate_risk_report.py                 # Template 9: E1-9 financial effects
├── integrations/
│   ├── __init__.py
│   ├── pack_orchestrator.py                   # Master orchestrator (10-phase DAG pipeline)
│   ├── ghg_app_bridge.py                      # Bridge to GL-GHG-APP (GHG inventory I/O)
│   ├── mrv_agent_bridge.py                    # Bridge to all 30 MRV agents
│   ├── dma_pack_bridge.py                     # Bridge to PACK-015 (DMA)
│   ├── decarbonization_bridge.py              # Bridge to GL-DECARB agents (21 agents)
│   ├── adaptation_bridge.py                   # Bridge to GL-ADAPT agents (12 agents)
│   ├── health_check.py                        # 10-category health check
│   └── setup_wizard.py                        # 8-step climate setup
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_manifest.py
    ├── test_config.py
    ├── test_demo.py
    ├── test_ghg_inventory.py
    ├── test_energy_mix.py
    ├── test_transition_plan.py
    ├── test_climate_target.py
    ├── test_climate_action.py
    ├── test_carbon_credit.py
    ├── test_carbon_pricing.py
    ├── test_workflows.py
    ├── test_templates.py
    ├── test_integrations.py
    ├── test_e2e.py
    └── test_agent_integration.py
```

### 4.2 Components Summary

| Category | Count | Description |
|----------|-------|-------------|
| Engines | 8 | GHG inventory, energy mix, transition plan, climate targets, climate actions, carbon credits, carbon pricing, climate risk |
| Workflows | 9 | 8 domain-specific workflows + 1 full E1 end-to-end workflow |
| Templates | 9 | One ESRS-aligned report template per disclosure requirement |
| Integrations | 8 | Orchestrator, GHG app bridge, MRV agent bridge, DMA bridge, decarbonization bridge, adaptation bridge, health check, setup wizard |
| Presets | 6 | Power generation, manufacturing, transport, financial services, real estate, multi-sector |
| Tests | 17 | conftest + manifest + config + demo + 6 engine tests + workflows + templates + integrations + e2e + agent integration |

### 4.3 Key Architectural Decisions

1. **One engine per disclosure cluster, not per DR**: E1-2 (policies) and E1-3 (actions) share the ClimateActionEngine because policies and actions are inherently linked. E1-6 (GHG emissions) gets its own dedicated GHGInventoryEngine due to calculation complexity.

2. **Deterministic calculation only**: All GHG calculations, energy aggregations, target progress computations, and financial risk quantifications use deterministic arithmetic and lookup tables. No LLM is used in any calculation path. LLM is used only for narrative generation (transition plan descriptions, policy summaries) with explicit human review gates.

3. **AR6 GWP enforcement**: The GHGInventoryEngine enforces IPCC AR6 100-year GWP values for all gas conversions. AR5 or AR4 values are rejected with an explicit warning. This ensures compliance with ESRS E1 AR 46.

4. **Dual Scope 2 calculation**: The GHGInventoryEngine always calculates both location-based and market-based Scope 2. Market-based instruments (GOs, RECs, PPAs) are validated against the GHG Protocol Scope 2 Guidance quality criteria hierarchy.

5. **Scope 3 completeness**: All 15 Scope 3 categories are supported via AGENT-MRV-014 through 028 plus cross-cutting mapper (029). Categories excluded by the company require documented justification per ESRS E1 para 52.

6. **Cross-DR consistency**: The pack enforces internal consistency across all 9 DRs. For example, targets in E1-4 must reference the same base year emissions as E1-6. Transition plan actions in E1-1 must align with resources in E1-3.

---

## 5. Engine Specifications

### 5.1 Engine 1: GHG Inventory Engine

**File**: `engines/ghg_inventory_engine.py`
**Purpose**: Calculate Scope 1, Scope 2 (location-based and market-based), and Scope 3 GHG emissions per GHG Protocol methodology with IPCC AR6 GWP values. Produces all datapoints required by ESRS E1 para 44-55.

**Key Features:**
- **Scope 1 calculation**: Stationary combustion, mobile combustion, process emissions, fugitive emissions, refrigerant leakage, land use change -- each via dedicated AGENT-MRV agents (001-008)
- **Gas-level disaggregation**: Reports CO2, CH4, N2O, HFCs, PFCs, SF6, NF3 individually before aggregation to CO2e using AR6 GWP-100 values
- **AR6 GWP values enforced**: CO2=1, CH4=27.9 (fossil) / 27.2 (biogenic), N2O=273, SF6=25,200, NF3=17,400, HFC-134a=1,530, plus 200+ HFC/PFC/CFC variants
- **Biogenic CO2 separation**: Biogenic CO2 from biomass combustion and biodegradation reported as memo item, not included in Scope 1 total per ESRS E1 para 46
- **Scope 2 dual reporting**: Location-based using grid emission factors (IEA, national registries) and market-based using contractual instruments hierarchy (supplier-specific, PPA, GO, residual mix)
- **Scope 3 all 15 categories**: Each category routed to dedicated AGENT-MRV agent with appropriate calculation method (supplier-specific, hybrid, average-data, spend-based)
- **Data quality scoring**: 5-level quality score per Scope 3 category (1=estimated, 5=verified primary data) per AR 58-59
- **EU ETS allocation tracking**: Percentage of Scope 1 from regulated emission trading schemes per para 44(b)
- **Organizational boundary**: Supports operational control, financial control, and equity share approaches per AR 39-41
- **Base year comparison**: Calculates year-over-year change and presents against base year per AR 60
- **Emission factor provenance**: Every emission factor tracked with source, vintage, and SHA-256 hash

**Core Calculations:**

```
# Scope 1 per source
Scope1_source = ActivityData * EmissionFactor_AR6
  where EmissionFactor_AR6 = EF_gas * GWP_AR6_100yr for each GHG

# Scope 1 total
Scope1_total = SUM(Scope1_stationary + Scope1_mobile + Scope1_process +
                    Scope1_fugitive + Scope1_refrigerant + Scope1_land_use)

# Scope 2 location-based
Scope2_location = SUM(Electricity_MWh * GridEF_country_tCO2e_per_MWh +
                       Heat_MWh * HeatEF_tCO2e_per_MWh +
                       Steam_MWh * SteamEF_tCO2e_per_MWh +
                       Cooling_MWh * CoolingEF_tCO2e_per_MWh)

# Scope 2 market-based (instrument hierarchy)
Scope2_market = SUM(
  PPA_MWh * PPA_EF +
  GO_MWh * GO_EF +              # GO = 0 tCO2e/MWh for renewable GOs
  Supplier_MWh * Supplier_EF +
  Residual_MWh * ResidualMix_EF
)

# Scope 3 per category (example: Cat 1 spend-based)
Scope3_Cat1 = SUM(SpendAmount_EUR * EEIO_EF_tCO2e_per_EUR_by_sector)

# Total GHG (location-based)
Total_GHG_location = Scope1_total + Scope2_location + Scope3_total

# Total GHG (market-based)
Total_GHG_market = Scope1_total + Scope2_market + Scope3_total

# GHG intensity
GHG_intensity = Total_GHG / Net_Revenue_EUR_million  # tCO2e / EUR M
```

**Models**: `GHGInventoryConfig`, `Scope1Data`, `Scope2Data`, `Scope3Data`, `GasDisaggregation`, `EmissionFactor`, `AR6GWPTable`, `GHGInventoryResult`, `DataQualityScore`, `OrganizationalBoundary`, `BiogenicCO2Memo`, `EUETSAllocation`

**Regulatory References:**
- ESRS E1 para 44-55 (Scope 1/2/3 disclosure)
- ESRS E1 AR 39-63 (calculation methodology)
- GHG Protocol Corporate Standard Chapters 3-8
- GHG Protocol Scope 2 Guidance Chapters 4-7
- GHG Protocol Scope 3 Standard Chapters 5-9
- IPCC AR6 WG1 Appendix 7.SM (GWP tables)

**Edge Cases:**
- Electricity from cogeneration (CHP): Allocate based on efficiency method or energy content per GHG Protocol
- Renewable energy with GOs in one country, consumption in another: Apply GO EF to market-based, grid EF of consumption country to location-based
- Scope 3 category with zero emissions: Include with zero value (do not omit)
- Scope 3 category not relevant: Exclude with documented justification per para 52
- Acquisition/divestiture mid-year: Pro-rata calculation and base year restatement
- Missing Scope 3 supplier data: Use spend-based with DQ score = 1 and flag for improvement

### 5.2 Engine 2: Energy Mix Engine

**File**: `engines/energy_mix_engine.py`
**Purpose**: Calculate energy consumption, renewable share, and energy intensity metrics for ESRS E1-5 (para 35-39). Provides the energy data foundation that feeds into the GHGInventoryEngine for emission calculations.

**Key Features:**
- **Non-renewable breakdown**: Coal and coal products, crude oil and petroleum products, natural gas, other fossil fuels, nuclear energy -- each in MWh
- **Renewable breakdown**: Solar (PV + thermal), wind (onshore + offshore), hydroelectric, biomass/biogas, geothermal, marine/tidal, other renewable -- each in MWh
- **Self-generated vs purchased**: Tracks energy produced on-site (rooftop solar, on-site CHP) separately from purchased energy
- **Energy sold/exported**: Deducts energy sold to the grid or third parties per AR 33
- **Conversion factors**: GJ to MWh (divide by 3.6), kWh to MWh, BTU to MWh, with configurable precision
- **NACE sector flagging**: Flags energy consumption from high climate impact sectors (NACE Sections A through H and L) per para 39
- **Intensity metrics**: Energy intensity per net revenue (MWh/EUR M), per employee (MWh/FTE), per unit of production (MWh/unit)
- **Renewable share calculation**: (Renewable MWh / Total MWh) * 100, with GO-backed renewable electricity counted toward market-based renewable share
- **Primary vs final energy**: Option to report primary energy (including conversion losses) or final energy (delivered), with conversion factors per energy carrier

**Core Calculations:**

```
# Non-renewable energy
NonRenewable_MWh = Coal_MWh + Oil_MWh + Gas_MWh + OtherFossil_MWh + Nuclear_MWh

# Renewable energy
Renewable_MWh = Solar_MWh + Wind_MWh + Hydro_MWh + Biomass_MWh +
                Geothermal_MWh + Marine_MWh + OtherRenewable_MWh

# Total energy consumption
Total_Energy_MWh = NonRenewable_MWh + Renewable_MWh - Energy_Sold_MWh

# Renewable share
Renewable_Share_pct = (Renewable_MWh / Total_Energy_MWh) * 100

# Energy intensity
Energy_Intensity = Total_Energy_MWh / Net_Revenue_EUR_M

# High climate impact sector energy
HCIS_Energy_MWh = SUM(Energy for facilities with NACE in {A,B,C,D,E,F,G,H,L})
```

**Models**: `EnergyMixConfig`, `EnergySource`, `EnergyConsumptionRecord`, `RenewableBreakdown`, `NonRenewableBreakdown`, `EnergyIntensity`, `EnergyMixResult`, `NACEClassification`, `ConversionFactors`

**Regulatory References:**
- ESRS E1 para 35-39 (energy disclosure)
- ESRS E1 AR 30-37 (energy methodology)
- Eurostat energy balance methodology
- IEA World Energy Balances

**Edge Cases:**
- Self-generated renewable consumed on-site: Count as both generation and consumption
- Biomass from non-sustainable sources: Report as renewable but flag non-certified sources
- Energy from waste incineration: Classify based on waste composition (biogenic/fossil split)
- District heating/cooling: Use supplier emission factor or local default per GHG Protocol

### 5.3 Engine 3: Transition Plan Engine

**File**: `engines/transition_plan_engine.py`
**Purpose**: Structure, validate, and quantify the transition plan for climate change mitigation per ESRS E1-1 (para 14-16). Ensures internal consistency with targets (E1-4), current emissions (E1-6), and actions (E1-3).

**Key Features:**
- **Plan structure generator**: Creates ESRS-compliant transition plan structure with all mandatory sections per para 14-16 and AR 1-13
- **Decarbonization lever quantification**: For each lever (energy efficiency, fuel switching, electrification, renewables, CCS/CCUS, process change), calculates expected tCO2e reduction per year against the baseline
- **Locked-in emissions calculator**: Estimates emissions locked in from existing assets based on asset lifetime, utilization, and emission intensity per AR 8
- **CapEx/OpEx allocation tracker**: Maps financial resources to specific decarbonization actions, tracking planned vs actual spend
- **Pathway alignment assessment**: Compares plan trajectory against IEA NZE 2050, IPCC SSP1-1.9, and SBTi sector pathways
- **Milestone tracking**: Defines interim milestones (2025, 2030, 2035, 2040, 2045, 2050) with expected emission levels at each point
- **Scenario compatibility**: Tests plan against at least two climate scenarios (1.5C aligned, 2C aligned) to demonstrate resilience
- **Cross-DR consistency validation**: Validates that plan references match E1-4 targets, E1-6 baseline, E1-3 actions, and E1-5 energy mix trajectory

**Core Calculations:**

```
# Locked-in emissions
Locked_In_tCO2e = SUM(Asset_i.AnnualEmissions * Asset_i.RemainingLife_years)
  for each asset_i with remaining economic life > 0

# Lever reduction potential
Lever_Reduction_tCO2e_yr = Baseline_tCO2e_yr * ReductionFactor_pct / 100
  where ReductionFactor is lever-specific (e.g., LED = 15% electricity reduction)

# Plan trajectory at year Y
Plan_Emissions_Y = Baseline_tCO2e - SUM(Lever_j.Reduction * Lever_j.Ramp_Y)
  where Ramp_Y = implementation ramp-up factor at year Y (0 to 1)

# 1.5C alignment gap
Alignment_Gap_Y = Plan_Emissions_Y - Pathway_1_5C_Y
  Positive = above pathway (not aligned), Negative = below (aligned)

# CapEx allocation by lever
CapEx_Lever = SUM(Investment_Action_k) for actions implementing lever
```

**Models**: `TransitionPlanConfig`, `DecarbonizationLever`, `LockedInEmission`, `AssetRegister`, `CapExAllocation`, `PlanMilestone`, `PathwayAlignment`, `TransitionPlanResult`, `ScenarioComparison`

**Regulatory References:**
- ESRS E1 para 14-16 (transition plan disclosure)
- ESRS E1 AR 1-13 (transition plan application)
- ESRS 2 GOV-3 (governance of climate transition)
- IEA Net Zero by 2050 roadmap
- IPCC AR6 SSP1-1.9 scenario

### 5.4 Engine 4: Climate Target Engine

**File**: `engines/climate_target_engine.py`
**Purpose**: Manage climate targets, validate against SBTi methodology, track progress, and handle base year recalculations per ESRS E1-4 (para 30-33).

**Key Features:**
- **Target types**: Absolute emission reduction targets, intensity targets (tCO2e per revenue, per unit, per FTE), renewable energy targets, energy efficiency targets
- **SBTi validation**: Checks near-term targets (5-10yr, 1.5C or WB2C), long-term targets (by 2050, min 90% reduction), net-zero targets (residual + neutralization)
- **Scope coverage**: Tracks which scopes and categories each target covers, validates minimum SBTi scope coverage (95% Scope 1+2, 67% Scope 3)
- **Base year management**: Stores base year emissions, applies recalculation triggers (structural changes, methodology changes, error corrections per AR 25)
- **Progress tracking**: Calculates current year progress as percentage of target, projects trajectory to target year
- **Intensity denominator tracking**: For intensity targets, tracks both numerator (tCO2e) and denominator (revenue, FTE, production units) to decompose organic vs decarbonization progress
- **Sectoral pathway alignment**: Validates against SBTi sector-specific decarbonization pathways (SDA, absolute contraction, FLAG)
- **Target decomposition**: Breaks group-level targets into business unit sub-targets with accountability tracking

**Core Calculations:**

```
# Absolute target progress
Progress_pct = (BaseYear_tCO2e - CurrentYear_tCO2e) /
               (BaseYear_tCO2e - TargetYear_tCO2e) * 100

# Intensity target progress
Intensity_Base = BaseYear_tCO2e / BaseYear_Revenue
Intensity_Current = CurrentYear_tCO2e / CurrentYear_Revenue
Intensity_Target = Target_tCO2e_per_M_EUR
Progress_pct = (Intensity_Base - Intensity_Current) /
               (Intensity_Base - Intensity_Target) * 100

# SBTi minimum ambition check (absolute contraction)
Required_Annual_Reduction = 4.2%  # for 1.5C aligned near-term
Actual_Annual_Reduction = (1 - (CurrentYear/BaseYear)^(1/Years_Elapsed)) * 100
SBTi_Aligned = Actual_Annual_Reduction >= Required_Annual_Reduction

# Base year recalculation
Recalculated_BaseYear = Original_BaseYear + Structural_Adjustment
  where Structural_Adjustment accounts for M&A, divestiture, methodology change
```

**Models**: `ClimateTargetConfig`, `EmissionTarget`, `IntensityTarget`, `SBTiValidation`, `BaseYearData`, `TargetProgress`, `SectoralPathway`, `TargetDecomposition`, `ClimateTargetResult`

**Regulatory References:**
- ESRS E1 para 30-33 (target disclosure)
- ESRS E1 AR 20-29 (target methodology)
- SBTi Corporate Net-Zero Standard v1.2
- SBTi Criteria and Recommendations v5.1

### 5.5 Engine 5: Climate Action Engine

**File**: `engines/climate_action_engine.py`
**Purpose**: Track climate change mitigation and adaptation actions with resource allocation, and manage climate policies. Covers both E1-2 (para 22-24) and E1-3 (para 26-28) since policies and actions are inherently linked.

**Key Features:**
- **Policy registry**: Structured repository of climate policies covering mitigation, adaptation, energy efficiency, and renewable energy deployment
- **Policy scope mapping**: Maps each policy to own operations, upstream value chain, downstream value chain per para 24
- **Action tracking**: Comprehensive register of actions with status (planned, in-progress, completed), timeline, expected emission reductions, and responsible party
- **Resource allocation**: CapEx and OpEx tracking per action, with planned vs actual spend and variance analysis
- **Emission reduction quantification**: For each action, calculates expected tCO2e reduction using engineering-grade calculation (not estimates)
- **Action-to-target linkage**: Maps each action to the target(s) it contributes toward, ensuring E1-3 actions sum to E1-4 targets
- **Adaptation action tracking**: Separate register for climate adaptation actions (physical risk mitigation) vs mitigation actions (emission reduction)
- **Stakeholder impact**: Documents how policies consider the interests of affected stakeholders per AR 15

**Core Calculations:**

```
# Action emission reduction (example: LED retrofit)
Reduction_tCO2e = Baseline_Electricity_kWh * (1 - LED_Efficiency_Ratio) *
                   Grid_EF_tCO2e_per_kWh

# Total reduction from all actions
Total_Planned_Reduction = SUM(Action_i.Expected_Reduction_tCO2e)

# Gap to target
Reduction_Gap = Target_Reduction_Required - Total_Planned_Reduction

# Resource efficiency
EUR_per_tCO2e_abated = Action_CapEx / Action_Reduction_tCO2e
```

**Models**: `ClimateActionConfig`, `ClimatePolicy`, `ClimateAction`, `ResourceAllocation`, `ActionReduction`, `PolicyScope`, `AdaptationAction`, `ClimateActionResult`

**Regulatory References:**
- ESRS E1 para 22-24 (policies)
- ESRS E1 para 26-28 (actions and resources)
- ESRS E1 AR 14-19 (policies and actions methodology)
- ESRS 2 MDR-P (minimum disclosure for policies)
- ESRS 2 MDR-A (minimum disclosure for actions)

### 5.6 Engine 6: Carbon Credit Engine

**File**: `engines/carbon_credit_engine.py`
**Purpose**: Track GHG removals in own operations and carbon credits purchased and retired per ESRS E1-7 (para 56-60). Enforces the principle that credits are never deducted from gross emissions.

**Key Features:**
- **Own removal tracking**: Tracks removals by activity type -- afforestation/reforestation, soil carbon sequestration, BECCS, DACCS, enhanced weathering, biochar, ocean-based
- **Permanence assessment**: Scores permanence risk (1-5) for each removal activity based on reversal risk (wildfire, disease, land use change for nature-based; technology failure for engineered)
- **Carbon credit registry**: Full lifecycle tracking: purchase date, vintage year, certification standard (VCS/Verra, Gold Standard, ACR, CAR, CDM, Article 6.4), project type, country of origin
- **Credit classification**: Distinguishes avoidance/reduction credits (e.g., renewable energy, cookstoves) from removal credits (e.g., afforestation, DACCS) per para 57(b)
- **Additionality verification**: Flags whether credits meet additionality criteria per certification standard
- **Retirement tracking**: Records retirement against specific reporting periods and claims
- **Non-deduction enforcement**: Programmatically prevents carbon credits from being deducted from E1-6 gross emissions -- credits are reported as a separate memo item only
- **Net-zero crediting rules**: For companies with net-zero targets, validates that only removal credits are counted toward neutralization of residual emissions per SBTi net-zero standard

**Core Calculations:**

```
# Own removals total
Own_Removals_tCO2e = SUM(RemovalActivity_i.tCO2e_removed *
                          RemovalActivity_i.Permanence_Factor)

# Credits purchased in period
Credits_Purchased_tCO2e = SUM(CreditPurchase_j.Quantity_tCO2e)

# Credits retired in period
Credits_Retired_tCO2e = SUM(CreditRetirement_k.Quantity_tCO2e)

# Credit inventory
Credit_Inventory = Previous_Inventory + Purchased - Retired

# Gross emissions remain unchanged
Gross_Scope1 = <from GHGInventoryEngine>  # Credits NOT deducted
Net_Emissions_Memo = Gross_Total - Own_Removals - Credits_Retired  # Memo only
```

**Models**: `CarbonCreditConfig`, `OwnRemovalActivity`, `CarbonCredit`, `CreditRetirement`, `PermanenceAssessment`, `CertificationStandard`, `CarbonCreditResult`

**Regulatory References:**
- ESRS E1 para 56-60 (removals and credits)
- ESRS E1 AR 64-68 (methodology)
- SBTi Corporate Net-Zero Standard Section 5 (neutralization)
- VCMI Claims Code of Practice

### 5.7 Engine 7: Carbon Pricing Engine

**File**: `engines/carbon_pricing_engine.py`
**Purpose**: Calculate and disclose internal carbon pricing schemes per ESRS E1-8 (para 62-64), linking to investment decision-making and transition plan support.

**Key Features:**
- **Pricing scheme types**: Shadow pricing (investment appraisal), internal carbon fee (operational charge), implicit price (derived from abatement costs)
- **Price level management**: Current price (EUR/tCO2e), planned escalation path (2025-2050), benchmark against EU ETS price and social cost of carbon
- **Scope of application**: Tracks which decisions use the carbon price -- CapEx approval, procurement, product pricing, supplier selection, R&D prioritization
- **Investment decision impact**: Calculates how the carbon price changes NPV of investment proposals (carbon-adjusted NPV)
- **Revenue tracking** (for internal fee): If internal carbon fee generates revenue, tracks allocation of funds (to decarbonization projects, green R&D, offsets)
- **EU ETS and CBAM linkage**: Compares internal price against EU ETS allowance price and CBAM certificate price to assess cost exposure
- **Price trajectory modeling**: Projects future carbon costs under different scenarios (IEA NZE, NGFS Orderly/Disorderly/Hot House)

**Core Calculations:**

```
# Shadow price impact on investment NPV
Carbon_Cost_Year_Y = Projected_Emissions_Y * Shadow_Price_Y
NPV_Carbon = SUM(Carbon_Cost_Year_Y / (1 + Discount_Rate)^Y)
Adjusted_NPV = Standard_NPV - NPV_Carbon

# Internal carbon fee revenue
Annual_Fee_Revenue = Total_Emissions_tCO2e * Internal_Fee_EUR_per_tCO2e

# EU ETS exposure comparison
ETS_Cost = ETS_Verified_Emissions * ETS_Allowance_Price
Internal_Price_Gap = Internal_Price - ETS_Allowance_Price
```

**Models**: `CarbonPricingConfig`, `PricingScheme`, `PriceTrajectory`, `InvestmentImpact`, `FeeRevenue`, `ETSComparison`, `CarbonPricingResult`

**Regulatory References:**
- ESRS E1 para 62-64 (internal carbon pricing)
- ESRS E1 AR 69-72 (pricing methodology)
- EU ETS Directive 2003/87/EC (amended)
- CBAM Regulation 2023/956

### 5.8 Engine 8: Climate Risk Engine

**File**: `engines/climate_risk_engine.py`
**Purpose**: Quantify anticipated financial effects from material physical and transition risks, and identify climate-related opportunities per ESRS E1-9 (para 66-69).

**Key Features:**
- **Physical risk assessment**: Acute risks (floods, storms, wildfires, heatwaves) and chronic risks (sea level rise, temperature increase, precipitation change, water stress) -- assessed per asset/site using geospatial hazard data from AGENT-DATA-020 (Climate Hazard Connector)
- **Transition risk assessment**: Policy/legal (carbon pricing, regulation), technology (stranded assets, substitution), market (demand shifts, commodity prices), reputation (stakeholder expectations, litigation) -- per TCFD framework
- **Asset-level quantification**: For each asset at risk, calculates monetary exposure (EUR) based on asset value, replacement cost, revenue contribution, and insurance coverage
- **Revenue-at-risk**: Quantifies net revenue exposed to physical and transition risks by product line, geography, and time horizon
- **Opportunity quantification**: Green revenue from climate-positive products/services, cost savings from efficiency, access to green finance, new market opportunities
- **Time horizon analysis**: Short-term (<1yr), medium-term (1-5yr), long-term (>5yr) per ESRS 1 ss77
- **Scenario analysis**: Quantitative scenario analysis under at least two scenarios (e.g., IEA NZE 1.5C, NGFS Hot House 3C+)
- **EU Taxonomy alignment**: Cross-references assets and revenue with EU Taxonomy climate mitigation/adaptation CapEx per AR 79

**Core Calculations:**

```
# Physical risk: Asset value at risk
Asset_Physical_Risk_EUR = Asset_Value * Hazard_Probability * Damage_Factor
  where Damage_Factor = f(hazard_type, asset_vulnerability, adaptation_measures)

# Physical risk: Total
Total_Physical_Risk_EUR = SUM(Asset_i.Physical_Risk_EUR)
Physical_Risk_pct_Assets = Total_Physical_Risk_EUR / Total_Assets_EUR * 100

# Transition risk: Carbon cost exposure
Carbon_Cost_Risk_EUR = Projected_Emissions_tCO2e * Projected_Carbon_Price_EUR
  where Projected_Carbon_Price per scenario (NZE, CPS, STEPS)

# Transition risk: Stranded asset risk
Stranded_Asset_Risk_EUR = SUM(Asset_j.BookValue * StrandingProbability_pct/100)

# Revenue at risk
Revenue_Physical_Risk = SUM(Revenue_Segment_k * Risk_Factor_k)
Revenue_Transition_Risk = SUM(Revenue_Product_l * ObsolescenceFactor_l)

# Climate opportunity: Green revenue
Green_Revenue_EUR = SUM(Revenue from Taxonomy-aligned activities)
```

**Models**: `ClimateRiskConfig`, `PhysicalRisk`, `TransitionRisk`, `AssetRiskExposure`, `RevenueAtRisk`, `ClimateOpportunity`, `ScenarioResult`, `FinancialEffectsResult`, `TaxonomyAlignment`

**Regulatory References:**
- ESRS E1 para 66-69 (financial effects)
- ESRS E1 AR 73-79 (risk quantification methodology)
- TCFD Recommendations (2017) and Implementation Annex
- NGFS Climate Scenarios (2024)
- EU Taxonomy Delegated Regulation 2021/2139

---

## 6. Workflow Specifications

### 6.1 Workflow 1: Transition Plan Workflow
**File**: `workflows/transition_plan_workflow.py`

**6-phase workflow (baseline assessment, lever identification, action planning, gap analysis, scenario validation, report generation):**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Data Collection | Gather strategic plans, decarbonization studies, asset register, CapEx budgets, scenario analysis inputs | 60 min |
| 2 | Lever Quantification | Run TransitionPlanEngine to quantify each decarbonization lever and calculate locked-in emissions | 30 min |
| 3 | Pathway Alignment | Compare plan trajectory against IEA NZE, IPCC SSP1-1.9, SBTi sector pathways | 15 min |
| 4 | Cross-DR Validation | Validate consistency with E1-4 targets, E1-6 baseline, E1-3 actions | 15 min |
| 5 | Report Generation | Generate E1-1 disclosure using TransitionPlanReport template with XBRL tagging | 15 min |

**Estimated total:** 135 minutes

### 6.2 Workflow 2: Climate Actions Workflow
**File**: `workflows/climate_actions_workflow.py`

**3-phase workflow:**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Policy Inventory | Collect and structure all climate-related policies (mitigation, adaptation, energy, renewables) | 45 min |
| 2 | Scope Mapping | Map each policy to value chain scope (own ops, upstream, downstream) and stakeholder relevance | 20 min |
| 3 | Report Generation | Generate E1-2 disclosure using ClimatePolicyReport template with MDR-P compliance | 15 min |

**Estimated total:** 80 minutes

### 6.3 Workflow 3: Target Setting Workflow
**File**: `workflows/target_setting_workflow.py`

**4-phase workflow:**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Action Inventory | Collect all climate actions (planned, in-progress, completed) with timelines | 45 min |
| 2 | Reduction Quantification | Calculate expected tCO2e reduction per action using ClimateActionEngine | 30 min |
| 3 | Resource Mapping | Map CapEx/OpEx allocation per action, reconcile with financial planning | 20 min |
| 4 | Report Generation | Generate E1-3 disclosure with MDR-A compliance and target linkage | 15 min |

**Estimated total:** 110 minutes

### 6.4 Workflow 4: GHG Inventory Workflow
**File**: `workflows/ghg_inventory_workflow.py`

**5-phase workflow:**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Target Inventory | Collect all climate targets with base years, target years, scope coverage | 30 min |
| 2 | SBTi Validation | Validate targets against SBTi criteria (ambition, scope coverage, methodology) | 20 min |
| 3 | Progress Calculation | Calculate current progress vs base year and project trajectory to target year | 15 min |
| 4 | Base Year Check | Assess whether base year recalculation is needed (structural changes, errors) | 15 min |
| 5 | Report Generation | Generate E1-4 disclosure with MDR-T compliance and SBTi status | 15 min |

**Estimated total:** 95 minutes

### 6.5 Workflow 5: Energy Assessment Workflow
**File**: `workflows/energy_assessment_workflow.py`

**4-phase workflow:**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Energy Data Collection | Gather energy bills, metering data, PPA contracts, renewable certificates | 45 min |
| 2 | Aggregation & Classification | Run EnergyMixEngine to aggregate by source type, classify renewable/non-renewable | 20 min |
| 3 | Intensity Calculation | Calculate energy intensity per revenue, per employee, per production unit | 10 min |
| 4 | Report Generation | Generate E1-5 disclosure with fuel-type breakdown, renewable share, NACE flagging | 15 min |

**Estimated total:** 90 minutes

### 6.6 Workflow 6: Carbon Credits Workflow
**File**: `workflows/carbon_credits_workflow.py`

**This is the most complex workflow, as E1-6 is the most data-intensive disclosure.**

**7-phase workflow:**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Scope 1 Calculation | Route data to MRV agents 001-008, aggregate stationary/mobile/process/fugitive/refrigerant/land use | 60 min |
| 2 | Scope 2 Dual Calc | Calculate location-based (grid EFs) and market-based (contractual instruments) via MRV agents 009-013 | 30 min |
| 3 | Scope 3 Calculation | Route to MRV agents 014-028 for all 15 categories, run Scope 3 Category Mapper (029) | 90 min |
| 4 | Gas Disaggregation | Disaggregate total by GHG type (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3) with AR6 GWP | 15 min |
| 5 | Quality Assessment | Score data quality per Scope 3 category (5-level), flag estimation methods | 15 min |
| 6 | Reconciliation | Cross-check Scope 1 with E1-5 energy data, validate totals, biogenic CO2 separation | 15 min |
| 7 | Report Generation | Generate E1-6 disclosure with all breakdowns, intensities, base year comparison, XBRL | 20 min |

**Estimated total:** 245 minutes

### 6.7 Workflow 7: Carbon Pricing Workflow
**File**: `workflows/carbon_pricing_workflow.py`

**3-phase workflow:**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Removals Assessment | Collect own removal activities, assess permanence, quantify tCO2e removed | 30 min |
| 2 | Credit Reconciliation | Reconcile credit registry (purchased, retired, inventory), validate certifications | 20 min |
| 3 | Report Generation | Generate E1-7 disclosure with non-deduction statement, credit breakdown, net-zero crediting | 15 min |

**Estimated total:** 65 minutes

### 6.8 Workflow 8: Climate Risk Workflow
**File**: `workflows/climate_risk_workflow.py`

**3-phase workflow:**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Scheme Documentation | Document carbon pricing scheme type, price level, scope of application | 20 min |
| 2 | Impact Analysis | Calculate carbon price impact on investment decisions, fee revenue allocation | 20 min |
| 3 | Report Generation | Generate E1-8 disclosure with price trajectory, EU ETS comparison | 10 min |

**Estimated total:** 50 minutes

### 6.9 Workflow 9: Full E1 Disclosure Workflow
**File**: `workflows/full_e1_workflow.py`

**5-phase workflow:**

| Phase | Name | Description | Duration |
|-------|------|-------------|----------|
| 1 | Physical Risk Mapping | Map assets and revenue to physical hazard exposure using geospatial data | 60 min |
| 2 | Transition Risk Assessment | Assess policy, technology, market, reputation risks per scenario | 45 min |
| 3 | Opportunity Identification | Quantify climate opportunities (green revenue, cost savings, green finance) | 30 min |
| 4 | Scenario Analysis | Run financial impact under 1.5C and 3C+ scenarios | 30 min |
| 5 | Report Generation | Generate E1-9 disclosure with risk tables, opportunity summary, Taxonomy cross-ref | 15 min |

**Estimated total:** 180 minutes

### 6.10 Full E1 Orchestration Pipeline
**File**: `integrations/pack_orchestrator.py`

**10-phase end-to-end pipeline executing all 9 DRs in DAG dependency order:**

| Phase | DR | Dependency |
|-------|----|------------|
| 1 | Initialization | Health check, config loading, DMA check (E1 materiality confirmed) |
| 2 | E1-5 Energy Mix | No dependency (data foundation) |
| 3 | E1-6 GHG Emissions | Depends on E1-5 (energy data feeds Scope 2) |
| 4 | E1-2 Climate Policies | No dependency |
| 5 | E1-3 Actions & Resources | Depends on E1-6 (baseline for reduction calcs) |
| 6 | E1-4 Climate Targets | Depends on E1-6 (base year emissions) |
| 7 | E1-1 Transition Plan | Depends on E1-4 + E1-6 + E1-3 (plan references all) |
| 8 | E1-7 Removals & Credits | Depends on E1-6 (gross emissions context) |
| 9 | E1-8 Carbon Pricing | Depends on E1-6 (emission volumes for price application) |
| 10 | E1-9 Financial Effects | Depends on E1-6 + E1-4 (risk linked to emissions trajectory) |
| 11 | Cross-DR Validation | Validates consistency across all 9 DRs |
| 12 | Final Report Assembly | Assembles complete E1 section with XBRL tagging |

**Estimated total:** 12-20 hours (with parallel phases 2+4, 5+6, 8+9)

---

## 7. Template Specifications

### 7.1 Template 1: Transition Plan Report
**File**: `templates/transition_plan_report.py`
**DR**: E1-1 (para 14-16)
**Sections**: Plan overview, decarbonization levers table, locked-in emissions, CapEx/OpEx allocation, pathway alignment chart (vs IEA NZE), milestone timeline, governance, year-over-year progress
**Formats**: PDF, HTML, XBRL
**XBRL Tags**: E1-1 datapoints from EFRAG taxonomy (transition plan description, locked-in emissions amount, CapEx for climate mitigation)

### 7.2 Template 2: Climate Policy Report
**File**: `templates/climate_policy_report.py`
**DR**: E1-2 (para 22-24)
**Sections**: Policy inventory table (name, scope, coverage, date adopted), mitigation policies, adaptation policies, energy efficiency policies, renewable energy policies, stakeholder engagement on policies, MDR-P compliance checklist
**Formats**: PDF, HTML, XBRL

### 7.3 Template 3: Climate Actions Report
**File**: `templates/climate_actions_report.py`
**DR**: E1-3 (para 26-28)
**Sections**: Action register (name, status, timeline, expected reduction, CapEx, OpEx), implementation progress, resource allocation by action and lever, gap analysis (actions vs targets), MDR-A compliance checklist
**Formats**: PDF, HTML, XBRL

### 7.4 Template 4: Climate Targets Report
**File**: `templates/climate_targets_report.py`
**DR**: E1-4 (para 30-33)
**Sections**: Target overview table (scope, base year, target year, target level, SBTi status), progress chart (base year to current to target), SBTi validation summary, base year recalculation log, intensity metric trends, sector pathway comparison, MDR-T compliance checklist
**Formats**: PDF, HTML, XBRL

### 7.5 Template 5: Energy Mix Report
**File**: `templates/energy_mix_report.py`
**DR**: E1-5 (para 35-39)
**Sections**: Total energy consumption table, non-renewable breakdown (by fuel type), renewable breakdown (by source), self-generated vs purchased, energy sold/exported, renewable share (% with trend), energy intensity (MWh/EUR M), high climate impact sector energy, year-over-year comparison
**Formats**: PDF, HTML, XBRL

### 7.6 Template 6: GHG Emissions Report
**File**: `templates/ghg_emissions_report.py`
**DR**: E1-6 (para 44-55)
**Sections**: Scope 1 summary and breakdown (by source, by gas, by country), Scope 2 location-based summary, Scope 2 market-based summary (with instrument details), Scope 3 category breakdown (15 categories with methodology and DQ score), biogenic CO2 memo, total GHG (location and market), GHG intensity, EU ETS allocation %, base year comparison, emission factor sources, organizational boundary statement
**Formats**: PDF, HTML, XBRL, CSV (data tables)

### 7.7 Template 7: Carbon Credits Report
**File**: `templates/carbon_credits_report.py`
**DR**: E1-7 (para 56-60)
**Sections**: Own removals table (activity type, tCO2e, permanence score), credits purchased (standard, vintage, project type, country), credits retired, credit inventory, non-deduction statement, net-zero crediting assessment
**Formats**: PDF, HTML, XBRL

### 7.8 Template 8: Carbon Pricing Report
**File**: `templates/carbon_pricing_report.py`
**DR**: E1-8 (para 62-64)
**Sections**: Scheme description, price level (current EUR/tCO2e), price trajectory (chart to 2050), scope of application, investment decision examples, fee revenue and allocation, EU ETS and CBAM price comparison
**Formats**: PDF, HTML, XBRL

### 7.9 Template 9: Climate Risk Report
**File**: `templates/climate_risk_report.py`
**DR**: E1-9 (para 66-69)
**Sections**: Physical risk summary (assets at risk EUR, % of total), physical risk breakdown (acute vs chronic, by geography), transition risk summary (assets at risk EUR, % of total), transition risk breakdown (policy, technology, market, reputation), revenue at risk, climate opportunities (green revenue, cost savings, green finance), CapEx at risk, scenario analysis results (1.5C vs 3C+), EU Taxonomy CapEx cross-reference
**Formats**: PDF, HTML, XBRL

---

## 8. Integration Specifications

### 8.1 Pack Orchestrator (E1PackOrchestrator)
**File**: `integrations/pack_orchestrator.py`

10-phase DAG pipeline with dependency resolution, parallel execution, retry with exponential backoff, and SHA-256 provenance tracking:

1. **materiality_check**: Verify E1 climate materiality from DMA (PACK-015)
2. **ghg_inventory**: Compute Scope 1/2/3 GHG inventory via MRV agents
3. **energy_assessment**: Assess energy consumption and mix (E1-5)
4. **transition_plan**: Evaluate or build climate transition plan (E1-1)
5. **target_setting**: Set and validate climate targets / SBTi alignment (E1-4)
6. **climate_actions**: Catalogue climate actions and resources (E1-2, E1-3)
7. **carbon_credits**: Assess carbon credit/removal portfolio (E1-7)
8. **carbon_pricing**: Evaluate carbon pricing exposure (E1-8)
9. **climate_risk**: Analyze physical and transition risks (E1-9)
10. **report_assembly**: Assemble the final E1 disclosure package with all 9 templates

**DAG Dependencies:**
- materiality_check --> ghg_inventory --> energy_assessment
- energy_assessment --> transition_plan --> target_setting
- target_setting --> climate_actions
- climate_actions --> carbon_credits (parallel with carbon_pricing)
- climate_actions --> carbon_pricing (parallel with carbon_credits)
- carbon_credits + carbon_pricing --> climate_risk
- climate_risk --> report_assembly

### 8.2 GHG Application Bridge (GHGAppBridge)
**File**: `integrations/ghg_app_bridge.py`

Bridge to GL-GHG-APP (applications/GL-GHG-APP) for bidirectional GHG data flow:
- **Import**: Pull GHG inventory data from GL-GHG-APP into PACK-016 E1 engines
- **Export**: Push calculated E1-6 emissions data back to GL-GHG-APP
- **Base year sync**: Synchronize base year emissions between GL-GHG-APP and E1 target engine
- **Scope 3 integration**: Route Scope 3 data through GL-GHG-APP for consolidated reporting
- **Provenance tracking**: All data flows tracked with SHA-256 hashes for audit trail

### 8.3 MRV Agent Bridge (MRVAgentBridge)
**File**: `integrations/mrv_agent_bridge.py`

Routes emissions data from all 30 AGENT-MRV agents to E1 GHG inventory engine:
- **Scope 1**: AGENT-MRV-001 (stationary combustion), 002 (refrigerants), 003 (mobile), 004 (process), 005 (fugitive), 006 (land use), 007 (waste treatment), 008 (agricultural)
- **Scope 2**: AGENT-MRV-009 (location-based), 010 (market-based), 011 (steam/heat), 012 (cooling), 013 (dual reporting reconciliation)
- **Scope 3**: AGENT-MRV-014 through 028 (categories 1-15), 029 (category mapper)
- **Audit**: AGENT-MRV-030 (audit trail and lineage)

Data flow: PACK-016 sends activity data to MRV agents, receives calculated emissions with provenance hashes.

### 8.4 DMA Pack Bridge (DMAPackBridge)
**File**: `integrations/dma_pack_bridge.py`

Bridge to PACK-015 (Double Materiality Assessment):
- **E1 materiality status**: Receives DMA results confirming E1 (Climate Change) is material, including materiality scores for E1 sub-topics (mitigation, adaptation, energy)
- **IRO import**: Links E1 disclosures to specific IROs from the DMA IRO register
- **Financial materiality**: Imports financial materiality assessment for E1-9 risk quantification
- **Conditional activation**: If DMA determines E1 is not material, pack enters "mandatory-only" mode (E1-6 Scope 1+2 remain mandatory per Omnibus)
- **Related disclosures**: Identifies which E1 disclosure requirements are triggered by materiality results

### 8.5 Decarbonization Bridge (DecarbonizationBridge)
**File**: `integrations/decarbonization_bridge.py`

Bridge to GL-DECARB agents (21 decarbonization agents):
- **Transition plan import**: Import decarbonization roadmap and pathway data for E1-1
- **Target synchronization**: Sync climate targets between decarbonization agents and E1-4 engine
- **Abatement option import**: Import abatement options and marginal abatement cost curves
- **MACC curve integration**: Feed MACC data into transition plan cost-effectiveness analysis
- **Progress tracking**: Synchronize decarbonization progress metrics with E1 disclosure tracking

### 8.6 Adaptation Bridge (AdaptationBridge)
**File**: `integrations/adaptation_bridge.py`

Bridge to GL-ADAPT agents (12 climate adaptation agents):
- **Physical risk import**: Import physical risk assessment data (acute and chronic) for E1-9
- **Transition risk import**: Import transition risk assessment data for E1-9
- **Opportunity assessment**: Import climate opportunity identification for E1-9
- **Scenario analysis**: Import climate scenario analysis results (NGFS, IEA, IPCC)
- **TCFD alignment**: Validate E1-9 disclosures against TCFD framework requirements

### 8.7 Health Check (E1HealthCheck)
**File**: `integrations/health_check.py`

10-category system health verification:
1. **Engines**: All 8 engines loaded, initialized, and operational
2. **Workflows**: All 9 workflows loaded and callable
3. **Templates**: All 9 templates loaded with markdown/html/json rendering
4. **Integrations**: All bridges (GHG app, MRV, DMA, decarbonization, adaptation) reachable
5. **Configuration**: Preset loaded, company profile complete, validation passing
6. **Manifest**: pack.yaml valid, version consistent, all components listed
7. **MRV Agents**: 30 MRV agents (Scope 1: 001-008, Scope 2: 009-013, Scope 3: 014-030) responsive
8. **PACK-015 Dependency**: DMA bridge connectivity and materiality data available
9. **Infrastructure**: Database, cache, and file system accessible
10. **Provenance**: SHA-256 hashing functional, audit trail operational

### 8.8 Setup Wizard
**File**: `integrations/setup_wizard.py`

8-step guided configuration:
1. **Company Profile**: Legal entity, NACE code, employee count, revenue, reporting period
2. **Organizational Boundary**: Operational control / financial control / equity share
3. **Scope Selection**: Scope 1 sources, Scope 2 method preference, Scope 3 category relevance
4. **Data Source Connection**: ERP, energy management, fleet management, procurement, financial planning
5. **Emission Factor Selection**: Default databases (IEA, national, custom), grid EF country selection
6. **Target Configuration**: Import existing SBTi targets or define new targets
7. **Carbon Pricing Setup**: Internal pricing scheme configuration (if applicable)
8. **Preset Selection**: Choose sector preset or customize

---

## 9. Configuration and Presets

### 9.1 Power Generation Preset
**File**: `config/presets/power_generation.yaml`

Sector-specific for utilities, IPPs, CHP operators, and renewable energy developers. High Scope 1 intensity (direct combustion in power generation). Low Scope 2 (self-generation, net exporter). Scope 3 dominated by Cat 3 (fuel supply chain) and Cat 11 (use of sold products). Energy mix is the core disclosure (fuel vs. renewable breakdown). Transition plan focuses on coal-to-gas-to-renewable pathway. SBTi SDA power sector pathway applies. Carbon pricing exposure through EU ETS. Methane intensity tracking for gas-fired generation.

### 9.2 Manufacturing Preset
**File**: `config/presets/manufacturing.yaml`

Covers NACE division C (heavy and light industry: steel, cement, chemicals, automotive, electronics, food processing). High Scope 1 (combustion + process emissions via AGENT-MRV-004). High Scope 2 (industrial electricity demand). Scope 3 dominated by Cat 1 (purchased goods), Cat 4 (upstream transport), Cat 5 (waste), Cat 11 (use of sold products). Process emissions require specific emission factors (e.g., clinker ratio for cement). GHG intensity per tonne of product is key metric. Transition plan involves electrification, process innovation (CCS, DRI-H2, electric furnace). SBTi sector-specific SDA pathways (steel, cement, chemicals). EU ETS and CBAM integration enabled. Full gas disaggregation including PFCs and N2O.

### 9.3 Transport and Logistics Preset
**File**: `config/presets/transport.yaml`

Sector-specific for road freight, maritime shipping, airlines, rail operators, and last-mile delivery. High Scope 1 (mobile combustion from fleet via AGENT-MRV-003). Modest Scope 2 (depots, terminals, rail traction). Scope 3 Cat 3 (fuel supply chain) is significant. GHG intensity per tonne-km or per passenger-km is key metric. Energy mix dominated by liquid fuels (diesel, jet fuel, marine fuel). Transition plan focuses on fleet electrification, sustainable aviation fuel (SAF), biofuels. SBTi transport sector SDA pathway applies.

### 9.4 Financial Services Preset
**File**: `config/presets/financial_services.yaml`

Sector-specific for banks, insurers, asset managers, pension funds, and development finance institutions. Low Scope 1+2 (offices, business travel, data centers). Dominant Scope 3 Category 15 (financed/insured emissions per PCAF Standard). GHG intensity per EUR million invested/lent is key metric. Portfolio alignment metrics (implied temperature rise). Transition plan focuses on portfolio decarbonization and exclusion policies. SBTi financial sector guidance applies. Climate risk assessment emphasizes portfolio-level physical and transition risk (ECB, EBA). SFDR PAI indicator integration. Green asset ratio.

### 9.5 Real Estate Preset
**File**: `config/presets/real_estate.yaml`

Sector-specific for commercial real estate owners, residential property companies, REITs, developers, and hotel operators. Moderate Scope 1 (on-site gas/oil heating). Significant Scope 2 (purchased electricity, district heating). Scope 3 dominated by Cat 1/2 (embodied carbon from construction materials), Cat 13 (tenant emissions), Cat 12 (end-of-life demolition). GHG intensity per sqm (kgCO2e/sqm/year) is key metric. Energy Performance Certificate (EPC) rating integration. Transition plan follows CRREM decarbonization pathways. Building-level energy and emissions tracking. Stranded asset risk from building energy performance requirements.

### 9.6 Multi-Sector Conglomerate Preset
**File**: `config/presets/multi_sector.yaml`

Default configuration for diversified groups, holding companies, and multi-divisional organizations. All Scope 1 sources potentially relevant (combustion + process). All Scope 2 sources relevant. All 15 Scope 3 categories screened for significance. Division-level reporting with group consolidation. Multiple GHG intensity denominators (revenue, production, FTE). Transition plan covers multiple decarbonization pathways. Climate risk assessment across all divisions. Comprehensive emission factor coverage. Suitable for first-time CSRD reporters with complex organizational structures.

---

## 10. Test Strategy

### 10.1 Test File Summary

| Test File | Focus | Target Tests |
|-----------|-------|-------------|
| conftest.py | Shared fixtures (company profiles, energy data, emissions data, targets, risk data) | N/A (fixtures) |
| test_manifest.py | Pack YAML validation, metadata, component listing | 65+ |
| test_config.py | Config system, presets, validation, defaults | 50+ |
| test_demo.py | Demo smoke tests with sample company data | 60+ |
| test_ghg_inventory.py | Engine 1: Scope 1/2/3 calculations, AR6 GWP, gas disaggregation, biogenic CO2 | 120+ |
| test_energy_mix.py | Engine 2: Energy aggregation, renewable share, intensity, NACE classification | 80+ |
| test_transition_plan.py | Engine 3: Locked-in emissions, lever quantification, pathway alignment | 70+ |
| test_climate_target.py | Engine 4: SBTi validation, progress tracking, base year recalculation | 75+ |
| test_climate_action.py | Engine 5: Action tracking, reduction quantification, policy registry | 60+ |
| test_carbon_credit.py | Engine 6: Removal tracking, credit lifecycle, non-deduction enforcement | 55+ |
| test_carbon_pricing.py | Engine 7: Shadow pricing, fee revenue, EU ETS comparison | 40+ |
| test_workflows.py | All 9 workflows (8 domain-specific + full E1) | 40+ |
| test_templates.py | All 9 templates + registry, XBRL tagging | 36+ |
| test_integrations.py | All 8 integrations including orchestrator | 30+ |
| test_e2e.py | End-to-end scenarios (manufacturing, financial, energy, multi-site) | 15+ |
| test_agent_integration.py | Agent wiring verification for all 30 MRV + DATA agents | 20+ |
| **Total** | | **816+** |

### 10.2 Key Test Scenarios

**Scenario 1: Manufacturing Company (Steel)**
- Scope 1: BF-BOF process emissions (AGENT-MRV-004), stationary combustion (001), fugitive (005)
- Scope 2: Location-based and market-based with PPA for renewable electricity
- Scope 3: Cat 1 (iron ore, coking coal), Cat 4 (transport), Cat 9 (steel delivery), Cat 11 (construction use)
- Target: SBTi sectoral decarbonization (steel pathway, 1.5C aligned)
- Transition plan: DRI-H2 migration, electric arc furnace, CCS retrofit
- E1-9: Carbon pricing risk (EU ETS Phase IV), stranded BF-BOF assets

**Scenario 2: Financial Services (Bank)**
- Scope 1+2: Office buildings (small)
- Scope 3 Cat 15: Financed emissions across loan and investment portfolio (PCAF)
- Target: Net-zero financed emissions by 2050 (SBTi Financial Institutions)
- Transition plan: Portfolio decarbonization, exclusion policies, green finance targets
- E1-9: Transition risk on high-carbon portfolio segments, stranded asset exposure

**Scenario 3: Energy Company (Utility)**
- Scope 1: Power generation (natural gas CCGT, coal phase-out), upstream methane
- Scope 2: Minimal (self-generating)
- Scope 3 Cat 11: Use of sold natural gas and electricity (fossil portion)
- Target: Coal phase-out 2030, net-zero Scope 1+2 by 2040
- Transition plan: Gas-to-renewables, battery storage, green hydrogen
- E1-9: Stranded fossil generation assets, physical risk to grid infrastructure

**Scenario 4: Retail Company (Multi-site)**
- Scope 1: Refrigerant leakage (HFCs from store cooling), vehicle fleet
- Scope 2: Large electricity consumption across 500+ stores
- Scope 3: Cat 1 (purchased products, packaging), Cat 4 (logistics), Cat 9 (customer delivery)
- Target: Absolute reduction 46% by 2030, net-zero by 2050
- Transition plan: Natural refrigerants, electric fleet, renewable PPAs, supplier engagement
- E1-8: Internal carbon fee of EUR 100/tCO2e on all CapEx decisions

**Scenario 5: SME (Omnibus Threshold)**
- Scope 1+2 only (mandatory per Omnibus)
- Scope 3 not disclosed (below materiality threshold with justification)
- Simplified E1-5 energy disclosure
- No SBTi targets (voluntary future consideration)
- No transition plan (explanation of timeline for adoption per AR 2)

### 10.3 Test Categories

**Unit Tests** (Engines):
- Correct AR6 GWP application for each GHG (CO2, CH4, N2O, SF6, NF3, all HFCs/PFCs)
- Biogenic CO2 excluded from Scope 1 total
- Location-based and market-based Scope 2 produce different results when contractual instruments differ
- Scope 3 exclusion requires documented justification
- SBTi target validation catches insufficient ambition, missing scope coverage, wrong methodology
- Base year recalculation correctly adjusts for M&A
- Locked-in emissions correctly sum asset-level emissions over remaining life
- Carbon credits never deducted from gross emissions
- Physical risk aggregation across sites produces correct total
- Cross-DR consistency validation catches mismatched base years

**Integration Tests** (Agent Wiring):
- AGENT-MRV-001 through 030 all produce valid emission outputs consumed by GHGInventoryEngine
- AGENT-DATA-020 hazard data correctly feeds ClimateRiskEngine
- PACK-015 DMA bridge correctly activates/deactivates E1 DRs based on materiality
- PACK-008 Taxonomy bridge correctly exchanges CapEx and revenue data

**End-to-End Tests** (Full Disclosure):
- Complete E1 disclosure for manufacturing company (all 9 DRs, all data populated)
- Complete E1 disclosure for financial services (financed emissions focus)
- Complete E1 disclosure for SME (mandatory-only mode)
- XBRL output validates against EFRAG E1 taxonomy schema

---

## 11. Dependencies

### 11.1 Pack Dependencies

| Dependency | Type | Required | Purpose |
|------------|------|----------|---------|
| PACK-015 Double Materiality | Pack | Required | E1 materiality determination (is E1 material?) |
| PACK-001/002/003 CSRD | Pack | Required | CSRD sustainability statement integration |
| PACK-008 EU Taxonomy | Pack | Optional | Taxonomy alignment cross-reference for E1-1, E1-9 |
| PACK-004/005 CBAM | Pack | Optional | CBAM embedded emissions for heavy industry preset |

### 11.2 Agent Dependencies

| Agent Layer | Count | Required | Purpose |
|-------------|-------|----------|---------|
| AGENT-MRV (001-030) | 30 | 30 required | Scope 1/2/3 emission calculations |
| AGENT-DATA (001-020) | 20 | 12 required | Data intake, quality, validation |
| AGENT-FOUND (001-010) | 10 | 10 required | Orchestration, schema, provenance, audit |
| **Total** | **60** | **52 required** | |

### 11.3 Application Dependencies

| Application | Required | Purpose |
|-------------|----------|---------|
| GL-GHG-APP v1.0 | Required | GHG inventory management (underlying data layer for E1-6) |
| GL-CSRD-APP v1.1 | Required | CSRD report assembly (E1 section integration) |
| GL-Taxonomy-APP v1.0 | Optional | Taxonomy alignment for E1-1/E1-9 cross-reference |
| GL-SBTi-APP v1.0 | Optional | SBTi target management for E1-4 validation |

### 11.4 Infrastructure Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | >=3.11 | Runtime |
| PostgreSQL | >=16 | Data storage, reporting |
| TimescaleDB | >=2.11 | Time-series emission data |
| Redis | >=7.0 | Caching emission factors, calculation results |
| pgvector | >=0.5 | Similarity search for sector benchmarks |

---

## 12. Performance Requirements

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Scope 1 calculation (single entity) | <30 seconds | Time for all Scope 1 sources aggregated |
| Scope 2 dual calculation | <15 seconds | Time for location + market-based |
| Scope 3 all 15 categories | <5 minutes | Time for complete Scope 3 inventory |
| E1-5 energy aggregation | <10 seconds | Time for energy mix calculation |
| E1-6 full GHG report generation | <10 minutes | Time from data ready to report output |
| E1-1 transition plan assembly | <5 minutes | Time for plan structuring and validation |
| E1-9 scenario analysis (2 scenarios) | <3 minutes | Time for financial effect quantification |
| Full E1 disclosure (all 9 DRs) | <60 minutes | Automated processing time (excludes human review) |
| XBRL tagging for all E1 datapoints | <2 minutes | Time for full XBRL package generation |
| Cache hit ratio | >75% | Emission factor and conversion factor lookups |
| Memory ceiling | 4096 MB | Maximum RAM during full E1 workflow |
| Concurrent users | 25 | Simultaneous E1 disclosure processes |

---

## 13. Security Requirements

| Requirement | Implementation |
|-------------|---------------|
| Authentication | JWT (RS256) via SEC-001 |
| Authorization | RBAC with E1-specific roles (climate_lead, emissions_analyst, target_manager, risk_analyst, auditor) |
| Encryption at rest | AES-256-GCM for all emission data, target data, financial risk data |
| Encryption in transit | TLS 1.3 for all API and agent communication |
| Audit logging | All E1 data access, calculation runs, report generation logged via SEC-005 |
| PII redaction | Employee names redacted from commuting data (Scope 3 Cat 7) |
| Data classification | CONFIDENTIAL (financial risk data, transition plans), RESTRICTED (emission factors, GHG data), INTERNAL (targets, policies), PUBLIC (published disclosures) |
| Provenance | SHA-256 hash for every emission calculation, factor lookup, and report output |
| Retention | Configurable per preset (5-10 years), minimum per CSRD audit requirements |

---

## 14. Acceptance Criteria

### 14.1 Launch Criteria (Go/No-Go)

- [ ] All 8 engines implemented with 100% of documented features
- [ ] All 10 workflows functional and tested
- [ ] All 9 templates generating correct ESRS-formatted output
- [ ] All 8 integrations connected and verified by health check
- [ ] AR6 GWP values correctly applied for all 200+ GHG variants
- [ ] Scope 2 dual reporting (location + market) produces correct distinct values
- [ ] Scope 3 all 15 categories operational via MRV agents
- [ ] SBTi target validation catches all non-compliant target types
- [ ] Carbon credits provably never deducted from gross E1-6 emissions
- [ ] Cross-DR consistency validation passes for all test scenarios
- [ ] XBRL output validates against EFRAG E1 taxonomy schema
- [ ] 85%+ test coverage across all code
- [ ] 896+ tests passing (0 critical/high failures)
- [ ] Security audit passed (Grade A)
- [ ] Performance targets met (full E1 <60 minutes automated)
- [ ] 5 beta customers successfully generating E1 disclosures
- [ ] External auditor review of 3 sample E1 outputs with <2 findings
- [ ] Documentation complete (API docs, user guide, ESRS mapping guide)

### 14.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 25 active customers generating E1 disclosures
- 100+ E1 reports generated
- <5 calculation accuracy issues reported
- <3 support tickets per customer
- NPS >40

**60 Days:**
- 75 active customers
- 500+ E1 reports generated
- Zero calculation accuracy issues (all resolved)
- <2 support tickets per customer
- NPS >50

**90 Days:**
- 150 active customers
- 1,500+ E1 reports generated
- 99.9% uptime
- External auditor acceptance rate >98%
- NPS >55

---

## 15. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| AR6 | IPCC Sixth Assessment Report (2021), source of GWP-100 values for GHG conversion |
| BECCS | Bioenergy with Carbon Capture and Storage, a technological removal method |
| Biogenic CO2 | CO2 from biological sources (biomass combustion, biodegradation), reported separately |
| CBAM | Carbon Border Adjustment Mechanism, EU regulation on embedded emissions in imports |
| CO2e | Carbon dioxide equivalent, a metric for comparing different GHGs using GWP factors |
| COP | Conference of the Parties (UNFCCC), annual climate negotiations |
| CSRD | Corporate Sustainability Reporting Directive (EU 2022/2464) |
| DACCS | Direct Air Carbon Capture and Storage, a technological removal method |
| DR | Disclosure Requirement (within ESRS) |
| EFRAG | European Financial Reporting Advisory Group, developer of ESRS |
| ETS | Emissions Trading System (EU ETS) |
| GHG | Greenhouse Gas |
| GHG Protocol | Greenhouse Gas Protocol, global standard for GHG accounting |
| GO | Guarantee of Origin, EU certificate for renewable electricity |
| GWP | Global Warming Potential, metric for comparing GHGs relative to CO2 |
| HFC | Hydrofluorocarbon, a potent greenhouse gas (F-gas) |
| IEA NZE | International Energy Agency Net Zero Emissions by 2050 scenario |
| IPCC | Intergovernmental Panel on Climate Change |
| IRO | Impact, Risk, Opportunity (ESRS terminology) |
| Locked-in emissions | Future emissions from existing assets based on remaining economic life |
| NGFS | Network for Greening the Financial System, source of climate scenarios |
| NZE | Net Zero Emissions |
| Omnibus I | EU Directive 2026/470, revising CSRD thresholds and reducing datapoints |
| PCAF | Partnership for Carbon Accounting Financials (financed emissions methodology) |
| PFC | Perfluorocarbon, a potent greenhouse gas |
| PPA | Power Purchase Agreement, contractual instrument for renewable electricity |
| REC | Renewable Energy Certificate (market-based Scope 2 instrument) |
| Residual mix | Grid emission factor after removing tracked/claimed renewable attributes |
| SBTi | Science Based Targets initiative |
| SDA | Sectoral Decarbonization Approach (SBTi target-setting method) |
| SF6 | Sulphur hexafluoride, most potent GHG (GWP=25,200 in AR6) |
| Shadow price | Internal carbon price used in investment appraisal but not as actual charge |
| Scope 1 | Direct GHG emissions from owned or controlled sources |
| Scope 2 | Indirect GHG emissions from purchased electricity, heat, steam, cooling |
| Scope 3 | Other indirect GHG emissions in the value chain (15 categories) |
| TCFD | Task Force on Climate-related Financial Disclosures |
| Transition plan | Plan describing how the company transitions to climate neutrality |
| XBRL | eXtensible Business Reporting Language (EU digital reporting format) |

### Appendix B: ESRS E1 Datapoint to Engine Mapping

| ESRS E1 Datapoint | DR | Engine | Method |
|--------------------|-----|--------|--------|
| Transition plan description | E1-1 | TransitionPlanEngine | generate_plan_structure() |
| Locked-in emissions (tCO2e) | E1-1 | TransitionPlanEngine | calculate_locked_in() |
| Decarbonization levers | E1-1 | TransitionPlanEngine | quantify_levers() |
| Transition plan CapEx (EUR) | E1-1 | TransitionPlanEngine | allocate_capex() |
| Pathway alignment assessment | E1-1 | TransitionPlanEngine | assess_alignment() |
| Climate mitigation policies | E1-2 | ClimateActionEngine | get_policies(type=mitigation) |
| Climate adaptation policies | E1-2 | ClimateActionEngine | get_policies(type=adaptation) |
| Energy efficiency policies | E1-2 | ClimateActionEngine | get_policies(type=energy) |
| Policy value chain scope | E1-2 | ClimateActionEngine | map_policy_scope() |
| Key climate actions | E1-3 | ClimateActionEngine | get_actions() |
| Expected emission reductions per action | E1-3 | ClimateActionEngine | quantify_reductions() |
| CapEx/OpEx per action (EUR) | E1-3 | ClimateActionEngine | allocate_resources() |
| GHG reduction targets | E1-4 | ClimateTargetEngine | get_targets() |
| Base year and base year emissions | E1-4 | ClimateTargetEngine | get_base_year() |
| SBTi validation status | E1-4 | ClimateTargetEngine | validate_sbti() |
| Target progress (%) | E1-4 | ClimateTargetEngine | calculate_progress() |
| Total energy consumption (MWh) | E1-5 | EnergyMixEngine | calculate_total() |
| Non-renewable breakdown (MWh) | E1-5 | EnergyMixEngine | breakdown_nonrenewable() |
| Renewable breakdown (MWh) | E1-5 | EnergyMixEngine | breakdown_renewable() |
| Renewable share (%) | E1-5 | EnergyMixEngine | calculate_renewable_share() |
| Energy intensity (MWh/EUR M) | E1-5 | EnergyMixEngine | calculate_intensity() |
| Gross Scope 1 GHG (tCO2e) | E1-6 | GHGInventoryEngine | calculate_scope1() |
| Scope 1 by GHG type | E1-6 | GHGInventoryEngine | disaggregate_gases() |
| Scope 1 EU ETS share (%) | E1-6 | GHGInventoryEngine | calculate_ets_share() |
| Biogenic CO2 (tCO2) | E1-6 | GHGInventoryEngine | calculate_biogenic() |
| Scope 2 location-based (tCO2e) | E1-6 | GHGInventoryEngine | calculate_scope2_location() |
| Scope 2 market-based (tCO2e) | E1-6 | GHGInventoryEngine | calculate_scope2_market() |
| Scope 3 total (tCO2e) | E1-6 | GHGInventoryEngine | calculate_scope3_total() |
| Scope 3 by category (tCO2e) | E1-6 | GHGInventoryEngine | calculate_scope3_categories() |
| Scope 3 data quality per category | E1-6 | GHGInventoryEngine | assess_data_quality() |
| Total GHG emissions (tCO2e) | E1-6 | GHGInventoryEngine | calculate_total_ghg() |
| GHG intensity (tCO2e/EUR M) | E1-6 | GHGInventoryEngine | calculate_ghg_intensity() |
| Own GHG removals (tCO2e) | E1-7 | CarbonCreditEngine | calculate_own_removals() |
| Carbon credits retired (tCO2e) | E1-7 | CarbonCreditEngine | get_retired_credits() |
| Credit type (avoidance/removal) | E1-7 | CarbonCreditEngine | classify_credits() |
| Certification standard | E1-7 | CarbonCreditEngine | get_certification() |
| Non-deduction statement | E1-7 | CarbonCreditEngine | generate_non_deduction() |
| Internal carbon pricing applied | E1-8 | CarbonPricingEngine | get_scheme_status() |
| Carbon price level (EUR/tCO2e) | E1-8 | CarbonPricingEngine | get_price_level() |
| Price trajectory | E1-8 | CarbonPricingEngine | get_trajectory() |
| Application scope | E1-8 | CarbonPricingEngine | get_application_scope() |
| Assets at physical risk (EUR) | E1-9 | ClimateRiskEngine | calculate_physical_risk() |
| Assets at transition risk (EUR) | E1-9 | ClimateRiskEngine | calculate_transition_risk() |
| Revenue at risk (EUR) | E1-9 | ClimateRiskEngine | calculate_revenue_risk() |
| Climate opportunities (EUR) | E1-9 | ClimateRiskEngine | quantify_opportunities() |
| Scenario analysis results | E1-9 | ClimateRiskEngine | run_scenarios() |

### Appendix C: IPCC AR6 GWP-100 Reference Values (Key Gases)

| Gas | Chemical Formula | AR6 GWP-100 | AR5 GWP-100 (deprecated) |
|-----|-----------------|-------------|--------------------------|
| Carbon dioxide | CO2 | 1 | 1 |
| Methane (fossil) | CH4 | 29.8 | 28 |
| Methane (non-fossil) | CH4 | 27.0 | 28 |
| Nitrous oxide | N2O | 273 | 265 |
| Sulphur hexafluoride | SF6 | 25,200 | 23,500 |
| Nitrogen trifluoride | NF3 | 17,400 | 16,100 |
| HFC-23 | CHF3 | 14,600 | 12,400 |
| HFC-32 | CH2F2 | 771 | 677 |
| HFC-125 | CHF2CF3 | 3,740 | 3,170 |
| HFC-134a | CH2FCF3 | 1,530 | 1,300 |
| HFC-143a | CH3CF3 | 5,810 | 4,800 |
| HFC-152a | CH3CHF2 | 164 | 138 |
| HFC-227ea | CF3CHFCF3 | 3,600 | 3,350 |
| HFC-245fa | CHF2CH2CF3 | 962 | 858 |
| PFC-14 | CF4 | 7,380 | 6,630 |
| PFC-116 | C2F6 | 12,400 | 11,100 |
| PFC-218 | C3F8 | 9,290 | 8,900 |

*Full AR6 GWP table (200+ gases) loaded in GHGInventoryEngine at runtime.*

### Appendix D: Regulatory Cross-Reference Matrix

| ESRS E1 DR | GHG Protocol | SBTi | TCFD | EU Taxonomy | ISO 14064-1 |
|------------|-------------|------|------|-------------|-------------|
| E1-1 Transition Plan | -- | Net-Zero Standard | Strategy (b) | Art 8 CapEx | -- |
| E1-2 Policies | -- | -- | Risk Mgmt (a) | -- | -- |
| E1-3 Actions | -- | Target-setting (actions) | Risk Mgmt (b) | Art 8 CapEx | -- |
| E1-4 Targets | -- | SBTi Criteria v5.1 | Metrics (a) | -- | -- |
| E1-5 Energy | -- | -- | Metrics (b) | DNSH energy | -- |
| E1-6 Scope 1 | Corporate Standard Ch. 4-5 | Scope 1 boundary | Metrics (b) | DNSH GHG | Clause 5.2.1-5.2.4 |
| E1-6 Scope 2 | Scope 2 Guidance | Scope 2 boundary | Metrics (b) | DNSH GHG | Clause 5.2.5 |
| E1-6 Scope 3 | Scope 3 Standard | Scope 3 boundary | Metrics (b) | -- | Clause 5.2.6 |
| E1-7 Removals | -- | Neutralization (Ch. 5) | -- | -- | Clause 5.3 |
| E1-8 Carbon Pricing | -- | -- | Risk Mgmt (c) | -- | -- |
| E1-9 Financial Effects | -- | -- | Strategy (c), Metrics (c) | Art 8 Revenue | -- |

### Appendix E: Data Dictionary

| Field | Type | Unit | Source | E1 DR |
|-------|------|------|--------|-------|
| scope1_total_tco2e | float | tCO2e | GHGInventoryEngine | E1-6 |
| scope2_location_tco2e | float | tCO2e | GHGInventoryEngine | E1-6 |
| scope2_market_tco2e | float | tCO2e | GHGInventoryEngine | E1-6 |
| scope3_total_tco2e | float | tCO2e | GHGInventoryEngine | E1-6 |
| scope3_cat1_tco2e through scope3_cat15_tco2e | float | tCO2e | GHGInventoryEngine | E1-6 |
| biogenic_co2_tco2 | float | tCO2 | GHGInventoryEngine | E1-6 |
| ghg_intensity_tco2e_per_m_eur | float | tCO2e/EUR M | GHGInventoryEngine | E1-6 |
| energy_total_mwh | float | MWh | EnergyMixEngine | E1-5 |
| energy_renewable_mwh | float | MWh | EnergyMixEngine | E1-5 |
| energy_nonrenewable_mwh | float | MWh | EnergyMixEngine | E1-5 |
| renewable_share_pct | float | % | EnergyMixEngine | E1-5 |
| energy_intensity_mwh_per_m_eur | float | MWh/EUR M | EnergyMixEngine | E1-5 |
| locked_in_emissions_tco2e | float | tCO2e | TransitionPlanEngine | E1-1 |
| transition_capex_eur | float | EUR | TransitionPlanEngine | E1-1 |
| target_base_year_tco2e | float | tCO2e | ClimateTargetEngine | E1-4 |
| target_current_year_tco2e | float | tCO2e | ClimateTargetEngine | E1-4 |
| target_progress_pct | float | % | ClimateTargetEngine | E1-4 |
| sbti_validated | boolean | -- | ClimateTargetEngine | E1-4 |
| own_removals_tco2e | float | tCO2e | CarbonCreditEngine | E1-7 |
| credits_retired_tco2e | float | tCO2e | CarbonCreditEngine | E1-7 |
| internal_carbon_price_eur | float | EUR/tCO2e | CarbonPricingEngine | E1-8 |
| assets_physical_risk_eur | float | EUR | ClimateRiskEngine | E1-9 |
| assets_transition_risk_eur | float | EUR | ClimateRiskEngine | E1-9 |
| revenue_at_risk_eur | float | EUR | ClimateRiskEngine | E1-9 |
| green_revenue_eur | float | EUR | ClimateRiskEngine | E1-9 |

---

**Approval Signatures:**

- Product Manager: ___________________
- Engineering Lead: ___________________
- Regulatory Lead: ___________________
- CEO: ___________________

---

*Document generated by GreenLang Product Team. All ESRS paragraph references are to Delegated Regulation (EU) 2023/2772 as amended. GWP values are IPCC AR6 WG1 (2021). SBTi references are to v1.2 of the Corporate Net-Zero Standard and v5.1 of the Criteria and Recommendations.*
