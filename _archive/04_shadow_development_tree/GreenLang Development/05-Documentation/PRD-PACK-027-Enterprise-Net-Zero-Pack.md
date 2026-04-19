# PRD-PACK-027: Enterprise Net Zero Pack

**Pack ID:** PACK-027-enterprise-net-zero
**Category:** Net Zero Packs
**Tier:** Enterprise
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-19
**Prerequisite:** None (standalone; enhanced with PACK-021/022/023/024/025 if present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Large enterprises -- defined as organizations with more than 250 employees and more than $50 million in annual revenue -- face a fundamentally different net-zero challenge than SMEs or mid-market companies. These organizations operate at a scale and complexity that renders simplified tools, spend-based estimation, and single-entity approaches wholly inadequate. Enterprise net-zero programs must contend with a distinct set of structural challenges:

1. **Full Scope 3 complexity across all 15 categories**: Unlike SMEs that can focus on 3-5 material Scope 3 categories using spend-based estimation, enterprises must calculate emissions across all 15 GHG Protocol Scope 3 categories using activity-based data where available. For a typical Fortune 500 company, Scope 3 represents 70-90% of total emissions, distributed across purchased goods and services (Cat 1, typically 30-50% alone), capital goods (Cat 2), fuel and energy activities (Cat 3), upstream transportation (Cat 4), waste (Cat 5), business travel (Cat 6), employee commuting (Cat 7), upstream leased assets (Cat 8), downstream transportation (Cat 9), processing of sold products (Cat 10), use of sold products (Cat 11), end-of-life treatment (Cat 12), downstream leased assets (Cat 13), franchises (Cat 14), and investments (Cat 15). Each category requires distinct calculation methodologies, data sources, and emission factors. Manual management of 15 categories across multiple business units, geographies, and reporting years is a multi-month, multi-FTE effort prone to material errors.

2. **Multi-entity consolidation complexity**: Large enterprises operate through complex corporate structures with 10 to 500+ subsidiaries, joint ventures, associates, and special-purpose vehicles across dozens of countries. GHG Protocol requires organizations to choose a consolidation approach -- financial control, operational control, or equity share -- and apply it consistently across all entities. Each approach yields materially different total emissions (often varying by 20-40%), and the choice must align with financial reporting boundaries. Intercompany transactions create double-counting risks that require systematic elimination. Acquisitions and divestitures trigger base year recalculation per the GHG Protocol's 5% significance threshold. No existing tool handles 100+ entity consolidation with intercompany elimination at the data quality standard enterprises require.

3. **SBTi Corporate Standard (not SME pathway)**: Enterprises must follow the full SBTi Corporate Manual V5.3 with 28 near-term criteria (C1-C28) and the Net-Zero Standard V1.3 with 14 net-zero criteria (NZ-C1 to NZ-C14). This requires choosing between Absolute Contraction Approach (ACA) at 4.2%/yr for 1.5C alignment and Sectoral Decarbonization Approach (SDA) for applicable sectors, setting both near-term (5-10 year) and long-term (by 2050) targets, achieving 95% coverage for Scope 1+2 and 67%+ coverage for Scope 3 near-term targets (90% for long-term), and potentially setting separate FLAG targets for land use emissions. The SME simplified pathway (PACK-026) is categorically inapplicable for organizations of this scale.

4. **Financial-grade data quality requirements**: Enterprise climate disclosures are increasingly subject to external assurance -- limited assurance today, trending toward reasonable assurance by 2028-2030. The SEC Climate Disclosure Rule, CSRD, and California SB 253 all mandate third-party verification. This requires data quality at the +/-3% accuracy level (comparable to financial statement materiality thresholds), complete audit trails with SHA-256 provenance hashing, documented methodology for every calculation, systematic data quality scoring per the GHG Protocol's data quality hierarchy, and full traceability from source data through calculation to reported figure. Enterprises cannot accept the +/-20-40% accuracy typical of spend-based SME approaches.

5. **Enterprise system integration**: Large organizations run complex ERP landscapes -- SAP S/4HANA, Oracle ERP Cloud, Workday HCM, Salesforce -- containing the activity data needed for GHG calculations. Manually extracting procurement volumes, energy consumption, fleet mileage, employee headcounts, and travel bookings from these systems is unsustainable. Enterprises need automated, bidirectional integration that pulls activity data from source systems and pushes carbon allocation back into financial reporting (carbon-adjusted P&L, EBITDA carbon intensity, carbon cost of goods sold).

6. **Advanced analytics for strategic decision-making**: Enterprise boards and C-suites require scenario modeling (1.5C, 2C, and business-as-usual pathways with Monte Carlo uncertainty analysis), internal carbon pricing ($50-$200/tCO2e applied to capital allocation decisions), Scope 4 avoided emissions quantification (for companies whose products displace higher-emission alternatives), and supply chain mapping across 3-5 tiers with engagement tracking for thousands of suppliers. These are not nice-to-have analytics -- they are prerequisites for climate-informed capital allocation at the enterprise scale.

7. **Multi-framework regulatory compliance**: Enterprises face simultaneous reporting obligations under the GHG Protocol Corporate Accounting and Reporting Standard, GHG Protocol Scope 2 Guidance, GHG Protocol Corporate Value Chain (Scope 3) Standard, SBTi Corporate Manual and Net-Zero Standard, CDP Climate Change Questionnaire, TCFD Recommendations (now ISSB S2), SEC Climate Disclosure Rule, EU Corporate Sustainability Reporting Directive (CSRD/ESRS E1), California SB 253 and SB 261, ISO 14064-1:2018, and numerous sector-specific frameworks. Each framework has overlapping but non-identical requirements, creating a reconciliation burden that grows quadratically with the number of applicable frameworks. A single source of truth with automated crosswalk to each framework is essential.

8. **External assurance readiness**: Under CSRD, enterprises must obtain limited assurance on sustainability disclosures from FY2025, transitioning to reasonable assurance by 2028. The SEC Climate Disclosure Rule requires attestation for large accelerated filers. ISO 14064-3:2019 defines verification and validation requirements. Enterprise net-zero programs must produce calculation workpapers, methodology documentation, control evidence, and data lineage that satisfy Big 4 auditor requirements without manual preparation -- a process that currently consumes 200-400 hours per assurance engagement.

### 1.2 Solution Overview

PACK-027 is the **Enterprise Net Zero Pack** -- the seventh pack in the "Net Zero Packs" category. It is the flagship enterprise-grade net-zero solution for large organizations, providing financial-grade GHG accounting, full SBTi Corporate Standard compliance, multi-entity consolidation, advanced analytics, and external assurance readiness across the complete net-zero lifecycle. The pack wraps and orchestrates existing GreenLang platform components into an enterprise-optimized experience with 8 new engines, 8 workflows, 10 templates, 13 integrations, and 8 presets.

Unlike PACK-021 (Starter, single-entity baseline), PACK-022 (Acceleration, up to 50 subsidiaries), or PACK-026 (SME, spend-based simplified), PACK-027 is designed for organizations where:

- **Scale demands automation**: 100+ entities, 50,000+ suppliers, 10,000+ employees across 30+ countries
- **Accuracy demands rigor**: +/-3% accuracy target, financial-grade data quality, external assurance
- **Complexity demands integration**: SAP/Oracle/Workday integration, multi-entity consolidation with eliminations
- **Strategy demands analytics**: Monte Carlo scenario modeling, internal carbon pricing, Scope 4 avoided emissions
- **Compliance demands completeness**: All 15 Scope 3 categories, all major regulatory frameworks simultaneously
- **Governance demands transparency**: Board-level reporting, quarterly climate dashboards, assurance workpapers

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | PACK-026 (SME) | PACK-022 (Acceleration) | PACK-027 (Enterprise) |
|-----------|----------------|------------------------|----------------------|
| Target organization | <250 employees, <$50M revenue | 250-5,000 employees | >250 employees, >$50M revenue, complex structures |
| Scope 3 coverage | 3 categories (Cat 1, 6, 7) | All 15 categories (spend-based + activity) | All 15 categories (activity-based, supplier-specific) |
| Data quality target | +/-20-40% (spend-based) | +/-10-15% (hybrid) | +/-3% (financial-grade) |
| SBTi pathway | SME simplified (50% by 2030) | ACA + SDA (12 sectors) | Full Corporate Standard (ACA/SDA/FLAG) + Net-Zero Standard |
| Entity consolidation | Single entity | Up to 50 subsidiaries | 100+ subsidiaries with intercompany elimination |
| ERP integration | Xero/QuickBooks | Basic ERP connectivity | SAP S/4HANA, Oracle ERP Cloud, Workday HCM |
| Scenario modeling | None | 3-scenario Monte Carlo | Full Monte Carlo (10,000 runs) with carbon pricing |
| Carbon pricing | None | Basic shadow pricing | Internal carbon price ($50-$200/tCO2e) with P&L allocation |
| Scope 4 | None | None | Full avoided emissions quantification |
| Supplier mapping | 10-100 suppliers | 50,000 suppliers | Multi-tier mapping (Tier 1-5), engagement tracking |
| External assurance | Not applicable | SHA-256 provenance | ISO 14064-3 ready, Big 4 workpaper generation |
| Regulatory frameworks | SME Climate Hub, B Corp | GHG Protocol, SBTi, CDP, TCFD | GHG Protocol, SBTi, CDP, TCFD, SEC, CSRD, ISO 14064, CA SB 253 |
| Implementation timeline | 2 hours (data collection) | 2-4 weeks | 6-12 weeks (phased enterprise rollout) |
| Implementation budget | <$10K | $50K-$150K | $100K-$500K |
| Max suppliers | 500 | 50,000 | 100,000+ |
| Max facilities | 50 | 2,000 | 5,000+ |
| Max employees modeled | 250 | 50,000 | 500,000+ |

### 1.4 Enterprise Market Overview

| Statistic | Value | Source |
|-----------|-------|--------|
| Large enterprises globally (>250 employees) | ~500,000 | OECD (2024) |
| Large enterprises with net-zero commitments | ~15,000 | SBTi, Race to Zero (2024) |
| Companies with validated SBTi targets | ~4,000 | SBTi Dashboard (2025) |
| Companies in CSRD scope (Omnibus I threshold) | ~10,000 | EU Commission (2026) |
| Companies in SEC Climate Rule scope | ~2,800 large accelerated filers | SEC (2024) |
| Average enterprise sustainability team size | 5-15 FTEs | Corporate Knights (2024) |
| Average enterprise climate program budget | $500K-$5M/year | CDP Supply Chain (2024) |
| Enterprises with reasonable assurance on GHG | <5% | IAASB (2024) |
| Average Scope 3 as % of total enterprise emissions | 75-90% | CDP (2024) |
| Enterprises with complete 15-category Scope 3 | <15% | CDP (2024) |

### 1.5 Target Users & Personas

**Persona 1: Chief Sustainability Officer (CSO)**
- **Context**: C-suite executive accountable for enterprise climate strategy, reporting to CEO and Board
- **Key challenges**: Aligning net-zero ambition with business reality; managing cross-functional stakeholders (CFO, COO, CTO); maintaining board confidence; navigating evolving regulatory landscape
- **PACK-027 value**: Single platform for strategy, reporting, and governance; board-ready dashboards; regulatory compliance across all frameworks; scenario analysis for strategic decisions
- **Critical features**: Executive dashboard, board climate report, scenario comparison, SBTi submission, regulatory filings

**Persona 2: Head of Sustainability / VP Sustainability**
- **Context**: Senior manager leading 5-15 person sustainability team; day-to-day program execution
- **Key challenges**: Coordinating data collection across 100+ entities; managing supplier engagement at scale; meeting multiple reporting deadlines (CDP April, SEC annual, CSRD annual); preparing for external assurance
- **PACK-027 value**: Automated data collection via ERP; multi-entity consolidation; supplier portal; template-driven reporting; assurance workpaper generation
- **Critical features**: Comprehensive baseline workflow, annual inventory workflow, supply chain heatmap, CDP auto-population, assurance package

**Persona 3: Chief Financial Officer (CFO)**
- **Context**: Finance leader integrating climate risk into financial planning; climate-related financial disclosures
- **Key challenges**: Carbon cost impact on P&L; CBAM exposure; green bond eligibility; EU Taxonomy CapEx alignment; SOX-compliant controls for climate data; investor communication
- **PACK-027 value**: Carbon-adjusted financials; internal carbon pricing; investment appraisal with carbon cost; ESRS E1-8/E1-9 disclosures; financial-grade data quality
- **Critical features**: Carbon pricing engine, financial integration engine, carbon-adjusted P&L, CBAM exposure, regulatory filings (SEC, CSRD)

**Persona 4: Board Member / Audit Committee Chair**
- **Context**: Non-executive director overseeing climate governance; audit committee oversight of climate disclosures
- **Key challenges**: Sufficient climate literacy for effective governance; confidence in data quality; liability exposure from incorrect disclosures; understanding climate risk to business model
- **PACK-027 value**: Board-level dashboard designed for non-specialists; quarterly climate report; assurance readiness; scenario analysis showing strategic implications
- **Critical features**: Board climate report, executive dashboard, scenario comparison, assurance statement

**Persona 5: Sustainability Analyst**
- **Context**: Technical specialist collecting, validating, and analyzing GHG data across multiple entities
- **Key challenges**: Data quality inconsistencies across entities; emission factor selection; Scope 3 calculation methodology for 15 categories; maintaining audit trail; time pressure for multiple reporting deadlines
- **PACK-027 value**: Automated data ingestion from ERP; calculation engine handling all 15 Scope 3 categories; DQ scoring and improvement tracking; SHA-256 provenance; template-driven outputs
- **Critical features**: Enterprise baseline engine, data quality guardian, all 30 MRV agents, audit trail

**Persona 6: Supply Chain Director**
- **Context**: Procurement/supply chain leader responsible for Scope 3 reduction through supplier engagement
- **Key challenges**: Engaging thousands of suppliers; collecting quality emission data; measuring impact of engagement; balancing climate with cost and reliability objectives
- **PACK-027 value**: Supplier tiering and hotspot identification; engagement program tracking; CDP Supply Chain integration; supplier scorecards; impact measurement
- **Critical features**: Supply chain mapping engine, supply chain portal, supply chain heatmap, engagement workflow

**Persona 7: External Auditor (Big 4)**
- **Context**: Assurance professional conducting limited or reasonable assurance on GHG statement
- **Key challenges**: Evidence quality; calculation verification; data lineage tracing; control documentation; materiality assessment; time pressure
- **PACK-027 value**: Pre-formatted workpapers; automated sample selection; calculation traces; provenance hashing; control documentation; management assertion template
- **Critical features**: Assurance statement template, external assurance workflow, workpapers (read-only auditor role)

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to complete enterprise GHG baseline (all 15 Scope 3 cats) | <6 weeks (vs. 6-12 months manual) | Time from data integration to validated baseline |
| GHG calculation accuracy vs. manual verification | +/-3% or better | Cross-validated against 1,000 known emission values |
| Scope 3 category coverage | 15/15 categories with activity-based data | Number of categories with primary data |
| Multi-entity consolidation accuracy | 100% match with financial consolidation | Cross-checked against group financial statements |
| SBTi target validation (42 criteria) | 95%+ first-pass approval rate | Targets submitted vs. targets validated |
| Assurance engagement efficiency | <80 hours auditor time (vs. 200-400 hours) | Hours billed by external auditor |
| Regulatory framework coverage | 8+ simultaneous frameworks | Frameworks with automated crosswalk |
| Scenario modeling throughput | 10,000 Monte Carlo runs in <30 minutes | Wall-clock time for full scenario analysis |
| Supplier engagement coverage | Top 80% of Scope 3 emissions engaged | Engaged supplier emissions / total Scope 3 |
| Customer NPS | >55 | Net Promoter Score survey |
| Board report generation | <15 minutes from data refresh | Time from latest data to board-ready report |
| Data quality score improvement | >15% improvement in first year | Weighted average DQ score across all categories |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| GHG Protocol Corporate Accounting and Reporting Standard | WRI/WBCSD (2004, amended 2015) | Core Scope 1+2 methodology; organizational boundary, operational boundary, quantification, reporting |
| GHG Protocol Scope 2 Guidance | WRI/WBCSD (2015) | Dual reporting (location-based and market-based); contractual instruments; residual mix |
| GHG Protocol Corporate Value Chain (Scope 3) Standard | WRI/WBCSD (2011) | All 15 categories; calculation approaches (supplier-specific, hybrid, average-data, spend-based); data quality scoring |
| SBTi Corporate Manual | V5.3 (2024) | 28 near-term criteria (C1-C28); ACA (4.2%/yr for 1.5C); SDA (12 sectors); coverage requirements (95% Scope 1+2, 67%+ Scope 3) |
| SBTi Corporate Net-Zero Standard | V1.3 (2024) | 14 net-zero criteria (NZ-C1 to NZ-C14); long-term targets (90%+ reduction by 2050); neutralization of residual emissions via permanent CDR |
| IPCC AR6 | IPCC Sixth Assessment Report (2021/2022) | GWP-100 values for all greenhouse gases; carbon budget alignment; 1.5C pathway constraints |
| Paris Agreement | UNFCCC (2015) | 1.5C temperature alignment; nationally determined contributions; global stocktake |

### 2.2 Regulatory Disclosure Frameworks

| Framework | Reference | Effective | Pack Relevance |
|-----------|-----------|-----------|----------------|
| SEC Climate Disclosure Rule | SEC Final Rule S7-10-22 (2024) | FY2025 (LAF), FY2026 (AF) | Scope 1+2 disclosure (phased); material Scope 3; attestation for LAF; financial statement footnotes; transition plan |
| CSRD / ESRS E1 | Directive (EU) 2022/2464 + Delegated Reg 2023/2772 | FY2025+ | E1-1 transition plan, E1-4 GHG reduction targets, E1-5 energy consumption, E1-6 Scope 1/2/3 emissions, E1-7 GHG removals, E1-8 internal carbon pricing, E1-9 anticipatory financial effects |
| California SB 253 (Climate Corporate Data Accountability Act) | Cal. Health & Safety Code 38532 (2023) | FY2026 (Scope 1+2), FY2027 (Scope 3) | All entities >$1B revenue doing business in CA; Scope 1+2+3; third-party assurance |
| California SB 261 (Climate-Related Financial Risk Act) | Cal. Health & Safety Code 38533 (2023) | Biennial from 2026 | Climate-related financial risk report per TCFD; >$500M revenue entities |
| ISSB S2 (Climate-related Disclosures) | IFRS S2 (2023) | Varies by jurisdiction | Successor to TCFD; governance, strategy, risk management, metrics and targets; Scope 1+2+3; transition plans |
| CDP Climate Change Questionnaire | CDP (2024/2025) | Annual | C0-C15 modules; SBTi target alignment; Scope 1+2+3; climate risk; energy; emissions reduction |
| TCFD Recommendations | FSB/TCFD (2017, final 2023) | Ongoing (absorbed into ISSB) | Governance, strategy, risk management, metrics and targets; scenario analysis (1.5C/2C/4C) |

### 2.3 Verification and Assurance Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| ISO 14064-1:2018 | ISO | Organization-level GHG quantification and reporting; principles; boundary; data quality |
| ISO 14064-3:2019 | ISO | Specification for verification and validation of GHG statements; levels of assurance; evidence requirements |
| ISAE 3410 | IAASB (2012) | Assurance Engagements on GHG Statements; limited and reasonable assurance procedures |
| ISAE 3000 (Revised) | IAASB (2013) | General assurance standard for non-financial information; applicable to sustainability disclosures |
| AA1000AS v3 | AccountAbility (2020) | Stakeholder assurance standard; inclusivity, materiality, responsiveness, impact |
| PCAF Global GHG Accounting Standard | PCAF (2022) | Financed emissions methodology for financial institutions; data quality scoring (1-5) |

### 2.4 Supporting Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| SBTi FLAG Guidance | SBTi V1.1 (2022) | FLAG targets for >20% land use emissions; 11 commodity categories; no-deforestation commitment |
| SBTi SDA Tool | SBTi V3.0 (2024) | Sectoral Decarbonization Approach for 12 sectors; intensity convergence pathways |
| SBTi Temperature Rating | SBTi V2.0 (2024) | Company-level (1.0-6.0C); portfolio aggregation (WATS/TETS/MOTS/EOTS/ECOTS/AOTS) |
| GHG Management Hierarchy | WRI/WBCSD | Avoid > Reduce > Replace > Offset; informs enterprise decarbonization strategy |
| ISO 14068-1:2023 | ISO | Carbon neutrality quantification and reporting; PAS 2060 successor |
| VCMI Claims Code | VCMI (2023) | Net-zero and carbon neutral claim validation; Silver/Gold/Platinum tiers |
| Oxford Principles | Oxford (2020) | Offset quality hierarchy; shift toward permanent removals |
| IEA Net Zero Roadmap | IEA (2023) | Sector pathway benchmarks; milestones for 1.5C alignment |
| TPI Carbon Performance | TPI (2024) | Sector benchmarks; alignment assessment for 16 sectors |
| EU Taxonomy Climate Delegated Act | EU Reg 2021/2139 | Climate CapEx alignment; substantial contribution criteria |
| WBCSD Avoided Emissions Guidance | WBCSD (2023) | Scope 4 methodology; baseline scenario; attributional vs. consequential |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 8 | Enterprise-grade deterministic calculation engines |
| Workflows | 8 | Multi-phase orchestration workflows for enterprise processes |
| Templates | 10 | Enterprise report and dashboard templates |
| Integrations | 13 | Agent, app, ERP, assurance, and platform bridges |
| Presets | 8 | Sector-specific enterprise configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `enterprise_baseline_engine.py` | Financial-grade Scope 1+2+3 calculation across all 30 MRV agents with full activity-based methodology, data quality scoring per GHG Protocol hierarchy, and dual Scope 2 reporting. Supports all 15 Scope 3 categories with supplier-specific, hybrid, average-data, and spend-based calculation approaches. Produces comprehensive baseline with per-category data quality scores, confidence intervals, and materiality assessment. Target accuracy: +/-3%. |
| 2 | `sbti_target_engine.py` | Full SBTi Corporate Standard target setting covering near-term (C1-C28), long-term, and net-zero (NZ-C1 to NZ-C14) targets. Supports ACA (4.2%/yr 1.5C), SDA (12 sectors), and FLAG pathways. Validates coverage requirements (95% Scope 1+2, 67%/90% Scope 3). Generates submission-ready target documentation with annual milestones and five-year review schedule. |
| 3 | `scenario_modeling_engine.py` | Monte Carlo pathway analysis with 10,000+ simulation runs across 1.5C, 2C, and BAU scenarios. Models technology adoption curves, policy scenarios, energy price trajectories, and carbon price evolution. Produces probability distributions for emission outcomes, investment requirements, and stranded asset risk. Sensitivity analysis identifies top 10 drivers of outcome uncertainty. |
| 4 | `carbon_pricing_engine.py` | Internal carbon price management ($50-$200/tCO2e with escalation trajectories). Applies shadow pricing to capital allocation decisions, calculates carbon-adjusted NPV for investment cases, allocates carbon costs to business units and product lines, and models carbon border adjustment mechanism (CBAM) exposure. Supports both implicit (regulation-driven) and explicit (voluntary) carbon pricing. |
| 5 | `scope4_avoided_emissions_engine.py` | Quantifies avoided emissions from products and services that displace higher-emission alternatives. Implements WBCSD Avoided Emissions Guidance with baseline scenario definition, attributional calculation, conservative estimation principles, and double-counting prevention. Covers product substitution, efficiency improvements, enabling effects, and systemic change contributions. |
| 6 | `supply_chain_mapping_engine.py` | Multi-tier supplier mapping (Tier 1 through Tier 5) with engagement tracking for 100,000+ suppliers. Categorizes suppliers into 4 engagement tiers (inform, engage, require, collaborate). Tracks supplier SBTi adoption, CDP disclosure rates, emission reduction progress, and data quality improvement. Identifies Scope 3 hotspots by supplier, category, geography, and commodity. Integrates with CDP Supply Chain, EcoVadis, SEDEX, and WBCSD PACT data exchange. |
| 7 | `multi_entity_consolidation_engine.py` | Consolidates GHG data across 100+ subsidiaries, joint ventures, and associates using financial control, operational control, or equity share approaches per GHG Protocol. Handles intercompany transaction elimination, partial ownership adjustments, mid-year acquisitions/divestitures with pro-rata allocation, and base year recalculation triggers (5% significance threshold). Produces consolidated and entity-level reports aligned with financial reporting boundaries. |
| 8 | `financial_integration_engine.py` | Integrates carbon data into financial reporting workflows. Allocates emissions to P&L line items (carbon cost of goods sold, carbon SG&A), calculates EBITDA carbon intensity, tracks carbon asset values (allowances, credits, PPAs), models carbon liability exposure (ETS, CBAM, litigation), and produces carbon-adjusted financial metrics for investor communication. Supports ESRS E1-8 (internal carbon pricing) and E1-9 (anticipated financial effects of climate change). |

### 3.3 Engine Specifications

#### 3.3.1 Engine 1: Enterprise Baseline Engine

**Purpose:** Calculate a financial-grade GHG baseline across Scope 1, 2, and all 15 Scope 3 categories for large enterprises with complex organizational structures.

**Data Quality Framework (5-level hierarchy per GHG Protocol):**

| Level | Data Type | Example | Accuracy | Scope 3 Application |
|-------|-----------|---------|----------|---------------------|
| 1 | Supplier-specific, verified | Supplier CDP disclosure, PACT data exchange, product-level PCF | +/-3% | Best: use for top 50 suppliers by spend |
| 2 | Supplier-specific, unverified | Supplier self-reported, unaudited questionnaire | +/-5-10% | Good: use for next 200 suppliers |
| 3 | Average data (physical) | Industry average emission factor per unit of product | +/-10-20% | Acceptable: use for remaining suppliers with quantity data |
| 4 | Spend-based (EEIO) | Economic input-output emission factor per $ spend | +/-20-40% | Minimum: use when only spend data available |
| 5 | Proxy/extrapolation | Revenue-based proxy, headcount-based estimation | +/-40-60% | Last resort: use for immaterial categories only |

**Scope 1 Calculation (all 8 MRV agents):**

| Source | MRV Agent | Methodology | Key Data |
|--------|-----------|-------------|----------|
| Stationary combustion | MRV-001 | Fuel-based: quantity x NCV x emission factor | Fuel invoices, meter readings |
| Refrigerants & F-gas | MRV-002 | Mass balance or simplified: charge x leak rate x GWP | Refrigerant logs, service records |
| Mobile combustion | MRV-003 | Fuel-based or distance-based | Fleet fuel cards, telematics |
| Process emissions | MRV-004 | Process-specific factors (cement, chemicals, metals) | Production volumes, process parameters |
| Fugitive emissions | MRV-005 | Component count, emission factors, or LDAR data | Equipment inventory, inspection data |
| Land use emissions | MRV-006 | IPCC tier 1/2/3 for land use changes | Land area, land use type, biomass |
| Waste treatment | MRV-007 | Waste type x treatment method x emission factor | Waste manifests, treatment records |
| Agricultural emissions | MRV-008 | Enteric fermentation, manure management, soil N2O | Livestock numbers, fertilizer use |

**Scope 2 Calculation (dual reporting per Scope 2 Guidance):**

| Method | MRV Agent | Methodology | Data |
|--------|-----------|-------------|------|
| Location-based | MRV-009 | Grid average emission factor x kWh | Electricity invoices, grid factors by country/region |
| Market-based | MRV-010 | Contractual instrument factor x kWh | PPAs, RECs/GOs, green tariffs, residual mix |
| Steam/heat purchase | MRV-011 | Steam/heat factor x MWh | District heating invoices |
| Cooling purchase | MRV-012 | Cooling factor x MWh | District cooling invoices |
| Dual reporting reconciliation | MRV-013 | Location vs. market delta analysis | Both methods calculated |

**Scope 3 Calculation (all 15 categories):**

| Cat | Category | MRV Agent | Primary Method (Enterprise) | Fallback Method | Typical % of Total |
|-----|----------|-----------|---------------------------|-----------------|-------------------|
| 1 | Purchased goods & services | MRV-014 | Supplier-specific (CDP/PACT) + hybrid | Spend-based EEIO | 30-50% |
| 2 | Capital goods | MRV-015 | Average-data per asset type | Spend-based EEIO | 2-8% |
| 3 | Fuel & energy activities | MRV-016 | WTT and T&D factors from Scope 1+2 data | Auto-calculated | 3-8% |
| 4 | Upstream transportation | MRV-017 | Distance-based: tonne-km x mode factor | Spend-based | 3-10% |
| 5 | Waste generated | MRV-018 | Waste-type-specific: tonnes x treatment factor | Average waste/employee | 0.5-2% |
| 6 | Business travel | MRV-019 | Distance-based: passenger-km x mode x class | Spend-based | 1-5% |
| 7 | Employee commuting | MRV-020 | Survey-based: distance x mode x frequency | Average commute model | 1-3% |
| 8 | Upstream leased assets | MRV-021 | Asset-specific: area x energy intensity | Lease value proxy | 0.5-5% |
| 9 | Downstream transportation | MRV-022 | Distance-based: tonne-km x mode | Revenue-based proxy | 2-8% |
| 10 | Processing of sold products | MRV-023 | Process-specific: mass x processing factor | Industry average | 0-15% |
| 11 | Use of sold products | MRV-024 | Product-specific: energy x lifetime x units | Average use profile | 0-40% |
| 12 | End-of-life treatment | MRV-025 | Waste-type: mass x treatment pathway x factor | Average disposal | 1-5% |
| 13 | Downstream leased assets | MRV-026 | Asset-specific: area x energy intensity | Lease value proxy | 0-5% |
| 14 | Franchises | MRV-027 | Franchise-specific: Scope 1+2 per franchisee | Average franchise profile | 0-10% |
| 15 | Investments | MRV-028 | PCAF methodology: financed emissions by asset class | Revenue-based proxy | 0-80% (FIs) |

**Materiality Assessment:**
- Categories contributing >1% of total emissions: full activity-based calculation required
- Categories contributing 0.1-1%: average-data method acceptable
- Categories contributing <0.1%: may be excluded with documented justification
- Total exclusions must not exceed 5% of anticipated total Scope 3

**Key Models:**
- `EnterpriseBaselineConfig` -- Organizational boundary (entities, approach), reporting year, base year, gases included (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3), data quality targets
- `EnterpriseBaselineInput` -- Per-entity data packages (energy, fuel, fleet, procurement, travel, waste, process, land use, leased assets, investments)
- `EnterpriseBaselineResult` -- Total CO2e by scope and category, per-entity breakdown, data quality score matrix (per-category per-entity), confidence intervals, materiality assessment, year-over-year comparison
- `DataQualityMatrix` -- Per-category data quality level (1-5), per-entity data quality level, weighted average DQ score, improvement targets
- `MaterialityAssessment` -- Per-category materiality (% of total), inclusion/exclusion justification, coverage calculation for SBTi

#### 3.3.2 Engine 2: SBTi Target Engine

**Purpose:** Set targets aligned with the full SBTi Corporate Standard covering near-term, long-term, and net-zero requirements.

**Target Types Supported:**

| Target Type | Standard | Scope | Methodology | Ambition | Timeline |
|-------------|----------|-------|-------------|----------|----------|
| Near-term (1.5C) | Corporate Manual V5.3 | Scope 1+2 (95%) + Scope 3 (67%) | ACA: 4.2%/yr absolute reduction | 1.5C-aligned | 5-10 years |
| Near-term (WB2C) | Corporate Manual V5.3 | Scope 1+2 (95%) + Scope 3 (67%) | ACA: 2.5%/yr absolute reduction | Well-below 2C | 5-10 years |
| Near-term (SDA) | Corporate Manual V5.3 | Scope 1+2 (95%) | Sector intensity convergence to IEA NZE | Sector-specific | 5-10 years |
| Long-term (net-zero) | Net-Zero Standard V1.3 | Scope 1+2 (95%) + Scope 3 (90%) | 90%+ absolute reduction from base year | Net-zero by 2050 | By 2050 |
| FLAG | FLAG Guidance V1.1 | FLAG emissions (if >20%) | 3.03%/yr FLAG pathway | 1.5C/no deforestation | By 2030 (deforestation), by 2050 (emissions) |

**28 Near-Term Criteria (C1-C28) Automated Validation:**

| Criterion Group | Criteria | Validation Logic |
|----------------|----------|------------------|
| Boundary & Coverage | C1-C5 | Entity coverage >= 95% Scope 1+2; Scope 3 >= 67%; boundary consistent with financial reporting |
| Base Year | C6-C9 | Base year within 2 most recent completed years; no older than target submission year minus 2; recalculation policy defined |
| Target Ambition | C10-C15 | ACA >= 4.2%/yr (1.5C) or >= 2.5%/yr (WB2C); SDA convergence validated against sector benchmark; FLAG 3.03%/yr if applicable |
| Target Timeframe | C16-C18 | Near-term: 5-10 years from submission; no more than 10 years from base year |
| Scope 3 Target | C19-C23 | 67% coverage of total Scope 3; supplier engagement target (if applicable); all material categories included |
| Reporting & Disclosure | C24-C28 | Annual disclosure commitment; progress tracking methodology; recalculation triggers defined |

**14 Net-Zero Criteria (NZ-C1 to NZ-C14) Automated Validation:**

| Criterion Group | Criteria | Validation Logic |
|----------------|----------|------------------|
| Long-term Target | NZ-C1 to NZ-C4 | 90%+ absolute reduction by 2050; Scope 1+2 coverage >= 95%; Scope 3 coverage >= 90% |
| Neutralization | NZ-C5 to NZ-C8 | Residual emissions <= 10%; neutralization via permanent CDR; credit quality per SBTi guidance |
| Interim Milestones | NZ-C9 to NZ-C11 | Near-term target set (C1-C28 satisfied); interim milestones every 5 years; linear or front-loaded pathway |
| Governance | NZ-C12 to NZ-C14 | Board-level oversight; annual progress reporting; five-year review and revalidation |

**SDA Sector Pathways (12 sectors):**

| Sector | Intensity Metric | 2030 Target | 2050 Target | Source |
|--------|-----------------|-------------|-------------|--------|
| Power generation | tCO2/MWh | 0.14 | 0.00 | IEA NZE |
| Cement | tCO2/t cement | 0.42 | 0.07 | SBTi SDA Tool |
| Iron & steel | tCO2/t crude steel | 1.06 | 0.05 | SBTi SDA Tool |
| Aluminium | tCO2/t aluminium | 3.10 | 0.20 | SBTi SDA Tool |
| Pulp & paper | tCO2/t product | 0.22 | 0.04 | SBTi SDA Tool |
| Chemicals | tCO2/t product | Varies | Varies | SBTi SDA Tool |
| Aviation | gCO2/pkm | 62.0 | 8.0 | SBTi SDA Tool |
| Maritime shipping | gCO2/tkm | 5.8 | 0.8 | SBTi SDA Tool |
| Road transport | gCO2/vkm | 85.0 | 0.0 | SBTi SDA Tool |
| Commercial buildings | kgCO2/sqm | 25.0 | 2.0 | SBTi SDA Tool |
| Residential buildings | kgCO2/sqm | 12.0 | 1.0 | SBTi SDA Tool |
| Food & beverage | tCO2/t product | Varies | Varies | SBTi SDA Tool |

**Key Models:**
- `SBTiTargetConfig` -- Pathway type (ACA/SDA/FLAG), ambition level (1.5C/WB2C), base year, target year, sector classification
- `SBTiTargetInput` -- Baseline result, entity list, sector allocation, FLAG assessment, prior targets (if revalidating)
- `SBTiTargetResult` -- Near-term target definition, long-term target definition, net-zero target, annual milestone pathway, criteria validation matrix (42 criteria with pass/fail/warning), submission readiness score, estimated time-to-validation
- `CriteriaValidation` -- Per-criterion assessment with pass/fail/warning, evidence reference, remediation guidance
- `TargetPathway` -- Annual absolute and intensity targets from base year to target year and through 2050

#### 3.3.3 Engine 3: Scenario Modeling Engine

**Purpose:** Model multiple decarbonization scenarios with Monte Carlo uncertainty analysis for enterprise strategic planning.

**Scenario Framework:**

| Scenario | Temperature Alignment | Key Assumptions | Typical Use |
|----------|---------------------|-----------------|-------------|
| Aggressive (1.5C) | 1.5C by 2100 | Rapid electrification, 100% RE by 2035, aggressive supplier engagement, high carbon price ($150+/tCO2e by 2030) | Stretch target, investor communication |
| Moderate (2C) | Well-below 2C | Steady transition, 80% RE by 2035, moderate supplier engagement, medium carbon price ($75-100/tCO2e by 2030) | Base case for planning |
| Conservative (BAU) | 3-4C | Current policies only, no additional action, low carbon price ($25-50/tCO2e by 2030) | Risk scenario, stranded asset analysis |
| Custom | User-defined | Configurable technology adoption rates, policy assumptions, carbon price trajectory | Specific investment case analysis |

**Monte Carlo Simulation Parameters:**

| Parameter | Distribution | Range | Source |
|-----------|-------------|-------|--------|
| Carbon price trajectory | Log-normal | $25-$300/tCO2e by 2030 | IEA, World Bank, NGFS scenarios |
| Electricity grid decarbonization rate | Beta | 2-8%/yr by country | IEA NZE, national plans |
| Technology adoption rate (EVs) | S-curve (logistic) | 20-80% by 2030 | BloombergNEF, IEA |
| Technology adoption rate (heat pumps) | S-curve (logistic) | 15-60% by 2030 | IEA, national plans |
| Renewable energy cost trajectory | Log-normal declining | LCOE reduction 3-8%/yr | IRENA, BNEF |
| Energy efficiency improvement rate | Normal | 1-4%/yr | IEA, historical trend |
| Supplier engagement success rate | Beta | 30-80% of suppliers setting SBTi | CDP Supply Chain data |
| Scope 3 data quality improvement | Linear | 0.5-2.0 DQ levels per year | Enterprise benchmark data |
| Regulatory stringency | Discrete (scenarios) | Current/enhanced/accelerated | NGFS scenarios |
| Physical climate risk | Varying by geography | RCP 2.6 / 4.5 / 8.5 | IPCC AR6, NGFS |

**Simulation Methodology:**
1. Define base case parameters from enterprise baseline and current trajectory
2. Specify probability distributions for each uncertain parameter
3. Generate 10,000 random samples from parameter distributions (Latin Hypercube Sampling for efficiency)
4. For each sample, calculate annual emissions trajectory through 2050
5. Compute aggregate statistics: mean, median, P10, P25, P75, P90 for each year
6. Identify key sensitivities via Sobol indices and tornado chart generation
7. Calculate probability of target achievement under each scenario

**Output Deliverables:**
- Fan chart showing emission trajectories with confidence bands (P10-P90)
- Tornado chart showing top 10 sensitivity drivers
- Probability of SBTi target achievement under each scenario
- Investment requirement distribution (P50 and P90 CapEx)
- Carbon budget consumption trajectory
- Stranded asset risk assessment by asset class

**Key Models:**
- `ScenarioConfig` -- Scenario definitions (1.5C/2C/BAU/custom), simulation parameters, Monte Carlo run count, confidence intervals
- `ScenarioInput` -- Baseline result, current portfolio, planned actions, technology adoption assumptions, policy assumptions
- `ScenarioResult` -- Per-scenario emission trajectories (annual, with P10/P25/P50/P75/P90), sensitivity analysis, probability of target achievement, investment requirements, carbon budget analysis
- `MonteCarloRun` -- Single simulation run with all sampled parameters and resulting trajectory
- `SensitivityAnalysis` -- Sobol first-order and total indices for each parameter, tornado chart data

#### 3.3.4 Engine 4: Carbon Pricing Engine

**Purpose:** Implement and manage internal carbon pricing across the enterprise for capital allocation, performance management, and regulatory preparation.

**Carbon Pricing Approaches:**

| Approach | Description | Typical Price Range | Application |
|----------|-------------|--------------------|-|
| Shadow price | Hypothetical carbon cost applied to investment decisions | $50-$200/tCO2e | CapEx appraisal, project selection |
| Internal carbon fee | Actual charge to business units based on emissions | $25-$100/tCO2e | P&L allocation, behavior change |
| Implicit carbon price | Derived from actual abatement costs and investments | Varies by action | Bottom-up cost analysis |
| Regulatory carbon price | Actual or expected regulatory cost (ETS, CBAM, carbon tax) | Market price | Compliance cost modeling |

**Price Trajectory Modeling:**

| Year | Low Scenario | Medium Scenario | High Scenario | Source Benchmark |
|------|-------------|----------------|--------------|-----------------|
| 2025 | $30/tCO2e | $60/tCO2e | $100/tCO2e | EU ETS ~$60-80 |
| 2030 | $50/tCO2e | $100/tCO2e | $200/tCO2e | IEA NZE: $130 |
| 2035 | $75/tCO2e | $150/tCO2e | $300/tCO2e | NGFS scenarios |
| 2040 | $100/tCO2e | $200/tCO2e | $400/tCO2e | NGFS scenarios |
| 2050 | $150/tCO2e | $300/tCO2e | $500/tCO2e | IEA NZE: $250 |

**Carbon-Adjusted Financial Metrics:**

| Metric | Calculation | Use |
|--------|-------------|-----|
| Carbon-adjusted NPV | Standard NPV minus PV of carbon costs over project life | Investment appraisal |
| Carbon cost of goods sold | Product-level emission intensity x internal carbon price | Product margin analysis |
| Carbon-adjusted EBITDA | EBITDA minus total carbon charge | Performance reporting |
| Carbon intensity of revenue | tCO2e per $M revenue x carbon price | Investor communication |
| Carbon liability | Forward-looking ETS/CBAM exposure at projected prices | Financial risk |
| Carbon asset value | Carbon credits, allowances, PPAs at current/forward price | Balance sheet |
| CBAM exposure | Imported goods emissions x CBAM certificate price | Trade exposure |

**Business Unit Allocation:**
- Allocation based on measured Scope 1+2 emissions per BU
- Scope 3 allocation by procurement spend, revenue, or activity-based drivers
- Intercompany carbon charges for shared services (data centers, fleet, logistics)
- Performance benchmarking: carbon intensity per BU vs. target and peer

**Key Models:**
- `CarbonPricingConfig` -- Pricing approach, price level, escalation trajectory, allocation methodology, BU structure
- `CarbonPricingInput` -- Enterprise baseline by BU, investment proposals, product portfolio, regulatory exposure
- `CarbonPricingResult` -- Carbon-adjusted financials by BU, investment ranking with/without carbon price, CBAM exposure, carbon P&L, carbon balance sheet items
- `InvestmentAppraisal` -- Per-project carbon-adjusted NPV, IRR, payback with and without carbon price
- `CBAMExposure` -- Per-product CBAM certificate cost by import origin and embedded emissions

#### 3.3.5 Engine 5: Scope 4 Avoided Emissions Engine

**Purpose:** Quantify avoided emissions from products and services that displace higher-emission alternatives, following WBCSD Avoided Emissions Guidance.

**Avoided Emissions Framework:**

| Category | Description | Example | Methodology |
|----------|-------------|---------|-------------|
| Product substitution | Product displaces a higher-emission alternative | EV displacing ICE vehicle | Baseline emissions (ICE) minus product emissions (EV) x units sold |
| Efficiency improvement | Product enables energy/resource efficiency | LED lighting, insulation, efficient HVAC | Baseline energy use minus product energy use x units sold x lifetime |
| Enabling effect | Product/service enables emission reductions by others | Teleconferencing reducing travel, smart grid reducing curtailment | Counterfactual travel emissions minus actual emissions post-adoption |
| Systemic change | Product contributes to systemic low-carbon transition | Renewable energy equipment, carbon capture technology | System-level emission reduction attributed to product contribution |

**Calculation Methodology (WBCSD-aligned):**

```
Avoided Emissions = (Baseline_Emissions - Product_Lifecycle_Emissions) x Units_Sold

Where:
  Baseline_Emissions = Reference product/service emissions over equivalent functional unit
  Product_Lifecycle_Emissions = Full lifecycle emissions of the assessed product (cradle-to-grave)
  Units_Sold = Number of units sold/deployed in reporting year

Conservative Principles:
  1. Baseline must be market average or regulatory minimum (not worst-case)
  2. Full lifecycle of assessed product included (not just use-phase savings)
  3. Rebound effects quantified and deducted
  4. Attribution share applied for enabling effects (not 100% credit)
  5. Double-counting with Scope 3 avoided through clear boundary definition
```

**Reporting Requirements:**
- Avoided emissions reported separately from Scope 1, 2, 3 (never netted against footprint)
- Methodology and baseline assumptions fully documented
- Uncertainty ranges provided (P10-P90)
- Time horizon and decay assumptions stated
- Comparison to total Scope 1+2+3 footprint for context

**Key Models:**
- `AvoidedEmissionsConfig` -- Product/service categories, baseline scenarios, attribution methodology, conservative principles
- `AvoidedEmissionsInput` -- Product inventory, sales volumes, baseline emission factors, product lifecycle assessment data
- `AvoidedEmissionsResult` -- Total avoided emissions by product/service category, methodology documentation, uncertainty ranges, ratio to footprint
- `BaselineScenario` -- Market average or regulatory minimum baseline with justification
- `AttributionShare` -- Attribution methodology for multi-party enabling effects

#### 3.3.6 Engine 6: Supply Chain Mapping Engine

**Purpose:** Map, analyze, and engage enterprise supply chains across multiple tiers for Scope 3 reduction.

**Supplier Tiering Model:**

| Tier | Definition | Typical Count (Enterprise) | Data Approach | Engagement Level |
|------|-----------|--------------------------|--------------|-----------------|
| Tier 1 (Critical) | Top 50 suppliers by Scope 3 contribution (typically 50-70% of Scope 3) | 50 | Supplier-specific (CDP, PACT, direct engagement) | Collaborate: joint reduction targets, data sharing, innovation |
| Tier 2 (Strategic) | Next 200 suppliers (next 15-25% of Scope 3) | 200 | Supplier-reported (questionnaire) + verification | Require: mandate SBTi commitment, annual CDP disclosure |
| Tier 3 (Managed) | Next 1,000 suppliers (next 5-10% of Scope 3) | 1,000 | Average-data + periodic questionnaire | Engage: encourage CDP disclosure, provide tools |
| Tier 4 (Monitored) | Remaining suppliers (long tail, 5-10%) | 10,000+ | Spend-based EEIO | Inform: communicate expectations, provide resources |
| Sub-tier (Upstream) | Tier 2-5 suppliers of Tier 1 suppliers | Varies | CDP Supply Chain cascade, industry databases | Cascade: Tier 1 suppliers engage their own suppliers |

**Engagement Program Design:**

| Stage | Activities | Metrics | Timeline |
|-------|-----------|---------|----------|
| Awareness | Send climate expectations letter, share resources | Letter sent rate (target: 100% Tier 1-3) | Month 1-3 |
| Measurement | Request CDP/GHG data, provide templates | Response rate (target: 80% Tier 1-2, 50% Tier 3) | Month 3-6 |
| Target Setting | Encourage SBTi commitment, support target design | SBTi commitment rate (target: 50% Tier 1, 25% Tier 2) | Month 6-12 |
| Reduction | Joint projects, innovation partnerships, procurement incentives | Year-over-year emission reduction, data quality improvement | Ongoing |

**Hotspot Analysis:**
- Rank suppliers by absolute Scope 3 contribution (tCO2e)
- Rank product categories by emission intensity (tCO2e/unit or tCO2e/$M)
- Identify geographic hotspots (high grid-factor regions, deforestation-risk countries)
- Cross-reference with commodity risk (EUDR, conflict minerals, forced labor)
- Prioritize engagement by influence (procurement spend) x impact (emission contribution)

**CDP Supply Chain Integration:**
- Automated request management for CDP Climate Change questionnaire
- Score analysis: A-list, Management, Awareness, Disclosure, Not Disclosed
- Engagement-to-disclosure ratio tracking
- Supplier action plan extraction from CDP responses
- Year-over-year supplier score improvement tracking

**Key Models:**
- `SupplyChainConfig` -- Tiering thresholds, engagement program design, data collection methodology, integration settings
- `SupplyChainInput` -- Supplier master data, procurement spend, CDP scores, questionnaire responses, prior year engagement data
- `SupplyChainResult` -- Supplier tier assignment, Scope 3 hotspot map, engagement program status, supplier scorecards, aggregate progress metrics
- `SupplierScorecard` -- Per-supplier: tier, emissions, DQ score, SBTi status, CDP score, year-over-year change, engagement actions
- `EngagementProgram` -- Program design, participation tracking, impact measurement, escalation triggers

#### 3.3.7 Engine 7: Multi-Entity Consolidation Engine

**Purpose:** Consolidate GHG data across 100+ entities using GHG Protocol consolidation approaches with intercompany elimination.

**Consolidation Approaches (per GHG Protocol Corporate Standard Chapter 3):**

| Approach | Rule | When to Use | Typical Delta vs. Others |
|----------|------|-------------|-------------------------|
| Financial control | Include 100% of emissions from entities where company has financial control (ability to direct financial and operating policies) | IFRS reporting basis; aligns with financial statements | Often highest total (includes all consolidated subs) |
| Operational control | Include 100% of emissions from entities where company has operational control (authority to introduce operating policies) | US GAAP preference; excludes JVs where not operator | May exclude significant JV emissions |
| Equity share | Include emissions proportional to equity ownership % | Most conservative; required by some regulators | Often lowest total; includes partial JV/associate shares |

**Multi-Entity Data Model:**

```
Corporate Group
  +-- Entity 1 (100% owned subsidiary) -- full consolidation
  +-- Entity 2 (100% owned subsidiary) -- full consolidation
  |   +-- Entity 2a (80% owned sub-subsidiary) -- full or 80% equity
  +-- Entity 3 (51% owned JV) -- depends on control approach
  +-- Entity 4 (33% associate) -- equity share only (not control approaches)
  +-- Entity 5 (acquired mid-year) -- pro-rata from acquisition date
  +-- Entity 6 (divested mid-year) -- pro-rata until divestiture date
```

**Intercompany Elimination:**
- Identify intercompany transactions (energy supply, shared services, internal logistics)
- Eliminate double-counted emissions from Scope 3 where one entity's Scope 3 Cat 1 is another entity's Scope 1
- Track and document all elimination entries with justification
- Reconcile consolidated GHG total with sum of entity-level totals minus eliminations

**Base Year Recalculation Triggers (GHG Protocol):**
- Structural change (M&A, divestiture) exceeding 5% significance threshold
- Methodology change (new emission factors, calculation approach)
- Discovery of significant error (>5% impact)
- Change in organizational boundary or consolidation approach
- Outsourcing/insourcing of emitting activities exceeding 5% threshold

**Recalculation Methodology:**
1. Assess significance: does the trigger event change base year emissions by >5%?
2. If yes: recalculate base year as if current structure/methodology had been in place
3. Adjust all interim years consistently
4. Document recalculation with full audit trail (old base year, new base year, trigger event, delta)
5. Report restated figures alongside original in trend analysis

**Key Models:**
- `ConsolidationConfig` -- Approach (financial/operational/equity), entity hierarchy, ownership percentages, control assessments, intercompany relationships
- `ConsolidationInput` -- Per-entity baseline results, intercompany transaction register, acquisition/divestiture events, base year recalculation triggers
- `ConsolidationResult` -- Consolidated emissions by scope and category, per-entity contribution, intercompany eliminations, reconciliation to entity-level sum, base year restated (if applicable)
- `EntityHierarchy` -- Tree structure of entities with ownership %, control type, consolidation method, reporting currency
- `BaseYearRecalculation` -- Trigger event, significance assessment, old vs. new base year values, restated trend data

#### 3.3.8 Engine 8: Financial Integration Engine

**Purpose:** Integrate carbon data into enterprise financial reporting and analysis workflows.

**P&L Carbon Allocation:**

| P&L Line Item | Carbon Allocation Methodology | Data Source |
|---------------|------------------------------|-------------|
| Revenue | Carbon intensity per $M revenue (tCO2e/$M) | Total Scope 1+2+3 / revenue |
| Cost of Goods Sold | Scope 1 (manufacturing) + Scope 2 (production energy) + Scope 3 Cat 1 (materials) + Cat 4 (inbound logistics) | Activity-based per product line |
| Gross Profit | Carbon intensity of gross profit | Derived |
| SG&A | Scope 2 (office energy) + Scope 3 Cat 6 (travel) + Cat 7 (commuting) + Cat 5 (waste) | Activity-based per function |
| R&D | Scope 2 (lab energy) + Scope 3 Cat 1 (R&D procurement) | Activity-based |
| EBITDA | Carbon-adjusted EBITDA = EBITDA - (total emissions x internal carbon price) | Derived |
| CapEx | Carbon impact of capital investments (positive or negative) | Per-project assessment |

**Carbon Balance Sheet Items:**

| Item | Description | Valuation Basis |
|------|-------------|----------------|
| Carbon allowances (ETS) | Free allocation + purchased allowances | Market price (EU ETS, UK ETS, etc.) |
| Carbon credits (voluntary) | Purchased voluntary credits (VCS, Gold Standard, etc.) | Purchase price + market value |
| Renewable energy certificates | RECs, GOs, I-RECs | Market price or PPA contract value |
| Carbon liability (current) | Current year ETS/carbon tax obligation | Regulatory rate x emissions |
| Carbon liability (long-term) | Forward-looking compliance cost at projected carbon prices | Discounted projected cost |
| Stranded asset risk | Assets at risk of write-down due to transition (fossil fuel reserves, high-carbon infrastructure) | Scenario-dependent valuation |
| Green bond proceeds | Proceeds allocated to eligible green projects | Bond terms |
| EU Taxonomy CapEx | Climate-aligned CapEx as % of total | Taxonomy screening |

**ESRS E1 Alignment:**

| ESRS Datapoint | Engine Output |
|----------------|---------------|
| E1-8: Internal carbon pricing | Carbon price level, scope of application, revenue generated, methodology |
| E1-9: Anticipated financial effects | Physical risk exposure (acute/chronic), transition risk exposure (policy, technology, market, reputation), climate opportunities |

**Key Models:**
- `FinancialIntegrationConfig` -- P&L structure, BU hierarchy, carbon price, allocation methodology, regulatory positions (ETS, CBAM)
- `FinancialIntegrationInput` -- Enterprise baseline, financial data (P&L, balance sheet, CapEx plan), carbon price trajectory, regulatory positions
- `FinancialIntegrationResult` -- Carbon-adjusted P&L, carbon balance sheet, EBITDA carbon intensity, carbon cost allocation by BU/product, CBAM exposure, EU Taxonomy CapEx alignment, ESRS E1-8/E1-9 disclosures
- `CarbonPnL` -- P&L with carbon allocation per line item
- `CarbonBalanceSheet` -- Carbon assets, liabilities, and net position

### 3.4 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `comprehensive_baseline_workflow.py` | 6: EntityMapping -> DataCollection -> QualityAssurance -> Calculation -> Consolidation -> Reporting | Full GHG inventory across all entities and all 15 Scope 3 categories. Phase 1 maps organizational boundary and entity hierarchy. Phase 2 orchestrates data collection from ERP systems and manual uploads per entity. Phase 3 runs data quality profiling, duplicate detection, outlier analysis, and gap filling. Phase 4 runs enterprise_baseline_engine for each entity. Phase 5 runs multi_entity_consolidation_engine. Phase 6 generates consolidated baseline report. Typical duration: 6-12 weeks. |
| 2 | `sbti_submission_workflow.py` | 5: BaselineValidation -> PathwaySelection -> TargetDefinition -> CriteriaValidation -> SubmissionPackage | Complete SBTi target submission preparation. Phase 1 validates baseline data quality meets SBTi requirements. Phase 2 evaluates ACA vs. SDA vs. FLAG pathway suitability. Phase 3 defines near-term, long-term, and net-zero targets with annual milestones. Phase 4 validates all 42 criteria (C1-C28 + NZ-C1 to NZ-C14). Phase 5 generates submission-ready documentation package. |
| 3 | `annual_inventory_workflow.py` | 5: DataRefresh -> Calculation -> BaseYearCheck -> Consolidation -> AnnualReport | Annual recalculation with base year adjustment review. Phase 1 refreshes data from ERP and other sources. Phase 2 recalculates current year emissions. Phase 3 checks base year recalculation triggers (M&A, methodology changes, errors). Phase 4 consolidates across entities. Phase 5 generates annual GHG inventory report with year-over-year and base-year-to-current comparison. |
| 4 | `scenario_analysis_workflow.py` | 5: ParameterSetup -> Simulation -> Sensitivity -> Comparison -> StrategyReport | Full scenario analysis for board-level strategic planning. Phase 1 defines scenario parameters and uncertainty distributions. Phase 2 runs Monte Carlo simulation (10,000 runs per scenario). Phase 3 performs sensitivity analysis (Sobol indices, tornado charts). Phase 4 compares scenarios (1.5C vs. 2C vs. BAU). Phase 5 generates executive strategy report with investment recommendations. |
| 5 | `supply_chain_engagement_workflow.py` | 5: SupplierMapping -> Tiering -> ProgramDesign -> Execution -> ImpactMeasurement | End-to-end supplier engagement program. Phase 1 maps suppliers to Scope 3 categories and tiers. Phase 2 assigns supplier engagement tiers (inform/engage/require/collaborate). Phase 3 designs engagement program with milestones and KPIs. Phase 4 tracks engagement activities (letters sent, questionnaires received, SBTi commitments). Phase 5 measures Scope 3 reduction impact from supplier engagement. |
| 6 | `internal_carbon_pricing_workflow.py` | 4: PriceDesign -> AllocationSetup -> ImpactAnalysis -> Reporting | Implement and report on internal carbon pricing. Phase 1 designs carbon pricing approach (shadow price vs. internal fee) with price level and escalation. Phase 2 configures allocation to business units and product lines. Phase 3 analyzes impact on investment decisions, product margins, and BU performance. Phase 4 generates carbon pricing report and ESRS E1-8 disclosure. |
| 7 | `multi_entity_rollup_workflow.py` | 5: EntityRefresh -> DataValidation -> EntityCalculation -> Elimination -> ConsolidatedReport | Consolidate 100+ entities with intercompany elimination. Phase 1 refreshes entity hierarchy (new entities, divestitures, ownership changes). Phase 2 validates per-entity data completeness and quality. Phase 3 calculates per-entity emissions. Phase 4 performs intercompany elimination (Scope 3 Cat 1 vs. Scope 1 overlap). Phase 5 generates consolidated report with per-entity breakdown and reconciliation. |
| 8 | `external_assurance_workflow.py` | 5: ScopeDefinition -> EvidenceCollection -> WorkpaperGeneration -> ControlTesting -> AssurancePackage | Prepare for ISO 14064-3 / ISAE 3410 external assurance. Phase 1 defines assurance scope (limited vs. reasonable) and boundary. Phase 2 collects all evidence (source data, methodology documentation, calculation traces, control documentation). Phase 3 generates audit workpapers per Big 4 format requirements. Phase 4 performs pre-assurance control testing (reconciliation, analytical review, sample testing). Phase 5 produces assurance-ready package with management assertion letter template. |

### 3.5 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `ghg_inventory_report.py` | MD, HTML, JSON, XLSX | Full GHG Protocol Corporate Standard report with Scope 1/2/3 breakdown, entity-level detail, data quality scores, methodology notes, and year-over-year trends. Conforms to GHG Protocol reporting requirements (Chapter 9). 20-40 page enterprise report. |
| 2 | `sbti_target_submission.py` | MD, HTML, JSON, PDF | Complete SBTi submission package with near-term + long-term + net-zero targets, criteria validation matrix (42 criteria), pathway visualization, coverage analysis, and supporting documentation. Formatted per SBTi submission template. |
| 3 | `cdp_climate_response.py` | MD, HTML, JSON | Full CDP Climate Change questionnaire response covering all modules (C0-C15). Auto-populates from pack data: C4 (targets), C5 (emissions methodology), C6 (emissions data), C7 (energy), C8 (energy expenditures), C12 (engagement), C15 (biodiversity). Maximizes scoring potential toward A-list. |
| 4 | `tcfd_report.py` | MD, HTML, JSON, PDF | Complete TCFD/ISSB S2 disclosure covering all four pillars: Governance (board oversight, management role), Strategy (climate risks/opportunities, scenario analysis, financial impact), Risk Management (identification, assessment, management processes), Metrics and Targets (Scope 1/2/3, intensity metrics, targets, progress). |
| 5 | `executive_dashboard.py` | MD, HTML, JSON | Board-level climate dashboard with 15-20 key KPIs, traffic-light status indicators, trend sparklines, peer benchmarking, and strategic commentary. Designed for quarterly board reporting. Single-page executive summary with drill-down capability. |
| 6 | `supply_chain_heatmap.py` | MD, HTML, JSON | Tier 1/2/3 supplier emissions heatmap by geography, commodity, and engagement status. Includes supplier scorecards for top 50 by emissions, CDP score distribution, SBTi adoption tracking, and year-over-year improvement trends. Identifies hotspots requiring immediate attention. |
| 7 | `scenario_comparison.py` | MD, HTML, JSON | 1.5C vs. 2C vs. BAU scenario comparison with fan charts (P10-P90), tornado sensitivity analysis, investment requirement comparison, probability of target achievement, and carbon budget consumption. Board-ready strategic decision document. |
| 8 | `assurance_statement.py` | MD, HTML, JSON, PDF | ISO 14064-3 assurance statement template (limited and reasonable assurance versions) with management assertion, scope of engagement, criteria, findings, and conclusion. Pre-populated with enterprise data; designed for external auditor review and customization. |
| 9 | `board_climate_report.py` | MD, HTML, JSON, PDF | Quarterly board climate report covering: (a) emission performance vs. target pathway, (b) key initiatives status (energy, fleet, procurement), (c) regulatory compliance update, (d) carbon pricing impact, (e) supply chain engagement progress, (f) upcoming milestones, (g) risk assessment. 5-10 page board paper format. |
| 10 | `regulatory_filings.py` | MD, HTML, JSON, PDF, XLSX | Multi-framework regulatory filing templates: SEC Climate Rule disclosure (S-X Article 14), CSRD ESRS E1 climate change chapter, California SB 253 emission report, ISO 14064-1 GHG statement, and CDP questionnaire extract. Each template auto-maps enterprise data to framework-specific format and datapoint requirements. |

### 3.6 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `sap_connector.py` | SAP S/4HANA integration for procurement data (MM module), finance data (FI/CO module), logistics data (SD/LE modules), plant maintenance data (PM module), and human resources data (HCM module). Extracts activity data for Scope 1 (fuel purchases, fleet), Scope 2 (electricity invoices), Scope 3 Cat 1 (procurement volumes and spend), Cat 4 (transportation), Cat 6 (travel bookings via Concur), and Cat 7 (employee data). Supports SAP Sustainability Control Tower data exchange. |
| 2 | `oracle_connector.py` | Oracle ERP Cloud integration covering Procurement Cloud (supplier spend, PO data), Financial Cloud (GL carbon allocation), SCM Cloud (logistics and transportation), and HCM Cloud (employee data for commuting). Extracts activity data for all material Scope 3 categories. Supports Oracle Sustainability Hub data exchange. |
| 3 | `workday_connector.py` | Workday HCM integration for employee data (headcount by location, commute surveys, remote work status), travel and expense data (business travel bookings, expense reports), and financial data (budget allocation for carbon initiatives). Primarily feeds Scope 3 Cat 6 (travel) and Cat 7 (commuting) calculations. |
| 4 | `cdp_bridge.py` | Automated CDP Climate Change questionnaire response generation. Maps enterprise data to CDP modules (C0-C15). Pulls prior year CDP responses for year-over-year consistency. Integrates CDP Supply Chain data for supplier emission collection. Targets A-list scoring through optimized response strategy. |
| 5 | `sbti_bridge.py` | Bridge to GL-SBTi-APP for target submission and validation. Manages SBTi commitment registration, target submission, validation tracking, progress reporting, and five-year review cycle. Exports data in SBTi target submission template format. |
| 6 | `assurance_provider_bridge.py` | Interface for Big 4 and specialist assurance providers (Deloitte, EY, KPMG, PwC, SGS, Bureau Veritas, DNV). Generates audit workpapers in provider-preferred format. Manages evidence requests, sample selection, management representation letters, and finding remediation tracking. Supports both limited and reasonable assurance engagements per ISAE 3410/3000. |
| 7 | `multi_entity_orchestrator.py` | Manages 100+ entity hierarchy for multi-entity consolidation. Handles entity onboarding/offboarding, ownership structure changes, consolidation approach configuration per entity, data collection scheduling per entity, and rollup orchestration. Maintains entity-level data completeness dashboard and escalation for overdue submissions. |
| 8 | `carbon_marketplace_bridge.py` | Integration with voluntary carbon credit marketplaces and registries. Supports Verra VCS, Gold Standard, American Carbon Registry, Climate Action Reserve, and Puro.earth for permanent CDR. Manages credit procurement, retirement, and portfolio tracking. Applies SBTi neutralization requirements (permanent CDR for residual emissions). Provides credit quality scoring per VCMI Claims Code and Oxford Principles. |
| 9 | `supply_chain_portal.py` | Enterprise supplier data collection portal. Generates and distributes supplier questionnaires (climate, environmental, social). Tracks submission status across 100,000+ suppliers. Validates and ingests supplier-reported emission data. Integrates with CDP Supply Chain, EcoVadis, SEDEX, and WBCSD PACT for data exchange. Provides supplier dashboard showing response rates, scores, and improvement trends. |
| 10 | `financial_system_bridge.py` | General ledger integration for carbon cost allocation. Posts carbon charges to GL accounts per business unit. Generates carbon-adjusted financial reports. Supports SAP FI, Oracle Financials, and generic GL formats. Reconciles carbon ledger to GHG inventory. Supports ESRS E1-8 (internal carbon pricing) disclosure data extraction. |
| 11 | `data_quality_guardian.py` | Automated data quality monitoring and improvement engine. Continuously assesses data quality across all sources against +/-3% accuracy target. Applies GHG Protocol 5-level data quality hierarchy. Generates data quality improvement plans per category per entity. Triggers alerts when DQ falls below threshold. Tracks DQ improvement over time. Integrates with DATA agents (DATA-010 Profiler, DATA-011 Dedup, DATA-013 Outlier, DATA-019 Validation). |
| 12 | `setup_wizard.py` | 10-step enterprise onboarding wizard: (1) Corporate profile (legal entity, sector, revenue, employees), (2) Organizational boundary definition (consolidation approach, entity hierarchy), (3) Reporting year and base year selection, (4) Entity registration (subsidiaries, JVs, associates), (5) ERP system connection (SAP/Oracle/Workday), (6) Data source mapping per entity, (7) Scope 3 materiality screening, (8) SBTi target configuration, (9) Carbon pricing setup, (10) Go-live health check and demo validation. |
| 13 | `health_check.py` | 25-category system health verification: (1) Pack manifest, (2) Configuration, (3) Preset loading, (4-11) Engines 1-8, (12-19) Workflows 1-8, (20) MRV agent connectivity (30 agents), (21) DATA agent connectivity (20 agents), (22) FOUND agent connectivity (10 agents), (23) ERP connector status, (24) Database connectivity, (25) Overall health score (0-100). |

### 3.7 Presets

| # | Preset | Sector | Key Characteristics |
|---|--------|--------|---------------------|
| 1 | `manufacturing_enterprise.yaml` | Manufacturing / Industrial | High Scope 1 (process, combustion); SDA pathway (cement/steel/chemicals); multi-facility consolidation; process emissions agents (MRV-004); capital-intensive CapEx planning; ETS compliance (EU/UK ETS); CBAM exposure for imports |
| 2 | `energy_utilities.yaml` | Power / Oil & Gas / Utilities | Very high Scope 1; SDA mandatory (power sector); FLAG if upstream oil/gas with land use; coal/gas asset stranded risk; renewable transition modeling; grid decarbonization scenarios; Cat 11 (use of sold energy) |
| 3 | `financial_services.yaml` | Banking / Insurance / Asset Management | Low direct emissions; Scope 3 Cat 15 (investments) dominant (often 95%+); PCAF financed emissions methodology; portfolio temperature scoring; FINZ targets; NZBA/NZAM commitments; mortgage/real estate Scope 3 |
| 4 | `technology.yaml` | Technology / Software / Data Centers | Low Scope 1; Scope 2 dominated by data centers (PUE tracking); Scope 3 Cat 1 (hardware procurement) + Cat 11 (use of sold products); RE100 alignment; cloud provider emission factors; avoided emissions (SaaS vs. on-premise) |
| 5 | `consumer_goods.yaml` | FMCG / Consumer Products | Scope 3 dominant (Cat 1 raw materials, Cat 12 end-of-life); supply chain engagement critical (1000+ ingredient suppliers); product-level carbon footprint; packaging emissions; FLAG if agricultural inputs |
| 6 | `transport_logistics.yaml` | Transport / Aviation / Shipping / Logistics | SDA for transport sectors; fleet electrification scenarios; SAF (aviation) / alternative fuels (shipping); Cat 3 fuel and energy; last-mile delivery optimization; modal shift modeling |
| 7 | `real_estate.yaml` | Real Estate / Construction / Property | SDA buildings pathway; CRREM alignment; building performance (kgCO2/sqm); Cat 13 (downstream leased assets) dominant; renovation vs. new build scenarios; green building certification (BREEAM, LEED) alignment |
| 8 | `healthcare_pharma.yaml` | Healthcare / Pharmaceuticals | Mixed Scope 1 (labs, manufacturing) + high Scope 3 (Cat 1 procurement); supply chain cold chain emissions; anesthetic gas Scope 1; clinical trial travel (Cat 6); product lifecycle (Cat 12); regulatory (FDA/EMA sustainability) |

### 3.8 Preset Specifications

#### 3.8.1 Manufacturing Enterprise Preset (`manufacturing_enterprise.yaml`)

**Target Organizations**: Siemens, BASF, 3M, Honeywell, ABB, Schneider Electric, and similar diversified industrial enterprises with significant process emissions, multi-site manufacturing, and complex supply chains.

**Configuration Overrides**:
- `sector: manufacturing`
- `sbti_pathway: mixed` (ACA for general operations, SDA for cement/chemicals/steel divisions)
- `baseline.scope1_agents: [MRV-001, MRV-002, MRV-003, MRV-004, MRV-005]` (process emissions critical)
- `baseline.scope3_priority_categories: [1, 2, 3, 4, 11, 12]` (materials, capital goods, fuel, transport, use-phase, end-of-life)
- `carbon_pricing.cbam_enabled: true` (EU CBAM exposure for imports)
- `carbon_pricing.ets_enabled: true` (EU ETS compliance for covered installations)
- `scenarios.technology_focus: [electrification, hydrogen, ccs, energy_efficiency]`
- `supply_chain.tier_depth: 4` (raw materials through to component suppliers)
- `flag_enabled: false` (unless significant land use in supply chain)

#### 3.8.2 Energy & Utilities Preset (`energy_utilities.yaml`)

**Target Organizations**: Shell, TotalEnergies, Enel, Iberdrola, NextEra Energy, Duke Energy, and similar energy companies navigating the transition from fossil fuels to renewables.

**Configuration Overrides**:
- `sector: energy`
- `sbti_pathway: sda` (SDA mandatory for power generation sector)
- `baseline.scope1_agents: [MRV-001, MRV-003, MRV-004, MRV-005, MRV-006]` (combustion, process, fugitive critical)
- `baseline.scope3_priority_categories: [1, 3, 9, 10, 11]` (fuel procurement, fuel & energy, downstream, processing, use of sold energy)
- `scenarios.stranded_asset_analysis: true` (fossil fuel asset write-down risk)
- `scenarios.technology_focus: [renewables, battery_storage, hydrogen, nuclear, grid_decarbonization]`
- `carbon_pricing.ets_enabled: true` (EU ETS for power generation)
- `avoided_emissions.scope4_enabled: true` (renewable energy displacing fossil)
- `flag_enabled: true` (if upstream oil/gas with land disturbance)

#### 3.8.3 Financial Services Preset (`financial_services.yaml`)

**Target Organizations**: JPMorgan, HSBC, Allianz, BlackRock, BNP Paribas, and similar banks, insurers, and asset managers with massive financed emissions portfolios.

**Configuration Overrides**:
- `sector: financial_services`
- `sbti_pathway: aca_15c` (ACA for direct emissions)
- `finz_enabled: true` (FINZ V1.0 for portfolio targets)
- `baseline.scope1_agents: [MRV-001, MRV-002]` (office heating, HVAC only)
- `baseline.scope3_priority_categories: [15, 1, 6, 7]` (investments dominant at 95%+)
- `baseline.pcaf_enabled: true` (PCAF methodology for financed emissions)
- `baseline.pcaf_asset_classes: [listed_equity, corporate_bonds, business_loans, mortgages, commercial_re, project_finance]`
- `scenarios.portfolio_temperature_scoring: true` (WATS, TETS, MOTS, EOTS)
- `supply_chain.engagement_via_lending: true` (engage borrowers on climate)
- `financial_integration.green_bond_screening: true`
- `financial_integration.taxonomy_alignment: true`
- `carbon_pricing.lending_carbon_price: true` (carbon cost in credit decisions)

#### 3.8.4 Technology Preset (`technology.yaml`)

**Target Organizations**: Microsoft, Google, Apple, SAP, Salesforce, and similar technology companies with data center energy as dominant Scope 2 and supply chain hardware as dominant Scope 3.

**Configuration Overrides**:
- `sector: technology`
- `sbti_pathway: aca_15c` (ACA pathway)
- `baseline.scope1_agents: [MRV-001, MRV-002]` (backup generators, data center cooling)
- `baseline.scope2_pue_tracking: true` (Power Usage Effectiveness for data centers)
- `baseline.scope3_priority_categories: [1, 2, 3, 11, 12]` (hardware procurement, capital goods, fuel, product use-phase, end-of-life)
- `baseline.data_center_metering: detailed` (per-facility kWh with PUE)
- `avoided_emissions.scope4_enabled: true` (SaaS vs. on-premise efficiency gains)
- `scenarios.re100_alignment: true` (100% renewable energy target)
- `supply_chain.hardware_supplier_focus: true` (semiconductor, PCB, assembly suppliers)
- `supply_chain.cloud_provider_emissions: true` (track cloud hosting emissions)
- `carbon_pricing.data_center_allocation: true` (carbon cost per compute hour)

#### 3.8.5 Consumer Goods Preset (`consumer_goods.yaml`)

**Target Organizations**: Unilever, P&G, Nestle, L'Oreal, Nike, and similar FMCG and consumer product companies with supply chain-dominated emission profiles.

**Configuration Overrides**:
- `sector: consumer_goods`
- `sbti_pathway: mixed` (ACA + FLAG for food/agriculture inputs)
- `baseline.scope3_priority_categories: [1, 4, 9, 11, 12]` (raw materials dominant)
- `flag_enabled: true` (agricultural inputs: palm oil, soy, dairy, cocoa, coffee)
- `supply_chain.tier_depth: 5` (farm-to-shelf traceability)
- `supply_chain.commodity_tracking: [palm_oil, soy, cocoa, coffee, cotton, rubber, timber]`
- `scenarios.product_innovation_modeling: true` (reformulation, lightweighting)
- `carbon_pricing.product_level_footprint: true` (PCF per SKU)
- `avoided_emissions.scope4_enabled: true` (eco-innovation products)
- `financial_integration.packaging_emissions_allocation: true`
- `csrd_enabled: true` (ESRS E5 circular economy likely material)

#### 3.8.6 Transport & Logistics Preset (`transport_logistics.yaml`)

**Target Organizations**: Maersk, DHL, FedEx, Ryanair, Delta Air Lines, and similar transport and logistics companies where fleet emissions dominate.

**Configuration Overrides**:
- `sector: transport`
- `sbti_pathway: sda` (SDA mandatory for transport sectors)
- `baseline.scope1_agents: [MRV-001, MRV-003, MRV-005]` (mobile combustion dominant)
- `baseline.scope3_priority_categories: [3, 1, 4, 9]` (fuel & energy, procurement, transport)
- `scenarios.fleet_electrification: true` (EV transition modeling)
- `scenarios.alternative_fuels: [saf, e_methanol, ammonia, hydrogen]` (for aviation/shipping)
- `scenarios.modal_shift_analysis: true` (road to rail, air to sea)
- `carbon_pricing.fuel_price_sensitivity: true` (fossil fuel price trajectory)
- `supply_chain.carrier_emissions_tracking: true` (subcontracted transport)
- `financial_integration.fuel_cost_carbon_overlay: true`
- `avoided_emissions.scope4_enabled: true` (efficient logistics vs. alternative)

#### 3.8.7 Real Estate Preset (`real_estate.yaml`)

**Target Organizations**: Prologis, Vonovia, British Land, Brookfield, and similar REITs and property companies where building emissions dominate.

**Configuration Overrides**:
- `sector: real_estate`
- `sbti_pathway: sda` (SDA for buildings sector: commercial and/or residential)
- `baseline.scope1_agents: [MRV-001, MRV-002]` (building heating, refrigerants)
- `baseline.scope3_priority_categories: [13, 1, 2, 11]` (downstream leased assets dominant)
- `baseline.crrem_alignment: true` (Carbon Risk Real Estate Monitor)
- `baseline.building_performance: kgCO2_per_sqm` (intensity metric)
- `scenarios.renovation_vs_newbuild: true` (building retrofit scenarios)
- `scenarios.green_building_certification: [breeam, leed, dgnb, well]`
- `supply_chain.construction_materials_focus: true` (embodied carbon)
- `financial_integration.green_lease_tracking: true`
- `financial_integration.stranded_asset_building_level: true` (CRREM stranding year per asset)
- `consolidation.approach: operational_control` (common for REITs operating managed properties)

#### 3.8.8 Healthcare & Pharma Preset (`healthcare_pharma.yaml`)

**Target Organizations**: Novartis, Roche, Johnson & Johnson, Pfizer, and similar pharmaceutical and healthcare companies with mixed emission profiles.

**Configuration Overrides**:
- `sector: healthcare`
- `sbti_pathway: aca_15c`
- `baseline.scope1_agents: [MRV-001, MRV-002, MRV-003, MRV-004]` (labs, HVAC, fleet, process)
- `baseline.scope1_anesthetic_gases: true` (N2O, desflurane, sevoflurane tracking)
- `baseline.scope3_priority_categories: [1, 2, 4, 6, 12]` (API procurement, capital goods, transport, clinical trial travel, product disposal)
- `baseline.cold_chain_emissions: true` (pharmaceutical cold chain transport)
- `supply_chain.api_supplier_focus: true` (Active Pharmaceutical Ingredient suppliers)
- `supply_chain.clinical_trial_emissions: true` (patient travel, site operations)
- `scenarios.drug_lifecycle_modeling: true` (R&D to manufacturing to distribution)
- `carbon_pricing.r_and_d_carbon_allocation: true` (carbon cost per molecule in development)
- `avoided_emissions.scope4_enabled: false` (pharmaceutical avoided emissions methodology immature)

---

## 4. Agent Dependencies

### 4.1 MRV Agents (all 30)

All 30 AGENT-MRV agents are direct dependencies via `mrv_bridge.py`. Unlike PACK-026 (SME, 7 agents) or PACK-022 (Acceleration, all 30 but spend-based fallback), PACK-027 requires all 30 agents operating at full activity-based precision.

| MRV Agent | Enterprise Application | Priority |
|-----------|----------------------|----------|
| MRV-001 Stationary Combustion | Boilers, furnaces, heaters, generators across all facilities | Critical |
| MRV-002 Refrigerants & F-Gas | Commercial/industrial refrigeration, HVAC, chillers | High |
| MRV-003 Mobile Combustion | Fleet vehicles (owned/leased), company aircraft | Critical |
| MRV-004 Process Emissions | Cement, chemicals, metals, glass, ceramics | Critical (industrial) |
| MRV-005 Fugitive Emissions | Gas distribution, coal mining, oil/gas operations | High (energy sector) |
| MRV-006 Land Use Emissions | Agriculture, forestry, land use change | Critical (FLAG sectors) |
| MRV-007 Waste Treatment | On-site wastewater, waste incineration | Medium |
| MRV-008 Agricultural Emissions | Enteric fermentation, manure, soil N2O, rice paddies | Critical (agri-food) |
| MRV-009 Scope 2 Location-Based | Grid electricity by country/region | Critical |
| MRV-010 Scope 2 Market-Based | PPAs, RECs/GOs, green tariffs, residual mix | Critical |
| MRV-011 Steam/Heat Purchase | District heating, industrial steam | High |
| MRV-012 Cooling Purchase | District cooling (where applicable) | Medium |
| MRV-013 Dual Reporting Reconciliation | Location vs. market Scope 2 delta | Critical |
| MRV-014 Purchased Goods (Cat 1) | Raw materials, components, services (dominant category) | Critical |
| MRV-015 Capital Goods (Cat 2) | Equipment, machinery, buildings, vehicles, IT infrastructure | High |
| MRV-016 Fuel & Energy (Cat 3) | WTT fuel emissions, T&D losses (auto-calculated from S1+S2) | High |
| MRV-017 Upstream Transport (Cat 4) | Inbound logistics by mode (road, rail, sea, air) | Critical |
| MRV-018 Waste Generated (Cat 5) | Operational waste by type and treatment | Medium |
| MRV-019 Business Travel (Cat 6) | Air, rail, car, hotel stays | High |
| MRV-020 Employee Commuting (Cat 7) | Survey-based or modeled commute patterns | Medium |
| MRV-021 Upstream Leased (Cat 8) | Leased facilities not in Scope 1+2 | Medium |
| MRV-022 Downstream Transport (Cat 9) | Outbound logistics, distribution, last-mile | High |
| MRV-023 Processing of Sold Products (Cat 10) | Third-party processing of intermediate products | Sector-dependent |
| MRV-024 Use of Sold Products (Cat 11) | Energy use of sold products over lifetime | Critical (tech/automotive) |
| MRV-025 End-of-Life (Cat 12) | Product disposal and recycling | Medium |
| MRV-026 Downstream Leased (Cat 13) | Assets leased to others | High (real estate) |
| MRV-027 Franchises (Cat 14) | Franchise operations emissions | High (franchisors) |
| MRV-028 Investments (Cat 15) | PCAF financed emissions by asset class | Critical (financial) |
| MRV-029 Category Mapper | Route data to appropriate Scope 3 category agents | Critical |
| MRV-030 Audit Trail & Lineage | SHA-256 provenance for all calculations | Critical |

### 4.2 Decarbonization Agents (all 21)

All 21 DECARB-X agents via `decarb_bridge.py` for reduction pathway planning:
- DECARB-X-001: Abatement Options Library (500+ enterprise-scale options)
- DECARB-X-002: MACC Generator (Marginal Abatement Cost Curves)
- DECARB-X-003: Target Setting Agent (SBTi pathway support)
- DECARB-X-004: Pathway Scenario Builder (multi-scenario modeling)
- DECARB-X-005: Investment Prioritization (CapEx/OpEx ranking)
- DECARB-X-006: Technology Readiness Assessor (TRL assessment)
- DECARB-X-007: Implementation Roadmap (phased execution planning)
- DECARB-X-008: Avoided Emissions Calculator (Scope 4 support)
- DECARB-X-009: Carbon Intensity Tracker (intensity metrics)
- DECARB-X-010: Renewable Energy Planner (RE100, PPA strategy)
- DECARB-X-011: Electrification Planner (fleet, heat pump, process)
- DECARB-X-012: Fuel Switching Agent (alternative fuels, hydrogen)
- DECARB-X-013: Energy Efficiency Identifier (facility-level audits)
- DECARB-X-014: Supplier Engagement Agent (cascade programs)
- DECARB-X-015: Carbon Capture Agent (CCS/CCUS for heavy industry)
- DECARB-X-016: Circular Economy Agent (material circularity)
- DECARB-X-017: Nature-Based Solutions Agent (afforestation, soil carbon)
- DECARB-X-018: Progress Monitoring Agent (annual tracking)
- DECARB-X-019: Residual Emissions Agent (neutralization planning)
- DECARB-X-020: Cost-Benefit Analyzer (financial analysis)
- DECARB-X-021: Transition Finance Agent (green bonds, Taxonomy)

### 4.3 Application Dependencies

| Application | Usage in PACK-027 |
|-------------|------------------|
| GL-GHG-APP | GHG inventory management, base year management, multi-year tracking |
| GL-SBTi-APP | SBTi target validation (42 criteria), temperature rating, SDA tools |
| GL-CDP-APP | CDP questionnaire auto-population, scoring optimization |
| GL-TCFD-APP | TCFD/ISSB S2 disclosure generation, scenario analysis |
| GL-ISO14064-APP | ISO 14064-1 GHG statement, ISO 14064-3 verification support |
| GL-CSRD-APP | ESRS E1 climate change disclosure datapoints |
| GL-Taxonomy-APP | EU Taxonomy climate CapEx alignment |

### 4.4 Data Agents (all 20)

All 20 AGENT-DATA agents via `data_bridge.py` for enterprise data management:
- DATA-001: PDF & Invoice Extractor (utility bills, supplier certificates)
- DATA-002: Excel/CSV Normalizer (bulk data imports)
- DATA-003: ERP/Finance Connector (SAP, Oracle, Workday integration)
- DATA-004: API Gateway Agent (external data feeds, emission factor APIs)
- DATA-005: EUDR Traceability Connector (commodity origin for FLAG)
- DATA-006: GIS/Mapping Connector (facility locations, supply chain geography)
- DATA-007: Deforestation Satellite Connector (FLAG verification)
- DATA-008: Supplier Questionnaire Processor (CDP/custom questionnaires)
- DATA-009: Spend Data Categorizer (procurement classification for Scope 3)
- DATA-010: Data Quality Profiler (enterprise DQ scoring)
- DATA-011: Duplicate Detection Agent (entity/supplier deduplication)
- DATA-012: Missing Value Imputer (gap filling for incomplete datasets)
- DATA-013: Outlier Detection Agent (anomalous emission values)
- DATA-014: Time Series Gap Filler (monthly/quarterly data gaps)
- DATA-015: Cross-Source Reconciliation (ERP vs. meter vs. invoice)
- DATA-016: Data Freshness Monitor (stale data alerts)
- DATA-017: Schema Migration Agent (year-over-year format changes)
- DATA-018: Data Lineage Tracker (source-to-report traceability)
- DATA-019: Validation Rule Engine (enterprise-specific validation rules)
- DATA-020: Climate Hazard Connector (physical risk for TCFD scenarios)

### 4.5 Foundation Agents (all 10)

All 10 AGENT-FOUND agents for orchestration and platform services:
- FOUND-001: GreenLang Orchestrator (DAG execution)
- FOUND-002: Schema Compiler & Validator (data model validation)
- FOUND-003: Unit & Reference Normalizer (unit conversion, currency)
- FOUND-004: Assumptions Registry (assumption documentation)
- FOUND-005: Citations & Evidence Agent (source references)
- FOUND-006: Access & Policy Guard (RBAC enforcement)
- FOUND-007: Agent Registry & Service Catalog (agent discovery)
- FOUND-008: Reproducibility Agent (deterministic replay)
- FOUND-009: QA Test Harness (automated quality checks)
- FOUND-010: Observability & Telemetry Agent (monitoring)

### 4.6 Optional Pack Dependencies

| Pack | Integration | Value |
|------|-------------|-------|
| PACK-021 Net Zero Starter | Baseline migration for companies upgrading from Starter | Preserve existing baseline data and recalculate with enterprise methodology |
| PACK-022 Net Zero Acceleration | Scenario and SDA data migration for companies upgrading | Import existing scenarios, supplier engagement data, temperature scores |
| PACK-023 SBTi Alignment | Extended SBTi lifecycle for companies with existing SBTi process | Bidirectional sync of targets, progress, and validation status |
| PACK-024 Carbon Neutral | Carbon neutrality claim for enterprises pursuing PAS 2060/ISO 14068-1 | Credit portfolio management, claim substantiation |
| PACK-025 Race to Zero | Race to Zero campaign lifecycle for committed enterprises | Starting Line criteria, HLEG reporting |
| PACK-001/002/003 CSRD | CSRD ESRS E1 disclosure integration for EU-reporting enterprises | Share E1 datapoints, avoid duplication |

---

## 5. Enterprise-Specific Considerations

### 5.1 Organizational Boundary Complexity

Enterprise organizational structures present challenges absent in SME or mid-market contexts:

| Structure | GHG Implication | PACK-027 Handling |
|-----------|----------------|-------------------|
| Wholly-owned subsidiaries | 100% consolidation under all approaches | Auto-consolidated; entity-level detail preserved |
| Partially-owned subsidiaries (>50%) | 100% under control approaches; equity % under equity share | Configurable per entity; ownership % tracked with effective dates |
| Joint ventures (50/50) | Depends on JV agreement and control assessment | Control assessment wizard; fallback to equity share |
| Associates (20-49%) | Equity share only | Equity % applied; not included under control approaches |
| SPVs / Project companies | Depends on control and purpose | Case-by-case control assessment |
| Franchises | Scope 3 Cat 14 (for franchisor); Scope 1+2 (for franchisee) | Franchise model with per-franchisee data collection or average profile |
| Outsourced operations | Scope 3 (outsourcer); Scope 1 (if operational control retained) | Classification wizard; ensures no gap between Scope 1 and Scope 3 |

### 5.2 ERP Integration Architecture

Enterprise ERP integration follows a standardized architecture:

```
ERP System (SAP / Oracle / Workday)
  |
  +-- Standard API / OData / REST connector
  |
  +-- GreenLang Data Agent (DATA-003 ERP Connector)
  |     |
  |     +-- Data extraction (scheduled: daily/weekly/monthly)
  |     +-- Data transformation (unit normalization, currency conversion)
  |     +-- Data quality validation (completeness, accuracy checks)
  |
  +-- GreenLang Carbon Data Store (PostgreSQL + TimescaleDB)
  |     |
  |     +-- Activity data tables (energy, fuel, travel, procurement, waste)
  |     +-- Emission factor tables (DEFRA, EPA, ecoinvent, supplier-specific)
  |     +-- Calculated emission tables (per-entity, per-category, per-period)
  |
  +-- Reverse integration (optional)
        |
        +-- Carbon cost allocation to GL accounts
        +-- Carbon-adjusted financial reports
        +-- BU-level carbon performance metrics
```

**SAP Integration Points:**

| SAP Module | Data Extracted | GreenLang Usage |
|------------|---------------|-----------------|
| MM (Materials Management) | Purchase orders, goods receipts, material masters | Scope 3 Cat 1, Cat 2 (procurement volumes) |
| FI (Financial Accounting) | GL postings, vendor invoices, utility bills | Scope 2 (energy invoices), Scope 3 (spend-based fallback) |
| CO (Controlling) | Cost center allocations, internal orders | BU-level carbon allocation |
| SD (Sales & Distribution) | Sales orders, deliveries, shipping | Scope 3 Cat 9 (downstream transport) |
| PM (Plant Maintenance) | Equipment records, refrigerant logs, maintenance orders | Scope 1 (refrigerants, equipment-based factors) |
| HCM (Human Capital Management) | Headcount by location, travel bookings, commute surveys | Scope 3 Cat 6 (travel), Cat 7 (commuting) |
| TM (Transportation Management) | Shipment records, carrier data, route information | Scope 3 Cat 4 (upstream transport) |
| S/4HANA Sustainability | Sustainability Control Tower data, product footprints | Cross-module carbon data integration |

### 5.3 Data Quality Management

Enterprise data quality management follows a continuous improvement cycle:

**Data Quality Improvement Targets:**

| Year | DQ Level Target (weighted avg) | Primary Improvement Actions |
|------|-------------------------------|---------------------------|
| Year 1 (baseline) | Level 3.5 (average-data dominant) | Establish measurement systems; engage top 50 suppliers; connect ERP |
| Year 2 | Level 2.8 (shift toward supplier-specific) | Expand CDP Supply Chain; improve metering; automate data collection |
| Year 3 | Level 2.2 (majority supplier-specific) | Deep supplier engagement; product-level footprints; verified data |
| Year 5 | Level 1.8 (approaching best practice) | PACT data exchange; real-time metering; verified supplier data |

**Data Quality Scoring per GHG Protocol:**

| DQ Indicator | Level 1 (Best) | Level 3 (Average) | Level 5 (Worst) |
|--------------|---------------|-------------------|-----------------|
| Technological representativeness | Same technology | Similar technology | Unknown technology |
| Temporal representativeness | Same year | Within 3 years | Older than 5 years |
| Geographical representativeness | Same country/region | Same continent | Global average |
| Completeness | 100% of relevant data | 50-80% | <50% or estimated |
| Reliability | Verified data, peer reviewed | Non-verified calculation | Expert estimate |

### 5.4 Multi-Year Trend Analysis

Enterprise reporting requires multi-year trend analysis with consistent methodology:

**Trend Metrics:**

| Metric | Description | Trend Period |
|--------|-------------|-------------|
| Absolute emissions (tCO2e) | Total emissions by scope | Base year to current, projected to target year |
| Emissions intensity (tCO2e/$M revenue) | Revenue-normalized | Base year to current |
| Emissions intensity (tCO2e/FTE) | Employee-normalized | Base year to current |
| Scope 3 data quality (weighted DQ score) | Coverage and quality improvement | Year-over-year |
| SBTi pathway alignment (% progress) | On-track vs. target pathway | Annual vs. linear pathway |
| Supplier engagement (% of Scope 3 engaged) | Active engagement coverage | Year-over-year |
| Renewable energy share (%) | RE as % of total electricity | Year-over-year |

**Base Year Adjustment Policy:**
- Trigger assessment performed annually as part of annual inventory workflow
- 5% significance threshold applied consistently across all triggers
- Pro-rata treatment for mid-year M&A and divestitures
- Like-for-like comparison methodology for organic vs. structural changes
- Full documentation of all adjustments in audit trail

### 5.5 Governance and Controls

Enterprise climate programs require formal governance structures:

**Three Lines Model for Climate Data:**

| Line | Responsibility | PACK-027 Support |
|------|---------------|-----------------|
| First Line (Operations) | Entity-level data owners; data collection and entry; source data accuracy | Entity-level dashboards; data completeness alerts; validation rules |
| Second Line (Risk/Compliance) | Central sustainability team; methodology design; quality oversight | DQ scoring; outlier detection; reconciliation; analytical review |
| Third Line (Assurance) | Internal audit; external auditor | Audit workpapers; sampling support; control testing; management assertions |

**Segregation of Duties:**
- Data entry personnel cannot modify emission factors
- Calculation methodology changes require two-person approval
- Base year recalculations require senior management sign-off
- External disclosures require board/audit committee approval
- Carbon credit transactions require finance team authorization

---

## 6. Performance Targets

| Metric | Target |
|--------|--------|
| Enterprise baseline calculation (100 entities, 15 Scope 3 cats) | <4 hours (wall-clock, after data ingestion) |
| Single-entity baseline calculation | <15 minutes |
| Multi-entity consolidation (100+ entities) | <30 minutes |
| Scenario modeling (10,000 Monte Carlo runs, 3 scenarios) | <30 minutes |
| SBTi 42-criteria validation | <10 minutes |
| Annual inventory recalculation | <2 hours |
| CDP questionnaire auto-population | <30 minutes |
| TCFD report generation | <20 minutes |
| Board climate report generation | <15 minutes |
| Assurance workpaper generation | <45 minutes |
| Supply chain heatmap (50,000 suppliers) | <60 minutes |
| Carbon pricing allocation (50 BUs) | <10 minutes |
| Regulatory filing generation (5 frameworks) | <30 minutes |
| Memory ceiling per engine | 4096 MB |
| Cache hit target | 85% |
| Max entities | 500 |
| Max suppliers | 100,000 |
| Max employees modeled | 500,000 |
| Max facilities | 5,000 |
| Max Scope 3 categories | 15 (all) |
| API response time (95th percentile) | <2 seconds |
| Batch processing throughput | 1,000 entity-years per hour |
| Data refresh frequency (ERP) | Daily (configurable) |

---

## 7. Security Requirements

### 7.1 Authentication and Authorization

- JWT RS256 authentication with enterprise SSO integration (SAML 2.0, OIDC)
- RBAC with 8 roles specific to enterprise climate governance:

| Role | Permissions | Typical User |
|------|------------|-------------|
| `enterprise_admin` | Full system configuration, user management, entity management | IT/Sustainability system admin |
| `cso` | All read/write, approve targets, approve disclosures, view all entities | Chief Sustainability Officer |
| `sustainability_manager` | Read/write data, run engines, generate reports, manage suppliers | Head of Sustainability |
| `entity_data_owner` | Read/write data for assigned entities only, run entity-level engines | BU sustainability lead |
| `analyst` | Read all data, run engines, generate reports (no configuration changes) | Sustainability analyst |
| `finance_viewer` | Read carbon-adjusted financial data, carbon pricing, ESRS E1-8/E1-9 | CFO, finance team |
| `auditor` | Read all data and workpapers, view audit trails, no write access | Internal/external auditor |
| `board_viewer` | Read executive dashboard, board reports, strategy documents only | Board member |

### 7.2 Data Protection

- AES-256-GCM encryption at rest for all emission and financial data
- TLS 1.3 for all data in transit (internal and external)
- SHA-256 provenance hashing on all calculation outputs (every engine, every phase)
- Full audit trail per SEC-005 (centralized audit logging)
- ERP API credentials encrypted via Vault (SEC-006)
- GDPR compliance for employee commuting and travel data (anonymization at individual level)
- SOX compliance for carbon data integrated with financial reporting
- Data residency controls (EU data stays in EU, US data stays in US) per regulatory requirements
- Supplier data access controls (suppliers see only their own data in portal)
- Board report access restricted to authorized governance roles

### 7.3 Audit and Compliance

- Every calculation produces a complete audit trail: input data hash, emission factors used, methodology reference, calculation steps, output hash, timestamp, user identity
- Audit trails are immutable (append-only log with cryptographic chaining)
- Retention period: 10 years minimum (aligned with financial record retention)
- External auditor access via dedicated `auditor` role with read-only permissions and workpaper download
- Management assertion letter template auto-generated for assurance engagements
- Control evidence automatically collected for ISAE 3410 / ISAE 3000 engagements

---

## 8. Database Migrations

Inherits platform migrations V001-V128. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V083-PACK027-001 | `ent_corporate_profiles` | Enterprise corporate profiles with entity hierarchy, consolidation approach, sector classification, regulatory jurisdictions |
| V083-PACK027-002 | `ent_entity_hierarchy` | Entity tree structure with ownership percentages, control assessments, effective dates, consolidation method per entity |
| V083-PACK027-003 | `ent_baselines` | Enterprise GHG baseline records with per-entity per-category per-scope detail, data quality matrix, confidence intervals |
| V083-PACK027-004 | `ent_sbti_targets` | SBTi target records (near-term, long-term, net-zero) with criteria validation results, pathway milestones, submission status |
| V083-PACK027-005 | `ent_scenarios` | Scenario modeling records with Monte Carlo parameters, simulation results (P10-P90), sensitivity analysis |
| V083-PACK027-006 | `ent_carbon_pricing` | Internal carbon pricing configuration with price trajectories, BU allocations, investment appraisals |
| V083-PACK027-007 | `ent_avoided_emissions` | Scope 4 avoided emissions records with product data, baseline scenarios, attribution methodology |
| V083-PACK027-008 | `ent_supply_chain` | Supplier master with tier assignment, engagement status, CDP scores, SBTi status, year-over-year emission data |
| V083-PACK027-009 | `ent_consolidation` | Multi-entity consolidation records with intercompany eliminations, reconciliation, base year recalculations |
| V083-PACK027-010 | `ent_financial_integration` | Carbon-adjusted financial data with P&L allocation, balance sheet items, CBAM exposure |
| V083-PACK027-011 | `ent_assurance` | Assurance engagement records with scope, workpapers, findings, management responses |
| V083-PACK027-012 | `ent_data_quality` | Data quality scoring records with per-category per-entity DQ levels, improvement plans, trend tracking |
| V083-PACK027-013 | `ent_regulatory_filings` | Regulatory filing records by framework (SEC, CSRD, CDP, TCFD, SB253) with submission status and dates |
| V083-PACK027-014 | `ent_erp_connections` | ERP connector configuration with system type, connection parameters, extraction schedules, data mapping |
| V083-PACK027-015 | `ent_audit_trail` | Extended audit trail for enterprise governance with cryptographic chaining, immutable append-only log |

---

## 9. File Structure

```
packs/net-zero/PACK-027-enterprise-net-zero/
  __init__.py
  pack.yaml
  config/
    __init__.py
    pack_config.py
    demo/
      __init__.py
      demo_config.yaml
    presets/
      __init__.py
      manufacturing_enterprise.yaml
      energy_utilities.yaml
      financial_services.yaml
      technology.yaml
      consumer_goods.yaml
      transport_logistics.yaml
      real_estate.yaml
      healthcare_pharma.yaml
  engines/
    __init__.py
    enterprise_baseline_engine.py
    sbti_target_engine.py
    scenario_modeling_engine.py
    carbon_pricing_engine.py
    scope4_avoided_emissions_engine.py
    supply_chain_mapping_engine.py
    multi_entity_consolidation_engine.py
    financial_integration_engine.py
  workflows/
    __init__.py
    comprehensive_baseline_workflow.py
    sbti_submission_workflow.py
    annual_inventory_workflow.py
    scenario_analysis_workflow.py
    supply_chain_engagement_workflow.py
    internal_carbon_pricing_workflow.py
    multi_entity_rollup_workflow.py
    external_assurance_workflow.py
  templates/
    __init__.py
    ghg_inventory_report.py
    sbti_target_submission.py
    cdp_climate_response.py
    tcfd_report.py
    executive_dashboard.py
    supply_chain_heatmap.py
    scenario_comparison.py
    assurance_statement.py
    board_climate_report.py
    regulatory_filings.py
  integrations/
    __init__.py
    sap_connector.py
    oracle_connector.py
    workday_connector.py
    cdp_bridge.py
    sbti_bridge.py
    assurance_provider_bridge.py
    multi_entity_orchestrator.py
    carbon_marketplace_bridge.py
    supply_chain_portal.py
    financial_system_bridge.py
    data_quality_guardian.py
    setup_wizard.py
    health_check.py
  data/
    __init__.py
    enterprise_emission_factors.json
    sbti_sda_benchmarks.json
    carbon_price_scenarios.json
    industry_benchmarks.json
    regulatory_requirements.json
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_baseline_engine.py
    test_sbti_target_engine.py
    test_scenario_modeling_engine.py
    test_carbon_pricing_engine.py
    test_scope4_engine.py
    test_supply_chain_engine.py
    test_consolidation_engine.py
    test_financial_integration_engine.py
    test_workflows.py
    test_templates.py
    test_integrations.py
    test_presets.py
    test_erp_connectors.py
    test_assurance.py
    test_data_quality.py
    test_e2e.py
    test_orchestrator.py
```

---

## 10. Testing Requirements

### 10.1 Test Files (20 files, 900+ tests target)

| Test File | Scope | Target Tests |
|-----------|-------|-------------|
| conftest.py | Shared fixtures: enterprise company (multi-entity), supplier portfolio, financial data, ERP mock data | N/A (fixtures) |
| test_manifest.py | Pack YAML validation, version, structure, component counts | 60+ |
| test_config.py | Config system, preset loading (8 presets), merge hierarchy, validation | 50+ |
| test_baseline_engine.py | Engine 1: all 15 Scope 3 categories, 5 DQ levels, multi-entity, materiality | 80+ |
| test_sbti_target_engine.py | Engine 2: 42 criteria validation, ACA/SDA/FLAG, near-term/long-term/net-zero | 70+ |
| test_scenario_modeling_engine.py | Engine 3: Monte Carlo (1.5C/2C/BAU), sensitivity analysis, probability outputs | 60+ |
| test_carbon_pricing_engine.py | Engine 4: shadow pricing, BU allocation, CBAM, carbon-adjusted financials | 50+ |
| test_scope4_engine.py | Engine 5: avoided emissions, baseline scenarios, attribution, conservative principles | 45+ |
| test_supply_chain_engine.py | Engine 6: tiering, engagement tracking, hotspot analysis, CDP integration | 55+ |
| test_consolidation_engine.py | Engine 7: 3 consolidation approaches, intercompany elimination, base year recalc | 65+ |
| test_financial_integration_engine.py | Engine 8: P&L allocation, carbon balance sheet, ESRS E1-8/E1-9 | 50+ |
| test_workflows.py | All 8 workflows end-to-end with synthetic enterprise data | 40+ |
| test_templates.py | All 10 templates in 4 formats (MD, HTML, JSON, PDF/XLSX) + template registry | 35+ |
| test_integrations.py | All 13 integrations with mock ERP, mock CDP, mock SBTi | 30+ |
| test_presets.py | All 8 presets with representative enterprise profiles | 25+ |
| test_erp_connectors.py | SAP, Oracle, Workday connector data extraction and transformation | 35+ |
| test_assurance.py | Workpaper generation, control testing, management assertion, evidence collection | 30+ |
| test_data_quality.py | DQ scoring, improvement tracking, guardian alerts, GHG Protocol hierarchy | 30+ |
| test_e2e.py | End-to-end flows (5 enterprise scenarios) | 20+ |
| test_orchestrator.py | Pipeline execution, retry, provenance, parallel phases | 20+ |

### 10.2 Key Test Scenarios

**Scenario 1: Global Manufacturer Full Baseline**
- Company: GlobalMfg Corp (synthetic), 120 entities across 35 countries, $25B revenue, 85,000 employees
- Flow: Load manufacturing preset -> Map entity hierarchy (3 levels) -> Import SAP data (energy, procurement, fleet, travel) -> Calculate Scope 1 (stationary + mobile + process) -> Calculate Scope 2 (dual reporting, 35 grids) -> Calculate Scope 3 (all 15 cats, supplier-specific for top 50, spend-based for tail) -> Consolidate (financial control) -> Validate SBTi criteria (ACA + SDA for cement division) -> Generate GHG inventory report + CDP response + ESRS E1
- Expected outputs: Consolidated baseline (~5M tCO2e), 42-criteria validation, CDP auto-populated, ESRS E1 chapter

**Scenario 2: Financial Institution Portfolio Targets**
- Company: GreenBank SE (synthetic), banking + asset management, $500B AUM, 30 countries
- Flow: Load financial services preset -> Calculate direct emissions (Scope 1+2, offices/data centers) -> Calculate financed emissions (Scope 3 Cat 15, PCAF methodology across 8 asset classes) -> Set SBTi FINZ targets -> Temperature scoring (WATS + portfolio) -> Carbon pricing for lending decisions -> Generate TCFD report + CDP response
- Expected outputs: Portfolio emissions by asset class, temperature scores, TCFD scenario analysis, FINZ target package

**Scenario 3: Multi-Entity Consolidation with M&A**
- Company: AcquireCo Ltd (synthetic), 200 entities, mid-year acquisition of 50-entity TargetCo
- Flow: Load manufacturing preset -> Map pre-acquisition hierarchy -> Calculate pre-acquisition baseline -> Register acquisition event (July 1) -> Map post-acquisition hierarchy (250 entities) -> Pro-rata current year calculation -> Base year recalculation (>5% significance) -> Intercompany elimination (3 internal transactions) -> Generate consolidated report with restated base year
- Expected outputs: Restated base year, current year with pro-rata acquisition, elimination workpapers, delta reconciliation

**Scenario 4: Enterprise Carbon Pricing Implementation**
- Company: CarbonLeader AG (synthetic), diversified industrial, 5 BUs, $10B revenue
- Flow: Load manufacturing preset -> Baseline by BU -> Set internal carbon price ($100/tCO2e, 5% annual escalation) -> Allocate to BUs (Scope 1+2 direct, Scope 3 by procurement spend) -> Evaluate 10 CapEx projects with/without carbon price -> Generate carbon-adjusted P&L -> Model CBAM exposure for EU imports -> Generate ESRS E1-8 disclosure
- Expected outputs: BU carbon charges, project ranking delta with carbon price, CBAM certificate cost, ESRS E1-8 datapoints

**Scenario 5: External Assurance Preparation**
- Company: AssuranceReady Inc (synthetic), Fortune 500, preparing for reasonable assurance
- Flow: Load technology preset -> Generate current year inventory -> Run data quality assessment (target: DQ Level 2.0) -> Generate audit workpapers (Big 4 format) -> Perform pre-assurance control testing (reconciliation, analytical review, sample tests on 60 items) -> Generate management assertion letter -> Package assurance evidence (source data, calculations, factors, methodology)
- Expected outputs: Workpaper set (15 workpapers), control test results, management assertion letter, evidence index, DQ improvement plan

### 10.3 Test Infrastructure

- **Dynamic loading**: All tests use `importlib` dynamic loading (no package installation required)
- **Fixtures**: Shared conftest.py provides synthetic enterprise data: multi-entity hierarchy (120 entities), supplier portfolio (50,000 suppliers), financial data (P&L, balance sheet), ERP mock data (SAP extracts)
- **Determinism**: All tests verify SHA-256 provenance hashes for identical inputs produce identical outputs
- **Coverage target**: 90%+ line coverage across all engines; 85%+ across workflows and integrations
- **CI integration**: Tests run in GitHub Actions via INFRA-007 CI/CD pipeline
- **Performance tests**: Subset of tests verify performance targets (consolidation <30 min, scenario <30 min)
- **Regulatory accuracy**: Template outputs validated against official regulatory formats (SEC, CSRD, CDP)

---

## 11. Configuration System

### 11.1 Pack Configuration

**File**: `config/pack_config.py`

**Configuration Hierarchy** (later overrides earlier):
1. Base `pack.yaml` manifest (defaults)
2. Preset YAML (manufacturing_enterprise / energy_utilities / financial_services / technology / consumer_goods / transport_logistics / real_estate / healthcare_pharma)
3. Environment overrides (`ENT_NET_ZERO_*` environment variables)
4. Explicit runtime overrides

**Top-Level Configuration Model** (`PackConfig`):

```python
class EnterpriseSector(str, Enum):
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    FINANCIAL_SERVICES = "financial_services"
    TECHNOLOGY = "technology"
    CONSUMER_GOODS = "consumer_goods"
    TRANSPORT = "transport"
    REAL_ESTATE = "real_estate"
    HEALTHCARE = "healthcare"

class ConsolidationApproach(str, Enum):
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"
    EQUITY_SHARE = "equity_share"

class SBTiPathway(str, Enum):
    ACA_15C = "aca_1.5c"
    ACA_WB2C = "aca_wb2c"
    SDA = "sda"
    FLAG = "flag"
    MIXED = "mixed"  # ACA for some divisions, SDA for others

class PackConfig(BaseModel):
    pack_id: str = "PACK-027-enterprise-net-zero"
    version: str = "1.0.0"
    sector: EnterpriseSector
    consolidation_approach: ConsolidationApproach
    sbti_pathway: SBTiPathway
    reporting_year: int
    base_year: int
    currency: str = "USD"

    # Entity configuration
    entity_count: int
    total_employees: int
    total_revenue: float
    countries_of_operation: List[str]

    # Engine configurations
    baseline: EnterpriseBaselineConfig
    sbti_targets: SBTiTargetConfig
    scenarios: ScenarioConfig
    carbon_pricing: CarbonPricingConfig
    avoided_emissions: AvoidedEmissionsConfig
    supply_chain: SupplyChainConfig
    consolidation: ConsolidationConfig
    financial_integration: FinancialIntegrationConfig

    # Feature flags
    scope4_enabled: bool = False  # Opt-in: avoided emissions
    carbon_pricing_enabled: bool = True
    flag_enabled: bool = False  # Auto-set if >20% FLAG emissions
    finz_enabled: bool = False  # Auto-set for financial services preset
    cbam_enabled: bool = True  # EU CBAM exposure tracking
    sec_climate_enabled: bool = True  # SEC Climate Rule
    csrd_enabled: bool = True  # CSRD ESRS E1

    # Data quality
    target_dq_level: float = 2.0  # Weighted average DQ target
    accuracy_target: float = 0.03  # +/-3%

    # Integration
    sap_enabled: bool = False
    oracle_enabled: bool = False
    workday_enabled: bool = False

    # Provenance
    provenance_enabled: bool = True
    provenance_algorithm: str = "sha256"
```

---

## 12. Pricing & Packaging

### 12.1 Pricing Tiers

| Tier | Entities | Employees | Annual License | Implementation | Total Year 1 |
|------|----------|-----------|---------------|---------------|-------------|
| Enterprise Standard | 10-50 | 250-5,000 | $75,000/year | $50,000-$100,000 | $125,000-$175,000 |
| Enterprise Professional | 50-200 | 5,000-50,000 | $150,000/year | $100,000-$250,000 | $250,000-$400,000 |
| Enterprise Premium | 200-500+ | 50,000-500,000+ | $300,000/year | $200,000-$500,000 | $500,000-$800,000 |

### 12.2 Implementation Services

| Service | Description | Typical Duration | Cost Range |
|---------|-------------|-----------------|-----------|
| Enterprise Onboarding | Entity mapping, ERP connection, initial configuration | 2-4 weeks | $25,000-$75,000 |
| Baseline Build | Full Scope 1+2+3 baseline with data quality improvement | 4-8 weeks | $50,000-$150,000 |
| SBTi Target Setting | Pathway selection, target definition, criteria validation | 2-4 weeks | $25,000-$75,000 |
| Assurance Preparation | Workpaper generation, control documentation, pre-audit support | 2-4 weeks | $25,000-$75,000 |
| ERP Integration | SAP/Oracle/Workday connector setup and data mapping | 3-6 weeks | $50,000-$150,000 |
| Training | Administrator training (2 days), analyst training (3 days), executive briefing (half day) | 1-2 weeks | $10,000-$30,000 |

### 12.3 Module Add-Ons

| Module | Description | Price |
|--------|-------------|-------|
| Scope 4 Avoided Emissions | Engine 5 + avoided emissions reporting | $25,000/year |
| Carbon Marketplace | Credit procurement and portfolio management | $15,000/year |
| FINZ Module | Financial institution portfolio targets (PCAF/SBTi FINZ) | $50,000/year |
| Additional ERP Connector | Each additional ERP system beyond first | $25,000/year |
| Advanced Scenario Modeling | Extended Monte Carlo + custom scenarios | $20,000/year |

---

## 13. Deployment Model

### 13.1 Infrastructure Requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| Compute | 4 vCPU, 8 GB RAM (per pack instance) | Scales horizontally via K8s; Monte Carlo may burst to 16 vCPU |
| Storage | 2 GB for emission factor databases + reference data | S3-backed |
| Database | Existing PostgreSQL + TimescaleDB | 15 pack-specific tables |
| Cache | Existing Redis cluster | Emission factor caching, intermediate results |
| Network | Outbound HTTPS for ERP APIs, CDP API, SBTi portal | Firewall rules per ERP system |

### 13.2 Deployment Configuration

```yaml
# Kubernetes deployment snippet
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pack-027-enterprise-net-zero
  labels:
    app: greenlang
    component: solution-pack
    pack: enterprise-net-zero
spec:
  replicas: 3
  selector:
    matchLabels:
      pack: enterprise-net-zero
  template:
    spec:
      containers:
      - name: pack-027
        image: greenlang/pack-027-enterprise-net-zero:1.0.0
        resources:
          requests:
            cpu: "2"
            memory: 4Gi
          limits:
            cpu: "4"
            memory: 8Gi
        env:
        - name: ENT_NET_ZERO_LOG_LEVEL
          value: "INFO"
        - name: ENT_NET_ZERO_PROVENANCE
          value: "true"
        - name: ENT_NET_ZERO_MONTE_CARLO_WORKERS
          value: "4"
```

### 13.3 Pack Registration

PACK-027 registers in the GreenLang Solution Pack registry with the following metadata:
- Pack ID: `PACK-027-enterprise-net-zero`
- Category: `net-zero`
- Sector: `cross-sector` (sector-specific via presets)
- Tier: `enterprise`
- Dependencies: None (standalone)
- Optional bridges: PACK-021/022/023/024/025 (net-zero suite), PACK-001/002/003 (CSRD)

---

## 14. Upgrade Path

### 14.1 Upgrade from Other Packs

| Source Pack | Migration Path | Data Preserved |
|------------|---------------|----------------|
| PACK-021 (Starter) | Automatic baseline data import; recalculate with enterprise methodology | Baseline emissions, target definitions, reduction roadmap |
| PACK-022 (Acceleration) | Import scenarios, supplier engagement, temperature scores | All PACK-022 data + scenario history |
| PACK-023 (SBTi) | Import SBTi targets, criteria validation, progress data | Full SBTi lifecycle data |
| PACK-026 (SME) | Import spend-based baseline; significant methodology upgrade required | Baseline data (recalculated); target data (revalidated under Corporate Standard) |

### 14.2 Graduation from SME to Enterprise

When an organization grows beyond the SME threshold (>250 employees), the upgrade from PACK-026 to PACK-027 involves:

1. **Baseline recalculation**: Spend-based estimates replaced with activity-based calculations where data available; accuracy improves from +/-20-40% to +/-3-10%
2. **Scope 3 expansion**: From 3 categories (Cat 1, 6, 7) to all 15 categories
3. **SBTi pathway change**: From SME simplified to Corporate Standard (ACA/SDA/FLAG)
4. **Entity expansion**: From single entity to multi-entity consolidation
5. **System integration**: From Xero/QuickBooks to SAP/Oracle/Workday
6. **Assurance readiness**: From SHA-256 provenance to full ISO 14064-3 workpapers
7. **Data quality**: From Bronze/Silver/Gold tiers to 5-level GHG Protocol hierarchy

All historical data from PACK-026 is preserved and restated where methodology changes require it.

---

## 15. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-19 |
| Phase 2 | Engine implementation (8 engines) | 2026-03-20 to 2026-03-22 |
| Phase 3 | Workflow implementation (8 workflows) | 2026-03-22 to 2026-03-23 |
| Phase 4 | Template implementation (10 templates) | 2026-03-23 to 2026-03-24 |
| Phase 5 | Integration implementation (13 integrations) | 2026-03-24 to 2026-03-26 |
| Phase 6 | ERP connector development (SAP, Oracle, Workday) | 2026-03-26 to 2026-03-28 |
| Phase 7 | Reference data population (emission factors, SDA benchmarks, carbon prices) | 2026-03-28 to 2026-03-29 |
| Phase 8 | Test suite (900+ tests) | 2026-03-29 to 2026-03-31 |
| Phase 9 | Documentation & Release | 2026-03-31 |

---

## 16. Appendix: SBTi Corporate Standard vs. SME Pathway

### Comparison Table

| Feature | SBTi Corporate Standard (PACK-027) | SBTi SME Pathway (PACK-026) |
|---------|-----------------------------------|----------------------------|
| Eligibility | Any company (no size limit) | <500 employees (varies by country) |
| Near-term Scope 1+2 | 4.2%/yr ACA (1.5C) or SDA | 50% by 2030 (simplified) |
| Near-term Scope 3 | 67%+ coverage, 2.5%/yr minimum | Measure and reduce (no formal target) |
| Long-term target | 90%+ reduction by 2050 | Not required |
| Net-zero commitment | Residual neutralization via permanent CDR | Not applicable |
| Pathway options | ACA, SDA (12 sectors), FLAG | Simplified absolute contraction only |
| Validation process | Queued review (4-12 weeks) | Immediate auto-validation |
| Cost | Varies by revenue ($2,500-$15,000+) | Free |
| Revalidation | Every 5 years | Annual self-assessment |
| Scope 3 screening | Full 15-category materiality assessment | 3 categories recommended |
| Progress reporting | Annual against pathway with recalculation triggers | Annual self-reported |
| FLAG assessment | Required if >20% FLAG emissions | Not applicable |

### PACK-027 SBTi Workflow

1. **Commitment**: Register commitment on SBTi platform (24-month window to submit)
2. **Inventory**: Complete Scope 1+2+3 baseline (enterprise_baseline_engine)
3. **Screening**: Scope 3 materiality screening across all 15 categories
4. **Pathway**: Select ACA/SDA/FLAG based on sector and emission profile
5. **Target**: Define near-term + long-term + net-zero targets with coverage check
6. **Validate**: Run 42-criteria validation (sbti_target_engine)
7. **Submit**: Generate submission package (sbti_target_submission template)
8. **Track**: Annual progress monitoring (annual_inventory_workflow)
9. **Review**: Five-year revalidation cycle
10. **Achieve**: Net-zero attainment and neutralization

---

## 17. Appendix: Multi-Entity Consolidation Worked Example

### Scenario: AcquireCo Ltd

```
AcquireCo Ltd (Parent) -- financial control approach
  |
  +-- SubCo Alpha (100% owned) -- Scope 1: 10,000 tCO2e, Scope 2: 5,000 tCO2e
  |
  +-- SubCo Beta (100% owned) -- Scope 1: 8,000 tCO2e, Scope 2: 3,000 tCO2e
  |   |
  |   +-- SubCo Beta-1 (80% owned by Beta) -- Scope 1: 2,000 tCO2e, Scope 2: 1,000 tCO2e
  |
  +-- JV Gamma (51% owned, financial control) -- Scope 1: 15,000 tCO2e, Scope 2: 7,000 tCO2e
  |
  +-- Associate Delta (33% owned, no control) -- Not consolidated (Scope 3 Cat 15)
```

**Financial Control Consolidation:**
- SubCo Alpha: 100% included = 15,000 tCO2e
- SubCo Beta: 100% included = 11,000 tCO2e
- SubCo Beta-1: 100% included (controlled via Beta) = 3,000 tCO2e
- JV Gamma: 100% included (financial control) = 22,000 tCO2e
- Associate Delta: Excluded from Scope 1+2; included in Scope 3 Cat 15 at 33% equity share

**Sum before eliminations:** 51,000 tCO2e (Scope 1+2)

**Intercompany elimination:**
- SubCo Alpha purchases electricity generated by JV Gamma (1,000 tCO2e)
- This is Scope 2 for Alpha but already Scope 1 for Gamma
- Eliminate 1,000 tCO2e from Alpha's Scope 2 to avoid double-count

**Consolidated total:** 50,000 tCO2e (Scope 1+2)

**Equity Share Alternative:**
- SubCo Alpha: 100% = 15,000
- SubCo Beta: 100% = 11,000
- SubCo Beta-1: 80% = 2,400
- JV Gamma: 51% = 11,220
- Associate Delta: 33% = 33% of Delta's emissions included
- Total: 39,620 tCO2e + 33% of Delta (lower total, but includes associates)

---

## 18. Appendix: Carbon Pricing Worked Example

### Scenario: IndustrialCo with $100/tCO2e Shadow Price

**Step 1: Baseline by Business Unit**

| Business Unit | Scope 1 (tCO2e) | Scope 2 (tCO2e) | Scope 3 (tCO2e) | Total (tCO2e) |
|--------------|-----------------|-----------------|-----------------|--------------|
| BU-Manufacturing | 50,000 | 20,000 | 80,000 | 150,000 |
| BU-Logistics | 30,000 | 5,000 | 10,000 | 45,000 |
| BU-Services | 1,000 | 3,000 | 15,000 | 19,000 |
| BU-R&D | 500 | 2,000 | 5,000 | 7,500 |
| Corporate | 200 | 1,000 | 3,000 | 4,200 |
| **Total** | **81,700** | **31,000** | **113,000** | **225,700** |

**Step 2: Carbon Charge at $100/tCO2e (Scope 1+2 only)**

| Business Unit | Scope 1+2 (tCO2e) | Carbon Charge | As % of BU Revenue |
|--------------|-------------------|---------------|-------------------|
| BU-Manufacturing | 70,000 | $7,000,000 | 2.3% |
| BU-Logistics | 35,000 | $3,500,000 | 3.5% |
| BU-Services | 4,000 | $400,000 | 0.2% |
| BU-R&D | 2,500 | $250,000 | 0.8% |
| Corporate | 1,200 | $120,000 | N/A |
| **Total** | **112,700** | **$11,270,000** | **1.1% of group revenue** |

**Step 3: Investment Decision Impact**

| Project | Standard NPV | Carbon Cost (lifetime) | Carbon-Adjusted NPV | Decision Change? |
|---------|-------------|----------------------|---------------------|-----------------|
| New gas boiler | $2,000,000 | -$4,500,000 | -$2,500,000 | YES: reject |
| Heat pump retrofit | $800,000 | +$3,200,000 (avoided) | $4,000,000 | YES: prioritize |
| Diesel fleet expansion | $1,500,000 | -$3,800,000 | -$2,300,000 | YES: reject |
| EV fleet replacement | -$200,000 | +$2,900,000 (avoided) | $2,700,000 | YES: approve |
| Solar PV installation | $500,000 | +$1,200,000 (avoided) | $1,700,000 | No: already positive |

---

## 19. Appendix: Regulatory Crosswalk Matrix

### Mapping PACK-027 Outputs to Regulatory Requirements

| Data Element | GHG Protocol | SBTi | CDP | TCFD/ISSB S2 | SEC Climate | CSRD E1 | ISO 14064-1 | CA SB 253 |
|-------------|-------------|------|-----|-------------|------------|---------|-------------|----------|
| Scope 1 total (tCO2e) | Chapter 4 | C6 | C6.1 | MT-a | Reg S-X 1504(b)(1) | E1-6(a) | 6.2.2 | Section 38532(c) |
| Scope 2 location (tCO2e) | Chapter 6 | C7 | C6.3 | MT-a | Reg S-X 1504(b)(2) | E1-6(b) | 6.2.3 | Section 38532(c) |
| Scope 2 market (tCO2e) | Scope 2 Guidance | C7 | C6.3 | MT-a | Reg S-X 1504(b)(2) | E1-6(b) | 6.2.3 | Section 38532(c) |
| Scope 3 by category | Scope 3 Standard | C6 | C6.5 | MT-a | Reg S-X 1504(b)(3) | E1-6(c) | 6.2.4 | Section 38532(d) |
| GHG reduction targets | Chapter 9 | C1-C28 | C4.1 | MT-b | Reg S-X 1504(c) | E1-4 | 6.4 | N/A |
| Transition plan | N/A | NZ-C12 | C3.1 | S-a | Reg S-X 1502 | E1-1 | N/A | N/A (SB 261) |
| Scenario analysis | N/A | N/A | C3.2 | S-b | Reg S-X 1502(d) | E1-9 | N/A | SB 261 |
| Internal carbon price | N/A | N/A | C11.3 | S | N/A | E1-8 | N/A | N/A |
| Base year emissions | Chapter 5 | C5 | C5.2 | MT | N/A | E1-6 | 6.3 | N/A |
| Progress vs. targets | Chapter 9 | C4.2 | C4.2 | MT-b | Reg S-X 1504(c)(2) | E1-4 | 6.5 | N/A |
| Data quality / assurance | Chapter 7 | N/A | C10 | N/A | Reg S-X 1505 (attestation) | Audit | ISO 14064-3 | Section 38532(e) |
| Energy consumption | Chapter 6 | N/A | C8 | MT | Reg S-X 1504(d) | E1-5 | 6.2.3 | N/A |
| Renewable energy share | N/A | N/A | C8.2 | MT | N/A | E1-5 | N/A | N/A |

---

## 20. Appendix: External Assurance Preparation

### 20.1 Assurance Levels

| Level | Standard | Evidence Required | Conclusion | Typical Scope |
|-------|----------|------------------|-----------|---------------|
| Limited assurance | ISAE 3410 / ISAE 3000 | Inquiry, analytical procedures, limited testing | "Nothing has come to our attention..." (negative assurance) | Scope 1+2 only, or total GHG statement |
| Reasonable assurance | ISAE 3410 / ISAE 3000 | Detailed testing, corroborating evidence, site visits, third-party confirmations | "In our opinion, the GHG statement is fairly stated..." (positive assurance) | Scope 1+2; extending to Scope 3 over time |
| ISO 14064-3 verification | ISO 14064-3:2019 | Verification plan, evidence gathering, data assessment, materiality evaluation | Verification statement with level of assurance | Full GHG statement per ISO 14064-1 |

### 20.2 Workpaper Set (generated by external_assurance_workflow)

| # | Workpaper | Content | Auditor Use |
|---|-----------|---------|-------------|
| 1 | WP-100: Engagement overview | Scope, boundary, criteria, materiality threshold | Planning and scoping |
| 2 | WP-200: Organizational boundary | Entity hierarchy, consolidation approach, ownership %, control assessments | Boundary verification |
| 3 | WP-300: Scope 1 detail | Per-source emissions with calculation methodology, emission factors, activity data | Substantive testing |
| 4 | WP-400: Scope 2 detail | Location-based and market-based with grid factors, contractual instruments | Substantive testing |
| 5 | WP-500: Scope 3 detail | Per-category with methodology, data quality scores, materiality justification for excluded categories | Substantive testing |
| 6 | WP-600: Emission factors | Full register of all emission factors used with source, version, vintage | Factor verification |
| 7 | WP-700: Data quality assessment | Per-category DQ scoring against GHG Protocol hierarchy | Data quality evaluation |
| 8 | WP-800: Base year recalculation | Trigger assessment, significance calculation, recalculated values, justification | Consistency testing |
| 9 | WP-900: Consolidation reconciliation | Entity-level to group-level reconciliation, intercompany eliminations | Consolidation verification |
| 10 | WP-1000: Calculation trace | Step-by-step calculation for sampled items (60 sample items minimum) | Recalculation testing |
| 11 | WP-1100: Provenance hashes | SHA-256 hashes for all inputs, outputs, and intermediate calculations | Integrity verification |
| 12 | WP-1200: Control documentation | Data collection controls, approval workflows, change management | Control testing |
| 13 | WP-1300: Management assertion | Template management representation letter covering completeness, accuracy, methodology | Management responsibilities |
| 14 | WP-1400: Prior year comparison | Year-over-year analysis with variance explanations for all >5% changes | Analytical review |
| 15 | WP-1500: Findings register | Open issues from pre-assurance testing with severity, remediation plan | Issue tracking |

### 20.3 Pre-Assurance Control Testing

PACK-027 performs automated pre-assurance testing before generating the workpaper package:

| Test Category | Test Description | Pass Criteria | Remediation if Fail |
|--------------|-----------------|---------------|---------------------|
| Completeness | All entities in hierarchy have submitted data | 100% entity coverage | Escalation to entity data owners |
| Completeness | All material Scope 3 categories have calculations | 100% material categories | Run calculations for missing categories |
| Accuracy | Recalculation of 60 sampled line items | Within +/-1% of original | Investigate and correct errors |
| Accuracy | Emission factor version check | All factors from current year publications | Update stale emission factors |
| Consistency | Year-over-year variance for each entity > 10% | All variances explained | Document variance drivers |
| Consistency | Scope 2 dual reporting reconciliation | Location-based >= market-based (or explained) | Verify contractual instruments |
| Cut-off | All data within reporting period boundaries | No out-of-period data included | Correct period allocation |
| Classification | Scope 1/2/3 classification per GHG Protocol | 100% correctly classified | Reclassify misclassified items |
| Existence | Emission factor source verification | All factors traceable to published source | Replace unsourced factors |
| Valuation | GWP values match IPCC AR6 | 100% match | Update GWP values |

### 20.4 Assurance Timeline

| Week | Activity | PACK-027 Support |
|------|----------|-----------------|
| 1-2 | Planning and scoping | Generate WP-100, WP-200; define materiality threshold |
| 3-4 | Risk assessment | Provide data quality matrix, variance analysis, control documentation |
| 5-8 | Fieldwork (testing) | Generate WP-300 through WP-1100; provide sample selections; answer queries |
| 9-10 | Findings and remediation | Update WP-1500; remediate findings; re-run calculations if needed |
| 11-12 | Reporting | Generate assurance statement template (WP-1300); finalize workpapers |

---

## 21. Appendix: Data Quality Improvement Program

### 21.1 Enterprise Data Quality Maturity Model

| Maturity Level | Description | Typical Year | DQ Score (weighted avg) | Characteristics |
|---------------|-------------|-------------|------------------------|-----------------|
| Level 1: Ad Hoc | Spreadsheet-based, inconsistent methodology, limited Scope 3 | Pre-implementation | 4.0-5.0 | Manual data collection, spend-based Scope 3, no formal DQ process |
| Level 2: Managed | Platform-implemented, standardized methodology, basic Scope 3 | Year 1 | 3.0-4.0 | ERP-connected, activity-based S1+S2, spend-based S3, DQ baseline established |
| Level 3: Defined | Systematic DQ improvement, expanding activity-based Scope 3 | Year 2 | 2.5-3.0 | Top 50 suppliers engaged, activity data for 5+ S3 categories, DQ targets set |
| Level 4: Quantitatively Managed | Continuous DQ monitoring, predominantly activity-based | Year 3-4 | 2.0-2.5 | CDP Supply Chain data, 80%+ activity-based, automated DQ alerts |
| Level 5: Optimizing | Near-real-time, verified data, best-in-class | Year 5+ | 1.5-2.0 | PACT data exchange, verified supplier data, continuous monitoring |

### 21.2 Data Quality Improvement Actions by Scope 3 Category

| Category | Current Typical DQ | Target DQ | Key Improvement Actions |
|----------|-------------------|-----------|------------------------|
| Cat 1 (Purchased Goods) | Level 4 (spend-based) | Level 2 (supplier-specific) | CDP Supply Chain, PACT data exchange, direct engagement of top 50 suppliers |
| Cat 2 (Capital Goods) | Level 4 (spend-based) | Level 3 (average-data) | Asset-specific EF by equipment type, manufacturer data |
| Cat 3 (Fuel & Energy) | Level 2 (auto-calculated) | Level 1 (metered + factors) | Automated calculation from Scope 1+2 with verified factors |
| Cat 4 (Upstream Transport) | Level 3 (average distance) | Level 2 (route-specific) | TMS integration, carrier-reported emissions, GLEC Framework |
| Cat 5 (Waste) | Level 3 (average/employee) | Level 2 (waste records) | Waste contractor reporting, waste stream analysis |
| Cat 6 (Business Travel) | Level 3 (spend-based) | Level 2 (distance-based) | Travel management system integration, per-trip calculation |
| Cat 7 (Employee Commuting) | Level 4 (national average) | Level 3 (survey-based) | Annual commute survey, HR location data, remote work tracking |
| Cat 8 (Upstream Leased) | Level 4 (floor area proxy) | Level 2 (metered) | Sub-metering, landlord energy data, green lease provisions |
| Cat 9 (Downstream Transport) | Level 4 (revenue proxy) | Level 3 (route model) | Distribution network model, carrier data, last-mile tracking |
| Cat 10 (Processing) | Level 4 (industry average) | Level 3 (process-specific) | Customer processing data, industry process factors |
| Cat 11 (Use of Sold Products) | Level 3 (product specification) | Level 2 (measured use) | Product energy labeling, IoT usage data, lifecycle models |
| Cat 12 (End-of-Life) | Level 4 (waste stream proxy) | Level 3 (product-specific) | Product waste stream analysis, recycling rate data |
| Cat 13 (Downstream Leased) | Level 3 (building type) | Level 2 (tenant data) | Tenant energy reporting, green lease data provisions |
| Cat 14 (Franchises) | Level 3 (average franchise) | Level 2 (per-franchise) | Franchise reporting system, per-store energy data |
| Cat 15 (Investments) | Level 4 (revenue proxy) | Level 2 (PCAF scores 1-3) | PCAF methodology, portfolio company engagement, disclosed data |

### 21.3 Data Quality Guardian Alerts

The `data_quality_guardian.py` integration continuously monitors data quality and generates alerts:

| Alert Type | Trigger | Severity | Action Required |
|-----------|---------|----------|----------------|
| Missing entity data | Entity has not submitted data within 30 days of deadline | Critical | Contact entity data owner; escalate after 7 days |
| Data quality regression | Category DQ score decreased vs. prior year | High | Investigate cause; restore prior DQ level |
| Stale emission factor | Emission factor is >2 years old | Medium | Update to current year publication |
| Outlier detected | Entity emission value >3 standard deviations from peer group | High | Investigate anomaly; confirm or correct |
| Coverage gap | Material Scope 3 category (<1% total) has no calculation | Critical | Initiate data collection for missing category |
| Supplier response lag | CDP/questionnaire response rate below 50% target | Medium | Send reminder; escalate engagement tier |
| Reconciliation mismatch | ERP data and utility bill data differ by >5% | High | Investigate source; reconcile discrepancy |
| Base year trigger | Structural change potentially exceeding 5% threshold | Critical | Run base year recalculation assessment |

---

## 22. Appendix: Enterprise Implementation Guide

### 22.1 Implementation Phases

| Phase | Duration | Activities | Deliverables | Key Stakeholders |
|-------|----------|-----------|-------------|-----------------|
| Phase 1: Scoping | 1-2 weeks | Define organizational boundary; identify entities; select consolidation approach; assess data availability; confirm regulatory requirements | Scoping document, entity register, data availability assessment | CSO, IT, Finance |
| Phase 2: Configuration | 1-2 weeks | Install PACK-027; select preset; configure entity hierarchy; set up ERP connectors; define user roles; run setup wizard | Configured system, connected ERPs, user accounts | IT, Sustainability, ERP team |
| Phase 3: Data Collection | 2-4 weeks | Extract data from ERPs; collect utility bills; distribute supplier questionnaires; collect fleet data; run data quality assessment | Complete activity data set, DQ baseline score | Entity data owners, procurement, fleet |
| Phase 4: Baseline Calculation | 1-2 weeks | Run enterprise_baseline_engine for all entities; run multi_entity_consolidation_engine; review and validate results | Draft enterprise GHG baseline, DQ matrix | Sustainability team, finance |
| Phase 5: Target Setting | 1-2 weeks | Run sbti_target_engine; validate 42 criteria; define annual milestones; prepare submission package | SBTi target package, criteria validation report | CSO, board |
| Phase 6: Reporting Setup | 1-2 weeks | Configure templates; run CDP auto-population; generate TCFD report; set up board dashboard; generate first assurance workpapers | Report templates configured, first outputs generated | Sustainability, IR, board secretary |
| Phase 7: Go-Live | 1 week | Final validation; health check; user training; hand-off to sustainability team | Production system, trained users, support handover | All stakeholders |

**Total implementation: 8-15 weeks** (depending on entity count, ERP complexity, and data availability)

### 22.2 Implementation Team Requirements

| Role | Responsibility | Allocation (% FTE) | Duration |
|------|---------------|-------------------|----------|
| Project Manager | Overall implementation governance; stakeholder coordination | 50% | Full duration |
| Sustainability Lead | Methodology decisions; data validation; target setting | 75% | Full duration |
| Data Engineer | ERP connector setup; data extraction; quality assurance | 100% | Phase 2-5 |
| ERP Consultant | SAP/Oracle/Workday integration; data mapping | 50% | Phase 2-3 |
| Entity Data Coordinators | Per-entity data collection and validation | 25% each | Phase 3-4 |
| Finance Partner | Financial integration setup; carbon pricing; P&L mapping | 25% | Phase 5-6 |
| IT Security | Authentication setup; RBAC configuration; data protection | 10% | Phase 2, 7 |
| GreenLang Implementation Partner | Technical implementation; configuration; training | 100% | Full duration |

### 22.3 Change Management

Enterprise climate program implementation requires organizational change management:

| Dimension | Challenge | PACK-027 Mitigation |
|-----------|----------|---------------------|
| Data ownership | Unclear who owns GHG data (sustainability vs. finance vs. operations) | Setup wizard defines entity data owners; RBAC enforces accountability |
| Process integration | GHG reporting treated as annual project, not embedded process | Automated ERP extraction; continuous monitoring; quarterly workflows |
| Board engagement | Board lacks climate literacy for governance role | Executive dashboard designed for non-specialist audience; commentary templates |
| Cross-functional alignment | Sustainability, finance, procurement, operations working in silos | Integrated platform; carbon pricing bridges finance; supply chain heatmap bridges procurement |
| Entity cooperation | Subsidiary resistance to additional data reporting burden | Minimal manual data collection (ERP-automated); entity-level dashboards showing value |

---

## 23. Appendix: Scope 4 Avoided Emissions Methodology

### 23.1 When to Use Scope 4

Scope 4 (avoided emissions) is appropriate when an enterprise's products or services demonstrably displace higher-emission alternatives. Common enterprise contexts:

| Sector | Product/Service | Baseline Displaced | Typical Avoided/Footprint Ratio |
|--------|----------------|-------------------|-------------------------------|
| Technology | Cloud computing (SaaS) | On-premise data centers | 2:1 to 10:1 |
| Technology | Video conferencing | Business air travel | 50:1 to 200:1 |
| Energy | Renewable energy (wind/solar) | Grid average or marginal fossil | 5:1 to 50:1 |
| Energy | Energy efficiency equipment | Incumbent equipment | 2:1 to 20:1 |
| Automotive | Electric vehicles | ICE vehicles | 2:1 to 5:1 |
| Buildings | High-performance insulation | Standard insulation | 5:1 to 15:1 |
| Manufacturing | Lightweight materials | Standard materials | 1:1 to 5:1 |
| Finance | Green loans/bonds | Standard financing | Indirect; measured via portfolio |

### 23.2 Conservative Principles (mandatory in PACK-027)

1. **Market average baseline**: Always use market average (not worst-in-class) as the baseline scenario. If a regulatory minimum exists, use that.
2. **Full lifecycle inclusion**: Include full cradle-to-grave emissions of the assessed product, not just use-phase savings.
3. **Rebound effect deduction**: Quantify and deduct rebound effects (e.g., increased usage due to lower cost).
4. **Attribution share**: For enabling effects (e.g., software enabling energy savings), apply conservative attribution (typically 10-30%, not 100%).
5. **Separate reporting**: Never net avoided emissions against Scope 1+2+3 footprint. Always report separately.
6. **Annual recalculation**: Recalculate annually as market baselines change (e.g., grid decarbonization reduces avoided emissions from renewable energy).
7. **Uncertainty disclosure**: Provide P10-P90 uncertainty range for all avoided emissions claims.
8. **No double-counting**: Ensure avoided emissions are not also counted in a customer's Scope 2 or Scope 3 reduction claims.

### 23.3 Avoided Emissions Reporting Format

```
Company: ExampleCorp
Reporting Year: 2025

Scope 1+2+3 Footprint: 500,000 tCO2e

Scope 4 Avoided Emissions (reported separately):
  Product A (EV fleet):       150,000 tCO2e avoided (baseline: ICE fleet average)
  Product B (Cloud services):  80,000 tCO2e avoided (baseline: on-premise)
  Product C (LED lighting):    45,000 tCO2e avoided (baseline: halogen/CFL)

Total Avoided Emissions: 275,000 tCO2e
Avoided/Footprint Ratio: 0.55x

Methodology: WBCSD Avoided Emissions Guidance (2023)
Baseline: Market average per product category
Uncertainty: +/- 30% (P10: 192,500; P90: 357,500)
Attribution: 100% for direct substitution; 25% for enabling effects
```

---

## 24. Appendix: Enterprise Net Zero Maturity Assessment

### 24.1 Maturity Framework

PACK-027 includes a built-in maturity assessment that evaluates the enterprise across 10 dimensions:

| # | Dimension | Level 1 (Beginning) | Level 3 (Developing) | Level 5 (Leading) |
|---|-----------|--------------------|--------------------|------------------|
| 1 | Governance | No board oversight; ad hoc responsibility | Board receives annual climate update; dedicated sustainability team | Board climate committee; C-suite KPIs tied to emissions; TCFD-aligned governance |
| 2 | GHG Inventory | Scope 1+2 only; manual; incomplete | Full Scope 1+2; partial Scope 3 (5-8 categories); annual calculation | Full Scope 1+2+3 (15 categories); quarterly updates; activity-based; DQ Level 2+ |
| 3 | Target Setting | No targets or vague commitments | SBTi committed; near-term target submitted | SBTi validated near-term + long-term + net-zero; annual pathway tracking |
| 4 | Reduction Strategy | Generic action list; no quantification | Prioritized actions with cost-benefit; partial implementation | MACC-driven roadmap; >50% actions implemented; measurable emission reduction |
| 5 | Supply Chain | No Scope 3 engagement | Top 50 suppliers engaged; CDP Supply Chain participant | Multi-tier mapping; top 80% Scope 3 engaged; supplier SBTi adoption >30% |
| 6 | Data Quality | Level 4-5 (spend-based, proxies) | Level 3 (average-data, partial activity) | Level 1-2 (supplier-specific, verified, PACT) |
| 7 | Financial Integration | No carbon in financial decisions | Shadow carbon price for major CapEx | Full internal carbon fee; carbon-adjusted P&L; CBAM compliance |
| 8 | Reporting | Single framework (e.g., CDP only) | 2-3 frameworks (CDP, TCFD, one regulator) | 5+ frameworks auto-generated from single source of truth |
| 9 | Assurance | No external verification | Limited assurance on Scope 1+2 | Reasonable assurance on Scope 1+2+3; workpapers auto-generated |
| 10 | Innovation | No Scope 4; no scenario modeling | Basic scenario analysis; considering avoided emissions | Monte Carlo scenarios; verified Scope 4; internal carbon pricing |

### 24.2 Maturity Scoring

| Overall Score | Maturity Level | Interpretation |
|--------------|---------------|----------------|
| 10-20 | Beginning | Significant gaps across most dimensions; foundation-building required |
| 21-30 | Emerging | Some dimensions addressed; need systematic approach |
| 31-38 | Developing | Most dimensions progressing; targeted improvements needed |
| 39-45 | Advanced | Strong performance; optimizing remaining gaps |
| 46-50 | Leading | Best-in-class; continuous improvement focus |

The maturity assessment is run at implementation start (baseline maturity) and annually thereafter to track program development. It feeds into the executive dashboard as a summary "Net Zero Readiness Score" for board reporting.

---

## 25. Future Roadmap

- **PACK-027 v1.1: Real-Time Carbon Monitoring** -- IoT sensor integration for continuous emissions monitoring (CEMS equivalent), real-time dashboard updates, anomaly detection, and automated regulatory reporting triggers
- **PACK-027 v1.2: AI-Augmented Data Quality** -- Machine learning for emission factor selection optimization, automated data gap identification and filling, supplier data verification, and anomaly detection (note: AI for data quality only, never for calculations)
- **PACK-027 v1.3: Extended Regulatory Coverage** -- Japan TCFD/SSBJ disclosure, Hong Kong ESG reporting, Singapore ISSB adoption, Brazil SEC climate rule, additional jurisdictions
- **PACK-027 v1.4: Nature and Biodiversity Integration** -- TNFD alignment, SBTN (Science Based Targets for Nature), biodiversity footprint, water risk assessment, integrated climate-nature reporting
- **PACK-027 v1.5: Value Chain Digital Twin** -- Digital twin of enterprise value chain with real-time emission modeling, what-if simulation, and supplier substitution analysis
- **PACK-028: Enterprise Climate Finance Pack** -- Green bond framework development, transition finance classification, ESG-linked lending parameters, climate VAR, NGFS scenario alignment for financial institutions

---

## 26. Glossary

| Term | Definition |
|------|-----------|
| ACA | Absolute Contraction Approach -- SBTi methodology requiring absolute emission reductions at a fixed annual rate (4.2%/yr for 1.5C alignment) |
| BAU | Business As Usual -- scenario assuming no additional climate action beyond current policies |
| CBAM | Carbon Border Adjustment Mechanism -- EU regulation imposing carbon costs on imports from countries without equivalent carbon pricing |
| CDR | Carbon Dioxide Removal -- permanent removal of CO2 from the atmosphere (e.g., DACS, BECCS, enhanced weathering) |
| CEMS | Continuous Emissions Monitoring System -- real-time measurement of emissions from point sources |
| CRREM | Carbon Risk Real Estate Monitor -- tool for assessing stranding risk of real estate assets under different decarbonization pathways |
| CSRD | Corporate Sustainability Reporting Directive -- EU directive mandating sustainability disclosures under ESRS standards |
| DQ Level | Data Quality Level -- 1 (best, verified primary data) through 5 (worst, proxy/extrapolation) per GHG Protocol |
| EEIO | Environmentally-Extended Input-Output -- economic model mapping spend to emissions via sector-level emission factors |
| ESRS | European Sustainability Reporting Standards -- disclosure standards under CSRD; E1 covers climate change |
| ETS | Emissions Trading System -- cap-and-trade carbon market (e.g., EU ETS, UK ETS) |
| FINZ | Financial Institutions Net-Zero -- SBTi standard for financial institution portfolio-level targets |
| FLAG | Forest, Land and Agriculture -- SBTi guidance for land use sector emissions; required if >20% of total emissions |
| GHG Protocol | Greenhouse Gas Protocol -- the global standard for corporate GHG accounting (Corporate Standard, Scope 2 Guidance, Scope 3 Standard) |
| GWP-100 | Global Warming Potential over 100 years -- metric for comparing greenhouse gas potency relative to CO2 |
| ISAE 3410 | International Standard on Assurance Engagements for GHG Statements -- assurance standard for GHG reports |
| ISSB S2 | International Sustainability Standards Board Standard S2 -- global climate disclosure standard (successor to TCFD) |
| MACC | Marginal Abatement Cost Curve -- visualization ranking emission reduction options by cost per tCO2e avoided |
| NCV | Net Calorific Value -- energy content of fuel (lower heating value); used in combustion emission calculations |
| NGFS | Network for Greening the Financial System -- central banking group providing climate scenario frameworks |
| PACT | Partnership for Carbon Transparency -- WBCSD initiative for standardized exchange of product-level carbon data |
| PCAF | Partnership for Carbon Accounting Financials -- standard for measuring financed emissions across asset classes |
| PCF | Product Carbon Footprint -- lifecycle GHG emissions of a specific product |
| PPA | Power Purchase Agreement -- long-term contract to purchase renewable electricity; qualifies as contractual instrument for market-based Scope 2 |
| REC/GO | Renewable Energy Certificate / Guarantee of Origin -- tradeable certificate proving renewable electricity generation |
| SBTi | Science Based Targets initiative -- framework for setting emission reduction targets aligned with Paris Agreement |
| SDA | Sectoral Decarbonization Approach -- SBTi methodology for sector-specific intensity convergence pathways |
| TCFD | Task Force on Climate-related Financial Disclosures -- framework for climate risk disclosure (now absorbed into ISSB) |
| VCMI | Voluntary Carbon Markets Integrity Initiative -- framework for credible net-zero and carbon neutral claims |
| WTT | Well-to-Tank -- upstream emissions from fuel extraction, processing, and delivery (part of Scope 3 Cat 3) |

---

*Document Version: 1.0.0 | Last Updated: 2026-03-19 | Status: Draft*
