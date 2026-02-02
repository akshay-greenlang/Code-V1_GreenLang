# GreenLang Agent Factory - Regulatory Agents Master Plan

**Version:** 1.0.0
**Date:** December 3, 2025
**Author:** GL-RegulatoryIntelligence
**Status:** APPROVED - Ready for Implementation

---

## Executive Summary

This document provides a comprehensive, prioritized to-do list for building 10 regulatory compliance agents using the GreenLang Agent Factory. Each agent is specified with regulatory deadlines, input data requirements, calculation methodologies, emission factors, tools, and output formats.

### Current State
- **Existing Agents:** 3 (Fuel Analyzer, CBAM Importer, Building Energy)
- **Target:** 10 additional agents for regulatory compliance
- **Deadline-Driven Priority:** Agents ordered by regulatory enforcement dates

### Regulatory Deadline Overview

| Priority | Agent | Regulation | Deadline | Days Until |
|----------|-------|------------|----------|------------|
| P0 | EUDR Compliance Agent | EU Deforestation Regulation | Dec 30, 2025 | 27 days |
| P1 | SB 253 Disclosure Agent | California Climate Disclosure | Jun 30, 2026 | 209 days |
| P1 | CSRD Reporting Agent | Corporate Sustainability Reporting | Jan 1, 2024 (effective) | In force |
| P2 | EU Taxonomy Agent | Green Investment Classification | Ongoing | Continuous |
| P2 | Green Claims Agent | Anti-Greenwashing Directive | Sep 27, 2026 | 298 days |
| P3 | CSDDD Agent | Supply Chain Due Diligence | Jul 26, 2027 | 600 days |
| P3 | PCF/DPP Agent | Product Carbon Footprint | Feb 2027 | ~450 days |
| P4 | Scope 3 Agent | GHG Protocol Supply Chain | Ongoing | Continuous |
| P4 | SBTi Validation Agent | Science-Based Targets | Ongoing | Continuous |
| P4 | Carbon Offset Agent | VCS/Gold Standard | Ongoing | Continuous |

---

## Agent 1: EUDR Compliance Agent [CRITICAL - P0]

### Regulation Details
- **Regulation:** EU Deforestation-Free Products Regulation (EU) 2023/1115
- **Deadline:** December 30, 2025 (Large operators), June 30, 2026 (SMEs)
- **Penalty:** Up to 4% of annual EU turnover
- **Scope:** Cattle, cocoa, coffee, palm oil, rubber, soya, wood and derived products

### Agent Specification

```yaml
agent_id: gl-eudr-compliance-v1
name: EUDR Deforestation Compliance Agent
version: "1.0.0"
type: due-diligence-validator
priority: P0-CRITICAL
deadline: "2025-12-30"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `commodity_type` | enum | User input | Yes |
| `origin_country` | ISO 3166-1 | User input | Yes |
| `geolocation_data` | GeoJSON | GPS/satellite | Yes |
| `production_date` | date | Supplier | Yes |
| `operator_info` | object | Registration | Yes |
| `supply_chain_docs` | array | Invoices/certs | Yes |
| `quantity_kg` | float | Shipment | Yes |
| `cn_code` | string | Customs | Yes |

### Calculations Needed

1. **Deforestation Risk Assessment**
   - Formula: `risk_score = f(country_risk, region_risk, supplier_history, certification_status)`
   - Output: High/Medium/Low risk classification

2. **Land Cover Change Detection**
   - Formula: `deforestation_flag = land_cover(T2) - land_cover(T1) < threshold`
   - Reference date: December 31, 2020 (EUDR cutoff)

3. **Supply Chain Traceability Score**
   - Formula: `traceability = (verified_nodes / total_nodes) * 100`
   - Target: 100% traceability to production plot

### Emission Factors Needed

| Factor | Source | Value | Unit |
|--------|--------|-------|------|
| Deforestation CO2 | IPCC | 450 | tCO2/ha |
| Forest degradation | IPCC | 150 | tCO2/ha |
| Land use change | IPCC | varies | tCO2/ha |

### Tools Required (5)

1. **geolocation_validator** - Validates GPS coordinates and plot polygons
2. **land_cover_analyzer** - Analyzes satellite imagery for deforestation
3. **supply_chain_tracer** - Traces commodities to production plots
4. **risk_assessment_engine** - Calculates country/operator risk scores
5. **eudr_schema_validator** - Validates against EU DDS schema

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| Due Diligence Statement | JSON | EU DDS v1.0 | EU Registry |
| Risk Assessment Report | PDF | Internal | Compliance Team |
| Traceability Map | GeoJSON | OGC | Auditors |
| Non-Compliance Alerts | Email/API | Custom | Risk Managers |

### Implementation Tasks

- [ ] **Week 1:** Design geolocation validator for GPS/polygon validation
- [ ] **Week 1:** Integrate satellite imagery API (Sentinel-2, Planet)
- [ ] **Week 2:** Build country risk database (EC benchmarking system)
- [ ] **Week 2:** Implement supply chain traceability graph
- [ ] **Week 3:** Create EU DDS JSON schema validator
- [ ] **Week 3:** Build due diligence statement generator
- [ ] **Week 4:** Create 200 golden tests (geolocation, risk, traceability)
- [ ] **Week 4:** Certification and deployment

---

## Agent 2: California SB 253 Disclosure Agent [P1]

### Regulation Details
- **Regulation:** California Climate Corporate Data Accountability Act (SB 253)
- **Deadline:** June 30, 2026 (Scope 1, 2), June 30, 2027 (Scope 3)
- **Penalty:** Up to $500,000 per year for non-compliance
- **Scope:** Companies with $1B+ revenue doing business in California

### Agent Specification

```yaml
agent_id: gl-sb253-disclosure-v1
name: SB 253 Climate Disclosure Agent
version: "1.0.0"
type: emissions-disclosure
priority: P1-HIGH
deadline: "2026-06-30"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `company_profile` | object | Registration | Yes |
| `california_revenue` | float | Finance | Yes |
| `facility_data` | array | Operations | Yes |
| `fuel_consumption` | array | ERP/meters | Yes |
| `electricity_usage` | array | Utility bills | Yes |
| `supply_chain_data` | array | Procurement | Scope 3 |
| `reporting_year` | int | User input | Yes |

### Calculations Needed

1. **Scope 1 Direct Emissions**
   - Formula: `scope1 = SUM(fuel_consumption_i * emission_factor_i)`
   - GHG Protocol Corporate Standard methodology

2. **Scope 2 Indirect Emissions**
   - Location-based: `scope2_loc = electricity_kwh * grid_factor`
   - Market-based: `scope2_mkt = electricity_kwh * supplier_factor`

3. **Scope 3 Value Chain Emissions** (15 categories)
   - Formula: `scope3 = SUM(category_emissions_1..15)`
   - Required categories: 1-8, 11, 12 (minimum)

4. **Third-Party Assurance Level**
   - Limited assurance (2026), Reasonable assurance (2030)

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| Stationary combustion | EPA eGRID | Annual |
| Mobile combustion | EPA GHG EF Hub | Annual |
| Grid electricity | CARB, EPA | Annual |
| Scope 3 categories | EPA EEIO, Exiobase | Annual |

### Tools Required (5)

1. **emissions_calculator** - GHG Protocol-compliant calculations
2. **egrid_factor_lookup** - EPA eGRID emission factors
3. **carb_verifier** - California Air Resources Board alignment
4. **assurance_tracker** - Third-party verification status
5. **sb253_report_generator** - CARB reporting format

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| SB 253 Annual Report | PDF/XML | CARB format | CARB Registry |
| GHG Inventory | Excel | GHG Protocol | Internal |
| Assurance Statement | PDF | ISAE 3410 | Public |
| Data Quality Report | PDF | Internal | Auditors |

### Implementation Tasks

- [ ] **Week 1:** Integrate EPA eGRID and GHG EF Hub databases
- [ ] **Week 1:** Build GHG Protocol-compliant calculator
- [ ] **Week 2:** Implement all 15 Scope 3 category calculators
- [ ] **Week 2:** Create CARB reporting format generator
- [ ] **Week 3:** Build assurance tracking workflow
- [ ] **Week 3:** Implement California revenue threshold checker
- [ ] **Week 4:** Create 300 golden tests (all scopes, all categories)
- [ ] **Week 4:** Certification and deployment

---

## Agent 3: CSRD Reporting Agent [P1]

### Regulation Details
- **Regulation:** Corporate Sustainability Reporting Directive (EU) 2022/2464
- **Effective:** January 1, 2024 (large PIEs), January 1, 2025 (large companies)
- **Penalty:** Member state defined (up to 10M EUR in some jurisdictions)
- **Scope:** ~50,000 EU companies + non-EU with significant EU operations

### Agent Specification

```yaml
agent_id: gl-csrd-reporting-v1
name: CSRD ESRS Reporting Agent
version: "1.0.0"
type: sustainability-reporting
priority: P1-HIGH
deadline: "2024-01-01"  # Already in force
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `company_profile` | object | Registration | Yes |
| `double_materiality` | object | Assessment | Yes |
| `e1_climate_data` | object | Operations | Yes |
| `e2_pollution_data` | object | Operations | Materiality |
| `e3_water_data` | object | Operations | Materiality |
| `e4_biodiversity_data` | object | Operations | Materiality |
| `e5_circular_economy` | object | Operations | Materiality |
| `s1_own_workforce` | object | HR | Yes |
| `s2_value_chain_workers` | object | Supply chain | Materiality |
| `s3_communities` | object | Stakeholders | Materiality |
| `s4_consumers` | object | Sales | Materiality |
| `g1_governance` | object | Corporate | Yes |

### Calculations Needed

1. **Double Materiality Assessment**
   - Impact materiality: `impact_score = stakeholder_impact * severity * likelihood`
   - Financial materiality: `fin_score = risk_exposure * probability * time_horizon`

2. **ESRS E1 Climate Metrics**
   - Scope 1, 2, 3 emissions (GHG Protocol aligned)
   - Energy consumption by source
   - Climate transition plan progress

3. **ESRS Datapoint Completeness**
   - Formula: `completeness = filled_datapoints / required_datapoints * 100`
   - Target: 100% for mandatory, >80% for material topics

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| GHG emissions | IPCC, IEA | Annual |
| Energy conversion | IEA | Annual |
| Water intensity | WRI Aqueduct | Annual |
| Waste factors | Eurostat | Annual |

### Tools Required (5)

1. **materiality_assessor** - Double materiality matrix generator
2. **esrs_gap_checker** - ESRS datapoint completeness analyzer
3. **esrs_calculator** - ESRS metric calculations (E1-E5, S1-S4, G1)
4. **xbrl_generator** - ESEF/iXBRL format generator
5. **csrd_validator** - ESRS standard compliance checker

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| CSRD Sustainability Report | iXBRL | ESRS Taxonomy | EU Single Access Point |
| Management Report Integration | DOCX/PDF | Company format | Annual Report |
| Materiality Matrix | PDF | GRI/ESRS | Stakeholders |
| Datapoint Inventory | Excel | EFRAG | Internal |

### Implementation Tasks

- [ ] **Week 1:** Build double materiality assessment engine
- [ ] **Week 1:** Create ESRS E1-E5 calculators
- [ ] **Week 2:** Build ESRS S1-S4 data collectors
- [ ] **Week 2:** Implement ESRS G1 governance framework
- [ ] **Week 3:** Create iXBRL/ESEF report generator
- [ ] **Week 3:** Build ESRS gap analysis tool
- [ ] **Week 4:** Create 500 golden tests (all ESRS standards)
- [ ] **Week 4:** Certification and deployment

---

## Agent 4: EU Taxonomy Agent [P2]

### Regulation Details
- **Regulation:** EU Taxonomy Regulation (EU) 2020/852
- **Deadline:** Ongoing (reporting cycle aligned with CSRD)
- **Penalty:** Integrated with CSRD penalties
- **Scope:** CSRD-subject companies, financial market participants

### Agent Specification

```yaml
agent_id: gl-eu-taxonomy-v1
name: EU Taxonomy Alignment Agent
version: "1.0.0"
type: green-investment-classifier
priority: P2-MEDIUM
deadline: "ongoing"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `economic_activities` | array | Finance | Yes |
| `revenue_breakdown` | object | Finance | Yes |
| `capex_data` | object | Finance | Yes |
| `opex_data` | object | Finance | Yes |
| `environmental_data` | object | Operations | Yes |
| `dnsh_assessments` | array | Assessment | Yes |
| `minimum_safeguards` | object | HR/Legal | Yes |

### Calculations Needed

1. **Taxonomy Eligibility Assessment**
   - Formula: `eligible = activity IN taxonomy_delegated_acts`
   - Check against 6 environmental objectives

2. **Substantial Contribution Criteria**
   - Technical screening criteria per activity
   - Quantitative thresholds (e.g., <100gCO2/kWh for electricity)

3. **DNSH (Do No Significant Harm) Assessment**
   - Check all 6 objectives for each activity
   - Formula: `dnsh_pass = ALL(objective_checks) == True`

4. **Taxonomy KPIs**
   - `taxonomy_aligned_revenue = aligned_revenue / total_revenue * 100`
   - `taxonomy_aligned_capex = aligned_capex / total_capex * 100`
   - `taxonomy_aligned_opex = aligned_opex / total_opex * 100`

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| Activity emission thresholds | EU Delegated Acts | Per update |
| Energy efficiency benchmarks | EU JRC | Annual |
| Water stress indicators | WRI Aqueduct | Annual |
| Biodiversity indicators | IBAT | Ongoing |

### Tools Required (5)

1. **activity_classifier** - NACE code to Taxonomy activity mapping
2. **tsc_evaluator** - Technical screening criteria evaluator
3. **dnsh_checker** - Do No Significant Harm assessment
4. **safeguards_validator** - Minimum safeguards compliance
5. **taxonomy_kpi_calculator** - Revenue/CapEx/OpEx alignment

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| Taxonomy Disclosure | iXBRL | ESRS E1 | CSRD Report |
| Activity Assessment | Excel | Internal | Finance |
| DNSH Documentation | PDF | EU format | Auditors |
| Investment Screening | JSON | API | Investors |

### Implementation Tasks

- [ ] **Week 1:** Build NACE to Taxonomy activity mapper
- [ ] **Week 1:** Implement all technical screening criteria
- [ ] **Week 2:** Build DNSH assessment engine
- [ ] **Week 2:** Implement minimum safeguards checker
- [ ] **Week 3:** Create Taxonomy KPI calculators
- [ ] **Week 3:** Build investor reporting API
- [ ] **Week 4:** Create 300 golden tests (all activities, all objectives)
- [ ] **Week 4:** Certification and deployment

---

## Agent 5: Green Claims Directive Agent [P2]

### Regulation Details
- **Regulation:** Green Claims Directive (proposal COM/2023/166)
- **Deadline:** September 27, 2026 (estimated transposition)
- **Penalty:** Up to 4% annual turnover, banned from public procurement
- **Scope:** All B2C environmental claims in EU market

### Agent Specification

```yaml
agent_id: gl-green-claims-v1
name: Green Claims Verification Agent
version: "1.0.0"
type: claim-substantiation
priority: P2-MEDIUM
deadline: "2026-09-27"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `claim_text` | string | Marketing | Yes |
| `claim_type` | enum | Classification | Yes |
| `product_lca` | object | LCA study | Yes |
| `environmental_footprint` | object | PEF study | If applicable |
| `comparison_baseline` | object | If comparative | Conditional |
| `certification_labels` | array | Third-party | If applicable |
| `supporting_evidence` | array | Documents | Yes |

### Calculations Needed

1. **Claim Substantiation Score**
   - Formula: `substantiation = evidence_quality * coverage * recency`
   - Minimum threshold: 80% for approval

2. **PEF/OEF Alignment Check**
   - Environmental footprint calculation per EU PEF methodology
   - 16 impact categories assessment

3. **Comparative Claim Validation**
   - Formula: `valid_comparison = same_methodology AND same_scope AND same_timeframe`
   - Statistical significance: p < 0.05

4. **Carbon Neutral Claim Assessment**
   - Formula: `net_zero = total_emissions - verified_offsets <= 0`
   - Offset quality: Gold Standard, VCS, or equivalent

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| Product LCA factors | EcoInvent, ELCD | Annual |
| PEF factors | EU JRC | Per update |
| Carbon offset registry | VCS, Gold Standard | Real-time |

### Tools Required (4)

1. **claim_analyzer** - NLP analysis of environmental claims
2. **pef_calculator** - Product Environmental Footprint calculator
3. **evidence_validator** - Supporting documentation checker
4. **offset_verifier** - Carbon offset quality validator

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| Claim Substantiation Report | PDF | GCD format | Legal/Marketing |
| PEF Study Summary | PDF | EU PEF | Auditors |
| Compliance Certificate | PDF | Internal | Regulators |
| Risk Assessment | JSON | API | Risk Managers |

### Implementation Tasks

- [ ] **Week 1:** Build claim text analyzer (NLP for green claims)
- [ ] **Week 1:** Implement PEF methodology calculator
- [ ] **Week 2:** Build evidence quality scorer
- [ ] **Week 2:** Integrate carbon offset registries (VCS, Gold Standard)
- [ ] **Week 3:** Create comparative claim validator
- [ ] **Week 3:** Build substantiation report generator
- [ ] **Week 4:** Create 200 golden tests (claim types, evidence levels)
- [ ] **Week 4:** Certification and deployment

---

## Agent 6: CSDDD Supply Chain Due Diligence Agent [P3]

### Regulation Details
- **Regulation:** Corporate Sustainability Due Diligence Directive (EU) 2024/1760
- **Deadline:** July 26, 2027 (large companies), July 26, 2028 (smaller in-scope)
- **Penalty:** Up to 5% annual worldwide turnover
- **Scope:** Large EU companies + non-EU with significant EU turnover

### Agent Specification

```yaml
agent_id: gl-csddd-v1
name: CSDDD Supply Chain Due Diligence Agent
version: "1.0.0"
type: supply-chain-due-diligence
priority: P3-MEDIUM
deadline: "2027-07-26"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `company_profile` | object | Registration | Yes |
| `supplier_list` | array | Procurement | Yes |
| `supplier_assessments` | array | Due diligence | Yes |
| `tier_mapping` | object | Supply chain | Yes |
| `risk_indicators` | array | Monitoring | Yes |
| `grievance_data` | array | Stakeholder | Yes |
| `remediation_plans` | array | Corrective actions | Yes |

### Calculations Needed

1. **Adverse Impact Identification**
   - Human rights impacts: 18 categories (ILO conventions)
   - Environmental impacts: 6 categories (Paris Agreement aligned)
   - Formula: `impact_score = severity * likelihood * reversibility`

2. **Supplier Risk Scoring**
   - Formula: `risk = country_risk * sector_risk * supplier_history`
   - Categories: High, Medium, Low

3. **Value Chain Coverage**
   - Formula: `coverage = assessed_suppliers / total_suppliers * 100`
   - Target: 100% Tier 1, prioritized Tier 2+

4. **Remediation Effectiveness**
   - Formula: `effectiveness = remediated_impacts / identified_impacts * 100`
   - Target: 90% within reporting period

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| Country human rights risk | BHRRC | Quarterly |
| Sector risk indicators | SASB, GRI | Annual |
| Environmental risk | WRI, WWF | Annual |

### Tools Required (5)

1. **supplier_risk_assessor** - Multi-factor supplier risk scoring
2. **value_chain_mapper** - Tier 1-N supply chain visualization
3. **adverse_impact_identifier** - Human rights/environmental impact detection
4. **grievance_tracker** - Stakeholder complaint management
5. **remediation_monitor** - Corrective action tracking

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| Due Diligence Report | PDF | CSDDD format | Annual Report |
| Supplier Risk Dashboard | Web | API | Procurement |
| Grievance Register | Excel | Internal | Legal |
| Remediation Tracker | Web | API | Operations |

### Implementation Tasks

- [ ] **Week 1:** Build supplier risk assessment engine
- [ ] **Week 1:** Implement value chain mapping (Tier 1-N)
- [ ] **Week 2:** Build adverse impact identifier
- [ ] **Week 2:** Create grievance tracking system
- [ ] **Week 3:** Implement remediation workflow
- [ ] **Week 3:** Build stakeholder engagement tools
- [ ] **Week 4:** Create 250 golden tests (risk, impacts, remediation)
- [ ] **Week 4:** Certification and deployment

---

## Agent 7: Product Carbon Footprint / Digital Product Passport Agent [P3]

### Regulation Details
- **Regulation:** Ecodesign for Sustainable Products Regulation (ESPR), Battery Regulation
- **Deadline:** February 2027 (Batteries), rolling (other products)
- **Penalty:** Market access denial, fines per member state
- **Scope:** Products sold in EU market (starting with batteries)

### Agent Specification

```yaml
agent_id: gl-pcf-dpp-v1
name: Product Carbon Footprint & Digital Passport Agent
version: "1.0.0"
type: product-lifecycle
priority: P3-MEDIUM
deadline: "2027-02-01"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `product_id` | string | System | Yes |
| `bill_of_materials` | array | Engineering | Yes |
| `manufacturing_data` | object | Production | Yes |
| `logistics_data` | object | Supply chain | Yes |
| `use_phase_data` | object | Product specs | Yes |
| `end_of_life_data` | object | Waste management | Yes |
| `battery_chemistry` | object | Engineering | Batteries |
| `recycled_content` | object | Materials | Yes |

### Calculations Needed

1. **Product Carbon Footprint (cradle-to-gate)**
   - Formula: `PCF = raw_materials + manufacturing + transport`
   - Methodology: ISO 14067, PEF

2. **Full Lifecycle Assessment (cradle-to-grave)**
   - Formula: `LCA = PCF + use_phase + end_of_life`
   - 16 PEF impact categories

3. **Battery Carbon Footprint** (specific for Battery Regulation)
   - Production: `battery_pcf = cell_production + module_assembly + pack_integration`
   - Performance class: A, B, C, D, E

4. **Recycled Content Calculation**
   - Formula: `recycled_pct = recycled_material / total_material * 100`
   - Minimum thresholds per product category

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| Material emission factors | EcoInvent | Annual |
| Energy factors by country | IEA | Annual |
| Transport factors | GLEC | Annual |
| Battery materials | Argonne GREET | Annual |

### Tools Required (5)

1. **lca_calculator** - Full lifecycle assessment engine
2. **bom_analyzer** - Bill of materials carbon analyzer
3. **dpp_generator** - Digital Product Passport creator
4. **battery_pcf_calculator** - Battery Regulation compliant PCF
5. **qr_code_generator** - DPP access QR code generator

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| Product Carbon Footprint | PDF | ISO 14067 | Marketing |
| Digital Product Passport | JSON-LD | EU DPP | EU Registry |
| Battery Passport | QR/JSON | Battery Reg | Consumers |
| LCA Report | PDF/Excel | ISO 14040 | Internal |

### Implementation Tasks

- [ ] **Week 1:** Build BOM carbon footprint calculator
- [ ] **Week 1:** Implement ISO 14067 PCF methodology
- [ ] **Week 2:** Build battery-specific PCF calculator
- [ ] **Week 2:** Create Digital Product Passport generator
- [ ] **Week 3:** Implement recycled content tracker
- [ ] **Week 3:** Build QR code and registry integration
- [ ] **Week 4:** Create 300 golden tests (products, batteries, LCA)
- [ ] **Week 4:** Certification and deployment

---

## Agent 8: Scope 3 Supply Chain Emissions Agent [P4]

### Regulation Details
- **Framework:** GHG Protocol Corporate Value Chain (Scope 3) Standard
- **Deadline:** Ongoing (required by CSRD, SB 253, ISSB)
- **Drivers:** CSRD, SB 253, CDP, ISSB IFRS S2
- **Scope:** All 15 Scope 3 categories

### Agent Specification

```yaml
agent_id: gl-scope3-v1
name: Scope 3 Supply Chain Emissions Agent
version: "1.0.0"
type: value-chain-emissions
priority: P4-STANDARD
deadline: "ongoing"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `procurement_data` | array | ERP | Yes |
| `supplier_emissions` | array | Supplier data | Preferred |
| `logistics_data` | array | TMS | Yes |
| `employee_data` | object | HR | Cat 6, 7 |
| `product_use_data` | object | Engineering | Cat 11 |
| `waste_data` | array | Operations | Cat 5 |
| `investment_data` | array | Finance | Cat 15 |

### Calculations Needed

All 15 GHG Protocol Scope 3 categories:

| Category | Calculation Method |
|----------|-------------------|
| 1. Purchased goods | Spend-based or supplier-specific |
| 2. Capital goods | Spend-based |
| 3. Fuel/energy activities | WTT factors |
| 4. Upstream transport | Distance-based or spend-based |
| 5. Waste generated | Waste-type specific |
| 6. Business travel | Distance-based |
| 7. Employee commuting | Distance-based or surveys |
| 8. Upstream leased assets | Asset-based |
| 9. Downstream transport | Distance-based |
| 10. Processing of sold products | Process-specific |
| 11. Use of sold products | Product lifecycle |
| 12. End-of-life treatment | Waste-type specific |
| 13. Downstream leased assets | Asset-based |
| 14. Franchises | Franchise-based |
| 15. Investments | Financed emissions |

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| Spend-based factors | EPA EEIO, Exiobase | Annual |
| Supplier-specific | CDP, SBTi | Ongoing |
| Transport factors | GLEC Framework | Annual |
| Travel factors | DEFRA, EPA | Annual |
| Waste factors | IPCC, EPA | Annual |

### Tools Required (5)

1. **spend_analyzer** - Spend-based emissions calculator
2. **supplier_data_collector** - Primary data collection from suppliers
3. **transport_calculator** - GLEC-compliant transport emissions
4. **data_quality_scorer** - GHG Protocol data quality indicators
5. **scope3_aggregator** - All-category consolidator

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| Scope 3 Inventory | Excel | GHG Protocol | Internal |
| Category Breakdown | PDF | Custom | Stakeholders |
| CDP Response | XML | CDP | CDP Platform |
| SBTi Submission | Excel | SBTi | SBTi Platform |

### Implementation Tasks

- [ ] **Week 1:** Build spend-based emissions calculator
- [ ] **Week 1:** Implement supplier data collection workflow
- [ ] **Week 2:** Build transport emissions calculator (GLEC)
- [ ] **Week 2:** Implement categories 6, 7 (travel, commuting)
- [ ] **Week 3:** Build product use-phase calculator (Cat 11)
- [ ] **Week 3:** Implement financed emissions (Cat 15)
- [ ] **Week 4:** Create 400 golden tests (all 15 categories)
- [ ] **Week 4:** Certification and deployment

---

## Agent 9: Science-Based Targets (SBTi) Validation Agent [P4]

### Regulation Details
- **Framework:** Science Based Targets initiative (SBTi) criteria
- **Deadline:** Ongoing (validation cycles)
- **Drivers:** Investor pressure, CSRD, Net-Zero commitments
- **Scope:** Companies with SBTi commitments or targets

### Agent Specification

```yaml
agent_id: gl-sbti-validation-v1
name: SBTi Target Validation Agent
version: "1.0.0"
type: target-validation
priority: P4-STANDARD
deadline: "ongoing"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `base_year_emissions` | object | GHG Inventory | Yes |
| `target_year` | int | Commitment | Yes |
| `target_type` | enum | SBTi choice | Yes |
| `sector` | string | Classification | Yes |
| `scope_coverage` | object | GHG Inventory | Yes |
| `reduction_pathway` | array | Planning | Yes |
| `current_emissions` | object | GHG Inventory | Yes |

### Calculations Needed

1. **1.5C Alignment Check**
   - Absolute contraction: 4.2% annual reduction
   - Sectoral decarbonization: sector-specific pathway

2. **Target Ambition Assessment**
   - Formula: `ambition = (base_year - target_year_emissions) / years`
   - Minimum: 1.5C or well-below 2C alignment

3. **Scope Coverage Validation**
   - Scope 1 + 2: 95% coverage required
   - Scope 3: 67% coverage if >40% of total emissions

4. **Progress Tracking**
   - Formula: `progress = (base_emissions - current_emissions) / (base_emissions - target_emissions) * 100`
   - Annual tracking required

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| Sector pathways | SBTi | Per update |
| 1.5C carbon budget | IPCC | Per AR cycle |
| Grid decarbonization | IEA | Annual |

### Tools Required (4)

1. **pathway_calculator** - 1.5C/2C pathway generator
2. **target_validator** - SBTi criteria compliance checker
3. **progress_tracker** - Annual progress monitoring
4. **scenario_modeler** - Target achievement scenarios

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| SBTi Submission Package | Excel | SBTi format | SBTi |
| Target Validation Report | PDF | Internal | Board |
| Progress Dashboard | Web | API | Sustainability Team |
| Investor Disclosure | PDF | TCFD | Investors |

### Implementation Tasks

- [ ] **Week 1:** Build 1.5C pathway calculator
- [ ] **Week 1:** Implement sector-specific pathways
- [ ] **Week 2:** Build target validation engine
- [ ] **Week 2:** Create progress tracking dashboard
- [ ] **Week 3:** Implement scenario modeling
- [ ] **Week 3:** Build SBTi submission package generator
- [ ] **Week 4:** Create 200 golden tests (pathways, targets, progress)
- [ ] **Week 4:** Certification and deployment

---

## Agent 10: Carbon Offset Verification Agent [P4]

### Regulation Details
- **Standards:** Verified Carbon Standard (VCS), Gold Standard, ACR, CAR
- **Deadline:** Ongoing (offset quality requirements increasing)
- **Drivers:** Net-Zero claims, Green Claims Directive, SBTi
- **Scope:** Organizations using carbon offsets

### Agent Specification

```yaml
agent_id: gl-carbon-offset-v1
name: Carbon Offset Verification Agent
version: "1.0.0"
type: offset-verification
priority: P4-STANDARD
deadline: "ongoing"
```

### Input Data Required

| Input Field | Type | Source | Required |
|-------------|------|--------|----------|
| `offset_portfolio` | array | Registry | Yes |
| `project_ids` | array | Registry | Yes |
| `vintage_years` | array | Registry | Yes |
| `project_types` | array | Registry | Yes |
| `verification_reports` | array | VVB | Yes |
| `retirement_records` | array | Registry | Yes |
| `additionality_docs` | array | Project | Yes |

### Calculations Needed

1. **Offset Quality Score**
   - Formula: `quality = additionality * permanence * verification * co_benefits`
   - Scale: 0-100

2. **Additionality Assessment**
   - Investment test, barrier analysis, common practice test
   - Binary: Pass/Fail

3. **Permanence Risk**
   - Buffer pool contribution (forest projects)
   - Risk rating: Low, Medium, High

4. **Double Counting Check**
   - Corresponding adjustments under Paris Agreement
   - Registry cross-check

### Emission Factors Needed

| Factor | Source | Update Frequency |
|--------|--------|------------------|
| Project baselines | Registry | Per project |
| Leakage factors | Methodology | Per project |
| Buffer pool rates | VCS | Annual |

### Tools Required (4)

1. **registry_connector** - VCS, Gold Standard, ACR, CAR API integration
2. **quality_assessor** - Multi-criteria offset quality scoring
3. **additionality_checker** - Additionality test automation
4. **retirement_tracker** - Offset retirement and claims matching

### Output Reports/Formats

| Output | Format | Schema | Recipient |
|--------|--------|--------|-----------|
| Offset Quality Report | PDF | Internal | Sustainability Team |
| Portfolio Dashboard | Web | API | Finance |
| Retirement Certificate | PDF | Registry | Auditors |
| Claims Substantiation | PDF | GCD | Marketing |

### Implementation Tasks

- [ ] **Week 1:** Build registry API connectors (VCS, Gold Standard)
- [ ] **Week 1:** Implement offset quality scoring
- [ ] **Week 2:** Build additionality assessment engine
- [ ] **Week 2:** Create double counting checker
- [ ] **Week 3:** Implement retirement tracking
- [ ] **Week 3:** Build claims substantiation generator
- [ ] **Week 4:** Create 150 golden tests (registries, quality, claims)
- [ ] **Week 4:** Certification and deployment

---

## Implementation Prioritization Matrix

### Priority Levels

| Priority | Criteria | Agents | Timeline |
|----------|----------|--------|----------|
| **P0-CRITICAL** | Deadline <60 days | EUDR | Week 1-4 |
| **P1-HIGH** | Deadline <12 months OR already in force | SB 253, CSRD | Week 5-12 |
| **P2-MEDIUM** | Deadline 12-18 months | EU Taxonomy, Green Claims | Week 13-20 |
| **P3-MEDIUM** | Deadline 18-24 months | CSDDD, PCF/DPP | Week 21-28 |
| **P4-STANDARD** | Ongoing/Framework-based | Scope 3, SBTi, Offsets | Week 29-36 |

### Resource Allocation

| Phase | Weeks | Agents | Engineers | Climate Scientists |
|-------|-------|--------|-----------|-------------------|
| Phase 1 | 1-4 | EUDR | 4 | 2 |
| Phase 2 | 5-12 | SB 253, CSRD | 6 | 2 |
| Phase 3 | 13-20 | EU Taxonomy, Green Claims | 4 | 1 |
| Phase 4 | 21-28 | CSDDD, PCF/DPP | 4 | 1 |
| Phase 5 | 29-36 | Scope 3, SBTi, Offsets | 4 | 1 |

---

## Shared Components (Build Once, Reuse)

### Emission Factor Database

All agents share a common emission factor database:

```yaml
emission_factor_database:
  sources:
    - IEA: Grid factors, energy statistics
    - IPCC: GHG methodology defaults
    - EPA: US-specific factors
    - DEFRA: UK-specific factors
    - EcoInvent: LCA database
    - Exiobase: MRIO factors
    - GLEC: Transport factors

  update_frequency: Quarterly

  coverage:
    - Stationary combustion: 50+ fuels
    - Mobile combustion: 30+ vehicle types
    - Grid electricity: 200+ countries
    - Materials: 1,000+ materials
    - Transport: 20+ modes
```

### Validation Framework

Common validation hooks across agents:

```python
# Shared validation hooks
validation_hooks = [
    "emissions_arithmetic_check",      # Sum validation
    "emission_factor_provenance",      # Source verification
    "data_quality_scorer",            # GHG Protocol DQI
    "regulatory_schema_validator",    # EU schema compliance
    "calculation_methodology_check",  # Method alignment
]
```

### Golden Test Infrastructure

```yaml
golden_tests:
  total_target: 2,500 tests
  distribution:
    - EUDR: 200 tests
    - SB 253: 300 tests
    - CSRD: 500 tests
    - EU Taxonomy: 300 tests
    - Green Claims: 200 tests
    - CSDDD: 250 tests
    - PCF/DPP: 300 tests
    - Scope 3: 400 tests
    - SBTi: 200 tests
    - Offsets: 150 tests
```

---

## Success Metrics

### Per-Agent Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Golden test pass rate | >98% | Automated |
| Regulatory compliance | 100% | Expert review |
| Calculation accuracy | >99% | Against known values |
| Generation time | <2 hours | Factory pipeline |
| Certification time | <5 days | Full cycle |

### Program Metrics

| Metric | Phase 1 | Phase 2 | Phase 3 | Final |
|--------|---------|---------|---------|-------|
| Agents deployed | 1 | 4 | 7 | 10 |
| Golden tests | 200 | 1,000 | 2,000 | 2,500 |
| Regulations covered | 1 | 4 | 6 | 10 |
| Customer pilots | 5 | 20 | 50 | 100 |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| EUDR deadline missed | Medium | Critical | Fast-track development, reduce scope |
| Regulation changes | High | Medium | Modular design, monitoring |
| Data quality issues | High | Medium | Data quality scoring, estimation fallbacks |
| Integration complexity | Medium | Medium | API-first design, sandbox testing |
| Expert reviewer bottleneck | Low | Medium | Train additional reviewers |

---

## Next Steps

1. **Week 1:** Kickoff EUDR agent development (P0-CRITICAL)
2. **Week 2:** Finalize shared emission factor database schema
3. **Week 3:** Begin SB 253 and CSRD agent specifications
4. **Week 4:** EUDR agent certification and deployment
5. **Week 5+:** Continue with P1-P4 agents per schedule

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-03 | GL-RegulatoryIntelligence | Initial regulatory agents master plan |

**Approvals:**

- Climate Science Lead: ___________________ Date: _______
- Engineering Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______
- Program Director: ___________________ Date: _______

---

**END OF DOCUMENT**
