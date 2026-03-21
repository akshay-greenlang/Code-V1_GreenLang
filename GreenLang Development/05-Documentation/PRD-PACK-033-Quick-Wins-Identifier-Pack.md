# PRD-PACK-033: Quick Wins Identifier Pack

**Pack ID:** PACK-033-quick-wins-identifier
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-21
**Prerequisite:** None (standalone; enhanced with PACK-031 Industrial Energy Audit Pack and PACK-032 Building Energy Assessment Pack if present; complemented by PACK-021/022/023 Net Zero Packs)

---

## 1. Executive Summary

### 1.1 Problem Statement

Across commercial buildings and industrial facilities, 15-30% of energy consumption can be eliminated through low-cost, fast-payback measures -- commonly termed "quick wins." The EU Energy Efficiency Directive (EED) 2023/1791 mandates cost-effective energy savings identification, yet most organizations struggle to systematically discover, quantify, and prioritize these opportunities. Despite abundant technical guidance from ISO 50001:2018, ASHRAE, and national energy agencies, organizations face persistent challenges:

1. **Scattered opportunity identification**: Quick win opportunities span lighting, HVAC scheduling, compressed air leaks, power factor correction, plug load management, building envelope sealing, and dozens of other categories. Facilities managers typically identify only 20-30% of available quick wins through informal walkthroughs because no systematic scanning methodology covers all 15+ quick win categories simultaneously. Opportunities in behavioral change, utility rebates, and controls optimization are almost always overlooked.

2. **Inaccurate savings estimation**: Estimating energy savings from quick win measures requires ASHRAE 14-2014 compliant methodology including interactive effects (e.g., LED lighting reduces both lighting energy and internal heat gains, altering HVAC loads), HDD/CDD normalization, rebound effects (occupants may increase comfort after efficiency improvements), and uncertainty quantification. Most facilities use manufacturer claims or rule-of-thumb percentages that systematically overestimate savings by 20-40%, leading to credibility erosion with management and finance teams.

3. **Simplistic financial analysis**: Quick win business cases typically present only simple payback periods. Decision-makers need discounted payback, NPV, IRR, ROI, and Levelized Cost of Energy (LCOE) to compare energy investments against alternative capital deployment. Furthermore, utility rate escalation (3-5% annual), tax incentives (Section 179D in the US, Enhanced Capital Allowances in the UK, EU Taxonomy-aligned green investment benefits), and maintenance cost changes must be factored into the analysis. Without rigorous financial modeling, sound quick wins are rejected while marginal projects with better marketing are approved.

4. **Missing carbon quantification**: Organizations increasingly need to link energy savings to tCO2e reduction for Scope 1, 2, and 3 GHG reporting per the GHG Protocol, SBTi target tracking, CSRD/ESRS E1 disclosure, and CDP Climate Change reporting. Quick win carbon reduction depends on fuel type, grid emission factors (location-based vs. market-based per GHG Protocol Scope 2 Guidance), and upstream energy emissions (Scope 3 Category 3). Most quick win assessments either ignore carbon entirely or apply a single average emission factor that fails to differentiate between fuel switching and electricity reduction.

5. **Ad hoc prioritization**: With 20-80 potential quick wins identified across a facility, organizations lack structured methods to prioritize implementation. Multi-criteria decision analysis (MCDA) considering energy savings, carbon reduction, capital cost, payback, implementation disruption, maintenance impact, occupant comfort, and regulatory alignment is rarely performed. Without Pareto frontier analysis and dependency graph modeling, organizations pursue measures in suboptimal sequence, missing synergies and creating unnecessary disruption.

6. **Behavioral change neglect**: 10-20% of building energy consumption can be reduced through behavioral changes alone (IEA estimates 5-20% for commercial buildings). However, behavioral energy programs suffer from low persistence (savings decay 30-50% within 12 months without reinforcement), inconsistent adoption across building populations, and lack of measurement. Without Rogers diffusion modeling, persistence factor tracking, and gamification frameworks, behavioral programs produce initial enthusiasm followed by rapid regression to baseline.

7. **Utility rebate blindspots**: Utility companies and government programs offer prescriptive and custom rebates that can fund 20-60% of quick win implementation costs. A typical commercial building is eligible for USD 5,000-50,000 in annual rebates across lighting, HVAC, controls, and motor efficiency programs. However, rebate programs differ by utility territory, change annually, have filing deadlines, and require specific documentation. Most facilities capture less than 10% of available rebates because no automated system matches measures to programs.

8. **Verification gaps**: After implementing quick wins, IPMVP-compliant Measurement and Verification (M&V) is rarely performed. Without Option A (key parameter measurement) or Option B (all parameter measurement) verification per IPMVP Core Concepts 2022, organizations cannot confirm actual savings, report verified reductions to management, or satisfy ISO 50001 continual improvement evidence requirements. This undermines the credibility of future quick win programs and makes energy performance contracting impossible.

### 1.2 Solution Overview

PACK-033 is the **Quick Wins Identifier Pack** -- the third pack in the "Energy Efficiency Packs" category, complementing PACK-031 (Industrial Energy Audit) and PACK-032 (Building Energy Assessment). While PACK-031 and PACK-032 provide deep, comprehensive energy audits, PACK-033 focuses specifically on fast-payback, low-capital measures that deliver measurable energy and carbon savings within 0-24 months.

The pack provides automated facility scanning across 15+ quick win categories with 80+ pre-defined actions, rigorous savings estimation with ASHRAE 14-2014 compliant uncertainty bands, full financial analysis (simple payback, discounted payback, NPV, IRR, ROI, LCOE), Scope 1/2/3 carbon reduction quantification per the GHG Protocol, multi-criteria prioritization with Pareto frontier optimization, behavioral change modeling with Rogers diffusion curves, utility rebate matching against 100+ utility programs, and IPMVP Option A/B verification for implemented measures.

The pack includes 8 engines, 6 workflows, 8 templates, 11 integrations, and 8 presets covering the complete quick wins lifecycle from identification through verification.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-033 Quick Wins Identifier Pack |
|-----------|------------------------------|--------------------------------------|
| Opportunity identification | Informal walkthrough (20-30% coverage) | Systematic scan across 15 categories, 80+ actions (90%+ coverage) |
| Time to identify quick wins | 2-4 weeks per facility | <4 hours per facility (10-20x faster) |
| Assessment cost | EUR 10,000-30,000 per facility | EUR 1,000-3,000 per facility (10x reduction) |
| Savings estimation accuracy | Rule-of-thumb (+/- 40%) | ASHRAE 14-2014 with uncertainty bands (+/- 10-15%) |
| Financial analysis | Simple payback only | NPV, IRR, ROI, LCOE, discounted payback, tax incentives |
| Carbon quantification | Single average emission factor | Scope 1/2/3 per GHG Protocol, location/market-based, SBTi aligned |
| Prioritization method | Subjective ranking | MCDA with Pareto frontier, dependency graphs, sequencing optimization |
| Behavioral change | Ad hoc awareness campaigns | 40+ actions, Rogers diffusion modeling, persistence tracking, gamification |
| Utility rebates | Manual lookup (capture <10%) | Automated matching against 100+ utility programs (capture 50-80%) |
| M&V verification | Rarely performed | IPMVP Option A/B integrated for every measure |
| Audit trail | Spreadsheet-based | SHA-256 provenance, full calculation lineage, digital audit trail |

### 1.4 Quick Win Definition

A "quick win" energy conservation measure meets ALL of the following criteria:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Simple payback | <= 24 months | Fast return on investment; minimal capital approval hurdles |
| Capital cost | <= EUR 50,000 per measure (configurable) | Within operational or minor capital expenditure authority |
| Implementation time | <= 6 months from approval to operational | Delivers savings within current fiscal year |
| Technical complexity | Low to medium | No major engineering design, structural modification, or permit required |
| Disruption | Minimal to moderate | Does not require facility shutdown or major process interruption |

### 1.5 Quick Win Categories

| # | Category | Typical Measures | Typical Savings (% of category energy) |
|---|----------|-----------------|----------------------------------------|
| 1 | Lighting | LED retrofit, occupancy sensors, daylight harvesting, de-lamping, task lighting | 30-60% |
| 2 | HVAC Scheduling | Optimized start/stop, night setback, weekend shutdown, holiday scheduling | 10-25% |
| 3 | HVAC Setpoints | Deadband widening, seasonal reset, zone optimization, unoccupied setback | 5-15% |
| 4 | Compressed Air | Leak repair, pressure reduction, VSD trim compressor, receiver sizing | 15-30% |
| 5 | Building Envelope | Weather-stripping, door closers, dock seals, caulking, pipe insulation | 5-15% |
| 6 | Plug Loads | Smart power strips, vending machine controllers, equipment scheduling | 10-30% |
| 7 | Motors & Drives | VSD retrofit on variable-load motors, right-sizing, IE class upgrade | 15-40% |
| 8 | Boiler/Heating | Combustion tuning, O2 trim, condensate return, insulation, blowdown | 5-15% |
| 9 | Water Heating | Temperature reduction, timer controls, low-flow fixtures, pipe insulation | 10-25% |
| 10 | Refrigeration | Condenser cleaning, door gaskets, strip curtains, defrost optimization | 10-20% |
| 11 | Power Quality | Power factor correction, harmonic filtering, demand management | 5-15% (demand charges) |
| 12 | Controls & BMS | Scheduling optimization, reset strategies, economizer enable, trending | 10-20% |
| 13 | Behavioral Change | Awareness campaigns, energy champions, competitions, dashboards | 5-15% |
| 14 | Steam/Process Heat | Trap repair, insulation, condensate return, flash steam recovery | 10-25% |
| 15 | Renewable Quick Wins | Solar PV (where incentives give <24-month payback), solar thermal preheat | 10-30% (of applicable load) |

### 1.6 Target Users

**Primary:**
- Facilities managers seeking fast-payback energy savings at commercial buildings and light industrial facilities
- Energy managers building a pipeline of quick win measures to demonstrate continuous improvement under ISO 50001
- Property portfolio managers wanting a rapid scan across multiple sites
- SME owners and operators without dedicated energy staff

**Secondary:**
- Energy consultants performing rapid energy assessments for multiple clients
- Corporate sustainability teams rolling out quick win programs across a building portfolio
- ESCOs (Energy Service Companies) identifying quick-start projects to build client relationships
- Utility program administrators matching customers to rebate programs
- Operations managers seeking to reduce facility operating costs
- CFOs evaluating low-risk energy investments with fast payback

### 1.7 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to complete quick wins scan (single facility) | <4 hours (vs. 2-4 weeks manual) | Time from data intake to prioritized quick wins list |
| Quick win identification coverage | >90% of available opportunities | Validated against detailed Type 2 energy audit findings |
| Savings estimation accuracy | Within 15% of actual post-implementation savings | Verified against IPMVP M&V results at 12 months |
| Financial calculation accuracy | 100% match with financial calculator reference values | Cross-validated against NPV/IRR/LCOE test cases |
| Carbon reduction accuracy | Within 5% of verified GHG inventory values | Cross-validated against MRV agent calculations |
| Utility rebate matching | >80% of available rebates identified | Validated against manual utility program research |
| Behavioral change persistence | >70% savings retention at 12 months | Measured via ongoing monitoring after behavioral intervention |
| Customer NPS | >55 | Net Promoter Score survey |
| Average payback of identified measures | <18 months | Portfolio average across all identified quick wins |
| Implementation rate of identified quick wins | >60% within 12 months | Tracking percentage of identified measures actually implemented |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| EU Energy Efficiency Directive (EED) | Directive 2023/1791 (recast) | Article 11 energy audit obligation includes quick win identification; Article 8 energy efficiency obligation schemes require quantified savings |
| ISO 50001:2018 | Energy management systems -- Requirements with guidance for use | Clause 6.2 objectives and targets (quick wins as short-term targets); Clause 10.2 continual improvement (quick wins demonstrate ongoing savings) |
| ASHRAE Guideline 14-2014 | Measurement of Energy, Demand, and Water Savings | Statistical requirements for savings estimation: CV(RMSE), NMBE, R-squared; uncertainty quantification methodology |
| IPMVP Core Concepts 2022 | International Performance Measurement and Verification Protocol | M&V Options A (key parameter) and B (all parameters) for quick win savings verification |
| GHG Protocol Corporate Standard | WRI/WBCSD (2015) | Scope 1, 2, 3 emissions quantification from energy consumption |
| GHG Protocol Scope 2 Guidance | WRI/WBCSD (2015) | Location-based and market-based emission factors for electricity savings |

### 2.2 Supporting Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| ISO 50006:2014 | Energy baseline and energy performance indicators | EnPI methodology for quick win savings tracking |
| ISO 50015:2014 | Measurement and verification of energy performance | M&V framework aligned with IPMVP |
| EN 16247-1:2022 | Energy audits -- Part 1: General requirements | Quick win identification as a component of energy audit process |
| EN 15193-1:2017 | Energy requirements for lighting | LENI calculation for lighting quick wins |
| EN 14825:2022 | Heat pump seasonal performance | SCOP/SEER for HVAC setpoint optimization impact |
| EN 15232-1:2017 | Impact of building automation and controls | BACS efficiency factors for controls quick wins |
| IEC 60034-30-1 | Motor efficiency classes IE1-IE5 | Motor upgrade quick win savings quantification |

### 2.3 Financial and Carbon Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| SBTi Corporate Framework | SBTi (2024) | Quick win carbon reductions contributing to science-based targets |
| ESRS E1 Climate Change | EU CSRD (2023) | E1-5 energy consumption and mix disclosure; quick wins reduce reported energy intensity |
| CDP Climate Change | CDP (2024) | C4 Targets and performance, C7 Energy breakdown |
| EU Taxonomy Regulation | Regulation 2020/852 | Climate mitigation substantial contribution criteria for energy efficiency investments |
| TCFD Recommendations | FSB/TCFD (2017) | Metrics and targets for energy efficiency |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 8 | Quick wins identification, estimation, and verification engines |
| Workflows | 6 | Multi-phase orchestration workflows |
| Templates | 8 | Report, dashboard, and analysis templates |
| Integrations | 11 | Agent, app, data, and system bridges |
| Presets | 8 | Building/facility-type-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `quick_wins_scanner_engine.py` | Automated facility and process scanning for quick win opportunities across 15 categories. Contains 80+ pre-defined quick win actions with applicability rules, typical savings ranges, and implementation requirements. Uses building type profiles (office, manufacturing, retail, warehouse, healthcare, education, data center, SME) to filter and weight applicable measures. Performs automated gap analysis against facility operational data (schedules, setpoints, equipment inventory, metering) to identify specific quick wins with estimated savings potential. Includes occupancy-driven analysis, after-hours energy profiling, and equipment runtime anomaly detection. |
| 2 | `payback_calculator_engine.py` | Comprehensive financial analysis engine for quick win measures: simple payback (CapEx / annual savings), discounted payback (time to positive cumulative NPV), Net Present Value (NPV) at configurable discount rate, Internal Rate of Return (IRR) via Newton-Raphson iteration, Return on Investment (ROI), and Levelized Cost of Energy (LCOE) per kWh saved. Incorporates utility rate escalation (configurable, default 3.5% annual), tax incentives and depreciation benefits, maintenance cost differentials (before/after implementation), equipment useful life, and rebate/incentive offsets. All arithmetic uses Python Decimal for financial precision (no floating-point rounding). |
| 3 | `energy_savings_estimator_engine.py` | Rigorous energy savings estimation in kWh, therms, and GJ with uncertainty quantification per ASHRAE Guideline 14-2014. Calculates savings for each measure using deterministic engineering formulas with lookups for equipment efficiency, load profiles, and operating hours. Applies interactive effects modeling (e.g., lighting savings reduce cooling load but increase heating load), HDD/CDD normalization for weather-dependent measures, rebound effect corrections (typically 5-15% for comfort-related measures), degradation factors for equipment aging, and learning curves for behavioral measures. Outputs include point estimate, 90% confidence interval (upper/lower bounds), and sensitivity analysis on key input parameters. |
| 4 | `carbon_reduction_engine.py` | Calculates tCO2e emission reductions for each quick win measure using grid and fuel emission factors per the GHG Protocol. Separates Scope 1 (direct fuel combustion savings), Scope 2 (purchased electricity, heat, cooling savings with both location-based and market-based factors per GHG Protocol Scope 2 Guidance), and Scope 3 (upstream fuel and energy activities per Category 3). Uses national/regional grid emission factors (DEFRA, UBA, ADEME, ISPRA, EPA eGRID), fuel-specific emission factors from IPCC Guidelines, and residual mix factors for market-based accounting. Includes SBTi alignment assessment showing quick win carbon reduction contribution toward science-based targets. |
| 5 | `implementation_prioritizer_engine.py` | Multi-Criteria Decision Analysis (MCDA) engine for prioritizing quick win measures. Supports weighted scoring across configurable criteria: energy savings (kWh/yr), carbon reduction (tCO2e/yr), capital cost (EUR), payback (years), implementation disruption (1-5 scale), maintenance impact (positive/negative/neutral), occupant comfort impact (1-5 scale), regulatory alignment score, and rebate availability. Computes Pareto frontier to identify non-dominated solutions. Builds dependency graphs for measures with prerequisites or synergies (e.g., BMS upgrade enables scheduling optimization). Runs sequencing optimization to determine optimal implementation order considering cash flow, resource constraints, and seasonal windows. |
| 6 | `behavioral_change_engine.py` | Models and tracks behavioral energy conservation actions across 40+ defined behavioral measures (thermostat awareness, equipment shutdown, natural ventilation, stair vs. elevator, print reduction, etc.). Implements Rogers diffusion of innovation modeling (innovators 2.5%, early adopters 13.5%, early majority 34%, late majority 34%, laggards 16%) to forecast adoption curves. Applies persistence factors based on intervention type (information-only: 30-50% persistence at 12 months; feedback with incentives: 60-80%; structural/default changes: 85-95%). Includes gamification framework for energy competitions, leaderboards, and badge systems. Tracks per-action adoption rates and savings realization against modeled projections. |
| 7 | `utility_rebate_engine.py` | Matches identified quick win measures to available utility incentive and rebate programs. Maintains a database of 100+ utility programs across major service territories with prescriptive rebates (fixed dollar per unit: e.g., USD 2-5/lamp for LED, USD 50-200/ton for high-efficiency HVAC) and custom/calculated rebates (based on verified kWh savings, typically USD 0.05-0.15/kWh). Tracks program eligibility requirements, filing deadlines, documentation requirements, pre-approval needs, and stacking rules (whether multiple rebates can be combined). Calculates total rebate value per measure and net cost after incentives. Generates rebate application packages with required documentation. |
| 8 | `quick_wins_reporting_engine.py` | Aggregates all quick win analysis results into configurable dashboard and report formats. Provides portfolio-level summary statistics (total savings potential kWh/yr, EUR/yr, tCO2e/yr; total investment; portfolio payback; portfolio IRR). Tracks implementation progress (planned, in-progress, completed, verified) with milestone tracking. Integrates IPMVP Option A/B verification results comparing actual vs. projected savings. Generates executive summaries, detailed measure reports, implementation schedules, and progress dashboards. Supports export in MD, HTML, PDF, JSON, and CSV formats. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `facility_scan_workflow.py` | 4: FacilityRegistration -> DataCollection -> QuickWinScanning -> SavingsEstimation | End-to-end facility scanning from setup through savings quantification for all applicable quick wins |
| 2 | `prioritization_workflow.py` | 3: CriteriaConfiguration -> MCDAScoring -> ParetoOptimization | Multi-criteria prioritization with dependency analysis and sequencing optimization |
| 3 | `implementation_planning_workflow.py` | 4: MeasureSelection -> ResourcePlanning -> ScheduleGeneration -> RebateApplication | Implementation planning from approved measure selection through rebate package preparation |
| 4 | `progress_tracking_workflow.py` | 3: ImplementationMonitoring -> SavingsVerification -> PerformanceReporting | Ongoing tracking of implementation status and IPMVP savings verification |
| 5 | `reporting_workflow.py` | 3: DataAggregation -> ReportGeneration -> DistributionDelivery | Report generation and distribution for executive, operational, and compliance audiences |
| 6 | `full_assessment_workflow.py` | 6: FacilitySetup -> DataIngestion -> QuickWinScan -> Prioritization -> ImplementationPlan -> ReportGeneration | Complete end-to-end workflow from facility onboarding to final deliverables |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `quick_wins_scan_report.py` | MD, HTML, PDF, JSON | Complete scan results showing all identified quick wins by category with applicability scoring and preliminary savings ranges |
| 2 | `prioritized_actions_report.py` | MD, HTML, PDF, JSON | MCDA-ranked quick win list with multi-criteria scores, Pareto frontier visualization data, and recommended implementation sequence |
| 3 | `payback_analysis_report.py` | MD, HTML, PDF, JSON | Financial analysis per measure and portfolio: simple payback, discounted payback, NPV, IRR, ROI, LCOE, sensitivity analysis |
| 4 | `carbon_reduction_report.py` | MD, HTML, PDF, JSON | Carbon impact assessment showing tCO2e reduction by measure, scope breakdown (1/2/3), SBTi contribution, and emission factor details |
| 5 | `implementation_plan_report.py` | MD, HTML, PDF, JSON | Phased implementation schedule with resource requirements, contractor specifications, procurement lists, and milestone dates |
| 6 | `progress_dashboard.py` | MD, HTML, JSON | Real-time implementation and savings tracking dashboard with measure status, verified savings, and variance analysis |
| 7 | `executive_summary_report.py` | MD, HTML, PDF, JSON | 2-4 page executive summary: total opportunity, top 10 measures, portfolio financials, carbon impact, and recommended next steps |
| 8 | `rebate_opportunities_report.py` | MD, HTML, PDF, JSON | Utility rebate matching results showing eligible programs, rebate values, filing requirements, deadlines, and net measure costs |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline: FacilitySetup -> DataIngestion -> QuickWinScan -> SavingsEstimation -> CarbonQuantification -> FinancialAnalysis -> Prioritization -> BehavioralModeling -> RebateMatching -> ReportGeneration. Conditional phases for behavioral change (if occupant engagement in scope) and rebate matching (if utility territory configured). Retry with exponential backoff, SHA-256 provenance chain, phase-level caching. |
| 2 | `mrv_bridge.py` | Routes to all 30 AGENT-MRV agents for GHG emissions linked to energy savings: MRV-001 (Stationary Combustion for fuel savings), MRV-009/010 (Scope 2 for electricity savings), MRV-016 (Fuel & Energy Activities Cat 3 for upstream savings), MRV-002 (Refrigerants for refrigeration quick wins). Bi-directional: MRV provides emission factors; quick wins provide savings data for emission reduction reporting. |
| 3 | `data_bridge.py` | Routes to 20 AGENT-DATA agents: DATA-002 (Excel/CSV for utility bills and meter exports), DATA-001 (PDF extraction for invoices and equipment specs), DATA-003 (ERP/Finance for energy procurement costs), DATA-010 (Data Quality Profiler for input data assessment), DATA-014 (Time Series Gap Filler for meter data), DATA-015 (Cross-Source Reconciliation). |
| 4 | `pack031_bridge.py` | PACK-031 Industrial Energy Audit integration: imports detailed equipment efficiency data, process energy maps, baseline models, and audit findings. Quick wins that overlap with PACK-031 ECMs are flagged to avoid double-counting. Shares equipment registry and metering infrastructure configuration. |
| 5 | `pack032_bridge.py` | PACK-032 Building Energy Assessment integration: imports building envelope data, HVAC system assessments, EPC ratings, and retrofit measure analysis. Quick wins are filtered against PACK-032 long-term retrofit roadmap to ensure complementary sequencing. Shares building type profiles and weather data. |
| 6 | `utility_rebate_bridge.py` | External utility incentive program data integration: connects to utility program databases (DSIRE, Energy Star rebate finder, regional utility APIs), refreshes program data quarterly, tracks program changes and expiration dates, and validates rebate eligibility against measure specifications and customer eligibility criteria. |
| 7 | `bms_data_bridge.py` | Building Management System data integration for real-time and historical operational data: HVAC schedules, setpoints, equipment runtimes, zone temperatures, after-hours energy profiles. Supports BACnet/IP, Modbus TCP/RTU, OPC-UA, and REST API protocols. Critical for identifying scheduling, setpoint, and controls quick wins. |
| 8 | `weather_bridge.py` | Weather data integration for savings normalization: HDD/CDD calculation with facility-specific base temperatures, TMY data for annual savings projection, real-time weather for M&V baseline adjustment. Sources: NOAA ISD, Meteostat, Open-Meteo. |
| 9 | `health_check.py` | 18-category system verification covering all 8 engines, 6 workflows, database connectivity, cache status, MRV bridge, DATA bridge, PACK-031/032 bridges, BMS connectivity, weather data freshness, utility rebate database currency, and authentication/authorization. |
| 10 | `setup_wizard.py` | 7-step guided facility configuration: facility profile (type, area, occupancy, operating hours), energy systems inventory (HVAC, lighting, motors, compressed air), utility accounts (electricity, gas, water tariff structures and rate schedules), metering infrastructure, utility territory (for rebate matching), quick win criteria thresholds (max payback, max cost, max disruption), and reporting preferences. |
| 11 | `alert_bridge.py` | Alert and notification integration for quick win implementation tracking: milestone reminders, rebate filing deadline warnings, savings verification due dates, behavioral campaign scheduling, and anomaly alerts when implemented measures show savings degradation. Supports email, SMS, webhook, and in-app notification channels. |

### 3.6 Presets

| # | Preset | Facility Type | Key Characteristics |
|---|--------|--------------|---------------------|
| 1 | `office_building.yaml` | Commercial Office | Lighting 25-35% of energy, HVAC 40-55%, plug loads 10-20%; focus on LED retrofit, scheduling optimization, setpoint deadband, plug load management, occupancy controls. Typical quick win savings: 15-25% of total energy. Building type profiles for open-plan, cellular, and mixed-use offices. |
| 2 | `manufacturing.yaml` | Light to Medium Manufacturing | Compressed air 20-35%, motors 30-50%, lighting 10-15%, HVAC 10-20%; focus on leak repair, motor VSD, pressure reduction, lighting LED, combustion tuning. Typical quick win savings: 10-20% of total energy. Production-normalized savings tracking. |
| 3 | `retail_store.yaml` | Retail / Shopping | Lighting 30-40%, HVAC 30-40%, refrigeration 15-30% (food retail); focus on LED, scheduling, setpoint optimization, refrigeration maintenance, plug loads. Typical quick win savings: 15-30% of total energy. Extended operating hours profile (12-16 hrs/day). |
| 4 | `warehouse.yaml` | Warehouse & Distribution | Lighting 35-50%, HVAC 20-35%, dock doors 10-15%; focus on LED high-bay, occupancy zoning, dock seal repair, destratification fans. Typical quick win savings: 20-40% of total energy. Large area with low occupancy density. |
| 5 | `healthcare.yaml` | Healthcare (Hospital/Clinic) | HVAC 45-60%, lighting 15-25%, DHW 10-15%, medical equipment 10-20%; focus on scheduling (operating hours vs. 24/7 areas), setpoint optimization, lighting controls, DHW temperature management. Typical quick win savings: 8-15% of total energy. Compliance constraints for ventilation rates and temperature ranges. |
| 6 | `education.yaml` | Education (School/University) | Heating 40-55%, lighting 20-30%, HVAC 15-25%; focus on scheduling (term-time vs. holidays), setback during unoccupied periods, LED classroom lighting, behavioral programs with students. Typical quick win savings: 15-30% of total energy. Intermittent occupancy pattern. |
| 7 | `data_center.yaml` | Data Center | Cooling 35-45%, IT load 45-55%, lighting 2-5%, UPS 5-10%; focus on hot/cold aisle containment, raised floor management, airflow optimization, setpoint increase (ASHRAE A1 envelope), blanking panels, cable management for airflow. Typical quick win savings: 5-15% of total energy (PUE improvement 0.05-0.2). |
| 8 | `sme_simplified.yaml` | SME (any sector, <250 employees) | Simplified 5-engine flow (scanner, savings, payback, carbon, reporting); skip behavioral modeling engine for very small facilities; simplified financial analysis; pre-populated typical savings ranges; guided walkthrough with checklist-based scanning; reduced data requirements. Typical quick win savings: 15-35% of total energy (often more untapped potential than large facilities). |

---

## 4. Engine Specifications

### 4.1 Engine 1: Quick Wins Scanner Engine

**Purpose:** Automated scanning of facility operations, equipment, and controls to identify applicable quick win opportunities across 15 categories.

**Scanning Methodology:**

| Scan Type | Data Source | Quick Wins Identified |
|-----------|------------ |----------------------|
| Schedule analysis | BMS schedules, occupancy data, after-hours meter profiles | HVAC/lighting running outside occupied hours |
| Setpoint analysis | BMS setpoints, zone temperature data | Unnecessarily tight deadbands, excessive heating/cooling |
| Equipment runtime | Meter data, BMS equipment status | Equipment running unnecessarily, oversized equipment |
| After-hours energy | Interval meter data (15-min), baseload analysis | Parasitic loads, equipment not shutting down |
| Lighting survey | Lighting inventory, LPD calculation, operating hours | Inefficient lamp types, over-illumination, missing controls |
| Compressed air | Compressor logs, pressure data, leak survey data | Leak losses, excess pressure, inefficient staging |
| Maintenance gaps | Equipment age, maintenance records, efficiency trends | Degraded equipment, dirty filters, fouled coils |
| Envelope walkthrough | Thermal imaging data, air tightness, visual inspection data | Air leaks, missing insulation, failed seals |
| Load profile analysis | 15-min interval data, demand profiles | Demand peaks, poor power factor, load shifting opportunities |

**Quick Win Action Library (80+ actions across 15 categories):**

Each action in the library contains:
- `action_id`: Unique identifier (e.g., `QW-LTG-001`)
- `category`: One of 15 categories
- `description`: Plain-language description
- `applicability_rules`: Conditions under which the action applies (building type, equipment present, climate zone)
- `typical_savings_pct`: Range (low, mid, high) as percentage of category energy
- `typical_cost_range`: EUR range (low, mid, high) for implementation
- `typical_payback_months`: Range for simple payback
- `implementation_complexity`: Low / Medium (quick wins exclude High)
- `disruption_level`: 1-5 scale (1 = no disruption, 5 = facility shutdown)
- `prerequisites`: Other actions or conditions required first
- `data_requirements`: Minimum data needed to quantify savings

**Selected Quick Win Actions (representative sample):**

| Action ID | Category | Description | Typical Savings | Typical Payback |
|-----------|----------|-------------|-----------------|-----------------|
| QW-LTG-001 | Lighting | Replace T8/T12 fluorescent with LED tubes | 40-60% of lighting energy | 6-18 months |
| QW-LTG-003 | Lighting | Install occupancy/vacancy sensors in intermittent spaces | 20-40% per zone | 6-12 months |
| QW-LTG-005 | Lighting | De-lamp over-illuminated areas to target lux levels | 15-25% per area | 0 months (no cost) |
| QW-HVS-001 | HVAC Scheduling | Optimize HVAC start/stop times to match occupancy | 10-20% of HVAC energy | 0-3 months |
| QW-HVS-003 | HVAC Scheduling | Implement night setback / weekend shutdown | 5-15% of heating/cooling | 0-3 months |
| QW-HVP-001 | HVAC Setpoints | Widen thermostat deadband from 1C to 3C | 5-10% of HVAC energy | 0 months |
| QW-CAR-001 | Compressed Air | Repair identified compressed air leaks | 15-25% of CA energy | 1-6 months |
| QW-CAR-003 | Compressed Air | Reduce system pressure by 1 bar | ~7% of CA energy | 0-1 months |
| QW-ENV-001 | Building Envelope | Install weather-stripping on exterior doors | 3-8% of heating energy | 3-12 months |
| QW-PLG-001 | Plug Loads | Install smart power strips at workstations | 10-20% of plug load energy | 6-18 months |
| QW-MTR-001 | Motors & Drives | Install VSD on variable-load pump/fan motors | 20-40% of motor energy | 12-24 months |
| QW-BLR-001 | Boiler/Heating | Tune boiler combustion (O2 adjustment) | 2-5% of boiler fuel | 0-3 months |
| QW-DHW-001 | Water Heating | Reduce DHW storage temperature from 65C to 60C | 5-10% of DHW energy | 0 months |
| QW-REF-001 | Refrigeration | Clean condenser coils | 5-10% of refrigeration energy | 0-1 months |
| QW-PQC-001 | Power Quality | Install power factor correction capacitors | 5-15% demand charge reduction | 12-24 months |
| QW-BMS-001 | Controls & BMS | Enable and commission economizer operation | 10-20% of cooling energy | 0-6 months |
| QW-BHV-001 | Behavioral Change | Launch "switch off" campaign (lights, monitors, equipment) | 3-8% of total energy | 0-3 months |
| QW-STM-001 | Steam/Process Heat | Repair failed steam traps | 10-20% of trap losses | 1-6 months |

**Key Models:**
- `ScanInput` - Facility profile, building type, equipment inventory, BMS data, meter data, maintenance records, occupancy schedules
- `ScanResult` - List of applicable quick wins with applicability score (0-100), preliminary savings range, cost estimate, and data confidence level
- `QuickWinAction` - Action ID, category, description, applicability rules, savings range, cost range, payback range, prerequisites
- `ApplicabilityAssessment` - Action ID, facility-specific applicability score, data quality score, confidence level, justification text

**Non-Functional Requirements:**
- Full facility scan (1,000 equipment items, 12 months of data): <10 minutes
- Action library lookup and filtering: <1 second
- Reproducibility: bit-perfect (same input produces same output, SHA-256 verified)

### 4.2 Engine 2: Payback Calculator Engine

**Purpose:** Comprehensive financial analysis for each quick win measure and the portfolio as a whole.

**Financial Metrics:**

| Metric | Formula | Description |
|--------|---------|-------------|
| Simple Payback | CapEx / Annual_Savings | Years to recover investment from annual savings |
| Discounted Payback | Time to cumulative NPV >= 0 | Years to recover investment at discount rate |
| NPV | Sum(CF_t / (1+r)^t) for t=0..N | Net present value at discount rate r over N years |
| IRR | Rate r where NPV = 0 | Internal rate of return via Newton-Raphson iteration |
| ROI | (NPV / CapEx) * 100 | Return on investment as percentage |
| LCOE | (Sum(Cost_t / (1+r)^t)) / (Sum(kWh_t / (1+r)^t)) | Levelized cost per kWh saved over equipment life |

**Utility Rate Escalation:**

```
Cost_savings_year_t = kWh_savings * rate_year_0 * (1 + escalation_rate)^t

Where:
  rate_year_0 = current utility rate (EUR/kWh or EUR/therm)
  escalation_rate = annual rate increase (default 3.5%, configurable)
  t = year number (0, 1, 2, ..., N)
```

**Tax Incentive Modeling:**

| Incentive Type | Modeling Approach | Examples |
|---------------|-------------------|----------|
| Accelerated depreciation | Reduced CapEx via tax shield | Section 179D (US), Enhanced Capital Allowances (UK) |
| Tax credits | Direct reduction of tax liability | ITC for solar, state energy credits |
| Grants | Direct CapEx reduction (non-repayable) | EU structural funds, national energy agency grants |
| Green finance discount | Reduced interest rate on financing | Green bonds, sustainability-linked loans |
| Carbon credit value | Revenue from verified emission reductions | Voluntary carbon market credits (if applicable) |

**Decimal Arithmetic:**
All financial calculations use Python `decimal.Decimal` with `ROUND_HALF_EVEN` (banker's rounding) to eliminate floating-point errors. Currency values are calculated to 2 decimal places; rates and factors to 6 decimal places.

**Key Models:**
- `FinancialInput` - CapEx, annual energy savings (kWh, therms), energy rates, discount rate, equipment life, escalation rate, tax incentives, rebates, maintenance cost differential
- `FinancialResult` - Simple payback, discounted payback, NPV, IRR, ROI, LCOE, annual cash flows, cumulative NPV series, sensitivity analysis
- `CashFlow` - Year, energy savings (EUR), maintenance delta (EUR), tax benefit (EUR), rebate (EUR, year 0 only), net cash flow, cumulative NPV
- `SensitivityResult` - Parameter varied, values tested, resulting NPV/IRR/payback for each value

**Non-Functional Requirements:**
- Single measure financial analysis: <100 milliseconds
- Portfolio analysis (100 measures): <5 seconds
- IRR convergence: within 0.01% in <100 Newton-Raphson iterations
- Decimal precision: 100% match with reference financial calculator outputs

### 4.3 Engine 3: Energy Savings Estimator Engine

**Purpose:** Estimate energy savings in kWh, therms, and GJ with ASHRAE 14-2014 compliant uncertainty bands.

**Savings Estimation Framework:**

```
Savings = Baseline_Energy - Post_Implementation_Energy +/- Adjustments

Where:
  Baseline_Energy = f(independent_variables) using regression or engineering calculation
  Post_Implementation_Energy = Baseline_Energy * (1 - savings_fraction)
  Adjustments = routine (weather, production) + non-routine (structural changes)
```

**Interactive Effects:**

| Primary Measure | Interactive Effect | Adjustment Factor |
|----------------|-------------------|-------------------|
| LED lighting | Reduced cooling load (less internal heat gain) | +5-15% additional HVAC cooling savings |
| LED lighting | Increased heating load (less waste heat contributing to heating) | -3-8% reduced net savings in heating season |
| Envelope sealing | Reduced HVAC infiltration load | Calculated via infiltration heat loss formula |
| VSD on AHU fans | Reduced duct leakage at lower pressure | +3-5% additional savings |
| HVAC setpoint change | Changed equipment runtime and cycling | Recalculated based on bin analysis |

**HDD/CDD Normalization:**

```
Savings_normalized = Savings_measured * (DD_normal / DD_actual)

Where:
  DD_normal = long-term average degree-days (HDD or CDD)
  DD_actual = degree-days during measurement period
  Applicable to weather-dependent measures only
```

**Rebound Effect:**

| Measure Type | Typical Rebound | Adjustment |
|-------------|-----------------|------------|
| Comfort-related (setpoints, insulation) | 10-15% | Reduce savings estimate by rebound percentage |
| Equipment efficiency (LED, VSD) | 0-5% | Minimal adjustment |
| Behavioral (awareness, competitions) | 5-10% | Moderate adjustment; captured in persistence factor |
| Controls (scheduling, BMS) | 0-3% | Minimal; controls enforce savings |

**Uncertainty Quantification per ASHRAE 14-2014:**

```
Uncertainty_savings = sqrt(
  (uncertainty_baseline)^2 +
  (uncertainty_measurement)^2 +
  (uncertainty_interactive)^2
)

90% confidence interval: Savings +/- (1.645 * Uncertainty_savings)
```

**Key Models:**
- `SavingsInput` - Measure specification, baseline energy data, operating parameters, weather data, occupancy data
- `SavingsResult` - Point estimate (kWh, therms, GJ), 90% CI upper/lower, interactive effects breakdown, normalization factors, rebound adjustment, degradation schedule
- `InteractiveEffect` - Primary measure, affected system, direction (increase/decrease), magnitude, calculation basis
- `UncertaintyAnalysis` - Component uncertainties, combined uncertainty, confidence interval, data quality score

### 4.4 Engine 4: Carbon Reduction Engine

**Purpose:** Calculate tCO2e emission reductions from quick win energy savings per GHG Protocol.

**Emission Factor Application:**

| Scope | Energy Type | Factor Source | Methodology |
|-------|-----------|---------------|-------------|
| Scope 1 | Natural gas savings | IPCC 2006 Guidelines (56.1 kgCO2/GJ NCV) | Direct fuel combustion reduction |
| Scope 1 | Fuel oil savings | IPCC 2006 Guidelines (77.4 kgCO2/GJ NCV) | Direct fuel combustion reduction |
| Scope 1 | LPG savings | IPCC 2006 Guidelines (63.1 kgCO2/GJ NCV) | Direct fuel combustion reduction |
| Scope 2 (location) | Electricity savings | National grid average EF (DEFRA, UBA, EPA eGRID) | Grid average emission factor |
| Scope 2 (market) | Electricity savings | Residual mix EF, supplier-specific, RE certificates | Contractual instruments |
| Scope 3 Cat 3 | Upstream fuel/energy | WTT emission factors from DEFRA/UBA databases | Well-to-tank and T&D losses |

**Carbon Reduction Calculation:**

```
CO2e_reduction_scope1 = fuel_savings_GJ * fuel_EF_kgCO2_per_GJ / 1000  (tCO2e)

CO2e_reduction_scope2_location = electricity_savings_kWh * grid_EF_location / 1000  (tCO2e)

CO2e_reduction_scope2_market = electricity_savings_kWh * grid_EF_market / 1000  (tCO2e)

CO2e_reduction_scope3 = fuel_savings_GJ * WTT_EF / 1000 + elec_savings_kWh * TD_loss_EF / 1000  (tCO2e)

Total_CO2e_reduction = scope1 + scope2 + scope3
```

**SBTi Alignment:**

| Assessment | Method | Output |
|-----------|--------|--------|
| Target contribution | Quick win tCO2e / annual SBTi target tCO2e | Percentage of annual target achieved through quick wins |
| Pathway alignment | Quick win reductions mapped against absolute contraction pathway | On-track / behind / ahead assessment |
| Scope coverage | Breakdown of reductions by scope vs. SBTi scope requirements | Coverage adequacy assessment |

**Key Models:**
- `CarbonInput` - Energy savings by fuel type and electricity, facility location, emission factor set, SBTi target parameters
- `CarbonResult` - tCO2e reduction by scope (1, 2-location, 2-market, 3), total reduction, emission factors used, SBTi alignment score, marginal abatement cost (EUR/tCO2e)
- `EmissionFactor` - Factor value (kgCO2e/kWh or kgCO2e/GJ), source, year, region, scope applicability
- `SBTiAlignment` - Target year, target reduction, actual reduction from quick wins, gap, trajectory assessment

### 4.5 Engine 5: Implementation Prioritizer Engine

**Purpose:** Multi-criteria prioritization of quick win measures with dependency management and sequencing optimization.

**MCDA Criteria:**

| Criterion | Weight (default) | Scale | Direction |
|-----------|-----------------|-------|-----------|
| Annual energy savings (kWh) | 20% | Continuous | Higher is better |
| Annual carbon reduction (tCO2e) | 15% | Continuous | Higher is better |
| Simple payback (years) | 20% | Continuous | Lower is better |
| Capital cost (EUR) | 10% | Continuous | Lower is better |
| Implementation disruption | 10% | 1-5 | Lower is better |
| Maintenance impact | 5% | -2 to +2 | Higher is better (positive = less maintenance) |
| Occupant comfort impact | 5% | 1-5 | Higher is better |
| Regulatory alignment | 5% | 0-100 | Higher is better |
| Rebate availability | 5% | 0-100% of CapEx | Higher is better |
| Implementation readiness | 5% | 1-5 | Higher is better |

**MCDA Scoring:**

```
Score_normalized = (value - min_value) / (max_value - min_value)  [for "higher is better"]
Score_normalized = (max_value - value) / (max_value - min_value)  [for "lower is better"]

Overall_score = Sum(weight_i * score_normalized_i) for all criteria
```

**Pareto Frontier:**
Identifies non-dominated solutions across two or more criteria (typically savings vs. cost). A measure is Pareto-optimal if no other measure provides both higher savings and lower cost.

**Dependency Graph:**
Models prerequisites and synergies:
- `QW-BMS-001` (commission economizer) enables `QW-HVS-004` (free cooling optimization)
- `QW-LTG-001` (LED retrofit) synergizes with `QW-LTG-003` (occupancy sensors) -- combined savings exceed sum of individual
- `QW-CAR-001` (leak repair) should precede `QW-CAR-003` (pressure reduction) for maximum benefit

**Sequencing Optimization:**
Determines implementation order considering:
- Cash flow (implement highest-ROI first to fund subsequent measures)
- Seasonal windows (envelope work in spring/fall, HVAC commissioning before heating/cooling season)
- Resource availability (contractor scheduling, equipment lead times)
- Dependency constraints (prerequisites must complete before dependents)

**Key Models:**
- `PrioritizationInput` - List of measures with all criteria values, criteria weights, dependency definitions, resource constraints, seasonal preferences
- `PrioritizationResult` - Ranked measure list with overall MCDA score, Pareto frontier members, dependency-aware sequence, implementation schedule
- `MCDAScore` - Per-criterion normalized score, weighted contribution, overall score, rank
- `DependencyGraph` - Nodes (measures), edges (prerequisite, synergy, conflict), topological sort order

### 4.6 Engine 6: Behavioral Change Engine

**Purpose:** Model, plan, and track behavioral energy conservation actions.

**Behavioral Action Library (40+ actions):**

| # | Action | Category | Typical Savings | Persistence (12-month) |
|---|--------|----------|-----------------|----------------------|
| 1 | Turn off lights when leaving | Lighting | 5-10% per zone | 40-60% (information) |
| 2 | Power down monitors/PCs at end of day | Plug loads | 5-15% of IT energy | 30-50% (information) |
| 3 | Adjust personal thermostat by 1C | HVAC | 3-5% of HVAC | 50-70% (with feedback) |
| 4 | Use stairs instead of elevator (1-3 floors) | Vertical transport | 2-5% of elevator energy | 20-40% (motivation decays) |
| 5 | Report equipment/lighting left on | General | 3-8% via rapid response | 60-80% (structural change) |
| 6 | Use natural ventilation when conditions allow | HVAC | 5-15% of cooling | 50-70% (seasonal) |
| 7 | Reduce print volume | Office equipment | 3-5% of print energy | 60-80% (digital habit formation) |
| 8 | Unplug chargers and peripherals when not in use | Plug loads | 1-3% of plug load | 30-50% (convenience barrier) |

**Rogers Diffusion Model:**

```
Adoption(t) = M / (1 + exp(-k * (t - t_mid)))

Where:
  M = maximum adoption rate (% of population)
  k = adoption speed coefficient
  t = time (weeks from campaign launch)
  t_mid = time to 50% of maximum adoption

Population segments:
  Innovators: first 2.5% (adopt within 2 weeks)
  Early adopters: next 13.5% (adopt within 4-8 weeks)
  Early majority: next 34% (adopt within 8-16 weeks)
  Late majority: next 34% (adopt within 16-32 weeks)
  Laggards: final 16% (adopt after 32 weeks, or never)
```

**Persistence Factor:**

| Intervention Type | 6-month Persistence | 12-month Persistence | 24-month Persistence |
|-------------------|--------------------|--------------------|---------------------|
| Information only (posters, emails) | 40-60% | 30-50% | 15-30% |
| Feedback (dashboards, reports) | 55-75% | 45-65% | 30-50% |
| Feedback + incentives (competitions, rewards) | 65-85% | 60-80% | 45-65% |
| Structural/default changes (auto-off, setback defaults) | 85-95% | 85-95% | 80-90% |
| Social norms + commitment devices | 60-80% | 55-70% | 40-55% |

**Gamification Framework:**

| Element | Implementation | Impact |
|---------|---------------|--------|
| Energy dashboards | Real-time floor/zone energy display | +5-10% savings awareness |
| Team competitions | Monthly energy reduction challenge by department/floor | +8-15% during competition |
| Individual badges | Achievement badges for sustained behavior change | +3-5% persistence improvement |
| Leaderboards | Weekly ranking of zones/floors by energy performance | +5-10% through social comparison |
| Reward programs | Points redeemable for small rewards (coffee, parking) | +10-15% participation rate |

**Key Models:**
- `BehavioralInput` - Building population, segmentation data, baseline behavioral assessment, planned interventions, historical campaign data
- `BehavioralResult` - Adoption curve projections, persistence-adjusted savings forecast, gamification metrics, campaign schedule, per-action savings estimate
- `AdoptionCurve` - Time series of adoption percentage by Rogers segment, cumulative adoption, savings realization
- `PersistenceForecast` - Month-by-month savings retention percentage, intervention reinforcement schedule, decay rate

### 4.7 Engine 7: Utility Rebate Engine

**Purpose:** Match quick win measures to available utility incentive and rebate programs.

**Program Types:**

| Type | Description | Typical Value | Documentation |
|------|-------------|---------------|---------------|
| Prescriptive rebate | Fixed amount per unit (lamp, motor, thermostat) | USD 1-200 per unit | Equipment spec sheet, invoice |
| Custom/calculated rebate | Variable amount based on verified kWh savings | USD 0.05-0.15 per kWh | Pre/post measurement, calculations |
| Upstream/midstream | Rebate paid to distributor/contractor (lower retail price) | 20-50% price reduction | None (applied at point of sale) |
| Direct install | Utility covers full installation cost | 100% of measure cost | Utility contractor performs work |
| Demand response | Payment for load curtailment during peak events | USD 50-200 per kW curtailed | Load monitoring, event participation |

**Matching Algorithm:**

```
For each quick_win_measure:
  1. Identify utility territory from facility location
  2. Filter programs by: territory, measure type, customer class, building type
  3. Check eligibility: minimum efficiency thresholds, equipment age, existing conditions
  4. Calculate rebate value: prescriptive (lookup) or custom (kWh * $/kWh)
  5. Check stacking rules: can multiple rebates combine?
  6. Apply program cap: maximum rebate per measure or per facility per year
  7. Verify timeline: program active, filing deadline not passed
  8. Generate application package: required forms, documentation list, submission instructions
```

**Utility Program Database:**
- 100+ utility programs covering major US, UK, and EU utility territories
- Quarterly refresh cycle for program updates
- Program data: eligibility, measure types, rebate amounts, caps, deadlines, documentation
- Historical tracking: previous year program changes, sunset programs, new programs

**Key Models:**
- `RebateInput` - Facility location, utility territory, customer class, list of quick win measures with specifications
- `RebateResult` - Per-measure: eligible programs, rebate amount, net cost after rebate, documentation requirements, filing deadline. Portfolio total rebate value.
- `UtilityProgram` - Program ID, utility name, territory, measure types, prescriptive values, custom rate, eligibility rules, deadline, documentation list, stacking rules
- `RebateApplication` - Measure details, program selected, rebate amount, required documents, submission instructions, deadline

### 4.8 Engine 8: Quick Wins Reporting Engine

**Purpose:** Aggregate analysis results into dashboards and reports with implementation tracking and M&V verification.

**Dashboard Panels:**

| Panel | Content | Update Frequency |
|-------|---------|-----------------|
| Portfolio Summary | Total measures, total savings (kWh, EUR, tCO2e), portfolio payback, portfolio IRR | On-demand |
| Category Breakdown | Savings by quick win category (15 categories), interactive Pareto chart | On-demand |
| Implementation Tracker | Measure status pipeline: Identified -> Approved -> Procured -> Installed -> Verified | Weekly |
| Savings Verification | Actual vs. projected savings per measure, cumulative CUSUM chart | Monthly |
| Carbon Impact | tCO2e reduction by scope, contribution to SBTi target, marginal abatement cost curve | Monthly |
| Rebate Tracker | Rebate applications submitted, approved, received; total rebate value; upcoming deadlines | Monthly |
| Behavioral Metrics | Campaign adoption rates, persistence tracking, gamification scores | Weekly |
| Timeline | Gantt chart of implementation schedule with milestones and dependencies | Weekly |

**IPMVP Verification Integration:**

| IPMVP Option | Application in Quick Wins | Statistical Requirements |
|-------------|--------------------------|------------------------|
| Option A (Key Parameter) | LED retrofit (measure power; stipulate hours), motor VSD (measure kW; stipulate flow) | Key parameter uncertainty +/- 5% at 90% confidence |
| Option B (All Parameters) | Compressed air optimization (measure power, flow, pressure continuously) | CV(RMSE) <= 20% monthly, NMBE +/- 5% |

**Export Formats:**
- Markdown (MD): for version control and documentation
- HTML: for web-based dashboards and interactive viewing
- PDF: for executive and compliance distribution
- JSON: for API consumption and system integration
- CSV: for data export and spreadsheet analysis

**Key Models:**
- `ReportInput` - Scan results, savings estimates, financial analysis, carbon reduction, prioritization, behavioral data, rebate matching, implementation status, M&V results
- `ReportResult` - Formatted report content, dashboard data, export files, provenance hash
- `ImplementationStatus` - Measure ID, status (identified/approved/procured/installed/verified), dates, responsible person, notes
- `VerificationResult` - Measure ID, IPMVP option, baseline energy, post-implementation energy, verified savings, projected savings, variance, statistical validity

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Facility Scan Workflow

**Purpose:** End-to-end facility scanning from registration through savings quantification.

**Phase 1: Facility Registration**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Create facility profile | Facility name, address, type, area, occupancy, operating hours | Facility record with unique ID | <5 minutes |
| 1.2 | Select building/facility preset | Building type selection from 8 presets | Pre-configured scanning parameters | <2 minutes |
| 1.3 | Configure quick win criteria | Max payback, max cost, max disruption thresholds | Quick win filter configuration | <5 minutes |
| 1.4 | Register utility accounts | Utility provider, account number, rate schedule, territory | Utility configuration for rebate matching | <10 minutes |

**Phase 2: Data Collection**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Import utility bills | PDF/Excel utility bills (12+ months) | Standardized energy consumption and cost series | <15 minutes |
| 2.2 | Import BMS data (if available) | Schedules, setpoints, equipment runtimes, zone data | Operational data for scanning analysis | <10 minutes |
| 2.3 | Import equipment inventory | Equipment type, nameplate data, age, condition | Equipment registry for efficiency analysis | <15 minutes |
| 2.4 | Import weather data | Auto-fetch from nearest station | HDD/CDD for normalization | <2 minutes (auto) |
| 2.5 | Data quality assessment | All imported data | Quality score (0-100), gap report, confidence rating | <3 minutes (auto) |

**Phase 3: Quick Win Scanning**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Run schedule analysis | BMS schedules, after-hours meter profiles | HVAC/lighting schedule quick wins | <3 minutes (auto) |
| 3.2 | Run setpoint analysis | BMS setpoints, zone temperature data | Setpoint optimization quick wins | <2 minutes (auto) |
| 3.3 | Run equipment analysis | Equipment inventory, runtime data, efficiency data | Equipment upgrade/maintenance quick wins | <5 minutes (auto) |
| 3.4 | Run lighting analysis | Lighting inventory, LPD data, operating hours | Lighting retrofit and controls quick wins | <3 minutes (auto) |
| 3.5 | Run category-wide scan | All data against 80+ action library | Full quick wins list with applicability scores | <5 minutes (auto) |

**Phase 4: Savings Estimation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Estimate per-measure savings | Applicable quick wins, facility data | kWh/therms/GJ savings with 90% CI | <5 minutes (auto) |
| 4.2 | Apply interactive effects | Measure list, interaction matrix | Adjusted savings with interaction corrections | <2 minutes (auto) |
| 4.3 | Normalize for weather | Savings estimates, HDD/CDD data | Weather-normalized annual savings | <1 minute (auto) |
| 4.4 | Apply rebound corrections | Savings estimates, measure types | Net savings after rebound adjustment | <1 minute (auto) |
| 4.5 | Generate scan summary | All estimated savings | Portfolio savings summary with confidence ratings | <2 minutes (auto) |

**Acceptance Criteria:**
- [ ] All 4 phases execute sequentially with data passing between phases
- [ ] Phase 2 data quality assessment flags gaps and provides confidence rating
- [ ] Phase 3 scans all 15 categories and reports applicability for each action
- [ ] Phase 4 savings estimates include 90% confidence intervals per ASHRAE 14
- [ ] Total workflow duration <2 hours for typical commercial building with BMS data
- [ ] Full SHA-256 provenance chain from input data through every calculation to scan results

### 5.2 Workflow 2: Prioritization Workflow

**Purpose:** Multi-criteria prioritization with dependency analysis and sequencing optimization.

**Phase 1: Criteria Configuration**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Set MCDA criteria weights | User preferences or preset defaults | Weighted criteria configuration | <5 minutes |
| 1.2 | Define constraints | Budget ceiling, maximum disruption, seasonal constraints | Constraint set for optimization | <5 minutes |
| 1.3 | Load dependency rules | Prerequisite and synergy relationships from action library | Dependency graph | <1 minute (auto) |

**Phase 2: MCDA Scoring**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Normalize scores | Raw criteria values across all measures | Normalized 0-1 scores per criterion | <1 minute (auto) |
| 2.2 | Calculate weighted scores | Normalized scores, criteria weights | Overall MCDA score per measure | <1 minute (auto) |
| 2.3 | Rank measures | MCDA scores | Ranked list with score breakdown | <1 minute (auto) |

**Phase 3: Pareto Optimization**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Compute Pareto frontier | Savings vs. cost for all measures | Non-dominated solution set | <1 minute (auto) |
| 3.2 | Apply dependency constraints | Ranked list, dependency graph | Dependency-aware implementation sequence | <2 minutes (auto) |
| 3.3 | Generate implementation sequence | Sequence, resource constraints, seasonal windows | Optimized implementation order with timeline | <2 minutes (auto) |

**Acceptance Criteria:**
- [ ] MCDA scoring produces deterministic ranking (same inputs = same ranking)
- [ ] Pareto frontier correctly identifies non-dominated solutions
- [ ] Dependency graph prevents prerequisite violations in sequence
- [ ] Total workflow duration <15 minutes
- [ ] Criteria weights are user-configurable and default to preset values

### 5.3 Workflow 3: Implementation Planning Workflow

**Purpose:** Detailed implementation planning from measure selection through rebate application.

**Phases:** MeasureSelection -> ResourcePlanning -> ScheduleGeneration -> RebateApplication

**Acceptance Criteria:**
- [ ] Selected measures include bill of materials and contractor specifications
- [ ] Resource plan identifies internal vs. external labor requirements
- [ ] Schedule respects dependency constraints and seasonal windows
- [ ] Rebate applications are pre-populated with measure data and facility information
- [ ] Total workflow duration <30 minutes for a portfolio of 20 measures

### 5.4 Workflow 4: Progress Tracking Workflow

**Purpose:** Ongoing tracking of implementation status and IPMVP savings verification.

**Phases:** ImplementationMonitoring -> SavingsVerification -> PerformanceReporting

**Acceptance Criteria:**
- [ ] Implementation status updates supported via API, CSV upload, and manual entry
- [ ] IPMVP Option A/B verification calculates verified savings with statistical validation
- [ ] Variance analysis flags measures where actual savings differ >25% from projected
- [ ] Performance reports generated automatically on configurable schedule (weekly/monthly)
- [ ] Savings degradation detection alerts when verified savings trend downward

### 5.5 Workflow 5: Reporting Workflow

**Purpose:** Report generation and distribution for executive, operational, and compliance audiences.

**Phases:** DataAggregation -> ReportGeneration -> DistributionDelivery

**Acceptance Criteria:**
- [ ] Reports generated in all supported formats (MD, HTML, PDF, JSON, CSV)
- [ ] Executive summary is <4 pages with portfolio-level metrics
- [ ] Distribution supports email, webhook, and file system delivery
- [ ] Report provenance hash chain verifiable from raw data through final report

### 5.6 Workflow 6: Full Assessment Workflow

**Purpose:** Complete end-to-end workflow combining all phases from facility onboarding through final deliverables.

**Phases:** FacilitySetup -> DataIngestion -> QuickWinScan -> Prioritization -> ImplementationPlan -> ReportGeneration

**Acceptance Criteria:**
- [ ] All 6 phases execute sequentially with full data handoff between phases
- [ ] Total workflow duration <4 hours for a typical facility
- [ ] All outputs include SHA-256 provenance chain
- [ ] Final deliverables include all 8 template outputs

---

## 6. Template Specifications

### 6.1 Template 1: Quick Wins Scan Report

**Purpose:** Complete scan results showing all identified quick wins with applicability scoring and savings ranges.

**Sections:**
- Executive summary: facility overview, total opportunities identified, total savings potential (kWh, EUR, tCO2e)
- Category-by-category findings: each of 15 categories with applicable/not-applicable justification
- Per-measure detail: action ID, description, applicability score, savings range, cost range, payback range, data confidence
- Gap analysis: categories where data was insufficient for assessment
- Data quality summary: input data sources, completeness scores, confidence ratings
- Recommended next steps: top 10 highest-impact measures, data collection recommendations for low-confidence measures

**Output Formats:** MD, HTML, PDF, JSON

### 6.2 Template 2: Prioritized Actions Report

**Purpose:** MCDA-ranked quick win list with multi-criteria scores and implementation sequence.

**Sections:**
- Prioritized measure list with overall MCDA score and per-criterion breakdown
- Pareto frontier chart data (savings vs. cost, savings vs. carbon)
- Dependency graph visualization data
- Recommended implementation sequence with phase groupings
- Sensitivity analysis: how ranking changes with different criteria weight profiles
- Portfolio-level summary: total savings, cost, payback if all measures implemented

**Output Formats:** MD, HTML, PDF, JSON

### 6.3 Template 3: Payback Analysis Report

**Purpose:** Financial analysis per measure and portfolio level.

**Sections:**
- Per-measure financial summary: CapEx, annual savings, simple payback, discounted payback, NPV, IRR, ROI, LCOE
- Annual cash flow tables with utility rate escalation
- Cumulative NPV chart data per measure
- Tax incentive and rebate impact on net cost and payback
- Sensitivity analysis: NPV/IRR sensitivity to discount rate, energy price escalation, equipment life
- Portfolio financial summary: total investment, total NPV, portfolio IRR, aggregate payback

**Output Formats:** MD, HTML, PDF, JSON

### 6.4 Template 4: Carbon Reduction Report

**Purpose:** Carbon impact assessment with scope breakdown and SBTi alignment.

**Sections:**
- Per-measure tCO2e reduction by scope (1, 2-location, 2-market, 3)
- Emission factors used with source citations
- Portfolio carbon reduction summary
- Marginal abatement cost curve data (EUR/tCO2e for each measure, ordered by cost)
- SBTi target contribution analysis
- Year-over-year carbon reduction trajectory from quick win implementation
- Regulatory disclosure alignment (ESRS E1, CDP C4/C7)

**Output Formats:** MD, HTML, PDF, JSON

### 6.5 Template 5: Implementation Plan Report

**Purpose:** Phased implementation schedule with resource and procurement details.

**Sections:**
- Implementation phases with milestone dates
- Per-measure implementation details: scope of work, contractor requirements, procurement list, lead time
- Resource requirements: internal labor hours, external contractor hours, budget per phase
- Dependency sequence: which measures must complete before others begin
- Seasonal constraints: optimal implementation windows
- Risk assessment: implementation risks and mitigation strategies
- Rebate filing timeline aligned with implementation schedule

**Output Formats:** MD, HTML, PDF, JSON

### 6.6 Template 6: Progress Dashboard

**Purpose:** Real-time implementation and savings tracking.

**Dashboard Panels:**
- Implementation pipeline: counts by status (identified, approved, procured, installed, verified)
- Savings tracker: cumulative verified savings vs. projected (kWh, EUR, tCO2e)
- CUSUM chart: cumulative sum of actual minus projected savings over time
- Variance analysis: measures with >25% deviation from projection, flagged for investigation
- Rebate status: submitted, approved, received amounts
- Behavioral campaign metrics: adoption rates, persistence tracking
- Timeline: Gantt chart with actual vs. planned implementation dates
- Budget tracker: actual spend vs. budgeted, rebates received vs. projected

**Output Formats:** MD, HTML, JSON

### 6.7 Template 7: Executive Summary Report

**Purpose:** 2-4 page executive summary for management decision-making.

**Sections:**
- One-paragraph facility overview and assessment scope
- Key findings: top 5 quick wins with savings and payback
- Portfolio financials: total investment, total annual savings, portfolio payback, NPV, IRR
- Carbon impact: total tCO2e reduction, SBTi contribution
- Available rebates: total rebate value, net investment after rebates
- Implementation recommendation: quick-start package (measures implementable within 30 days)
- Risk and confidence assessment

**Output Formats:** MD, HTML, PDF, JSON

### 6.8 Template 8: Rebate Opportunities Report

**Purpose:** Utility rebate matching results with application guidance.

**Sections:**
- Per-measure rebate eligibility: eligible programs, rebate value, net cost after rebate
- Program details: utility name, program name, requirements, filing deadline, documentation needed
- Stacking analysis: where multiple rebates can be combined for maximum benefit
- Total rebate value summary: by program, by measure category, portfolio total
- Application checklist: step-by-step instructions for each rebate application
- Timeline: filing deadlines and recommended submission dates
- Historical rebate data: prior year program values for trend awareness

**Output Formats:** MD, HTML, PDF, JSON

---

## 7. Integration Specifications

### 7.1 Integration 1: Pack Orchestrator

**Purpose:** Master orchestration pipeline for all PACK-033 engines.

**DAG Pipeline (10 phases):**

```
Phase 1: Facility Setup (setup_wizard)
  |
Phase 2: Data Ingestion (data_bridge, bms_data_bridge, weather_bridge)
  |
Phase 3: Quick Win Scanning (quick_wins_scanner_engine)
  |
Phase 4: Savings Estimation (energy_savings_estimator_engine)
  |
Phase 5: Carbon Quantification (carbon_reduction_engine)
  |
Phase 6: Financial Analysis (payback_calculator_engine)
  |
Phase 7: Prioritization (implementation_prioritizer_engine)
  |
Phase 8: Behavioral Modeling [conditional] (behavioral_change_engine)
  [if behavioral_change_in_scope and building_occupant_count > 10]
  |
Phase 9: Rebate Matching [conditional] (utility_rebate_engine)
  [if utility_territory_configured]
  |
Phase 10: Report Generation (quick_wins_reporting_engine + all templates)
```

**Orchestrator Features:**
- Conditional phase execution based on facility type and preset configuration
- Retry with exponential backoff (max 3 retries per phase)
- SHA-256 provenance chain across all phases
- Phase-level caching (skip re-execution if inputs unchanged)
- Progress tracking with percentage completion per phase
- Error isolation (failed conditional phase does not block required phases)

### 7.2 Integration 2: MRV Bridge

**Purpose:** Connect quick win energy savings to GHG emission calculations.

**Data Flow:**
- Energy savings (kWh electricity, m3 gas, L fuel oil) -> MRV agents for emission reduction calculation
- MRV-001: fuel combustion savings -> Scope 1 CO2e reduction
- MRV-009/010: grid electricity savings -> Scope 2 CO2e reduction (location and market-based)
- MRV-016: upstream fuel/energy savings -> Scope 3 Category 3 CO2e reduction
- MRV-002: refrigerant quick wins -> refrigerant emission reduction

**Bi-directional Benefits:**
- Quick wins -> MRV: accurate savings data for emission reduction reporting
- MRV -> Quick wins: verified emission factors for carbon reduction calculations

### 7.3 Integration 3: Data Bridge

**Purpose:** Route quick win data through AGENT-DATA agents for quality assurance.

**Data Pipeline:**
- Utility bills -> DATA-002 (Excel/CSV Normalizer) -> standardized consumption series
- Equipment specs -> DATA-001 (PDF extraction) -> equipment data
- ERP cost data -> DATA-003 (ERP/Finance Connector) -> cost time series
- All data -> DATA-010 (Data Quality Profiler) -> quality score and gap report
- Gapped meter data -> DATA-014 (Time Series Gap Filler) -> complete time series
- Multiple sources -> DATA-015 (Cross-Source Reconciliation) -> reconciled dataset

### 7.4 Integration 4: PACK-031 Bridge

**Purpose:** Integration with PACK-031 Industrial Energy Audit Pack.

**Key Functions:**
- Import equipment efficiency data from PACK-031 equipment registry
- Import process energy maps for industrial quick win identification
- Import baseline models for savings estimation normalization
- Cross-reference: flag quick wins that overlap with PACK-031 ECMs to avoid double-counting
- Share metering infrastructure and weather data configuration

### 7.5 Integration 5: PACK-032 Bridge

**Purpose:** Integration with PACK-032 Building Energy Assessment Pack.

**Key Functions:**
- Import building envelope assessment data for envelope quick win identification
- Import HVAC system assessments for HVAC quick win quantification
- Import EPC ratings for regulatory alignment scoring
- Cross-reference: align quick wins with PACK-032 retrofit roadmap staging
- Share building type profiles and BMS data configuration

### 7.6 Integration 6: Utility Rebate Bridge

**Purpose:** External utility incentive program data integration.

**Data Sources:**
- DSIRE (Database of State Incentives for Renewables & Efficiency) -- US programs
- Energy Star Rebate Finder -- US programs
- Regional utility APIs and program websites
- EU Member State energy agency incentive databases
- Manual program data entry for unlisted programs

**Refresh Cycle:** Quarterly automatic refresh with manual override capability

### 7.7 Integration 7: BMS Data Bridge

**Purpose:** Building Management System data integration for operational quick win identification.

**Supported Protocols:**
- BACnet/IP: HVAC schedules, setpoints, equipment status
- Modbus TCP/RTU: power meters, industrial equipment
- OPC-UA: industrial automation, SCADA systems
- REST API: cloud-based BMS platforms

**Critical Data Points for Quick Win Scanning:**
- HVAC operating schedules vs. occupancy schedules (after-hours runtime detection)
- Zone temperature setpoints vs. comfort requirements (over-conditioning detection)
- Equipment runtime hours vs. production schedules (unnecessary runtime detection)
- After-hours electrical load profiles (baseload anomaly detection)

### 7.8 Integration 8: Weather Bridge

**Purpose:** Weather data integration for savings normalization.

**Data Sources:**
- NOAA ISD (Integrated Surface Database) for global coverage
- Meteostat / Open-Meteo for European coverage
- Degree-day calculation with facility-specific base temperatures

**Output:**
- HDD and CDD for facility location, calculated at facility-specific base temperature
- TMY data for annual savings projection
- Real-time weather for ongoing M&V baseline adjustment

### 7.9 Integration 9: Health Check

**Purpose:** System verification for all PACK-033 components.

**18 Verification Categories:**

| # | Category | Checks |
|---|----------|--------|
| 1 | Engine availability | All 8 engines respond to health ping |
| 2 | Workflow availability | All 6 workflows respond to health ping |
| 3 | Template availability | All 8 templates generate test output |
| 4 | Database connectivity | PostgreSQL connection, migration status |
| 5 | Redis cache | Cache connectivity and response time |
| 6 | MRV bridge | Connection to MRV agents (001, 002, 009, 010, 016) |
| 7 | Data bridge | Connection to DATA agents (001, 002, 003, 010, 014, 015) |
| 8 | Foundation bridge | Connection to FOUND agents (001-010) |
| 9 | PACK-031 bridge | Connection to PACK-031 engines (if deployed) |
| 10 | PACK-032 bridge | Connection to PACK-032 engines (if deployed) |
| 11 | BMS connectivity | Protocol adapters responding (BACnet, Modbus, OPC-UA) |
| 12 | Weather data feed | Latest weather data within 24 hours |
| 13 | Utility rebate database | Program data loaded and within refresh cycle |
| 14 | Action library | 80+ actions loaded with validation checksums |
| 15 | Authentication | JWT RS256 token issuance/validation |
| 16 | Authorization | RBAC permission checks for all 5 roles |
| 17 | Provenance | SHA-256 hash generation/verification |
| 18 | Disk/memory | Storage <80% capacity, memory <80% ceiling |

### 7.10 Integration 10: Setup Wizard

**Purpose:** Guided 7-step facility configuration for new deployments.

**Steps:**

| Step | Configuration | Inputs Required |
|------|--------------|-----------------|
| 1. Facility Profile | Name, address, building type, floor area, occupancy, operating hours | Basic facility information |
| 2. Energy Systems | HVAC type, lighting type, motor inventory, compressed air, boiler/heating | System-level equipment data |
| 3. Utility Accounts | Provider, rate schedule, account number, territory | Utility configuration |
| 4. Metering Infrastructure | Main meters, sub-meters, BMS data availability | Data source configuration |
| 5. Utility Territory | Service territory for rebate matching | Location-based configuration |
| 6. Quick Win Criteria | Max payback, max cost, max disruption, priority weights | Assessment configuration |
| 7. Reporting Preferences | Report format, distribution list, schedule, dashboard preferences | Output configuration |

### 7.11 Integration 11: Alert Bridge

**Purpose:** Alert and notification integration for implementation tracking and verification.

**Alert Types:**

| Alert | Trigger | Channel | Audience |
|-------|---------|---------|----------|
| Milestone reminder | Implementation milestone approaching (7 days) | Email, in-app | Facility manager |
| Rebate deadline | Filing deadline approaching (30 days) | Email, SMS | Energy manager |
| Savings verification due | M&V measurement period complete | Email | Energy manager |
| Savings degradation | Verified savings trending >15% below projected | Email, SMS | Energy manager, facility engineer |
| Campaign milestone | Behavioral campaign phase transition | Email, in-app | Energy champion |
| Budget alert | Implementation spending >80% of budget | Email | CFO, energy manager |

---

## 8. Preset Specifications

### 8.1 Preset 1: Office Building

**Facility Type:** Commercial office (open-plan, cellular, mixed)
**Energy Profile:** HVAC-dominated with significant lighting and plug loads

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 8 |
| Behavioral change engine | Enabled (high occupant count) |
| Utility rebate engine | Enabled |
| Primary quick win categories | Lighting, HVAC scheduling, HVAC setpoints, plug loads, controls/BMS, behavioral |
| Secondary categories | Building envelope, water heating, power quality |
| Typical total savings | 15-25% of total energy cost |
| Key metrics | kWh/m2/yr, EUR/m2/yr, tCO2e/m2/yr |

### 8.2 Preset 2: Manufacturing

**Facility Type:** Light to medium manufacturing
**Energy Profile:** Motor and compressed air dominated

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 8 (behavioral conditional on office/admin area size) |
| Primary quick win categories | Compressed air, motors & drives, lighting, boiler/heating, steam/process heat |
| Secondary categories | HVAC scheduling, controls/BMS, power quality |
| Typical total savings | 10-20% of total energy cost |
| Key metrics | kWh/unit produced, EUR/unit, tCO2e/unit |

### 8.3 Preset 3: Retail Store

**Facility Type:** Retail / shopping
**Energy Profile:** Lighting and HVAC dominated; refrigeration in food retail

| Parameter | Value |
|-----------|-------|
| Engines enabled | 7 (behavioral simplified for customer-facing) |
| Primary quick win categories | Lighting, HVAC scheduling, HVAC setpoints, refrigeration, plug loads |
| Secondary categories | Building envelope, controls/BMS, water heating |
| Typical total savings | 15-30% of total energy cost |

### 8.4 Preset 4: Warehouse

**Facility Type:** Warehouse and distribution
**Energy Profile:** Lighting dominated; envelope losses significant

| Parameter | Value |
|-----------|-------|
| Engines enabled | 7 (behavioral reduced for low occupancy) |
| Primary quick win categories | Lighting, building envelope (dock doors/seals), HVAC scheduling, controls/BMS |
| Secondary categories | Motors & drives, compressed air (if present), power quality |
| Typical total savings | 20-40% of total energy cost |

### 8.5 Preset 5: Healthcare

**Facility Type:** Hospital, clinic, care home
**Energy Profile:** HVAC and ventilation dominant; 24/7 areas with intermittent zones

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 8 |
| Primary quick win categories | HVAC scheduling (non-24/7 zones), lighting controls, water heating, controls/BMS |
| Secondary categories | Building envelope, plug loads, power quality |
| Constraints | Minimum ventilation rates non-negotiable; patient comfort paramount |
| Typical total savings | 8-15% of total energy cost |

### 8.6 Preset 6: Education

**Facility Type:** School, university
**Energy Profile:** Heating dominated; intermittent occupancy

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 8 (behavioral strongly emphasized for student engagement) |
| Primary quick win categories | HVAC scheduling (term/holiday), lighting, behavioral change, controls/BMS |
| Secondary categories | Building envelope, plug loads, water heating |
| Typical total savings | 15-30% of total energy cost |

### 8.7 Preset 7: Data Center

**Facility Type:** Colocation, enterprise, hyperscale
**Energy Profile:** Cooling dominated; PUE optimization focus

| Parameter | Value |
|-----------|-------|
| Engines enabled | 6 (exclude behavioral, exclude steam) |
| Primary quick win categories | HVAC setpoints (raise cold aisle temp), controls/BMS (airflow optimization), lighting |
| Secondary categories | Power quality, plug loads (blanking panels as "equipment"), building envelope |
| Constraints | IT uptime and thermal envelope per ASHRAE A1 class |
| Key metrics | PUE improvement, kW cooling saved, EUR/yr |
| Typical total savings | 5-15% of total energy cost |

### 8.8 Preset 8: SME Simplified

**Facility Type:** Any sector, <250 employees
**Energy Profile:** Simplified analysis for facilities with limited data

| Parameter | Value |
|-----------|-------|
| Engines enabled | 5 (scanner, savings, payback, carbon, reporting) |
| Behavioral change engine | Disabled (optional manual enable) |
| Utility rebate engine | Disabled (optional manual enable) |
| Data requirements | Utility bills (12 months minimum), basic equipment inventory, operating hours |
| Analysis approach | Checklist-based scanning with pre-populated savings ranges |
| Typical total savings | 15-35% of total energy cost |

---

## 9. Agent Dependencies

### 9.1 MRV Agents (30)

All 30 AGENT-MRV agents are available via `mrv_bridge.py`, with primary relevance for:
- **MRV-001 Stationary Combustion**: Scope 1 emissions from fuel savings (boiler tuning, insulation)
- **MRV-002 Refrigerants & F-Gas**: Refrigerant emissions from refrigeration quick wins
- **MRV-009 Scope 2 Location-Based**: Grid electricity emission reductions from all electrical quick wins
- **MRV-010 Scope 2 Market-Based**: Electricity emissions with RE certificates and contractual instruments
- **MRV-016 Fuel & Energy Activities (Cat 3)**: Upstream energy emissions from fuel and electricity savings

### 9.2 Data Agents (20)

All 20 AGENT-DATA agents via `data_bridge.py`, with primary relevance for:
- **DATA-001 PDF Extractor**: Utility bill and equipment specification PDF extraction
- **DATA-002 Excel/CSV Normalizer**: Utility bill and meter data import
- **DATA-003 ERP/Finance Connector**: Energy procurement and cost data
- **DATA-010 Data Quality Profiler**: Input data completeness and accuracy assessment
- **DATA-014 Time Series Gap Filler**: Meter data gap filling for savings estimation
- **DATA-015 Cross-Source Reconciliation**: Reconciling meter data vs. utility bills

### 9.3 Foundation Agents (10)

All 10 AGENT-FOUND agents for orchestration, schema validation, unit normalization (kWh, GJ, therms, BTU), assumptions registry, citations, access control, and provenance tracking.

### 9.4 Pack Dependencies

- **PACK-031 Industrial Energy Audit**: Equipment efficiency data, process maps, baselines (optional, enhances manufacturing preset)
- **PACK-032 Building Energy Assessment**: Envelope data, HVAC assessments, EPC ratings (optional, enhances building presets)
- **PACK-021/022/023 Net Zero Packs**: SBTi targets for carbon alignment scoring (optional)

### 9.5 Application Dependencies

- **GL-GHG-APP**: GHG inventory for emission factor sourcing and reduction reporting
- **GL-CSRD-APP**: ESRS E1 energy consumption and efficiency disclosure
- **GL-CDP-APP**: CDP C4 targets and C7 energy breakdown
- **GL-ISO14064-APP**: GHG quantification for verified emission reductions

---

## 10. Performance Targets

| Metric | Target |
|--------|--------|
| Full facility quick wins scan (80+ actions, 12 months data) | <10 minutes |
| Savings estimation per measure | <5 seconds |
| Savings estimation batch (50 measures with interactions) | <3 minutes |
| Financial analysis per measure (NPV, IRR, payback, LCOE) | <100 milliseconds |
| Financial analysis portfolio (100 measures) | <5 seconds |
| Carbon reduction calculation (50 measures, 3 scopes) | <30 seconds |
| MCDA prioritization and Pareto frontier (100 measures) | <30 seconds |
| Behavioral change adoption curve modeling (40 actions) | <10 seconds |
| Utility rebate matching (50 measures, 100 programs) | <1 minute |
| Full report generation (all 8 templates) | <5 minutes |
| Full assessment workflow (end-to-end) | <4 hours |
| Real-time dashboard refresh | <3 seconds |
| Memory ceiling | 2048 MB |
| Cache hit target | 70% |
| Max facilities | 1,000 |
| Max measures per facility | 200 |
| Max concurrent assessments | 50 |

---

## 11. Security Requirements

- JWT RS256 authentication
- RBAC with 5 roles: `energy_manager`, `facility_engineer`, `sustainability_officer`, `external_consultant`, `admin`
- Facility-level access control (users see only facilities assigned to them)
- AES-256-GCM encryption at rest for all energy data, financial analysis, and assessment results
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all calculation outputs (savings estimates, financial analysis, carbon reduction, prioritization scores)
- Full audit trail per SEC-005 (who changed what, when, with provenance chain)
- BMS credentials encrypted via Vault (SEC-006)
- Utility account credentials and API keys encrypted via Vault
- Read-only mode for external consultants (no data modification, no deletion)
- Data retention: minimum 5 years for quick win assessment records, 8 years for financial and compliance records

**RBAC Permission Matrix:**

| Permission | energy_manager | facility_engineer | sustainability_officer | external_consultant | admin |
|------------|---------------|-------------------|----------------------|--------------------|----- |
| Create/edit facility | Yes | No | No | No | Yes |
| Upload data | Yes | Yes | No | No | Yes |
| Run quick win scan | Yes | Yes | No | No | Yes |
| View scan results | Yes | Yes | Yes | Yes (assigned) | Yes |
| Approve measures | Yes | No | No | No | Yes |
| Run financial analysis | Yes | Yes | Yes | Yes (assigned) | Yes |
| Generate reports | Yes | Yes | Yes | Yes (assigned) | Yes |
| Export data | Yes | Yes | Yes | Yes (assigned) | Yes |
| Configure rebate programs | Yes | No | No | No | Yes |
| Manage users | No | No | No | No | Yes |
| View all facilities | No | No | Yes | No | Yes |
| Delete records | No | No | No | No | Yes |

---

## 12. Database Migrations

Inherits platform migrations V001-V245. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V246__pack033_quick_wins_001 | `qwi_facilities`, `qwi_facility_profiles`, `qwi_scan_configurations`, `qwi_quick_win_criteria` | Facility registry, building type profiles, scan configuration parameters, and quick win threshold criteria (max payback, max cost, max disruption) |
| V247__pack033_scan_results_002 | `qwi_scan_runs`, `qwi_scan_results`, `qwi_measure_applicability`, `qwi_action_library` | Quick win scan execution records, per-measure scan results with applicability scores, measure-to-action applicability assessments, and pre-defined action library (80+ actions across 15 categories) |
| V248__pack033_savings_003 | `qwi_savings_estimates`, `qwi_interactive_effects`, `qwi_uncertainty_analysis`, `qwi_normalization_factors` | Per-measure energy savings estimates with confidence intervals, interactive effect calculations between measures, ASHRAE 14-2014 uncertainty analysis, and HDD/CDD normalization factors |
| V249__pack033_financial_004 | `qwi_financial_analysis`, `qwi_cash_flows`, `qwi_sensitivity_results`, `qwi_tax_incentives` | Financial analysis per measure (NPV, IRR, payback, ROI, LCOE), annual cash flow projections, sensitivity analysis results, and tax incentive/depreciation configuration |
| V250__pack033_carbon_005 | `qwi_carbon_reductions`, `qwi_emission_factors`, `qwi_sbti_alignment`, `qwi_scope_breakdown` | Per-measure tCO2e reductions by scope, emission factor registry with source citations, SBTi target alignment assessments, and Scope 1/2/3 breakdown detail |
| V251__pack033_prioritization_006 | `qwi_mcda_configurations`, `qwi_mcda_scores`, `qwi_pareto_frontiers`, `qwi_dependency_graphs`, `qwi_implementation_sequences` | MCDA criteria weights configuration, per-measure multi-criteria scores, Pareto frontier members, measure dependency relationships, and optimized implementation sequences |
| V252__pack033_behavioral_007 | `qwi_behavioral_campaigns`, `qwi_behavioral_actions`, `qwi_adoption_tracking`, `qwi_gamification_scores`, `qwi_persistence_records` | Behavioral change campaign records, 40+ behavioral action definitions, adoption rate tracking by Rogers segment, gamification scores and badges, and persistence measurement records |
| V253__pack033_rebates_008 | `qwi_utility_programs`, `qwi_rebate_matches`, `qwi_rebate_applications`, `qwi_program_updates` | Utility incentive program database (100+ programs), measure-to-program matching results, rebate application records with status tracking, and program update/refresh history |
| V254__pack033_implementation_009 | `qwi_implementation_plans`, `qwi_implementation_status`, `qwi_milestones`, `qwi_resource_allocations` | Implementation plan records, per-measure implementation status (identified/approved/procured/installed/verified), milestone tracking, and resource allocation records |
| V255__pack033_verification_010 | `qwi_verification_results`, `qwi_mv_measurements`, `qwi_variance_analysis`, `qwi_savings_degradation` | IPMVP Option A/B verification results, pre/post measurement records, actual vs. projected variance analysis, and savings degradation detection records |

**Table Prefix:** `qwi_` (Quick Wins Identifier)

**Row-Level Security (RLS):**
- All tables have `facility_id` column for facility-level access control
- RLS policies enforce that users can only see data for facilities assigned to their role
- External consultants have read-only access to specifically assigned assessment records
- Admin role bypasses RLS for cross-facility reporting

**Indexes:**
- Composite indexes on `(facility_id, created_at)` for time-series queries
- GIN indexes on JSONB columns for flexible metadata storage (action library attributes, scan results)
- Partial indexes on `status` columns for active-record filtering
- B-tree indexes on `measure_id`, `scan_id`, `campaign_id` for foreign key joins
- Full-text search index on `qwi_action_library.description` for action search

---

## 13. File Structure

```
packs/energy-efficiency/PACK-033-quick-wins-identifier/
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
      office_building.yaml
      manufacturing.yaml
      retail_store.yaml
      warehouse.yaml
      healthcare.yaml
      education.yaml
      data_center.yaml
      sme_simplified.yaml
  engines/
    __init__.py
    quick_wins_scanner_engine.py
    payback_calculator_engine.py
    energy_savings_estimator_engine.py
    carbon_reduction_engine.py
    implementation_prioritizer_engine.py
    behavioral_change_engine.py
    utility_rebate_engine.py
    quick_wins_reporting_engine.py
  workflows/
    __init__.py
    facility_scan_workflow.py
    prioritization_workflow.py
    implementation_planning_workflow.py
    progress_tracking_workflow.py
    reporting_workflow.py
    full_assessment_workflow.py
  templates/
    __init__.py
    quick_wins_scan_report.py
    prioritized_actions_report.py
    payback_analysis_report.py
    carbon_reduction_report.py
    implementation_plan_report.py
    progress_dashboard.py
    executive_summary_report.py
    rebate_opportunities_report.py
  integrations/
    __init__.py
    pack_orchestrator.py
    mrv_bridge.py
    data_bridge.py
    pack031_bridge.py
    pack032_bridge.py
    utility_rebate_bridge.py
    bms_data_bridge.py
    weather_bridge.py
    health_check.py
    setup_wizard.py
    alert_bridge.py
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_quick_wins_scanner_engine.py
    test_payback_calculator_engine.py
    test_energy_savings_estimator_engine.py
    test_carbon_reduction_engine.py
    test_implementation_prioritizer_engine.py
    test_behavioral_change_engine.py
    test_utility_rebate_engine.py
    test_quick_wins_reporting_engine.py
    test_workflows.py
    test_templates.py
    test_integrations.py
    test_presets.py
    test_e2e.py
    test_orchestrator.py
```

---

## 14. Testing Requirements

| Test Type | Coverage Target | Scope |
|-----------|-----------------|-------|
| Unit Tests | >90% line coverage | All 8 engines, all config models, all presets |
| Workflow Tests | >85% | All 6 workflows with synthetic facility data |
| Template Tests | 100% | All 8 templates in 3+ formats (MD, HTML, JSON, PDF where applicable) |
| Integration Tests | >80% | All 11 integrations with mock agents, BMS simulators, and utility program data |
| E2E Tests | Core happy path | Full pipeline from facility setup to prioritized quick wins report |
| Scanner Tests | 100% | All 80+ actions in action library with applicability rules for all 8 presets |
| Financial Tests | 100% | NPV, IRR, payback, ROI, LCOE calculations with financial calculator cross-validation; Decimal precision verification |
| Savings Tests | >90% | ASHRAE 14-2014 uncertainty bands, interactive effects, HDD/CDD normalization, rebound corrections |
| Carbon Tests | 100% | Scope 1/2/3 emission factor application, location vs. market-based, SBTi alignment |
| Prioritization Tests | 100% | MCDA scoring, Pareto frontier computation, dependency graph resolution, sequencing |
| Behavioral Tests | >85% | Rogers diffusion curves, persistence factors, gamification scoring |
| Rebate Tests | >90% | Program matching, eligibility checking, stacking rules, deadline validation |
| Preset Tests | 100% | All 8 facility-type presets with representative scenarios |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

**Test Count Target:** 600+ tests (50-70 per engine, 30-40 integration, 20-30 E2E)

**Known-Value Validation Sets:**
- 50 NPV/IRR/payback calculations validated against Excel financial functions
- 30 savings estimation calculations validated against ASHRAE handbook engineering methods
- 25 carbon reduction calculations validated against GHG Protocol worked examples
- 20 MCDA scoring scenarios validated against hand calculations
- 15 Rogers diffusion curve fits validated against published adoption data
- 10 utility rebate matching scenarios validated against actual utility program documentation
- 30 interactive effect calculations validated against energy simulation (EnergyPlus reference cases)

---

## 15. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-21 |
| Phase 2 | Engine implementation (8 engines) | 2026-03-22 to 2026-03-24 |
| Phase 3 | Workflow implementation (6 workflows) | 2026-03-24 to 2026-03-25 |
| Phase 4 | Template implementation (8 templates) | 2026-03-25 to 2026-03-26 |
| Phase 5 | Integration implementation (11 integrations) | 2026-03-26 to 2026-03-27 |
| Phase 6 | Test suite (600+ tests) | 2026-03-27 to 2026-03-29 |
| Phase 7 | Database migrations (V246-V255) | 2026-03-29 |
| Phase 8 | Documentation & Release | 2026-03-30 |

---

## 16. Appendix: Quick Win Action Library Categories

### Category 1: Lighting (QW-LTG-001 through QW-LTG-012)

| Action ID | Description | Typical Savings | Typical Payback |
|-----------|-------------|-----------------|-----------------|
| QW-LTG-001 | Replace T8/T12 fluorescent tubes with LED tubes | 40-60% of fixture energy | 6-18 months |
| QW-LTG-002 | Replace HID (metal halide/HPS) with LED high-bay | 50-70% of fixture energy | 12-24 months |
| QW-LTG-003 | Install occupancy/vacancy sensors (intermittent spaces) | 20-40% per controlled zone | 6-12 months |
| QW-LTG-004 | Install daylight harvesting (perimeter zones) | 15-30% of perimeter lighting | 12-24 months |
| QW-LTG-005 | De-lamp over-illuminated areas | 15-25% per area | 0 months (zero cost) |
| QW-LTG-006 | Replace magnetic ballasts with electronic | 10-20% per fixture | 12-18 months |
| QW-LTG-007 | Install task lighting and reduce ambient levels | 10-20% of area lighting | 6-12 months |
| QW-LTG-008 | Add time-clock scheduling for exterior lighting | 15-25% of exterior lighting | 3-12 months |
| QW-LTG-009 | Clean luminaires and reflectors | 5-10% recovery | 0-1 months |
| QW-LTG-010 | Replace exit signs with LED | 80% per sign | 6-12 months |
| QW-LTG-011 | Install photocell controls for exterior/parking lighting | 10-20% of exterior lighting | 6-18 months |
| QW-LTG-012 | Reduce decorative/display lighting hours | 10-30% of decorative energy | 0 months |

### Category 2: HVAC Scheduling (QW-HVS-001 through QW-HVS-006)

| Action ID | Description | Typical Savings | Typical Payback |
|-----------|-------------|-----------------|-----------------|
| QW-HVS-001 | Optimize HVAC start/stop to match actual occupancy | 10-20% of HVAC | 0-3 months |
| QW-HVS-002 | Implement night setback (heating 15C, cooling off) | 5-15% of heating/cooling | 0-3 months |
| QW-HVS-003 | Implement weekend/holiday shutdown or setback | 5-15% of HVAC | 0-3 months |
| QW-HVS-004 | Enable free cooling/economizer when conditions allow | 10-20% of cooling | 0-6 months |
| QW-HVS-005 | Reduce ventilation to minimum during unoccupied warm-up | 5-10% of HVAC | 0-3 months |
| QW-HVS-006 | Stagger AHU start-up to reduce peak demand | 3-8% demand charge reduction | 0-3 months |

### Category 13: Behavioral Change (QW-BHV-001 through QW-BHV-010)

| Action ID | Description | Typical Savings | Typical Persistence |
|-----------|-------------|-----------------|-------------------|
| QW-BHV-001 | "Switch off" campaign (lights, monitors, equipment) | 3-8% of total energy | 30-50% at 12 months |
| QW-BHV-002 | Thermostat awareness program (+/- 1C acceptance) | 3-5% of HVAC | 50-70% at 12 months |
| QW-BHV-003 | Energy champion network (per floor/department) | 5-10% of managed areas | 60-80% at 12 months |
| QW-BHV-004 | Monthly energy consumption feedback dashboards | 3-7% via Hawthorne effect | 45-65% at 12 months |
| QW-BHV-005 | Inter-department energy competition | 8-15% during competition | 40-60% post-competition |
| QW-BHV-006 | "Last out" equipment checklist | 2-5% of after-hours energy | 50-70% at 12 months |
| QW-BHV-007 | Seasonal energy awareness tips program | 2-4% seasonally | 30-50% at 12 months |
| QW-BHV-008 | New employee energy orientation | 2-3% of new hire areas | 60-80% (embedded in culture) |
| QW-BHV-009 | Management walk-and-talk energy rounds | 3-5% via visible leadership | 50-70% at 12 months |
| QW-BHV-010 | Print-to-digital initiative | 3-5% of print energy + paper cost | 60-80% (digital habit) |

---

## 17. Appendix: Financial Calculation Reference

### NPV Calculation

```
NPV = -CapEx + Sum(CF_t / (1 + r)^t)  for t = 1 to N

Where:
  CapEx = initial capital expenditure (EUR)
  CF_t = net cash flow in year t = energy_savings_t + maintenance_delta_t + rebate_t - opex_t
  energy_savings_t = kWh_savings * rate * (1 + escalation)^t
  r = discount rate (e.g., 0.08 for 8%)
  N = equipment useful life (years)
```

### IRR Calculation (Newton-Raphson)

```
Find r such that NPV(r) = 0

Iteration: r_new = r_old - NPV(r_old) / NPV'(r_old)
Convergence: |NPV(r)| < 0.01 EUR
Max iterations: 100
```

### LCOE Calculation

```
LCOE = Sum(Cost_t / (1 + r)^t) / Sum(kWh_t / (1 + r)^t)  for t = 0 to N

Where:
  Cost_t = CapEx (t=0) + OpEx_t - Rebate (t=0 or t=1)
  kWh_t = annual energy savings in year t (with degradation factor)
  r = discount rate
  N = equipment useful life
```

### Discounted Payback

```
Find smallest T such that:
  Sum(CF_t / (1 + r)^t) >= CapEx  for t = 1 to T

If no T exists within equipment life, discounted payback = "N/A" (project does not recover investment at discount rate)
```

---

## 18. Appendix: ASHRAE 14-2014 Statistical Requirements

| Metric | Monthly Models | Daily Models | Hourly Models |
|--------|---------------|-------------|---------------|
| CV(RMSE) | <= 15-20% | <= 20-25% | <= 25-30% |
| NMBE | +/- 5% | +/- 5-10% | +/- 10% |
| R-squared | >= 0.75 | >= 0.70 | >= 0.65 |
| Minimum data points | N >= 3p + 1 | N >= 3p + 1 | N >= 3p + 1 |

Where:
- CV(RMSE) = Coefficient of Variation of Root Mean Square Error
- NMBE = Normalized Mean Bias Error
- p = number of independent variables
- N = number of data points in baseline period

```
CV(RMSE) = (1/y_bar) * sqrt(Sum((y_i - y_hat_i)^2) / (n - p))

NMBE = Sum(y_i - y_hat_i) / ((n - p) * y_bar)

Where:
  y_i = actual energy in period i
  y_hat_i = predicted energy in period i
  y_bar = mean actual energy
  n = number of data points
  p = number of model parameters
```

---

## 19. Appendix: Rogers Diffusion of Innovation Parameters

| Segment | Population Share | Adoption Timing | Key Characteristics |
|---------|-----------------|-----------------|---------------------|
| Innovators | 2.5% | First to adopt (weeks 1-2) | Venturesome, risk-tolerant, intrinsically motivated |
| Early Adopters | 13.5% | Weeks 2-6 | Opinion leaders, respected by peers, see strategic value |
| Early Majority | 34% | Weeks 6-16 | Deliberate, need evidence of success, follow opinion leaders |
| Late Majority | 34% | Weeks 16-32 | Skeptical, adopt from peer pressure or necessity |
| Laggards | 16% | After week 32 (or never) | Traditional, suspicious of change, adopt only when unavoidable |

**Adoption rate modeling:**

```
cumulative_adoption(t) = M / (1 + exp(-k * (t - t0)))

Where:
  M = maximum adoption (typically 70-90% for workplace measures)
  k = adoption speed (typically 0.1-0.3 per week)
  t0 = inflection point (typically 8-16 weeks)
  t = time in weeks
```

---

## 20. Appendix: Energy Unit Quick Reference

| From | To | Factor |
|------|-----|--------|
| 1 kWh | MJ | 3.6 |
| 1 kWh | BTU | 3,412.14 |
| 1 therm | kWh | 29.31 |
| 1 GJ | kWh | 277.78 |
| 1 m3 natural gas | kWh | ~10.55 (NCV) |
| 1 kg fuel oil | kWh | ~11.86 (NCV) |
| 1 kW demand reduction | EUR/yr | ~EUR 50-150/kW (demand charge dependent) |
