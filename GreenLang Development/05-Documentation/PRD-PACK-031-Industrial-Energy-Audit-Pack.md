# PRD-PACK-031: Industrial Energy Audit Pack

**Pack ID:** PACK-031-industrial-energy-audit
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-20
**Prerequisite:** None (standalone; enhanced with PACK-021/022/023 Net Zero Packs if present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Industrial facilities account for approximately 37% of global final energy consumption and 24% of direct CO2 emissions (IEA World Energy Outlook 2023). The EU Energy Efficiency Directive (EED) 2023/1791 -- the recast directive replacing 2012/27/EU -- imposes binding energy efficiency obligations on EU Member States and mandates energy audits for all non-SME enterprises at least every four years (Article 11). ISO 50001:2018 provides the international standard for energy management systems (EnMS), and the EN 16247 series (Parts 1-5) defines the requirements for energy audits. Despite this robust regulatory and standards framework, industrial facilities face significant operational challenges:

1. **Energy audit complexity and cost**: EN 16247-compliant energy audits require systematic data collection across dozens of energy-consuming systems (motors, pumps, compressors, HVAC, boilers, furnaces, lighting, steam systems, compressed air). A detailed (Type 2) audit of a medium manufacturing plant takes 200-400 person-hours and costs EUR 30,000-80,000 when performed by external consultants. Many facilities conduct only walk-through (Type 1) audits that miss 60-80% of savings opportunities.

2. **Baseline establishment difficulty**: Energy baselines per ISO 50006:2014 require regression analysis against relevant variables (production volume, degree-days, occupancy), normalization for weather and production changes, and identification of Energy Performance Indicators (EnPIs). Most facilities lack the statistical tools and metering infrastructure to establish robust baselines, resulting in baselines that do not accurately represent normal operating conditions.

3. **Equipment-level efficiency blindspots**: Industrial facilities operate thousands of pieces of energy-consuming equipment. Motors alone account for 70% of industrial electricity consumption globally (IEA). Without equipment-level efficiency tracking -- motor loading, pump affinity law calculations, compressor specific power, boiler combustion efficiency -- facilities cannot prioritize the highest-impact savings opportunities.

4. **Compressed air system waste**: Compressed air is often called the "fourth utility" in manufacturing, yet it is the most expensive form of energy delivery (only 10-15% of input electricity reaches the point of use as pneumatic energy). Typical compressed air systems waste 25-30% of generated air through leaks, and most facilities operate at higher pressure than necessary. Specialized compressed air audits are rarely performed despite offering the highest ROI of any energy saving measure.

5. **Steam system inefficiency**: Steam systems in process industries (chemical, food and beverage, pulp and paper, petroleum refining) often operate at 55-65% overall efficiency when best-practice achievable efficiency is 80-85%. Steam trap failure rates of 15-30% are common, condensate return is often incomplete, and insulation degradation goes undetected. Systematic steam system audits require specialized knowledge that most facility engineers lack.

6. **Waste heat recovery underutilization**: Industrial processes reject enormous quantities of thermal energy -- typically 20-50% of input energy is lost as waste heat. Pinch analysis and heat exchanger network design can recover 40-90% of this waste heat, but the analytical complexity and capital investment requirements deter most facilities. Without systematic waste heat identification and economic analysis, these opportunities remain unrealized.

7. **Regulatory compliance burden**: The EED recast (2023/1791) tightens energy audit obligations, introduces mandatory energy management systems for enterprises consuming >85 TJ/year, and requires energy audits to identify waste heat recovery potential. The EU Emissions Trading System (EU ETS) Phase 4 requires energy-intensive installations to demonstrate energy efficiency improvements. The Industrial Emissions Directive (IED) 2010/75/EU mandates Best Available Techniques (BAT) compliance, including energy efficiency BAT-AELs from sector-specific BREF documents. Navigating these overlapping obligations is time-consuming and error-prone.

8. **Measurement and Verification (M&V) gaps**: After implementing energy conservation measures (ECMs), facilities must verify actual savings against projected savings per the International Performance Measurement and Verification Protocol (IPMVP). Most facilities lack M&V plans, resulting in inability to demonstrate savings to management, financiers, or regulators. Without M&V, energy performance contracts and green financing become impossible to substantiate.

9. **Benchmarking data scarcity**: Facilities need to benchmark their energy performance against industry peers using Specific Energy Consumption (SEC), Energy Intensity Indices, and BAT-AEL comparisons. However, reliable benchmarking data is scattered across BREF documents, industry associations, and national energy agencies. No integrated tool exists to perform multi-dimensional energy benchmarking against EU BAT-AEL values.

10. **ISO 50001 implementation complexity**: ISO 50001:2018 certification requires establishing an energy management system with energy policy, planning (energy review, EnPIs, baselines, objectives/targets), support (competence, awareness, communication, documentation), operation (design, procurement, operational control), performance evaluation (monitoring, measurement, analysis, internal audit, management review), and improvement (nonconformity, continual improvement). The standard's Plan-Do-Check-Act cycle demands continuous, systematic energy management that spreadsheets cannot support.

### 1.2 Solution Overview

PACK-031 is the **Industrial Energy Audit Pack** -- the first pack in the new "Energy Efficiency Packs" category. It provides a comprehensive industrial energy audit and management solution purpose-built for EN 16247-compliant energy audits, ISO 50001:2018 energy management systems, EED 2023/1791 compliance, and EU ETS/IED energy efficiency requirements. The pack covers the full energy audit lifecycle: baseline establishment, multi-level energy auditing (walk-through to investment-grade), process energy mapping, equipment-level efficiency analysis, energy savings identification and prioritization, specialized subsystem audits (compressed air, steam, waste heat, lighting/HVAC), energy benchmarking, and regulatory compliance management.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete energy audit and management lifecycle for industrial facilities of all types and sizes.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-031 Industrial Energy Audit Pack |
|-----------|------------------------------|---------------------------------------|
| Time to complete energy audit | 200-400 person-hours (Type 2) | <40 hours (5-10x faster) |
| Audit cost | EUR 30,000-80,000 per facility | EUR 3,000-8,000 per facility (10x reduction) |
| Equipment-level analysis | Sampling-based (10-20% of equipment) | Comprehensive (100% of metered equipment) |
| Savings identification | Qualitative/semi-quantitative | Fully quantified with NPV/IRR/payback for every ECM |
| Compressed air audit | Rarely performed separately | Dedicated engine with leak quantification, specific power analysis |
| Steam system audit | Specialist consultant required | Automated steam trap survey, condensate analysis, boiler efficiency |
| Waste heat recovery | Pinch analysis by specialists only | Automated pinch analysis with heat exchanger sizing and ROI |
| M&V planning | Rarely included in audits | IPMVP Options A-D integrated into every ECM recommendation |
| Benchmarking | Manual lookup of BREF/BAT data | Automated SEC and BAT-AEL benchmarking across 50+ subsectors |
| Regulatory compliance | Manual tracking of EED/ETS/IED | Automated compliance mapping with deadline tracking |
| Audit trail | Paper-based work papers | SHA-256 provenance, full calculation lineage, digital audit trail |
| ISO 50001 support | Separate consulting engagement | Integrated EnMS support with EnPI tracking, management review packages |

### 1.4 Energy Audit Types (EN 16247 Classification)

| Audit Type | EN 16247 Reference | Depth | Typical Duration | Cost | Savings Identified |
|------------|-------------------|-------|------------------|------|--------------------|
| Type 1: Walk-Through | EN 16247-1 (basic) | Low | 1-3 days | EUR 5,000-15,000 | 5-15% of energy cost |
| Type 2: Detailed | EN 16247-1 (standard) | Medium | 2-4 weeks | EUR 30,000-80,000 | 15-30% of energy cost |
| Type 3: Investment-Grade | EN 16247-1 (comprehensive) | High | 4-8 weeks | EUR 80,000-200,000 | 25-40% of energy cost |

### 1.5 Target Users

**Primary:**
- Energy managers at manufacturing plants and process industry facilities
- Facility engineers responsible for energy efficiency improvements
- EHS (Environment, Health, Safety) managers with energy audit obligations under EED
- Plant managers seeking to reduce energy costs and improve competitiveness

**Secondary:**
- Energy auditors conducting EN 16247 audits for clients
- Corporate sustainability teams rolling out energy efficiency programs across multiple sites
- Utility managers at data centers and warehouse operations
- ISO 50001 implementation consultants
- EU ETS compliance officers at energy-intensive installations
- ESCOs (Energy Service Companies) developing energy performance contracts
- CFOs evaluating energy efficiency investment portfolios

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to complete energy audit (Type 2 equivalent) | <40 hours (vs. 200-400 manual) | Time from data intake to final audit report |
| Energy baseline accuracy | R-squared >= 0.85 for regression models | Statistical fit of baseline model to historical data |
| Savings identification accuracy | Within 10% of actual post-implementation savings | Verified against IPMVP M&V results at 12 months |
| Equipment efficiency calculation accuracy | 100% match with manual engineering calculations | Tested against 1,000 known equipment efficiency values |
| Compressed air specific power accuracy | Within 5% of measured values | Validated against flow meter and power measurements |
| Steam system efficiency calculation | Within 3% of combustion analyzer readings | Cross-validated against stack gas analysis |
| Regulatory compliance coverage | 100% of applicable EED/ETS/IED requirements mapped | Automated compliance checklist against regulatory database |
| EN 16247 audit report compliance | 100% of mandatory report sections present | Validated against EN 16247-1 Annex A |
| Customer NPS | >50 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| EU Energy Efficiency Directive (EED) | Directive 2023/1791 (recast) | Core regulatory driver; Article 11 energy audit obligation, Article 12 energy management systems |
| ISO 50001:2018 | Energy management systems -- Requirements with guidance for use | International EnMS standard; Plan-Do-Check-Act energy management framework |
| EN 16247-1:2022 | Energy audits -- Part 1: General requirements | Core energy audit methodology; audit process, reporting, competence |
| EN 16247-2:2022 | Energy audits -- Part 2: Buildings | Building-specific audit requirements (HVAC, lighting, envelope) |
| EN 16247-3:2022 | Energy audits -- Part 3: Processes | Process-specific audit requirements (industrial processes, heat treatment) |
| EN 16247-4:2022 | Energy audits -- Part 4: Transport | Transport-specific audit requirements (fleet, logistics) |
| EN 16247-5:2022 | Energy audits -- Part 5: Competence of energy auditors | Auditor qualification and competence requirements |
| ISO 50006:2014 | Energy baseline and energy performance indicators | EnPI methodology, baseline establishment, normalization |
| ISO 50015:2014 | Measurement and verification of energy performance | M&V methodology for energy management systems |

### 2.2 EU Regulatory Framework

| Regulation | Reference | Pack Relevance |
|------------|-----------|----------------|
| EU Emissions Trading System (EU ETS) | Directive 2003/87/EC (as amended by 2023/959) | Phase 4 energy efficiency requirements for installations; free allocation conditioned on efficiency |
| Industrial Emissions Directive (IED) | Directive 2010/75/EU | BAT and BAT-AEL compliance for energy efficiency; BREF documents |
| Energy Performance of Buildings Directive (EPBD) | Directive 2024/1275 (recast) | Building energy performance for industrial buildings with conditioned spaces |
| Ecodesign for Sustainable Products Regulation | Regulation 2024/1781 | Motor, pump, fan, compressor efficiency requirements (IE classes) |
| EU Energy Labelling Regulation | Regulation 2017/1369 | Energy labelling for applicable equipment categories |
| EU Taxonomy Regulation | Regulation 2020/852 | Climate mitigation substantial contribution criteria for energy efficiency investments |

### 2.3 International Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| IPMVP (International Performance Measurement and Verification Protocol) | EVO IPMVP Core Concepts 2022 | M&V methodology; Options A (Key Parameter), B (All Parameters), C (Whole Facility), D (Calibrated Simulation) |
| ASHRAE Standard 90.1 | Energy Standard for Buildings (2022) | Commercial/industrial building energy efficiency baselines |
| ASHRAE Guideline 14 | Measurement of Energy, Demand, and Water Savings (2014) | M&V statistical requirements (CV(RMSE), NMBE, R-squared) |
| IEC 60034-30-1 | Rotating electrical machines -- Efficiency classes (IE code) | Motor efficiency classes IE1 through IE5 |
| ISO 11011:2013 | Compressed air -- Energy efficiency -- Assessment | Compressed air system audit methodology |
| EN 12953/EN 12952 | Shell/Water-tube boilers | Boiler efficiency calculation and testing standards |
| ISO 50002:2014 | Energy audits -- Requirements with guidance for use | International energy audit standard (complementary to EN 16247) |
| ISO 50004:2020 | Guidance for implementation, maintenance and improvement of an EnMS | ISO 50001 implementation guidance |

### 2.4 BREF Documents (Best Available Techniques Reference)

| BREF | Reference | Pack Relevance |
|------|-----------|----------------|
| Energy Efficiency BREF | JRC (2009, revision pending) | Cross-sector energy efficiency BAT and BAT-AELs |
| Large Combustion Plants BREF | JRC (2017) | Boiler and CHP efficiency BAT-AELs |
| Iron and Steel BREF | JRC (2012) | Steel sector SEC and BAT-AELs |
| Cement, Lime and Magnesium Oxide BREF | JRC (2013) | Cement sector SEC and BAT-AELs |
| Non-Ferrous Metals BREF | JRC (2016) | Aluminium and non-ferrous SEC and BAT-AELs |
| Refining of Mineral Oil and Gas BREF | JRC (2015) | Refinery energy intensity index (EII) benchmarks |
| Production of Pulp, Paper and Board BREF | JRC (2015) | Paper sector SEC and BAT-AELs |
| Food, Drink and Milk BREF | JRC (2019) | Food and beverage SEC and BAT-AELs |
| Common Waste Water/Gas Treatment BREF | JRC (2016) | Wastewater and gas treatment energy BAT |
| Surface Treatment BREF | JRC (2006) | Surface treatment process energy BAT |

### 2.5 Supporting Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015) | Scope 1+2 emissions from energy consumption |
| ISO 14064-1:2018 | Organization GHG quantification | GHG emissions linked to energy use |
| ESRS E1 Climate Change | EU CSRD (2023) | E1-5 energy consumption and mix disclosure |
| CDP Climate Change | CDP (2024) | C7 Energy breakdown, C8 Energy-related emissions |
| TCFD Recommendations | FSB/TCFD (2017) | Metrics and targets for energy efficiency |
| EU Taxonomy Technical Screening Criteria | Delegated Reg. 2021/2139 | Substantial contribution criteria for energy efficiency activities |
| IEA Energy Efficiency Indicators | IEA (annual) | International energy efficiency benchmarking data |
| Eurostat Energy Balance Sheets | Eurostat (annual) | EU Member State energy consumption statistics |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Industrial energy audit calculation engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report, dashboard, and compliance templates |
| Integrations | 12 | Agent, app, data, and system bridges |
| Presets | 8 | Facility-type-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `energy_baseline_engine.py` | Establishes energy consumption baselines per facility, production line, and equipment using multivariate regression analysis, degree-day normalization (HDD/CDD), production volume normalization, and Energy Performance Indicators (EnPIs) per ISO 50006. Supports static, dynamic, and rolling baselines with automatic variable selection, statistical significance testing (R-squared, CV(RMSE), NMBE, t-test, F-test), and baseline adjustment for structural changes. |
| 2 | `energy_audit_engine.py` | Conducts EN 16247-compliant energy audits at three levels: Type 1 (walk-through with checklist-based assessment), Type 2 (detailed with metered data analysis and engineering calculations), Type 3 (investment-grade with calibrated models and financial analysis). Handles automatic audit scheduling per EED Article 11 (4-year cycle), auditor competence tracking per EN 16247-5, and multi-site audit coordination. |
| 3 | `process_energy_mapping_engine.py` | Maps energy flows through industrial processes using Sankey diagram data structures, identifies energy losses at each process step, calculates process-level efficiency (useful output / total input), tracks energy balance per production unit, and identifies the largest loss points. Supports mass and energy balance reconciliation, process integration opportunities, and production-normalized energy metrics. |
| 4 | `equipment_efficiency_engine.py` | Calculates equipment-level efficiency for all major industrial energy consumers: electric motors (IE1-IE5 classification, load factor, oversizing detection), pumps (affinity laws, BEP deviation, system curve analysis), compressors (isentropic/volumetric efficiency, specific power), HVAC systems (COP/EER, part-load performance), boilers (combustion efficiency, stack loss, radiation loss, blowdown loss), furnaces (thermal efficiency, wall loss, opening loss), and steam systems (isentropic efficiency, trap performance, insulation R-value). |
| 5 | `energy_savings_opportunity_engine.py` | Identifies, quantifies, and prioritizes energy conservation measures (ECMs) and energy conservation opportunities (ECOs) using deterministic engineering calculations. For each ECM: calculates annual energy savings (kWh, therms, kg steam), annual cost savings (EUR), implementation cost (CapEx + OpEx), simple payback (years), NPV at configurable discount rate, IRR, lifecycle cost analysis, IPMVP M&V option recommendation, learning curves, and degradation factors. Applies IPMVP protocols for savings verification planning. |
| 6 | `waste_heat_recovery_engine.py` | Identifies and quantifies waste heat recovery opportunities from flue gases, cooling water, compressed air aftercoolers, process exhaust streams, and equipment radiation losses. Performs pinch analysis (composite curves, grand composite curve, minimum hot/cold utility targets), heat exchanger sizing (LMTD method, effectiveness-NTU method), technology selection (economizers, air preheaters, heat recovery steam generators, organic Rankine cycle, heat pumps, thermoelectric generators), and economic analysis (CapEx, annual savings, payback, NPV/IRR). |
| 7 | `compressed_air_engine.py` | Specialized audit engine for compressed air systems: system mapping (compressors, dryers, receivers, distribution, end uses), leak detection and quantification (orifice method, ultrasonic survey extrapolation), pressure profile analysis (generation pressure, distribution losses, point-of-use requirements), compressor performance (specific power kW/m3/min, load/unload cycles, VSD retrofit potential), air receiver sizing (demand buffering, pressure stabilization), artificial demand identification, and heat recovery from compression. |
| 8 | `steam_system_engine.py` | Analyzes steam generation, distribution, and condensate recovery systems: boiler efficiency (direct method via stack analysis, indirect method via loss accounting per EN 12953/ASME PTC 4), steam trap surveys (live/failed/blocked classification, condensate loss quantification), distribution insulation assessment (heat loss per meter, economic insulation thickness), flash steam recovery potential, condensate return optimization (return rate, contamination assessment, polishing requirements), deaerator performance, blowdown heat recovery, and CHP/cogeneration opportunities. |
| 9 | `lighting_hvac_engine.py` | Calculates energy savings from lighting and HVAC upgrades: lighting analysis (installed power density W/m2, luminous efficacy lm/W, LED retrofit savings, daylight harvesting potential, occupancy/vacancy control savings, task/ambient lighting optimization), HVAC analysis (cooling load calculation, heating load calculation, economizer potential, VSD fan/pump retrofit savings, heat recovery ventilation effectiveness, chiller optimization, cooling tower optimization, free cooling hours, demand-controlled ventilation). |
| 10 | `energy_benchmark_engine.py` | Benchmarks facility energy performance against industry peers and regulatory thresholds: Specific Energy Consumption (SEC) calculation (kWh/unit, GJ/tonne, kWh/m2), Energy Intensity Index (EII), comparison against EU BAT-AEL values from BREF documents, comparison against national energy efficiency benchmarks, percentile ranking within subsector, gap-to-best-practice quantification, energy efficiency maturity assessment (1-5 scale covering management, data, technology, behavior, investment), and improvement potential estimation. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `initial_energy_audit_workflow.py` | 5: FacilityRegistration -> DataCollection -> BaselineEstablishment -> AuditExecution -> ReportGeneration | End-to-end initial energy audit from facility setup to EN 16247-compliant report |
| 2 | `continuous_monitoring_workflow.py` | 4: RealTimeDataIngestion -> DeviationDetection -> AlertGeneration -> TrendAnalysis | Ongoing energy performance monitoring with anomaly detection and alerts |
| 3 | `energy_savings_verification_workflow.py` | 4: BaselinePeriod -> Implementation -> PostImplementation -> MVReport | IPMVP-compliant savings verification for implemented ECMs (Options A through D) |
| 4 | `compressed_air_audit_workflow.py` | 4: SystemMapping -> LeakSurvey -> PerformanceTesting -> OptimizationRecommendations | Specialized compressed air system audit with leak quantification and VSD analysis |
| 5 | `steam_system_audit_workflow.py` | 4: BoilerAssessment -> DistributionSurvey -> CondensateAnalysis -> RecoveryOptimization | Comprehensive steam system audit from generation to condensate return |
| 6 | `waste_heat_recovery_workflow.py` | 4: HeatSourceIdentification -> PinchAnalysis -> TechnologySelection -> ROICalculation | Waste heat recovery feasibility analysis with pinch analysis and technology sizing |
| 7 | `regulatory_compliance_workflow.py` | 3: EEDObligationCheck -> AuditScheduling -> ComplianceReporting | EED/ETS/IED regulatory compliance management with deadline tracking |
| 8 | `iso50001_certification_workflow.py` | 4: EnMSGapAnalysis -> EnergyPolicyDevelopment -> EnPITracking -> ManagementReviewPreparation | ISO 50001 certification support from gap analysis to management review |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `energy_audit_report.py` | MD, HTML, PDF, JSON | EN 16247-1 compliant energy audit report with all mandatory sections per Annex A |
| 2 | `energy_baseline_report.py` | MD, HTML, JSON | Energy baseline documentation with regression models, EnPIs, and normalization factors |
| 3 | `energy_savings_verification_report.py` | MD, HTML, PDF, JSON | IPMVP-compliant M&V report with baseline, post-implementation, and verified savings |
| 4 | `energy_management_dashboard.py` | MD, HTML, JSON | Real-time energy management dashboard with EnPI tracking, targets, and alerts |
| 5 | `compressed_air_system_report.py` | MD, HTML, JSON | Compressed air audit report with system map, leak register, and optimization plan |
| 6 | `steam_system_assessment_report.py` | MD, HTML, JSON | Steam system audit report with boiler efficiency, trap survey, and recovery opportunities |
| 7 | `waste_heat_recovery_report.py` | MD, HTML, PDF, JSON | Waste heat recovery feasibility report with pinch analysis and technology recommendations |
| 8 | `equipment_efficiency_report.py` | MD, HTML, JSON | Equipment-level efficiency assessment with motor loading, pump curves, and retrofit recommendations |
| 9 | `regulatory_compliance_summary.py` | MD, HTML, JSON | EED/ETS/IED compliance summary with obligation status, deadlines, and action items |
| 10 | `iso50001_management_review.py` | MD, HTML, PDF, JSON | ISO 50001 management review package with EnPI performance, objectives status, and improvement plan |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline with retry, provenance, conditional subsystem audit phases (compressed air, steam, waste heat based on facility type) |
| 2 | `mrv_energy_bridge.py` | Routes to AGENT-MRV-001 (Stationary Combustion), MRV-009/010 (Scope 2 Location/Market-Based), MRV-016 (Fuel & Energy Activities Cat 3) for GHG emissions linked to energy consumption |
| 3 | `data_energy_bridge.py` | Routes to AGENT-DATA agents for meter data ingestion (DATA-002 Excel/CSV, DATA-003 ERP/Finance), data quality profiling (DATA-010), and time series gap filling (DATA-014) |
| 4 | `eed_compliance_bridge.py` | EU Energy Efficiency Directive 2023/1791 obligation tracking: Article 11 audit scheduling, Article 12 EnMS requirements, Article 26 energy efficiency obligation schemes |
| 5 | `iso50001_bridge.py` | ISO 50001:2018 energy management system integration: EnPI registry, objective/target tracking, internal audit scheduling, management review scheduling, continual improvement register |
| 6 | `bms_scada_bridge.py` | Building Management System (BMS) and SCADA data integration: real-time energy data feeds, equipment status, setpoint data, alarm data via BACnet, Modbus, OPC-UA protocols |
| 7 | `utility_metering_bridge.py` | Smart meter, sub-metering, and AMI (Advanced Metering Infrastructure) data integration: interval data (15-min, hourly), demand data, power quality data, utility bill reconciliation |
| 8 | `equipment_registry_bridge.py` | Asset management and CMMS (Computerized Maintenance Management System) integration: equipment inventory, nameplate data, maintenance records, runtime hours, replacement schedules |
| 9 | `weather_normalization_bridge.py` | Weather data integration for baseline normalization: degree-day data (HDD/CDD from local weather stations), TMY (Typical Meteorological Year) data, wet-bulb temperature for cooling, solar radiation for lighting |
| 10 | `health_check.py` | 22-category system verification covering all 10 engines, 8 workflows, metering connectivity, BMS/SCADA status, and data freshness |
| 11 | `setup_wizard.py` | 8-step guided facility configuration: facility profile, metering infrastructure, equipment inventory, production data, weather station, regulatory obligations, audit history, ISO 50001 status |
| 12 | `eu_ets_bridge.py` | EU Emissions Trading System integration for energy-intensive installations: free allocation benchmarks, carbon cost of energy, emissions intensity monitoring, and ETS compliance energy efficiency reporting |

### 3.6 Presets

| # | Preset | Facility Type | Key Characteristics |
|---|--------|--------------|---------------------|
| 1 | `manufacturing_discrete.yaml` | Discrete Manufacturing | Motor-driven systems dominant (70%+ of electricity), production-line-based energy mapping, batch/continuous production normalization, IE motor class upgrade focus |
| 2 | `process_industry.yaml` | Chemical / Petrochemical / Pharmaceutical | Steam-intensive (40-60% of energy), process heat dominant, waste heat recovery high priority, complex energy balance, hazardous area considerations |
| 3 | `food_beverage.yaml` | Food & Beverage Processing | Refrigeration-heavy (20-40% of electricity), steam for pasteurization/sterilization, compressed air for packaging, seasonal production variation, hygiene constraints on heat recovery |
| 4 | `data_center.yaml` | Data Center | Cooling-dominated (PUE optimization), UPS efficiency, IT load management, free cooling maximization, heat reuse potential, redundancy vs. efficiency trade-off |
| 5 | `warehouse_logistics.yaml` | Warehouse & Logistics | Lighting-dominated, HVAC for temperature-controlled storage, forklift charging, dock door energy loss, large envelope area, occupancy-based controls |
| 6 | `automotive_manufacturing.yaml` | Automotive Manufacturing | Paint shop energy intensity (40%+ of plant energy), compressed air intensive, welding energy, body-in-white process heat, assembly line motor systems |
| 7 | `steel_metals.yaml` | Steel & Metals | Electric arc furnace / blast furnace energy, rolling mill motors, high-temperature process heat, waste heat recovery from slag/flue gas, EU ETS intensive |
| 8 | `sme_industrial.yaml` | SME Industrial (any subsector) | Simplified 6-engine flow (baseline, audit, equipment, savings, lighting/HVAC, benchmark), lower metering density, aggregate-level analysis, EED SME exemption check |

---

## 4. Engine Specifications

### 4.1 Engine 1: Energy Baseline Engine

**Purpose:** Establish energy consumption baselines per facility, production line, and equipment using regression analysis, degree-day normalization, and Energy Performance Indicators (EnPIs) per ISO 50006.

**Baseline Types:**

| Type | Description | Use Case |
|------|-------------|----------|
| Static Baseline | Fixed reference period (typically 12 months) | Initial audit, M&V baseline period |
| Dynamic Baseline | Rolling 12-month window, recalculated monthly | Continuous performance monitoring |
| Adjusted Baseline | Static baseline with normalization adjustments | Weather/production-corrected comparison |
| Segmented Baseline | Separate baselines for distinct operating modes | Facilities with seasonal or shift-based patterns |

**Regression Analysis Features:**

| Feature | Description |
|---------|-------------|
| Single-variable regression | Energy vs. production volume, or energy vs. degree-days |
| Multi-variable regression | Energy vs. production + degree-days + occupancy + working days |
| Change-point models | 3-parameter (heating only), 4-parameter (heating + cooling), 5-parameter (full HVAC) |
| Variable selection | Stepwise regression with AIC/BIC criteria for optimal variable selection |
| Statistical validation | R-squared >= 0.75 required (0.85 target), CV(RMSE) <= 25%, NMBE within +/- 10% per ASHRAE 14 |
| Outlier detection | Studentized residual analysis with automatic outlier flagging (> 2.5 sigma) |
| Autocorrelation check | Durbin-Watson test for time-series autocorrelation |

**Energy Performance Indicators (EnPIs) per ISO 50006:**

| EnPI Type | Formula | Example |
|-----------|---------|---------|
| Simple ratio | Energy / Activity | kWh / tonne product |
| Regression-based | E_predicted = a + b*X1 + c*X2 | kWh = f(tonnes, HDD) |
| Cusum (cumulative sum) | Sum(E_actual - E_baseline) over time | Cumulative savings tracking |
| Energy intensity index | (E_actual / E_baseline_adjusted) * 100 | 95 = 5% improvement |
| Specific Energy Consumption | E / Production_normalized | GJ/tonne_normalized |

**Degree-Day Normalization:**

| Parameter | Description |
|-----------|-------------|
| HDD (Heating Degree Days) | Sum of (T_base - T_avg) for days where T_avg < T_base; default T_base = 15.5C (EU) or 18C (configurable) |
| CDD (Cooling Degree Days) | Sum of (T_avg - T_base) for days where T_avg > T_base; default T_base = 22C (configurable) |
| Balance point | Facility-specific balance point determination via change-point regression |
| Data sources | Weather station data via weather_normalization_bridge, TMY for forecasting |

**Key Models:**
- `BaselineInput` - Facility ID, energy consumption time series (monthly/weekly/daily/15-min), production data, weather data, operating schedule, meter inventory
- `BaselineResult` - Regression model parameters, R-squared, CV(RMSE), NMBE, EnPI values, baseline period, normalization factors, statistical validation results
- `RegressionModel` - Coefficients, p-values, confidence intervals, residual analysis, variable importance
- `EnPIDefinition` - EnPI name, formula, baseline value, current value, target value, improvement percentage
- `BaselineAdjustment` - Adjustment type (weather, production, structural), adjustment factor, justification

**Edge Cases:**
- Insufficient data (< 12 months) -> Use minimum 9 months with seasonal adjustment warning
- Poor regression fit (R-squared < 0.75) -> Flag for manual review, suggest additional variables
- Production data unavailable -> Fall back to degree-day-only model with weather normalization
- Multi-fuel facility -> Separate baselines per energy carrier with primary energy conversion
- Shift-based operations -> Separate weekday/weekend baselines or operating-hours normalization
- Structural change mid-period -> Segmented baseline with change-point detection

**Non-Functional Requirements:**
- Baseline calculation per facility: <30 seconds for 3 years of monthly data
- Baseline calculation per facility: <5 minutes for 3 years of 15-minute interval data
- Statistical significance: all reported metrics include 95% confidence intervals
- Reproducibility: bit-perfect (same input produces same output, SHA-256 verified)

### 4.2 Engine 2: Energy Audit Engine

**Purpose:** Conduct EN 16247-compliant energy audits at three levels (Type 1 walk-through, Type 2 detailed, Type 3 investment-grade).

**EN 16247-1 Audit Process:**

| Phase | EN 16247-1 Reference | Activities |
|-------|---------------------|------------|
| 1. Preliminary Contact | Section 5.2 | Scope definition, facility information, audit objectives, resource requirements |
| 2. Start-Up Meeting | Section 5.3 | Stakeholder introduction, data request list, site access arrangements, schedule |
| 3. Data Collection | Section 5.4 | Energy bills (24+ months), production data, equipment inventories, drawings, metering data |
| 4. Field Work | Section 5.5 | On-site measurements, equipment inspection, process observation, occupant interviews |
| 5. Analysis | Section 5.6 | Energy balance, loss identification, savings calculation, financial analysis |
| 6. Reporting | Section 5.7 | Audit report per Annex A, executive summary, recommendations, implementation plan |

**Audit Scope Configuration:**

| Scope Element | Options |
|---------------|---------|
| Facility boundary | Whole site, specific building(s), specific production line(s), specific system(s) |
| Energy carriers | Electricity, natural gas, LPG, fuel oil, district heat, district cooling, steam, compressed air, biomass, solar thermal, other |
| Time boundary | Current year, rolling 12 months, specific period |
| Depth | Type 1 (walk-through), Type 2 (detailed), Type 3 (investment-grade) |
| Systems in scope | All, or selected from: motors/drives, pumps, fans, compressors, HVAC, lighting, boilers/furnaces, steam, refrigeration, process heat, building envelope, compressed air, water |

**Audit Checklist per System Type:**

| System | Key Checklist Items | Measurements Required |
|--------|--------------------|-----------------------|
| Motors/Drives | Loading, IE class, oversizing, VSD potential, coupling, alignment | Voltage, current, power factor, speed |
| Pumps | BEP deviation, throttling, bypass, impeller trim, VSD potential | Flow, pressure, power |
| Fans | System curve, damper control, inlet guide vanes, VSD potential | Flow, static pressure, power |
| Compressors | Load/unload profile, specific power, VSD, staging, pressure band | Airflow, power, pressure, temperature |
| Boilers | Combustion efficiency, O2/CO/CO2, radiation loss, blowdown, cycling | Stack temp, O2, CO, ambient temp |
| HVAC | COP/EER, economizer, setpoints, scheduling, controls, filtration | Supply/return temp, flow, power, humidity |
| Lighting | W/m2, lux levels, occupancy, daylight, lamp/ballast type, controls | Illuminance, power, operating hours |
| Refrigeration | COP, condensing temp, suction pressure, superheat, subcooling | Temperatures, pressures, power |
| Steam | Trap condition, condensate return, insulation, flash steam, deaerator | Temperature, pressure, flow, trap test |
| Building envelope | U-values, air infiltration, thermal bridges, solar gain, insulation | Thermography, blower door, U-value calc |

**EED Article 11 Compliance:**

| Requirement | Implementation |
|-------------|---------------|
| Audit frequency | Every 4 years for non-SME enterprises (automated scheduling with 6-month advance warning) |
| Auditor independence | External auditor or internal auditor not directly involved in audited activities (tracked in auditor registry) |
| Coverage | Energy audit must cover at least 85% of total energy consumption |
| Report requirements | Must include measured/calculated data, recommendations ranked by cost-effectiveness, life-cycle cost analysis |
| EnMS exemption | Facilities with certified ISO 50001 EnMS are exempt from mandatory audit (exemption tracker) |
| >85 TJ threshold | Enterprises consuming >85 TJ/year must implement an EnMS (Article 12) |

**Key Models:**
- `AuditInput` - Facility profile, scope definition, audit type, energy data, equipment inventory, metering data, production data
- `AuditResult` - EN 16247 compliant audit results, energy balance, loss breakdown, savings opportunities list, financial analysis, compliance status
- `EnergyBalance` - Input energy by carrier, useful energy by end use, losses by category, balance residual
- `SystemAuditResult` - Per-system audit findings, efficiency metrics, savings identified, priority rating
- `AuditSchedule` - Next audit due date, audit history, EED compliance status, ISO 50001 exemption status

### 4.3 Engine 3: Process Energy Mapping Engine

**Purpose:** Map energy flows through industrial processes, identify losses, and calculate process efficiency.

**Sankey Diagram Data Structure:**

| Element | Description |
|---------|-------------|
| Node | Energy source, process step, end use, or loss point |
| Link | Energy flow between nodes with magnitude (kW, kWh/yr, GJ/yr) |
| Layer | Hierarchical level (facility -> area -> process -> equipment) |
| Efficiency | Ratio of useful output to total input at each node |

**Process Energy Analysis:**

| Analysis Type | Description | Output |
|---------------|-------------|--------|
| Mass and energy balance | Conservation-based balance across process boundary | Input = Output + Losses + Accumulation |
| Process efficiency | Useful thermal/mechanical output / total energy input | Percentage (e.g., 65% for a dryer) |
| Energy intensity | Energy per unit of product at process level | kWh/unit, GJ/tonne |
| Loss categorization | Classify losses: flue gas, radiation, convection, conduction, latent, electrical | kW and % by category |
| Process integration | Identify heat exchange opportunities between processes | Pinch temperature, recovery potential |
| Production normalization | Energy per unit product adjusted for product mix and utilization | Normalized SEC |

**Process-Specific Models:**

| Process Type | Key Parameters | Typical Efficiency Range |
|-------------|----------------|-------------------------|
| Drying | Evaporation rate, inlet/outlet moisture, exhaust temp, recirculation | 30-60% |
| Melting | Charge weight, tap temperature, holding time, radiation loss | 25-65% |
| Heat treatment | Furnace type, temperature profile, atmosphere, quench method | 20-50% |
| Distillation | Reflux ratio, column efficiency, reboiler duty, condenser duty | 10-30% (thermodynamic) |
| Evaporation | Multiple-effect configuration, vapor recompression, fouling | 50-80% (with multi-effect) |
| Pasteurization | Temperature/time, regeneration %, CIP energy | 85-95% (with regeneration) |
| Extrusion | Specific energy (kWh/kg), barrel heating/cooling, motor load | 60-85% |
| Injection molding | Cycle time, mold cooling, barrel heating, hydraulic/servo | 40-70% |

**Key Models:**
- `ProcessMappingInput` - Facility process flow diagram, energy meter data, production data, process parameters
- `ProcessMappingResult` - Sankey diagram data structure, per-process efficiency, loss breakdown, integration opportunities
- `SankeyNode` - Node ID, name, type (source/process/end_use/loss), energy value, parent node
- `SankeyLink` - Source node, target node, energy value, energy carrier, loss percentage
- `ProcessEfficiency` - Process ID, useful output, total input, efficiency, loss breakdown by category

### 4.4 Engine 4: Equipment Efficiency Engine

**Purpose:** Calculate equipment-level efficiency for all major industrial energy consumers.

**Motor Efficiency (IEC 60034-30-1):**

| IE Class | Typical Efficiency (4-pole, 11 kW) | Annual Electricity Cost (8000 h, EUR 0.15/kWh) |
|----------|--------------------------------------|------------------------------------------------|
| IE1 (Standard) | 87.6% | EUR 15,068 |
| IE2 (High) | 89.8% | EUR 14,699 |
| IE3 (Premium) | 91.4% | EUR 14,441 |
| IE4 (Super Premium) | 93.0% | EUR 14,194 |
| IE5 (Ultra Premium) | 94.0% | EUR 14,043 |

**Motor Loading Analysis:**

| Load Factor | Status | Action |
|-------------|--------|--------|
| <40% | Severely oversized | Replace with correctly sized motor or add VSD |
| 40-60% | Moderately oversized | Consider downsizing at next replacement or add VSD |
| 60-80% | Slightly oversized | Monitor; VSD if variable load |
| 80-100% | Properly sized | No action; consider VSD if variable load |
| >100% | Overloaded | Replace with larger motor (overheating risk) |

**Motor load calculation:**
```
Load_factor = (P_measured / P_rated) = (sqrt(3) * V * I * PF) / P_rated
Annual_savings = P_rated * (1/eta_old - 1/eta_new) * hours * electricity_cost
Payback = motor_cost / annual_savings
```

**Pump Efficiency (Affinity Laws):**

```
Flow: Q2/Q1 = (N2/N1)
Head: H2/H1 = (N2/N1)^2
Power: P2/P1 = (N2/N1)^3

Wire-to-water efficiency = (Q * H * rho * g) / (P_electrical * 1000)
Where: Q = flow (m3/s), H = head (m), rho = density (kg/m3), g = 9.81 m/s2

VSD savings potential = 1 - (Q_reduced/Q_full)^3 (for systems with pure friction loss)
```

**Pump Analysis Points:**

| Check | Threshold | Action |
|-------|-----------|--------|
| BEP deviation | >20% from BEP | Impeller trim or replacement |
| Throttle valve | >10% pressure drop across valve | VSD retrofit (3-year payback typical) |
| Bypass flow | Any continuous bypass | Eliminate bypass, add VSD |
| Cavitation | NPSH_available < NPSH_required + 0.5m | Raise suction head or reduce speed |
| Seal leakage | >0.5 L/min | Replace mechanical seal |

**Compressor Efficiency:**

| Metric | Formula | Target |
|--------|---------|--------|
| Specific power | kW_input / m3/min_FAD | <6.5 kW/m3/min (7 bar) for screw |
| Isentropic efficiency | W_isentropic / W_actual | >85% for new screw, >70% acceptable |
| Volumetric efficiency | V_actual / V_swept | >90% for screw, monitored for wear |
| Part-load efficiency | kW_input / m3/min_actual at part load | VSD: near-linear; load/unload: degrades rapidly below 50% |

**Boiler Efficiency (Indirect Method per EN 12953):**

```
Boiler_efficiency = 100% - L_flue_gas - L_radiation - L_blowdown - L_unburned - L_other

Where:
L_flue_gas = m_gas * cp_gas * (T_stack - T_ambient) / HHV (typically 5-15%)
L_radiation = f(boiler_rating, insulation_condition) (typically 0.5-3%)
L_blowdown = (blowdown_rate / (1 - blowdown_rate)) * (h_blowdown - h_feedwater) / HHV (typically 0.5-3%)
L_unburned = CO_measured * combustion_factor (typically 0.1-0.5%)
```

**Stack Gas Analysis Targets:**

| Parameter | Natural Gas | Fuel Oil | Coal |
|-----------|------------|----------|------|
| O2 | 2-3% | 3-4% | 4-6% |
| CO2 | 10-11% | 12-14% | 14-16% |
| CO | <50 ppm | <100 ppm | <200 ppm |
| Stack temp | <150C above steam temp | <180C | <200C |
| Excess air | 10-15% | 15-25% | 25-40% |

**Key Models:**
- `EquipmentInput` - Equipment inventory with nameplate data, operating data, measurements
- `EquipmentResult` - Per-equipment efficiency, loading analysis, upgrade recommendations, savings potential
- `MotorAnalysis` - IE class, load factor, efficiency, VSD potential, replacement recommendation
- `PumpAnalysis` - BEP deviation, wire-to-water efficiency, affinity law savings, VSD/trim potential
- `CompressorAnalysis` - Specific power, load profile, VSD retrofit analysis, staging optimization
- `BoilerAnalysis` - Combustion efficiency, loss breakdown, O2 trim potential, blowdown optimization
- `HVACAnalysis` - COP/EER, part-load performance, economizer savings, VSD fan/pump potential

### 4.5 Engine 5: Energy Savings Opportunity Engine

**Purpose:** Identify, quantify, and prioritize energy conservation measures (ECMs) using deterministic engineering calculations and IPMVP protocols.

**ECM Categories:**

| Category | Typical Measures | Savings Range |
|----------|-----------------|---------------|
| Motor systems | IE class upgrade, VSD retrofit, right-sizing, synchronous belts, alignment | 10-40% of motor system energy |
| Pumping systems | VSD retrofit, impeller trim, bypass elimination, system optimization | 15-50% of pumping energy |
| Compressed air | Leak repair, pressure reduction, VSD, receiver sizing, heat recovery | 20-50% of compressed air energy |
| Steam systems | Trap repair, insulation, condensate return, blowdown recovery, economizer | 10-30% of steam energy |
| HVAC | Economizer, VSD, heat recovery, setpoint optimization, controls upgrade | 15-40% of HVAC energy |
| Lighting | LED retrofit, daylight harvesting, occupancy controls, task lighting | 30-70% of lighting energy |
| Process heat | Waste heat recovery, insulation, scheduling, temperature optimization | 10-25% of process heat energy |
| Building envelope | Insulation upgrade, air sealing, cool roof, high-performance glazing | 5-20% of heating/cooling energy |
| Power quality | Power factor correction, harmonic filtering, transformer efficiency | 2-5% of total electricity |
| Controls/automation | BMS optimization, demand response, load shifting, predictive control | 5-15% of total energy |

**Financial Analysis per ECM:**

| Metric | Formula | Decision Threshold |
|--------|---------|-------------------|
| Simple payback | CapEx / Annual_savings | <3 years (quick win), 3-7 years (standard), 7-15 years (strategic) |
| Net Present Value (NPV) | Sum(savings_t / (1+r)^t) - CapEx | NPV > 0 |
| Internal Rate of Return (IRR) | Rate where NPV = 0 | IRR > WACC + risk premium |
| Lifecycle Cost Analysis (LCCA) | NPV of all costs over equipment life | Lowest LCCA option preferred |
| Cost of conserved energy (CCE) | Annualized_CapEx / Annual_energy_saved | CCE < current energy price |
| Benefit-Cost Ratio (BCR) | NPV_benefits / NPV_costs | BCR > 1.0 |

**IPMVP M&V Options:**

| Option | Name | Method | Best For |
|--------|------|--------|----------|
| A | Retrofit Isolation: Key Parameter | Measure key parameter, stipulate others | Single equipment retrofit (motor, VSD) |
| B | Retrofit Isolation: All Parameters | Measure all energy parameters continuously | Equipment with variable load (pump, compressor) |
| C | Whole Facility | Utility meter analysis with regression | Multiple ECMs, whole-building retrofit |
| D | Calibrated Simulation | Energy model calibrated to pre/post data | New construction, major renovation |

**Savings Calculation Framework:**

```
Annual_energy_savings = E_baseline_adjusted - E_post_implementation
Where:
  E_baseline_adjusted = f(baseline_model, current_conditions)  [weather, production normalized]
  E_post_implementation = actual metered energy in post period

Annual_cost_savings = energy_savings * energy_price + demand_savings * demand_charge
Implementation_cost = equipment_cost + installation_cost + commissioning_cost + downtime_cost
Simple_payback = implementation_cost / annual_cost_savings
NPV = -implementation_cost + Sum(annual_cost_savings * (1+escalation)^t / (1+discount_rate)^t, t=1..life)
```

**Degradation and Learning Factors:**

| Factor | Description | Typical Value |
|--------|-------------|---------------|
| Savings degradation | Annual reduction in savings due to equipment wear, drift | 1-3% per year |
| Learning curve | Improved savings in first 1-2 years as operators optimize | +5-10% in year 2 |
| Persistence factor | Fraction of savings maintained after N years without intervention | 70-90% at year 10 |
| Rebound effect | Increased consumption due to reduced cost (take-back) | 5-20% for comfort measures |

**Key Models:**
- `SavingsInput` - Equipment/system data, baseline energy, operating conditions, financial parameters
- `SavingsResult` - Ranked ECM list with energy/cost savings, financial metrics, M&V plan, implementation schedule
- `ECMRecommendation` - ECM description, energy savings (kWh/yr), cost savings (EUR/yr), CapEx, payback, NPV, IRR, M&V option, priority
- `FinancialAnalysis` - NPV, IRR, payback, LCCA, CCE, BCR with sensitivity analysis
- `MVPlan` - M&V option, measurement points, baseline period, post-implementation period, reporting frequency

### 4.6 Engine 6: Waste Heat Recovery Engine

**Purpose:** Identify and quantify waste heat recovery opportunities using pinch analysis and heat exchanger sizing.

**Waste Heat Sources in Industrial Facilities:**

| Source | Typical Temperature | Recovery Potential | Technology |
|--------|--------------------|--------------------|------------|
| Boiler flue gas | 150-250C | 10-20% of boiler input | Economizer, air preheater |
| Furnace exhaust | 300-1000C | 15-40% of furnace input | Recuperator, regenerator |
| Compressor aftercooler | 60-90C | 90%+ of compression heat | Water/air heat exchanger |
| Process cooling water | 30-80C | 50-80% of cooling duty | Heat pump, preheat |
| Refrigeration condenser | 30-50C | 80%+ of condenser heat | Space heating, preheat |
| Kiln/oven exhaust | 200-600C | 20-50% of kiln input | Heat recovery steam generator |
| Engine exhaust (CHP) | 300-500C | 25-35% of fuel input | HRSG, ORC |
| Engine jacket water (CHP) | 80-100C | 15-25% of fuel input | Water-to-water HX |

**Pinch Analysis Methodology:**

| Step | Description | Output |
|------|-------------|--------|
| 1. Stream data extraction | Identify all hot streams (to be cooled) and cold streams (to be heated) with supply/target temps and heat capacity flow rates | Stream table |
| 2. Composite curves | Plot cumulative enthalpy vs. temperature for hot and cold streams | Hot and cold composite curves |
| 3. Pinch point identification | Minimum approach temperature (delta_T_min) where curves are closest | Pinch temperature |
| 4. Minimum utility targets | Minimum hot utility (QH_min) and cold utility (QC_min) | Utility targets (kW) |
| 5. Grand composite curve | Net heat flow vs. temperature intervals | Cascade diagram |
| 6. Heat exchanger network design | Design HX network respecting pinch rules (no cross-pinch transfer) | HX network with duties |
| 7. Economic optimization | Trade off HX area (CapEx) vs. utility savings (OpEx) | Optimal delta_T_min |

**Pinch Analysis Rules:**
- Rule 1: No heat transfer across the pinch
- Rule 2: No external cooling above the pinch
- Rule 3: No external heating below the pinch
- Violation of any rule increases total utility consumption above minimum

**Heat Exchanger Sizing:**

```
LMTD Method:
Q = U * A * LMTD * F
Where:
  Q = heat duty (kW)
  U = overall heat transfer coefficient (W/m2.K)
  A = heat transfer area (m2)
  LMTD = log mean temperature difference (K)
  F = correction factor for non-counterflow (0.75-1.0)

LMTD = (dT1 - dT2) / ln(dT1/dT2)

Effectiveness-NTU Method:
epsilon = Q_actual / Q_max
NTU = U * A / C_min
epsilon = f(NTU, C_min/C_max, flow arrangement)
```

**Typical Overall Heat Transfer Coefficients (U):**

| Fluid Pair | U (W/m2.K) |
|------------|------------|
| Gas to gas | 10-50 |
| Gas to liquid | 20-200 |
| Liquid to liquid | 200-1500 |
| Steam to liquid (condensing) | 1000-4000 |
| Steam to gas | 30-300 |
| Water to water | 800-2500 |

**Waste Heat Recovery Technologies:**

| Technology | Input Temperature | Output | Typical Efficiency | CapEx Range |
|------------|------------------|--------|--------------------|-------------|
| Economizer | 150-300C flue gas | Hot water / feedwater preheat | 80-90% of available heat | EUR 10,000-50,000 |
| Air preheater | 200-400C flue gas | Preheated combustion air | 60-80% | EUR 15,000-80,000 |
| HRSG | 300-600C exhaust | Steam | 70-85% | EUR 200,000-2,000,000 |
| ORC (Organic Rankine Cycle) | 80-350C | Electricity | 10-25% (thermal-to-electric) | EUR 2,000-4,000/kW_e |
| Heat pump (industrial) | 30-80C | 60-120C useful heat | COP 3.0-6.0 | EUR 300-800/kW_th |
| Thermoelectric generator | 200-500C | Electricity | 5-8% | EUR 5,000-15,000/kW_e |
| Absorption chiller | 80-200C | Chilled water | COP 0.7-1.4 | EUR 500-1,500/kW_cooling |
| Regenerative burner | 800-1400C | Combustion air preheat | 50-70% energy recovery | EUR 50,000-300,000 |

**Key Models:**
- `WasteHeatInput` - Hot stream data (temp, flow, Cp), cold stream data, delta_T_min, financial parameters
- `WasteHeatResult` - Pinch analysis results, HX network design, technology recommendations, economic analysis
- `PinchAnalysis` - Composite curves, grand composite curve, pinch temperature, minimum utility targets
- `HeatExchangerDesign` - HX type, duty, LMTD, U, area, cost estimate
- `RecoveryOpportunity` - Source, sink, duty (kW), annual savings (EUR), technology, CapEx, payback

### 4.7 Engine 7: Compressed Air System Engine

**Purpose:** Specialized audit engine for compressed air systems covering leak detection, pressure optimization, VSD retrofit, and system efficiency.

**System Components Mapped:**

| Component | Key Parameters | Audit Focus |
|-----------|---------------|-------------|
| Compressors (screw, reciprocating, centrifugal) | Rating (kW), FAD (m3/min), operating pressure (bar), control mode (load/unload, modulation, VSD) | Specific power, part-load efficiency, staging |
| Dryers (refrigerated, desiccant, membrane) | Pressure dewpoint, purge air consumption, energy consumption | Oversizing, type selection, regeneration energy |
| Receivers (primary, secondary) | Volume (L), operating pressure range | Demand buffering, pressure stability, VSD optimization |
| Distribution (headers, branches, drops) | Pipe diameter, length, material, pressure drop | Pressure loss, undersizing, dead legs |
| End uses (tools, actuators, blow-off, agitation) | Pressure requirement, flow demand, duty cycle | Artificial demand, inappropriate use, pressure optimization |

**Leak Detection and Quantification:**

```
Leak flow rate (orifice method):
Q_leak = C_d * A * sqrt(2 * P / rho)  [for unchoked flow]
Q_leak = C_d * A * sqrt(gamma * P * rho * (2/(gamma+1))^((gamma+1)/(gamma-1)))  [for choked flow, typical]

Simplified leak estimation (ultrasonic survey):
Q_leak_total = N_leaks * Q_avg_per_leak * load_factor
Where:
  N_leaks = number of leaks detected
  Q_avg_per_leak = average flow per leak (classified: small 0.4 L/s, medium 1.5 L/s, large 5.0 L/s at 7 bar)
  load_factor = fraction of time system is pressurized

Annual cost of leaks:
Cost_leaks = Q_leak_total * specific_power * hours * electricity_cost / compressor_efficiency
```

**Leak Classification:**

| Size | Orifice Equivalent | Flow at 7 bar | Annual Cost (EUR 0.15/kWh, 6000h) | Ultrasonic dB Level |
|------|-------------------|---------------|-----------------------------------|---------------------|
| Small | 1-2 mm | 0.2-0.8 L/s | EUR 150-600 | <5 dB above ambient |
| Medium | 3-5 mm | 1.0-3.0 L/s | EUR 750-2,250 | 5-15 dB above ambient |
| Large | 6-10 mm | 4.0-15.0 L/s | EUR 3,000-11,250 | >15 dB above ambient |

**Pressure Optimization Analysis:**

```
Every 1 bar reduction in system pressure saves approximately 7% of compressor energy input.

Optimal system pressure = max(point_of_use_requirements) + distribution_loss + filter_loss + dryer_loss + safety_margin

Artificial demand reduction from pressure reduction:
Q_artificial = V_system * (P_high - P_optimal) / P_atmospheric
Where: V_system = total system volume (receivers + piping)
```

**VSD Compressor Analysis:**

| Load Profile | VSD Savings vs. Load/Unload | VSD Savings vs. Modulation |
|-------------|---------------------------|---------------------------|
| 40-60% average load | 25-35% energy savings | 15-25% energy savings |
| 60-80% average load | 15-25% energy savings | 10-15% energy savings |
| 80-100% average load | 5-15% energy savings | 3-8% energy savings |
| Steady 100% load | VSD less efficient (2-3% loss) | VSD less efficient |

**Key Compressed Air KPIs:**

| KPI | Formula | Target |
|-----|---------|--------|
| Specific power | kW_input / (m3/min FAD) | <6.5 kW/m3/min at 7 bar (screw) |
| Leak rate | Q_leak / Q_total * 100% | <10% (good), <5% (excellent) |
| Pressure drop | P_compressor - P_point_of_use | <1.0 bar (entire system) |
| Artificial demand | Q at P_high - Q at P_optimal | Minimize to zero |
| System efficiency | Useful pneumatic work / electrical input | >20% (typical is 10-15%) |

**Key Models:**
- `CompressedAirInput` - Compressor inventory, distribution layout, end-use inventory, metering data, leak survey data
- `CompressedAirResult` - System map, leak register, pressure profile, specific power analysis, optimization recommendations
- `LeakSurvey` - Leak register with location, size classification, estimated flow, repair cost, annual savings
- `PressureAnalysis` - Generation pressure, distribution losses, point-of-use requirements, optimal pressure, artificial demand
- `CompressorOptimization` - Current vs. optimal staging, VSD retrofit analysis, receiver sizing, heat recovery potential

### 4.8 Engine 8: Steam System Optimization Engine

**Purpose:** Analyze steam generation, distribution, and condensate recovery for efficiency optimization.

**Steam System Components:**

| Component | Key Parameters | Audit Focus |
|-----------|---------------|-------------|
| Boilers | Rating (tonnes/hr), pressure (bar), fuel type, turndown, controls | Combustion efficiency, O2 control, cycling losses |
| Economizers | Feedwater temperature rise, approach temp | Heat recovery from flue gas |
| Deaerators | Operating pressure, O2 removal, vent rate | Vent steam recovery, performance |
| Steam headers | Pressure levels (HP/MP/LP), flow, pressure drop | Pressure optimization, header losses |
| PRV stations | Inlet/outlet pressure, flow | Replace with backpressure turbine |
| Steam traps | Type, size, application, condition | Failed traps (stuck open/closed) |
| Condensate return | Return rate, temperature, contamination | Maximize return, flash steam recovery |
| Insulation | Pipe diameter, insulation type/thickness, condition | Missing/damaged insulation, economic thickness |

**Boiler Efficiency (Direct Method):**

```
Efficiency_direct = (m_steam * (h_steam - h_feedwater)) / (m_fuel * HHV) * 100%

Where:
  m_steam = steam mass flow (kg/hr)
  h_steam = specific enthalpy of steam at outlet conditions (kJ/kg)
  h_feedwater = specific enthalpy of feedwater (kJ/kg)
  m_fuel = fuel mass flow (kg/hr or m3/hr)
  HHV = higher heating value of fuel (kJ/kg or kJ/m3)
```

**Boiler Efficiency (Indirect/Loss Method per EN 12953):**

| Loss Category | Typical % (Gas) | Typical % (Oil) | Calculation Method |
|---------------|-----------------|-----------------|-------------------|
| Dry flue gas loss | 4-8% | 5-9% | L1 = m_gas * Cp_gas * (T_stack - T_amb) / HHV |
| Moisture in fuel | 0% (gas) | 0.5% | L2 = m_H2O_fuel * (h_steam_100C - h_water_amb) / HHV |
| Moisture from H2 combustion | 10-11% (gas) | 6-7% | L3 = 9*H2_fraction * m_fuel * (h_steam_Tstack - h_water_amb) / HHV |
| Radiation and convection | 0.5-2% | 0.5-2% | L4 = f(boiler_rating, insulation) per ASME PTC 4 |
| Blowdown loss | 0.5-3% | 0.5-3% | L5 = m_blowdown * (h_blowdown - h_makeup) / (m_fuel * HHV) |
| Unburned combustibles | 0-0.5% | 0-0.5% | L6 = CO_ppm * 12.64 / CO2_% (approximate) |
| Total losses (typical) | 15-22% | 13-20% | Sum of L1 through L6 |
| Net efficiency | 78-85% | 80-87% | 100% - total losses |

**Steam Trap Survey:**

| Trap Condition | Description | Action | Typical Failure Rate |
|---------------|-------------|--------|---------------------|
| Live (passing) | Operating correctly, passing condensate, blocking steam | None | N/A |
| Failed open (blowing) | Stuck open, passing live steam continuously | Replace immediately | 15-30% of installed traps |
| Failed closed (blocked) | Stuck closed, not passing condensate | Replace (causes waterlogging) | 5-10% of installed traps |
| Leaking (passing steam) | Partially failed, passing some live steam | Schedule replacement | 5-15% of installed traps |

**Steam Trap Loss Calculation:**

```
For a failed-open trap:
Steam_loss = C_d * A_orifice * sqrt(2 * rho_steam * delta_P) * 3600  [kg/hr]
Annual_loss_cost = steam_loss * operating_hours * steam_cost_per_kg

Simplified (Spirax Sarco method):
Steam_loss (kg/hr) = orifice_factor * sqrt(P_upstream_bar) * orifice_area_mm2 / 25.4
Where orifice_factor depends on trap type and failure mode

Rule of thumb: A failed 15mm trap at 7 bar loses approximately 25-50 kg/hr of steam
Annual cost: 25 kg/hr * 8000 hr * EUR 0.035/kg = EUR 7,000 per failed trap
```

**Condensate Return Optimization:**

| Parameter | Current Typical | Best Practice | Savings |
|-----------|-----------------|---------------|---------|
| Condensate return rate | 50-70% | 85-95% | 10-20% of steam cost |
| Condensate temperature | 70-90C | 90-95C | Reduced feedwater heating |
| Flash steam recovery | 0% (vented) | 80-90% captured | 5-10% of LP steam demand |
| Make-up water treatment | Full treatment | Reduced proportionally | Water + chemical savings |

**Key Models:**
- `SteamSystemInput` - Boiler inventory, distribution layout, trap inventory, condensate system, metering data
- `SteamSystemResult` - Boiler efficiency, trap survey results, condensate analysis, insulation assessment, optimization plan
- `BoilerEfficiency` - Direct and indirect method results, loss breakdown, O2 trim recommendation, blowdown optimization
- `TrapSurveyResult` - Per-trap condition (live/failed_open/failed_closed/leaking), steam loss, annual cost, priority
- `CondensateAnalysis` - Return rate, temperature, flash steam potential, contamination risk, recovery optimization

### 4.9 Engine 9: Lighting & HVAC Optimization Engine

**Purpose:** Calculate energy savings from lighting upgrades and HVAC optimization.

**Lighting Analysis:**

| Metric | Formula | EN 12464-1 Requirement |
|--------|---------|----------------------|
| Installed power density | W_installed / Area_m2 | Varies by space type (e.g., 8-12 W/m2 for offices) |
| Luminous efficacy | lm / W | LED: 120-180 lm/W; Fluorescent: 80-100 lm/W; HID: 70-120 lm/W |
| Maintained illuminance | E_m (lux) on task plane | Office: 500 lux; Warehouse: 100-200 lux; Manufacturing: 300-750 lux |
| Lighting Energy Numeric Indicator (LENI) | kWh/m2/yr | EN 15193-1 calculation method |
| Uniformity ratio | E_min / E_avg | >= 0.6 for task area, >= 0.4 for surrounding |

**Lighting Savings Calculations:**

```
LED Retrofit Savings:
Savings_kWh = (P_existing - P_LED) * hours * quantity * (1 - controls_factor)
Where:
  P_existing = existing lamp + ballast power (W)
  P_LED = replacement LED power (W)
  hours = annual operating hours
  controls_factor = additional savings from controls (0 to 0.5)

Daylight Harvesting Savings:
Savings_daylight = P_installed * hours_daylight * daylight_factor * area_daylight / area_total
Where:
  daylight_factor = 0.3-0.7 depending on glazing ratio and orientation

Occupancy Control Savings:
Savings_occupancy = P_installed * hours * (1 - occupancy_factor)
Where:
  occupancy_factor = fraction of time space is occupied (typically 0.5-0.8 for offices)
```

**HVAC Analysis:**

| System | Key Parameters | Savings Opportunities |
|--------|---------------|----------------------|
| Chillers | COP/EER, part-load curve (IPLV/NPLV), condenser approach | Staging optimization, condenser cleaning, free cooling, VSD |
| Boilers (HVAC) | Seasonal efficiency, modulation range, reset schedule | Condensing boiler, outdoor reset, staging |
| AHU/RTU | Supply air temperature, economizer, fan power, filtration | Economizer repair/add, VSD, demand-controlled ventilation |
| Fans | Operating point, VSD, inlet guide vanes, dampers | VSD retrofit, duct sealing, system optimization |
| Pumps (HVAC) | Delta-T, VSD, 3-way valves, balancing | VSD, 2-way valve conversion, balancing |
| Cooling towers | Approach temperature, fan staging, VSD, water treatment | VSD, fill replacement, optimize cycles |
| Heat recovery | Effectiveness, leakage, cleaning schedule | Upgrade HRV, add runaround coil |

**HVAC Savings Calculations:**

```
Economizer Savings:
Free_cooling_hours = hours where T_outdoor < T_return - dT_min
Savings_kWh = cooling_load_avg * free_cooling_hours / COP

VSD Fan Savings (Affinity Laws):
Savings_kWh = P_fan_rated * hours * (1 - (Q_avg/Q_design)^3)
Note: Cubic relationship means 20% flow reduction = 49% power reduction

Heat Recovery Ventilation Savings:
Q_recovered = effectiveness * m_air * Cp_air * (T_exhaust - T_supply)
Savings_kWh = Q_recovered * operating_hours / (eta_heating or COP_cooling)

Demand-Controlled Ventilation Savings:
Savings_kWh = (1 - occupancy_fraction) * Q_ventilation * Cp_air * dT * hours / eta_system
```

**Key Models:**
- `LightingHVACInput` - Lighting inventory, HVAC equipment inventory, space data, metering data, schedule data
- `LightingHVACResult` - Lighting analysis with LED/controls recommendations, HVAC analysis with optimization measures
- `LightingAnalysis` - Per-space: installed power density, luminous efficacy, LENI, LED savings, controls savings
- `HVACAnalysis` - Per-system: efficiency, part-load performance, economizer status, VSD potential, savings
- `LightingECM` - LED retrofit, daylight harvesting, occupancy controls with savings and payback per space
- `HVACECM` - VSD, economizer, heat recovery, DCV with savings and payback per system

### 4.10 Engine 10: Energy Benchmark Engine

**Purpose:** Benchmark facility energy performance against industry peers and regulatory thresholds.

**Benchmarking Dimensions:**

| Dimension | Metric | Source |
|-----------|--------|--------|
| Specific Energy Consumption (SEC) | kWh/unit, GJ/tonne, kWh/m2 | Facility production/area data |
| Energy Intensity Index (EII) | Actual SEC / Reference SEC * 100 | Solomon (refining), internal benchmarks |
| BAT-AEL comparison | Facility metric vs. BAT-AEL range | BREF documents (see Section 2.4) |
| National benchmark | Facility vs. national average and best practice | Member State energy agency data |
| Sector percentile | Ranking within subsector peer group | IEA, Eurostat, industry association data |
| Energy efficiency maturity | 1-5 scale across 5 dimensions | GreenLang maturity model |

**EU BAT-AEL Energy Benchmarks (Selected Subsectors):**

| Subsector | BREF | BAT-AEL Range | Unit |
|-----------|------|---------------|------|
| Cement (dry process) | Cement BREF | 2.9-3.3 GJ/tonne clinker | GJ/t |
| Steel (EAF) | Iron & Steel BREF | 350-450 kWh/tonne liquid steel | kWh/t |
| Steel (BF-BOF) | Iron & Steel BREF | 17-21 GJ/tonne hot rolled coil | GJ/t |
| Aluminium (electrolysis) | Non-Ferrous BREF | 13.0-14.5 MWh/tonne aluminium | MWh/t |
| Paper (fine paper) | Pulp & Paper BREF | 4.5-7.0 GJ/tonne | GJ/t |
| Dairy processing | Food BREF | 0.3-0.7 GJ/tonne product | GJ/t |
| Brewery | Food BREF | 0.15-0.25 GJ/hL beer | GJ/hL |
| Glass (container) | Glass BREF | 4.0-7.0 GJ/tonne melted glass | GJ/t |
| Refinery (overall) | Refining BREF | EII 85-100 (Solomon index) | Index |
| LCP (gas boiler) | LCP BREF | 91-95% (NCV efficiency) | % |
| LCP (gas CCGT) | LCP BREF | 57-60.5% (NCV efficiency) | % |

**Energy Efficiency Maturity Model (5 Levels):**

| Level | Management | Data | Technology | Behavior | Investment |
|-------|-----------|------|------------|----------|------------|
| 1 - Initial | No energy policy | Utility bills only | No efficiency standards | No awareness programs | No energy budget |
| 2 - Developing | Energy policy exists | Main meter + bills | Some IE3 motors | Occasional campaigns | Ad hoc project funding |
| 3 - Defined | ISO 50001 planned | Sub-metering (50%+) | IE3 standard, some VSD | Regular training | Annual energy budget |
| 4 - Managed | ISO 50001 certified | Sub-metering (80%+) | IE4/VSD standard, BMS | Continuous engagement | 3-year investment plan |
| 5 - Optimizing | EnMS integrated into business | Real-time 100% sub-metered | IE5/VSD/AI controls | Energy culture embedded | Strategic energy investment |

**Benchmarking Output:**

| Output Element | Description |
|----------------|-------------|
| SEC comparison | Facility SEC vs. BAT-AEL, national, sector best practice |
| Percentile ranking | Where the facility stands in its subsector (quartile, decile) |
| Gap-to-best-practice | Absolute and percentage gap to BAT-AEL or best-in-class |
| Improvement potential | Estimated energy savings if facility achieves BAT-AEL (kWh/yr, EUR/yr) |
| Maturity score | 1-5 maturity level with per-dimension scores and improvement roadmap |
| Trend analysis | Year-over-year SEC trend with target trajectory |

**Key Models:**
- `BenchmarkInput` - Facility energy data, production data, subsector classification (NACE), area, employee count
- `BenchmarkResult` - SEC, EII, BAT-AEL comparison, percentile ranking, gap analysis, maturity score, improvement potential
- `SECCalculation` - Numerator (energy), denominator (production/area), normalization, confidence interval
- `BATComparison` - Facility metric, BAT-AEL lower, BAT-AEL upper, gap, BREF reference
- `MaturityAssessment` - Per-dimension score (1-5), overall score, improvement actions per dimension

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Initial Energy Audit Workflow

**Purpose:** End-to-end initial energy audit from facility registration through EN 16247-compliant audit report delivery.

**Phase 1: Facility Registration**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Create facility profile | Facility name, address, NACE code, floor area, operating hours, production type | Facility record with unique ID | <5 minutes |
| 1.2 | Define organizational boundary | Legal entity, operational/financial control, multi-site hierarchy | Boundary definition document | <10 minutes |
| 1.3 | Register energy carriers | Electricity, gas, oil, LPG, steam, district heat, renewables | Energy carrier inventory | <5 minutes |
| 1.4 | Configure metering infrastructure | Main meters, sub-meters, check meters, virtual meters | Meter registry with hierarchy | <15 minutes |
| 1.5 | Import equipment inventory | Equipment type, nameplate data, location, runtime hours | Equipment registry | <30 minutes |

**Phase 2: Data Collection**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Import utility bills | PDF/Excel utility bills (24+ months) | Standardized energy consumption time series | <15 minutes |
| 2.2 | Import meter data | Interval data (15-min/hourly) from BMS/SCADA/AMI | High-resolution energy time series | <10 minutes |
| 2.3 | Import production data | Production volumes, product mix, operating hours, shifts | Production time series | <10 minutes |
| 2.4 | Import weather data | Temperature, humidity, wind, solar from nearest station | Weather time series (HDD/CDD calculated) | <5 minutes (auto) |
| 2.5 | Data quality assessment | All imported data | Data quality score (0-100), gap report, reconciliation | <5 minutes (auto) |

**Phase 3: Baseline Establishment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Select baseline period | Energy, production, weather data; user preference | Baseline period (typically 12 months) | <5 minutes |
| 3.2 | Variable selection | Candidate variables (production, HDD, CDD, occupancy, etc.) | Selected independent variables with statistical significance | <2 minutes (auto) |
| 3.3 | Regression modeling | Energy vs. selected variables | Regression model with R-squared, CV(RMSE), NMBE | <30 seconds (auto) |
| 3.4 | EnPI definition | Baseline model, production metrics | EnPI definitions with baseline values | <5 minutes |
| 3.5 | Baseline validation | Statistical tests, visual inspection, engineering review | Validated baseline report | <10 minutes |

**Phase 4: Audit Execution**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Energy balance | All energy data, process flow | Facility energy balance (Sankey) | <10 minutes (auto) |
| 4.2 | System-level analysis | Per-system metered data, equipment data | System efficiency assessments | <30 minutes (auto) |
| 4.3 | Equipment-level analysis | Equipment inventory, measurements | Per-equipment efficiency, loading, upgrade potential | <30 minutes (auto) |
| 4.4 | Subsystem audits | Compressed air, steam, waste heat data (if applicable) | Subsystem audit results | <30 minutes (auto) |
| 4.5 | ECM identification | All analysis results, financial parameters | Prioritized ECM list with savings and financials | <15 minutes (auto) |

**Phase 5: Report Generation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 5.1 | Generate EN 16247 report | All audit results | Compliant audit report (PDF, HTML, JSON) | <10 minutes (auto) |
| 5.2 | Executive summary | Top findings, total savings, payback summary | 2-page executive summary | <5 minutes (auto) |
| 5.3 | Implementation plan | Prioritized ECMs, resource requirements | Phased implementation roadmap | <5 minutes (auto) |
| 5.4 | M&V plans | Per-ECM M&V approach | IPMVP-compliant M&V plans | <5 minutes (auto) |
| 5.5 | Quality assurance | Report completeness check vs. EN 16247 Annex A | QA checklist with pass/fail per section | <2 minutes (auto) |

**Acceptance Criteria:**
- [ ] All 5 phases execute sequentially with data passing between phases
- [ ] Phase 2 data quality assessment flags gaps and offers auto-fill options
- [ ] Phase 3 baseline achieves R-squared >= 0.75 or escalates for manual review
- [ ] Phase 4 covers all systems representing >= 85% of total energy per EED
- [ ] Phase 5 report passes EN 16247-1 Annex A completeness check at 100%
- [ ] Total workflow duration < 60 minutes for a typical Type 2 audit (data pre-loaded)
- [ ] Full SHA-256 provenance chain from input data through every calculation to final report

### 5.2 Workflow 2: Continuous Monitoring Workflow

**Purpose:** Ongoing energy performance monitoring with anomaly detection and trend analysis.

**Phase 1: Real-Time Data Ingestion**

| Step | Action | Frequency |
|------|--------|-----------|
| 1.1 | Ingest meter interval data | Every 15 minutes (configurable: 1-min to 1-hour) |
| 1.2 | Ingest BMS/SCADA data | Every 1-5 minutes |
| 1.3 | Ingest production data | Every shift / every hour |
| 1.4 | Ingest weather data | Every hour |
| 1.5 | Data validation and cleansing | On every ingestion cycle |

**Phase 2: Deviation Detection**

| Detection Method | Algorithm | Threshold |
|-----------------|-----------|-----------|
| CUSUM control chart | Cumulative sum of deviations from expected | >3 sigma cumulative deviation |
| EWMA control chart | Exponentially weighted moving average | >2 sigma from weighted mean |
| Regression residual | Actual vs. predicted from baseline model | >2.5 sigma residual |
| EnPI threshold | EnPI value vs. target/limit | >10% deviation from target |
| Equipment anomaly | Equipment power vs. expected range | >15% deviation from nameplate/historical |

**Phase 3: Alert Generation**

| Alert Level | Trigger | Action | Notification |
|-------------|---------|--------|--------------|
| Information | EnPI within 5-10% of target | Log for trend analysis | Dashboard indicator |
| Warning | EnPI 10-20% above target or CUSUM trend detected | Review within 1 week | Email to energy_manager |
| Critical | EnPI >20% above target or sudden step change | Immediate investigation | SMS + email to energy_manager + facility_engineer |
| Emergency | Equipment failure detected or safety threshold exceeded | Immediate response | All channels including escalation to plant_manager |

**Phase 4: Trend Analysis**

| Analysis | Frequency | Output |
|----------|-----------|--------|
| Daily energy profile | Daily | 24-hour load profile with peak/base/off-peak |
| Weekly summary | Weekly | Week-over-week comparison, weekend vs. weekday analysis |
| Monthly EnPI tracking | Monthly | EnPI values vs. target, CUSUM chart, trend direction |
| Quarterly review | Quarterly | Seasonal analysis, year-over-year comparison, projection |
| Annual performance review | Annual | Full-year EnPI performance, savings achieved, targets for next year |

**Acceptance Criteria:**
- [ ] Data ingestion latency < 5 seconds from meter read to database
- [ ] Deviation detection runs within 10 seconds of data ingestion
- [ ] False positive alert rate < 5% (validated against 12 months of historical data)
- [ ] CUSUM sensitivity detects 10% sustained deviation within 5 data points
- [ ] Dashboard refreshes within 3 seconds of new data availability
- [ ] Trend analysis reports auto-generated on schedule without manual intervention

### 5.3 Workflow 3: Energy Savings Verification Workflow

**Purpose:** IPMVP-compliant savings verification for implemented ECMs using Options A through D.

**Phase 1: Baseline Period**

| Step | Action | Duration |
|------|--------|----------|
| 1.1 | Define M&V boundary | 1 hour |
| 1.2 | Select IPMVP option (A, B, C, or D) | 30 minutes |
| 1.3 | Install/verify measurement points | Implementation-dependent |
| 1.4 | Collect baseline data (minimum 3 months, recommended 12 months) | 3-12 months |
| 1.5 | Build baseline model and validate per ASHRAE 14 | 1 hour (auto) |

**Phase 2: Implementation**

| Step | Action |
|------|--------|
| 2.1 | Document ECM implementation details (date, scope, cost) |
| 2.2 | Record any non-routine adjustments during implementation |
| 2.3 | Define post-implementation measurement start date |
| 2.4 | Verify measurement system continuity |

**Phase 3: Post-Implementation**

| Step | Action | Duration |
|------|--------|----------|
| 3.1 | Collect post-implementation data (minimum 3 months, recommended 12 months) | 3-12 months |
| 3.2 | Apply routine adjustments (normalize for weather, production) | Automatic |
| 3.3 | Apply non-routine adjustments (structural changes, additions) | Manual review |
| 3.4 | Calculate adjusted baseline for post conditions | Automatic |
| 3.5 | Calculate verified savings = adjusted_baseline - actual_post | Automatic |

**Phase 4: M&V Report**

| Step | Action |
|------|--------|
| 4.1 | Statistical validation of savings (t-test, confidence interval) |
| 4.2 | Savings uncertainty analysis (fractional savings uncertainty per ASHRAE 14) |
| 4.3 | Comparison of verified savings vs. projected savings |
| 4.4 | Persistence analysis (savings degradation trend) |
| 4.5 | Generate IPMVP-compliant M&V report |

**Savings Uncertainty Calculation (ASHRAE Guideline 14):**

```
Fractional Savings Uncertainty:
FSU = t * sqrt((CV_RMSE * n_post / (n_baseline * F))^2 + (sigma_regression / savings)^2)

Where:
  t = t-statistic for desired confidence level (1.96 for 95%)
  CV_RMSE = coefficient of variation of RMSE from baseline model
  n_post = number of post-implementation data points
  n_baseline = number of baseline data points
  F = savings fraction = savings / baseline_energy
  sigma_regression = uncertainty from regression model

Acceptable FSU: < 50% at 68% confidence for savings > 10% of baseline
```

**Acceptance Criteria:**
- [ ] Supports all 4 IPMVP options (A, B, C, D) with option-specific data requirements
- [ ] Baseline model meets ASHRAE 14 criteria: CV(RMSE) <= 20% (monthly), NMBE +/- 5%
- [ ] Savings calculation includes routine and non-routine adjustments
- [ ] Uncertainty analysis produces fractional savings uncertainty per ASHRAE 14
- [ ] Report includes comparison of verified vs. projected savings with variance explanation
- [ ] All calculation inputs, parameters, and results are SHA-256 hashed for audit trail

### 5.4 Workflow 4: Compressed Air Audit Workflow

**Purpose:** Specialized compressed air system audit from system mapping through optimization recommendations.

**Phase 1: System Mapping**
- Map all compressors with ratings, types, control modes
- Map dryers, receivers, filters, regulators
- Map distribution piping (headers, branches, drops)
- Identify all end uses with pressure and flow requirements
- Calculate system total capacity and average demand

**Phase 2: Leak Survey**
- Ultrasonic leak survey of entire distribution system
- Classify leaks by size (small/medium/large)
- Quantify leak rate using orifice method
- Calculate total leak cost and tag for repair
- Prioritize repairs by cost impact (largest leaks first)

**Phase 3: Performance Testing**
- Measure compressor specific power (kW/m3/min FAD)
- Log load/unload cycles and part-load efficiency
- Measure system pressure at generation, distribution, and point of use
- Calculate pressure drop through dryers, filters, regulators
- Assess artificial demand from excess pressure

**Phase 4: Optimization Recommendations**
- Leak repair program with ROI (typically 3-6 month payback)
- Pressure optimization (1 bar reduction = ~7% savings)
- VSD compressor retrofit analysis
- Receiver sizing optimization (demand buffering)
- Heat recovery from compression (80-93% of input energy)
- Control system upgrade (sequencer, master controller)
- End-use optimization (replace pneumatic with electric where possible)

**Acceptance Criteria:**
- [ ] System map captures 100% of compressors, dryers, and receivers
- [ ] Leak survey quantifies total leak rate to within +/- 20% of actual
- [ ] Specific power calculation accurate to within +/- 5% of measured
- [ ] Pressure optimization savings calculated using ideal gas relationships
- [ ] VSD analysis includes energy model for full operating profile (not just design point)
- [ ] All recommendations include CapEx, annual savings, simple payback, NPV

### 5.5 Workflow 5: Steam System Audit Workflow

**Purpose:** Comprehensive steam system audit from boiler assessment through condensate return optimization.

**Phase 1: Boiler Assessment**
- Combustion efficiency measurement (stack O2, CO, temperature)
- Boiler efficiency calculation (direct and indirect method)
- Excess air assessment and O2 trim recommendation
- Blowdown rate assessment and flash heat recovery potential
- Cycling/short-cycling loss assessment for multi-boiler plants
- Fuel-to-steam cost calculation

**Phase 2: Distribution Survey**
- Steam header pressure optimization (HP/MP/LP staging)
- PRV station assessment (backpressure turbine replacement potential)
- Insulation survey (missing, damaged, wet insulation identification)
- Heat loss calculation per meter of uninsulated/degraded pipe
- Economic insulation thickness calculation
- Distribution pressure drop assessment

**Phase 3: Condensate Analysis**
- Condensate return rate measurement (% of steam generated)
- Condensate temperature at collection points
- Contamination assessment (conductivity, pH, oil)
- Flash steam generation at let-down points
- Condensate polishing requirements assessment
- Make-up water treatment cost baseline

**Phase 4: Recovery Optimization**
- Steam trap survey (ultrasonic, temperature, visual)
- Failed trap quantification (steam loss in kg/hr and EUR/yr)
- Flash steam recovery sizing (flash vessels, secondary LP steam)
- Condensate return piping optimization
- Blowdown heat recovery heat exchanger sizing
- CHP/cogeneration feasibility (if PRV stations with significant flow)

**Acceptance Criteria:**
- [ ] Boiler efficiency calculated by both direct and indirect method with < 3% deviation
- [ ] Steam trap survey classifies 100% of accessible traps as live/failed_open/failed_closed/leaking
- [ ] Insulation survey identifies all bare/damaged sections > 1 meter
- [ ] Condensate return rate calculated to within +/- 5% of actual
- [ ] Flash steam recovery potential quantified using steam tables at actual pressures
- [ ] All optimization recommendations include CapEx, annual savings, payback

### 5.6 Workflow 6: Waste Heat Recovery Workflow

**Purpose:** Waste heat recovery feasibility analysis from heat source identification through technology selection and ROI calculation.

**Phase 1: Heat Source Identification**
- Catalogue all heat rejection points (flue gas, cooling water, exhaust, radiation)
- Measure/estimate temperature and flow rate for each source
- Calculate available heat duty (kW) per source
- Rank sources by temperature grade (high >400C, medium 100-400C, low <100C)
- Identify intermittency and seasonal variation

**Phase 2: Pinch Analysis**
- Define hot streams (to be cooled) and cold streams (to be heated) with supply/target temperatures
- Calculate heat capacity flow rates (Cp * mass_flow)
- Construct composite curves (temperature vs. enthalpy)
- Identify pinch point at specified minimum approach temperature (delta_T_min)
- Calculate minimum hot utility and cold utility targets
- Construct grand composite curve for utility targeting

**Phase 3: Technology Selection**
- Map each recovery opportunity to appropriate technology
- Size heat exchangers using LMTD/effectiveness-NTU method
- Evaluate ORC potential for medium-grade waste heat to power
- Evaluate heat pump upgrade potential for low-grade heat
- Assess practical constraints (space, fouling, corrosion, intermittency)
- Calculate technology-specific CapEx using cost correlations

**Phase 4: ROI Calculation**
- Annual energy savings per recovery opportunity (kWh_th, kWh_e, or both)
- Annual cost savings based on avoided fuel/electricity purchase
- Implementation cost (equipment, installation, piping, controls, commissioning)
- Simple payback, NPV, IRR per opportunity
- Rank opportunities by NPV or IRR
- Sensitivity analysis on energy price, utilization, and discount rate

**Acceptance Criteria:**
- [ ] Pinch analysis minimum utility targets within 5% of theoretical minimum
- [ ] Heat exchanger sizing uses appropriate U values for fluid pairs (validated against TEMA)
- [ ] ORC efficiency estimation within 2% of manufacturer performance curves
- [ ] Heat pump COP estimation within 5% of manufacturer data
- [ ] CapEx estimation within +/- 30% (AACE Class 4 estimate) using cost correlations
- [ ] Sensitivity analysis covers +/- 20% on energy price and utilization factor

### 5.7 Workflow 7: Regulatory Compliance Workflow

**Purpose:** EED/ETS/IED regulatory compliance management with obligation tracking and deadline management.

**Phase 1: EED Obligation Check**
- Determine enterprise size (SME vs. non-SME per EU definition)
- Calculate total energy consumption (>85 TJ threshold for mandatory EnMS)
- Check ISO 50001 certification status (audit exemption)
- Identify applicable Member State transposition requirements
- Determine audit obligation status and next audit due date

**Phase 2: Audit Scheduling**
- Calculate next mandatory audit date (4-year cycle from last compliant audit)
- Generate 6-month, 3-month, and 1-month advance warnings
- Track auditor procurement timeline
- Verify auditor independence and EN 16247-5 competence
- Schedule audit execution within compliance window

**Phase 3: Compliance Reporting**
- Generate EED compliance summary for Member State authority
- Generate EU ETS energy efficiency documentation for free allocation
- Generate IED BAT compliance assessment for IPPC permit
- Track all compliance deadlines in unified calendar
- Generate management compliance dashboard with RAG status

**Acceptance Criteria:**
- [ ] Correctly classifies enterprise as SME or non-SME per EU Recommendation 2003/361
- [ ] Correctly calculates 85 TJ threshold using total primary energy consumption
- [ ] Correctly identifies ISO 50001 exemption status with certificate expiry tracking
- [ ] Audit scheduling generates alerts at 6, 3, and 1 month before due date
- [ ] Compliance dashboard shows all obligations with RAG status and next action

### 5.8 Workflow 8: ISO 50001 Certification Workflow

**Purpose:** ISO 50001 certification support from initial gap analysis through management review preparation.

**Phase 1: EnMS Gap Analysis**
- Assess current state against all ISO 50001:2018 clauses (4 through 10)
- Score each clause: COMPLIANT / PARTIALLY_COMPLIANT / NON_COMPLIANT / NOT_ADDRESSED
- Identify documentation gaps (energy policy, procedures, records)
- Identify process gaps (energy review, EnPI definition, operational control)
- Estimate effort and timeline to close gaps
- Generate gap analysis report with prioritized action plan

**Phase 2: Energy Policy Development**
- Draft energy policy per ISO 50001 Clause 5.2 requirements
- Ensure policy includes commitment to continual improvement of energy performance
- Define roles, responsibilities, and authorities for EnMS
- Generate documented information templates (procedures, forms, records)
- Prepare energy planning documentation (scope, boundary, energy review)

**Phase 3: EnPI Tracking**
- Define EnPIs per ISO 50006 methodology (from Engine 1)
- Establish energy baselines (from Engine 1)
- Set energy objectives and energy targets with measurable EnPI values
- Configure automated EnPI monitoring and reporting
- Track progress against objectives and targets on monthly basis
- Generate control charts (CUSUM, EWMA) for continual improvement evidence

**Phase 4: Management Review Preparation**
- Compile management review inputs per ISO 50001 Clause 9.3:
  - Status of actions from previous reviews
  - Energy policy suitability review
  - Energy performance and EnPI improvement
  - Compliance obligations status
  - Audit results (internal and external)
  - Nonconformities and corrective actions
  - Strategic direction alignment
  - Improvement opportunities
- Generate management review package (Template 10)
- Record management review outputs and decisions

**Acceptance Criteria:**
- [ ] Gap analysis covers all ISO 50001:2018 clauses with evidence-based scoring
- [ ] Energy policy template includes all mandatory elements per Clause 5.2
- [ ] EnPI tracking provides automated monthly reports with trend analysis
- [ ] Management review package includes all required inputs per Clause 9.3
- [ ] Full audit trail for all EnMS documentation changes
- [ ] Gap-to-certification timeline estimation within +/- 2 months of actual

---

## 6. Template Specifications

### 6.1 Template 1: Energy Audit Report (EN 16247 Format)

**Purpose:** Fully compliant EN 16247-1 Annex A energy audit report.

**Mandatory Sections per EN 16247-1:**

| Section | Content | Data Source |
|---------|---------|-------------|
| Cover page | Facility name, audit type, date, auditor, report number | Configuration |
| Executive summary | Key findings, total savings potential, top 5 ECMs, ROI summary | All engines |
| 1. Background | Audit scope, boundary, objectives, methodology, team | Audit engine |
| 2. Description of audited object | Facility profile, production, operating hours, layout | Facility registry |
| 3. Energy data | Energy carriers, consumption (24+ months), costs, trends | Baseline engine |
| 4. Current energy use analysis | Energy balance (Sankey), end-use breakdown, load profiles | Process mapping engine |
| 5. Energy performance indicators | EnPIs, baseline comparison, benchmarks | Baseline + benchmark engines |
| 6. Significant energy uses | SEU identification, ranking by consumption | Audit + equipment engines |
| 7. Proposals for improvement | Ranked ECMs with savings, costs, payback per measure | Savings engine |
| 8. Economic evaluation | NPV, IRR, LCCA for each ECM and total portfolio | Savings engine financials |
| 9. Interaction effects | ECM interactions, combined savings adjustment | Savings engine interaction matrix |
| 10. Implementation plan | Phased schedule, resource requirements, milestones | Savings engine prioritization |
| 11. M&V plan | IPMVP option per ECM, measurement requirements | Savings engine M&V output |
| Appendices | Raw data, calculation details, drawings, photographs | All engines, data imports |

**Output Formats:** MD, HTML, PDF, JSON
**Typical Length:** 60-150 pages (Type 2 audit)

### 6.2 Template 2: Energy Baseline Report

**Purpose:** Document energy baselines, regression models, and EnPI definitions per ISO 50006.

**Sections:**
- Baseline period definition and justification
- Data summary (energy, production, weather, operating schedule)
- Regression model selection and variable analysis
- Model coefficients with statistical significance (p-values, confidence intervals)
- Model validation metrics (R-squared, CV(RMSE), NMBE, Durbin-Watson)
- Residual analysis with outlier identification
- EnPI definitions with baseline values
- Normalization factors and adjustment procedures
- Baseline boundary conditions and applicability limits

**Output Formats:** MD, HTML, JSON

### 6.3 Template 3: Energy Savings Verification Report (IPMVP)

**Purpose:** IPMVP-compliant M&V report documenting verified energy savings.

**Sections:**
- ECM description and implementation details
- IPMVP option selected (A, B, C, or D) with justification
- Baseline model documentation
- Post-implementation measurement results
- Routine adjustments applied (weather, production normalization)
- Non-routine adjustments applied (structural changes)
- Verified savings calculation with uncertainty analysis
- Comparison: verified vs. projected savings
- Persistence analysis and degradation assessment
- Cost savings verification and financial performance

**Output Formats:** MD, HTML, PDF, JSON

### 6.4 Template 4: Energy Management Dashboard

**Purpose:** Real-time energy management dashboard for continuous monitoring.

**Dashboard Panels:**
- Facility energy consumption (real-time, daily, weekly, monthly trends)
- EnPI tracking with target lines and control limits
- CUSUM chart showing cumulative savings/losses vs. baseline
- Energy cost tracking (actual vs. budget vs. target)
- Equipment efficiency summary (motors, pumps, compressors, boilers)
- Alert status (active warnings and critical alerts)
- Weather normalization overlay (actual vs. degree-day-adjusted)
- ECM implementation tracker (planned, in-progress, completed, verified)
- Benchmark comparison (facility vs. sector average vs. BAT-AEL)

**Output Formats:** MD, HTML, JSON

### 6.5 Template 5: Compressed Air System Report

**Purpose:** Comprehensive compressed air audit report.

**Sections:**
- System overview (compressors, dryers, receivers, capacity)
- System schematic / piping diagram
- Compressor performance data (specific power, load profiles, efficiency)
- Leak survey results (leak register, total leak rate, cost)
- Pressure profile analysis (generation to point of use)
- Artificial demand assessment
- Optimization recommendations (leak repair, pressure reduction, VSD, receivers)
- Heat recovery potential from compression
- Financial analysis per recommendation
- Implementation priority matrix

**Output Formats:** MD, HTML, JSON

### 6.6 Template 6: Steam System Assessment Report

**Purpose:** Comprehensive steam system audit report.

**Sections:**
- Steam system overview (boilers, headers, distribution, condensate)
- Boiler efficiency analysis (direct and indirect method)
- Stack gas analysis results and O2 trim recommendation
- Steam trap survey results with map showing failed traps
- Insulation survey results with economic insulation thickness
- Condensate return analysis with return rate and recovery potential
- Flash steam recovery opportunities
- Blowdown optimization and heat recovery
- CHP/cogeneration feasibility (if applicable)
- Financial analysis per recommendation

**Output Formats:** MD, HTML, JSON

### 6.7 Template 7: Waste Heat Recovery Feasibility Report

**Purpose:** Waste heat recovery feasibility report with pinch analysis and technology recommendations.

**Sections:**
- Waste heat source inventory (temperature, flow, duty, intermittency)
- Pinch analysis results (composite curves, grand composite curve, pinch temperature)
- Minimum utility targets (hot utility, cold utility)
- Heat exchanger network design
- Technology recommendations per recovery opportunity
- ORC / heat pump / absorption chiller assessment (where applicable)
- CapEx estimates using cost correlations
- Economic analysis (payback, NPV, IRR per opportunity)
- Sensitivity analysis
- Implementation priority and phasing

**Output Formats:** MD, HTML, PDF, JSON

### 6.8 Template 8: Equipment Efficiency Assessment Report

**Purpose:** Equipment-level efficiency report with upgrade recommendations.

**Sections:**
- Equipment inventory summary (count by type, age, rating)
- Motor analysis (IE class distribution, loading, VSD potential)
- Pump analysis (BEP deviation, wire-to-water efficiency, optimization)
- Compressor analysis (type, specific power, capacity utilization)
- Fan analysis (system curves, VSD potential, damper elimination)
- Boiler analysis (combustion efficiency, loss breakdown)
- HVAC analysis (COP/EER, part-load, economizer status)
- Refrigeration analysis (COP, condenser/evaporator approach)
- Upgrade recommendations per equipment item with savings and payback
- Capital planning summary (immediate, 1-year, 3-year, 5-year)

**Output Formats:** MD, HTML, JSON

### 6.9 Template 9: Regulatory Compliance Summary

**Purpose:** EED/ETS/IED compliance status report.

**Sections:**
- Enterprise classification (SME/non-SME, energy consumption, sector)
- EED Article 11 compliance status (audit obligation, schedule, exemptions)
- EED Article 12 compliance status (EnMS requirement for >85 TJ enterprises)
- EU ETS compliance status (energy efficiency benchmarks, free allocation)
- IED/BAT compliance status (BAT-AEL comparison by installation)
- Upcoming deadlines with RAG status
- Compliance action items with responsible person and due date
- Historical compliance record
- Risk assessment for non-compliance (penalties, permit conditions)

**Output Formats:** MD, HTML, JSON

### 6.10 Template 10: ISO 50001 Management Review Package

**Purpose:** Complete management review package per ISO 50001 Clause 9.3.

**Sections:**
- Review date, attendees, agenda
- Status of actions from previous management review
- Energy policy review (suitability, adequacy, effectiveness)
- Energy performance summary (EnPI trends, target achievement)
- EnMS internal audit results summary
- Compliance obligations review (EED, ETS, IED, other)
- Nonconformity and corrective action register
- Opportunities for improvement (new ECMs, technology, processes)
- Changes in external/internal issues affecting EnMS
- Resource needs for upcoming period
- Decisions and actions arising from review
- Next review date

**Output Formats:** MD, HTML, PDF, JSON

---

## 7. Integration Specifications

### 7.1 Integration 1: Pack Orchestrator

**Purpose:** Master orchestration pipeline for all PACK-031 engines.

**DAG Pipeline (10 phases):**

```
Phase 1: Facility Setup (setup_wizard)
  |
Phase 2: Data Ingestion (data_energy_bridge, utility_metering_bridge, bms_scada_bridge)
  |
Phase 3: Baseline Establishment (energy_baseline_engine)
  |
Phase 4: Energy Audit (energy_audit_engine + process_energy_mapping_engine)
  |
Phase 5: Equipment Analysis (equipment_efficiency_engine)
  |
Phase 6: Subsystem Audits [conditional]
  |-- 6a: Compressed Air (compressed_air_engine) [if compressed_air_present]
  |-- 6b: Steam System (steam_system_engine) [if steam_system_present]
  |-- 6c: Waste Heat (waste_heat_recovery_engine) [if waste_heat_sources_present]
  |-- 6d: Lighting/HVAC (lighting_hvac_engine) [if lighting_hvac_in_scope]
  |
Phase 7: Savings Identification (energy_savings_opportunity_engine)
  |
Phase 8: Benchmarking (energy_benchmark_engine)
  |
Phase 9: Compliance Check (regulatory_compliance_workflow)
  |
Phase 10: Report Generation (all templates)
```

**Orchestrator Features:**
- Conditional phase execution based on facility type and preset configuration
- Retry with exponential backoff (max 3 retries per phase)
- SHA-256 provenance chain across all phases
- Phase-level caching (skip re-execution if inputs unchanged)
- Progress tracking with percentage completion per phase
- Error isolation (failed subsystem audit does not block other phases)

### 7.2 Integration 2: MRV Energy Bridge

**Purpose:** Connect energy audit findings to GHG emission calculations.

**Data Flow:**
- Energy consumption (kWh electricity, m3 gas, L fuel oil) -> MRV agents for Scope 1+2 emissions
- MRV-001: fuel combustion from boilers, furnaces, kilns -> Scope 1 CO2e
- MRV-009/010: grid electricity -> Scope 2 CO2e (location-based and market-based)
- MRV-016: upstream fuel/energy -> Scope 3 Category 3 CO2e
- Energy savings from ECMs -> projected emission reduction (tCO2e/yr)

**Bi-directional Benefits:**
- Energy audit -> MRV: accurate fuel/energy data improves emission calculation accuracy
- MRV -> Energy audit: carbon cost (EUR/tCO2e from EU ETS) enhances ECM financial analysis

### 7.3 Integration 3: Data Energy Bridge

**Purpose:** Route energy data through AGENT-DATA agents for quality assurance.

**Data Pipeline:**
- Raw meter data / utility bills -> DATA-002 (Excel/CSV Normalizer) -> standardized time series
- ERP energy cost data -> DATA-003 (ERP/Finance Connector) -> cost time series
- All energy data -> DATA-010 (Data Quality Profiler) -> quality score and gap report
- Gapped time series -> DATA-014 (Time Series Gap Filler) -> complete time series
- Multiple data sources -> DATA-015 (Cross-Source Reconciliation) -> reconciled dataset
- Ongoing feeds -> DATA-016 (Data Freshness Monitor) -> staleness alerts

### 7.4 Integration 4: EED Compliance Bridge

**Purpose:** Track and manage EU Energy Efficiency Directive obligations.

**Key Functions:**
- Enterprise classification engine (non-SME determination per EU Recommendation 2003/361)
- 85 TJ threshold calculator (total primary energy across all sites)
- 4-year audit cycle scheduler with advance warning system
- ISO 50001 exemption tracker with certificate expiry monitoring
- Member State specific requirement database (transposition variations)
- Compliance evidence package generator for national authority submission

### 7.5 Integration 5: ISO 50001 Bridge

**Purpose:** Support ISO 50001 energy management system lifecycle.

**Key Functions:**
- EnPI registry: define, track, and report EnPIs per ISO 50006
- Objective/target tracker: manage energy objectives with measurable targets
- Internal audit scheduler: schedule and track EnMS internal audits
- Management review scheduler: ensure regular management reviews
- Continual improvement register: log improvements with evidence
- Document control: version tracking for EnMS documentation
- Nonconformity register: track NCRs with root cause analysis and corrective action

### 7.6 Integration 6: BMS/SCADA Bridge

**Purpose:** Real-time data integration with building and industrial management systems.

**Supported Protocols:**
- BACnet/IP: building automation (HVAC, lighting, metering)
- Modbus TCP/RTU: industrial equipment, PLCs, power meters
- OPC-UA: industrial automation, SCADA systems
- MQTT: IoT sensors, smart meters
- REST API: cloud-based BMS platforms

**Data Points:**
- Energy meters (kWh, kW, kVAR, PF, voltage, current)
- Equipment status (on/off, speed, setpoint, output)
- Environmental sensors (temperature, humidity, CO2, lux)
- Process parameters (pressure, flow, temperature, level)

### 7.7 Integration 7: Utility Metering Bridge

**Purpose:** Smart meter and sub-metering data integration.

**Supported Data Sources:**
- Smart meter interval data (AMI feeds via utility API or file export)
- Sub-meter pulse/Modbus data via BMS/SCADA
- Manual meter reads (CSV/Excel upload for sites without AMR)
- Virtual meters (calculated from other meters: parent - sum of sub-meters)
- Utility bill data (monthly totals for reconciliation)

**Data Processing:**
- 15-minute interval alignment and gap detection
- Demand (kW) calculation from energy (kWh) intervals
- Power factor calculation from kW and kVAR
- Reactive energy tracking for power quality
- Utility bill reconciliation against meter totals (+/- 2% tolerance)

### 7.8 Integration 8: Equipment Registry Bridge

**Purpose:** Asset management and CMMS integration for equipment data.

**Supported CMMS Platforms:**
- SAP Plant Maintenance (PM)
- IBM Maximo
- Infor EAM
- eMaint / Fluke
- Maintenance Connection

**Data Exchange:**
- Equipment nameplate data (manufacturer, model, rating, IE class, year of manufacture)
- Runtime hours and duty cycle
- Maintenance records (overhaul, rebuild, replacement)
- Replacement schedule (planned end-of-life, procurement timeline)
- Condition monitoring data (vibration, thermography, oil analysis)

### 7.9 Integration 9: Weather Normalization Bridge

**Purpose:** Weather data integration for baseline normalization and HVAC analysis.

**Data Sources:**
- Local weather station API (temperature, humidity, wind, solar radiation)
- NOAA ISD (Integrated Surface Database) for global coverage
- Meteostat / Open-Meteo for European coverage
- Degree-day calculation (HDD/CDD with configurable base temperature)
- TMY (Typical Meteorological Year) data for forecasting and simulation

**Output:**
- Hourly/daily weather data aligned with energy data timestamps
- HDD and CDD calculated for facility-specific base temperature
- Wet-bulb temperature for cooling tower analysis
- Solar radiation for daylight harvesting calculations
- Wind data for infiltration analysis

### 7.10 Integration 10: Health Check

**Purpose:** System verification for all PACK-031 components.

**22 Verification Categories:**

| # | Category | Checks |
|---|----------|--------|
| 1 | Engine availability | All 10 engines respond to health ping |
| 2 | Workflow availability | All 8 workflows respond to health ping |
| 3 | Template availability | All 10 templates generate test output |
| 4 | Database connectivity | PostgreSQL connection, migration status |
| 5 | Redis cache | Cache connectivity and response time |
| 6 | MRV bridge | Connection to MRV agents (001, 009, 010, 016) |
| 7 | Data bridge | Connection to DATA agents (002, 003, 010, 014, 015, 016) |
| 8 | Foundation bridge | Connection to FOUND agents (001-010) |
| 9 | BMS/SCADA connectivity | Protocol adapters responding |
| 10 | Meter data feed | Latest data timestamp within freshness threshold |
| 11 | Weather data feed | Latest weather data within 24 hours |
| 12 | Equipment registry | CMMS sync status |
| 13 | Baseline model validity | Regression models within statistical thresholds |
| 14 | EnPI calculation | Test EnPI calculation produces expected result |
| 15 | Regulatory data | EED/ETS/IED obligation database currency |
| 16 | BREF data | BAT-AEL benchmark data loaded and current |
| 17 | Authentication | JWT RS256 token issuance/validation |
| 18 | Authorization | RBAC permission checks for all 5 roles |
| 19 | Encryption | AES-256-GCM encrypt/decrypt test |
| 20 | Audit trail | SEC-005 audit logging operational |
| 21 | Provenance | SHA-256 hash generation/verification |
| 22 | Disk/memory | Storage < 80% capacity, memory < 80% ceiling |

### 7.11 Integration 11: Setup Wizard

**Purpose:** Guided 8-step facility configuration for new deployments.

**Steps:**

| Step | Configuration | Inputs Required |
|------|--------------|-----------------|
| 1. Facility Profile | Name, address, NACE code, floor area, employees, production type | Basic facility information |
| 2. Metering Infrastructure | Main meters, sub-meters, data sources, intervals | Meter inventory and data feeds |
| 3. Equipment Inventory | Import from CMMS or manual entry; motors, pumps, compressors, boilers, HVAC | Equipment nameplate data |
| 4. Production Data | Production units, volumes, schedules, product mix | Production system or manual entry |
| 5. Weather Station | Nearest weather station selection or manual coordinates | Location data |
| 6. Regulatory Obligations | Enterprise size, EU/national requirements, ISO 50001 status | Legal entity information |
| 7. Audit History | Previous audit reports, dates, auditors, findings | Historical audit records |
| 8. ISO 50001 Status | Current EnMS maturity, certification status, EnPI definitions | EnMS documentation |

### 7.12 Integration 12: EU ETS Bridge

**Purpose:** EU Emissions Trading System integration for energy-intensive installations.

**Key Functions:**
- Free allocation benchmark comparison (product benchmarks in tCO2e/unit)
- Carbon cost integration into ECM financial analysis (EUR/tCO2e from ETS allowance price)
- Emissions intensity monitoring (tCO2e per unit production)
- Energy efficiency documentation for NIMs (National Implementation Measures)
- Carbon leakage risk assessment based on energy intensity

---

## 8. Preset Specifications

### 8.1 Preset 1: Manufacturing Discrete

**Facility Type:** Discrete manufacturing (automotive, electronics, machinery, metal fabrication)
**Energy Profile:** Motor-driven systems dominate (60-75% of electricity)

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Compressed air audit | Enabled (high relevance) |
| Steam system audit | Conditional (enabled if steam present) |
| Waste heat recovery | Conditional (enabled if process heat present) |
| Primary energy carrier | Electricity (60-80%), Natural gas (15-30%), Other (5-10%) |
| Key SEUs | Motor systems, compressed air, HVAC, lighting |
| EnPI default | kWh/unit produced, kWh/production-hour |
| Baseline variables | Production volume, HDD, CDD, operating days |
| Benchmarking subsectors | NACE C.25-C.30 (machinery, automotive, electrical) |
| Typical savings potential | 15-25% of total energy cost |

### 8.2 Preset 2: Process Industry

**Facility Type:** Chemical, petrochemical, pharmaceutical processing
**Energy Profile:** Steam and process heat intensive (40-65% thermal energy)

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Compressed air audit | Enabled |
| Steam system audit | Enabled (critical) |
| Waste heat recovery | Enabled (high priority - large waste heat streams) |
| Primary energy carrier | Natural gas (40-60%), Electricity (25-40%), Steam (10-20%) |
| Key SEUs | Boilers/steam, process heat, reactors, distillation, pumping |
| EnPI default | GJ/tonne product, kWh/tonne product |
| Baseline variables | Production volume (tonnes), product mix, ambient temperature |
| Benchmarking subsectors | NACE C.20-C.21 (chemicals, pharmaceuticals) |
| Typical savings potential | 10-20% of total energy cost |

### 8.3 Preset 3: Food & Beverage Processing

**Facility Type:** Food processing, brewing, dairy, bakery, meat processing
**Energy Profile:** Refrigeration + steam/hot water intensive

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Compressed air audit | Enabled (packaging lines) |
| Steam system audit | Enabled (pasteurization, sterilization, CIP) |
| Waste heat recovery | Enabled (refrigeration condenser heat, process exhaust) |
| Primary energy carrier | Electricity (40-60%), Natural gas (30-50%), Other (5-15%) |
| Key SEUs | Refrigeration, boilers/steam, compressed air, HVAC, process heat |
| EnPI default | kWh/tonne product, kWh/hL (beverage), GJ/tonne product |
| Baseline variables | Production volume, product mix, ambient temperature, seasonal variation |
| Benchmarking subsectors | NACE C.10-C.11 (food, beverages) with BREF Food BREF BAT-AELs |
| Typical savings potential | 15-30% of total energy cost |

### 8.4 Preset 4: Data Center

**Facility Type:** Colocation, enterprise, hyperscale data centers
**Energy Profile:** Cooling-dominated, UPS losses, IT load

| Parameter | Value |
|-----------|-------|
| Engines enabled | 8 (exclude steam, exclude waste heat if no district heating) |
| Compressed air audit | Disabled (typically not present) |
| Steam system audit | Disabled |
| Waste heat recovery | Conditional (enabled if district heat export possible) |
| Primary energy carrier | Electricity (95-100%) |
| Key SEUs | Cooling (CRAC/CRAH/chiller), UPS, IT load, lighting |
| EnPI default | PUE (primary), DCiE, CUE, WUE |
| Baseline variables | IT load (kW), ambient temperature, wet-bulb temperature |
| Benchmarking subsectors | The Green Grid PUE benchmarks, EED Article 25 requirements |
| Typical savings potential | PUE improvement from 1.6 to 1.3 = 18.75% cooling energy reduction |

### 8.5 Preset 5: Warehouse & Logistics

**Facility Type:** Distribution centers, cold storage, logistics hubs
**Energy Profile:** Lighting and HVAC dominated

| Parameter | Value |
|-----------|-------|
| Engines enabled | 7 (lighting/HVAC primary; exclude steam, compressed air minimal) |
| Compressed air audit | Conditional (enabled if pneumatic systems present) |
| Steam system audit | Disabled |
| Waste heat recovery | Conditional (cold storage condenser heat) |
| Primary energy carrier | Electricity (70-90%), Natural gas (10-25%), Diesel (forklift, 5-10%) |
| Key SEUs | Lighting, HVAC/refrigeration, dock doors, forklift charging |
| EnPI default | kWh/m2, kWh/pallet-movement |
| Baseline variables | Throughput (pallets), HDD, CDD, operating hours |
| Benchmarking subsectors | BREEAM warehouse, national warehouse energy benchmarks |
| Typical savings potential | 20-40% of total energy cost (lighting LED retrofit often dominant) |

### 8.6 Preset 6: Automotive Manufacturing

**Facility Type:** Automotive OEM and Tier 1/2 supplier plants
**Energy Profile:** Paint shop dominant, compressed air intensive

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Compressed air audit | Enabled (critical - body shop, assembly) |
| Steam system audit | Conditional (paint shop ovens, E-coat) |
| Waste heat recovery | Enabled (paint shop exhaust, oven exhaust) |
| Primary energy carrier | Electricity (50-65%), Natural gas (30-45%), Other (5-10%) |
| Key SEUs | Paint shop (30-50% of plant energy), compressed air, welding, HVAC |
| EnPI default | kWh/vehicle, kWh/unit produced, GJ/vehicle |
| Baseline variables | Vehicle production count, vehicle mix (platform), ambient temperature |
| Benchmarking subsectors | Automotive sector energy benchmarks, VDA sustainability standards |
| Typical savings potential | 10-20% of total energy cost |

### 8.7 Preset 7: Steel & Metals

**Facility Type:** Electric arc furnace, blast furnace, rolling mills, foundries
**Energy Profile:** Very high energy intensity, process heat dominant

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Compressed air audit | Enabled |
| Steam system audit | Conditional (BF-BOF route) |
| Waste heat recovery | Enabled (critical - slag heat, furnace exhaust, rolling mill cooling) |
| Primary energy carrier | Electricity (40-70% for EAF), Natural gas/coal (50-80% for BF-BOF) |
| Key SEUs | EAF/BF, rolling mill, reheat furnace, compressed air, water cooling |
| EnPI default | kWh/tonne liquid steel (EAF), GJ/tonne hot rolled coil (integrated) |
| Baseline variables | Production tonnes, product mix, scrap ratio (EAF), tapping temperature |
| Benchmarking subsectors | Iron & Steel BREF BAT-AELs, EU ETS product benchmarks |
| Typical savings potential | 5-15% of total energy cost (lower % due to already optimized processes) |

### 8.8 Preset 8: SME Industrial

**Facility Type:** Small/medium industrial facility (any subsector)
**Energy Profile:** Simplified analysis for facilities with limited metering

| Parameter | Value |
|-----------|-------|
| Engines enabled | 6 (baseline, audit, equipment, savings, lighting/HVAC, benchmark) |
| Compressed air audit | Simplified (no detailed leak survey) |
| Steam system audit | Disabled (unless specifically requested) |
| Waste heat recovery | Disabled |
| Primary energy carrier | Electricity + Natural gas (typical) |
| Key SEUs | Lighting, HVAC, motors, compressed air |
| EnPI default | kWh/m2, EUR/employee, kWh/unit (if measurable) |
| Baseline variables | Degree-days (HDD/CDD), operating hours |
| Benchmarking subsectors | National SME energy benchmarks by NACE code |
| Typical savings potential | 15-35% of total energy cost (often low-hanging fruit not yet captured) |
| EED applicability | SME exemption check (may be exempt from mandatory audit) |

---

## 9. Agent Dependencies

### 9.1 MRV Agents (30)

All 30 AGENT-MRV agents are available as dependencies via `mrv_energy_bridge.py`, with primary relevance for:
- **MRV-001 Stationary Combustion**: Boiler, furnace, kiln fuel combustion emissions linked to energy consumption
- **MRV-003 Mobile Combustion**: On-site mobile equipment fuel consumption
- **MRV-009 Scope 2 Location-Based**: Grid electricity emissions from metered consumption
- **MRV-010 Scope 2 Market-Based**: Electricity emissions with RE certificates and contracts
- **MRV-016 Fuel & Energy Activities (Cat 3)**: Upstream energy emissions (T&D losses, well-to-tank)
- **MRV-011 Steam/Heat Purchase**: Purchased steam/heat emissions
- **MRV-012 Cooling Purchase**: Purchased cooling emissions

Additional MRV agents available for comprehensive GHG-energy linking across all scopes.

### 9.2 Data Agents (20)

All 20 AGENT-DATA agents via `data_energy_bridge.py`, with primary relevance for:
- **DATA-002 Excel/CSV Normalizer**: Utility bill and meter data import
- **DATA-003 ERP/Finance Connector**: Energy procurement and cost data from SAP, Oracle, etc.
- **DATA-010 Data Quality Profiler**: Energy data completeness and accuracy assessment
- **DATA-014 Time Series Gap Filler**: Filling gaps in energy meter time series data
- **DATA-015 Cross-Source Reconciliation**: Reconciling meter data vs. utility bills vs. ERP
- **DATA-016 Data Freshness Monitor**: Ensuring energy data is current for real-time monitoring

### 9.3 Foundation Agents (10)

All 10 AGENT-FOUND agents for orchestration, schema validation, unit normalization (critical for energy unit conversions: kWh, GJ, therms, BTU, tonnes steam), assumptions registry, citations, access control, etc.

### 9.4 Application Dependencies

- **GL-GHG-APP**: GHG inventory management for energy-related emissions
- **GL-CSRD-APP**: ESRS E1-5 energy consumption and mix disclosure
- **GL-CDP-APP**: CDP C7 Energy and C8 Energy-related emissions modules
- **GL-Taxonomy-APP**: EU Taxonomy climate mitigation criteria for energy efficiency investments

### 9.5 Optional Pack Dependencies

- PACK-021 Net Zero Starter Pack: Baseline emissions linked to energy consumption, reduction pathways (via integration if present)
- PACK-022 Net Zero Acceleration Pack: Advanced decarbonization scenarios including energy efficiency (via integration if present)
- PACK-023 SBTi Alignment Pack: SBTi target-linked energy efficiency programs (via integration if present)

---

## 10. Performance Targets

| Metric | Target |
|--------|--------|
| Energy baseline calculation per facility (monthly data, 3 years) | <30 seconds |
| Energy baseline calculation per facility (15-min interval, 3 years) | <5 minutes |
| Energy audit report generation (Type 2, full facility) | <10 minutes |
| Process energy mapping (50 process nodes) | <2 minutes |
| Equipment efficiency calculation (1,000 items) | <3 minutes |
| ECM identification and financial analysis (100 measures) | <5 minutes |
| Pinch analysis (20 hot/cold streams) | <1 minute |
| Compressed air system analysis (10 compressors) | <2 minutes |
| Steam system analysis (5 boilers, 500 traps) | <3 minutes |
| Lighting/HVAC analysis (100 spaces, 50 systems) | <5 minutes |
| Benchmarking (single facility, 5 subsector comparisons) | <1 minute |
| Full initial energy audit workflow (Type 2) | <60 minutes |
| Real-time data processing latency | <5 seconds |
| Memory ceiling | 4096 MB |
| Cache hit target | 70% |
| Max facilities | 500 |
| Max equipment items per facility | 50,000 |
| Max meters per facility | 10,000 |
| Max concurrent audits | 20 |

---

## 11. Security Requirements

- JWT RS256 authentication
- RBAC with 5 roles: `energy_manager`, `facility_engineer`, `compliance_officer`, `external_auditor`, `admin`
- Facility-level access control (users see only facilities assigned to them)
- AES-256-GCM encryption at rest for all energy data, production data, and audit results
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all calculation outputs (baselines, EnPIs, ECM savings, audit results)
- Full audit trail per SEC-005 (who changed what, when, with provenance chain)
- BMS/SCADA credentials encrypted via Vault (SEC-006)
- Metering API keys and utility account credentials encrypted via Vault
- Read-only mode for external auditors (no data modification, no deletion)
- Data retention: minimum 8 years for EED/ETS compliance records

**RBAC Permission Matrix:**

| Permission | energy_manager | facility_engineer | compliance_officer | external_auditor | admin |
|------------|---------------|-------------------|-------------------|-----------------|-------|
| Create/edit facility | Yes | No | No | No | Yes |
| Upload meter data | Yes | Yes | No | No | Yes |
| Run energy audit | Yes | Yes | No | No | Yes |
| View audit results | Yes | Yes | Yes | Yes (assigned) | Yes |
| Approve ECMs | Yes | No | No | No | Yes |
| Edit baselines | Yes | Yes | No | No | Yes |
| Generate compliance reports | Yes | No | Yes | No | Yes |
| Export data | Yes | Yes | Yes | Yes (assigned) | Yes |
| Manage users | No | No | No | No | Yes |
| View all facilities | No | No | Yes | No | Yes |
| Delete records | No | No | No | No | Yes |

---

## 12. Database Migrations

Inherits platform migrations V001-V180. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V181__pack031_energy_baselines_001 | `iea_facilities`, `iea_energy_baselines`, `iea_enpi_definitions`, `iea_baseline_adjustments` | Facility registry, energy consumption baselines with regression models, EnPI definitions per ISO 50006, and baseline adjustment records |
| V182__pack031_energy_audits_002 | `iea_energy_audits`, `iea_audit_findings`, `iea_audit_schedules`, `iea_auditor_registry` | EN 16247 energy audit records with type/scope/status, per-system audit findings, EED Article 11 audit scheduling, and auditor competence tracking |
| V183__pack031_equipment_003 | `iea_equipment_registry`, `iea_equipment_efficiency`, `iea_motor_analysis`, `iea_pump_analysis`, `iea_compressor_analysis`, `iea_boiler_analysis` | Equipment inventory with nameplate data, equipment-level efficiency calculations, and per-type analysis records (motors, pumps, compressors, boilers) |
| V184__pack031_savings_004 | `iea_ecm_recommendations`, `iea_financial_analysis`, `iea_mv_plans`, `iea_savings_verification` | Energy conservation measures with savings and financial metrics, M&V plans per IPMVP, and post-implementation savings verification records |
| V185__pack031_steam_005 | `iea_steam_systems`, `iea_boiler_efficiency`, `iea_steam_trap_surveys`, `iea_condensate_analysis`, `iea_insulation_assessment` | Steam system records, boiler efficiency (direct and indirect method), steam trap survey results, condensate return analysis, and insulation assessment |
| V186__pack031_compressed_air_006 | `iea_compressed_air_systems`, `iea_leak_surveys`, `iea_pressure_profiles`, `iea_compressor_optimization` | Compressed air system records, leak survey registers, pressure profile analysis, and compressor staging/VSD optimization results |
| V187__pack031_waste_heat_007 | `iea_waste_heat_sources`, `iea_pinch_analysis`, `iea_heat_exchangers`, `iea_recovery_opportunities` | Waste heat source inventory, pinch analysis results (composite curves, utility targets), heat exchanger sizing, and recovery opportunity economic analysis |
| V188__pack031_benchmarks_008 | `iea_benchmarks`, `iea_sec_calculations`, `iea_bat_comparisons`, `iea_maturity_assessments` | Facility energy benchmarks, SEC calculations, BAT-AEL comparisons from BREF documents, and energy efficiency maturity assessments |
| V189__pack031_compliance_009 | `iea_eed_obligations`, `iea_ets_compliance`, `iea_ied_bat_status`, `iea_compliance_deadlines` | EED Article 11/12 obligation tracking, EU ETS energy efficiency records, IED BAT compliance status, and compliance deadline management |
| V190__pack031_iso50001_010 | `iea_enms_status`, `iea_energy_policies`, `iea_energy_objectives`, `iea_management_reviews`, `iea_continual_improvement` | ISO 50001 EnMS status tracking, energy policy records, energy objectives and targets, management review packages, and continual improvement register |

**Table Prefix:** `iea_` (Industrial Energy Audit)

**Row-Level Security (RLS):**
- All tables have `facility_id` column for facility-level access control
- RLS policies enforce that users can only see data for facilities assigned to their role
- External auditors have read-only access to specifically assigned audit records
- Admin role bypasses RLS for cross-facility reporting

**Indexes:**
- Composite indexes on `(facility_id, created_at)` for time-series queries
- GIN indexes on JSONB columns for flexible metadata storage
- Partial indexes on `status` columns for active-record filtering
- B-tree indexes on `equipment_id`, `audit_id`, `baseline_id` for foreign key joins

---

## 13. File Structure

```
packs/energy-efficiency/PACK-031-industrial-energy-audit/
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
      manufacturing_discrete.yaml
      process_industry.yaml
      food_beverage.yaml
      data_center.yaml
      warehouse_logistics.yaml
      automotive_manufacturing.yaml
      steel_metals.yaml
      sme_industrial.yaml
  engines/
    __init__.py
    energy_baseline_engine.py
    energy_audit_engine.py
    process_energy_mapping_engine.py
    equipment_efficiency_engine.py
    energy_savings_opportunity_engine.py
    waste_heat_recovery_engine.py
    compressed_air_engine.py
    steam_system_engine.py
    lighting_hvac_engine.py
    energy_benchmark_engine.py
  workflows/
    __init__.py
    initial_energy_audit_workflow.py
    continuous_monitoring_workflow.py
    energy_savings_verification_workflow.py
    compressed_air_audit_workflow.py
    steam_system_audit_workflow.py
    waste_heat_recovery_workflow.py
    regulatory_compliance_workflow.py
    iso50001_certification_workflow.py
  templates/
    __init__.py
    energy_audit_report.py
    energy_baseline_report.py
    energy_savings_verification_report.py
    energy_management_dashboard.py
    compressed_air_system_report.py
    steam_system_assessment_report.py
    waste_heat_recovery_report.py
    equipment_efficiency_report.py
    regulatory_compliance_summary.py
    iso50001_management_review.py
  integrations/
    __init__.py
    pack_orchestrator.py
    mrv_energy_bridge.py
    data_energy_bridge.py
    eed_compliance_bridge.py
    iso50001_bridge.py
    bms_scada_bridge.py
    utility_metering_bridge.py
    equipment_registry_bridge.py
    weather_normalization_bridge.py
    health_check.py
    setup_wizard.py
    eu_ets_bridge.py
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_energy_baseline_engine.py
    test_energy_audit_engine.py
    test_process_energy_mapping_engine.py
    test_equipment_efficiency_engine.py
    test_energy_savings_engine.py
    test_waste_heat_recovery_engine.py
    test_compressed_air_engine.py
    test_steam_system_engine.py
    test_lighting_hvac_engine.py
    test_energy_benchmark_engine.py
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
| Unit Tests | >90% line coverage | All 10 engines, all config models, all presets |
| Workflow Tests | >85% | All 8 workflows with synthetic facility data |
| Template Tests | 100% | All 10 templates in 3+ formats (MD, HTML, JSON, PDF where applicable) |
| Integration Tests | >80% | All 12 integrations with mock agents, BMS/SCADA simulators, and meter data |
| E2E Tests | Core happy path | Full pipeline from facility setup to EN 16247 audit report |
| Baseline Tests | 100% | All regression model types, degree-day normalization, EnPI calculations with known-value validation |
| Equipment Tests | >90% | Motor IE1-IE5, pump affinity laws, compressor specific power, boiler efficiency with engineering reference values |
| Compressed Air Tests | 100% | Leak quantification, pressure optimization, VSD analysis with measured-value validation |
| Steam System Tests | 100% | Boiler efficiency (direct/indirect), trap survey, condensate analysis with combustion analyzer cross-validation |
| Pinch Analysis Tests | 100% | Composite curves, pinch point, minimum utility targets with textbook problem validation |
| Financial Tests | 100% | NPV, IRR, payback, LCCA calculations with financial calculator cross-validation |
| IPMVP Tests | 100% | All 4 M&V options (A, B, C, D) with ASHRAE 14 statistical requirements |
| Benchmark Tests | >90% | BAT-AEL comparisons for all 50+ subsectors in BREF documents |
| Preset Tests | 100% | All 8 facility-type presets with representative facility scenarios |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

**Test Count Target:** 700+ tests (50-70 per engine, 40-50 integration, 20-30 E2E)

**Known-Value Validation Sets:**
- 100 motor efficiency calculations validated against IEC 60034-30-1 tables
- 50 pump affinity law calculations validated against pump curve data
- 30 boiler efficiency calculations validated against ASME PTC 4 examples
- 20 pinch analysis problems validated against published textbook solutions (Linnhoff, Smith)
- 50 degree-day regression models validated against ASHRAE 14 statistical criteria
- 25 compressed air leak calculations validated against orifice flow equations
- 30 steam trap loss calculations validated against Spirax Sarco engineering data

---

## 15. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-20 |
| Phase 2 | Engine implementation (10 engines) | 2026-03-21 to 2026-03-23 |
| Phase 3 | Workflow implementation (8 workflows) | 2026-03-23 to 2026-03-24 |
| Phase 4 | Template implementation (10 templates) | 2026-03-24 to 2026-03-25 |
| Phase 5 | Integration implementation (12 integrations) | 2026-03-25 to 2026-03-26 |
| Phase 6 | Test suite (700+ tests) | 2026-03-26 to 2026-03-28 |
| Phase 7 | Database migrations (V181-V190) | 2026-03-28 |
| Phase 8 | Documentation & Release | 2026-03-29 |

---

## 16. Appendix: EN 16247-1 Audit Report Structure (Mandatory Sections)

Per EN 16247-1:2022 Annex A, the energy audit report must contain:

| Section | Content | PACK-031 Mapping |
|---------|---------|-----------------|
| A.1 Executive Summary | Key findings, total energy use, top savings opportunities, payback summary | Template 1: Executive Summary section |
| A.2 Background | Audit objectives, scope, boundary, team, dates, methodology | Template 1: Background section |
| A.3 Energy Audit Description | Audit type, systems audited, data sources, measurement equipment | Template 1: Methodology section |
| A.4 Current Energy Use | Energy balance by carrier, end use breakdown, cost breakdown, historical trend | Engine 2 + Engine 3 output |
| A.5 Energy Performance Indicators | EnPIs for all significant energy uses, baseline comparison, benchmarks | Engine 1 + Engine 10 output |
| A.6 Proposals for Improving Energy Performance | Ranked ECM list with energy savings, cost savings, investment, payback, NPV | Engine 5 output |
| A.7 Economic Evaluation | Life-cycle cost analysis for each proposed measure, sensitivity analysis | Engine 5 financial analysis |
| A.8 Interaction Effects | ECM interactions (savings overlap, sequential dependency, synergies) | Engine 5 interaction matrix |
| A.9 Implementation Plan | Prioritized implementation sequence, resource requirements, timeline | Engine 5 + Workflow 1 Phase 5 |
| A.10 Measurement and Verification Plan | IPMVP M&V approach for each ECM, monitoring requirements | Engine 5 M&V plan output |

---

## 17. Appendix: EED 2023/1791 Key Articles for Energy Audits

| Article | Title | Pack Relevance |
|---------|-------|----------------|
| Article 2(30) | Definition of energy audit | EN 16247-compliant audit methodology |
| Article 8 | Energy efficiency obligation schemes | Savings quantification for obligation reporting |
| Article 11 | Energy audits | Mandatory audit every 4 years for non-SME; must cover >=85% of total energy; must identify waste heat potential |
| Article 11(2) | Exemption for ISO 50001 certified | EnMS exemption tracking in compliance engine |
| Article 11(3) | Transport coverage | EN 16247-4 transport energy audit inclusion |
| Article 11(10) | Penalties for non-compliance | Deadline tracking and warning system |
| Article 12 | Energy management systems | Mandatory EnMS for enterprises >85 TJ/year |
| Article 25 | Energy efficiency in data centres | Data center specific audit requirements (PUE reporting) |
| Article 26 | Energy efficiency obligation schemes | ECM savings quantification for obligation reporting |
| Article 27 | Energy efficiency national fund | Financial mechanism awareness for ECM investment |

---

## 18. Appendix: ISO 50001:2018 Clause Mapping

| ISO 50001 Clause | Requirement | PACK-031 Support |
|-----------------|-------------|-----------------|
| 4. Context of the organization | Interested parties, scope, EnMS boundary | Setup wizard: facility configuration |
| 5. Leadership | Energy policy, roles, responsibilities | Template 10: energy policy template, RBAC roles |
| 6.1 Actions to address risks | Risk-based thinking for EnMS | Compliance engine: risk identification |
| 6.2 Objectives, targets, action plans | Energy objectives with measurable targets | Engine 1: EnPIs and targets, Engine 5: ECM action plans |
| 6.3 Energy review | SEUs, EnPIs, baselines, data collection | Engine 1: baseline, Engine 3: process mapping, Engine 4: equipment |
| 6.4 EnPIs | Energy performance indicators | Engine 1: EnPI definitions per ISO 50006 |
| 6.5 EnBs | Energy baselines | Engine 1: baseline establishment per ISO 50006 |
| 6.6 Planning for collection of energy data | Data collection plan | Setup wizard: metering plan, Integration 7: utility metering |
| 7.1 Resources | Competence, resources | Auditor registry, resource planning |
| 7.2 Competence | Training needs | EN 16247-5 competence tracking |
| 7.5 Documented information | Document control | SHA-256 provenance, version tracking, audit trail |
| 8.1 Operational planning and control | SEU operational criteria | Engine 4: equipment efficiency monitoring |
| 8.2 Design | Energy-efficient design criteria | Benchmark engine: BAT-AEL reference for new designs |
| 8.3 Procurement | Energy-efficient procurement | Equipment registry: IE class tracking for procurement |
| 9.1 Monitoring, measurement, analysis | EnPI tracking, key characteristics | Engine 1: continuous EnPI monitoring, Workflow 2 |
| 9.2 Internal audit | EnMS audit program | Workflow 8: internal audit scheduling |
| 9.3 Management review | Management review inputs/outputs | Template 10: management review package |
| 10.1 Nonconformity | Corrective action | Compliance engine: nonconformity register |
| 10.2 Continual improvement | EnMS improvement | Engine 10: maturity model, Engine 5: continuous ECM identification |

---

## 19. Appendix: IPMVP M&V Options Detail

### Option A: Retrofit Isolation -- Key Parameter Measurement

**Method:** Measure the key performance parameter(s) that define the ECM's energy savings. Stipulate (estimate) the remaining parameters based on historical data or manufacturer specifications.

**Example:** Motor replacement -- measure motor power draw (kW) at typical load; stipulate operating hours from production records.

**Statistical Requirements:**
- Key parameter measurement uncertainty: +/- 5% at 90% confidence
- Stipulated parameters: engineering judgment with documented basis
- Reporting: monthly or quarterly savings

**Best For:** Simple retrofits with one dominant variable (motor replacement, lighting upgrade, insulation addition)

### Option B: Retrofit Isolation -- All Parameter Measurement

**Method:** Measure all energy parameters for the retrofitted system continuously or at high frequency.

**Example:** VSD pump installation -- measure pump power (kW) and flow (m3/hr) continuously; calculate savings from difference between baseline and post-implementation energy at same operating conditions.

**Statistical Requirements per ASHRAE Guideline 14:**
- CV(RMSE) <= 20% for monthly models
- CV(RMSE) <= 30% for hourly models
- NMBE within +/- 5% for monthly models
- NMBE within +/- 10% for hourly models
- R-squared >= 0.75

**Best For:** Variable-load systems where savings depend on operating conditions (VSD, process optimization, compressed air optimization)

### Option C: Whole Facility

**Method:** Analyze whole-facility energy consumption using utility meter data. Apply regression analysis to normalize for weather, production, and other relevant variables. Savings = baseline-predicted energy minus actual post-implementation energy.

**Statistical Requirements per ASHRAE Guideline 14:**
- CV(RMSE) <= 20% for monthly models
- NMBE within +/- 5%
- R-squared >= 0.75
- Minimum 12 months baseline, 12 months post-implementation (3 months for interim reporting)
- Sufficient data points: N >= 3 * number of independent variables + 1

**Best For:** Multiple simultaneous ECMs, whole-building retrofits, energy management programs

### Option D: Calibrated Simulation

**Method:** Energy simulation model calibrated to match actual pre-retrofit energy consumption. Post-retrofit model updated with ECM parameters. Savings = calibrated baseline model minus calibrated post-retrofit model.

**Statistical Requirements per ASHRAE Guideline 14:**
- Monthly CV(RMSE) <= 15%
- Monthly NMBE within +/- 5%
- Model must replicate end-use profiles and load shapes

**Best For:** New construction (no pre-retrofit data), complex interactive effects, facilities with limited metering

---

## 20. Appendix: Energy Unit Conversion Reference

| From | To | Factor |
|------|-----|--------|
| 1 kWh | MJ | 3.6 |
| 1 kWh | BTU | 3,412.14 |
| 1 GJ | kWh | 277.78 |
| 1 therm | kWh | 29.31 |
| 1 therm | MJ | 105.51 |
| 1 tonne of steam (from 100C water, at 10 bar) | GJ | ~2.68 (varies with pressure) |
| 1 m3 natural gas | kWh | ~10.55 (varies with composition, NCV) |
| 1 kg fuel oil | kWh | ~11.86 (NCV, varies with grade) |
| 1 kg LPG | kWh | ~12.78 (NCV) |
| 1 tonne coal | GJ | ~25-30 (varies with grade) |
| 1 kg biomass (wood pellet) | kWh | ~4.8 (NCV, 10% moisture) |
| Primary energy factor (electricity, EU avg) | - | 2.1 (decreasing with renewable share) |
| Primary energy factor (natural gas) | - | 1.1 |
| Primary energy factor (district heat) | - | 0.7-1.3 (varies by source) |

---

## 21. Appendix: Compressed Air System Rules of Thumb

| Rule | Value | Source |
|------|-------|--------|
| Cost of compressed air per m3 | EUR 0.02-0.04/m3 (at 7 bar) | Industry average |
| Electricity cost as % of total compressed air lifecycle cost | 75-80% | US DOE |
| Every 1 bar pressure reduction saves | ~7% of compressor energy | Compressed Air Challenge |
| Every 6C reduction in inlet air temperature saves | ~2% of compressor energy | Thermodynamic calculation |
| Typical leak rate (well-maintained system) | <10% of generated air | ISO 11011 benchmark |
| Typical leak rate (poorly maintained system) | 25-40% of generated air | Industry survey data |
| VSD compressor turndown ratio | 20-100% (typical), 25-100% (conservative) | Manufacturer specifications |
| Specific power target (screw, 7 bar) | <6.5 kW/m3/min | Industry best practice |
| Air receiver sizing rule | 3-5 L per L/s of compressor capacity | Compressed Air Challenge |
| Heat recovery potential from compression | 80-93% of input electrical energy | Thermodynamic first law |
| Pressure drop limit (entire distribution) | <1.0 bar (generation to point of use) | ISO 11011 recommendation |
| Artificial demand per 1 bar excess pressure | ~1% of system volume per second | Ideal gas law |

---

## 22. Appendix: Steam System Rules of Thumb

| Rule | Value | Source |
|------|-------|--------|
| Cost of steam per tonne (natural gas, 10 bar) | EUR 25-40/tonne | Fuel cost dependent |
| Failed steam trap (stuck open, 15mm, 7 bar) | ~25-50 kg/hr steam loss | Spirax Sarco |
| Annual cost per failed trap | EUR 5,000-10,000 | At EUR 0.03-0.04/kg steam |
| Typical trap failure rate (4-year survey cycle) | 15-30% of installed traps | Armstrong, Spirax Sarco surveys |
| 1% reduction in boiler O2 saves | ~0.5% fuel | Combustion stoichiometry |
| Bare pipe heat loss (100mm, 10 bar, uninsulated) | ~500 W/m | Heat transfer calculation |
| Insulated pipe heat loss (100mm, 10 bar, 50mm insulation) | ~50 W/m (90% reduction) | Heat transfer calculation |
| Flash steam percentage at pressure reduction | ~10-15% when reducing from 10 bar to 1 bar | Steam tables |
| Condensate return value (thermal + water + chemical) | ~20-25% of steam cost | Industry analysis |
| Boiler efficiency improvement from economizer | 3-5% (condensing), 1-2% (non-condensing) | Heat recovery calculation |
| Blowdown heat recovery savings | 50-80% of blowdown energy loss | Heat exchanger efficiency |
| Optimal condensate return rate | >85% | Industry best practice |

---

## 23. Appendix: Data Center Specific Metrics

| Metric | Formula | Target | Source |
|--------|---------|--------|--------|
| PUE (Power Usage Effectiveness) | Total_facility_power / IT_equipment_power | <1.3 (good), <1.2 (excellent) | The Green Grid |
| DCiE (Data Center infrastructure Efficiency) | 1/PUE * 100% | >77% (good), >83% (excellent) | The Green Grid |
| CUE (Carbon Usage Effectiveness) | Total_CO2 / IT_equipment_energy | Minimize (depends on grid factor) | The Green Grid |
| WUE (Water Usage Effectiveness) | Annual_water_use_L / IT_equipment_energy_kWh | <1.0 L/kWh (efficient) | The Green Grid |
| ERE (Energy Reuse Effectiveness) | (Total_energy - Reused_energy) / IT_energy | <1.0 (with heat reuse) | The Green Grid |
| CADE (Corporate Average Data center Efficiency) | Composite metric | Maximize | Uptime Institute |
| Temperature range (ASHRAE A1) | 18-27C inlet | Widen to maximize free cooling | ASHRAE TC 9.9 |
| Free cooling hours (northern Europe) | Hours where T_outdoor < 18C | 5,000-7,000 hours/year | Climate data |
| UPS efficiency at partial load | % at 25%, 50%, 75%, 100% load | >96% at 50% load (online double-conversion) | Manufacturer data |

---

## 24. Appendix: Glossary

| Term | Definition |
|------|-----------|
| **BAT** | Best Available Techniques -- techniques developed at a scale that allows implementation under economically and technically viable conditions |
| **BAT-AEL** | BAT-Associated Energy Level -- energy performance range achievable using BAT |
| **BEP** | Best Efficiency Point -- the pump operating point with highest efficiency |
| **BMS** | Building Management System -- centralized system for monitoring and controlling building services |
| **BREF** | BAT Reference Document -- published by the EU JRC for each industrial sector |
| **CDD** | Cooling Degree Days -- measure of cooling demand based on outdoor temperature |
| **CMMS** | Computerized Maintenance Management System -- asset management software |
| **COP** | Coefficient of Performance -- ratio of useful heating/cooling output to energy input |
| **CV(RMSE)** | Coefficient of Variation of the Root Mean Square Error -- statistical fit metric for M&V |
| **ECM** | Energy Conservation Measure -- specific action to reduce energy consumption |
| **ECO** | Energy Conservation Opportunity -- identified potential for energy saving |
| **EED** | EU Energy Efficiency Directive (2023/1791) |
| **EER** | Energy Efficiency Ratio -- cooling output (BTU/hr) per energy input (W) |
| **EnMS** | Energy Management System -- per ISO 50001 |
| **EnPI** | Energy Performance Indicator -- quantitative measure of energy performance per ISO 50006 |
| **ESCO** | Energy Service Company -- company providing energy efficiency services, often via performance contracts |
| **EU ETS** | EU Emissions Trading System -- cap-and-trade system for GHG emissions |
| **FAD** | Free Air Delivery -- compressed air volume at standard conditions (1 bar, 20C) |
| **HDD** | Heating Degree Days -- measure of heating demand based on outdoor temperature |
| **HRSG** | Heat Recovery Steam Generator -- boiler using exhaust gas heat |
| **HHV** | Higher Heating Value -- calorific value including latent heat of water vapor |
| **IED** | Industrial Emissions Directive (2010/75/EU) |
| **IE1-IE5** | International Efficiency classes for electric motors per IEC 60034-30-1 |
| **IPMVP** | International Performance Measurement and Verification Protocol |
| **IRR** | Internal Rate of Return -- discount rate at which NPV equals zero |
| **LCCA** | Life-Cycle Cost Analysis -- total cost of ownership over equipment lifetime |
| **LMTD** | Log Mean Temperature Difference -- driving force for heat exchange |
| **M&V** | Measurement and Verification -- process to verify energy savings |
| **NMBE** | Normalized Mean Bias Error -- statistical metric for M&V model bias |
| **NPV** | Net Present Value -- discounted future cash flows minus initial investment |
| **NTU** | Number of Transfer Units -- dimensionless HX performance parameter |
| **ORC** | Organic Rankine Cycle -- thermodynamic cycle for low-grade heat-to-power conversion |
| **PUE** | Power Usage Effectiveness -- data center energy efficiency metric |
| **RLS** | Row-Level Security -- database access control at the row level |
| **SCADA** | Supervisory Control and Data Acquisition -- industrial process monitoring system |
| **SEC** | Specific Energy Consumption -- energy per unit of production or service |
| **SEU** | Significant Energy Use -- per ISO 50001, energy uses that account for substantial consumption or offer substantial improvement potential |
| **VSD** | Variable Speed Drive -- electronic device controlling motor speed (also known as VFD) |

---

## 25. User Stories & Acceptance Criteria

### US-001: Energy Baseline Establishment

```
As an energy manager,
I want to establish a statistically robust energy baseline for my facility,
So that I can measure energy performance improvement accurately over time.
```

**Acceptance Criteria:**

```
GIVEN a facility with 24+ months of monthly energy consumption data,
  AND corresponding production volume and weather data,
WHEN I run the energy baseline engine,
THEN it SHALL produce a regression model with R-squared >= 0.75,
  AND CV(RMSE) <= 25%,
  AND NMBE within +/- 10%,
  AND define at least one EnPI per ISO 50006,
  AND complete baseline calculation in < 30 seconds,
  AND produce a SHA-256 provenance hash of all inputs and outputs.

GIVEN a facility with less than 12 months of data,
WHEN I run the energy baseline engine,
THEN it SHALL warn that data is insufficient for a robust baseline,
  AND offer to proceed with a preliminary baseline flagged as "provisional",
  AND require manual confirmation before proceeding.

GIVEN a facility where no regression model achieves R-squared >= 0.75,
WHEN the engine completes variable selection and model fitting,
THEN it SHALL flag the model for manual review,
  AND suggest additional variables that may improve the fit,
  AND document the best-available model with caveats.
```

### US-002: EN 16247 Energy Audit

```
As a facility engineer,
I want to conduct an EN 16247-compliant energy audit of my facility,
So that I can identify energy saving opportunities and comply with EED Article 11.
```

**Acceptance Criteria:**

```
GIVEN a facility with complete energy data and equipment inventory,
WHEN I run a Type 2 (detailed) energy audit,
THEN it SHALL produce an energy balance covering >= 85% of total energy consumption,
  AND identify at least 5 energy conservation measures (ECMs),
  AND calculate energy savings (kWh/yr) and cost savings (EUR/yr) per ECM,
  AND calculate simple payback, NPV, and IRR per ECM,
  AND generate an EN 16247-1 Annex A compliant report,
  AND complete the audit in < 60 minutes (data pre-loaded).

GIVEN an ECM that interacts with another ECM (e.g., VSD on pump reduces waste heat),
WHEN savings are calculated for both ECMs,
THEN the interaction effect SHALL be quantified and documented,
  AND the combined savings SHALL be less than or equal to the sum of individual savings,
  AND the interaction matrix SHALL be included in the audit report.
```

### US-003: Equipment Efficiency Analysis

```
As a facility engineer,
I want to assess the efficiency of all major energy-consuming equipment,
So that I can identify oversized, underperforming, and upgrade-eligible equipment.
```

**Acceptance Criteria:**

```
GIVEN an equipment inventory with motors, pumps, compressors, and boilers,
  AND operating measurements (voltage, current, power factor, pressure, flow, temperature),
WHEN I run the equipment efficiency engine,
THEN it SHALL classify each motor by IE class and load factor,
  AND flag motors with load factor < 40% as "severely oversized",
  AND calculate wire-to-water efficiency for each pump,
  AND flag pumps operating > 20% from BEP,
  AND calculate specific power (kW/m3/min) for each compressor,
  AND calculate boiler efficiency by indirect method with loss breakdown,
  AND produce upgrade recommendations with savings per equipment item,
  AND complete analysis of 1,000 equipment items in < 3 minutes.
```

### US-004: Compressed Air System Audit

```
As an energy manager,
I want to perform a comprehensive compressed air system audit,
So that I can reduce air leaks, optimize pressure, and improve specific power.
```

**Acceptance Criteria:**

```
GIVEN a compressed air system with compressors, dryers, receivers, and distribution,
  AND ultrasonic leak survey data with leak locations and sizes,
WHEN I run the compressed air audit workflow,
THEN it SHALL map the complete compressed air system (compressors to end uses),
  AND quantify total leak rate to within +/- 20% of actual,
  AND calculate the annual cost of leaks,
  AND assess system specific power against target of < 6.5 kW/m3/min (7 bar),
  AND calculate pressure optimization savings (7% per bar reduction),
  AND evaluate VSD retrofit potential with energy model for full operating profile,
  AND generate a leak register sorted by annual cost impact (largest first),
  AND produce compressed air audit report with all findings and recommendations.

GIVEN a system where leak rate exceeds 25% of generated air,
WHEN the results are reported,
THEN it SHALL flag the system as "critical" condition,
  AND prioritize leak repair as first action,
  AND estimate payback for a systematic leak repair program.
```

### US-005: Steam System Optimization

```
As a facility engineer at a process industry plant,
I want to audit my steam system from boiler to condensate return,
So that I can improve boiler efficiency, fix failed traps, and optimize condensate recovery.
```

**Acceptance Criteria:**

```
GIVEN a steam system with boilers, distribution headers, steam traps, and condensate return,
  AND stack gas analysis data (O2, CO, temperature) for each boiler,
  AND steam trap survey data (condition for each trap),
WHEN I run the steam system audit workflow,
THEN it SHALL calculate boiler efficiency by both direct and indirect methods,
  AND the two methods SHALL agree within 3% (or flag for investigation),
  AND classify each steam trap as live/failed_open/failed_closed/leaking,
  AND calculate steam loss (kg/hr) and annual cost for each failed trap,
  AND assess condensate return rate and flash steam recovery potential,
  AND identify insulation deficiencies with economic insulation thickness calculation,
  AND produce ranked optimization recommendations with savings and payback.

GIVEN a boiler with stack O2 > 5% (natural gas),
WHEN efficiency is calculated,
THEN it SHALL recommend O2 trim control,
  AND calculate the savings from reducing excess air to optimal range (2-3% O2).
```

### US-006: Waste Heat Recovery Assessment

```
As an energy manager,
I want to identify and quantify waste heat recovery opportunities,
So that I can reduce energy costs by recovering heat currently rejected to ambient.
```

**Acceptance Criteria:**

```
GIVEN a facility with identifiable hot streams (flue gas, cooling water, exhaust) and cold streams (feedwater, process preheat, space heating),
WHEN I run the waste heat recovery workflow with pinch analysis,
THEN it SHALL construct hot and cold composite curves,
  AND identify the pinch point at the specified delta_T_min,
  AND calculate minimum hot utility (QH_min) and cold utility (QC_min),
  AND the calculated minimums SHALL be within 5% of theoretical minimum,
  AND recommend heat exchanger types and sizes for each recovery opportunity,
  AND calculate CapEx, annual savings, payback, NPV, and IRR per opportunity,
  AND rank opportunities by NPV (highest first),
  AND include sensitivity analysis on energy price (+/- 20%).
```

### US-007: Regulatory Compliance Management

```
As a compliance officer,
I want to track all energy-related regulatory obligations (EED, EU ETS, IED),
So that I can ensure timely compliance and avoid penalties.
```

**Acceptance Criteria:**

```
GIVEN a non-SME enterprise within the EU,
WHEN I configure the regulatory compliance workflow,
THEN it SHALL correctly identify EED Article 11 audit obligation,
  AND calculate the next audit due date based on 4-year cycle,
  AND generate advance warnings at 6, 3, and 1 months before due date,
  AND check ISO 50001 certification for audit exemption,
  AND track the 85 TJ threshold for mandatory EnMS (Article 12),
  AND generate a unified compliance dashboard with RAG status for all obligations.

GIVEN an ISO 50001 certified facility,
WHEN checking EED Article 11 compliance,
THEN it SHALL mark the facility as exempt from mandatory energy audit,
  AND track certificate expiry date,
  AND generate a warning 6 months before certificate expires.
```

### US-008: ISO 50001 Certification Support

```
As an energy manager pursuing ISO 50001 certification,
I want gap analysis and EnMS documentation support,
So that I can prepare for certification efficiently.
```

**Acceptance Criteria:**

```
GIVEN a facility that is not yet ISO 50001 certified,
WHEN I run the ISO 50001 certification workflow,
THEN it SHALL assess current state against all ISO 50001:2018 clauses,
  AND produce a gap analysis with per-clause scores (compliant/partial/non-compliant),
  AND estimate effort to close each gap (person-hours),
  AND estimate timeline to certification readiness,
  AND provide energy policy template per Clause 5.2,
  AND configure automated EnPI tracking per Clause 6.4,
  AND generate management review package per Clause 9.3.
```

### US-009: Energy Benchmarking

```
As a corporate sustainability manager,
I want to benchmark my facilities against industry peers and BAT-AEL values,
So that I can identify underperforming sites and prioritize improvement investments.
```

**Acceptance Criteria:**

```
GIVEN a facility with energy and production data and a NACE code classification,
WHEN I run the energy benchmark engine,
THEN it SHALL calculate Specific Energy Consumption (SEC) with appropriate units,
  AND compare SEC against BAT-AEL values from the relevant BREF document,
  AND calculate gap-to-best-practice in kWh/yr and EUR/yr,
  AND provide percentile ranking within the subsector,
  AND assess energy efficiency maturity on a 1-5 scale across 5 dimensions,
  AND estimate improvement potential if facility achieves BAT-AEL performance.

GIVEN multiple facilities in the same subsector,
WHEN benchmarking results are aggregated,
THEN it SHALL rank facilities by SEC (worst to best),
  AND identify the top 3 facilities with highest improvement potential,
  AND produce a portfolio-level energy efficiency improvement roadmap.
```

### US-010: Energy Savings Financial Analysis

```
As a CFO,
I want financial analysis of energy saving investments,
So that I can allocate capital to the highest-return energy efficiency projects.
```

**Acceptance Criteria:**

```
GIVEN a portfolio of 50+ identified ECMs with energy savings estimates,
  AND financial parameters (discount rate, energy price escalation, equipment life),
WHEN I run the financial analysis,
THEN it SHALL calculate simple payback, NPV, IRR, and LCCA for each ECM,
  AND rank ECMs by NPV (configurable: payback, IRR, or NPV ranking),
  AND produce a capital planning summary grouped by payback period (<1yr, 1-3yr, 3-7yr, >7yr),
  AND include sensitivity analysis on discount rate (+/- 2%) and energy price (+/- 20%),
  AND flag ECMs with IRR < WACC as "below hurdle rate",
  AND calculate portfolio-level total NPV, weighted average payback, and total CapEx.
```

---

## 26. Appendix: Sankey Diagram Energy Flow Categories

### Typical Manufacturing Facility Energy Flow

```
ENERGY INPUTS                    END USES                        LOSSES
=============                    =========                       ======

Electricity ----+----> Motors/Drives (35%) --------> Mechanical work (25%) + Motor losses (10%)
(60% of total)  |
                +----> Compressed Air (15%) -------> Pneumatic work (2%) + Compression heat (13%)
                |
                +----> Lighting (5%) ----------------> Illumination (3%) + Heat (2%)
                |
                +----> HVAC (10%) -----------------> Conditioned space (6%) + Losses (4%)
                |
                +----> Process (30%) ----------------> Useful process (20%) + Losses (10%)
                |
                +----> Auxiliaries (5%) -------------> Building services (3%) + Losses (2%)

Natural Gas ----+----> Boilers (60%) ----------------> Steam/hot water (48%) + Stack/radiation (12%)
(35% of total)  |
                +----> Furnaces/Ovens (30%) ---------> Process heat (18%) + Exhaust/wall (12%)
                |
                +----> Space Heating (10%) ----------> Conditioned space (7%) + Losses (3%)

Other Fuels ----+----> Process (70%) ----------------> Various (50%) + Various (20%)
(5% of total)   |
                +----> Transport (30%) --------------> Motive power (9%) + Exhaust/friction (21%)
```

### Energy Loss Distribution (Typical Manufacturing)

| Loss Category | % of Total Input | Recovery Potential |
|---------------|-----------------|-------------------|
| Motor system losses | 8-12% | 30-50% via IE upgrade, VSD, right-sizing |
| Compressed air losses | 10-15% | 40-60% via leak repair, pressure optimization, VSD |
| Boiler stack losses | 5-10% | 50-80% via economizer, air preheater, condensing |
| Furnace exhaust losses | 5-12% | 30-60% via recuperator, regenerator |
| Steam distribution losses | 3-8% | 50-80% via trap repair, insulation, condensate return |
| HVAC losses | 3-6% | 30-50% via VSD, heat recovery, economizer |
| Lighting heat losses | 1-3% | 50-70% via LED, controls |
| Building envelope losses | 2-5% | 20-40% via insulation, air sealing |
| Electrical distribution losses | 1-3% | 20-50% via PF correction, transformer upgrade |

---

## 27. Appendix: Motor Efficiency Lookup Tables (IEC 60034-30-1:2014)

### 4-Pole Motors, 50 Hz (Nominal Efficiency %)

| Rating (kW) | IE1 | IE2 | IE3 | IE4 | IE5 |
|-------------|-----|-----|-----|-----|-----|
| 0.75 | 72.1 | 77.4 | 80.7 | 82.5 | 85.0 |
| 1.1 | 75.0 | 79.6 | 82.7 | 84.1 | 86.5 |
| 1.5 | 77.2 | 81.3 | 84.2 | 85.3 | 87.5 |
| 2.2 | 79.7 | 83.2 | 85.9 | 86.7 | 88.5 |
| 3.0 | 81.5 | 84.6 | 87.1 | 87.7 | 89.5 |
| 4.0 | 83.1 | 85.8 | 88.1 | 88.6 | 90.2 |
| 5.5 | 84.7 | 87.0 | 89.2 | 89.6 | 91.0 |
| 7.5 | 86.0 | 88.1 | 90.1 | 90.4 | 91.7 |
| 11 | 87.6 | 89.4 | 91.2 | 91.4 | 92.6 |
| 15 | 88.7 | 90.3 | 91.9 | 92.1 | 93.3 |
| 18.5 | 89.3 | 90.9 | 92.4 | 92.6 | 93.7 |
| 22 | 89.9 | 91.3 | 92.7 | 93.0 | 94.0 |
| 30 | 90.7 | 92.0 | 93.3 | 93.6 | 94.5 |
| 37 | 91.2 | 92.5 | 93.7 | 93.9 | 94.8 |
| 45 | 91.7 | 92.9 | 94.0 | 94.2 | 95.0 |
| 55 | 92.1 | 93.2 | 94.3 | 94.6 | 95.3 |
| 75 | 92.7 | 93.8 | 94.7 | 95.0 | 95.6 |
| 90 | 93.0 | 94.1 | 95.0 | 95.2 | 95.8 |
| 110 | 93.3 | 94.3 | 95.2 | 95.4 | 96.0 |
| 132 | 93.5 | 94.6 | 95.4 | 95.6 | 96.2 |
| 160 | 93.8 | 94.8 | 95.6 | 95.8 | 96.3 |
| 200 | 94.0 | 95.0 | 95.8 | 96.0 | 96.5 |
| 250 | 94.0 | 95.0 | 95.8 | 96.0 | 96.5 |
| 315 | 94.0 | 95.0 | 95.8 | 96.0 | 96.5 |
| 355 | 94.0 | 95.0 | 95.8 | 96.0 | 96.5 |

**Motor Upgrade Savings Calculation:**

```
Annual_savings_kWh = P_rated_kW * (1/eta_old - 1/eta_new) * annual_hours * load_factor
Annual_savings_EUR = Annual_savings_kWh * electricity_price_EUR_per_kWh
Payback_years = motor_cost_EUR / Annual_savings_EUR

Example:
  11 kW motor, IE1 (87.6%) -> IE4 (91.4%), 6000 hrs/yr, 75% load, EUR 0.15/kWh
  Savings = 11 * (1/0.876 - 1/0.914) * 6000 * 0.75 = 2,238 kWh/yr = EUR 336/yr
  Motor cost ~EUR 800 -> Payback = 2.4 years
```

---

## 28. Appendix: Steam Tables Reference (Saturated Steam)

| Pressure (bar_g) | Temperature (C) | Specific Enthalpy h_f (kJ/kg) | Specific Enthalpy h_g (kJ/kg) | Latent Heat h_fg (kJ/kg) |
|-------------------|-----------------|-------------------------------|-------------------------------|--------------------------|
| 0.0 | 100.0 | 419 | 2,676 | 2,257 |
| 0.5 | 111.4 | 468 | 2,693 | 2,226 |
| 1.0 | 120.2 | 505 | 2,707 | 2,201 |
| 2.0 | 133.5 | 562 | 2,725 | 2,163 |
| 3.0 | 143.6 | 605 | 2,738 | 2,133 |
| 4.0 | 151.8 | 641 | 2,749 | 2,108 |
| 5.0 | 158.8 | 671 | 2,757 | 2,086 |
| 6.0 | 164.9 | 698 | 2,764 | 2,066 |
| 7.0 | 170.4 | 721 | 2,769 | 2,048 |
| 8.0 | 175.4 | 743 | 2,774 | 2,031 |
| 9.0 | 179.9 | 763 | 2,778 | 2,015 |
| 10.0 | 184.1 | 782 | 2,781 | 1,999 |
| 12.0 | 191.6 | 815 | 2,787 | 1,972 |
| 14.0 | 198.3 | 845 | 2,790 | 1,946 |
| 16.0 | 204.3 | 872 | 2,793 | 1,921 |
| 18.0 | 209.8 | 897 | 2,795 | 1,898 |
| 20.0 | 214.9 | 920 | 2,797 | 1,877 |
| 25.0 | 226.0 | 972 | 2,800 | 1,828 |
| 30.0 | 235.8 | 1,017 | 2,801 | 1,784 |
| 40.0 | 252.0 | 1,094 | 2,798 | 1,704 |

**Flash Steam Percentage:**

```
Flash_steam_% = (h_f_high - h_f_low) / h_fg_low * 100%

Example: Flash from 10 bar_g to 0 bar_g (atmospheric)
Flash_% = (782 - 419) / 2,257 * 100% = 16.1%

This means 16.1% of the condensate will flash to steam when pressure drops from 10 bar to atmospheric.
At a condensate flow of 1,000 kg/hr, this represents 161 kg/hr of flash steam -- potentially
recoverable for LP heating applications.
```

---

## 29. Appendix: Boiler Efficiency Worked Example

### Indirect Method Loss Calculation (Natural Gas Boiler, 10 bar)

**Given Data:**
- Fuel: Natural gas (HHV = 39.8 MJ/m3)
- Stack temperature: 200C
- Ambient temperature: 20C
- Stack O2: 4.5% (excess air = 25%)
- Steam output: 5,000 kg/hr at 10 bar_g (184C)
- Feedwater temperature: 80C
- Blowdown rate: 3%
- Boiler rating: 5 MW (thermal)

**Loss 1: Dry Flue Gas Loss**
```
Excess_air = O2 / (21 - O2) * 100 = 4.5 / (21 - 4.5) * 100 = 27.3%
m_gas = (1 + excess_air/100) * stoichiometric_air * fuel_flow
Cp_gas = 1.05 kJ/kg.K (average for flue gas)
L1 = m_gas * Cp_gas * (200 - 20) / HHV_per_kg
L1 = approximately 6.8% (typical for 200C stack with 25% excess air)
```

**Loss 2: Moisture from Hydrogen Combustion**
```
Natural gas (CH4): H2 mass fraction ~25%
L2 = 9 * 0.25 * (2,676 + 1.88*(200-100) - 4.18*20) / 39,800 * 100
L2 = approximately 10.5%
```

**Loss 3: Radiation and Convection Loss**
```
For a 5 MW boiler (typical insulated shell boiler):
L3 = approximately 1.0% (from ASME PTC 4 radiation loss chart)
```

**Loss 4: Blowdown Loss**
```
L4 = blowdown_rate / (1 - blowdown_rate) * (h_blowdown - h_makeup) / (fuel_flow * HHV)
h_blowdown = 782 kJ/kg (saturated water at 10 bar_g)
h_makeup = 84 kJ/kg (water at 20C)
L4 = 0.03 / 0.97 * (782 - 84) / ...
L4 = approximately 1.5%
```

**Loss 5: Unburned Combustibles**
```
CO < 50 ppm for well-tuned gas burner
L5 = approximately 0.1%
```

**Total Losses and Efficiency:**
```
Total_losses = 6.8 + 10.5 + 1.0 + 1.5 + 0.1 = 19.9%
Boiler_efficiency = 100 - 19.9 = 80.1% (on HHV basis)

Note: On NCV (Net Calorific Value) basis, efficiency would be approximately 88-89%
as NCV excludes latent heat of water vapor.
```

**Improvement Opportunities from this Example:**
1. Reduce stack temperature from 200C to 150C: saves ~1.5% efficiency -> EUR 7,500/yr at EUR 500K fuel
2. Reduce O2 from 4.5% to 2.5%: saves ~1.0% -> EUR 5,000/yr
3. Install blowdown heat recovery: recover 50% of blowdown loss -> EUR 3,750/yr
4. Install condensing economizer: recover latent heat -> additional 5-8% efficiency for DHWS preheat

---

## 30. Appendix: Pinch Analysis Worked Example

### Simple 4-Stream Problem

**Stream Data:**

| Stream | Type | T_supply (C) | T_target (C) | Heat Capacity Rate CP (kW/K) | Duty (kW) |
|--------|------|-------------|-------------|------------------------------|-----------|
| H1 | Hot | 200 | 80 | 3.0 | 360 (to be cooled) |
| H2 | Hot | 150 | 50 | 1.5 | 150 (to be cooled) |
| C1 | Cold | 30 | 180 | 2.0 | 300 (to be heated) |
| C2 | Cold | 60 | 120 | 4.0 | 240 (to be heated) |

**Minimum Approach Temperature:** delta_T_min = 10C

**Step 1: Shifted Temperatures**
- Hot streams shifted down by delta_T_min/2 = 5C
- Cold streams shifted up by delta_T_min/2 = 5C
- H1: 195-75C, H2: 145-45C
- C1: 35-185C, C2: 65-125C

**Step 2: Temperature Intervals and Heat Cascade**

| Interval | T_shifted (C) | CP_hot | CP_cold | Net (kW) | Cascade (kW) |
|----------|--------------|--------|---------|----------|--------------|
| 195-185 | 195-185 | 3.0 | 0 | +30 | +30 |
| 185-145 | 185-145 | 3.0 | 2.0 | +40 | +70 |
| 145-125 | 145-125 | 4.5 | 2.0 | +50 | +120 |
| 125-75 | 125-75 | 4.5 | 6.0 | -75 | +45 |
| 75-65 | 75-65 | 1.5 | 4.0 | -25 | +20 |
| 65-45 | 65-45 | 1.5 | 0 | +30 | +50 |
| 45-35 | 45-35 | 0 | 2.0 | -20 | +30 |

**Step 3: Results**
- Pinch temperature: 125C (shifted) = 130C hot / 120C cold
- Minimum hot utility (QH_min): 30 kW (from cascade: minimum deficit is 0 at some interval)
- Minimum cold utility (QC_min): 30 kW
- Maximum heat recovery: 510 - 30 = 480 kW

**Without heat recovery:** QH = 540 kW, QC = 510 kW (total utility = 1,050 kW)
**With pinch-optimal recovery:** QH = 30 kW, QC = 30 kW (total utility = 60 kW)
**Utility reduction: 94.3%**

This example demonstrates the power of pinch analysis: systematic heat integration reduces external utility requirements by over 90% in this case.

---

## 31. Appendix: Financial Analysis Reference Formulas

### Net Present Value (NPV)

```
NPV = -C_0 + Sum(CF_t / (1 + r)^t, t=1..N)

Where:
  C_0 = initial investment (CapEx)
  CF_t = net cash flow in year t (annual savings - annual costs, with escalation)
  r = discount rate (WACC or hurdle rate)
  N = equipment lifetime (years)

With energy price escalation:
  CF_t = S_0 * (1 + e)^t - O&M_t
  S_0 = annual energy cost savings at current prices
  e = annual energy price escalation rate (typically 2-5%)
```

### Internal Rate of Return (IRR)

```
IRR = rate r where NPV = 0
Solve: -C_0 + Sum(CF_t / (1 + r)^t, t=1..N) = 0

Decision rule: Accept if IRR > WACC + risk_premium
Typical hurdle rates:
  - Quick wins: IRR > 15%
  - Standard projects: IRR > 12%
  - Strategic investments: IRR > 8%
```

### Life-Cycle Cost Analysis (LCCA)

```
LCCA = C_initial + Sum(C_energy_t / (1+r)^t) + Sum(C_maintenance_t / (1+r)^t) + Sum(C_replacement_t / (1+r)^t) - C_residual / (1+r)^N

Compare LCCA of existing equipment vs. efficient replacement:
  LCCA_savings = LCCA_existing - LCCA_replacement
  If LCCA_savings > 0, replacement is economically justified
```

### Cost of Conserved Energy (CCE)

```
CCE = (C_0 * CRF) / E_saved

Where:
  CRF = Capital Recovery Factor = r * (1+r)^N / ((1+r)^N - 1)
  E_saved = annual energy savings (kWh/yr)

Decision rule: Accept if CCE < current_energy_price
If CCE = EUR 0.08/kWh and electricity costs EUR 0.15/kWh, the ECM is cost-effective.
```

### Sensitivity Analysis Parameters

| Parameter | Base Case | Low Case | High Case | Impact |
|-----------|-----------|----------|-----------|--------|
| Electricity price (EUR/kWh) | 0.15 | 0.10 (-33%) | 0.22 (+47%) | Direct on annual savings |
| Gas price (EUR/MWh) | 35 | 25 (-29%) | 55 (+57%) | Direct on thermal savings |
| Discount rate | 8% | 5% | 12% | Affects NPV, less effect on payback |
| Equipment life (years) | 15 | 10 | 20 | Affects NPV and LCCA |
| Operating hours (hrs/yr) | 6,000 | 4,000 | 8,000 | Direct on annual savings |
| Energy price escalation (%/yr) | 3% | 1% | 5% | Cumulative effect over equipment life |
| Load factor | 75% | 50% | 90% | Direct on motor/pump/compressor savings |

---

## 32. Future Roadmap

- **PACK-032: Building Energy Optimization Pack** -- Commercial and institutional building energy audits with EPBD compliance, DEC/EPC generation, HVAC-dominant analysis, occupant comfort optimization, smart building integration, and WELL/LEED energy credit alignment
- **PACK-033: Renewable Energy Integration Pack** -- On-site renewable energy assessment (solar PV, wind, biomass CHP, geothermal), grid interaction optimization, PPA evaluation, storage sizing, self-consumption maximization, and RE certificate management
- **PACK-034: Energy Performance Contracting Pack** -- ESCO support with guaranteed savings contracts, EPC financial modeling, risk allocation, M&V-based payment schedules, baseline dispute resolution, and contract lifecycle management
- **PACK-035: Industrial Decarbonization Pack** -- Deep industrial decarbonization covering fuel switching, electrification, hydrogen readiness, CCS readiness, process innovation, and technology roadmapping for hard-to-abate sectors

---

*Document Version: 1.0.0 | Last Updated: 2026-03-20 | Status: Draft*
